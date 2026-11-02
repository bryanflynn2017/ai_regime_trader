"""Research runs never use a caller's persistent state or process-global randomness."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.backtester import Backtester
from backtest.research_manifest import revision
from backtest.stress_tests import inject_missing_data, run_standard_stress_tests
from config.settings import Settings
from data_providers.synthetic_provider import SyntheticDataProvider
from risk.circuit_breakers import CircuitBreakerManager
from strategy.signals import RegimeLabel
from strategy.strategy_orchestrator import StrategyOrchestrator, _check_buy_conditions


def frames():
    return SyntheticDataProvider(rows=258).get_multiple_bars(["AAA"])


def test_repeated_runs_order_lock_isolation_and_seeds(tmp_path):
    marker = tmp_path / "private-marker"
    marker.write_text("unchanged")
    settings = Settings(RISK_LOCK_FILE=str(marker), DEFAULT_STOP_LOSS_PCT=.04,
                        MAX_TRADES_PER_DAY=1, LIVE_TRADING_ALLOWED=False)
    paths = []

    def factory(cfg, seed):
        path = cfg.risk_lock_path()
        paths.append(path)
        assert path != marker and not path.exists()
        rng = np.random.default_rng(seed)
        daily = StrategyOrchestrator(settings_override=cfg)
        regime = SimpleNamespace(label=RegimeLabel.BULL, confidence=.8,
                                 is_stable=True, is_uncertain=False)

        def generate(ticker, bars):
            if ticker == "SHOCK":
                CircuitBreakerManager(settings_override=cfg).evaluate(
                    current_equity=50, starting_day_equity=100,
                    starting_week_equity=100, peak_equity=100)
                assert path.exists()
            sig = daily._make_buy(ticker, bars, bars.iloc[-1], regime, str(rng.integers(100000)))
            assert sig.stop_loss == pytest.approx(sig.entry_price * .96)
            return sig

        return SimpleNamespace(generate_signal=generate)

    bt = Backtester(orchestrator_factory=factory, settings_override=settings, seed=7)
    settings.DEFAULT_STOP_LOSS_PCT = .2  # Runner owns the original scalar snapshot.
    a = frames()
    a["BBB"] = a["AAA"].copy()
    b = {"SHOCK": a["AAA"]}
    a1, b1, b2, a2 = (bt.run(x) for x in (a, b, b, a))
    for first, repeat in [(a1, a2), (b1, b2)]:
        pd.testing.assert_series_equal(first.equity_curve, repeat.equity_curve)
        assert first.risk_decisions == repeat.risk_decisions
        assert first.signals == repeat.signals and first.trades == repeat.trades
        assert first.config_snapshot == repeat.config_snapshot
    assert any(d["approved"] for d in a1.risk_decisions)
    # Both candidates can pass at close; only one may execute the next day.
    assert [o["status"] for o in a1.metadata["orders"][:2]] == ["filled", "cancelled_risk"]
    assert b1.risk_decisions and not any(d["approved"] for d in b1.risk_decisions)
    assert len(set(paths)) == 4 and not any(p.parent.exists() for p in paths)
    assert marker.read_text() == "unchanged"
    assert a1.config_snapshot["accounting_version"] == 2
    assert a1.config_snapshot["research_manifest"]["strategy_seed"] == 7


def test_strategy_and_risk_use_declared_settings():
    cfg = Settings(MIN_PRICE=900, REGIME_STABILITY_BARS=5, MAX_OPEN_POSITIONS=1)
    orch = StrategyOrchestrator(settings_override=cfg)
    regime = SimpleNamespace(label=RegimeLabel.BULL, confidence=.8)
    _, _, reasons = _check_buy_conditions(pd.Series({"close": 100}), regime, orch._settings)
    assert any("below MIN_PRICE 900" in r for r in reasons)
    assert not orch._regime_engine._check_stability(RegimeLabel.BULL, [RegimeLabel.BULL.value] * 3)[0]
    assert orch._regime_engine._check_stability(RegimeLabel.BULL, [RegimeLabel.BULL.value] * 5)[0]


def test_manifest_is_allowlisted_and_sensitive_inputs_only_appear_as_digests(tmp_path):
    marker = "private-fixture-marker"
    cfg = Settings(ALPACA_SECRET_KEY=marker, RISK_LOCK_FILE=str(tmp_path / marker))
    bt = Backtester(settings_override=cfg)
    df = frames()["AAA"]
    df.attrs["private"] = marker
    m = bt.run({marker: df}).config_snapshot["research_manifest"]
    assert marker not in json.dumps(m) and str(tmp_path) not in json.dumps(m)
    assert m["inputs"][0]["synthetic_seed"] == df.attrs["synthetic_seed"] == 495955936
    assert "code_revision" in m and "pandas" in m["dependencies"]
    changed = df.copy()
    changed.iloc[0, 0] += .01
    other = bt.run({marker: changed}).config_snapshot["research_manifest"]
    assert m["inputs"] != other["inputs"]


def test_stress_injection_is_order_independent_and_records_actual_seed():
    data = SyntheticDataProvider(rows=260).get_multiple_bars(["BBB", "AAA"])
    one = inject_missing_data(data, seed=11)
    two = inject_missing_data(dict(reversed(data.items())), seed=11)
    for symbol in data:
        pd.testing.assert_frame_equal(one[symbol], two[symbol])
    result = run_standard_stress_tests(Backtester(), data, seed=11)
    for scenario, run in result.items():
        assert run.config_snapshot["research_manifest"]["stress_scenario"] == scenario
    assert result["missing_data"].config_snapshot["research_manifest"]["missing_data_seed"] == 11


def test_factory_exception_cleans_up_run_directory():
    paths = []

    def fail(cfg, seed):
        paths.append(Path(cfg.STATE_DIR))
        raise RuntimeError("fixture failure")

    with pytest.raises(RuntimeError, match="fixture failure"):
        Backtester(orchestrator_factory=fail).run({})
    assert paths and not paths[0].exists()


def test_declared_custom_identity_round_trip():
    def first(cfg, seed):
        return object()

    def second(cfg, seed):
        return object()

    def run(factory, **labels):
        return Backtester(orchestrator_factory=factory, **labels).run({}).config_snapshot["research_manifest"]

    labels = dict(strategy_id="research-first", strategy_version="1.0")
    one = run(first, **labels)
    assert one == run(first, **labels)
    assert one["strategy_identity_status"] == "declared"
    assert one["strategy_id"].startswith("sha256:") and "research-first" not in json.dumps(one)
    assert one["strategy_id"] != run(second, strategy_id="research-second", strategy_version="1.0")["strategy_id"]
    newer = run(first, strategy_id="research-first", strategy_version="1.1")
    assert newer["strategy_id"] == one["strategy_id"] and newer["strategy_version"] == "1.1"
    unidentified = run(second)
    assert unidentified["strategy_identity_status"] == "unverified"
    assert unidentified["strategy_id"] is None and unidentified["strategy_version"] is None
    for invalid in ({"strategy_id": "alone"}, {"strategy_version": "1"},
                    {"strategy_id": "/private/fixture", "strategy_version": "1"}):
        with pytest.raises(ValueError, match="public strategy slug"):
            run(first, **invalid)


def test_revision_classifies_without_exporting_untracked_names(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    (tmp_path / ".gitignore").write_text("ignored/\n")
    git("add", ".gitignore")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
    sha = git("rev-parse", "HEAD")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "private.py").touch()
    (tmp_path / "private.log").touch()
    assert revision(tmp_path) == (sha, False, "clean")
    source = tmp_path / "nested" / "sensitive-name.py"
    source.parent.mkdir()
    source.write_text("private fixture contents")
    assert revision(tmp_path) == (sha, False, "untracked-source")
    (tmp_path / ".gitignore").write_text("ignored/\n*.log\n")
    assert revision(tmp_path) == (sha, True, "untracked-source")
    git("add", "nested")
    assert revision(tmp_path) == (sha, True, "modified")


def test_revision_unavailable_is_unknown(tmp_path, monkeypatch):
    assert revision(tmp_path) == (None, None, "unknown")
    (tmp_path / ".git").mkdir()
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("private fixture path")
    monkeypatch.setattr(subprocess, "check_output", unavailable)
    assert revision(tmp_path) == (None, None, "unknown")
