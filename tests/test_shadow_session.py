"""
Tests for automated shadow forward testing Phase 1 (trades/shadow_session.py).

Covers the pure ShadowEntryPolicy, the append-only decision log, shadow trade
creation, journal separation (shadow trades never touch manual_trades.jsonl),
and the safety guarantees (no broker imports, no secrets in shadow logs).
All tests are offline: no network, no broker, no real orders.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from strategy.day_trade_setups import DayTradeSetup, SetupType
from trades.shadow_session import (
    DECISION_ENTERED,
    DECISION_SKIPPED,
    ENTRY_SOURCE_AUTOMATED,
    SHADOW_TEST_TYPE,
    ShadowDecisionLog,
    ShadowEntryConfig,
    ShadowEntryPolicy,
    ShadowSessionStats,
    ShadowSkipReason,
    alert_age_seconds,
    create_shadow_trade_from_setup,
    parse_entry_cutoff_et,
    shadow_exit_price,
)
from trades.trade_tracker import ManualTrade, TradeStatus, TradeTracker

NOW = datetime(2026, 6, 12, 10, 30, 0)
SESSION_ID = "shadow-20260612-100000"


def _setup(**overrides) -> DayTradeSetup:
    """A fresh rank-1 high-confidence setup that passes every entry rule."""
    defaults = dict(
        ticker="SPY",
        timestamp=(NOW - timedelta(seconds=60)).isoformat(),
        setup_type=SetupType.VWAP_RECLAIM_LONG,
        is_valid=True,
        confidence=0.85,
        entry_trigger=100.0,
        stop_loss=98.0,
        target_1=104.0,
        target_2=106.0,
        risk_reward_1=2.0,
        risk_reward_2=3.0,
        reason="test setup",
        metadata={"rank": 1},
    )
    defaults.update(overrides)
    return DayTradeSetup(**defaults)


def _no_cooldowns() -> dict:
    return {"cooled_setup_types": {}, "cooled_tickers": {}}


def _evaluate(setup=None, open_trades=(), cooldowns=None, stale=False,
              entries_today=0, config=None, now=NOW):
    policy = ShadowEntryPolicy(config)
    evaluated_setup = setup
    if evaluated_setup is None:
        evaluated_setup = _setup(timestamp=(now - timedelta(seconds=60)).isoformat())
    return policy.evaluate(
        evaluated_setup,
        now=now,
        open_shadow_trades=list(open_trades),
        cooldown_summary=cooldowns if cooldowns is not None else _no_cooldowns(),
        stale_data=stale,
        entries_today=entries_today,
    )


def _open_shadow_trade(ticker="SPY", setup_type="vwap_reclaim_long") -> ManualTrade:
    return ManualTrade(
        ticker=ticker,
        setup_type=setup_type,
        entry_price=100.0,
        stop_loss=98.0,
        target_1=104.0,
        target_2=106.0,
        shares=5.0,
        entry_time=NOW.isoformat(),
        test_type=SHADOW_TEST_TYPE,
    )


# ---------------------------------------------------------------------------
# Entry policy: accept path
# ---------------------------------------------------------------------------

def test_qualifying_setup_is_entered():
    decision = _evaluate()
    assert decision.accepted
    assert decision.decision == "ENTER"
    assert decision.skip_reasons == []


def test_confidence_exactly_at_threshold_is_entered():
    decision = _evaluate(_setup(confidence=0.80))
    assert decision.accepted


def test_entry_before_default_cutoff_is_entered():
    decision = _evaluate(now=datetime(2026, 6, 12, 12, 59, 59))
    assert decision.accepted


def test_entry_at_or_after_default_cutoff_is_skipped():
    at_cutoff = _evaluate(now=datetime(2026, 6, 12, 13, 0, 0))
    after_cutoff = _evaluate(now=datetime(2026, 6, 12, 15, 0, 0))

    assert at_cutoff.skip_reasons == [ShadowSkipReason.AFTER_ENTRY_CUTOFF.value]
    assert after_cutoff.skip_reasons == [ShadowSkipReason.AFTER_ENTRY_CUTOFF.value]


def test_configurable_entry_cutoff_is_applied():
    config = ShadowEntryConfig(entry_cutoff_et=dt_time(14, 30))
    assert _evaluate(config=config, now=datetime(2026, 6, 12, 14, 29, 59)).accepted
    decision = _evaluate(config=config, now=datetime(2026, 6, 12, 14, 30, 0))
    assert decision.skip_reasons == [ShadowSkipReason.AFTER_ENTRY_CUTOFF.value]


# ---------------------------------------------------------------------------
# Entry policy: each skip reason
# ---------------------------------------------------------------------------

def test_low_confidence_is_skipped():
    decision = _evaluate(_setup(confidence=0.79))
    assert not decision.accepted
    assert decision.skip_reasons == [ShadowSkipReason.LOW_CONFIDENCE.value]


def test_rank_below_top_is_skipped():
    decision = _evaluate(_setup(metadata={"rank": 2}))
    assert decision.skip_reasons == [ShadowSkipReason.NOT_TOP_RANK.value]


def test_missing_rank_is_skipped_conservatively():
    decision = _evaluate(_setup(metadata={}))
    assert ShadowSkipReason.NOT_TOP_RANK.value in decision.skip_reasons


def test_old_alert_is_skipped():
    old = _setup(timestamp=(NOW - timedelta(seconds=301)).isoformat())
    decision = _evaluate(old)
    assert decision.skip_reasons == [ShadowSkipReason.ALERT_TOO_OLD.value]


def test_unparseable_alert_timestamp_is_skipped_conservatively():
    decision = _evaluate(_setup(timestamp="not-a-time"))
    assert ShadowSkipReason.ALERT_TOO_OLD.value in decision.skip_reasons


def test_stale_data_blocks_entry():
    decision = _evaluate(stale=True)
    assert decision.skip_reasons == [ShadowSkipReason.STALE_DATA.value]


def test_cooled_down_setup_type_is_skipped():
    cooldowns = {"cooled_setup_types": {"vwap_reclaim_long": 2}, "cooled_tickers": {}}
    decision = _evaluate(cooldowns=cooldowns)
    assert decision.skip_reasons == [ShadowSkipReason.SETUP_COOLED_DOWN.value]


def test_cooled_down_ticker_is_skipped():
    cooldowns = {"cooled_setup_types": {}, "cooled_tickers": {"SPY": 3}}
    decision = _evaluate(cooldowns=cooldowns)
    assert decision.skip_reasons == [ShadowSkipReason.TICKER_COOLED_DOWN.value]


def test_duplicate_open_shadow_trade_is_skipped():
    decision = _evaluate(open_trades=[_open_shadow_trade()])
    assert ShadowSkipReason.DUPLICATE_OPEN_TRADE.value in decision.skip_reasons


def test_same_ticker_different_setup_type_is_not_duplicate():
    other = _open_shadow_trade(setup_type="opening_range_breakout_long")
    decision = _evaluate(open_trades=[other])
    assert ShadowSkipReason.DUPLICATE_OPEN_TRADE.value not in decision.skip_reasons


def test_setup_type_not_allowed_is_skipped():
    config = ShadowEntryConfig(allowed_setup_types=["opening_range_breakout_long"])
    decision = _evaluate(config=config)
    assert decision.skip_reasons == [ShadowSkipReason.SETUP_TYPE_NOT_ALLOWED.value]


def test_max_open_trades_is_enforced():
    open_trades = [
        _open_shadow_trade(ticker=t, setup_type="opening_range_breakout_long")
        for t in ("QQQ", "NVDA", "AMD")
    ]
    decision = _evaluate(open_trades=open_trades)
    assert decision.skip_reasons == [ShadowSkipReason.MAX_OPEN_TRADES.value]


def test_max_daily_trades_is_enforced():
    decision = _evaluate(entries_today=10)
    assert decision.skip_reasons == [ShadowSkipReason.MAX_DAILY_TRADES.value]


def test_multiple_skip_reasons_are_all_preserved():
    old_low_conf = _setup(
        confidence=0.5,
        timestamp=(NOW - timedelta(seconds=600)).isoformat(),
    )
    decision = _evaluate(old_low_conf, stale=True, entries_today=10)
    assert set(decision.skip_reasons) == {
        ShadowSkipReason.LOW_CONFIDENCE.value,
        ShadowSkipReason.ALERT_TOO_OLD.value,
        ShadowSkipReason.STALE_DATA.value,
        ShadowSkipReason.MAX_DAILY_TRADES.value,
    }


def test_alert_age_seconds_parses_iso_timestamp():
    setup = _setup(timestamp=(NOW - timedelta(seconds=90)).isoformat())
    assert alert_age_seconds(setup, NOW) == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Shadow trade creation
# ---------------------------------------------------------------------------

def test_accepted_setup_creates_labelled_shadow_trade():
    trade = create_shadow_trade_from_setup(
        _setup(), shares=7.0, session_id=SESSION_ID, now=NOW, alert_id="a-1",
    )
    assert trade.test_type == SHADOW_TEST_TYPE
    assert ENTRY_SOURCE_AUTOMATED in trade.notes
    assert trade.ticker == "SPY"
    assert trade.setup_type == "vwap_reclaim_long"
    assert trade.entry_price == 100.0
    assert trade.shares == 7.0
    assert trade.setup_alert_id == "a-1"
    assert trade.setup_rank == 1
    assert trade.setup_confidence == pytest.approx(0.85)
    assert trade.status == TradeStatus.OPEN.value
    assert trade.metadata["session_id"] == SESSION_ID
    assert trade.metadata["entry_source"] == ENTRY_SOURCE_AUTOMATED
    assert trade.metadata["alert_age_seconds"] == pytest.approx(60.0)
    assert trade.metadata["simulated_fill"] is True
    assert trade.metadata["fill_basis"] == "entry_trigger"


# ---------------------------------------------------------------------------
# Journal separation: shadow trades never touch the manual journal
# ---------------------------------------------------------------------------

def test_shadow_trades_write_to_shadow_file_only(tmp_path):
    manual_path = tmp_path / "manual_trades.jsonl"
    shadow_path = tmp_path / "shadow_trades.jsonl"
    manual_path.write_text(
        json.dumps(_open_shadow_trade().to_dict()) + "\n", encoding="utf-8"
    )
    manual_before = manual_path.read_bytes()

    tracker = TradeTracker(storage_path=shadow_path)
    trade = create_shadow_trade_from_setup(
        _setup(), shares=5.0, session_id=SESSION_ID, now=NOW,
    )
    tracker.add_trade(trade)

    assert shadow_path.exists()
    assert manual_path.read_bytes() == manual_before


def test_all_persisted_shadow_trades_have_shadow_test_type(tmp_path):
    shadow_path = tmp_path / "shadow_trades.jsonl"
    tracker = TradeTracker(storage_path=shadow_path)
    for ticker in ("SPY", "QQQ"):
        tracker.add_trade(create_shadow_trade_from_setup(
            _setup(ticker=ticker), shares=1.0, session_id=SESSION_ID, now=NOW,
        ))

    persisted = tracker.list_all_trades()
    assert len(persisted) == 2
    assert all(t.test_type == SHADOW_TEST_TYPE for t in persisted)


def test_shadow_tracker_close_does_not_create_manual_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    trade = tracker.add_trade(create_shadow_trade_from_setup(
        _setup(), shares=5.0, session_id=SESSION_ID, now=NOW,
    ))
    tracker.close_trade(trade.trade_id, exit_price=104.0)

    assert not (tmp_path / "logs" / "manual_trades.jsonl").exists()
    closed = tracker.get_trade(trade.trade_id)
    assert closed.status == TradeStatus.CLOSED.value
    assert closed.realized_pnl == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Atomic journal rewrite
# ---------------------------------------------------------------------------

def test_write_all_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "shadow_trades.jsonl"
    tracker = TradeTracker(storage_path=path)
    trade = tracker.add_trade(create_shadow_trade_from_setup(
        _setup(), shares=2.0, session_id=SESSION_ID, now=NOW,
    ))
    tracker.close_trade(trade.trade_id, exit_price=101.0)

    assert path.exists()
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == TradeStatus.CLOSED.value


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------

def test_decision_log_records_entered_and_skipped(tmp_path):
    log = ShadowDecisionLog(tmp_path / "shadow_decisions.jsonl")
    entered = _evaluate()
    skipped = _evaluate(_setup(confidence=0.1))
    log.record(SESSION_ID, _setup(), entered, trade_id="t-1", alert_id="a-1", timestamp=NOW)
    log.record(SESSION_ID, _setup(confidence=0.1), skipped, alert_id="a-2", timestamp=NOW)

    lines = (tmp_path / "shadow_decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 2

    first, second = records
    assert first["decision"] == DECISION_ENTERED
    assert first["trade_id"] == "t-1"
    assert first["alert_id"] == "a-1"
    assert first["session_id"] == SESSION_ID
    assert first["ticker"] == "SPY"
    assert first["setup_type"] == "vwap_reclaim_long"
    assert first["rank"] == 1
    assert first["confidence"] == pytest.approx(0.85)
    assert first["skip_reasons"] == []

    assert second["decision"] == DECISION_SKIPPED
    assert second["trade_id"] is None
    assert second["skip_reasons"] == [ShadowSkipReason.LOW_CONFIDENCE.value]


def test_decision_log_is_append_only(tmp_path):
    path = tmp_path / "shadow_decisions.jsonl"
    log = ShadowDecisionLog(path)
    log.record(SESSION_ID, _setup(), _evaluate(), trade_id="t-1", timestamp=NOW)
    first_line = path.read_text(encoding="utf-8").splitlines()[0]

    log.record(SESSION_ID, _setup(), _evaluate(_setup(confidence=0.1)), timestamp=NOW)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == first_line  # earlier records are never rewritten


def test_decision_log_counts_entered_today(tmp_path):
    log = ShadowDecisionLog(tmp_path / "shadow_decisions.jsonl")
    log.record(SESSION_ID, _setup(), _evaluate(), trade_id="t-1", timestamp=NOW)
    log.record(SESSION_ID, _setup(), _evaluate(_setup(confidence=0.1)), timestamp=NOW)
    log.record(SESSION_ID, _setup(), _evaluate(), trade_id="t-2",
               timestamp=NOW - timedelta(days=1))

    assert log.count_entered_on(NOW.date()) == 1


# ---------------------------------------------------------------------------
# Safety: no broker modules, no secrets
# ---------------------------------------------------------------------------

def test_shadow_session_module_imports_no_broker_modules():
    source = Path("trades/shadow_session.py").read_text(encoding="utf-8")
    import_lines = [
        line for line in source.splitlines()
        if re.match(r"^\s*(import|from)\s+broker", line)
    ]
    assert import_lines == [], f"shadow_session must never import broker modules: {import_lines}"
    # Order/execution modules must not appear anywhere in the module source.
    assert "order_executor" not in source
    assert "alpaca" not in source.lower()


def test_no_secrets_in_shadow_logs(tmp_path, monkeypatch):
    fake_key = "FAKE_KEY_SHOULD_NEVER_APPEAR"
    fake_secret = "FAKE_SECRET_SHOULD_NEVER_APPEAR"
    monkeypatch.setenv("ALPACA_API_KEY", fake_key)
    monkeypatch.setenv("ALPACA_SECRET_KEY", fake_secret)

    shadow_path = tmp_path / "shadow_trades.jsonl"
    decisions_path = tmp_path / "shadow_decisions.jsonl"
    tracker = TradeTracker(storage_path=shadow_path)
    trade = tracker.add_trade(create_shadow_trade_from_setup(
        _setup(), shares=5.0, session_id=SESSION_ID, now=NOW,
    ))
    ShadowDecisionLog(decisions_path).record(
        SESSION_ID, _setup(), _evaluate(), trade_id=trade.trade_id, timestamp=NOW,
    )

    for path in (shadow_path, decisions_path):
        content = path.read_text(encoding="utf-8")
        assert fake_key not in content
        assert fake_secret not in content
        assert "webhook" not in content.lower()


# ===========================================================================
# Phase 2: exit pricing, session stats, and the forward-test-session command
# ===========================================================================

import pandas as pd
import typer.testing
from unittest.mock import patch

from main import app


@pytest.fixture
def cli_runner():
    return typer.testing.CliRunner()


def _bars(close=100.5, high=101.0, low=100.0, session_vwap=100.0) -> pd.DataFrame:
    """One-bar OHLCV frame for TradeMonitor."""
    return pd.DataFrame([{
        "close": close, "high": high, "low": low,
        "open": 100.2, "volume": 1000, "session_vwap": session_vwap,
    }])


#: Bars that trigger no exit for the default _setup() levels (stop 98, t1 104).
HOLD_BARS = _bars()
#: Bars where close < session_vwap but stop/targets untouched -> EXIT_VWAP_LOST.
VWAP_LOST_BARS = _bars(close=99.0, high=99.5, low=98.5, session_vwap=100.0)


def _scan_result(setups=None, stale=False, bars=None, data_age_seconds=None):
    """Build a patched _run_shadow_scan_once return value."""
    setups = setups if setups is not None else [_setup()]
    frame = (bars if bars is not None else HOLD_BARS).copy()
    if data_age_seconds is not None:
        frame.attrs["data_age_seconds"] = data_age_seconds
    return (setups, stale, {"SPY": frame})


def _run_session(
    cli_runner,
    scan_returns,
    extra_args=(),
    iterations=None,
    entry_time_et=datetime(2026, 6, 12, 10, 30, 0),
):
    """Invoke forward-test-session with a patched scan; returns CLI result."""
    if iterations is None:
        iterations = len(scan_returns) if isinstance(scan_returns, list) else 1
    side_effect = scan_returns if isinstance(scan_returns, list) else [scan_returns]
    with (
        patch("main._run_shadow_scan_once", side_effect=side_effect),
        patch("trades.shadow_session.now_eastern", return_value=entry_time_et),
    ):
        return cli_runner.invoke(app, [
            "forward-test-session", "--synthetic",
            "--max-iterations", str(iterations), "--interval-seconds", "0",
            *extra_args,
        ])


# -- exit price helper ------------------------------------------------------

def _freshness_frame(age_seconds, timestamp="2026-09-07T14:00:00", comparison="2026-09-07T14:20:00"):
    frame = HOLD_BARS.copy()
    frame.attrs.update(data_age_seconds=age_seconds, latest_bar_time=timestamp, provider_time=comparison)
    return frame


@pytest.mark.parametrize("ages,classification", [
    ([60, 900, 120], "ALL_FRESH"),
    ([60, 901, 120], "SYMBOL_LOCAL_STALE"),
    ([901, 1200, 60], "MIXED_FRESHNESS"),
    ([901, 1200, 1300], "FEED_WIDE_STALE"),
])
def test_freshness_symbol_classifications_preserve_gate(ages, classification):
    from data_providers.freshness import freshness_snapshot
    from main import _warn_if_provider_data_stale
    data = {symbol: _freshness_frame(age) for symbol, age in zip(["SPY", "QQQ", "ARM"], ages)}
    snapshot = freshness_snapshot(data, list(data), "5Min", "AlpacaData", "iex", NOW.isoformat())
    assert snapshot["classification"] == classification
    assert snapshot["symbols_checked"] == 3
    assert snapshot["stale_count"] == sum(age > 900 for age in ages)
    assert snapshot["median_bar_age_minutes"] == sorted(ages)[1] / 60
    assert snapshot["max_bar_age_minutes"] == max(ages) / 60
    assert _warn_if_provider_data_stale(data, "AlpacaData", "5Min") == any(age > 900 for age in ages)


def test_current_freshness_gate_first_warning_not_max_and_missing_ignored(capsys):
    from main import _warn_if_provider_data_stale
    data = {"SPY": _freshness_frame(960), "ARM": _freshness_frame(60000), "MISSING": None}
    assert _warn_if_provider_data_stale(data, "AlpacaData", "5Min")
    output = capsys.readouterr().out
    assert "16.0 minutes" in output
    assert "1000.0" not in output
    assert not _warn_if_provider_data_stale({"MISSING": None}, "AlpacaData", "5Min")
    assert not _warn_if_provider_data_stale(data, "Synthetic", "5Min")
    assert not _warn_if_provider_data_stale({"SPY": _freshness_frame(180)}, "AlpacaData", "1Min")
    assert _warn_if_provider_data_stale({"SPY": _freshness_frame(181)}, "AlpacaData", "1Min")


def test_missing_symbols_not_fresh_or_stale():
    from data_providers.freshness import freshness_snapshot
    snapshot = freshness_snapshot({"SPY": _freshness_frame(1200)}, ["SPY", "QQQ"],
                                  "5Min", "AlpacaData", "iex", NOW.isoformat())
    assert snapshot["classification"] == "INSUFFICIENT_DATA"
    assert snapshot["missing_symbols"] == ["QQQ"]
    assert snapshot["stale_symbols"] == ["SPY"]
    missing = snapshot["symbols"][1]
    assert missing["latest_bar_timestamp"] is None
    assert missing["stale"] is None
    assert missing["bar_age_minutes"] is None
    assert missing["comparison_timestamp_et"] is None


@pytest.mark.parametrize("timestamp", ["2026-09-07T10:00:00-04:00", "2026-09-07T14:00:00+00:00", "2026-09-07T14:00:00"])
def test_freshness_utc_naive_and_aware_timestamp_path(timestamp):
    from data_providers.alpaca_provider import AlpacaDataProvider
    from data_providers.freshness import freshness_snapshot
    frame = HOLD_BARS.tail(1).copy()
    frame["timestamp"] = timestamp
    cleaned = AlpacaDataProvider._clean_dataframe(frame)
    assert cleaned.index[0] == pd.Timestamp("2026-09-07T14:00:00")
    AlpacaDataProvider._attach_freshness_metadata(cleaned)
    observed = pd.Timestamp(cleaned.attrs["provider_time"])
    assert cleaned.attrs["data_age_seconds"] == max(0, (observed - cleaned.index[0]).total_seconds())
    snap = freshness_snapshot({"SPY": cleaned}, ["SPY"], "5Min", "AlpacaData", "iex", NOW.isoformat())
    assert snap["symbols"][0]["latest_bar_timestamp"] == "2026-09-07T14:00:00+00:00"
    assert pd.Timestamp(snap["symbols"][0]["comparison_timestamp_et"]).tz_convert("UTC").tz_localize(None) == observed


def test_latest_bar_missing_symbol_never_substitutes_another_symbol():
    from data_providers.alpaca_provider import AlpacaDataProvider
    from data_providers.base import DataProviderError
    with pytest.raises(DataProviderError, match="no latest bar"):
        AlpacaDataProvider._latest_response_to_dataframe({"QQQ": {"close": 100}}, "SPY")


def test_historical_missing_symbol_never_substitutes_another_symbol():
    from types import SimpleNamespace
    from data_providers.alpaca_provider import AlpacaDataProvider
    from data_providers.base import DataProviderError
    frame = HOLD_BARS.tail(1).copy()
    frame.index = pd.MultiIndex.from_tuples([("QQQ", pd.Timestamp("2026-09-07T14:00:00"))], names=["symbol", "timestamp"])
    with pytest.raises(DataProviderError, match="no bars for 'SPY'"):
        AlpacaDataProvider._response_to_dataframe(SimpleNamespace(df=frame), "SPY")


@pytest.mark.parametrize("layout", ["column", "multiindex", "mapping", "latest"])
def test_alpaca_parser_selects_only_requested_symbol(layout):
    from types import SimpleNamespace
    from data_providers.alpaca_provider import AlpacaDataProvider
    from data_providers.base import DataProviderError
    bars = [{"timestamp": "2026-09-17T14:00:00Z", "open": 100, "high": 102,
             "low": 99, "close": price, "volume": 10} for price in (101, 102)]
    if layout in {"column", "multiindex"}:
        df = pd.DataFrame(bars).assign(symbol=["SPY", "QQQ"])
        if layout == "multiindex":
            df = df.set_index(["symbol", "timestamp"])
        response = SimpleNamespace(df=df)
        parse = AlpacaDataProvider._response_to_dataframe
    elif layout == "mapping":
        response = {"SPY": [bars[0]], "QQQ": [bars[1]]}
        parse = AlpacaDataProvider._response_to_dataframe
    else:
        response = {"SPY": bars[0], "QQQ": bars[1]}
        parse = AlpacaDataProvider._latest_response_to_dataframe
    actual = parse(response, "SPY")
    expected = AlpacaDataProvider._clean_dataframe(pd.DataFrame([bars[0]]))
    pd.testing.assert_frame_equal(actual, expected)
    with pytest.raises(DataProviderError):
        parse(response, "ARM")


def test_snapshot_deduplicates_requests_and_preserves_evidence():
    from data_providers.freshness import freshness_snapshot
    data = {"SPY": _freshness_frame(1200), "QQQ": HOLD_BARS.copy()}
    snapshot = freshness_snapshot(data, ["SPY", "SPY", "QQQ", "ARM"], "5Min", "AlpacaData", "iex", NOW.isoformat())
    assert snapshot["symbols_requested"] == 3
    assert snapshot["symbols_checked"] == 1
    assert snapshot["stale_symbols"] == ["SPY"]
    assert snapshot["unknown_age_symbols"] == ["QQQ"]
    assert snapshot["missing_symbols"] == ["ARM"]
    assert snapshot["classification"] == "INSUFFICIENT_DATA"
    row = snapshot["symbols"][0]
    assert row == {"symbol": "SPY", "provider": "AlpacaData", "feed": "iex",
                   "latest_bar_timestamp": "2026-09-07T14:00:00+00:00",
                   "comparison_timestamp_et": "2026-09-07T10:20:00-04:00",
                   "bar_age_minutes": 20, "stale": True, "missing": False}


@pytest.mark.parametrize("stale", [False, True])
def test_session_persists_symbol_details_only_on_stale_event(cli_runner, tmp_path, monkeypatch, stale):
    from types import SimpleNamespace
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("main._get_named_data_provider", lambda *a, **kw: SimpleNamespace(provider_name="AlpacaData", feed="iex"))
    frame = _freshness_frame(1200 if stale else 60)
    result = _run_session(cli_runner, ([], stale, {"SPY": frame}), extra_args=["--ticker", "SPY,QQQ"])
    assert result.exit_code == 0, result.output
    state = ShadowSessionState.load()
    assert state.counters["missing_bar_observations"] == 1
    events = state.counters["stale_data_events"]
    assert len(events) == int(stale)
    if stale:
        assert events[0]["freshness_snapshot"]["stale_symbols"] == ["SPY"]
        assert events[0]["freshness_snapshot"]["missing_symbols"] == ["QQQ"]
        assert "INSUFFICIENT_DATA=1" in result.output
    assert state.counters["data_quality"] == ("PARTIAL_DATA" if stale else "CLEAN")

def test_shadow_exit_price_uses_levels_and_current_price():
    trade = _open_shadow_trade()
    assert shadow_exit_price("STOP_HIT", trade, 97.5) == 98.0
    assert shadow_exit_price("TARGET_1_HIT", trade, 104.4) == 104.0
    assert shadow_exit_price("TARGET_2_HIT", trade, 106.6) == 106.0
    assert shadow_exit_price("EXIT_VWAP_LOST", trade, 99.0) == 99.0


def test_shadow_exit_price_falls_back_on_malformed_level():
    trade = _open_shadow_trade()
    trade.stop_loss = 0.0
    assert shadow_exit_price("STOP_HIT", trade, 97.5) == 97.5


# -- session stats ----------------------------------------------------------

def test_session_stats_summary_lines():
    stats = ShadowSessionStats()
    stats.record_iteration(1)
    stats.record_entry()
    stats.record_skip(["LOW_CONFIDENCE", "STALE_DATA"])
    stats.record_skip(["LOW_CONFIDENCE"])
    stats.record_close(-12.5)
    stats.record_discord_failure()

    lines = stats.summary_lines(open_count=1)
    text = "\n".join(lines)
    assert "SIMULATION ONLY" in text
    assert "Data quality: CLEAN" in text
    assert "Entered: 1" in text
    assert "Skipped: 2" in text
    assert "LOW_CONFIDENCE: 2" in text
    assert "STALE_DATA: 1" in text
    assert "Closed: 1" in text
    assert "Open shadow trades: 1" in text
    assert "-12.50" in text
    assert "Discord failures: 1" in text


def test_session_stats_records_data_gap_as_partial_data():
    stats = ShadowSessionStats()
    stats.record_data_gap(
        "2026-06-25T10:00:00",
        "2026-06-25T11:10:00",
        elapsed_seconds=4200,
        expected_interval_seconds=300,
    )

    assert stats.data_quality == "PARTIAL_DATA"
    assert stats.data_gap_count == 1
    assert stats.largest_data_gap_seconds == pytest.approx(4200)
    counters = stats.counters()
    assert counters["has_data_gaps"] is True
    assert counters["data_quality"] == "PARTIAL_DATA"
    assert counters["data_gaps"][0]["estimated_missed_intervals"] == 13


def test_session_stats_reports_stale_provider_events():
    stats = ShadowSessionStats()
    stats.record_stale_data_event("2026-06-25T10:05:00", 600)
    stats.record_data_quality_failure("STALE_DATA", "2026-06-25T10:05:00")
    stats.record_stale_data_event("2026-06-25T10:10:00", 900)

    counters = stats.counters()
    assert counters["stale_warning_count"] == 2
    assert counters["max_stale_bar_age_seconds"] == pytest.approx(900)
    assert counters["first_stale_timestamp"] == "2026-06-25T10:05:00"
    assert counters["last_stale_timestamp"] == "2026-06-25T10:10:00"
    assert counters["stale_data_caused_partial"] is True
    text = "\n".join(stats.summary_lines(open_count=0))
    assert "Stale warnings: 2" in text
    assert "Max stale bar age: 15.0 min" in text
    assert "Stale data caused PARTIAL_DATA: yes" in text


# -- CLI safety gates -------------------------------------------------------

def test_session_flagless_refused_when_market_closed(cli_runner, tmp_path, monkeypatch):
    """Flagless live invocation outside market hours must be refused (Phase 3:
    during market hours a flagless live session is legitimate and bounded by
    market close, so the clock is pinned to a Saturday here)."""
    import trades.shadow_session as shadow_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        shadow_module, "now_eastern",
        lambda: datetime(2026, 6, 13, 10, 0, 0),  # Saturday — market closed
    )
    result = cli_runner.invoke(app, ["forward-test-session"])
    assert result.exit_code == 1
    assert "market is closed" in result.output.lower()


def test_synthetic_bounded_session_completes(cli_runner, tmp_path, monkeypatch):
    """Real synthetic scan (no patching): session must terminate and stay offline."""
    monkeypatch.chdir(tmp_path)
    result = cli_runner.invoke(app, [
        "forward-test-session", "--synthetic",
        "--max-iterations", "2", "--interval-seconds", "0",
    ])
    assert result.exit_code == 0
    assert "SIMULATION ONLY" in result.output
    assert "No real orders were placed." in result.output
    assert not (tmp_path / "logs" / "manual_trades.jsonl").exists()


def test_forward_test_session_accepts_momentum_universe_watchlist(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    watchlists_dir = tmp_path / "watchlists"
    watchlists_dir.mkdir()
    (watchlists_dir / "momentum_universe.json").write_text(json.dumps({
        "name": "momentum_universe",
        "tickers": ["SPY", "QQQ", "HOOD", "SMCI"],
    }), encoding="utf-8")

    result = _run_session(
        cli_runner,
        _scan_result(setups=[]),
        extra_args=["--watchlist", "momentum_universe"],
    )

    assert result.exit_code == 0, result.output
    assert "Tickers:" in result.output
    assert "HOOD" in result.output
    assert "SMCI" in result.output
    assert "No real orders were placed." in result.output
    assert not (tmp_path / "logs" / "manual_trades.jsonl").exists()

    state = ShadowSessionState.load(tmp_path / "state" / "shadow_session.json")
    assert state.config["watchlist_name"] == "momentum_universe"
    assert state.config["ticker_count"] == 4

    history_records = [
        json.loads(line)
        for line in (tmp_path / "logs" / "shadow_sessions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert history_records[-1]["config"]["watchlist_name"] == "momentum_universe"
    assert history_records[-1]["config"]["ticker_count"] == 4


def test_session_never_constructs_order_executor_or_broker_client(
    cli_runner, tmp_path, monkeypatch,
):
    import broker.order_executor as order_executor_module
    import broker.alpaca_client as alpaca_client_module

    def _boom(*args, **kwargs):
        raise AssertionError("forward-test-session must never touch broker order modules")

    monkeypatch.setattr(order_executor_module.OrderExecutor, "__init__", _boom)
    monkeypatch.setattr(alpaca_client_module.AlpacaClientWrapper, "__init__", _boom)
    monkeypatch.chdir(tmp_path)

    result = _run_session(cli_runner, _scan_result())
    assert result.exit_code == 0


def test_session_keyboard_interrupt_leaves_trades_open(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run_session(
        cli_runner,
        [_scan_result(), KeyboardInterrupt()],
        iterations=5,
    )
    assert result.exit_code == 0
    assert "left OPEN" in result.output

    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    open_trades = tracker.list_open_trades()
    assert len(open_trades) == 1
    assert open_trades[0].status == TradeStatus.OPEN.value


# -- CLI functional behavior -------------------------------------------------

def test_session_accepted_setup_creates_shadow_trade(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run_session(cli_runner, _scan_result())
    assert result.exit_code == 0
    assert "SHADOW ENTRY" in result.output
    assert "no real order placed" in result.output.lower()

    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    trades = tracker.list_all_trades()
    assert len(trades) == 1
    trade = trades[0]
    assert trade.test_type == SHADOW_TEST_TYPE
    assert trade.ticker == "SPY"
    assert trade.metadata["entry_source"] == ENTRY_SOURCE_AUTOMATED
    assert trade.metadata["simulated_fill"] is True
    # sim-notional default 1000 at entry 100.0 -> 10 shares
    assert trade.shares == pytest.approx(10.0)
    assert not (tmp_path / "logs" / "manual_trades.jsonl").exists()


def test_session_open_trade_stays_open_when_session_end_close_disabled(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    result = _run_session(cli_runner, _scan_result())
    assert result.exit_code == 0

    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    trades = tracker.list_all_trades()
    assert len(trades) == 1
    assert trades[0].status == TradeStatus.OPEN.value
    assert "exit_signal" not in trades[0].metadata


def test_session_open_trade_closes_at_session_end_when_enabled(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    result = _run_session(
        cli_runner,
        _scan_result(bars=_bars(close=101.25, high=101.5, low=100.5, session_vwap=100.0)),
        extra_args=["--close-open-at-session-end"],
    )
    assert result.exit_code == 0, result.output
    assert "SESSION_END" in result.output
    assert "no real order placed" in result.output.lower()

    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    trades = tracker.list_all_trades()
    assert len(trades) == 1
    trade = trades[0]
    assert trade.status == TradeStatus.CLOSED.value
    assert trade.exit_price == pytest.approx(101.25)
    assert trade.metadata["exit_signal"] == "SESSION_END"
    assert trade.metadata["session_end_exit"] is True
    assert trade.realized_pnl == pytest.approx((101.25 - 100.0) * trade.shares)


def test_session_end_close_flag_does_not_force_close_on_ctrl_c(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    result = _run_session(
        cli_runner,
        [_scan_result(), KeyboardInterrupt()],
        extra_args=["--close-open-at-session-end"],
        iterations=5,
    )
    assert result.exit_code == 0, result.output

    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    trades = tracker.list_all_trades()
    assert len(trades) == 1
    assert trades[0].status == TradeStatus.OPEN.value
    assert "exit_signal" not in trades[0].metadata


def test_session_low_confidence_logs_skipped_and_no_trade(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run_session(cli_runner, _scan_result(setups=[_setup(confidence=0.5)]))
    assert result.exit_code == 0
    assert "SKIP" in result.output

    decisions = [
        json.loads(line) for line in
        (tmp_path / "logs" / "shadow_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(decisions) == 1
    assert decisions[0]["decision"] == DECISION_SKIPPED
    assert ShadowSkipReason.LOW_CONFIDENCE.value in decisions[0]["skip_reasons"]

    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    assert tracker.list_all_trades() == []


def test_session_stale_data_blocks_entry(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run_session(
        cli_runner,
        _scan_result(stale=True, data_age_seconds=1500),
    )
    assert result.exit_code == 0

    decisions = [
        json.loads(line) for line in
        (tmp_path / "logs" / "shadow_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert decisions[0]["decision"] == DECISION_SKIPPED
    assert ShadowSkipReason.STALE_DATA.value in decisions[0]["skip_reasons"]
    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    assert tracker.list_all_trades() == []
    state = ShadowSessionState.load(tmp_path / "state" / "shadow_session.json")
    assert state.counters["stale_warning_count"] == 1
    assert state.counters["max_stale_bar_age_seconds"] == pytest.approx(1500)
    assert state.counters["stale_data_caused_partial"] is True
    assert "Max stale bar age: 25.0 min" in result.output


def test_forward_session_enters_before_entry_cutoff(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run_session(
        cli_runner,
        _scan_result(),
        entry_time_et=datetime(2026, 6, 12, 12, 59, 59),
    )

    assert result.exit_code == 0, result.output
    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    assert len(tracker.list_open_trades()) == 1
    assert "SHADOW ENTRY (simulated)" in result.output


def test_forward_session_logs_skip_at_entry_cutoff(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run_session(
        cli_runner,
        _scan_result(),
        entry_time_et=datetime(2026, 6, 12, 13, 0, 0),
    )

    assert result.exit_code == 0, result.output
    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    assert tracker.list_open_trades() == []
    decisions = [
        json.loads(line) for line in
        (tmp_path / "logs" / "shadow_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert decisions[0]["decision"] == DECISION_SKIPPED
    assert decisions[0]["skip_reasons"] == [ShadowSkipReason.AFTER_ENTRY_CUTOFF.value]
    assert "AFTER_ENTRY_CUTOFF" in result.output


def test_open_trade_can_exit_after_entry_cutoff(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    prior = tracker.add_trade(create_shadow_trade_from_setup(
        _setup(), shares=5.0, session_id="shadow-OLD", now=NOW,
    ))

    _save_recoverable_state(prior)

    result = _run_session(
        cli_runner,
        _scan_result(setups=[_setup(ticker="QQQ")], bars=VWAP_LOST_BARS),
        entry_time_et=datetime(2026, 6, 12, 14, 0, 0),
    )

    assert result.exit_code == 0, result.output
    assert tracker.get_trade(prior.trade_id).status == TradeStatus.CLOSED.value
    decisions = [
        json.loads(line) for line in
        (tmp_path / "logs" / "shadow_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert ShadowSkipReason.AFTER_ENTRY_CUTOFF.value in decisions[0]["skip_reasons"]
    assert "SHADOW EXIT (simulated)" in result.output


def test_session_duplicate_open_trade_skipped_on_second_iteration(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    result = _run_session(cli_runner, [_scan_result(), _scan_result()])
    assert result.exit_code == 0

    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    assert len(tracker.list_all_trades()) == 1

    decisions = [
        json.loads(line) for line in
        (tmp_path / "logs" / "shadow_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [d["decision"] for d in decisions] == [DECISION_ENTERED, DECISION_SKIPPED]
    assert ShadowSkipReason.DUPLICATE_OPEN_TRADE.value in decisions[1]["skip_reasons"]


def test_session_vwap_lost_closes_shadow_trade(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run_session(cli_runner, [
        _scan_result(),                                  # iter 1: enter, bars HOLD
        _scan_result(setups=[], bars=VWAP_LOST_BARS),    # iter 2: VWAP lost -> close
    ])
    assert result.exit_code == 0
    assert "SHADOW EXIT" in result.output

    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    trades = tracker.list_all_trades()
    assert len(trades) == 1
    closed = trades[0]
    assert closed.status == TradeStatus.CLOSED.value
    assert closed.metadata["exit_signal"] == "EXIT_VWAP_LOST"
    # EXIT_VWAP_LOST fills at the monitor's current price (bar close)
    assert closed.exit_price == pytest.approx(99.0)
    assert closed.realized_pnl == pytest.approx((99.0 - 100.0) * closed.shares)


def test_session_decision_log_records_every_evaluated_setup(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    setups = [
        _setup(ticker="SPY"),
        _setup(ticker="QQQ", confidence=0.5, metadata={"rank": 2}),
    ]
    result = _run_session(cli_runner, _scan_result(setups=setups))
    assert result.exit_code == 0

    decisions = [
        json.loads(line) for line in
        (tmp_path / "logs" / "shadow_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(decisions) == 2
    assert decisions[0]["decision"] == DECISION_ENTERED
    assert decisions[1]["decision"] == DECISION_SKIPPED
    assert set(decisions[1]["skip_reasons"]) == {
        ShadowSkipReason.LOW_CONFIDENCE.value,
        ShadowSkipReason.NOT_TOP_RANK.value,
    }


def test_session_summary_prints_counts(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run_session(cli_runner, [
        _scan_result(),
        _scan_result(setups=[_setup(confidence=0.5)], bars=VWAP_LOST_BARS),
    ])
    assert result.exit_code == 0
    assert "Shadow Session Summary" in result.output
    assert "Entered: 1" in result.output
    assert "Skipped: 1" in result.output
    assert "Closed: 1" in result.output
    assert "Open shadow trades: 0" in result.output


def test_session_counts_discord_failures(cli_runner, tmp_path, monkeypatch):
    import alerts.discord_alerts as discord_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(discord_module, "is_discord_configured", lambda: True)
    monkeypatch.setattr(discord_module, "send_discord_message", lambda content: False)

    result = _run_session(cli_runner, _scan_result())
    assert result.exit_code == 0
    assert "Discord failures: 1" in result.output


def test_session_sends_shadow_discord_messages(cli_runner, tmp_path, monkeypatch):
    import alerts.discord_alerts as discord_module

    sent = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(discord_module, "is_discord_configured", lambda: True)
    monkeypatch.setattr(
        discord_module, "send_discord_message",
        lambda content: sent.append(content) or True,
    )

    result = _run_session(cli_runner, [
        _scan_result(),
        _scan_result(setups=[], bars=VWAP_LOST_BARS),
    ])
    assert result.exit_code == 0
    # entry, exit, end-of-session summary
    assert len(sent) == 3
    assert "SHADOW ENTRY" in sent[0]
    assert "SHADOW EXIT" in sent[1]
    assert "SHADOW SESSION SUMMARY" in sent[2]
    assert all("No real order placed." in message for message in sent)


# ===========================================================================
# Phase 3: market hours, session state, resume, failure handling
# ===========================================================================

from trades.shadow_session import (
    SESSION_ENDED,
    SESSION_ERROR,
    SESSION_INTERRUPTED,
    SESSION_RUNNING,
    SHADOW_SESSION_LOCK_FILE,
    ShadowSessionState,
    adopt_open_shadow_trades,
    is_market_open_et,
    market_close_et,
    parse_end_time_et,
)

#: Tuesday 2026-06-09 10:00 ET — regular market hours.
MARKET_OPEN_NOW = datetime(2026, 6, 9, 10, 0, 0)
#: Saturday 2026-06-13 10:00 ET — market closed.
MARKET_CLOSED_NOW = datetime(2026, 6, 13, 10, 0, 0)


class _FakeProvider:
    provider_name = "AlpacaData"


def _patch_live_mode(monkeypatch, clock):
    """Patch the ET clock and the data-provider factory for live-mode CLI tests."""
    import trades.shadow_session as shadow_module

    monkeypatch.setattr(shadow_module, "now_eastern", clock)
    monkeypatch.setattr("main._get_named_data_provider",
                        lambda *args, **kwargs: _FakeProvider())


def _fixed_clock(moment):
    return lambda: moment


def _stepping_clock(times):
    """Clock returning each time once, then repeating the last forever."""
    sequence = list(times)

    def _now():
        return sequence.pop(0) if len(sequence) > 1 else sequence[0]

    return _now


def _run_live_session(cli_runner, scan_returns, extra_args=()):
    side_effect = scan_returns if isinstance(scan_returns, list) else [scan_returns]
    with patch("main._run_shadow_scan_once", side_effect=side_effect):
        return cli_runner.invoke(app, [
            "forward-test-session", "--provider", "alpaca",
            "--interval-seconds", "0", *extra_args,
        ])


# -- market-hours helpers ------------------------------------------------------

def test_market_hours_helpers():
    assert is_market_open_et(MARKET_OPEN_NOW)
    assert not is_market_open_et(MARKET_CLOSED_NOW)                       # Saturday
    assert not is_market_open_et(datetime(2026, 6, 9, 9, 29, 0))          # pre-open
    assert not is_market_open_et(datetime(2026, 6, 9, 16, 0, 0))          # at close
    assert market_close_et(MARKET_OPEN_NOW) == datetime(2026, 6, 9, 16, 0, 0)
    assert parse_end_time_et("15:30", MARKET_OPEN_NOW) == datetime(2026, 6, 9, 15, 30, 0)
    assert parse_entry_cutoff_et("13:00") == dt_time(13, 0)
    with pytest.raises(ValueError):
        parse_end_time_et("not-a-time", MARKET_OPEN_NOW)
    with pytest.raises(ValueError):
        parse_entry_cutoff_et("not-a-time")


# -- live mode gates -----------------------------------------------------------

def test_live_mode_allowed_during_market_hours(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch_live_mode(monkeypatch, _fixed_clock(MARKET_OPEN_NOW))

    result = _run_live_session(cli_runner, _scan_result(), ["--max-iterations", "1"])
    assert result.exit_code == 0
    assert "Market status: open" in result.output
    assert "Session end: 16:00 ET" in result.output


def test_closed_market_refused_without_ignore_flag(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch_live_mode(monkeypatch, _fixed_clock(MARKET_CLOSED_NOW))

    result = _run_live_session(cli_runner, _scan_result(), ["--max-iterations", "1"])
    assert result.exit_code == 1
    assert "market is closed" in result.output.lower()
    assert "--ignore-market-hours" in result.output


def test_closed_market_allowed_with_ignore_flag_and_bound(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch_live_mode(monkeypatch, _fixed_clock(MARKET_CLOSED_NOW))

    result = _run_live_session(
        cli_runner, _scan_result(),
        ["--ignore-market-hours", "--max-iterations", "1"],
    )
    assert result.exit_code == 0
    assert "Market status: closed" in result.output


def test_ignore_market_hours_without_bound_is_refused(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch_live_mode(monkeypatch, _fixed_clock(MARKET_CLOSED_NOW))

    result = _run_live_session(cli_runner, _scan_result(), ["--ignore-market-hours"])
    assert result.exit_code == 1
    assert "unbounded" in result.output.lower()


def test_end_time_stops_session(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Validation sees 10:00; iteration 1 runs at 10:01; iteration 2 sees 10:10 >= 10:05.
    _patch_live_mode(monkeypatch, _stepping_clock([
        MARKET_OPEN_NOW,
        datetime(2026, 6, 9, 10, 1, 0),
        datetime(2026, 6, 9, 10, 10, 0),
    ]))

    result = _run_live_session(
        cli_runner,
        [_scan_result(), _scan_result()],
        ["--end-time", "10:05"],
    )
    assert result.exit_code == 0
    assert "Session end: 10:05 ET" in result.output
    assert "Shadow iteration 1" in result.output
    assert "Shadow iteration 2" not in result.output
    assert "Reached session end time" in result.output


def test_end_time_in_past_is_refused(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch_live_mode(monkeypatch, _fixed_clock(MARKET_OPEN_NOW))

    result = _run_live_session(cli_runner, _scan_result(), ["--end-time", "09:45"])
    assert result.exit_code == 1
    assert "already passed" in result.output


# -- session state ---------------------------------------------------------------

def test_session_state_written_running_then_ended(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    observed = []
    original_save = ShadowSessionState.save

    def _spy_save(self, *args, **kwargs):
        observed.append(self.status)
        return original_save(self, *args, **kwargs)

    monkeypatch.setattr(ShadowSessionState, "save", _spy_save)

    result = _run_session(cli_runner, _scan_result())
    assert result.exit_code == 0
    assert observed[0] == SESSION_RUNNING
    assert observed[-1] == SESSION_ENDED

    state = ShadowSessionState.load(tmp_path / "state" / "shadow_session.json")
    assert state is not None
    assert state.status == SESSION_ENDED
    assert state.counters["entered"] == 1
    assert state.counters["closed"] == 0
    assert state.last_iteration_at is not None
    assert len(state.open_trade_ids) == 1


def test_forward_test_session_creates_lock(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("trades.shadow_session.release_shadow_session_lock", lambda session_id: False)

    result = _run_session(cli_runner, _scan_result(setups=[]))

    assert result.exit_code == 0, result.output
    lock = json.loads((tmp_path / SHADOW_SESSION_LOCK_FILE).read_text(encoding="utf-8"))
    assert lock["session_id"].startswith("shadow-")
    assert lock["metadata"]["command"] == "forward-test-session"
    assert lock["metadata"]["ticker_count"] == 1


def test_completed_forward_test_session_removes_lock(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = _run_session(cli_runner, _scan_result(setups=[]))

    assert result.exit_code == 0, result.output
    assert not (tmp_path / SHADOW_SESSION_LOCK_FILE).exists()


def test_forward_test_session_refuses_active_running_lock(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = ShadowSessionState(
        session_id="shadow-active",
        started_at=datetime.now().isoformat(),
        status=SESSION_RUNNING,
    )
    state.save(tmp_path / "state" / "shadow_session.json")
    lock_path = tmp_path / SHADOW_SESSION_LOCK_FILE
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({
        "session_id": "shadow-active",
        "started_at": datetime.now().isoformat(),
        "pid": 12345,
        "metadata": {"command": "forward-test-session"},
    }), encoding="utf-8")

    result = _run_session(cli_runner, _scan_result(setups=[]))

    assert result.exit_code == 1
    assert "already be running" in result.output
    assert "shadow-active" in result.output


def test_forward_test_session_clears_ended_lock_and_starts(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = ShadowSessionState(
        session_id="shadow-ended",
        started_at=datetime.now().isoformat(),
        status=SESSION_ENDED,
    )
    state.save(tmp_path / "state" / "shadow_session.json")
    lock_path = tmp_path / SHADOW_SESSION_LOCK_FILE
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({
        "session_id": "shadow-ended",
        "started_at": datetime.now().isoformat(),
        "pid": 12345,
        "metadata": {"command": "forward-test-session"},
    }), encoding="utf-8")

    result = _run_session(cli_runner, _scan_result(setups=[]))

    assert result.exit_code == 0, result.output
    assert not lock_path.exists()


def test_forward_test_session_clears_stale_running_lock_and_starts(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    stale_started = (datetime.now() - timedelta(hours=13)).isoformat()
    state = ShadowSessionState(
        session_id="shadow-stale",
        started_at=stale_started,
        status=SESSION_RUNNING,
    )
    state.save(tmp_path / "state" / "shadow_session.json")
    lock_path = tmp_path / SHADOW_SESSION_LOCK_FILE
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({
        "session_id": "shadow-stale",
        "started_at": stale_started,
        "pid": 12345,
        "metadata": {"command": "forward-test-session"},
    }), encoding="utf-8")

    result = _run_session(cli_runner, _scan_result(setups=[]))

    assert result.exit_code == 0, result.output
    assert not lock_path.exists()


def test_session_state_tracks_no_ranked_setup_iterations_without_fake_logs(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    result = _run_session(
        cli_runner,
        [_scan_result(setups=[]), _scan_result(setups=[])],
        iterations=2,
    )

    assert result.exit_code == 0, result.output
    state = ShadowSessionState.load(tmp_path / "state" / "shadow_session.json")
    assert state is not None
    assert state.counters["iterations_total"] == 2
    assert state.counters["no_ranked_setup_iterations"] == 2
    assert state.counters["ranked_setup_iterations"] == 0
    assert state.counters["evaluated_setups"] == 0
    assert state.counters["entered"] == 0
    assert state.counters["skipped"] == 0
    assert state.counters["closed"] == 0
    assert state.counters["open"] == 0
    assert "No-ranked-setup iterations: 2" in result.output
    assert not (tmp_path / "logs" / "shadow_trades.jsonl").exists()
    assert not (tmp_path / "logs" / "shadow_decisions.jsonl").exists()
    assert not (tmp_path / "logs" / "manual_trades.jsonl").exists()
    history_path = tmp_path / "logs" / "shadow_sessions.jsonl"
    records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["session_id"] == state.session_id
    assert records[0]["status"] == SESSION_ENDED
    assert records[0]["counters"]["iterations_total"] == 2
    assert records[0]["counters"]["no_ranked_setup_iterations"] == 2


def test_session_detects_large_iteration_gap_and_marks_partial_data(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    class FakeDateTime(datetime):
        _times = [
            datetime(2026, 6, 25, 10, 0, 0),  # session id
            datetime(2026, 6, 25, 10, 0, 0),  # state started_at
            datetime(2026, 6, 25, 10, 0, 0),  # iteration 1
            datetime(2026, 6, 25, 11, 10, 0), # iteration 2: 70-minute gap
            datetime(2026, 6, 25, 11, 10, 0), # final timestamps
        ]

        @classmethod
        def now(cls, tz=None):
            value = cls._times.pop(0) if len(cls._times) > 1 else cls._times[0]
            return value

    monkeypatch.setattr("main.datetime", FakeDateTime)

    result = _run_session(
        cli_runner,
        [_scan_result(setups=[]), _scan_result(setups=[])],
        iterations=2,
    )

    assert result.exit_code == 0, result.output
    assert "WARNING: data gap detected between iterations: 70.0 minutes" in result.output
    assert "Data quality: PARTIAL_DATA" in result.output
    assert "Data gaps: 1" in result.output
    assert "Largest data gap: 70.0 min" in result.output
    assert "SESSION_SUSPENSION" in result.output
    assert "Partial-data reason: ITERATION_GAP" in result.output
    assert "Extreme/session-suspension gaps: 1" in result.output

    state = ShadowSessionState.load(tmp_path / "state" / "shadow_session.json")
    assert state is not None
    assert state.counters["data_quality"] == "PARTIAL_DATA"
    assert state.counters["data_gap_count"] == 1
    assert state.counters["largest_data_gap_seconds"] == pytest.approx(4200)
    assert state.counters["normal_data_gap_count"] == 0
    assert state.counters["extreme_data_gap_count"] == 1
    assert state.counters["max_iteration_gap_seconds"] == 4200
    assert state.counters["partial_data_reasons"] == ["ITERATION_GAP"]
    assert state.counters["data_gaps"][0]["start_timestamp"] == "2026-06-25T10:00:00"
    assert state.counters["data_gaps"][0]["end_timestamp"] == "2026-06-25T11:10:00"


def test_partial_session_blocks_new_entries_after_large_data_gap(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    class FakeDateTime(datetime):
        _times = [
            datetime(2026, 6, 25, 10, 0, 0),
            datetime(2026, 6, 25, 10, 0, 0),
            datetime(2026, 6, 25, 10, 0, 0),
            datetime(2026, 6, 25, 11, 10, 0),
            datetime(2026, 6, 25, 11, 10, 0),
        ]

        @classmethod
        def now(cls, tz=None):
            value = cls._times.pop(0) if len(cls._times) > 1 else cls._times[0]
            return value

    monkeypatch.setattr("main.datetime", FakeDateTime)
    result = _run_session(
        cli_runner,
        [_scan_result(setups=[]), _scan_result(setups=[_setup(ticker="QQQ")])],
        iterations=2,
    )

    assert result.exit_code == 0, result.output
    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    assert tracker.list_all_trades() == []
    decisions = [
        json.loads(line) for line in
        (tmp_path / "logs" / "shadow_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert decisions[-1]["decision"] == DECISION_SKIPPED
    assert ShadowSkipReason.PARTIAL_SESSION.value in decisions[-1]["skip_reasons"]


@pytest.mark.parametrize("gap_seconds,extreme,interval_seconds", [
    (1200, 0, 300), (3600, 1, 300), (7200, 1, 300),
    (3600, 0, 600), (7200, 1, 600), (3600, 1, 60),
])
def test_gap_persisted_before_scan_and_existing_trade_can_exit(
    cli_runner, tmp_path, monkeypatch, gap_seconds, extreme, interval_seconds,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("main.time.sleep", lambda seconds: None)
    moments = iter([NOW, NOW, NOW, NOW + timedelta(seconds=gap_seconds)])
    class GapClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(moments, NOW + timedelta(seconds=gap_seconds))
    monkeypatch.setattr("main.datetime", GapClock)
    calls = 0
    def scan(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _scan_result()
        state = ShadowSessionState.load()
        assert state.counters["data_quality"] == "PARTIAL_DATA"
        assert state.counters["partial_data_reasons"] == ["ITERATION_GAP"]
        assert state.counters["extreme_data_gap_count"] == extreme
        assert state.counters["normal_data_gap_count"] == 1 - extreme
        assert state.counters["open"] == 1
        return _scan_result(setups=[_setup(ticker="QQQ")], bars=VWAP_LOST_BARS)
    with patch("main._run_shadow_scan_once", side_effect=scan), patch(
        "trades.shadow_session.now_eastern", return_value=NOW,
    ):
        result = cli_runner.invoke(app, ["forward-test-session", "--synthetic",
                                        "--max-iterations", "2", "--interval-seconds", str(interval_seconds)])
    assert result.exit_code == 0, result.output
    trades = TradeTracker(storage_path="logs/shadow_trades.jsonl").list_all_trades()
    assert len(trades) == 1
    assert trades[0].status == TradeStatus.CLOSED.value
    assert "PARTIAL_SESSION" in result.output
    assert ShadowSessionState.load().counters["max_iteration_gap_seconds"] == gap_seconds


def test_open_trade_can_close_after_session_becomes_partial_from_stale_data(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    prior = tracker.add_trade(create_shadow_trade_from_setup(
        _setup(), shares=5.0, session_id="shadow-OLD", now=NOW,
    ))

    _save_recoverable_state(prior)

    result = _run_session(
        cli_runner,
        _scan_result(setups=[_setup(ticker="QQQ")], stale=True, bars=VWAP_LOST_BARS),
    )

    assert result.exit_code == 0, result.output
    state = ShadowSessionState.load(tmp_path / "state" / "shadow_session.json")
    assert state.counters["data_quality"] == "PARTIAL_DATA"
    closed = tracker.get_trade(prior.trade_id)
    assert closed.status == TradeStatus.CLOSED.value
    decisions = [
        json.loads(line) for line in
        (tmp_path / "logs" / "shadow_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert ShadowSkipReason.PARTIAL_SESSION.value in decisions[-1]["skip_reasons"]


def test_keyboard_interrupt_marks_state_interrupted(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run_session(
        cli_runner, [_scan_result(), KeyboardInterrupt()], iterations=5,
    )
    assert result.exit_code == 0

    state = ShadowSessionState.load(tmp_path / "state" / "shadow_session.json")
    assert state is not None
    assert state.status == SESSION_INTERRUPTED
    # Open trades stay open and are recorded in state.
    assert len(state.open_trade_ids) == 1
    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    assert len(tracker.list_open_trades()) == 1


# -- resume ------------------------------------------------------------------
import trades.shadow_session as shadow_module

def _save_recoverable_state(trade, status=SESSION_INTERRUPTED):
    state = ShadowSessionState(
        session_id=trade.metadata["session_id"], started_at=NOW.isoformat(),
        status=status, open_trade_ids=[trade.trade_id],
    )
    state.save()
    return state


@pytest.mark.parametrize("status", [SESSION_ENDED, SESSION_INTERRUPTED, SESSION_ERROR, SESSION_RUNNING])
def test_recovery_respects_source_status(cli_runner, tmp_path, monkeypatch, status):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path="logs/shadow_trades.jsonl")
    prior = tracker.add_trade(create_shadow_trade_from_setup(
        _setup(), shares=5.0, session_id="shadow-OLD", now=NOW,
    ))
    _save_recoverable_state(prior, status)
    result = _run_session(cli_runner, _scan_result(setups=[], bars=VWAP_LOST_BARS),
                          extra_args=["--close-open-at-session-end"])
    assert result.exit_code == 0, result.output
    state = ShadowSessionState.load()
    assert state.status == SESSION_ENDED
    assert state.open_trade_ids == []
    assert state.counters["open"] == 0
    assert not (tmp_path / SHADOW_SESSION_LOCK_FILE).exists()
    trade = tracker.get_trade(prior.trade_id)
    if status == SESSION_ENDED:
        assert trade == prior  # No adoption, exit, repricing, or metadata mutation.
        assert state.counters["unadopted_open"] == 1
    else:
        assert trade.status == TradeStatus.CLOSED.value
        assert trade.metadata["adopted_from_session"] == "shadow-OLD"


def test_ended_history_vetoes_stale_recoverable_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path="logs/shadow_trades.jsonl")
    trade = tracker.add_trade(create_shadow_trade_from_setup(
        _setup(), shares=1.0, session_id="shadow-OLD", now=NOW,
    ))
    state = _save_recoverable_state(trade)
    shadow_module.append_shadow_session_history(ShadowSessionState(
        session_id=state.session_id, started_at=state.started_at, status=SESSION_ENDED,
    ))
    assert adopt_open_shadow_trades(tracker, "shadow-NEW", state) == []
    assert tracker.get_trade(trade.trade_id) == trade


def test_unrelated_and_closed_trades_are_not_recovered(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path="logs/shadow_trades.jsonl")
    trade = tracker.add_trade(create_shadow_trade_from_setup(
        _setup(), shares=1.0, session_id="shadow-OLD", now=NOW,
    ))
    state = _save_recoverable_state(trade)
    state.open_trade_ids = []
    state.session_id = "unrelated"
    assert adopt_open_shadow_trades(tracker, "shadow-NEW", state) == []
    state.session_id = "shadow-OLD"
    state.open_trade_ids = [trade.trade_id]
    tracker.close_trade(trade.trade_id, exit_price=101.0, exit_time=NOW.isoformat())
    assert adopt_open_shadow_trades(tracker, "shadow-NEW", state) == []


def test_journal_entry_before_state_snapshot_is_recoverable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path="logs/shadow_trades.jsonl")
    trade = tracker.add_trade(create_shadow_trade_from_setup(
        _setup(), shares=1.0, session_id="shadow-OLD", now=NOW,
    ))
    state = _save_recoverable_state(trade)
    state.open_trade_ids = []  # Process stopped after journal write, before state update.
    assert [t.trade_id for t in adopt_open_shadow_trades(tracker, "shadow-NEW", state)] == [trade.trade_id]


@pytest.mark.parametrize("content", [None, "{bad json", "[]", "null", '{"session_id":"old"}',
                                    '{"session_id":"old","started_at":"x","status":"RUNNING","open_trade_ids":"trade"}'])
def test_missing_or_corrupt_state_never_adopts(cli_runner, tmp_path, monkeypatch, content):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path="logs/shadow_trades.jsonl")
    trade = tracker.add_trade(create_shadow_trade_from_setup(
        _setup(), shares=1.0, session_id="old", now=NOW,
    ))
    if content is not None:
        (tmp_path / "state").mkdir()
        (tmp_path / "state/shadow_session.json").write_text(content)
    assert ShadowSessionState.load() is None
    result = _run_session(cli_runner, _scan_result(setups=[], bars=VWAP_LOST_BARS))
    assert result.exit_code == 0, result.output
    assert tracker.get_trade(trade.trade_id) == trade


def test_partial_adoption_failure_remains_recoverable(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path="logs/shadow_trades.jsonl")
    trades = [tracker.add_trade(create_shadow_trade_from_setup(
        _setup(ticker=ticker), shares=1.0, session_id="old", now=NOW,
    )) for ticker in ("SPY", "QQQ")]
    state = _save_recoverable_state(trades[0])
    state.open_trade_ids = [t.trade_id for t in trades]
    state.save()
    original = TradeTracker.update_metadata
    calls = 0
    def fail_second(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated journal write failure")
        return original(self, *args, **kwargs)
    with patch.object(TradeTracker, "update_metadata", fail_second):
        result = _run_session(cli_runner, _scan_result(setups=[]))
    assert result.exit_code != 0
    state = ShadowSessionState.load()
    assert state.status == SESSION_ERROR
    assert set(state.open_trade_ids) == {t.trade_id for t in trades}
    assert not (tmp_path / SHADOW_SESSION_LOCK_FILE).exists()
    recovered = adopt_open_shadow_trades(tracker, "retry", state)
    assert {t.trade_id for t in recovered} == {t.trade_id for t in trades}


def test_recovery_intent_save_failure_does_not_reassign_journal(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path="logs/shadow_trades.jsonl")
    trade = tracker.add_trade(create_shadow_trade_from_setup(
        _setup(), shares=1.0, session_id="old", now=NOW,
    ))
    _save_recoverable_state(trade)
    with patch.object(ShadowSessionState, "save", side_effect=OSError("simulated state write failure")):
        result = _run_session(cli_runner, _scan_result(setups=[]))
    assert result.exit_code != 0
    assert ShadowSessionState.load().session_id == "old"
    assert tracker.get_trade(trade.trade_id) == trade
    assert not (tmp_path / SHADOW_SESSION_LOCK_FILE).exists()


@pytest.mark.parametrize("price", [float("nan"), float("inf"), 0, -1, None, "bad"])
def test_invalid_session_end_price_never_fabricates_exit(cli_runner, tmp_path, monkeypatch, price):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path="logs/shadow_trades.jsonl")
    trade = tracker.add_trade(create_shadow_trade_from_setup(
        _setup(), shares=1.0, session_id="old", now=NOW,
    ))
    _save_recoverable_state(trade)
    bars = HOLD_BARS.copy()
    bars["close"] = price
    with patch("trades.trade_monitor.TradeMonitor.monitor_open_trades", return_value=[]):
        result = _run_session(cli_runner, _scan_result(setups=[], bars=bars),
                              extra_args=["--close-open-at-session-end"])
    assert result.exit_code == 0, result.output
    assert ShadowSessionState.load().status == SESSION_ERROR
    assert tracker.get_trade(trade.trade_id).status == TradeStatus.OPEN.value
    assert tracker.get_trade(trade.trade_id).exit_price is None


def test_cleanup_exception_persists_actual_journal_state(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch.object(TradeTracker, "close_trade", side_effect=OSError("simulated close failure")):
        result = _run_session(cli_runner, _scan_result(), extra_args=["--close-open-at-session-end"])
    assert result.exit_code != 0
    state = ShadowSessionState.load()
    assert state.status == SESSION_ERROR
    journal = TradeTracker(storage_path="logs/shadow_trades.jsonl")
    assert state.open_trade_ids == [t.trade_id for t in journal.list_open_trades()]
    assert state.counters["open"] == 1
    assert "SESSION_LIFECYCLE_ERROR" in state.counters["partial_data_reasons"]
    assert not (tmp_path / SHADOW_SESSION_LOCK_FILE).exists()


def test_lock_write_failure_removes_its_partial_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("trades.shadow_session.json.dump", side_effect=OSError("simulated write failure")):
        with pytest.raises(OSError):
            shadow_module.acquire_shadow_session_lock("failed")
    assert not (tmp_path / SHADOW_SESSION_LOCK_FILE).exists()


def test_lock_timestamp_offsets_compare_same_instant():
    assert shadow_module._parse_lock_timestamp("2026-09-07T10:00:00-04:00") == shadow_module._parse_lock_timestamp("2026-09-07T14:00:00+00:00")


def test_concurrent_ended_lock_reclamation_has_only_one_winner(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    monkeypatch.chdir(tmp_path)
    ShadowSessionState(session_id="old", started_at=NOW.isoformat(), status=SESSION_ENDED).save()
    (tmp_path / SHADOW_SESSION_LOCK_FILE).write_text(json.dumps({
        "session_id": "old", "started_at": NOW.isoformat(),
    }))
    barrier = Barrier(2)
    def acquire(session_id):
        barrier.wait(timeout=5)
        return shadow_module.acquire_shadow_session_lock(session_id, now=NOW)[0]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(acquire, ["one", "two"]))
    assert sum(results) == 1
    winner = "one" if results[0] else "two"
    assert shadow_module.load_shadow_session_lock()["session_id"] == winner
    assert shadow_module.release_shadow_session_lock(winner)
    # A retained guard is not a stale session lock and does not block reuse.
    assert shadow_module.acquire_shadow_session_lock("three", now=NOW)[0]


def test_session_end_closes_recovered_trades_and_persists_zero(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path="logs/shadow_trades.jsonl")
    prior = tracker.add_trade(create_shadow_trade_from_setup(
        _setup(), shares=1.0, session_id="shadow-OLD", now=NOW,
    ))
    _save_recoverable_state(prior)
    result = _run_session(cli_runner, _scan_result(setups=[]),
                          extra_args=["--close-open-at-session-end"])
    assert result.exit_code == 0, result.output
    assert tracker.get_trade(prior.trade_id).metadata["exit_signal"] == "SESSION_END"
    assert tracker.list_open_trades() == []
    state = ShadowSessionState.load()
    assert state.open_trade_ids == []
    assert state.counters["open"] == 0
    history = json.loads((tmp_path / "logs/shadow_sessions.jsonl").read_text().splitlines()[-1])
    assert history["open_trade_ids"] == []
    assert history["counters"]["open"] == 0


def test_missing_close_price_keeps_session_recoverable(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path="logs/shadow_trades.jsonl")
    prior = tracker.add_trade(create_shadow_trade_from_setup(
        _setup(ticker="QQQ"), shares=1.0, session_id="shadow-OLD", now=NOW,
    ))
    _save_recoverable_state(prior)
    result = _run_session(cli_runner, _scan_result(setups=[]),
                          extra_args=["--close-open-at-session-end"])
    assert result.exit_code == 0, result.output
    state = ShadowSessionState.load()
    assert state.status == SESSION_ERROR
    assert state.open_trade_ids == [prior.trade_id]
    assert state.counters["open"] == 1
    assert "SESSION_END_CLOSE_INCOMPLETE" in result.output
    assert tracker.get_trade(prior.trade_id).status == TradeStatus.OPEN.value
    assert tracker.get_trade(prior.trade_id).exit_price is None
    assert not (tmp_path / SHADOW_SESSION_LOCK_FILE).exists()


@pytest.mark.parametrize("failure_point", ["append_shadow_session_history", "adopt_open_shadow_trades"])
def test_lock_released_even_if_setup_or_reporting_fails(cli_runner, tmp_path, monkeypatch, failure_point):
    monkeypatch.chdir(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("simulated persistence failure")
    monkeypatch.setattr(shadow_module, failure_point, fail)
    result = _run_session(cli_runner, _scan_result(setups=[]))
    assert result.exit_code != 0
    assert not (tmp_path / SHADOW_SESSION_LOCK_FILE).exists()


def test_fresh_lock_with_unrelated_ended_state_is_not_stolen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ShadowSessionState(session_id="old", started_at=NOW.isoformat(), status=SESSION_ENDED).save()
    acquired, _, _ = shadow_module.acquire_shadow_session_lock("new", now=NOW)
    assert acquired
    acquired, _, _ = shadow_module.acquire_shadow_session_lock("competitor", now=NOW)
    assert not acquired
    assert shadow_module.load_shadow_session_lock()["session_id"] == "new"

def test_resume_adopts_prior_session_open_trades(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    prior = create_shadow_trade_from_setup(
        _setup(), shares=5.0, session_id="shadow-OLD", now=NOW,
    )
    tracker.add_trade(prior)

    _save_recoverable_state(prior)

    result = _run_session(cli_runner, _scan_result(setups=[]))
    assert result.exit_code == 0
    assert "Adopted open shadow trade" in result.output

    adopted = tracker.get_trade(prior.trade_id)
    assert adopted.status == TradeStatus.OPEN.value          # never force-closed
    assert adopted.metadata["adopted_from_session"] == "shadow-OLD"


def test_no_resume_skips_prior_open_trades(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    prior = create_shadow_trade_from_setup(
        _setup(), shares=5.0, session_id="shadow-OLD", now=NOW,
    )
    tracker.add_trade(prior)

    # VWAP-lost bars would close the prior trade if it were monitored.
    result = _run_session(
        cli_runner,
        _scan_result(setups=[], bars=VWAP_LOST_BARS),
        extra_args=["--no-resume"],
    )
    assert result.exit_code == 0
    assert "will NOT be monitored" in result.output

    unchanged = tracker.get_trade(prior.trade_id)
    assert unchanged.status == TradeStatus.OPEN.value
    assert "adopted_from_session" not in unchanged.metadata


def test_adopt_open_shadow_trades_helper(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path=tmp_path / "shadow_trades.jsonl")
    old = tracker.add_trade(create_shadow_trade_from_setup(
        _setup(), shares=1.0, session_id="shadow-OLD", now=NOW,
    ))
    current = tracker.add_trade(create_shadow_trade_from_setup(
        _setup(ticker="QQQ"), shares=1.0, session_id="shadow-NEW", now=NOW,
    ))

    _save_recoverable_state(old)
    adopted = adopt_open_shadow_trades(tracker, "shadow-NEW")
    assert [t.trade_id for t in adopted] == [old.trade_id]
    assert tracker.get_trade(old.trade_id).metadata["adopted_from_session"] == "shadow-OLD"
    assert "adopted_from_session" not in tracker.get_trade(current.trade_id).metadata


# -- failure handling ----------------------------------------------------------

def test_consecutive_data_failures_end_session(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    failures = [RuntimeError(f"data outage {i}") for i in range(10)]
    result = _run_session(cli_runner, failures, iterations=10,
                          extra_args=["--max-data-failures", "3"])
    assert result.exit_code == 0
    assert "ending session cleanly" in result.output
    assert "Data failures: 3" in result.output

    state = ShadowSessionState.load(tmp_path / "state" / "shadow_session.json")
    assert state.status == SESSION_ENDED
    assert state.counters["data_failures"] == 3


def test_data_failure_counter_resets_on_success(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run_session(cli_runner, [
        RuntimeError("outage 1"),
        _scan_result(setups=[]),
        RuntimeError("outage 2"),
    ], iterations=3, extra_args=["--max-data-failures", "2"])
    assert result.exit_code == 0
    # Two non-consecutive failures never hit the threshold of 2.
    assert "ending session cleanly" not in result.output
    assert "Data failures: 2" in result.output


def test_stale_data_blocks_entries_but_processes_tagged_exits(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    prior = tracker.add_trade(create_shadow_trade_from_setup(
        _setup(), shares=5.0, session_id="shadow-OLD", now=NOW,
    ))

    # Stale iteration: a fresh setup is offered AND the open trade loses VWAP.
    _save_recoverable_state(prior)

    result = _run_session(
        cli_runner,
        _scan_result(setups=[_setup(ticker="QQQ")], stale=True, bars=VWAP_LOST_BARS),
    )
    assert result.exit_code == 0

    # No new entry (stale blocks entries)...
    decisions = [
        json.loads(line) for line in
        (tmp_path / "logs" / "shadow_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert decisions[0]["decision"] == DECISION_SKIPPED
    assert ShadowSkipReason.STALE_DATA.value in decisions[0]["skip_reasons"]

    # ...but the adopted open trade's exit was processed and tagged.
    closed = tracker.get_trade(prior.trade_id)
    assert closed.status == TradeStatus.CLOSED.value
    assert closed.metadata["exit_signal"] == "EXIT_VWAP_LOST"
    assert closed.metadata["stale_exit"] is True


# -- Discord session summary -----------------------------------------------------

def test_discord_session_summary_is_clearly_labelled(cli_runner, tmp_path, monkeypatch):
    import alerts.discord_alerts as discord_module

    sent = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(discord_module, "is_discord_configured", lambda: True)
    monkeypatch.setattr(
        discord_module, "send_discord_message",
        lambda content: sent.append(content) or True,
    )

    result = _run_session(cli_runner, _scan_result(setups=[]))
    assert result.exit_code == 0
    summary = sent[-1]
    assert "SHADOW SESSION SUMMARY" in summary
    assert "SIMULATION ONLY" in summary
    assert "No real order placed." in summary
    assert "Entered: 0" in summary


def test_interrupted_session_sends_discord_summary(cli_runner, tmp_path, monkeypatch):
    import alerts.discord_alerts as discord_module

    sent = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(discord_module, "is_discord_configured", lambda: True)
    monkeypatch.setattr(
        discord_module, "send_discord_message",
        lambda content: sent.append(content) or True,
    )

    result = _run_session(
        cli_runner, [_scan_result(setups=[]), KeyboardInterrupt()], iterations=5,
    )
    assert result.exit_code == 0
    assert any("SHADOW SESSION SUMMARY" in m and "INTERRUPTED" in m for m in sent)


def test_no_secrets_in_session_state_and_output(cli_runner, tmp_path, monkeypatch):
    fake_key = "FAKE_KEY_SHOULD_NEVER_APPEAR"
    monkeypatch.setenv("ALPACA_API_KEY", fake_key)
    monkeypatch.setenv("ALPACA_SECRET_KEY", fake_key)
    monkeypatch.chdir(tmp_path)

    result = _run_session(cli_runner, _scan_result())
    assert result.exit_code == 0
    assert fake_key not in result.output

    state_content = (tmp_path / "state" / "shadow_session.json").read_text(encoding="utf-8")
    assert fake_key not in state_content
    assert "webhook" not in state_content.lower()


# ===========================================================================
# Phase 4: analytics and manual-vs-shadow comparison
# ===========================================================================

from trades.shadow_session import (
    average_time_in_trade_seconds,
    shadow_after_15_summary,
    build_orb_no_progress_snapshot,
    record_orb_no_progress_diagnostics,
    shadow_entry_time_breakdown,
    shadow_exit_reason_breakdown,
    shadow_orb_no_progress_diagnostics,
    shadow_stop_hit_diagnostics,
    summarize_shadow_decision_records,
)


def _closed_shadow_trade(ticker="SPY", pnl=10.0, exit_signal="EXIT_VWAP_LOST",
                         minutes_in_trade=30, entry_time=None,
                         setup_type="vwap_reclaim_long", session_id=SESSION_ID) -> ManualTrade:
    entry = entry_time or NOW
    exit_time = entry + timedelta(minutes=minutes_in_trade)
    return ManualTrade(
        ticker=ticker,
        setup_type=setup_type,
        entry_price=100.0,
        stop_loss=98.0,
        target_1=104.0,
        target_2=106.0,
        shares=10.0,
        entry_time=entry.isoformat(),
        status=TradeStatus.CLOSED.value,
        exit_price=100.0 + pnl / 10.0,
        exit_time=exit_time.isoformat(),
        realized_pnl=pnl,
        test_type=SHADOW_TEST_TYPE,
        metadata={"exit_signal": exit_signal, "session_id": session_id},
    )


def test_summarize_shadow_decision_records_counts_and_reasons():
    records = [
        {"timestamp": NOW.isoformat(), "decision": "ENTERED", "skip_reasons": []},
        {"timestamp": NOW.isoformat(), "decision": "SKIPPED",
         "skip_reasons": ["LOW_CONFIDENCE", "STALE_DATA"]},
        {"timestamp": NOW.isoformat(), "decision": "SKIPPED",
         "skip_reasons": ["LOW_CONFIDENCE"]},
        {"timestamp": (NOW - timedelta(days=1)).isoformat(), "decision": "SKIPPED",
         "skip_reasons": ["STALE_DATA"]},
    ]
    summary = summarize_shadow_decision_records(records)
    assert summary["entered"] == 1
    assert summary["skipped"] == 3
    assert summary["skip_reasons"] == {"LOW_CONFIDENCE": 2, "STALE_DATA": 2}

    today_only = summarize_shadow_decision_records(records, NOW.date())
    assert today_only["skipped"] == 2
    assert today_only["skip_reasons"] == {"LOW_CONFIDENCE": 2, "STALE_DATA": 1}


def test_decision_log_summarize_handles_missing_file(tmp_path):
    summary = ShadowDecisionLog(tmp_path / "missing.jsonl").summarize()
    assert summary == {"entered": 0, "skipped": 0, "skip_reasons": {}}


def test_shadow_exit_reason_breakdown_groups_closed_trades():
    trades = [
        _closed_shadow_trade(pnl=20.0, exit_signal="TARGET_1_HIT"),
        _closed_shadow_trade(pnl=-10.0, exit_signal="EXIT_VWAP_LOST"),
        _closed_shadow_trade(pnl=-5.0, exit_signal="EXIT_VWAP_LOST"),
        _open_shadow_trade(),  # OPEN — must be ignored
    ]
    rows = shadow_exit_reason_breakdown(trades)
    assert [row["exit_reason"] for row in rows] == ["TARGET_1_HIT", "EXIT_VWAP_LOST"]
    target = rows[0]
    assert target["trades"] == 1 and target["win_rate"] == 1.0
    vwap = rows[1]
    assert vwap["trades"] == 2
    assert vwap["total_pnl"] == pytest.approx(-15.0)
    assert vwap["win_rate"] == 0.0


def test_shadow_entry_time_breakdown_groups_closed_trades():
    trades = [
        _closed_shadow_trade(pnl=10.0, entry_time=NOW.replace(hour=9, minute=45)),
        _closed_shadow_trade(pnl=-5.0, entry_time=NOW.replace(hour=10, minute=15)),
        _closed_shadow_trade(pnl=7.0, entry_time=NOW.replace(hour=11, minute=30)),
        _closed_shadow_trade(pnl=-2.0, entry_time=NOW.replace(hour=12, minute=30)),
        _closed_shadow_trade(pnl=3.0, entry_time=NOW.replace(hour=13, minute=15)),
        _closed_shadow_trade(pnl=-4.0, entry_time=NOW.replace(hour=14, minute=59)),
        _closed_shadow_trade(pnl=-6.0, entry_time=NOW.replace(hour=15, minute=0)),
        _closed_shadow_trade(pnl=8.0, entry_time=NOW.replace(hour=15, minute=30)),
        _open_shadow_trade(),
    ]

    rows = {row["entry_time_bucket"]: row for row in shadow_entry_time_breakdown(trades)}

    assert rows["before 10:00"]["trades"] == 1
    assert rows["10:00-11:00"]["total_pnl"] == pytest.approx(-5.0)
    assert rows["11:00-12:00"]["win_rate"] == 1.0
    assert rows["12:00-13:00"]["avg_pnl"] == pytest.approx(-2.0)
    assert rows["13:00-14:00"]["total_pnl"] == pytest.approx(3.0)
    assert rows["14:00-15:00"]["trades"] == 1
    assert rows["after 15:00"]["trades"] == 2
    assert rows["after 15:00"]["total_pnl"] == pytest.approx(2.0)
    assert rows["after 15:00"]["win_rate"] == pytest.approx(0.5)
    assert rows["after 15:00"]["profit_factor"] == pytest.approx(8.0 / 6.0)


def test_shadow_entry_time_breakdown_empty_buckets_do_not_crash():
    rows = shadow_entry_time_breakdown([])

    assert [row["entry_time_bucket"] for row in rows] == [
        "before 10:00",
        "10:00-11:00",
        "11:00-12:00",
        "12:00-13:00",
        "13:00-14:00",
        "14:00-15:00",
        "after 15:00",
    ]
    assert all(row["trades"] == 0 for row in rows)
    assert shadow_after_15_summary([])["total_pnl"] == 0.0


def test_shadow_stop_hit_diagnostics_counts_and_compares_trades():
    trades = [
        _closed_shadow_trade(
            ticker="TSLA", pnl=-20.0, exit_signal="STOP_HIT", minutes_in_trade=90,
            entry_time=NOW.replace(hour=15, minute=15),
            setup_type="vwap_reclaim_long",
        ),
        _closed_shadow_trade(
            ticker="NVDA", pnl=-10.0, exit_signal="STOP_HIT", minutes_in_trade=30,
            entry_time=NOW.replace(hour=10, minute=15),
            setup_type="opening_range_breakout_long",
        ),
        _closed_shadow_trade(ticker="SPY", pnl=30.0, exit_signal="TARGET_1_HIT"),
        _open_shadow_trade(),
    ]

    diagnostics = shadow_stop_hit_diagnostics(trades)

    assert diagnostics["stop_hit"]["trades"] == 2
    assert diagnostics["stop_hit"]["total_pnl"] == pytest.approx(-30.0)
    assert diagnostics["stop_hit"]["avg_pnl"] == pytest.approx(-15.0)
    assert diagnostics["stop_hit"]["avg_time_seconds"] == pytest.approx(60 * 60)
    assert diagnostics["non_stop_hit"]["trades"] == 1
    assert diagnostics["non_stop_hit"]["total_pnl"] == pytest.approx(30.0)
    assert {row["setup_type"]: row["trades"] for row in diagnostics["by_setup_type"]} == {
        "vwap_reclaim_long": 1,
        "opening_range_breakout_long": 1,
    }
    assert {row["ticker"]: row["trades"] for row in diagnostics["by_ticker"]} == {
        "TSLA": 1,
        "NVDA": 1,
    }
    assert {row["entry_time_bucket"]: row["trades"] for row in diagnostics["by_entry_time"]} == {
        "10:00-11:00": 1,
        "after 15:00": 1,
    }


def test_shadow_stop_hit_diagnostics_worst_losses_sort_ascending():
    trades = [
        _closed_shadow_trade(ticker="A", pnl=-5.0, exit_signal="STOP_HIT"),
        _closed_shadow_trade(ticker="B", pnl=-25.0, exit_signal="STOP_HIT"),
        _closed_shadow_trade(ticker="C", pnl=-10.0, exit_signal="STOP_HIT"),
        _closed_shadow_trade(ticker="D", pnl=20.0, exit_signal="TARGET_1_HIT"),
    ]

    losses = shadow_stop_hit_diagnostics(trades)["largest_losses"]

    assert [loss["ticker"] for loss in losses] == ["B", "C", "A"]
    assert [loss["realized_pnl"] for loss in losses] == [-25.0, -10.0, -5.0]


def test_shadow_stop_hit_diagnostics_empty_case_does_not_crash():
    diagnostics = shadow_stop_hit_diagnostics([
        _closed_shadow_trade(ticker="SPY", pnl=20.0, exit_signal="TARGET_1_HIT"),
    ])

    assert diagnostics["stop_hit"]["trades"] == 0
    assert diagnostics["stop_hit"]["total_pnl"] == 0.0
    assert diagnostics["stop_hit"]["avg_time_seconds"] is None
    assert diagnostics["by_setup_type"] == []
    assert diagnostics["by_ticker"] == []
    assert diagnostics["by_entry_time"] == []
    assert diagnostics["largest_losses"] == []


def _diagnostic_bars(start: datetime) -> pd.DataFrame:
    index = pd.to_datetime([
        start,
        start + timedelta(minutes=15),
        start + timedelta(minutes=30),
        start + timedelta(minutes=60),
    ])
    return pd.DataFrame({
        "open": [100.0, 100.4, 100.6, 100.8],
        "high": [100.2, 100.8, 101.2, 101.4],
        "low": [99.8, 99.6, 99.4, 99.7],
        "close": [100.1, 99.9, 100.5, 100.7],
        "volume": [1000, 1200, 1100, 1300],
        "session_vwap": [100.0, 100.1, 100.8, 100.9],
        "session_opening_range_high": [100.3, 100.3, 100.3, 100.3],
    }, index=index)


def test_orb_no_progress_snapshot_flags_no_progress():
    trade = _open_shadow_trade(setup_type="opening_range_breakout_long")
    now = NOW + timedelta(minutes=30)
    result = SimpleNamespace(current_price=100.5)

    snapshot = build_orb_no_progress_snapshot(
        trade, result, now, 30, bars=_diagnostic_bars(NOW),
    )

    assert snapshot["current_price"] == 100.5
    assert snapshot["unrealized_pnl"] == pytest.approx(2.5)
    assert snapshot["r_multiple"] == pytest.approx(0.25)
    assert snapshot["progress_to_target_1"] == pytest.approx(0.125)
    assert snapshot["distance_remaining_to_stop_pct"] == pytest.approx(1.25)
    assert snapshot["price_above_entry"] is True
    assert snapshot["made_meaningful_progress"] is False
    assert snapshot["no_progress"] is True
    assert snapshot["mfe_reached_025r"] is True
    assert snapshot["closed_back_below_entry"] is True
    assert snapshot["vwap_loss_seen"] is True
    assert snapshot["opening_range_reentry_seen"] is True


def test_orb_no_progress_snapshot_flags_meaningful_progress():
    trade = _open_shadow_trade(setup_type="opening_range_breakout_long")
    snapshot = build_orb_no_progress_snapshot(
        trade,
        SimpleNamespace(current_price=102.0),
        NOW + timedelta(minutes=60),
        60,
    )

    assert snapshot["progress_to_target_1"] == pytest.approx(0.5)
    assert snapshot["made_meaningful_progress"] is True
    assert snapshot["no_progress"] is False


def test_orb_no_progress_diagnostics_survive_persistence_reload(tmp_path):
    tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    trade = _open_shadow_trade(setup_type="opening_range_breakout_long")
    trade.entry_time = (NOW - timedelta(minutes=31)).isoformat()
    tracker.add_trade(trade)

    updated = record_orb_no_progress_diagnostics(
        tracker,
        trade,
        SimpleNamespace(current_price=100.5),
        NOW,
        bars=_diagnostic_bars(NOW - timedelta(minutes=31)),
    )

    assert updated is not None
    reloaded = tracker.get_trade(trade.trade_id)
    diagnostics = reloaded.metadata["orb_no_progress_diagnostics"]
    assert diagnostics["version"] == 1
    assert "30" in diagnostics["snapshots"]
    assert diagnostics["snapshots"]["30"]["no_progress"] is True
    assert "60" not in diagnostics["snapshots"]


def test_orb_no_progress_candidate_impact_math():
    stop_trade = _closed_shadow_trade(
        ticker="TSLA", pnl=-20.0, exit_signal="STOP_HIT",
        setup_type="opening_range_breakout_long",
    )
    stop_trade.metadata["orb_no_progress_diagnostics"] = {
        "snapshots": {
            "30": {"no_progress": True, "unrealized_pnl": -5.0},
            "60": {"no_progress": True, "unrealized_pnl": -9.0},
        }
    }
    winner_trade = _closed_shadow_trade(
        ticker="AVGO", pnl=30.0, exit_signal="TARGET_1_HIT",
        setup_type="opening_range_breakout_long",
    )
    winner_trade.metadata["orb_no_progress_diagnostics"] = {
        "snapshots": {
            "30": {"no_progress": True, "unrealized_pnl": 2.0},
            "60": {"no_progress": False, "unrealized_pnl": 15.0},
        }
    }
    no_snapshot = _closed_shadow_trade(
        ticker="QQQ", pnl=-4.0, exit_signal="STOP_HIT",
        setup_type="opening_range_breakout_long",
    )

    diagnostics = shadow_orb_no_progress_diagnostics([stop_trade, winner_trade, no_snapshot])

    thirty = diagnostics["milestones"]["30"]
    assert thirty["snapshot_count"] == 2
    assert thirty["no_progress_count"] == 2
    assert thirty["no_progress_actual_pnl"] == pytest.approx(10.0)
    assert thirty["no_progress_hypothetical_pnl"] == pytest.approx(-3.0)
    assert thirty["candidate_pnl_delta"] == pytest.approx(-13.0)
    assert thirty["eventual_stop_hit_flagged"] == 1
    assert thirty["eventual_target_1_hit_flagged"] == 1

    sixty = diagnostics["milestones"]["60"]
    assert sixty["snapshot_count"] == 2
    assert sixty["no_progress_count"] == 1
    assert sixty["no_progress_actual_pnl"] == pytest.approx(-20.0)
    assert sixty["no_progress_hypothetical_pnl"] == pytest.approx(-9.0)
    assert sixty["candidate_pnl_delta"] == pytest.approx(11.0)


def test_orb_no_progress_analytics_handles_no_diagnostics():
    diagnostics = shadow_orb_no_progress_diagnostics([
        _closed_shadow_trade(setup_type="opening_range_breakout_long", pnl=-10.0),
    ])

    assert diagnostics["orb_closed_trades"] == 1
    assert diagnostics["milestones"]["30"]["snapshot_count"] == 0
    assert diagnostics["milestones"]["30"]["no_progress_count"] == 0
    assert diagnostics["milestones"]["60"]["candidate_pnl_delta"] == 0.0


def test_average_time_in_trade_seconds():
    trades = [
        _closed_shadow_trade(minutes_in_trade=30),
        _closed_shadow_trade(minutes_in_trade=60),
        _open_shadow_trade(),  # OPEN — ignored
    ]
    assert average_time_in_trade_seconds(trades) == pytest.approx(45 * 60)
    assert average_time_in_trade_seconds([]) is None


def _seed_phase4_journals(tmp_path):
    """Create manual + shadow journals and a decision log under tmp cwd."""
    manual_tracker = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl")
    manual_tracker.add_trade(ManualTrade(
        ticker="SPY", setup_type="vwap_reclaim_long",
        entry_price=100.0, stop_loss=98.0, target_1=104.0, target_2=106.0,
        shares=10.0, entry_time=NOW.isoformat(),
        status=TradeStatus.CLOSED.value, exit_price=102.0,
        exit_time=(NOW + timedelta(minutes=45)).isoformat(), realized_pnl=20.0,
    ))
    shadow_tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    shadow_tracker.add_trade(_closed_shadow_trade(pnl=-10.0))
    shadow_tracker.add_trade(_closed_shadow_trade(ticker="QQQ", pnl=30.0,
                                                  exit_signal="TARGET_1_HIT"))
    log = ShadowDecisionLog(tmp_path / "logs" / "shadow_decisions.jsonl")
    log.record(SESSION_ID, _setup(), _evaluate(), trade_id="t-1", timestamp=NOW)
    log.record(SESSION_ID, _setup(confidence=0.5), _evaluate(_setup(confidence=0.5)),
               timestamp=NOW)


def test_trade_analytics_include_shadow_prints_all_sections(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_phase4_journals(tmp_path)

    result = cli_runner.invoke(app, ["trade-analytics", "--include-shadow"])
    assert result.exit_code == 0, result.output

    out = result.output
    assert "=== Manual Forward Test ===" in out
    assert "=== Shadow Forward Test ===" in out
    assert "=== Manual vs Shadow ===" in out
    assert "=== Shadow Decisions ===" in out
    assert "SIMULATION ONLY" in out
    # Shadow side: 2 closed, 1 win, +20 sim P&L
    assert "Total shadow trades: 2" in out
    assert "Closed / Open:       2 / 0" in out
    assert "Total sim P&L:       +20.00" in out
    # Comparison shows both sides and the followed-signal line
    assert "Closed trades" in out
    assert "followed first exit signal" in out
    assert "EXIT_VWAP_LOST=1" in out and "TARGET_1_HIT=1" in out
    assert "-- By Entry Time --" in out
    assert "10:00-11:00" in out
    assert "trades=2" in out
    assert "-- Late-Day Summary --" in out
    assert "After 15:00 trades:   0" in out
    assert "-- Stop-Hit Diagnostics --" in out
    assert "STOP_HIT:     count=0" in out
    assert "Largest stop-hit losses:" in out
    assert "-- ORB No-Progress Diagnostics --" in out
    assert "30-min snapshots: 0" in out
    assert "60-min snapshots: 0" in out
    # Decisions
    assert "ENTERED: 1" in out
    assert "SKIPPED: 1" in out
    assert "LOW_CONFIDENCE: 1" in out


def test_trade_analytics_default_unchanged_and_shadow_isolated(cli_runner, tmp_path, monkeypatch):
    """Without --include-shadow: no shadow sections, and shadow trades never
    leak into manual stats."""
    monkeypatch.chdir(tmp_path)
    _seed_phase4_journals(tmp_path)
    manual_before = (tmp_path / "logs" / "manual_trades.jsonl").read_bytes()

    result = cli_runner.invoke(app, ["trade-analytics"])
    assert result.exit_code == 0, result.output
    assert "Shadow" not in result.output
    # Manual stats reflect ONLY the single manual trade (+20.00), not shadow P&L.
    assert "Total trades:       1" in result.output
    assert "Total realized P&L: +20.00" in result.output
    assert (tmp_path / "logs" / "manual_trades.jsonl").read_bytes() == manual_before


def test_trade_analytics_include_shadow_handles_missing_logs(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = cli_runner.invoke(app, ["trade-analytics", "--include-shadow"])
    assert result.exit_code == 0, result.output
    assert "=== Shadow Forward Test ===" in result.output
    assert "No closed shadow trades yet." in result.output
    assert "ENTERED: 0" in result.output


def test_trade_analytics_include_shadow_shows_state_only_session(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    state_path = tmp_path / "state" / "shadow_session.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "session_id": "shadow-20260615-114419",
        "started_at": NOW.isoformat(),
        "status": SESSION_ENDED,
        "config": {"provider": "alpaca", "timeframe": "5Min"},
        "open_trade_ids": [],
        "last_iteration_at": NOW.isoformat(),
        "counters": {
            "iterations_total": 12,
            "no_ranked_setup_iterations": 12,
            "ranked_setup_iterations": 0,
            "evaluated_setups": 0,
            "entered": 0,
            "skipped": 0,
            "closed": 0,
            "open": 0,
        },
    }), encoding="utf-8")

    result = cli_runner.invoke(app, ["trade-analytics", "--include-shadow"])

    assert result.exit_code == 0, result.output
    assert "Latest shadow session: shadow-20260615-114419 [ENDED]" in result.output
    assert "Iterations:            12" in result.output
    assert "No-ranked-setup iterations: 12" in result.output
    assert "No real orders placed." in result.output
    assert "No closed shadow trades yet." in result.output


def test_trade_analytics_include_shadow_shows_today_session_history(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    history_path = tmp_path / "logs" / "shadow_sessions.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "timestamp": datetime.now().replace(hour=10, minute=0, second=0, microsecond=0).isoformat(),
            "session_id": "shadow-A",
            "status": SESSION_ENDED,
            "counters": {
                "iterations_total": 12,
                "evaluated_setups": 0,
                "no_ranked_setup_iterations": 12,
                "data_gap_count": 1,
                "largest_data_gap_seconds": 4200.0,
                "data_quality": "PARTIAL_DATA",
                "stale_warning_count": 3,
                "max_stale_bar_age_seconds": 1800.0,
                "first_stale_timestamp": "2026-06-25T10:05:00",
                "last_stale_timestamp": "2026-06-25T10:15:00",
                "stale_data_caused_partial": True,
            },
        },
        {
            "timestamp": datetime.now().replace(hour=11, minute=0, second=0, microsecond=0).isoformat(),
            "session_id": "shadow-B",
            "status": SESSION_INTERRUPTED,
            "counters": {
                "iterations_total": 1,
                "evaluated_setups": 2,
                "no_ranked_setup_iterations": 0,
            },
        },
    ]
    history_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    result = cli_runner.invoke(app, ["trade-analytics", "--include-shadow"])

    assert result.exit_code == 0, result.output
    assert "Today's shadow sessions: 2" in result.output
    assert "Today's iterations:      13" in result.output
    assert "Today's evaluated setups: 2" in result.output
    assert "Today's data quality:  PARTIAL_DATA" in result.output
    assert "Today's data gaps:     1" in result.output
    assert "Today's largest gap:   70.0 min" in result.output
    assert "Today's stale warnings: 3" in result.output
    assert "Today's max stale age:  30.0 min" in result.output
    assert "PARTIAL_DATA sessions=1 stale_warnings=3 max_age=30.0 min stale_caused_partial=yes" in result.output
    assert "invalid for strategy proof and excluded from clean analytics" in result.output


def _write_shadow_session_history(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def _seed_clean_shadow_filter_sample(tmp_path):
    history_path = tmp_path / "logs" / "shadow_sessions.jsonl"
    _write_shadow_session_history(history_path, [
        {
            "timestamp": datetime.now().replace(hour=10, minute=0, second=0, microsecond=0).isoformat(),
            "session_id": "shadow-CLEAN",
            "status": SESSION_ENDED,
            "config": {"watchlist_name": "momentum_universe"},
            "counters": {
                "iterations_total": 40,
                "evaluated_setups": 12,
                "entered": 2,
                "closed": 2,
                "open": 0,
                "realized_pnl": 30.0,
                "data_quality": "CLEAN",
                "data_gap_count": 0,
            },
        },
        {
            "timestamp": datetime.now().replace(hour=11, minute=0, second=0, microsecond=0).isoformat(),
            "session_id": "shadow-PARTIAL",
            "status": SESSION_ENDED,
            "config": {"watchlist_name": "momentum_universe"},
            "counters": {
                "iterations_total": 5,
                "evaluated_setups": 1,
                "entered": 1,
                "closed": 1,
                "open": 0,
                "realized_pnl": -50.0,
                "data_quality": "PARTIAL_DATA",
                "data_gap_count": 1,
            },
        },
        {
            "timestamp": datetime.now().replace(hour=12, minute=0, second=0, microsecond=0).isoformat(),
            "session_id": "shadow-INTERRUPTED",
            "status": SESSION_INTERRUPTED,
            "config": {"watchlist_name": "momentum_universe"},
            "counters": {
                "iterations_total": 3,
                "evaluated_setups": 1,
                "entered": 1,
                "closed": 1,
                "open": 0,
                "realized_pnl": -25.0,
                "data_quality": "CLEAN",
                "data_gap_count": 0,
            },
        },
    ])
    shadow_tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    shadow_tracker.add_trade(_closed_shadow_trade(
        ticker="SPY", pnl=20.0, exit_signal="TARGET_1_HIT", session_id="shadow-CLEAN",
        setup_type="opening_range_breakout_long",
    ))
    shadow_tracker.add_trade(_closed_shadow_trade(
        ticker="NVDA", pnl=10.0, exit_signal="SESSION_END", session_id="shadow-CLEAN",
    ))
    shadow_tracker.add_trade(_closed_shadow_trade(
        ticker="TSLA", pnl=-50.0, exit_signal="STOP_HIT", session_id="shadow-PARTIAL",
    ))
    shadow_tracker.add_trade(_closed_shadow_trade(
        ticker="AMD", pnl=-25.0, exit_signal="STOP_HIT", session_id="shadow-INTERRUPTED",
    ))


def _seed_clean_shadow_large_loser_sample(tmp_path):
    _write_shadow_session_history(tmp_path / "logs" / "shadow_sessions.jsonl", [
        {
            "timestamp": datetime.now().replace(hour=10, minute=0, second=0, microsecond=0).isoformat(),
            "session_id": "shadow-CLEAN-LOSERS",
            "status": SESSION_ENDED,
            "config": {"watchlist_name": "momentum_universe"},
            "counters": {
                "iterations_total": 40,
                "evaluated_setups": 20,
                "entered": 6,
                "closed": 6,
                "open": 0,
                "realized_pnl": -60.0,
                "data_quality": "CLEAN",
                "data_gap_count": 0,
            },
        },
    ])
    shadow_tracker = TradeTracker(storage_path=tmp_path / "logs" / "shadow_trades.jsonl")
    shadow_tracker.add_trade(_closed_shadow_trade(
        ticker="ARM",
        pnl=-25.0,
        exit_signal="STOP_HIT",
        session_id="shadow-CLEAN-LOSERS",
        setup_type="opening_range_breakout_long",
        entry_time=NOW.replace(hour=13, minute=15),
        minutes_in_trade=70,
    ))
    shadow_tracker.add_trade(_closed_shadow_trade(
        ticker="QCOM",
        pnl=-20.0,
        exit_signal="STOP_HIT",
        session_id="shadow-CLEAN-LOSERS",
        setup_type="opening_range_breakout_long",
        entry_time=NOW.replace(hour=14, minute=30),
        minutes_in_trade=60,
    ))
    shadow_tracker.add_trade(_closed_shadow_trade(
        ticker="TSLA",
        pnl=-12.0,
        exit_signal="SESSION_END",
        session_id="shadow-CLEAN-LOSERS",
        setup_type="opening_range_breakout_long",
        entry_time=NOW.replace(hour=14, minute=10),
        minutes_in_trade=55,
    ))
    shadow_tracker.add_trade(_closed_shadow_trade(
        ticker="CRWD",
        pnl=-10.0,
        exit_signal="STOP_HIT",
        session_id="shadow-CLEAN-LOSERS",
        setup_type="high_relative_volume_continuation_long",
        entry_time=NOW.replace(hour=13, minute=40),
        minutes_in_trade=30,
    ))
    shadow_tracker.add_trade(_closed_shadow_trade(
        ticker="NVDA",
        pnl=-8.0,
        exit_signal="STOP_HIT",
        session_id="shadow-CLEAN-LOSERS",
        setup_type="high_relative_volume_continuation_long",
        entry_time=NOW.replace(hour=10, minute=30),
        minutes_in_trade=25,
    ))
    shadow_tracker.add_trade(_closed_shadow_trade(
        ticker="AAPL",
        pnl=15.0,
        exit_signal="TARGET_1_HIT",
        session_id="shadow-CLEAN-LOSERS",
        setup_type="high_relative_volume_continuation_long",
        entry_time=NOW.replace(hour=9, minute=45),
        minutes_in_trade=20,
    ))


def test_trade_analytics_default_shadow_still_includes_all_shadow_trades(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _seed_clean_shadow_filter_sample(tmp_path)

    result = cli_runner.invoke(app, ["trade-analytics", "--include-shadow"])

    assert result.exit_code == 0, result.output
    assert "=== Shadow Forward Test ===" in result.output
    assert "Total shadow trades: 4" in result.output
    assert "Total sim P&L:       -45.00" in result.output
    assert "TSLA=1" in result.output or "TSLA" in result.output
    assert "-- Hypothetical Filter Diagnostics --" not in result.output
    assert "-- Candidate Rule Tracking --" not in result.output


def test_trade_analytics_clean_shadow_only_filters_to_clean_ended_sessions(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _seed_clean_shadow_filter_sample(tmp_path)

    result = cli_runner.invoke(app, [
        "trade-analytics", "--include-shadow", "--clean-shadow-only",
    ])

    assert result.exit_code == 0, result.output
    assert "=== Clean Shadow Sample ===" in result.output
    assert "Clean sessions: 1" in result.output
    assert "Session IDs: shadow-CLEAN" in result.output
    assert "Watchlists: momentum_universe=1" in result.output
    assert "Closed trades: 2" in result.output
    assert "Open trades:             0" in result.output
    assert "Total P&L:     +30.00" in result.output
    assert "Win rate:      100.0%" in result.output
    assert "SPY" in result.output
    assert "NVDA" in result.output
    assert "TSLA" not in result.output
    assert "AMD" not in result.output
    assert "-- Large Loser Diagnostics --" in result.output
    assert "No losing clean shadow trades." in result.output
    assert "Clean shadow sample is still small" in result.output


def test_trade_analytics_clean_shadow_large_loser_diagnostics(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _seed_clean_shadow_large_loser_sample(tmp_path)

    result = cli_runner.invoke(app, [
        "trade-analytics", "--include-shadow", "--clean-shadow-only",
    ])

    assert result.exit_code == 0, result.output
    assert "-- Large Loser Diagnostics --" in result.output
    loser_block = result.output.split("Largest losing clean shadow trades:", 1)[1]
    loser_block = loser_block.split("Largest losses by ticker", 1)[0]
    assert loser_block.index("ARM") < loser_block.index("TSLA")
    assert loser_block.index("TSLA") < loser_block.index("NVDA")
    assert "STOP_HIT loss" in result.output
    assert "SESSION_END loss" in result.output
    assert "Largest losses by ticker" in result.output
    assert "Largest losses by setup type" in result.output
    assert "Largest losses by entry time bucket" in result.output
    assert "STOP_HIT losses vs SESSION_END losses" in result.output
    assert "-- Large ORB Losers --" in result.output
    assert "opening_range_breakout_long" in result.output
    assert "-- HRV Continuation Diagnostics --" in result.output
    assert "Total HRV trades: 3" in result.output
    assert "HRV P&L: -3.00" in result.output
    assert "HRV win rate: 33.3%" in result.output
    assert "HRV losses: 2" in result.output
    assert "HRV average loss: -9.00" in result.output
    assert "Worst HRV tickers" in result.output


def test_trade_analytics_clean_shadow_hypothetical_filter_diagnostics(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _seed_clean_shadow_large_loser_sample(tmp_path)

    result = cli_runner.invoke(app, [
        "trade-analytics", "--include-shadow", "--clean-shadow-only",
    ])

    assert result.exit_code == 0, result.output
    assert "-- Hypothetical Filter Diagnostics --" in result.output
    assert "Baseline: closed=6 pnl=-60.00" in result.output
    assert "Hypothetical filters are for research only" in result.output
    assert "Exclude high_relative_volume_continuation_long" in result.output
    assert "excluded=3" in result.output
    assert "delta=+3.00" in result.output
    assert "Exclude ORB entries at/after 13:00" in result.output
    assert "delta=+57.00" in result.output
    assert "Exclude ARM" in result.output
    assert "delta=+25.00" in result.output
    assert "Exclude QCOM" in result.output
    assert "delta=+20.00" in result.output
    assert "Exclude CRWD" in result.output
    assert "delta=+10.00" in result.output
    assert "Exclude ARM + QCOM + CRWD" in result.output
    assert "Exclude top 3 worst tickers by clean P&L" in result.output
    assert "hypothetical=-3.00" in result.output


def test_trade_analytics_clean_shadow_candidate_rule_tracking(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _seed_clean_shadow_large_loser_sample(tmp_path)

    result = cli_runner.invoke(app, [
        "trade-analytics", "--include-shadow", "--clean-shadow-only",
    ])

    assert result.exit_code == 0, result.output
    assert "-- Candidate Rule Tracking --" in result.output
    assert "Candidate rules are research-only" in result.output

    def _candidate_line(name: str) -> str:
        return next(
            line for line in result.output.splitlines()
            if line.strip() and line.strip().split(maxsplit=1)[0] == name
        )

    no_entries = _candidate_line("candidate_no_entries_after_13")
    assert "remaining=2" in no_entries
    assert "excluded=4" in no_entries
    assert "candidate_pnl=+7.00" in no_entries
    assert "delta=+67.00" in no_entries
    assert "interpretation=HELPFUL so far" in no_entries

    no_orb = _candidate_line("candidate_no_orb_after_13")
    assert "remaining=3" in no_orb
    assert "excluded=3" in no_orb
    assert "candidate_pnl=-3.00" in no_orb
    assert "delta=+57.00" in no_orb

    ticker_rule = _candidate_line("candidate_exclude_arm_qcom_crwd")
    assert "remaining=3" in ticker_rule
    assert "excluded=3" in ticker_rule
    assert "candidate_pnl=-5.00" in ticker_rule
    assert "delta=+55.00" in ticker_rule

    arm_qcom = _candidate_line("candidate_exclude_arm_qcom")
    assert "remaining=4" in arm_qcom
    assert "excluded=2" in arm_qcom
    assert "delta=+45.00" in arm_qcom

    no_hrv = _candidate_line("candidate_no_hrv")
    assert "remaining=3" in no_hrv
    assert "excluded=3" in no_hrv
    assert "candidate_pnl=-57.00" in no_hrv
    assert "delta=+3.00" in no_hrv
    assert "interpretation=MIXED/INSUFFICIENT" in no_hrv


def test_trade_analytics_clean_shadow_watchlists_use_new_field_and_unknown_fallback(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _write_shadow_session_history(tmp_path / "logs" / "shadow_sessions.jsonl", [
        {
            "timestamp": datetime.now().isoformat(),
            "session_id": "shadow-MOMENTUM",
            "status": SESSION_ENDED,
            "config": {"watchlist_name": "momentum_universe"},
            "counters": {"data_quality": "CLEAN", "data_gap_count": 0},
        },
        {
            "timestamp": datetime.now().isoformat(),
            "session_id": "shadow-OLD",
            "status": SESSION_ENDED,
            "config": {},
            "counters": {"data_quality": "CLEAN", "data_gap_count": 0},
        },
    ])

    result = cli_runner.invoke(app, [
        "trade-analytics", "--include-shadow", "--clean-shadow-only",
    ])

    assert result.exit_code == 0, result.output
    assert "Watchlists: momentum_universe=1, unknown=1" in result.output
    assert "Note: some older shadow sessions do not store a watchlist name." in result.output


def test_trade_analytics_clean_shadow_only_handles_no_clean_sessions(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _write_shadow_session_history(tmp_path / "logs" / "shadow_sessions.jsonl", [
        {
            "timestamp": datetime.now().isoformat(),
            "session_id": "shadow-PARTIAL",
            "status": SESSION_ENDED,
            "counters": {"data_quality": "PARTIAL_DATA", "data_gap_count": 1},
        }
    ])

    result = cli_runner.invoke(app, [
        "trade-analytics", "--include-shadow", "--clean-shadow-only",
    ])

    assert result.exit_code == 0, result.output
    assert "=== Clean Shadow Sample ===" in result.output
    assert "No clean completed shadow sessions found." in result.output
    assert "Clean shadow sample is still small" in result.output


# -- Discord formatters -------------------------------------------------------

def test_shadow_discord_formatters_are_clearly_labelled():
    from alerts.discord_alerts import (
        format_shadow_entry_discord_alert,
        format_shadow_exit_discord_alert,
    )

    trade = create_shadow_trade_from_setup(
        _setup(), shares=10.0, session_id=SESSION_ID, now=NOW,
    )
    entry_message = format_shadow_entry_discord_alert(trade)
    assert "SHADOW ENTRY" in entry_message
    assert "SIMULATION ONLY" in entry_message
    assert "No real order placed." in entry_message
    assert "SPY" in entry_message

    trade.exit_price = 99.0
    trade.realized_pnl = -10.0
    exit_message = format_shadow_exit_discord_alert(trade, "EXIT_VWAP_LOST")
    assert "SHADOW EXIT" in exit_message
    assert "SIMULATION ONLY" in exit_message
    assert "No real order placed." in exit_message
    assert "EXIT_VWAP_LOST" in exit_message
