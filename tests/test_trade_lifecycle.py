"""
Tests for trade lifecycle analytics (trades/trade_lifecycle.py).

Lifecycle analytics join closed manual trades with trade_monitor.jsonl history.
Read-only: never places orders or mutates journals.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import typer.testing

from main import app
from trades.trade_lifecycle import (
    EXIT_REASON_MANUAL,
    TradeLifecycleAnalytics,
)
from trades.trade_tracker import ManualTrade, TradeStatus, TradeTracker


@pytest.fixture
def cli_runner():
    return typer.testing.CliRunner()


def _trade(**overrides) -> ManualTrade:
    defaults = dict(
        trade_id="t1",
        ticker="SPY",
        setup_type="vwap_reclaim_long",
        entry_price=100.0,
        stop_loss=98.0,
        target_1=104.0,
        target_2=106.0,
        shares=10.0,
        entry_time="2026-06-08T10:00:00",
        status=TradeStatus.CLOSED.value,
        exit_price=105.0,
        exit_time="2026-06-08T10:30:00",
        realized_pnl=50.0,
    )
    defaults.update(overrides)
    return ManualTrade(**defaults)


def _write_monitor_events(path: Path, events: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def _analytics(tmp_path, trades, events=None) -> TradeLifecycleAnalytics:
    trades_path = tmp_path / "manual_trades.jsonl"
    monitor_path = tmp_path / "trade_monitor.jsonl"
    tracker = TradeTracker(storage_path=trades_path)
    for trade in trades:
        tracker.add_trade(trade)
    if events is not None:
        _write_monitor_events(monitor_path, events)
    return TradeLifecycleAnalytics(trades_path=trades_path, monitor_path=monitor_path)


def _event(trade_id="t1", action="HOLD", timestamp="2026-06-08T10:05:00") -> dict:
    return {"timestamp": timestamp, "trade_id": trade_id, "action": action}


# ---------------------------------------------------------------------------
# Per-trade lifecycle fields
# ---------------------------------------------------------------------------

def test_first_monitor_action_and_first_alert(tmp_path):
    analytics = _analytics(tmp_path, [_trade()], [
        _event(action="HOLD", timestamp="2026-06-08T10:05:00"),
        _event(action="WATCH", timestamp="2026-06-08T10:10:00"),
        _event(action="TARGET_1_HIT", timestamp="2026-06-08T10:20:00"),
    ])

    lc = analytics.build_lifecycles()[0]
    assert lc.first_monitor_action == "HOLD"
    assert lc.first_alert_action == "WATCH"
    # 10:00 entry -> 10:10 first non-HOLD alert = 600 seconds
    assert lc.time_to_first_alert_seconds == pytest.approx(600.0)
    assert lc.monitor_event_count == 3


def test_time_in_trade(tmp_path):
    analytics = _analytics(tmp_path, [_trade()], [])
    lc = analytics.build_lifecycles()[0]
    # 10:00 -> 10:30 = 1800 seconds
    assert lc.time_in_trade_seconds == pytest.approx(1800.0)


@pytest.mark.parametrize("action", ["STOP_HIT", "TARGET_1_HIT", "TARGET_2_HIT", "EXIT_VWAP_LOST"])
def test_exit_reason_classification(tmp_path, action):
    analytics = _analytics(tmp_path, [_trade()], [
        _event(action=action, timestamp="2026-06-08T10:25:00"),
    ])

    lc = analytics.build_lifecycles()[0]
    assert lc.exit_reason == action
    assert lc.exit_action_followed is True


def test_exit_reason_manual_when_no_exit_signal(tmp_path):
    analytics = _analytics(tmp_path, [_trade()], [
        _event(action="HOLD"),
        _event(action="WATCH", timestamp="2026-06-08T10:10:00"),
    ])

    lc = analytics.build_lifecycles()[0]
    assert lc.exit_reason == EXIT_REASON_MANUAL
    assert lc.exit_action_followed is False


def test_exit_reason_uses_last_signal_before_close(tmp_path):
    analytics = _analytics(tmp_path, [_trade()], [
        _event(action="TARGET_1_HIT", timestamp="2026-06-08T10:10:00"),
        _event(action="TARGET_2_HIT", timestamp="2026-06-08T10:20:00"),
    ])

    lc = analytics.build_lifecycles()[0]
    assert lc.exit_reason == "TARGET_2_HIT"


def test_exit_signals_after_close_are_ignored(tmp_path):
    # Trade closed at 10:30; STOP_HIT logged afterwards must not explain the exit.
    analytics = _analytics(tmp_path, [_trade()], [
        _event(action="STOP_HIT", timestamp="2026-06-08T11:00:00"),
    ])

    lc = analytics.build_lifecycles()[0]
    assert lc.exit_reason == EXIT_REASON_MANUAL
    assert lc.exit_action_followed is False


def test_no_monitor_history_is_manual_with_no_alert_times(tmp_path):
    analytics = _analytics(tmp_path, [_trade()])  # monitor file never written

    lc = analytics.build_lifecycles()[0]
    assert lc.exit_reason == EXIT_REASON_MANUAL
    assert lc.first_monitor_action is None
    assert lc.first_alert_action is None
    assert lc.time_to_first_alert_seconds is None
    assert lc.monitor_event_count == 0


def test_open_and_excluded_trades_skipped(tmp_path):
    analytics = _analytics(tmp_path, [
        _trade(trade_id="closed1"),
        _trade(trade_id="open1", status=TradeStatus.OPEN.value, exit_price=None,
               exit_time=None, realized_pnl=None),
        _trade(trade_id="excl1", exclude_from_analytics=True, exclusion_reason="test"),
    ], [])

    ids = [lc.trade_id for lc in analytics.build_lifecycles()]
    assert ids == ["closed1"]


def test_legacy_trade_without_new_fields_compatible(tmp_path):
    # Simulate an old journal line written before the newer ManualTrade fields.
    trades_path = tmp_path / "manual_trades.jsonl"
    legacy = {
        "ticker": "QQQ", "setup_type": "vwap_reclaim_long",
        "entry_price": 400.0, "stop_loss": 398.0, "target_1": 404.0,
        "target_2": 408.0, "shares": 5.0, "entry_time": "2026-06-08T09:00:00",
        "trade_id": "legacy1", "status": "CLOSED", "exit_price": 404.0,
        "exit_time": "2026-06-08T09:45:00", "realized_pnl": 20.0, "notes": "",
    }
    trades_path.write_text(json.dumps(legacy) + "\n")

    analytics = TradeLifecycleAnalytics(
        trades_path=trades_path, monitor_path=tmp_path / "trade_monitor.jsonl"
    )
    lc = analytics.build_lifecycles()[0]
    assert lc.trade_id == "legacy1"
    assert lc.exit_reason == EXIT_REASON_MANUAL
    assert lc.time_in_trade_seconds == pytest.approx(45 * 60)


# ---------------------------------------------------------------------------
# Aggregated stats
# ---------------------------------------------------------------------------

def test_by_exit_reason_stats(tmp_path):
    analytics = _analytics(tmp_path, [
        _trade(trade_id="v1", realized_pnl=30.0),
        _trade(trade_id="v2", realized_pnl=-10.0),
        _trade(trade_id="m1", realized_pnl=15.0),
    ], [
        _event(trade_id="v1", action="EXIT_VWAP_LOST", timestamp="2026-06-08T10:20:00"),
        _event(trade_id="v2", action="EXIT_VWAP_LOST", timestamp="2026-06-08T10:21:00"),
        # m1 has no exit signal -> MANUAL
    ])

    result = analytics.analyze()
    by_reason = {row["exit_reason"]: row for row in result.by_exit_reason}

    vwap = by_reason["EXIT_VWAP_LOST"]
    assert vwap["trades"] == 2
    assert vwap["avg_pnl"] == pytest.approx(10.0)
    assert vwap["win_rate"] == pytest.approx(0.5)
    assert vwap["followed_count"] == 2
    assert vwap["label"] == "VWAP-loss exits"

    manual = by_reason[EXIT_REASON_MANUAL]
    assert manual["trades"] == 1
    assert manual["followed_count"] == 0


def test_analyze_effectiveness_mirrors_exit_reason(tmp_path):
    analytics = _analytics(tmp_path, [_trade()], [
        _event(action="TARGET_1_HIT", timestamp="2026-06-08T10:20:00"),
    ])

    result = analytics.analyze()
    assert result.effectiveness == result.by_exit_reason
    assert result.lifecycles[0]["exit_reason"] == "TARGET_1_HIT"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_trade_analytics_cli_prints_lifecycle_sections(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker()  # default logs/manual_trades.jsonl under cwd
    tracker.add_trade(_trade(trade_id="cli1", realized_pnl=40.0))
    _write_monitor_events(Path("logs/trade_monitor.jsonl"), [
        _event(trade_id="cli1", action="TARGET_1_HIT", timestamp="2026-06-08T10:20:00"),
    ])

    result = cli_runner.invoke(app, ["trade-analytics"])

    assert result.exit_code == 0
    assert "=== By Exit Reason ===" in result.stdout
    assert "=== Monitor Effectiveness ===" in result.stdout
    assert "TARGET_1_HIT" in result.stdout
    assert "Target 1 exits" in result.stdout


def test_trade_analytics_cli_lifecycle_manual_only(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker()
    tracker.add_trade(_trade(trade_id="cli2"))

    result = cli_runner.invoke(app, ["trade-analytics"])

    assert result.exit_code == 0
    assert "MANUAL" in result.stdout
    assert "Manual exits" in result.stdout


# ---------------------------------------------------------------------------
# Dashboard frames entry point
# ---------------------------------------------------------------------------

def test_analyze_frames_matches_file_analytics(tmp_path):
    trades_df = pd.DataFrame([_trade(trade_id="d1", realized_pnl=25.0).to_dict()])
    monitor_df = pd.DataFrame([
        _event(trade_id="d1", action="TARGET_2_HIT", timestamp="2026-06-08T10:20:00"),
    ])

    result = TradeLifecycleAnalytics.analyze_frames(trades_df, monitor_df)

    assert result.lifecycles[0]["exit_reason"] == "TARGET_2_HIT"
    assert result.by_exit_reason[0]["trades"] == 1


def test_analyze_frames_empty_inputs(tmp_path):
    empty = TradeLifecycleAnalytics.analyze_frames(pd.DataFrame(), pd.DataFrame())
    assert empty.lifecycles == []
    assert empty.by_exit_reason == []

    none_result = TradeLifecycleAnalytics.analyze_frames(None, None)
    assert none_result.lifecycles == []


def test_dashboard_helper_summarize_trade_lifecycle():
    from dashboard.helpers import summarize_trade_lifecycle

    trades_df = pd.DataFrame([_trade(trade_id="h1").to_dict()])
    monitor_df = pd.DataFrame([
        _event(trade_id="h1", action="STOP_HIT", timestamp="2026-06-08T10:20:00"),
    ])

    summary = summarize_trade_lifecycle(trades_df, monitor_df)
    assert summary["lifecycles"][0]["exit_reason"] == "STOP_HIT"
    assert summary["by_exit_reason"][0]["label"] == "Stop exits"


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def test_lifecycle_source_has_no_broker_imports():
    src = Path("trades/trade_lifecycle.py").read_text().lower()
    for pat in ["import alpaca", "from alpaca", "submit_order", "order_executor",
                "tradeapi", "place_order"]:
        assert pat not in src, f"Lifecycle analytics must not reference {pat!r}"
