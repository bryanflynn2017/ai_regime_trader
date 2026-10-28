"""
Tests for manual trade tracking (trades/trade_tracker.py + CLI commands).

The tracker is a JOURNAL ONLY:
  - It persists trades to a JSONL file.
  - It computes realized P&L on close for long trades.
  - It NEVER places orders or contacts a broker.

Coverage:
  - add/list/get/close on TradeTracker
  - P&L math and OPEN/CLOSED transitions
  - CLI commands trade-add / trade-list / trade-close
  - No broker / order executor is ever touched
  - Invalid inputs are handled clearly
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import typer.testing

from main import app
from trades.trade_tracker import TradeTracker, ManualTrade, TradeStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cli_runner():
    return typer.testing.CliRunner()


@pytest.fixture
def tracker(tmp_path):
    """A TradeTracker backed by an isolated temp file."""
    return TradeTracker(storage_path=tmp_path / "manual_trades.jsonl")


def _make_trade(**overrides) -> ManualTrade:
    defaults = dict(
        ticker="SPY",
        setup_type="vwap_reclaim_long",
        entry_price=500.0,
        stop_loss=498.0,
        target_1=504.0,
        target_2=508.0,
        shares=10.0,
        entry_time="2026-06-04T10:00:00",
    )
    defaults.update(overrides)
    return ManualTrade(**defaults)


# ---------------------------------------------------------------------------
# TradeTracker unit tests
# ---------------------------------------------------------------------------

def test_add_trade_persists_to_jsonl(tracker):
    trade = _make_trade()
    saved = tracker.add_trade(trade)

    assert saved.trade_id
    assert tracker.storage_path.exists()

    # File should contain exactly one JSON line for this trade
    lines = [l for l in tracker.storage_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["ticker"] == "SPY"
    assert record["status"] == "OPEN"


def test_trade_id_generated_when_missing(tracker):
    trade = _make_trade(trade_id="")
    saved = tracker.add_trade(trade)
    assert saved.trade_id  # non-empty generated id
    assert "SPY" in saved.trade_id


def test_trade_id_preserved_when_provided(tracker):
    trade = _make_trade(trade_id="my-custom-id")
    saved = tracker.add_trade(trade)
    assert saved.trade_id == "my-custom-id"


def test_manual_trade_loads_old_record_without_setup_fields(tmp_path):
    log_file = tmp_path / "manual_trades.jsonl"
    old_record = {
        "trade_id": "old",
        "ticker": "SPY",
        "setup_type": "vwap_reclaim_long",
        "entry_price": 500.0,
        "stop_loss": 498.0,
        "target_1": 504.0,
        "target_2": 508.0,
        "shares": 10.0,
        "entry_time": "2026-06-08T10:00:00",
        "status": "OPEN",
    }
    log_file.write_text(json.dumps(old_record) + "\n")

    trade = TradeTracker(storage_path=log_file).get_trade("old")

    assert trade is not None
    assert trade.setup_alert_id is None
    assert trade.setup_rank is None
    assert trade.setup_score is None
    assert trade.setup_confidence is None
    assert trade.exclude_from_analytics is False
    assert trade.exclusion_reason == ""
    assert trade.test_type == "forward_test"
    assert trade.thesis == ""
    assert trade.market_context == ""
    assert trade.entry_quality == ""
    assert trade.mistake is False
    assert trade.mistake_notes == ""
    assert trade.screenshot_path == ""


def test_edit_closed_trade_exit_price_recalculates_pnl(tracker):
    trade = _make_trade(trade_id="closed-edit")
    tracker.add_trade(trade)
    tracker.close_trade("closed-edit", exit_price=505.0)

    corrected = tracker.edit_trade("closed-edit", exit_price=503.0, correction_notes="wrong fill")

    assert corrected.exit_price == pytest.approx(503.0)
    assert corrected.realized_pnl == pytest.approx(30.0)
    assert corrected.metadata["corrected_at"]
    assert corrected.metadata["correction_notes"] == "wrong fill"
    assert corrected.metadata["corrections"][-1]["fields"] == ["exit_price", "realized_pnl"]


def test_edit_open_trade_exit_price_rejected(tracker):
    tracker.add_trade(_make_trade(trade_id="open-edit"))

    with pytest.raises(ValueError, match="CLOSED trades"):
        tracker.edit_trade("open-edit", exit_price=503.0)


def test_list_open_trades(tracker):
    tracker.add_trade(_make_trade(trade_id="t1"))
    tracker.add_trade(_make_trade(trade_id="t2", ticker="QQQ"))

    open_trades = tracker.list_open_trades()
    assert len(open_trades) == 2
    assert all(t.status == TradeStatus.OPEN.value for t in open_trades)


def test_list_all_trades(tracker):
    tracker.add_trade(_make_trade(trade_id="t1"))
    tracker.add_trade(_make_trade(trade_id="t2"))
    tracker.close_trade("t1", exit_price=505.0)

    all_trades = tracker.list_all_trades()
    assert len(all_trades) == 2


def test_get_trade(tracker):
    tracker.add_trade(_make_trade(trade_id="findme"))
    found = tracker.get_trade("findme")
    assert found is not None
    assert found.trade_id == "findme"


def test_get_trade_missing_returns_none(tracker):
    assert tracker.get_trade("nope") is None


def test_close_trade_calculates_pnl(tracker):
    tracker.add_trade(_make_trade(trade_id="t1", entry_price=500.0, shares=10.0))
    closed = tracker.close_trade("t1", exit_price=505.0)

    # (505 - 500) * 10 = 50
    assert closed.realized_pnl == pytest.approx(50.0)
    assert closed.status == TradeStatus.CLOSED.value
    assert closed.exit_price == 505.0
    assert closed.exit_time is not None


def test_close_trade_negative_pnl(tracker):
    tracker.add_trade(_make_trade(trade_id="t1", entry_price=500.0, shares=4.0))
    closed = tracker.close_trade("t1", exit_price=497.5)
    # (497.5 - 500) * 4 = -10
    assert closed.realized_pnl == pytest.approx(-10.0)


def test_closed_trade_not_in_open_list(tracker):
    tracker.add_trade(_make_trade(trade_id="t1"))
    tracker.add_trade(_make_trade(trade_id="t2"))
    tracker.close_trade("t1", exit_price=505.0)

    open_ids = [t.trade_id for t in tracker.list_open_trades()]
    assert "t1" not in open_ids
    assert "t2" in open_ids


def test_close_unknown_trade_raises(tracker):
    with pytest.raises(ValueError):
        tracker.close_trade("does-not-exist", exit_price=100.0)


def test_close_already_closed_raises(tracker):
    tracker.add_trade(_make_trade(trade_id="t1"))
    tracker.close_trade("t1", exit_price=505.0)
    with pytest.raises(ValueError):
        tracker.close_trade("t1", exit_price=506.0)


def test_close_trade_persists(tracker):
    tracker.add_trade(_make_trade(trade_id="t1"))
    tracker.close_trade("t1", exit_price=505.0)

    # Re-read from disk via a fresh tracker
    fresh = TradeTracker(storage_path=tracker.storage_path)
    reloaded = fresh.get_trade("t1")
    assert reloaded.status == TradeStatus.CLOSED.value
    assert reloaded.realized_pnl == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

def test_cli_trade_add(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = cli_runner.invoke(
        app,
        ["trade-add", "--ticker", "SPY", "--setup-type", "vwap_reclaim_long",
         "--entry", "500", "--stop", "498", "--target1", "504",
         "--target2", "508", "--shares", "10"],
    )
    assert result.exit_code == 0
    assert "Trade ID:" in result.stdout
    # File written under cwd/logs
    log_file = tmp_path / "logs" / "manual_trades.jsonl"
    assert log_file.exists()


def test_cli_trade_add_accepts_setup_link_fields(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = cli_runner.invoke(
        app,
        [
            "trade-add", "--ticker", "SPY", "--setup-type", "vwap_reclaim_long",
            "--entry", "500", "--stop", "498", "--target1", "504",
            "--target2", "508", "--shares", "10",
            "--setup-alert-id", "alert-123",
            "--setup-rank", "2",
            "--setup-score", "4.25",
            "--setup-confidence", "0.72",
        ],
    )

    assert result.exit_code == 0, result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").list_all_trades()[0]
    assert trade.setup_alert_id == "alert-123"
    assert trade.setup_rank == 2
    assert trade.setup_score == 4.25
    assert trade.setup_confidence == 0.72


def test_cli_trade_add_can_create_excluded_trade(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = cli_runner.invoke(
        app,
        [
            "trade-add", "--ticker", "SPY", "--setup-type", "workflow_test",
            "--entry", "100", "--stop", "99", "--target1", "102",
            "--target2", "104", "--shares", "1",
            "--exclude-from-analytics", "--exclusion-reason", "paper workflow test",
        ],
    )

    assert result.exit_code == 0, result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").list_all_trades()[0]
    assert trade.exclude_from_analytics is True
    assert trade.exclusion_reason == "paper workflow test"


def test_cli_trade_add_stores_forward_test_fields(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = cli_runner.invoke(
        app,
        [
            "trade-add", "--ticker", "SPY", "--setup-type", "vwap_reclaim_long",
            "--entry", "100", "--stop", "99", "--target1", "102",
            "--target2", "104", "--shares", "1",
            "--test-type", "live_paper",
            "--thesis", "VWAP reclaim with relative volume",
            "--market-context", "QQQ strong",
            "--entry-quality", "A",
            "--mistake",
            "--mistake-notes", "chased entry",
            "--screenshot-path", "screenshots/spy.png",
        ],
    )

    assert result.exit_code == 0, result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").list_all_trades()[0]
    assert trade.test_type == "live_paper"
    assert trade.thesis == "VWAP reclaim with relative volume"
    assert trade.market_context == "QQQ strong"
    assert trade.entry_quality == "A"
    assert trade.mistake is True
    assert trade.mistake_notes == "chased entry"
    assert trade.screenshot_path == "screenshots/spy.png"


def _write_setup_alert(tmp_path, **overrides):
    import main as m

    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    record = {
        "timestamp": "2026-06-08T10:00:00",
        "ticker": "SPY",
        "setup_type": "vwap_reclaim_long",
        "rank": 2,
        "score": 4.25,
        "confidence": 0.72,
        "entry_trigger": 500.0,
        "stop_loss": 498.0,
        "target_1": 504.0,
        "target_2": 508.0,
    }
    record.update(overrides)
    (log_dir / "setup_alerts.jsonl").write_text(json.dumps(record) + "\n")
    return record, m._setup_alert_display_id(record)


def _write_setup_alerts(tmp_path, records):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / "setup_alerts.jsonl").write_text("\n".join(json.dumps(record) for record in records))


def _alert_minutes_ago(minutes: float) -> str:
    return (datetime.now() - timedelta(minutes=minutes)).isoformat()


def test_cli_trade_add_from_alert_creates_trade(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, alert_id = _write_setup_alert(tmp_path)

    result = cli_runner.invoke(
        app,
        ["trade-add-from-alert", "--alert-id", alert_id, "--shares", "10"],
    )

    assert result.exit_code == 0, result.stdout
    assert "Trade Recorded From Setup Alert" in result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").list_all_trades()[0]
    assert trade.ticker == "SPY"
    assert trade.entry_price == 500.0
    assert trade.stop_loss == 498.0
    assert trade.target_1 == 504.0
    assert trade.target_2 == 508.0


def test_cli_trade_add_from_alert_persists_setup_link_fields(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, alert_id = _write_setup_alert(tmp_path)

    result = cli_runner.invoke(app, ["trade-add-from-alert", "--alert-id", alert_id, "--shares", "10"])

    assert result.exit_code == 0, result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").list_all_trades()[0]
    assert trade.setup_alert_id == alert_id
    assert trade.setup_rank == 2
    assert trade.setup_score == 4.25
    assert trade.setup_confidence == 0.72


def test_cli_trade_add_from_alert_overrides_prices(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, alert_id = _write_setup_alert(tmp_path)

    result = cli_runner.invoke(
        app,
        [
            "trade-add-from-alert", "--alert-id", alert_id, "--shares", "10",
            "--entry", "501", "--stop", "499", "--target1", "505", "--target2", "509",
        ],
    )

    assert result.exit_code == 0, result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").list_all_trades()[0]
    assert trade.entry_price == 501.0
    assert trade.stop_loss == 499.0
    assert trade.target_1 == 505.0
    assert trade.target_2 == 509.0


def test_cli_trade_add_from_alert_missing_alert_errors(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = cli_runner.invoke(
        app,
        ["trade-add-from-alert", "--alert-id", "missing", "--shares", "10"],
    )

    assert result.exit_code == 1
    assert "not found" in result.stdout.lower()
    assert not (tmp_path / "logs" / "manual_trades.jsonl").exists()


def test_cli_trade_add_from_alert_invalid_shares_errors(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, alert_id = _write_setup_alert(tmp_path)

    result = cli_runner.invoke(
        app,
        ["trade-add-from-alert", "--alert-id", alert_id, "--shares", "0"],
    )

    assert result.exit_code == 1
    assert "shares" in result.stdout.lower()


def test_cli_trade_add_from_alert_stop_above_entry_errors(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, alert_id = _write_setup_alert(tmp_path)

    result = cli_runner.invoke(
        app,
        ["trade-add-from-alert", "--alert-id", alert_id, "--shares", "10", "--stop", "501"],
    )

    assert result.exit_code == 1
    assert "stop" in result.stdout.lower()


def test_cli_trade_add_from_alert_no_broker_or_order_execution(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, alert_id = _write_setup_alert(tmp_path)

    from unittest.mock import patch
    with patch("broker.order_executor.OrderExecutor") as order_executor, patch(
        "broker.alpaca_adapter.AlpacaBrokerAdapter"
    ) as alpaca_adapter:
        result = cli_runner.invoke(
            app,
            ["trade-add-from-alert", "--alert-id", alert_id, "--shares", "10"],
        )

    assert result.exit_code == 0, result.stdout
    order_executor.assert_not_called()
    alpaca_adapter.assert_not_called()


def test_cli_trade_add_from_alert_can_create_excluded_trade(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, alert_id = _write_setup_alert(tmp_path)

    result = cli_runner.invoke(
        app,
        [
            "trade-add-from-alert", "--alert-id", alert_id, "--shares", "10",
            "--exclude-from-analytics", "--exclusion-reason", "stale alert test",
        ],
    )

    assert result.exit_code == 0, result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").list_all_trades()[0]
    assert trade.exclude_from_analytics is True
    assert trade.exclusion_reason == "stale alert test"


def test_cli_trade_add_from_alert_stores_forward_test_fields(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, alert_id = _write_setup_alert(tmp_path)

    result = cli_runner.invoke(
        app,
        [
            "trade-add-from-alert", "--alert-id", alert_id, "--shares", "10",
            "--test-type", "workflow_test",
            "--thesis", "testing alert journal flow",
            "--market-context", "synthetic context",
            "--entry-quality", "B",
            "--mistake-notes", "none",
            "--screenshot-path", "screenshots/test.png",
        ],
    )

    assert result.exit_code == 0, result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").list_all_trades()[0]
    assert trade.test_type == "workflow_test"
    assert trade.thesis == "testing alert journal flow"
    assert trade.market_context == "synthetic context"
    assert trade.entry_quality == "B"
    assert trade.mistake_notes == "none"
    assert trade.screenshot_path == "screenshots/test.png"


def test_cli_trade_add_latest_alert_creates_from_newest(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": _alert_minutes_ago(20), "ticker": "OLD", "setup_type": "vwap", "rank": 2, "score": 2.0, "confidence": 0.6, "entry_trigger": 90, "stop_loss": 88, "target_1": 94, "target_2": 96},
        {"timestamp": _alert_minutes_ago(5), "ticker": "NEW", "setup_type": "orb", "rank": 1, "score": 4.0, "confidence": 0.8, "entry_trigger": 100, "stop_loss": 98, "target_1": 104, "target_2": 106},
    ])

    result = cli_runner.invoke(app, ["trade-add-latest-alert", "--shares", "5"])

    assert result.exit_code == 0, result.stdout
    assert "Selected alert_id:" in result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").list_all_trades()[0]
    assert trade.ticker == "NEW"
    assert trade.setup_type == "orb"
    assert trade.setup_rank == 1


def test_cli_trade_add_latest_alert_ticker_filter(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": _alert_minutes_ago(10), "ticker": "SPY", "setup_type": "vwap", "confidence": 0.6, "entry_trigger": 90, "stop_loss": 88, "target_1": 94, "target_2": 96},
        {"timestamp": _alert_minutes_ago(5), "ticker": "QQQ", "setup_type": "orb", "confidence": 0.8, "entry_trigger": 100, "stop_loss": 98, "target_1": 104, "target_2": 106},
    ])

    result = cli_runner.invoke(app, ["trade-add-latest-alert", "--shares", "5", "--ticker", "SPY"])

    assert result.exit_code == 0, result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").list_all_trades()[0]
    assert trade.ticker == "SPY"


def test_cli_trade_add_latest_alert_setup_type_filter(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": _alert_minutes_ago(10), "ticker": "SPY", "setup_type": "vwap", "confidence": 0.6, "entry_trigger": 90, "stop_loss": 88, "target_1": 94, "target_2": 96},
        {"timestamp": _alert_minutes_ago(5), "ticker": "QQQ", "setup_type": "orb", "confidence": 0.8, "entry_trigger": 100, "stop_loss": 98, "target_1": 104, "target_2": 106},
    ])

    result = cli_runner.invoke(app, ["trade-add-latest-alert", "--shares", "5", "--setup-type", "vwap"])

    assert result.exit_code == 0, result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").list_all_trades()[0]
    assert trade.setup_type == "vwap"


def test_cli_trade_add_latest_alert_excludes_synthetic_by_default(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": _alert_minutes_ago(10), "ticker": "SPY", "setup_type": "vwap", "confidence": 0.6, "entry_trigger": 90, "stop_loss": 88, "target_1": 94, "target_2": 96, "metadata": {"source": "yfinance"}},
        {"timestamp": _alert_minutes_ago(5), "ticker": "SYNTHETIC", "setup_type": "orb", "confidence": 0.8, "entry_trigger": 100, "stop_loss": 98, "target_1": 104, "target_2": 106, "metadata": {"source": "synthetic"}},
    ])

    result = cli_runner.invoke(app, ["trade-add-latest-alert", "--shares", "5"])

    assert result.exit_code == 0, result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").list_all_trades()[0]
    assert trade.ticker == "SPY"


def test_cli_trade_add_latest_alert_overrides(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": _alert_minutes_ago(5), "ticker": "SPY", "setup_type": "vwap", "confidence": 0.6, "entry_trigger": 90, "stop_loss": 88, "target_1": 94, "target_2": 96},
    ])

    result = cli_runner.invoke(
        app,
        ["trade-add-latest-alert", "--shares", "5", "--entry", "91", "--stop", "89", "--target1", "95", "--target2", "97"],
    )

    assert result.exit_code == 0, result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").list_all_trades()[0]
    assert trade.entry_price == 91.0
    assert trade.stop_loss == 89.0
    assert trade.target_1 == 95.0
    assert trade.target_2 == 97.0


def test_cli_trade_add_latest_alert_no_matching_errors(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = cli_runner.invoke(app, ["trade-add-latest-alert", "--shares", "5"])
    assert result.exit_code == 1
    assert "no matching setup alert" in result.stdout.lower()


def test_cli_trade_add_latest_alert_invalid_shares_errors(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": "2026-06-08T10:00:00", "ticker": "SPY", "setup_type": "vwap", "confidence": 0.6, "entry_trigger": 90, "stop_loss": 88, "target_1": 94, "target_2": 96},
    ])

    result = cli_runner.invoke(app, ["trade-add-latest-alert", "--shares", "0"])
    assert result.exit_code == 1
    assert "shares" in result.stdout.lower()


def test_cli_trade_add_latest_alert_stop_above_entry_errors(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": _alert_minutes_ago(5), "ticker": "SPY", "setup_type": "vwap", "confidence": 0.6, "entry_trigger": 90, "stop_loss": 88, "target_1": 94, "target_2": 96},
    ])

    result = cli_runner.invoke(app, ["trade-add-latest-alert", "--shares", "5", "--stop", "91"])
    assert result.exit_code == 1
    assert "stop" in result.stdout.lower()


def test_cli_trade_add_latest_alert_fresh_alert_selected(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": _alert_minutes_ago(10), "ticker": "TSLA", "setup_type": "orb", "confidence": 0.8, "entry_trigger": 200, "stop_loss": 198, "target_1": 204, "target_2": 206},
    ])

    result = cli_runner.invoke(app, ["trade-add-latest-alert", "--shares", "5", "--ticker", "TSLA"])

    assert result.exit_code == 0, result.stdout
    assert "Selected alert_id:" in result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").list_all_trades()[0]
    assert trade.ticker == "TSLA"


def test_cli_trade_add_latest_alert_stale_rejected_by_default(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": _alert_minutes_ago(45), "ticker": "GOOGL", "setup_type": "vwap", "confidence": 0.7, "entry_trigger": 180, "stop_loss": 178, "target_1": 184, "target_2": 186},
    ])

    result = cli_runner.invoke(app, ["trade-add-latest-alert", "--shares", "5", "--ticker", "GOOGL"])

    assert result.exit_code == 1
    assert "No matching setup alert found within 30 minutes." in result.stdout
    assert not (tmp_path / "logs" / "manual_trades.jsonl").exists()


def test_cli_trade_add_latest_alert_allow_stale_selects_old_alert(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": _alert_minutes_ago(45), "ticker": "GOOGL", "setup_type": "vwap", "confidence": 0.7, "entry_trigger": 180, "stop_loss": 178, "target_1": 184, "target_2": 186},
    ])

    result = cli_runner.invoke(
        app,
        ["trade-add-latest-alert", "--shares", "5", "--ticker", "GOOGL", "--allow-stale-alert"],
    )

    assert result.exit_code == 0, result.stdout
    assert "Warning: using stale setup alert." in result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").list_all_trades()[0]
    assert trade.ticker == "GOOGL"


def test_cli_trade_add_latest_alert_stale_error_suggests_recent_command(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": _alert_minutes_ago(45), "ticker": "GOOGL", "setup_type": "vwap", "confidence": 0.7, "entry_trigger": 180, "stop_loss": 178, "target_1": 184, "target_2": 186},
    ])

    result = cli_runner.invoke(app, ["trade-add-latest-alert", "--shares", "5", "--ticker", "GOOGL"])

    assert result.exit_code == 1
    assert "python main.py setup-alerts-recent --ticker GOOGL --since-hours 4" in result.stdout


def test_cli_trade_add_latest_alert_selected_age_printed(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": _alert_minutes_ago(7), "ticker": "NVDA", "setup_type": "orb", "confidence": 0.8, "entry_trigger": 150, "stop_loss": 148, "target_1": 154, "target_2": 156},
    ])

    result = cli_runner.invoke(app, ["trade-add-latest-alert", "--shares", "5", "--ticker", "NVDA"])

    assert result.exit_code == 0, result.stdout
    assert "Selected alert_id:" in result.stdout
    assert "(age:" in result.stdout
    assert "minutes)" in result.stdout


def test_cli_trade_add_latest_alert_can_create_excluded_trade(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": _alert_minutes_ago(7), "ticker": "NVDA", "setup_type": "orb", "confidence": 0.8, "entry_trigger": 150, "stop_loss": 148, "target_1": 154, "target_2": 156},
    ])

    result = cli_runner.invoke(
        app,
        [
            "trade-add-latest-alert", "--shares", "5", "--ticker", "NVDA",
            "--exclude-from-analytics", "--exclusion-reason", "workflow test",
        ],
    )

    assert result.exit_code == 0, result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").list_all_trades()[0]
    assert trade.exclude_from_analytics is True
    assert trade.exclusion_reason == "workflow test"


def test_cli_trade_add_latest_alert_stores_forward_test_fields(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": _alert_minutes_ago(7), "ticker": "NVDA", "setup_type": "orb", "confidence": 0.8, "entry_trigger": 150, "stop_loss": 148, "target_1": 154, "target_2": 156},
    ])

    result = cli_runner.invoke(
        app,
        [
            "trade-add-latest-alert", "--shares", "5", "--ticker", "NVDA",
            "--test-type", "forward_test",
            "--thesis", "opening range continuation",
            "--market-context", "semis strong",
            "--entry-quality", "A",
            "--mistake",
            "--mistake-notes", "entered late",
            "--screenshot-path", "screenshots/nvda.png",
        ],
    )

    assert result.exit_code == 0, result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").list_all_trades()[0]
    assert trade.test_type == "forward_test"
    assert trade.thesis == "opening range continuation"
    assert trade.market_context == "semis strong"
    assert trade.entry_quality == "A"
    assert trade.mistake is True
    assert trade.mistake_notes == "entered late"
    assert trade.screenshot_path == "screenshots/nvda.png"


def test_setup_alerts_recent_newest_first(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": "2026-06-08T09:00:00", "ticker": "OLD", "setup_type": "vwap", "confidence": 0.7, "entry_trigger": 1, "stop_loss": 0.9, "target_1": 1.2, "target_2": 1.3},
        {"timestamp": "2026-06-08T10:00:00", "ticker": "NEW", "setup_type": "vwap", "confidence": 0.7, "entry_trigger": 1, "stop_loss": 0.9, "target_1": 1.2, "target_2": 1.3},
    ])

    result = cli_runner.invoke(app, ["setup-alerts-recent"])

    assert result.exit_code == 0, result.stdout
    assert result.stdout.index("NEW") < result.stdout.index("OLD")


def test_setup_alerts_recent_limit_works(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": "2026-06-08T09:00:00", "ticker": "A", "setup_type": "vwap", "confidence": 0.7, "entry_trigger": 1, "stop_loss": 0.9, "target_1": 1.2, "target_2": 1.3},
        {"timestamp": "2026-06-08T10:00:00", "ticker": "B", "setup_type": "vwap", "confidence": 0.7, "entry_trigger": 1, "stop_loss": 0.9, "target_1": 1.2, "target_2": 1.3},
    ])

    result = cli_runner.invoke(app, ["setup-alerts-recent", "--limit", "1"])

    assert result.exit_code == 0, result.stdout
    assert "B" in result.stdout
    assert "2026-06-08T09:00:00" not in result.stdout


def test_setup_alerts_recent_ticker_filter(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": "2026-06-08T09:00:00", "ticker": "SPY", "setup_type": "vwap", "confidence": 0.7, "entry_trigger": 1, "stop_loss": 0.9, "target_1": 1.2, "target_2": 1.3},
        {"timestamp": "2026-06-08T10:00:00", "ticker": "QQQ", "setup_type": "vwap", "confidence": 0.7, "entry_trigger": 1, "stop_loss": 0.9, "target_1": 1.2, "target_2": 1.3},
    ])

    result = cli_runner.invoke(app, ["setup-alerts-recent", "--ticker", "SPY"])

    assert result.exit_code == 0, result.stdout
    assert "SPY" in result.stdout
    assert "QQQ" not in result.stdout


def test_setup_alerts_recent_since_hours_filter():
    import main as m
    from datetime import datetime

    records = [
        {"timestamp": "2026-06-08T09:30:00", "ticker": "NEW", "setup_type": "vwap"},
        {"timestamp": "2026-06-08T07:00:00", "ticker": "OLD", "setup_type": "vwap"},
    ]

    filtered = m._filter_setup_alert_records(
        records,
        since_hours=2,
        now=datetime.fromisoformat("2026-06-08T10:00:00"),
    )

    assert [record["ticker"] for record in filtered] == ["NEW"]


def test_setup_alerts_recent_excludes_synthetic(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": "2026-06-08T09:00:00", "ticker": "SYN", "setup_type": "vwap", "confidence": 0.7, "entry_trigger": 1, "stop_loss": 0.9, "target_1": 1.2, "target_2": 1.3, "metadata": {"source": "synthetic"}},
        {"timestamp": "2026-06-08T10:00:00", "ticker": "SPY", "setup_type": "vwap", "confidence": 0.7, "entry_trigger": 1, "stop_loss": 0.9, "target_1": 1.2, "target_2": 1.3, "metadata": {"source": "yfinance"}},
    ])

    result = cli_runner.invoke(app, ["setup-alerts-recent"])

    assert result.exit_code == 0, result.stdout
    assert "SPY" in result.stdout
    assert "SYN" not in result.stdout


def test_setup_alerts_recent_missing_metadata_handled(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_setup_alerts(tmp_path, [
        {"timestamp": "2026-06-08T10:00:00", "ticker": "SPY", "setup_type": "vwap", "confidence": 0.7, "entry_trigger": 1, "stop_loss": 0.9, "target_1": 1.2, "target_2": 1.3},
    ])

    result = cli_runner.invoke(app, ["setup-alerts-recent"])

    assert result.exit_code == 0, result.stdout
    assert "SPY" in result.stdout


def test_cli_trade_list(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli_runner.invoke(
        app,
        ["trade-add", "--ticker", "QQQ", "--setup-type", "vwap_reclaim_long",
         "--entry", "400", "--stop", "398", "--target1", "404",
         "--target2", "408", "--shares", "5"],
    )
    result = cli_runner.invoke(app, ["trade-list"])
    assert result.exit_code == 0
    assert "QQQ" in result.stdout
    assert "Open Trades" in result.stdout


def test_trade_mark_excluded_and_included(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker()
    trade = tracker.add_trade(_make_trade(trade_id="mark-me"))

    excluded = cli_runner.invoke(
        app,
        ["trade-mark-excluded", "--trade-id", trade.trade_id, "--reason", "paper workflow test"],
    )
    assert excluded.exit_code == 0, excluded.stdout
    updated = tracker.get_trade(trade.trade_id)
    assert updated.exclude_from_analytics is True
    assert updated.exclusion_reason == "paper workflow test"

    included = cli_runner.invoke(app, ["trade-mark-included", "--trade-id", trade.trade_id])
    assert included.exit_code == 0, included.stdout
    updated = tracker.get_trade(trade.trade_id)
    assert updated.exclude_from_analytics is False
    assert updated.exclusion_reason == ""


def test_trade_list_shows_excluded_marker(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(
        _make_trade(
            trade_id="excluded-open",
            exclude_from_analytics=True,
            exclusion_reason="paper workflow test",
        )
    )

    result = cli_runner.invoke(app, ["trade-list"])

    assert result.exit_code == 0, result.stdout
    assert "[EXCLUDED]" in result.stdout
    assert "paper workflow test" in result.stdout


def test_trade_list_shows_test_type_thesis_and_mistake(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(
        _make_trade(
            trade_id="rich-open",
            test_type="live_paper",
            thesis="VWAP reclaim with volume",
            mistake=True,
            mistake_notes="chased entry",
        )
    )

    result = cli_runner.invoke(app, ["trade-list"])

    assert result.exit_code == 0, result.stdout
    assert "test_type: live_paper" in result.stdout
    assert "thesis: VWAP reclaim with volume" in result.stdout
    assert "[MISTAKE]" in result.stdout
    assert "chased entry" in result.stdout


def test_cli_trade_close(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    add = cli_runner.invoke(
        app,
        ["trade-add", "--ticker", "SPY", "--setup-type", "vwap_reclaim_long",
         "--entry", "500", "--stop", "498", "--target1", "504",
         "--target2", "508", "--shares", "10"],
    )
    # Extract trade id from output
    trade_id = None
    for line in add.stdout.splitlines():
        if "Trade ID:" in line:
            trade_id = line.split("Trade ID:")[1].strip()
            break
    assert trade_id

    result = cli_runner.invoke(
        app, ["trade-close", "--trade-id", trade_id, "--exit-price", "505"]
    )
    assert result.exit_code == 0
    assert "Realized P&L:" in result.stdout
    assert "+50.00" in result.stdout


def test_cli_trade_close_unknown_id(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = cli_runner.invoke(
        app, ["trade-close", "--trade-id", "nope", "--exit-price", "100"]
    )
    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_cli_trade_add_rejects_bad_shares(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = cli_runner.invoke(
        app,
        ["trade-add", "--ticker", "SPY", "--setup-type", "vwap_reclaim_long",
         "--entry", "500", "--stop", "498", "--target1", "504",
         "--target2", "508", "--shares", "0"],
    )
    assert result.exit_code == 1
    assert "shares" in result.stdout.lower()


def test_cli_trade_add_rejects_stop_above_entry(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = cli_runner.invoke(
        app,
        ["trade-add", "--ticker", "SPY", "--setup-type", "vwap_reclaim_long",
         "--entry", "500", "--stop", "501", "--target1", "504",
         "--target2", "508", "--shares", "10"],
    )
    assert result.exit_code == 1
    assert "stop" in result.stdout.lower()


def test_cli_trade_list_all_shows_closed(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    add = cli_runner.invoke(
        app,
        ["trade-add", "--ticker", "SPY", "--setup-type", "vwap_reclaim_long",
         "--entry", "500", "--stop", "498", "--target1", "504",
         "--target2", "508", "--shares", "10"],
    )
    trade_id = None
    for line in add.stdout.splitlines():
        if "Trade ID:" in line:
            trade_id = line.split("Trade ID:")[1].strip()
            break
    cli_runner.invoke(app, ["trade-close", "--trade-id", trade_id, "--exit-price", "505"])

    # Default list (open only) should be empty
    open_result = cli_runner.invoke(app, ["trade-list"])
    assert trade_id not in open_result.stdout

    # --all should include the closed trade
    all_result = cli_runner.invoke(app, ["trade-list", "--all"])
    assert trade_id in all_result.stdout
    assert "CLOSED" in all_result.stdout


def test_cli_trade_edit_closed_exit_price_recalculates_pnl(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker()
    tracker.add_trade(_make_trade(trade_id="edit-cli", ticker="TSLA", entry_price=390.0, shares=2.0))
    tracker.close_trade("edit-cli", exit_price=395.13)

    result = cli_runner.invoke(
        app,
        ["trade-edit", "--trade-id", "edit-cli", "--exit-price", "398.48", "--notes", "corrected fill"],
    )

    assert result.exit_code == 0, result.stdout
    assert "Trade Corrected" in result.stdout
    assert "398.48" in result.stdout
    assert "+16.96" in result.stdout

    corrected = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").get_trade("edit-cli")
    assert corrected.exit_price == pytest.approx(398.48)
    assert corrected.realized_pnl == pytest.approx(16.96)
    assert corrected.metadata["corrected_at"]
    assert corrected.metadata["correction_notes"] == "corrected fill"


def test_cli_trade_edit_open_exit_price_rejected(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(trade_id="open-cli"))

    result = cli_runner.invoke(
        app,
        ["trade-edit", "--trade-id", "open-cli", "--exit-price", "505"],
    )

    assert result.exit_code == 1
    assert "Exit price can only be edited for CLOSED trades" in result.stdout


def test_cli_trade_exclude_marks_trade_and_reason(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker()
    tracker.add_trade(_make_trade(trade_id="exclude-cli"))

    result = cli_runner.invoke(
        app,
        ["trade-exclude", "--trade-id", "exclude-cli", "--reason", "wrong journal entry"],
    )

    assert result.exit_code == 0, result.stdout
    assert "Trade Excluded From Analytics" in result.stdout
    trade = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl").get_trade("exclude-cli")
    assert trade.exclude_from_analytics is True
    assert trade.exclusion_reason == "wrong journal entry"
    assert "wrong journal entry" in trade.mistake_notes


def test_cli_trade_edit_output_does_not_print_secrets(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker()
    tracker.add_trade(_make_trade(trade_id="safe-cli"))
    tracker.close_trade("safe-cli", exit_price=505.0)

    result = cli_runner.invoke(
        app,
        ["trade-edit", "--trade-id", "safe-cli", "--exit-price", "506", "--notes", "routine correction"],
    )

    assert result.exit_code == 0, result.stdout
    output = result.stdout.lower()
    for banned in ("api_key", "secret", "password", "token"):
        assert banned not in output


# ---------------------------------------------------------------------------
# Safety: no broker / order executor
# ---------------------------------------------------------------------------

def test_tracker_source_has_no_broker_imports():
    """The tracker module must not import or call any broker/order execution code."""
    src = Path("trades/trade_tracker.py").read_text().lower()
    # Look for actual usage patterns, not the word "broker" in a safety docstring.
    forbidden_patterns = [
        "import alpaca",
        "from alpaca",
        "submit_order",
        "order_executor",
        "tradeapi",
        "place_order",
    ]
    for pat in forbidden_patterns:
        assert pat not in src, f"Tracker must not reference {pat!r}"
