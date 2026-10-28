"""Tests for the read-only daily-review CLI command."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import typer.testing

from main import app
from trades.trade_tracker import ManualTrade, TradeTracker


@pytest.fixture
def cli_runner():
    return typer.testing.CliRunner()


def _today_iso(hour: int = 10) -> str:
    now = datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)
    return now.isoformat()


@pytest.mark.parametrize("use_history", [True, False])
def test_shadow_partial_reasons_and_gap_diagnostics(cli_runner, tmp_path, monkeypatch, use_history):
    from trades.shadow_session import ShadowSessionState, append_shadow_session_history
    monkeypatch.chdir(tmp_path)
    state = ShadowSessionState(
        session_id="partial", started_at=_today_iso(), status="ENDED",
        config={"provider": "alpaca", "feed": "iex"},
        counters={
            "data_quality": "PARTIAL_DATA", "data_gap_count": 2,
            "normal_data_gap_count": 1, "extreme_data_gap_count": 1,
            "max_iteration_gap_seconds": 7200,
            "stale_warning_count": 3, "max_stale_bar_age_seconds": 1800,
            "first_stale_timestamp": _today_iso(10), "last_stale_timestamp": _today_iso(11),
            "stale_data_caused_partial": True,
            "partial_data_reasons": ["STALE_PROVIDER_DATA", "ITERATION_GAP"],
        },
    )
    if use_history:
        append_shadow_session_history(state)
    else:
        state.save()
    result = cli_runner.invoke(app, ["daily-review", "--no-save"])
    assert result.exit_code == 0, result.output
    assert "MULTIPLE_REASONS (ITERATION_GAP, STALE_PROVIDER_DATA)" in result.output
    assert "Normal data gaps: 1" in result.output
    assert "Extreme/session-suspension gaps: 1" in result.output
    assert "Max iteration gap: 120.0 min" in result.output
    assert "Stale warnings: 3" in result.output
    assert "Max stale bar age: 30.0 min" in result.output
    assert "Provider/feed: alpaca/iex" in result.output
    assert "invalid for strategy proof and excluded from clean analytics" in result.output


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


@pytest.mark.parametrize("history", [False, True])
def test_daily_review_symbol_freshness_evidence(cli_runner, tmp_path, monkeypatch, history):
    from trades.shadow_session import ShadowSessionState, append_shadow_session_history
    monkeypatch.chdir(tmp_path)
    state = ShadowSessionState(session_id="symbols", started_at=_today_iso(), status="ENDED", counters={
        "data_quality": "PARTIAL_DATA", "stale_warning_count": 1, "stale_data_events": [
            {"freshness_snapshot": {"classification": "SYMBOL_LOCAL_STALE", "stale_symbols": ["ARM"], "max_bar_age_minutes": 20}},
        ], "missing_bar_observations": 0,
    })
    state.save()
    if history:
        append_shadow_session_history(state)
    result = cli_runner.invoke(app, ["daily-review", "--no-save"])
    assert result.exit_code == 0, result.output
    assert "1/1 stale events with symbol details" in result.output
    assert "SYMBOL_LOCAL_STALE=1" in result.output
    assert "Most frequently stale: ARM=1" in result.output


def test_daily_review_empty_logs_runs(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "Daily Review" in result.stdout
    assert "Setup Alerts Today" in result.stdout
    assert "Count: 0" in result.stdout


def test_daily_review_setup_alert_counts_shown(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(
        Path("logs/setup_alerts.jsonl"),
        [
            {"timestamp": _today_iso(9), "ticker": "NVDA", "setup_type": "vwap_reclaim_long", "rank": 2},
            {"timestamp": _today_iso(10), "ticker": "NVDA", "setup_type": "vwap_reclaim_long", "rank": 1},
            {"timestamp": _today_iso(11), "ticker": "TSLA", "setup_type": "opening_range_breakout_long", "rank": 3},
        ],
    )

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "Setup Alerts Today" in result.stdout
    assert "Count: 3" in result.stdout
    assert "NVDA=2" in result.stdout
    assert "vwap_reclaim_long=2" in result.stdout
    assert "Top ranked alert: NVDA vwap_reclaim_long rank=1" in result.stdout


def test_daily_review_trades_taken_today_shown(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(
        ManualTrade(
            trade_id="t-open",
            ticker="TSLA",
            setup_type="high_relative_volume_continuation_long",
            entry_price=300.0,
            stop_loss=295.0,
            target_1=310.0,
            target_2=315.0,
            shares=5,
            entry_time=_today_iso(10),
            test_type="forward_test",
            thesis="strong relative volume",
        )
    )

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "Trades Taken Today" in result.stdout
    assert "TSLA" in result.stdout
    assert "test_type=forward_test" in result.stdout
    assert "thesis=strong relative volume" in result.stdout


def test_daily_review_closed_trade_pnl_shown(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(
        ManualTrade(
            trade_id="t-closed",
            ticker="AAPL",
            setup_type="vwap_reclaim_long",
            entry_price=100.0,
            stop_loss=98.0,
            target_1=104.0,
            target_2=106.0,
            shares=10,
            entry_time=_today_iso(9),
            status="CLOSED",
            exit_price=103.0,
            exit_time=_today_iso(15),
            realized_pnl=30.0,
            notes="target exit",
        )
    )

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "Closed Trades Today" in result.stdout
    assert "AAPL" in result.stdout
    assert "realized P&L +30.00" in result.stdout
    assert "Total P&L today: +30.00" in result.stdout


def test_daily_review_monitor_alert_counts_shown(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(
        Path("logs/trade_monitor.jsonl"),
        [
            {"timestamp": _today_iso(10), "ticker": "NVDA", "action": "WATCH", "reason": "above 1R"},
            {"timestamp": _today_iso(11), "ticker": "NVDA", "action": "TARGET_1_HIT", "reason": "target reached"},
            {"timestamp": _today_iso(12), "ticker": "TSLA", "action": "HOLD", "reason": "between levels"},
        ],
    )

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "Monitor Alerts Today" in result.stdout
    assert "TARGET_1_HIT=1" in result.stdout
    assert "WATCH=1" in result.stdout
    assert "NVDA: TARGET_1_HIT - target reached" in result.stdout


def test_daily_review_excluded_trades_omitted_from_performance(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker()
    tracker.add_trade(
        ManualTrade(
            trade_id="included",
            ticker="MSFT",
            setup_type="vwap_reclaim_long",
            entry_price=100.0,
            stop_loss=98.0,
            target_1=104.0,
            target_2=106.0,
            shares=10,
            entry_time=_today_iso(9),
            status="CLOSED",
            exit_price=105.0,
            exit_time=_today_iso(15),
            realized_pnl=50.0,
        )
    )
    tracker.add_trade(
        ManualTrade(
            trade_id="excluded",
            ticker="GOOGL",
            setup_type="workflow_test",
            entry_price=100.0,
            stop_loss=98.0,
            target_1=104.0,
            target_2=106.0,
            shares=10,
            entry_time=_today_iso(9),
            status="CLOSED",
            exit_price=200.0,
            exit_time=_today_iso(15),
            realized_pnl=1000.0,
            exclude_from_analytics=True,
            exclusion_reason="workflow test",
        )
    )

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "Total P&L today: +50.00" in result.stdout
    assert "Win rate today: 100.0%" in result.stdout
    assert "Expectancy today: +50.00" in result.stdout
    assert "Excluded trades: 1" in result.stdout


def test_daily_review_includes_setup_cooldown_section(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker()
    for idx, ticker in enumerate(["TSLA", "NVDA", "TSLA"]):
        tracker.add_trade(
            ManualTrade(
                trade_id=f"cooldown-{idx}",
                ticker=ticker,
                setup_type="vwap_reclaim_long",
                entry_price=100.0,
                stop_loss=98.0,
                target_1=104.0,
                target_2=106.0,
                shares=1,
                entry_time=_today_iso(9),
                status="CLOSED",
                exit_price=99.0,
                exit_time=_today_iso(10 + idx),
                realized_pnl=-1.0,
                exclude_from_analytics=(idx == 2),
                exclusion_reason="logging issue" if idx == 2 else "",
            )
        )

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "Setup Cooldowns" in result.stdout
    assert "vwap_reclaim_long: cooled down due to 3 failed/excluded trades" in result.stdout
    assert "TSLA: cooled down due to 2 failed/excluded trades" in result.stdout


def test_daily_review_no_secrets_printed(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(
        Path("logs/setup_alerts.jsonl"),
        [
            {
                "timestamp": _today_iso(10),
                "ticker": "SPY",
                "setup_type": "vwap_reclaim_long",
                "metadata": {"api_key": "SHOULD_NOT_PRINT", "secret": "SHOULD_NOT_PRINT"},
            }
        ],
    )

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    output_lower = result.stdout.lower()
    assert result.exit_code == 0, result.stdout
    assert "should_not_print" not in output_lower
    assert "api_key" not in output_lower
    assert "secret" not in output_lower


def test_daily_review_saves_markdown_file(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = cli_runner.invoke(app, ["daily-review"])

    assert result.exit_code == 0, result.stdout
    today = datetime.now().date().isoformat()
    review_file = tmp_path / "logs" / "daily_reviews" / f"{today}.md"
    assert review_file.exists()
    content = review_file.read_text(encoding="utf-8")
    assert "# Daily Review" in content
    assert "## Setup Alerts Today" in content
    assert f"Saved daily review to: logs\\daily_reviews\\{today}.md" in result.stdout or f"Saved daily review to: logs/daily_reviews/{today}.md" in result.stdout


def test_daily_review_no_save_does_not_write_file(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "Daily Review" in result.stdout
    assert not (tmp_path / "logs" / "daily_reviews").exists()


def test_daily_review_date_writes_matching_date_file(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = cli_runner.invoke(app, ["daily-review", "--date", "2026-06-10"])

    assert result.exit_code == 0, result.stdout
    review_file = tmp_path / "logs" / "daily_reviews" / "2026-06-10.md"
    assert review_file.exists()
    assert "2026-06-10" in review_file.read_text(encoding="utf-8")
    assert "2026-06-10" in result.stdout


def test_daily_review_missing_logs_saved_gracefully(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = cli_runner.invoke(app, ["daily-review", "--date", "2026-06-10"])

    assert result.exit_code == 0, result.stdout
    assert "Count: 0" in result.stdout
    assert (tmp_path / "logs" / "daily_reviews" / "2026-06-10.md").exists()


def test_daily_review_no_secrets_saved(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(
        Path("logs/setup_alerts.jsonl"),
        [
            {
                "timestamp": _today_iso(10),
                "ticker": "SPY",
                "setup_type": "vwap_reclaim_long",
                "metadata": {"api_key": "SHOULD_NOT_SAVE", "secret": "SHOULD_NOT_SAVE"},
            }
        ],
    )

    result = cli_runner.invoke(app, ["daily-review"])

    assert result.exit_code == 0, result.stdout
    today = datetime.now().date().isoformat()
    content = (tmp_path / "logs" / "daily_reviews" / f"{today}.md").read_text(encoding="utf-8").lower()
    assert "should_not_save" not in content
    assert "api_key" not in content
    assert "secret" not in content


def test_daily_review_discord_sends_when_configured(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
        with patch("alerts.discord_alerts.send_discord_message", return_value=True) as send_mock:
            result = cli_runner.invoke(app, ["daily-review", "--discord", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "Discord daily review sent." in result.stdout
    sent_message = send_mock.call_args.args[0]
    assert "DAILY REVIEW" in sent_message
    assert sent_message.count("DAILY REVIEW") == 1
    assert "# Daily Review" not in sent_message
    assert "Scoreboard" in sent_message
    assert "Date:" in sent_message


def test_daily_review_discord_missing_webhook_does_not_crash(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with patch("alerts.discord_alerts.is_discord_configured", return_value=False):
        result = cli_runner.invoke(app, ["daily-review", "--discord", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "Discord webhook not configured; daily review not sent." in result.stdout
    assert "Daily Review" in result.stdout


def test_daily_review_discord_failure_does_not_crash(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
        with patch("alerts.discord_alerts.send_discord_message", return_value=False):
            result = cli_runner.invoke(app, ["daily-review", "--discord", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "Discord daily review failed to send." in result.stdout
    assert "Daily Review" in result.stdout


def test_daily_review_discord_long_review_is_compact(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(
        Path("logs/setup_alerts.jsonl"),
        [
            {
                "timestamp": _today_iso(10),
                "ticker": f"TICKER{i}",
                "setup_type": "vwap_reclaim_long",
            }
            for i in range(300)
        ],
    )

    with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
        with patch("alerts.discord_alerts.send_discord_message", return_value=True) as send_mock:
            result = cli_runner.invoke(app, ["daily-review", "--discord", "--no-save"])

    assert result.exit_code == 0, result.stdout
    sent_message = send_mock.call_args.args[0]
    assert len(sent_message) <= 1900
    assert "Scoreboard" in sent_message
    assert "Full review saved locally." in sent_message
    assert "TICKER299" not in sent_message


def test_daily_review_discord_never_prints_webhook_url(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    webhook_url = "https://discord.com/api/webhooks/123/SECRET_TOKEN"

    with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
        with patch("alerts.discord_alerts.send_discord_message", return_value=True):
            result = cli_runner.invoke(app, ["daily-review", "--discord", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert webhook_url not in result.stdout
    assert "SECRET_TOKEN" not in result.stdout


def test_daily_review_discord_local_save_still_works(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
        with patch("alerts.discord_alerts.send_discord_message", return_value=True):
            result = cli_runner.invoke(app, ["daily-review", "--discord", "--date", "2026-06-10"])

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "logs" / "daily_reviews" / "2026-06-10.md").exists()
    assert "Discord daily review sent." in result.stdout


def test_daily_review_discord_preserves_notes_bullets(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
        with patch("alerts.discord_alerts.send_discord_message", return_value=True) as send_mock:
            result = cli_runner.invoke(app, ["daily-review", "--discord", "--no-save"])

    assert result.exit_code == 0, result.stdout
    sent_message = send_mock.call_args.args[0]
    assert "Key Notes" in sent_message
    assert "- Full review saved locally." in sent_message
    assert "Daily Review" in result.stdout


def test_daily_review_discord_compact_scoreboard_from_real_logs(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(
        Path("logs/setup_alerts.jsonl"),
        [
            {"timestamp": _today_iso(9), "ticker": "SPY", "setup_type": "vwap_reclaim_long", "rank": 1},
            {"timestamp": _today_iso(10), "ticker": "NVDA", "setup_type": "high_relative_volume_continuation_long", "rank": 2},
            {"timestamp": _today_iso(11), "ticker": "NVDA", "setup_type": "high_relative_volume_continuation_long", "rank": 3},
            {"timestamp": _today_iso(12), "ticker": "AMZN", "setup_type": "vwap_reclaim_long", "rank": 4},
        ],
    )
    _write_jsonl(
        Path("logs/trade_monitor.jsonl"),
        [
            {"timestamp": _today_iso(10), "ticker": "SPY", "action": "EXIT_VWAP_LOST", "reason": "below vwap"}
            for _ in range(3)
        ] + [
            {"timestamp": _today_iso(11), "ticker": "TSLA", "action": "WATCH", "reason": "above 1R"}
        ],
    )
    tracker = TradeTracker()
    tracker.add_trade(
        ManualTrade(
            trade_id="included-1",
            ticker="SPY",
            setup_type="vwap_reclaim_long",
            entry_price=100.0,
            stop_loss=98.0,
            target_1=104.0,
            target_2=106.0,
            shares=1,
            entry_time=_today_iso(9),
            status="CLOSED",
            exit_price=98.19,
            exit_time=_today_iso(14),
            realized_pnl=-1.81,
            notes="long detailed local-only notes should not dominate discord output",
        )
    )
    tracker.add_trade(
        ManualTrade(
            trade_id="excluded-1",
            ticker="TSLA",
            setup_type="vwap_reclaim_long",
            entry_price=100.0,
            stop_loss=98.0,
            target_1=104.0,
            target_2=106.0,
            shares=1,
            entry_time=_today_iso(9),
            status="CLOSED",
            exit_price=99.0,
            exit_time=_today_iso(14),
            realized_pnl=-1.0,
            exclude_from_analytics=True,
        )
    )

    with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
        with patch("alerts.discord_alerts.send_discord_message", return_value=True) as send_mock:
            result = cli_runner.invoke(app, ["daily-review", "--discord"])

    assert result.exit_code == 0, result.stdout
    sent_message = send_mock.call_args.args[0]
    assert "Scoreboard" in sent_message
    assert "Clean P&L: -1.81" in sent_message
    assert "Included Trades: 1" in sent_message
    assert "Excluded Trades: 1" in sent_message
    assert "Included (clean scoreboard):" in sent_message
    assert "Excluded (not in clean scoreboard):" in sent_message
    assert "[EXCLUDED]" in sent_message
    assert "Monitor Alerts:" in sent_message
    assert "EXIT_VWAP_LOST=3" in sent_message
    assert "WATCH=1" in sent_message
    assert "Alerts: 4" in sent_message
    assert "NVDA=2" in sent_message
    assert len(sent_message) <= 1900
    today = datetime.now().date().isoformat()
    local_content = (tmp_path / "logs" / "daily_reviews" / f"{today}.md").read_text(encoding="utf-8")
    assert "long detailed local-only notes should not dominate discord output" in local_content
    assert "long detailed local-only notes should not dominate discord output" not in sent_message


def test_daily_review_discord_separates_included_and_excluded_trades(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker()
    tracker.add_trade(
        ManualTrade(
            trade_id="included",
            ticker="SPY",
            setup_type="vwap_reclaim_long",
            entry_price=100.0,
            stop_loss=98.0,
            target_1=104.0,
            target_2=106.0,
            shares=1,
            entry_time=_today_iso(9),
            status="CLOSED",
            exit_price=98.19,
            exit_time=_today_iso(14),
            realized_pnl=-1.81,
            notes="included local detail",
        )
    )
    tracker.add_trade(
        ManualTrade(
            trade_id="excluded",
            ticker="TSLA",
            setup_type="vwap_reclaim_long",
            entry_price=100.0,
            stop_loss=98.0,
            target_1=104.0,
            target_2=106.0,
            shares=1,
            entry_time=_today_iso(9),
            status="CLOSED",
            exit_price=50.0,
            exit_time=_today_iso(14),
            realized_pnl=-50.0,
            exclude_from_analytics=True,
            exclusion_reason="bad fill",
            notes="excluded local detail",
        )
    )

    with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
        with patch("alerts.discord_alerts.send_discord_message", return_value=True) as send_mock:
            result = cli_runner.invoke(app, ["daily-review", "--discord"])

    assert result.exit_code == 0, result.stdout
    sent_message = send_mock.call_args.args[0]
    assert "Clean P&L: -1.81" in sent_message
    assert "Included (clean scoreboard):" in sent_message
    assert "Excluded (not in clean scoreboard):" in sent_message
    assert "[EXCLUDED]" in sent_message
    assert "TSLA realized P&L -50.00" in sent_message
    assert len(sent_message) <= 1900
    today = datetime.now().date().isoformat()
    local_content = (tmp_path / "logs" / "daily_reviews" / f"{today}.md").read_text(encoding="utf-8")
    assert "included local detail" in local_content
    assert "excluded local detail" in local_content
    assert "included local detail" not in sent_message
    assert "excluded local detail" not in sent_message


# ---------------------------------------------------------------------------
# Phase 4: Shadow Forward Test section
# ---------------------------------------------------------------------------

def test_daily_review_includes_shadow_section_when_logs_exist(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from trades.shadow_session import create_shadow_trade_from_setup  # noqa: F401 (shadow module import sanity)

    shadow_tracker = TradeTracker(storage_path=Path("logs/shadow_trades.jsonl"))
    trade = ManualTrade(
        ticker="SPY",
        setup_type="vwap_reclaim_long",
        entry_price=100.0,
        stop_loss=98.0,
        target_1=104.0,
        target_2=106.0,
        shares=10.0,
        entry_time=_today_iso(10),
        status="CLOSED",
        exit_price=101.2,
        exit_time=_today_iso(11),
        realized_pnl=12.0,
        test_type="shadow_test",
        metadata={"exit_signal": "TARGET_1_HIT", "session_id": "shadow-X"},
    )
    shadow_tracker.add_trade(trade)
    _write_jsonl(
        Path("logs/shadow_decisions.jsonl"),
        [
            {"timestamp": _today_iso(10), "decision": "ENTERED", "skip_reasons": []},
            {"timestamp": _today_iso(10), "decision": "SKIPPED",
             "skip_reasons": ["LOW_CONFIDENCE"]},
        ],
    )

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "## Shadow Forward Test" in result.stdout
    assert "Entered: 1" in result.stdout
    assert "Closed: 1" in result.stdout
    assert "Open: 0" in result.stdout
    assert "Skipped: 1" in result.stdout
    assert "Sim P&L: +12.00" in result.stdout
    assert "Win Rate: 100.0%" in result.stdout
    assert "Top skip reasons: LOW_CONFIDENCE=1" in result.stdout
    assert "Simulation only" in result.stdout


def test_daily_review_includes_shadow_section_when_only_state_exists(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    state_path = Path("state/shadow_session.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "session_id": "shadow-20260615-114419",
        "started_at": _today_iso(11),
        "status": "ENDED",
        "config": {"provider": "alpaca", "timeframe": "5Min"},
        "open_trade_ids": [],
        "last_iteration_at": _today_iso(12),
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

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "## Shadow Forward Test" in result.stdout
    assert "Session: shadow-20260615-114419 [ENDED]" in result.stdout
    assert "Iterations: 12" in result.stdout
    assert "No-ranked-setup iterations: 12" in result.stdout
    assert "Ranked-setup iterations: 0" in result.stdout
    assert "Evaluated setups: 0" in result.stdout
    assert "Entered: 0" in result.stdout
    assert "Skipped: 0" in result.stdout
    assert "Closed: 0" in result.stdout
    assert "Open: 0" in result.stdout
    assert "Sim P&L: +0.00" in result.stdout
    assert "Result: No ranked setups appeared during the session." in result.stdout
    assert "Simulation only - no real orders placed." in result.stdout


def test_daily_review_aggregates_multiple_shadow_sessions_for_review_date(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    yesterday = datetime.now().replace(day=max(1, datetime.now().day - 1))
    _write_jsonl(
        Path("logs/shadow_sessions.jsonl"),
        [
            {
                "timestamp": _today_iso(9),
                "session_id": "shadow-A",
                "status": "ENDED",
                "counters": {
                    "iterations_total": 12,
                    "no_ranked_setup_iterations": 12,
                    "ranked_setup_iterations": 0,
                    "evaluated_setups": 0,
                    "data_gap_count": 1,
                    "largest_data_gap_seconds": 4200.0,
                    "data_quality": "PARTIAL_DATA",
                    "stale_warning_count": 2,
                    "max_stale_bar_age_seconds": 1500.0,
                    "first_stale_timestamp": _today_iso(9),
                    "last_stale_timestamp": _today_iso(10),
                    "stale_data_caused_partial": True,
                    "entered": 0,
                    "skipped": 0,
                    "closed": 0,
                    "open": 0,
                    "realized_pnl": 0.0,
                },
            },
            {
                "timestamp": _today_iso(10),
                "session_id": "shadow-B",
                "status": "INTERRUPTED",
                "counters": {
                    "iterations_total": 1,
                    "no_ranked_setup_iterations": 0,
                    "ranked_setup_iterations": 1,
                    "evaluated_setups": 2,
                    "entered": 1,
                    "skipped": 2,
                    "closed": 0,
                    "open": 1,
                    "realized_pnl": 0.0,
                },
            },
        ],
    )
    _write_jsonl(
        Path("logs/shadow_decisions.jsonl"),
        [
            {"timestamp": _today_iso(10), "session_id": "shadow-B", "decision": "SKIPPED",
             "skip_reasons": ["LOW_CONFIDENCE"]},
            {"timestamp": _today_iso(10), "session_id": "shadow-B", "decision": "SKIPPED",
             "skip_reasons": ["LOW_CONFIDENCE"]},
            {"timestamp": _today_iso(10), "session_id": "shadow-OUTSIDE", "decision": "SKIPPED",
             "skip_reasons": ["OUTSIDE_REASON"]},
            {"timestamp": yesterday.isoformat(), "decision": "SKIPPED", "skip_reasons": ["OLD_REASON"]},
        ],
    )
    state_path = Path("state/shadow_session.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "session_id": "shadow-LATEST",
        "started_at": _today_iso(11),
        "status": "ENDED",
        "open_trade_ids": [],
        "last_iteration_at": _today_iso(11),
        "counters": {
            "iterations_total": 1,
            "no_ranked_setup_iterations": 1,
            "ranked_setup_iterations": 0,
            "evaluated_setups": 0,
            "entered": 0,
            "skipped": 0,
            "closed": 0,
            "open": 0,
        },
    }), encoding="utf-8")

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "Sessions: 2" in result.stdout
    assert "Session statuses: ENDED=1, INTERRUPTED=1" in result.stdout
    assert "Iterations: 13" in result.stdout
    assert "No-ranked-setup iterations: 12" in result.stdout
    assert "Ranked-setup iterations: 1" in result.stdout
    assert "Evaluated setups: 2" in result.stdout
    assert "Data quality: PARTIAL_DATA" in result.stdout
    assert "Data gaps: 1" in result.stdout
    assert "Largest data gap: 70.0 min" in result.stdout
    assert "Stale warnings: 2" in result.stdout
    assert "Max stale bar age: 25.0 min" in result.stdout
    assert "First stale timestamp:" in result.stdout
    assert "Last stale timestamp:" in result.stdout
    assert "Stale data caused PARTIAL_DATA: yes" in result.stdout
    assert "Entered: 1" in result.stdout
    assert "Skipped: 2" in result.stdout
    assert "Open: 1" in result.stdout
    assert "Top skip reasons: LOW_CONFIDENCE=2" in result.stdout
    assert "OUTSIDE_REASON" not in result.stdout
    assert "OLD_REASON" not in result.stdout
    assert "shadow-LATEST" not in result.stdout


def test_daily_review_shadow_one_clean_session_stays_clean(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(
        Path("logs/shadow_sessions.jsonl"),
        [{
            "timestamp": _today_iso(10),
            "session_id": "shadow-CLEAN",
            "status": "ENDED",
            "counters": {
                "iterations_total": 20,
                "evaluated_setups": 5,
                "entered": 1,
                "skipped": 4,
                "closed": 1,
                "open": 0,
                "realized_pnl": 12.5,
                "data_quality": "CLEAN",
                "data_gap_count": 0,
            },
        }],
    )

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "Data quality: CLEAN" in result.stdout
    assert "CLEAN sessions: 1" in result.stdout
    assert "PARTIAL_DATA sessions: 0" in result.stdout
    assert "Clean Sim P&L: +12.50" in result.stdout
    assert "PARTIAL_DATA stats:" not in result.stdout


def test_daily_review_shadow_one_partial_session_says_partial(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(
        Path("logs/shadow_sessions.jsonl"),
        [{
            "timestamp": _today_iso(10),
            "session_id": "shadow-PARTIAL",
            "status": "ENDED",
            "counters": {
                "iterations_total": 10,
                "evaluated_setups": 0,
                "entered": 0,
                "skipped": 0,
                "closed": 0,
                "open": 0,
                "realized_pnl": 0.0,
                "data_quality": "PARTIAL_DATA",
                "data_gap_count": 1,
                "largest_data_gap_seconds": 3600.0,
            },
        }],
    )

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "Data quality: PARTIAL_DATA" in result.stdout
    assert "CLEAN sessions: 0" in result.stdout
    assert "PARTIAL_DATA sessions: 1" in result.stdout
    assert "Partial Sim P&L: +0.00" in result.stdout
    assert "PARTIAL_DATA sessions are invalid for strategy proof" in result.stdout
    assert "CLEAN-only stats:" not in result.stdout


def test_daily_review_shadow_mixed_clean_partial_shows_clean_only_stats(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(
        Path("logs/shadow_sessions.jsonl"),
        [
            {
                "timestamp": _today_iso(9),
                "session_id": "shadow-PARTIAL",
                "status": "ENDED",
                "counters": {
                    "iterations_total": 5,
                    "evaluated_setups": 0,
                    "entered": 0,
                    "skipped": 0,
                    "closed": 0,
                    "open": 0,
                    "realized_pnl": 0.0,
                    "data_quality": "PARTIAL_DATA",
                    "data_gap_count": 1,
                    "largest_data_gap_seconds": 7200.0,
                },
            },
            {
                "timestamp": _today_iso(11),
                "session_id": "shadow-CLEAN",
                "status": "ENDED",
                "counters": {
                    "iterations_total": 75,
                    "evaluated_setups": 69,
                    "entered": 3,
                    "skipped": 66,
                    "closed": 3,
                    "open": 0,
                    "realized_pnl": 64.51,
                    "data_quality": "CLEAN",
                    "data_gap_count": 0,
                },
            },
        ],
    )
    _write_jsonl(
        Path("logs/shadow_decisions.jsonl"),
        [
            {"timestamp": _today_iso(11), "session_id": "shadow-CLEAN",
             "decision": "SKIPPED", "skip_reasons": ["LOW_CONFIDENCE"]},
            {"timestamp": _today_iso(11), "session_id": "shadow-CLEAN",
             "decision": "SKIPPED", "skip_reasons": ["LOW_CONFIDENCE"]},
        ],
    )
    shadow_tracker = TradeTracker(storage_path=Path("logs/shadow_trades.jsonl"))
    for i, pnl in enumerate([20.0, 30.0, 14.51], start=1):
        shadow_tracker.add_trade(ManualTrade(
            trade_id=f"shadow-clean-{i}",
            ticker="SPY",
            setup_type="opening_range_breakout_long",
            entry_price=100.0,
            stop_loss=98.0,
            target_1=104.0,
            target_2=106.0,
            shares=1,
            entry_time=_today_iso(11),
            status="CLOSED",
            exit_price=100.0 + pnl,
            exit_time=_today_iso(12),
            realized_pnl=pnl,
            test_type="shadow_test",
            metadata={"session_id": "shadow-CLEAN", "exit_signal": "TARGET_1_HIT"},
        ))

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "Data quality: PARTIAL_DATA" in result.stdout
    assert "CLEAN sessions: 1" in result.stdout
    assert "PARTIAL_DATA sessions: 1" in result.stdout
    assert "Iterations: 80" in result.stdout
    assert "Evaluated setups: 69" in result.stdout
    assert "Sim P&L: +64.51" in result.stdout
    assert "CLEAN-only stats:" in result.stdout
    assert "Clean iterations: 75" in result.stdout
    assert "Clean evaluated setups: 69" in result.stdout
    assert "Clean entered: 3" in result.stdout
    assert "Clean skipped: 66" in result.stdout
    assert "Clean closed: 3" in result.stdout
    assert "Clean Sim P&L: +64.51" in result.stdout
    assert "Clean Win Rate: 100.0%" in result.stdout
    assert "Clean top skip reasons: LOW_CONFIDENCE=2" in result.stdout
    assert "Partial Sim P&L: +0.00" in result.stdout


def test_daily_review_state_fallback_filters_decisions_to_latest_session(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    state_path = Path("state/shadow_session.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "session_id": "shadow-20260615-152055",
        "started_at": _today_iso(15),
        "status": "ENDED",
        "open_trade_ids": [],
        "last_iteration_at": _today_iso(15),
        "counters": {
            "iterations_total": 1,
            "no_ranked_setup_iterations": 1,
            "ranked_setup_iterations": 0,
            "evaluated_setups": 0,
            "entered": 0,
            "skipped": 0,
            "closed": 0,
            "open": 0,
        },
    }), encoding="utf-8")
    _write_jsonl(
        Path("logs/shadow_decisions.jsonl"),
        [
            {"timestamp": _today_iso(15), "session_id": "shadow-20260615-150456",
             "decision": "SKIPPED", "skip_reasons": ["LOW_CONFIDENCE"]},
            {"timestamp": _today_iso(15), "session_id": "shadow-20260615-150456",
             "decision": "SKIPPED", "skip_reasons": ["LOW_CONFIDENCE"]},
        ],
    )

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "Session: shadow-20260615-152055 [ENDED]" in result.stdout
    assert "Evaluated setups: 0" in result.stdout
    assert "Skipped: 0" in result.stdout
    assert "Top skip reasons: none" in result.stdout
    assert "LOW_CONFIDENCE" not in result.stdout


def test_daily_review_log_only_all_zero_is_clear(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("logs/shadow_decisions.jsonl").write_text("", encoding="utf-8")

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "## Shadow Forward Test" in result.stdout
    assert "No shadow session was run today." in result.stdout
    assert "Simulation only - no real orders placed." in result.stdout
    assert "Shadow activity recorded" not in result.stdout
    assert "Session: unknown" not in result.stdout


def test_daily_review_decisions_only_logs_still_show_skip_reasons(
    cli_runner, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(
        Path("logs/shadow_decisions.jsonl"),
        [
            {"timestamp": _today_iso(10), "decision": "SKIPPED",
             "skip_reasons": ["LOW_CONFIDENCE"]},
        ],
    )

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "## Shadow Forward Test" in result.stdout
    assert "Skipped: 1" in result.stdout
    assert "Top skip reasons: LOW_CONFIDENCE=1" in result.stdout
    assert "No shadow session was run today." not in result.stdout


def test_daily_review_omits_shadow_section_without_logs(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = cli_runner.invoke(app, ["daily-review", "--no-save"])

    assert result.exit_code == 0, result.stdout
    assert "Shadow Forward Test" not in result.stdout
