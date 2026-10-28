"""Tests for optional Discord webhook alerts."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest
import typer.testing

from alerts.discord_alerts import (
    format_daily_review_discord_message,
    format_setup_discord_alert,
    format_trade_monitor_discord_alert,
    get_last_discord_error,
    is_discord_configured,
    send_discord_message,
)
from main import app
from strategy.day_trade_setups import DayTradeSetup, SetupType
from trades.trade_monitor import ACTION_GUIDANCE, TradeMonitorAction, TradeMonitorResult


@pytest.fixture
def cli_runner():
    return typer.testing.CliRunner()


def _setup() -> DayTradeSetup:
    return DayTradeSetup(
        ticker="NVDA",
        timestamp=datetime.now(),
        setup_type=SetupType.OPENING_RANGE_BREAKOUT_LONG,
        is_valid=True,
        confidence=0.72,
        entry_trigger=100.0,
        stop_loss=98.0,
        target_1=104.0,
        target_2=106.0,
        risk_reward_1=2.0,
        risk_reward_2=3.0,
        reason="Opening range breakout confirmed.",
        warnings=[],
        metadata={"rank": 1},
    )


def _monitor_result() -> TradeMonitorResult:
    return TradeMonitorResult(
        trade_id="trade-1",
        ticker="NVDA",
        action=TradeMonitorAction.TARGET_1_HIT,
        current_price=104.0,
        entry_price=100.0,
        stop_loss=98.0,
        target_1=104.0,
        target_2=106.0,
        shares=10,
        unrealized_pnl=40.0,
        r_multiple=2.0,
        reason="Target 1 touched.",
        guidance=ACTION_GUIDANCE[TradeMonitorAction.TARGET_1_HIT],
        warnings=[],
        metadata={},
    )


class _FakeHttpResponse:
    def __init__(self, status: int):
        self.status = status
        self.code = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_format_setup_discord_alert():
    message = format_setup_discord_alert(_setup())

    assert "\U0001f6a8 **NEW SETUP ALERT**" in message
    assert "\u2501" * 20 in message
    assert "**Ticker:** `NVDA`" in message
    assert "**Rank:** `#1`" in message
    assert "**Setup:** `opening_range_breakout_long`" in message
    assert "**Entry:** `$100.00`" in message
    assert "**Stop:** `$98.00`" in message
    assert "**Target 1:** `$104.00`" in message
    assert "**Target 2:** `$106.00`" in message


def test_format_trade_monitor_discord_alert():
    message = format_trade_monitor_discord_alert(_monitor_result())

    assert "\u26a0\ufe0f **TRADE MONITOR ALERT**" in message
    assert "\u2501" * 20 in message
    assert "**Ticker:** `NVDA`" in message
    assert "**Action:** `TARGET_1_HIT`" in message
    assert "**Current:** `$104.00`" in message
    assert "**R Multiple:** `2.00R`" in message
    assert "**Unrealized P&L:** `$40.00`" in message
    assert "**Guidance:**" in message


def test_discord_setup_alert_stays_under_limit_with_long_reason():
    setup = _setup()
    setup.reason = "Opening range breakout.\n" + ("very long reason " * 300)

    message = format_setup_discord_alert(setup)

    assert len(message) < 2000
    assert "..." in message


def test_discord_trade_monitor_alert_stays_under_limit_with_long_reason():
    result = _monitor_result()
    result.reason = "Target touched.\n" + ("very long reason " * 300)
    result.guidance = "Take action.\n" + ("very long guidance " * 300)

    message = format_trade_monitor_discord_alert(result)

    assert len(message) < 2000
    assert "..." in message


def test_discord_formatted_messages_never_include_webhook_url():
    webhook_url = "https://discord.com/api/webhooks/123/SECRET_TOKEN"
    setup = _setup()
    setup.reason = f"Do not leak {webhook_url}"
    result = _monitor_result()
    result.reason = f"Do not leak {webhook_url}"
    result.guidance = f"Do not leak {webhook_url}"

    setup_message = format_setup_discord_alert(setup)
    monitor_message = format_trade_monitor_discord_alert(result)

    assert webhook_url not in setup_message
    assert webhook_url not in monitor_message
    assert "SECRET_TOKEN" not in setup_message
    assert "SECRET_TOKEN" not in monitor_message


def test_format_daily_review_discord_message_compacts_long_review():
    message = format_daily_review_discord_message("# Daily Review\n" + ("long line\n" * 1000))

    assert message.startswith("\U0001f4ca **DAILY REVIEW**")
    assert len(message) <= 1900
    assert "Scoreboard" in message
    assert "Full review saved locally." in message
    assert "long line" not in message


def test_format_daily_review_discord_message_redacts_webhook_url():
    webhook_url = "https://discord.com/api/webhooks/123/SECRET_TOKEN"

    message = format_daily_review_discord_message(f"# Daily Review\n{webhook_url}")

    assert webhook_url not in message
    assert "SECRET_TOKEN" not in message


def test_format_daily_review_discord_message_removes_duplicate_title():
    message = format_daily_review_discord_message("# Daily Review\n\n## Date Reviewed\n  2026-06-11")

    assert message.count("DAILY REVIEW") == 1
    assert "# Daily Review" not in message
    assert "Date: 2026-06-11" in message


def test_format_daily_review_discord_message_includes_compact_notes():
    message = format_daily_review_discord_message(
        "# Daily Review\n\n"
        "## Notes / Next Questions\n"
        "- Which setup looked best?\n"
        "- Did VWAP-loss exits help?"
    )

    assert "Key Notes" in message
    assert "- Full review saved locally." in message


def test_format_daily_review_discord_message_includes_scoreboard_counts():
    message = format_daily_review_discord_message(
        "# Daily Review\n\n"
        "## Date Reviewed\n  2026-06-12\n\n"
        "## Setup Alerts Today\n"
        "  Count: 4\n"
        "  By ticker: AMZN=1, NVDA=2, SPY=1\n"
        "  By setup_type: high_relative_volume_continuation_long=2, vwap_reclaim_long=2\n\n"
        "## Closed Trades Today\n"
        "  Count: 5\n"
        "  SPY      realized P&L -1.81 notes=full verbose note that should stay compact\n\n"
        "## Monitor Alerts Today\n"
        "  Counts: EXIT_VWAP_LOST=15, WATCH=1\n\n"
        "## Performance Summary\n"
        "  Total P&L today: -2.81\n"
        "  Win rate today: 0.0%\n"
        "  Expectancy today: -1.40\n"
        "  Excluded trades: 3\n"
    )

    assert "Scoreboard" in message
    assert "Clean P&L: -2.81" in message
    assert "Win Rate: 0.0%" in message
    assert "Included Trades: 2" in message
    assert "Excluded Trades: 3" in message
    assert "Monitor Alerts: EXIT_VWAP_LOST=15, WATCH=1" in message
    assert "By ticker: AMZN=1, NVDA=2, SPY=1" in message
    assert "By setup: high_relative_volume_continuation_long=2, vwap_reclaim_long=2" in message
    assert "Included (clean scoreboard):" in message
    assert "Excluded (not in clean scoreboard):" in message
    assert "[EXCLUDED]" in message
    assert len(message) <= 1900


def test_format_daily_review_discord_message_includes_shadow_clean_partial_split():
    message = format_daily_review_discord_message(
        "# Daily Review\n\n"
        "## Date Reviewed\n  2026-06-30\n\n"
        "## Shadow Forward Test\n"
        "  Sessions: 2\n"
        "  CLEAN sessions: 1\n"
        "  PARTIAL_DATA sessions: 1\n"
        "  Iterations: 80\n"
        "  Evaluated setups: 69\n"
        "  Data quality: PARTIAL_DATA\n"
        "  Data gaps: 1\n"
        "  Entered: 3\n"
        "  Sim P&L: +64.51\n"
        "  CLEAN-only stats:\n"
        "    Clean Sim P&L: +64.51\n"
        "  PARTIAL_DATA stats:\n"
        "    Partial Sim P&L: +0.00\n"
    )

    assert "Shadow: 80 iterations, 69 setups, 3 entries, sim P&L +64.51" in message
    assert "(clean sessions: +64.51, partial sessions: +0.00)" in message
    assert "data PARTIAL_DATA, gaps=1" in message
    assert len(message) <= 1900


def test_discord_no_webhook_returns_false(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "DISCORD_WEBHOOK_URL", "")

    assert is_discord_configured() is False
    assert send_discord_message("hello") is False


def test_discord_http_failure_returns_false(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "DISCORD_WEBHOOK_URL", "https://discord.example/webhook")

    with patch("urllib.request.urlopen", side_effect=OSError("network down")):
        assert send_discord_message("hello") is False


def test_discord_204_response_returns_true(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "DISCORD_WEBHOOK_URL", "https://discord.example/webhook")

    with patch("urllib.request.urlopen", return_value=_FakeHttpResponse(204)) as urlopen_mock:
        assert send_discord_message("hello") is True

    request = urlopen_mock.call_args.args[0]
    headers = {key.lower(): value for key, value in request.header_items()}
    assert headers["content-type"] == "application/json"
    assert headers["user-agent"] == "AI-Regime-Trader"
    assert request.data == b'{"content": "hello"}'


def test_discord_200_response_returns_true(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "DISCORD_WEBHOOK_URL", "https://discord.example/webhook")

    with patch("urllib.request.urlopen", return_value=_FakeHttpResponse(200)):
        assert send_discord_message("hello") is True


def test_discord_non_2xx_returns_false(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "DISCORD_WEBHOOK_URL", "https://discord.example/webhook")

    with patch("urllib.request.urlopen", return_value=_FakeHttpResponse(500)):
        assert send_discord_message("hello") is False

    assert "HTTP 500" in get_last_discord_error()


def test_discord_exception_redacts_webhook_url(monkeypatch):
    from config.settings import settings

    webhook_url = "https://discord.com/api/webhooks/123/SECRET_TOKEN"
    monkeypatch.setattr(settings, "DISCORD_WEBHOOK_URL", webhook_url)

    with patch("urllib.request.urlopen", side_effect=OSError(f"failed for {webhook_url}")):
        assert send_discord_message("hello") is False

    assert webhook_url not in get_last_discord_error()
    assert "SECRET_TOKEN" not in get_last_discord_error()


def test_discord_test_missing_webhook_handled(cli_runner):
    with patch("alerts.discord_alerts.is_discord_configured", return_value=False):
        result = cli_runner.invoke(app, ["discord-test"])

    assert result.exit_code == 0, result.stdout
    assert "Discord webhook is not configured" in result.stdout


def test_discord_test_configured_sends(cli_runner):
    with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
        with patch("alerts.discord_alerts.send_discord_message", return_value=True) as send_mock:
            result = cli_runner.invoke(app, ["discord-test"])

    assert result.exit_code == 0, result.stdout
    sent_message = send_mock.call_args.args[0]
    assert "DISCORD TEST ALERT" in sent_message
    assert "AI Regime Trader Discord alerts are configured correctly." in sent_message
    assert "sent successfully" in result.stdout


def test_discord_test_failure_does_not_crash(cli_runner):
    with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
        with patch("alerts.discord_alerts.send_discord_message", return_value=False):
            with patch("alerts.discord_alerts.get_last_discord_error", return_value="Discord returned HTTP 500."):
                result = cli_runner.invoke(app, ["discord-test"])

    assert result.exit_code == 0, result.stdout
    assert "failed" in result.stdout.lower()
    assert "Reason: Discord returned HTTP 500." in result.stdout


def test_discord_test_webhook_url_not_printed(cli_runner):
    fake_url = "https://discord.example/webhook/SECRET_TOKEN"
    with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
        with patch("alerts.discord_alerts.send_discord_message", return_value=True):
            result = cli_runner.invoke(app, ["discord-test"])

    assert result.exit_code == 0, result.stdout
    assert fake_url not in result.stdout
    assert "SECRET_TOKEN" not in result.stdout


# ---------------------------------------------------------------------------
# Phase 4: compact daily review shadow line
# ---------------------------------------------------------------------------

_SHADOW_REVIEW_TEXT = (
    "# Daily Review\n\n"
    "## Date Reviewed\n  2026-06-12\n\n"
    "## Shadow Forward Test\n"
    "  (Simulation only - no real orders placed.)\n"
    "  Iterations: 12\n"
    "  No-ranked-setup iterations: 10\n"
    "  Ranked-setup iterations: 2\n"
    "  Evaluated setups: 5\n"
    "  Entered: 2\n"
    "  Closed: 1\n"
    "  Open: 1\n"
    "  Skipped: 3\n"
    "  Sim P&L: +12.00\n"
    "  Win Rate: 100.0%\n"
    "  Top skip reasons: LOW_CONFIDENCE=2\n"
)


def test_format_daily_review_discord_message_includes_shadow_line():
    message = format_daily_review_discord_message(_SHADOW_REVIEW_TEXT)

    assert "Shadow: 12 iterations, 5 setups, 2 entries, sim P&L +12.00" in message


def test_format_daily_review_discord_message_includes_zero_setup_shadow_summary():
    message = format_daily_review_discord_message(
        "# Daily Review\n\n"
        "## Date Reviewed\n  2026-06-15\n\n"
        "## Shadow Forward Test\n"
        "  Session: shadow-20260615-114419 [ENDED]\n"
        "  Iterations: 12\n"
        "  No-ranked-setup iterations: 12\n"
        "  Ranked-setup iterations: 0\n"
        "  Evaluated setups: 0\n"
        "  Entered: 0\n"
        "  Sim P&L: +0.00\n"
    )

    assert "Shadow: 12 iterations, 0 setups, 0 entries, sim P&L +0.00" in message


def test_format_daily_review_discord_message_includes_partial_shadow_data_quality():
    message = format_daily_review_discord_message(
        "# Daily Review\n\n"
        "## Date Reviewed\n  2026-06-25\n\n"
        "## Shadow Forward Test\n"
        "  Iterations: 2\n"
        "  Evaluated setups: 0\n"
        "  Data quality: PARTIAL_DATA\n"
        "  Data gaps: 1\n"
        "  Entered: 0\n"
        "  Sim P&L: +0.00\n"
    )

    assert "Shadow: 2 iterations, 0 setups, 0 entries, sim P&L +0.00, data PARTIAL_DATA, gaps=1" in message


def test_format_daily_review_discord_message_uses_no_shadow_session_wording():
    message = format_daily_review_discord_message(
        "# Daily Review\n\n"
        "## Date Reviewed\n  2026-06-16\n\n"
        "## Shadow Forward Test\n"
        "  No shadow session was run today.\n"
        "  Simulation only - no real orders placed.\n"
    )

    assert "Shadow: no session run today" in message
    assert "Shadow: 0 iterations" not in message


def test_format_daily_review_discord_message_omits_shadow_line_without_section():
    message = format_daily_review_discord_message(
        "# Daily Review\n\n## Date Reviewed\n  2026-06-12\n"
    )

    assert "Shadow:" not in message
