"""Tests for the read-only daily-plan CLI command."""
from __future__ import annotations

import json
from datetime import datetime

import typer.testing

from main import app
from trades.trade_tracker import ManualTrade, TradeStatus, TradeTracker


def _write_jsonl(path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_daily_plan_runs_with_empty_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = typer.testing.CliRunner().invoke(app, ["daily-plan"])

    assert result.exit_code == 0
    assert "Daily Plan" in result.output
    assert "Recent Setup Alerts" in result.output


def test_daily_plan_prints_safety_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = typer.testing.CliRunner().invoke(app, ["daily-plan"])

    assert result.exit_code == 0
    assert "Safety Status" in result.output
    assert "PAPER_TRADING" in result.output
    assert "LIVE_TRADING_ALLOWED" in result.output
    assert "ALPACA_PAPER" in result.output
    assert "Risk lock exists" in result.output


def test_daily_plan_lists_watchlists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    watchlists_dir = tmp_path / "watchlists"
    watchlists_dir.mkdir()
    (watchlists_dir / "test_list.json").write_text(
        json.dumps({"name": "test_list", "tickers": ["SPY", "QQQ"]})
    )

    result = typer.testing.CliRunner().invoke(app, ["daily-plan"])

    assert result.exit_code == 0
    assert "Watchlists" in result.output
    assert "test_list: 2 tickers" in result.output


def test_daily_plan_displays_recent_alert_count(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(
        tmp_path / "logs" / "setup_alerts.jsonl",
        [{
            "timestamp": datetime.now().isoformat(),
            "ticker": "AAPL",
            "setup_type": "vwap_reclaim_long",
            "is_valid": True,
            "confidence": 0.72,
        }],
    )

    result = typer.testing.CliRunner().invoke(app, ["daily-plan"])

    assert result.exit_code == 0
    assert "Count in last 4 hours: 1" in result.output
    assert "AAPL vwap_reclaim_long confidence=0.72" in result.output


def test_daily_plan_displays_open_trades(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl")
    tracker.add_trade(
        ManualTrade(
            trade_id="open-1",
            ticker="MSFT",
            setup_type="opening_range_breakout_long",
            entry_price=420.0,
            stop_loss=416.0,
            target_1=428.0,
            target_2=432.0,
            shares=5,
            entry_time=datetime.now().isoformat(),
        )
    )

    result = typer.testing.CliRunner().invoke(app, ["daily-plan"])

    assert result.exit_code == 0
    assert "Open Trades" in result.output
    assert "Count: 1" in result.output
    assert "MSFT" in result.output
    assert "entry 420.00" in result.output


def test_daily_plan_displays_today_pnl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker(storage_path=tmp_path / "logs" / "manual_trades.jsonl")
    tracker.add_trade(
        ManualTrade(
            trade_id="closed-1",
            ticker="NVDA",
            setup_type="high_relative_volume_continuation_long",
            entry_price=100.0,
            stop_loss=98.0,
            target_1=104.0,
            target_2=106.0,
            shares=10,
            entry_time=datetime.now().isoformat(),
            status=TradeStatus.CLOSED.value,
            exit_price=103.0,
            exit_time=datetime.now().isoformat(),
            realized_pnl=30.0,
        )
    )

    result = typer.testing.CliRunner().invoke(app, ["daily-plan"])

    assert result.exit_code == 0
    assert "Today's Closed Trades / P&L" in result.output
    assert "Closed trades today: 1" in result.output
    assert "Realized P&L today:  +30.00" in result.output


def test_daily_plan_prints_suggested_commands(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from config.settings import settings
    monkeypatch.setattr(settings, "ALPACA_API_KEY", "")
    monkeypatch.setattr(settings, "ALPACA_SECRET_KEY", "")

    result = typer.testing.CliRunner().invoke(app, ["daily-plan"])

    assert result.exit_code == 0
    assert "Suggested Commands" in result.output
    assert "setup-scan-loop" in result.output
    assert "setup-alerts-recent" in result.output
    assert "trade-add-latest-alert" in result.output
    assert "trade-monitor" in result.output
    assert "trade-analytics" in result.output
    assert "dashboard" in result.output


def test_daily_plan_recommends_alpaca_iex_when_credentials_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from config.settings import settings
    monkeypatch.setattr(settings, "ALPACA_API_KEY", "SET")
    monkeypatch.setattr(settings, "ALPACA_SECRET_KEY", "SET")

    result = typer.testing.CliRunner().invoke(app, ["daily-plan"])

    assert result.exit_code == 0
    assert "Recommended live scan" in result.output
    assert "setup-scan-loop --watchlist mega_caps --timeframe 5Min --provider alpaca --alpaca-feed iex" in result.output
    assert "trade-monitor --provider alpaca --alpaca-feed iex --timeframe 1Min" in result.output
    assert "YFinance intraday may be delayed or stale" not in result.output
    assert "SET" not in result.output


def test_daily_plan_recommends_yfinance_with_warning_without_alpaca_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from config.settings import settings
    monkeypatch.setattr(settings, "ALPACA_API_KEY", "")
    monkeypatch.setattr(settings, "ALPACA_SECRET_KEY", "")

    result = typer.testing.CliRunner().invoke(app, ["daily-plan"])

    assert result.exit_code == 0
    assert "setup-scan-loop --watchlist mega_caps --timeframe 5Min --provider yfinance" in result.output
    assert "trade-monitor --provider yfinance --timeframe 1Min" in result.output
    assert "Note: Alpaca credentials are not configured. YFinance intraday may be delayed or stale." in result.output


def test_daily_plan_does_not_print_secrets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = typer.testing.CliRunner().invoke(app, ["daily-plan"])

    assert result.exit_code == 0
    output = result.output.lower()
    for banned in ("api_key", "secret", "password", "token"):
        assert banned not in output


def test_morning_start_prints_safety_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = typer.testing.CliRunner().invoke(app, ["morning-start"])

    assert result.exit_code == 0
    assert "Morning Start" in result.output
    assert "Safety Status" in result.output
    assert "PAPER_TRADING" in result.output
    assert "LIVE_TRADING_ALLOWED" in result.output
    assert "ALPACA_PAPER" in result.output
    assert "Risk lock exists" in result.output


def test_morning_start_recommends_alpaca_iex_when_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from config.settings import settings
    monkeypatch.setattr(settings, "ALPACA_API_KEY", "SET")
    monkeypatch.setattr(settings, "ALPACA_SECRET_KEY", "SET")
    monkeypatch.setattr(settings, "DISCORD_WEBHOOK_URL", "https://discord.example/webhook/SECRET_TOKEN")

    result = typer.testing.CliRunner().invoke(app, ["morning-start"])

    assert result.exit_code == 0
    assert "Recommended provider: Alpaca IEX" in result.output
    assert "setup-scan-loop --watchlist mega_caps --timeframe 5Min --provider alpaca --alpaca-feed iex" in result.output
    assert "trade-monitor --provider alpaca --alpaca-feed iex --timeframe 1Min" in result.output
    assert "--discord-alerts" in result.output
    assert "daily-review --discord" in result.output
    assert "SET" not in result.output
    assert "SECRET_TOKEN" not in result.output


def test_morning_start_recommends_yfinance_warning_when_alpaca_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from config.settings import settings
    monkeypatch.setattr(settings, "ALPACA_API_KEY", "")
    monkeypatch.setattr(settings, "ALPACA_SECRET_KEY", "")
    monkeypatch.setattr(settings, "DISCORD_WEBHOOK_URL", "")

    result = typer.testing.CliRunner().invoke(app, ["morning-start"])

    assert result.exit_code == 0
    assert "Recommended provider: YFinance" in result.output
    assert "YFinance intraday may be delayed or stale" in result.output
    assert "Warning: Discord webhook missing" in result.output
    assert "setup-scan-loop --watchlist mega_caps --timeframe 5Min --provider yfinance" in result.output
    assert "trade-monitor --provider yfinance --timeframe 1Min" in result.output
    assert "--discord-alerts" not in result.output
    assert "daily-review --discord" not in result.output


def test_morning_start_includes_manual_trade_reminder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = typer.testing.CliRunner().invoke(app, ["morning-start"])

    assert result.exit_code == 0
    assert "Replace TICKER and SHARES before adding a trade." in result.output
    assert 'trade-add-latest-alert --ticker TICKER --shares SHARES --notes "forward test"' in result.output


def test_morning_start_does_not_print_secrets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from config.settings import settings
    monkeypatch.setattr(settings, "ALPACA_API_KEY", "PK_SECRET_VALUE")
    monkeypatch.setattr(settings, "ALPACA_SECRET_KEY", "SUPER_SECRET_VALUE")
    monkeypatch.setattr(settings, "DISCORD_WEBHOOK_URL", "https://discord.example/webhook/SECRET_TOKEN")

    result = typer.testing.CliRunner().invoke(app, ["morning-start"])

    assert result.exit_code == 0
    assert "PK_SECRET_VALUE" not in result.output
    assert "SUPER_SECRET_VALUE" not in result.output
    assert "SECRET_TOKEN" not in result.output
