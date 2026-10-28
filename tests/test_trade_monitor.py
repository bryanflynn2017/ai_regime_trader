"""
Tests for manual trade monitoring.

The monitor is advisory only:
  - It reads ManualTrade records.
  - It evaluates latest OHLCV bars.
  - It NEVER places orders or contacts broker execution.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import typer.testing

from main import app
from trades.trade_monitor import ACTION_GUIDANCE, TradeMonitor, TradeMonitorAction, TradeMonitorResult
from trades.trade_tracker import ManualTrade, TradeTracker


@pytest.fixture
def cli_runner():
    return typer.testing.CliRunner()


def _make_trade(**overrides) -> ManualTrade:
    defaults = dict(
        trade_id="trade-1",
        ticker="SPY",
        setup_type="vwap_reclaim_long",
        entry_price=100.0,
        stop_loss=98.0,
        target_1=104.0,
        target_2=106.0,
        shares=10.0,
        entry_time="2026-06-06T10:00:00",
    )
    defaults.update(overrides)
    return ManualTrade(**defaults)


def _bars(close: float, high: float | None = None, low: float | None = None, session_vwap=None) -> pd.DataFrame:
    data = {
        "open": [100.0],
        "high": [high if high is not None else close],
        "low": [low if low is not None else close],
        "close": [close],
        "volume": [1000],
    }
    if session_vwap is not None:
        data["session_vwap"] = [session_vwap]
    return pd.DataFrame(data, index=pd.DatetimeIndex(["2026-06-06 10:00:00"]))


def test_monitor_hold_below_1r():
    result = TradeMonitor().monitor_trade(_make_trade(), _bars(close=101.0, high=101.5, low=100.5))
    assert result.action == TradeMonitorAction.HOLD
    assert result.r_multiple == pytest.approx(0.5)
    assert result.guidance == ACTION_GUIDANCE[TradeMonitorAction.HOLD]


def test_monitor_watch_above_1r():
    result = TradeMonitor().monitor_trade(_make_trade(), _bars(close=102.5, high=103.0, low=102.0))
    assert result.action == TradeMonitorAction.WATCH
    assert result.r_multiple == pytest.approx(1.25)
    assert result.guidance == ACTION_GUIDANCE[TradeMonitorAction.WATCH]


def test_monitor_target_1_hit():
    result = TradeMonitor().monitor_trade(_make_trade(), _bars(close=103.0, high=104.2, low=102.0))
    assert result.action == TradeMonitorAction.TARGET_1_HIT
    assert result.guidance == ACTION_GUIDANCE[TradeMonitorAction.TARGET_1_HIT]


def test_monitor_target_2_hit():
    result = TradeMonitor().monitor_trade(_make_trade(), _bars(close=105.0, high=106.1, low=104.0))
    assert result.action == TradeMonitorAction.TARGET_2_HIT
    assert result.guidance == ACTION_GUIDANCE[TradeMonitorAction.TARGET_2_HIT]


def test_monitor_stop_hit_wins_over_target_same_bar():
    result = TradeMonitor().monitor_trade(_make_trade(), _bars(close=103.0, high=106.5, low=97.5))
    assert result.action == TradeMonitorAction.STOP_HIT
    assert result.guidance == ACTION_GUIDANCE[TradeMonitorAction.STOP_HIT]


def test_monitor_exit_vwap_lost():
    result = TradeMonitor().monitor_trade(
        _make_trade(),
        _bars(close=99.5, high=100.0, low=99.0, session_vwap=100.0),
    )
    assert result.action == TradeMonitorAction.EXIT_VWAP_LOST
    assert result.guidance == ACTION_GUIDANCE[TradeMonitorAction.EXIT_VWAP_LOST]


def test_each_monitor_action_has_expected_guidance():
    expected = {
        TradeMonitorAction.HOLD: "Hold. Trade remains between management levels.",
        TradeMonitorAction.WATCH: "Watch closely. Trade is at or above +1R; consider moving stop to breakeven.",
        TradeMonitorAction.TARGET_1_HIT: "Target 1 hit. Consider taking partial profit and/or moving stop to breakeven.",
        TradeMonitorAction.TARGET_2_HIT: "Target 2 hit. Consider closing remaining position.",
        TradeMonitorAction.STOP_HIT: "Stop hit. Trade invalidated.",
        TradeMonitorAction.EXIT_VWAP_LOST: "Session VWAP lost. Consider exiting or reducing.",
    }

    assert ACTION_GUIDANCE == expected


def test_monitor_invalid_risk_per_share_warning():
    trade = _make_trade(stop_loss=101.0)
    result = TradeMonitor().monitor_trade(trade, _bars(close=102.0, high=103.0, low=101.5))
    assert result.r_multiple == 0.0
    assert any("Invalid risk_per_share" in warning for warning in result.warnings)


def test_monitor_missing_data_warning():
    result = TradeMonitor().monitor_trade(_make_trade(), None)
    assert result.action == TradeMonitorAction.WATCH
    assert result.current_price == 0.0
    assert "No market data available." in result.warnings


def test_monitor_open_trades_handles_multiple_trades():
    trades = [_make_trade(trade_id="t1", ticker="SPY"), _make_trade(trade_id="t2", ticker="QQQ")]
    data = {
        "SPY": _bars(close=101.0, high=101.5, low=100.5),
        "QQQ": _bars(close=105.0, high=106.1, low=104.0),
    }
    results = TradeMonitor().monitor_open_trades(trades, data)
    assert len(results) == 2
    assert {result.ticker for result in results} == {"SPY", "QQQ"}


def test_trade_monitor_cli_with_synthetic_data(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker()
    tracker.add_trade(_make_trade(ticker="SPY"))

    result = cli_runner.invoke(app, ["trade-monitor", "--synthetic"])

    assert result.exit_code == 0, result.stdout
    assert "Trade Monitor" in result.stdout
    assert "SPY" in result.stdout


def test_trade_monitor_writes_jsonl(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(ticker="SYNTHETIC"))

    result = cli_runner.invoke(app, ["trade-monitor", "--synthetic"])

    assert result.exit_code == 0, result.stdout
    log_file = tmp_path / "logs" / "trade_monitor.jsonl"
    assert log_file.exists()
    records = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
    assert records
    assert records[0]["ticker"] == "SYNTHETIC"
    assert records[0]["guidance"]
    assert "ALPACA" not in log_file.read_text().upper()


def test_trade_monitor_output_includes_guidance(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(ticker="SYNTHETIC"))

    result = cli_runner.invoke(app, ["trade-monitor", "--synthetic"])

    assert result.exit_code == 0, result.stdout
    assert "Guidance:" in result.stdout


def test_trade_monitor_loop_respects_max_iterations(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(ticker="SYNTHETIC"))

    result = cli_runner.invoke(
        app,
        ["trade-monitor", "--synthetic", "--loop", "--max-iterations", "2", "--interval-seconds", "0"],
    )

    assert result.exit_code == 0, result.stdout
    assert result.stdout.count("Iteration") == 2
    records = [
        json.loads(line)
        for line in (tmp_path / "logs" / "trade_monitor.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 2


def test_trade_monitor_synthetic_loop_without_max_iterations_rejected(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = cli_runner.invoke(app, ["trade-monitor", "--synthetic", "--loop"])
    assert result.exit_code == 1
    assert "requires --max-iterations" in result.stdout


def test_trade_monitor_no_broker_or_order_executor_called(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(ticker="SPY"))

    with patch("broker.order_executor.OrderExecutor") as order_executor, patch(
        "broker.alpaca_adapter.AlpacaBrokerAdapter"
    ) as alpaca_adapter:
        result = cli_runner.invoke(app, ["trade-monitor", "--synthetic"])

    assert result.exit_code == 0, result.stdout
    order_executor.assert_not_called()
    alpaca_adapter.assert_not_called()


def test_trade_monitor_provider_alpaca_routes_through_alpaca_data_provider(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(ticker="SPY"))

    class MockAlpacaProvider:
        provider_name = "AlpacaData"
        latest_calls = []
        multiple_calls = []

        def get_latest_bar(self, ticker, timeframe="1Min"):
            self.latest_calls.append((ticker, timeframe))
            assert ticker == "SPY"
            assert timeframe == "1Min"
            df = _bars(close=101.0, high=101.5, low=100.5)
            df.attrs["latest_bar_time"] = "2026-06-06T10:00:00"
            df.attrs["provider_time"] = "2026-06-06T10:00:30"
            df.attrs["data_age_seconds"] = 30.0
            return df

        def get_multiple_bars(self, tickers, start=None, end=None, timeframe="1Min"):
            self.multiple_calls.append((tickers, start, end, timeframe))
            return {"SPY": _bars(close=101.0, high=101.5, low=100.5)}

    provider = MockAlpacaProvider()
    with patch("data_providers.AlpacaDataProvider", return_value=provider) as provider_cls, patch(
        "broker.order_executor.OrderExecutor"
    ) as order_executor:
        result = cli_runner.invoke(app, ["trade-monitor", "--provider", "alpaca"])

    assert result.exit_code == 0, result.stdout
    assert "Provider: AlpacaData" in result.stdout
    assert "Data provider: Alpaca IEX" in result.stdout
    provider_cls.assert_called_once_with(feed="iex")
    assert provider.latest_calls == [("SPY", "1Min")]
    assert provider.multiple_calls == []
    order_executor.assert_not_called()


def test_trade_monitor_explicit_yfinance_provider_is_respected(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(ticker="SPY"))

    class MockYFinanceProvider:
        provider_name = "YFinance"

        def get_multiple_bars(self, tickers, start=None, end=None, timeframe="1Min"):
            assert tickers == ["SPY"]
            assert timeframe == "1Min"
            return {"SPY": _bars(close=101.0, high=101.5, low=100.5)}

    with patch("data_providers.YFinanceDataProvider", return_value=MockYFinanceProvider()) as provider_cls, patch(
        "broker.order_executor.OrderExecutor"
    ) as order_executor:
        result = cli_runner.invoke(app, ["trade-monitor", "--provider", "yfinance"])

    assert result.exit_code == 0, result.stdout
    assert "Data provider: YFinance" in result.stdout
    provider_cls.assert_called_once()
    order_executor.assert_not_called()


def test_trade_monitor_provider_alpaca_sip_subscription_error_is_helpful(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(ticker="SPY"))

    from data_providers.base import DataProviderError

    class MockAlpacaProvider:
        provider_name = "AlpacaData"

        def get_latest_bar(self, ticker, timeframe="1Min"):
            raise DataProviderError("subscription does not permit querying recent SIP data")

    with patch("data_providers.AlpacaDataProvider", return_value=MockAlpacaProvider()), patch(
        "broker.order_executor.OrderExecutor"
    ) as order_executor:
        result = cli_runner.invoke(
            app, ["trade-monitor", "--provider", "alpaca", "--alpaca-feed", "sip"]
        )

    assert result.exit_code == 1
    assert "Your Alpaca plan does not permit SIP data" in result.stdout
    assert "--alpaca-feed iex" in result.stdout
    order_executor.assert_not_called()


def test_trade_monitor_provider_alpaca_warns_when_latest_bar_stale(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(ticker="SPY"))

    class MockAlpacaProvider:
        provider_name = "AlpacaData"

        def get_latest_bar(self, ticker, timeframe="1Min"):
            df = _bars(close=101.0, high=101.5, low=100.5)
            df.attrs["latest_bar_time"] = "2026-06-06T09:30:00"
            df.attrs["provider_time"] = "2026-06-06T10:00:00"
            df.attrs["data_age_seconds"] = 1800.0
            return df

    with patch("data_providers.AlpacaDataProvider", return_value=MockAlpacaProvider()), patch(
        "broker.order_executor.OrderExecutor"
    ) as order_executor:
        result = cli_runner.invoke(app, ["trade-monitor", "--provider", "alpaca"])

    assert result.exit_code == 0, result.stdout
    assert "Warning: Alpaca latest bar appears stale: 30.0 minutes old" in result.stdout
    order_executor.assert_not_called()


def test_trade_monitor_synthetic_skips_real_ticker_by_default(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(ticker="GOOGL"))

    result = cli_runner.invoke(app, ["trade-monitor", "--synthetic"])

    assert result.exit_code == 0, result.stdout
    assert "Warning: synthetic data is fake and may not match real tickers: GOOGL" in result.stdout
    assert "Skipped real ticker trades in synthetic mode" in result.stdout
    assert "No compatible open trades to monitor." in result.stdout
    log_file = tmp_path / "logs" / "trade_monitor.jsonl"
    assert not log_file.exists()


def test_trade_monitor_synthetic_ticker_still_monitored(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker()
    tracker.add_trade(_make_trade(ticker="GOOGL", trade_id="real"))
    tracker.add_trade(_make_trade(ticker="SYNTHETIC", trade_id="synthetic"))

    result = cli_runner.invoke(app, ["trade-monitor", "--synthetic"])

    assert result.exit_code == 0, result.stdout
    assert "Warning: synthetic data is fake" in result.stdout
    log_file = tmp_path / "logs" / "trade_monitor.jsonl"
    records = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
    assert [record["ticker"] for record in records] == ["SYNTHETIC"]


def test_trade_monitor_allow_synthetic_real_tickers_monitors_real_ticker(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(ticker="GOOGL"))

    result = cli_runner.invoke(
        app,
        ["trade-monitor", "--synthetic", "--allow-synthetic-real-tickers"],
    )

    assert result.exit_code == 0, result.stdout
    assert "Warning: synthetic data is fake" not in result.stdout
    log_file = tmp_path / "logs" / "trade_monitor.jsonl"
    records = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
    assert records[0]["ticker"] == "GOOGL"


def test_trade_monitor_all_skipped_shows_no_compatible_trades(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = TradeTracker()
    tracker.add_trade(_make_trade(ticker="GOOGL", trade_id="t1"))
    tracker.add_trade(_make_trade(ticker="AAPL", trade_id="t2"))

    result = cli_runner.invoke(app, ["trade-monitor", "--synthetic"])

    assert result.exit_code == 0, result.stdout
    assert "Warning: synthetic data is fake and may not match real tickers: AAPL, GOOGL" in result.stdout
    assert "No compatible open trades to monitor." in result.stdout


def _monitor_result(action: TradeMonitorAction, trade_id: str = "trade-1") -> TradeMonitorResult:
    return TradeMonitorResult(
        trade_id=trade_id,
        ticker="SYNTHETIC",
        action=action,
        current_price=104.0,
        entry_price=100.0,
        stop_loss=98.0,
        target_1=104.0,
        target_2=106.0,
        shares=10.0,
        unrealized_pnl=40.0,
        r_multiple=2.0,
        reason=f"{action.value}: test reason.",
        guidance=ACTION_GUIDANCE[action],
        warnings=[],
        metadata={},
    )


def test_trade_monitor_loop_alert_banner_for_target_1(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(ticker="SYNTHETIC"))

    with patch("trades.trade_monitor.TradeMonitor.monitor_open_trades") as mock_monitor:
        mock_monitor.return_value = [_monitor_result(TradeMonitorAction.TARGET_1_HIT)]
        result = cli_runner.invoke(
            app,
            ["trade-monitor", "--synthetic", "--loop", "--max-iterations", "1", "--interval-seconds", "0"],
        )

    assert result.exit_code == 0, result.stdout
    assert "TRADE MONITOR ALERT" in result.stdout
    assert result.stdout.count("TRADE MONITOR ALERT") == 1
    assert "TARGET_1_HIT" in result.stdout


def test_trade_monitor_alert_banner_title_once(capsys):
    from alerts.local_alerts import print_trade_monitor_alert_banner

    print_trade_monitor_alert_banner(_monitor_result(TradeMonitorAction.TARGET_1_HIT))

    captured = capsys.readouterr()
    assert captured.out.count("TRADE MONITOR ALERT") == 1


def test_trade_monitor_loop_no_alert_for_hold(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(ticker="SYNTHETIC"))

    with patch("trades.trade_monitor.TradeMonitor.monitor_open_trades") as mock_monitor:
        mock_monitor.return_value = [_monitor_result(TradeMonitorAction.HOLD)]
        result = cli_runner.invoke(
            app,
            ["trade-monitor", "--synthetic", "--loop", "--max-iterations", "1", "--interval-seconds", "0"],
        )

    assert result.exit_code == 0, result.stdout
    assert "TRADE MONITOR ALERT" not in result.stdout


def test_trade_monitor_has_no_beep_flag_or_sound_hooks():
    main_source = Path("main.py").read_text(encoding="utf-8").lower()
    alerts_source = Path("alerts/local_alerts.py").read_text(encoding="utf-8").lower()
    trade_monitor_source = main_source.split("def trade_monitor", 1)[1].split("# ---------------------------------------------------------------------------\n# entry point", 1)[0]

    assert "--beep" not in trade_monitor_source
    assert "emit_beep" not in trade_monitor_source
    assert "winsound" not in alerts_source
    assert "\\a" not in alerts_source
    assert "emit_beep" not in alerts_source


def test_trade_monitor_loop_duplicate_alert_suppressed(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(ticker="SYNTHETIC"))

    with patch("trades.trade_monitor.TradeMonitor.monitor_open_trades") as mock_monitor:
        mock_monitor.return_value = [_monitor_result(TradeMonitorAction.STOP_HIT)]
        result = cli_runner.invoke(
            app,
            ["trade-monitor", "--synthetic", "--loop", "--max-iterations", "2", "--interval-seconds", "0"],
        )

    assert result.exit_code == 0, result.stdout
    assert result.stdout.count("TRADE MONITOR ALERT") == 1
    assert "Repeated trade monitor alerts suppressed: 1" in result.stdout


def test_trade_monitor_loop_discord_sends_only_new_important_alert(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(ticker="SYNTHETIC"))

    with patch("trades.trade_monitor.TradeMonitor.monitor_open_trades") as mock_monitor:
        mock_monitor.return_value = [_monitor_result(TradeMonitorAction.WATCH)]
        with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
            with patch("alerts.discord_alerts.send_discord_message", return_value=True) as send_mock:
                result = cli_runner.invoke(
                    app,
                    ["trade-monitor", "--synthetic", "--loop", "--max-iterations", "2", "--interval-seconds", "0"],
                )

    assert result.exit_code == 0, result.stdout
    assert "Discord alerts: enabled" in result.stdout
    assert send_mock.call_count == 1


def test_trade_monitor_loop_discord_no_send_for_hold(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(ticker="SYNTHETIC"))

    with patch("trades.trade_monitor.TradeMonitor.monitor_open_trades") as mock_monitor:
        mock_monitor.return_value = [_monitor_result(TradeMonitorAction.HOLD)]
        with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
            with patch("alerts.discord_alerts.send_discord_message", return_value=True) as send_mock:
                result = cli_runner.invoke(
                    app,
                    ["trade-monitor", "--synthetic", "--loop", "--max-iterations", "1", "--interval-seconds", "0"],
                )

    assert result.exit_code == 0, result.stdout
    assert send_mock.call_count == 0


def test_trade_monitor_loop_discord_no_webhook_url_in_output(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(ticker="SYNTHETIC"))
    fake_url = "https://discord.example/webhook/SECRET_TOKEN"

    with patch("trades.trade_monitor.TradeMonitor.monitor_open_trades") as mock_monitor:
        mock_monitor.return_value = [_monitor_result(TradeMonitorAction.STOP_HIT)]
        with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
            with patch("alerts.discord_alerts.send_discord_message", return_value=True):
                result = cli_runner.invoke(
                    app,
                    ["trade-monitor", "--synthetic", "--loop", "--max-iterations", "1", "--interval-seconds", "0"],
                )

    assert result.exit_code == 0, result.stdout
    assert fake_url not in result.stdout
    assert "SECRET_TOKEN" not in result.stdout


def test_trade_monitor_alert_includes_guidance(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_make_trade(ticker="SYNTHETIC"))

    with patch("trades.trade_monitor.TradeMonitor.monitor_open_trades") as mock_monitor:
        mock_monitor.return_value = [_monitor_result(TradeMonitorAction.EXIT_VWAP_LOST)]
        result = cli_runner.invoke(
            app,
            ["trade-monitor", "--synthetic", "--loop", "--max-iterations", "1", "--interval-seconds", "0"],
        )

    assert result.exit_code == 0, result.stdout
    assert "Guidance:" in result.stdout
    assert ACTION_GUIDANCE[TradeMonitorAction.EXIT_VWAP_LOST] in result.stdout
