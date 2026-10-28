"""
Tests for Phase 11 adapter abstractions.

All tests pass without Alpaca credentials or network access.

Coverage:
  BrokerAdapter ABC — cannot be instantiated directly
  SimulatedBrokerAdapter — full offline operation
  AlpacaBrokerAdapter — paper-safety enforcement
  DataProvider ABC — cannot be instantiated directly
  YFinanceDataProvider — mockable interface
  CSVDataProvider — local file loading
  SyntheticDataProvider — offline data generation
  main.py _get_data_provider — correct provider selection
  main.py dry-run — uses SimulatedBrokerAdapter
  No secrets appear in any output
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from data.market_data import generate_synthetic_ohlcv


# ---------------------------------------------------------------------------
# BrokerAdapter — abstract base class
# ---------------------------------------------------------------------------

def test_broker_adapter_is_abstract():
    """BrokerAdapter must not be directly instantiable."""
    from broker.base import BrokerAdapter
    with pytest.raises(TypeError):
        BrokerAdapter()  # type: ignore[abstract]


def test_broker_error_is_exception():
    from broker.base import BrokerError
    err = BrokerError("test")
    assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# SimulatedBrokerAdapter — works without credentials
# ---------------------------------------------------------------------------

def test_simulated_adapter_is_configured():
    from broker.simulated_adapter import SimulatedBrokerAdapter
    adapter = SimulatedBrokerAdapter()
    assert adapter.is_configured is True


def test_simulated_adapter_is_paper():
    from broker.simulated_adapter import SimulatedBrokerAdapter
    assert SimulatedBrokerAdapter().is_paper is True


def test_simulated_adapter_get_account():
    from broker.simulated_adapter import SimulatedBrokerAdapter
    adapter = SimulatedBrokerAdapter(starting_equity=50_000.0)
    account = adapter.get_account()

    assert account["status"] == "ACTIVE"
    assert float(account["equity"]) == 50_000.0
    assert account["paper"] is True
    assert "secret" not in str(account).lower()
    assert "key" not in str(account).lower()


def test_simulated_adapter_is_market_open():
    from broker.simulated_adapter import SimulatedBrokerAdapter
    assert SimulatedBrokerAdapter().is_market_open() is True


def test_simulated_adapter_empty_positions():
    from broker.simulated_adapter import SimulatedBrokerAdapter
    adapter = SimulatedBrokerAdapter()
    assert adapter.list_positions() == []
    assert adapter.get_position("SPY") is None


def test_simulated_adapter_submit_order_returns_id():
    from broker.simulated_adapter import SimulatedBrokerAdapter
    adapter = SimulatedBrokerAdapter()
    result  = adapter.submit_order({
        "symbol": "SPY", "qty": 10.0, "side": "buy", "type": "market"
    })
    assert "id" in result
    assert result["status"] == "filled"
    assert result["symbol"] == "SPY"


def test_simulated_adapter_submit_order_zero_qty_raises():
    from broker.simulated_adapter import SimulatedBrokerAdapter
    from broker.base import BrokerError
    adapter = SimulatedBrokerAdapter()
    with pytest.raises(BrokerError, match="qty"):
        adapter.submit_order({"symbol": "SPY", "qty": 0.0, "side": "buy"})


def test_simulated_adapter_cancel_order():
    from broker.simulated_adapter import SimulatedBrokerAdapter
    adapter = SimulatedBrokerAdapter()
    assert adapter.cancel_order("any-id") is True


def test_simulated_adapter_cancel_all_orders():
    from broker.simulated_adapter import SimulatedBrokerAdapter
    adapter = SimulatedBrokerAdapter()
    assert isinstance(adapter.cancel_all_orders(), int)


def test_simulated_adapter_log_safe_no_secrets():
    from broker.simulated_adapter import SimulatedBrokerAdapter
    info = SimulatedBrokerAdapter().log_safe_info()
    info_str = str(info).lower()
    for banned in ("secret", "password", "token", "api_key"):
        assert banned not in info_str, f"Banned term '{banned}' in log_safe_info"


# ---------------------------------------------------------------------------
# AlpacaBrokerAdapter — paper-safety enforcement
# ---------------------------------------------------------------------------

def test_alpaca_adapter_refuses_live_url():
    """AlpacaBrokerAdapter must block calls to the live Alpaca URL."""
    from broker.alpaca_adapter import AlpacaBrokerAdapter
    from broker.alpaca_client import AlpacaSafetyError
    from broker.base import BrokerError
    from config.settings import Settings

    s = Settings(
        ALPACA_BASE_URL="https://api.alpaca.markets",   # LIVE URL — not paper!
        PAPER_TRADING=True,
        LIVE_TRADING_ALLOWED=False,
    )
    adapter = AlpacaBrokerAdapter(settings_override=s)

    # Adapter itself creates fine, but any real call triggers safety check
    with pytest.raises((AlpacaSafetyError, BrokerError)):
        adapter.submit_order({"symbol": "SPY", "qty": 1, "side": "buy"})


def test_alpaca_adapter_is_paper():
    from broker.alpaca_adapter import AlpacaBrokerAdapter
    adapter = AlpacaBrokerAdapter()
    assert adapter.is_paper is True


def test_alpaca_adapter_is_configured_without_keys():
    from broker.alpaca_adapter import AlpacaBrokerAdapter
    from config.settings import Settings
    s = Settings(ALPACA_API_KEY="", ALPACA_SECRET_KEY="")
    adapter = AlpacaBrokerAdapter(settings_override=s)
    assert adapter.is_configured is False


def test_alpaca_adapter_log_safe_no_secrets():
    from broker.alpaca_adapter import AlpacaBrokerAdapter
    from config.settings import Settings
    s = Settings(ALPACA_API_KEY="TEST_KEY", ALPACA_SECRET_KEY="SUPER_SECRET")
    adapter = AlpacaBrokerAdapter(settings_override=s)
    info_str = str(adapter.log_safe_info()).lower()
    assert "super_secret" not in info_str
    assert "test_key"    not in info_str


# ---------------------------------------------------------------------------
# DataProvider — abstract base class
# ---------------------------------------------------------------------------

def test_data_provider_is_abstract():
    """DataProvider must not be directly instantiable."""
    from data_providers.base import DataProvider
    with pytest.raises(TypeError):
        DataProvider()  # type: ignore[abstract]


def test_data_provider_error_is_exception():
    from data_providers.base import DataProviderError
    err = DataProviderError("test")
    assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# YFinanceDataProvider — mockable
# ---------------------------------------------------------------------------

def test_yfinance_provider_mocked():
    """YFinanceDataProvider must delegate to data.market_data.fetch_ticker."""
    from data_providers import YFinanceDataProvider

    df = generate_synthetic_ohlcv(rows=200)
    with patch("data.market_data.fetch_ticker", return_value=df):
        provider = YFinanceDataProvider()
        result   = provider.get_bars("SPY", start="2022-01-01", end="2023-12-31")

    assert len(result) == 200
    assert "close" in result.columns


def test_yfinance_provider_wraps_data_fetch_error():
    """DataFetchError from fetch_ticker must be re-raised as DataProviderError."""
    from data_providers import YFinanceDataProvider
    from data_providers.base import DataProviderError
    from data.market_data import DataFetchError

    with patch("data.market_data.fetch_ticker", side_effect=DataFetchError("no data")):
        with pytest.raises(DataProviderError, match="YFinanceDataProvider"):
            YFinanceDataProvider().get_bars("SPY")


def test_yfinance_provider_name():
    from data_providers import YFinanceDataProvider
    assert YFinanceDataProvider().provider_name == "YFinance"


# ---------------------------------------------------------------------------
# AlpacaDataProvider — market-data only, mockable
# ---------------------------------------------------------------------------

def test_alpaca_data_provider_declares_intraday_timeframes():
    from data_providers import AlpacaDataProvider

    provider = AlpacaDataProvider(client=object())

    assert provider.supported_timeframes == ["1Min", "5Min"]
    assert provider.provider_name == "AlpacaData"
    assert provider.feed == "iex"


def test_alpaca_data_provider_accepts_sip_feed():
    from data_providers import AlpacaDataProvider

    provider = AlpacaDataProvider(client=object(), feed="sip")

    assert provider.feed == "sip"


def test_alpaca_data_provider_mocked_client_returns_ohlcv():
    from types import SimpleNamespace

    from data_providers import AlpacaDataProvider

    df = pd.DataFrame({
        "symbol": ["SPY", "SPY"],
        "timestamp": pd.to_datetime(["2026-06-10 09:30", "2026-06-10 09:31"]),
        "open": [100.0, 101.0],
        "high": [101.0, 102.0],
        "low": [99.5, 100.5],
        "close": [100.5, 101.5],
        "volume": [1000, 1200],
    }).set_index(["symbol", "timestamp"])
    client = SimpleNamespace(get_stock_bars=lambda request: SimpleNamespace(df=df))

    result = AlpacaDataProvider(client=client).get_bars("SPY", timeframe="1Min")

    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert len(result) == 2
    assert result["close"].iloc[-1] == 101.5
    assert "data_age_seconds" in result.attrs


def test_alpaca_data_provider_repeated_calls_use_fresh_end_time():
    from datetime import datetime
    from types import SimpleNamespace

    from data_providers import AlpacaDataProvider

    df = pd.DataFrame({
        "symbol": ["SPY"],
        "timestamp": pd.to_datetime(["2026-06-10 09:31"]),
        "open": [100.0],
        "high": [101.0],
        "low": [99.5],
        "close": [100.5],
        "volume": [1000],
    }).set_index(["symbol", "timestamp"])

    captured_ends = []

    def get_stock_bars(request):
        captured_ends.append(request.get("end") if isinstance(request, dict) else getattr(request, "end"))
        return SimpleNamespace(df=df)

    client = SimpleNamespace(get_stock_bars=get_stock_bars)
    provider = AlpacaDataProvider(client=client)

    with patch.object(
        AlpacaDataProvider,
        "_current_utc",
        side_effect=[
            datetime(2026, 6, 10, 14, 0, 0),
            datetime(2026, 6, 10, 14, 0, 1),
            datetime(2026, 6, 10, 14, 1, 0),
            datetime(2026, 6, 10, 14, 1, 1),
        ],
    ):
        provider.get_bars("SPY", timeframe="1Min")
        provider.get_bars("SPY", timeframe="1Min")

    assert len(captured_ends) == 2
    assert captured_ends[1] > captured_ends[0]


def test_alpaca_data_provider_passes_feed_to_mocked_request():
    from types import SimpleNamespace

    from data_providers import AlpacaDataProvider

    df = pd.DataFrame({
        "symbol": ["SPY"],
        "timestamp": pd.to_datetime(["2026-06-10 09:31"]),
        "open": [100.0],
        "high": [101.0],
        "low": [99.5],
        "close": [100.5],
        "volume": [1000],
    }).set_index(["symbol", "timestamp"])
    captured_feeds = []

    def get_stock_bars(request):
        feed = request.get("feed") if isinstance(request, dict) else getattr(request, "feed", None)
        captured_feeds.append(str(feed).lower())
        return SimpleNamespace(df=df)

    client = SimpleNamespace(get_stock_bars=get_stock_bars)
    AlpacaDataProvider(client=client, feed="iex").get_bars("SPY", timeframe="1Min")

    assert captured_feeds
    assert "iex" in captured_feeds[0]


def test_alpaca_data_provider_get_latest_bar_uses_latest_endpoint():
    from datetime import datetime
    from types import SimpleNamespace

    from data_providers import AlpacaDataProvider

    calls = []

    def get_stock_latest_bar(request):
        calls.append(request)
        return {
            "SPY": {
                "timestamp": datetime(2026, 6, 10, 15, 55),
                "open": 100.0,
                "high": 101.0,
                "low": 99.5,
                "close": 100.5,
                "volume": 1000,
            }
        }

    client = SimpleNamespace(get_stock_latest_bar=get_stock_latest_bar)
    result = AlpacaDataProvider(client=client).get_latest_bar("SPY", timeframe="1Min")

    assert calls
    assert len(result) == 1
    assert result["close"].iloc[-1] == 100.5
    assert result.attrs["latest_bar_time"].startswith("2026-06-10T15:55")
    assert "provider_time" in result.attrs
    assert "data_age_seconds" in result.attrs


def test_alpaca_data_provider_missing_keys_clear_error():
    from config.settings import Settings
    from data_providers import AlpacaDataProvider
    from data_providers.base import DataProviderError

    settings = Settings(ALPACA_API_KEY="", ALPACA_SECRET_KEY="")

    with pytest.raises(DataProviderError, match="requires Alpaca paper data credentials"):
        AlpacaDataProvider(settings_override=settings).get_bars("SPY", timeframe="1Min")


def test_alpaca_data_provider_rejects_unsupported_timeframe():
    from data_providers import AlpacaDataProvider
    from data_providers.base import DataProviderError

    with pytest.raises(DataProviderError, match="does not support timeframe"):
        AlpacaDataProvider(client=object()).get_bars("SPY", timeframe="1Day")


# ---------------------------------------------------------------------------
# CSVDataProvider — local file loading
# ---------------------------------------------------------------------------

def test_csv_provider_loads_valid_file(tmp_path):
    from data_providers import CSVDataProvider

    df  = generate_synthetic_ohlcv(rows=150)
    csv = tmp_path / "test.csv"
    df.index.name = "date"
    df.to_csv(csv)

    provider = CSVDataProvider(file_path=csv)
    result   = provider.get_bars("IGNORED_TICKER")

    assert len(result) == 150
    assert "close" in result.columns


def test_csv_provider_raises_for_missing_file(tmp_path):
    from data_providers import CSVDataProvider
    from data_providers.base import DataProviderError

    with pytest.raises(DataProviderError, match="not found"):
        CSVDataProvider(file_path=tmp_path / "nonexistent.csv")


def test_csv_provider_name(tmp_path):
    from data_providers import CSVDataProvider

    df  = generate_synthetic_ohlcv(rows=100)
    csv = tmp_path / "mydata.csv"
    df.index.name = "date"
    df.to_csv(csv)

    provider = CSVDataProvider(file_path=csv)
    assert "mydata.csv" in provider.provider_name


# ---------------------------------------------------------------------------
# SyntheticDataProvider — offline data generation
# ---------------------------------------------------------------------------

def test_synthetic_provider_produces_valid_data():
    from data_providers import SyntheticDataProvider
    from data.market_data import validate_ohlcv

    provider = SyntheticDataProvider(rows=300, seed=42)
    df       = provider.get_bars("SPY")

    assert len(df) == 300
    validate_ohlcv(df)   # Must not raise


def test_synthetic_provider_is_deterministic():
    from data_providers import SyntheticDataProvider

    df1 = SyntheticDataProvider(rows=100, seed=7).get_bars("SPY")
    df2 = SyntheticDataProvider(rows=100, seed=7).get_bars("SPY")
    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_provider_different_tickers_not_identical():
    from data_providers import SyntheticDataProvider

    provider = SyntheticDataProvider(rows=100, seed=42)
    spy = provider.get_bars("SPY")
    qqq = provider.get_bars("QQQ")
    # Different tickers should have different close prices (different seeds)
    assert not spy["close"].equals(qqq["close"])


def test_synthetic_provider_get_multiple_bars():
    from data_providers import SyntheticDataProvider
    from data.market_data import validate_ohlcv

    data = SyntheticDataProvider(rows=200).get_multiple_bars(["SPY", "QQQ", "AAPL"])
    assert set(data.keys()) == {"SPY", "QQQ", "AAPL"}
    for ticker, df in data.items():
        assert len(df) == 200
        validate_ohlcv(df)


def test_synthetic_provider_name():
    from data_providers import SyntheticDataProvider
    assert SyntheticDataProvider().provider_name == "Synthetic"


def test_synthetic_provider_no_secrets_in_repr():
    from data_providers import SyntheticDataProvider
    provider = SyntheticDataProvider()
    r = repr(provider)
    for banned in ("secret", "key", "password", "api"):
        assert banned not in r.lower(), f"Banned term '{banned}' in repr"


# ---------------------------------------------------------------------------
# main.py _get_data_provider — correct provider selection
# ---------------------------------------------------------------------------

def test_get_data_provider_synthetic():
    from main import _get_data_provider
    from data_providers.synthetic_provider import SyntheticDataProvider
    provider = _get_data_provider(synthetic=True, csv_file=None)
    assert isinstance(provider, SyntheticDataProvider)


def test_get_data_provider_csv(tmp_path):
    from main import _get_data_provider
    from data_providers.csv_provider import CSVDataProvider

    csv = tmp_path / "data.csv"
    df  = generate_synthetic_ohlcv(rows=100)
    df.index.name = "date"
    df.to_csv(csv)

    provider = _get_data_provider(synthetic=False, csv_file=str(csv))
    assert isinstance(provider, CSVDataProvider)


def test_get_data_provider_default():
    from main import _get_data_provider
    from data_providers.yfinance_provider import YFinanceDataProvider
    provider = _get_data_provider(synthetic=False, csv_file=None)
    assert isinstance(provider, YFinanceDataProvider)


# ---------------------------------------------------------------------------
# main.py dry-run uses SimulatedBrokerAdapter (CLI test)
# ---------------------------------------------------------------------------

def test_main_paper_once_dry_run_uses_simulated_broker():
    """paper-once --synthetic --dry-run must complete without Alpaca credentials."""
    from typer.testing import CliRunner
    import main as m

    runner = CliRunner()
    result = runner.invoke(m.app, ["paper-once", "--ticker", "SPY", "--synthetic"])

    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert "DRY" in result.output.upper() or "dry" in result.output.lower()


def test_main_paper_loop_dry_run_uses_simulated_broker():
    """paper-loop --synthetic --max-iterations 1 must not need credentials."""
    from typer.testing import CliRunner
    import main as m

    runner = CliRunner()
    result = runner.invoke(m.app, [
        "paper-loop", "--synthetic", "--max-iterations", "1",
        "--ticker", "SPY", "--interval-seconds", "0",
    ])

    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"


def test_main_backtest_synthetic_uses_data_provider():
    """backtest --synthetic must route through SyntheticDataProvider."""
    from typer.testing import CliRunner
    import main as m
    from data_providers.synthetic_provider import SyntheticDataProvider

    captured = []

    original_get_multiple = SyntheticDataProvider.get_multiple_bars

    def spy_get_multiple(self, *args, **kwargs):
        captured.append("called")
        return original_get_multiple(self, *args, **kwargs)

    with patch.object(SyntheticDataProvider, "get_multiple_bars", spy_get_multiple):
        runner = CliRunner()
        runner.invoke(m.app, ["backtest", "--synthetic"])

    assert captured, "SyntheticDataProvider.get_multiple_bars was not called"


# ---------------------------------------------------------------------------
# No secrets leak in any output
# ---------------------------------------------------------------------------

def test_simulated_adapter_get_clock_no_secrets():
    from broker.simulated_adapter import SimulatedBrokerAdapter
    clock = SimulatedBrokerAdapter().get_clock()
    for banned in ("secret", "key", "password", "api"):
        assert banned not in str(clock).lower()


# ---------------------------------------------------------------------------
# PositionTracker integration with adapters
# ---------------------------------------------------------------------------

def test_position_tracker_with_simulated_adapter():
    """PositionTracker should work with SimulatedBrokerAdapter."""
    from broker.simulated_adapter import SimulatedBrokerAdapter
    from broker.position_tracker import PositionTracker

    broker = SimulatedBrokerAdapter(starting_equity=100_000.0)
    tracker = PositionTracker(broker_adapter=broker)

    # No positions yet, should return empty portfolio
    snapshot = tracker.get_portfolio_snapshot()
    assert snapshot.equity == 100_000.0
    assert snapshot.open_positions_count == 0
    assert len(snapshot.positions) == 0
    assert snapshot.portfolio_exposure == 0.0


def test_position_tracker_with_mocked_broker():
    """PositionTracker should work with mocked BrokerAdapter."""
    from broker.base import BrokerAdapter
    from broker.position_tracker import PositionTracker
    from unittest.mock import MagicMock

    mock_broker = MagicMock(spec=BrokerAdapter)
    mock_broker.get_account.return_value = {
        "equity": "100000.00",
        "cash": "50000.00",
        "buying_power": "50000.00",
        "status": "ACTIVE",
    }
    mock_broker.list_positions.return_value = [
        {
            "symbol": "SPY",
            "qty": "100",
            "market_value": "50000.00",
            "avg_entry_price": "400.00",
            "current_price": "500.00",
            "unrealized_pl": "10000.00",
            "unrealized_plpc": "0.25",
            "side": "long",
        }
    ]

    tracker = PositionTracker(broker_adapter=mock_broker)
    snapshot = tracker.get_portfolio_snapshot()

    assert snapshot.equity == 100_000.0
    assert snapshot.open_positions_count == 1
    assert snapshot.portfolio_exposure == pytest.approx(0.5)
    assert "SPY" in snapshot.positions
    spy_pos = snapshot.positions["SPY"]
    assert spy_pos.quantity == 100.0
    assert spy_pos.market_value == 50_000.0


def test_position_tracker_ticker_exposure():
    """PositionTracker should calculate ticker exposure correctly."""
    from broker.base import BrokerAdapter
    from broker.position_tracker import PositionTracker
    from unittest.mock import MagicMock

    mock_broker = MagicMock(spec=BrokerAdapter)
    mock_broker.get_account.return_value = {
        "equity": "100000.00",
        "cash": "70000.00",
        "buying_power": "70000.00",
        "status": "ACTIVE",
    }
    mock_broker.list_positions.return_value = [
        {
            "symbol": "AAPL",
            "qty": "100",
            "market_value": "30000.00",
            "avg_entry_price": "150.00",
            "current_price": "300.00",
            "unrealized_pl": "15000.00",
            "unrealized_plpc": "1.0",
            "side": "long",
        }
    ]

    tracker = PositionTracker(broker_adapter=mock_broker)
    aapl_exposure = tracker.get_current_ticker_exposure("AAPL")
    unknown_exposure = tracker.get_current_ticker_exposure("UNKNOWN")

    assert aapl_exposure == pytest.approx(0.30)  # 30000 / 100000
    assert unknown_exposure == 0.0


# ---------------------------------------------------------------------------
# DataProvider timeframe support/validation
# ---------------------------------------------------------------------------

def test_yfinance_provider_declares_supported_timeframes():
    """YFinanceDataProvider should declare daily plus intraday timeframes."""
    from data_providers import YFinanceDataProvider
    provider = YFinanceDataProvider()
    assert provider.supported_timeframes == ["1Day", "1Min", "5Min", "15Min"]


def test_yfinance_provider_maps_timeframe_to_interval():
    """YFinanceDataProvider should pass the mapped yfinance interval to fetch_ticker."""
    from data_providers import YFinanceDataProvider
    from unittest.mock import patch

    df = generate_synthetic_ohlcv(rows=50)
    provider = YFinanceDataProvider()

    with patch("data.market_data.fetch_ticker", return_value=df) as mock_fetch:
        provider.get_bars("SPY", start="2026-05-01", end="2026-05-30", timeframe="5Min")

    # interval kwarg must be the yfinance code for 5Min
    assert mock_fetch.call_args.kwargs.get("interval") == "5m"


def test_yfinance_provider_rejects_unsupported_timeframe():
    """YFinanceDataProvider should raise a clear error for a timeframe it can't map."""
    from data_providers import YFinanceDataProvider
    from data_providers.base import DataProviderError

    provider = YFinanceDataProvider()

    with pytest.raises(DataProviderError, match="does not support.*1Hour"):
        provider.get_bars("SPY", timeframe="1Hour")


def test_csv_provider_supports_intraday_timeframes(tmp_path):
    """CSVDataProvider should accept daily and intraday timeframe strings."""
    from data_providers import CSVDataProvider

    df = generate_synthetic_ohlcv(rows=100)
    csv = tmp_path / "test.csv"
    df.index.name = "date"
    df.to_csv(csv)

    provider = CSVDataProvider(file_path=csv)
    assert provider.supported_timeframes == ["1Day", "1Min", "5Min", "15Min"]

    # Intraday strings load the file as-is (no resampling), without raising.
    result = provider.get_bars("IGNORED", timeframe="5Min")
    assert "close" in result.columns


def test_csv_provider_rejects_unsupported_timeframe(tmp_path):
    """CSVDataProvider should raise DataProviderError for an unsupported timeframe."""
    from data_providers import CSVDataProvider
    from data_providers.base import DataProviderError

    df = generate_synthetic_ohlcv(rows=100)
    csv = tmp_path / "test.csv"
    df.index.name = "date"
    df.to_csv(csv)

    provider = CSVDataProvider(file_path=csv)

    with pytest.raises(DataProviderError, match="does not support.*1Hour"):
        provider.get_bars("SPY", timeframe="1Hour")


def test_synthetic_provider_supports_intraday_timeframes():
    """SyntheticDataProvider should declare daily plus intraday timeframes."""
    from data_providers import SyntheticDataProvider
    provider = SyntheticDataProvider()
    assert provider.supported_timeframes == ["1Day", "1Min", "5Min", "15Min"]


def test_synthetic_provider_intraday_returns_datetimeindex():
    """SyntheticDataProvider 5Min data must have a DatetimeIndex with intraday steps."""
    from data_providers import SyntheticDataProvider
    from data.market_data import validate_ohlcv

    provider = SyntheticDataProvider(rows=120, seed=11)
    df = provider.get_bars("SPY", timeframe="5Min")

    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 120
    validate_ohlcv(df)  # must not raise

    # Consecutive intraday bars step by 5 minutes within a session.
    deltas = df.index.to_series().diff().dropna()
    assert (deltas == pd.Timedelta(minutes=5)).any()
    # At least one bar should sit at the 09:30 session open.
    assert any((ts.hour == 9 and ts.minute == 30) for ts in df.index)


def test_synthetic_provider_daily_index_unchanged():
    """Daily synthetic data must keep its business-day index (no time component)."""
    from data_providers import SyntheticDataProvider

    df = SyntheticDataProvider(rows=50, seed=11).get_bars("SPY")  # default 1Day
    assert isinstance(df.index, pd.DatetimeIndex)
    # Daily bars have no intraday time component.
    assert all((ts.hour == 0 and ts.minute == 0) for ts in df.index)


def test_synthetic_provider_rejects_unsupported_timeframe():
    """SyntheticDataProvider should raise DataProviderError for an unsupported timeframe."""
    from data_providers import SyntheticDataProvider
    from data_providers.base import DataProviderError

    provider = SyntheticDataProvider()

    with pytest.raises(DataProviderError, match="does not support.*1Hour"):
        provider.get_bars("SPY", timeframe="1Hour")


def test_data_provider_validate_timeframe_helper():
    """DataProvider._validate_timeframe() should reject unsupported timeframes clearly."""
    from data_providers import SyntheticDataProvider
    from data_providers.base import DataProviderError

    provider = SyntheticDataProvider()

    # Should not raise for supported timeframes
    provider._validate_timeframe("1Day")
    provider._validate_timeframe("5Min")

    # Should raise for unsupported timeframe, listing what IS supported
    with pytest.raises(DataProviderError, match="Synthetic.*1Hour.*Supported"):
        provider._validate_timeframe("1Hour")


# ---------------------------------------------------------------------------
# OrderExecutor integration with adapters
# ---------------------------------------------------------------------------

def test_order_executor_with_simulated_adapter():
    """OrderExecutor should work with SimulatedBrokerAdapter (dry_run)."""
    from broker.simulated_adapter import SimulatedBrokerAdapter
    from broker.order_executor import OrderExecutor, OrderExecutionResult
    from strategy.signals import Signal, SignalAction, RegimeLabel
    import pandas as pd

    broker = SimulatedBrokerAdapter()
    executor = OrderExecutor(broker_adapter=broker, dry_run=True)

    signal = Signal(
        ticker="SPY", timestamp=pd.Timestamp.now(),
        action=SignalAction.BUY, confidence=0.8, reason="test",
        regime=RegimeLabel.BULL, entry_price=400.0, stop_loss=390.0,
        take_profit=420.0, max_position_value=0.0,
        metadata={"rejection_reasons": []},
    )

    class MockDecision:
        approved = True
        final_action = SignalAction.BUY
        reason = "test"
        class PositionSize:
            approved = True
            shares = 10.0
            position_value = 4000.0
            risk_dollars = 100.0
        position_size = PositionSize()

    result = executor.execute_risk_decision(signal, MockDecision())
    assert result.submitted is True
    assert result.dry_run is True
    assert result.action == "BUY"
    assert result.quantity == 10.0
    assert "DRY" in result.status.upper() or "simulated" in result.status.lower()


def test_order_executor_with_alpaca_adapter_mocked():
    """OrderExecutor should work with AlpacaBrokerAdapter (mocked)."""
    from broker.base import BrokerAdapter, BrokerError
    from broker.order_executor import OrderExecutor
    from strategy.signals import Signal, SignalAction, RegimeLabel
    import pandas as pd
    from unittest.mock import MagicMock

    # Mock the broker that simulates successful order submission
    mock_broker = MagicMock(spec=BrokerAdapter)
    mock_broker.is_paper = True
    mock_broker.submit_order.return_value = {
        "id": "abc-123",
        "status": "filled",
        "symbol": "SPY",
        "qty": "10.0",
        "side": "buy",
    }

    executor = OrderExecutor(broker_adapter=mock_broker, dry_run=False)

    signal = Signal(
        ticker="SPY", timestamp=pd.Timestamp.now(),
        action=SignalAction.BUY, confidence=0.8, reason="test",
        regime=RegimeLabel.BULL, entry_price=400.0, stop_loss=390.0,
        take_profit=420.0, max_position_value=0.0,
        metadata={"rejection_reasons": []},
    )

    class MockDecision:
        approved = True
        final_action = SignalAction.BUY
        reason = "test"
        class PositionSize:
            approved = True
            shares = 10.0
            position_value = 4000.0
            risk_dollars = 100.0
        position_size = PositionSize()

    result = executor.execute_risk_decision(signal, MockDecision())
    assert result.submitted is True
    assert result.dry_run is False
    assert result.action == "BUY"
    assert result.order_id == "abc-123"
    # Verify submit_order was called with proper order dict
    mock_broker.submit_order.assert_called_once()
    call_args = mock_broker.submit_order.call_args[0][0]
    assert call_args["symbol"] == "SPY"
    assert call_args["qty"] == 10.0
    assert call_args["side"] == "buy"
    assert call_args["stop_loss"] == 390.0
    assert call_args["take_profit"] == 420.0


def test_order_executor_dry_run_never_submits():
    """OrderExecutor in dry_run mode should never call submit_order."""
    from broker.base import BrokerAdapter
    from broker.order_executor import OrderExecutor
    from strategy.signals import Signal, SignalAction, RegimeLabel
    import pandas as pd
    from unittest.mock import MagicMock

    mock_broker = MagicMock(spec=BrokerAdapter)
    mock_broker.is_paper = True
    executor = OrderExecutor(broker_adapter=mock_broker, dry_run=True)

    signal = Signal(
        ticker="SPY", timestamp=pd.Timestamp.now(),
        action=SignalAction.BUY, confidence=0.8, reason="test",
        regime=RegimeLabel.BULL, entry_price=400.0, stop_loss=390.0,
        take_profit=420.0, max_position_value=0.0,
        metadata={"rejection_reasons": []},
    )

    class MockDecision:
        approved = True
        final_action = SignalAction.BUY
        reason = "test"
        class PositionSize:
            approved = True
            shares = 10.0
            position_value = 4000.0
            risk_dollars = 100.0
        position_size = PositionSize()

    result = executor.execute_risk_decision(signal, MockDecision())
    assert result.submitted is True
    assert result.dry_run is True
    # In dry_run mode, submit_order should NEVER be called
    mock_broker.submit_order.assert_not_called()
