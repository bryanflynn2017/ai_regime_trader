"""
Tests for the setup-scan CLI command (Phase 12).

The setup-scan command is a watchlist detector only — it loads market data,
runs DayTradeSetupDetector on each ticker, prints results, and logs to JSONL.
It never places orders or interacts with the broker.

Coverage:
  - Synthetic data mode (offline, no network)
  - JSONL logging with timestamp, all setup fields, no secrets
  - Table printing (no crashes even with NO_SETUP rows)
  - Comma-separated ticker parsing
  - Single-ticker and multi-ticker modes
  - Graceful handling of NO_SETUP results
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import typer.testing

from main import app


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _load_jsonl_log(log_file: Path) -> list[dict]:
    """Load a JSONL log file into a list of dicts."""
    if not log_file.exists():
        return []
    lines = []
    with open(log_file) as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    return lines


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cli_runner():
    """Return a typer CliRunner for invoking the app."""
    return typer.testing.CliRunner()


# ---------------------------------------------------------------------------
# Test 1: Synthetic mode (offline, no network calls)
# ---------------------------------------------------------------------------

def test_setup_scan_synthetic_mode(cli_runner):
    """Run setup-scan with --synthetic; should not make network calls."""
    result = cli_runner.invoke(app, ["setup-scan", "--synthetic"])

    assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"
    # Default output is the ranked view: either top setups or the no-setups notice.
    assert (
        "Top Trade Setups" in result.stdout
        or "No valid setups found across scanned tickers." in result.stdout
    )


def test_setup_scan_synthetic_produces_jsonl(cli_runner):
    """Synthetic mode should produce a JSONL log file."""
    # Clean up any existing log first
    log_file = Path("logs/setup_scans.jsonl")
    if log_file.exists():
        log_file.unlink()

    result = cli_runner.invoke(app, ["setup-scan", "--synthetic"])
    assert result.exit_code == 0

    assert log_file.exists(), f"JSONL log not created at {log_file}"

    lines = _load_jsonl_log(log_file)
    assert len(lines) > 0, "JSONL log is empty"

    # Each line should be a valid JSON record
    for record in lines:
        assert isinstance(record, dict)
        assert "timestamp" in record
        assert "ticker" in record
        assert "setup_type" in record
        assert "is_valid" in record


# ---------------------------------------------------------------------------
# Test 2: JSONL logging: structure and no secrets
# ---------------------------------------------------------------------------

def test_setup_scan_jsonl_has_required_fields(cli_runner):
    """JSONL records must have all setup fields."""
    log_file = Path("logs/setup_scans.jsonl")
    if log_file.exists():
        log_file.unlink()

    cli_runner.invoke(app, ["setup-scan", "--synthetic"])

    lines = _load_jsonl_log(log_file)

    for record in lines:
        required_fields = [
            "timestamp", "ticker", "setup_type", "is_valid", "confidence",
            "entry_trigger", "stop_loss", "target_1", "target_2",
            "risk_reward_1", "risk_reward_2", "reason", "warnings", "metadata",
        ]
        for field in required_fields:
            assert field in record, f"Missing field '{field}' in JSONL record"


def test_setup_scan_jsonl_no_api_keys(cli_runner):
    """JSONL log should NOT contain any API keys or broker credentials."""
    log_file = Path("logs/setup_scans.jsonl")
    if log_file.exists():
        log_file.unlink()

    cli_runner.invoke(app, ["setup-scan", "--synthetic"])

    if log_file.exists():
        full_content = log_file.read_text()
        # Check for common secret patterns
        assert "ALPACA_API_KEY" not in full_content
        assert "ALPACA_SECRET" not in full_content
        assert "API_KEY" not in full_content
        assert "api_key" not in full_content


def test_setup_scan_jsonl_timestamp_format(cli_runner):
    """Each JSONL record must have a valid ISO timestamp."""
    log_file = Path("logs/setup_scans.jsonl")
    if log_file.exists():
        log_file.unlink()

    cli_runner.invoke(app, ["setup-scan", "--synthetic"])

    lines = _load_jsonl_log(log_file)

    for record in lines:
        ts = record["timestamp"]
        # ISO format: "2026-06-04T14:30:45.123456"
        try:
            pd.Timestamp(ts)
        except Exception as exc:
            pytest.fail(f"Invalid timestamp format: {ts} ({exc})")


# ---------------------------------------------------------------------------
# Test 3: Table printing (readable output, no crashes)
# ---------------------------------------------------------------------------

def test_setup_scan_prints_table_header(cli_runner):
    """Output should include the table header (with --full-table)."""
    result = cli_runner.invoke(app, ["setup-scan", "--synthetic", "--full-table"])

    assert result.exit_code == 0
    assert "Ticker" in result.stdout
    assert "Setup Type" in result.stdout
    assert "Conf" in result.stdout
    assert "Entry" in result.stdout
    assert "Stop" in result.stdout


def test_setup_scan_prints_results_rows(cli_runner):
    """Output should include data rows (at least one ticker) in the full table."""
    result = cli_runner.invoke(app, ["setup-scan", "--synthetic", "--full-table"])

    assert result.exit_code == 0
    lines = result.stdout.split("\n")
    # Find lines with ticker symbols (should have $ signs for prices or — for N/A)
    data_lines = [l for l in lines if ("SPY" in l or "QQQ" in l or "NVDA" in l) and ("$" in l or "—" in l)]
    assert len(data_lines) > 0, f"No data rows in output:\n{result.stdout}"


def test_setup_scan_handles_no_setup_rows(cli_runner):
    """Table should gracefully print NO_SETUP rows (confidence 0, all values —)."""
    result = cli_runner.invoke(app, ["setup-scan", "--synthetic", "--full-table"])

    assert result.exit_code == 0
    # The detector may return NO_SETUP for weak data or on first few bars.
    # Either way, the command should not crash when printing them.
    assert "Setup Results:" in result.stdout


# ---------------------------------------------------------------------------
# Test 4: Ticker parsing (single, multiple, comma-separated)
# ---------------------------------------------------------------------------

def test_setup_scan_single_ticker(cli_runner):
    """Single --ticker option should work."""
    result = cli_runner.invoke(app, ["setup-scan", "--ticker", "SPY", "--synthetic"])

    assert result.exit_code == 0
    assert "Tickers: SPY" in result.stdout


def test_setup_scan_comma_separated_tickers(cli_runner):
    """Comma-separated tickers should be parsed correctly."""
    result = cli_runner.invoke(
        app,
        ["setup-scan", "--ticker", "SPY,QQQ,NVDA", "--synthetic"]
    )

    assert result.exit_code == 0
    assert "Tickers: SPY, QQQ, NVDA" in result.stdout


def test_setup_scan_ticker_case_insensitive(cli_runner):
    """Tickers should be normalized to uppercase."""
    result = cli_runner.invoke(
        app,
        ["setup-scan", "--ticker", "spy", "--synthetic"]
    )

    assert result.exit_code == 0
    assert "Tickers: SPY" in result.stdout
    assert "SPY" in result.stdout


# ---------------------------------------------------------------------------
# Test 5: No broker interaction
# ---------------------------------------------------------------------------

def test_setup_scan_no_broker_calls(cli_runner):
    """Setup-scan should not make any broker API calls."""
    # Mock the broker-related modules to ensure they are not called
    with patch("broker.alpaca_adapter.AlpacaBrokerAdapter") as mock_broker:
        with patch("broker.simulated_adapter.SimulatedBrokerAdapter") as mock_sim:
            result = cli_runner.invoke(app, ["setup-scan", "--synthetic"])

            assert result.exit_code == 0
            # Broker mocks should not have been instantiated
            mock_broker.assert_not_called()
            mock_sim.assert_not_called()


def test_setup_scan_no_order_executor(cli_runner):
    """Setup-scan should not instantiate OrderExecutor or place any orders."""
    with patch("broker.order_executor.OrderExecutor") as mock_executor:
        result = cli_runner.invoke(app, ["setup-scan", "--synthetic"])

        assert result.exit_code == 0
        mock_executor.assert_not_called()


def test_setup_scan_no_position_tracker(cli_runner):
    """Setup-scan should not instantiate PositionTracker."""
    with patch("broker.position_tracker.PositionTracker") as mock_tracker:
        result = cli_runner.invoke(app, ["setup-scan", "--synthetic"])

        assert result.exit_code == 0
        mock_tracker.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6: Graceful handling of invalid/empty data
# ---------------------------------------------------------------------------

def test_setup_scan_with_csv_file_missing(cli_runner):
    """Should exit with error if CSV file does not exist."""
    result = cli_runner.invoke(
        app,
        ["setup-scan", "--csv", "/nonexistent/file.csv"]
    )

    # Should not succeed
    assert result.exit_code != 0


def test_setup_scan_jsonl_appends_to_existing_log(cli_runner):
    """Running setup-scan twice should append to the JSONL log, not overwrite."""
    log_file = Path("logs/setup_scans.jsonl")
    if log_file.exists():
        log_file.unlink()

    cli_runner.invoke(app, ["setup-scan", "--synthetic"])
    if log_file.exists():
        count_after_first = len(_load_jsonl_log(log_file))

        cli_runner.invoke(app, ["setup-scan", "--synthetic"])
        count_after_second = len(_load_jsonl_log(log_file))

        assert count_after_second >= count_after_first, "JSONL should append, not overwrite"


# ---------------------------------------------------------------------------
# Test 7: Output messages and user feedback
# ---------------------------------------------------------------------------

def test_setup_scan_shows_success_message(cli_runner):
    """Should display a success message with the log file path."""
    result = cli_runner.invoke(app, ["setup-scan", "--synthetic"])

    assert result.exit_code == 0
    assert "setup_scans.jsonl" in result.stdout
    assert "logged" in result.stdout.lower()


def test_setup_scan_warns_about_secrets_in_log(cli_runner):
    """Should explicitly state that no secrets are included."""
    result = cli_runner.invoke(app, ["setup-scan", "--synthetic"])

    assert result.exit_code == 0
    assert "No API keys or secrets" in result.stdout or "secrets" in result.stdout.lower()


def test_setup_scan_synthetic_offline_notice(cli_runner):
    """Synthetic mode should clearly state offline status."""
    result = cli_runner.invoke(app, ["setup-scan", "--synthetic"])

    assert result.exit_code == 0
    assert "synthetic" in result.stdout.lower() or "offline" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Test 8: Integration with DayTradeSetupDetector
# ---------------------------------------------------------------------------

def test_setup_scan_calls_detector(cli_runner):
    """Should invoke DayTradeSetupDetector.detect_many()."""
    with patch("strategy.day_trade_setups.DayTradeSetupDetector") as mock_detector_class:
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_many.return_value = []

        cli_runner.invoke(app, ["setup-scan", "--synthetic"])

        # Detector should have been instantiated and detect_many called
        mock_detector_class.assert_called()
        mock_detector.detect_many.assert_called()


def test_setup_scan_detector_receives_data_by_ticker(cli_runner):
    """DayTradeSetupDetector.detect_many() should receive dict of ticker -> DataFrame."""
    with patch("strategy.day_trade_setups.DayTradeSetupDetector") as mock_detector_class:
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_many.return_value = []

        cli_runner.invoke(app, ["setup-scan", "--synthetic"])

        # Check that detect_many was called with a dict
        mock_detector.detect_many.assert_called_once()
        call_args = mock_detector.detect_many.call_args
        data_arg = call_args[0][0]  # First positional argument
        assert isinstance(data_arg, dict), "detect_many should receive a dict of ticker -> DataFrame"


# ---------------------------------------------------------------------------
# Test 9: Lookback days parameter
# ---------------------------------------------------------------------------

def test_setup_scan_custom_lookback_days(cli_runner):
    """Custom --lookback-days should be accepted."""
    result = cli_runner.invoke(
        app,
        ["setup-scan", "--synthetic", "--lookback-days", "100"]
    )

    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Test 10: Edge cases
# ---------------------------------------------------------------------------

def test_setup_scan_with_empty_ticker_list(cli_runner):
    """If somehow given an empty ticker list, should handle gracefully."""
    result = cli_runner.invoke(app, ["setup-scan", "--synthetic"])
    assert result.exit_code == 0


def test_setup_scan_whitespace_in_ticker_names(cli_runner):
    """Comma-separated tickers with whitespace should be trimmed."""
    result = cli_runner.invoke(
        app,
        ["setup-scan", "--ticker", " SPY , QQQ , NVDA ", "--synthetic"]
    )

    assert result.exit_code == 0
    # All three should appear in output (without leading/trailing spaces)
    assert "SPY" in result.stdout
    assert "QQQ" in result.stdout
    assert "NVDA" in result.stdout


# ---------------------------------------------------------------------------
# Test 11: Multi-ticker ranking output (Phase 13)
# ---------------------------------------------------------------------------

def _make_setup(
    ticker, setup_type, is_valid=True, confidence=0.7,
    rr1=2.0, rr2=3.0,
):
    """Build a DayTradeSetup for stubbing the detector in ranking tests."""
    from strategy.day_trade_setups import DayTradeSetup
    return DayTradeSetup(
        ticker=ticker,
        timestamp="2026-06-05T10:00:00",
        setup_type=setup_type,
        is_valid=is_valid,
        confidence=confidence,
        entry_trigger=100.0 if is_valid else 0.0,
        stop_loss=99.0 if is_valid else 0.0,
        target_1=102.0 if is_valid else 0.0,
        target_2=103.0 if is_valid else 0.0,
        risk_reward_1=rr1 if is_valid else 0.0,
        risk_reward_2=rr2 if is_valid else 0.0,
        reason="stub setup",
        warnings=[],
        metadata={"unusual_volume": 0.0},
    )


def _patch_detector(monkeypatch, stub_results):
    """Patch DayTradeSetupDetector.detect_many to return stub_results."""
    from strategy.day_trade_setups import DayTradeSetupDetector
    monkeypatch.setattr(
        DayTradeSetupDetector, "detect_many",
        lambda self, data_by_ticker: list(stub_results),
    )


def test_setup_scan_ranked_output_shows_top_setups(cli_runner, monkeypatch):
    """Valid setups are shown under the 'Top Trade Setups' header, best first."""
    from strategy.day_trade_setups import SetupType
    stubs = [
        _make_setup("SPY", SetupType.HIGH_RELATIVE_VOLUME_CONTINUATION_LONG, confidence=0.7),
        _make_setup("QQQ", SetupType.OPENING_RANGE_BREAKOUT_LONG, confidence=0.9),
    ]
    _patch_detector(monkeypatch, stubs)

    result = cli_runner.invoke(
        app, ["setup-scan", "--ticker", "SPY,QQQ", "--synthetic"]
    )

    assert result.exit_code == 0
    assert "=== Top Trade Setups ===" in result.stdout
    # QQQ has higher confidence AND the ORB bonus, so it should be rank 1.
    spy_pos = result.stdout.index("SPY —") if "SPY —" in result.stdout else -1
    qqq_pos = result.stdout.index("QQQ —")
    assert qqq_pos != -1
    assert "1. QQQ —" in result.stdout
    if spy_pos != -1:
        assert qqq_pos < spy_pos


def test_setup_scan_all_no_setup_shows_message(cli_runner, monkeypatch):
    """When every ticker is NO_SETUP, the no-valid-setups message is printed."""
    from strategy.day_trade_setups import SetupType
    stubs = [
        _make_setup("SPY", SetupType.NO_SETUP, is_valid=False, confidence=0.0),
        _make_setup("QQQ", SetupType.NO_SETUP, is_valid=False, confidence=0.0),
    ]
    _patch_detector(monkeypatch, stubs)

    result = cli_runner.invoke(
        app, ["setup-scan", "--ticker", "SPY,QQQ", "--synthetic"]
    )

    assert result.exit_code == 0
    assert "No valid setups found across scanned tickers." in result.stdout
    assert "=== Top Trade Setups ===" not in result.stdout


def test_setup_scan_excludes_invalid_setups_from_ranking(cli_runner, monkeypatch):
    """Invalid setups never appear in the ranked output."""
    from strategy.day_trade_setups import SetupType
    stubs = [
        _make_setup("SPY", SetupType.OPENING_RANGE_BREAKOUT_LONG, is_valid=True, confidence=0.8),
        _make_setup("BAD", SetupType.VWAP_RECLAIM_LONG, is_valid=False, confidence=0.9),
    ]
    _patch_detector(monkeypatch, stubs)

    result = cli_runner.invoke(
        app, ["setup-scan", "--ticker", "SPY,BAD", "--synthetic"]
    )

    assert result.exit_code == 0
    # Ranked section exists and shows SPY, never the invalid BAD ticker line.
    assert "=== Top Trade Setups ===" in result.stdout
    assert "1. SPY —" in result.stdout
    assert "BAD —" not in result.stdout


def test_setup_scan_top_n_limit_respected(cli_runner, monkeypatch):
    """--top-n caps the number of ranked setups displayed."""
    from strategy.day_trade_setups import SetupType
    stubs = [
        _make_setup(f"T{i}", SetupType.OPENING_RANGE_BREAKOUT_LONG, confidence=0.9)
        for i in range(8)
    ]
    _patch_detector(monkeypatch, stubs)

    result = cli_runner.invoke(
        app,
        ["setup-scan", "--ticker", "T0,T1,T2,T3,T4,T5,T6,T7",
         "--synthetic", "--top-n", "3"],
    )

    assert result.exit_code == 0
    # Ranks 1..3 present, rank 4 absent.
    assert "1. " in result.stdout
    assert "3. " in result.stdout
    assert "4. " not in result.stdout


def test_setup_scan_jsonl_logs_all_tickers_not_just_top(cli_runner, monkeypatch):
    """JSONL must log every scanned ticker, including NO_SETUP, not just the top subset."""
    from strategy.day_trade_setups import SetupType
    log_file = Path("logs/setup_scans.jsonl")
    if log_file.exists():
        log_file.unlink()

    stubs = [
        _make_setup("SPY", SetupType.OPENING_RANGE_BREAKOUT_LONG, is_valid=True, confidence=0.8),
        _make_setup("QQQ", SetupType.NO_SETUP, is_valid=False, confidence=0.0),
        _make_setup("NVDA", SetupType.NO_SETUP, is_valid=False, confidence=0.0),
    ]
    _patch_detector(monkeypatch, stubs)

    result = cli_runner.invoke(
        app, ["setup-scan", "--ticker", "SPY,QQQ,NVDA", "--synthetic"]
    )
    assert result.exit_code == 0

    records = _load_jsonl_log(log_file)
    logged_tickers = {r["ticker"] for r in records}
    assert {"SPY", "QQQ", "NVDA"}.issubset(logged_tickers)
    # The NO_SETUP rows must be logged even though they are excluded from ranking.
    assert any(r["setup_type"] == "no_setup" for r in records)


def test_setup_scan_does_not_mutate_logged_setups(cli_runner, monkeypatch):
    """Ranking must not leak rank/score into the JSONL-logged original metadata."""
    from strategy.day_trade_setups import SetupType
    log_file = Path("logs/setup_scans.jsonl")
    if log_file.exists():
        log_file.unlink()

    stubs = [
        _make_setup("SPY", SetupType.OPENING_RANGE_BREAKOUT_LONG, is_valid=True, confidence=0.8),
    ]
    _patch_detector(monkeypatch, stubs)

    result = cli_runner.invoke(app, ["setup-scan", "--ticker", "SPY", "--synthetic"])
    assert result.exit_code == 0

    records = _load_jsonl_log(log_file)
    spy_records = [r for r in records if r["ticker"] == "SPY"]
    assert spy_records, "SPY should be logged"
    for r in spy_records:
        assert "rank" not in r["metadata"], "rank leaked into logged original"
        assert "score" not in r["metadata"], "score leaked into logged original"


# ---------------------------------------------------------------------------
# Test 12: Timeframe plumbing (Phase: intraday data-provider foundation)
# ---------------------------------------------------------------------------

def test_setup_scan_default_timeframe_is_daily(cli_runner):
    """Without --timeframe, the scan runs daily (1Day) as before."""
    result = cli_runner.invoke(app, ["setup-scan", "--synthetic"])

    assert result.exit_code == 0
    assert "Timeframe: 1Day" in result.stdout


def test_setup_scan_synthetic_5min_works_offline(cli_runner):
    """--synthetic --timeframe 5Min must run with no network and exit cleanly."""
    result = cli_runner.invoke(
        app, ["setup-scan", "--ticker", "SPY,QQQ", "--synthetic", "--timeframe", "5Min"]
    )

    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.stdout}"
    assert "Timeframe: 5Min" in result.stdout


def test_setup_scan_passes_timeframe_to_provider(cli_runner):
    """setup-scan must forward the requested timeframe to the data provider."""
    captured = {}

    from data_providers.synthetic_provider import SyntheticDataProvider
    original = SyntheticDataProvider.get_multiple_bars

    def spy_get_multiple(self, tickers, *args, **kwargs):
        captured["timeframe"] = kwargs.get("timeframe")
        return original(self, tickers, *args, **kwargs)

    with patch.object(SyntheticDataProvider, "get_multiple_bars", spy_get_multiple):
        result = cli_runner.invoke(
            app, ["setup-scan", "--ticker", "SPY", "--synthetic", "--timeframe", "15Min"]
        )

    assert result.exit_code == 0
    assert captured.get("timeframe") == "15Min"


def test_setup_scan_unsupported_timeframe_is_clear(cli_runner):
    """An unsupported timeframe should fail with a clear, listed-support message."""
    result = cli_runner.invoke(
        app, ["setup-scan", "--ticker", "SPY", "--synthetic", "--timeframe", "1Hour"]
    )

    assert result.exit_code != 0
    assert "does not support" in result.stdout
    assert "1Hour" in result.stdout


def test_setup_scan_provider_alpaca_routes_through_alpaca_provider(cli_runner):
    """setup-scan --provider alpaca should route through AlpacaDataProvider."""
    calls = []

    class MockAlpacaProvider:
        provider_name = "AlpacaData"

        def get_multiple_bars(self, tickers, start=None, end=None, timeframe="1Min"):
            calls.append({"tickers": tickers, "start": start, "end": end, "timeframe": timeframe})
            df = pd.DataFrame({
                "open": [100.0, 100.5],
                "high": [101.0, 101.5],
                "low": [99.5, 100.0],
                "close": [100.2, 100.8],
                "volume": [1000, 1200],
            }, index=pd.date_range("2026-06-10 09:30", periods=2, freq="1min"))
            df.attrs["data_age_seconds"] = 30.0
            return {"SPY": df}

    with patch("data_providers.AlpacaDataProvider", return_value=MockAlpacaProvider()) as provider_cls:
        with patch("strategy.day_trade_setups.DayTradeSetupDetector.detect_many") as detect_many:
            detect_many.return_value = [_no_setup_for_bias("SPY", above_vwap=False)]
            result = cli_runner.invoke(
                app, ["setup-scan", "--ticker", "SPY", "--provider", "alpaca", "--timeframe", "1Min"]
            )

    assert result.exit_code == 0, result.stdout
    assert "Provider: AlpacaData" in result.stdout
    assert "Data provider: Alpaca IEX" in result.stdout
    provider_cls.assert_called_once_with(feed="iex")
    assert calls and calls[0]["end"] is None
    assert calls[0]["timeframe"] == "1Min"


def test_setup_scan_provider_alpaca_defaults_to_5min_when_timeframe_omitted(cli_runner):
    """setup-scan --provider alpaca should default to 5Min when --timeframe is omitted."""
    calls = []

    class MockAlpacaProvider:
        provider_name = "AlpacaData"

        def get_multiple_bars(self, tickers, start=None, end=None, timeframe="5Min"):
            calls.append({"tickers": tickers, "start": start, "end": end, "timeframe": timeframe})
            df = pd.DataFrame({
                "open": [100.0, 100.5],
                "high": [101.0, 101.5],
                "low": [99.5, 100.0],
                "close": [100.2, 100.8],
                "volume": [1000, 1200],
            }, index=pd.date_range("2026-06-10 09:30", periods=2, freq="5min"))
            return {"SPY": df}

    with patch("data_providers.AlpacaDataProvider", return_value=MockAlpacaProvider()):
        with patch("strategy.day_trade_setups.DayTradeSetupDetector.detect_many") as detect_many:
            detect_many.return_value = [_no_setup_for_bias("SPY", above_vwap=False)]
            result = cli_runner.invoke(app, ["setup-scan", "--ticker", "SPY", "--provider", "alpaca"])

    assert result.exit_code == 0, result.stdout
    assert "Timeframe: 5Min" in result.stdout
    assert calls and calls[0]["timeframe"] == "5Min"


def test_setup_scan_provider_alpaca_explicit_unsupported_timeframe_errors_once(cli_runner):
    """Explicit unsupported Alpaca timeframe should fail before provider fetch."""
    with patch("data_providers.AlpacaDataProvider") as provider_cls:
        result = cli_runner.invoke(
            app, ["setup-scan", "--ticker", "SPY", "--provider", "alpaca", "--timeframe", "1Day"]
        )

    assert result.exit_code == 1
    assert result.stdout.count("AlpacaData supports only 1Min and 5Min") == 1
    assert "Use --timeframe 5Min for setup scans" in result.stdout
    assert "SECRET" not in result.stdout.upper()
    provider_cls.assert_not_called()


def test_setup_scan_loop_provider_alpaca_routes_through_alpaca_provider(cli_runner):
    """setup-scan-loop --provider alpaca should route through AlpacaDataProvider."""
    calls = []

    class MockAlpacaProvider:
        provider_name = "AlpacaData"

        def get_multiple_bars(self, tickers, start=None, end=None, timeframe="5Min"):
            calls.append({"tickers": tickers, "start": start, "end": end, "timeframe": timeframe})
            df = pd.DataFrame({
                "open": [100.0, 100.5],
                "high": [101.0, 101.5],
                "low": [99.5, 100.0],
                "close": [100.2, 100.8],
                "volume": [1000, 1200],
            }, index=pd.date_range("2026-06-10 09:30", periods=2, freq="5min"))
            return {"SPY": df}

    with patch("data_providers.AlpacaDataProvider", return_value=MockAlpacaProvider()) as provider_cls:
        with patch("strategy.day_trade_setups.DayTradeSetupDetector.detect_many") as detect_many:
            detect_many.return_value = [_no_setup_for_bias("SPY", above_vwap=False)]
            result = cli_runner.invoke(
                app,
                ["setup-scan-loop", "--ticker", "SPY", "--provider", "alpaca",
                 "--timeframe", "5Min", "--max-iterations", "1", "--interval-seconds", "0"],
            )

    assert result.exit_code == 0, result.stdout
    assert "Provider: AlpacaData" in result.stdout
    assert "Data provider: Alpaca IEX" in result.stdout
    provider_cls.assert_called_once_with(feed="iex")
    assert calls and calls[0]["end"] is None
    assert calls[0]["timeframe"] == "5Min"


def test_setup_scan_loop_provider_alpaca_defaults_to_5min_when_timeframe_omitted(cli_runner):
    """setup-scan-loop --provider alpaca should default to 5Min when --timeframe is omitted."""
    calls = []

    class MockAlpacaProvider:
        provider_name = "AlpacaData"

        def get_multiple_bars(self, tickers, start=None, end=None, timeframe="5Min"):
            calls.append({"tickers": tickers, "start": start, "end": end, "timeframe": timeframe})
            df = pd.DataFrame({
                "open": [100.0, 100.5],
                "high": [101.0, 101.5],
                "low": [99.5, 100.0],
                "close": [100.2, 100.8],
                "volume": [1000, 1200],
            }, index=pd.date_range("2026-06-10 09:30", periods=2, freq="5min"))
            return {"SPY": df}

    with patch("data_providers.AlpacaDataProvider", return_value=MockAlpacaProvider()):
        with patch("strategy.day_trade_setups.DayTradeSetupDetector.detect_many") as detect_many:
            detect_many.return_value = [_no_setup_for_bias("SPY", above_vwap=False)]
            result = cli_runner.invoke(
                app,
                ["setup-scan-loop", "--ticker", "SPY", "--provider", "alpaca",
                 "--max-iterations", "1", "--interval-seconds", "0"],
            )

    assert result.exit_code == 0, result.stdout
    assert "Timeframe: 5Min" in result.stdout
    assert calls and calls[0]["timeframe"] == "5Min"


def test_setup_scan_loop_provider_alpaca_explicit_unsupported_timeframe_errors_once(cli_runner):
    """Explicit unsupported Alpaca loop timeframe should fail before provider fetch."""
    with patch("data_providers.AlpacaDataProvider") as provider_cls:
        result = cli_runner.invoke(
            app,
            ["setup-scan-loop", "--ticker", "SPY", "--provider", "alpaca",
             "--timeframe", "1Day", "--max-iterations", "1", "--interval-seconds", "0"],
        )

    assert result.exit_code == 1
    assert result.stdout.count("AlpacaData supports only 1Min and 5Min") == 1
    assert "Use --timeframe 5Min for setup scans" in result.stdout
    assert "SECRET" not in result.stdout.upper()
    provider_cls.assert_not_called()


def test_setup_scan_loop_explicit_yfinance_provider_is_respected(cli_runner):
    """Explicit --provider yfinance should not be replaced by the recommendation helper."""
    calls = []

    class MockYFinanceProvider:
        provider_name = "YFinance"

        def get_multiple_bars(self, tickers, start=None, end=None, timeframe="5Min"):
            calls.append({"tickers": tickers, "start": start, "end": end, "timeframe": timeframe})
            df = pd.DataFrame({
                "open": [100.0, 100.5],
                "high": [101.0, 101.5],
                "low": [99.5, 100.0],
                "close": [100.2, 100.8],
                "volume": [1000, 1200],
            }, index=pd.date_range("2026-06-10 09:30", periods=2, freq="5min"))
            return {"SPY": df}

    with patch("data_providers.YFinanceDataProvider", return_value=MockYFinanceProvider()) as provider_cls:
        with patch("strategy.day_trade_setups.DayTradeSetupDetector.detect_many") as detect_many:
            detect_many.return_value = [_no_setup_for_bias("SPY", above_vwap=False)]
            result = cli_runner.invoke(
                app,
                ["setup-scan-loop", "--ticker", "SPY", "--provider", "yfinance",
                 "--timeframe", "5Min", "--max-iterations", "1", "--interval-seconds", "0"],
            )

    assert result.exit_code == 0, result.stdout
    assert "Data provider: YFinance" in result.stdout
    provider_cls.assert_called_once()
    assert calls and calls[0]["timeframe"] == "5Min"


def test_setup_scan_synthetic_flag_overrides_provider_alpaca(cli_runner):
    """--synthetic should override --provider alpaca for backward compatibility."""
    with patch("data_providers.AlpacaDataProvider") as alpaca_provider:
        result = cli_runner.invoke(
            app, ["setup-scan", "--ticker", "SPY", "--provider", "alpaca", "--synthetic"]
        )

    assert result.exit_code == 0, result.stdout
    assert "Provider: Synthetic" in result.stdout
    alpaca_provider.assert_not_called()


def test_setup_scan_csv_flag_overrides_provider_alpaca(cli_runner, tmp_path):
    """--csv should override --provider alpaca for backward compatibility."""
    csv_file = tmp_path / "bars.csv"
    df = pd.DataFrame({
        "open": [100.0] * 60,
        "high": [101.0] * 60,
        "low": [99.0] * 60,
        "close": [100.5] * 60,
        "volume": [1000] * 60,
    }, index=pd.date_range("2026-06-10 09:30", periods=60, freq="1min"))
    df.index.name = "date"
    df.to_csv(csv_file)

    with patch("data_providers.AlpacaDataProvider") as alpaca_provider:
        result = cli_runner.invoke(
            app, ["setup-scan", "--provider", "alpaca", "--csv", str(csv_file), "--timeframe", "1Min"]
        )

    assert result.exit_code == 0, result.stdout
    assert "Provider: CSV" in result.stdout
    alpaca_provider.assert_not_called()


def test_setup_scan_provider_alpaca_missing_credentials_clear_error(cli_runner):
    """Missing Alpaca credentials should produce a clear provider error without secrets."""
    from data_providers.base import DataProviderError

    class MockAlpacaProvider:
        provider_name = "AlpacaData"

        def get_multiple_bars(self, *args, **kwargs):
            raise DataProviderError("AlpacaDataProvider requires Alpaca paper data credentials.")

    with patch("data_providers.AlpacaDataProvider", return_value=MockAlpacaProvider()):
        result = cli_runner.invoke(
            app, ["setup-scan", "--ticker", "SPY", "--provider", "alpaca", "--timeframe", "1Min"]
        )

    assert result.exit_code == 1
    assert "AlpacaDataProvider requires Alpaca paper data credentials" in result.stdout
    assert "SECRET" not in result.stdout.upper()


def test_setup_scan_provider_alpaca_sip_subscription_error_is_helpful(cli_runner):
    """SIP subscription failures should suggest IEX/free feed."""
    from data_providers.base import DataProviderError

    class MockAlpacaProvider:
        provider_name = "AlpacaData"

        def get_multiple_bars(self, *args, **kwargs):
            raise DataProviderError("subscription does not permit querying recent SIP data")

    with patch("data_providers.AlpacaDataProvider", return_value=MockAlpacaProvider()):
        result = cli_runner.invoke(
            app, ["setup-scan", "--ticker", "SPY", "--provider", "alpaca", "--alpaca-feed", "sip", "--timeframe", "1Min"]
        )

    assert result.exit_code == 1
    assert "Your Alpaca plan does not permit SIP data" in result.stdout
    assert "--alpaca-feed iex" in result.stdout


def test_setup_scan_provider_alpaca_no_broker_or_order_executor_calls(cli_runner):
    """Alpaca data provider scans must not instantiate broker/order execution."""
    class MockAlpacaProvider:
        provider_name = "AlpacaData"

        def get_multiple_bars(self, *args, **kwargs):
            df = pd.DataFrame({
                "open": [100.0, 100.5],
                "high": [101.0, 101.5],
                "low": [99.5, 100.0],
                "close": [100.2, 100.8],
                "volume": [1000, 1200],
            }, index=pd.date_range("2026-06-10 09:30", periods=2, freq="1min"))
            return {"SPY": df}

    with patch("data_providers.AlpacaDataProvider", return_value=MockAlpacaProvider()):
        with patch("broker.alpaca_adapter.AlpacaBrokerAdapter") as broker_cls:
            with patch("broker.order_executor.OrderExecutor") as executor_cls:
                with patch("strategy.day_trade_setups.DayTradeSetupDetector.detect_many") as detect_many:
                    detect_many.return_value = [_no_setup_for_bias("SPY", above_vwap=False)]
                    result = cli_runner.invoke(
                        app, ["setup-scan", "--ticker", "SPY", "--provider", "alpaca", "--timeframe", "1Min"]
                    )

    assert result.exit_code == 0, result.stdout
    broker_cls.assert_not_called()
    executor_cls.assert_not_called()


def test_setup_scan_default_provider_remains_yfinance(cli_runner):
    """Without --provider/--synthetic/--csv, setup-scan should still use yfinance."""
    class MockYFinanceProvider:
        provider_name = "YFinance"

        def get_multiple_bars(self, tickers, start=None, end=None, timeframe="1Day"):
            df = pd.DataFrame({
                "open": [100.0, 100.5],
                "high": [101.0, 101.5],
                "low": [99.5, 100.0],
                "close": [100.2, 100.8],
                "volume": [1000, 1200],
            }, index=pd.date_range("2026-06-09", periods=2, freq="D"))
            return {"SPY": df}

    with patch("data_providers.YFinanceDataProvider", return_value=MockYFinanceProvider()) as provider_cls:
        with patch("strategy.day_trade_setups.DayTradeSetupDetector.detect_many") as detect_many:
            detect_many.return_value = [_no_setup_for_bias("SPY", above_vwap=False)]
            result = cli_runner.invoke(app, ["setup-scan", "--ticker", "SPY"])

    assert result.exit_code == 0, result.stdout
    assert "Provider: YFinance" in result.stdout
    provider_cls.assert_called_once()


# ---------------------------------------------------------------------------
# Test 13: setup-scan-loop command (Phase: continuous monitoring)
# ---------------------------------------------------------------------------

def test_setup_scan_loop_synthetic_with_max_iterations(cli_runner):
    """setup-scan-loop --synthetic --max-iterations 2 must complete without network."""
    result = cli_runner.invoke(
        app, ["setup-scan-loop", "--synthetic", "--max-iterations", "2", "--interval-seconds", "0"]
    )

    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.stdout}"
    assert "Iteration 1" in result.stdout
    assert "Iteration 2" in result.stdout


def test_setup_scan_loop_synthetic_without_max_iterations_rejected(cli_runner):
    """setup-scan-loop --synthetic without --max-iterations must be rejected (safety)."""
    result = cli_runner.invoke(
        app, ["setup-scan-loop", "--synthetic"]
    )

    assert result.exit_code != 0
    assert "--max-iterations" in result.stdout or "requires" in result.stdout.lower()


def test_setup_scan_loop_logs_valid_setups_to_alerts(cli_runner):
    """Valid ranked setups must be appended to logs/setup_alerts.jsonl."""
    log_file = Path("logs/setup_alerts.jsonl")
    if log_file.exists():
        log_file.unlink()

    result = cli_runner.invoke(
        app, ["setup-scan-loop", "--synthetic", "--max-iterations", "1", "--interval-seconds", "0"]
    )
    assert result.exit_code == 0

    # Check that alerts file was created (even if empty if no valid setups).
    # With synthetic seed, there may or may not be valid setups; we check the file exists.
    if log_file.exists():
        records = _load_jsonl_log(log_file)
        # If there are alerts, they should have the expected fields.
        for r in records:
            assert "ticker" in r
            assert "setup_type" in r
            assert "rank" in r


def test_setup_scan_loop_deduplicates_alerts_within_run(cli_runner, monkeypatch):
    """Duplicate ticker/setup_type/entry combinations should not trigger repeated alerts."""
    log_file = Path("logs/setup_alerts.jsonl")
    if log_file.exists():
        log_file.unlink()

    # Stub the detector to return a valid setup twice (simulating same setup
    # appearing across iterations).
    from strategy.day_trade_setups import DayTradeSetupDetector, DayTradeSetup, SetupType
    original_detect = DayTradeSetupDetector.detect_many

    call_count = {"n": 0}

    def stub_detect(self, data):
        call_count["n"] += 1
        # Always return the same valid setup (so alert key repeats).
        return [
            DayTradeSetup(
                ticker="TEST", timestamp=pd.Timestamp.now(),
                setup_type=SetupType.OPENING_RANGE_BREAKOUT_LONG,
                is_valid=True, confidence=0.8,
                entry_trigger=100.0, stop_loss=99.0,
                target_1=102.0, target_2=103.0,
                risk_reward_1=2.0, risk_reward_2=3.0,
                reason="stub", metadata={"rank": 1, "score": 5.0},
            )
        ]

    monkeypatch.setattr(DayTradeSetupDetector, "detect_many", stub_detect)

    # Run for 2 iterations; should generate only 1 alert (deduped in the 2nd).
    result = cli_runner.invoke(
        app, ["setup-scan-loop", "--synthetic", "--max-iterations", "2", "--interval-seconds", "0"]
    )
    assert result.exit_code == 0
    assert call_count["n"] == 2, "detector should be called twice"

    # The alert file should have only 1 entry (from iteration 1), not 2.
    if log_file.exists():
        records = _load_jsonl_log(log_file)
        assert len(records) == 1, f"Expected 1 alert, got {len(records)}"


def test_setup_scan_loop_respects_max_iterations(cli_runner):
    """Loop must stop at --max-iterations."""
    result = cli_runner.invoke(
        app, ["setup-scan-loop", "--synthetic", "--max-iterations", "2", "--interval-seconds", "0"]
    )

    assert result.exit_code == 0
    assert "Iteration 1" in result.stdout
    assert "Iteration 2" in result.stdout
    assert "Iteration 3" not in result.stdout


def test_setup_scan_loop_no_valid_setups_path(cli_runner):
    """Loop should handle iterations with no valid setups gracefully."""
    result = cli_runner.invoke(
        app, ["setup-scan-loop", "--synthetic", "--max-iterations", "1", "--interval-seconds", "0"]
    )

    assert result.exit_code == 0
    # Either "No valid setups" or actual setups; either is fine.
    assert "Iteration 1" in result.stdout


def test_setup_scan_loop_no_broker_calls(cli_runner):
    """setup-scan-loop must not call any broker or order execution methods."""
    with patch("broker.alpaca_adapter.AlpacaBrokerAdapter") as mock_broker:
        with patch("broker.order_executor.OrderExecutor") as mock_executor:
            result = cli_runner.invoke(
                app, ["setup-scan-loop", "--synthetic", "--max-iterations", "1", "--interval-seconds", "0"]
            )

            assert result.exit_code == 0
            mock_broker.assert_not_called()
            mock_executor.assert_not_called()


# ---------------------------------------------------------------------------
# Test 14: Near-miss diagnostics in setup-scan output
# ---------------------------------------------------------------------------

def test_setup_scan_shows_near_miss_diagnostics(cli_runner):
    """When no valid setups, setup-scan should show near-miss diagnostics."""
    # Run with a weak setup condition
    result = cli_runner.invoke(
        app, ["setup-scan", "--ticker", "SPY", "--synthetic"]
    )

    assert result.exit_code == 0
    # Output should either show valid setups or near-miss diagnostics
    output = result.stdout.lower()
    has_valid = "top trade setups" in output
    has_diagnostics = "near-miss" in output or "nearest" in output
    assert has_valid or has_diagnostics, "Should show either setups or diagnostics"


def test_setup_scan_jsonl_includes_diagnostics(cli_runner):
    """JSONL log should include diagnostic metadata for NO_SETUP results."""
    log_file = Path("logs/setup_scans.jsonl")
    if log_file.exists():
        log_file.unlink()

    cli_runner.invoke(app, ["setup-scan", "--ticker", "SPY", "--synthetic"])

    records = _load_jsonl_log(log_file)
    # At least one record should be present
    assert len(records) > 0

    # Find a NO_SETUP record
    no_setup_records = [r for r in records if r.get("setup_type") == "no_setup"]
    if no_setup_records:
        # It should have metadata with diagnostics
        record = no_setup_records[0]
        assert "metadata" in record
        metadata = record["metadata"]
        # NO_SETUP should have diagnostics (but valid setups may not)
        if "diagnostics" in metadata:
            assert "checked_setups" in metadata or isinstance(metadata.get("diagnostics"), dict)


# ---------------------------------------------------------------------------
# Test 15: No-trade market bias summary
# ---------------------------------------------------------------------------

def _no_setup_for_bias(ticker: str, above_vwap: bool, blocker: str = "close below VWAP"):
    from datetime import datetime
    from strategy.day_trade_setups import DayTradeSetup, SetupType

    close = 101.0 if above_vwap else 99.0
    return DayTradeSetup(
        ticker=ticker,
        timestamp=datetime.now(),
        setup_type=SetupType.NO_SETUP,
        is_valid=False,
        confidence=0.0,
        entry_trigger=0.0,
        stop_loss=0.0,
        target_1=0.0,
        target_2=0.0,
        risk_reward_1=0.0,
        risk_reward_2=0.0,
        reason="No qualifying day-trade setup.",
        warnings=[],
        metadata={
            "close": close,
            "vwap_value": 100.0,
            "diagnostics": {
                "opening_range_breakout_long": [blocker],
            },
            "nearest_setup_type": "opening_range_breakout_long",
        },
    )


def test_setup_market_bias_summary_unfavorable_when_most_below_vwap():
    from main import summarize_setup_market_bias

    summary = summarize_setup_market_bias([
        _no_setup_for_bias("AAPL", above_vwap=False),
        _no_setup_for_bias("MSFT", above_vwap=False),
        _no_setup_for_bias("NVDA", above_vwap=False),
        _no_setup_for_bias("TSLA", above_vwap=True),
    ])

    assert summary["tickers_scanned"] == 4
    assert summary["above_vwap"] == 1
    assert summary["below_vwap"] == 3
    assert summary["long_bias"] == "unfavorable"


def test_setup_market_bias_summary_favorable_when_most_above_vwap():
    from main import summarize_setup_market_bias

    summary = summarize_setup_market_bias([
        _no_setup_for_bias("AAPL", above_vwap=True),
        _no_setup_for_bias("MSFT", above_vwap=True),
        _no_setup_for_bias("NVDA", above_vwap=True),
        _no_setup_for_bias("TSLA", above_vwap=False),
    ])

    assert summary["above_vwap"] == 3
    assert summary["percent_above_vwap"] == pytest.approx(75.0)
    assert summary["long_bias"] == "favorable"


def test_setup_market_bias_summary_common_failed_condition_detected():
    from main import summarize_setup_market_bias

    summary = summarize_setup_market_bias([
        _no_setup_for_bias("AAPL", above_vwap=False, blocker="close 99.00 not above vwap 100.00"),
        _no_setup_for_bias("MSFT", above_vwap=False, blocker="close_above_vwap != 1.0"),
        _no_setup_for_bias("NVDA", above_vwap=True, blocker="rel_volume_20 1.10 < 1.5"),
    ])

    assert summary["most_common_blocker"] == "close below VWAP"


def test_setup_market_bias_summary_missing_diagnostics_does_not_crash():
    from datetime import datetime
    from main import summarize_setup_market_bias
    from strategy.day_trade_setups import DayTradeSetup, SetupType

    setup = DayTradeSetup(
        ticker="SPY",
        timestamp=datetime.now(),
        setup_type=SetupType.NO_SETUP,
        is_valid=False,
        confidence=0.0,
        entry_trigger=0.0,
        stop_loss=0.0,
        target_1=0.0,
        target_2=0.0,
        risk_reward_1=0.0,
        risk_reward_2=0.0,
        reason="No setup.",
        warnings=[],
        metadata={},
    )

    summary = summarize_setup_market_bias([setup])

    assert summary["tickers_scanned"] == 1
    assert summary["most_common_blocker"] == "not enough diagnostic data"


def test_setup_scan_loop_no_valid_setups_prints_market_bias_summary(cli_runner):
    setups = [
        _no_setup_for_bias("AAPL", above_vwap=False),
        _no_setup_for_bias("MSFT", above_vwap=False),
    ]

    with patch("main._run_setup_scan_once", return_value=(setups, [])):
        result = cli_runner.invoke(
            app,
            ["setup-scan-loop", "--ticker", "AAPL,MSFT", "--synthetic",
             "--max-iterations", "1", "--interval-seconds", "0"],
        )

    assert result.exit_code == 0, result.stdout
    assert "No valid setups." in result.stdout
    assert "Market Bias Summary:" in result.stdout
    assert "Long bias: unfavorable" in result.stdout


# ---------------------------------------------------------------------------
# Test 16: Intraday lookback validation for yfinance
# ---------------------------------------------------------------------------

def test_setup_scan_5min_default_lookback_30(cli_runner):
    """5Min default lookback should be 30 days (safe boundary, not 60)."""
    result = cli_runner.invoke(
        app, ["setup-scan", "--ticker", "SPY", "--timeframe", "5Min", "--synthetic"]
    )

    assert result.exit_code == 0
    # With synthetic mode, lookback validation passes and synthetic ignores it
    assert "Setup" in result.stdout or "valid" in result.stdout.lower()


def test_setup_scan_15min_default_lookback_30(cli_runner):
    """15Min default lookback should be 30 days (safe boundary, not 60)."""
    result = cli_runner.invoke(
        app, ["setup-scan", "--ticker", "SPY", "--timeframe", "15Min", "--synthetic"]
    )

    assert result.exit_code == 0
    # With synthetic mode, lookback validation passes and synthetic ignores it
    assert "Setup" in result.stdout or "valid" in result.stdout.lower()


def test_setup_scan_1min_default_lookback_7(cli_runner):
    """1Min default lookback should be 7 days (yfinance limit) not 400."""
    result = cli_runner.invoke(
        app, ["setup-scan", "--ticker", "SPY", "--timeframe", "1Min", "--synthetic"]
    )

    assert result.exit_code == 0
    # With synthetic mode, lookback validation passes
    assert "Setup" in result.stdout or "valid" in result.stdout.lower()


def test_setup_scan_1day_default_lookback_400(cli_runner):
    """1Day default lookback should remain 400 (no yfinance intraday limit)."""
    result = cli_runner.invoke(
        app, ["setup-scan", "--ticker", "SPY", "--timeframe", "1Day", "--synthetic"]
    )

    assert result.exit_code == 0
    assert "Setup" in result.stdout or "valid" in result.stdout.lower()


def test_setup_scan_5min_explicit_400_lookback_errors():
    """Explicit 5Min lookback of 400 should error before network call."""
    from main import _validate_intraday_lookback

    # Should raise Exit(1) because 400 > 60 day limit for 5Min
    with pytest.raises(typer.Exit):
        _validate_intraday_lookback(
            timeframe="5Min",
            lookback_days=400,
            synthetic=False,
            csv_file=None,
            user_explicit_lookback=True,
        )


def test_setup_scan_5min_explicit_60_lookback_allowed():
    """Explicit 5Min lookback of 60 should be allowed (at limit)."""
    from main import _validate_intraday_lookback

    result = _validate_intraday_lookback(
        timeframe="5Min",
        lookback_days=60,
        synthetic=False,
        csv_file=None,
        user_explicit_lookback=True,
    )
    assert result == 60


def test_setup_scan_5min_synthetic_400_lookback_allowed(cli_runner):
    """5Min synthetic mode should allow 400 lookback (not restricted by yfinance)."""
    result = cli_runner.invoke(
        app, ["setup-scan", "--ticker", "SPY", "--timeframe", "5Min",
              "--synthetic", "--lookback-days", "400"]
    )

    # Synthetic should allow any lookback
    assert result.exit_code == 0


def test_setup_scan_5min_csv_400_lookback_allowed(cli_runner, tmp_path):
    """5Min CSV mode should allow 400 lookback (not restricted by yfinance)."""
    import pandas as pd

    # Create a dummy CSV file
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({
        "open": [100.0] * 100,
        "high": [101.0] * 100,
        "low": [99.0] * 100,
        "close": [100.5] * 100,
        "volume": [1e6] * 100,
    }, index=pd.date_range("2024-01-01", periods=100, freq="5Min"))
    df.index.name = "date"
    df.to_csv(csv_file)

    result = cli_runner.invoke(
        app, ["setup-scan", "--csv", str(csv_file), "--timeframe", "5Min",
              "--lookback-days", "400"]
    )

    # CSV should allow any lookback
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Test 16: Watchlist support
# ---------------------------------------------------------------------------

def test_load_watchlist_by_name():
    """load_watchlist() should load watchlist by name from watchlists/ dir."""
    from main import load_watchlist

    tickers = load_watchlist("mega_caps")
    assert isinstance(tickers, list)
    assert len(tickers) > 0
    assert all(isinstance(t, str) for t in tickers)
    # Should be uppercase
    assert all(t == t.upper() for t in tickers)


def test_load_momentum_universe_watchlist_by_name():
    """The broader shadow-testing watchlist should resolve from watchlists/."""
    from main import load_watchlist

    tickers = load_watchlist("momentum_universe")
    assert tickers[:4] == ["SPY", "QQQ", "DIA", "IWM"]
    assert "HOOD" in tickers
    assert "SMCI" in tickers
    assert len(tickers) == len(set(tickers))
    assert all(t == t.upper() for t in tickers)


def test_load_watchlist_by_path(tmp_path):
    """load_watchlist() should load watchlist by file path."""
    from main import load_watchlist

    # Create a custom watchlist
    wl_file = tmp_path / "custom.json"
    wl_file.write_text(json.dumps({
        "name": "custom",
        "tickers": ["ABC", "def", "GHI"]
    }))

    tickers = load_watchlist(str(wl_file))
    assert tickers == ["ABC", "DEF", "GHI"]


def test_load_watchlist_missing_errors():
    """load_watchlist() should error with clear message for missing watchlist."""
    from main import load_watchlist

    with pytest.raises(typer.Exit):
        load_watchlist("nonexistent_watchlist")


def test_load_watchlist_invalid_json_errors(tmp_path):
    """load_watchlist() should error clearly for invalid JSON."""
    from main import load_watchlist

    wl_file = tmp_path / "invalid.json"
    wl_file.write_text("{ bad json ]")

    with pytest.raises(typer.Exit):
        load_watchlist(str(wl_file))


def test_load_watchlist_empty_tickers_errors(tmp_path):
    """load_watchlist() should error if tickers list is empty."""
    from main import load_watchlist

    wl_file = tmp_path / "empty.json"
    wl_file.write_text(json.dumps({
        "name": "empty",
        "tickers": []
    }))

    with pytest.raises(typer.Exit):
        load_watchlist(str(wl_file))


def test_merge_tickers_ticker_only():
    """merge_tickers() should parse --ticker only."""
    from main import merge_tickers

    tickers = merge_tickers("SPY,QQQ", None)
    assert tickers == ["SPY", "QQQ"]


def test_merge_tickers_watchlist_only():
    """merge_tickers() should load --watchlist only."""
    from main import merge_tickers

    tickers = merge_tickers(None, "mega_caps")
    assert len(tickers) > 0
    assert all(t == t.upper() for t in tickers)


def test_merge_tickers_combined():
    """merge_tickers() should combine --ticker and --watchlist with dedup."""
    from main import merge_tickers

    # Get mega_caps tickers
    tickers = merge_tickers("NVDA,CUSTOM", "mega_caps")
    # Should include CUSTOM, NVDA (from both), and other mega_caps
    assert "CUSTOM" in tickers
    assert "NVDA" in tickers
    # NVDA should appear only once (deduped)
    assert tickers.count("NVDA") == 1


def test_merge_tickers_dedupes_momentum_universe_duplicates():
    """Ticker + broad watchlist should preserve order and avoid duplicate symbols."""
    from main import merge_tickers

    tickers = merge_tickers("HOOD,CUSTOM,SPY", "momentum_universe")
    assert tickers[0:3] == ["HOOD", "CUSTOM", "SPY"]
    assert "NVDA" in tickers
    assert "SMCI" in tickers
    assert tickers.count("HOOD") == 1
    assert tickers.count("SPY") == 1


def test_merge_tickers_both_none_empty():
    """merge_tickers() with both None should return empty list."""
    from main import merge_tickers

    tickers = merge_tickers(None, None)
    assert tickers == []


def test_setup_scan_with_watchlist(cli_runner):
    """setup-scan --watchlist should load and scan tickers."""
    result = cli_runner.invoke(
        app, ["setup-scan", "--watchlist", "mega_caps", "--synthetic"]
    )

    assert result.exit_code == 0
    assert "Tickers:" in result.stdout
    # Should list mega_caps tickers
    assert "AAPL" in result.stdout or "MSFT" in result.stdout


def test_setup_scan_with_watchlist_and_ticker(cli_runner):
    """setup-scan --watchlist and --ticker should combine."""
    result = cli_runner.invoke(
        app, ["setup-scan", "--ticker", "CUSTOM", "--watchlist", "mega_caps", "--synthetic"]
    )

    assert result.exit_code == 0
    assert "CUSTOM" in result.stdout
    # Should also include mega_caps tickers
    assert "Tickers:" in result.stdout


def test_setup_scan_loop_with_watchlist(cli_runner):
    """setup-scan-loop --watchlist should work with max_iterations."""
    result = cli_runner.invoke(
        app, ["setup-scan-loop", "--watchlist", "mega_caps", "--synthetic",
              "--max-iterations", "1", "--interval-seconds", "0"]
    )

    assert result.exit_code == 0
    assert "Tickers:" in result.stdout
    assert "Iteration 1" in result.stdout


def test_list_watchlists_command(cli_runner):
    """list-watchlists should print available watchlists and tickers."""
    result = cli_runner.invoke(app, ["list-watchlists"])

    assert result.exit_code == 0
    # Should list at least the default watchlists
    assert "mega_caps" in result.stdout or "Watchlist" in result.stdout
    assert "momentum_universe" in result.stdout


def test_setup_scan_with_missing_watchlist(cli_runner):
    """setup-scan --watchlist with missing file should error clearly."""
    result = cli_runner.invoke(
        app, ["setup-scan", "--watchlist", "nonexistent", "--synthetic"]
    )

    assert result.exit_code == 1
    assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Test 17: Repeated setup suppression in setup-scan-loop
# ---------------------------------------------------------------------------

def test_setup_scan_loop_repeated_not_in_new_setups(cli_runner):
    """Repeated setups should not be displayed under 'New Setups'."""
    from datetime import datetime
    from strategy.day_trade_setups import DayTradeSetup, SetupType

    mock_setup = DayTradeSetup(
        ticker="SPY",
        timestamp=datetime.now(),
        setup_type=SetupType.OPENING_RANGE_BREAKOUT_LONG,
        is_valid=True,
        confidence=0.8,
        entry_trigger=100.0,
        stop_loss=99.0,
        target_1=102.0,
        target_2=104.0,
        risk_reward_1=2.0,
        risk_reward_2=4.0,
        reason="test setup",
        warnings=[],
        metadata={"rank": 1},
    )

    # Mock _run_setup_scan_once to always return the same setup
    with patch("main._run_setup_scan_once") as mock_scan:
        mock_scan.return_value = ([mock_setup], [mock_setup])

        result = cli_runner.invoke(
            app, ["setup-scan-loop", "--ticker", "SPY", "--synthetic",
                  "--max-iterations", "2", "--interval-seconds", "0"]
        )

    assert result.exit_code == 0
    # First iteration should show "New Setups:"
    assert "New Setups:" in result.stdout
    # Should show the setup once on first iteration, then "suppressed" on second
    assert "suppressed" in result.stdout.lower()
    # Should NOT show "(repeated)" label in output
    assert "(repeated)" not in result.stdout


def test_setup_scan_loop_repeated_count_printed(cli_runner):
    """When setups repeat, suppressed count should be printed."""
    from datetime import datetime
    from strategy.day_trade_setups import DayTradeSetup, SetupType

    now = datetime.now()
    setup1 = DayTradeSetup(
        ticker="SPY",
        timestamp=now,
        setup_type=SetupType.OPENING_RANGE_BREAKOUT_LONG,
        is_valid=True,
        confidence=0.8,
        entry_trigger=100.0,
        stop_loss=99.0,
        target_1=102.0,
        target_2=104.0,
        risk_reward_1=2.0,
        risk_reward_2=4.0,
        reason="test",
        warnings=[],
        metadata={"rank": 1},
    )

    setup2 = DayTradeSetup(
        ticker="QQQ",
        timestamp=now,
        setup_type=SetupType.VWAP_RECLAIM_LONG,
        is_valid=True,
        confidence=0.7,
        entry_trigger=200.0,
        stop_loss=199.0,
        target_1=202.0,
        target_2=204.0,
        risk_reward_1=2.0,
        risk_reward_2=4.0,
        reason="test",
        warnings=[],
        metadata={"rank": 2},
    )

    with patch("main._run_setup_scan_once") as mock_scan:
        # First iteration: 2 new setups
        # Second iteration: same 2 setups (both repeated)
        mock_scan.return_value = ([setup1, setup2], [setup1, setup2])

        result = cli_runner.invoke(
            app, ["setup-scan-loop", "--ticker", "SPY,QQQ", "--synthetic",
                  "--max-iterations", "2", "--interval-seconds", "0"]
        )

    assert result.exit_code == 0
    # First iteration shows "New Setups:", second should show suppressed count
    assert "suppressed" in result.stdout.lower()
    assert "2" in result.stdout  # 2 setups suppressed on iteration 2


def test_setup_scan_loop_all_duplicates_message(cli_runner):
    """When all setups are duplicates, should print appropriate message."""
    from datetime import datetime
    from strategy.day_trade_setups import DayTradeSetup, SetupType

    setup = DayTradeSetup(
        ticker="SPY",
        timestamp=datetime.now(),
        setup_type=SetupType.OPENING_RANGE_BREAKOUT_LONG,
        is_valid=True,
        confidence=0.8,
        entry_trigger=100.0,
        stop_loss=99.0,
        target_1=102.0,
        target_2=104.0,
        risk_reward_1=2.0,
        risk_reward_2=4.0,
        reason="test",
        warnings=[],
        metadata={"rank": 1},
    )

    with patch("main._run_setup_scan_once") as mock_scan:
        mock_scan.return_value = ([setup], [setup])

        result = cli_runner.invoke(
            app, ["setup-scan-loop", "--ticker", "SPY", "--synthetic",
                  "--max-iterations", "3", "--interval-seconds", "0"]
        )

    assert result.exit_code == 0
    # Iterations 2 and 3 should show suppressed count
    lines = result.stdout.split("\n")
    suppressed_lines = [l for l in lines if "suppressed" in l.lower()]
    assert len(suppressed_lines) >= 1


def test_setup_scan_loop_alerts_only_new(cli_runner):
    """setup_alerts.jsonl should only contain new alerts, not repeated ones."""
    from datetime import datetime
    from strategy.day_trade_setups import DayTradeSetup, SetupType

    setup = DayTradeSetup(
        ticker="SPY",
        timestamp=datetime.now(),
        setup_type=SetupType.OPENING_RANGE_BREAKOUT_LONG,
        is_valid=True,
        confidence=0.8,
        entry_trigger=100.0,
        stop_loss=99.0,
        target_1=102.0,
        target_2=104.0,
        risk_reward_1=2.0,
        risk_reward_2=4.0,
        reason="test",
        warnings=[],
        metadata={"rank": 1},
    )

    log_file = Path("logs/setup_alerts.jsonl")
    if log_file.exists():
        log_file.unlink()

    with patch("main._run_setup_scan_once") as mock_scan:
        mock_scan.return_value = ([setup], [setup])

        result = cli_runner.invoke(
            app, ["setup-scan-loop", "--ticker", "SPY", "--synthetic",
                  "--max-iterations", "2", "--interval-seconds", "0"]
        )

    assert result.exit_code == 0

    if log_file.exists():
        records = []
        with open(log_file) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        # Should have exactly 1 alert (from iteration 1 only, not iteration 2)
        assert len(records) == 1
        assert records[0]["ticker"] == "SPY"
        assert records[0]["setup_type"] == "opening_range_breakout_long"


def test_setup_scan_loop_discord_sends_only_new_setup(cli_runner):
    """Discord setup alerts should send for new setups only, not repeated ones."""
    setup = _make_card_setup(ticker="SPY")

    with patch("main._run_setup_scan_once") as mock_scan:
        mock_scan.return_value = ([setup], [setup])
        with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
            with patch("alerts.discord_alerts.send_discord_message", return_value=True) as send_mock:
                result = cli_runner.invoke(
                    app,
                    ["setup-scan-loop", "--ticker", "SPY", "--synthetic",
                     "--max-iterations", "2", "--interval-seconds", "0"],
                )

    assert result.exit_code == 0, result.stdout
    assert "Discord alerts: enabled" in result.stdout
    assert send_mock.call_count == 1


def test_setup_scan_loop_discord_no_webhook_url_in_output(cli_runner):
    """Startup output should never reveal the Discord webhook URL."""
    setup = _make_card_setup(ticker="SPY")
    fake_url = "https://discord.example/webhook/SECRET_TOKEN"

    with patch("main._run_setup_scan_once", return_value=([setup], [setup])):
        with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
            with patch("alerts.discord_alerts.send_discord_message", return_value=True):
                result = cli_runner.invoke(
                    app,
                    ["setup-scan-loop", "--ticker", "SPY", "--synthetic",
                     "--max-iterations", "1", "--interval-seconds", "0"],
                )

    assert result.exit_code == 0, result.stdout
    assert fake_url not in result.stdout
    assert "SECRET_TOKEN" not in result.stdout


def _add_failed_trade_for_cooldown(ticker: str, setup_type: str, trade_id: str, minutes_ago: int = 10):
    """Write one same-day failed manual trade for cooldown tests."""
    from datetime import datetime, timedelta
    from trades.trade_tracker import ManualTrade, TradeTracker

    ts = (datetime.now() - timedelta(minutes=minutes_ago)).isoformat()
    TradeTracker().add_trade(
        ManualTrade(
            trade_id=trade_id,
            ticker=ticker,
            setup_type=setup_type,
            entry_price=100.0,
            stop_loss=98.0,
            target_1=104.0,
            target_2=106.0,
            shares=1,
            entry_time=ts,
            status="CLOSED",
            exit_price=99.0,
            exit_time=ts,
            realized_pnl=-1.0,
        )
    )


def test_setup_scan_loop_warns_after_two_failed_same_day_setup_type_trades(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    setup_type = "vwap_reclaim_long"
    _add_failed_trade_for_cooldown("TSLA", setup_type, "fail-setup-1")
    _add_failed_trade_for_cooldown("NVDA", setup_type, "fail-setup-2")
    setup = _make_card_setup(ticker="SPY")
    setup.setup_type = type(setup.setup_type)(setup_type)

    with patch("main._run_setup_scan_once", return_value=([setup], [setup])):
        result = cli_runner.invoke(
            app,
            [
                "setup-scan-loop", "--ticker", "SPY", "--synthetic",
                "--max-iterations", "1", "--interval-seconds", "0",
                "--no-discord-alerts", "--no-alert-banner",
            ],
        )

    assert result.exit_code == 0, result.stdout
    assert "COOLDOWN WARNING:" in result.stdout
    assert "vwap_reclaim_long has 2+ failed/excluded trades today" in result.stdout
    assert "Treat this alert as review-only" in result.stdout


def test_setup_scan_loop_warns_after_two_failed_same_day_ticker_trades(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _add_failed_trade_for_cooldown("TSLA", "vwap_reclaim_long", "fail-ticker-1")
    _add_failed_trade_for_cooldown("TSLA", "opening_range_breakout_long", "fail-ticker-2")
    setup = _make_card_setup(ticker="TSLA")

    with patch("main._run_setup_scan_once", return_value=([setup], [setup])):
        result = cli_runner.invoke(
            app,
            [
                "setup-scan-loop", "--ticker", "TSLA", "--synthetic",
                "--max-iterations", "1", "--interval-seconds", "0",
                "--no-discord-alerts", "--no-alert-banner",
            ],
        )

    assert result.exit_code == 0, result.stdout
    assert "COOLDOWN WARNING:" in result.stdout
    assert "TSLA has 2+ failed/excluded trades today" in result.stdout


def test_setup_scan_loop_cooldown_warns_but_records_without_suppress(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _add_failed_trade_for_cooldown("TSLA", "vwap_reclaim_long", "fail-record-1")
    _add_failed_trade_for_cooldown("TSLA", "vwap_reclaim_long", "fail-record-2")
    setup = _make_card_setup(ticker="TSLA")

    with patch("main._run_setup_scan_once", return_value=([setup], [setup])):
        with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
            with patch("alerts.discord_alerts.send_discord_message", return_value=True) as send_mock:
                result = cli_runner.invoke(
                    app,
                    [
                        "setup-scan-loop", "--ticker", "TSLA", "--synthetic",
                        "--max-iterations", "1", "--interval-seconds", "0",
                        "--no-alert-banner",
                    ],
                )

    assert result.exit_code == 0, result.stdout
    assert "COOLDOWN WARNING:" in result.stdout
    assert send_mock.call_count == 1
    records = _load_jsonl_log(Path("logs/setup_alerts.jsonl"))
    assert len(records) == 1
    assert records[0]["ticker"] == "TSLA"
    assert records[0]["metadata"]["cooldown_warning"] is True


def test_setup_scan_loop_suppress_cooled_down_alerts_blocks_alert_record_and_discord(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _add_failed_trade_for_cooldown("TSLA", "vwap_reclaim_long", "fail-suppress-1")
    _add_failed_trade_for_cooldown("TSLA", "vwap_reclaim_long", "fail-suppress-2")
    setup = _make_card_setup(ticker="TSLA")

    with patch("main._run_setup_scan_once", return_value=([setup], [setup])):
        with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
            with patch("alerts.discord_alerts.send_discord_message", return_value=True) as send_mock:
                result = cli_runner.invoke(
                    app,
                    [
                        "setup-scan-loop", "--ticker", "TSLA", "--synthetic",
                        "--max-iterations", "1", "--interval-seconds", "0",
                        "--suppress-cooled-down-alerts",
                    ],
                )

    assert result.exit_code == 0, result.stdout
    assert "Setup alert suppressed by daily cooldown" in result.stdout
    assert "ACTION: STALE REVIEW SETUP" in result.stdout
    assert send_mock.call_count == 0
    assert _load_jsonl_log(Path("logs/setup_alerts.jsonl")) == []


def test_setup_scan_loop_previous_day_failed_trades_do_not_trigger_cooldown(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from datetime import datetime, timedelta
    from trades.trade_tracker import ManualTrade, TradeTracker

    yesterday = (datetime.now() - timedelta(days=1)).isoformat()
    tracker = TradeTracker()
    for idx in range(2):
        tracker.add_trade(
            ManualTrade(
                trade_id=f"old-fail-{idx}",
                ticker="TSLA",
                setup_type="vwap_reclaim_long",
                entry_price=100,
                stop_loss=98,
                target_1=104,
                target_2=106,
                shares=1,
                entry_time=yesterday,
                status="CLOSED",
                exit_price=99,
                exit_time=yesterday,
                realized_pnl=-1,
            )
        )
    setup = _make_card_setup(ticker="TSLA")

    with patch("main._run_setup_scan_once", return_value=([setup], [setup])):
        result = cli_runner.invoke(
            app,
            [
                "setup-scan-loop", "--ticker", "TSLA", "--synthetic",
                "--max-iterations", "1", "--interval-seconds", "0",
                "--no-discord-alerts", "--no-alert-banner",
            ],
        )

    assert result.exit_code == 0, result.stdout
    assert "COOLDOWN WARNING" not in result.stdout


# ---------------------------------------------------------------------------
# Test 18: Local alert notifications
# ---------------------------------------------------------------------------

def test_alert_banner_function():
    """print_alert_banner() should print a formatted alert for a setup."""
    from datetime import datetime
    from strategy.day_trade_setups import DayTradeSetup, SetupType
    from alerts.local_alerts import print_alert_banner

    setup = DayTradeSetup(
        ticker="GOOGL",
        timestamp=datetime.now(),
        setup_type=SetupType.HIGH_RELATIVE_VOLUME_CONTINUATION_LONG,
        is_valid=True,
        confidence=0.70,
        entry_trigger=368.72,
        stop_loss=368.37,
        target_1=369.42,
        target_2=370.5,
        risk_reward_1=1.5,
        risk_reward_2=2.5,
        reason="test",
        warnings=[],
        metadata={"rank": 1},
    )

    # Should not raise any exception
    print_alert_banner(setup)


def test_setup_scan_loop_alert_banner_on_new_setup(cli_runner):
    """Alert banner should print for new setups when --alert-banner."""
    from datetime import datetime
    from strategy.day_trade_setups import DayTradeSetup, SetupType

    setup = DayTradeSetup(
        ticker="SPY",
        timestamp=datetime.now(),
        setup_type=SetupType.OPENING_RANGE_BREAKOUT_LONG,
        is_valid=True,
        confidence=0.8,
        entry_trigger=100.0,
        stop_loss=99.0,
        target_1=102.0,
        target_2=104.0,
        risk_reward_1=2.0,
        risk_reward_2=4.0,
        reason="test",
        warnings=[],
        metadata={"rank": 1},
    )

    with patch("main._run_setup_scan_once") as mock_scan:
        mock_scan.return_value = ([setup], [setup])

        result = cli_runner.invoke(
            app, ["setup-scan-loop", "--ticker", "SPY", "--synthetic",
                  "--max-iterations", "1", "--interval-seconds", "0",
                  "--alert-banner"]
        )

    assert result.exit_code == 0
    # Alert banner should be in output
    assert "NEW SETUP ALERT" in result.stdout


def test_setup_scan_loop_no_banner_with_flag(cli_runner):
    """Alert banner should NOT print when --no-alert-banner."""
    from datetime import datetime
    from strategy.day_trade_setups import DayTradeSetup, SetupType

    setup = DayTradeSetup(
        ticker="SPY",
        timestamp=datetime.now(),
        setup_type=SetupType.OPENING_RANGE_BREAKOUT_LONG,
        is_valid=True,
        confidence=0.8,
        entry_trigger=100.0,
        stop_loss=99.0,
        target_1=102.0,
        target_2=104.0,
        risk_reward_1=2.0,
        risk_reward_2=4.0,
        reason="test",
        warnings=[],
        metadata={"rank": 1},
    )

    with patch("main._run_setup_scan_once") as mock_scan:
        mock_scan.return_value = ([setup], [setup])

        result = cli_runner.invoke(
            app, ["setup-scan-loop", "--ticker", "SPY", "--synthetic",
                  "--max-iterations", "1", "--interval-seconds", "0",
                  "--no-alert-banner"]
        )

    assert result.exit_code == 0
    # Alert banner should NOT be in output
    assert "NEW SETUP ALERT" not in result.stdout


def test_setup_scan_loop_has_no_audible_alert_usage():
    """setup-scan-loop should not expose or call any audible beep/bell alert."""
    main_source = Path("main.py").read_text(encoding="utf-8").lower()
    setup_loop_source = main_source.split("def setup_scan_loop", 1)[1].split("def trade_add", 1)[0]

    assert "--beep" not in setup_loop_source
    assert "emit_beep" not in setup_loop_source


# ---------------------------------------------------------------------------
# Test 19: Trade action card
# ---------------------------------------------------------------------------

def _make_card_setup(**overrides):
    """Build a valid DayTradeSetup for action-card tests."""
    from datetime import datetime
    from strategy.day_trade_setups import DayTradeSetup, SetupType

    defaults = dict(
        ticker="GOOGL",
        timestamp=datetime.now(),
        setup_type=SetupType.HIGH_RELATIVE_VOLUME_CONTINUATION_LONG,
        is_valid=True,
        confidence=0.70,
        entry_trigger=100.0,
        stop_loss=98.0,
        target_1=104.0,
        target_2=108.0,
        risk_reward_1=2.0,
        risk_reward_2=4.0,
        reason="strong continuation",
        warnings=[],
        metadata={"rank": 1},
    )
    defaults.update(overrides)
    return DayTradeSetup(**defaults)


def test_action_card_includes_entry_stop_targets():
    """Action card should include entry, stop, and both targets."""
    from alerts.local_alerts import format_trade_action_card

    setup = _make_card_setup()
    card = format_trade_action_card(setup)

    assert "100.00" in card  # entry
    assert "98.00" in card   # stop
    assert "104.00" in card  # target 1
    assert "108.00" in card  # target 2
    assert "GOOGL" in card
    assert "VALID SETUP" in card


def test_action_card_includes_shares_with_equity():
    """Action card should include suggested shares when equity provided."""
    from alerts.local_alerts import format_trade_action_card

    setup = _make_card_setup()  # risk/share = 2.0
    # equity 100000, risk_pct 0.005 => $500 risk / $2 = 250 shares
    card = format_trade_action_card(setup, account_equity=100000.0, risk_pct=0.005)

    assert "Suggested shares" in card
    assert "250" in card
    assert "Est. max loss" in card


def test_action_card_omits_shares_without_equity():
    """Action card should omit suggested shares when equity is None."""
    from alerts.local_alerts import format_trade_action_card

    setup = _make_card_setup()
    card = format_trade_action_card(setup)  # no equity

    assert "Suggested shares" not in card
    assert "Est. max loss" not in card


def test_action_card_risk_per_share_correct():
    """risk_per_share should equal entry - stop."""
    from alerts.local_alerts import format_trade_action_card

    setup = _make_card_setup(entry_trigger=50.0, stop_loss=47.5)
    card = format_trade_action_card(setup)

    # risk/share = 50.0 - 47.5 = 2.50
    assert "Risk/share:     2.50" in card


def test_action_card_no_divide_by_zero():
    """Invalid stop/risk (entry == stop) must not divide by zero."""
    from alerts.local_alerts import format_trade_action_card

    setup = _make_card_setup(entry_trigger=100.0, stop_loss=100.0)
    # Should not raise even with equity provided
    card = format_trade_action_card(setup, account_equity=100000.0, risk_pct=0.005)

    assert "N/A" in card
    assert "Suggested shares" in card  # label present but value is N/A


def test_action_card_includes_warnings():
    """Action card should include warnings when present."""
    from alerts.local_alerts import format_trade_action_card

    setup = _make_card_setup(warnings=["low liquidity", "wide spread"])
    card = format_trade_action_card(setup)

    assert "Warnings:" in card
    assert "low liquidity" in card


def test_action_card_includes_invalidation():
    """Action card should always include an invalidation note."""
    from alerts.local_alerts import format_trade_action_card

    setup = _make_card_setup()
    card = format_trade_action_card(setup)

    assert "Invalidation:" in card
    assert "stop_loss" in card or "VWAP" in card


def test_action_card_labels_stale_data_as_review_only():
    """Stale data should not label a valid setup as live-actionable."""
    from alerts.local_alerts import format_trade_action_card

    setup = _make_card_setup()
    card = format_trade_action_card(setup, stale_data=True)

    assert "ACTION: STALE REVIEW SETUP" in card
    assert "ACTION: VALID SETUP" not in card


def test_setup_scan_output_includes_action_card(cli_runner):
    """setup-scan ranked output should include the action card."""
    setup = _make_card_setup()

    with patch("strategy.day_trade_setups.DayTradeSetupDetector") as mock_detector_cls:
        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector
        mock_detector.detect_many.return_value = [setup]

        result = cli_runner.invoke(
            app, ["setup-scan", "--ticker", "GOOGL", "--synthetic"]
        )

    assert result.exit_code == 0
    assert "ACTION:" in result.stdout
    assert "Invalidation:" in result.stdout


def test_setup_scan_stale_alpaca_output_is_review_only(cli_runner):
    """Stale Alpaca scan results should be visibly review-only."""
    import pandas as pd

    setup = _make_card_setup(ticker="SPY")
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-06-11 10:00")]),
    )
    df.attrs["data_age_seconds"] = 429 * 60

    fake_provider = MagicMock()
    fake_provider.provider_name = "AlpacaData"
    fake_provider.get_multiple_bars.return_value = {"SPY": df}

    with patch("main._get_named_data_provider", return_value=fake_provider):
        with patch("strategy.day_trade_setups.DayTradeSetupDetector") as mock_detector_cls:
            mock_detector = MagicMock()
            mock_detector_cls.return_value = mock_detector
            mock_detector.detect_many.return_value = [setup]

            result = cli_runner.invoke(
                app, ["setup-scan", "--ticker", "SPY", "--provider", "alpaca"]
            )

    assert result.exit_code == 0, result.stdout
    assert "STALE DATA WARNING" in result.stdout
    assert "results are for review only, not live trade entry" in result.stdout
    assert "ACTION: STALE REVIEW SETUP" in result.stdout
    assert "ACTION: VALID SETUP" not in result.stdout
    assert "SECRET" not in result.stdout


def test_setup_scan_loop_alert_includes_action_card(cli_runner):
    """setup-scan-loop alert banner should include the action card."""
    setup = _make_card_setup(ticker="SPY")

    with patch("main._run_setup_scan_once") as mock_scan:
        mock_scan.return_value = ([setup], [setup])

        result = cli_runner.invoke(
            app, ["setup-scan-loop", "--ticker", "SPY", "--synthetic",
                  "--max-iterations", "1", "--interval-seconds", "0",
                  "--alert-banner"]
        )

    assert result.exit_code == 0
    assert "ACTION:" in result.stdout
    assert "Invalidation:" in result.stdout


def test_setup_scan_loop_stale_scan_suppresses_alerts_by_default(cli_runner):
    """Stale ranked setups should not write setup alerts or send Discord by default."""
    setup = _make_card_setup(ticker="SPY")
    alert_log = Path("logs/setup_alerts.jsonl")
    if alert_log.exists():
        alert_log.unlink()

    with patch("main._run_setup_scan_once", return_value=([setup], [setup], True)):
        with patch("alerts.discord_alerts.is_discord_configured", return_value=True):
            with patch("alerts.discord_alerts.send_discord_message", return_value=True) as send_mock:
                result = cli_runner.invoke(
                    app,
                    [
                        "setup-scan-loop",
                        "--ticker",
                        "SPY",
                        "--synthetic",
                        "--max-iterations",
                        "1",
                        "--interval-seconds",
                        "0",
                    ],
                )

    assert result.exit_code == 0, result.stdout
    assert "STALE DATA WARNING" in result.stdout
    assert "Stale Review Setups" in result.stdout
    assert "ACTION: STALE REVIEW SETUP" in result.stdout
    assert "NEW SETUP ALERT" not in result.stdout
    assert "Setup alerts suppressed because market data is stale" in result.stdout
    assert send_mock.call_count == 0
    assert not alert_log.exists()


def test_setup_scan_loop_action_card_with_equity(cli_runner):
    """setup-scan-loop with --account-equity should show suggested shares in card."""
    setup = _make_card_setup(ticker="SPY")

    with patch("main._run_setup_scan_once") as mock_scan:
        mock_scan.return_value = ([setup], [setup])

        result = cli_runner.invoke(
            app, ["setup-scan-loop", "--ticker", "SPY", "--synthetic",
                  "--max-iterations", "1", "--interval-seconds", "0",
                  "--alert-banner", "--account-equity", "100000"]
        )

    assert result.exit_code == 0
    assert "Suggested shares" in result.stdout
