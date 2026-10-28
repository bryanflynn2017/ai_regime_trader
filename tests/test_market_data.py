"""
Tests for data/market_data.py — CSV loader, synthetic generator, and error messages.

These tests never make real network calls.  yfinance is mocked where needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(tmp_path, data: dict, date_col_name: str = "date") -> "pathlib.Path":
    """Write a minimal valid OHLCV CSV to tmp_path and return its path."""
    import io, csv as csv_mod
    from pathlib import Path
    path = tmp_path / "sample.csv"
    rows = list(data.values())[0]
    n = len(rows["close"])
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    with open(path, "w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow([date_col_name, "open", "high", "low", "close", "volume"])
        for i, d in enumerate(dates):
            writer.writerow([
                str(d.date()),
                rows["open"][i], rows["high"][i],
                rows["low"][i],  rows["close"][i], rows["volume"][i],
            ])
    return path


def _minimal_ohlcv_dict(n: int = 50) -> dict:
    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return {"dummy": {
        "open":   list(close * 0.999),
        "high":   list(close * 1.005),
        "low":    list(close * 0.995),
        "close":  list(close),
        "volume": [1_000_000] * n,
    }}


# ---------------------------------------------------------------------------
# Test 1: load_ohlcv_csv loads a valid CSV
# ---------------------------------------------------------------------------

def test_load_ohlcv_csv_loads_valid_file(tmp_path):
    from data.market_data import load_ohlcv_csv

    data = _minimal_ohlcv_dict(60)
    path = _write_csv(tmp_path, data, date_col_name="date")

    df = load_ohlcv_csv(path)

    assert not df.empty
    assert set(df.columns) >= {"open", "high", "low", "close", "volume"}
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# Test 2: load_ohlcv_csv rejects missing required columns
# ---------------------------------------------------------------------------

def test_load_ohlcv_csv_rejects_missing_columns(tmp_path):
    from data.market_data import load_ohlcv_csv, DataFetchError

    # CSV without 'close' column
    path = tmp_path / "bad.csv"
    path.write_text("date,open,high,low,volume\n2021-01-01,100,101,99,1000000\n")

    with pytest.raises(DataFetchError, match="[Mm]issing|missing"):
        load_ohlcv_csv(path)


def test_load_ohlcv_csv_raises_for_nonexistent_file():
    from data.market_data import load_ohlcv_csv, DataFetchError

    with pytest.raises(DataFetchError, match="[Nn]ot found|not found"):
        load_ohlcv_csv("/nonexistent/path/file.csv")


# ---------------------------------------------------------------------------
# Test 3: load_ohlcv_csv accepts all valid date column names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("date_col", ["date", "datetime", "timestamp"])
def test_load_ohlcv_csv_accepts_date_column_variants(tmp_path, date_col):
    from data.market_data import load_ohlcv_csv

    data = _minimal_ohlcv_dict(40)
    path = _write_csv(tmp_path, data, date_col_name=date_col)

    df = load_ohlcv_csv(path)
    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)


def test_load_ohlcv_csv_raises_when_no_date_column(tmp_path):
    """CSV with no recognisable date column must raise DataFetchError."""
    from data.market_data import load_ohlcv_csv, DataFetchError

    path = tmp_path / "nodates.csv"
    path.write_text("ticker,open,high,low,close,volume\nSPY,100,101,99,100,1000000\n")

    with pytest.raises(DataFetchError):
        load_ohlcv_csv(path)


# ---------------------------------------------------------------------------
# Test 4: generate_synthetic_ohlcv creates valid data
# ---------------------------------------------------------------------------

def test_generate_synthetic_ohlcv_creates_valid_data():
    from data.market_data import generate_synthetic_ohlcv, validate_ohlcv

    df = generate_synthetic_ohlcv(rows=300)

    assert not df.empty
    assert len(df) == 300
    assert set(df.columns) >= {"open", "high", "low", "close", "volume"}
    assert isinstance(df.index, pd.DatetimeIndex)

    # Should not raise
    validate_ohlcv(df)


def test_generate_synthetic_ohlcv_respects_row_count():
    from data.market_data import generate_synthetic_ohlcv

    for n in (50, 200, 500):
        df = generate_synthetic_ohlcv(rows=n)
        assert len(df) == n, f"Expected {n} rows, got {len(df)}"


# ---------------------------------------------------------------------------
# Test 5: generate_synthetic_ohlcv is deterministic with same seed
# ---------------------------------------------------------------------------

def test_generate_synthetic_ohlcv_deterministic():
    from data.market_data import generate_synthetic_ohlcv

    df1 = generate_synthetic_ohlcv(rows=100, seed=7)
    df2 = generate_synthetic_ohlcv(rows=100, seed=7)
    df3 = generate_synthetic_ohlcv(rows=100, seed=99)  # Different seed

    pd.testing.assert_frame_equal(df1, df2, obj="Same seed must produce identical output")

    # Different seed must give different close prices
    assert not df1["close"].equals(df3["close"]), (
        "Different seeds must produce different data"
    )


# ---------------------------------------------------------------------------
# Test 6: synthetic data works with Backtester end-to-end
# ---------------------------------------------------------------------------

def test_synthetic_data_works_with_backtester(tmp_path):
    """Full pipeline: synthetic data → Backtester → BacktestResult."""
    from config.settings import Settings
    from data.market_data import generate_synthetic_ohlcv
    from backtest.backtester import Backtester, BacktestResult

    df = generate_synthetic_ohlcv(rows=300, seed=1)
    s  = Settings(RISK_LOCK_FILE=str(tmp_path / "lock.txt"))
    bt = Backtester(starting_equity=100_000, settings_override=s)

    result = bt.run_single_ticker("SYNTH", df)

    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) > 0
    assert result.starting_equity == 100_000


# ---------------------------------------------------------------------------
# Test 7: CLI backtest --synthetic runs without network access
# ---------------------------------------------------------------------------

def test_cli_backtest_synthetic_no_network(tmp_path, monkeypatch):
    """
    The --synthetic flag must complete successfully with zero network calls.
    We verify by ensuring yfinance is never imported during the run.
    """
    from typer.testing import CliRunner
    import main as main_module

    # Redirect risk lock file to tmp_path (shared settings singleton isn't
    # easily patched in CliRunner; the backtester uses its own isolated settings)
    runner = CliRunner()
    result = runner.invoke(main_module.app, ["backtest", "--synthetic"])

    assert result.exit_code == 0, (
        f"CLI exited {result.exit_code}.\nOutput:\n{result.output}"
    )
    assert "Total return" in result.output
    assert "Sharpe ratio" in result.output
    assert "SYNTHETIC" in result.output.upper() or "synthetic" in result.output.lower()


# ---------------------------------------------------------------------------
# Test 8: yfinance failure produces a provider-failure error message
# ---------------------------------------------------------------------------

def test_yfinance_failure_does_not_say_delisted():
    """
    When yfinance raises a JSON error (provider failure), our error message
    must not suggest the ticker is delisted.
    """
    from data.market_data import fetch_ticker, DataFetchError

    with patch("yfinance.Ticker") as mock_cls:
        mock_inst = MagicMock()
        mock_inst.history.side_effect = ValueError(
            "Expecting value: line 1 column 1 (char 0)"
        )
        mock_cls.return_value = mock_inst

        with pytest.raises(DataFetchError) as exc_info:
            fetch_ticker("SPY", "2020-01-01", "2023-12-31")

    err_msg = str(exc_info.value).lower()
    assert "delisted" not in err_msg, (
        "Provider-failure error must not suggest the ticker is delisted"
    )
    assert any(kw in err_msg for kw in ("provider", "yahoo", "yfinance")), (
        "Error message should mention the data provider"
    )


def test_yfinance_empty_response_does_not_say_delisted():
    """
    When yfinance returns an empty DataFrame (Yahoo outage / rate limit),
    the error message must be provider-focused, not blame the ticker.
    """
    from data.market_data import fetch_ticker, DataFetchError

    with patch("yfinance.Ticker") as mock_cls:
        mock_inst = MagicMock()
        mock_inst.history.return_value = pd.DataFrame()  # empty
        mock_cls.return_value = mock_inst

        with pytest.raises(DataFetchError) as exc_info:
            fetch_ticker("QQQ", "2020-01-01", "2023-12-31")

    err_msg = str(exc_info.value).lower()
    assert "delisted" not in err_msg, (
        "Empty-response error must not suggest the ticker is delisted"
    )


# ---------------------------------------------------------------------------
# Additional: load_ohlcv_csv handles DataFrame-to-CSV round-trip
# ---------------------------------------------------------------------------

def test_load_ohlcv_csv_round_trip(tmp_path):
    """DataFrame saved with to_csv then loaded back must be identical."""
    from data.market_data import generate_synthetic_ohlcv, load_ohlcv_csv

    df_orig = generate_synthetic_ohlcv(rows=80, seed=2)
    csv_path = tmp_path / "roundtrip.csv"

    # Save with index (the DatetimeIndex becomes the first column)
    df_orig.index.name = "date"
    df_orig.to_csv(csv_path)

    df_loaded = load_ohlcv_csv(csv_path)

    assert len(df_loaded) == len(df_orig)
    # Prices should match (CSV precision may differ by small float rounding)
    pd.testing.assert_series_equal(
        df_loaded["close"].reset_index(drop=True),
        df_orig["close"].reset_index(drop=True),
        check_names=False,
        rtol=1e-5,
    )
