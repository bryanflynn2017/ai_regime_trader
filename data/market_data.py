"""
Market data fetching and validation layer.

Primary source: yfinance (backtesting / no credentials needed).
Alternative sources:
  - load_ohlcv_csv()         : load from a user-supplied CSV file
  - generate_synthetic_ohlcv(): deterministic random walk for smoke testing
Future source:
  - Alpaca data API (Phase 7)

Design rules:
- Never return a DataFrame that fails validate_ohlcv().
- Raise DataFetchError with a human-readable reason on any failure.
- Never silently swallow exceptions that affect data integrity.
- Sort by date ascending; remove duplicate dates.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The only columns we care about — everything else (Dividends, etc.) is dropped.
REQUIRED_COLUMNS: frozenset = frozenset({"open", "high", "low", "close", "volume"})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DataFetchError(Exception):
    """
    Raised when market data cannot be fetched or fails validation.

    Always includes a human-readable message explaining what went wrong.
    Never silently absorbed — the caller must handle or propagate it.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with all column names lowercased and stripped.

    Handles pandas MultiIndex columns (returned by yfinance for some queries)
    by taking the first level only.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower().strip() for col in df.columns]
    else:
        df.columns = [str(col).lower().strip() for col in df.columns]
    return df


def _strip_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """Remove timezone info from a DatetimeIndex, returning a tz-naive index."""
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


# ---------------------------------------------------------------------------
# Public: validation
# ---------------------------------------------------------------------------

def validate_ohlcv(df: pd.DataFrame) -> None:
    """
    Validate that df is a well-formed OHLCV dataset.

    Checks (in order):
    1. Not empty.
    2. All required columns present (open, high, low, close, volume).
    3. All price columns strictly positive.
    4. Volume is non-negative.
    5. high >= low for every row.

    Raises:
        DataFetchError: On the first validation failure, with a descriptive message.

    Note:
        Does NOT check that close is between low and high because adjusted-price
        data (auto_adjust=True from yfinance) can technically violate this.
    """
    if df is None or df.empty:
        raise DataFetchError("DataFrame is empty — no data was returned.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataFetchError(
            f"Missing required columns: {sorted(missing)}. "
            f"Present columns: {sorted(df.columns.tolist())}"
        )

    for col in ("open", "high", "low", "close"):
        n_bad = int((df[col] <= 0).sum())
        if n_bad > 0:
            raise DataFetchError(
                f"Column '{col}' contains {n_bad} non-positive value(s). "
                "All price columns must be strictly positive (> 0)."
            )

    n_bad_vol = int((df["volume"] < 0).sum())
    if n_bad_vol > 0:
        raise DataFetchError(
            f"Column 'volume' contains {n_bad_vol} negative value(s). "
            "Volume must be >= 0."
        )

    n_bad_hl = int((df["high"] < df["low"]).sum())
    if n_bad_hl > 0:
        raise DataFetchError(
            f"'high' < 'low' in {n_bad_hl} row(s). "
            "This indicates corrupted or malformed OHLC data."
        )


# ---------------------------------------------------------------------------
# Internal: post-fetch cleaning
# ---------------------------------------------------------------------------

def _clean_fetched(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize, filter, sort, and deduplicate a freshly fetched DataFrame.

    Steps:
    1. Normalize column names to lowercase.
    2. Keep only OHLCV columns (drop Dividends, Stock Splits, etc.).
    3. Remove timezone info from the DatetimeIndex.
    4. Sort ascending by date.
    5. Drop duplicate index entries (keep first).
    """
    df = normalize_columns(df)

    keep = [c for c in REQUIRED_COLUMNS if c in df.columns]
    if not keep:
        raise DataFetchError(
            f"None of the required OHLCV columns found after normalization. "
            f"Columns present: {df.columns.tolist()}"
        )
    df = df[keep]
    df = _strip_timezone(df)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    return df


# ---------------------------------------------------------------------------
# Public: data fetching
# ---------------------------------------------------------------------------

def fetch_ticker(
    ticker: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    source: str = "yfinance",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a single ticker.

    Args:
        ticker:   Equity symbol, e.g. 'SPY'.
        start:    Start date as 'YYYY-MM-DD' string or datetime.
        end:      End date as 'YYYY-MM-DD' string or datetime.
        source:   Data source. Currently only 'yfinance' is supported.
        interval: yfinance bar interval, e.g. '1d', '1m', '5m', '15m'.
                  Defaults to '1d' (daily) so existing callers are unaffected.
                  Note: yfinance only serves intraday history for a limited
                  recent window (roughly the last 60 days for minute bars).

    Returns:
        Clean DataFrame with lowercase OHLCV columns, DatetimeIndex sorted
        ascending, duplicates removed.  Passes validate_ohlcv() before return.

    Raises:
        DataFetchError: If source is unsupported, yfinance is missing, the
                        network call fails, or the returned data is malformed.
    """
    if source not in ("yfinance",):
        raise DataFetchError(
            f"Unsupported data source: '{source}'. Supported: 'yfinance'."
        )

    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataFetchError(
            "yfinance is not installed. Run: pip install yfinance"
        ) from exc

    try:
        raw = yf.Ticker(ticker.upper()).history(
            start=start, end=end, interval=interval, auto_adjust=True,
        )
    except Exception as exc:
        # Distinguish between a provider-side failure and a genuine ticker problem.
        # JSON/network errors are always provider failures — the ticker is NOT delisted.
        exc_str = str(exc).lower()
        provider_fail = any(kw in exc_str for kw in (
            "expecting value", "json", "connection", "timeout",
            "503", "429", "rate limit", "ssl",
        ))
        if provider_fail:
            raise DataFetchError(
                f"Yahoo Finance / yfinance data provider failed for '{ticker}' "
                f"({start} to {end}): {exc}. "
                "This is a provider-side error — the ticker is almost certainly valid. "
                "Suggestions: (1) retry later, (2) upgrade yfinance, "
                "(3) use --synthetic or --csv."
            ) from exc
        raise DataFetchError(
            f"yfinance error fetching '{ticker}' ({start} to {end}): {exc}"
        ) from exc

    if raw is None or raw.empty:
        raise DataFetchError(
            f"No data returned for '{ticker}' from {start} to {end}. "
            "Yahoo Finance / yfinance returned an empty response — "
            "this is usually a provider outage or API rate limit, "
            "not a problem with the ticker symbol. "
            "Suggestions: (1) retry later, "
            "(2) pip install --upgrade yfinance, "
            "(3) run with --synthetic for smoke testing, "
            "(4) run with --csv if you have a local data file."
        )

    df = _clean_fetched(raw)
    validate_ohlcv(df)

    logger.info("Fetched %d rows for %s (%s to %s)", len(df), ticker, start, end)
    return df


def fetch_multiple_tickers(
    tickers: List[str],
    start: Union[str, datetime],
    end: Union[str, datetime],
    source: str = "yfinance",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for a list of tickers.

    Tickers that fail are skipped and logged as warnings rather than aborting
    the entire batch.  If every ticker fails, raises DataFetchError.

    Returns:
        Dict mapping ticker symbol -> clean OHLCV DataFrame.
        Only successfully fetched tickers are included.
    """
    results: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            results[ticker] = fetch_ticker(ticker, start, end, source=source)
        except DataFetchError as exc:
            logger.warning("Skipping %s: %s", ticker, exc)

    if not results:
        raise DataFetchError(
            f"Failed to fetch data for any ticker in {tickers}. "
            "Check network connectivity and ticker symbols."
        )

    logger.info("Fetched %d/%d tickers successfully.", len(results), len(tickers))
    return results


# ---------------------------------------------------------------------------
# Public: freshness check
# ---------------------------------------------------------------------------

def is_data_stale(
    df: pd.DataFrame,
    max_age_calendar_days: int = 5,
) -> bool:
    """
    Return True if the most recent row in df is older than max_age_calendar_days.

    Uses calendar days (not trading days) for simplicity.
    A setting of 5 covers a full trading week plus a weekend buffer.

    Returns True for empty or None DataFrames (treat as stale).
    """
    if df is None or df.empty:
        return True

    most_recent = pd.Timestamp(df.index[-1])
    if hasattr(most_recent, "tz") and most_recent.tz is not None:
        most_recent = most_recent.tz_localize(None)

    age_days = (pd.Timestamp.now().normalize() - most_recent).days
    return age_days > max_age_calendar_days


# ---------------------------------------------------------------------------
# Public: CSV loader
# ---------------------------------------------------------------------------

#: Column names that are recognised as the date/datetime index column in a CSV.
_DATE_COL_CANDIDATES = ("date", "datetime", "timestamp")


def load_ohlcv_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load OHLCV data from a user-supplied CSV file.

    The CSV must contain a date column named one of:
        date, datetime, timestamp

    All other column names are normalised to lowercase.
    The resulting DataFrame passes validate_ohlcv() before being returned.

    Args:
        path: Path to the CSV file (str or pathlib.Path).

    Returns:
        Clean OHLCV DataFrame with DatetimeIndex, sorted ascending,
        duplicates removed.

    Raises:
        DataFetchError: If the file is missing, unreadable, lacks the required
                        date column, or fails OHLCV validation.
    """
    path = Path(path)
    if not path.exists():
        raise DataFetchError(f"CSV file not found: {path.resolve()}")

    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        raise DataFetchError(
            f"Failed to read CSV '{path}': {exc}"
        ) from exc

    # Normalise all column names
    df = normalize_columns(raw)

    # Locate the date column
    date_col: Optional[str] = None
    for candidate in _DATE_COL_CANDIDATES:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        # Last resort: if the first column looks like dates, use it
        if len(df.columns) > 0:
            first_col = df.columns[0]
            try:
                pd.to_datetime(df[first_col].head(3))
                date_col = first_col
            except (ValueError, TypeError):
                pass

    if date_col is None:
        raise DataFetchError(
            f"CSV must contain a date column named one of "
            f"{list(_DATE_COL_CANDIDATES)}. "
            f"Found columns: {df.columns.tolist()}"
        )

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df.index.name = "Date"
    except Exception as exc:
        raise DataFetchError(
            f"Failed to parse '{date_col}' column as dates: {exc}"
        ) from exc

    # Strip timezone if present
    df = _strip_timezone(df)

    # Keep only OHLCV columns (drop anything extra the user's CSV might have)
    available = [c for c in REQUIRED_COLUMNS if c in df.columns]
    if available:
        df = df[available]

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    validate_ohlcv(df)

    logger.info(
        "Loaded %d rows from CSV '%s'", len(df), path.name
    )
    return df


# ---------------------------------------------------------------------------
# Public: synthetic data generator
# ---------------------------------------------------------------------------

def _synthetic_intraday_index(
    start: str,
    rows: int,
    interval_minutes: int,
) -> pd.DatetimeIndex:
    """
    Build a deterministic intraday DatetimeIndex of regular-session bars.

    Bars run from 09:30 up to (but not including) 16:00 — the 390-minute regular
    US equity session — stepping every `interval_minutes`, on business days only.
    When a session fills up, bars roll to the next business day's 09:30 open.

    Limitation: this is synthetic data for offline smoke tests. It ignores market
    holidays, half-days/early closes, and timezones — it only approximates a
    weekday session shape so intraday timeframes produce a plausible index.
    """
    session_minutes = 390  # 6.5 hours, 09:30 -> 16:00
    bars_per_day = max(1, session_minutes // interval_minutes)

    timestamps: List[pd.Timestamp] = []
    day = pd.Timestamp(start).normalize()
    while len(timestamps) < rows:
        if day.weekday() < 5:  # Monday-Friday only
            open_ts = day + pd.Timedelta(hours=9, minutes=30)
            for i in range(bars_per_day):
                if len(timestamps) >= rows:
                    break
                timestamps.append(open_ts + pd.Timedelta(minutes=i * interval_minutes))
        day += pd.Timedelta(days=1)

    return pd.DatetimeIndex(timestamps)


def generate_synthetic_ohlcv(
    rows: int = 500,
    start: str = "2020-01-01",
    seed: int = 42,
    start_price: float = 100.0,
    interval_minutes: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate deterministic synthetic OHLCV data for smoke testing.

    *** NOT real market data. Results have no financial meaning. ***
    Use only to verify that the strategy pipeline runs end-to-end
    without relying on a live data provider.

    Args:
        rows:             Number of trading bars to generate.
        start:            Start date string 'YYYY-MM-DD'.
        seed:             Random seed — same seed always produces the same DataFrame.
        start_price:      Starting close price in dollars.
        interval_minutes: None -> daily bars on a business-day index (default,
                          unchanged behavior). An int (1/5/15) -> intraday bars
                          on a regular-session index (see _synthetic_intraday_index).

    Returns:
        DataFrame with lowercase OHLCV columns and a DatetimeIndex (business-day
        for daily, regular-session for intraday). Passes validate_ohlcv().
    """
    rng = np.random.default_rng(seed)
    if interval_minutes is None:
        dates = pd.date_range(start=start, periods=rows, freq="B")
    else:
        dates = _synthetic_intraday_index(start, rows, interval_minutes)

    # Random walk for close prices with slight positive drift
    log_returns = rng.normal(0.0003, 0.012, rows)
    close = start_price * np.exp(np.cumsum(log_returns))

    # Intraday price range (proportional to close)
    daily_range = close * np.abs(rng.normal(0.0, 0.008, rows))
    open_ = np.abs(close * (1.0 + rng.normal(0.0, 0.004, rows)))  # abs ensures positive

    high = np.maximum(close, open_) + daily_range
    low  = np.maximum(np.minimum(close, open_) - daily_range, 0.01)  # floor at $0.01

    volume = rng.integers(1_000_000, 50_000_000, rows).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )

    validate_ohlcv(df)  # Sanity-check our own generator output
    return df
