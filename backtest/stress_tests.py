"""
Synthetic stress tests for the backtester.

All injection functions return COPIES of the input data — originals are
never modified.  Use `run_standard_stress_tests` to run the full battery.

Available scenarios:
  inject_market_drop       — sudden price drop at a specific bar
  inject_volatility_spike  — widen daily high/low range by a multiplier
  inject_missing_data      — randomly drop a fraction of rows
  inject_stale_data        — repeat the last N rows (simulates a data feed freeze)
  run_standard_stress_tests — run all standard scenarios and return results dict
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from loguru import logger  # type: ignore
except ImportError:
    logger = logging.getLogger(__name__)  # type: ignore


# ---------------------------------------------------------------------------
# Individual injection functions
# ---------------------------------------------------------------------------

def inject_market_drop(
    data_by_ticker: Dict[str, pd.DataFrame],
    drop_pct: float,
    date_index: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Apply a sudden price drop of `drop_pct` to all tickers from `date_index` onward.

    Args:
        data_by_ticker: Original OHLCV data (not mutated).
        drop_pct:       Fraction to drop, e.g. 0.10 for -10%.
        date_index:     Bar index where the drop occurs.
                        Defaults to the midpoint of each ticker's series.

    Returns:
        New dict of DataFrames with prices scaled down from date_index onward.
    """
    result = {}
    for ticker, df in data_by_ticker.items():
        df_copy = df.copy()
        idx = date_index if date_index is not None else len(df) // 2
        idx = max(0, min(idx, len(df) - 1))

        scale = 1.0 - drop_pct
        price_cols = [c for c in ("open", "high", "low", "close") if c in df_copy.columns]
        df_copy.iloc[idx:, df_copy.columns.get_indexer(price_cols)] *= scale
        result[ticker] = df_copy

    return result


def inject_volatility_spike(
    data_by_ticker: Dict[str, pd.DataFrame],
    multiplier: float = 2.0,
) -> Dict[str, pd.DataFrame]:
    """
    Widen every bar's high/low range by `multiplier` while keeping close unchanged.

    This simulates a volatility spike (e.g., VIX doubling) across the whole dataset.
    Close prices and volume are not modified.
    """
    result = {}
    for ticker, df in data_by_ticker.items():
        df_copy = df.copy()
        if "high" in df_copy.columns and "low" in df_copy.columns and "close" in df_copy.columns:
            mid       = (df_copy["high"] + df_copy["low"]) / 2
            half_range = (df_copy["high"] - df_copy["low"]) / 2
            df_copy["high"] = mid + half_range * multiplier
            df_copy["low"]  = (mid - half_range * multiplier).clip(lower=0.01)
        result[ticker] = df_copy

    return result


def inject_missing_data(
    data_by_ticker: Dict[str, pd.DataFrame],
    missing_fraction: float = 0.05,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Randomly drop `missing_fraction` of rows from each ticker's DataFrame.

    Models data-feed gaps or corporate-action blackout periods.
    """
    rng = np.random.default_rng(seed)
    result = {}
    for ticker, df in data_by_ticker.items():
        n_drop = max(1, int(len(df) * missing_fraction))
        drop_idx = rng.choice(len(df), size=n_drop, replace=False)
        mask = np.ones(len(df), dtype=bool)
        mask[drop_idx] = False
        result[ticker] = df.iloc[mask].copy()

    return result


def inject_stale_data(
    data_by_ticker: Dict[str, pd.DataFrame],
    repeat_last_n: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Replace the final `repeat_last_n` rows with copies of the bar just before them.

    Simulates a data feed that stops updating (stale/frozen feed).
    """
    result = {}
    for ticker, df in data_by_ticker.items():
        df_copy = df.copy()
        n = min(repeat_last_n, len(df) - 1)
        if n > 0:
            stale_bar = df_copy.iloc[-(n + 1)]
            for i in range(n, 0, -1):
                df_copy.iloc[-i] = stale_bar
        result[ticker] = df_copy

    return result


# ---------------------------------------------------------------------------
# Standard stress-test battery
# ---------------------------------------------------------------------------

def run_standard_stress_tests(
    backtester: Any,
    data_by_ticker: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    Run the standard battery of stress tests and return named results.

    Args:
        backtester:     A Backtester instance (already configured).
        data_by_ticker: Clean original OHLCV data.

    Returns:
        Dict mapping scenario name → BacktestResult (or an error string).

    Scenarios:
        "baseline"         — unmodified data
        "drop_10"          — sudden -10% at midpoint
        "drop_15"          — sudden -15% at midpoint
        "volatility_spike" — 2× intraday range
        "missing_data"     — 5% of rows dropped
        "stale_data"       — last 5 bars frozen
    """
    scenarios = {
        "baseline":         data_by_ticker,
        "drop_10":          inject_market_drop(data_by_ticker, drop_pct=0.10),
        "drop_15":          inject_market_drop(data_by_ticker, drop_pct=0.15),
        "volatility_spike": inject_volatility_spike(data_by_ticker, multiplier=2.0),
        "missing_data":     inject_missing_data(data_by_ticker),
        "stale_data":       inject_stale_data(data_by_ticker),
    }

    results = {}
    for name, data in scenarios.items():
        try:
            logger.info("Running stress scenario: {}", name)
            results[name] = backtester.run(data)
        except Exception as exc:
            logger.warning("Stress test '{}' raised: {}", name, exc)
            results[name] = str(exc)

    return results
