"""
Feature engineering for regime detection and signal generation.

All calculations are strictly backward-looking:
- Rolling windows use only the current and past rows (no min_periods tricks
  that could use future data).
- No forward-fill or back-fill that could propagate future information.
- Early rows with insufficient history produce NaN — that is correct behavior.

Usage:
    from data.feature_engineering import add_features, prepare_features_for_model

    df_raw  = fetch_ticker("SPY", "2020-01-01", "2023-12-31")
    df_feat = add_features(df_raw)           # keeps all rows; early rows have NaN
    df_model = prepare_features_for_model(df_raw)  # drops NaN rows, model-ready
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from data.market_data import DataFetchError

# ---------------------------------------------------------------------------
# Column name constants — used by tests and downstream consumers
# ---------------------------------------------------------------------------

#: Every column that add_features() appends to the input DataFrame.
ALL_FEATURE_COLUMNS: List[str] = [
    "returns",
    "log_returns",
    "rolling_volatility_20",
    "rolling_volatility_60",
    "true_range",
    "atr_14",
    "sma_20",
    "sma_50",
    "sma_100",
    "sma_200",
    "ema_20",
    "ema_50",
    "rsi_14",
    "volume_sma_20",
    "volume_ratio",
    "above_sma_50",
    "above_sma_200",
    "trend_score",
    "volatility_percentile",
    "rolling_high_20",
    "drawdown_from_20_high",
    "rolling_high_252",
    "drawdown_from_252_high",
    "vwap",
    "close_above_vwap",
    "distance_from_vwap_pct",
    "volume_sma_50",
    "rel_volume_20",
    "rel_volume_50",
    "unusual_volume",
    "opening_range_high",
    "opening_range_low",
    "opening_range_mid",
    "opening_range_width_pct",
    "broke_above_opening_range",
    "broke_below_opening_range",
    # Session-aware intraday features (VWAP / opening range reset per session).
    # For daily data each date is its own one-row session, so session_vwap equals
    # the bar's typical price and the session opening range is never established
    # (NaN range, 0.0 flags) — daily behavior is unaffected by these columns.
    "session_vwap",
    "close_above_session_vwap",
    "distance_from_session_vwap_pct",
    "session_opening_range_high",
    "session_opening_range_low",
    "session_opening_range_mid",
    "session_opening_range_width_pct",
    "broke_above_session_opening_range",
    "broke_below_session_opening_range",
]

#: Subset used as model inputs — normalized/stationary features only.
#: Absolute price levels (sma_20, etc.) are intentionally excluded because
#: they don't generalise across stocks with different price magnitudes.
MODEL_FEATURES: List[str] = [
    "log_returns",
    "rolling_volatility_20",
    "rolling_volatility_60",
    "atr_14",
    "rsi_14",
    "volume_ratio",
    "above_sma_50",
    "above_sma_200",
    "trend_score",
    "volatility_percentile",
    "drawdown_from_20_high",
    "drawdown_from_252_high",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's RSI using exponential smoothing (alpha = 1/period).

    Returns values in [0, 100].  First (period - 1) rows are NaN.
    No lookahead: each value depends only on current and prior closes.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # EWM with Wilder's alpha; min_periods suppresses output until we have
    # enough observations for a meaningful average.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Avoid division by zero when avg_loss is 0 (all gains, RSI = 100)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where avg_loss was 0 (pure uptrend), RSI is 100
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    return rsi


def _compute_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|).

    First row is NaN because PrevClose is undefined.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def _compute_trend_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite trend score in the range [-1, +1].

    Components:
    - +1 / -1 for price above / below each SMA (20, 50, 100, 200)
    - +1 / -1 for fast SMA above / below slow SMA (20>50, 50>200)

    Each component contributes only when its underlying SMA is available
    (not NaN).  The mean of available components is returned.
    For rows with no available SMAs, the result is NaN.

    No lookahead: all SMAs are computed solely from past data.
    """
    close = df["close"]
    signals: List[pd.Series] = []

    # Price vs each SMA
    for col in ("sma_20", "sma_50", "sma_100", "sma_200"):
        if col not in df.columns:
            continue
        sma = df[col]
        sig = pd.Series(
            np.where(close > sma, 1.0, -1.0),
            index=df.index,
            dtype=float,
        ).where(sma.notna())  # NaN wherever the SMA hasn't computed yet
        signals.append(sig)

    # SMA ordering (bullish alignment)
    for fast, slow in (("sma_20", "sma_50"), ("sma_50", "sma_200")):
        if fast not in df.columns or slow not in df.columns:
            continue
        both_valid = df[fast].notna() & df[slow].notna()
        sig = pd.Series(
            np.where(df[fast] > df[slow], 1.0, -1.0),
            index=df.index,
            dtype=float,
        ).where(both_valid)
        signals.append(sig)

    if not signals:
        return pd.Series(np.nan, index=df.index, dtype=float)

    # mean(axis=1, skipna=True) averages only the non-NaN components
    return pd.concat(signals, axis=1).mean(axis=1)


def _compute_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Volume Weighted Average Price (VWAP).

    VWAP = cumulative_sum(typical_price * volume) / cumulative_sum(volume)
    where typical_price = (high + low + close) / 3

    For daily data, this computes the cumulative VWAP from the start of the series.
    Handles zero volume safely by replacing cumulative_sum(volume) == 0 with NaN.

    No lookahead: each row depends only on itself and prior rows.
    """
    typical_price = (high + low + close) / 3.0
    cumulative_tp_vol = (typical_price * volume).cumsum()
    cumulative_vol = volume.cumsum()

    # Avoid division by zero: where cumsum(volume) == 0, return NaN
    vwap = cumulative_tp_vol / cumulative_vol.replace(0.0, np.nan)

    return vwap


def _add_opening_range_features(
    df: pd.DataFrame,
    range_bars: int = 6,
) -> pd.DataFrame:
    """
    Add opening range breakout features based on the first range_bars rows.

    Opening range is defined as the high/low of the first range_bars rows.
    Once the range is established (at row range_bars - 1), it remains constant
    for the rest of the DataFrame.

    Rows before the range is established have NaN for range values and 0.0 for flags.

    Args:
        df: DataFrame with OHLCV columns
        range_bars: number of rows used to establish the opening range (default 6)

    Returns:
        DataFrame with new columns:
        - opening_range_high: constant from row (range_bars-1) onwards
        - opening_range_low: constant from row (range_bars-1) onwards
        - opening_range_mid: (high + low) / 2
        - opening_range_width_pct: (high - low) / mid
        - broke_above_opening_range: 1.0 if close > high, 0.0 otherwise
        - broke_below_opening_range: 1.0 if close < low, 0.0 otherwise

    No lookahead: using first range_bars rows for all subsequent rows is allowed
    because those rows are already in the past.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Compute the opening range from the first range_bars rows
    if len(df) < range_bars:
        # Dataset too small; all rows are pre-range
        range_high = high.iloc[:range_bars].max()
        range_low = low.iloc[:range_bars].min()
        range_ready = False
    else:
        range_high = high.iloc[:range_bars].max()
        range_low = low.iloc[:range_bars].min()
        range_ready = True

    # opening_range_high: NaN before range is ready, constant after
    opening_range_high = pd.Series(np.nan, index=df.index, dtype=float)
    if range_ready:
        opening_range_high.iloc[range_bars - 1:] = range_high

    # opening_range_low: NaN before range is ready, constant after
    opening_range_low = pd.Series(np.nan, index=df.index, dtype=float)
    if range_ready:
        opening_range_low.iloc[range_bars - 1:] = range_low

    # opening_range_mid: (high + low) / 2
    opening_range_mid = (opening_range_high + opening_range_low) / 2.0

    # opening_range_width_pct: (high - low) / mid
    opening_range_width_pct = (
        (opening_range_high - opening_range_low) / opening_range_mid.replace(0.0, np.nan)
    )

    # Breakout flags: 0.0 before range is ready, then compare close to range
    broke_above = pd.Series(0.0, index=df.index, dtype=float)
    if range_ready:
        broke_above.iloc[range_bars - 1:] = np.where(
            close.iloc[range_bars - 1:] > range_high, 1.0, 0.0
        )

    broke_below = pd.Series(0.0, index=df.index, dtype=float)
    if range_ready:
        broke_below.iloc[range_bars - 1:] = np.where(
            close.iloc[range_bars - 1:] < range_low, 1.0, 0.0
        )

    # Return tuple of (high, low, mid, width_pct, above, below)
    return opening_range_high, opening_range_low, opening_range_mid, opening_range_width_pct, broke_above, broke_below


# ---------------------------------------------------------------------------
# Public: session-aware intraday features
# ---------------------------------------------------------------------------

def add_session_intraday_features(
    df: pd.DataFrame,
    opening_range_bars: int = 6,
) -> pd.DataFrame:
    """
    Add session-aware VWAP and opening-range features that reset each session.

    A "session" is one calendar date taken from the DatetimeIndex. Every
    quantity below is computed within each session independently and resets at
    the next date:

      session_vwap                      cumulative VWAP from the session open
      close_above_session_vwap          1.0 if close > session_vwap else 0.0 (NaN if vwap NaN)
      distance_from_session_vwap_pct    (close - session_vwap) / session_vwap

      session_opening_range_high        max(high) of the session's first
                                        opening_range_bars rows
      session_opening_range_low         min(low)  of those rows
      session_opening_range_mid         (high + low) / 2
      session_opening_range_width_pct   (high - low) / mid
      broke_above_session_opening_range 1.0 if close > session range high else 0.0
      broke_below_session_opening_range 1.0 if close < session range low  else 0.0

    Behavior notes:
      - For DAILY data each date is its own one-row session: session_vwap equals
        the bar's typical price and the opening range is never established
        (< opening_range_bars rows), so range columns are NaN and breakout flags
        are 0.0. This leaves daily detection behavior unchanged.
      - Range values are NaN until a session has accumulated opening_range_bars
        rows (established at position opening_range_bars - 1); breakout flags are
        0.0 until then.
      - Zero volume is handled safely: a zero session volume cumsum yields NaN
        session_vwap rather than a divide-by-zero.

    No lookahead: each value depends only on rows at or before it *within the
    same session*. Appending later rows (same or future sessions) never changes
    an earlier row's value — earlier rows before the opening range is established
    are masked to NaN/0.0, and groupby operates per-session.
    """
    required = {"high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"add_session_intraday_features requires OHLCV columns. "
            f"Missing: {sorted(missing)}."
        )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "add_session_intraday_features requires a DatetimeIndex to derive "
            "trading sessions from the bar dates."
        )

    df = df.copy()
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]

    # Session key = calendar date of each bar (midnight-normalised timestamp).
    session_key = df.index.normalize()

    # ------------------------------------------------------------------
    # Session VWAP (cumulative within each session, reset per date)
    # ------------------------------------------------------------------
    typical_price = (high + low + close) / 3.0
    cum_tp_vol = (typical_price * volume).groupby(session_key).cumsum()
    cum_vol = volume.groupby(session_key).cumsum()

    # Zero-volume safety: a zero cumulative session volume becomes NaN.
    session_vwap = cum_tp_vol / cum_vol.replace(0.0, np.nan)
    df["session_vwap"] = session_vwap

    df["close_above_session_vwap"] = (
        pd.Series(np.where(close > session_vwap, 1.0, 0.0), index=df.index)
        .where(session_vwap.notna())
    )
    df["distance_from_session_vwap_pct"] = (
        (close - session_vwap) / session_vwap.replace(0.0, np.nan)
    )

    # ------------------------------------------------------------------
    # Session opening range (first opening_range_bars rows of each session)
    # ------------------------------------------------------------------
    # Position of each row within its session (0-indexed) and session length.
    pos = volume.groupby(session_key).cumcount()
    session_size = volume.groupby(session_key).transform("size")

    # High/low restricted to the opening window, then the per-session
    # max/min broadcast back to every row of the session. transform("max")
    # ignores the NaN-masked non-opening rows.
    in_opening = pos < opening_range_bars
    range_high_full = high.where(in_opening).groupby(session_key).transform("max")
    range_low_full = low.where(in_opening).groupby(session_key).transform("min")

    # The range is only "established" once the session has at least
    # opening_range_bars rows AND we are at/after the establishing bar. Earlier
    # rows are masked to NaN so they never carry the (future-derived) range.
    established = (pos >= (opening_range_bars - 1)) & (session_size >= opening_range_bars)

    session_or_high = range_high_full.where(established)
    session_or_low = range_low_full.where(established)
    session_or_mid = (session_or_high + session_or_low) / 2.0
    session_or_width_pct = (
        (session_or_high - session_or_low) / session_or_mid.replace(0.0, np.nan)
    )

    df["session_opening_range_high"] = session_or_high
    df["session_opening_range_low"] = session_or_low
    df["session_opening_range_mid"] = session_or_mid
    df["session_opening_range_width_pct"] = session_or_width_pct

    # Breakout flags: NaN range (pre-establishment) compares False -> 0.0,
    # and the comparison naturally resets at each new session.
    df["broke_above_session_opening_range"] = pd.Series(
        np.where(close > session_or_high, 1.0, 0.0), index=df.index, dtype=float,
    )
    df["broke_below_session_opening_range"] = pd.Series(
        np.where(close < session_or_low, 1.0, 0.0), index=df.index, dtype=float,
    )

    return df


# ---------------------------------------------------------------------------
# Public: feature generation
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical features to an OHLCV DataFrame.

    Input requirements:
        - Must have lowercase columns: open, high, low, close, volume
        - Index must be a DatetimeIndex sorted ascending
        - Must not be empty

    Returns:
        A new DataFrame containing all original OHLCV columns plus every
        column listed in ALL_FEATURE_COLUMNS.  Early rows where a rolling
        window has insufficient history contain NaN — do NOT drop them here.

    Raises:
        ValueError: If required columns are missing or the DataFrame is empty.

    No-lookahead guarantee:
        For any row at position T, every feature value depends only on rows
        0 through T.  Adding rows at positions T+1, T+2, ... does not change
        any value at position T.  This is verified by tests/test_feature_engineering.py.
    """
    _validate_input(df)
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------
    df["returns"] = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))

    # ------------------------------------------------------------------
    # Volatility (annualised, assuming 252 trading days)
    # ------------------------------------------------------------------
    df["rolling_volatility_20"] = df["returns"].rolling(20).std() * np.sqrt(252)
    df["rolling_volatility_60"] = df["returns"].rolling(60).std() * np.sqrt(252)

    # ------------------------------------------------------------------
    # True Range and ATR (Wilder's EWM smoothing)
    # ------------------------------------------------------------------
    df["true_range"] = _compute_true_range(high, low, close)
    # alpha = 1/14 gives Wilder's smoothing; min_periods avoids early noise
    df["atr_14"] = df["true_range"].ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()

    # ------------------------------------------------------------------
    # Moving averages
    # ------------------------------------------------------------------
    df["sma_20"]  = close.rolling(20).mean()
    df["sma_50"]  = close.rolling(50).mean()
    df["sma_100"] = close.rolling(100).mean()
    df["sma_200"] = close.rolling(200).mean()
    df["ema_20"]  = close.ewm(span=20, adjust=False).mean()
    df["ema_50"]  = close.ewm(span=50, adjust=False).mean()

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------
    df["rsi_14"] = _compute_rsi(close, period=14)

    # ------------------------------------------------------------------
    # Volume indicators
    # ------------------------------------------------------------------
    df["volume_sma_20"] = volume.rolling(20).mean()
    # volume_ratio: today's volume vs 20-day average
    # NaN where volume_sma_20 is NaN or zero
    df["volume_ratio"] = volume / df["volume_sma_20"].replace(0.0, np.nan)

    # ------------------------------------------------------------------
    # Relative volume indicators (for day-trading setups)
    # ------------------------------------------------------------------
    # rel_volume_20: today's volume relative to 20-day average
    df["volume_sma_50"] = volume.rolling(50).mean()

    # rel_volume_20: ratio of current volume to 20-day average volume
    # NaN where volume_sma_20 is NaN or zero
    df["rel_volume_20"] = volume / df["volume_sma_20"].replace(0.0, np.nan)

    # rel_volume_50: ratio of current volume to 50-day average volume
    # NaN where volume_sma_50 is NaN or zero
    df["rel_volume_50"] = volume / df["volume_sma_50"].replace(0.0, np.nan)

    # unusual_volume: flag for significantly higher volume (>= 1.5x the 20-day average)
    # 1.0 if rel_volume_20 >= 1.5, 0.0 otherwise; NaN where rel_volume_20 is NaN
    df["unusual_volume"] = (
        pd.Series(np.where(df["rel_volume_20"] >= 1.5, 1.0, 0.0), index=df.index)
        .where(df["rel_volume_20"].notna())
    )

    # ------------------------------------------------------------------
    # Price vs SMA flags (1.0 = above, 0.0 = below, NaN = SMA not ready)
    # ------------------------------------------------------------------
    df["above_sma_50"] = (
        pd.Series(np.where(close > df["sma_50"], 1.0, 0.0), index=df.index)
        .where(df["sma_50"].notna())
    )
    df["above_sma_200"] = (
        pd.Series(np.where(close > df["sma_200"], 1.0, 0.0), index=df.index)
        .where(df["sma_200"].notna())
    )

    # ------------------------------------------------------------------
    # Composite trend score in [-1, +1]
    # ------------------------------------------------------------------
    df["trend_score"] = _compute_trend_score(df)

    # ------------------------------------------------------------------
    # Volatility percentile (where current vol sits in its 252-day history)
    # raw=True passes a numpy array for speed; x[-1] is the current value.
    # ------------------------------------------------------------------
    df["volatility_percentile"] = (
        df["rolling_volatility_20"]
        .rolling(252, min_periods=20)
        .apply(lambda x: float((x < x[-1]).mean()), raw=True)
    )

    # ------------------------------------------------------------------
    # Drawdown from rolling highs
    # ------------------------------------------------------------------
    df["rolling_high_20"]       = close.rolling(20).max()
    df["drawdown_from_20_high"] = (close - df["rolling_high_20"]) / df["rolling_high_20"]

    df["rolling_high_252"]       = close.rolling(252).max()
    df["drawdown_from_252_high"] = (close - df["rolling_high_252"]) / df["rolling_high_252"]

    # ------------------------------------------------------------------
    # Volume Weighted Average Price (VWAP) and related indicators
    # ------------------------------------------------------------------
    df["vwap"] = _compute_vwap(high, low, close, volume)

    # close_above_vwap: 1.0 if close > vwap, 0.0 otherwise, NaN where vwap is NaN
    df["close_above_vwap"] = (
        pd.Series(np.where(close > df["vwap"], 1.0, 0.0), index=df.index)
        .where(df["vwap"].notna())
    )

    # distance_from_vwap_pct: (close - vwap) / vwap as a percentage
    # NaN where vwap is 0 or NaN
    df["distance_from_vwap_pct"] = (close - df["vwap"]) / df["vwap"].replace(0.0, np.nan)

    # ------------------------------------------------------------------
    # Opening range breakout features (for day-trading detection)
    # ------------------------------------------------------------------
    (
        df["opening_range_high"],
        df["opening_range_low"],
        df["opening_range_mid"],
        df["opening_range_width_pct"],
        df["broke_above_opening_range"],
        df["broke_below_opening_range"],
    ) = _add_opening_range_features(df, range_bars=6)

    # ------------------------------------------------------------------
    # Session-aware intraday features (VWAP + opening range reset per date).
    # Kept separate from the columns above so existing daily behavior and the
    # backward-compatible vwap/opening_range_* columns are unchanged.
    # ------------------------------------------------------------------
    df = add_session_intraday_features(df, opening_range_bars=6)

    return df


def prepare_features_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a model-ready DataFrame: add features, select model columns, drop NaN rows.

    Steps:
    1. Call add_features() — appends all technical indicators.
    2. Select only MODEL_FEATURES columns.
    3. Drop any row that still contains NaN after feature computation.
       (These are always the earliest rows where rolling windows lack data.)

    The returned DataFrame is safe to pass directly to a regime classifier or
    any scikit-learn estimator.

    Raises:
        ValueError: If input fails basic OHLCV validation.
        DataFetchError: Propagated from add_features() if columns are missing.
    """
    df_feat = add_features(df)

    # Only select model features that actually exist in the output
    available = [c for c in MODEL_FEATURES if c in df_feat.columns]
    df_model = df_feat[available].copy()

    # Drop early rows where rolling windows haven't filled yet.
    # This is the ONLY place we drop rows — never inside add_features().
    n_before = len(df_model)
    df_model = df_model.dropna()
    n_dropped = n_before - len(df_model)

    if n_dropped > 0:
        import logging
        logging.getLogger(__name__).debug(
            "prepare_features_for_model: dropped %d rows with NaN (insufficient history).",
            n_dropped,
        )

    return df_model


# ---------------------------------------------------------------------------
# Private: input validation
# ---------------------------------------------------------------------------

def _validate_input(df: pd.DataFrame) -> None:
    """Raise ValueError if df is not a valid OHLCV input for add_features."""
    if df is None or df.empty:
        raise ValueError("add_features received an empty DataFrame.")

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"add_features requires OHLCV columns. Missing: {sorted(missing)}. "
            f"Present: {sorted(df.columns.tolist())}. "
            "Ensure the DataFrame was produced by fetch_ticker() or validate_ohlcv()."
        )
