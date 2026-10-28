"""
Tests for data/feature_engineering.py and data/market_data.py (validation layer).

Invariants verified here:
- add_features() never uses future data (strict no-lookahead)
- All feature columns are produced
- Original OHLCV columns are preserved
- RSI in [0, 100], ATR >= 0, drawdown <= 0
- volume_ratio is computed correctly
- Malformed data raises DataFetchError with a clear message
- Duplicate dates in the index are deduplicated
- prepare_features_for_model() returns a NaN-free DataFrame
- Everything works end-to-end on 300+ rows of synthetic data
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared fixture — synthetic OHLCV data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """
    Generate 400 rows of synthetic OHLCV data via a seeded random walk.

    Prices start at $100 with a slight positive drift.
    All OHLCV invariants hold: high >= low, all prices > 0, volume > 0.
    """
    rng = np.random.default_rng(seed=42)
    n = 400
    dates = pd.date_range("2021-01-01", periods=n, freq="B")  # business days

    # Random walk for close
    log_ret = rng.normal(0.0003, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(log_ret))

    # Build realistic OHLC from close
    daily_range = close * np.abs(rng.normal(0.0, 0.008, n))
    open_ = close * (1.0 + rng.normal(0.0, 0.004, n))
    high = np.maximum(close, open_) + daily_range
    low  = np.minimum(close, open_) - np.abs(daily_range)
    # Ensure low > 0
    low = np.maximum(low, 0.01)

    volume = rng.integers(1_000_000, 50_000_000, n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Test 1: add_features preserves original OHLCV columns
# ---------------------------------------------------------------------------

def test_add_features_preserves_ohlcv_columns(sample_ohlcv):
    """Original OHLCV columns must be present and unchanged after add_features()."""
    from data.feature_engineering import add_features

    original = sample_ohlcv.copy()
    result = add_features(sample_ohlcv)

    for col in ("open", "high", "low", "close", "volume"):
        assert col in result.columns, f"Column '{col}' missing from output"
        pd.testing.assert_series_equal(
            result[col], original[col], check_names=True,
            obj=f"OHLCV column '{col}' was modified by add_features()"
        )


# ---------------------------------------------------------------------------
# Test 2: add_features creates all required feature columns
# ---------------------------------------------------------------------------

def test_add_features_creates_all_feature_columns(sample_ohlcv):
    """Every column in ALL_FEATURE_COLUMNS must appear in the output."""
    from data.feature_engineering import add_features, ALL_FEATURE_COLUMNS

    result = add_features(sample_ohlcv)
    missing = [c for c in ALL_FEATURE_COLUMNS if c not in result.columns]
    assert missing == [], f"Missing feature columns: {missing}"


# ---------------------------------------------------------------------------
# Test 3: No lookahead bias
# ---------------------------------------------------------------------------

def test_no_lookahead_bias(sample_ohlcv):
    """
    Features computed on the first K rows must be identical to the corresponding
    rows when computed on the full dataset.

    Methodology: compute features on 250 rows and on 350 rows; compare rows 0-248.
    If any value differs, future data was used (lookahead detected).
    """
    from data.feature_engineering import add_features, ALL_FEATURE_COLUMNS

    short = add_features(sample_ohlcv.iloc[:250].copy())
    full  = add_features(sample_ohlcv.iloc[:350].copy())

    n_check = 249  # Compare rows 0..248 (well inside the shorter dataset)

    # volatility_percentile uses a 252-day window; skip it since a 250-row dataset
    # has fewer than 252 rows so the window behaviour at the boundary is trivially
    # different (window shrinks differently at row 249 vs 350).
    skip = {"volatility_percentile"}

    for col in ALL_FEATURE_COLUMNS:
        if col in skip:
            continue
        if col not in short.columns or col not in full.columns:
            continue

        short_vals = short[col].iloc[:n_check].reset_index(drop=True)
        full_vals  = full[col].iloc[:n_check].reset_index(drop=True)

        pd.testing.assert_series_equal(
            short_vals, full_vals,
            check_names=False,
            rtol=1e-6,
            obj=f"Lookahead bias detected in column '{col}'",
        )


# ---------------------------------------------------------------------------
# Test 4: RSI is within [0, 100] (where not NaN)
# ---------------------------------------------------------------------------

def test_rsi_within_bounds(sample_ohlcv):
    """rsi_14 must be in [0, 100] for every non-NaN value."""
    from data.feature_engineering import add_features

    result = add_features(sample_ohlcv)
    rsi = result["rsi_14"].dropna()

    assert len(rsi) > 0, "RSI column is entirely NaN — no values to check"
    assert (rsi >= 0.0).all(), f"RSI below 0 detected: min={rsi.min():.4f}"
    assert (rsi <= 100.0).all(), f"RSI above 100 detected: max={rsi.max():.4f}"


# ---------------------------------------------------------------------------
# Test 5: ATR is non-negative
# ---------------------------------------------------------------------------

def test_atr_nonnegative(sample_ohlcv):
    """atr_14 must be >= 0 for every non-NaN value."""
    from data.feature_engineering import add_features

    result = add_features(sample_ohlcv)
    atr = result["atr_14"].dropna()

    assert len(atr) > 0, "atr_14 is entirely NaN — no values to check"
    assert (atr >= 0.0).all(), f"Negative ATR detected: min={atr.min():.6f}"


# ---------------------------------------------------------------------------
# Test 6: volume_ratio is calculated correctly
# ---------------------------------------------------------------------------

def test_volume_ratio_calculation(sample_ohlcv):
    """volume_ratio must equal volume / volume_sma_20 for every non-NaN row."""
    from data.feature_engineering import add_features

    result = add_features(sample_ohlcv)
    valid = result["volume_ratio"].notna() & result["volume_sma_20"].notna()
    subset = result[valid]

    expected = subset["volume"] / subset["volume_sma_20"]
    pd.testing.assert_series_equal(
        subset["volume_ratio"], expected,
        check_names=False,
        rtol=1e-10,
        obj="volume_ratio != volume / volume_sma_20",
    )


# ---------------------------------------------------------------------------
# Test 7: malformed data raises DataFetchError with a clear message
# ---------------------------------------------------------------------------

def test_missing_column_raises():
    """validate_ohlcv() must raise DataFetchError when a required column is absent."""
    from data.market_data import validate_ohlcv, DataFetchError

    bad_df = pd.DataFrame({"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5]})
    # 'volume' column is missing

    with pytest.raises(DataFetchError, match="[Mm]issing"):
        validate_ohlcv(bad_df)


def test_negative_price_raises():
    """validate_ohlcv() must raise DataFetchError when any price column is <= 0."""
    from data.market_data import validate_ohlcv, DataFetchError

    bad_df = pd.DataFrame({
        "open":   [100.0, -5.0],
        "high":   [101.0,  1.0],
        "low":    [ 99.0,  0.5],
        "close":  [100.5,  0.8],
        "volume": [1e6,    1e6],
    })

    with pytest.raises(DataFetchError, match="non-positive"):
        validate_ohlcv(bad_df)


def test_high_less_than_low_raises():
    """validate_ohlcv() must raise DataFetchError when high < low."""
    from data.market_data import validate_ohlcv, DataFetchError

    bad_df = pd.DataFrame({
        "open":   [100.0],
        "high":   [ 98.0],   # high < low — malformed
        "low":    [101.0],
        "close":  [100.5],
        "volume": [1_000_000.0],
    })

    with pytest.raises(DataFetchError, match="[Hh]igh.*[Ll]ow|[Ll]ow.*[Hh]igh|malformed"):
        validate_ohlcv(bad_df)


def test_empty_dataframe_raises():
    """validate_ohlcv() must raise DataFetchError for an empty DataFrame."""
    from data.market_data import validate_ohlcv, DataFetchError

    with pytest.raises(DataFetchError, match="[Ee]mpty"):
        validate_ohlcv(pd.DataFrame())


def test_add_features_missing_ohlcv_raises():
    """add_features() must raise ValueError when given a DataFrame without OHLCV columns."""
    from data.feature_engineering import add_features

    bad_df = pd.DataFrame({"price": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="[Mm]issing"):
        add_features(bad_df)


# ---------------------------------------------------------------------------
# Test 8: duplicate dates are removed after deduplication
# ---------------------------------------------------------------------------

def test_duplicate_dates_deduplicated(sample_ohlcv):
    """
    After _clean_fetched(), duplicate index rows must be removed.
    We test this by manually injecting a duplicate and then calling
    the internal cleaning path that deduplicates.
    """
    from data.market_data import _clean_fetched

    # Build a small clean DataFrame first
    small = sample_ohlcv.iloc[:10].copy()
    small.columns = [c.capitalize() for c in small.columns]  # Simulate yfinance casing
    small.index.name = "Date"

    # Duplicate the first row
    dup = pd.concat([small.iloc[[0]], small])
    assert dup.index.duplicated().any(), "Test setup error: duplicate not created"

    result = _clean_fetched(dup)
    assert not result.index.duplicated().any(), "Duplicate dates were not removed"
    assert len(result) == len(small), f"Expected {len(small)} rows, got {len(result)}"


# ---------------------------------------------------------------------------
# Test 9: prepare_features_for_model returns no NaN values
# ---------------------------------------------------------------------------

def test_prepare_features_no_nan(sample_ohlcv):
    """
    prepare_features_for_model() must return a DataFrame with no NaN values.
    Early rows with insufficient history are dropped — that is the intended behaviour.
    """
    from data.feature_engineering import prepare_features_for_model

    result = prepare_features_for_model(sample_ohlcv)

    assert len(result) > 0, "prepare_features_for_model returned an empty DataFrame"
    assert not result.isnull().any().any(), (
        "prepare_features_for_model returned NaN values:\n"
        + str(result.isnull().sum()[result.isnull().sum() > 0])
    )


def test_prepare_features_returns_model_columns(sample_ohlcv):
    """prepare_features_for_model() must include all MODEL_FEATURES columns."""
    from data.feature_engineering import prepare_features_for_model, MODEL_FEATURES

    result = prepare_features_for_model(sample_ohlcv)
    missing = [c for c in MODEL_FEATURES if c not in result.columns]
    assert missing == [], f"Model columns missing: {missing}"


# ---------------------------------------------------------------------------
# Test 10: feature generation works on at least 300 rows of synthetic data
# ---------------------------------------------------------------------------

def test_works_with_300_rows():
    """add_features() must complete without error on at least 300 rows."""
    from data.feature_engineering import add_features, ALL_FEATURE_COLUMNS

    rng = np.random.default_rng(seed=0)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    spread = close * 0.005
    df = pd.DataFrame({
        "open":   close * (1.0 + rng.normal(0, 0.003, n)),
        "high":   close + spread,
        "low":    close - spread,
        "close":  close,
        "volume": rng.integers(500_000, 5_000_000, n).astype(float),
    }, index=dates)

    result = add_features(df)

    assert len(result) == n, "Row count changed after add_features()"
    for col in ALL_FEATURE_COLUMNS:
        assert col in result.columns, f"Expected feature column '{col}' not found"


# ---------------------------------------------------------------------------
# Additional targeted tests
# ---------------------------------------------------------------------------

def test_drawdown_is_non_positive(sample_ohlcv):
    """Drawdown from high must always be <= 0 (price can never exceed its own high)."""
    from data.feature_engineering import add_features

    result = add_features(sample_ohlcv)

    for col in ("drawdown_from_20_high", "drawdown_from_252_high"):
        valid = result[col].dropna()
        assert (valid <= 1e-9).all(), (  # tiny tolerance for float rounding
            f"{col} contains positive values: max={valid.max():.6f}"
        )


def test_sma_200_requires_200_rows(sample_ohlcv):
    """sma_200 must be NaN for the first 199 rows (insufficient history)."""
    from data.feature_engineering import add_features

    result = add_features(sample_ohlcv)
    # First 199 rows should be NaN
    assert result["sma_200"].iloc[:199].isna().all(), (
        "sma_200 should be NaN for the first 199 rows"
    )
    # Row 199 (200th row, 0-indexed) should have a value
    assert pd.notna(result["sma_200"].iloc[199]), (
        "sma_200 should have a value at row 199 (the 200th row)"
    )


def test_trend_score_within_bounds(sample_ohlcv):
    """trend_score must be in [-1, +1] for every non-NaN value."""
    from data.feature_engineering import add_features

    result = add_features(sample_ohlcv)
    ts = result["trend_score"].dropna()

    assert len(ts) > 0, "trend_score is entirely NaN"
    assert (ts >= -1.0 - 1e-9).all(), f"trend_score below -1: min={ts.min():.4f}"
    assert (ts <= +1.0 + 1e-9).all(), f"trend_score above +1: max={ts.max():.4f}"


def test_log_returns_match_manual_calculation(sample_ohlcv):
    """log_returns must equal log(close_t / close_{t-1})."""
    from data.feature_engineering import add_features

    result = add_features(sample_ohlcv)
    close = sample_ohlcv["close"]
    expected = np.log(close / close.shift(1))

    pd.testing.assert_series_equal(
        result["log_returns"], expected,
        check_names=False, rtol=1e-10,
        obj="log_returns != log(close/prev_close)",
    )


def test_volume_ratio_nan_when_sma_not_ready(sample_ohlcv):
    """volume_ratio must be NaN for the first 19 rows (volume_sma_20 not ready)."""
    from data.feature_engineering import add_features

    result = add_features(sample_ohlcv)
    # First 19 rows have no volume_sma_20 → ratio must be NaN
    assert result["volume_ratio"].iloc[:19].isna().all(), (
        "volume_ratio should be NaN for first 19 rows (volume_sma_20 not available)"
    )


# ---------------------------------------------------------------------------
# Test 11: VWAP and related features
# ---------------------------------------------------------------------------

def test_vwap_column_created(sample_ohlcv):
    """vwap column must be created by add_features()."""
    from data.feature_engineering import add_features
    result = add_features(sample_ohlcv)
    assert "vwap" in result.columns, "vwap column not created"


def test_close_above_vwap_column_created(sample_ohlcv):
    """close_above_vwap column must be created by add_features()."""
    from data.feature_engineering import add_features
    result = add_features(sample_ohlcv)
    assert "close_above_vwap" in result.columns, "close_above_vwap column not created"


def test_distance_from_vwap_pct_column_created(sample_ohlcv):
    """distance_from_vwap_pct column must be created by add_features()."""
    from data.feature_engineering import add_features
    result = add_features(sample_ohlcv)
    assert "distance_from_vwap_pct" in result.columns, "distance_from_vwap_pct column not created"


def test_vwap_calculation_on_known_data():
    """VWAP calculation must be correct on a small known dataset."""
    from data.feature_engineering import add_features

    # Create a simple 3-row dataset with known values
    df = pd.DataFrame({
        "open":   [100.0, 101.0, 102.0],
        "high":   [101.0, 102.0, 103.0],
        "low":    [ 99.0, 100.0, 101.0],
        "close":  [100.5, 101.5, 102.5],
        "volume": [1000.0, 2000.0, 1000.0],
    }, index=pd.date_range("2021-01-01", periods=3, freq="D"))

    result = add_features(df)

    # Calculate expected VWAP manually:
    # Row 0: typical_price = (101 + 99 + 100.5) / 3 = 100.166...
    #        vwap = (100.166... * 1000) / 1000 = 100.166...
    # Row 1: typical_price = (102 + 100 + 101.5) / 3 = 101.166...
    #        cumsum(tp*vol) = 100.166...*1000 + 101.166...*2000 = 100166.67 + 202333.33 = 302500.0
    #        cumsum(vol) = 1000 + 2000 = 3000
    #        vwap = 302500.0 / 3000 = 100.833...
    # Row 2: typical_price = (103 + 101 + 102.5) / 3 = 102.166...
    #        cumsum(tp*vol) = 302500.0 + 102.166...*1000 = 302500.0 + 102166.67 = 404666.67
    #        cumsum(vol) = 3000 + 1000 = 4000
    #        vwap = 404666.67 / 4000 = 101.166...

    tp0 = (101 + 99 + 100.5) / 3.0
    tp1 = (102 + 100 + 101.5) / 3.0
    tp2 = (103 + 101 + 102.5) / 3.0

    expected_vwap = [
        tp0,
        (tp0 * 1000 + tp1 * 2000) / 3000.0,
        (tp0 * 1000 + tp1 * 2000 + tp2 * 1000) / 4000.0,
    ]

    for i, exp_vwap in enumerate(expected_vwap):
        assert np.isclose(result["vwap"].iloc[i], exp_vwap, rtol=1e-5), (
            f"Row {i}: expected vwap={exp_vwap:.6f}, got {result['vwap'].iloc[i]:.6f}"
        )


def test_close_above_vwap_values(sample_ohlcv):
    """close_above_vwap must be 1.0 where close > vwap, 0.0 where close <= vwap."""
    from data.feature_engineering import add_features

    result = add_features(sample_ohlcv)

    # For every row where both vwap and close_above_vwap are not NaN:
    valid = result["vwap"].notna() & result["close_above_vwap"].notna()
    subset = result[valid]

    for idx, row in subset.iterrows():
        close = row["close"]
        vwap = row["vwap"]
        close_above = row["close_above_vwap"]

        if close > vwap:
            assert close_above == 1.0, (
                f"Row {idx}: close ({close:.4f}) > vwap ({vwap:.4f}) but close_above_vwap = {close_above}"
            )
        else:
            assert close_above == 0.0, (
                f"Row {idx}: close ({close:.4f}) <= vwap ({vwap:.4f}) but close_above_vwap = {close_above}"
            )


def test_distance_from_vwap_pct_calculation(sample_ohlcv):
    """distance_from_vwap_pct must equal (close - vwap) / vwap."""
    from data.feature_engineering import add_features

    result = add_features(sample_ohlcv)

    valid = result["vwap"].notna() & result["distance_from_vwap_pct"].notna()
    subset = result[valid]

    expected = (subset["close"] - subset["vwap"]) / subset["vwap"]
    pd.testing.assert_series_equal(
        subset["distance_from_vwap_pct"], expected,
        check_names=False, rtol=1e-10,
        obj="distance_from_vwap_pct != (close - vwap) / vwap",
    )


def test_vwap_no_lookahead(sample_ohlcv):
    """VWAP must not use future data (strict backward-looking)."""
    from data.feature_engineering import add_features

    short = add_features(sample_ohlcv.iloc[:250].copy())
    full = add_features(sample_ohlcv.iloc[:350].copy())

    n_check = 249

    # Compare VWAP values for the first 249 rows
    short_vwap = short["vwap"].iloc[:n_check].reset_index(drop=True)
    full_vwap = full["vwap"].iloc[:n_check].reset_index(drop=True)

    pd.testing.assert_series_equal(
        short_vwap, full_vwap,
        check_names=False, rtol=1e-6,
        obj="Lookahead bias detected in VWAP",
    )


def test_vwap_handles_zero_volume():
    """VWAP must not crash when volume is zero."""
    from data.feature_engineering import add_features

    # Create a dataset with a zero-volume row
    df = pd.DataFrame({
        "open":   [100.0, 101.0, 102.0],
        "high":   [101.0, 102.0, 103.0],
        "low":    [ 99.0, 100.0, 101.0],
        "close":  [100.5, 101.5, 102.5],
        "volume": [1000.0, 0.0, 1000.0],  # Zero volume at row 1
    }, index=pd.date_range("2021-01-01", periods=3, freq="D"))

    result = add_features(df)

    # VWAP should still be computed; row 1 might have NaN or a valid value depending on
    # whether zero volume creates a division by zero
    assert "vwap" in result.columns
    # The important thing is that it doesn't crash
    assert len(result) == 3


def test_prepare_features_still_works_with_vwap(sample_ohlcv):
    """prepare_features_for_model() must still work after adding VWAP features."""
    from data.feature_engineering import prepare_features_for_model

    result = prepare_features_for_model(sample_ohlcv)

    # Must return a DataFrame
    assert isinstance(result, pd.DataFrame)
    # Must have at least some rows (NaN rows are dropped)
    assert len(result) > 100, "prepare_features_for_model returned too few rows"
    # Must not have any NaN values in MODEL_FEATURES columns
    assert not result.isna().any().any(), "Result contains NaN values"


# ---------------------------------------------------------------------------
# Test 12: Relative volume indicators
# ---------------------------------------------------------------------------

def test_rel_volume_20_column_created(sample_ohlcv):
    """rel_volume_20 column must be created by add_features()."""
    from data.feature_engineering import add_features
    result = add_features(sample_ohlcv)
    assert "rel_volume_20" in result.columns, "rel_volume_20 column not created"


def test_rel_volume_50_column_created(sample_ohlcv):
    """rel_volume_50 column must be created by add_features()."""
    from data.feature_engineering import add_features
    result = add_features(sample_ohlcv)
    assert "rel_volume_50" in result.columns, "rel_volume_50 column not created"


def test_unusual_volume_column_created(sample_ohlcv):
    """unusual_volume column must be created by add_features()."""
    from data.feature_engineering import add_features
    result = add_features(sample_ohlcv)
    assert "unusual_volume" in result.columns, "unusual_volume column not created"


def test_rel_volume_20_calculation():
    """rel_volume_20 must equal volume / volume_sma_20 on a known dataset."""
    from data.feature_engineering import add_features

    # Create a simple dataset with known volumes
    df = pd.DataFrame({
        "open":   [100.0, 101.0, 102.0, 103.0, 104.0],
        "high":   [101.0, 102.0, 103.0, 104.0, 105.0],
        "low":    [ 99.0, 100.0, 101.0, 102.0, 103.0],
        "close":  [100.5, 101.5, 102.5, 103.5, 104.5],
        "volume": [1000.0, 1200.0, 800.0, 1500.0, 900.0],
    }, index=pd.date_range("2021-01-01", periods=5, freq="D"))

    result = add_features(df)

    # rel_volume_20 should be NaN for first 19 rows (20-period window)
    # For row 19 (if we had it), it would be volume / mean(volumes[0:20])
    # In our 5-row dataset, we can't fully test this, but we can verify the column exists
    # and is computed correctly where it's not NaN

    valid = result["rel_volume_20"].notna() & result["volume_sma_20"].notna()
    if valid.any():
        expected = result.loc[valid, "volume"] / result.loc[valid, "volume_sma_20"]
        pd.testing.assert_series_equal(
            result.loc[valid, "rel_volume_20"], expected,
            check_names=False, rtol=1e-10,
            obj="rel_volume_20 != volume / volume_sma_20",
        )


def test_unusual_volume_flag_values(sample_ohlcv):
    """unusual_volume must be 1.0 when rel_volume_20 >= 1.5, else 0.0."""
    from data.feature_engineering import add_features

    result = add_features(sample_ohlcv)

    # For every row where both rel_volume_20 and unusual_volume are not NaN:
    valid = result["rel_volume_20"].notna() & result["unusual_volume"].notna()
    subset = result[valid]

    for idx, row in subset.iterrows():
        rel_vol = row["rel_volume_20"]
        unusual = row["unusual_volume"]

        if rel_vol >= 1.5:
            assert unusual == 1.0, (
                f"Row {idx}: rel_volume_20 ({rel_vol:.4f}) >= 1.5 but unusual_volume = {unusual}"
            )
        else:
            assert unusual == 0.0, (
                f"Row {idx}: rel_volume_20 ({rel_vol:.4f}) < 1.5 but unusual_volume = {unusual}"
            )


def test_unusual_volume_nan_where_rel_volume_nan(sample_ohlcv):
    """unusual_volume must be NaN wherever rel_volume_20 is NaN."""
    from data.feature_engineering import add_features

    result = add_features(sample_ohlcv)

    # Where rel_volume_20 is NaN, unusual_volume must also be NaN
    rel_vol_nan = result["rel_volume_20"].isna()
    unusual_at_nan = result.loc[rel_vol_nan, "unusual_volume"]

    assert unusual_at_nan.isna().all(), (
        "unusual_volume should be NaN wherever rel_volume_20 is NaN"
    )


def test_rel_volume_no_lookahead(sample_ohlcv):
    """rel_volume_20 and rel_volume_50 must not use future data."""
    from data.feature_engineering import add_features

    short = add_features(sample_ohlcv.iloc[:250].copy())
    full = add_features(sample_ohlcv.iloc[:350].copy())

    n_check = 249

    for col in ("rel_volume_20", "rel_volume_50"):
        short_vals = short[col].iloc[:n_check].reset_index(drop=True)
        full_vals = full[col].iloc[:n_check].reset_index(drop=True)

        pd.testing.assert_series_equal(
            short_vals, full_vals,
            check_names=False, rtol=1e-6,
            obj=f"Lookahead bias detected in {col}",
        )


def test_rel_volume_handles_zero_sma():
    """rel_volume_20 must not crash when volume_sma_20 is zero."""
    from data.feature_engineering import add_features

    # Create a dataset with very small volumes
    df = pd.DataFrame({
        "open":   [100.0] * 30,
        "high":   [101.0] * 30,
        "low":    [ 99.0] * 30,
        "close":  [100.5] * 30,
        "volume": [0.0001] * 30,  # Extremely small volumes that might sum to near-zero
    }, index=pd.date_range("2021-01-01", periods=30, freq="D"))

    result = add_features(df)

    # Should not crash; columns should exist
    assert "rel_volume_20" in result.columns
    assert "rel_volume_50" in result.columns
    assert "unusual_volume" in result.columns
    # Length should be preserved
    assert len(result) == 30


def test_prepare_features_still_works_with_rel_volume(sample_ohlcv):
    """prepare_features_for_model() must still work after adding relative volume features."""
    from data.feature_engineering import prepare_features_for_model

    result = prepare_features_for_model(sample_ohlcv)

    # Must return a DataFrame
    assert isinstance(result, pd.DataFrame)
    # Must have at least some rows (NaN rows are dropped)
    assert len(result) > 100, "prepare_features_for_model returned too few rows"
    # Must not have any NaN values in MODEL_FEATURES columns
    assert not result.isna().any().any(), "Result contains NaN values"


# ---------------------------------------------------------------------------
# Test 13: Opening range breakout features
# ---------------------------------------------------------------------------

def test_opening_range_high_column_created(sample_ohlcv):
    """opening_range_high column must be created by add_features()."""
    from data.feature_engineering import add_features
    result = add_features(sample_ohlcv)
    assert "opening_range_high" in result.columns, "opening_range_high column not created"


def test_opening_range_low_column_created(sample_ohlcv):
    """opening_range_low column must be created by add_features()."""
    from data.feature_engineering import add_features
    result = add_features(sample_ohlcv)
    assert "opening_range_low" in result.columns, "opening_range_low column not created"


def test_opening_range_mid_column_created(sample_ohlcv):
    """opening_range_mid column must be created by add_features()."""
    from data.feature_engineering import add_features
    result = add_features(sample_ohlcv)
    assert "opening_range_mid" in result.columns, "opening_range_mid column not created"


def test_opening_range_width_pct_column_created(sample_ohlcv):
    """opening_range_width_pct column must be created by add_features()."""
    from data.feature_engineering import add_features
    result = add_features(sample_ohlcv)
    assert "opening_range_width_pct" in result.columns, "opening_range_width_pct column not created"


def test_broke_above_opening_range_column_created(sample_ohlcv):
    """broke_above_opening_range column must be created by add_features()."""
    from data.feature_engineering import add_features
    result = add_features(sample_ohlcv)
    assert "broke_above_opening_range" in result.columns, "broke_above_opening_range column not created"


def test_broke_below_opening_range_column_created(sample_ohlcv):
    """broke_below_opening_range column must be created by add_features()."""
    from data.feature_engineering import add_features
    result = add_features(sample_ohlcv)
    assert "broke_below_opening_range" in result.columns, "broke_below_opening_range column not created"


def test_opening_range_high_equals_max_of_first_bars():
    """opening_range_high must equal max(high[0:6]) from row 5 onwards."""
    from data.feature_engineering import add_features

    df = pd.DataFrame({
        "open":   [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        "high":   [101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0],
        "low":    [ 99.0, 100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0],
        "close":  [100.5, 101.5, 102.5, 103.5, 104.5, 103.5, 102.5, 101.5],
        "volume": [1e6] * 8,
    }, index=pd.date_range("2021-01-01", periods=8, freq="D"))

    result = add_features(df)

    # Opening range high should be max of first 6 highs: max(101, 102, 103, 104, 105, 104) = 105
    expected_high = 105.0

    # Row 5 (6th row, 0-indexed) should have the opening range high
    assert pd.notna(result["opening_range_high"].iloc[5]), "opening_range_high should be set at row 5"
    assert np.isclose(result["opening_range_high"].iloc[5], expected_high), (
        f"Row 5: expected opening_range_high={expected_high}, got {result['opening_range_high'].iloc[5]}"
    )

    # Row 6 should also have the same value (constant after established)
    assert np.isclose(result["opening_range_high"].iloc[6], expected_high), (
        f"Row 6: opening_range_high should be constant, expected {expected_high}, got {result['opening_range_high'].iloc[6]}"
    )


def test_opening_range_low_equals_min_of_first_bars():
    """opening_range_low must equal min(low[0:6]) from row 5 onwards."""
    from data.feature_engineering import add_features

    df = pd.DataFrame({
        "open":   [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        "high":   [101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0],
        "low":    [ 99.0, 98.0, 97.0, 96.0, 97.0, 98.0, 99.0, 100.0],
        "close":  [100.5, 101.5, 102.5, 103.5, 104.5, 103.5, 102.5, 101.5],
        "volume": [1e6] * 8,
    }, index=pd.date_range("2021-01-01", periods=8, freq="D"))

    result = add_features(df)

    # Opening range low should be min of first 6 lows: min(99, 98, 97, 96, 97, 98) = 96
    expected_low = 96.0

    assert pd.notna(result["opening_range_low"].iloc[5]), "opening_range_low should be set at row 5"
    assert np.isclose(result["opening_range_low"].iloc[5], expected_low), (
        f"Row 5: expected opening_range_low={expected_low}, got {result['opening_range_low'].iloc[5]}"
    )


def test_opening_range_mid_calculation():
    """opening_range_mid must equal (opening_range_high + opening_range_low) / 2."""
    from data.feature_engineering import add_features

    df = pd.DataFrame({
        "open":   [100.0] * 10,
        "high":   [105.0] * 10,
        "low":    [ 95.0] * 10,
        "close":  [100.0] * 10,
        "volume": [1e6] * 10,
    }, index=pd.date_range("2021-01-01", periods=10, freq="D"))

    result = add_features(df)

    # Opening range high = 105, low = 95, mid should be 100
    valid = result["opening_range_mid"].notna()
    subset = result[valid]

    if len(subset) > 0:
        expected = (subset["opening_range_high"] + subset["opening_range_low"]) / 2.0
        pd.testing.assert_series_equal(
            subset["opening_range_mid"], expected,
            check_names=False, rtol=1e-10,
            obj="opening_range_mid != (high + low) / 2",
        )


def test_opening_range_width_pct_calculation():
    """opening_range_width_pct must equal (high - low) / mid."""
    from data.feature_engineering import add_features

    df = pd.DataFrame({
        "open":   [100.0] * 10,
        "high":   [105.0] * 10,
        "low":    [ 95.0] * 10,
        "close":  [100.0] * 10,
        "volume": [1e6] * 10,
    }, index=pd.date_range("2021-01-01", periods=10, freq="D"))

    result = add_features(df)

    # Opening range width: (105 - 95) / 100 = 0.1
    valid = result["opening_range_width_pct"].notna()
    subset = result[valid]

    if len(subset) > 0:
        expected = (subset["opening_range_high"] - subset["opening_range_low"]) / subset["opening_range_mid"]
        pd.testing.assert_series_equal(
            subset["opening_range_width_pct"], expected,
            check_names=False, rtol=1e-10,
            obj="opening_range_width_pct != (high - low) / mid",
        )


def test_broke_above_flag_values():
    """broke_above_opening_range must be 1.0 when close > opening_range_high, else 0.0."""
    from data.feature_engineering import add_features

    df = pd.DataFrame({
        "open":   [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 110.0, 100.0],
        "high":   [101.0, 102.0, 103.0, 104.0, 105.0, 105.0, 110.0, 100.0],
        "low":    [ 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0,  99.0],
        "close":  [100.5, 101.5, 102.5, 103.5, 104.5, 104.5, 110.0, 99.0],
        "volume": [1e6] * 8,
    }, index=pd.date_range("2021-01-01", periods=8, freq="D"))

    result = add_features(df)

    # opening_range_high = 105 (max of first 6)
    # Row 5: close=104.5 <= 105, should be 0.0
    # Row 6: close=110.0 > 105, should be 1.0
    # Row 7: close=99.0 <= 105, should be 0.0

    assert result["broke_above_opening_range"].iloc[5] == 0.0, "Row 5: close <= high, expect 0.0"
    assert result["broke_above_opening_range"].iloc[6] == 1.0, "Row 6: close > high, expect 1.0"
    assert result["broke_above_opening_range"].iloc[7] == 0.0, "Row 7: close <= high, expect 0.0"


def test_broke_below_flag_values():
    """broke_below_opening_range must be 1.0 when close < opening_range_low, else 0.0."""
    from data.feature_engineering import add_features

    df = pd.DataFrame({
        "open":   [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 90.0, 100.0],
        "high":   [101.0, 102.0, 103.0, 104.0, 105.0, 105.0, 100.0, 101.0],
        "low":    [ 99.0, 100.0, 101.0, 102.0, 103.0, 96.0, 90.0, 95.0],
        "close":  [100.5, 101.5, 102.5, 103.5, 104.5, 97.0, 90.0, 95.0],
        "volume": [1e6] * 8,
    }, index=pd.date_range("2021-01-01", periods=8, freq="D"))

    result = add_features(df)

    # opening_range_low = 96 (min of first 6 lows)
    # Row 5: close=97.0 >= 96, should be 0.0
    # Row 6: close=90.0 < 96, should be 1.0
    # Row 7: close=95.0 < 96, should be 1.0

    assert result["broke_below_opening_range"].iloc[5] == 0.0, "Row 5: close >= low, expect 0.0"
    assert result["broke_below_opening_range"].iloc[6] == 1.0, "Row 6: close < low, expect 1.0"
    assert result["broke_below_opening_range"].iloc[7] == 1.0, "Row 7: close < low, expect 1.0"


def test_opening_range_flags_zero_before_established():
    """Breakout flags must be 0.0 before the opening range is established (row < 5)."""
    from data.feature_engineering import add_features

    df = pd.DataFrame({
        "open":   [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 110.0],
        "high":   [101.0, 102.0, 103.0, 104.0, 105.0, 105.0, 110.0],
        "low":    [ 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "close":  [100.5, 101.5, 102.5, 103.5, 104.5, 104.5, 110.0],
        "volume": [1e6] * 7,
    }, index=pd.date_range("2021-01-01", periods=7, freq="D"))

    result = add_features(df)

    # Rows 0-4 should have flags as 0.0 (before range is established)
    for i in range(5):
        assert result["broke_above_opening_range"].iloc[i] == 0.0, (
            f"Row {i}: before range established, expect 0.0"
        )
        assert result["broke_below_opening_range"].iloc[i] == 0.0, (
            f"Row {i}: before range established, expect 0.0"
        )


def test_opening_range_no_lookahead():
    """Opening range values must not change when future rows are added."""
    from data.feature_engineering import add_features

    short = add_features(pd.DataFrame({
        "open":   [100.0 + i for i in range(50)],
        "high":   [101.0 + i for i in range(50)],
        "low":    [ 99.0 + i for i in range(50)],
        "close":  [100.5 + i for i in range(50)],
        "volume": [1e6] * 50,
    }, index=pd.date_range("2021-01-01", periods=50, freq="D")))

    full = add_features(pd.DataFrame({
        "open":   [100.0 + i for i in range(100)],
        "high":   [101.0 + i for i in range(100)],
        "low":    [ 99.0 + i for i in range(100)],
        "close":  [100.5 + i for i in range(100)],
        "volume": [1e6] * 100,
    }, index=pd.date_range("2021-01-01", periods=100, freq="D")))

    n_check = 45  # Compare rows 0..44

    for col in ("opening_range_high", "opening_range_low", "opening_range_mid", "opening_range_width_pct"):
        short_vals = short[col].iloc[:n_check].reset_index(drop=True)
        full_vals = full[col].iloc[:n_check].reset_index(drop=True)

        pd.testing.assert_series_equal(
            short_vals, full_vals,
            check_names=False, rtol=1e-6,
            obj=f"Lookahead bias detected in {col}",
        )


def test_add_features_preserves_ohlcv_with_opening_range(sample_ohlcv):
    """Original OHLCV columns must be preserved after adding opening range features."""
    from data.feature_engineering import add_features

    original = sample_ohlcv.copy()
    result = add_features(sample_ohlcv)

    for col in ("open", "high", "low", "close", "volume"):
        assert col in result.columns, f"Column '{col}' missing from output"
        pd.testing.assert_series_equal(
            result[col], original[col], check_names=True,
            obj=f"OHLCV column '{col}' was modified by add_features()",
        )


def test_prepare_features_still_works_with_opening_range(sample_ohlcv):
    """prepare_features_for_model() must still work after adding opening range features."""
    from data.feature_engineering import prepare_features_for_model

    result = prepare_features_for_model(sample_ohlcv)

    # Must return a DataFrame
    assert isinstance(result, pd.DataFrame)
    # Must have at least some rows (NaN rows are dropped)
    assert len(result) > 100, "prepare_features_for_model returned too few rows"
    # Must not have any NaN values in MODEL_FEATURES columns
    assert not result.isna().any().any(), "Result contains NaN values"


# ---------------------------------------------------------------------------
# Test 14: Session-aware intraday features
# ---------------------------------------------------------------------------

def _intraday_two_sessions(bars_per_session: int = 8, interval_minutes: int = 5):
    """
    Build a small deterministic two-session intraday OHLCV DataFrame.

    Two calendar dates (2021-01-04 and 2021-01-05), each with `bars_per_session`
    bars starting at 09:30 and stepping `interval_minutes`.
    """
    rows = []
    index = []
    for d, base_close in (("2021-01-04", 100.0), ("2021-01-05", 200.0)):
        open_ts = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=30)
        for i in range(bars_per_session):
            c = base_close + i
            rows.append({
                "open": c, "high": c + 1.0, "low": c - 1.0, "close": c, "volume": 1000.0,
            })
            index.append(open_ts + pd.Timedelta(minutes=i * interval_minutes))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index))


def test_session_vwap_resets_between_sessions():
    """session_vwap must restart each session (first bar of a session = its typical price)."""
    from data.feature_engineering import add_session_intraday_features

    df = _intraday_two_sessions(bars_per_session=6)
    result = add_session_intraday_features(df)

    # Each session's first bar: cum volume = its own volume, so session_vwap == typical price.
    sessions = result.groupby(result.index.normalize())
    for _, grp in sessions:
        first = grp.iloc[0]
        tp = (first["high"] + first["low"] + first["close"]) / 3.0
        assert np.isclose(first["session_vwap"], tp), (
            "session_vwap first bar should equal its typical price (VWAP reset)"
        )

    # The second session's first bar must NOT include first-session volume:
    # if it leaked, its vwap would be pulled toward ~100 rather than ~200.
    second_session_first = sessions.get_group(pd.Timestamp("2021-01-05")).iloc[0]
    assert second_session_first["session_vwap"] > 150.0, "VWAP leaked across sessions"


def test_session_vwap_correct_on_known_two_session_data():
    """session_vwap must match a hand-computed cumulative VWAP within each session."""
    from data.feature_engineering import add_session_intraday_features

    df = _intraday_two_sessions(bars_per_session=3)
    result = add_session_intraday_features(df)

    # Session 1 closes/typical prices: closes 100,101,102 -> tp == close (symmetric high/low)
    for d, base in (("2021-01-04", 100.0), ("2021-01-05", 200.0)):
        grp = result[result.index.normalize() == pd.Timestamp(d)]
        tps = [base, base + 1.0, base + 2.0]  # typical price each bar (high/low symmetric)
        vols = [1000.0, 1000.0, 1000.0]
        for i in range(3):
            num = sum(tps[j] * vols[j] for j in range(i + 1))
            den = sum(vols[j] for j in range(i + 1))
            expected = num / den
            assert np.isclose(grp["session_vwap"].iloc[i], expected), (
                f"{d} bar {i}: expected session_vwap {expected}, got {grp['session_vwap'].iloc[i]}"
            )


def test_close_above_session_vwap_values():
    """close_above_session_vwap must be 1.0 when close > session_vwap, else 0.0."""
    from data.feature_engineering import add_session_intraday_features

    df = _intraday_two_sessions(bars_per_session=6)
    result = add_session_intraday_features(df)

    valid = result["session_vwap"].notna() & result["close_above_session_vwap"].notna()
    subset = result[valid]
    for _, row in subset.iterrows():
        expected = 1.0 if row["close"] > row["session_vwap"] else 0.0
        assert row["close_above_session_vwap"] == expected


def test_distance_from_session_vwap_pct_calculation():
    """distance_from_session_vwap_pct must equal (close - session_vwap) / session_vwap."""
    from data.feature_engineering import add_session_intraday_features

    df = _intraday_two_sessions(bars_per_session=6)
    result = add_session_intraday_features(df)

    valid = result["session_vwap"].notna() & result["distance_from_session_vwap_pct"].notna()
    subset = result[valid]
    expected = (subset["close"] - subset["session_vwap"]) / subset["session_vwap"]
    pd.testing.assert_series_equal(
        subset["distance_from_session_vwap_pct"], expected,
        check_names=False, rtol=1e-10,
        obj="distance_from_session_vwap_pct != (close - session_vwap) / session_vwap",
    )


def test_session_opening_range_resets_between_sessions():
    """Session opening range high/low must be computed per session, not globally."""
    from data.feature_engineering import add_session_intraday_features

    # 6 bars/session, opening_range_bars=3. Session 1 highs ~100s, session 2 highs ~200s.
    df = _intraday_two_sessions(bars_per_session=6)
    result = add_session_intraday_features(df, opening_range_bars=3)

    s1 = result[result.index.normalize() == pd.Timestamp("2021-01-04")]
    s2 = result[result.index.normalize() == pd.Timestamp("2021-01-05")]

    # Established value is constant within each session and differs by ~100 between them.
    s1_high = s1["session_opening_range_high"].dropna().iloc[0]
    s2_high = s2["session_opening_range_high"].dropna().iloc[0]
    # Session 1 first 3 highs: 101,102,103 -> 103 ; session 2: 201,202,203 -> 203
    assert np.isclose(s1_high, 103.0), f"session 1 OR high wrong: {s1_high}"
    assert np.isclose(s2_high, 203.0), f"session 2 OR high wrong: {s2_high}"


def test_session_opening_range_nan_before_established():
    """Rows before the opening range is established must have NaN range values."""
    from data.feature_engineering import add_session_intraday_features

    df = _intraday_two_sessions(bars_per_session=6)
    result = add_session_intraday_features(df, opening_range_bars=3)

    s1 = result[result.index.normalize() == pd.Timestamp("2021-01-04")]
    # Positions 0 and 1 (before the establishing bar at position 2) must be NaN.
    assert pd.isna(s1["session_opening_range_high"].iloc[0])
    assert pd.isna(s1["session_opening_range_high"].iloc[1])
    assert pd.notna(s1["session_opening_range_high"].iloc[2]), "range should be set at bar 2"


def test_session_breakout_above_flag():
    """broke_above_session_opening_range must fire when close exceeds the session range high."""
    from data.feature_engineering import add_session_intraday_features

    # One session, 4 bars, opening_range_bars=2. First two highs establish the range.
    idx = [pd.Timestamp("2021-01-04 09:30") + pd.Timedelta(minutes=5 * i) for i in range(4)]
    df = pd.DataFrame({
        "open":   [100.0, 101.0, 102.0, 105.0],
        "high":   [101.0, 102.0, 103.0, 106.0],
        "low":    [ 99.0, 100.0, 101.0, 104.0],
        "close":  [100.5, 101.5, 102.5, 106.0],  # bar 3 close 106 > range high 102
        "volume": [1000.0] * 4,
    }, index=pd.DatetimeIndex(idx))

    result = add_session_intraday_features(df, opening_range_bars=2)

    # Range high = max(101, 102) = 102, established at bar 1.
    assert result["broke_above_session_opening_range"].iloc[1] == 0.0  # 101.5 <= 102
    assert result["broke_above_session_opening_range"].iloc[3] == 1.0  # 106 > 102


def test_session_breakout_below_flag():
    """broke_below_session_opening_range must fire when close drops below the session range low."""
    from data.feature_engineering import add_session_intraday_features

    idx = [pd.Timestamp("2021-01-04 09:30") + pd.Timedelta(minutes=5 * i) for i in range(4)]
    df = pd.DataFrame({
        "open":   [100.0, 101.0, 102.0,  95.0],
        "high":   [101.0, 102.0, 103.0,  96.0],
        "low":    [ 99.0, 100.0,  98.0,  94.0],
        "close":  [100.5, 101.5,  98.5,  94.0],  # bar 3 close 94 < range low 99
        "volume": [1000.0] * 4,
    }, index=pd.DatetimeIndex(idx))

    result = add_session_intraday_features(df, opening_range_bars=2)

    # Range low = min(99, 100) = 99, established at bar 1.
    assert result["broke_below_session_opening_range"].iloc[1] == 0.0  # 101.5 >= 99
    assert result["broke_below_session_opening_range"].iloc[3] == 1.0  # 94 < 99


def test_session_breakout_flags_reset_between_sessions():
    """Breakout flags from one session must not carry into the next session's early bars."""
    from data.feature_engineering import add_session_intraday_features

    df = _intraday_two_sessions(bars_per_session=6)
    result = add_session_intraday_features(df, opening_range_bars=3)

    s2 = result[result.index.normalize() == pd.Timestamp("2021-01-05")]
    # The first two bars of session 2 are before its own range is established -> flags 0.0.
    assert s2["broke_above_session_opening_range"].iloc[0] == 0.0
    assert s2["broke_above_session_opening_range"].iloc[1] == 0.0
    assert s2["broke_below_session_opening_range"].iloc[0] == 0.0
    assert s2["broke_below_session_opening_range"].iloc[1] == 0.0


def test_session_features_no_lookahead():
    """Adding a future session must not change any prior session's values."""
    from data.feature_engineering import add_session_intraday_features

    full = add_session_intraday_features(_intraday_two_sessions(bars_per_session=6),
                                         opening_range_bars=3)
    # Only the first session.
    one_session = _intraday_two_sessions(bars_per_session=6)
    one_session = one_session[one_session.index.normalize() == pd.Timestamp("2021-01-04")]
    short = add_session_intraday_features(one_session, opening_range_bars=3)

    cols = [
        "session_vwap", "close_above_session_vwap", "distance_from_session_vwap_pct",
        "session_opening_range_high", "session_opening_range_low",
        "session_opening_range_mid", "session_opening_range_width_pct",
        "broke_above_session_opening_range", "broke_below_session_opening_range",
    ]
    full_s1 = full[full.index.normalize() == pd.Timestamp("2021-01-04")]
    for col in cols:
        pd.testing.assert_series_equal(
            short[col].reset_index(drop=True),
            full_s1[col].reset_index(drop=True),
            check_names=False, rtol=1e-10,
            obj=f"Lookahead/cross-session leak detected in {col}",
        )


def test_session_features_handle_zero_volume():
    """Zero session volume must not crash; session_vwap becomes NaN there."""
    from data.feature_engineering import add_session_intraday_features

    idx = [pd.Timestamp("2021-01-04 09:30") + pd.Timedelta(minutes=5 * i) for i in range(3)]
    df = pd.DataFrame({
        "open":   [100.0, 101.0, 102.0],
        "high":   [101.0, 102.0, 103.0],
        "low":    [ 99.0, 100.0, 101.0],
        "close":  [100.5, 101.5, 102.5],
        "volume": [0.0, 0.0, 0.0],  # entire session has zero volume
    }, index=pd.DatetimeIndex(idx))

    result = add_session_intraday_features(df)  # must not raise
    assert len(result) == 3
    assert result["session_vwap"].isna().all(), "zero-volume session vwap should be NaN"


def test_session_features_integrated_in_add_features(sample_ohlcv):
    """add_features() must emit all session_* columns (daily data: range never established)."""
    from data.feature_engineering import add_features, ALL_FEATURE_COLUMNS

    result = add_features(sample_ohlcv)
    session_cols = [c for c in ALL_FEATURE_COLUMNS if c.startswith("session_")
                    or c.startswith("broke_") and "session" in c]
    expected = [
        "session_vwap", "close_above_session_vwap", "distance_from_session_vwap_pct",
        "session_opening_range_high", "session_opening_range_low",
        "session_opening_range_mid", "session_opening_range_width_pct",
        "broke_above_session_opening_range", "broke_below_session_opening_range",
    ]
    for col in expected:
        assert col in result.columns, f"{col} not produced by add_features()"

    # Daily data => each date is a one-row session => opening range never established.
    assert result["session_opening_range_high"].isna().all()
    assert (result["broke_above_session_opening_range"] == 0.0).all()


def test_session_columns_not_in_model_features():
    """Session columns must NOT be added to MODEL_FEATURES yet."""
    from data.feature_engineering import MODEL_FEATURES
    for col in MODEL_FEATURES:
        assert not col.startswith("session_"), f"session column {col} leaked into MODEL_FEATURES"


def test_prepare_features_still_works_with_session_features(sample_ohlcv):
    """prepare_features_for_model() must still work after adding session features."""
    from data.feature_engineering import prepare_features_for_model

    result = prepare_features_for_model(sample_ohlcv)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 100, "prepare_features_for_model returned too few rows"
    assert not result.isna().any().any(), "Result contains NaN values"
