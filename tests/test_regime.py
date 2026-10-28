"""
Tests for regime/fallback_regime.py and regime/hmm_engine.py.

Coverage:
1.  Fallback returns a valid RegimeResult.
2.  Fallback returns neutral + low confidence for insufficient data.
3.  Fallback classifies a clear bull market correctly.
4.  Fallback classifies a clear bear market correctly.
5.  Fallback classifies a crash-like market correctly.
6.  Fallback confidence is always in [0, 1].
7.  classify_series returns one regime label per row (after warmup).
8.  Stability filter marks flickering regimes as uncertain.
9.  HMMRegimeEngine.is_available() returns a bool.
10. HMM predict_latest returns a valid RegimeResult (or graceful fallback result).
11. HMM walk-forward: row 249 prediction unchanged when extra rows are appended.
12. HMM handles insufficient data gracefully (no exception, valid result).
13. HMM handles missing hmmlearn gracefully (mocked unavailability).
14. RegimeResult includes non-empty reason and features_used list.
15. Neither classifier mutates the input DataFrame.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from regime.fallback_regime import RegimeResult
from strategy.signals import RegimeLabel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_ohlcv() -> pd.DataFrame:
    """400 rows of generic random-walk OHLCV data."""
    rng = np.random.default_rng(seed=7)
    n = 400
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    log_ret = rng.normal(0.0003, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    spread = close * 0.005
    open_ = close * (1.0 + rng.normal(0, 0.003, n))
    high = np.maximum(close, open_) + spread
    low  = np.maximum(np.minimum(close, open_) - spread, 0.01)
    vol  = rng.integers(1_000_000, 50_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


@pytest.fixture
def bull_ohlcv() -> pd.DataFrame:
    """
    Synthetically clean bull-market data: 300-day steady uptrend.
    close well above SMA200, golden cross (sma50 > sma200), positive trend.
    """
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    # 0.12% daily return → ~30% annual; tight range
    close = 100.0 * np.exp(np.arange(n) * 0.0012)
    spread = close * 0.002
    return pd.DataFrame({
        "open":   close * 0.9995,
        "high":   close + spread,
        "low":    close - spread,
        "close":  close,
        "volume": np.full(n, 5_000_000, dtype=float),
    }, index=dates)


@pytest.fixture
def bear_ohlcv() -> pd.DataFrame:
    """
    Synthetically clean bear-market data: 300-day steady downtrend.
    close well below SMA200, death cross (sma50 < sma200).
    """
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    # –0.10% daily return
    close = 100.0 * np.exp(np.arange(n) * -0.0010)
    spread = close * 0.002
    return pd.DataFrame({
        "open":   close * 1.0005,
        "high":   close + spread,
        "low":    np.maximum(close - spread, 0.01),
        "close":  close,
        "volume": np.full(n, 5_000_000, dtype=float),
    }, index=dates)


@pytest.fixture
def crash_ohlcv() -> pd.DataFrame:
    """
    300 rows: gentle rise for 260 bars, then sudden –35% drop over 40 bars.
    After the crash: deep drawdown, high volatility, strongly negative trend.
    """
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed=99)

    # Phase 1: steady rise
    phase1 = np.exp(np.arange(260) * 0.0003) * 100.0
    # Phase 2: –35% crash with added noise
    drop = np.exp(-0.015 * np.arange(40))
    noise = rng.normal(0, 0.005, 40)
    phase2 = phase1[-1] * drop * (1 + noise)
    close = np.concatenate([phase1, phase2])

    spread = close * 0.003
    return pd.DataFrame({
        "open":   close * (1.0 + rng.normal(0, 0.003, n)),
        "high":   close + spread,
        "low":    np.maximum(close - spread, 0.01),
        "close":  close,
        "volume": np.full(n, 5_000_000, dtype=float),
    }, index=dates)


@pytest.fixture
def insufficient_ohlcv() -> pd.DataFrame:
    """Only 30 rows — not enough to compute sma_200."""
    n = 30
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    return pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.full(n, 1_000_000, dtype=float),
    }, index=dates)


# ---------------------------------------------------------------------------
# Test 1: Fallback returns a valid RegimeResult
# ---------------------------------------------------------------------------

def test_fallback_returns_regime_result(base_ohlcv):
    from regime.fallback_regime import FallbackRegimeClassifier
    clf = FallbackRegimeClassifier()
    result = clf.classify_latest(base_ohlcv)

    assert isinstance(result, RegimeResult), "classify_latest must return a RegimeResult"
    assert isinstance(result.label, RegimeLabel), f"label must be a RegimeLabel, got {type(result.label)}"
    assert result.label in RegimeLabel.__members__.values(), f"Unknown label: {result.label}"
    assert result.method == "fallback"
    assert isinstance(result.timestamp, pd.Timestamp)


# ---------------------------------------------------------------------------
# Test 2: Fallback returns neutral + low confidence for insufficient data
# ---------------------------------------------------------------------------

def test_fallback_insufficient_data(insufficient_ohlcv):
    from regime.fallback_regime import FallbackRegimeClassifier
    result = FallbackRegimeClassifier().classify_latest(insufficient_ohlcv)

    assert result.label == RegimeLabel.NEUTRAL, (
        f"Expected NEUTRAL for insufficient data, got {result.label}"
    )
    assert result.confidence < 0.25, (
        f"Expected low confidence for insufficient data, got {result.confidence:.3f}"
    )
    assert result.is_uncertain is True
    assert "Insufficient" in result.reason or "insufficient" in result.reason


# ---------------------------------------------------------------------------
# Test 3: Fallback classifies a clear bull market
# ---------------------------------------------------------------------------

def test_fallback_classifies_bull(bull_ohlcv):
    from regime.fallback_regime import FallbackRegimeClassifier
    result = FallbackRegimeClassifier().classify_latest(bull_ohlcv)

    assert result.label in (RegimeLabel.BULL, RegimeLabel.EUPHORIA), (
        f"Expected BULL or EUPHORIA for steady uptrend, got {result.label.value}. "
        f"Reason: {result.reason}"
    )
    assert result.confidence >= 0.40


# ---------------------------------------------------------------------------
# Test 4: Fallback classifies a clear bear market
# ---------------------------------------------------------------------------

def test_fallback_classifies_bear(bear_ohlcv):
    from regime.fallback_regime import FallbackRegimeClassifier
    result = FallbackRegimeClassifier().classify_latest(bear_ohlcv)

    assert result.label in (RegimeLabel.BEAR, RegimeLabel.CRASH), (
        f"Expected BEAR or CRASH for steady downtrend, got {result.label.value}. "
        f"Reason: {result.reason}"
    )
    assert result.confidence >= 0.40


# ---------------------------------------------------------------------------
# Test 5: Fallback classifies a crash-like market
# ---------------------------------------------------------------------------

def test_fallback_classifies_crash(crash_ohlcv):
    from regime.fallback_regime import FallbackRegimeClassifier
    result = FallbackRegimeClassifier().classify_latest(crash_ohlcv)

    assert result.label in (RegimeLabel.CRASH, RegimeLabel.BEAR), (
        f"Expected CRASH or BEAR after sharp drop, got {result.label.value}. "
        f"Reason: {result.reason}"
    )


# ---------------------------------------------------------------------------
# Test 6: Fallback confidence is always in [0, 1]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", ["base_ohlcv", "bull_ohlcv", "bear_ohlcv", "crash_ohlcv"])
def test_fallback_confidence_in_range(fixture_name, request):
    from regime.fallback_regime import FallbackRegimeClassifier
    ohlcv = request.getfixturevalue(fixture_name)
    result = FallbackRegimeClassifier().classify_latest(ohlcv)

    assert 0.0 <= result.confidence <= 1.0, (
        f"Confidence {result.confidence:.4f} out of [0, 1] for {fixture_name}"
    )


# ---------------------------------------------------------------------------
# Test 7: classify_series returns one regime per row (after warmup)
# ---------------------------------------------------------------------------

def test_classify_series_one_regime_per_row(base_ohlcv):
    from regime.fallback_regime import FallbackRegimeClassifier
    result = FallbackRegimeClassifier().classify_series(base_ohlcv)

    assert "regime_label" in result.columns
    assert len(result) == len(base_ohlcv), "Output must have same number of rows as input"

    # After warmup (row >= 51), every label must be a known value
    after_warmup = result.iloc[60:]
    valid_labels = {l.value for l in RegimeLabel}
    for lbl in after_warmup["regime_label"]:
        assert lbl in valid_labels, f"Unknown label in series: {lbl!r}"


# ---------------------------------------------------------------------------
# Test 8: Stability filter marks flickering regimes as uncertain
# ---------------------------------------------------------------------------

def test_stability_filter_uncertain(base_ohlcv):
    """
    When regime labels flip every few bars, is_uncertain must be True
    for the affected window.
    """
    from regime.fallback_regime import _compute_is_uncertain

    # Synthesise a rapidly flickering label series
    labels = pd.Series(
        ["neutral", "bull", "bear", "neutral", "crash", "bull",
         "bear", "neutral", "bull", "bear", "crash", "neutral",
         "bull", "bear", "neutral", "bull", "bear", "neutral",
         "bull", "bear", "crash"]
    )

    uncertain = _compute_is_uncertain(labels, n=10, max_unique=3)
    # The window from index 10 onward has >= 4 unique labels → uncertain
    assert uncertain.iloc[-1] is True or bool(uncertain.iloc[-1]), (
        "Expected is_uncertain=True for rapidly flickering label window"
    )


def test_stability_filter_stable(base_ohlcv):
    """A run of identical labels must be marked stable."""
    from regime.fallback_regime import _compute_is_stable

    labels = pd.Series(["bull"] * 20)
    stable = _compute_is_stable(labels, n=3)

    assert stable.iloc[-1] is True or bool(stable.iloc[-1]), (
        "Expected is_stable=True when all recent labels are identical"
    )


# ---------------------------------------------------------------------------
# Test 9: HMMRegimeEngine.is_available() returns a bool
# ---------------------------------------------------------------------------

def test_hmm_is_available_returns_bool():
    from regime.hmm_engine import HMMRegimeEngine
    result = HMMRegimeEngine.is_available()
    assert isinstance(result, bool), f"is_available() must return bool, got {type(result)}"


# ---------------------------------------------------------------------------
# Test 10: HMM predict_latest returns a valid RegimeResult (or fallback)
# ---------------------------------------------------------------------------

def test_hmm_predict_latest_valid_result(base_ohlcv):
    from regime.hmm_engine import HMMRegimeEngine
    engine = HMMRegimeEngine()
    result = engine.fit(base_ohlcv).predict_latest(base_ohlcv)

    assert isinstance(result, RegimeResult)
    assert result.label in RegimeLabel.__members__.values()
    assert 0.0 <= result.confidence <= 1.0
    assert result.method in ("hmm", "fallback")
    assert len(result.reason) > 0


# ---------------------------------------------------------------------------
# Test 11: Walk-forward no-lookahead — prediction at a given row must not
#          change when future rows are appended.
# ---------------------------------------------------------------------------

def test_hmm_walk_forward_no_lookahead(base_ohlcv):
    """
    Core no-lookahead test for the walk-forward prediction path.

    The regime label at row 379 must be identical whether the engine saw
    380 rows or 400 rows.

    Why 379 / 380 vs 400?
        drawdown_from_252_high is NaN for the first 251 rows.
        The HMM needs MIN_TRAINING_ROWS (100) valid feature rows before its
        first fit, so the first HMM prediction occurs around row 350.
        Using 380 vs 400 rows exercises the REAL HMM path (not fallback)
        and verifies no future data leaks backward.

    Why NOT 249 / 250 vs 350 (the old indices)?
        With only 250 rows, drawdown_from_252_high is NaN for ALL rows, so
        X_full is empty and every prediction falls back to the deterministic
        classifier.  That original test would pass regardless of the
        IndexError bug because the fallback path was always used.
    """
    from regime.hmm_engine import HMMRegimeEngine

    engine_short = HMMRegimeEngine(random_state=42)
    engine_full  = HMMRegimeEngine(random_state=42)

    # base_ohlcv has 400 rows; both slices share data for rows 0–379.
    short_result = engine_short.predict_series_walk_forward(base_ohlcv.iloc[:380])
    full_result  = engine_full.predict_series_walk_forward(base_ohlcv.iloc[:400])

    label_at_379_short = short_result["regime_label"].iloc[379]
    label_at_379_full  = full_result["regime_label"].iloc[379]

    assert label_at_379_short == label_at_379_full, (
        f"Lookahead detected: label at row 379 differs between 380-row run "
        f"({label_at_379_short!r}) and 400-row run ({label_at_379_full!r})."
    )


def test_hmm_walk_forward_uses_real_hmm_when_available(base_ohlcv):
    """
    When hmmlearn is available, the walk-forward must switch from fallback
    to the real HMM once enough valid training rows are accumulated.

    Specifically, for the 400-row base_ohlcv fixture:
    - drawdown_from_252_high is NaN for rows 0–250 (251 NaN rows)
    - 100 valid feature rows accumulate at bar ~350
    - From bar 350 onward the HMM is active

    If hmmlearn is NOT installed, the test is skipped.
    """
    from regime.hmm_engine import HMMRegimeEngine

    if not HMMRegimeEngine.is_available():
        import pytest
        pytest.skip("hmmlearn not installed — real HMM path not available")

    engine = HMMRegimeEngine(random_state=42)
    result = engine.predict_series_walk_forward(base_ohlcv)  # 400 rows

    hmm_rows = (result["regime_method"] == "hmm").sum()
    assert hmm_rows > 0, (
        "hmmlearn is installed but walk-forward used only the fallback classifier. "
        "Check MIN_TRAINING_ROWS and that drawdown_from_252_high is non-NaN "
        "for enough rows in the fixture."
    )


# ---------------------------------------------------------------------------
# Test 12: HMM handles insufficient data gracefully
# ---------------------------------------------------------------------------

def test_hmm_insufficient_data_no_exception(insufficient_ohlcv):
    from regime.hmm_engine import HMMRegimeEngine

    engine = HMMRegimeEngine()
    # fit on tiny data must not raise
    engine.fit(insufficient_ohlcv)
    # predict_latest on tiny data must return a valid result
    result = engine.predict_latest(insufficient_ohlcv)

    assert isinstance(result, RegimeResult)
    assert result.label in RegimeLabel.__members__.values()


# ---------------------------------------------------------------------------
# Test 13: HMM handles missing hmmlearn gracefully (mocked)
# ---------------------------------------------------------------------------

def test_hmm_missing_hmmlearn_uses_fallback(base_ohlcv):
    """
    Simulate hmmlearn being unavailable at runtime.
    The engine must return a valid RegimeResult from the fallback.
    """
    import regime.hmm_engine as hmm_mod

    with patch.object(hmm_mod, "_HMM_AVAILABLE", False):
        engine = hmm_mod.HMMRegimeEngine()
        engine.fit(base_ohlcv)   # must be a no-op, not raise
        result = engine.predict_latest(base_ohlcv)

    assert isinstance(result, RegimeResult)
    assert result.method == "fallback", (
        f"Expected method='fallback' when hmmlearn mocked out, got '{result.method}'"
    )


def test_hmm_walk_forward_missing_hmmlearn(base_ohlcv):
    """Walk-forward with mocked-out hmmlearn must produce fallback labels."""
    import regime.hmm_engine as hmm_mod

    with patch.object(hmm_mod, "_HMM_AVAILABLE", False):
        result = hmm_mod.HMMRegimeEngine().predict_series_walk_forward(base_ohlcv.iloc[:100])

    assert "regime_label" in result.columns
    assert len(result) == 100


# ---------------------------------------------------------------------------
# Test 14: RegimeResult has non-empty reason and features_used
# ---------------------------------------------------------------------------

def test_regime_result_has_reason_and_features(base_ohlcv):
    from regime.fallback_regime import FallbackRegimeClassifier
    result = FallbackRegimeClassifier().classify_latest(base_ohlcv)

    assert len(result.reason) > 0, "RegimeResult.reason must not be empty"
    assert isinstance(result.features_used, list), "features_used must be a list"


def test_regime_result_dataclass_fields():
    """Verify RegimeResult has all required fields."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(RegimeResult)}
    required = {
        "label", "confidence", "timestamp", "method", "reason",
        "features_used", "is_stable", "is_uncertain", "metadata",
    }
    missing = required - field_names
    assert not missing, f"RegimeResult is missing fields: {missing}"


# ---------------------------------------------------------------------------
# Test 15: Neither classifier mutates the input DataFrame
# ---------------------------------------------------------------------------

def test_fallback_does_not_mutate_input(base_ohlcv):
    from regime.fallback_regime import FallbackRegimeClassifier
    original_cols = list(base_ohlcv.columns)
    original_shape = base_ohlcv.shape

    FallbackRegimeClassifier().classify_latest(base_ohlcv)

    assert list(base_ohlcv.columns) == original_cols, (
        "classify_latest must not add columns to the input DataFrame"
    )
    assert base_ohlcv.shape == original_shape


def test_fallback_classify_series_does_not_mutate_input(base_ohlcv):
    from regime.fallback_regime import FallbackRegimeClassifier
    original_cols = set(base_ohlcv.columns)

    FallbackRegimeClassifier().classify_series(base_ohlcv)

    assert set(base_ohlcv.columns) == original_cols, (
        "classify_series must not modify the input DataFrame's columns"
    )


def test_hmm_does_not_mutate_input(base_ohlcv):
    from regime.hmm_engine import HMMRegimeEngine
    original_cols = set(base_ohlcv.columns)
    original_shape = base_ohlcv.shape

    engine = HMMRegimeEngine()
    engine.fit(base_ohlcv)
    engine.predict_latest(base_ohlcv)

    assert set(base_ohlcv.columns) == original_cols
    assert base_ohlcv.shape == original_shape
