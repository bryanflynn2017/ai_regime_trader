"""
Tests for strategy/day_trade_setups.py.

The detector is a watchlist tool only — it never places orders. These tests
verify detection rules, risk/reward math, confidence bounds, safe handling of
malformed stops, and the no-mutation guarantee.

Most tests build a DataFrame that ALREADY contains the feature columns the
detector reads, so the rules can be exercised deterministically without relying
on add_features() producing specific values. One test verifies that
add_features() IS invoked when those columns are absent.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from strategy.day_trade_setups import (
    DayTradeSetup,
    DayTradeSetupDetector,
    SetupType,
    compute_risk_reward,
)


# ---------------------------------------------------------------------------
# Helpers: build a featured DataFrame the detector can read directly
# ---------------------------------------------------------------------------

_FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "vwap", "rel_volume_20", "rel_volume_50", "unusual_volume",
    "trend_score", "rsi_14",
    "close_above_vwap", "distance_from_vwap_pct",
    "opening_range_high", "opening_range_low", "opening_range_mid",
    "opening_range_width_pct",
    "broke_above_opening_range", "broke_below_opening_range",
]


def _featured_df(last: dict, prev: dict | None = None, n_pad: int = 3) -> pd.DataFrame:
    """
    Build a small DataFrame with explicit feature columns.

    `last` and `prev` are dicts of column -> value for the final two rows.
    Padding rows (neutral, no-setup) are prepended so len >= 2 comfortably.
    Any column not supplied defaults to a neutral value.
    """
    neutral = {
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6,
        "vwap": 100.0, "rel_volume_20": 1.0, "rel_volume_50": 1.0,
        "unusual_volume": 0.0, "trend_score": 0.0, "rsi_14": 50.0,
        "close_above_vwap": 0.0, "distance_from_vwap_pct": 0.0,
        "opening_range_high": 101.0, "opening_range_low": 99.0,
        "opening_range_mid": 100.0, "opening_range_width_pct": 0.02,
        "broke_above_opening_range": 0.0, "broke_below_opening_range": 0.0,
    }

    rows = []
    for _ in range(n_pad):
        rows.append(dict(neutral))
    rows.append({**neutral, **(prev or {})})
    rows.append({**neutral, **last})

    idx = pd.date_range("2021-06-01", periods=len(rows), freq="D")
    return pd.DataFrame(rows, index=idx)[_FEATURE_COLUMNS]


def _orb_last() -> dict:
    """Row that triggers an opening-range breakout long."""
    return {
        "close": 110.0, "vwap": 105.0, "rel_volume_20": 1.6,
        "trend_score": 0.6, "broke_above_opening_range": 1.0,
        "opening_range_mid": 100.0, "opening_range_high": 102.0,
        "opening_range_low": 98.0, "close_above_vwap": 1.0,
        "unusual_volume": 0.0, "rsi_14": 60.0,
    }


def _vwap_reclaim_rows() -> tuple[dict, dict]:
    """(prev, last) that trigger a VWAP reclaim long but not ORB/HRVC."""
    prev = {"close": 99.0, "vwap": 100.0}
    last = {
        "close": 101.0, "vwap": 100.0, "rel_volume_20": 1.3,
        "trend_score": 0.1, "broke_above_opening_range": 0.0,
        "close_above_vwap": 1.0, "unusual_volume": 0.0, "rsi_14": 55.0,
    }
    return prev, last


def _hrvc_rows() -> tuple[dict, dict]:
    """(prev, last) that trigger high rel-volume continuation but not ORB/reclaim."""
    # prev close > prev vwap so the VWAP-reclaim precondition fails.
    prev = {"close": 119.0, "vwap": 109.0}
    last = {
        "close": 120.0, "vwap": 110.0, "rel_volume_20": 2.5,
        "trend_score": 0.6, "broke_above_opening_range": 0.0,
        "close_above_vwap": 1.0, "unusual_volume": 1.0, "rsi_14": 60.0,
    }
    return prev, last


# ---------------------------------------------------------------------------
# Test 1: DayTradeSetup object can be created
# ---------------------------------------------------------------------------

def test_day_trade_setup_can_be_created():
    setup = DayTradeSetup(
        ticker="SPY",
        timestamp=pd.Timestamp("2021-06-01"),
        setup_type=SetupType.NO_SETUP,
        is_valid=False,
        confidence=0.0,
        entry_trigger=0.0,
        stop_loss=0.0,
        target_1=0.0,
        target_2=0.0,
        risk_reward_1=0.0,
        risk_reward_2=0.0,
        reason="manual",
    )
    assert setup.ticker == "SPY"
    assert setup.setup_type == SetupType.NO_SETUP
    assert setup.warnings == []
    assert setup.metadata == {}


# ---------------------------------------------------------------------------
# Test 2: NO_SETUP returned when data is insufficient
# ---------------------------------------------------------------------------

def test_no_setup_when_data_insufficient():
    detector = DayTradeSetupDetector()

    # Single raw OHLCV row — far too little to compute features.
    df = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0],
         "close": [100.0], "volume": [1e6]},
        index=pd.date_range("2021-06-01", periods=1, freq="D"),
    )
    setup = detector.detect_latest("SPY", df)
    assert setup.setup_type == SetupType.NO_SETUP
    assert setup.is_valid is False


def test_no_setup_when_empty_dataframe():
    detector = DayTradeSetupDetector()
    setup = detector.detect_latest("SPY", pd.DataFrame())
    assert setup.setup_type == SetupType.NO_SETUP
    assert setup.is_valid is False


# ---------------------------------------------------------------------------
# Test 3: Opening range breakout long is detected
# ---------------------------------------------------------------------------

def test_opening_range_breakout_detected():
    detector = DayTradeSetupDetector()
    df = _featured_df(_orb_last())

    setup = detector.detect_latest("SPY", df)

    assert setup.setup_type == SetupType.OPENING_RANGE_BREAKOUT_LONG
    assert setup.is_valid is True
    assert setup.entry_trigger == 110.0
    # stop = min(opening_range_mid=100, vwap=105) = 100
    assert setup.stop_loss == 100.0
    assert setup.stop_loss < setup.entry_trigger


# ---------------------------------------------------------------------------
# Test 4: VWAP reclaim long is detected
# ---------------------------------------------------------------------------

def test_vwap_reclaim_detected():
    detector = DayTradeSetupDetector()
    prev, last = _vwap_reclaim_rows()
    df = _featured_df(last, prev)

    setup = detector.detect_latest("QQQ", df)

    assert setup.setup_type == SetupType.VWAP_RECLAIM_LONG
    assert setup.is_valid is True
    assert setup.entry_trigger == 101.0
    # stop = vwap * 0.995 = 99.5
    assert np.isclose(setup.stop_loss, 99.5)
    assert setup.stop_loss < setup.entry_trigger


# ---------------------------------------------------------------------------
# Test 5: High relative volume continuation long is detected
# ---------------------------------------------------------------------------

def test_high_rel_volume_continuation_detected():
    detector = DayTradeSetupDetector()
    prev, last = _hrvc_rows()
    df = _featured_df(last, prev)

    setup = detector.detect_latest("NVDA", df)

    assert setup.setup_type == SetupType.HIGH_RELATIVE_VOLUME_CONTINUATION_LONG
    assert setup.is_valid is True
    assert setup.entry_trigger == 120.0
    # stop = max(vwap=110, close*0.98=117.6) = 117.6
    assert np.isclose(setup.stop_loss, 117.6)
    assert setup.stop_loss < setup.entry_trigger


# ---------------------------------------------------------------------------
# Test 6: Invalid stop (not below entry) handled safely
# ---------------------------------------------------------------------------

def test_invalid_stop_above_entry_is_handled():
    """A stop at/above entry must invalidate the setup with a warning, not crash.

    The live rules always yield a stop below entry, so this guard is exercised
    by driving the construction helper with a malformed stop directly.
    """
    detector = DayTradeSetupDetector()
    last_row = pd.Series(_featured_df(_orb_last()).iloc[-1])

    setup = detector._build_long_setup(
        ticker="SPY",
        ts=pd.Timestamp("2021-06-05"),
        setup_type=SetupType.OPENING_RANGE_BREAKOUT_LONG,
        entry=100.0,
        stop=105.0,             # stop ABOVE entry — malformed
        last=last_row,
        reason="forced",
    )
    assert setup.is_valid is False
    assert setup.risk_reward_1 == 0.0
    assert setup.risk_reward_2 == 0.0
    assert any("not below entry" in w for w in setup.warnings)


def test_compute_risk_reward_safe_on_bad_stop():
    # Zero / negative risk must not divide by zero.
    assert compute_risk_reward(entry=100.0, stop=100.0, target=110.0) == 0.0
    assert compute_risk_reward(entry=100.0, stop=105.0, target=110.0) == 0.0


# ---------------------------------------------------------------------------
# Test 7: risk_reward_1 and risk_reward_2 are correct
# ---------------------------------------------------------------------------

def test_risk_reward_values_are_correct():
    detector = DayTradeSetupDetector()
    df = _featured_df(_orb_last())

    setup = detector.detect_latest("SPY", df)

    # entry=110, stop=100 -> R=10; target_1=130 (2R), target_2=140 (3R)
    assert np.isclose(setup.target_1, 130.0)
    assert np.isclose(setup.target_2, 140.0)
    assert np.isclose(setup.risk_reward_1, 2.0)
    assert np.isclose(setup.risk_reward_2, 3.0)


# ---------------------------------------------------------------------------
# Test 8: confidence is between 0 and 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows_fn", [
    lambda: (_featured_df(_orb_last())),
    lambda: (_featured_df(*reversed(_vwap_reclaim_rows()))),
    lambda: (_featured_df(*reversed(_hrvc_rows()))),
])
def test_confidence_within_bounds(rows_fn):
    detector = DayTradeSetupDetector()
    setup = detector.detect_latest("SPY", rows_fn())
    assert 0.0 <= setup.confidence <= 1.0


def test_no_setup_confidence_is_zero():
    detector = DayTradeSetupDetector()
    df = _featured_df({"rel_volume_20": 0.8})  # weak, nothing triggers
    setup = detector.detect_latest("SPY", df)
    assert setup.setup_type == SetupType.NO_SETUP
    assert setup.confidence == 0.0


# ---------------------------------------------------------------------------
# Test 9: detect_many returns one setup per ticker
# ---------------------------------------------------------------------------

def test_detect_many_one_per_ticker():
    detector = DayTradeSetupDetector()
    prev_v, last_v = _vwap_reclaim_rows()
    data = {
        "SPY": _featured_df(_orb_last()),
        "QQQ": _featured_df(last_v, prev_v),
        "WEAK": _featured_df({"rel_volume_20": 0.5}),
    }
    results = detector.detect_many(data)

    assert len(results) == 3
    tickers = [s.ticker for s in results]
    assert tickers == ["SPY", "QQQ", "WEAK"]
    assert results[0].setup_type == SetupType.OPENING_RANGE_BREAKOUT_LONG
    assert results[1].setup_type == SetupType.VWAP_RECLAIM_LONG
    assert results[2].setup_type == SetupType.NO_SETUP


# ---------------------------------------------------------------------------
# Test 10: input DataFrame is not mutated
# ---------------------------------------------------------------------------

def test_input_dataframe_not_mutated():
    detector = DayTradeSetupDetector()
    df = _featured_df(_orb_last())
    before = df.copy(deep=True)

    detector.detect_latest("SPY", df)

    pd.testing.assert_frame_equal(df, before)


def test_input_raw_ohlcv_not_mutated():
    """Even when add_features() runs internally, the caller's frame is untouched."""
    detector = DayTradeSetupDetector()
    rng = np.random.default_rng(7)
    n = 40
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.integers(1e6, 5e6, n).astype(float),
    }, index=pd.date_range("2021-01-01", periods=n, freq="B"))
    before = df.copy(deep=True)

    detector.detect_latest("SPY", df)

    pd.testing.assert_frame_equal(df, before)
    # No feature columns leaked into the caller's frame.
    assert "vwap" not in df.columns


# ---------------------------------------------------------------------------
# Test 11: detector calls add_features when feature columns are missing
# ---------------------------------------------------------------------------

def test_detector_calls_add_features_when_missing():
    detector = DayTradeSetupDetector()
    rng = np.random.default_rng(3)
    n = 40
    close = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    raw = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.integers(1e6, 5e6, n).astype(float),
    }, index=pd.date_range("2021-01-01", periods=n, freq="B"))

    from data.feature_engineering import add_features as real_add_features
    called = {"n": 0}

    def spy_add_features(df):
        called["n"] += 1
        return real_add_features(df)

    with patch("strategy.day_trade_setups.add_features", spy_add_features):
        setup = detector.detect_latest("SPY", raw)

    assert called["n"] == 1, "add_features should be called exactly once"
    assert isinstance(setup, DayTradeSetup)
    assert isinstance(setup.setup_type, SetupType)


def test_detector_skips_add_features_when_present():
    """If feature columns already exist, add_features must NOT be called."""
    detector = DayTradeSetupDetector()
    df = _featured_df(_orb_last())

    with patch("strategy.day_trade_setups.add_features") as mock_add:
        detector.detect_latest("SPY", df)

    mock_add.assert_not_called()


# ---------------------------------------------------------------------------
# Test 12: no setup detected when volume is weak
# ---------------------------------------------------------------------------

def test_no_setup_when_volume_weak():
    detector = DayTradeSetupDetector()
    # Strong trend and above VWAP, but relative volume is far too low and no
    # breakout / reclaim conditions hold.
    last = {
        "close": 110.0, "vwap": 105.0, "rel_volume_20": 0.7,
        "trend_score": 0.8, "broke_above_opening_range": 0.0,
        "close_above_vwap": 1.0, "unusual_volume": 0.0, "rsi_14": 60.0,
    }
    prev = {"close": 109.0, "vwap": 104.0}  # prev above vwap -> no reclaim
    df = _featured_df(last, prev)

    setup = detector.detect_latest("SPY", df)
    assert setup.setup_type == SetupType.NO_SETUP
    assert setup.is_valid is False


# ---------------------------------------------------------------------------
# Test 13: Session-aware feature resolution
# ---------------------------------------------------------------------------

_SESSION_COLUMNS = [
    "session_vwap", "close_above_session_vwap", "distance_from_session_vwap_pct",
    "session_opening_range_high", "session_opening_range_low",
    "session_opening_range_mid", "session_opening_range_width_pct",
    "broke_above_session_opening_range", "broke_below_session_opening_range",
]


def _intraday_df(last: dict, prev: dict | None = None, n_pad: int = 3) -> pd.DataFrame:
    """
    Build an intraday DataFrame: many bars on a SINGLE session date so the
    latest bar belongs to a multi-bar session. Includes both generic and
    session_* feature columns. Any column not supplied defaults to neutral.
    """
    neutral = {
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6,
        "vwap": 100.0, "rel_volume_20": 1.0, "rel_volume_50": 1.0,
        "unusual_volume": 0.0, "trend_score": 0.0, "rsi_14": 50.0,
        "close_above_vwap": 0.0, "distance_from_vwap_pct": 0.0,
        "opening_range_high": 101.0, "opening_range_low": 99.0,
        "opening_range_mid": 100.0, "opening_range_width_pct": 0.02,
        "broke_above_opening_range": 0.0, "broke_below_opening_range": 0.0,
        # Session columns (default: established, neutral).
        "session_vwap": 100.0, "close_above_session_vwap": 0.0,
        "distance_from_session_vwap_pct": 0.0,
        "session_opening_range_high": 101.0, "session_opening_range_low": 99.0,
        "session_opening_range_mid": 100.0, "session_opening_range_width_pct": 0.02,
        "broke_above_session_opening_range": 0.0,
        "broke_below_session_opening_range": 0.0,
    }
    rows = []
    for _ in range(n_pad):
        rows.append(dict(neutral))
    rows.append({**neutral, **(prev or {})})
    rows.append({**neutral, **last})

    cols = list(neutral.keys())
    # All bars share one date (2021-06-01) at 5-minute steps -> multi-bar session.
    idx = pd.date_range("2021-06-01 09:30", periods=len(rows), freq="5min")
    return pd.DataFrame(rows, index=idx)[cols]


def test_resolver_uses_session_vwap_when_available():
    """get_vwap_column must return session_vwap for an intraday multi-bar session."""
    detector = DayTradeSetupDetector()
    df = _intraday_df({"session_vwap": 123.0})
    assert detector.get_vwap_column(df) == "session_vwap"


def test_resolver_falls_back_to_generic_vwap_when_session_nan():
    """get_vwap_column must fall back to generic vwap when session_vwap is NaN."""
    detector = DayTradeSetupDetector()
    df = _intraday_df({"session_vwap": np.nan})
    assert detector.get_vwap_column(df) == "vwap"


def test_resolver_uses_generic_vwap_for_daily_single_bar_sessions():
    """Daily data (one bar per date) must keep generic vwap even if session_vwap is present."""
    detector = DayTradeSetupDetector()
    df = _featured_df(_orb_last())  # daily index, no session cols
    assert detector.get_vwap_column(df) == "vwap"


def test_resolver_uses_session_opening_range_when_available():
    """get_opening_range_high_column must return the session column when established."""
    detector = DayTradeSetupDetector()
    df = _intraday_df({"session_opening_range_high": 105.0})
    assert detector.get_opening_range_high_column(df) == "session_opening_range_high"
    # The breakout flag column is tied to the same source.
    assert detector.get_broke_above_opening_range_column(df) == "broke_above_session_opening_range"


def test_resolver_falls_back_to_generic_opening_range_when_session_nan():
    """get_opening_range_high_column must fall back to generic when session range is NaN."""
    detector = DayTradeSetupDetector()
    df = _intraday_df({"session_opening_range_high": np.nan})
    assert detector.get_opening_range_high_column(df) == "opening_range_high"
    assert detector.get_broke_above_opening_range_column(df) == "broke_above_opening_range"


def test_orb_detected_using_session_columns():
    """ORB must trigger from session columns even when generic breakout is 0."""
    detector = DayTradeSetupDetector()
    df = _intraday_df({
        "close": 110.0,
        # Session columns drive the breakout:
        "session_vwap": 105.0, "session_opening_range_mid": 100.0,
        "session_opening_range_high": 102.0, "session_opening_range_low": 98.0,
        "broke_above_session_opening_range": 1.0,
        # Generic columns would NOT trigger (broke flag 0):
        "vwap": 105.0, "opening_range_mid": 100.0,
        "broke_above_opening_range": 0.0,
        "rel_volume_20": 1.6, "trend_score": 0.6,
    })

    setup = detector.detect_latest("SPY", df)

    assert setup.setup_type == SetupType.OPENING_RANGE_BREAKOUT_LONG
    assert setup.is_valid is True
    assert setup.entry_trigger == 110.0
    assert setup.stop_loss == 100.0  # min(session_mid=100, session_vwap=105)
    assert setup.metadata["opening_range_source"] == "session"
    assert setup.metadata["vwap_source"] == "session"


def test_vwap_reclaim_detected_using_session_columns():
    """VWAP reclaim must use session_vwap (current and previous bar)."""
    detector = DayTradeSetupDetector()
    prev = {"close": 99.0, "session_vwap": 100.0, "vwap": 98.0}  # generic would fail (99 > 98)
    last = {
        "close": 101.0, "session_vwap": 100.0, "vwap": 98.0,
        "rel_volume_20": 1.3, "trend_score": 0.1,
        "broke_above_session_opening_range": 0.0, "broke_above_opening_range": 0.0,
    }
    df = _intraday_df(last, prev)

    setup = detector.detect_latest("QQQ", df)

    assert setup.setup_type == SetupType.VWAP_RECLAIM_LONG
    assert setup.is_valid is True
    assert setup.entry_trigger == 101.0
    assert np.isclose(setup.stop_loss, 99.5)  # session_vwap * 0.995
    assert setup.metadata["vwap_source"] == "session"
    assert np.isclose(setup.metadata["vwap_value"], 100.0)


def test_hrvc_detected_using_session_columns():
    """High rel-volume continuation must use session_vwap / close_above_session_vwap."""
    detector = DayTradeSetupDetector()
    prev = {"close": 119.0, "session_vwap": 109.0}  # prev above session vwap -> no reclaim
    last = {
        "close": 120.0,
        "session_vwap": 110.0, "close_above_session_vwap": 1.0,
        # Generic columns would FAIL HRVC (close below generic vwap, flag 0):
        "vwap": 125.0, "close_above_vwap": 0.0,
        "rel_volume_20": 2.5, "trend_score": 0.6, "rsi_14": 60.0,
        "broke_above_session_opening_range": 0.0, "broke_above_opening_range": 0.0,
    }
    df = _intraday_df(last, prev)

    setup = detector.detect_latest("NVDA", df)

    assert setup.setup_type == SetupType.HIGH_RELATIVE_VOLUME_CONTINUATION_LONG
    assert setup.is_valid is True
    assert setup.entry_trigger == 120.0
    assert np.isclose(setup.stop_loss, 117.6)  # max(session_vwap=110, close*0.98=117.6)
    assert setup.metadata["vwap_source"] == "session"


def test_daily_generic_behavior_records_generic_source():
    """Daily detection must report generic sources in metadata (backward compat)."""
    detector = DayTradeSetupDetector()
    df = _featured_df(_orb_last())  # daily, generic columns only

    setup = detector.detect_latest("SPY", df)

    assert setup.setup_type == SetupType.OPENING_RANGE_BREAKOUT_LONG
    assert setup.metadata["vwap_source"] == "generic"
    assert setup.metadata["opening_range_source"] == "generic"
    assert np.isclose(setup.metadata["vwap_value"], 105.0)


def test_intraday_input_dataframe_not_mutated():
    """Session-aware detection must not mutate the caller's intraday DataFrame."""
    detector = DayTradeSetupDetector()
    df = _intraday_df({
        "close": 110.0, "session_vwap": 105.0,
        "session_opening_range_mid": 100.0, "session_opening_range_high": 102.0,
        "broke_above_session_opening_range": 1.0,
        "rel_volume_20": 1.6, "trend_score": 0.6,
    })
    before = df.copy(deep=True)

    detector.detect_latest("SPY", df)

    pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------------------
# Test 14: Near-miss diagnostics (Phase: diagnostic output)
# ---------------------------------------------------------------------------

def test_no_setup_includes_diagnostics():
    """NO_SETUP results must include diagnostic metadata."""
    detector = DayTradeSetupDetector()
    # Weak setup: high volume but low trend and low confidence
    df = _featured_df({"rel_volume_20": 0.5, "trend_score": -0.5})

    setup = detector.detect_latest("SPY", df)

    assert setup.setup_type == SetupType.NO_SETUP
    assert "diagnostics" in setup.metadata
    assert "checked_setups" in setup.metadata
    assert "nearest_setup_type" in setup.metadata


def test_diagnostics_list_failed_conditions():
    """Diagnostics must list which conditions failed for the nearest setup."""
    detector = DayTradeSetupDetector()
    # Fails ORB: low volume
    df = _featured_df({
        "close": 110.0, "vwap": 105.0, "rel_volume_20": 1.0,
        "trend_score": 0.6, "broke_above_opening_range": 1.0,
    })

    setup = detector.detect_latest("SPY", df)

    assert setup.setup_type == SetupType.NO_SETUP
    orb_failures = setup.metadata["diagnostics"].get("opening_range_breakout_long", [])
    assert any("rel_volume_20" in f for f in orb_failures), "Should list rel_volume failure"


def test_nearest_setup_type_is_populated():
    """nearest_setup_type should be the setup that came closest."""
    detector = DayTradeSetupDetector()
    # Close to ORB but fails one condition
    df = _featured_df({"rel_volume_20": 1.0})

    setup = detector.detect_latest("SPY", df)

    assert setup.setup_type == SetupType.NO_SETUP
    assert setup.metadata.get("nearest_setup_type") is not None


def test_valid_setup_unchanged():
    """Valid setups should have minimal diagnostic metadata."""
    detector = DayTradeSetupDetector()
    df = _featured_df(_orb_last())

    setup = detector.detect_latest("SPY", df)

    assert setup.setup_type == SetupType.OPENING_RANGE_BREAKOUT_LONG
    assert setup.is_valid is True
    # Valid setups may have minimal metadata
    assert "diagnostics" not in setup.metadata or len(setup.metadata.get("diagnostics", {})) == 0


def test_passed_conditions_count():
    """Metadata should include passed_conditions_count for nearest setup."""
    detector = DayTradeSetupDetector()
    df = _featured_df({"rel_volume_20": 0.5})

    setup = detector.detect_latest("SPY", df)

    assert setup.setup_type == SetupType.NO_SETUP
    assert "passed_conditions_count" in setup.metadata
    assert setup.metadata["passed_conditions_count"] >= 0
