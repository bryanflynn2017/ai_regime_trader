"""
Tests for strategy/strategy_orchestrator.py and strategy/allocation.py.

Coverage:
1.  BUY signal generated in a clear bull setup.
2.  HOLD when regime confidence is too low (< 0.40).
3.  HOLD when data is insufficient (< 51 rows).
4.  HOLD when price is below 200 SMA.
5.  HOLD when RSI is above 70.
6.  HOLD when volume_ratio is below 1.0.
7.  HOLD when price is outside configured MIN_PRICE / MAX_PRICE.
8.  REDUCE when price closes below 50 SMA.
9.  SELL when crash regime is detected.
10. Euphoria regime rejects overextended entries (RSI > 70).
11. BUY signal includes valid stop_loss and take_profit.
12. Every signal includes a human-readable reason string.
13. Signal metadata includes rejection_reasons list.
14. generate_signals handles multiple tickers.
15. Input DataFrame is not mutated by generate_signal.
16. AllocationPolicy returns correct target exposure by regime.
17. Uncertain regime cuts target exposure and position multiplier by 50%.
18. Confidence below 0.40 blocks new longs.
19. Bear regime blocks new longs by default.
20. All generated signal actions are valid SignalAction enum values.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from regime.fallback_regime import RegimeResult
from strategy.signals import RegimeLabel, SignalAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_regime(
    label: RegimeLabel,
    confidence: float = 0.75,
    is_stable: bool = True,
    is_uncertain: bool = False,
) -> RegimeResult:
    """Create a mock-style RegimeResult with controlled values."""
    return RegimeResult(
        label=label,
        confidence=confidence,
        timestamp=pd.Timestamp("2023-01-01"),
        method="fallback",
        reason="test fixture",
        features_used=["sma_200"],
        is_stable=is_stable,
        is_uncertain=is_uncertain,
    )


def _mock_regime_engine(result: RegimeResult) -> MagicMock:
    """Return a mock regime engine that always returns `result`."""
    mock = MagicMock()
    mock.classify_latest.return_value = result
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bull_features_df() -> pd.DataFrame:
    """
    300 rows of features already computed for a moderate uptrend.
    The LAST ROW is manually set to pass all BUY criteria with known values.
    """
    from data.feature_engineering import add_features

    rng = np.random.default_rng(seed=11)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    log_ret = rng.normal(0.0004, 0.009, n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    spread = close * 0.003
    df = pd.DataFrame({
        "open":   close * (1 + rng.normal(0, 0.002, n)),
        "high":   close + spread,
        "low":    np.maximum(close - spread, 0.01),
        "close":  close,
        "volume": np.full(n, 10_000_000.0),
    }, index=dates)
    df = add_features(df)

    # Force the last row to known-good BUY values so the test is deterministic.
    # We override individual features without touching OHLCV.
    last = df.index[-1]
    sma_50_val  = float(df.loc[last, "sma_50"])
    sma_200_val = float(df.loc[last, "sma_200"])
    # Ensure close is above both SMAs
    df.loc[last, "close"]      = max(close[-1], sma_50_val * 1.03, sma_200_val * 1.05)
    df.loc[last, "rsi_14"]     = 58.0   # mid-range [45, 70]
    df.loc[last, "volume_ratio"] = 1.25
    df.loc[last, "volume"]     = 12_000_000.0
    df.loc[last, "trend_score"]  = 0.55
    df.loc[last, "drawdown_from_252_high"] = -0.05  # minimal drawdown
    df.loc[last, "sma_20"]     = sma_50_val * 1.02  # sma_20 > sma_50

    return df


@pytest.fixture
def insufficient_df() -> pd.DataFrame:
    """Only 30 rows — below _MIN_ROWS threshold."""
    n = 30
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": np.full(n, 100.0), "high": np.full(n, 101.0),
        "low":  np.full(n, 99.0),  "close": np.full(n, 100.0),
        "volume": np.full(n, 5_000_000.0),
    }, index=dates)


# ---------------------------------------------------------------------------
# Test 1: BUY in clear bull setup
# ---------------------------------------------------------------------------

def test_buy_in_clear_bull_setup(bull_features_df):
    """StrategyOrchestrator must produce a BUY for a textbook bull setup."""
    from strategy.strategy_orchestrator import StrategyOrchestrator

    orch = StrategyOrchestrator(
        regime_engine=_mock_regime_engine(_make_regime(RegimeLabel.BULL, confidence=0.75))
    )
    signal = orch.generate_signal("AAPL", bull_features_df)

    assert signal.action == SignalAction.BUY, (
        f"Expected BUY in bull setup, got {signal.action.value}. Reason: {signal.reason}"
    )
    assert signal.ticker == "AAPL"


# ---------------------------------------------------------------------------
# Test 2: HOLD when regime confidence < 0.40
# ---------------------------------------------------------------------------

def test_hold_low_regime_confidence(bull_features_df):
    from strategy.strategy_orchestrator import StrategyOrchestrator

    orch = StrategyOrchestrator(
        regime_engine=_mock_regime_engine(_make_regime(RegimeLabel.BULL, confidence=0.30))
    )
    signal = orch.generate_signal("SPY", bull_features_df)

    assert signal.action == SignalAction.HOLD, (
        f"Expected HOLD for low confidence, got {signal.action.value}"
    )


# ---------------------------------------------------------------------------
# Test 3: HOLD when data is insufficient
# ---------------------------------------------------------------------------

def test_hold_insufficient_data(insufficient_df):
    from strategy.strategy_orchestrator import StrategyOrchestrator

    signal = StrategyOrchestrator().generate_signal("MSFT", insufficient_df)

    assert signal.action == SignalAction.HOLD
    assert "Insufficient" in signal.reason or "insufficient" in signal.reason.lower()


# ---------------------------------------------------------------------------
# Test 4: HOLD when price is below 200 SMA
# ---------------------------------------------------------------------------

def test_hold_price_below_200_sma(bull_features_df):
    """
    BUY requires close > sma_200.  When that fails and no SELL/REDUCE conditions
    independently fire, the result must be HOLD.

    Strategy: set close BETWEEN sma_50 and sma_200 so that:
      - close < sma_200  → BUY blocked
      - close > sma_50   → REDUCE (close < sma_50) does NOT fire
    """
    from strategy.strategy_orchestrator import StrategyOrchestrator

    df = bull_features_df.copy()
    last = df.index[-1]
    sma_200_val = float(df.loc[last, "sma_200"])
    sma_50_val  = float(df.loc[last, "sma_50"])

    # close is 2% below sma_200 but we ensure it stays above sma_50
    new_close = sma_200_val * 0.98
    # Push sma_50 down so close > sma_50 (avoids REDUCE trigger)
    df.loc[last, "close"]  = new_close
    df.loc[last, "sma_50"] = new_close * 0.97  # ensure close > sma_50

    orch = StrategyOrchestrator(
        regime_engine=_mock_regime_engine(_make_regime(RegimeLabel.NEUTRAL, confidence=0.55))
    )
    signal = orch.generate_signal("GOOGL", df)

    assert signal.action == SignalAction.HOLD, (
        f"Expected HOLD (close < sma_200 blocks BUY), got {signal.action.value}. "
        f"Reason: {signal.reason}"
    )


# ---------------------------------------------------------------------------
# Test 5: HOLD when RSI > 70
# ---------------------------------------------------------------------------

def test_hold_rsi_overbought(bull_features_df):
    from strategy.strategy_orchestrator import StrategyOrchestrator

    df = bull_features_df.copy()
    df.iloc[-1, df.columns.get_loc("rsi_14")] = 75.0  # Above 70 threshold

    orch = StrategyOrchestrator(
        regime_engine=_mock_regime_engine(_make_regime(RegimeLabel.BULL, confidence=0.75))
    )
    signal = orch.generate_signal("NVDA", df)

    assert signal.action == SignalAction.HOLD, (
        f"Expected HOLD (RSI overbought), got {signal.action.value}"
    )
    # RSI should appear in the rejection reason
    reason_lower = signal.reason.lower()
    assert "rsi" in reason_lower or "overbought" in reason_lower, (
        f"Expected RSI/overbought in reason, got: {signal.reason}"
    )


# ---------------------------------------------------------------------------
# Test 6: HOLD when volume_ratio < 1.0
# ---------------------------------------------------------------------------

def test_hold_low_volume_ratio(bull_features_df):
    from strategy.strategy_orchestrator import StrategyOrchestrator

    df = bull_features_df.copy()
    df.iloc[-1, df.columns.get_loc("volume_ratio")] = 0.7  # Below 1.0

    orch = StrategyOrchestrator(
        regime_engine=_mock_regime_engine(_make_regime(RegimeLabel.BULL, confidence=0.75))
    )
    signal = orch.generate_signal("AMD", df)

    assert signal.action == SignalAction.HOLD, (
        f"Expected HOLD (low volume_ratio), got {signal.action.value}"
    )


# ---------------------------------------------------------------------------
# Test 7: HOLD when price outside configured MIN_PRICE / MAX_PRICE
# ---------------------------------------------------------------------------

def test_hold_price_below_min_price(bull_features_df):
    """
    BUY requires close >= MIN_PRICE.  When price is below the floor, BUY is
    blocked and HOLD is returned.

    We also push sma_50 and sma_200 below close so REDUCE (close < sma_50)
    does not fire before the price-range check.
    """
    from strategy.strategy_orchestrator import StrategyOrchestrator
    from config.settings import settings

    df = bull_features_df.copy()
    last = df.index[-1]
    penny_price = settings.MIN_PRICE - 1.0  # e.g. 4.0

    df.loc[last, "close"]   = penny_price
    df.loc[last, "sma_50"]  = penny_price * 0.97   # close > sma_50 → no REDUCE
    df.loc[last, "sma_200"] = penny_price * 0.94   # close > sma_200 → no SELL

    orch = StrategyOrchestrator(
        regime_engine=_mock_regime_engine(_make_regime(RegimeLabel.BULL, confidence=0.75))
    )
    signal = orch.generate_signal("PENNY", df)

    assert signal.action == SignalAction.HOLD, (
        f"Expected HOLD (price below MIN_PRICE), got {signal.action.value}. "
        f"Reason: {signal.reason}"
    )


def test_hold_price_above_max_price(bull_features_df):
    from strategy.strategy_orchestrator import StrategyOrchestrator
    from config.settings import settings

    df = bull_features_df.copy()
    df.iloc[-1, df.columns.get_loc("close")] = settings.MAX_PRICE + 100.0

    orch = StrategyOrchestrator(
        regime_engine=_mock_regime_engine(_make_regime(RegimeLabel.BULL, confidence=0.75))
    )
    signal = orch.generate_signal("EXPENSIVE", df)

    assert signal.action == SignalAction.HOLD


# ---------------------------------------------------------------------------
# Test 8: REDUCE when price closes below 50 SMA
# ---------------------------------------------------------------------------

def test_reduce_when_price_below_50_sma(bull_features_df):
    from strategy.strategy_orchestrator import StrategyOrchestrator

    df = bull_features_df.copy()
    last = df.index[-1]
    sma_50_val = float(df.loc[last, "sma_50"])
    # Set close below sma_50 (but above sma_200 so no SELL fires)
    sma_200_val = float(df.loc[last, "sma_200"])
    df.loc[last, "close"] = sma_50_val * 0.97  # below sma_50
    # Ensure close is still above sma_200 to avoid SELL
    if df.loc[last, "close"] < sma_200_val:
        df.loc[last, "sma_200"] = sma_200_val * 0.90  # push sma_200 down

    # Use NEUTRAL regime — not CRASH (which would give SELL) and not BEAR with high conf
    orch = StrategyOrchestrator(
        regime_engine=_mock_regime_engine(_make_regime(RegimeLabel.NEUTRAL, confidence=0.55))
    )
    signal = orch.generate_signal("META", df)

    assert signal.action == SignalAction.REDUCE, (
        f"Expected REDUCE (close < sma_50), got {signal.action.value}. "
        f"Reason: {signal.reason}"
    )
    assert "sma_50" in signal.reason.lower() or "50" in signal.reason


# ---------------------------------------------------------------------------
# Test 9: SELL when crash regime detected
# ---------------------------------------------------------------------------

def test_sell_on_crash_regime(bull_features_df):
    from strategy.strategy_orchestrator import StrategyOrchestrator

    # High-confidence crash
    orch = StrategyOrchestrator(
        regime_engine=_mock_regime_engine(
            _make_regime(RegimeLabel.CRASH, confidence=0.85)
        )
    )
    signal = orch.generate_signal("SPY", bull_features_df)

    assert signal.action == SignalAction.SELL, (
        f"Expected SELL on CRASH regime, got {signal.action.value}. "
        f"Reason: {signal.reason}"
    )
    assert "crash" in signal.reason.lower()


# ---------------------------------------------------------------------------
# Test 10: Euphoria regime rejects overextended RSI > 70
# ---------------------------------------------------------------------------

def test_euphoria_rejects_overbought(bull_features_df):
    """
    Euphoria regime must block BUY when RSI > 70.

    Use RSI = 72 — above the euphoria cap (70) but below the generic
    REDUCE threshold (75) so REDUCE does not fire first.
    """
    from strategy.strategy_orchestrator import StrategyOrchestrator

    df = bull_features_df.copy()
    df.iloc[-1, df.columns.get_loc("rsi_14")] = 72.0  # 70 < rsi < 75

    orch = StrategyOrchestrator(
        regime_engine=_mock_regime_engine(
            _make_regime(RegimeLabel.EUPHORIA, confidence=0.70)
        )
    )
    signal = orch.generate_signal("TSLA", df)

    assert signal.action == SignalAction.HOLD, (
        f"Euphoria must reject RSI > 70, got {signal.action.value}"
    )


def test_euphoria_rejects_price_extended_above_sma50(bull_features_df):
    from strategy.strategy_orchestrator import StrategyOrchestrator

    df = bull_features_df.copy()
    last = df.index[-1]
    sma_50_val = float(df.loc[last, "sma_50"])
    # Price 20% above sma_50 — overextended in euphoria
    df.loc[last, "close"]  = sma_50_val * 1.20
    df.loc[last, "rsi_14"] = 65.0  # RSI fine, but price extension blocks it

    orch = StrategyOrchestrator(
        regime_engine=_mock_regime_engine(
            _make_regime(RegimeLabel.EUPHORIA, confidence=0.70)
        )
    )
    signal = orch.generate_signal("TSLA", df)

    assert signal.action == SignalAction.HOLD


# ---------------------------------------------------------------------------
# Test 11: BUY signal has valid stop_loss and take_profit
# ---------------------------------------------------------------------------

def test_buy_signal_has_stop_and_target(bull_features_df):
    from strategy.strategy_orchestrator import StrategyOrchestrator
    from config.settings import settings

    orch = StrategyOrchestrator(
        regime_engine=_mock_regime_engine(_make_regime(RegimeLabel.BULL, confidence=0.75))
    )
    signal = orch.generate_signal("AAPL", bull_features_df)

    assert signal.action == SignalAction.BUY, f"Expected BUY, got {signal.action.value}"
    assert signal.stop_loss > 0, "stop_loss must be positive"
    assert signal.take_profit > 0, "take_profit must be positive"
    assert signal.stop_loss < signal.entry_price, (
        f"stop_loss {signal.stop_loss:.2f} must be below entry {signal.entry_price:.2f}"
    )
    assert signal.take_profit > signal.entry_price, (
        f"take_profit {signal.take_profit:.2f} must be above entry {signal.entry_price:.2f}"
    )
    # Verify the percentages match settings
    assert abs(signal.stop_loss - signal.entry_price * (1 - settings.DEFAULT_STOP_LOSS_PCT)) < 0.01
    assert abs(signal.take_profit - signal.entry_price * (1 + settings.DEFAULT_TAKE_PROFIT_PCT)) < 0.01


# ---------------------------------------------------------------------------
# Test 12: Every signal has a human-readable reason string
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("regime_label,expected_action", [
    (RegimeLabel.BULL,    SignalAction.BUY),
    (RegimeLabel.CRASH,   SignalAction.SELL),
])
def test_signal_has_reason_string(regime_label, expected_action, bull_features_df):
    from strategy.strategy_orchestrator import StrategyOrchestrator

    orch = StrategyOrchestrator(
        regime_engine=_mock_regime_engine(_make_regime(regime_label, confidence=0.80))
    )
    signal = orch.generate_signal("SPY", bull_features_df)

    assert len(signal.reason) > 0, "reason must not be empty"
    assert signal.action.value in signal.reason or signal.action.value.lower() in signal.reason.lower()


# ---------------------------------------------------------------------------
# Test 13: Signal metadata includes rejection_reasons
# ---------------------------------------------------------------------------

def test_hold_signal_has_rejection_reasons(bull_features_df):
    """
    A HOLD caused by a BUY criterion failure must carry rejection_reasons in metadata.

    Use RSI = 72 — above BUY cap (70) but below REDUCE threshold (75)
    so REDUCE does not intercept first.
    """
    from strategy.strategy_orchestrator import StrategyOrchestrator

    df = bull_features_df.copy()
    df.iloc[-1, df.columns.get_loc("rsi_14")] = 72.0  # 70 < rsi < 75

    orch = StrategyOrchestrator(
        regime_engine=_mock_regime_engine(_make_regime(RegimeLabel.BULL, confidence=0.75))
    )
    signal = orch.generate_signal("AAPL", df)

    assert signal.action == SignalAction.HOLD
    assert "rejection_reasons" in signal.metadata, "metadata must have rejection_reasons key"
    assert isinstance(signal.metadata["rejection_reasons"], list)


def test_buy_signal_metadata_has_empty_rejections(bull_features_df):
    from strategy.strategy_orchestrator import StrategyOrchestrator

    orch = StrategyOrchestrator(
        regime_engine=_mock_regime_engine(_make_regime(RegimeLabel.BULL, confidence=0.75))
    )
    signal = orch.generate_signal("AAPL", bull_features_df)

    if signal.action == SignalAction.BUY:
        assert signal.metadata["rejection_reasons"] == [], (
            "BUY signal must have empty rejection_reasons"
        )


# ---------------------------------------------------------------------------
# Test 14: generate_signals handles multiple tickers
# ---------------------------------------------------------------------------

def test_generate_signals_multiple_tickers(bull_features_df):
    from strategy.strategy_orchestrator import StrategyOrchestrator

    tickers = ["AAPL", "MSFT", "SPY"]
    data_by_ticker = {t: bull_features_df.copy() for t in tickers}

    orch = StrategyOrchestrator(
        regime_engine=_mock_regime_engine(_make_regime(RegimeLabel.BULL, confidence=0.75))
    )
    signals = orch.generate_signals(data_by_ticker)

    assert len(signals) == len(tickers), (
        f"Expected {len(tickers)} signals, got {len(signals)}"
    )
    returned_tickers = {s.ticker for s in signals}
    assert returned_tickers == set(tickers)


# ---------------------------------------------------------------------------
# Test 15: Input DataFrame is not mutated
# ---------------------------------------------------------------------------

def test_generate_signal_does_not_mutate_input(bull_features_df):
    from strategy.strategy_orchestrator import StrategyOrchestrator

    original_cols  = set(bull_features_df.columns)
    original_shape = bull_features_df.shape

    orch = StrategyOrchestrator(
        regime_engine=_mock_regime_engine(_make_regime(RegimeLabel.BULL, confidence=0.75))
    )
    orch.generate_signal("AAPL", bull_features_df)

    assert set(bull_features_df.columns) == original_cols, (
        "generate_signal must not add columns to the input DataFrame"
    )
    assert bull_features_df.shape == original_shape


# ---------------------------------------------------------------------------
# Test 16: AllocationPolicy target exposure by regime
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("regime,expected_exposure", [
    (RegimeLabel.CRASH,    0.00),
    (RegimeLabel.BEAR,     0.10),
    (RegimeLabel.NEUTRAL,  0.25),
    (RegimeLabel.BULL,     0.50),
    (RegimeLabel.EUPHORIA, 0.25),
])
def test_allocation_target_exposure_by_regime(regime, expected_exposure):
    from strategy.allocation import AllocationPolicy

    policy = AllocationPolicy()
    result = policy.get_target_exposure(regime, confidence=0.75, is_uncertain=False)

    assert abs(result - expected_exposure) < 1e-9, (
        f"Expected exposure {expected_exposure} for {regime.value}, got {result}"
    )


# ---------------------------------------------------------------------------
# Test 17: Uncertain regime cuts exposure and position multiplier by 50%
# ---------------------------------------------------------------------------

def test_uncertain_regime_halves_exposure():
    from strategy.allocation import AllocationPolicy

    policy = AllocationPolicy()
    normal    = policy.get_target_exposure(RegimeLabel.BULL, 0.75, is_uncertain=False)
    uncertain = policy.get_target_exposure(RegimeLabel.BULL, 0.75, is_uncertain=True)

    assert abs(uncertain - normal * 0.50) < 1e-9, (
        f"Uncertain regime must halve exposure: normal={normal}, uncertain={uncertain}"
    )


def test_uncertain_regime_halves_position_multiplier():
    from strategy.allocation import AllocationPolicy

    policy = AllocationPolicy()
    normal    = policy.get_position_size_multiplier(RegimeLabel.BULL, 0.75, is_uncertain=False)
    uncertain = policy.get_position_size_multiplier(RegimeLabel.BULL, 0.75, is_uncertain=True)

    assert abs(uncertain - normal * 0.50) < 1e-9, (
        f"Uncertain regime must halve size multiplier: normal={normal}, uncertain={uncertain}"
    )


# ---------------------------------------------------------------------------
# Test 18: Confidence below 0.40 blocks new longs
# ---------------------------------------------------------------------------

def test_low_confidence_blocks_new_longs():
    from strategy.allocation import AllocationPolicy

    policy = AllocationPolicy()
    for regime in (RegimeLabel.NEUTRAL, RegimeLabel.BULL, RegimeLabel.EUPHORIA):
        result = policy.should_allow_new_longs(regime, confidence=0.35)
        assert result is False, (
            f"Confidence 0.35 must block new longs in {regime.value}, got {result}"
        )


def test_low_confidence_returns_zero_multiplier():
    from strategy.allocation import AllocationPolicy

    policy = AllocationPolicy()
    multiplier = policy.get_position_size_multiplier(RegimeLabel.BULL, confidence=0.35)
    assert multiplier == 0.0


# ---------------------------------------------------------------------------
# Test 19: Bear regime blocks new longs by default
# ---------------------------------------------------------------------------

def test_bear_regime_blocks_new_longs():
    from strategy.allocation import AllocationPolicy

    policy = AllocationPolicy()
    result = policy.should_allow_new_longs(RegimeLabel.BEAR, confidence=0.80)
    assert result is False, "Bear regime must block new longs regardless of confidence"


def test_bear_regime_reduce_flag():
    from strategy.allocation import AllocationPolicy

    policy = AllocationPolicy()
    assert policy.should_reduce_existing_positions(RegimeLabel.BEAR, confidence=0.65) is True


# ---------------------------------------------------------------------------
# Test 20: All generated signal actions are valid SignalAction enum values
# ---------------------------------------------------------------------------

def test_all_signal_actions_valid_enum(bull_features_df, insufficient_df):
    from strategy.strategy_orchestrator import StrategyOrchestrator

    scenarios = [
        (bull_features_df, _make_regime(RegimeLabel.BULL,    0.75)),
        (bull_features_df, _make_regime(RegimeLabel.CRASH,   0.85)),
        (bull_features_df, _make_regime(RegimeLabel.NEUTRAL, 0.25)),
        (insufficient_df,  _make_regime(RegimeLabel.BULL,    0.75)),
    ]

    valid_actions = set(SignalAction)

    for df, regime in scenarios:
        orch = StrategyOrchestrator(regime_engine=_mock_regime_engine(regime))
        signal = orch.generate_signal("SPY", df)
        assert signal.action in valid_actions, (
            f"Unexpected action {signal.action!r} — must be a SignalAction member"
        )
