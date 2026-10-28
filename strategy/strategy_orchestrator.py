"""
Strategy orchestrator — the single entry point for signal generation.

Responsibilities:
1. Ensure technical features are available (calls add_features if needed).
2. Classify the current market regime via the injected regime engine.
3. Apply allocation rules to determine whether new longs are allowed.
4. Evaluate ticker-specific BUY, SELL, and REDUCE criteria.
5. Return a fully populated Signal with a human-readable reason.
6. Log every rejection with an explicit reason.

This layer does NOT:
- Place orders (Phase 7).
- Check portfolio-level exposure (Phase 5 risk manager).
- Connect to any broker (Phase 7).

The signals produced here are CANDIDATES. The Phase 5 risk manager has
veto power over all of them before any order is sent.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import settings
from strategy.allocation import AllocationPolicy
from strategy.signals import RegimeLabel, Signal, SignalAction

logger = logging.getLogger(__name__)

# Minimum rows needed before we can attempt regime classification.
_MIN_ROWS = 51

# Required feature columns for a full BUY evaluation.
_REQUIRED_FEATURES = [
    "close", "sma_20", "sma_50", "sma_200",
    "rsi_14", "volume_ratio", "trend_score",
    "drawdown_from_252_high", "drawdown_from_20_high",
    "volume",
]


class StrategyOrchestrator:
    """
    Generates candidate trading signals from market data and regime context.

    Inject a custom regime_engine for testing or to swap between fallback/HMM:

        orch = StrategyOrchestrator(regime_engine=my_classifier)
        signal = orch.generate_signal("SPY", df)
    """

    def __init__(self, regime_engine: Optional[Any] = None) -> None:
        if regime_engine is None:
            from regime.fallback_regime import FallbackRegimeClassifier
            regime_engine = FallbackRegimeClassifier()
        self._regime_engine = regime_engine
        self._allocation = AllocationPolicy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signal(self, ticker: str, df: pd.DataFrame) -> Signal:
        """
        Generate a candidate Signal for a single ticker.

        The input DataFrame should contain at minimum raw OHLCV columns.
        Features will be computed automatically if absent.
        The input DataFrame is NEVER modified.

        Returns a Signal with action BUY / SELL / REDUCE / HOLD.
        All decisions (including rejections) are described in Signal.reason.
        """
        df = df.copy()

        # ---- Stage 1: Basic data validation ----
        if len(df) < _MIN_ROWS:
            return self._make_hold(
                ticker, df,
                f"Insufficient data: {len(df)} rows (need >= {_MIN_ROWS})",
            )

        if "close" not in df.columns or "volume" not in df.columns:
            return self._make_hold(
                ticker, df,
                "Missing required columns: close and/or volume",
            )

        # ---- Stage 2: Ensure features ----
        df = _ensure_features(df)

        row = df.iloc[-1]

        if not _has_valid_price_volume(row):
            return self._make_hold(
                ticker, df,
                f"Invalid price ({row.get('close', 'N/A')}) or volume ({row.get('volume', 'N/A')})",
            )

        # ---- Stage 3: Regime classification ----
        try:
            regime_result = self._regime_engine.classify_latest(df)
        except Exception as exc:
            logger.warning("Regime classification failed for %s: %s", ticker, exc)
            return self._make_hold(ticker, df, f"Regime classification error: {exc}")

        logger.debug(
            "%s regime=%s confidence=%.2f stable=%s uncertain=%s",
            ticker,
            regime_result.label.value,
            regime_result.confidence,
            regime_result.is_stable,
            regime_result.is_uncertain,
        )

        # ---- Stage 4: SELL conditions (highest priority) ----
        sell_reason = _check_sell_conditions(row, regime_result)
        if sell_reason:
            return self._make_sell(ticker, df, row, regime_result, sell_reason)

        # ---- Stage 5: REDUCE conditions ----
        reduce_reason = _check_reduce_conditions(row, regime_result)
        if reduce_reason:
            return self._make_reduce(ticker, df, row, regime_result, reduce_reason)

        # ---- Stage 6: BUY check ----
        # First gate: allocation policy
        allow_long = self._allocation.should_allow_new_longs(
            regime_result.label,
            regime_result.confidence,
            regime_result.is_uncertain,
        )
        if not allow_long:
            regime_str = regime_result.label.value
            conf_str   = f"{regime_result.confidence:.0%}"
            uncertain_str = " (regime uncertain)" if regime_result.is_uncertain else ""
            return self._make_hold(
                ticker, df,
                f"Regime {regime_str} (confidence {conf_str}){uncertain_str} does not allow new longs",
                rejection_reasons=[f"Regime gate blocked: {regime_str}"],
                regime_result=regime_result,
            )

        # Second gate: per-ticker technical criteria
        buy_passed, buy_reason, rejections = _check_buy_conditions(row, regime_result)
        if buy_passed:
            return self._make_buy(ticker, df, row, regime_result, buy_reason)

        # ---- Stage 7: Default HOLD ----
        return self._make_hold(
            ticker, df,
            buy_reason,  # Contains the rejection summary from _check_buy_conditions
            rejection_reasons=rejections,
            regime_result=regime_result,
        )

    def generate_signals(
        self, data_by_ticker: Dict[str, pd.DataFrame]
    ) -> List[Signal]:
        """
        Generate candidate signals for multiple tickers.

        Tickers that fail signal generation are logged as warnings and skipped.
        Never raises — always returns a (possibly empty) list.
        """
        signals: List[Signal] = []
        for ticker, df in data_by_ticker.items():
            try:
                sig = self.generate_signal(ticker, df)
                signals.append(sig)
                logger.debug("Signal for %s: %s — %s", ticker, sig.action.value, sig.reason)
            except Exception as exc:
                logger.warning("Signal generation failed for %s: %s", ticker, exc)
        return signals

    # ------------------------------------------------------------------
    # Private: signal constructors
    # ------------------------------------------------------------------

    def _make_hold(
        self,
        ticker: str,
        df: pd.DataFrame,
        reason: str,
        rejection_reasons: Optional[List[str]] = None,
        regime_result: Optional[Any] = None,
    ) -> Signal:
        close = _safe_float(df["close"].iloc[-1]) if len(df) > 0 and "close" in df.columns else 0.0
        regime = regime_result.label if regime_result else RegimeLabel.UNKNOWN
        ts     = pd.Timestamp(df.index[-1]) if len(df) > 0 else pd.Timestamp.now()

        return Signal(
            ticker=ticker,
            timestamp=ts,
            action=SignalAction.HOLD,
            confidence=0.0,
            reason=f"HOLD: {reason}",
            regime=regime,
            entry_price=close,
            stop_loss=0.0,
            take_profit=0.0,
            max_position_value=0.0,  # populated by Phase 5 risk manager
            metadata={
                "rejection_reasons": rejection_reasons or [],
                "regime": regime.value,
                "regime_confidence": regime_result.confidence if regime_result else 0.0,
            },
        )

    def _make_buy(
        self,
        ticker: str,
        df: pd.DataFrame,
        row: pd.Series,
        regime_result: Any,
        reason: str,
    ) -> Signal:
        close      = _safe_float(row.get("close", 0.0))
        stop_loss  = close * (1.0 - settings.DEFAULT_STOP_LOSS_PCT)
        take_profit = close * (1.0 + settings.DEFAULT_TAKE_PROFIT_PCT)

        target_exp  = self._allocation.get_target_exposure(
            regime_result.label, regime_result.confidence, regime_result.is_uncertain
        )
        size_mult   = self._allocation.get_position_size_multiplier(
            regime_result.label, regime_result.confidence, regime_result.is_uncertain
        )

        return Signal(
            ticker=ticker,
            timestamp=pd.Timestamp(df.index[-1]),
            action=SignalAction.BUY,
            confidence=float(np.clip(regime_result.confidence * size_mult, 0.0, 1.0)),
            reason=f"BUY: {reason}",
            regime=regime_result.label,
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_position_value=0.0,  # Phase 5 risk manager will populate
            metadata=_build_metadata(row, regime_result, target_exp, size_mult),
        )

    def _make_sell(
        self,
        ticker: str,
        df: pd.DataFrame,
        row: pd.Series,
        regime_result: Any,
        reason: str,
    ) -> Signal:
        close = _safe_float(row.get("close", 0.0))
        return Signal(
            ticker=ticker,
            timestamp=pd.Timestamp(df.index[-1]),
            action=SignalAction.SELL,
            confidence=float(regime_result.confidence),
            reason=f"SELL: {reason}",
            regime=regime_result.label,
            entry_price=close,
            stop_loss=0.0,
            take_profit=0.0,
            max_position_value=0.0,
            metadata=_build_metadata(row, regime_result, 0.0, 0.0),
        )

    def _make_reduce(
        self,
        ticker: str,
        df: pd.DataFrame,
        row: pd.Series,
        regime_result: Any,
        reason: str,
    ) -> Signal:
        close = _safe_float(row.get("close", 0.0))
        return Signal(
            ticker=ticker,
            timestamp=pd.Timestamp(df.index[-1]),
            action=SignalAction.REDUCE,
            confidence=float(regime_result.confidence),
            reason=f"REDUCE: {reason}",
            regime=regime_result.label,
            entry_price=close,
            stop_loss=0.0,
            take_profit=0.0,
            max_position_value=0.0,
            metadata=_build_metadata(row, regime_result, 0.0, 0.0),
        )


# ---------------------------------------------------------------------------
# Module-level check functions (pure, no side effects, easy to unit-test)
# ---------------------------------------------------------------------------

def _check_buy_conditions(
    row: pd.Series,
    regime_result: Any,
) -> Tuple[bool, str, List[str]]:
    """
    Evaluate all BUY entry criteria for a single bar.

    Returns:
        (passed, reason_string, rejection_reasons_list)
    """
    rejections: List[str] = []

    close  = _safe_float(row.get("close",  np.nan))
    sma_20 = _safe_float(row.get("sma_20", np.nan))
    sma_50 = _safe_float(row.get("sma_50", np.nan))
    sma_200= _safe_float(row.get("sma_200",np.nan))
    vol    = _safe_float(row.get("volume", np.nan))
    vr     = _safe_float(row.get("volume_ratio", np.nan))
    rsi    = _safe_float(row.get("rsi_14", np.nan))
    trend  = _safe_float(row.get("trend_score", np.nan))
    dd_252 = _safe_float(row.get("drawdown_from_252_high", np.nan))

    # -- Price range --
    if np.isnan(close) or close <= 0:
        rejections.append("close price invalid")
    else:
        if close < settings.MIN_PRICE:
            rejections.append(
                f"price {close:.2f} below MIN_PRICE {settings.MIN_PRICE}"
            )
        if close > settings.MAX_PRICE:
            rejections.append(
                f"price {close:.2f} above MAX_PRICE {settings.MAX_PRICE}"
            )

    # -- Volume (absolute liquidity) --
    if np.isnan(vol) or vol < settings.MIN_AVG_DAILY_VOLUME:
        rejections.append(
            f"volume {vol:,.0f} below MIN_AVG_DAILY_VOLUME {settings.MIN_AVG_DAILY_VOLUME:,.0f}"
        )

    # -- Relative volume --
    if np.isnan(vr) or vr < 1.0:
        rejections.append(f"volume_ratio {vr:.2f} below 1.0 (below-average volume)")

    # -- Price above both SMAs --
    if np.isnan(sma_50) or np.isnan(close) or close <= sma_50:
        rejections.append(
            f"close {close:.2f} not above sma_50 {sma_50:.2f}"
        )
    if np.isnan(sma_200) or np.isnan(close) or close <= sma_200:
        rejections.append(
            f"close {close:.2f} not above sma_200 {sma_200:.2f}"
        )

    # -- Bullish SMA alignment (sma_20 > sma_50) --
    if not np.isnan(sma_20) and not np.isnan(sma_50) and sma_20 <= sma_50:
        rejections.append(
            f"sma_20 {sma_20:.2f} not above sma_50 {sma_50:.2f} — no bullish SMA stack"
        )

    # -- RSI in acceptable range [45, 70] --
    if np.isnan(rsi):
        rejections.append("RSI is NaN (insufficient history)")
    elif rsi < 45:
        rejections.append(f"RSI {rsi:.1f} below 45 — insufficient momentum")
    elif rsi > 70:
        rejections.append(f"RSI {rsi:.1f} above 70 — overbought")

    # -- Positive trend --
    if np.isnan(trend) or trend <= 0:
        rejections.append(f"trend_score {trend:.2f} not positive")

    # -- Not in deep drawdown --
    if not np.isnan(dd_252) and dd_252 <= -0.20:
        rejections.append(
            f"drawdown from 252-high {dd_252:.1%} — too extended from long-term high"
        )

    # -- Regime confidence --
    if regime_result.confidence < 0.40:
        rejections.append(
            f"regime confidence {regime_result.confidence:.0%} below 0.40"
        )

    # -- Euphoria-specific: reject overextended entries --
    if regime_result.label == RegimeLabel.EUPHORIA:
        if not np.isnan(rsi) and rsi > 70:
            rejections.append(
                f"RSI {rsi:.1f} > 70 in euphoria regime — overextended, no entry"
            )
        if not np.isnan(sma_50) and not np.isnan(close) and close > sma_50 * 1.15:
            pct_above = (close / sma_50 - 1)
            rejections.append(
                f"close {pct_above:.1%} above sma_50 in euphoria — overextended"
            )

    if rejections:
        summary = "; ".join(rejections[:3])  # First 3 rejections in the reason string
        return False, f"BUY rejected: {summary}", rejections

    regime_str = regime_result.label.value
    return (
        True,
        (
            f"{regime_str} regime, price above 50/200 SMA, trend_score={trend:.2f}, "
            f"RSI={rsi:.1f}, volume_ratio={vr:.2f}"
        ),
        [],
    )


def _check_sell_conditions(
    row: pd.Series,
    regime_result: Any,
) -> Optional[str]:
    """
    Return a SELL reason string if exit conditions are met, else None.

    SELL is the most drastic action — reserved for crash detection and
    severe drawdown breaches.
    """
    reasons: List[str] = []

    close  = _safe_float(row.get("close",  np.nan))
    sma_200= _safe_float(row.get("sma_200", np.nan))
    dd_252 = _safe_float(row.get("drawdown_from_252_high", np.nan))

    # Crash regime with meaningful confidence
    if (
        regime_result.label == RegimeLabel.CRASH
        and regime_result.confidence >= 0.50
    ):
        reasons.append(
            f"crash regime detected (confidence {regime_result.confidence:.0%})"
        )

    # Price below 200 SMA AND severe drawdown → defensive exit
    if (
        not np.isnan(close)
        and not np.isnan(sma_200)
        and close < sma_200
        and not np.isnan(dd_252)
        and dd_252 < -0.25
    ):
        reasons.append(
            f"close below sma_200 AND drawdown {dd_252:.1%} from 1-yr high"
        )

    return "; ".join(reasons) if reasons else None


def _check_reduce_conditions(
    row: pd.Series,
    regime_result: Any,
) -> Optional[str]:
    """
    Return a REDUCE reason string if trimming conditions are met, else None.

    REDUCE is for moderate deterioration — used to trim existing longs
    before conditions become SELL territory.
    """
    reasons: List[str] = []

    close  = _safe_float(row.get("close",  np.nan))
    sma_50 = _safe_float(row.get("sma_50", np.nan))
    rsi    = _safe_float(row.get("rsi_14", np.nan))
    trend  = _safe_float(row.get("trend_score", np.nan))
    dd_20  = _safe_float(row.get("drawdown_from_20_high", np.nan))

    # Bear regime with confirmed confidence
    if (
        regime_result.label == RegimeLabel.BEAR
        and regime_result.confidence >= 0.50
    ):
        reasons.append(
            f"bear regime (confidence {regime_result.confidence:.0%})"
        )

    # Price closed below 50 SMA — classic momentum exit signal
    if not np.isnan(close) and not np.isnan(sma_50) and close < sma_50:
        reasons.append(f"close {close:.2f} below sma_50 {sma_50:.2f}")

    # Negative trend outside of a confirmed bull regime
    if (
        not np.isnan(trend)
        and trend < 0
        and regime_result.label != RegimeLabel.BULL
    ):
        reasons.append(f"trend_score {trend:.2f} negative")

    # Overbought in non-bull regime (extended, risk of reversal)
    if not np.isnan(rsi) and rsi > 75:
        reasons.append(f"RSI {rsi:.1f} overbought")

    # Short-term drawdown exceeds 8%
    if not np.isnan(dd_20) and dd_20 < -0.08:
        reasons.append(f"drawdown from 20-day high {dd_20:.1%}")

    return "; ".join(reasons) if reasons else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical features if the key columns are absent."""
    if "sma_200" not in df.columns:
        from data.feature_engineering import add_features
        df = add_features(df)
    return df


def _has_valid_price_volume(row: pd.Series) -> bool:
    """Return False if price or volume are clearly invalid."""
    close  = _safe_float(row.get("close",  np.nan))
    volume = _safe_float(row.get("volume", np.nan))
    return not (np.isnan(close) or close <= 0 or np.isnan(volume) or volume < 0)


def _safe_float(v: Any) -> float:
    """Convert to float; return NaN if not convertible."""
    try:
        result = float(v)
        return result
    except (TypeError, ValueError):
        return np.nan


def _build_metadata(
    row: pd.Series,
    regime_result: Any,
    allocation_target: float,
    size_multiplier: float,
) -> Dict[str, Any]:
    """Assemble the standard metadata dict for a Signal."""
    return {
        "close":                  _safe_float(row.get("close")),
        "volume":                 _safe_float(row.get("volume")),
        "volume_ratio":           _safe_float(row.get("volume_ratio")),
        "rsi_14":                 _safe_float(row.get("rsi_14")),
        "trend_score":            _safe_float(row.get("trend_score")),
        "sma_50":                 _safe_float(row.get("sma_50")),
        "sma_200":                _safe_float(row.get("sma_200")),
        "drawdown_from_252_high": _safe_float(row.get("drawdown_from_252_high")),
        "regime_confidence":      float(regime_result.confidence),
        "regime_is_uncertain":    bool(regime_result.is_uncertain),
        "allocation_target":      allocation_target,
        "position_size_multiplier": size_multiplier,
        "rejection_reasons":      [],
    }
