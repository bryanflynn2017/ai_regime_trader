"""
Deterministic fallback regime classifier.

This is the always-available regime detector. It uses only observable price and
volume features — no model training, no external ML dependencies.

Design principles:
- Every classification is fully explainable: the reason string describes exactly
  which conditions fired.
- No lookahead: features at row T are backward-looking by construction
  (see data/feature_engineering.py). classify_series() processes each row using
  only data available up to that row.
- Graceful degradation: if key features are missing (insufficient history),
  return neutral with low confidence and a clear reason.
- The stability filter uses only past labels, never future ones.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from strategy.signals import RegimeLabel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RegimeResult — shared output type for both classifiers
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    """
    The output of any regime classifier.

    Carries enough information to make a fully auditable allocation decision:
    what was detected, how confident we are, why, and whether the regime is
    stable enough to act on.
    """
    label: RegimeLabel
    confidence: float          # 0.0 (total uncertainty) → 1.0 (very confident)
    timestamp: pd.Timestamp
    method: str                # "fallback" or "hmm"
    reason: str                # Human-readable explanation of the classification
    features_used: List[str]   # Which feature names influenced this result
    is_stable: bool            # True if same label persisted >= REGIME_STABILITY_BARS
    is_uncertain: bool         # True if labels have flickered too much recently
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))


# ---------------------------------------------------------------------------
# FallbackRegimeClassifier
# ---------------------------------------------------------------------------

# Minimum bars before we can classify (need sma_50 at minimum).
_MIN_BARS = 51

# Columns that must not be NaN for a full classification.
_CLASSIFICATION_FEATURES = [
    "close", "sma_50", "sma_200",
    "trend_score", "drawdown_from_252_high",
]


class FallbackRegimeClassifier:
    """
    Deterministic market regime classifier based on technical indicator thresholds.

    Regime ladder (highest-priority checked first):
        1. crash    — strong bearish evidence, elevated volatility, deep drawdown
        2. bear     — price below 200 SMA with negative trend
        3. euphoria — extended bull with overbought signals
        4. bull     — price above 200 SMA with positive trend
        5. neutral  — mixed signals (default)
    """

    def classify_latest(self, df: pd.DataFrame) -> RegimeResult:
        """
        Classify the market regime as of the last row of df.

        Calls add_features() internally if feature columns are absent.
        Never modifies the input DataFrame.

        Returns:
            RegimeResult with label, confidence, reason, stability flags.
        """
        df = self._ensure_features(df.copy())

        if len(df) < _MIN_BARS or df["sma_200"].isna().all():
            return self._insufficient_data_result(df)

        row = df.iloc[-1]
        if pd.isna(row.get("sma_200", np.nan)):
            return self._insufficient_data_result(df)

        label, confidence, reason, features_used = self._classify_row(row)

        # Apply stability filter using only past rows.
        past_labels = self._get_past_labels(df)
        is_stable, is_uncertain = self._check_stability(label, past_labels)

        if is_uncertain:
            confidence = max(confidence * 0.70, 0.10)
            reason = f"{reason} [uncertain — regime flickering]"

        return RegimeResult(
            label=label,
            confidence=confidence,
            timestamp=pd.Timestamp(df.index[-1]),
            method="fallback",
            reason=reason,
            features_used=features_used,
            is_stable=is_stable,
            is_uncertain=is_uncertain,
            metadata={"row_count": len(df)},
        )

    def classify_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify every row in df, returning a copy with regime columns appended.

        No-lookahead guarantee:
            add_features() uses only backward-looking rolling windows.
            _classify_row() uses only the current row's feature values.
            The stability filter uses only past labels (rolling backward window).

        Columns added:
            regime_label, regime_confidence, regime_is_stable,
            regime_is_uncertain, regime_reason, regime_method
        """
        df = self._ensure_features(df.copy())

        labels: List[RegimeLabel] = []
        confidences: List[float] = []
        reasons: List[str] = []

        for i in range(len(df)):
            row = df.iloc[i]
            if i < _MIN_BARS or pd.isna(row.get("sma_200", np.nan)):
                labels.append(RegimeLabel.NEUTRAL)
                confidences.append(0.10)
                reasons.append(f"Insufficient data at row {i} (need {_MIN_BARS})")
            else:
                lbl, conf, rsn, _ = self._classify_row(row)
                labels.append(lbl)
                confidences.append(conf)
                reasons.append(rsn)

        label_series = pd.Series(
            [l.value for l in labels], index=df.index, dtype=str
        )

        # Stability / uncertainty applied across the label series (backward-looking).
        from config.settings import settings
        is_stable = _compute_is_stable(label_series, n=settings.REGIME_STABILITY_BARS)
        is_uncertain = _compute_is_uncertain(label_series, n=20, max_unique=3)

        # Dampen confidence when uncertain.
        conf_series = pd.Series(confidences, index=df.index, dtype=float)
        conf_series = conf_series.where(~is_uncertain, conf_series * 0.70)

        result = df.copy()
        result["regime_label"]        = label_series
        result["regime_confidence"]   = conf_series
        result["regime_is_stable"]    = is_stable
        result["regime_is_uncertain"] = is_uncertain
        result["regime_reason"]       = reasons
        result["regime_method"]       = "fallback"

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_row(
        self, row: pd.Series
    ) -> Tuple[RegimeLabel, float, str, List[str]]:
        """
        Core classification logic for a single row.

        Returns (label, confidence, reason, features_used).
        Confidence is computed as (signals_fired / max_possible_signals).
        """
        def _get(name: str) -> float:
            v = row.get(name, np.nan)
            return float(v) if not pd.isna(v) else np.nan

        close       = _get("close")
        sma_50      = _get("sma_50")
        sma_200     = _get("sma_200")
        trend       = _get("trend_score")
        dd_252      = _get("drawdown_from_252_high")
        rsi         = _get("rsi_14")
        vol_pct     = _get("volatility_percentile")
        vol_20      = _get("rolling_volatility_20")

        if np.isnan(sma_200) or np.isnan(close):
            return (
                RegimeLabel.NEUTRAL, 0.10,
                "Cannot classify: sma_200 or close is NaN",
                [],
            )

        # ----------------------------------------------------------------
        # CRASH — checked first (most severe)
        # Requires at least 3 of 4 conditions.
        # ----------------------------------------------------------------
        crash_hits: List[str] = []

        if not np.isnan(sma_200) and close < sma_200:
            crash_hits.append(f"close({close:.2f}) < sma_200({sma_200:.2f})")

        if not np.isnan(dd_252) and dd_252 <= -0.20:
            crash_hits.append(f"drawdown {dd_252:.1%} from 1-yr high (>= 20%)")

        if not np.isnan(vol_pct) and vol_pct >= 0.85:
            crash_hits.append(f"vol at {vol_pct:.0%} percentile (elevated)")

        if not np.isnan(trend) and trend < -0.50:
            crash_hits.append(f"trend_score={trend:.2f} (strongly bearish)")

        if len(crash_hits) >= 3:
            conf = min(len(crash_hits) / 4.0, 0.95)
            return (
                RegimeLabel.CRASH,
                conf,
                "Crash: " + "; ".join(crash_hits),
                ["close", "sma_200", "drawdown_from_252_high", "volatility_percentile", "trend_score"],
            )

        # ----------------------------------------------------------------
        # BEAR — price below 200 SMA with negative trend alignment
        # ----------------------------------------------------------------
        bear_hits: List[str] = []

        if not np.isnan(sma_200) and close < sma_200:
            bear_hits.append(f"close < sma_200")

        if not np.isnan(sma_50) and not np.isnan(sma_200) and sma_50 < sma_200:
            bear_hits.append("sma_50 < sma_200 (death cross)")

        if not np.isnan(trend) and trend < -0.20:
            bear_hits.append(f"trend_score={trend:.2f}")

        if len(bear_hits) >= 2:
            conf = min(len(bear_hits) / 3.0, 0.85)
            return (
                RegimeLabel.BEAR,
                conf,
                "Bear: " + "; ".join(bear_hits),
                ["close", "sma_50", "sma_200", "trend_score"],
            )

        # ----------------------------------------------------------------
        # EUPHORIA — price extended well above moving averages, RSI elevated
        # Requires bull conditions plus two overbought signals.
        # ----------------------------------------------------------------
        bull_hits: List[str] = []
        if not np.isnan(sma_200) and close > sma_200:
            bull_hits.append("close > sma_200")
        if not np.isnan(sma_50) and not np.isnan(sma_200) and sma_50 > sma_200:
            bull_hits.append("sma_50 > sma_200 (golden cross)")
        if not np.isnan(trend) and trend > 0.30:
            bull_hits.append(f"trend_score={trend:.2f}")

        euphoria_hits: List[str] = []
        if len(bull_hits) >= 2:  # Needs bull base
            if not np.isnan(rsi) and rsi > 70.0:
                euphoria_hits.append(f"RSI={rsi:.0f} (overbought)")
            if not np.isnan(sma_50) and close > sma_50 * 1.10:
                euphoria_hits.append(f"close {(close/sma_50 - 1):.1%} above sma_50")
            if not np.isnan(trend) and trend > 0.60:
                euphoria_hits.append(f"trend_score={trend:.2f} (very strong)")

        if len(bull_hits) >= 2 and len(euphoria_hits) >= 2:
            conf = min(0.65 + len(euphoria_hits) * 0.08, 0.88)
            return (
                RegimeLabel.EUPHORIA,
                conf,
                "Euphoria (overextended): " + "; ".join(bull_hits + euphoria_hits),
                ["close", "sma_50", "sma_200", "rsi_14", "trend_score"],
            )

        # ----------------------------------------------------------------
        # BULL — price above 200 SMA with positive trend
        # ----------------------------------------------------------------
        if len(bull_hits) >= 2:
            conf = min(len(bull_hits) / 3.0 + 0.20, 0.90)
            return (
                RegimeLabel.BULL,
                conf,
                "Bull: " + "; ".join(bull_hits),
                ["close", "sma_50", "sma_200", "trend_score"],
            )

        # ----------------------------------------------------------------
        # NEUTRAL — default when signals are mixed
        # ----------------------------------------------------------------
        neutral_reason_parts: List[str] = []
        if not np.isnan(trend):
            neutral_reason_parts.append(f"trend_score={trend:.2f}")
        if not np.isnan(sma_200):
            rel = (close / sma_200 - 1)
            neutral_reason_parts.append(f"close {rel:+.1%} vs sma_200")

        # Confidence is highest when trend is truly near zero (genuinely neutral).
        trend_val = trend if not np.isnan(trend) else 0.0
        neutral_conf = 0.35 + 0.25 * max(0.0, 1.0 - abs(trend_val) * 2.0)

        return (
            RegimeLabel.NEUTRAL,
            neutral_conf,
            "Neutral (mixed signals): " + ("; ".join(neutral_reason_parts) or "no strong directional bias"),
            ["close", "sma_200", "trend_score"],
        )

    def _get_past_labels(self, df: pd.DataFrame) -> List[str]:
        """
        Return a list of the most recent regime labels by re-classifying a
        small trailing window. Used only for stability assessment.
        """
        from config.settings import settings
        n = max(settings.REGIME_STABILITY_BARS, 20)
        start = max(0, len(df) - n)
        past = []
        for i in range(start, len(df)):
            row = df.iloc[i]
            if i < _MIN_BARS or pd.isna(row.get("sma_200", np.nan)):
                past.append(RegimeLabel.NEUTRAL.value)
            else:
                lbl, _, _, _ = self._classify_row(row)
                past.append(lbl.value)
        return past

    def _check_stability(
        self, current_label: RegimeLabel, past_labels: List[str]
    ) -> Tuple[bool, bool]:
        """
        Determine is_stable and is_uncertain from a list of recent labels.

        is_stable:   last REGIME_STABILITY_BARS labels are all current_label
        is_uncertain: more than 3 distinct labels in the recent window
        """
        from config.settings import settings
        n_stable = settings.REGIME_STABILITY_BARS

        if len(past_labels) < n_stable:
            return False, False

        recent_stable_window = past_labels[-n_stable:]
        is_stable = all(l == current_label.value for l in recent_stable_window)

        n_uncertain_window = min(20, len(past_labels))
        uncertain_window = past_labels[-n_uncertain_window:]
        is_uncertain = len(set(uncertain_window)) > 3

        return is_stable, is_uncertain

    @staticmethod
    def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical features if the key columns are absent."""
        if "sma_200" not in df.columns:
            from data.feature_engineering import add_features
            df = add_features(df)
        return df

    @staticmethod
    def _insufficient_data_result(df: pd.DataFrame) -> RegimeResult:
        n = len(df)
        return RegimeResult(
            label=RegimeLabel.NEUTRAL,
            confidence=0.10,
            timestamp=pd.Timestamp(df.index[-1]) if n > 0 else pd.Timestamp.now(),
            method="fallback",
            reason=f"Insufficient data: {n} rows available (need >= {_MIN_BARS} with sma_200)",
            features_used=[],
            is_stable=False,
            is_uncertain=True,
            metadata={"row_count": n, "min_required": _MIN_BARS},
        )


# ---------------------------------------------------------------------------
# Vectorised stability helpers (used by classify_series)
# ---------------------------------------------------------------------------

def _compute_is_stable(labels: pd.Series, n: int) -> pd.Series:
    """
    Return True at position t if the last n labels are all identical.

    Rolling min == rolling max on integer codes is the fast vectorised check.
    pd.Categorical.codes returns a numpy ndarray — wrap in Series to use .rolling().
    """
    codes = pd.Series(
        pd.Categorical(labels.astype(str)).codes.astype(float),
        index=labels.index,
    )
    r_min = codes.rolling(n, min_periods=n).min()
    r_max = codes.rolling(n, min_periods=n).max()
    return (r_min == r_max).fillna(False)


def _compute_is_uncertain(
    labels: pd.Series, n: int = 20, max_unique: int = 3
) -> pd.Series:
    """
    Return True at position t if there are more than max_unique distinct
    labels in the most recent n bars.
    """
    codes = pd.Series(
        pd.Categorical(labels.astype(str)).codes.astype(float),
        index=labels.index,
    )
    unique_count = codes.rolling(n, min_periods=n).apply(
        lambda x: float(len(set(x))), raw=True
    )
    return (unique_count > max_unique).fillna(False)
