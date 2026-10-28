"""
Regime-based allocation policy.

AllocationPolicy translates a RegimeResult into concrete sizing guidance:
- How much of the portfolio should be invested?
- How large should new positions be relative to normal?
- Should new longs be allowed at all?
- Should existing positions be reduced?

These are MAXIMUMS and DEFAULTS — the Phase 5 risk manager applies
further per-trade and portfolio-level limits on top of these.

Financial assumption:
    The regime is an environment classifier.  More risky environments
    (crash, bear) warrant lower exposure.  Uncertain regimes warrant
    conservative sizing even when the nominal regime is bullish.
"""
from __future__ import annotations

from typing import Dict

from strategy.signals import RegimeLabel

# ---------------------------------------------------------------------------
# Target portfolio exposure ceiling by regime (fraction of equity in positions).
# These are CEILINGS — actual exposure can be lower after individual risk checks.
# ---------------------------------------------------------------------------

TARGET_EXPOSURE_BY_REGIME: Dict[RegimeLabel, float] = {
    RegimeLabel.CRASH:    0.00,   # No exposure — exit everything
    RegimeLabel.BEAR:     0.10,   # Minimal; only hold very strong existing longs
    RegimeLabel.NEUTRAL:  0.25,   # Moderate; high-quality setups only
    RegimeLabel.BULL:     0.50,   # Normal allocation; trend-following longs allowed
    RegimeLabel.EUPHORIA: 0.25,   # Reduced; avoid chasing overextended moves
    RegimeLabel.UNKNOWN:  0.00,   # Unknown = treat as unsafe
}


class AllocationPolicy:
    """
    Translate a regime classification into portfolio exposure and sizing guidance.

    All methods are pure functions of (regime, confidence, is_uncertain) with no
    side effects — safe to call in tests and backtesting without state concerns.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_target_exposure(
        self,
        regime: RegimeLabel,
        confidence: float,
        is_uncertain: bool = False,
    ) -> float:
        """
        Return the target fraction of portfolio equity to deploy.

        Reductions applied on top of the regime baseline:
        - uncertain:         × 0.50
        - confidence < 0.40: × 0.50
        - confidence < 0.60: × 0.75

        Returns a value in [0.0, MAX_PORTFOLIO_EXPOSURE].
        """
        base = TARGET_EXPOSURE_BY_REGIME.get(regime, 0.0)

        if is_uncertain:
            base *= 0.50

        if confidence < 0.40:
            base *= 0.50
        elif confidence < 0.60:
            base *= 0.75

        return base

    def get_position_size_multiplier(
        self,
        regime: RegimeLabel,
        confidence: float,
        is_uncertain: bool = False,
    ) -> float:
        """
        Return a multiplier [0.0, 1.0] to scale individual position sizes.

        A value of 1.0 means full size as calculated by the position sizer.
        A value of 0.0 means no position should be entered.

        Regime-specific caps:
        - crash / unknown: 0.0 (no new positions)
        - bear:            0.40 (very small positions if somehow allowed)
        - euphoria:        0.60 (smaller positions; risk of reversal is elevated)
        - neutral / bull:  1.0 base (further scaled by confidence/uncertainty)
        """
        # Never enter new positions in crash or unknown regime
        if regime in (RegimeLabel.CRASH, RegimeLabel.UNKNOWN):
            return 0.0

        # Confidence below threshold → no entry
        if confidence < 0.40:
            return 0.0

        multiplier = 1.0

        # Uncertain regime: cut position size in half
        if is_uncertain:
            multiplier *= 0.50

        # Confidence-based scaling
        if confidence < 0.60:
            multiplier *= 0.75

        # Regime-specific adjustments
        if regime == RegimeLabel.BEAR:
            multiplier *= 0.40
        elif regime == RegimeLabel.EUPHORIA:
            multiplier *= 0.60

        return multiplier

    def should_allow_new_longs(
        self,
        regime: RegimeLabel,
        confidence: float,
        is_uncertain: bool = False,
    ) -> bool:
        """
        Return True if new long positions are permitted under current conditions.

        Blocked unconditionally:
        - crash or unknown regime
        - bear regime (default: no new longs; protecting capital takes priority)
        - confidence < 0.40

        Blocked conditionally:
        - uncertain regime with confidence < 0.60 (regime flickering too much)
        """
        if regime in (RegimeLabel.CRASH, RegimeLabel.UNKNOWN):
            return False

        # Bear: we do not open new longs by default.  The spec requires "extremely
        # strong setup" to override — that is handled above this layer via hard
        # technical criteria, not by relaxing this gate.
        if regime == RegimeLabel.BEAR:
            return False

        if confidence < 0.40:
            return False

        if is_uncertain and confidence < 0.60:
            return False

        return True

    def should_reduce_existing_positions(
        self,
        regime: RegimeLabel,
        confidence: float,
        is_uncertain: bool = False,
    ) -> bool:
        """
        Return True if existing long positions should be reduced or exited.

        True when:
        - crash regime (reduce regardless of confidence)
        - bear regime with confidence >= 0.50 (meaningful bear signal)
        """
        if regime == RegimeLabel.CRASH:
            return True
        if regime == RegimeLabel.BEAR and confidence >= 0.50:
            return True
        return False
