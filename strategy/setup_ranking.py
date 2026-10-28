"""
Setup ranking — prioritise detected day-trade setups (Phase 13).

This module is a RANKER ONLY. Given a list of DayTradeSetup candidates (produced
by DayTradeSetupDetector), it scores each one and returns a new list ordered from
best to worst. It does NOT place orders, size positions, evaluate portfolio risk,
or talk to any broker. The rank it assigns is advisory metadata for human review
or for a future execution strategy — never an order.

Scoring (higher is better):
    score  = confidence    * 2.00
           + risk_reward_1 * 0.50
           + risk_reward_2 * 0.25
           + 0.50  if unusual_volume == 1.0
           + 0.50  if setup_type == OPENING_RANGE_BREAKOUT_LONG
           + 0.25  if setup_type == VWAP_RECLAIM_LONG
           - 1.00  if is_valid is False

Safety properties:
  - The input list and the original DayTradeSetup objects are never mutated; the
    ranker works on deep copies and returns those.
  - `metadata["rank"]` (1 = best) and `metadata["score"]` are written only on the
    returned copies.
"""
from __future__ import annotations

import copy
from typing import List

from strategy.day_trade_setups import DayTradeSetup, SetupType


class SetupRanker:
    """
    Rank DayTradeSetup candidates so the best trades can be prioritised.

    Usage:
        ranker = SetupRanker()
        ranked = ranker.rank_setups(setups)        # all, best first
        best   = ranker.filter_top_setups(setups)  # valid + high-confidence only
    """

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_setup(self, setup: DayTradeSetup) -> float:
        """
        Compute the ranking score for a single setup (higher is better).

        Reads `unusual_volume` from the setup's metadata (the detector stores it
        there). Missing/unconvertible values contribute nothing rather than
        raising.
        """
        score = 0.0
        score += _safe_float(setup.confidence) * 2.0
        score += _safe_float(setup.risk_reward_1) * 0.5
        score += _safe_float(setup.risk_reward_2) * 0.25

        unusual = _safe_float((setup.metadata or {}).get("unusual_volume"))
        if unusual == 1.0:
            score += 0.5

        if setup.setup_type == SetupType.OPENING_RANGE_BREAKOUT_LONG:
            score += 0.5
        elif setup.setup_type == SetupType.VWAP_RECLAIM_LONG:
            score += 0.25

        if not setup.is_valid:
            score -= 1.0

        return score

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_setups(self, setups: List[DayTradeSetup]) -> List[DayTradeSetup]:
        """
        Return a new list of setups ordered best-to-worst by score.

        The original list and its objects are not mutated — deep copies are
        ranked and returned. Each returned copy gets `metadata["rank"]` (1 = best)
        and `metadata["score"]`.
        """
        # Deep-copy so neither the input list nor the original setups change.
        copies = [copy.deepcopy(s) for s in setups]

        # Score once, then sort descending. Stable sort preserves input order
        # for ties so the result is deterministic.
        scored = sorted(copies, key=self.score_setup, reverse=True)

        for position, setup in enumerate(scored, start=1):
            if setup.metadata is None:
                setup.metadata = {}
            setup.metadata["rank"] = position
            setup.metadata["score"] = self.score_setup(setup)

        return scored

    def filter_top_setups(
        self,
        setups: List[DayTradeSetup],
        min_confidence: float = 0.6,
        max_results: int = 5,
    ) -> List[DayTradeSetup]:
        """
        Return at most `max_results` of the best setups, keeping only valid ones
        whose confidence is >= `min_confidence`.

        The result is ranked (best first) and carries the same rank/score
        metadata as rank_setups().
        """
        ranked = self.rank_setups(setups)
        strong = [
            s for s in ranked
            if s.is_valid and _safe_float(s.confidence) >= min_confidence
        ]
        return strong[:max_results]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value) -> float:
    """Convert *value* to float; return 0.0 for None/NaN/unconvertible input."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if result != result:  # NaN
        return 0.0
    return result
