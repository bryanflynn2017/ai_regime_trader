"""
Position sizer — translates a Signal's stop-loss distance into a concrete share count.

Financial model (fixed-fractional risk):
    risk_dollars  = equity × MAX_RISK_PER_TRADE × position_size_multiplier
    stop_distance = entry_price − stop_loss
    raw_shares    = risk_dollars / stop_distance

Three independent dollar caps are applied, whichever is most restrictive:
  1. MAX_SINGLE_POSITION_EXPOSURE × equity          (per-position ceiling)
  2. (MAX_PORTFOLIO_EXPOSURE × equity) − current_portfolio  (remaining headroom)
  3. (MAX_SINGLE_POSITION_EXPOSURE × equity) − current_ticker  (ticker ceiling)

This layer is a pure calculation with no side effects.  It never connects to
any broker and never modifies external state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PositionSizingResult:
    """
    Output of a single position sizing calculation.

    If approved is False, shares == 0 and position_value == 0.
    Every rejection includes a human-readable reason.
    """
    approved: bool
    shares: float
    position_value: float
    risk_dollars: float
    max_loss_dollars: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PositionSizer:
    """
    Calculates position size using the fixed-fractional risk model.

    Inject a custom Settings for testing:
        sizer = PositionSizer(settings_override=my_settings)
    """

    def __init__(self, settings_override: Any = None) -> None:
        if settings_override is None:
            from config.settings import settings
            self._s = settings
        else:
            self._s = settings_override

    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float,
        current_portfolio_exposure: float = 0.0,
        current_ticker_exposure: float = 0.0,
        position_size_multiplier: float = 1.0,
    ) -> PositionSizingResult:
        """
        Calculate the approved share count for a prospective long trade.

        Args:
            equity:                     Current account equity in dollars.
            entry_price:                Intended purchase price per share.
            stop_loss:                  Stop-loss price (must be < entry for longs).
            current_portfolio_exposure: Total market value of all open positions.
            current_ticker_exposure:    Market value already allocated to this ticker.
            position_size_multiplier:   Scaling factor [0, 1] from circuit breaker / regime.

        Returns:
            PositionSizingResult with approved=True/False and full audit trail.
        """
        # ---- Input validation ----
        if equity <= 0:
            return self._reject(
                f"equity {equity:.2f} must be positive",
                equity=equity,
            )
        if entry_price <= 0:
            return self._reject(
                f"entry_price {entry_price:.2f} must be positive",
                entry_price=entry_price,
            )
        if stop_loss <= 0:
            return self._reject(
                f"stop_loss {stop_loss:.2f} must be positive",
                stop_loss=stop_loss,
            )
        if stop_loss >= entry_price:
            return self._reject(
                f"stop_loss {stop_loss:.2f} >= entry_price {entry_price:.2f} "
                "(invalid for a long trade — stop must be below entry)",
                entry_price=entry_price,
                stop_loss=stop_loss,
            )

        # ---- Core calculation ----
        stop_distance = entry_price - stop_loss
        multiplier    = max(0.0, position_size_multiplier)
        risk_dollars  = equity * self._s.MAX_RISK_PER_TRADE * multiplier

        raw_shares         = risk_dollars / stop_distance
        raw_position_value = raw_shares * entry_price

        # ---- Dollar caps ----
        # Cap 1: single position ceiling
        max_by_single = self._s.MAX_SINGLE_POSITION_EXPOSURE * equity

        # Cap 2: remaining portfolio headroom
        max_portfolio_total = self._s.MAX_PORTFOLIO_EXPOSURE * equity
        remaining_portfolio = max(0.0, max_portfolio_total - current_portfolio_exposure)

        # Cap 3: remaining ticker headroom
        max_ticker_total   = self._s.MAX_SINGLE_POSITION_EXPOSURE * equity
        remaining_ticker   = max(0.0, max_ticker_total - current_ticker_exposure)

        capped_value = min(
            raw_position_value,
            max_by_single,
            remaining_portfolio,
            remaining_ticker,
        )

        final_shares = capped_value / entry_price

        # ---- Fractional share handling ----
        fractional_floored = False
        if not self._s.ALLOW_FRACTIONAL_SHARES:
            final_shares     = math.floor(final_shares)
            fractional_floored = True

        if final_shares <= 0:
            return self._reject(
                "Shares calculated as <= 0 after applying exposure caps.",
                raw_shares=raw_shares,
                raw_position_value=raw_position_value,
                remaining_portfolio=remaining_portfolio,
                remaining_ticker=remaining_ticker,
            )

        final_position_value = final_shares * entry_price
        max_loss_dollars     = final_shares * stop_distance

        # ---- Build audit trail ----
        caps_applied: List[str] = []
        if raw_position_value > max_by_single + 1e-9:
            caps_applied.append(
                f"MAX_SINGLE_POSITION ({self._s.MAX_SINGLE_POSITION_EXPOSURE:.0%} "
                f"of equity = ${max_by_single:,.0f})"
            )
        if raw_position_value > remaining_portfolio + 1e-9:
            caps_applied.append(
                f"portfolio headroom (${remaining_portfolio:,.0f} remaining)"
            )
        if raw_position_value > remaining_ticker + 1e-9:
            caps_applied.append(
                f"ticker headroom (${remaining_ticker:,.0f} remaining)"
            )
        if fractional_floored:
            caps_applied.append("fractional shares disabled (floored to whole shares)")

        reason = (
            f"Approved: {final_shares:.4f} shares @ ${entry_price:.2f} "
            f"(value=${final_position_value:,.0f}, "
            f"max_loss=${max_loss_dollars:,.0f})"
        )
        if caps_applied:
            reason += f". Caps: {'; '.join(caps_applied)}"

        return PositionSizingResult(
            approved=True,
            shares=final_shares,
            position_value=final_position_value,
            risk_dollars=risk_dollars,
            max_loss_dollars=max_loss_dollars,
            reason=reason,
            metadata={
                "raw_shares":            raw_shares,
                "raw_position_value":    raw_position_value,
                "stop_distance":         stop_distance,
                "max_by_single":         max_by_single,
                "remaining_portfolio":   remaining_portfolio,
                "remaining_ticker":      remaining_ticker,
                "position_size_multiplier": multiplier,
                "caps_applied":          caps_applied,
            },
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _reject(self, reason: str, **context) -> PositionSizingResult:
        return PositionSizingResult(
            approved=False,
            shares=0.0,
            position_value=0.0,
            risk_dollars=0.0,
            max_loss_dollars=0.0,
            reason=f"Rejected: {reason}",
            metadata=context,
        )
