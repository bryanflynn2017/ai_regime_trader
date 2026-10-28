"""
Risk manager — absolute veto authority over every trading signal.

The risk manager is the last gate before any order reaches the broker.
A signal that passes the strategy orchestrator still has no guarantee of
execution; it must pass every check here first.

Conservative design principle:
  When in doubt, BLOCK.  An order that was never sent cannot cause a loss.
  Any unexpected exception fails CLOSED (returns a blocking RiskDecision).

Checks performed for BUY signals (in order):
  1. PAPER_TRADING flag is True
  2. LIVE_TRADING_ALLOWED flag is False (this phase: paper only)
  3. Circuit breakers (drawdown, lock file)
  4. Stop-loss present and valid
  5. Daily trade count < MAX_TRADES_PER_DAY
  6. Open positions < MAX_OPEN_POSITIONS
  7. Portfolio exposure headroom available
  8. Ticker exposure headroom available
  9. Position sizer approves the size

Phase 5 does NOT connect to Alpaca or place real orders.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from loguru import logger  # type: ignore
except ImportError:
    logger = logging.getLogger(__name__)  # type: ignore

from risk.circuit_breakers import CircuitBreakerManager, CircuitBreakerStatus
from risk.position_sizing import PositionSizer, PositionSizingResult
from strategy.signals import Signal, SignalAction


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RiskDecision:
    """
    The complete risk evaluation result for one Signal.

    Read final_action to know what the bot should actually do.
    Read rejection_reasons to understand why the original action was blocked.
    Read position_size for the approved share count (BUY only).
    """
    approved:                bool
    original_action:         SignalAction
    final_action:            SignalAction
    reason:                  str
    position_size:           Optional[PositionSizingResult]
    circuit_breaker_status:  CircuitBreakerStatus
    rejection_reasons:       List[str] = field(default_factory=list)
    warnings:                List[str] = field(default_factory=list)
    metadata:                Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Evaluates a Signal against all risk rules and returns a RiskDecision.

    Inject dependencies for testing:
        rm = RiskManager(
            circuit_breaker_manager=CircuitBreakerManager(settings_override=s),
            settings_override=s,
        )
    """

    def __init__(
        self,
        circuit_breaker_manager: Optional[CircuitBreakerManager] = None,
        settings_override: Any = None,
    ) -> None:
        if settings_override is None:
            from config.settings import settings
            self._s = settings
        else:
            self._s = settings_override

        self._cb     = circuit_breaker_manager or CircuitBreakerManager(
            settings_override=self._s
        )
        self._sizer  = PositionSizer(settings_override=self._s)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_signal(
        self,
        signal:                      Signal,
        equity:                      float,
        starting_day_equity:         float,
        starting_week_equity:        float,
        peak_equity:                 float,
        current_portfolio_exposure:  float = 0.0,
        current_ticker_exposure:     float = 0.0,
        open_positions_count:        int   = 0,
        trades_today:                int   = 0,
        has_existing_position:       bool  = False,
    ) -> RiskDecision:
        """
        Evaluate a Signal against all risk rules.

        Never raises.  Any unexpected exception returns a blocking RiskDecision
        so the bot cannot accidentally execute a trade during an error.

        Args:
            signal:                     The candidate signal from the strategy layer.
            equity:                     Current account equity in dollars.
            starting_day_equity:        Equity at today's market open.
            starting_week_equity:       Equity at this week's Monday open.
            peak_equity:                Highest equity ever recorded (for total drawdown).
            current_portfolio_exposure: Total market value of all open positions.
            current_ticker_exposure:    Market value of any existing position in signal.ticker.
            open_positions_count:       Number of currently open positions.
            trades_today:               Number of trades already executed today.
            has_existing_position:      True if there is an open position in signal.ticker.
        """
        rejections: List[str] = []
        warnings:   List[str] = []

        try:
            return self._evaluate_inner(
                signal, equity,
                starting_day_equity, starting_week_equity, peak_equity,
                current_portfolio_exposure, current_ticker_exposure,
                open_positions_count, trades_today, has_existing_position,
                rejections, warnings,
            )
        except Exception as exc:
            msg = f"Unexpected exception in RiskManager for {signal.ticker}: {exc}"
            logger.exception(msg)
            rejections.append(f"Exception: {exc}")
            return self._blocked(
                signal=signal,
                reason=f"Internal risk manager error — failing closed. {exc}",
                rejections=rejections,
                warnings=warnings,
                cb_status=_blocking_cb_status("Error in risk manager"),
            )

    # ------------------------------------------------------------------
    # Private: main evaluation path
    # ------------------------------------------------------------------

    def _evaluate_inner(
        self,
        signal, equity,
        starting_day_equity, starting_week_equity, peak_equity,
        current_portfolio_exposure, current_ticker_exposure,
        open_positions_count, trades_today, has_existing_position,
        rejections, warnings,
    ) -> RiskDecision:

        # ---- Safety switch #1: paper trading must be active ----
        if not self._s.PAPER_TRADING:
            rejections.append(
                "PAPER_TRADING is False. This system requires paper trading mode."
            )
            logger.error("PAPER_TRADING=False. Blocking all signals.")
            return self._blocked(
                signal, "PAPER_TRADING is not True",
                rejections, warnings,
                _blocking_cb_status("PAPER_TRADING=False"),
            )

        # ---- Safety switch #2: live trading is not supported in Phase 5 ----
        if self._s.LIVE_TRADING_ALLOWED:
            msg = (
                "LIVE_TRADING_ALLOWED is True. Phase 5 only supports paper trading. "
                "No broker connection exists yet. Blocking all BUY signals."
            )
            warnings.append(msg)
            rejections.append(msg)
            logger.warning(msg)
            return self._blocked(
                signal, "LIVE_TRADING_ALLOWED=True but no live broker is configured",
                rejections, warnings,
                _blocking_cb_status("LIVE_TRADING_ALLOWED=True"),
            )

        # ---- Circuit breakers ----
        cb_status = self._cb.evaluate(
            current_equity=equity,
            starting_day_equity=starting_day_equity,
            starting_week_equity=starting_week_equity,
            peak_equity=peak_equity,
        )

        if not cb_status.trading_allowed:
            rejections.append(
                f"Circuit breaker halted all trading: {cb_status.reason}"
            )
            return self._blocked(
                signal, cb_status.reason, rejections, warnings, cb_status
            )

        # ---- Route by action ----
        action = signal.action

        if action == SignalAction.HOLD:
            return self._approve_hold(signal, cb_status, rejections, warnings)

        if action in (SignalAction.REDUCE, SignalAction.SELL):
            return self._handle_exit(
                signal, cb_status, has_existing_position, rejections, warnings
            )

        if action == SignalAction.BUY:
            return self._handle_buy(
                signal, equity, cb_status,
                current_portfolio_exposure, current_ticker_exposure,
                open_positions_count, trades_today,
                rejections, warnings,
            )

        rejections.append(f"Unknown signal action: {action!r}")
        return self._blocked(
            signal, f"Unknown action: {action}",
            rejections, warnings, cb_status
        )

    # ------------------------------------------------------------------
    # Private: action handlers
    # ------------------------------------------------------------------

    def _approve_hold(
        self, signal, cb_status, rejections, warnings
    ) -> RiskDecision:
        return RiskDecision(
            approved=True,
            original_action=SignalAction.HOLD,
            final_action=SignalAction.HOLD,
            reason="HOLD approved — no action required",
            position_size=None,
            circuit_breaker_status=cb_status,
            rejection_reasons=rejections,
            warnings=warnings,
            metadata={"ticker": signal.ticker},
        )

    def _handle_exit(
        self, signal, cb_status, has_existing_position, rejections, warnings
    ) -> RiskDecision:
        if not has_existing_position:
            # No position to exit — convert to HOLD with a warning
            msg = (
                f"{signal.action.value} signal for {signal.ticker} "
                "but no existing position. Converting to HOLD."
            )
            warnings.append(msg)
            logger.debug(msg)
            return RiskDecision(
                approved=True,
                original_action=signal.action,
                final_action=SignalAction.HOLD,
                reason=f"{signal.action.value} with no existing position -> HOLD",
                position_size=None,
                circuit_breaker_status=cb_status,
                rejection_reasons=rejections,
                warnings=warnings,
                metadata={"ticker": signal.ticker},
            )

        logger.debug(
            "{} approved for {} (existing position confirmed).",
            signal.action.value, signal.ticker,
        )
        return RiskDecision(
            approved=True,
            original_action=signal.action,
            final_action=signal.action,
            reason=f"{signal.action.value} approved — existing position confirmed",
            position_size=None,
            circuit_breaker_status=cb_status,
            rejection_reasons=rejections,
            warnings=warnings,
            metadata={"ticker": signal.ticker},
        )

    def _handle_buy(
        self,
        signal, equity, cb_status,
        current_portfolio_exposure, current_ticker_exposure,
        open_positions_count, trades_today,
        rejections, warnings,
    ) -> RiskDecision:

        def block(reason: str) -> RiskDecision:
            return self._blocked(signal, reason, rejections, warnings, cb_status)

        # Circuit breaker: no new positions
        if not cb_status.new_positions_allowed:
            rejections.append(
                f"Circuit breaker blocks new positions: {cb_status.reason}"
            )
            return block(cb_status.reason)

        # Stop loss presence check
        if self._s.REQUIRE_STOP_LOSS:
            if signal.stop_loss is None or signal.stop_loss <= 0:
                rejections.append(
                    f"Stop loss missing or zero (stop_loss={signal.stop_loss}). "
                    "REQUIRE_STOP_LOSS=True."
                )
                return block("Missing valid stop loss")
            if signal.stop_loss >= signal.entry_price:
                rejections.append(
                    f"stop_loss {signal.stop_loss:.2f} >= entry_price "
                    f"{signal.entry_price:.2f} — invalid for long trade"
                )
                return block("Invalid stop loss (>= entry)")

        # Daily trade limit
        if trades_today >= self._s.MAX_TRADES_PER_DAY:
            rejections.append(
                f"Max trades per day reached: "
                f"{trades_today}/{self._s.MAX_TRADES_PER_DAY}"
            )
            return block("Max trades per day reached")

        # Open position limit
        if open_positions_count >= self._s.MAX_OPEN_POSITIONS:
            rejections.append(
                f"Max open positions reached: "
                f"{open_positions_count}/{self._s.MAX_OPEN_POSITIONS}"
            )
            return block("Max open positions reached")

        # Portfolio exposure headroom
        max_portfolio = self._s.MAX_PORTFOLIO_EXPOSURE * equity
        if current_portfolio_exposure >= max_portfolio:
            rejections.append(
                f"Portfolio exposure maxed: "
                f"${current_portfolio_exposure:,.0f} / ${max_portfolio:,.0f} "
                f"({self._s.MAX_PORTFOLIO_EXPOSURE:.0%})"
            )
            return block("Portfolio exposure at maximum")

        # Per-ticker exposure headroom
        max_ticker = self._s.MAX_SINGLE_POSITION_EXPOSURE * equity
        if current_ticker_exposure >= max_ticker:
            rejections.append(
                f"Ticker {signal.ticker} exposure maxed: "
                f"${current_ticker_exposure:,.0f} / ${max_ticker:,.0f}"
            )
            return block(f"Ticker {signal.ticker} exposure at maximum")

        # Position sizing (applies circuit breaker size multiplier)
        ps = self._sizer.calculate_position_size(
            equity=equity,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            current_portfolio_exposure=current_portfolio_exposure,
            current_ticker_exposure=current_ticker_exposure,
            position_size_multiplier=cb_status.size_multiplier,
        )

        if not ps.approved:
            rejections.append(f"Position sizer rejected: {ps.reason}")
            return block(f"Position sizing failed: {ps.reason}")

        logger.info(
            "BUY APPROVED {}: {:.4f} shares @ ${:.2f}. "
            "Risk=${:.0f} Max-loss=${:.0f}. CB_mult={:.2f}. {}",
            signal.ticker, ps.shares, signal.entry_price,
            ps.risk_dollars, ps.max_loss_dollars,
            cb_status.size_multiplier, ps.reason,
        )

        return RiskDecision(
            approved=True,
            original_action=SignalAction.BUY,
            final_action=SignalAction.BUY,
            reason=f"BUY approved. {ps.reason}",
            position_size=ps,
            circuit_breaker_status=cb_status,
            rejection_reasons=rejections,
            warnings=warnings,
            metadata={
                "ticker":          signal.ticker,
                "equity":          equity,
                "size_multiplier": cb_status.size_multiplier,
            },
        )

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    def _blocked(
        self,
        signal:     Signal,
        reason:     str,
        rejections: List[str],
        warnings:   List[str],
        cb_status:  CircuitBreakerStatus,
    ) -> RiskDecision:
        logger.debug(
            "Signal BLOCKED {} (original={}): {}",
            signal.ticker, signal.action.value, reason,
        )
        return RiskDecision(
            approved=False,
            original_action=signal.action,
            final_action=SignalAction.HOLD,
            reason=f"Blocked: {reason}",
            position_size=None,
            circuit_breaker_status=cb_status,
            rejection_reasons=rejections,
            warnings=warnings,
            metadata={"ticker": signal.ticker},
        )


# ---------------------------------------------------------------------------
# Helper: create a blocking CB status without a real evaluation
# ---------------------------------------------------------------------------

def _blocking_cb_status(reason: str) -> CircuitBreakerStatus:
    """Return a maximally-restrictive CircuitBreakerStatus for error / guard cases."""
    return CircuitBreakerStatus(
        trading_allowed=False,
        new_positions_allowed=False,
        reduce_only=True,
        must_close_all=False,
        lock_created=False,
        size_multiplier=0.0,
        reason=reason,
        triggered_rules=["GUARD"],
    )
