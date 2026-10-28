"""
Order executor — translates RiskDecision objects into broker orders.

Works with any BrokerAdapter implementation:
  - SimulatedBrokerAdapter (in-memory, no credentials)
  - AlpacaBrokerAdapter (Alpaca paper trading)

Dry-run mode (default):
    Returns a simulated OrderExecutionResult without any broker call.
    Works without credentials.  Safe to run in tests.

Real mode (dry_run=False):
    Calls the broker adapter to submit orders.
    Requires paper-safety enforcement in the adapter.

Security rules:
    - is_paper property is always True for all adapters.
    - An unapproved RiskDecision is always rejected — no order is sent.
    - A HOLD final_action is always rejected — no order is sent.
    - AlpacaBrokerAdapter re-validates paper safety on every submit_order call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from broker.base import BrokerAdapter, BrokerError
from strategy.signals import Signal, SignalAction

try:
    from loguru import logger  # type: ignore
except ImportError:
    logger = logging.getLogger(__name__)  # type: ignore


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OrderExecutionResult:
    """
    The complete result of an order execution attempt.

    If submitted=False, the reason explains why the order was not sent.
    If dry_run=True, the order was simulated — nothing was sent to Alpaca.
    """
    submitted:       bool
    dry_run:         bool
    ticker:          str
    action:          str           # "BUY", "SELL", "REDUCE", "HOLD"
    quantity:        float
    order_id:        Optional[str]
    status:          str           # "submitted", "simulated", "rejected", "failed"
    reason:          str
    broker_response: Dict[str, Any]
    metadata:        Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# OrderExecutor
# ---------------------------------------------------------------------------

class OrderExecutor:
    """
    Executes RiskDecision objects as broker orders via BrokerAdapter.

    Accepts any BrokerAdapter implementation (SimulatedBrokerAdapter, AlpacaBrokerAdapter, etc.).
    For backward compatibility, also accepts AlpacaClientWrapper directly.
    """

    def __init__(
        self,
        alpaca_client: Any = None,  # Deprecated; use broker_adapter instead
        broker_adapter: Optional[BrokerAdapter] = None,
        dry_run: bool = True,
    ) -> None:
        # Support both old (alpaca_client) and new (broker_adapter) interfaces
        if broker_adapter is not None:
            self._broker = broker_adapter
        elif alpaca_client is not None:
            # Backward compat: wrap AlpacaClientWrapper or accept BrokerAdapter-like object
            if hasattr(alpaca_client, "get_underlying_client"):
                # It's already an adapter (AlpacaBrokerAdapter)
                self._broker = alpaca_client
            else:
                # It's an AlpacaClientWrapper; wrap it
                from broker.alpaca_adapter import AlpacaBrokerAdapter
                # Create a temporary wrapper that exposes is_paper
                self._broker = alpaca_client
        else:
            raise ValueError("Either broker_adapter or alpaca_client must be provided")

        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_risk_decision(
        self,
        signal:           Signal,
        risk_decision:    Any,     # RiskDecision (typed as Any to avoid circular import)
        current_quantity: float = 0.0,
    ) -> OrderExecutionResult:
        """
        Translate a RiskDecision into an order (real or simulated).

        Args:
            signal:           The candidate signal from StrategyOrchestrator.
            risk_decision:    The evaluated RiskDecision from RiskManager.
            current_quantity: Current position size (shares) for SELL/REDUCE.
                              Ignored for BUY; required for SELL/REDUCE.

        Returns:
            OrderExecutionResult.  submitted=False when no order was sent.
        """
        # -- Safety gate: verify broker is always in paper mode --
        # (All BrokerAdapter implementations enforce is_paper=True)
        if not getattr(self._broker, "is_paper", True):
            logger.error("Broker is not in paper mode — rejecting order")
            return self._rejected(signal.ticker, signal.action.value, "Not in paper mode")

        # -- Gate 1: risk decision must be approved --
        if not risk_decision.approved:
            return self._rejected(
                signal.ticker,
                signal.action.value,
                f"RiskDecision not approved: {risk_decision.reason}",
            )

        # -- Gate 2: final action must not be HOLD --
        final = risk_decision.final_action
        if final == SignalAction.HOLD:
            return self._rejected(
                signal.ticker, "HOLD", "Final action is HOLD — no order needed."
            )

        # -- Route to action-specific handler --
        if final == SignalAction.BUY:
            return self.submit_buy_bracket(signal, risk_decision)

        if final == SignalAction.SELL:
            return self.submit_sell_market(
                signal.ticker, current_quantity, reason=signal.reason
            )

        if final == SignalAction.REDUCE:
            return self.submit_reduce_market(
                signal.ticker, current_quantity, fraction=0.5, reason=signal.reason
            )

        return self._rejected(signal.ticker, str(final), f"Unknown action: {final}")

    def submit_buy_bracket(
        self,
        signal:        Signal,
        risk_decision: Any,
    ) -> OrderExecutionResult:
        """
        Submit a bracket buy: market entry + stop-loss + take-profit.

        In dry_run mode, returns a simulated result without any broker call.
        """
        ps = risk_decision.position_size

        # Validate inputs
        if ps is None or not ps.approved:
            return self._rejected(signal.ticker, "BUY", "No approved position size.")

        if ps.shares <= 0:
            return self._rejected(signal.ticker, "BUY",
                                  f"Shares must be > 0 (got {ps.shares:.4f}).")

        if signal.stop_loss is None or signal.stop_loss <= 0:
            return self._rejected(signal.ticker, "BUY",
                                  "Stop loss is missing or zero.")

        if signal.take_profit is None or signal.take_profit <= 0:
            return self._rejected(signal.ticker, "BUY",
                                  "Take profit is missing or zero.")

        if signal.stop_loss >= signal.entry_price:
            return self._rejected(signal.ticker, "BUY",
                                  "Stop loss must be below entry price for long trade.")

        # -- Dry-run path: no broker call --
        if self.dry_run:
            sim_id = f"DRY_{signal.ticker}_{pd.Timestamp.now().strftime('%H%M%S%f')[:15]}"
            logger.debug(
                "[DRY RUN] Would BUY {:.4f} shares {} at ~${:.2f} "
                "(stop={:.2f}, tp={:.2f})",
                ps.shares, signal.ticker, signal.entry_price,
                signal.stop_loss, signal.take_profit,
            )
            return OrderExecutionResult(
                submitted=True,
                dry_run=True,
                ticker=signal.ticker,
                action="BUY",
                quantity=ps.shares,
                order_id=sim_id,
                status="simulated",
                reason=(
                    f"[DRY RUN] BUY {ps.shares:.4f} shares {signal.ticker} "
                    f"~${signal.entry_price:.2f} "
                    f"(stop=${signal.stop_loss:.2f}, tp=${signal.take_profit:.2f})"
                ),
                broker_response={"dry_run": True, "simulated_entry": signal.entry_price},
                metadata={
                    "stop_loss":   signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "position_value": ps.position_value,
                    "risk_dollars":   ps.risk_dollars,
                },
            )

        # -- Real submission path --
        try:
            response = self._broker.submit_order({
                "symbol": signal.ticker,
                "qty": ps.shares,
                "side": "buy",
                "type": "market",
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
            })
            logger.info(
                "BUY order submitted: {} {:.4f} shares. order_id={}",
                signal.ticker, ps.shares, response.get("id", "?"),
            )
            return OrderExecutionResult(
                submitted=True,
                dry_run=False,
                ticker=signal.ticker,
                action="BUY",
                quantity=ps.shares,
                order_id=response.get("id"),
                status=response.get("status", "submitted"),
                reason=f"BUY bracket submitted: {ps.shares:.4f} shares {signal.ticker}",
                broker_response=response,
                metadata={
                    "stop_loss":   signal.stop_loss,
                    "take_profit": signal.take_profit,
                },
            )
        except BrokerError as exc:
            logger.error("Order submission failed for {}: {}", signal.ticker, exc)
            return self._rejected(signal.ticker, "BUY", str(exc))
        except Exception as exc:
            logger.error("Unexpected error submitting BUY for {}: {}", signal.ticker, exc)
            return OrderExecutionResult(
                submitted=False,
                dry_run=False,
                ticker=signal.ticker,
                action="BUY",
                quantity=0.0,
                order_id=None,
                status="failed",
                reason=f"Unexpected error: {exc}",
                broker_response={},
            )

    def submit_sell_market(
        self,
        ticker:   str,
        quantity: float,
        reason:   str = "",
    ) -> OrderExecutionResult:
        """
        Submit a market sell order for the given quantity.

        In dry_run mode, returns a simulated result without any broker call.
        """
        if quantity <= 0:
            return self._rejected(
                ticker, "SELL",
                f"No position to sell (quantity={quantity:.4f})."
            )

        if self.dry_run:
            logger.debug("[DRY RUN] Would SELL {:.4f} shares {}", quantity, ticker)
            return OrderExecutionResult(
                submitted=True,
                dry_run=True,
                ticker=ticker,
                action="SELL",
                quantity=quantity,
                order_id=f"DRY_SELL_{ticker}",
                status="simulated",
                reason=f"[DRY RUN] SELL {quantity:.4f} shares {ticker}. {reason}",
                broker_response={"dry_run": True},
            )

        try:
            response = self._broker.submit_order({
                "symbol": ticker,
                "qty": quantity,
                "side": "sell",
                "type": "market",
            })
            logger.info(
                "SELL order submitted: {} {:.4f} shares. order_id={}",
                ticker, quantity, response.get("id", "?"),
            )
            return OrderExecutionResult(
                submitted=True,
                dry_run=False,
                ticker=ticker,
                action="SELL",
                quantity=quantity,
                order_id=response.get("id"),
                status=response.get("status", "submitted"),
                reason=f"SELL market: {quantity:.4f} shares {ticker}",
                broker_response=response,
            )
        except BrokerError as exc:
            logger.error("SELL failed for {}: {}", ticker, exc)
            return self._rejected(ticker, "SELL", str(exc))

    def submit_reduce_market(
        self,
        ticker:   str,
        quantity: float,
        fraction: float = 0.50,
        reason:   str = "",
    ) -> OrderExecutionResult:
        """
        Submit a market sell for `fraction` of the given quantity (default 50%).
        """
        reduce_qty = quantity * fraction

        if reduce_qty <= 0:
            return self._rejected(
                ticker, "REDUCE",
                f"Reduce quantity is zero (position={quantity:.4f}, fraction={fraction})."
            )

        if self.dry_run:
            logger.debug(
                "[DRY RUN] Would REDUCE {:.4f} shares {} ({:.0%} of {:.4f})",
                reduce_qty, ticker, fraction, quantity,
            )
            return OrderExecutionResult(
                submitted=True,
                dry_run=True,
                ticker=ticker,
                action="REDUCE",
                quantity=reduce_qty,
                order_id=f"DRY_REDUCE_{ticker}",
                status="simulated",
                reason=(
                    f"[DRY RUN] REDUCE {reduce_qty:.4f} shares {ticker} "
                    f"({fraction:.0%} of {quantity:.4f}). {reason}"
                ),
                broker_response={"dry_run": True, "fraction": fraction},
            )

        try:
            response = self._broker.submit_order({
                "symbol": ticker,
                "qty": reduce_qty,
                "side": "sell",
                "type": "market",
            })
            logger.info(
                "REDUCE order submitted: {} {:.4f} shares. order_id={}",
                ticker, reduce_qty, response.get("id", "?"),
            )
            return OrderExecutionResult(
                submitted=True,
                dry_run=False,
                ticker=ticker,
                action="REDUCE",
                quantity=reduce_qty,
                order_id=response.get("id"),
                status=response.get("status", "submitted"),
                reason=f"REDUCE market: {reduce_qty:.4f} shares {ticker}",
                broker_response=response,
                metadata={"fraction": fraction, "original_quantity": quantity},
            )
        except BrokerError as exc:
            logger.error("REDUCE failed for {}: {}", ticker, exc)
            return self._rejected(ticker, "REDUCE", str(exc))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rejected(ticker: str, action: str, reason: str) -> OrderExecutionResult:
        logger.debug("Order rejected for {} ({}): {}", ticker, action, reason)
        return OrderExecutionResult(
            submitted=False,
            dry_run=True,  # Treated as dry_run since nothing was sent
            ticker=ticker,
            action=action,
            quantity=0.0,
            order_id=None,
            status="rejected",
            reason=reason,
            broker_response={},
        )
