"""
BrokerAdapter abstract base class.

The rest of the system depends only on this interface — not on Alpaca or
any specific brokerage.  To add a new broker, subclass BrokerAdapter and
implement all abstract methods.

Safety invariants that every implementation MUST uphold:
  - is_paper must always return True
  - submit_order must refuse live-market routing
  - validate_paper_safety() is called before mutating calls in real adapters
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BrokerError(Exception):
    """Raised when a broker adapter operation fails."""


class OrderNotSubmitted(BrokerError):
    """Definitive local preflight failure: no order request reached transport."""


class BrokerAdapter(ABC):
    """
    Abstract interface for paper-trading broker backends.

    Concrete implementations:
        AlpacaBrokerAdapter    — Alpaca paper trading (requires alpaca-py + 64-bit Python)
        SimulatedBrokerAdapter — In-memory simulation (no credentials needed)

    Every implementation must set is_paper = True and must never route
    orders to a live exchange.
    """

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """True if the adapter has everything needed to operate (credentials, etc.)."""

    @property
    @abstractmethod
    def is_paper(self) -> bool:
        """True if in paper/simulation mode — must always be True for all adapters."""

    @abstractmethod
    def get_account(self) -> Dict[str, Any]:
        """Return account equity, cash, buying_power, and status."""

    @abstractmethod
    def get_clock(self) -> Dict[str, Any]:
        """Return market clock: is_open, next_open, next_close."""

    @abstractmethod
    def is_market_open(self) -> bool:
        """Return True if the primary market is currently open."""

    @abstractmethod
    def list_positions(self) -> List[Dict[str, Any]]:
        """Return all open positions as a list of dicts."""

    @abstractmethod
    def get_position(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return the open position for ticker, or None if no position."""

    @abstractmethod
    def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit an order.

        Minimum required keys in `order`:
            symbol (str)   — equity ticker
            qty    (float) — number of shares
            side   (str)   — 'buy' or 'sell'
            type   (str)   — 'market' or 'limit'

        Returns a dict with at minimum:
            id     (str) — broker order ID
            status (str) — e.g. 'filled', 'pending_new'
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID.  Returns True if cancelled successfully."""

    @abstractmethod
    def cancel_all_orders(self) -> int:
        """Cancel all open orders.  Returns count of cancelled orders."""

    def log_safe_info(self) -> Dict[str, Any]:
        """Return a dict safe for logging — no credentials or secrets."""
        return {
            "adapter":       type(self).__name__,
            "is_configured": self.is_configured,
            "is_paper":      self.is_paper,
        }

    def verify_paper_account(self) -> Dict[str, Any]:
        """Managed execution requires positive endpoint/account verification."""
        raise BrokerError("Adapter does not support verified managed paper execution")

    def get_order_by_client_id(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        """None only for definitive absence; all transport failures must raise."""
        raise BrokerError("Adapter does not support durable order lookup")

    def list_open_orders(self) -> List[Dict[str, Any]]:
        """Complete account snapshot; truncated responses must raise."""
        raise BrokerError("Adapter does not support order reconciliation")
