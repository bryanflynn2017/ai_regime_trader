"""
AlpacaBrokerAdapter — wraps AlpacaClientWrapper for the BrokerAdapter interface.

All calls delegate to the existing AlpacaClientWrapper so no Alpaca-specific
logic is duplicated.  The adapter adds the standard BrokerAdapter interface,
re-enforces paper-safety on every mutating call, and exposes the underlying
client for components (e.g. OrderExecutor) that still require it directly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from broker.base import BrokerAdapter, BrokerError


class AlpacaBrokerAdapter(BrokerAdapter):
    """
    BrokerAdapter backed by Alpaca paper trading.

    Requires alpaca-py (64-bit Python 3.10+ or 3.11).
    Refuses any configuration that would allow live trading.
    """

    def __init__(self, settings_override: Any = None, *, execute_paper: bool = False) -> None:
        from broker.alpaca_client import AlpacaClientWrapper
        self._client = AlpacaClientWrapper(settings_override=settings_override, dry_run=not execute_paper)
        self._submission_guard = None

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def is_configured(self) -> bool:
        return self._client.is_configured()

    @property
    def is_paper(self) -> bool:
        return True  # AlpacaClientWrapper enforces paper-only

    # ------------------------------------------------------------------
    # Account & clock
    # ------------------------------------------------------------------

    def get_account(self) -> Dict[str, Any]:
        try:
            return self._client.get_account()
        except Exception as exc:
            raise BrokerError(f"AlpacaBrokerAdapter.get_account failed: {exc}") from exc

    def get_clock(self) -> Dict[str, Any]:
        try:
            return self._client.get_clock()
        except Exception as exc:
            raise BrokerError(f"AlpacaBrokerAdapter.get_clock failed: {exc}") from exc

    def is_market_open(self) -> bool:
        try:
            return self._client.is_market_open()
        except Exception:
            return False  # Fail safe on error

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def list_positions(self) -> List[Dict[str, Any]]:
        try:
            return self._client.list_positions()
        except Exception as exc:
            raise BrokerError(f"AlpacaBrokerAdapter.list_positions failed: {exc}") from exc

    def get_position(self, ticker: str) -> Optional[Dict[str, Any]]:
        try:
            return self._client.get_position(ticker)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a durable managed order through Alpaca paper trading only.
        Legacy non-durable orders cannot bypass the lifecycle coordinator.
        """
        if "client_order_id" in order:
            return self._client.submit_managed_order(order, before_submit=self._submission_guard)
        raise BrokerError("Legacy submission disabled; use managed paper-session")

    def cancel_order(self, order_id: str) -> bool:
        try:
            if self._client.dry_run or self._client._s.EXECUTION_MODE != "paper":
                raise BrokerError("Paper cancellation requires explicit activation")
            self._client.verify_paper_account()
            return self._client.cancel_order(order_id)
        except Exception:
            return False

    def cancel_all_orders(self) -> int:
        try:
            return self._client.cancel_all_orders()
        except Exception as exc:
            raise BrokerError(f"cancel_all_orders failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Convenience / compatibility helpers
    # ------------------------------------------------------------------

    def validate_paper_safety(self) -> None:
        """Delegates to the underlying AlpacaClientWrapper safety check."""
        self._client.validate_paper_safety()

    def get_underlying_client(self):
        """
        Return the underlying AlpacaClientWrapper.

        Used by OrderExecutor which currently requires the wrapper directly.
        This method will be removed once OrderExecutor is updated to use
        BrokerAdapter.submit_order() directly.
        """
        return self._client

    def log_safe_info(self) -> Dict[str, Any]:
        return self._client.log_safe_info()

    def verify_paper_account(self) -> Dict[str, Any]:
        return self._client.verify_paper_account()

    def set_submission_guard(self, guard) -> None:
        """Durable owner installs final exact-approval/data/account/risk checks."""
        self._submission_guard = guard

    def get_order_by_client_id(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        return self._client.get_order_by_client_id(client_order_id)

    def list_open_orders(self) -> List[Dict[str, Any]]:
        orders = self._client.get_orders(status="open", limit=500)
        if len(orders) >= 500:
            raise BrokerError("Order snapshot may be truncated; reconciliation blocked")
        return orders
