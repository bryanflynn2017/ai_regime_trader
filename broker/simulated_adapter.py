"""
SimulatedBrokerAdapter — in-memory broker for dry-run and offline testing.

Works without any credentials or network access.
Always reports paper=True and is_configured=True.
Orders are accepted and logged in-memory but not routed anywhere.
The market is always considered open (no real clock check).
No secrets are ever stored or logged.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import pandas as pd

from broker.base import BrokerAdapter, BrokerError


class SimulatedBrokerAdapter(BrokerAdapter):
    """
    In-memory broker simulation for dry-run and offline testing.

    Provides a full BrokerAdapter interface without requiring:
      - Alpaca credentials
      - alpaca-py package
      - Network access

    This is the adapter used when --dry-run is active and no credentials
    are available.  It is NOT a realistic market simulator — fills are
    instantaneous and at the requested price.
    """

    def __init__(self, starting_equity: float = 100_000.0) -> None:
        self._equity          = starting_equity
        self._cash            = starting_equity
        self._positions:      Dict[str, Dict[str, Any]] = {}
        self._open_orders:    List[Dict[str, Any]]      = []
        self._filled_orders:  List[Dict[str, Any]]      = []

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def is_configured(self) -> bool:
        return True  # Always ready — no credentials needed

    @property
    def is_paper(self) -> bool:
        return True  # Always paper/simulation

    # ------------------------------------------------------------------
    # Account & clock
    # ------------------------------------------------------------------

    def get_account(self) -> Dict[str, Any]:
        return {
            "status":       "ACTIVE",
            "equity":       str(self._equity),
            "cash":         str(self._cash),
            "buying_power": str(self._cash),
            "paper":        True,
            "currency":     "USD",
        }

    def get_clock(self) -> Dict[str, Any]:
        now = str(pd.Timestamp.now())
        return {
            "is_open":    True,
            "next_open":  now,
            "next_close": now,
        }

    def is_market_open(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def list_positions(self) -> List[Dict[str, Any]]:
        return list(self._positions.values())

    def get_position(self, ticker: str) -> Optional[Dict[str, Any]]:
        return self._positions.get(ticker.upper())

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate an order fill.

        No real execution — generates a synthetic order ID and records
        the fill in memory.  Always succeeds for qty > 0.
        """
        qty = float(order.get("qty", 0))
        if qty <= 0:
            raise BrokerError(
                f"SimulatedBrokerAdapter: qty must be > 0, got {qty}"
            )

        order_id = f"SIM-{uuid.uuid4().hex[:8].upper()}"
        filled = {
            "id":     order_id,
            "status": "filled",
            "symbol": str(order.get("symbol", "")).upper(),
            "qty":    str(qty),
            "side":   str(order.get("side", "buy")).lower(),
            "type":   str(order.get("type", "market")),
        }
        self._filled_orders.append(filled)
        return filled

    def cancel_order(self, order_id: str) -> bool:
        return True  # All cancellations succeed in simulation

    def cancel_all_orders(self) -> int:
        n = len(self._open_orders)
        self._open_orders.clear()
        return n

    # ------------------------------------------------------------------
    # Logging safety
    # ------------------------------------------------------------------

    def log_safe_info(self) -> Dict[str, Any]:
        return {
            "adapter":       "SimulatedBrokerAdapter",
            "equity":        self._equity,
            "n_positions":   len(self._positions),
            "n_filled":      len(self._filled_orders),
            # No credentials to expose
        }
