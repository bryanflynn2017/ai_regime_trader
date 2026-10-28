"""
Position tracker — reads live portfolio state from any BrokerAdapter.

Converts string-valued numeric fields to Python floats safely.
All calculations return 0.0 rather than crashing when equity is zero or
positions are empty.

Unit-testable without real credentials: inject a mock BrokerAdapter or use
SimulatedBrokerAdapter.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from broker.base import BrokerAdapter, BrokerError

try:
    from loguru import logger  # type: ignore
except ImportError:
    logger = logging.getLogger(__name__)  # type: ignore


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PositionSnapshot:
    """State of one open position at a moment in time."""
    ticker:           str
    quantity:         float
    market_value:     float
    avg_entry_price:  float
    current_price:    float
    unrealized_pl:    float
    unrealized_plpc:  float        # fraction (e.g. 0.05 = +5%)
    side:             str          # "long" or "short"
    metadata:         Dict[str, Any] = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        return self.side.lower() == "long"


@dataclass
class PortfolioSnapshot:
    """Full portfolio state at a moment in time."""
    equity:               float
    cash:                 float
    buying_power:         float
    portfolio_exposure:   float          # total long value / equity
    positions:            Dict[str, PositionSnapshot]
    open_positions_count: int
    ticker_exposures:     Dict[str, float]  # ticker -> market_value / equity
    timestamp:            pd.Timestamp
    metadata:             Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PositionTracker
# ---------------------------------------------------------------------------

class PositionTracker:
    """
    Reads current position and account state from any BrokerAdapter.

    Works with any BrokerAdapter implementation:
        from broker.simulated_adapter import SimulatedBrokerAdapter
        tracker = PositionTracker(broker_adapter=SimulatedBrokerAdapter())

    For backward compatibility, also accepts AlpacaClientWrapper directly.
    """

    def __init__(
        self,
        alpaca_client: Any = None,  # Deprecated; use broker_adapter instead
        broker_adapter: Optional[BrokerAdapter] = None,
    ) -> None:
        # Support both old (alpaca_client) and new (broker_adapter) interfaces
        if broker_adapter is not None:
            self._broker = broker_adapter
        elif alpaca_client is not None:
            # Backward compat: use alpaca_client directly (duck typing)
            self._broker = alpaca_client
        else:
            raise ValueError("Either broker_adapter or alpaca_client must be provided")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        """
        Fetch a complete snapshot of the current portfolio from the broker.

        Returns:
            PortfolioSnapshot with equity, cash, positions, and exposures.

        Raises:
            BrokerError: If the broker call fails.
        """
        account    = self._broker.get_account()
        positions_raw = self._broker.list_positions()

        equity       = _to_float(account.get("equity",       "0"))
        cash         = _to_float(account.get("cash",         "0"))
        buying_power = _to_float(account.get("buying_power", "0"))

        if equity <= 0:
            logger.warning(
                "Portfolio equity is {:.2f} — exposure calculations will be 0.0.",
                equity,
            )

        positions:         Dict[str, PositionSnapshot] = {}
        ticker_exposures:  Dict[str, float]            = {}
        total_long_value   = 0.0

        for pos_dict in positions_raw:
            ticker       = pos_dict.get("symbol", "")
            market_value = _to_float(pos_dict.get("market_value", "0"))

            snap = PositionSnapshot(
                ticker          = ticker,
                quantity        = _to_float(pos_dict.get("qty",             "0")),
                market_value    = market_value,
                avg_entry_price = _to_float(pos_dict.get("avg_entry_price", "0")),
                current_price   = _to_float(pos_dict.get("current_price",   "0")),
                unrealized_pl   = _to_float(pos_dict.get("unrealized_pl",   "0")),
                unrealized_plpc = _to_float(pos_dict.get("unrealized_plpc", "0")),
                side            = pos_dict.get("side", "long"),
            )
            positions[ticker] = snap

            if snap.is_long:
                total_long_value += market_value

            ticker_exposures[ticker] = (
                market_value / equity if equity > 0 else 0.0
            )

        portfolio_exposure = total_long_value / equity if equity > 0 else 0.0

        return PortfolioSnapshot(
            equity               = equity,
            cash                 = cash,
            buying_power         = buying_power,
            portfolio_exposure   = portfolio_exposure,
            positions            = positions,
            open_positions_count = len(positions),
            ticker_exposures     = ticker_exposures,
            timestamp            = pd.Timestamp.now(),
            metadata             = {"account_status": account.get("status", "unknown")},
        )

    def get_position_snapshot(self, ticker: str) -> Optional[PositionSnapshot]:
        """
        Return the snapshot for a single ticker, or None if no open position.
        """
        pos_dict = self._broker.get_position(ticker.upper())
        if pos_dict is None:
            return None

        return PositionSnapshot(
            ticker          = pos_dict.get("symbol", ticker),
            quantity        = _to_float(pos_dict.get("qty",             "0")),
            market_value    = _to_float(pos_dict.get("market_value",    "0")),
            avg_entry_price = _to_float(pos_dict.get("avg_entry_price", "0")),
            current_price   = _to_float(pos_dict.get("current_price",   "0")),
            unrealized_pl   = _to_float(pos_dict.get("unrealized_pl",   "0")),
            unrealized_plpc = _to_float(pos_dict.get("unrealized_plpc", "0")),
            side            = pos_dict.get("side", "long"),
        )

    def get_current_portfolio_exposure(self) -> float:
        """
        Return total long market value / equity as a fraction [0, 1].
        Returns 0.0 if equity is zero or no positions exist.
        """
        try:
            return self.get_portfolio_snapshot().portfolio_exposure
        except BrokerError as exc:
            logger.warning("Could not get portfolio exposure: {}", exc)
            return 0.0

    def get_current_ticker_exposure(self, ticker: str) -> float:
        """
        Return ticker market value / equity as a fraction [0, 1].
        Returns 0.0 if the ticker has no position or equity is zero.
        """
        try:
            snapshot = self.get_portfolio_snapshot()
            return snapshot.ticker_exposures.get(ticker.upper(), 0.0)
        except AlpacaConnectionError as exc:
            logger.warning("Could not get ticker exposure for {}: {}", ticker, exc)
            return 0.0

    def get_open_positions_count(self) -> int:
        """Return the number of currently open positions."""
        try:
            return self.get_portfolio_snapshot().open_positions_count
        except AlpacaConnectionError as exc:
            logger.warning("Could not get open positions count: {}", exc)
            return 0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value (possibly a string from Alpaca) to float.
    Returns `default` on any conversion error.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
