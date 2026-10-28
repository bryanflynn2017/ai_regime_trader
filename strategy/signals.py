"""
Signal dataclass definition.

A Signal represents a candidate trading action produced by the strategy layer.
It is NOT an order — it must pass through risk validation before becoming one.

Phase 4 will implement:
- Signal dataclass with all required fields
- SignalAction enum: BUY, SELL, HOLD, REDUCE
- Signal factory functions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class SignalAction(str, Enum):
    """Possible actions a signal can recommend."""
    BUY    = "BUY"
    SELL   = "SELL"
    HOLD   = "HOLD"
    REDUCE = "REDUCE"


class RegimeLabel(str, Enum):
    """Market regime labels used throughout the system."""
    CRASH    = "crash"
    BEAR     = "bear"
    NEUTRAL  = "neutral"
    BULL     = "bull"
    EUPHORIA = "euphoria"
    UNKNOWN  = "unknown"


@dataclass
class Signal:
    """
    A candidate trade signal produced by the strategy orchestrator.

    Fields are intentionally explicit — every signal must carry enough
    information to be audited without looking up additional state.
    """
    ticker: str
    timestamp: datetime
    action: SignalAction
    confidence: float            # 0.0 – 1.0
    reason: str                  # Human-readable explanation
    regime: RegimeLabel
    entry_price: float
    stop_loss: float             # Absolute price level (not percentage)
    take_profit: float           # Absolute price level (not percentage)
    max_position_value: float    # Dollar ceiling for this position
    metadata: Dict[str, Any] = field(default_factory=dict)

    def stop_loss_pct(self) -> float:
        """Return stop loss as a fraction of entry price."""
        return abs(self.entry_price - self.stop_loss) / self.entry_price

    def take_profit_pct(self) -> float:
        """Return take profit as a fraction of entry price."""
        return abs(self.take_profit - self.entry_price) / self.entry_price

    def reward_risk_ratio(self) -> float:
        """Return reward/risk ratio. A value >= 2.0 is generally acceptable."""
        stop_dist = abs(self.entry_price - self.stop_loss)
        if stop_dist == 0:
            return 0.0
        return abs(self.take_profit - self.entry_price) / stop_dist
