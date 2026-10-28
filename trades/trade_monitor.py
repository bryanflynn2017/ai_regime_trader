"""
Manual trade monitoring (journal + market data only).

This module evaluates open ManualTrade records against recent OHLCV data and
returns advisory actions. It never places orders, contacts a broker, or changes
the trade journal by itself.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from trades.trade_tracker import ManualTrade


class TradeMonitorAction(str, Enum):
    """Advisory actions for a manually tracked long trade."""

    HOLD = "HOLD"
    WATCH = "WATCH"
    TARGET_1_HIT = "TARGET_1_HIT"
    TARGET_2_HIT = "TARGET_2_HIT"
    STOP_HIT = "STOP_HIT"
    EXIT_VWAP_LOST = "EXIT_VWAP_LOST"


ACTION_GUIDANCE: Dict[TradeMonitorAction, str] = {
    TradeMonitorAction.HOLD: "Hold. Trade remains between management levels.",
    TradeMonitorAction.WATCH: "Watch closely. Trade is at or above +1R; consider moving stop to breakeven.",
    TradeMonitorAction.TARGET_1_HIT: "Target 1 hit. Consider taking partial profit and/or moving stop to breakeven.",
    TradeMonitorAction.TARGET_2_HIT: "Target 2 hit. Consider closing remaining position.",
    TradeMonitorAction.STOP_HIT: "Stop hit. Trade invalidated.",
    TradeMonitorAction.EXIT_VWAP_LOST: "Session VWAP lost. Consider exiting or reducing.",
}


def guidance_for_action(action: TradeMonitorAction) -> str:
    """Return human-readable management guidance for a monitor action."""
    return ACTION_GUIDANCE[action]


@dataclass
class TradeMonitorResult:
    """Result of evaluating one ManualTrade against latest market data."""

    trade_id: str
    ticker: str
    action: TradeMonitorAction
    current_price: float
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    shares: float
    unrealized_pnl: float
    r_multiple: float
    reason: str
    guidance: str
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.value
        return data


class TradeMonitor:
    """Evaluate open manual trades using latest OHLCV bars."""

    def monitor_trade(self, trade: ManualTrade, df: Optional[pd.DataFrame]) -> TradeMonitorResult:
        """Return an advisory monitor result for one trade."""
        warnings: List[str] = []

        if df is None or df.empty:
            warnings.append("No market data available.")
            return TradeMonitorResult(
                trade_id=trade.trade_id,
                ticker=trade.ticker,
                action=TradeMonitorAction.WATCH,
                current_price=0.0,
                entry_price=float(trade.entry_price),
                stop_loss=float(trade.stop_loss),
                target_1=float(trade.target_1),
                target_2=float(trade.target_2),
                shares=float(trade.shares),
                unrealized_pnl=0.0,
                r_multiple=0.0,
                reason="WATCH: No market data available.",
                guidance=guidance_for_action(TradeMonitorAction.WATCH),
                warnings=warnings,
                metadata={"data_available": False},
            )

        latest = df.iloc[-1]
        close = float(latest["close"])
        high = float(latest["high"])
        low = float(latest["low"])

        entry = float(trade.entry_price)
        stop = float(trade.stop_loss)
        target_1 = float(trade.target_1)
        target_2 = float(trade.target_2)
        shares = float(trade.shares)

        unrealized_pnl = (close - entry) * shares
        risk_per_share = entry - stop
        if risk_per_share <= 0:
            r_multiple = 0.0
            warnings.append("Invalid risk_per_share: entry_price must be above stop_loss.")
        else:
            r_multiple = (close - entry) / risk_per_share

        session_vwap = latest.get("session_vwap")
        has_session_vwap = pd.notna(session_vwap)
        session_vwap_value = float(session_vwap) if has_session_vwap else None

        if low <= stop:
            action = TradeMonitorAction.STOP_HIT
            reason = "STOP_HIT: latest low touched or crossed stop loss."
        elif high >= target_2:
            action = TradeMonitorAction.TARGET_2_HIT
            reason = "TARGET_2_HIT: latest high touched or crossed target 2."
        elif high >= target_1:
            action = TradeMonitorAction.TARGET_1_HIT
            reason = "TARGET_1_HIT: latest high touched or crossed target 1."
        elif has_session_vwap and close < float(session_vwap):
            action = TradeMonitorAction.EXIT_VWAP_LOST
            reason = "EXIT_VWAP_LOST: close is below session VWAP."
        elif r_multiple >= 1.0:
            action = TradeMonitorAction.WATCH
            reason = "WATCH: trade is at or above 1R; monitor for target or reversal."
        else:
            action = TradeMonitorAction.HOLD
            reason = "HOLD: trade remains between entry management levels."

        return TradeMonitorResult(
            trade_id=trade.trade_id,
            ticker=trade.ticker,
            action=action,
            current_price=close,
            entry_price=entry,
            stop_loss=stop,
            target_1=target_1,
            target_2=target_2,
            shares=shares,
            unrealized_pnl=unrealized_pnl,
            r_multiple=r_multiple,
            reason=reason,
            guidance=guidance_for_action(action),
            warnings=warnings,
            metadata={
                "data_available": True,
                "latest_high": high,
                "latest_low": low,
                "risk_per_share": risk_per_share,
                "session_vwap": session_vwap_value,
            },
        )

    def monitor_open_trades(
        self,
        trades: List[ManualTrade],
        data_by_ticker: Dict[str, pd.DataFrame],
    ) -> List[TradeMonitorResult]:
        """Evaluate every open trade using the data frame for its ticker."""
        results: List[TradeMonitorResult] = []
        for trade in trades:
            df = data_by_ticker.get(trade.ticker)
            if df is None:
                df = data_by_ticker.get(trade.ticker.upper())
            results.append(self.monitor_trade(trade, df))
        return results
