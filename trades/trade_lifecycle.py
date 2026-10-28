"""
Trade lifecycle analytics (journal/analytics only).

Joins closed ManualTrade records with their trade_monitor.jsonl history to
understand how monitor actions relate to outcomes. This module is READ-ONLY:
it never places orders, contacts a broker, or mutates any journal.

Definitions:
  - first_monitor_action: the action of the earliest monitor event for a trade.
  - "alert": the first monitor event whose action is not HOLD
    (WATCH / TARGET_1_HIT / TARGET_2_HIT / STOP_HIT / EXIT_VWAP_LOST).
  - time_to_first_alert_seconds: first alert timestamp minus entry_time.
  - time_in_trade_seconds: exit_time minus entry_time.
  - exit_reason: the action of the LAST exit-class monitor event
    (STOP_HIT / TARGET_1_HIT / TARGET_2_HIT / EXIT_VWAP_LOST) recorded at or
    before exit_time. If no exit-class event preceded the close, the exit is
    classified as MANUAL.
  - exit_action_followed: True when the trade was closed after the monitor
    signalled an exit (exit_reason != MANUAL); False for MANUAL exits.

Older trade records without the newer journal fields remain fully compatible:
trades with no monitor history simply classify as MANUAL with no alert times.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from trades.trade_tracker import ManualTrade, TradeStatus, TradeTracker

# Monitor actions that signal an exit (mirrors TradeMonitorAction values
# without importing it, so analytics never depends on monitor internals).
EXIT_ACTIONS = ("STOP_HIT", "TARGET_1_HIT", "TARGET_2_HIT", "EXIT_VWAP_LOST")
EXIT_REASON_MANUAL = "MANUAL"

#: Display order for exit-reason reports.
EXIT_REASON_ORDER = ("EXIT_VWAP_LOST", "STOP_HIT", "TARGET_1_HIT", "TARGET_2_HIT", EXIT_REASON_MANUAL)

#: Friendly labels for the Monitor Effectiveness report.
EXIT_REASON_LABELS = {
    "EXIT_VWAP_LOST": "VWAP-loss exits",
    "STOP_HIT": "Stop exits",
    "TARGET_1_HIT": "Target 1 exits",
    "TARGET_2_HIT": "Target 2 exits",
    EXIT_REASON_MANUAL: "Manual exits",
}


def _parse_dt(value: Any) -> Optional[datetime]:
    """Parse an ISO timestamp string; return None on failure. Naive datetimes."""
    if value is None or value == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        return None


@dataclass
class TradeLifecycle:
    """Lifecycle summary for one closed manual trade."""

    trade_id: str
    ticker: str
    setup_type: str
    realized_pnl: float
    first_monitor_action: Optional[str] = None
    first_alert_action: Optional[str] = None
    time_to_first_alert_seconds: Optional[float] = None
    time_in_trade_seconds: Optional[float] = None
    exit_reason: str = EXIT_REASON_MANUAL
    exit_action_followed: bool = False
    monitor_event_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradeLifecycleResult:
    """Computed lifecycle analytics for closed manual trades."""

    lifecycles: List[Dict[str, Any]] = field(default_factory=list)
    by_exit_reason: List[Dict[str, Any]] = field(default_factory=list)
    effectiveness: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lifecycles": self.lifecycles,
            "by_exit_reason": self.by_exit_reason,
            "effectiveness": self.effectiveness,
        }


class TradeLifecycleAnalytics:
    """Join closed trades with monitor history and compute lifecycle stats."""

    def __init__(
        self,
        trades_path: str | Path = "logs/manual_trades.jsonl",
        monitor_path: str | Path = "logs/trade_monitor.jsonl",
    ) -> None:
        self.tracker = TradeTracker(storage_path=trades_path)
        self.monitor_path = Path(monitor_path)

    # -- loading -------------------------------------------------------------

    def _load_closed_trades(self) -> List[ManualTrade]:
        return [
            trade
            for trade in self.tracker.list_all_trades()
            if str(trade.status).upper() == TradeStatus.CLOSED.value
            and not bool(trade.exclude_from_analytics)
        ]

    def _load_monitor_events(self) -> List[Dict[str, Any]]:
        if not self.monitor_path.exists():
            return []
        events: List[Dict[str, Any]] = []
        with open(self.monitor_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events

    # -- lifecycle construction ----------------------------------------------

    @staticmethod
    def _build_lifecycle(trade: ManualTrade, events: List[Dict[str, Any]]) -> TradeLifecycle:
        """Build one TradeLifecycle from a closed trade and its monitor events."""
        entry_dt = _parse_dt(trade.entry_time)
        exit_dt = _parse_dt(trade.exit_time)

        # Sort events chronologically; events with unparseable timestamps sort first.
        dated = sorted(
            ((_parse_dt(e.get("timestamp")), e) for e in events),
            key=lambda pair: (pair[0] is not None, pair[0] or datetime.min),
        )

        first_monitor_action: Optional[str] = None
        first_alert_action: Optional[str] = None
        time_to_first_alert: Optional[float] = None
        exit_reason = EXIT_REASON_MANUAL

        for event_dt, event in dated:
            action = str(event.get("action", "")).upper()
            if not action:
                continue
            if first_monitor_action is None:
                first_monitor_action = action
            if first_alert_action is None and action != "HOLD":
                first_alert_action = action
                if entry_dt is not None and event_dt is not None:
                    time_to_first_alert = (event_dt - entry_dt).total_seconds()
            if action in EXIT_ACTIONS:
                # Only signals at/before the close explain the exit.
                if exit_dt is None or event_dt is None or event_dt <= exit_dt:
                    exit_reason = action

        time_in_trade: Optional[float] = None
        if entry_dt is not None and exit_dt is not None:
            time_in_trade = (exit_dt - entry_dt).total_seconds()

        return TradeLifecycle(
            trade_id=trade.trade_id,
            ticker=trade.ticker,
            setup_type=trade.setup_type,
            realized_pnl=float(trade.realized_pnl or 0.0),
            first_monitor_action=first_monitor_action,
            first_alert_action=first_alert_action,
            time_to_first_alert_seconds=time_to_first_alert,
            time_in_trade_seconds=time_in_trade,
            exit_reason=exit_reason,
            exit_action_followed=exit_reason != EXIT_REASON_MANUAL,
            monitor_event_count=len(events),
        )

    def build_lifecycles(self) -> List[TradeLifecycle]:
        """Return a TradeLifecycle for every closed, non-excluded trade."""
        events_by_trade: Dict[str, List[Dict[str, Any]]] = {}
        for event in self._load_monitor_events():
            trade_id = str(event.get("trade_id", ""))
            if trade_id:
                events_by_trade.setdefault(trade_id, []).append(event)

        return [
            self._build_lifecycle(trade, events_by_trade.get(trade.trade_id, []))
            for trade in self._load_closed_trades()
        ]

    # -- aggregation -----------------------------------------------------------

    @staticmethod
    def _exit_reason_stats(lifecycles: List[TradeLifecycle]) -> List[Dict[str, Any]]:
        """Group lifecycles by exit reason and compute outcome stats."""
        rows: List[Dict[str, Any]] = []
        for reason in EXIT_REASON_ORDER:
            group = [lc for lc in lifecycles if lc.exit_reason == reason]
            if not group:
                continue
            pnl = pd.Series([lc.realized_pnl for lc in group], dtype=float)
            times = [lc.time_in_trade_seconds for lc in group if lc.time_in_trade_seconds is not None]
            rows.append({
                "exit_reason": reason,
                "label": EXIT_REASON_LABELS.get(reason, reason),
                "trades": int(len(group)),
                "total_pnl": float(pnl.sum()),
                "avg_pnl": float(pnl.mean()),
                "win_rate": float((pnl > 0).sum() / len(group)),
                "followed_count": int(sum(1 for lc in group if lc.exit_action_followed)),
                "avg_time_in_trade_seconds": float(sum(times) / len(times)) if times else None,
            })
        return rows

    def analyze(self) -> TradeLifecycleResult:
        """Return per-trade lifecycles plus exit-reason and effectiveness stats."""
        lifecycles = self.build_lifecycles()
        by_exit_reason = self._exit_reason_stats(lifecycles)
        return TradeLifecycleResult(
            lifecycles=[lc.to_dict() for lc in lifecycles],
            by_exit_reason=by_exit_reason,
            # Effectiveness is the same grouping presented per-reason with labels;
            # kept as a separate list so callers can render it independently.
            effectiveness=by_exit_reason,
        )

    # -- dashboard entry point -------------------------------------------------

    @staticmethod
    def analyze_frames(trades_df: pd.DataFrame, monitor_df: pd.DataFrame) -> TradeLifecycleResult:
        """
        Analyze DataFrames loaded from manual_trades.jsonl / trade_monitor.jsonl.

        Mirrors analyze() so the dashboard and CLI stay in sync without re-reading
        files. Missing/empty input returns an empty result.
        """
        if trades_df is None or trades_df.empty or "status" not in trades_df.columns:
            return TradeLifecycleResult()

        status = trades_df["status"].fillna("").astype(str).str.upper()
        closed = trades_df[status == TradeStatus.CLOSED.value].copy()
        if "exclude_from_analytics" in closed.columns:
            closed = closed[~closed["exclude_from_analytics"].fillna(False).astype(bool)]
        if closed.empty:
            return TradeLifecycleResult()
        # JSONL rows missing newer fields load as NaN; ManualTrade expects None.
        closed = closed.astype(object).where(pd.notna(closed), None)

        events_by_trade: Dict[str, List[Dict[str, Any]]] = {}
        if monitor_df is not None and not monitor_df.empty and "trade_id" in monitor_df.columns:
            for record in monitor_df.to_dict(orient="records"):
                trade_id = str(record.get("trade_id", ""))
                if trade_id:
                    events_by_trade.setdefault(trade_id, []).append(record)

        lifecycles = [
            TradeLifecycleAnalytics._build_lifecycle(
                ManualTrade.from_dict(record),
                events_by_trade.get(str(record.get("trade_id", "")), []),
            )
            for record in closed.to_dict(orient="records")
        ]
        by_exit_reason = TradeLifecycleAnalytics._exit_reason_stats(lifecycles)
        return TradeLifecycleResult(
            lifecycles=[lc.to_dict() for lc in lifecycles],
            by_exit_reason=by_exit_reason,
            effectiveness=by_exit_reason,
        )
