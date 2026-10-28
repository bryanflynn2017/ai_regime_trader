"""
Read-only analytics for manually tracked trades.

This module analyzes completed ManualTrade records. It never places orders,
contacts brokers, or mutates the trade journal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from trades.trade_tracker import ManualTrade, TradeStatus, TradeTracker


@dataclass
class TradeAnalyticsResult:
    """Computed performance analytics for closed manual trades."""

    overall: Dict[str, Any]
    by_setup_type: List[Dict[str, Any]] = field(default_factory=list)
    by_ticker: List[Dict[str, Any]] = field(default_factory=list)
    by_setup_rank: List[Dict[str, Any]] = field(default_factory=list)
    by_test_type: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "by_setup_type": self.by_setup_type,
            "by_ticker": self.by_ticker,
            "by_setup_rank": self.by_setup_rank,
            "by_test_type": self.by_test_type,
        }


class TradeAnalytics:
    """Compute read-only analytics over closed manual trades."""

    def __init__(self, storage_path: str | Path = "logs/manual_trades.jsonl") -> None:
        self.tracker = TradeTracker(storage_path=storage_path)

    def load_closed_trades(self) -> List[ManualTrade]:
        """Load only CLOSED trades from the manual trade journal."""
        return [
            trade
            for trade in self.tracker.list_all_trades()
            if str(trade.status).upper() == TradeStatus.CLOSED.value
            and not bool(trade.exclude_from_analytics)
        ]

    def count_excluded_closed_trades(self) -> int:
        """Count CLOSED trades intentionally excluded from analytics."""
        return sum(
            1
            for trade in self.tracker.list_all_trades()
            if str(trade.status).upper() == TradeStatus.CLOSED.value
            and bool(trade.exclude_from_analytics)
        )

    def analyze(self) -> TradeAnalyticsResult:
        """Return overall and grouped analytics for closed manual trades."""
        excluded_count = self.count_excluded_closed_trades()
        trades = self.load_closed_trades()
        df = self._trades_to_dataframe(trades)
        if df.empty:
            overall = self._stats_from_pnl(pd.Series(dtype=float))
            overall["excluded_trades"] = excluded_count
            return TradeAnalyticsResult(
                overall=overall,
                by_setup_type=[],
                by_ticker=[],
                by_setup_rank=[],
                by_test_type=[],
            )

        overall = self._stats_from_frame(df)
        overall["excluded_trades"] = excluded_count
        by_setup_type = self._breakdown(df, "setup_type")
        by_ticker = self._breakdown(df, "ticker")
        by_setup_rank = self._breakdown(df, "setup_rank")
        by_test_type = self._breakdown(df, "test_type")
        return TradeAnalyticsResult(
            overall=overall,
            by_setup_type=by_setup_type,
            by_ticker=by_ticker,
            by_setup_rank=by_setup_rank,
            by_test_type=by_test_type,
        )

    @staticmethod
    def analyze_dataframe(df: pd.DataFrame) -> TradeAnalyticsResult:
        """Analyze a DataFrame loaded from manual_trades.jsonl."""
        if df is None or df.empty or "status" not in df.columns:
            return TradeAnalyticsResult(
                overall=TradeAnalytics._stats_from_pnl(pd.Series(dtype=float)),
                by_setup_type=[],
                by_ticker=[],
                by_setup_rank=[],
                by_test_type=[],
            )

        closed = df[df["status"].fillna("").astype(str).str.upper() == TradeStatus.CLOSED.value].copy()
        excluded_count = 0
        if "exclude_from_analytics" in closed.columns:
            excluded_mask = closed["exclude_from_analytics"].fillna(False).astype(bool)
            excluded_count = int(excluded_mask.sum())
            closed = closed[~excluded_mask].copy()
        if closed.empty:
            overall = TradeAnalytics._stats_from_pnl(pd.Series(dtype=float))
            overall["excluded_trades"] = excluded_count
            return TradeAnalyticsResult(
                overall=overall,
                by_setup_type=[],
                by_ticker=[],
                by_setup_rank=[],
                by_test_type=[],
            )

        closed["realized_pnl"] = pd.to_numeric(closed.get("realized_pnl"), errors="coerce").fillna(0.0)
        overall = TradeAnalytics._stats_from_frame(closed)
        overall["excluded_trades"] = excluded_count
        return TradeAnalyticsResult(
            overall=overall,
            by_setup_type=TradeAnalytics._breakdown(closed, "setup_type"),
            by_ticker=TradeAnalytics._breakdown(closed, "ticker"),
            by_setup_rank=TradeAnalytics._breakdown(closed, "setup_rank"),
            by_test_type=TradeAnalytics._breakdown(closed, "test_type"),
        )

    @staticmethod
    def _trades_to_dataframe(trades: List[ManualTrade]) -> pd.DataFrame:
        records = [trade.to_dict() for trade in trades]
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
        return df

    @staticmethod
    def _stats_from_frame(df: pd.DataFrame) -> Dict[str, Any]:
        stats = TradeAnalytics._stats_from_pnl(df["realized_pnl"])
        stats["average_setup_confidence"] = TradeAnalytics._average_optional_numeric(df, "setup_confidence")
        stats["average_setup_score"] = TradeAnalytics._average_optional_numeric(df, "setup_score")
        stats["mistake_count"] = TradeAnalytics._mistake_count(df)
        return stats

    @staticmethod
    def _mistake_count(df: pd.DataFrame) -> int:
        if df.empty or "mistake" not in df.columns:
            return 0
        return int(df["mistake"].fillna(False).astype(bool).sum())

    @staticmethod
    def _average_optional_numeric(df: pd.DataFrame, column: str) -> float:
        if df.empty or column not in df.columns:
            return 0.0
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        return float(values.mean()) if not values.empty else 0.0

    @staticmethod
    def _stats_from_pnl(pnl_values: pd.Series) -> Dict[str, Any]:
        pnl = pd.to_numeric(pnl_values, errors="coerce").dropna()
        total_trades = int(len(pnl))
        if total_trades == 0:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_realized_pnl": 0.0,
                "average_pnl": 0.0,
                "average_winner": 0.0,
                "average_loser": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "average_setup_confidence": 0.0,
                "average_setup_score": 0.0,
                "excluded_trades": 0,
                "mistake_count": 0,
            }

        winners = pnl[pnl > 0]
        losers = pnl[pnl < 0]
        wins = int(len(winners))
        losses = int(len(losers))
        total_wins = float(winners.sum())
        total_losses_abs = float(abs(losers.sum()))
        average_winner = float(winners.mean()) if wins else 0.0
        average_loser = float(losers.mean()) if losses else 0.0

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": float(wins / total_trades),
            "total_realized_pnl": float(pnl.sum()),
            "average_pnl": float(pnl.mean()),
            "average_winner": average_winner,
            "average_loser": average_loser,
            "profit_factor": float(total_wins / total_losses_abs) if total_losses_abs > 0 else (float("inf") if total_wins > 0 else 0.0),
            "expectancy": float(pnl.mean()),
            "excluded_trades": 0,
            "mistake_count": 0,
        }

    @staticmethod
    def _breakdown(df: pd.DataFrame, column: str) -> List[Dict[str, Any]]:
        if df.empty or column not in df.columns:
            return []
        if df[column].isna().all():
            return []

        rows: List[Dict[str, Any]] = []
        for value, group in df.dropna(subset=[column]).groupby(column, dropna=False):
            stats = TradeAnalytics._stats_from_frame(group)
            rows.append({column: value, **stats})

        return sorted(rows, key=lambda row: row["total_realized_pnl"], reverse=True)
