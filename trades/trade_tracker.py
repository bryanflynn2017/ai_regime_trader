"""
Manual trade tracking (journal only).

This module records trades the user has MANUALLY decided to take, based on a
detected setup. It is a JOURNAL / TRACKER ONLY:
  - It NEVER places orders or talks to any broker.
  - It NEVER sizes positions or evaluates portfolio risk.
  - It stores nothing but the plain trade fields the user supplies.

Storage is an append-only JSONL file (default logs/manual_trades.jsonl). Each
line is one trade record. Updates (e.g. closing a trade) rewrite the file with
the current state of all trades.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class TradeStatus(str, Enum):
    """Lifecycle states for a manual trade."""
    OPEN   = "OPEN"
    CLOSED = "CLOSED"


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class ManualTrade:
    """
    A single manually-recorded trade.

    All price fields are absolute levels. realized_pnl is computed on close for
    a long trade as (exit_price - entry_price) * shares. For an OPEN trade the
    exit_* and realized_pnl fields are None.
    """
    ticker:       str
    setup_type:   str
    entry_price:  float
    stop_loss:    float
    target_1:     float
    target_2:     float
    shares:       float
    entry_time:   str
    trade_id:     str            = ""
    status:       str            = TradeStatus.OPEN.value
    exit_price:   Optional[float] = None
    exit_time:    Optional[str]   = None
    realized_pnl: Optional[float] = None
    setup_alert_id: Optional[str] = None
    setup_rank: Optional[int] = None
    setup_score: Optional[float] = None
    setup_confidence: Optional[float] = None
    exclude_from_analytics: bool = False
    exclusion_reason: str = ""
    test_type: str = "forward_test"
    thesis: str = ""
    market_context: str = ""
    entry_quality: str = ""
    mistake: bool = False
    mistake_notes: str = ""
    screenshot_path: str = ""
    notes:        str            = ""
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Generate a trade_id if one was not supplied.
        if not self.trade_id:
            self.trade_id = self._generate_id()

    def _generate_id(self) -> str:
        """Build a readable id: <timestamp>-<ticker>-<short-uuid>."""
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        short = uuid.uuid4().hex[:6]
        return f"{stamp}-{self.ticker}-{short}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManualTrade":
        # Only keep keys we recognise (forward-compatible with extra fields).
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class TradeTracker:
    """
    JSONL-backed store of ManualTrade records.

    This is a journal only — it never submits orders or contacts a broker.
    """

    def __init__(self, storage_path: str | Path = "logs/manual_trades.jsonl") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    # -- internal IO --------------------------------------------------------

    def _read_all(self) -> List[ManualTrade]:
        if not self.storage_path.exists():
            return []
        trades: List[ManualTrade] = []
        with open(self.storage_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                trades.append(ManualTrade.from_dict(json.loads(line)))
        return trades

    def _write_all(self, trades: List[ManualTrade]) -> None:
        # Atomic rewrite: write a temp file in the same directory, then
        # os.replace() so a crash mid-write can never corrupt the journal.
        tmp_path = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for trade in trades:
                f.write(json.dumps(trade.to_dict()) + "\n")
        os.replace(tmp_path, self.storage_path)

    # -- public API ---------------------------------------------------------

    def add_trade(self, trade: ManualTrade) -> ManualTrade:
        """Append a new trade to the journal and return it (with id ensured)."""
        if not trade.trade_id:
            trade.trade_id = trade._generate_id()
        if trade.entry_time is None or trade.entry_time == "":
            trade.entry_time = datetime.now().isoformat()
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trade.to_dict()) + "\n")
        return trade

    def list_all_trades(self) -> List[ManualTrade]:
        """Return every trade in the journal."""
        return self._read_all()

    def list_open_trades(self) -> List[ManualTrade]:
        """Return only trades whose status is OPEN."""
        return [t for t in self._read_all() if t.status == TradeStatus.OPEN.value]

    def get_trade(self, trade_id: str) -> Optional[ManualTrade]:
        """Return the trade with the given id, or None if not found."""
        for t in self._read_all():
            if t.trade_id == trade_id:
                return t
        return None

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: Optional[str] = None,
        notes: str = "",
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> ManualTrade:
        """
        Close an open trade, compute realized P&L, and persist the change.

        For a long trade: realized_pnl = (exit_price - entry_price) * shares.
        metadata_updates (e.g. exit_signal for shadow trades) are merged into
        the trade's metadata dict.

        Raises:
            ValueError: if the trade is not found or is already closed.
        """
        trades = self._read_all()
        target: Optional[ManualTrade] = None
        for t in trades:
            if t.trade_id == trade_id:
                target = t
                break

        if target is None:
            raise ValueError(f"Trade not found: {trade_id}")
        if target.status == TradeStatus.CLOSED.value:
            raise ValueError(f"Trade already closed: {trade_id}")

        target.exit_price = float(exit_price)
        target.exit_time = exit_time or datetime.now().isoformat()
        target.realized_pnl = (target.exit_price - target.entry_price) * target.shares
        target.status = TradeStatus.CLOSED.value
        if notes:
            target.notes = (target.notes + " | " + notes).strip(" |") if target.notes else notes
        if metadata_updates:
            target.metadata.update(metadata_updates)

        self._write_all(trades)
        return target

    def edit_trade(
        self,
        trade_id: str,
        exit_price: Optional[float] = None,
        notes: str = "",
        mistake: Optional[bool] = None,
        mistake_notes: Optional[str] = None,
        exclude_from_analytics: Optional[bool] = None,
        correction_notes: str = "",
    ) -> ManualTrade:
        """
        Safely correct journal fields without deleting history.

        Exit-price edits are allowed only for CLOSED trades and recompute long
        realized P&L as (exit_price - entry_price) * shares. Correction details
        are stored in metadata so the journal remains auditable.
        """
        trades = self._read_all()
        target: Optional[ManualTrade] = None
        for trade in trades:
            if trade.trade_id == trade_id:
                target = trade
                break

        if target is None:
            raise ValueError(f"Trade not found: {trade_id}")

        corrected_fields: List[str] = []
        if exit_price is not None:
            if target.status != TradeStatus.CLOSED.value:
                raise ValueError("Exit price can only be edited for CLOSED trades.")
            target.exit_price = float(exit_price)
            target.realized_pnl = (target.exit_price - target.entry_price) * target.shares
            corrected_fields.extend(["exit_price", "realized_pnl"])

        if notes:
            target.notes = (target.notes + " | " + notes).strip(" |") if target.notes else notes
            corrected_fields.append("notes")

        if mistake is not None:
            target.mistake = bool(mistake)
            corrected_fields.append("mistake")

        if mistake_notes is not None:
            target.mistake_notes = mistake_notes
            corrected_fields.append("mistake_notes")

        if exclude_from_analytics is not None:
            target.exclude_from_analytics = bool(exclude_from_analytics)
            corrected_fields.append("exclude_from_analytics")

        if not corrected_fields and not correction_notes:
            raise ValueError("No editable fields were provided.")

        corrections = list(target.metadata.get("corrections", []))
        correction_record = {
            "corrected_at": datetime.now().isoformat(),
            "fields": corrected_fields,
        }
        if correction_notes:
            correction_record["correction_notes"] = correction_notes
        corrections.append(correction_record)
        target.metadata["corrections"] = corrections
        target.metadata["corrected_at"] = correction_record["corrected_at"]
        if correction_notes:
            target.metadata["correction_notes"] = correction_notes

        self._write_all(trades)
        return target

    def update_metadata(self, trade_id: str, updates: Dict[str, Any]) -> ManualTrade:
        """
        Merge keys into a trade's metadata dict without touching any other
        field (used e.g. for shadow-session adoption tags).

        Raises:
            ValueError: if the trade is not found or updates is empty.
        """
        if not updates:
            raise ValueError("metadata updates are required")
        trades = self._read_all()
        target: Optional[ManualTrade] = None
        for trade in trades:
            if trade.trade_id == trade_id:
                target = trade
                break

        if target is None:
            raise ValueError(f"Trade not found: {trade_id}")

        target.metadata.update(updates)
        self._write_all(trades)
        return target

    def mark_excluded(self, trade_id: str, reason: str) -> ManualTrade:
        """Exclude a trade from analytics without changing its trade outcome."""
        if not reason:
            raise ValueError("Exclusion reason is required")
        trades = self._read_all()
        target: Optional[ManualTrade] = None
        for trade in trades:
            if trade.trade_id == trade_id:
                target = trade
                break

        if target is None:
            raise ValueError(f"Trade not found: {trade_id}")

        target.exclude_from_analytics = True
        target.exclusion_reason = reason
        target.mistake_notes = (target.mistake_notes + " | " + reason).strip(" |") if target.mistake_notes else reason
        self._write_all(trades)
        return target

    def mark_included(self, trade_id: str) -> ManualTrade:
        """Include a trade in analytics again without changing its outcome."""
        trades = self._read_all()
        target: Optional[ManualTrade] = None
        for trade in trades:
            if trade.trade_id == trade_id:
                target = trade
                break

        if target is None:
            raise ValueError(f"Trade not found: {trade_id}")

        target.exclude_from_analytics = False
        target.exclusion_reason = ""
        self._write_all(trades)
        return target
