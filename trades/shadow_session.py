"""
Automated shadow forward testing — Phase 1 foundation (journal/simulation only).

This module contains the pure decision pieces for the future
`forward-test-session` command:

  - ShadowEntryPolicy: pure rules deciding whether a detected setup would have
    been entered by a bot that follows the rules perfectly.
  - ShadowDecisionLog: append-only JSONL log of every ENTERED/SKIPPED decision.
  - create_shadow_trade_from_setup: builds a clearly-labelled simulated trade
    record for the separate shadow journal (logs/shadow_trades.jsonl).

Hard safety guarantees:
  - It NEVER places orders, contacts a broker, or imports broker modules.
  - It NEVER writes to the manual journal (logs/manual_trades.jsonl); shadow
    trades live in their own file via TradeTracker(storage_path=...).
  - Every shadow trade is labelled test_type="shadow_test" and
    entry_source="automated_forward_test" so it can never be confused with a
    manually entered forward_test trade.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as dt_time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

from trades.trade_tracker import ManualTrade, TradeTracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Shadow journal path. Manual trades stay in logs/manual_trades.jsonl —
#: shadow records are never mixed into the manual journal.
SHADOW_TRADES_FILE = "logs/shadow_trades.jsonl"

#: Append-only decision log (one record per evaluated alert).
SHADOW_DECISIONS_FILE = "logs/shadow_decisions.jsonl"

#: Append-only session history (one final record per shadow session).
SHADOW_SESSIONS_FILE = "logs/shadow_sessions.jsonl"

#: test_type value for every shadow trade record.
SHADOW_TEST_TYPE = "shadow_test"

#: entry_source metadata value for automated shadow entries.
ENTRY_SOURCE_AUTOMATED = "automated_forward_test"

#: Decision values written to the decision log.
DECISION_ENTERED = "ENTERED"
DECISION_SKIPPED = "SKIPPED"

#: Monitor actions that close a shadow trade immediately (close_full mode).
SHADOW_EXIT_ACTIONS = ("STOP_HIT", "TARGET_1_HIT", "TARGET_2_HIT", "EXIT_VWAP_LOST")

#: Session state file (recovery + observability). Atomic writes only.
SHADOW_SESSION_STATE_FILE = "state/shadow_session.json"

#: Shadow session lock file. Prevents overlapping writes to state/log files.
SHADOW_SESSION_LOCK_FILE = "state/shadow_session.lock"

#: Lock age after which a RUNNING session lock is considered stale.
SHADOW_SESSION_LOCK_STALE_SECONDS = 12 * 60 * 60

#: Shadow-only ORB diagnostics. These thresholds are reporting-only.
ORB_SETUP_TYPE = "opening_range_breakout_long"
ORB_NO_PROGRESS_THRESHOLDS = {30: 0.25, 60: 0.40}
ORB_DIAGNOSTIC_METADATA_KEY = "orb_no_progress_diagnostics"

#: Session lifecycle states.
SESSION_RUNNING = "RUNNING"
SESSION_ENDED = "ENDED"
SESSION_INTERRUPTED = "INTERRUPTED"
SESSION_ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Market-hours clock (pure ET wall-clock; no network, no broker)
# ---------------------------------------------------------------------------
# Note: weekday/time check only — US market holidays are NOT modelled. On a
# holiday the market looks "open" here, but the stale-data block and data
# failures prevent shadow entries on dead data, so this stays conservative.

_EASTERN = ZoneInfo("America/New_York")
MARKET_OPEN_ET = dt_time(9, 30)
MARKET_CLOSE_ET = dt_time(16, 0)


def now_eastern() -> datetime:
    """Current US/Eastern wall-clock time as a naive datetime."""
    return datetime.now(_EASTERN).replace(tzinfo=None)


def is_market_open_et(now_et: datetime) -> bool:
    """True during regular US equity hours (Mon–Fri 9:30–16:00 ET, no holidays)."""
    if now_et.weekday() >= 5:
        return False
    return MARKET_OPEN_ET <= now_et.time() < MARKET_CLOSE_ET


def market_close_et(now_et: datetime) -> datetime:
    """Today's 16:00 ET as a naive datetime."""
    return now_et.replace(hour=16, minute=0, second=0, microsecond=0)


def parse_end_time_et(value: str, now_et: datetime) -> datetime:
    """Parse an HH:MM end time into today's ET datetime. Raises ValueError."""
    parsed = datetime.strptime(str(value).strip(), "%H:%M").time()
    return now_et.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)


def parse_entry_cutoff_et(value: str) -> dt_time:
    """Parse an HH:MM shadow-entry cutoff as an ET wall-clock time."""
    return datetime.strptime(str(value).strip(), "%H:%M").time()


class ShadowSkipReason(str, Enum):
    """Why the entry policy refused a setup alert."""

    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NOT_TOP_RANK = "NOT_TOP_RANK"
    ALERT_TOO_OLD = "ALERT_TOO_OLD"
    STALE_DATA = "STALE_DATA"
    PARTIAL_SESSION = "PARTIAL_SESSION"
    AFTER_ENTRY_CUTOFF = "AFTER_ENTRY_CUTOFF"
    SETUP_COOLED_DOWN = "SETUP_COOLED_DOWN"
    TICKER_COOLED_DOWN = "TICKER_COOLED_DOWN"
    DUPLICATE_OPEN_TRADE = "DUPLICATE_OPEN_TRADE"
    SETUP_TYPE_NOT_ALLOWED = "SETUP_TYPE_NOT_ALLOWED"
    MAX_OPEN_TRADES = "MAX_OPEN_TRADES"
    MAX_DAILY_TRADES = "MAX_DAILY_TRADES"


# ---------------------------------------------------------------------------
# Config and decision dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ShadowEntryConfig:
    """Conservative entry rules for automated shadow forward testing."""

    min_confidence: float = 0.80
    max_rank: int = 1
    max_alert_age_seconds: float = 300.0
    #: None or empty means all setup types are allowed.
    allowed_setup_types: Optional[Sequence[str]] = None
    max_open_trades: int = 3
    max_new_trades_per_day: int = 10
    #: New entries are blocked at or after this US/Eastern wall-clock time.
    entry_cutoff_et: Optional[dt_time] = dt_time(13, 0)
    #: When False, daily cooldowns are warn-only and never block entries.
    enforce_cooldowns: bool = True


@dataclass
class ShadowEntryDecision:
    """Result of evaluating one setup alert against the entry rules."""

    decision: str  # "ENTER" or "SKIP"
    skip_reasons: List[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.decision == "ENTER"


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _setup_type_value(setup: Any) -> str:
    """Return the plain setup_type string from a DayTradeSetup or alert dict."""
    raw = getattr(setup, "setup_type", "")
    return str(getattr(raw, "value", raw) or "")


def _setup_rank(setup: Any) -> Optional[int]:
    """Return the alert rank from setup metadata, or None when absent."""
    metadata = getattr(setup, "metadata", None) or {}
    rank = metadata.get("rank")
    try:
        return int(rank)
    except (TypeError, ValueError):
        return None


def _parse_alert_time(value: Any) -> Optional[datetime]:
    """Parse a setup timestamp (pd.Timestamp / datetime / ISO str) to naive datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    # pd.Timestamp exposes to_pydatetime(); fall through to string parsing otherwise.
    to_pydatetime = getattr(value, "to_pydatetime", None)
    if callable(to_pydatetime):
        try:
            parsed = to_pydatetime()
            return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
        except (TypeError, ValueError):
            return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        return None


def alert_age_seconds(setup: Any, now: datetime) -> Optional[float]:
    """Age of a setup alert in seconds, or None when the timestamp is unusable."""
    alert_time = _parse_alert_time(getattr(setup, "timestamp", None))
    if alert_time is None:
        return None
    return (now - alert_time).total_seconds()


# ---------------------------------------------------------------------------
# Entry policy (pure — no IO, no clock, no network)
# ---------------------------------------------------------------------------

class ShadowEntryPolicy:
    """
    Pure entry rules for automated shadow forward testing.

    evaluate() never reads files, never looks at the wall clock, and never
    talks to a broker or data provider — every input is passed in, so the
    policy is deterministic and fully unit-testable.
    """

    def __init__(self, config: Optional[ShadowEntryConfig] = None) -> None:
        self.config = config or ShadowEntryConfig()

    def evaluate(
        self,
        setup: Any,
        now: datetime,
        open_shadow_trades: Sequence[ManualTrade],
        cooldown_summary: Dict[str, Any],
        stale_data: bool,
        entries_today: int,
        now_et: Optional[datetime] = None,
    ) -> ShadowEntryDecision:
        """
        Evaluate one setup alert. Returns ENTER only when every rule passes;
        otherwise SKIP with ALL failing reasons (not just the first) so the
        decision log captures the full picture.
        """
        cfg = self.config
        reasons: List[str] = []

        ticker = str(getattr(setup, "ticker", "") or "").upper()
        setup_type = _setup_type_value(setup)

        # 1. Confidence
        try:
            confidence = float(getattr(setup, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < cfg.min_confidence:
            reasons.append(ShadowSkipReason.LOW_CONFIDENCE.value)

        # 2. Rank (missing rank is treated as not auto-tradable — conservative)
        rank = _setup_rank(setup)
        if rank is None or rank > cfg.max_rank:
            reasons.append(ShadowSkipReason.NOT_TOP_RANK.value)

        # 3. Alert age (unparseable timestamp is treated as too old — conservative)
        age = alert_age_seconds(setup, now)
        if age is None or age > cfg.max_alert_age_seconds or age < 0:
            reasons.append(ShadowSkipReason.ALERT_TOO_OLD.value)

        # 4. Stale market data blocks all automated entries (no override)
        if stale_data:
            reasons.append(ShadowSkipReason.STALE_DATA.value)

        # 5. Shadow entry cutoff uses an explicitly supplied ET clock. Pure
        #    policy callers may omit it when ``now`` is already ET.
        entry_clock_et = now_et or now
        if cfg.entry_cutoff_et and entry_clock_et.time() >= cfg.entry_cutoff_et:
            reasons.append(ShadowSkipReason.AFTER_ENTRY_CUTOFF.value)

        # 6/7. Daily cooldowns (same summary shape as _daily_setup_cooldown_summary)
        if cfg.enforce_cooldowns:
            if setup_type and setup_type in (cooldown_summary.get("cooled_setup_types") or {}):
                reasons.append(ShadowSkipReason.SETUP_COOLED_DOWN.value)
            if ticker and ticker in (cooldown_summary.get("cooled_tickers") or {}):
                reasons.append(ShadowSkipReason.TICKER_COOLED_DOWN.value)

        # 8. No duplicate open shadow trade for the same ticker/setup_type.
        #    Deliberately checks SHADOW trades only — an open manual trade in the
        #    same ticker must not block the shadow comparison.
        for trade in open_shadow_trades:
            if (
                str(trade.ticker or "").upper() == ticker
                and str(trade.setup_type or "") == setup_type
            ):
                reasons.append(ShadowSkipReason.DUPLICATE_OPEN_TRADE.value)
                break

        # 9. Allowed setup types (None/empty = all allowed)
        if cfg.allowed_setup_types and setup_type not in cfg.allowed_setup_types:
            reasons.append(ShadowSkipReason.SETUP_TYPE_NOT_ALLOWED.value)

        # 10/11. Hard caps
        if len(open_shadow_trades) >= cfg.max_open_trades:
            reasons.append(ShadowSkipReason.MAX_OPEN_TRADES.value)
        if entries_today >= cfg.max_new_trades_per_day:
            reasons.append(ShadowSkipReason.MAX_DAILY_TRADES.value)

        if reasons:
            return ShadowEntryDecision(decision="SKIP", skip_reasons=reasons)
        return ShadowEntryDecision(decision="ENTER")


# ---------------------------------------------------------------------------
# Shadow trade creation
# ---------------------------------------------------------------------------

def create_shadow_trade_from_setup(
    setup: Any,
    shares: float,
    session_id: str,
    now: datetime,
    alert_id: Optional[str] = None,
) -> ManualTrade:
    """
    Build a ManualTrade record for an ACCEPTED setup, clearly labelled as a
    simulated shadow trade. The caller persists it with a TradeTracker pointed
    at SHADOW_TRADES_FILE — never the manual journal.

    The simulated fill is the alert's entry_trigger (recorded in metadata as
    fill_basis="entry_trigger"); real fills would include slippage, so shadow
    results are an optimistic bound.
    """
    metadata_in = getattr(setup, "metadata", None) or {}
    age = alert_age_seconds(setup, now)
    return ManualTrade(
        ticker=str(getattr(setup, "ticker", "") or "").upper(),
        setup_type=_setup_type_value(setup),
        entry_price=float(getattr(setup, "entry_trigger", 0.0) or 0.0),
        stop_loss=float(getattr(setup, "stop_loss", 0.0) or 0.0),
        target_1=float(getattr(setup, "target_1", 0.0) or 0.0),
        target_2=float(getattr(setup, "target_2", 0.0) or 0.0),
        shares=float(shares),
        entry_time=now.isoformat(),
        setup_alert_id=alert_id,
        setup_rank=_setup_rank(setup),
        setup_score=metadata_in.get("score"),
        setup_confidence=float(getattr(setup, "confidence", 0.0) or 0.0),
        test_type=SHADOW_TEST_TYPE,
        notes=ENTRY_SOURCE_AUTOMATED,
        metadata={
            "session_id": session_id,
            "entry_source": ENTRY_SOURCE_AUTOMATED,
            "alert_age_seconds": age,
            "simulated_fill": True,
            "fill_basis": "entry_trigger",
        },
    )


# ---------------------------------------------------------------------------
# Exit handling (Phase 2: close_full only)
# ---------------------------------------------------------------------------

def shadow_exit_price(action: Any, trade: ManualTrade, current_price: float) -> float:
    """
    Simulated exit fill for a monitor exit action (close_full mode).

    STOP_HIT / TARGET_1_HIT / TARGET_2_HIT fill at the trade's stop/target
    level when that level is usable; EXIT_VWAP_LOST (and any malformed level)
    fills at the monitor's current price.
    """
    action_value = str(getattr(action, "value", action))
    level = {
        "STOP_HIT": trade.stop_loss,
        "TARGET_1_HIT": trade.target_1,
        "TARGET_2_HIT": trade.target_2,
    }.get(action_value)
    if level is not None and float(level) > 0:
        return float(level)
    return float(current_price)


def _bars_since_entry(bars: Any, entry_time: datetime, now: datetime) -> Any:
    if bars is None or getattr(bars, "empty", True):
        return None
    index = getattr(bars, "index", None)
    if index is None:
        return None
    try:
        mask = (index >= entry_time) & (index <= now)
        return bars.loc[mask]
    except (TypeError, ValueError, KeyError):
        return None


def build_orb_no_progress_snapshot(
    trade: ManualTrade,
    monitor_result: Any,
    now: datetime,
    milestone_minutes: int,
    bars: Any = None,
) -> Dict[str, Any]:
    """
    Build one reporting-only ORB no-progress diagnostic snapshot.

    This does not change the trade, does not create an exit signal, and does
    not affect shadow strategy behavior.
    """
    current_price = float(getattr(monitor_result, "current_price", 0.0) or 0.0)
    entry = float(trade.entry_price or 0.0)
    stop = float(trade.stop_loss or 0.0)
    target_1 = float(trade.target_1 or 0.0)
    shares = float(trade.shares or 0.0)
    risk_per_share = entry - stop
    target_distance = target_1 - entry
    unrealized_pnl = (current_price - entry) * shares
    r_multiple = (current_price - entry) / risk_per_share if risk_per_share > 0 else 0.0
    progress_to_target_1 = (
        (current_price - entry) / target_distance if target_distance > 0 else 0.0
    )
    distance_remaining_to_stop_pct = (
        (current_price - stop) / risk_per_share if risk_per_share > 0 else 0.0
    )
    threshold = float(ORB_NO_PROGRESS_THRESHOLDS.get(int(milestone_minutes), 0.0))
    price_above_entry = current_price > entry
    made_meaningful_progress = bool(price_above_entry and progress_to_target_1 >= threshold)

    entry_dt = _parse_alert_time(trade.entry_time)
    window = _bars_since_entry(bars, entry_dt, now) if entry_dt is not None else None
    mfe = mae = mfe_r = mae_r = None
    reached_025r = reached_05r = None
    closed_back_below_entry = None
    vwap_loss_seen = None
    opening_range_reentry_seen = None
    if window is not None and not getattr(window, "empty", True):
        if "high" in window.columns:
            mfe = float(window["high"].max()) - entry
            mfe_r = mfe / risk_per_share if risk_per_share > 0 else 0.0
            reached_025r = mfe_r >= 0.25
            reached_05r = mfe_r >= 0.50
        if "low" in window.columns:
            mae = float(window["low"].min()) - entry
            mae_r = mae / risk_per_share if risk_per_share > 0 else 0.0
        if "close" in window.columns:
            closed_back_below_entry = bool((window["close"] < entry).any())
            if "session_vwap" in window.columns:
                vwap_loss_seen = bool((window["close"] < window["session_vwap"]).any())
            elif "vwap" in window.columns:
                vwap_loss_seen = bool((window["close"] < window["vwap"]).any())
            if "session_opening_range_high" in window.columns:
                opening_range_reentry_seen = bool(
                    (window["close"] < window["session_opening_range_high"]).any()
                )
            elif "opening_range_high" in window.columns:
                opening_range_reentry_seen = bool((window["close"] < window["opening_range_high"]).any())

    return {
        "milestone_minutes": int(milestone_minutes),
        "timestamp": now.isoformat(),
        "current_price": current_price,
        "unrealized_pnl": unrealized_pnl,
        "r_multiple": r_multiple,
        "progress_to_target_1": progress_to_target_1,
        "distance_remaining_to_stop_pct": distance_remaining_to_stop_pct,
        "price_above_entry": price_above_entry,
        "made_meaningful_progress": made_meaningful_progress,
        "no_progress": not made_meaningful_progress,
        "threshold_progress_to_target_1": threshold,
        "mfe": mfe,
        "mae": mae,
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "mfe_reached_025r": reached_025r,
        "mfe_reached_05r": reached_05r,
        "closed_back_below_entry": closed_back_below_entry,
        "vwap_loss_seen": vwap_loss_seen,
        "opening_range_reentry_seen": opening_range_reentry_seen,
    }


def record_orb_no_progress_diagnostics(
    tracker: TradeTracker,
    trade: ManualTrade,
    monitor_result: Any,
    now: datetime,
    bars: Any = None,
) -> Optional[ManualTrade]:
    """
    Persist 30/60-minute ORB diagnostic snapshots when milestones are reached.

    Returns an updated trade only when metadata was written.
    """
    if str(trade.setup_type) != ORB_SETUP_TYPE:
        return None
    if str(trade.status).upper() != "OPEN":
        return None
    entry_dt = _parse_alert_time(trade.entry_time)
    if entry_dt is None:
        return None
    elapsed_minutes = (now - entry_dt).total_seconds() / 60.0
    metadata = dict(trade.metadata or {})
    diagnostics = dict(metadata.get(ORB_DIAGNOSTIC_METADATA_KEY) or {})
    snapshots = dict(diagnostics.get("snapshots") or {})
    wrote_snapshot = False
    for milestone in sorted(ORB_NO_PROGRESS_THRESHOLDS):
        key = str(milestone)
        if elapsed_minutes >= milestone and key not in snapshots:
            snapshots[key] = build_orb_no_progress_snapshot(
                trade, monitor_result, now, milestone, bars=bars,
            )
            wrote_snapshot = True
    if not wrote_snapshot:
        return None
    diagnostics.update({
        "version": 1,
        "snapshots": snapshots,
    })
    return tracker.update_metadata(trade.trade_id, {ORB_DIAGNOSTIC_METADATA_KEY: diagnostics})


# ---------------------------------------------------------------------------
# Session statistics (pure accumulator — rendering only, no IO)
# ---------------------------------------------------------------------------

@dataclass
class ShadowSessionStats:
    """Counts for one shadow forward-test session, used for the end summary."""

    iterations_total: int = 0
    no_ranked_setup_iterations: int = 0
    ranked_setup_iterations: int = 0
    evaluated_setups: int = 0
    entered: int = 0
    skipped: int = 0
    closed: int = 0
    wins: int = 0
    realized_pnl: float = 0.0
    skip_reason_counts: Dict[str, int] = field(default_factory=dict)
    data_failures: int = 0
    discord_failures: int = 0
    data_gaps: List[Dict[str, Any]] = field(default_factory=list)
    max_iteration_gap_seconds: float = 0.0
    data_quality_failures: List[Dict[str, Any]] = field(default_factory=list)
    stale_data_events: List[Dict[str, Any]] = field(default_factory=list)
    missing_bar_observations: Optional[int] = None

    def record_iteration(self, ranked_setup_count: int) -> None:
        """Record one completed scan iteration without creating fake decisions."""
        count = max(0, int(ranked_setup_count or 0))
        self.iterations_total += 1
        self.evaluated_setups += count
        if count:
            self.ranked_setup_iterations += 1
        else:
            self.no_ranked_setup_iterations += 1

    def record_entry(self) -> None:
        self.entered += 1

    def record_skip(self, skip_reasons: Sequence[str]) -> None:
        self.skipped += 1
        for reason in skip_reasons:
            self.skip_reason_counts[reason] = self.skip_reason_counts.get(reason, 0) + 1

    def record_close(self, realized_pnl: float) -> None:
        self.closed += 1
        pnl = float(realized_pnl or 0.0)
        self.realized_pnl += pnl
        if pnl > 0:
            self.wins += 1

    def record_data_failure(self) -> None:
        self.data_failures += 1

    def record_discord_failure(self) -> None:
        self.discord_failures += 1

    def record_data_gap(
        self,
        start_timestamp: str,
        end_timestamp: str,
        elapsed_seconds: float,
        expected_interval_seconds: float,
        extreme: bool = False,
    ) -> None:
        """Record a large wall-clock gap between iterations."""
        expected = max(1.0, float(expected_interval_seconds or 1.0))
        self.max_iteration_gap_seconds = max(self.max_iteration_gap_seconds, float(elapsed_seconds))
        self.data_gaps.append({
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "elapsed_seconds": float(elapsed_seconds),
            "kind": "SESSION_SUSPENSION" if extreme else "DATA_GAP",
            "estimated_missed_intervals": max(0, int(float(elapsed_seconds) // expected) - 1),
        })
        self.record_data_quality_failure("ITERATION_GAP", end_timestamp)

    def record_data_quality_failure(self, reason: str, timestamp: str) -> None:
        """Latch the session to PARTIAL_DATA after an unrecoverable quality failure."""
        failure = {"reason": str(reason), "timestamp": str(timestamp)}
        if failure not in self.data_quality_failures:
            self.data_quality_failures.append(failure)

    def record_stale_data_event(self, timestamp: str, max_bar_age_seconds: float, freshness_snapshot=None) -> None:
        """Record one stale provider warning without changing policy behavior."""
        self.stale_data_events.append({
            "timestamp": str(timestamp),
            "max_bar_age_seconds": max(0.0, float(max_bar_age_seconds or 0.0)),
        })
        if freshness_snapshot is not None:
            self.stale_data_events[-1]["freshness_snapshot"] = freshness_snapshot

    @property
    def stale_warning_count(self) -> int:
        return len(self.stale_data_events)

    @property
    def max_stale_bar_age_seconds(self) -> float:
        return max(
            [float(event.get("max_bar_age_seconds", 0.0) or 0.0) for event in self.stale_data_events]
            or [0.0]
        )

    @property
    def first_stale_timestamp(self) -> Optional[str]:
        return str(self.stale_data_events[0].get("timestamp")) if self.stale_data_events else None

    @property
    def last_stale_timestamp(self) -> Optional[str]:
        return str(self.stale_data_events[-1].get("timestamp")) if self.stale_data_events else None

    @property
    def stale_data_caused_partial(self) -> bool:
        return any(
            str(failure.get("reason", "")).upper() == "STALE_DATA"
            for failure in self.data_quality_failures
        )

    @property
    def data_gap_count(self) -> int:
        return len(self.data_gaps)

    @property
    def largest_data_gap_seconds(self) -> float:
        if not self.data_gaps:
            return 0.0
        return max(float(gap.get("elapsed_seconds", 0.0) or 0.0) for gap in self.data_gaps)

    @property
    def has_data_gaps(self) -> bool:
        return bool(self.data_gaps)

    @property
    def data_quality(self) -> str:
        return "PARTIAL_DATA" if self.has_data_gaps or self.data_quality_failures else "CLEAN"

    def counters(self) -> Dict[str, Any]:
        """Counter snapshot persisted in the session state file."""
        return {
            **({"missing_bar_observations": self.missing_bar_observations}
               if self.missing_bar_observations is not None else {}),
            "iterations_total": self.iterations_total,
            "no_ranked_setup_iterations": self.no_ranked_setup_iterations,
            "ranked_setup_iterations": self.ranked_setup_iterations,
            "evaluated_setups": self.evaluated_setups,
            "entered": self.entered,
            "skipped": self.skipped,
            "closed": self.closed,
            "realized_pnl": self.realized_pnl,
            "data_failures": self.data_failures,
            "discord_failures": self.discord_failures,
            "data_gap_count": self.data_gap_count,
            "largest_data_gap_seconds": self.largest_data_gap_seconds,
            "has_data_gaps": self.has_data_gaps,
            "data_quality": self.data_quality,
            "data_gaps": list(self.data_gaps),
            "normal_data_gap_count": sum(g.get("kind") != "SESSION_SUSPENSION" for g in self.data_gaps),
            "extreme_data_gap_count": sum(g.get("kind") == "SESSION_SUSPENSION" for g in self.data_gaps),
            "max_iteration_gap_seconds": self.max_iteration_gap_seconds,
            "partial_data_reasons": sorted({
                "STALE_PROVIDER_DATA" if f["reason"] == "STALE_DATA" else f["reason"]
                for f in self.data_quality_failures
            }),
            "data_quality_failures": list(self.data_quality_failures),
            "stale_warning_count": self.stale_warning_count,
            "max_stale_bar_age_seconds": self.max_stale_bar_age_seconds,
            "first_stale_timestamp": self.first_stale_timestamp,
            "last_stale_timestamp": self.last_stale_timestamp,
            "stale_data_caused_partial": self.stale_data_caused_partial,
            "stale_data_events": list(self.stale_data_events),
        }

    def summary_lines(self, open_count: int) -> List[str]:
        """Plain-text end-of-session summary lines."""
        lines = [
            "Shadow Session Summary (SIMULATION ONLY — no real orders placed)",
            f"  Iterations: {self.iterations_total}",
            f"  No-ranked-setup iterations: {self.no_ranked_setup_iterations}",
            f"  Ranked-setup iterations: {self.ranked_setup_iterations}",
            f"  Evaluated setups: {self.evaluated_setups}",
            f"  Data quality: {self.data_quality}",
            f"  Data gaps: {self.data_gap_count}",
            f"  Largest data gap: {self.largest_data_gap_seconds / 60:.1f} min",
            f"  Stale warnings: {self.stale_warning_count}",
            f"  Max stale bar age: {self.max_stale_bar_age_seconds / 60:.1f} min",
            f"  Entered: {self.entered}",
            f"  Skipped: {self.skipped}",
        ]
        lines.extend(shadow_quality_diagnostic_lines(self.counters()))
        from data_providers.freshness import freshness_report_lines
        lines.extend(freshness_report_lines(self.counters()))
        if self.data_quality == "PARTIAL_DATA":
            lines.append("  PARTIAL_DATA sessions are invalid for strategy proof and excluded from clean analytics.")
        if self.first_stale_timestamp:
            lines.append(f"  First stale timestamp: {self.first_stale_timestamp}")
            lines.append(f"  Last stale timestamp: {self.last_stale_timestamp}")
            lines.append(
                "  Stale data caused PARTIAL_DATA: "
                f"{'yes' if self.stale_data_caused_partial else 'no'}"
            )
        for reason in sorted(self.skip_reason_counts):
            lines.append(f"    {reason}: {self.skip_reason_counts[reason]}")
        lines.append(f"  Closed: {self.closed}")
        lines.append(f"  Open shadow trades: {open_count}")
        if self.closed:
            lines.append(f"  Shadow realized P&L: {self.realized_pnl:+.2f}")
            lines.append(f"  Win rate: {self.wins / self.closed * 100:.1f}%")
        if self.data_failures:
            lines.append(f"  Data failures: {self.data_failures}")
        if self.discord_failures:
            lines.append(f"  Discord failures: {self.discord_failures}")
        return lines


# ---------------------------------------------------------------------------
# Decision log (append-only JSONL — never rewritten)
# ---------------------------------------------------------------------------

class ShadowDecisionLog:
    """Append-only JSONL log of every evaluated shadow entry decision."""

    def __init__(self, path: str | Path = SHADOW_DECISIONS_FILE) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        session_id: str,
        setup: Any,
        decision: ShadowEntryDecision,
        trade_id: Optional[str] = None,
        alert_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Append one decision record and return it."""
        record = {
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "session_id": session_id,
            "alert_id": alert_id,
            "ticker": str(getattr(setup, "ticker", "") or "").upper(),
            "setup_type": _setup_type_value(setup),
            "rank": _setup_rank(setup),
            "confidence": float(getattr(setup, "confidence", 0.0) or 0.0),
            "decision": DECISION_ENTERED if decision.accepted else DECISION_SKIPPED,
            "skip_reasons": list(decision.skip_reasons),
            "trade_id": trade_id,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def _read_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def count_entered_on(self, day: date) -> int:
        """Count ENTERED decisions on a calendar day (feeds the daily-trades cap)."""
        count = 0
        for record in self._read_all():
            if record.get("decision") != DECISION_ENTERED:
                continue
            parsed = _parse_alert_time(record.get("timestamp"))
            if parsed is not None and parsed.date() == day:
                count += 1
        return count

    def summarize(self, day: Optional[date] = None) -> Dict[str, Any]:
        """
        Count ENTERED/SKIPPED decisions and skip reasons (optionally for one
        calendar day). Missing log file yields all-zero counts.
        """
        return summarize_shadow_decision_records(self._read_all(), day)


def summarize_shadow_decision_records(
    records: Sequence[Dict[str, Any]], day: Optional[date] = None,
) -> Dict[str, Any]:
    """Pure ENTERED/SKIPPED + skip-reason summary over decision records."""
    entered = 0
    skipped = 0
    reasons: Dict[str, int] = {}
    for record in records:
        if day is not None:
            parsed = _parse_alert_time(record.get("timestamp"))
            if parsed is None or parsed.date() != day:
                continue
        decision = str(record.get("decision", "")).upper()
        if decision == DECISION_ENTERED:
            entered += 1
        elif decision == DECISION_SKIPPED:
            skipped += 1
        for reason in record.get("skip_reasons") or []:
            reason = str(reason)
            reasons[reason] = reasons.get(reason, 0) + 1
    sorted_reasons = dict(
        sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
    )
    return {"entered": entered, "skipped": skipped, "skip_reasons": sorted_reasons}


# ---------------------------------------------------------------------------
# Shadow analytics helpers (read-only, pure over trade lists)
# ---------------------------------------------------------------------------

def shadow_exit_reason_breakdown(trades: Sequence[ManualTrade]) -> List[Dict[str, Any]]:
    """
    Group CLOSED shadow trades by metadata.exit_signal (MANUAL when absent)
    and compute count / wins / win rate / total and average simulated P&L.
    """
    groups: Dict[str, List[float]] = {}
    for trade in trades:
        if str(trade.status).upper() != "CLOSED":
            continue
        reason = str((trade.metadata or {}).get("exit_signal") or "MANUAL")
        groups.setdefault(reason, []).append(float(trade.realized_pnl or 0.0))

    rows: List[Dict[str, Any]] = []
    for reason, pnls in groups.items():
        wins = sum(1 for pnl in pnls if pnl > 0)
        rows.append({
            "exit_reason": reason,
            "trades": len(pnls),
            "wins": wins,
            "win_rate": wins / len(pnls),
            "total_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / len(pnls),
        })
    return sorted(rows, key=lambda row: row["total_pnl"], reverse=True)


_ENTRY_TIME_BUCKETS = (
    ("before 10:00", None, dt_time(10, 0)),
    ("10:00-11:00", dt_time(10, 0), dt_time(11, 0)),
    ("11:00-12:00", dt_time(11, 0), dt_time(12, 0)),
    ("12:00-13:00", dt_time(12, 0), dt_time(13, 0)),
    ("13:00-14:00", dt_time(13, 0), dt_time(14, 0)),
    ("14:00-15:00", dt_time(14, 0), dt_time(15, 0)),
    ("after 15:00", dt_time(15, 0), None),
)


def _entry_time_bucket(value: datetime) -> str:
    trade_time = value.time()
    for label, start, end in _ENTRY_TIME_BUCKETS:
        if start is None and trade_time < end:
            return label
        if end is None and trade_time >= start:
            return label
        if start is not None and end is not None and start <= trade_time < end:
            return label
    return "after 15:00"


def _pnl_performance_row(label: str, pnls: Sequence[float]) -> Dict[str, Any]:
    wins = sum(1 for pnl in pnls if pnl > 0)
    gross_wins = sum(pnl for pnl in pnls if pnl > 0)
    gross_losses = abs(sum(pnl for pnl in pnls if pnl < 0))
    profit_factor = (
        float("inf") if gross_losses == 0 and gross_wins > 0
        else gross_wins / gross_losses if gross_losses > 0
        else 0.0
    )
    total_pnl = sum(pnls)
    trade_count = len(pnls)
    return {
        "entry_time_bucket": label,
        "trades": trade_count,
        "wins": wins,
        "win_rate": wins / trade_count if trade_count else 0.0,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / trade_count if trade_count else 0.0,
        "profit_factor": profit_factor,
    }


def shadow_entry_time_breakdown(trades: Sequence[ManualTrade]) -> List[Dict[str, Any]]:
    """
    Group CLOSED shadow trades by entry-time bucket.

    This is shadow-only analytics. It does not alter trades or affect entry/exit
    rules; SESSION_END exits are counted by their original entry time.
    """
    groups: Dict[str, List[float]] = {label: [] for label, _, _ in _ENTRY_TIME_BUCKETS}
    for trade in trades:
        if str(trade.status).upper() != "CLOSED":
            continue
        entry_dt = _parse_alert_time(trade.entry_time)
        if entry_dt is None:
            continue
        groups[_entry_time_bucket(entry_dt)].append(float(trade.realized_pnl or 0.0))
    return [_pnl_performance_row(label, groups[label]) for label, _, _ in _ENTRY_TIME_BUCKETS]


def shadow_after_15_summary(trades: Sequence[ManualTrade]) -> Dict[str, Any]:
    """Return the after-15:00 bucket summary for quick late-day review."""
    for row in shadow_entry_time_breakdown(trades):
        if row["entry_time_bucket"] == "after 15:00":
            return row
    return _pnl_performance_row("after 15:00", [])


def _time_in_trade_seconds(trade: ManualTrade) -> Optional[float]:
    entry_dt = _parse_alert_time(trade.entry_time)
    exit_dt = _parse_alert_time(trade.exit_time)
    if entry_dt is None or exit_dt is None:
        return None
    return (exit_dt - entry_dt).total_seconds()


def _performance_comparison_row(label: str, trades: Sequence[ManualTrade]) -> Dict[str, Any]:
    pnls = [float(trade.realized_pnl or 0.0) for trade in trades]
    durations = [
        duration for trade in trades
        if (duration := _time_in_trade_seconds(trade)) is not None
    ]
    row = _pnl_performance_row(label, pnls)
    row["avg_time_seconds"] = sum(durations) / len(durations) if durations else None
    return row


def _group_trade_performance(
    trades: Sequence[ManualTrade],
    key_name: str,
    key_func,
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[float]] = {}
    for trade in trades:
        key = str(key_func(trade) or "unknown")
        groups.setdefault(key, []).append(float(trade.realized_pnl or 0.0))
    rows: List[Dict[str, Any]] = []
    for key, pnls in groups.items():
        row = _pnl_performance_row(key, pnls)
        row[key_name] = row.pop("entry_time_bucket")
        rows.append(row)
    return sorted(rows, key=lambda row: row["total_pnl"])


def shadow_stop_hit_diagnostics(trades: Sequence[ManualTrade]) -> Dict[str, Any]:
    """
    Diagnose CLOSED shadow trades that exited via STOP_HIT.

    Analytics only: this helper never changes trades or shadow-session behavior.
    """
    closed = [trade for trade in trades if str(trade.status).upper() == "CLOSED"]
    stop_hits = [
        trade for trade in closed
        if str((trade.metadata or {}).get("exit_signal") or "").upper() == "STOP_HIT"
    ]
    non_stop_hits = [
        trade for trade in closed
        if str((trade.metadata or {}).get("exit_signal") or "").upper() != "STOP_HIT"
    ]

    largest_losses: List[Dict[str, Any]] = []
    for trade in sorted(stop_hits, key=lambda item: float(item.realized_pnl or 0.0))[:5]:
        largest_losses.append({
            "ticker": trade.ticker,
            "setup_type": trade.setup_type,
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "realized_pnl": float(trade.realized_pnl or 0.0),
            "time_in_trade_seconds": _time_in_trade_seconds(trade),
        })

    return {
        "stop_hit": _performance_comparison_row("STOP_HIT", stop_hits),
        "non_stop_hit": _performance_comparison_row("non-STOP_HIT", non_stop_hits),
        "by_setup_type": _group_trade_performance(
            stop_hits, "setup_type", lambda trade: trade.setup_type,
        ),
        "by_ticker": _group_trade_performance(
            stop_hits, "ticker", lambda trade: trade.ticker,
        ),
        "by_entry_time": [
            row for row in shadow_entry_time_breakdown(stop_hits)
            if row["trades"] > 0
        ],
        "largest_losses": largest_losses,
    }


def shadow_orb_no_progress_diagnostics(trades: Sequence[ManualTrade]) -> Dict[str, Any]:
    """
    Summarize persisted ORB no-progress snapshots for closed shadow trades.

    Candidate impact is hypothetical analytics only: it compares actual P&L
    with the snapshot unrealized P&L for trades flagged no-progress.
    """
    orb_closed = [
        trade for trade in trades
        if str(trade.status).upper() == "CLOSED" and str(trade.setup_type) == ORB_SETUP_TYPE
    ]
    milestones: Dict[str, Dict[str, Any]] = {}
    for milestone in sorted(ORB_NO_PROGRESS_THRESHOLDS):
        key = str(milestone)
        with_snapshot: List[ManualTrade] = []
        flagged: List[tuple[ManualTrade, Dict[str, Any]]] = []
        for trade in orb_closed:
            diagnostics = (trade.metadata or {}).get(ORB_DIAGNOSTIC_METADATA_KEY) or {}
            snapshot = (diagnostics.get("snapshots") or {}).get(key)
            if not isinstance(snapshot, dict):
                continue
            with_snapshot.append(trade)
            if bool(snapshot.get("no_progress")):
                flagged.append((trade, snapshot))

        actual_pnl = sum(float(trade.realized_pnl or 0.0) for trade, _ in flagged)
        hypothetical_pnl = sum(float(snapshot.get("unrealized_pnl") or 0.0) for _, snapshot in flagged)
        milestones[key] = {
            "milestone_minutes": milestone,
            "snapshot_count": len(with_snapshot),
            "no_progress_count": len(flagged),
            "no_progress_actual_pnl": actual_pnl,
            "no_progress_hypothetical_pnl": hypothetical_pnl,
            "candidate_pnl_delta": hypothetical_pnl - actual_pnl,
            "eventual_stop_hit_flagged": sum(
                1 for trade, _ in flagged
                if str((trade.metadata or {}).get("exit_signal") or "").upper() == "STOP_HIT"
            ),
            "eventual_target_1_hit_flagged": sum(
                1 for trade, _ in flagged
                if str((trade.metadata or {}).get("exit_signal") or "").upper() == "TARGET_1_HIT"
            ),
        }

    return {
        "orb_closed_trades": len(orb_closed),
        "milestones": milestones,
    }


def average_time_in_trade_seconds(trades: Sequence[ManualTrade]) -> Optional[float]:
    """Mean (exit_time - entry_time) over CLOSED trades with parseable times."""
    durations: List[float] = []
    for trade in trades:
        if str(trade.status).upper() != "CLOSED":
            continue
        entry_dt = _parse_alert_time(trade.entry_time)
        exit_dt = _parse_alert_time(trade.exit_time)
        if entry_dt is not None and exit_dt is not None:
            durations.append((exit_dt - entry_dt).total_seconds())
    if not durations:
        return None
    return sum(durations) / len(durations)


# ---------------------------------------------------------------------------
# Session state (state/shadow_session.json — recovery + observability)
# ---------------------------------------------------------------------------

@dataclass
class ShadowSessionState:
    """Persisted shadow-session state. Never stores secrets."""

    session_id: str
    started_at: str
    status: str = SESSION_RUNNING
    config: Dict[str, Any] = field(default_factory=dict)
    open_trade_ids: List[str] = field(default_factory=list)
    counters: Dict[str, Any] = field(default_factory=dict)
    last_iteration_at: Optional[str] = None

    def save(self, path: str | Path = SHADOW_SESSION_STATE_FILE) -> None:
        """Atomically persist session state (temp file + os.replace)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        os.replace(tmp_path, target)

    @classmethod
    def load(cls, path: str | Path = SHADOW_SESSION_STATE_FILE) -> Optional["ShadowSessionState"]:
        """Load prior session state, or None when missing/corrupt."""
        target = Path(path)
        if not target.exists():
            return None
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None
            if not all(isinstance(data.get(key), str) and data[key] for key in ("session_id", "started_at", "status")):
                return None
            if not isinstance(data.get("open_trade_ids", []), list) or not all(
                isinstance(item, str) for item in data.get("open_trade_ids", [])
            ):
                return None
            if not all(isinstance(data.get(key, {}), dict) for key in ("config", "counters")):
                return None
            known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            return cls(**{k: v for k, v in data.items() if k in known})
        except (json.JSONDecodeError, TypeError, OSError):
            return None


def append_shadow_session_history(
    state: ShadowSessionState,
    summary_lines: Optional[Sequence[str]] = None,
    path: str | Path = SHADOW_SESSIONS_FILE,
) -> Dict[str, Any]:
    """Append one final shadow-session state record for daily aggregation."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = asdict(state)
    record["timestamp"] = datetime.now().isoformat()
    record["summary_lines"] = list(summary_lines or [])
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def _parse_lock_timestamp(value: Any) -> Optional[datetime]:
    """Parse a lock timestamp; invalid values are treated as unknown."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone().replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        return None


def load_shadow_session_lock(path: str | Path = SHADOW_SESSION_LOCK_FILE) -> Optional[Dict[str, Any]]:
    """Load the shadow-session lock, or None when missing/corrupt."""
    target = Path(path)
    if not target.exists():
        return None
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError, TypeError):
        return None


@contextmanager
def _shadow_lock_guard(lock_path: str | Path):
    """Serialize lock reclamation/removal, including across processes.

    The sidecar is deliberately retained: its OS lock is released on close or
    process death, so an old sidecar never represents a running session.
    """
    guard = Path(str(lock_path) + ".guard")
    guard.parent.mkdir(parents=True, exist_ok=True)
    with open(guard, "a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if guard.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def acquire_shadow_session_lock(
    session_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    lock_path: str | Path = SHADOW_SESSION_LOCK_FILE,
    state_path: str | Path = SHADOW_SESSION_STATE_FILE,
    stale_after_seconds: float = SHADOW_SESSION_LOCK_STALE_SECONDS,
    now: Optional[datetime] = None,
) -> tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Create the shadow-session lock unless an active RUNNING session owns it.

    Returns (acquired, existing_or_new_lock, reason). Stale, ended,
    interrupted, and error locks are cleared before the new lock is written.
    """
    with _shadow_lock_guard(lock_path):
        current_time = now or datetime.now()
        if current_time.tzinfo is not None:
            current_time = current_time.astimezone().replace(tzinfo=None)
        target = Path(lock_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        existing = load_shadow_session_lock(target)
        if existing is not None:
            state = ShadowSessionState.load(state_path)
            started_at = _parse_lock_timestamp(existing.get("started_at"))
            if started_at is None and state is not None:
                started_at = _parse_lock_timestamp(state.started_at)
            lock_age = (current_time - started_at).total_seconds() if started_at else None
            is_stale = lock_age is not None and lock_age > stale_after_seconds
            matching_state = state is not None and state.session_id == existing.get("session_id")
            if not is_stale and (not matching_state or state.status == SESSION_RUNNING):
                return False, existing, "active"
            try:
                target.unlink()
            except FileNotFoundError:
                pass

        record = {
            "session_id": session_id,
            "started_at": current_time.isoformat(),
            "pid": os.getpid() if hasattr(os, "getpid") else None,
            "metadata": metadata or {},
        }
        created = False
        try:
            # Exclusive creation also protects the startup window before state.save().
            with open(target, "x", encoding="utf-8") as f:
                created = True
                json.dump(record, f, indent=2)
        except FileExistsError:
            return False, load_shadow_session_lock(target), "active"
        except BaseException:
            if created:
                target.unlink(missing_ok=True)
            raise
        return True, record, "created"


def release_shadow_session_lock(
    session_id: str,
    lock_path: str | Path = SHADOW_SESSION_LOCK_FILE,
) -> bool:
    """Remove this session's lock without deleting a different active lock."""
    with _shadow_lock_guard(lock_path):
        target = Path(lock_path)
        existing = load_shadow_session_lock(target)
        if existing is None:
            return False
        if str(existing.get("session_id") or "") != str(session_id):
            return False
        try:
            target.unlink()
            return True
        except FileNotFoundError:
            return False


# ---------------------------------------------------------------------------
# Resume: adopt prior-session open shadow trades (never force-closes)
# ---------------------------------------------------------------------------

def adopt_open_shadow_trades(
    tracker: TradeTracker, session_id: str,
    prior_state: Optional[ShadowSessionState] = None,
    destination_state: Optional[ShadowSessionState] = None,
) -> List[ManualTrade]:
    """
    Tag open shadow trades from earlier sessions with adopted_from_session so
    this session monitors them. Trades stay OPEN — adoption never closes,
    re-prices, or otherwise alters the trade outcome fields.
    """
    adopted: List[ManualTrade] = []
    prior_state = prior_state or ShadowSessionState.load()
    if prior_state is None or prior_state.status not in {SESSION_RUNNING, SESSION_INTERRUPTED, SESSION_ERROR}:
        return adopted
    # Final history is authoritative even if an older state snapshot survived.
    ended_sessions = set()
    history = Path(SHADOW_SESSIONS_FILE)
    if history.exists():
        for line in history.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(record, dict) and str(record.get("status", "")).upper() == SESSION_ENDED:
                ended_sessions.add(record.get("session_id"))
    if prior_state.session_id in ended_sessions:
        return adopted
    pending_owners = prior_state.config.get("recovery_trade_owners") or {}
    candidates = []
    for trade in tracker.list_open_trades():
        metadata = trade.metadata or {}
        origin = str(metadata.get("session_id") or "")
        owner = str(metadata.get("monitoring_session_id") or origin)
        pending_owner = pending_owners.get(trade.trade_id) if isinstance(pending_owners, dict) else None
        # A journal entry may precede its next state snapshot. Its explicit
        # unfinished owner is sufficient; cross-session recovery needs saved intent.
        recoverable = owner == prior_state.session_id or (
            trade.trade_id in prior_state.open_trade_ids and owner == pending_owner
        )
        if origin == session_id or not recoverable or origin in ended_sessions or owner in ended_sessions:
            continue
        candidates.append((trade, owner))
    if destination_state is not None and candidates:
        # Write recovery intent BEFORE journal metadata. A crash or failed write
        # midway through adoption leaves every candidate discoverable on retry.
        destination_state.open_trade_ids = [t.trade_id for t, _ in candidates]
        destination_state.counters["open"] = len(candidates)
        destination_state.config["recovery_trade_owners"] = {t.trade_id: owner for t, owner in candidates}
        destination_state.save()
    for trade, owner in candidates:
        adopted.append(tracker.update_metadata(
            trade.trade_id, {"adopted_from_session": owner, "monitoring_session_id": session_id},
        ))
    return adopted


def shadow_quality_diagnostics(counters: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize diagnostic metadata only; never reclassify legacy sessions."""
    reasons = set(counters.get("partial_data_reasons") or [])
    reasons.update(f.get("reason") for f in (counters.get("data_quality_failures") or []) if isinstance(f, dict) and f.get("reason"))
    if counters.get("stale_data_caused_partial"):
        reasons.add("STALE_PROVIDER_DATA")
    if counters.get("data_gap_count"):
        reasons.add("ITERATION_GAP")
    reasons = {"STALE_PROVIDER_DATA" if r == "STALE_DATA" else r for r in reasons}
    if not reasons and counters.get("data_quality") == "PARTIAL_DATA":
        reasons.add("UNKNOWN_LEGACY_REASON")
    extreme = int(counters.get("extreme_data_gap_count", 0) or 0)
    return {
        "partial_data_reasons": sorted(reasons),
        "normal_data_gap_count": int(counters.get("normal_data_gap_count", max(0, int(counters.get("data_gap_count", 0) or 0) - extreme)) or 0),
        "extreme_data_gap_count": extreme,
        "max_iteration_gap_seconds": float(counters.get("max_iteration_gap_seconds", counters.get("largest_data_gap_seconds", 0.0)) or 0.0),
    }


def shadow_quality_diagnostic_lines(counters: Dict[str, Any]) -> List[str]:
    detail = shadow_quality_diagnostics(counters)
    reasons = detail["partial_data_reasons"]
    label = "MULTIPLE_REASONS (" + ", ".join(reasons) + ")" if len(reasons) > 1 else (reasons[0] if reasons else "none")
    return [
        f"  Partial-data reason: {label}",
        f"  Normal data gaps: {detail['normal_data_gap_count']}",
        f"  Extreme/session-suspension gaps: {detail['extreme_data_gap_count']}",
        f"  Max iteration gap: {detail['max_iteration_gap_seconds'] / 60:.1f} min",
    ]
