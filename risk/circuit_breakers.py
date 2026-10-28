"""
Circuit breaker layer — automatic trading halts based on drawdown thresholds.

Rules (most severe checked first):
  current_equity <= 0              → immediate lock, halt all trading
  Risk lock file already exists    → halt all trading, require manual reset
  total drawdown >= MAX_TOTAL      → create lock file, halt permanently
  daily drawdown >= MAX_DAILY_STOP → halt new positions for the day
  weekly drawdown >= MAX_WEEKLY    → halt new positions for the week
  daily drawdown >= MAX_DAILY_WARN → reduce position sizes by 50%

Drawdown formula (same for all three):
  drawdown = (reference_equity − current_equity) / reference_equity

The risk lock file requires MANUAL deletion before trading can resume.
Run `python main.py unlock-risk` for instructions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from loguru import logger  # type: ignore
except ImportError:
    logger = logging.getLogger(__name__)  # type: ignore

# Rule name constants (appear in triggered_rules and log messages)
RULE_DAILY_WARN    = "DAILY_DRAWDOWN_WARN"
RULE_DAILY_STOP    = "DAILY_DRAWDOWN_STOP"
RULE_WEEKLY_STOP   = "WEEKLY_DRAWDOWN_STOP"
RULE_TOTAL_LOCK    = "TOTAL_DRAWDOWN_LOCK"
RULE_LOCK_EXISTS   = "LOCK_FILE_EXISTS"
RULE_INVALID_INPUT = "INVALID_REFERENCE_EQUITY"
RULE_ZERO_EQUITY   = "ZERO_OR_NEGATIVE_EQUITY"


@dataclass
class CircuitBreakerStatus:
    """
    Snapshot of all circuit breaker states for a single evaluation.

    Callers must inspect all flags; no single flag tells the whole story.
    """
    trading_allowed:        bool          # False = bot must not execute any order
    new_positions_allowed:  bool          # False = no new BUY orders
    reduce_only:            bool          # True  = only exit / reduce allowed
    must_close_all:         bool          # True  = close all open positions ASAP
    lock_created:           bool          # True  = lock file was created this call
    size_multiplier:        float         # [0, 1] applied to every new position size
    reason:                 str           # Human-readable summary
    triggered_rules:        List[str]     = field(default_factory=list)
    metadata:               Dict[str, Any] = field(default_factory=dict)


class CircuitBreakerManager:
    """
    Evaluates circuit breaker rules and manages the risk lock file.

    Inject settings_override for testability (prevents writing to the real
    state/ directory during tests):

        cb = CircuitBreakerManager(settings_override=Settings(
            RISK_LOCK_FILE=str(tmp_path / "RISK_LOCKED.txt")
        ))
    """

    def __init__(self, settings_override: Any = None) -> None:
        if settings_override is None:
            from config.settings import settings
            self._s = settings
        else:
            self._s = settings_override

    # ------------------------------------------------------------------
    # Public: evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        current_equity:      float,
        starting_day_equity: float,
        starting_week_equity: float,
        peak_equity:         float,
    ) -> CircuitBreakerStatus:
        """
        Run all circuit breaker rules against the current equity snapshot.

        Rules are checked from most severe to least severe.  The first
        triggered rule determines the returned status.

        Returns:
            CircuitBreakerStatus describing what is (dis)allowed right now.
        """

        # ---- Guard: equity has collapsed to zero/negative ----
        if current_equity <= 0:
            reason = (
                f"current_equity={current_equity:.2f} is non-positive. "
                "All trading halted immediately."
            )
            self.create_risk_lock(
                reason=reason,
                metadata={"current_equity": current_equity},
            )
            logger.critical("ZERO EQUITY: {}", reason)
            return CircuitBreakerStatus(
                trading_allowed=False,
                new_positions_allowed=False,
                reduce_only=True,
                must_close_all=True,
                lock_created=True,
                size_multiplier=0.0,
                reason=reason,
                triggered_rules=[RULE_ZERO_EQUITY, RULE_TOTAL_LOCK],
                metadata={"current_equity": current_equity},
            )

        # ---- Guard: risk lock already exists from a previous run ----
        if self.risk_lock_exists():
            return CircuitBreakerStatus(
                trading_allowed=False,
                new_positions_allowed=False,
                reduce_only=True,
                must_close_all=False,
                lock_created=False,
                size_multiplier=0.0,
                reason="Risk lock file exists. Manual review required before trading.",
                triggered_rules=[RULE_LOCK_EXISTS],
                metadata={"lock_path": str(self._s.risk_lock_path())},
            )

        # ---- Compute drawdowns (handle bad reference equities safely) ----
        daily_dd, weekly_dd, total_dd = 0.0, 0.0, 0.0

        if starting_day_equity <= 0:
            logger.warning("starting_day_equity={} <= 0. Blocking new positions.", starting_day_equity)
            return CircuitBreakerStatus(
                trading_allowed=True,
                new_positions_allowed=False,
                reduce_only=True,
                must_close_all=False,
                lock_created=False,
                size_multiplier=0.0,
                reason=(
                    f"starting_day_equity={starting_day_equity} is invalid "
                    "(must be > 0). Blocking new positions as a precaution."
                ),
                triggered_rules=[RULE_INVALID_INPUT],
                metadata={
                    "starting_day_equity": starting_day_equity,
                    "current_equity": current_equity,
                },
            )

        daily_dd = (starting_day_equity - current_equity) / starting_day_equity

        if starting_week_equity > 0:
            weekly_dd = (starting_week_equity - current_equity) / starting_week_equity
        else:
            logger.warning("starting_week_equity={} <= 0. Weekly drawdown set to 0.", starting_week_equity)

        if peak_equity > 0:
            total_dd = (peak_equity - current_equity) / peak_equity
        else:
            logger.warning("peak_equity={} <= 0. Total drawdown set to 0.", peak_equity)

        meta = {
            "current_equity":        current_equity,
            "starting_day_equity":   starting_day_equity,
            "starting_week_equity":  starting_week_equity,
            "peak_equity":           peak_equity,
            "daily_drawdown":        daily_dd,
            "weekly_drawdown":       weekly_dd,
            "total_drawdown":        total_dd,
        }

        # ---- Rule: TOTAL DRAWDOWN LOCK ----
        if total_dd >= self._s.MAX_TOTAL_DRAWDOWN_LOCK:
            reason = (
                f"Total drawdown {total_dd:.1%} from peak has exceeded "
                f"MAX_TOTAL_DRAWDOWN_LOCK ({self._s.MAX_TOTAL_DRAWDOWN_LOCK:.0%}). "
                "Bot halted permanently. Manual review required."
            )
            self.create_risk_lock(reason=reason, metadata=meta)
            logger.critical("TOTAL DRAWDOWN LOCK TRIGGERED: {}", reason)
            return CircuitBreakerStatus(
                trading_allowed=False,
                new_positions_allowed=False,
                reduce_only=True,
                must_close_all=True,
                lock_created=True,
                size_multiplier=0.0,
                reason=reason,
                triggered_rules=[RULE_TOTAL_LOCK],
                metadata=meta,
            )

        # ---- Rule: DAILY DRAWDOWN STOP ----
        if daily_dd >= self._s.MAX_DAILY_DRAWDOWN_STOP:
            reason = (
                f"Daily drawdown {daily_dd:.1%} has reached the STOP threshold "
                f"({self._s.MAX_DAILY_DRAWDOWN_STOP:.0%}). "
                "No new positions for the remainder of the day."
            )
            logger.warning("DAILY DRAWDOWN STOP: {}", reason)
            return CircuitBreakerStatus(
                trading_allowed=True,
                new_positions_allowed=False,
                reduce_only=True,
                must_close_all=True,
                lock_created=False,
                size_multiplier=0.0,
                reason=reason,
                triggered_rules=[RULE_DAILY_STOP],
                metadata=meta,
            )

        # ---- Rule: WEEKLY DRAWDOWN STOP ----
        if weekly_dd >= self._s.MAX_WEEKLY_DRAWDOWN_STOP:
            reason = (
                f"Weekly drawdown {weekly_dd:.1%} has reached the STOP threshold "
                f"({self._s.MAX_WEEKLY_DRAWDOWN_STOP:.0%}). "
                "No new positions for the remainder of the week."
            )
            logger.warning("WEEKLY DRAWDOWN STOP: {}", reason)
            return CircuitBreakerStatus(
                trading_allowed=True,
                new_positions_allowed=False,
                reduce_only=True,
                must_close_all=False,
                lock_created=False,
                size_multiplier=0.0,
                reason=reason,
                triggered_rules=[RULE_WEEKLY_STOP],
                metadata=meta,
            )

        # ---- Rule: DAILY DRAWDOWN WARN ----
        if daily_dd >= self._s.MAX_DAILY_DRAWDOWN_WARN:
            reason = (
                f"Daily drawdown {daily_dd:.1%} has reached the WARNING threshold "
                f"({self._s.MAX_DAILY_DRAWDOWN_WARN:.0%}). "
                "New position sizes reduced by 50%."
            )
            logger.warning("DAILY DRAWDOWN WARN: {}", reason)
            return CircuitBreakerStatus(
                trading_allowed=True,
                new_positions_allowed=True,
                reduce_only=False,
                must_close_all=False,
                lock_created=False,
                size_multiplier=0.50,
                reason=reason,
                triggered_rules=[RULE_DAILY_WARN],
                metadata=meta,
            )

        # ---- No rules triggered ----
        return CircuitBreakerStatus(
            trading_allowed=True,
            new_positions_allowed=True,
            reduce_only=False,
            must_close_all=False,
            lock_created=False,
            size_multiplier=1.0,
            reason="All circuit breaker checks passed. Normal trading conditions.",
            triggered_rules=[],
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Public: risk lock file management
    # ------------------------------------------------------------------

    def create_risk_lock(
        self,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Write the risk lock file to disk and return its Path.

        If the file already exists it is overwritten (additional context
        appended to a running record would be useful but keeping it simple
        for Phase 5).

        This method always succeeds or raises — callers should not catch
        the exception unless they have a specific recovery plan.
        """
        path = self._s.risk_lock_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "=" * 62,
            "  RISK LOCK — AUTOMATED TRADING HALTED",
            "=" * 62,
            f"  Created : {timestamp}",
            f"  Reason  : {reason}",
            "",
        ]

        if metadata:
            lines.append("  Equity snapshot:")
            for k, v in metadata.items():
                if isinstance(v, float):
                    fmt = f"{v:.2%}" if "drawdown" in k else f"${v:,.2f}"
                    lines.append(f"    {k:<26} {fmt}")
                else:
                    lines.append(f"    {k:<26} {v}")
            lines.append("")

        lines += [
            "=" * 62,
            "  MANUAL REVIEW REQUIRED BEFORE RESUMING TRADING.",
            "  1. Check logs/ for the full event history.",
            "  2. Review all open positions in your Alpaca paper account.",
            "  3. Understand what caused this lock.",
            "  4. Run: python main.py unlock-risk",
            "  5. Delete this file only after steps 1–4 are complete.",
            "=" * 62,
        ]

        content = "\n".join(lines)
        path.write_text(content)

        logger.critical(
            "Risk lock file written to {}. Reason: {}", path, reason
        )
        return path

    def risk_lock_exists(self) -> bool:
        """Return True if the risk lock file is present on disk."""
        return self._s.risk_lock_path().exists()

    def read_risk_lock(self) -> Optional[str]:
        """Return the contents of the risk lock file, or None if it does not exist."""
        path = self._s.risk_lock_path()
        return path.read_text() if path.exists() else None
