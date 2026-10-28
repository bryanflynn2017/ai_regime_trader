"""
Day-trade setup detection (skeleton).

This module is a DETECTOR ONLY. It identifies basic long watchlist setups from
already-computed technical features (VWAP, relative volume, opening range). It
does NOT place orders, size positions, evaluate portfolio risk, or talk to any
broker. Every setup it returns is a candidate for human review or for a future
execution strategy — never an order.

Setups recognised (all long-only for now):
  - OPENING_RANGE_BREAKOUT_LONG
  - VWAP_RECLAIM_LONG
  - HIGH_RELATIVE_VOLUME_CONTINUATION_LONG

If nothing qualifies, a NO_SETUP result is returned with is_valid=False.

Safety properties:
  - The input DataFrame is never mutated (a copy is taken).
  - Features are computed via add_features() only if required columns are absent.
  - A stop that is not strictly below the entry invalidates the setup and is
    reported in `warnings` rather than producing a nonsensical risk/reward.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from data.feature_engineering import add_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Setup taxonomy
# ---------------------------------------------------------------------------

class SetupType(str, Enum):
    """The kinds of day-trade setups this detector can recognise."""
    OPENING_RANGE_BREAKOUT_LONG            = "opening_range_breakout_long"
    VWAP_RECLAIM_LONG                      = "vwap_reclaim_long"
    HIGH_RELATIVE_VOLUME_CONTINUATION_LONG = "high_relative_volume_continuation_long"
    NO_SETUP                              = "no_setup"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DayTradeSetup:
    """
    A detected (or rejected) day-trade setup for a single ticker at one instant.

    All price fields are absolute levels. risk_reward_* are multiples of the
    per-share risk (entry - stop). For a NO_SETUP or an invalidated setup,
    numeric fields are 0.0 and is_valid is False.
    """
    ticker:        str
    timestamp:     Any                     # pd.Timestamp / datetime
    setup_type:    SetupType
    is_valid:      bool
    confidence:    float                   # 0.0 – 1.0
    entry_trigger: float
    stop_loss:     float
    target_1:      float
    target_2:      float
    risk_reward_1: float
    risk_reward_2: float
    reason:        str
    warnings:      List[str]        = field(default_factory=list)
    metadata:      Dict[str, Any]   = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Required feature columns
# ---------------------------------------------------------------------------

#: Columns the detector reads. If any are absent, add_features() is invoked.
_REQUIRED_COLUMNS: List[str] = [
    "close",
    "vwap",
    "rel_volume_20",
    "trend_score",
    "broke_above_opening_range",
    "opening_range_mid",
    "opening_range_high",
    "opening_range_low",
    "close_above_vwap",
    "rsi_14",
    "unusual_volume",
]


# ---------------------------------------------------------------------------
# Pure helpers (module-level for easy unit testing)
# ---------------------------------------------------------------------------

def compute_risk_reward(entry: float, stop: float, target: float) -> float:
    """
    Reward/risk multiple = (target - entry) / (entry - stop).

    Returns 0.0 when the risk (entry - stop) is zero or negative, so a
    malformed stop can never produce a misleading ratio or a divide-by-zero.
    """
    risk = entry - stop
    if risk <= 0:
        return 0.0
    return (target - entry) / risk


def _score_confidence(
    rel_volume_20: float,
    close: float,
    vwap: float,
    trend_score: float,
    unusual_volume: float,
) -> float:
    """
    Simple additive confidence in [0, 1].

    base 0.50 for a valid setup, plus 0.10 for each of:
      - rel_volume_20 >= 2.0
      - close > vwap
      - trend_score > 0.5
      - unusual_volume == 1.0
    """
    conf = 0.50
    if rel_volume_20 >= 2.0:
        conf += 0.10
    if close > vwap:
        conf += 0.10
    if trend_score > 0.5:
        conf += 0.10
    if unusual_volume == 1.0:
        conf += 0.10
    return min(conf, 1.0)


def _f(row: pd.Series, col: str) -> float:
    """Read a column from a row as float; NaN if missing or unconvertible."""
    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class DayTradeSetupDetector:
    """
    Detect basic long day-trade setups from feature-engineered OHLCV data.

    Usage:
        detector = DayTradeSetupDetector()
        setup = detector.detect_latest("SPY", df)
        if setup.is_valid:
            ...  # candidate only — never auto-executed here
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_latest(self, ticker: str, df: pd.DataFrame) -> DayTradeSetup:
        """
        Detect the best qualifying setup on the most recent bar.

        The input DataFrame is copied and never mutated. Features are added
        automatically when the required columns are missing. Returns a
        NO_SETUP result (is_valid=False) when nothing qualifies or the data
        is insufficient.
        """
        if df is None or len(df) == 0:
            return self._no_setup(ticker, None, "Empty or missing DataFrame.")

        # Never mutate the caller's frame.
        work = df.copy()

        try:
            work = self._ensure_features(work)
        except Exception as exc:  # add_features validation, etc.
            logger.debug("Feature computation failed for %s: %s", ticker, exc)
            return self._no_setup(ticker, _safe_timestamp(work), f"Feature error: {exc}")

        if len(work) < 2:
            return self._no_setup(
                ticker, _safe_timestamp(work),
                f"Insufficient rows: {len(work)} (need >= 2).",
            )

        last = work.iloc[-1]
        prev = work.iloc[-2]
        ts   = _safe_timestamp(work)

        # Bail out early if the core feature values are not numeric yet
        # (e.g. very short history where rolling features are still NaN).
        if not self._core_features_ready(last):
            return self._no_setup(
                ticker, ts,
                "Required feature values are not available (NaN).",
            )

        # Evaluate setups in priority order; first qualifying wins.
        # Track which setup came closest for diagnostics.
        nearest_setup_type = None
        max_passed = -1
        all_diagnostics = {}

        # Evaluate each setup type for diagnostics
        diag_orb = self._diagnose_opening_range_breakout(last, prev, work)
        diag_vwap = self._diagnose_vwap_reclaim(last, prev, work)
        diag_hrvc = self._diagnose_high_rel_volume_continuation(last, prev, work)

        all_diagnostics[SetupType.OPENING_RANGE_BREAKOUT_LONG.value] = diag_orb[2]
        all_diagnostics[SetupType.VWAP_RECLAIM_LONG.value] = diag_vwap[2]
        all_diagnostics[SetupType.HIGH_RELATIVE_VOLUME_CONTINUATION_LONG.value] = diag_hrvc[2]

        # Track the setup that came closest (most conditions passed)
        for setup_type, diag_tuple in [
            (SetupType.OPENING_RANGE_BREAKOUT_LONG, diag_orb),
            (SetupType.VWAP_RECLAIM_LONG, diag_vwap),
            (SetupType.HIGH_RELATIVE_VOLUME_CONTINUATION_LONG, diag_hrvc),
        ]:
            if diag_tuple[0] > max_passed:
                max_passed = diag_tuple[0]
                nearest_setup_type = setup_type

        # Try actual setup builders in priority order
        for builder in (
            self._try_opening_range_breakout,
            self._try_vwap_reclaim,
            self._try_high_rel_volume_continuation,
        ):
            setup = builder(ticker, ts, last, prev, work)
            if setup is not None:
                return setup

        return self._no_setup(
            ticker, ts, "No qualifying day-trade setup.",
            diagnostics=all_diagnostics,
            nearest_setup_type=nearest_setup_type,
            passed_count=max_passed if max_passed >= 0 else 0,
            extra_metadata={
                "close": _f(last, "close"),
                "vwap_value": _f(last, self.get_vwap_column(work)),
                "close_above_vwap": _f(last, self.get_close_above_vwap_column(work)),
            },
        )

    def detect_many(
        self,
        data_by_ticker: Dict[str, pd.DataFrame],
    ) -> List[DayTradeSetup]:
        """
        Run detect_latest() for every ticker. Returns exactly one DayTradeSetup
        per input ticker, in input order.
        """
        results: List[DayTradeSetup] = []
        for ticker, df in data_by_ticker.items():
            results.append(self.detect_latest(ticker, df))
        return results

    # ------------------------------------------------------------------
    # Feature-column resolvers (session-aware vs generic)
    # ------------------------------------------------------------------
    #
    # Intraday scans should use the per-session columns produced by
    # add_session_intraday_features() (session_vwap, session_opening_range_*),
    # while daily/full-series scans keep using the generic columns. Each
    # resolver returns the COLUMN NAME the rules should read for the latest bar.
    #
    # A session column is preferred only when the latest bar belongs to a
    # multi-bar session (i.e. genuinely intraday) AND the session value is
    # present (non-NaN). For daily data every date is a one-row session, so the
    # gate is False and the generic columns are always used — preserving daily
    # behavior even though session_vwap is technically non-null on daily bars.

    def _session_features_active(self, df: pd.DataFrame) -> bool:
        """True when the latest bar sits in a multi-bar (intraday) session."""
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex) or len(idx) == 0:
            return False
        try:
            last_date = idx[-1].normalize()
            return int((idx.normalize() == last_date).sum()) > 1
        except Exception:
            return False

    def _resolve_column(self, df: pd.DataFrame, session_col: str, generic_col: str) -> str:
        """Return session_col when it is active and non-null on the latest bar."""
        if self._session_features_active(df) and session_col in df.columns:
            if not np.isnan(_f(df.iloc[-1], session_col)):
                return session_col
        return generic_col

    def get_vwap_column(self, df: pd.DataFrame) -> str:
        return self._resolve_column(df, "session_vwap", "vwap")

    def get_close_above_vwap_column(self, df: pd.DataFrame) -> str:
        return self._resolve_column(df, "close_above_session_vwap", "close_above_vwap")

    def get_distance_from_vwap_column(self, df: pd.DataFrame) -> str:
        return self._resolve_column(
            df, "distance_from_session_vwap_pct", "distance_from_vwap_pct",
        )

    def get_opening_range_high_column(self, df: pd.DataFrame) -> str:
        return self._resolve_column(
            df, "session_opening_range_high", "opening_range_high",
        )

    def get_opening_range_low_column(self, df: pd.DataFrame) -> str:
        return self._resolve_column(
            df, "session_opening_range_low", "opening_range_low",
        )

    def get_opening_range_mid_column(self, df: pd.DataFrame) -> str:
        return self._resolve_column(
            df, "session_opening_range_mid", "opening_range_mid",
        )

    def get_broke_above_opening_range_column(self, df: pd.DataFrame) -> str:
        # Tie the breakout flag to the range source so the opening-range family
        # is always read coherently from one source.
        if self.get_opening_range_high_column(df).startswith("session_"):
            return "broke_above_session_opening_range"
        return "broke_above_opening_range"

    def get_broke_below_opening_range_column(self, df: pd.DataFrame) -> str:
        if self.get_opening_range_high_column(df).startswith("session_"):
            return "broke_below_session_opening_range"
        return "broke_below_opening_range"

    # ------------------------------------------------------------------
    # Setup rules
    # ------------------------------------------------------------------

    def _try_opening_range_breakout(
        self, ticker: str, ts: Any, last: pd.Series, prev: pd.Series, df: pd.DataFrame,
    ) -> Optional[DayTradeSetup]:
        """
        Opening range breakout long:
          broke_above_opening_range == 1.0
          close > vwap
          rel_volume_20 >= 1.5
          trend_score > 0
        Stop = lower of {opening_range_mid, vwap} (naturally below entry).

        Uses session_opening_range_* / broke_above_session_opening_range and
        session_vwap when intraday session features are available; otherwise the
        generic full-series columns.
        """
        vwap_col       = self.get_vwap_column(df)
        broke_col      = self.get_broke_above_opening_range_column(df)
        or_mid_col     = self.get_opening_range_mid_column(df)

        close   = _f(last, "close")
        vwap    = _f(last, vwap_col)
        rel_vol = _f(last, "rel_volume_20")
        trend   = _f(last, "trend_score")
        broke   = _f(last, broke_col)
        or_mid  = _f(last, or_mid_col)

        qualifies = (
            broke == 1.0
            and close > vwap
            and rel_vol >= 1.5
            and trend > 0.0
        )
        if not qualifies:
            return None

        # Lower of the two candidate stops. Because close > vwap, vwap < entry,
        # so this is always below entry when the rules above hold.
        stop_candidates = [v for v in (or_mid, vwap) if not np.isnan(v)]
        stop = min(stop_candidates) if stop_candidates else np.nan

        return self._build_long_setup(
            ticker, ts, SetupType.OPENING_RANGE_BREAKOUT_LONG,
            entry=close, stop=stop, last=last,
            reason=(
                f"Opening-range breakout: close {close:.2f} > VWAP {vwap:.2f}, "
                f"rel_vol {rel_vol:.2f}, trend {trend:.2f}."
            ),
            df=df,
        )

    def _try_vwap_reclaim(
        self, ticker: str, ts: Any, last: pd.Series, prev: pd.Series, df: pd.DataFrame,
    ) -> Optional[DayTradeSetup]:
        """
        VWAP reclaim long:
          previous close <= previous vwap
          current close  > current vwap
          rel_volume_20 >= 1.2
          trend_score >= 0
        Stop = vwap * 0.995.

        Uses session_vwap (current and previous bar) when intraday session
        features are available; otherwise the generic full-series vwap.
        """
        vwap_col   = self.get_vwap_column(df)

        close      = _f(last, "close")
        vwap       = _f(last, vwap_col)
        rel_vol    = _f(last, "rel_volume_20")
        trend      = _f(last, "trend_score")
        prev_close = _f(prev, "close")
        prev_vwap  = _f(prev, vwap_col)

        if np.isnan(prev_close) or np.isnan(prev_vwap):
            return None

        qualifies = (
            prev_close <= prev_vwap
            and close > vwap
            and rel_vol >= 1.2
            and trend >= 0.0
        )
        if not qualifies:
            return None

        stop = vwap * 0.995

        return self._build_long_setup(
            ticker, ts, SetupType.VWAP_RECLAIM_LONG,
            entry=close, stop=stop, last=last,
            reason=(
                f"VWAP reclaim: prev close {prev_close:.2f} <= prev VWAP "
                f"{prev_vwap:.2f}, now {close:.2f} > {vwap:.2f}, "
                f"rel_vol {rel_vol:.2f}."
            ),
            df=df,
        )

    def _try_high_rel_volume_continuation(
        self, ticker: str, ts: Any, last: pd.Series, prev: pd.Series, df: pd.DataFrame,
    ) -> Optional[DayTradeSetup]:
        """
        High relative-volume continuation long:
          close > vwap
          rel_volume_20 >= 2.0
          close_above_vwap == 1.0
          trend_score > 0
          rsi_14 <= 75
        Stop = max(vwap, close * 0.98) (both below entry since close > vwap).

        Uses session_vwap / close_above_session_vwap when intraday session
        features are available; otherwise the generic full-series columns.
        """
        vwap_col        = self.get_vwap_column(df)
        close_above_col = self.get_close_above_vwap_column(df)

        close     = _f(last, "close")
        vwap      = _f(last, vwap_col)
        rel_vol   = _f(last, "rel_volume_20")
        trend     = _f(last, "trend_score")
        above     = _f(last, close_above_col)
        rsi       = _f(last, "rsi_14")

        qualifies = (
            close > vwap
            and rel_vol >= 2.0
            and above == 1.0
            and trend > 0.0
            and not np.isnan(rsi)
            and rsi <= 75.0
        )
        if not qualifies:
            return None

        stop = max(vwap, close * 0.98)

        return self._build_long_setup(
            ticker, ts, SetupType.HIGH_RELATIVE_VOLUME_CONTINUATION_LONG,
            entry=close, stop=stop, last=last,
            reason=(
                f"High rel-volume continuation: close {close:.2f} > VWAP "
                f"{vwap:.2f}, rel_vol {rel_vol:.2f}, RSI {rsi:.1f}."
            ),
            df=df,
        )

    # ------------------------------------------------------------------
    # Diagnostic helpers (near-miss analysis)
    # ------------------------------------------------------------------

    def _diagnose_opening_range_breakout(
        self, last: pd.Series, prev: pd.Series, df: pd.DataFrame,
    ) -> tuple[int, int, List[str]]:
        """
        Evaluate opening-range breakout conditions. Returns
        (passed_count, failed_count, failed_reasons).
        """
        failed = []
        vwap_col = self.get_vwap_column(df)
        broke_col = self.get_broke_above_opening_range_column(df)
        or_mid_col = self.get_opening_range_mid_column(df)

        close   = _f(last, "close")
        vwap    = _f(last, vwap_col)
        rel_vol = _f(last, "rel_volume_20")
        trend   = _f(last, "trend_score")
        broke   = _f(last, broke_col)

        if broke != 1.0:
            failed.append("broke_above_opening_range != 1.0")
        if not (close > vwap):
            failed.append(f"close {close:.2f} not above vwap {vwap:.2f}")
        if not (rel_vol >= 1.5):
            failed.append(f"rel_volume_20 {rel_vol:.2f} < 1.5")
        if not (trend > 0.0):
            failed.append(f"trend_score {trend:.2f} <= 0")

        return 4 - len(failed), len(failed), failed

    def _diagnose_vwap_reclaim(
        self, last: pd.Series, prev: pd.Series, df: pd.DataFrame,
    ) -> tuple[int, int, List[str]]:
        """
        Evaluate VWAP reclaim conditions. Returns
        (passed_count, failed_count, failed_reasons).
        """
        failed = []
        vwap_col = self.get_vwap_column(df)

        close      = _f(last, "close")
        vwap       = _f(last, vwap_col)
        rel_vol    = _f(last, "rel_volume_20")
        trend      = _f(last, "trend_score")
        prev_close = _f(prev, "close")
        prev_vwap  = _f(prev, vwap_col)

        if not (prev_close <= prev_vwap):
            failed.append(f"prev_close {prev_close:.2f} > prev_vwap {prev_vwap:.2f}")
        if not (close > vwap):
            failed.append(f"close {close:.2f} not above vwap {vwap:.2f}")
        if not (rel_vol >= 1.2):
            failed.append(f"rel_volume_20 {rel_vol:.2f} < 1.2")
        if not (trend >= 0.0):
            failed.append(f"trend_score {trend:.2f} < 0")

        return 4 - len(failed), len(failed), failed

    def _diagnose_high_rel_volume_continuation(
        self, last: pd.Series, prev: pd.Series, df: pd.DataFrame,
    ) -> tuple[int, int, List[str]]:
        """
        Evaluate high-rel-volume continuation conditions. Returns
        (passed_count, failed_count, failed_reasons).
        """
        failed = []
        vwap_col = self.get_vwap_column(df)
        close_above_col = self.get_close_above_vwap_column(df)

        close     = _f(last, "close")
        vwap      = _f(last, vwap_col)
        rel_vol   = _f(last, "rel_volume_20")
        trend     = _f(last, "trend_score")
        above     = _f(last, close_above_col)
        rsi       = _f(last, "rsi_14")
        prev_close = _f(prev, "close")
        prev_vwap  = _f(prev, vwap_col)

        # Exclude if prev close already > prev vwap (VWAP reclaim takes priority)
        if prev_close > prev_vwap:
            failed.append("prev_close > prev_vwap (VWAP reclaim priority)")
        if not (close > vwap):
            failed.append(f"close {close:.2f} not above vwap {vwap:.2f}")
        if not (rel_vol >= 2.0):
            failed.append(f"rel_volume_20 {rel_vol:.2f} < 2.0")
        if above != 1.0:
            failed.append("close_above_vwap != 1.0")
        if not (trend > 0.0):
            failed.append(f"trend_score {trend:.2f} <= 0")
        if np.isnan(rsi) or rsi > 75.0:
            failed.append(f"rsi_14 {rsi:.1f} > 75 (or NaN)")

        return 6 - len(failed), len(failed), failed

    # ------------------------------------------------------------------
    # Setup construction & validation
    # ------------------------------------------------------------------

    def _build_long_setup(
        self,
        ticker: str,
        ts: Any,
        setup_type: SetupType,
        entry: float,
        stop: float,
        last: pd.Series,
        reason: str,
        df: Optional[pd.DataFrame] = None,
    ) -> DayTradeSetup:
        """
        Finalise a long setup: validate the stop, compute targets at 2R / 3R,
        compute risk/reward and confidence. A stop that is not strictly below
        the entry invalidates the setup (is_valid=False) with a warning, and
        targets / ratios are zeroed.

        When `df` is provided, the resolved feature sources (session vs generic)
        and their values are recorded in metadata for provenance.
        """
        warnings: List[str] = []

        rel_vol = _f(last, "rel_volume_20")
        vwap    = _f(last, "vwap")
        trend   = _f(last, "trend_score")
        unusual = _f(last, "unusual_volume")

        # Resolve which feature source the rules read for this bar. Falls back to
        # generic names when df is unavailable (e.g. direct _build_long_setup use).
        if df is not None:
            vwap_col    = self.get_vwap_column(df)
            or_high_col = self.get_opening_range_high_column(df)
            or_low_col  = self.get_opening_range_low_column(df)
        else:
            vwap_col, or_high_col, or_low_col = "vwap", "opening_range_high", "opening_range_low"

        vwap_source = "session" if vwap_col.startswith("session_") else "generic"
        or_source   = "session" if or_high_col.startswith("session_") else "generic"

        metadata = {
            "rel_volume_20":   rel_vol,
            "vwap":            vwap,
            "trend_score":     trend,
            "unusual_volume":  unusual,
            "rsi_14":          _f(last, "rsi_14"),
            "opening_range_high": _f(last, "opening_range_high"),
            "opening_range_low":  _f(last, "opening_range_low"),
            # Provenance: which feature source the detector actually used.
            "vwap_source":             vwap_source,
            "opening_range_source":    or_source,
            "vwap_value":              _f(last, vwap_col),
            "opening_range_high_value": _f(last, or_high_col),
            "opening_range_low_value":  _f(last, or_low_col),
        }

        # --- Validate the stop ---
        if np.isnan(stop) or np.isnan(entry):
            warnings.append("Entry or stop is not available (NaN).")
            return self._invalid_setup(ticker, ts, setup_type, reason, warnings, metadata)

        if stop >= entry:
            warnings.append(
                f"Stop ({stop:.4f}) is not below entry ({entry:.4f}); "
                "setup invalidated."
            )
            return self._invalid_setup(ticker, ts, setup_type, reason, warnings, metadata)

        # --- Valid: targets at 2R and 3R ---
        risk     = entry - stop
        target_1 = entry + 2.0 * risk
        target_2 = entry + 3.0 * risk
        rr_1     = compute_risk_reward(entry, stop, target_1)
        rr_2     = compute_risk_reward(entry, stop, target_2)

        confidence = _score_confidence(
            rel_volume_20=rel_vol if not np.isnan(rel_vol) else 0.0,
            close=entry,
            vwap=vwap if not np.isnan(vwap) else entry,  # neutral if missing
            trend_score=trend if not np.isnan(trend) else 0.0,
            unusual_volume=unusual if not np.isnan(unusual) else 0.0,
        )

        return DayTradeSetup(
            ticker=ticker,
            timestamp=ts,
            setup_type=setup_type,
            is_valid=True,
            confidence=confidence,
            entry_trigger=entry,
            stop_loss=stop,
            target_1=target_1,
            target_2=target_2,
            risk_reward_1=rr_1,
            risk_reward_2=rr_2,
            reason=reason,
            warnings=warnings,
            metadata=metadata,
        )

    def _invalid_setup(
        self, ticker: str, ts: Any, setup_type: SetupType,
        reason: str, warnings: List[str], metadata: Dict[str, Any],
    ) -> DayTradeSetup:
        """Build an invalidated (is_valid=False) setup of a known type."""
        return DayTradeSetup(
            ticker=ticker,
            timestamp=ts,
            setup_type=setup_type,
            is_valid=False,
            confidence=0.0,
            entry_trigger=0.0,
            stop_loss=0.0,
            target_1=0.0,
            target_2=0.0,
            risk_reward_1=0.0,
            risk_reward_2=0.0,
            reason=reason,
            warnings=warnings,
            metadata=metadata,
        )

    def _no_setup(
        self, ticker: str, ts: Any, reason: str,
        diagnostics: Optional[Dict[str, List[str]]] = None,
        nearest_setup_type: Optional[SetupType] = None,
        passed_count: int = 0,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> DayTradeSetup:
        """Build a NO_SETUP result with optional near-miss diagnostics."""
        metadata: Dict[str, Any] = {}
        if extra_metadata:
            metadata.update(extra_metadata)
        if diagnostics:
            metadata["diagnostics"] = diagnostics
            metadata["checked_setups"] = list(diagnostics.keys())
        if nearest_setup_type:
            metadata["nearest_setup_type"] = nearest_setup_type.value
            failed_count = len(diagnostics.get(nearest_setup_type.value, []))
            metadata["passed_conditions_count"] = passed_count
            metadata["failed_conditions_count"] = failed_count

        return DayTradeSetup(
            ticker=ticker,
            timestamp=ts,
            setup_type=SetupType.NO_SETUP,
            is_valid=False,
            confidence=0.0,
            entry_trigger=0.0,
            stop_loss=0.0,
            target_1=0.0,
            target_2=0.0,
            risk_reward_1=0.0,
            risk_reward_2=0.0,
            reason=reason,
            warnings=[],
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Feature plumbing
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute features only if a required column is missing."""
        if any(col not in df.columns for col in _REQUIRED_COLUMNS):
            df = add_features(df)
        return df

    @staticmethod
    def _core_features_ready(row: pd.Series) -> bool:
        """True if the always-needed feature values are numeric on this row."""
        for col in ("close", "vwap", "rel_volume_20", "trend_score"):
            if np.isnan(_f(row, col)):
                return False
        return True


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _safe_timestamp(df: Optional[pd.DataFrame]) -> Any:
    """Return the last index value, or None if unavailable."""
    if df is None or len(df) == 0:
        return None
    try:
        return df.index[-1]
    except Exception:
        return None
