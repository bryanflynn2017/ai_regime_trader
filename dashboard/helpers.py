"""
Dashboard helper functions — pure Python, no Streamlit imports.

All functions here are importable and testable without Streamlit being installed.
They handle file I/O, parsing, and summarisation for the dashboard.

Security rules:
  - Never read or return .env contents.
  - Never return API keys or secrets.
  - mask_or_exclude_sensitive_values() is available to sanitise any dict.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Defaults (mirrors main.py _DEFAULT_PAPER_STATE — kept separate to avoid
# importing typer / CLI machinery when loading the dashboard)
# ---------------------------------------------------------------------------

_DEFAULT_STATE: Dict[str, Any] = {
    "equity":               100_000.0,
    "starting_day_equity":  100_000.0,
    "starting_week_equity": 100_000.0,
    "peak_equity":          100_000.0,
    "trades_today":         0,
    "last_run_timestamp":   None,
    "last_run_date":        None,
    "last_run_iso_week":    None,
}

# Keys that must never appear as values in dashboard output
_SENSITIVE_KEYWORDS = frozenset({
    "key", "secret", "token", "password", "credential",
    "api_key", "api_secret", "alpaca_api_key", "alpaca_secret_key",
    "alpaca_secret", "secret_key",
})


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_jsonl(path) -> pd.DataFrame:
    """
    Load a JSONL (newline-delimited JSON) file as a pandas DataFrame.

    Returns an empty DataFrame if the file is missing, empty, or entirely corrupt.
    Malformed individual lines are silently skipped so one bad record never
    prevents the rest from loading.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    records: List[Dict] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(mask_or_exclude_sensitive_values(json.loads(line)))
            except json.JSONDecodeError:
                pass  # Skip malformed lines

    except OSError:
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def load_state(path) -> Dict[str, Any]:
    """
    Load paper_state.json.  Returns default values if the file is missing
    or contains invalid JSON.
    """
    path = Path(path)
    if not path.exists():
        return _DEFAULT_STATE.copy()

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        state  = _DEFAULT_STATE.copy()
        state.update(loaded)
        return state
    except (json.JSONDecodeError, OSError, TypeError):
        return _DEFAULT_STATE.copy()


def risk_lock_status(path) -> Dict[str, Any]:
    """
    Return information about the risk lock file.

    Returns:
        {"exists": bool, "content": str | None}

    The content is the raw text of the lock file.  If the file is unreadable
    the content field contains a short error message.
    """
    path = Path(path)
    if not path.exists():
        return {"exists": False, "content": None}

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        content = f"[Could not read lock file: {exc}]"

    return {"exists": True, "content": content}


def tail_file(path, n: int = 100) -> str:
    """
    Return the last *n* lines of a text file as a single string.

    Returns a helpful placeholder message if the file is missing or unreadable.
    """
    path = Path(path)
    if not path.exists():
        return f"[File not found: {path}]"

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:]) if lines else "[File is empty]"
    except OSError as exc:
        return f"[Error reading file: {exc}]"


def summarize_decisions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute summary statistics from a paper-decisions DataFrame.

    Handles missing columns gracefully — every key is always present in
    the returned dict so callers don't need to check.
    """
    empty = {
        "total":           0,
        "by_signal_action":  {},
        "by_final_action":   {},
        "approved":        0,
        "blocked":         0,
        "dry_run_count":   0,
        "submitted_count": 0,
    }

    if df is None or df.empty:
        return empty

    summary: Dict[str, Any] = {"total": len(df)}

    summary["by_signal_action"] = (
        df["signal_action"].value_counts().to_dict()
        if "signal_action" in df.columns else {}
    )

    summary["by_final_action"] = (
        df["final_action"].value_counts().to_dict()
        if "final_action" in df.columns else {}
    )

    if "risk_approved" in df.columns:
        approved = int(df["risk_approved"].fillna(False).astype(bool).sum())
        summary["approved"] = approved
        summary["blocked"]  = len(df) - approved
    else:
        summary["approved"] = 0
        summary["blocked"]  = 0

    if "dry_run" in df.columns:
        dry_count = int(df["dry_run"].fillna(True).astype(bool).sum())
        summary["dry_run_count"]   = dry_count
        summary["submitted_count"] = len(df) - dry_count
    else:
        summary["dry_run_count"]   = 0
        summary["submitted_count"] = 0

    return summary


def mask_or_exclude_sensitive_values(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of *obj* with any sensitive values replaced by '***REDACTED***'.

    A key is considered sensitive if it contains one of the known sensitive
    keyword fragments (case-insensitive).  Nested dicts are recursed into.

    This function is used to sanitise dicts before displaying them in the
    dashboard UI — ensuring no API keys, secrets, or passwords are ever shown.
    """
    result: Dict[str, Any] = {}
    for k, v in obj.items():
        k_lower = k.lower()
        if any(s in k_lower for s in _SENSITIVE_KEYWORDS):
            result[k] = "***REDACTED***"
        elif isinstance(v, dict):
            result[k] = mask_or_exclude_sensitive_values(v)
        else:
            result[k] = v
    return result


def load_latest_result(results_dir) -> Optional[Dict[str, Any]]:
    """
    Load the most recently saved result JSON from a directory.

    Returns None if the directory is missing or contains no JSON files.
    """
    path = Path(results_dir)
    if not path.exists():
        return None

    json_files = sorted(path.glob("*.json"), reverse=True)
    if not json_files:
        return None

    try:
        return json.loads(json_files[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def list_result_files(results_dir, max_count: int = 10) -> List[Dict[str, Any]]:
    """
    Return metadata for the most recent result JSON files in a directory.

    Each entry has 'filename', 'timestamp', and any top-level metrics keys.
    """
    path = Path(results_dir)
    if not path.exists():
        return []

    entries: List[Dict[str, Any]] = []
    for f in sorted(path.glob("*.json"), reverse=True)[:max_count]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            entries.append({
                "filename":    f.name,
                "timestamp":   data.get("timestamp", "unknown"),
                "final_equity":data.get("final_equity"),
                "total_return":data.get("metrics", {}).get("total_return"),
                "sharpe":      data.get("metrics", {}).get("sharpe_ratio"),
            })
        except Exception:
            entries.append({"filename": f.name, "timestamp": "unreadable"})

    return entries


def get_recent_rejections(df: pd.DataFrame, top_n: int = 5) -> List[str]:
    """
    Extract the most common rejection reasons from a decisions DataFrame.

    Returns a list of strings (reason, count).
    """
    if df is None or df.empty or "reason" not in df.columns:
        return []

    blocked_reasons = df.loc[
        df.get("risk_approved", pd.Series(True, index=df.index)).fillna(True) == False,
        "reason",
    ]
    if blocked_reasons.empty:
        return []

    counts = blocked_reasons.value_counts().head(top_n)
    return [f"{reason} ({count}x)" for reason, count in counts.items()]


def summarize_setup_scans(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute summary statistics from a setup scans DataFrame.

    Returns a dict with counts of valid setups, setup types, and average confidence.
    Handles missing columns and empty DataFrames gracefully.
    """
    empty_summary = {
        "total_scans": 0,
        "valid_setups": 0,
        "no_setup_count": 0,
        "by_setup_type": {},
        "avg_confidence_valid": 0.0,
        "latest_timestamp": None,
    }

    if df is None or df.empty:
        return empty_summary

    summary: Dict[str, Any] = {
        "total_scans": len(df),
    }

    # Count valid setups
    if "is_valid" in df.columns:
        valid_count = int(df["is_valid"].fillna(False).astype(bool).sum())
        summary["valid_setups"] = valid_count
    else:
        summary["valid_setups"] = 0

    # Count NO_SETUP rows
    if "setup_type" in df.columns:
        no_setup_count = int((df["setup_type"] == "no_setup").sum())
        summary["no_setup_count"] = no_setup_count
        # Count by setup type
        summary["by_setup_type"] = (
            df["setup_type"].value_counts().to_dict()
            if "setup_type" in df.columns else {}
        )
    else:
        summary["no_setup_count"] = 0
        summary["by_setup_type"] = {}

    # Average confidence for valid setups
    if "is_valid" in df.columns and "confidence" in df.columns:
        valid_df = df[df["is_valid"].fillna(False).astype(bool)]
        if not valid_df.empty:
            summary["avg_confidence_valid"] = float(
                valid_df["confidence"].fillna(0).mean()
            )
        else:
            summary["avg_confidence_valid"] = 0.0
    else:
        summary["avg_confidence_valid"] = 0.0

    # Latest timestamp
    if "timestamp" in df.columns:
        try:
            summary["latest_timestamp"] = df["timestamp"].iloc[-1]
        except (IndexError, KeyError):
            summary["latest_timestamp"] = None
    else:
        summary["latest_timestamp"] = None

    return summary


def split_manual_trades(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split a manual-trades DataFrame into OPEN and CLOSED trades.

    Missing/empty input returns two empty DataFrames. Status comparisons are
    case-insensitive so older logs remain displayable.
    """
    if df is None or df.empty or "status" not in df.columns:
        return {"open": pd.DataFrame(), "closed": pd.DataFrame()}

    status = df["status"].fillna("").astype(str).str.upper()
    return {
        "open": df[status == "OPEN"].copy(),
        "closed": df[status == "CLOSED"].copy(),
    }


def summarize_manual_trades(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute dashboard summary metrics for manual trades.

    Returns counts, total realized P&L, and win rate for closed trades. Missing
    columns are handled gracefully.
    """
    empty = {
        "open_count": 0,
        "closed_count": 0,
        "total_realized_pnl": 0.0,
        "win_rate": 0.0,
    }

    if df is None or df.empty:
        return empty

    split = split_manual_trades(df)
    open_df = split["open"]
    closed_df = split["closed"]

    summary = {
        "open_count": int(len(open_df)),
        "closed_count": int(len(closed_df)),
        "total_realized_pnl": 0.0,
        "win_rate": 0.0,
    }

    if not closed_df.empty and "realized_pnl" in closed_df.columns:
        pnl = pd.to_numeric(closed_df["realized_pnl"], errors="coerce").fillna(0.0)
        summary["total_realized_pnl"] = float(pnl.sum())
        summary["win_rate"] = float((pnl > 0).sum() / len(closed_df))

    return summary


def summarize_trade_analytics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute closed-trade analytics for dashboard display.

    This delegates to trades.trade_analytics so CLI and dashboard math stay in
    sync. Returns plain dictionaries suitable for metrics and tables.
    """
    from trades.trade_analytics import TradeAnalytics

    result = TradeAnalytics.analyze_dataframe(df)
    return result.to_dict()


def summarize_shadow_decisions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize shadow decision records (logs/shadow_decisions.jsonl) for the
    dashboard: ENTERED/SKIPPED counts plus skip-reason counts.

    Missing/empty input returns all-zero counts. Read-only and offline.
    """
    empty = {"entered": 0, "skipped": 0, "skip_reasons": {}}
    if df is None or df.empty or "decision" not in df.columns:
        return empty

    decisions = df["decision"].fillna("").astype(str).str.upper()
    reasons: Dict[str, int] = {}
    if "skip_reasons" in df.columns:
        for value in df["skip_reasons"].dropna():
            if isinstance(value, (list, tuple)):
                for reason in value:
                    key = str(reason)
                    reasons[key] = reasons.get(key, 0) + 1

    return {
        "entered": int((decisions == "ENTERED").sum()),
        "skipped": int((decisions == "SKIPPED").sum()),
        "skip_reasons": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
    }


def summarize_trade_lifecycle(trades_df: pd.DataFrame, monitor_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute trade lifecycle analytics for dashboard display.

    Joins closed manual trades with trade_monitor.jsonl history via
    trades.trade_lifecycle so CLI and dashboard math stay in sync.
    """
    from trades.trade_lifecycle import TradeLifecycleAnalytics

    result = TradeLifecycleAnalytics.analyze_frames(trades_df, monitor_df)
    return result.to_dict()
