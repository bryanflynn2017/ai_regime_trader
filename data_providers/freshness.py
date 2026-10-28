"""Read-only evidence from provider metadata; never a trading eligibility gate."""
from collections import Counter
from math import isfinite
from statistics import median

import pandas as pd


def _number(value):
    try:
        number = float(value)
        return number if isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _timestamp(value, zone="UTC"):
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return None
        # Alpaca provider attrs are normalized UTC with the timezone removed.
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp
        return stamp.tz_convert(zone).isoformat()
    except (TypeError, ValueError):
        return None


def freshness_snapshot(data_by_ticker, symbols, timeframe, provider, feed, event_timestamp):
    """Describe all requested symbols without changing the existing any-stale decision.

    Age is the provider's original measurement, not recomputed at reporting time.
    Missing includes failed/omitted/empty frames, not necessarily an empty API response.
    """
    threshold = max(180, {"1Min": 1, "5Min": 5}.get(timeframe, 1) * 180)
    rows = []
    for symbol in dict.fromkeys(symbols):
        frame = data_by_ticker.get(symbol)
        missing = frame is None or getattr(frame, "empty", True)
        attrs = getattr(frame, "attrs", {}) if not missing else {}
        age = _number(attrs.get("data_age_seconds"))
        rows.append({
            "symbol": symbol, "provider": provider, "feed": feed,
            "latest_bar_timestamp": _timestamp(attrs.get("latest_bar_time")),
            "comparison_timestamp_et": _timestamp(attrs.get("provider_time"), "America/New_York"),
            "bar_age_minutes": age / 60 if age is not None else None,
            "stale": age > threshold if age is not None else None,
            "missing": missing,
        })
    checked = [row for row in rows if row["stale"] is not None]
    stale = [row for row in checked if row["stale"]]
    ordered = sorted(checked, key=lambda row: (row["bar_age_minutes"], row["symbol"]))
    if len(checked) != len(rows) or not checked:
        classification = "INSUFFICIENT_DATA"
    elif not stale:
        classification = "ALL_FRESH"
    elif len(stale) == len(checked):
        classification = "FEED_WIDE_STALE"
    elif len(stale) == 1:
        classification = "SYMBOL_LOCAL_STALE"
    else:
        classification = "MIXED_FRESHNESS"
    def extreme(row):
        return {"symbol": row["symbol"], "bar_age_minutes": row["bar_age_minutes"]} if row else None
    return {
        "event_timestamp": event_timestamp, "provider": provider, "feed": feed,
        "request_kind": "historical_bars_per_symbol", "timeframe": timeframe,
        "stale_threshold_minutes": threshold / 60,
        "classification": classification,
        "symbols_requested": len(rows), "symbols_checked": len(checked),
        "stale_count": len(stale), "stale_symbols": [r["symbol"] for r in stale],
        "missing_symbols": [r["symbol"] for r in rows if r["missing"]],
        "unknown_age_symbols": [r["symbol"] for r in rows if not r["missing"] and r["stale"] is None],
        "max_bar_age_minutes": ordered[-1]["bar_age_minutes"] if ordered else None,
        "median_bar_age_minutes": median(r["bar_age_minutes"] for r in checked) if checked else None,
        "freshest": extreme(ordered[0] if ordered else None),
        "stalest": extreme(ordered[-1] if ordered else None),
        "symbols": rows,
    }


def aggregate_freshness(counters_list):
    """Summarize recorded evidence only; absent legacy details remain unavailable."""
    snapshots = [e["freshness_snapshot"] for c in counters_list
                 for e in (c.get("stale_data_events") or []) if e.get("freshness_snapshot")]
    events = sum(int(c.get("stale_warning_count", len(c.get("stale_data_events") or [
        f for f in (c.get("data_quality_failures") or []) if f.get("reason") == "STALE_DATA"
    ])) or 0)
                 for c in counters_list)
    counts = Counter(s["classification"] for s in snapshots)
    symbols = Counter(symbol for s in snapshots for symbol in dict.fromkeys(s["stale_symbols"]))
    ages = [age for s in snapshots if (age := _number(s.get("max_bar_age_minutes"))) is not None]
    missing = [c["missing_bar_observations"] for c in counters_list if c.get("missing_bar_observations") is not None]
    return {
        "stale_events": events, "detailed_events": len(snapshots),
        "classification_counts": dict(counts) if snapshots else None,
        "stale_symbol_observations": sum(symbols.values()) if snapshots else None,
        "most_frequently_stale": dict(symbols.most_common(5)) if snapshots else None,
        "worst_bar_age_minutes": max(ages) if ages else None,
        "missing_bar_observations": sum(missing) if missing else None,
        "missing_coverage_sessions": len(missing), "sessions": len(counters_list),
    }


def freshness_report_lines(counters):
    summary = counters.get("freshness_diagnostics") or aggregate_freshness([counters])
    counts = summary["classification_counts"]
    coverage = f"{summary['detailed_events']}/{summary['stale_events']} stale events with symbol details"
    lines = [f"  Freshness evidence: {coverage}"]
    if counts is None:
        lines.append("  Symbol freshness classification/observations/worst age: unavailable (not recorded)")
    else:
        lines.append("  Recorded classifications: " + ", ".join(
            f"{key}={counts.get(key, 0)}" for key in (
                "FEED_WIDE_STALE", "SYMBOL_LOCAL_STALE", "MIXED_FRESHNESS", "INSUFFICIENT_DATA")))
        age = summary['worst_bar_age_minutes']
        age_text = f"{age:g} min" if age is not None else "unavailable"
        lines.append(f"  Stale-symbol observations: {summary['stale_symbol_observations']} | "
                     f"Worst recorded age: {age_text}")
        leaders = ", ".join(f"{s}={n}" for s, n in summary["most_frequently_stale"].items())
        lines.append(f"  Most frequently stale: {leaders or 'none recorded'}")
    missing = summary["missing_bar_observations"]
    lines.append(f"  Missing-bar observations: {missing if missing is not None else 'unavailable'} "
                 f"(coverage {summary['missing_coverage_sessions']}/{summary['sessions']} sessions)")
    return lines
