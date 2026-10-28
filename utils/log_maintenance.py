"""
Read-only-first log maintenance helpers.

These helpers archive JSONL logs and optionally filter obvious synthetic/test
records. They never delete archive files and never inspect or print secrets.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


TARGET_LOG_FILES = [
    Path("logs/setup_scans.jsonl"),
    Path("logs/setup_alerts.jsonl"),
    Path("logs/manual_trades.jsonl"),
    Path("logs/trade_monitor.jsonl"),
]


def archive_log_file(path, archive_dir="logs/archive") -> Path | None:
    """
    Copy a log file to the archive directory with a timestamped filename.

    Returns None if the source file is missing. The original file is never
    modified or deleted by this helper.
    """
    source = Path(path)
    if not source.exists():
        return None

    archive_path = Path(archive_dir)
    archive_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = archive_path / f"{source.stem}_{timestamp}{source.suffix}"
    shutil.copy2(source, destination)
    return destination


def load_jsonl_records(path) -> List[Dict[str, Any]]:
    """Load valid JSONL records from a file. Missing files return an empty list."""
    source = Path(path)
    if not source.exists():
        return []

    records: List[Dict[str, Any]] = []
    with open(source, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def write_jsonl_records(path, records: Iterable[Dict[str, Any]]) -> None:
    """Rewrite a JSONL file with the supplied records."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def filter_test_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split records into (kept, removed) using conservative test/demo rules."""
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    for record in records:
        if is_test_demo_synthetic_record(record):
            removed.append(record)
        else:
            kept.append(record)
    return kept, removed


def is_test_demo_synthetic_record(record: Dict[str, Any]) -> bool:
    """Return True when a record looks like synthetic/test/demo data."""
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    if str(record.get("ticker", "")).upper() == "SYNTHETIC":
        return True

    for key in ("provider", "source", "data_source"):
        if _is_synthetic_value(record.get(key)) or _is_synthetic_value(metadata.get(key)):
            return True

    if metadata.get("synthetic") is True:
        return True

    notes = str(record.get("notes", "")).lower()
    if any(term in notes for term in ("monitor test", "synthetic", "test")):
        return True

    if str(record.get("setup_type", "")).lower() == "test":
        return True

    if _has_fixture_prices(record):
        return True

    return False


def resolve_log_paths(log_names: Iterable[str] | None) -> List[Path]:
    """Resolve CLI --logs values to paths, defaulting to all target logs."""
    if not log_names:
        return TARGET_LOG_FILES.copy()

    resolved: List[Path] = []
    known_by_name = {path.name: path for path in TARGET_LOG_FILES}
    known_by_stem = {path.stem: path for path in TARGET_LOG_FILES}
    for item in log_names:
        candidate = Path(item)
        if candidate.name in known_by_name:
            resolved.append(known_by_name[candidate.name])
        elif candidate.stem in known_by_stem and not candidate.parent.parts:
            resolved.append(known_by_stem[candidate.stem])
        else:
            resolved.append(candidate)
    return resolved


def _is_synthetic_value(value: Any) -> bool:
    return str(value).lower() == "synthetic"


def _numeric(record: Dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in record and record.get(key) is not None:
            try:
                return float(record.get(key))
            except (TypeError, ValueError):
                return None
    return None


def _has_fixture_prices(record: Dict[str, Any]) -> bool:
    entry = _numeric(record, "entry", "entry_price", "entry_trigger")
    stop = _numeric(record, "stop", "stop_loss")
    target1 = _numeric(record, "target1", "target_1")
    target2 = _numeric(record, "target2", "target_2")
    if entry is None or stop is None or target1 is None or target2 is None:
        return False

    return (
        abs(entry - 100.0) < 1e-9
        and abs(stop - 99.0) < 1e-9
        and abs(target1 - 102.0) < 1e-9
        and (abs(target2 - 103.0) < 1e-9 or abs(target2 - 104.0) < 1e-9)
    )
