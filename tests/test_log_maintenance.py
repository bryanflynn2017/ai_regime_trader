"""Tests for safe JSONL log maintenance commands."""
from __future__ import annotations

import json
from pathlib import Path

import typer.testing

from main import app
from utils.log_maintenance import (
    archive_log_file,
    filter_test_records,
    load_jsonl_records,
    write_jsonl_records,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_log_cleanup_dry_run_does_not_modify_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "logs" / "setup_alerts.jsonl"
    records = [{"ticker": "SYNTHETIC"}, {"ticker": "SPY"}]
    _write_jsonl(path, records)

    result = typer.testing.CliRunner().invoke(app, ["log-cleanup", "--logs", "setup_alerts"])

    assert result.exit_code == 0, result.stdout
    assert _read_jsonl(path) == records
    assert not (tmp_path / "logs" / "archive").exists()


def test_log_cleanup_apply_archives_before_rewrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "logs" / "setup_alerts.jsonl"
    _write_jsonl(path, [{"ticker": "SYNTHETIC"}, {"ticker": "SPY"}])

    result = typer.testing.CliRunner().invoke(
        app,
        ["log-cleanup", "--apply", "--logs", "setup_alerts"],
    )

    assert result.exit_code == 0, result.stdout
    assert _read_jsonl(path) == [{"ticker": "SPY"}]
    archives = list((tmp_path / "logs" / "archive").glob("setup_alerts_*.jsonl"))
    assert len(archives) == 1
    assert _read_jsonl(archives[0]) == [{"ticker": "SYNTHETIC"}, {"ticker": "SPY"}]


def test_log_cleanup_archive_only_does_not_modify_original(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "logs" / "manual_trades.jsonl"
    records = [{"ticker": "SYNTHETIC"}, {"ticker": "SPY"}]
    _write_jsonl(path, records)

    result = typer.testing.CliRunner().invoke(
        app,
        ["log-cleanup", "--archive-only", "--logs", "manual_trades"],
    )

    assert result.exit_code == 0, result.stdout
    assert _read_jsonl(path) == records
    assert list((tmp_path / "logs" / "archive").glob("manual_trades_*.jsonl"))


def test_filter_synthetic_ticker_removed():
    kept, removed = filter_test_records([{"ticker": "SYNTHETIC"}, {"ticker": "AAPL"}])
    assert kept == [{"ticker": "AAPL"}]
    assert removed == [{"ticker": "SYNTHETIC"}]


def test_filter_notes_containing_test_removed():
    kept, removed = filter_test_records([
        {"ticker": "SPY", "notes": "monitor test record"},
        {"ticker": "QQQ", "notes": "real trade"},
    ])
    assert kept == [{"ticker": "QQQ", "notes": "real trade"}]
    assert removed == [{"ticker": "SPY", "notes": "monitor test record"}]


def test_filter_normal_real_record_kept():
    record = {"ticker": "AAPL", "provider": "yfinance", "entry": 187.25, "notes": "real idea"}
    kept, removed = filter_test_records([record])
    assert kept == [record]
    assert removed == []


def test_missing_files_handled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = typer.testing.CliRunner().invoke(
        app,
        ["log-cleanup", "--logs", "logs/missing.jsonl"],
    )
    assert result.exit_code == 0, result.stdout
    assert "missing, skipped" in result.stdout


def test_log_cleanup_counts_printed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(tmp_path / "logs" / "trade_monitor.jsonl", [{"ticker": "SYNTHETIC"}, {"ticker": "MSFT"}])

    result = typer.testing.CliRunner().invoke(app, ["log-cleanup", "--logs", "trade_monitor"])

    assert result.exit_code == 0, result.stdout
    assert "total=2" in result.stdout
    assert "test/demo/synthetic=1" in result.stdout
    assert "remain=1" in result.stdout


def test_archive_filename_contains_timestamp(tmp_path):
    path = tmp_path / "logs" / "setup_scans.jsonl"
    _write_jsonl(path, [{"ticker": "SPY"}])

    archived = archive_log_file(path, archive_dir=tmp_path / "logs" / "archive")

    assert archived is not None
    assert archived.name.startswith("setup_scans_")
    assert archived.suffix == ".jsonl"
    timestamp_part = archived.stem.replace("setup_scans_", "")
    assert len(timestamp_part) >= len("YYYYmmdd_HHMMSS")


def test_log_cleanup_output_does_not_print_secrets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(
        tmp_path / "logs" / "setup_scans.jsonl",
        [{"ticker": "SYNTHETIC", "api_key": "SECRET_VALUE"}],
    )

    result = typer.testing.CliRunner().invoke(app, ["log-cleanup", "--logs", "setup_scans"])

    assert result.exit_code == 0, result.stdout
    assert "SECRET_VALUE" not in result.stdout
    assert "api_key" not in result.stdout


def test_load_and_write_jsonl_records(tmp_path):
    path = tmp_path / "logs" / "sample.jsonl"
    write_jsonl_records(path, [{"ticker": "SPY"}, {"ticker": "QQQ"}])
    assert load_jsonl_records(path) == [{"ticker": "SPY"}, {"ticker": "QQQ"}]
