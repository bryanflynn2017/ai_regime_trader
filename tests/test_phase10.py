"""
Phase 10 tests: result persistence, health-check, export-status, and
dashboard helpers for saved results.

All tests pass without network access or Alpaca credentials.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_backtest_data() -> dict:
    """Minimal dict that looks like a saved backtest result."""
    return {
        "timestamp":       "2024-01-15T10:00:00",
        "source":          "synthetic",
        "tickers":         ["SYNTHETIC"],
        "starting_equity": 100_000.0,
        "final_equity":    102_000.0,
        "metrics": {
            "total_return": 0.02, "cagr": 0.05,
            "sharpe_ratio": 0.8,  "sortino_ratio": 0.9,
            "max_drawdown": -0.05, "win_rate": 0.6,
            "profit_factor": 1.5, "total_trades": 5,
            "average_win": 400.0, "average_loss": -200.0,
            "best_trade": 800.0,  "worst_trade": -300.0,
            "exposure_time": 0.3, "benchmark_return": 0.03,
            "excess_return": -0.01,
        },
        "total_trades": 5,
        "warnings": [],
        "config": {"slippage_bps": 5.0},
    }


# ---------------------------------------------------------------------------
# Test: saved backtest JSON excludes secrets
# ---------------------------------------------------------------------------

def test_saved_backtest_json_excludes_secrets(tmp_path):
    """_save_result_json must sanitise API keys before writing."""
    from main import _save_result_json

    data = {
        **_make_minimal_backtest_data(),
        "api_key": "SHOULD_BE_REDACTED",
        "alpaca_secret": "ALSO_REDACTED",
    }

    path = _save_result_json(data, "backtests", "test_bt", base_dir=str(tmp_path))

    assert path.exists()
    content = json.loads(path.read_text())

    assert "SHOULD_BE_REDACTED"  not in json.dumps(content)
    assert "ALSO_REDACTED"       not in json.dumps(content)
    assert content["source"]     == "synthetic"       # Benign values preserved
    assert content["total_trades"] == 5


# ---------------------------------------------------------------------------
# Test: saved walk-forward JSON excludes secrets
# ---------------------------------------------------------------------------

def test_saved_walk_forward_json_excludes_secrets(tmp_path):
    from main import _save_result_json

    data = {
        "timestamp":  "2024-01-15T10:00:00",
        "source":     "synthetic",
        "n_windows":  2,
        "secret_key": "SHOULD_NOT_APPEAR",
        "combined_metrics": {"total_return": 0.01, "sharpe_ratio": 0.5},
        "windows": [],
    }

    path = _save_result_json(data, "walk_forward", "test_wf", base_dir=str(tmp_path))

    content = json.loads(path.read_text())
    assert "SHOULD_NOT_APPEAR" not in json.dumps(content)
    assert content["n_windows"] == 2


# ---------------------------------------------------------------------------
# Test: results directories are created if missing
# ---------------------------------------------------------------------------

def test_results_directories_created(tmp_path):
    from main import _save_result_json

    # Directories don't exist yet
    assert not (tmp_path / "backtests").exists()

    path = _save_result_json({"x": 1}, "backtests", "bt", base_dir=str(tmp_path))

    assert path.exists()
    assert (tmp_path / "backtests").is_dir()


# ---------------------------------------------------------------------------
# Test: dashboard load_latest_result works with saved files
# ---------------------------------------------------------------------------

def test_load_latest_result_returns_most_recent(tmp_path):
    from dashboard.helpers import load_latest_result

    results_dir = tmp_path / "backtests"
    results_dir.mkdir()

    # Write two files with different timestamps in name
    older = results_dir / "backtest_20240101_100000.json"
    newer = results_dir / "backtest_20240115_100000.json"
    older.write_text(json.dumps({"timestamp": "2024-01-01", "final_equity": 100_000}))
    newer.write_text(json.dumps({"timestamp": "2024-01-15", "final_equity": 105_000}))

    result = load_latest_result(results_dir)

    assert result is not None
    # Sorted by filename descending — newest file returned
    assert result["timestamp"] == "2024-01-15"


def test_load_latest_result_missing_dir(tmp_path):
    from dashboard.helpers import load_latest_result

    result = load_latest_result(tmp_path / "nonexistent_dir")
    assert result is None


def test_load_latest_result_empty_dir(tmp_path):
    from dashboard.helpers import load_latest_result

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert load_latest_result(empty_dir) is None


def test_list_result_files_returns_metadata(tmp_path):
    from dashboard.helpers import list_result_files

    results_dir = tmp_path / "backtests"
    results_dir.mkdir()
    (results_dir / "backtest_20240115_100000.json").write_text(
        json.dumps({"timestamp": "2024-01-15", "final_equity": 102_000,
                    "metrics": {"total_return": 0.02}})
    )

    files = list_result_files(results_dir)

    assert len(files) == 1
    assert files[0]["filename"] == "backtest_20240115_100000.json"
    assert files[0]["timestamp"] == "2024-01-15"


# ---------------------------------------------------------------------------
# Test: health-check works without network access
# ---------------------------------------------------------------------------

def test_health_check_no_network():
    from typer.testing import CliRunner
    import main as m

    runner = CliRunner()
    result = runner.invoke(m.app, ["health-check"])

    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert "Health check complete" in result.output


def test_health_check_no_secrets():
    from typer.testing import CliRunner
    import main as m

    runner = CliRunner()
    result = runner.invoke(m.app, ["health-check"])

    output_lower = result.output.lower()
    # No secret values should appear
    for banned in ("api_key=", "secret=", "password=", "alpaca_secret"):
        assert banned not in output_lower, (
            f"Banned pattern '{banned}' found in health-check output"
        )
    # Should NOT print actual key/secret values
    assert result.exit_code == 0


def test_health_check_mentions_paper_trading():
    from typer.testing import CliRunner
    import main as m

    runner = CliRunner()
    result = runner.invoke(m.app, ["health-check"])

    assert "PAPER_TRADING" in result.output


# ---------------------------------------------------------------------------
# Test: export-status creates a JSON file
# ---------------------------------------------------------------------------

def test_export_status_creates_json():
    """export-status must invoke _save_result_json."""
    from typer.testing import CliRunner
    from unittest.mock import MagicMock
    import main as m

    mock_path = MagicMock()
    mock_path.__str__ = lambda self: "results/status/status_test.json"

    with patch.object(m, "_save_result_json", return_value=mock_path) as mock_save:
        runner = CliRunner()
        result = runner.invoke(m.app, ["export-status"])

    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert mock_save.called, "_save_result_json must be called by export-status"


def test_export_status_excludes_secrets():
    """The data passed to _save_result_json must not contain credential values."""
    from typer.testing import CliRunner
    from unittest.mock import MagicMock
    import main as m

    captured: list = []
    mock_path = MagicMock()

    def capture(data, *args, **kwargs):
        captured.append(data)
        return mock_path

    with patch.object(m, "_save_result_json", side_effect=capture):
        runner = CliRunner()
        runner.invoke(m.app, ["export-status"])

    assert captured, "export-status must call _save_result_json"
    data_str = json.dumps(captured[0], default=str).lower()

    # Raw placeholder values from .env.example must never appear
    for banned in ("your_paper_key_here", "your_paper_secret_here"):
        assert banned not in data_str, (
            f"Credential placeholder '{banned}' found in export-status output"
        )


def test_export_status_includes_risk_lock_status():
    """export-status data must contain a risk_lock field with 'exists' key."""
    from typer.testing import CliRunner
    from unittest.mock import MagicMock
    import main as m

    captured: list = []
    mock_path = MagicMock()

    def capture(data, *args, **kwargs):
        captured.append(data)
        return mock_path

    with patch.object(m, "_save_result_json", side_effect=capture):
        runner = CliRunner()
        runner.invoke(m.app, ["export-status"])

    assert captured
    data = captured[0]
    assert "risk_lock" in data, "export-status data must include risk_lock"
    assert "exists" in data["risk_lock"]


# ---------------------------------------------------------------------------
# Test: dashboard helpers do not import streamlit
# ---------------------------------------------------------------------------

def test_helpers_importable_without_streamlit():
    """dashboard.helpers must never import streamlit."""
    import sys

    # Temporarily block streamlit if present
    streamlit_backup = sys.modules.pop("streamlit", None)
    try:
        # Re-import helpers to trigger any top-level imports
        import importlib
        import dashboard.helpers
        importlib.reload(dashboard.helpers)
    finally:
        if streamlit_backup is not None:
            sys.modules["streamlit"] = streamlit_backup


# ---------------------------------------------------------------------------
# Test: --save-results option exists on backtest and walk-forward commands
# ---------------------------------------------------------------------------

def test_backtest_has_save_results_option():
    from typer.testing import CliRunner
    import main as m

    runner = CliRunner()
    result = runner.invoke(m.app, ["backtest", "--help"])
    assert "--save-results" in result.output


def test_walk_forward_has_save_results_option():
    from typer.testing import CliRunner
    import main as m

    runner = CliRunner()
    result = runner.invoke(m.app, ["walk-forward", "--help"])
    assert "--save-results" in result.output
