"""
Tests for dashboard/helpers.py.

All tests operate on pure helper functions — no Streamlit UI is tested.
No network calls, no Alpaca credentials required.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Test 1: load_jsonl returns empty DataFrame for missing file
# ---------------------------------------------------------------------------

def test_load_jsonl_missing_file(tmp_path):
    from dashboard.helpers import load_jsonl

    result = load_jsonl(tmp_path / "nonexistent.jsonl")

    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ---------------------------------------------------------------------------
# Test 2: load_jsonl reads valid JSONL records
# ---------------------------------------------------------------------------

def test_load_jsonl_valid_records(tmp_path):
    from dashboard.helpers import load_jsonl

    log_file = tmp_path / "decisions.jsonl"
    records  = [
        {"timestamp": "2024-01-15T10:00:00", "ticker": "SPY",  "final_action": "BUY",  "risk_approved": True},
        {"timestamp": "2024-01-15T10:05:00", "ticker": "QQQ",  "final_action": "HOLD", "risk_approved": False},
        {"timestamp": "2024-01-15T10:10:00", "ticker": "AAPL", "final_action": "BUY",  "risk_approved": True},
    ]
    log_file.write_text("\n".join(json.dumps(r) for r in records))

    df = load_jsonl(log_file)

    assert len(df) == 3
    assert list(df["ticker"]) == ["SPY", "QQQ", "AAPL"]
    assert df["risk_approved"].sum() == 2


# ---------------------------------------------------------------------------
# Test 3: load_jsonl skips malformed lines gracefully
# ---------------------------------------------------------------------------

def test_load_jsonl_skips_malformed_lines(tmp_path):
    from dashboard.helpers import load_jsonl

    log_file = tmp_path / "mixed.jsonl"
    log_file.write_text(
        '{"ticker": "SPY", "final_action": "BUY"}\n'
        'THIS IS NOT JSON\n'
        '{"ticker": "QQQ", "final_action": "HOLD"}\n'
        '\n'
        '{"incomplete": \n'   # bad JSON
    )

    df = load_jsonl(log_file)

    assert len(df) == 2, f"Expected 2 valid records, got {len(df)}"
    assert set(df["ticker"]) == {"SPY", "QQQ"}


# ---------------------------------------------------------------------------
# Test 4: load_state returns defaults for missing file
# ---------------------------------------------------------------------------

def test_load_state_missing_file(tmp_path):
    from dashboard.helpers import load_state, _DEFAULT_STATE

    state = load_state(tmp_path / "nonexistent.json")

    assert isinstance(state, dict)
    assert state["trades_today"]   == _DEFAULT_STATE["trades_today"]
    assert state["peak_equity"]    == _DEFAULT_STATE["peak_equity"]
    assert state["last_run_date"]  == _DEFAULT_STATE["last_run_date"]


def test_load_state_reads_valid_file(tmp_path):
    from dashboard.helpers import load_state

    state_file = tmp_path / "paper_state.json"
    state_file.write_text(json.dumps({
        "equity": 105_000.0,
        "peak_equity": 107_000.0,
        "trades_today": 3,
        "last_run_date": "2024-01-15",
    }))

    state = load_state(state_file)

    assert abs(state["equity"] - 105_000.0)    < 1e-9
    assert abs(state["peak_equity"] - 107_000.0) < 1e-9
    assert state["trades_today"]   == 3
    assert state["last_run_date"]  == "2024-01-15"


def test_load_state_returns_defaults_for_corrupt_file(tmp_path):
    from dashboard.helpers import load_state, _DEFAULT_STATE

    bad_file = tmp_path / "bad.json"
    bad_file.write_text("this is not valid JSON {{{")

    state = load_state(bad_file)

    assert state["trades_today"] == _DEFAULT_STATE["trades_today"]


# ---------------------------------------------------------------------------
# Test 5: risk_lock_status detects no lock
# ---------------------------------------------------------------------------

def test_risk_lock_status_no_lock(tmp_path):
    from dashboard.helpers import risk_lock_status

    status = risk_lock_status(tmp_path / "RISK_LOCKED.txt")

    assert status["exists"]  is False
    assert status["content"] is None


# ---------------------------------------------------------------------------
# Test 6: risk_lock_status reads existing lock file
# ---------------------------------------------------------------------------

def test_risk_lock_status_reads_lock_file(tmp_path):
    from dashboard.helpers import risk_lock_status

    lock_file = tmp_path / "RISK_LOCKED.txt"
    lock_file.write_text("Locked for test reason\nTimestamp: 2024-01-15")

    status = risk_lock_status(lock_file)

    assert status["exists"] is True
    assert "Locked for test reason" in status["content"]
    assert status["content"] is not None


# ---------------------------------------------------------------------------
# Test 7: tail_file returns last N lines
# ---------------------------------------------------------------------------

def test_tail_file_returns_last_n_lines(tmp_path):
    from dashboard.helpers import tail_file

    log_file = tmp_path / "test.log"
    lines    = [f"Line {i}" for i in range(50)]
    log_file.write_text("\n".join(lines))

    result = tail_file(log_file, n=10)

    result_lines = result.splitlines()
    assert len(result_lines) == 10
    assert result_lines[-1] == "Line 49"
    assert result_lines[0]  == "Line 40"


def test_tail_file_returns_all_when_fewer_than_n_lines(tmp_path):
    from dashboard.helpers import tail_file

    log_file = tmp_path / "short.log"
    log_file.write_text("Line 1\nLine 2\nLine 3")

    result = tail_file(log_file, n=100)

    assert "Line 1" in result
    assert "Line 3" in result


# ---------------------------------------------------------------------------
# Test 8: tail_file handles missing file
# ---------------------------------------------------------------------------

def test_tail_file_missing_file(tmp_path):
    from dashboard.helpers import tail_file

    result = tail_file(tmp_path / "missing.log", n=50)

    assert isinstance(result, str)
    assert len(result) > 0
    assert "not found" in result.lower() or "File" in result


# ---------------------------------------------------------------------------
# Test 9: summarize_decisions counts final actions correctly
# ---------------------------------------------------------------------------

def test_summarize_decisions_counts_final_actions():
    from dashboard.helpers import summarize_decisions

    df = pd.DataFrame({
        "final_action":  ["BUY", "BUY", "HOLD", "HOLD", "HOLD", "SELL"],
        "signal_action": ["BUY", "BUY", "BUY",  "HOLD", "HOLD", "SELL"],
        "risk_approved": [True,  True,  False,  True,   True,   True],
        "dry_run":       [True,  True,  True,   True,   True,   True],
    })

    summary = summarize_decisions(df)

    assert summary["total"]                          == 6
    assert summary["by_final_action"]["BUY"]         == 2
    assert summary["by_final_action"]["HOLD"]        == 3
    assert summary["by_final_action"]["SELL"]        == 1
    assert summary["by_signal_action"]["BUY"]        == 3


# ---------------------------------------------------------------------------
# Test 10: summarize_decisions counts approved vs blocked
# ---------------------------------------------------------------------------

def test_summarize_decisions_approved_vs_blocked():
    from dashboard.helpers import summarize_decisions

    df = pd.DataFrame({
        "risk_approved": [True, True, False, True, False],
        "dry_run":       [True, True, True,  True, True],
    })

    summary = summarize_decisions(df)

    assert summary["approved"] == 3
    assert summary["blocked"]  == 2
    assert summary["total"]    == 5


# ---------------------------------------------------------------------------
# Test 11: summarize_decisions handles empty DataFrame
# ---------------------------------------------------------------------------

def test_summarize_decisions_empty_dataframe():
    from dashboard.helpers import summarize_decisions

    summary = summarize_decisions(pd.DataFrame())

    assert summary["total"]            == 0
    assert summary["approved"]         == 0
    assert summary["blocked"]          == 0
    assert summary["by_final_action"]  == {}
    assert summary["dry_run_count"]    == 0


def test_summarize_decisions_none():
    from dashboard.helpers import summarize_decisions

    summary = summarize_decisions(None)  # type: ignore
    assert summary["total"] == 0


# ---------------------------------------------------------------------------
# Test 12: sensitive values are not returned by helper output
# ---------------------------------------------------------------------------

def test_mask_or_exclude_sensitive_values():
    from dashboard.helpers import mask_or_exclude_sensitive_values

    obj = {
        "api_key":   "ACTUAL_KEY_VALUE",
        "secret_key":"ACTUAL_SECRET",
        "equity":    100_000.0,
        "ticker":    "SPY",
        "nested":    {"api_secret": "NESTED_SECRET", "cash": 50_000.0},
    }

    result = mask_or_exclude_sensitive_values(obj)

    # Sensitive values must be redacted
    assert "ACTUAL_KEY_VALUE" not in str(result)
    assert "ACTUAL_SECRET"    not in str(result)
    assert "NESTED_SECRET"    not in str(result)

    # Safe values must be preserved
    assert result["equity"] == 100_000.0
    assert result["ticker"] == "SPY"
    assert result["nested"]["cash"] == 50_000.0


def test_mask_sensitive_value_placeholders():
    """Redacted values must contain a clear placeholder, not be deleted."""
    from dashboard.helpers import mask_or_exclude_sensitive_values

    result = mask_or_exclude_sensitive_values({"api_key": "MY_KEY"})

    assert "api_key" in result, "Sensitive keys should not be deleted, just masked"
    assert result["api_key"] == "***REDACTED***"


# ---------------------------------------------------------------------------
# Test 13: dashboard helpers do not require Alpaca credentials
# ---------------------------------------------------------------------------

def test_helpers_require_no_credentials(tmp_path):
    """Importing and calling helpers must never fail due to missing credentials."""
    from dashboard.helpers import (
        load_jsonl, load_state, mask_or_exclude_sensitive_values,
        risk_lock_status, summarize_decisions, tail_file,
    )

    # All of these should work with no credentials
    assert isinstance(load_jsonl(tmp_path / "x.jsonl"),             pd.DataFrame)
    assert isinstance(load_state(tmp_path / "x.json"),              dict)
    assert isinstance(risk_lock_status(tmp_path / "x.txt"),         dict)
    assert isinstance(tail_file(tmp_path / "x.log"),                str)
    assert isinstance(summarize_decisions(pd.DataFrame()),           dict)
    assert isinstance(mask_or_exclude_sensitive_values({"k": "v"}), dict)


# ---------------------------------------------------------------------------
# Test 14: dashboard helpers do not make network calls
# ---------------------------------------------------------------------------

def test_helpers_make_no_network_calls(tmp_path, monkeypatch):
    """
    helpers.py must never make HTTP/socket calls.
    We verify this by blocking socket creation and confirming helpers still work.
    """
    import socket

    original_create_connection = socket.create_connection

    def _no_network(*args, **kwargs):
        raise RuntimeError("Network call detected in dashboard helper! This is not allowed.")

    monkeypatch.setattr(socket, "create_connection", _no_network)

    from dashboard.helpers import (
        load_jsonl, load_state, risk_lock_status, tail_file, summarize_decisions,
    )

    # These must all succeed without network
    load_jsonl(tmp_path / "x.jsonl")
    load_state(tmp_path / "x.json")
    risk_lock_status(tmp_path / "x.txt")
    tail_file(tmp_path / "x.log")
    summarize_decisions(pd.DataFrame())


# ---------------------------------------------------------------------------
# Test 15: dashboard command handles missing streamlit gracefully
# ---------------------------------------------------------------------------

def test_dashboard_command_handles_missing_streamlit(monkeypatch):
    """
    The 'dashboard' CLI command must not crash when Streamlit is unavailable.
    It must print helpful install instructions and exit cleanly (code 0).

    When Streamlit IS installed, the command launches a blocking server via
    subprocess.run — stub it out so the test never starts a real server or
    hangs the suite.
    """
    import subprocess
    from typer.testing import CliRunner
    import main as m

    launched = {}

    def _fake_run(cmd, *args, **kwargs):
        launched["cmd"] = [str(part) for part in cmd]
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    runner = CliRunner()
    result = runner.invoke(m.app, ["dashboard"])

    assert result.exit_code == 0, (
        f"Dashboard command exited {result.exit_code}: {result.output}"
    )

    # Either streamlit launched (and exited) OR install instructions were shown
    output_lower = result.output.lower()
    # Accept either scenario
    any_valid = (
        "streamlit" in output_lower     # Mentioned streamlit
        or "pip install" in output_lower # Showed install instruction
        or "localhost" in output_lower   # Dashboard started
    )
    assert any_valid, f"Expected streamlit-related output, got: {result.output}"

    # If streamlit was installed, the command must have launched it (stubbed)
    # rather than printing install instructions.
    if launched:
        assert any("streamlit" in part for part in launched["cmd"])


# ---------------------------------------------------------------------------
# Test 19: summarize_setup_scans returns empty summary for missing file
# ---------------------------------------------------------------------------

def test_summarize_setup_scans_empty_dataframe():
    from dashboard.helpers import summarize_setup_scans

    summary = summarize_setup_scans(pd.DataFrame())

    assert summary["total_scans"] == 0
    assert summary["valid_setups"] == 0
    assert summary["no_setup_count"] == 0
    assert summary["by_setup_type"] == {}
    assert summary["avg_confidence_valid"] == 0.0


def test_summarize_setup_scans_none():
    from dashboard.helpers import summarize_setup_scans

    summary = summarize_setup_scans(None)

    assert summary["total_scans"] == 0
    assert summary["valid_setups"] == 0


# ---------------------------------------------------------------------------
# Test 20: summarize_setup_scans counts valid setups correctly
# ---------------------------------------------------------------------------

def test_summarize_setup_scans_counts_valid():
    from dashboard.helpers import summarize_setup_scans

    records = [
        {"setup_type": "opening_range_breakout_long", "is_valid": True, "confidence": 0.70},
        {"setup_type": "vwap_reclaim_long", "is_valid": True, "confidence": 0.60},
        {"setup_type": "no_setup", "is_valid": False, "confidence": 0.0},
        {"setup_type": "no_setup", "is_valid": False, "confidence": 0.0},
    ]
    df = pd.DataFrame(records)

    summary = summarize_setup_scans(df)

    assert summary["total_scans"] == 4
    assert summary["valid_setups"] == 2
    assert summary["no_setup_count"] == 2
    assert summary["avg_confidence_valid"] > 0.6 and summary["avg_confidence_valid"] < 0.71


# ---------------------------------------------------------------------------
# Test 21: summarize_setup_scans counts by setup type
# ---------------------------------------------------------------------------

def test_summarize_setup_scans_by_setup_type():
    from dashboard.helpers import summarize_setup_scans

    records = [
        {"setup_type": "opening_range_breakout_long", "is_valid": True, "confidence": 0.70},
        {"setup_type": "opening_range_breakout_long", "is_valid": True, "confidence": 0.75},
        {"setup_type": "vwap_reclaim_long", "is_valid": True, "confidence": 0.60},
        {"setup_type": "no_setup", "is_valid": False, "confidence": 0.0},
    ]
    df = pd.DataFrame(records)

    summary = summarize_setup_scans(df)

    assert summary["by_setup_type"]["opening_range_breakout_long"] == 2
    assert summary["by_setup_type"]["vwap_reclaim_long"] == 1
    assert summary["by_setup_type"]["no_setup"] == 1


# ---------------------------------------------------------------------------
# Test 22: summarize_setup_scans includes latest timestamp
# ---------------------------------------------------------------------------

def test_summarize_setup_scans_latest_timestamp():
    from dashboard.helpers import summarize_setup_scans

    records = [
        {"timestamp": "2026-06-05T10:00:00", "setup_type": "no_setup", "is_valid": False, "confidence": 0.0},
        {"timestamp": "2026-06-05T10:05:00", "setup_type": "no_setup", "is_valid": False, "confidence": 0.0},
    ]
    df = pd.DataFrame(records)

    summary = summarize_setup_scans(df)

    assert summary["latest_timestamp"] == "2026-06-05T10:05:00"


# ---------------------------------------------------------------------------
# Test 23: summarize_setup_scans handles missing columns gracefully
# ---------------------------------------------------------------------------

def test_summarize_setup_scans_missing_columns():
    from dashboard.helpers import summarize_setup_scans

    # DataFrame with missing critical columns
    records = [
        {"ticker": "SPY"},
        {"ticker": "QQQ"},
    ]
    df = pd.DataFrame(records)

    summary = summarize_setup_scans(df)

    # Should not crash and return default values
    assert summary["total_scans"] == 2
    assert summary["valid_setups"] == 0
    assert summary["by_setup_type"] == {}


# ---------------------------------------------------------------------------
# Test 24: load_jsonl correctly loads setup scans from JSONL file
# ---------------------------------------------------------------------------

def test_load_jsonl_setup_scans(tmp_path):
    from dashboard.helpers import load_jsonl

    log_file = tmp_path / "setup_scans.jsonl"
    records = [
        {
            "timestamp": "2026-06-05T10:00:00",
            "ticker": "SPY",
            "setup_type": "opening_range_breakout_long",
            "is_valid": True,
            "confidence": 0.70,
            "entry_trigger": 420.50,
            "stop_loss": 418.00,
            "target_1": 425.00,
            "target_2": 430.00,
            "risk_reward_1": 2.0,
            "risk_reward_2": 3.0,
            "reason": "Test setup",
            "warnings": [],
            "metadata": {},
        },
        {
            "timestamp": "2026-06-05T10:05:00",
            "ticker": "QQQ",
            "setup_type": "no_setup",
            "is_valid": False,
            "confidence": 0.0,
            "entry_trigger": 0.0,
            "stop_loss": 0.0,
            "target_1": 0.0,
            "target_2": 0.0,
            "risk_reward_1": 0.0,
            "risk_reward_2": 0.0,
            "reason": "No qualifying setup",
            "warnings": [],
            "metadata": {},
        },
    ]
    log_file.write_text("\n".join(json.dumps(r) for r in records))

    df = load_jsonl(log_file)

    assert len(df) == 2
    assert list(df["ticker"]) == ["SPY", "QQQ"]
    assert df[df["is_valid"]]["confidence"].iloc[0] == 0.70
    assert "entry_trigger" in df.columns


def test_manual_trades_missing_returns_empty(tmp_path):
    from dashboard.helpers import load_jsonl

    df = load_jsonl(tmp_path / "manual_trades.jsonl")

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_split_manual_trades_open_closed():
    from dashboard.helpers import split_manual_trades

    df = pd.DataFrame([
        {"trade_id": "open-1", "ticker": "SPY", "status": "OPEN"},
        {"trade_id": "closed-1", "ticker": "QQQ", "status": "CLOSED"},
    ])

    split = split_manual_trades(df)

    assert list(split["open"]["trade_id"]) == ["open-1"]
    assert list(split["closed"]["trade_id"]) == ["closed-1"]


def test_summarize_manual_trades_realized_pnl_and_win_rate():
    from dashboard.helpers import summarize_manual_trades

    df = pd.DataFrame([
        {"trade_id": "open-1", "status": "OPEN", "realized_pnl": None},
        {"trade_id": "win", "status": "CLOSED", "realized_pnl": 120.0},
        {"trade_id": "loss", "status": "CLOSED", "realized_pnl": -40.0},
    ])

    summary = summarize_manual_trades(df)

    assert summary["open_count"] == 1
    assert summary["closed_count"] == 2
    assert summary["total_realized_pnl"] == 80.0
    assert summary["win_rate"] == 0.5


def test_dashboard_trade_analytics_excludes_marked_trades():
    from dashboard.helpers import summarize_trade_analytics

    df = pd.DataFrame([
        {"trade_id": "included", "status": "CLOSED", "realized_pnl": 120.0, "exclude_from_analytics": False},
        {"trade_id": "excluded", "status": "CLOSED", "realized_pnl": -999.0, "exclude_from_analytics": True},
    ])

    analytics = summarize_trade_analytics(df)

    assert analytics["overall"]["total_trades"] == 1
    assert analytics["overall"]["total_realized_pnl"] == 120.0
    assert analytics["overall"]["excluded_trades"] == 1


def test_dashboard_trade_analytics_handles_forward_test_fields():
    from dashboard.helpers import summarize_trade_analytics

    df = pd.DataFrame([
        {
            "trade_id": "trade-1",
            "status": "CLOSED",
            "realized_pnl": 120.0,
            "test_type": "forward_test",
            "thesis": "VWAP reclaim",
            "market_context": "QQQ strong",
            "entry_quality": "A",
            "mistake": True,
            "mistake_notes": "chased entry",
            "screenshot_path": "screenshots/trade-1.png",
        },
    ])

    analytics = summarize_trade_analytics(df)
    by_test_type = {row["test_type"]: row for row in analytics["by_test_type"]}

    assert analytics["overall"]["mistake_count"] == 1
    assert by_test_type["forward_test"]["total_trades"] == 1


def test_trade_monitor_jsonl_loads(tmp_path):
    from dashboard.helpers import load_jsonl

    log_file = tmp_path / "trade_monitor.jsonl"
    records = [
        {
            "timestamp": "2026-06-06T10:00:00",
            "ticker": "SPY",
            "action": "WATCH",
            "current_price": 101.5,
            "r_multiple": 1.2,
            "unrealized_pnl": 15.0,
            "reason": "WATCH: trade is above 1R",
            "guidance": "Watch closely. Trade is at or above +1R; consider moving stop to breakeven.",
        }
    ]
    log_file.write_text("\n".join(json.dumps(r) for r in records))

    df = load_jsonl(log_file)

    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "SPY"
    assert df.iloc[0]["action"] == "WATCH"
    assert "Watch closely" in df.iloc[0]["guidance"]


def test_trade_monitor_empty_file_handled(tmp_path):
    from dashboard.helpers import load_jsonl

    log_file = tmp_path / "trade_monitor.jsonl"
    log_file.write_text("")

    df = load_jsonl(log_file)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_dashboard_jsonl_secrets_redacted(tmp_path):
    from dashboard.helpers import load_jsonl

    log_file = tmp_path / "manual_trades.jsonl"
    log_file.write_text(json.dumps({
        "trade_id": "t1",
        "ticker": "SPY",
        "status": "OPEN",
        "api_key": "SHOULD_NOT_SHOW",
        "metadata": {"secret_key": "ALSO_HIDDEN"},
    }))

    df = load_jsonl(log_file)
    rendered = df.to_string()

    assert "SHOULD_NOT_SHOW" not in rendered
    assert "ALSO_HIDDEN" not in rendered
    assert "***REDACTED***" in rendered


# ---------------------------------------------------------------------------
# Phase 4: shadow decision summary helper
# ---------------------------------------------------------------------------

def test_summarize_shadow_decisions_counts_and_reasons():
    from dashboard.helpers import summarize_shadow_decisions

    df = pd.DataFrame([
        {"decision": "ENTERED", "skip_reasons": []},
        {"decision": "SKIPPED", "skip_reasons": ["LOW_CONFIDENCE", "STALE_DATA"]},
        {"decision": "SKIPPED", "skip_reasons": ["LOW_CONFIDENCE"]},
    ])
    summary = summarize_shadow_decisions(df)

    assert summary["entered"] == 1
    assert summary["skipped"] == 2
    assert summary["skip_reasons"] == {"LOW_CONFIDENCE": 2, "STALE_DATA": 1}


def test_summarize_shadow_decisions_empty_and_missing_columns():
    from dashboard.helpers import summarize_shadow_decisions

    empty = {"entered": 0, "skipped": 0, "skip_reasons": {}}
    assert summarize_shadow_decisions(pd.DataFrame()) == empty
    assert summarize_shadow_decisions(None) == empty
    assert summarize_shadow_decisions(pd.DataFrame([{"other": 1}])) == empty
