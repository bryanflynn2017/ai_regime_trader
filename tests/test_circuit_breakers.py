"""
Tests for risk/circuit_breakers.py.

All tests that create lock files use tmp_path so they never write to the
real state/ directory.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cb(tmp_path):
    """CircuitBreakerManager wired to a temp lock-file path."""
    from config.settings import Settings
    from risk.circuit_breakers import CircuitBreakerManager
    s = Settings(RISK_LOCK_FILE=str(tmp_path / "RISK_LOCKED.txt"))
    return CircuitBreakerManager(settings_override=s)


def _equity(day_pct=0.0, week_pct=0.0, total_pct=0.0, peak=100_000.0):
    """Helper: return (current, day_start, week_start, peak) for given drawdowns."""
    day_current  = peak * (1 - total_pct)  # use total as the "real" current
    return (
        peak * (1 - total_pct),    # current
        peak * (1 - day_pct),      # starting_day_equity that yields day_pct drawdown
        peak * (1 - week_pct),     # starting_week_equity that yields week_pct drawdown
        peak,
    )


def _eval(cb_obj, current, day_start, week_start, peak):
    return cb_obj.evaluate(
        current_equity=current,
        starting_day_equity=day_start,
        starting_week_equity=week_start,
        peak_equity=peak,
    )


# ---------------------------------------------------------------------------
# Test 1: No drawdown → normal trading
# ---------------------------------------------------------------------------

def test_no_drawdown_allows_normal_trading(cb):
    status = _eval(cb, 100_000, 100_000, 100_000, 100_000)

    assert status.trading_allowed        is True
    assert status.new_positions_allowed  is True
    assert status.reduce_only            is False
    assert status.must_close_all         is False
    assert status.size_multiplier        == 1.0
    assert status.triggered_rules        == []


# ---------------------------------------------------------------------------
# Test 2: Daily warn cuts size multiplier to 0.5
# ---------------------------------------------------------------------------

def test_daily_warn_cuts_size_multiplier_to_50pct(cb):
    """2.5% daily drawdown → WARN → size_multiplier=0.5, trading still allowed."""
    # day_start=100_000, current=97_500 → daily_dd=2.5%
    status = _eval(cb, 97_500, 100_000, 100_000, 100_000)

    from risk.circuit_breakers import RULE_DAILY_WARN
    assert status.trading_allowed       is True
    assert status.new_positions_allowed is True
    assert status.reduce_only           is False
    assert abs(status.size_multiplier - 0.50) < 1e-9
    assert RULE_DAILY_WARN in status.triggered_rules


# ---------------------------------------------------------------------------
# Test 3: Daily stop blocks new positions and sets must_close_all
# ---------------------------------------------------------------------------

def test_daily_stop_blocks_new_positions(cb):
    """3.5% daily drawdown → STOP → no new buys, must close all."""
    # day_start=100_000, current=96_500 → daily_dd=3.5%
    status = _eval(cb, 96_500, 100_000, 100_000, 100_000)

    from risk.circuit_breakers import RULE_DAILY_STOP
    assert status.trading_allowed       is True  # can still reduce/close
    assert status.new_positions_allowed is False
    assert status.must_close_all        is True
    assert status.size_multiplier       == 0.0
    assert RULE_DAILY_STOP in status.triggered_rules


# ---------------------------------------------------------------------------
# Test 4: Weekly stop blocks new positions
# ---------------------------------------------------------------------------

def test_weekly_stop_blocks_new_positions(cb):
    """5.5% weekly drawdown → weekly STOP → no new positions."""
    # day flat, week_start gave 5.5% weekly dd, total flat
    status = _eval(cb, 94_500, 94_500, 100_000, 100_000)

    from risk.circuit_breakers import RULE_WEEKLY_STOP
    assert status.new_positions_allowed is False
    assert status.reduce_only           is True
    assert status.size_multiplier       == 0.0
    assert RULE_WEEKLY_STOP in status.triggered_rules


# ---------------------------------------------------------------------------
# Test 5: Total drawdown creates risk lock file
# ---------------------------------------------------------------------------

def test_total_drawdown_creates_lock_file(cb, tmp_path):
    """11% total drawdown → LOCK → file created, all trading halted."""
    lock_path = tmp_path / "RISK_LOCKED.txt"
    assert not lock_path.exists()

    status = _eval(cb, 89_000, 89_000, 89_000, 100_000)  # 11% total

    from risk.circuit_breakers import RULE_TOTAL_LOCK
    assert status.trading_allowed       is False
    assert status.new_positions_allowed is False
    assert status.lock_created          is True
    assert RULE_TOTAL_LOCK in status.triggered_rules
    assert lock_path.exists(), "Lock file must be created on disk"


# ---------------------------------------------------------------------------
# Test 6: Existing lock file blocks all trading
# ---------------------------------------------------------------------------

def test_existing_lock_file_blocks_trading(tmp_path):
    """If the lock file already exists, all trading is blocked immediately."""
    lock_path = tmp_path / "RISK_LOCKED.txt"
    lock_path.write_text("Previous lock — test")

    from config.settings import Settings
    from risk.circuit_breakers import CircuitBreakerManager, RULE_LOCK_EXISTS
    s = Settings(RISK_LOCK_FILE=str(lock_path))
    cb_obj = CircuitBreakerManager(settings_override=s)

    status = _eval(cb_obj, 100_000, 100_000, 100_000, 100_000)

    assert status.trading_allowed       is False
    assert status.new_positions_allowed is False
    assert RULE_LOCK_EXISTS in status.triggered_rules


# ---------------------------------------------------------------------------
# Test 7: Invalid reference equity fails safely, blocks new positions
# ---------------------------------------------------------------------------

def test_invalid_starting_day_equity_fails_safely(cb):
    """starting_day_equity=0 is invalid → fail safe, block new positions."""
    from risk.circuit_breakers import RULE_INVALID_INPUT
    status = _eval(cb, 100_000, 0, 100_000, 100_000)

    assert status.new_positions_allowed is False
    assert RULE_INVALID_INPUT in status.triggered_rules


# ---------------------------------------------------------------------------
# Test 8: current_equity <= 0 creates lock and blocks all trading
# ---------------------------------------------------------------------------

def test_zero_current_equity_creates_lock_and_blocks(cb, tmp_path):
    lock_path = tmp_path / "RISK_LOCKED.txt"
    status = _eval(cb, 0, 100_000, 100_000, 100_000)

    assert status.trading_allowed       is False
    assert status.new_positions_allowed is False
    assert lock_path.exists(), "Lock file must be created for zero equity"


# ---------------------------------------------------------------------------
# Test 9: Lock file contains the reason and a timestamp
# ---------------------------------------------------------------------------

def test_lock_file_content_has_reason_and_timestamp(cb, tmp_path):
    lock_path = tmp_path / "RISK_LOCKED.txt"

    # Trigger a lock
    _eval(cb, 89_000, 89_000, 89_000, 100_000)

    assert lock_path.exists()
    content = lock_path.read_text()

    # Must have non-trivial content
    assert len(content) > 50
    # Must mention the reason
    assert "drawdown" in content.lower() or "halted" in content.lower()
    # Must have a timestamp-like pattern (YYYY-)
    import re
    assert re.search(r"\d{4}-\d{2}-\d{2}", content), (
        "Lock file must contain a date/timestamp"
    )


# ---------------------------------------------------------------------------
# Test 10: read_risk_lock returns the file contents
# ---------------------------------------------------------------------------

def test_read_risk_lock_returns_contents(cb, tmp_path):
    lock_path = tmp_path / "RISK_LOCKED.txt"

    assert cb.read_risk_lock() is None, "Should return None when file does not exist"

    # Trigger lock
    _eval(cb, 89_000, 89_000, 89_000, 100_000)

    contents = cb.read_risk_lock()
    assert contents is not None
    assert len(contents) > 0


# ---------------------------------------------------------------------------
# Additional: create_risk_lock writes readable content
# ---------------------------------------------------------------------------

def test_create_risk_lock_directly(cb, tmp_path):
    lock_path = tmp_path / "RISK_LOCKED.txt"
    path = cb.create_risk_lock(
        reason="Unit test: manual lock",
        metadata={"current_equity": 95_000.0, "peak_equity": 100_000.0},
    )

    assert path.exists()
    content = path.read_text()
    assert "Unit test" in content
    assert "MANUAL REVIEW" in content
