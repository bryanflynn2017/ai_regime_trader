"""
Tests for risk/position_sizing.py.

All tests are pure calculations — no network, no broker, no file I/O.
"""
from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# Fixture: default settings pointing to no real files
# ---------------------------------------------------------------------------

@pytest.fixture
def sizer():
    from risk.position_sizing import PositionSizer
    return PositionSizer()


@pytest.fixture
def sizer_no_fractional():
    """PositionSizer with fractional shares disabled."""
    from config.settings import Settings
    from risk.position_sizing import PositionSizer
    s = Settings(ALLOW_FRACTIONAL_SHARES=False)
    return PositionSizer(settings_override=s)


# ---------------------------------------------------------------------------
# Test 1: Reject zero / negative equity
# ---------------------------------------------------------------------------

def test_rejects_zero_equity(sizer):
    result = sizer.calculate_position_size(equity=0, entry_price=100, stop_loss=97)
    assert not result.approved
    assert "equity" in result.reason.lower()


def test_rejects_negative_equity(sizer):
    result = sizer.calculate_position_size(equity=-50_000, entry_price=100, stop_loss=97)
    assert not result.approved


# ---------------------------------------------------------------------------
# Test 2: Reject invalid entry price
# ---------------------------------------------------------------------------

def test_rejects_zero_entry_price(sizer):
    result = sizer.calculate_position_size(equity=100_000, entry_price=0, stop_loss=97)
    assert not result.approved
    assert "entry" in result.reason.lower()


def test_rejects_negative_entry_price(sizer):
    result = sizer.calculate_position_size(equity=100_000, entry_price=-10, stop_loss=-15)
    assert not result.approved


# ---------------------------------------------------------------------------
# Test 3: Reject stop_loss >= entry_price for long trades
# ---------------------------------------------------------------------------

def test_rejects_stop_loss_equal_to_entry(sizer):
    result = sizer.calculate_position_size(equity=100_000, entry_price=100, stop_loss=100)
    assert not result.approved
    assert "stop" in result.reason.lower()


def test_rejects_stop_loss_above_entry(sizer):
    result = sizer.calculate_position_size(equity=100_000, entry_price=100, stop_loss=105)
    assert not result.approved


def test_rejects_zero_stop_loss(sizer):
    result = sizer.calculate_position_size(equity=100_000, entry_price=100, stop_loss=0)
    assert not result.approved


# ---------------------------------------------------------------------------
# Test 4: Correct shares from risk dollars and stop distance
# ---------------------------------------------------------------------------

def test_calculates_shares_from_risk_and_stop(sizer):
    """
    Verify the base formula: shares = (equity × MAX_RISK_PER_TRADE) / stop_distance.

    With equity=100_000, MAX_RISK_PER_TRADE=0.005, entry=100, stop=97:
      risk_dollars   = 100_000 × 0.005 = 500
      stop_distance  = 3
      raw_shares     = 500 / 3 ≈ 166.67
      position_value = 16,667 → capped at MAX_SINGLE_POS (10,000)
      final_shares   = 100
    """
    from config.settings import settings
    result = sizer.calculate_position_size(
        equity=100_000,
        entry_price=100.0,
        stop_loss=97.0,
    )

    assert result.approved
    # Max-loss must not exceed uncapped risk_dollars
    assert result.max_loss_dollars <= result.risk_dollars + 1e-9

    # The formula must hold before capping:
    stop_dist   = 100.0 - 97.0
    risk_dollars = 100_000 * settings.MAX_RISK_PER_TRADE
    expected_raw = risk_dollars / stop_dist
    assert abs(result.metadata["raw_shares"] - expected_raw) < 1e-9


# ---------------------------------------------------------------------------
# Test 5: Cap by MAX_SINGLE_POSITION_EXPOSURE
# ---------------------------------------------------------------------------

def test_caps_by_max_single_position_exposure(sizer):
    """
    Position value must not exceed MAX_SINGLE_POSITION_EXPOSURE × equity.
    """
    from config.settings import settings

    equity = 100_000
    result = sizer.calculate_position_size(
        equity=equity,
        entry_price=100.0,
        stop_loss=97.0,  # wide stop → large raw size → must be capped
    )

    assert result.approved
    max_allowed = settings.MAX_SINGLE_POSITION_EXPOSURE * equity
    assert result.position_value <= max_allowed + 1e-9


# ---------------------------------------------------------------------------
# Test 6: Cap by remaining MAX_PORTFOLIO_EXPOSURE headroom
# ---------------------------------------------------------------------------

def test_caps_by_portfolio_exposure_headroom(sizer):
    """
    Remaining portfolio headroom = MAX_PORTFOLIO_EXPOSURE × equity − current.
    Position value must not exceed that headroom.
    """
    from config.settings import settings

    equity = 100_000
    current_portfolio = 42_000  # headroom = 50_000 − 42_000 = 8_000

    result = sizer.calculate_position_size(
        equity=equity,
        entry_price=100.0,
        stop_loss=97.0,
        current_portfolio_exposure=current_portfolio,
    )

    assert result.approved
    headroom = max(0.0, settings.MAX_PORTFOLIO_EXPOSURE * equity - current_portfolio)
    assert result.position_value <= headroom + 1e-9


# ---------------------------------------------------------------------------
# Test 7: Cap by remaining per-ticker exposure headroom
# ---------------------------------------------------------------------------

def test_caps_by_ticker_exposure_headroom(sizer):
    """
    Remaining ticker headroom = MAX_SINGLE_POSITION_EXPOSURE × equity − current.
    """
    from config.settings import settings

    equity = 100_000
    current_ticker = 7_500  # headroom = 10_000 − 7_500 = 2_500

    result = sizer.calculate_position_size(
        equity=equity,
        entry_price=100.0,
        stop_loss=97.0,
        current_ticker_exposure=current_ticker,
    )

    assert result.approved
    headroom = max(0.0, settings.MAX_SINGLE_POSITION_EXPOSURE * equity - current_ticker)
    assert result.position_value <= headroom + 1e-9


# ---------------------------------------------------------------------------
# Test 8: Floor shares when fractional shares are disabled
# ---------------------------------------------------------------------------

def test_floors_shares_when_fractional_disabled(sizer_no_fractional):
    result = sizer_no_fractional.calculate_position_size(
        equity=100_000,
        entry_price=100.0,
        stop_loss=97.0,
    )

    if result.approved:
        # Shares must be a whole number when fractional is disabled
        assert result.shares == math.floor(result.shares), (
            f"Expected integer shares, got {result.shares}"
        )


# ---------------------------------------------------------------------------
# Test 9: Reject when final shares <= 0
# ---------------------------------------------------------------------------

def test_rejects_when_shares_become_zero_after_caps(sizer):
    """
    When portfolio headroom is exactly zero, position size reduces to 0 → rejected.
    """
    from config.settings import settings

    equity = 100_000
    # Fill up portfolio entirely
    max_portfolio = settings.MAX_PORTFOLIO_EXPOSURE * equity

    result = sizer.calculate_position_size(
        equity=equity,
        entry_price=100.0,
        stop_loss=97.0,
        current_portfolio_exposure=max_portfolio,  # zero headroom
    )

    assert not result.approved, (
        "Expected rejection when portfolio exposure is fully maxed out"
    )


# ---------------------------------------------------------------------------
# Test 10: max_loss_dollars must not exceed configured risk_dollars
# ---------------------------------------------------------------------------

def test_max_loss_does_not_exceed_risk_budget(sizer):
    """
    After all caps, max_loss = shares × stop_distance must be ≤ original risk_dollars.
    Caps can only reduce position size, never increase it.
    """
    result = sizer.calculate_position_size(
        equity=100_000,
        entry_price=100.0,
        stop_loss=97.0,
    )

    assert result.approved
    assert result.max_loss_dollars <= result.risk_dollars + 1e-9, (
        f"max_loss={result.max_loss_dollars:.2f} > risk_dollars={result.risk_dollars:.2f}"
    )


# ---------------------------------------------------------------------------
# Additional: metadata has expected keys
# ---------------------------------------------------------------------------

def test_metadata_contains_audit_fields(sizer):
    result = sizer.calculate_position_size(
        equity=100_000,
        entry_price=100.0,
        stop_loss=97.0,
    )

    assert result.approved
    required = {
        "raw_shares", "raw_position_value", "stop_distance",
        "max_by_single", "remaining_portfolio", "remaining_ticker",
    }
    missing = required - set(result.metadata)
    assert not missing, f"metadata missing keys: {missing}"
