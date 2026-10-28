"""
Tests for risk/risk_manager.py.

Every test uses a temp lock-file path so no test writes to the real state/ dir.
Signal fixtures use valid BUY/SELL/REDUCE/HOLD templates.
"""
from __future__ import annotations

import pandas as pd
import pytest

from strategy.signals import RegimeLabel, Signal, SignalAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _buy(entry=100.0, stop=97.0, target=106.0, ticker="AAPL") -> Signal:
    return Signal(
        ticker=ticker,
        timestamp=pd.Timestamp.now(),
        action=SignalAction.BUY,
        confidence=0.75,
        reason="Test BUY",
        regime=RegimeLabel.BULL,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        max_position_value=0.0,
        metadata={"rejection_reasons": []},
    )


def _sell(ticker="AAPL") -> Signal:
    return Signal(
        ticker=ticker, timestamp=pd.Timestamp.now(),
        action=SignalAction.SELL, confidence=0.80,
        reason="Test SELL", regime=RegimeLabel.CRASH,
        entry_price=95.0, stop_loss=0.0, take_profit=0.0,
        max_position_value=0.0, metadata={"rejection_reasons": []},
    )


def _reduce(ticker="AAPL") -> Signal:
    return Signal(
        ticker=ticker, timestamp=pd.Timestamp.now(),
        action=SignalAction.REDUCE, confidence=0.65,
        reason="Test REDUCE", regime=RegimeLabel.BEAR,
        entry_price=98.0, stop_loss=0.0, take_profit=0.0,
        max_position_value=0.0, metadata={"rejection_reasons": []},
    )


def _hold(ticker="AAPL") -> Signal:
    return Signal(
        ticker=ticker, timestamp=pd.Timestamp.now(),
        action=SignalAction.HOLD, confidence=0.0,
        reason="Test HOLD", regime=RegimeLabel.NEUTRAL,
        entry_price=100.0, stop_loss=0.0, take_profit=0.0,
        max_position_value=0.0, metadata={"rejection_reasons": []},
    )


# ---------------------------------------------------------------------------
# Fixture: RiskManager wired to temp lock file
# ---------------------------------------------------------------------------

@pytest.fixture
def rm(tmp_path):
    from config.settings import Settings
    from risk.risk_manager import RiskManager
    s = Settings(RISK_LOCK_FILE=str(tmp_path / "RISK_LOCKED.txt"))
    return RiskManager(settings_override=s)


# Default safe equity snapshot (no circuit breakers fire)
_EQ = dict(
    equity=100_000,
    starting_day_equity=100_000,
    starting_week_equity=100_000,
    peak_equity=100_000,
)


# ---------------------------------------------------------------------------
# Test 1: HOLD approved when no circuit breaker active
# ---------------------------------------------------------------------------

def test_hold_approved_with_no_circuit_breaker(rm):
    decision = rm.evaluate_signal(_hold(), **_EQ)

    assert decision.approved is True
    assert decision.final_action == SignalAction.HOLD


# ---------------------------------------------------------------------------
# Test 2: BUY approved with valid signal and equity
# ---------------------------------------------------------------------------

def test_buy_approved_valid_signal(rm):
    decision = rm.evaluate_signal(_buy(), **_EQ)

    assert decision.approved is True
    assert decision.final_action == SignalAction.BUY
    assert decision.position_size is not None
    assert decision.position_size.approved is True


# ---------------------------------------------------------------------------
# Test 3: BUY blocked when stop_loss is missing / zero
# ---------------------------------------------------------------------------

def test_buy_blocked_missing_stop_loss(rm):
    signal = _buy(stop=0.0)
    decision = rm.evaluate_signal(signal, **_EQ)

    assert decision.approved is False
    assert decision.final_action == SignalAction.HOLD
    assert any("stop" in r.lower() for r in decision.rejection_reasons)


# ---------------------------------------------------------------------------
# Test 4: BUY blocked when risk lock file exists
# ---------------------------------------------------------------------------

def test_buy_blocked_when_lock_file_exists(tmp_path):
    lock = tmp_path / "RISK_LOCKED.txt"
    lock.write_text("Existing lock — test")

    from config.settings import Settings
    from risk.risk_manager import RiskManager
    s = Settings(RISK_LOCK_FILE=str(lock))
    risk_mgr = RiskManager(settings_override=s)

    decision = risk_mgr.evaluate_signal(_buy(), **_EQ)

    assert decision.approved is False
    assert decision.final_action == SignalAction.HOLD


# ---------------------------------------------------------------------------
# Test 5: BUY blocked when daily drawdown stop triggered (>= 3%)
# ---------------------------------------------------------------------------

def test_buy_blocked_daily_stop_circuit_breaker(rm):
    # 3.6% daily drawdown → daily STOP fires
    decision = rm.evaluate_signal(
        _buy(),
        equity=96_400,
        starting_day_equity=100_000,
        starting_week_equity=100_000,
        peak_equity=100_000,
    )

    assert decision.approved is False
    assert decision.final_action == SignalAction.HOLD


# ---------------------------------------------------------------------------
# Test 6: BUY size reduced when daily warning triggered (>= 2%, < 3%)
# ---------------------------------------------------------------------------

def test_buy_size_reduced_daily_warn(rm):
    # 2.6% daily drawdown → WARN → size_multiplier=0.5
    decision = rm.evaluate_signal(
        _buy(),
        equity=97_400,
        starting_day_equity=100_000,
        starting_week_equity=100_000,
        peak_equity=100_000,
    )

    assert decision.approved is True
    assert decision.final_action == SignalAction.BUY
    assert abs(decision.circuit_breaker_status.size_multiplier - 0.5) < 1e-9
    assert decision.position_size is not None

    # Shares must be smaller than without the warning
    decision_normal = rm.evaluate_signal(_buy(), **_EQ)
    assert decision.position_size.shares < decision_normal.position_size.shares


# ---------------------------------------------------------------------------
# Test 7: BUY blocked when max trades per day reached
# ---------------------------------------------------------------------------

def test_buy_blocked_max_trades_per_day(rm):
    from config.settings import settings as default_settings

    decision = rm.evaluate_signal(
        _buy(),
        **_EQ,
        trades_today=default_settings.MAX_TRADES_PER_DAY,  # exactly at limit
    )

    assert decision.approved is False
    assert any("trades" in r.lower() or "day" in r.lower() for r in decision.rejection_reasons)


# ---------------------------------------------------------------------------
# Test 8: BUY blocked when max open positions reached
# ---------------------------------------------------------------------------

def test_buy_blocked_max_open_positions(rm):
    from config.settings import settings as default_settings

    decision = rm.evaluate_signal(
        _buy(),
        **_EQ,
        open_positions_count=default_settings.MAX_OPEN_POSITIONS,
    )

    assert decision.approved is False
    assert any("position" in r.lower() for r in decision.rejection_reasons)


# ---------------------------------------------------------------------------
# Test 9: BUY blocked when portfolio exposure is maxed
# ---------------------------------------------------------------------------

def test_buy_blocked_portfolio_exposure_maxed(rm):
    from config.settings import settings as default_settings

    equity = 100_000
    max_portfolio = default_settings.MAX_PORTFOLIO_EXPOSURE * equity

    decision = rm.evaluate_signal(
        _buy(),
        **_EQ,
        current_portfolio_exposure=max_portfolio,
    )

    assert decision.approved is False
    assert any("portfolio" in r.lower() for r in decision.rejection_reasons)


# ---------------------------------------------------------------------------
# Test 10: BUY blocked when ticker exposure is maxed
# ---------------------------------------------------------------------------

def test_buy_blocked_ticker_exposure_maxed(rm):
    from config.settings import settings as default_settings

    equity = 100_000
    max_ticker = default_settings.MAX_SINGLE_POSITION_EXPOSURE * equity

    decision = rm.evaluate_signal(
        _buy(),
        **_EQ,
        current_ticker_exposure=max_ticker,
    )

    assert decision.approved is False
    assert any(
        "ticker" in r.lower() or "exposure" in r.lower()
        for r in decision.rejection_reasons
    )


# ---------------------------------------------------------------------------
# Test 11: REDUCE allowed when existing position exists
# ---------------------------------------------------------------------------

def test_reduce_allowed_with_existing_position(rm):
    decision = rm.evaluate_signal(
        _reduce(),
        **_EQ,
        has_existing_position=True,
    )

    assert decision.approved is True
    assert decision.final_action == SignalAction.REDUCE


# ---------------------------------------------------------------------------
# Test 12: SELL allowed when existing position exists
# ---------------------------------------------------------------------------

def test_sell_allowed_with_existing_position(rm):
    decision = rm.evaluate_signal(
        _sell(),
        **_EQ,
        has_existing_position=True,
    )

    assert decision.approved is True
    assert decision.final_action == SignalAction.SELL


# ---------------------------------------------------------------------------
# Test 13: REDUCE with no position converts to HOLD with warning
# ---------------------------------------------------------------------------

def test_reduce_no_position_converts_to_hold(rm):
    decision = rm.evaluate_signal(
        _reduce(),
        **_EQ,
        has_existing_position=False,
    )

    assert decision.approved is True
    assert decision.final_action == SignalAction.HOLD
    assert len(decision.warnings) > 0


# ---------------------------------------------------------------------------
# Test 14: SELL with no position converts to HOLD with warning
# ---------------------------------------------------------------------------

def test_sell_no_position_converts_to_hold(rm):
    decision = rm.evaluate_signal(
        _sell(),
        **_EQ,
        has_existing_position=False,
    )

    assert decision.approved is True
    assert decision.final_action == SignalAction.HOLD
    assert len(decision.warnings) > 0


# ---------------------------------------------------------------------------
# Test 15: LIVE_TRADING_ALLOWED=True blocks all BUY signals
# ---------------------------------------------------------------------------

def test_live_trading_allowed_blocks_buy(tmp_path):
    from config.settings import Settings
    from risk.risk_manager import RiskManager

    s = Settings(
        LIVE_TRADING_ALLOWED=True,
        RISK_LOCK_FILE=str(tmp_path / "RISK_LOCKED.txt"),
    )
    rm_live = RiskManager(settings_override=s)

    decision = rm_live.evaluate_signal(_buy(), **_EQ)

    assert decision.approved is False
    assert decision.final_action == SignalAction.HOLD


# ---------------------------------------------------------------------------
# Test 16: Exceptions fail closed (return blocking RiskDecision)
# ---------------------------------------------------------------------------

def test_exception_in_evaluation_fails_closed(tmp_path):
    """
    If the circuit breaker manager raises an exception, the risk manager
    must still return a blocking (not approved) RiskDecision.
    """
    from unittest.mock import MagicMock, patch
    from config.settings import Settings
    from risk.risk_manager import RiskManager

    s = Settings(RISK_LOCK_FILE=str(tmp_path / "RISK_LOCKED.txt"))
    rm_err = RiskManager(settings_override=s)

    # Monkey-patch the circuit breaker to raise
    rm_err._cb.evaluate = MagicMock(side_effect=RuntimeError("simulated CB failure"))

    decision = rm_err.evaluate_signal(_buy(), **_EQ)

    assert decision.approved is False
    assert decision.final_action == SignalAction.HOLD


# ---------------------------------------------------------------------------
# Test 17: RiskDecision includes non-empty rejection_reasons when blocked
# ---------------------------------------------------------------------------

def test_blocked_decision_has_rejection_reasons(rm):
    signal = _buy(stop=0.0)  # invalid stop → blocked
    decision = rm.evaluate_signal(signal, **_EQ)

    assert decision.approved is False
    assert isinstance(decision.rejection_reasons, list)
    assert len(decision.rejection_reasons) > 0
    assert all(isinstance(r, str) and len(r) > 0 for r in decision.rejection_reasons)


# ---------------------------------------------------------------------------
# Test 18: RiskDecision always contains a CircuitBreakerStatus
# ---------------------------------------------------------------------------

def test_decision_always_has_circuit_breaker_status(rm):
    from risk.circuit_breakers import CircuitBreakerStatus

    for signal in (_buy(), _sell(), _reduce(), _hold()):
        decision = rm.evaluate_signal(signal, **_EQ, has_existing_position=True)
        assert isinstance(decision.circuit_breaker_status, CircuitBreakerStatus), (
            f"Missing CircuitBreakerStatus for {signal.action.value}"
        )


# ---------------------------------------------------------------------------
# Test 19: Approved BUY includes a PositionSizingResult
# ---------------------------------------------------------------------------

def test_approved_buy_has_position_sizing_result(rm):
    from risk.position_sizing import PositionSizingResult

    decision = rm.evaluate_signal(_buy(), **_EQ)

    assert decision.approved is True
    assert isinstance(decision.position_size, PositionSizingResult)
    assert decision.position_size.approved is True
    assert decision.position_size.shares > 0


# ---------------------------------------------------------------------------
# Test 20: All final actions are valid SignalAction enum values
# ---------------------------------------------------------------------------

def test_all_final_actions_are_valid_enum_values(rm):
    scenarios = [
        (_buy(),    dict(**_EQ)),
        (_sell(),   dict(**_EQ, has_existing_position=True)),
        (_reduce(), dict(**_EQ, has_existing_position=True)),
        (_hold(),   dict(**_EQ)),
        (_buy(stop=0.0), dict(**_EQ)),  # blocked BUY
    ]

    valid_actions = set(SignalAction)

    for signal, kwargs in scenarios:
        decision = rm.evaluate_signal(signal, **kwargs)
        assert decision.final_action in valid_actions, (
            f"Unexpected final_action {decision.final_action!r} — "
            "must be a SignalAction member"
        )
