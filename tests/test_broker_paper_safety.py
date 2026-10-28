"""
Tests for broker/alpaca_client.py, broker/order_executor.py,
broker/position_tracker.py, and related CLI commands.

ALL tests pass without real Alpaca credentials or network access.
Every Alpaca API call is mocked — no actual HTTP requests are made.

Safety invariants verified:
- Paper mode checks cannot be bypassed
- Secrets never appear in output, repr, or log-safe dicts
- dry_run mode never calls the broker
- Unapproved signals are always rejected
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from strategy.signals import RegimeLabel, Signal, SignalAction


# ---------------------------------------------------------------------------
# Helpers: signal and decision factories
# ---------------------------------------------------------------------------

def _buy_signal(ticker="AAPL", entry=150.0, stop=145.5, tp=159.0) -> Signal:
    return Signal(
        ticker=ticker,
        timestamp=pd.Timestamp.now(),
        action=SignalAction.BUY,
        confidence=0.75,
        reason="test BUY",
        regime=RegimeLabel.BULL,
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp,
        max_position_value=0.0,
        metadata={"rejection_reasons": []},
    )


def _sell_signal(ticker="AAPL") -> Signal:
    return Signal(
        ticker=ticker, timestamp=pd.Timestamp.now(),
        action=SignalAction.SELL, confidence=0.80,
        reason="test SELL", regime=RegimeLabel.CRASH,
        entry_price=145.0, stop_loss=0.0, take_profit=0.0,
        max_position_value=0.0, metadata={"rejection_reasons": []},
    )


def _hold_signal(ticker="AAPL") -> Signal:
    return Signal(
        ticker=ticker, timestamp=pd.Timestamp.now(),
        action=SignalAction.HOLD, confidence=0.0,
        reason="test HOLD", regime=RegimeLabel.NEUTRAL,
        entry_price=150.0, stop_loss=0.0, take_profit=0.0,
        max_position_value=0.0, metadata={"rejection_reasons": []},
    )


def _approved_buy_decision(shares=10.5):
    """Return a fully approved BUY RiskDecision with a position size."""
    from risk.position_sizing import PositionSizingResult
    from risk.circuit_breakers import CircuitBreakerStatus
    from risk.risk_manager import RiskDecision

    ps = PositionSizingResult(
        approved=True, shares=shares, position_value=shares * 150.0,
        risk_dollars=500.0, max_loss_dollars=shares * 4.5,
        reason="test position size", metadata={},
    )
    cb = CircuitBreakerStatus(
        trading_allowed=True, new_positions_allowed=True,
        reduce_only=False, must_close_all=False,
        lock_created=False, size_multiplier=1.0,
        reason="ok", triggered_rules=[],
    )
    return RiskDecision(
        approved=True,
        original_action=SignalAction.BUY,
        final_action=SignalAction.BUY,
        reason="approved",
        position_size=ps,
        circuit_breaker_status=cb,
        rejection_reasons=[],
        warnings=[],
    )


def _rejected_decision():
    """Return a blocked RiskDecision."""
    from risk.circuit_breakers import CircuitBreakerStatus
    from risk.risk_manager import RiskDecision

    cb = CircuitBreakerStatus(
        trading_allowed=False, new_positions_allowed=False,
        reduce_only=True, must_close_all=False,
        lock_created=False, size_multiplier=0.0,
        reason="test block", triggered_rules=["TEST"],
    )
    return RiskDecision(
        approved=False,
        original_action=SignalAction.BUY,
        final_action=SignalAction.HOLD,
        reason="blocked by test",
        position_size=None,
        circuit_breaker_status=cb,
        rejection_reasons=["test block"],
        warnings=[],
    )


def _hold_decision():
    """Return a HOLD RiskDecision."""
    from risk.circuit_breakers import CircuitBreakerStatus
    from risk.risk_manager import RiskDecision

    cb = CircuitBreakerStatus(
        trading_allowed=True, new_positions_allowed=True,
        reduce_only=False, must_close_all=False,
        lock_created=False, size_multiplier=1.0,
        reason="ok", triggered_rules=[],
    )
    return RiskDecision(
        approved=True,
        original_action=SignalAction.HOLD,
        final_action=SignalAction.HOLD,
        reason="hold",
        position_size=None,
        circuit_breaker_status=cb,
        rejection_reasons=[],
        warnings=[],
    )


def _paper_settings(**overrides):
    """Return Settings wired for paper trading with optional overrides."""
    from config.settings import Settings
    defaults = dict(
        PAPER_TRADING=True,
        LIVE_TRADING_ALLOWED=False,
        ALPACA_PAPER=True,
        ALPACA_BASE_URL="https://paper-api.alpaca.markets",
        ALPACA_API_KEY="",
        ALPACA_SECRET_KEY="",
    )
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# Test 1: is_configured() returns False without keys
# ---------------------------------------------------------------------------

def test_is_configured_false_without_keys():
    from broker.alpaca_client import AlpacaClientWrapper

    client = AlpacaClientWrapper(settings_override=_paper_settings(
        ALPACA_API_KEY="", ALPACA_SECRET_KEY=""
    ))
    assert client.is_configured() is False


def test_is_configured_true_with_keys():
    from broker.alpaca_client import AlpacaClientWrapper

    client = AlpacaClientWrapper(settings_override=_paper_settings(
        ALPACA_API_KEY="TEST_KEY_123", ALPACA_SECRET_KEY="TEST_SECRET_456"
    ))
    assert client.is_configured() is True


# ---------------------------------------------------------------------------
# Test 2: Missing credentials raises AlpacaConnectionError
# ---------------------------------------------------------------------------

def test_missing_credentials_raises_connection_error():
    from broker.alpaca_client import AlpacaClientWrapper, AlpacaConnectionError

    client = AlpacaClientWrapper(settings_override=_paper_settings(
        ALPACA_API_KEY="", ALPACA_SECRET_KEY=""
    ))

    with pytest.raises(AlpacaConnectionError, match="[Cc]redentials|configured"):
        client.get_account()  # Should raise before any network call


# ---------------------------------------------------------------------------
# Test 3: validate_paper_safety passes with correct paper settings
# ---------------------------------------------------------------------------

def test_validate_paper_safety_passes():
    from broker.alpaca_client import AlpacaClientWrapper

    client = AlpacaClientWrapper(settings_override=_paper_settings())
    # Must not raise
    client.validate_paper_safety()


# ---------------------------------------------------------------------------
# Test 4: validate_paper_safety fails if PAPER_TRADING is False
# ---------------------------------------------------------------------------

def test_validate_paper_safety_fails_if_paper_trading_false():
    from broker.alpaca_client import AlpacaClientWrapper, AlpacaSafetyError

    client = AlpacaClientWrapper(settings_override=_paper_settings(PAPER_TRADING=False))

    with pytest.raises(AlpacaSafetyError, match="PAPER_TRADING"):
        client.validate_paper_safety()


# ---------------------------------------------------------------------------
# Test 5: validate_paper_safety fails if LIVE_TRADING_ALLOWED is True
# ---------------------------------------------------------------------------

def test_validate_paper_safety_fails_if_live_trading_allowed():
    from broker.alpaca_client import AlpacaClientWrapper, AlpacaSafetyError

    client = AlpacaClientWrapper(settings_override=_paper_settings(LIVE_TRADING_ALLOWED=True))

    with pytest.raises(AlpacaSafetyError, match="LIVE_TRADING_ALLOWED"):
        client.validate_paper_safety()


# ---------------------------------------------------------------------------
# Test 6: validate_paper_safety fails if ALPACA_PAPER is False
# ---------------------------------------------------------------------------

def test_validate_paper_safety_fails_if_alpaca_paper_false():
    from broker.alpaca_client import AlpacaClientWrapper, AlpacaSafetyError

    client = AlpacaClientWrapper(settings_override=_paper_settings(ALPACA_PAPER=False))

    with pytest.raises(AlpacaSafetyError, match="ALPACA_PAPER"):
        client.validate_paper_safety()


# ---------------------------------------------------------------------------
# Test 7: validate_paper_safety fails if base URL is the live URL
# ---------------------------------------------------------------------------

def test_validate_paper_safety_fails_if_live_url():
    from broker.alpaca_client import AlpacaClientWrapper, AlpacaSafetyError

    client = AlpacaClientWrapper(settings_override=_paper_settings(
        ALPACA_BASE_URL="https://api.alpaca.markets"  # Live URL!
    ))

    with pytest.raises(AlpacaSafetyError, match="paper-api"):
        client.validate_paper_safety()


# ---------------------------------------------------------------------------
# Test 8: Secret key never appears in repr or log_safe_info
# ---------------------------------------------------------------------------

def test_secret_key_not_in_repr_or_log_safe():
    from broker.alpaca_client import AlpacaClientWrapper

    secret = "SUPER_SECRET_KEY_NEVER_PRINT"
    client = AlpacaClientWrapper(settings_override=_paper_settings(
        ALPACA_API_KEY="TESTKEY", ALPACA_SECRET_KEY=secret
    ))

    assert secret not in repr(client), "Secret must not appear in __repr__"
    assert secret not in str(client.log_safe_info()), "Secret must not appear in log_safe_info"

    info = client.log_safe_info()
    assert "secret_key" not in str(list(info.values())), (
        "Secret key value must not be in log_safe_info values"
    )
    assert info["secret_key_set"] is True, "log_safe_info should confirm the key is set"


# ---------------------------------------------------------------------------
# Test 9: OrderExecutor dry_run BUY does not call submit_bracket_buy
# ---------------------------------------------------------------------------

def test_dry_run_buy_does_not_call_broker():
    from broker.simulated_adapter import SimulatedBrokerAdapter
    from broker.order_executor import OrderExecutor

    # Use SimulatedBrokerAdapter instead of mocking
    broker = SimulatedBrokerAdapter()
    executor = OrderExecutor(broker_adapter=broker, dry_run=True)
    signal   = _buy_signal()
    decision = _approved_buy_decision(shares=10.0)

    result = executor.execute_risk_decision(signal, decision)

    # In dry_run mode, no orders are actually submitted
    assert result.submitted is True
    assert result.dry_run   is True
    assert result.action    == "BUY"
    assert result.quantity  == pytest.approx(10.0)
    assert "DRY" in result.status.upper() or "simulated" in result.status.lower()
    # Verify no real orders were logged in the adapter
    assert len(broker.list_positions()) == 0


# ---------------------------------------------------------------------------
# Test 10: OrderExecutor blocks unapproved risk decision
# ---------------------------------------------------------------------------

def test_executor_blocks_unapproved_decision():
    from broker.order_executor import OrderExecutor
    from broker.simulated_adapter import SimulatedBrokerAdapter

    broker = SimulatedBrokerAdapter()
    executor = OrderExecutor(broker_adapter=broker, dry_run=True)
    signal   = _buy_signal()
    decision = _rejected_decision()

    result = executor.execute_risk_decision(signal, decision)

    assert result.submitted is False
    assert "approved" in result.reason.lower() or "RiskDecision" in result.reason


# ---------------------------------------------------------------------------
# Test 11: OrderExecutor blocks HOLD final action
# ---------------------------------------------------------------------------

def test_executor_blocks_hold_action():
    from broker.order_executor import OrderExecutor
    from broker.simulated_adapter import SimulatedBrokerAdapter

    broker = SimulatedBrokerAdapter()
    executor = OrderExecutor(broker_adapter=broker, dry_run=True)
    signal   = _hold_signal()
    decision = _hold_decision()

    result = executor.execute_risk_decision(signal, decision)

    assert result.submitted is False
    assert result.action    == "HOLD"


# ---------------------------------------------------------------------------
# Test 12: OrderExecutor requires stop_loss for BUY
# ---------------------------------------------------------------------------

def test_executor_rejects_buy_without_stop_loss():
    from broker.order_executor import OrderExecutor
    from broker.simulated_adapter import SimulatedBrokerAdapter

    broker = SimulatedBrokerAdapter()
    executor = OrderExecutor(broker_adapter=broker, dry_run=True)
    bad_signal = _buy_signal(stop=0.0)  # No stop loss
    decision   = _approved_buy_decision()

    result = executor.execute_risk_decision(bad_signal, decision)

    assert result.submitted is False
    assert "stop" in result.reason.lower()


# ---------------------------------------------------------------------------
# Test 13: OrderExecutor requires take_profit for BUY
# ---------------------------------------------------------------------------

def test_executor_rejects_buy_without_take_profit():
    from broker.order_executor import OrderExecutor
    from broker.simulated_adapter import SimulatedBrokerAdapter

    broker = SimulatedBrokerAdapter()
    executor = OrderExecutor(broker_adapter=broker, dry_run=True)
    bad_signal = _buy_signal(tp=0.0)  # No take profit
    decision   = _approved_buy_decision()

    result = executor.execute_risk_decision(bad_signal, decision)

    assert result.submitted is False
    assert "take" in result.reason.lower() or "profit" in result.reason.lower()


# ---------------------------------------------------------------------------
# Test 14: OrderExecutor requires positive shares
# ---------------------------------------------------------------------------

def test_executor_rejects_buy_with_zero_shares():
    from broker.order_executor import OrderExecutor
    from broker.simulated_adapter import SimulatedBrokerAdapter

    broker = SimulatedBrokerAdapter()
    executor = OrderExecutor(broker_adapter=broker, dry_run=True)
    signal   = _buy_signal()
    decision = _approved_buy_decision(shares=0.0)  # Zero shares

    result = executor.execute_risk_decision(signal, decision)

    assert result.submitted is False
    assert "share" in result.reason.lower() or "0" in result.reason


# ---------------------------------------------------------------------------
# Test 15: OrderExecutor validates paper safety before real order
# ---------------------------------------------------------------------------

def test_executor_validates_safety_before_real_order():
    from broker.base import BrokerAdapter, BrokerError
    from broker.order_executor import OrderExecutor

    # Mock BrokerAdapter that raises BrokerError on submit_order
    mock_broker = MagicMock(spec=BrokerAdapter)
    mock_broker.is_paper = True
    mock_broker.submit_order.side_effect = BrokerError(
        "Paper safety check failed: LIVE_TRADING_ALLOWED is True"
    )

    executor = OrderExecutor(broker_adapter=mock_broker, dry_run=False)
    signal   = _buy_signal()
    decision = _approved_buy_decision()

    result = executor.execute_risk_decision(signal, decision)

    assert result.submitted is False
    assert "Paper safety" in result.reason or "safety" in result.reason.lower()


# ---------------------------------------------------------------------------
# Test 16: PositionTracker calculates portfolio exposure correctly
# ---------------------------------------------------------------------------

def test_position_tracker_portfolio_exposure():
    from broker.base import BrokerAdapter
    from broker.position_tracker import PositionTracker

    mock_broker = MagicMock(spec=BrokerAdapter)
    mock_broker.get_account.return_value = {
        "equity": "100000.00", "cash": "70000.00",
        "buying_power": "70000.00", "status": "ACTIVE",
    }
    mock_broker.list_positions.return_value = [
        {
            "symbol": "AAPL", "qty": "100", "market_value": "15000.00",
            "avg_entry_price": "145.00", "current_price": "150.00",
            "unrealized_pl": "500.00", "unrealized_plpc": "0.0333",
            "side": "long",
        },
        {
            "symbol": "MSFT", "qty": "50", "market_value": "15000.00",
            "avg_entry_price": "295.00", "current_price": "300.00",
            "unrealized_pl": "250.00", "unrealized_plpc": "0.0169",
            "side": "long",
        },
    ]

    tracker  = PositionTracker(broker_adapter=mock_broker)
    snapshot = tracker.get_portfolio_snapshot()

    # Total long value = 15000 + 15000 = 30000; equity = 100000
    assert abs(snapshot.portfolio_exposure - 0.30) < 1e-9
    assert snapshot.open_positions_count == 2
    assert snapshot.equity == 100_000.0


# ---------------------------------------------------------------------------
# Test 17: PositionTracker calculates per-ticker exposure correctly
# ---------------------------------------------------------------------------

def test_position_tracker_ticker_exposure():
    from broker.base import BrokerAdapter
    from broker.position_tracker import PositionTracker

    mock_broker = MagicMock(spec=BrokerAdapter)
    mock_broker.get_account.return_value = {
        "equity": "100000.00", "cash": "90000.00",
        "buying_power": "90000.00", "status": "ACTIVE",
    }
    mock_broker.list_positions.return_value = [
        {
            "symbol": "SPY", "qty": "50", "market_value": "10000.00",
            "avg_entry_price": "195.00", "current_price": "200.00",
            "unrealized_pl": "250.00", "unrealized_plpc": "0.025",
            "side": "long",
        }
    ]

    tracker  = PositionTracker(broker_adapter=mock_broker)
    exposure = tracker.get_current_ticker_exposure("SPY")

    # 10000 / 100000 = 0.10
    assert abs(exposure - 0.10) < 1e-9


# ---------------------------------------------------------------------------
# Test 18: PositionTracker handles no positions (empty portfolio)
# ---------------------------------------------------------------------------

def test_position_tracker_empty_portfolio():
    from broker.base import BrokerAdapter
    from broker.position_tracker import PositionTracker

    mock_broker = MagicMock(spec=BrokerAdapter)
    mock_broker.get_account.return_value = {
        "equity": "100000.00", "cash": "100000.00",
        "buying_power": "100000.00", "status": "ACTIVE",
    }
    mock_broker.list_positions.return_value = []

    tracker  = PositionTracker(broker_adapter=mock_broker)
    snapshot = tracker.get_portfolio_snapshot()

    assert snapshot.portfolio_exposure   == 0.0
    assert snapshot.open_positions_count == 0
    assert len(snapshot.positions)       == 0
    assert tracker.get_current_ticker_exposure("AAPL") == 0.0
    assert tracker.get_open_positions_count() == 0


# ---------------------------------------------------------------------------
# Test 19: PositionTracker handles string numeric fields from Alpaca
# ---------------------------------------------------------------------------

def test_position_tracker_converts_string_numerics():
    """Brokers return numeric fields as strings — must convert safely."""
    from broker.base import BrokerAdapter
    from broker.position_tracker import PositionTracker, _to_float

    # Test the helper directly
    assert _to_float("123.45")     == pytest.approx(123.45)
    assert _to_float("0")          == 0.0
    assert _to_float(None)         == 0.0
    assert _to_float("invalid")    == 0.0
    assert _to_float("-50.5")      == pytest.approx(-50.5)
    assert _to_float(42.0)         == 42.0  # Already a float

    # Via tracker
    mock_broker = MagicMock(spec=BrokerAdapter)
    mock_broker.get_account.return_value = {
        "equity": "99999.99", "cash": "50000.00",
        "buying_power": "50000.00", "status": "ACTIVE",
    }
    mock_broker.list_positions.return_value = [{
        "symbol": "TSLA", "qty": "5", "market_value": "1250.00",
        "avg_entry_price": "245.00", "current_price": "250.00",
        "unrealized_pl": "25.00", "unrealized_plpc": "0.0204",
        "side": "long",
    }]

    tracker  = PositionTracker(broker_adapter=mock_broker)
    snapshot = tracker.get_portfolio_snapshot()

    assert isinstance(snapshot.equity, float)
    pos = snapshot.positions["TSLA"]
    assert isinstance(pos.quantity,      float)
    assert isinstance(pos.current_price, float)
    assert pos.current_price == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# Test 20: test-connection command does not print secrets
# ---------------------------------------------------------------------------

def test_test_connection_no_secrets_in_output():
    """
    The test-connection command must never print the API secret.
    When credentials are absent it prints a helpful message and exits 0.
    """
    from typer.testing import CliRunner
    import main as main_module

    runner = CliRunner()
    # Isolate this test from workstation credentials and network access.
    with patch("broker.alpaca_client.AlpacaClientWrapper.is_configured", return_value=False):
        result = runner.invoke(main_module.app, ["test-connection"])

    # Exit code 0 — graceful handling of no credentials
    assert result.exit_code == 0, (
        f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
    )

    output = result.output.lower()
    # No secret key value should appear
    assert "alpaca_secret_key" not in output or "not found" in output or "missing" in output
    # Standard secrets pattern should not appear
    for banned in ("secret=", "secret_key=", "api_secret="):
        assert banned not in output, f"Banned string '{banned}' found in output"


def test_test_connection_missing_alpaca_py_exits_zero_without_broker_call():
    """
    Missing alpaca-py is an environment readiness issue, not a failure of
    paper-safety/no-secret behavior.
    """
    from typer.testing import CliRunner
    import main as main_module

    mock_client = MagicMock()
    mock_client.is_configured.return_value = True
    mock_client.is_available.return_value = False
    mock_client.validate_paper_safety.return_value = None

    with patch("broker.alpaca_client.AlpacaClientWrapper", return_value=mock_client):
        result = CliRunner().invoke(main_module.app, ["test-connection"])

    assert result.exit_code == 0, result.output
    assert "alpaca-py is not installed; live connection test skipped" in result.output
    mock_client.validate_paper_safety.assert_called_once()
    mock_client.get_account.assert_not_called()
    mock_client.get_clock.assert_not_called()

    output = result.output.lower()
    for banned in ("secret=", "secret_key=", "api_secret=", "discord.com/api/webhooks"):
        assert banned not in output


# ---------------------------------------------------------------------------
# Test 21: paper-dry-run command does not place a real order
# ---------------------------------------------------------------------------

def test_paper_dry_run_no_real_order():
    """
    The paper-dry-run command must complete without placing a real order.
    It should report [DRY RUN] in output.
    """
    from typer.testing import CliRunner
    import main as main_module

    runner = CliRunner()
    result = runner.invoke(main_module.app, ["paper-dry-run", "--synthetic"])

    assert result.exit_code == 0, (
        f"paper-dry-run exited {result.exit_code}. Output:\n{result.output}"
    )

    output_lower = result.output.lower()
    assert "dry" in output_lower or "no real order" in output_lower, (
        "paper-dry-run must mention dry run in output"
    )


# ---------------------------------------------------------------------------
# Test 22: All broker tests pass without real credentials (meta-test)
# ---------------------------------------------------------------------------

def test_all_broker_tests_require_no_real_credentials():
    """
    This test confirms the test module itself never needs ALPACA_API_KEY.
    Simply importing and instantiating the core broker classes must not raise.
    """
    from config.settings import Settings
    from broker.alpaca_client import AlpacaClientWrapper
    from broker.order_executor import OrderExecutor
    from broker.position_tracker import PositionTracker
    from broker.simulated_adapter import SimulatedBrokerAdapter

    s      = _paper_settings()
    client = AlpacaClientWrapper(settings_override=s)
    mock   = MagicMock()
    mock.validate_paper_safety.return_value = None
    broker = SimulatedBrokerAdapter()

    executor = OrderExecutor(broker_adapter=broker, dry_run=True)
    tracker  = PositionTracker(alpaca_client=mock)

    # All constructors work; no network calls made
    assert client is not None
    assert executor is not None
    assert tracker is not None
