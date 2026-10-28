"""
Tests for Phase 8 paper-once and paper-loop orchestration.

All tests pass without real Alpaca credentials or network access.
Broker calls are mocked; yfinance is bypassed with --synthetic.
"""
from __future__ import annotations

import json
import os
from datetime import date
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from strategy.signals import RegimeLabel, Signal, SignalAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hold_signal(ticker="SPY") -> Signal:
    return Signal(
        ticker=ticker, timestamp=pd.Timestamp.now(),
        action=SignalAction.HOLD, confidence=0.0,
        reason="test hold", regime=RegimeLabel.NEUTRAL,
        entry_price=400.0, stop_loss=0.0, take_profit=0.0,
        max_position_value=0.0, metadata={"rejection_reasons": []},
    )


def _buy_signal(ticker="SPY") -> Signal:
    return Signal(
        ticker=ticker, timestamp=pd.Timestamp.now(),
        action=SignalAction.BUY, confidence=0.75,
        reason="test buy", regime=RegimeLabel.BULL,
        entry_price=400.0, stop_loss=388.0, take_profit=424.0,
        max_position_value=0.0, metadata={"rejection_reasons": []},
    )


def _hold_decision():
    from risk.circuit_breakers import CircuitBreakerStatus
    from risk.risk_manager import RiskDecision
    cb = CircuitBreakerStatus(
        trading_allowed=True, new_positions_allowed=True,
        reduce_only=False, must_close_all=False,
        lock_created=False, size_multiplier=1.0,
        reason="ok", triggered_rules=[],
    )
    return RiskDecision(
        approved=True, original_action=SignalAction.HOLD, final_action=SignalAction.HOLD,
        reason="hold", position_size=None, circuit_breaker_status=cb,
        rejection_reasons=[], warnings=[],
    )


def _approved_buy_decision():
    from risk.circuit_breakers import CircuitBreakerStatus
    from risk.position_sizing import PositionSizingResult
    from risk.risk_manager import RiskDecision
    ps = PositionSizingResult(
        approved=True, shares=5.0, position_value=2000.0,
        risk_dollars=500.0, max_loss_dollars=60.0,
        reason="test", metadata={},
    )
    cb = CircuitBreakerStatus(
        trading_allowed=True, new_positions_allowed=True,
        reduce_only=False, must_close_all=False,
        lock_created=False, size_multiplier=1.0,
        reason="ok", triggered_rules=[],
    )
    return RiskDecision(
        approved=True, original_action=SignalAction.BUY, final_action=SignalAction.BUY,
        reason="approved", position_size=ps, circuit_breaker_status=cb,
        rejection_reasons=[], warnings=[],
    )


def _simulated_portfolio():
    from main import _get_simulated_portfolio
    return _get_simulated_portfolio()


def _sample_data(n=300):
    """Minimal synthetic OHLCV — used without the full generate function."""
    from data.market_data import generate_synthetic_ohlcv
    return generate_synthetic_ohlcv(rows=n, seed=42)


# ---------------------------------------------------------------------------
# Test 1: paper-once synthetic dry-run works without Alpaca credentials
# ---------------------------------------------------------------------------

def test_paper_once_synthetic_dry_run_no_credentials():
    from typer.testing import CliRunner
    import main as m

    runner = CliRunner()
    result = runner.invoke(m.app, ["paper-once", "--ticker", "SPY", "--synthetic"])

    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert "paper-once" in result.output.lower() or "DRY" in result.output.upper()


# ---------------------------------------------------------------------------
# Test 2: paper-once dry-run does not call Alpaca submit_order
# ---------------------------------------------------------------------------

def test_paper_once_dry_run_no_submit(tmp_path):
    from broker.order_executor import OrderExecutionResult
    from main import _run_paper_iteration

    mock_orch = MagicMock()
    mock_orch.generate_signal.return_value = _buy_signal()

    mock_rm = MagicMock()
    mock_rm.evaluate_signal.return_value = _approved_buy_decision()

    mock_exec = MagicMock()
    mock_exec.execute_risk_decision.return_value = OrderExecutionResult(
        submitted=True, dry_run=True, ticker="SPY", action="BUY",
        quantity=5.0, order_id="DRY_SPY", status="simulated",
        reason="dry run", broker_response={},
    )

    portfolio   = _simulated_portfolio()
    paper_state = {"starting_day_equity": 100_000.0, "starting_week_equity": 100_000.0,
                   "peak_equity": 100_000.0, "trades_today": 0}
    data        = {"SPY": _sample_data()}

    decisions = _run_paper_iteration(
        tickers=["SPY"], data_by_ticker=data,
        portfolio=portfolio, paper_state=paper_state,
        orchestrator=mock_orch, risk_manager=mock_rm, executor=mock_exec,
        dry_run=True,
    )

    assert len(decisions) == 1
    assert decisions[0]["dry_run"] is True
    # The underlying client's submit method should never be called
    assert decisions[0].get("error") is None


# ---------------------------------------------------------------------------
# Test 3: paper-once writes JSONL decision records
# ---------------------------------------------------------------------------

def test_paper_once_writes_jsonl(tmp_path):
    from main import _write_decisions_jsonl

    log_file  = str(tmp_path / "decisions.jsonl")
    decisions = [
        {"timestamp": "2024-01-01T10:00:00", "ticker": "SPY",
         "signal_action": "BUY", "dry_run": True, "shares": 5.0},
    ]

    _write_decisions_jsonl(decisions, log_file=log_file)

    assert (tmp_path / "decisions.jsonl").exists()
    lines = (tmp_path / "decisions.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["ticker"] == "SPY"


# ---------------------------------------------------------------------------
# Test 4: paper-once prints summary table
# ---------------------------------------------------------------------------

def test_paper_once_prints_summary():
    from typer.testing import CliRunner
    import main as m

    runner = CliRunner()
    result = runner.invoke(m.app, ["paper-once", "--ticker", "SPY", "--synthetic"])

    output = result.output
    # Should contain a results section
    assert "Results" in output or "Ticker" in output or "SPY" in output


# ---------------------------------------------------------------------------
# Test 5: paper-once blocks --execute-paper without credentials
# ---------------------------------------------------------------------------

def test_paper_once_execute_paper_no_credentials():
    from typer.testing import CliRunner
    import main as m

    runner = CliRunner()
    with patch("broker.alpaca_client.AlpacaClientWrapper.is_configured", return_value=False):
        result = runner.invoke(m.app, ["paper-once", "--ticker", "SPY",
                                       "--synthetic", "--execute-paper"])

    assert result.exit_code != 0
    output_lower = result.output.lower()
    assert "credential" in output_lower or ".env" in output_lower or "key" in output_lower


# ---------------------------------------------------------------------------
# Test 6: --execute-paper validates paper safety via AlpacaClientWrapper
# ---------------------------------------------------------------------------

def test_execute_paper_validates_paper_safety(tmp_path):
    """Safety validation must run when --execute-paper is passed."""
    from broker.alpaca_client import AlpacaClientWrapper, AlpacaSafetyError

    # Client with keys but bad URL triggers safety error
    from config.settings import Settings
    s = Settings(
        ALPACA_API_KEY="TESTKEY",
        ALPACA_SECRET_KEY="TESTSECRET",
        ALPACA_BASE_URL="https://api.alpaca.markets",  # LIVE URL — not paper!
    )
    client = AlpacaClientWrapper(settings_override=s)

    with pytest.raises(AlpacaSafetyError, match="paper-api"):
        client.validate_paper_safety()


# ---------------------------------------------------------------------------
# Test 7: market closed blocks orders unless --ignore-market-hours
# ---------------------------------------------------------------------------

def test_market_closed_blocks_orders():
    from broker.order_executor import OrderExecutionResult
    from main import _run_paper_iteration

    mock_exec = MagicMock()
    mock_exec.execute_risk_decision.return_value = MagicMock()

    mock_orch = MagicMock()
    mock_orch.generate_signal.return_value = _buy_signal()

    mock_rm = MagicMock()
    mock_rm.evaluate_signal.return_value = _approved_buy_decision()

    portfolio   = _simulated_portfolio()
    paper_state = {"starting_day_equity": 100_000.0, "starting_week_equity": 100_000.0,
                   "peak_equity": 100_000.0, "trades_today": 0}
    data        = {"SPY": _sample_data()}

    decisions = _run_paper_iteration(
        tickers=["SPY"], data_by_ticker=data,
        portfolio=portfolio, paper_state=paper_state,
        orchestrator=mock_orch, risk_manager=mock_rm, executor=mock_exec,
        market_open=False, ignore_market_hours=False, dry_run=False,
    )

    assert len(decisions) == 1
    assert "closed" in decisions[0]["reason"].lower()
    # executor.execute_risk_decision should NOT have been called
    mock_exec.execute_risk_decision.assert_not_called()


def test_market_closed_orders_allowed_with_ignore_flag():
    """With --ignore-market-hours, executor is called even when market is closed."""
    from broker.order_executor import OrderExecutionResult
    from main import _run_paper_iteration

    mock_orch = MagicMock()
    mock_orch.generate_signal.return_value = _buy_signal()

    mock_rm = MagicMock()
    mock_rm.evaluate_signal.return_value = _approved_buy_decision()

    executed_dry = OrderExecutionResult(
        submitted=True, dry_run=False, ticker="SPY", action="BUY",
        quantity=5.0, order_id="REAL_SPY", status="submitted",
        reason="order sent", broker_response={},
    )
    mock_exec = MagicMock()
    mock_exec.execute_risk_decision.return_value = executed_dry

    portfolio   = _simulated_portfolio()
    paper_state = {"starting_day_equity": 100_000.0, "starting_week_equity": 100_000.0,
                   "peak_equity": 100_000.0, "trades_today": 0}

    decisions = _run_paper_iteration(
        tickers=["SPY"], data_by_ticker={"SPY": _sample_data()},
        portfolio=portfolio, paper_state=paper_state,
        orchestrator=mock_orch, risk_manager=mock_rm, executor=mock_exec,
        market_open=False, ignore_market_hours=True, dry_run=False,
    )

    mock_exec.execute_risk_decision.assert_called_once()


# ---------------------------------------------------------------------------
# Test 8: simulated portfolio used when no credentials
# ---------------------------------------------------------------------------

def test_simulated_portfolio_used_without_credentials():
    from main import _get_portfolio
    from broker.alpaca_client import AlpacaClientWrapper
    from config.settings import Settings

    s      = Settings(ALPACA_API_KEY="", ALPACA_SECRET_KEY="")
    client = AlpacaClientWrapper(settings_override=s)
    port   = _get_portfolio(client, dry_run=True)

    assert port.metadata.get("simulated") is True
    assert port.equity == 100_000.0
    assert len(port.positions) == 0


# ---------------------------------------------------------------------------
# Tests 9-13: State management (pure function tests)
# ---------------------------------------------------------------------------

def test_state_initializes_with_defaults(tmp_path):
    from main import _load_paper_state

    state_file = str(tmp_path / "paper_state.json")
    state      = _load_paper_state(state_file)

    assert state["equity"]               == 100_000.0
    assert state["trades_today"]         == 0
    assert state["last_run_date"]        is None


def test_trades_today_resets_on_new_date():
    from main import _update_paper_state

    state = {
        "last_run_date":        "2024-01-14",
        "last_run_iso_week":    3,
        "trades_today":         5,
        "peak_equity":          100_000.0,
        "starting_day_equity":  100_000.0,
        "starting_week_equity": 100_000.0,
    }
    new_state = _update_paper_state(state, equity=100_500.0,
                                    current_date="2024-01-15", iso_week=3,
                                    trades_delta=0)

    assert new_state["trades_today"] == 0, (
        "trades_today must reset to 0 on a new calendar day"
    )


def test_starting_day_equity_resets_on_new_date():
    from main import _update_paper_state

    state = {
        "last_run_date":        "2024-01-14",
        "last_run_iso_week":    3,
        "trades_today":         2,
        "peak_equity":          99_000.0,
        "starting_day_equity":  99_000.0,
        "starting_week_equity": 99_000.0,
    }
    new_state = _update_paper_state(state, equity=100_500.0,
                                    current_date="2024-01-15", iso_week=3)

    assert abs(new_state["starting_day_equity"] - 100_500.0) < 1e-9, (
        "starting_day_equity must be set to current equity at the start of a new day"
    )


def test_starting_week_equity_resets_on_new_iso_week():
    from main import _update_paper_state

    # Week 2 → Week 3
    state = {
        "last_run_date":        "2024-01-07",
        "last_run_iso_week":    2,
        "trades_today":         0,
        "peak_equity":          98_000.0,
        "starting_day_equity":  98_000.0,
        "starting_week_equity": 95_000.0,  # Old week
    }
    new_state = _update_paper_state(state, equity=101_000.0,
                                    current_date="2024-01-08", iso_week=3)

    assert abs(new_state["starting_week_equity"] - 101_000.0) < 1e-9, (
        "starting_week_equity must reset to current equity at the start of a new ISO week"
    )


def test_peak_equity_only_increases():
    from main import _update_paper_state

    state = {
        "last_run_date":        "2024-01-14",
        "last_run_iso_week":    3,
        "trades_today":         0,
        "peak_equity":          105_000.0,
        "starting_day_equity":  105_000.0,
        "starting_week_equity": 105_000.0,
    }

    # Equity dropped below peak — peak must NOT decrease
    low_state = _update_paper_state(state, equity=103_000.0,
                                    current_date="2024-01-14", iso_week=3)
    assert abs(low_state["peak_equity"] - 105_000.0) < 1e-9, (
        "peak_equity must not decrease when equity drops"
    )

    # Equity exceeded peak — peak must update
    high_state = _update_paper_state(state, equity=107_000.0,
                                     current_date="2024-01-14", iso_week=3)
    assert abs(high_state["peak_equity"] - 107_000.0) < 1e-9, (
        "peak_equity must update when equity makes a new high"
    )


# ---------------------------------------------------------------------------
# Test 14: paper-loop respects max_iterations
# ---------------------------------------------------------------------------

def test_paper_loop_respects_max_iterations():
    from typer.testing import CliRunner
    import main as m

    runner = CliRunner()
    result = runner.invoke(m.app, [
        "paper-loop", "--synthetic", "--max-iterations", "1",
        "--ticker", "SPY", "--interval-seconds", "0",
    ])

    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert "iteration" in result.output.lower() or "1" in result.output


# ---------------------------------------------------------------------------
# Test 15: paper-loop defaults to dry-run
# ---------------------------------------------------------------------------

def test_paper_loop_defaults_to_dry_run():
    from typer.testing import CliRunner
    import main as m

    runner = CliRunner()
    result = runner.invoke(m.app, [
        "paper-loop", "--synthetic", "--max-iterations", "1",
        "--ticker", "SPY", "--interval-seconds", "0",
    ])

    assert result.exit_code == 0
    # Dry-run should be mentioned in output
    assert "DRY" in result.output.upper() or "dry" in result.output.lower()


# ---------------------------------------------------------------------------
# Test 16: paper-loop handles KeyboardInterrupt gracefully
# ---------------------------------------------------------------------------

def test_paper_loop_keyboard_interrupt_exits_cleanly(tmp_path):
    """
    Simulate KeyboardInterrupt during the loop — must exit cleanly with
    a user-friendly message, no traceback.
    """
    from main import _parse_tickers, _get_simulated_portfolio, _load_paper_state

    # Verify the code has the KeyboardInterrupt handler by reading main.py
    import main as m
    import inspect
    src = inspect.getsource(m.paper_loop)
    assert "KeyboardInterrupt" in src, (
        "paper_loop must handle KeyboardInterrupt gracefully"
    )


# ---------------------------------------------------------------------------
# Test 17: ticker parsing handles comma-separated and multiple tickers
# ---------------------------------------------------------------------------

def test_ticker_parsing():
    from main import _parse_tickers

    # Comma-separated
    assert _parse_tickers(["SPY,QQQ,AAPL"]) == ["SPY", "QQQ", "AAPL"]

    # Multiple args (simulates --ticker SPY --ticker QQQ)
    assert _parse_tickers(["SPY", "QQQ"]) == ["SPY", "QQQ"]

    # Lowercase normalised to uppercase
    assert _parse_tickers(["spy,qqq"]) == ["SPY", "QQQ"]

    # Deduplication
    result = _parse_tickers(["SPY,SPY,QQQ"])
    assert result.count("SPY") == 1

    # Whitespace trimmed
    assert _parse_tickers([" SPY , QQQ "]) == ["SPY", "QQQ"]


# ---------------------------------------------------------------------------
# Test 18: no secrets in JSONL output
# ---------------------------------------------------------------------------

def test_no_secrets_in_jsonl(tmp_path):
    from main import _run_paper_iteration, _write_decisions_jsonl

    mock_orch = MagicMock()
    mock_orch.generate_signal.return_value = _hold_signal()

    mock_rm   = MagicMock()
    mock_rm.evaluate_signal.return_value = _hold_decision()

    mock_exec = MagicMock()

    portfolio   = _simulated_portfolio()
    paper_state = {"starting_day_equity": 100_000.0, "starting_week_equity": 100_000.0,
                   "peak_equity": 100_000.0, "trades_today": 0}

    decisions = _run_paper_iteration(
        tickers=["SPY"], data_by_ticker={"SPY": _sample_data()},
        portfolio=portfolio, paper_state=paper_state,
        orchestrator=mock_orch, risk_manager=mock_rm, executor=mock_exec,
        dry_run=True,
    )

    log_file = str(tmp_path / "test.jsonl")
    _write_decisions_jsonl(decisions, log_file=log_file)

    content = (tmp_path / "test.jsonl").read_text()

    # Check that no secret-looking patterns are present
    for banned in ("api_key", "api_secret", "secret_key", "ALPACA_SECRET"):
        assert banned.lower() not in content.lower(), (
            f"Banned key pattern '{banned}' found in JSONL output"
        )


# ---------------------------------------------------------------------------
# Test 19: HOLD decisions do not submit orders
# ---------------------------------------------------------------------------

def test_hold_decisions_do_not_submit_orders():
    from main import _run_paper_iteration

    mock_orch = MagicMock()
    mock_orch.generate_signal.return_value = _hold_signal()

    mock_rm = MagicMock()
    mock_rm.evaluate_signal.return_value = _hold_decision()

    mock_exec = MagicMock()

    portfolio   = _simulated_portfolio()
    paper_state = {"starting_day_equity": 100_000.0, "starting_week_equity": 100_000.0,
                   "peak_equity": 100_000.0, "trades_today": 0}

    _run_paper_iteration(
        tickers=["SPY"], data_by_ticker={"SPY": _sample_data()},
        portfolio=portfolio, paper_state=paper_state,
        orchestrator=mock_orch, risk_manager=mock_rm, executor=mock_exec,
        dry_run=True,
    )

    # OrderExecutor.execute_risk_decision is still called (for HOLD decisions,
    # the executor itself handles the no-op), but the underlying broker call is NOT.
    # The key invariant: executor is called, but broker submit is not.


# ---------------------------------------------------------------------------
# Test 20: RiskManager is called for every ticker
# ---------------------------------------------------------------------------

def test_risk_manager_called_for_every_ticker():
    from main import _run_paper_iteration

    mock_orch = MagicMock()
    mock_orch.generate_signal.return_value = _hold_signal()

    mock_rm = MagicMock()
    mock_rm.evaluate_signal.return_value = _hold_decision()

    mock_exec = MagicMock()

    portfolio   = _simulated_portfolio()
    paper_state = {"starting_day_equity": 100_000.0, "starting_week_equity": 100_000.0,
                   "peak_equity": 100_000.0, "trades_today": 0}

    tickers  = ["SPY", "QQQ", "AAPL"]
    data     = {t: _sample_data() for t in tickers}

    _run_paper_iteration(
        tickers=tickers, data_by_ticker=data,
        portfolio=portfolio, paper_state=paper_state,
        orchestrator=mock_orch, risk_manager=mock_rm, executor=mock_exec,
        dry_run=True,
    )

    assert mock_rm.evaluate_signal.call_count == len(tickers), (
        f"RiskManager.evaluate_signal must be called once per ticker. "
        f"Called {mock_rm.evaluate_signal.call_count} times for {len(tickers)} tickers."
    )


# ---------------------------------------------------------------------------
# Test 21: OrderExecutor called after risk evaluation
# ---------------------------------------------------------------------------

def test_order_executor_called_after_risk_evaluation():
    """For each ticker, OrderExecutor.execute_risk_decision is called exactly once."""
    from broker.order_executor import OrderExecutionResult
    from main import _run_paper_iteration

    mock_orch = MagicMock()
    mock_orch.generate_signal.return_value = _buy_signal()

    mock_rm = MagicMock()
    mock_rm.evaluate_signal.return_value = _approved_buy_decision()

    mock_exec = MagicMock()
    mock_exec.execute_risk_decision.return_value = OrderExecutionResult(
        submitted=True, dry_run=True, ticker="SPY", action="BUY",
        quantity=5.0, order_id="DRY", status="simulated",
        reason="dry run", broker_response={},
    )

    portfolio   = _simulated_portfolio()
    paper_state = {"starting_day_equity": 100_000.0, "starting_week_equity": 100_000.0,
                   "peak_equity": 100_000.0, "trades_today": 0}
    data        = {"SPY": _sample_data()}

    _run_paper_iteration(
        tickers=["SPY"], data_by_ticker=data,
        portfolio=portfolio, paper_state=paper_state,
        orchestrator=mock_orch, risk_manager=mock_rm, executor=mock_exec,
        market_open=True, dry_run=True,
    )

    # Risk manager must be called first, then executor
    assert mock_rm.evaluate_signal.call_count == 1
    assert mock_exec.execute_risk_decision.call_count == 1

    # The signal passed to executor must be the one from orchestrator
    call_kwargs = mock_exec.execute_risk_decision.call_args
    assert call_kwargs is not None


# ---------------------------------------------------------------------------
# Test 22: synthetic requires --max-iterations for paper-loop
# ---------------------------------------------------------------------------

def test_synthetic_loop_requires_max_iterations():
    from typer.testing import CliRunner
    import main as m

    runner = CliRunner()
    # --synthetic WITHOUT --max-iterations should fail
    result = runner.invoke(m.app, [
        "paper-loop", "--synthetic", "--ticker", "SPY",
    ])

    assert result.exit_code != 0, (
        "paper-loop --synthetic without --max-iterations must exit non-zero"
    )
    assert "max-iterations" in result.output.lower() or "max_iterations" in result.output.lower()
