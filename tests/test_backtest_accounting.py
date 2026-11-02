"""Synthetic fill-level invariants for legacy research accounting version 2."""
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from backtest.backtester import Backtester, WARMUP_BARS, _Portfolio
from strategy.signals import SignalAction


def decision(action, qty):
    return SimpleNamespace(approved=True, final_action=SignalAction(action),
                           position_size=SimpleNamespace(approved=True, shares=qty))


@pytest.fixture
def ledger(tmp_path):
    from config.settings import Settings
    bt = Backtester(commission_per_trade=2, slippage_bps=0, orchestrator=object(),
                    settings_override=Settings(LIVE_TRADING_ALLOWED=False,
                                               RISK_LOCK_FILE=str(tmp_path / "lock")))
    return bt, _Portfolio(1000)


def execute(ledger, action, qty, price, day):
    bt, portfolio = ledger
    bt._execute(portfolio, "ABC", pd.Timestamp(2020, 1, day),
                SimpleNamespace(stop_loss=price-10, take_profit=price+10),
                decision(action, qty), pd.Series({"close": price}))


def reconcile(portfolio, cash, holdings, equity):
    assert portfolio.cash == pytest.approx(cash)
    assert portfolio.portfolio_exposure == pytest.approx(holdings)
    assert portfolio.equity == pytest.approx(equity)
    realized = sum(t["pnl"] for t in portfolio.trades)
    unrealized = sum((p.current_price - p.entry_price) * p.shares - p.entry_fees_remaining
                     for p in portfolio.positions.values())
    assert realized + unrealized == pytest.approx(equity - portfolio.starting_equity)


def test_two_identical_buys_and_flat_round_trip(ledger):
    _, p = ledger
    execute(ledger, "BUY", 1, 100, 1)
    reconcile(p, 898, 100, 998)
    execute(ledger, "BUY", 1, 100, 2)
    reconcile(p, 796, 200, 996)
    assert p.positions["ABC"].shares == 2
    execute(ledger, "SELL", 0, 100, 3)
    reconcile(p, 994, 0, 994)
    assert p.trades[0]["pnl"] == -6


def test_weighted_basis_and_fee_allocation(ledger):
    _, p = ledger
    execute(ledger, "BUY", 3, 100, 1)
    reconcile(p, 698, 300, 998)
    execute(ledger, "BUY", 1, 120, 2)
    reconcile(p, 576, 480, 1056)
    pos = p.positions["ABC"]
    assert (pos.shares, pos.entry_price, pos.entry_fees_remaining) == (4, 105, 4)
    assert (pos.stop_loss, pos.take_profit) == (110, 130)
    execute(ledger, "REDUCE", 0, 115, 3)
    reconcile(p, 804, 230, 1034)
    assert (pos.shares, pos.entry_price, pos.entry_fees_remaining) == (2, 105, 2)
    execute(ledger, "SELL", 0, 110, 4)
    reconcile(p, 1022, 0, 1022)
    assert [t["pnl"] for t in p.trades] == [16, 6]
    assert [t["entry_fee"] for t in p.trades] == [2, 2]
    assert all(t["entry_date"] == pd.Timestamp("2020-01-01") for t in p.trades)


def test_unfunded_add_changes_only_warnings(ledger):
    _, p = ledger
    execute(ledger, "BUY", 1, 100, 1)
    before = deepcopy(vars(p))
    execute(ledger, "BUY", 1, 896.50, 2)  # Cash 898; fee pushes cost 50 cents over.
    assert len(p.warnings) == 1
    assert {**vars(p), "warnings": []} == before


@pytest.mark.parametrize("reason,low,high,price", [
    ("stop_loss", 89, 111, 90), ("take_profit", 95, 111, 110),
])
def test_trigger_exit_after_reduce(ledger, reason, low, high, price):
    bt, p = ledger
    execute(ledger, "BUY", 2, 100, 1)
    execute(ledger, "REDUCE", 0, 100, 2)
    reconcile(p, 896, 100, 996)
    bt._check_exits(p, "ABC", pd.Timestamp("2020-01-03"), pd.Series({"low": low, "high": high}))
    reconcile(p, 894 + price, 0, 894 + price)
    assert p.trades[-1]["exit_reason"] == reason
    assert p.trades[-1]["pnl"] == price - 103


def test_fractional_remainder_is_not_written_off(ledger):
    _, p = ledger
    execute(ledger, "BUY", .001, 100, 1)
    execute(ledger, "REDUCE", 0, 100, 2)
    reconcile(p, 995.95, .05, 996)
    execute(ledger, "SELL", 0, 100, 3)
    reconcile(p, 994, 0, 994)


@pytest.mark.parametrize("last_action,equity,pnls", [("HOLD", 99998, []), ("SELL", 99996, [-4])])
def test_terminal_mark_or_actual_exit(ledger, last_action, equity, pnls):
    bt, _ = ledger
    dates = pd.bdate_range("2020-01-01", periods=WARMUP_BARS + 2)
    df = pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume=100000.), index=dates)

    def signal(ticker, history):
        action = "BUY" if history.index[-1] == dates[-2] else last_action
        return SimpleNamespace(action=SignalAction(action), reason="fixture", stop_loss=90, take_profit=110)

    def approve(signal, **kwargs):
        d = decision(signal.action, 1)
        d.original_action = signal.action
        return d

    bt._orch = SimpleNamespace(generate_signal=signal)
    bt._risk_mgr = SimpleNamespace(evaluate_signal=approve)
    result = bt.run_single_ticker("ABC", df)
    assert result.config_snapshot["accounting_version"] == 2
    assert result.final_equity == result.equity_curve.iloc[-1] == equity
    assert [t["pnl"] for t in result.trades] == pnls
