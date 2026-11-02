"""Daily-bar replay with synthetic prints, a scripted strategy and the real risk manager."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.backtester import Backtester, WARMUP_BARS
from strategy.signals import RegimeLabel, Signal, SignalAction


def series(rows):
    history = [(100, 101, 99, 100, 1000)] * WARMUP_BARS
    return pd.DataFrame(history + rows, columns=["open", "high", "low", "close", "volume"],
                        index=pd.bdate_range("2020-01-01", periods=len(history) + len(rows)))


FLAT = (100, 101, 99, 100, 1000)


def replay(df, actions, *, stop=85, target=130, fee=0, slip=0, other=None):
    def generate(ticker, history):
        date = history.index[-1]
        action = actions.get(date, "HOLD") if ticker == "ABC" else "HOLD"
        return Signal(ticker, date, SignalAction(action), .8, "script", RegimeLabel.BULL,
                      history.iloc[-1]["close"], stop, target, 0)
    bt = Backtester(commission_per_trade=fee, slippage_bps=slip,
                    orchestrator=SimpleNamespace(generate_signal=generate))
    return bt.run({"ABC": df, **({"OTHER": other} if other is not None else {})})


def test_open_fill_is_causal_even_when_later_extrema_or_signal_change():
    df = series([FLAT, (105, 110, 100, 107, 1000), FLAT])
    seen, eligible, last = df.index[-3:]
    actions = {seen: "BUY", last: "SELL"}
    run = replay(df, actions)
    buy, sell = run.metadata["orders"]
    assert (buy["observed_bar"], buy["decision_bar"], buy["eligible_bar"],
            buy["execution_bar"], buy["fill_price"]) == (seen, seen, eligible, eligible, 105)
    assert run.equity_curve[seen] == run.starting_equity
    assert sell["status"] == "expired_end_of_data" and sell["execution_bar"] is None
    assert not run.trades and run.metadata["terminal_open_positions"] == 1
    changed = df.copy()
    changed.loc[eligible, ["high", "low", "close"]] = [150, 75, 80]
    changed.loc[last, "volume"] = 0
    later = replay(changed, actions)
    assert later.metadata["orders"][0] == buy  # Same opening quantity despite intraday crash.
    assert later.trades[0]["exit_reason"] == "stop_loss"
    sim = run.config_snapshot["research_manifest"]["simulation"]
    assert sim["accounting_version"] == 2 and sim["fill_model_version"] and sim["timing_version"]


@pytest.mark.parametrize("bar,expected", [((80, 84, 75, 82, 1000), 80),
                                           ((100, 115, 90, 103, 1000), 95)])
def test_gap_and_ambiguous_bracket(bar, expected):
    df = series([FLAT, FLAT, bar])
    r = replay(df, {df.index[-3]: "BUY"}, stop=95, target=110, slip=10)
    trade, = r.trades
    assert trade["exit_reason"] == "stop_loss"
    assert trade["exit_price"] == pytest.approx(expected * .999)
    assert trade["exit_date"] == df.index[-1]


def test_pending_waits_for_next_usable_symbol_open():
    df = series([FLAT] * 8)
    dates = df.index[-8:]
    other = df.copy()
    df = df.drop(dates[1])
    df.loc[dates[2], "volume"] = 0
    df.loc[dates[3], "low"] = np.nan
    df.loc[dates[4], "high"] = 80
    df.loc[dates[5], "open"] = 0
    df["tradable"] = True
    df.loc[dates[6], "tradable"] = False
    r = replay(df, {dates[0]: "BUY"}, other=other)
    order, = r.metadata["orders"]
    assert order["execution_bar"] == dates[-1]
    assert r.metadata["unusable_bars"]["ABC"] == 5
    assert list(r.equity_curve.loc[dates[:-1]]) == [r.starting_equity] * 7


def test_quantity_cannot_exceed_close_authorization_and_invalid_bracket_cancels():
    df = series([FLAT, (99, 101, 98, 100, 1000)])
    actions = {df.index[-2]: "BUY"}
    order, = replay(df, actions, stop=90).metadata["orders"]
    # $500 risk / $10 original stop distance: a better open cannot enlarge the order.
    assert order["shares"] == 50
    df.iloc[-1] = (80, 82, 78, 80, 1000)
    r = replay(df, actions, stop=90)
    assert r.metadata["orders"][0]["status"] == "cancelled_bracket"
    assert r.final_equity == r.starting_equity


def test_slippage_and_both_sides_of_fees_reconcile_through_reduce_and_sell():
    df = series([FLAT, (102, 105, 100, 104, 1000), (106, 109, 105, 107, 1000),
                 (108, 111, 106, 109, 1000)])
    d = df.index[-4:]
    r = replay(df, {d[0]: "BUY", d[1]: "REDUCE", d[2]: "SELL"}, fee=2, slip=10)
    orders = r.metadata["orders"]
    assert [o["fill_price"] for o in orders] == pytest.approx([102*1.001, 106*.999, 108*.999])
    q = orders[0]["shares"]
    assert [o["shares"] for o in orders] == pytest.approx([q, q/2, q/2])
    expected = q * ((106+108)/2 * .999 - 102*1.001) - 6
    assert sum(t["entry_fee"] for t in r.trades) == 2
    assert sum(t["pnl"] for t in r.trades) == pytest.approx(expected)
    assert r.final_equity - r.starting_equity == pytest.approx(expected)
    assert r.metadata["terminal_open_positions"] == 0
