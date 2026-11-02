"""Daily research replay, not a broker adapter. Dates are bar labels, not UTC instants."""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy.signals import SignalAction


def usable_bars(df):
    """Reject bad rows individually: a later bad print must not erase earlier fills."""
    cols = ["open", "high", "low", "close", "volume"]
    values = df[cols].apply(pd.to_numeric, errors="coerce")
    valid = (np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
             & (values.low <= values[["open", "close"]].min(axis=1))
             & (values.high >= values[["open", "close"]].max(axis=1))
             & ~df.index.duplicated(keep=False) & df.index.notna())
    if "tradable" in df:
        valid &= df["tradable"].eq(True).fillna(False)
    clean = df.loc[valid].sort_index().copy()
    clean[cols] = values.loc[clean.index]
    return clean


def assess(manager, portfolio, ticker, signal):
    return manager.evaluate_signal(
        signal=signal, equity=portfolio.equity,
        starting_day_equity=portfolio.day_start_equity,
        starting_week_equity=portfolio.week_start_equity, peak_equity=portfolio.peak_equity,
        current_portfolio_exposure=portfolio.portfolio_exposure,
        current_ticker_exposure=portfolio.ticker_exposure(ticker),
        open_positions_count=len(portfolio.positions), trades_today=portfolio.trades_today,
        has_existing_position=ticker in portfolio.positions)


@dataclass
class Pending:
    signal: object
    cap: float
    record: dict

    @classmethod
    def queue(cls, signal, decision, ticker, date):
        cap = decision.position_size.shares if decision.final_action == SignalAction.BUY else 0
        return cls(deepcopy(signal), cap, {
            "ticker": ticker, "action": decision.final_action.value,
            "observed_bar": date, "observation_phase": "close",
            "decision_bar": date, "decision_phase": "after_close",
            "eligibility": "first_subsequent_usable_open", "eligible_bar": None,
            "execution_bar": None, "status": "pending",
        })

    def attempt(self, bt, manager, portfolio, ticker, date, opening):
        record = self.record
        record["eligible_bar"] = date
        buy = self.signal.action == SignalAction.BUY
        fill = opening * (1 + (1 if buy else -1) * bt.slippage_bps / 10_000)
        if buy and (opening <= self.signal.stop_loss or
                    (self.signal.take_profit > 0 and fill >= self.signal.take_profit)):
            record["status"] = "cancelled_bracket"
            return
        signal = self.signal
        if buy:
            signal.entry_price = fill
        decision = assess(manager, portfolio, ticker, signal)
        if not decision.approved or decision.final_action.value != record["action"]:
            record["status"] = "cancelled_risk"
            return
        if buy:
            decision.position_size.shares = min(self.cap, decision.position_size.shares)
        shares = bt._execute(portfolio, ticker, date, signal, decision, opening)
        if not shares:
            record["status"] = "cancelled_unfilled"
            return
        if ticker in portfolio.positions:
            portfolio.positions[ticker].current_price = opening
        record.update(status="filled", execution_bar=date, execution_phase="open",
                      fill_price=fill, shares=shares)
