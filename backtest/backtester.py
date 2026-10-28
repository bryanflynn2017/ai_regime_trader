"""
Event-driven daily-bar backtester.

Pipeline per bar (strict no-lookahead):
  1. Slice historical data up to and including the current bar.
  2. Check stop-loss / take-profit for all open positions.
  3. For each ticker, generate a signal using StrategyOrchestrator.
  4. Evaluate each signal through RiskManager.
  5. Simulate order execution (slippage, commission).
  6. Mark portfolio to market.
  7. Record equity, signal, and risk decision.

No-lookahead guarantee:
  Features are pre-computed once on the full dataset (backward-looking windows).
  At bar T the orchestrator receives features_df.iloc[:T+1] — it cannot see
  bar T+1 or later.  The regime classifier uses only the last row of that slice.

Safety:
  The backtester uses an isolated temporary directory for the risk lock file so
  it never writes to the real state/RISK_LOCKED.txt during simulation.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.metrics import PerformanceMetrics, calculate_performance_metrics

try:
    from loguru import logger  # type: ignore
except ImportError:
    logger = logging.getLogger(__name__)  # type: ignore

# Minimum bars of feature history required before the first trade
WARMUP_BARS: int = 252

# Default REDUCE fraction (sell this fraction of the position on REDUCE signal)
REDUCE_FRACTION: float = 0.50


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Full output of a completed backtest run.

    equity_curve:    pd.Series indexed by date; includes warmup bars (flat).
    trades:          List of closed-trade dicts with entry/exit/pnl fields.
    signals:         Log of all non-HOLD signals generated.
    risk_decisions:  Log of all non-HOLD risk decisions.
    metrics:         Aggregated PerformanceMetrics.
    benchmark_curve: Buy-and-hold equity curve for comparison.
    """
    starting_equity: float
    final_equity:    float
    equity_curve:    pd.Series
    trades:          List[Dict[str, Any]]
    signals:         List[Dict[str, Any]]
    risk_decisions:  List[Dict[str, Any]]
    metrics:         PerformanceMetrics
    benchmark_curve: pd.Series
    config_snapshot: Dict[str, Any]
    warnings:        List[str]
    metadata:        Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal portfolio state
# ---------------------------------------------------------------------------

@dataclass
class _Position:
    ticker:      str
    shares:      float
    entry_price: float
    entry_date:  Any           # pd.Timestamp
    stop_loss:   float
    take_profit: float
    current_price: float = 0.0


class _Portfolio:
    """Mutable portfolio state for the backtest simulation."""

    def __init__(self, starting_equity: float) -> None:
        self.starting_equity   = starting_equity
        self.cash              = starting_equity
        self.positions:        Dict[str, _Position] = {}
        self.trades:           List[Dict] = []
        self.equity_curve:     Dict = {}
        self.peak_equity       = starting_equity
        self.day_start_equity  = starting_equity
        self.week_start_equity = starting_equity
        self.trades_today      = 0
        self.warnings:         List[str] = []
        self._last_date        = None
        self._last_iso_week    = None

    @property
    def equity(self) -> float:
        pos_value = sum(
            p.shares * p.current_price for p in self.positions.values()
        )
        return self.cash + pos_value

    @property
    def portfolio_exposure(self) -> float:
        return sum(p.shares * p.current_price for p in self.positions.values())

    def ticker_exposure(self, ticker: str) -> float:
        pos = self.positions.get(ticker)
        return pos.shares * pos.current_price if pos else 0.0

    def begin_day(self, date: pd.Timestamp) -> None:
        """Update day/week tracking at the start of each new bar."""
        if self._last_date is None:
            self._last_date     = date
            self._last_iso_week = date.isocalendar()[1]
            return

        if date == self._last_date:
            return  # Same date (shouldn't happen with daily bars, but be safe)

        # New calendar day
        self.day_start_equity = self.equity
        self.trades_today     = 0

        # New ISO week
        new_week = date.isocalendar()[1]
        if new_week != self._last_iso_week or date.year != self._last_date.year:
            self.week_start_equity = self.equity
            self._last_iso_week    = new_week

        self._last_date = date

    def update_peak(self) -> None:
        self.peak_equity = max(self.peak_equity, self.equity)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Simulates the full trading pipeline on historical daily-bar data.

    Inject a custom `orchestrator` for deterministic testing:
        bt = Backtester(orchestrator=my_mock_orch)

    Use `settings_override` to control the risk-lock-file path so
    backtests never touch the real state/ directory.
    """

    def __init__(
        self,
        starting_equity:      float = 100_000.0,
        slippage_bps:         float = 5.0,
        commission_per_trade: float = 0.0,
        orchestrator:         Any   = None,
        settings_override:    Any   = None,
    ) -> None:
        self.starting_equity      = starting_equity
        self.slippage_bps         = slippage_bps
        self.commission_per_trade = commission_per_trade

        # Use an isolated temp directory so backtest never writes to state/
        if settings_override is None:
            from config.settings import Settings
            tmp = tempfile.mkdtemp(prefix="bt_")
            self._settings = Settings(
                STATE_DIR=tmp,
                RISK_LOCK_FILE=os.path.join(tmp, "RISK_LOCKED.txt"),
            )
        else:
            self._settings = settings_override

        from risk.risk_manager import RiskManager
        self._risk_mgr = RiskManager(settings_override=self._settings)

        if orchestrator is None:
            from strategy.strategy_orchestrator import StrategyOrchestrator
            self._orch = StrategyOrchestrator()
        else:
            self._orch = orchestrator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_single_ticker(
        self,
        ticker: str,
        df: pd.DataFrame,
    ) -> BacktestResult:
        """Convenience wrapper: run backtest on one ticker."""
        return self.run({ticker: df})

    def run(
        self,
        data_by_ticker: Dict[str, pd.DataFrame],
    ) -> BacktestResult:
        """
        Run the full backtest across all tickers.

        At each date, all tickers are processed in alphabetical order.
        Portfolio constraints (max positions, exposure) are applied
        across tickers simultaneously via the RiskManager.
        """
        if not data_by_ticker:
            return self._empty_result("No ticker data provided")

        # -- Pre-compute features for all tickers (O(N) once per ticker) --
        features: Dict[str, pd.DataFrame] = {}
        for ticker, df in data_by_ticker.items():
            try:
                from data.market_data import validate_ohlcv
                from data.feature_engineering import add_features
                validate_ohlcv(df)
                features[ticker] = add_features(df)
                logger.debug("Features computed for {} ({} bars)", ticker, len(df))
            except Exception as exc:
                logger.warning("Skipping {}: {}", ticker, exc)

        if not features:
            return self._empty_result("No valid ticker data after preprocessing")

        # -- Find common trading dates --
        date_sets = [set(df.index) for df in features.values()]
        common = sorted(set.intersection(*date_sets))

        if len(common) <= WARMUP_BARS:
            return self._empty_result(
                f"Insufficient common dates: {len(common)} (need > {WARMUP_BARS})"
            )

        logger.info(
            "Backtest starting: {} tickers, {} common dates, "
            "{} trading bars (warmup={})",
            len(features), len(common),
            len(common) - WARMUP_BARS, WARMUP_BARS,
        )

        # -- Benchmark: buy-and-hold SPY or first ticker --
        bench_ticker = "SPY" if "SPY" in features else sorted(features.keys())[0]
        bench_start_price: Optional[float] = None
        bench_curve: Dict = {}

        # -- Portfolio & signal logs --
        portfolio   = _Portfolio(self.starting_equity)
        signals_log = []
        decisions_log = []

        # -- Main simulation loop --
        for i, date in enumerate(common):
            if i < WARMUP_BARS:
                portfolio.equity_curve[date] = self.starting_equity
                continue

            portfolio.begin_day(date)

            # Step 1: check stop-loss and take-profit for all open positions
            for t in list(portfolio.positions.keys()):
                if t in features and date in features[t].index:
                    self._check_exits(portfolio, t, date, features[t].loc[date])

            # Step 2: generate and evaluate signals for each ticker
            for ticker in sorted(features.keys()):
                if date not in features[ticker].index:
                    continue

                # Strict no-lookahead: only rows up to and including current date
                df_to_date = features[ticker].loc[:date]

                try:
                    from strategy.signals import SignalAction
                    signal = self._orch.generate_signal(ticker, df_to_date)

                    if signal.action != SignalAction.HOLD:
                        signals_log.append({
                            "date":   date,
                            "ticker": ticker,
                            "action": signal.action.value,
                            "reason": signal.reason,
                        })

                    if signal.action == SignalAction.HOLD:
                        continue

                    decision = self._risk_mgr.evaluate_signal(
                        signal                     = signal,
                        equity                     = portfolio.equity,
                        starting_day_equity        = portfolio.day_start_equity,
                        starting_week_equity       = portfolio.week_start_equity,
                        peak_equity                = portfolio.peak_equity,
                        current_portfolio_exposure = portfolio.portfolio_exposure,
                        current_ticker_exposure    = portfolio.ticker_exposure(ticker),
                        open_positions_count       = len(portfolio.positions),
                        trades_today               = portfolio.trades_today,
                        has_existing_position      = ticker in portfolio.positions,
                    )

                    decisions_log.append({
                        "date":     date,
                        "ticker":   ticker,
                        "original": decision.original_action.value,
                        "final":    decision.final_action.value,
                        "approved": decision.approved,
                    })

                    bar = features[ticker].loc[date]
                    self._execute(portfolio, ticker, date, signal, decision, bar)

                except Exception as exc:
                    logger.debug("Bar {} / {} error: {}", date, ticker, exc)

            # Step 3: mark to market
            for ticker, pos in portfolio.positions.items():
                if ticker in features and date in features[ticker].index:
                    pos.current_price = float(features[ticker].loc[date, "close"])

            portfolio.equity_curve[date] = portfolio.equity
            portfolio.update_peak()

            # Benchmark
            if bench_start_price is None:
                bench_start_price = float(features[bench_ticker].loc[date, "close"])
            cur_bench = float(features[bench_ticker].loc[date, "close"])
            bench_curve[date] = self.starting_equity * cur_bench / bench_start_price

        # -- Build output series --
        equity_curve  = pd.Series(portfolio.equity_curve, name="equity",    dtype=float)
        benchmark_ser = pd.Series(bench_curve,             name="benchmark", dtype=float)

        metrics = calculate_performance_metrics(
            equity_curve    = equity_curve,
            trades          = portfolio.trades,
            benchmark_curve = benchmark_ser if len(benchmark_ser) > 1 else None,
        )

        logger.info(
            "Backtest complete: return={:.1f}% sharpe={:.2f} max_dd={:.1f}% trades={}",
            metrics.total_return * 100,
            metrics.sharpe_ratio,
            metrics.max_drawdown * 100,
            metrics.total_trades,
        )

        return BacktestResult(
            starting_equity = self.starting_equity,
            final_equity    = portfolio.equity,
            equity_curve    = equity_curve,
            trades          = portfolio.trades,
            signals         = signals_log,
            risk_decisions  = decisions_log,
            metrics         = metrics,
            benchmark_curve = benchmark_ser,
            config_snapshot = {
                "slippage_bps":         self.slippage_bps,
                "commission_per_trade": self.commission_per_trade,
                "warmup_bars":          WARMUP_BARS,
                "starting_equity":      self.starting_equity,
            },
            warnings  = portfolio.warnings,
            metadata  = {
                "tickers":       sorted(features.keys()),
                "total_bars":    len(common),
                "backtest_bars": len(common) - WARMUP_BARS,
                "bench_ticker":  bench_ticker,
            },
        )

    # ------------------------------------------------------------------
    # Private: stop / take-profit exits
    # ------------------------------------------------------------------

    def _check_exits(
        self,
        portfolio: _Portfolio,
        ticker:    str,
        date:      pd.Timestamp,
        bar:       pd.Series,
    ) -> None:
        """
        Check if a position's stop-loss or take-profit was hit on this bar.

        Conservative rule: stop-loss checked before take-profit.
        If the bar's low is at or below the stop, we fill at the stop
        (with adverse slippage).  Only if the stop was NOT hit do we
        check the take-profit.
        """
        pos = portfolio.positions.get(ticker)
        if pos is None:
            return

        low  = float(bar["low"])
        high = float(bar["high"])
        slip = self.slippage_bps / 10_000

        # -- Stop-loss: fill below stop, conservative --
        if pos.stop_loss > 0 and low <= pos.stop_loss:
            fill   = pos.stop_loss * (1.0 - slip)
            pnl    = (fill - pos.entry_price) * pos.shares - self.commission_per_trade
            proceeds = fill * pos.shares - self.commission_per_trade

            portfolio.cash += proceeds
            portfolio.trades.append({
                "ticker":      ticker,
                "action":      "SELL",
                "entry_date":  pos.entry_date,
                "exit_date":   date,
                "entry_price": pos.entry_price,
                "exit_price":  fill,
                "shares":      pos.shares,
                "pnl":         pnl,
                "exit_reason": "stop_loss",
            })
            del portfolio.positions[ticker]
            return

        # -- Take-profit --
        if pos.take_profit > 0 and high >= pos.take_profit:
            fill   = pos.take_profit * (1.0 - slip)
            pnl    = (fill - pos.entry_price) * pos.shares - self.commission_per_trade
            proceeds = fill * pos.shares - self.commission_per_trade

            portfolio.cash += proceeds
            portfolio.trades.append({
                "ticker":      ticker,
                "action":      "SELL",
                "entry_date":  pos.entry_date,
                "exit_date":   date,
                "entry_price": pos.entry_price,
                "exit_price":  fill,
                "shares":      pos.shares,
                "pnl":         pnl,
                "exit_reason": "take_profit",
            })
            del portfolio.positions[ticker]

    # ------------------------------------------------------------------
    # Private: order execution
    # ------------------------------------------------------------------

    def _execute(
        self,
        portfolio: _Portfolio,
        ticker:    str,
        date:      pd.Timestamp,
        signal:    Any,
        decision:  Any,
        bar:       pd.Series,
    ) -> None:
        from strategy.signals import SignalAction

        close = float(bar["close"])
        slip  = self.slippage_bps / 10_000

        action = decision.final_action
        if not decision.approved and action != SignalAction.HOLD:
            return

        if action == SignalAction.BUY:
            ps = decision.position_size
            if ps is None or not ps.approved or ps.shares <= 0:
                return

            fill = close * (1.0 + slip)
            cost = fill * ps.shares + self.commission_per_trade

            if cost > portfolio.cash + 1.0:  # 1$ tolerance for float rounding
                portfolio.warnings.append(
                    f"{date}/{ticker}: insufficient cash (need ${cost:.0f}, "
                    f"have ${portfolio.cash:.0f})"
                )
                return

            portfolio.cash -= cost
            portfolio.positions[ticker] = _Position(
                ticker      = ticker,
                shares      = ps.shares,
                entry_price = fill,
                entry_date  = date,
                stop_loss   = signal.stop_loss,
                take_profit = signal.take_profit,
                current_price = fill,
            )
            portfolio.trades_today += 1

        elif action == SignalAction.SELL:
            pos = portfolio.positions.get(ticker)
            if pos is None:
                return

            fill     = close * (1.0 - slip)
            proceeds = fill * pos.shares - self.commission_per_trade
            pnl      = (fill - pos.entry_price) * pos.shares - self.commission_per_trade

            portfolio.cash += proceeds
            portfolio.trades.append({
                "ticker":      ticker,
                "action":      "SELL",
                "entry_date":  pos.entry_date,
                "exit_date":   date,
                "entry_price": pos.entry_price,
                "exit_price":  fill,
                "shares":      pos.shares,
                "pnl":         pnl,
                "exit_reason": "signal",
            })
            del portfolio.positions[ticker]
            portfolio.trades_today += 1

        elif action == SignalAction.REDUCE:
            pos = portfolio.positions.get(ticker)
            if pos is None:
                return

            reduce_shares = pos.shares * REDUCE_FRACTION
            fill          = close * (1.0 - slip)
            proceeds      = fill * reduce_shares - self.commission_per_trade
            pnl           = (fill - pos.entry_price) * reduce_shares - self.commission_per_trade

            portfolio.cash += proceeds
            portfolio.trades.append({
                "ticker":      ticker,
                "action":      "REDUCE",
                "entry_date":  pos.entry_date,
                "exit_date":   date,
                "entry_price": pos.entry_price,
                "exit_price":  fill,
                "shares":      reduce_shares,
                "pnl":         pnl,
                "exit_reason": "signal",
            })
            pos.shares -= reduce_shares
            if pos.shares < 0.001:
                del portfolio.positions[ticker]
            else:
                pos.current_price = fill
            portfolio.trades_today += 1

    # ------------------------------------------------------------------
    # Private: empty result
    # ------------------------------------------------------------------

    def _empty_result(self, reason: str) -> BacktestResult:
        from backtest.metrics import _empty_metrics
        logger.warning("Backtester returning empty result: {}", reason)
        return BacktestResult(
            starting_equity = self.starting_equity,
            final_equity    = self.starting_equity,
            equity_curve    = pd.Series(dtype=float),
            trades          = [],
            signals         = [],
            risk_decisions  = [],
            metrics         = _empty_metrics(),
            benchmark_curve = pd.Series(dtype=float),
            config_snapshot = {},
            warnings        = [reason],
            metadata        = {"empty": True, "reason": reason},
        )
