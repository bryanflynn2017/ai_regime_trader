"""
Event-driven daily-bar backtester.

Daily replay: mark at usable opens, apply gap stops, recheck and execute prior
decisions; check intrabar brackets (stop first when ambiguous); mark at close,
observe signals, then queue for the next usable open. No terminal liquidation.

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

from backtest.causal_fills import Pending, assess, usable_bars

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
    entry_fees_remaining: float = 0.0


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

    def book_exit(self, pos: _Position, shares: float, fill: float, fee: float,
                  date: pd.Timestamp, action: str, reason: str) -> None:
        """Record one simulated exit. Entry fees already left cash on purchase."""
        full_exit = shares == pos.shares
        entry_fee = (pos.entry_fees_remaining if full_exit else
                     pos.entry_fees_remaining * (shares / pos.shares))
        self.cash += shares * fill - fee
        self.trades.append({
            "ticker": pos.ticker, "action": action, "entry_date": pos.entry_date,
            "exit_date": date, "entry_price": pos.entry_price, "exit_price": fill,
            "shares": shares, "entry_fee": entry_fee, "exit_fee": fee,
            "pnl": (fill - pos.entry_price) * shares - entry_fee - fee,
            "exit_reason": reason,
        })
        if full_exit:
            del self.positions[pos.ticker]
        else:
            pos.entry_fees_remaining -= entry_fee
            pos.shares -= shares
            pos.current_price = fill

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

    Settings overrides supply research thresholds only. State paths are always
    disposable. For repeatable custom runs, supply orchestrator_factory(cfg, seed)
    returning a fresh strategy with a local RNG and no external mutable state.
    Declare strategy_id (a public lowercase slug) and strategy_version (e.g. "1.2.0")
    for custom runs. Bump the version when implementation or hidden configuration
    changes. Only the slug digest is exported; declarations are not code attestation.
    """

    def __init__(
        self,
        starting_equity:      float = 100_000.0,
        slippage_bps:         float = 5.0,
        commission_per_trade: float = 0.0,
        orchestrator:         Any   = None,
        settings_override:    Any   = None,
        *, orchestrator_factory=None, seed: int = 42,
        strategy_id: Optional[str] = None, strategy_version: Optional[str] = None,
    ) -> None:
        self.starting_equity      = starting_equity
        self.slippage_bps         = slippage_bps
        self.commission_per_trade = commission_per_trade

        from config.settings import settings
        from backtest.research_manifest import declared, strategy_identity

        if orchestrator is not None and orchestrator_factory is not None:
            raise ValueError("Choose orchestrator or orchestrator_factory")
        if type(seed) is not int:
            raise ValueError("Expected an integer research seed")
        self.seed = seed
        self._identity = strategy_identity(orchestrator_factory is not None or orchestrator is not None,
                                           strategy_id, strategy_version)
        self._declared = declared(settings if settings_override is None else settings_override)
        self._factory = orchestrator_factory
        self._prototype = orchestrator
        self._legacy_used = False

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
        from copy import deepcopy
        from backtest.research_manifest import manifest
        from config.settings import Settings
        from risk.risk_manager import RiskManager
        from strategy.strategy_orchestrator import StrategyOrchestrator

        with tempfile.TemporaryDirectory(prefix="bt_") as tmp:
            # Reuse already validated scalar settings, never caller credentials or paths.
            cfg = Settings.model_construct(**self._declared, STATE_DIR=tmp, LOG_DIR=tmp,
                                           RISK_LOCK_FILE=os.path.join(tmp, "RISK_LOCKED.txt"))
            if self._factory is not None:
                orch = self._factory(cfg.model_copy(deep=True), self.seed)
            elif self._prototype is not None:
                # deepcopy cannot isolate mutable closures or external RNGs.
                if self._legacy_used:
                    raise ValueError("Repeated custom runs require orchestrator_factory(cfg, seed)")
                self._legacy_used = True
                orch = deepcopy(self._prototype)
            else:
                orch = StrategyOrchestrator(settings_override=cfg)
            provenance = manifest(data_by_ticker, cfg, self._identity,
                                  self.seed if self._factory is not None else None)
            result = self._run(data_by_ticker, orch, RiskManager(settings_override=cfg))
            provenance["simulation"] = {
                "accounting_version": 2, "starting_equity": self.starting_equity,
                "slippage_bps": self.slippage_bps, "commission_per_trade": self.commission_per_trade,
                "warmup_bars": WARMUP_BARS, "reduce_fraction": REDUCE_FRACTION,
                "fill_model_version": "daily-open-gap-stop-v1",
                "timing_version": "completed-close-to-next-usable-open-v1",
            }
            result.config_snapshot.update(accounting_version=2, research_manifest=provenance)
            return result

    def _run(self, data_by_ticker, orch, risk_mgr) -> BacktestResult:
        if not data_by_ticker:
            return self._empty_result("No ticker data provided")

        # -- Pre-compute features for all tickers (O(N) once per ticker) --
        features: Dict[str, pd.DataFrame] = {}
        calendar = set()
        skipped = {}
        for ticker, df in data_by_ticker.items():
            try:
                from data.market_data import validate_ohlcv
                from data.feature_engineering import add_features
                clean = usable_bars(df)
                calendar.update(df.index.dropna())
                skipped[ticker] = len(df) - len(clean)
                validate_ohlcv(clean)
                features[ticker] = add_features(clean)
                logger.debug("Features computed for {} ({} bars)", ticker, len(df))
            except Exception as exc:
                logger.warning("Skipping {}: {}", ticker, exc)

        if not features:
            return self._empty_result("No valid ticker data after preprocessing")

        # A missing symbol is not a market-wide holiday.
        common = sorted(calendar)

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
        pending = {}
        orders = []

        # -- Main simulation loop --
        for i, date in enumerate(common):
            if i < WARMUP_BARS:
                portfolio.equity_curve[date] = self.starting_equity
                continue

            portfolio.begin_day(date)

            bars = {t: frame.loc[date] for t, frame in features.items() if date in frame.index}
            for t, pos in portfolio.positions.items():
                if t in bars:
                    pos.current_price = float(bars[t]["open"])
            # Resting protective gap stops precede queued market orders.
            for t in list(portfolio.positions):
                if t in bars and bars[t]["open"] <= portfolio.positions[t].stop_loss:
                    self._check_exits(portfolio, t, date, bars[t])
            for t in sorted(pending.keys() & bars.keys()):
                pending.pop(t).attempt(self, risk_mgr, portfolio, t, date, float(bars[t]["open"]))
            for t in list(portfolio.positions):
                if t in bars:
                    self._check_exits(portfolio, t, date, bars[t])
            for t, pos in portfolio.positions.items():
                if t in bars:
                    pos.current_price = float(bars[t]["close"])
            portfolio.update_peak()

            # Step 2: generate and evaluate signals for each ticker
            for ticker in sorted(features.keys()):
                if date not in features[ticker].index:
                    continue

                # Strict no-lookahead: only rows up to and including current date
                df_to_date = features[ticker].loc[:date]
                if len(df_to_date) <= WARMUP_BARS:
                    continue

                try:
                    from strategy.signals import SignalAction
                    signal = orch.generate_signal(ticker, df_to_date)

                    if signal.action != SignalAction.HOLD:
                        signals_log.append({
                            "date":   date,
                            "ticker": ticker,
                            "action": signal.action.value,
                            "reason": signal.reason,
                        })

                    if signal.action == SignalAction.HOLD:
                        continue

                    decision = assess(risk_mgr, portfolio, ticker, signal)

                    decisions_log.append({
                        "date":     date,
                        "ticker":   ticker,
                        "original": decision.original_action.value,
                        "final":    decision.final_action.value,
                        "approved": decision.approved,
                    })

                    if decision.approved and decision.final_action != SignalAction.HOLD:
                        order = Pending.queue(signal, decision, ticker, date)
                        pending[ticker] = order
                        orders.append(order.record)

                except Exception as exc:
                    logger.debug("Bar {} / {} error: {}", date, ticker, exc)

            portfolio.equity_curve[date] = portfolio.equity
            if bench_ticker in bars:
                cur_bench = float(bars[bench_ticker]["close"])
                if bench_start_price is None:
                    bench_start_price = cur_bench
                bench_curve[date] = self.starting_equity * cur_bench / bench_start_price

        for order in pending.values():
            order.record.update(status="expired_end_of_data", cancellation_bar=common[-1])

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
                "accounting_version":   2,  # Unversioned runs predate pooled basis/fee allocation.
                "warmup_bars":          WARMUP_BARS,
                "starting_equity":      self.starting_equity,
            },
            warnings  = portfolio.warnings,
            metadata  = {
                "orders": orders, "unusable_bars": skipped,
                "terminal_open_positions": len(portfolio.positions),
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
        A gap through the stop uses the lower opening price before adverse
        slippage. Targets receive no favorable gap-price improvement.
        """
        pos = portfolio.positions.get(ticker)
        if pos is None:
            return

        low  = float(bar["low"])
        high = float(bar["high"])
        slip = self.slippage_bps / 10_000

        # -- Stop-loss: fill below stop, conservative --
        if pos.stop_loss > 0 and low <= pos.stop_loss:
            fill = min(float(bar["open"]), pos.stop_loss) * (1.0 - slip)
            portfolio.book_exit(pos, pos.shares, fill, self.commission_per_trade,
                                date, "SELL", "stop_loss")
            portfolio.trades[-1]["execution_phase"] = "open" if bar["open"] <= pos.stop_loss else "intrabar"
        elif pos.take_profit > 0 and high >= pos.take_profit:
            fill = pos.take_profit * (1.0 - slip)
            portfolio.book_exit(pos, pos.shares, fill, self.commission_per_trade,
                                date, "SELL", "take_profit")
            portfolio.trades[-1]["execution_phase"] = "bar_trigger"

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
        opening:   float,
    ) -> Optional[float]:
        from strategy.signals import SignalAction

        slip = self.slippage_bps / 10_000

        action = decision.final_action
        if not decision.approved and action != SignalAction.HOLD:
            return

        if action == SignalAction.BUY:
            ps = decision.position_size
            if ps is None or not ps.approved or ps.shares <= 0:
                return

            fill = opening * (1.0 + slip)
            cost = fill * ps.shares + self.commission_per_trade

            if not cost <= portfolio.cash:
                portfolio.warnings.append(
                    f"{date}/{ticker}: insufficient cash (need ${cost:.0f}, "
                    f"have ${portfolio.cash:.0f})"
                )
                return

            pos = portfolio.positions.get(ticker)
            if pos is None:
                pos = _Position(ticker, ps.shares, fill, date, signal.stop_loss,
                                signal.take_profit, fill, self.commission_per_trade)
                portfolio.positions[ticker] = pos
            else:
                total = pos.shares + ps.shares
                pos.entry_price = (pos.entry_price * pos.shares + fill * ps.shares) / total
                pos.shares = total
                pos.entry_fees_remaining += self.commission_per_trade
                # Keep the original position date and the legacy latest-BUY exit levels.
                pos.stop_loss = signal.stop_loss
                pos.take_profit = signal.take_profit
                pos.current_price = fill
            portfolio.cash -= cost
            portfolio.trades_today += 1
            return ps.shares

        elif action in (SignalAction.SELL, SignalAction.REDUCE):
            pos = portfolio.positions.get(ticker)
            if pos is None:
                return
            shares = pos.shares if action == SignalAction.SELL else pos.shares * REDUCE_FRACTION
            fill = opening * (1.0 - slip)
            portfolio.book_exit(pos, shares, fill, self.commission_per_trade,
                                date, action.value, "signal")
            portfolio.trades[-1]["execution_phase"] = "open"
            portfolio.trades_today += 1
            return shares

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
