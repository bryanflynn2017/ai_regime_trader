"""
Performance metrics for backtesting and live portfolio monitoring.

All functions are pure — no side effects, no I/O.

Edge cases handled explicitly:
  - Empty or single-element equity curves → return 0.0 / sensible defaults.
  - Flat equity (std == 0) → Sharpe/Sortino return 0.0.
  - No negative returns → Sortino returns 0.0 (undefined in practice).
  - Zero gross profit → profit_factor returns 0.0.
  - All values are in the same units as the inputs (dollars, fractions, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    """
    Full performance summary for a backtest or live run.

    Monetary values are in the same currency as starting_equity.
    Ratio values (total_return, sharpe, etc.) are dimensionless.
    Drawdown values are fractions (negative).
    """
    starting_equity:   float
    final_equity:      float
    total_return:      float          # (final − start) / start
    benchmark_return:  float          # same formula for benchmark
    excess_return:     float          # total_return − benchmark_return
    cagr:              float          # compound annual growth rate
    sharpe_ratio:      float
    sortino_ratio:     float
    max_drawdown:      float          # most negative peak-to-trough fraction
    win_rate:          float          # fraction of trades with pnl > 0
    average_win:       float          # mean pnl of winning trades
    average_loss:      float          # mean pnl of losing trades (≤ 0)
    profit_factor:     float          # gross_wins / |gross_losses|
    total_trades:      int
    exposure_time:     float          # fraction of bars with an open position
    best_trade:        float
    worst_trade:       float
    metadata:          Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def calculate_returns(equity_curve: pd.Series) -> pd.Series:
    """Return daily percent-change returns.  First element is always NaN."""
    return equity_curve.pct_change()


def calculate_cagr(
    equity_curve: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Compound Annual Growth Rate.

    Returns 0.0 for curves shorter than 2 bars or with non-positive starting value.
    """
    if len(equity_curve) < 2:
        return 0.0
    start = float(equity_curve.iloc[0])
    end   = float(equity_curve.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    years = len(equity_curve) / periods_per_year
    if years <= 0:
        return 0.0
    return float((end / start) ** (1.0 / years) - 1.0)


def calculate_sharpe(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Annualised Sharpe ratio (assumes zero risk-free rate by default).

    Returns 0.0 when std == 0 (constant or flat equity).
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    excess = clean - risk_free_rate / periods_per_year
    std = float(excess.std())
    if std == 0.0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def calculate_sortino(
    returns: pd.Series,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float:
    """
    Annualised Sortino ratio.

    Returns 0.0 when there are no negative returns or insufficient data.
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    downside = clean[clean < target_return] - target_return
    if len(downside) == 0:
        # No negative returns — technically infinite, but return 0.0 for safety
        return 0.0
    downside_std = float(downside.std())
    if downside_std == 0.0:
        return 0.0
    return float((clean.mean() - target_return) / downside_std * np.sqrt(periods_per_year))


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown as a negative fraction.

    Example: a 20% drawdown returns -0.20.
    Returns 0.0 for curves shorter than 2 bars.
    """
    if len(equity_curve) < 2:
        return 0.0
    rolling_peak = equity_curve.cummax()
    drawdown = (equity_curve - rolling_peak) / rolling_peak.replace(0, np.nan)
    return float(drawdown.min()) if not drawdown.isna().all() else 0.0


def calculate_trade_stats(trades: List[Dict]) -> Dict[str, Any]:
    """
    Summarise a list of trade records.

    Each trade dict must have a 'pnl' key.  Missing pnl is treated as 0.
    Returns sensible defaults for an empty list.
    """
    if not trades:
        return {
            "total_trades":  0,
            "win_rate":      0.0,
            "average_win":   0.0,
            "average_loss":  0.0,
            "profit_factor": 0.0,
            "best_trade":    0.0,
            "worst_trade":   0.0,
        }

    pnls = [float(t.get("pnl", 0.0)) for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_wins   = sum(wins)
    gross_losses = abs(sum(losses))

    return {
        "total_trades":  len(pnls),
        "win_rate":      len(wins) / len(pnls) if pnls else 0.0,
        "average_win":   float(np.mean(wins))   if wins   else 0.0,
        "average_loss":  float(np.mean(losses)) if losses else 0.0,
        "profit_factor": gross_wins / gross_losses if gross_losses > 0 else 0.0,
        "best_trade":    max(pnls) if pnls else 0.0,
        "worst_trade":   min(pnls) if pnls else 0.0,
    }


def calculate_performance_metrics(
    equity_curve:    pd.Series,
    trades:          List[Dict],
    benchmark_curve: Optional[pd.Series] = None,
    periods_per_year: int = 252,
) -> PerformanceMetrics:
    """
    Build a PerformanceMetrics from an equity curve and trade list.

    Args:
        equity_curve:     Daily equity indexed by date.
        trades:           List of completed trade dicts with 'pnl' key.
        benchmark_curve:  Optional buy-and-hold equity curve, same index.
        periods_per_year: Trading periods per year (252 for daily).
    """
    # Guard empty curve
    if equity_curve is None or len(equity_curve) < 2:
        return _empty_metrics()

    start_eq = float(equity_curve.iloc[0])
    end_eq   = float(equity_curve.iloc[-1])
    if start_eq <= 0:
        return _empty_metrics()

    total_ret = (end_eq - start_eq) / start_eq

    # Benchmark
    bench_ret = 0.0
    if benchmark_curve is not None and len(benchmark_curve) >= 2:
        b_start = float(benchmark_curve.iloc[0])
        b_end   = float(benchmark_curve.iloc[-1])
        if b_start > 0:
            bench_ret = (b_end - b_start) / b_start

    rets = calculate_returns(equity_curve)

    trade_stats = calculate_trade_stats(trades)

    # Exposure: fraction of bars where position was open
    # We estimate this from trades: bars_in_trade / total_bars
    total_bars = len(equity_curve) - 1
    in_trade_bars = 0
    for t in trades:
        entry = pd.Timestamp(t.get("entry_date", equity_curve.index[0]))
        exit_ = pd.Timestamp(t.get("exit_date",  equity_curve.index[-1]))
        dur = max(1, (exit_ - entry).days)
        in_trade_bars += dur
    exposure = min(1.0, in_trade_bars / total_bars) if total_bars > 0 else 0.0

    return PerformanceMetrics(
        starting_equity  = start_eq,
        final_equity     = end_eq,
        total_return     = total_ret,
        benchmark_return = bench_ret,
        excess_return    = total_ret - bench_ret,
        cagr             = calculate_cagr(equity_curve, periods_per_year),
        sharpe_ratio     = calculate_sharpe(rets, periods_per_year),
        sortino_ratio    = calculate_sortino(rets, periods_per_year),
        max_drawdown     = calculate_max_drawdown(equity_curve),
        win_rate         = trade_stats["win_rate"],
        average_win      = trade_stats["average_win"],
        average_loss     = trade_stats["average_loss"],
        profit_factor    = trade_stats["profit_factor"],
        total_trades     = trade_stats["total_trades"],
        best_trade       = trade_stats["best_trade"],
        worst_trade      = trade_stats["worst_trade"],
        exposure_time    = exposure,
        metadata         = {"periods_per_year": periods_per_year},
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _empty_metrics() -> PerformanceMetrics:
    """Return a zeroed PerformanceMetrics for edge cases."""
    return PerformanceMetrics(
        starting_equity=0.0, final_equity=0.0, total_return=0.0,
        benchmark_return=0.0, excess_return=0.0, cagr=0.0,
        sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0,
        win_rate=0.0, average_win=0.0, average_loss=0.0,
        profit_factor=0.0, total_trades=0, exposure_time=0.0,
        best_trade=0.0, worst_trade=0.0,
        metadata={"note": "empty equity curve"},
    )
