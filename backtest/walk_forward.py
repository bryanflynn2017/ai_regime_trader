"""
Walk-forward validation.

Splits the data into rolling train/test windows and reports performance
separately for each.  The combined test equity curve stitches the test
windows together end-to-end.

Window layout (default):
  train_days=252, test_days=126, step_days=126

  Window 1:  train [0, 252),    test [252, 378)
  Window 2:  train [126, 378),  test [378, 504)
  Window 3:  train [252, 504),  test [504, 630)
  ...

No-lookahead:
  Each test window's backtester only sees data up to the test window's end.
  Historical data prior to the test start is included ONLY for feature warmup
  (backward-looking rolling windows), never to influence forward-looking
  decisions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.backtester import Backtester, BacktestResult, WARMUP_BARS
from backtest.metrics import PerformanceMetrics, calculate_performance_metrics

try:
    from loguru import logger  # type: ignore
except ImportError:
    logger = logging.getLogger(__name__)  # type: ignore


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardWindowResult:
    """Results for a single train/test split."""
    train_start:  pd.Timestamp
    train_end:    pd.Timestamp
    test_start:   pd.Timestamp
    test_end:     pd.Timestamp
    train_result: BacktestResult
    test_result:  BacktestResult
    metadata:     Dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk-forward windows."""
    windows:                   List[WalkForwardWindowResult]
    combined_test_equity_curve: pd.Series     # stitched test windows
    combined_metrics:          PerformanceMetrics
    warnings:                  List[str]
    metadata:                  Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# WalkForwardValidator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """
    Runs rolling walk-forward validation on historical data.

    The test period of each window starts with the same `starting_equity`.
    Train and test performance are reported separately so you can detect
    strategy overfitting (train Sharpe >> test Sharpe).
    """

    def __init__(
        self,
        train_days:      int   = 252,
        test_days:       int   = 126,
        step_days:       int   = 126,
        starting_equity: float = 100_000.0,
        settings_override: Any = None,
    ) -> None:
        self.train_days       = train_days
        self.test_days        = test_days
        self.step_days        = step_days
        self.starting_equity  = starting_equity
        self._settings_override = settings_override

    def run(
        self,
        data_by_ticker: Dict[str, pd.DataFrame],
    ) -> WalkForwardResult:
        """
        Run all walk-forward windows and return aggregated results.

        Each window:
          - Trains (backtests) on data[train_start:train_end].
          - Tests  (backtests) on data[0:test_end], then extracts
            only the test-period equity for performance measurement.

        The extraction is necessary because the backtester needs full
        historical data for feature warmup (sma_200 needs 200 bars).
        """
        if not data_by_ticker:
            return self._empty_result("No data provided")

        # Collect all sorted dates across tickers
        all_date_sets = [set(df.index) for df in data_by_ticker.values()]
        all_dates = sorted(set.intersection(*all_date_sets) if all_date_sets else set())

        if len(all_dates) < self.train_days + self.test_days:
            return self._empty_result(
                f"Insufficient dates for walk-forward: "
                f"{len(all_dates)} available, "
                f"need >= {self.train_days + self.test_days}"
            )

        windows: List[WalkForwardWindowResult] = []
        combined_equity_segments: List[pd.Series] = []
        combined_trades: List[dict] = []
        all_warnings: List[str] = []

        start_idx = 0
        window_num = 0

        while start_idx + self.train_days + self.test_days <= len(all_dates):
            train_start = all_dates[start_idx]
            train_end   = all_dates[start_idx + self.train_days - 1]
            test_start  = all_dates[start_idx + self.train_days]
            test_end    = all_dates[min(
                start_idx + self.train_days + self.test_days - 1,
                len(all_dates) - 1
            )]

            window_num += 1
            logger.info(
                "Walk-forward window {}: train {} to {}, test {} to {}",
                window_num, train_start.date(), train_end.date(),
                test_start.date(), test_end.date(),
            )

            # -- Train backtest (data up to train_end) --
            train_data = {
                t: df.loc[:train_end] for t, df in data_by_ticker.items()
            }
            train_result = Backtester(
                starting_equity=self.starting_equity,
                settings_override=self._settings_override,
            ).run(train_data)
            all_warnings.extend(train_result.warnings)

            # -- Test backtest (data up to test_end, extract test-period slice) --
            full_data_to_test_end = {
                t: df.loc[:test_end] for t, df in data_by_ticker.items()
            }
            full_test_run = Backtester(
                starting_equity=self.starting_equity,
                settings_override=self._settings_override,
            ).run(full_data_to_test_end)
            all_warnings.extend(full_test_run.warnings)

            # Extract only test-period equity and trades
            test_equity = _extract_period(
                full_test_run.equity_curve, test_start, test_end
            )
            test_trades = [
                t for t in full_test_run.trades
                if pd.Timestamp(t.get("exit_date", test_end)) >= test_start
            ]

            # Re-normalise test equity to start at starting_equity
            if len(test_equity) > 0 and float(test_equity.iloc[0]) > 0:
                scale = self.starting_equity / float(test_equity.iloc[0])
                test_equity = test_equity * scale
                for t in test_trades:
                    t["pnl"] = t.get("pnl", 0.0) * scale

            test_metrics = calculate_performance_metrics(
                equity_curve=test_equity,
                trades=test_trades,
            )

            # Build a synthetic BacktestResult for the test period
            test_result = BacktestResult(
                starting_equity = self.starting_equity,
                final_equity    = float(test_equity.iloc[-1]) if len(test_equity) > 0 else self.starting_equity,
                equity_curve    = test_equity,
                trades          = test_trades,
                signals         = full_test_run.signals,
                risk_decisions  = full_test_run.risk_decisions,
                metrics         = test_metrics,
                benchmark_curve = _extract_period(full_test_run.benchmark_curve, test_start, test_end),
                config_snapshot = full_test_run.config_snapshot,
                warnings        = full_test_run.warnings,
            )

            windows.append(WalkForwardWindowResult(
                train_start  = train_start,
                train_end    = train_end,
                test_start   = test_start,
                test_end     = test_end,
                train_result = train_result,
                test_result  = test_result,
                metadata     = {"window": window_num},
            ))

            combined_equity_segments.append(test_equity)
            combined_trades.extend(test_trades)

            start_idx += self.step_days

        if not windows:
            return self._empty_result("No complete windows were generated")

        # Stitch test equity segments end-to-end
        stitched = self._stitch_equity(combined_equity_segments)

        combined_metrics = calculate_performance_metrics(
            equity_curve=stitched,
            trades=combined_trades,
        )

        logger.info(
            "Walk-forward complete: {} windows, combined test return={:.1f}% sharpe={:.2f}",
            len(windows),
            combined_metrics.total_return * 100,
            combined_metrics.sharpe_ratio,
        )

        return WalkForwardResult(
            windows                    = windows,
            combined_test_equity_curve = stitched,
            combined_metrics           = combined_metrics,
            warnings                   = all_warnings,
            metadata                   = {
                "n_windows":   len(windows),
                "train_days":  self.train_days,
                "test_days":   self.test_days,
                "step_days":   self.step_days,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stitch_equity(segments: List[pd.Series]) -> pd.Series:
        """
        Concatenate test equity segments into one continuous curve.
        Each segment restarts at starting_equity; stitching preserves
        relative performance within each window.
        """
        if not segments:
            return pd.Series(dtype=float)
        if len(segments) == 1:
            return segments[0]
        return pd.concat(segments)

    @staticmethod
    def _empty_result(reason: str) -> WalkForwardResult:
        from backtest.metrics import _empty_metrics
        return WalkForwardResult(
            windows=[],
            combined_test_equity_curve=pd.Series(dtype=float),
            combined_metrics=_empty_metrics(),
            warnings=[reason],
            metadata={"empty": True, "reason": reason},
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_period(
    series:     pd.Series,
    start_date: pd.Timestamp,
    end_date:   pd.Timestamp,
) -> pd.Series:
    """Return the sub-series in [start_date, end_date]."""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    return series.loc[
        (series.index >= start_date) & (series.index <= end_date)
    ]
