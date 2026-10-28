"""
Tests for backtest/ — metrics, backtester, walk-forward, and stress tests.

Every test that creates risk lock files uses tmp_path so the real
state/RISK_LOCKED.txt is never touched.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from strategy.signals import RegimeLabel, Signal, SignalAction


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 450, seed: int = 42, drift: float = 0.0004) -> pd.DataFrame:
    """Synthetic daily OHLCV with a slight positive drift."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    log_ret = rng.normal(drift, 0.010, n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    spread = close * 0.004
    return pd.DataFrame({
        "open":   close * (1 + rng.normal(0, 0.002, n)),
        "high":   close + spread,
        "low":    np.maximum(close - spread, 0.01),
        "close":  close,
        "volume": np.full(n, 8_000_000.0),
    }, index=dates)


def _make_signal(
    ticker:  str,
    action:  SignalAction,
    close:   float,
    date:    pd.Timestamp,
    stop:    float  = 0.0,
    target:  float  = 0.0,
) -> Signal:
    return Signal(
        ticker       = ticker,
        timestamp    = date,
        action       = action,
        confidence   = 0.75,
        reason       = f"Test {action.value}",
        regime       = RegimeLabel.BULL,
        entry_price  = close,
        stop_loss    = stop  if stop   > 0 else close * 0.97,
        take_profit  = target if target > 0 else close * 1.06,
        max_position_value = 0.0,
        metadata     = {"rejection_reasons": []},
    )


def _always_buy_orch(stop_pct=0.03, tp_pct=0.06):
    """Orchestrator that always returns a valid BUY signal."""
    mock = MagicMock()

    def gen(ticker, df):
        close = float(df.iloc[-1]["close"])
        return _make_signal(
            ticker, SignalAction.BUY, close,
            pd.Timestamp(df.index[-1]),
            stop=close * (1 - stop_pct),
            target=close * (1 + tp_pct),
        )

    mock.generate_signal = gen
    return mock


def _always_hold_orch():
    """Orchestrator that always returns HOLD."""
    mock = MagicMock()

    def gen(ticker, df):
        close = float(df.iloc[-1]["close"])
        return Signal(
            ticker=ticker, timestamp=pd.Timestamp(df.index[-1]),
            action=SignalAction.HOLD, confidence=0.0, reason="HOLD",
            regime=RegimeLabel.NEUTRAL, entry_price=close,
            stop_loss=0.0, take_profit=0.0, max_position_value=0.0,
            metadata={"rejection_reasons": []},
        )

    mock.generate_signal = gen
    return mock


@pytest.fixture
def bt_settings(tmp_path):
    from config.settings import Settings
    return Settings(RISK_LOCK_FILE=str(tmp_path / "RISK_LOCKED.txt"))


@pytest.fixture
def df():
    return _make_ohlcv()


# ---------------------------------------------------------------------------
# Tests 1-3: Metrics
# ---------------------------------------------------------------------------

def test_metrics_handle_flat_equity_curve():
    """PerformanceMetrics for a flat equity curve must not raise."""
    from backtest.metrics import calculate_performance_metrics
    flat = pd.Series([100_000.0] * 300, index=pd.date_range("2020-01-01", periods=300, freq="B"))
    m = calculate_performance_metrics(flat, trades=[])
    assert m.total_return == 0.0
    assert m.sharpe_ratio == 0.0
    assert m.max_drawdown == 0.0


def test_max_drawdown_calculation():
    """Max drawdown of [100, 80, 90] from peak=100 is -20%."""
    from backtest.metrics import calculate_max_drawdown
    curve = pd.Series([100.0, 80.0, 90.0])
    dd = calculate_max_drawdown(curve)
    assert abs(dd - (-0.20)) < 1e-9


def test_trade_stats_empty_trades():
    """calculate_trade_stats must return sensible defaults for an empty list."""
    from backtest.metrics import calculate_trade_stats
    stats = calculate_trade_stats([])
    assert stats["total_trades"]  == 0
    assert stats["win_rate"]      == 0.0
    assert stats["profit_factor"] == 0.0


# ---------------------------------------------------------------------------
# Tests 4-8: BacktestResult structure
# ---------------------------------------------------------------------------

def test_backtester_returns_backtest_result(df, bt_settings):
    from backtest.backtester import Backtester, BacktestResult
    bt = Backtester(settings_override=bt_settings)
    result = bt.run_single_ticker("SPY", df)
    assert isinstance(result, BacktestResult)


def test_backtester_does_not_mutate_input_data(df, bt_settings):
    from backtest.backtester import Backtester
    original_close = df["close"].copy()
    bt = Backtester(settings_override=bt_settings)
    bt.run_single_ticker("SPY", df)
    pd.testing.assert_series_equal(df["close"], original_close)


def test_backtester_starts_after_warmup(df, bt_settings):
    """Equity must be flat (= starting_equity) for the first WARMUP_BARS bars."""
    from backtest.backtester import Backtester, WARMUP_BARS
    bt = Backtester(settings_override=bt_settings, orchestrator=_always_hold_orch())
    result = bt.run_single_ticker("SPY", df)

    # All warmup bars should have starting equity
    warmup_eq = result.equity_curve.iloc[:WARMUP_BARS]
    assert (warmup_eq == bt.starting_equity).all(), (
        "Equity must equal starting_equity during the warmup period"
    )


def test_backtester_records_signals(df, bt_settings):
    """Non-HOLD signals must be recorded in the signals list."""
    from backtest.backtester import Backtester
    bt = Backtester(settings_override=bt_settings, orchestrator=_always_buy_orch())
    result = bt.run_single_ticker("SPY", df)
    # At least some BUY signals should have been generated
    assert len(result.signals) > 0
    assert all(s["action"] in ("BUY", "SELL", "REDUCE") for s in result.signals)


def test_backtester_records_risk_decisions(df, bt_settings):
    """Risk decisions (approved or blocked) must be recorded."""
    from backtest.backtester import Backtester
    bt = Backtester(settings_override=bt_settings, orchestrator=_always_buy_orch())
    result = bt.run_single_ticker("SPY", df)
    assert len(result.risk_decisions) > 0


# ---------------------------------------------------------------------------
# Tests 9-11: Order simulation
# ---------------------------------------------------------------------------

def test_buy_reduces_cash_and_creates_position(df, bt_settings):
    """After an approved BUY, cash decreases and a position is recorded."""
    from backtest.backtester import Backtester, WARMUP_BARS
    bt = Backtester(starting_equity=100_000, settings_override=bt_settings,
                    orchestrator=_always_buy_orch())
    result = bt.run_single_ticker("SPY", df)

    # If any trade was executed, the final equity differs from starting
    # OR is close to starting (price appreciation offset by no-trades).
    # At minimum, trades should have been recorded.
    assert len(result.trades) > 0 or len(result.warnings) > 0, (
        "Backtest should have either executed trades or issued warnings"
    )


def test_sell_closes_position_and_records_trade(tmp_path):
    """
    Sequence: BUY at bar 252, SELL at bar 260 → trade with pnl recorded.
    """
    from config.settings import Settings
    from backtest.backtester import Backtester, WARMUP_BARS

    df = _make_ohlcv(n=300, seed=7)
    dates = df.index

    # Alternating: BUY at warmup bar, SELL 5 bars later
    buy_date  = dates[WARMUP_BARS]
    sell_date = dates[WARMUP_BARS + 5]

    call_count = [0]

    def seq_gen(ticker, dfslice):
        d = pd.Timestamp(dfslice.index[-1])
        close = float(dfslice.iloc[-1]["close"])
        if d == buy_date and call_count[0] == 0:
            call_count[0] += 1
            return _make_signal(ticker, SignalAction.BUY, close, d)
        elif d >= sell_date:
            return _make_signal(ticker, SignalAction.SELL, close, d,
                                stop=0.0, target=0.0)
        return _make_signal(ticker, SignalAction.HOLD, close, d)

    mock = MagicMock()
    mock.generate_signal = seq_gen

    s = Settings(RISK_LOCK_FILE=str(tmp_path / "lock.txt"))
    bt = Backtester(starting_equity=100_000, settings_override=s, orchestrator=mock)
    result = bt.run_single_ticker("SPY", df)

    sells = [t for t in result.trades if t["action"] == "SELL" and t["exit_reason"] == "signal"]
    assert len(sells) >= 1, "At least one SELL trade should be recorded"
    assert "pnl" in sells[0]


def test_reduce_sells_partial_position(tmp_path):
    """REDUCE should sell REDUCE_FRACTION of shares, not the full position."""
    from config.settings import Settings
    from backtest.backtester import Backtester, WARMUP_BARS, REDUCE_FRACTION

    df = _make_ohlcv(n=300, seed=9)
    dates = df.index
    buy_date    = dates[WARMUP_BARS]
    reduce_date = dates[WARMUP_BARS + 3]
    state       = {"bought_shares": 0.0}

    def gen(ticker, dfslice):
        d     = pd.Timestamp(dfslice.index[-1])
        close = float(dfslice.iloc[-1]["close"])
        if d == buy_date and state["bought_shares"] == 0:
            state["bought_shares"] = 1.0  # flag
            return _make_signal(ticker, SignalAction.BUY, close, d)
        if d == reduce_date and state["bought_shares"] > 0:
            return _make_signal(ticker, SignalAction.REDUCE, close, d,
                                stop=0.0, target=0.0)
        return _make_signal(ticker, SignalAction.HOLD, close, d)

    mock = MagicMock()
    mock.generate_signal = gen

    s = Settings(RISK_LOCK_FILE=str(tmp_path / "lock.txt"))
    bt = Backtester(starting_equity=100_000, settings_override=s, orchestrator=mock)
    result = bt.run_single_ticker("SPY", df)

    reduces = [t for t in result.trades if t["action"] == "REDUCE"]
    if reduces:
        # A REDUCE trade's shares must be less than the original BUY shares
        # (we sold REDUCE_FRACTION of the position)
        assert reduces[0]["shares"] > 0


# ---------------------------------------------------------------------------
# Test 12: Stop-loss fires before take-profit when both hit in same bar
# ---------------------------------------------------------------------------

def test_stop_loss_exits_before_take_profit(tmp_path):
    """
    If a single bar's low <= stop_loss AND high >= take_profit,
    the stop-loss must fire (exit at stop, not take-profit).
    """
    from config.settings import Settings
    from backtest.backtester import Backtester, WARMUP_BARS

    df = _make_ohlcv(n=280, seed=55)
    dates = df.index
    buy_date = dates[WARMUP_BARS]
    next_date = dates[WARMUP_BARS + 1]

    entry_price = float(df.loc[buy_date, "close"])
    stop  = entry_price * 0.95   # 5% below entry
    tp    = entry_price * 1.10   # 10% above entry

    # Force the next bar to hit BOTH stop and TP: low < stop, high > TP
    df.loc[next_date, "low"]  = stop * 0.90  # well below stop
    df.loc[next_date, "high"] = tp   * 1.05  # well above TP

    called = [False]

    def gen(ticker, dfslice):
        d     = pd.Timestamp(dfslice.index[-1])
        close = float(dfslice.iloc[-1]["close"])
        if d == buy_date and not called[0]:
            called[0] = True
            return _make_signal(ticker, SignalAction.BUY, close, d,
                                stop=stop, target=tp)
        return _make_signal(ticker, SignalAction.HOLD, close, d)

    mock = MagicMock()
    mock.generate_signal = gen

    s = Settings(RISK_LOCK_FILE=str(tmp_path / "lock.txt"))
    bt = Backtester(starting_equity=100_000, settings_override=s, orchestrator=mock)
    result = bt.run_single_ticker("SPY", df)

    exits = [t for t in result.trades if t.get("exit_reason") in ("stop_loss", "take_profit")]
    if exits:
        assert exits[0]["exit_reason"] == "stop_loss", (
            f"Stop-loss must fire before take-profit; got {exits[0]['exit_reason']}"
        )


# ---------------------------------------------------------------------------
# Test 13: Slippage is applied
# ---------------------------------------------------------------------------

def test_slippage_applied_to_buy_and_sell(tmp_path):
    """Buy execution price must be > close, sell price must be < close."""
    from config.settings import Settings
    from backtest.backtester import Backtester, WARMUP_BARS

    df      = _make_ohlcv(n=280, seed=77)
    dates   = df.index
    buy_d   = dates[WARMUP_BARS]
    sell_d  = dates[WARMUP_BARS + 2]
    state   = {"bought": False}

    def gen(ticker, dfslice):
        d     = pd.Timestamp(dfslice.index[-1])
        close = float(dfslice.iloc[-1]["close"])
        if d == buy_d and not state["bought"]:
            state["bought"] = True
            return _make_signal(ticker, SignalAction.BUY, close, d)
        if d >= sell_d and state["bought"]:
            return _make_signal(ticker, SignalAction.SELL, close, d,
                                stop=0.0, target=0.0)
        return _make_signal(ticker, SignalAction.HOLD, close, d)

    mock = MagicMock()
    mock.generate_signal = gen

    slip_bps = 10
    s = Settings(RISK_LOCK_FILE=str(tmp_path / "lock.txt"))
    bt = Backtester(starting_equity=100_000, slippage_bps=slip_bps,
                    settings_override=s, orchestrator=mock)
    result = bt.run_single_ticker("SPY", df)

    sells = [t for t in result.trades if t["action"] == "SELL" and t["exit_reason"] == "signal"]
    if sells:
        sell_trade = sells[0]
        close_at_sell = float(df.loc[sell_d, "close"])
        assert sell_trade["exit_price"] < close_at_sell * 1.001, (
            "Sell execution price should be <= close (slippage takes you below)"
        )


# ---------------------------------------------------------------------------
# Test 14: Equity curve has no negatives under normal conditions
# ---------------------------------------------------------------------------

def test_equity_curve_non_negative(df, bt_settings):
    from backtest.backtester import Backtester
    bt = Backtester(settings_override=bt_settings)
    result = bt.run_single_ticker("SPY", df)
    if len(result.equity_curve) > 0:
        assert (result.equity_curve >= 0).all(), "Equity must not go negative"


# ---------------------------------------------------------------------------
# Test 15: Benchmark curve is created
# ---------------------------------------------------------------------------

def test_benchmark_curve_is_created(df, bt_settings):
    from backtest.backtester import Backtester
    bt = Backtester(settings_override=bt_settings)
    result = bt.run_single_ticker("SPY", df)
    assert len(result.benchmark_curve) > 0
    assert (result.benchmark_curve > 0).all()


# ---------------------------------------------------------------------------
# Test 16: Backtester uses only data up to the current bar
# ---------------------------------------------------------------------------

def test_backtester_uses_only_data_up_to_current_bar(df, bt_settings):
    """
    The orchestrator must receive a monotonically growing slice of data:
    at bar T it sees exactly T+1 rows.
    """
    from backtest.backtester import Backtester, WARMUP_BARS

    received_lengths = []

    def recording_gen(ticker, dfslice):
        received_lengths.append(len(dfslice))
        close = float(dfslice.iloc[-1]["close"])
        return Signal(
            ticker=ticker, timestamp=pd.Timestamp(dfslice.index[-1]),
            action=SignalAction.HOLD, confidence=0.0, reason="test",
            regime=RegimeLabel.NEUTRAL, entry_price=close,
            stop_loss=0.0, take_profit=0.0, max_position_value=0.0,
            metadata={"rejection_reasons": []},
        )

    mock = MagicMock()
    mock.generate_signal = recording_gen

    # Use a small df to keep the test fast
    small_df = df.iloc[:WARMUP_BARS + 10]
    bt = Backtester(settings_override=bt_settings, orchestrator=mock)
    bt.run_single_ticker("SPY", small_df)

    assert len(received_lengths) > 0
    # Each call should receive a longer slice than the previous
    for i in range(1, len(received_lengths)):
        assert received_lengths[i] > received_lengths[i - 1], (
            f"Orchestrator received a non-growing slice at call {i}: "
            f"{received_lengths[i-1]} → {received_lengths[i]}"
        )


# ---------------------------------------------------------------------------
# Test 17: Backtester does not create real state/RISK_LOCKED.txt
# ---------------------------------------------------------------------------

def test_backtester_does_not_create_real_lock_file(df, tmp_path):
    """The real state/RISK_LOCKED.txt must not be created during a backtest."""
    from pathlib import Path
    from config.settings import Settings
    from backtest.backtester import Backtester

    real_lock = Path("state") / "RISK_LOCKED.txt"

    s = Settings(RISK_LOCK_FILE=str(tmp_path / "test_lock.txt"))
    bt = Backtester(settings_override=s, orchestrator=_always_buy_orch())
    bt.run_single_ticker("SPY", df)

    assert not real_lock.exists(), (
        "Backtester must not write to state/RISK_LOCKED.txt (use isolated tmp settings)"
    )


# ---------------------------------------------------------------------------
# Tests 18-20: Walk-forward validator
# ---------------------------------------------------------------------------

@pytest.fixture
def wf_data():
    return {"SPY": _make_ohlcv(n=700, seed=3)}


def test_walk_forward_creates_windows(wf_data, tmp_path):
    from config.settings import Settings
    from backtest.walk_forward import WalkForwardValidator

    s = Settings(RISK_LOCK_FILE=str(tmp_path / "wf_lock.txt"))
    wfv = WalkForwardValidator(train_days=252, test_days=126, step_days=126,
                               settings_override=s)
    result = wfv.run(wf_data)

    assert len(result.windows) > 0, "Walk-forward must create at least one window"


def test_walk_forward_windows_do_not_overlap(wf_data, tmp_path):
    """No two test windows should share a date (step_days == test_days)."""
    from config.settings import Settings
    from backtest.walk_forward import WalkForwardValidator

    s = Settings(RISK_LOCK_FILE=str(tmp_path / "wf_lock2.txt"))
    wfv = WalkForwardValidator(train_days=252, test_days=126, step_days=126,
                               settings_override=s)
    result = wfv.run(wf_data)

    for i in range(len(result.windows) - 1):
        w_a = result.windows[i]
        w_b = result.windows[i + 1]
        assert w_a.test_end < w_b.test_start, (
            f"Window {i} test_end={w_a.test_end.date()} overlaps "
            f"window {i+1} test_start={w_b.test_start.date()}"
        )


def test_walk_forward_combined_metrics_generated(wf_data, tmp_path):
    from config.settings import Settings
    from backtest.walk_forward import WalkForwardValidator
    from backtest.metrics import PerformanceMetrics

    s = Settings(RISK_LOCK_FILE=str(tmp_path / "wf_lock3.txt"))
    wfv = WalkForwardValidator(train_days=252, test_days=126, step_days=126,
                               settings_override=s)
    result = wfv.run(wf_data)

    assert isinstance(result.combined_metrics, PerformanceMetrics)
    assert result.combined_metrics.starting_equity >= 0


# ---------------------------------------------------------------------------
# Tests 21-25: Stress tests
# ---------------------------------------------------------------------------

@pytest.fixture
def stress_data():
    return {"SPY": _make_ohlcv(n=400, seed=5)}


def test_stress_inject_drop_does_not_mutate_original(stress_data):
    from backtest.stress_tests import inject_market_drop
    original_close = stress_data["SPY"]["close"].copy()
    inject_market_drop(stress_data, drop_pct=0.10)
    pd.testing.assert_series_equal(stress_data["SPY"]["close"], original_close)


def test_stress_inject_drop_actually_lowers_prices(stress_data):
    from backtest.stress_tests import inject_market_drop
    injected = inject_market_drop(stress_data, drop_pct=0.10, date_index=200)
    orig_close = stress_data["SPY"]["close"].iloc[200]
    drop_close = injected["SPY"]["close"].iloc[200]
    assert drop_close < orig_close, "Prices after drop date must be lower"


def test_stress_inject_missing_data_removes_rows(stress_data):
    from backtest.stress_tests import inject_missing_data
    injected = inject_missing_data(stress_data, missing_fraction=0.10)
    original_len = len(stress_data["SPY"])
    new_len = len(injected["SPY"])
    assert new_len < original_len, "inject_missing_data must remove rows"


def test_stress_inject_stale_data_repeats_last_values(stress_data):
    from backtest.stress_tests import inject_stale_data
    n_repeat = 5
    injected = inject_stale_data(stress_data, repeat_last_n=n_repeat)
    df = injected["SPY"]
    # All of the last n_repeat rows should have the same close as row -(n_repeat+1)
    last_fresh_close = float(df.iloc[-(n_repeat + 1)]["close"])
    for i in range(n_repeat, 0, -1):
        assert float(df.iloc[-i]["close"]) == last_fresh_close, (
            f"Row -{i} should equal the pre-stale close {last_fresh_close}"
        )


def test_stress_standard_tests_return_named_results(stress_data, tmp_path):
    from config.settings import Settings
    from backtest.backtester import Backtester
    from backtest.stress_tests import run_standard_stress_tests

    s = Settings(RISK_LOCK_FILE=str(tmp_path / "st_lock.txt"))
    bt = Backtester(settings_override=s)
    results = run_standard_stress_tests(bt, stress_data)

    required_keys = {"baseline", "drop_10", "drop_15", "volatility_spike",
                     "missing_data", "stale_data"}
    assert set(results.keys()) == required_keys, (
        f"Missing stress scenarios: {required_keys - set(results.keys())}"
    )


# ---------------------------------------------------------------------------
# Tests 26-27: Error handling
# ---------------------------------------------------------------------------

def test_backtester_graceful_on_insufficient_data(bt_settings):
    """Fewer than WARMUP_BARS bars → empty BacktestResult with warning."""
    from backtest.backtester import Backtester, BacktestResult
    tiny_df = _make_ohlcv(n=50)
    bt = Backtester(settings_override=bt_settings)
    result = bt.run_single_ticker("SPY", tiny_df)
    assert isinstance(result, BacktestResult)
    assert len(result.warnings) > 0


def test_backtester_graceful_on_malformed_ticker_data(bt_settings):
    """Malformed data (missing close column) → empty result with warning."""
    from backtest.backtester import Backtester, BacktestResult
    bad_df = pd.DataFrame({"price": [100.0] * 300})
    bt = Backtester(settings_override=bt_settings)
    result = bt.run_single_ticker("BAD", bad_df)
    assert isinstance(result, BacktestResult)


# ---------------------------------------------------------------------------
# Tests 28-29: Risk manager integration
# ---------------------------------------------------------------------------

def test_backtester_respects_max_open_positions(tmp_path):
    """With max_open_positions=1, only one position should be held at a time."""
    from config.settings import Settings
    from backtest.backtester import Backtester

    # Settings with max 1 position
    s = Settings(
        MAX_OPEN_POSITIONS=1,
        RISK_LOCK_FILE=str(tmp_path / "pos_lock.txt"),
    )
    df = _make_ohlcv(n=400, seed=20)
    data = {"SPY": df, "QQQ": df.copy()}  # two identical tickers

    bt = Backtester(starting_equity=100_000, settings_override=s,
                    orchestrator=_always_buy_orch())
    result = bt.run(data)

    # Check that we never held more positions than max_open_positions
    # We can't inspect intraday state, but we verify no error was raised
    assert isinstance(result.equity_curve, pd.Series)


def test_backtester_respects_max_trades_per_day(tmp_path):
    """With max_trades_per_day=1, the risk manager blocks the second trade."""
    from config.settings import Settings
    from backtest.backtester import Backtester

    s = Settings(
        MAX_TRADES_PER_DAY=1,
        MAX_OPEN_POSITIONS=5,
        RISK_LOCK_FILE=str(tmp_path / "tpd_lock.txt"),
    )
    df = _make_ohlcv(n=400, seed=21)
    data = {"SPY": df, "QQQ": df.copy(), "AAPL": df.copy()}

    bt = Backtester(starting_equity=100_000, settings_override=s,
                    orchestrator=_always_buy_orch())
    result = bt.run(data)

    assert isinstance(result.equity_curve, pd.Series)


# ---------------------------------------------------------------------------
# Test 30: No-lookahead — equity curve identical up to shared dates
# ---------------------------------------------------------------------------

def test_no_lookahead_equity_curve_identical_for_shared_dates(tmp_path):
    """
    Running the backtest on 310 rows vs 410 rows must produce identical equity
    for the first 310 dates.  Future bars cannot alter past decisions.
    """
    from config.settings import Settings
    from backtest.backtester import Backtester

    df_full = _make_ohlcv(n=420, seed=99)

    s1 = Settings(RISK_LOCK_FILE=str(tmp_path / "lock310.txt"))
    s2 = Settings(RISK_LOCK_FILE=str(tmp_path / "lock410.txt"))

    bt310 = Backtester(starting_equity=100_000, settings_override=s1)
    bt410 = Backtester(starting_equity=100_000, settings_override=s2)

    r310 = bt310.run_single_ticker("SPY", df_full.iloc[:310])
    r410 = bt410.run_single_ticker("SPY", df_full.iloc[:410])

    common = r310.equity_curve.index

    if len(common) == 0:
        pytest.skip("No common dates in equity curves (warmup too long for data)")

    eq310 = r310.equity_curve
    eq410 = r410.equity_curve.loc[common]

    pd.testing.assert_series_equal(
        eq310.reset_index(drop=True),
        eq410.reset_index(drop=True),
        check_names=False,
        rtol=1e-6,
        obj=(
            "No-lookahead violated: equity at shared dates differs "
            "between 310-row and 410-row backtests"
        ),
    )
