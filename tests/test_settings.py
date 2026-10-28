"""
Tests for config/settings.py.

These tests verify that:
- Safety defaults are never accidentally broken
- Risk limits are internally consistent
- Validation rejects dangerous configurations
- File-system helpers work correctly
"""
from __future__ import annotations

import pytest
from pathlib import Path
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Safety defaults — these are the most critical tests in the project.
# If any of these fail, do NOT trade.
# ---------------------------------------------------------------------------

def test_paper_trading_default_true():
    """PAPER_TRADING must default to True. Core safety invariant."""
    from config.settings import Settings
    s = Settings()
    assert s.PAPER_TRADING is True


def test_live_trading_disabled_by_default():
    """LIVE_TRADING_ALLOWED must default to False."""
    from config.settings import Settings
    s = Settings()
    assert s.LIVE_TRADING_ALLOWED is False


def test_alpaca_paper_default_true():
    """ALPACA_PAPER flag must default to True."""
    from config.settings import Settings
    s = Settings()
    assert s.ALPACA_PAPER is True


def test_alpaca_base_url_is_paper_endpoint():
    """ALPACA_BASE_URL must default to the paper trading endpoint."""
    from config.settings import Settings
    s = Settings()
    assert "paper-api.alpaca.markets" in s.ALPACA_BASE_URL


def test_require_stop_loss_default_true():
    """REQUIRE_STOP_LOSS must default to True."""
    from config.settings import Settings
    s = Settings()
    assert s.REQUIRE_STOP_LOSS is True


# ---------------------------------------------------------------------------
# Universe configuration
# ---------------------------------------------------------------------------

def test_tickers_not_empty():
    """Default ticker list must not be empty."""
    from config.settings import Settings
    s = Settings()
    assert len(s.TICKERS) > 0


def test_benchmark_in_tickers():
    """Default benchmark (SPY) must be in the default ticker list."""
    from config.settings import Settings
    s = Settings()
    assert s.BENCHMARK in s.TICKERS


def test_tickers_are_uppercase():
    """Tickers must be normalized to uppercase by the validator."""
    from config.settings import Settings
    s = Settings(TICKERS=["spy", "qqq", "aapl"])
    for ticker in s.TICKERS:
        assert ticker == ticker.upper(), f"Ticker {ticker!r} not uppercase"


def test_empty_tickers_raises():
    """Settings must reject an empty TICKERS list."""
    from config.settings import Settings
    with pytest.raises(ValidationError, match="TICKERS"):
        Settings(TICKERS=[])


# ---------------------------------------------------------------------------
# Risk limits — internal consistency
# ---------------------------------------------------------------------------

def test_take_profit_greater_than_stop_loss():
    """DEFAULT_TAKE_PROFIT_PCT must exceed DEFAULT_STOP_LOSS_PCT (positive R:R)."""
    from config.settings import Settings
    s = Settings()
    assert s.DEFAULT_TAKE_PROFIT_PCT > s.DEFAULT_STOP_LOSS_PCT


def test_risk_per_trade_in_safe_range():
    """MAX_RISK_PER_TRADE must be within a sane range (0, 5%]."""
    from config.settings import Settings
    s = Settings()
    assert 0 < s.MAX_RISK_PER_TRADE <= 0.05


def test_drawdown_thresholds_in_correct_order():
    """Drawdown thresholds must be: warn < daily_stop <= weekly_stop <= total_lock."""
    from config.settings import Settings
    s = Settings()
    assert s.MAX_DAILY_DRAWDOWN_WARN < s.MAX_DAILY_DRAWDOWN_STOP
    assert s.MAX_DAILY_DRAWDOWN_STOP < s.MAX_TOTAL_DRAWDOWN_LOCK
    assert s.MAX_WEEKLY_DRAWDOWN_STOP <= s.MAX_TOTAL_DRAWDOWN_LOCK


def test_portfolio_exposure_not_greater_than_one():
    """MAX_PORTFOLIO_EXPOSURE must not exceed 100%."""
    from config.settings import Settings
    s = Settings()
    assert 0 < s.MAX_PORTFOLIO_EXPOSURE <= 1.0


def test_single_position_not_greater_than_portfolio():
    """Single position limit must not exceed total portfolio limit."""
    from config.settings import Settings
    s = Settings()
    assert s.MAX_SINGLE_POSITION_EXPOSURE <= s.MAX_PORTFOLIO_EXPOSURE


def test_regime_stability_bars_at_least_one():
    """REGIME_STABILITY_BARS must be a positive integer."""
    from config.settings import Settings
    s = Settings()
    assert s.REGIME_STABILITY_BARS >= 1


# ---------------------------------------------------------------------------
# Validation — should reject dangerous values
# ---------------------------------------------------------------------------

def test_max_risk_per_trade_over_five_percent_raises():
    """Settings must reject MAX_RISK_PER_TRADE > 5%."""
    from config.settings import Settings
    with pytest.raises(ValidationError):
        Settings(MAX_RISK_PER_TRADE=0.10)  # 10% — dangerously high


def test_max_risk_per_trade_zero_raises():
    """Settings must reject MAX_RISK_PER_TRADE of 0."""
    from config.settings import Settings
    with pytest.raises(ValidationError):
        Settings(MAX_RISK_PER_TRADE=0.0)


def test_portfolio_exposure_over_100_raises():
    """Settings must reject MAX_PORTFOLIO_EXPOSURE > 1.0."""
    from config.settings import Settings
    with pytest.raises(ValidationError):
        Settings(MAX_PORTFOLIO_EXPOSURE=1.5)


# ---------------------------------------------------------------------------
# Settings without .env — backtest mode must not require broker credentials
# ---------------------------------------------------------------------------

def test_settings_loads_without_env_file():
    """Settings must load with defaults even when no .env file is present."""
    from config.settings import Settings
    # _env_file=None tells pydantic-settings to skip file loading entirely
    s = Settings(_env_file=None)
    assert s is not None
    assert s.PAPER_TRADING is True
    assert s.LIVE_TRADING_ALLOWED is False


def test_empty_credentials_allowed():
    """Empty API credentials are allowed (needed for backtest-only mode)."""
    from config.settings import Settings
    s = Settings(ALPACA_API_KEY="", ALPACA_SECRET_KEY="")
    assert s.has_broker_credentials() is False


def test_has_broker_credentials_true_when_set():
    """has_broker_credentials() returns True when both keys are non-empty."""
    from config.settings import Settings
    s = Settings(ALPACA_API_KEY="FAKE_KEY", ALPACA_SECRET_KEY="FAKE_SECRET")
    assert s.has_broker_credentials() is True


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def test_risk_lock_path_returns_path_object():
    """risk_lock_path() must return a pathlib.Path instance."""
    from config.settings import Settings
    s = Settings()
    assert isinstance(s.risk_lock_path(), Path)


def test_is_risk_locked_false_when_no_file(tmp_path: Path):
    """is_risk_locked() returns False when the lock file does not exist."""
    from config.settings import Settings
    lock_file = tmp_path / "RISK_LOCKED.txt"
    s = Settings(RISK_LOCK_FILE=str(lock_file))
    assert s.is_risk_locked() is False


def test_is_risk_locked_true_when_file_exists(tmp_path: Path):
    """is_risk_locked() returns True when the lock file exists."""
    from config.settings import Settings
    lock_file = tmp_path / "RISK_LOCKED.txt"
    lock_file.write_text("Test lock — drawdown exceeded")
    s = Settings(RISK_LOCK_FILE=str(lock_file))
    assert s.is_risk_locked() is True


def test_ensure_dirs_creates_state_and_log(tmp_path: Path):
    """ensure_dirs() must create both STATE_DIR and LOG_DIR."""
    from config.settings import Settings
    state_dir = str(tmp_path / "state")
    log_dir = str(tmp_path / "logs")
    s = Settings(STATE_DIR=state_dir, LOG_DIR=log_dir)
    s.ensure_dirs()
    assert Path(state_dir).is_dir()
    assert Path(log_dir).is_dir()


def test_ensure_dirs_idempotent(tmp_path: Path):
    """ensure_dirs() must not raise if directories already exist."""
    from config.settings import Settings
    state_dir = str(tmp_path / "state")
    log_dir = str(tmp_path / "logs")
    s = Settings(STATE_DIR=state_dir, LOG_DIR=log_dir)
    s.ensure_dirs()
    s.ensure_dirs()  # Second call must not raise
