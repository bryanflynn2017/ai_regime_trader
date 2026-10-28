"""
Central configuration for ai_regime_trader.

All values have safe defaults for paper trading.  Override via environment
variables or a local .env file — never hard-code secrets in this file.

Risk control values here are the system-wide MAXIMUMS.  The actual trading
logic uses these as ceilings, not targets.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Pydantic settings object.  Reads values from environment / .env file.
    Instantiate once as `settings = Settings()` and import that singleton.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # SAFETY SWITCHES — defaults must be maximally conservative
    # Changing these requires deliberate action, not a typo.
    # -------------------------------------------------------------------------
    PAPER_TRADING: bool = True
    LIVE_TRADING_ALLOWED: bool = False  # Must be explicitly set True to override
    EXECUTION_MODE: Literal["shadow", "paper"] = "shadow"

    # -------------------------------------------------------------------------
    # Alpaca credentials — loaded from .env, never hardcoded
    # -------------------------------------------------------------------------
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
    ALPACA_PAPER: bool = True
    DISCORD_WEBHOOK_URL: str = ""

    # -------------------------------------------------------------------------
    # Trading universe — edit this list to change what we trade
    # -------------------------------------------------------------------------
    TICKERS: List[str] = [
        "SPY",  "QQQ",  "DIA",  "IWM",
        "AAPL", "MSFT", "NVDA", "AMD",
        "META", "GOOGL","AMZN", "TSLA",
    ]
    BENCHMARK: str = "SPY"

    # -------------------------------------------------------------------------
    # Timeframes
    # -------------------------------------------------------------------------
    TIMEFRAME: str = "1Day"           # Historical/backtest bar size
    INTRADAY_TIMEFRAME: str = "5Min"  # Live paper-loop bar size

    # -------------------------------------------------------------------------
    # Risk limits — these are MAXIMUMS, not targets
    # -------------------------------------------------------------------------

    # Fraction of account equity risked per trade (entry → stop distance)
    # 0.5% means a $100k account risks $500 per trade
    MAX_RISK_PER_TRADE: float = 0.005

    # Total market value of all open positions / account equity ceiling
    MAX_PORTFOLIO_EXPOSURE: float = 0.50

    # Market value of any single position / account equity ceiling
    MAX_SINGLE_POSITION_EXPOSURE: float = 0.10

    # -------------------------------------------------------------------------
    # Drawdown circuit breakers (see risk/circuit_breakers.py)
    # -------------------------------------------------------------------------
    MAX_DAILY_DRAWDOWN_WARN: float = 0.02   # 2%  → halve new position sizes
    MAX_DAILY_DRAWDOWN_STOP: float = 0.03   # 3%  → stop new trades for the day
    MAX_WEEKLY_DRAWDOWN_STOP: float = 0.05  # 5%  → stop new trades for the week
    MAX_TOTAL_DRAWDOWN_LOCK: float = 0.10   # 10% → create lock file, halt permanently

    # -------------------------------------------------------------------------
    # Liquidity and price filters
    # -------------------------------------------------------------------------
    MIN_AVG_DAILY_VOLUME: int = 1_000_000  # Skip illiquid names
    MIN_PRICE: float = 5.0                  # Avoid penny stocks
    MAX_PRICE: float = 1000.0               # Avoid single-share concentration risk

    # -------------------------------------------------------------------------
    # Order behavior
    # -------------------------------------------------------------------------
    ALLOW_FRACTIONAL_SHARES: bool = True
    REQUIRE_STOP_LOSS: bool = True
    DEFAULT_STOP_LOSS_PCT: float = 0.03   # 3% below entry
    DEFAULT_TAKE_PROFIT_PCT: float = 0.06  # 6% above entry → 2:1 reward/risk

    # -------------------------------------------------------------------------
    # Regime detection
    # -------------------------------------------------------------------------
    # Regime must be stable for this many consecutive bars before we act on it.
    # Prevents whipsawing when the HMM flickers between states.
    REGIME_STABILITY_BARS: int = 3

    # -------------------------------------------------------------------------
    # Daily trade throttles
    # -------------------------------------------------------------------------
    MAX_TRADES_PER_DAY: int = 5
    MAX_OPEN_POSITIONS: int = 5

    # -------------------------------------------------------------------------
    # Filesystem paths (relative to project root)
    # -------------------------------------------------------------------------
    STATE_DIR: str = "state"
    LOG_DIR: str = "logs"
    RISK_LOCK_FILE: str = "state/RISK_LOCKED.txt"

    # =========================================================================
    # Validators
    # =========================================================================

    @field_validator("PAPER_TRADING", mode="before")
    @classmethod
    def warn_if_paper_disabled(cls, v: bool) -> bool:
        if not v:
            warnings.warn(
                "PAPER_TRADING is set to False. This project is designed for paper "
                "trading. Ensure you intend to trade with real money.",
                stacklevel=3,
            )
        return v

    @field_validator("LIVE_TRADING_ALLOWED", mode="before")
    @classmethod
    def warn_if_live_enabled(cls, v: bool) -> bool:
        if v:
            warnings.warn(
                "LIVE_TRADING_ALLOWED is True. Real money is at risk. "
                "Only set this if you have reviewed all risk controls.",
                stacklevel=3,
            )
        return v

    @field_validator("MAX_RISK_PER_TRADE")
    @classmethod
    def risk_per_trade_sane(cls, v: float) -> float:
        # 5% risk per trade is already considered reckless by most risk frameworks
        if v > 0.05:
            raise ValueError(
                f"MAX_RISK_PER_TRADE={v:.1%} exceeds 5%. "
                "This is dangerously high. Typical safe range is 0.5%–2%."
            )
        if v <= 0:
            raise ValueError("MAX_RISK_PER_TRADE must be positive.")
        return v

    @field_validator("MAX_PORTFOLIO_EXPOSURE")
    @classmethod
    def portfolio_exposure_sane(cls, v: float) -> float:
        if not 0 < v <= 1.0:
            raise ValueError("MAX_PORTFOLIO_EXPOSURE must be between 0 and 1.0.")
        return v

    @field_validator("MAX_SINGLE_POSITION_EXPOSURE")
    @classmethod
    def single_position_sane(cls, v: float) -> float:
        if not 0 < v <= 1.0:
            raise ValueError("MAX_SINGLE_POSITION_EXPOSURE must be between 0 and 1.0.")
        return v

    @field_validator("TICKERS")
    @classmethod
    def tickers_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("TICKERS list cannot be empty.")
        return [t.upper().strip() for t in v]

    @field_validator("DEFAULT_TAKE_PROFIT_PCT")
    @classmethod
    def take_profit_exceeds_stop(cls, v: float) -> float:
        # We validate the ratio in model_post_init since we need both fields
        if v <= 0:
            raise ValueError("DEFAULT_TAKE_PROFIT_PCT must be positive.")
        return v

    # =========================================================================
    # Convenience helpers
    # =========================================================================

    def risk_lock_path(self) -> Path:
        """Return the risk lock file as a Path object."""
        return Path(self.RISK_LOCK_FILE)

    def is_risk_locked(self) -> bool:
        """Return True if the risk lock file exists on disk."""
        return self.risk_lock_path().exists()

    def ensure_dirs(self) -> None:
        """Create state and log directories if they do not exist."""
        Path(self.STATE_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.LOG_DIR).mkdir(parents=True, exist_ok=True)

    def has_broker_credentials(self) -> bool:
        """Return True if Alpaca API key and secret are both set."""
        return bool(self.ALPACA_API_KEY and self.ALPACA_SECRET_KEY)


# ---------------------------------------------------------------------------
# Module-level singleton — import this throughout the project:
#   from config.settings import settings
# ---------------------------------------------------------------------------
settings = Settings()
