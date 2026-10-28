"""
DataProvider abstract base class.

The rest of the system depends only on this interface — not on yfinance,
Alpaca data, or any specific vendor.  To add a new data source, subclass
DataProvider and implement get_bars().

get_multiple_bars() has a default implementation that calls get_bars() per
ticker and skips failures, so subclasses only need to implement get_bars().
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class DataProviderError(Exception):
    """Raised when a data provider fails to return usable data."""


class DataProvider(ABC):
    """
    Abstract interface for OHLCV market-data sources.

    Concrete implementations:
        YFinanceDataProvider  — Yahoo Finance via yfinance (daily only)
        CSVDataProvider       — Local CSV file (whatever is in the file)
        SyntheticDataProvider — Deterministic random-walk (daily only)

    All implementations must return DataFrames that pass validate_ohlcv().
    Each implementation declares supported_timeframes() — only those values
    are accepted. Requesting an unsupported timeframe raises DataProviderError.
    """

    @property
    def supported_timeframes(self) -> List[str]:
        """
        Return the list of supported timeframe strings.

        Default: ["1Day"] (daily bars only).
        Subclasses must override if they support intraday data.
        """
        return ["1Day"]

    @abstractmethod
    def get_bars(
        self,
        ticker: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a single ticker.

        Args:
            ticker:    Equity symbol, e.g. 'SPY'.
            start:     Start date as 'YYYY-MM-DD' or datetime.  None = provider default.
            end:       End date.  None = today.
            timeframe: Bar size, e.g. '1Day', '5Min'.  Must be in supported_timeframes.

        Returns:
            DataFrame with lowercase OHLCV columns (open/high/low/close/volume),
            sorted ascending by date, duplicates removed.

        Raises:
            DataProviderError: If timeframe is not supported, on any fetch failure,
                              or if validation fails.
        """

    def get_multiple_bars(
        self,
        tickers: List[str],
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        timeframe: str = "1Day",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV bars for multiple tickers.

        Default implementation calls get_bars() per ticker and skips failures.
        Raises DataProviderError if EVERY ticker fails.
        Subclasses may override for more efficient batch fetching.
        """
        results: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                results[ticker] = self.get_bars(ticker, start, end, timeframe)
            except DataProviderError as exc:
                logger.warning("Skipping %s: %s", ticker, exc)

        if not results:
            raise DataProviderError(
                f"Failed to fetch data for any ticker in {tickers}."
            )
        return results

    @property
    def provider_name(self) -> str:
        """Human-readable name for logging and display."""
        return type(self).__name__

    def _validate_timeframe(self, timeframe: str) -> None:
        """
        Validate that the requested timeframe is supported.

        Raises:
            DataProviderError: If timeframe is not in supported_timeframes.
        """
        if timeframe not in self.supported_timeframes:
            raise DataProviderError(
                f"{self.provider_name} does not support timeframe '{timeframe}'. "
                f"Supported: {', '.join(self.supported_timeframes)}"
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
