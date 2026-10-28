"""
YFinanceDataProvider — fetches OHLCV data from Yahoo Finance via yfinance.

Delegates to the existing data.market_data.fetch_ticker function which
handles column normalisation, deduplication, and error messages.

Daily ("1Day") and intraday ("1Min", "5Min", "15Min") bars are supported by
mapping the timeframe to a yfinance interval string. Note that Yahoo Finance
only serves intraday history for a limited recent window; older intraday
requests (or provider outages) raise a clear DataProviderError.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd

from data_providers.base import DataProvider, DataProviderError

# Map our timeframe vocabulary to yfinance interval strings.
_TIMEFRAME_TO_INTERVAL: Dict[str, str] = {
    "1Day": "1d",
    "1Min": "1m",
    "5Min": "5m",
    "15Min": "15m",
}


class YFinanceDataProvider(DataProvider):
    """Market data from Yahoo Finance via yfinance (daily and intraday bars)."""

    @property
    def supported_timeframes(self) -> List[str]:
        """Daily plus intraday bars, mapped to yfinance intervals."""
        return list(_TIMEFRAME_TO_INTERVAL.keys())

    @property
    def provider_name(self) -> str:
        return "YFinance"

    def get_bars(
        self,
        ticker: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars from Yahoo Finance for the requested timeframe.

        Raises:
            DataProviderError: If timeframe is unsupported or the fetch fails.
        """
        self._validate_timeframe(timeframe)
        interval = _TIMEFRAME_TO_INTERVAL[timeframe]

        from data.market_data import fetch_ticker, DataFetchError

        if start is None:
            start = "2020-01-01"
        if end is None:
            end = str(pd.Timestamp.now().date())

        try:
            return fetch_ticker(
                ticker, start=str(start), end=str(end), interval=interval,
            )
        except DataFetchError as exc:
            raise DataProviderError(
                f"YFinanceDataProvider failed for '{ticker}': {exc}"
            ) from exc
        except Exception as exc:
            raise DataProviderError(
                f"YFinanceDataProvider unexpected error for '{ticker}': {exc}"
            ) from exc
