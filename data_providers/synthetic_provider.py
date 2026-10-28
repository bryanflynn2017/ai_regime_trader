"""
SyntheticDataProvider — deterministic random-walk OHLCV data.

NOT real market data.  For offline smoke testing and CI only.
Each ticker gets data from a different seed (base_seed + hash(ticker))
so tickers are not identical to each other.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd

from data_providers.base import DataProvider, DataProviderError

# Map our timeframe vocabulary to the synthetic generator's interval (minutes).
# None means daily (business-day index); an int means intraday bars.
_TIMEFRAME_TO_MINUTES: Dict[str, Optional[int]] = {
    "1Day": None,
    "1Min": 1,
    "5Min": 5,
    "15Min": 15,
}


class SyntheticDataProvider(DataProvider):
    """
    Generate synthetic OHLCV data without any network calls (daily or intraday).

    Useful for:
      - offline smoke tests (--synthetic flag)
      - CI pipelines without market-data credentials
      - fast iteration on strategy logic

    Intraday timeframes produce a regular-session DatetimeIndex (see
    data.market_data._synthetic_intraday_index). This is synthetic data and
    ignores holidays/timezones — it only approximates a weekday session shape.
    """

    def __init__(
        self,
        rows: int = 500,
        seed: int = 42,
        start: str = "2020-01-01",
    ) -> None:
        self._rows  = rows
        self._seed  = seed
        self._start = start

    @property
    def supported_timeframes(self) -> List[str]:
        """Daily plus intraday bars are generated."""
        return list(_TIMEFRAME_TO_MINUTES.keys())

    @property
    def provider_name(self) -> str:
        return "Synthetic"

    def get_bars(
        self,
        ticker: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """
        Generate synthetic bars for the requested timeframe.

        ticker and start/end parameters are accepted for interface
        compatibility but ignored — the data is randomly generated.
        Each ticker receives a different random seed so tickers are
        not all identical. The timeframe selects daily vs. intraday
        (1/5/15-minute) bar generation.

        Raises:
            DataProviderError: If timeframe is unsupported.
        """
        self._validate_timeframe(timeframe)
        interval_minutes = _TIMEFRAME_TO_MINUTES[timeframe]

        from data.market_data import generate_synthetic_ohlcv, DataFetchError

        try:
            # Vary seed by ticker so multiple tickers are not identical
            ticker_seed = (self._seed + abs(hash(ticker))) % (2**31)
            return generate_synthetic_ohlcv(
                rows=self._rows,
                start=self._start,
                seed=ticker_seed,
                interval_minutes=interval_minutes,
            )
        except DataFetchError as exc:
            raise DataProviderError(
                f"SyntheticDataProvider failed: {exc}"
            ) from exc

    def get_multiple_bars(
        self,
        tickers: List[str],
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        timeframe: str = "1Day",
    ) -> Dict[str, pd.DataFrame]:
        """Generate independent synthetic bars for each ticker."""
        return {
            ticker: self.get_bars(ticker, start, end, timeframe)
            for ticker in tickers
        }
