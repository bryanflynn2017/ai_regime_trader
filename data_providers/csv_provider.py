"""
CSVDataProvider — load OHLCV data from a local CSV file.

The same file is returned for every ticker request (typically used when
testing with a single-ticker dataset).  ticker, start, end parameters are
accepted for interface compatibility but ignored.

A CSV may contain bars at any frequency, so all timeframes ("1Day", "1Min",
"5Min", "15Min") are accepted. The file is loaded and validated as-is — the
provider does NOT resample, so it trusts that the file's bar frequency matches
the requested timeframe.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from data_providers.base import DataProvider, DataProviderError


class CSVDataProvider(DataProvider):
    """Load OHLCV data from a local CSV file (any bar frequency)."""

    def __init__(self, file_path: Union[str, Path]) -> None:
        self._file_path = Path(file_path)
        if not self._file_path.exists():
            raise DataProviderError(
                f"CSVDataProvider: file not found: {self._file_path.resolve()}"
            )

    @property
    def supported_timeframes(self) -> List[str]:
        """A CSV can hold any frequency; the file is loaded as-is (no resampling)."""
        return ["1Day", "1Min", "5Min", "15Min"]

    @property
    def provider_name(self) -> str:
        return f"CSV({self._file_path.name})"

    def __repr__(self) -> str:
        return f"CSVDataProvider(file_path={self._file_path!r})"

    def get_bars(
        self,
        ticker: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """
        Load the CSV file.

        ticker, start, end parameters are accepted for interface
        compatibility but ignored — the file contains one dataset. The file
        is loaded as-is for any supported timeframe (no resampling).

        Raises:
            DataProviderError: If timeframe is unsupported or the load fails.
        """
        self._validate_timeframe(timeframe)

        from data.market_data import load_ohlcv_csv, DataFetchError

        try:
            return load_ohlcv_csv(self._file_path)
        except DataFetchError as exc:
            raise DataProviderError(
                f"CSVDataProvider failed to load '{self._file_path}': {exc}"
            ) from exc
        except Exception as exc:
            raise DataProviderError(
                f"CSVDataProvider unexpected error: {exc}"
            ) from exc
