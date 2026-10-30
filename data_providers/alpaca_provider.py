"""
AlpacaDataProvider — market data only via Alpaca paper credentials.

This provider does not submit orders and does not instantiate broker trading
clients. It only fetches historical equity bars through alpaca-py data APIs.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Dict, List, Optional, Union

import pandas as pd

from data.market_data import DataFetchError, validate_ohlcv
from data_providers.base import DataProvider, DataProviderError
from utils.provider_errors import safe_provider_error


_TIMEFRAME_MINUTES: Dict[str, int] = {
    "1Min": 1,
    "5Min": 5,
}


class AlpacaDataProvider(DataProvider):
    """Market data from Alpaca Data API for intraday monitoring."""

    def __init__(self, settings_override=None, client=None, feed: str = "iex") -> None:
        self.settings = settings_override
        self._client = client
        feed_normalized = str(feed or "iex").lower()
        if feed_normalized not in {"iex", "sip"}:
            raise DataProviderError("AlpacaDataProvider feed must be one of: iex, sip")
        self.feed = feed_normalized

    @property
    def supported_timeframes(self) -> List[str]:
        return list(_TIMEFRAME_MINUTES.keys())

    @property
    def provider_name(self) -> str:
        return "AlpacaData"

    def get_bars(
        self,
        ticker: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        timeframe: str = "1Min",
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from Alpaca market data."""
        self._validate_timeframe(timeframe)

        client = self._get_client()
        try:
            request = self._build_request(ticker, start, end, timeframe)
            try:
                response = client.get_stock_bars(request)
            except Exception as exc:
                raise DataProviderError(self._format_provider_error("Historical bar request failed", exc)) from None
            df = self._response_to_dataframe(response, ticker)
            validate_ohlcv(df)
            self._attach_freshness_metadata(df)
            return df
        except DataProviderError:
            raise
        except DataFetchError as exc:
            raise DataProviderError(self._format_provider_error("AlpacaDataProvider OHLCV validation failed", exc)) from None
        except Exception as exc:
            raise DataProviderError(
                self._format_provider_error("AlpacaDataProvider failed", exc)
            ) from None

    def get_latest_bar(self, ticker: str, timeframe: str = "1Min") -> pd.DataFrame:
        """
        Fetch the latest Alpaca bar for a ticker.

        Alpaca exposes a latest-bar endpoint. Trade monitoring should prefer it
        over historical date-window requests so repeated loop iterations do not
        keep reading the same stale completed bar range.
        """
        self._validate_timeframe(timeframe)
        client = self._get_client()

        try:
            if hasattr(client, "get_stock_latest_bar"):
                request = self._build_latest_request(ticker)
                try:
                    response = client.get_stock_latest_bar(request)
                except Exception as exc:
                    raise DataProviderError(self._format_provider_error("Latest bar request failed", exc)) from None
                df = self._latest_response_to_dataframe(response, ticker)
            else:
                end = self._current_utc()
                start = end - timedelta(days=1)
                df = self.get_bars(ticker, start=start, end=end, timeframe=timeframe).tail(1)
            validate_ohlcv(df)
            self._attach_freshness_metadata(df)
            return df
        except DataProviderError:
            raise
        except DataFetchError as exc:
            raise DataProviderError(self._format_provider_error("AlpacaDataProvider latest OHLCV validation failed", exc)) from None
        except Exception as exc:
            raise DataProviderError(
                self._format_provider_error("AlpacaDataProvider latest bar failed", exc)
            ) from None

    def _get_client(self):
        if self._client is not None:
            return self._client

        settings = self.settings
        if settings is None:
            from config.settings import settings as default_settings
            settings = default_settings
            self.settings = settings

        from config.secret_values import secret_value
        api_key = secret_value(getattr(settings, "ALPACA_API_KEY", None))
        secret_key = secret_value(getattr(settings, "ALPACA_SECRET_KEY", None))
        if not api_key or not secret_key:
            raise DataProviderError(
                "AlpacaDataProvider requires Alpaca paper data credentials. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env."
            )

        try:
            from alpaca.data.historical import StockHistoricalDataClient
        except Exception as exc:
            raise DataProviderError(
                "alpaca-py is unavailable. Install alpaca-py to use --provider alpaca."
            ) from exc

        try:
            self._client = StockHistoricalDataClient(api_key, secret_key)
        except Exception as exc:
            raise DataProviderError(self._format_provider_error("Market data client initialization failed", exc)) from None
        return self._client

    def _build_request(
        self,
        ticker: str,
        start: Optional[Union[str, datetime]],
        end: Optional[Union[str, datetime]],
        timeframe: str,
    ):
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        except Exception as exc:
            if self._client is not None:
                return {
                    "symbol_or_symbols": [ticker.upper()],
                    "timeframe": timeframe,
                    "start": self._to_datetime(start, default_days_back=5),
                    "end": self._to_datetime(end, default_days_back=0),
                    "feed": self.feed,
                }
            raise DataProviderError(
                "alpaca-py data request classes are unavailable. Install alpaca-py."
            ) from exc

        minutes = _TIMEFRAME_MINUTES[timeframe]
        tf = TimeFrame.Minute if minutes == 1 else TimeFrame(minutes, TimeFrameUnit.Minute)
        request_kwargs = {
            "symbol_or_symbols": [ticker.upper()],
            "timeframe": tf,
            "start": self._to_datetime(start, default_days_back=5),
            "end": self._to_datetime(end, default_days_back=0),
            "feed": self._feed_request_value(),
        }
        try:
            return StockBarsRequest(**request_kwargs)
        except TypeError:
            request_kwargs.pop("feed", None)
            return StockBarsRequest(**request_kwargs)

    @staticmethod
    def _to_datetime(value: Optional[Union[str, datetime]], default_days_back: int) -> datetime:
        if value is None:
            base = AlpacaDataProvider._current_utc()
            if default_days_back:
                base = base - timedelta(days=default_days_back)
                return datetime.combine(base.date(), time.min)
            return base
        if isinstance(value, datetime):
            return value
        parsed = pd.to_datetime(value, errors="raise")
        return parsed.to_pydatetime()

    @staticmethod
    def _current_utc() -> datetime:
        return datetime.now(timezone.utc).replace(tzinfo=None)

    def _build_latest_request(self, ticker: str):
        try:
            from alpaca.data.requests import StockLatestBarRequest
        except Exception as exc:
            if self._client is not None:
                return {"symbol_or_symbols": [ticker.upper()], "feed": self.feed}
            raise DataProviderError(
                "alpaca-py latest bar request classes are unavailable. Install alpaca-py."
            ) from exc

        request_kwargs = {
            "symbol_or_symbols": [ticker.upper()],
            "feed": self._feed_request_value(),
        }
        try:
            return StockLatestBarRequest(**request_kwargs)
        except TypeError:
            request_kwargs.pop("feed", None)
            return StockLatestBarRequest(**request_kwargs)

    def _feed_request_value(self):
        try:
            from alpaca.data.enums import DataFeed
            return DataFeed.IEX if self.feed == "iex" else DataFeed.SIP
        except Exception:
            return self.feed

    def _format_provider_error(self, operation: str, error: Exception) -> str:
        text = safe_provider_error(operation, error)
        if self.feed == "sip":
            return (
                f"{text} If your plan does not permit SIP data, "
                "use --alpaca-feed iex or review your data subscription."
            )
        return text

    @staticmethod
    def _response_to_dataframe(response, ticker: str) -> pd.DataFrame:
        if hasattr(response, "df") and response.df is not None:
            df = response.df.copy()
            # Some clients expose symbol as a column rather than an index level.
            symbol_column = next((c for c in df.columns if str(c).lower() == "symbol"), None)
            if symbol_column is not None:
                df = df.loc[df[symbol_column].astype(str).str.upper() == ticker.upper()]
                if df.empty:
                    raise DataProviderError(f"AlpacaDataProvider returned no bars for '{ticker}'.")
            if isinstance(df.index, pd.MultiIndex):
                symbol_level = 0
                try:
                    symbols = df.index.get_level_values("symbol")
                    if ticker.upper() in set(str(s).upper() for s in symbols):
                        symbol_level = "symbol"
                except Exception:
                    pass
                try:
                    df = df.xs(ticker.upper(), level=symbol_level)
                except (KeyError, ValueError) as exc:
                    raise DataProviderError(f"AlpacaDataProvider returned no bars for '{ticker}'.") from exc
            return AlpacaDataProvider._clean_dataframe(df)

        data = getattr(response, "data", response)
        if isinstance(data, dict):
            bars = data.get(ticker.upper()) or data.get(ticker) or []
        else:
            bars = data or []

        records = []
        for bar in bars:
            records.append({
                "timestamp": AlpacaDataProvider._bar_value(bar, "timestamp", "t"),
                "open": AlpacaDataProvider._bar_value(bar, "open", "o"),
                "high": AlpacaDataProvider._bar_value(bar, "high", "h"),
                "low": AlpacaDataProvider._bar_value(bar, "low", "l"),
                "close": AlpacaDataProvider._bar_value(bar, "close", "c"),
                "volume": AlpacaDataProvider._bar_value(bar, "volume", "v"),
            })
        df = pd.DataFrame(records)
        return AlpacaDataProvider._clean_dataframe(df)

    @staticmethod
    def _latest_response_to_dataframe(response, ticker: str) -> pd.DataFrame:
        data = getattr(response, "data", response)
        bar = data
        if isinstance(data, dict):
            bar = data.get(ticker.upper()) or data.get(ticker)

        if isinstance(bar, list):
            bar = bar[-1] if bar else None
        if bar is None:
            raise DataProviderError(f"AlpacaDataProvider returned no latest bar for '{ticker}'.")

        record = {
            "timestamp": AlpacaDataProvider._bar_value(bar, "timestamp", "t"),
            "open": AlpacaDataProvider._bar_value(bar, "open", "o"),
            "high": AlpacaDataProvider._bar_value(bar, "high", "h"),
            "low": AlpacaDataProvider._bar_value(bar, "low", "l"),
            "close": AlpacaDataProvider._bar_value(bar, "close", "c"),
            "volume": AlpacaDataProvider._bar_value(bar, "volume", "v"),
        }
        return AlpacaDataProvider._clean_dataframe(pd.DataFrame([record]))

    @staticmethod
    def _bar_value(bar, *names):
        for name in names:
            if isinstance(bar, dict) and name in bar:
                return bar[name]
            if hasattr(bar, name):
                return getattr(bar, name)
        return None

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            raise DataProviderError("AlpacaDataProvider returned no bars.")

        df = df.copy()
        df.columns = [str(col).lower() for col in df.columns]
        if "timestamp" in df.columns:
            df.index = pd.to_datetime(df.pop("timestamp"), errors="coerce")
        elif "time" in df.columns:
            df.index = pd.to_datetime(df.pop("time"), errors="coerce")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")

        column_aliases = {
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
        df = df.rename(columns=column_aliases)
        keep = ["open", "high", "low", "close", "volume"]
        missing = [col for col in keep if col not in df.columns]
        if missing:
            raise DataProviderError(f"AlpacaDataProvider missing OHLCV columns: {missing}")

        df = df[keep]
        df = df[~df.index.isna()]
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert(None)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        for col in keep:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=keep)
        return df

    @staticmethod
    def _attach_freshness_metadata(df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        provider_time = pd.Timestamp.now(tz="UTC").tz_convert(None)
        latest_bar_time = pd.Timestamp(df.index.max())
        if latest_bar_time.tzinfo is not None:
            latest_bar_time = latest_bar_time.tz_convert(None)
        data_age_seconds = max(0.0, float((provider_time - latest_bar_time).total_seconds()))
        df.attrs["latest_bar_time"] = latest_bar_time.isoformat()
        df.attrs["provider_time"] = provider_time.isoformat()
        df.attrs["data_age_seconds"] = data_age_seconds
