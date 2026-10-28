"""
Data provider implementations for ai_regime_trader.

Usage:
    from data_providers import YFinanceDataProvider, SyntheticDataProvider
    from data_providers import CSVDataProvider, DataProvider, DataProviderError

    provider = YFinanceDataProvider()
    df = provider.get_bars("SPY", start="2022-01-01", end="2023-12-31")

    provider = SyntheticDataProvider(rows=400)
    data = provider.get_multiple_bars(["SPY", "QQQ"])
"""
from data_providers.base import DataProvider, DataProviderError
from data_providers.alpaca_provider import AlpacaDataProvider
from data_providers.csv_provider import CSVDataProvider
from data_providers.synthetic_provider import SyntheticDataProvider
from data_providers.yfinance_provider import YFinanceDataProvider

__all__ = [
    "DataProvider",
    "DataProviderError",
    "AlpacaDataProvider",
    "CSVDataProvider",
    "SyntheticDataProvider",
    "YFinanceDataProvider",
]
