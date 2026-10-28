# Data Providers

Data providers load OHLCV bars from yfinance, CSV, or synthetic sources.

## Current Status
Implemented with daily and intraday timeframe support.

## Related Commands
- `python main.py setup-scan --ticker SPY --timeframe 5Min`
- `python main.py setup-scan --csv path/to/data.csv --timeframe 5Min`
- `python main.py setup-scan --synthetic --timeframe 5Min`

## Role
Data providers feed [[Setup Scanner]], [[Trade Monitor]], [[Backtesting]], and [[Walk-Forward Validation]].

## Related Notes
- [[Provider Freshness Diagnostics]]
- [[Real-World Validation]]
- [[Safety Rules]]
