# Market Data Providers

Market data providers supply bars and context used by scans, monitors, backtests, and validation.

## Current Status
Priority 1 post-1.0 integration area.

## Plan
- Maintain the modular [[Data Providers]] interface.
- Keep yfinance as a free fallback provider.
- Add providers later such as Polygon, Alpaca Data, TwelveData, IEX, or Interactive Brokers data.
- New providers must not break CSV, synthetic, or yfinance paths.
- Provider failures must be clear and graceful.

## Role
Better providers make [[Setup Scanner]], [[Trade Monitor]], [[Backtesting]], and [[Real-World Validation]] more durable.

## Related Notes
- [[Data Providers]]
- [[Long-Term Maintenance]]
- [[Safety Rules]]
