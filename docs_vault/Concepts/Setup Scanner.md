# Setup Scanner

The setup scanner loads market data, detects watchlist setups, ranks valid candidates, and logs scan results.

## Current Status
Implemented for Version 1.0.

## Related Commands
- `python main.py setup-scan --watchlist mega_caps --timeframe 5Min`
- `python main.py setup-scan-loop --watchlist mega_caps --timeframe 5Min --interval-seconds 300`

## Role
Primary discovery tool in [[Version 1.0]]. It feeds [[Setup Alerts]], [[Setup Ranking]], and [[Trade Action Cards]].

Post-1.0, the fixed-watchlist scanner should evolve alongside the
[[Opportunity Discovery Engine]], which searches broader markets for bullish
movers outside saved watchlists.

## Related Notes
- [[Data Providers]]
- [[Opportunity Discovery Engine]]
- [[ORB Strategy]]
- [[VWAP Reclaim Strategy]]
- [[High Relative Volume Strategy]]
- [[Real-World Validation]]
