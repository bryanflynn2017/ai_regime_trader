# Opportunity Discovery Engine

The Opportunity Discovery Engine is a post-[[Version 1.0]] goal: a system that
searches beyond fixed watchlists to find HOOD-style bullish movers, explain why
they are moving, rank the opportunity, and propose a shadow-test trade plan
without placing real orders.

## Current Status
Post-1.0 direction. It depends on stable [[Market Data Providers]], cleaner
context feeds, and enough [[Real-World Validation]] data from Version 1.0.

## Role
The engine should eventually:

- scan broad market universes outside saved watchlists
- detect unusual bullish movers, momentum, volume, news, sector, and regime
  context
- explain the likely reason a ticker is moving
- compare the move against known strategies such as [[VWAP Reclaim Strategy]],
  [[High Relative Volume Strategy]], and [[ORB Strategy]]
- rank the setup with [[Setup Ranking]]
- produce a [[Trade Action Cards|trade action card]]
- propose a shadow-test trade plan for review
- log the idea into the [[AI Learning Dataset]]

## Safety Boundary
This is discovery and planning only. It must not place real orders. Any proposed
trade remains subject to [[Human-in-the-Loop Workflow|human approval]],
[[Backtesting]], [[Walk-Forward Validation]], and forward testing.

## Related Notes
- [[Long-Term AI Direction]]
- [[AI Trade Decision Engine]]
- [[AI Strategy Generation]]
- [[Strategy Optimization]]
- [[Setup Scanner]]
- [[Market Data Providers]]
- [[Economic Calendar]]
- [[Sector Tracking]]
- [[Market Regime Tracking]]
- [[Safety Rules]]
