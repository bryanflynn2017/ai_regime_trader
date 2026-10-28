# AI Trade Decision Engine

The AI trade decision engine is the post-1.0 goal: the app analyzes data,
creates and evaluates strategies, proposes trades, and monitors exits while the
user remains the final approval gate.

## Current Status
Post-1.0 direction. It depends on stable [[Version 1.0]] workflows and clean
data in the [[AI Learning Dataset]].

## Role
The engine should eventually:
- analyze market data and context
- generate strategy ideas through [[AI Strategy Generation]]
- improve scoring through [[Strategy Optimization]]
- propose entry candidates with [[Trade Action Cards]]
- propose exit, reduce, hold, or target-hit actions through [[Trade Monitor]]
- log recommendations for review and learning

## Approval Boundary
The system may prepare and recommend trades, but the user approves orders. It
should not silently submit live orders.

## Related Notes
- [[Long-Term AI Direction]]
- [[Human-in-the-Loop Workflow]]
- [[Safety Rules]]
- [[Backtesting]]
- [[Walk-Forward Validation]]
- [[Real-World Validation]]
