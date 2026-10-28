# Long-Term AI Direction

The long-term goal is an AI trading intelligence system that learns from its
own history, evaluates strategies, improves scoring, proposes new ideas, and
produces trade recommendations for user approval.

## Current Status
Post-1.0 direction. Not blocked, but it depends on clean Version 1.0 data.

## Role
After [[Version 1.0]] proves the workflow, the system should:
- learn from [[Manual Trade Journal|manual trade history]]
- learn from clean [[Shadow Forward Testing]] evidence
- analyze market data, event context, regime, sector, and trade history
- evaluate [[Trade Analytics|outcomes]]
- use [[Clean Shadow Analytics]] and [[Candidate Rule Tracking]] to avoid
  overfitting strategy changes
- compare [[ORB Strategy]], [[VWAP Reclaim Strategy]], [[High Relative Volume Strategy]], and future strategies
- improve [[Setup Ranking|ranking]] and thresholds
- propose new strategies through [[AI Strategy Generation]]
- discover bullish movers outside fixed watchlists through
  [[Opportunity Discovery Engine]]
- generate entry candidates and complete [[Trade Action Cards]]
- monitor open trades and recommend hold, reduce, exit, stop, or target actions
  through [[Trade Monitor]]

## Approval Boundary
Post-1.0, the app should do nearly everything except final order approval. It
can analyze, propose, plan, and manage recommendations. The user approves orders.

## Validation Rule
AI-generated strategies must be [[Backtesting|backtested]], [[Walk-Forward Validation|walk-forward validated]], and [[Real-World Validation|forward-tested]] before adoption.

## Related Notes
- [[Strategy Optimization]]
- [[AI Strategy Generation]]
- [[AI Trade Decision Engine]]
- [[Backtesting]]
- [[Walk-Forward Validation]]
- [[Real-World Validation]]
- [[Clean Shadow Analytics]]
- [[Candidate Rule Tracking]]
- [[Shadow Forward Testing]]
- [[Safety Rules]]
