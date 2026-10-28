# Clean Shadow Analytics

Clean shadow analytics filters shadow evidence to completed, clean sessions so
useful signal is not mixed with partial, stale, interrupted, or noisy runs.

## Current Status
Implemented for `trade-analytics --include-shadow --clean-shadow-only`.

## Related Commands
- `python main.py trade-analytics --include-shadow --clean-shadow-only`

## What It Shows
- clean session count
- watchlist breakdown
- closed/open shadow trades
- P&L, win rate, profit factor, expectancy
- setup/ticker/exit-reason breakdowns
- entry-time performance
- stop-hit diagnostics
- ORB no-progress diagnostics
- large loser diagnostics
- hypothetical filter diagnostics
- [[Candidate Rule Tracking]]

## Role in Version 1.0
Research and validation only. It helps decide what deserves more testing before
strategy rule changes. A rule affects shadow behavior only after an explicit
tested promotion; the 13:00 ET cutoff is the first promoted rule in
[[Version 1.1]].

## Related Notes
- [[Shadow Forward Testing]]
- [[Trade Analytics]]
- [[Candidate Rule Tracking]]
- [[ORB Strategy]]
- [[High Relative Volume Strategy]]
- [[Real-World Validation]]
