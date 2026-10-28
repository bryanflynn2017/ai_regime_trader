# Version 1.0

Version 1.0 is a [[Human-in-the-Loop Workflow|human-in-the-loop]] intraday trading assistant.

## Current Status
Functionally complete; validation continues. The app supports scanning, alerts, optional
[[Discord Alerts]], manual trade journaling, monitoring, dashboard review,
analytics, daily reviews, and simulation-only [[Shadow Forward Testing]].

[[Version 1.1]] is the active validation release and adds the promoted 13:00 ET
shadow new-entry cutoff.

## Role
Version 1.0 collects clean manual and shadow data and proves the workflow before
deeper [[Long-Term AI Direction|AI learning]] or [[Strategy Optimization|optimization]].

## Related Commands
- `python main.py daily-plan`
- `python main.py morning-start`
- `python main.py setup-scan-loop --watchlist mega_caps --timeframe 5Min --provider alpaca --alpaca-feed iex --interval-seconds 300 --discord-alerts`
- `python main.py trade-add-latest-alert --ticker TICKER --shares SHARES`
- `python main.py trade-monitor --provider alpaca --alpaca-feed iex --timeframe 1Min --loop --interval-seconds 60 --discord-alerts`
- `python main.py forward-test-session --provider alpaca --alpaca-feed iex --watchlist momentum_universe --timeframe 5Min --max-iterations 72 --close-open-at-session-end`
- `python main.py trade-analytics --include-shadow --clean-shadow-only`
- `python main.py daily-review --discord`
- `python main.py dashboard`

## Related Notes
- [[Human-in-the-Loop Workflow]]
- [[Setup Scanner]]
- [[Manual Trade Journal]]
- [[Trade Monitor]]
- [[Trade Analytics]]
- [[Shadow Forward Testing]]
- [[Clean Shadow Analytics]]
- [[Candidate Rule Tracking]]
- [[Safety Rules]]
- [[Real-World Validation]]
