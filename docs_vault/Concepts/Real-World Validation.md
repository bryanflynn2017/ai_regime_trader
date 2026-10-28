# Real-World Validation

Real-world validation means observing the workflow and setup behavior during actual market sessions.

## Current Status
Ongoing after Version 1.0 functional completion and the active focus for
[[Version 1.1]].

## Role
It proves whether setup alerts, action cards, manual journaling, monitoring, and analytics are useful outside tests.

It also includes [[Shadow Forward Testing]] so simulated evidence can be
compared against manual workflow evidence without broker execution.

## Related Commands
- `python main.py daily-plan`
- `python main.py setup-scan-loop --watchlist mega_caps --timeframe 5Min`
- `python main.py trade-monitor --timeframe 1Min --loop`
- `python main.py forward-test-session --provider alpaca --alpaca-feed iex --watchlist momentum_universe --timeframe 5Min --max-iterations 72 --close-open-at-session-end`
- `python main.py trade-analytics`

## Related Notes
- [[Version 1.0]]
- [[Setup Alerts]]
- [[Manual Trade Journal]]
- [[Trade Analytics]]
- [[Shadow Forward Testing]]
- [[Clean Shadow Analytics]]
- [[Long-Term AI Direction]]
