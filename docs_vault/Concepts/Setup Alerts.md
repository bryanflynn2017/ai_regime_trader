# Setup Alerts

Setup alerts are logged valid setup opportunities produced by the setup scanner loop.

## Current Status
Implemented for Version 1.0.

## Related Commands
- `python main.py setup-alerts-recent --since-hours 4`
- `python main.py trade-add-latest-alert --ticker TICKER --shares SHARES`

## Role
Alerts bridge [[Setup Scanner]] output to the [[Manual Trade Journal]]. They should be clean, traceable, and free of test noise.

## Related Notes
- [[Setup Ranking]]
- [[Trade Action Cards]]
- [[Manual Trade Journal]]
- [[Real-World Validation]]
