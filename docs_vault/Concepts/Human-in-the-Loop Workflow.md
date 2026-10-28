# Human-in-the-Loop Workflow

The app gives structured guidance while the user manually approves, enters, and closes trades.

## Current Status
Core Version 1.0 design.

## Role
This keeps [[Version 1.0]] safe while still making the daily workflow faster and more systematic.

## Related Commands
- `python main.py daily-plan`
- `python main.py setup-alerts-recent --since-hours 4`
- `python main.py trade-add-latest-alert --ticker TICKER --shares SHARES`
- `python main.py trade-monitor --timeframe 1Min --loop`

## Related Notes
- [[Safety Rules]]
- [[Trade Action Cards]]
- [[Manual Trade Journal]]
- [[Trade Monitor]]
- [[Dashboard]]
