# Trade Monitor

Trade monitor reviews open manual trades against current market data and provides action guidance.

## Current Status
Implemented for Version 1.0.

## Related Commands
- `python main.py trade-monitor --timeframe 1Min`
- `python main.py trade-monitor --timeframe 1Min --loop --interval-seconds 60`

## Role
It advises HOLD, WATCH, TARGET HIT, STOP HIT, or EXIT VWAP LOST. It does not place orders.

## Related Notes
- [[Manual Trade Journal]]
- [[Human-in-the-Loop Workflow]]
- [[Safety Rules]]
- [[Trade Analytics]]
