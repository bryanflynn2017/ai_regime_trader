# Version 1.1

Version 1.1 is the active prospective-validation release after the functional
completion of [[Version 1.0]].

## Current Status
Active. New shadow entries are blocked at or after 13:00 ET by default.

## Entry Cutoff
- Configure with `--entry-cutoff HH:MM`.
- Default: `13:00` ET.
- Blocked candidates are logged with `AFTER_ENTRY_CUTOFF`.
- Existing open shadow trades continue to be monitored and exited normally.
- The rule affects simulation-only [[Shadow Forward Testing]], not live orders.

## Validation Goal
Continue collecting clean shadow sessions to determine whether the cutoff
improves future results. Historical candidate performance justified promotion,
but prospective evidence is required before treating the improvement as proven.

## Safety Boundary
- No live trading is authorized.
- No real-money automation is authorized.
- No automatic broker execution is authorized for the intraday assistant.
- Future paper execution work means integration and hardening of existing
  [[Broker Adapters|paper components]], not immediate real-money trading.

## Related Notes
- [[Version 1.0]]
- [[Shadow Forward Testing]]
- [[Candidate Rule Tracking]]
- [[Clean Shadow Analytics]]
- [[Real-World Validation]]
- [[Safety Rules]]
