# Broker Adapters

v1.4 adds mandatory one-use human consent for each paper entry through the same
durable lifecycle. See [[Version 1.4 Human-Approved Paper Execution]]. Paper-mode
activation alone is not order approval; exits and cleanup stay non-interactive.

Broker adapters isolate broker access behind replaceable interfaces.

## Current Status
[[Version 1.3 Paper Execution]] integrates the existing Alpaca adapter with a
durable paper-only lifecycle for the frozen intraday strategy. Shadow remains
the default. Paper submission requires explicit activation and supervised
acceptance; live execution remains unavailable.

## Role
The adapter verifies the pinned paper endpoint/account, validates managed
orders, and supports client-ID lookup and order/position reconciliation under
[[Safety Rules]]. Legacy non-durable submission paths are disabled; simulation
remains available. See [[Paper Execution Runbook]] for activation and recovery.

## Related Notes
- [[Human-in-the-Loop Workflow]]
- [[Safety Rules]]
- [[Dashboard]]
