# Paper execution safety boundary

This publication snapshot is for source review and offline tests, not an activated
trading deployment. No credentials, previous approvals, account identity, orders,
positions or state are included. Do not restore someone else's ledger or copy a
running SQLite database into this source checkout.

Use `python scripts/offline_tests.py` for validation. The wrapper installs no
credentials and never starts a paper session. CI must never execute broker workflows.

Before any later supervised paper acceptance, the owner must separately authorize
the account-connected exercise, provide a dedicated paper account locally, verify
market data and clock freshness, and retain the existing exact-order approval,
reconciliation, risk, cutoff, kill and recovery controls. The supported code has no
live execution mode. Do not add one or treat an environment flag as approval.

Activation alone does not authorize an entry. Each immutable proposal expires within
60 seconds (or sooner for data/cutoff expiry), and must be approved and fully
revalidated. Exits/cancellations/cleanup remain automatic and must not wait for consent.

The owner must have access to the paper broker dashboard during acceptance. An error,
process exit or local kill file is not proof of liquidation. Confirm flat broker
positions and no pending owned orders. Preserve unknown intents; never delete state
to force a retry. The process/network may fail before an exit; no broker-hosted stop
is implied by the local stop-loss field.

See [[Publication]] for what was verified in this source snapshot and its limits.
