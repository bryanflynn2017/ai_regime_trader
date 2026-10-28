# Version 1.4 — Human-Approved Paper Execution

Implemented on the v1.3 paper lifecycle; review and supervised acceptance remain
required. **Not live execution, not strategy optimization.** Shadow remains the
default and no orders are submitted by automated tests.

## Architecture

The frozen scanner/policy and existing risk manager produce an exact proposed
BUY limit order. `broker/paper_approvals.py` extends the existing SQLite paper
ledger with a proposals table. Canonical JSON and SHA-256 bind the exact symbol,
quantity, price, validity deadline, session, paper-account fingerprint and setup.
The operator inspects and approves/rejects one proposal through local CLI commands.
The decision and audit persist through process restarts.

Maximum lifetime is 60 seconds from planning, bounded by data expiry and the
unchanged 13:00 ET cutoff. No renewal, blanket approval, implicit approval from
paper activation, or approval transfer is supported.

Consent consumption and the existing UNKNOWN order intent commit atomically
before POST. At the Alpaca adapter's final submission hook, the coordinator reads
fresh historical bars, verifies the same paper account, reconciles orders and
positions, reevaluates existing risk calculations and frozen entry policy, and
compares the resulting order to the approved order. Any changed sizing refuses
submission instead of resizing. Bar ages, deadline, broker clock, cutoff and
kill/risk locks are checked again after potentially slow work.

The final reconciliation excludes only its own definitively unsent UNKNOWN
intent from pending exposure/trade counts. It still queries that client ID and
refuses if it already exists at the broker. Normal crash recovery never makes
this exception: unresolved intents are never automatically replayed.

One process retains v1.3 OS ownership of execution. Separate operator connections
use short SQLite `BEGIN IMMEDIATE` transactions; they cannot execute broker calls.
No transaction stays open across network IO. An approval is consumed at most once.
The adapter remains pinned to the verified Alpaca paper endpoint and legacy
non-durable submission commands remain disabled.

## States and recovery

- PENDING -> APPROVED or REJECTED by explicit exact-digest operator decision.
- PENDING/APPROVED -> EXPIRED at the original deadline, or INVALIDATED by cleanup,
  new session, partial data, error or changed recovery context.
- APPROVED -> CONSUMED atomically with the UNKNOWN order intent.
- Order outcome is independent: acknowledged, partial fill, filled, rejected,
  canceled, or unresolved. CONSUMED alone is not a fill confirmation.
- Definitive pre-POST refusal marks the order rejected. Uncertain submission
  preserves UNKNOWN for reconciliation; no automatic replay or consent reuse.
- Still-valid approvals survive only the same RUNNING session. ERROR and cleanup
  recovery cannot submit entries. Old v1.3 orders need no retroactive consent to
  reconcile or exit; no consent is inferred for new entries.

Exits, partial-fill cancellation, monitoring, session-end flattening and emergency
cleanup use the existing lifecycle without interactive approval. Clean shadow
analytics, detectors, ranking, sizing formulas and monitor exit rules are unchanged.

## Limits

Local OS access is the trust boundary, not authenticated multi-user approval.
The digest prevents accidental/order-substitution mismatches, not malicious local
database/application replacement. Network operations are not broker-side atomic
with revalidation; an account/feed can change after the last read. Use a dedicated
paper account with no concurrent manual activity. Slow reads can expire consent.
Keep the supervisor alive and the paper dashboard available: this does not add
broker-hosted protective exits, and outage/halts can still require intervention.

See [[Paper Execution Runbook]] for commands, acceptance and incident handling.
