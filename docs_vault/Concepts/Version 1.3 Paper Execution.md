# Version 1.3 — Managed Alpaca PAPER execution

The current workflow adds mandatory per-entry human approval on top of this
lifecycle. See [[Version 1.4 Human-Approved Paper Execution]] and the updated
[[Paper Execution Runbook]]. Paper activation alone no longer authorizes entries.

Implementation date: 2026-10-04. Acceptance testing is still required.
No live-money execution is implemented or authorized. Do not treat passing
unit tests as evidence of unattended operational readiness.

## Architecture and frozen strategy

`paper-session` defaults to the existing shadow workflow. Paper mode requires
`--execution-mode paper` (or `EXECUTION_MODE=paper`), and order submission also
requires `--execute-paper`. `--dry-run` is the default and never submits or
cancels broker orders. There is no live execution mode.

The managed session reuses `_run_shadow_scan_once`, `DayTradeSetupDetector`,
`SetupRanker`, `ShadowEntryPolicy`, `TradeMonitor`, `RiskManager`,
`CircuitBreakerManager`, `PositionSizer`, `AlpacaBrokerAdapter`, and the atomic
`TradeTracker` journal writer. It does not retrain or optimize anything.
The scan uses the same 400-day historical request window and rank/confidence
floors as the existing default Alpaca shadow session, not the old regime-based
`paper-loop` strategy. Recovered positions are monitored without injecting
additional tickers into the strategy ranking universe.

The frozen default entry rules remain confidence >= 0.80, rank <= 1, maximum
three open strategy slots, ten daily strategy entries, cooldowns enabled, and
no new entries at/after **13:00 America/New_York**. Existing project risk limits
can be stricter (for example MAX_TRADES_PER_DAY=5). Pending buys reserve slots,
exposure and trade-count capacity. A ticker cannot have overlapping managed
positions/orders. Shadow's separately configurable cutoff remains untouched;
v1.3 paper acceptance uses the frozen 13:00 cutoff without an override.

Execution details, not strategy optimization:

- Entry is a simple regular-hours DAY limit buy, capped at the frozen trigger
  (price rounded down to exchange precision). No price chasing or extended hours.
- Quantity is capped by the existing risk sizer, cash/buying power, and the
  existing default $1,000 shadow notional ceiling. Fractional quantity is rounded
  down to two decimals, or whole shares when fractions are disabled.
- Existing monitor exit actions request full remaining-position market sells.
  Actual fills, including slippage and partial fills, replace hypothetical fills.
- There are deliberately no broker brackets or new stop/target algorithms:
  those could change the frozen monitor's exit behavior. This means the process
  and network must remain available to act on monitor exits.
- Outstanding entries are canceled at the cutoff, upon partial-data/kill/risk
  blocking, or after the existing five-minute alert age. A cancellation request
  is not confirmation; fills racing cancellation remain owned and reconciled.
- Paper-only IO guards also reject missing, unknown, or future data timestamps.
  The existing stale thresholds remain >3 minutes for 1Min, >15 minutes for 5Min.
  Age is rechecked at decision time. Shadow CLEAN/PARTIAL selection is unchanged.

## Safety boundary

The wrapper only constructs `TradingClient(paper=True)`. Configuration must
name exactly `https://paper-api.alpaca.markets` (optional trailing slash).
A pinned HTTP transport rejects other origins, disables redirects/environment
proxies, enforces TLS verification and bounded connect/read timeouts, and
disables automatic SDK retries of ambiguous submissions.
Requests are paced (at least 0.4 seconds apart per owner) so reconciliation and
partial-fill cleanup do not produce uncontrolled request bursts.

Paper status is verified by successful authentication against that pinned
endpoint, not a locally asserted `is_paper` boolean. Alpaca account records do
not provide a standalone authoritative paper flag. The account must be ACTIVE,
with trading/account/user suspension flags explicitly false. A SHA-256 account
fingerprint binds durable state to the account; raw account IDs and credentials
are neither stored in the paper ledger nor printed. Assets must be active,
tradable US equities and fractionable when needed.

The broker clock must be open and within 60 seconds of the local aware UTC
clock. It supplies early-close/holiday boundaries; no market-hours bypass exists.
Submission repeats the clock, cutoff, deadline and local kill/risk checks at
the final preflight. The host clock must be synchronized.

Legacy `paper-once --execute-paper`, `paper-loop --execute-paper`, direct legacy
bracket/sell helpers and account-wide cancel are disabled. Their simulation
paths remain available. Only managed, durable, owned orders may be submitted.

## Durable lifecycle and recovery

`state/paper/ledger.sqlite3` is the authoritative SQLite WAL/FULL-synchronous
store for order intents, cumulative broker fills, session/recovery state,
risk references, and structured audit events. `state/paper/paper_trades.jsonl`
is a rebuildable projection using the existing journal schema, explicitly
`test_type=alpaca_paper` and excluded from shadow strategy proof.

An OS-held `owner.lock` prevents concurrent owners in this checkout and releases
on crash. The sidecar's existence is normal; **never delete it while a process
may be running**. SQLite intent and SUBMISSION_INTENT audit event commit before
the network call. Stable client IDs derive from date/symbol/setup/bar identity;
exit IDs also incorporate cumulative confirmed sold quantity.

An explicit local preflight rejection records LOCAL_PREFLIGHT_REJECTED, without
pretending the broker received an order. An ambiguous response stays UNKNOWN.
Recovery queries by client ID and never
blindly submits again, including when lookup returns 404. Reconciliation checks
order identity, quantity, cumulative-fill monotonicity and account positions.
Foreign orders, unowned positions, shorts, corrupt/missing state, account
changes, or mismatches stop new execution. No foreign trade is adopted or sold.

Partial entry fills are canceled/confirmed before exit orders. Pending sell
orders block duplicate exits. Canceled partially filled exits can submit only
the remaining owned quantity. Rejected zero-fill exits do not retry forever;
they require operator review. No fill price is invented.

RUNNING sessions resume the same identity/configuration with their quality
latch intact. ERROR/CLEANUP_PENDING sessions perform cleanup before allowing
new sessions. ENDED sessions with unfinished state are rejected, not adopted.
End-of-session cancels residual entries, reconciles, requests owned exits, and
waits a bounded number of polls for flat confirmation. Only confirmed flat,
no-pending-order state becomes ENDED; otherwise recovery remains necessary.

Daily/weekly/peak equity references survive restarts. The first deployment
uses its first verified observation as the initial reference, not a fabricated
historical market-open equity. Do not delete/reset state to bypass risk limits.

## Operational limits / acceptance risks

- Use a **dedicated, initially flat Alpaca paper account**, one checkout/process,
  no manual orders and no other trading bot. The local lock does not coordinate
  independent computers or copied checkouts. Account reconciliation fails closed
  if foreign activity is observed, but cannot eliminate cross-host races.
- Market sells can slip; stop-based risk sizing is not a guaranteed maximum
  realized loss. No software can guarantee fills during halts/outages.
- The kill switch is cooperative, not an exchange-side dead-man order. Broker
  polling continues during a background read-only data scan, but blocked broker
  requests can delay it until the bounded timeout. A terminated process cannot
  liquidate positions. Check the paper dashboard and supervise acceptance runs.
- A scan exceeding five minutes requests cleanup and records SCAN_TIMEOUT.
  The 400-day historical window may be expensive. It was not shortened silently
  because that could change frozen strategy features.
- Paper fills/partial-fill frequency differ from actual markets. A pre-cutoff
  order may fill while its cutoff cancellation is in flight; reconciliation
  records the actual event. This is not proof that a new post-cutoff signal
  was authorized.
- Unknown acknowledgements, unsupported broker status, trade corrections/busts,
  rejected exits, persistent position lag, or missing state require operator
  investigation. Never edit the ledger simply to clear a HOLD.
- Broker acceptance, permissions, IEX availability, fills, early close and real
  disconnect/restart behavior still require supervised paper acceptance. No
  broker-connected acceptance/smoke run was performed by this implementation.

See [[Paper Execution Runbook]] for deployment and acceptance instructions.

Provider references: [Alpaca paper environment](https://docs.alpaca.markets/us/docs/paper-trading)
and [order lifecycle and client IDs](https://docs.alpaca.markets/us/docs/orders-at-alpaca).
