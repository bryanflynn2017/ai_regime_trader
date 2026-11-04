# Robinhood read-only foundation (PR #7)

Status: NON-CONNECTED. No account access, authentication, MCP calls, network
transport, CLI, persisted account data, or order operations are implemented.
This is an internal integration contract, not a Python wrapper around Robinhood.
Evidence reviewed 2026-11-04; provider capabilities can change.

## Official evidence and limits

[External-agent onboarding](https://robinhood.com/us/en/support/articles/onboarding-an-external-agent/)
documents the Trading MCP endpoint `https://agent.robinhood.com/mcp/trading`,
Streamable HTTP setup, and interactive platform authentication/onboarding.
It permits reads across accounts, including account numbers, balances, positions
and transaction history, but restricts trading to the dedicated Agentic/MCP
account. Opening that self-directed individual account requires an existing
primary individual account in good standing.

[Trading documentation](https://robinhood.com/us/en/support/articles/trading-with-your-agent/)
lists external-agent reads including `get_accounts`, `get_portfolio`,
`get_equity_positions`, and `get_equity_orders`. It separately lists equity
review/place/cancel tools. These are not implemented here. Supported advertised
trading includes long equities, options and crypto, with market, limit and stop
order variants. MCP trade approvals are rolling out, unavailable to some users,
and OFF by default; when enabled, users manually place proposed trades in the
Robinhood app. Never assume MCP enforces this project's exact-order consent.

These public pages do not provide a sufficient versioned wire schema, token
lifecycle, read-only permission scope, pagination/completeness guarantees, or
snapshot timestamps for a verified Python implementation. No authenticated tool
discovery was performed. OAuth details and a supported Python client mapping
remain UNVERIFIED, not inferred from third-party wrappers. The separate
[official API documentation](https://docs.robinhood.com/) describes crypto,
not evidence of a stock/ETF REST brokerage API usable here.

## Existing architecture audit

- `broker/base.py`: BrokerAdapter includes submission/cancellation and requires
  paper safety. A real-account reader must NOT implement that mutation interface.
- `broker/alpaca_adapter.py`, `broker/paper_transport.py`: existing paper adapter
  and pinned paper-only HTTPS transport remain untouched.
- `broker/paper_execution.py`: durable intents, account fingerprint binding,
  fill/position reconciliation, unknown-order blocking, kill switch, market/data
  freshness and final risk/cutoff checks are unchanged.
- `broker/paper_approvals.py`: durable exact-payload approvals, expiration and
  single-use consumption remain exclusively in the paper lifecycle.
- `broker/position_tracker.py`: legacy reporting converts missing values to zero
  and some convenience methods suppress errors. Do not reuse those fallbacks for
  a future real-account read boundary. No changes made to legacy consumers.

## Internal read contract v1

`broker/read_only.py` defines AccountReader, validated AccountSnapshot (account
identity/type, USD balances, stock/ETF positions, complete open orders), and
OrderSnapshot for one explicitly requested order's status. This schema is ours,
NOT an asserted Robinhood response schema. A future reviewed translator must
bind every row to the selected account, exhaust pagination, map statuses
explicitly, and prove timestamp meaning. Never infer absence from a failed read.

Identity is explicitly supplied as SecretStr and compared exactly. Account type
is explicitly selected (`individual` or `mcp`); Robinhood acceptance must select
`mcp`. Narrowing reads to that account is our proposed least-privilege policy,
not a claim that Robinhood forbids reading other accounts. No account selection,
permission, or trading eligibility can be inferred from a display name.

Unknown fields/statuses, missing balances, incomplete snapshots, duplicate
positions/orders, invalid quantities and nonfinite amounts fail closed. Each
observation requires an aware timestamp; caller-supplied UTC-comparable time and
maximum age (default 60 seconds) make offline tests deterministic. Stale/future
observations fail; never replace unknown data with zero. This does not establish
an atomic broker snapshot or prove provider timestamps are trustworthy.

Models are immutable with tuples and redact every field from normal repr,
model_dump and JSON serialization. Explicit in-memory attributes remain private
financial data, not safe logs/exports. Generic reflection, debugger access or
intentional attribute extraction are not a security boundary. Only fixed error
codes and the placeholder's fixed log_safe_info are intended for logging.
No credentials are accepted or loaded; no account observations enter research
manifests. New code has no imports of execution, providers, settings or ledgers.

`broker/robinhood_read_only.py` always reports disconnected, advertises no
capabilities and rejects reads. It has no submit, cancel, connect or auth method.
Tests use synthetic internal fixtures, not captured financial records. It is not
wired into main.py, the dashboard, PositionTracker, or Alpaca execution.

## Acceptance before any future connection

1. Obtain a supported official schema/auth contract without reverse engineering.
   Confirm equity/ETF coverage, account-type markers, status/pagination semantics,
   timestamp meaning, token refresh/revocation and least-privilege read permissions.
2. Separately authorize a supervised authentication/read acceptance task. This PR
   does NOT authorize it. Store tokens in an OS credential store, never source,
   environment dumps, manifests, logs, fixtures or public issue attachments.
3. Build an audited exact read-tool allowlist. Tool annotations alone are not an
   authorization boundary. Block all mutation/review/order/approval tools and
   redirects; fail on schema drift or unsupported capabilities.
4. Verify selected dedicated MCP account identity/type, redacted errors, complete
   account-bound reads, stale data rejection and token revocation with controlled
   acceptance tests. Never import real responses into offline fixtures.
5. Order placement needs a separate design, authorization and safety review;
   there is no order route or activation switch in this foundation. Existing
   Alpaca paper approvals cannot authorize Robinhood real-money orders.

Offline validation: `python -B scripts/offline_tests.py tests/test_read_only_broker.py`.
There is deliberately no Robinhood connection or trading command to run.
