# PR #8: official MCP discovery protocol foundation

Reviewed 2026-11-04. Status: **offline-only protocol subset, NOT connected**.
This is the smallest verified increment while provider auth/schema evidence is
missing. It is not a complete Streamable HTTP transport or working brokerage
connection. No actual MCP endpoint requests, authentication or account reads
were made in development. Synthetic tests are not provider acceptance evidence.

## Official evidence

- [Robinhood onboarding](https://robinhood.com/us/en/support/articles/onboarding-an-external-agent/)
  identifies the official Trading MCP endpoint and interactive authentication.
  Reads can expose multiple accounts; trades are restricted to the dedicated
  MCP/Agentic account. Our future read integration will deliberately narrow to
  one explicitly selected dedicated account, not every accessible account.
- [Robinhood tools](https://robinhood.com/us/en/support/articles/trading-with-your-agent/)
  lists `get_accounts`, `get_portfolio`, `get_equity_positions`, and
  `get_equity_orders` for external agents. This is the exact discovery candidate
  allowlist. Review, place, cancel, and approval tools are excluded entirely.
  MCP approvals are not universally available and default off; they cannot
  replace project authorization safeguards.
- [MCP versioning](https://modelcontextprotocol.io/docs/2026-07-28/learn/versioning)
  identifies 2026-07-28 as current. The older 2025-11-25 lifecycle uses an
  initialize handshake; the current revision uses per-request metadata.
  Robinhood's supported revision is UNVERIFIED. There is no automatic downgrade.
- [Current transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
  specifies POST envelopes, matching version/method headers, JSON or SSE
  responses, and no protocol-level sessions. The codec prepares these request
  metadata fields but only decodes the JSON response branch. SSE is deliberately
  rejected; full conformance requires both. Accept headers in prepared envelopes
  describe the eventual transport, not a claim this codec implements SSE.
- [Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
  defines `tools/list`, input schemas and optional output schemas. Annotations
  are not an authorization boundary. Discovering a schema neither approves it
  nor proves account identity, freshness or financial pagination semantics.
- [Pagination](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/pagination)
  uses opaque cursors; an empty cursor is valid. Discovery is complete only when
  the cursor is absent/null. Tool-list pagination is NOT account-data pagination.
- [Authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
  specifies OAuth-based protected-resource/issuer discovery, PKCE-associated
  authorization validation, resource-bound tokens, and bearer headers rather
  than query-string tokens. Refresh-token issuance must not be assumed. These
  generic requirements do not prove Robinhood's scopes, registration support,
  refresh policy or account selection fields. No OAuth flow is implemented.

## Exact implementation and boundaries

`broker/robinhood_mcp_protocol.py` is a pure in-memory state machine:

1. Construct a fixed-endpoint `tools/list` request with version 2026-07-28 and
   no client interaction capabilities. No arbitrary method or endpoint input.
2. Accept bounded synthetic JSON response bytes; check status, media type,
   JSON-RPC version, matching integer request ID, result shape and completion.
3. Continue opaque pagination. Reject repeated cursors, duplicate names,
   overlapping requests, invalid JSON/duplicate keys, nonfinite constants,
   unrequested server messages and incomplete listings.
4. Expose only candidate tool names, canonical schema hashes and whether an
   output schema was present, after all pages complete. These are discovery
   evidence, NEVER executable capabilities. Descriptions, instructions and raw
   schemas are neither exposed nor persisted. Schemas are not approved or
   validated as JSON Schema; hashes are not compatibility guarantees.

Limits are 1 MiB per response, 32 pages, 2,048 unique tools and 4 KiB cursors.
Failures clear accumulated evidence and end the state machine. Status 401/403
fails closed without triggering login, scope escalation or retries. Redirects,
SSE, protocol errors and unknown result types are not followed/interpreted.
Request repr and normal model serialization omit all fields, including opaque
cursors. Explicit attributes remain sensitive; do not dump or persist them.

No HTTP sender, browser, socket, token parameter/store, file persistence, logs,
CLI, connected adapter or `tools/call` exists. No external URLs, schema refs or
instructions are executed. `RobinhoodReadOnly` remains disconnected with zero
capabilities. Account reads cannot occur, so missing identity/type/freshness
evidence cannot accidentally produce a successful account observation.

The PR #7 internal read contract still enforces selected identity/type, complete
snapshots and freshness for synthetic fixtures. It is NOT a Robinhood mapping.
No changes to shared providers, Alpaca paper execution, approvals, strategy,
risk, research, or the 13:00 cutoff. See [[Robinhood Read-Only Foundation]].

## Evidence required before the next milestone

- Establish supported protocol versions and official auth metadata/scopes via
  published material or separately authorized public-only metadata discovery.
  Do not guess endpoints, reuse browser cookies, reverse engineer APIs, or use
  third-party brokerage wrappers.
- Obtain official input/output schemas (tool discovery may require auth) and
  documented dedicated-account markers, identity binding, status vocabulary,
  account-data pagination, timestamps and error semantics. Discovery hashes alone
  are insufficient. If no output schema is offered, obtain a documented contract.
- Design/review full bounded HTTP/SSE handling, deadlines, TLS, redirect/SSRF
  denial, schema drift and transport privacy. Pin supported versions; do not
  add permissive fallback based on failures.
- Any future authentication requires separate explicit user authorization and
  validated issuer/resource/PKCE handling. Keep tokens in memory for the consented
  session; no token or financial persistence. Fail closed if least-privilege
  permissions or explicit dedicated-account binding cannot be verified.
- Only then consider separately authorized read acceptance. No mutation tools,
  including order review and approval tools, may be callable. Robinhood order
  placement is outside this milestone and needs a separate safety design.

Offline test command:
`python -B scripts/offline_tests.py tests/test_robinhood_mcp_protocol.py tests/test_read_only_broker.py`
There is no live connection command or activation switch.
