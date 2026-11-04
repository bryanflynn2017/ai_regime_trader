# PR #9: secure MCP discovery transport

Reviewed 2026-11-04. Implemented transport foundation; **not an authenticated
Robinhood connection**. No real endpoint was contacted during development.
No financial observations or credentials were used in tests.

## Official contract

[MCP Streamable HTTP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
defines POST requests with version/method metadata, JSON or request-scoped SSE
responses, and response correlation. This implementation supports only tools/list
at that explicit revision. It does not implement protocol sessions, retries,
version fallback, tool calls, MRTR, subscriptions, or interactive server requests.
JSON-RPC errors (including unsupported-version errors) fail closed. No claim is
made that Robinhood supports this revision.

[Robinhood onboarding](https://robinhood.com/us/en/support/articles/onboarding-an-external-agent/)
documents `https://agent.robinhood.com/mcp/trading` and interactive authentication.
The endpoint is the only allowlisted destination. This PR does not authenticate
or assume public discovery succeeds without authentication. A 401/403 stops the
operation, without scope escalation or token handling.

[WHATWG SSE parsing](https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation)
informs UTF-8/BOM, comment, line-ending, multi-line data and blank-line dispatch
handling. We ignore id/retry/unknown SSE fields and do not reconnect. Only the
default/message event type is supported. Logging notifications are discarded,
not logged. No progress token is requested; progress and other notifications
are refused. A matching final JSON-RPC response ends consumption and closes the
response; following bytes are not interpreted. EOF before a fully dispatched
matching response is failure. Invalid UTF-8 is rejected rather than repaired.

## Implementation and limits

`broker/mcp_discovery_transport.py` adds:

- DiscoveryExchange: injectable exchange interface for synthetic offline tests.
- StreamableHTTPDiscovery: network-disabled by default. Explicit construction
  with enable_network=True is a code-level gate, NOT user authorization. No CLI
  or settings switch enables it. Do not run network discovery without a separate
  user-authorized acceptance task.
- discover: caller must supply the supported protocol version. Runs bounded
  tool-list pagination through the existing protocol validator and returns only
  its filtered discovery evidence. Tool presence/hash never grants permission.
- A private Requests worker validates endpoint, method, headers, version,
  metadata and tools/list request shape again immediately before sending. No
  arbitrary URLs, Authorization headers, cookies, extra RPC methods or redirects.

TLS verification is always enabled. Each page uses a fresh session with
trust_env=False, no environment proxy/netrc, cleared cookies/headers and no auth.
Compression is not accepted, avoiding decompression before size checks. No new
dependencies; Requests remains at the project's existing pin.

Bounds: 1 MiB total response body (including SSE comments/notifications), 64 Ki
characters per SSE line, 256 dispatched data events, and existing 32-page/
2,048-tool discovery limits. Response Content-Length is checked when present;
the actual stream is counted independently. No body is read on HTTP errors or
redirects. Input requests are bounded to 8 KiB; oversized escaped cursors are
conservatively refused. Unknown media types are rejected.

[Requests timeouts](https://requests.readthedocs.io/en/latest/user/advanced/#timeouts)
are inactivity/connect timeouts, not whole-operation deadlines. Consequently,
real I/O runs in a fresh spawn worker with a 15-second page cap and an overall
monotonic discovery deadline (30 seconds default, maximum 60). The parent waits
only to its deadline, then terminates/kills an unfinished worker. This covers
stalled DNS/header reads and slow-drip streams that inactivity timeouts alone
cannot bound. Cleanup can add up to two one-second waits plus OS scheduling
overhead; this is not a hard-real-time guarantee. Per-page HTTP connect/read
timeouts are also capped at five seconds. No automatic retry occurs.

Bounded shared memory carries only the final discovery JSON or a fixed error
code back to the parent; no files, raw exception text or logs. Worker/session/
response cleanup occurs on success and failure. Tests inject both fake HTTP
sessions and fake process contexts; they never launch a network worker. Spawn
requires a normal guarded Python entry point if used in future acceptance.

## Security boundaries and remaining work

The RobinhoodReadOnly adapter remains unchanged and disconnected. No account
read tools are called, so this transport does not validate actual account
identity, MCP account type, financial pagination or position freshness. The
existing synthetic internal contract is not a provider response mapping.

Before any genuine account connection:

1. Obtain official supported-version and schema evidence, including dedicated
   account markers, account-bound responses, pagination and timestamp semantics.
2. Separately authorize public discovery/acceptance; inspect only sanitized
   metadata. Do not put provider responses, cookies or identifiers into fixtures.
3. Design/review OAuth issuer/resource/PKCE validation and least-privilege scopes.
   Authentication requires separate explicit user authorization. No login,
   bearer-token parameter, token refresh/storage or browser flow exists here.
4. Add a separately reviewed read-only schema translator and identity/freshness
   checks before enabling any financial reads. Keep all trading/review/cancel/
   approval capabilities absent, even if the server advertises them.

No Alpaca routing, approvals, risk, strategy, research, runtime records, or
13:00 cutoff changed. No brokerage data retrieved or orders submitted.

Offline validation:
`python -B scripts/offline_tests.py tests/test_mcp_discovery_transport.py tests/test_robinhood_mcp_protocol.py tests/test_read_only_broker.py`

See [[Robinhood MCP Protocol Foundation]] and [[Robinhood Read-Only Foundation]].
