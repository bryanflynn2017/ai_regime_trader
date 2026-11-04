"""Offline-only MCP discovery codec; NOT a connected Streamable HTTP client.

Only constructs tools/list envelopes. No socket, HTTP sender, credentials,
tool invocation, authentication, logging, file access, or broker adapter wiring.
Pinned to the documented 2026-07-28 protocol, not a claim Robinhood supports it.
"""
import hashlib
import json
import math
import re

from broker.read_only import PrivateModel, private_field, ReadCode, ReadOnlyError

VERSION = "2026-07-28"
ENDPOINT = "https://agent.robinhood.com/mcp/trading"
# Discovery candidates only. Presence is never authorization to call a tool.
READ_CANDIDATES = frozenset({"get_accounts", "get_portfolio",
                             "get_equity_positions", "get_equity_orders"})
MAX_BYTES = 1_048_576
MAX_PAGES = 32
MAX_TOOLS = 2048


class DiscoveryRequest(PrivateModel):
    url: str = private_field()
    method: str = private_field()
    headers: dict[str, str] = private_field()
    body: bytes = private_field()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _invalid_constant(_):
    raise ValueError


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError
    return number


class DiscoveryProtocol:
    """Single-use paginated discovery state machine driven by synthetic I/O.

    A failed/incomplete listing never exposes partial evidence. Start a new
    instance for another listing. No auto retries, downgrade, or cached auth.
    """
    def __init__(self):
        self._state = "ready"
        self._page = 0
        self._cursor = None
        self._cursors = set()
        self._names = set()
        self._evidence = {}

    def __repr__(self):
        return "DiscoveryProtocol(disconnected=True)"

    def _fail(self, code=ReadCode.MALFORMED_RESPONSE):
        self._state = "failed"
        self._cursor = None
        self._cursors.clear()
        self._names.clear()
        self._evidence.clear()
        raise ReadOnlyError(code) from None

    def next_request(self):
        if self._state != "ready" or self._page >= MAX_PAGES:
            self._fail()
        self._page += 1
        params = {"_meta": {
            "io.modelcontextprotocol/protocolVersion": VERSION,
            "io.modelcontextprotocol/clientInfo": {"name": "ai-regime-read-discovery", "version": "1"},
            "io.modelcontextprotocol/clientCapabilities": {},
        }}
        if self._cursor is not None:
            params["cursor"] = self._cursor  # Opaque, including the empty string.
        body = {"jsonrpc": "2.0", "id": self._page, "method": "tools/list", "params": params}
        self._state = "waiting"
        return DiscoveryRequest(url=ENDPOINT, method="POST", headers={
            "Content-Type": "application/json", "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": VERSION, "Mcp-Method": "tools/list",
        }, body=json.dumps(body, allow_nan=False).encode("utf-8"))

    def accept_json_response(self, *, status: int, content_type: str, body: bytes):
        """Bounded JSON branch only. SSE requires a future reviewed transport.

        Status/type come from a future trusted HTTP layer, never tool content.
        This function never performs requests, follows URLs, or handles tokens.
        """
        if self._state != "waiting":
            self._fail()
        if type(status) is not int:
            self._fail()
        if status in (401, 403):
            self._fail(ReadCode.AUTHENTICATION_FAILED)
        if status != 200:
            self._fail(ReadCode.CONNECTION_FAILED)
        if not isinstance(content_type, str) or content_type.split(";", 1)[0].strip().lower() != "application/json":
            self._fail(ReadCode.UNSUPPORTED_CAPABILITY)
        if not isinstance(body, bytes) or len(body) > MAX_BYTES:
            self._fail()
        try:
            response = json.loads(body.decode("utf-8"), object_pairs_hook=_pairs,
                                  parse_constant=_invalid_constant, parse_float=_finite_float)
            if (not isinstance(response, dict) or response.get("jsonrpc") != "2.0"
                    or type(response.get("id")) is not int or response["id"] != self._page
                    or set(response) != {"jsonrpc", "id", "result"}):
                self._fail()
            result = response["result"]
            if (not isinstance(result, dict) or result.get("resultType") != "complete"
                    or set(result) - {"resultType", "tools", "nextCursor", "ttlMs", "cacheScope", "cacheKey", "_meta"}
                    or not isinstance(result.get("tools"), list)):
                self._fail()
            for tool in result["tools"]:
                if not isinstance(tool, dict):
                    self._fail()
                name = tool.get("name")
                if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name)
                        or name in self._names or len(self._names) >= MAX_TOOLS):
                    self._fail()
                self._names.add(name)
                if name not in READ_CANDIDATES:
                    continue  # Never expose mutation/approval tools or descriptions.
                if (not isinstance(tool.get("inputSchema"), dict)
                        or ("outputSchema" in tool and not isinstance(tool["outputSchema"], dict))):
                    self._fail()
                # This is evidence of bytes seen, NOT schema approval/validation.
                schema = {k: tool[k] for k in ("inputSchema", "outputSchema") if k in tool}
                digest = hashlib.sha256(json.dumps(schema, sort_keys=True,
                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
                self._evidence[name] = (digest, "outputSchema" in tool)
            cursor = result.get("nextCursor")
            if cursor is not None:
                if (not isinstance(cursor, str) or len(cursor.encode("utf-8")) > 4096
                        or cursor in self._cursors or self._page >= MAX_PAGES):
                    self._fail()
                self._cursors.add(cursor)
                self._cursor = cursor
                self._state = "ready"
            else:
                self._state = "complete"
                self._cursor = None
                self._cursors.clear()
                self._names.clear()
        except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
            self._fail()

    def evidence(self):
        if self._state != "complete":
            raise ReadOnlyError(ReadCode.UNSUPPORTED_CAPABILITY)
        return {name: {"schema_sha256": digest, "output_schema_present": present}
                for name, (digest, present) in sorted(self._evidence.items())}
