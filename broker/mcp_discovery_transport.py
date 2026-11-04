"""Bounded, unauthenticated tools/list transport. Never wired to broker execution.

Network use is disabled by default. No tokens, cookies, OAuth, tool calls, logging,
or disk persistence. The only endpoint is Robinhood's documented Trading MCP.
"""
import codecs
import json
import math
import multiprocessing
import re
from time import monotonic
from typing import Protocol

from broker.read_only import ReadCode, ReadOnlyError
from broker.robinhood_mcp_protocol import (
    DiscoveryProtocol, DiscoveryRequest, ENDPOINT, VERSION, MAX_BYTES,
    _pairs, _finite_float, _invalid_constant,
)

MAX_LINE = 65_536
MAX_EVENTS = 256


def _error(code=ReadCode.MALFORMED_RESPONSE):
    raise ReadOnlyError(code) from None


def _json(body):
    try:
        return json.loads(body, object_pairs_hook=_pairs, parse_float=_finite_float,
                          parse_constant=_invalid_constant)
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        _error()


def _request(request):
    """Validate again at the network boundary, not just at the request builder."""
    template = DiscoveryProtocol().next_request()
    try:
        if (request.url != ENDPOINT or request.method != "POST"
                or request.headers != template.headers or len(request.body) > 8192):
            _error(ReadCode.UNSUPPORTED_CAPABILITY)
        body = _json(request.body)
        expected = _json(template.body)
        if (set(body) != set(expected) or body["jsonrpc"] != "2.0"
                or body["method"] != "tools/list" or type(body["id"]) is not int
                or not 1 <= body["id"] <= 32 or not isinstance(body["params"], dict)
                or set(body["params"]) - {"_meta", "cursor"}
                or body["params"].get("_meta") != expected["params"]["_meta"]):
            _error(ReadCode.UNSUPPORTED_CAPABILITY)
        if "cursor" in body["params"]:
            cursor = body["params"]["cursor"]
            if not isinstance(cursor, str) or len(cursor.encode()) > 4096:
                _error()
        # Copy/re-encode so caller mutations cannot change the dispatched request.
        return json.dumps(body, allow_nan=False).encode(), body["id"]
    except (AttributeError, KeyError, TypeError, ValueError, UnicodeError):
        _error()


def _remaining(deadline, clock):
    remaining = deadline - clock()
    if not math.isfinite(remaining) or remaining <= 0:
        _error(ReadCode.CONNECTION_FAILED)
    return remaining


def _chunks(response, deadline, clock):
    total = 0
    for chunk in response.iter_content(chunk_size=4096):
        _remaining(deadline, clock)
        if not isinstance(chunk, bytes):
            _error()
        total += len(chunk)
        if total > MAX_BYTES:
            _error()
        yield chunk
    _remaining(deadline, clock)


def _lines(chunks):
    decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
    pending = ""
    for chunk in chunks:
        pending += decoder.decode(chunk)
        while True:
            match = re.search(r"\r\n|\r|\n", pending)
            if not match or (match[0] == "\r" and match.end() == len(pending)):
                break  # Preserve CR split from LF across chunks.
            line, pending = pending[:match.start()], pending[match.end():]
            if len(line) > MAX_LINE:
                _error()
            yield line
        if len(pending) > MAX_LINE:
            _error()
    pending += decoder.decode(b"", final=True)
    if pending.endswith("\r"):
        yield pending[:-1]
    elif pending:
        # Unterminated final event cannot constitute a complete response.
        _error()


def _correlate(message, request_id):
    if (not isinstance(message, dict) or message.get("jsonrpc") != "2.0"
            or type(message.get("id")) is not int or message["id"] != request_id
            or set(message) != {"jsonrpc", "id", "result"}):
        _error()  # Includes JSON-RPC version errors: never downgrade/retry.
    return json.dumps(message, allow_nan=False).encode()


def _decode_response(response, request_id, deadline, clock):
    if response.status_code in (401, 403):
        _error(ReadCode.AUTHENTICATION_FAILED)
    if response.status_code != 200:
        _error(ReadCode.CONNECTION_FAILED)
    if response.headers.get("Content-Encoding", "identity").lower() not in ("", "identity"):
        _error()  # No compressed expansion before byte limits.
    length = response.headers.get("Content-Length")
    if length is not None and (not length.isascii() or not length.isdigit() or int(length) > MAX_BYTES):
        _error()
    media = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    chunks = _chunks(response, deadline, clock)
    if media == "application/json":
        return _correlate(_json(b"".join(chunks)), request_id)
    if media != "text/event-stream":
        _error(ReadCode.UNSUPPORTED_CAPABILITY)
    data, event, count = [], "", 0
    for line in _lines(chunks):
        if not line:
            if data:
                count += 1
                if count > MAX_EVENTS or event not in ("", "message"):
                    _error()
                message = _json("\n".join(data))
                # Only logging notifications are supported, and discarded, never logged.
                # No progress token was requested, so progress notifications are refused.
                if (isinstance(message, dict) and message.get("jsonrpc") == "2.0"
                        and message.get("method") == "notifications/message"
                        and set(message) <= {"jsonrpc", "method", "params"}
                        and isinstance(message.get("params"), dict)):
                    pass
                else:
                    return _correlate(message, request_id)
            data, event = [], ""
        elif not line.startswith(":"):
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                data.append(value)
            elif field == "event":
                event = value
            # id/retry/unknown fields ignored. No reconnection or replay.
    _error()  # EOF without a matching, fully dispatched response.


def _http_exchange(request, deadline, *, session_factory=None, clock=monotonic):
    """Private synchronous worker implementation; injectable for offline tests."""
    body, request_id = _request(request)
    if session_factory is None:
        from requests import Session
        session_factory = Session
    try:
        with session_factory() as session:
            session.trust_env = False
            session.headers.clear()
            session.cookies.clear()
            session.auth = None
            timeout = min(5.0, _remaining(deadline, clock))
            with session.post(ENDPOINT, data=body,
                    headers={**DiscoveryProtocol().next_request().headers, "Accept-Encoding": "identity"},
                    allow_redirects=False, verify=True, stream=True,
                    timeout=(timeout, timeout), proxies={}) as response:
                return _decode_response(response, request_id, deadline, clock)
    except ReadOnlyError:
        raise
    except Exception:
        _error(ReadCode.CONNECTION_FAILED)


def _worker(request, deadline, output, size):
    try:
        body = _http_exchange(request, deadline)
        if len(body) > MAX_BYTES:
            _error()
        output[:len(body)] = body
        size.value = len(body)
    except ReadOnlyError as exc:
        size.value = -1 - list(ReadCode).index(exc.code)
    except Exception:
        size.value = -1 - list(ReadCode).index(ReadCode.CONNECTION_FAILED)


class DiscoveryExchange(Protocol):
    def exchange(self, request: DiscoveryRequest, *, deadline: float) -> bytes: ...


class StreamableHTTPDiscovery:
    def __init__(self, *, enable_network=False):
        self._enabled = enable_network is True

    def exchange(self, request, *, deadline):
        if not self._enabled:
            _error(ReadCode.DISCONNECTED)
        _request(request)
        remaining = min(15.0, _remaining(deadline, monotonic))
        context = multiprocessing.get_context("spawn")
        output = context.Array("B", MAX_BYTES, lock=False)
        size = context.Value("i", 0, lock=False)
        worker = context.Process(target=_worker,
            args=(request, min(deadline, monotonic() + remaining), output, size), daemon=True)
        try:
            worker.start()
            worker.join(min(remaining, _remaining(deadline, monotonic)))
            if worker.is_alive() or size.value == 0:
                _error(ReadCode.CONNECTION_FAILED)
            _remaining(deadline, monotonic)
            if size.value < 0:
                _error(list(ReadCode)[-size.value - 1])
            return bytes(output[:size.value])
        except ReadOnlyError:
            raise
        except Exception:
            _error(ReadCode.CONNECTION_FAILED)
        finally:
            if worker.pid is not None:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(1)
                if worker.is_alive():
                    worker.kill()
                    worker.join(1)
                if not worker.is_alive():
                    worker.close()


def discover(exchange: DiscoveryExchange, *, protocol_version: str, timeout_seconds=30.0):
    """Explicit single supported version. Return discovery evidence, not account data."""
    if protocol_version != VERSION:
        _error(ReadCode.UNSUPPORTED_CAPABILITY)
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 60):
        _error()
    deadline = monotonic() + timeout_seconds
    protocol = DiscoveryProtocol()
    try:
        for _ in range(32):
            _remaining(deadline, monotonic)
            body = exchange.exchange(protocol.next_request(), deadline=deadline)
            _remaining(deadline, monotonic)
            protocol.accept_json_response(status=200, content_type="application/json", body=body)
            if protocol._state == "complete":
                return protocol.evidence()
        _error()
    except ReadOnlyError:
        raise
    except Exception:
        _error(ReadCode.CONNECTION_FAILED)
