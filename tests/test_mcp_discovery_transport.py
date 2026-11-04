"""No network, accounts, captured responses or tokens: synthetic transport tests."""
import json
from types import SimpleNamespace
from time import monotonic

import pytest

import broker.mcp_discovery_transport as transport
from broker.read_only import ReadOnlyError
from broker.robinhood_mcp_protocol import DiscoveryProtocol, VERSION, MAX_BYTES
from broker.robinhood_read_only import RobinhoodReadOnly


def body(request_id=1, **extra):
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "result": {
        "resultType": "complete", "tools": [], **extra}}).encode()


class Response:
    def __init__(self, chunks, *, media="application/json", status=200, headers=None):
        self.status_code = status
        self.headers = {"Content-Type": media, **(headers or {})}
        self.chunks = chunks
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def iter_content(self, **kwargs):
        yield from self.chunks


class Session:
    def __init__(self, response):
        self.response = response
        self.headers, self.cookies = {"unwanted": "synthetic"}, {"unwanted": "synthetic"}
        self.calls, self.closed = [], False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def run(response, request=None, clock=lambda: 0):
    session = Session(response)
    result = transport._http_exchange(request or DiscoveryProtocol().next_request(), 10,
                                     session_factory=lambda: session, clock=clock)
    return result, session


def test_json_tls_redirect_proxy_cookie_and_cleanup_policy():
    response = Response([body()[:20], body()[20:]])
    result, session = run(response)
    assert json.loads(result)["id"] == 1
    url, kwargs = session.calls[0]
    assert url == transport.ENDPOINT
    assert kwargs["verify"] is True and kwargs["allow_redirects"] is False
    assert kwargs["stream"] is True and kwargs["timeout"] == (5.0, 5.0)
    assert kwargs["proxies"] == {} and kwargs["headers"]["Accept-Encoding"] == "identity"
    assert not session.trust_env and session.auth is None
    assert not session.headers and not session.cookies
    assert session.closed and response.closed


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
def test_sse_bom_comments_multiline_and_arbitrary_chunk_boundaries(newline):
    payload = body().replace(b', "id"', b',\n "id"')
    stream = b"\xef\xbb\xbf: keepalive" + newline + newline
    stream += b"event: message" + newline
    stream += b"id: ignored" + newline + b"retry: 10" + newline
    stream += newline.join(b"data: " + part for part in payload.split(b"\n")) + newline + newline
    result, session = run(Response([bytes([b]) for b in stream], media="text/event-stream"))
    assert json.loads(result)["id"] == 1 and session.closed


def test_sse_discards_log_notification_and_closes_at_correlated_result():
    notification = json.dumps({"jsonrpc": "2.0", "method": "notifications/message",
                               "params": {"data": "synthetic-private"}}).encode()
    stream = b"data: " + notification + b"\n\ndata: " + body() + b"\n\n"
    result, session = run(Response([stream], media="text/event-stream"))
    assert b"synthetic-private" not in result and session.closed


@pytest.mark.parametrize("stream", [
    b"data: not-json\n\n", b"data: " + body(), b"data: " + body() + b"\n",
    b"data: " + body(2) + b"\n\n", b"data: []\n\n", b"data: \xff\n\n",
    b"event: unsupported\ndata: " + body() + b"\n\n",
    b'data: {"jsonrpc":"2.0","id":1,"method":"sampling/createMessage"}\n\n',
    b'data: {"jsonrpc":"2.0","method":"notifications/progress","params":{}}\n\n',
    b'data: {"jsonrpc":"2.0","id":1,"error":{"code":-32601}}\n\n',
])
def test_malformed_sse_and_correlation_fail_closed(stream):
    response = Response([stream], media="text/event-stream")
    with pytest.raises(ReadOnlyError):
        run(response)
    assert response.closed


@pytest.mark.parametrize("status", [301, 302, 307, 308, 400, 401, 403, 404, 429, 500])
def test_no_redirect_auth_retry_or_body_echo(status):
    response = Response([b"synthetic-private"], status=status, headers={"Location": "https://example.invalid"})
    with pytest.raises(ReadOnlyError) as caught:
        run(response)
    assert "synthetic-private" not in str(caught.value)
    assert response.closed


@pytest.mark.parametrize("response", [
    Response([body()], media="text/html"), Response([body(2)]),
    Response([b"x" * (MAX_BYTES + 1)]),
    Response([body()], headers={"Content-Length": str(MAX_BYTES + 1)}),
    Response([body()], headers={"Content-Encoding": "gzip"}),
    Response([b"data: " + b"x" * transport.MAX_LINE], media="text/event-stream"),
])
def test_size_media_compression_and_id_limits(response):
    with pytest.raises(ReadOnlyError):
        run(response)
    assert response.closed


@pytest.mark.parametrize("change", [
    {"url": "http://agent.robinhood.com/mcp/trading"},
    {"url": "https://agent.robinhood.com/mcp/trading?token=synthetic"},
    {"url": "https://example.invalid/mcp/trading"}, {"method": "GET"},
    {"headers": {"Authorization": "synthetic"}},
    {"body": b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{}}'},
])
def test_request_boundary_rejects_mutation_before_io(change):
    request = DiscoveryProtocol().next_request().model_copy(update=change)
    with pytest.raises(ReadOnlyError):
        transport._http_exchange(request, 10, session_factory=lambda: pytest.fail("I/O attempted"))


def test_deadline_stream_failures_and_notification_bound(monkeypatch):
    times = iter([0, 11])
    with pytest.raises(ReadOnlyError):
        run(Response([body()]), clock=lambda: next(times))
    def broken():
        yield b"partial"
        raise TimeoutError("synthetic-private")
    with pytest.raises(ReadOnlyError) as caught:
        run(Response(broken()))
    assert "synthetic-private" not in str(caught.value)
    monkeypatch.setattr(transport, "MAX_EVENTS", 1)
    notification = b'data: {"jsonrpc":"2.0","method":"notifications/message","params":{}}\n\n'
    with pytest.raises(ReadOnlyError):
        run(Response([notification * 2], media="text/event-stream"))


def test_default_disconnected_and_explicit_version_no_downgrade():
    with pytest.raises(ReadOnlyError, match="DISCONNECTED"):
        transport.discover(transport.StreamableHTTPDiscovery(), protocol_version=VERSION)
    class Fake:
        def exchange(self, request, **kwargs):
            request_id = json.loads(request.body)["id"]
            return body(request_id, **({"nextCursor": ""} if request_id == 1 else {}))
    assert transport.discover(Fake(), protocol_version=VERSION) == {}
    with pytest.raises(ReadOnlyError, match="UNSUPPORTED_CAPABILITY"):
        transport.discover(Fake(), protocol_version="2025-11-25")
    assert not RobinhoodReadOnly().is_connected


def test_parent_deadline_terminates_worker_without_network(monkeypatch):
    class Worker:
        pid = None
        alive = True
        def start(self): self.pid = 1
        def join(self, seconds): pass
        def is_alive(self): return self.alive
        def terminate(self): self.alive = False
        def close(self): self.closed = True
    worker = Worker()
    context = SimpleNamespace(Array=lambda *a, **k: bytearray(MAX_BYTES),
        Value=lambda *a, **k: SimpleNamespace(value=0), Process=lambda **k: worker)
    monkeypatch.setattr(transport.multiprocessing, "get_context", lambda _: context)
    with pytest.raises(ReadOnlyError):
        transport.StreamableHTTPDiscovery(enable_network=True).exchange(
            DiscoveryProtocol().next_request(), deadline=monotonic() + 5)
    assert not worker.alive and worker.closed


def test_worker_returns_only_bounded_result_or_fixed_error(monkeypatch):
    output, size = bytearray(MAX_BYTES), SimpleNamespace(value=0)
    monkeypatch.setattr(transport, "_http_exchange", lambda *a: body())
    transport._worker(None, 0, output, size)
    assert bytes(output[:size.value]) == body()
    def fail(*args): raise RuntimeError("synthetic-private")
    monkeypatch.setattr(transport, "_http_exchange", fail)
    transport._worker(None, 0, output, size)
    assert size.value < 0


def test_dispatch_uses_frozen_policy_not_mutable_caller_headers():
    request = DiscoveryProtocol().next_request()
    class MutatingSession(Session):
        def __enter__(self):
            request.headers["Authorization"] = "synthetic"
            request.headers["Mcp-Method"] = "tools/call"
            return self
    session = MutatingSession(Response([body()]))
    transport._http_exchange(request, 10, session_factory=lambda: session, clock=lambda: 0)
    headers = session.calls[0][1]["headers"]
    assert "Authorization" not in headers and headers["Mcp-Method"] == "tools/list"
