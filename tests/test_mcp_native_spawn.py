"""Real Windows-compatible spawn tests. Every child replaces HTTP before work.

Only module-level functions are child targets; importing this module never starts
a process. Parent spies delegate to genuine spawn Processes and shared memory.
"""
import json
import multiprocessing
import os
from time import monotonic
import traceback

import pytest

from broker import mcp_discovery_transport as transport
from broker.read_only import ReadOnlyError
from broker.robinhood_mcp_protocol import DiscoveryProtocol, MAX_BYTES


def _deny_network(*args, **kwargs):
    raise AssertionError("NATIVE_TEST_NETWORK_FORBIDDEN")


def _synthetic_http(request, deadline):
    # This runs in the genuine child, after request/Pydantic pickling.
    encoded, request_id = transport._request(request)
    assert request_id == 1 and isinstance(deadline, float)
    assert deadline > 0
    mode = json.loads(encoded)["params"]["cursor"]
    if mode == "error":
        raise RuntimeError("synthetic-private-worker-body")
    if mode == "crash":
        os._exit(7)
    if mode == "blocked":
        multiprocessing.Event().wait(60)
        raise AssertionError("blocked child was not stopped")
    if mode == "oversized":
        return b"x" * (MAX_BYTES + 1)
    payload = json.dumps({"jsonrpc": "2.0", "id": request_id,
                          "result": {"resultType": "complete", "tools": [], "_meta": {"padding": ""}}}).encode()
    if mode == "boundary":
        payload = payload.replace(b'"padding": ""', b'"padding": "' + b"x" * (MAX_BYTES - len(payload)) + b'"')
        assert len(payload) == MAX_BYTES
    return payload


def _offline_child(request, deadline, output, size, ready):
    # Spawn does not inherit the offline runner's monkeypatches. Install local
    # guards before invoking the production worker; never create a Session.
    import socket
    import http.client
    import urllib.request
    import requests
    socket.create_connection = _deny_network
    socket.getaddrinfo = _deny_network
    socket.socket.connect = _deny_network
    socket.socket.connect_ex = _deny_network
    http.client.HTTPConnection.connect = _deny_network
    urllib.request.urlopen = _deny_network
    requests.Session = _deny_network
    requests.sessions.Session.request = _deny_network
    transport._http_exchange = _synthetic_http
    ready.set()
    transport._worker(request, deadline, output, size)


class ProcessSpy:
    """Observe lifecycle; all execution/waits/kill operations use the real OS."""
    def __init__(self, process, ready, force_kill):
        self.process, self.ready, self.force_kill = process, ready, force_kill
        self.started_pid = None
        self.closed = False
        self.exitcode = None
        self.terminations = self.kills = 0

    @property
    def pid(self): return self.process.pid

    def start(self):
        self.process.start()
        self.started_pid = self.process.pid
        # Explicitly prove the synthetic worker booted, not just that startup timed out.
        assert self.ready.wait(12), "native child did not initialize"

    def join(self, timeout): self.process.join(timeout)
    def is_alive(self): return self.process.is_alive()

    def terminate(self):
        self.terminations += 1
        if not self.force_kill:
            self.process.terminate()

    def kill(self):
        self.kills += 1
        self.process.kill()

    def close(self):
        assert not self.process.is_alive()
        self.exitcode = self.process.exitcode
        self.process.close()
        self.closed = True


class NativeContext:
    def __init__(self, real, force_kill):
        self.real, self.force_kill = real, force_kill
        self.children = []
        self.output = self.size = None

    def Array(self, *args, **kwargs):
        self.output = self.real.Array(*args, **kwargs)
        return self.output

    def Value(self, *args, **kwargs):
        self.size = self.real.Value(*args, **kwargs)
        return self.size

    def Process(self, *, target, args, daemon):
        assert target is transport._worker and daemon is True
        ready = self.real.Event()
        process = self.real.Process(target=_offline_child, args=(*args, ready), daemon=daemon)
        spy = ProcessSpy(process, ready, self.force_kill)
        self.children.append(spy)
        return spy


@pytest.mark.parametrize("mode,force_kill", [
    ("success", False), ("boundary", False), ("oversized", False),
    ("error", False), ("crash", False), ("blocked", False), ("blocked", True),
])
def test_native_spawn_worker_and_cleanup(mode, force_kill, monkeypatch, capfd):
    real = multiprocessing.get_context("spawn")
    context = NativeContext(real, force_kill)
    original_children = {p.pid for p in multiprocessing.active_children()}
    def get_context(method):
        assert method == "spawn"
        return context
    monkeypatch.setattr(transport.multiprocessing, "get_context", get_context)
    request = DiscoveryProtocol().next_request()
    message = json.loads(request.body)
    message["params"]["cursor"] = mode
    request = request.model_copy(update={"body": json.dumps(message).encode()})
    started = monotonic()
    budget = 5 if mode == "blocked" else 14
    try:
        if mode in ("success", "boundary"):
            result = transport.StreamableHTTPDiscovery(enable_network=True).exchange(
                request, deadline=started + budget)
            assert json.loads(result)["id"] == 1
            assert len(result) == context.size.value <= MAX_BYTES
            assert bytes(context.output[:context.size.value]) == result
            if mode == "boundary":
                assert len(result) == MAX_BYTES
        else:
            with pytest.raises(ReadOnlyError) as caught:
                transport.StreamableHTTPDiscovery(enable_network=True).exchange(
                    request, deadline=started + budget)
            rendered = "".join(traceback.format_exception(caught.value))
            assert "synthetic-private-worker-body" not in rendered
            assert "Traceback" not in str(caught.value)
            assert context.size.value <= 0
            assert not any(context.output)  # No error bodies written into shared memory.
        assert len(context.children) == 1
        child = context.children[0]
        assert child.started_pid != os.getpid() and child.ready.is_set()
        assert child.closed and child.exitcode is not None
        with pytest.raises(ValueError):
            _ = child.process.sentinel  # Process.close() released the OS handle.
        if mode == "blocked":
            assert child.terminations == 1
            assert child.kills == int(force_kill)
            assert monotonic() - started < budget + 8  # cleanup + scheduling allowance
        else:
            assert child.terminations == child.kills == 0
            assert child.exitcode == (7 if mode == "crash" else 0)
        assert {p.pid for p in multiprocessing.active_children()} == original_children
        captured = capfd.readouterr()
        assert not captured.out and not captured.err
    finally:
        # Test failure must not itself leak the real synthetic blocked child.
        for child in context.children:
            if not child.closed:
                if child.process.is_alive():
                    child.process.kill()
                child.process.join(3)
                child.process.close()
