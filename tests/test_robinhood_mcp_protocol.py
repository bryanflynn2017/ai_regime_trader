"""Synthetic protocol messages only; no captured Robinhood schemas or account data."""
import json
import logging
import traceback

import pytest

from broker.read_only import ReadOnlyError
from broker.robinhood_read_only import RobinhoodReadOnly
from broker.robinhood_mcp_protocol import DiscoveryProtocol, ENDPOINT, VERSION, MAX_BYTES, MAX_PAGES


def tool(name="get_accounts", **extra):
    return {"name": name, "inputSchema": {"type": "object"}, **extra}


def response(page=1, tools=None, **extra):
    return json.dumps({"jsonrpc": "2.0", "id": page, "result": {
        "resultType": "complete", "tools": [tool()] if tools is None else tools, **extra,
    }}).encode()


def accept(protocol, body, **kwargs):
    protocol.accept_json_response(status=kwargs.get("status", 200),
        content_type=kwargs.get("content_type", "application/json"), body=body)


def test_request_is_fixed_discovery_only_and_has_required_current_metadata():
    protocol = DiscoveryProtocol()
    request = protocol.next_request()
    assert request.url == ENDPOINT and request.method == "POST"
    message = json.loads(request.body)
    assert message["method"] == "tools/list"
    assert message["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"] == VERSION
    assert message["params"]["_meta"]["io.modelcontextprotocol/clientCapabilities"] == {}
    assert request.headers == {"Content-Type": "application/json",
        "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": VERSION,
        "Mcp-Method": "tools/list"}
    assert request.model_dump() == {} and request.model_dump_json() == "{}"
    for name in ("connect", "authenticate", "call_tool", "submit_order", "cancel_order"):
        assert not hasattr(protocol, name)


def test_pagination_including_empty_cursor_is_complete_before_evidence():
    protocol = DiscoveryProtocol()
    protocol.next_request()
    accept(protocol, response(nextCursor=""))
    with pytest.raises(ReadOnlyError):
        protocol.evidence()
    assert json.loads(protocol.next_request().body)["params"]["cursor"] == ""
    accept(protocol, response(2, [tool("get_equity_positions")], nextCursor="opaque-synthetic"))
    assert json.loads(protocol.next_request().body)["params"]["cursor"] == "opaque-synthetic"
    accept(protocol, response(3, [tool("get_equity_orders")]))
    assert set(protocol.evidence()) == {"get_accounts", "get_equity_positions", "get_equity_orders"}
    result = protocol.evidence()
    result.clear()
    assert protocol.evidence()
    assert RobinhoodReadOnly().is_connected is False
    with pytest.raises(ReadOnlyError, match="DISCONNECTED"):
        RobinhoodReadOnly().snapshot()


def test_mutations_annotations_and_descriptions_cannot_expand_allowlist(caplog):
    protocol = DiscoveryProtocol()
    protocol.next_request()
    tools = [tool("get_portfolio", description="synthetic-private-text",
                  outputSchema={"type": "object"})]
    for name in ("place_equity_order", "review_equity_order", "cancel_equity_order",
                 "get_trade_approvals", "decline_trade_approval", "get_trade_approval_setting",
                 "get_accounts_evil", "GET_ACCOUNTS"):
        tools.append(tool(name, annotations={"readOnlyHint": True}))
    accept(protocol, response(tools=tools))
    evidence = protocol.evidence()
    assert set(evidence) == {"get_portfolio"}
    assert evidence["get_portfolio"]["output_schema_present"] is True
    assert len(evidence["get_portfolio"]["schema_sha256"]) == 64
    with caplog.at_level(logging.INFO):
        logging.info("%r %s", protocol, evidence)
    assert "synthetic-private-text" not in caplog.text


@pytest.mark.parametrize("status", [301, 302, 307, 400, 401, 403, 404, 429, 500])
def test_http_failures_do_not_retry_authenticate_follow_or_echo(status):
    protocol = DiscoveryProtocol()
    protocol.next_request()
    private_body = b"synthetic-private-body"
    with pytest.raises(ReadOnlyError) as caught:
        accept(protocol, private_body, status=status)
    assert "synthetic-private-body" not in "".join(traceback.format_exception(caught.value))
    with pytest.raises(ReadOnlyError):
        protocol.next_request()
    with pytest.raises(ReadOnlyError):
        protocol.evidence()


@pytest.mark.parametrize("body", [
    b"not-json", b"[]", b"\xff", b'{"jsonrpc":"2.0","id":1,"id":1,"result":{}}',
    b'{"jsonrpc":"2.0","id":1,"result":{"tools":[],"resultType":"complete","ttlMs":NaN}}',
    b'{"jsonrpc":"2.0","id":1,"result":{"tools":[],"resultType":"complete","ttlMs":1e400}}',
    b'{"jsonrpc":"2.0","id":1,"error":{"code":-1,"message":"synthetic-private-body"}}',
    b'{"jsonrpc":"2.0","id":1,"method":"sampling/createMessage","params":{}}',
    response(page=True), response(page=2),
    response(tools=[tool(), tool()]),
    response(tools=[tool(inputSchema=None)]),
    response(tools=[tool(outputSchema=None)]),
    response(resultType="input_required"),
    response(nextCursor=42), response(nextCursor="x" * 4097),
])
def test_malformed_unknown_or_server_initiated_messages_fail_closed(body):
    protocol = DiscoveryProtocol()
    protocol.next_request()
    with pytest.raises(ReadOnlyError):
        accept(protocol, body)
    with pytest.raises(ReadOnlyError):
        protocol.evidence()


def test_repeated_cursor_or_duplicate_across_pages_clears_evidence():
    for second in (response(2, [], nextCursor="same"), response(2)):
        protocol = DiscoveryProtocol()
        protocol.next_request()
        accept(protocol, response(nextCursor="same"))
        protocol.next_request()
        with pytest.raises(ReadOnlyError):
            accept(protocol, second)
        with pytest.raises(ReadOnlyError):
            protocol.evidence()


def test_bounded_pages_and_body_and_no_sse_fallback():
    protocol = DiscoveryProtocol()
    for page in range(1, MAX_PAGES + 1):
        protocol.next_request()
        if page < MAX_PAGES:
            accept(protocol, response(page, [], nextCursor=str(page)))
        else:
            with pytest.raises(ReadOnlyError):
                accept(protocol, response(page, [], nextCursor=str(page)))
    for body, content_type in ((b" " * (MAX_BYTES + 1), "application/json"),
                               (b"data: {}\n\n", "text/event-stream")):
        protocol = DiscoveryProtocol()
        protocol.next_request()
        with pytest.raises(ReadOnlyError):
            accept(protocol, body, content_type=content_type)


def test_ordering_and_schema_fingerprint_are_deterministic():
    def collect(tools):
        protocol = DiscoveryProtocol()
        protocol.next_request()
        accept(protocol, response(tools=tools))
        return protocol.evidence()
    a, b = tool(), tool("get_portfolio")
    assert collect([a, b]) == collect([b, a])
    a["inputSchema"]["additionalProperties"] = False
    assert collect([a]) != collect([tool()])


def test_out_of_order_response_or_overlapping_request_is_terminal():
    protocol = DiscoveryProtocol()
    with pytest.raises(ReadOnlyError):
        accept(protocol, response())
    with pytest.raises(ReadOnlyError):
        protocol.next_request()
    protocol = DiscoveryProtocol()
    protocol.next_request()
    with pytest.raises(ReadOnlyError):
        protocol.next_request()


def test_empty_listing_null_cursor_and_tool_limit(monkeypatch):
    protocol = DiscoveryProtocol()
    protocol.next_request()
    accept(protocol, response(tools=[], nextCursor=None))
    assert protocol.evidence() == {}
    monkeypatch.setattr("broker.robinhood_mcp_protocol.MAX_TOOLS", 1)
    protocol = DiscoveryProtocol()
    protocol.next_request()
    with pytest.raises(ReadOnlyError):
        accept(protocol, response(tools=[tool(), tool("get_portfolio")]))
