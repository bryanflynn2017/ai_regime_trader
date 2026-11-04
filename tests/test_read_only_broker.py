"""Synthetic internal contract fixtures; not captured Robinhood responses."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import logging
import traceback

from pydantic import SecretStr
import pytest

from broker.base import BrokerAdapter
from broker.read_only import ReadCode, ReadOnlyError, read_snapshot, read_order_status
from broker.robinhood_read_only import RobinhoodReadOnly

NOW = datetime(2026, 1, 5, 15, tzinfo=timezone.utc)
IDENTITY = SecretStr("synthetic-account-only")
ORDER = SecretStr("synthetic-order-only")


def fixture():
    return dict(account_id=IDENTITY, account_type="mcp", as_of=NOW,
                balances=dict(currency="USD", cash="100", equity="200", buying_power="100"),
                positions=[dict(symbol="TEST", asset_type="etf", quantity="2", as_of=NOW)],
                open_orders=[dict(order_id=ORDER, symbol="TEST", side="buy", status="open",
                                  quantity="2", filled_quantity="0", as_of=NOW)], complete=True)


class FakeReader:
    def __init__(self, payload=None, error=None):
        self.payload = fixture() if payload is None else payload
        self.error = error

    def snapshot(self):
        if self.error:
            raise self.error
        return deepcopy(self.payload)

    def order_status(self, order_id):
        result = self.snapshot()
        return {key: result[key] for key in ("account_id", "account_type", "as_of")} | {
            "order": result["open_orders"][0]}


def read(reader, **kwargs):
    return read_snapshot(reader, expected_account=IDENTITY, expected_type="mcp", now=NOW, **kwargs)


def test_default_disconnected_and_no_execution_surface():
    reader = RobinhoodReadOnly()
    assert not reader.is_connected and not reader.capabilities
    assert not isinstance(reader, BrokerAdapter)
    for name in ("submit_order", "cancel_order", "cancel_all_orders", "connect", "authenticate"):
        assert not hasattr(reader, name)
    for action in (lambda: read(reader), lambda: reader.order_status(ORDER)):
        with pytest.raises(ReadOnlyError, match="^DISCONNECTED$"):
            action()
    with pytest.raises(ReadOnlyError, match="^UNSUPPORTED_CAPABILITY$"):
        reader.require_capability("submit_order")
    assert reader.log_safe_info()["connected"] is False


def test_valid_snapshot_and_no_financial_serialization(caplog):
    result = read(FakeReader())
    assert result.balances.cash == 100 and result.positions[0].quantity == 2
    assert result.account_id == IDENTITY
    for obj in (result, result.balances, result.positions[0], result.open_orders[0]):
        assert obj.model_dump() == {} and obj.model_dump_json() == "{}"
        with caplog.at_level(logging.INFO):
            logging.info("read contract %s %r", obj, obj)
    assert "synthetic" not in caplog.text and "TEST" not in caplog.text
    assert "100" not in caplog.text
    # Returned data cannot be mutated through normal model assignment.
    with pytest.raises(ValueError):
        result.complete = False


@pytest.mark.parametrize("error,code", [
    (ConnectionError("synthetic-private-response"), ReadCode.CONNECTION_FAILED),
    (PermissionError("synthetic-private-auth"), ReadCode.AUTHENTICATION_FAILED),
    (RuntimeError("synthetic-private-response"), ReadCode.CONNECTION_FAILED),
    (ReadOnlyError(ReadCode.UNSUPPORTED_CAPABILITY), ReadCode.UNSUPPORTED_CAPABILITY),
])
def test_failures_are_closed_and_sanitized(error, code):
    with pytest.raises(ReadOnlyError) as caught:
        read(FakeReader(error=error))
    assert caught.value.code == code
    rendered = "".join(traceback.format_exception(caught.value))
    assert "synthetic-private" not in rendered


@pytest.mark.parametrize("change,code", [
    (lambda d: d.update(account_id="another-synthetic-account"), ReadCode.ACCOUNT_MISMATCH),
    (lambda d: d.update(account_type="individual"), ReadCode.ACCOUNT_TYPE_UNSUPPORTED),
    (lambda d: d.pop("balances"), ReadCode.MALFORMED_RESPONSE),
    (lambda d: d["balances"].pop("buying_power"), ReadCode.MALFORMED_RESPONSE),
    (lambda d: d["balances"].update(cash="NaN"), ReadCode.MALFORMED_RESPONSE),
    (lambda d: d.update(complete=False), ReadCode.MALFORMED_RESPONSE),
    (lambda d: d.update(complete="true"), ReadCode.MALFORMED_RESPONSE),
    (lambda d: d.update(as_of=NOW.replace(tzinfo=None)), ReadCode.MALFORMED_RESPONSE),
    (lambda d: d.update(as_of=NOW + timedelta(seconds=1)), ReadCode.STALE_DATA),
    (lambda d: d["positions"][0].update(as_of=NOW - timedelta(seconds=61)), ReadCode.STALE_DATA),
    (lambda d: d["open_orders"][0].update(as_of=NOW - timedelta(seconds=61)), ReadCode.STALE_DATA),
    (lambda d: d["positions"].append(deepcopy(d["positions"][0])), ReadCode.MALFORMED_RESPONSE),
    (lambda d: d["open_orders"].append(deepcopy(d["open_orders"][0])), ReadCode.MALFORMED_RESPONSE),
    (lambda d: d["open_orders"][0].update(status="new-unmapped-state"), ReadCode.MALFORMED_RESPONSE),
    (lambda d: d["open_orders"][0].update(filled_quantity="3"), ReadCode.MALFORMED_RESPONSE),
    (lambda d: d.update(unrecognized="synthetic-private-response"), ReadCode.MALFORMED_RESPONSE),
])
def test_invalid_snapshots_never_become_empty_or_zero_success(change, code):
    data = fixture()
    change(data)
    with pytest.raises(ReadOnlyError) as caught:
        read(FakeReader(data))
    assert caught.value.code == code
    assert "synthetic-private" not in str(caught.value)


def test_empty_complete_snapshot_is_valid_but_unknown_is_not():
    data = fixture()
    data.update(positions=[], open_orders=[])
    assert not read(FakeReader(data)).positions
    data["positions"] = None
    with pytest.raises(ReadOnlyError):
        read(FakeReader(data))


def test_order_status_identity_and_terminal_semantics():
    data = fixture()
    data["open_orders"][0].update(status="filled", filled_quantity="2")
    kwargs = dict(expected_account=IDENTITY, expected_type="mcp", now=NOW)
    result = read_order_status(FakeReader(data), ORDER, **kwargs)
    assert result.order.status == "filled" and result.model_dump() == {}
    with pytest.raises(ReadOnlyError):
        read_order_status(FakeReader(data), SecretStr("different-synthetic-order"), **kwargs)
    data["open_orders"][0]["filled_quantity"] = "1"
    with pytest.raises(ReadOnlyError):
        read_order_status(FakeReader(data), ORDER, **kwargs)


def test_timezone_and_freshness_boundary():
    data = fixture()
    data["as_of"] = (NOW - timedelta(seconds=60)).astimezone(timezone(timedelta(hours=-5)))
    assert read(FakeReader(data)).as_of == NOW - timedelta(seconds=60)
    with pytest.raises(ReadOnlyError):
        read(FakeReader(), max_age=timedelta(0))


def test_error_cannot_echo_arbitrary_provider_text():
    assert str(ReadOnlyError("synthetic-private-response")) == "MALFORMED_RESPONSE"
