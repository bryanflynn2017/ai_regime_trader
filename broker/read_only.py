"""Internal read-only contract v1, NOT a provider wire schema or execution adapter.

No transport, credential loading, persistence, CLI, or order mutation is provided.
Financial models intentionally omit ALL fields from repr and standard serialization.
Only explicit attribute access is for trusted in-memory consumers, never logging.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError


class ReadCode(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    CONNECTION_FAILED = "CONNECTION_FAILED"
    ACCOUNT_MISMATCH = "ACCOUNT_MISMATCH"
    ACCOUNT_TYPE_UNSUPPORTED = "ACCOUNT_TYPE_UNSUPPORTED"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    STALE_DATA = "STALE_DATA"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"


class ReadOnlyError(Exception):
    def __init__(self, code: ReadCode):
        # Never accept arbitrary provider messages, IDs, or response bodies.
        self.code = code if isinstance(code, ReadCode) else ReadCode.MALFORMED_RESPONSE
        super().__init__(self.code.value)


def private_field(**kwargs):
    return Field(repr=False, exclude=True, **kwargs)


class PrivateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class Binding(PrivateModel):
    account_id: SecretStr = private_field()
    account_type: Literal["individual", "mcp"] = private_field()
    as_of: datetime = private_field()


class Balances(PrivateModel):
    currency: Literal["USD"] = private_field()
    cash: Decimal = private_field(allow_inf_nan=False)
    equity: Decimal = private_field(ge=0, allow_inf_nan=False)
    buying_power: Decimal = private_field(ge=0, allow_inf_nan=False)


class Position(PrivateModel):
    symbol: str = private_field(pattern=r"^[A-Z][A-Z0-9.\-]{0,14}$")
    asset_type: Literal["stock", "etf"] = private_field()
    quantity: Decimal = private_field(gt=0, allow_inf_nan=False)
    as_of: datetime = private_field()


class Order(PrivateModel):
    order_id: SecretStr = private_field()
    symbol: str = private_field(pattern=r"^[A-Z][A-Z0-9.\-]{0,14}$")
    side: Literal["buy", "sell"] = private_field()
    status: Literal["open", "partially_filled", "filled", "canceled", "rejected", "expired"] = private_field()
    quantity: Decimal = private_field(gt=0, allow_inf_nan=False)
    filled_quantity: Decimal = private_field(ge=0, allow_inf_nan=False)
    as_of: datetime = private_field()


class AccountSnapshot(Binding):
    balances: Balances = private_field()
    positions: tuple[Position, ...] = private_field()
    open_orders: tuple[Order, ...] = private_field()
    complete: bool = private_field(strict=True)


class OrderSnapshot(Binding):
    order: Order = private_field()


class AccountReader(Protocol):
    """Future trusted translator returns contract objects/dicts, NOT raw MCP data.

    snapshot must bind every row to one selected account and exhaust pagination.
    order_status must fail on absence/unknown status, never manufacture a fill.
    Implementations must not have order-submission or cancellation methods.
    """
    def snapshot(self) -> dict: ...
    def order_status(self, order_id: SecretStr) -> dict: ...


def _fresh(stamp, now, max_age):
    if (stamp.tzinfo is None or stamp.utcoffset() is None
            or now.tzinfo is None or now.utcoffset() is None
            or max_age <= timedelta(0)):
        raise ReadOnlyError(ReadCode.MALFORMED_RESPONSE)
    age = now.astimezone(timezone.utc) - stamp.astimezone(timezone.utc)
    if age < timedelta(0) or age > max_age:
        raise ReadOnlyError(ReadCode.STALE_DATA)


def _validate(payload, model, expected_account, expected_type, now, max_age):
    try:
        result = model.model_validate(payload)
        if (not isinstance(expected_account, SecretStr)
                or not expected_account.get_secret_value()
                or result.account_id.get_secret_value() != expected_account.get_secret_value()):
            raise ReadOnlyError(ReadCode.ACCOUNT_MISMATCH)
        if expected_type not in ("individual", "mcp") or result.account_type != expected_type:
            raise ReadOnlyError(ReadCode.ACCOUNT_TYPE_UNSUPPORTED)
        _fresh(result.as_of, now, max_age)
        if isinstance(result, AccountSnapshot):
            if not result.complete:
                raise ReadOnlyError(ReadCode.MALFORMED_RESPONSE)
            if len({p.symbol for p in result.positions}) != len(result.positions):
                raise ReadOnlyError(ReadCode.MALFORMED_RESPONSE)
            orders = result.open_orders
            if any(o.status not in ("open", "partially_filled") for o in orders):
                raise ReadOnlyError(ReadCode.MALFORMED_RESPONSE)
            for position in result.positions:
                _fresh(position.as_of, now, max_age)
        else:
            orders = (result.order,)
        if len({o.order_id.get_secret_value() for o in orders}) != len(orders):
            raise ReadOnlyError(ReadCode.MALFORMED_RESPONSE)
        for order in orders:
            _fresh(order.as_of, now, max_age)
            if (not order.order_id.get_secret_value() or order.filled_quantity > order.quantity
                    or (order.status == "filled" and order.filled_quantity != order.quantity)
                    or (order.status == "partially_filled" and not 0 < order.filled_quantity < order.quantity)):
                raise ReadOnlyError(ReadCode.MALFORMED_RESPONSE)
        return result
    except (ValidationError, ValueError, TypeError, AttributeError):
        raise ReadOnlyError(ReadCode.MALFORMED_RESPONSE) from None


def _invoke(call):
    # Normalize transport failures without retaining raw exception context in output.
    try:
        return call()
    except ReadOnlyError as exc:
        code = exc.code
    except PermissionError:
        code = ReadCode.AUTHENTICATION_FAILED
    except Exception:
        code = ReadCode.CONNECTION_FAILED
    raise ReadOnlyError(code) from None


def read_snapshot(reader: AccountReader, *, expected_account: SecretStr,
                  expected_type: str, now: datetime, max_age=timedelta(seconds=60)):
    return _validate(_invoke(reader.snapshot), AccountSnapshot,
                     expected_account, expected_type, now, max_age)


def read_order_status(reader: AccountReader, order_id: SecretStr, *,
                      expected_account: SecretStr, expected_type: str,
                      now: datetime, max_age=timedelta(seconds=60)):
    if not isinstance(order_id, SecretStr) or not order_id.get_secret_value():
        raise ReadOnlyError(ReadCode.MALFORMED_RESPONSE)
    result = _validate(_invoke(lambda: reader.order_status(order_id)), OrderSnapshot,
                       expected_account, expected_type, now, max_age)
    if result.order.order_id.get_secret_value() != order_id.get_secret_value():
        raise ReadOnlyError(ReadCode.MALFORMED_RESPONSE)
    return result
