"""Synthetic-only regression tests; no credential or broker network access."""
import io
import json
import logging
import traceback
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr, ValidationError
from requests.exceptions import ConnectionError, Timeout

from config.settings import Settings
from utils.provider_errors import safe_provider_error

FIELDS = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "DISCORD_WEBHOOK_URL")


def markers():
    return {name: "synthetic-only-" + name.lower() for name in FIELDS}


def assert_redacted(settings):
    values = markers().values()
    outputs = (repr(settings), str(settings), repr(dict(settings)),
               repr(settings.model_dump()), settings.model_dump_json(),
               json.dumps(settings.model_dump(mode="json")),
               repr(settings.model_dump(serialize_as_any=True)))
    for output in outputs:
        assert all(value not in output for value in values)
    for name in FIELDS:
        assert isinstance(getattr(settings, name), SecretStr)
        assert settings.model_dump()[name] == "**********"


def test_settings_redact_all_serialization_and_logging(caplog):
    s = Settings(_env_file=None, **markers())
    assert_redacted(s)
    logging.getLogger("security-test").warning("settings=%r dump=%s", s, s.model_dump())
    assert all(value not in caplog.text for value in markers().values())
    from loguru import logger
    sink = io.StringIO()
    handle = logger.add(sink)
    try:
        logger.info("settings={} dump={}", s, s.model_dump())
    finally:
        logger.remove(handle)
    assert all(value not in sink.getvalue() for value in markers().values())
    # Deprecated APIs still route through the same redaction.
    with pytest.warns(DeprecationWarning):
        assert all(v not in repr(s.dict()) for v in markers().values())
    with pytest.warns(DeprecationWarning):
        assert all(v not in s.json() for v in markers().values())


def test_secret_assignment_and_copy_are_redacted():
    s = Settings(_env_file=None)
    for name, value in markers().items():
        setattr(s, name, value)
    assert_redacted(s)
    assert_redacted(Settings(_env_file=None).model_copy(update=markers(), deep=True))
    assert s.has_broker_credentials()
    s.ALPACA_API_KEY = ""
    assert not s.has_broker_credentials()
    assert s.model_dump()["ALPACA_API_KEY"] == ""


@pytest.mark.parametrize("source", ["env", "dotenv", "constructor"])
def test_configuration_paths_preserve_values_without_exposing_them(source, monkeypatch, tmp_path):
    for name in FIELDS:
        monkeypatch.delenv(name, raising=False)
    if source == "env":
        for name, value in markers().items():
            monkeypatch.setenv(name, value)
        s = Settings(_env_file=None)
    elif source == "dotenv":
        path = tmp_path / "synthetic.env"
        path.write_text("\n".join(f"{k}={v}" for k, v in markers().items()))
        s = Settings(_env_file=path)
    else:
        s = Settings(_env_file=None, **markers())
    assert_redacted(s)
    assert all(getattr(s, k).get_secret_value() == v for k, v in markers().items())
    assert s.EXECUTION_MODE == "shadow" and not s.LIVE_TRADING_ALLOWED


def test_invalid_configuration_error_text_does_not_echo_inputs():
    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None, **markers(), EXECUTION_MODE=markers()[FIELDS[0]])
    assert all(v not in str(caught.value) for v in markers().values())


@pytest.mark.parametrize("via_copy", [True, False])
def test_invalid_secret_update_does_not_echo_input(via_copy):
    s = Settings(_env_file=None)
    value = {"sensitive": markers()[FIELDS[0]]}
    with pytest.raises(ValidationError) as caught:
        if via_copy:
            s.model_copy(update={FIELDS[0]: value})
        else:
            s.ALPACA_API_KEY = value
    assert markers()[FIELDS[0]] not in str(caught.value)


def test_sdk_and_discord_boundaries_receive_original_values(monkeypatch):
    from broker import alpaca_client
    from data_providers.alpaca_provider import AlpacaDataProvider
    from alerts import discord_alerts
    import config.settings as config_module
    import alpaca.data.historical
    s = Settings(_env_file=None, **markers())
    trading = MagicMock()
    historical = MagicMock()
    monkeypatch.setattr(alpaca_client, "_ALPACA_AVAILABLE", True)
    monkeypatch.setattr(alpaca_client, "_TradingClient", trading)
    monkeypatch.setattr(alpaca.data.historical, "StockHistoricalDataClient", historical)
    monkeypatch.setattr(config_module, "settings", s)
    alpaca_client.AlpacaClientWrapper(s).get_trading_client()
    trading.assert_called_once_with(api_key=markers()[FIELDS[0]], secret_key=markers()[FIELDS[1]], paper=True)
    AlpacaDataProvider(settings_override=s)._get_client()
    historical.assert_called_once_with(markers()[FIELDS[0]], markers()[FIELDS[1]])
    assert discord_alerts._webhook_url() == markers()[FIELDS[2]]
    assert markers()[FIELDS[2]] not in discord_alerts._sanitize_error(markers()[FIELDS[2]])
    trading.return_value.submit_order.assert_not_called()


class SensitiveProviderError(Exception):
    def __init__(self, status=None):
        super().__init__("Authorization: synthetic-bearer; account: synthetic-account; " + " ".join(markers().values()))
        self.status_code = status


@pytest.mark.parametrize("error,category", [
    (SensitiveProviderError(401), "AUTHENTICATION_OR_PERMISSION"),
    (SensitiveProviderError(403), "AUTHENTICATION_OR_PERMISSION"),
    (SensitiveProviderError(429), "RATE_LIMIT"),
    (SensitiveProviderError(404), "NOT_FOUND"),
    (SensitiveProviderError(503), "PROVIDER_UNAVAILABLE"),
    (SensitiveProviderError(422), "REQUEST_REJECTED"),
    (Timeout("synthetic-bearer"), "TIMEOUT"),
    (ConnectionError("synthetic-bearer"), "CONNECTION"),
    (SensitiveProviderError("synthetic-bearer"), "PROVIDER_ERROR"),
])
def test_allowlisted_error_categories(error, category):
    text = safe_provider_error("Read failed", error)
    assert category in text
    assert all(v not in text for v in [*markers().values(), "synthetic-bearer", "synthetic-account"])


@pytest.mark.parametrize("method,sdk_method,kwargs", [
    ("get_account", "get_account", {}),
    ("get_clock", "get_clock", {}),
    ("list_positions", "get_all_positions", {}),
    ("get_orders", "get_orders", {}),
    ("get_order_by_client_id", "get_order_by_client_id", {"client_order_id": "synthetic-id"}),
    ("verify_paper_account", "get_account", {}),
])
def test_broker_errors_and_tracebacks_never_echo_provider_content(method, sdk_method, kwargs, caplog):
    from broker.alpaca_client import AlpacaClientWrapper, AlpacaConnectionError
    client = MagicMock()
    client._base_url = "https://paper-api.alpaca.markets"
    getattr(client, sdk_method).side_effect = SensitiveProviderError(403)
    wrapper = AlpacaClientWrapper(Settings(_env_file=None, **markers()), _client_for_testing=client)
    with pytest.raises(AlpacaConnectionError) as caught:
        getattr(wrapper, method)(**kwargs)
    text = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    logging.getLogger("security-test").error("%s", caught.value)
    assert "AUTHENTICATION_OR_PERMISSION" in text
    assert all(v not in text + caplog.text for v in [*markers().values(), "synthetic-bearer", "synthetic-account"])
    assert caught.value.__suppress_context__
    client.submit_order.assert_not_called()


def test_sdk_constructor_error_is_sanitized(monkeypatch):
    from broker import alpaca_client
    monkeypatch.setattr(alpaca_client, "_ALPACA_AVAILABLE", True)
    monkeypatch.setattr(alpaca_client, "_TradingClient", MagicMock(side_effect=SensitiveProviderError(401)))
    with pytest.raises(alpaca_client.AlpacaConnectionError) as caught:
        alpaca_client.AlpacaClientWrapper(Settings(_env_file=None, **markers())).get_trading_client()
    assert "AUTHENTICATION_OR_PERMISSION" in str(caught.value)
    assert all(v not in str(caught.value) for v in markers().values())
    assert caught.value.__suppress_context__


def test_order_lookup_absence_and_uncertainty_semantics_unchanged():
    from broker.alpaca_client import AlpacaClientWrapper, AlpacaConnectionError
    client = MagicMock()
    wrapper = AlpacaClientWrapper(Settings(_env_file=None), _client_for_testing=client)
    client.get_order_by_client_id.side_effect = SensitiveProviderError(404)
    assert wrapper.get_order_by_client_id("synthetic-id") is None
    client.get_order_by_client_id.side_effect = Timeout("synthetic-bearer")
    with pytest.raises(AlpacaConnectionError, match="TIMEOUT"):
        wrapper.get_order_by_client_id("synthetic-id")


def test_error_classifier_never_stringifies_response():
    class UnprintableError(Exception):
        response = SimpleNamespace(status_code=429)

        def __str__(self):
            raise AssertionError("Do not stringify provider errors")
    assert "RATE_LIMIT" in safe_provider_error("Read failed", UnprintableError())


@pytest.mark.parametrize("latest", [True, False])
def test_market_data_sdk_failures_are_sanitized(latest):
    from data_providers.alpaca_provider import AlpacaDataProvider
    from data_providers.base import DataProviderError
    client = MagicMock()
    method = "get_stock_latest_bar" if latest else "get_stock_bars"
    getattr(client, method).side_effect = SensitiveProviderError(403)
    provider = AlpacaDataProvider(client=client, feed="sip")
    with pytest.raises(DataProviderError) as caught:
        if latest:
            provider.get_latest_bar("SPY")
        else:
            provider.get_bars("SPY")
    text = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert "AUTHENTICATION_OR_PERMISSION" in text and "--alpaca-feed iex" in text
    assert all(v not in text for v in [*markers().values(), "synthetic-account", "synthetic-bearer"])


def test_market_data_constructor_failure_is_sanitized(monkeypatch):
    import alpaca.data.historical
    from data_providers.alpaca_provider import AlpacaDataProvider
    from data_providers.base import DataProviderError
    monkeypatch.setattr(alpaca.data.historical, "StockHistoricalDataClient", MagicMock(side_effect=SensitiveProviderError(401)))
    with pytest.raises(DataProviderError) as caught:
        AlpacaDataProvider(settings_override=Settings(_env_file=None, **markers()))._get_client()
    assert "AUTHENTICATION_OR_PERMISSION" in str(caught.value)
    assert all(v not in str(caught.value) for v in markers().values())
