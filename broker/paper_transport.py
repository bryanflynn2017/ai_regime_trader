"""Pinned, bounded HTTP transport and order-schema validation. No live route."""
from math import isfinite
import re
from threading import Lock
from time import monotonic, sleep
from urllib.parse import urlsplit

from requests import Session

from broker.base import BrokerError


class PaperTransport(Session):
    def __init__(self):
        super().__init__()
        self.trust_env = False  # no environment proxy/netrc credential redirection
        self._pacing_lock = Lock()
        self._last_request = 0.0

    def request(self, method, url, **kwargs):
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or parsed.netloc != "paper-api.alpaca.markets"
                or not parsed.path.startswith("/v2/") or parsed.fragment):
            raise BrokerError("Non-paper transport refused")
        kwargs["allow_redirects"] = False
        kwargs["timeout"] = (5, 15)
        kwargs["verify"] = True
        # Bound bursts from reconciliation/cleanup (including several partial
        # fills). One owner, <=150 requests/min steady-state, never auto-retry.
        with self._pacing_lock:
            delay = .4 - (monotonic() - self._last_request)
            if delay > 0:
                sleep(delay)
            self._last_request = monotonic()
        return super().request(method, url, **kwargs)


def validate_order(order):
    """Only long entry limit buys / position-reducing market sells are supported."""
    try:
        if set(order) - {"symbol", "qty", "side", "type", "limit_price", "client_order_id", "valid_until"}:
            raise ValueError
        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", order["symbol"]):
            raise ValueError
        if not re.fullmatch(r"art13-[a-f0-9]{32}", order["client_order_id"]):
            raise ValueError
        qty = float(order["qty"])
        if not isfinite(qty) or qty <= 0:
            raise ValueError
        if order["side"] == "buy" and order["type"] == "limit":
            price = float(order["limit_price"])
            if not isfinite(price) or price <= 0:
                raise ValueError
        elif order["side"] != "sell" or order["type"] != "market":
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise BrokerError("Invalid managed paper order") from None
