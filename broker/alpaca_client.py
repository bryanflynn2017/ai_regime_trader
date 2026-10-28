"""
Alpaca paper trading client wrapper.

Safety hierarchy — every method that touches the broker checks these first:
  PAPER_TRADING == True
  LIVE_TRADING_ALLOWED == False
  ALPACA_PAPER == True
  ALPACA_BASE_URL is exactly the HTTPS paper endpoint

If ANY of these checks fail, AlpacaSafetyError is raised and no order is sent.

alpaca-py availability is detected at import time.
All real broker calls require alpaca-py to be installed.
dry_run mode works without alpaca-py and without credentials.
Secrets are never logged, printed, or included in any repr.
"""
from __future__ import annotations

import logging
import hashlib
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional

from broker.base import OrderNotSubmitted

try:
    from loguru import logger  # type: ignore
except ImportError:
    logger = logging.getLogger(__name__)  # type: ignore

# ---------------------------------------------------------------------------
# Optional heavy dependency — graceful degradation if absent
# ---------------------------------------------------------------------------

try:
    from alpaca.trading.client import TradingClient as _TradingClient  # type: ignore
    from alpaca.trading.requests import (  # type: ignore
        MarketOrderRequest as _MarketOrderRequest,
        LimitOrderRequest as _LimitOrderRequest,
        GetOrdersRequest as _GetOrdersRequest,
    )
    from alpaca.trading.enums import (  # type: ignore
        QueryOrderStatus as _QueryOrderStatus,
    )
    _ALPACA_AVAILABLE = True
    logger.debug("alpaca-py loaded successfully.")
except ImportError:
    _ALPACA_AVAILABLE = False
    logger.info(
        "alpaca-py not installed. Broker calls will raise AlpacaConnectionError. "
        "dry_run mode works without alpaca-py."
    )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AlpacaConnectionError(Exception):
    """
    Raised when a broker API call fails or alpaca-py is unavailable.
    Always includes a human-readable explanation and remediation hint.
    """


class AlpacaSafetyError(OrderNotSubmitted):
    """
    Raised when a paper-trading safety check fails.
    Trading must stop immediately when this is raised.
    """


# ---------------------------------------------------------------------------
# AlpacaClientWrapper
# ---------------------------------------------------------------------------

class AlpacaClientWrapper:
    """
    Safe wrapper around alpaca-py's TradingClient.

    Inject `_client_for_testing` to bypass the real Alpaca connection in tests.
    All real API calls are guarded by validate_paper_safety().
    Secrets are never included in __repr__, log_safe_info(), or any log output.
    """

    def __init__(
        self,
        settings_override: Any = None,
        dry_run: bool = True,
        _client_for_testing: Any = None,   # Inject a mock in tests
    ) -> None:
        if settings_override is None:
            from config.settings import settings
            self._s = settings
        else:
            self._s = settings_override

        self.dry_run = dry_run
        # _client_for_testing allows injecting a mock TradingClient in unit tests
        self._trading_client: Any = _client_for_testing

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Return True if alpaca-py is installed."""
        return _ALPACA_AVAILABLE

    def is_configured(self) -> bool:
        """Return True if both API key and secret key are non-empty."""
        return bool(self._s.ALPACA_API_KEY and self._s.ALPACA_SECRET_KEY)

    def log_safe_info(self) -> Dict[str, Any]:
        """
        Return a dict safe for logging.
        NEVER includes the actual key or secret values.
        """
        return {
            "alpaca_py_available": _ALPACA_AVAILABLE,
            "configured":          self.is_configured(),
            "paper":               self._s.ALPACA_PAPER,
            "base_url":            "https://paper-api.alpaca.markets" if self._s.ALPACA_BASE_URL.rstrip("/") == "https://paper-api.alpaca.markets" else "INVALID",
            "paper_trading":       self._s.PAPER_TRADING,
            "live_trading_allowed":self._s.LIVE_TRADING_ALLOWED,
            "api_key_set":         bool(self._s.ALPACA_API_KEY),
            "secret_key_set":      bool(self._s.ALPACA_SECRET_KEY),
            # Deliberately NOT including actual key or secret values
        }

    def __repr__(self) -> str:
        return (
            f"AlpacaClientWrapper("
            f"configured={self.is_configured()}, "
            f"dry_run={self.dry_run}, "
            f"paper={self._s.ALPACA_PAPER}"
            f")"
        )

    # ------------------------------------------------------------------
    # Safety gate
    # ------------------------------------------------------------------

    def validate_paper_safety(self) -> None:
        """
        Assert that all paper-trading safety conditions are met.

        Raises AlpacaSafetyError if any condition fails.
        This check is a no-op for settings (no network call), so it can be
        called freely in both dry_run and live paths.
        """
        errors: List[str] = []

        if not self._s.PAPER_TRADING:
            errors.append("PAPER_TRADING is not True")

        if self._s.LIVE_TRADING_ALLOWED:
            errors.append("LIVE_TRADING_ALLOWED is True")

        if not self._s.ALPACA_PAPER:
            errors.append("ALPACA_PAPER is not True")

        if self._s.ALPACA_BASE_URL.rstrip("/") != "https://paper-api.alpaca.markets":
            errors.append(
                "ALPACA_BASE_URL must be exactly https://paper-api.alpaca.markets"
            )

        if errors:
            raise AlpacaSafetyError(
                "Paper trading safety check failed: " + "; ".join(errors) + ". "
                "Resolve these settings before any broker call."
            )

    # ------------------------------------------------------------------
    # Client factory
    # ------------------------------------------------------------------

    def get_trading_client(self) -> Any:
        """
        Return an initialised alpaca-py TradingClient.

        The client is created lazily and cached.  Calls validate_paper_safety()
        first so it is impossible to create a live client by accident.

        Raises:
            AlpacaSafetyError:    If any paper-safety check fails.
            AlpacaConnectionError: If alpaca-py is missing or credentials absent.
        """
        self.validate_paper_safety()
        if self._trading_client is not None:
            return self._trading_client

        self.validate_paper_safety()

        if not self.is_configured():
            raise AlpacaConnectionError(
                "Alpaca credentials not configured. "
                "Copy .env.example to .env and fill in "
                "ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )

        if not _ALPACA_AVAILABLE:
            raise AlpacaConnectionError(
                "alpaca-py is not installed. "
                "Run: pip install alpaca-py  "
                "(requires 64-bit Python or a system with build tools)."
            )

        self._trading_client = _TradingClient(
            api_key=self._s.ALPACA_API_KEY,
            secret_key=self._s.ALPACA_SECRET_KEY,
            paper=True,  # ALWAYS paper — never live
        )
        from broker.paper_transport import PaperTransport
        self._trading_client._session = PaperTransport()
        # Never automatically replay an ambiguously acknowledged POST.
        self._trading_client._retry = 0
        logger.info("Alpaca TradingClient initialised (paper=True).")
        return self._trading_client

    # ------------------------------------------------------------------
    # Account & clock
    # ------------------------------------------------------------------

    def get_account(self) -> Dict[str, Any]:
        """Return a dict with key account fields (no secrets included)."""
        client = self.get_trading_client()
        try:
            acct = client.get_account()
            return {
                "status":         str(acct.status),
                "equity":         str(acct.equity),
                "cash":           str(acct.cash),
                "buying_power":   str(acct.buying_power),
                "currency":       str(getattr(acct, "currency", "USD")),
                "paper":          True,
                "account_number": str(getattr(acct, "account_number", "")),
            }
        except AlpacaConnectionError:
            raise
        except Exception as exc:
            raise AlpacaConnectionError(
                f"Failed to retrieve account: {exc}"
            ) from exc

    def get_clock(self) -> Dict[str, Any]:
        """Return market clock information."""
        client = self.get_trading_client()
        try:
            clock = client.get_clock()
            return {
                "is_open":   bool(clock.is_open),
                "timestamp": str(clock.timestamp),
                "next_open": str(clock.next_open),
                "next_close":str(clock.next_close),
            }
        except AlpacaConnectionError:
            raise
        except Exception as exc:
            raise AlpacaConnectionError(
                f"Failed to retrieve market clock: {exc}"
            ) from exc

    def is_market_open(self) -> bool:
        """Return True if the US equity market is currently open."""
        return bool(self.get_clock().get("is_open", False))

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def list_positions(self) -> List[Dict[str, Any]]:
        """Return all open positions as a list of dicts."""
        client = self.get_trading_client()
        try:
            positions = client.get_all_positions()
            return [self._position_to_dict(p) for p in positions]
        except AlpacaConnectionError:
            raise
        except Exception as exc:
            raise AlpacaConnectionError(
                f"Failed to list positions: {exc}"
            ) from exc

    def get_position(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return the open position for `ticker`, or None if no position."""
        client = self.get_trading_client()
        try:
            pos = client.get_open_position(ticker.upper())
            return self._position_to_dict(pos)
        except AlpacaConnectionError:
            raise
        except Exception:
            return None  # No position for this ticker

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def get_orders(self, status: str = "all", limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent orders as a list of dicts."""
        client = self.get_trading_client()
        try:
            if _ALPACA_AVAILABLE:
                try:
                    q = _QueryOrderStatus(status)
                except Exception:
                    q = None
                if q is not None:
                    req = _GetOrdersRequest(status=q, limit=limit)
                    orders = client.get_orders(filter=req)
                else:
                    orders = client.get_orders()
            else:
                orders = client.get_orders()
            return [self._order_to_dict(o) for o in orders]
        except AlpacaConnectionError:
            raise
        except Exception as exc:
            raise AlpacaConnectionError(
                f"Failed to retrieve orders: {exc}"
            ) from exc

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by ID.  Returns True if cancelled."""
        if self.dry_run or self._s.EXECUTION_MODE != "paper":
            raise AlpacaSafetyError("Explicit paper cancellation activation required")
        self.verify_paper_account()
        client = self.get_trading_client()
        try:
            client.cancel_order_by_id(order_id)
            return True
        except Exception:
            logger.warning("Paper cancellation acknowledgement unavailable")
            return False

    def cancel_all_orders(self) -> int:
        """Account-wide mutation is forbidden: cancel only durable owned orders."""
        raise AlpacaSafetyError("Account-wide cancellation disabled; use managed paper-session")

    # ------------------------------------------------------------------
    # Order submission (called by OrderExecutor for real trades)
    # ------------------------------------------------------------------

    def submit_bracket_buy(
        self,
        ticker:           str,
        qty:              float,
        stop_loss_price:  float,
        take_profit_price: float,
    ) -> Dict[str, Any]:
        """Legacy non-durable submission is intentionally unavailable."""
        self.validate_paper_safety()
        raise AlpacaSafetyError("Legacy submission disabled; use managed paper-session")

    def submit_market_sell(
        self,
        ticker:   str,
        qty:      float,
    ) -> Dict[str, Any]:
        """Legacy non-durable submission is intentionally unavailable."""
        self.validate_paper_safety()
        raise AlpacaSafetyError("Legacy submission disabled; use managed paper-session")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _position_to_dict(pos: Any) -> Dict[str, Any]:
        return {
            "symbol":          str(pos.symbol),
            "qty":             str(pos.qty),
            "market_value":    str(pos.market_value),
            "avg_entry_price": str(pos.avg_entry_price),
            "current_price":   str(pos.current_price),
            "unrealized_pl":   str(pos.unrealized_pl),
            "unrealized_plpc": str(pos.unrealized_plpc),
            "side":            str(pos.side.value) if hasattr(pos.side, "value") else str(pos.side),
        }

    @staticmethod
    def _order_to_dict(order: Any) -> Dict[str, Any]:
        return {
            "id":           str(order.id),
            "status":       str(getattr(order.status, "value", order.status)),
            "client_order_id": str(getattr(order, "client_order_id", "")),
            "symbol":       str(order.symbol),
            "qty":          str(order.qty),
            "side":         str(order.side.value) if hasattr(order.side, "value") else str(order.side),
            "filled_qty":   str(getattr(order, "filled_qty", "0")),
            "filled_avg_price": str(getattr(order, "filled_avg_price", "") or ""),
            "filled_at": str(getattr(order, "filled_at", "") or ""),
            "updated_at": str(getattr(order, "updated_at", "") or ""),
            "type":         str(getattr(order, "order_type", "market")),
            "created_at":   str(getattr(order, "created_at", "")),
        }

    def verify_paper_account(self) -> Dict[str, Any]:
        """Authenticate against a pinned paper transport; never trust a boolean alone.

        Alpaca has no account `paper` field. Endpoint identity plus successful
        authentication is the proof. The opaque fingerprint binds durable state
        to this account without storing/logging its ID or credentials.
        """
        self.validate_paper_safety()
        client = self.get_trading_client()
        base = getattr(client, "_base_url", "")
        if str(getattr(base, "value", base)) != "https://paper-api.alpaca.markets":
            raise AlpacaSafetyError("Paper transport cannot be verified")
        try:
            account = client.get_account()
            status = str(getattr(account.status, "value", account.status)).upper()
            if (status != "ACTIVE" or not account.id
                    or account.trading_blocked is not False
                    or account.account_blocked is not False
                    or account.trade_suspended_by_user is not False):
                raise AlpacaSafetyError("Paper account is not eligible for trading")
            return {
                "paper_verified": True,
                "fingerprint": hashlib.sha256(str(account.id).encode()).hexdigest(),
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
            }
        except AlpacaSafetyError:
            raise
        except Exception:
            raise AlpacaConnectionError("Paper account verification failed") from None

    def get_order_by_client_id(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        try:
            order = self.get_trading_client().get_order_by_client_id(client_order_id)
            return self._order_to_dict(order)
        except Exception as exc:
            # Only an actual HTTP 404 establishes absence; timeouts/auth failures
            # are NOT absence and must leave submission state unresolved.
            if getattr(exc, "status_code", None) == 404:
                return None
            raise AlpacaConnectionError("Paper order lookup failed") from None

    def submit_managed_order(self, order: Dict[str, Any], *, before_submit=None) -> Dict[str, Any]:
        """v1.3 simple DAY order, with durable caller-supplied client ID."""
        from broker.paper_transport import validate_order
        validate_order(order)
        if self._s.EXECUTION_MODE != "paper" or self.dry_run:
            raise AlpacaSafetyError("Explicit paper execution activation required")
        self.verify_paper_account()
        if not self.is_market_open():
            raise AlpacaSafetyError("Market closed; paper submission blocked")
        client = self.get_trading_client()
        try:
            asset = client.get_asset(order["symbol"])
            if (asset.tradable is not True or str(getattr(asset.status, "value", asset.status)) != "active"
                    or str(getattr(asset.asset_class, "value", asset.asset_class)) != "us_equity"
                    or (float(order["qty"]) % 1 and asset.fractionable is not True)):
                raise AlpacaSafetyError("Asset is not eligible for this paper order")
        except AlpacaSafetyError:
            raise
        except Exception:
            raise AlpacaSafetyError("Asset eligibility unavailable") from None
        if order["side"] == "buy":
            try:
                deadline = datetime.fromisoformat(order["valid_until"])
                now = datetime.now(timezone.utc)
                if deadline.tzinfo is None or now >= deadline or now.astimezone(ZoneInfo("America/New_York")).hour >= 13:
                    raise ValueError
            except (KeyError, ValueError, TypeError):
                raise AlpacaSafetyError("Entry deadline/cutoff blocks submission") from None
        request_type = _LimitOrderRequest if order["type"] == "limit" else _MarketOrderRequest
        kwargs = dict(
            symbol=order["symbol"], qty=order["qty"], side=order["side"],
            time_in_force="day", extended_hours=False,
            client_order_id=order["client_order_id"],
        )
        if order["type"] == "limit":
            kwargs["limit_price"] = order["limit_price"]
        # Last network preflight after asset lookup. Validate freshness as well
        # as is_open, so a cached or skewed clock cannot queue an overnight order.
        try:
            clock = self.get_clock()
            now = datetime.now(timezone.utc)
            stamp = datetime.fromisoformat(clock["timestamp"])
            closing = datetime.fromisoformat(clock["next_close"])
            if (clock["is_open"] is not True or stamp.tzinfo is None or closing.tzinfo is None
                    or abs((stamp - now).total_seconds()) > 60 or closing <= now):
                raise ValueError
            if order["side"] == "buy" and (now >= deadline or now.astimezone(ZoneInfo("America/New_York")).hour >= 13):
                raise ValueError
            if before_submit is None or before_submit(order) is not True:
                raise ValueError
        except Exception:
            raise AlpacaSafetyError("Final paper submission preflight blocked") from None
        try:
            return self._order_to_dict(client.submit_order(
                order_data=request_type(**kwargs),
            ))
        except Exception:
            # Includes explicit API errors: reconcile instead of guessing whether
            # an order was accepted. Never expose raw broker response/exception.
            raise AlpacaConnectionError("Paper submission acknowledgement unavailable") from None
