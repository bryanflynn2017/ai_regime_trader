"""Durable v1.3 paper lifecycle. Strategy/monitor remain outside this module.

SQLite is authoritative; the existing TradeTracker journal is a rebuildable
projection of broker-confirmed cumulative fills. A committed submission intent
ALWAYS precedes network IO. Uncertain orders are looked up, never resubmitted.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_DOWN
from dataclasses import asdict
from types import SimpleNamespace
import hashlib
import json
from math import isfinite
from pathlib import Path
import os
import sqlite3
from zoneinfo import ZoneInfo

from broker.base import BrokerError, OrderNotSubmitted
from broker.paper_transport import validate_order
from broker.paper_approvals import Approvals, initialize, digest
from risk.risk_manager import RiskManager
from strategy.signals import Signal, SignalAction, RegimeLabel
from trades.trade_tracker import ManualTrade, TradeTracker

ET = ZoneInfo("America/New_York")
TERMINAL = {"filled", "canceled", "expired", "rejected"}
ACTIVE = {"new", "accepted", "pending_new", "partially_filled", "pending_cancel",
          "accepted_for_bidding", "done_for_day", "stopped", "calculated"}


def number(value, *, positive=False):
    try:
        result = float(value)
        if not isfinite(result) or result < 0 or (positive and result == 0):
            raise ValueError
        return result
    except (ValueError, TypeError, OverflowError):
        raise BrokerError("Invalid numeric broker/state value") from None


def utc(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise BrokerError("Broker clock must be timezone-aware")
    return parsed.astimezone(timezone.utc)


class PaperLedger(AbstractContextManager):
    """Single-process ownership held by OS, released even on crash; never aged out.

    A lock sidecar remains on disk intentionally. Its existence is not a lease.
    Do not unlink it (that could allow two processes to lock different inodes).
    """
    def __init__(self, root="state/paper"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = open(self.root / "owner.lock", "a+b")
        self.db = None
        try:
            self.lock.seek(0, 2)
            if not self.lock.tell():
                self.lock.write(b"0")
                self.lock.flush()
            self.lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.db = sqlite3.connect(self.root / "ledger.sqlite3", timeout=5)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS orders (
                    cid TEXT PRIMARY KEY, payload TEXT NOT NULL,
                    status TEXT NOT NULL, broker_id TEXT NOT NULL DEFAULT '',
                    filled REAL NOT NULL DEFAULT 0, average REAL NOT NULL DEFAULT 0,
                    updated TEXT NOT NULL DEFAULT '');
                CREATE TABLE IF NOT EXISTS audit (
                    sequence INTEGER PRIMARY KEY, at TEXT NOT NULL,
                    event TEXT NOT NULL, details TEXT NOT NULL);
            """)
            initialize(self.db)
            if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise BrokerError("Paper ledger integrity check failed")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        if self.db is not None:
            self.db.close()
            self.db = None
        self.lock.close()  # OS releases ownership

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))

    def audit(self, event, **details):
        # Callers pass only locally defined fields; never broker dicts/exceptions.
        with self.db:
            self.db.execute("INSERT INTO audit(at,event,details) VALUES (?,?,?)", (
                datetime.now(timezone.utc).isoformat(), event, json.dumps(details, allow_nan=False),
            ))

    def orders(self):
        return [dict(row, payload=json.loads(row["payload"]))
                for row in self.db.execute("SELECT * FROM orders ORDER BY rowid")]

    def intent(self, cid, payload):
        with self.db:
            self.db.execute("INSERT INTO orders(cid,payload,status) VALUES (?,?,?)",
                            (cid, json.dumps(payload, allow_nan=False), "UNKNOWN"))
            self.db.execute("INSERT INTO audit(at,event,details) VALUES (?,?,?)", (
                payload["at"], "SUBMISSION_INTENT", json.dumps({"cid": cid, "side": payload["order"]["side"]}),
            ))

    def observe(self, row, observed):
        expected = row["payload"]["order"]
        try:
            status = str(observed["status"])
            filled = number(observed["filled_qty"])
            average = number(observed.get("filled_avg_price"), positive=True) if filled else 0.0
            if (observed["client_order_id"] != row["cid"]
                    or observed["symbol"] != expected["symbol"]
                    or observed["side"] != expected["side"]
                    or number(observed["qty"]) != expected["qty"]
                    or not observed["id"]
                    or (row["broker_id"] and row["broker_id"] != observed["id"])
                    or status not in ACTIVE | TERMINAL
                    or (row["status"] in TERMINAL and status != row["status"])
                    or filled < row["filled"] or filled > expected["qty"]
                    or (status == "filled" and filled != expected["qty"])):
                raise BrokerError("Order acknowledgement/reconciliation conflict")
            if filled and expected["side"] == "buy" and average > expected["limit_price"]:
                raise BrokerError("Entry fill exceeds submitted limit; reconciliation blocked")
            updated = observed.get("filled_at") or observed.get("updated_at") or row["payload"]["at"]
            utc(updated)
        except (KeyError, TypeError, ValueError):
            raise BrokerError("Incomplete order acknowledgement") from None
        with self.db:
            self.db.execute("UPDATE orders SET status=?,broker_id=?,filled=?,average=?,updated=? WHERE cid=?",
                            (status, observed["id"], filled, average, updated, row["cid"]))
            self.db.execute("INSERT INTO audit(at,event,details) VALUES (?,?,?)", (
                datetime.now(timezone.utc).isoformat(), "ORDER_OBSERVED",
                json.dumps({"cid": row["cid"], "status": status, "filled": filled, "average": average}),
            ))

    def reject_unsent(self, cid):
        """Only used for an explicit adapter guarantee that POST never started."""
        with self.db:
            self.db.execute("UPDATE orders SET status='rejected' WHERE cid=? AND status='UNKNOWN' AND filled=0",
                            (cid,))
            self.db.execute("INSERT INTO audit(at,event,details) VALUES (?,?,?)", (
                datetime.now(timezone.utc).isoformat(), "LOCAL_PREFLIGHT_REJECTED", json.dumps({"cid": cid}),
            ))


class PaperExecution:
    def __init__(self, broker, ledger, settings, *, execute=False, now=None):
        self.broker, self.ledger, self.settings = broker, ledger, settings
        self.execute = execute
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.risk = RiskManager(settings_override=settings)
        self.tracker = TradeTracker(ledger.root / "paper_trades.jsonl")
        self.kill_path = ledger.root / "KILL_SWITCH"
        self.approvals = Approvals(ledger.db)
        self._approval_context = None
        if execute and settings.EXECUTION_MODE != "paper":
            raise BrokerError("EXECUTION_MODE=paper is required for paper submission")
        if settings.LIVE_TRADING_ALLOWED or not settings.PAPER_TRADING or not settings.ALPACA_PAPER:
            raise BrokerError("Paper-only safety configuration required")
        if hasattr(broker, "set_submission_guard"):
            broker.set_submission_guard(self.submission_guard)

    def submission_guard(self, order):
        """Last local check immediately before HTTP POST, including durable intent."""
        owned = any(r["cid"] == order["client_order_id"] and r["payload"]["order"] == order
                    and r["status"] == "UNKNOWN" for r in self.ledger.orders())
        if not self.execute or not owned:
            return False
        if order["side"] == "sell":
            return True
        basic = (not self.kill_path.exists() and not self.ledger.get("partial", False)
                and not self.settings.is_risk_locked()
                and self.now().astimezone(ET).hour < 13
                and self.now() < utc(order["valid_until"]))
        if not basic or not self._approval_context:
            return False
        cid, freshness, eligibility = self._approval_context
        row = self.approvals.get(cid)
        if (not row or row["status"] != "CONSUMED" or row["payload"]["order"] != order
                or digest(row["payload"]) != row["digest"]):
            return False
        return self._revalidate(row, freshness, eligibility, ignore_intent=cid)

    def verify(self):
        if self.broker.is_paper is not True:
            raise BrokerError("Paper broker not verified")
        account = self.broker.verify_paper_account()
        if account.get("paper_verified") is not True or not account.get("fingerprint"):
            raise BrokerError("Paper account not verified")
        previous = self.ledger.get("account_fingerprint")
        if previous and previous != account["fingerprint"]:
            self.ledger.audit("SAFETY_BLOCK", reason="ACCOUNT_CHANGED")
            raise BrokerError("Account differs from durable paper state")
        self.ledger.set("account_fingerprint", account["fingerprint"])
        for key in ("equity", "cash", "buying_power"):
            account[key] = number(account[key], positive=(key == "equity"))
        return account

    def clock(self):
        clock = self.broker.get_clock()
        current = self.now()
        if abs((utc(clock["timestamp"]) - current).total_seconds()) > 60:
            self.ledger.audit("SAFETY_BLOCK", reason="CLOCK_SKEW_OR_STALE")
            raise BrokerError("Broker clock stale or local clock skewed")
        if clock.get("is_open") is not True:
            self.ledger.audit("SAFETY_BLOCK", reason="MARKET_CLOSED")
            raise BrokerError("Market is closed")
        if utc(clock["next_close"]) <= current:
            raise BrokerError("Invalid market-close timestamp")
        return clock

    def reconcile(self, *, _unsent_intent=None):
        account = self.verify()
        for row in self.ledger.orders():
            if row["cid"] == _unsent_intent:
                if row["status"] != "UNKNOWN" or self.broker.get_order_by_client_id(row["cid"]) is not None:
                    raise BrokerError("Unexpected broker order for unsent approved intent")
                continue
            if row["status"] in TERMINAL:
                continue
            observed = self.broker.get_order_by_client_id(row["cid"])
            if observed is None:
                self.ledger.audit("SAFETY_BLOCK", reason="UNRESOLVED_ORDER_INTENT", cid=row["cid"])
                raise BrokerError("Unresolved order intent; manual review required, never auto-resubmitted")
            self.ledger.observe(row, observed)
        orders = self.ledger.orders()
        known = {row["cid"]: row for row in orders}
        open_orders = self.broker.list_open_orders()
        if any(order.get("client_order_id") not in known for order in open_orders):
            self.ledger.audit("SAFETY_BLOCK", reason="UNOWNED_BROKER_ORDER")
            raise BrokerError("Unowned broker order; reconciliation blocked")
        for order in open_orders:
            if order.get("client_order_id") == _unsent_intent:
                raise BrokerError("Approved intent already exists at broker")
            self.ledger.observe(known[order["client_order_id"]], order)
        orders = self.ledger.orders()
        positions = self.broker.list_positions()
        expected = {}
        for row in orders:
            order = row["payload"]["order"]
            sign = 1 if order["side"] == "buy" else -1
            expected[order["symbol"]] = expected.get(order["symbol"], 0) + sign * row["filled"]
        actual = {}
        for pos in positions:
            if pos["symbol"] in actual or str(pos.get("side")) != "long":
                raise BrokerError("Duplicate or non-long broker position")
            actual[pos["symbol"]] = number(pos["qty"])
            number(pos["market_value"])
        if any(abs(actual.get(s, 0) - expected.get(s, 0)) > 0.000001 for s in actual.keys() | expected.keys()):
            self.ledger.audit("SAFETY_BLOCK", reason="POSITION_FILL_MISMATCH")
            raise BrokerError("Broker position/fill mismatch; no new orders")
        self.project_journal()
        self.ledger.audit("RECONCILED", positions=len(positions), open_orders=len(open_orders))
        return account, positions

    def project_journal(self):
        orders = self.ledger.orders()
        trades = []
        for row in orders:
            payload = row["payload"]
            if payload["order"]["side"] != "buy" or not row["filled"]:
                continue
            exits = [r for r in orders if r["payload"].get("entry_cid") == row["cid"]]
            sold_exact = sum((Decimal(str(r["filled"])) for r in exits), Decimal(0))
            entry_exact = Decimal(str(row["filled"]))
            sold = float(sold_exact)
            if sold_exact > entry_exact:
                raise BrokerError("Exit fills exceed entry fills")
            remaining = float(entry_exact - sold_exact)
            trade = ManualTrade(
                **payload["trade"], trade_id=row["cid"], entry_price=row["average"],
                shares=remaining or row["filled"], entry_time=payload["at"],
                test_type="alpaca_paper", exclude_from_analytics=True,
                exclusion_reason="Broker paper fills; not clean shadow strategy proof",
                metadata={"paper": True, "entry_filled_qty": row["filled"],
                          "exit_filled_qty": sold, "entry_order_status": row["status"],
                          "partial_realized_pnl": sum(r["filled"] * r["average"] for r in exits) - sold * row["average"],
                          "session_id": payload["session_id"]},
            )
            if not remaining:
                trade.status = "CLOSED"
                trade.exit_price = sum(r["filled"] * r["average"] for r in exits) / sold
                trade.exit_time = max(r["updated"] for r in exits if r["filled"])
                trade.realized_pnl = (trade.exit_price - row["average"]) * sold
            trades.append(trade)
        self.tracker._write_all(trades)  # existing atomic journal writer; projection only

    def references(self, equity):
        now = self.now().astimezone(ET)
        day, week = now.date().isoformat(), f"{now.isocalendar().year}-{now.isocalendar().week}"
        refs = self.ledger.get("risk_references", {})
        if not refs and self.ledger.orders():
            raise BrokerError("Missing risk references for existing order history")
        if refs:
            for name in ("day_equity", "week_equity", "peak_equity"):
                number(refs[name], positive=True)
            if not refs.get("day") or not refs.get("week"):
                raise BrokerError("Incomplete risk reference dates")
        if refs.get("day") != day:
            refs.update(day=day, day_equity=equity)
        if refs.get("week") != week:
            refs.update(week=week, week_equity=equity)
        refs["peak_equity"] = max(equity, refs.get("peak_equity", equity))
        self.ledger.set("risk_references", refs)
        return refs

    def entries_today(self, *, exclude=None):
        day = self.now().astimezone(ET).date()
        return sum(r["payload"]["order"]["side"] == "buy" and
                   utc(r["payload"]["at"]).astimezone(ET).date() == day
                   for r in self.ledger.orders() if r["cid"] != exclude)

    def circuit_status(self, account):
        refs = self.references(account["equity"])
        return self.risk._cb.evaluate(account["equity"], refs["day_equity"],
                                      refs["week_equity"], refs["peak_equity"])

    def _submit(self, payload, key, *, approved_cid=None):
        cid = approved_cid or ("art13-" + hashlib.sha256(key.encode()).hexdigest()[:32])
        if any(r["cid"] == cid for r in self.ledger.orders()):
            self.ledger.audit("DUPLICATE_BLOCKED", cid=cid)
            return None
        payload["order"]["client_order_id"] = cid
        validate_order(payload["order"])
        if not self.execute:
            self.ledger.audit("DRY_RUN_ORDER", **payload["order"])
            return None
        self.verify()
        self.clock()
        if payload["order"]["side"] == "buy":
            if (self.kill_path.exists() or self.ledger.get("partial", False)
                    or self.now().astimezone(ET).hour >= 13
                    or self.now() >= utc(payload["order"]["valid_until"])):
                self.ledger.audit("ENTRY_BLOCKED", reason="FINAL_PREFLIGHT_EXPIRED_OR_HALTED")
                return None
        if payload["order"]["side"] == "buy":
            if not approved_cid or not self._approval_context:
                raise BrokerError("Human approval required for every paper entry")
            self.approvals.consume(cid, payload, self.now())
        else:
            self.ledger.intent(cid, payload)
        try:
            # Alpaca invokes the gate after its network preflight, just before
            # POST. Adapters without that capability are gated here instead.
            if not hasattr(self.broker, "set_submission_guard"):
                try:
                    allowed = self.submission_guard(payload["order"])
                except Exception:
                    raise OrderNotSubmitted("Final submission revalidation failed") from None
                if not allowed:
                    raise OrderNotSubmitted("Final submission gate refused order")
            response = self.broker.submit_order(payload["order"])
            row = next(r for r in self.ledger.orders() if r["cid"] == cid)
            self.ledger.observe(row, response)
        except OrderNotSubmitted:
            self.ledger.reject_unsent(cid)
            return None
        except Exception:
            self.ledger.audit("SUBMISSION_UNCERTAIN", cid=cid)
            raise BrokerError("Submission uncertain; reconcile before further action") from None
        return cid

    def _plan_entry(self, setup, *, session_id, stale=False, valid_until=None, ignore_intent=None):
        account, positions = self.reconcile(_unsent_intent=ignore_intent)
        self.clock()
        if (self.kill_path.exists() or self.ledger.get("partial", False) or stale
                or self.now().astimezone(ET).hour >= 13):
            self.ledger.audit("ENTRY_BLOCKED", reason="KILL_PARTIAL_STALE_OR_CUTOFF")
            return None
        orders = [r for r in self.ledger.orders() if r["cid"] != ignore_intent]
        active = [r for r in orders if r["status"] not in TERMINAL]
        symbol = setup.ticker.upper()
        if any(p["symbol"] == symbol for p in positions) or any(r["payload"]["order"]["symbol"] == symbol for r in active):
            self.ledger.audit("ENTRY_BLOCKED", reason="POSITION_OR_PENDING_ORDER")
            return None
        entry, stop, target = (number(v, positive=True) for v in (setup.entry_trigger, setup.stop_loss, setup.target_1))
        if not stop < entry < target:
            raise BrokerError("Invalid frozen setup price levels")
        # Limit is rounded DOWN to exchange precision; never pay above trigger.
        limit = float(Decimal(str(entry)).quantize(Decimal(".01" if entry >= 1 else ".0001"), rounding=ROUND_DOWN))
        pending = sum((r["payload"]["order"]["qty"] - r["filled"]) * r["payload"]["order"]["limit_price"]
                      for r in active if r["payload"]["order"]["side"] == "buy")
        exposure = sum(number(p["market_value"]) for p in positions) + pending
        refs = self.references(account["equity"])
        signal = Signal(symbol, self.now(), SignalAction.BUY, setup.confidence, "Frozen intraday setup",
                        RegimeLabel.UNKNOWN, entry, stop, target, 0)
        decision = self.risk.evaluate_signal(
            signal, account["equity"], refs["day_equity"], refs["week_equity"], refs["peak_equity"],
            current_portfolio_exposure=exposure,
            open_positions_count=len({p["symbol"] for p in positions} | {r["payload"]["order"]["symbol"] for r in active}),
            trades_today=self.entries_today(exclude=ignore_intent),
        )
        if not decision.approved or not decision.position_size:
            self.ledger.audit("ENTRY_BLOCKED", reason="RISK_MANAGER")
            return None
        # Preserve the shadow workflow's $1000 notional ceiling, also bounded by
        # existing risk sizing/buying power. No leverage or new risk thresholds.
        qty = min(number(decision.position_size.shares), 1000 / entry,
                  max(0, min(account["cash"], account["buying_power"]) - pending) / entry)
        precision = Decimal(".01") if self.settings.ALLOW_FRACTIONAL_SHARES else Decimal("1")
        qty = float(Decimal(str(qty)).quantize(precision, rounding=ROUND_DOWN))
        if qty <= 0:
            return None
        trade = dict(ticker=symbol, setup_type=str(getattr(setup.setup_type, "value", setup.setup_type)),
                     stop_loss=stop, target_1=target, target_2=number(setup.target_2, positive=True),
                     setup_rank=(setup.metadata or {}).get("rank"), setup_confidence=setup.confidence)
        deadline = valid_until or (self.now() + timedelta(seconds=300))
        return dict(order=dict(symbol=symbol, qty=qty, side="buy", type="limit", limit_price=limit,
                                           valid_until=deadline.isoformat()),
                    trade=trade, at=self.now().isoformat(), session_id=session_id)

    def enter(self, setup, *, key, session_id, stale=False, valid_until=None):
        """Propose only. Even explicit paper activation is NOT entry approval."""
        deadline = min(valid_until or self.now() + timedelta(seconds=300),
                       self.now() + timedelta(seconds=60),
                       self.now().astimezone(ET).replace(hour=13, minute=0, second=0, microsecond=0))
        if self.now() >= deadline:
            return None
        payload = self._plan_entry(setup, session_id=session_id, stale=stale, valid_until=deadline)
        if not payload or self.now() >= deadline:
            return None
        cid = "art13-" + hashlib.sha256(("entry:" + key).encode()).hexdigest()[:32]
        if any(r["cid"] == cid for r in self.ledger.orders()):
            return None
        payload["order"]["client_order_id"] = cid
        validate_order(payload["order"])
        if not self.execute:
            self.ledger.audit("DRY_RUN_ORDER", **payload["order"])
            return None
        if self.ledger.get("session_id") != session_id or self.ledger.get("status") != "RUNNING":
            raise BrokerError("An active paper session is required for proposals")
        frozen = asdict(setup)
        frozen["timestamp"] = setup.timestamp.isoformat()
        frozen["setup_type"] = str(getattr(setup.setup_type, "value", setup.setup_type))
        payload.update(setup=frozen, account_fingerprint=self.ledger.get("account_fingerprint"))
        return self.approvals.create(cid, payload, self.now())

    def _revalidate(self, row, freshness, eligibility, *, ignore_intent=None):
        """Fresh reads, unchanged risk sizing, exact-order comparison. No resizing."""
        from trades.paper_session import entry_data_invalid
        payload = row["payload"]
        if (self.ledger.get("status") != "RUNNING" or self.ledger.get("partial") is not False
                or self.ledger.get("session_id") != payload["session_id"]
                or self.ledger.get("account_fingerprint") != payload["account_fingerprint"]
                or self.now() >= utc(row["expires"]) or digest(payload) != row["digest"]):
            return False
        data, symbols, timeframe = freshness()
        config = self.ledger.get("config", {})
        if (not symbols or symbols != config.get("symbols") or timeframe != config.get("timeframe")
                or payload["order"]["symbol"] not in symbols):
            return False
        if entry_data_invalid(data, symbols, timeframe, self.now()):
            return False
        frozen = dict(payload["setup"])
        frozen["timestamp"] = datetime.fromisoformat(frozen["timestamp"])
        setup = SimpleNamespace(**frozen)
        planned = self._plan_entry(setup, session_id=payload["session_id"],
                                   valid_until=utc(row["expires"]), ignore_intent=ignore_intent)
        if not planned or not eligibility(setup, ignore_intent):
            return False
        clock = self.clock()
        if (utc(clock["next_close"]) - self.now()).total_seconds() <= 60:
            return False
        planned["order"]["client_order_id"] = row["cid"]
        return (planned["order"] == payload["order"] and planned["trade"] == payload["trade"]
                and not entry_data_invalid(data, symbols, timeframe, self.now())
                and not self.kill_path.exists() and not self.settings.is_risk_locked()
                and not self.ledger.get("partial") and self.now().astimezone(ET).hour < 13
                and self.now() < utc(row["expires"]))

    def submit_approved(self, cid, *, freshness, eligibility):
        self.approvals.expire(self.now())
        row = self.approvals.get(cid)
        if not self.execute or not row or row["status"] != "APPROVED":
            self.ledger.audit("APPROVAL_REQUIRED", cid=cid)
            return None
        try:
            if (row["payload"]["session_id"] != self.ledger.get("session_id")
                    or self.ledger.get("status") != "RUNNING" or self.ledger.get("partial") is not False
                    or digest(row["payload"]) != row["digest"]):
                self.approvals.invalidate(self.now(), "REVALIDATION_FAILED", cid)
                return None
            self._approval_context = (cid, freshness, eligibility)
            result = self._submit(row["payload"], "", approved_cid=cid)
            if result is None:
                self.approvals.invalidate(self.now(), "SUBMISSION_REFUSED", cid)
            return result
        except Exception:
            self.approvals.invalidate(self.now(), "REVALIDATION_OR_SUBMISSION_ERROR", cid)
            raise
        finally:
            self._approval_context = None

    def cancel_entries(self):
        self.approvals.invalidate(self.now(), "ENTRIES_HALTED")
        self.verify()
        for row in self.ledger.orders():
            if row["payload"]["order"]["side"] == "buy" and row["status"] not in TERMINAL:
                if not row["broker_id"]:
                    raise BrokerError("Cannot cancel unacknowledged entry; reconcile first")
                if self.execute:
                    self.ledger.audit("CANCEL_REQUESTED", cid=row["cid"])
                    if not self.broker.cancel_order(row["broker_id"]):
                        raise BrokerError("Cancellation acknowledgement unavailable")
                else:
                    self.ledger.audit("DRY_RUN_CANCEL", cid=row["cid"])

    def exit(self, trade_id, reason):
        self.reconcile()
        self.clock()
        rows = self.ledger.orders()
        entry = next(r for r in rows if r["cid"] == trade_id)
        if entry["status"] not in TERMINAL:
            # Cancel residual entry BEFORE selling a partial fill. A cancel
            # request is not confirmation; the next reconciliation must settle.
            self.cancel_entries()
            return None
        exits = [r for r in rows if r["payload"].get("entry_cid") == trade_id]
        if any(r["status"] not in TERMINAL for r in exits):
            return None
        sold = sum((Decimal(str(r["filled"])) for r in exits), Decimal(0))
        qty = float(Decimal(str(entry["filled"])) - sold)
        if qty <= 0:
            return None
        # Same cumulative-fill key prevents endless retries of rejected exits.
        return self._submit(dict(
            order=dict(symbol=entry["payload"]["order"]["symbol"], qty=qty, side="sell", type="market"),
            entry_cid=trade_id, at=self.now().isoformat(), reason=reason,
            session_id=entry["payload"]["session_id"],
        ), f"exit:{trade_id}:{sold:.8f}")

    def finish(self, *, flatten=True):
        self.approvals.invalidate(self.now(), "SESSION_CLEANUP")
        self.ledger.set("status", "CLEANUP_PENDING")
        self.reconcile()
        self.cancel_entries()
        self.reconcile()
        if flatten:
            for trade in self.tracker.list_open_trades():
                self.exit(trade.trade_id, "SESSION_END")
        _, remaining_positions = self.reconcile()
        pending = any(r["status"] not in TERMINAL for r in self.ledger.orders())
        complete = not pending and not self.tracker.list_open_trades() and not remaining_positions
        self.ledger.set("status", "ENDED" if complete else "CLEANUP_PENDING")
        self.ledger.audit("SESSION_END", complete=complete)
        return complete
