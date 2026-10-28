"""Frozen intraday strategy -> managed paper lifecycle; shadow paths untouched."""
from dataclasses import replace
from datetime import timedelta
from concurrent.futures import Future
from threading import Thread
from time import monotonic, sleep
from uuid import uuid4

import pandas as pd

from broker.base import BrokerError
from broker.paper_execution import ET, TERMINAL, utc
from trades.shadow_session import ShadowEntryPolicy, SHADOW_EXIT_ACTIONS
from trades.trade_monitor import TradeMonitor


class CleanupRequested(Exception):
    """Supervisor requested orderly cleanup while a read-only scan was running."""


def supervised_scan(engine, scan, symbols, monitoring, wait, monotonic_clock):
    """The worker only reads market data/computes signals. All broker IO remains
    on the single owning thread. Slow/hung data cannot starve the kill switch.
    A daemon worker cannot prevent process shutdown; its result is discarded
    when cleanup is requested. Never start another scan after a scan timeout.
    """
    result = Future()

    def read_only_work():
        try:
            result.set_result(scan(symbols, monitoring))
        except BaseException as exc:
            result.set_exception(exc)

    Thread(target=read_only_work, name="paper-data-scan", daemon=True).start()
    deadline = monotonic_clock() + 300
    while not result.done():
        account, _ = engine.reconcile()
        clock = engine.clock()
        current = engine.now()
        if (engine.kill_path.exists() or engine.circuit_status(account).must_close_all
                or (utc(clock["next_close"]) - current).total_seconds() <= 60):
            raise CleanupRequested
        if current.astimezone(ET).hour >= 13 or engine.ledger.get("partial"):
            engine.cancel_entries()
        if monotonic_clock() >= deadline:
            engine.ledger.set("partial", True)
            engine.ledger.set("partial_reasons", sorted(set(engine.ledger.get("partial_reasons", []) + ["SCAN_TIMEOUT"])))
            engine.ledger.audit("SCAN_TIMEOUT")
            raise CleanupRequested
        wait(5)
    return result.result()


def entry_data_invalid(data, symbols, timeframe, now):
    """Paper IO guard: same stale threshold; unknown/missing timestamps fail closed.

    Recompute age at decision time (a long scan can age earlier symbol data).
    Alpaca internally represents UTC timestamps without tzinfo.
    """
    threshold = {"1Min": 180, "5Min": 900}[timeframe]
    for symbol in symbols:
        frame = data.get(symbol)
        if frame is None or frame.empty:
            return True
        try:
            stamp = pd.Timestamp(frame.attrs["latest_bar_time"])
            if pd.isna(stamp):
                return True
            stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
            age = (now - stamp.to_pydatetime()).total_seconds()
            if age < 0 or age > threshold:
                return True
        except (KeyError, ValueError, TypeError):
            return True
    return False


def approval_freshness(engine, refresh, symbols, timeframe, deadline, wait=sleep):
    """Bounded read-only fetch; no broker/SQLite access in the data worker.

    During the final gate an UNKNOWN intent already exists, so normal recovery
    must NOT run here. Refusal is definitively before POST. Kill/expiry interrupt
    the wait even when a market-data endpoint hangs.
    """
    if refresh is None:
        raise BrokerError("Fresh provider read required for approval revalidation")
    result = Future()

    def read():
        try:
            result.set_result(refresh(symbols))
        except BaseException as exc:
            result.set_exception(exc)

    Thread(target=read, name="paper-approval-data", daemon=True).start()
    started = monotonic()
    while not result.done():
        if (engine.kill_path.exists() or engine.now() >= deadline or monotonic() - started >= 60):
            raise BrokerError("Approval refresh interrupted or expired")
        wait(.25)
    return result.result(), symbols, timeframe


def run_paper_session(engine, scan, cooldowns, symbols, *, timeframe="5Min",
                      interval_seconds=300, max_iterations=72, wait=sleep,
                      monotonic_clock=monotonic, report=print, refresh_data=None):
    """Bounded supervisor. Poll broker/kill switch at most every five seconds.

    Crashed unfinished sessions resume the same durable identity, never adopt
    historical ENDED trades. Foreign account activity blocks, never gets sold.
    """
    ledger = engine.ledger
    policy, monitor = ShadowEntryPolicy(), TradeMonitor()
    config = {"symbols": symbols, "timeframe": timeframe, "interval": interval_seconds}
    status = ledger.get("status")
    recovering = status is not None and status != "ENDED"
    if ledger.orders() and status is None:
        raise BrokerError("Missing paper session state with historical orders")
    if recovering and ledger.get("config") != config:
        raise BrokerError("Recovery configuration differs; restore previous session configuration")
    if recovering and (type(ledger.get("partial")) is not bool
                       or not isinstance(ledger.get("partial_reasons"), list)
                       or not ledger.get("started_at") or not ledger.get("session_id")):
        raise BrokerError("Incomplete recoverable session state; manual review required")
    session_active = False
    try:
        account, _ = engine.reconcile()
        if status == "ENDED" and (engine.tracker.list_open_trades() or
                any(r["status"] not in TERMINAL for r in ledger.orders())):
            raise BrokerError("ENDED session contains unfinished orders/trades")
        engine.clock()
        engine.references(account["equity"])
        if not recovering:
            engine.approvals.invalidate(engine.now(), "NEW_SESSION")
            ledger.set("session_id", "paper-" + uuid4().hex)
            ledger.set("partial", False)
            ledger.set("partial_reasons", [])
            ledger.set("config", config)
            ledger.set("started_at", engine.now().isoformat())
            ledger.set("last_scan", None)
        ledger.set("status", "RUNNING")
        session_active = True
        ledger.audit("SESSION_STARTED", recovery=recovering, execute=engine.execute)
        session_id = ledger.get("session_id")
        if not session_id:
            raise BrokerError("Missing recoverable session identity")
        iteration, next_scan = 0, monotonic_clock()
        finish_only = recovering and (status in {"CLEANUP_PENDING", "ERROR"} or
                        utc(ledger.get("started_at")).astimezone(ET).date() != engine.now().astimezone(ET).date())

        def partial(reason):
            ledger.set("partial", True)
            reasons = ledger.get("partial_reasons", [])
            if reason not in reasons:
                ledger.set("partial_reasons", reasons + [reason])
            ledger.audit("PARTIAL_DATA", reason=reason)

        def eligible(setup, unsent):
            now_et = engine.now().astimezone(ET).replace(tzinfo=None)
            trades = engine.tracker.list_open_trades()
            active = {r["payload"]["order"]["symbol"] for r in ledger.orders()
                      if r["cid"] != unsent and r["status"] not in TERMINAL}
            decision = policy.evaluate(setup, now_et, trades, cooldowns(), False,
                                       engine.entries_today(exclude=unsent), now_et=now_et)
            return (decision.accepted and
                    len({t.ticker for t in trades} | active) < policy.config.max_open_trades)

        while not finish_only:
            account, _ = engine.reconcile()
            clock = engine.clock()
            current = engine.now()
            if engine.kill_path.exists():
                ledger.audit("KILL_SWITCH", action="CANCEL_ENTRIES_AND_FLATTEN_OWNED")
                break
            cb = engine.circuit_status(account)
            if cb.must_close_all:
                ledger.audit("RISK_CLOSE_ALL", rules=cb.triggered_rules)
                break
            if (utc(clock["next_close"]) - current).total_seconds() <= 60:
                break
            after_cutoff = current.astimezone(ET).hour >= 13
            expired_entry = any(r["payload"]["order"]["side"] == "buy" and
                                r["status"] not in TERMINAL and
                                (current - utc(r["payload"]["at"])).total_seconds() >= 300
                                for r in ledger.orders())
            if after_cutoff or expired_entry or ledger.get("partial") or not cb.new_positions_allowed:
                engine.cancel_entries()
            engine.approvals.expire(engine.now())
            for proposal in engine.approvals.rows():
                if proposal["status"] != "APPROVED":
                    continue
                deadline = utc(proposal["expires"])
                engine.submit_approved(proposal["cid"],
                    freshness=lambda: approval_freshness(engine, refresh_data, symbols, timeframe, deadline, wait),
                    eligibility=eligible)
            if monotonic_clock() < next_scan:
                wait(min(5, max(0, next_scan - monotonic_clock())))
                continue
            last_scan = ledger.get("last_scan")
            if last_scan and (current - utc(last_scan)).total_seconds() > max(900, interval_seconds * 2.5):
                partial("ITERATION_GAP")
            ledger.set("last_scan", current.isoformat())
            ledger.set("scan_attempts", ledger.get("scan_attempts", 0) + 1)
            iteration += 1
            monitor_symbols = [t.ticker for t in engine.tracker.list_open_trades()]
            try:
                setups, stale, data = supervised_scan(engine, scan, symbols, monitor_symbols, wait, monotonic_clock)
            except CleanupRequested:
                break
            except Exception:
                partial("PROVIDER_FAILURE")
                setups, stale, data = [], True, {}
            if stale or entry_data_invalid(data, symbols, timeframe, engine.now()):
                partial("STALE_OR_MISSING_PROVIDER_DATA")
                engine.cancel_entries()
            open_trades = engine.tracker.list_open_trades()
            current = engine.now()
            now_et = current.astimezone(ET).replace(tzinfo=None)
            for setup in setups:
                evaluated = replace(setup, timestamp=now_et)
                pending_symbols = {r["payload"]["order"]["symbol"] for r in ledger.orders()
                                   if r["status"] not in TERMINAL and r["payload"]["order"]["side"] == "buy"}
                decision = policy.evaluate(evaluated, now_et, open_trades, cooldowns(),
                                           bool(stale), engine.entries_today(), now_et=now_et)
                reasons = list(decision.skip_reasons)
                if len({t.ticker for t in open_trades} | pending_symbols) >= policy.config.max_open_trades:
                    reasons.append("MAX_OPEN_TRADES")
                if ledger.get("partial"):
                    reasons.append("PARTIAL_SESSION")
                ledger.audit("SETUP_DECISION", symbol=setup.ticker, reasons=sorted(set(reasons)))
                if decision.accepted and not reasons:
                    frame = data[setup.ticker]
                    kind = str(getattr(setup.setup_type, "value", setup.setup_type))
                    key = f"{now_et.date()}:{setup.ticker}:{kind}:{frame.index[-1]}"
                    deadlines = []
                    for symbol in symbols:
                        stamp = pd.Timestamp(data[symbol].attrs["latest_bar_time"])
                        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
                        deadlines.append(stamp.to_pydatetime() + timedelta(seconds={"1Min": 180, "5Min": 900}[timeframe]))
                    proposed = engine.enter(evaluated, key=key, session_id=session_id,
                                 stale=entry_data_invalid(data, symbols, timeframe, engine.now()),
                                 valid_until=min(current + timedelta(seconds=300), *deadlines))
                    if proposed:
                        proposal = engine.approvals.get(proposed)
                        report(f"PENDING HUMAN APPROVAL: {proposed}; {setup.ticker}; "
                               f"expires={proposal['expires']}. Use paper-proposals to inspect.")
                    engine.reconcile()
                    open_trades = engine.tracker.list_open_trades()
            # Same monitor and exit actions as shadow. Prices come from actual
            # paper fills, never shadow_exit_price or invented last-close fills.
            open_trades = engine.tracker.list_open_trades()
            for trade, result in zip(open_trades, monitor.monitor_open_trades(open_trades, data)):
                action = str(getattr(result.action, "value", result.action))
                if action in SHADOW_EXIT_ACTIONS:
                    engine.exit(trade.trade_id, action)
            report(f"Paper scan {iteration}: quality={'PARTIAL_DATA' if ledger.get('partial') else 'CLEAN'}; "
                   f"owned open trades={len(engine.tracker.list_open_trades())}; "
                   f"mode={'PAPER' if engine.execute else 'READ_ONLY_DRY_RUN'}")
            # Scheduled cleanup never waits for a human approval. Pending
            # proposals are invalidated by finish, including on a one-scan run.
            if iteration >= max_iterations:
                break
            next_scan = monotonic_clock() + interval_seconds

        for _ in range(12):
            if engine.finish():
                report("Paper session ENDED: broker confirmed flat; no pending owned orders.")
                return True
            wait(5)
        report("Paper session CLEANUP_PENDING: operator review/restart required.")
        return False
    except BaseException as exc:
        engine.approvals.invalidate(engine.now(), "SESSION_ERROR")
        if session_active:
            ledger.set("status", "ERROR")
        ledger.audit("SESSION_ERROR", action="RECOVER_BEFORE_NEW_ENTRIES", error_type=type(exc).__name__)
        if engine.execute:
            try:
                engine.cancel_entries()
            except Exception:
                ledger.audit("CLEANUP_UNCONFIRMED")
        raise
