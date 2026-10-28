"""v1.4 consent fault tests: temporary ledgers and fake brokers only."""
from datetime import datetime, timedelta
import json
from threading import Event
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from broker.base import BrokerError, OrderNotSubmitted
from broker.paper_approvals import operator_store
from broker.paper_execution import PaperExecution, PaperLedger
from tests.test_paper_execution import (env, forbid_network, setup, bars, recovery_state,
                                        FakeBroker, Settings)
from trades.paper_session import run_paper_session, approval_freshness


def propose(e, b, l, *, key="one", approve=False):
    if l.get("status") is None:
        recovery_state(e, l, b)
    cid = e.enter(setup(), key=key, session_id=l.get("session_id"))
    if approve:
        e.approvals.decide(cid, e.approvals.get(cid)["digest"], approve=True, now=b.now)
    return cid


def submit(e, cid, *, age=0, symbols=None, eligible=True):
    return e.submit_approved(cid,
        freshness=lambda: ({s: bars(e.now(), age=age) for s in (symbols or ["SPY"])},
                           symbols or ["SPY"], "5Min"), eligibility=lambda *_: eligible)


def test_activation_never_implies_consent(env):
    e, b, l = env
    cid = propose(e, b, l)
    assert e.approvals.get(cid)["status"] == "PENDING"
    assert not b.submissions and not l.orders()
    assert submit(e, cid) is None
    assert not b.submissions and not l.orders()


def test_exact_approved_order_submitted_once_and_audited(env):
    e, b, l = env
    cid = propose(e, b, l, approve=True)
    proposed = e.approvals.get(cid)["payload"]["order"]
    assert submit(e, cid) == cid
    assert b.submissions == [proposed]
    assert submit(e, cid) is None
    events = [r[0] for r in l.db.execute("SELECT event FROM audit ORDER BY sequence")]
    for first, second in zip(["ENTRY_PROPOSED", "HUMAN_APPROVED", "APPROVAL_CONSUMED", "SUBMISSION_INTENT"],
                              ["HUMAN_APPROVED", "APPROVAL_CONSUMED", "SUBMISSION_INTENT", "ORDER_OBSERVED"]):
        assert events.index(first) < events.index(second)


@pytest.mark.parametrize("approve", [True, False])
def test_operator_connection_does_not_need_owner_lock(env, approve):
    e, b, l = env
    cid = propose(e, b, l)
    with operator_store(l.root) as store:
        store.decide(cid, store.get(cid)["digest"], approve=approve, now=b.now)
    assert e.approvals.get(cid)["status"] == ("APPROVED" if approve else "REJECTED")
    assert not b.submissions


@pytest.mark.parametrize("seconds", [60, 61, 3600])
def test_expired_proposal_cannot_be_approved(env, seconds):
    e, b, l = env
    cid = propose(e, b, l)
    b.now += timedelta(seconds=seconds)
    with pytest.raises(BrokerError):
        e.approvals.decide(cid, e.approvals.get(cid)["digest"], approve=True, now=b.now)
    assert e.approvals.get(cid)["status"] == "EXPIRED"
    assert not b.submissions


def test_approval_does_not_extend_deadline(env):
    e, b, l = env
    cid = propose(e, b, l)
    original = e.approvals.get(cid)["expires"]
    b.now += timedelta(seconds=59)
    e.approvals.decide(cid, e.approvals.get(cid)["digest"], approve=True, now=b.now)
    assert e.approvals.get(cid)["expires"] == original
    b.now += timedelta(seconds=1)
    assert submit(e, cid) is None
    assert not b.submissions


def test_approval_expiring_while_waiting_for_database_lock_is_refused(env):
    e, b, l = env
    cid = propose(e, b, l)
    with patch("broker.paper_approvals.datetime", wraps=datetime) as dt:
        dt.now.side_effect = [b.now + timedelta(seconds=59), b.now + timedelta(seconds=60)]
        with pytest.raises(BrokerError):
            e.approvals.decide(cid, e.approvals.get(cid)["digest"], approve=True)
    assert e.approvals.get(cid)["status"] != "APPROVED" and not b.submissions


def test_data_deadline_and_cutoff_cap_approval_ttl(env):
    e, b, l = env
    recovery_state(e, l, b)
    cid = e.enter(setup(), key="data", session_id=l.get("session_id"), valid_until=b.now + timedelta(seconds=7))
    assert e.approvals.get(cid)["expires"] == (b.now + timedelta(seconds=7)).isoformat()
    b.now = b.now.replace(hour=16, minute=59, second=50)
    cid = e.enter(setup(), key="cutoff", session_id=l.get("session_id"))
    from broker.paper_execution import utc
    assert utc(e.approvals.get(cid)["expires"]) == b.now + timedelta(seconds=10)


@pytest.mark.parametrize("action", ["reject", "wrong_digest", "second_approval", "duplicate_proposal"])
def test_rejection_identity_and_duplicate_rules(env, action):
    e, b, l = env
    cid = propose(e, b, l)
    row = e.approvals.get(cid)
    if action == "wrong_digest":
        with pytest.raises(BrokerError):
            e.approvals.decide(cid, "0" * 64, approve=True, now=b.now)
    elif action == "reject":
        e.approvals.decide(cid, row["digest"], approve=False, now=b.now)
        with pytest.raises(BrokerError):
            e.approvals.decide(cid, row["digest"], approve=True, now=b.now)
    elif action == "second_approval":
        e.approvals.decide(cid, row["digest"], approve=True, now=b.now)
        with pytest.raises(BrokerError):
            e.approvals.decide(cid, row["digest"], approve=True, now=b.now)
    else:
        b.now += timedelta(seconds=10)
        assert propose(e, b, l) is None
        assert e.approvals.get(cid) == row
    assert not b.submissions


@pytest.mark.parametrize("field,value", [("symbol", "QQQ"), ("qty", 11), ("limit_price", 101),
                                        ("side", "sell"), ("valid_until", "2099-01-01T00:00:00+00:00")])
def test_changed_order_cannot_reuse_approval(env, field, value):
    e, b, l = env
    cid = propose(e, b, l, approve=True)
    payload = e.approvals.get(cid)["payload"]
    payload["order"][field] = value
    with l.db:
        l.db.execute("UPDATE proposals SET payload=? WHERE cid=?", (json.dumps(payload), cid))
    assert submit(e, cid) is None
    assert not b.submissions


@pytest.mark.parametrize("failure", ["cash", "risk", "cutoff", "partial", "kill", "account", "closed",
                                    "foreign_order", "foreign_position", "eligibility", "stale", "wrong_symbols"])
def test_revalidate_after_approval_fails_closed(env, failure):
    e, b, l = env
    cid = propose(e, b, l, approve=True)
    if failure == "cash": b.cash = 50
    if failure == "risk": b.equity = 80000
    if failure == "cutoff": b.now = b.now.replace(hour=17)
    if failure == "partial": l.set("partial", True)
    if failure == "kill": e.kill_path.touch()
    if failure == "account": b.verified = False
    if failure == "closed": b.market_open = False
    if failure == "foreign_order":
        b.orders["foreign"] = dict(client_order_id="foreign", status="new", filled_qty=0, symbol="QQQ", side="buy")
    if failure == "foreign_position":
        b.extra_positions = [dict(symbol="QQQ", qty=1, market_value=100, side="long")]
    try:
        result = submit(e, cid, age=901 if failure == "stale" else 0,
                        symbols=["QQQ"] if failure == "wrong_symbols" else None,
                        eligible=failure != "eligibility")
        assert result is None
    except BrokerError:
        pass
    assert not b.submissions


def test_missing_data_is_not_fresh(env):
    e, b, l = env
    cid = propose(e, b, l, approve=True)
    assert e.submit_approved(cid, freshness=lambda: ({}, ["SPY"], "5Min"), eligibility=lambda *_: True) is None
    assert not b.submissions and l.orders()[0]["status"] == "rejected"


def test_freshness_must_cover_the_exact_proposed_symbol(env):
    e, b, l = env
    recovery_state(e, l, b)
    cid = e.enter(setup("QQQ"), key="outside", session_id=l.get("session_id"))
    e.approvals.decide(cid, e.approvals.get(cid)["digest"], approve=True, now=b.now)
    assert submit(e, cid) is None  # Fresh SPY is not evidence about approved QQQ.
    assert not b.submissions


def test_final_gate_rereads_buying_power_after_adapter_preflight(env):
    e, b, l = env
    gate = {}
    b.set_submission_guard = lambda fn: gate.update(fn=fn)
    e = PaperExecution(b, l, e.settings, execute=True, now=lambda: b.now)
    cid = propose(e, b, l, approve=True)
    original_submit = b.submit_order
    def preflight(order):
        b.cash = 10  # Changed AFTER intent, just before final adapter gate.
        if not gate["fn"](order):
            raise OrderNotSubmitted("refused")
        return original_submit(order)
    b.submit_order = preflight
    assert submit(e, cid) is None
    assert not b.submissions and l.orders()[0]["status"] == "rejected"


@pytest.mark.parametrize("failure", ["kill", "expire", "stale_after_risk", "account_change"])
def test_late_revalidation_races(env, failure):
    e, b, l = env
    cid = propose(e, b, l, approve=True)
    def fresh():
        data = {"SPY": bars(b.now, age=899 if failure == "stale_after_risk" else 0)}
        if failure == "kill": e.kill_path.touch()
        if failure == "expire": b.now += timedelta(seconds=60)
        if failure == "account_change": b.fingerprint = "changed"
        return data, ["SPY"], "5Min"
    def eligible(*_):
        if failure == "stale_after_risk": b.now += timedelta(seconds=2)
        return True
    assert e.submit_approved(cid, freshness=fresh, eligibility=eligible) is None
    assert not b.submissions


def test_approval_is_consumed_and_intent_committed_before_post(env):
    e, b, l = env
    cid = propose(e, b, l, approve=True)
    original = b.submit_order
    def inspect(order):
        with operator_store(l.root) as store:
            assert store.get(cid)["status"] == "CONSUMED"
            assert store.db.execute("SELECT status FROM orders WHERE cid=?", (cid,)).fetchone()[0] == "UNKNOWN"
        return original(order)
    b.submit_order = inspect
    assert submit(e, cid) == cid


def test_atomic_consumption_rolls_back_on_intent_failure(env):
    e, b, l = env
    cid = propose(e, b, l, approve=True)
    l.db.execute("CREATE TRIGGER fail_intent BEFORE INSERT ON orders BEGIN SELECT RAISE(ABORT, 'test crash'); END")
    with pytest.raises(Exception):
        e.approvals.consume(cid, e.approvals.get(cid)["payload"], b.now)
    assert e.approvals.get(cid)["status"] == "APPROVED" and not l.orders()


@pytest.mark.parametrize("approved", [False, True])
def test_restart_preserves_exact_proposal_and_consent(tmp_path, approved):
    b = FakeBroker()
    settings = Settings(_env_file=None, EXECUTION_MODE="paper", STATE_DIR=str(tmp_path / "risk"),
                        RISK_LOCK_FILE=str(tmp_path / "risk" / "RISK_LOCKED.txt"))
    with PaperLedger(tmp_path / "paper") as l:
        e = PaperExecution(b, l, settings, execute=True, now=lambda: b.now)
        cid = propose(e, b, l, approve=approved)
        before = e.approvals.get(cid)
    with PaperLedger(tmp_path / "paper") as l:
        e = PaperExecution(b, l, settings, execute=True, now=lambda: b.now)
        assert e.approvals.get(cid) == before
        assert submit(e, cid) == (cid if approved else None)
        assert len(b.submissions) == int(approved)


def test_ambiguous_post_never_replays_consent(env):
    e, b, l = env
    cid = propose(e, b, l, approve=True)
    b.raise_after_submit = True
    with pytest.raises(BrokerError, match="uncertain"):
        submit(e, cid)
    assert e.approvals.get(cid)["status"] == "CONSUMED"
    assert submit(e, cid) is None
    e.reconcile()
    assert len(b.submissions) == 1


@pytest.mark.parametrize("reason", ["finish", "cancel", "new_session", "error", "ended"])
def test_cleanup_or_wrong_session_cannot_submit_approval(env, reason):
    e, b, l = env
    cid = propose(e, b, l, approve=True)
    if reason == "finish": assert e.finish()
    if reason == "cancel": e.cancel_entries()
    if reason == "new_session": l.set("session_id", "new")
    if reason == "error": l.set("status", "ERROR")
    if reason == "ended": l.set("status", "ENDED")
    assert submit(e, cid) is None
    assert not b.submissions


def test_exit_and_kill_cleanup_do_not_require_approval(env):
    e, b, l = env
    cid = propose(e, b, l, approve=True)
    assert submit(e, cid) == cid
    e.kill_path.touch()
    b.now = b.now.replace(hour=18)
    assert e.finish()
    assert [o["side"] for o in b.submissions] == ["buy", "sell"]
    assert len(e.approvals.rows()) == 1


def test_no_operator_session_expires_and_closes_without_orders(env):
    e, b, l = env
    messages = []
    def wait(seconds): b.now += timedelta(seconds=seconds)
    assert run_paper_session(e, lambda *_: ([setup()], False, {"SPY": bars(b.now)}),
        lambda: {}, ["SPY"], max_iterations=2, wait=wait, report=messages.append,
        monotonic_clock=lambda: b.now.timestamp())
    assert not b.submissions and l.get("status") == "ENDED"
    assert any("PENDING HUMAN APPROVAL" in m for m in messages)
    assert e.approvals.rows()[0]["status"] == "EXPIRED"


def test_session_end_does_not_wait_for_pending_approval(env):
    e, b, l = env
    def wait(_):
        assert not e.approvals.rows(), "Session end must not wait for approval"
    assert run_paper_session(e, lambda *_: ([setup()], False, {"SPY": bars(b.now)}),
        lambda: {}, ["SPY"], max_iterations=1, wait=wait, report=lambda _: None)
    assert not b.submissions
    assert e.approvals.rows()[0]["status"] == "INVALIDATED"


def test_slow_refresh_kill_interrupts_before_post(env):
    e, b, l = env
    release = Event()
    def refresh(_):
        release.wait(2)
        return {"SPY": bars(b.now)}
    def wait(_): e.kill_path.touch()
    try:
        with pytest.raises(BrokerError):
            approval_freshness(e, refresh, ["SPY"], "5Min", b.now + timedelta(seconds=60), wait)
    finally:
        release.set()
    assert not b.submissions


def test_cli_inspect_approve_reject_history_without_broker(env):
    import main
    e, b, l = env
    # Proposal fixture stays during an open pre-cutoff ET time. CLI clock is
    # explicitly faked, never dependent on the actual test machine's hour.
    cid = propose(e, b, l)
    row = e.approvals.get(cid)
    class FixedDateTime:
        @staticmethod
        def now(tz=None): return b.now
    original = operator_store
    with patch("broker.paper_approvals.operator_store", lambda: original(l.root)), \
         patch("main.datetime", FixedDateTime), patch("broker.paper_approvals.datetime", wraps=datetime) as dt, \
         patch("broker.alpaca_adapter.AlpacaBrokerAdapter") as adapter:
        dt.now.return_value = b.now
        runner = CliRunner()
        result = runner.invoke(main.app, ["paper-proposals"])
        assert result.exit_code == 0 and cid in result.output and row["digest"] in result.output
        result = runner.invoke(main.app, ["paper-approve", cid, "--digest", row["digest"]])
        assert result.exit_code == 0 and "no order submitted" in result.output
        result = runner.invoke(main.app, ["paper-approve", cid, "--digest", row["digest"]])
        assert result.exit_code == 1
        second = propose(e, b, l, key="two")
        result = runner.invoke(main.app, ["paper-reject", second, "--digest", e.approvals.get(second)["digest"]])
        assert result.exit_code == 0
        result = runner.invoke(main.app, ["paper-proposals", "--history"])
        assert "APPROVED" in result.output and "REJECTED" in result.output
        assert "test-paper-account-hash" not in result.output
        adapter.assert_not_called()
    assert not b.submissions


def test_two_operator_decisions_serialize(env):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    e, b, l = env
    cid = propose(e, b, l)
    identity = e.approvals.get(cid)["digest"]
    barrier = Barrier(2)
    def decide(approve):
        with operator_store(l.root) as store:
            barrier.wait(timeout=5)
            try:
                store.decide(cid, identity, approve=approve, now=b.now)
                return True
            except BrokerError:
                return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(decide, [True, False])) == 1
    assert not b.submissions
    assert len(list(l.db.execute("SELECT event FROM audit WHERE event IN ('HUMAN_APPROVED','HUMAN_REJECTED')"))) == 1


def test_crash_after_consent_consumption_is_never_replayed(env):
    e, b, l = env
    cid = propose(e, b, l, approve=True)
    e.approvals.consume(cid, e.approvals.get(cid)["payload"], b.now)
    restarted = PaperExecution(b, l, e.settings, execute=True, now=lambda: b.now)
    assert submit(restarted, cid) is None
    with pytest.raises(BrokerError, match="Unresolved"):
        restarted.reconcile()
    assert not b.submissions


def test_private_submit_cannot_bypass_human_approval(env):
    e, b, l = env
    cid = propose(e, b, l)
    with pytest.raises(BrokerError, match="approval"):
        e._submit(e.approvals.get(cid)["payload"], "bypass")
    assert not b.submissions and not l.orders()


def test_preview_creates_no_approvable_proposal(env):
    e, b, l = env
    e.execute = False
    assert propose(e, b, l) is None
    assert not e.approvals.rows() and not b.submissions


def test_approved_same_symbol_proposals_cannot_double_enter(env):
    e, b, l = env
    first = propose(e, b, l, key="first", approve=True)
    second = propose(e, b, l, key="second", approve=True)
    assert submit(e, first) == first
    assert submit(e, second) is None
    assert len(b.submissions) == 1


def test_revalidation_does_not_count_own_unsent_intent_as_trade(env):
    e, b, l = env
    e.settings.MAX_TRADES_PER_DAY = 1
    cid = propose(e, b, l, approve=True)
    assert submit(e, cid) == cid
    assert e.entries_today() == 1


def test_old_v13_order_history_still_reconciles_without_new_consent(env):
    e, b, l = env
    cid = propose(e, b, l, approve=True)
    assert submit(e, cid) == cid
    # Old orders did not carry approval context. No retroactive approval is
    # needed to observe their fills or exit them; never used for a new entry.
    payload = l.orders()[0]["payload"]
    payload.pop("setup")
    payload.pop("account_fingerprint")
    with l.db:
        l.db.execute("UPDATE orders SET payload=? WHERE cid=?", (json.dumps(payload), cid))
        l.db.execute("DROP TABLE proposals")
    from broker.paper_approvals import initialize
    initialize(l.db)
    e.reconcile()
    assert e.finish() and len(b.submissions) == 2


def test_session_final_gate_enforces_frozen_policy(env):
    e, b, l = env
    cid = propose(e, b, l, approve=True)
    # The operator approved before another alert caused an existing cooldown.
    cooldowns = {"ticker_counts": {"SPY": 20}, "setup_counts": {}}
    # Force a documented policy rejection using the existing confidence rule,
    # without modifying a stored proposal/hash or the policy implementation.
    from trades.shadow_session import ShadowEntryPolicy, ShadowEntryDecision
    with patch.object(ShadowEntryPolicy, "evaluate", return_value=ShadowEntryDecision("SKIP", ["TICKER_COOLED_DOWN"])):
        assert run_paper_session(e, lambda *_: ([], False, {"SPY": bars(b.now)}),
            lambda: cooldowns, ["SPY"], max_iterations=1, wait=lambda _: None,
            refresh_data=lambda _: {"SPY": bars(b.now)}, report=lambda _: None)
    assert not b.submissions and l.orders()[0]["status"] == "rejected"


@pytest.mark.parametrize("stale", [False, True])
def test_real_adapter_and_sdk_hook_require_exact_consumed_approval(env, stale):
    from types import SimpleNamespace
    from broker.alpaca_adapter import AlpacaBrokerAdapter
    from tests.test_paper_execution import managed_wrapper, frozen_datetime
    original, b, l = env
    wrapper, sdk, _, now = managed_wrapper()
    adapter = AlpacaBrokerAdapter(original.settings, execute_paper=True)
    adapter._client = wrapper
    # Read-only broker snapshots are faked; the REAL adapter->wrapper->gate->SDK
    # request construction path remains in use. SDK submit itself is a mock.
    adapter.list_positions = b.list_positions
    adapter.list_open_orders = b.list_open_orders
    adapter.get_order_by_client_id = b.get_order_by_client_id
    def acknowledge(*, order_data):
        order = dict(symbol=order_data.symbol, qty=order_data.qty, side=order_data.side.value,
                     type=order_data.type.value, limit_price=order_data.limit_price,
                     client_order_id=order_data.client_order_id)
        return SimpleNamespace(**b.submit_order(order))
    sdk.submit_order.side_effect = acknowledge
    e = PaperExecution(adapter, l, original.settings, execute=True, now=lambda: b.now)
    with patch("broker.alpaca_client.datetime", frozen_datetime(now)):
        cid = propose(e, b, l)
        assert submit(e, cid) is None
        sdk.submit_order.assert_not_called()
        e.approvals.decide(cid, e.approvals.get(cid)["digest"], approve=True, now=b.now)
        result = submit(e, cid, age=901 if stale else 0)
        assert result == (None if stale else cid)
    assert sdk.submit_order.call_count == int(not stale)
    assert e.approvals.get(cid)["status"] == "CONSUMED"
