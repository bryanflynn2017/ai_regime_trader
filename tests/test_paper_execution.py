"""Offline v1.3 fault-injection tests. Never use workstation keys or real orders."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from broker.base import BrokerError
from broker.paper_execution import PaperLedger, PaperExecution, TERMINAL
from config.settings import Settings
from strategy.day_trade_setups import DayTradeSetup, SetupType
from trades.paper_session import entry_data_invalid, run_paper_session


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Execution tests must never contact the network")
    monkeypatch.setattr("requests.Session.request", blocked)


class FakeBroker:
    is_paper = True

    def __init__(self):
        self.now = datetime(2026, 10, 5, 14, tzinfo=timezone.utc)
        self.orders = {}
        self.submissions = []
        self.cancellations = []
        self.equity = 100000
        self.cash = 100000
        self.fingerprint = "test-paper-account-hash"
        self.verified = True
        self.market_open = True
        self.fill_immediately = True
        self.raise_after_submit = False
        self.cancel_immediately = True
        self.extra_positions = []
        self.clock_offset = 0
        self.close_at = self.now.replace(hour=20)

    def verify_paper_account(self):
        return dict(paper_verified=self.verified, fingerprint=self.fingerprint,
                    equity=self.equity, cash=self.cash, buying_power=self.cash)

    def get_clock(self):
        return dict(is_open=self.market_open, timestamp=(self.now + timedelta(seconds=self.clock_offset)).isoformat(),
                    next_close=self.close_at.isoformat())

    def list_positions(self):
        positions = {}
        for order in self.orders.values():
            sign = 1 if order["side"] == "buy" else -1
            positions[order["symbol"]] = positions.get(order["symbol"], Decimal(0)) + sign * Decimal(str(order["filled_qty"]))
        return [dict(symbol=s, qty=q, market_value=q * 100, side="long")
                for s, q in positions.items() if q] + self.extra_positions

    def list_open_orders(self):
        return [deepcopy(o) for o in self.orders.values() if o["status"] not in TERMINAL]

    def get_order_by_client_id(self, cid):
        return deepcopy(self.orders.get(cid))

    def submit_order(self, order):
        assert order["client_order_id"] not in self.orders, "duplicate broker POST"
        self.submissions.append(deepcopy(order))
        cid = order["client_order_id"]
        response = dict(order, id="fake-" + cid, status="new", filled_qty=0,
                        filled_avg_price="", updated_at=self.now.isoformat())
        self.orders[cid] = response
        if self.fill_immediately:
            self.fill(cid, order["qty"], 100 if order["side"] == "buy" else 101)
        if self.raise_after_submit:
            raise RuntimeError("SECRET account=DO_NOT_LOG")
        return deepcopy(self.orders[cid])

    def fill(self, cid, qty, price):
        o = self.orders[cid]
        o.update(filled_qty=qty, filled_avg_price=price,
                 status="filled" if qty == o["qty"] else "partially_filled",
                 filled_at=self.now.isoformat())

    def cancel_order(self, oid):
        self.cancellations.append(oid)
        order = next(o for o in self.orders.values() if o["id"] == oid)
        order["status"] = "canceled" if self.cancel_immediately else "pending_cancel"
        return True


def setup(symbol="SPY"):
    return DayTradeSetup(symbol, datetime(2026, 10, 5, 10), SetupType.OPENING_RANGE_BREAKOUT_LONG,
                         True, .9, 100, 98, 104, 106, 2, 3, "test", metadata={"rank": 1})


def bars(now, *, age=0):
    frame = pd.DataFrame({"open": [100], "high": [101], "low": [99], "close": [100],
                          "volume": [1000], "vwap": [99]}, index=[pd.Timestamp(now).tz_localize(None)])
    frame.attrs["latest_bar_time"] = (now - timedelta(seconds=age)).replace(tzinfo=None).isoformat()
    return frame


@pytest.fixture
def env(tmp_path):
    s = Settings(_env_file=None, ALPACA_API_KEY="", ALPACA_SECRET_KEY="", EXECUTION_MODE="paper",
                 STATE_DIR=str(tmp_path / "risk"), RISK_LOCK_FILE=str(tmp_path / "risk" / "RISK_LOCKED.txt"))
    broker = FakeBroker()
    with PaperLedger(tmp_path / "paper") as ledger:
        engine = PaperExecution(broker, ledger, s, execute=True, now=lambda: broker.now)
        yield engine, broker, ledger


def enter(engine, key="1", symbol="SPY"):
    # Explicit fake operator, never a production auto-approval path.
    l = engine.ledger
    if l.get("status") is None:
        l.set("status", "RUNNING")
        l.set("session_id", "test-session")
        l.set("partial", False)
        l.set("partial_reasons", [])
        l.set("started_at", engine.now().isoformat())
        l.set("config", dict(symbols=["SPY"], timeframe="5Min", interval=300))
    config = l.get("config")
    config["symbols"] = sorted(set(config["symbols"]) | {symbol})
    l.set("config", config)
    cid = engine.enter(setup(symbol), key=key, session_id=l.get("session_id"))
    if not cid:
        return None
    engine.approvals.decide(cid, engine.approvals.get(cid)["digest"], approve=True, now=engine.now())
    return engine.submit_approved(cid,
        freshness=lambda: ({s: bars(engine.now()) for s in config["symbols"]}, config["symbols"], "5Min"),
        eligibility=lambda *_: True)


def test_default_configuration_has_no_live_mode():
    assert Settings(_env_file=None).EXECUTION_MODE == "shadow"
    with pytest.raises(ValueError):
        Settings(_env_file=None, EXECUTION_MODE="live")


def test_explicit_activation_required(env):
    e, b, l = env
    e.settings.EXECUTION_MODE = "shadow"
    with pytest.raises(BrokerError, match="activation|MODE"):
        PaperExecution(b, l, e.settings, execute=True)
    assert not b.submissions


@pytest.mark.parametrize("url", ["https://api.alpaca.markets", "http://paper-api.alpaca.markets",
    "https://paper-api.alpaca.markets.attacker.test", "https://attacker.test/paper-api.alpaca.markets",
    "https://paper-api.alpaca.markets@attacker.test", "https://paper-api.alpaca.markets:443",
    "https://paper-api.alpaca.markets?credential=SECRET"])
def test_exact_endpoint_enforcement(url):
    from broker.alpaca_client import AlpacaClientWrapper, AlpacaSafetyError
    s = Settings(_env_file=None, ALPACA_BASE_URL=url)
    with pytest.raises(AlpacaSafetyError) as exc:
        AlpacaClientWrapper(s).validate_paper_safety()
    assert "SECRET" not in str(exc.value)


@pytest.mark.parametrize("url", ["https://api.alpaca.markets/v2/orders", "http://paper-api.alpaca.markets/v2/orders",
                                "https://paper-api.alpaca.markets.attacker.test/v2/orders"])
def test_transport_rejects_nonpaper_without_network(url):
    from broker.paper_transport import PaperTransport
    with patch("requests.Session.request") as request, pytest.raises(BrokerError):
        PaperTransport().post(url)
    request.assert_not_called()


def test_transport_pins_timeout_redirect_and_tls():
    from broker.paper_transport import PaperTransport
    transport = PaperTransport()
    with patch("requests.Session.request") as request:
        transport.post("https://paper-api.alpaca.markets/v2/orders", allow_redirects=True, verify=False)
    kwargs = request.call_args.kwargs
    assert kwargs["allow_redirects"] is False and kwargs["verify"] is True
    assert kwargs["timeout"] == (5, 15) and transport.trust_env is False


def account_client():
    from alpaca.common.enums import BaseURL
    client = MagicMock()
    client._base_url = BaseURL.TRADING_PAPER
    client.get_account.return_value = SimpleNamespace(
        id="private-account-id", status="ACTIVE", trading_blocked=False, account_blocked=False,
        trade_suspended_by_user=False, equity="100000", cash="100000", buying_power="100000")
    return client


def test_account_positive_verification_no_id_in_result():
    from broker.alpaca_client import AlpacaClientWrapper
    wrapper = AlpacaClientWrapper(Settings(_env_file=None), _client_for_testing=account_client())
    result = wrapper.verify_paper_account()
    assert result["paper_verified"] is True and len(result["fingerprint"]) == 64
    assert "private-account-id" not in str(result)


@pytest.mark.parametrize("field,value", [("status", "INACTIVE"), ("trading_blocked", True),
    ("account_blocked", True), ("trade_suspended_by_user", True), ("trading_blocked", None)])
def test_account_unverifiable_or_blocked_fails_closed(field, value):
    from broker.alpaca_client import AlpacaClientWrapper, AlpacaSafetyError
    client = account_client()
    setattr(client.get_account.return_value, field, value)
    with pytest.raises(AlpacaSafetyError):
        AlpacaClientWrapper(Settings(_env_file=None), _client_for_testing=client).verify_paper_account()


@pytest.mark.parametrize("method", ["submit_bracket_buy", "submit_market_sell", "cancel_all_orders"])
def test_legacy_mutation_path_disabled(method):
    from broker.alpaca_client import AlpacaClientWrapper, AlpacaSafetyError
    client = account_client()
    wrapper = AlpacaClientWrapper(Settings(_env_file=None), _client_for_testing=client)
    args = {"submit_bracket_buy": ("SPY", 1, 98, 104), "submit_market_sell": ("SPY", 1), "cancel_all_orders": ()}
    with pytest.raises(AlpacaSafetyError):
        getattr(wrapper, method)(*args[method])
    client.submit_order.assert_not_called()
    client.cancel_orders.assert_not_called()


def test_dry_run_never_submits_or_cancels(env):
    e, b, l = env
    e.execute = False
    enter(e)
    e.finish()
    assert not b.submissions and not b.cancellations and not l.orders()
    assert "DRY_RUN_ORDER" in [r[0] for r in l.db.execute("SELECT event FROM audit")]


def test_fill_projection_and_exact_pnl(env):
    e, b, l = env
    cid = enter(e)
    e.reconcile()
    trade = e.tracker.list_open_trades()[0]
    assert trade.shares == 10 and trade.entry_price == 100
    e.exit(cid, "TARGET_1_HIT")
    e.reconcile()
    trade = e.tracker.list_all_trades()[0]
    assert trade.status == "CLOSED" and trade.realized_pnl == 10
    assert trade.exit_price == 101  # broker fill, NOT hypothetical target 104
    assert trade.test_type == "alpaca_paper" and trade.exclude_from_analytics


def test_duplicate_signal_and_pending_ticker_blocked(env):
    e, b, l = env
    b.fill_immediately = False
    cid = enter(e)
    assert enter(e) is None and enter(e, "another") is None
    assert len(b.submissions) == 1
    assert l.orders()[0]["cid"] == cid


def test_duplicate_signal_after_closed_trade_still_blocked(env):
    e, b, l = env
    cid = enter(e)
    e.exit(cid, "STOP_HIT")
    assert enter(e) is None
    assert len(b.submissions) == 2


def test_timeout_after_acceptance_reconciles_without_second_post(env):
    e, b, l = env
    b.raise_after_submit = True
    with pytest.raises(BrokerError, match="uncertain"):
        enter(e)
    assert l.orders()[0]["status"] == "UNKNOWN"
    b.raise_after_submit = False
    e.reconcile()
    assert l.orders()[0]["status"] == "filled"
    assert enter(e) is None and len(b.submissions) == 1
    assert "SECRET" not in "".join(r[0] for r in l.db.execute("SELECT details FROM audit"))


def test_unknown_404_never_replays(env):
    e, b, l = env
    b.raise_after_submit = True
    with pytest.raises(BrokerError):
        enter(e)
    b.orders.clear()
    with pytest.raises(BrokerError, match="Unresolved"):
        enter(e)
    assert len(b.submissions) == 1


def test_partial_entry_cancel_confirm_before_sell(env):
    e, b, l = env
    b.fill_immediately = False
    cid = enter(e)
    b.fill(cid, 4, 99.8)
    b.cancel_immediately = False
    assert e.exit(cid, "STOP_HIT") is None
    assert len(b.submissions) == 1 and b.cancellations
    b.orders[cid]["status"] = "canceled"
    b.fill_immediately = True
    e.exit(cid, "STOP_HIT")
    e.reconcile()
    assert b.submissions[-1]["qty"] == 4
    assert e.tracker.list_all_trades()[0].realized_pnl == pytest.approx(4.8)


def test_partial_sell_does_not_duplicate_then_canceled_remainder_recovered(env):
    e, b, l = env
    cid = enter(e)
    b.fill_immediately = False
    sell = e.exit(cid, "STOP_HIT")
    b.fill(sell, 3, 99)
    assert e.exit(cid, "STOP_HIT") is None
    e.reconcile()
    assert e.tracker.list_open_trades()[0].shares == 7
    b.orders[sell]["status"] = "canceled"
    b.fill_immediately = True
    e.exit(cid, "STOP_HIT")
    e.reconcile()
    assert b.submissions[-1]["qty"] == 7
    assert e.tracker.list_all_trades()[0].realized_pnl == pytest.approx(4)


@pytest.mark.parametrize("status", ["rejected", "canceled", "expired"])
def test_terminal_unfilled_orders_never_make_trades(env, status):
    e, b, l = env
    b.fill_immediately = False
    cid = enter(e)
    b.orders[cid]["status"] = status
    e.reconcile()
    assert not e.tracker.list_all_trades()
    assert enter(e) is None


@pytest.mark.parametrize("time", [(12, 59, True), (13, 0, False), (14, 0, False)])
def test_cutoff_et_and_exits(env, time):
    e, b, l = env
    hour, minute, allowed = time
    b.now = b.now.replace(hour=hour + 4, minute=minute)
    cid = enter(e)
    assert bool(cid) == allowed
    if cid:
        b.now = b.now.replace(hour=18)
        assert e.exit(cid, "STOP_HIT")


@pytest.mark.parametrize("gate", ["partial", "kill", "stale", "market", "clock", "account"])
def test_entry_safety_gates(env, gate):
    e, b, l = env
    if gate == "partial": l.set("partial", True)
    if gate == "kill": e.kill_path.touch()
    if gate == "market": b.market_open = False
    if gate == "clock": b.clock_offset = 120
    if gate == "account": b.verified = False
    try:
        e.enter(setup(), key="one", session_id="s", stale=gate == "stale")
    except BrokerError:
        pass
    assert not b.submissions


def test_foreign_position_not_adopted_or_liquidated(env):
    e, b, l = env
    b.extra_positions = [dict(symbol="AAPL", qty=1, market_value=100, side="long")]
    with pytest.raises(BrokerError, match="mismatch"):
        enter(e)
    assert not b.submissions


def test_foreign_order_blocks_without_cancel(env):
    e, b, l = env
    b.orders["foreign"] = dict(client_order_id="foreign", id="foreign", symbol="SPY", side="buy",
                               status="new", filled_qty=0)
    with pytest.raises(BrokerError, match="Unowned"):
        e.reconcile()
    assert not b.cancellations


def test_account_binding_survives_restart(env):
    e, b, l = env
    e.reconcile()
    b.fingerprint = "different-account"
    with pytest.raises(BrokerError, match="Account differs"):
        e.reconcile()


def test_journal_rebuilt_from_durable_fills(env):
    e, b, l = env
    enter(e)
    e.tracker._write_all([])
    recovered = PaperExecution(b, l, e.settings, execute=True, now=lambda: b.now)
    recovered.reconcile()
    assert len(recovered.tracker.list_open_trades()) == 1
    assert len(b.submissions) == 1


def test_process_lock_and_release(tmp_path):
    with PaperLedger(tmp_path):
        with pytest.raises((OSError, BlockingIOError)):
            PaperLedger(tmp_path)
    with PaperLedger(tmp_path) as ledger:
        assert ledger.orders() == []


@pytest.mark.parametrize("qty", [float("nan"), float("inf"), -1, 0])
def test_order_rejects_invalid_quantity(qty):
    from broker.paper_transport import validate_order
    with pytest.raises(BrokerError):
        validate_order(dict(symbol="SPY", qty=qty, side="buy", type="limit", limit_price=100,
                            client_order_id="art13-" + "a" * 32))


def test_risk_sizing_and_pending_exposure(env):
    e, b, l = env
    b.equity = b.cash = 2000
    b.fill_immediately = False
    cid = enter(e)
    assert b.submissions[0]["qty"] == 2  # 10% per-position ceiling, not $1000
    assert e.risk._s.MAX_SINGLE_POSITION_EXPOSURE == .10
    for i, symbol in enumerate(["AAPL", "QQQ", "MSFT", "TSLA", "NVDA"]):
        enter(e, str(i + 2), symbol)
    assert sum(o["qty"] * o["limit_price"] for o in b.submissions) <= 1000
    assert len(b.submissions) <= e.settings.MAX_OPEN_POSITIONS


def test_day_week_peak_references_are_durable(env):
    e, b, l = env
    e.references(100000)
    b.equity = 96000
    assert enter(e) is None
    assert l.get("risk_references")["day_equity"] == 100000
    b.now += timedelta(days=1)
    refs = e.references(96000)
    assert refs["day_equity"] == 96000 and refs["week_equity"] == 100000 and refs["peak_equity"] == 100000


def test_incomplete_cleanup_never_ended(env):
    e, b, l = env
    enter(e)
    b.fill_immediately = False
    assert not e.finish()
    assert l.get("status") == "CLEANUP_PENDING"
    assert e.tracker.list_open_trades()


@pytest.mark.parametrize("minutes,stale", [(15, False), (15.01, True), (3, False)])
def test_paper_freshness_threshold_unchanged(minutes, stale):
    now = datetime(2026, 10, 5, 14, tzinfo=timezone.utc)
    assert entry_data_invalid({"SPY": bars(now, age=minutes * 60)}, ["SPY"], "5Min", now) is stale


def test_missing_and_future_timestamps_block():
    now = datetime(2026, 10, 5, 14, tzinfo=timezone.utc)
    assert entry_data_invalid({}, ["SPY"], "5Min", now)
    assert entry_data_invalid({"SPY": bars(now, age=-10)}, ["SPY"], "5Min", now)


def run_one(e, b, setups=None, stale=False):
    def fake_operator_wait(seconds):
        b.now += timedelta(seconds=seconds)
        for proposal in e.approvals.rows():
            if proposal["status"] == "PENDING":
                e.approvals.decide(proposal["cid"], proposal["digest"], approve=True, now=b.now)
    return run_paper_session(e, lambda *args: (setups or [], stale, {"SPY": bars(b.now)}),
                            lambda: {}, ["SPY"], max_iterations=2, wait=fake_operator_wait, report=lambda _: None,
                            monotonic_clock=lambda: b.now.timestamp(),
                            refresh_data=lambda symbols: {s: bars(b.now) for s in symbols})


def test_frozen_pipeline_full_session_closes_and_persists_flat(env):
    e, b, l = env
    assert run_one(e, b, [setup()])
    assert l.get("status") == "ENDED" and not e.tracker.list_open_trades()
    assert [o["side"] for o in b.submissions] == ["buy", "sell"]


def test_partial_scan_cannot_enter(env):
    e, b, l = env
    run_one(e, b, [setup()], stale=True)
    assert not b.submissions and l.get("partial") is True


def test_ended_session_cannot_donate_open_trade(env):
    e, b, l = env
    enter(e)
    l.set("status", "ENDED")
    with pytest.raises(BrokerError, match="ENDED"):
        run_one(e, b)
    assert l.get("status") == "ENDED" and len(b.submissions) == 1


def test_missing_state_cannot_donate_open_trade(env):
    e, b, l = env
    enter(e)
    with l.db:
        l.db.execute("DELETE FROM meta WHERE key='status'")
    with pytest.raises(BrokerError, match="Missing paper session"):
        run_one(e, b)
    assert len(b.submissions) == 1


def recovery_state(e, l, b):
    l.set("status", "RUNNING")
    l.set("config", dict(symbols=["SPY"], timeframe="5Min", interval=300))
    l.set("session_id", "recover-me")
    l.set("started_at", b.now.isoformat())
    l.set("partial", False)
    l.set("partial_reasons", [])


def test_recover_running_session_and_gap_blocks_entry(env):
    e, b, l = env
    recovery_state(e, l, b)
    l.set("last_scan", (b.now - timedelta(hours=2)).isoformat())
    run_one(e, b, [setup()])
    assert l.get("session_id") == "recover-me"
    assert "ITERATION_GAP" in l.get("partial_reasons") and not b.submissions


def test_kill_switch_flattens_owned_only(env):
    e, b, l = env
    enter(e)
    recovery_state(e, l, b)
    e.kill_path.touch()
    run_one(e, b)
    assert not e.tracker.list_open_trades() and l.get("status") == "ENDED"
    assert e.kill_path.exists()


@pytest.mark.parametrize("args", [["--execution-mode", "live"], ["--execution-mode", "shadow", "--execute-paper"]])
def test_cli_blocks_live_and_implicit_activation(args):
    from typer.testing import CliRunner
    import main
    with patch("broker.alpaca_adapter.AlpacaBrokerAdapter") as adapter:
        result = CliRunner().invoke(main.app, ["paper-session", *args])
    assert result.exit_code != 0
    adapter.assert_not_called()


def test_cli_default_dispatches_original_shadow_without_broker():
    from typer.testing import CliRunner
    import main
    with patch.object(main, "forward_test_session") as shadow, patch("broker.alpaca_adapter.AlpacaBrokerAdapter") as adapter:
        result = CliRunner().invoke(main.app, ["paper-session", "--execution-mode", "shadow", "--max-iterations", "1"])
    assert result.exit_code == 0
    adapter.assert_not_called()
    assert shadow.call_args.kwargs["entry_cutoff"] == "13:00"


def managed_wrapper():
    from broker.alpaca_client import AlpacaClientWrapper
    client = account_client()
    now = datetime(2026, 10, 5, 14, tzinfo=timezone.utc)
    client.get_clock.return_value = SimpleNamespace(is_open=True, timestamp=now,
                                                    next_open=now, next_close=now.replace(hour=20))
    client.get_asset.return_value = SimpleNamespace(tradable=True, status="active",
                                                    asset_class="us_equity", fractionable=True)
    order = dict(symbol="SPY", qty=1.25, side="buy", type="limit", limit_price=100,
                 client_order_id="art13-" + "a" * 32, valid_until=(now + timedelta(minutes=3)).isoformat())
    client.submit_order.return_value = SimpleNamespace(
        **{k: v for k, v in order.items() if k != "type"}, id="broker-order", status="new",
        filled_qty=0, filled_avg_price=None, updated_at=now)
    s = Settings(_env_file=None, EXECUTION_MODE="paper")
    return AlpacaClientWrapper(s, dry_run=False, _client_for_testing=client), client, order, now


def frozen_datetime(now):
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)
    return Frozen


def test_sdk_order_schema_paper_limit_day_idempotency():
    wrapper, client, order, now = managed_wrapper()
    with patch("broker.alpaca_client.datetime", frozen_datetime(now)):
        response = wrapper.submit_managed_order(order, before_submit=lambda _: True)
    request = client.submit_order.call_args.kwargs["order_data"]
    assert request.symbol == "SPY" and request.qty == 1.25 and request.limit_price == 100
    assert request.time_in_force.value == "day" and request.extended_hours is False
    assert request.client_order_id == order["client_order_id"] and response["status"] == "new"


@pytest.mark.parametrize("field,value", [("tradable", False), ("status", "inactive"),
    ("asset_class", "crypto"), ("fractionable", False)])
def test_asset_validation_before_submission(field, value):
    from broker.alpaca_client import AlpacaSafetyError
    wrapper, client, order, now = managed_wrapper()
    setattr(client.get_asset.return_value, field, value)
    with patch("broker.alpaca_client.datetime", frozen_datetime(now)), pytest.raises(AlpacaSafetyError):
        wrapper.submit_managed_order(order, before_submit=lambda _: True)
    client.submit_order.assert_not_called()


@pytest.mark.parametrize("gate", ["dry_run", "mode", "missing_guard", "false_guard", "cutoff", "deadline", "clock", "closed"])
def test_wrapper_rechecks_safety_before_actual_post(gate):
    from broker.alpaca_client import AlpacaSafetyError
    wrapper, client, order, now = managed_wrapper()
    guard = lambda _: True
    if gate == "dry_run": wrapper.dry_run = True
    if gate == "mode": wrapper._s.EXECUTION_MODE = "shadow"
    if gate == "missing_guard": guard = None
    if gate == "false_guard": guard = lambda _: False
    if gate == "cutoff": now = now.replace(hour=17)
    if gate == "deadline": order["valid_until"] = now.isoformat()
    if gate == "clock": client.get_clock.return_value.timestamp = now - timedelta(minutes=2)
    if gate == "closed": client.get_clock.return_value.is_open = False
    with patch("broker.alpaca_client.datetime", frozen_datetime(now)), pytest.raises(AlpacaSafetyError):
        wrapper.submit_managed_order(order, before_submit=guard)
    client.submit_order.assert_not_called()


@pytest.mark.parametrize("code", [404, 401, 500])
def test_order_lookup_only_404_is_absence(code):
    from broker.alpaca_client import AlpacaConnectionError
    wrapper, client, order, now = managed_wrapper()
    class Failure(Exception):
        status_code = code
    client.get_order_by_client_id.side_effect = Failure("SECRET")
    if code == 404:
        assert wrapper.get_order_by_client_id("id") is None
    else:
        with pytest.raises(AlpacaConnectionError) as exc:
            wrapper.get_order_by_client_id("id")
        assert "SECRET" not in str(exc.value)


def test_lookup_failure_does_not_become_empty_position(env):
    e, b, l = env
    b.fill_immediately = False
    enter(e)
    b.get_order_by_client_id = MagicMock(side_effect=TimeoutError("secret"))
    with pytest.raises(TimeoutError):
        e.reconcile()
    assert len(b.submissions) == 1 and l.orders()[0]["status"] == "new"


@pytest.mark.parametrize("field,value", [("side", "sell"), ("symbol", "QQQ"),
    ("qty", 99), ("filled_qty", 99), ("status", "unrecognized"), ("client_order_id", "wrong")])
def test_bad_acknowledgement_never_creates_fake_trade(env, field, value):
    e, b, l = env
    b.fill_immediately = False
    cid = enter(e)
    b.orders[cid][field] = value
    with pytest.raises(BrokerError):
        e.reconcile()
    assert not e.tracker.list_all_trades()


def test_late_fill_after_cancel_request_still_owned(env):
    e, b, l = env
    b.fill_immediately = False
    cid = enter(e)
    b.cancel_immediately = False
    e.cancel_entries()
    b.fill(cid, 10, 100)
    b.fill_immediately = True
    assert e.finish()
    assert len(b.submissions) == 2 and b.submissions[-1]["qty"] == 10


def test_real_ledger_reopen_and_journal_recovery(tmp_path):
    b = FakeBroker()
    s = Settings(_env_file=None, EXECUTION_MODE="paper", STATE_DIR=str(tmp_path / "risk"),
                 RISK_LOCK_FILE=str(tmp_path / "risk" / "RISK_LOCKED.txt"))
    with PaperLedger(tmp_path / "orders") as l:
        e = PaperExecution(b, l, s, execute=True, now=lambda: b.now)
        cid = enter(e)
    with PaperLedger(tmp_path / "orders") as l:
        e = PaperExecution(b, l, s, execute=True, now=lambda: b.now)
        e.reconcile()
        assert e.tracker.list_open_trades()[0].trade_id == cid
        assert enter(e) is None
        assert len(b.submissions) == 1


def test_corrupt_risk_baselines_fail_closed(env):
    e, b, l = env
    l.set("risk_references", dict(day="2026-10-05", week="2026-41", day_equity=float("nan"),
                                  week_equity=100000, peak_equity=100000))
    with pytest.raises(BrokerError):
        enter(e)
    assert not b.submissions


def test_final_deadline_blocks_after_slow_preflight(env):
    e, b, l = env
    deadline = b.now - timedelta(seconds=1)
    assert e.enter(setup(), key="old", session_id="s", valid_until=deadline) is None
    assert not b.submissions


def test_kill_guard_rechecked_after_durable_intent(env):
    e, b, l = env
    b.fill_immediately = False
    cid = enter(e)
    row = l.orders()[0]
    with l.db:
        l.db.execute("UPDATE orders SET status='UNKNOWN' WHERE cid=?", (cid,))
    # Restarting/replaying an intent cannot reuse consumed approval context.
    assert e.submission_guard(row["payload"]["order"]) is False
    e.kill_path.touch()
    assert e.submission_guard(row["payload"]["order"]) is False


def test_hung_data_scan_does_not_starve_kill_switch(env):
    from threading import Event
    from trades.paper_session import supervised_scan, CleanupRequested
    e, b, l = env
    release = Event()
    def scan(*args):
        release.wait(2)
        return [], False, {}
    def wait(_):
        e.kill_path.touch()
    try:
        with pytest.raises(CleanupRequested):
            supervised_scan(e, scan, ["SPY"], [], wait, lambda: 0)
    finally:
        release.set()
    assert not b.submissions


def test_early_close_cleanup_before_regular_close(env):
    e, b, l = env
    enter(e)
    recovery_state(e, l, b)
    b.close_at = b.now + timedelta(seconds=50)
    assert run_one(e, b)
    assert l.get("status") == "ENDED" and not e.tracker.list_open_trades()


@pytest.mark.parametrize("field", ["partial", "partial_reasons", "started_at", "session_id"])
def test_incomplete_recovery_metadata_blocks(env, field):
    e, b, l = env
    recovery_state(e, l, b)
    with l.db:
        l.db.execute("DELETE FROM meta WHERE key=?", (field,))
    with pytest.raises(BrokerError, match="Incomplete recoverable"):
        run_one(e, b, [setup()])
    assert not b.submissions


def test_rejected_exit_stays_open_and_recoverable(env):
    e, b, l = env
    cid = enter(e)
    b.fill_immediately = False
    sell = e.exit(cid, "STOP_HIT")
    b.orders[sell]["status"] = "rejected"
    assert not e.finish()
    assert len(b.submissions) == 2  # never spin on the same rejected exit
    assert e.tracker.list_open_trades()


def test_adapter_fails_closed_on_truncated_open_orders():
    from broker.alpaca_adapter import AlpacaBrokerAdapter
    adapter = AlpacaBrokerAdapter(Settings(_env_file=None))
    adapter._client = MagicMock()
    adapter._client.get_orders.return_value = [{}] * 500
    with pytest.raises(BrokerError, match="truncated"):
        adapter.list_open_orders()


def test_impossible_fill_above_limit_fails_closed(env):
    e, b, l = env
    b.fill_immediately = False
    cid = enter(e)
    b.fill(cid, 10, 110)
    with pytest.raises(BrokerError, match="exceeds submitted limit"):
        e.reconcile()


def test_sdk_sell_remains_available_after_cutoff():
    wrapper, client, order, now = managed_wrapper()
    now = now.replace(hour=18)
    client.get_clock.return_value.timestamp = now
    order.update(side="sell", type="market")
    order.pop("limit_price")
    order.pop("valid_until")
    with patch("broker.alpaca_client.datetime", frozen_datetime(now)):
        wrapper.submit_managed_order(order, before_submit=lambda _: True)
    request = client.submit_order.call_args.kwargs["order_data"]
    assert request.side.value == "sell" and request.time_in_force.value == "day"


def test_winter_cutoff_uses_new_york_not_fixed_utc(env):
    e, b, l = env
    b.now = datetime(2026, 11, 2, 18, tzinfo=timezone.utc)
    b.close_at = b.now.replace(hour=21)
    assert enter(e) is None
    assert not b.submissions


def test_known_unsent_preflight_rejection_does_not_block_cleanup(env):
    from broker.base import OrderNotSubmitted
    e, b, l = env
    enter(e, symbol="AAPL")
    original = b.submit_order
    def reject_after_intent(order):
        assert any(r["cid"] == order["client_order_id"] for r in l.orders())
        e.kill_path.touch()
        raise OrderNotSubmitted("final local kill check")
    b.submit_order = reject_after_intent
    assert enter(e, "second", "SPY") is None
    assert l.orders()[-1]["status"] == "rejected"
    b.submit_order = original
    assert e.finish()
    assert not e.tracker.list_open_trades()


def test_fractional_partial_fills_do_not_leave_float_dust(env):
    e, b, l = env
    b.cash = 30
    cid = enter(e)
    assert b.submissions[0]["qty"] == .3
    b.fill_immediately = False
    sell = e.exit(cid, "STOP_HIT")
    b.fill(sell, .1, 101)
    b.orders[sell]["status"] = "canceled"
    b.fill_immediately = True
    e.exit(cid, "STOP_HIT")
    assert b.submissions[-1]["qty"] == .2
    assert e.finish()
    assert e.tracker.list_all_trades()[0].status == "CLOSED"


def test_broker_dust_prevents_ended_even_below_reconciliation_tolerance(env):
    e, b, l = env
    b.extra_positions = [dict(symbol="SPY", qty=.000000001, market_value=.0000001, side="long")]
    assert not e.finish()
    assert l.get("status") == "CLEANUP_PENDING"


def test_transport_paces_reconciliation_bursts():
    from broker.paper_transport import PaperTransport
    transport = PaperTransport()
    with patch("requests.Session.request"), patch("broker.paper_transport.monotonic", return_value=100), \
            patch("broker.paper_transport.sleep") as wait:
        transport.get("https://paper-api.alpaca.markets/v2/clock")
        transport.get("https://paper-api.alpaca.markets/v2/clock")
    wait.assert_called_once_with(.4)
