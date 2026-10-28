"""
ai_regime_trader — CLI entry point.

EDUCATIONAL / PAPER TRADING ONLY.
This bot defaults to Alpaca paper trading and has hard-coded risk controls.

Usage:
    python main.py --help
    python main.py check-config
    python main.py test-connection
    python main.py backtest --start 2020-01-01 --end 2023-12-31
    python main.py paper-once
    python main.py paper-loop
    python main.py setup-scan
    python main.py dashboard
    python main.py unlock-risk
"""
from __future__ import annotations

import json
import hashlib
import sys
import time
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional

import typer

app = typer.Typer(
    name="ai_regime_trader",
    help=(
        "AI Regime Trader — Educational paper trading bot.\n\n"
        "NOT financial advice. Paper trading only by default."
    ),
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Managed paper execution (shadow remains the default)
# ---------------------------------------------------------------------------

@app.command()
def paper_session(
    execution_mode: Optional[str] = typer.Option(None, "--execution-mode", help="shadow (default) or paper; live unavailable"),
    execute_paper: bool = typer.Option(False, "--execute-paper/--dry-run", help="Explicit paper submission activation; default read-only"),
    watchlist: str = typer.Option("momentum_universe", "--watchlist"),
    timeframe: str = typer.Option("5Min", "--timeframe"),
    interval_seconds: int = typer.Option(300, "--interval-seconds", min=1),
    max_iterations: int = typer.Option(72, "--max-iterations", min=1),
) -> None:
    """v1.4 human-approved frozen strategy. Paper dry run NEVER mutates broker."""
    from config.settings import settings
    mode = execution_mode or settings.EXECUTION_MODE
    if mode not in {"shadow", "paper"} or (execute_paper and mode != "paper"):
        raise typer.BadParameter("Use shadow or explicitly activate paper; live unavailable")
    if timeframe not in {"1Min", "5Min"}:
        raise typer.BadParameter("Only existing intraday 1Min/5Min timeframes supported")
    if mode == "shadow":
        return forward_test_session(
            ticker="SPY", watchlist=watchlist, synthetic=False, provider_name="alpaca",
            alpaca_feed="iex", timeframe=timeframe, lookback_days=None,
            interval_seconds=interval_seconds, max_iterations=max_iterations,
            top_n=5, scan_min_confidence=0.6, min_confidence=0.80, max_rank=1,
            max_open_trades=3, max_new_trades_per_day=10, entry_cutoff="13:00",
            sim_notional=1000, discord=False, end_time=None, resume=True,
            ignore_market_hours=False, max_data_failures=5,
            suppress_cooled_down_alerts=True, close_open_at_session_end=True,
        )
    from broker.alpaca_adapter import AlpacaBrokerAdapter
    from broker.paper_execution import PaperLedger, PaperExecution
    from trades.paper_session import run_paper_session
    selected = settings.model_copy(update={"EXECUTION_MODE": "paper"})
    symbols = merge_tickers(None, watchlist)
    typer.echo("PAPER ONLY | " + ("HUMAN APPROVAL REQUIRED FOR EACH ENTRY" if execute_paper else "READ-ONLY DRY RUN: no submission/cancellation"))
    try:
        broker = AlpacaBrokerAdapter(settings_override=selected, execute_paper=execute_paper)
        provider = _get_named_data_provider("alpaca", synthetic=False, csv_file=None, alpaca_feed="iex")

        def scan(tickers, monitoring):
            setups, stale, bars = _run_shadow_scan_once(tickers, provider, timeframe, 400, 0.6, 5, False)
            extra = [s for s in monitoring if s not in tickers]
            if extra:
                bars.update(provider.get_multiple_bars(extra, timeframe=timeframe))
            return setups, stale, bars

        root = "state/paper" if execute_paper else "state/paper_preview"
        with PaperLedger(root) as ledger:
            engine = PaperExecution(broker, ledger, selected, execute=execute_paper)
            complete = run_paper_session(engine, scan, _daily_setup_cooldown_summary, symbols,
                                        timeframe=timeframe, interval_seconds=interval_seconds,
                                        max_iterations=max_iterations, report=typer.echo,
                                        refresh_data=lambda tickers: provider.get_multiple_bars(tickers, timeframe=timeframe))
        if not complete:
            raise typer.Exit(1)
    except (KeyboardInterrupt, Exception) as exc:
        if isinstance(exc, typer.Exit):
            raise
        typer.echo("HOLD: paper session could not complete safely. No live routing is available. "
                   "Review local paper audit/state and broker paper dashboard before restarting.")
        raise typer.Exit(1) from None


@app.command()
def paper_proposals(history: bool = typer.Option(False, "--history")) -> None:
    """Inspect exact paper entry proposals; --history includes decisions/fills."""
    from broker.paper_approvals import operator_store, digest
    from datetime import timezone
    try:
        with operator_store() as store:
            store.expire(datetime.now(timezone.utc))
            orders = {r["cid"]: r for r in store.db.execute("SELECT cid,status,filled,average FROM orders")}
            for row in store.rows():
                if not history and row["status"] not in {"PENDING", "APPROVED"}:
                    continue
                if digest(row["payload"]) != row["digest"]:
                    raise ValueError("Invalid proposal")
                order = row["payload"]["order"]
                outcome = dict(orders[row["cid"]]) if row["cid"] in orders else None
                typer.echo(json.dumps(dict(id=row["cid"], status=row["status"], digest=row["digest"],
                    order=order, trade_plan=row["payload"]["trade"], created_at=row["payload"]["at"],
                    expires=row["expires"], decided_at=row["decided_at"],
                    reason=row["reason"], outcome=outcome), sort_keys=True))
    except Exception:
        typer.echo("HOLD: paper proposal ledger unavailable or invalid; no broker action taken.")
        raise typer.Exit(1) from None


def _paper_decision(proposal_id: str, digest_value: str, approve: bool):
    from broker.paper_approvals import operator_store
    try:
        with operator_store() as store:
            store.decide(proposal_id, digest_value, approve=approve)
        typer.echo("APPROVED for exact-order revalidation only; no order submitted by this command."
                   if approve else "REJECTED; no order submitted.")
    except Exception:
        typer.echo("REFUSED: proposal unavailable, expired, changed, already decided, or session ineligible.")
        raise typer.Exit(1) from None


@app.command()
def paper_approve(proposal_id: str, digest_value: str = typer.Option(..., "--digest")) -> None:
    """Approve ONE inspected exact paper entry; never extends its expiry."""
    _paper_decision(proposal_id, digest_value, True)


@app.command()
def paper_reject(proposal_id: str, digest_value: str = typer.Option(..., "--digest")) -> None:
    """Reject ONE exact pending entry. No broker access."""
    _paper_decision(proposal_id, digest_value, False)


@app.command()
def paper_kill() -> None:
    """Latch paper kill switch: supervisor cancels buys and flattens owned positions."""
    path = Path("state/paper/KILL_SWITCH")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    typer.echo("Paper kill switch latched. Verify broker paper orders/positions; network failures can delay cleanup. "
               "Remove state/paper/KILL_SWITCH only after manual review. Shadow is unaffected.")


# ---------------------------------------------------------------------------
# Watchlist support
# ---------------------------------------------------------------------------

def load_watchlist(name_or_path: str) -> list[str]:
    """
    Load tickers from a watchlist.

    If name_or_path is a JSON filename without path, look in watchlists/ dir.
    If name_or_path is a path, load that file directly.
    Raises typer.Exit on file not found or invalid JSON.
    Returns list of tickers (normalized to uppercase).
    """
    watchlist_dir = Path("watchlists")

    # Try as filename in watchlists/ directory first
    if "/" not in name_or_path and "\\" not in name_or_path:
        file_path = watchlist_dir / f"{name_or_path}.json"
    else:
        file_path = Path(name_or_path)

    # If name_or_path had no extension, try adding .json
    if not file_path.suffix:
        file_path = Path(str(file_path) + ".json")

    if not file_path.exists():
        typer.secho(
            f"Error: Watchlist not found: {file_path}\n",
            fg=typer.colors.RED,
        )
        typer.echo(f"Available watchlists:")
        for wl_name in get_available_watchlists():
            typer.echo(f"  - {wl_name}")
        raise typer.Exit(1)

    try:
        data = json.loads(file_path.read_text())
        tickers = data.get("tickers", [])
        if not tickers:
            typer.secho(
                f"Error: Watchlist {file_path} has no tickers.\n",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
        return [t.upper() for t in tickers]
    except json.JSONDecodeError as e:
        typer.secho(
            f"Error: Invalid JSON in {file_path}: {e}\n",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)


def get_available_watchlists() -> list[str]:
    """Return list of available watchlist names (without .json extension)."""
    watchlist_dir = Path("watchlists")
    if not watchlist_dir.exists():
        return []
    return sorted([f.stem for f in watchlist_dir.glob("*.json")])


def merge_tickers(ticker_str: Optional[str], watchlist_name: Optional[str]) -> list[str]:
    """
    Merge tickers from --ticker and --watchlist options.

    Returns deduplicated, uppercase tickers preserving order.
    If both are empty, returns empty list (caller handles default SPY).
    """
    tickers = []
    seen = set()

    # Load from --ticker
    if ticker_str:
        for t in ticker_str.split(","):
            t = t.strip().upper()
            if t and t not in seen:
                tickers.append(t)
                seen.add(t)

    # Load from --watchlist
    if watchlist_name:
        for t in load_watchlist(watchlist_name):
            if t not in seen:
                tickers.append(t)
                seen.add(t)

    return tickers


# ---------------------------------------------------------------------------
# Phase 1: check-config (fully implemented)
# ---------------------------------------------------------------------------

@app.command()
def check_config() -> None:
    """Validate environment variables and settings configuration."""
    from config.settings import settings

    typer.secho("\n=== ai_regime_trader — Configuration Check ===\n", bold=True)

    rows = [
        ("PAPER_TRADING",            settings.PAPER_TRADING),
        ("LIVE_TRADING_ALLOWED",     settings.LIVE_TRADING_ALLOWED),
        ("ALPACA_PAPER",             settings.ALPACA_PAPER),
        ("ALPACA_BASE_URL",          settings.ALPACA_BASE_URL),
        ("ALPACA_API_KEY set",       bool(settings.ALPACA_API_KEY)),
        ("ALPACA_SECRET_KEY set",    bool(settings.ALPACA_SECRET_KEY)),
        ("BENCHMARK",                settings.BENCHMARK),
        ("TICKERS",                  ", ".join(settings.TICKERS)),
        ("MAX_RISK_PER_TRADE",       f"{settings.MAX_RISK_PER_TRADE:.1%}"),
        ("MAX_PORTFOLIO_EXPOSURE",   f"{settings.MAX_PORTFOLIO_EXPOSURE:.0%}"),
        ("MAX_SINGLE_POS_EXPOSURE",  f"{settings.MAX_SINGLE_POSITION_EXPOSURE:.0%}"),
        ("MAX_DAILY_DD_WARN",        f"{settings.MAX_DAILY_DRAWDOWN_WARN:.0%}"),
        ("MAX_DAILY_DD_STOP",        f"{settings.MAX_DAILY_DRAWDOWN_STOP:.0%}"),
        ("MAX_WEEKLY_DD_STOP",       f"{settings.MAX_WEEKLY_DRAWDOWN_STOP:.0%}"),
        ("MAX_TOTAL_DD_LOCK",        f"{settings.MAX_TOTAL_DRAWDOWN_LOCK:.0%}"),
        ("REQUIRE_STOP_LOSS",        settings.REQUIRE_STOP_LOSS),
        ("DEFAULT_STOP_LOSS_PCT",    f"{settings.DEFAULT_STOP_LOSS_PCT:.0%}"),
        ("DEFAULT_TAKE_PROFIT_PCT",  f"{settings.DEFAULT_TAKE_PROFIT_PCT:.0%}"),
        ("MAX_TRADES_PER_DAY",       settings.MAX_TRADES_PER_DAY),
        ("MAX_OPEN_POSITIONS",       settings.MAX_OPEN_POSITIONS),
        ("RISK_LOCK_FILE",           settings.RISK_LOCK_FILE),
        ("RISK_LOCKED NOW",          settings.is_risk_locked()),
    ]

    for key, val in rows:
        typer.echo(f"  {key:<28} {val}")

    # Highlight any concerning values
    warnings_found = False
    typer.echo("")
    if not settings.PAPER_TRADING:
        typer.secho("  [WARN] PAPER_TRADING is False!", fg=typer.colors.RED, bold=True)
        warnings_found = True
    if settings.LIVE_TRADING_ALLOWED:
        typer.secho("  [WARN] LIVE_TRADING_ALLOWED is True!", fg=typer.colors.RED, bold=True)
        warnings_found = True
    if not settings.ALPACA_PAPER:
        typer.secho("  [WARN] ALPACA_PAPER is False!", fg=typer.colors.RED, bold=True)
        warnings_found = True
    if "paper-api" not in settings.ALPACA_BASE_URL:
        typer.secho("  [WARN] ALPACA_BASE_URL does not point to paper endpoint!", fg=typer.colors.RED, bold=True)
        warnings_found = True
    if not settings.has_broker_credentials():
        typer.secho("  [NOTE] Broker credentials not set — needed for live paper loop.", fg=typer.colors.YELLOW)
    if settings.is_risk_locked():
        typer.secho("  [WARN] Risk lock file exists! Run `unlock-risk` for details.", fg=typer.colors.RED, bold=True)
        warnings_found = True

    if not warnings_found:
        typer.secho("\nAll safety checks passed.", fg=typer.colors.GREEN)
    else:
        typer.secho("\nWarnings found — review before proceeding.", fg=typer.colors.YELLOW)


# ---------------------------------------------------------------------------
# Phase 7: test-connection (fully implemented)
# ---------------------------------------------------------------------------

@app.command()
def test_connection() -> None:
    """
    Verify Alpaca PAPER account connection and print account summary.

    Credentials are read from .env — never from this command.
    The secret key is never printed.
    """
    from broker.alpaca_client import AlpacaClientWrapper, AlpacaConnectionError, AlpacaSafetyError

    client = AlpacaClientWrapper()

    typer.secho("\n=== Alpaca Paper Connection Test ===\n", bold=True)

    if not client.is_configured():
        typer.secho("  Alpaca paper credentials not found.", fg=typer.colors.YELLOW)
        typer.echo("  To connect to your paper account:")
        typer.echo("    1. Sign up at https://alpaca.markets (free)")
        typer.echo("    2. Go to Paper Trading -> API Keys")
        typer.echo("    3. Copy .env.example to .env:")
        typer.echo("         copy .env.example .env  (Windows)")
        typer.echo("    4. Fill in ALPACA_API_KEY and ALPACA_SECRET_KEY")
        typer.echo("  NEVER paste API keys into Claude or any chat tool.")
        return

    try:
        client.validate_paper_safety()
    except AlpacaSafetyError as exc:
        typer.secho(f"  Safety check failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)

    if not client.is_available():
        typer.secho("  alpaca-py is not installed; live connection test skipped.", fg=typer.colors.YELLOW)
        typer.echo("  Install alpaca-py in this Python environment to test the paper account connection.")
        typer.echo("  Paper safety settings were checked; no broker request was made.")
        return

    typer.echo("  Connecting to Alpaca paper endpoint...")
    try:
        account = client.get_account()
        clock   = client.get_clock()
    except AlpacaConnectionError as exc:
        typer.secho(f"  Connection failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)

    equity       = float(account.get("equity",       "0"))
    buying_power = float(account.get("buying_power", "0"))

    typer.secho("\n  Account summary:", bold=True)
    typer.echo(f"    Status:        {account.get('status', 'unknown')}")
    typer.echo(f"    Paper mode:    True (hard-coded)")
    typer.echo(f"    Equity:        ${equity:,.2f}")
    typer.echo(f"    Buying power:  ${buying_power:,.2f}")
    typer.echo(f"    Market open:   {clock.get('is_open', False)}")
    if not clock.get("is_open", True):
        typer.echo(f"    Next open:     {clock.get('next_open', 'unknown')}")

    typer.secho("\n  Connection verified. Paper trading only.", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------
# Phase 7: paper-dry-run (fully implemented)
# ---------------------------------------------------------------------------

@app.command()
def paper_dry_run(
    ticker:    str  = typer.Option("SPY",   help="Ticker to analyze"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Use synthetic data (offline)"),
) -> None:
    """
    Fetch data, generate a signal, evaluate risk, and simulate an order.

    Runs the full pipeline with dry_run=True — no real order is placed.
    Use this to verify the system before running paper-loop.
    """
    import pandas as pd
    from data.market_data import DataFetchError, fetch_ticker, generate_synthetic_ohlcv
    from strategy.strategy_orchestrator import StrategyOrchestrator
    from strategy.signals import SignalAction
    from risk.risk_manager import RiskManager
    from broker.alpaca_client import AlpacaClientWrapper
    from broker.order_executor import OrderExecutor

    typer.secho(f"\n=== Paper Dry-Run: {ticker.upper()} ===\n", bold=True)

    # -- Data --
    if synthetic:
        typer.secho("  Using synthetic data (offline mode).", fg=typer.colors.YELLOW)
        df = generate_synthetic_ohlcv(rows=300)
        display_ticker = "SYNTHETIC"
    else:
        typer.echo(f"  Fetching recent data for {ticker.upper()}...")
        try:
            end   = str(pd.Timestamp.now().date())
            start = str((pd.Timestamp.now() - pd.Timedelta(days=730)).date())
            df    = fetch_ticker(ticker.upper(), start=start, end=end)
            display_ticker = ticker.upper()
        except DataFetchError as exc:
            typer.secho(f"  Data fetch failed: {exc}", fg=typer.colors.RED)
            typer.echo("  Try: py -3.10 main.py paper-dry-run --synthetic")
            raise typer.Exit(1)

    typer.echo(f"  Data: {len(df)} bars loaded.")

    # -- Signal --
    orch   = StrategyOrchestrator()
    signal = orch.generate_signal(display_ticker, df)

    typer.secho("\n  Signal:", bold=True)
    typer.echo(f"    Action:     {signal.action.value}")
    typer.echo(f"    Regime:     {signal.regime.value}")
    typer.echo(f"    Confidence: {signal.confidence:.0%}")
    typer.echo(f"    Reason:     {signal.reason}")
    if signal.action == SignalAction.BUY:
        typer.echo(f"    Entry:      ${signal.entry_price:.2f}")
        typer.echo(f"    Stop loss:  ${signal.stop_loss:.2f}")
        typer.echo(f"    Take profit:${signal.take_profit:.2f}")

    # -- Risk evaluation (placeholder equity) --
    rm       = RiskManager()
    decision = rm.evaluate_signal(
        signal               = signal,
        equity               = 100_000.0,
        starting_day_equity  = 100_000.0,
        starting_week_equity = 100_000.0,
        peak_equity          = 100_000.0,
    )

    typer.secho("\n  Risk decision:", bold=True)
    typer.echo(f"    Approved:     {decision.approved}")
    typer.echo(f"    Final action: {decision.final_action.value}")
    if decision.rejection_reasons:
        typer.echo(f"    Rejected:     {decision.rejection_reasons[0]}")
    if decision.position_size and decision.position_size.approved:
        ps = decision.position_size
        typer.echo(f"    Shares:       {ps.shares:.4f}")
        typer.echo(f"    Pos value:    ${ps.position_value:,.0f}")

    # -- Dry-run order execution via SimulatedBrokerAdapter --
    from broker.simulated_adapter import SimulatedBrokerAdapter
    broker   = SimulatedBrokerAdapter()
    executor = OrderExecutor(broker_adapter=broker, dry_run=True)
    result   = executor.execute_risk_decision(signal, decision)

    typer.secho("\n  [DRY RUN] Order simulation:", bold=True)
    typer.echo(f"    Would submit: {result.submitted}")
    typer.echo(f"    Action:       {result.action}")
    typer.echo(f"    Quantity:     {result.quantity:.4f}")
    typer.echo(f"    Status:       {result.status}")
    typer.echo(f"    Reason:       {result.reason}")

    typer.secho("\n  No real order was placed.", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------
# Phase 6: backtest (fully implemented)
# ---------------------------------------------------------------------------

@app.command()
def backtest(
    ticker:    str  = typer.Option("SPY",        help="Ticker symbol (yfinance) or 'all' for full universe"),
    start:     str  = typer.Option("2020-01-01", help="Start date YYYY-MM-DD (yfinance only)"),
    end:       str  = typer.Option("2023-12-31", help="End date YYYY-MM-DD (yfinance only)"),
    csv_file:     str  = typer.Option(None,  "--csv",          help="Load OHLCV from a CSV file instead of yfinance"),
    synthetic:    bool = typer.Option(False, "--synthetic",    help="Use synthetic random-walk data (offline smoke test)"),
    save_results: bool = typer.Option(False, "--save-results", help="Save metrics to results/backtests/backtest_TIMESTAMP.json"),
) -> None:
    """
    Run historical strategy backtest.

    Data sources (in priority order):
      1. --synthetic : offline random-walk data, no network needed
      2. --csv FILE  : load OHLCV from a CSV file
      3. default     : fetch from Yahoo Finance via yfinance

    Use --save-results to persist metrics for later review in the dashboard.
    WARNING: Good backtest results do not guarantee live trading results.
    """
    from config.settings import settings
    from data_providers.base import DataProviderError
    from backtest.backtester import Backtester

    provider = _get_data_provider(synthetic, csv_file)

    # -- Load data via DataProvider --
    if synthetic:
        typer.secho("\n=== ai_regime_trader Backtest (SYNTHETIC DATA) ===\n", bold=True)
        typer.secho(
            f"  NOTE: Using {provider.provider_name} — NOT real market data.\n"
            "  Results are for pipeline smoke-testing only.",
            fg=typer.colors.YELLOW,
        )
        data = provider.get_multiple_bars(["SYNTHETIC"])
        data_label = f"{list(data.values())[0].__len__()} synthetic bars"

    elif csv_file:
        typer.secho("\n=== ai_regime_trader Backtest (CSV) ===\n", bold=True)
        typer.echo(f"  Source: {csv_file} (provider: {provider.provider_name})")
        try:
            from pathlib import Path as _Path
            label = _Path(csv_file).stem.upper()
            df    = provider.get_bars(label)
            data  = {label: df}
            data_label = f"{len(df)} rows from {csv_file}"
        except DataProviderError as exc:
            typer.secho(f"\nCSV load failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(1)

    else:
        typer.secho("\n=== ai_regime_trader Backtest ===\n", bold=True)
        typer.echo(f"  Period:  {start} to {end}")
        tickers = settings.TICKERS if ticker.lower() == "all" else [ticker.upper()]
        typer.echo(f"  Tickers: {', '.join(tickers)}")
        typer.echo(f"\nFetching data via {provider.provider_name}...")
        try:
            data = provider.get_multiple_bars(tickers, start=start, end=end)
        except DataProviderError as exc:
            typer.secho(f"\nData fetch failed:\n  {exc}", fg=typer.colors.RED)
            typer.secho("\nSuggestions:", bold=True)
            typer.echo("  1. Retry later (Yahoo Finance is sometimes temporarily unavailable)")
            typer.echo("  2. Upgrade yfinance:")
            typer.echo("       py -3.11 -m pip install --upgrade yfinance")
            typer.echo("  3. Smoke-test with synthetic data (no network needed):")
            typer.echo("       py -3.11 main.py backtest --synthetic")
            typer.echo("  4. Use a local CSV file:")
            typer.echo("       py -3.11 main.py backtest --csv path/to/data.csv")
            raise typer.Exit(1)
        data_label = f"{len(data)} ticker(s)"

    # -- Run backtest --
    typer.echo(f"\n  Data: {data_label}. Running backtest...")
    bt     = Backtester(starting_equity=100_000.0, slippage_bps=5.0)
    result = bt.run(data)
    m      = result.metrics

    typer.echo(f"\n{'-'*42}")
    typer.echo(f"  Total return:      {m.total_return:+.1%}")
    typer.echo(f"  Benchmark return:  {m.benchmark_return:+.1%}")
    typer.echo(f"  Excess return:     {m.excess_return:+.1%}")
    typer.echo(f"  CAGR:              {m.cagr:+.1%}")
    typer.echo(f"  Sharpe ratio:      {m.sharpe_ratio:.2f}")
    typer.echo(f"  Sortino ratio:     {m.sortino_ratio:.2f}")
    typer.echo(f"  Max drawdown:      {m.max_drawdown:.1%}")
    typer.echo(f"  Total trades:      {m.total_trades}")
    typer.echo(f"  Win rate:          {m.win_rate:.0%}")
    typer.echo(f"  Profit factor:     {m.profit_factor:.2f}")
    typer.echo(f"  Final equity:      ${result.final_equity:,.0f}")
    typer.echo(f"{'-'*42}")

    if result.warnings:
        typer.secho(f"\nWarnings ({len(result.warnings)}):", fg=typer.colors.YELLOW)
        for w in result.warnings[:5]:
            typer.echo(f"  {w}")

    if save_results:
        import dataclasses
        bt_data = {
            "timestamp":       datetime.now().isoformat(),
            "source":          "synthetic" if synthetic else ("csv" if csv_file else "yfinance"),
            "tickers":         list(data.keys()),
            "starting_equity": 100_000.0,
            "final_equity":    result.final_equity,
            "metrics":         dataclasses.asdict(result.metrics),
            "total_trades":    len(result.trades),
            "warnings":        result.warnings[:10],
            "config":          result.config_snapshot,
        }
        saved = _save_result_json(bt_data, "backtests", "backtest")
        typer.secho(f"\nResults saved: {saved}", fg=typer.colors.GREEN)

    typer.secho(
        "\nWARNING: Backtest results do not guarantee live performance.",
        fg=typer.colors.YELLOW,
    )


# ---------------------------------------------------------------------------
# Phase 6: walk-forward (fully implemented)
# ---------------------------------------------------------------------------

@app.command()
def walk_forward(
    ticker:    str  = typer.Option("SPY",        help="Ticker (yfinance)"),
    start:     str  = typer.Option("2018-01-01", help="Data start date YYYY-MM-DD"),
    end:       str  = typer.Option("2023-12-31", help="Data end date YYYY-MM-DD"),
    csv_file:     str  = typer.Option(None,  "--csv",          help="Load OHLCV from a CSV file"),
    synthetic:    bool = typer.Option(False, "--synthetic",    help="Use synthetic data (offline)"),
    save_results: bool = typer.Option(False, "--save-results", help="Save metrics to results/walk_forward/wf_TIMESTAMP.json"),
) -> None:
    """
    Run walk-forward validation (train 252 days, test 126 days, step 126 days).

    Supports --synthetic and --csv for offline use when yfinance fails.
    Use --save-results to persist metrics for later review in the dashboard.
    WARNING: Good walk-forward results do not guarantee live performance.
    """
    from data_providers.base import DataProviderError
    from backtest.walk_forward import WalkForwardValidator

    provider = _get_data_provider(synthetic, csv_file)

    # -- Load data via DataProvider --
    if synthetic:
        typer.secho("\n=== Walk-Forward Validation (SYNTHETIC DATA) ===\n", bold=True)
        typer.secho(
            f"  NOTE: Using {provider.provider_name} — NOT real market data.",
            fg=typer.colors.YELLOW,
        )
        from data_providers.synthetic_provider import SyntheticDataProvider
        data = SyntheticDataProvider(rows=750).get_multiple_bars(["SYNTHETIC"])
    elif csv_file:
        typer.secho("\n=== Walk-Forward Validation (CSV) ===\n", bold=True)
        typer.echo(f"  Source: {csv_file} (provider: {provider.provider_name})")
        try:
            from pathlib import Path as _Path
            label = _Path(csv_file).stem.upper()
            data = {label: provider.get_bars(label)}
        except DataProviderError as exc:
            typer.secho(f"\nCSV load failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(1)
    else:
        typer.secho("\n=== Walk-Forward Validation ===\n", bold=True)
        typer.echo(f"  Ticker: {ticker.upper()}  ({start} to {end})")
        try:
            data = {ticker.upper(): provider.get_bars(ticker.upper(), start=start, end=end)}
        except DataProviderError as exc:
            typer.secho(f"\nData fetch failed:\n  {exc}", fg=typer.colors.RED)
            typer.echo("\nSuggestions:")
            typer.echo("  py -3.11 main.py walk-forward --synthetic")
            typer.echo("  py -3.11 main.py walk-forward --csv path/to/data.csv")
            raise typer.Exit(1)

    wfv    = WalkForwardValidator(train_days=252, test_days=126, step_days=126)
    result = wfv.run(data)

    if not result.windows:
        typer.secho("Not enough data for walk-forward validation.", fg=typer.colors.YELLOW)
        return

    typer.echo(f"\n  Windows run:   {len(result.windows)}")
    typer.echo(f"\n  {'Window':<8} {'Train ret':>10} {'Test ret':>10} "
               f"{'Test Sharpe':>12} {'Test MaxDD':>12}")
    typer.echo("  " + "-" * 54)
    for i, w in enumerate(result.windows, 1):
        tr = w.train_result.metrics.total_return
        te = w.test_result.metrics.total_return
        sh = w.test_result.metrics.sharpe_ratio
        dd = w.test_result.metrics.max_drawdown
        typer.echo(f"  {i:<8} {tr:>+9.1%}  {te:>+9.1%}  {sh:>11.2f}  {dd:>11.1%}")

    m = result.combined_metrics
    typer.echo(f"\n  Combined test: return={m.total_return:+.1%}  "
               f"sharpe={m.sharpe_ratio:.2f}  max_dd={m.max_drawdown:.1%}")
    if save_results:
        import dataclasses
        wf_data = {
            "timestamp":        datetime.now().isoformat(),
            "source":           "synthetic" if synthetic else ("csv" if csv_file else f"yfinance:{ticker}"),
            "tickers":          list(data.keys()),
            "train_days":       252,
            "test_days":        126,
            "step_days":        126,
            "n_windows":        len(result.windows),
            "combined_metrics": dataclasses.asdict(result.combined_metrics),
            "windows": [
                {
                    "window":       i + 1,
                    "train_start":  str(w.train_start.date()),
                    "train_end":    str(w.train_end.date()),
                    "test_start":   str(w.test_start.date()),
                    "test_end":     str(w.test_end.date()),
                    "train_return": w.train_result.metrics.total_return,
                    "test_return":  w.test_result.metrics.total_return,
                    "test_sharpe":  w.test_result.metrics.sharpe_ratio,
                    "test_max_dd":  w.test_result.metrics.max_drawdown,
                }
                for i, w in enumerate(result.windows)
            ],
            "warnings": result.warnings[:10],
        }
        saved = _save_result_json(wf_data, "walk_forward", "wf")
        typer.secho(f"\nResults saved: {saved}", fg=typer.colors.GREEN)

    typer.secho(
        "\nWARNING: Walk-forward results do not guarantee live performance.",
        fg=typer.colors.YELLOW,
    )


# ===========================================================================
# Phase 8: State management, helpers, paper-once, paper-loop
# ===========================================================================

_PAPER_STATE_FILE    = "state/paper_state.json"
_RESULTS_BASE        = "results"
_PAPER_DECISIONS_LOG = "logs/paper_decisions.jsonl"

_DEFAULT_PAPER_STATE: Dict = {
    "equity":               100_000.0,
    "starting_day_equity":  100_000.0,
    "starting_week_equity": 100_000.0,
    "peak_equity":          100_000.0,
    "trades_today":         0,
    "last_run_timestamp":   None,
    "last_run_date":        None,
    "last_run_iso_week":    None,
}


def _save_result_json(
    data:     Dict,
    subdir:   str,
    prefix:   str,
    base_dir: str = _RESULTS_BASE,
) -> Path:
    """
    Persist a result dict to a timestamped JSON file.

    The dict is sanitised through mask_or_exclude_sensitive_values before
    writing so API keys and secrets can never end up on disk via this path.
    Returns the Path of the written file.
    """
    from dashboard.helpers import mask_or_exclude_sensitive_values

    clean     = mask_or_exclude_sensitive_values(data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path      = Path(base_dir) / subdir / f"{prefix}_{timestamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2, default=str), encoding="utf-8")
    return path


def _parse_tickers(ticker_args: List[str]) -> List[str]:
    """
    Parse ticker arguments, handling comma-separated strings.

    Example inputs:
      ["SPY,QQQ,AAPL"]   -> ["SPY", "QQQ", "AAPL"]
      ["SPY", "QQQ"]     -> ["SPY", "QQQ"]
      ["spy,qqq"]        -> ["SPY", "QQQ"]
      ["SPY,SPY"]        -> ["SPY"]   (deduplicated)
    """
    seen: Dict[str, None] = {}
    for arg in ticker_args:
        for t in arg.split(","):
            t = t.strip().upper()
            if t:
                seen[t] = None
    return list(seen.keys())


def _load_paper_state(state_file: str = _PAPER_STATE_FILE) -> Dict:
    """Load paper state from JSON. Returns defaults if file is missing or corrupt."""
    path = Path(state_file)
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            state  = _DEFAULT_PAPER_STATE.copy()
            state.update(loaded)
            return state
        except (json.JSONDecodeError, OSError, TypeError):
            pass
    return _DEFAULT_PAPER_STATE.copy()


def _save_paper_state(state: Dict, state_file: str = _PAPER_STATE_FILE) -> None:
    """Persist paper state to JSON. Never stores secrets."""
    path = Path(state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str))


def _update_paper_state(
    state:        Dict,
    equity:       float,
    current_date: str,     # "YYYY-MM-DD"
    iso_week:     int,
    trades_delta: int = 0,
) -> Dict:
    """
    Apply date-based resets and update equity tracking.
    Returns a NEW dict; does not modify the input.

    Resets applied:
    - New calendar day  -> starting_day_equity = equity, trades_today = 0
    - New ISO week      -> starting_week_equity = equity
    - Always            -> peak_equity = max(peak, equity)
    """
    s         = state.copy()
    last_date = s.get("last_run_date")
    last_week = s.get("last_run_iso_week")

    is_first_run = last_date is None
    is_new_day   = is_first_run or (str(current_date) != str(last_date))
    is_new_week  = is_first_run or (iso_week != last_week)

    if is_new_day:
        s["starting_day_equity"] = equity
        s["trades_today"]        = 0

    if is_new_week:
        s["starting_week_equity"] = equity

    # Peak only ever increases
    s["peak_equity"] = max(float(s.get("peak_equity", equity)), equity)

    s["equity"]          = equity
    s["trades_today"]    = s.get("trades_today", 0) + trades_delta
    s["last_run_date"]   = str(current_date)
    s["last_run_iso_week"] = iso_week
    s["last_run_timestamp"] = datetime.now().isoformat()

    return s


def _get_simulated_portfolio():
    """Return an empty $100,000 portfolio for dry-run use without real credentials."""
    from broker.position_tracker import PortfolioSnapshot
    import pandas as pd
    return PortfolioSnapshot(
        equity               = 100_000.0,
        cash                 = 100_000.0,
        buying_power         = 100_000.0,
        portfolio_exposure   = 0.0,
        positions            = {},
        open_positions_count = 0,
        ticker_exposures     = {},
        timestamp            = pd.Timestamp.now(),
        metadata             = {"simulated": True},
    )


def _get_data_provider(synthetic: bool, csv_file: Optional[str]):
    """
    Create the appropriate DataProvider based on CLI flags.

    Returns a DataProvider instance:
        synthetic=True   -> SyntheticDataProvider
        csv_file set     -> CSVDataProvider
        default          -> YFinanceDataProvider
    """
    from data_providers import SyntheticDataProvider, CSVDataProvider, YFinanceDataProvider
    if synthetic:
        return SyntheticDataProvider(rows=500)
    if csv_file:
        return CSVDataProvider(file_path=csv_file)
    return YFinanceDataProvider()


def _get_named_data_provider(
    provider_name: str,
    synthetic: bool = False,
    csv_file: Optional[str] = None,
    alpaca_feed: str = "iex",
):
    """
    Create a DataProvider by explicit provider name.

    The synthetic/csv flags remain supported as backward-compatible aliases for
    commands that predate --provider.
    """
    selected = (provider_name or "yfinance").lower()
    if synthetic:
        selected = "synthetic"
    if csv_file:
        selected = "csv"

    from data_providers import (
        AlpacaDataProvider,
        CSVDataProvider,
        SyntheticDataProvider,
        YFinanceDataProvider,
    )

    if selected == "yfinance":
        return YFinanceDataProvider()
    if selected == "alpaca":
        return AlpacaDataProvider(feed=alpaca_feed)
    if selected == "synthetic":
        return SyntheticDataProvider(rows=500)
    if selected == "csv":
        if not csv_file:
            raise typer.BadParameter("--provider csv requires --csv FILE")
        return CSVDataProvider(file_path=csv_file)

    raise typer.BadParameter("--provider must be one of: yfinance, alpaca, synthetic, csv")


def _alpaca_credentials_configured() -> bool:
    """Return True when Alpaca paper/data credentials are present without exposing them."""
    try:
        from config.settings import settings
    except Exception:
        return False
    return bool(getattr(settings, "ALPACA_API_KEY", "") and getattr(settings, "ALPACA_SECRET_KEY", ""))


def _recommended_intraday_provider(explicit_provider: Optional[str] = None) -> tuple[str, str, Optional[str]]:
    """
    Choose the recommended live intraday data provider.

    Explicit CLI provider choices always win. When omitted, Alpaca IEX is
    preferred if credentials exist; otherwise yfinance remains the fallback.
    """
    if explicit_provider:
        return explicit_provider.lower(), "iex", None
    if _alpaca_credentials_configured():
        return "alpaca", "iex", None
    return (
        "yfinance",
        "iex",
        "Note: Alpaca credentials are not configured. YFinance intraday may be delayed or stale.",
    )


def _data_provider_display_name(selected_provider: str, provider_name: str = "", alpaca_feed: str = "iex") -> str:
    """Return a concise provider label for terminal output."""
    selected = (selected_provider or provider_name or "yfinance").lower()
    if selected == "alpaca" or provider_name == "AlpacaData":
        return f"Alpaca {str(alpaca_feed or 'iex').upper()}"
    if selected == "yfinance" or provider_name == "YFinance":
        return "YFinance"
    if selected == "synthetic" or provider_name == "Synthetic":
        return "Synthetic"
    if selected == "csv" or provider_name == "CSV":
        return "CSV"
    return provider_name or selected_provider


def _resolve_scan_timeframe(selected_provider: str, timeframe: Optional[str], default_timeframe: str) -> str:
    """Resolve setup-scan timeframe defaults and fail fast for unsupported Alpaca timeframes."""
    resolved = timeframe or ("5Min" if selected_provider == "alpaca" else default_timeframe)
    if selected_provider == "alpaca" and resolved not in {"1Min", "5Min"}:
        typer.secho(
            "AlpacaData supports only 1Min and 5Min. Use --timeframe 5Min for setup scans.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    return resolved


def _print_data_provider_error(exc: Exception, indent: str = "") -> None:
    """Print provider failures with a helpful SIP subscription hint when relevant."""
    message = str(exc)
    typer.secho(f"{indent}Data fetch failed: {message}", fg=typer.colors.RED)
    lowered = message.lower()
    if "sip" in lowered and ("subscription" in lowered or "permit" in lowered or "permission" in lowered):
        typer.echo(
            f"{indent}Your Alpaca plan does not permit SIP data. "
            "Use --alpaca-feed iex or upgrade Alpaca data subscription."
        )


def _get_portfolio(client_or_adapter, dry_run: bool):
    """
    Return real portfolio if possible, otherwise simulated.

    Accepts either AlpacaClientWrapper (is_configured is a method) or
    BrokerAdapter (is_configured is a property) via duck typing.
    """
    from broker.position_tracker import PositionTracker

    # Duck typing: support is_configured() as method OR property
    is_conf_raw = client_or_adapter.is_configured
    configured  = is_conf_raw() if callable(is_conf_raw) else bool(is_conf_raw)

    if configured and not dry_run:
        try:
            return PositionTracker(client_or_adapter).get_portfolio_snapshot()
        except Exception as exc:
            typer.secho(f"  [WARN] Could not fetch real portfolio: {exc}", fg=typer.colors.YELLOW)
    return _get_simulated_portfolio()


def _check_market_open(client_or_adapter) -> bool:
    """
    Return True if market appears open, or True if we cannot verify.

    Accepts either AlpacaClientWrapper or BrokerAdapter via duck typing.
    """
    is_conf_raw = client_or_adapter.is_configured
    configured  = is_conf_raw() if callable(is_conf_raw) else bool(is_conf_raw)

    if not configured:
        return True   # Assume open for dry-run without credentials
    try:
        return client_or_adapter.is_market_open()
    except Exception:
        return False  # Fail safe


def _run_paper_iteration(
    tickers:             List[str],
    data_by_ticker:      Dict,
    portfolio,
    paper_state:         Dict,
    orchestrator,
    risk_manager,
    executor,
    market_open:         bool = True,
    ignore_market_hours: bool = False,
    dry_run:             bool = True,
) -> List[Dict]:
    """
    Execute one full paper-trading scan cycle.

    For each ticker:
      - generate signal
      - evaluate with risk manager
      - execute via order executor (or block if market closed)
      - return a list of decision dicts (no secrets included)

    This function is extracted for unit-testability: inject mocks for
    orchestrator, risk_manager, and executor to test without broker calls.
    """
    decisions: List[Dict] = []

    for ticker in tickers:
        if ticker not in data_by_ticker:
            continue

        df = data_by_ticker[ticker]

        # -- Signal --
        try:
            from strategy.signals import SignalAction
            signal = orchestrator.generate_signal(ticker, df)
        except Exception as exc:
            decisions.append({
                "timestamp": datetime.now().isoformat(),
                "ticker":    ticker,
                "error":     str(exc),
                "signal_action": "ERROR",
            })
            continue

        # -- Risk evaluation --
        pos = portfolio.positions.get(ticker)
        has_pos  = pos is not None
        cur_qty  = float(pos.quantity) if pos else 0.0

        try:
            risk_decision = risk_manager.evaluate_signal(
                signal                     = signal,
                equity                     = portfolio.equity,
                starting_day_equity        = float(paper_state.get("starting_day_equity", portfolio.equity)),
                starting_week_equity       = float(paper_state.get("starting_week_equity", portfolio.equity)),
                peak_equity                = float(paper_state.get("peak_equity",          portfolio.equity)),
                current_portfolio_exposure = portfolio.portfolio_exposure,
                current_ticker_exposure    = portfolio.ticker_exposures.get(ticker, 0.0),
                open_positions_count       = portfolio.open_positions_count,
                trades_today               = int(paper_state.get("trades_today", 0)),
                has_existing_position      = has_pos,
            )
        except Exception as exc:
            decisions.append({
                "timestamp":     datetime.now().isoformat(),
                "ticker":        ticker,
                "error":         f"Risk evaluation failed: {exc}",
                "signal_action": signal.action.value,
            })
            continue

        # -- Market hours check --
        market_blocked = (not ignore_market_hours) and (not market_open)

        if market_blocked:
            order_submitted = False
            order_action    = risk_decision.final_action.value
            order_shares    = 0.0
            order_dry       = True
            order_id        = None
            order_reason    = "Market is closed. No orders submitted."
        else:
            try:
                order = executor.execute_risk_decision(
                    signal           = signal,
                    risk_decision    = risk_decision,
                    current_quantity = cur_qty,
                )
                order_submitted = order.submitted
                order_action    = order.action
                order_shares    = order.quantity
                order_dry       = order.dry_run
                order_id        = order.order_id
                order_reason    = order.reason
            except Exception as exc:
                order_submitted = False
                order_action    = signal.action.value
                order_shares    = 0.0
                order_dry       = True
                order_id        = None
                order_reason    = f"Order execution error: {exc}"

        decisions.append({
            "timestamp":       datetime.now().isoformat(),
            "ticker":          ticker,
            "signal_action":   signal.action.value,
            "risk_approved":   risk_decision.approved,
            "final_action":    risk_decision.final_action.value,
            "order_submitted": order_submitted,
            "order_action":    order_action,
            "shares":          order_shares,
            "dry_run":         order_dry,
            "order_id":        order_id,
            "reason":          order_reason,
            "regime":          signal.regime.value,
            "confidence":      signal.confidence,
            "entry_price":     signal.entry_price,
            "stop_loss":       signal.stop_loss or 0.0,
            "take_profit":     signal.take_profit or 0.0,
            "market_open":     market_open,
            "portfolio_equity":portfolio.equity,
            # Explicitly NO api_key, api_secret, or any credential
        })

    return decisions


def _write_decisions_jsonl(
    decisions: List[Dict],
    log_file:  str = _PAPER_DECISIONS_LOG,
) -> None:
    """Append decision records to a JSONL file. Never writes secrets."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for d in decisions:
            f.write(json.dumps(d, default=str) + "\n")


def _print_iteration_summary(decisions: List[Dict]) -> None:
    """Print a concise table of this iteration's decisions."""
    if not decisions:
        typer.echo("  (no decisions)")
        return

    typer.echo(
        f"  {'Ticker':<8} {'Signal':<8} {'Risk':<6} {'Final':<8} "
        f"{'Shares':>8}  Mode     Reason"
    )
    typer.echo("  " + "-" * 75)
    for d in decisions:
        if "error" in d:
            typer.secho(f"  {d['ticker']:<8} ERROR   {d['error'][:50]}", fg=typer.colors.RED)
            continue
        mode   = "DRY_RUN" if d.get("dry_run") else "PAPER"
        shares = f"{d.get('shares', 0):.2f}" if d.get("shares", 0) > 0 else "-"
        risk   = "OK" if d.get("risk_approved") else "NO"
        reason = d.get("reason", "")[:40]
        typer.echo(
            f"  {d.get('ticker',''):<8} "
            f"{d.get('signal_action',''):<8} "
            f"{risk:<6} "
            f"{d.get('final_action',''):<8} "
            f"{shares:>8}  {mode:<8} {reason}"
        )


def _load_data_for_paper(
    ticker_list:   List[str],
    lookback_days: int,
    synthetic:     bool,
    csv_file:      Optional[str],
) -> Dict:
    """
    Fetch OHLCV data for paper-once/paper-loop via a DataProvider.

    Routes to the appropriate DataProvider based on flags:
        synthetic=True -> SyntheticDataProvider
        csv_file set   -> CSVDataProvider
        default        -> YFinanceDataProvider
    """
    import pandas as pd
    from data_providers.base import DataProviderError

    provider = _get_data_provider(synthetic, csv_file)

    if synthetic:
        typer.secho(
            f"  Using {provider.provider_name} (offline mode).", fg=typer.colors.YELLOW
        )
        return provider.get_multiple_bars(ticker_list)

    if csv_file:
        typer.echo(f"  Loading from {provider.provider_name}...")
        try:
            # CSV provider ignores ticker — one file, one dataset
            from pathlib import Path as _P
            label = ticker_list[0] if ticker_list else _P(csv_file).stem.upper()
            df    = provider.get_bars(label)
            return {label: df}
        except DataProviderError as exc:
            typer.secho(f"  CSV load failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(1)

    # yfinance path
    end   = str(pd.Timestamp.now().date())
    start = str((pd.Timestamp.now() - pd.Timedelta(days=lookback_days)).date())

    data: Dict = {}
    for t in ticker_list:
        try:
            data[t] = provider.get_bars(t, start=start, end=end)
            typer.echo(f"  {t}: {len(data[t])} bars loaded")
        except DataProviderError as exc:
            typer.secho(f"  {t}: {exc}", fg=typer.colors.YELLOW)

    if not data:
        typer.secho(
            "\n  No data loaded. Try --synthetic or --csv.\n"
            "  Upgrade yfinance:  py -3.11 -m pip install --upgrade yfinance",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    return data


# ---------------------------------------------------------------------------
# Phase 8: paper-once (fully implemented)
# ---------------------------------------------------------------------------

@app.command()
def paper_once(
    ticker:              str  = typer.Option("SPY",   "--ticker",              help="Ticker(s), comma-separated: 'SPY,QQQ'"),
    synthetic:           bool = typer.Option(False,   "--synthetic",            help="Use synthetic data (no network needed)"),
    csv_file:            str  = typer.Option(None,    "--csv",                  help="Load OHLCV from a CSV file"),
    execute_paper:       bool = typer.Option(False,   "--execute-paper/--dry-run", help="Submit real paper orders (default: dry-run simulation)"),
    ignore_market_hours: bool = typer.Option(False,   "--ignore-market-hours",  help="Run even when market is closed"),
    lookback_days:       int  = typer.Option(400,     "--lookback-days",        help="Days of history to fetch"),
) -> None:
    """
    Run one full signal-to-decision cycle for each ticker.

    Default mode is dry-run (no orders placed).
    Legacy execution is disabled. Use paper-session for managed Alpaca paper execution.

    WARNING: Paper trading results do not guarantee live trading results.
    """
    from broker.alpaca_client import AlpacaClientWrapper, AlpacaSafetyError
    from broker.order_executor import OrderExecutor
    from risk.risk_manager import RiskManager
    from strategy.strategy_orchestrator import StrategyOrchestrator

    dry_run     = not execute_paper
    if execute_paper:
        typer.echo("Legacy paper execution is disabled. Use paper-session with explicit paper activation and paper credentials.")
        raise typer.Exit(1)
    ticker_list = _parse_tickers([ticker])

    typer.secho("\n=== paper-once ===", bold=True)
    typer.echo(f"  Tickers: {', '.join(ticker_list)}")
    typer.echo(f"  Mode:    {'DRY RUN (no orders placed)' if dry_run else 'EXECUTE PAPER (real Alpaca paper orders)'}")

    # -- Safety: --execute-paper requires credentials --
    # AlpacaClientWrapper is still used by OrderExecutor (backward compat).
    alpaca_client = AlpacaClientWrapper(dry_run=dry_run)
    if not dry_run:
        if not alpaca_client.is_configured():
            typer.secho(
                "\nCannot use --execute-paper without Alpaca credentials.\n"
                "Copy .env.example to .env and fill in paper keys.\n"
                "NEVER paste keys into Claude or any chat tool.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
        try:
            alpaca_client.validate_paper_safety()
        except AlpacaSafetyError as exc:
            typer.secho(f"\nSafety check failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(1)

    # -- Broker adapter for portfolio tracking (Phase 11) --
    from broker.simulated_adapter import SimulatedBrokerAdapter
    from broker.alpaca_adapter import AlpacaBrokerAdapter
    broker = SimulatedBrokerAdapter() if dry_run else AlpacaBrokerAdapter()

    # -- Load data --
    typer.echo("\nLoading data...")
    data_by_ticker = _load_data_for_paper(ticker_list, lookback_days, synthetic, csv_file)

    # -- Portfolio state (uses BrokerAdapter) --
    portfolio  = _get_portfolio(broker, dry_run=dry_run)
    if portfolio.metadata.get("simulated"):
        typer.secho("  Portfolio: simulated ($100,000 equity, no positions)", fg=typer.colors.YELLOW)

    # -- Paper state --
    paper_state = _load_paper_state()
    today       = datetime.now().date()
    iso_week    = today.isocalendar()[1]
    paper_state = _update_paper_state(paper_state, portfolio.equity, str(today), iso_week)

    # -- Market hours --
    market_open = _check_market_open(broker)
    if not market_open and not ignore_market_hours:
        typer.secho("  Market is closed. No orders will be submitted.", fg=typer.colors.YELLOW)
        if not dry_run:
            typer.echo("  Pass --ignore-market-hours to override.")

    # -- Execute one cycle --
    orchestrator  = StrategyOrchestrator()
    risk_manager  = RiskManager()
    executor      = OrderExecutor(broker_adapter=broker, dry_run=dry_run)

    decisions = _run_paper_iteration(
        tickers             = ticker_list,
        data_by_ticker      = data_by_ticker,
        portfolio           = portfolio,
        paper_state         = paper_state,
        orchestrator        = orchestrator,
        risk_manager        = risk_manager,
        executor            = executor,
        market_open         = market_open,
        ignore_market_hours = ignore_market_hours,
        dry_run             = dry_run,
    )

    # -- Persist --
    trades_this_iter = sum(
        1 for d in decisions
        if d.get("order_submitted") and d.get("order_action") != "HOLD"
    )
    paper_state = _update_paper_state(
        paper_state, portfolio.equity, str(today), iso_week, trades_delta=trades_this_iter
    )
    _save_paper_state(paper_state)
    _write_decisions_jsonl(decisions)

    # -- Summary --
    typer.echo("\n  Results:")
    _print_iteration_summary(decisions)
    typer.echo(f"\n  Decisions logged to: {_PAPER_DECISIONS_LOG}")
    typer.secho(
        "\n  NOTE: Paper trading results do not guarantee live trading results.",
        fg=typer.colors.YELLOW,
    )


# ---------------------------------------------------------------------------
# Phase 8: paper-loop (fully implemented)
# ---------------------------------------------------------------------------

@app.command()
def paper_loop(
    ticker:              str           = typer.Option("SPY",   "--ticker",               help="Ticker(s), comma-separated"),
    interval_seconds:    int           = typer.Option(300,     "--interval-seconds",      help="Seconds between iterations"),
    max_iterations:      Optional[int] = typer.Option(None,    "--max-iterations",        help="Stop after N iterations (omit for indefinite)"),
    execute_paper:       bool          = typer.Option(False,   "--execute-paper/--dry-run", help="Submit real paper orders (default: dry-run)"),
    ignore_market_hours: bool          = typer.Option(False,   "--ignore-market-hours",   help="Run even when market is closed"),
    synthetic:           bool          = typer.Option(False,   "--synthetic",             help="Use synthetic data (requires --max-iterations)"),
    lookback_days:       int           = typer.Option(400,     "--lookback-days",         help="Days of history per iteration"),
) -> None:
    """
    Repeatedly run paper-once on a fixed interval.

    Legacy simulation loop. Use paper-session for managed Alpaca paper execution.
    Stop with Ctrl+C.

    WARNING: Still paper trading only. No live orders.
    """
    from broker.alpaca_client import AlpacaClientWrapper, AlpacaSafetyError
    from broker.order_executor import OrderExecutor
    from risk.risk_manager import RiskManager
    from strategy.strategy_orchestrator import StrategyOrchestrator

    dry_run     = not execute_paper
    if execute_paper:
        typer.echo("Legacy paper execution is disabled. Use paper-session with explicit paper activation and paper credentials.")
        raise typer.Exit(1)
    ticker_list = _parse_tickers([ticker])

    # -- Synthetic requires max_iterations to prevent infinite loops --
    if synthetic and max_iterations is None:
        typer.secho(
            "--synthetic requires --max-iterations to prevent an infinite loop "
            "on meaningless data.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    typer.secho("\n=== paper-loop ===", bold=True)
    typer.echo(f"  Tickers:  {', '.join(ticker_list)}")
    typer.echo(f"  Mode:     {'DRY RUN' if dry_run else 'EXECUTE PAPER'}")
    typer.echo(f"  Interval: {interval_seconds}s")
    if max_iterations:
        typer.echo(f"  Max iters:{max_iterations}")
    typer.echo("  Press Ctrl+C to stop.\n")

    # -- Safety: --execute-paper requires credentials --
    # AlpacaClientWrapper is still used by OrderExecutor (backward compat).
    alpaca_client = AlpacaClientWrapper(dry_run=dry_run)
    if not dry_run:
        if not alpaca_client.is_configured():
            typer.secho(
                "Cannot use --execute-paper without Alpaca credentials.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
        try:
            alpaca_client.validate_paper_safety()
        except AlpacaSafetyError as exc:
            typer.secho(f"Safety check failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(1)

    # -- Broker adapter for portfolio tracking (Phase 11) --
    from broker.simulated_adapter import SimulatedBrokerAdapter
    from broker.alpaca_adapter import AlpacaBrokerAdapter
    broker = SimulatedBrokerAdapter() if dry_run else AlpacaBrokerAdapter()

    orchestrator = StrategyOrchestrator()
    risk_manager = RiskManager()
    executor     = OrderExecutor(broker_adapter=broker, dry_run=dry_run)

    iteration = 0
    try:
        while True:
            iteration += 1
            max_str = f"/{max_iterations}" if max_iterations else ""
            typer.secho(f"--- Iteration {iteration}{max_str} ---", bold=True)

            # Load data fresh each iteration via DataProvider
            data_by_ticker = _load_data_for_paper(
                ticker_list, lookback_days, synthetic, csv_file=None
            )

            portfolio   = _get_portfolio(broker, dry_run=dry_run)
            paper_state = _load_paper_state()
            today       = datetime.now().date()
            iso_week    = today.isocalendar()[1]
            paper_state = _update_paper_state(paper_state, portfolio.equity, str(today), iso_week)

            market_open = _check_market_open(broker)
            if not market_open and not ignore_market_hours and not dry_run:
                typer.secho("  Market closed — skipping orders.", fg=typer.colors.YELLOW)

            decisions = _run_paper_iteration(
                tickers             = ticker_list,
                data_by_ticker      = data_by_ticker,
                portfolio           = portfolio,
                paper_state         = paper_state,
                orchestrator        = orchestrator,
                risk_manager        = risk_manager,
                executor            = executor,
                market_open         = market_open,
                ignore_market_hours = ignore_market_hours,
                dry_run             = dry_run,
            )

            trades_this_iter = sum(
                1 for d in decisions
                if d.get("order_submitted") and d.get("order_action") != "HOLD"
            )
            paper_state = _update_paper_state(
                paper_state, portfolio.equity, str(today), iso_week,
                trades_delta=trades_this_iter,
            )
            _save_paper_state(paper_state)
            _write_decisions_jsonl(decisions)
            _print_iteration_summary(decisions)

            if max_iterations and iteration >= max_iterations:
                typer.echo(f"\nReached max_iterations={max_iterations}. Stopping.")
                break

            typer.echo(f"\nSleeping {interval_seconds}s... (Ctrl+C to stop)")
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        typer.echo("\nStopped by user (Ctrl+C).")

    typer.secho(
        f"\npaper-loop finished after {iteration} iteration(s). "
        "No live orders were placed.",
        fg=typer.colors.GREEN,
    )


# ---------------------------------------------------------------------------
# Phase 9: dashboard (fully implemented)
# ---------------------------------------------------------------------------

@app.command()
def dashboard() -> None:
    """
    Launch the read-only Streamlit monitoring dashboard.

    Shows portfolio state, recent decisions, risk status, signals, logs,
    and command help.  Does NOT place orders.

    Requires streamlit: py -3.10 -m pip install streamlit plotly
    """
    import subprocess

    # Check streamlit availability
    try:
        import streamlit  # noqa: F401
    except ImportError:
        typer.secho("Streamlit is not installed.", fg=typer.colors.YELLOW)
        typer.echo("\nInstall it with:")
        typer.echo("  py -3.10 -m pip install streamlit plotly")
        typer.echo("\nNote: 32-bit Python 3.10 may not have Streamlit wheels.")
        typer.echo("Try 64-bit Python 3.10+ or Python 3.11.")
        typer.echo("\nOnce installed, launch with:")
        typer.echo("  streamlit run dashboard/app.py")
        return

    app_path = Path(__file__).parent / "dashboard" / "app.py"
    typer.echo(f"Launching dashboard: {app_path}")
    typer.echo("Open http://localhost:8501 in your browser. Ctrl+C to stop.\n")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)],
                       check=False)
    except KeyboardInterrupt:
        typer.echo("\nDashboard stopped.")


# ---------------------------------------------------------------------------
# Phase 10: health-check
# ---------------------------------------------------------------------------

@app.command(name="health-check")
def health_check() -> None:
    """
    Check project health without making network calls or requiring credentials.

    Reports: Python version/architecture, dependency availability,
    safety settings, credential status, and risk lock state.
    """
    import importlib
    import struct

    typer.secho("\n=== ai_regime_trader Health Check ===\n", bold=True)

    # -- Python environment --
    bits = struct.calcsize("P") * 8
    typer.echo(f"  Python:  {sys.version.split()[0]}  ({bits}-bit)")
    if bits == 32:
        typer.secho(
            "  [WARN] 32-bit Python: alpaca-py and streamlit require 64-bit.",
            fg=typer.colors.YELLOW,
        )

    # -- Dependencies --
    typer.echo("\n  Dependencies:")
    deps = [
        ("pandas",    True,  "required — data manipulation"),
        ("numpy",     True,  "required — numerics"),
        ("yfinance",  False, "recommended — market data"),
        ("hmmlearn",  False, "optional — HMM regime detection"),
        ("sklearn",   False, "optional — needed by hmmlearn"),
        ("alpaca",    False, "optional — Alpaca paper broker"),
        ("streamlit", False, "optional — monitoring dashboard"),
        ("loguru",    False, "recommended — enhanced logging"),
    ]
    for mod, required, desc in deps:
        try:
            m_obj = importlib.import_module(mod)
            ver   = getattr(m_obj, "__version__", "?")
            typer.secho(f"    [OK]      {mod:<12} {ver:<10}  {desc}", fg=typer.colors.GREEN)
        except ImportError:
            color  = typer.colors.RED if required else typer.colors.YELLOW
            marker = "[MISSING]" if required else "[OPTIONAL]"
            typer.secho(f"    {marker:<10} {mod:<12} {'?':<10}  {desc}", fg=color)

    # -- Safety settings --
    typer.echo("\n  Safety settings:")
    from config.settings import settings
    for name, val, safe_val in [
        ("PAPER_TRADING",        settings.PAPER_TRADING,        True),
        ("LIVE_TRADING_ALLOWED", settings.LIVE_TRADING_ALLOWED, False),
        ("ALPACA_PAPER",         settings.ALPACA_PAPER,         True),
    ]:
        ok    = (val == safe_val)
        color = typer.colors.GREEN if ok else typer.colors.RED
        tag   = "OK  " if ok else "WARN"
        typer.secho(f"    [{tag}] {name} = {val}", fg=color)

    # -- Credentials (existence only — never prints values) --
    typer.echo("\n  Credentials:")
    if settings.has_broker_credentials():
        typer.secho("    [OK]  Alpaca paper credentials are set (values hidden)", fg=typer.colors.GREEN)
    else:
        typer.echo("    [INFO] Alpaca credentials not set — OK for backtesting/synthetic mode")

    if Path(".env").exists():
        typer.secho("    [OK]  .env file exists", fg=typer.colors.GREEN)
    else:
        typer.echo("    [INFO] .env not found — copy .env.example to .env to add paper keys")

    # -- Risk lock --
    typer.echo("\n  Risk lock:")
    if settings.is_risk_locked():
        typer.secho(
            f"    [WARN] Risk lock active at {settings.RISK_LOCK_FILE}\n"
            "           Run: py -3.10 main.py unlock-risk",
            fg=typer.colors.RED,
        )
    else:
        typer.secho("    [OK]  No risk lock detected", fg=typer.colors.GREEN)

    # -- State files --
    typer.echo("\n  State files:")
    for label, path_str in [
        ("paper_state.json",         "state/paper_state.json"),
        ("paper_decisions.jsonl",    "logs/paper_decisions.jsonl"),
    ]:
        exists = Path(path_str).exists()
        color  = typer.colors.GREEN if exists else typer.colors.YELLOW
        typer.secho(f"    {'[OK]' if exists else '[INFO]'}  {label}", fg=color)

    # -- Test hint --
    typer.echo("\n  Tests:")
    typer.echo("    py -3.10 -m pytest -q --tb=short  (expected: 260+ passing)")

    typer.secho("\nHealth check complete. No network calls were made.", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------
# Phase 10: export-status
# ---------------------------------------------------------------------------

@app.command(name="export-status")
def export_status() -> None:
    """
    Export a full system status snapshot to results/status/status_TIMESTAMP.json.

    Includes: safety settings, Python environment, risk lock state,
    paper state, recent decision summary, and latest backtest/walk-forward
    summaries (if available).  Never includes API keys or secrets.
    """
    import struct
    from dashboard.helpers import load_jsonl, load_latest_result, summarize_decisions

    typer.echo("Collecting status...")

    bits = struct.calcsize("P") * 8
    from config.settings import settings

    status: Dict = {
        "timestamp": datetime.now().isoformat(),
        "python": {
            "version": sys.version.split()[0],
            "bits":    bits,
        },
        "safety_settings": {
            "PAPER_TRADING":        settings.PAPER_TRADING,
            "LIVE_TRADING_ALLOWED": settings.LIVE_TRADING_ALLOWED,
            "ALPACA_PAPER":         settings.ALPACA_PAPER,
            "ALPACA_BASE_URL":      settings.ALPACA_BASE_URL,
            "MAX_RISK_PER_TRADE":   settings.MAX_RISK_PER_TRADE,
            "MAX_PORTFOLIO_EXPOSURE": settings.MAX_PORTFOLIO_EXPOSURE,
            "MAX_TOTAL_DRAWDOWN_LOCK": settings.MAX_TOTAL_DRAWDOWN_LOCK,
            # API_KEY and SECRET_KEY deliberately excluded
        },
        "credentials_configured": settings.has_broker_credentials(),
        "risk_lock": {
            "exists": settings.is_risk_locked(),
            "path":   settings.RISK_LOCK_FILE,
        },
        "paper_state": _load_paper_state(),
        "recent_decisions": summarize_decisions(load_jsonl(_PAPER_DECISIONS_LOG)),
        "latest_backtest":   load_latest_result(f"{_RESULTS_BASE}/backtests"),
        "latest_walk_forward": load_latest_result(f"{_RESULTS_BASE}/walk_forward"),
    }

    saved = _save_result_json(status, "status", "status")
    typer.secho(f"\nStatus exported to: {saved}", fg=typer.colors.GREEN)
    typer.echo("No API keys or secrets were included.")


# ---------------------------------------------------------------------------
# unlock-risk (fully implemented — shows info, does NOT auto-delete)
# ---------------------------------------------------------------------------

@app.command()
def unlock_risk() -> None:
    """
    Show risk lock status and explain how to manually remove the lock.

    This command does NOT delete the lock file automatically.
    You must review logs and delete it yourself.
    """
    from config.settings import settings

    lock_path = settings.risk_lock_path().resolve()

    typer.secho("\n=== RISK LOCK STATUS ===\n", bold=True)
    typer.echo(f"Lock file path: {lock_path}\n")

    if lock_path.exists():
        typer.secho("STATUS: LOCKED", fg=typer.colors.RED, bold=True)
        typer.echo(f"\nLock file contents:\n{'-' * 50}")
        typer.echo(lock_path.read_text())
        typer.echo(f"{'-' * 50}")

        typer.secho("\nThis lock was created because a drawdown circuit breaker triggered.", fg=typer.colors.YELLOW)
        typer.secho("\nBefore unlocking, you MUST:", bold=True)
        typer.echo("  1. Review all log files in logs/")
        typer.echo("  2. Check all open positions in your Alpaca paper account")
        typer.echo("  3. Understand exactly why the circuit breaker triggered")
        typer.echo("  4. Decide if you need to adjust risk settings")
        typer.echo("  5. Accept that the same conditions could occur again\n")

        typer.secho("To unlock, MANUALLY delete the lock file:", bold=True)
        typer.echo(f"  Windows:   del \"{lock_path}\"")
        typer.echo(f"  PowerShell: Remove-Item '{lock_path}'")
        typer.echo(f"  Mac/Linux:  rm '{lock_path}'\n")

        typer.secho("WARNING: Never delete this file without reviewing the above steps.", fg=typer.colors.RED)
    else:
        typer.secho("STATUS: NOT LOCKED", fg=typer.colors.GREEN)
        typer.echo("The bot is not currently risk-locked and can trade normally.")


@app.command(name="list-watchlists")
def show_watchlists() -> None:
    """Show all available watchlists and their tickers."""
    watchlist_names = get_available_watchlists()

    if not watchlist_names:
        typer.secho("No watchlists found in watchlists/ directory.", fg=typer.colors.YELLOW)
        return

    typer.secho("\n=== Available Watchlists ===\n", bold=True)

    for name in watchlist_names:
        try:
            tickers = load_watchlist(name)
            typer.secho(f"{name}:", bold=True)
            typer.echo(f"  Tickers: {', '.join(tickers)}")
            typer.echo()
        except typer.Exit:
            # Skip watchlists with errors
            pass


# ---------------------------------------------------------------------------
# Intraday lookback validation (yfinance limits)
# ---------------------------------------------------------------------------

def _validate_intraday_lookback(
    timeframe: str,
    lookback_days: int,
    synthetic: bool,
    csv_file: Optional[str],
    user_explicit_lookback: bool,
) -> int:
    """
    Validate and adjust lookback days for intraday timeframes.

    yfinance has strict intraday history limits:
      - 1Min: ~7 days only
      - 5Min: ~60 days (default 30 for safety)
      - 15Min: ~60 days (default 30 for safety)
      - 1Day: historical data (no limit)

    If using yfinance (not synthetic/csv) with intraday data, apply limits.
    If user did NOT explicitly pass --lookback-days, use safe defaults.
    If user DID pass too-large lookback for yfinance, error before fetching.

    Returns: adjusted lookback_days (or raises Exit if invalid)
    """
    # For synthetic or CSV, no restrictions apply
    if synthetic or csv_file:
        return lookback_days

    # Determine safe defaults by timeframe (yfinance only)
    defaults = {
        "1Day":  400,
        "15Min": 30,
        "5Min":  30,
        "1Min":  7,
    }

    # Limits for yfinance intraday data
    limits = {
        "1Day":  None,  # No limit
        "15Min": 60,
        "5Min":  60,
        "1Min":  7,
    }

    default_for_tf = defaults.get(timeframe, 400)
    limit_for_tf = limits.get(timeframe)

    # If user didn't explicitly pass --lookback-days, use the safe default
    if not user_explicit_lookback:
        return default_for_tf

    # User explicitly passed --lookback-days; validate it for yfinance
    if limit_for_tf is not None and lookback_days > limit_for_tf:
        typer.secho(
            f"Error: Yahoo/yfinance intraday limit: {timeframe} bars are only available for "
            f"the last {limit_for_tf} days, but you requested {lookback_days} days.\n",
            fg=typer.colors.RED,
        )
        typer.echo("Solutions:")
        typer.echo(f"  1. Reduce --lookback-days to {limit_for_tf} or less")
        typer.echo("  2. Use --synthetic for offline smoke testing")
        typer.echo("  3. Use --csv for your own data file")
        raise typer.Exit(1)

    return lookback_days


def _normalize_setup_blocker(reason: str) -> str:
    """Convert verbose setup diagnostic text into a concise market blocker."""
    text = str(reason or "").lower()
    if "not above vwap" in text or "close_above_vwap" in text:
        return "close below VWAP"
    if "rel_volume_20" in text:
        return "relative volume too low"
    if "trend_score" in text:
        return "trend not positive"
    if "broke_above_opening_range" in text:
        return "close not above opening range"
    if "rsi_14" in text:
        return "RSI overextended or unavailable"
    if "prev_close" in text and "prev_vwap" in text:
        return "VWAP reclaim not confirmed"
    return str(reason or "unknown blocker")


def summarize_setup_market_bias(setups: list) -> Dict:
    """
    Summarize market bias when setup scanning finds no valid ranked setups.

    This is display-only and intentionally does not alter detector or ranking
    decisions.
    """
    from collections import Counter
    import math

    total = len(setups or [])
    above_vwap = 0
    below_vwap = 0
    blockers = []

    for setup in setups or []:
        metadata = getattr(setup, "metadata", {}) or {}
        close = metadata.get("close")
        vwap = metadata.get("vwap_value", metadata.get("vwap"))
        close_above = metadata.get("close_above_vwap")

        try:
            close_float = float(close)
            vwap_float = float(vwap)
            if not math.isnan(close_float) and not math.isnan(vwap_float):
                if close_float > vwap_float:
                    above_vwap += 1
                else:
                    below_vwap += 1
            elif float(close_above) == 1.0:
                above_vwap += 1
            elif float(close_above) == 0.0:
                below_vwap += 1
        except Exception:
            diagnostics_text = " ".join(
                str(item)
                for failures in metadata.get("diagnostics", {}).values()
                for item in failures
            ).lower()
            if "not above vwap" in diagnostics_text or "close_above_vwap" in diagnostics_text:
                below_vwap += 1

        diagnostics = metadata.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            for failures in diagnostics.values():
                for failure in failures or []:
                    blockers.append(_normalize_setup_blocker(failure))

    percent_above = (above_vwap / total * 100.0) if total else 0.0
    most_common = Counter(blockers).most_common(1)
    most_common_blocker = most_common[0][0] if most_common else "not enough diagnostic data"

    if total and percent_above >= 65.0:
        long_bias = "favorable"
    elif total and percent_above >= 35.0:
        long_bias = "mixed"
    else:
        long_bias = "unfavorable"

    return {
        "tickers_scanned": total,
        "above_vwap": above_vwap,
        "below_vwap": below_vwap,
        "percent_above_vwap": percent_above,
        "most_common_blocker": most_common_blocker,
        "long_bias": long_bias,
    }


def print_setup_market_bias_summary(setups: list, indent: str = "") -> None:
    """Print a concise no-trade market-bias summary."""
    summary = summarize_setup_market_bias(setups)
    typer.secho(f"{indent}Market Bias Summary:", bold=True)
    typer.echo(f"{indent}  Tickers scanned: {summary['tickers_scanned']}")
    typer.echo(f"{indent}  Above VWAP: {summary['above_vwap']}")
    typer.echo(f"{indent}  Below VWAP: {summary['below_vwap']}")
    typer.echo(f"{indent}  Percent above VWAP: {summary['percent_above_vwap']:.0f}%")
    typer.echo(f"{indent}  Long bias: {summary['long_bias']}")
    typer.echo(f"{indent}  Most common blocker: {summary['most_common_blocker']}")


def _print_stale_data_review_banner(indent: str = "") -> None:
    """Print a strong review-only warning for stale market data."""
    typer.secho(
        f"{indent}STALE DATA WARNING \u2014 results are for review only, not live trade entry.",
        fg=typer.colors.YELLOW,
        bold=True,
    )


SETUP_TYPE_COOLDOWN_FAILED_THRESHOLD = 2
TICKER_COOLDOWN_FAILED_THRESHOLD = 2


def _daily_setup_cooldown_summary(review_date=None) -> dict:
    """Summarize same-day failed/excluded manual trades by setup type and ticker."""
    from collections import Counter
    from trades.trade_tracker import TradeStatus, TradeTracker

    target_date = review_date or datetime.now().date()
    included_by_setup = Counter()
    included_by_ticker = Counter()
    excluded_by_setup = Counter()
    excluded_by_ticker = Counter()
    failed_by_setup = Counter()
    failed_by_ticker = Counter()

    for trade in TradeTracker().list_all_trades():
        entry_dt = _parse_review_datetime(getattr(trade, "entry_time", None))
        exit_dt = _parse_review_datetime(getattr(trade, "exit_time", None))
        is_today = (
            (entry_dt is not None and entry_dt.date() == target_date)
            or (exit_dt is not None and exit_dt.date() == target_date)
        )
        if not is_today:
            continue

        setup_type = str(getattr(trade, "setup_type", "") or "").strip()
        ticker = str(getattr(trade, "ticker", "") or "").upper().strip()
        if not setup_type and not ticker:
            continue

        is_excluded_or_mistake = bool(getattr(trade, "exclude_from_analytics", False)) or bool(getattr(trade, "mistake", False))
        if is_excluded_or_mistake:
            if setup_type:
                excluded_by_setup[setup_type] += 1
            if ticker:
                excluded_by_ticker[ticker] += 1
        else:
            if setup_type:
                included_by_setup[setup_type] += 1
            if ticker:
                included_by_ticker[ticker] += 1

        is_loss = (
            str(getattr(trade, "status", "")).upper() == TradeStatus.CLOSED.value
            and not bool(getattr(trade, "exclude_from_analytics", False))
            and float(getattr(trade, "realized_pnl", 0.0) or 0.0) < 0
        )
        if is_loss or is_excluded_or_mistake:
            if setup_type:
                failed_by_setup[setup_type] += 1
            if ticker:
                failed_by_ticker[ticker] += 1

    cooled_setup_types = {
        setup: count
        for setup, count in failed_by_setup.items()
        if count >= SETUP_TYPE_COOLDOWN_FAILED_THRESHOLD
    }
    cooled_tickers = {
        ticker: count
        for ticker, count in failed_by_ticker.items()
        if count >= TICKER_COOLDOWN_FAILED_THRESHOLD
    }
    return {
        "included_by_setup_type": dict(included_by_setup),
        "included_by_ticker": dict(included_by_ticker),
        "excluded_by_setup_type": dict(excluded_by_setup),
        "excluded_by_ticker": dict(excluded_by_ticker),
        "failed_by_setup_type": dict(failed_by_setup),
        "failed_by_ticker": dict(failed_by_ticker),
        "cooled_setup_types": cooled_setup_types,
        "cooled_tickers": cooled_tickers,
    }


def _setup_cooldown_reasons(setup, cooldown_summary: dict) -> list[str]:
    """Return review-only cooldown reasons matching a detected setup."""
    setup_type = getattr(getattr(setup, "setup_type", None), "value", getattr(setup, "setup_type", ""))
    ticker = str(getattr(setup, "ticker", "") or "").upper()
    reasons = []
    setup_count = cooldown_summary.get("cooled_setup_types", {}).get(setup_type)
    ticker_count = cooldown_summary.get("cooled_tickers", {}).get(ticker)
    if setup_count:
        reasons.append(f"{setup_type} has {setup_count}+ failed/excluded trades today.")
    if ticker_count:
        reasons.append(f"{ticker} has {ticker_count}+ failed/excluded trades today.")
    return reasons


def _print_setup_cooldown_warning(reasons: list[str], indent: str = "") -> None:
    """Print a clear cooldown warning for a setup alert."""
    if not reasons:
        return
    typer.secho(f"{indent}COOLDOWN WARNING:", fg=typer.colors.YELLOW, bold=True)
    for reason in reasons:
        typer.echo(f"{indent}{reason}")
    typer.echo(f"{indent}Treat this alert as review-only unless intentionally overriding.")


def _warn_if_provider_data_stale(data_by_ticker: Dict, provider_name: str, timeframe: str, indent: str = "") -> bool:
    """Print a concise stale-data warning when provider metadata exposes age."""
    if provider_name != "AlpacaData":
        return False

    timeframe_minutes = {"1Min": 1, "5Min": 5}.get(timeframe, 1)
    stale_threshold_seconds = max(180, timeframe_minutes * 3 * 60)
    for df in (data_by_ticker or {}).values():
        data_age_seconds = getattr(df, "attrs", {}).get("data_age_seconds")
        if data_age_seconds is not None and float(data_age_seconds) > stale_threshold_seconds:
            age_minutes = float(data_age_seconds) / 60.0
            typer.secho(
                f"{indent}Warning: Alpaca latest bar appears stale: {age_minutes:.1f} minutes old",
                fg=typer.colors.YELLOW,
            )
            return True

    return False


def _provider_max_data_age_seconds(data_by_ticker: Dict) -> float:
    """Return the maximum provider-reported bar age without logging data or secrets."""
    ages = []
    for df in (data_by_ticker or {}).values():
        value = getattr(df, "attrs", {}).get("data_age_seconds")
        if value is None:
            continue
        try:
            ages.append(max(0.0, float(value)))
        except (TypeError, ValueError):
            continue
    return max(ages or [0.0])


# ---------------------------------------------------------------------------
# setup-scan (Phase 12: Day-trade setup detection)
# ---------------------------------------------------------------------------

@app.command()
def setup_scan(
    ticker:       str  = typer.Option("SPY",   help="Ticker(s): use --ticker multiple times or comma-separated (e.g., --ticker 'SPY,QQQ,NVDA')"),
    watchlist:    str  = typer.Option(None, "--watchlist", help="Load tickers from a watchlist (e.g., mega_caps, momentum_universe, or path/to/file.json)"),
    synthetic:    bool = typer.Option(False, "--synthetic", help="Use synthetic data (offline, no network required)"),
    csv_file:     str  = typer.Option(None,  "--csv",      help="Load OHLCV from a CSV file instead of yfinance"),
    provider_name: str = typer.Option("yfinance", "--provider", help="Market data provider: yfinance, alpaca, synthetic, csv"),
    alpaca_feed: str = typer.Option("iex", "--alpaca-feed", help="Alpaca market data feed: iex or sip (default iex/free feed)"),
    lookback_days: Optional[int] = typer.Option(None, "--lookback-days", help="Days of history (default: 400 for daily, 30 for 5/15Min, 7 for 1Min). yfinance intraday limits apply."),
    timeframe:    Optional[str]  = typer.Option(None, "--timeframe", show_default=False, help="Bar size: 1Day (default for yfinance), 1Min, 5Min, 15Min. Alpaca defaults to 5Min."),
    full_table:   bool = typer.Option(False, "--full-table", help="Also print the full per-ticker results table (every setup, including NO_SETUP)"),
    top_n:        int  = typer.Option(5, "--top-n", help="Maximum number of ranked setups to display (default 5)"),
    min_confidence: float = typer.Option(0.6, "--min-confidence", help="Minimum confidence for a setup to appear in the ranked list (default 0.6)"),
    account_equity: Optional[float] = typer.Option(None, "--account-equity", help="Optional account equity for suggested share sizing (no broker call)"),
    risk_pct:     float = typer.Option(0.005, "--risk-pct", help="Fraction of equity to risk per trade for sizing (default 0.005 = 0.5%)"),
) -> None:
    """
    Scan for day-trade setups (watchlist detector only, no orders).

    Identifies three types of long setups:
      - Opening range breakout (close > VWAP, broke above opening range, high volume)
      - VWAP reclaim (transitioned from below to above VWAP, decent volume)
      - High relative-volume continuation (2x normal volume, strong trend)

    By default the best opportunities across all scanned tickers are ranked and
    only the top few are shown. Pass --full-table to also see the raw per-ticker
    table. Every detected setup (including NO_SETUP) is logged to
    logs/setup_scans.jsonl. This is a watchlist tool — it never places orders or
    affects your broker state.

    Examples:
        python main.py setup-scan
        python main.py setup-scan --ticker SPY --ticker QQQ
        python main.py setup-scan --ticker 'SPY,QQQ,NVDA,TSLA'
        python main.py setup-scan --ticker 'SPY,QQQ,NVDA' --full-table
        python main.py setup-scan --synthetic  (offline smoke test)
        python main.py setup-scan --csv path/to/data.csv --ticker CUSTOM
    """
    import pandas as pd
    from pathlib import Path

    from data_providers.base import DataProviderError
    from strategy.day_trade_setups import DayTradeSetupDetector, SetupType
    from strategy.setup_ranking import SetupRanker

    typer.secho("\n=== Day-Trade Setup Scan ===\n", bold=True)

    # -- Parse tickers from --ticker and/or --watchlist --
    tickers = merge_tickers(ticker if ticker != "SPY" or watchlist else None, watchlist)

    # If no tickers from CLI, default to SPY
    if not tickers:
        tickers = ["SPY"]

    selected_provider = (provider_name or "yfinance").lower()
    if synthetic:
        selected_provider = "synthetic"
    if csv_file:
        selected_provider = "csv"
    timeframe = _resolve_scan_timeframe(selected_provider, timeframe, "1Day")

    typer.echo(f"  Tickers: {', '.join(tickers)}")
    typer.echo(f"  Timeframe: {timeframe}")

    # -- Validate and adjust lookback for intraday yfinance limits --
    user_passed_lookback = lookback_days is not None
    lookback_days = _validate_intraday_lookback(
        timeframe,
        lookback_days or 400,
        selected_provider != "yfinance",
        csv_file if selected_provider == "csv" else None,
        user_passed_lookback,
    )

    # -- Get data provider --
    try:
        provider = _get_named_data_provider(
            provider_name, synthetic=synthetic, csv_file=csv_file, alpaca_feed=alpaca_feed
        )
    except typer.BadParameter as exc:
        typer.secho(f"  Provider error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)

    # -- Load data --
    typer.echo(f"  Provider: {provider.provider_name}")
    typer.echo(f"  Data provider: {_data_provider_display_name(selected_provider, provider.provider_name, alpaca_feed)}")

    data_by_ticker: Dict = {}

    if selected_provider == "synthetic":
        typer.secho("  (Using synthetic offline data; no network calls)\n", fg=typer.colors.YELLOW)
        try:
            data_by_ticker = provider.get_multiple_bars(tickers, timeframe=timeframe)
        except DataProviderError as exc:
            typer.secho(f"  Data generation failed: {exc}\n", fg=typer.colors.RED)
            raise typer.Exit(1)

    elif selected_provider == "csv":
        typer.echo(f"  Source: {csv_file}\n")
        try:
            from pathlib import Path as _Path
            label = _Path(csv_file).stem.upper()
            df = provider.get_bars(label, timeframe=timeframe)
            data_by_ticker = {label: df}
        except DataProviderError as exc:
            typer.secho(f"  CSV load failed: {exc}\n", fg=typer.colors.RED)
            raise typer.Exit(1)

    else:
        typer.echo(f"  Lookback: {lookback_days} days\n")
        end = None if selected_provider == "alpaca" else str(pd.Timestamp.now().date())
        start = str((pd.Timestamp.now() - pd.Timedelta(days=lookback_days)).date())
        try:
            data_by_ticker = provider.get_multiple_bars(
                tickers, start=start, end=end, timeframe=timeframe,
            )
        except DataProviderError as exc:
            _print_data_provider_error(exc, indent="  ")
            typer.echo("")
            if selected_provider == "alpaca":
                typer.echo("  Check Alpaca paper data credentials in .env and confirm the timeframe is supported.")
            else:
                typer.secho("Suggestions:", bold=True)
                typer.echo("  1. Retry later (Yahoo Finance may be temporarily unavailable)")
                typer.echo("  2. Use synthetic data (offline): --synthetic")
                typer.echo("  3. Use a CSV file: --csv path/to/data.csv")
            raise typer.Exit(1)

    stale_data = _warn_if_provider_data_stale(data_by_ticker, provider.provider_name, timeframe)
    if stale_data:
        _print_stale_data_review_banner()

    # -- Detect setups --
    typer.echo("Running day-trade setup detection...\n")
    detector = DayTradeSetupDetector()
    results = detector.detect_many(data_by_ticker)

    # -- Print full per-ticker table (only when requested) --
    if full_table:
        typer.secho("Setup Results:", bold=True)
        typer.echo(
            f"{'Ticker':<8} {'Setup Type':<35} {'Valid':<7} {'Conf':<6} "
            f"{'Entry':<9} {'Stop':<9} {'T1 (2R)':<9} {'T2 (3R)':<9} {'RR1':<5} {'RR2':<5} {'Reason':<50}"
        )
        typer.echo("-" * 145)

        for setup in results:
            setup_name = setup.setup_type.value if setup.setup_type else "unknown"
            valid_str = "✓ YES" if setup.is_valid else "NO"
            confidence_str = f"{setup.confidence:.0%}" if setup.confidence > 0 else "—"
            entry_str = f"${setup.entry_trigger:.2f}" if setup.entry_trigger > 0 else "—"
            stop_str = f"${setup.stop_loss:.2f}" if setup.stop_loss > 0 else "—"
            t1_str = f"${setup.target_1:.2f}" if setup.target_1 > 0 else "—"
            t2_str = f"${setup.target_2:.2f}" if setup.target_2 > 0 else "—"
            rr1_str = f"{setup.risk_reward_1:.1f}R" if setup.risk_reward_1 > 0 else "—"
            rr2_str = f"{setup.risk_reward_2:.1f}R" if setup.risk_reward_2 > 0 else "—"
            reason_trunc = (setup.reason[:47] + "...") if len(setup.reason) > 50 else setup.reason

            typer.echo(
                f"{setup.ticker:<8} {setup_name:<35} {valid_str:<7} {confidence_str:<6} "
                f"{entry_str:<9} {stop_str:<9} {t1_str:<9} {t2_str:<9} {rr1_str:<5} {rr2_str:<5} {reason_trunc:<50}"
            )
        typer.echo("")

    # -- Rank valid setups across all tickers and show the top opportunities --
    ranker = SetupRanker()
    top_setups = ranker.filter_top_setups(
        results, min_confidence=min_confidence, max_results=top_n,
    )

    if not top_setups:
        typer.echo("No valid setups found across scanned tickers.\n")
        print_setup_market_bias_summary(results)
        typer.echo("")
        # Show near-miss diagnostics for each ticker
        typer.secho("Near-Miss Diagnostics:", bold=True)
        for setup in results:
            if not setup.is_valid:
                nearest = setup.metadata.get("nearest_setup_type", "unknown")
                failed = setup.metadata.get("diagnostics", {}).get(nearest, [])
                if failed:
                    failed_str = ", ".join(failed[:2])  # Show first 2 failures
                    if len(failed) > 2:
                        failed_str += f", ... ({len(failed) - 2} more)"
                    typer.echo(f"  {setup.ticker}: Nearest: {nearest}; failed: {failed_str}")
        typer.echo("")
    else:
        from alerts.local_alerts import format_trade_action_card

        typer.secho("=== Top Trade Setups ===\n", bold=True)
        for setup in top_setups:
            typer.echo(format_trade_action_card(setup, account_equity=account_equity, risk_pct=risk_pct, stale_data=stale_data))
            typer.echo("")

    # -- Log results to JSONL (ALL setups, including NO_SETUP) --
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    log_file = logs_dir / "setup_scans.jsonl"
    timestamp = datetime.now().isoformat()

    jsonl_lines = []
    for setup in results:
        metadata = dict(setup.metadata or {})
        if stale_data:
            metadata["stale_data"] = True
        record = {
            "timestamp": timestamp,
            "ticker": setup.ticker,
            "setup_type": setup.setup_type.value,
            "is_valid": setup.is_valid,
            "confidence": float(setup.confidence),
            "entry_trigger": float(setup.entry_trigger),
            "stop_loss": float(setup.stop_loss),
            "target_1": float(setup.target_1),
            "target_2": float(setup.target_2),
            "risk_reward_1": float(setup.risk_reward_1),
            "risk_reward_2": float(setup.risk_reward_2),
            "reason": setup.reason,
            "warnings": setup.warnings,
            "metadata": metadata,
        }
        jsonl_lines.append(json.dumps(record))

    with open(log_file, "a") as f:
        for line in jsonl_lines:
            f.write(line + "\n")

    typer.secho(f"\nResults logged to: {log_file}", fg=typer.colors.GREEN)
    typer.echo("No API keys or secrets were included in the log.")


# ---------------------------------------------------------------------------
# setup-scan-loop (Phase: continuous monitoring)
# ---------------------------------------------------------------------------

def _run_setup_scan_once(
    tickers: list,
    provider,
    timeframe: str,
    lookback_days: int,
    min_confidence: float,
    top_n: int,
    synthetic: bool,
    csv_file: str,
) -> tuple:
    """
    Execute one setup-scan iteration and return (all_results, top_ranked_setups, stale_data).
    Raises DataProviderError on data load failure.
    """
    from data_providers.base import DataProviderError
    from strategy.day_trade_setups import DayTradeSetupDetector
    from strategy.setup_ranking import SetupRanker
    import pandas as pd

    data_by_ticker = {}

    if synthetic:
        data_by_ticker = provider.get_multiple_bars(tickers, timeframe=timeframe)
    elif csv_file:
        from pathlib import Path as _Path
        label = _Path(csv_file).stem.upper()
        df = provider.get_bars(label, timeframe=timeframe)
        data_by_ticker = {label: df}
    else:
        provider_name = getattr(provider, "provider_name", "")
        end = None if provider_name == "AlpacaData" else str(pd.Timestamp.now().date())
        start = str((pd.Timestamp.now() - pd.Timedelta(days=lookback_days)).date())
        data_by_ticker = provider.get_multiple_bars(
            tickers, start=start, end=end, timeframe=timeframe,
        )

    stale_data = _warn_if_provider_data_stale(data_by_ticker, getattr(provider, "provider_name", ""), timeframe, indent="  ")

    detector = DayTradeSetupDetector()
    results = detector.detect_many(data_by_ticker)

    ranker = SetupRanker()
    top_setups = ranker.filter_top_setups(
        results, min_confidence=min_confidence, max_results=top_n,
    )

    return results, top_setups, stale_data


@app.command()
def setup_scan_loop(
    ticker:       str  = typer.Option("SPY", help="Ticker(s): use --ticker multiple times or comma-separated"),
    watchlist:    str  = typer.Option(None, "--watchlist", help="Load tickers from a watchlist (e.g., mega_caps, momentum_universe, or path/to/file.json)"),
    synthetic:    bool = typer.Option(False, "--synthetic", help="Use synthetic data (offline)"),
    csv_file:     str  = typer.Option(None, "--csv", help="Load OHLCV from a CSV file"),
    provider_name: Optional[str] = typer.Option(None, "--provider", help="Market data provider: yfinance, alpaca, synthetic, csv"),
    alpaca_feed: str = typer.Option("iex", "--alpaca-feed", help="Alpaca market data feed: iex or sip (default iex/free feed)"),
    lookback_days: Optional[int] = typer.Option(None, "--lookback-days", help="Days of history (default: 400 for daily, 30 for 5/15Min, 7 for 1Min). yfinance intraday limits apply."),
    timeframe:    Optional[str]  = typer.Option(None, "--timeframe", show_default=False, help="Bar size: 1Day, 1Min, 5Min, 15Min (default 5Min for setup-scan-loop)"),
    interval_seconds: int = typer.Option(300, "--interval-seconds", help="Wait this many seconds between scans (default 300)"),
    max_iterations: int = typer.Option(None, "--max-iterations", help="Stop after this many iterations (required for --synthetic)"),
    top_n:        int  = typer.Option(5, "--top-n", help="Top N setups to display per iteration (default 5)"),
    min_confidence: float = typer.Option(0.6, "--min-confidence", help="Minimum confidence threshold (default 0.6)"),
    alert_banner: bool = typer.Option(True, "--alert-banner/--no-alert-banner", help="Print alert banner on new setup (default alert-banner)"),
    allow_stale_alerts: bool = typer.Option(False, "--allow-stale-alerts", help="Allow setup alerts even when Alpaca market data is stale"),
    suppress_cooled_down_alerts: bool = typer.Option(False, "--suppress-cooled-down-alerts", help="Suppress setup alert records and Discord pings for setup types/tickers that failed repeatedly today"),
    discord_alerts: bool = typer.Option(True, "--discord-alerts/--no-discord-alerts", help="Send Discord alerts when webhook is configured"),
    account_equity: Optional[float] = typer.Option(None, "--account-equity", help="Optional account equity for suggested share sizing (no broker call)"),
    risk_pct:     float = typer.Option(0.005, "--risk-pct", help="Fraction of equity to risk per trade for sizing (default 0.005 = 0.5%)"),
) -> None:
    """
    Continuously scan for day-trade setups at regular intervals.

    Repeatedly runs setup-scan logic, prints the latest top setups each iteration,
    and logs valid alerts to logs/setup_alerts.jsonl (deduplicating within the run).

    Safety: --synthetic mode requires --max-iterations (to prevent infinite loops).

    Examples:
        python main.py setup-scan-loop --synthetic --max-iterations 2
        python main.py setup-scan-loop --ticker SPY,QQQ --interval-seconds 60
    """
    import time
    from pathlib import Path
    from alerts.local_alerts import format_trade_action_card, print_alert_banner
    from alerts.discord_alerts import (
        format_setup_discord_alert,
        is_discord_configured,
        send_discord_message,
    )

    # -- Validate synthetic without max-iterations (safety gate) --
    selected_provider, recommended_feed, provider_warning = _recommended_intraday_provider(provider_name)
    if provider_name is None and selected_provider == "alpaca":
        alpaca_feed = recommended_feed
    if synthetic:
        selected_provider = "synthetic"
    if csv_file:
        selected_provider = "csv"
    timeframe = _resolve_scan_timeframe(selected_provider, timeframe, "5Min")

    if selected_provider == "synthetic" and max_iterations is None:
        typer.secho(
            "Error: --synthetic mode requires --max-iterations to prevent infinite loops.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    # -- Parse tickers from --ticker and/or --watchlist --
    tickers = merge_tickers(ticker if ticker != "SPY" or watchlist else None, watchlist)

    # If no tickers from CLI, default to SPY
    if not tickers:
        tickers = ["SPY"]

    # -- Get data provider --
    try:
        provider = _get_named_data_provider(
            selected_provider, synthetic=synthetic, csv_file=csv_file, alpaca_feed=alpaca_feed
        )
    except typer.BadParameter as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho("\n=== Day-Trade Setup Scan Loop ===\n", bold=True)
    typer.echo(f"  Tickers: {', '.join(tickers)}")
    typer.echo(f"  Timeframe: {timeframe}")
    typer.echo(f"  Interval: {interval_seconds}s")
    if max_iterations:
        typer.echo(f"  Max iterations: {max_iterations}")
    typer.echo(f"  Provider: {provider.provider_name}")
    typer.echo(f"  Data provider: {_data_provider_display_name(selected_provider, provider.provider_name, alpaca_feed)}")
    if provider_warning and selected_provider == "yfinance":
        typer.echo(f"  {provider_warning}")
    discord_enabled = bool(discord_alerts and is_discord_configured())
    if discord_enabled:
        typer.echo("  Discord alerts: enabled\n")
    else:
        typer.echo("  Discord alerts: not configured\n")

    # -- Validate and adjust lookback for intraday yfinance limits --
    user_passed_lookback = lookback_days is not None
    lookback_days = _validate_intraday_lookback(
        timeframe,
        lookback_days or 400,
        selected_provider != "yfinance",
        csv_file if selected_provider == "csv" else None,
        user_passed_lookback,
    )

    # -- Setup logging --
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    scan_log = logs_dir / "setup_scans.jsonl"
    alert_log = logs_dir / "setup_alerts.jsonl"

    # -- Track seen alerts within this run to avoid duplicates --
    seen_alerts = set()

    iteration = 0
    try:
        while True:
            iteration += 1
            if max_iterations and iteration > max_iterations:
                break

            timestamp_now = datetime.now().isoformat()
            typer.echo(f"\n[{timestamp_now}] Iteration {iteration}")

            try:
                scan_result = _run_setup_scan_once(
                    tickers, provider, timeframe, lookback_days,
                    min_confidence, top_n, synthetic, csv_file,
                )
                if len(scan_result) == 3:
                    all_results, top_setups, stale_data = scan_result
                else:
                    all_results, top_setups = scan_result
                    stale_data = False
            except Exception as exc:
                message = str(exc)
                typer.secho(f"  Scan failed: {message}", fg=typer.colors.RED)
                lowered = message.lower()
                if "sip" in lowered and ("subscription" in lowered or "permit" in lowered or "permission" in lowered):
                    typer.echo(
                        "  Your Alpaca plan does not permit SIP data. "
                        "Use --alpaca-feed iex or upgrade Alpaca data subscription."
                    )
                time.sleep(interval_seconds)
                continue

            # -- Log all scan results --
            jsonl_lines = []
            for setup in all_results:
                metadata = dict(setup.metadata or {})
                if stale_data:
                    metadata["stale_data"] = True
                record = {
                    "timestamp": timestamp_now,
                    "ticker": setup.ticker,
                    "setup_type": setup.setup_type.value,
                    "is_valid": setup.is_valid,
                    "confidence": float(setup.confidence),
                    "entry_trigger": float(setup.entry_trigger),
                    "stop_loss": float(setup.stop_loss),
                    "target_1": float(setup.target_1),
                    "target_2": float(setup.target_2),
                    "risk_reward_1": float(setup.risk_reward_1),
                    "risk_reward_2": float(setup.risk_reward_2),
                    "reason": setup.reason,
                    "warnings": setup.warnings,
                    "metadata": metadata,
                }
                jsonl_lines.append(json.dumps(record))

            with open(scan_log, "a") as f:
                for line in jsonl_lines:
                    f.write(line + "\n")

            # -- Display and alert on valid ranked setups --
            cooldown_summary = _daily_setup_cooldown_summary()
            if stale_data:
                _print_stale_data_review_banner(indent="  ")

            if not top_setups:
                typer.echo("  No valid setups.")
                print_setup_market_bias_summary(all_results, indent="  ")
            else:
                if stale_data and not allow_stale_alerts:
                    typer.secho("  Stale Review Setups:", bold=True)
                    for setup in top_setups:
                        typer.echo(format_trade_action_card(
                            setup,
                            account_equity=account_equity,
                            risk_pct=risk_pct,
                            stale_data=True,
                        ))
                        typer.echo("")
                    typer.echo("  Setup alerts suppressed because market data is stale. Use --allow-stale-alerts to override.")
                    if max_iterations is None or iteration < max_iterations:
                        time.sleep(interval_seconds)
                    continue

                # Separate new and repeated setups
                new_setups = []
                repeated_count = 0

                for setup in top_setups:
                    alert_key = (setup.ticker, setup.setup_type.value, setup.entry_trigger)

                    if alert_key not in seen_alerts:
                        new_setups.append(setup)
                        seen_alerts.add(alert_key)
                    else:
                        repeated_count += 1

                # Display new setups
                if new_setups:
                    typer.secho("  New Setups:", bold=True)
                    for setup in new_setups:
                        rank = setup.metadata.get("rank", "?")
                        cooldown_reasons = _setup_cooldown_reasons(setup, cooldown_summary)
                        is_cooled_down = bool(cooldown_reasons)
                        typer.secho(f"    {rank}. {setup.ticker} {setup.setup_type.value}", fg=typer.colors.GREEN)
                        typer.echo(f"       Entry: {setup.entry_trigger:.2f} | Stop: {setup.stop_loss:.2f} | Conf: {setup.confidence:.2f}")
                        if is_cooled_down:
                            _print_setup_cooldown_warning(cooldown_reasons, indent="       ")

                        if is_cooled_down and suppress_cooled_down_alerts:
                            typer.echo("       Setup alert suppressed by daily cooldown. Review-only output shown.")
                            typer.echo(format_trade_action_card(
                                setup,
                                account_equity=account_equity,
                                risk_pct=risk_pct,
                                stale_data=True,
                            ))
                            typer.echo("")
                            continue

                        # Trigger alerts for new setup
                        if alert_banner:
                            print_alert_banner(setup, account_equity=account_equity, risk_pct=risk_pct, stale_data=stale_data)
                        if discord_enabled:
                            send_discord_message(format_setup_discord_alert(setup))
                        # Log new alert to file
                        metadata = dict(setup.metadata or {})
                        if stale_data:
                            metadata["stale_data"] = True
                        if is_cooled_down:
                            metadata["cooldown_warning"] = True
                            metadata["cooldown_reasons"] = cooldown_reasons
                        alert_record = {
                            "timestamp": timestamp_now,
                            "rank": rank,
                            "ticker": setup.ticker,
                            "setup_type": setup.setup_type.value,
                            "is_valid": setup.is_valid,
                            "confidence": float(setup.confidence),
                            "entry_trigger": float(setup.entry_trigger),
                            "stop_loss": float(setup.stop_loss),
                            "target_1": float(setup.target_1),
                            "target_2": float(setup.target_2),
                            "risk_reward_1": float(setup.risk_reward_1),
                            "risk_reward_2": float(setup.risk_reward_2),
                            "reason": setup.reason,
                            "metadata": metadata,
                        }
                        with open(alert_log, "a") as f:
                            f.write(json.dumps(alert_record) + "\n")

                # Show suppression summary
                if repeated_count > 0:
                    if new_setups:
                        typer.echo(f"  Repeated setups suppressed: {repeated_count}")
                    else:
                        typer.echo(f"  No new setups. Repeated setups suppressed: {repeated_count}")

            # -- Wait for next iteration --
            if max_iterations is None or iteration < max_iterations:
                time.sleep(interval_seconds)

    except KeyboardInterrupt:
        typer.secho("\n\nLoop stopped by user (Ctrl+C).", fg=typer.colors.YELLOW)
        raise typer.Exit(0)


# ---------------------------------------------------------------------------
# forward-test-session (automated shadow forward testing — SIMULATION ONLY)
# ---------------------------------------------------------------------------

def _run_shadow_scan_once(
    tickers: list,
    provider,
    timeframe: str,
    lookback_days: int,
    min_confidence: float,
    top_n: int,
    synthetic: bool,
) -> tuple:
    """
    Execute one shadow-session scan and return (top_setups, stale_data, data_by_ticker).

    Same scan pipeline as _run_setup_scan_once, but also returns the loaded
    bars so open shadow trades can be monitored without a second data fetch.
    Raises DataProviderError on data load failure.
    """
    from strategy.day_trade_setups import DayTradeSetupDetector
    from strategy.setup_ranking import SetupRanker
    import pandas as pd

    if synthetic:
        data_by_ticker = provider.get_multiple_bars(tickers, timeframe=timeframe)
    else:
        provider_name = getattr(provider, "provider_name", "")
        end = None if provider_name == "AlpacaData" else str(pd.Timestamp.now().date())
        start = str((pd.Timestamp.now() - pd.Timedelta(days=lookback_days)).date())
        data_by_ticker = provider.get_multiple_bars(
            tickers, start=start, end=end, timeframe=timeframe,
        )

    stale_data = _warn_if_provider_data_stale(
        data_by_ticker, getattr(provider, "provider_name", ""), timeframe, indent="  ",
    )

    detector = DayTradeSetupDetector()
    results = detector.detect_many(data_by_ticker)

    ranker = SetupRanker()
    top_setups = ranker.filter_top_setups(
        results, min_confidence=min_confidence, max_results=top_n,
    )
    return top_setups, stale_data, data_by_ticker


@app.command(name="forward-test-session")
def forward_test_session(
    ticker:       str  = typer.Option("SPY", "--ticker", help="Ticker(s), comma-separated"),
    watchlist:    Optional[str] = typer.Option(None, "--watchlist", help="Load tickers from a watchlist (e.g., mega_caps or momentum_universe)"),
    synthetic:    bool = typer.Option(False, "--synthetic", help="Use synthetic data (offline)"),
    provider_name: Optional[str] = typer.Option(None, "--provider", help="Market data provider: yfinance, alpaca, synthetic"),
    alpaca_feed:  str  = typer.Option("iex", "--alpaca-feed", help="Alpaca market data feed: iex or sip (default iex/free feed)"),
    lookback_days: Optional[int] = typer.Option(None, "--lookback-days", help="Days of history (defaults match setup-scan-loop)"),
    timeframe:    Optional[str]  = typer.Option(None, "--timeframe", show_default=False, help="Bar size: 1Day, 1Min, 5Min, 15Min (default 5Min)"),
    interval_seconds: int = typer.Option(300, "--interval-seconds", help="Wait this many seconds between iterations (default 300)"),
    max_iterations: Optional[int] = typer.Option(None, "--max-iterations", help="Stop after this many iterations (required unless --synthetic)"),
    top_n:        int  = typer.Option(5, "--top-n", help="Top N ranked setups evaluated per iteration (default 5)"),
    scan_min_confidence: float = typer.Option(0.6, "--scan-min-confidence", help="Ranker confidence floor so near-miss setups are still logged as SKIPPED (default 0.6)"),
    min_confidence: float = typer.Option(0.80, "--min-confidence", help="Shadow ENTRY confidence threshold (default 0.80)"),
    max_rank:     int  = typer.Option(1, "--max-rank", help="Only auto-enter setups ranked at or above this (default 1)"),
    max_open_trades: int = typer.Option(3, "--max-open-trades", help="Maximum simultaneous open shadow trades (default 3)"),
    max_new_trades_per_day: int = typer.Option(10, "--max-new-trades-per-day", help="Maximum shadow entries per calendar day (default 10)"),
    entry_cutoff: str = typer.Option("13:00", "--entry-cutoff", help="Block new shadow entries at or after HH:MM ET (default 13:00)"),
    sim_notional: float = typer.Option(1000.0, "--sim-notional", help="Simulated dollars per shadow trade; shares = notional / entry (default 1000)"),
    discord:      bool = typer.Option(True, "--discord/--no-discord", help="Send SHADOW ENTRY/EXIT Discord messages when webhook is configured"),
    end_time:     Optional[str] = typer.Option(None, "--end-time", help="Stop the session at HH:MM ET (default in live mode: market close)"),
    resume:       bool = typer.Option(True, "--resume/--no-resume", help="Adopt and monitor open shadow trades from prior sessions (default resume)"),
    ignore_market_hours: bool = typer.Option(False, "--ignore-market-hours", help="Allow a live-data session while the market is closed (review/testing only)"),
    max_data_failures: int = typer.Option(5, "--max-data-failures", help="End the session after this many consecutive data failures (default 5)"),
    suppress_cooled_down_alerts: bool = typer.Option(True, "--suppress-cooled-down-alerts/--no-suppress-cooled-down-alerts", help="Skip entries for cooled-down setup types/tickers (default on; off = warn-only)"),
    close_open_at_session_end: bool = typer.Option(False, "--close-open-at-session-end", help="Close current-session open shadow trades at normal session end (default off)"),
) -> None:
    """
    Automated shadow forward-test session — SIMULATION ONLY.

    Scans for setups each iteration, auto-enters qualifying alerts as SHADOW
    journal trades (logs/shadow_trades.jsonl), monitors them, and auto-closes
    on the first exit signal. Answers: "what would have happened if the bot
    followed the rules perfectly?"

    Safety: NEVER places orders or contacts a broker for execution. Shadow
    trades are journal records only; logs/manual_trades.jsonl is never touched.
    Live-data sessions run only during market hours (or --ignore-market-hours)
    and stop at --end-time / market close; --synthetic stays bounded by
    --max-iterations.

    Examples:
        python main.py forward-test-session --synthetic --max-iterations 2 --interval-seconds 0
        python main.py forward-test-session --provider alpaca --alpaca-feed iex --watchlist mega_caps --timeframe 5Min
        python main.py forward-test-session --provider alpaca --alpaca-feed iex --watchlist momentum_universe --timeframe 5Min --max-iterations 72 --close-open-at-session-end
        python main.py forward-test-session --watchlist mega_caps --end-time 15:30
    """
    from dataclasses import replace as _dc_replace
    from datetime import date as _date

    from alerts.discord_alerts import (
        format_shadow_entry_discord_alert,
        format_shadow_exit_discord_alert,
        format_shadow_session_summary_discord_message,
        is_discord_configured,
        send_discord_message,
    )
    from trades.shadow_session import (
        SESSION_ENDED,
        SESSION_ERROR,
        SESSION_INTERRUPTED,
        SESSION_RUNNING,
        SHADOW_EXIT_ACTIONS,
        SHADOW_TRADES_FILE,
        ShadowDecisionLog,
        ShadowEntryConfig,
        ShadowEntryDecision,
        ShadowEntryPolicy,
        ShadowSkipReason,
        ShadowSessionState,
        ShadowSessionStats,
        adopt_open_shadow_trades,
        append_shadow_session_history,
        acquire_shadow_session_lock,
        create_shadow_trade_from_setup,
        is_market_open_et,
        market_close_et,
        now_eastern,
        parse_entry_cutoff_et,
        parse_end_time_et,
        record_orb_no_progress_diagnostics,
        release_shadow_session_lock,
        shadow_exit_price,
    )
    from trades.trade_monitor import TradeMonitor
    from trades.trade_tracker import TradeTracker

    # -- Bounded-run safety gates: never run unbounded by accident --
    now_et = now_eastern()
    session_end_et = None
    try:
        entry_cutoff_et = parse_entry_cutoff_et(entry_cutoff)
    except ValueError:
        typer.secho(
            f"Error: invalid --entry-cutoff '{entry_cutoff}' (expected HH:MM).",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    if synthetic:
        if max_iterations is None:
            max_iterations = 1
        market_status = "n/a (synthetic data)"
    else:
        market_open = is_market_open_et(now_et)
        market_status = "open" if market_open else "closed"
        if not market_open and not ignore_market_hours:
            typer.secho(
                "Error: market is closed (Mon-Fri 9:30-16:00 ET). Use "
                "--ignore-market-hours for a review/testing session, or --synthetic.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
        if end_time:
            try:
                session_end_et = parse_end_time_et(end_time, now_et)
            except ValueError:
                typer.secho(f"Error: invalid --end-time '{end_time}' (expected HH:MM).", fg=typer.colors.RED)
                raise typer.Exit(1)
            if session_end_et <= now_et:
                typer.secho(f"Error: --end-time {end_time} ET has already passed.", fg=typer.colors.RED)
                raise typer.Exit(1)
        elif market_open:
            session_end_et = market_close_et(now_et)
        if session_end_et is None and max_iterations is None:
            typer.secho(
                "Error: unbounded session. Provide --end-time or --max-iterations "
                "when running with --ignore-market-hours.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

    # -- Provider selection (Alpaca IEX preferred when configured) --
    selected_provider, recommended_feed, provider_warning = _recommended_intraday_provider(provider_name)
    if provider_name is None and selected_provider == "alpaca":
        alpaca_feed = recommended_feed
    if synthetic:
        selected_provider = "synthetic"
    timeframe = _resolve_scan_timeframe(selected_provider, timeframe, "5Min")

    tickers = merge_tickers(ticker if ticker != "SPY" or watchlist else None, watchlist)
    if not tickers:
        tickers = ["SPY"]

    try:
        provider = _get_named_data_provider(
            selected_provider, synthetic=synthetic, csv_file=None, alpaca_feed=alpaca_feed
        )
    except typer.BadParameter as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)

    user_passed_lookback = lookback_days is not None
    lookback_days = _validate_intraday_lookback(
        timeframe,
        lookback_days or 400,
        selected_provider != "yfinance",
        None,
        user_passed_lookback,
    )

    session_id = f"shadow-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    lock_acquired, existing_lock, lock_reason = acquire_shadow_session_lock(
        session_id,
        metadata={
            "command": "forward-test-session",
            "watchlist_name": watchlist or "",
            "ticker_count": len(tickers),
            "tickers": tickers,
            "provider": selected_provider,
            "timeframe": timeframe,
            "entry_cutoff_et": entry_cutoff_et.strftime("%H:%M"),
        },
    )
    if not lock_acquired:
        existing_session_id = str((existing_lock or {}).get("session_id") or "unknown")
        typer.secho(
            "A shadow forward-test session appears to already be running. "
            "Refusing to start another one.",
            fg=typer.colors.RED,
        )
        typer.echo(f"Existing session: {existing_session_id}")
        raise typer.Exit(1)

    try:
        policy = ShadowEntryPolicy(ShadowEntryConfig(
            min_confidence=min_confidence,
            max_rank=max_rank,
            max_open_trades=max_open_trades,
            max_new_trades_per_day=max_new_trades_per_day,
            entry_cutoff_et=entry_cutoff_et,
            enforce_cooldowns=suppress_cooled_down_alerts,
        ))
        tracker = TradeTracker(storage_path=SHADOW_TRADES_FILE)
        decision_log = ShadowDecisionLog()
        monitor = TradeMonitor()
        stats = ShadowSessionStats()
        discord_enabled = bool(discord and is_discord_configured())

        typer.secho("\n=== Shadow Forward Test Session (SIMULATION ONLY) ===\n", bold=True)
        typer.secho("  No real orders will be placed. Shadow journal trades only.\n", fg=typer.colors.GREEN)
        typer.echo(f"  Session: {session_id}")
        typer.echo(f"  Tickers: {', '.join(tickers)}")
        typer.echo(f"  Timeframe: {timeframe}")
        typer.echo(f"  Market status: {market_status}")
        typer.echo(
            "  Session end: "
            + (f"{session_end_et.strftime('%H:%M')} ET" if session_end_et
               else "n/a (bounded by --max-iterations)")
        )
        typer.echo(f"  Interval: {interval_seconds}s | Max iterations: {max_iterations}")
        typer.echo(f"  Data provider: {_data_provider_display_name(selected_provider, provider.provider_name, alpaca_feed)}")
        if provider_warning and selected_provider == "yfinance":
            typer.echo(f"  {provider_warning}")
        typer.echo(f"  Entry rules: conf>={min_confidence:.2f} rank<={max_rank} "
                   f"max_open={max_open_trades} max_daily={max_new_trades_per_day}")
        typer.echo(f"  New-entry cutoff: {entry_cutoff_et.strftime('%H:%M')} ET")
        typer.echo(f"  Cooldowns: {'enforced' if suppress_cooled_down_alerts else 'warn-only'}")
        typer.echo(f"  Discord: {'enabled' if discord_enabled else 'not configured/disabled'}")

        # Read recovery evidence before replacing the previous state snapshot.
        prior_state = ShadowSessionState.load()
        # -- Session state (state/shadow_session.json) --
        state = ShadowSessionState(
            session_id=session_id,
            started_at=datetime.now().isoformat(),
            status=SESSION_RUNNING,
            config={
                "watchlist_name": watchlist or "",
                "ticker_count": len(tickers),
                "tickers": tickers,
                "timeframe": timeframe,
                "provider": selected_provider,
                "min_confidence": min_confidence,
                "max_rank": max_rank,
                "max_open_trades": max_open_trades,
                "max_new_trades_per_day": max_new_trades_per_day,
                "entry_cutoff_et": entry_cutoff_et.strftime("%H:%M"),
                "sim_notional": sim_notional,
                "end_time": end_time,
                "resume": resume,
                "suppress_cooled_down_alerts": suppress_cooled_down_alerts,
                "close_open_at_session_end": close_open_at_session_end,
                "feed": alpaca_feed if selected_provider == "alpaca" else None,
            },
        )
        # -- Resume: adopt prior-session open shadow trades (never force-closed) --
        adopted = []
        if resume:
            adopted = adopt_open_shadow_trades(tracker, session_id, prior_state, state)
            for trade in adopted:
                typer.echo(
                    f"  Adopted open shadow trade: {trade.trade_id} {trade.ticker} "
                    f"{trade.setup_type} (left OPEN, will be monitored)"
                )
        else:
            prior_open = [
                t for t in tracker.list_open_trades()
                if (t.metadata or {}).get("session_id") != session_id
            ]
            if prior_open:
                typer.echo(
                    f"  --no-resume: {len(prior_open)} prior open shadow trade(s) "
                    "will NOT be monitored this session (left OPEN)."
                )

        def _monitored_this_session(trade) -> bool:
            return ((trade.metadata or {}).get("session_id") == session_id
                    or trade.trade_id in adopted_ids)

        adopted_ids = {trade.trade_id for trade in adopted}
        unadopted = [t for t in tracker.list_open_trades() if not _monitored_this_session(t)]
        if unadopted:
            typer.echo(f"  Unadopted historical open shadow trades: {len(unadopted)}; left unchanged and unmonitored.")
        state.open_trade_ids = [t.trade_id for t in tracker.list_open_trades() if _monitored_this_session(t)]
        state.counters["open"] = len(state.open_trade_ids)
        state.save()

        def _send_shadow_discord(message: str) -> None:
            if not discord_enabled:
                return
            if not send_discord_message(message):
                stats.record_discord_failure()

        iteration = 0
        consecutive_data_failures = 0
        latest_data_by_ticker = {}
        last_iteration_started_at = None
        data_gap_threshold_seconds = max(900.0, float(interval_seconds) * 2.5)
        extreme_gap_threshold_seconds = max(3600.0, float(interval_seconds) * 12)
        pending_exception = None
        try:
            while True:
                iteration += 1
                if max_iterations and iteration > max_iterations:
                    break

                now = datetime.now()
                if last_iteration_started_at is not None:
                    elapsed_seconds = (now - last_iteration_started_at).total_seconds()
                    stats.max_iteration_gap_seconds = max(stats.max_iteration_gap_seconds, elapsed_seconds)
                    if elapsed_seconds > data_gap_threshold_seconds:
                        stats.record_data_gap(
                            last_iteration_started_at.isoformat(),
                            now.isoformat(),
                            elapsed_seconds,
                            interval_seconds,
                            extreme=elapsed_seconds >= extreme_gap_threshold_seconds,
                        )
                        typer.secho(
                            "  WARNING: data gap detected between iterations: "
                            f"{elapsed_seconds / 60:.1f} minutes. Session will be marked partial.",
                            fg=typer.colors.YELLOW,
                        )
                        if elapsed_seconds >= extreme_gap_threshold_seconds:
                            typer.echo("  SESSION_SUSPENSION: extreme iteration gap; PARTIAL_DATA: ITERATION_GAP. New entries blocked.")
                        state.counters = stats.counters()
                        state.counters["open"] = len(state.open_trade_ids)
                        state.last_iteration_at = now.isoformat()
                        state.save()
                last_iteration_started_at = now
                now_et = now_eastern()
                if session_end_et is not None and now_et >= session_end_et:
                    typer.echo(
                        f"\nReached session end time ({session_end_et.strftime('%H:%M')} ET). Stopping."
                    )
                    break

                typer.echo(f"\n[{now.isoformat()}] Shadow iteration {iteration}")

                # -- Scan --
                try:
                    top_setups, stale_data, data_by_ticker = _run_shadow_scan_once(
                        tickers, provider, timeframe, lookback_days,
                        scan_min_confidence, top_n, synthetic,
                    )
                except Exception as exc:  # noqa: BLE001 — loop must survive provider hiccups (KeyboardInterrupt passes through)
                    stats.record_data_failure()
                    consecutive_data_failures += 1
                    typer.secho(f"  Scan failed: {exc}", fg=typer.colors.RED)
                    typer.echo(
                        f"  Skipping iteration (no entries, no exits, on failed data). "
                        f"Consecutive failures: {consecutive_data_failures}/{max_data_failures}"
                    )
                    if consecutive_data_failures >= max_data_failures:
                        typer.secho(
                            f"  {consecutive_data_failures} consecutive data failures — "
                            "ending session cleanly.",
                            fg=typer.colors.RED,
                        )
                        break
                    if max_iterations is None or iteration < max_iterations:
                        time.sleep(interval_seconds)
                    continue
                consecutive_data_failures = 0
                latest_data_by_ticker = data_by_ticker
                stats.record_iteration(len(top_setups))

                if getattr(provider, "provider_name", "") == "AlpacaData":
                    stats.missing_bar_observations = (stats.missing_bar_observations or 0) + sum(
                        symbol not in data_by_ticker or getattr(data_by_ticker[symbol], "empty", True)
                        for symbol in dict.fromkeys(tickers)
                    )

                if stale_data:
                    from data_providers.freshness import freshness_snapshot
                    stats.record_stale_data_event(
                        now_et.isoformat(), _provider_max_data_age_seconds(data_by_ticker),
                        freshness_snapshot=freshness_snapshot(
                            data_by_ticker, tickers, timeframe, provider.provider_name,
                            getattr(provider, "feed", None), now_et.isoformat(),
                        ) if getattr(provider, "provider_name", "") == "AlpacaData" else None,
                    )
                    stats.record_data_quality_failure("STALE_DATA", now_et.isoformat())
                    _print_stale_data_review_banner(indent="  ")

                cooldown_summary = _daily_setup_cooldown_summary()
                entries_today = decision_log.count_entered_on(_date.today())

                # -- Evaluate top setups with the shadow entry policy --
                open_trades = tracker.list_open_trades()
                if not top_setups:
                    typer.echo("  No ranked setups this iteration.")
                for setup in top_setups:
                    # Setups were detected THIS iteration, so evaluate alert age
                    # from detection time; old-bars risk is covered by the
                    # stale-data block above.
                    eval_setup = _dc_replace(setup, timestamp=now)
                    decision = policy.evaluate(
                        eval_setup,
                        now=now,
                        open_shadow_trades=open_trades,
                        cooldown_summary=cooldown_summary,
                        stale_data=bool(stale_data),
                        entries_today=entries_today,
                        now_et=now_et,
                    )

                    # Data quality failure is a one-way latch for this session:
                    # continue monitoring exits, but never create another entry.
                    if stats.data_quality == "PARTIAL_DATA":
                        partial_reason = ShadowSkipReason.PARTIAL_SESSION.value
                        reasons = list(decision.skip_reasons)
                        if partial_reason not in reasons:
                            reasons.append(partial_reason)
                        decision = ShadowEntryDecision(decision="SKIP", skip_reasons=reasons)

                    if decision.accepted:
                        if not suppress_cooled_down_alerts:
                            cooldown_reasons = _setup_cooldown_reasons(eval_setup, cooldown_summary)
                            if cooldown_reasons:
                                _print_setup_cooldown_warning(cooldown_reasons, indent="  ")
                        entry_price = float(setup.entry_trigger or 0.0)
                        shares = round(sim_notional / entry_price, 2) if entry_price > 0 else 0.0
                        trade = create_shadow_trade_from_setup(
                            eval_setup, shares=shares, session_id=session_id, now=now,
                        )
                        tracker.add_trade(trade)
                        decision_log.record(session_id, eval_setup, decision,
                                            trade_id=trade.trade_id, timestamp=now)
                        stats.record_entry()
                        entries_today += 1
                        open_trades = tracker.list_open_trades()
                        typer.secho(
                            f"  SHADOW ENTRY (simulated): {trade.ticker} {trade.setup_type} "
                            f"@ {trade.entry_price:.2f} x {trade.shares} — no real order placed",
                            fg=typer.colors.CYAN,
                        )
                        _send_shadow_discord(format_shadow_entry_discord_alert(trade))
                    else:
                        decision_log.record(session_id, eval_setup, decision, timestamp=now)
                        stats.record_skip(decision.skip_reasons)
                        typer.echo(
                            f"  SKIP {setup.ticker} { _setup_type_display(setup) }: "
                            f"{', '.join(decision.skip_reasons)}"
                        )

                # -- Monitor open shadow trades; close on first exit signal.
                #    Stale data blocks ENTRIES (policy), but exits still process
                #    and get tagged stale_exit=true for analytics segmentation. --
                open_trades = [
                    t for t in tracker.list_open_trades() if _monitored_this_session(t)
                ]
                if open_trades:
                    results = monitor.monitor_open_trades(open_trades, data_by_ticker)
                    for trade, result in zip(open_trades, results):
                        record_orb_no_progress_diagnostics(
                            tracker,
                            trade,
                            result,
                            now,
                            bars=data_by_ticker.get(trade.ticker),
                        )
                        action_value = str(getattr(result.action, "value", result.action))
                        if action_value not in SHADOW_EXIT_ACTIONS:
                            continue
                        exit_price = shadow_exit_price(action_value, trade, result.current_price)
                        metadata_updates = {"exit_signal": action_value}
                        if stale_data:
                            metadata_updates["stale_exit"] = True
                        closed = tracker.close_trade(
                            trade.trade_id,
                            exit_price=exit_price,
                            exit_time=now.isoformat(),
                            metadata_updates=metadata_updates,
                        )
                        stats.record_close(closed.realized_pnl or 0.0)
                        typer.secho(
                            f"  SHADOW EXIT (simulated): {closed.ticker} @ {closed.exit_price:.2f} "
                            f"({action_value}) P&L {closed.realized_pnl:+.2f} — no real order placed",
                            fg=typer.colors.YELLOW,
                        )
                        _send_shadow_discord(format_shadow_exit_discord_alert(closed, action_value))

                # -- Persist session state after every iteration --
                state.counters = stats.counters()
                state.open_trade_ids = [t.trade_id for t in tracker.list_open_trades() if _monitored_this_session(t)]
                state.counters["open"] = len(state.open_trade_ids)
                state.last_iteration_at = now.isoformat()
                state.save()

                # -- Wait for next iteration --
                if max_iterations is None or iteration < max_iterations:
                    time.sleep(interval_seconds)

        except KeyboardInterrupt:
            session_status = SESSION_INTERRUPTED
            typer.secho("\n\nSession stopped by user (Ctrl+C).", fg=typer.colors.YELLOW)
            typer.echo("Open shadow trades were left OPEN (not force-closed); "
                       "the next session can adopt them.")
        except Exception as exc:  # noqa: BLE001 - persist ERROR state before surfacing unexpected failures
            pending_exception = exc
            session_status = SESSION_ERROR
            typer.secho(f"\nShadow session stopped after unexpected error: {exc}", fg=typer.colors.RED)
        else:
            session_status = SESSION_ENDED

        if session_status == SESSION_ENDED and close_open_at_session_end:
            now = datetime.now()
            current_session_open = [
                t for t in tracker.list_open_trades()
                if _monitored_this_session(t)
            ]
            for trade in current_session_open:
                bars = latest_data_by_ticker.get(trade.ticker)
                if bars is None or getattr(bars, "empty", True) or "close" not in bars.columns:
                    typer.secho(
                        f"  SESSION_END close skipped for {trade.ticker}: no latest close available.",
                        fg=typer.colors.YELLOW,
                    )
                    continue
                from math import isfinite
                try:
                    exit_price = float(bars["close"].iloc[-1])
                except (TypeError, ValueError, OverflowError):
                    exit_price = float("nan")
                if not isfinite(exit_price) or exit_price <= 0:
                    typer.echo(f"  SESSION_END close skipped for {trade.ticker}: invalid latest close.")
                    continue
                closed = tracker.close_trade(
                    trade.trade_id,
                    exit_price=exit_price,
                    exit_time=now.isoformat(),
                    metadata_updates={"exit_signal": "SESSION_END", "session_end_exit": True},
                )
                stats.record_close(closed.realized_pnl or 0.0)
                typer.secho(
                    f"  SHADOW EXIT (simulated): {closed.ticker} @ {closed.exit_price:.2f} "
                    f"(SESSION_END) P&L {closed.realized_pnl:+.2f} — no real order placed",
                    fg=typer.colors.YELLOW,
                )
                _send_shadow_discord(format_shadow_exit_discord_alert(closed, "SESSION_END"))

            if any(_monitored_this_session(t) for t in tracker.list_open_trades()):
                session_status = SESSION_ERROR
                stats.record_data_quality_failure("SESSION_END_CLOSE_INCOMPLETE", now.isoformat())
                typer.echo("  Session-end cleanup incomplete: trades remain OPEN for recovery; session is not ENDED.")

        # -- Final session state --
        state.status = session_status
        state.counters = stats.counters()
        state.open_trade_ids = [t.trade_id for t in tracker.list_open_trades() if _monitored_this_session(t)]
        state.counters["open"] = len(state.open_trade_ids)
        state.counters["unadopted_open"] = sum(not _monitored_this_session(t) for t in tracker.list_open_trades())
        state.save()

        # -- End-of-session summary (terminal + optional Discord) --
        open_count = len(state.open_trade_ids)
        summary = stats.summary_lines(open_count)
        summary.append(f"  Provider/feed: {selected_provider}/{state.config.get('feed') or 'n/a'}")
        summary.append(f"  Unadopted historical open trades: {state.counters['unadopted_open']}")
        append_shadow_session_history(state, summary)
        typer.echo("")
        typer.echo(f"Session: {session_id} [{session_status}]")
        for line in summary:
            typer.echo(line)
        typer.echo(f"Shadow journal: {SHADOW_TRADES_FILE}")
        typer.echo("Decision log: logs/shadow_decisions.jsonl")
        typer.echo("Session state: state/shadow_session.json")
        typer.secho("No real orders were placed.", fg=typer.colors.GREEN)
        _send_shadow_discord(
            format_shadow_session_summary_discord_message(session_id, summary, status=session_status)
        )
        if pending_exception is not None:
            raise pending_exception
    except BaseException as exc:
        # Startup/adoption and finalization sit outside the iteration handler.
        # Preserve the journal's actual open set if either fails. A successfully
        # persisted ENDED session must not become recoverable on a reporting error.
        persisted = ShadowSessionState.load()
        if persisted is not None and persisted.session_id == session_id and persisted.status != SESSION_ENDED:
            persisted.status = SESSION_INTERRUPTED if isinstance(exc, KeyboardInterrupt) else SESSION_ERROR
            stats.record_data_quality_failure("SESSION_LIFECYCLE_ERROR", datetime.now().isoformat())
            persisted.counters = stats.counters()
            try:
                persisted.open_trade_ids = [
                    t.trade_id for t in tracker.list_open_trades()
                    if t.trade_id in persisted.open_trade_ids
                    or (t.metadata or {}).get("session_id") == session_id
                    or (t.metadata or {}).get("monitoring_session_id") == session_id
                ]
                persisted.counters["open"] = len(persisted.open_trade_ids)
                persisted.save()
            except Exception:
                typer.echo("  ERROR: recovery state could not be persisted; manual journal/state review required.")
        raise
    finally:
        release_shadow_session_lock(session_id)


def _setup_type_display(setup) -> str:
    """Plain setup_type string for terminal output."""
    raw = getattr(setup, "setup_type", "")
    return str(getattr(raw, "value", raw) or "")


# ---------------------------------------------------------------------------
# Manual trade tracking (journal only — never places orders)
# ---------------------------------------------------------------------------

@app.command(name="trade-add")
def trade_add(
    ticker:     str   = typer.Option(..., "--ticker", help="Ticker symbol you entered"),
    setup_type: str   = typer.Option(..., "--setup-type", help="Setup type (e.g. opening_range_breakout_long)"),
    entry:      float = typer.Option(..., "--entry", help="Entry fill price"),
    stop:       float = typer.Option(..., "--stop", help="Stop loss price"),
    target1:    float = typer.Option(..., "--target1", help="First target price"),
    target2:    float = typer.Option(..., "--target2", help="Second target price"),
    shares:     float = typer.Option(..., "--shares", help="Number of shares"),
    setup_alert_id: Optional[str] = typer.Option(None, "--setup-alert-id", help="Optional setup alert id this trade came from"),
    setup_rank: Optional[int] = typer.Option(None, "--setup-rank", help="Optional setup rank when the idea was scanned"),
    setup_score: Optional[float] = typer.Option(None, "--setup-score", help="Optional setup ranking score"),
    setup_confidence: Optional[float] = typer.Option(None, "--setup-confidence", help="Optional setup confidence"),
    exclude_from_analytics: bool = typer.Option(False, "--exclude-from-analytics", help="Exclude this trade from trade-analytics results"),
    exclusion_reason: str = typer.Option("", "--exclusion-reason", help="Reason this trade is excluded from analytics"),
    test_type: str = typer.Option("forward_test", "--test-type", help="Trade test category: forward_test, workflow_test, live_paper, real_trade"),
    thesis: str = typer.Option("", "--thesis", help="Why this trade was taken"),
    market_context: str = typer.Option("", "--market-context", help="Market context at entry"),
    entry_quality: str = typer.Option("", "--entry-quality", help="Entry quality grade, e.g. A, B, C, bad"),
    mistake: bool = typer.Option(False, "--mistake", help="Mark this trade as a mistake for later review"),
    mistake_notes: str = typer.Option("", "--mistake-notes", help="Notes about the mistake"),
    screenshot_path: str = typer.Option("", "--screenshot-path", help="Optional local chart screenshot path"),
    notes:      str   = typer.Option("", "--notes", help="Optional free-text notes"),
) -> None:
    """
    Manually record that you entered a trade (journal only — NEVER places orders).

    Example:
        python main.py trade-add --ticker SPY --setup-type vwap_reclaim_long \\
            --entry 500.0 --stop 498.0 --target1 504.0 --target2 508.0 --shares 10
    """
    from datetime import datetime
    from trades.trade_tracker import TradeTracker, ManualTrade

    if shares <= 0:
        typer.secho("Error: --shares must be greater than 0.", fg=typer.colors.RED)
        raise typer.Exit(1)
    if stop >= entry:
        typer.secho(
            "Error: --stop must be below --entry for a long trade.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    tracker = TradeTracker()
    trade = ManualTrade(
        ticker=ticker.upper(),
        setup_type=setup_type,
        entry_price=float(entry),
        stop_loss=float(stop),
        target_1=float(target1),
        target_2=float(target2),
        shares=float(shares),
        entry_time=datetime.now().isoformat(),
        setup_alert_id=setup_alert_id,
        setup_rank=setup_rank,
        setup_score=setup_score,
        setup_confidence=setup_confidence,
        exclude_from_analytics=exclude_from_analytics,
        exclusion_reason=exclusion_reason,
        test_type=test_type,
        thesis=thesis,
        market_context=market_context,
        entry_quality=entry_quality,
        mistake=mistake,
        mistake_notes=mistake_notes,
        screenshot_path=screenshot_path,
        notes=notes,
    )
    saved = tracker.add_trade(trade)

    typer.secho("\n=== Trade Recorded ===\n", bold=True)
    typer.secho(f"  Trade ID: {saved.trade_id}", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  {saved.ticker} {saved.setup_type}")
    typer.echo(f"  Entry: {saved.entry_price:.2f} | Stop: {saved.stop_loss:.2f} | Shares: {saved.shares:g}")
    typer.echo(f"  Targets: {saved.target_1:.2f} / {saved.target_2:.2f}")
    typer.echo(f"  Test type: {saved.test_type}")
    if saved.setup_alert_id:
        typer.echo(f"  Setup alert: {saved.setup_alert_id}")
    if saved.exclude_from_analytics:
        typer.echo(f"  [EXCLUDED] {saved.exclusion_reason}")
    typer.secho("\n  (Journal only — no order was placed.)", fg=typer.colors.YELLOW)


def _stable_setup_alert_id(record: dict) -> str:
    """Derive a stable display id for setup alert records without alert_id."""
    raw = "|".join(str(record.get(key, "")) for key in [
        "timestamp", "ticker", "setup_type", "entry_trigger",
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _load_setup_alert_records(log_file: Path = Path("logs/setup_alerts.jsonl")) -> list[dict]:
    """Load setup alert JSONL records, skipping malformed lines."""
    if not log_file.exists():
        return []

    records = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _setup_alert_display_id(record: dict) -> str:
    """Return explicit alert_id or stable derived id for older alert records."""
    return str(record.get("alert_id") or _stable_setup_alert_id(record))


def _parse_alert_timestamp(record: dict):
    """Parse an alert timestamp for sorting/filtering; invalid timestamps sort old."""
    try:
        import pandas as pd
        ts = pd.to_datetime(record.get("timestamp"), errors="coerce")
        if pd.isna(ts):
            return None
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert(None)
        return ts.to_pydatetime()
    except Exception:
        return None


def _record_is_synthetic_alert(record: dict) -> bool:
    """Best-effort synthetic alert detection from optional source metadata."""
    metadata = record.get("metadata") or {}
    candidates = [
        record.get("provider"),
        record.get("source"),
        record.get("timeframe"),
        metadata.get("provider"),
        metadata.get("source"),
        metadata.get("data_source"),
        metadata.get("timeframe"),
    ]
    return any("synthetic" in str(value).lower() for value in candidates if value is not None)


def _filter_setup_alert_records(
    records: list[dict],
    ticker: Optional[str] = None,
    since_hours: Optional[float] = None,
    valid_only: bool = True,
    exclude_synthetic: bool = True,
    now: Optional[datetime] = None,
) -> list[dict]:
    """Filter setup alert records and sort newest first."""
    import pandas as pd

    cutoff = None
    if since_hours is not None:
        cutoff = (now or datetime.now()) - pd.Timedelta(hours=float(since_hours))

    filtered = []
    ticker_upper = ticker.upper() if ticker else None
    for record in records:
        if ticker_upper and str(record.get("ticker", "")).upper() != ticker_upper:
            continue
        if valid_only and "is_valid" in record and not bool(record.get("is_valid")):
            continue
        if exclude_synthetic and _record_is_synthetic_alert(record):
            continue
        ts = _parse_alert_timestamp(record)
        if cutoff is not None and (ts is None or ts < cutoff):
            continue
        filtered.append(record)

    return sorted(
        filtered,
        key=lambda record: _parse_alert_timestamp(record) or datetime.min,
        reverse=True,
    )


def _create_manual_trade_from_alert_record(
    record: dict,
    alert_id: str,
    shares: float,
    notes: str = "",
    entry: Optional[float] = None,
    stop: Optional[float] = None,
    target1: Optional[float] = None,
    target2: Optional[float] = None,
    exclude_from_analytics: bool = False,
    exclusion_reason: str = "",
    test_type: str = "forward_test",
    thesis: str = "",
    market_context: str = "",
    entry_quality: str = "",
    mistake: bool = False,
    mistake_notes: str = "",
    screenshot_path: str = "",
):
    """Create and persist a ManualTrade from one setup alert record."""
    from datetime import datetime
    from trades.trade_tracker import ManualTrade, TradeTracker

    if shares <= 0:
        typer.secho("Error: --shares must be greater than 0.", fg=typer.colors.RED)
        raise typer.Exit(1)

    metadata = record.get("metadata") or {}
    entry_price = float(entry if entry is not None else record.get("entry_trigger", 0.0))
    stop_loss = float(stop if stop is not None else record.get("stop_loss", 0.0))
    target_1 = float(target1 if target1 is not None else record.get("target_1", 0.0))
    target_2 = float(target2 if target2 is not None else record.get("target_2", 0.0))

    if stop_loss >= entry_price:
        typer.secho("Error: --stop must be below entry for a long trade.", fg=typer.colors.RED)
        raise typer.Exit(1)

    score = record.get("score", metadata.get("score"))
    rank = record.get("rank", metadata.get("rank"))
    trade = ManualTrade(
        ticker=str(record.get("ticker", "")).upper(),
        setup_type=str(record.get("setup_type", "")),
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
        shares=float(shares),
        entry_time=datetime.now().isoformat(),
        setup_alert_id=alert_id,
        setup_rank=int(rank) if rank not in (None, "") else None,
        setup_score=float(score) if score not in (None, "") else None,
        setup_confidence=float(record.get("confidence")) if record.get("confidence") is not None else None,
        exclude_from_analytics=exclude_from_analytics,
        exclusion_reason=exclusion_reason,
        test_type=test_type,
        thesis=thesis,
        market_context=market_context,
        entry_quality=entry_quality,
        mistake=mistake,
        mistake_notes=mistake_notes,
        screenshot_path=screenshot_path,
        notes=notes,
    )
    return TradeTracker().add_trade(trade)


def _print_trade_from_alert(saved) -> None:
    """Print created trade details for alert-derived manual trades."""
    typer.secho("\n=== Trade Recorded From Setup Alert ===\n", bold=True)
    typer.secho(f"  Trade ID: {saved.trade_id}", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  Setup alert: {saved.setup_alert_id}")
    typer.echo(f"  {saved.ticker} {saved.setup_type}")
    typer.echo(f"  Entry: {saved.entry_price:.2f} | Stop: {saved.stop_loss:.2f} | Shares: {saved.shares:g}")
    typer.echo(f"  Targets: {saved.target_1:.2f} / {saved.target_2:.2f}")
    typer.secho("\n  (Journal only — no order was placed.)", fg=typer.colors.YELLOW)


@app.command(name="setup-alerts-recent")
def setup_alerts_recent(
    ticker: Optional[str] = typer.Option(None, "--ticker", help="Filter to one ticker"),
    since_hours: Optional[float] = typer.Option(None, "--since-hours", help="Only show alerts newer than this many hours"),
    valid_only: bool = typer.Option(True, "--valid-only/--all-alerts", help="Only show valid setup alerts when validity is present"),
    limit: int = typer.Option(10, "--limit", help="Number of recent setup alerts to show"),
    exclude_synthetic: bool = typer.Option(True, "--exclude-synthetic/--include-synthetic", help="Hide synthetic/test alerts when source metadata indicates synthetic"),
) -> None:
    """Print recent setup alerts with stable ids for manual trade linking."""
    log_file = Path("logs/setup_alerts.jsonl")
    typer.secho("\n=== Recent Setup Alerts ===\n", bold=True)

    records = _filter_setup_alert_records(
        _load_setup_alert_records(log_file),
        ticker=ticker,
        since_hours=since_hours,
        valid_only=valid_only,
        exclude_synthetic=exclude_synthetic,
    )[:limit]
    if not records:
        typer.echo("No setup alerts found.")
        return

    typer.echo(
        f"{'alert_id':<14} {'timestamp':<20} {'ticker':<8} {'setup_type':<34} "
        f"{'rank':>5} {'score':>8} {'conf':>8} {'entry':>10} {'stop':>10} {'t1':>10} {'t2':>10}"
    )
    typer.echo("-" * 145)
    for record in records:
        alert_id = _setup_alert_display_id(record)
        metadata = record.get("metadata") or {}
        score = record.get("score", metadata.get("score", ""))
        rank = record.get("rank", metadata.get("rank", ""))
        typer.echo(
            f"{str(alert_id):<14} "
            f"{str(record.get('timestamp', ''))[:19]:<20} "
            f"{str(record.get('ticker', '')):<8} "
            f"{str(record.get('setup_type', '')):<34} "
            f"{str(rank):>5} "
            f"{str(score):>8} "
            f"{float(record.get('confidence', 0.0)):>8.2f} "
            f"{float(record.get('entry_trigger', 0.0)):>10.2f} "
            f"{float(record.get('stop_loss', 0.0)):>10.2f} "
            f"{float(record.get('target_1', 0.0)):>10.2f} "
            f"{float(record.get('target_2', 0.0)):>10.2f}"
        )


@app.command(name="trade-add-from-alert")
def trade_add_from_alert(
    alert_id: str = typer.Option(..., "--alert-id", help="Setup alert id from setup-alerts-recent"),
    shares: float = typer.Option(..., "--shares", help="Number of shares manually entered"),
    notes: str = typer.Option("", "--notes", help="Optional free-text notes"),
    entry: Optional[float] = typer.Option(None, "--entry", help="Override alert entry price"),
    stop: Optional[float] = typer.Option(None, "--stop", help="Override alert stop loss"),
    target1: Optional[float] = typer.Option(None, "--target1", help="Override alert target 1"),
    target2: Optional[float] = typer.Option(None, "--target2", help="Override alert target 2"),
    exclude_from_analytics: bool = typer.Option(False, "--exclude-from-analytics", help="Exclude this trade from trade-analytics results"),
    exclusion_reason: str = typer.Option("", "--exclusion-reason", help="Reason this trade is excluded from analytics"),
    test_type: str = typer.Option("forward_test", "--test-type", help="Trade test category: forward_test, workflow_test, live_paper, real_trade"),
    thesis: str = typer.Option("", "--thesis", help="Why this trade was taken"),
    market_context: str = typer.Option("", "--market-context", help="Market context at entry"),
    entry_quality: str = typer.Option("", "--entry-quality", help="Entry quality grade, e.g. A, B, C, bad"),
    mistake: bool = typer.Option(False, "--mistake", help="Mark this trade as a mistake for later review"),
    mistake_notes: str = typer.Option("", "--mistake-notes", help="Notes about the mistake"),
    screenshot_path: str = typer.Option("", "--screenshot-path", help="Optional local chart screenshot path"),
) -> None:
    """Create a manual trade from a setup alert (journal only; no order placed)."""
    records = _load_setup_alert_records()
    matched = None
    for record in records:
        if _setup_alert_display_id(record) == alert_id:
            matched = record
            break

    if matched is None:
        typer.secho(f"Error: setup alert not found: {alert_id}", fg=typer.colors.RED)
        raise typer.Exit(1)

    saved = _create_manual_trade_from_alert_record(
        matched,
        alert_id=alert_id,
        shares=shares,
        notes=notes,
        entry=entry,
        stop=stop,
        target1=target1,
        target2=target2,
        exclude_from_analytics=exclude_from_analytics,
        exclusion_reason=exclusion_reason,
        test_type=test_type,
        thesis=thesis,
        market_context=market_context,
        entry_quality=entry_quality,
        mistake=mistake,
        mistake_notes=mistake_notes,
        screenshot_path=screenshot_path,
    )
    _print_trade_from_alert(saved)


@app.command(name="trade-add-latest-alert")
def trade_add_latest_alert(
    shares: float = typer.Option(..., "--shares", help="Number of shares manually entered"),
    ticker: Optional[str] = typer.Option(None, "--ticker", help="Filter to newest alert for this ticker"),
    setup_type: Optional[str] = typer.Option(None, "--setup-type", help="Filter to newest alert with this setup type"),
    notes: str = typer.Option("", "--notes", help="Optional free-text notes"),
    entry: Optional[float] = typer.Option(None, "--entry", help="Override alert entry price"),
    stop: Optional[float] = typer.Option(None, "--stop", help="Override alert stop loss"),
    target1: Optional[float] = typer.Option(None, "--target1", help="Override alert target 1"),
    target2: Optional[float] = typer.Option(None, "--target2", help="Override alert target 2"),
    max_age_minutes: float = typer.Option(30.0, "--max-age-minutes", help="Only use alerts newer than this many minutes"),
    allow_stale_alert: bool = typer.Option(False, "--allow-stale-alert", help="Allow older matching alerts after printing a stale warning"),
    exclude_from_analytics: bool = typer.Option(False, "--exclude-from-analytics", help="Exclude this trade from trade-analytics results"),
    exclusion_reason: str = typer.Option("", "--exclusion-reason", help="Reason this trade is excluded from analytics"),
    test_type: str = typer.Option("forward_test", "--test-type", help="Trade test category: forward_test, workflow_test, live_paper, real_trade"),
    thesis: str = typer.Option("", "--thesis", help="Why this trade was taken"),
    market_context: str = typer.Option("", "--market-context", help="Market context at entry"),
    entry_quality: str = typer.Option("", "--entry-quality", help="Entry quality grade, e.g. A, B, C, bad"),
    mistake: bool = typer.Option(False, "--mistake", help="Mark this trade as a mistake for later review"),
    mistake_notes: str = typer.Option("", "--mistake-notes", help="Notes about the mistake"),
    screenshot_path: str = typer.Option("", "--screenshot-path", help="Optional local chart screenshot path"),
) -> None:
    """Create a manual trade from the newest matching setup alert."""
    if shares <= 0:
        typer.secho("Error: --shares must be greater than 0.", fg=typer.colors.RED)
        raise typer.Exit(1)

    records = _filter_setup_alert_records(
        _load_setup_alert_records(),
        ticker=ticker,
        valid_only=True,
        exclude_synthetic=True,
    )
    if setup_type:
        records = [record for record in records if str(record.get("setup_type", "")) == setup_type]

    if not records:
        age_text = f"{max_age_minutes:g}"
        typer.secho(f"Error: No matching setup alert found within {age_text} minutes.", fg=typer.colors.RED)
        typer.echo(
            f"Suggestion: python main.py setup-alerts-recent --ticker {ticker or 'TICKER'} --since-hours 4"
        )
        raise typer.Exit(1)

    now = datetime.now()
    cutoff = now - timedelta(minutes=float(max_age_minutes))
    fresh_records = []
    for record in records:
        ts = _parse_alert_timestamp(record)
        if ts is not None and ts >= cutoff:
            fresh_records.append(record)

    if fresh_records:
        selected = fresh_records[0]
    elif allow_stale_alert:
        selected = records[0]
        typer.secho("Warning: using stale setup alert.", fg=typer.colors.YELLOW)
    else:
        age_text = f"{max_age_minutes:g}"
        typer.secho(f"Error: No matching setup alert found within {age_text} minutes.", fg=typer.colors.RED)
        typer.echo(
            f"Suggestion: python main.py setup-alerts-recent --ticker {ticker or 'TICKER'} --since-hours 4"
        )
        raise typer.Exit(1)

    alert_id = _setup_alert_display_id(selected)
    selected_ts = _parse_alert_timestamp(selected)
    age_minutes = ((now - selected_ts).total_seconds() / 60.0) if selected_ts else None
    saved = _create_manual_trade_from_alert_record(
        selected,
        alert_id=alert_id,
        shares=shares,
        notes=notes,
        entry=entry,
        stop=stop,
        target1=target1,
        target2=target2,
        exclude_from_analytics=exclude_from_analytics,
        exclusion_reason=exclusion_reason,
        test_type=test_type,
        thesis=thesis,
        market_context=market_context,
        entry_quality=entry_quality,
        mistake=mistake,
        mistake_notes=mistake_notes,
        screenshot_path=screenshot_path,
    )
    if age_minutes is None:
        typer.echo(f"Selected alert_id: {alert_id} (age: unknown)")
    else:
        typer.echo(f"Selected alert_id: {alert_id} (age: {age_minutes:.1f} minutes)")
    _print_trade_from_alert(saved)


@app.command(name="trade-list")
def trade_list(
    show_all: bool = typer.Option(False, "--all", help="Show all trades (default: open only)"),
) -> None:
    """List recorded trades. Defaults to open trades; use --all for everything."""
    from trades.trade_tracker import TradeTracker

    tracker = TradeTracker()
    trades = tracker.list_all_trades() if show_all else tracker.list_open_trades()

    heading = "All Trades" if show_all else "Open Trades"
    typer.secho(f"\n=== {heading} ===\n", bold=True)

    if not trades:
        typer.echo("  (none)")
        return

    for t in trades:
        excluded = " [EXCLUDED]" if getattr(t, "exclude_from_analytics", False) else ""
        mistake_marker = " [MISTAKE]" if getattr(t, "mistake", False) else ""
        line = (
            f"  [{t.status}]{excluded}{mistake_marker} {t.trade_id} | {t.ticker} {t.setup_type} | "
            f"entry {t.entry_price:.2f} stop {t.stop_loss:.2f} shares {t.shares:g}"
        )
        typer.echo(line)
        typer.echo(f"        test_type: {getattr(t, 'test_type', 'forward_test')}")
        if getattr(t, "thesis", ""):
            typer.echo(f"        thesis: {t.thesis}")
        if t.status == "CLOSED" and t.realized_pnl is not None:
            color = typer.colors.GREEN if t.realized_pnl >= 0 else typer.colors.RED
            typer.secho(f"        exit {t.exit_price:.2f} | P&L {t.realized_pnl:+.2f}", fg=color)
        if t.notes:
            typer.echo(f"        notes: {t.notes}")
        if getattr(t, "mistake", False) and getattr(t, "mistake_notes", ""):
            typer.echo(f"        mistake: {t.mistake_notes}")
        if getattr(t, "exclude_from_analytics", False) and getattr(t, "exclusion_reason", ""):
            typer.echo(f"        exclusion: {t.exclusion_reason}")


def _parse_optional_bool(value: Optional[str], option_name: str) -> Optional[bool]:
    """Parse optional true/false CLI text for correction commands."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise typer.BadParameter(f"{option_name} must be true or false")


@app.command(name="trade-edit")
def trade_edit(
    trade_id: str = typer.Option(..., "--trade-id", help="ID of the trade to correct"),
    exit_price: Optional[float] = typer.Option(None, "--exit-price", help="Corrected exit price for an already CLOSED trade"),
    notes: str = typer.Option("", "--notes", help="Optional correction note appended to trade notes"),
    mistake: Optional[str] = typer.Option(None, "--mistake", help="Set mistake true/false"),
    mistake_notes: Optional[str] = typer.Option(None, "--mistake-notes", help="Replace mistake notes"),
    exclude_from_analytics: Optional[str] = typer.Option(None, "--exclude-from-analytics", help="Set analytics exclusion true/false"),
) -> None:
    """Safely correct manual journal fields without deleting trade history."""
    from trades.trade_tracker import TradeTracker

    try:
        corrected = TradeTracker().edit_trade(
            trade_id=trade_id,
            exit_price=exit_price,
            notes=notes,
            mistake=_parse_optional_bool(mistake, "--mistake"),
            mistake_notes=mistake_notes,
            exclude_from_analytics=_parse_optional_bool(exclude_from_analytics, "--exclude-from-analytics"),
            correction_notes=notes,
        )
    except (ValueError, typer.BadParameter) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho("\n=== Trade Corrected ===\n", bold=True)
    typer.echo(f"  Trade ID: {corrected.trade_id}")
    typer.echo(f"  {corrected.ticker} {corrected.setup_type}")
    if corrected.exit_price is not None:
        typer.echo(f"  Exit: {corrected.exit_price:.2f}")
    if corrected.realized_pnl is not None:
        color = typer.colors.GREEN if corrected.realized_pnl >= 0 else typer.colors.RED
        typer.secho(f"  Realized P&L: {corrected.realized_pnl:+.2f}", fg=color, bold=True)
    typer.echo("  Correction metadata saved.")


@app.command(name="trade-exclude")
def trade_exclude(
    trade_id: str = typer.Option(..., "--trade-id", help="ID of the trade to exclude from analytics"),
    reason: str = typer.Option(..., "--reason", help="Why this trade should be excluded from analytics"),
) -> None:
    """Alias for excluding a trade from analytics while preserving the journal."""
    from trades.trade_tracker import TradeTracker

    try:
        trade = TradeTracker().mark_excluded(trade_id, reason)
    except ValueError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho("\n=== Trade Excluded From Analytics ===\n", bold=True)
    typer.echo(f"  Trade ID: {trade.trade_id}")
    typer.echo(f"  Reason: {trade.exclusion_reason}")


@app.command(name="trade-mark-excluded")
def trade_mark_excluded(
    trade_id: str = typer.Option(..., "--trade-id", help="ID of the trade to exclude from analytics"),
    reason: str = typer.Option(..., "--reason", help="Why this trade should be excluded from analytics"),
) -> None:
    """Mark a journaled trade as excluded from analytics."""
    from trades.trade_tracker import TradeTracker

    try:
        trade = TradeTracker().mark_excluded(trade_id, reason)
    except ValueError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho("\n=== Trade Marked Excluded ===\n", bold=True)
    typer.echo(f"  Trade ID: {trade.trade_id}")
    typer.echo(f"  Reason: {trade.exclusion_reason}")


@app.command(name="trade-mark-included")
def trade_mark_included(
    trade_id: str = typer.Option(..., "--trade-id", help="ID of the trade to include in analytics"),
) -> None:
    """Mark a journaled trade as included in analytics again."""
    from trades.trade_tracker import TradeTracker

    try:
        trade = TradeTracker().mark_included(trade_id)
    except ValueError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho("\n=== Trade Marked Included ===\n", bold=True)
    typer.echo(f"  Trade ID: {trade.trade_id}")


@app.command(name="trade-close")
def trade_close(
    trade_id:   str   = typer.Option(..., "--trade-id", help="ID of the trade to close"),
    exit_price: float = typer.Option(..., "--exit-price", help="Exit fill price"),
    notes:      str   = typer.Option("", "--notes", help="Optional free-text notes"),
) -> None:
    """
    Close a recorded trade and compute realized P&L (journal only).

    For a long trade: realized_pnl = (exit_price - entry_price) * shares.
    """
    from trades.trade_tracker import TradeTracker

    tracker = TradeTracker()
    try:
        closed = tracker.close_trade(trade_id, exit_price=float(exit_price), notes=notes)
    except ValueError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho("\n=== Trade Closed ===\n", bold=True)
    typer.echo(f"  Trade ID: {closed.trade_id}")
    typer.echo(f"  {closed.ticker} {closed.setup_type}")
    typer.echo(f"  Entry: {closed.entry_price:.2f} | Exit: {closed.exit_price:.2f} | Shares: {closed.shares:g}")
    color = typer.colors.GREEN if (closed.realized_pnl or 0) >= 0 else typer.colors.RED
    typer.secho(f"  Realized P&L: {closed.realized_pnl:+.2f}", fg=color, bold=True)


@app.command(name="trade-analytics")
def trade_analytics(
    include_shadow: bool = typer.Option(
        False, "--include-shadow/--no-include-shadow",
        help="Also show shadow forward-test analytics and the manual-vs-shadow comparison",
    ),
    clean_shadow_only: bool = typer.Option(
        False, "--clean-shadow-only/--no-clean-shadow-only",
        help="With --include-shadow, analyze only CLEAN ENDED shadow sessions with no data gaps",
    ),
) -> None:
    """Analyze closed manual trades (read-only; no broker interaction)."""
    from trades.trade_analytics import TradeAnalytics

    result = TradeAnalytics().analyze()
    overall = result.overall

    if include_shadow:
        typer.secho("\n=== Manual Forward Test ===", bold=True)
    typer.secho("\n=== Overall Performance ===\n", bold=True)
    typer.echo(f"  Excluded trades:    {overall.get('excluded_trades', 0)}")
    if overall["total_trades"] == 0:
        typer.echo("No closed manual trades found.")
        if include_shadow:
            if clean_shadow_only:
                _print_clean_shadow_sample_sections()
            else:
                _print_shadow_analytics_sections(overall)
        return

    typer.echo(f"  Total trades:       {overall['total_trades']}")
    typer.echo(f"  Wins / Losses:      {overall['wins']} / {overall['losses']}")
    typer.echo(f"  Win rate:           {overall['win_rate']:.1%}")
    typer.echo(f"  Total realized P&L: {overall['total_realized_pnl']:+.2f}")
    typer.echo(f"  Average P&L:        {overall['average_pnl']:+.2f}")
    typer.echo(f"  Average winner:     {overall['average_winner']:+.2f}")
    typer.echo(f"  Average loser:      {overall['average_loser']:+.2f}")
    typer.echo(f"  Avg setup conf:     {overall.get('average_setup_confidence', 0.0):.2f}")
    typer.echo(f"  Avg setup score:    {overall.get('average_setup_score', 0.0):.2f}")
    typer.echo(f"  Mistake count:      {overall.get('mistake_count', 0)}")
    pf = overall["profit_factor"]
    pf_text = "inf" if pf == float("inf") else f"{pf:.2f}"
    typer.echo(f"  Profit factor:      {pf_text}")
    typer.echo(f"  Expectancy:         {overall['expectancy']:+.2f}")

    def _print_breakdown(title: str, rows: list[dict], label_key: str) -> None:
        typer.secho(f"\n=== {title} ===\n", bold=True)
        if not rows:
            typer.echo("  (none)")
            return
        typer.echo(f"{label_key:<28} {'Trades':>6} {'Win%':>8} {'P&L':>12} {'Avg':>10} {'PF':>8}")
        typer.echo("-" * 78)
        for row in rows:
            pf_value = row["profit_factor"]
            pf_display = "inf" if pf_value == float("inf") else f"{pf_value:.2f}"
            typer.echo(
                f"{str(row.get(label_key, 'unknown')):<28} "
                f"{row['total_trades']:>6} "
                f"{row['win_rate']:>7.1%} "
                f"{row['total_realized_pnl']:>+12.2f} "
                f"{row['average_pnl']:>+10.2f} "
                f"{pf_display:>8}"
            )

    _print_breakdown("By Setup Type", result.by_setup_type, "setup_type")
    _print_breakdown("By Ticker", result.by_ticker, "ticker")
    _print_breakdown("By Setup Rank", result.by_setup_rank, "setup_rank")
    _print_breakdown("By Test Type", result.by_test_type, "test_type")

    # -- Lifecycle analytics (joins trade_monitor.jsonl history) --
    from trades.trade_lifecycle import TradeLifecycleAnalytics

    lifecycle = TradeLifecycleAnalytics().analyze()

    typer.secho("\n=== By Exit Reason ===\n", bold=True)
    if not lifecycle.by_exit_reason:
        typer.echo("  (none)")
    else:
        typer.echo(f"{'exit_reason':<18} {'Trades':>6} {'Win%':>8} {'P&L':>12} {'Avg':>10} {'Followed':>9}")
        typer.echo("-" * 70)
        for row in lifecycle.by_exit_reason:
            typer.echo(
                f"{row['exit_reason']:<18} "
                f"{row['trades']:>6} "
                f"{row['win_rate']:>7.1%} "
                f"{row['total_pnl']:>+12.2f} "
                f"{row['avg_pnl']:>+10.2f} "
                f"{row['followed_count']:>9}"
            )

    typer.secho("\n=== Monitor Effectiveness ===\n", bold=True)
    if not lifecycle.effectiveness:
        typer.echo("  (no closed trades with lifecycle data)")
    else:
        for row in lifecycle.effectiveness:
            typer.secho(f"{row['label']}:", bold=True)
            typer.echo(f"  Trades:   {row['trades']}")
            typer.echo(f"  Avg P&L:  {row['avg_pnl']:+.2f}")
            typer.echo(f"  Win Rate: {row['win_rate']:.1%}")
            if row.get("avg_time_in_trade_seconds") is not None:
                typer.echo(f"  Avg time in trade: {row['avg_time_in_trade_seconds'] / 60:.1f} min")
            typer.echo("")

    if include_shadow:
        if clean_shadow_only:
            _print_clean_shadow_sample_sections()
        else:
            _print_shadow_analytics_sections(overall)


def _format_minutes(seconds) -> str:
    """Render a seconds duration as minutes, or n/a."""
    if seconds is None:
        return "n/a"
    return f"{float(seconds) / 60:.1f} min"


def _format_reason_mix(counts: dict) -> str:
    """Render an exit-reason/skip-reason counter as 'A=2, B=1' (or 'none')."""
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _shadow_record_data_quality(record: dict) -> str:
    """Return CLEAN/PARTIAL_DATA for a persisted shadow session record."""
    counters = record.get("counters") or {}
    quality = str(counters.get("data_quality") or "").upper()
    if quality:
        return quality
    return "PARTIAL_DATA" if int(counters.get("data_gap_count", 0) or 0) else "CLEAN"


def _sum_shadow_counters(records: list[dict]) -> dict:
    """Sum numeric shadow-session counters across session history records."""
    from trades.shadow_session import shadow_quality_diagnostics
    from data_providers.freshness import aggregate_freshness
    counters_list = [record.get("counters") or {} for record in records]
    quality_details = [shadow_quality_diagnostics(c) for c in counters_list]
    stale_details = []
    for counters in counters_list:
        events = list(counters.get("stale_data_events") or [])
        if not events:
            events = [
                failure for failure in (counters.get("data_quality_failures") or [])
                if str(failure.get("reason", "")).upper() == "STALE_DATA"
            ]
        timestamps = [str(event.get("timestamp")) for event in events if event.get("timestamp")]
        stale_details.append({
            "count": int(counters.get("stale_warning_count", len(events)) or 0),
            "max_age": max(
                [float(event.get("max_bar_age_seconds", 0.0) or 0.0) for event in events]
                + [float(counters.get("max_stale_bar_age_seconds", 0.0) or 0.0)]
            ),
            "first": counters.get("first_stale_timestamp") or (min(timestamps) if timestamps else None),
            "last": counters.get("last_stale_timestamp") or (max(timestamps) if timestamps else None),
            "caused_partial": bool(counters.get("stale_data_caused_partial", bool(events))),
        })
    stale_timestamps = [
        str(detail[key])
        for detail in stale_details
        for key in ("first", "last")
        if detail[key]
    ]
    return {
        "freshness_diagnostics": aggregate_freshness(counters_list),
        "partial_data_reasons": sorted({r for d in quality_details for r in d["partial_data_reasons"]}),
        "normal_data_gap_count": sum(d["normal_data_gap_count"] for d in quality_details),
        "extreme_data_gap_count": sum(d["extreme_data_gap_count"] for d in quality_details),
        "max_iteration_gap_seconds": max([d["max_iteration_gap_seconds"] for d in quality_details] or [0.0]),
        "iterations_total": sum(int(c.get("iterations_total", 0) or 0) for c in counters_list),
        "no_ranked_setup_iterations": sum(int(c.get("no_ranked_setup_iterations", 0) or 0) for c in counters_list),
        "ranked_setup_iterations": sum(int(c.get("ranked_setup_iterations", 0) or 0) for c in counters_list),
        "evaluated_setups": sum(int(c.get("evaluated_setups", 0) or 0) for c in counters_list),
        "data_gap_count": sum(int(c.get("data_gap_count", 0) or 0) for c in counters_list),
        "largest_data_gap_seconds": max(
            [float(c.get("largest_data_gap_seconds", 0.0) or 0.0) for c in counters_list] or [0.0]
        ),
        "stale_warning_count": sum(detail["count"] for detail in stale_details),
        "max_stale_bar_age_seconds": max(
            [detail["max_age"] for detail in stale_details] or [0.0]
        ),
        "first_stale_timestamp": min(stale_timestamps) if stale_timestamps else None,
        "last_stale_timestamp": max(stale_timestamps) if stale_timestamps else None,
        "stale_data_caused_partial": any(detail["caused_partial"] for detail in stale_details),
        "entered": sum(int(c.get("entered", 0) or 0) for c in counters_list),
        "skipped": sum(int(c.get("skipped", 0) or 0) for c in counters_list),
        "closed": sum(int(c.get("closed", 0) or 0) for c in counters_list),
        "open": int((counters_list[-1].get("open", 0) if counters_list else 0) or 0),
        "realized_pnl": sum(float(c.get("realized_pnl", 0.0) or 0.0) for c in counters_list),
    }


def _print_shadow_freshness_diagnostics(records: list[dict]) -> None:
    """Print aggregate provider freshness for clean and partial shadow sessions."""
    from trades.shadow_session import shadow_quality_diagnostic_lines
    clean_records = [record for record in records if _shadow_record_data_quality(record) == "CLEAN"]
    partial_records = [record for record in records if _shadow_record_data_quality(record) != "CLEAN"]
    typer.secho("\n  -- Provider Freshness Diagnostics --", bold=True)
    for label, sample in (("All", records), ("CLEAN", clean_records), ("PARTIAL_DATA", partial_records)):
        totals = _sum_shadow_counters(sample)
        typer.echo(
            f"    {label:<12} sessions={len(sample)} "
            f"stale_warnings={totals['stale_warning_count']} "
            f"max_age={totals['max_stale_bar_age_seconds'] / 60:.1f} min "
            f"stale_caused_partial={'yes' if totals['stale_data_caused_partial'] else 'no'}"
        )
        if totals["first_stale_timestamp"]:
            typer.echo(
                f"      first={totals['first_stale_timestamp']} "
                f"last={totals['last_stale_timestamp']}"
            )
        for line in shadow_quality_diagnostic_lines(totals):
            typer.echo("    " + line)
        from data_providers.freshness import freshness_report_lines
        for line in freshness_report_lines(totals):
            typer.echo("    " + line)
        providers = sorted({
            f"{(r.get('config') or {}).get('provider') or 'unknown'}/{(r.get('config') or {}).get('feed') or 'unknown'}"
            for r in sample
        })
        typer.echo(f"      Provider/feed: {', '.join(providers) or 'none'}")
    typer.echo("    PARTIAL_DATA sessions are invalid for strategy proof and excluded from clean analytics.")


def _clean_shadow_session_records(records: list[dict]) -> list[dict]:
    """Return CLEAN, ENDED, no-gap shadow session history records."""
    clean_records = []
    for record in records:
        counters = record.get("counters") or {}
        if str(record.get("status", "")).upper() != "ENDED":
            continue
        if _shadow_record_data_quality(record) != "CLEAN":
            continue
        if int(counters.get("data_gap_count", 0) or 0) != 0:
            continue
        clean_records.append(record)
    return clean_records


def _shadow_session_watchlist_label(record: dict) -> str:
    """Best-effort watchlist label from shadow session config, if present."""
    config = record.get("config") or {}
    for key in ("watchlist_name", "watchlist"):
        value = config.get(key)
        if value:
            return str(value)
    return "unknown"


def _print_trade_analytics_overall(prefix: str, overall: dict) -> None:
    pf = overall["profit_factor"]
    pf_text = "inf" if pf == float("inf") else f"{pf:.2f}"
    typer.echo(f"  {prefix}Closed trades: {overall['total_trades']}")
    typer.echo(f"  {prefix}Win rate:      {overall['win_rate']:.1%}")
    typer.echo(f"  {prefix}Total P&L:     {overall['total_realized_pnl']:+.2f}")
    typer.echo(f"  {prefix}Average P&L:   {overall['average_pnl']:+.2f}")
    typer.echo(f"  {prefix}Profit factor: {pf_text}")
    typer.echo(f"  {prefix}Expectancy:    {overall['expectancy']:+.2f}")


def _print_shadow_result_breakdown(title: str, rows: list, label_key: str) -> None:
    typer.secho(f"\n  -- {title} --", bold=True)
    if not rows:
        typer.echo("    (none)")
        return
    for row in rows:
        typer.echo(
            f"    {str(row.get(label_key, 'unknown')):<32} "
            f"trades={row.get('total_trades', row.get('trades', 0))} "
            f"win={row['win_rate']:.1%} "
            f"pnl={row.get('total_realized_pnl', row.get('total_pnl', 0.0)):+.2f}"
        )


def _shadow_trade_datetime(value) -> Optional[datetime]:
    """Parse a shadow trade timestamp for diagnostics; invalid values return None."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        return None


def _shadow_entry_bucket(value) -> str:
    """Entry-time bucket used by shadow analytics output."""
    parsed = _shadow_trade_datetime(value)
    if parsed is None:
        return "unknown"
    trade_time = parsed.time()
    if trade_time < dt_time(10, 0):
        return "before 10:00"
    if trade_time < dt_time(11, 0):
        return "10:00-11:00"
    if trade_time < dt_time(12, 0):
        return "11:00-12:00"
    if trade_time < dt_time(13, 0):
        return "12:00-13:00"
    if trade_time < dt_time(14, 0):
        return "13:00-14:00"
    if trade_time < dt_time(15, 0):
        return "14:00-15:00"
    return "after 15:00"


def _shadow_trade_minutes_held(trade) -> Optional[float]:
    """Minutes between entry and exit for a closed shadow trade."""
    entry_dt = _shadow_trade_datetime(getattr(trade, "entry_time", None))
    exit_dt = _shadow_trade_datetime(getattr(trade, "exit_time", None))
    if entry_dt is None or exit_dt is None:
        return None
    return (exit_dt - entry_dt).total_seconds() / 60.0


def _shadow_exit_signal(trade) -> str:
    """Exit reason stored in shadow trade metadata."""
    return str((getattr(trade, "metadata", None) or {}).get("exit_signal") or "unknown")


def _shadow_loss_group_rows(trades: list, label_key: str, key_func) -> list[dict]:
    """Group losing shadow trades by label and sort by total loss ascending."""
    groups = {}
    for trade in trades:
        label = str(key_func(trade) or "unknown")
        groups.setdefault(label, []).append(float(getattr(trade, "realized_pnl", 0.0) or 0.0))
    rows = []
    for label, pnls in groups.items():
        rows.append({
            label_key: label,
            "losses": len(pnls),
            "total_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
        })
    return sorted(rows, key=lambda row: row["total_pnl"])


def _print_shadow_loss_group(title: str, rows: list[dict], label_key: str) -> None:
    typer.echo(f"    {title}:")
    if not rows:
        typer.echo("      (none)")
        return
    for row in rows:
        typer.echo(
            f"      {str(row.get(label_key, 'unknown')):<24} "
            f"losses={row['losses']} pnl={row['total_pnl']:+.2f} avg={row['avg_pnl']:+.2f}"
        )


def _print_shadow_loser_list(trades: list, limit: int = 5) -> None:
    """Print losing shadow trades worst-first."""
    if not trades:
        typer.echo("    (none)")
        return
    for trade in trades[:limit]:
        minutes = _shadow_trade_minutes_held(trade)
        session_id = str((getattr(trade, "metadata", None) or {}).get("session_id") or "unknown")
        exit_signal = _shadow_exit_signal(trade)
        loss_kind = (
            "STOP_HIT loss" if exit_signal == "STOP_HIT"
            else "SESSION_END loss" if exit_signal == "SESSION_END"
            else f"{exit_signal} loss"
        )
        typer.echo(
            f"    {trade.ticker:<6} {str(trade.setup_type):<34} "
            f"entry={trade.entry_time} exit={trade.exit_time} "
            f"bucket={_shadow_entry_bucket(trade.entry_time)} "
            f"exit={exit_signal} pnl={float(trade.realized_pnl or 0.0):+.2f} "
            f"held={minutes:.1f}m session={session_id} ({loss_kind})"
            if minutes is not None else
            f"    {trade.ticker:<6} {str(trade.setup_type):<34} "
            f"entry={trade.entry_time} exit={trade.exit_time} "
            f"bucket={_shadow_entry_bucket(trade.entry_time)} "
            f"exit={exit_signal} pnl={float(trade.realized_pnl or 0.0):+.2f} "
            f"held=n/a session={session_id} ({loss_kind})"
        )


def _print_clean_shadow_large_loser_diagnostics(sample_closed: list) -> None:
    """Print clean-shadow-only large loser diagnostics."""
    losers = sorted(
        [trade for trade in sample_closed if float(getattr(trade, "realized_pnl", 0.0) or 0.0) < 0],
        key=lambda trade: float(getattr(trade, "realized_pnl", 0.0) or 0.0),
    )

    typer.secho("\n  -- Large Loser Diagnostics --", bold=True)
    if not losers:
        typer.echo("    No losing clean shadow trades.")
    else:
        typer.echo("    Largest losing clean shadow trades:")
        _print_shadow_loser_list(losers)
        _print_shadow_loss_group(
            "Largest losses by ticker",
            _shadow_loss_group_rows(losers, "ticker", lambda trade: trade.ticker),
            "ticker",
        )
        _print_shadow_loss_group(
            "Largest losses by setup type",
            _shadow_loss_group_rows(losers, "setup_type", lambda trade: trade.setup_type),
            "setup_type",
        )
        _print_shadow_loss_group(
            "Largest losses by entry time bucket",
            _shadow_loss_group_rows(losers, "entry_time_bucket", lambda trade: _shadow_entry_bucket(trade.entry_time)),
            "entry_time_bucket",
        )
        _print_shadow_loss_group(
            "STOP_HIT losses vs SESSION_END losses",
            _shadow_loss_group_rows(
                [trade for trade in losers if _shadow_exit_signal(trade) in {"STOP_HIT", "SESSION_END"}],
                "exit_reason",
                _shadow_exit_signal,
            ),
            "exit_reason",
        )

    orb_losers = [
        trade for trade in losers
        if str(getattr(trade, "setup_type", "")) == "opening_range_breakout_long"
    ]
    typer.secho("\n  -- Large ORB Losers --", bold=True)
    _print_shadow_loser_list(orb_losers)

    hrv_trades = [
        trade for trade in sample_closed
        if str(getattr(trade, "setup_type", "")) == "high_relative_volume_continuation_long"
    ]
    hrv_losses = [
        trade for trade in hrv_trades
        if float(getattr(trade, "realized_pnl", 0.0) or 0.0) < 0
    ]
    hrv_pnls = [float(getattr(trade, "realized_pnl", 0.0) or 0.0) for trade in hrv_trades]
    hrv_wins = sum(1 for pnl in hrv_pnls if pnl > 0)
    typer.secho("\n  -- HRV Continuation Diagnostics --", bold=True)
    typer.echo(f"    Total HRV trades: {len(hrv_trades)}")
    typer.echo(f"    HRV P&L: {sum(hrv_pnls):+.2f}")
    typer.echo(f"    HRV win rate: {(hrv_wins / len(hrv_trades)):.1%}" if hrv_trades else "    HRV win rate: 0.0%")
    typer.echo(f"    HRV losses: {len(hrv_losses)}")
    avg_loss = sum(float(trade.realized_pnl or 0.0) for trade in hrv_losses) / len(hrv_losses) if hrv_losses else 0.0
    typer.echo(f"    HRV average loss: {avg_loss:+.2f}")
    _print_shadow_loss_group(
        "Worst HRV tickers",
        _shadow_loss_group_rows(hrv_losses, "ticker", lambda trade: trade.ticker),
        "ticker",
    )


def _shadow_trade_at_or_after(trade, hour: int) -> bool:
    """True when a trade entry timestamp is parseable and at/after the hour."""
    parsed = _shadow_trade_datetime(getattr(trade, "entry_time", None))
    return parsed is not None and parsed.time() >= dt_time(hour, 0)


def _shadow_metric_summary(trades: list) -> dict:
    """Use the project's trade analytics math for a list of shadow trades."""
    from trades.trade_analytics import TradeAnalytics

    df = TradeAnalytics._trades_to_dataframe(trades)
    return TradeAnalytics.analyze_dataframe(df).overall


def _format_profit_factor(value: float) -> str:
    return "inf" if value == float("inf") else f"{value:.2f}"


def _clean_shadow_filter_row(name: str, sample_closed: list, predicate, baseline: dict) -> dict:
    """Return baseline-vs-filter metrics for one analytics-only scenario."""
    baseline_pnl = float(baseline.get("total_realized_pnl", 0.0) or 0.0)
    excluded = [trade for trade in sample_closed if predicate(trade)]
    remaining = [trade for trade in sample_closed if not predicate(trade)]
    hypothetical = _shadow_metric_summary(remaining)
    hypothetical_pnl = float(hypothetical.get("total_realized_pnl", 0.0) or 0.0)
    return {
        "name": name,
        "excluded_count": len(excluded),
        "remaining_count": len(remaining),
        "baseline_closed_trades": int(baseline.get("total_trades", 0) or 0),
        "baseline_pnl": baseline_pnl,
        "hypothetical_pnl": hypothetical_pnl,
        "delta": hypothetical_pnl - baseline_pnl,
        "baseline_profit_factor": float(baseline.get("profit_factor", 0.0) or 0.0),
        "win_rate": float(hypothetical.get("win_rate", 0.0) or 0.0),
        "profit_factor": float(hypothetical.get("profit_factor", 0.0) or 0.0),
        "expectancy": float(hypothetical.get("expectancy", 0.0) or 0.0),
    }


def _print_clean_shadow_hypothetical_filter_diagnostics(sample_closed: list) -> None:
    """Print analytics-only hypothetical filters for the clean shadow sample."""
    from collections import defaultdict

    baseline = _shadow_metric_summary(sample_closed)
    baseline_pnl = float(baseline.get("total_realized_pnl", 0.0) or 0.0)

    typer.secho("\n  -- Hypothetical Filter Diagnostics --", bold=True)
    typer.echo(
        "    Baseline: "
        f"closed={baseline['total_trades']} pnl={baseline_pnl:+.2f} "
        f"win={baseline['win_rate']:.1%} "
        f"pf={_format_profit_factor(baseline['profit_factor'])} "
        f"expectancy={baseline['expectancy']:+.2f}"
    )
    typer.secho(
        "    Hypothetical filters are for research only. Do not treat them as strategy changes without more data.",
        fg=typer.colors.YELLOW,
    )

    if not sample_closed:
        typer.echo("    No clean closed shadow trades to filter.")
        return

    pnl_by_ticker = defaultdict(float)
    for trade in sample_closed:
        pnl_by_ticker[str(trade.ticker).upper()] += float(trade.realized_pnl or 0.0)
    worst_tickers = [
        ticker for ticker, pnl in sorted(pnl_by_ticker.items(), key=lambda item: item[1])
        if pnl < 0
    ][:3]

    scenarios = [
        (
            "Exclude high_relative_volume_continuation_long",
            lambda trade: str(trade.setup_type) == "high_relative_volume_continuation_long",
        ),
        (
            "Exclude ORB entries at/after 13:00",
            lambda trade: str(trade.setup_type) == "opening_range_breakout_long"
            and _shadow_trade_at_or_after(trade, 13),
        ),
        (
            "Exclude ORB entries at/after 14:00",
            lambda trade: str(trade.setup_type) == "opening_range_breakout_long"
            and _shadow_trade_at_or_after(trade, 14),
        ),
        (
            "Exclude all entries at/after 13:00",
            lambda trade: _shadow_trade_at_or_after(trade, 13),
        ),
        (
            "Exclude all entries at/after 14:00",
            lambda trade: _shadow_trade_at_or_after(trade, 14),
        ),
        ("Exclude ARM", lambda trade: str(trade.ticker).upper() == "ARM"),
        ("Exclude QCOM", lambda trade: str(trade.ticker).upper() == "QCOM"),
        ("Exclude CRWD", lambda trade: str(trade.ticker).upper() == "CRWD"),
        (
            "Exclude ARM + QCOM",
            lambda trade: str(trade.ticker).upper() in {"ARM", "QCOM"},
        ),
        (
            "Exclude ARM + QCOM + CRWD",
            lambda trade: str(trade.ticker).upper() in {"ARM", "QCOM", "CRWD"},
        ),
        (
            "Exclude top 3 worst tickers by clean P&L",
            lambda trade: str(trade.ticker).upper() in set(worst_tickers),
        ),
    ]

    rows = []
    for name, predicate in scenarios:
        rows.append(_clean_shadow_filter_row(name, sample_closed, predicate, baseline))

    rows.sort(key=lambda row: row["delta"], reverse=True)
    typer.echo("    Scenarios sorted by P&L improvement:")
    for row in rows:
        typer.echo(
            f"    {row['name']:<48} "
            f"excluded={row['excluded_count']:<2} remaining={row['remaining_count']:<2} "
            f"baseline={row['baseline_pnl']:+.2f} hypothetical={row['hypothetical_pnl']:+.2f} "
            f"delta={row['delta']:+.2f} win={row['win_rate']:.1%} "
            f"pf={_format_profit_factor(row['profit_factor'])} expectancy={row['expectancy']:+.2f}"
        )


def _candidate_rule_interpretation(row: dict) -> str:
    """Simple research-only interpretation for a candidate filter row."""
    if row["delta"] > 0 and row["profit_factor"] > row["baseline_profit_factor"]:
        return "HELPFUL so far"
    if row["delta"] < 0:
        return "HARMFUL so far"
    return "MIXED/INSUFFICIENT"


def _print_clean_shadow_candidate_rule_tracking(sample_closed: list) -> None:
    """Track fixed candidate rules across future clean shadow samples."""
    baseline = _shadow_metric_summary(sample_closed)

    typer.secho("\n  -- Candidate Rule Tracking --", bold=True)
    typer.secho(
        "    Candidate rules are research-only and are not applied to trading behavior.",
        fg=typer.colors.YELLOW,
    )
    if not sample_closed:
        typer.echo("    No clean closed shadow trades to evaluate.")
        return

    candidate_rules = [
        (
            "candidate_no_entries_after_13",
            lambda trade: _shadow_trade_at_or_after(trade, 13),
        ),
        (
            "candidate_no_orb_after_13",
            lambda trade: str(trade.setup_type) == "opening_range_breakout_long"
            and _shadow_trade_at_or_after(trade, 13),
        ),
        (
            "candidate_exclude_arm_qcom_crwd",
            lambda trade: str(trade.ticker).upper() in {"ARM", "QCOM", "CRWD"},
        ),
        (
            "candidate_exclude_arm_qcom",
            lambda trade: str(trade.ticker).upper() in {"ARM", "QCOM"},
        ),
        (
            "candidate_no_hrv",
            lambda trade: str(trade.setup_type) == "high_relative_volume_continuation_long",
        ),
    ]

    for name, predicate in candidate_rules:
        row = _clean_shadow_filter_row(name, sample_closed, predicate, baseline)
        typer.echo(
            f"    {row['name']:<34} "
            f"baseline_closed={row['baseline_closed_trades']} "
            f"remaining={row['remaining_count']} excluded={row['excluded_count']} "
            f"baseline_pnl={row['baseline_pnl']:+.2f} candidate_pnl={row['hypothetical_pnl']:+.2f} "
            f"delta={row['delta']:+.2f} "
            f"baseline_pf={_format_profit_factor(row['baseline_profit_factor'])} "
            f"candidate_pf={_format_profit_factor(row['profit_factor'])} "
            f"candidate_win={row['win_rate']:.1%} "
            f"candidate_expectancy={row['expectancy']:+.2f} "
            f"interpretation={_candidate_rule_interpretation(row)}"
        )


def _print_clean_shadow_sample_sections() -> None:
    """
    Print filtered shadow analytics for CLEAN, ENDED, no-gap sessions only.

    Read-only: does not modify shadow trades, manual trades, or session logs.
    """
    from collections import Counter

    from trades.shadow_session import (
        SHADOW_SESSIONS_FILE,
        SHADOW_TRADES_FILE,
        shadow_entry_time_breakdown,
        shadow_exit_reason_breakdown,
        shadow_orb_no_progress_diagnostics,
        shadow_stop_hit_diagnostics,
    )
    from trades.trade_analytics import TradeAnalytics
    from trades.trade_tracker import TradeStatus, TradeTracker

    session_records = _load_review_jsonl(Path(SHADOW_SESSIONS_FILE))
    clean_records = _clean_shadow_session_records(session_records)
    clean_session_ids = {
        str(record.get("session_id")) for record in clean_records if record.get("session_id")
    }

    typer.secho("\n=== Clean Shadow Sample ===\n", bold=True)
    typer.echo("  SIMULATION ONLY — no real orders were placed.")
    typer.echo("  Filter: status=ENDED, data_quality=CLEAN, data_gaps=0")
    _print_shadow_freshness_diagnostics(session_records)

    if not clean_session_ids:
        typer.echo("  No clean completed shadow sessions found.")
        typer.echo("  Clean shadow sample is still small; avoid overfitting until 30-50 closed clean trades.")
        return

    all_shadow_trades = TradeTracker(storage_path=SHADOW_TRADES_FILE).list_all_trades()
    sample_trades = [
        trade for trade in all_shadow_trades
        if str((trade.metadata or {}).get("session_id")) in clean_session_ids
    ]
    sample_closed = [
        trade for trade in sample_trades
        if str(trade.status).upper() == TradeStatus.CLOSED.value
    ]
    sample_open_count = len(sample_trades) - len(sample_closed)
    sample_df = TradeAnalytics._trades_to_dataframe(sample_trades)
    sample_result = TradeAnalytics.analyze_dataframe(sample_df)
    overall = sample_result.overall
    counters = _sum_shadow_counters(clean_records)
    watchlists = Counter(_shadow_session_watchlist_label(record) for record in clean_records)

    typer.echo(f"  Clean sessions: {len(clean_records)}")
    typer.echo(f"  Session IDs: {', '.join(sorted(clean_session_ids))}")
    typer.echo(f"  Watchlists: {_format_reason_mix(watchlists)}")
    if "unknown" in watchlists:
        typer.echo("  Note: some older shadow sessions do not store a watchlist name.")
    typer.echo(f"  Clean session iterations: {counters['iterations_total']}")
    typer.echo(f"  Clean evaluated setups:  {counters['evaluated_setups']}")
    typer.echo(f"  Open trades:             {sample_open_count}")
    _print_trade_analytics_overall("", overall)
    if overall["total_trades"] < 30:
        typer.secho(
            "  Clean shadow sample is still small; avoid overfitting until 30-50 closed clean trades.",
            fg=typer.colors.YELLOW,
        )

    _print_shadow_result_breakdown("By Setup Type", sample_result.by_setup_type, "setup_type")
    _print_shadow_result_breakdown("By Ticker", sample_result.by_ticker, "ticker")
    _print_shadow_result_breakdown(
        "By Exit Reason", shadow_exit_reason_breakdown(sample_closed), "exit_reason",
    )

    typer.secho("\n  -- By Entry Time --", bold=True)
    for row in shadow_entry_time_breakdown(sample_closed):
        profit_factor = row["profit_factor"]
        pf_text = "inf" if profit_factor == float("inf") else f"{profit_factor:.2f}"
        typer.echo(
            f"    {row['entry_time_bucket']:<16} "
            f"trades={row['trades']} "
            f"win={row['win_rate']:.1%} "
            f"pnl={row['total_pnl']:+.2f} "
            f"avg={row['avg_pnl']:+.2f} "
            f"pf={pf_text}"
        )

    _print_clean_shadow_large_loser_diagnostics(sample_closed)
    _print_clean_shadow_hypothetical_filter_diagnostics(sample_closed)
    _print_clean_shadow_candidate_rule_tracking(sample_closed)

    stop_diag = shadow_stop_hit_diagnostics(sample_closed)
    stop_hit = stop_diag["stop_hit"]
    non_stop = stop_diag["non_stop_hit"]
    typer.secho("\n  -- Stop-Hit Diagnostics --", bold=True)
    typer.echo(
        "    STOP_HIT:     "
        f"count={stop_hit['trades']} pnl={stop_hit['total_pnl']:+.2f} "
        f"avg={stop_hit['avg_pnl']:+.2f} avg_time={_format_minutes(stop_hit['avg_time_seconds'])}"
    )
    typer.echo(
        "    non-STOP_HIT: "
        f"count={non_stop['trades']} pnl={non_stop['total_pnl']:+.2f} "
        f"avg={non_stop['avg_pnl']:+.2f} avg_time={_format_minutes(non_stop['avg_time_seconds'])}"
    )

    orb_diag = shadow_orb_no_progress_diagnostics(sample_closed)
    typer.secho("\n  -- ORB No-Progress Diagnostics --", bold=True)
    typer.echo(f"    ORB closed trades: {orb_diag['orb_closed_trades']}")
    for key in ("30", "60"):
        row = orb_diag["milestones"].get(key, {})
        typer.echo(
            f"    {key}-min snapshots: {row.get('snapshot_count', 0)} | "
            f"no-progress: {row.get('no_progress_count', 0)} | "
            f"actual P&L: {row.get('no_progress_actual_pnl', 0.0):+.2f} | "
            f"hypothetical P&L: {row.get('no_progress_hypothetical_pnl', 0.0):+.2f} | "
            f"delta: {row.get('candidate_pnl_delta', 0.0):+.2f}"
        )


def _print_shadow_analytics_sections(manual_overall: dict) -> None:
    """
    Print Shadow Forward Test / Manual vs Shadow / Shadow Decisions sections.

    Read-only: loads logs/shadow_trades.jsonl and logs/shadow_decisions.jsonl;
    never touches the manual journal. Missing shadow logs print empty sections.
    """
    from collections import Counter

    from trades.shadow_session import (
        SHADOW_TRADES_FILE,
        SHADOW_SESSIONS_FILE,
        ShadowSessionState,
        ShadowDecisionLog,
        average_time_in_trade_seconds,
        shadow_after_15_summary,
        shadow_entry_time_breakdown,
        shadow_exit_reason_breakdown,
        shadow_orb_no_progress_diagnostics,
        shadow_stop_hit_diagnostics,
    )
    from trades.trade_analytics import TradeAnalytics
    from trades.trade_lifecycle import TradeLifecycleAnalytics
    from trades.trade_tracker import TradeStatus, TradeTracker

    shadow_tracker = TradeTracker(storage_path=SHADOW_TRADES_FILE)
    shadow_trades = shadow_tracker.list_all_trades()
    shadow_closed = [t for t in shadow_trades if str(t.status).upper() == TradeStatus.CLOSED.value]
    shadow_open_count = len(shadow_trades) - len(shadow_closed)
    shadow_result = TradeAnalytics(storage_path=SHADOW_TRADES_FILE).analyze()
    shadow_overall = shadow_result.overall
    shadow_state = ShadowSessionState.load(Path("state/shadow_session.json"))

    # -- Shadow Forward Test --
    typer.secho("\n=== Shadow Forward Test ===\n", bold=True)
    typer.echo("  SIMULATION ONLY — no real orders were placed.")
    if shadow_state is not None:
        counters = shadow_state.counters or {}
        from trades.shadow_session import shadow_quality_diagnostic_lines
        for line in shadow_quality_diagnostic_lines(counters):
            typer.echo(line)
        from data_providers.freshness import freshness_report_lines
        for line in freshness_report_lines(counters):
            typer.echo(line)
        typer.echo(f"  Provider/feed: {shadow_state.config.get('provider') or 'unknown'}/{shadow_state.config.get('feed') or 'unknown'}")
        typer.echo(f"  Latest shadow session: {shadow_state.session_id} [{shadow_state.status}]")
        typer.echo(f"  Iterations:            {int(counters.get('iterations_total', 0) or 0)}")
        typer.echo(f"  Data quality:          {counters.get('data_quality', 'CLEAN')}")
        typer.echo(f"  Data gaps:             {int(counters.get('data_gap_count', 0) or 0)}")
        typer.echo(
            "  Largest data gap:      "
            f"{float(counters.get('largest_data_gap_seconds', 0.0) or 0.0) / 60:.1f} min"
        )
        typer.echo(f"  Stale warnings:        {int(counters.get('stale_warning_count', 0) or 0)}")
        typer.echo(
            "  Max stale bar age:      "
            f"{float(counters.get('max_stale_bar_age_seconds', 0.0) or 0.0) / 60:.1f} min"
        )
        if counters.get("first_stale_timestamp"):
            typer.echo(f"  First stale timestamp: {counters.get('first_stale_timestamp')}")
            typer.echo(f"  Last stale timestamp:  {counters.get('last_stale_timestamp')}")
            typer.echo(
                "  Stale caused partial:   "
                f"{'yes' if counters.get('stale_data_caused_partial') else 'no'}"
            )
        typer.echo(
            "  No-ranked-setup iterations: "
            f"{int(counters.get('no_ranked_setup_iterations', 0) or 0)}"
        )
        typer.echo("  No real orders placed.")
    today = datetime.now().date()
    session_records_today = [
        record for record in _load_review_jsonl(Path(SHADOW_SESSIONS_FILE))
        if any(
            (parsed := _parse_review_datetime(record.get(key))) is not None
            and parsed.date() == today
            for key in ("timestamp", "last_iteration_at", "started_at")
        )
    ]
    if session_records_today:
        today_totals = _sum_shadow_counters(session_records_today)
        clean_records_today = [
            record for record in session_records_today
            if _shadow_record_data_quality(record) == "CLEAN"
        ]
        partial_records_today = [
            record for record in session_records_today
            if _shadow_record_data_quality(record) != "CLEAN"
        ]
        clean_totals = _sum_shadow_counters(clean_records_today)
        typer.echo(f"  Today's shadow sessions: {len(session_records_today)}")
        typer.echo(f"  Today's CLEAN sessions:  {len(clean_records_today)}")
        typer.echo(f"  Today's PARTIAL_DATA sessions: {len(partial_records_today)}")
        typer.echo(
            "  Today's iterations:      "
            f"{today_totals['iterations_total']}"
        )
        typer.echo(
            "  Today's evaluated setups: "
            f"{today_totals['evaluated_setups']}"
        )
        today_gap_count = today_totals["data_gap_count"]
        today_largest_gap = today_totals["largest_data_gap_seconds"]
        today_quality = "PARTIAL_DATA" if partial_records_today or today_gap_count else "CLEAN"
        typer.echo(f"  Today's data quality:  {today_quality}")
        typer.echo(f"  Today's data gaps:     {today_gap_count}")
        typer.echo(f"  Today's largest gap:   {today_largest_gap / 60:.1f} min")
        typer.echo(f"  Today's stale warnings: {today_totals['stale_warning_count']}")
        typer.echo(
            "  Today's max stale age:  "
            f"{today_totals['max_stale_bar_age_seconds'] / 60:.1f} min"
        )
        typer.echo(
            "  Stale caused partial:   "
            f"{'yes' if today_totals['stale_data_caused_partial'] else 'no'}"
        )
        _print_shadow_freshness_diagnostics(session_records_today)
        if clean_records_today:
            typer.echo(
                "  Today's CLEAN-only:    "
                f"iterations={clean_totals['iterations_total']} "
                f"setups={clean_totals['evaluated_setups']} "
                f"entered={clean_totals['entered']} "
                f"closed={clean_totals['closed']} "
                f"pnl={clean_totals['realized_pnl']:+.2f}"
            )
    typer.echo(f"  Total shadow trades: {len(shadow_trades)}")
    typer.echo(f"  Closed / Open:       {len(shadow_closed)} / {shadow_open_count}")
    if shadow_overall["total_trades"] == 0:
        typer.echo("  No closed shadow trades yet.")
    else:
        pf = shadow_overall["profit_factor"]
        pf_text = "inf" if pf == float("inf") else f"{pf:.2f}"
        typer.echo(f"  Win rate:            {shadow_overall['win_rate']:.1%}")
        typer.echo(f"  Total sim P&L:       {shadow_overall['total_realized_pnl']:+.2f}")
        typer.echo(f"  Average P&L:         {shadow_overall['average_pnl']:+.2f}")
        typer.echo(f"  Profit factor:       {pf_text}")
        typer.echo(f"  Expectancy:          {shadow_overall['expectancy']:+.2f}")

        def _print_shadow_breakdown(title: str, rows: list, label_key: str) -> None:
            typer.secho(f"\n  -- {title} --", bold=True)
            if not rows:
                typer.echo("    (none)")
                return
            for row in rows:
                typer.echo(
                    f"    {str(row.get(label_key, 'unknown')):<32} "
                    f"trades={row.get('total_trades', row.get('trades', 0))} "
                    f"win={row['win_rate']:.1%} "
                    f"pnl={row.get('total_realized_pnl', row.get('total_pnl', 0.0)):+.2f}"
                )

        _print_shadow_breakdown("By Setup Type", shadow_result.by_setup_type, "setup_type")
        _print_shadow_breakdown("By Ticker", shadow_result.by_ticker, "ticker")
        _print_shadow_breakdown(
            "By Exit Reason", shadow_exit_reason_breakdown(shadow_closed), "exit_reason",
        )

    typer.secho("\n  -- By Entry Time --", bold=True)
    for row in shadow_entry_time_breakdown(shadow_closed):
        profit_factor = row["profit_factor"]
        pf_text = "inf" if profit_factor == float("inf") else f"{profit_factor:.2f}"
        typer.echo(
            f"    {row['entry_time_bucket']:<16} "
            f"trades={row['trades']} "
            f"win={row['win_rate']:.1%} "
            f"pnl={row['total_pnl']:+.2f} "
            f"avg={row['avg_pnl']:+.2f} "
            f"pf={pf_text}"
        )
    late_day = shadow_after_15_summary(shadow_closed)
    typer.secho("\n  -- Late-Day Summary --", bold=True)
    typer.echo(f"    After 15:00 trades:   {late_day['trades']}")
    typer.echo(f"    After 15:00 P&L:      {late_day['total_pnl']:+.2f}")
    typer.echo(f"    After 15:00 win rate: {late_day['win_rate']:.1%}")

    stop_diag = shadow_stop_hit_diagnostics(shadow_closed)
    stop_hit = stop_diag["stop_hit"]
    non_stop = stop_diag["non_stop_hit"]

    typer.secho("\n  -- Stop-Hit Diagnostics --", bold=True)
    typer.echo(
        "    STOP_HIT:     "
        f"count={stop_hit['trades']} "
        f"pnl={stop_hit['total_pnl']:+.2f} "
        f"avg={stop_hit['avg_pnl']:+.2f} "
        f"avg_time={_format_minutes(stop_hit['avg_time_seconds'])}"
    )
    typer.echo(
        "    non-STOP_HIT: "
        f"count={non_stop['trades']} "
        f"pnl={non_stop['total_pnl']:+.2f} "
        f"avg={non_stop['avg_pnl']:+.2f} "
        f"avg_time={_format_minutes(non_stop['avg_time_seconds'])}"
    )

    def _print_stop_breakdown(title: str, rows: list, label_key: str) -> None:
        typer.echo(f"    {title}:")
        if not rows:
            typer.echo("      (none)")
            return
        for row in rows:
            typer.echo(
                f"      {str(row.get(label_key, 'unknown')):<24} "
                f"trades={row['trades']} "
                f"win={row['win_rate']:.1%} "
                f"pnl={row['total_pnl']:+.2f} "
                f"avg={row['avg_pnl']:+.2f}"
            )

    _print_stop_breakdown("By setup type", stop_diag["by_setup_type"], "setup_type")
    _print_stop_breakdown("By ticker", stop_diag["by_ticker"], "ticker")
    _print_stop_breakdown("By entry time", stop_diag["by_entry_time"], "entry_time_bucket")

    typer.echo("    Largest stop-hit losses:")
    if not stop_diag["largest_losses"]:
        typer.echo("      (none)")
    else:
        for trade in stop_diag["largest_losses"]:
            typer.echo(
                f"      {trade['ticker']:<6} {trade['setup_type']:<32} "
                f"entry={trade['entry_time']} exit={trade['exit_time']} "
                f"pnl={trade['realized_pnl']:+.2f} "
                f"time={_format_minutes(trade['time_in_trade_seconds'])}"
            )

    orb_diag = shadow_orb_no_progress_diagnostics(shadow_closed)
    typer.secho("\n  -- ORB No-Progress Diagnostics --", bold=True)
    typer.echo(f"    ORB closed trades: {orb_diag['orb_closed_trades']}")
    for key in ("30", "60"):
        row = orb_diag["milestones"].get(key, {})
        typer.echo(
            f"    {key}-min snapshots: {row.get('snapshot_count', 0)} | "
            f"no-progress: {row.get('no_progress_count', 0)} | "
            f"actual P&L: {row.get('no_progress_actual_pnl', 0.0):+.2f} | "
            f"hypothetical P&L: {row.get('no_progress_hypothetical_pnl', 0.0):+.2f} | "
            f"delta: {row.get('candidate_pnl_delta', 0.0):+.2f}"
        )
        typer.echo(
            f"      eventual STOP_HIT flagged: {row.get('eventual_stop_hit_flagged', 0)} | "
            f"eventual TARGET_1_HIT flagged: {row.get('eventual_target_1_hit_flagged', 0)}"
        )

    # -- Manual vs Shadow --
    manual_lifecycles = TradeLifecycleAnalytics().build_lifecycles()
    manual_followed_pct = (
        sum(1 for lc in manual_lifecycles if lc.exit_action_followed) / len(manual_lifecycles)
        if manual_lifecycles else None
    )
    manual_durations = [
        lc.time_in_trade_seconds for lc in manual_lifecycles
        if lc.time_in_trade_seconds is not None
    ]
    manual_avg_time = sum(manual_durations) / len(manual_durations) if manual_durations else None
    manual_exit_mix = Counter(lc.exit_reason for lc in manual_lifecycles)
    shadow_exit_mix = Counter(
        str((t.metadata or {}).get("exit_signal") or "MANUAL") for t in shadow_closed
    )
    shadow_avg_time = average_time_in_trade_seconds(shadow_closed)

    typer.secho("\n=== Manual vs Shadow ===\n", bold=True)
    typer.echo(f"  {'Metric':<24} {'Manual':>16} {'Shadow':>16}")
    typer.echo("  " + "-" * 58)
    typer.echo(f"  {'Closed trades':<24} {manual_overall['total_trades']:>16} {shadow_overall['total_trades']:>16}")
    typer.echo(f"  {'Total P&L':<24} {manual_overall['total_realized_pnl']:>+16.2f} {shadow_overall['total_realized_pnl']:>+16.2f}")
    typer.echo(f"  {'Win rate':<24} {manual_overall['win_rate']:>15.1%} {shadow_overall['win_rate']:>15.1%}")
    typer.echo(f"  {'Avg P&L':<24} {manual_overall['average_pnl']:>+16.2f} {shadow_overall['average_pnl']:>+16.2f}")
    typer.echo(f"  {'Avg time in trade':<24} {_format_minutes(manual_avg_time):>16} {_format_minutes(shadow_avg_time):>16}")
    typer.echo(f"  Exit-reason mix (manual): {_format_reason_mix(manual_exit_mix)}")
    typer.echo(f"  Exit-reason mix (shadow): {_format_reason_mix(shadow_exit_mix)}")
    followed_text = f"{manual_followed_pct:.1%}" if manual_followed_pct is not None else "n/a"
    typer.echo(f"  Manual trades that followed first exit signal: {followed_text}")
    typer.echo("  (Shadow trades follow exit signals 100% by construction.)")

    # -- Shadow Decisions --
    decision_summary = ShadowDecisionLog().summarize()
    typer.secho("\n=== Shadow Decisions ===\n", bold=True)
    typer.echo(f"  ENTERED: {decision_summary['entered']}")
    typer.echo(f"  SKIPPED: {decision_summary['skipped']}")
    if decision_summary["skip_reasons"]:
        typer.echo("  Skipped by reason:")
        for reason, count in decision_summary["skip_reasons"].items():
            typer.echo(f"    {reason}: {count}")
    else:
        typer.echo("  Skipped by reason: (none)")


@app.command(name="daily-plan")
def daily_plan() -> None:
    """
    Print a read-only daily operating summary and suggested next commands.

    This command only reads local settings, watchlists, setup-alert logs, and
    manual-trade logs. It never fetches market data and never touches brokers.
    """
    from config.settings import settings
    from trades.trade_tracker import TradeStatus, TradeTracker

    def _parse_dt(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
        except ValueError:
            return None

    typer.secho("\n=== Daily Plan ===\n", bold=True)

    typer.secho("Safety Status", bold=True)
    typer.echo(f"  PAPER_TRADING:        {settings.PAPER_TRADING}")
    typer.echo(f"  LIVE_TRADING_ALLOWED: {settings.LIVE_TRADING_ALLOWED}")
    typer.echo(f"  ALPACA_PAPER:         {settings.ALPACA_PAPER}")
    typer.echo(f"  Risk lock exists:     {'yes' if settings.is_risk_locked() else 'no'}")

    typer.secho("\nWatchlists", bold=True)
    watchlist_names = get_available_watchlists()
    if not watchlist_names:
        typer.echo("  (none found)")
    else:
        for name in watchlist_names:
            try:
                tickers = load_watchlist(name)
            except typer.Exit:
                continue
            typer.echo(f"  {name}: {len(tickers)} tickers")

    typer.secho("\nRecent Setup Alerts", bold=True)
    recent_alerts = _filter_setup_alert_records(
        _load_setup_alert_records(),
        since_hours=4,
        valid_only=True,
        exclude_synthetic=True,
    )
    typer.echo(f"  Count in last 4 hours: {len(recent_alerts)}")
    newest_alert = recent_alerts[0] if recent_alerts else None
    if newest_alert:
        typer.echo(
            "  Newest: "
            f"{str(newest_alert.get('ticker', '')).upper()} "
            f"{newest_alert.get('setup_type', '')} "
            f"confidence={float(newest_alert.get('confidence', 0.0)):.2f}"
        )
    else:
        typer.echo("  Newest: none")

    tracker = TradeTracker()
    open_trades = tracker.list_open_trades()
    all_trades = tracker.list_all_trades()

    typer.secho("\nOpen Trades", bold=True)
    typer.echo(f"  Count: {len(open_trades)}")
    if not open_trades:
        typer.echo("  (none)")
    else:
        for trade in open_trades:
            typer.echo(
                f"  {trade.ticker:<8} entry {trade.entry_price:.2f} "
                f"stop {trade.stop_loss:.2f} shares {trade.shares:g}"
            )

    today = datetime.now().date()
    closed_today = [
        trade for trade in all_trades
        if trade.status == TradeStatus.CLOSED.value
        and _parse_dt(trade.exit_time) is not None
        and _parse_dt(trade.exit_time).date() == today
    ]
    pnl_today = sum(float(trade.realized_pnl or 0.0) for trade in closed_today)

    typer.secho("\nToday's Closed Trades / P&L", bold=True)
    typer.echo(f"  Closed trades today: {len(closed_today)}")
    typer.echo(f"  Realized P&L today:  {pnl_today:+.2f}")

    suggested_ticker = str(newest_alert.get("ticker", "TICKER")).upper() if newest_alert else "TICKER"
    recommended_provider, recommended_feed, provider_warning = _recommended_intraday_provider()
    provider_args = f"--provider {recommended_provider}"
    feed_args = f" --alpaca-feed {recommended_feed}" if recommended_provider == "alpaca" else ""

    typer.secho("\nSuggested Commands", bold=True)
    typer.echo("  Recommended live scan:")
    typer.echo(
        "  - "
        f"python main.py setup-scan-loop --watchlist mega_caps --timeframe 5Min "
        f"{provider_args}{feed_args} --interval-seconds 300"
        + (" --discord-alerts" if recommended_provider == "alpaca" else "")
    )
    typer.echo("  Recommended monitor:")
    typer.echo(
        "  - "
        f"python main.py trade-monitor {provider_args}{feed_args} "
        "--timeframe 1Min --loop --interval-seconds 60"
        + (" --discord-alerts" if recommended_provider == "alpaca" else "")
    )
    if provider_warning:
        typer.echo(f"  {provider_warning}")

    commands = [
        "python main.py setup-alerts-recent --since-hours 4",
        f'python main.py trade-add-latest-alert --ticker {suggested_ticker} --shares SHARES --notes "entered manually"',
        "python main.py trade-analytics",
        "python main.py dashboard",
    ]
    for command in commands:
        typer.echo(f"  - {command}")


@app.command(name="morning-start")
def morning_start() -> None:
    """
    Print a read-only morning workflow checklist with exact commands to run.

    This command only reads local settings and configuration status. It does not
    start scanners, fetch market data, contact brokers, or place orders.
    """
    from config.settings import settings
    from alerts.discord_alerts import is_discord_configured

    recommended_provider, recommended_feed, provider_warning = _recommended_intraday_provider()
    provider_args = f"--provider {recommended_provider}"
    feed_args = f" --alpaca-feed {recommended_feed}" if recommended_provider == "alpaca" else ""
    discord_enabled = is_discord_configured()
    discord_flag = " --discord-alerts" if discord_enabled else ""
    daily_review_discord = " --discord" if discord_enabled else ""

    typer.secho("\n=== Morning Start ===\n", bold=True)

    typer.secho("Safety Status", bold=True)
    typer.echo(f"  PAPER_TRADING:        {settings.PAPER_TRADING}")
    typer.echo(f"  LIVE_TRADING_ALLOWED: {settings.LIVE_TRADING_ALLOWED}")
    typer.echo(f"  ALPACA_PAPER:         {settings.ALPACA_PAPER}")
    typer.echo(f"  Risk lock exists:     {'yes' if settings.is_risk_locked() else 'no'}")

    typer.secho("\nProvider Status", bold=True)
    if recommended_provider == "alpaca":
        typer.echo("  Recommended provider: Alpaca IEX")
    else:
        typer.echo("  Recommended provider: YFinance")
    if provider_warning:
        typer.echo(f"  Warning: {provider_warning}")
    if discord_enabled:
        typer.echo("  Discord alerts: configured")
    else:
        typer.echo("  Warning: Discord webhook missing; Discord alert flags are omitted.")

    typer.secho("\nExact Commands", bold=True)
    commands = [
        (
            "python main.py setup-scan-loop --watchlist mega_caps --timeframe 5Min "
            f"{provider_args}{feed_args} --interval-seconds 300{discord_flag}"
        ),
        "python main.py setup-alerts-recent --since-hours 4 --limit 10",
        (
            'python main.py trade-add-latest-alert --ticker TICKER --shares SHARES '
            '--notes "forward test" --test-type forward_test --thesis "..." '
            '--market-context "..." --entry-quality A'
        ),
        (
            f"python main.py trade-monitor {provider_args}{feed_args} "
            f"--timeframe 1Min --loop --interval-seconds 60{discord_flag}"
        ),
        "python main.py trade-analytics",
        f"python main.py daily-review{daily_review_discord}",
    ]
    for command in commands:
        typer.echo(f"  {command}")

    typer.secho("\nReminder", bold=True)
    typer.echo("  Replace TICKER and SHARES before adding a trade.")


def _parse_review_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-ish datetime for local log review commands."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except (TypeError, ValueError):
        return None


def _load_review_jsonl(path: Path) -> list[dict]:
    """Load JSONL records, skipping malformed lines."""
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _records_on_date(records: list[dict], date_value, timestamp_key: str = "timestamp") -> list[dict]:
    """Return JSONL records whose timestamp_key falls on date_value."""
    selected = []
    for record in records:
        ts = _parse_review_datetime(record.get(timestamp_key))
        if ts is not None and ts.date() == date_value:
            selected.append(record)
    return selected


def _build_daily_review_lines(review_date) -> list[str]:
    """Build the daily review text as Markdown-friendly plain lines."""
    from collections import Counter
    from trades.trade_analytics import TradeAnalytics
    from trades.trade_tracker import TradeStatus, TradeTracker

    logs_dir = Path("logs")
    lines: list[str] = []

    lines.extend(["# Daily Review", "", "## Date Reviewed", f"  {review_date.isoformat()}"])

    setup_alerts = _records_on_date(_load_review_jsonl(logs_dir / "setup_alerts.jsonl"), review_date)
    lines.extend(["", "## Setup Alerts Today", f"  Count: {len(setup_alerts)}"])
    if setup_alerts:
        by_ticker = Counter(str(alert.get("ticker", "")).upper() for alert in setup_alerts if alert.get("ticker"))
        by_setup = Counter(str(alert.get("setup_type", "")) for alert in setup_alerts if alert.get("setup_type"))
        lines.append("  By ticker: " + ", ".join(f"{ticker}={count}" for ticker, count in sorted(by_ticker.items())))
        lines.append("  By setup_type: " + ", ".join(f"{setup}={count}" for setup, count in sorted(by_setup.items())))
        ranked = [
            alert for alert in setup_alerts
            if alert.get("rank") is not None
        ]
        top_alert = min(ranked, key=lambda alert: int(alert.get("rank", 999999))) if ranked else setup_alerts[-1]
        lines.append(
            "  Top ranked alert: "
            f"{str(top_alert.get('ticker', '')).upper()} "
            f"{top_alert.get('setup_type', '')} "
            f"rank={top_alert.get('rank', '-')}"
        )
    else:
        lines.append("  (none)")

    cooldown_summary = _daily_setup_cooldown_summary(review_date)
    lines.extend(["", "## Setup Cooldowns"])
    cooldown_lines = []
    for setup_type, count in sorted(cooldown_summary.get("cooled_setup_types", {}).items()):
        cooldown_lines.append(f"  {setup_type}: cooled down due to {count} failed/excluded trades")
    for ticker, count in sorted(cooldown_summary.get("cooled_tickers", {}).items()):
        cooldown_lines.append(f"  {ticker}: cooled down due to {count} failed/excluded trades")
    if cooldown_lines:
        lines.extend(cooldown_lines)
    else:
        lines.append("  (none)")

    tracker = TradeTracker()
    all_trades = tracker.list_all_trades()
    open_trades = tracker.list_open_trades()

    opened_today = [
        trade for trade in all_trades
        if _parse_review_datetime(trade.entry_time) is not None
        and _parse_review_datetime(trade.entry_time).date() == review_date
    ]
    lines.extend(["", "## Trades Taken Today", f"  Count: {len(opened_today)}"])
    if opened_today:
        for trade in opened_today:
            thesis = f" thesis={trade.thesis}" if trade.thesis else ""
            lines.append(
                f"  {trade.ticker:<8} {trade.setup_type} entry {trade.entry_price:.2f} "
                f"stop {trade.stop_loss:.2f} shares {trade.shares:g} "
                f"test_type={trade.test_type}{thesis}"
            )
    else:
        lines.append("  (none)")

    closed_today = [
        trade for trade in all_trades
        if trade.status == TradeStatus.CLOSED.value
        and _parse_review_datetime(trade.exit_time) is not None
        and _parse_review_datetime(trade.exit_time).date() == review_date
    ]
    lines.extend(["", "## Closed Trades Today", f"  Count: {len(closed_today)}"])
    if closed_today:
        for trade in closed_today:
            lines.append(
                f"  {trade.ticker:<8} realized P&L {float(trade.realized_pnl or 0.0):+.2f} "
                f"notes={trade.notes or '-'}"
            )
    else:
        lines.append("  (none)")

    lines.extend(["", "## Open Trades", f"  Count: {len(open_trades)}"])
    if open_trades:
        for trade in open_trades:
            lines.append(
                f"  {trade.ticker:<8} entry {trade.entry_price:.2f} "
                f"stop {trade.stop_loss:.2f} shares {trade.shares:g}"
            )
    else:
        lines.append("  (none)")

    important_actions = {"STOP_HIT", "TARGET_1_HIT", "TARGET_2_HIT", "EXIT_VWAP_LOST", "WATCH"}
    monitor_alerts = [
        record for record in _records_on_date(_load_review_jsonl(logs_dir / "trade_monitor.jsonl"), review_date)
        if str(record.get("action", "")).upper() in important_actions
    ]
    lines.extend(["", "## Monitor Alerts Today"])
    action_counts = Counter(str(record.get("action", "")).upper() for record in monitor_alerts)
    if action_counts:
        lines.append("  Counts: " + ", ".join(f"{action}={action_counts.get(action, 0)}" for action in sorted(important_actions)))
    else:
        lines.append("  Counts: none")
    latest_by_ticker: Dict[str, dict] = {}
    for record in monitor_alerts:
        ticker = str(record.get("ticker", "")).upper()
        if ticker:
            latest_by_ticker[ticker] = record
    if latest_by_ticker:
        for ticker, record in sorted(latest_by_ticker.items()):
            lines.append(f"  {ticker}: {record.get('action', '')} - {record.get('reason', '')}")
    else:
        lines.append("  Latest important alert: none")

    included_closed_today = [trade for trade in closed_today if not bool(trade.exclude_from_analytics)]
    excluded_closed_today = [trade for trade in closed_today if bool(trade.exclude_from_analytics)]
    performance_df = None
    if included_closed_today or excluded_closed_today:
        import pandas as pd
        performance_df = pd.DataFrame([trade.to_dict() for trade in closed_today])
    analytics = TradeAnalytics.analyze_dataframe(performance_df) if performance_df is not None else TradeAnalytics.analyze_dataframe(None)
    overall = analytics.overall

    lines.extend([
        "",
        "## Performance Summary",
        f"  Total P&L today: {float(overall.get('total_realized_pnl', 0.0)):+.2f}",
        f"  Win rate today: {float(overall.get('win_rate', 0.0)):.1%}",
        f"  Expectancy today: {float(overall.get('expectancy', 0.0)):+.2f}",
        f"  Excluded trades: {len(excluded_closed_today)}",
    ])

    lines.extend(_build_shadow_review_lines(review_date, logs_dir))

    lines.extend(["", "## Notes / Next Questions"])
    for question in [
        "Which setup looked best?",
        "Did VWAP-loss exits help?",
        "Were any trades mistakes?",
        "Did Alpaca data look current?",
        "What should be improved next?",
    ]:
        lines.append(f"  - {question}")

    return lines


def _build_shadow_review_lines(review_date, logs_dir: Path) -> list[str]:
    """
    Build the '## Shadow Forward Test' daily-review section (simulation only).

    Returns [] when no shadow logs or same-day session state exist, so
    reviews without shadow sessions are unchanged.
    """
    from collections import Counter

    from trades.shadow_session import ShadowSessionState, summarize_shadow_decision_records
    from trades.trade_tracker import TradeStatus, TradeTracker

    shadow_trades_path = logs_dir / "shadow_trades.jsonl"
    shadow_decisions_path = logs_dir / "shadow_decisions.jsonl"
    shadow_sessions_path = logs_dir / "shadow_sessions.jsonl"
    shadow_state = ShadowSessionState.load(Path("state/shadow_session.json"))

    def _record_is_for_review_date(record: dict) -> bool:
        for key in ("timestamp", "last_iteration_at", "started_at"):
            parsed = _parse_review_datetime(record.get(key))
            if parsed is not None and parsed.date() == review_date:
                return True
        return False

    session_records_today = [
        record for record in _load_review_jsonl(shadow_sessions_path)
        if _record_is_for_review_date(record)
    ]

    def _state_is_for_review_date() -> bool:
        if shadow_state is None:
            return False
        for key in ("last_iteration_at", "started_at"):
            parsed = _parse_review_datetime(getattr(shadow_state, key, None))
            if parsed is not None and parsed.date() == review_date:
                return True
        return False

    has_shadow_logs = shadow_trades_path.exists() or shadow_decisions_path.exists()
    has_shadow_state_today = _state_is_for_review_date()
    has_session_history_today = bool(session_records_today)
    if not has_shadow_logs and not has_shadow_state_today and not has_session_history_today:
        return []

    shadow_all = (
        TradeTracker(storage_path=shadow_trades_path).list_all_trades()
        if shadow_trades_path.exists() else []
    )
    entered_today = [
        t for t in shadow_all
        if _parse_review_datetime(t.entry_time) is not None
        and _parse_review_datetime(t.entry_time).date() == review_date
    ]
    closed_today = [
        t for t in shadow_all
        if str(t.status).upper() == TradeStatus.CLOSED.value
        and _parse_review_datetime(t.exit_time) is not None
        and _parse_review_datetime(t.exit_time).date() == review_date
    ]
    open_now = [t for t in shadow_all if str(t.status).upper() == TradeStatus.OPEN.value]

    sim_pnl = sum(float(t.realized_pnl or 0.0) for t in closed_today)
    wins = sum(1 for t in closed_today if float(t.realized_pnl or 0.0) > 0)
    win_rate = wins / len(closed_today) if closed_today else 0.0

    reported_session_ids = {
        str(record.get("session_id"))
        for record in session_records_today
        if record.get("session_id")
    }
    if not reported_session_ids and has_shadow_state_today and shadow_state is not None:
        reported_session_ids = {shadow_state.session_id}

    all_decision_records = _load_review_jsonl(shadow_decisions_path)
    decision_records = list(all_decision_records)
    if reported_session_ids:
        decision_records = [
            record for record in decision_records
            if str(record.get("session_id")) in reported_session_ids
        ]
        decisions = summarize_shadow_decision_records(decision_records, None)
    else:
        decisions = summarize_shadow_decision_records(decision_records, review_date)
    top_reasons = list(decisions["skip_reasons"].items())[:3]
    reasons_text = (
        ", ".join(f"{reason}={count}" for reason, count in top_reasons)
        if top_reasons else "none"
    )
    if has_session_history_today:
        status_counts = Counter(str(record.get("status", "unknown")) for record in session_records_today)
        clean_session_records = [
            record for record in session_records_today
            if _shadow_record_data_quality(record) == "CLEAN"
        ]
        partial_session_records = [
            record for record in session_records_today
            if _shadow_record_data_quality(record) != "CLEAN"
        ]
        counters = _sum_shadow_counters(session_records_today)
        clean_counters = _sum_shadow_counters(clean_session_records)
        partial_counters = _sum_shadow_counters(partial_session_records)
        iterations_total = counters["iterations_total"]
        no_ranked_iterations = counters["no_ranked_setup_iterations"]
        ranked_iterations = counters["ranked_setup_iterations"]
        evaluated_setups = counters["evaluated_setups"]
        data_gap_count = counters["data_gap_count"]
        largest_data_gap_seconds = counters["largest_data_gap_seconds"]
        stale_warning_count = counters["stale_warning_count"]
        max_stale_bar_age_seconds = counters["max_stale_bar_age_seconds"]
        first_stale_timestamp = counters["first_stale_timestamp"]
        last_stale_timestamp = counters["last_stale_timestamp"]
        stale_data_caused_partial = counters["stale_data_caused_partial"]
        data_quality = "PARTIAL_DATA" if partial_session_records or data_gap_count else "CLEAN"
        entered_count = counters["entered"]
        skipped_count = counters["skipped"]
        closed_count = max(len(closed_today), counters["closed"])
        open_count = max(len(open_now), counters["open"])
        session_id = f"{len(session_records_today)} session(s)"
        status = ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items()))
        if any("realized_pnl" in (record.get("counters") or {}) for record in session_records_today):
            sim_pnl = counters["realized_pnl"]
        clean_session_ids = {
            str(record.get("session_id")) for record in clean_session_records if record.get("session_id")
        }
        clean_decisions = summarize_shadow_decision_records([
            record for record in all_decision_records
            if str(record.get("session_id")) in clean_session_ids
        ], None)
        clean_top_reasons = list(clean_decisions["skip_reasons"].items())[:3]
        clean_reasons_text = (
            ", ".join(f"{reason}={count}" for reason, count in clean_top_reasons)
            if clean_top_reasons else "none"
        )
        clean_closed_today = [
            trade for trade in closed_today
            if str((trade.metadata or {}).get("session_id")) in clean_session_ids
        ]
        clean_win_rate = (
            sum(1 for trade in clean_closed_today if float(trade.realized_pnl or 0.0) > 0) / len(clean_closed_today)
            if clean_closed_today else 0.0
        )
    else:
        clean_session_records = []
        partial_session_records = []
        clean_counters = {}
        partial_counters = {}
        clean_reasons_text = "none"
        clean_win_rate = 0.0
        counters = shadow_state.counters if has_shadow_state_today and shadow_state else {}
        iterations_total = int(counters.get("iterations_total", 0) or 0)
        no_ranked_iterations = int(counters.get("no_ranked_setup_iterations", 0) or 0)
        ranked_iterations = int(counters.get("ranked_setup_iterations", 0) or 0)
        evaluated_setups = int(counters.get("evaluated_setups", 0) or 0)
        data_gap_count = int(counters.get("data_gap_count", 0) or 0)
        largest_data_gap_seconds = float(counters.get("largest_data_gap_seconds", 0.0) or 0.0)
        stale_warning_count = int(counters.get("stale_warning_count", 0) or 0)
        max_stale_bar_age_seconds = float(counters.get("max_stale_bar_age_seconds", 0.0) or 0.0)
        first_stale_timestamp = counters.get("first_stale_timestamp")
        last_stale_timestamp = counters.get("last_stale_timestamp")
        stale_data_caused_partial = bool(counters.get("stale_data_caused_partial", False))
        data_quality = str(counters.get("data_quality", "PARTIAL_DATA" if data_gap_count else "CLEAN"))
        entered_count = max(len(entered_today), int(counters.get("entered", 0) or 0))
        skipped_count = max(int(decisions["skipped"]), int(counters.get("skipped", 0) or 0))
        closed_count = max(len(closed_today), int(counters.get("closed", 0) or 0))
        open_count = max(len(open_now), int(counters.get("open", 0) or 0))
        session_id = shadow_state.session_id if has_shadow_state_today and shadow_state else "unknown"
        status = shadow_state.status if has_shadow_state_today and shadow_state else "LOGS_ONLY"
    all_shadow_counts_zero = all(
        int(value) == 0
        for value in (
            iterations_total,
            no_ranked_iterations,
            ranked_iterations,
            evaluated_setups,
            data_gap_count,
            entered_count,
            skipped_count,
            closed_count,
            open_count,
        )
    ) and abs(float(sim_pnl or 0.0)) < 0.000001
    if (
        (status == "LOGS_ONLY" or session_id == "unknown")
        and all_shadow_counts_zero
        and not decisions["skip_reasons"]
    ):
        return [
            "",
            "## Shadow Forward Test",
            "  No shadow session was run today.",
            "  Simulation only - no real orders placed.",
        ]
    if evaluated_setups == 0 and no_ranked_iterations > 0:
        result_text = "No ranked setups appeared during the session."
    elif evaluated_setups > 0:
        result_text = "Ranked setups were evaluated during the session."
    else:
        result_text = "Shadow activity recorded."

    lines = [
        "",
        "## Shadow Forward Test",
        "  (Simulation only — no real orders placed.)",
        f"  Sessions: {len(session_records_today) if has_session_history_today else 1}",
        f"  CLEAN sessions: {len(clean_session_records) if has_session_history_today else (1 if data_quality == 'CLEAN' else 0)}",
        f"  PARTIAL_DATA sessions: {len(partial_session_records) if has_session_history_today else (1 if data_quality != 'CLEAN' else 0)}",
        f"  Session: {session_id} [{status}]",
        f"  Session statuses: {status}",
        f"  Iterations: {iterations_total}",
        f"  No-ranked-setup iterations: {no_ranked_iterations}",
        f"  Ranked-setup iterations: {ranked_iterations}",
        f"  Evaluated setups: {evaluated_setups}",
        f"  Data quality: {data_quality}",
        f"  Data gaps: {data_gap_count}",
        f"  Largest data gap: {largest_data_gap_seconds / 60:.1f} min",
        f"  Stale warnings: {stale_warning_count}",
        f"  Max stale bar age: {max_stale_bar_age_seconds / 60:.1f} min",
        f"  Entered: {entered_count}",
        f"  Skipped: {skipped_count}",
        f"  Closed: {closed_count}",
        f"  Open: {open_count}",
        f"  Sim P&L: {sim_pnl:+.2f}",
        f"  Win Rate: {win_rate:.1%}",
        f"  Top skip reasons: {reasons_text}",
        f"  Result: {result_text}",
        "  Simulation only - no real orders placed.",
    ]
    from trades.shadow_session import shadow_quality_diagnostic_lines
    lines.extend(shadow_quality_diagnostic_lines(counters))
    from data_providers.freshness import freshness_report_lines
    lines.extend(freshness_report_lines(counters))
    report_configs = [r.get("config") or {} for r in session_records_today] if has_session_history_today else [shadow_state.config if shadow_state else {}]
    lines.append("  Provider/feed: " + ", ".join(sorted({
        f"{c.get('provider') or 'unknown'}/{c.get('feed') or 'unknown'}" for c in report_configs
    })))
    if first_stale_timestamp:
        lines.extend([
            f"  First stale timestamp: {first_stale_timestamp}",
            f"  Last stale timestamp: {last_stale_timestamp}",
            "  Stale data caused PARTIAL_DATA: "
            f"{'yes' if stale_data_caused_partial else 'no'}",
        ])
    if data_quality != "CLEAN":
        lines.append("  PARTIAL_DATA sessions are invalid for strategy proof and excluded from clean analytics.")
    if has_session_history_today and clean_session_records:
        lines.extend([
            "  CLEAN-only stats:",
            f"    Clean iterations: {clean_counters['iterations_total']}",
            f"    Clean evaluated setups: {clean_counters['evaluated_setups']}",
            f"    Clean entered: {clean_counters['entered']}",
            f"    Clean skipped: {clean_counters['skipped']}",
            f"    Clean closed: {clean_counters['closed']}",
            f"    Clean open: {clean_counters['open']}",
            f"    Clean Sim P&L: {clean_counters['realized_pnl']:+.2f}",
            f"    Clean Win Rate: {clean_win_rate:.1%}",
            f"    Clean top skip reasons: {clean_reasons_text}",
        ])
    if has_session_history_today and partial_session_records:
        lines.extend([
            "  PARTIAL_DATA stats:",
            f"    Partial iterations: {partial_counters['iterations_total']}",
            f"    Partial evaluated setups: {partial_counters['evaluated_setups']}",
            f"    Partial entered: {partial_counters['entered']}",
            f"    Partial Sim P&L: {partial_counters['realized_pnl']:+.2f}",
            f"    Partial stale warnings: {partial_counters['stale_warning_count']}",
            "    Partial max stale bar age: "
            f"{partial_counters['max_stale_bar_age_seconds'] / 60:.1f} min",
            "    Stale data caused PARTIAL_DATA: "
            f"{'yes' if partial_counters['stale_data_caused_partial'] else 'no'}",
        ])
    return lines


@app.command(name="daily-review")
def daily_review(
    save: bool = typer.Option(True, "--save/--no-save", help="Save review to logs/daily_reviews/YYYY-MM-DD.md"),
    date: Optional[str] = typer.Option(None, "--date", help="Review a specific date YYYY-MM-DD instead of today"),
    discord: bool = typer.Option(False, "--discord/--no-discord", help="Optionally send the daily review summary to Discord"),
) -> None:
    """
    Print and optionally save a read-only end-of-day review from local logs.

    This command never fetches market data, contacts brokers, or places orders.
    """
    if date:
        try:
            review_date = datetime.fromisoformat(date).date()
        except ValueError:
            typer.secho("Error: --date must be YYYY-MM-DD.", fg=typer.colors.RED)
            raise typer.Exit(1)
    else:
        review_date = datetime.now().date()

    lines = _build_daily_review_lines(review_date)
    typer.echo("\n".join(lines))

    if save:
        review_dir = Path("logs/daily_reviews")
        review_dir.mkdir(parents=True, exist_ok=True)
        review_file = review_dir / f"{review_date.isoformat()}.md"
        review_file.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        typer.secho(f"\nSaved daily review to: {review_file}", fg=typer.colors.GREEN)

    if discord:
        from alerts.discord_alerts import (
            format_daily_review_discord_message,
            is_discord_configured,
            send_discord_message,
        )

        if not is_discord_configured():
            typer.secho("Discord webhook not configured; daily review not sent.", fg=typer.colors.YELLOW)
            return

        message = format_daily_review_discord_message("\n".join(lines))
        if send_discord_message(message):
            typer.secho("Discord daily review sent.", fg=typer.colors.GREEN)
        else:
            typer.secho("Discord daily review failed to send.", fg=typer.colors.RED)


@app.command(name="discord-test")
def discord_test() -> None:
    """
    Send a safe Discord webhook test message.

    This command only verifies notification configuration. It never places
    orders and never prints the webhook URL.
    """
    from alerts.discord_alerts import get_last_discord_error, is_discord_configured, send_discord_message

    typer.secho("\n=== Discord Alert Test ===\n", bold=True)
    if not is_discord_configured():
        typer.secho("Discord webhook is not configured.", fg=typer.colors.YELLOW)
        typer.echo("Set DISCORD_WEBHOOK_URL in your local .env to enable Discord alerts.")
        return

    sent = send_discord_message("\n".join([
        "\U0001f9ea **DISCORD TEST ALERT**",
        "\u2501" * 20,
        "AI Regime Trader Discord alerts are configured correctly.",
    ]))
    if sent:
        typer.secho("Discord test alert sent successfully.", fg=typer.colors.GREEN)
    else:
        typer.secho("Discord test alert failed. Check webhook configuration and network access.", fg=typer.colors.RED)
        failure_reason = get_last_discord_error()
        if failure_reason:
            typer.echo(f"Reason: {failure_reason}")


@app.command(name="trade-monitor")
def trade_monitor(
    timeframe: str = typer.Option("1Min", "--timeframe", help="Bar size: 1Day, 1Min, 5Min, 15Min"),
    lookback_days: Optional[int] = typer.Option(None, "--lookback-days", help="Days of history to load. yfinance intraday limits apply."),
    provider_name: Optional[str] = typer.Option(None, "--provider", help="Market data provider: yfinance, alpaca, synthetic, csv"),
    alpaca_feed: str = typer.Option("iex", "--alpaca-feed", help="Alpaca market data feed: iex or sip (default iex/free feed)"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Use synthetic data (offline, no network required)"),
    csv_file: str = typer.Option(None, "--csv", help="Load OHLCV from a CSV file for single-ticker/offline monitoring"),
    watchlist: str = typer.Option(None, "--watchlist", help="Optional extra watchlist to load; open trades determine monitored tickers"),
    loop: bool = typer.Option(False, "--loop", help="Repeat monitoring until stopped"),
    interval_seconds: int = typer.Option(60, "--interval-seconds", help="Wait this many seconds between loop iterations"),
    max_iterations: int = typer.Option(None, "--max-iterations", help="Stop after this many iterations"),
    alert_banner: bool = typer.Option(True, "--alert-banner/--no-alert-banner", help="Print alert banners for new loop alerts"),
    discord_alerts: bool = typer.Option(True, "--discord-alerts/--no-discord-alerts", help="Send Discord alerts when webhook is configured"),
    allow_synthetic_real_tickers: bool = typer.Option(
        False,
        "--allow-synthetic-real-tickers",
        help="Allow fake synthetic data to monitor real ticker trades (disabled by default).",
    ),
) -> None:
    """
    Monitor open manual trades and print advisory HOLD/WATCH/EXIT/TARGET status.

    This is a journal + market-data tool only. It never submits orders and never
    contacts broker execution.
    """
    import pandas as pd

    from data.feature_engineering import add_features
    from data_providers.base import DataProviderError
    from trades.trade_monitor import TradeMonitor
    from trades.trade_tracker import TradeTracker
    from trades.trade_monitor import TradeMonitorAction
    from alerts.discord_alerts import (
        format_trade_monitor_discord_alert,
        is_discord_configured,
        send_discord_message,
    )

    selected_provider, recommended_feed, provider_warning = _recommended_intraday_provider(provider_name)
    if provider_name is None and selected_provider == "alpaca":
        alpaca_feed = recommended_feed
    if synthetic:
        selected_provider = "synthetic"
    if csv_file:
        selected_provider = "csv"

    if selected_provider == "synthetic" and loop and max_iterations is None:
        typer.secho(
            "Error: --synthetic loop mode requires --max-iterations to prevent infinite loops.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    user_passed_lookback = lookback_days is not None
    lookback_days = _validate_intraday_lookback(
        timeframe,
        lookback_days or 400,
        selected_provider == "synthetic",
        csv_file if selected_provider == "csv" else None,
        user_passed_lookback,
    )

    tracker = TradeTracker()
    monitor = TradeMonitor()
    try:
        provider = _get_named_data_provider(
            selected_provider, synthetic=synthetic, csv_file=csv_file, alpaca_feed=alpaca_feed
        )
    except typer.BadParameter as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    monitor_log = logs_dir / "trade_monitor.jsonl"

    iteration = 0
    alerted_trade_actions = set()
    important_actions = {
        TradeMonitorAction.WATCH,
        TradeMonitorAction.TARGET_1_HIT,
        TradeMonitorAction.TARGET_2_HIT,
        TradeMonitorAction.STOP_HIT,
        TradeMonitorAction.EXIT_VWAP_LOST,
    }
    discord_enabled = bool(discord_alerts and is_discord_configured())
    if discord_enabled:
        typer.echo("Discord alerts: enabled")
    else:
        typer.echo("Discord alerts: not configured")
    typer.echo(f"Data provider: {_data_provider_display_name(selected_provider, provider.provider_name, alpaca_feed)}")
    if provider_warning and selected_provider == "yfinance":
        typer.echo(provider_warning)

    def run_once() -> None:
        open_trades = tracker.list_open_trades()
        timestamp_now = datetime.now().isoformat()

        typer.secho("\n=== Trade Monitor ===\n", bold=True)
        typer.echo(f"  Timeframe: {timeframe}")
        typer.echo(f"  Provider: {provider.provider_name}")
        typer.echo(f"  Data provider: {_data_provider_display_name(selected_provider, provider.provider_name, alpaca_feed)}")
        typer.echo(f"  Open trades: {len(open_trades)}\n")

        if not open_trades:
            typer.echo("  (no open manual trades)")
            return

        if selected_provider == "synthetic" and not allow_synthetic_real_tickers:
            real_ticker_trades = [trade for trade in open_trades if trade.ticker.upper() != "SYNTHETIC"]
            if real_ticker_trades:
                real_symbols = sorted({trade.ticker.upper() for trade in real_ticker_trades})
                typer.secho(
                    "Warning: synthetic data is fake and may not match real tickers: "
                    + ", ".join(real_symbols),
                    fg=typer.colors.YELLOW,
                )
                typer.echo("Skipped real ticker trades in synthetic mode. Use --allow-synthetic-real-tickers to override.")
                open_trades = [trade for trade in open_trades if trade.ticker.upper() == "SYNTHETIC"]
                if not open_trades:
                    typer.echo("No compatible open trades to monitor.")
                    return

        trade_tickers = [trade.ticker.upper() for trade in open_trades]
        extra_tickers = load_watchlist(watchlist) if watchlist else []
        tickers = []
        seen = set()
        for ticker_value in trade_tickers + extra_tickers:
            symbol = ticker_value.strip().upper()
            if symbol and symbol not in seen:
                tickers.append(symbol)
                seen.add(symbol)

        try:
            if selected_provider == "synthetic":
                data_by_ticker = provider.get_multiple_bars(tickers, timeframe=timeframe)
            elif selected_provider == "csv":
                label = Path(csv_file).stem.upper()
                df = provider.get_bars(label, timeframe=timeframe)
                data_by_ticker = {ticker_value: df.copy() for ticker_value in trade_tickers}
            elif selected_provider == "alpaca" and hasattr(provider, "get_latest_bar"):
                data_by_ticker = {
                    ticker_value: provider.get_latest_bar(ticker_value, timeframe=timeframe)
                    for ticker_value in tickers
                }
            else:
                end = None if selected_provider == "alpaca" else str(pd.Timestamp.now().date())
                start = str((pd.Timestamp.now() - pd.Timedelta(days=lookback_days)).date())
                data_by_ticker = provider.get_multiple_bars(
                    tickers, start=start, end=end, timeframe=timeframe,
                )
        except DataProviderError as exc:
            _print_data_provider_error(exc, indent="  ")
            raise typer.Exit(1)

        freshness_by_ticker = {}
        if selected_provider == "alpaca":
            timeframe_minutes = {"1Min": 1, "5Min": 5}.get(timeframe, 1)
            stale_threshold_seconds = max(180, timeframe_minutes * 3 * 60)
            for ticker_value, df in data_by_ticker.items():
                metadata = {
                    "latest_bar_time": df.attrs.get("latest_bar_time"),
                    "provider_time": df.attrs.get("provider_time"),
                    "data_age_seconds": df.attrs.get("data_age_seconds"),
                }
                freshness_by_ticker[ticker_value.upper()] = {
                    key: value for key, value in metadata.items() if value is not None
                }
                data_age_seconds = metadata.get("data_age_seconds")
                if data_age_seconds is not None and float(data_age_seconds) > stale_threshold_seconds:
                    age_minutes = float(data_age_seconds) / 60.0
                    typer.secho(
                        f"Warning: Alpaca latest bar appears stale: {age_minutes:.1f} minutes old",
                        fg=typer.colors.YELLOW,
                    )

        featured_data = {}
        for ticker_value, df in data_by_ticker.items():
            try:
                featured_data[ticker_value.upper()] = add_features(df)
            except Exception:
                featured_data[ticker_value.upper()] = df

        results = monitor.monitor_open_trades(open_trades, featured_data)
        for result in results:
            freshness = freshness_by_ticker.get(result.ticker.upper())
            if freshness:
                result.metadata["market_data"] = freshness

        typer.echo(
            f"{'Ticker':<8} {'Action':<16} {'Current':>10} {'R':>8} {'Unrl P&L':>12}  Reason / Guidance"
        )
        typer.echo("-" * 120)
        for result in results:
            current = f"{result.current_price:.2f}" if result.current_price > 0 else "-"
            typer.echo(
                f"{result.ticker:<8} {result.action.value:<16} {current:>10} "
                f"{result.r_multiple:>8.2f} {result.unrealized_pnl:>12.2f}  {result.reason}"
            )
            typer.echo(f"  Guidance: {result.guidance}")
            for warning in result.warnings:
                typer.secho(f"  warning: {warning}", fg=typer.colors.YELLOW)

        with open(monitor_log, "a", encoding="utf-8") as f:
            for result in results:
                record = {"timestamp": timestamp_now, **result.to_dict()}
                f.write(json.dumps(record) + "\n")

        if loop:
            from alerts.local_alerts import print_trade_monitor_alert_banner

            new_alerts = []
            repeated_alerts = 0
            for result in results:
                if result.action not in important_actions:
                    continue
                alert_key = (result.trade_id, result.action.value)
                if alert_key in alerted_trade_actions:
                    repeated_alerts += 1
                    continue
                alerted_trade_actions.add(alert_key)
                new_alerts.append(result)

            for result in new_alerts:
                if alert_banner:
                    print_trade_monitor_alert_banner(result)
                if discord_enabled:
                    send_discord_message(format_trade_monitor_discord_alert(result))

            if repeated_alerts:
                typer.echo(f"Repeated trade monitor alerts suppressed: {repeated_alerts}")

        typer.secho(f"\nResults logged to: {monitor_log}", fg=typer.colors.GREEN)
        typer.echo("No API keys or secrets were included in the log.")

    try:
        while True:
            iteration += 1
            if max_iterations is not None and iteration > max_iterations:
                break

            if loop:
                typer.echo(f"\nIteration {iteration} @ {datetime.now().isoformat()}")
            run_once()

            if not loop or (max_iterations is not None and iteration >= max_iterations):
                break
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        typer.secho("\nTrade monitor stopped by user (Ctrl+C).", fg=typer.colors.YELLOW)
        raise typer.Exit(0)


@app.command(name="log-archive")
def log_archive() -> None:
    """Archive target JSONL logs without modifying originals."""
    from utils.log_maintenance import TARGET_LOG_FILES, archive_log_file

    typer.secho("\n=== Log Archive ===\n", bold=True)
    for path in TARGET_LOG_FILES:
        archived = archive_log_file(path)
        if archived is None:
            typer.echo(f"{path}: missing, skipped")
        else:
            typer.echo(f"{path}: archived -> {archived}")


@app.command(name="log-cleanup")
def log_cleanup(
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Preview cleanup without modifying logs"),
    apply: bool = typer.Option(False, "--apply", help="Archive original logs, then remove test/demo/synthetic records"),
    archive_only: bool = typer.Option(False, "--archive-only", help="Archive selected logs without rewriting them"),
    logs: Optional[List[str]] = typer.Option(None, "--logs", help="Specific log path/name to process; repeatable"),
) -> None:
    """
    Archive and optionally remove obvious test/demo/synthetic JSONL records.

    Defaults to dry-run. --apply always archives before rewriting. --archive-only
    archives originals and leaves them untouched.
    """
    from utils.log_maintenance import (
        archive_log_file,
        filter_test_records,
        load_jsonl_records,
        resolve_log_paths,
        write_jsonl_records,
    )

    selected_logs = resolve_log_paths(logs)
    typer.secho("\n=== Log Cleanup ===\n", bold=True)

    if archive_only:
        mode = "ARCHIVE ONLY"
    elif apply:
        mode = "APPLY"
    else:
        mode = "DRY RUN"
    typer.echo(f"Mode: {mode}")

    for path in selected_logs:
        records = load_jsonl_records(path)
        kept, removed = filter_test_records(records)
        typer.echo(
            f"{path}: total={len(records)} test/demo/synthetic={len(removed)} remain={len(kept)}"
        )

        if not Path(path).exists():
            typer.echo("  missing, skipped")
            continue

        if archive_only:
            archived = archive_log_file(path)
            typer.echo(f"  archived -> {archived}" if archived else "  archive skipped")
            continue

        if apply:
            archived = archive_log_file(path)
            typer.echo(f"  archived -> {archived}" if archived else "  archive skipped")
            write_jsonl_records(path, kept)
            typer.echo("  rewritten with kept records")
        elif dry_run:
            typer.echo("  dry-run only; no changes made")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
