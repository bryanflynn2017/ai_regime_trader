"""
ai_regime_trader — Streamlit monitoring dashboard.

READ-ONLY: This dashboard does NOT place orders, enable live trading,
or expose API keys/secrets.  It is for monitoring and inspection only.

Launch:
    streamlit run dashboard/app.py

Requires:
    pip install streamlit plotly
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Add project root to sys.path so imports work regardless of cwd
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Streamlit guard: fail clearly if not installed
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    import plotly.express as px
except ImportError:
    print(
        "\nStreamlit is not installed.\n"
        "Install it with:\n"
        "  pip install streamlit plotly\n"
        "Then run:\n"
        "  streamlit run dashboard/app.py\n"
    )
    sys.exit(1)

import pandas as pd

from dashboard.helpers import (
    get_recent_rejections,
    list_result_files,
    load_jsonl,
    load_latest_result,
    load_state,
    mask_or_exclude_sensitive_values,
    risk_lock_status,
    summarize_decisions,
    split_manual_trades,
    summarize_shadow_decisions,
    summarize_trade_analytics,
    summarize_trade_lifecycle,
    summarize_manual_trades,
    summarize_setup_scans,
    tail_file,
)

# ---------------------------------------------------------------------------
# Paths (resolved from project root — works regardless of launch directory)
# ---------------------------------------------------------------------------
DECISIONS_LOG  = _PROJECT_ROOT / "logs" / "paper_decisions.jsonl"
SETUP_SCANS_LOG = _PROJECT_ROOT / "logs" / "setup_scans.jsonl"
MANUAL_TRADES_LOG = _PROJECT_ROOT / "logs" / "manual_trades.jsonl"
TRADE_MONITOR_LOG = _PROJECT_ROOT / "logs" / "trade_monitor.jsonl"
SHADOW_TRADES_LOG = _PROJECT_ROOT / "logs" / "shadow_trades.jsonl"
SHADOW_DECISIONS_LOG = _PROJECT_ROOT / "logs" / "shadow_decisions.jsonl"
STATE_FILE     = _PROJECT_ROOT / "state" / "paper_state.json"
RISK_LOCK_FILE = _PROJECT_ROOT / "state" / "RISK_LOCKED.txt"

LOG_FILES = {
    "paper_decisions.jsonl": DECISIONS_LOG,
    "setup_scans.jsonl":     SETUP_SCANS_LOG,
    "manual_trades.jsonl":    MANUAL_TRADES_LOG,
    "trade_monitor.jsonl":    TRADE_MONITOR_LOG,
    "shadow_trades.jsonl":    SHADOW_TRADES_LOG,
    "shadow_decisions.jsonl": SHADOW_DECISIONS_LOG,
    "system.log":            _PROJECT_ROOT / "logs" / "system.log",
    "signals.log":           _PROJECT_ROOT / "logs" / "signals.log",
    "orders.log":            _PROJECT_ROOT / "logs" / "orders.log",
    "risk.log":              _PROJECT_ROOT / "logs" / "risk.log",
    "backtest.log":          _PROJECT_ROOT / "logs" / "backtest.log",
}


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title   = "ai_regime_trader Dashboard",
    page_icon    = "📈",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)


# ---------------------------------------------------------------------------
# Sidebar: auto-refresh and nav
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("ai_regime_trader")
    st.caption("Educational paper trading only.")
    st.markdown("---")

    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        import time
        st.caption(f"Last refresh: {time.strftime('%H:%M:%S')}")
        st.rerun  # will be called after main() runs

    n_decisions = st.slider("Recent decisions to show", 10, 500, 100, step=10)
    n_log_lines = st.slider("Log tail lines",           10, 500, 100, step=10)
    st.markdown("---")

    st.markdown("**Quick links**")
    st.markdown("[Phase 9 docs](#)")
    st.caption("v0.9.0-Phase9")


# ---------------------------------------------------------------------------
# Safety banner (always visible)
# ---------------------------------------------------------------------------
st.warning(
    "**DASHBOARD IS READ-ONLY.** "
    "It does NOT place trades, enable live trading, or expose API secrets. "
    "For educational monitoring use only. Not financial advice."
)

st.title("ai_regime_trader — Monitoring Dashboard")
st.caption("Educational paper trading only. Paper results ≠ live results.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "Safety Status",
    "Portfolio",
    "Risk State",
    "Decisions",
    "Signals",
    "Positions",
    "Trades",
    "Analytics",
    "Setup Scans",
    "Logs",
    "Backtest",
    "Commands",
    "Shadow Testing",
])


# ============================================================
# Tab 1: Safety Status
# ============================================================
with tabs[0]:
    st.header("Safety Status")

    try:
        from config.settings import settings
    except Exception as exc:
        st.error(f"Could not load settings: {exc}")
        settings = None

    lock    = risk_lock_status(RISK_LOCK_FILE)
    state   = load_state(STATE_FILE)

    if lock["exists"]:
        st.error("RISK LOCK ACTIVE — Trading is halted. Manual review required.")
    else:
        st.success("Risk lock: NOT active")

    c1, c2, c3, c4 = st.columns(4)
    if settings:
        c1.metric("PAPER_TRADING",          str(settings.PAPER_TRADING))
        c2.metric("LIVE_TRADING_ALLOWED",   str(settings.LIVE_TRADING_ALLOWED))
        c3.metric("ALPACA_PAPER",           str(settings.ALPACA_PAPER))

        try:
            from broker.alpaca_client import AlpacaClientWrapper
            client = AlpacaClientWrapper()
            c4.metric("Alpaca Configured", "Yes" if client.is_configured() else "No")
        except Exception:
            c4.metric("Alpaca Configured", "Unknown")

        mode = "LIVE DISABLED (paper only)" if not settings.LIVE_TRADING_ALLOWED else "PAPER"
        st.info(f"**Trading Mode:** {mode}")

    with st.expander("What do these settings mean?"):
        st.markdown("""
| Setting | Safe value | What it means |
|---|---|---|
| PAPER_TRADING | True | Bot trades on Alpaca paper endpoint only |
| LIVE_TRADING_ALLOWED | False | Hard block on real-money orders |
| ALPACA_PAPER | True | API URL points to paper server |
| Alpaca Configured | — | .env credentials are set |
| Risk Lock | NOT active | No circuit breaker has fired |
        """)

    if state.get("last_run_timestamp"):
        st.info(f"Last paper-once run: **{state['last_run_timestamp']}**")
    else:
        st.info("paper-once has not run yet.")


# ============================================================
# Tab 2: Portfolio / Account Snapshot
# ============================================================
with tabs[1]:
    st.header("Portfolio / Account Snapshot")

    st.info(
        "Network calls are opt-in. Enable the checkbox below to fetch "
        "live Alpaca paper account data."
    )

    load_account = st.checkbox("Load Alpaca paper account snapshot (requires .env credentials)")

    if load_account:
        try:
            from broker.alpaca_client import AlpacaClientWrapper
            client = AlpacaClientWrapper()

            if not client.is_configured():
                st.warning(
                    "Alpaca credentials not configured. "
                    "Copy `.env.example` to `.env` and fill in paper keys. "
                    "**NEVER paste keys here.**"
                )
            else:
                client.validate_paper_safety()
                account = client.get_account()

                c1, c2, c3 = st.columns(3)
                c1.metric("Equity",       f"${float(account.get('equity', 0)):,.2f}")
                c2.metric("Cash",         f"${float(account.get('cash', 0)):,.2f}")
                c3.metric("Buying Power", f"${float(account.get('buying_power', 0)):,.2f}")

                st.metric("Account Status", account.get("status", "unknown"))

                try:
                    clock = client.get_clock()
                    st.metric("Market Open", str(clock.get("is_open", False)))
                    if not clock.get("is_open"):
                        st.info(f"Next open: {clock.get('next_open', 'unknown')}")
                except Exception as exc:
                    st.warning(f"Could not fetch market clock: {exc}")

        except Exception as exc:
            st.error(f"Failed to load account: {exc}")
    else:
        st.markdown("""
**Offline mode** — No account data loaded.

To see live Alpaca paper account status:
1. Fill in `.env` with paper credentials
2. Enable the checkbox above

Paper trading credentials are needed; live credentials are **blocked**.
""")


# ============================================================
# Tab 3: Risk State
# ============================================================
with tabs[2]:
    st.header("Risk State")

    state = load_state(STATE_FILE)
    lock  = risk_lock_status(RISK_LOCK_FILE)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak Equity",         f"${float(state.get('peak_equity', 0)):,.2f}")
    c2.metric("Day Start Equity",    f"${float(state.get('starting_day_equity', 0)):,.2f}")
    c3.metric("Week Start Equity",   f"${float(state.get('starting_week_equity', 0)):,.2f}")
    c4.metric("Trades Today",        int(state.get("trades_today", 0)))

    st.caption(f"Last updated: {state.get('last_run_timestamp', 'never')}")

    st.subheader("Circuit Breaker / Risk Lock")
    if lock["exists"]:
        st.error("**RISK LOCK IS ACTIVE**")
        st.markdown("""
**Trading is halted.** A drawdown circuit breaker triggered.

To investigate and manually remove the lock:
```
py -3.10 main.py unlock-risk
```

**Do NOT delete the lock file from this dashboard.**
Manual review of logs is required before removing it.
""")
        with st.expander("View lock file contents"):
            st.code(lock["content"] or "[Empty]")
    else:
        st.success("No risk lock detected. Circuit breakers have not fired.")

    with st.expander("Drawdown thresholds (from settings)"):
        try:
            from config.settings import settings
            data = {
                "Threshold": ["Daily warn", "Daily stop", "Weekly stop", "Total lock"],
                "Value": [
                    f"{settings.MAX_DAILY_DRAWDOWN_WARN:.0%}",
                    f"{settings.MAX_DAILY_DRAWDOWN_STOP:.0%}",
                    f"{settings.MAX_WEEKLY_DRAWDOWN_STOP:.0%}",
                    f"{settings.MAX_TOTAL_DRAWDOWN_LOCK:.0%}",
                ],
                "Action": [
                    "Halve new position sizes",
                    "Stop new trades for the day",
                    "Stop new trades for the week",
                    "Create lock file, halt bot permanently",
                ],
            }
            st.table(pd.DataFrame(data))
        except Exception as exc:
            st.warning(f"Could not load settings: {exc}")


# ============================================================
# Tab 4: Recent Paper Decisions
# ============================================================
with tabs[3]:
    st.header("Recent Paper Decisions")

    df_all = load_jsonl(DECISIONS_LOG)

    if df_all.empty:
        st.info(
            "No decisions logged yet.\n\n"
            "Run: `py -3.10 main.py paper-once --ticker SPY --synthetic`"
        )
    else:
        st.caption(f"Total records in log: {len(df_all)}")

        # -- Filters --
        col1, col2, col3, col4 = st.columns(4)

        tickers = sorted(df_all["ticker"].dropna().unique().tolist()) if "ticker" in df_all.columns else []
        all_actions = sorted(df_all["final_action"].dropna().unique().tolist()) if "final_action" in df_all.columns else []

        with col1:
            selected_tickers = st.multiselect("Ticker", tickers, default=tickers)
        with col2:
            selected_actions = st.multiselect("Final Action", all_actions, default=all_actions)
        with col3:
            approved_only = st.checkbox("Approved only")
        with col4:
            n_show = st.number_input("Last N records", min_value=1, max_value=len(df_all), value=min(n_decisions, len(df_all)))

        df = df_all.copy()

        if selected_tickers and "ticker" in df.columns:
            df = df[df["ticker"].isin(selected_tickers)]
        if selected_actions and "final_action" in df.columns:
            df = df[df["final_action"].isin(selected_actions)]
        if approved_only and "risk_approved" in df.columns:
            df = df[df["risk_approved"].fillna(False).astype(bool)]

        df = df.tail(int(n_show))

        display_cols = [c for c in [
            "timestamp", "ticker", "signal_action", "final_action",
            "risk_approved", "shares", "dry_run", "order_submitted",
            "reason", "regime",
        ] if c in df.columns]

        st.dataframe(df[display_cols], use_container_width=True)
        st.caption(f"Showing {len(df)} of {len(df_all)} records")


# ============================================================
# Tab 5: Signals Summary
# ============================================================
with tabs[4]:
    st.header("Signals Summary")

    df_sum = load_jsonl(DECISIONS_LOG)
    summary = summarize_decisions(df_sum)

    if summary["total"] == 0:
        st.info("No decisions to summarise yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Decisions", summary["total"])
        c2.metric("Risk Approved",   summary["approved"])
        c3.metric("Risk Blocked",    summary["blocked"])
        c4.metric("Dry-Run",         summary["dry_run_count"])

        col_left, col_right = st.columns(2)

        with col_left:
            if summary["by_signal_action"]:
                fig = px.pie(
                    names=list(summary["by_signal_action"].keys()),
                    values=list(summary["by_signal_action"].values()),
                    title="Signal Actions Generated",
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_right:
            if summary["by_final_action"]:
                fig2 = px.pie(
                    names=list(summary["by_final_action"].keys()),
                    values=list(summary["by_final_action"].values()),
                    title="Final Actions (after risk gate)",
                )
                st.plotly_chart(fig2, use_container_width=True)

        rejections = get_recent_rejections(df_sum)
        if rejections:
            st.subheader("Most Common Rejection Reasons")
            for r in rejections:
                st.markdown(f"- {r}")
        else:
            st.info("No rejection reasons found (all approved, or 'reason' column missing).")

        # Equity curve from decisions
        if "portfolio_equity" in df_sum.columns and "timestamp" in df_sum.columns:
            st.subheader("Portfolio Equity Over Time (from decision log)")
            eq_df = df_sum[["timestamp", "portfolio_equity"]].dropna()
            if not eq_df.empty:
                eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"], errors="coerce")
                eq_df = eq_df.sort_values("timestamp")
                fig3 = px.line(eq_df, x="timestamp", y="portfolio_equity",
                               title="Equity from paper decisions")
                st.plotly_chart(fig3, use_container_width=True)


# ============================================================
# Tab 6: Positions
# ============================================================
with tabs[5]:
    st.header("Positions")

    load_pos = st.checkbox("Load current Alpaca paper positions (requires .env credentials)")

    if load_pos:
        try:
            from broker.alpaca_client import AlpacaClientWrapper
            from broker.position_tracker import PositionTracker

            client = AlpacaClientWrapper()
            if not client.is_configured():
                st.warning("No Alpaca credentials. Showing dry-run decisions only.")
            else:
                client.validate_paper_safety()
                tracker  = PositionTracker(alpaca_client=client)
                snapshot = tracker.get_portfolio_snapshot()

                st.metric("Open Positions", snapshot.open_positions_count)
                st.metric("Portfolio Exposure", f"{snapshot.portfolio_exposure:.1%}")

                if snapshot.positions:
                    rows = [
                        {
                            "Ticker":        ticker,
                            "Shares":        pos.quantity,
                            "Entry Price":   f"${pos.avg_entry_price:.2f}",
                            "Current Price": f"${pos.current_price:.2f}",
                            "Market Value":  f"${pos.market_value:,.2f}",
                            "Unrealized P&L":f"${pos.unrealized_pl:,.2f}",
                            "P&L %":         f"{pos.unrealized_plpc:.2%}",
                        }
                        for ticker, pos in snapshot.positions.items()
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("No open positions in paper account.")
        except Exception as exc:
            st.error(f"Could not load positions: {exc}")
    else:
        # Show dry-run BUY decisions as a reference (clearly labelled as simulation)
        df_pos = load_jsonl(DECISIONS_LOG)
        buys   = df_pos[
            (df_pos.get("final_action", pd.Series()) == "BUY") &
            (df_pos.get("order_submitted", pd.Series(False)).fillna(False))
        ] if not df_pos.empty and "final_action" in df_pos.columns else pd.DataFrame()

        st.markdown("""
**No live paper account connected.**

Enable the checkbox above to view real Alpaca paper positions.

The table below shows recent **simulated BUY decisions** from the decision log
— these are NOT real positions.
""")
        if not buys.empty:
            st.caption("Dry-run BUY decisions (simulation only — NOT real positions):")
            show_cols = [c for c in ["timestamp","ticker","shares","entry_price","stop_loss","take_profit"] if c in buys.columns]
            st.dataframe(buys[show_cols].tail(20), use_container_width=True)
        else:
            st.info("No simulated BUY decisions in log yet.")


# ============================================================
# Tab 7: Trades
# ============================================================
with tabs[6]:
    st.header("Manual Trades")
    st.caption("Read-only journal and monitor view. No orders can be placed from this dashboard.")

    trades_df = load_jsonl(MANUAL_TRADES_LOG)
    monitor_df = load_jsonl(TRADE_MONITOR_LOG)
    trade_summary = summarize_manual_trades(trades_df)
    split_trades = split_manual_trades(trades_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open Trades", trade_summary["open_count"])
    c2.metric("Closed Trades", trade_summary["closed_count"])
    c3.metric("Realized P&L", f"${trade_summary['total_realized_pnl']:,.2f}")
    c4.metric("Win Rate", f"{trade_summary['win_rate']:.0%}" if trade_summary["closed_count"] else "—")

    st.markdown("---")

    st.subheader("Open Trades")
    open_cols = [
        "trade_id", "ticker", "setup_type", "entry_price", "stop_loss",
        "target_1", "target_2", "shares", "entry_time",
        "test_type", "thesis", "market_context", "entry_quality", "mistake",
        "mistake_notes", "screenshot_path", "exclude_from_analytics",
        "exclusion_reason", "notes",
    ]
    open_df = split_trades["open"]
    if open_df.empty:
        st.info("No open manual trades found.")
    else:
        st.dataframe(open_df[[c for c in open_cols if c in open_df.columns]], use_container_width=True)

    st.subheader("Closed Trades")
    closed_cols = [
        "trade_id", "ticker", "entry_price", "exit_price",
        "realized_pnl", "exit_time", "test_type", "thesis",
        "market_context", "entry_quality", "mistake", "mistake_notes",
        "screenshot_path", "exclude_from_analytics", "exclusion_reason", "notes",
    ]
    closed_df = split_trades["closed"]
    if closed_df.empty:
        st.info("No closed manual trades found.")
    else:
        st.dataframe(closed_df[[c for c in closed_cols if c in closed_df.columns]], use_container_width=True)

    st.subheader("Recent Trade Monitor Results")
    monitor_cols = [
        "timestamp", "ticker", "action", "current_price",
        "r_multiple", "unrealized_pnl", "reason", "guidance",
    ]
    if monitor_df.empty:
        st.info("No trade monitor results found. Run `python main.py trade-monitor` to populate this section.")
    else:
        recent_monitor = monitor_df.tail(n_decisions)
        st.dataframe(
            recent_monitor[[c for c in monitor_cols if c in recent_monitor.columns]],
            use_container_width=True,
        )


# ============================================================
# Tab 8: Analytics
# ============================================================
with tabs[7]:
    st.header("Trade Analytics")
    st.caption("Read-only analytics from CLOSED manual trades only.")

    trades_df = load_jsonl(MANUAL_TRADES_LOG)
    analytics = summarize_trade_analytics(trades_df)
    overall = analytics["overall"]

    if overall["total_trades"] == 0:
        st.info("No closed manual trades found. Close trades with `python main.py trade-close` to populate analytics.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Closed Trades", overall["total_trades"])
        c2.metric("Win Rate", f"{overall['win_rate']:.1%}")
        c3.metric("Realized P&L", f"${overall['total_realized_pnl']:,.2f}")
        c4.metric("Profit Factor", "inf" if overall["profit_factor"] == float("inf") else f"{overall['profit_factor']:.2f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Average P&L", f"${overall['average_pnl']:,.2f}")
        c6.metric("Average Winner", f"${overall['average_winner']:,.2f}")
        c7.metric("Average Loser", f"${overall['average_loser']:,.2f}")
        c8.metric("Expectancy", f"${overall['expectancy']:,.2f}")

        c9, c10 = st.columns(2)
        c9.metric("Avg Setup Confidence", f"{overall.get('average_setup_confidence', 0.0):.2f}")
        c10.metric("Avg Setup Score", f"{overall.get('average_setup_score', 0.0):.2f}")
        st.caption(
            f"Excluded closed trades: {overall.get('excluded_trades', 0)} | "
            f"Mistakes: {overall.get('mistake_count', 0)}"
        )

        st.subheader("By Setup Type")
        setup_df = pd.DataFrame(analytics["by_setup_type"])
        if setup_df.empty:
            st.info("No setup-type breakdown available.")
        else:
            st.dataframe(setup_df, use_container_width=True)

        st.subheader("By Ticker")
        ticker_df = pd.DataFrame(analytics["by_ticker"])
        if ticker_df.empty:
            st.info("No ticker breakdown available.")
        else:
            st.dataframe(ticker_df, use_container_width=True)

        st.subheader("By Setup Rank")
        rank_df = pd.DataFrame(analytics["by_setup_rank"])
        if rank_df.empty:
            st.info("No setup-rank breakdown available.")
        else:
            st.dataframe(rank_df, use_container_width=True)

        st.subheader("By Test Type")
        test_type_df = pd.DataFrame(analytics.get("by_test_type", []))
        if test_type_df.empty:
            st.info("No test-type breakdown available.")
        else:
            st.dataframe(test_type_df, use_container_width=True)

        # -- Trade lifecycle (joins trade monitor history) --
        st.subheader("Trade Lifecycle")
        st.caption("How monitor actions relate to outcomes (joined with trade_monitor.jsonl).")
        lifecycle = summarize_trade_lifecycle(trades_df, load_jsonl(TRADE_MONITOR_LOG))

        if not lifecycle["by_exit_reason"]:
            st.info("No lifecycle data yet. Run `python main.py trade-monitor` while trades are open.")
        else:
            exit_reason_df = pd.DataFrame(lifecycle["by_exit_reason"])
            st.markdown("**By Exit Reason**")
            st.dataframe(exit_reason_df, use_container_width=True)

            followed = sum(r["followed_count"] for r in lifecycle["by_exit_reason"])
            total = sum(r["trades"] for r in lifecycle["by_exit_reason"])
            st.caption(f"Exit signal followed: {followed} of {total} closed trades")

            lifecycle_df = pd.DataFrame(lifecycle["lifecycles"])
            if not lifecycle_df.empty:
                st.markdown("**Per-Trade Lifecycle**")
                lifecycle_cols = [
                    "trade_id", "ticker", "first_monitor_action", "first_alert_action",
                    "time_to_first_alert_seconds", "time_in_trade_seconds",
                    "exit_reason", "exit_action_followed", "realized_pnl",
                ]
                st.dataframe(
                    lifecycle_df[[c for c in lifecycle_cols if c in lifecycle_df.columns]],
                    use_container_width=True,
                )


# ============================================================
# Tab 9: Setup Scans
# ============================================================
with tabs[8]:
    st.header("Day-Trade Setup Scans")

    df_scans = load_jsonl(SETUP_SCANS_LOG)

    if df_scans.empty:
        st.info("No setup scans found. Run `python main.py setup-scan` to populate this section.")
    else:
        # -- Summary metrics --
        summary = summarize_setup_scans(df_scans)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Scans", summary["total_scans"])
        c2.metric("Valid Setups", summary["valid_setups"])
        c3.metric("No Setup", summary["no_setup_count"])
        c4.metric("Avg Confidence", f"{summary['avg_confidence_valid']:.0%}")
        if summary["latest_timestamp"]:
            c5.metric("Latest", str(summary["latest_timestamp"])[:10])

        st.markdown("---")

        # -- Setup type breakdown --
        if summary["by_setup_type"]:
            st.subheader("Setup Type Breakdown")
            setup_type_df = pd.DataFrame(
                list(summary["by_setup_type"].items()),
                columns=["Setup Type", "Count"]
            )
            col_chart, col_table = st.columns([1, 1])
            with col_chart:
                st.bar_chart(setup_type_df.set_index("Setup Type"))
            with col_table:
                st.table(setup_type_df)

        st.markdown("---")

        # -- Filters --
        st.subheader("Filter & Explore")
        col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)

        # Ticker filter
        tickers = sorted(df_scans["ticker"].dropna().unique().tolist())
        selected_ticker = col_filter1.multiselect(
            "Ticker(s)",
            tickers,
            default=tickers if len(tickers) <= 5 else [],
            key="setup_ticker_filter"
        )

        # Valid only
        valid_only = col_filter2.checkbox("Valid setups only", value=False, key="setup_valid_filter")

        # Setup type filter
        setup_types = sorted(
            [t for t in df_scans["setup_type"].dropna().unique().tolist() if t != "no_setup"]
        )
        selected_setup_types = col_filter3.multiselect(
            "Setup Type(s)",
            setup_types,
            default=setup_types,
            key="setup_type_filter"
        )

        # Last N records
        n_records = col_filter4.slider(
            "Show last N records",
            10, min(500, len(df_scans)), 50,
            step=10,
            key="setup_records_slider"
        )

        # -- Apply filters --
        df_filtered = df_scans.copy()

        if selected_ticker:
            df_filtered = df_filtered[df_filtered["ticker"].isin(selected_ticker)]

        if valid_only:
            df_filtered = df_filtered[df_filtered["is_valid"].fillna(False).astype(bool)]

        if selected_setup_types:
            df_filtered = df_filtered[df_filtered["setup_type"].isin(selected_setup_types)]

        df_filtered = df_filtered.tail(n_records)

        # -- Display filtered table --
        st.subheader(f"Setup Scans ({len(df_filtered)} records)")

        display_columns = [
            "timestamp", "ticker", "setup_type", "is_valid", "confidence",
            "entry_trigger", "stop_loss", "target_1", "target_2",
            "risk_reward_1", "risk_reward_2", "reason",
        ]

        available_cols = [c for c in display_columns if c in df_filtered.columns]
        df_display = df_filtered[available_cols].copy()

        # Format numeric columns
        for col in ["confidence", "entry_trigger", "stop_loss", "target_1", "target_2", "risk_reward_1", "risk_reward_2"]:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "—"
                )

        st.dataframe(df_display, use_container_width=True)

        # -- Export option --
        if not df_filtered.empty:
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="Download filtered results as CSV",
                data=csv,
                file_name=f"setup_scans_{pd.Timestamp.now().date()}.csv",
                mime="text/csv",
            )


# ============================================================
# Tab 10: Logs Viewer
# ============================================================
with tabs[9]:
    st.header("Logs Viewer")

    log_choice = st.selectbox("Select log file", list(LOG_FILES.keys()))
    n_tail     = st.slider("Lines to show", 10, 500, n_log_lines, step=10, key="log_tail_slider")

    log_path = LOG_FILES[log_choice]
    content  = tail_file(log_path, n_tail)

    st.caption(f"File: `{log_path}`")
    st.code(content, language="text")


# ============================================================
# Tab 11: Backtest Summary
# ============================================================
with tabs[10]:
    st.header("Backtest / Walk-Forward Summary")

    # -- Saved backtest results --
    bt_results_dir = _PROJECT_ROOT / "results" / "backtests"
    wf_results_dir = _PROJECT_ROOT / "results" / "walk_forward"

    latest_bt = load_latest_result(bt_results_dir)
    latest_wf = load_latest_result(wf_results_dir)

    if latest_bt or latest_wf:
        st.subheader("Latest saved results")
        col_b, col_w = st.columns(2)

        with col_b:
            st.markdown("**Backtest**")
            if latest_bt:
                m = latest_bt.get("metrics", {})
                st.metric("Total Return",  f"{float(m.get('total_return', 0)):+.1%}")
                st.metric("Sharpe Ratio",  f"{float(m.get('sharpe_ratio', 0)):.2f}")
                st.metric("Max Drawdown",  f"{float(m.get('max_drawdown', 0)):.1%}")
                st.metric("Total Trades",  m.get("total_trades", 0))
                st.caption(f"Source: {latest_bt.get('source', '?')} | "
                           f"Saved: {latest_bt.get('timestamp', '?')[:19]}")
            else:
                st.info("No saved backtest found.")

        with col_w:
            st.markdown("**Walk-Forward**")
            if latest_wf:
                m = latest_wf.get("combined_metrics", {})
                st.metric("Combined Return", f"{float(m.get('total_return', 0)):+.1%}")
                st.metric("Combined Sharpe", f"{float(m.get('sharpe_ratio', 0)):.2f}")
                st.metric("Windows",         latest_wf.get("n_windows", 0))
                st.caption(f"Source: {latest_wf.get('source', '?')} | "
                           f"Saved: {latest_wf.get('timestamp', '?')[:19]}")
            else:
                st.info("No saved walk-forward found.")

        # File history
        bt_files = list_result_files(bt_results_dir)
        wf_files = list_result_files(wf_results_dir)
        if bt_files:
            with st.expander(f"Backtest history ({len(bt_files)} files)"):
                st.dataframe(pd.DataFrame(bt_files), use_container_width=True)
        if wf_files:
            with st.expander(f"Walk-forward history ({len(wf_files)} files)"):
                st.dataframe(pd.DataFrame(wf_files), use_container_width=True)
    else:
        st.info(
            "No saved results found. Run a backtest with --save-results to populate this panel."
        )

    st.subheader("Run a backtest (terminal commands)")
    st.code("py -3.10 main.py backtest --ticker SPY --synthetic --save-results")
    st.code("py -3.10 main.py backtest --ticker SPY --start 2020-01-01 --end 2023-12-31 --save-results")

    st.subheader("Run walk-forward validation")
    st.code("py -3.10 main.py walk-forward --synthetic --save-results")
    st.code("py -3.10 main.py walk-forward --ticker SPY --start 2018-01-01 --end 2023-12-31 --save-results")

    st.warning(
        "**Important:** Good backtest results do not guarantee live trading performance. "
        "Past performance is not indicative of future results. "
        "Synthetic data results have no financial meaning."
    )

    # Check for any backtest log content
    bt_content = tail_file(LOG_FILES["backtest.log"], 50)
    if not bt_content.startswith("[File not found"):
        with st.expander("Recent backtest.log output"):
            st.code(bt_content)


# ============================================================
# Tab 12: Manual Commands Help
# ============================================================
with tabs[11]:
    st.header("Manual Commands Help")
    st.markdown("Copy and paste these commands into your terminal:")

    commands = [
        ("Check configuration",         "py -3.10 main.py check-config"),
        ("Test Alpaca paper connection", "py -3.10 main.py test-connection"),
        ("Scan for day-trade setups",    "py -3.10 main.py setup-scan --synthetic"),
        ("Scan with real data",          "py -3.10 main.py setup-scan --ticker SPY,QQQ"),
        ("Backtest (synthetic)",         "py -3.10 main.py backtest --synthetic"),
        ("Backtest (real ticker)",       "py -3.10 main.py backtest --ticker SPY --start 2020-01-01 --end 2023-12-31"),
        ("Walk-forward (synthetic)",     "py -3.10 main.py walk-forward --synthetic"),
        ("Paper-once (synthetic)",       "py -3.10 main.py paper-once --ticker SPY --synthetic"),
        ("Paper-once (real data)",       "py -3.10 main.py paper-once --ticker SPY"),
        ("Paper-loop (synthetic, 2x)",   "py -3.10 main.py paper-loop --synthetic --max-iterations 2 --interval-seconds 0"),
        ("Paper dry-run",                "py -3.10 main.py paper-dry-run --synthetic"),
        ("Unlock risk (view info)",      "py -3.10 main.py unlock-risk"),
        ("Launch this dashboard",        "streamlit run dashboard/app.py"),
        ("Run all tests",                "py -3.10 -m pytest -q --tb=short"),
    ]

    for label, cmd in commands:
        col_l, col_r = st.columns([1, 2])
        col_l.markdown(f"**{label}**")
        col_r.code(cmd)

    st.markdown("---")
    st.markdown("""
### Key safety reminders

- **Dashboard is read-only.** No orders are placed from here.
- **Live trading is permanently disabled** unless you intentionally change settings.
- **Risk lock files** must be deleted manually after reviewing logs.
- **Paper results ≠ live results.** Do not use paper performance as a trading signal.
- **API keys/secrets** must only be stored in `.env` — never in code, never in chat.
""")


# ============================================================
# Tab 13: Shadow Testing (read-only; simulation only)
# ============================================================
with tabs[12]:
    st.header("Shadow Testing")
    st.info(
        "Automated shadow forward-test results — **SIMULATION ONLY**. "
        "No real orders are ever placed. Run "
        "`python main.py forward-test-session` to populate this tab."
    )

    shadow_df = load_jsonl(SHADOW_TRADES_LOG)
    shadow_decisions_df = load_jsonl(SHADOW_DECISIONS_LOG)

    # -- Shadow performance --
    st.subheader("Shadow Performance")
    if shadow_df.empty:
        st.info("No shadow trades found yet.")
    else:
        shadow_summary = summarize_manual_trades(shadow_df)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Open shadow trades", shadow_summary["open_count"])
        col2.metric("Closed shadow trades", shadow_summary["closed_count"])
        col3.metric("Sim P&L", f"{shadow_summary['total_realized_pnl']:+.2f}")
        col4.metric("Win rate", f"{shadow_summary['win_rate']:.1%}")

        shadow_analytics = summarize_trade_analytics(shadow_df)
        if shadow_analytics["by_setup_type"]:
            st.caption("By setup type (closed shadow trades)")
            st.dataframe(pd.DataFrame(shadow_analytics["by_setup_type"]), use_container_width=True)
        if shadow_analytics["by_ticker"]:
            st.caption("By ticker (closed shadow trades)")
            st.dataframe(pd.DataFrame(shadow_analytics["by_ticker"]), use_container_width=True)

        shadow_split = split_manual_trades(shadow_df)
        if not shadow_split["open"].empty:
            st.caption("Open shadow trades")
            st.dataframe(shadow_split["open"], use_container_width=True)

    # -- Skipped reasons --
    st.subheader("Shadow Decisions")
    decision_summary = summarize_shadow_decisions(shadow_decisions_df)
    col1, col2 = st.columns(2)
    col1.metric("Entered", decision_summary["entered"])
    col2.metric("Skipped", decision_summary["skipped"])
    if decision_summary["skip_reasons"]:
        st.caption("Skipped by reason")
        st.dataframe(
            pd.DataFrame(
                [{"skip_reason": reason, "count": count}
                 for reason, count in decision_summary["skip_reasons"].items()]
            ),
            use_container_width=True,
        )
    else:
        st.caption("No skipped shadow decisions recorded yet.")

    # -- Manual vs Shadow --
    st.subheader("Manual vs Shadow")
    manual_df = load_jsonl(MANUAL_TRADES_LOG)
    manual_overall = summarize_trade_analytics(manual_df)["overall"]
    shadow_overall = summarize_trade_analytics(shadow_df)["overall"]
    comparison = pd.DataFrame([
        {
            "metric": "Closed trades",
            "manual": manual_overall["total_trades"],
            "shadow": shadow_overall["total_trades"],
        },
        {
            "metric": "Total P&L",
            "manual": round(manual_overall["total_realized_pnl"], 2),
            "shadow": round(shadow_overall["total_realized_pnl"], 2),
        },
        {
            "metric": "Win rate",
            "manual": f"{manual_overall['win_rate']:.1%}",
            "shadow": f"{shadow_overall['win_rate']:.1%}",
        },
        {
            "metric": "Avg P&L",
            "manual": round(manual_overall["average_pnl"], 2),
            "shadow": round(shadow_overall["average_pnl"], 2),
        },
    ])
    st.dataframe(comparison, use_container_width=True)
    st.caption(
        "Shadow trades follow exit signals 100% by construction; manual "
        "follow-through is shown in the Analytics tab (Monitor Effectiveness)."
    )
