"""Local visual alert output for setup-scan-loop and trade-monitor."""
import typer


def format_trade_action_card(setup, account_equity=None, risk_pct: float = 0.005, stale_data: bool = False) -> str:
    """
    Build a complete trade plan / action card string for a setup.

    This is a watchlist planning aid only — it never places orders. Position
    sizing is a suggestion based on a fixed-fractional risk model.

    Args:
        setup: DayTradeSetup object with setup details.
        account_equity: Optional account equity. If provided, suggested share
            count and estimated max loss are included.
        risk_pct: Fraction of equity to risk per trade (default 0.005 = 0.5%).
        stale_data: If True, label valid setups as review-only because the
            underlying market data is stale.

    Returns:
        Multi-line string describing the trade plan.
    """
    action = "STALE REVIEW SETUP" if stale_data and setup.is_valid else ("VALID SETUP" if setup.is_valid else "WATCH")
    rank = setup.metadata.get("rank", "?")

    risk_per_share = setup.entry_trigger - setup.stop_loss

    lines = []
    lines.append(f"ACTION: {action}")
    lines.append(f"  {rank}. {setup.ticker} — {setup.setup_type.value}")
    lines.append(f"  Confidence:     {setup.confidence:.2f}")
    lines.append(f"  Entry trigger:  {setup.entry_trigger:.2f}")
    lines.append(f"  Stop loss:      {setup.stop_loss:.2f}")
    lines.append(f"  Target 1:       {setup.target_1:.2f}")
    lines.append(f"  Target 2:       {setup.target_2:.2f}")
    lines.append(f"  Risk/Reward 1:  {setup.risk_reward_1:.2f}")
    lines.append(f"  Risk/Reward 2:  {setup.risk_reward_2:.2f}")
    lines.append(f"  Risk/share:     {risk_per_share:.2f}")

    # Suggested sizing only when equity provided and risk is positive
    if account_equity is not None:
        if risk_per_share > 0:
            suggested_shares = int((account_equity * risk_pct) / risk_per_share)
            estimated_max_loss = suggested_shares * risk_per_share
            lines.append(f"  Suggested shares: {suggested_shares}  (risk {risk_pct * 100:.2f}% of {account_equity:.2f})")
            lines.append(f"  Est. max loss:    {estimated_max_loss:.2f}")
        else:
            # Non-positive risk/share: cannot size safely
            lines.append("  Suggested shares: N/A (invalid risk/share <= 0)")
            lines.append("  Est. max loss:    N/A")

    lines.append(f"  Reason: {setup.reason}")

    if setup.warnings:
        lines.append(f"  Warnings: {'; '.join(setup.warnings)}")

    lines.append("  Invalidation: price closes below stop_loss or loses session VWAP")

    return "\n".join(lines)


def print_alert_banner(setup, account_equity=None, risk_pct: float = 0.005, stale_data: bool = False) -> None:
    """
    Print a visual alert banner for a new setup.

    Args:
        setup: DayTradeSetup object with setup details.
        account_equity: Optional account equity for position sizing.
        risk_pct: Fraction of equity to risk per trade (default 0.005).
        stale_data: If True, label the setup as review-only in the action card.
    """
    banner = "=" * 50
    typer.secho(banner, fg=typer.colors.YELLOW, bold=True)
    typer.secho("NEW SETUP ALERT", fg=typer.colors.GREEN, bold=True)
    typer.echo(format_trade_action_card(setup, account_equity=account_equity, risk_pct=risk_pct, stale_data=stale_data))
    typer.secho(banner, fg=typer.colors.YELLOW, bold=True)


def format_trade_monitor_alert(result) -> str:
    """
    Format an advisory alert for a manual trade monitor result.

    This is display-only. It never places orders or changes trade state.
    """
    return "\n".join([
        "TRADE MONITOR ALERT",
        f"  Ticker:         {result.ticker}",
        f"  Action:         {result.action.value}",
        f"  Current price:  {result.current_price:.2f}",
        f"  R multiple:     {result.r_multiple:.2f}",
        f"  Unrealized P&L: {result.unrealized_pnl:.2f}",
        f"  Reason:         {result.reason}",
        f"  Guidance:       {result.guidance}",
    ])


def print_trade_monitor_alert_banner(result) -> None:
    """Print a visual banner for an important trade monitor result."""
    banner = "=" * 50
    typer.secho(banner, fg=typer.colors.YELLOW, bold=True)
    typer.echo(format_trade_monitor_alert(result))
    typer.secho(banner, fg=typer.colors.YELLOW, bold=True)
