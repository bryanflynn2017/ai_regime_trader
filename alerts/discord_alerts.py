"""Optional Discord webhook alerts.

This module is intentionally small and dependency-free. Discord alerts are
disabled unless DISCORD_WEBHOOK_URL is configured in the local environment or
.env file. The webhook URL is never printed or logged by this module.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

_LAST_DISCORD_ERROR = ""
_DISCORD_MESSAGE_LIMIT = 1900
_SETUP_ALERT_TITLE = "\U0001f6a8 **NEW SETUP ALERT**"
_TRADE_MONITOR_ALERT_TITLE = "\u26a0\ufe0f **TRADE MONITOR ALERT**"
_DAILY_REVIEW_TITLE = "\U0001f4ca **DAILY REVIEW**"
_SHADOW_ENTRY_TITLE = "\U0001f312 **SHADOW ENTRY \u2014 SIMULATION ONLY**"
_SHADOW_EXIT_TITLE = "\U0001f318 **SHADOW EXIT \u2014 SIMULATION ONLY**"
_SHADOW_SUMMARY_TITLE = "\U0001f319 **SHADOW SESSION SUMMARY \u2014 SIMULATION ONLY**"
_SHADOW_DISCLAIMER = "\u26a0\ufe0f No real order placed. Automated shadow forward test."
_SEPARATOR = "\u2501" * 20


def _webhook_url() -> str:
    from config.secret_values import secret_value
    from config.settings import settings

    return secret_value(getattr(settings, "DISCORD_WEBHOOK_URL", "")).strip()


def is_discord_configured() -> bool:
    """Return True when a local Discord webhook URL is configured."""
    return bool(_webhook_url())


def _redact_webhook_urls(message: object) -> str:
    """Return text with Discord webhook URLs removed."""
    text = str(message or "")
    webhook = _webhook_url()
    if webhook:
        text = text.replace(webhook, "[redacted webhook]")
    return re.sub(r"https://(?:discord|discordapp)\.com/api/webhooks/\S+", "[redacted webhook]", text)


def _sanitize_error(message: object) -> str:
    """Return a short Discord error message with webhook URLs removed."""
    return _redact_webhook_urls(message)[:300]


def _clean_message_part(value: object, max_length: int = 500) -> str:
    """Compact user/model text so Discord markdown stays readable."""
    text = _redact_webhook_urls(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_length:
        return text[: max_length - 3].rstrip() + "..."
    return text


def _format_money(value: object) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "`n/a`"
    prefix = "-$" if amount < 0 else "$"
    return f"`{prefix}{abs(amount):.2f}`"


def _format_number(value: object, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "`n/a`"
    return f"`{number:.2f}{suffix}`"


def _truncate_discord_message(message: str) -> str:
    if len(message) <= _DISCORD_MESSAGE_LIMIT:
        return message
    return message[: _DISCORD_MESSAGE_LIMIT - 3].rstrip() + "..."


def _clean_daily_review_body(review_text: str) -> str:
    """Prepare local daily-review Markdown for Discord without changing content structure."""
    body = _redact_webhook_urls(review_text)
    body = body.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = body.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip().lower() in {"# daily review", "daily review"}:
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


def _daily_review_sections(review_text: str) -> dict[str, list[str]]:
    """Parse local daily-review Markdown into simple section line buckets."""
    body = _clean_daily_review_body(review_text)
    sections: dict[str, list[str]] = {}
    current = ""
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
            continue
        if current:
            sections[current].append(line)
    return sections


def _section_value(sections: dict[str, list[str]], section: str, prefix: str, default: str = "0") -> str:
    """Return the value after a prefix in a parsed daily-review section."""
    for line in sections.get(section, []):
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()
    return default


def _first_section_line(sections: dict[str, list[str]], section: str, prefix: str) -> str:
    """Return a full stripped line from a section that starts with prefix."""
    for line in sections.get(section, []):
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped
    return ""


def _count_from_section(sections: dict[str, list[str]], section: str) -> int:
    value = _section_value(sections, section, "Count:", "0")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _clean_compact_detail(value: str, max_length: int = 220) -> str:
    """Trim low-priority compact daily-review detail lines."""
    value = _redact_webhook_urls(value).strip()
    value = re.sub(r"\s+", " ", value)
    if len(value) > max_length:
        return value[: max_length - 3].rstrip() + "..."
    return value


def _format_compact_daily_review_body(review_text: str) -> str:
    """Build a compact mobile Discord daily review from the full local review."""
    sections = _daily_review_sections(review_text)
    review_date = _section_value(sections, "Date Reviewed", "", "")
    if not review_date:
        date_lines = [line.strip() for line in sections.get("Date Reviewed", []) if line.strip()]
        review_date = date_lines[0] if date_lines else "unknown"

    alerts_count = _section_value(sections, "Setup Alerts Today", "Count:", "0")
    by_ticker = _section_value(sections, "Setup Alerts Today", "By ticker:", "none")
    by_setup = _section_value(sections, "Setup Alerts Today", "By setup_type:", "none")
    closed_count = _count_from_section(sections, "Closed Trades Today")
    excluded_count = _section_value(sections, "Performance Summary", "Excluded trades:", "0")
    try:
        excluded_count_int = int(float(excluded_count))
        included_count = max(0, closed_count - excluded_count_int)
    except (TypeError, ValueError):
        excluded_count_int = 0
        included_count = closed_count

    monitor_counts = _section_value(sections, "Monitor Alerts Today", "Counts:", "none")
    closed_lines = []
    for line in sections.get("Closed Trades Today", []):
        stripped = line.strip()
        if not stripped or stripped.startswith("Count:") or stripped == "(none)":
            continue
        compact_line = stripped.split(" notes=", 1)[0]
        closed_lines.append(_clean_compact_detail(compact_line))

    lines = [
        f"Date: {review_date}",
        "",
        "Scoreboard",
        f"  Clean P&L: {_section_value(sections, 'Performance Summary', 'Total P&L today:', '+0.00')}",
        f"  Win Rate: {_section_value(sections, 'Performance Summary', 'Win rate today:', '0.0%')}",
        f"  Included Trades: {included_count}",
        f"  Excluded Trades: {excluded_count}",
        f"  Monitor Alerts: {monitor_counts}",
    ]

    # Shadow forward-test one-liner (simulation only; section exists only
    # when shadow logs are present).
    if "Shadow Forward Test" in sections:
        shadow_section_text = "\n".join(sections.get("Shadow Forward Test", []))
        if "No shadow session was run today." in shadow_section_text:
            lines.append("  Shadow: no session run today")
        else:
            shadow_iterations = _section_value(sections, "Shadow Forward Test", "Iterations:", "0")
            shadow_setups = _section_value(sections, "Shadow Forward Test", "Evaluated setups:", "0")
            shadow_entered = _section_value(sections, "Shadow Forward Test", "Entered:", "0")
            shadow_pnl = _section_value(sections, "Shadow Forward Test", "Sim P&L:", "+0.00")
            shadow_quality = _section_value(sections, "Shadow Forward Test", "Data quality:", "CLEAN")
            shadow_gap_count = _section_value(sections, "Shadow Forward Test", "Data gaps:", "0")
            clean_pnl = _section_value(sections, "Shadow Forward Test", "Clean Sim P&L:", "")
            partial_pnl = _section_value(sections, "Shadow Forward Test", "Partial Sim P&L:", "")
            shadow_line = (
                f"  Shadow: {shadow_iterations} iterations, {shadow_setups} setups, "
                f"{shadow_entered} entries, sim P&L {shadow_pnl}"
            )
            if clean_pnl or partial_pnl:
                shadow_line += (
                    f" (clean sessions: {clean_pnl or '+0.00'}, "
                    f"partial sessions: {partial_pnl or '+0.00'})"
                )
            if shadow_quality != "CLEAN":
                shadow_line += f", data {shadow_quality}, gaps={shadow_gap_count}"
            lines.append(shadow_line)

    lines.extend([
        "",
        "Setups Today",
        f"  Alerts: {alerts_count}",
        f"  By ticker: {_clean_compact_detail(by_ticker)}",
        f"  By setup: {_clean_compact_detail(by_setup)}",
        "",
        "Closed Trades",
    ])

    included_lines = closed_lines[:included_count] if included_count > 0 else []
    excluded_lines = closed_lines[included_count:included_count + excluded_count_int] if excluded_count_int > 0 else []

    lines.append("  Included (clean scoreboard):")
    if included_lines:
        lines.extend(f"    {line}" for line in included_lines[:4])
        if len(included_lines) > 4:
            lines.append(f"    ... {len(included_lines) - 4} more included trades in local review")
    else:
        lines.append("    (none)")

    lines.append("  Excluded (not in clean scoreboard):")
    if excluded_count_int > 0:
        if excluded_lines:
            lines.extend(f"    [EXCLUDED] {line}" for line in excluded_lines[:4])
            if len(excluded_lines) > 4:
                lines.append(f"    ... {len(excluded_lines) - 4} more excluded trades in local review")
        else:
            lines.append(f"    [EXCLUDED] {excluded_count_int} trade(s) in local review")
    else:
        lines.append("    (none)")

    lines.extend([
        "",
        "Key Notes",
        "  - Review VWAP-loss and WATCH alerts before the next session.",
        "  - Check whether repeated setups were clean or choppy.",
        "  - Review excluded trades for execution/logging issues.",
        "  - Full review saved locally.",
    ])
    return "\n".join(lines)


def _set_last_discord_error(message: object) -> None:
    global _LAST_DISCORD_ERROR
    _LAST_DISCORD_ERROR = _sanitize_error(message)


def get_last_discord_error() -> str:
    """Return the latest safe Discord failure reason, if any."""
    return _LAST_DISCORD_ERROR


def send_discord_message(content: str) -> bool:
    """
    Send a plain text Discord webhook message.

    Returns False when Discord is not configured or when the HTTP request fails.
    Network failures are intentionally swallowed so alerts can never crash scans.
    """
    global _LAST_DISCORD_ERROR
    _LAST_DISCORD_ERROR = ""

    url = _webhook_url()
    if not url:
        _set_last_discord_error("Discord webhook is not configured.")
        return False
    if not content:
        _set_last_discord_error("Discord message content is empty.")
        return False

    payload = json.dumps({"content": str(content)[:1900]}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "AI-Regime-Trader",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = int(getattr(response, "status", getattr(response, "code", 0)) or 0)
            if 200 <= status < 300:
                return True
            _set_last_discord_error(f"Discord returned HTTP {status}.")
            return False
    except urllib.error.HTTPError as exc:
        _set_last_discord_error(f"Discord returned HTTP {exc.code}.")
        return False
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        _set_last_discord_error(f"Discord send failed: {exc}")
        return False


def format_setup_discord_alert(setup) -> str:
    """Format a setup alert for Discord."""
    rank = setup.metadata.get("rank", "?") if getattr(setup, "metadata", None) else "?"
    setup_type = getattr(setup.setup_type, "value", setup.setup_type)
    try:
        confidence = f"{float(setup.confidence) * 100:.0f}%"
    except (TypeError, ValueError):
        confidence = "n/a"

    message = "\n".join([
        _SETUP_ALERT_TITLE,
        _SEPARATOR,
        f"**Ticker:** `{setup.ticker}`",
        f"**Rank:** `#{rank}`",
        f"**Setup:** `{setup_type}`",
        "",
        f"**Confidence:** `{confidence}`",
        f"**Entry:** {_format_money(setup.entry_trigger)}",
        f"**Stop:** {_format_money(setup.stop_loss)}",
        f"**Target 1:** {_format_money(setup.target_1)}",
        f"**Target 2:** {_format_money(setup.target_2)}",
        "",
        f"**Reason:** {_clean_message_part(setup.reason, max_length=700)}",
    ])
    return _truncate_discord_message(message)


def format_trade_monitor_discord_alert(result) -> str:
    """Format a trade monitor alert for Discord."""
    action = getattr(result.action, "value", result.action)
    message = "\n".join([
        _TRADE_MONITOR_ALERT_TITLE,
        _SEPARATOR,
        f"**Ticker:** `{result.ticker}`",
        f"**Action:** `{action}`",
        "",
        f"**Current:** {_format_money(result.current_price)}",
        f"**R Multiple:** {_format_number(result.r_multiple, suffix='R')}",
        f"**Unrealized P&L:** {_format_money(result.unrealized_pnl)}",
        "",
        f"**Guidance:** {_clean_message_part(result.guidance, max_length=500)}",
        f"**Reason:** {_clean_message_part(result.reason, max_length=700)}",
    ])
    return _truncate_discord_message(message)


def format_daily_review_discord_message(review_text: str) -> str:
    """Format a compact daily review summary for Discord."""
    body = _format_compact_daily_review_body(review_text)
    message = "\n".join([
        _DAILY_REVIEW_TITLE,
        _SEPARATOR,
        "",
        body,
    ])
    suffix = "\n\n(truncated; full review saved locally)"
    if len(message) <= _DISCORD_MESSAGE_LIMIT:
        return message
    protected_sections = message.split("\n\n")
    protected = "\n\n".join(protected_sections[:3])
    remaining_budget = _DISCORD_MESSAGE_LIMIT - len(protected) - len(suffix) - 2
    if remaining_budget <= 0:
        return protected[: _DISCORD_MESSAGE_LIMIT - len(suffix)].rstrip() + suffix
    low_priority = "\n\n".join(protected_sections[3:])
    return (protected + "\n\n" + low_priority[:remaining_budget].rstrip() + suffix)


def format_shadow_entry_discord_alert(trade) -> str:
    """Format an automated shadow-trade ENTRY for Discord (simulation only)."""
    try:
        confidence = f"{float(trade.setup_confidence) * 100:.0f}%"
    except (TypeError, ValueError):
        confidence = "n/a"

    message = "\n".join([
        _SHADOW_ENTRY_TITLE,
        _SEPARATOR,
        f"**Ticker:** `{trade.ticker}`",
        f"**Setup:** `{trade.setup_type}`",
        f"**Confidence:** `{confidence}`",
        "",
        f"**Sim Entry:** {_format_money(trade.entry_price)}",
        f"**Stop:** {_format_money(trade.stop_loss)}",
        f"**Target 1:** {_format_money(trade.target_1)}",
        f"**Target 2:** {_format_money(trade.target_2)}",
        f"**Shares (sim):** {_format_number(trade.shares)}",
        "",
        _SHADOW_DISCLAIMER,
    ])
    return _truncate_discord_message(message)


def format_shadow_session_summary_discord_message(
    session_id: str, summary_lines, status: str = "ENDED",
) -> str:
    """Format the end-of-session shadow summary for Discord (simulation only)."""
    body = [_redact_webhook_urls(line).rstrip() for line in summary_lines]
    message = "\n".join([
        _SHADOW_SUMMARY_TITLE,
        _SEPARATOR,
        f"**Session:** `{_clean_message_part(session_id, max_length=80)}`",
        f"**Status:** `{_clean_message_part(status, max_length=20)}`",
        "",
        *body,
        "",
        _SHADOW_DISCLAIMER,
    ])
    return _truncate_discord_message(message)


def format_shadow_exit_discord_alert(trade, exit_signal: str) -> str:
    """Format an automated shadow-trade EXIT for Discord (simulation only)."""
    message = "\n".join([
        _SHADOW_EXIT_TITLE,
        _SEPARATOR,
        f"**Ticker:** `{trade.ticker}`",
        f"**Setup:** `{trade.setup_type}`",
        f"**Exit signal:** `{_clean_message_part(exit_signal, max_length=50)}`",
        "",
        f"**Sim Entry:** {_format_money(trade.entry_price)}",
        f"**Sim Exit:** {_format_money(trade.exit_price)}",
        f"**Sim P&L:** {_format_money(trade.realized_pnl)}",
        "",
        _SHADOW_DISCLAIMER,
    ])
    return _truncate_discord_message(message)
