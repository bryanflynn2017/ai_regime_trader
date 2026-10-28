"""
Tests for read-only manual trade analytics.
"""
from __future__ import annotations

import json

import pytest
import typer.testing

from main import app
from trades.trade_analytics import TradeAnalytics
from trades.trade_tracker import ManualTrade, TradeStatus, TradeTracker


@pytest.fixture
def cli_runner():
    return typer.testing.CliRunner()


def test_legacy_partial_reasons_do_not_change_clean_selection():
    from main import _sum_shadow_counters, _clean_shadow_session_records
    records = [
        {"session_id": "clean", "status": "ENDED", "counters": {"data_quality": "CLEAN"}},
        {"session_id": "gap", "status": "ENDED", "counters": {"data_gap_count": 1, "largest_data_gap_seconds": 4200}},
        {"session_id": "stale", "status": "ENDED", "counters": {"data_quality": "PARTIAL_DATA", "data_quality_failures": [{"reason": "STALE_DATA", "timestamp": "2026-09-07T10:00:00"}]}},
        {"session_id": "unknown", "status": "ENDED", "counters": {"data_quality": "PARTIAL_DATA"}},
    ]
    assert _clean_shadow_session_records(records) == records[:1]
    totals = _sum_shadow_counters(records)
    assert totals["partial_data_reasons"] == ["ITERATION_GAP", "STALE_PROVIDER_DATA", "UNKNOWN_LEGACY_REASON"]
    assert totals["normal_data_gap_count"] == 1
    assert totals["extreme_data_gap_count"] == 0  # No invented classification for old gaps.
    assert totals["max_iteration_gap_seconds"] == 4200
    assert totals["stale_warning_count"] == 1


def test_partial_reason_aliases_are_deduplicated_and_gap_counts_sum():
    from main import _sum_shadow_counters
    records = [{"counters": {
        "partial_data_reasons": ["STALE_DATA", "STALE_PROVIDER_DATA", "ITERATION_GAP"],
        "data_quality_failures": [{"reason": "STALE_DATA"}, {"reason": "ITERATION_GAP"}],
        "stale_data_caused_partial": True,
        "data_gap_count": 3, "normal_data_gap_count": 2, "extreme_data_gap_count": 1,
    }}]
    totals = _sum_shadow_counters(records)
    assert totals["partial_data_reasons"] == ["ITERATION_GAP", "STALE_PROVIDER_DATA"]
    assert totals["normal_data_gap_count"] + totals["extreme_data_gap_count"] == totals["data_gap_count"] == 3


@pytest.mark.parametrize("clean_only", [False, True])
def test_symbol_freshness_reporting_and_legacy_coverage(cli_runner, tmp_path, monkeypatch, clean_only):
    from trades.shadow_session import ShadowSessionState, append_shadow_session_history
    from main import _sum_shadow_counters, _clean_shadow_session_records
    monkeypatch.chdir(tmp_path)
    counters = {"data_quality": "PARTIAL_DATA", "stale_warning_count": 2,
                "missing_bar_observations": 1, "stale_data_events": [
        {"freshness_snapshot": {"classification": "SYMBOL_LOCAL_STALE", "stale_symbols": ["ARM"], "max_bar_age_minutes": 20}},
        {"freshness_snapshot": {"classification": "FEED_WIDE_STALE", "stale_symbols": ["ARM", "SPY"], "max_bar_age_minutes": 25}},
    ]}
    records = []
    for sid, c in [("new", counters), ("legacy", {"data_quality": "PARTIAL_DATA", "stale_warning_count": 3})]:
        state = ShadowSessionState(session_id=sid, started_at="2026-09-07T10:00:00", status="ENDED", counters=c)
        state.save()
        records.append(append_shadow_session_history(state))
    assert _clean_shadow_session_records(records) == []
    summary = _sum_shadow_counters(records)["freshness_diagnostics"]
    assert summary["detailed_events"] == 2
    assert summary["stale_events"] == 5
    assert summary["stale_symbol_observations"] == 3
    assert summary["most_frequently_stale"] == {"ARM": 2, "SPY": 1}
    args = ["trade-analytics", "--include-shadow"] + (["--clean-shadow-only"] if clean_only else [])
    result = cli_runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "2/5 stale events with symbol details" in result.output
    assert "FEED_WIDE_STALE=1" in result.output
    assert "ARM=2" in result.output
    assert "Worst recorded age: 25 min" in result.output


def test_old_symbol_details_are_unavailable_not_zero():
    from data_providers.freshness import freshness_report_lines
    text = "\n".join(freshness_report_lines({"data_quality": "PARTIAL_DATA", "stale_warning_count": 3}))
    assert "0/3 stale events with symbol details" in text
    assert "unavailable (not recorded)" in text
    assert "Missing-bar observations: unavailable" in text
    assert "FEED_WIDE_STALE=0" not in text


def test_repeated_events_count_once_per_symbol_each_time_and_nulls_are_unknown():
    from data_providers.freshness import aggregate_freshness, freshness_report_lines
    event = {"freshness_snapshot": {"classification": "INSUFFICIENT_DATA",
                                   "stale_symbols": ["ARM", "ARM"], "max_bar_age_minutes": None}}
    counters = {"stale_warning_count": 2, "stale_data_events": [event, event],
                "missing_bar_observations": None}
    summary = aggregate_freshness([counters])
    assert summary["detailed_events"] == summary["stale_events"] == 2
    assert summary["most_frequently_stale"] == {"ARM": 2}
    assert summary["stale_symbol_observations"] == 2
    assert summary["classification_counts"] == {"INSUFFICIENT_DATA": 2}
    text = "\n".join(freshness_report_lines(counters))
    assert "Worst recorded age: unavailable" in text
    assert "Missing-bar observations: unavailable" in text
    assert "None min" not in text


def test_clean_analytics_reports_partial_gap_reasons(cli_runner, tmp_path, monkeypatch):
    from trades.shadow_session import ShadowSessionState, append_shadow_session_history
    monkeypatch.chdir(tmp_path)
    append_shadow_session_history(ShadowSessionState(
        session_id="suspended", started_at="2026-09-07T10:00:00", status="ENDED",
        config={"provider": "alpaca", "feed": "iex"},
        counters={"data_quality": "PARTIAL_DATA", "data_gap_count": 1,
                  "extreme_data_gap_count": 1, "normal_data_gap_count": 0,
                  "max_iteration_gap_seconds": 7200, "partial_data_reasons": ["ITERATION_GAP"]},
    ))
    result = cli_runner.invoke(app, ["trade-analytics", "--include-shadow", "--clean-shadow-only"])
    assert result.exit_code == 0, result.output
    assert "Partial-data reason: ITERATION_GAP" in result.output
    assert "Extreme/session-suspension gaps: 1" in result.output
    assert "Max iteration gap: 120.0 min" in result.output
    assert "Provider/feed: alpaca/iex" in result.output
    assert "invalid for strategy proof and excluded from clean analytics" in result.output


def _trade(**overrides) -> ManualTrade:
    defaults = dict(
        trade_id="t1",
        ticker="SPY",
        setup_type="vwap_reclaim_long",
        entry_price=100.0,
        stop_loss=98.0,
        target_1=104.0,
        target_2=106.0,
        shares=10.0,
        entry_time="2026-06-08T10:00:00",
        status=TradeStatus.CLOSED.value,
        exit_price=105.0,
        exit_time="2026-06-08T10:30:00",
        realized_pnl=50.0,
    )
    defaults.update(overrides)
    return ManualTrade(**defaults)


def _tracker_with_trades(path, trades):
    tracker = TradeTracker(storage_path=path)
    for trade in trades:
        tracker.add_trade(trade)
    return tracker


def test_trade_analytics_win_rate(tmp_path):
    _tracker_with_trades(tmp_path / "manual_trades.jsonl", [
        _trade(trade_id="w1", realized_pnl=100.0),
        _trade(trade_id="w2", realized_pnl=50.0),
        _trade(trade_id="l1", realized_pnl=-50.0),
    ])

    result = TradeAnalytics(tmp_path / "manual_trades.jsonl").analyze()

    assert result.overall["wins"] == 2
    assert result.overall["losses"] == 1
    assert result.overall["win_rate"] == pytest.approx(2 / 3)


def test_trade_analytics_profit_factor(tmp_path):
    _tracker_with_trades(tmp_path / "manual_trades.jsonl", [
        _trade(trade_id="w1", realized_pnl=100.0),
        _trade(trade_id="w2", realized_pnl=50.0),
        _trade(trade_id="l1", realized_pnl=-75.0),
    ])

    result = TradeAnalytics(tmp_path / "manual_trades.jsonl").analyze()

    assert result.overall["profit_factor"] == pytest.approx(2.0)


def test_trade_analytics_expectancy(tmp_path):
    _tracker_with_trades(tmp_path / "manual_trades.jsonl", [
        _trade(trade_id="w1", realized_pnl=100.0),
        _trade(trade_id="l1", realized_pnl=-40.0),
        _trade(trade_id="l2", realized_pnl=-10.0),
    ])

    result = TradeAnalytics(tmp_path / "manual_trades.jsonl").analyze()

    assert result.overall["expectancy"] == pytest.approx(50.0 / 3.0)
    assert result.overall["average_pnl"] == pytest.approx(50.0 / 3.0)


def test_trade_analytics_setup_breakdown(tmp_path):
    _tracker_with_trades(tmp_path / "manual_trades.jsonl", [
        _trade(trade_id="s1", setup_type="orb", realized_pnl=100.0),
        _trade(trade_id="s2", setup_type="orb", realized_pnl=-25.0),
        _trade(trade_id="s3", setup_type="vwap", realized_pnl=50.0),
    ])

    result = TradeAnalytics(tmp_path / "manual_trades.jsonl").analyze()
    by_setup = {row["setup_type"]: row for row in result.by_setup_type}

    assert by_setup["orb"]["total_trades"] == 2
    assert by_setup["orb"]["total_realized_pnl"] == pytest.approx(75.0)
    assert by_setup["vwap"]["win_rate"] == 1.0


def test_trade_analytics_ticker_breakdown(tmp_path):
    _tracker_with_trades(tmp_path / "manual_trades.jsonl", [
        _trade(trade_id="t1", ticker="SPY", realized_pnl=100.0),
        _trade(trade_id="t2", ticker="QQQ", realized_pnl=-25.0),
        _trade(trade_id="t3", ticker="QQQ", realized_pnl=50.0),
    ])

    result = TradeAnalytics(tmp_path / "manual_trades.jsonl").analyze()
    by_ticker = {row["ticker"]: row for row in result.by_ticker}

    assert by_ticker["SPY"]["total_realized_pnl"] == pytest.approx(100.0)
    assert by_ticker["QQQ"]["total_trades"] == 2
    assert by_ticker["QQQ"]["win_rate"] == 0.5


def test_trade_analytics_empty_history(tmp_path):
    result = TradeAnalytics(tmp_path / "missing.jsonl").analyze()

    assert result.overall["total_trades"] == 0
    assert result.by_setup_type == []
    assert result.by_ticker == []


def test_trade_analytics_ignores_open_trades(tmp_path):
    _tracker_with_trades(tmp_path / "manual_trades.jsonl", [
        _trade(trade_id="closed", realized_pnl=100.0),
        _trade(trade_id="open", status=TradeStatus.OPEN.value, realized_pnl=None),
    ])

    result = TradeAnalytics(tmp_path / "manual_trades.jsonl").analyze()

    assert result.overall["total_trades"] == 1
    assert result.overall["total_realized_pnl"] == pytest.approx(100.0)


def test_trade_analytics_ignores_excluded_closed_trades(tmp_path):
    _tracker_with_trades(tmp_path / "manual_trades.jsonl", [
        _trade(trade_id="included", realized_pnl=100.0),
        _trade(
            trade_id="excluded",
            realized_pnl=-999.0,
            exclude_from_analytics=True,
            exclusion_reason="paper workflow test",
        ),
    ])

    result = TradeAnalytics(tmp_path / "manual_trades.jsonl").analyze()

    assert result.overall["total_trades"] == 1
    assert result.overall["total_realized_pnl"] == pytest.approx(100.0)
    assert result.overall["excluded_trades"] == 1


def test_trade_analytics_reflects_corrected_exit_price(tmp_path):
    tracker = TradeTracker(storage_path=tmp_path / "manual_trades.jsonl")
    tracker.add_trade(
        ManualTrade(
            trade_id="corrected",
            ticker="TSLA",
            setup_type="vwap_reclaim_long",
            entry_price=390.0,
            stop_loss=388.0,
            target_1=394.0,
            target_2=398.0,
            shares=2.0,
            entry_time="2026-06-08T10:00:00",
        )
    )
    tracker.close_trade("corrected", exit_price=395.13)
    tracker.edit_trade("corrected", exit_price=398.48, correction_notes="correct fill")

    result = TradeAnalytics(tmp_path / "manual_trades.jsonl").analyze()

    assert result.overall["total_trades"] == 1
    assert result.overall["total_realized_pnl"] == pytest.approx(16.96)
    trade = tracker.get_trade("corrected")
    assert trade.metadata["corrected_at"]


def test_trade_analytics_cli_outputs_sections(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_trade(trade_id="closed", realized_pnl=100.0))

    result = cli_runner.invoke(app, ["trade-analytics"])

    assert result.exit_code == 0, result.stdout
    assert "Overall Performance" in result.stdout
    assert "By Setup Type" in result.stdout
    assert "By Ticker" in result.stdout


def test_trade_analytics_cli_prints_excluded_count(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    TradeTracker().add_trade(_trade(trade_id="included", realized_pnl=100.0))
    TradeTracker().add_trade(
        _trade(
            trade_id="excluded",
            realized_pnl=-999.0,
            exclude_from_analytics=True,
            exclusion_reason="paper workflow test",
        )
    )

    result = cli_runner.invoke(app, ["trade-analytics"])

    assert result.exit_code == 0, result.stdout
    assert "Excluded trades:    1" in result.stdout


def test_trade_analytics_cli_empty_history(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = cli_runner.invoke(app, ["trade-analytics"])

    assert result.exit_code == 0, result.stdout
    assert "No closed manual trades found." in result.stdout


def test_trade_analytics_avg_setup_confidence(tmp_path):
    _tracker_with_trades(tmp_path / "manual_trades.jsonl", [
        _trade(trade_id="c1", realized_pnl=100.0, setup_confidence=0.6),
        _trade(trade_id="c2", realized_pnl=-20.0, setup_confidence=0.8),
    ])

    result = TradeAnalytics(tmp_path / "manual_trades.jsonl").analyze()

    assert result.overall["average_setup_confidence"] == pytest.approx(0.7)


def test_trade_analytics_avg_setup_score(tmp_path):
    _tracker_with_trades(tmp_path / "manual_trades.jsonl", [
        _trade(trade_id="s1", realized_pnl=100.0, setup_score=3.0),
        _trade(trade_id="s2", realized_pnl=-20.0, setup_score=5.0),
    ])

    result = TradeAnalytics(tmp_path / "manual_trades.jsonl").analyze()

    assert result.overall["average_setup_score"] == pytest.approx(4.0)


def test_trade_analytics_breakdown_by_setup_rank(tmp_path):
    _tracker_with_trades(tmp_path / "manual_trades.jsonl", [
        _trade(trade_id="r1", realized_pnl=100.0, setup_rank=1),
        _trade(trade_id="r2", realized_pnl=-20.0, setup_rank=1),
        _trade(trade_id="r3", realized_pnl=40.0, setup_rank=2),
    ])

    result = TradeAnalytics(tmp_path / "manual_trades.jsonl").analyze()
    by_rank = {row["setup_rank"]: row for row in result.by_setup_rank}

    assert by_rank[1]["total_trades"] == 2
    assert by_rank[1]["total_realized_pnl"] == pytest.approx(80.0)
    assert by_rank[2]["win_rate"] == 1.0


def test_trade_analytics_breakdown_by_test_type_and_mistakes(tmp_path):
    _tracker_with_trades(tmp_path / "manual_trades.jsonl", [
        _trade(trade_id="f1", test_type="forward_test", realized_pnl=100.0),
        _trade(trade_id="f2", test_type="forward_test", realized_pnl=-20.0, mistake=True),
        _trade(trade_id="p1", test_type="live_paper", realized_pnl=40.0),
    ])

    result = TradeAnalytics(tmp_path / "manual_trades.jsonl").analyze()
    by_test_type = {row["test_type"]: row for row in result.by_test_type}

    assert by_test_type["forward_test"]["total_trades"] == 2
    assert by_test_type["forward_test"]["mistake_count"] == 1
    assert by_test_type["live_paper"]["total_realized_pnl"] == pytest.approx(40.0)
    assert result.overall["mistake_count"] == 1


def test_trade_analytics_missing_setup_fields_graceful(tmp_path):
    _tracker_with_trades(tmp_path / "manual_trades.jsonl", [
        _trade(trade_id="old", realized_pnl=100.0),
    ])

    result = TradeAnalytics(tmp_path / "manual_trades.jsonl").analyze()

    assert result.overall["average_setup_confidence"] == 0.0
    assert result.overall["average_setup_score"] == 0.0
    assert result.by_setup_rank == []


def test_setup_alerts_recent_prints_recent_alerts(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    records = [
        {
            "timestamp": "2026-06-08T10:00:00",
            "ticker": "SPY",
            "setup_type": "vwap_reclaim_long",
            "rank": 1,
            "score": 3.5,
            "confidence": 0.7,
            "entry_trigger": 100.0,
            "stop_loss": 98.0,
            "target_1": 104.0,
            "target_2": 106.0,
        }
    ]
    (log_dir / "setup_alerts.jsonl").write_text("\n".join(json.dumps(record) for record in records))

    result = cli_runner.invoke(app, ["setup-alerts-recent"])

    assert result.exit_code == 0, result.stdout
    assert "Recent Setup Alerts" in result.stdout
    assert "SPY" in result.stdout
    assert "vwap_reclaim_long" in result.stdout


def test_stable_alert_id_generated_for_old_alert_records():
    import main as m

    record = {
        "timestamp": "2026-06-08T10:00:00",
        "ticker": "SPY",
        "setup_type": "vwap_reclaim_long",
        "entry_trigger": 100.0,
    }

    assert m._stable_setup_alert_id(record) == m._stable_setup_alert_id(record)
    assert len(m._stable_setup_alert_id(record)) == 12


def test_dashboard_trade_analytics_loading(tmp_path):
    from dashboard.helpers import load_jsonl, summarize_trade_analytics

    log_file = tmp_path / "manual_trades.jsonl"
    records = [
        _trade(trade_id="w", ticker="SPY", realized_pnl=100.0).to_dict(),
        _trade(trade_id="l", ticker="QQQ", realized_pnl=-50.0).to_dict(),
    ]
    log_file.write_text("\n".join(json.dumps(record) for record in records))

    df = load_jsonl(log_file)
    summary = summarize_trade_analytics(df)

    assert summary["overall"]["total_trades"] == 2
    assert summary["overall"]["win_rate"] == 0.5
    assert {row["ticker"] for row in summary["by_ticker"]} == {"SPY", "QQQ"}
