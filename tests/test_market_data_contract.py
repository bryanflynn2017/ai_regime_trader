"""Offline contract tests: calendar rules and synthetic raw provider responses only."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from backtest.market_data_contract import (
    ContractError, DataContract, normalize_research_bars, safe_contract_metadata,
)


def raw(times):
    return pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100., "volume": 1000.},
                        index=pd.DatetimeIndex(times))


def contract(**changes):
    return replace(DataContract("synthetic", "raw", "start"), **changes)


@pytest.mark.parametrize("date,opening", [
    ("2024-03-08", "14:30"), ("2024-03-11", "13:30"),
    ("2024-11-01", "13:30"), ("2024-11-04", "14:30"),
])
def test_dst_preserves_local_session_and_utc_boundaries(date, opening):
    frame = raw([date + " 09:30"]).tz_localize("America/New_York")
    result = normalize_research_bars(frame, contract())
    bar = result.bars.iloc[0]
    assert str(bar.bar_start) == date + " " + opening + ":00+00:00"
    assert bar.session_date == pd.Timestamp(date).date()
    assert bar.original_timestamp == frame.index[0]
    assert result.bars.index.tz is not None
    pd.testing.assert_frame_equal(result.original, frame)


def test_early_close_regular_extended_and_boundary():
    times = ["2024-11-29 09:25", "2024-11-29 09:30",
             "2024-11-29 12:55", "2024-11-29 13:00"]
    df = raw(times).tz_localize("America/New_York")
    normalized = normalize_research_bars(df, contract())
    assert len(normalized.bars) == 2
    assert normalized.bars.iloc[-1].bar_end == pd.Timestamp("2024-11-29 18:00Z")
    assert normalized.provenance()["excluded_extended_bars"] == 2
    all_hours = normalize_research_bars(df, contract(sessions="all"))
    assert list(all_hours.bars.session_type) == ["extended", "regular", "regular", "extended"]
    with pytest.raises(ContractError, match="straddles"):
        normalize_research_bars(raw(["2024-11-29 17:58Z"]), contract())


@pytest.mark.parametrize("holiday", ["2024-07-04", "2024-11-28", "2024-03-10"])
def test_holidays_and_weekends_rejected(holiday):
    with pytest.raises(ContractError, match="non-session"):
        normalize_research_bars(raw([holiday + " 14:30Z"]), contract())


def test_available_only_at_completion_or_later_receipt():
    df = raw(["2024-03-11 13:30Z", "2024-03-11 13:35Z"])
    df["available_at"] = pd.to_datetime(["2024-03-11 13:37Z", "2024-03-11 13:40Z"])
    bars = normalize_research_bars(df, contract())
    assert bars.as_of("2024-03-11 13:36Z").empty
    assert len(bars.as_of("2024-03-11 13:37Z")) == 1
    assert bars.provenance()["delayed_bars"] == 1
    df.iloc[0, df.columns.get_loc("available_at")] = pd.Timestamp("2024-03-11 13:34Z")
    with pytest.raises(ContractError, match="before its end"):
        normalize_research_bars(df, contract())


def test_cross_provider_start_end_and_naive_semantics():
    yahoo = raw(["2024-03-11 09:30"]).tz_localize("America/New_York")
    alpaca = raw(["2024-03-11 13:30Z"])
    a = normalize_research_bars(yahoo, contract(provider="yahoo")).bars
    b = normalize_research_bars(alpaca, contract(provider="alpaca")).bars
    pd.testing.assert_series_equal(a.bar_start, b.bar_start)
    pd.testing.assert_series_equal(a.bar_end, b.bar_end)
    naive = raw(["2024-03-11 09:30"])
    with pytest.raises(ContractError, match="timezone"):
        normalize_research_bars(naive, contract())
    explicit = normalize_research_bars(naive, contract(source_timezone="America/New_York"))
    assert explicit.bars.iloc[0].bar_start == a.iloc[0].bar_start
    end = normalize_research_bars(raw(["2024-03-11 13:35Z"]), contract(timestamp_kind="end"))
    assert end.bars.iloc[0].bar_start == a.iloc[0].bar_start


def test_missing_duplicates_order_and_malformed():
    df = raw(["2024-03-11 13:40Z", "2024-03-11 13:30Z"])
    bars = normalize_research_bars(df, contract())
    assert bars.bars.index.is_monotonic_increasing
    assert bars.provenance()["out_of_order_input"]
    assert pd.Timestamp("2024-03-11 13:35Z") in bars.missing_regular_starts()
    assert len(bars.bars) == 2  # Missing bars are never synthesized.
    with pytest.raises(ContractError, match="duplicate"):
        normalize_research_bars(pd.concat([df, df]), contract())
    with pytest.raises(ContractError, match="Missing"):
        normalize_research_bars(raw([pd.NaT]), contract())
    df.loc[df.index[0], "volume"] = np.nan
    with pytest.raises(ContractError, match="Malformed"):
        normalize_research_bars(df, contract())


def test_adjustment_metadata_preserved_and_never_mixed():
    df = raw(["2024-03-11 13:30Z"])
    df.attrs = {"adjustment_policy": "split", "private_note": "fixture-private"}
    df["stock splits"] = 0
    normalized = normalize_research_bars(df, contract(adjustment="split"))
    pd.testing.assert_frame_equal(normalized.original, df)
    assert normalized.provenance()["adjustment"] == "split"
    assert "fixture-private" not in json.dumps(normalized.provenance())
    with pytest.raises(ContractError, match="conflicts"):
        normalize_research_bars(df, contract())
    df.attrs = {}
    df["adjustment_policy"] = "split"
    with pytest.raises(ContractError, match="Mixed"):
        normalize_research_bars(df, contract())


def test_features_use_only_observed_regular_data():
    df = raw(["2024-03-11 13:25Z", "2024-03-11 13:30Z", "2024-03-11 13:35Z"])
    df.iloc[0, :4] = [10000, 10001, 9999, 10000]  # Premarket must not contaminate VWAP.
    normalized = normalize_research_bars(df, contract())
    before = normalized.features_as_of("2024-03-11 13:35Z")
    assert len(before) == 1 and before.iloc[-1]["session_vwap"] == 100
    assert normalized.features_as_of("2024-03-11 13:34:59Z").empty
    changed = df.copy()
    changed.iloc[-1, :4] = [200, 201, 199, 200]
    after = normalize_research_bars(changed, contract()).features_as_of("2024-03-11 13:35Z")
    pd.testing.assert_frame_equal(before, after)


def test_daily_bridge_versions_early_close_and_legacy_compatibility():
    import exchange_calendars as xcals
    from backtest.backtester import Backtester
    cal = xcals.get_calendar("XNYS", start="2023-01-01", end="2024-12-31")
    dates = cal.sessions[:260]
    daily = DataContract("synthetic", "raw", "session_label", timeframe="1Day")
    normalized = normalize_research_bars(raw(dates), daily)
    frame = normalized.to_daily_research()
    bt = Backtester()
    legacy = bt.run({"ABC": raw(dates)})
    strict = bt.run_contracts({"ABC": normalized})
    pd.testing.assert_series_equal(legacy.equity_curve, strict.equity_curve)
    sim = strict.config_snapshot["research_manifest"]["simulation"]
    assert sim["accounting_version"] == 2 and sim["fill_model_version"].endswith("v2")
    source = strict.config_snapshot["research_manifest"]["inputs"][0]["data_contract"]
    assert source["version"] == "us-equity-pit-v1"
    assert legacy.config_snapshot["research_manifest"]["inputs"][0]["data_contract"]["version"] == "legacy-unverified"
    early = normalize_research_bars(raw(["2024-11-29"]), daily)
    assert early.bars.iloc[0].bar_end == pd.Timestamp("2024-11-29 18:00Z")
    frame.attrs["research_data_contract"]["secret"] = "fixture-private"
    assert "fixture-private" not in json.dumps(safe_contract_metadata(frame))


def test_daily_bridge_rejects_unsupported_latency_intraday_and_actions():
    daily = DataContract("synthetic", "raw", "session_label", timeframe="1Day")
    df = raw(["2024-11-29"])
    df["available_at"] = [pd.Timestamp("2024-11-29 18:01Z")]
    with pytest.raises(ContractError, match="Delayed"):
        normalize_research_bars(df, daily).to_daily_research()
    with pytest.raises(ContractError, match="intraday"):
        normalize_research_bars(raw(["2024-11-29 14:30Z"]), contract()).to_daily_research()
    df = raw(["2024-11-29"])
    adjusted = normalize_research_bars(df, replace(daily, adjustment="split_dividend"))
    with pytest.raises(ContractError, match="retrospective"):
        adjusted.to_daily_research()
    assert len(adjusted.to_daily_research(allow_retrospective_adjustments=True)) == 1
    df["dividends"] = .5
    with pytest.raises(ContractError, match="Corporate-action"):
        normalize_research_bars(df, daily).to_daily_research()


def test_delayed_feature_dependency_cannot_be_backdated():
    df = raw(["2024-03-11 13:30Z", "2024-03-11 13:35Z"])
    df["available_at"] = pd.to_datetime(["2024-03-11 13:45Z", "2024-03-11 13:40Z"])
    bars = normalize_research_bars(df, contract())
    assert len(bars.features_as_of("2024-03-11 13:40Z")) == 1
    complete = bars.features_as_of("2024-03-11 13:45Z")
    assert (complete.feature_available_at == pd.Timestamp("2024-03-11 13:45Z")).all()


def test_split_events_and_mixed_portfolio_inputs_fail_closed():
    from backtest.backtester import Backtester
    daily = DataContract("synthetic", "raw", "session_label", timeframe="1Day")
    df = raw(["2024-11-29"])
    df["stock splits"] = 2.
    with pytest.raises(ContractError, match="Corporate-action"):
        normalize_research_bars(df, daily).to_daily_research()
    base = normalize_research_bars(raw(["2024-11-29"]), daily)
    adjusted = normalize_research_bars(raw(["2024-11-29"]), replace(daily, adjustment="split"))
    with pytest.raises(ContractError, match="mix"):
        Backtester().run_contracts({"ABC": base, "DEF": adjusted},
                                  allow_retrospective_adjustments=True)
    with pytest.raises(ContractError, match="mix"):
        Backtester().run({"ABC": base.to_daily_research(), "DEF": raw(["2024-11-29"])})


def test_provenance_rejects_private_values_and_overlapping_timestamps():
    daily = DataContract("synthetic", "raw", "session_label", timeframe="1Day")
    frame = normalize_research_bars(raw(["2024-11-29"]), daily).to_daily_research()
    frame.attrs["research_data_contract"]["provider"] = "/private/fixture"
    with pytest.raises(ContractError, match="provider"):
        safe_contract_metadata(frame)
    with pytest.raises(ContractError, match="Overlapping"):
        normalize_research_bars(raw(["2024-03-11 13:30Z", "2024-03-11 13:32Z"]), contract())


def test_delayed_receipt_revises_vwap_without_backdating_or_mutating_prior_snapshot():
    df = raw(["2024-03-11 13:30Z", "2024-03-11 13:35Z"])
    df.iloc[0, :4] = [100, 100, 100, 100]
    df.iloc[1, :4] = [200, 200, 200, 200]
    df["available_at"] = pd.to_datetime(["2024-03-11 13:45Z", "2024-03-11 13:40Z"])
    bars = normalize_research_bars(df, contract())
    earlier = bars.features_as_of("2024-03-11 13:40Z")
    saved = earlier.copy(deep=True)
    assert list(earlier.session_vwap) == [200]
    assert earlier.index[0] == pd.Timestamp("2024-03-11 09:35", tz="America/New_York")
    assert earlier.feature_available_at.iloc[0] == pd.Timestamp("2024-03-11 13:40Z")
    later = bars.features_as_of("2024-03-11 13:45Z")
    assert list(later.session_vwap) == [100, 150]
    assert list(later.feature_available_at) == [pd.Timestamp("2024-03-11 13:45Z")] * 2
    pd.testing.assert_frame_equal(saved, earlier)
    pd.testing.assert_frame_equal(saved, bars.features_as_of("2024-03-11 13:40Z"))
    # Caller mutations of a returned snapshot must not change subsequent queries.
    later.iloc[-1, later.columns.get_loc("session_vwap")] = -1
    assert bars.features_as_of("2024-03-11 13:45Z").session_vwap.iloc[-1] == 150


@pytest.mark.parametrize("receipts", [(35, 40, 45), (55, 50, 45), (50, 40, 45)])
@pytest.mark.parametrize("response_order", [(0, 1, 2), (2, 0, 1)])
def test_snapshot_values_follow_event_order_not_receipt_or_response_order(receipts, response_order):
    df = raw(["2024-03-11 13:30Z", "2024-03-11 13:35Z", "2024-03-11 13:40Z"])
    prices = [100., 200., 400.]
    for i, price in enumerate(prices):
        df.iloc[i, :4] = [price] * 4
    df["available_at"] = pd.to_datetime([f"2024-03-11 13:{m}:00Z" for m in receipts])
    bars = normalize_research_bars(df.iloc[list(response_order)], contract())
    saved = {}
    # Deliberately query backwards and repeatedly: no retained arrival-state.
    for minute in (55, 40, 50, 35, 45, 40, 55):
        at = pd.Timestamp(f"2024-03-11 13:{minute}:00Z")
        result = bars.features_as_of(at)
        known = [i for i in range(3) if receipts[i] <= minute]
        assert len(result) == len(known)
        if known:
            # Equal-volume VWAP, independently derived in market-event order.
            assert list(result.session_vwap) == pytest.approx([
                sum(prices[i] for i in known[:n]) / n for n in range(1, len(known) + 1)])
            expected_available = [
                pd.Timestamp(f"2024-03-11 13:{max(receipts[i] for i in known[:n])}:00Z")
                for n in range(1, len(known) + 1)]
            assert list(result.feature_available_at) == expected_available
            assert (result.feature_available_at <= at).all()
            assert list(result.index) == [df.index[i].tz_convert("America/New_York") for i in known]
        if minute in saved:
            pd.testing.assert_frame_equal(saved[minute], result)
        saved[minute] = result.copy(deep=True)
