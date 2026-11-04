"""Opt-in US-equity research contract. No provider fetches or broker dependencies.

Consume raw provider timestamps, BEFORE the legacy normalizers strip timezones.
Contract v1 uses XNYS regular sessions; extended-hours inclusion is explicit.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version

import exchange_calendars as xcals
import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]
TZ = "America/New_York"
VERSION = "us-equity-pit-v1"


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class DataContract:
    provider: str
    adjustment: str
    timestamp_kind: str  # start, end, or daily session_label
    timeframe: str = "5Min"
    source_timezone: str | None = None  # Explicit interpretation for naive intraday input.
    sessions: str = "regular"

    def __post_init__(self):
        choices = {
            "provider": ("alpaca", "yahoo", "synthetic", "csv"),
            "adjustment": ("raw", "split", "split_dividend"),
            "timestamp_kind": ("start", "end", "session_label"),
            "timeframe": ("1Min", "5Min", "15Min", "1Day"),
            "source_timezone": (None, "UTC", TZ),
            "sessions": ("regular", "all"),
        }
        for key, allowed in choices.items():
            if getattr(self, key) not in allowed:
                raise ContractError("Unsupported contract field: " + key)
        if (self.timeframe == "1Day") != (self.timestamp_kind == "session_label"):
            raise ContractError("Daily input requires explicit session_label semantics")
        if self.timeframe == "1Day" and self.sessions != "regular":
            raise ContractError("Daily all-hours aggregation is not supported")


def _utc(value, source_timezone=None):
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            raise ValueError()
        if ts.tzinfo is None:
            if source_timezone is None:
                raise ValueError()
            ts = ts.tz_localize(source_timezone, ambiguous="raise", nonexistent="raise")
        return ts.tz_convert("UTC")
    except (ValueError, TypeError) as exc:
        raise ContractError("Timestamp needs an unambiguous declared timezone") from exc


class ResearchBars:
    """Normalized snapshot; accessors return copies. Receipt times are never inferred."""

    def __init__(self, original, bars, contract, diagnostics, schedule):
        self._original = original.copy(deep=True)
        self._bars = bars.copy(deep=True)
        self.contract = contract
        self._diagnostics = dict(diagnostics)
        self._schedule = schedule.copy()

    @property
    def original(self):
        return self._original.copy(deep=True)

    @property
    def bars(self):
        return self._bars.copy(deep=True)

    def as_of(self, observation_time):
        """No completed OHLCV row before its end AND declared availability."""
        at = _utc(observation_time)
        return self._bars.loc[self._bars["available_at"] <= at].copy(deep=True)

    def features_as_of(self, observation_time):
        """Backward-looking existing features, computed only from observable rows.

        This is a research view, not the frozen intraday strategy or a replay adapter.
        Regular-only filtering happens before any feature calculation.
        """
        from data.feature_engineering import add_features
        visible = self.as_of(observation_time)
        if visible.empty:
            return visible
        values = visible[OHLCV].copy()
        values.index = pd.DatetimeIndex(visible["bar_start"]).tz_convert(TZ)
        result = add_features(values)
        # A later event's feature can depend on a delayed earlier event. Never
        # backdate that dependency to the event label.
        result["feature_available_at"] = visible["available_at"].cummax().to_numpy()
        return result

    def missing_regular_starts(self):
        """Expected regular-session slots across the supplied date span; no fills."""
        expected = []
        for _, row in self._schedule.iterrows():
            if self.contract.timeframe == "1Day":
                expected.append(row["open"])
            else:
                step = pd.Timedelta(self.contract.timeframe.replace("Min", "min"))
                expected.extend(pd.date_range(row["open"], row["close"], freq=step, inclusive="left"))
        observed = pd.DatetimeIndex(self._bars.loc[self._bars["session_type"] == "regular", "bar_start"])
        return pd.DatetimeIndex(expected).difference(observed)

    def provenance(self):
        return {
            "version": VERSION, **asdict(self.contract), "event_timezone": "UTC",
            "session_timezone": TZ, "calendar": "XNYS",
            "calendar_version": version("exchange-calendars"),
            "tzdata_version": version("tzdata"),
            "availability_policy": ("declared-receipt" if "available_at" in self._original
                                    else "assumed-no-earlier-than-bar-end"),
            "adjustment_point_in_time_verified": False,
            "corporate_action_accounting": "not-implemented",
            "timing_sha256": sha256(pd.util.hash_pandas_object(
                self._bars[["bar_start", "bar_end", "available_at", "original_timestamp"]],
                index=True).to_numpy(dtype="<u8").tobytes()).hexdigest(),
            **self._diagnostics,
            "missing_regular_intervals": len(self.missing_regular_starts()),
        }

    def to_daily_research(self, *, allow_retrospective_adjustments=False):
        """Explicit bridge to the legacy daily replay, not an intraday adapter.

        Reject delayed receipts: the current replay cannot schedule them causally.
        """
        if self.contract.timeframe != "1Day":
            raise ContractError("Daily replay cannot consume intraday contract bars")
        if (self._bars["available_at"] != self._bars["bar_end"]).any():
            raise ContractError("Delayed daily replay requires a future event-time adapter")
        if self.contract.adjustment != "raw" and not allow_retrospective_adjustments:
            raise ContractError("Adjusted history requires explicit retrospective research consent")
        for name in ("dividends", "stock splits", "split", "dividend"):
            if name in self._original:
                values = pd.to_numeric(self._original[name], errors="coerce")
                if values.isna().any() or values.ne(0).any():
                    raise ContractError("Corporate-action portfolio accounting is not implemented")
        out = self._bars[OHLCV].copy()
        out.index = pd.DatetimeIndex(self._bars["session_date"]).as_unit("ns")
        out.attrs = {"research_data_contract": self.provenance()}
        return out


def normalize_research_bars(raw, contract):
    """Fail closed on malformed/duplicate/overlapping bars; sort out-of-order input.

    Daily session labels are dates, not instants. Intraday event timestamps retain
    their declared start/end meaning. Half-open [start,end) regular bars must fit
    fully inside the calendar session. Crossing a boundary is rejected.
    """
    if raw.empty or not raw.columns.is_unique or not set(OHLCV).issubset(raw.columns):
        raise ContractError("Expected nonempty single-symbol lowercase OHLCV data")
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.hasnans or not raw.index.is_unique:
        raise ContractError("Missing or duplicate event timestamps")
    # Do not silently override a known input basis/provider.
    for name, expected in (("adjustment_policy", contract.adjustment), ("provider", contract.provider)):
        if name in raw.attrs and raw.attrs[name] != expected:
            raise ContractError("Input metadata conflicts with contract")
        if name in raw and not raw[name].eq(expected).all():
            raise ContractError("Mixed or undeclared provider/adjustment basis")
    prices = raw[OHLCV].apply(pd.to_numeric, errors="coerce")
    valid = (np.isfinite(prices).all(axis=1) & (prices[OHLCV[:4]] > 0).all(axis=1)
             & (prices.volume >= 0) & (prices.low <= prices[["open", "close"]].min(axis=1))
             & (prices.high >= prices[["open", "close"]].max(axis=1)))
    if not valid.all():
        raise ContractError("Malformed OHLCV; repair upstream without fabricating prints")

    if contract.timeframe == "1Day":
        if any(ts != ts.normalize() for ts in raw.index):
            raise ContractError("Daily session labels must be midnight date labels")
        local_dates = [ts.date() for ts in raw.index]
        events = None
    else:
        events = [_utc(ts, contract.source_timezone) for ts in raw.index]
        step = pd.Timedelta(contract.timeframe.replace("Min", "min"))
        starts = [ts if contract.timestamp_kind == "start" else ts - step for ts in events]
        local_dates = [ts.tz_convert(TZ).date() for ts in starts]

    first, last = pd.Timestamp(min(local_dates)), pd.Timestamp(max(local_dates))
    # Explicit bounds avoid a calendar horizon determined by today's wall clock.
    calendar = xcals.get_calendar("XNYS", start=first - pd.Timedelta(days=370),
                                  end=last + pd.Timedelta(days=370))
    schedule = calendar.schedule.loc[first:last, ["open", "close"]]
    rows = []
    for i, day in enumerate(local_dates):
        label = pd.Timestamp(day)
        if label not in schedule.index:
            raise ContractError("Bar is on a non-session date")
        opening, closing = schedule.loc[label, ["open", "close"]]
        if contract.timeframe == "1Day":
            start, end, event = opening, closing, opening
        else:
            start, end, event = starts[i], starts[i] + step, events[i]
        regular = start >= opening and end <= closing
        if not regular and start < closing and end > opening:
            raise ContractError("Bar straddles a regular-session boundary")
        available = _utc(raw.iloc[i]["available_at"]) if "available_at" in raw else end
        if available < end:
            raise ContractError("Completed bar cannot be available before its end")
        rows.append({
            **prices.iloc[i].to_dict(), "original_timestamp": raw.index[i],
            "event_timestamp": event, "bar_start": start, "bar_end": end,
            "available_at": available, "session_date": day,
            "session_type": "regular" if regular else "extended",
        })
    bars = pd.DataFrame(rows).sort_values("bar_start")
    if bars["bar_start"].duplicated().any():
        raise ContractError("Duplicate normalized bar boundaries")
    if len(bars) > 1 and (bars["bar_start"].iloc[1:].reset_index(drop=True)
                         < bars["bar_end"].iloc[:-1].reset_index(drop=True)).any():
        raise ContractError("Overlapping bars are not supported")
    excluded = 0
    if contract.sessions == "regular":
        excluded = int((bars["session_type"] != "regular").sum())
        bars = bars.loc[bars["session_type"] == "regular"]
    bars = bars.set_index("event_timestamp", drop=False)
    diagnostics = {"out_of_order_input": not raw.index.is_monotonic_increasing,
                   "excluded_extended_bars": excluded,
                   "delayed_bars": int((bars.available_at > bars.bar_end).sum())}
    return ResearchBars(raw, bars, contract, diagnostics, schedule)


def safe_contract_metadata(frame):
    """Allowlisted metadata for exported manifests; never dump arbitrary attrs."""
    value = frame.attrs.get("research_data_contract")
    if value is None:
        return {"version": "legacy-unverified", "availability_policy": "unverified"}
    if not isinstance(value, dict) or value.get("version") != VERSION:
        raise ContractError("Unknown research contract metadata")
    config = DataContract(**{key: value.get(key) for key in DataContract.__dataclass_fields__})
    result = {"version": VERSION, **asdict(config)}
    enums = {
        "event_timezone": ("UTC",), "session_timezone": (TZ,), "calendar": ("XNYS",),
        "availability_policy": ("declared-receipt", "assumed-no-earlier-than-bar-end"),
        "corporate_action_accounting": ("not-implemented",),
    }
    for key, allowed in enums.items():
        if value.get(key) not in allowed:
            raise ContractError("Invalid contract provenance")
        result[key] = value[key]
    for key in ("out_of_order_input", "adjustment_point_in_time_verified"):
        if type(value.get(key)) is not bool:
            raise ContractError("Invalid provenance flag")
        result[key] = value[key]
    for key in ("excluded_extended_bars", "delayed_bars", "missing_regular_intervals"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ContractError("Invalid provenance count")
        result[key] = value[key]
    import re
    for key, pattern in (("calendar_version", r"[0-9]+(?:\.[0-9]+)*"),
                         ("tzdata_version", r"[0-9]+(?:\.[0-9]+)*"),
                         ("timing_sha256", r"[0-9a-f]{64}")):
        if not isinstance(value.get(key), str) or not re.fullmatch(pattern, value[key]):
            raise ContractError("Invalid provenance version or digest")
        result[key] = value[key]
    return result
