# Research market data contract v1

## Scope and migration

PR #6 is the timestamp/session foundation, not a replacement for the shared data
providers, frozen intraday strategies, or broker execution. The new API is
opt-in in `backtest/market_data_contract.py`. No network call is made by it.

Existing `Backtester.run(DataFrames)` is unchanged economically. Its inputs are
now labeled `legacy-unverified` rather than retroactively claiming timestamps
or adjustments were verified. Do not relabel or overwrite historical results.

Use raw provider responses BEFORE legacy normalization. Timezones, duplicate
records or adjustment evidence already discarded upstream cannot be recovered.
The caller must select one symbol and provide lowercase OHLCV columns.

## Audited existing paths

- `data/market_data.py::_strip_timezone` drops Yahoo timezone information while
  retaining local wall time. `fetch_ticker` requests `auto_adjust=True`;
  its cleaning path discards action columns. A naive old index is not proof
  of UTC. Raw Yahoo daily date labels require explicit session-label semantics.
- `data_providers/alpaca_provider.py::_normalize_frame` converts aware timestamps
  to UTC-naive, discards invalid/duplicate records and keeps OHLCV columns only.
  Historical request construction does not explicitly declare an adjustment
  policy. Do not infer raw/adjusted status from provider name or credentials.
- `data_providers/base.py::get_bars` promises sorted/deduplicated OHLCV but has
  no observation-time or adjustment contract.
- `data/feature_engineering.py::add_session_intraday_features` groups normalized
  calendar dates and constructs a fixed row-count opening range. This PR does
  not alter that implementation or its strategy interpretation.
- `backtest/backtester.py` uses date labels and open/close phases, not real receipt
  times. Its causal fill/accounting versions remain v2.

## Explicit contract

`DataContract(provider, adjustment, timestamp_kind, timeframe, source_timezone,
sessions)` uses enumerated values, not unrestricted provenance strings.

- Provider is alpaca, yahoo, synthetic or csv. This is a caller declaration,
  not authenticated evidence of origin; feed-specific fetch adapters remain future work.
- Adjustment is raw, split, or split_dividend. No price adjustment is performed.
  Conflicting per-row or attribute declarations are rejected. Preserve the raw
  response and action columns via the snapshot's `original` accessor.
- Intraday timestamps must be aware, or require explicit UTC/New York
  interpretation. Ambiguous/nonexistent local times fail rather than guess.
  A timestamp is explicitly bar start or bar end. Canonical event timestamps
  and half-open [bar_start, bar_end) boundaries are UTC-aware.
- Daily timestamps are explicitly date labels, not instants. They must be
  midnight labels. Their calendar date identifies the session; canonical event
  time is the session open and completion is its actual close.
- Session dates are America/New_York dates. XNYS rules from pinned
  exchange-calendars 4.13.2 supply holidays, DST-aware opens, and early closes.
  Calendar bounds derive from input dates, not the machine's current date.
- Regular bars must fit entirely inside the session. Boundary-straddling bars
  and bars on non-session dates are rejected. Default regular-only mode excludes
  extended bars BEFORE calculating features; their count is reported.
  Explicit `sessions="all"` includes same-session-date extended bars without
  claiming that the calendar verifies venue-specific extended-hours liquidity.
- Completed bars cannot be available before bar end. A supplied timezone-aware
  `available_at` is the declared receipt time. Without one, availability at end
  is a labeled approximation, not verified provider latency.
- `as_of(utc_time)` and `features_as_of(utc_time)` expose only available bars.
  The latter uses existing backward-looking feature functions and adds
  `feature_available_at`: a cumulative availability maximum so a delayed prior
  row cannot backdate a later feature that depends on it.
- Out-of-order responses are sorted and flagged. Duplicates, overlaps, NaT,
  nonfinite/inconsistent prices, negative volume and missing columns fail closed.
  Zero volume is valid data, not proof of opening liquidity.
  `missing_regular_starts()` lists expected slots across full regular sessions
  in the input date span; it never forward-fills or invents bars. Counts can
  include intentionally partial first/last sessions and are not an outage diagnosis.

## Minimal offline usage

```python
from backtest.market_data_contract import DataContract, normalize_research_bars

spec = DataContract(
    provider="synthetic", adjustment="raw", timestamp_kind="start",
    timeframe="5Min", source_timezone="America/New_York",
)
snapshot = normalize_research_bars(raw_single_symbol_frame, spec)
visible = snapshot.features_as_of("2024-03-11T13:35:00Z")
```

A 09:30 ET five-minute bar starts 14:30 UTC on March 8, 2024, but 13:30
UTC on March 11. It is not visible at 09:34:59 ET. On November 29, 2024,
daily completion is 13:00 ET / 18:00 UTC, not 16:00 ET.

## Daily compatibility bridge and exclusions

`Backtester.run_contracts({"ABC": snapshot})` accepts regular DAILY snapshots
whose availability equals session close. It intentionally converts session dates
to legacy daily labels and embeds the sanitized contract in research provenance.
The normalized snapshot retains the original timezone-aware data separately.
There is no new broker or intraday replay path.

Delayed daily receipts and intraday snapshots are rejected by this bridge:
scheduling those requires a separate event-time replay PR. Raw and adjusted
portfolios, or contracted and legacy-unverified inputs, cannot be silently mixed.

Adjusted snapshots require `allow_retrospective_adjustments=True`. This is
explicit retrospective research consent, not point-in-time verification.
Provider back-adjustments can incorporate subsequently known splits/dividends.
The bridge rejects supplied nonzero split/dividend events because the research
portfolio does not yet implement share/cash corporate-action accounting.
Absent event columns do not prove no event occurred. Raw histories with missing
action evidence can still produce misleading returns and require data review.
No claim of verified profitability is justified by contract admission alone.

## Provenance and next PRs

Each contracted research input carries contract/calendar/tzdata versions, enum
provider and adjustment declarations, availability policy, missing/excluded/delayed
counts, an original-timestamp/availability digest, and the usual OHLCV fingerprint.
Private attributes, paths, original data and corporate-action records are NOT
copied into the exported manifest. They remain local snapshot data.
The calendar dependency is Apache-2.0 and is not vendored:
[package/version](https://pypi.org/project/exchange_calendars/4.13.2/).

Follow-ups: raw-provider ingestion adapters preserving receipt/action revisions;
event-time replay for delayed and intraday data; duration-based opening ranges
and same-time volume research; split/dividend cash/share accounting and
point-in-time adjustment/universe datasets. Do not modify deployed strategies
as an incidental part of those research changes.
