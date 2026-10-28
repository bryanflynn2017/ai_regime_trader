# Provider Freshness Diagnostics

Read-only shadow-session evidence. This does not change entry/exit rules, the
13:00 ET cutoff, ranking, sizing, ticker selection, or CLEAN/PARTIAL_DATA rules.

## Current implementation (2026-09-07)

- `forward-test-session` calls `_run_shadow_scan_once`, which calls
  `get_multiple_bars`. Alpaca inherits the base implementation: sequential
  `get_bars` calls, one symbol per `StockBarsRequest`, using `get_stock_bars`.
  It does **not** call the latest-bar endpoint. Each request includes the chosen
  timeframe (1Min or 5Min), feed, lookback start, and a newly generated UTC end.
- Freshness uses the maximum timestamp in each returned historical frame,
  not necessarily a bar returned by the latest-bar API. The normal console
  warning's wording "latest bar" refers to this newest returned historical bar.
- Timestamp source is the response dataframe timestamp index/column, or the
  bar object's `timestamp` (`t` for raw dictionaries). Aware timestamps are
  converted to UTC and made naive with `tz_convert(None)`. Naive timestamps
  remain unchanged and are treated as UTC, not ET.
- `_attach_freshness_metadata` subtracts the newest bar timestamp from
  `pd.Timestamp.now(tz="UTC").tz_convert(None)`, clamps negative ages to zero,
  and records `latest_bar_time`, `provider_time`, and `data_age_seconds`.
  Each symbol has its own comparison time because requests are sequential.
- `_warn_if_provider_data_stale` checks only provider name `AlpacaData` and
  returns true for **any** returned frame with age strictly greater than
  `max(180 seconds, 3 * timeframe minutes * 60)`: 3 minutes for 1Min and
  15 minutes for 5Min. Exactly at the threshold is not stale. It short-circuits
  on the first stale frame; the console age is not necessarily the maximum.
  The session's existing maximum-age diagnostic separately scans all frames.
- Individual `DataProviderError` failures are logged and omitted by
  `get_multiple_bars`. If all fail, it raises and the existing scan-failure
  path runs. Missing symbols do not themselves trip the stale gate. Returned
  frames without age metadata are also ignored by that gate. These semantics
  remain unchanged; missing/unknown does not mean fresh in diagnostics.
- Separately, `get_latest_bar` uses `StockLatestBarRequest` with one symbol
  and feed, then `get_stock_latest_bar` (`/stocks/bars/latest`). The installed
  SDK documents this as the latest **minute** bar; the request has no timeframe
  parameter. The provider's 5Min argument does not aggregate that response.
  If an injected client lacks the method, existing code uses one day of
  historical bars and takes the last row. No new fallback was added.

## Recorded evidence

Only stale events retain `freshness_snapshot` in `stale_data_events`. A snapshot
contains every requested symbol's normalized UTC latest timestamp, the original
provider comparison time expressed in ET with offset, age in minutes, stale
boolean (null if unknown), missing flag, provider, and feed. It also records
event time (existing naive ET convention), request kind, threshold, requested/
checked/stale counts, stale/missing/unknown lists, median/max age, and freshest/
stalest symbol. Reporting never recomputes ages using a later clock.

Classification is descriptive, not a diagnosis or eligibility gate:

- FEED_WIDE_STALE: all requested symbols have known ages and all are stale.
- SYMBOL_LOCAL_STALE: exactly one is stale and all others have known fresh ages.
- MIXED_FRESHNESS: several, but not all, are stale with complete age coverage.
- INSUFFICIENT_DATA: at least one requested symbol is missing/has unknown age,
  or there are no symbols. Available stale observations are still retained.
- ALL_FRESH: all have known fresh ages (available to the diagnostic helper;
  normal fresh scans do not persist detailed snapshots).

"Feed-wide" means the observed requested sample, not proof of a feed outage.
One requested stale symbol alone cannot establish an exchange-wide problem.
Missing means no usable returned frame; it does not distinguish empty API
responses from per-symbol fetch/validation errors. All-failed scans retain
existing failure counters but no fabricated per-symbol timestamps or snapshots.

Successful Alpaca scans also accumulate missing-bar observation counts without
per-symbol normal-console spam. A session with no successful scan has unavailable
missing-bar evidence, not zero. Summary, daily review, and analytics show detail
coverage, classification counts, stale-symbol observations, top five stale
symbols, worst detailed age, and missing-count coverage. Mixed old/new histories
report only recorded evidence; absent legacy symbol details are unavailable.

## Correctness fixes and interpretation

Review found wrong-symbol substitution in two response adapters: latest-bar
dictionaries could fall back to another symbol, and historical MultiIndex
selection failure could reset the index and use another symbol's rows. Both
now raise the existing data-provider error instead. No timestamp-arithmetic
bug was demonstrated in the supported UTC-naive/aware paths.

Review (2026-09-17): historical frames carrying a `symbol` column are also
filtered explicitly, preventing the same wrong-symbol substitution outside
the MultiIndex layout. Valid symbol responses are regression-tested unchanged.
Aggregation counts a symbol once within each event but counts repeated events
separately. Null missing counts and absent detailed ages display as unavailable.

Old session-level age maxima cannot distinguish symbol-local, feed-wide,
sparse trading, or historical-request semantics. Collect new stale-event
snapshots before attributing a cause or proposing a strategy/provider change.
PARTIAL_DATA remains invalid for strategy proof and excluded from clean analytics.

Related: [[Shadow Forward Testing]], [[Data Providers]], [[Clean Shadow Analytics]].
