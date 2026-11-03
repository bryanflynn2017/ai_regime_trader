# Backtesting

Backtesting simulates strategy behavior on historical data. The following
execution conventions apply only to the legacy daily research simulator.

## Current Status
Implemented as a research and validation tool.

## Daily execution timeline

1. At a usable bar's close, value holdings and observe the completed data.
   After close, evaluate the signal against the current portfolio. An approved
   market order is pending, not filled. Dates label sessions; phases order
   events within a session without inventing precise timestamps.
2. At the next usable open for that symbol, mark available positions at their
   opens and process resting gap stops first. Recheck pending orders in ticker
   order against current cash, exposure and the existing risk rules. A BUY
   uses the slipped opening price and the original absolute bracket. Cancel an
   invalidated bracket or a failed check; never enlarge the previously approved
   share count. Signal exits also require an extant position and risk approval.
3. For positions still open (including new fills), check the bar's range.
   When both levels were touched and the path is unknown, assume the stop
   first. A long stop uses the lesser of open and stop, less slippage; a target
   uses the target less slippage, without favorable gap improvement.
4. Mark remaining positions to close. On a symbol's missing/unusable bars,
   retain its last mark, generate no signal or fill, and keep its order pending
   (explicit opening flags have the distinct behavior described below).
   At the end, expire outstanding orders; do not liquidate open positions.

### Eligibility and lifecycle correction (v2)

Opening eligibility and retrospective data integrity are separate policies.
For admitted records, daily volume (including zero) and later high/low/close
do not choose the opening fill or quantity. A caller may supply a boolean
`opening_tradable` known at the open; false or unknown values prevent execution.
Without that column, the model ASSUMES the valid opening quote is available.
This is not evidence of liquidity, an auction fill or actual market access.
When an explicit flag blocks the open, this conservative daily model also
skips bracket fills for that session because reopening time is unknown, but
can still mark and observe valid completed-bar data at close.

The separately versioned retrospective quarantine still rejects nonfinite or
inconsistent OHLCV, negative volume, nonpositive prices, duplicate labels and
legacy `tradable=False` records. Thus invalid later prints CAN exclude a whole
record; these results remain conditional on a cleaned historical dataset.
Zero daily volume alone is no longer such an exclusion.

A new non-HOLD closing signal explicitly cancels an older pending order before
evaluating the replacement, even if the replacement fails risk checks. HOLD,
missing data and quarantined bars leave an order pending. All order records
retain transition history with session labels, phases and reasons, including
supersession, risk/bracket cancellation, fill and end-of-data expiration.

Replay uses the union of input dates and each symbol's own usable-bar warmup.
Missing or nonfinite OHLCV, inconsistent ranges, duplicate bar labels,
nonpositive prices, negative volume and explicitly nontradable rows are not market
prints. Whole-bar validity is a historical data admission rule, not a claim
that daily volume or extremes were known at the open. Opening risk checks do
not use the day's range, close or volume to price or size orders. Daily bars
cannot resolve auction liquidity, halts within a valid bar, impact, queues or
partial fills. Slippage is a fixed adverse cost represented in the fill price.

Inspect `BacktestResult.metadata["orders"]` for order timing, fills and
cancellations. The manifest's simulation section records
`daily-open-gap-stop-v2` and `completed-close-to-next-eligible-open-v2`,
plus opening-eligibility, data-quality and order-lifecycle policies and
`opening_liquidity_verified=False`.
Accounting stays at version 2. Runs without these timing/fill versions used
different execution assumptions: rerun comparisons; do not relabel old results.

## Related Commands
- `python main.py backtest --synthetic`
- `python main.py backtest --csv path/to/data.csv`

## Role
Backtesting is required before trusting new strategies or AI-generated ideas.

## Related Notes
- [[Research Market Data Contract]] — opt-in UTC/session/availability contract;
  legacy inputs remain explicitly unverified.
- [[Walk-Forward Validation]]
- [[Strategy Optimization]]
- [[AI Strategy Generation]]
- [[Long-Term AI Direction]]
- [[Safety Rules]]
