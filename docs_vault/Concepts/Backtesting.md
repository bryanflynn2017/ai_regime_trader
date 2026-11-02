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
   retain its last mark, generate no signal or fill, and keep its order pending.
   At the end, expire outstanding orders; do not liquidate open positions.

Replay uses the union of input dates and each symbol's own usable-bar warmup.
Missing or nonfinite OHLCV, inconsistent ranges, duplicate bar labels,
nonpositive prices/volume and explicitly nontradable rows are not market
prints. Whole-bar validity is a historical data admission rule, not a claim
that daily volume or extremes were known at the open. Opening risk checks do
not use the day's range, close or volume to price or size orders. Daily bars
cannot resolve auction liquidity, halts within a valid bar, impact, queues or
partial fills. Slippage is a fixed adverse cost represented in the fill price.

Inspect `BacktestResult.metadata["orders"]` for order timing, fills and
cancellations. The manifest's simulation section records
`daily-open-gap-stop-v1` and `completed-close-to-next-usable-open-v1`.
Accounting stays at version 2. Runs without these timing/fill versions used
different execution assumptions: rerun comparisons; do not relabel old results.

## Related Commands
- `python main.py backtest --synthetic`
- `python main.py backtest --csv path/to/data.csv`

## Role
Backtesting is required before trusting new strategies or AI-generated ideas.

## Related Notes
- [[Walk-Forward Validation]]
- [[Strategy Optimization]]
- [[AI Strategy Generation]]
- [[Long-Term AI Direction]]
- [[Safety Rules]]
