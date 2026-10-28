# Shadow Forward Testing

Shadow forward testing is a simulation-only workflow that lets the app journal
what it would have done if the current setup/ranking/exit rules were followed
automatically.

## Current Status
Implemented and actively used for validation. [[Version 1.1]] blocks new shadow
entries at or after 13:00 ET by default.

## Related Commands
- `python main.py forward-test-session --provider alpaca --alpaca-feed iex --watchlist momentum_universe --timeframe 5Min --max-iterations 72 --close-open-at-session-end`
- `python main.py trade-analytics --include-shadow`
- `python main.py trade-analytics --include-shadow --clean-shadow-only`

## Role in Version 1.0
Supports [[Real-World Validation]] without placing real orders. Shadow trades
are separate from the [[Manual Trade Journal]] and are labeled as simulation
records.

## Safety Rules
- No real orders.
- No broker execution.
- Manual journal remains separate.
- Session lock prevents overlapping `forward-test-session` runs.
- Partial-data sessions are marked so they do not pollute clean analysis.
- Once a session becomes partial, no new entries are created; existing open
  trades may still be monitored and closed.
- New entries are blocked at or after the configurable `--entry-cutoff HH:MM`
  cutoff, default 13:00 ET.
- Entry cutoffs never disable monitoring or normal exits for existing trades.

## Session recovery and freshness diagnostics

Shadow-session lifecycle hardening (2026-09-07):

- Recovery requires an OPEN journal trade owned by the saved RUNNING,
  INTERRUPTED, or ERROR session (including a journal entry written just before
  a state snapshot was interrupted), or explicit persisted adoption intent.
  Adoption intent is saved before ownership metadata changes, so a failed
  adoption can be retried without orphaning trades. ENDED sessions are never adopted; an ENDED
  history record also vetoes an older unfinished state snapshot.
- Historical open trades without recovery evidence remain unchanged and are
  reported separately, not silently monitored or closed by a new session.
- Session-end cleanup includes adopted trades. Successful cleanup persists
  zero session-owned open trades in state and history. If price data is missing
  or invalid (non-numeric, non-finite, zero, or negative),
  no exit is invented: the session is ERROR with SESSION_END_CLOSE_INCOMPLETE
  and retains the open trade IDs for recovery.
- Lock creation is exclusive and lock release runs even if initialization or
  reporting raises an error. Each new session has a microsecond-resolution ID.
  A retained `.lock.guard` sidecar serializes lock reclamation/removal using an
  OS lock that releases on process exit. The sidecar alone is not an active or
  stale session and should not be deleted while sessions are starting/running.
  Lifecycle exceptions persist ERROR/INTERRUPTED with SESSION_LIFECYCLE_ERROR
  and the actual journal open set where storage is writable. A reporting error
  after ENDED was saved never makes that session recoverable again.
- Normal gap detection remains greater than max(15 minutes, 2.5 intervals).
  Gaps at least max(60 minutes, 12 intervals) are separately tagged
  SESSION_SUSPENSION. Gap detection immediately persists PARTIAL_DATA with
  ITERATION_GAP, before the next scan; the existing partial-session entry block
  applies. Monitoring and exits continue without reconstructing missed bars.
- Summary, daily review, and analytics show partial-data reasons, normal and
  extreme gap counts, maximum iteration gap, and provider/feed when known.
  Stale warning count, maximum stale bar age, first/last stale timestamps, and
  the stale-caused-partial flag remain available. STALE_DATA is displayed as
  STALE_PROVIDER_DATA; combined causes display MULTIPLE_REASONS.
- Older records retain their original clean-analytics classification. Missing
  partial reasons are labeled UNKNOWN_LEGACY_REASON when no cause can be
  inferred. Old unclassified gaps are not retroactively labeled suspension gaps.

PARTIAL_DATA sessions remain invalid for strategy proof and excluded from clean
analytics. This hardening does not change strategy rules, the 13:00 ET cutoff,
risk sizing, ranking, paper execution, or live execution. No real orders.

## Related Notes
- [[Clean Shadow Analytics]]
- [[Candidate Rule Tracking]]
- [[Trade Analytics]]
- [[Safety Rules]]
- [[Real-World Validation]]
