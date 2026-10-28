# Safety Rules

v1.4 paper entries require exact-order, expiring human approval plus final
data/account/order/risk revalidation. Approval never changes sizing or extends
the 13:00 ET cutoff. Exits and emergency cleanup must not wait for approval.
See [[Version 1.4 Human-Approved Paper Execution]]. Live execution is unavailable.

Safety rules keep the project useful without drifting into unsafe automation.

## Current Status
Permanent project constraint.

## Role
Safety rules define the boundaries for [[Version 1.0]] and any future
[[Long-Term AI Direction|AI intelligence]] work.

Version 1.0 is conservative: no automatic live execution and manual trade entry.
Post-1.0 may become more autonomous in analysis, strategy generation, trade
proposal, and trade management recommendations, but the user remains the order
approval gate unless a later design explicitly changes that boundary.

## Rules
- No real-money orders. Explicitly activated v1.3 Alpaca PAPER orders are
  permitted only through the managed lifecycle in [[Version 1.3 Paper Execution]].
- No live trading enablement.
- No automatic live execution in Version 1.0.
- No audible beeps, winsound, or terminal bell.
- No secrets in logs, state, dashboard, or terminal output.
- Dashboard remains read-only.
- User manually approves and enters trades.
- Shadow forward testing is simulation-only and must stay separate from the
  manual trade journal.
- AI recommendations must be logged and testable.
- No strategy should be trusted without backtesting and forward testing.
- Candidate rules are research-only until explicitly promoted by a tested
  design change. The Version 1.1 13:00 ET shadow entry cutoff is promoted;
  remaining candidates are not.
- Promotion of a shadow rule does not authorize broker execution, live trading,
  or real-money automation.
- v1.3 integrates/hardens existing paper components for the frozen strategy;
  it defaults to shadow, fails closed on unverifiable paper status, and does not
  enable live routing. Follow [[Paper Execution Runbook]] for supervised acceptance.
- Post-1.0 systems may prepare recommendations and trade plans, but must not
  silently submit live orders.

## Related Notes
- [[Human-in-the-Loop Workflow]]
- [[Broker Adapters]]
- [[Dashboard]]
- [[Shadow Forward Testing]]
- [[Candidate Rule Tracking]]
- [[Backtesting]]
- [[Walk-Forward Validation]]
- [[Long-Term AI Direction]]
- [[AI Trade Decision Engine]]
