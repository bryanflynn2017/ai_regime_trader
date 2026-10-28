# Candidate Rule Tracking

Candidate rule tracking monitors proposed filters over future clean shadow
samples. Candidates do not affect behavior unless an explicit tested design
change promotes one.

## Current Status
Implemented in [[Clean Shadow Analytics]]. `candidate_no_entries_after_13` was
promoted into active [[Version 1.1]] shadow behavior; the tracker remains in
place to compare historical and prospective evidence.

## Related Commands
- `python main.py trade-analytics --include-shadow --clean-shadow-only`

## Promoted Rule
- `candidate_no_entries_after_13` — promoted in Version 1.1. New shadow entries
  are blocked at or after 13:00 ET by default. The cutoff is configurable with
  `--entry-cutoff HH:MM`; monitoring and exits remain active.

## Research-Only Candidate Rules
- `candidate_no_orb_after_13`
- `candidate_exclude_arm_qcom_crwd`
- `candidate_exclude_arm_qcom`
- `candidate_no_hrv`

## Interpretation
Each candidate compares baseline clean shadow performance to a hypothetical
filtered sample:

- `HELPFUL so far`: P&L delta is positive and profit factor improves.
- `HARMFUL so far`: P&L delta is negative.
- `MIXED/INSUFFICIENT`: evidence is inconclusive.

## Role
Postpone additional rule changes until a candidate remains useful over more
clean trades and is explicitly promoted. Continue tracking the promoted 13:00
cutoff prospectively to determine whether it improves future results rather
than only the historical sample. This is guardrail work against overfitting.

## Related Notes
- [[Clean Shadow Analytics]]
- [[Shadow Forward Testing]]
- [[Strategy Optimization]]
- [[Real-World Validation]]
- [[Safety Rules]]
