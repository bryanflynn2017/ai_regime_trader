# Source publication instructions

- Preserve the frozen strategy, 13:00 ET cutoff, risk calculations and paper-only guards.
- Never place orders or enable live execution as part of development or CI.
- Every paper entry requires exact-order human consent; do not auto-approve proposals.
- Do not read, print, commit or upload credentials, account identifiers or runtime ledgers.
- Keep dashboard access read-only. Do not add audible alerts.
- Run tests with `python scripts/offline_tests.py`, not an account-connected workflow.
- Use only synthetic fixtures and fake broker submissions in tests.
- Never copy `.git`, `.env`, environments, logs, databases or state from the original checkout.
- A new feature requires an explicit task; do not change strategy behavior to make CI pass.
- Private GitHub publication and pushes require owner authorization.
