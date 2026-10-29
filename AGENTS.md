# Source publication instructions

## Primary development checkout and GitHub

- Develop in the `ai_regime_trader_source` checkout, not the sibling
  `ai_regime_trader` archive/runtime directory. Verify `git rev-parse --show-toplevel`
  and use an explicit working directory before every edit or Git operation.
- Expected origin: `https://github.com/bryanflynn2017/ai_regime_trader.git`.
  Verify the authenticated owner is `bryanflynn2017` and the repository is private.
- Use existing Git and authenticated GitHub CLI for requested operations. On
  Windows, if `gh` is not on PATH, use `C:/Program Files/GitHub CLI/gh.exe`.
  The connected GitHub integration can inspect pushed code, not local-only edits.
  Never ask the user to paste tokens; use browser authentication when required.
- Never print credential helper responses, token values, or full environment dumps.
  Keep Git authentication outside project files and credential-bearing remote URLs.
- Preserve the original checkout. Never copy its `.git`, credentials, environments,
  logs, account records, databases, or approval ledgers into this checkout.
- Before synchronization, fetch and compare histories. If branches diverge or have
  unrelated roots, stop and request direction. Never automatically merge unrelated
  histories, force-push, mirror-push, reset, or replace user work.
- The authoritative development history is the reviewed source-only snapshot.
  Legacy history belongs only in the separate local backup and private archived
  `ai_regime_trader_legacy_20261029` repository. Never merge or push legacy refs,
  trading records, or runtime artifacts into this repository.

## Development and publication workflow

- Use `codex/<topic>` feature branches for substantial changes; inspect existing
  changes first and preserve work belonging to the user.
- Commit and push only when the user explicitly requests each action. Prefer pull
  requests for substantial changes. Never create test commits or test pushes.
- Run relevant offline tests before committing, and the complete offline suite for
  broad or safety-sensitive changes: `python -B scripts/offline_tests.py`.
  Use an environment with requirements installed; never run broker workflows in CI.
- Before committing/pushing, review the exact staged diff, run
  `python -B scripts/publication_policy.py`, `git diff --cached --check`, and pinned
  Gitleaks source/history scans with full redaction. Review the exact outgoing
  commit range; scan failures or unreviewed history block publication.
- `.env` and all real credential variants must NEVER be staged, committed, shared,
  attached, or printed. Only the reviewed blank `.env.example` is permitted.
  Do not use `git add -f` to bypass exclusions. Check the index as well as ignore
  rules: ignored files already tracked are still publishable.
- Keep Alpaca keys, account identifiers, trade records, runtime data, and broker
  state out of Git. Never put broker credentials into GitHub Actions secrets.
- Report changed files, exact test results, commit SHAs (if committed), PR URLs
  (if created), and any incomplete security checks. Never claim local edits are
  visible through GitHub until their authorized push is verified.

## Trading safety

- Preserve the frozen strategy, 13:00 ET cutoff, risk calculations and paper-only guards.
- Never place orders or enable live execution as part of development or CI.
- Every paper entry requires exact-order human consent; do not auto-approve proposals.
- Do not read, print, commit or upload credentials, account identifiers or runtime ledgers.
- Keep dashboard access read-only. Do not add audible alerts.
- Run tests with `python scripts/offline_tests.py`, not an account-connected workflow.
- Use only synthetic fixtures and fake broker submissions in tests.
- Never copy `.git`, `.env`, environments, logs, databases or state from the original checkout.
- A new feature requires an explicit task; do not change strategy behavior to make CI pass.
- Never weaken independent exposure, loss, sizing, or risk limits merely to make
  a strategy or test pass. Diagnose the failure and request any needed design change.
- Never enable live trading. Broker-order submission requires a separate explicit
  authorization for supervised paper acceptance and the existing exact-order
  approval safeguards; development, Git operations and CI never authorize orders.
- Private GitHub publication and pushes require owner authorization.
