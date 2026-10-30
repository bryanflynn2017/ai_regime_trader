# Source publication and public-release gates

Updated 2026-10-30. **PRIVATE until separate owner authorization.** Preparing a
license, documentation or remediation branch does not authorize public visibility,
commits, pushes, merges, broker access or trading.

## Authoritative source and preserved history

The authoritative repository is `bryanflynn2017/ai_regime_trader`, GitHub ID
`1438747798`, developed from the `ai_regime_trader_source` checkout. The three-commit
reviewed baseline is `a181325b9857bbcdf32b3c17e22ad9b39c0dc72d`. It has no shared
ancestry with the 87-commit legacy history. A separate verified local mirror/bundle
preserves that old history; the legacy GitHub repository is PRIVATE and archived.
Never copy, merge or push legacy refs, backups, records or runtime artifacts here.
Never use `push --mirror` or `push --all` for source publication.

The original snapshot retained 144 reviewed files: application/tests, watchlists,
safe examples and selected concept notes. This remediation adds secret-handling
helpers, synthetic tests, LICENSE and third-party review documentation. Review the
actual candidate file inventory, not an old fixed count, before publication.

Excluded: `.env` and real credential variants, virtual environments, logs, databases
and sidecars, broker/approval/session state, historical trading/account records,
market-data caches, local tool settings and model artifacts. Only the reviewed blank
`.env.example` is allowed. An ignore rule does not protect a tracked or force-added file.

## Licensing and privacy

Original code/documentation carries the MIT License, copyright (c) 2026 Bryan Flynn.
See the root LICENSE and THIRD_PARTY.md for review scope and dependency/data caveats.
The owner-approved copyright name, GitHub username, no-reply identity, commit times
and history are intentional public information if release is later approved.

No real credentials or private trading records were found in the three-commit
baseline. Gitleaks and exact in-memory comparisons with known local credentials
passed, including the initial Actions logs. Scans are bounded evidence, not proof
that every unknown/encoded secret is absent. Never publish raw scanner findings.
The narrow synthetic test exception is limited by exact path AND exact fixture
expression; do not expand it to an entire rule or test directory.

Settings now redact secret fields in repr and ordinary dict/JSON serialization;
explicit extraction is restricted to SDK/HTTP consumption. Broker error messages
use fixed operations/categories, not raw responses or exception text. These controls
do not protect against malicious local code, explicit secret extraction, arbitrary
validation-input dumps, debuggers, or memory inspection. Keep runtime access private.

## CI/security review and recommended settings

The workflow remains unchanged: push to main and pull_request on hosted Ubuntu,
read-only contents permission, no persisted checkout credential, full-SHA-pinned
checkout/setup-python, checksum-verified Gitleaks, and no broker/deployment commands.
The offline wrapper clears credentials and blocks socket/HTTP/curl request entry
points before application imports. Dependencies download before that guard; neither
the guard nor secret wrappers are a hostile-code sandbox. No credential-bearing
self-hosted runners or local testing of untrusted PR code on the trading host.

Owner settings to verify before public release (not changed by this work):

1. Protect main with PR review and successful `offline-tests`/`secret-scan` checks;
   restrict bypasses and disallow force pushes/deletion. Review access permissions.
2. Enable and verify GitHub private vulnerability reporting; SECURITY.md describes
   the private route and a safe fallback without publishing personal contact details.
3. Enable/verify secret scanning, push protection, dependency graph and alerts as
   available. Require approval for outside-contributor workflows. Never execute
   untrusted PR code with `pull_request_target`, write tokens or broker secrets.
4. Review Actions logs/artifacts before visibility changes: existing logs become
   public. Do not upload complete workspaces or runtime data as artifacts.
5. Preserve the legacy archive's PRIVATE visibility independently of this repository.

Known dependency risks: most requirements use minimum-version ranges; only requests
and alpaca-py are pinned. No reproducible transitive lock or complete vulnerability
inventory is provided. Plan a separately reviewed locked-environment/SBOM/security
audit; do not silently upgrade dependencies during security remediation. The inherited
packaging backend is not a certified build path: use requirements.txt, not `pip install .`.

## Candidate validation and authorization

Run from the source-only checkout with dependencies installed:

```text
python -B scripts/offline_tests.py tests/test_secret_security.py tests/test_settings.py tests/test_broker_paper_safety.py tests/test_adapters.py tests/test_discord_alerts.py
python -B scripts/offline_tests.py
python -B scripts/publication_policy.py
git diff --check
gitleaks dir . --redact=100 --no-banner
gitleaks git . --redact=100 --no-banner --log-opts=HEAD
```

The index policy checks tracked/indexed paths. Also explicitly review every new
untracked candidate file before staging. Source scans cover uncommitted edits;
history scans cannot cover edits until an authorized commit exists. Repeat the
history scan after that commit and review the exact outgoing ancestry before push.
Use the existing exact synthetic exception, never a blanket test-file exclusion.

The earlier hosted baseline passed 1,132 tests. New remediation results must be
reported separately; do not claim hosted validation of an unpushed branch. Passing
tests is engineering evidence, not proof of profit or production brokerage acceptance.

Before release, review the entire proposed diff, dependency provenance, no-reply
metadata, all tracked files and history, and public Actions surfaces. Obtain separate
authorization for commit/push/merge and for visibility. No command in this document
changes repository visibility or submits orders. Public forks/clones cannot be
recalled merely by making a repository private again.
