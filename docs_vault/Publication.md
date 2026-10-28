# Source-only publication review

Date: 2026-10-28. **Prepared for owner review and private source publication;
not approved for unattended trading deployment.** No GitHub repository, remote,
push, broker call or order was created by this preparation.

## Provenance and separation

This is an independent sibling directory named `ai_regime_trader_source`, not a
worktree sharing the original repository's object database. It starts with a new
`.git` directory on `main`; reviewed files are staged, but there is deliberately
no initial commit until owner review. No original branches, refs, objects,
reflogs, hooks or remote configuration were copied.

Source baseline: `004530221b34455beca593e304dc82a2c7c81dda`. The original repository
and its history were preserved. Its pre-existing untracked security/competitive
review remains in the original vault. That report was the starting point, but
was not copied here because it contains machine-local paths and private trading
context. Do not push the original repository or use `--mirror`.

## Included and excluded inventory

The snapshot contains 144 reviewed files:

| Group | Files | Treatment |
|---|---:|---|
| Application, tests, watchlists, dependency manifests | 92 | Byte-identical to current original files; SHA-256 comparison found zero omissions or differences |
| `docs_vault/Concepts/*.md` | 41 | Design and safety notes retained; historical/design statements are not proof of production acceptance |
| Publication controls, README, blank template, runbook and this report | 11 | New or replacement publication-only files |

The 92-file set includes `main.py`, `requirements.txt`, `pyproject.toml`, and all
tracked files under `alerts`, `backtest`, `broker`, `config`, `dashboard`, `data`,
`data_providers`, `regime`, `risk`, `strategy`, `tests`, `trades`, `utils`, and
`watchlists`. All 30 test-package files (28 test modules) are retained. Comparing
the remaining tracked non-environment paths identified only replaced root policy
files, documentation/local Obsidian settings, and intentionally excluded runtime
artifacts; no required application resource was omitted.

Excluded entirely:

- Original `.git` history, virtual environments, installed dependencies, binaries,
  caches, editor settings, local agent settings and Obsidian workspace settings.
- `.env` and all credential-bearing variants; only a new blank `.env.example` is included.
- `state`, SQLite databases and sidecars, broker journals, approval ledgers,
  session locks, logs, archived logs, daily reviews, result/status exports,
  runtime market-data files and saved models.
- Historical performance/status notes, the full private audit report and other
  root vault notes outside the reviewed Concepts allowlist. No account balances,
  broker identifiers, actual orders or private trading records are published.

Watchlists and synthetic in-code test fixtures are source configuration, not
runtime records. `data/` contains Python source, not a copied market-data cache.

## Security validation

- All included files were covered by content screening, including exact in-memory
  comparisons with three currently configured local secret values; no matches.
  Secret values were not printed, written to the snapshot or sent to scanners
  for online credential validation.
- Independent Gitleaks **8.30.1**, official Windows x64 release, verified SHA-256:
  `d29144deff3a68aa93ced33dddf84b7fdc26070add4aa0f4513094c8332afc4e`.
- The default source scan found one synthetic redaction-test fixture in
  `tests/test_broker_paper_safety.py:266`, not an owner credential. The final
  default-rules-plus-exception scan passed with **zero findings**. `.gitleaks.toml`
  allows only the exact fake fixture value or literal variable assignment AND that exact test path, not an
  entire test directory or rule. Scanning the full test file through stdin,
  without the permitted file path, still detects the fixture (exit 1).
- Broad ignores cover environments, nested logs, state, databases/sidecars,
  credentials and local artifacts. Eleven representative sensitive paths were
  confirmed ignored; `.env.example` remains includable. The index policy also
  rejects prohibited tracked paths, including force-added files. Twenty policy
  assertions passed. Ignore rules are not a guarantee against deliberate bypass.
- `git diff --cached --check` passed. The index contains only reviewed files.
- No old history is present to scan. Run the history scanner on the initial
  commit before the first push; CI also scans all available new history.

Scanning is bounded evidence, not a proof that every possible secret format or
encoded value is absent. The discarded environment/old history is not cleared
for publication by this source-only review.

### Initial-commit verification follow-up

The authorized initial source-only commit was created after the preparation
review. The first Git-history scan flagged the literal code expression
`ALPACA_SECRET_KEY=secret` in the same redaction test. In-memory comparison
confirmed this is a variable assignment, not a credential. The exception now
also includes that exact expression, still restricted to the exact test path.
This scanner-only correction is recorded in a separate follow-up commit; no
application or test code changed. The preparation-time statements and example
commands below describe the original pre-commit state: do not repeat initial
commit creation. Scan all new history before any separately authorized push.

## Tests and CI review

Final command, using an existing Python 3.11 interpreter with installed dependencies:

```powershell
python -B scripts/offline_tests.py
```

Result: **1,132 passed, 250 warnings, 0 failures in 70.47 seconds**. A prior run
also passed 1,132 tests. The final run additionally blocked HTTP and native-curl
request entry points. Warnings include deliberate negative safety fixtures,
websockets deprecation and a joblib physical-core detection fallback. Live-flag
warnings arise from tests verifying refusal, not an activated trading process.

The wrapper copies only application/test source into a disposable directory,
clears inherited credentials, uses a temporary home, disables pytest plugin
autoload, and blocks socket/DNS, urllib, requests and curl_cffi network calls
before application imports. Tests use their existing fake transports. This is
defense in depth for trusted tests, not an OS sandbox for hostile code. No broker
workflow is launched. The original environment was used only as the interpreter
and installed-library source; it is not included in this repository.

GitHub Actions is restricted to dependency installation, index policy, offline
tests and redacted source/history secret scans. It has read-only repository
permissions, pinned checkout/setup-python commit SHAs, disabled persisted checkout
credentials, and a checksum-verified scanner download. It has no broker secrets,
schedule, deployment, artifact upload, or order command. Dependency installation
and scanner download require network access; test execution does not.

Dependency review found no credential URLs, private package endpoints or execution
hooks added by this snapshot. Original minimum-version constraints are retained;
only Alpaca SDK and requests are pinned. A fresh Linux installation/hosted Actions
run and comprehensive dependency vulnerability audit have **not** been performed.
The inherited `pyproject.toml` build backend is not a certified package-install
path: use `pip install -r requirements.txt`, not `pip install .`. Do not confuse
successful local tests with reproducible dependency resolution on another host.

## Publication commands — owner action only

These commands were **not executed**. They create an initial source-only commit,
a **private** repository, and the first push only after review/authorization.
Install GitHub CLI and Gitleaks 8.30.1 separately from their official releases;
verify the scanner checksum above. Do not copy an old `.git` directory here.

```powershell
Set-Location (Join-Path $env:USERPROFILE 'Desktop\ai_regime_trader_source')
# Use the Python environment in which requirements.txt has been installed.
python -B scripts/offline_tests.py
if ($LASTEXITCODE -ne 0) { throw 'Offline tests failed' }
git add -- .
python -B scripts/publication_policy.py
if ($LASTEXITCODE -ne 0) { throw 'Publication policy failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Staged whitespace check failed' }
gitleaks dir . --redact=100 --no-banner
if ($LASTEXITCODE -ne 0) { throw 'Source scan failed' }
git diff --cached --stat
# Inspect the complete staged diff locally before continuing.
git commit -m 'Initial reviewed source-only snapshot'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
gitleaks git . --redact=100 --no-banner --log-opts=--all
if ($LASTEXITCODE -ne 0) { throw 'History scan failed' }

gh auth login --hostname github.com --git-protocol https --web
if ($LASTEXITCODE -ne 0) { throw 'Authentication failed' }
gh auth setup-git
if ($LASTEXITCODE -ne 0) { throw 'Git authentication setup failed' }
$owner = gh api user --jq .login
if ($LASTEXITCODE -ne 0 -or !$owner) { throw 'Cannot determine GitHub owner' }
$repo = "$owner/ai_regime_trader"
gh repo create $repo --private --description 'Private source-only shadow and supervised paper research'
if ($LASTEXITCODE -ne 0) { throw 'Repository creation failed; do not reuse an unknown repository' }
$visibility = gh repo view $repo --json visibility --jq .visibility
if ($LASTEXITCODE -ne 0 -or $visibility.Trim() -ne 'PRIVATE') { throw 'Private visibility not verified' }
git remote add origin "https://github.com/$repo.git"
if ($LASTEXITCODE -ne 0) { throw 'Remote already exists or creation failed; review it' }
git push -u origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
```

The commands target the authenticated personal account. For an organization,
explicitly replace `$owner` with the approved organization before repository
creation. Do not put authentication tokens into remote URLs or command arguments.
If account identity for commits is missing, configure an owner-approved Git name
and GitHub no-reply email locally; do not invent an identity.

After the initial push, require pull requests, reviews, and both CI jobs on `main`;
disallow force pushes/deletion and minimize collaborators. Enable secret scanning,
push protection and dependency alerts where the private-repository plan supports
them. Do not assume those controls are available or enabled until verified in
Settings. Never add broker credentials to repository or Actions secrets. Require
approval for outside-contributor workflows and avoid `pull_request_target` execution.

For future synchronization, work in this clean repository (or a clone of it), use
reviewed feature branches and pull requests, and repeat the staged path/content
checks. Do not merge the original history or recursively copy its working folder.
Runtime availability/backups are separate from GitHub source synchronization.

Official references: [GitHub CLI repository creation](https://cli.github.com/manual/gh_repo_create),
[Gitleaks source and usage](https://github.com/gitleaks/gitleaks),
[verified scanner release](https://github.com/gitleaks/gitleaks/releases/tag/v8.30.1).

## Remaining decisions and limitations

1. Owner review, initial commit, post-commit history scan and explicit authorization
   are still required before publication. No remote or push has occurred.
2. Verify hosted CI, private visibility and protection settings after authorized
   creation. Fresh dependency resolution and a vulnerability/license inventory
   remain follow-up work; this is not a production deployment certificate.
3. Resolve ownership/license policy before redistribution: inherited metadata says
   MIT, but the original has no corresponding root license grant. No competitor
   source was copied into this snapshot.
4. Historical concept links to intentionally omitted private notes may be unresolved.
   The new runbook intentionally omits order-activation commands. Any account-connected
   paper acceptance requires a separate task and authorization.
5. Original known operational/backtest limitations remain. Strategy, 13:00 ET
   cutoff, sizing, approvals, monitoring, recovery and paper/live safeguards were
   not changed. This snapshot does not authorize unattended or live trading.
