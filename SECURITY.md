# Security policy

## Release status and disclosure

The source repository remains PRIVATE pending separate owner approval for public
release. The legacy archive and local backups must always remain private.

For a vulnerability, use GitHub's **Security > Report a vulnerability** on this
repository when private vulnerability reporting is enabled:
https://github.com/bryanflynn2017/ai_regime_trader/security/advisories/new

The maintainer must enable and verify that route before public release. If it is
unavailable, do not create a public vulnerability issue: request a private contact
channel from the maintainer without including exploit details or sensitive data.
No private email address or response-time guarantee is published here.

Include the affected commit, impact, and minimal synthetic reproduction. Never
include real keys, authorization headers, account identifiers, raw broker responses,
SQLite ledgers, trading history, or private backups. Do not test against someone
else's account. Only the current source branch is maintained; no supported live
trading release exists. Coordinate fixes privately before publishing details.

## Boundaries

- Shadow is the default. No live-money execution is supported. Paper activation
  and exact-order human approvals remain mandatory for entries.
- Never weaken endpoint verification, risk, cutoff, data freshness, ownership,
  approval binding or recovery to make tests pass. Exits/cleanup remain automatic.
- Local OS access is the approval trust boundary; this is not a multi-user service.
  Keep the workstation and runtime directories access-controlled. Do not expose
  the dashboard, paper ledger or approval CLI through a public server.
- Settings secrets are redacted in representations and ordinary dict/JSON exports.
  Values must be explicitly unwrapped at SDK/HTTP boundaries. This is accidental
  disclosure protection, not encryption or protection against trusted Python code.
  Do not dump validation-error inputs, memory, SDK objects or raw provider responses.
- No real credentials in tests. `.env` must never be committed or shared. Only a
  blank reviewed `.env.example` is allowed. Never use live keys with this application.

## Contributor and CI security

Use hosted disposable runners with no broker secrets. The offline test wrapper
clears inherited credentials and blocks network entry points for trusted tests;
it is not an OS sandbox against hostile PR code or package install hooks.
Review untrusted contributions before local execution. Do not run public PRs on
the trading host or a self-hosted runner holding secrets.

Keep workflow permissions read-only, checkout credentials non-persistent, Actions
pinned to reviewed full SHAs, and scanner downloads checksum-verified. Never use
`pull_request_target` to execute PR code. Require approval for outside-contributor
workflows where available. Do not automatically approve PRs or upload workspaces.

Before committing or pushing:

1. Review the exact staged file list and diff, including new files.
2. Run `python -B scripts/publication_policy.py` and `git diff --cached --check`.
3. Run targeted/full offline tests appropriate to the change.
4. Scan source and the exact outgoing history with pinned Gitleaks, `--redact=100`.
5. Obtain explicit commit/push authorization and report the resulting SHAs/checks.

Ignore rules do not protect already tracked files or prevent deliberate force-add.
Require reviews and successful `offline-tests` and `secret-scan` checks on main;
disallow force pushes/deletion. Verify push protection, secret scanning, dependency
alerts, and private vulnerability reporting before public launch. Settings changes
require owner authorization and must not weaken existing controls.

If an actual secret is exposed, stop publication and notify the owner privately.
The owner should revoke/rotate it, review provider activity, and authorize any
history/cache remediation. A later deletion commit or private visibility does not
erase forks, clones, Actions logs, or cached copies. Never rotate credentials or
rewrite history silently.
