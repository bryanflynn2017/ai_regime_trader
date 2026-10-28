# Security and publication policy

This repository is intended to remain private. Report vulnerabilities privately to
the repository owner; do not put credentials, account identifiers, journals or raw
scanner matches in issues or pull requests. Rotate confirmed exposed credentials
before investigating history cleanup.

No live-money execution is supported. Do not weaken paper endpoint verification,
approval binding, cutoff, stale-data gates, risk controls, ownership or recovery to
make a test pass. Broker tests must use fakes and deny external network access.

Before publication or every push:

1. Run `python scripts/publication_policy.py` after staging reviewed files.
2. Run a pinned Gitleaks scan with `--redact=100` over the source and Git history.
3. Run `python scripts/offline_tests.py` and inspect all failures.
4. Review staged changes and `git diff --cached --check`.

Ignore rules are not authorization controls: force-add can bypass them. Require CI
and review on the protected default branch, enable private-repository secret scanning
and push protection where the account plan supports them, and limit collaborator access.
Never configure production/broker secrets in Actions. CI is not a trading scheduler.

Third-party Actions and scanner downloads must be pinned to immutable revisions or
checksums. Do not use untrusted pull-request code with elevated workflow permissions.
Dependencies are installed from PyPI; their provenance and vulnerability review remain
separate from a clean secret scan. Do not upload entire workspaces as CI artifacts.
