# ai_regime_trader

A Python 3.11+ intraday trading research and shadow-testing command center, with
an explicitly activated, human-approved Alpaca **paper-only** execution lifecycle.
This source-only snapshot contains no account data, credentials or trading history.

Public-release preparation is in progress. The repository remains PRIVATE until
the owner separately approves visibility changes. This is experimental research
software, not investment advice, a profitability claim, or production certification.

## Safety boundary

- Shadow/simulation is the default; live execution is unavailable.
- The frozen strategy includes a 13:00 ET new-entry cutoff.
- Each paper entry requires exact-order, short-lived approval and revalidation.
- Exits and emergency cleanup do not depend on interactive approval.
- No CI job may submit orders, contact a broker, or use broker credentials.
- A passing unit suite does not certify broker acceptance or profitability.
- Monitoring depends on the process/network; this is not an unattended deployment.

Do not copy runtime state from another checkout or run simultaneous execution
owners against one account. Keep secrets local and runtime backups encrypted and
separate from Git. Never commit `.env`, journals, databases or approval ledgers.

## Offline installation and tests

From a clean source checkout, create an isolated environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python scripts/offline_tests.py
```

Dependency installation requires internet access. Test execution is network-blocked
and credential-isolated by the wrapper. The wrapper copies source into a temporary
workspace so tests cannot read this checkout's `.env` or write its runtime state.
It does not import or start a trading workflow. For a focused run:

```powershell
python scripts/offline_tests.py tests/test_paper_approvals.py tests/test_paper_execution.py
```

Install from `requirements.txt`, not `pip install .`: the inherited packaging
backend metadata has not been certified for package builds. Dependency minimums
are retained from upstream; only the Alpaca SDK and requests versions are pinned.
A fully locked clean-install validation is still recommended before deployment.

## Architecture

- `data_providers/`, `data/`: bars, freshness and feature engineering.
- `strategy/`: deterministic ORB, VWAP reclaim and relative-volume setup detection/ranking.
- `trades/`: journals, monitoring, shadow sessions and paper supervision.
- `broker/`: paper transport, durable order ledger, approvals and reconciliation.
- `risk/`: position sizing, exposure limits and circuit breakers.
- `backtest/`, `regime/`: older regime-based research path, separate from intraday execution.
- `alerts/`, `dashboard/`: reporting and read-only review.
- `tests/`: original automated suite, with mocked broker submissions.

The optional HMM is not the active intraday decision engine. Saved profitability
claims and account records are deliberately absent. The legacy backtest accounting
and walk-forward limitations identified during review must be addressed before using
that path to promote strategies. Strategy invention and autonomous promotion are
not implemented by this publication snapshot.

See [publication notes](docs_vault/Publication.md) and
[security policy](SECURITY.md). Concept notes preserve design context and may
describe historical milestones; code and current safety policy take precedence.

## License and contributions

Original project code and documentation are licensed under the [MIT License](LICENSE),
copyright (c) 2026 Bryan Flynn. Dependencies retain their own licenses; see
[third-party review](THIRD_PARTY.md). No market-data redistribution rights are granted.

Use small feature branches and pull requests, synthetic fixtures, offline tests,
and redacted secret scans. Do not upload account records or credentials in issues,
screenshots, logs, or PRs. Report vulnerabilities privately as described in SECURITY.md.

Settings use redacted secret fields. Supply credentials only through local environment
variables or an ignored, access-controlled `.env`; never put live brokerage keys in
this application. Serialization is diagnostic, not a credential backup/round trip.
Explicit secret extraction occurs only at SDK/HTTP boundaries. Secret wrappers are
not encryption: trusted Python code, debuggers and memory dumps can still access values.
Do not expose the local dashboard or approval database to untrusted users.
