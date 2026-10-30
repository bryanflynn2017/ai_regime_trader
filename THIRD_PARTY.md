# Licensing and provenance review

Review date: 2026-10-30. The owner authorized the MIT grant for original project
code/documentation with copyright holder Bryan Flynn. The source-only inventory
contains no vendored environments, dependency source trees, binaries, model weights,
or market-data datasets. Review of source attribution/copyright markers found no
conflicting third-party notices or copied competitor implementation. No third-party
implementation was introduced by the public-release remediation.

This is a bounded source review, not independent proof of authorship of every line.
Contributors must identify copied/adapted material, its source and compatible license;
retain required notices. Do not assume the project MIT grant relicenses dependencies
or data supplied by external providers.

## Dependency declarations (not bundled code)

Observed installed direct-package metadata was reviewed without importing trading
workflows. These are license families, not a complete locked transitive SBOM:

| Packages | Observed license family |
|---|---|
| pandas, scipy, scikit-learn, hmmlearn, python-dotenv | BSD family |
| numpy | BSD-3-Clause plus component 0BSD/MIT/Zlib/CC0 notices |
| yfinance, alpaca-py, streamlit, requests | Apache-2.0 |
| pydantic, pydantic-settings, plotly, loguru, typer, pytest, pytest-cov | MIT |

Requirements remain unchanged. Minimum-version ranges mean installations can
resolve different versions. Before distributing a container, executable, environment
or vendored package, inventory the exact transitive versions and preserve all
applicable copyright, license, NOTICE and bundled-library notices. No blanket
license-compatibility or vulnerability certification is asserted for future versions.

Market-data rights are separate from software licenses. Do not publish downloaded
Yahoo/Alpaca/IEX/SIP bars, account exports or datasets unless the relevant provider
terms and permissions allow it. yfinance's upstream documentation specifically
distinguishes its software license from Yahoo data-use terms.

Sources:

- MIT text: https://choosealicense.com/licenses/mit/
- yfinance licensing/data guidance: https://github.com/ranaroussi/yfinance
- Alpaca SDK license: https://github.com/alpacahq/alpaca-py/blob/master/LICENSE

No additional third-party attribution is added merely for calling dependencies.
New third-party copied code or redistributed artifacts require a fresh review.
