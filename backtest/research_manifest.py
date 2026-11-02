"""Allowlisted provenance for legacy research; no environment/config dumps."""
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import subprocess

import pandas as pd


# Only numeric/boolean inputs consumed by the daily strategy and risk engine.
FIELDS = (
    "PAPER_TRADING", "LIVE_TRADING_ALLOWED", "REQUIRE_STOP_LOSS",
    "MAX_TRADES_PER_DAY", "MAX_OPEN_POSITIONS", "MAX_PORTFOLIO_EXPOSURE",
    "MAX_SINGLE_POSITION_EXPOSURE", "MAX_RISK_PER_TRADE", "ALLOW_FRACTIONAL_SHARES",
    "MAX_TOTAL_DRAWDOWN_LOCK", "MAX_DAILY_DRAWDOWN_STOP", "MAX_WEEKLY_DRAWDOWN_STOP",
    "MAX_DAILY_DRAWDOWN_WARN", "MIN_PRICE", "MAX_PRICE", "MIN_AVG_DAILY_VOLUME",
    "REGIME_STABILITY_BARS", "DEFAULT_STOP_LOSS_PCT", "DEFAULT_TAKE_PROFIT_PCT",
)


def declared(settings):
    values = {key: getattr(settings, key) for key in FIELDS}
    if any(type(v) not in (float, int, bool) for v in values.values()):
        raise ValueError("Invalid nonnumeric research setting")
    return values


def revision():
    root = Path(__file__).resolve().parents[1]
    if (root / ".git").exists():
        try:
            value = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                            stderr=subprocess.DEVNULL, timeout=3, text=True).strip()
            if re.fullmatch("[0-9a-f]{40,64}", value):
                dirty = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"],
                                       stderr=subprocess.DEVNULL, timeout=3).returncode
                return value, {0: False, 1: True}.get(dirty)
        except (OSError, subprocess.SubprocessError):
            pass
    return None, None  # E.g. installed source distribution, not a checkout.


def fingerprint(symbol, df):
    # Hash schema/labels too: row hashes alone do not include column names or index timezone.
    schema = [symbol, list(map(str, df.columns)), list(map(str, df.dtypes)), str(df.index.dtype)]
    digest = sha256(json.dumps(schema, ensure_ascii=True).encode())
    digest.update(pd.util.hash_pandas_object(df, index=True).to_numpy(dtype="<u8").tobytes())
    record = {"sha256": digest.hexdigest(), "rows": len(df)}
    seed = df.attrs.get("synthetic_seed")
    if type(seed) is int:
        record["synthetic_seed"] = seed
    return record


def manifest(data, settings, custom, seed):
    commit, modified = revision()
    return {
        "schema_version": 1, "code_revision": commit, "tracked_files_modified": modified,
        "strategy_version": "custom-unverified" if custom else "legacy-daily-v1",
        "settings": declared(settings), "strategy_seed": seed if custom else None,
        "python": platform.python_version(),
        "dependencies": {p: version(p) for p in ("pandas", "numpy", "pydantic", "pydantic-settings")},
        "inputs": [fingerprint(symbol, data[symbol]) for symbol in sorted(data)],
    }
