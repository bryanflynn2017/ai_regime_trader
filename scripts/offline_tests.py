"""Credential-free, no-network pytest runner in a disposable source workspace.

No application module is imported until the isolated child installs its network
guard. This is defense in depth for trusted tests, not a sandbox for hostile code.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (
    "alerts", "backtest", "broker", "config", "dashboard", "data",
    "data_providers", "regime", "risk", "strategy", "tests", "trades", "utils", "watchlists",
)


def main():
    args = sys.argv[1:] or ["tests"]
    # Only relative test selections/options; prevent a caller selecting old checkout tests.
    for arg in args:
        if Path(arg.split("::", 1)[0]).is_absolute() or ".." in Path(arg.split("::", 1)[0]).parts:
            raise SystemExit("Use relative test paths inside the publication snapshot")
    with tempfile.TemporaryDirectory(prefix="ai-regime-offline-") as temp:
        work = Path(temp)
        for name in SOURCE_DIRS:
            shutil.copytree(ROOT / name, work / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("main.py", "pyproject.toml"):
            shutil.copy2(ROOT / name, work / name)
        # Keep required OS variables, not arbitrary inherited broker/cloud tokens.
        keep = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "LANG"}
        env = {k: v for k, v in os.environ.items() if k.upper() in keep}
        env.update({
            "HOME": str(work), "USERPROFILE": str(work),
            "ALPACA_API_KEY": "", "ALPACA_SECRET_KEY": "", "DISCORD_WEBHOOK_URL": "",
            "PAPER_TRADING": "true", "ALPACA_PAPER": "true", "LIVE_TRADING_ALLOWED": "false",
            "EXECUTION_MODE": "shadow", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONIOENCODING": "utf-8",
        })
        child = r'''
import sys, socket, http.client, urllib.request
sys.path.insert(0, ".")
def denied(*args, **kwargs):
    raise RuntimeError("OFFLINE_TEST_NETWORK_BLOCKED")
class OfflineSocket(socket.socket):
    connect = denied
    connect_ex = denied
    sendto = denied
socket.socket = OfflineSocket
socket.create_connection = denied
socket.getaddrinfo = denied
http.client.HTTPConnection.connect = denied
urllib.request.urlopen = denied
import requests
requests.sessions.Session.request = denied
# yfinance can use native libcurl, bypassing Python's socket implementation.
try:
    import curl_cffi.requests
except ImportError:
    pass
else:
    curl_cffi.requests.Session.request = denied
    curl_cffi.requests.AsyncSession.request = denied
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
'''
        result = subprocess.run([sys.executable, "-B", "-c", child, *args, "-q", "--tb=short"], cwd=work, env=env)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
