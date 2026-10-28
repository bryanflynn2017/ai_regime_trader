"""Check indexed paths: ignore rules alone do not protect tracked files."""
from pathlib import PurePosixPath
import subprocess

DENY_DIRS = {"state", "logs", "results", "data_cache", ".obsidian", ".aws", ".codex", ".claude", ".agents", "__pycache__", "env", ".streamlit", ".pytest_cache", "build", "dist"}
DENY_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".pem", ".key", ".p12", ".pfx", ".jsonl", ".log", ".exe", ".dll", ".so", ".zip", ".pyc", ".pkl", ".joblib", ".csv", ".parquet", ".lock", ".guard", ".h5", ".pt", ".pth"}


def forbidden(name):
    path = PurePosixPath(name)
    return (
        any(p.lower() in DENY_DIRS or p.lower().startswith((".venv", "venv")) for p in path.parts)
        or (path.name.lower().startswith(".env") and path.name != ".env.example")
        or path.name.lower().startswith(("credentials", "secrets", "client_secret"))
        or path.suffix.lower() in DENY_SUFFIXES
        or any(ext in path.name.lower() for ext in (".sqlite-", ".sqlite3-", ".db-"))
    )


if __name__ == "__main__":
    paths = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    paths = [p for p in paths if p]
    if not paths:
        raise SystemExit("No indexed files: stage the reviewed snapshot first")
    bad = [p for p in paths if forbidden(p)]
    if bad:
        for p in bad:
            print("Forbidden publication path:", p)
        raise SystemExit(1)
    print(f"Publication path policy passed: {len(paths)} indexed files")
