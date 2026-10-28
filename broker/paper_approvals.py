"""Local, one-use entry consent in the existing paper SQLite ledger.

The operator connection intentionally does NOT take the session ownership lock.
SQLite transactions serialize decisions against consumption. No broker imports
or network calls occur in this module. Local OS access is the trust boundary.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from broker.base import BrokerError

OPEN = {"PENDING", "APPROVED"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def initialize(db):
    db.execute("""CREATE TABLE IF NOT EXISTS proposals (
        cid TEXT PRIMARY KEY, payload TEXT NOT NULL, digest TEXT NOT NULL,
        expires TEXT NOT NULL, status TEXT NOT NULL, decided_at TEXT,
        reason TEXT NOT NULL DEFAULT '')""")
    db.commit()


@contextmanager
def operator_store(root="state/paper"):
    path = Path(root) / "ledger.sqlite3"
    if not path.is_file():
        raise BrokerError("No paper ledger; start an explicitly activated paper session first")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, timeout=5)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA synchronous=FULL")
    try:
        yield Approvals(db)
    finally:
        db.close()


class Approvals:
    def __init__(self, db):
        self.db = db

    def _audit(self, now, event, cid, **fields):
        self.db.execute("INSERT INTO audit(at,event,details) VALUES (?,?,?)",
                        (now.isoformat(), event, canonical(dict(cid=cid, **fields))))

    def rows(self):
        return [dict(r, payload=json.loads(r["payload"])) for r in
                self.db.execute("SELECT * FROM proposals ORDER BY rowid")]

    def get(self, cid):
        return next((r for r in self.rows() if r["cid"] == cid), None)

    def create(self, cid, payload, now):
        with self.db:
            result = self.db.execute("INSERT OR IGNORE INTO proposals(cid,payload,digest,expires,status) VALUES (?,?,?,?,?)",
                (cid, canonical(payload), digest(payload), payload["order"]["valid_until"], "PENDING"))
            if not result.rowcount:
                return None  # Never refresh/replace an expired/rejected/consumed proposal.
            self._audit(now, "ENTRY_PROPOSED", cid, digest=digest(payload))
        return cid

    def expire(self, now):
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            for row in self.rows():
                if row["status"] in OPEN and now >= datetime.fromisoformat(row["expires"]):
                    self.db.execute("UPDATE proposals SET status='EXPIRED',reason='DEADLINE' WHERE cid=? AND status IN ('PENDING','APPROVED')", (row["cid"],))
                    self._audit(now, "PROPOSAL_EXPIRED", row["cid"])

    def invalidate(self, now, reason, cid=None):
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            for row in self.rows():
                if row["status"] in OPEN and (cid is None or row["cid"] == cid):
                    self.db.execute("UPDATE proposals SET status='INVALIDATED',reason=? WHERE cid=?", (reason, row["cid"]))
                    self._audit(now, "PROPOSAL_INVALIDATED", row["cid"], reason=reason)

    def decide(self, cid, expected_digest, *, approve, now=None):
        self.expire(now or datetime.now(timezone.utc))
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            # A contended SQLite lock can consume the remaining consent window.
            # CLI wall time must be sampled AFTER acquiring it, not before.
            decided_now = now or datetime.now(timezone.utc)
            row = self.get(cid)
            if (not row or row["status"] != "PENDING" or
                    row["digest"] != expected_digest or digest(row["payload"]) != expected_digest
                    or decided_now >= datetime.fromisoformat(row["expires"])):
                raise BrokerError("Proposal unavailable, changed, expired, or already decided")
            meta = {r["key"]: json.loads(r["value"]) for r in self.db.execute("SELECT * FROM meta")}
            if approve and (meta.get("status") != "RUNNING" or meta.get("partial") is not False
                            or meta.get("session_id") != row["payload"]["session_id"]
                            or meta.get("account_fingerprint") != row["payload"]["account_fingerprint"]):
                raise BrokerError("Proposal session/account is no longer eligible")
            status = "APPROVED" if approve else "REJECTED"
            self.db.execute("UPDATE proposals SET status=?,decided_at=? WHERE cid=?", (status, decided_now.isoformat(), cid))
            self._audit(decided_now, "HUMAN_" + status, cid, digest=expected_digest, source="LOCAL_OPERATOR_CLI")

    def consume(self, cid, payload, now):
        """Consent consumption + intent are a single FULL-synchronous commit."""
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.get(cid)
            if (not row or row["status"] != "APPROVED" or row["digest"] != digest(payload)
                    or row["payload"] != payload or now >= datetime.fromisoformat(row["expires"])):
                raise BrokerError("Exact, unexpired human approval required")
            self.db.execute("INSERT INTO orders(cid,payload,status) VALUES (?,?,'UNKNOWN')", (cid, canonical(payload)))
            self.db.execute("UPDATE proposals SET status='CONSUMED' WHERE cid=?", (cid,))
            self._audit(now, "APPROVAL_CONSUMED", cid, digest=row["digest"])
            self._audit(now, "SUBMISSION_INTENT", cid, side="buy")
