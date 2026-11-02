"""Agent-side durable state (SQLite): workflow sessions, approvals, operations, memory.

Everything that must survive a restart lives here. Writes commit immediately so
a crash between two statements never loses an approval or an operation record.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

APPROVAL_PROPOSED = "proposed"
APPROVAL_GRANTED = "granted"
APPROVAL_CONSUMED = "consumed"
APPROVAL_VOID = "void"

OP_PENDING = "pending"
OP_DONE = "done"
OP_FAILED = "failed"
OP_UNKNOWN = "unknown"  # retries exhausted without an authoritative answer


def action_fingerprint(action: str, provider_id: str, slot: str, price: int, booking_id: str | None) -> str:
    raw = json.dumps([action, provider_id, slot, int(price), booking_id or ""])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class Approval:
    id: str
    user_id: str
    action: str  # book | cancel | reschedule
    provider_id: str
    slot: str
    price: int
    booking_id: str | None
    hold_id: str | None
    hold_expires_at: str | None
    fingerprint: str
    status: str

    def matches(self, action: str, provider_id: str, slot: str, price: int, booking_id: str | None) -> bool:
        return self.fingerprint == action_fingerprint(action, provider_id, slot, price, booking_id)


@dataclass
class Operation:
    idempotency_key: str
    approval_id: str
    user_id: str
    action: str
    status: str
    result: dict


@dataclass
class MemoryFact:
    key: str
    value: str
    source: str  # stated | inferred
    confirmed: bool
    status: str  # active | forgotten
    updated_at: str

    @property
    def usable(self) -> bool:
        return self.status == "active" and (self.source == "stated" or self.confirmed)


class Store:
    def __init__(self, db_path: str | Path):
        self._db = sqlite3.connect(str(db_path))
        self._db.row_factory = sqlite3.Row
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (user_id TEXT PRIMARY KEY, state TEXT, updated_at TEXT);
            CREATE TABLE IF NOT EXISTS approvals (
                id TEXT PRIMARY KEY, user_id TEXT, action TEXT, provider_id TEXT, slot TEXT, price INTEGER,
                booking_id TEXT, hold_id TEXT, hold_expires_at TEXT, fingerprint TEXT, status TEXT);
            CREATE TABLE IF NOT EXISTS operations (
                idempotency_key TEXT PRIMARY KEY, approval_id TEXT, user_id TEXT, action TEXT,
                status TEXT, result TEXT);
            CREATE TABLE IF NOT EXISTS memory (
                user_id TEXT, key TEXT, value TEXT, source TEXT, confirmed INTEGER, status TEXT,
                updated_at TEXT, PRIMARY KEY (user_id, key));
            """
        )
        self._db.commit()

    # ---------- sessions ----------

    def load_session(self, user_id: str) -> dict | None:
        row = self._db.execute("SELECT state FROM sessions WHERE user_id=?", (user_id,)).fetchone()
        return json.loads(row["state"]) if row else None

    def save_session(self, user_id: str, state: dict, now: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?)", (user_id, json.dumps(state), now))
        self._db.commit()

    # ---------- approvals ----------

    def create_approval(self, user_id: str, action: str, provider_id: str, slot: str, price: int,
                        booking_id: str | None = None, hold_id: str | None = None,
                        hold_expires_at: str | None = None) -> Approval:
        appr = Approval(
            id=f"A-{uuid.uuid4().hex[:8]}", user_id=user_id, action=action, provider_id=provider_id, slot=slot,
            price=int(price), booking_id=booking_id, hold_id=hold_id, hold_expires_at=hold_expires_at,
            fingerprint=action_fingerprint(action, provider_id, slot, price, booking_id), status=APPROVAL_PROPOSED,
        )
        self._db.execute(
            "INSERT INTO approvals VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            tuple(asdict(appr).values()),
        )
        self._db.commit()
        return appr

    def get_approval(self, approval_id: str) -> Approval | None:
        row = self._db.execute("SELECT * FROM approvals WHERE id=?", (approval_id,)).fetchone()
        return Approval(**dict(row)) if row else None

    def grant_approval(self, approval_id: str) -> bool:
        """Atomic proposed -> granted. False if someone else already granted, voided or consumed it."""
        cur = self._db.execute("UPDATE approvals SET status=? WHERE id=? AND status=?",
                               (APPROVAL_GRANTED, approval_id, APPROVAL_PROPOSED))
        self._db.commit()
        return cur.rowcount == 1

    def set_approval_status(self, approval_id: str, status: str) -> None:
        self._db.execute("UPDATE approvals SET status=? WHERE id=?", (status, approval_id))
        self._db.commit()

    def update_hold(self, approval_id: str, hold_id: str, expires_at: str) -> None:
        """Holds can be renewed without re-approval: the fingerprint excludes the hold."""
        self._db.execute(
            "UPDATE approvals SET hold_id=?, hold_expires_at=? WHERE id=?", (hold_id, expires_at, approval_id)
        )
        self._db.commit()

    def void_open_approvals(self, user_id: str) -> list[str]:
        rows = self._db.execute(
            "SELECT id FROM approvals WHERE user_id=? AND status IN (?,?)",
            (user_id, APPROVAL_PROPOSED, APPROVAL_GRANTED),
        ).fetchall()
        ids = [r["id"] for r in rows]
        for approval_id in ids:
            self.set_approval_status(approval_id, APPROVAL_VOID)
        return ids

    # ---------- operations ----------

    def get_operation(self, key: str) -> Operation | None:
        row = self._db.execute("SELECT * FROM operations WHERE idempotency_key=?", (key,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["result"] = json.loads(d["result"] or "{}")
        return Operation(**d)

    def put_operation(self, key: str, approval_id: str, user_id: str, action: str, status: str,
                      result: dict | None = None) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO operations VALUES (?,?,?,?,?,?)",
            (key, approval_id, user_id, action, status, json.dumps(result or {})),
        )
        self._db.commit()

    def unresolved_operations(self, user_id: str) -> list[Operation]:
        rows = self._db.execute(
            "SELECT idempotency_key FROM operations WHERE user_id=? AND status IN (?,?)",
            (user_id, OP_PENDING, OP_UNKNOWN),
        ).fetchall()
        return [self.get_operation(r["idempotency_key"]) for r in rows]

    # ---------- memory ----------

    def get_fact(self, user_id: str, key: str) -> MemoryFact | None:
        row = self._db.execute("SELECT * FROM memory WHERE user_id=? AND key=?", (user_id, key)).fetchone()
        if not row:
            return None
        d = dict(row)
        d.pop("user_id")
        d["confirmed"] = bool(d["confirmed"])
        return MemoryFact(**d)

    def all_facts(self, user_id: str) -> list[MemoryFact]:
        rows = self._db.execute("SELECT key FROM memory WHERE user_id=? ORDER BY key", (user_id,)).fetchall()
        return [self.get_fact(user_id, r["key"]) for r in rows]

    def put_fact(self, user_id: str, fact: MemoryFact) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO memory VALUES (?,?,?,?,?,?,?)",
            (user_id, fact.key, fact.value, fact.source, int(fact.confirmed), fact.status, fact.updated_at),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()
