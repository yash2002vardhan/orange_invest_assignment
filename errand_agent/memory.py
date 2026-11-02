"""Long-term user preferences with provenance.

Rules (enforced here, not in a prompt):
- every fact is `stated` (user said it) or `inferred` (agent derived it);
- inferred facts are not usable until the user confirms them;
- forgetting leaves a tombstone, and inference can never revive a tombstoned key;
  only a fresh user statement can.
"""

from __future__ import annotations

from .store import MemoryFact, Store

STABLE_KEYS = ("area", "earliest_hour")
TECH_PREFIX = "preferred_technician:"


def is_valid_key(key: str) -> bool:
    return key in STABLE_KEYS or key.startswith(TECH_PREFIX)


class Memory:
    def __init__(self, store: Store, user_id: str, clock, tracer):
        self.store = store
        self.user_id = user_id
        self.clock = clock
        self.tracer = tracer

    def _now(self) -> str:
        return self.clock.now().isoformat()

    def usable(self, key: str) -> str | None:
        fact = self.store.get_fact(self.user_id, key)
        return fact.value if fact and fact.usable else None

    def pending_confirmation(self, key: str) -> MemoryFact | None:
        fact = self.store.get_fact(self.user_id, key)
        if fact and fact.status == "active" and fact.source == "inferred" and not fact.confirmed:
            return fact
        return None

    def state(self, value: str, key: str) -> None:
        if not is_valid_key(key):
            raise ValueError(f"not a memory key: {key}")
        self.store.put_fact(self.user_id, MemoryFact(key, str(value), "stated", True, "active", self._now()))
        self.tracer.event("memory_write", key=key, value=value, source="stated")

    def infer(self, key: str, value: str) -> bool:
        existing = self.store.get_fact(self.user_id, key)
        if existing and (existing.status == "forgotten" or existing.source == "stated"):
            self.tracer.event("memory_infer_skipped", key=key, reason=existing.status if existing.status == "forgotten" else "stated_exists")
            return False
        if existing and existing.value == str(value):
            return False
        self.store.put_fact(self.user_id, MemoryFact(key, str(value), "inferred", False, "active", self._now()))
        self.tracer.event("memory_write", key=key, value=value, source="inferred")
        return True

    def confirm(self, key: str) -> None:
        fact = self.store.get_fact(self.user_id, key)
        if fact and fact.status == "active":
            fact.confirmed = True
            fact.updated_at = self._now()
            self.store.put_fact(self.user_id, fact)
            self.tracer.event("memory_confirmed", key=key)

    def forget(self, key: str) -> list[str]:
        """`key` may be exact, a prefix such as 'preferred_technician', or 'all'."""
        forgotten = []
        for fact in self.store.all_facts(self.user_id):
            if fact.status == "forgotten":
                continue
            if key == "all" or fact.key == key or fact.key.startswith(key.rstrip(":") + ":"):
                fact.status = "forgotten"
                fact.value = ""
                fact.updated_at = self._now()
                self.store.put_fact(self.user_id, fact)
                forgotten.append(fact.key)
        self.tracer.event("memory_forget", requested=key, forgotten=forgotten)
        return forgotten

    def view(self) -> list[MemoryFact]:
        return [f for f in self.store.all_facts(self.user_id) if f.status == "active"]
