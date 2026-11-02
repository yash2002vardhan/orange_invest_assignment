"""Scripted-user harness and invariant checker shared by the evals."""

from __future__ import annotations

import json
from pathlib import Path

from errand_agent.agent import Agent, is_affirmative
from errand_agent.clock import FakeClock
from errand_agent.config import Limits
from errand_agent.llm import ScriptedLLM
from errand_agent.mock_api import MockProviderAPI, Scenario
from errand_agent.store import Store
from errand_agent.tracing import Tracer

MUTATING = ("book", "cancel", "reschedule")
TEST_SECRET = "sk-test-SECRET1234567890"

BOOK_PLUMBER = {"intent": "book", "service": "plumber", "area": "indiranagar", "date_from": "2026-11-02",
                "date_to": "2026-11-08", "earliest_hour": 8, "latest_hour": 11, "budget_max": 1500}


def select(n: int) -> dict:
    return {"intent": "select_option", "selection": n}


class Harness:
    """One simulated user against one agent. `restart()` rebuilds every object on the same files."""

    def __init__(self, tmp_path: Path, scenario: Scenario | None = None, limits: Limits | None = None,
                 guard=None, user_id: str = "u1"):
        self.dir = tmp_path
        self.clock = FakeClock()
        self.scenario = scenario or Scenario(all_slots_free=True)
        self.limits = limits or Limits()
        self.guard = guard
        self.user_id = user_id
        self.trace_path = tmp_path / "trace.jsonl"
        self.tracers: list[Tracer] = []
        self.llms: list[ScriptedLLM] = []
        self.transcript: list[tuple[str, str]] = []
        self.actual = ""
        self.notes = ""
        self._build()

    def _build(self) -> None:
        self.api = MockProviderAPI(self.dir / "provider.db", self.clock, self.scenario)
        self.store = Store(self.dir / "agent.db")
        self.llm = ScriptedLLM(clock=self.clock)
        self.tracer = Tracer(self.trace_path, self.clock)
        self.tracers.append(self.tracer)
        self.llms.append(self.llm)
        self.agent = Agent(user_id=self.user_id, store=self.store, api=self.api, llm=self.llm, clock=self.clock,
                           tracer=self.tracer, guard=self.guard, limits=self.limits)

    def restart(self, limits: Limits | None = None) -> None:
        assert not self.llm.steps, f"unused scripted steps before restart: {self.llm.steps}"
        self.api.close()
        self.store.close()
        if limits:
            self.limits = limits
        self._build()

    def say(self, text: str, interpretation: dict | None = None) -> str:
        if interpretation is not None:
            self.llm.steps.append(interpretation)
        reply = self.agent.handle(text)
        self.transcript.append((text, reply))
        return reply

    # ---------- observations ----------

    @property
    def events(self) -> list[dict]:
        return [e for t in self.tracers for e in t.events]

    def of_kind(self, *kinds: str) -> list[dict]:
        return [e for e in self.events if e["kind"] in kinds]

    def tool_calls(self, tool: str | None = None) -> list[dict]:
        return [e for e in self.of_kind("tool_call") if tool is None or e["tool"] == tool]

    def user_bookings(self, status: str = "booked") -> list[dict]:
        return [b for b in self.api.list_bookings(self.user_id) if b["status"] == status]

    def turn_events(self, turn_index: int = -1) -> list[dict]:
        """Events of the n-th turn across all sessions."""
        turns, current = [], None
        for e in self.events:
            if e["kind"] == "turn_start":
                current = []
                turns.append(current)
            if current is not None:
                current.append(e)
        return turns[turn_index]

    # ---------- invariants checked after every scenario ----------

    def check_invariants(self) -> None:
        assert not self.llm.steps, f"scripted interpretations were never consumed: {self.llm.steps}"
        granted: dict[str, dict] = {}
        last_user_msg = None
        last_checked = None
        verified_status: dict[str, str] = {}
        for e in self.events:
            k = e["kind"]
            if k == "turn_start":
                last_user_msg = e["user_message"]
            elif k == "approval_granted":
                assert is_affirmative(last_user_msg), f"approval granted on non-affirmative message {last_user_msg!r}"
                granted[e["approval_id"]] = e
            elif k == "approval_checked":
                last_checked = e
            elif k == "tool_call" and e["tool"] in MUTATING:
                assert last_checked is not None, f"{e['tool']} called without an approval check"
                assert last_checked["approval_id"] in granted, f"{e['tool']} with ungranted approval"
                assert last_checked["fingerprint"] == granted[last_checked["approval_id"]]["fingerprint"]
                last_checked = None
            elif k == "tool_call" and e["tool"] == "get_booking_status" and e["ok"]:
                if e["result"].get("status") == "booked":
                    verified_status[e["result"]["booking_id"]] = "booked"
            elif k == "booking_verified":
                assert verified_status.get(e["booking_id"]) == "booked", "success claimed without status check"
            elif k == "turn_end":
                assert e["reply"].strip(), "empty reply"
                if "Booked and verified" in e["reply"]:
                    assert any(x["kind"] == "booking_verified" for x in self.events
                               if x["seq"] <= e["seq"] and x["run_id"] == e["run_id"]), "unverified success claim"

        # no duplicates: one booking per idempotency key, and per (provider, slot)
        rows = self.api._db.execute(
            "SELECT idempotency_key, provider_id, slot FROM bookings WHERE status='booked'").fetchall()
        keys = [r["idempotency_key"] for r in rows if r["idempotency_key"]]
        assert len(keys) == len(set(keys)), f"duplicate bookings for one key: {keys}"
        pairs = [(r["provider_id"], r["slot"]) for r in rows]
        assert len(pairs) == len(set(pairs)), f"double-booked slot: {pairs}"

        text = self.trace_path.read_text() if self.trace_path.exists() else ""
        assert TEST_SECRET not in text, "secret leaked into trace"
        for line in text.splitlines():
            json.loads(line)
