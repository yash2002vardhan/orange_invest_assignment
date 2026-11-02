"""Produce docs/trace_timeout_after_success.jsonl: a full scripted booking whose `book` call times out
after the provider actually booked it. Offline and deterministic."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from errand_agent.agent import Agent
from errand_agent.clock import FakeClock
from errand_agent.llm import ScriptedLLM
from errand_agent.mock_api import MockProviderAPI, Scenario
from errand_agent.store import Store
from errand_agent.tracing import Tracer

OUT = Path(__file__).resolve().parent.parent / "docs" / "trace_timeout_after_success.jsonl"


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    OUT.unlink(missing_ok=True)
    clock = FakeClock()
    with tempfile.TemporaryDirectory() as d:
        api = MockProviderAPI(Path(d) / "provider.db", clock, Scenario(book_outcomes=["timeout_booked"]))
        llm = ScriptedLLM([
            {"intent": "book", "service": "plumber", "area": "indiranagar", "date_from": "2026-11-02",
             "date_to": "2026-11-08", "earliest_hour": 8, "latest_hour": 11, "budget_max": 1500},
            {"intent": "show_reviews", "selection": 2},
            {"intent": "select_option", "selection": 1},
        ], clock=clock)
        tracer = Tracer(OUT, clock, run_id="run-demo-timeout")
        agent = Agent(user_id="u1", store=Store(Path(d) / "agent.db"), api=api, llm=llm, clock=clock, tracer=tracer)
        for msg in ["Find a plumber for my kitchen tap this week, mornings only, under ₹1,500.",
                    "what do the reviews for option 2 say?", "1", "yes", "yes"]:
            print(f"you>   {msg}")
            print("agent> " + agent.handle(msg).replace("\n", "\n       ") + "\n")
            clock.advance(20)
        tracer.event("run_end")
    print(f"trace written to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
