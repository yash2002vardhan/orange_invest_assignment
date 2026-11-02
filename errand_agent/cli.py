"""Interactive chat. Run `errand-chat`, and `errand-trace --latest -f` in a second pane for the live trace."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .agent import Agent
from .clock import Clock
from .config import load_dotenv
from .guard import default_guard
from .llm import default_llm
from .mock_api import MockProviderAPI, Scenario
from .store import Store
from .tracing import Tracer


def main() -> None:
    p = argparse.ArgumentParser(description="Chat with the errand agent.")
    p.add_argument("--user", default="u1")
    p.add_argument("--scenario", default="default",
                   help="default | timeout_after_success | slot_taken | price_change | happy")
    p.add_argument("--data-dir", default=".errand", help="where agent and mock-provider state live")
    p.add_argument("--reset", action="store_true", help="wipe saved state (memory, bookings, sessions) first")
    args = p.parse_args()
    load_dotenv()

    data = Path(args.data_dir)
    if args.reset and data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True, exist_ok=True)

    clock = Clock()
    tracer = Tracer(None, clock)
    tracer.path = Path("traces") / f"{tracer.run_id}.jsonl"
    tracer.path.parent.mkdir(exist_ok=True)
    guard = default_guard()
    agent = Agent(user_id=args.user, store=Store(data / "agent.db"),
                  api=MockProviderAPI(data / "provider.db", clock, Scenario.named(args.scenario)),
                  llm=default_llm(clock), clock=clock, tracer=tracer, guard=guard)
    print(f"Errand agent ready (user={args.user}, scenario={args.scenario}, guard={guard.name}).")
    print(f"Trace: {tracer.path}   (live view: uv run errand-trace --latest -f)")
    print("Type 'quit' to exit.\n")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg.lower() in ("quit", "exit"):
            break
        if msg:
            print("agent> " + agent.handle(msg).replace("\n", "\n       ") + "\n")
    tracer.event("run_end")


if __name__ == "__main__":
    main()
