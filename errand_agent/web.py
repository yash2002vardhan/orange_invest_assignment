"""Local web UI server: chat on the left, live trace on the right (frontend/ folder).

Standard library only. All agent and SQLite work runs on one worker thread
(SQLite connections are thread-bound and the agent is not concurrent), while
the trace endpoint reads the in-memory event list so it can update live
during a slow model call.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from .agent import Agent
from .clock import Clock
from .config import load_dotenv
from .guard import default_guard
from .llm import default_llm
from .mock_api import MockProviderAPI, Scenario
from .store import Store
from .tracing import Tracer

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SCENARIOS = ["default", "timeout_after_success", "slot_taken", "price_change", "happy"]


class Session:
    """Owns the agent. Every method must run on the single worker thread."""

    def __init__(self, data_dir: Path, user_id: str, llm_factory: Callable):
        self.data_dir = data_dir
        self.user_id = user_id
        self.llm_factory = llm_factory
        self.clock = Clock()
        self.scenario = "default"
        self.past_events: list[dict] = []  # events of earlier runs (before a restart)
        self.transcript: list[dict] = []  # full chat, so a page reload can restore it
        self.tracer: Tracer | None = None
        self.agent: Agent | None = None
        self.store: Store | None = None
        self.api: MockProviderAPI | None = None

    def build(self) -> None:
        if self.tracer is not None:
            self.tracer.event("run_end", reason="agent restarted")
            self.past_events.extend(self.tracer.events)
        for obj in (self.api, self.store):
            if obj is not None:
                obj.close()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tracer = Tracer(None, self.clock)
        self.tracer.path = ROOT / "traces" / f"{self.tracer.run_id}.jsonl"
        self.tracer.path.parent.mkdir(exist_ok=True)
        self.store = Store(self.data_dir / "agent.db")
        self.api = MockProviderAPI(self.data_dir / "provider.db", self.clock, Scenario.named(self.scenario))
        self.agent = Agent(user_id=self.user_id, store=self.store, api=self.api, llm=self.llm_factory(self.clock),
                           clock=self.clock, tracer=self.tracer, guard=default_guard())

    def reset(self, scenario: str) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario}")
        for obj in (self.api, self.store):
            if obj is not None:
                obj.close()
        self.api = self.store = None
        shutil.rmtree(self.data_dir, ignore_errors=True)
        self.scenario = scenario
        self.past_events = []
        self.transcript = []
        self.tracer = None
        self.build()

    def chat(self, message: str) -> dict:
        self.transcript.append({"role": "user", "text": message})
        reply = self.agent.handle(message)
        self.transcript.append({"role": "agent", "text": reply})
        return {"reply": reply, "phase": self.agent.state.phase}

    def events(self) -> list[dict]:
        return self.past_events + list(self.tracer.events)

    def status(self) -> dict:
        return {"phase": self.agent.state.phase, "scenario": self.scenario, "run_id": self.tracer.run_id,
                "trace_file": str(self.tracer.path.relative_to(ROOT)), "guard": getattr(self.agent.guard, "name", "?"),
                "user_id": self.user_id}

    def bookings(self) -> list[dict]:
        return self.api.list_bookings(self.user_id)

    def memory(self) -> list[dict]:
        return [{"key": f.key, "value": f.value, "source": f.source, "confirmed": f.confirmed, "status": f.status,
                 "updated_at": f.updated_at} for f in self.store.all_facts(self.user_id)]


def make_handler(session: Session, worker: ThreadPoolExecutor):
    def run(fn, *args):
        return worker.submit(fn, *args).result()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(FRONTEND), **kw)

        def log_message(self, *args):  # keep the terminal quiet for recording
            pass

        def _json(self, payload, code: int = 200) -> None:
            body = json.dumps(payload, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            path, _, query = self.path.partition("?")
            if path == "/api/events":
                after = int(dict(p.split("=") for p in query.split("&") if "=" in p).get("after", 0))
                events = session.events()  # list read only; safe off the worker thread
                return self._json({"events": events[after:], "total": len(events)})
            if path == "/api/status":
                return self._json(run(session.status) | {"scenarios": SCENARIOS})
            if path == "/api/transcript":
                return self._json(run(lambda: list(session.transcript)))
            if path == "/api/bookings":
                return self._json(run(session.bookings))
            if path == "/api/memory":
                return self._json(run(session.memory))
            return super().do_GET()

        def do_POST(self):
            try:
                body = self._body()
                if self.path == "/api/chat":
                    msg = str(body.get("message", "")).strip()
                    if not msg:
                        return self._json({"error": "empty message"}, 400)
                    return self._json(run(session.chat, msg))
                if self.path == "/api/reset":
                    run(session.reset, body.get("scenario", "default"))
                    return self._json(run(session.status))
                if self.path == "/api/restart":
                    run(session.build)
                    return self._json(run(session.status))
                return self._json({"error": "not found"}, 404)
            except Exception as e:  # surface server faults to the UI instead of hanging it
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def serve(port: int, data_dir: Path, user_id: str, llm_factory: Callable, scenario: str = "default",
          reset: bool = False) -> ThreadingHTTPServer:
    worker = ThreadPoolExecutor(max_workers=1)
    session = Session(data_dir, user_id, llm_factory)
    session.scenario = scenario
    if reset:
        worker.submit(session.reset, scenario).result()
    else:
        worker.submit(session.build).result()
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(session, worker))


def main() -> None:
    p = argparse.ArgumentParser(description="Web UI for the errand agent (chat + live trace).")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--user", default="u1")
    p.add_argument("--scenario", default="default", choices=SCENARIOS)
    p.add_argument("--data-dir", default=".errand-web")
    p.add_argument("--reset", action="store_true", help="start from a clean world")
    args = p.parse_args()
    load_dotenv()
    default_llm()  # fail fast with a clear message if OPENAI_API_KEY is missing
    server = serve(args.port, ROOT / args.data_dir, args.user, default_llm, args.scenario, args.reset)
    print(f"Errand agent UI: http://127.0.0.1:{args.port}   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
