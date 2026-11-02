"""Readable, optionally live view of a JSONL trace (for showing chat and trace side by side)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

COLORS = {"tool_call": "36", "model_call": "35", "state_transition": "33", "approval_granted": "32",
          "approval_proposed": "32", "approval_voided": "31", "approval_rejected": "31", "booking_verified": "32;1",
          "error": "31;1", "limit_exceeded": "31;1", "guard_check": "34", "turn_start": "1", "turn_end": "2"}


def fmt(e: dict) -> str:
    k = e["kind"]
    t = e["ts"][11:19]
    if k == "tool_call":
        status = "ok" if e["ok"] else f"ERR {e.get('error')}"
        body = f"{e['tool']}({json.dumps(e['args'])[1:-1]}) -> {status} [{e['latency_ms']}ms]"
        if e["ok"] and e["tool"] in ("get_booking_status", "hold_slot", "book", "cancel", "reschedule"):
            body += f" {json.dumps(e['result'])}"
    elif k == "model_call":
        body = (f"{e['model']} ({e['tier']}: {e['route_reason']}) {e['latency_ms']}ms "
                f"tok={e['input_tokens']}/{e['output_tokens']} ${e['cost_usd']} -> {json.dumps(e['interpretation'])}")
    elif k == "state_transition":
        body = f"{e['from_phase']} -> {e['to_phase']} ({e['reason']})"
    elif k == "turn_start":
        body = f"USER: {e['user_message']}"
    elif k == "turn_end":
        body = f"AGENT replied ({e['model_calls']} model / {e['tool_calls']} tool calls)"
    else:
        body = json.dumps({x: y for x, y in e.items() if x not in ("run_id", "seq", "ts", "kind")})
    color = COLORS.get(k, "0")
    return f"\033[2m{t}\033[0m \033[{color}m{k:<22}\033[0m {body}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?")
    p.add_argument("--latest", action="store_true", help="newest file in ./traces")
    p.add_argument("-f", "--follow", action="store_true")
    args = p.parse_args()
    if args.latest:
        files = sorted(Path("traces").glob("*.jsonl"), key=lambda f: f.stat().st_mtime)
        if not files:
            sys.exit("no traces yet")
        path = files[-1]
    elif args.path:
        path = Path(args.path)
    else:
        sys.exit("give a trace path or --latest")
    print(f"== {path} ==")
    with path.open() as f:
        while True:
            line = f.readline()
            if line:
                print(fmt(json.loads(line)), flush=True)
            elif args.follow:
                time.sleep(0.2)
            else:
                break


if __name__ == "__main__":
    main()
