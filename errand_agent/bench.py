"""Model-routing benchmark (live; needs OPENAI_API_KEY).

Runs a labelled set of user messages through the cheap model, the strong model,
and the router, and reports field accuracy, latency and cost per strategy to
reports/routing_report.md. The offline evals never call this.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from .config import load_dotenv
from .llm import OpenAILLM, route

CTX = {"today": "2026-11-02", "weekday": "Monday", "request": {}, "options": [], "last_question": None}
OPTS = [{"n": 1, "label": "Anita Rao Plumber (PLU-IND-3), Mon 02 Nov 10:00 AM, ₹600"},
        {"n": 2, "label": "Kavya Reddy Plumber (PLU-IND-1), Mon 02 Nov 10:00 AM, ₹850"}]

# (message, phase, expected fields). Only the listed fields are scored.
CASES: list[tuple[str, str, dict]] = [
    ("Find a plumber for my kitchen tap this week, mornings only, under ₹1,500.", "idle",
     {"intent": "book", "service": "plumber", "date_from": "2026-11-02", "date_to": "2026-11-08", "budget_max": 1500}),
    ("Move my AC servicing to Saturday.", "idle",
     {"intent": "reschedule", "service": "ac_servicing", "date_from": "2026-11-07"}),
    ("Book the same electrician as last time, but not before 10 a.m.", "idle",
     {"intent": "book", "service": "electrician", "same_provider_as_last": True, "earliest_hour": 10}),
    ("Cancel my AC servicing please", "idle", {"intent": "cancel", "service": "ac_servicing"}),
    ("I need a carpenter in Whitefield tomorrow", "idle",
     {"intent": "book", "service": "carpenter", "area": "whitefield", "date_from": "2026-11-03"}),
    ("cleaning in HSR on friday, budget 2000", "idle",
     {"intent": "book", "service": "cleaning", "area": "hsr_layout", "date_from": "2026-11-06", "budget_max": 2000}),
    ("2", "choosing", {"intent": "select_option", "selection": 2}),
    ("the first one", "choosing", {"intent": "select_option", "selection": 1}),
    ("Koramangala", "collecting", {"area": "koramangala"}),
    ("tomorrow", "collecting", {"date_from": "2026-11-03"}),
    ("no preference", "collecting", {"no_preference": True}),
    ("actually make it under 700", "choosing", {"budget_max": 700}),
    ("actually can we do Wednesday instead", "choosing", {"date_from": "2026-11-04"}),
    ("what do the reviews for option 2 say?", "choosing", {"intent": "show_reviews", "selection": 2}),
    ("what do you remember about me?", "idle", {"intent": "memory_view"}),
    ("forget my area", "idle", {"intent": "memory_forget", "memory_key": "area"}),
    ("I moved to Indiranagar, update that", "idle",
     {"intent": "memory_correct", "memory_key": "area", "memory_value": "indiranagar"}),
    ("I never want anything before 9am", "idle", {"remember_earliest_hour": 9}),
    ("electrician, after 5pm any day this week", "idle",
     {"intent": "book", "service": "electrician", "earliest_hour": 17}),
    ("none of these work", "choosing", {"intent": "decline"}),
]


def _score(interp, expected: dict) -> float:
    got = asdict(interp)
    return sum(got.get(k) == v for k, v in expected.items()) / len(expected)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="reports/routing_report.md")
    args = p.parse_args()
    load_dotenv()
    llm = OpenAILLM()
    rows = []
    per_tier: dict[str, list[dict]] = {"cheap": [], "strong": []}
    for msg, phase, expected in CASES:
        ctx = dict(CTX, phase=phase, options=OPTS if phase == "choosing" else [])
        routed_tier, _ = route(msg, phase)
        rec = {"message": msg, "phase": phase, "routed": routed_tier}
        for tier in ("cheap", "strong"):
            r = llm.interpret(msg, ctx, tier)
            m = {"score": _score(r.interpretation, expected), "latency_ms": r.latency_ms, "cost": r.cost_usd or 0.0,
                 "model": r.model}
            per_tier[tier].append(m)
            rec[tier] = m
        rows.append(rec)

    def summary(ms: list[dict]) -> tuple[float, float, float]:
        return (statistics.mean(m["score"] for m in ms), statistics.median(m["latency_ms"] for m in ms),
                sum(m["cost"] for m in ms))

    routed = [r[r["routed"]] for r in rows]
    lines = ["# Model routing benchmark", "",
             f"{len(CASES)} labelled messages; field accuracy = share of expected fields matched.", "",
             "| Strategy | Accuracy | Median latency (ms) | Total cost (USD) |", "|---|---|---|---|"]
    for name, ms in (("cheap only", per_tier["cheap"]), ("strong only", per_tier["strong"]), ("routed", routed)):
        acc, lat, cost = summary(ms)
        lines.append(f"| {name} | {acc:.1%} | {lat:.0f} | {cost:.5f} |")
    lines += ["", "| Message | Routed | Cheap score | Strong score |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['message']} | {r['routed']} | {r['cheap']['score']:.2f} | {r['strong']['score']:.2f} |")
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    (out.with_suffix(".json")).write_text(json.dumps(rows, indent=2))
    print("\n".join(lines[:9]))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
