"""The errand agent: an explicit, persisted workflow state machine.

The model only interprets the user's words (llm.py). This module decides what
to ask, which tools to call and when an action may run:

- Workflow state is persisted after every turn; corrections merge into it.
- Approval is granted only by `_on_approve`, reached only when the raw user text
  is a plain affirmative while a proposal is pending. The gateway re-checks the
  stored record before any mutating call.
- Every mutating call uses an idempotency key derived from the approval id and
  recorded before the call, so retries, repeated "yes" and restarts converge on
  one booking. Success is reported only after `get_booking_status` (or
  `list_bookings` for cancel/reschedule) confirms it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .budget import Budget, LimitExceeded
from .config import Limits
from .gateway import ApprovalError, ToolGateway
from .guard import HeuristicGuard
from .llm import LLM, Interpretation, ScriptExhausted, route
from .memory import TECH_PREFIX, Memory
from .mock_api import MockProviderAPI, ToolError, parse_slot
from .store import (APPROVAL_CONSUMED, APPROVAL_GRANTED, APPROVAL_PROPOSED, APPROVAL_VOID, OP_DONE, OP_FAILED,
                    OP_PENDING, OP_UNKNOWN, Approval, Store)

IDLE, COLLECTING, CONFIRM_MEMORY, CHOOSING, AWAITING_APPROVAL, DONE = (
    "idle", "collecting", "confirm_memory", "choosing", "awaiting_approval", "done")

SEARCH_FIELDS = ("service", "area", "date_from", "date_to", "earliest_hour", "latest_hour", "budget_max", "provider_id")
REQUEST_FIELDS = ("service", "area", "date_from", "date_to", "earliest_hour", "latest_hour", "budget_max")
MUTATING = ("book", "cancel", "reschedule")

_AFFIRM = re.compile(
    r"^(yes|y|yep|yeah|yup|sure|ok|okay|confirm|confirmed|go ahead|book it|do it|please do|yes please|"
    r"yes,? (please )?(book|confirm|go ahead|do it|proceed)( it)?)[\s.!]*$", re.I)
_NEGATE = re.compile(r"^(no|n|nope|nah|no thanks|not now|don'?t|stop|cancel that)[\s.!]*$", re.I)

ASK_TEXT = {
    "service": "what service you need (plumber, electrician, AC servicing, cleaning or carpenter)",
    "area": "which area you're in (Indiranagar, Koramangala, HSR Layout or Whitefield)",
    "date_from": "which day or date range works",
    "prefs": "any time-of-day or budget limits (or say 'no preference')",
}


def is_affirmative(text: str) -> bool:
    return bool(_AFFIRM.match(text.strip()))


def is_negative(text: str) -> bool:
    return bool(_NEGATE.match(text.strip()))


def human_slot(slot: str) -> str:
    return parse_slot(slot).strftime("%a %d %b, %-I:%M %p")


def human_service(service: str | None) -> str:
    return (service or "service").replace("_", " ")


class SimulatedCrash(BaseException):
    """Raised by test fault hooks to mimic the process dying (escapes all handlers)."""


@dataclass
class Request:
    intent: str | None = None
    service: str | None = None
    area: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    earliest_hour: int | None = None
    latest_hour: int | None = None
    budget_max: int | None = None
    prefs_answered: bool = False
    provider_id: str | None = None
    booking: dict | None = None  # target booking for cancel / reschedule
    sources: dict = field(default_factory=dict)  # field -> user | memory | history


@dataclass
class WorkflowState:
    phase: str = IDLE
    request: Request = field(default_factory=Request)
    options: list[dict] = field(default_factory=list)
    approval_id: str | None = None
    memory_question: str | None = None
    asked: list[str] = field(default_factory=list)
    holds: dict = field(default_factory=dict)  # "provider|slot" -> our live hold (there is no release tool)
    last_booking_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowState":
        d = dict(d)
        d["request"] = Request(**d.get("request", {}))
        return cls(**d)


class Agent:
    def __init__(self, *, user_id: str, store: Store, api: MockProviderAPI, llm: LLM, clock, tracer,
                 guard=None, limits: Limits | None = None):
        self.user_id = user_id
        self.store = store
        self.llm = llm
        self.clock = clock
        self.tracer = tracer
        self.guard = guard or HeuristicGuard()
        self.limits = limits or Limits()
        self.gateway = ToolGateway(api, store, tracer, clock, user_id)
        self.memory = Memory(store, user_id, clock, tracer)
        saved = store.load_session(user_id)
        self.state = WorkflowState.from_dict(saved) if saved else WorkflowState()
        self.turns = 0
        self.budget: Budget | None = None
        self.tracer.event("run_start", user_id=user_id, resumed_phase=self.state.phase, guard=getattr(self.guard, "name", "?"))

    # ======================= turn loop =======================

    @property
    def req(self) -> Request:
        return self.state.request

    def handle(self, message: str) -> str:
        self.turns += 1
        self.budget = Budget(self.limits, self.clock, self.tracer)
        self.gateway.budget = self.budget
        self._turn_start_seq = len(self.tracer.events)
        self.tracer.event("turn_start", turn=self.turns, phase=self.state.phase, user_message=message)
        try:
            if self.turns > self.limits.max_turns_per_session:
                self.tracer.event("limit_exceeded", limit="session_turns", cap=self.limits.max_turns_per_session)
                raise LimitExceeded("session_turns", self.limits.max_turns_per_session)
            notes = self._reconcile()
            reply = self._dispatch(message)
            if notes:
                reply = "\n".join(notes + [reply])
        except LimitExceeded as e:
            reply = self._limit_message(e)
        except ScriptExhausted:
            raise
        except Exception as e:  # every exit ends with a clear message
            self.tracer.event("error", error_type=type(e).__name__, error=str(e))
            reply = ("Something went wrong on my side, so I stopped. " + self._safety_status() +
                     f" (reference: {self.tracer.run_id}). Your request details are saved; you can try again.")
        self._save()
        self.tracer.event("turn_end", turn=self.turns, phase=self.state.phase, reply=reply,
                          model_calls=self.budget.model_calls, tool_calls=self.budget.tool_calls)
        return reply

    def _save(self) -> None:
        self.store.save_session(self.user_id, self.state.to_dict(), self.clock.now().isoformat())

    def _set_phase(self, phase: str, reason: str) -> None:
        if phase != self.state.phase:
            self.tracer.event("state_transition", from_phase=self.state.phase, to_phase=phase, reason=reason)
            self.state.phase = phase

    def _dispatch(self, message: str) -> str:
        text = message.strip()
        phase = self.state.phase
        if phase == AWAITING_APPROVAL:
            if is_affirmative(text):
                return self._on_approve()
            if is_negative(text):
                return self._on_reject()
        elif phase == CONFIRM_MEMORY:
            if is_affirmative(text):
                return self._on_memory_answer(True)
            if is_negative(text):
                return self._on_memory_answer(False)
        elif is_affirmative(text):
            return self._nothing_to_confirm()
        return self._apply(self._interpret(text))

    # ======================= model =======================

    def _context(self) -> dict:
        now = self.clock.now()
        r = self.req
        return {
            "today": now.date().isoformat(),
            "weekday": now.strftime("%A"),
            "phase": self.state.phase,
            "request": {"intent": r.intent, **{f: getattr(r, f) for f in REQUEST_FIELDS}},
            "options": [{"n": o["n"], "label": o["label"]} for o in self.state.options],
            "last_question": self.state.asked[-1:] or None,
        }

    def _interpret(self, text: str) -> Interpretation:
        tier, reason = route(text, self.state.phase)
        tiers = [tier] + (["strong"] if tier == "cheap" else [])
        for i, t in enumerate(tiers):
            self.budget.spend_model_call()
            try:
                res = self.llm.interpret(text, self._context(), t)
            except ScriptExhausted:
                raise
            except Exception as e:
                self.tracer.event("model_error", tier=t, error_type=type(e).__name__, error=str(e))
                if i + 1 < len(tiers):
                    continue
                raise
            self.tracer.event("model_call", model=res.model, tier=t, route_reason=reason if i == 0 else "escalated",
                              latency_ms=res.latency_ms, input_tokens=res.input_tokens,
                              output_tokens=res.output_tokens, cost_usd=res.cost_usd,
                              interpretation={k: v for k, v in asdict(res.interpretation).items() if v not in (None, False)})
            if res.interpretation.intent == "unknown" and i + 1 < len(tiers):
                continue
            return res.interpretation
        return Interpretation()

    # ======================= applying an interpretation =======================

    def _apply(self, i: Interpretation) -> str:
        if i.remember_earliest_hour is not None:
            self.memory.state(str(i.remember_earliest_hour), "earliest_hour")
            if self.req.earliest_hour is None:
                i.earliest_hour = i.earliest_hour if i.earliest_hour is not None else i.remember_earliest_hour

        if i.intent == "memory_view":
            return self._render_memory()
        if i.intent == "memory_forget":
            return self._forget(i.memory_key or "all")
        if i.intent == "memory_correct":
            return self._correct_memory(i.memory_key, i.memory_value)
        if i.intent == "show_reviews":
            return self._show_reviews(i.selection)

        if i.intent in ("book", "reschedule", "cancel") and (
                self.req.intent != i.intent or self.state.phase in (IDLE, DONE)):
            self._start_request(i.intent)

        if self.req.intent is None:
            return ("I can book, reschedule or cancel home services, and show or forget what I remember about you. "
                    "What do you need?")

        changed = self._merge(i)
        if i.no_preference:
            self.req.prefs_answered = True
        if changed & set(SEARCH_FIELDS) and self.state.phase in (CHOOSING, AWAITING_APPROVAL):
            self._void_pending(f"details changed: {sorted(changed)}")
            self.state.options = []
            self._set_phase(COLLECTING, "correction")
            return self._advance()

        if i.selection is not None and i.intent in ("select_option", "provide_info", "book", "reschedule", "cancel") \
                and self.state.phase == CHOOSING:
            return self._select(i.selection)
        if i.intent == "decline" and self.state.phase == CHOOSING:
            return "No problem. Pick another option, or tell me what to change (day, time, budget or provider)."
        if i.intent in ("smalltalk", "unknown", "decline") and not changed:
            return self._reprompt()
        return self._advance()

    def _start_request(self, intent: str) -> None:
        self._void_pending("new request")
        self.state.request = Request(intent=intent)
        self.state.options = []
        self.state.asked = []
        self.state.memory_question = None
        self._set_phase(COLLECTING, f"new {intent} request")

    def _merge(self, i: Interpretation) -> set[str]:
        r = self.req
        changed = set()
        for f in REQUEST_FIELDS:
            v = getattr(i, f)
            if v is not None and v != getattr(r, f):
                setattr(r, f, v)
                r.sources[f] = "user"
                changed.add(f)
        if {"earliest_hour", "latest_hour", "budget_max"} & changed:
            r.prefs_answered = True
        if "date_from" in changed and "date_to" not in changed:
            r.date_to = None
        if "area" in changed and self.memory.usable("area") != r.area:
            self.memory.state(r.area, "area")
        if i.same_provider_as_last:
            changed |= self._use_last_provider()
        if changed:
            self.tracer.event("request_updated", changed=sorted(changed),
                              request={f: getattr(r, f) for f in REQUEST_FIELDS + ("provider_id",)})
        return changed

    def _use_last_provider(self) -> set[str]:
        r = self.req
        past = [b for b in self.gateway.list_bookings()
                if (r.service is None or b["service"] == r.service) and b["status"] in ("completed", "booked")]
        if not past:
            return set()
        last = max(past, key=lambda b: b["slot"])
        r.provider_id = last["provider"]
        r.sources["provider_id"] = "history"
        changed = {"provider_id"}
        if r.service is None:
            r.service, r.sources["service"] = last["service"], "history"
            changed.add("service")
        if r.area is None:
            r.area = self.gateway.get_provider(last["provider"])["area"]
            r.sources["area"] = "history"
            changed.add("area")
        return changed

    # ======================= advancing the workflow =======================

    def _advance(self) -> str:
        if self.req.intent == "book":
            return self._advance_book()
        if self.req.intent in ("reschedule", "cancel"):
            return self._advance_change()
        return self._reprompt()

    def _ask(self, fields: list[str]) -> str:
        r = self.req
        known = [f for f in fields if f != "prefs" and getattr(r, f) is not None]
        assert not known, f"bug: asking for known fields {known}"
        self.state.asked.extend(fields)
        self.tracer.event("ask", fields=fields)
        self._set_phase(COLLECTING, "missing details")
        parts = [ASK_TEXT[f] for f in fields]
        return "Could you tell me " + (", and ".join(parts) if len(parts) <= 2 else ", ".join(parts[:-1]) + ", and " + parts[-1]) + "?"

    def _fill_from_memory(self) -> str | None:
        r = self.req
        if r.area is None:
            if (v := self.memory.usable("area")):
                r.area, r.sources["area"] = v, "memory"
            elif (f := self.memory.pending_confirmation("area")):
                return self._ask_memory("area", f"Should I use {f.value.replace('_', ' ').title()} as your area?")
        if r.earliest_hour is None and (v := self.memory.usable("earliest_hour")):
            r.earliest_hour, r.sources["earliest_hour"] = int(v), "memory"
        if r.service and r.provider_id is None:
            key = TECH_PREFIX + r.service
            if (v := self.memory.usable(key)):
                r.provider_id, r.sources["provider_id"] = v, "memory"
            elif (f := self.memory.pending_confirmation(key)):
                name = self.gateway.get_provider(f.value)["name"]
                return self._ask_memory(key, f"Last time you booked {name} ({f.value}) for {human_service(r.service)}. "
                                             "Should I prefer them again?")
        return None

    def _ask_memory(self, key: str, question: str) -> str:
        self.state.memory_question = key
        self.tracer.event("ask_memory_confirmation", key=key)
        self._set_phase(CONFIRM_MEMORY, f"inferred fact {key} needs confirmation")
        return question + " (yes/no)"

    def _on_memory_answer(self, yes: bool) -> str:
        key = self.state.memory_question
        self.state.memory_question = None
        self._set_phase(COLLECTING, "memory answer")
        if key is None:
            return self._advance()
        if yes:
            self.memory.confirm(key)
            prefix = "Thanks, I'll remember that. "
        else:
            self.memory.forget(key)
            prefix = "OK, I won't use that. "
        return prefix + self._advance()

    def _advance_book(self) -> str:
        if (q := self._fill_from_memory()):
            return q
        r = self.req
        missing = [f for f in ("service", "area", "date_from") if getattr(r, f) is None]
        if missing:
            return self._ask(missing)
        if r.date_to is None:
            r.date_to = r.date_from
        if not r.prefs_answered and r.earliest_hour is None and r.latest_hour is None and r.budget_max is None:
            r.prefs_answered = True
            return self._ask(["prefs"])
        return self._search_and_offer()

    def _hour_ok(self, slot: str) -> bool:
        h = parse_slot(slot).hour
        r = self.req
        return (r.earliest_hour is None or h >= r.earliest_hour) and (r.latest_hour is None or h <= r.latest_hour)

    def _search_and_offer(self) -> str:
        r = self.req
        results = self.gateway.search_providers(r.service, r.area, (r.date_from, r.date_to))
        hard_provider = r.provider_id if r.sources.get("provider_id") in ("user", "history") else None
        excluded = {"budget": 0, "time": 0, "provider": 0}
        opts = []
        for p in results:
            if hard_provider and p["id"] != hard_provider:
                excluded["provider"] += 1
                continue
            if r.budget_max is not None and p["price"] > r.budget_max:
                excluded["budget"] += 1
                continue
            slots = [s for s in p["free_slots"] if self._hour_ok(s)]
            if not slots:
                excluded["time"] += 1
                continue
            opts.append({"kind": "provider_slot", "provider_id": p["id"], "name": p["name"], "price": p["price"],
                         "rating": p["rating"], "slot": slots[0], "preferred": p["id"] == r.provider_id})
        opts.sort(key=lambda o: (not o["preferred"], o["price"], -o["rating"]))
        opts = opts[:3]
        for n, o in enumerate(opts, 1):
            o["n"] = n
            o["label"] = f"{o['name']} ({o['provider_id']}), rating {o['rating']}, {human_slot(o['slot'])}, ₹{o['price']}"
        self.tracer.event("options_built", count=len(opts), searched=len(results), excluded=excluded)
        if not opts:
            self._set_phase(COLLECTING, "no matching options")
            reasons = [f"{n} over budget" if k == "budget" else f"{n} with no slot in your time window" if k == "time"
                       else f"{n} not your chosen provider" for k, n in excluded.items() if n]
            return (f"I found {len(results)} {human_service(r.service)}s but none fit ({'; '.join(reasons)}). "
                    "Would you like to change the day, time or budget?")
        self.state.options = opts
        self._set_phase(CHOOSING, "options offered")
        return self._render_options()

    def _render_options(self, header: str | None = None) -> str:
        r = self.req
        header = header or f"Here are the best {human_service(r.service)} options:"
        lines = [header] + [f"  {o['n']}. {o['label']}" + ("  (your preferred)" if o.get("preferred") else "")
                            for o in self.state.options]
        lines.append("Reply with a number to pick one, or ask to see an option's reviews.")
        return "\n".join(lines)

    def _advance_change(self) -> str:
        r = self.req
        if r.booking is None:
            now = self.clock.now().strftime("%Y-%m-%dT%H:%M")
            upcoming = [b for b in self.gateway.list_bookings() if b["status"] == "booked" and b["slot"] > now
                        and (r.service is None or b["service"] == r.service)]
            if not upcoming:
                self._set_phase(IDLE, "no booking to change")
                self.state.request = Request()
                return f"I couldn't find an upcoming {human_service(r.service) if r.service else ''} booking to {r.intent}.".replace("  ", " ")
            if len(upcoming) > 1:
                self.state.options = [{"kind": "booking", "n": n, "booking": b,
                                       "label": f"{b['id']}: {human_service(b['service'])} with {b['provider']}, {human_slot(b['slot'])}"}
                                      for n, b in enumerate(upcoming, 1)]
                self._set_phase(CHOOSING, "pick booking")
                return self._render_options(f"Which booking do you want to {r.intent}?")
            r.booking = upcoming[0]
            r.service = r.service or r.booking["service"]
        b = r.booking
        if r.intent == "cancel":
            return self._propose("cancel", b["provider"], b["slot"], b["price"], booking_id=b["id"])
        if r.date_from is None:
            return self._ask(["date_from"])
        provider = self.gateway.get_provider(b["provider"])
        results = self.gateway.search_providers(b["service"], provider["area"], (r.date_from, r.date_to or r.date_from))
        slots = next((p["free_slots"] for p in results if p["id"] == b["provider"]), [])
        slots = [s for s in slots if self._hour_ok(s) and s != b["slot"]][:5]
        if not slots:
            self._set_phase(COLLECTING, "no slots for reschedule")
            r.date_from = r.date_to = None
            return f"{provider['name']} has no free slots then. Which other day works?"
        self.state.options = [{"kind": "slot", "n": n, "slot": s, "provider_id": b["provider"], "price": b["price"],
                               "label": f"{human_slot(s)} with {provider['name']}"} for n, s in enumerate(slots, 1)]
        self._set_phase(CHOOSING, "slots offered")
        return self._render_options(f"Free slots to move your {human_service(b['service'])} ({b['id']}, now {human_slot(b['slot'])}):")

    def _reprompt(self) -> str:
        phase = self.state.phase
        if phase == CHOOSING and self.state.options:
            return self._render_options("Please pick one of these:")
        if phase == AWAITING_APPROVAL and (appr := self._pending_approval()):
            return self._proposal_text(appr)
        if phase == CONFIRM_MEMORY:
            return "Please answer yes or no to my question above."
        if self.req.intent:
            return self._advance()
        return ("I can book, reschedule or cancel home services, and show or forget what I remember about you. "
                "What do you need?")

    # ======================= selection & proposals =======================

    def _select(self, n: int) -> str:
        opts = self.state.options
        opt = next((o for o in opts if o["n"] == n), None)
        if opt is None:
            return f"Please pick a number between 1 and {len(opts)}."
        if opt["kind"] == "booking":
            self.req.booking = opt["booking"]
            self.req.service = self.req.service or opt["booking"]["service"]
            self.state.options = []
            return self._advance_change()
        if opt["kind"] == "slot":
            b = self.req.booking
            return self._propose("reschedule", b["provider"], opt["slot"], b["price"], booking_id=b["id"])
        try:
            hold = self._hold(opt["provider_id"], opt["slot"])
        except ToolError as e:
            if e.code != "slot_unavailable":
                raise
            return self._drop_option(opt, "That slot was just taken.")
        quoted = hold["quoted_price"]
        if self.req.budget_max is not None and quoted > self.req.budget_max:
            return self._drop_option(opt, f"{opt['name']} quoted ₹{quoted}, above your ₹{self.req.budget_max} budget.")
        note = f"Note: the provider quoted ₹{quoted}, not the ₹{opt['price']} listed.\n" if quoted != opt["price"] else ""
        return note + self._propose("book", opt["provider_id"], opt["slot"], quoted, hold=hold)

    def _hold(self, provider_id: str, slot: str, *, fresh: bool = False) -> dict:
        """Hold a slot, reusing our own unexpired hold so we never block ourselves."""
        k = f"{provider_id}|{slot}"
        now = self.clock.now().isoformat()
        self.state.holds = {key: h for key, h in self.state.holds.items() if h["expires_at"] > now}
        if not fresh and k in self.state.holds:
            self.tracer.event("hold_reused", hold_id=self.state.holds[k]["hold_id"])
            return self.state.holds[k]
        hold = self.gateway.hold_slot(provider_id, slot)
        self.state.holds[k] = hold
        return hold

    def _drop_option(self, opt: dict, reason: str) -> str:
        self.state.options = [o for o in self.state.options if o is not opt]
        for n, o in enumerate(self.state.options, 1):
            o["n"] = n
        if not self.state.options:
            self._set_phase(COLLECTING, "options exhausted")
            return reason + " " + self._search_and_offer()
        self._set_phase(CHOOSING, "option dropped")
        return self._render_options(reason + " Remaining options:")

    def _propose(self, action: str, provider_id: str, slot: str, price: int, *, booking_id: str | None = None,
                 hold: dict | None = None) -> str:
        self._void_pending("superseded by new proposal")
        appr = self.store.create_approval(self.user_id, action, provider_id, slot, price, booking_id=booking_id,
                                          hold_id=hold and hold["hold_id"], hold_expires_at=hold and hold["expires_at"])
        self.state.approval_id = appr.id
        self.tracer.event("approval_proposed", approval_id=appr.id, action=action, provider_id=provider_id, slot=slot,
                          price=price, booking_id=booking_id, fingerprint=appr.fingerprint)
        self._set_phase(AWAITING_APPROVAL, f"{action} proposed")
        return self._proposal_text(appr)

    def _proposal_text(self, appr: Approval) -> str:
        what = {"book": "BOOK", "cancel": "CANCEL booking " + (appr.booking_id or ""),
                "reschedule": f"MOVE booking {appr.booking_id} to"}[appr.action]
        return (f"Please confirm: {what} {appr.provider_id} for {human_service(self.req.service)} on "
                f"{human_slot(appr.slot)} at ₹{appr.price}. Reply 'yes' to confirm or 'no' to stop.")

    def _pending_approval(self) -> Approval | None:
        aid = self.state.approval_id
        return self.store.get_approval(aid) if aid else None

    def _void_pending(self, reason: str) -> None:
        for aid in self.store.void_open_approvals(self.user_id):
            self.tracer.event("approval_voided", approval_id=aid, reason=reason)
        self.state.approval_id = None

    def _on_reject(self) -> str:
        self._void_pending("user said no")
        if self.state.options:
            self._set_phase(CHOOSING, "user declined proposal")
            return self._render_options("OK, I haven't done anything. Here are the options again:")
        self._set_phase(IDLE, "user declined proposal")
        self.state.request = Request()
        return "OK, I haven't done anything. What would you like to do?"

    def _nothing_to_confirm(self) -> str:
        extra = f" Your last booking {self.state.last_booking_id} is already confirmed." if self.state.last_booking_id else ""
        self.tracer.event("confirmation_ignored", reason="no pending proposal")
        return "There's nothing waiting for your confirmation right now." + extra

    # ======================= execution =======================

    def _on_approve(self) -> str:
        appr = self._pending_approval()
        if appr is None or not self.store.grant_approval(appr.id):
            self._set_phase(CHOOSING if self.state.options else IDLE, "stale approval")
            return "That proposal is no longer valid, so I haven't done anything. " + self._reprompt()
        self.tracer.event("approval_granted", approval_id=appr.id, action=appr.action, provider_id=appr.provider_id,
                          slot=appr.slot, price=appr.price, fingerprint=appr.fingerprint)
        appr.status = APPROVAL_GRANTED
        try:
            if appr.action == "book":
                return self._execute_book(appr)
            if appr.action == "cancel":
                return self._execute_cancel(appr)
            return self._execute_reschedule(appr)
        except ApprovalError as e:
            self._void_pending(f"gateway refused: {e}")
            return f"I stopped because the approval check failed ({e}). Nothing was changed."

    def _execute_book(self, appr: Approval) -> str:
        key = f"book:{appr.id}"
        op = self.store.get_operation(key)
        if op and op.status == OP_DONE:
            return self._already_done(op)
        self.store.put_operation(key, appr.id, self.user_id, "book", OP_PENDING)
        if op and op.status in (OP_PENDING, OP_UNKNOWN):
            st = self.gateway.get_booking_status(key)
            if st["status"] == "booked":
                return self._book_succeeded(appr, key, st["booking_id"])
        if appr.hold_expires_at is None or appr.hold_expires_at <= self.clock.now().isoformat():
            if (msg := self._renew_hold(appr, key)):
                return msg
            appr = self.store.get_approval(appr.id)

        last_status = None
        for attempt in range(1, self.limits.max_book_attempts + 1):
            try:
                self.gateway.book(appr.id, appr.hold_id, key, provider_id=appr.provider_id, slot=appr.slot,
                                  price=appr.price)
            except ToolError as e:
                if e.code == "timeout":
                    st = self.gateway.get_booking_status(key)
                    self.tracer.event("booking_status_checked", after="timeout", attempt=attempt, status=st["status"])
                    if st["status"] == "booked":
                        return self._book_succeeded(appr, key, st["booking_id"])
                    if st["status"] == "failed":
                        return self._book_failed(appr, key, "the provider rejected the booking")
                    last_status = st["status"]
                    self.clock.sleep(self.limits.retry_backoff_seconds * attempt)
                    continue  # not_found: the request never landed, so retrying with the same key is safe
                if e.code == "hold_expired":
                    if (msg := self._renew_hold(appr, key)):
                        return msg
                    appr = self.store.get_approval(appr.id)
                    continue
                if e.code == "slot_taken":
                    return self._book_failed(appr, key, "someone else took that slot")
                raise
            st = self.gateway.get_booking_status(key)
            self.tracer.event("booking_status_checked", after="book_ok", attempt=attempt, status=st["status"])
            if st["status"] == "booked":
                return self._book_succeeded(appr, key, st["booking_id"])
            self.store.put_operation(key, appr.id, self.user_id, "book", OP_UNKNOWN)
            return ("The provider accepted the request but its status check doesn't show a booking yet, so I'm not "
                    "treating it as booked. I'll re-check before doing anything else.")
        if last_status == "not_found":
            return self._book_failed(appr, key, f"the provider didn't respond after {self.limits.max_book_attempts} "
                                                "attempts (I checked: no booking exists)")
        self.store.put_operation(key, appr.id, self.user_id, "book", OP_UNKNOWN)
        return ("I couldn't confirm the booking after several attempts. I won't retry blindly; I'll check its status "
                "with the provider before doing anything else.")

    def _renew_hold(self, appr: Approval, key: str) -> str | None:
        """Get a fresh hold. Returns a user message if the approval can no longer be used as-is."""
        try:
            hold = self._hold(appr.provider_id, appr.slot, fresh=True)
        except ToolError as e:
            if e.code != "slot_unavailable":
                raise
            return self._book_failed(appr, key, "the slot is no longer available")
        if hold["quoted_price"] != appr.price:
            self.store.put_operation(key, appr.id, self.user_id, "book", OP_FAILED, {"reason": "price_changed"})
            self.tracer.event("approval_voided", approval_id=appr.id, reason="price_changed",
                              approved_price=appr.price, new_price=hold["quoted_price"])
            self.store.set_approval_status(appr.id, APPROVAL_VOID)
            msg = self._propose("book", appr.provider_id, appr.slot, hold["quoted_price"], hold=hold)
            return f"The price changed from ₹{appr.price} to ₹{hold['quoted_price']} since you approved, so I have NOT booked.\n" + msg
        self.store.update_hold(appr.id, hold["hold_id"], hold["expires_at"])
        self.tracer.event("hold_renewed", approval_id=appr.id, hold_id=hold["hold_id"])
        return None

    def _book_succeeded(self, appr: Approval, key: str, booking_id: str) -> str:
        self.store.put_operation(key, appr.id, self.user_id, "book", OP_DONE, {"booking_id": booking_id})
        self.store.set_approval_status(appr.id, APPROVAL_CONSUMED)
        self.tracer.event("booking_verified", approval_id=appr.id, booking_id=booking_id)
        service = self.req.service
        if service and self.req.sources.get("provider_id") != "memory":
            self.memory.infer(TECH_PREFIX + service, appr.provider_id)
        self.state.last_booking_id = booking_id
        self.state.approval_id = None
        self.state.options = []
        self._set_phase(DONE, "booking verified")
        return (f"Booked and verified with the provider: {booking_id}, {appr.provider_id} on "
                f"{human_slot(appr.slot)} at ₹{appr.price}.")

    def _book_failed(self, appr: Approval, key: str, reason: str) -> str:
        self.store.put_operation(key, appr.id, self.user_id, "book", OP_FAILED, {"reason": reason})
        self.store.set_approval_status(appr.id, APPROVAL_VOID)
        self.tracer.event("approval_voided", approval_id=appr.id, reason=reason)
        self.state.approval_id = None
        opt = next((o for o in self.state.options if o.get("provider_id") == appr.provider_id
                    and o.get("slot") == appr.slot), None)
        if opt:
            return self._drop_option(opt, f"I couldn't book: {reason}. Nothing was booked.")
        self._set_phase(COLLECTING, "booking failed")
        return f"I couldn't book: {reason}. Nothing was booked. " + self._search_and_offer()

    def _already_done(self, op) -> str:
        self._set_phase(DONE, "already executed")
        self.state.approval_id = None
        return f"That's already done ({op.result}). I didn't repeat it."

    def _execute_cancel(self, appr: Approval) -> str:
        key = f"cancel:{appr.id}"
        op = self.store.get_operation(key)
        if op and op.status == OP_DONE:
            return self._already_done(op)
        self.store.put_operation(key, appr.id, self.user_id, "cancel", OP_PENDING)
        for attempt in range(1, self.limits.max_cancel_attempts + 1):
            try:
                self.gateway.cancel(appr.id, appr.booking_id, key, provider_id=appr.provider_id, slot=appr.slot,
                                    price=appr.price)
                break
            except ToolError as e:
                if not e.retryable:
                    raise
                self.tracer.event("retry", tool="cancel", attempt=attempt, error=e.code)
                self.clock.sleep(self.limits.retry_backoff_seconds * 2 ** (attempt - 1))
        else:
            self.store.put_operation(key, appr.id, self.user_id, "cancel", OP_UNKNOWN)
            return (f"The provider kept failing, so I couldn't cancel {appr.booking_id} after "
                    f"{self.limits.max_cancel_attempts} tries. It is still booked. I'll re-check before anything else.")
        return self._finish_change(appr, key, "cancel")

    def _execute_reschedule(self, appr: Approval) -> str:
        key = f"reschedule:{appr.id}"
        op = self.store.get_operation(key)
        if op and op.status == OP_DONE:
            return self._already_done(op)
        self.store.put_operation(key, appr.id, self.user_id, "reschedule", OP_PENDING)
        try:
            self.gateway.reschedule(appr.id, appr.booking_id, appr.slot, key, provider_id=appr.provider_id,
                                    price=appr.price)
        except ToolError as e:
            if e.code != "slot_conflict":
                raise
            self.store.put_operation(key, appr.id, self.user_id, "reschedule", OP_FAILED, {"reason": "slot_conflict"})
            self.store.set_approval_status(appr.id, APPROVAL_VOID)
            self.tracer.event("approval_voided", approval_id=appr.id, reason="slot_conflict")
            self.state.approval_id = None
            opt = next((o for o in self.state.options if o.get("slot") == appr.slot), None)
            msg = f"{human_slot(appr.slot)} conflicts with the provider's schedule, so nothing was changed."
            if opt:
                return self._drop_option(opt, msg)
            self._set_phase(COLLECTING, "reschedule conflict")
            return msg + " Which other day works?"
        return self._finish_change(appr, key, "reschedule")

    def _finish_change(self, appr: Approval, key: str, action: str) -> str:
        booking = next((b for b in self.gateway.list_bookings() if b["id"] == appr.booking_id), None)
        ok = booking is not None and (booking["status"] == "cancelled" if action == "cancel"
                                      else booking["slot"] == appr.slot)
        if not ok:
            self.store.put_operation(key, appr.id, self.user_id, action, OP_UNKNOWN)
            return f"The provider acknowledged the {action}, but I couldn't verify it yet. I'll re-check before anything else."
        self.store.put_operation(key, appr.id, self.user_id, action, OP_DONE, {"booking": booking})
        self.store.set_approval_status(appr.id, APPROVAL_CONSUMED)
        self.tracer.event(f"{action}_verified", approval_id=appr.id, booking_id=appr.booking_id)
        self.state.approval_id = None
        self.state.options = []
        self._set_phase(DONE, f"{action} verified")
        if action == "cancel":
            return f"Cancelled and verified: {appr.booking_id} ({appr.provider_id}, {human_slot(appr.slot)})."
        return f"Rescheduled and verified: {appr.booking_id} is now on {human_slot(appr.slot)} with {appr.provider_id}."

    # ======================= recovery =======================

    def _reconcile(self) -> list[str]:
        """Resolve operations left pending by a crash or exhausted retries, before doing anything new."""
        notes = []
        for op in self.store.unresolved_operations(self.user_id):
            appr = self.store.get_approval(op.approval_id)
            self.tracer.event("reconcile_start", idempotency_key=op.idempotency_key, status=op.status)
            if op.action == "book":
                st = self.gateway.get_booking_status(op.idempotency_key)
                self.tracer.event("booking_status_checked", after="restart", status=st["status"])
                if st["status"] == "booked":
                    notes.append("Update on your earlier request: " +
                                 self._book_succeeded(appr, op.idempotency_key, st["booking_id"]))
                    continue
                reason = "it never reached the provider" if st["status"] == "not_found" else "the provider rejected it"
            else:
                booking = next((b for b in self.gateway.list_bookings() if b["id"] == appr.booking_id), None)
                done = booking and (booking["status"] == "cancelled" if op.action == "cancel" else booking["slot"] == appr.slot)
                if done:
                    notes.append("Update on your earlier request: " + self._finish_change(appr, op.idempotency_key, op.action))
                    continue
                reason = "it did not go through"
            self.store.put_operation(op.idempotency_key, op.approval_id, self.user_id, op.action, OP_FAILED,
                                     {"reason": reason})
            if appr.status in (APPROVAL_GRANTED, APPROVAL_PROPOSED):
                self.store.set_approval_status(appr.id, APPROVAL_VOID)
                self.tracer.event("approval_voided", approval_id=appr.id, reason="interrupted: " + reason)
            if self.state.approval_id == appr.id:
                self.state.approval_id = None
                self._set_phase(CHOOSING if self.state.options else COLLECTING, "interrupted action resolved")
            notes.append(f"Your earlier {op.action} was interrupted and {reason}; nothing was changed.")
        return notes

    # ======================= memory & reviews =======================

    def _render_memory(self) -> str:
        facts = self.memory.view()
        if not facts:
            return "I don't have any saved preferences for you."
        lines = ["Here's what I remember:"]
        for f in facts:
            tag = "you told me" if f.source == "stated" else ("inferred, confirmed by you" if f.confirmed
                                                              else "inferred, not used until you confirm")
            lines.append(f"  - {f.key}: {f.value} ({tag})")
        lines.append("You can correct or forget any of these.")
        return "\n".join(lines)

    def _forget(self, key: str) -> str:
        forgotten = self.memory.forget(key)
        r = self.req
        # Clear the forgotten value from the request in progress too, whatever its source:
        # a persisted workflow must not keep using what the user asked us to forget.
        cleared = []
        for f in ("area", "earliest_hour", "provider_id"):
            mem_key = TECH_PREFIX + (r.service or "") if f == "provider_id" else f
            if mem_key in forgotten and getattr(r, f) is not None:
                setattr(r, f, None)
                r.sources.pop(f, None)
                cleared.append(f)
        if cleared and self.state.phase in (CHOOSING, AWAITING_APPROVAL):
            self._void_pending(f"forgotten: {cleared}")
            self.state.options = []
            self._set_phase(COLLECTING, "forgotten fact cleared from request")
        if not forgotten:
            return "I didn't have anything saved for that."
        return "Done. I've forgotten: " + ", ".join(forgotten) + "."

    def _correct_memory(self, key: str | None, value: str | None) -> str:
        if not key or value is None or key in ("all", "preferred_technician"):
            return "Which preference should I change, and to what?"
        self.memory.state(value, key)
        if self.req.sources.get(key) == "memory":
            setattr(self.req, key, int(value) if key == "earliest_hour" else value)
        return f"Updated: {key} is now {value}."

    def _show_reviews(self, n: int | None) -> str:
        opt = next((o for o in self.state.options if o["n"] == n and o.get("provider_id")), None)
        if opt is None:
            return "Which option's reviews would you like? Give me its number."
        provider = self.gateway.get_provider(opt["provider_id"])
        reviews = provider["reviews"]
        verdicts = self.guard.check(reviews)
        flagged = [i for i, v in enumerate(verdicts) if v.flagged]
        self.tracer.event("guard_check", provider_id=opt["provider_id"], guard=getattr(self.guard, "name", "?"),
                          reviews=len(reviews), flagged=flagged,
                          scores=[v.score for v in verdicts], sources=[v.source for v in verdicts],
                          guard_error=getattr(self.guard, "last_error", None))
        lines = [f"Customer reviews for {provider['name']} (shown as data, not instructions):"]
        for i, text in enumerate(reviews):
            lines.append("  - [withheld: this review contained instructions aimed at an assistant, so I ignored it]"
                         if i in flagged else f'  - "{text}"')
        return "\n".join(lines) + "\n" + self._reprompt()

    # ======================= exits =======================

    def _mutations_this_turn(self) -> list[dict]:
        return [e for e in self.tracer.events[self._turn_start_seq:]
                if e["kind"] == "tool_call" and e["tool"] in MUTATING and e["ok"]]

    def _safety_status(self) -> str:
        if self.store.unresolved_operations(self.user_id):
            return ("An action is unconfirmed; I'll check its status with the provider before doing anything else, "
                    "so nothing will be duplicated.")
        if self._mutations_this_turn():
            return "An action completed this turn; see your bookings for the result."
        return "Nothing was booked, cancelled or changed."

    def _limit_message(self, e: LimitExceeded) -> str:
        label = {"wall_clock": "time", "model_calls": "model-call", "tool_calls": "tool-call",
                 "session_turns": "conversation-length"}.get(e.limit, e.limit)
        return f"I had to stop: this turn hit the {label} limit ({e.cap}). " + self._safety_status()
