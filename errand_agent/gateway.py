"""The only path from the agent to the provider API.

Read tools are traced and budgeted. Mutating tools (book, cancel, reschedule)
additionally require an approval that the gateway re-loads from the store and
checks itself: status must be `granted` and its fingerprint must match the exact
action, provider, slot and price being executed. Nothing the model says, and no
argument the caller passes, can substitute for that record.
"""

from __future__ import annotations

from typing import Callable

from .budget import Budget
from .mock_api import MockProviderAPI, ToolError
from .store import APPROVAL_GRANTED, Store


class ApprovalError(Exception):
    pass


class ToolGateway:
    def __init__(self, api: MockProviderAPI, store: Store, tracer, clock, user_id: str):
        self.api = api
        self.store = store
        self.tracer = tracer
        self.clock = clock
        self.user_id = user_id
        self.budget: Budget | None = None
        # test-only fault injection: name -> callable(result) run after a tool returns
        self.after_hooks: dict[str, Callable[[object], None]] = {}

    def _call(self, name: str, fn: Callable, **args):
        assert self.budget is not None, "gateway used outside a turn"
        self.budget.spend_tool_call()
        start = self.clock.monotonic()
        try:
            result = fn(**args)
        except ToolError as e:
            self.tracer.event("tool_call", tool=name, args=args, ok=False, error=e.code,
                              latency_ms=round((self.clock.monotonic() - start) * 1000, 2))
            raise
        self.tracer.event("tool_call", tool=name, args=args, ok=True, result=result,
                          latency_ms=round((self.clock.monotonic() - start) * 1000, 2))
        if name in self.after_hooks:
            self.after_hooks[name](result)
        return result

    def _authorize(self, approval_id: str, action: str, provider_id: str, slot: str, price: int,
                   booking_id: str | None) -> None:
        appr = self.store.get_approval(approval_id)
        reason = None
        if appr is None:
            reason = "no such approval"
        elif appr.user_id != self.user_id:
            reason = "approval belongs to another user"
        elif appr.status != APPROVAL_GRANTED:
            reason = f"approval status is {appr.status}"
        elif not appr.matches(action, provider_id, slot, price, booking_id):
            reason = "action differs from what the user approved"
        if reason:
            self.tracer.event("approval_rejected", approval_id=approval_id, action=action, reason=reason)
            raise ApprovalError(reason)
        self.tracer.event("approval_checked", approval_id=approval_id, action=action, fingerprint=appr.fingerprint)

    # ---------- read tools ----------

    def search_providers(self, service: str, area: str, date_range: tuple[str, str]):
        return self._call("search_providers", self.api.search_providers,
                          service=service, area=area, date_range=list(date_range))

    def get_provider(self, provider_id: str):
        return self._call("get_provider", self.api.get_provider, provider_id=provider_id)

    def list_bookings(self):
        return self._call("list_bookings", self.api.list_bookings, user_id=self.user_id)

    def hold_slot(self, provider_id: str, slot: str):
        # A hold commits nothing; it is how we learn the real quoted price before asking for approval.
        return self._call("hold_slot", self.api.hold_slot, provider_id=provider_id, slot=slot)

    def get_booking_status(self, idempotency_key: str):
        return self._call("get_booking_status", self.api.get_booking_status, idempotency_key=idempotency_key)

    # ---------- mutating tools ----------

    def book(self, approval_id: str, hold_id: str, idempotency_key: str, *, provider_id: str, slot: str,
             price: int):
        self._authorize(approval_id, "book", provider_id, slot, price, None)
        return self._call("book", self.api.book, hold_id=hold_id, idempotency_key=idempotency_key,
                          user_id=self.user_id)

    def cancel(self, approval_id: str, booking_id: str, idempotency_key: str, *, provider_id: str, slot: str,
               price: int):
        self._authorize(approval_id, "cancel", provider_id, slot, price, booking_id)
        return self._call("cancel", self.api.cancel, booking_id=booking_id, idempotency_key=idempotency_key)

    def reschedule(self, approval_id: str, booking_id: str, new_slot: str, idempotency_key: str, *,
                   provider_id: str, price: int):
        self._authorize(approval_id, "reschedule", provider_id, new_slot, price, booking_id)
        return self._call("reschedule", self.api.reschedule, booking_id=booking_id, new_slot=new_slot,
                          idempotency_key=idempotency_key)
