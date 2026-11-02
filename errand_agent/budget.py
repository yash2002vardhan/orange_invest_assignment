"""Hard per-turn caps on model calls, tool calls and wall-clock time."""

from __future__ import annotations

from .config import Limits


class LimitExceeded(Exception):
    def __init__(self, limit: str, cap):
        super().__init__(f"{limit} limit reached ({cap})")
        self.limit = limit
        self.cap = cap


class Budget:
    def __init__(self, limits: Limits, clock, tracer):
        self.limits = limits
        self.clock = clock
        self.tracer = tracer
        self.started = clock.monotonic()
        self.model_calls = 0
        self.tool_calls = 0

    def _check_time(self) -> None:
        elapsed = self.clock.monotonic() - self.started
        if elapsed > self.limits.max_turn_seconds:
            self.tracer.event("limit_exceeded", limit="wall_clock", cap=self.limits.max_turn_seconds,
                              elapsed_s=round(elapsed, 2))
            raise LimitExceeded("wall_clock", f"{self.limits.max_turn_seconds:.0f}s")

    def spend_model_call(self) -> None:
        self._check_time()
        if self.model_calls >= self.limits.max_model_calls_per_turn:
            self.tracer.event("limit_exceeded", limit="model_calls", cap=self.limits.max_model_calls_per_turn)
            raise LimitExceeded("model_calls", self.limits.max_model_calls_per_turn)
        self.model_calls += 1

    def spend_tool_call(self) -> None:
        self._check_time()
        if self.tool_calls >= self.limits.max_tool_calls_per_turn:
            self.tracer.event("limit_exceeded", limit="tool_calls", cap=self.limits.max_tool_calls_per_turn)
            raise LimitExceeded("tool_calls", self.limits.max_tool_calls_per_turn)
        self.tool_calls += 1
