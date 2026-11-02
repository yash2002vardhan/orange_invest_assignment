"""Injectable clock so hold expiry and wall-clock limits are testable.

All datetimes are naive local time (the mock world is assumed to be IST).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta


class Clock:
    def now(self) -> datetime:
        return datetime.now().replace(microsecond=0)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class FakeClock(Clock):
    def __init__(self, start: datetime | None = None):
        self._now = start or datetime(2026, 11, 2, 9, 0)  # a Monday
        self._mono = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._mono += seconds

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)
