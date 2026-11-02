"""One JSONL trace per run. Every record passes through `sanitize` before it is written."""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

_SECRET_KEYS = re.compile(r"^(api_?key|secret|authorization|password|access_token|bearer)$", re.I)
_SECRET_VALUES = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9_\-.]{8,})")
MAX_STR = 240
MAX_LIST = 8


def _env_secrets() -> list[str]:
    names = ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "JEV_API_KEY", "TYPESAFE_API_KEY")
    return [v for n in names if (v := os.environ.get(n)) and len(v) >= 8]


def sanitize(obj, _secrets: list[str] | None = None):
    secrets = _env_secrets() if _secrets is None else _secrets
    if isinstance(obj, dict):
        return {k: "[REDACTED]" if _SECRET_KEYS.match(str(k)) else sanitize(v, secrets) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        items = [sanitize(v, secrets) for v in list(obj)[:MAX_LIST]]
        if len(obj) > MAX_LIST:
            items.append(f"...(+{len(obj) - MAX_LIST} more)")
        return items
    if isinstance(obj, str):
        for s in secrets:
            obj = obj.replace(s, "[REDACTED]")
        obj = _SECRET_VALUES.sub("[REDACTED]", obj)
        return obj if len(obj) <= MAX_STR else obj[:MAX_STR] + f"...(+{len(obj) - MAX_STR} chars)"
    return obj


class Tracer:
    def __init__(self, path: str | Path | None, clock, run_id: str | None = None):
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:10]}"
        self.clock = clock
        self.path = Path(path) if path else None
        self.events: list[dict] = []
        self._seq = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, **data) -> dict:
        self._seq += 1
        record = {"run_id": self.run_id, "seq": self._seq, "ts": self.clock.now().isoformat(), "kind": kind}
        record.update(sanitize(data))
        self.events.append(record)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        return record

    def of_kind(self, *kinds: str) -> list[dict]:
        return [e for e in self.events if e["kind"] in kinds]
