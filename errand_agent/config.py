from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines into os.environ. Variables already set in the shell win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Limits:
    max_model_calls_per_turn: int = 3
    max_tool_calls_per_turn: int = 12
    max_turn_seconds: float = 45.0
    max_turns_per_session: int = 40
    max_book_attempts: int = 3
    max_cancel_attempts: int = 3
    retry_backoff_seconds: float = 0.5


# USD per 1M tokens (input, output). These are the published launch prices of
# the default models; check current OpenAI pricing before quoting costs.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
}


@dataclass(frozen=True)
class ModelConfig:
    cheap: str = "gpt-5-mini"
    strong: str = "gpt-5"
    prices: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            cheap=os.environ.get("ERRAND_CHEAP_MODEL", cls.cheap),
            strong=os.environ.get("ERRAND_STRONG_MODEL", cls.strong),
        )

    def cost_usd(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        price = self.prices.get(model)
        if price is None:
            return None
        return round(input_tokens / 1e6 * price[0] + output_tokens / 1e6 * price[1], 6)
