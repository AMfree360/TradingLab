from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class LLMProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    provider: str
    model: str
    content: str
    raw: dict[str, Any] | None = None
    request_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_s: float | None = None


class LLMProvider(Protocol):
    name: str

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: str | None = None,
        timeout_s: float = 120.0,
    ) -> LLMResponse:
        raise NotImplementedError
