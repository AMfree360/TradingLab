from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

from adapters.llm.providers.base import LLMProvider, LLMProviderError, LLMResponse


@dataclass
class OllamaClient(LLMProvider):
    """Ollama chat client.

    Uses the Ollama HTTP API. Defaults to local ollama at http://localhost:11434.

    Notes:
    - We keep this intentionally minimal.
    - Structured output is best-effort via prompting (and optional `format`).
    """

    base_url: str = "http://localhost:11434"

    @property
    def name(self) -> str:  # type: ignore[override]
        return "ollama"

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: str | None = None,
        timeout_s: float = 120.0,
    ) -> LLMResponse:
        url = self.base_url.rstrip("/") + "/api/chat"

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        # Ollama supports `format: "json"` for some endpoints/models.
        # We treat it as best-effort.
        if response_format == "json":
            payload["format"] = "json"

        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=payload, timeout=timeout_s)
        except requests.RequestException as e:
            raise LLMProviderError(
                f"Failed to reach Ollama at {url}. Is `ollama serve` running? ({e})"
            ) from e

        latency_s = time.perf_counter() - t0
        if r.status_code >= 400:
            raise LLMProviderError(f"Ollama error {r.status_code}: {r.text[:500]}")

        data = r.json()
        msg = (data.get("message") or {})
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("Ollama response missing message.content")

        # Token counts may exist as `prompt_eval_count` / `eval_count`.
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        total_tokens = None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = prompt_tokens + completion_tokens

        return LLMResponse(
            provider=self.name,
            model=model,
            content=content.strip(),
            raw=data,
            request_id=str(data.get("id")) if data.get("id") is not None else None,
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
            total_tokens=total_tokens,
            latency_s=latency_s,
        )
