from __future__ import annotations

from dataclasses import dataclass

from adapters.llm.providers.base import LLMProvider, LLMResponse
from ai_assist.controlled_english import (
    draft_controlled_english,
    repair_controlled_english,
    repair_controlled_english_for_clarifications,
)


@dataclass
class FakeProvider(LLMProvider):
    name: str = "fake"

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: str | None = None,
        timeout_s: float = 120.0,
    ) -> LLMResponse:
        # Return something that the existing parser should understand.
        return LLMResponse(
            provider=self.name,
            model=model,
            content="Go long when EMA(20) crosses above EMA(50) on 4h\nStop ATR(14) * 3",
            raw={"messages": messages},
        )


def test_ai_draft_controlled_english_smoke(tmp_path):
    provider = FakeProvider()
    out = draft_controlled_english(
        notes="some messy notes",
        provider=provider,
        model="fake-model",
        strategy_name="test_strategy",
        default_entry_tf="1h",
        artifacts_base_dir=tmp_path,
    )

    assert "Go long" in out.controlled_english
    assert out.bundle_dir is not None
    assert out.bundle_dir.exists()


def test_ai_repair_controlled_english_smoke(tmp_path):
    provider = FakeProvider()
    out = repair_controlled_english(
        notes="some messy notes",
        previous_controlled_english="nonsense",
        errors="Parser error: ValueError: could not parse",
        provider=provider,
        model="fake-model",
        strategy_name="test_strategy",
        default_entry_tf="1h",
        artifacts_base_dir=tmp_path,
        attempt=1,
    )

    assert "Go long" in out.controlled_english
    assert out.bundle_dir is not None
    assert out.bundle_dir.exists()


def test_ai_repair_controlled_english_for_clarifications_smoke(tmp_path):
    provider = FakeProvider()
    out = repair_controlled_english_for_clarifications(
        notes="some messy notes",
        previous_controlled_english="Go long when EMA(20) crosses above EMA(50)",
        clarifications="- condition_tf: missing timeframe (default=1h)",
        provider=provider,
        model="fake-model",
        strategy_name="test_strategy",
        default_entry_tf="1h",
        artifacts_base_dir=tmp_path,
        attempt=1,
    )

    assert "Go long" in out.controlled_english
    assert out.bundle_dir is not None
    assert out.bundle_dir.exists()
