from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adapters.llm.providers.base import LLMProvider, LLMResponse
from ai_assist.artifacts import create_ai_bundle_dir, write_ai_bundle


PROMPT_VERSION = "controlled_english_v1"
REPAIR_PROMPT_VERSION = "controlled_english_repair_v1"
CLARIFICATIONS_REPAIR_PROMPT_VERSION = "controlled_english_clarifications_repair_v1"


def _load_prompt_template(prompt_version: str) -> str:
    prompt_path = Path(__file__).parent.parent / "ai" / "prompts" / f"{prompt_version}.md"
    return prompt_path.read_text()


def load_prompt_template() -> str:
    return _load_prompt_template(PROMPT_VERSION)


def load_repair_prompt_template() -> str:
    return _load_prompt_template(REPAIR_PROMPT_VERSION)


def load_clarifications_repair_prompt_template() -> str:
    return _load_prompt_template(CLARIFICATIONS_REPAIR_PROMPT_VERSION)


@dataclass(frozen=True)
class ControlledEnglishDraft:
    controlled_english: str
    llm: LLMResponse
    bundle_dir: Path | None = None


def draft_controlled_english(
    *,
    notes: str,
    provider: LLMProvider,
    model: str,
    strategy_name: str,
    default_entry_tf: str | None,
    artifacts_base_dir: Path | None,
    temperature: float = 0.0,
) -> ControlledEnglishDraft:
    template = load_prompt_template()
    user_prompt = template.format(
        strategy_name=strategy_name,
        default_entry_tf=(default_entry_tf or ""),
        notes=notes,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a trading strategy transcription assistant. "
                "Output only controlled-English strategy lines. "
                "No commentary, no markdown fences."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    llm = provider.chat(model=model, messages=messages, temperature=temperature)

    bundle_dir = None
    if artifacts_base_dir is not None:
        bundle_dir = create_ai_bundle_dir(base_dir=artifacts_base_dir, prefix="controlled_english")
        prompt_obj: dict[str, Any] = {
            "prompt_version": PROMPT_VERSION,
            "messages": messages,
            "temperature": temperature,
        }
        meta = {
            "provider": llm.provider,
            "model": llm.model,
            "request_id": llm.request_id,
            "latency_s": llm.latency_s,
            "prompt_tokens": llm.prompt_tokens,
            "completion_tokens": llm.completion_tokens,
            "total_tokens": llm.total_tokens,
        }
        write_ai_bundle(
            root=bundle_dir,
            input_notes=notes,
            prompt=prompt_obj,
            response_raw=llm.raw,
            output_text=llm.content,
            meta=meta,
        )

    return ControlledEnglishDraft(controlled_english=llm.content, llm=llm, bundle_dir=bundle_dir)


def repair_controlled_english(
    *,
    notes: str,
    previous_controlled_english: str,
    errors: str,
    provider: LLMProvider,
    model: str,
    strategy_name: str,
    default_entry_tf: str | None,
    artifacts_base_dir: Path | None,
    temperature: float = 0.0,
    attempt: int = 1,
) -> ControlledEnglishDraft:
    """Ask the LLM to fix its previous controlled-English output.

    This is a best-effort helper. The deterministic parser + schema validation
    remain the source of truth.
    """

    template = load_repair_prompt_template()
    user_prompt = template.format(
        strategy_name=strategy_name,
        default_entry_tf=(default_entry_tf or ""),
        notes=notes,
        previous=previous_controlled_english,
        errors=errors,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a trading strategy transcription assistant. "
                "Output only controlled-English strategy lines. "
                "No commentary, no markdown fences."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    llm = provider.chat(model=model, messages=messages, temperature=temperature)

    bundle_dir = None
    if artifacts_base_dir is not None:
        bundle_dir = create_ai_bundle_dir(base_dir=artifacts_base_dir, prefix=f"controlled_english_repair_a{attempt}")
        prompt_obj: dict[str, Any] = {
            "prompt_version": REPAIR_PROMPT_VERSION,
            "messages": messages,
            "temperature": temperature,
            "attempt": attempt,
        }
        meta = {
            "provider": llm.provider,
            "model": llm.model,
            "request_id": llm.request_id,
            "latency_s": llm.latency_s,
            "prompt_tokens": llm.prompt_tokens,
            "completion_tokens": llm.completion_tokens,
            "total_tokens": llm.total_tokens,
        }
        # Embed the previous output + errors inside the prompt; also record them
        # in output.txt/input_notes.txt to keep bundles self-contained.
        write_ai_bundle(
            root=bundle_dir,
            input_notes=notes,
            prompt=prompt_obj,
            response_raw=llm.raw,
            output_text=llm.content,
            meta={
                **meta,
                "previous_controlled_english": previous_controlled_english,
                "errors": errors,
            },
        )

    return ControlledEnglishDraft(controlled_english=llm.content, llm=llm, bundle_dir=bundle_dir)


def repair_controlled_english_for_clarifications(
    *,
    notes: str,
    previous_controlled_english: str,
    clarifications: str,
    provider: LLMProvider,
    model: str,
    strategy_name: str,
    default_entry_tf: str | None,
    artifacts_base_dir: Path | None,
    temperature: float = 0.0,
    attempt: int = 1,
) -> ControlledEnglishDraft:
    """Ask the LLM to update controlled-English output to eliminate clarifications.

    This is conservative by prompt design: it should use defaults when provided
    and avoid inventing missing values.
    """

    template = load_clarifications_repair_prompt_template()
    user_prompt = template.format(
        strategy_name=strategy_name,
        default_entry_tf=(default_entry_tf or ""),
        notes=notes,
        previous=previous_controlled_english,
        clarifications=clarifications,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a trading strategy transcription assistant. "
                "Output only controlled-English strategy lines. "
                "No commentary, no markdown fences."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    llm = provider.chat(model=model, messages=messages, temperature=temperature)

    bundle_dir = None
    if artifacts_base_dir is not None:
        bundle_dir = create_ai_bundle_dir(
            base_dir=artifacts_base_dir, prefix=f"controlled_english_clarify_repair_a{attempt}"
        )
        prompt_obj: dict[str, Any] = {
            "prompt_version": CLARIFICATIONS_REPAIR_PROMPT_VERSION,
            "messages": messages,
            "temperature": temperature,
            "attempt": attempt,
        }
        meta = {
            "provider": llm.provider,
            "model": llm.model,
            "request_id": llm.request_id,
            "latency_s": llm.latency_s,
            "prompt_tokens": llm.prompt_tokens,
            "completion_tokens": llm.completion_tokens,
            "total_tokens": llm.total_tokens,
        }
        write_ai_bundle(
            root=bundle_dir,
            input_notes=notes,
            prompt=prompt_obj,
            response_raw=llm.raw,
            output_text=llm.content,
            meta={
                **meta,
                "previous_controlled_english": previous_controlled_english,
                "clarifications": clarifications,
            },
        )

    return ControlledEnglishDraft(controlled_english=llm.content, llm=llm, bundle_dir=bundle_dir)
