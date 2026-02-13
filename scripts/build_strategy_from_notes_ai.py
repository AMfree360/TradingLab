#!/usr/bin/env python3

"""Draft a research-layer strategy spec from messy notes using an LLM.

This script is intentionally designed so AI is **not** a source of truth.
The LLM only rewrites notes into controlled-English lines.
TradingLab then parses + validates the output deterministically.

Typical usage (Ollama):
  python scripts/build_strategy_from_notes_ai.py --name my_strategy --in notes.txt \
    --provider ollama --model llama3.1 --entry-tf 1h --compile

Artifacts:
  Writes an AI bundle under data/logs/ai/ (prompt + raw response + output + hashes)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.llm.providers.ollama import OllamaClient
from ai_assist.controlled_english import (
    draft_controlled_english,
    repair_controlled_english,
    repair_controlled_english_for_clarifications,
)
from research.compiler import StrategyCompiler
from research.nl_parser import apply_clarifications, parse_english_strategy
from research.spec import StrategySpec


def _read_text(path: str | None) -> str:
    if path:
        return Path(path).read_text()
    return sys.stdin.read()


def _prompt_one(key: str, question: str, options: list[str] | None, default: str | None) -> str:
    print("\n" + question)
    if options:
        for i, opt in enumerate(options, start=1):
            print(f"  {i}) {opt}")
        if default:
            print(f"Default: {default}")
        while True:
            raw = input("Select option # or type a value: ").strip()
            if not raw and default:
                return default
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return options[int(raw) - 1]
            if raw:
                return raw
    else:
        if default:
            raw = input(f"{key} [{default}]: ").strip()
            return raw or default
        return input(f"{key}: ").strip()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Use an LLM to rewrite messy notes into controlled-English, then compile a TradingLab research spec"
    )
    p.add_argument("--name", required=True, help="Strategy name (used for folder + spec)")
    p.add_argument("--in", dest="in_path", help="Path to a .txt file (default: read from stdin)")
    p.add_argument("--entry-tf", help="Default entry timeframe if not specified (e.g., 1h)")

            p.add_argument(
                "--parse-mode",
                choices=["strict", "lenient"],
                default="strict",
                help=(
                    "Parser mode. 'strict' requires explicit controlled-English templates; "
                    "'lenient' broadens which lines count as entry rules without guessing values."
                ),
            )
    p.add_argument("--provider", choices=["ollama"], default="ollama")
    p.add_argument("--model", required=True, help="Model name (e.g., llama3.1)")
    p.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument(
        "--out-spec",
        help="Where to write the YAML spec (default: research_specs/<name>.yml)",
    )
    p.add_argument("--compile", action="store_true", help="Compile the spec into strategies/<name>/")
    p.add_argument("--non-interactive", action="store_true", help="Fail if clarifications are required")

    p.add_argument(
        "--artifacts-dir",
        default="data/logs/ai",
        help="Base directory for AI prompt/response artifacts",
    )
    p.add_argument(
        "--print-controlled-english",
        action="store_true",
        help="Print the controlled-English draft before parsing",
    )

    p.add_argument(
        "--max-repairs",
        type=int,
        default=2,
        help="Max number of LLM repair attempts when parsing/validation fails",
    )

    p.add_argument(
        "--auto-resolve-clarifications",
        action="store_true",
        help=(
            "Ask the LLM to rewrite controlled-English to eliminate parser clarifications using defaults when possible"
        ),
    )
    p.add_argument(
        "--max-clarify-repairs",
        type=int,
        default=1,
        help="Max number of LLM attempts to eliminate clarifications (only when --auto-resolve-clarifications)",
    )

    args = p.parse_args()

    repo_root = Path(__file__).parent.parent
    notes = _read_text(args.in_path)

    if args.provider == "ollama":
        provider = OllamaClient(base_url=args.ollama_url)
    else:
        raise SystemExit("Unsupported provider")

    artifacts_base = repo_root / args.artifacts_dir if args.artifacts_dir else None
    if artifacts_base is not None:
        artifacts_base.mkdir(parents=True, exist_ok=True)

    draft = draft_controlled_english(
        notes=notes,
        provider=provider,
        model=args.model,
        strategy_name=args.name,
        default_entry_tf=args.entry_tf,
        artifacts_base_dir=artifacts_base,
        temperature=args.temperature,
    )

    def _print_draft(label: str, text: str) -> None:
        if not args.print_controlled_english:
            return
        print(f"\n--- Controlled English ({label}) ---\n")
        print(text)
        print("\n--- End draft ---\n")

    def _format_validation_error(e: ValidationError) -> str:
        # Keep stable, readable output for LLM repair.
        lines: list[str] = []
        for err in e.errors():
            loc = ".".join(str(p) for p in err.get("loc", []))
            msg = err.get("msg", "")
            typ = err.get("type", "")
            lines.append(f"{loc}: {msg} ({typ})".strip())
        return "\n".join(lines) if lines else str(e)

    controlled = draft.controlled_english
    _print_draft("initial", controlled)

    # Repair loop: if parsing or schema validation fails, ask the LLM to fix
    # its controlled-English output using deterministic error messages.
    result = None
    last_error: str | None = None
    for attempt in range(0, max(0, int(args.max_repairs)) + 1):
        try:
                    result = parse_english_strategy(
                        controlled,
                        name=args.name,
                        default_entry_tf=args.entry_tf,
                        mode=args.parse_mode,
                    )
        except Exception as e:  # parser/template failure
            last_error = f"Parser error: {type(e).__name__}: {e}"
            if attempt >= int(args.max_repairs):
                raise

            repaired = repair_controlled_english(
                notes=notes,
                previous_controlled_english=controlled,
                errors=last_error,
                provider=provider,
                model=args.model,
                strategy_name=args.name,
                default_entry_tf=args.entry_tf,
                artifacts_base_dir=artifacts_base,
                temperature=args.temperature,
                attempt=attempt + 1,
            )
            controlled = repaired.controlled_english
            _print_draft(f"repair {attempt + 1}", controlled)
            continue

        # Try schema validation using the parse result.
        try:
            _ = StrategySpec.model_validate(result.spec_dict)
        except ValidationError as e:
            last_error = "Schema validation error:\n" + _format_validation_error(e)
            if attempt >= int(args.max_repairs):
                raise

            repaired = repair_controlled_english(
                notes=notes,
                previous_controlled_english=controlled,
                errors=last_error,
                provider=provider,
                model=args.model,
                strategy_name=args.name,
                default_entry_tf=args.entry_tf,
                artifacts_base_dir=artifacts_base,
                temperature=args.temperature,
                attempt=attempt + 1,
            )
            controlled = repaired.controlled_english
            _print_draft(f"repair {attempt + 1}", controlled)
            continue

        break

    if result is None:
        raise SystemExit(last_error or "Failed to parse/validate controlled-English")

    def _eligible_clarifications_for_ai(clarifications: list[Any]) -> list[Any]:
        eligible: list[Any] = []
        for c in clarifications:
            key = getattr(c, "key", "")
            default = getattr(c, "default", None)

            # Never attempt to guess symbols/market identifiers.
            if isinstance(key, str) and key.startswith("market."):
                continue

            # Only attempt clarifications with deterministic defaults.
            if default is None or (isinstance(default, str) and not default.strip()):
                continue

            eligible.append(c)
        return eligible

    def _format_clarifications(clarifications: list[Any]) -> str:
        lines: list[str] = []
        for c in clarifications:
            key = getattr(c, "key", "")
            q = getattr(c, "question", "")
            options = getattr(c, "options", None)
            default = getattr(c, "default", None)
            line = f"- {key}: {q}"
            if options:
                line += f" (options={options})"
            if default:
                line += f" (default={default})"
            lines.append(line)
        return "\n".join(lines)

    # Optional clarification-repair loop: ask the LLM to make defaults explicit
    # in controlled-English so the deterministic parser produces fewer questions.
    if args.auto_resolve_clarifications and result.clarifications:
        clarify_attempts = 0
        while result.clarifications and clarify_attempts < int(args.max_clarify_repairs):
            eligible = _eligible_clarifications_for_ai(result.clarifications)
            if not eligible:
                break

            clarify_attempts += 1
            clarifications_text = _format_clarifications(eligible)

            repaired = repair_controlled_english_for_clarifications(
                notes=notes,
                previous_controlled_english=controlled,
                clarifications=clarifications_text,
                provider=provider,
                model=args.model,
                strategy_name=args.name,
                default_entry_tf=args.entry_tf,
                artifacts_base_dir=artifacts_base,
                temperature=args.temperature,
                attempt=clarify_attempts,
            )
            controlled = repaired.controlled_english
            _print_draft(f"clarify repair {clarify_attempts}", controlled)
            result = parse_english_strategy(
                controlled,
                name=args.name,
                default_entry_tf=args.entry_tf,
                mode=args.parse_mode,
            )

            # If schema validation is now broken, fall back into the existing repair loop.
            try:
                _ = StrategySpec.model_validate(result.spec_dict)
            except ValidationError as e:
                last_error = "Schema validation error:\n" + _format_validation_error(e)
                # Reuse the standard repair path, consuming from remaining max-repairs.
                remaining = max(0, int(args.max_repairs))
                for attempt in range(0, remaining + 1):
                    if attempt >= remaining:
                        raise
                    repaired2 = repair_controlled_english(
                        notes=notes,
                        previous_controlled_english=controlled,
                        errors=last_error,
                        provider=provider,
                        model=args.model,
                        strategy_name=args.name,
                        default_entry_tf=args.entry_tf,
                        artifacts_base_dir=artifacts_base,
                        temperature=args.temperature,
                        attempt=attempt + 1,
                    )
                    controlled = repaired2.controlled_english
                    _print_draft(f"repair {attempt + 1}", controlled)
                    result = parse_english_strategy(
                        controlled,
                        name=args.name,
                        default_entry_tf=args.entry_tf,
                        mode=args.parse_mode,
                    )
                    _ = StrategySpec.model_validate(result.spec_dict)
                    break

    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    spec_dict = result.spec_dict
    answers: dict[str, str] = {}
    if result.clarifications:
        if args.non_interactive:
            print("\nCannot proceed without clarification:")
            for c in result.clarifications:
                print(f"  - {c.key}: {c.question}")
            return 2

        print("\nI need a few clarifications to make this strategy unambiguous.")
        for c in result.clarifications:
            answers[c.key] = _prompt_one(c.key, c.question, c.options, c.default)

        spec_dict = apply_clarifications(spec_dict, answers)

    # Final validation via Pydantic (hard fail on inconsistent output)
    spec = StrategySpec.model_validate(spec_dict)

    out_spec = Path(args.out_spec) if args.out_spec else (repo_root / "research_specs" / f"{args.name}.yml")
    out_spec.parent.mkdir(parents=True, exist_ok=True)
    out_spec.write_text(yaml.safe_dump(spec.model_dump(), sort_keys=False))
    print(f"\n✓ Wrote spec: {out_spec}")

    if draft.bundle_dir is not None:
        print(f"✓ AI artifacts: {draft.bundle_dir}")

    if args.compile:
        compiler = StrategyCompiler(repo_root=repo_root)
        built_path = compiler.compile_to_folder(spec)
        print(f"✓ Built strategy folder: {built_path}")
        print(f"  Config: {built_path / 'config.yml'}")
        print(f"  Strategy: {built_path / 'strategy.py'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
