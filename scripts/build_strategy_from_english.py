#!/usr/bin/env python3

"""Build a TradingLab strategy from controlled plain English.

This is an offline (non-ML) parser that supports a small, explicit template.

Examples of input text:
  - Go long when EMA(20) crosses above EMA(50) on 4h
  - Enter short when close < SMA(200) on 1h
  - Stop ATR(14) * 3
  - Take profit 2R 50%, 3R 50%
  - Partial exit 1.5R 80%
  - Trailing EMA 21 after 0.5R
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    p = argparse.ArgumentParser(description="Parse controlled plain English into a TradingLab research spec")
    p.add_argument("--name", required=True, help="Strategy name (used for folder + spec)")
    p.add_argument("--in", dest="in_path", help="Path to a .txt file (default: read from stdin)")
    p.add_argument("--entry-tf", help="Default entry timeframe if not specified in text (e.g., 1h)")
    p.add_argument(
        "--parse-mode",
        choices=["strict", "lenient"],
        default="strict",
        help=(
            "Parser mode. 'strict' requires explicit controlled-English templates; "
            "'lenient' broadens which lines count as entry rules without guessing values."
        ),
    )
    p.add_argument("--out-spec", help="Where to write the YAML spec (default: research_specs/<name>.yml)")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing spec file (default: refuse if it already exists).",
    )
    p.add_argument("--compile", action="store_true", help="Compile the spec into strategies/<name>/")
    p.add_argument("--non-interactive", action="store_true", help="Fail if clarifications are required")
    p.add_argument(
        "--answers-json",
        help=(
            "Optional path to a JSON file containing clarification answers (key -> value). "
            "If provided, the script applies these answers instead of prompting."
        ),
    )
    args = p.parse_args()

    repo_root = Path(__file__).parent.parent

    raw = _read_text(args.in_path)
    result = parse_english_strategy(raw, name=args.name, default_entry_tf=args.entry_tf, mode=args.parse_mode)

    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    spec_dict = result.spec_dict
    answers: dict[str, str] = {}
    if result.clarifications:
        if args.answers_json:
            raw_answers = json.loads(Path(args.answers_json).read_text())
            if not isinstance(raw_answers, dict):
                raise ValueError("answers-json must contain a JSON object mapping clarification keys to values")
            provided = {str(k): str(v) for k, v in raw_answers.items() if v is not None}

            # Fill required answers deterministically:
            # - If provided in answers-json, use it.
            # - Else if the clarification has a default, use the default.
            # - Else fail clearly (GUI should have asked the user).
            missing_required: list[str] = []
            for c in result.clarifications:
                if c.key in provided and str(provided[c.key]).strip():
                    answers[c.key] = str(provided[c.key]).strip()
                    continue
                if c.default is not None and str(c.default).strip():
                    answers[c.key] = str(c.default).strip()
                    continue
                missing_required.append(f"{c.key}: {c.question}")

            if missing_required:
                print("\nCannot proceed without clarification (missing required answers in --answers-json):")
                for line in missing_required:
                    print(f"  - {line}")
                return 2

            spec_dict = apply_clarifications(spec_dict, answers)

        elif args.non_interactive:
            print("\nCannot proceed without clarification:")
            for c in result.clarifications:
                print(f"  - {c.key}: {c.question}")
            return 2

        # Interactive fallback
        if args.answers_json:
            pass
        else:
            print("\nI need a few clarifications to make this strategy unambiguous.")
            for c in result.clarifications:
                answers[c.key] = _prompt_one(c.key, c.question, c.options, c.default)

            spec_dict = apply_clarifications(spec_dict, answers)

    if "long" not in spec_dict and "short" not in spec_dict:
        print(
            "\nCannot proceed: I couldn't extract a LONG or SHORT entry rule from the English text. "
            "Try wording entries like: 'Go long when RSI is above 70' and/or 'Go short when RSI is below 30'."
        )
        return 2

    # Validate via Pydantic (hard fail on inconsistent output)
    spec = StrategySpec.model_validate(spec_dict)

    out_spec = Path(args.out_spec) if args.out_spec else (repo_root / "research_specs" / f"{args.name}.yml")
    out_spec.parent.mkdir(parents=True, exist_ok=True)

    existing: Path | None = None
    if args.out_spec:
        existing = out_spec if out_spec.exists() else None
    else:
        for ext in (".yml", ".yaml"):
            pth = repo_root / "research_specs" / f"{args.name}{ext}"
            if pth.exists():
                existing = pth
                break

    if existing is not None and not args.overwrite:
        print(f"\nRefusing to overwrite existing spec: {existing}\nUse --overwrite if this is intentional.")
        return 2

    out_spec.write_text(yaml.safe_dump(spec.model_dump(), sort_keys=False))
    print(f"\n✓ Wrote spec: {out_spec}")

    if args.compile:
        compiler = StrategyCompiler(repo_root=repo_root)
        built_path = compiler.compile_to_folder(spec)
        print(f"✓ Built strategy folder: {built_path}")
        print(f"  Config: {built_path / 'config.yml'}")
        print(f"  Strategy: {built_path / 'strategy.py'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
