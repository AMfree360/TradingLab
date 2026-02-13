#!/usr/bin/env python3

"""Build a TradingLab strategy from a human-readable research spec.

Example:
  python3 scripts/build_strategy_from_spec.py --spec research_specs/my_strategy.yml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.compiler import StrategyCompiler


def main() -> int:
    p = argparse.ArgumentParser(description="Compile a research-layer StrategySpec into a TradingLab strategy folder")
    p.add_argument("--spec", required=True, help="Path to strategy spec YAML")
    p.add_argument("--out", help="Optional output dir (default: strategies/<name>/)")
    p.add_argument("--print-summary", action="store_true", help="Print spec summary JSON")
    args = p.parse_args()

    repo_root = Path(__file__).parent.parent
    compiler = StrategyCompiler(repo_root=repo_root)

    spec = compiler.load_spec(Path(args.spec))

    if args.print_summary:
        print(json.dumps(spec.summary(), indent=2, sort_keys=True))

    out_dir = Path(args.out) if args.out else None
    built_path = compiler.compile_to_folder(spec, out_dir=out_dir)
    print(f"✓ Built strategy folder: {built_path}")
    print(f"  Config: {built_path / 'config.yml'}")
    print(f"  Strategy: {built_path / 'strategy.py'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
