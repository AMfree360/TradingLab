#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.registry import ExperimentRegistry


def _format_passed(passed_val) -> str:
    if passed_val is None:
        return "?"
    if passed_val == 1 or passed_val is True:
        return "PASS"
    return "FAIL"


def cmd_catalog(args: argparse.Namespace) -> int:
    reg = ExperimentRegistry(args.db)
    rows = reg.list_latest_runs(strategy=args.strategy, phase=args.phase)
    if not rows:
        print("No runs found")
        return 0

    # Optional status filtering
    def _status_match(r) -> bool:
        if args.status == "any":
            return True
        passed = r.get("passed")
        if args.status == "pass":
            return passed == 1
        if args.status == "fail":
            return passed == 0
        if args.status == "unknown":
            return passed is None
        return True

    rows = [r for r in rows if _status_match(r)]
    if not rows:
        print("No runs found")
        return 0

    if args.json:
        import json

        print(json.dumps(rows, indent=2, default=str))
        return 0

    # Group by strategy for a clearer “passed vs failed strategies” view.
    by_strategy = {}
    for r in rows:
        by_strategy.setdefault(r["strategy"], []).append(r)

    for strategy, recs in sorted(by_strategy.items()):
        print(strategy)
        for r in sorted(recs, key=lambda x: x.get("phase") or ""):
            passed_str = _format_passed(r.get("passed"))

            outcome = {}
            try:
                import json as _json

                if r.get("outcome_json"):
                    outcome = _json.loads(r["outcome_json"]) or {}
            except Exception:
                outcome = {}

            failure_reasons = outcome.get("failure_reasons") or []
            failure_summary = "" if not failure_reasons else f" | reasons={'; '.join(map(str, failure_reasons[:3]))}"
            if failure_reasons and len(failure_reasons) > 3:
                failure_summary += f" (+{len(failure_reasons) - 3} more)"

            print(
                f"  {r.get('phase')} {passed_str}"
                f" | when={r.get('created_at')}"
                f" | dataset={r.get('dataset_manifest_hash') or 'n/a'}"
                f" | report={r.get('report_path') or 'n/a'}"
                f"{failure_summary}"
            )
        print("")

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    reg = ExperimentRegistry(args.db)
    rows = reg.list_runs(strategy=args.strategy, phase=args.phase, limit=args.limit)
    if not rows:
        print("No runs found")
        return 0

    for r in rows:
        passed = r.get("passed")
        passed_str = "?" if passed is None else ("PASS" if passed == 1 else "FAIL")
        print(
            f"#{r['id']} {r['created_at']} {passed_str} "
            f"{r['strategy']} {r['phase']} dataset={r.get('dataset_manifest_hash') or 'n/a'} "
            f"report={r.get('report_path') or 'n/a'}"
        )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    reg = ExperimentRegistry(args.db)
    out = reg.compare_runs(args.a, args.b)

    a = out["a"]
    b = out["b"]
    print(f"A: #{a['id']} {a['created_at']} {a['strategy']} {a['phase']}")
    print(f"B: #{b['id']} {b['created_at']} {b['strategy']} {b['phase']}")

    if a.get("dataset_manifest_hash") != b.get("dataset_manifest_hash"):
        print(f"dataset_manifest_hash differs: {a.get('dataset_manifest_hash')} vs {b.get('dataset_manifest_hash')}")

    if a.get("config_hash") != b.get("config_hash"):
        print(f"config_hash differs: {a.get('config_hash')} vs {b.get('config_hash')}")

    diffs = out["metric_diffs"]
    if not diffs:
        print("No metric diffs recorded")
        return 0

    print("Metric diffs:")
    for k, v in diffs.items():
        print(f"  {k}: {v.get('a')} -> {v.get('b')}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Experiment registry utilities")
    p.add_argument("--db", type=str, default=".experiments/registry.sqlite", help="Path to registry sqlite")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List recent runs")
    p_list.add_argument("--strategy", type=str)
    p_list.add_argument("--phase", type=str)
    p_list.add_argument("--limit", type=int, default=50)
    p_list.set_defaults(func=cmd_list)

    p_cmp = sub.add_parser("compare", help="Compare two runs by id")
    p_cmp.add_argument("--a", type=int, required=True)
    p_cmp.add_argument("--b", type=int, required=True)
    p_cmp.set_defaults(func=cmd_compare)

    p_cat = sub.add_parser("catalog", help="List latest status per strategy+phase")
    p_cat.add_argument("--strategy", type=str)
    p_cat.add_argument("--phase", type=str)
    p_cat.add_argument("--status", type=str, default="any", choices=["any", "pass", "fail", "unknown"])
    p_cat.add_argument("--json", action="store_true", help="Output raw JSON rows")
    p_cat.set_defaults(func=cmd_catalog)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
