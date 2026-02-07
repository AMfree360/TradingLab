from __future__ import annotations

from pathlib import Path

import pytest


LOOKAHEAD_PATTERNS = [
    "shift(-1)",
    "shift(-2)",
    "shift(-3)",
    "get_loc(idx) + 1",
]


@pytest.mark.parametrize("root", [Path(__file__).resolve().parents[1] / "strategies"])
def test_strategies_do_not_use_obvious_lookahead_patterns(root: Path) -> None:
    offenders: list[tuple[Path, str]] = []

    for path in root.rglob("*.py"):
        # Skip caches / compiled artifacts
        if "__pycache__" in path.parts:
            continue

        text = path.read_text(encoding="utf-8")
        for pat in LOOKAHEAD_PATTERNS:
            if pat in text:
                offenders.append((path, pat))

    if offenders:
        details = "\n".join(f"- {p}: contains {pat!r}" for p, pat in offenders)
        raise AssertionError(
            "Obvious forward-looking (lookahead) patterns detected in strategies.\n"
            "These are disallowed because they can leak future information into signals.\n\n"
            f"{details}\n"
        )
