import subprocess
import sys


def test_state_store_registry_js():
    cmd = [
        "node",
        "scripts/js_tests/test_state_store.js",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
    assert proc.returncode == 0, f"JS registry tests failed: {proc.stderr}"
