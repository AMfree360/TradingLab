#!/usr/bin/env python3

from __future__ import annotations

import argparse
import socket
import sys
import webbrowser
from pathlib import Path


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the TradingLab local web launcher.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=0, help="Bind port (0 = pick a free port)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    host = str(args.host)
    port = int(args.port) if int(args.port) != 0 else _pick_free_port(host)

    url = f"http://{host}:{port}/"
    if not args.no_browser:
        try:
            # open launcher home (dashboard) in a browser tab; do not auto-open Builder V3
            webbrowser.open_new_tab(url)
        except Exception:
            pass

    import uvicorn

    uvicorn.run(
        "gui_launcher.app:app",
        host=host,
        port=port,
        reload=bool(args.reload),
        log_level="info",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
