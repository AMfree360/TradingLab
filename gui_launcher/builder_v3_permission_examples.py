"""Example permission-checker callables for Builder V3 save flows.

Drop one of these into `gui_launcher.app` at startup to enable custom permission
behavior for `/api/builder_v3/save` operations. For production, implement a real
checker that consults your auth/ACL system (e.g., user roles, groups, or an
external policy service).

Usage:

    from gui_launcher import app as launcher_app
    from gui_launcher.builder_v3_permission_examples import example_sync_checker

    # attach the checker
    launcher_app.state.builder_v3_permission_checker = example_sync_checker

The checker signature is `(request, spec_name, op)` and should return True to
permit, False to deny. The checker may be async.
"""
from typing import Any
from fastapi import Request
import asyncio


def example_sync_checker(request: Request, spec_name: str, op: str) -> bool:
    """Simple synchronous checker example.

    Rules:
      - Allow 'create' for any request
      - Allow 'overwrite' only from local host
      - Deny 'rename' by default
    """
    client_ip = "unknown"
    try:
        client_ip = request.client.host or "unknown"
    except Exception:
        pass

    if op == "create":
        return True
    if op == "overwrite":
        return client_ip in ("127.0.0.1", "::1", "localhost")
    if op == "rename":
        return False
    return False


async def example_async_checker(request: Request, spec_name: str, op: str) -> bool:
    """Async example that simulates a remote permission call (sleep).

    In real deployments, call an authz service here and return its decision.
    """
    await asyncio.sleep(0.01)
    # For demo, allow everything except rename
    return op != "rename"
