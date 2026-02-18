from __future__ import annotations

import json
import io
import platform
import re
import sys
import time
import uuid
import zipfile
from pathlib import Path
import shutil
from functools import lru_cache
from typing import Any, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from gui_launcher.bundles import (
    create_run_bundle,
    write_bundle_command,
    write_bundle_inputs,
    write_bundle_logs,
    write_bundle_meta,
)
from gui_launcher.runner import JobRunner

# Repo imports
from research.nl_parser import parse_english_strategy
from research.spec import StrategySpec


REPO_ROOT = Path(__file__).parent.parent
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@lru_cache(maxsize=1)
def _get_master_calendar_defaults() -> dict[str, Any]:
    """Load canonical calendar defaults from config/master_filters.yml.

    Normalized return shape:
      {"allowed_days": [...], "sessions": {"Asia": {"enabled","start","end"}, ...}}
    """

    import yaml

    defaults: dict[str, Any] = {
        "allowed_days": [0, 1, 2, 3, 4],
        "sessions": {
            "Asia": {"enabled": True, "start": "23:00", "end": "08:00"},
            "London": {"enabled": True, "start": "07:00", "end": "16:00"},
            "NewYork": {"enabled": True, "start": "13:00", "end": "21:00"},
        },
    }

    path = (REPO_ROOT / "config" / "master_filters.yml").resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
        cal = (((raw.get("master_filters") or {}).get("calendar_filters")) or {}) if isinstance(raw, dict) else {}

        day_cfg = cal.get("day_of_week") if isinstance(cal, dict) else None
        allowed_days = day_cfg.get("allowed_days") if isinstance(day_cfg, dict) else None
        if isinstance(allowed_days, list) and allowed_days:
            cleaned: list[int] = []
            for d in allowed_days:
                try:
                    di = int(d)
                except Exception:
                    continue
                if 0 <= di <= 6 and di not in cleaned:
                    cleaned.append(di)
            if cleaned:
                defaults["allowed_days"] = cleaned

        ts = cal.get("trading_sessions") if isinstance(cal, dict) else None
        if isinstance(ts, dict):
            for k in ["Asia", "London", "NewYork"]:
                v = ts.get(k)
                if not isinstance(v, dict):
                    continue
                if "enabled" in v:
                    defaults["sessions"][k]["enabled"] = bool(v.get("enabled"))
                if v.get("start"):
                    defaults["sessions"][k]["start"] = str(v.get("start"))
                if v.get("end"):
                    defaults["sessions"][k]["end"] = str(v.get("end"))
    except Exception:
        pass

    return defaults


@lru_cache(maxsize=1)
def _get_master_trade_management_defaults() -> dict[str, Any]:
    """Load canonical trade-management defaults from config/master_trade_management.yml.

    Normalized return shape (UI-focused):
      {"take_profit": {...}, "partial_exit": {...}, "trailing_stop": {...}}
    """

    import yaml

    defaults: dict[str, Any] = {
        "take_profit": {"enabled": True, "target_r": 3.0, "exit_pct": 100.0},
        "partial_exit": {"enabled": False, "level_r": 1.5, "exit_pct": 50.0},
        "trailing_stop": {"enabled": False, "length": 21, "activation_r": 0.5, "stepped": True, "min_move_pips": 7.0},
    }

    path = (REPO_ROOT / "config" / "master_trade_management.yml").resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
        mtm = (raw.get("master_trade_management") or {}) if isinstance(raw, dict) else {}

        tp = mtm.get("take_profit") if isinstance(mtm, dict) else None
        if isinstance(tp, dict):
            if "enabled" in tp:
                defaults["take_profit"]["enabled"] = bool(tp.get("enabled"))
            levels = tp.get("levels")
            if isinstance(levels, list) and levels and isinstance(levels[0], dict):
                lvl0 = levels[0]
                if "target_r" in lvl0:
                    try:
                        defaults["take_profit"]["target_r"] = float(lvl0.get("target_r"))
                    except Exception:
                        pass
                if "exit_pct" in lvl0:
                    try:
                        defaults["take_profit"]["exit_pct"] = float(lvl0.get("exit_pct"))
                    except Exception:
                        pass

        pe = mtm.get("partial_exit") if isinstance(mtm, dict) else None
        if isinstance(pe, dict):
            if "enabled" in pe:
                defaults["partial_exit"]["enabled"] = bool(pe.get("enabled"))
            levels = pe.get("levels")
            if isinstance(levels, list) and levels and isinstance(levels[0], dict):
                lvl0 = levels[0]
                if "level_r" in lvl0:
                    try:
                        defaults["partial_exit"]["level_r"] = float(lvl0.get("level_r"))
                    except Exception:
                        pass
                if "exit_pct" in lvl0:
                    try:
                        defaults["partial_exit"]["exit_pct"] = float(lvl0.get("exit_pct"))
                    except Exception:
                        pass

        ts = mtm.get("trailing_stop") if isinstance(mtm, dict) else None
        if isinstance(ts, dict):
            if "enabled" in ts:
                defaults["trailing_stop"]["enabled"] = bool(ts.get("enabled"))
            if "length" in ts:
                try:
                    defaults["trailing_stop"]["length"] = int(ts.get("length"))
                except Exception:
                    pass
            if "activation_r" in ts:
                try:
                    defaults["trailing_stop"]["activation_r"] = float(ts.get("activation_r"))
                except Exception:
                    pass
            if "stepped" in ts:
                defaults["trailing_stop"]["stepped"] = bool(ts.get("stepped"))
            if "min_move_pips" in ts:
                try:
                    defaults["trailing_stop"]["min_move_pips"] = float(ts.get("min_move_pips"))
                except Exception:
                    pass
    except Exception:
        pass

    return defaults


def _prune_empty(obj: object) -> object:
    """Recursively drop empty containers and None values.

    Keeps falsy-but-meaningful values like 0 and False.
    """

    if obj is None:
        return None

    if isinstance(obj, dict):
        out: dict[object, object] = {}
        for k, v in obj.items():
            pv = _prune_empty(v)
            if pv is None:
                continue
            if isinstance(pv, (dict, list)) and len(pv) == 0:
                continue
            out[k] = pv
        return out

    if isinstance(obj, list):
        out_list: list[object] = []
        for v in obj:
            pv = _prune_empty(v)
            if pv is None:
                continue
            if isinstance(pv, (dict, list)) and len(pv) == 0:
                continue
            out_list.append(pv)
        return out_list

    return obj


def _spec_to_yaml_compact(spec: StrategySpec) -> str:
    import yaml

    # Exclude default values to avoid dumping UI-unseen defaults (empty filters/news/volume sections,
    # default risk_per_trade_pct, etc). Then prune any remaining empty containers.
    data = spec.model_dump(exclude_none=True, exclude_defaults=True)
    data = _prune_empty(data)
    return yaml.safe_dump(data, sort_keys=False)


def _write_validated_spec_to_repo(*, spec: StrategySpec, old_spec_name: str) -> Path:
    """Write spec to research_specs and handle safe rename. Returns new path."""

    new_name = str(getattr(spec, "name", "") or "").strip()
    if not _is_safe_strategy_name(new_name):
        raise ValueError("Invalid name")

    specs_dir = REPO_ROOT / "research_specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    # Prefer writing to an existing file if it already exists (.yml or .yaml),
    # otherwise default to .yml.
    new_path = _safe_spec_path(new_name)
    if new_path is None:
        raise ValueError("Invalid name")

    # Safe-guard: only write under research_specs
    base = specs_dir.resolve()
    if not str(new_path).startswith(str(base)):
        raise ValueError("Unsafe path")

    # Do not allow creating/renaming into an existing spec (unless it's the same file).
    old_path = _safe_spec_path(str(old_spec_name)) if _is_safe_strategy_name(str(old_spec_name)) else None
    if new_path.exists() and (old_path is None or new_path.resolve() != old_path.resolve()):
        raise ValueError(f"Strategy name already exists: {new_name}")

    new_path.write_text(_spec_to_yaml_compact(spec), encoding="utf-8")

    # If user renamed the spec, remove the old file (best-effort)
    if new_name != str(old_spec_name) and _is_safe_strategy_name(str(old_spec_name)):
        old_path = _safe_spec_path(str(old_spec_name))
        if old_path is not None and old_path.exists():
            try:
                old_path.unlink()
            except Exception:
                pass

    return new_path

import os

app = FastAPI(title="TradingLab Launcher", docs_url=None, redoc_url=None)

# Runtime feature flags and security settings (can be set via env vars)
app.state.builder_v3_csrf_secret = os.environ.get("BUILDER_V3_CSRF_SECRET")
app.state.builder_v3_csrf_enforce = str(os.environ.get("BUILDER_V3_CSRF_ENFORCE", "false")).lower() in ("1", "true", "yes")
app.state.builder_v3_rate_limit_enabled = str(os.environ.get("BUILDER_V3_RATE_LIMIT_ENABLED", "false")).lower() in ("1", "true", "yes")
# per-key default rate (requests) and window (seconds)
try:
    app.state.builder_v3_rate_limit = int(os.environ.get("BUILDER_V3_RATE_LIMIT", "60"))
except Exception:
    app.state.builder_v3_rate_limit = 60
try:
    app.state.builder_v3_rate_window = int(os.environ.get("BUILDER_V3_RATE_WINDOW", "60"))
except Exception:
    app.state.builder_v3_rate_window = 60

# Save semantics flags (Phase 4)
# When True, `/api/builder_v3/save` permits clients to supply `force: true` to overwrite existing specs.
app.state.builder_v3_allow_overwrite = str(os.environ.get("BUILDER_V3_ALLOW_OVERWRITE", "false")).lower() in ("1", "true", "yes")
# When True, name conflicts return a strict 409 conflict instead of falling back to draft JSON.
app.state.builder_v3_strict_save_conflict = str(os.environ.get("BUILDER_V3_STRICT_SAVE_CONFLICT", "false")).lower() in ("1", "true", "yes")
# Optional permission checker hook. If set to a callable, it will be called as
# `res = checker(request, spec_name, op)` where `op` is one of "create","overwrite","rename".
# The checker may be synchronous (returning bool) or async (awaitable). Default: None.
app.state.builder_v3_permission_checker = None

# Static + reports
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
app.mount("/reports", StaticFiles(directory=str(REPO_ROOT / "reports")), name="reports")

# Builder V3 routes removed in branch `remove/builder-v3`.
# Previously the app included `gui_launcher.builder_v3_routes` here.
# Keeping flags on `app.state` for backward-compatibility if needed.

runner = JobRunner()

# Draft state stored in memory (MVP). Keyed by draft_id.
DRAFTS: dict[str, dict[str, Any]] = {}

# Minimal job metadata stored in memory (MVP). Keyed by job_id.
JOB_META: dict[str, dict[str, Any]] = {}

# Minimal request log (for diagnosing local UI issues). Not persisted.
REQUEST_LOG: list[dict[str, Any]] = []
REQUEST_LOG_MAX = 50


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    # Attach a per-request id for structured logging and correlation
    try:
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
    except Exception:
        request.state.request_id = None
    t0 = time.time()
    try:
        response = await call_next(request)
        status = getattr(response, "status_code", None)
    except Exception:
        status = 500
        raise
    finally:
        try:
            REQUEST_LOG.append(
                {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "ms": int((time.time() - t0) * 1000),
                    "method": request.method,
                    "path": request.url.path,
                    "query": request.url.query,
                    "status": status,
                    "request_id": getattr(request.state, "request_id", None),
                }
            )
            if len(REQUEST_LOG) > REQUEST_LOG_MAX:
                del REQUEST_LOG[: len(REQUEST_LOG) - REQUEST_LOG_MAX]
        except Exception:
            pass

    # Expose request id to callers for easier correlation in logs
    try:
        rid = getattr(request.state, "request_id", None)
        if rid is not None:
            response.headers["X-Request-ID"] = rid
    except Exception:
        pass

    return response


@app.get("/debug/requests", response_class=HTMLResponse)
def debug_requests(request: Request):
    rows = list(reversed(REQUEST_LOG))
    html = [
        "<h1>Debug: Recent Requests</h1>",
        "<div style='opacity:0.85;margin-bottom:10px;'>Newest first. This page is local-only debug for the launcher.</div>",
        "<div style='font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;'>",
    ]
    for r in rows[:REQUEST_LOG_MAX]:
        q = ("?" + r.get("query")) if r.get("query") else ""
        html.append(
            f"<div>{r.get('ts')}  {r.get('ms')}ms  {r.get('status')}  {r.get('method')}  {r.get('path')}{q}</div>"
        )
    html.append("</div>")
    return HTMLResponse("\n".join(html))


@app.get("/builder-v3")
def builder_v3_page(request: Request):
    """Builder V3 removed: redirect to launcher home."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/")




def _is_safe_strategy_name(name: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9_\-]+$", (name or "").strip()))


def _suggest_next_version_name(repo_root: Path, requested_name: str) -> str:
    """Suggest a versioned name when a spec already exists.

    Examples:
      - pin_bar -> pin_bar_v2
      - pin_bar_v2 -> pin_bar_v3
    """

    name = str(requested_name or "").strip()
    if not _is_safe_strategy_name(name):
        return name

    p0 = _safe_spec_path(name)
    if p0 is None:
        return name
    if not p0.exists():
        return name

    m = re.match(r"^(?P<base>.+?)(?:_v(?P<n>\d+))?$", name)
    base = (m.group("base") if m else name).strip() if m else name
    n0 = int(m.group("n")) if (m and m.group("n")) else 1

    # Avoid infinite loops; 2..9999 is plenty.
    for n in range(max(2, n0 + 1), 10000):
        candidate = f"{base}_v{n}"
        if not _is_safe_strategy_name(candidate):
            continue
        pc = _safe_spec_path(candidate)
        if pc is None:
            continue
        if not pc.exists():
            return candidate

    return name


def _available_split_policy_names(repo_root: Path) -> list[str]:
    try:
        from config.data_splits import load_data_splits

        doc = load_data_splits(repo_root / "config" / "data_splits.yml")
        policies = doc.get("policies") or {}
        if isinstance(policies, dict):
            return sorted([str(k) for k in policies.keys()])
    except Exception:
        pass
    return []


def _parse_pagination(*, page: int | str | None, per_page: int | str | None, max_per_page: int = 200) -> tuple[int, int]:
    try:
        p = int(str(page or "1").strip())
    except Exception:
        p = 1
    try:
        pp = int(str(per_page or "50").strip())
    except Exception:
        pp = 50

    if p < 1:
        p = 1
    if pp < 1:
        pp = 1
    if pp > max_per_page:
        pp = max_per_page
    return p, pp


def _paginate_list(items: list[Any], *, page: int, per_page: int) -> tuple[list[Any], int]:
    total = len(items)
    start = (page - 1) * per_page
    if start < 0:
        start = 0
    end = start + per_page
    return items[start:end], total


def _list_strategy_specs(repo_root: Path) -> list[dict[str, Any]]:
    specs_dir = repo_root / "research_specs"
    rows: list[dict[str, Any]] = []
    if not specs_dir.exists():
        return rows
    for p in specs_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".yml", ".yaml"}:
            continue
        name = p.stem
        if not _is_safe_strategy_name(name):
            continue

        meta = _read_guided_meta(name)
        guided_template = None
        try:
            guided_template = str((meta or {}).get("template") or "").strip().lower() or None
        except Exception:
            guided_template = None
        rows.append(
            {
                "name": name,
                "path": str(p.relative_to(repo_root)),
                "has_guided_meta": bool(meta),
                "guided_template": guided_template,
                "has_v2_guided_meta": guided_template == "builder_v2",
            }
        )
    rows.sort(key=lambda r: r["name"].lower())
    return rows


def _list_code_strategies(repo_root: Path) -> list[dict[str, str]]:
    root = repo_root / "strategies"
    rows: list[dict[str, str]] = []
    if not root.exists():
        return rows

    # Top-level strategy modules: strategies/<name>.py
    for p in root.iterdir():
        if p.is_file() and p.suffix == ".py" and p.name != "__init__.py":
            name = p.stem
            if _is_safe_strategy_name(name):
                rows.append({"name": name, "path": str(p.relative_to(repo_root))})

    # Package strategies: strategies/<name>/...
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("__"):
            continue
        name = p.name
        if not _is_safe_strategy_name(name):
            continue
        # Heuristic: treat as strategy if it contains an __init__.py or <name>.py
        if (p / "__init__.py").exists() or (p / f"{name}.py").exists():
            rows.append({"name": name, "path": str(p.relative_to(repo_root))})

    # De-dupe by name (prefer packages over modules if both exist)
    deduped: dict[str, dict[str, str]] = {}
    for r in rows:
        n = r["name"]
        prev = deduped.get(n)
        if prev is None or ("/" in r["path"] and "/" not in prev["path"]):
            deduped[n] = r
    out = list(deduped.values())
    out.sort(key=lambda r: r["name"].lower())
    return out


def _guided_instruments() -> dict[str, list[str]]:
    # Static lists (MVP). Keep this small and human-friendly.
    return {
        "crypto": [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "BNBUSDT",
            "XRPUSDT",
            "ADAUSDT",
            "DOGEUSDT",
        ],
        "fx": [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "USDCHF",
            "USDCAD",
            "AUDUSD",
            "NZDUSD",
        ],
        "futures": [
            # Micros
            "MES",
            "MNQ",
            "MYM",
            "M2K",
            "MCL",
            "MGC",
            # Minis
            "ES",
            "NQ",
            "YM",
            "RTY",
            "CL",
            "GC",
            "SI",
            "ZB",
        ],
    }


def _expr_from_condition(form_prefix: str, form: dict[str, Any]) -> Optional[dict[str, str]]:
    """Build a single ConditionSpec-like dict from flat form values.

    Returns None if the condition row is disabled/blank.
    Raises ValueError for partially-filled invalid rows.
    """

    ctype = str(form.get(f"{form_prefix}_type") or "").strip()
    if not ctype or ctype == "none":
        return None

    tf = str(form.get(f"{form_prefix}_tf") or "").strip()
    if not tf:
        raise ValueError(f"{form_prefix}: timeframe is required")

    if ctype == "rsi":
        length = int(str(form.get(f"{form_prefix}_rsi_len") or "14").strip())
        op = str(form.get(f"{form_prefix}_op") or "<").strip()
        level = float(str(form.get(f"{form_prefix}_level") or "").strip())
        if op not in {"<", ">", "<=", ">="}:
            raise ValueError(f"{form_prefix}: invalid operator")
        expr = f"rsi(close, {length}) {op} {level}"
        return {"tf": tf, "expr": expr}

    if ctype == "ema_cross":
        fast = int(str(form.get(f"{form_prefix}_fast") or "").strip())
        slow = int(str(form.get(f"{form_prefix}_slow") or "").strip())
        direction = str(form.get(f"{form_prefix}_direction") or "above").strip().lower()
        fn = "crosses_above" if direction in {"above", "over"} else "crosses_below"
        expr = f"{fn}(ema(close, {fast}), ema(close, {slow}))"
        return {"tf": tf, "expr": expr}

    if ctype == "close_vs_ma":
        ma = str(form.get(f"{form_prefix}_ma") or "ema").strip().lower()
        length = int(str(form.get(f"{form_prefix}_ma_len") or "").strip())
        op = str(form.get(f"{form_prefix}_op") or ">").strip()
        if ma not in {"ema", "sma"}:
            raise ValueError(f"{form_prefix}: invalid MA type")
        if op not in {"<", ">", "<=", ">="}:
            raise ValueError(f"{form_prefix}: invalid operator")
        expr = f"close {op} {ma}(close, {length})"
        return {"tf": tf, "expr": expr}

    if ctype == "donchian":
        length = int(str(form.get(f"{form_prefix}_don_len") or "").strip())
        direction = str(form.get(f"{form_prefix}_direction") or "breakout").strip().lower()
        if direction in {"breakout", "above", "over"}:
            expr = f"crosses_above(close, shift(donchian_high({length}), 1))"
        else:
            expr = f"crosses_below(close, shift(donchian_low({length}), 1))"
        return {"tf": tf, "expr": expr}

    raise ValueError(f"{form_prefix}: unknown condition type")


def _guided_defaults(template_key: str) -> dict[str, Any]:
    t = (template_key or "custom").strip().lower()
    if t == "trend":
        return {
            "long1": {"type": "ema_cross", "fast": 20, "slow": 50, "direction": "above"},
            "short1": {"type": "ema_cross", "fast": 20, "slow": 50, "direction": "below"},
        }
    if t == "breakout":
        return {
            "long1": {"type": "donchian", "don_len": 20, "direction": "breakout"},
            "short1": {"type": "donchian", "don_len": 20, "direction": "breakdown"},
        }
    if t == "meanrev":
        return {
            "long1": {"type": "rsi", "rsi_len": 14, "op": "<", "level": 30},
            "short1": {"type": "rsi", "rsi_len": 14, "op": ">", "level": 70},
        }
    if t == "momentum":
        return {
            "long1": {"type": "close_vs_ma", "ma": "sma", "ma_len": 200, "op": ">"},
            "short1": {"type": "close_vs_ma", "ma": "sma", "ma_len": 200, "op": "<"},
        }
    if t == "builder_v2":
        # Conditions are built by the Builder v2 step4 UI, not the legacy step4 UI.
        return {}
    return {}


def _builder_v2_preview(draft: dict[str, Any]) -> dict[str, Any]:
    """Compute a human-friendly preview of the effective Builder v2 rules.

    Returns a dict suitable for template rendering:
      {
        "entry_tf": "15m",
        "context_tf": "1d" | "(entry)",
        "long": {"enabled": bool, "context": [{"label","expr"}...], "trigger": {"label","expr"}},
        "short": { ... },
      }
    """

    def _tf_expr(atom: str, tf: str | None) -> str:
        tf = (tf or "").strip()
        if not tf:
            return atom
        return f"{atom}@{tf}"

    def _expr_has_explicit_tf(expr: str) -> bool:
        try:
            return bool(re.search(r"@\s*\d+[mhdw]\b", expr, flags=re.IGNORECASE))
        except Exception:
            return False

    def _wrap_tf_if_missing(expr: str, tf: str | None) -> str:
        tf = (tf or "").strip()
        if not tf:
            return expr
        if _expr_has_explicit_tf(expr):
            return expr
        return f"({expr})@{tf}"

    def _apply_valid_for_bars(expr: str, bars: int | None) -> str:
        try:
            n = int(bars or 0)
        except Exception:
            n = 0
        if n <= 0:
            return expr
        # "Valid for N bars" means: the condition is true now OR it was true in the last N bars.
        # This is a simple, DSL-friendly implementation using shift().
        parts: list[str] = [f"({expr})"]
        for i in range(1, n + 1):
            parts.append(f"shift(({expr}), {i})")
        return "(" + " or ".join(parts) + ")"

    def _ma_expr(ma_type: str, length: int, tf: str | None) -> str:
        fn = "ema" if (ma_type or "").strip().lower() != "sma" else "sma"
        return _tf_expr(f"{fn}(close, {int(length)})", tf)

    entry_tf = str(draft.get("entry_tf") or "1h").strip()
    signal_tf_raw = str(draft.get("signal_tf") or "").strip()
    trigger_tf_raw = str(draft.get("trigger_tf") or "").strip()
    signal_tf_default = signal_tf_raw or entry_tf
    trigger_tf_default = trigger_tf_raw or entry_tf
    context_tf_raw = str(draft.get("context_tf") or "").strip()
    ctx_tf = context_tf_raw or None
    ctx_tf_label = context_tf_raw if context_tf_raw else "(entry)"
    sig_tf_label = signal_tf_raw if signal_tf_raw else "(entry)"
    trg_tf_label = trigger_tf_raw if trigger_tf_raw else "(entry)"

    align = bool(draft.get("align_with_context"))
    long_enabled = bool(draft.get("long_enabled"))
    short_enabled = bool(draft.get("short_enabled"))

    primary_context_type = str(draft.get("primary_context_type") or "ma_stack").strip().lower()

    ma_type = str(draft.get("ma_type") or "ema").strip().lower()
    ma_fast = int(draft.get("ma_fast") or 20)
    ma_mid = int(draft.get("ma_mid") or 50)
    ma_slow = int(draft.get("ma_slow") or 200)
    stack_mode = str(draft.get("stack_mode") or "none").strip().lower()
    slope_mode = str(draft.get("slope_mode") or "none").strip().lower()
    slope_lookback = int(draft.get("slope_lookback") or 10)
    try:
        min_ma_dist_pct = float(draft.get("min_ma_dist_pct") or 0.0)
    except Exception:
        min_ma_dist_pct = 0.0

    trigger_type = str(draft.get("trigger_type") or "pin_bar").strip().lower()
    try:
        trigger_valid_for_bars = int(draft.get("trigger_valid_for_bars") or 0)
    except Exception:
        trigger_valid_for_bars = 0
    try:
        pin_wick_body = float(draft.get("pin_wick_body") or 2.0)
    except Exception:
        pin_wick_body = 2.0
    try:
        pin_opp_wick_body_max = float(draft.get("pin_opp_wick_body_max") or 1.0)
    except Exception:
        pin_opp_wick_body_max = 1.0
    try:
        pin_min_body_pct = float(draft.get("pin_min_body_pct") or 0.2)
    except Exception:
        pin_min_body_pct = 0.2

    trigger_ma_type = str(draft.get("trigger_ma_type") or "ema").strip().lower()
    try:
        trigger_ma_len = int(draft.get("trigger_ma_len") or 20)
    except Exception:
        trigger_ma_len = 20
    try:
        trigger_don_len = int(draft.get("trigger_don_len") or 20)
    except Exception:
        trigger_don_len = 20
    try:
        trigger_range_len = int(draft.get("trigger_range_len") or 20)
    except Exception:
        trigger_range_len = 20
    try:
        trigger_atr_len = int(draft.get("trigger_atr_len") or 14)
    except Exception:
        trigger_atr_len = 14
    try:
        trigger_atr_mult = float(draft.get("trigger_atr_mult") or 2.0)
    except Exception:
        trigger_atr_mult = 2.0
    trigger_custom_bull_expr = str(draft.get("trigger_custom_bull_expr") or "").strip()
    trigger_custom_bear_expr = str(draft.get("trigger_custom_bear_expr") or "").strip()

    fast = _ma_expr(ma_type, ma_fast, ctx_tf)
    mid = _ma_expr(ma_type, ma_mid, ctx_tf)
    slow = _ma_expr(ma_type, ma_slow, ctx_tf)

    extra_context_rules = draft.get("context_rules")
    if not isinstance(extra_context_rules, list):
        extra_context_rules = []
    signal_rules = draft.get("signal_rules")
    if not isinstance(signal_rules, list):
        signal_rules = []
    extra_trigger_rules = draft.get("trigger_rules")
    if not isinstance(extra_trigger_rules, list):
        extra_trigger_rules = []

    def _resolve_rule_tf(rule: dict[str, Any], default_tf: str) -> str:
        raw = str(rule.get("tf") or "").strip().lower()
        if not raw or raw in {"default", "(default)", "entry", "(entry)"}:
            return default_tf
        return raw

    def _context_rule_items(rule: dict[str, Any], direction: str) -> list[dict[str, str]]:
        # Disable volume-based context rules temporarily
        rtype = str(rule.get("type") or "").strip().lower()
        if rtype in {"relative_volume", "volume_osc_increase", "volume_above_ma"} or "volume" in rtype:
            return []

        # Relative Volume (RVOL)
        if rtype == "relative_volume":
            ma_type = str(rule.get("ma_type") or "sma").strip().lower()
            length = int(rule.get("length") or 20)
            op = str(rule.get("op") or ">=").strip()
            threshold = float(rule.get("threshold") or 1.5)
            rvol = f"volume / {ma_type}(volume, {length})"
            expr = f"{rvol} {op} {threshold}"
            label = f"RVOL: {rvol} {op} {threshold} on {ctx_tf_label}"
            items.append({"group": "Volume", "label": label, "expr": _tf_expr(expr, ctx_tf)})
            return items

        # Volume Oscillator Increase
        if rtype == "volume_osc_increase":
            fast = int(rule.get("fast") or 12)
            slow = int(rule.get("slow") or 26)
            min_pct = float(rule.get("min_pct") or 0.1)
            lookback = int(rule.get("lookback") or 3)
            osc = f"(ema(volume, {fast}) - ema(volume, {slow})) / ema(volume, {slow})"
            expr = f"{osc} > {min_pct} and {osc} > shift({osc}, {lookback})"
            label = f"Volume Oscillator Increase: fast={fast}, slow={slow}, min%={min_pct}, N={lookback} on {ctx_tf_label}"
            items.append({"group": "Volume", "label": label, "expr": _tf_expr(expr, ctx_tf)})
            return items

        # Volume Above MA
        if rtype == "volume_above_ma":
            ma_type = str(rule.get("ma_type") or "ema").strip().lower()
            length = int(rule.get("length") or 20)
            min_pct = float(rule.get("min_pct") or 0.1)
            ma = f"{ma_type}(volume, {length})"
            expr = f"(volume - {ma}) / {ma} > {min_pct}"
            label = f"Volume Above {ma_type.upper()}({length}): min%={min_pct} on {ctx_tf_label}"
            items.append({"group": "Volume", "label": label, "expr": _tf_expr(expr, ctx_tf)})
            return items
        d = direction.strip().lower()
        items: list[dict[str, str]] = []

        if rtype in {"", "none"}:
            return items

        if rtype == "price_vs_ma":
            r_ma_type = str(rule.get("ma_type") or "ema").strip().lower()
            length = int(rule.get("length") or 200)
            ma = _ma_expr(r_ma_type, length, ctx_tf)
            px = _tf_expr("close", ctx_tf)

            slope_enabled = bool(rule.get("slope_enabled"))
            try:
                slope_lookback = int(rule.get("slope_lookback") or 10)
            except Exception:
                slope_lookback = 10
            try:
                min_dist_pct = float(rule.get("min_dist_pct") or 0.0)
            except Exception:
                min_dist_pct = 0.0

            if d == "bull":
                items.append(
                    {
                        "group": "Regime",
                        "label": f"Price above {r_ma_type.upper()}({length}) on {ctx_tf_label}",
                        "expr": f"{px} > {ma}",
                    }
                )
            else:
                items.append(
                    {
                        "group": "Regime",
                        "label": f"Price below {r_ma_type.upper()}({length}) on {ctx_tf_label}",
                        "expr": f"{px} < {ma}",
                    }
                )

            if slope_enabled and slope_lookback > 0:
                if d == "bull":
                    items.append(
                        {
                            "group": "Regime",
                            "label": f"MA slope up (now > {slope_lookback} bars ago) on {ctx_tf_label}",
                            "expr": f"{ma} > shift({ma}, {int(slope_lookback)})",
                        }
                    )
                else:
                    items.append(
                        {
                            "group": "Regime",
                            "label": f"MA slope down (now < {slope_lookback} bars ago) on {ctx_tf_label}",
                            "expr": f"{ma} < shift({ma}, {int(slope_lookback)})",
                        }
                    )

            if min_dist_pct and min_dist_pct > 0:
                thr = float(min_dist_pct)
                if d == "bull":
                    expr = f"({px} - {ma}) / {ma} > {thr}"
                else:
                    expr = f"({ma} - {px}) / {ma} > {thr}"
                items.append(
                    {
                        "group": "Separation",
                        "label": f"Price/MA distance >= {thr:g} (directional) on {ctx_tf_label}",
                        "expr": expr,
                    }
                )
            return items

        if rtype == "ma_cross_state":
            r_ma_type = str(rule.get("ma_type") or "ema").strip().lower()
            fast_len = int(rule.get("fast") or 20)
            slow_len = int(rule.get("slow") or 50)
            f = _ma_expr(r_ma_type, fast_len, ctx_tf)
            s = _ma_expr(r_ma_type, slow_len, ctx_tf)

            slope_enabled = bool(rule.get("slope_enabled"))
            slope_on = str(rule.get("slope_on") or "slow").strip().lower()
            if slope_on not in {"fast", "slow"}:
                slope_on = "slow"
            try:
                slope_lookback = int(rule.get("slope_lookback") or 10)
            except Exception:
                slope_lookback = 10
            try:
                min_spread_pct = float(rule.get("min_spread_pct") or 0.0)
            except Exception:
                min_spread_pct = 0.0

            if d == "bull":
                expr = f"{f} > {s}"
                label = f"{r_ma_type.upper()}({fast_len}) above {r_ma_type.upper()}({slow_len}) on {ctx_tf_label}"
            else:
                expr = f"{f} < {s}"
                label = f"{r_ma_type.upper()}({fast_len}) below {r_ma_type.upper()}({slow_len}) on {ctx_tf_label}"
            items.append({"group": "Regime", "label": label, "expr": expr})

            if slope_enabled and slope_lookback > 0:
                target = f if slope_on == "fast" else s
                which = "Fast" if slope_on == "fast" else "Slow"
                if d == "bull":
                    items.append(
                        {
                            "group": "Regime",
                            "label": f"{which} MA slope up (now > {slope_lookback} bars ago) on {ctx_tf_label}",
                            "expr": f"{target} > shift({target}, {int(slope_lookback)})",
                        }
                    )
                else:
                    items.append(
                        {
                            "group": "Regime",
                            "label": f"{which} MA slope down (now < {slope_lookback} bars ago) on {ctx_tf_label}",
                            "expr": f"{target} < shift({target}, {int(slope_lookback)})",
                        }
                    )

            if min_spread_pct and min_spread_pct > 0:
                thr = float(min_spread_pct)
                if d == "bull":
                    spread_expr = f"({f} - {s}) / {s} > {thr}"
                else:
                    spread_expr = f"({s} - {f}) / {s} > {thr}"
                items.append(
                    {
                        "group": "Separation",
                        "label": f"MA spread >= {thr:g} ({r_ma_type.upper()} {fast_len}/{slow_len}) on {ctx_tf_label}",
                        "expr": spread_expr,
                    }
                )
            return items

        if rtype == "atr_pct":
            atr_len = int(rule.get("atr_len") or 14)
            op = str(rule.get("op") or ">").strip()
            try:
                threshold = float(rule.get("threshold") or 0.005)
            except Exception:
                threshold = 0.005
            if op not in {"<", ">", "<=", ">="}:
                op = ">"
            atr = _tf_expr(f"atr({atr_len})", ctx_tf)
            px = _tf_expr("close", ctx_tf)
            items.append(
                {
                    "group": "Volatility",
                    "label": f"ATR({atr_len})/Price {op} {threshold:g} on {ctx_tf_label}",
                    "expr": f"{atr} / {px} {op} {threshold}",
                }
            )
            return items

        if rtype == "structure_breakout_state":
            length = int(rule.get("length") or 20)
            px = _tf_expr("close", ctx_tf)
            hi = _tf_expr(f"highest(high, {length})", ctx_tf)
            lo = _tf_expr(f"lowest(low, {length})", ctx_tf)
            if d == "bull":
                items.append(
                    {
                        "group": "Structure",
                        "label": f"Close above prior {length}-bar high on {ctx_tf_label}",
                        "expr": f"{px} > shift({hi}, 1)",
                    }
                )
            else:
                items.append(
                    {
                        "group": "Structure",
                        "label": f"Close below prior {length}-bar low on {ctx_tf_label}",
                        "expr": f"{px} < shift({lo}, 1)",
                    }
                )
            return items

        if rtype == "ma_spread_pct":
            r_ma_type = str(rule.get("ma_type") or "ema").strip().lower()
            fast_len = int(rule.get("fast") or 20)
            slow_len = int(rule.get("slow") or 200)
            try:
                threshold = float(rule.get("threshold") or 0.01)
            except Exception:
                threshold = 0.01
            f = _ma_expr(r_ma_type, fast_len, ctx_tf)
            s = _ma_expr(r_ma_type, slow_len, ctx_tf)

            slope_enabled = bool(rule.get("slope_enabled"))
            slope_on = str(rule.get("slope_on") or "slow").strip().lower()
            if slope_on not in {"fast", "slow"}:
                slope_on = "slow"
            try:
                slope_lookback = int(rule.get("slope_lookback") or 10)
            except Exception:
                slope_lookback = 10

            if d == "bull":
                expr = f"({f} - {s}) / {s} > {threshold}"
            else:
                expr = f"({s} - {f}) / {s} > {threshold}"
            items.append(
                {
                    "group": "Separation",
                    "label": f"MA spread >= {threshold:g} ({r_ma_type.upper()} {fast_len}/{slow_len}) on {ctx_tf_label}",
                    "expr": expr,
                }
            )

            if slope_enabled and slope_lookback > 0:
                target = f if slope_on == "fast" else s
                which = "Fast" if slope_on == "fast" else "Slow"
                if d == "bull":
                    items.append(
                        {
                            "group": "Regime",
                            "label": f"{which} MA slope up (now > {slope_lookback} bars ago) on {ctx_tf_label}",
                            "expr": f"{target} > shift({target}, {int(slope_lookback)})",
                        }
                    )
                else:
                    items.append(
                        {
                            "group": "Regime",
                            "label": f"{which} MA slope down (now < {slope_lookback} bars ago) on {ctx_tf_label}",
                            "expr": f"{target} < shift({target}, {int(slope_lookback)})",
                        }
                    )
            return items

        if rtype == "custom":
            bull_expr = str(rule.get("bull_expr") or "").strip()
            bear_expr = str(rule.get("bear_expr") or "").strip()
            expr = bull_expr if d == "bull" else bear_expr
            expr = _wrap_tf_if_missing(expr, ctx_tf)
            items.append(
                {
                    "group": "Custom",
                    "label": f"Custom context ({'bull' if d == 'bull' else 'bear'}) on {ctx_tf_label}",
                    "expr": expr or "(empty)",
                }
            )
            return items

        items.append({"group": "Context", "label": f"Context: {rtype}", "expr": "(unknown)"})
        return items

    def _signal_rule_items(rule: dict[str, Any], direction: str) -> list[dict[str, str]]:
        # Disable volume-based signal rules temporarily
        rtype = str(rule.get("type") or "").strip().lower()
        if rtype in {"relative_volume", "volume_osc_increase", "volume_above_ma"} or "volume" in rtype:
            return []

        # Relative Volume (RVOL)
        if rtype == "relative_volume":
            ma_type = str(rule.get("ma_type") or "sma").strip().lower()
            length = int(rule.get("length") or 20)
            op = str(rule.get("op") or ">=").strip()
            threshold = float(rule.get("threshold") or 1.5)
            rvol = f"volume / {ma_type}(volume, {length})"
            expr = f"{rvol} {op} {threshold}"
            label = f"RVOL: {rvol} {op} {threshold} on {tf}"
            expr = _apply_valid_for_bars(expr, valid_for)
            items.append({"group": "Volume", "label": label, "expr": _tf_expr(expr, tf)})
            return items

        # Volume Oscillator Increase
        if rtype == "volume_osc_increase":
            fast = int(rule.get("fast") or 12)
            slow = int(rule.get("slow") or 26)
            min_pct = float(rule.get("min_pct") or 0.1)
            lookback = int(rule.get("lookback") or 3)
            osc = f"(ema(volume, {fast}) - ema(volume, {slow})) / ema(volume, {slow})"
            expr = f"{osc} > {min_pct} and {osc} > shift({osc}, {lookback})"
            label = f"Volume Oscillator Increase: fast={fast}, slow={slow}, min%={min_pct}, N={lookback} on {tf}"
            expr = _apply_valid_for_bars(expr, valid_for)
            items.append({"group": "Volume", "label": label, "expr": _tf_expr(expr, tf)})
            return items

        # Volume Above MA
        if rtype == "volume_above_ma":
            ma_type = str(rule.get("ma_type") or "ema").strip().lower()
            length = int(rule.get("length") or 20)
            min_pct = float(rule.get("min_pct") or 0.1)
            ma = f"{ma_type}(volume, {length})"
            expr = f"(volume - {ma}) / {ma} > {min_pct}"
            label = f"Volume Above {ma_type.upper()}({length}): min%={min_pct} on {tf}"
            expr = _apply_valid_for_bars(expr, valid_for)
            items.append({"group": "Volume", "label": label, "expr": _tf_expr(expr, tf)})
            return items
        d = direction.strip().lower()
        rtype = str(rule.get("type") or "").strip().lower()
        items: list[dict[str, str]] = []

        tf = _resolve_rule_tf(rule, signal_tf_default)
        valid_for = rule.get("valid_for_bars")

        if rtype in {"", "none"}:
            return items

        if rtype == "ma_cross":
            r_ma_type = str(rule.get("ma_type") or "ema").strip().lower()
            fast_len = int(rule.get("fast") or 20)
            slow_len = int(rule.get("slow") or 50)
            f = _ma_expr(r_ma_type, fast_len, tf)
            s = _ma_expr(r_ma_type, slow_len, tf)
            if d == "bull":
                expr = f"crosses_above({f}, {s})"
                label = f"MA cross up ({r_ma_type.upper()} {fast_len}/{slow_len}) on {tf}"
            else:
                expr = f"crosses_below({f}, {s})"
                label = f"MA cross down ({r_ma_type.upper()} {fast_len}/{slow_len}) on {tf}"
            expr = _apply_valid_for_bars(expr, valid_for)
            if valid_for:
                try:
                    n = int(valid_for)
                    if n > 0:
                        label = f"{label} (valid {n} bars)"
                except Exception:
                    pass
            items.append({"group": "Signal", "label": label, "expr": expr})
            return items

        if rtype == "rsi_threshold":
            length = int(rule.get("length") or 14)
            try:
                bull_level = float(rule.get("bull_level") or 30)
            except Exception:
                bull_level = 30
            try:
                bear_level = float(rule.get("bear_level") or 70)
            except Exception:
                bear_level = 70
            r = _tf_expr(f"rsi(close, {length})", tf)
            expr = None
            label = None
            if d == "bull":
                items.append(
                    {
                        "group": "Signal",
                        "label": f"RSI({length}) < {bull_level:g} on {tf}",
                        "expr": f"{r} < {bull_level}",
                    }
                )
            else:
                items.append(
                    {
                        "group": "Signal",
                        "label": f"RSI({length}) > {bear_level:g} on {tf}",
                        "expr": f"{r} > {bear_level}",
                    }
                )
            return items

        if rtype == "rsi_cross_back":
            length = int(rule.get("length") or 14)
            try:
                bull_level = float(rule.get("bull_level") or 30)
            except Exception:
                bull_level = 30
            try:
                bear_level = float(rule.get("bear_level") or 70)
            except Exception:
                bear_level = 70
            r = _tf_expr(f"rsi(close, {length})", tf)
            if d == "bull":
                expr = f"({r} > {bull_level}) and shift({r}, 1) <= {bull_level}"
                label = f"RSI({length}) crosses above {bull_level:g} on {tf}"
            else:
                expr = f"({r} < {bear_level}) and shift({r}, 1) >= {bear_level}"
                label = f"RSI({length}) crosses below {bear_level:g} on {tf}"
            expr = _apply_valid_for_bars(expr, valid_for)
            if valid_for:
                try:
                    n = int(valid_for)
                    if n > 0:
                        label = f"{label} (valid {n} bars)"
                except Exception:
                    pass
            items.append({"group": "Signal", "label": label, "expr": expr})
            return items

        if rtype == "donchian_breakout":
            length = int(rule.get("length") or 20)
            px = _tf_expr("close", tf)
            if d == "bull":
                ch = _tf_expr(f"donchian_high({length})", tf)
                expr = f"crosses_above({px}, shift({ch}, 1))"
                label = f"Donchian breakout ({length}) on {tf}"
            else:
                ch = _tf_expr(f"donchian_low({length})", tf)
                expr = f"crosses_below({px}, shift({ch}, 1))"
                label = f"Donchian breakdown ({length}) on {tf}"
            expr = _apply_valid_for_bars(expr, valid_for)
            if valid_for:
                try:
                    n = int(valid_for)
                    if n > 0:
                        label = f"{label} (valid {n} bars)"
                except Exception:
                    pass
            items.append({"group": "Signal", "label": label, "expr": expr})
            return items

        if rtype == "new_high_low_breakout":
            length = int(rule.get("length") or 20)
            px = _tf_expr("close", tf)
            hi = _tf_expr(f"highest(high, {length})", tf)
            lo = _tf_expr(f"lowest(low, {length})", tf)
            if d == "bull":
                expr = f"crosses_above({px}, shift({hi}, 1))"
                label = f"Breakout above prior {length}-bar high on {tf}"
            else:
                expr = f"crosses_below({px}, shift({lo}, 1))"
                label = f"Breakdown below prior {length}-bar low on {tf}"
            expr = _apply_valid_for_bars(expr, valid_for)
            if valid_for:
                try:
                    n = int(valid_for)
                    if n > 0:
                        label = f"{label} (valid {n} bars)"
                except Exception:
                    pass
            items.append({"group": "Signal", "label": label, "expr": expr})
            return items

        if rtype == "pullback_to_ma":
            r_ma_type = str(rule.get("ma_type") or "ema").strip().lower()
            length = int(rule.get("length") or 20)
            ma = _ma_expr(r_ma_type, length, tf)
            px = _tf_expr("close", tf)
            if d == "bull":
                expr = f"{px} < {ma}"
                label = f"Pullback: close below {r_ma_type.upper()}({length}) on {tf}"
            else:
                expr = f"{px} > {ma}"
                label = f"Pullback: close above {r_ma_type.upper()}({length}) on {tf}"
            expr = _apply_valid_for_bars(expr, valid_for)
            if valid_for:
                try:
                    n = int(valid_for)
                    if n > 0:
                        label = f"{label} (valid {n} bars)"
                except Exception:
                    pass
            items.append({"group": "Signal", "label": label, "expr": expr})
            return items

        if rtype == "custom":
            bull_expr = str(rule.get("bull_expr") or "").strip()
            bear_expr = str(rule.get("bear_expr") or "").strip()
            expr = bull_expr if d == "bull" else bear_expr
            expr = _wrap_tf_if_missing(expr, tf)
            if expr:
                expr = _apply_valid_for_bars(expr, valid_for)
            items.append(
                {
                    "group": "Custom",
                    "label": f"Custom signal ({'bull' if d == 'bull' else 'bear'}) on {tf}",
                    "expr": expr or "(empty)",
                }
            )
            return items

        items.append({"group": "Signal", "label": f"Signal: {rtype}", "expr": "(unknown)"})
        return items

    def _trigger_rule_items(rule: dict[str, Any], direction: str) -> list[dict[str, str]]:
        d = direction.strip().lower()
        rtype = str(rule.get("type") or "").strip().lower()
        items: list[dict[str, str]] = []

        tf = _resolve_rule_tf(rule, trigger_tf_default)
        valid_for = rule.get("valid_for_bars")

        if rtype in {"", "none"}:
            return items

        if rtype == "pin_bar":
            wbr = float(rule.get("pin_wick_body") or pin_wick_body)
            ow = float(rule.get("pin_opp_wick_body_max") or pin_opp_wick_body_max)
            mb = float(rule.get("pin_min_body_pct") or pin_min_body_pct)
            expr = _tf_expr(
                f"pin_bar(open, high, low, close, \"{'bull' if d == 'bull' else 'bear'}\", {wbr}, {ow}, {mb})",
                tf,
            )
            expr = _apply_valid_for_bars(expr, valid_for)
            label = f"Pin bar ({'bull' if d == 'bull' else 'bear'}) on {tf}"
            if valid_for:
                try:
                    n = int(valid_for)
                    if n > 0:
                        label = f"{label} (valid {n} bars)"
                except Exception:
                    pass
            items.append(
                {
                    "group": "Trigger",
                    "label": label,
                    "expr": expr,
                }
            )
            return items

        if rtype == "inside_bar_breakout":
            inside = _tf_expr("inside_bar(high, low)", tf)
            if d == "bull":
                expr = f"shift({inside}, 1) and {_tf_expr('close', tf)} > shift({_tf_expr('high', tf)}, 2)"
                label = f"Inside-bar breakout up (confirm next bar) on {tf}"
            else:
                expr = f"shift({inside}, 1) and {_tf_expr('close', tf)} < shift({_tf_expr('low', tf)}, 2)"
                label = f"Inside-bar breakout down (confirm next bar) on {tf}"
            expr = _apply_valid_for_bars(expr, valid_for)
            if valid_for:
                try:
                    n = int(valid_for)
                    if n > 0:
                        label = f"{label} (valid {n} bars)"
                except Exception:
                    pass
            items.append({"group": "Trigger", "label": label, "expr": expr})
            return items

        if rtype == "engulfing":
            expr = _tf_expr(
                f"engulfing(open, close, \"{'bull' if d == 'bull' else 'bear'}\")",
                tf,
            )
            expr = _apply_valid_for_bars(expr, valid_for)
            label = f"Engulfing ({'bull' if d == 'bull' else 'bear'}) on {tf}"
            if valid_for:
                try:
                    n = int(valid_for)
                    if n > 0:
                        label = f"{label} (valid {n} bars)"
                except Exception:
                    pass
            items.append(
                {
                    "group": "Trigger",
                    "label": label,
                    "expr": expr,
                }
            )
            return items

        if rtype == "ma_reclaim":
            r_ma_type = str(rule.get("ma_type") or "ema").strip().lower()
            length = int(rule.get("length") or 20)
            ma = _ma_expr(r_ma_type, length, tf)
            if d == "bull":
                expr = f"{_tf_expr('low', tf)} <= {ma} and {_tf_expr('close', tf)} > {ma}"
                label = f"Reclaim {r_ma_type.upper()}({length}) on {tf}"
            else:
                expr = f"{_tf_expr('high', tf)} >= {ma} and {_tf_expr('close', tf)} < {ma}"
                label = f"Reject {r_ma_type.upper()}({length}) on {tf}"
            expr = _apply_valid_for_bars(expr, valid_for)
            if valid_for:
                try:
                    n = int(valid_for)
                    if n > 0:
                        label = f"{label} (valid {n} bars)"
                except Exception:
                    pass
            items.append({"group": "Trigger", "label": label, "expr": expr})
            return items

        if rtype == "prior_bar_break":
            px = _tf_expr("close", tf)
            if d == "bull":
                expr = f"crosses_above({px}, shift({_tf_expr('high', tf)}, 1))"
                label = f"Break above prior high on {tf}"
            else:
                expr = f"crosses_below({px}, shift({_tf_expr('low', tf)}, 1))"
                label = f"Break below prior low on {tf}"
            expr = _apply_valid_for_bars(expr, valid_for)
            if valid_for:
                try:
                    n = int(valid_for)
                    if n > 0:
                        label = f"{label} (valid {n} bars)"
                except Exception:
                    pass
            items.append({"group": "Trigger", "label": label, "expr": expr})
            return items

        if rtype == "donchian_breakout":
            length = int(rule.get("length") or rule.get("don_len") or 20)
            px = _tf_expr("close", tf)
            if d == "bull":
                ch = _tf_expr(f"donchian_high({length})", tf)
                expr = f"crosses_above({px}, shift({ch}, 1))"
                label = f"Donchian breakout ({length}) on {tf}"
            else:
                ch = _tf_expr(f"donchian_low({length})", tf)
                expr = f"crosses_below({px}, shift({ch}, 1))"
                label = f"Donchian breakdown ({length}) on {tf}"
            expr = _apply_valid_for_bars(expr, valid_for)
            if valid_for:
                try:
                    n = int(valid_for)
                    if n > 0:
                        label = f"{label} (valid {n} bars)"
                except Exception:
                    pass
            items.append({"group": "Trigger", "label": label, "expr": expr})
            return items

        if rtype == "range_breakout":
            length = int(rule.get("length") or rule.get("range_len") or 20)
            px = _tf_expr("close", tf)
            hi = _tf_expr(f"highest(high, {length})", tf)
            lo = _tf_expr(f"lowest(low, {length})", tf)
            if d == "bull":
                expr = f"crosses_above({px}, shift({hi}, 1))"
                label = f"Breakout above {length}-bar range on {tf}"
            else:
                expr = f"crosses_below({px}, shift({lo}, 1))"
                label = f"Breakdown below {length}-bar range on {tf}"
            expr = _apply_valid_for_bars(expr, valid_for)
            if valid_for:
                try:
                    n = int(valid_for)
                    if n > 0:
                        label = f"{label} (valid {n} bars)"
                except Exception:
                    pass
            items.append({"group": "Trigger", "label": label, "expr": expr})
            return items

        if rtype == "wide_range_candle":
            atr_len = int(rule.get("atr_len") or 14)
            try:
                mult = float(rule.get("mult") or 2.0)
            except Exception:
                mult = 2.0
            atr = _tf_expr(f"atr({atr_len})", tf)
            rng = f"({_tf_expr('high', tf)} - {_tf_expr('low', tf)})"
            if d == "bull":
                expr = f"{rng} > {mult} * {atr} and {_tf_expr('close', tf)} > {_tf_expr('open', tf)}"
                label = f"Wide-range bull candle (> {mult:g}x ATR({atr_len})) on {tf}"
            else:
                expr = f"{rng} > {mult} * {atr} and {_tf_expr('close', tf)} < {_tf_expr('open', tf)}"
                label = f"Wide-range bear candle (> {mult:g}x ATR({atr_len})) on {tf}"
            expr = _apply_valid_for_bars(expr, valid_for)
            if valid_for:
                try:
                    n = int(valid_for)
                    if n > 0:
                        label = f"{label} (valid {n} bars)"
                except Exception:
                    pass
            items.append({"group": "Trigger", "label": label, "expr": expr})
            return items

        if rtype == "custom":
            bull_expr = str(rule.get("bull_expr") or "").strip()
            bear_expr = str(rule.get("bear_expr") or "").strip()
            expr = bull_expr if d == "bull" else bear_expr
            expr = _wrap_tf_if_missing(expr, tf)
            if expr:
                expr = _apply_valid_for_bars(expr, valid_for)
            items.append(
                {
                    "group": "Custom",
                    "label": f"Custom trigger ({'bull' if d == 'bull' else 'bear'}) on {tf}",
                    "expr": expr or "(empty)",
                }
            )
            return items

        items.append({"group": "Trigger", "label": f"Trigger: {rtype}", "expr": "(unknown)"})
        return items

    def _context_items(direction: str) -> list[dict[str, str]]:
        d = direction.strip().lower()
        items: list[dict[str, str]] = []

        # Only include MA-stack context when selected as the primary context.
        if primary_context_type != "ma_stack":
            return items

        if stack_mode in {"bull", "bear"}:
            if d == "bull":
                items.append(
                    {
                        "group": "Regime",
                        "label": f"MA stack aligned (fast > mid > slow) on {ctx_tf_label}",
                        "expr": f"{fast} > {mid} and {mid} > {slow}",
                    }
                )
            else:
                items.append(
                    {
                        "group": "Regime",
                        "label": f"MA stack aligned (fast < mid < slow) on {ctx_tf_label}",
                        "expr": f"{fast} < {mid} and {mid} < {slow}",
                    }
                )

        if slope_mode in {"up", "down"}:
            if d == "bull":
                items.append(
                    {
                        "group": "Regime",
                        "label": f"Mid MA slope up (now > {slope_lookback} bars ago) on {ctx_tf_label}",
                        "expr": f"{mid} > shift({mid}, {int(slope_lookback)})",
                    }
                )
            else:
                items.append(
                    {
                        "group": "Regime",
                        "label": f"Mid MA slope down (now < {slope_lookback} bars ago) on {ctx_tf_label}",
                        "expr": f"{mid} < shift({mid}, {int(slope_lookback)})",
                    }
                )

        if min_ma_dist_pct and min_ma_dist_pct > 0:
            pct = float(min_ma_dist_pct)
            if d == "bull":
                items.append(
                    {
                        "group": "Separation",
                        "label": f"Fast/Slow distance >= {pct:.3g} (directional) on {ctx_tf_label}",
                        "expr": f"({fast} - {slow}) / {slow} > {pct}",
                    }
                )
            else:
                items.append(
                    {
                        "group": "Separation",
                        "label": f"Fast/Slow distance >= {pct:.3g} (directional) on {ctx_tf_label}",
                        "expr": f"({slow} - {fast}) / {slow} > {pct}",
                    }
                )

        return items

    def _signal_items(direction: str) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for r in signal_rules:
            if isinstance(r, dict):
                out.extend(_signal_rule_items(r, direction))
        return out

    def _trigger_items(direction: str) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []

        # Trigger 1 (from legacy single-trigger fields) for backward compatibility.
        if trigger_type and trigger_type not in {"none", ""}:
            out.extend(
                _trigger_rule_items(
                    {
                        "type": trigger_type,
                        "tf": trigger_tf_raw or "default",
                        "valid_for_bars": trigger_valid_for_bars if trigger_valid_for_bars > 0 else None,
                        "pin_wick_body": pin_wick_body,
                        "pin_opp_wick_body_max": pin_opp_wick_body_max,
                        "pin_min_body_pct": pin_min_body_pct,
                        "ma_type": trigger_ma_type,
                        "length": trigger_ma_len,
                        "atr_len": trigger_atr_len,
                        "mult": trigger_atr_mult,
                        "bull_expr": trigger_custom_bull_expr,
                        "bear_expr": trigger_custom_bear_expr,
                        "don_len": trigger_don_len,
                        "range_len": trigger_range_len,
                    },
                    direction,
                )
            )

        for r in extra_trigger_rules:
            if isinstance(r, dict):
                out.extend(_trigger_rule_items(r, direction))
        return out

    # Context direction (bull/bear regime) can be aligned across sides.
    # Trade direction (signals/triggers) should always be bull-for-LONG and bear-for-SHORT.
    aligned_dual = align and long_enabled and short_enabled

    def _single_direction_default(side: str) -> str:
        # If the user explicitly selected bear/down in the UI, treat that as the direction.
        # (This is primarily driven by MA-stack settings today.)
        if side == "long":
            if stack_mode == "bear" or slope_mode == "down":
                return "bear"
            return "bull"
        # short
        if stack_mode == "bull" or slope_mode == "up":
            return "bull"
        return "bear"

    long_ctx_dir = "bull" if aligned_dual else _single_direction_default("long")
    short_ctx_dir = "bear" if aligned_dual else _single_direction_default("short")

    long_trade_dir = "bull"
    short_trade_dir = "bear"

    return {
        "entry_tf": entry_tf,
        "context_tf": ctx_tf_label,
        "signal_tf": sig_tf_label,
        "trigger_tf": trg_tf_label,
        "aligned_dual": aligned_dual,
        "long": {
            "enabled": long_enabled,
            "direction": long_trade_dir,
            "context_direction": long_ctx_dir,
            "context": (
                (
                    _context_items(long_ctx_dir)
                    + sum((_context_rule_items(r, long_ctx_dir) for r in extra_context_rules if isinstance(r, dict)), [])
                )
                if long_enabled
                else []
            ),
            "signals": _signal_items(long_trade_dir) if long_enabled else [],
            "triggers": _trigger_items(long_trade_dir) if long_enabled else [],
        },
        "short": {
            "enabled": short_enabled,
            "direction": short_trade_dir,
            "context_direction": short_ctx_dir,
            "context": (
                (
                    _context_items(short_ctx_dir)
                    + sum((_context_rule_items(r, short_ctx_dir) for r in extra_context_rules if isinstance(r, dict)), [])
                )
                if short_enabled
                else []
            ),
            "signals": _signal_items(short_trade_dir) if short_enabled else [],
            "triggers": _trigger_items(short_trade_dir) if short_enabled else [],
        },
    }


@app.post("/api/guided/builder_v2/preview", response_class=JSONResponse)
async def api_guided_builder_v2_preview(request: Request) -> JSONResponse:
    """Live preview for Builder v2 Step 4.

    Expects JSON payload matching the Step 4 form fields.
    """
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}

    # Start from a minimal base (so missing keys don't crash preview).
    base: dict[str, Any] = {
        "entry_tf": str(payload.get("entry_tf") or "1h"),
        "context_tf": str(payload.get("context_tf") or ""),
        "signal_tf": str(payload.get("signal_tf") or ""),
        "trigger_tf": str(payload.get("trigger_tf") or ""),
        "primary_context_type": str(payload.get("primary_context_type") or "ma_stack"),
        "align_with_context": bool(payload.get("align_with_context")),
        "long_enabled": bool(payload.get("long_enabled")),
        "short_enabled": bool(payload.get("short_enabled")),
        "ma_type": str(payload.get("ma_type") or "ema"),
        "ma_fast": payload.get("ma_fast") or 20,
        "ma_mid": payload.get("ma_mid") or 50,
        "ma_slow": payload.get("ma_slow") or 200,
        "stack_mode": str(payload.get("stack_mode") or "none"),
        "slope_mode": str(payload.get("slope_mode") or "none"),
        "slope_lookback": payload.get("slope_lookback") or 10,
        "min_ma_dist_pct": payload.get("min_ma_dist_pct") or 0,
        "trigger_type": str(payload.get("trigger_type") or "pin_bar"),
        "trigger_valid_for_bars": payload.get("trigger_valid_for_bars") or 0,
        "pin_wick_body": payload.get("pin_wick_body") or 2.0,
        "pin_opp_wick_body_max": payload.get("pin_opp_wick_body_max") or 1.0,
        "pin_min_body_pct": payload.get("pin_min_body_pct") or 0.2,
        "trigger_ma_type": str(payload.get("trigger_ma_type") or "ema"),
        "trigger_ma_len": payload.get("trigger_ma_len") or 20,
        "trigger_don_len": payload.get("trigger_don_len") or 20,
        "trigger_range_len": payload.get("trigger_range_len") or 20,
        "trigger_atr_len": payload.get("trigger_atr_len") or 14,
        "trigger_atr_mult": payload.get("trigger_atr_mult") or 2.0,
        "trigger_custom_bull_expr": str(payload.get("trigger_custom_bull_expr") or ""),
        "trigger_custom_bear_expr": str(payload.get("trigger_custom_bear_expr") or ""),
        "context_rules": payload.get("context_rules") if isinstance(payload.get("context_rules"), list) else [],
        "signal_rules": payload.get("signal_rules") if isinstance(payload.get("signal_rules"), list) else [],
        "trigger_rules": payload.get("trigger_rules") if isinstance(payload.get("trigger_rules"), list) else [],
    }

    # Normalize numeric fields
    for k in (
        "ma_fast",
        "ma_mid",
        "ma_slow",
        "slope_lookback",
        "trigger_valid_for_bars",
        "trigger_ma_len",
        "trigger_don_len",
        "trigger_range_len",
        "trigger_atr_len",
    ):
        try:
            base[k] = int(base[k])
        except Exception:
            pass
    for k in (
        "min_ma_dist_pct",
        "pin_wick_body",
        "pin_opp_wick_body_max",
        "pin_min_body_pct",
        "trigger_atr_mult",
    ):
        try:
            base[k] = float(base[k])
        except Exception:
            pass

    preview = _builder_v2_preview(base)
    return JSONResponse(preview)


def _find_latest_gui_dataset(repo_root: Path) -> Optional[Path]:
    base = repo_root / "data" / "logs" / "gui_runs"
    if not base.exists() or not base.is_dir():
        return None

    candidates: list[Path] = []
    try:
        for p in base.rglob("dataset_*"):
            try:
                if not p.is_file():
                    continue
                if "_inspect" in p.parts:
                    continue
                candidates.append(p)
            except Exception:
                continue
    except Exception:
        return None

    if not candidates:
        return None

    try:
        return max(candidates, key=lambda x: x.stat().st_mtime)
    except Exception:
        return None


@app.post("/api/guided/builder_v2/context_visual", response_class=JSONResponse)
async def api_guided_builder_v2_context_visual(request: Request) -> JSONResponse:
    """Candlestick context preview for Builder v2 Step 4.

    Uses the newest dataset saved under data/logs/gui_runs/**/dataset_*.*.
    Returns a Plotly figure JSON payload suitable for Plotly.react.
    """

    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}

    dataset_path = _find_latest_gui_dataset(REPO_ROOT)
    if dataset_path is None:
        return JSONResponse(
            {
                "ok": False,
                "message": "No dataset found under data/logs/gui_runs/. Run a backtest from the GUI (with an uploaded dataset) to enable this preview.",
            }
        )

    entry_tf = str(payload.get("entry_tf") or "1h").strip()
    context_tf = str(payload.get("context_tf") or "").strip()
    target_tf = context_tf or entry_tf or "1h"
    primary_context_type = str(payload.get("primary_context_type") or "ma_stack").strip()

    # Pull context params from the payload (best-effort)
    ma_type = str(payload.get("ma_type") or "ema").strip().lower()
    ma_fast = int(payload.get("ma_fast") or 20)
    ma_mid = int(payload.get("ma_mid") or 50)
    ma_slow = int(payload.get("ma_slow") or 200)

    try:
        from adapters.data.data_loader import DataLoader

        loader = DataLoader()
        df = loader.load(dataset_path)
    except Exception as e:
        return JSONResponse({"ok": False, "message": f"Failed to load dataset: {type(e).__name__}: {e}"})

    try:
        import pandas as pd

        if not isinstance(df.index, pd.DatetimeIndex):
            return JSONResponse({"ok": False, "message": "Dataset index is not datetime; cannot render preview."})

        # Ensure required columns exist.
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                return JSONResponse({"ok": False, "message": f"Dataset missing required column: {col}"})
        if "volume" not in df.columns:
            df = df.copy()
            df["volume"] = 0.0

        # Resample to target timeframe (best-effort). If it fails, fall back to raw.
        try:
            from engine.resampler import resample_ohlcv

            df_tf = resample_ohlcv(df[["open", "high", "low", "close", "volume"]], target_tf)
        except Exception:
            df_tf = df[["open", "high", "low", "close", "volume"]].copy()

        df_tf = df_tf.dropna(subset=["open", "high", "low", "close"], how="any")
        if len(df_tf) < 20:
            return JSONResponse({"ok": False, "message": "Not enough rows to render context preview."})

        # Keep a manageable window.
        df_tf = df_tf.tail(300)

        close = df_tf["close"].astype(float)
        high = df_tf["high"].astype(float)
        low = df_tf["low"].astype(float)

        def _ma(series: pd.Series, length: int, kind: str) -> pd.Series:
            length_i = max(1, int(length))
            if kind == "sma":
                return series.rolling(length_i).mean()
            return series.ewm(span=length_i, adjust=False).mean()

        def _segments(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
            mask_b = mask.fillna(False).astype(bool)
            segs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            start = None
            for ts, on in mask_b.items():
                if on and start is None:
                    start = ts
                if (not on) and start is not None:
                    segs.append((start, ts))
                    start = None
            if start is not None:
                segs.append((start, mask_b.index[-1]))
            return segs

        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df_tf.index,
                    open=df_tf["open"],
                    high=df_tf["high"],
                    low=df_tf["low"],
                    close=df_tf["close"],
                    name="Price",
                )
            ]
        )

        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=18, b=10),
            xaxis=dict(rangeslider=dict(visible=False)),
            yaxis=dict(title=None),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )

        msg_bits: list[str] = []
        shaded_any = False

        # Primary overlays
        if primary_context_type == "ma_stack":
            fast = _ma(close, ma_fast, ma_type)
            mid = _ma(close, ma_mid, ma_type)
            slow = _ma(close, ma_slow, ma_type)
            fig.add_trace(go.Scatter(x=df_tf.index, y=fast, mode="lines", name=f"{ma_type.upper()} {ma_fast}"))
            fig.add_trace(go.Scatter(x=df_tf.index, y=mid, mode="lines", name=f"{ma_type.upper()} {ma_mid}"))
            fig.add_trace(go.Scatter(x=df_tf.index, y=slow, mode="lines", name=f"{ma_type.upper()} {ma_slow}"))

            bull = (fast > mid) & (mid > slow)
            bear = (fast < mid) & (mid < slow)
            for x0, x1 in _segments(bull):
                fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(0, 200, 0, 0.08)", line_width=0)
                shaded_any = True
            for x0, x1 in _segments(bear):
                fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(200, 0, 0, 0.08)", line_width=0)
                shaded_any = True
            msg_bits.append("Shading: bull/bear MA stack")

        elif primary_context_type in {"price_vs_ma", "ma_cross_state", "structure_breakout_state", "ma_spread_pct", "atr_pct"}:
            rules = payload.get("context_rules") if isinstance(payload.get("context_rules"), list) else []
            rule0 = rules[0] if rules and isinstance(rules[0], dict) else {}

            if primary_context_type == "price_vs_ma":
                r_ma_type = str(rule0.get("ma_type") or "ema").lower()
                length = int(rule0.get("length") or 200)
                ma = _ma(close, length, r_ma_type)
                fig.add_trace(go.Scatter(x=df_tf.index, y=ma, mode="lines", name=f"{r_ma_type.upper()} {length}"))
                bull = close > ma
                bear = close < ma
                for x0, x1 in _segments(bull):
                    fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(0, 200, 0, 0.08)", line_width=0)
                    shaded_any = True
                for x0, x1 in _segments(bear):
                    fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(200, 0, 0, 0.08)", line_width=0)
                    shaded_any = True
                msg_bits.append("Shading: close vs MA")

            elif primary_context_type == "ma_cross_state":
                r_ma_type = str(rule0.get("ma_type") or "ema").lower()
                fast_len = int(rule0.get("fast") or 20)
                slow_len = int(rule0.get("slow") or 50)
                fast = _ma(close, fast_len, r_ma_type)
                slow = _ma(close, slow_len, r_ma_type)
                fig.add_trace(go.Scatter(x=df_tf.index, y=fast, mode="lines", name=f"{r_ma_type.upper()} {fast_len}"))
                fig.add_trace(go.Scatter(x=df_tf.index, y=slow, mode="lines", name=f"{r_ma_type.upper()} {slow_len}"))
                bull = fast > slow
                bear = fast < slow
                for x0, x1 in _segments(bull):
                    fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(0, 200, 0, 0.08)", line_width=0)
                    shaded_any = True
                for x0, x1 in _segments(bear):
                    fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(200, 0, 0, 0.08)", line_width=0)
                    shaded_any = True
                msg_bits.append("Shading: fast MA above/below slow MA")

            elif primary_context_type == "structure_breakout_state":
                length = int(rule0.get("length") or 20)

                # Visualize "breakout state" with a Bollinger-style channel.
                # This tends to be more readable at a glance than raw prior-high/low lines.
                basis = close.rolling(length).mean()
                dev = close.rolling(length).std()
                mult = 2.0
                upper = basis + mult * dev
                lower = basis - mult * dev

                fig.add_trace(
                    go.Scatter(
                        x=df_tf.index,
                        y=lower,
                        mode="lines",
                        name=f"BB lower ({length}, {mult:g})",
                        line=dict(color="rgba(0,0,0,0.30)", width=1),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_tf.index,
                        y=upper,
                        mode="lines",
                        name=f"BB upper ({length}, {mult:g})",
                        line=dict(color="rgba(0,0,0,0.30)", width=1),
                        fill="tonexty",
                        fillcolor="rgba(0,0,0,0.06)",
                    )
                )

                bull = close > upper
                bear = close < lower
                for x0, x1 in _segments(bull):
                    fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(0, 200, 0, 0.08)", line_width=0)
                    shaded_any = True
                for x0, x1 in _segments(bear):
                    fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(200, 0, 0, 0.08)", line_width=0)
                    shaded_any = True
                msg_bits.append("Shading: close outside Bollinger channel")

            elif primary_context_type == "ma_spread_pct":
                r_ma_type = str(rule0.get("ma_type") or "ema").lower()
                fast_len = int(rule0.get("fast") or 20)
                slow_len = int(rule0.get("slow") or 200)
                thr = float(rule0.get("threshold") or 0.01)
                fast = _ma(close, fast_len, r_ma_type)
                slow = _ma(close, slow_len, r_ma_type)
                fig.add_trace(go.Scatter(x=df_tf.index, y=fast, mode="lines", name=f"{r_ma_type.upper()} {fast_len}"))
                fig.add_trace(go.Scatter(x=df_tf.index, y=slow, mode="lines", name=f"{r_ma_type.upper()} {slow_len}"))
                spread = (fast - slow).abs() / slow.abs().replace(0.0, float("nan"))
                spread_pct = spread * 100.0

                # Show trend strength as a simple line on the right axis.
                fig.update_layout(
                    yaxis2=dict(
                        title="Spread %",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                        rangemode="tozero",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_tf.index,
                        y=spread_pct,
                        mode="lines",
                        name="MA spread %",
                        yaxis="y2",
                        line=dict(color="rgba(0, 0, 0, 0.55)", width=1),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_tf.index,
                        y=[thr * 100.0] * len(df_tf.index),
                        mode="lines",
                        name="Spread threshold",
                        yaxis="y2",
                        line=dict(color="rgba(0, 0, 0, 0.35)", width=1, dash="dot"),
                    )
                )
                bull = (fast > slow) & (spread >= thr)
                bear = (fast < slow) & (spread >= thr)
                for x0, x1 in _segments(bull):
                    fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(0, 200, 0, 0.08)", line_width=0)
                    shaded_any = True
                for x0, x1 in _segments(bear):
                    fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(200, 0, 0, 0.08)", line_width=0)
                    shaded_any = True
                msg_bits.append("Shading: direction + spread>=threshold; line: spread%")

            elif primary_context_type == "atr_pct":
                atr_len = int(rule0.get("atr_len") or 14)
                op = str(rule0.get("op") or ">")
                thr = float(rule0.get("threshold") or 0.005)
                prev_close = close.shift(1)
                tr = pd.concat(
                    [
                        (high - low),
                        (high - prev_close).abs(),
                        (low - prev_close).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                atr = tr.rolling(atr_len).mean()
                atr_pct = atr / close.replace(0.0, float("nan"))

                # Show volatility as ATR% line on the right axis.
                atr_pct_100 = atr_pct * 100.0
                fig.update_layout(
                    yaxis2=dict(
                        title="ATR %",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                        rangemode="tozero",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_tf.index,
                        y=atr_pct_100,
                        mode="lines",
                        name=f"ATR% ({atr_len})",
                        yaxis="y2",
                        line=dict(color="rgba(0, 120, 255, 0.65)", width=1),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_tf.index,
                        y=[thr * 100.0] * len(df_tf.index),
                        mode="lines",
                        name="ATR% threshold",
                        yaxis="y2",
                        line=dict(color="rgba(0, 120, 255, 0.35)", width=1, dash="dot"),
                    )
                )
                if op == ">":
                    ok = atr_pct > thr
                elif op == ">=":
                    ok = atr_pct >= thr
                elif op == "<":
                    ok = atr_pct < thr
                elif op == "<=":
                    ok = atr_pct <= thr
                else:
                    ok = atr_pct > thr
                for x0, x1 in _segments(ok):
                    fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(0, 120, 255, 0.08)", line_width=0)
                    shaded_any = True
                msg_bits.append(f"Shading: ATR% {op} {thr:g}; line: ATR%")

        else:
            msg_bits.append("No context overlay for selected type")

        if not shaded_any:
            msg_bits.append("(no shaded regime detected in window)")

        rel = None
        try:
            rel = str(dataset_path.relative_to(REPO_ROOT))
        except Exception:
            rel = str(dataset_path)

        message = f"Data: {rel} • TF: {target_tf} • " + "; ".join(msg_bits)

        import json

        fig_json = json.loads(fig.to_json())
        return JSONResponse({"ok": True, "message": message, "fig": fig_json})

    except Exception as e:
        return JSONResponse({"ok": False, "message": f"Context preview failed: {type(e).__name__}: {e}"})


@app.post("/api/guided/builder_v2/setup_visual", response_class=JSONResponse)
async def api_guided_builder_v2_setup_visual(request: Request) -> JSONResponse:
    """Full setup preview (context + signals + triggers) for Builder v2 Step 4.

    Returns a Plotly figure JSON payload with:
      - entry timeframe candles
      - context shading (per-side)
      - markers where ALL(Context+Signals+Triggers) is true
    """

    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}

    dataset_path = _find_latest_gui_dataset(REPO_ROOT)
    if dataset_path is None:
        return JSONResponse(
            {
                "ok": False,
                "message": "No dataset found under data/logs/gui_runs/. Run a backtest from the GUI (with an uploaded dataset) to enable this preview.",
            }
        )

    # Normalize a preview base (mirrors /api/guided/builder_v2/preview)
    entry_tf = str(payload.get("entry_tf") or "1h")
    base: dict[str, Any] = {
        "entry_tf": entry_tf,
        "context_tf": str(payload.get("context_tf") or ""),
        "primary_context_type": str(payload.get("primary_context_type") or "ma_stack"),
        "align_with_context": bool(payload.get("align_with_context")),
        "long_enabled": bool(payload.get("long_enabled")),
        "short_enabled": bool(payload.get("short_enabled")),
        "ma_type": str(payload.get("ma_type") or "ema"),
        "ma_fast": payload.get("ma_fast") or 20,
        "ma_mid": payload.get("ma_mid") or 50,
        "ma_slow": payload.get("ma_slow") or 200,
        "stack_mode": str(payload.get("stack_mode") or "none"),
        "slope_mode": str(payload.get("slope_mode") or "none"),
        "slope_lookback": payload.get("slope_lookback") or 10,
        "min_ma_dist_pct": payload.get("min_ma_dist_pct") or 0,
        "trigger_type": str(payload.get("trigger_type") or "pin_bar"),
        "pin_wick_body": payload.get("pin_wick_body") or 2.0,
        "pin_opp_wick_body_max": payload.get("pin_opp_wick_body_max") or 1.0,
        "pin_min_body_pct": payload.get("pin_min_body_pct") or 0.2,
        "trigger_ma_type": str(payload.get("trigger_ma_type") or "ema"),
        "trigger_ma_len": payload.get("trigger_ma_len") or 20,
        "trigger_don_len": payload.get("trigger_don_len") or 20,
        "trigger_range_len": payload.get("trigger_range_len") or 20,
        "trigger_atr_len": payload.get("trigger_atr_len") or 14,
        "trigger_atr_mult": payload.get("trigger_atr_mult") or 2.0,
        "trigger_custom_bull_expr": str(payload.get("trigger_custom_bull_expr") or ""),
        "trigger_custom_bear_expr": str(payload.get("trigger_custom_bear_expr") or ""),
        "context_rules": payload.get("context_rules") if isinstance(payload.get("context_rules"), list) else [],
        "signal_rules": payload.get("signal_rules") if isinstance(payload.get("signal_rules"), list) else [],
        "trigger_rules": payload.get("trigger_rules") if isinstance(payload.get("trigger_rules"), list) else [],
    }

    # Numeric normalization
    for k in (
        "ma_fast",
        "ma_mid",
        "ma_slow",
        "slope_lookback",
        "trigger_ma_len",
        "trigger_don_len",
        "trigger_range_len",
        "trigger_atr_len",
    ):
        try:
            base[k] = int(base[k])
        except Exception:
            pass
    for k in (
        "min_ma_dist_pct",
        "pin_wick_body",
        "pin_opp_wick_body_max",
        "pin_min_body_pct",
        "trigger_atr_mult",
    ):
        try:
            base[k] = float(base[k])
        except Exception:
            pass

    try:
        from adapters.data.data_loader import DataLoader

        loader = DataLoader()
        df_raw = loader.load(dataset_path)
    except Exception as e:
        return JSONResponse({"ok": False, "message": f"Failed to load dataset: {type(e).__name__}: {e}"})

    try:
        import json
        import pandas as pd
        import plotly.graph_objects as go

        from research.dsl import EvalContext, IndicatorRequest, compile_condition, extract_indicator_requests, extract_timeframe_refs

        if not isinstance(df_raw.index, pd.DatetimeIndex):
            return JSONResponse({"ok": False, "message": "Dataset index is not datetime; cannot render preview."})
        for col in ("open", "high", "low", "close"):
            if col not in df_raw.columns:
                return JSONResponse({"ok": False, "message": f"Dataset missing required column: {col}"})
        if "volume" not in df_raw.columns:
            df_raw = df_raw.copy()
            df_raw["volume"] = 0.0

        preview = _builder_v2_preview(base)

        def _exprs(side: dict[str, Any], key: str) -> list[str]:
            out: list[str] = []
            for item in (side.get(key) or []):
                if isinstance(item, dict) and item.get("expr"):
                    out.append(str(item.get("expr")))
            return out

        long_side = preview.get("long") if isinstance(preview.get("long"), dict) else {}
        short_side = preview.get("short") if isinstance(preview.get("short"), dict) else {}

        long_enabled = bool(long_side.get("enabled"))
        short_enabled = bool(short_side.get("enabled"))

        long_ctx_exprs = _exprs(long_side, "context")
        long_sig_exprs = _exprs(long_side, "signals")
        long_trg_exprs = _exprs(long_side, "triggers")
        short_ctx_exprs = _exprs(short_side, "context")
        short_sig_exprs = _exprs(short_side, "signals")
        short_trg_exprs = _exprs(short_side, "triggers")

        exprs_all: list[str] = []
        if long_enabled:
            exprs_all.extend(long_ctx_exprs)
            exprs_all.extend(long_sig_exprs)
            exprs_all.extend(long_trg_exprs)
        if short_enabled:
            exprs_all.extend(short_ctx_exprs)
            exprs_all.extend(short_sig_exprs)
            exprs_all.extend(short_trg_exprs)

        if not exprs_all:
            return JSONResponse({"ok": False, "message": "No rules enabled to preview (enable LONG/SHORT and add rules)."})

        # Determine timeframes needed by the expressions.
        tfs: set[str] = set()
        for expr in exprs_all:
            try:
                tfs |= extract_timeframe_refs(expr, default_tf=entry_tf)
            except Exception:
                tfs.add(entry_tf)

        # Build df_by_tf by resampling raw data.
        df_by_tf: dict[str, pd.DataFrame] = {}
        try:
            from engine.resampler import resample_ohlcv

            for tf in sorted(tfs):
                if tf == entry_tf:
                    df_tf = resample_ohlcv(df_raw[["open", "high", "low", "close", "volume"]], tf)
                else:
                    df_tf = resample_ohlcv(df_raw[["open", "high", "low", "close", "volume"]], tf)
                df_tf = df_tf.dropna(subset=["open", "high", "low", "close"], how="any")
                df_by_tf[tf] = df_tf
        except Exception as e:
            return JSONResponse({"ok": False, "message": f"Failed to resample data for preview: {type(e).__name__}: {e}"})

        if entry_tf not in df_by_tf or len(df_by_tf[entry_tf]) < 50:
            return JSONResponse({"ok": False, "message": "Not enough rows on entry timeframe to render setup preview."})

        # Window the entry TF for charting; then window other TFs to cover the same span.
        df_entry_full = df_by_tf[entry_tf]
        df_entry = df_entry_full.tail(400)
        entry_index = df_entry.index
        start_ts = entry_index.min()
        end_ts = entry_index.max()

        for tf, dft in list(df_by_tf.items()):
            try:
                df_by_tf[tf] = dft.loc[(dft.index >= start_ts) & (dft.index <= end_ts)].copy()
            except Exception:
                df_by_tf[tf] = dft.copy()

        # Precompute indicator columns needed by all expressions.
        reqs: list[IndicatorRequest] = []
        for expr in exprs_all:
            try:
                reqs.extend(extract_indicator_requests(expr, tf=entry_tf))
            except Exception:
                continue

        def _series(df: pd.DataFrame, name: str) -> pd.Series:
            if name in df.columns:
                return df[name]
            if name == "hl2":
                s = (df["high"] + df["low"]) / 2
                s.name = "hl2"
                return s
            if name == "hlc3":
                s = (df["high"] + df["low"] + df["close"]) / 3
                s.name = "hlc3"
                return s
            if name == "ohlc4":
                s = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
                s.name = "ohlc4"
                return s
            raise ValueError(f"unknown series: {name}")

        def _ema(series: pd.Series, length: int) -> pd.Series:
            return series.ewm(span=max(1, int(length)), adjust=False).mean()

        def _sma(series: pd.Series, length: int) -> pd.Series:
            return series.rolling(window=max(1, int(length))).mean()

        def _rsi(series: pd.Series, length: int) -> pd.Series:
            n = max(1, int(length))
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0.0, float("nan"))
            return 100.0 - (100.0 / (1.0 + rs))

        def _atr(df: pd.DataFrame, length: int) -> pd.Series:
            n = max(1, int(length))
            prev_close = df["close"].shift(1)
            tr = pd.concat(
                [
                    (df["high"] - df["low"]),
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            return tr.rolling(n).mean()

        def _apply_request(r: IndicatorRequest) -> None:
            if r.tf not in df_by_tf:
                return
            df = df_by_tf[r.tf]
            if r.name in {"ema", "sma", "rsi"}:
                series_key = str(r.args[0])
                length = int(r.args[1])
                s = _series(df, series_key)
                if r.name == "ema":
                    df[f"ema_{series_key}_{length}"] = _ema(s, length)
                elif r.name == "sma":
                    df[f"sma_{series_key}_{length}"] = _sma(s, length)
                else:
                    df[f"rsi_{series_key}_{length}"] = _rsi(s, length)
            elif r.name == "atr":
                length = int(r.args[0])
                df[f"atr_{length}"] = _atr(df, length)
            elif r.name == "donchian_high":
                length = int(r.args[0])
                df[f"donchian_high_{length}"] = df["high"].rolling(length, min_periods=1).max()
            elif r.name == "donchian_low":
                length = int(r.args[0])
                df[f"donchian_low_{length}"] = df["low"].rolling(length, min_periods=1).min()

        # Deduplicate
        seen_req = set()
        for r in reqs:
            key = (r.name, r.tf, r.args)
            if key in seen_req:
                continue
            seen_req.add(key)
            _apply_request(r)

        # Evaluate expressions into masks aligned to entry index.
        ctx = EvalContext(df_by_tf=df_by_tf, tf=entry_tf, target_index=entry_index)

        unsupported: list[str] = []

        def _eval_expr(expr: str) -> pd.Series:
            try:
                fn = compile_condition(expr)
                out = fn(ctx)
                if not isinstance(out, pd.Series):
                    out = pd.Series([bool(out)] * len(entry_index), index=entry_index)
                out = out.reindex(entry_index, fill_value=False)
                return out.fillna(False).astype(bool)
            except Exception:
                unsupported.append(expr)
                return pd.Series([False] * len(entry_index), index=entry_index)

        def _and_all(exprs: list[str]) -> pd.Series:
            if not exprs:
                return pd.Series([True] * len(entry_index), index=entry_index)
            out = _eval_expr(exprs[0])
            for e in exprs[1:]:
                out = out & _eval_expr(e)
            return out

        long_ctx = _and_all(long_ctx_exprs) if long_enabled else pd.Series([False] * len(entry_index), index=entry_index)
        long_sig = _and_all(long_sig_exprs) if long_enabled else pd.Series([True] * len(entry_index), index=entry_index)
        long_trg = _and_all(long_trg_exprs) if long_enabled else pd.Series([True] * len(entry_index), index=entry_index)
        long_all = (long_ctx & long_sig & long_trg) if long_enabled else pd.Series([False] * len(entry_index), index=entry_index)

        short_ctx = _and_all(short_ctx_exprs) if short_enabled else pd.Series([False] * len(entry_index), index=entry_index)
        short_sig = _and_all(short_sig_exprs) if short_enabled else pd.Series([True] * len(entry_index), index=entry_index)
        short_trg = _and_all(short_trg_exprs) if short_enabled else pd.Series([True] * len(entry_index), index=entry_index)
        short_all = (short_ctx & short_sig & short_trg) if short_enabled else pd.Series([False] * len(entry_index), index=entry_index)

        def _segments(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
            mask_b = mask.fillna(False).astype(bool)
            segs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            start = None
            for ts, on in mask_b.items():
                if on and start is None:
                    start = ts
                if (not on) and start is not None:
                    segs.append((start, ts))
                    start = None
            if start is not None:
                segs.append((start, mask_b.index[-1]))
            return segs

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=entry_index,
                    open=df_entry["open"],
                    high=df_entry["high"],
                    low=df_entry["low"],
                    close=df_entry["close"],
                    name="Price",
                )
            ]
        )
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=18, b=10),
            xaxis=dict(rangeslider=dict(visible=False)),
            yaxis=dict(title=None),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )

        # Add primary context overlays so the user can visually understand the shading/markers.
        pct = str(base.get("primary_context_type") or "ma_stack")
        ctx_tf = str(base.get("context_tf") or "").strip() or entry_tf
        df_ctx = df_by_tf.get(ctx_tf, df_entry)

        if ctx_tf != entry_tf:
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.01,
                y=0.99,
                text=f"Context TF: {ctx_tf} → {entry_tf} (forward-filled)",
                showarrow=False,
                align="left",
                font=dict(size=11, color="rgba(0,0,0,0.75)"),
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
                borderpad=4,
            )

        def _ma(series: pd.Series, length: int, kind: str) -> pd.Series:
            n = max(1, int(length))
            if kind == "sma":
                return series.rolling(n).mean()
            return series.ewm(span=n, adjust=False).mean()

        try:
            if pct == "ma_stack":
                ma_kind = str(base.get("ma_type") or "ema").lower()
                f = _ma(df_ctx["close"].astype(float), int(base.get("ma_fast") or 20), ma_kind)
                m = _ma(df_ctx["close"].astype(float), int(base.get("ma_mid") or 50), ma_kind)
                s = _ma(df_ctx["close"].astype(float), int(base.get("ma_slow") or 200), ma_kind)
                # Align to entry index for plotting
                f = f.reindex(entry_index, method="ffill")
                m = m.reindex(entry_index, method="ffill")
                s = s.reindex(entry_index, method="ffill")
                fig.add_trace(go.Scatter(x=entry_index, y=f, mode="lines", name=f"{ma_kind.upper()} {int(base.get('ma_fast') or 20)}", line=dict(width=1)))
                fig.add_trace(go.Scatter(x=entry_index, y=m, mode="lines", name=f"{ma_kind.upper()} {int(base.get('ma_mid') or 50)}", line=dict(width=1)))
                fig.add_trace(go.Scatter(x=entry_index, y=s, mode="lines", name=f"{ma_kind.upper()} {int(base.get('ma_slow') or 200)}", line=dict(width=1)))

            elif pct == "price_vs_ma":
                # Use the first context rule if available.
                rules = base.get("context_rules") if isinstance(base.get("context_rules"), list) else []
                r0 = rules[0] if rules and isinstance(rules[0], dict) else {}
                ma_kind = str(r0.get("ma_type") or "ema").lower()
                length = int(r0.get("length") or 200)
                ma = _ma(df_ctx["close"].astype(float), length, ma_kind).reindex(entry_index, method="ffill")
                fig.add_trace(go.Scatter(x=entry_index, y=ma, mode="lines", name=f"{ma_kind.upper()} {length}", line=dict(width=1)))

            elif pct == "ma_cross_state":
                rules = base.get("context_rules") if isinstance(base.get("context_rules"), list) else []
                r0 = rules[0] if rules and isinstance(rules[0], dict) else {}
                ma_kind = str(r0.get("ma_type") or "ema").lower()
                fast_len = int(r0.get("fast") or 20)
                slow_len = int(r0.get("slow") or 50)
                fast = _ma(df_ctx["close"].astype(float), fast_len, ma_kind).reindex(entry_index, method="ffill")
                slow = _ma(df_ctx["close"].astype(float), slow_len, ma_kind).reindex(entry_index, method="ffill")
                fig.add_trace(go.Scatter(x=entry_index, y=fast, mode="lines", name=f"{ma_kind.upper()} {fast_len}", line=dict(width=1)))
                fig.add_trace(go.Scatter(x=entry_index, y=slow, mode="lines", name=f"{ma_kind.upper()} {slow_len}", line=dict(width=1)))

            elif pct == "structure_breakout_state":
                rules = base.get("context_rules") if isinstance(base.get("context_rules"), list) else []
                r0 = rules[0] if rules and isinstance(rules[0], dict) else {}
                length = int(r0.get("length") or 20)
                c = df_ctx["close"].astype(float)
                basis = c.rolling(length).mean().reindex(entry_index, method="ffill")
                dev = c.rolling(length).std().reindex(entry_index, method="ffill")
                mult = 2.0
                upper = basis + mult * dev
                lower = basis - mult * dev
                fig.add_trace(go.Scatter(x=entry_index, y=lower, mode="lines", name=f"BB lower ({length})", line=dict(color="rgba(0,0,0,0.28)", width=1)))
                fig.add_trace(
                    go.Scatter(
                        x=entry_index,
                        y=upper,
                        mode="lines",
                        name=f"BB upper ({length})",
                        line=dict(color="rgba(0,0,0,0.28)", width=1),
                        fill="tonexty",
                        fillcolor="rgba(0,0,0,0.05)",
                    )
                )

            elif pct == "atr_pct":
                rules = base.get("context_rules") if isinstance(base.get("context_rules"), list) else []
                r0 = rules[0] if rules and isinstance(rules[0], dict) else {}
                atr_len = int(r0.get("atr_len") or 14)
                thr = float(r0.get("threshold") or 0.005)
                atr = _atr(df_ctx, atr_len).reindex(entry_index, method="ffill")
                atrp = (atr / df_entry["close"].astype(float).replace(0.0, float("nan"))) * 100.0
                fig.update_layout(yaxis2=dict(title="ATR %", overlaying="y", side="right", showgrid=False, rangemode="tozero"))
                fig.add_trace(go.Scatter(x=entry_index, y=atrp, mode="lines", name=f"ATR% ({atr_len})", yaxis="y2", line=dict(color="rgba(0, 120, 255, 0.6)", width=1)))
                fig.add_trace(go.Scatter(x=entry_index, y=[thr * 100.0] * len(entry_index), mode="lines", name="ATR% threshold", yaxis="y2", line=dict(color="rgba(0, 120, 255, 0.35)", width=1, dash="dot")))

            elif pct == "ma_spread_pct":
                rules = base.get("context_rules") if isinstance(base.get("context_rules"), list) else []
                r0 = rules[0] if rules and isinstance(rules[0], dict) else {}
                ma_kind = str(r0.get("ma_type") or "ema").lower()
                fast_len = int(r0.get("fast") or 20)
                slow_len = int(r0.get("slow") or 200)
                thr = float(r0.get("threshold") or 0.01)
                fast = _ma(df_ctx["close"].astype(float), fast_len, ma_kind).reindex(entry_index, method="ffill")
                slow = _ma(df_ctx["close"].astype(float), slow_len, ma_kind).reindex(entry_index, method="ffill")
                fig.add_trace(go.Scatter(x=entry_index, y=fast, mode="lines", name=f"{ma_kind.upper()} {fast_len}", line=dict(width=1)))
                fig.add_trace(go.Scatter(x=entry_index, y=slow, mode="lines", name=f"{ma_kind.upper()} {slow_len}", line=dict(width=1)))
                spread_pct = ((fast - slow).abs() / slow.abs().replace(0.0, float("nan"))) * 100.0
                fig.update_layout(yaxis2=dict(title="Spread %", overlaying="y", side="right", showgrid=False, rangemode="tozero"))
                fig.add_trace(go.Scatter(x=entry_index, y=spread_pct, mode="lines", name="MA spread %", yaxis="y2", line=dict(color="rgba(0,0,0,0.55)", width=1)))
                fig.add_trace(go.Scatter(x=entry_index, y=[thr * 100.0] * len(entry_index), mode="lines", name="Spread threshold", yaxis="y2", line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot")))
        except Exception:
            # Overlays are best-effort; the shading + markers still convey setup.
            pass

        # Regime shading (context only)
        for x0, x1 in _segments(long_ctx):
            fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(0, 200, 0, 0.08)", line_width=0)
        for x0, x1 in _segments(short_ctx):
            fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(200, 0, 0, 0.08)", line_width=0)

        # Setup markers (ALL)
        # Optional layers: signals-only / triggers-only (hidden by default; toggle in legend)
        if long_enabled and long_sig_exprs:
            idx = entry_index[long_sig.values]
            if len(idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=idx,
                        y=df_entry.loc[idx, "close"],
                        mode="markers",
                        name="LONG signals (all)",
                        visible="legendonly",
                        marker=dict(symbol="x", size=7, color="rgba(50, 110, 255, 0.55)"),
                    )
                )
        if long_enabled and long_trg_exprs:
            idx = entry_index[long_trg.values]
            if len(idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=idx,
                        y=df_entry.loc[idx, "close"],
                        mode="markers",
                        name="LONG triggers (all)",
                        visible="legendonly",
                        marker=dict(symbol="circle-open", size=7, color="rgba(0, 160, 140, 0.6)", line=dict(width=1)),
                    )
                )
        if short_enabled and short_sig_exprs:
            idx = entry_index[short_sig.values]
            if len(idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=idx,
                        y=df_entry.loc[idx, "close"],
                        mode="markers",
                        name="SHORT signals (all)",
                        visible="legendonly",
                        marker=dict(symbol="x", size=7, color="rgba(255, 140, 0, 0.55)"),
                    )
                )
        if short_enabled and short_trg_exprs:
            idx = entry_index[short_trg.values]
            if len(idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=idx,
                        y=df_entry.loc[idx, "close"],
                        mode="markers",
                        name="SHORT triggers (all)",
                        visible="legendonly",
                        marker=dict(symbol="circle-open", size=7, color="rgba(200, 0, 160, 0.55)", line=dict(width=1)),
                    )
                )

        if long_enabled:
            idx = entry_index[long_all.values]
            if len(idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=idx,
                        y=df_entry.loc[idx, "low"],
                        mode="markers",
                        name="LONG setup",
                        marker=dict(symbol="triangle-up", size=9, color="rgba(0, 140, 0, 0.9)"),
                    )
                )
        if short_enabled:
            idx = entry_index[short_all.values]
            if len(idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=idx,
                        y=df_entry.loc[idx, "high"],
                        mode="markers",
                        name="SHORT setup",
                        marker=dict(symbol="triangle-down", size=9, color="rgba(180, 0, 0, 0.9)"),
                    )
                )

        rel = None
        try:
            rel = str(dataset_path.relative_to(REPO_ROOT))
        except Exception:
            rel = str(dataset_path)

        ctx_tf_label = str(base.get("context_tf") or "").strip() or entry_tf
        if ctx_tf_label == entry_tf:
            ctx_tf_msg = "Context TF: (entry)"
        else:
            ctx_tf_msg = f"Context TF: {ctx_tf_label} (forward-filled → {entry_tf})"

        msg = (
            f"Data: {rel} • Entry TF: {entry_tf} • {ctx_tf_msg} "
            f"• LONG bars: {int(long_all.sum())} • SHORT bars: {int(short_all.sum())}"
        )
        if unsupported:
            msg += f" • ({len(unsupported)} rule(s) not visualized)"

        fig_json = json.loads(fig.to_json())
        return JSONResponse({"ok": True, "message": msg, "fig": fig_json})

    except Exception as e:
        return JSONResponse({"ok": False, "message": f"Setup preview failed: {type(e).__name__}: {e}"})


def _clarification_to_dict(c: Any) -> dict[str, Any]:
    return {
        "key": getattr(c, "key", ""),
        "question": getattr(c, "question", ""),
        "options": getattr(c, "options", None),
        "default": getattr(c, "default", None),
    }


def _extract_path(prefix: str, stdout: str) -> Optional[str]:
    # Example: "✓ Wrote spec: /path/to/file.yml"
    m = re.search(re.escape(prefix) + r"\s*(?P<p>.+)$", stdout, flags=re.MULTILINE)
    if not m:
        return None
    return m.group("p").strip()


def _extract_report_path(stdout: str) -> Optional[str]:
    m = re.search(r"Report saved to:\s*(?P<p>.+)$", stdout, flags=re.MULTILINE)
    if not m:
        return None
    return m.group("p").strip()


def _report_json_from_report_path(repo_root: Path, report_path: Optional[str]) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Best-effort load a report JSON file given a report path extracted from stdout.

    Returns:
        (json_href, json_dict)
    """
    if not report_path:
        return (None, None)

    try:
        basename = str(report_path).split("/")[-1]
        if not basename:
            return (None, None)
        if basename.endswith(".html"):
            json_name = basename[:-5] + ".json"
        elif basename.endswith(".json"):
            json_name = basename
        else:
            return (None, None)

        json_path = repo_root / "reports" / json_name
        if not json_path.exists():
            return (None, None)

        import json

        data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        return ("/reports/" + json_name, data)
    except Exception:
        return (None, None)


def _quick_diagnosis(report_json: Optional[dict[str, Any]], *, stderr: str = "") -> dict[str, Any]:
    """Return a compact diagnosis payload for UI.

    The intent is not to be 'smart'—just provide obvious next knobs.
    """

    diag: dict[str, Any] = {"headline": None, "metrics": {}, "suggestions": [], "exit_reasons": []}

    if stderr and ("Traceback" in stderr or "ERROR" in stderr.upper()):
        diag["headline"] = "Run error (check stderr)"

    if not report_json:
        if diag["headline"] is None:
            diag["headline"] = "No report JSON found"
        diag["suggestions"].append(
            {
                "label": "If this was a spec-based strategy, open the run and click Tune",
                "preset": None,
            }
        )
        return diag

    basic = report_json.get("basic_metrics") or {}
    enh = report_json.get("enhanced_metrics") or {}

    mfe_mae = enh.get("mfe_mae") or {}
    mfe_by_exit = mfe_mae.get("mfe_by_exit_reason") or {}

    exit_counts: dict[str, int] = {}
    try:
        if isinstance(mfe_by_exit, dict):
            for reason, stats in mfe_by_exit.items():
                if not isinstance(stats, dict):
                    continue
                c = stats.get("count")
                if c is None:
                    continue
                exit_counts[str(reason)] = int(c)
    except Exception:
        exit_counts = {}

    exit_total = sum(exit_counts.values())
    if exit_total > 0:
        top = sorted(exit_counts.items(), key=lambda kv: kv[1], reverse=True)[:4]
        diag["exit_reasons"] = [
            {"reason": reason, "count": count, "share": (count / exit_total)} for reason, count in top
        ]

    def _f(key: str, default: float | None = None) -> Optional[float]:
        try:
            v = basic.get(key)
            if v is None:
                v = enh.get(key)
            return float(v) if v is not None else default
        except Exception:
            return default

    total_trades = int(_f("total_trades", 0) or 0)
    total_pnl = _f("total_pnl", 0.0) or 0.0
    profit_factor = _f("profit_factor", None)
    sharpe = _f("sharpe_ratio", None)
    max_dd = _f("max_drawdown_pct", None)
    win_rate = _f("win_rate", None)
    commission = _f("total_commission", 0.0) or 0.0

    diag["metrics"] = {
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "win_rate": win_rate,
        "total_commission": commission,
    }

    # Headline
    if total_trades == 0:
        diag["headline"] = "No trades triggered"
    elif total_pnl < 0:
        diag["headline"] = "Unprofitable backtest"
    else:
        diag["headline"] = "Profitable backtest (still validate)"

    # Suggestions (actionable knobs in Tune)
    if total_trades == 0:
        diag["suggestions"].append({"label": "Disable calendar filters (common cause of zero trades)", "preset": "disable_filters"})
        diag["suggestions"].append({"label": "If still zero trades, loosen entry conditions (not in Tune yet)", "preset": None})

    if total_trades > 0 and total_trades < 30:
        diag["suggestions"].append({"label": "Consider disabling filters to increase sample size", "preset": "disable_filters"})

    if max_dd is not None and max_dd >= 20.0:
        diag["suggestions"].append({"label": "Reduce risk per trade", "preset": "lower_risk"})

    if profit_factor is not None and profit_factor < 1.0:
        diag["suggestions"].append({"label": "Try widening the stop (reduces stop-outs)", "preset": "widen_stop"})
        diag["suggestions"].append({"label": "Enable a partial exit to smooth outcomes", "preset": "enable_partial"})

    if sharpe is not None and sharpe < 0:
        diag["suggestions"].append({"label": "Disable trailing stop (if enabled) to reduce chop", "preset": "disable_trailing"})

    # Exit-reason-driven hints (uses enhanced_metrics.mfe_mae.mfe_by_exit_reason)
    def _share(reason: str) -> float:
        if exit_total <= 0:
            return 0.0
        return float(exit_counts.get(reason, 0)) / float(exit_total)

    stop_force_share = _share("stop_force")
    trailing_share = _share("trailing_stop")
    flatten_share = _share("flatten_time")

    if exit_total >= 10:
        if stop_force_share >= 0.40:
            diag["suggestions"].append({"label": "Many exits were forced stops: try widening the stop", "preset": "widen_stop"})
            if max_dd is None or max_dd >= 10.0:
                diag["suggestions"].append({"label": "If drawdown feels high, reduce risk per trade", "preset": "lower_risk"})

        if trailing_share >= 0.40 and (total_pnl < 0 or (sharpe is not None and sharpe < 0)):
            diag["suggestions"].append({"label": "Trailing-stop exits dominate in a weak run: try disabling trailing", "preset": "disable_trailing"})

        if flatten_share >= 0.40:
            diag["suggestions"].append(
                {
                    "label": "Many exits were time-flattened: consider widening sessions / holding rules (not in Tune yet)",
                    "preset": None,
                }
            )

    # Costs sanity
    try:
        if abs(total_pnl) > 0 and commission > abs(total_pnl):
            diag["suggestions"].append({"label": "Commissions dominated PnL: reduce churn (filters/timeframe/entries)", "preset": None})
    except Exception:
        pass

    return diag


def _safe_run_dir(run_name: str) -> Optional[Path]:
    # Prevent path traversal; run names are directory basenames under data/logs/gui_runs.
    if not run_name or run_name in {".", ".."}:
        return None
    if "/" in run_name or "\\" in run_name:
        return None

    base = REPO_ROOT / "data" / "logs" / "gui_runs"
    p = (base / run_name).resolve()
    try:
        base_resolved = base.resolve()
    except FileNotFoundError:
        return None
    if not str(p).startswith(str(base_resolved)):
        return None
    if not p.exists() or not p.is_dir():
        return None
    return p


def _extract_command_arg(command_text: str, flag: str) -> Optional[str]:
    """Extract a flag value from a command string written by the GUI.

    This is best-effort and assumes the command is roughly shell-like.
    """

    if not command_text:
        return None
    tokens = str(command_text).strip().split()
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def _build_backtest_argv_from_inputs(
    inputs: dict[str, Any],
    *,
    data_path: str | None = None,
) -> list[str]:
    strategy = str(inputs.get("strategy") or "").strip()
    argv: list[str] = [
        sys.executable,
        "scripts/run_backtest.py",
        "--strategy",
        strategy,
        "--auto-resample",
    ]

    spm = str(inputs.get("split_policy_mode") or "auto").strip().lower()
    if spm not in {"enforce", "auto", "none"}:
        spm = "auto"
    argv.extend(["--split-policy-mode", spm])

    spn = str(inputs.get("split_policy_name") or "").strip()
    if spn:
        argv.extend(["--split-policy-name", spn])

    if bool(inputs.get("override_split_policy")):
        argv.append("--override-split-policy")

    sd = str(inputs.get("start_date") or "").strip()
    ed = str(inputs.get("end_date") or "").strip()
    if sd:
        argv.extend(["--start-date", sd])
    if ed:
        argv.extend(["--end-date", ed])

    mkt = str(inputs.get("market") or "").strip()
    if mkt:
        argv.extend(["--market", mkt])

    prof = str(inputs.get("config_profile") or "").strip()
    if prof:
        argv.extend(["--config-profile", prof])

    if data_path:
        argv.extend(["--data", str(data_path)])

    # Slippage (optional)
    try:
        sl = inputs.get("slippage")
        if sl is not None and str(sl).strip() != "":
            argv.extend(["--slippage", str(float(sl))])
    except Exception:
        pass

    if bool(inputs.get("report_visuals")):
        argv.append("--report-visuals")

    if bool(inputs.get("download_data")) and not data_path:
        argv.append("--download-data")
        sym = str(inputs.get("download_symbol") or "BTCUSDT").strip().upper()
        interval = str(inputs.get("download_interval") or "15m").strip()
        argv.extend(["--download-symbol", sym])
        argv.extend(["--download-interval", interval])

    return argv


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    try:
        return TEMPLATES.TemplateResponse("home.html", {"request": request, "title": "TradingLab Launcher"})
    except Exception:
        return HTMLResponse("<h1>TradingLab Launcher</h1>")


@app.get("/help", response_class=HTMLResponse)
def help_page(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "help.html",
        {
            "title": "Help",
        },
    )


@app.get("/strategies", response_class=HTMLResponse)
def strategies_list(request: Request, message: str | None = None, page: int = 1, per_page: int = 50):
    page_i, per_page_i = _parse_pagination(page=page, per_page=per_page, max_per_page=200)

    all_specs = _list_strategy_specs(REPO_ROOT)
    spec_page, total_specs = _paginate_list(all_specs, page=page_i, per_page=per_page_i)
    total_pages = max(1, (total_specs + per_page_i - 1) // per_page_i)
    if page_i > total_pages:
        page_i = total_pages
        spec_page, total_specs = _paginate_list(all_specs, page=page_i, per_page=per_page_i)

    return TEMPLATES.TemplateResponse(
        request,
        "strategies.html",
        {
            "title": "Strategies",
            "spec_strategies": spec_page,
            "code_strategies": _list_code_strategies(REPO_ROOT),
            "message": message,
            "spec_page": page_i,
            "spec_per_page": per_page_i,
            "spec_total": total_specs,
            "spec_total_pages": total_pages,
        },
    )


def _safe_spec_path(spec_name: str) -> Optional[Path]:
    name = str(spec_name or "").strip()
    if not _is_safe_strategy_name(name):
        return None
    base = (REPO_ROOT / "research_specs").resolve()
    candidates = []
    for ext in (".yml", ".yaml"):
        p = (REPO_ROOT / "research_specs" / f"{name}{ext}").resolve()
        if str(p).startswith(str(base)):
            candidates.append(p)

    # Prefer whichever exists on disk.
    for p in candidates:
        if p.exists():
            return p

    # Otherwise default to .yml for new writes.
    return (REPO_ROOT / "research_specs" / f"{name}.yml").resolve() if candidates else None


def _guided_meta_path(spec_name: str) -> Optional[Path]:
    name = str(spec_name or "").strip()
    if not _is_safe_strategy_name(name):
        return None
    base = (REPO_ROOT / "research_specs" / ".ui").resolve()
    p = (REPO_ROOT / "research_specs" / ".ui" / f"{name}.guided.json").resolve()
    if not str(p).startswith(str(base)):
        return None
    return p


_GUIDED_META_KEYS: tuple[str, ...] = (
    "template",
    "entry_tf",
    "asset_class",
    "exchange",
    "market_type",
    "symbol",
    "risk_per_trade_pct",
    "sizing_mode",
    "account_size",
    "max_trades_per_day",
    "max_daily_loss_pct",
    "commissions",
    "slippage_ticks",
    "holding_style",
    "flatten_enabled",
    "flatten_time",
    "stop_signal_search_minutes_before",
    "use_intraday_margin",
    "long_enabled",
    "short_enabled",
    "stop_separate",
    "stop_type",
    "stop_percent",
    "atr_length",
    "atr_multiplier",
    "stop_ma_type",
    "stop_ma_length",
    "stop_buffer",
    "stop_structure_lookback_bars",
    "stop_candle_bars_back",
    "long_stop_type",
    "long_stop_percent",
    "long_atr_length",
    "long_atr_multiplier",
    "long_stop_ma_type",
    "long_stop_ma_length",
    "long_stop_buffer",
    "long_stop_structure_lookback_bars",
    "long_stop_candle_bars_back",
    "short_stop_type",
    "short_stop_percent",
    "short_atr_length",
    "short_atr_multiplier",
    "short_stop_ma_type",
    "short_stop_ma_length",
    "short_stop_buffer",
    "short_stop_structure_lookback_bars",
    "short_stop_candle_bars_back",
    "calendar_filters_mode",
    "calendar_allowed_days",
    "calendar_sessions",
    "calendar_session_times",
    "tm_tp_mode",
    "tm_tp_template",
    "tm_tp_levels",
    "tm_tp_target_r",
    "tm_tp_exit_pct",
    "tm_partial_enabled",
    "tm_partial_template",
    "tm_partial_levels",
    "tm_partial_level_r",
    "tm_partial_exit_pct",
    "tm_trailing_enabled",
    "tm_trailing_length",
    "tm_trailing_activation_r",
    "tm_trailing_stepped",
    "tm_trailing_min_move_pips",
    # Builder v2 fields
    "context_tf",
    "signal_tf",
    "trigger_tf",
    "align_with_context",
    "primary_context_type",
    "ma_type",
    "ma_fast",
    "ma_mid",
    "ma_slow",
    "stack_mode",
    "slope_mode",
    "slope_lookback",
    "min_ma_dist_pct",
    "trigger_type",
    "pin_wick_body",
    "pin_opp_wick_body_max",
    "pin_min_body_pct",
    "trigger_ma_type",
    "trigger_ma_len",
    "trigger_don_len",
    "trigger_range_len",
    "trigger_atr_len",
    "trigger_atr_mult",
    "trigger_custom_bull_expr",
    "trigger_custom_bear_expr",
    "context_rules",
    "signal_rules",
    "trigger_rules",

    # Phase 4 execution options
    "exit_ctx_enabled",
    "exit_ctx_mode",
)


def _guided_meta_from_draft(draft: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"version": 1}
    # Store a whitelisted subset only (avoid leaking huge/unrelated data).
    for k in _GUIDED_META_KEYS:
        if k in draft:
            out[k] = draft.get(k)
    # Also store computed conditions so legacy templates can round-trip.
    if "long_conds" in draft:
        out["long_conds"] = list(draft.get("long_conds") or [])
    if "short_conds" in draft:
        out["short_conds"] = list(draft.get("short_conds") or [])
    return out


def _write_guided_meta(*, spec_name: str, draft: dict[str, Any]) -> None:
    p = _guided_meta_path(spec_name)
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_guided_meta_from_draft(draft), indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        # UI metadata is best-effort.
        return


def _read_guided_meta(spec_name: str) -> dict[str, Any] | None:
    p = _guided_meta_path(spec_name)
    if p is None or not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _safe_generated_strategy_dir(strategy_name: str) -> Path | None:
    """Return strategies/<name> dir if it is inside repo and exists."""

    name = str(strategy_name or "").strip()
    if not _is_safe_strategy_name(name):
        return None
    base = (REPO_ROOT / "strategies").resolve()
    p = (REPO_ROOT / "strategies" / name).resolve()
    if not str(p).startswith(str(base)):
        return None
    if not p.exists() or not p.is_dir():
        return None
    return p


def _looks_like_generated_strategy_dir(p: Path) -> bool:
    try:
        if not (p / "config.yml").exists():
            return False
        if not (p / "strategy.py").exists():
            return False
        readme = p / "README.md"
        if readme.exists():
            txt = readme.read_text(encoding="utf-8", errors="replace").lower()
            if "generated by the tradinglab research layer" in txt:
                return True
        # If README missing, still treat as generated if core artifacts exist.
        return True
    except Exception:
        return False


def _relpath_for_ui(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except Exception:
        return str(p)


@app.get("/strategies/spec/{spec_name}", response_class=HTMLResponse)
def strategy_spec_detail(request: Request, spec_name: str, message: str | None = None):
    spec_path = _safe_spec_path(spec_name)
    if spec_path is None or not spec_path.exists():
        return RedirectResponse(url="/strategies", status_code=303)

    try:
        yaml_text = spec_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        yaml_text = ""

    meta = _read_guided_meta(spec_name) or {}
    guided_template = None
    try:
        guided_template = str(meta.get("template") or "").strip().lower() or None
    except Exception:
        guided_template = None

    return TEMPLATES.TemplateResponse(
        request,
        "strategy_spec.html",
        {
            "title": f"Spec: {spec_name}",
            "spec_name": spec_name,
            "spec_rel_path": str(spec_path.relative_to(REPO_ROOT)),
            "yaml_text": yaml_text,
            "message": message,
            "has_guided_meta": bool(guided_template),
            "guided_template": guided_template,
            "has_v2_guided_meta": guided_template == "builder_v2",
        },
    )


@app.get("/strategies/spec/{spec_name}/delete", response_class=HTMLResponse)
def strategy_spec_delete_confirm(request: Request, spec_name: str):
    """Two-step delete: show a preview of what will be deleted, then POST to execute."""

    spec_path = _safe_spec_path(spec_name)
    if spec_path is None or not spec_path.exists():
        return RedirectResponse(url="/strategies", status_code=303)

    meta_p = _guided_meta_path(spec_name)
    meta_exists = bool(meta_p and meta_p.exists())

    compiled_dir = _safe_generated_strategy_dir(spec_name)
    compiled_exists = bool(compiled_dir and compiled_dir.exists())
    compiled_is_generated = bool(compiled_dir and _looks_like_generated_strategy_dir(compiled_dir))

    delete_targets: list[dict[str, Any]] = []
    delete_targets.append(
        {
            "label": "Spec file",
            "path": _relpath_for_ui(spec_path),
            "exists": True,
            "will_delete": True,
            "note": "Strategy definition (YAML spec)",
        }
    )

    if meta_p is not None:
        delete_targets.append(
            {
                "label": "Guided UI metadata",
                "path": _relpath_for_ui(meta_p),
                "exists": meta_exists,
                "will_delete": meta_exists,
                "note": "Only present for strategies created via the guided builder",
            }
        )

    if compiled_dir is not None:
        note = "Generated strategy code (can be rebuilt from the spec)" if compiled_is_generated else "Not recognized as generated; will not be deleted"
        delete_targets.append(
            {
                "label": "Compiled strategy folder",
                "path": _relpath_for_ui(compiled_dir),
                "exists": compiled_exists,
                "will_delete": bool(compiled_exists and compiled_is_generated),
                "note": note,
            }
        )

    will_not_delete = [
        {"label": "Past runs / datasets / logs", "path": "data/logs/", "note": "Preserved"},
        {"label": "Reports", "path": "reports/", "note": "Preserved"},
        {"label": "Run bundles", "path": "data/logs/gui_runs/.../bundle.zip", "note": "Preserved"},
    ]

    return TEMPLATES.TemplateResponse(
        request,
        "strategy_delete_confirm.html",
        {
            "title": f"Delete: {spec_name}",
            "spec_name": spec_name,
            "delete_targets": delete_targets,
            "will_not_delete": will_not_delete,
            "compiled_exists": compiled_exists,
            "compiled_is_generated": compiled_is_generated,
        },
    )


@app.post("/strategies/spec/{spec_name}/delete", response_class=HTMLResponse)
async def strategy_spec_delete(request: Request, spec_name: str):
    spec_path = _safe_spec_path(spec_name)
    if spec_path is None or not spec_path.exists():
        return RedirectResponse(url="/strategies", status_code=303)

    form = await request.form()
    confirm_name = str(form.get("confirm_name") or "").strip()
    delete_compiled = str(form.get("delete_compiled") or "0").strip() == "1"

    if confirm_name != spec_name:
        return RedirectResponse(
            url=f"/strategies/spec/{spec_name}?message=Type+the+strategy+name+to+confirm+delete",
            status_code=303,
        )

    # 1) Delete spec file
    try:
        spec_path.unlink(missing_ok=True)
    except Exception:
        return RedirectResponse(
            url=f"/strategies/spec/{spec_name}?message=Failed+to+delete+spec",
            status_code=303,
        )

    # 2) Delete guided UI metadata sidecar (best effort)
    try:
        meta_p = _guided_meta_path(spec_name)
        if meta_p is not None:
            meta_p.unlink(missing_ok=True)
    except Exception:
        pass

    msg_parts: list[str] = ["Deleted+spec"]

    # 3) Optional: delete compiled generated strategy folder
    if delete_compiled:
        strat_dir = _safe_generated_strategy_dir(spec_name)
        if strat_dir is None:
            msg_parts.append("compiled+folder+not+found")
        else:
            if _looks_like_generated_strategy_dir(strat_dir):
                try:
                    shutil.rmtree(strat_dir)
                    msg_parts.append("deleted+compiled")
                except Exception:
                    msg_parts.append("failed+to+delete+compiled")
            else:
                msg_parts.append("compiled+folder+not+deleted+(not+recognized+as+generated)")

    return RedirectResponse(url=f"/strategies?message={'%2C+'.join(msg_parts)}", status_code=303)


@app.get("/strategies/spec/{spec_name}/edit-guided", response_class=HTMLResponse)
def strategy_spec_edit_guided(request: Request, spec_name: str):
    """Reopen the guided wizard in edit mode, prepopulated with the choices the user originally made.

    This only works for strategies created via the guided flow (we persist a sidecar .guided.json).
    """

    spec_path = _safe_spec_path(spec_name)
    if spec_path is None or not spec_path.exists():
        return RedirectResponse(url="/strategies", status_code=303)

    import yaml

    try:
        spec_data = yaml.safe_load(spec_path.read_text(encoding="utf-8", errors="replace")) or {}
        spec = StrategySpec.model_validate(spec_data)
    except Exception:
        return RedirectResponse(url=f"/strategies/spec/{spec_name}?message=Invalid+spec", status_code=303)

    meta = _read_guided_meta(spec.name)
    if not meta:
        return RedirectResponse(
            url=f"/strategies/spec/{spec_name}/form?message=No+guided+edit+data+found+for+this+spec",
            status_code=303,
        )

    template = str(meta.get("template") or "custom").strip().lower()
    if template != "builder_v2":
        return RedirectResponse(
            url=f"/strategies/spec/{spec_name}/form?message=Guided+edit+is+available+only+for+Builder+v2+specs.+Use+Edit+(advanced)+to+edit+YAML",
            status_code=303,
        )

    # Build an edit draft by combining: (guided defaults) <- (meta) <- (spec as source of truth)
    draft_id = str(uuid.uuid4())
    draft: dict[str, Any] = {
        "kind": "guided",
        "name": str(spec.name),
        "old_spec_name": str(spec.name),
        "template": template,
        "defaults": _guided_defaults(template),
        "heading": "Edit Strategy (Guided)",
        "edit_mode": True,
    }

    # Apply meta fields
    for k in _GUIDED_META_KEYS:
        if k in meta:
            draft[k] = meta.get(k)

    # Apply spec fields (source of truth)
    try:
        draft["entry_tf"] = str(spec.entry_tf)
        draft["exchange"] = str(getattr(spec.market, "exchange", None) or draft.get("exchange") or "unknown")
        draft["market_type"] = str(getattr(spec.market, "market_type", None) or draft.get("market_type") or "spot")
        draft["symbol"] = str(getattr(spec.market, "symbol", None) or draft.get("symbol") or "UNKNOWN")
        draft["risk_per_trade_pct"] = float(getattr(spec, "risk_per_trade_pct", 1.0) or 1.0)
        draft["sizing_mode"] = str(getattr(spec, "sizing_mode", "account_size") or "account_size")
        draft["account_size"] = float(getattr(spec, "account_size", 10000.0) or 10000.0)
        draft["max_trades_per_day"] = int(getattr(spec, "max_trades_per_day", 1) or 1)
        draft["max_daily_loss_pct"] = getattr(spec, "max_daily_loss_pct", None)
        draft["commissions"] = getattr(spec, "commissions", None)
        draft["slippage_ticks"] = getattr(spec, "slippage_ticks", None)
        draft["context_tf"] = "" if not getattr(spec, "context_tf", None) else str(spec.context_tf)

        draft["long_enabled"] = bool(spec.long.enabled) if getattr(spec, "long", None) else False
        draft["short_enabled"] = bool(spec.short.enabled) if getattr(spec, "short", None) else False

        # Stops: if long/short stops differ, enable separate stop UI; otherwise use shared fields.
        def _stop_sig(side_obj: Any) -> dict[str, Any]:
            if side_obj is None:
                return {}
            st = str(getattr(side_obj, "stop_type", None) or "atr")
            sig: dict[str, Any] = {"stop_type": st}
            if st == "percent":
                sig["stop_percent"] = getattr(side_obj, "stop_percent", None)
            elif st == "atr":
                sig["atr_length"] = getattr(side_obj, "atr_length", None)
                sig["atr_multiplier"] = getattr(side_obj, "atr_multiplier", None)
            elif st == "ma":
                sig["stop_ma_type"] = getattr(side_obj, "stop_ma_type", None)
                sig["stop_ma_length"] = getattr(side_obj, "stop_ma_length", None)
                sig["stop_buffer"] = getattr(side_obj, "stop_buffer", None)
            elif st == "structure":
                sig["stop_structure_lookback_bars"] = getattr(side_obj, "stop_structure_lookback_bars", None)
                sig["stop_buffer"] = getattr(side_obj, "stop_buffer", None)
            elif st == "candle":
                sig["stop_candle_bars_back"] = getattr(side_obj, "stop_candle_bars_back", None)
                sig["stop_buffer"] = getattr(side_obj, "stop_buffer", None)
            return sig

        long_sig = _stop_sig(getattr(spec, "long", None))
        short_sig = _stop_sig(getattr(spec, "short", None))
        have_long = bool(getattr(spec, "long", None))
        have_short = bool(getattr(spec, "short", None))

        separate_stops = False
        if have_long and have_short and long_sig and short_sig:
            separate_stops = long_sig != short_sig

        draft["stop_separate"] = bool(separate_stops)

        if separate_stops:
            # Populate per-side fields from the spec.
            if long_sig:
                draft["long_stop_type"] = str(long_sig.get("stop_type") or "atr")
                if long_sig.get("stop_percent") is not None:
                    draft["long_stop_percent"] = float(long_sig["stop_percent"])
                if long_sig.get("atr_length") is not None:
                    draft["long_atr_length"] = int(long_sig["atr_length"])
                if long_sig.get("atr_multiplier") is not None:
                    draft["long_atr_multiplier"] = float(long_sig["atr_multiplier"])
                if long_sig.get("stop_ma_type") is not None:
                    draft["long_stop_ma_type"] = str(long_sig["stop_ma_type"])
                if long_sig.get("stop_ma_length") is not None:
                    draft["long_stop_ma_length"] = int(long_sig["stop_ma_length"])
                if long_sig.get("stop_structure_lookback_bars") is not None:
                    draft["long_stop_structure_lookback_bars"] = int(long_sig["stop_structure_lookback_bars"])
                if long_sig.get("stop_candle_bars_back") is not None:
                    draft["long_stop_candle_bars_back"] = int(long_sig["stop_candle_bars_back"])
                if long_sig.get("stop_buffer") is not None:
                    draft["long_stop_buffer"] = float(long_sig["stop_buffer"])

            if short_sig:
                draft["short_stop_type"] = str(short_sig.get("stop_type") or "atr")
                if short_sig.get("stop_percent") is not None:
                    draft["short_stop_percent"] = float(short_sig["stop_percent"])
                if short_sig.get("atr_length") is not None:
                    draft["short_atr_length"] = int(short_sig["atr_length"])
                if short_sig.get("atr_multiplier") is not None:
                    draft["short_atr_multiplier"] = float(short_sig["atr_multiplier"])
                if short_sig.get("stop_ma_type") is not None:
                    draft["short_stop_ma_type"] = str(short_sig["stop_ma_type"])
                if short_sig.get("stop_ma_length") is not None:
                    draft["short_stop_ma_length"] = int(short_sig["stop_ma_length"])
                if short_sig.get("stop_structure_lookback_bars") is not None:
                    draft["short_stop_structure_lookback_bars"] = int(short_sig["stop_structure_lookback_bars"])
                if short_sig.get("stop_candle_bars_back") is not None:
                    draft["short_stop_candle_bars_back"] = int(short_sig["stop_candle_bars_back"])
                if short_sig.get("stop_buffer") is not None:
                    draft["short_stop_buffer"] = float(short_sig["stop_buffer"])
        else:
            # Populate shared fields from whichever side is present.
            stop_side = spec.long if getattr(spec, "long", None) else spec.short
            if stop_side is not None:
                draft["stop_type"] = str(getattr(stop_side, "stop_type", None) or draft.get("stop_type") or "atr")
                if getattr(stop_side, "stop_percent", None) is not None:
                    draft["stop_percent"] = float(getattr(stop_side, "stop_percent"))
                if getattr(stop_side, "atr_length", None) is not None:
                    draft["atr_length"] = int(getattr(stop_side, "atr_length"))
                if getattr(stop_side, "atr_multiplier", None) is not None:
                    draft["atr_multiplier"] = float(getattr(stop_side, "atr_multiplier"))
                if getattr(stop_side, "stop_ma_type", None) is not None:
                    draft["stop_ma_type"] = str(getattr(stop_side, "stop_ma_type"))
                if getattr(stop_side, "stop_ma_length", None) is not None:
                    draft["stop_ma_length"] = int(getattr(stop_side, "stop_ma_length"))
                if getattr(stop_side, "stop_structure_lookback_bars", None) is not None:
                    draft["stop_structure_lookback_bars"] = int(getattr(stop_side, "stop_structure_lookback_bars"))
                if getattr(stop_side, "stop_candle_bars_back", None) is not None:
                    draft["stop_candle_bars_back"] = int(getattr(stop_side, "stop_candle_bars_back"))
                if getattr(stop_side, "stop_buffer", None) is not None:
                    draft["stop_buffer"] = float(getattr(stop_side, "stop_buffer"))

        # Execution / holding controls (best effort)
        ex = getattr(spec, "execution", None)
        if ex is not None:
            if getattr(ex, "flatten_enabled", None) is not None:
                draft["flatten_enabled"] = bool(ex.flatten_enabled)
            if getattr(ex, "flatten_time", None) is not None:
                draft["flatten_time"] = ex.flatten_time
            if getattr(ex, "stop_signal_search_minutes_before", None) is not None:
                draft["stop_signal_search_minutes_before"] = ex.stop_signal_search_minutes_before
            if getattr(ex, "use_intraday_margin", None) is not None:
                draft["use_intraday_margin"] = bool(ex.use_intraday_margin)

            # Phase 4: context invalidation behavior
            try:
                exit_ctx = getattr(ex, "exit_if_context_invalid", None)
                if exit_ctx is not None and bool(getattr(exit_ctx, "enabled", False)):
                    draft["exit_ctx_enabled"] = True
                    draft["exit_ctx_mode"] = str(getattr(exit_ctx, "mode", None) or "immediate")
            except Exception:
                pass

        # Trade management (source of truth): map spec -> guided draft keys.
        # This prevents TP/partial/trailing settings from snapping back to defaults during guided edits.
        try:
            spec_dump = spec.model_dump(exclude_none=True)
            tm = spec_dump.get("trade_management") or {}
            if isinstance(tm, dict):
                tp = tm.get("take_profit")
                if isinstance(tp, dict):
                    if tp.get("enabled") is False:
                        draft["tm_tp_mode"] = "disable"
                        draft.pop("tm_tp_levels", None)
                    else:
                        levels = tp.get("levels")
                        if isinstance(levels, list) and levels:
                            draft["tm_tp_mode"] = "custom"
                            draft["tm_tp_levels"] = levels
                            try:
                                draft["tm_tp_target_r"] = float(levels[0].get("target_r"))
                                draft["tm_tp_exit_pct"] = float(levels[0].get("exit_pct"))
                            except Exception:
                                pass

                pe = tm.get("partial_exit")
                if isinstance(pe, dict):
                    levels = pe.get("levels")
                    if isinstance(levels, list) and levels:
                        draft["tm_partial_enabled"] = True
                        draft["tm_partial_levels"] = levels
                        try:
                            draft["tm_partial_level_r"] = float(levels[0].get("level_r"))
                            draft["tm_partial_exit_pct"] = float(levels[0].get("exit_pct"))
                        except Exception:
                            pass
                    else:
                        # If explicitly disabled in spec, preserve that; otherwise leave draft value from meta/defaults.
                        if pe.get("enabled") is False:
                            draft["tm_partial_enabled"] = False
                            draft.pop("tm_partial_levels", None)

                ts = tm.get("trailing_stop")
                if isinstance(ts, dict):
                    if ts.get("enabled") is False:
                        draft["tm_trailing_enabled"] = False
                    elif ts.get("enabled") is True:
                        draft["tm_trailing_enabled"] = True
                        if ts.get("length") is not None:
                            draft["tm_trailing_length"] = int(ts.get("length"))
                        if ts.get("activation_r") is not None:
                            draft["tm_trailing_activation_r"] = float(ts.get("activation_r"))
                        if ts.get("stepped") is not None:
                            draft["tm_trailing_stepped"] = bool(ts.get("stepped"))
                        if ts.get("min_move_pips") is not None:
                            draft["tm_trailing_min_move_pips"] = float(ts.get("min_move_pips"))
        except Exception:
            pass

        # Conditions (for legacy flows): if meta stored them, keep; else pull from spec
        if "long_conds" in meta:
            draft["long_conds"] = list(meta.get("long_conds") or [])
        else:
            if getattr(spec, "long", None) is not None:
                draft["long_conds"] = [c.model_dump() for c in (spec.long.conditions_all or [])]
        if "short_conds" in meta:
            draft["short_conds"] = list(meta.get("short_conds") or [])
        else:
            if getattr(spec, "short", None) is not None:
                draft["short_conds"] = [c.model_dump() for c in (spec.short.conditions_all or [])]
    except Exception:
        pass

    # Ensure the draft looks like a normal guided draft so existing steps work.
    # Some keys are expected by templates.
    draft.setdefault("asset_class", "crypto")
    draft.setdefault("market_type", "spot")
    draft.setdefault("exchange", "binance")
    draft.setdefault("symbol", "BTCUSDT")
    draft.setdefault("entry_tf", "1h")
    draft.setdefault("holding_style", "intraday")

    DRAFTS[draft_id] = draft
    return RedirectResponse(url=f"/create-strategy-guided/step2?draft_id={draft_id}", status_code=303)


@app.get("/strategies/spec/{spec_name}/form", response_class=HTMLResponse)
def strategy_spec_form(request: Request, spec_name: str, message: str | None = None):
    spec_path = _safe_spec_path(spec_name)
    if spec_path is None or not spec_path.exists():
        return RedirectResponse(url="/strategies", status_code=303)

    import yaml

    try:
        spec_data = yaml.safe_load(spec_path.read_text(encoding="utf-8", errors="replace")) or {}
        spec = StrategySpec.model_validate(spec_data)
    except Exception:
        return RedirectResponse(url=f"/strategies/spec/{spec_name}?message=Invalid+spec", status_code=303)

    spec_dump = spec.model_dump(exclude_none=True)
    market = spec_dump.get("market") or {}

    long_side = spec_dump.get("long") or {}
    short_side = spec_dump.get("short") or {}

    long_conds = list(long_side.get("conditions_all") or [])
    short_conds = list(short_side.get("conditions_all") or [])

    # Condition builder UI: try to parse known expression shapes so the form can
    # show user-friendly fields (normal users) while preserving a custom-expr escape hatch.
    import re

    _RE_RSI = re.compile(r"^rsi\(close,\s*(\d+)\)\s*([<>]=?)\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE)
    _RE_CLOSE_VS_MA = re.compile(r"^close\s*([<>]=?)\s*(ema|sma)\(close,\s*(\d+)\)\s*$", re.IGNORECASE)
    _RE_MA_CROSS = re.compile(
        r"^crosses_(above|below)\(\s*(ema|sma)\(close,\s*(\d+)\)\s*,\s*(ema|sma)\(close,\s*(\d+)\)\s*\)\s*$",
        re.IGNORECASE,
    )
    _RE_DONCHIAN = re.compile(
        r"^crosses_(above|below)\(\s*close\s*,\s*shift\(donchian_(high|low)\((\d+)\),\s*1\)\s*\)\s*$",
        re.IGNORECASE,
    )

    def _cond_to_ui(c: dict[str, Any]) -> dict[str, Any]:
        tf = str(c.get("tf") or "").strip()
        expr = str(c.get("expr") or "").strip()

        out: dict[str, Any] = {
            "tf": tf,
            "expr": expr,
            "type": "custom",
            # MA cross
            "cross_ma": "ema",
            "cross_fast": "",
            "cross_slow": "",
            "cross_direction": "above",
            # RSI
            "rsi_len": 14,
            "rsi_op": "<",
            "rsi_level": "",
            # Close vs MA
            "cv_ma": "ema",
            "cv_len": "",
            "cv_op": ">",
            # Donchian
            "don_len": "",
            "don_direction": "breakout",
        }

        m = _RE_RSI.match(expr)
        if m:
            out["type"] = "rsi"
            out["rsi_len"] = int(m.group(1))
            out["rsi_op"] = m.group(2)
            out["rsi_level"] = m.group(3)
            return out

        m = _RE_CLOSE_VS_MA.match(expr)
        if m:
            out["type"] = "close_vs_ma"
            out["cv_op"] = m.group(1)
            out["cv_ma"] = m.group(2).lower()
            out["cv_len"] = int(m.group(3))
            return out

        m = _RE_MA_CROSS.match(expr)
        if m:
            direction = m.group(1).lower()
            ma1 = m.group(2).lower()
            fast = int(m.group(3))
            ma2 = m.group(4).lower()
            slow = int(m.group(5))
            if ma1 == ma2:
                out["type"] = "ma_cross"
                out["cross_direction"] = direction
                out["cross_ma"] = ma1
                out["cross_fast"] = fast
                out["cross_slow"] = slow
                return out

        m = _RE_DONCHIAN.match(expr)
        if m:
            above_below = m.group(1).lower()
            length = int(m.group(3))
            out["type"] = "donchian"
            out["don_len"] = length
            out["don_direction"] = "breakout" if above_below == "above" else "breakdown"
            return out

        return out

    long_conds = [_cond_to_ui(c) for c in long_conds]
    short_conds = [_cond_to_ui(c) for c in short_conds]

    # Pull a single stop config for the form (applied to enabled sides on save)
    stop_type = (long_side.get("stop_type") or short_side.get("stop_type") or "atr")
    stop_percent = (long_side.get("stop_percent") or short_side.get("stop_percent") or 0.02)
    atr_length = (long_side.get("atr_length") or short_side.get("atr_length") or 14)
    atr_multiplier = (long_side.get("atr_multiplier") or short_side.get("atr_multiplier") or 3.0)

    # Advanced sections: only show if explicitly present in YAML (or enabled later by user).
    has_risk_override = isinstance(spec_data, dict) and ("risk_per_trade_pct" in spec_data)
    has_filters = isinstance(spec_data, dict) and isinstance(spec_data.get("filters"), dict) and bool(spec_data.get("filters"))
    has_calendar_filters = has_filters and ("calendar_filters" in (spec_data.get("filters") or {}))
    has_trade_management = isinstance(spec_data, dict) and isinstance(spec_data.get("trade_management"), dict) and bool(spec_data.get("trade_management"))
    has_execution = isinstance(spec_data, dict) and isinstance(spec_data.get("execution"), dict) and bool(spec_data.get("execution"))

    # Advanced: risk
    risk_per_trade_pct = float(getattr(spec, "risk_per_trade_pct", 1.0) or 1.0)

    # Advanced: calendar filters
    master_calendar = _get_master_calendar_defaults()
    master_tm = _get_master_trade_management_defaults()

    calendar_filters = {}
    try:
        calendar_filters = (spec.filters.calendar_filters or {}) if getattr(spec, "filters", None) else {}
    except Exception:
        calendar_filters = {}

    calendar_mode = "master" if bool(calendar_filters.get("master_filters_enabled", True)) else "disable"
    day_cfg = (calendar_filters.get("day_of_week") or {}) if isinstance(calendar_filters, dict) else {}
    allowed_days = day_cfg.get("allowed_days") if isinstance(day_cfg, dict) else None
    if not isinstance(allowed_days, list) or not allowed_days:
        allowed_days = list(master_calendar.get("allowed_days") or [0, 1, 2, 3, 4])

    ts_cfg = calendar_filters.get("trading_sessions") if isinstance(calendar_filters, dict) else None
    sessions = {
        "Asia": bool(((master_calendar.get("sessions") or {}).get("Asia") or {}).get("enabled", True)),
        "London": bool(((master_calendar.get("sessions") or {}).get("London") or {}).get("enabled", True)),
        "NewYork": bool(((master_calendar.get("sessions") or {}).get("NewYork") or {}).get("enabled", True)),
    }
    if isinstance(ts_cfg, dict) and ts_cfg:
        for k in ["Asia", "London", "NewYork"]:
            v = ts_cfg.get(k)
            if isinstance(v, dict) and "enabled" in v:
                sessions[k] = bool(v.get("enabled"))

    # Advanced: trade management
    tm = getattr(spec, "trade_management", None)
    tp_mode = "master"
    tp_target_r = float((master_tm.get("take_profit") or {}).get("target_r", 3.0))
    tp_exit_pct = float((master_tm.get("take_profit") or {}).get("exit_pct", 100.0))
    partial_enabled = bool((master_tm.get("partial_exit") or {}).get("enabled", False))
    partial_level_r = float((master_tm.get("partial_exit") or {}).get("level_r", 1.5))
    partial_exit_pct = float((master_tm.get("partial_exit") or {}).get("exit_pct", 50.0))
    trailing_enabled = bool((master_tm.get("trailing_stop") or {}).get("enabled", False))
    trailing_length = int((master_tm.get("trailing_stop") or {}).get("length", 21))
    trailing_activation_r = float((master_tm.get("trailing_stop") or {}).get("activation_r", 0.5))
    trailing_stepped = bool((master_tm.get("trailing_stop") or {}).get("stepped", True))
    trailing_min_move_pips = float((master_tm.get("trailing_stop") or {}).get("min_move_pips", 7.0))

    try:
        if tm is not None and tm.take_profit is not None and tm.take_profit.enabled is False:
            tp_mode = "disable"
        elif tm is not None and tm.take_profit is not None and (tm.take_profit.levels or []):
            tp_mode = "custom"
            lvl0 = tm.take_profit.levels[0]
            tp_target_r = float(lvl0.target_r)
            tp_exit_pct = float(lvl0.exit_pct)
    except Exception:
        pass

    try:
        if tm is not None and tm.partial_exit is not None and (tm.partial_exit.levels or []):
            partial_enabled = True
            lvl0 = tm.partial_exit.levels[0]
            partial_level_r = float(lvl0.level_r)
            partial_exit_pct = float(lvl0.exit_pct)
    except Exception:
        pass

    try:
        if tm is not None and tm.trailing_stop is not None and bool(tm.trailing_stop.enabled):
            trailing_enabled = True
            trailing_length = int(tm.trailing_stop.length)
            trailing_activation_r = float(tm.trailing_stop.activation_r)
            trailing_stepped = bool(tm.trailing_stop.stepped)
            trailing_min_move_pips = float(tm.trailing_stop.min_move_pips)
    except Exception:
        pass

    # Advanced: execution
    exec_dump = (spec_dump.get("execution") or {}) if isinstance(spec_dump, dict) else {}
    flatten_enabled = bool(exec_dump.get("flatten_enabled", False))
    flatten_time = str(exec_dump.get("flatten_time") or "")
    stop_signal_search_minutes_before = exec_dump.get("stop_signal_search_minutes_before")
    use_intraday_margin = bool(exec_dump.get("use_intraday_margin", False))

    return TEMPLATES.TemplateResponse(
        request,
        "strategy_spec_form.html",
        {
            "title": f"Spec (Form): {spec_name}",
            "spec_name": spec_name,
            "spec_rel_path": str(spec_path.relative_to(REPO_ROOT)),
            "message": message,
            "name": str(spec_dump.get("name") or spec_name),
            "description": str(spec_dump.get("description") or ""),
            "symbol": str(market.get("symbol") or ""),
            "exchange": str(market.get("exchange") or ""),
            "market_type": str(market.get("market_type") or "spot"),
            "entry_tf": str(spec_dump.get("entry_tf") or "1h"),
            "context_tf": str(spec_dump.get("context_tf") or ""),
            "long_enabled": bool(long_side.get("enabled", False)) if long_side else False,
            "short_enabled": bool(short_side.get("enabled", False)) if short_side else False,
            "long_conds": long_conds,
            "short_conds": short_conds,
            "stop_type": str(stop_type or "atr"),
            "stop_percent": float(stop_percent or 0.02),
            "atr_length": int(atr_length or 14),
            "atr_multiplier": float(atr_multiplier or 3.0),

            # Advanced: which sections are explicitly present in the YAML
            "adv_risk_enabled": bool(has_risk_override),
            "adv_calendar_enabled": bool(has_calendar_filters),
            "adv_tm_enabled": bool(has_trade_management),
            "adv_exec_enabled": bool(has_execution),

            # Advanced: risk
            "risk_per_trade_pct": float(risk_per_trade_pct),

            # Advanced: calendar filters
            "calendar_mode": calendar_mode,
            "calendar_allowed_days": allowed_days,
            "calendar_sessions": sessions,

            # Advanced: trade management
            "tp_mode": tp_mode,
            "tp_target_r": tp_target_r,
            "tp_exit_pct": tp_exit_pct,
            "partial_enabled": partial_enabled,
            "partial_level_r": partial_level_r,
            "partial_exit_pct": partial_exit_pct,
            "trailing_enabled": trailing_enabled,
            "trailing_length": trailing_length,
            "trailing_activation_r": trailing_activation_r,
            "trailing_stepped": trailing_stepped,
            "trailing_min_move_pips": trailing_min_move_pips,

            # Advanced: execution
            "flatten_enabled": flatten_enabled,
            "flatten_time": flatten_time,
            "stop_signal_search_minutes_before": "" if stop_signal_search_minutes_before is None else str(stop_signal_search_minutes_before),
            "use_intraday_margin": use_intraday_margin,

            # Master defaults (for transparency + copy-to-custom)
            "master_calendar": master_calendar,
            "master_tm": master_tm,
        },
    )


@app.post("/strategies/spec/{spec_name}/form", response_class=HTMLResponse)
async def strategy_spec_form_submit(request: Request, spec_name: str):
    spec_path = _safe_spec_path(spec_name)
    if spec_path is None:
        return RedirectResponse(url="/strategies", status_code=303)

    form = await request.form()
    act = str(form.get("action") or "save").strip().lower()

    # Compile-only path (use saved file)
    if act == "compile":
        if not spec_path.exists():
            return RedirectResponse(url="/strategies", status_code=303)
        bundle = create_run_bundle(repo_root=REPO_ROOT, workflow="compile", strategy_name=spec_name)
        write_bundle_meta(bundle, python_executable=sys.executable, repo_root=REPO_ROOT)
        write_bundle_inputs(bundle, {"workflow": "compile", "spec_path": str(spec_path)})
        argv = [sys.executable, "scripts/build_strategy_from_spec.py", "--spec", str(spec_path)]
        write_bundle_command(bundle, argv)
        job = runner.create(argv=argv, cwd=REPO_ROOT)
        JOB_META[job.id] = {
            "bundle_dir": str(bundle.dir),
            "workflow": "compile",
            "strategy_name": spec_name,
            "spec_path": str(spec_path),
        }
        runner.start(job.id)
        return TEMPLATES.TemplateResponse(
            request,
            "job.html",
            {"title": "Compiling strategy", "job_id": job.id, "next_hint": "backtest"},
        )

    import yaml

    base: dict[str, Any] = {}
    if spec_path.exists():
        try:
            base = yaml.safe_load(spec_path.read_text(encoding="utf-8", errors="replace")) or {}
        except Exception:
            base = {}

    name = str(form.get("name") or "").strip()
    description = str(form.get("description") or "").strip()
    symbol = str(form.get("symbol") or "").strip() or "UNKNOWN"
    exchange = str(form.get("exchange") or "").strip() or "unknown"
    market_type = str(form.get("market_type") or "spot").strip() or "spot"
    entry_tf = str(form.get("entry_tf") or "1h").strip() or "1h"
    context_tf_raw = str(form.get("context_tf") or "").strip()

    long_enabled = bool(form.get("long_enabled"))
    short_enabled = bool(form.get("short_enabled"))

    stop_type = str(form.get("stop_type") or "atr").strip().lower()
    if stop_type not in {"atr", "percent"}:
        stop_type = "atr"

    try:
        stop_percent = float(str(form.get("stop_percent") or "0.02").strip())
    except Exception:
        stop_percent = 0.02
    try:
        atr_length = int(str(form.get("atr_length") or "14").strip())
    except Exception:
        atr_length = 14
    try:
        atr_multiplier = float(str(form.get("atr_multiplier") or "3.0").strip())
    except Exception:
        atr_multiplier = 3.0

    def _conds_from(prefix: str) -> list[dict[str, str]]:
        def _get(lst: list[Any], i: int, default: str = "") -> str:
            try:
                v = lst[i]
            except Exception:
                v = default
            return str(default if v is None else v)

        tfs = list(form.getlist(f"{prefix}_tf"))
        types = list(form.getlist(f"{prefix}_type"))
        exprs = list(form.getlist(f"{prefix}_expr"))

        cross_ma = list(form.getlist(f"{prefix}_cross_ma"))
        cross_fast = list(form.getlist(f"{prefix}_cross_fast"))
        cross_slow = list(form.getlist(f"{prefix}_cross_slow"))
        cross_dir = list(form.getlist(f"{prefix}_cross_direction"))

        rsi_len = list(form.getlist(f"{prefix}_rsi_len"))
        rsi_op = list(form.getlist(f"{prefix}_rsi_op"))
        rsi_level = list(form.getlist(f"{prefix}_rsi_level"))

        cv_ma = list(form.getlist(f"{prefix}_cv_ma"))
        cv_len = list(form.getlist(f"{prefix}_cv_len"))
        cv_op = list(form.getlist(f"{prefix}_cv_op"))

        don_len = list(form.getlist(f"{prefix}_don_len"))
        don_dir = list(form.getlist(f"{prefix}_don_direction"))

        n = max(len(tfs), len(types), len(exprs))
        out: list[dict[str, str]] = []
        for i in range(n):
            tf_s = _get(tfs, i).strip() or entry_tf
            typ = _get(types, i).strip().lower()
            raw_expr = _get(exprs, i).strip()

            if not typ and not raw_expr:
                continue
            if typ in {"", "none"}:
                typ = "custom" if raw_expr else ""
            if not typ:
                continue

            if typ == "custom":
                if not raw_expr:
                    continue
                out.append({"tf": tf_s, "expr": raw_expr})
                continue

            if typ == "ma_cross":
                ma = _get(cross_ma, i, "ema").strip().lower() or "ema"
                direction = _get(cross_dir, i, "above").strip().lower() or "above"
                try:
                    fast = int(_get(cross_fast, i).strip())
                    slow = int(_get(cross_slow, i).strip())
                except Exception as e:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: MA cross requires fast/slow lengths") from e
                if ma not in {"ema", "sma"}:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: invalid MA type")
                fn = "crosses_above" if direction in {"above", "over"} else "crosses_below"
                expr = f"{fn}({ma}(close, {fast}), {ma}(close, {slow}))"
                out.append({"tf": tf_s, "expr": expr})
                continue

            if typ == "rsi":
                op = _get(rsi_op, i, "<").strip() or "<"
                if op not in {"<", ">", "<=", ">="}:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: invalid RSI operator")
                try:
                    length = int(_get(rsi_len, i, "14").strip() or "14")
                except Exception as e:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: RSI length is required") from e
                level_raw = _get(rsi_level, i).strip()
                if not level_raw:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: RSI level is required")
                try:
                    level = float(level_raw)
                except Exception as e:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: RSI level must be a number") from e
                expr = f"rsi(close, {length}) {op} {level}"
                out.append({"tf": tf_s, "expr": expr})
                continue

            if typ == "close_vs_ma":
                ma = _get(cv_ma, i, "ema").strip().lower() or "ema"
                op = _get(cv_op, i, ">").strip() or ">"
                if ma not in {"ema", "sma"}:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: invalid MA type")
                if op not in {"<", ">", "<=", ">="}:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: invalid operator")
                try:
                    length = int(_get(cv_len, i).strip())
                except Exception as e:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: MA length is required") from e
                expr = f"close {op} {ma}(close, {length})"
                out.append({"tf": tf_s, "expr": expr})
                continue

            if typ == "donchian":
                try:
                    length = int(_get(don_len, i).strip())
                except Exception as e:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: Donchian length is required") from e
                direction = _get(don_dir, i, "breakout").strip().lower() or "breakout"
                if direction in {"breakout", "above", "over"}:
                    expr = f"crosses_above(close, shift(donchian_high({length}), 1))"
                else:
                    expr = f"crosses_below(close, shift(donchian_low({length}), 1))"
                out.append({"tf": tf_s, "expr": expr})
                continue

            raise ValueError(f"{prefix.upper()} condition {i+1}: unknown type '{typ}'")

        return out

    import urllib.parse

    try:
        long_conds = _conds_from("long")
        short_conds = _conds_from("short")
    except ValueError as e:
        return RedirectResponse(
            url=f"/strategies/spec/{spec_name}/form?message={urllib.parse.quote(str(e))}",
            status_code=303,
        )

    # Advanced override toggles
    adv_risk_enabled = bool(form.get("adv_risk_enabled"))
    adv_calendar_enabled = bool(form.get("adv_calendar_enabled"))
    adv_tm_enabled = bool(form.get("adv_tm_enabled"))
    adv_exec_enabled = bool(form.get("adv_exec_enabled"))

    # Apply core edits while preserving any advanced keys the user didn't touch.
    base["name"] = name
    base["description"] = description
    base["market"] = {"symbol": symbol, "exchange": exchange, "market_type": market_type}
    base["entry_tf"] = entry_tf

    if context_tf_raw:
        base["context_tf"] = context_tf_raw
    else:
        base.pop("context_tf", None)

    def _upsert_side(side_key: str, enabled: bool, conds: list[dict[str, str]]):
        if not enabled:
            base.pop(side_key, None)
            return
        side = base.get(side_key)
        if not isinstance(side, dict):
            side = {}
        side["enabled"] = True
        side["conditions_all"] = conds
        side.setdefault("exit_conditions_all", [])
        side["stop_type"] = stop_type
        side["stop_percent"] = float(stop_percent) if stop_type == "percent" else None
        side["atr_length"] = int(atr_length) if stop_type == "atr" else None
        side["atr_multiplier"] = float(atr_multiplier) if stop_type == "atr" else None
        base[side_key] = side

    _upsert_side("long", long_enabled, long_conds)
    _upsert_side("short", short_enabled, short_conds)

    # --- Advanced: Risk override ---
    if adv_risk_enabled:
        try:
            r = float(str(form.get("risk_per_trade_pct") or "1.0").strip())
            if r <= 0:
                r = 1.0
        except Exception:
            r = 1.0
        base["risk_per_trade_pct"] = float(r)
    else:
        base.pop("risk_per_trade_pct", None)

    # --- Advanced: Calendar filters override ---
    if adv_calendar_enabled:
        master_calendar = _get_master_calendar_defaults()
        master_allowed_days = list(master_calendar.get("allowed_days") or [0, 1, 2, 3, 4])
        master_sessions_cfg = master_calendar.get("sessions") if isinstance(master_calendar, dict) else None
        if not isinstance(master_sessions_cfg, dict):
            master_sessions_cfg = {}

        calendar_mode = str(form.get("calendar_mode") or "master").strip().lower()
        if calendar_mode not in {"master", "disable", "custom"}:
            calendar_mode = "master"

        cal: dict[str, Any] = {}
        if calendar_mode == "disable":
            cal = {"master_filters_enabled": False}
        elif calendar_mode == "master":
            cal = {"master_filters_enabled": True}
        else:
            # Custom: allow days + sessions. Keep structure minimal.
            allowed_days: list[int] = []
            for d in form.getlist("dow"):
                try:
                    allowed_days.append(int(d))
                except Exception:
                    pass
            if not allowed_days:
                allowed_days = master_allowed_days

            sessions_enabled = {
                "Asia": bool(form.get("session_Asia")),
                "London": bool(form.get("session_London")),
                "NewYork": bool(form.get("session_NewYork")),
            }

            def _sess_times(name: str) -> tuple[str, str]:
                cfg = master_sessions_cfg.get(name)
                if not isinstance(cfg, dict):
                    return ("", "")
                return (str(cfg.get("start") or ""), str(cfg.get("end") or ""))

            a_start, a_end = _sess_times("Asia")
            l_start, l_end = _sess_times("London")
            n_start, n_end = _sess_times("NewYork")
            cal = {
                "master_filters_enabled": True,
                "day_of_week": {"enabled": True, "allowed_days": allowed_days},
                "trading_sessions_enabled": True,
                "trading_sessions": {
                    "Asia": {"enabled": sessions_enabled["Asia"], "start": a_start, "end": a_end},
                    "London": {"enabled": sessions_enabled["London"], "start": l_start, "end": l_end},
                    "NewYork": {"enabled": sessions_enabled["NewYork"], "start": n_start, "end": n_end},
                },
            }

        filt = base.get("filters")
        if not isinstance(filt, dict):
            filt = {}
        filt["calendar_filters"] = cal
        base["filters"] = filt
    else:
        # Remove calendar_filters; keep other filter dicts if user had them.
        if isinstance(base.get("filters"), dict):
            base["filters"].pop("calendar_filters", None)
            if not base["filters"]:
                base.pop("filters", None)

    # --- Advanced: Trade management override ---
    if adv_tm_enabled:
        trade_management: dict[str, Any] = {}

        tp_mode = str(form.get("tp_mode") or "master").strip().lower()
        if tp_mode == "disable":
            trade_management["take_profit"] = {"enabled": False, "levels": []}
        elif tp_mode == "custom":
            try:
                tp_target_r = float(str(form.get("tp_target_r") or "3.0").strip())
            except Exception:
                tp_target_r = 3.0
            try:
                tp_exit_pct = float(str(form.get("tp_exit_pct") or "100").strip())
            except Exception:
                tp_exit_pct = 100.0
            trade_management["take_profit"] = {
                "enabled": True,
                "levels": [{"enabled": True, "target_r": tp_target_r, "exit_pct": tp_exit_pct}],
            }

        if bool(form.get("partial_enabled")):
            try:
                partial_level_r = float(str(form.get("partial_level_r") or "1.5").strip())
            except Exception:
                partial_level_r = 1.5
            try:
                partial_exit_pct = float(str(form.get("partial_exit_pct") or "50").strip())
            except Exception:
                partial_exit_pct = 50.0
            trade_management["partial_exit"] = {
                "enabled": True,
                "levels": [{"enabled": True, "level_r": partial_level_r, "exit_pct": partial_exit_pct}],
            }

        if bool(form.get("trailing_enabled")):
            try:
                trailing_length = int(str(form.get("trailing_length") or "21").strip())
            except Exception:
                trailing_length = 21
            try:
                trailing_activation_r = float(str(form.get("trailing_activation_r") or "0.5").strip())
            except Exception:
                trailing_activation_r = 0.5
            trailing_stepped = bool(form.get("trailing_stepped"))
            try:
                trailing_min_move_pips = float(str(form.get("trailing_min_move_pips") or "7.0").strip())
            except Exception:
                trailing_min_move_pips = 7.0

            trade_management["trailing_stop"] = {
                "enabled": True,
                "type": "EMA",
                "length": trailing_length,
                "activation_type": "r_based",
                "activation_r": trailing_activation_r,
                "stepped": trailing_stepped,
                "min_move_pips": trailing_min_move_pips,
            }

        if trade_management:
            base["trade_management"] = trade_management
        else:
            base.pop("trade_management", None)
    else:
        base.pop("trade_management", None)

    # --- Advanced: Execution override ---
    if adv_exec_enabled:
        exec_cfg: dict[str, Any] = {}
        exec_cfg["flatten_enabled"] = bool(form.get("flatten_enabled"))
        ft = str(form.get("flatten_time") or "").strip()
        exec_cfg["flatten_time"] = ft if ft else None

        ss = str(form.get("stop_signal_search_minutes_before") or "").strip()
        if ss:
            try:
                exec_cfg["stop_signal_search_minutes_before"] = int(ss)
            except Exception:
                exec_cfg["stop_signal_search_minutes_before"] = None

        exec_cfg["use_intraday_margin"] = bool(form.get("use_intraday_margin"))
        base["execution"] = exec_cfg
    else:
        base.pop("execution", None)

    # Validate
    try:
        spec = StrategySpec.model_validate(base)
    except Exception as e:
        return RedirectResponse(url=f"/strategies/spec/{spec_name}/form?message=Invalid+spec%3A+{type(e).__name__}", status_code=303)

    new_name = str(getattr(spec, "name", "") or "").strip()
    if not _is_safe_strategy_name(new_name):
        return RedirectResponse(url=f"/strategies/spec/{spec_name}/form?message=Invalid+name", status_code=303)

    try:
        new_path = _write_validated_spec_to_repo(spec=spec, old_spec_name=spec_name)
    except ValueError as e:
        return RedirectResponse(
            url=f"/strategies/spec/{spec_name}/form?message={urllib.parse.quote(str(e))}",
            status_code=303,
        )
    except Exception:
        return RedirectResponse(url=f"/strategies/spec/{spec_name}/form?message=Failed+to+save", status_code=303)

    if act == "save":
        return RedirectResponse(url=f"/strategies/spec/{new_name}/form?message=Saved", status_code=303)

    if act == "save_compile":
        bundle = create_run_bundle(repo_root=REPO_ROOT, workflow="compile", strategy_name=new_name)
        write_bundle_meta(bundle, python_executable=sys.executable, repo_root=REPO_ROOT)
        write_bundle_inputs(bundle, {"workflow": "compile", "spec_path": str(new_path)})
        argv = [sys.executable, "scripts/build_strategy_from_spec.py", "--spec", str(new_path)]
        write_bundle_command(bundle, argv)
        job = runner.create(argv=argv, cwd=REPO_ROOT)
        JOB_META[job.id] = {
            "bundle_dir": str(bundle.dir),
            "workflow": "compile",
            "strategy_name": new_name,
            "spec_path": str(new_path),
        }
        runner.start(job.id)
        return TEMPLATES.TemplateResponse(
            request,
            "job.html",
            {"title": "Saving + compiling", "job_id": job.id, "next_hint": "backtest"},
        )

    return RedirectResponse(url=f"/strategies/spec/{new_name}/form", status_code=303)


@app.post("/strategies/spec/{spec_name}/form/preview", response_class=JSONResponse)
async def strategy_spec_form_preview(request: Request, spec_name: str):
    """Validate current form state and return the compact YAML that would be saved."""

    spec_path = _safe_spec_path(spec_name)
    if spec_path is None:
        return JSONResponse({"ok": False, "error": "Invalid spec name"}, status_code=400)

    form = await request.form()

    import yaml

    base: dict[str, Any] = {}
    if spec_path.exists():
        try:
            base = yaml.safe_load(spec_path.read_text(encoding="utf-8", errors="replace")) or {}
        except Exception:
            base = {}

    import copy

    before = copy.deepcopy(base)

    # Reuse the same logic as the submit handler (duplicated intentionally to keep the endpoint simple).
    name = str(form.get("name") or "").strip()
    description = str(form.get("description") or "").strip()
    symbol = str(form.get("symbol") or "").strip() or "UNKNOWN"
    exchange = str(form.get("exchange") or "").strip() or "unknown"
    market_type = str(form.get("market_type") or "spot").strip() or "spot"
    entry_tf = str(form.get("entry_tf") or "1h").strip() or "1h"
    context_tf_raw = str(form.get("context_tf") or "").strip()

    long_enabled = bool(form.get("long_enabled"))
    short_enabled = bool(form.get("short_enabled"))

    stop_type = str(form.get("stop_type") or "atr").strip().lower()
    if stop_type not in {"atr", "percent"}:
        stop_type = "atr"

    try:
        stop_percent = float(str(form.get("stop_percent") or "0.02").strip())
    except Exception:
        stop_percent = 0.02
    try:
        atr_length = int(str(form.get("atr_length") or "14").strip())
    except Exception:
        atr_length = 14
    try:
        atr_multiplier = float(str(form.get("atr_multiplier") or "3.0").strip())
    except Exception:
        atr_multiplier = 3.0

    def _conds_from(prefix: str) -> list[dict[str, str]]:
        def _get(lst: list[Any], i: int, default: str = "") -> str:
            try:
                v = lst[i]
            except Exception:
                v = default
            return str(default if v is None else v)

        tfs = list(form.getlist(f"{prefix}_tf"))
        types = list(form.getlist(f"{prefix}_type"))
        exprs = list(form.getlist(f"{prefix}_expr"))

        cross_ma = list(form.getlist(f"{prefix}_cross_ma"))
        cross_fast = list(form.getlist(f"{prefix}_cross_fast"))
        cross_slow = list(form.getlist(f"{prefix}_cross_slow"))
        cross_dir = list(form.getlist(f"{prefix}_cross_direction"))

        rsi_len = list(form.getlist(f"{prefix}_rsi_len"))
        rsi_op = list(form.getlist(f"{prefix}_rsi_op"))
        rsi_level = list(form.getlist(f"{prefix}_rsi_level"))

        cv_ma = list(form.getlist(f"{prefix}_cv_ma"))
        cv_len = list(form.getlist(f"{prefix}_cv_len"))
        cv_op = list(form.getlist(f"{prefix}_cv_op"))

        don_len = list(form.getlist(f"{prefix}_don_len"))
        don_dir = list(form.getlist(f"{prefix}_don_direction"))

        n = max(len(tfs), len(types), len(exprs))
        out: list[dict[str, str]] = []
        for i in range(n):
            tf_s = _get(tfs, i).strip() or entry_tf
            typ = _get(types, i).strip().lower()
            raw_expr = _get(exprs, i).strip()

            if not typ and not raw_expr:
                continue
            if typ in {"", "none"}:
                typ = "custom" if raw_expr else ""
            if not typ:
                continue

            if typ == "custom":
                if not raw_expr:
                    continue
                out.append({"tf": tf_s, "expr": raw_expr})
                continue

            if typ == "ma_cross":
                ma = _get(cross_ma, i, "ema").strip().lower() or "ema"
                direction = _get(cross_dir, i, "above").strip().lower() or "above"
                try:
                    fast = int(_get(cross_fast, i).strip())
                    slow = int(_get(cross_slow, i).strip())
                except Exception as e:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: MA cross requires fast/slow lengths") from e
                if ma not in {"ema", "sma"}:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: invalid MA type")
                fn = "crosses_above" if direction in {"above", "over"} else "crosses_below"
                out.append({"tf": tf_s, "expr": f"{fn}({ma}(close, {fast}), {ma}(close, {slow}))"})
                continue

            if typ == "rsi":
                op = _get(rsi_op, i, "<").strip() or "<"
                if op not in {"<", ">", "<=", ">="}:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: invalid RSI operator")
                try:
                    length = int(_get(rsi_len, i, "14").strip() or "14")
                except Exception as e:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: RSI length is required") from e
                level_raw = _get(rsi_level, i).strip()
                if not level_raw:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: RSI level is required")
                try:
                    level = float(level_raw)
                except Exception as e:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: RSI level must be a number") from e
                out.append({"tf": tf_s, "expr": f"rsi(close, {length}) {op} {level}"})
                continue

            if typ == "close_vs_ma":
                ma = _get(cv_ma, i, "ema").strip().lower() or "ema"
                op = _get(cv_op, i, ">").strip() or ">"
                if ma not in {"ema", "sma"}:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: invalid MA type")
                if op not in {"<", ">", "<=", ">="}:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: invalid operator")
                try:
                    length = int(_get(cv_len, i).strip())
                except Exception as e:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: MA length is required") from e
                out.append({"tf": tf_s, "expr": f"close {op} {ma}(close, {length})"})
                continue

            if typ == "donchian":
                try:
                    length = int(_get(don_len, i).strip())
                except Exception as e:
                    raise ValueError(f"{prefix.upper()} condition {i+1}: Donchian length is required") from e
                direction = _get(don_dir, i, "breakout").strip().lower() or "breakout"
                if direction in {"breakout", "above", "over"}:
                    expr = f"crosses_above(close, shift(donchian_high({length}), 1))"
                else:
                    expr = f"crosses_below(close, shift(donchian_low({length}), 1))"
                out.append({"tf": tf_s, "expr": expr})
                continue

            raise ValueError(f"{prefix.upper()} condition {i+1}: unknown type '{typ}'")

        return out

    try:
        long_conds = _conds_from("long")
        short_conds = _conds_from("short")
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)

    adv_risk_enabled = bool(form.get("adv_risk_enabled"))
    adv_calendar_enabled = bool(form.get("adv_calendar_enabled"))
    adv_tm_enabled = bool(form.get("adv_tm_enabled"))
    adv_exec_enabled = bool(form.get("adv_exec_enabled"))

    base["name"] = name
    base["description"] = description
    base["market"] = {"symbol": symbol, "exchange": exchange, "market_type": market_type}
    base["entry_tf"] = entry_tf

    if context_tf_raw:
        base["context_tf"] = context_tf_raw
    else:
        base.pop("context_tf", None)

    def _upsert_side(side_key: str, enabled: bool, conds: list[dict[str, str]]):
        if not enabled:
            base.pop(side_key, None)
            return
        side = base.get(side_key)
        if not isinstance(side, dict):
            side = {}
        side["enabled"] = True
        side["conditions_all"] = conds
        side.setdefault("exit_conditions_all", [])
        side["stop_type"] = stop_type
        side["stop_percent"] = float(stop_percent) if stop_type == "percent" else None
        side["atr_length"] = int(atr_length) if stop_type == "atr" else None
        side["atr_multiplier"] = float(atr_multiplier) if stop_type == "atr" else None
        base[side_key] = side

    _upsert_side("long", long_enabled, long_conds)
    _upsert_side("short", short_enabled, short_conds)

    if adv_risk_enabled:
        try:
            r = float(str(form.get("risk_per_trade_pct") or "1.0").strip())
            if r <= 0:
                r = 1.0
        except Exception:
            r = 1.0
        base["risk_per_trade_pct"] = float(r)
    else:
        base.pop("risk_per_trade_pct", None)

    if adv_calendar_enabled:
        master_calendar = _get_master_calendar_defaults()
        master_allowed_days = list(master_calendar.get("allowed_days") or [0, 1, 2, 3, 4])
        master_sessions_cfg = master_calendar.get("sessions") if isinstance(master_calendar, dict) else None
        if not isinstance(master_sessions_cfg, dict):
            master_sessions_cfg = {}

        calendar_mode = str(form.get("calendar_mode") or "master").strip().lower()
        if calendar_mode not in {"master", "disable", "custom"}:
            calendar_mode = "master"

        cal: dict[str, Any] = {}
        if calendar_mode == "disable":
            cal = {"master_filters_enabled": False}
        elif calendar_mode == "master":
            cal = {"master_filters_enabled": True}
        else:
            allowed_days: list[int] = []
            for d in form.getlist("dow"):
                try:
                    allowed_days.append(int(d))
                except Exception:
                    pass
            if not allowed_days:
                allowed_days = master_allowed_days

            sessions_enabled = {
                "Asia": bool(form.get("session_Asia")),
                "London": bool(form.get("session_London")),
                "NewYork": bool(form.get("session_NewYork")),
            }

            def _sess_times(name: str) -> tuple[str, str]:
                cfg = master_sessions_cfg.get(name)
                if not isinstance(cfg, dict):
                    return ("", "")
                return (str(cfg.get("start") or ""), str(cfg.get("end") or ""))

            a_start, a_end = _sess_times("Asia")
            l_start, l_end = _sess_times("London")
            n_start, n_end = _sess_times("NewYork")
            cal = {
                "master_filters_enabled": True,
                "day_of_week": {"enabled": True, "allowed_days": allowed_days},
                "trading_sessions_enabled": True,
                "trading_sessions": {
                    "Asia": {"enabled": sessions_enabled["Asia"], "start": a_start, "end": a_end},
                    "London": {"enabled": sessions_enabled["London"], "start": l_start, "end": l_end},
                    "NewYork": {"enabled": sessions_enabled["NewYork"], "start": n_start, "end": n_end},
                },
            }

        filt = base.get("filters")
        if not isinstance(filt, dict):
            filt = {}
        filt["calendar_filters"] = cal
        base["filters"] = filt
    else:
        if isinstance(base.get("filters"), dict):
            base["filters"].pop("calendar_filters", None)
            if not base["filters"]:
                base.pop("filters", None)

    if adv_tm_enabled:
        trade_management: dict[str, Any] = {}

        tp_mode = str(form.get("tp_mode") or "master").strip().lower()
        if tp_mode == "disable":
            trade_management["take_profit"] = {"enabled": False, "levels": []}
        elif tp_mode == "custom":
            try:
                tp_target_r = float(str(form.get("tp_target_r") or "3.0").strip())
            except Exception:
                tp_target_r = 3.0
            try:
                tp_exit_pct = float(str(form.get("tp_exit_pct") or "100").strip())
            except Exception:
                tp_exit_pct = 100.0
            trade_management["take_profit"] = {
                "enabled": True,
                "levels": [{"enabled": True, "target_r": tp_target_r, "exit_pct": tp_exit_pct}],
            }

        if bool(form.get("partial_enabled")):
            try:
                partial_level_r = float(str(form.get("partial_level_r") or "1.5").strip())
            except Exception:
                partial_level_r = 1.5
            try:
                partial_exit_pct = float(str(form.get("partial_exit_pct") or "50").strip())
            except Exception:
                partial_exit_pct = 50.0
            trade_management["partial_exit"] = {
                "enabled": True,
                "levels": [{"enabled": True, "level_r": partial_level_r, "exit_pct": partial_exit_pct}],
            }

        if bool(form.get("trailing_enabled")):
            try:
                trailing_length = int(str(form.get("trailing_length") or "21").strip())
            except Exception:
                trailing_length = 21
            try:
                trailing_activation_r = float(str(form.get("trailing_activation_r") or "0.5").strip())
            except Exception:
                trailing_activation_r = 0.5
            trailing_stepped = bool(form.get("trailing_stepped"))
            try:
                trailing_min_move_pips = float(str(form.get("trailing_min_move_pips") or "7.0").strip())
            except Exception:
                trailing_min_move_pips = 7.0

            trade_management["trailing_stop"] = {
                "enabled": True,
                "type": "EMA",
                "length": trailing_length,
                "activation_type": "r_based",
                "activation_r": trailing_activation_r,
                "stepped": trailing_stepped,
                "min_move_pips": trailing_min_move_pips,
            }

        if trade_management:
            base["trade_management"] = trade_management
        else:
            base.pop("trade_management", None)
    else:
        base.pop("trade_management", None)

    if adv_exec_enabled:
        exec_cfg: dict[str, Any] = {}
        exec_cfg["flatten_enabled"] = bool(form.get("flatten_enabled"))
        ft = str(form.get("flatten_time") or "").strip()
        exec_cfg["flatten_time"] = ft if ft else None

        ss = str(form.get("stop_signal_search_minutes_before") or "").strip()
        if ss:
            try:
                exec_cfg["stop_signal_search_minutes_before"] = int(ss)
            except Exception:
                exec_cfg["stop_signal_search_minutes_before"] = None

        exec_cfg["use_intraday_margin"] = bool(form.get("use_intraday_margin"))
        base["execution"] = exec_cfg
    else:
        base.pop("execution", None)

    try:
        spec = StrategySpec.model_validate(base)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=400)

    yaml_text = _spec_to_yaml_compact(spec)

    def _has_path(obj: Any, path: list[str]) -> bool:
        cur = obj
        for p in path:
            if not isinstance(cur, dict):
                return False
            if p not in cur:
                return False
            cur = cur[p]
        return True

    tracked_paths: list[tuple[str, list[str]]] = [
        ("context_tf", ["context_tf"]),
        ("long", ["long"]),
        ("short", ["short"]),
        ("risk_per_trade_pct", ["risk_per_trade_pct"]),
        ("filters.calendar_filters", ["filters", "calendar_filters"]),
        ("trade_management", ["trade_management"]),
        ("execution", ["execution"]),
    ]

    removed_paths: list[str] = []
    added_paths: list[str] = []
    for label, path in tracked_paths:
        had_before = _has_path(before, path)
        has_after = _has_path(base, path)
        if had_before and not has_after:
            removed_paths.append(label)
        if (not had_before) and has_after:
            added_paths.append(label)

    new_name = str(getattr(spec, "name", "") or "").strip()
    rename_to = new_name if (new_name and new_name != spec_name and _is_safe_strategy_name(new_name)) else ""

    return JSONResponse(
        {
            "ok": True,
            "yaml_text": yaml_text,
            "name": new_name,
            "rename_to": rename_to,
            "removed_paths": removed_paths,
            "added_paths": added_paths,
        }
    )


@app.post("/strategies/spec/{spec_name}", response_class=HTMLResponse)
async def strategy_spec_submit(
    request: Request,
    spec_name: str,
    action: str = Form("save"),
    yaml_text: str | None = Form(None),
):
    spec_path = _safe_spec_path(spec_name)
    if spec_path is None:
        return RedirectResponse(url="/strategies", status_code=303)

    act = str(action or "save").strip().lower()

    # Compile-only path (use saved file)
    if act == "compile":
        if not spec_path.exists():
            return RedirectResponse(url="/strategies", status_code=303)
        bundle = create_run_bundle(repo_root=REPO_ROOT, workflow="compile", strategy_name=spec_name)
        write_bundle_meta(bundle, python_executable=sys.executable, repo_root=REPO_ROOT)
        write_bundle_inputs(bundle, {"workflow": "compile", "spec_path": str(spec_path)})
        argv = [sys.executable, "scripts/build_strategy_from_spec.py", "--spec", str(spec_path)]
        write_bundle_command(bundle, argv)
        job = runner.create(argv=argv, cwd=REPO_ROOT)
        JOB_META[job.id] = {
            "bundle_dir": str(bundle.dir),
            "workflow": "compile",
            "strategy_name": spec_name,
            "spec_path": str(spec_path),
        }
        runner.start(job.id)
        return TEMPLATES.TemplateResponse(
            request,
            "job.html",
            {"title": "Compiling strategy", "job_id": job.id, "next_hint": "backtest"},
        )

    # Save (+ optional compile) path
    raw = (yaml_text or "").strip()
    if not raw:
        return RedirectResponse(url=f"/strategies/spec/{spec_name}?message=Empty+YAML", status_code=303)

    import yaml

    try:
        spec_data = yaml.safe_load(raw) or {}
        spec = StrategySpec.model_validate(spec_data)
    except Exception as e:
        return RedirectResponse(url=f"/strategies/spec/{spec_name}?message=Invalid+spec%3A+{type(e).__name__}", status_code=303)

    new_name = str(getattr(spec, "name", "") or "").strip()
    if not _is_safe_strategy_name(new_name):
        return RedirectResponse(url=f"/strategies/spec/{spec_name}?message=Invalid+name", status_code=303)

    try:
        new_path = _write_validated_spec_to_repo(spec=spec, old_spec_name=spec_name)
    except ValueError as e:
        return RedirectResponse(
            url=f"/strategies/spec/{spec_name}?message={urllib.parse.quote(str(e))}",
            status_code=303,
        )
    except Exception:
        return RedirectResponse(url=f"/strategies/spec/{spec_name}?message=Failed+to+save", status_code=303)

    if act == "save":
        return RedirectResponse(url=f"/strategies/spec/{new_name}?message=Saved", status_code=303)

    if act == "save_compile":
        bundle = create_run_bundle(repo_root=REPO_ROOT, workflow="compile", strategy_name=new_name)
        write_bundle_meta(bundle, python_executable=sys.executable, repo_root=REPO_ROOT)
        write_bundle_inputs(bundle, {"workflow": "compile", "spec_path": str(new_path)})
        argv = [sys.executable, "scripts/build_strategy_from_spec.py", "--spec", str(new_path)]
        write_bundle_command(bundle, argv)

        job = runner.create(argv=argv, cwd=REPO_ROOT)
        JOB_META[job.id] = {
            "bundle_dir": str(bundle.dir),
            "workflow": "compile",
            "strategy_name": new_name,
            "spec_path": str(new_path),
        }
        runner.start(job.id)
        return TEMPLATES.TemplateResponse(
            request,
            "job.html",
            {"title": "Saving + compiling", "job_id": job.id, "next_hint": "backtest"},
        )

    return RedirectResponse(url=f"/strategies/spec/{new_name}", status_code=303)


@app.get("/strategies/from-run/{run_name}", response_class=HTMLResponse)
def strategy_from_run(request: Request, run_name: str, overwrite: str | None = None):
    run_dir = _safe_run_dir(run_name)
    if run_dir is None:
        return RedirectResponse(url="/runs", status_code=303)

    spec_path = run_dir / "spec.yml"
    if not spec_path.exists():
        return RedirectResponse(url=f"/runs/{run_name}", status_code=303)

    import yaml

    try:
        spec_data = yaml.safe_load(spec_path.read_text(encoding="utf-8", errors="replace")) or {}
        spec = StrategySpec.model_validate(spec_data)
    except Exception:
        return RedirectResponse(url=f"/runs/{run_name}", status_code=303)

    name = str(getattr(spec, "name", "") or "").strip()
    if not _is_safe_strategy_name(name):
        return RedirectResponse(url=f"/runs/{run_name}", status_code=303)

    research_specs_dir = REPO_ROOT / "research_specs"
    research_specs_dir.mkdir(parents=True, exist_ok=True)
    repo_spec = research_specs_dir / f"{name}.yml"

    if overwrite is not None or (not repo_spec.exists()):
        repo_spec.write_text(_spec_to_yaml_compact(spec), encoding="utf-8")

    return RedirectResponse(url=f"/strategies/spec/{name}?message=Loaded+from+run+{run_name}", status_code=303)


@app.get("/create-strategy-guided", response_class=HTMLResponse)
def create_strategy_guided_step1(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "guided_step1.html",
        {
            "title": "Create Strategy (Guided)",
            "template": "builder_v2",
            "templates": [("builder_v2", "Builder v2 (Context + Trigger)")],
        },
    )



@app.post("/create-strategy-guided/step1", response_class=HTMLResponse)
async def create_strategy_guided_step1_submit(
    request: Request,
    name: str = Form(...),
    template: str = Form("builder_v2"),
):
    name_clean = str(name).strip()
    if not _is_safe_strategy_name(name_clean):
        return TEMPLATES.TemplateResponse(
            request,
            "guided_step1.html",
            {
                "title": "Create Strategy (Guided)",
                "error": "Strategy name must use only letters/numbers/_/- (no spaces).",
                "name": name,
                "template": template,
                "templates": [("builder_v2", "Builder v2 (Context + Trigger)")],
            },
        )

    # Guided strategy creation is Builder v2 only.
    template_clean = "builder_v2"

    draft_id = str(uuid.uuid4())
    DRAFTS[draft_id] = {
        "kind": "guided",
        "name": name_clean,
        "template": template_clean,
        "entry_tf": "1h",
        "asset_class": "crypto",
        "exchange": "binance",
        "market_type": "spot",
        "symbol": "BTCUSDT",
        "risk_per_trade_pct": 1.0,
        "holding_style": "intraday",
        "flatten_enabled": True,
        "flatten_time": "21:30",
        "stop_signal_search_minutes_before": None,
        "use_intraday_margin": True,
        "long_enabled": True,
        "short_enabled": True,
        "stop_type": "atr",
        "stop_percent": 0.02,
        "atr_length": 14,
        "atr_multiplier": 3.0,
        # Advanced (optional) step defaults (newbie-friendly)
        # Use master calendar filter defaults unless user explicitly disables.
        "calendar_filters_mode": "master",
        "calendar_filters": {"master_filters_enabled": True},
        "trade_management": None,
        "tm_tp_mode": "master",
        "tm_partial_enabled": False,
        "tm_trailing_enabled": False,
        "defaults": _guided_defaults(str(template)),
    }

    if str(template).strip().lower() == "builder_v2":
        # Builder v2 defaults (MA stack + candle triggers). Kept inside the draft
        # so the UI can round-trip the form values.
        DRAFTS[draft_id].update(
            {
                "entry_tf": "15m",
                "context_tf": "",
                "align_with_context": True,
                "ma_type": "ema",
                "ma_fast": 20,
                "ma_mid": 50,
                "ma_slow": 200,
                "stack_mode": "none",
                "slope_mode": "none",
                "slope_lookback": 10,
                "min_ma_dist_pct": 0.0,
                "trigger_type": "pin_bar",
                "pin_wick_body": 2.0,
                "pin_opp_wick_body_max": 1.0,
                "pin_min_body_pct": 0.2,
            }
        )
    return RedirectResponse(url=f"/create-strategy-guided/step2?draft_id={draft_id}", status_code=303)


@app.get("/create-strategy-guided/step2", response_class=HTMLResponse)
def create_strategy_guided_step2(request: Request, draft_id: str):
    draft = DRAFTS.get(draft_id)
    if not draft or draft.get("kind") != "guided":
        return RedirectResponse(url="/create-strategy-guided", status_code=303)

    instruments = _guided_instruments()
    return TEMPLATES.TemplateResponse(
        request,
        "guided_step2.html",
        {
            "title": "Create Strategy (Guided)",
            "draft_id": draft_id,
            "draft": draft,
            "instruments": instruments,
        },
    )


@app.post("/create-strategy-guided/step2", response_class=HTMLResponse)
async def create_strategy_guided_step2_submit(
    request: Request,
    draft_id: str = Form(...),
    asset_class: str = Form("crypto"),
    exchange: str = Form("binance"),
    market_type: str = Form("spot"),
    symbol: str = Form("BTCUSDT"),
    entry_tf: str | None = Form(None),
):
    draft = DRAFTS.get(draft_id)
    if not draft or draft.get("kind") != "guided":
        return RedirectResponse(url="/create-strategy-guided", status_code=303)

    asset_class = str(asset_class).strip().lower()
    symbol = str(symbol).strip().upper()
    entry_tf_clean = "" if entry_tf is None else str(entry_tf).strip()
    exchange = str(exchange).strip().lower()
    market_type = str(market_type).strip().lower()

    # Reduce confusion: some combinations don't make sense.
    # - FX is spot
    # - futures are futures
    if asset_class == "futures":
        market_type = "futures"
    elif asset_class == "fx":
        market_type = "spot"

    instruments = _guided_instruments()
    allowed = instruments.get(asset_class) or []
    if symbol not in allowed and symbol != "CUSTOM":
        return TEMPLATES.TemplateResponse(
            request,
            "guided_step2.html",
            {
                "title": "Create Strategy (Guided)",
                "error": "Please select an instrument from the list (or use Custom).",
                "draft_id": draft_id,
                "draft": draft,
                "instruments": instruments,
            },
        )

    if symbol == "CUSTOM":
        # User will fill it in on step 3 (kept for MVP simplicity)
        symbol = ""

    template = str(draft.get("template") or "").strip().lower()

    # Builder v2 chooses Entry TF in Step 4 (so Step 2 must not clobber it).
    # Keep legacy behavior for older templates that still post entry_tf.
    update: dict[str, Any] = {
        "asset_class": asset_class,
        "exchange": exchange,
        "market_type": market_type if market_type in {"spot", "futures"} else "spot",
        "symbol": symbol,
    }
    if template != "builder_v2":
        if entry_tf_clean:
            update["entry_tf"] = entry_tf_clean
        else:
            update["entry_tf"] = str(draft.get("entry_tf") or "1h")
    else:
        # Ensure the key exists for initial rendering defaults, but never overwrite an existing value.
        if not draft.get("entry_tf"):
            update["entry_tf"] = "1h"

    draft.update(update)
    return RedirectResponse(url=f"/create-strategy-guided/step3?draft_id={draft_id}", status_code=303)


@app.get("/create-strategy-guided/step3", response_class=HTMLResponse)
def create_strategy_guided_step3_holding(request: Request, draft_id: str):
    draft = DRAFTS.get(draft_id)
    if not draft or draft.get("kind") != "guided":
        return RedirectResponse(url="/create-strategy-guided", status_code=303)

    return TEMPLATES.TemplateResponse(
        request,
        "guided_step3_holding.html",
        {
            "title": "Create Strategy (Guided)",
            "draft_id": draft_id,
            "draft": draft,
        },
    )


@app.post("/create-strategy-guided/step3", response_class=HTMLResponse)
async def create_strategy_guided_step3_holding_submit(
    request: Request,
    draft_id: str = Form(...),
    holding_style: str = Form("intraday"),
    flatten_enabled: str | None = Form(None),
    flatten_time: str = Form("21:30"),
    stop_signal_search_minutes_before: str | None = Form(None),
    use_intraday_margin: str | None = Form(None),
):
    draft = DRAFTS.get(draft_id)
    if not draft or draft.get("kind") != "guided":
        return RedirectResponse(url="/create-strategy-guided", status_code=303)

    holding_style = str(holding_style).strip().lower()
    if holding_style not in {"intraday", "swing"}:
        holding_style = "intraday"

    # Defaults by style
    if holding_style == "intraday":
        draft["flatten_enabled"] = True
        draft["flatten_time"] = (flatten_time or "21:30").strip()
        # Use intraday margin by default for futures
        draft["use_intraday_margin"] = True if use_intraday_margin is None else bool(use_intraday_margin)
    else:
        draft["flatten_enabled"] = False
        draft["flatten_time"] = None
        draft["use_intraday_margin"] = False

    # Allow user override for flatten_enabled checkbox
    if flatten_enabled is not None:
        draft["flatten_enabled"] = True
        draft["flatten_time"] = (flatten_time or "21:30").strip()
    elif holding_style == "intraday":
        # Keep default True already set
        pass

    # stop-signal-search minutes
    ssm = None
    try:
        raw = (stop_signal_search_minutes_before or "").strip()
        if raw:
            ssm = int(raw)
            if ssm < 0:
                ssm = 0
    except Exception:
        ssm = None
    draft["stop_signal_search_minutes_before"] = ssm

    draft["holding_style"] = holding_style
    return RedirectResponse(url=f"/create-strategy-guided/step4?draft_id={draft_id}", status_code=303)


@app.get("/create-strategy-guided/step4", response_class=HTMLResponse)
def create_strategy_guided_step4(request: Request, draft_id: str):
    draft = DRAFTS.get(draft_id)
    if not draft or draft.get("kind") != "guided":
        return RedirectResponse(url="/create-strategy-guided", status_code=303)

    if str(draft.get("template") or "").strip().lower() == "builder_v2":
        return TEMPLATES.TemplateResponse(
            request,
            "guided_builder_v2.html",
            {
                "title": "Create Strategy (Guided)",
                "draft_id": draft_id,
                "draft": draft,
                "preview": _builder_v2_preview(draft),
                "error": None,
            },
        )

    return TEMPLATES.TemplateResponse(
        request,
        "guided_step3.html",
        {
            "title": "Create Strategy (Guided)",
            "draft_id": draft_id,
            "draft": draft,
        },
    )


@app.post("/create-strategy-guided/step4", response_class=HTMLResponse)
async def create_strategy_guided_step4_submit(request: Request, draft_id: str = Form(...)):
    draft = DRAFTS.get(draft_id)
    if not draft or draft.get("kind") != "guided":
        return RedirectResponse(url="/create-strategy-guided", status_code=303)

    form = await request.form()
    form_dict = {k: v for k, v in form.items()}

    builder_mode = str(form_dict.get("builder_mode") or "").strip().lower()
    if str(draft.get("template") or "").strip().lower() == "builder_v2" or builder_mode == "v2":
        def _parse_rules_json(field_name: str) -> list[dict[str, Any]]:
            raw = str(form_dict.get(field_name) or "").strip()
            if not raw:
                return []
            try:
                import json

                data = json.loads(raw)
                if not isinstance(data, list):
                    return []
                out: list[dict[str, Any]] = []
                for item in data:
                    if isinstance(item, dict):
                        out.append(item)
                return out
            except Exception:
                return []


        def _parse_int(name: str, default: int) -> int:
            try:
                return int(str(form_dict.get(name) or default).strip())
            except Exception:
                return default

        def _parse_float(name: str, default: float) -> float:
            try:
                return float(str(form_dict.get(name) or default).strip())
            except Exception:
                return default

        def _tf_expr(atom: str, tf: str | None) -> str:
            tf = (tf or "").strip()
            if not tf:
                return atom
            return f"{atom}@{tf}"

        def _ma_expr(ma_type: str, length: int, tf: str | None) -> str:
            fn = "ema" if (ma_type or "").strip().lower() != "sma" else "sma"
            return _tf_expr(f"{fn}(close, {int(length)})", tf)

        # Allow custom symbol if step2 picked CUSTOM
        symbol = str(form_dict.get("symbol") or draft.get("symbol") or "").strip().upper()
        if not symbol:
            return TEMPLATES.TemplateResponse(
                request,
                "guided_builder_v2.html",
                {
                    "title": "Create Strategy (Guided)",
                    "error": "Symbol is required.",
                    "draft_id": draft_id,
                    "draft": draft,
                    "preview": _builder_v2_preview(draft),
                },
            )

        # Risk
        try:
            risk_per_trade_pct = float(str(form_dict.get("risk_per_trade_pct") or "1.0").strip())
            if risk_per_trade_pct <= 0:
                raise ValueError("Risk per trade (%) must be > 0")
        except Exception as e:
            return TEMPLATES.TemplateResponse(
                request,
                "guided_builder_v2.html",
                {
                    "title": "Create Strategy (Guided)",
                    "error": str(e) or "Invalid risk per trade (%)",
                    "draft_id": draft_id,
                    "draft": draft,
                    "preview": _builder_v2_preview(draft),
                },
            )

        draft["symbol"] = symbol
        draft["long_enabled"] = bool(form_dict.get("long_enabled"))
        draft["short_enabled"] = bool(form_dict.get("short_enabled"))
        draft["risk_per_trade_pct"] = float(risk_per_trade_pct)

        if not draft["long_enabled"] and not draft["short_enabled"]:
            return TEMPLATES.TemplateResponse(
                request,
                "guided_builder_v2.html",
                {
                    "title": "Create Strategy (Guided)",
                    "error": "Enable LONG and/or SHORT.",
                    "draft_id": draft_id,
                    "draft": draft,
                    "preview": _builder_v2_preview(draft),
                },
            )

        entry_tf = str(form_dict.get("entry_tf") or draft.get("entry_tf") or "1h").strip()
        draft["entry_tf"] = entry_tf

        signal_tf = str(form_dict.get("signal_tf") or "").strip()
        trigger_tf = str(form_dict.get("trigger_tf") or "").strip()
        draft["signal_tf"] = signal_tf
        draft["trigger_tf"] = trigger_tf

        context_tf = str(form_dict.get("context_tf") or "").strip()
        primary_context_type = str(form_dict.get("primary_context_type") or draft.get("primary_context_type") or "ma_stack").strip().lower()
        align_with_context = bool(form_dict.get("align_with_context"))
        ma_type = str(form_dict.get("ma_type") or "ema").strip().lower()
        ma_fast = _parse_int("ma_fast", int(draft.get("ma_fast") or 20))
        ma_mid = _parse_int("ma_mid", int(draft.get("ma_mid") or 50))
        ma_slow = _parse_int("ma_slow", int(draft.get("ma_slow") or 200))
        stack_mode = str(form_dict.get("stack_mode") or "none").strip().lower()
        slope_mode = str(form_dict.get("slope_mode") or "none").strip().lower()
        slope_lookback = _parse_int("slope_lookback", int(draft.get("slope_lookback") or 10))
        min_ma_dist_pct = _parse_float("min_ma_dist_pct", float(draft.get("min_ma_dist_pct") or 0.0))

        trigger_type = str(form_dict.get("trigger_type") or "pin_bar").strip().lower()
        trigger_valid_for_bars = _parse_int(
            "trigger_valid_for_bars", int(draft.get("trigger_valid_for_bars") or 0)
        )
        pin_wick_body = _parse_float("pin_wick_body", float(draft.get("pin_wick_body") or 2.0))
        pin_opp_wick_body_max = _parse_float(
            "pin_opp_wick_body_max", float(draft.get("pin_opp_wick_body_max") or 1.0)
        )
        pin_min_body_pct = _parse_float("pin_min_body_pct", float(draft.get("pin_min_body_pct") or 0.2))

        trigger_ma_type = str(form_dict.get("trigger_ma_type") or draft.get("trigger_ma_type") or "ema").strip().lower()
        trigger_ma_len = _parse_int("trigger_ma_len", int(draft.get("trigger_ma_len") or 20))
        trigger_don_len = _parse_int("trigger_don_len", int(draft.get("trigger_don_len") or 20))
        trigger_range_len = _parse_int("trigger_range_len", int(draft.get("trigger_range_len") or 20))
        trigger_atr_len = _parse_int("trigger_atr_len", int(draft.get("trigger_atr_len") or 14))
        trigger_atr_mult = _parse_float("trigger_atr_mult", float(draft.get("trigger_atr_mult") or 2.0))
        trigger_custom_bull_expr = str(
            form_dict.get("trigger_custom_bull_expr") or draft.get("trigger_custom_bull_expr") or ""
        ).strip()
        trigger_custom_bear_expr = str(
            form_dict.get("trigger_custom_bear_expr") or draft.get("trigger_custom_bear_expr") or ""
        ).strip()

        context_rules = _parse_rules_json("context_rules_json")
        signal_rules = _parse_rules_json("signal_rules_json")
        extra_trigger_rules = _parse_rules_json("trigger_rules_json")

        draft.update(
            {
                "context_tf": context_tf,
                "primary_context_type": primary_context_type,
                "align_with_context": align_with_context,
                "ma_type": ma_type,
                "ma_fast": ma_fast,
                "ma_mid": ma_mid,
                "ma_slow": ma_slow,
                "stack_mode": stack_mode,
                "slope_mode": slope_mode,
                "slope_lookback": slope_lookback,
                "min_ma_dist_pct": min_ma_dist_pct,
                "trigger_type": trigger_type,
                "trigger_valid_for_bars": trigger_valid_for_bars,
                "pin_wick_body": pin_wick_body,
                "pin_opp_wick_body_max": pin_opp_wick_body_max,
                "pin_min_body_pct": pin_min_body_pct,
                "trigger_ma_type": trigger_ma_type,
                "trigger_ma_len": trigger_ma_len,
                "trigger_don_len": trigger_don_len,
                "trigger_range_len": trigger_range_len,
                "trigger_atr_len": trigger_atr_len,
                "trigger_atr_mult": trigger_atr_mult,
                "trigger_custom_bull_expr": trigger_custom_bull_expr,
                "trigger_custom_bear_expr": trigger_custom_bear_expr,
                "context_rules": context_rules,
                "signal_rules": signal_rules,
                "trigger_rules": extra_trigger_rules,
            }
        )

        # Context conditions on context_tf (or entry_tf if blank)
        ctx_tf = context_tf or None
        fast = _ma_expr(ma_type, ma_fast, ctx_tf)
        mid = _ma_expr(ma_type, ma_mid, ctx_tf)
        slow = _ma_expr(ma_type, ma_slow, ctx_tf)

        def _context_exprs(direction: str) -> list[str]:
            d = direction.strip().lower()
            out: list[str] = []

            # Stack requirement
            if stack_mode in {"bull", "bear"}:
                if d == "bull":
                    out.append(f"{fast} > {mid} and {mid} > {slow}")
                elif d == "bear":
                    out.append(f"{fast} < {mid} and {mid} < {slow}")

            # Slope requirement (on mid)
            if slope_mode in {"up", "down"}:
                if d == "bull":
                    out.append(f"{mid} > shift({mid}, {int(slope_lookback)})")
                elif d == "bear":
                    out.append(f"{mid} < shift({mid}, {int(slope_lookback)})")

            # Distance requirement (directional)
            if min_ma_dist_pct and min_ma_dist_pct > 0:
                if d == "bull":
                    out.append(f"({fast} - {slow}) / {slow} > {float(min_ma_dist_pct)}")
                elif d == "bear":
                    out.append(f"({slow} - {fast}) / {slow} > {float(min_ma_dist_pct)}")

            return out

        # Build per-side context expressions.
        # If alignment is ON and both sides enabled, we automatically invert direction for SHORT.
        long_context_exprs: list[str] = []
        short_context_exprs: list[str] = []

        if draft["long_enabled"]:
            if align_with_context and draft["short_enabled"]:
                long_context_exprs = _context_exprs("bull")
            else:
                # single-direction mode: respect user direction selection
                if stack_mode == "bear" or slope_mode == "down":
                    long_context_exprs = _context_exprs("bear")
                else:
                    long_context_exprs = _context_exprs("bull")

        if draft["short_enabled"]:
            if align_with_context and draft["long_enabled"]:
                short_context_exprs = _context_exprs("bear")
            else:
                if stack_mode == "bull" or slope_mode == "up":
                    short_context_exprs = _context_exprs("bull")
                else:
                    short_context_exprs = _context_exprs("bear")

        # Alignment semantics (Builder v2):
        # - If align_with_context is ON and BOTH directions are enabled,
        #   apply bullish context constraints to LONG and bearish constraints to SHORT.
        # - If only one direction is enabled, apply whichever directional constraints the user picked.
        # This matches the common discretionary workflow: “trade long in bull regime, short in bear regime”.

        # Compile final entry conditions from the previewed effective rules.
        preview = _builder_v2_preview(draft)

        def _collect_exprs(side_key: str) -> tuple[list[str], Optional[str]]:
            side = preview.get(side_key) or {}
            if not isinstance(side, dict):
                return ([], None)

            enabled = bool(side.get("enabled"))
            if not enabled:
                return ([], None)

            exprs: list[str] = []
            for section_key in ("context", "signals", "triggers"):
                items = side.get(section_key) or []
                if not isinstance(items, list):
                    continue
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    expr = str(it.get("expr") or "").strip()
                    if expr in {"(unknown)", "(empty)", ""}:
                        label = str(it.get("label") or section_key)
                        return ([], f"Invalid {section_key} rule: {label}")
                    exprs.append(expr)
            return (exprs, None)

        def _infer_tf(expr: str, default_tf: str) -> str:
            try:
                from research.dsl import extract_timeframe_refs

                tfs = extract_timeframe_refs(expr, default_tf=default_tf)
                if not tfs:
                    return default_tf
                # If multiple are referenced, pick one deterministically.
                return sorted(tfs)[0]
            except Exception:
                return default_tf

        long_exprs, long_err = _collect_exprs("long")
        if long_err:
            return TEMPLATES.TemplateResponse(
                request,
                "guided_builder_v2.html",
                {
                    "title": "Create Strategy (Guided)",
                    "error": long_err,
                    "draft_id": draft_id,
                    "draft": draft,
                    "preview": preview,
                },
            )

        short_exprs, short_err = _collect_exprs("short")
        if short_err:
            return TEMPLATES.TemplateResponse(
                request,
                "guided_builder_v2.html",
                {
                    "title": "Create Strategy (Guided)",
                    "error": short_err,
                    "draft_id": draft_id,
                    "draft": draft,
                    "preview": preview,
                },
            )

        long_conds: list[dict[str, str]] = []
        short_conds: list[dict[str, str]] = []

        if draft["long_enabled"]:
            for e in long_exprs:
                long_conds.append({"tf": _infer_tf(e, entry_tf), "expr": e})

        if draft["short_enabled"]:
            for e in short_exprs:
                short_conds.append({"tf": _infer_tf(e, entry_tf), "expr": e})

        if draft["long_enabled"] and not long_conds:
            return TEMPLATES.TemplateResponse(
                request,
                "guided_builder_v2.html",
                {
                    "title": "Create Strategy (Guided)",
                    "error": "Please provide at least one LONG entry condition.",
                    "draft_id": draft_id,
                    "draft": draft,
                    "preview": preview,
                },
            )
        if draft["short_enabled"] and not short_conds:
            return TEMPLATES.TemplateResponse(
                request,
                "guided_builder_v2.html",
                {
                    "title": "Create Strategy (Guided)",
                    "error": "Please provide at least one SHORT entry condition.",
                    "draft_id": draft_id,
                    "draft": draft,
                    "preview": preview,
                },
            )

        draft["long_conds"] = long_conds
        draft["short_conds"] = short_conds

        url = f"/create-strategy-guided/step5?draft_id={draft_id}"
        return TEMPLATES.TemplateResponse(
            request,
            "guided_redirect.html",
            {
                "title": "Create Strategy (Guided)",
                "url": url,
                "message": "Opening Advanced settings (optional)…",
                "draft": draft,
            },
        )

    # Allow custom symbol if step2 picked CUSTOM
    symbol = str(form_dict.get("symbol") or draft.get("symbol") or "").strip().upper()
    if not symbol:
        return TEMPLATES.TemplateResponse(
            request,
            "guided_step3.html",
            {
                "title": "Create Strategy (Guided)",
                "error": "Symbol is required.",
                "draft_id": draft_id,
                "draft": draft,
            },
        )

    draft["symbol"] = symbol

    draft["long_enabled"] = bool(form_dict.get("long_enabled"))
    draft["short_enabled"] = bool(form_dict.get("short_enabled"))

    draft["stop_type"] = str(form_dict.get("stop_type") or "atr").strip().lower()
    try:
        draft["stop_percent"] = float(str(form_dict.get("stop_percent") or "0.02").strip())
    except Exception:
        draft["stop_percent"] = 0.02
    try:
        draft["atr_length"] = int(str(form_dict.get("atr_length") or "14").strip())
    except Exception:
        draft["atr_length"] = 14
    try:
        draft["atr_multiplier"] = float(str(form_dict.get("atr_multiplier") or "3.0").strip())
    except Exception:
        draft["atr_multiplier"] = 3.0

    try:
        long_conds: list[dict[str, str]] = []
        short_conds: list[dict[str, str]] = []

        if draft["long_enabled"]:
            for pref in ("long1", "long2"):
                c = _expr_from_condition(pref, form_dict)
                if c is not None:
                    long_conds.append(c)

        if draft["short_enabled"]:
            for pref in ("short1", "short2"):
                c = _expr_from_condition(pref, form_dict)
                if c is not None:
                    short_conds.append(c)

        if draft["long_enabled"] and not long_conds:
            raise ValueError("Please provide at least one LONG entry condition.")
        if draft["short_enabled"] and not short_conds:
            raise ValueError("Please provide at least one SHORT entry condition.")
        if not long_conds and not short_conds:
            raise ValueError("Enable LONG and/or SHORT and provide an entry condition.")

        draft["long_conds"] = long_conds
        draft["short_conds"] = short_conds
    except Exception as e:
        return TEMPLATES.TemplateResponse(
            request,
            "guided_step3.html",
            {
                "title": "Create Strategy (Guided)",
                "error": str(e),
                "draft_id": draft_id,
                "draft": draft,
            },
        )

    # Use a visible redirect page to avoid "nothing happened" perception on some browsers/extensions.
    url = f"/create-strategy-guided/step5?draft_id={draft_id}"
    return TEMPLATES.TemplateResponse(
        request,
        "guided_redirect.html",
        {
            "title": "Create Strategy (Guided)",
            "url": url,
            "message": "Opening Advanced settings (optional)…",
            "draft": draft,
        },
    )


@app.get("/create-strategy-guided/step5", response_class=HTMLResponse)
def create_strategy_guided_step5_advanced(request: Request, draft_id: str):
    draft = DRAFTS.get(draft_id)
    if not draft or draft.get("kind") != "guided":
        return RedirectResponse(url="/create-strategy-guided", status_code=303)

    master_calendar = _get_master_calendar_defaults()
    master_tm = _get_master_trade_management_defaults()

    advanced_defaults: dict[str, Any] | None = None
    try:
        from config.advanced_builder_defaults import resolve_advanced_builder_defaults

        sym = str(draft.get("symbol") or "").strip()
        if sym:
            advanced_defaults = resolve_advanced_builder_defaults(sym).as_dict()
    except Exception:
        advanced_defaults = None

    return TEMPLATES.TemplateResponse(
        request,
        "guided_step5_advanced.html",
        {
            "title": "Create Strategy (Guided)",
            "draft_id": draft_id,
            "draft": draft,
            "master_calendar": master_calendar,
            "master_tm": master_tm,
            "advanced_defaults": advanced_defaults,
        },
    )


@app.post("/create-strategy-guided/step5", response_class=HTMLResponse)
async def create_strategy_guided_step5_advanced_submit(
    request: Request,
    draft_id: str = Form(...),
    action: str = Form("save"),
):
    draft = DRAFTS.get(draft_id)
    if not draft or draft.get("kind") != "guided":
        return RedirectResponse(url="/create-strategy-guided", status_code=303)

    form = await request.form()
    form_dict = {k: v for k, v in form.items()}

    master_calendar = _get_master_calendar_defaults()
    master_tm = _get_master_trade_management_defaults()
    master_allowed_days = list(master_calendar.get("allowed_days") or [0, 1, 2, 3, 4])
    master_sessions_cfg = master_calendar.get("sessions") if isinstance(master_calendar, dict) else None
    if not isinstance(master_sessions_cfg, dict):
        master_sessions_cfg = {
            "Asia": {"enabled": True, "start": "23:00", "end": "08:00"},
            "London": {"enabled": True, "start": "07:00", "end": "16:00"},
            "NewYork": {"enabled": True, "start": "13:00", "end": "21:00"},
        }

    advanced_defaults: dict[str, Any] | None = None
    try:
        from config.advanced_builder_defaults import resolve_advanced_builder_defaults

        sym = str(draft.get("symbol") or "").strip()
        if sym:
            advanced_defaults = resolve_advanced_builder_defaults(sym).as_dict()
    except Exception:
        advanced_defaults = None

    def _normalize_hhmm(v: str) -> str:
        s = str(v or "").strip()
        if not s:
            return ""
        # Accept HH:MM or HH:MM:SS and normalize to HH:MM.
        parts = s.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid time '{s}' (expected HH:MM)")
        try:
            hh = int(parts[0])
            mm = int(parts[1])
        except Exception:
            raise ValueError(f"Invalid time '{s}' (expected HH:MM)")
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            raise ValueError(f"Invalid time '{s}' (expected HH:MM)")
        return f"{hh:02d}:{mm:02d}"

    def _session_default_times(name: str) -> tuple[str, str]:
        cfg = master_sessions_cfg.get(name) if isinstance(master_sessions_cfg, dict) else None
        if not isinstance(cfg, dict):
            return ("", "")
        return (str(cfg.get("start") or ""), str(cfg.get("end") or ""))

    def _reset_to_defaults() -> None:
        draft["calendar_filters_mode"] = "master"
        draft["calendar_filters"] = {"master_filters_enabled": True}
        draft["calendar_allowed_days"] = master_allowed_days
        draft["calendar_sessions"] = {
            "Asia": bool((master_sessions_cfg.get("Asia") or {}).get("enabled", True)),
            "London": bool((master_sessions_cfg.get("London") or {}).get("enabled", True)),
            "NewYork": bool((master_sessions_cfg.get("NewYork") or {}).get("enabled", True)),
        }
        a_start, a_end = _session_default_times("Asia")
        l_start, l_end = _session_default_times("London")
        n_start, n_end = _session_default_times("NewYork")
        draft["calendar_session_times"] = {
            "Asia": {"start": a_start, "end": a_end},
            "London": {"start": l_start, "end": l_end},
            "NewYork": {"start": n_start, "end": n_end},
        }

        draft["trade_management"] = None
        draft["tm_tp_mode"] = "master"
        draft["tm_partial_enabled"] = bool((master_tm.get("partial_exit") or {}).get("enabled", False))
        draft["tm_trailing_enabled"] = bool((master_tm.get("trailing_stop") or {}).get("enabled", False))

        draft.pop("exit_ctx_enabled", None)
        draft.pop("exit_ctx_mode", None)

    if str(action).strip().lower() == "skip":
        _reset_to_defaults()
        return RedirectResponse(url=f"/create-strategy-guided/review?draft_id={draft_id}", status_code=303)

    try:
        # --- Risk / daily limits / backtest cost overrides ---
        sizing_mode = str(form_dict.get("sizing_mode") or draft.get("sizing_mode") or "account_size").strip().lower()
        if sizing_mode not in {"account_size", "daily_equity", "equity"}:
            sizing_mode = "account_size"

        try:
            account_size = float(str(form_dict.get("account_size") or draft.get("account_size") or "10000").strip())
        except Exception:
            account_size = float(draft.get("account_size") or 10000.0)
        if account_size <= 0:
            raise ValueError("Account size must be > 0")

        try:
            max_trades_per_day = int(str(form_dict.get("max_trades_per_day") or draft.get("max_trades_per_day") or "1").strip())
        except Exception:
            max_trades_per_day = int(draft.get("max_trades_per_day") or 1)
        if max_trades_per_day <= 0:
            raise ValueError("Max trades per day must be >= 1")

        max_daily_loss_pct = None
        if bool(form_dict.get("max_daily_loss_enabled")):
            try:
                max_daily_loss_pct = float(str(form_dict.get("max_daily_loss_pct") or "").strip())
            except Exception:
                raise ValueError("Max daily loss (%) must be a number")
            if max_daily_loss_pct <= 0:
                raise ValueError("Max daily loss (%) must be > 0")

        commissions = None
        if bool(form_dict.get("override_commissions")):
            raw = str(form_dict.get("commissions") or "").strip()
            if raw == "":
                raise ValueError("Commissions override is enabled but commissions is blank")
            try:
                commissions = float(raw)
            except Exception:
                raise ValueError("Commissions must be a number")
            if commissions < 0:
                raise ValueError("Commissions must be >= 0")

        slippage_ticks = None
        if bool(form_dict.get("override_slippage")):
            raw = str(form_dict.get("slippage_ticks") or "").strip()
            if raw == "":
                raise ValueError("Slippage override is enabled but slippage is blank")
            try:
                slippage_ticks = float(raw)
            except Exception:
                raise ValueError("Slippage must be a number")
            if slippage_ticks < 0:
                raise ValueError("Slippage must be >= 0")

        draft["sizing_mode"] = sizing_mode
        draft["account_size"] = float(account_size)
        draft["max_trades_per_day"] = int(max_trades_per_day)
        draft["max_daily_loss_pct"] = max_daily_loss_pct
        draft["commissions"] = commissions
        draft["slippage_ticks"] = slippage_ticks

        # --- Calendar filters ---
        calendar_mode = str(form_dict.get("calendar_mode") or "disable").strip().lower()
        if calendar_mode not in {"disable", "master", "custom"}:
            calendar_mode = "disable"

        draft["calendar_filters_mode"] = calendar_mode

        if calendar_mode == "disable":
            calendar_filters = {"master_filters_enabled": False}
        elif calendar_mode == "master":
            calendar_filters = {"master_filters_enabled": True}
        else:
            dow_raw = form.getlist("dow")
            allowed_days: list[int] = []
            for d in dow_raw:
                try:
                    di = int(str(d).strip())
                except Exception:
                    continue
                if 0 <= di <= 6 and di not in allowed_days:
                    allowed_days.append(di)

            if not allowed_days:
                raise ValueError("Please select at least one allowed day (Mon–Sun).")

            sessions_enabled = {
                "Asia": bool(form_dict.get("session_Asia")),
                "London": bool(form_dict.get("session_London")),
                "NewYork": bool(form_dict.get("session_NewYork")),
            }

            # Session window override (stored in UTC).
            a_start_raw = form_dict.get("session_Asia_start")
            a_end_raw = form_dict.get("session_Asia_end")
            l_start_raw = form_dict.get("session_London_start")
            l_end_raw = form_dict.get("session_London_end")
            n_start_raw = form_dict.get("session_NewYork_start")
            n_end_raw = form_dict.get("session_NewYork_end")

            a_start_def, a_end_def = _session_default_times("Asia")
            l_start_def, l_end_def = _session_default_times("London")
            n_start_def, n_end_def = _session_default_times("NewYork")

            a_start = _normalize_hhmm(a_start_raw) if a_start_raw is not None else _normalize_hhmm(a_start_def)
            a_end = _normalize_hhmm(a_end_raw) if a_end_raw is not None else _normalize_hhmm(a_end_def)
            l_start = _normalize_hhmm(l_start_raw) if l_start_raw is not None else _normalize_hhmm(l_start_def)
            l_end = _normalize_hhmm(l_end_raw) if l_end_raw is not None else _normalize_hhmm(l_end_def)
            n_start = _normalize_hhmm(n_start_raw) if n_start_raw is not None else _normalize_hhmm(n_start_def)
            n_end = _normalize_hhmm(n_end_raw) if n_end_raw is not None else _normalize_hhmm(n_end_def)

            if sessions_enabled.get("Asia") and (not a_start or not a_end):
                raise ValueError("Asia session requires start and end times")
            if sessions_enabled.get("London") and (not l_start or not l_end):
                raise ValueError("London session requires start and end times")
            if sessions_enabled.get("NewYork") and (not n_start or not n_end):
                raise ValueError("NewYork session requires start and end times")

            calendar_filters = {
                "master_filters_enabled": True,
                "day_of_week": {"enabled": True, "allowed_days": allowed_days},
                "trading_sessions_enabled": True,
                "trading_sessions": {
                    "Asia": {"enabled": sessions_enabled["Asia"], "start": a_start, "end": a_end},
                    "London": {"enabled": sessions_enabled["London"], "start": l_start, "end": l_end},
                    "NewYork": {"enabled": sessions_enabled["NewYork"], "start": n_start, "end": n_end},
                },
            }

            draft["calendar_allowed_days"] = allowed_days
            draft["calendar_sessions"] = sessions_enabled
            draft["calendar_session_times"] = {
                "Asia": {"start": a_start, "end": a_end},
                "London": {"start": l_start, "end": l_end},
                "NewYork": {"start": n_start, "end": n_end},
            }

        draft["calendar_filters"] = calendar_filters

        # --- Stop loss ---
        stop_separate = bool(form_dict.get("stop_separate"))
        draft["stop_separate"] = stop_separate

        allowed_stop_types = {"atr", "percent", "ma", "structure", "candle"}

        def _float_field(key: str, default: float) -> float:
            raw = form_dict.get(key)
            if raw is None or str(raw).strip() == "":
                return default
            return float(str(raw).strip())

        def _int_field(key: str, default: int) -> int:
            raw = form_dict.get(key)
            if raw is None or str(raw).strip() == "":
                return default
            return int(str(raw).strip())

        def _parse_stop(prefix: str | None) -> dict[str, Any]:
            p = (str(prefix) + "_") if prefix else ""

            t_key = f"{p}stop_type"
            stop_type = str(form_dict.get(t_key) or draft.get(t_key) or draft.get("stop_type") or "atr").strip().lower()
            if stop_type not in allowed_stop_types:
                stop_type = "atr"

            out: dict[str, Any] = {"stop_type": stop_type}

            if stop_type == "percent":
                sp = _float_field(f"{p}stop_percent", default=float(draft.get(f"{p}stop_percent") or draft.get("stop_percent") or 0.02))
                if sp <= 0:
                    raise ValueError(f"{prefix.upper() + ' ' if prefix else ''}Stop percent must be > 0")
                out["stop_percent"] = float(sp)

            elif stop_type == "atr":
                al = _int_field(f"{p}atr_length", default=int(draft.get(f"{p}atr_length") or draft.get("atr_length") or 14))
                am = _float_field(
                    f"{p}atr_multiplier",
                    default=float(draft.get(f"{p}atr_multiplier") or draft.get("atr_multiplier") or 3.0),
                )
                if al <= 0:
                    raise ValueError(f"{prefix.upper() + ' ' if prefix else ''}ATR length must be >= 1")
                if am <= 0:
                    raise ValueError(f"{prefix.upper() + ' ' if prefix else ''}ATR multiplier must be > 0")
                out["atr_length"] = int(al)
                out["atr_multiplier"] = float(am)

            elif stop_type == "ma":
                ma_type = str(form_dict.get(f"{p}stop_ma_type") or draft.get(f"{p}stop_ma_type") or draft.get("stop_ma_type") or "ema").strip().lower()
                if ma_type not in {"ema", "sma"}:
                    ma_type = "ema"
                ma_len = _int_field(f"{p}stop_ma_length", default=int(draft.get(f"{p}stop_ma_length") or draft.get("stop_ma_length") or 50))
                buf = _float_field(f"{p}stop_buffer", default=float(draft.get(f"{p}stop_buffer") or draft.get("stop_buffer") or 0.0))
                if ma_len <= 0:
                    raise ValueError(f"{prefix.upper() + ' ' if prefix else ''}MA length must be >= 1")
                if buf < 0:
                    raise ValueError(f"{prefix.upper() + ' ' if prefix else ''}Stop buffer must be >= 0")
                out["stop_ma_type"] = ma_type
                out["stop_ma_length"] = int(ma_len)
                out["stop_buffer"] = float(buf)

            elif stop_type == "structure":
                lb = _int_field(
                    f"{p}stop_structure_lookback_bars",
                    default=int(draft.get(f"{p}stop_structure_lookback_bars") or draft.get("stop_structure_lookback_bars") or 20),
                )
                buf = _float_field(f"{p}stop_buffer", default=float(draft.get(f"{p}stop_buffer") or draft.get("stop_buffer") or 0.0))
                if lb <= 0:
                    raise ValueError(f"{prefix.upper() + ' ' if prefix else ''}Structure lookback bars must be >= 1")
                if buf < 0:
                    raise ValueError(f"{prefix.upper() + ' ' if prefix else ''}Stop buffer must be >= 0")
                out["stop_structure_lookback_bars"] = int(lb)
                out["stop_buffer"] = float(buf)

            elif stop_type == "candle":
                bb = _int_field(
                    f"{p}stop_candle_bars_back",
                    default=int(draft.get(f"{p}stop_candle_bars_back") or draft.get("stop_candle_bars_back") or 1),
                )
                buf = _float_field(f"{p}stop_buffer", default=float(draft.get(f"{p}stop_buffer") or draft.get("stop_buffer") or 0.0))
                if bb <= 0:
                    raise ValueError(f"{prefix.upper() + ' ' if prefix else ''}Candle bars-back must be >= 1")
                if buf < 0:
                    raise ValueError(f"{prefix.upper() + ' ' if prefix else ''}Stop buffer must be >= 0")
                out["stop_candle_bars_back"] = int(bb)
                out["stop_buffer"] = float(buf)

            return out

        if stop_separate:
            long_stop = _parse_stop("long")
            short_stop = _parse_stop("short")

            draft["long_stop_type"] = long_stop.get("stop_type")
            draft["long_stop_percent"] = long_stop.get("stop_percent")
            draft["long_atr_length"] = long_stop.get("atr_length")
            draft["long_atr_multiplier"] = long_stop.get("atr_multiplier")
            draft["long_stop_ma_type"] = long_stop.get("stop_ma_type")
            draft["long_stop_ma_length"] = long_stop.get("stop_ma_length")
            draft["long_stop_structure_lookback_bars"] = long_stop.get("stop_structure_lookback_bars")
            draft["long_stop_candle_bars_back"] = long_stop.get("stop_candle_bars_back")
            draft["long_stop_buffer"] = long_stop.get("stop_buffer")

            draft["short_stop_type"] = short_stop.get("stop_type")
            draft["short_stop_percent"] = short_stop.get("stop_percent")
            draft["short_atr_length"] = short_stop.get("atr_length")
            draft["short_atr_multiplier"] = short_stop.get("atr_multiplier")
            draft["short_stop_ma_type"] = short_stop.get("stop_ma_type")
            draft["short_stop_ma_length"] = short_stop.get("stop_ma_length")
            draft["short_stop_structure_lookback_bars"] = short_stop.get("stop_structure_lookback_bars")
            draft["short_stop_candle_bars_back"] = short_stop.get("stop_candle_bars_back")
            draft["short_stop_buffer"] = short_stop.get("stop_buffer")

        else:
            shared_stop = _parse_stop(None)
            draft["stop_type"] = shared_stop.get("stop_type")
            draft["stop_percent"] = shared_stop.get("stop_percent")
            draft["atr_length"] = shared_stop.get("atr_length")
            draft["atr_multiplier"] = shared_stop.get("atr_multiplier")
            draft["stop_ma_type"] = shared_stop.get("stop_ma_type")
            draft["stop_ma_length"] = shared_stop.get("stop_ma_length")
            draft["stop_structure_lookback_bars"] = shared_stop.get("stop_structure_lookback_bars")
            draft["stop_candle_bars_back"] = shared_stop.get("stop_candle_bars_back")
            draft["stop_buffer"] = shared_stop.get("stop_buffer")

        # --- Trade management ---
        trade_management: dict[str, Any] = {}

        tp_mode = str(form_dict.get("tp_mode") or "master").strip().lower()
        if tp_mode not in {"master", "disable", "custom"}:
            tp_mode = "master"
        draft["tm_tp_mode"] = tp_mode

        if tp_mode == "disable":
            trade_management["take_profit"] = {"enabled": False, "levels": []}
        elif tp_mode == "custom":
            tp_template = str(form_dict.get("tp_template") or "").strip() or None
            if tp_template:
                draft["tm_tp_template"] = tp_template

            levels: list[dict[str, Any]] = []

            # Prefer the new multi-level fields; fall back to legacy single-level inputs.
            any_new = any((form_dict.get(f"tp{i}_target_r") is not None) for i in (1, 2, 3))
            if any_new:
                for i in (1, 2, 3):
                    if not bool(form_dict.get(f"tp{i}_enabled")):
                        continue
                    target_r = _float_field(f"tp{i}_target_r", default=0.0)
                    exit_pct = _float_field(f"tp{i}_exit_pct", default=0.0)
                    if target_r <= 0:
                        raise ValueError(f"Take profit Level {i} Target R must be > 0")
                    if exit_pct <= 0 or exit_pct > 100:
                        raise ValueError(f"Take profit Level {i} Exit % must be between 1 and 100")
                    levels.append({"enabled": True, "target_r": target_r, "exit_pct": exit_pct})
            else:
                target_r = _float_field("tp_target_r", default=3.0)
                exit_pct = _float_field("tp_exit_pct", default=100.0)
                if target_r <= 0:
                    raise ValueError("Take profit Target R must be > 0")
                if exit_pct <= 0 or exit_pct > 100:
                    raise ValueError("Take profit Exit % must be between 1 and 100")
                levels = [{"enabled": True, "target_r": target_r, "exit_pct": exit_pct}]

            if not levels:
                raise ValueError("Please enable at least one take-profit level.")

            total_exit = 0.0
            for lvl in levels:
                try:
                    total_exit += float(lvl.get("exit_pct") or 0.0)
                except Exception:
                    pass
            if total_exit > 100.0 + 1e-9:
                raise ValueError("Take profit Exit % total must be ≤ 100")

            draft["tm_tp_levels"] = levels
            # Keep legacy draft fields populated for continuity.
            try:
                draft["tm_tp_target_r"] = float(levels[0]["target_r"])
                draft["tm_tp_exit_pct"] = float(levels[0]["exit_pct"])
            except Exception:
                pass

            trade_management["take_profit"] = {"enabled": True, "levels": levels}

        partial_enabled = bool(form_dict.get("partial_enabled"))
        draft["tm_partial_enabled"] = partial_enabled
        if partial_enabled:
            pe_template = str(form_dict.get("partial_template") or "").strip() or None
            if pe_template:
                draft["tm_partial_template"] = pe_template

            levels: list[dict[str, Any]] = []
            any_new = any((form_dict.get(f"partial{i}_level_r") is not None) for i in (1, 2))
            if any_new:
                for i in (1, 2):
                    if not bool(form_dict.get(f"partial{i}_enabled")):
                        continue
                    try:
                        level_r = float(str(form_dict.get(f"partial{i}_level_r") or "").strip())
                    except Exception:
                        raise ValueError(f"Partial exit Level {i} Level R must be a number")
                    try:
                        exit_pct = float(str(form_dict.get(f"partial{i}_exit_pct") or "").strip())
                    except Exception:
                        raise ValueError(f"Partial exit Level {i} Exit % must be a number")
                    if level_r <= 0:
                        raise ValueError(f"Partial exit Level {i} Level R must be > 0")
                    if exit_pct <= 0 or exit_pct > 100:
                        raise ValueError(f"Partial exit Level {i} Exit % must be between 1 and 100")
                    levels.append({"enabled": True, "level_r": level_r, "exit_pct": exit_pct})
            else:
                try:
                    level_r = float(str(form_dict.get("partial_level_r") or "1.5").strip())
                except Exception:
                    level_r = 1.5
                try:
                    exit_pct = float(str(form_dict.get("partial_exit_pct") or "50.0").strip())
                except Exception:
                    exit_pct = 50.0
                if level_r <= 0:
                    raise ValueError("Partial exit Level R must be > 0")
                if exit_pct <= 0 or exit_pct > 100:
                    raise ValueError("Partial exit Exit % must be between 1 and 100")
                levels = [{"enabled": True, "level_r": level_r, "exit_pct": exit_pct}]

            if not levels:
                raise ValueError("Please enable at least one partial-exit level.")

            total_exit = 0.0
            for lvl in levels:
                try:
                    total_exit += float(lvl.get("exit_pct") or 0.0)
                except Exception:
                    pass
            if total_exit > 100.0 + 1e-9:
                raise ValueError("Partial exit Exit % total must be ≤ 100")

            draft["tm_partial_levels"] = levels
            try:
                draft["tm_partial_level_r"] = float(levels[0]["level_r"])
                draft["tm_partial_exit_pct"] = float(levels[0]["exit_pct"])
            except Exception:
                pass

            trade_management["partial_exit"] = {"enabled": True, "levels": levels}

        trailing_enabled = bool(form_dict.get("trailing_enabled"))
        draft["tm_trailing_enabled"] = trailing_enabled
        if trailing_enabled:
            try:
                length = int(str(form_dict.get("trailing_length") or "21").strip())
            except Exception:
                length = 21
            try:
                activation_r = float(str(form_dict.get("trailing_activation_r") or "0.5").strip())
            except Exception:
                activation_r = 0.5
            stepped = bool(form_dict.get("trailing_stepped"))
            try:
                min_move_pips = float(str(form_dict.get("trailing_min_move_pips") or "7.0").strip())
            except Exception:
                min_move_pips = 7.0

            if length <= 0:
                raise ValueError("Trailing stop length must be >= 1")
            if activation_r < 0:
                raise ValueError("Trailing activation R must be >= 0")
            if min_move_pips < 0:
                raise ValueError("Trailing min move must be >= 0")

            draft["tm_trailing_length"] = length
            draft["tm_trailing_activation_r"] = activation_r
            draft["tm_trailing_stepped"] = stepped
            draft["tm_trailing_min_move_pips"] = min_move_pips

            trade_management["trailing_stop"] = {
                "enabled": True,
                "type": "EMA",
                "length": length,
                "activation_type": "r_based",
                "activation_r": activation_r,
                "stepped": stepped,
                "min_move_pips": min_move_pips,
            }

        draft["trade_management"] = trade_management if trade_management else None

        # --- Execution (Phase 4) ---
        exit_ctx_enabled = bool(form_dict.get("exit_ctx_enabled"))
        exit_ctx_mode = str(form_dict.get("exit_ctx_mode") or "immediate").strip().lower()
        if exit_ctx_mode not in {"immediate", "tighten_stop"}:
            exit_ctx_mode = "immediate"
        draft["exit_ctx_enabled"] = exit_ctx_enabled
        draft["exit_ctx_mode"] = exit_ctx_mode

    except Exception as e:
        return TEMPLATES.TemplateResponse(
            request,
            "guided_step5_advanced.html",
            {
                "title": "Create Strategy (Guided)",
                "error": str(e),
                "draft_id": draft_id,
                "draft": draft,
                "master_calendar": master_calendar,
                "master_tm": master_tm,
                "advanced_defaults": advanced_defaults,
            },
        )

    return RedirectResponse(url=f"/create-strategy-guided/review?draft_id={draft_id}", status_code=303)


@app.get("/create-strategy-guided/review", response_class=HTMLResponse)
def create_strategy_guided_review(request: Request, draft_id: str, message: str | None = None):
    draft = DRAFTS.get(draft_id)
    if not draft or draft.get("kind") != "guided":
        return RedirectResponse(url="/create-strategy-guided", status_code=303)

    try:
        calendar_filters = draft.get("calendar_filters") or {"master_filters_enabled": True}
        trade_management = draft.get("trade_management")

        spec_dict = {
            "name": str(draft["name"]),
            "description": f"(guided template: {draft.get('template')})",
            "market": {
                "symbol": str(draft.get("symbol") or "UNKNOWN"),
                "exchange": str(draft.get("exchange") or "unknown"),
                "market_type": str(draft.get("market_type") or "spot"),
            },
            "entry_tf": str(draft.get("entry_tf") or "1h"),
            "context_tf": (str(draft.get("context_tf") or "").strip() or None),
            "extra_tfs": [],
            "filters": {"calendar_filters": calendar_filters},
            "execution": {
                "flatten_enabled": bool(draft.get("flatten_enabled")) if draft.get("flatten_enabled") is not None else None,
                "flatten_time": draft.get("flatten_time"),
                "stop_signal_search_minutes_before": draft.get("stop_signal_search_minutes_before"),
                "use_intraday_margin": draft.get("use_intraday_margin"),
            },
            "risk_per_trade_pct": float(draft.get("risk_per_trade_pct") or 1.0),
        }

        # Risk / limits / cost overrides
        spec_dict["sizing_mode"] = str(draft.get("sizing_mode") or "account_size")
        spec_dict["account_size"] = float(draft.get("account_size") or 10000.0)
        spec_dict["max_trades_per_day"] = int(draft.get("max_trades_per_day") or 1)
        if draft.get("max_daily_loss_pct") is not None:
            spec_dict["max_daily_loss_pct"] = float(draft.get("max_daily_loss_pct"))
        if draft.get("commissions") is not None:
            spec_dict["commissions"] = float(draft.get("commissions"))
        if draft.get("slippage_ticks") is not None:
            spec_dict["slippage_ticks"] = float(draft.get("slippage_ticks"))

        if draft.get("exit_ctx_enabled"):
            try:
                mode = str(draft.get("exit_ctx_mode") or "immediate").strip().lower()
            except Exception:
                mode = "immediate"
            if mode not in {"immediate", "tighten_stop"}:
                mode = "immediate"
            spec_dict["execution"]["exit_if_context_invalid"] = {"enabled": True, "mode": mode}

        # Keep extra_tfs informative for humans (context_tf is a common extra tf).
        try:
            ctx_tf = spec_dict.get("context_tf")
            entry_tf = spec_dict.get("entry_tf")
            if ctx_tf and ctx_tf != entry_tf:
                spec_dict["extra_tfs"] = [ctx_tf]
        except Exception:
            pass

        if trade_management:
            spec_dict["trade_management"] = trade_management

        stop_separate = bool(draft.get("stop_separate"))
        allowed_stop_types = {"atr", "percent", "ma", "structure", "candle"}

        def _stop_payload(prefix: str | None) -> dict[str, Any]:
            p = (str(prefix) + "_") if prefix else ""
            st = str(draft.get(f"{p}stop_type") or draft.get("stop_type") or "atr").strip().lower()
            if st not in allowed_stop_types:
                st = "atr"

            out: dict[str, Any] = {
                "stop_type": st,
                "stop_percent": None,
                "atr_length": None,
                "atr_multiplier": None,
                "stop_ma_type": None,
                "stop_ma_length": None,
                "stop_buffer": None,
                "stop_structure_lookback_bars": None,
                "stop_candle_bars_back": None,
            }

            if st == "percent":
                out["stop_percent"] = float(draft.get(f"{p}stop_percent") or draft.get("stop_percent") or 0.02)
            elif st == "atr":
                out["atr_length"] = int(draft.get(f"{p}atr_length") or draft.get("atr_length") or 14)
                out["atr_multiplier"] = float(draft.get(f"{p}atr_multiplier") or draft.get("atr_multiplier") or 3.0)
            elif st == "ma":
                out["stop_ma_type"] = str(draft.get(f"{p}stop_ma_type") or draft.get("stop_ma_type") or "ema")
                out["stop_ma_length"] = int(draft.get(f"{p}stop_ma_length") or draft.get("stop_ma_length") or 50)
                out["stop_buffer"] = float(draft.get(f"{p}stop_buffer") or draft.get("stop_buffer") or 0.0)
            elif st == "structure":
                out["stop_structure_lookback_bars"] = int(
                    draft.get(f"{p}stop_structure_lookback_bars") or draft.get("stop_structure_lookback_bars") or 20
                )
                out["stop_buffer"] = float(draft.get(f"{p}stop_buffer") or draft.get("stop_buffer") or 0.0)
            elif st == "candle":
                out["stop_candle_bars_back"] = int(draft.get(f"{p}stop_candle_bars_back") or draft.get("stop_candle_bars_back") or 1)
                out["stop_buffer"] = float(draft.get(f"{p}stop_buffer") or draft.get("stop_buffer") or 0.0)

            return out

        if draft.get("long_enabled"):
            stop_cfg = _stop_payload("long") if stop_separate else _stop_payload(None)
            spec_dict["long"] = {
                "enabled": True,
                "conditions_all": list(draft.get("long_conds") or []),
                "exit_conditions_all": [],
                **stop_cfg,
            }
        if draft.get("short_enabled"):
            stop_cfg = _stop_payload("short") if stop_separate else _stop_payload(None)
            spec_dict["short"] = {
                "enabled": True,
                "conditions_all": list(draft.get("short_conds") or []),
                "exit_conditions_all": [],
                **stop_cfg,
            }

        spec = StrategySpec.model_validate(spec_dict)

        yaml_text = _spec_to_yaml_compact(spec)
    except Exception as e:
        return TEMPLATES.TemplateResponse(
            request,
            "guided_step3.html",
            {
                "title": "Create Strategy (Guided)",
                "error": f"Invalid configuration: {e}",
                "draft_id": draft_id,
                "draft": draft,
            },
        )

    return TEMPLATES.TemplateResponse(
        request,
        "guided_review.html",
        {
            "title": "Create Strategy (Guided)",
            "draft_id": draft_id,
            "draft": draft,
            "yaml_text": yaml_text,
            "message": message,
        },
    )


@app.post("/create-strategy-guided/run", response_class=HTMLResponse)
async def create_strategy_guided_run(
    request: Request,
    draft_id: str = Form(...),
    compile_strategy: str | None = Form(None),
    edit_reason: str | None = Form(None),
    save_as_new_version: str | None = Form(None),
):
    draft = DRAFTS.get(draft_id)
    if not draft or draft.get("kind") != "guided":
        return RedirectResponse(url="/create-strategy-guided", status_code=303)

    want_version_copy = bool(draft.get("edit_mode")) and bool(save_as_new_version)

    # Persist the change note into the draft so a redirect doesn't lose it.
    try:
        note = str(edit_reason or "").strip()
        if note:
            draft["edit_reason"] = note
    except Exception:
        pass

    # Persist UI choice into the draft so a redirect doesn't lose it.
    try:
        if bool(draft.get("edit_mode")):
            draft["save_as_new_version"] = bool(want_version_copy)
    except Exception:
        pass

    # Build + validate spec, then run compiler via existing script
    calendar_filters = draft.get("calendar_filters") or {"master_filters_enabled": True}
    trade_management = draft.get("trade_management")

    spec_dict = {
        "name": str(draft["name"]),
        "description": f"(guided template: {draft.get('template')})",
        "market": {
            "symbol": str(draft.get("symbol") or "UNKNOWN"),
            "exchange": str(draft.get("exchange") or "unknown"),
            "market_type": str(draft.get("market_type") or "spot"),
        },
        "entry_tf": str(draft.get("entry_tf") or "1h"),
        "context_tf": (str(draft.get("context_tf") or "").strip() or None),
        "extra_tfs": [],
        "filters": {"calendar_filters": calendar_filters},
        "execution": {
            "flatten_enabled": bool(draft.get("flatten_enabled")) if draft.get("flatten_enabled") is not None else None,
            "flatten_time": draft.get("flatten_time"),
            "stop_signal_search_minutes_before": draft.get("stop_signal_search_minutes_before"),
            "use_intraday_margin": draft.get("use_intraday_margin"),
        },
        "risk_per_trade_pct": float(draft.get("risk_per_trade_pct") or 1.0),
    }

    # Risk / limits / cost overrides
    spec_dict["sizing_mode"] = str(draft.get("sizing_mode") or "account_size")
    spec_dict["account_size"] = float(draft.get("account_size") or 10000.0)
    spec_dict["max_trades_per_day"] = int(draft.get("max_trades_per_day") or 1)
    if draft.get("max_daily_loss_pct") is not None:
        spec_dict["max_daily_loss_pct"] = float(draft.get("max_daily_loss_pct"))
    if draft.get("commissions") is not None:
        spec_dict["commissions"] = float(draft.get("commissions"))
    if draft.get("slippage_ticks") is not None:
        spec_dict["slippage_ticks"] = float(draft.get("slippage_ticks"))

    if draft.get("exit_ctx_enabled"):
        try:
            mode = str(draft.get("exit_ctx_mode") or "immediate").strip().lower()
        except Exception:
            mode = "immediate"
        if mode not in {"immediate", "tighten_stop"}:
            mode = "immediate"
        spec_dict["execution"]["exit_if_context_invalid"] = {"enabled": True, "mode": mode}

    # Keep extra_tfs informative for humans (context_tf is a common extra tf).
    try:
        ctx_tf = spec_dict.get("context_tf")
        entry_tf = spec_dict.get("entry_tf")
        if ctx_tf and ctx_tf != entry_tf:
            spec_dict["extra_tfs"] = [ctx_tf]
    except Exception:
        pass

    if trade_management:
        spec_dict["trade_management"] = trade_management

    stop_separate = bool(draft.get("stop_separate"))
    allowed_stop_types = {"atr", "percent", "ma", "structure", "candle"}

    def _stop_payload(prefix: str | None) -> dict[str, Any]:
        p = (str(prefix) + "_") if prefix else ""
        st = str(draft.get(f"{p}stop_type") or draft.get("stop_type") or "atr").strip().lower()
        if st not in allowed_stop_types:
            st = "atr"

        out: dict[str, Any] = {
            "stop_type": st,
            "stop_percent": None,
            "atr_length": None,
            "atr_multiplier": None,
            "stop_ma_type": None,
            "stop_ma_length": None,
            "stop_buffer": None,
            "stop_structure_lookback_bars": None,
            "stop_candle_bars_back": None,
        }

        if st == "percent":
            out["stop_percent"] = float(draft.get(f"{p}stop_percent") or draft.get("stop_percent") or 0.02)
        elif st == "atr":
            out["atr_length"] = int(draft.get(f"{p}atr_length") or draft.get("atr_length") or 14)
            out["atr_multiplier"] = float(draft.get(f"{p}atr_multiplier") or draft.get("atr_multiplier") or 3.0)
        elif st == "ma":
            out["stop_ma_type"] = str(draft.get(f"{p}stop_ma_type") or draft.get("stop_ma_type") or "ema")
            out["stop_ma_length"] = int(draft.get(f"{p}stop_ma_length") or draft.get("stop_ma_length") or 50)
            out["stop_buffer"] = float(draft.get(f"{p}stop_buffer") or draft.get("stop_buffer") or 0.0)
        elif st == "structure":
            out["stop_structure_lookback_bars"] = int(
                draft.get(f"{p}stop_structure_lookback_bars") or draft.get("stop_structure_lookback_bars") or 20
            )
            out["stop_buffer"] = float(draft.get(f"{p}stop_buffer") or draft.get("stop_buffer") or 0.0)
        elif st == "candle":
            out["stop_candle_bars_back"] = int(draft.get(f"{p}stop_candle_bars_back") or draft.get("stop_candle_bars_back") or 1)
            out["stop_buffer"] = float(draft.get(f"{p}stop_buffer") or draft.get("stop_buffer") or 0.0)

        return out

    if draft.get("long_enabled"):
        stop_cfg = _stop_payload("long") if stop_separate else _stop_payload(None)
        spec_dict["long"] = {
            "enabled": True,
            "conditions_all": list(draft.get("long_conds") or []),
            "exit_conditions_all": [],
            **stop_cfg,
        }
    if draft.get("short_enabled"):
        stop_cfg = _stop_payload("short") if stop_separate else _stop_payload(None)
        spec_dict["short"] = {
            "enabled": True,
            "conditions_all": list(draft.get("short_conds") or []),
            "exit_conditions_all": [],
            **stop_cfg,
        }

    spec = StrategySpec.model_validate(spec_dict)

    # Enforce unique names in create mode; allow overwriting in edit mode.
    old_spec_name = ""
    if bool(draft.get("edit_mode")):
        old_spec_name = str(draft.get("old_spec_name") or spec.name or "").strip()

    # In edit mode, optionally save as a new version (keep original).
    # This writes as "create" (old_spec_name="") so the original is not overwritten or deleted.
    if want_version_copy:
        parent = old_spec_name
        requested = str(spec.name or "").strip()
        # If user kept the same name, auto-bump based on the original spec name.
        if parent and requested == parent:
            suggested = _suggest_next_version_name(REPO_ROOT, parent)
            if suggested:
                draft["name"] = suggested
                draft["version_parent"] = parent
                spec = spec.model_copy(update={"name": suggested})
        else:
            # User changed the name manually but still wants a copy.
            # Record lineage and ensure we don't delete the original.
            if parent:
                draft["version_parent"] = parent
        old_spec_name = ""

    try:
        repo_spec = _write_validated_spec_to_repo(spec=spec, old_spec_name=old_spec_name)
    except ValueError as e:
        import urllib.parse

        # If user is creating a new strategy (or saving an edited strategy as a new version)
        # and the name already exists, suggest/apply a versioned name.
        msg = str(e)
        if ((not bool(draft.get("edit_mode"))) or bool(draft.get("save_as_new_version"))) and (
            "already exists" in msg.lower()
        ):
            requested = str(draft.get("name") or "").strip()
            suggested = _suggest_next_version_name(REPO_ROOT, requested)
            if suggested and suggested != requested:
                draft["name"] = suggested
                if not str(draft.get("version_parent") or "").strip():
                    if bool(draft.get("save_as_new_version")):
                        parent = str(draft.get("old_spec_name") or "").strip() or requested
                        draft["version_parent"] = parent
                    else:
                        draft["version_parent"] = requested
                DRAFTS[draft_id] = draft
                msg = f"Strategy name already exists: {requested}. Suggested: {suggested} (applied). Review and click Create again."

        return RedirectResponse(
            url=f"/create-strategy-guided/review?draft_id={draft_id}&message={urllib.parse.quote(msg)}",
            status_code=303,
        )

    bundle = create_run_bundle(repo_root=REPO_ROOT, workflow="create_guided", strategy_name=str(draft["name"]))
    write_bundle_meta(bundle, python_executable=sys.executable, repo_root=REPO_ROOT)
    write_bundle_inputs(
        bundle,
        {
            "workflow": "create_strategy_guided",
            "draft": draft,
            "change_note": str(draft.get("edit_reason") or "").strip() or None,
            "edit_mode": bool(draft.get("edit_mode")),
            "save_as_new_version": bool(draft.get("save_as_new_version")),
            "old_spec_name": str(draft.get("old_spec_name") or "").strip() or None,
            "version_parent": str(draft.get("version_parent") or "").strip() or None,
        },
    )

    import yaml

    # Write spec to the run bundle (repo spec already persisted above).
    bundle_spec = bundle.dir / "spec.yml"
    bundle_spec.write_text(_spec_to_yaml_compact(spec))

    # Persist UI metadata so users can re-open the strategy using the same guided wizard later.
    _write_guided_meta(spec_name=str(spec.name), draft=draft)

    argv = [sys.executable, "scripts/build_strategy_from_spec.py", "--spec", str(repo_spec)]
    write_bundle_command(bundle, argv)

    job = runner.create(argv=argv, cwd=REPO_ROOT)
    JOB_META[job.id] = {
        "bundle_dir": str(bundle.dir),
        "workflow": "create_strategy_guided",
        "strategy_name": str(spec.name),
        "spec_path": str(repo_spec),
    }
    runner.start(job.id)

    return TEMPLATES.TemplateResponse(
        request,
        "job.html",
        {
            "title": "Running",
            "job_id": job.id,
            "next_hint": "create",
        },
    )


@app.get("/create-strategy-english", response_class=HTMLResponse)
def create_strategy_english_form(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "create_strategy_english.html",
        {
            "title": "Create Strategy (English)",
        },
    )


@app.post("/create-strategy-english/parse", response_class=HTMLResponse)
async def create_strategy_english_parse(
    request: Request,
    name: str = Form(...),
    entry_tf: str | None = Form(None),
    parse_mode: str = Form("strict"),
    compile_strategy: str | None = Form(None),
    english_text: str | None = Form(None),
    english_file: UploadFile | None = File(None),
):
    raw_text = (english_text or "").strip()
    if english_file is not None and (not raw_text):
        raw_text = (await english_file.read()).decode("utf-8", errors="replace").strip()

    if not raw_text:
        return TEMPLATES.TemplateResponse(
            request,
            "create_strategy_english.html",
            {
                "title": "Create Strategy (English)",
                "error": "Please paste English text or upload a .txt file.",
                "name": name,
                "entry_tf": entry_tf,
                "parse_mode": parse_mode,
                "compile_strategy": bool(compile_strategy),
                "english_text": raw_text,
            },
        )

    name_clean = str(name or "").strip()
    if not _is_safe_strategy_name(name_clean):
        return TEMPLATES.TemplateResponse(
            request,
            "create_strategy_english.html",
            {
                "title": "Create Strategy (English)",
                "error": "Invalid strategy name (use letters, numbers, '_' or '-').",
                "name": name_clean,
                "entry_tf": entry_tf,
                "parse_mode": parse_mode,
                "compile_strategy": bool(compile_strategy),
                "english_text": raw_text,
            },
        )

    existing = _safe_spec_path(name_clean)
    if existing is not None and existing.exists():
        return TEMPLATES.TemplateResponse(
            request,
            "create_strategy_english.html",
            {
                "title": "Create Strategy (English)",
                "error": f"Strategy name already exists: {name_clean}",
                "name": name_clean,
                "entry_tf": entry_tf,
                "parse_mode": parse_mode,
                "compile_strategy": bool(compile_strategy),
                "english_text": raw_text,
            },
        )

    result = parse_english_strategy(raw_text, name=name_clean, default_entry_tf=entry_tf, mode=parse_mode)  # type: ignore[arg-type]
    draft_id = str(uuid.uuid4())
    DRAFTS[draft_id] = {
        "name": name,
        "entry_tf": entry_tf,
        "parse_mode": parse_mode,
        "compile": bool(compile_strategy),
        "english": raw_text,
        "warnings": list(result.warnings),
        "clarifications": [_clarification_to_dict(c) for c in result.clarifications],
    }

    if result.clarifications:
        return TEMPLATES.TemplateResponse(
            request,
            "clarifications.html",
            {
                "title": "Clarifications",
                "draft_id": draft_id,
                "name": name,
                "warnings": list(result.warnings),
                "clarifications": [_clarification_to_dict(c) for c in result.clarifications],
            },
        )

    return RedirectResponse(url=f"/create-strategy-english/run?draft_id={draft_id}", status_code=303)


@app.get("/create-strategy-english/run", response_class=HTMLResponse)
def create_strategy_english_run(request: Request, draft_id: str):
    draft = DRAFTS.get(draft_id)
    if not draft:
        return RedirectResponse(url="/create-strategy-english", status_code=303)

    # No clarifications → run immediately with empty answers
    return _start_build_job(request, draft_id=draft_id, answers={})


@app.post("/create-strategy-english/clarifications", response_class=HTMLResponse)
async def create_strategy_english_clarifications_submit(request: Request, draft_id: str = Form(...)):
    draft = DRAFTS.get(draft_id)
    if not draft:
        return RedirectResponse(url="/create-strategy-english", status_code=303)

    form = await request.form()
    answers: dict[str, str] = {}
    for c in draft.get("clarifications", []) or []:
        key = str(c.get("key") or "")
        if not key:
            continue
        form_key = f"ans::{key}"
        val = form.get(form_key)
        if val is None:
            continue
        val_s = str(val).strip()
        if val_s:
            answers[key] = val_s

    return _start_build_job(request, draft_id=draft_id, answers=answers)


def _start_build_job(request: Request, *, draft_id: str, answers: dict[str, str]) -> HTMLResponse:
    draft = DRAFTS[draft_id]
    name = str(draft["name"])

    bundle = create_run_bundle(repo_root=REPO_ROOT, workflow="create_english", strategy_name=name)
    write_bundle_meta(bundle, python_executable=sys.executable, repo_root=REPO_ROOT)

    english_path = bundle.dir / "input_english.txt"
    english_path.write_text(str(draft["english"]))

    answers_path = bundle.dir / "answers.json"
    answers_path.write_text(json.dumps(answers, indent=2, sort_keys=True))

    inputs = {
        "workflow": "create_strategy_english",
        "name": name,
        "entry_tf": draft.get("entry_tf"),
        "parse_mode": draft.get("parse_mode"),
        "compile": bool(draft.get("compile")),
        "draft_id": draft_id,
        "answers": answers,
    }
    write_bundle_inputs(bundle, inputs)

    argv: list[str] = [
        sys.executable,
        "scripts/build_strategy_from_english.py",
        "--name",
        name,
        "--in",
        str(english_path),
        "--parse-mode",
        str(draft.get("parse_mode") or "strict"),
        "--answers-json",
        str(answers_path),
        "--non-interactive",
    ]

    if draft.get("entry_tf"):
        argv.extend(["--entry-tf", str(draft["entry_tf"])])

    if draft.get("compile"):
        argv.append("--compile")

    write_bundle_command(bundle, argv)

    job = runner.create(argv=argv, cwd=REPO_ROOT)
    # Stash bundle location on the draft for follow-up steps
    draft["bundle_dir"] = str(bundle.dir)
    draft["job_id"] = job.id
    JOB_META[job.id] = {
        "bundle_dir": str(bundle.dir),
        "workflow": "create_strategy_english",
        "strategy_name": name,
    }
    runner.start(job.id)

    return TEMPLATES.TemplateResponse(
        request,
        "job.html",
        {
            "title": "Running",
            "job_id": job.id,
            "next_hint": "create",
        },
    )


@app.get("/job/{job_id}", response_class=HTMLResponse)
def job_status(request: Request, job_id: str, next_hint: str | None = None):
    job = runner.get(job_id)
    if not job:
        return RedirectResponse(url="/", status_code=303)

    # If finished, extract outputs and persist logs to bundle when available.
    report_path = _extract_report_path(job.stdout)
    spec_path = _extract_path("✓ Wrote spec:", job.stdout)
    built_path = _extract_path("✓ Built strategy folder:", job.stdout)

    job.report_path = report_path
    job.spec_path = spec_path
    job.built_strategy_path = built_path

    meta = JOB_META.get(job_id) or {}
    bundle_dir = meta.get("bundle_dir")
    strategy_name = meta.get("strategy_name")

    if spec_path is None and meta.get("spec_path"):
        spec_path = str(meta.get("spec_path"))

    # Best-effort: persist logs to run bundle.
    if job.finished_at is not None and bundle_dir:
        try:
            from gui_launcher.bundles import RunBundle

            bundle = RunBundle(dir=Path(str(bundle_dir)))
            write_bundle_logs(bundle, stdout=job.stdout, stderr=job.stderr)
        except Exception:
            pass

    return TEMPLATES.TemplateResponse(
        request,
        "job_result.html",
        {
            "title": "Run Status",
            "job": job,
            "report_path": report_path,
            "spec_path": spec_path,
            "built_path": built_path,
            "bundle_dir": bundle_dir,
            "strategy_name": strategy_name,
        },
    )


@app.get("/backtest", response_class=HTMLResponse)
def backtest_form(request: Request, strategy: str | None = None):
    split_policy_names = _available_split_policy_names(REPO_ROOT)

    # Prefill slippage defaults based on the default download symbol.
    slippage_enabled_default = True
    slippage_default_value: float | None = None
    slippage_unit = "ticks"
    try:
        from config.advanced_builder_defaults import resolve_advanced_builder_defaults

        sym = "BTCUSDT"
        resolved = resolve_advanced_builder_defaults(sym).as_dict()
        bv = (resolved.get("backtest_validation") or {}) if isinstance(resolved, dict) else {}
        labels = (resolved.get("labels") or {}) if isinstance(resolved, dict) else {}

        slippage_enabled_default = bool((bv.get("slippage_enabled_default") is True) or (bv.get("slippage_enabled_default") is None))
        slippage_unit = str(labels.get("slippage_unit") or "ticks")

        # Prefer profile-provided slippage_ticks; fall back to slippage_default.
        if bv.get("slippage_ticks") is not None:
            slippage_default_value = float(bv.get("slippage_ticks"))
        elif bv.get("slippage_default") is not None:
            slippage_default_value = float(bv.get("slippage_default"))
    except Exception:
        slippage_enabled_default = True
        slippage_default_value = None
        slippage_unit = "ticks"

    return TEMPLATES.TemplateResponse(
        request,
        "backtest.html",
        {
            "title": "Backtest",
            "strategy": strategy or "",
            "split_policy_names": split_policy_names,
            "default_split_policy_name": "default_btcusdt_4h",
            "default_split_policy_mode": "auto",
            "slippage_enabled_default": slippage_enabled_default,
            "slippage_default_value": slippage_default_value,
            "slippage_unit": slippage_unit,
        },
    )


@app.get("/api/advanced-defaults", response_class=JSONResponse)
def api_advanced_defaults(symbol: str):
    sym = str(symbol or "").strip().upper()
    if not sym:
        return JSONResponse({"ok": False, "error": "Missing symbol"}, status_code=400)

    try:
        from config.advanced_builder_defaults import resolve_advanced_builder_defaults

        out = resolve_advanced_builder_defaults(sym).as_dict()
        if not isinstance(out, dict):
            return JSONResponse({"ok": False, "error": "Defaults resolver returned invalid payload"}, status_code=500)

        # Only return the parts the UI needs; keep this stable and small.
        labels = out.get("labels") if isinstance(out.get("labels"), dict) else {}
        bv = out.get("backtest_validation") if isinstance(out.get("backtest_validation"), dict) else {}

        payload = {
            "ok": True,
            "symbol": sym,
            "labels": {
                "distance_unit": labels.get("distance_unit"),
                "slippage_unit": labels.get("slippage_unit"),
                "timezone": labels.get("timezone"),
                "tick_size": labels.get("tick_size"),
            },
            "backtest_validation": {
                "slippage_enabled_default": bv.get("slippage_enabled_default"),
                "slippage_ticks": bv.get("slippage_ticks"),
                "slippage_default": bv.get("slippage_default"),
                "commission_rate": bv.get("commission_rate"),
                "commission_per_contract": bv.get("commission_per_contract"),
            },
        }
        return JSONResponse(payload)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.post("/backtest/run", response_class=HTMLResponse)
async def backtest_run(
    request: Request,
    strategy: str = Form(...),
    report_visuals: str | None = Form(None),
    slippage_enabled: str | None = Form(None),
    slippage_value: str | None = Form(None),
    data_file: UploadFile | None = File(None),
    download_data: str | None = Form(None),
    download_symbol: str = Form("BTCUSDT"),
    download_interval: str = Form("15m"),
    split_policy_mode: str = Form("auto"),
    split_policy_name: str = Form("default_btcusdt_4h"),
    override_split_policy: str | None = Form(None),
    start_date: str | None = Form(None),
    end_date: str | None = Form(None),
    market: str | None = Form(None),
    config_profile: str | None = Form(None),
):
    bundle = create_run_bundle(repo_root=REPO_ROOT, workflow="backtest", strategy_name=strategy)
    write_bundle_meta(bundle, python_executable=sys.executable, repo_root=REPO_ROOT)

    inputs = {
        "workflow": "backtest",
        "strategy": strategy,
        "report_visuals": bool(report_visuals),
        "slippage_enabled": bool(slippage_enabled),
        "download_data": bool(download_data),
        "download_symbol": download_symbol,
        "download_interval": download_interval,
        "split_policy_mode": split_policy_mode,
        "split_policy_name": split_policy_name,
        "override_split_policy": bool(override_split_policy),
        "start_date": start_date,
        "end_date": end_date,
        "market": market,
        "config_profile": config_profile,
    }

    # Resolve slippage default (market-aware) if enabled.
    resolved_slippage: float | None = None
    if bool(slippage_enabled):
        raw = str(slippage_value or "").strip()
        if raw:
            try:
                resolved_slippage = float(raw)
            except Exception:
                raise ValueError("Invalid slippage value")
        else:
            # Pick the best symbol hint we have.
            sym_hint: str | None = None
            mkt = str(market or "").strip()
            if mkt:
                sym_hint = mkt
            elif bool(download_data):
                sym_hint = str(download_symbol or "").strip()
            elif data_file is not None and data_file.filename:
                fn = str(data_file.filename)
                # Common patterns: EURUSD_M1_2025.csv, dataset_EURUSD_M1.csv
                base = Path(fn).name
                stem = base.rsplit(".", 1)[0]
                parts = stem.split("_")
                if parts and parts[0].lower() == "dataset" and len(parts) >= 2:
                    sym_hint = parts[1]
                elif parts:
                    sym_hint = parts[0]

            try:
                from config.advanced_builder_defaults import resolve_advanced_builder_defaults

                sym = str(sym_hint or "BTCUSDT").strip().upper()
                out = resolve_advanced_builder_defaults(sym).as_dict()
                bv = out.get("backtest_validation") or {}
                if bv.get("slippage_ticks") is not None:
                    resolved_slippage = float(bv.get("slippage_ticks"))
                elif bv.get("slippage_default") is not None:
                    resolved_slippage = float(bv.get("slippage_default"))
                else:
                    resolved_slippage = 0.0
            except Exception:
                resolved_slippage = 0.0

    inputs["slippage"] = resolved_slippage
    write_bundle_inputs(bundle, inputs)

    argv: list[str] = [
        sys.executable,
        "scripts/run_backtest.py",
        "--strategy",
        strategy,
        "--auto-resample",
    ]

    # Advanced options
    spm = str(split_policy_mode or "auto").strip().lower()
    if spm not in {"enforce", "auto", "none"}:
        spm = "auto"
    argv.extend(["--split-policy-mode", spm])

    spn = str(split_policy_name or "default_btcusdt_4h").strip()
    if spn:
        argv.extend(["--split-policy-name", spn])

    if override_split_policy:
        argv.append("--override-split-policy")

    sd = str(start_date).strip() if start_date else ""
    ed = str(end_date).strip() if end_date else ""
    if sd:
        argv.extend(["--start-date", sd])
    if ed:
        argv.extend(["--end-date", ed])

    mkt = str(market).strip() if market else ""
    if mkt:
        argv.extend(["--market", mkt])

    prof = str(config_profile).strip() if config_profile else ""
    if prof:
        argv.extend(["--config-profile", prof])

    if resolved_slippage is not None:
        argv.extend(["--slippage", str(float(resolved_slippage))])

    data_path = None
    if data_file is not None and data_file.filename:
        data_path = bundle.dir / ("dataset_" + Path(data_file.filename).name)
        data_path.write_bytes(await data_file.read())
        argv.extend(["--data", str(data_path)])

    if report_visuals:
        argv.append("--report-visuals")

    if download_data and data_path is None:
        argv.append("--download-data")
        argv.extend(["--download-symbol", download_symbol])
        argv.extend(["--download-interval", download_interval])

    write_bundle_command(bundle, argv)

    job = runner.create(argv=argv, cwd=REPO_ROOT)
    runner.start(job.id)

    JOB_META[job.id] = {
        "bundle_dir": str(bundle.dir),
        "workflow": "backtest",
        "strategy_name": strategy,
    }

    return TEMPLATES.TemplateResponse(
        request,
        "job.html",
        {
            "title": "Running backtest",
            "job_id": job.id,
            "next_hint": "backtest",
        },
    )


@app.post("/backtest/inspect", response_class=JSONResponse)
async def backtest_inspect_dataset(data_file: UploadFile = File(...)):
    """Inspect an uploaded dataset (CSV/Parquet) to infer date range/timeframe.

    This is used by the GUI to prefill start/end date inputs and to warn users
    when a dataset is clearly out-of-range for their intended backtest.
    """

    if not data_file.filename:
        return JSONResponse({"ok": False, "error": "No filename provided"}, status_code=400)

    tmp_dir = REPO_ROOT / "data" / "logs" / "gui_runs" / "_inspect"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(str(data_file.filename)).suffix
    tmp_path = tmp_dir / f"inspect_{uuid.uuid4().hex}{suffix}"

    try:
        tmp_path.write_bytes(await data_file.read())

        # Load using the same universal loader as the CLI backtest.
        from adapters.data.data_loader import DataLoader

        loader = DataLoader()

        # For inspection we want to preserve any extra columns (like symbol/ticker)
        # so we load raw + process timestamp, instead of calling loader.load() which
        # drops non-OHLCV columns.
        df_raw = loader._load_raw_data(tmp_path)
        df = loader._process_timestamp(df_raw, tmp_path)
        if len(df) == 0:
            return JSONResponse({"ok": False, "error": "Dataset loaded but has 0 rows"}, status_code=400)

        idx = df.index
        start_ts = idx.min()
        end_ts = idx.max()

        # Infer timeframe (best-effort)
        tf = None
        try:
            import pandas as pd

            freq = pd.infer_freq(idx)
            if freq:
                tf = str(freq)
            else:
                diffs = idx.to_series().diff().dropna().dt.total_seconds()
                if len(diffs) > 0:
                    med_s = float(diffs.median())
                    if med_s >= 86400 and med_s % 86400 == 0:
                        tf = f"{int(med_s // 86400)}D"
                    elif med_s >= 3600 and med_s % 3600 == 0:
                        tf = f"{int(med_s // 3600)}H"
                    elif med_s >= 60 and med_s % 60 == 0:
                        tf = f"{int(med_s // 60)}T"
        except Exception:
            tf = None

        def _infer_symbol_hint() -> tuple[str | None, str | None]:
            # 1) Try common column names (case-insensitive).
            try:
                candidates = ["symbol", "ticker", "instrument", "market", "pair"]
                cols = list(df.columns)
                col_lut = {str(c).strip().lower(): c for c in cols}
                for want in candidates:
                    if want in col_lut:
                        col = col_lut[want]
                        s = df[col]
                        try:
                            s = s.dropna()
                        except Exception:
                            continue
                        if len(s) == 0:
                            continue

                        # Normalize to strings.
                        try:
                            values = [str(x).strip() for x in s.tolist()[:5000]]
                        except Exception:
                            values = [str(x).strip() for x in list(s)[:5000]]
                        values = [v for v in values if v]
                        if not values:
                            continue
                        uniq = sorted(set(values))
                        if len(uniq) == 1:
                            return (uniq[0].upper(), f"column:{str(col)}")
            except Exception:
                pass

            # 2) Fallback: infer from filename.
            try:
                fn = str(data_file.filename or "")
                base = Path(fn).name
                stem = base.rsplit(".", 1)[0]
                parts = [p for p in stem.split("_") if p]
                for tok in parts:
                    t = str(tok).strip()
                    if not t:
                        continue
                    if t.lower() == "dataset":
                        continue
                    if t.isdigit() and len(t) == 4 and t.startswith("20"):
                        continue
                    # Skip timeframe-ish tokens like M1, H4, 15m
                    if re.match(r"^(m|h|d|w)\d+$", t, flags=re.IGNORECASE):
                        continue
                    if re.match(r"^\d+(m|min|h|d|w)$", t, flags=re.IGNORECASE):
                        continue
                    if re.match(r"^[A-Za-z0-9]{3,}$", t):
                        return (t.upper(), "filename")
            except Exception:
                pass

            return (None, None)

        symbol_hint, symbol_hint_source = _infer_symbol_hint()

        payload = {
            "ok": True,
            "rows": int(len(df)),
            "start": str(start_ts),
            "end": str(end_ts),
            "start_date": str(getattr(start_ts, "date", lambda: start_ts)()),
            "end_date": str(getattr(end_ts, "date", lambda: end_ts)()),
            "timeframe": tf,
            "symbol_hint": symbol_hint,
            "symbol_hint_source": symbol_hint_source,
        }
        return JSONResponse(payload)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=400)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/runs", response_class=HTMLResponse)
def runs_list(request: Request, message: str | None = None, page: int = 1, per_page: int = 50):
    base = REPO_ROOT / "data" / "logs" / "gui_runs"
    runs: list[Path] = []
    if base.exists():
        runs = [p for p in base.iterdir() if p.is_dir()]
        runs.sort(key=lambda p: p.name, reverse=True)

    page_i, per_page_i = _parse_pagination(page=page, per_page=per_page, max_per_page=200)
    run_page, total_runs = _paginate_list(runs, page=page_i, per_page=per_page_i)
    total_pages = max(1, (total_runs + per_page_i - 1) // per_page_i)
    if page_i > total_pages:
        page_i = total_pages
        run_page, total_runs = _paginate_list(runs, page=page_i, per_page=per_page_i)

    return TEMPLATES.TemplateResponse(
        request,
        "runs.html",
        {
            "title": "Runs",
            "runs": run_page,
            "message": message,
            "run_page": page_i,
            "run_per_page": per_page_i,
            "run_total": total_runs,
            "run_total_pages": total_pages,
        },
    )


@app.get("/runs/{run_name}/delete", response_class=HTMLResponse)
def run_delete_confirm(request: Request, run_name: str, message: str | None = None):
    run_dir = _safe_run_dir(run_name)
    if run_dir is None:
        return RedirectResponse(url="/runs", status_code=303)

    return TEMPLATES.TemplateResponse(
        request,
        "run_delete_confirm.html",
        {
            "title": f"Delete run: {run_name}",
            "run_name": run_name,
            "run_dir": str(run_dir),
            "message": message,
        },
    )


@app.post("/runs/{run_name}/delete", response_class=HTMLResponse)
async def run_delete(request: Request, run_name: str):
    run_dir = _safe_run_dir(run_name)
    if run_dir is None:
        return RedirectResponse(url="/runs", status_code=303)

    form = await request.form()
    confirm_name = str(form.get("confirm_name") or "").strip()
    if confirm_name != run_name:
        return RedirectResponse(url=f"/runs/{run_name}/delete?message=Type+the+run+name+to+confirm", status_code=303)

    try:
        shutil.rmtree(run_dir)
    except Exception:
        return RedirectResponse(url=f"/runs/{run_name}?message=Failed+to+delete+run", status_code=303)

    import urllib.parse

    return RedirectResponse(url=f"/runs?message=Deleted+run+{urllib.parse.quote(run_name)}", status_code=303)


@app.get("/runs/{run_name}", response_class=HTMLResponse)
def run_detail(request: Request, run_name: str):
    run_dir = _safe_run_dir(run_name)
    if run_dir is None:
        return RedirectResponse(url="/runs", status_code=303)

    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return ""

    inputs = _read_text(run_dir / "inputs.json")
    meta = _read_text(run_dir / "meta.json")
    command = _read_text(run_dir / "command.txt")
    stdout = _read_text(run_dir / "stdout.log")
    stderr = _read_text(run_dir / "stderr.log")

    report_path = _extract_report_path(stdout)
    report_href = None
    if report_path and report_path.endswith(".html"):
        report_href = "/reports/" + report_path.split("/")[-1]

    report_json_href, report_json = _report_json_from_report_path(REPO_ROOT, report_path)
    diagnosis = _quick_diagnosis(report_json, stderr=stderr)

    spec_exists = (run_dir / "spec.yml").exists()

    strategy_hint = None
    workflow = None
    try:
        inputs_obj = json.loads(inputs) if inputs else {}
        if isinstance(inputs_obj, dict):
            strategy_hint = inputs_obj.get("strategy") or inputs_obj.get("strategy_name")
            workflow = inputs_obj.get("workflow")
    except Exception:
        strategy_hint = None
        workflow = None

    can_rerun_backtest = bool(workflow == "backtest" and strategy_hint)

    return TEMPLATES.TemplateResponse(
        request,
        "run_detail.html",
        {
            "title": f"Run: {run_name}",
            "run_name": run_name,
            "run_dir": str(run_dir),
            "inputs": inputs,
            "meta": meta,
            "command": command,
            "stdout": stdout,
            "stderr": stderr,
            "report_href": report_href,
            "report_json_href": report_json_href,
            "diagnosis": diagnosis,
            "spec_exists": spec_exists,
            "strategy_hint": strategy_hint,
            "workflow": workflow,
            "can_rerun_backtest": can_rerun_backtest,
        },
    )


@app.get("/runs/{run_name}/export.zip")
def export_run_bundle_zip(run_name: str):
    """Download a small zip with everything needed to reproduce a GUI run.

    Intended for a 'zero support' workflow: users can attach this zip to an issue.
    """

    run_dir = _safe_run_dir(run_name)
    if run_dir is None:
        return RedirectResponse(url="/runs", status_code=303)

    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    # Collect common bundle files (best-effort)
    files = [
        (run_dir / "inputs.json", "inputs.json"),
        (run_dir / "meta.json", "meta.json"),
        (run_dir / "command.txt", "command.txt"),
        (run_dir / "stdout.log", "stdout.log"),
        (run_dir / "stderr.log", "stderr.log"),
        (run_dir / "spec.yml", "spec.yml"),
        (run_dir / "answers.json", "answers.json"),
        (run_dir / "input_english.txt", "input_english.txt"),
    ]

    stdout = _read_text(run_dir / "stdout.log")
    report_path = _extract_report_path(stdout)
    report_json_href, report_json = _report_json_from_report_path(REPO_ROOT, report_path)

    # Best-effort context for copy/paste issue reports
    inputs_text = _read_text(run_dir / "inputs.json")
    command_text = _read_text(run_dir / "command.txt")
    stderr_text = _read_text(run_dir / "stderr.log")
    workflow = None
    strategy = None
    try:
        obj = json.loads(inputs_text) if inputs_text else {}
        if isinstance(obj, dict):
            workflow = obj.get("workflow")
            strategy = obj.get("strategy") or obj.get("strategy_name")
    except Exception:
        workflow = None
        strategy = None

    env_payload = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "repo_root": str(REPO_ROOT),
        "run_name": run_name,
        "workflow": workflow,
        "strategy": strategy,
    }

    issue_md = "\n".join(
        [
            "# Bug report",
            "",
            "## Summary",
            "(1–2 sentences. What did you expect vs what happened?)",
            "",
            "## Reproduction",
            f"- Run: `{run_name}`",
            f"- Workflow: `{workflow}`" if workflow else "- Workflow: (unknown)",
            f"- Strategy: `{strategy}`" if strategy else "- Strategy: (unknown)",
            "- Attach the exported zip from the run detail page",
            "",
            "## Command (from bundle)",
            "```",
            command_text.strip(),
            "```",
            "",
            "## Stderr (if any)",
            "```",
            (stderr_text.strip()[:4000] + ("\n…(truncated)…" if len(stderr_text.strip()) > 4000 else "")),
            "```",
            "",
            "## Report",
            f"- report_path: `{report_path}`" if report_path else "- report_path: (none)",
            f"- report_json: `{report_json_href}`" if report_json_href else "- report_json: (none)",
        ]
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("run_name.txt", str(run_name))
        z.writestr("environment.json", json.dumps(env_payload, indent=2, sort_keys=True))
        z.writestr("ISSUE_TEMPLATE.md", issue_md)
        if report_path:
            z.writestr("report_path.txt", str(report_path))
        if report_json_href:
            z.writestr("report_json_href.txt", str(report_json_href))
        if report_json is not None:
            z.writestr("report.json", json.dumps(report_json, indent=2, sort_keys=True))

        for src, arc in files:
            try:
                if src.exists() and src.is_file():
                    # Avoid storing absolute paths inside the zip
                    z.write(src, arcname=arc)
            except Exception:
                pass

    buf.seek(0)
    headers = {
        "Content-Disposition": f'attachment; filename="{run_name}_bundle.zip"'
    }
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


@app.post("/runs/{run_name}/rerun", response_class=HTMLResponse)
def rerun_backtest_from_bundle(request: Request, run_name: str):
    run_dir = _safe_run_dir(run_name)
    if run_dir is None:
        return RedirectResponse(url="/runs", status_code=303)

    inputs_path = run_dir / "inputs.json"
    cmd_path = run_dir / "command.txt"
    if not inputs_path.exists():
        return RedirectResponse(url=f"/runs/{run_name}", status_code=303)

    try:
        inputs_obj = json.loads(inputs_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        inputs_obj = {}

    if not isinstance(inputs_obj, dict) or inputs_obj.get("workflow") != "backtest":
        return RedirectResponse(url=f"/runs/{run_name}", status_code=303)

    strategy = str(inputs_obj.get("strategy") or "").strip()
    if not strategy:
        return RedirectResponse(url=f"/runs/{run_name}", status_code=303)

    command_text = ""
    try:
        command_text = cmd_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        command_text = ""

    data_path = _extract_command_arg(command_text, "--data")
    if data_path and not Path(data_path).exists():
        data_path = None

    bundle = create_run_bundle(repo_root=REPO_ROOT, workflow="backtest_rerun", strategy_name=strategy)
    write_bundle_meta(bundle, python_executable=sys.executable, repo_root=REPO_ROOT)
    write_bundle_inputs(bundle, {"workflow": "backtest_rerun", "source_run": run_name, "inputs": inputs_obj})

    argv = _build_backtest_argv_from_inputs(inputs_obj, data_path=data_path)
    write_bundle_command(bundle, argv)

    job = runner.create(argv=argv, cwd=REPO_ROOT)
    runner.start(job.id)

    JOB_META[job.id] = {
        "bundle_dir": str(bundle.dir),
        "workflow": "backtest_rerun",
        "strategy_name": strategy,
    }

    return TEMPLATES.TemplateResponse(
        request,
        "job.html",
        {
            "title": "Rerunning backtest",
            "job_id": job.id,
            "next_hint": "backtest",
        },
    )


@app.get("/tune/{run_name}", response_class=HTMLResponse)
def tune_from_run(request: Request, run_name: str, preset: str | None = None):
    run_dir = _safe_run_dir(run_name)
    if run_dir is None:
        return RedirectResponse(url="/runs", status_code=303)

    master_calendar = _get_master_calendar_defaults()
    master_tm = _get_master_trade_management_defaults()

    spec_path = run_dir / "spec.yml"
    if not spec_path.exists():
        mc_days = list(master_calendar.get("allowed_days") or [0, 1, 2, 3, 4])
        mc_sessions_cfg = master_calendar.get("sessions") if isinstance(master_calendar, dict) else None
        if not isinstance(mc_sessions_cfg, dict):
            mc_sessions_cfg = {}
        mt_tp = master_tm.get("take_profit") if isinstance(master_tm, dict) else {}
        mt_pe = master_tm.get("partial_exit") if isinstance(master_tm, dict) else {}
        mt_ts = master_tm.get("trailing_stop") if isinstance(master_tm, dict) else {}
        return TEMPLATES.TemplateResponse(
            request,
            "tune.html",
            {
                "title": "Tune Strategy",
                "run_name": run_name,
                "error": "This run bundle has no spec.yml to tune.",
                "source_spec_rel": str(spec_path),
                "suggested_name": "strategy_v2",
                "risk_per_trade_pct": 1.0,
                "stop_type": "atr",
                "atr_length": 14,
                "atr_multiplier": 3.0,
                "stop_percent": 0.02,
                "calendar_mode": "master",
                "calendar_allowed_days": mc_days,
                "calendar_sessions": {
                    "Asia": bool((mc_sessions_cfg.get("Asia") or {}).get("enabled", True)),
                    "London": bool((mc_sessions_cfg.get("London") or {}).get("enabled", True)),
                    "NewYork": bool((mc_sessions_cfg.get("NewYork") or {}).get("enabled", True)),
                },
                "tp_mode": "master",
                "tp_target_r": float((mt_tp or {}).get("target_r", 3.0)),
                "tp_exit_pct": float((mt_tp or {}).get("exit_pct", 100.0)),
                "partial_enabled": bool((mt_pe or {}).get("enabled", False)),
                "partial_level_r": float((mt_pe or {}).get("level_r", 1.5)),
                "partial_exit_pct": float((mt_pe or {}).get("exit_pct", 50.0)),
                "trailing_enabled": bool((mt_ts or {}).get("enabled", False)),
                "trailing_length": int((mt_ts or {}).get("length", 21)),
                "trailing_activation_r": float((mt_ts or {}).get("activation_r", 0.5)),
                "trailing_stepped": bool((mt_ts or {}).get("stepped", True)),
                "trailing_min_move_pips": float((mt_ts or {}).get("min_move_pips", 7.0)),
                "master_calendar": master_calendar,
                "master_tm": master_tm,
            },
        )

    import yaml

    try:
        spec_data = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
        spec = StrategySpec.model_validate(spec_data)
    except Exception as e:
        mc_days = list(master_calendar.get("allowed_days") or [0, 1, 2, 3, 4])
        mc_sessions_cfg = master_calendar.get("sessions") if isinstance(master_calendar, dict) else None
        if not isinstance(mc_sessions_cfg, dict):
            mc_sessions_cfg = {}
        mt_tp = master_tm.get("take_profit") if isinstance(master_tm, dict) else {}
        mt_pe = master_tm.get("partial_exit") if isinstance(master_tm, dict) else {}
        mt_ts = master_tm.get("trailing_stop") if isinstance(master_tm, dict) else {}
        return TEMPLATES.TemplateResponse(
            request,
            "tune.html",
            {
                "title": "Tune Strategy",
                "run_name": run_name,
                "error": f"Invalid spec.yml: {e}",
                "source_spec_rel": str(spec_path),
                "suggested_name": "strategy_v2",
                "risk_per_trade_pct": 1.0,
                "stop_type": "atr",
                "atr_length": 14,
                "atr_multiplier": 3.0,
                "stop_percent": 0.02,
                "calendar_mode": "master",
                "calendar_allowed_days": mc_days,
                "calendar_sessions": {
                    "Asia": bool((mc_sessions_cfg.get("Asia") or {}).get("enabled", True)),
                    "London": bool((mc_sessions_cfg.get("London") or {}).get("enabled", True)),
                    "NewYork": bool((mc_sessions_cfg.get("NewYork") or {}).get("enabled", True)),
                },
                "tp_mode": "master",
                "tp_target_r": float((mt_tp or {}).get("target_r", 3.0)),
                "tp_exit_pct": float((mt_tp or {}).get("exit_pct", 100.0)),
                "partial_enabled": bool((mt_pe or {}).get("enabled", False)),
                "partial_level_r": float((mt_pe or {}).get("level_r", 1.5)),
                "partial_exit_pct": float((mt_pe or {}).get("exit_pct", 50.0)),
                "trailing_enabled": bool((mt_ts or {}).get("enabled", False)),
                "trailing_length": int((mt_ts or {}).get("length", 21)),
                "trailing_activation_r": float((mt_ts or {}).get("activation_r", 0.5)),
                "trailing_stepped": bool((mt_ts or {}).get("stepped", True)),
                "trailing_min_move_pips": float((mt_ts or {}).get("min_move_pips", 7.0)),
                "master_calendar": master_calendar,
                "master_tm": master_tm,
            },
        )

    # Pull current values from spec (best-effort)
    risk_per_trade_pct = float(getattr(spec, "risk_per_trade_pct", 1.0) or 1.0)

    def _stop_tuple(side_name: str) -> tuple[str, float, int, float]:
        side = getattr(spec, side_name, None)
        if side is None:
            return ("atr", 0.02, 14, 3.0)
        st = getattr(side, "stop_type", None) or "atr"
        sp = float(getattr(side, "stop_percent", None) or 0.02)
        al = int(getattr(side, "atr_length", None) or 14)
        am = float(getattr(side, "atr_multiplier", None) or 3.0)
        return (str(st), sp, al, am)

    stop_type, stop_percent, atr_length, atr_multiplier = _stop_tuple("long")

    calendar_filters = {}
    try:
        calendar_filters = (spec.filters.calendar_filters or {}) if getattr(spec, "filters", None) else {}
    except Exception:
        calendar_filters = {}

    # Derive calendar mode for UI
    calendar_mode = "master" if bool(calendar_filters.get("master_filters_enabled", True)) else "disable"
    day_cfg = (calendar_filters.get("day_of_week") or {}) if isinstance(calendar_filters, dict) else {}
    allowed_days = day_cfg.get("allowed_days") if isinstance(day_cfg, dict) else None
    if not isinstance(allowed_days, list) or not allowed_days:
        allowed_days = list(master_calendar.get("allowed_days") or [0, 1, 2, 3, 4])

    ts_cfg = calendar_filters.get("trading_sessions") if isinstance(calendar_filters, dict) else None
    sessions = {
        "Asia": bool(((master_calendar.get("sessions") or {}).get("Asia") or {}).get("enabled", True)),
        "London": bool(((master_calendar.get("sessions") or {}).get("London") or {}).get("enabled", True)),
        "NewYork": bool(((master_calendar.get("sessions") or {}).get("NewYork") or {}).get("enabled", True)),
    }
    if isinstance(ts_cfg, dict) and ts_cfg:
        for k in ["Asia", "London", "NewYork"]:
            v = ts_cfg.get(k)
            if isinstance(v, dict) and "enabled" in v:
                sessions[k] = bool(v.get("enabled"))

    tm = getattr(spec, "trade_management", None)
    tp_mode = "master"
    tp_target_r = float((master_tm.get("take_profit") or {}).get("target_r", 3.0))
    tp_exit_pct = float((master_tm.get("take_profit") or {}).get("exit_pct", 100.0))
    partial_enabled = bool((master_tm.get("partial_exit") or {}).get("enabled", False))
    partial_level_r = float((master_tm.get("partial_exit") or {}).get("level_r", 1.5))
    partial_exit_pct = float((master_tm.get("partial_exit") or {}).get("exit_pct", 50.0))
    trailing_enabled = bool((master_tm.get("trailing_stop") or {}).get("enabled", False))
    trailing_length = int((master_tm.get("trailing_stop") or {}).get("length", 21))
    trailing_activation_r = float((master_tm.get("trailing_stop") or {}).get("activation_r", 0.5))
    trailing_stepped = bool((master_tm.get("trailing_stop") or {}).get("stepped", True))
    trailing_min_move_pips = float((master_tm.get("trailing_stop") or {}).get("min_move_pips", 7.0))

    try:
        if tm is not None and tm.take_profit is not None and tm.take_profit.enabled is False:
            tp_mode = "disable"
        elif tm is not None and tm.take_profit is not None and (tm.take_profit.levels or []):
            tp_mode = "custom"
            lvl0 = tm.take_profit.levels[0]
            tp_target_r = float(lvl0.target_r)
            tp_exit_pct = float(lvl0.exit_pct)
    except Exception:
        pass

    try:
        if tm is not None and tm.partial_exit is not None and (tm.partial_exit.levels or []):
            partial_enabled = True
            lvl0 = tm.partial_exit.levels[0]
            partial_level_r = float(lvl0.level_r)
            partial_exit_pct = float(lvl0.exit_pct)
    except Exception:
        pass

    try:
        if tm is not None and tm.trailing_stop is not None and bool(tm.trailing_stop.enabled):
            trailing_enabled = True
            trailing_length = int(tm.trailing_stop.length)
            trailing_activation_r = float(tm.trailing_stop.activation_r)
            trailing_stepped = bool(tm.trailing_stop.stepped)
            trailing_min_move_pips = float(tm.trailing_stop.min_move_pips)
    except Exception:
        pass

    suggested_name = f"{spec.name}_v2" if getattr(spec, "name", None) else "strategy_v2"

    # Apply preset (UI-only; does not write until user clicks Save)
    preset = str(preset).strip().lower() if preset else None
    allowed_presets = {"lower_risk", "widen_stop", "disable_filters", "disable_trailing", "enable_partial"}
    if preset not in allowed_presets:
        preset = None

    if preset == "lower_risk":
        risk_per_trade_pct = max(0.1, round(risk_per_trade_pct * 0.5, 3))
    elif preset == "widen_stop":
        if str(stop_type) == "atr":
            atr_multiplier = float(atr_multiplier) + 0.5
        else:
            stop_percent = float(stop_percent) * 1.25
    elif preset == "disable_filters":
        calendar_mode = "disable"
    elif preset == "disable_trailing":
        trailing_enabled = False
    elif preset == "enable_partial":
        partial_enabled = True

    return TEMPLATES.TemplateResponse(
        request,
        "tune.html",
        {
            "title": "Tune Strategy",
            "run_name": run_name,
            "source_spec_rel": str(spec_path.relative_to(REPO_ROOT)) if str(spec_path).startswith(str(REPO_ROOT)) else str(spec_path),
            "suggested_name": suggested_name,
            "risk_per_trade_pct": risk_per_trade_pct,
            "stop_type": stop_type,
            "atr_length": atr_length,
            "atr_multiplier": atr_multiplier,
            "stop_percent": stop_percent,
            "calendar_mode": calendar_mode,
            "calendar_allowed_days": allowed_days,
            "calendar_sessions": sessions,
            "tp_mode": tp_mode,
            "tp_target_r": tp_target_r,
            "tp_exit_pct": tp_exit_pct,
            "partial_enabled": partial_enabled,
            "partial_level_r": partial_level_r,
            "partial_exit_pct": partial_exit_pct,
            "trailing_enabled": trailing_enabled,
            "trailing_length": trailing_length,
            "trailing_activation_r": trailing_activation_r,
            "trailing_stepped": trailing_stepped,
            "trailing_min_move_pips": trailing_min_move_pips,
            "preset": preset,
            "master_calendar": master_calendar,
            "master_tm": master_tm,
        },
    )


@app.post("/tune/run", response_class=HTMLResponse)
async def tune_from_run_submit(
    request: Request,
    run_name: str = Form(...),
    new_name: str = Form(...),
):
    run_dir = _safe_run_dir(run_name)
    if run_dir is None:
        return RedirectResponse(url="/runs", status_code=303)

    spec_path = run_dir / "spec.yml"
    if not spec_path.exists():
        return RedirectResponse(url=f"/tune/{run_name}", status_code=303)

    name_clean = str(new_name).strip()
    if not _is_safe_strategy_name(name_clean):
        return RedirectResponse(url=f"/tune/{run_name}", status_code=303)

    form = await request.form()
    form_dict = {k: v for k, v in form.items()}

    import yaml

    try:
        spec_data = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return TEMPLATES.TemplateResponse(
            request,
            "tune.html",
            {"title": "Tune Strategy", "run_name": run_name, "error": f"Could not read spec.yml: {e}"},
        )

    # --- Parse inputs ---
    try:
        risk_per_trade_pct = float(str(form_dict.get("risk_per_trade_pct") or "1.0").strip())
        if risk_per_trade_pct <= 0:
            raise ValueError("Risk per trade must be > 0")
    except Exception as e:
        return RedirectResponse(url=f"/tune/{run_name}", status_code=303)

    stop_type = str(form_dict.get("stop_type") or "atr").strip().lower()
    if stop_type not in {"atr", "percent"}:
        stop_type = "atr"

    stop_percent = None
    atr_length = None
    atr_multiplier = None

    if stop_type == "percent":
        try:
            stop_percent = float(str(form_dict.get("stop_percent") or "0.02").strip())
        except Exception:
            stop_percent = 0.02
        if stop_percent is None or stop_percent <= 0:
            return RedirectResponse(url=f"/tune/{run_name}", status_code=303)
    else:
        try:
            atr_length = int(str(form_dict.get("atr_length") or "14").strip())
        except Exception:
            atr_length = 14
        try:
            atr_multiplier = float(str(form_dict.get("atr_multiplier") or "3.0").strip())
        except Exception:
            atr_multiplier = 3.0
        if atr_length <= 0 or atr_multiplier <= 0:
            return RedirectResponse(url=f"/tune/{run_name}", status_code=303)

    calendar_mode = str(form_dict.get("calendar_mode") or "master").strip().lower()
    if calendar_mode not in {"master", "disable", "custom"}:
        calendar_mode = "master"

    master_calendar = _get_master_calendar_defaults()
    master_sessions_cfg = master_calendar.get("sessions") if isinstance(master_calendar, dict) else None
    if not isinstance(master_sessions_cfg, dict):
        master_sessions_cfg = {}

    if calendar_mode == "disable":
        calendar_filters = {"master_filters_enabled": False}
    elif calendar_mode == "master":
        calendar_filters = {"master_filters_enabled": True}
    else:
        dow_raw = form.getlist("dow")
        allowed_days: list[int] = []
        for d in dow_raw:
            try:
                di = int(str(d).strip())
            except Exception:
                continue
            if 0 <= di <= 6 and di not in allowed_days:
                allowed_days.append(di)
        if not allowed_days:
            return RedirectResponse(url=f"/tune/{run_name}", status_code=303)

        sessions_enabled = {
            "Asia": bool(form_dict.get("session_Asia")),
            "London": bool(form_dict.get("session_London")),
            "NewYork": bool(form_dict.get("session_NewYork")),
        }

        def _sess_times(name: str) -> tuple[str, str]:
            cfg = master_sessions_cfg.get(name)
            if not isinstance(cfg, dict):
                return ("", "")
            return (str(cfg.get("start") or ""), str(cfg.get("end") or ""))

        a_start, a_end = _sess_times("Asia")
        l_start, l_end = _sess_times("London")
        n_start, n_end = _sess_times("NewYork")

        calendar_filters = {
            "master_filters_enabled": True,
            "day_of_week": {"enabled": True, "allowed_days": allowed_days},
            "trading_sessions_enabled": True,
            "trading_sessions": {
                "Asia": {"enabled": sessions_enabled["Asia"], "start": a_start, "end": a_end},
                "London": {"enabled": sessions_enabled["London"], "start": l_start, "end": l_end},
                "NewYork": {"enabled": sessions_enabled["NewYork"], "start": n_start, "end": n_end},
            },
        }

    trade_management: dict[str, Any] = {}

    tp_mode = str(form_dict.get("tp_mode") or "master").strip().lower()
    if tp_mode not in {"master", "disable", "custom"}:
        tp_mode = "master"
    if tp_mode == "disable":
        trade_management["take_profit"] = {"enabled": False, "levels": []}
    elif tp_mode == "custom":
        try:
            target_r = float(str(form_dict.get("tp_target_r") or "3.0").strip())
        except Exception:
            target_r = 3.0
        try:
            exit_pct = float(str(form_dict.get("tp_exit_pct") or "100.0").strip())
        except Exception:
            exit_pct = 100.0
        if target_r <= 0 or exit_pct <= 0 or exit_pct > 100:
            return RedirectResponse(url=f"/tune/{run_name}", status_code=303)
        trade_management["take_profit"] = {
            "enabled": True,
            "levels": [{"enabled": True, "target_r": target_r, "exit_pct": exit_pct}],
        }

    partial_enabled = bool(form_dict.get("partial_enabled"))
    if partial_enabled:
        try:
            level_r = float(str(form_dict.get("partial_level_r") or "1.5").strip())
        except Exception:
            level_r = 1.5
        try:
            exit_pct = float(str(form_dict.get("partial_exit_pct") or "50.0").strip())
        except Exception:
            exit_pct = 50.0
        if level_r <= 0 or exit_pct <= 0 or exit_pct > 100:
            return RedirectResponse(url=f"/tune/{run_name}", status_code=303)
        trade_management["partial_exit"] = {
            "enabled": True,
            "levels": [{"enabled": True, "level_r": level_r, "exit_pct": exit_pct}],
        }

    trailing_enabled = bool(form_dict.get("trailing_enabled"))
    if trailing_enabled:
        try:
            length = int(str(form_dict.get("trailing_length") or "21").strip())
        except Exception:
            length = 21
        try:
            activation_r = float(str(form_dict.get("trailing_activation_r") or "0.5").strip())
        except Exception:
            activation_r = 0.5
        stepped = bool(form_dict.get("trailing_stepped"))
        try:
            min_move_pips = float(str(form_dict.get("trailing_min_move_pips") or "7.0").strip())
        except Exception:
            min_move_pips = 7.0
        if length <= 0 or activation_r < 0 or min_move_pips < 0:
            return RedirectResponse(url=f"/tune/{run_name}", status_code=303)
        trade_management["trailing_stop"] = {
            "enabled": True,
            "type": "EMA",
            "length": length,
            "activation_type": "r_based",
            "activation_r": activation_r,
            "stepped": stepped,
            "min_move_pips": min_move_pips,
        }

    # --- Apply changes to spec dict ---
    spec_data["name"] = name_clean
    spec_data["risk_per_trade_pct"] = float(risk_per_trade_pct)
    spec_data.setdefault("filters", {})
    if isinstance(spec_data["filters"], dict):
        spec_data["filters"]["calendar_filters"] = calendar_filters
    else:
        spec_data["filters"] = {"calendar_filters": calendar_filters}

    if trade_management:
        spec_data["trade_management"] = trade_management
    else:
        spec_data.pop("trade_management", None)

    for side_key in ("long", "short"):
        side = spec_data.get(side_key)
        if not isinstance(side, dict):
            continue
        if not bool(side.get("enabled", True)):
            continue
        side["stop_type"] = stop_type
        side["stop_percent"] = float(stop_percent) if stop_type == "percent" else None
        side["atr_length"] = int(atr_length) if stop_type == "atr" else None
        side["atr_multiplier"] = float(atr_multiplier) if stop_type == "atr" else None

    # Validate final spec and write to research_specs
    try:
        spec = StrategySpec.model_validate(spec_data)
    except Exception:
        return RedirectResponse(url=f"/tune/{run_name}", status_code=303)

    bundle = create_run_bundle(repo_root=REPO_ROOT, workflow="tune", strategy_name=spec.name)
    write_bundle_meta(bundle, python_executable=sys.executable, repo_root=REPO_ROOT)
    write_bundle_inputs(bundle, {"workflow": "tune", "source_run": run_name, "new_name": spec.name})

    # Write tuned spec into both bundle and repo research_specs
    bundle_spec = bundle.dir / "spec.yml"
    bundle_spec.write_text(_spec_to_yaml_compact(spec))

    research_specs_dir = REPO_ROOT / "research_specs"
    research_specs_dir.mkdir(parents=True, exist_ok=True)
    repo_spec = research_specs_dir / f"{spec.name}.yml"
    repo_spec.write_text(bundle_spec.read_text())

    argv = [sys.executable, "scripts/build_strategy_from_spec.py", "--spec", str(repo_spec)]
    write_bundle_command(bundle, argv)

    job = runner.create(argv=argv, cwd=REPO_ROOT)
    JOB_META[job.id] = {
        "bundle_dir": str(bundle.dir),
        "workflow": "tune",
        "strategy_name": str(spec.name),
        "spec_path": str(repo_spec),
    }
    runner.start(job.id)

    return TEMPLATES.TemplateResponse(
        request,
        "job.html",
        {
            "title": "Tuning + Rebuilding",
            "job_id": job.id,
            "next_hint": "backtest",
        },
    )
