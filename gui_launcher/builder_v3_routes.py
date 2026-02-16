"""
FastAPI router stubs for Builder V3 API endpoints.
These are minimal stubs returning example responses. Backend devs should replace logic with real implementations.
"""
from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Any
import sys
import logging
import json
from pathlib import Path
import uuid
from datetime import datetime
import time
import hmac
import hashlib
import os

# Simple in-memory rate limit store: {key: [timestamps]}
_RATE_LIMIT_STORE: dict[str, list[float]] = {}

from config.advanced_builder_defaults import _load_advanced_builder_defaults, _load_market_profiles, resolve_advanced_builder_defaults

router = APIRouter()


def _apply_rate_limit(request: Request, key: str, limit: int, window: int):
    """Simple sliding-window rate limiter keyed by client IP + endpoint key."""
    try:
        app_mod = sys.modules.get("gui_launcher.app")
        app_obj = getattr(app_mod, "app", None)
        if app_obj is None:
            return
        if not getattr(getattr(app_obj, "state", None), "builder_v3_rate_limit_enabled", False):
            return
        # client IP
        client_ip = "unknown"
        try:
            client_ip = request.client.host or "unknown"
        except Exception:
            pass
        key_name = f"{key}:{client_ip}"
        now = time.time()
        arr = _RATE_LIMIT_STORE.get(key_name) or []
        # drop old
        arr = [t for t in arr if t > now - window]
        if len(arr) >= limit:
            raise HTTPException(status_code=429, detail={"ok": False, "message": "Rate limit exceeded"})
        arr.append(now)
        _RATE_LIMIT_STORE[key_name] = arr
    except HTTPException:
        raise
    except Exception:
        # don't fail hard on rate limiter errors
        logging.exception("Rate limiter error")


def _verify_csrf_token(app_obj, token: str | None) -> bool:
    """Verify incoming CSRF token. Supports legacy static token on `app.state.builder_v3_csrf`
    or HMAC-signed rotating tokens using `app.state.builder_v3_csrf_secret`.
    Rotating tokens are of the form "{ts}.{sig_hex}" where sig_hex = HMAC(secret, ts).
    Accept tokens within a 15-minute window by default.
    """
    try:
        # Legacy static token support
        legacy = getattr(getattr(app_obj, "state", None), "builder_v3_csrf", None)
        if legacy and token == legacy:
            return True

        secret = getattr(getattr(app_obj, "state", None), "builder_v3_csrf_secret", None)
        if not secret or not token:
            return False
        parts = str(token).split(".")
        if len(parts) != 2:
            return False
        ts_s, sig = parts
        try:
            ts = int(ts_s)
        except Exception:
            return False
        # check freshness (default 15 minutes)
        window = int(os.environ.get("BUILDER_V3_CSRF_TOKEN_TTL", "900"))
        now = int(time.time())
        if abs(now - ts) > window:
            return False
        expected = hmac.new(str(secret).encode("utf-8"), str(ts).encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, sig)
    except Exception:
        logging.exception("Error verifying CSRF token")
        return False


@router.get("/api/builder_v3/csrf_token")
async def csrf_token(request: Request):
    """Issue a short-lived CSRF token signed by server secret (if configured).
    Returns JSON: {ok: True, token: "...", expires_in: seconds}
    """
    app_mod = sys.modules.get("gui_launcher.app")
    app_obj = getattr(app_mod, "app", None)
    secret = getattr(getattr(app_obj, "state", None), "builder_v3_csrf_secret", None)
    if not secret:
        # No secret configured
        return JSONResponse({"ok": False, "message": "CSRF rotation not configured"})
    ts = int(time.time())
    sig = hmac.new(str(secret).encode("utf-8"), str(ts).encode("utf-8"), hashlib.sha256).hexdigest()
    token = f"{ts}.{sig}"
    ttl = int(os.environ.get("BUILDER_V3_CSRF_TOKEN_TTL", "900"))
    return JSONResponse({"ok": True, "token": token, "expires_in": ttl})


@router.get("/api/builder_v3/metadata")
async def metadata(symbol: str | None = None) -> JSONResponse:
    try:
        # If a symbol is provided, resolve symbol-specific defaults for better UI hints.
        if symbol:
            try:
                adv = resolve_advanced_builder_defaults(symbol).as_dict().get("defaults") or {}
            except Exception:
                adv = _load_advanced_builder_defaults() or {}
        else:
            adv = _load_advanced_builder_defaults() or {}
    except Exception:
        adv = {}

    try:
        profiles = _load_market_profiles() or {}
    except Exception:
        profiles = {}

    # Extract presets from advanced defaults if present
    tp_ladders = {}
    partial_ladders = {}
    try:
        g = adv.get("global") if isinstance(adv, dict) else None
        tm = (g.get("trade_management") or {}) if isinstance(g, dict) else {}
        tp_ladders = tm.get("tp_ladders") or {}
        partial_ladders = tm.get("partial_ladders") or {}
    except Exception:
        tp_ladders = {}
        partial_ladders = {}

    # Build simple instruments list from market profiles if available
    instruments = []
    try:
        markets = profiles.get("markets") if isinstance(profiles, dict) else {}
        if isinstance(markets, dict):
            for k, v in markets.items():
                if not isinstance(v, dict):
                    continue
                instruments.append(
                    {
                        "symbol": str(v.get("symbol") or k),
                        "asset_class": str(v.get("asset_class") or ""),
                        "instrument_group": v.get("instrument_group"),
                        "tick_size": v.get("tick_size"),
                        "pip_value": v.get("pip_value"),
                    }
                )
    except Exception:
        instruments = []

    return JSONResponse({"ok": True, "data": {"advanced_defaults": adv, "tp_ladders": tp_ladders, "partial_ladders": partial_ladders, "instruments": instruments}})


@router.post("/api/builder_v3/preview")
async def preview(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"ok": False, "message": "Invalid JSON"})

    # Rate limiting (optional)
    try:
        app_mod = sys.modules.get("gui_launcher.app")
        app_obj = getattr(app_mod, "app", None)
        _apply_rate_limit(request, "preview", getattr(getattr(app_obj, "state", None), "builder_v3_rate_limit", 60), getattr(getattr(app_obj, "state", None), "builder_v3_rate_window", 60))
    except HTTPException:
        raise
    except Exception:
        pass

    # Normalize similar to the v2 preview endpoint and delegate to the
    # existing server-side preview helper when available. We access the
    # helper via sys.modules to avoid circular imports (app imports this router).
    logger = logging.getLogger(__name__)
    logger.info("/api/builder_v3/preview called")
    if not isinstance(payload, dict):
        payload = {}
    logger.debug("preview payload: %s", payload)

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

    # Try to locate the preview helper in the running app module and call it.
    app_mod = sys.modules.get("gui_launcher.app")
    if app_mod and hasattr(app_mod, "_builder_v2_preview"):
        try:
            preview_resp = getattr(app_mod, "_builder_v2_preview")(base)
            logger.info("preview generated successfully via v2 helper")
            return JSONResponse(preview_resp)
        except Exception as e:
            logger.exception("preview generation failed in v2 helper; falling back to builtin generator: %s", e)

    # Fallback: generate a simple textual preview from rules
    def _rule_to_label(r):
        try:
            t = r.get("type") or r.get("rule_type") or "rule"
            if t == "rsi_threshold":
                return f"RSI length={r.get('length')} bull<={r.get('bull_level')} bear>={r.get('bear_level')}"
            if t == "ma_cross":
                return f"MA cross fast={r.get('fast')} slow={r.get('slow')}"
            return json.dumps(r)
        except Exception:
            return str(r)

    def _build_side(side_enabled: bool):
        if not side_enabled:
            return []
        rows = []
        for i, cr in enumerate(base.get("context_rules", []) or []):
            rows.append({"label": f"Context {i}", "expr": _rule_to_label(cr)})
        for i, sr in enumerate(base.get("signal_rules", []) or []):
            rows.append({"label": f"Signal {i}", "expr": _rule_to_label(sr)})
        for i, tr in enumerate(base.get("trigger_rules", []) or []):
            rows.append({"label": f"Trigger {i}", "expr": _rule_to_label(tr)})
        return rows

    preview = {
        "ok": True,
        "data": {
            "entry_tf": base.get("entry_tf"),
            "context_tf": base.get("context_tf"),
            "signal_tf": base.get("signal_tf"),
            "trigger_tf": base.get("trigger_tf"),
            "long": _build_side(bool(base.get("long_enabled"))),
            "short": _build_side(bool(base.get("short_enabled"))),
        },
    }
    return JSONResponse(preview)


@router.post("/api/builder_v3/context_visual")
async def context_visual(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"ok": False, "message": "Invalid JSON"})
    try:
        app_mod = sys.modules.get("gui_launcher.app")
        app_obj = getattr(app_mod, "app", None)
        _apply_rate_limit(request, "context_visual", getattr(getattr(app_obj, "state", None), "builder_v3_rate_limit", 60), getattr(getattr(app_obj, "state", None), "builder_v3_rate_window", 60))
    except HTTPException:
        raise
    except Exception:
        pass
    # Delegate to existing Builder v2 context visual endpoint when available to
    # reuse plotting logic and keep visuals consistent.
    app_mod = sys.modules.get("gui_launcher.app")
    if app_mod and hasattr(app_mod, "api_guided_builder_v2_context_visual"):
        try:
            class _Req:
                def __init__(self, payload):
                    self._payload = payload

                async def json(self):
                    return self._payload

            return await getattr(app_mod, "api_guided_builder_v2_context_visual")(_Req(payload))
        except Exception:
            # Fall through to stub on failure
            pass

    fig = {"data": [{"x": [], "y": [], "type": "scatter"}], "layout": {"title": "Context preview"}}
    # simple dummy time-series: small synthetic series for client preview
    try:
        # If payload indicates a max_bars hint, use it for length
        mb = int(payload.get("max_bars") or 50)
        fig["data"][0]["x"] = list(range(mb))
        fig["data"][0]["y"] = [float((i % 10) + (i * 0.01)) for i in range(mb)]
    except Exception:
        pass
    return JSONResponse({"ok": True, "data": {"message": "Context preview", "fig": fig}})


@router.post("/api/builder_v3/setup_visual")
async def setup_visual(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"ok": False, "message": "Invalid JSON"})
    try:
        app_mod = sys.modules.get("gui_launcher.app")
        app_obj = getattr(app_mod, "app", None)
        _apply_rate_limit(request, "setup_visual", getattr(getattr(app_obj, "state", None), "builder_v3_rate_limit", 60), getattr(getattr(app_obj, "state", None), "builder_v3_rate_window", 60))
    except HTTPException:
        raise
    except Exception:
        pass
    # Delegate to existing Builder v2 setup visual endpoint when available.
    app_mod = sys.modules.get("gui_launcher.app")
    if app_mod and hasattr(app_mod, "api_guided_builder_v2_setup_visual"):
        try:
            class _Req:
                def __init__(self, payload):
                    self._payload = payload

                async def json(self):
                    return self._payload

            return await getattr(app_mod, "api_guided_builder_v2_setup_visual")(_Req(payload))
        except Exception:
            # Fall through to stub on failure
            pass

    fig = {"data": [{"x": [], "y": [], "type": "scatter"}], "layout": {"title": "Setup preview"}}
    try:
        mb = int(payload.get("max_bars") or 100)
        fig["data"][0]["x"] = list(range(mb))
        fig["data"][0]["y"] = [float((i % 5) + (i * 0.02)) for i in range(mb)]
    except Exception:
        pass
    return JSONResponse({"ok": True, "data": {"message": "Setup preview", "fig": fig}})


@router.post("/api/builder_v3/validate")
async def validate(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"ok": False, "message": "Invalid JSON"})
    logger = logging.getLogger(__name__)

    # Rate limiting (optional)
    try:
        app_mod = sys.modules.get("gui_launcher.app")
        app_obj = getattr(app_mod, "app", None)
        _apply_rate_limit(request, "validate", getattr(getattr(app_obj, "state", None), "builder_v3_rate_limit", 60), getattr(getattr(app_obj, "state", None), "builder_v3_rate_window", 60))
    except HTTPException:
        raise
    except Exception:
        pass

    # Load the canonical v3 payload schema from docs (if present)
    try:
        schema_path = Path(__file__).resolve().parents[1] / "docs" / "dev" / "v3" / "PAYLOAD_SCHEMA.json"
        with schema_path.open("r", encoding="utf-8") as fh:
            schema = json.load(fh)
    except Exception:
        logger.exception("Failed to load v3 payload schema; proceeding without schema")
        schema = None

    # Prefer authoritative validation via the `jsonschema` library when available
    try:
        import jsonschema
        from jsonschema import Draft7Validator

        if schema is None:
            # If schema failed to load earlier, return OK and avoid false negatives
            return JSONResponse({"ok": True, "valid": True, "errors": []})

        # Create a Draft7Validator which supports $ref, oneOf, allOf, etc.
        try:
            validator = Draft7Validator(schema)
        except Exception:
            # If schema is invalid for Draft7, fall back to simple success
            logger.exception("Schema is invalid for Draft7Validator; skipping jsonschema validation")
            return JSONResponse({"ok": True, "valid": True, "errors": []})

        errors = []
        for e in validator.iter_errors(payload):
            # Build dotted path from the error absolute_path
            try:
                path = '.'.join([str(p) for p in e.absolute_path])
            except Exception:
                path = ''
            # Prefer the validator message, but include the validator name for clarity
            msg = e.message if hasattr(e, 'message') else str(e)
            errors.append({"path": path, "message": msg})

        if errors:
            return JSONResponse(status_code=422, content={"ok": False, "valid": False, "errors": errors})

        return JSONResponse({"ok": True, "valid": True, "errors": []})
    except ImportError:
        logger.warning("jsonschema not installed; falling back to builtin validator")
        # Fall back to the simple validator implemented earlier

        def _type_of(val):
            if val is None:
                return "null"
            if isinstance(val, list):
                return "array"
            if isinstance(val, bool):
                return "boolean"
            if isinstance(val, int) and not isinstance(val, bool):
                return "integer"
            if isinstance(val, (int, float)):
                return "number"
            if isinstance(val, str):
                return "string"
            if isinstance(val, dict):
                return "object"
            return type(val).__name__

        def _match_type(expected, val):
            actual = _type_of(val)
            if isinstance(expected, list):
                return actual in expected
            if expected == "integer":
                return actual == "integer"
            if expected == "number":
                return actual in ("number", "integer")
            return actual == expected

        errors = []

        def _push(path, msg):
            errors.append({"path": path or "", "message": msg})

        def _validate_node(node_schema, value, path):
            if not node_schema:
                return
            t = node_schema.get("type")
            if t is not None and value is not None:
                if not _match_type(t, value):
                    tdesc = "|".join(t) if isinstance(t, list) else t
                    _push(path, f"Expected type {tdesc}, got {_type_of(value)}")
                    return

            # string checks
            if ((t == "string") or (isinstance(t, list) and "string" in t)) and isinstance(value, str):
                if node_schema.get("minLength") is not None and len(value) < node_schema["minLength"]:
                    _push(path, f"String too short (len {len(value)} < {node_schema['minLength']})")
                if node_schema.get("maxLength") is not None and len(value) > node_schema["maxLength"]:
                    _push(path, f"String too long (len {len(value)} > {node_schema['maxLength']})")
                if node_schema.get("pattern"):
                    try:
                        import re

                        if not re.search(node_schema["pattern"], value):
                            _push(path, f"Value does not match pattern {node_schema['pattern']}")
                    except Exception:
                        pass

            # numeric bounds
            if ((t == "number" or t == "integer") or (isinstance(t, list) and ("number" in t or "integer" in t))) and isinstance(value, (int, float)):
                if node_schema.get("minimum") is not None and value < node_schema["minimum"]:
                    _push(path, f"Value {value} < minimum {node_schema['minimum']}")
                if node_schema.get("maximum") is not None and value > node_schema["maximum"]:
                    _push(path, f"Value {value} > maximum {node_schema['maximum']}")

            # arrays
            if ((t == "array") or (isinstance(t, list) and "array" in t)) and isinstance(value, list):
                if node_schema.get("minItems") is not None and len(value) < node_schema["minItems"]:
                    _push(path, f"Array has {len(value)} items < minItems {node_schema['minItems']}")
                if node_schema.get("maxItems") is not None and len(value) > node_schema["maxItems"]:
                    _push(path, f"Array has {len(value)} items > maxItems {node_schema['maxItems']}")
                items_schema = node_schema.get("items")
                if items_schema:
                    for i, item in enumerate(value):
                        _validate_node(items_schema, item, f"{path}.{i}" if path else str(i))

            # objects
            if ((t == "object") or (t is None and isinstance(value, dict))) and isinstance(value, dict):
                props = node_schema.get("properties", {})
                req = node_schema.get("required", []) if isinstance(node_schema.get("required"), list) else []
                for r in req:
                    if r not in value:
                        _push(f"{path}.{r}" if path else r, "Missing required property")
                for k, v in value.items():
                    sub_schema = props.get(k)
                    if sub_schema:
                        _validate_node(sub_schema, v, f"{path}.{k}" if path else k)
                    else:
                        if node_schema.get("additionalProperties") is False:
                            _push(f"{path}.{k}" if path else k, "Additional property not allowed")

        if schema:
            try:
                _validate_node(schema, payload, "")
            except Exception:
                logger.exception("Error while validating payload (fallback)")

        if errors:
            return JSONResponse(status_code=422, content={"ok": False, "valid": False, "errors": errors})

        return JSONResponse({"ok": True, "valid": True, "errors": []})


@router.post("/api/builder_v3/save")
async def save(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"ok": False, "message": "Invalid JSON"})

    # CSRF enforcement: if the app sets a `builder_v3_csrf` token on app.state,
    # require the incoming request to include a matching token via header or body.
    try:
        # Rate limiting (optional)
        app_mod = sys.modules.get("gui_launcher.app")
        app_obj = getattr(app_mod, "app", None)
        _apply_rate_limit(request, "save", getattr(getattr(app_obj, "state", None), "builder_v3_rate_limit", 60), getattr(getattr(app_obj, "state", None), "builder_v3_rate_window", 60))
    except HTTPException:
        raise
    except Exception:
        pass

    # CSRF enforcement: support legacy static token (`app.state.builder_v3_csrf`) or
    # new rotating HMAC tokens using `app.state.builder_v3_csrf_secret`. Enforcement
    # is controlled by `app.state.builder_v3_csrf_enforce` (default False).
    try:
        app_mod = sys.modules.get("gui_launcher.app")
        app_obj = getattr(app_mod, "app", None)
        # Enforce if explicit flag set, or legacy static token exists on app.state
        enforce = bool(getattr(getattr(app_obj, "state", None), "builder_v3_csrf_enforce", False)) or bool(getattr(getattr(app_obj, "state", None), "builder_v3_csrf", None))
        if enforce:
            hdr = request.headers.get("X-CSRF-Token")
            body_token = body.get("csrf_token") if isinstance(body, dict) else None
            token = hdr or body_token
            if not _verify_csrf_token(app_obj, token):
                raise HTTPException(status_code=403, detail={"ok": False, "message": "CSRF token missing or invalid"})
    except HTTPException:
        raise
    except Exception:
        logging.exception("Error checking CSRF token")
    # Basic server-side save flow:
    # 1. Validate payload using the same schema used by /validate
    # 2. If invalid, return 422 with errors
    # 3. Attempt to convert payload -> StrategySpec and persist canonical YAML
    # 4. If conversion or write fails, fall back to YAML write or JSON draft

    # Load schema path
    schema_path = Path(__file__).resolve().parents[1] / "docs" / "dev" / "v3" / "PAYLOAD_SCHEMA.json"
    schema = None
    try:
        with schema_path.open("r", encoding="utf-8") as fh:
            schema = json.load(fh)
    except Exception:
        schema = None

    # Validate using jsonschema if available
    errors = []
    try:
        import jsonschema
        from jsonschema import Draft7Validator

        if schema is not None:
            try:
                validator = Draft7Validator(schema)
                for e in validator.iter_errors(body):
                    path = '.'.join([str(p) for p in e.absolute_path])
                    errors.append({"path": path, "message": e.message})
            except Exception:
                # Fall back to no validation
                pass
    except Exception:
        # jsonschema missing; do a lightweight check for required top-level fields
        reqs = (schema.get("required") if isinstance(schema, dict) else []) or []
        for r in reqs:
            if r not in body:
                errors.append({"path": r, "message": "Missing required field"})

    if errors:
        return JSONResponse(status_code=422, content={"ok": False, "errors": errors})

    # Try to convert payload -> StrategySpec and write YAML spec via app helper
    spec_path = None
    spec_name = None
    try:
        from research.spec import StrategySpec, MarketSpecLite, SideRuleSpec
    except Exception:
        StrategySpec = None

    if StrategySpec is not None:
        try:
            market = {
                "symbol": body.get("symbol") or (body.get("market") and body.get("market").get("symbol")) or "UNKNOWN",
                "exchange": body.get("exchange") or (body.get("market") and body.get("market").get("exchange")),
                "market_type": body.get("market_type") or (body.get("market") and body.get("market").get("market_type")),
            }

            name = (body.get("name") or f"draft-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}").strip()
            long_spec = None
            short_spec = None
            if body.get("long_enabled"):
                long_spec = SideRuleSpec(enabled=True)
            if body.get("short_enabled"):
                short_spec = SideRuleSpec(enabled=True)

            spec = StrategySpec(
                name=name,
                description=str(body.get("description") or ""),
                market=MarketSpecLite(**market),
                entry_tf=str(body.get("entry_tf") or "1h"),
                context_tf=body.get("context_tf"),
                long=long_spec,
                short=short_spec,
                risk_per_trade_pct=float(body.get("risk_per_trade_pct") or 1.0),
                account_size=float(body.get("account_size") or 10000.0),
                max_trades_per_day=int(body.get("max_trades_per_day") or 1),
            )

            # Enhanced save semantics: honor `force` (overwrite), `rename_to`, and permission hooks.
            app_mod = sys.modules.get("gui_launcher.app")
            app_obj = getattr(app_mod, "app", None)
            force = bool(body.get("force")) if isinstance(body, dict) else False
            rename_to = (body.get("rename_to") or "").strip() if isinstance(body, dict) else ""

            target_name = rename_to or spec.name

            # Validate safe name
            try:
                if app_mod and hasattr(app_mod, "_is_safe_strategy_name"):
                    ok_name = app_mod._is_safe_strategy_name(target_name)
                else:
                    ok_name = bool(target_name and isinstance(target_name, str))
                if not ok_name:
                    raise ValueError("Invalid strategy name")
            except Exception:
                spec_path = None
                spec_name = None

            if spec_path is None:
                try:
                    import yaml
                    import re as _re

                    out_dir = Path(__file__).resolve().parents[1] / "research_specs"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    safe_name = _re.sub(r"[^A-Za-z0-9_\-]", "_", target_name)
                    out_file = out_dir / f"{safe_name}.yml"

                    # If file exists and it's not a rename of the same file, enforce overwrite semantics
                    exists = out_file.exists()
                    old_name = (body.get("old_name") or "").strip() if isinstance(body, dict) else ""
                    is_rename_of_same = False
                    try:
                        if old_name and app_mod and hasattr(app_mod, "_safe_spec_path"):
                            old_path = app_mod._safe_spec_path(old_name)
                            if old_path is not None and old_path.resolve() == out_file.resolve():
                                is_rename_of_same = True
                    except Exception:
                        is_rename_of_same = False

                    if exists and not is_rename_of_same:
                        # Not permitted without explicit force and permission
                        perm_checker = getattr(getattr(app_obj, "state", None), "builder_v3_permission_checker", None)
                        allow_overwrite_flag = bool(getattr(getattr(app_obj, "state", None), "builder_v3_allow_overwrite", False))
                        permitted = False
                        if force and allow_overwrite_flag:
                            permitted = True
                        elif callable(perm_checker):
                            try:
                                res = perm_checker(request, safe_name, "overwrite")
                                if hasattr(res, "__await__"):
                                    # awaitable
                                    res = await res
                                permitted = bool(res)
                            except Exception:
                                permitted = False

                        if not permitted:
                            # Suggest next available name. By default, preserve legacy
                            # behavior and fall back to writing a draft JSON instead of
                            # returning a hard conflict. If the app enables strict
                            # conflict mode, return 409.
                            try:
                                if app_mod and hasattr(app_mod, "_suggest_next_version_name"):
                                    sug = app_mod._suggest_next_version_name(Path(__file__).resolve().parents[1], target_name)
                                else:
                                    sug = target_name + "_v2"
                            except Exception:
                                sug = target_name + "_v2"
                            strict = bool(getattr(getattr(app_obj, "state", None), "builder_v3_strict_save_conflict", False))
                            if strict:
                                raise HTTPException(status_code=409, detail={"ok": False, "message": "Strategy name already exists", "suggested_name": sug})
                            # legacy fallback: do not write YAML; let the outer flow persist a JSON draft
                            spec_path = None
                            spec_name = None
                            # skip the overwrite/write block
                            raise Exception("name_conflict_fallback")

                        # permitted: create a timestamped backup and then atomically overwrite
                        try:
                            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                            bak = out_file.with_suffix(out_file.suffix + f".bak.{ts}")
                            try:
                                shutil.copy2(str(out_file), str(bak))
                            except Exception:
                                pass
                            # atomic write: write to tmp then replace
                            tmp = out_file.with_suffix(out_file.suffix + ".tmp")
                            if app_mod and hasattr(app_mod, "_spec_to_yaml_compact"):
                                yaml_text = app_mod._spec_to_yaml_compact(spec)
                            else:
                                try:
                                    import yaml as _yaml

                                    yaml_text = _yaml.safe_dump(spec.model_dump(exclude_none=True, exclude_defaults=True), sort_keys=False)
                                except Exception:
                                    yaml_text = str(spec)
                            tmp.write_text(yaml_text, encoding="utf-8")
                            os.replace(str(tmp), str(out_file))
                        except Exception:
                            raise
                    else:
                        # Normal create/rename: write if not exists
                        tmp = out_file.with_suffix(out_file.suffix + ".tmp")
                        if app_mod and hasattr(app_mod, "_spec_to_yaml_compact"):
                            yaml_text = app_mod._spec_to_yaml_compact(spec)
                        else:
                            try:
                                import yaml as _yaml

                                yaml_text = _yaml.safe_dump(spec.model_dump(exclude_none=True, exclude_defaults=True), sort_keys=False)
                            except Exception:
                                yaml_text = str(spec)
                        tmp.write_text(yaml_text, encoding="utf-8")
                        os.replace(str(tmp), str(out_file))

                    spec_path = str(out_file)
                    spec_name = target_name
                except HTTPException:
                    raise
                except Exception:
                    logging.exception("Failed to persist YAML spec; will fall back to draft JSON")
                    spec_path = None

        except HTTPException:
            # Re-raise HTTP errors (permission/conflict) so they reach the client
            raise
        except Exception:
            logging.exception("Failed to convert payload to StrategySpec; falling back to raw draft JSON")

    # If we have a spec_path, return that as the saved spec
    if spec_path:
        # For backward compatibility with existing tests and UI flows,
        # also write a JSON draft file (draft-*.json) so callers expecting
        # a draft file can find it. Return the draft_id and the canonical spec info.
        draft_id = f"draft-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        try:
            out_dir = Path(__file__).resolve().parents[1] / "research_specs"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{draft_id}.json"
            with out_file.open("w", encoding="utf-8") as fh:
                json.dump({"draft_id": draft_id, "created_at": datetime.utcnow().isoformat() + "Z", "payload": body}, fh, indent=2)
        except Exception:
            logging.exception("Failed to write draft file alongside spec")

        next_link = f"/create-strategy-guided/review?name={spec_name}"
        return JSONResponse({"ok": True, "data": {"draft_id": draft_id, "next": next_link, "spec_name": spec_name, "spec_path": spec_path}})

    # fallback: persist as JSON draft
    draft_id = f"draft-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    out_dir = Path(__file__).resolve().parents[1] / "research_specs"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{draft_id}.json"
        with out_file.open("w", encoding="utf-8") as fh:
            json.dump({"draft_id": draft_id, "created_at": datetime.utcnow().isoformat() + "Z", "payload": body}, fh, indent=2)
    except Exception as e:
        logging.exception("Failed to write draft file: %s", e)
        raise HTTPException(status_code=500, detail={"ok": False, "message": "Failed to persist draft"})

    next_link = f"/create-strategy-guided/review?draft_id={draft_id}"
    return JSONResponse({"ok": True, "data": {"draft_id": draft_id, "next": next_link}})
    # Try to convert payload -> StrategySpec and write YAML spec via app helper
    try:
        from research.spec import StrategySpec, MarketSpecLite, SideRuleSpec
    except Exception:
        StrategySpec = None

    spec_path = None
    spec_name = None
    if StrategySpec is not None:
        try:
            # Build minimal market spec
            market = {
                "symbol": body.get("symbol") or (body.get("market") and body.get("market").get("symbol")) or "UNKNOWN",
                "exchange": body.get("exchange") or (body.get("market") and body.get("market").get("exchange")),
                "market_type": body.get("market_type") or (body.get("market") and body.get("market").get("market_type")),
            }

            name = (body.get("name") or f"draft-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}").strip()
            # Create minimal side specs
            long_spec = None
            short_spec = None
            if body.get("long_enabled"):
                long_spec = SideRuleSpec(enabled=True)
            if body.get("short_enabled"):
                short_spec = SideRuleSpec(enabled=True)

            spec = StrategySpec(
                name=name,
                description=str(body.get("description") or ""),
                market=MarketSpecLite(**market),
                entry_tf=str(body.get("entry_tf") or "1h"),
                context_tf=body.get("context_tf"),
                long=long_spec,
                short=short_spec,
                risk_per_trade_pct=float(body.get("risk_per_trade_pct") or 1.0),
                account_size=float(body.get("account_size") or 10000.0),
                max_trades_per_day=int(body.get("max_trades_per_day") or 1),
            )

            # Attempt to use app helper to write validated spec
            app_mod = sys.modules.get("gui_launcher.app")
            if app_mod and hasattr(app_mod, "_write_validated_spec_to_repo"):
                try:
                    new_path = app_mod._write_validated_spec_to_repo(spec=spec, old_spec_name=(body.get("old_name") or ""))
                    spec_path = str(new_path)
                    spec_name = spec.name
                except Exception:
                    # fallback to writing YAML ourselves
                    spec_path = None

            if spec_path is None:
                # write YAML file under research_specs
                import yaml
                import re as _re

                out_dir = Path(__file__).resolve().parents[1] / "research_specs"
                out_dir.mkdir(parents=True, exist_ok=True)
                safe_name = _re.sub(r"[^A-Za-z0-9_\-]", "_", spec.name)
                out_file = out_dir / f"{safe_name}.yml"
                out_file.write_text(_spec_to_yaml_compact(spec), encoding="utf-8")
                spec_path = str(out_file)
                spec_name = spec.name

        except Exception:
            logging.exception("Failed to convert payload to StrategySpec; falling back to raw draft JSON")

    # If we have a spec_path, return that as the saved spec; otherwise fall back to JSON draft
    if spec_path:
        next_link = f"/create-strategy-guided/review?name={spec_name}"
        return JSONResponse({"ok": True, "data": {"draft_id": spec_name, "next": next_link, "spec_path": spec_path}})

    # fallback: persist as JSON draft (existing behavior)
    draft_id = f"draft-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    out_dir = Path(__file__).resolve().parents[1] / "research_specs"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{draft_id}.json"
        with out_file.open("w", encoding="utf-8") as fh:
            json.dump({"draft_id": draft_id, "created_at": datetime.utcnow().isoformat() + "Z", "payload": body}, fh, indent=2)
    except Exception as e:
        logging.exception("Failed to write draft file: %s", e)
        raise HTTPException(status_code=500, detail={"ok": False, "message": "Failed to persist draft"})

    next_link = f"/create-strategy-guided/review?draft_id={draft_id}"
    return JSONResponse({"ok": True, "data": {"draft_id": draft_id, "next": next_link}})
