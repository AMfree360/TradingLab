# Builder V3 — API Contracts

This document lists the v3 endpoints, request/response examples, and error envelopes.

Common envelope
```json
{ "ok": true|false, "message": "optional", "data": { ... }, "errors": { "field.path": "message" } }
```

DEPRECATED: Builder V3 removed. API contracts archived.
- Purpose: bootstrap frontend with advanced defaults and templates.
- Success response example:
```json
{
  "ok": true,
  "data": {
    "advanced_defaults": { /* same shape as resolved config/advanced_builder_defaults.yml */ },
    "tp_ladders": { /* named ladders */ },
    "partial_ladders": { /* named partials */ },
    "instruments": [ { "symbol": "BTCUSDT", "asset_class": "crypto", "tick_size": 0.01, "pip_value": null }, ... ]
  }
}
```

2) POST /api/builder_v3/preview
- Input: full builder payload (see `PAYLOAD_SCHEMA.json`).
- Output: textual preview for LONG/SHORT.
- Sample response:
```json
{
  "ok": true,
  "data": {
    "entry_tf": "1h",
    "context_tf": "4h",
    "signal_tf": "15m",
    "trigger_tf": "1h",
    "aligned_dual": true,
    "long": { "enabled": true, "direction": "long", "context": [ { "label": "MA stack", "expr": "..." } ], "signals": [], "triggers": [] },
    "short": { /* same shape */ }
  }
}
```

3) POST /api/builder_v3/context_visual
- Input: subset payload (symbol, entry_tf, context_tf, context_rules)
- Output: Plotly `fig` JSON under `data.fig`.

4) POST /api/builder_v3/setup_visual
- Input: full payload
- Output: Plotly `fig` JSON under `data.fig`.

5) POST /api/builder_v3/validate
- Input: full payload
- Output: 200 OK if valid. If invalid, 422 with `errors` mapping.

6) POST /api/builder_v3/save
- Input: { action: 'save'|'skip'|'review', draft_id?, payload }
- Output: on success, `data: { draft_id, next }`.

Error handling
- 400: JSON parse / syntactic issues.
- 422: validation problems. Response example:
```json
HTTP/1.1 422 Unprocessable Entity
{ "ok": false, "message": "Validation failed", "errors": { "context_rules[0].fast": "Must be >= 1", "trade_management.tp_levels[0].exit_pct": "Sum > 100%" } }
```

Security
- Require CSRF token via `X-CSRF-Token` header or `csrf_token` in request body for mutating endpoints.

CSRF frontend usage example

Client code should read the token rendered by the server template and include it on mutating requests. Example:

```javascript
// read token from hidden input in `builder_v3.html`
const csrfEl = document.getElementById('csrf_token');
const csrf = csrfEl ? csrfEl.value : null;

// Include as header on mutating requests
fetch('/api/builder_v3/save', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    ...(csrf ? { 'X-CSRF-Token': csrf } : {})
  },
  body: JSON.stringify(payload)
});
```

Server-side behaviour: the v3 routes will accept either the `X-CSRF-Token` header or a `csrf_token` field in the JSON body. The app optionally enforces a configured token (`app.state.builder_v3_csrf`).

Notes
- Web client should handle absent or malformed `data` gracefully and show fallback messages in UI.
- Visual endpoints should accept optional `max_bars` to limit returned series.
