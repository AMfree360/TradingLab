# TradingLab Builder V3 — Implementation & Architecture Spec

Status: Draft

Purpose
- Define a single-page Builder V3 that reproduces and improves Builder V2 functionality without reusing any V2 frontend code.
- Provide frontend and backend implementers explicit specs, API contracts, payload schemas and a minimal scaffold to start development.

High-level goals
- One-page UI consolidating previous v2 steps (market, entries, context, signals, triggers, stops, trade management, calendar, execution, preview, review/save).
- No reuse of any `guided_builder_v2.js` or v2 JS code; new client implementation only.
- New backend API surface under `/api/builder_v3/*` to support preview, visuals, validation and save.
- Maintain server-side validation and advanced defaults (`config/advanced_builder_defaults.yml`) but expose them through a `metadata` endpoint for client bootstrapping.

Contents
1. Component architecture (frontend)
2. API contracts (backend)
3. Payload & JSON Schema
4. UI behavior, validation and UX rules
5. File map & scaffolding provided
6. Implementation steps & integration instructions

---

## 1. Frontend component architecture
Split the single-page UI into focused, reusable components. Avoid framework lock-in but favour a modular approach (vanilla JS modules or a component framework like React/Vue if team prefers).

Top-level components
- App (root orchestrator, global state store)
- HeaderMeta (strategy name, template, draft actions)
- MarketSelector (asset class, exchange, symbol/custom)
- SidesToggle (LONG/SHORT toggles + align_with_context)
- TimeframeSelector (entry/context/signal/trigger TFs)
- RiskSizing (risk per trade, sizing, account size, max trades/day, max daily loss)
- ContextBuilder (primary context + context rule rows)
- SignalsBuilder (signal rule rows)
- TriggersBuilder (trigger rule rows)
- EntryConditions (per-side two-condition UI)
- StopsConfig (shared/separate stop configs)
- TradeManagement (TP ladders, partials, trailing)
- CalendarSessions (allowed days + named sessions)
- ExecutionOptions (exit-if-context-invalid)
- PreviewPanel (textual effective rules + context/setup Plotly charts)
- ValidationEngine (shared client validation set)
- ApiClient (HTTP client wrapper for v3 endpoints)
- StateStore (single source of truth; local persistence optional)

Each rule row (context/signal/trigger) is a small subcomponent with its own local inputs and emits change events to the StateStore.

Design principles
- Stateless UI components where possible; central `StateStore` holds canonical form model.
- Components communicate via small events or callbacks to minimize coupling.
- All numeric values use native numbers in the state (not strings).
- Provide `serializeForSubmit()` in `StateStore` to produce the canonical payload.

---

## 2. API contracts (summary)
All APIs live under `/api/builder_v3/*`. New APIs (no v2 reuse):

- GET `/api/builder_v3/metadata`
  - Returns `advanced_defaults`, `tp_ladders`, `partial_ladders`, `instruments` and market hints.

- POST `/api/builder_v3/preview`
  - Input: full serialized payload
  - Output: textual preview object: `long` and `short` sections with `context | signals | triggers` arrays containing `{ label, expr }` and metadata (TFs).

- POST `/api/builder_v3/context_visual`
  - Input: payload (symbol, entry_tf, context_tf, context_rules, optional max_bars)
  - Output: Plotly-compatible `fig` JSON ({ data: [...], layout: {...} })

- POST `/api/builder_v3/setup_visual`
  - Input: full payload
  - Output: Plotly-compatible `fig` JSON showing entries/stops/preview overlays.

- POST `/api/builder_v3/validate`
  - Input: full payload
  - Output: 200 if OK, 422 with `errors` map if validation fails.

- POST `/api/builder_v3/save`
  - Input: { action: 'save'|'skip'|'review', draft_id?, payload }
  - Output: success with `draft_id` and `next` link or 422 `errors`.

Common response envelope
```json
{ "ok": true|false, "message": "optional", "data": {...}, "errors": { "field.path": "message" } }
```

Security
- Require CSRF on mutating endpoints: accept `X-CSRF-Token` header or `csrf_token` in body.
- Rate-limit preview/visual endpoints.

CSRF notes for frontend implementers
- The server template `gui_launcher/templates/builder_v3.html` renders a hidden input with the CSRF token when available:

  ```html
  <input type="hidden" id="csrf_token" value="{{ csrf_token() }}" />
  ```

- Frontend code should read this token and include it on mutating requests (save/validate). Example (already implemented in `ApiClient`):

  ```javascript
  const csrfEl = document.getElementById('csrf_token');
  const csrf = csrfEl ? csrfEl.value : null;
  // ApiClient automatically sets X-CSRF-Token header when provided
  const api = new ApiClient({ csrfToken: csrf });
  ```

- Server-side enforcement is optional and controlled by runtime configuration. In our backend the handler will check `app.state.builder_v3_csrf` (if set) and require a matching token in `X-CSRF-Token` or `csrf_token` in the request body. CI/dev environments can leave this unset for convenience; production should enable and rotate a secure token.

Save semantics: overwrite, rename, and permission hooks

- Runtime flags exposed via environment variables and `app.state`:
  - `BUILDER_V3_ALLOW_OVERWRITE` / `app.state.builder_v3_allow_overwrite` (default: false)
    - When true, clients may pass `"force": true` in the save payload to request an overwrite of an existing spec.
  - `BUILDER_V3_STRICT_SAVE_CONFLICT` / `app.state.builder_v3_strict_save_conflict` (default: false)
    - When true, name conflicts produce a 409 HTTP response. When false (legacy default), name conflicts fall back to persisting a JSON draft and returning a 200.

- Permission-checker hook (`app.state.builder_v3_permission_checker`):
  - A server may provide a callable to decide per-save permissions. The callable will be invoked as `res = checker(request, spec_name, op)` where `op` is one of `"create"`, `"overwrite"`, or `"rename"`.
  - The checker may be synchronous (return `bool`) or async (`await`able). If it returns truthy, the operation is permitted; otherwise it is denied.
  - If a permission checker is configured, it takes precedence over the simple `BUILDER_V3_ALLOW_OVERWRITE` flag for overwrite decisions.

Example (attach at app startup):

```py
from gui_launcher import app as launcher_app
from gui_launcher.builder_v3_permission_examples import example_sync_checker

# Attach the example checker (replace with your real authz check)
launcher_app.state.builder_v3_permission_checker = example_sync_checker
```

Notes:
- The router performs an atomic write when persisting YAML (write-to-temp + os.replace) to avoid partial writes.
- On overwrite, the previous file is copied to a timestamped `.bak.<ts>` file as a best-effort backup.
- The endpoint accepts `rename_to` in the payload to rename the canonical spec; permission checks are invoked with `op == "rename"`.

Notes for backend: the v3 endpoints can reuse server-side rule generation/preview logic but must expose the new routes and response envelopes. Keep v2 endpoints untouched until v3 is stable.

---

## 3. Payload & JSON Schema
Canonical payload (high-level fields):

```json
{
  "draft_id": "optional-string",
  "name": "strategy_name",
  "symbol": "BTCUSDT",
  "asset_class": "crypto",
  "exchange": "binance",
  "market_type": "spot",
  "long_enabled": true,
  "short_enabled": true,
  "align_with_context": true,
  "entry_tf": "1h",
  "context_tf": "4h",
  "signal_tf": "15m",
  "trigger_tf": "1h",
  "risk_per_trade_pct": 1.0,
  "sizing_mode": "account_size",
  "account_size": 10000,
  "max_trades_per_day": 1,
  "max_daily_loss_pct": 3.0,
  "context_rules": [ ... ],
  "signal_rules": [ ... ],
  "trigger_rules": [ ... ],
  "stops": { ... },
  "trade_management": { ... },
  "calendar": { ... },
  "execution": { ... }
}
```

A machine-readable JSON Schema is provided at `docs/dev/v3/PAYLOAD_SCHEMA.json` (see file in repository). Frontend should use it to validate before making preview calls.

---

## 4. UI behavior & validation rules
- Debounce preview/setup_visual calls (150-450ms). Expose a optional `force` flag for manual refresh.
- Seed primary context to first context rule for non-MA-stack selections.
- Per-rule TF: if rule omits `tf`, server uses global TF for that category.
- Provide "Copy master defaults" buttons for TP/partials/trailing/calendar, pulling defaults from `/api/builder_v3/metadata`.
- Enforce TP exit% totals ≤ 100 on client before `save`.
- Use `advanced_defaults` for min/max ranges and label hints.

Validation priorities
1. Client-side: presence checks, numeric ranges, TP% totals, basic DSL formatting sanity (non-empty).
2. Server-side via `/api/builder_v3/validate`: full semantic validation and guardrail application.

---

## 5. File map & scaffolding provided
The repository now includes a minimal scaffold to begin V3 work:
- `docs/dev/v3/IMPLEMENTATION_SPEC.md` (this file)
- `docs/dev/v3/API_CONTRACTS.md` (detailed API examples)
- `docs/dev/v3/PAYLOAD_SCHEMA.json` (JSON Schema)
- `docs/dev/v3/FILES_NEEDED.md` (recommended file list for full implementation)
- `gui_launcher/templates/builder_v3.html` (UI template skeleton)
- `gui_launcher/static/js/builder_v3/main.js` (frontend skeleton - no v2 code)
- `gui_launcher/static/css/builder_v3.css` (minimal styles)
- `gui_launcher/builder_v3_routes.py` (Flask blueprint stubs for v3 endpoints)

Refer to `FILES_NEEDED.md` for the complete list recommended for a full implementation.

---

## 6. Implementation steps & integration instructions (recommended)
1. Backend: implement `GET /api/builder_v3/metadata` endpoint. Return `advanced_defaults` resolved per symbol.
2. Backend: implement `POST /api/builder_v3/preview` using builder logic to produce textual preview. Return `ok` envelope.
3. Backend: implement `context_visual` and `setup_visual` returning Plotly `fig` objects.
4. Frontend: implement modular `StateStore` + `ApiClient`. Wire UI components to state.
5. Frontend: implement `PreviewPanel` and hook up debounced calls to preview/visual endpoints, respecting CSRF token.
6. Frontend: implement client validation using `PAYLOAD_SCHEMA.json` and `ValidationEngine` for guardrails.
7. Add E2E tests for major flows: create draft, add rules, preview, save.

Phase 5 details (UX, accessibility & QA)
- Accessibility: add ARIA landmarks and live regions for preview/error updates; ensure all interactive elements have keyboard focusable semantics. Example: the root is a `<main>` and the error summary uses `role="region"` and `aria-live="polite"`.
- Keyboard navigation: provide keyboard shortcuts for frequent actions (preview, save, jump to first error). Document these in the developer README.
- Error grouping: server validation responses returned as dotted paths should be rendered in a single summary panel; clicking an error should focus the corresponding form field.
- Visual polish: ensure responsive layout, chart container min-heights, and adequate color contrast for error states.
- E2E tests: scaffold basic Cypress tests to cover the smoke flow (open page, trigger validate/save, check errors). These can be expanded to cover full save/validate/preview flows once CSRF and auth flows are finalized.

Tips
- Keep server payload numeric types as numbers.
- Validate on server even if client validates.

---

Contact points
- Frontend lead: define preferred UI framework (vanilla vs React/Vue).
- Backend lead: adapt existing builder logic to new `builder_v3` controllers.


End of spec.
