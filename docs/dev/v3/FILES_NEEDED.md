# Files recommended for full Builder V3 implementation

This list describes the files the frontend and backend teams should add/implement when building Builder V3. The repository already contains these scaffolds; implementers should expand them.

Frontend (suggested)
- `gui_launcher/templates/builder_v3.html` — server-rendered HTML template for Builder V3 page.
- `gui_launcher/static/css/builder_v3.css` — styles specific to V3.
- `gui_launcher/static/js/builder_v3/main.js` — entry script; implements `App`, `StateStore`, `ApiClient`, components.
- `gui_launcher/static/js/builder_v3/components/*.js` — per-component modules (ContextBuilder.js, SignalsBuilder.js, PreviewPanel.js, etc.)
- `gui_launcher/static/js/builder_v3/validation.js` — client validation engine using `docs/dev/v3/PAYLOAD_SCHEMA.json`.
- `gui_launcher/static/js/builder_v3/state_store.js` — single source of truth and local persistence.

Backend (suggested)
- `gui_launcher/builder_v3_routes.py` — Flask blueprint with endpoints:
  - `GET /api/builder_v3/metadata`
  - `POST /api/builder_v3/preview`
  - `POST /api/builder_v3/context_visual`
  - `POST /api/builder_v3/setup_visual`
  - `POST /api/builder_v3/validate`
  - `POST /api/builder_v3/save`
- Update `gui_launcher/app.py` to register the new blueprint (when ready).

Docs & tests
- `docs/dev/v3/IMPLEMENTATION_SPEC.md` (this file)
- `docs/dev/v3/API_CONTRACTS.md`
- `docs/dev/v3/PAYLOAD_SCHEMA.json`
- Frontend unit tests (e.g. Jest) for StateStore & ValidationEngine
- Integration/E2E tests (Cypress/Playwright) for the full UI flow

Notes
- Keep v2 files untouched while v3 is developed.
- The frontend may use the metadata endpoint to load `advanced_defaults` rather than reading config files directly.
