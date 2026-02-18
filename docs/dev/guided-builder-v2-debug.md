# Guided Builder v2 — Frontend Debugging

This document describes the small on-page diagnostic console used by Guided Builder v2 and how developers can enable it for troubleshooting.

Location
- Template: `gui_launcher/templates/guided_builder_v2.html`
- Client JS: `gui_launcher/static/js/*` (main logic in `guided_builder_v2.js` and related modules)

What it is
- A lightweight, human-readable debug pane that captures:
  - fetch requests and responses (URL + truncated body + status)
  - uncaught errors and unhandled promise rejections
  - helper presence / script reload attempts
- The debug pane is purely diagnostic UI — it does not change API behavior or the generated strategy spec.

Element
- DOM id: `guided-debug` (element selector: `#guided-debug`)

How to enable (devs)
- URL flag: append `?guided_debug=1` to the page URL and reload.
- LocalStorage: in the browser console run:

```javascript
localStorage.setItem('guided_debug', '1');
location.reload();
```

Behavior
- Console-first: messages are always written to the browser console with the prefix `[guided-debug]` and a timestamp.
- In-page box: shown only when debug is enabled; otherwise it is hidden to normal users.
- Fetch wrapper: logs `fetch -> <url> body=...` and `fetch resp -> <url> status=<code>` useful for diagnosing `/api/guided/builder_v2/*` endpoints and preview rendering.

Why keep it
- Developer convenience: quick visibility of failing preview calls and request payloads without opening devtools.
- Lightweight and safe: no server changes, no persisted secrets, and no effect on production behavior when hidden.

Notes & maintenance
- Removing the element from the template is safe for functionality, but you'll lose the in-page convenience. Prefer hiding by default instead.
- If you add e2e checks that depend on this pane, reference `#guided-debug` explicitly in tests.
- Keep the fetch wrapper and console logs intact — automated tools and debugging workflows expect console output.

Quick troubleshooting checklist
1. Enable debug via URL/localStorage.
2. Reproduce the issue on Step 4 (Context + Trigger).
3. Check console for `[guided-debug]` lines and network tab for failing requests.
4. If necessary, copy the payload from the console and reproduce server-side using `curl` or a small Python script.

That's it — short, actionable guidance for devs working on Guided Builder v2.
