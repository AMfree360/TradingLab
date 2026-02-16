const { JSDOM } = require('jsdom');

(async () => {
  const dom = new JSDOM(`<!doctype html><html><body>
    <div id="context-visual"></div>
    <div id="context-visual-message"></div>
    <div id="setup-visual"></div>
    <div id="setup-visual-message"></div>
  </body></html>`, { runScripts: 'dangerously', resources: 'usable' });
  const { window } = dom;

  // mock Plotly
  window.Plotly = { react: (el, data, layout) => { window._plotly_calls = window._plotly_calls || []; window._plotly_calls.push({elId: el.id, data, layout}); return Promise.resolve(); } };

  // mock fetch to return ok fig objects
  window.fetch = (url, opts) => Promise.resolve({ ok: true, json: () => Promise.resolve({ ok: true, fig: { data: [{x:[1], y:[2]}], layout: { title: url } }, message: 'ok' }) });

  // mini implementations mirroring updateContextVisual / updateSetupVisual
  async function miniUpdateContextVisual(payload, opts) {
    const el = window.document.getElementById('context-visual');
    const msg = window.document.getElementById('context-visual-message');
    if (!el || !msg) return false;
    msg.textContent = 'Loading context preview…';
    try {
      const res = await window.fetch('/api/guided/builder_v2/context_visual', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const out = await res.json();
      if (!out || !out.ok) { msg.textContent = 'Context preview unavailable.'; return false; }
      msg.textContent = out.message || '';
      if (out.fig && out.fig.data && out.fig.layout) {
        await window.Plotly.react(el, out.fig.data, out.fig.layout, {});
        return true;
      }
    } catch (e) { msg.textContent = 'Context preview unavailable.'; }
    return false;
  }

  async function miniUpdateSetupVisual(payload, opts) {
    const el = window.document.getElementById('setup-visual');
    const msg = window.document.getElementById('setup-visual-message');
    if (!el || !msg) return false;
    msg.textContent = 'Loading setup preview…';
    try {
      const res = await window.fetch('/api/guided/builder_v2/setup_visual', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const out = await res.json();
      if (!out || !out.ok) { msg.textContent = 'Setup preview unavailable.'; return false; }
      msg.textContent = out.message || '';
      if (out.fig && out.fig.data && out.fig.layout) {
        await window.Plotly.react(el, out.fig.data, out.fig.layout, {});
        return true;
      }
    } catch (e) { msg.textContent = 'Setup preview unavailable.'; }
    return false;
  }

  const payload = { sample: true };
  const c = await miniUpdateContextVisual(payload, { force: true });
  const s = await miniUpdateSetupVisual(payload, { force: true });

  console.log('context ok?', c, 'setup ok?', s);
  console.log('Plotly calls:', window._plotly_calls && window._plotly_calls.length);
  if (window._plotly_calls && window._plotly_calls.length === 2) process.exit(0);
  else process.exit(2);
})();
