const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

function read(p) { return fs.readFileSync(path.resolve(p), 'utf8'); }

async function runTest() {
  const htmlPath = 'gui_launcher/static/test/guided_builder_harness.html';
  const jsPath = 'gui_launcher/static/js/guided_builder_v2.js';
  const modelJs = read('gui_launcher/static/js/guided_builder_v2_model.js');
  const stateJs = read('gui_launcher/static/js/guided_builder_v2_state.js');
  const domJs = read('gui_launcher/static/js/guided_builder_v2_dom.js');
  const html = read(htmlPath);

  // Remove external script tag to avoid jsdom attempting to fetch it
  const safeHtml = html.replace(/<script\s+src=["']\/static\/js\/guided_builder_v2\.js["']>\s*<\/script>/i, '');
  const dom = new JSDOM(safeHtml, { runScripts: 'dangerously', resources: 'usable', url: 'http://localhost' });
  const { window } = dom;

  // Ensure preview target elements exist (harness is minimal)
  const doc = window.document;
  // ensure form has expected action used by the preview IIFE
  const f = doc.querySelector('form#guided_step4_form');
  if (f) f.setAttribute('action', '/create-strategy-guided/step4');
  const ctxMsg = doc.createElement('div'); ctxMsg.id = 'context-visual-message'; doc.body.appendChild(ctxMsg);
  const ctxDiv = doc.createElement('div'); ctxDiv.id = 'context-visual'; ctxDiv.style.height = '200px'; doc.body.appendChild(ctxDiv);
  const setupMsg = doc.createElement('div'); setupMsg.id = 'setup-visual-message'; doc.body.appendChild(setupMsg);
  const setupDiv = doc.createElement('div'); setupDiv.id = 'setup-visual'; setupDiv.style.height = '200px'; doc.body.appendChild(setupDiv);

  // Capture global errors for easier diagnostics
  window.onerror = function(msg, url, line, col, err) { window.__last_err = (err && err.stack) || String(msg); };

  // Inject a simple mock Plotly before the renderer
  const plotlyScript = window.document.createElement('script');
  plotlyScript.textContent = `window.Plotly = { react: function(el, data, layout) { window._plotly_calls = window._plotly_calls || []; window._plotly_calls.push({elId: el && el.id, data, layout}); return Promise.resolve(); } };`;
  window.document.head.appendChild(plotlyScript);

  // Mock fetch to return expected preview JSON
  const fetchScript = window.document.createElement('script');
  fetchScript.textContent = `window.fetch = function(url, opts) {
    // return a simple fig for both endpoints
    const out = { ok: true, fig: { data: [{ x: [1,2,3], y: [1,2,3] }], layout: { title: url } }, message: 'ok', context_tf: '1h', entry_tf: '15m', signal_tf: '', trigger_tf: '' };
    return Promise.resolve({ ok: true, json: () => Promise.resolve(out) });
  };`;
  window.document.head.appendChild(fetchScript);

  // Inject model, state, dom binder, then main JS
  const sModel = window.document.createElement('script'); sModel.textContent = modelJs; window.document.head.appendChild(sModel);
  const sState = window.document.createElement('script'); sState.textContent = stateJs; window.document.head.appendChild(sState);
  const sDom = window.document.createElement('script'); sDom.textContent = domJs; window.document.head.appendChild(sDom);
  const sMain = window.document.createElement('script'); sMain.textContent = read(jsPath); window.document.head.appendChild(sMain);

  // wait for init
  await new Promise(res => setTimeout(res, 300));

  if (window.__last_err) console.error('Script error during init:', window.__last_err);

  // Trigger the form input/change handlers so the page's own `schedule()` ->
  // `updatePreview()` path runs (which calls updateContextVisual/updateSetupVisual).
  try {
    const form = window.document.querySelector('form#guided_step4_form');
    if (form) {
      // set entry TF to trigger a change
      const entryTf = form.querySelector('select[name="entry_tf"]');
      if (entryTf) { entryTf.value = '15m'; entryTf.dispatchEvent(new window.Event('input', { bubbles: true })); }
      // also dispatch a change event on the form
      form.dispatchEvent(new window.Event('change', { bubbles: true }));
    }
  } catch (e) { console.error('trigger error', e); }

  // wait for scheduled preview to run
  // also fire a pageshow to force preview refresh (some initializers listen for it)
  try { window.dispatchEvent(new window.Event('pageshow')); } catch (e) {}
  await new Promise(res => setTimeout(res, 800));

  const calls = window._plotly_calls || [];
  console.log('Plotly react calls:', calls.length);
  calls.forEach((c, i) => console.log(`#${i+1}`, c.elId, c.data && c.data.length, c.layout && c.layout.title));
  console.log('effective-preview-body:', (window.document.getElementById('effective-preview-body') || {}).innerHTML);
  console.log('context-visual-message:', (window.document.getElementById('context-visual-message') || {}).textContent);
  console.log('setup-visual-message:', (window.document.getElementById('setup-visual-message') || {}).textContent);

  if (calls.length >= 2) {
    console.log('SUCCESS: Both context and setup visuals invoked Plotly.react');
    process.exit(0);
  } else {
    console.error('FAIL: expected 2 Plotly.react calls');
    process.exit(2);
  }
}

runTest().catch((e) => { console.error('Test error', e); process.exit(2); });
