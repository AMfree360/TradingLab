const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

function read(p) { return fs.readFileSync(path.resolve(p), 'utf8'); }

async function testType(type) {
  const htmlPath = 'gui_launcher/static/test/guided_builder_harness.html';
  const jsPath = 'gui_launcher/static/js/guided_builder_v2.js';
  let html = read(htmlPath);
  // Ensure primary_context_type exists in the harness HTML so the v2 script
  // can find it during init and wire its change handler.
  if (!/id="primary_context_type"/.test(html)) {
    const selectHtml = `\n  <label>Primary context</label>\n  <select id="primary_context_type" name="primary_context_type">\n    <option value="ma_stack">ma_stack</option>\n    <option value="price_vs_ma">price_vs_ma</option>\n    <option value="ma_cross_state">ma_cross_state</option>\n    <option value="atr_pct">atr_pct</option>\n    <option value="structure_breakout_state">structure_breakout_state</option>\n    <option value="ma_spread_pct">ma_spread_pct</option>\n    <option value="custom">custom</option>\n  </select>\n`;
    html = html.replace(/(<form[^>]*>)/i, `$1\n${selectHtml}`);
  }
  html = html.replace(/<script\s+src=["']\/static\/js\/guided_builder_v2\.js["']>\s*<\/script>/i, '');
  const js = read(jsPath);
  const modelJs = read('gui_launcher/static/js/guided_builder_v2_model.js');
  const stateJs = read('gui_launcher/static/js/guided_builder_v2_state.js');
  const domJs = read('gui_launcher/static/js/guided_builder_v2_dom.js');

  const dom = new JSDOM(html, { runScripts: 'dangerously', resources: 'usable', url: 'http://localhost' });
  const { window } = dom;
  // Inject state manager first, then DOM binder, then main renderer
  const sm = window.document.createElement('script'); sm.textContent = modelJs; window.document.head.appendChild(sm);
  const s1 = window.document.createElement('script'); s1.textContent = stateJs; window.document.head.appendChild(s1);
  const s2 = window.document.createElement('script'); s2.textContent = domJs; window.document.head.appendChild(s2);
  const s3 = window.document.createElement('script'); s3.textContent = js; window.document.head.appendChild(s3);

  await new Promise((res) => setTimeout(res, 200));

  // ensure primary select
  let prim = window.document.getElementById('primary_context_type');
  if (!prim) {
    prim = window.document.createElement('select');
    prim.id = 'primary_context_type';
    prim.name = 'primary_context_type';
    ['ma_stack','price_vs_ma','ma_cross_state','atr_pct','structure_breakout_state','ma_spread_pct','custom'].forEach(v => {
      const o = window.document.createElement('option'); o.value = v; o.textContent = v; prim.appendChild(o);
    });
    const form = window.document.querySelector('form');
    form.insertBefore(prim, form.firstChild);
  }

  // set primary first, then add a context row so the new row picks up the default
  prim.value = type;
  prim.dispatchEvent(new window.Event('change', { bubbles: true }));
  await new Promise((res) => setTimeout(res, 200));

  const add = window.document.getElementById('add-context');
  add.click();
  await new Promise((res) => setTimeout(res, 200));

  const hidden = window.document.querySelector('#context_rules_json');
  const val = hidden ? hidden.value : null;
  let parsed = null;
  try { parsed = JSON.parse(val || '[]'); } catch (e) { parsed = null; }

  return { type, hidden: val, parsed };
}

(async () => {
  const types = ['ma_stack','price_vs_ma','ma_cross_state','atr_pct','structure_breakout_state','ma_spread_pct','custom'];
  for (const t of types) {
    const out = await testType(t);
    console.log('TYPE:', t);
    console.log('  hidden:', out.hidden);
    console.log('  parsed first:', (out.parsed && out.parsed[0]) || null);
  }
})();
