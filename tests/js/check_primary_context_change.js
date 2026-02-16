const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

function read(p) { return fs.readFileSync(path.resolve(p), 'utf8'); }

async function run() {
  const htmlPath = 'gui_launcher/static/test/guided_builder_harness.html';
  const jsPath = 'gui_launcher/static/js/guided_builder_v2.js';

  let html = read(htmlPath);
  html = html.replace(/<script\s+src=["']\/static\/js\/guided_builder_v2\.js["']>\s*<\/script>/i, '');
  const js = read(jsPath);
  const modelJs = read('gui_launcher/static/js/guided_builder_v2_model.js');
  const stateJs = read('gui_launcher/static/js/guided_builder_v2_state.js');
  const domJs = read('gui_launcher/static/js/guided_builder_v2_dom.js');

  const dom = new JSDOM(html, { runScripts: 'dangerously', resources: 'usable', url: 'http://localhost' });
  const { window } = dom;

  // inject state manager + dom binder + renderer in order
  const sm = window.document.createElement('script'); sm.textContent = modelJs; window.document.head.appendChild(sm);
  const s1 = window.document.createElement('script');
  s1.textContent = stateJs;
  window.document.head.appendChild(s1);
  const s2 = window.document.createElement('script');
  s2.textContent = domJs;
  window.document.head.appendChild(s2);
  const scriptEl = window.document.createElement('script');
  scriptEl.textContent = js;
  window.document.head.appendChild(scriptEl);

  await new Promise((res) => setTimeout(res, 300));

  let prim = window.document.getElementById('primary_context_type');
  if (!prim) {
    // create minimal primary select if harness doesn't include it
    const s = window.document.createElement('select');
    s.id = 'primary_context_type';
    s.name = 'primary_context_type';
    ['ma_stack','price_vs_ma','ma_cross_state','atr_pct','structure_breakout_state','ma_spread_pct','custom'].forEach(v => {
      const o = window.document.createElement('option'); o.value = v; o.textContent = v; s.appendChild(o);
    });
    const form = window.document.querySelector('form');
    form.insertBefore(s, form.firstChild);
    prim = s;
  }

  // ensure first row exists
  const add = window.document.getElementById('add-context');
  add.click();
  await new Promise((res) => setTimeout(res, 200));

  // change primary context to price_vs_ma
  prim.value = 'price_vs_ma';
  prim.dispatchEvent(new window.Event('change', { bubbles: true }));
  await new Promise((res) => setTimeout(res, 200));

  const ctx = window.document.getElementById('context-rules');
  console.log('context rows:', ctx.children.length);
    const first = ctx.querySelector('.rule-row');
    if (!first) { console.error('no first row'); return; }
    console.log('FIRST ROW HTML:\n', first.outerHTML);
    const maType = first.querySelector('[data-field="ma_type"]') || first.querySelector('[data-key="ma_type"]') || first.querySelector('[name="ma_type"]') || first.querySelector('.param[data-key="ma_type"]');
    const len = first.querySelector('[data-field="length"]') || first.querySelector('[data-key="length"]') || first.querySelector('[name="length"]') || first.querySelector('.param[data-key="length"]');
    console.log('ma_type element found:', !!maType);
    console.log('length element found:', !!len);
    console.log('ma_type value:', maType ? (maType.value || maType.textContent) : null);
    console.log('length value:', len ? (len.value || len.textContent) : null);
    const hidden = window.document.querySelector('#context_rules_json');
    console.log('context_rules_json hidden value:', hidden ? hidden.value : null);
}

run().then(()=>process.exit(0)).catch((e)=>{ console.error(e); process.exit(2); });