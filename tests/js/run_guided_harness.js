const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

function read(p) { return fs.readFileSync(path.resolve(p), 'utf8'); }

async function run() {
  const htmlPath = 'gui_launcher/static/test/guided_builder_harness.html';
  // load the full v2 script so we can test its initialization; it is wrapped
  // with a try/catch fallback by our patch if it throws.
  const jsPath = 'gui_launcher/static/js/guided_builder_v2.js';

  let html = read(htmlPath);
  // remove script src to prevent jsdom attempting to fetch external files
  html = html.replace(/<script\s+src=["']\/static\/js\/guided_builder_v2\.js["']>\s*<\/script>/i, '');
  const js = read(jsPath);

  const dom = new JSDOM(html, { runScripts: 'dangerously', resources: 'usable', url: 'http://localhost' });
  const { window } = dom;

  // capture console
  const logs = [];
  ['log','warn','error','info'].forEach((k) => {
    const orig = window.console[k].bind(window.console);
    window.console[k] = function(...args) { logs.push({ level: k, args }); orig(...args); };
  });

  // inject the script so it runs in page context
  const scriptEl = window.document.createElement('script');
  scriptEl.textContent = js;
  window.document.head.appendChild(scriptEl);

  // wait a short time for DOMContent and async timers
  await new Promise((res) => setTimeout(res, 500));

  // Helpers
  const dump = (label) => {
    console.log('---', label, '---');
    for (const l of logs) console.log(l.level.toUpperCase(), ...l.args);
    console.log('DOM snapshot: context-rules children:', window.document.getElementById('context-rules').children.length);
    console.log('DOM snapshot: signal-rules children:', window.document.getElementById('signal-rules').children.length);
    console.log('DOM snapshot: trigger-rules children:', window.document.getElementById('trigger-rules').children.length);
  };

  // simulate clicks
  const click = (id) => {
    const el = window.document.getElementById(id);
    if (!el) { console.error('missing element', id); return; }
    el.click();
  };

  try {
    dump('Initial load');

    click('add-context');
    await new Promise((res) => setTimeout(res, 200));
    dump('After add-context');

    click('add-signal');
    await new Promise((res) => setTimeout(res, 200));
    dump('After add-signal');

    click('add-trigger');
    await new Promise((res) => setTimeout(res, 200));
    dump('After add-trigger');

  } catch (e) {
    console.error('Test runner error', e);
  }
}

run().then(()=>process.exit(0)).catch((e)=>{ console.error(e); process.exit(2); });
