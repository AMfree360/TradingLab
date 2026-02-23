const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

(async () => {
  const root = path.resolve(__dirname, '..');
  const harnessPath = path.join(root, 'gui_launcher', 'static', 'test', 'guided_builder_harness.html');
  const builderPath = path.join(root, 'gui_launcher', 'static', 'js', 'guided_builder_v2.js');

  const harness = fs.readFileSync(harnessPath, 'utf8');
  const builder = fs.readFileSync(builderPath, 'utf8');

  // Replace external script tag with inline content
  const content = harness.replace(/<script\s+src=["']\/static\/js\/guided_builder_v2\.js["']><\/script>/i, `<script>\n${builder}\n</script>`);

  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  page.on('console', msg => {
    try {
      const text = msg.text();
      console.log('PAGE-LOG>', text);
    } catch (e) {
      console.log('PAGE-CONSOLE-ERR', e && e.message);
    }
  });

  page.on('pageerror', err => console.log('PAGE-ERROR>', err.stack || err.toString()));

  await page.setContent(content, { waitUntil: 'load' });
  await new Promise(res => setTimeout(res, 300));

  // perform interactions: add one context/signal/trigger and set some params
  await page.evaluate(() => {
    try {
      const addCtx = document.getElementById('add-context');
      const addSig = document.getElementById('add-signal');
      const addTrg = document.getElementById('add-trigger');
      if (addCtx) addCtx.click();
      if (addSig) addSig.click();
      if (addTrg) addTrg.click();
    } catch (e) { console.error('interaction-clicks failed', e); }
  });

  await new Promise(res => setTimeout(res, 200));

  // fill some inputs inside created rows
  await page.evaluate(() => {
    try {
      // set first signal to rsi_threshold if present
      const sigRow = document.querySelector('#signal-rules .rule-row');
      if (sigRow) {
        const sel = sigRow.querySelector('.rule-type') || sigRow.querySelector('select[data-field="type"]');
        if (sel) sel.value = 'rsi_threshold';
        const len = sigRow.querySelector('input[data-field="length"]') || sigRow.querySelector('.param[data-key="length"]') || sigRow.querySelector('input[name="length"]');
        if (len) { len.value = '14'; len.dispatchEvent(new Event('input', { bubbles: true })); }
        const thr = sigRow.querySelector('input[data-field="threshold"]') || sigRow.querySelector('.param[data-key="threshold"]');
        if (thr) { thr.value = '70'; thr.dispatchEvent(new Event('input', { bubbles: true })); }
      }

      // set first trigger to pin_bar custom param
      const trgRow = document.querySelector('#trigger-rules .rule-row');
      if (trgRow) {
        const sel = trgRow.querySelector('.rule-type') || trgRow.querySelector('select[data-field="type"]');
        if (sel) sel.value = 'pin_bar';
        const custom = trgRow.querySelector('.rule-custom') || trgRow.querySelector('textarea[data-field="bull_expr"]') || trgRow.querySelector('.param[data-key="custom"]');
        if (custom) { custom.value = 'pinbar'; custom.dispatchEvent(new Event('input', { bubbles: true })); }
      }

      // commit sections if commitSectionFromDOM exists
      if (window.commitSectionFromDOM) {
        try { commitSectionFromDOM('context'); commitSectionFromDOM('signal'); commitSectionFromDOM('trigger'); } catch (e) {}
      }
    } catch (e) { console.error('fill-rows failed', e); }
  });

  await new Promise(res => setTimeout(res, 200));

  // Click submit button (we'll intercept navigation by preventing default on submit)
  await page.evaluate(() => {
    const form = document.querySelector('form');
    if (!form) return;
    // prevent navigation but allow submit handlers to run
    form.addEventListener('submit', function (ev) { ev.preventDefault(); }, { once: true, capture: true });
    const btn = form.querySelector('button[type="submit"]') || form.querySelector('button');
    if (btn) btn.click();
  });

  await new Promise(res => setTimeout(res, 500));

  // read the last write stamp if present
  const last = await page.evaluate(() => { return window.__guided_v2_last_write || null; });
  console.log('LAST_WRITE>', JSON.stringify(last, null, 2));

  await browser.close();
  process.exit(0);
})();
