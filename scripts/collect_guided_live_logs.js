const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

if (process.argv.length < 3) {
  console.error('Usage: node collect_guided_live_logs.js <URL>');
  process.exit(2);
}

(async () => {
  const url = process.argv[2];
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

  console.log('Navigating to', url);
  await page.goto(url, { waitUntil: 'load', timeout: 30000 });
  await new Promise((res) => setTimeout(res, 300));

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

  await new Promise((res) => setTimeout(res, 200));

  // fill some inputs inside created rows
  await page.evaluate(() => {
    try {
      const sigRow = document.querySelector('#signal-rules .rule-row');
      if (sigRow) {
        const sel = sigRow.querySelector('.rule-type') || sigRow.querySelector('select[data-field="type"]');
        if (sel) sel.value = 'rsi_threshold';
        const len = sigRow.querySelector('input[data-field="length"]') || sigRow.querySelector('.param[data-key="length"]') || sigRow.querySelector('input[name="length"]');
        if (len) { len.value = '14'; len.dispatchEvent(new Event('input', { bubbles: true })); }
        const thr = sigRow.querySelector('input[data-field="threshold"]') || sigRow.querySelector('.param[data-key="threshold"]');
        if (thr) { thr.value = '70'; thr.dispatchEvent(new Event('input', { bubbles: true })); }
      }

      const trgRow = document.querySelector('#trigger-rules .rule-row');
      if (trgRow) {
        const sel = trgRow.querySelector('.rule-type') || trgRow.querySelector('select[data-field="type"]');
        if (sel) sel.value = 'pin_bar';
        const custom = trgRow.querySelector('.rule-custom') || trgRow.querySelector('textarea[data-field="bull_expr"]') || trgRow.querySelector('.param[data-key="custom"]');
        if (custom) { custom.value = 'pinbar'; custom.dispatchEvent(new Event('input', { bubbles: true })); }
      }

      if (window.commitSectionFromDOM) {
        try { commitSectionFromDOM('context'); commitSectionFromDOM('signal'); commitSectionFromDOM('trigger'); } catch (e) {}
      }
    } catch (e) { console.error('fill-rows failed', e); }
  });

  await new Promise((res) => setTimeout(res, 200));

  // click submit and wait for navigation to complete
  console.log('Clicking submit...');
  const submitted = await page.evaluate(() => {
    const form = document.querySelector('form');
    if (!form) return false;
    // ensure submit handlers run; do not intercept navigation
    const btn = form.querySelector('button[type="submit"]') || form.querySelector('button');
    if (btn) { btn.click(); return true; }
    return false;
  });

  if (submitted) {
    try {
      await page.waitForNavigation({ waitUntil: 'load', timeout: 10000 });
      console.log('Navigation after submit complete');
    } catch (e) {
      console.log('No navigation or timeout waiting for navigation', e && e.message);
    }
  } else {
    console.log('Submit button not found');
  }

  // read the last write stamp if present (may be on previous page)
  try {
    const last = await page.evaluate(() => { return window.__guided_v2_last_write || null; });
    console.log('LAST_WRITE>', JSON.stringify(last, null, 2));
  } catch (e) {
    console.log('Error reading LAST_WRITE from page', e && e.message);
  }

  await browser.close();
  process.exit(0);
})();
