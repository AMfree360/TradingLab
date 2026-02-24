const fs = require('fs');
const puppeteer = require('puppeteer');

if (process.argv.length < 3) {
  console.error('Usage: node collect_with_history.js <URL>');
  process.exit(2);
}

(async () => {
  const url = process.argv[2];
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  // Ensure the page's scripts see the debug flag so debug-only persistence
  // (window globals and localStorage writes) is enabled during the run.
  await page.evaluateOnNewDocument(() => {
    try {
      // Ensure debug gate is disabled by default for normal harness runs.
      // Tests that need debug persistence should explicitly enable it.
        // Enable debug gate for harness runs so client-side debug persistence
        // (write history and DOM snapshots) is active. Also point client debug
        // beacons to the server `/debug/beacon` endpoint so the server can
        // persist them to disk.
        window.__GUIDED_V2_DEBUG = true;
        window.__guided_v2_debug_beacon_url = '/debug/beacon';
    } catch (e) {}
  });

  page.on('console', msg => {
    try { console.log('PAGE-LOG>', msg.text()); } catch (e) { console.log('PAGE-CONSOLE-ERR', e && e.message); }
  });
  page.on('pageerror', err => console.log('PAGE-ERROR>', err.stack || err.toString()));

  // Capture outgoing POST requests (beacons/XHR) so we can inspect payloads
  page.on('request', req => {
    try {
      if (req.method && req.method() === 'POST') {
        const url = req.url();
        const post = req.postData ? req.postData() : null;
        try { console.log('PAGE-REQ>', JSON.stringify({ url: url, postData: post })); } catch (e) {}
      }
    } catch (e) {}
  });

  console.log('Navigating to', url);
  await page.goto(url, { waitUntil: 'load', timeout: 30000 });
  await new Promise(res => setTimeout(res, 300));

  // interactions
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

  await page.evaluate(() => {
    try {
      const sigRow = document.querySelector('#signal-rules .rule-row');
      if (sigRow) {
        const sel = sigRow.querySelector('.rule-type'); if (sel) sel.value = 'rsi_threshold';
        const len = sigRow.querySelector('.param[data-key="length"]') || sigRow.querySelector('input[data-field="length"]');
        if (len) { len.value = '14'; len.dispatchEvent(new Event('input', { bubbles: true })); }
        const thr = sigRow.querySelector('.param[data-key="threshold"]'); if (thr) { thr.value = '70'; thr.dispatchEvent(new Event('input', { bubbles: true })); }
      }
      const trgRow = document.querySelector('#trigger-rules .rule-row');
      if (trgRow) { const sel = trgRow.querySelector('.rule-type'); if (sel) sel.value = 'pin_bar'; const custom = trgRow.querySelector('.param[data-key="custom"]'); if (custom) { custom.value = 'pinbar'; custom.dispatchEvent(new Event('input', { bubbles: true })); } }
      if (window.commitSectionFromDOM) { try { commitSectionFromDOM('context'); commitSectionFromDOM('signal'); commitSectionFromDOM('trigger'); } catch (e) {} }
    } catch (e) { console.error('fill-rows failed', e); }
  });
  await new Promise(res => setTimeout(res, 200));

  console.log('Clicking submit...');
  const submitted = await page.evaluate(() => {
    const form = document.querySelector('form'); if (!form) return false; const btn = form.querySelector('button[type="submit"]') || form.querySelector('button'); if (btn) { btn.click(); return true; } return false;
  });

  if (submitted) {
    // Give client handlers a short moment to run and stamp hidden inputs / write history
    await new Promise(res => setTimeout(res, 200));
    try {
      const lastPreNav = await page.evaluate(() => ({ last: window.__guided_v2_last_write || null, history: window.__guided_v2_write_history || [] }));
      console.log('PAGE-WRITES-PRE-NAV>', JSON.stringify(lastPreNav, null, 2));
    } catch (e) { console.log('Error reading pre-nav write history', e && e.message); }
    try { await page.waitForNavigation({ waitUntil: 'load', timeout: 10000 }); console.log('Navigation after submit complete'); } catch (e) { console.log('No navigation or timeout waiting for navigation', e && e.message); }
  } else { console.log('Submit button not found'); }

  try {
    // Give the page a short moment after navigation to allow client handlers
    // to persist debug state to localStorage; read persisted history if present.
    await new Promise(res => setTimeout(res, 500));
    const last = await page.evaluate(() => {
      try {
        const histRaw = (typeof localStorage !== 'undefined') ? localStorage.getItem('__guided_v2_write_history') : null;
        const lastRaw = (typeof localStorage !== 'undefined') ? localStorage.getItem('__guided_v2_last_write') : null;
        const domSnapRaw = (typeof localStorage !== 'undefined') ? localStorage.getItem('__guided_v2_dom_snapshot') : null;
        const domSnapsRaw = (typeof localStorage !== 'undefined') ? localStorage.getItem('__guided_v2_dom_snapshots') : null;
        const hist = histRaw ? JSON.parse(histRaw) : (window.__guided_v2_write_history || []);
        const last = lastRaw ? JSON.parse(lastRaw) : (window.__guided_v2_last_write || null);
        const domSnap = domSnapRaw ? JSON.parse(domSnapRaw) : (window.__guided_v2_last_dom_snapshot || null);
        const domSnaps = domSnapsRaw ? JSON.parse(domSnapsRaw) : (window.__guided_v2_dom_snapshots || []);
        return { last: last, history: hist, domSnapshot: domSnap, domSnapshots: domSnaps };
      } catch (e) {
        return ({ last: window.__guided_v2_last_write || null, history: window.__guided_v2_write_history || [], domSnapshot: window.__guided_v2_last_dom_snapshot || null, domSnapshots: window.__guided_v2_dom_snapshots || [] });
      }
    });
    console.log('PAGE-WRITES>', JSON.stringify(last, null, 2));
    try {
      if (last && last.domSnapshot) console.log('PAGE-DOM-SNAPSHOT>', JSON.stringify(last.domSnapshot));
      if (last && last.domSnapshots && Array.isArray(last.domSnapshots)) {
        const tail = last.domSnapshots.slice(-6);
        for (const s of tail) console.log('DOM-HIST>', JSON.stringify(s));
      }
    } catch (e) { }
    // print top 12 history entries succinctly
    const hist = last.history && Array.isArray(last.history) ? last.history.slice(-12) : [];
    for (const h of hist) console.log('HIST>', JSON.stringify(h));
  } catch (e) { console.log('Error reading write history', e && e.message); }

  await browser.close();
  process.exit(0);
})();
