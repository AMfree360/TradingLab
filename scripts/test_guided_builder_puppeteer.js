const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const page = await browser.newPage();
  page.setDefaultTimeout(30000);
  // Capture page console + errors for diagnostics
  page.on('console', msg => { try { console.log('PAGE LOG:', msg.text()); } catch (e) {} });
  page.on('pageerror', err => { try { console.error('PAGE ERROR:', err && (err.stack||err.message||err)); } catch (e) {} });

  const base = 'http://127.0.0.1:8000';
  try {
    await page.goto(base + '/create-strategy-guided', { waitUntil: 'networkidle2' });

    // Step 1: create draft
    await page.waitForSelector('form[action="/create-strategy-guided/step1"]');
    const name = 'puppeteer_test_' + Date.now();
    await page.type('form[action="/create-strategy-guided/step1"] input[name="name"]', name);
    await Promise.all([
      page.waitForNavigation({ waitUntil: 'networkidle2' }),
      page.click('form[action="/create-strategy-guided/step1"] button[type=submit]')
    ]);

    // Step 2: submit defaults
    await page.waitForSelector('form[action="/create-strategy-guided/step2"]');
    await Promise.all([
      page.waitForNavigation({ waitUntil: 'networkidle2' }),
      page.click('form[action="/create-strategy-guided/step2"] button[type=submit]')
    ]);

    // Step 3: submit defaults (entry + risk)
    await page.waitForSelector('form[action="/create-strategy-guided/step3"] , form#guided_step4_form');
    // Some deployments use step3->step4 redirect; try clicking the Next/Review button if present
    const step3Form = await page.$('form[action="/create-strategy-guided/step3"]');
    if (step3Form) {
      await Promise.all([
        page.waitForNavigation({ waitUntil: 'networkidle2' }),
        page.click('form[action="/create-strategy-guided/step3"] button[type=submit], form[action="/create-strategy-guided/step3"] button.primary')
      ]);
    }

    // Now on Step 4 (builder v2)
    await page.waitForSelector('#guided_step4_form');
    await page.waitForSelector('#context-rules');

    // Add a context rule and assert the full renderer is active (presence of .rule-side control)
    await page.click('#add-context');
    await page.waitForSelector('.rule-row');
    // Dump first rule-row and container HTML for debugging
    const rowInfo = await page.evaluate(() => {
      const r = document.querySelector('.rule-row');
      const container = document.getElementById('context-rules');
      return {
        rowHtml: r ? r.outerHTML : null,
        containerHtml: container ? container.outerHTML : null,
        sidePresent: r ? !!r.querySelector('.rule-side') : false,
        dataSidePresent: r ? !!r.querySelector('[data-field="side"]') : false,
      };
    });
    console.log('ROW OUTERHTML:', rowInfo.rowHtml ? rowInfo.rowHtml.slice(0,2000) : 'NO ROW');
    console.log('CONTAINER HTML SNIPPET:', rowInfo.containerHtml ? rowInfo.containerHtml.slice(0,2000) : 'NO CONTAINER');
    if (!(rowInfo.sidePresent || rowInfo.dataSidePresent)) throw new Error('Full renderer missing: side selector not found (class or data-field)');

    // Add a signal and trigger row to exercise wiring
    await page.click('#add-signal');
    await page.waitForSelector('#signal-rules .rule-row');
    await page.click('#add-trigger');
    await page.waitForSelector('#trigger-rules .rule-row');

    // Verify single-click remove behavior for context/signal/trigger rows
    async function testSingleClickRemove(containerId) {
      // Wait until a row exists and contains a remove button (either `.rule-remove`
      // or a button whose text is 'Remove'). Then click it once.
      await page.waitForFunction((c) => {
        const row = document.querySelector(c + ' .rule-row');
        if (!row) return false;
        const btn = row.querySelector('.rule-remove') || Array.from(row.querySelectorAll('button')).find(b => (b.textContent||'').trim().toLowerCase() === 'remove');
        return !!btn;
      }, { timeout: 5000 }, containerId);
      await page.evaluate((c) => {
        const row = document.querySelector(c + ' .rule-row');
        if (!row) return;
        const btn = row.querySelector('.rule-remove') || Array.from(row.querySelectorAll('button')).find(b => (b.textContent||'').trim().toLowerCase() === 'remove');
        if (btn) btn.click();
      }, containerId);
      try {
        await page.waitForFunction((c) => document.querySelectorAll(c + ' .rule-row').length === 0, { timeout: 3000 }, containerId);
        return { ok: true };
      } catch (e) {
        // capture diagnostic
        const html = await page.$eval(containerId, el => el.outerHTML);
        return { ok: false, html };
      }
    }

    const ctxRemove = await testSingleClickRemove('#context-rules');
    console.log('CTX REMOVE RESULT:', ctxRemove);
    const sigRemove = await testSingleClickRemove('#signal-rules');
    console.log('SIG REMOVE RESULT:', sigRemove);
    const trgRemove = await testSingleClickRemove('#trigger-rules');
    console.log('TRG REMOVE RESULT:', trgRemove);
    if (!ctxRemove.ok || !sigRemove.ok || !trgRemove.ok) throw new Error('One or more sections failed single-click remove');

    // Reproduce: add a context row, set a specific type, then remove and
    // check whether an empty row is re-inserted shortly after removal.
    async function reproduceRemoveReplace(containerId) {
      // Add a fresh row
      await page.click('#add-context');
      await page.waitForSelector(containerId + ' .rule-row');
      // Set the first row type to 'price_vs_ma' so params render.
      // Support both classic renderer (.rule-type) and advanced renderer ([data-field="type"]).
      await page.waitForFunction((c) => {
        const row = document.querySelector(c + ' .rule-row');
        if (!row) return false;
        return !!(row.querySelector('.rule-type') || row.querySelector('[data-field="type"]'));
      }, { timeout: 2000 }, containerId);
      await page.evaluate((c) => {
        const row = document.querySelector(c + ' .rule-row');
        if (!row) return;
        const sel = row.querySelector('.rule-type') || row.querySelector('[data-field="type"]');
        if (!sel) return;
        sel.value = 'price_vs_ma';
        sel.dispatchEvent(new Event('change', { bubbles: true }));
      }, containerId);
      // wait for params container to include a param (ma_type select)
      await page.waitForFunction((c) => !!document.querySelector(c + ' .rule-row .rule-params select[data-key="ma_type"]' ) || !!document.querySelector(c + ' .rule-row .rule-params [data-key="ma_type"]'), { timeout: 2000 }, containerId);
      // click remove once
      await page.evaluate((c) => {
        const row = document.querySelector(c + ' .rule-row');
        const btn = row && (row.querySelector('.rule-remove') || Array.from(row.querySelectorAll('button')).find(b => (b.textContent||'').trim().toLowerCase() === 'remove'));
        if (btn) btn.click();
      }, containerId);
      // wait briefly to see if a new empty row appears
      await new Promise(r => setTimeout(r, 600));
      const after = await page.evaluate((c) => {
        const container = document.querySelector(c);
        const rows = container ? Array.from(container.querySelectorAll('.rule-row')) : [];
        return rows.map(r => ({ outer: r.outerHTML.slice(0,1000), hasTypeSelect: !!r.querySelector('.rule-type'), hasSide: !!r.querySelector('.rule-side'), hasValid: !!r.querySelector('.rule-valid'), paramsHtml: (r.querySelector('.rule-params')||{}).innerHTML.slice(0,200) }));
      }, containerId);
      return after;
    }

    const reproduce = await reproduceRemoveReplace('#context-rules');
    console.log('REPRODUCE AFTER REMOVE:', JSON.stringify(reproduce, null, 2));
    // If any rows remain, fail so we can inspect and fix
    if (reproduce && reproduce.length > 0) throw new Error('Remove left behind rows — reproducer detected');

    // Debug: inspect add-trade UI wiring before clicking
    const debugBefore = await page.evaluate(() => {
      const btn = document.getElementById('add-trade-filter');
      const container = document.getElementById('trade-filters');
      return {
        addTradeExists: !!btn,
        addTradeHtml: btn ? btn.outerHTML : null,
        addTradeDataset: btn ? Object.assign({}, btn.dataset) : null,
        tradeContainerExists: !!container,
        tradeContainerHtml: container ? container.outerHTML : null,
        hasState: !!window.GuidedBuilderV2State,
        preferFull: !!window.guidedBuilderV2PreferFullRenderer,
        lastWrite: window.__guided_v2_last_write || null
      };
    });
    console.log('DEBUG BEFORE ADD-TRADE:', JSON.stringify(debugBefore));
    // Collect diagnostics before clicking Add Trade Filter
    const diagBefore = await page.evaluate(() => {
      try {
        return {
          addTradeBtn: !!document.getElementById('add-trade-filter'),
          addTradeDataset: (function(){ const b = document.getElementById('add-trade-filter'); if (!b) return null; const d = {}; for (const k of Object.keys(b.dataset||{})) d[k]=b.dataset[k]; return d; })(),
          tradeContainerExists: !!document.getElementById('trade-filters'),
          tradeContainerOuter: (document.getElementById('trade-filters')||{}).outerHTML || null,
          apiKeys: (window._guided_builder_v2_api) ? Object.keys(window._guided_builder_v2_api) : null,
          ruleRowType: typeof window.ruleRow,
          ruleRowFnType: typeof window.ruleRow,
          guidedPreferFull: !!window.guidedBuilderV2PreferFullRenderer,
          hasState: !!window.GuidedBuilderV2State,
          lastWrite: window.__guided_v2_last_write || null
        };
      } catch (e) { return { error: String(e) }; }
    });
    console.log('DIAG BEFORE ADD-TRADE:', JSON.stringify(diagBefore, null, 2));

    // Click Add Trade Filter
    await page.click('#add-trade-filter');
    try {
      await page.waitForSelector('#trade-filters .rule-row', { timeout: 5000 });
      // Verify trade filter row does NOT include the `valid_for_bars` control
      const tradeHasValid = await page.evaluate(() => {
        const r = document.querySelector('#trade-filters .rule-row');
        return !!(r && r.querySelector('.rule-valid'));
      });
      if (tradeHasValid) throw new Error('Trade filter row unexpectedly contains valid-for-bars control');
    } catch (err) {
      const inner = await page.$eval('#trade-filters', el => el.innerHTML);
      console.error('TRADE-FILTERS HTML AFTER CLICK:', inner ? inner.slice(0,2000) : '<empty>');
      try {
        const diagAfter = await page.evaluate(() => {
          try {
            return {
              tradeContainerOuter: document.getElementById('trade-filters') ? document.getElementById('trade-filters').outerHTML : null,
              ruleRows: Array.from(document.querySelectorAll('#trade-filters .rule-row')).map(r => r.outerHTML.slice(0,1000)),
              apiKeys: (window._guided_builder_v2_api) ? Object.keys(window._guided_builder_v2_api) : null,
              globalFns: { ruleRow: typeof window.ruleRow },
              preferFull: !!window.guidedBuilderV2PreferFullRenderer,
              hasState: !!window.GuidedBuilderV2State,
              lastWrite: window.__guided_v2_last_write || null
            };
          } catch (e) { return { error: String(e) }; }
        });
        console.error('DIAG AFTER ADD-TRADE:', JSON.stringify(diagAfter, null, 2));
      } catch (e) {}
      throw err;
    }

    // Trigger a preview reload and wait for the setup_visual API to respond
    const setupReq = page.waitForResponse(resp => resp.url().includes('/api/guided/builder_v2/setup_visual') && resp.status() === 200, { timeout: 15000 });
    await page.click('#reload-setup-visual');
    await setupReq;

    // Check effective preview body updated
    await page.waitForFunction(() => {
      const el = document.getElementById('effective-preview-body');
      if (!el) return false;
      return el.innerText && !/Preview unavailable/i.test(el.innerText);
    }, { timeout: 10000 });

    console.log('PUPPETEER: full renderer active and preview updated — SUCCESS');
    await browser.close();
    process.exit(0);
  } catch (e) {
    console.error('PUPPETEER ERROR:', e && (e.stack || e.message || e));
    await browser.close();
    process.exit(2);
  }
})();
