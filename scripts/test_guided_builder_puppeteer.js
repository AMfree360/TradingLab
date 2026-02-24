const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const page = await browser.newPage();
  page.setDefaultTimeout(30000);

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
