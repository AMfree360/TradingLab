const puppeteer = require('puppeteer');

async function waitForNavigationQuiet(page) {
  try { await page.waitForNavigation({ waitUntil: 'networkidle2', timeout: 15000 }); } catch (e) {}
}

(async () => {
  const browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const page = await browser.newPage();
  page.setDefaultTimeout(30000);
  const base = 'http://127.0.0.1:8000';

  try {
    await page.goto(base + '/create-strategy-guided', { waitUntil: 'networkidle2' });

    // Step1 create
    await page.waitForSelector('form[action="/create-strategy-guided/step1"]');
    const name = 'puppeteer_roundtrip_' + Date.now();
    await page.type('form[action="/create-strategy-guided/step1"] input[name="name"]', name);
    await Promise.all([page.waitForNavigation({ waitUntil: 'networkidle2' }), page.click('form[action="/create-strategy-guided/step1"] button[type=submit]')]);

    // Step2
    await page.waitForSelector('form[action="/create-strategy-guided/step2"]');
    await Promise.all([page.waitForNavigation({ waitUntil: 'networkidle2' }), page.click('form[action="/create-strategy-guided/step2"] button[type=submit]')]);

    // Step3 -> Step4
    await page.waitForSelector('form[action="/create-strategy-guided/step3"] , form#guided_step4_form');
    const step3Form = await page.$('form[action="/create-strategy-guided/step3"]');
    if (step3Form) {
      await Promise.all([page.waitForNavigation({ waitUntil: 'networkidle2' }), page.click('form[action="/create-strategy-guided/step3"] button[type=submit], form[action="/create-strategy-guided/step3"] button.primary')]);
    }

    // Ensure we're on Step4
    await page.waitForSelector('#guided_step4_form');

    // Add a context rule and set its side to 'long' and a custom param
    await page.click('#add-context');
    await page.waitForSelector('#context-rules .rule-row');
    // Set type to price_vs_ma (if not already) and set custom param via data-field or textarea
    await page.evaluate(() => {
      const r = document.querySelector('#context-rules .rule-row');
      if (!r) return;
      const type = r.querySelector('[data-field="type"]') || r.querySelector('.rule-type');
      if (type) { type.value = 'price_vs_ma'; type.dispatchEvent(new Event('change', { bubbles: true })); }
      const side = r.querySelector('[data-field="side"]') || r.querySelector('.rule-side');
      if (side) { side.value = 'long'; side.dispatchEvent(new Event('change', { bubbles: true })); }
      const maType = r.querySelector('[data-field="ma_type"]') || r.querySelector('.param[data-key="ma_type"]');
      if (maType) { maType.value = 'ema'; maType.dispatchEvent(new Event('change', { bubbles: true })); }
      const len = r.querySelector('[data-field="length"]') || r.querySelector('.param[data-key="length"]');
      if (len) { len.value = '123'; len.dispatchEvent(new Event('input', { bubbles: true })); }
    });

    // Capture draft_id (present on the Step4 form) before submit
    const preDraftId = await page.evaluate(() => {
      const el = document.getElementById('draft_id') || document.querySelector('input[name="draft_id"]');
      return el ? el.value || null : null;
    });
    console.log('Captured draft_id before submit:', preDraftId);

    // Force sync and submit Review (Review button submits form)
    await page.evaluate(() => {
      const f = document.getElementById('guided_step4_form') || document.querySelector('form[action="/create-strategy-guided/step4"]');
      if (!f) return;
      try { if (window.__guided_v2_invoke_final_pre_write) window.__guided_v2_invoke_final_pre_write(f); } catch (e) {}
    });

    await Promise.all([page.waitForNavigation({ waitUntil: 'networkidle2' }), page.click('#guided_step4_form button[type=submit]')]);

    // After submit, we expect to be on review or next page; capture draft id from URL or hidden
    const url = page.url();
    console.log('After submit URL:', url);

    // Reopen Step4 using the pre-captured draft id
    const draftId = preDraftId;
    if (!draftId) throw new Error('draft_id not captured before save');
    console.log('Draft ID (used to reopen):', draftId);

    // Open edit Step4
    await page.goto(base + '/create-strategy-guided/step4?draft_id=' + encodeURIComponent(draftId), { waitUntil: 'networkidle2' });
    await page.waitForSelector('#guided_step4_form');

    // Read back the first context rule's side and length
    const persisted = await page.evaluate(() => {
      const r = document.querySelector('#context-rules .rule-row');
      if (!r) return null;
      const side = (r.querySelector('[data-field="side"]') || r.querySelector('.rule-side'));
      const len = (r.querySelector('[data-field="length"]') || r.querySelector('.param[data-key="length"]') || r.querySelector('input[name="length"]'));
      return { side: side ? side.value : null, length: len ? (len.value || null) : null };
    });

    console.log('Persisted values:', persisted);
    if (!persisted) throw new Error('No rule row found on reopen');
    if (String(persisted.side).toLowerCase() !== 'long') throw new Error('Side did not persist as long: ' + persisted.side);
    if (String(persisted.length) !== '123') throw new Error('Length did not persist as 123: ' + persisted.length);

    console.log('ROUNDTRIP: persisted values match — SUCCESS');
    await browser.close();
    process.exit(0);
  } catch (e) {
    console.error('ROUNDTRIP ERROR:', e && (e.stack || e.message || e));
    await browser.close();
    process.exit(3);
  }
})();
