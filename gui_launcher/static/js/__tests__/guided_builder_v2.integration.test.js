/**
 * Integration DOM tests for guided_builder_v2.js
 * - Adds context/signal/trigger rows, simulates Review click
 * - Verifies hidden JSON inputs contain the expected rule objects
 */
/* eslint-env jest */

const path = require('path');

describe('guided_builder_v2 integration', () => {
  beforeEach(() => {
    document.body.innerHTML = `
      <form id="guided_step4_form" action="/create-strategy-guided/step4">
        <input type="hidden" id="draft_id" name="draft_id" value="test-draft" />
        <div id="context-rules"></div>
        <div id="signal-rules"></div>
        <div id="trigger-rules"></div>
        <button type="submit" id="review_btn">Review</button>
      </form>
    `;
  });

  afterEach(() => {
    jest.resetModules();
    document.body.innerHTML = '';
  });

  test('serializes DOM rule rows into hidden JSON on Review', async () => {
    const scriptPath = path.resolve(__dirname, '..', 'guided_builder_v2.js');
    require(scriptPath);

    const form = document.getElementById('guided_step4_form');
    expect(form).toBeTruthy();

    // Create a context rule row (ma_cross) with params
    const ctx = document.getElementById('context-rules');
    ctx.innerHTML = `
      <div class="card rule-row">
        <select class="rule-type"><option value="ma_cross">MA cross</option></select>
        <select class="rule-side"><option value="both">both</option></select>
        <input class="rule-valid" type="number" value="5" />
        <div class="rule-params"><input class="param" data-key="fast" value="5" /><input class="param" data-key="slow" value="21" /></div>
      </div>`;

    // Create a signal rule row (rsi_threshold)
    const sig = document.getElementById('signal-rules');
    sig.innerHTML = `
      <div class="card rule-row">
        <select class="rule-type"><option value="rsi_threshold">RSI</option></select>
        <select class="rule-side"><option value="long">long</option></select>
        <input class="rule-valid" type="number" value="3" />
        <div class="rule-params"><input class="param" data-key="length" value="14" /><input class="param" data-key="threshold" value="70" /></div>
      </div>`;

    // Create a trigger rule row (pin_bar)
    const trg = document.getElementById('trigger-rules');
    trg.innerHTML = `
      <div class="card rule-row">
        <select class="rule-type"><option value="pin_bar">Pin bar</option></select>
        <select class="rule-side"><option value="both">both</option></select>
        <input class="rule-valid" type="number" value="1" />
        <div class="rule-params"><textarea class="param" data-key="custom">pinbar</textarea></div>
      </div>`;

    // Spy on native submit to avoid navigating
    const submitSpy = jest.spyOn(HTMLFormElement.prototype, 'submit').mockImplementation(() => {});

    // Click Review
    const btn = document.getElementById('review_btn');
    btn.click();

    // Validate hidden inputs
    const ctxH = document.getElementById('context_rules_json');
    const sigH = document.getElementById('signal_rules_json');
    const trgH = document.getElementById('trigger_rules_json');
    expect(ctxH).toBeTruthy(); expect(sigH).toBeTruthy(); expect(trgH).toBeTruthy();

    const parsedCtx = JSON.parse(ctxH.value);
    const parsedSig = JSON.parse(sigH.value);
    const parsedTrg = JSON.parse(trgH.value);

    expect(Array.isArray(parsedCtx)).toBe(true);
    expect(parsedCtx.length).toBe(1);
    expect(parsedCtx[0].type).toBe('ma_cross');

    expect(Array.isArray(parsedSig)).toBe(true);
    expect(parsedSig.length).toBe(1);
    expect(parsedSig[0].type).toBe('rsi_threshold');

    expect(Array.isArray(parsedTrg)).toBe(true);
    expect(parsedTrg.length).toBe(1);
    expect(parsedTrg[0].type).toBe('pin_bar');

    submitSpy.mockRestore();
  });
});
