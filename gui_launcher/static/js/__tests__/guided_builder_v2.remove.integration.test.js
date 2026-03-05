const fs = require('fs');
const path = require('path');
// Ensure TextEncoder/TextDecoder globals for jsdom in Node environments
if (typeof global.TextEncoder === 'undefined') global.TextEncoder = require('util').TextEncoder;
if (typeof global.TextDecoder === 'undefined') global.TextDecoder = require('util').TextDecoder;
const { JSDOM } = require('jsdom');

describe('guided_builder_v2 delegated remove click', () => {
  let dom; let window; let document; let script;
  beforeAll(() => {
    const builder = fs.readFileSync(path.resolve(__dirname, '..', 'guided_builder_v2.js'), 'utf8');
    // Include the elements needed by both IIFEs in guided_builder_v2.js:
    // - The main init() path (wires add/remove handlers)
    // - The effective-preview path (exposes window.ruleRow for add-row rendering)
    const harness = `<!doctype html><html><body>
      <form id="guided_step4_form">
        <input id="context_rules_json" />
        <input id="signal_rules_json" />
        <input id="trigger_rules_json" />
        <input id="trade_filters_json" />
      </form>
      <div id="effective-preview-body"></div>
      <div id="context-rules"></div>
      <div id="signal-rules"></div>
      <div id="trigger-rules"></div>
      <div id="trade-filters"></div>
      <button id="add-trade-filter">Add Trade Filter</button>
    </body></html>`;
    dom = new JSDOM(harness, { runScripts: 'dangerously', resources: 'usable' });
    window = dom.window; document = window.document;
    // expose minimal globals expected by the script
    window.GuidedBuilderV2State = { set: () => {}, flush: () => {}, raw: () => ({}) };
    window.guidedBuilderV2PreferFullRenderer = true;
    // evaluate the builder script in the window context
    const scriptEl = document.createElement('script');
    scriptEl.textContent = builder;
    document.body.appendChild(scriptEl);

    // jsdom often executes injected scripts while document.readyState is
    // still "loading"; guided_builder_v2.js defers init() to DOMContentLoaded
    // in that case. Fire it explicitly so wiring is installed.
    try { document.dispatchEvent(new window.Event('DOMContentLoaded', { bubbles: true })); } catch (e) {}
  });

  test('remove by button text triggers commitSectionFromDOM', (done) => {
    const calls = [];
    const original = window.__guided_v2_commitSectionFromDOM;
    expect(typeof original).toBe('function');
    window.__guided_v2_commitSectionFromDOM = function (s) { calls.push(s); return original(s); };
    // add a row that uses the advanced renderer remove button (no class)
    const container = document.getElementById('context-rules');
    // create DOM snippet similar to advanced renderer
    container.innerHTML = `<div class="card rule-row" data-section="context"><div><button type="button">Remove</button></div></div>`;
    // click the remove button
    const btn = container.querySelector('button');
    btn.click();
    setTimeout(() => {
      try {
        expect(calls).toContain('context');
        done();
      } catch (e) { done(e); }
    }, 0);
  });

  test('remove by .rule-remove class triggers commitSectionFromDOM', (done) => {
    const calls = [];
    const original = window.__guided_v2_commitSectionFromDOM;
    expect(typeof original).toBe('function');
    window.__guided_v2_commitSectionFromDOM = function (s) { calls.push(s); return original(s); };
    const container = document.getElementById('signal-rules');
    container.innerHTML = `<div class="card rule-row" data-section="signal"><div><button type="button" class="rule-remove">Remove</button></div></div>`;
    const btn = container.querySelector('button');
    btn.click();
    setTimeout(() => {
      try { expect(calls).toContain('signal'); done(); } catch (e) { done(e); }
    }, 0);
  });

    test('trade-filters remove by button text triggers commitSectionFromDOM', (done) => {
      const calls = [];
      const original = window.__guided_v2_commitSectionFromDOM;
      expect(typeof original).toBe('function');
      window.__guided_v2_commitSectionFromDOM = function (s) { calls.push(s); return original(s); };
      const container = document.getElementById('trade-filters');
      container.innerHTML = `<div class="card rule-row" data-section="trade_filters"><div><button type="button">Remove</button></div></div>`;
      const btn = container.querySelector('button');
      btn.click();
      setTimeout(() => {
        try { expect(calls).toContain('trade_filters'); done(); } catch (e) { done(e); }
      }, 0);
    });

    test('add-trade appends a rule-row when clicked', (done) => {
      // Ensure GUIDED state is absent so the DOM path is exercised
      window.GuidedBuilderV2State = null;
      const btn = document.getElementById('add-trade-filter');
      try { btn.click(); } catch (e) {}
      // Appending can be deferred by the retry strategy; allow a short window.
      setTimeout(() => {
        try {
          const container = document.getElementById('trade-filters');
          const row = container.querySelector('.rule-row');
          expect(row).not.toBeNull();
          done();
        } catch (e) { done(e); }
      }, 350);
    });
});
