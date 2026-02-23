/**
 * Basic DOM integration tests for guided_builder_v2.js
 * - Ensures submit handler wired and clicking Review serializes hidden inputs
 */
/* eslint-env jest */

const fs = require('fs');
const path = require('path');

describe('guided_builder_v2 DOM wiring', () => {
  beforeEach(() => {
    // Setup minimal DOM for step4 form
    document.body.innerHTML = `
      <form id="guided_step4_form" action="/create-strategy-guided/step4">
        <input type="hidden" id="draft_id" name="draft_id" value="" />
        <div id="context-rules"></div>
        <div id="signal-rules"></div>
        <div id="trigger-rules"></div>
        <button type="submit" id="review_btn">Review</button>
      </form>
    `;
  });

  afterEach(() => {
    // clear modules so requiring script re-runs
    jest.resetModules();
    document.body.innerHTML = '';
  });

  test('clicking Review populates hidden JSON inputs without throwing', async () => {
    // load the script after DOM is prepared
    const scriptPath = path.resolve(__dirname, '..', 'guided_builder_v2.js');
    // require executes the IIFE and wires handlers
    require(scriptPath);

    const form = document.getElementById('guided_step4_form');
    expect(form).toBeTruthy();

    const btn = document.getElementById('review_btn');
    expect(btn).toBeTruthy();

    // Spy on form submission to prevent actually submitting
    const submitSpy = jest.spyOn(HTMLFormElement.prototype, 'submit').mockImplementation(() => {});

    // Click the button
    btn.click();

    // Hidden inputs should exist and contain JSON arrays
    const ctx = form.querySelector('#context_rules_json');
    const sig = form.querySelector('#signal_rules_json');
    const trg = form.querySelector('#trigger_rules_json');
    expect(ctx).toBeTruthy();
    expect(sig).toBeTruthy();
    expect(trg).toBeTruthy();
    expect(() => JSON.parse(ctx.value)).not.toThrow();
    expect(() => JSON.parse(sig.value)).not.toThrow();
    expect(() => JSON.parse(trg.value)).not.toThrow();

    submitSpy.mockRestore();
  });
});
