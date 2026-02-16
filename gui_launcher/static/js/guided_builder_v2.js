/* Guided Builder v2 — State-integrated rule renderer (Phase 4)
   - Renders rule rows with per-type parameter blocks
   - Two-way sync: DOM <-> `GuidedBuilderV2State` and hidden JSON inputs
   - Keeps volume features behind FEATURE_FLAG.volume (not rendered)
*/
(function () {
  const FEATURE_FLAG = { volume: false };

  const contextTypes = [
    ['price_vs_ma', 'Trend: Price vs MA'],
    ['ma_cross_state', 'Trend: MA state (fast vs slow)'],
    ['atr_pct', 'Volatility: ATR% filter'],
    ['structure_breakout_state', 'Structure: Breakout state'],
    ['ma_spread_pct', 'Trend strength: MA spread %'],
    ['custom', 'Custom (DSL)']
  ];

  const signalTypes = [
    ['ma_cross', 'MA cross (fast vs slow)'],
    ['rsi_threshold', 'RSI threshold'],
    ['rsi_cross_back', 'RSI cross back'],
    ['donchian_breakout', 'Donchian breakout/breakdown'],
    ['new_high_low_breakout', 'New high/low breakout'],
    ['pullback_to_ma', 'Pullback to MA'],
    ['custom', 'Custom (DSL)']
  ];

  const triggerTypes = [
    ['pin_bar', 'Pin bar'],
    ['inside_bar_breakout', 'Inside bar breakout'],
    ['engulfing', 'Engulfing'],
    ['ma_reclaim', 'MA reclaim'],
    ['prior_bar_break', 'Break prior bar high/low'],
    ['donchian_breakout', 'Donchian breakout/breakdown'],
    ['range_breakout', 'Range breakout'],
    ['wide_range_candle', 'Wide-range candle (vs ATR)'],
    ['custom', 'Custom (DSL)']
  ];

  function el(tag, attrs) {
    const node = document.createElement(tag);
    if (!attrs) return node;
    for (const k of Object.keys(attrs)) {
      if (k === 'text') node.textContent = attrs[k];
      else if (k === 'html') node.innerHTML = attrs[k];
      else if (k === 'class') node.className = attrs[k];
      else node.setAttribute(k, String(attrs[k]));
    }
    return node;
  }

  function ensureHidden(form, id) {
    let elHidden = form.querySelector('#' + id);
    if (!elHidden) {
      elHidden = el('input', { type: 'hidden', id: id, name: id });
      elHidden.value = '[]';
      form.appendChild(elHidden);
    }
    return elHidden;
  }

  function parseJsonSafe(txt) {
    try { const p = JSON.parse(String(txt || '[]')); return Array.isArray(p) ? p : []; } catch (e) { return []; }
  }

  function renderParamsForType(container, section, type, rule) {
    container.innerHTML = '';
    if (!type) return;
    // Simple, explicit parameter sets for a few types
    if (type === 'price_vs_ma') {
      // ma_type, length, op, threshold
      const maType = el('select', { class: 'param', 'data-key': 'ma_type' });
      maType.appendChild(el('option', { value: 'sma', text: 'SMA' }));
      maType.appendChild(el('option', { value: 'ema', text: 'EMA' }));
      maType.value = (rule && rule.ma_type) ? rule.ma_type : 'sma';

      const len = el('input', { type: 'number', class: 'param', 'data-key': 'length', min: 1, value: (rule && rule.length) ? String(rule.length) : '20' });
      const op = el('select', { class: 'param', 'data-key': 'op' });
      op.appendChild(el('option', { value: '>=' , text: '≥' }));
      op.appendChild(el('option', { value: '<=' , text: '≤' }));
      op.appendChild(el('option', { value: '>' , text: '>' }));
      op.appendChild(el('option', { value: '<' , text: '<' }));
      op.value = (rule && rule.op) ? rule.op : '>=';
      const thr = el('input', { type: 'number', class: 'param', 'data-key': 'threshold', step: 0.01, min: 0, value: (rule && rule.threshold) ? String(rule.threshold) : '1.0' });

      container.appendChild(el('label', { text: 'MA type: ' })); container.appendChild(maType);
      container.appendChild(el('label', { text: ' MA length: ' })); container.appendChild(len);
      container.appendChild(el('label', { text: ' Op: ' })); container.appendChild(op);
      container.appendChild(el('label', { text: ' Threshold: ' })); container.appendChild(thr);
      return;
    }

    if (type === 'ma_cross' || type === 'ma_cross_state') {
      const fast = el('input', { type: 'number', class: 'param', 'data-key': 'fast', min: 1, value: (rule && rule.fast) ? String(rule.fast) : '8' });
      const slow = el('input', { type: 'number', class: 'param', 'data-key': 'slow', min: 1, value: (rule && rule.slow) ? String(rule.slow) : '21' });
      container.appendChild(el('label', { text: 'Fast: ' })); container.appendChild(fast);
      container.appendChild(el('label', { text: ' Slow: ' })); container.appendChild(slow);
      return;
    }

    if (type === 'rsi_threshold') {
      const len = el('input', { type: 'number', class: 'param', 'data-key': 'length', min: 1, value: (rule && rule.length) ? String(rule.length) : '14' });
      const thr = el('input', { type: 'number', class: 'param', 'data-key': 'threshold', min: 0, max: 100, value: (rule && rule.threshold) ? String(rule.threshold) : '70' });
      container.appendChild(el('label', { text: 'Length: ' })); container.appendChild(len);
      container.appendChild(el('label', { text: ' Threshold: ' })); container.appendChild(thr);
      return;
    }

    if (type === 'atr_pct') {
      const len = el('input', { type: 'number', class: 'param', 'data-key': 'length', min: 1, value: (rule && rule.length) ? String(rule.length) : '14' });
      const pct = el('input', { type: 'number', class: 'param', 'data-key': 'pct', step: 0.01, min: 0, value: (rule && rule.pct) ? String(rule.pct) : '1.0' });
      container.appendChild(el('label', { text: 'Length: ' })); container.appendChild(len);
      container.appendChild(el('label', { text: ' %: ' })); container.appendChild(pct);
      return;
    }

    // default: custom DSL
    const ta = el('textarea', { class: 'param', 'data-key': 'custom', rows: 2 });
    if (rule && rule.custom) ta.value = rule.custom;
    container.appendChild(ta);
  }

  function readParamsFromRow(row) {
    const params = {};
    const paramEls = Array.from(row.querySelectorAll('.param'));
    for (const p of paramEls) {
      const key = p.getAttribute('data-key');
      if (!key) continue;
      if (p.tagName === 'SELECT' || p.tagName === 'INPUT' || p.tagName === 'TEXTAREA') {
        const val = p.value;
        if (p.type === 'number') {
          const n = Number(val);
          params[key] = Number.isFinite(n) ? n : val;
        } else {
          params[key] = val;
        }
      }
    }
    return params;
  }

  function createRuleRow(section, rule) {
    const row = el('div', { class: 'card rule-row' });
    row.dataset.section = section;

    const left = el('div', { class: 'rule-left' });
    const right = el('div', { class: 'rule-right' });

    const sel = el('select', { class: 'rule-type' });
    const types = section === 'context' ? contextTypes : (section === 'signal' ? signalTypes : triggerTypes);
    for (const [v, label] of types) { const o = el('option', { value: v }); o.textContent = label; sel.appendChild(o); }
    if (rule && rule.type) sel.value = rule.type;

    left.appendChild(el('div', { class: 'rule-row-top' }));
    left.querySelector('.rule-row-top').appendChild(sel);

    if (section === 'signal' || section === 'trigger') {
      const tf = el('select', { class: 'rule-tf' }); tf.appendChild(el('option', { value: 'default', text: 'default' })); tf.appendChild(el('option', { value: '1m', text: '1m' })); tf.appendChild(el('option', { value: '5m', text: '5m' })); tf.appendChild(el('option', { value: '15m', text: '15m' })); if (rule && rule.tf) tf.value = rule.tf; left.querySelector('.rule-row-top').appendChild(tf);
    }

    const valid = el('input', { type: 'number', class: 'rule-valid', min: 1, placeholder: 'valid_for_bars' });
    if (rule && rule.valid_for_bars) valid.value = String(rule.valid_for_bars);
    left.appendChild(valid);

    // params container
    const params = el('div', { class: 'rule-params' });
    renderParamsForType(params, section, (rule && rule.type) ? rule.type : sel.value, rule || {});

    // wire type change to re-render params
    sel.addEventListener('change', () => {
      renderParamsForType(params, section, sel.value, {});
      // commit entire section to state after change
      commitSectionFromDOM(section);
    });

    // update on param changes
    params.addEventListener('input', () => commitSectionFromDOM(section));
    params.addEventListener('change', () => commitSectionFromDOM(section));

    // remove button
    const remove = el('button', { type: 'button', class: 'rule-remove' }); remove.textContent = 'Remove';
    remove.addEventListener('click', () => { row.remove(); commitSectionFromDOM(section); });

    right.appendChild(params);
    right.appendChild(remove);

    row.appendChild(left); row.appendChild(right);

    return row;
  }

  function renderRulesInto(container, rules, section) {
    if (!container) return;
    container.innerHTML = '';
    for (const r of (rules || [])) {
      const row = createRuleRow(section, r);
      container.appendChild(row);
    }
  }

  function serializeRulesFromDOM(container) {
    const out = [];
    if (!container) return out;
    const rows = Array.from(container.querySelectorAll('.rule-row'));
    for (const r of rows) {
      const type = (r.querySelector('.rule-type') || {}).value || '';
      if (!type) continue;
      const rule = { type };
      const tfEl = r.querySelector('.rule-tf'); if (tfEl && tfEl.value && tfEl.value !== 'default') rule.tf = tfEl.value;
      const validEl = r.querySelector('.rule-valid'); if (validEl && validEl.value !== '') { const n = Number(validEl.value); if (Number.isFinite(n)) rule.valid_for_bars = n; }
      const params = readParamsFromRow(r);
      for (const k of Object.keys(params)) {
        rule[k] = params[k];
      }
      out.push(rule);
    }
    return out;
  }

  function commitSectionFromDOM(section) {
    try {
      const container = document.getElementById(section + '-rules');
      if (!container || !window.GuidedBuilderV2State) return;
      const rules = serializeRulesFromDOM(container);
      window.GuidedBuilderV2State.set(section + '_rules', rules);
    } catch (e) { console.error('commitSectionFromDOM', e); }
  }

  function syncHiddenFromState(form, state) {
    try {
      const ctx = ensureHidden(form, 'context_rules_json');
      const sig = ensureHidden(form, 'signal_rules_json');
      const trg = ensureHidden(form, 'trigger_rules_json');
      ctx.value = JSON.stringify(state.context_rules || []);
      sig.value = JSON.stringify(state.signal_rules || []);
      trg.value = JSON.stringify(state.trigger_rules || []);
    } catch (e) { console.error('syncHiddenFromState', e); }
  }

  function init() {
    const form = document.querySelector('form[action="/create-strategy-guided/step4"]') || document.getElementById('guided_step4_form');
    if (!form) return;

    const ctxContainer = document.getElementById('context-rules');
    const sigContainer = document.getElementById('signal-rules');
    const trgContainer = document.getElementById('trigger-rules');

    ensureHidden(form, 'context_rules_json'); ensureHidden(form, 'signal_rules_json'); ensureHidden(form, 'trigger_rules_json');

    // initial load: prefer state if present, else hidden inputs
    const initialState = (window.GuidedBuilderV2State && typeof window.GuidedBuilderV2State.get === 'function') ? window.GuidedBuilderV2State.get() : null;
    const initial = initialState || {};

    // if state has no rules, read hidden inputs
    if (!initial || !initial.context_rules || !initial.signal_rules || !initial.trigger_rules) {
      initial.context_rules = parseJsonSafe(form.querySelector('#context_rules_json')?.value);
      initial.signal_rules = parseJsonSafe(form.querySelector('#signal_rules_json')?.value);
      initial.trigger_rules = parseJsonSafe(form.querySelector('#trigger_rules_json')?.value);
    }

    if (window.GuidedBuilderV2State && typeof window.GuidedBuilderV2State.load === 'function') {
      window.GuidedBuilderV2State.load(initial);
    }

    // subscribe to state updates and render + sync hidden
    if (window.GuidedBuilderV2State && typeof window.GuidedBuilderV2State.subscribe === 'function') {
      window.GuidedBuilderV2State.subscribe((s) => {
        try {
          renderRulesInto(ctxContainer, s.context_rules || [], 'context');
          renderRulesInto(sigContainer, s.signal_rules || [], 'signal');
          renderRulesInto(trgContainer, s.trigger_rules || [], 'trigger');
          syncHiddenFromState(form, s);
        } catch (e) { console.error('state subscribe render error', e); }
      });
    }

    // wire add buttons
    const addCtx = document.getElementById('add-context');
    const addSig = document.getElementById('add-signal');
    const addTrg = document.getElementById('add-trigger');
    if (addCtx && ctxContainer) addCtx.addEventListener('click', (ev) => { ev.preventDefault(); ctxContainer.appendChild(createRuleRow('context', {})); commitSectionFromDOM('context'); });
    if (addSig && sigContainer) addSig.addEventListener('click', (ev) => { ev.preventDefault(); sigContainer.appendChild(createRuleRow('signal', {})); commitSectionFromDOM('signal'); });
    if (addTrg && trgContainer) addTrg.addEventListener('click', (ev) => { ev.preventDefault(); trgContainer.appendChild(createRuleRow('trigger', {})); commitSectionFromDOM('trigger'); });

    // edits/removes: delegate to container and commit on change
    const delegateEvents = (container, section) => {
      if (!container) return;
      container.addEventListener('input', () => commitSectionFromDOM(section));
      container.addEventListener('change', () => commitSectionFromDOM(section));
      container.addEventListener('click', (ev) => { if (ev.target && ev.target.classList && ev.target.classList.contains('rule-remove')) commitSectionFromDOM(section); });
    };
    delegateEvents(ctxContainer, 'context'); delegateEvents(sigContainer, 'signal'); delegateEvents(trgContainer, 'trigger');

    // ensure hidden sync on submit
    form.addEventListener('submit', () => { if (window.GuidedBuilderV2State) { window.GuidedBuilderV2State.flush && window.GuidedBuilderV2State.flush(); const s = window.GuidedBuilderV2State.raw ? window.GuidedBuilderV2State.raw() : {}; syncHiddenFromState(form, s); } else { syncHidden(form, ctxContainer, sigContainer, trgContainer); } });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();

})();
/* Guided Builder v2 — Clean rewrite (Phase 4 starter)
   Self-contained module: reads the Step 4 form, renders simple rule rows,
   two-way syncs hidden JSON inputs: context_rules_json, signal_rules_json,
   trigger_rules_json. Keeps implementation minimal and robust for iterative
   expansion (per-type parameter blocks come next).
*/
(function () {
  const contextTypes = [
    ['price_vs_ma', 'Trend: Price vs MA'],
    ['ma_cross_state', 'Trend: MA state (fast vs slow)'],
    ['atr_pct', 'Volatility: ATR% filter'],
    ['structure_breakout_state', 'Structure: Breakout state'],
    ['ma_spread_pct', 'Trend strength: MA spread %'],
    ['custom', 'Custom (DSL)']
  ];

  const signalTypes = [
    ['ma_cross', 'MA cross (fast vs slow)'],
    ['rsi_threshold', 'RSI threshold'],
    ['rsi_cross_back', 'RSI cross back'],
    ['donchian_breakout', 'Donchian breakout/breakdown'],
    ['new_high_low_breakout', 'New high/low breakout'],
    ['pullback_to_ma', 'Pullback to MA'],
    ['custom', 'Custom (DSL)']
  ];

  const triggerTypes = [
    ['pin_bar', 'Pin bar'],
    ['inside_bar_breakout', 'Inside bar breakout'],
    ['engulfing', 'Engulfing'],
    ['ma_reclaim', 'MA reclaim'],
    ['prior_bar_break', 'Break prior bar high/low'],
    ['donchian_breakout', 'Donchian breakout/breakdown'],
    ['range_breakout', 'Range breakout'],
    ['wide_range_candle', 'Wide-range candle (vs ATR)'],
    ['custom', 'Custom (DSL)']
  ];

  function el(tag, attrs) {
    const node = document.createElement(tag);
    if (!attrs) return node;
    for (const k of Object.keys(attrs)) {
      if (k === 'text') node.textContent = attrs[k];
      else if (k === 'html') node.innerHTML = attrs[k];
      else if (k === 'class') node.className = attrs[k];
      else node.setAttribute(k, String(attrs[k]));
    }
    return node;
  }

  function ensureHidden(form, id) {
    let elHidden = form.querySelector('#' + id);
    if (!elHidden) {
      elHidden = el('input', { type: 'hidden', id: id, name: id });
      elHidden.value = '[]';
      form.appendChild(elHidden);
    }
    return elHidden;
  }

  function parseJsonSafe(txt) {
    try { const p = JSON.parse(String(txt || '[]')); return Array.isArray(p) ? p : []; } catch (e) { return []; }
  }

  function serializeRules(container) {
    const out = [];
    if (!container) return out;
    const rows = Array.from(container.querySelectorAll('.rule-row'));
    for (const r of rows) {
      const type = (r.querySelector('.rule-type') || {}).value || '';
      if (!type) continue;
      const rule = { type: String(type) };
      const tfEl = r.querySelector('.rule-tf');
      if (tfEl && tfEl.value && tfEl.value !== 'default') rule.tf = String(tfEl.value);
      const validEl = r.querySelector('.rule-valid');
      if (validEl && validEl.value !== '') {
        const n = Number(validEl.value);
        if (Number.isFinite(n)) rule.valid_for_bars = n;
      }
      const customEl = r.querySelector('.rule-custom');
      if (customEl && customEl.value && customEl.value.trim() !== '') rule.custom = customEl.value.trim();
      out.push(rule);
    }
    return out;
  }

  function syncHidden(form, ctxContainer, sigContainer, trgContainer) {
    try {
      const ctxHidden = ensureHidden(form, 'context_rules_json');
      const sigHidden = ensureHidden(form, 'signal_rules_json');
      const trgHidden = ensureHidden(form, 'trigger_rules_json');
      ctxHidden.value = JSON.stringify(serializeRules(ctxContainer));
      sigHidden.value = JSON.stringify(serializeRules(sigContainer));
      trgHidden.value = JSON.stringify(serializeRules(trgContainer));
    } catch (e) { console.error('syncHidden error', e); }
  }

  function createRuleRow(section, rule) {
    const row = el('div', { class: 'rule-row' });
    const left = el('div', { class: 'rule-left' });
    const right = el('div', { class: 'rule-right' });

    // type select
    const sel = el('select', { class: 'rule-type', name: `type` });
    const types = section === 'context' ? contextTypes : (section === 'signal' ? signalTypes : triggerTypes);
    for (const [v, label] of types) {
      const o = el('option', { value: v }); o.textContent = label; sel.appendChild(o);
    }
    if (rule && rule.type) sel.value = rule.type;

    left.appendChild(sel);

    // TF select for signal/trigger
    if (section === 'signal' || section === 'trigger') {
      const tf = el('select', { class: 'rule-tf', name: 'tf' });
      tf.appendChild(el('option', { value: 'default', text: 'default' }));
      tf.appendChild(el('option', { value: '1m', text: '1m' }));
      tf.appendChild(el('option', { value: '5m', text: '5m' }));
      tf.appendChild(el('option', { value: '15m', text: '15m' }));
      if (rule && rule.tf) tf.value = rule.tf;
      left.appendChild(tf);
    }

    // valid_for_bars
    const valid = el('input', { type: 'number', class: 'rule-valid', name: 'valid_for_bars', min: 1, value: (rule && rule.valid_for_bars) ? String(rule.valid_for_bars) : '' });
    left.appendChild(valid);

    // parameter area (simple custom DSL textarea for now)
    const ta = el('textarea', { class: 'rule-custom', name: 'custom', rows: 2 });
    if (rule && rule.custom) ta.value = rule.custom;
    right.appendChild(ta);

    // remove button
    const remove = el('button', { type: 'button', class: 'rule-remove' }); remove.textContent = 'Remove';
    right.appendChild(remove);

    row.appendChild(left); row.appendChild(right);

    return row;
  }

  function renderRules(container, rules, section) {
    if (!container) return;
    container.innerHTML = '';
    if (!Array.isArray(rules) || rules.length === 0) return;
    for (const r of rules) {
      const row = createRuleRow(section, r);
      container.appendChild(row);
    }
  }

  function wire(container, form, ctxContainer, sigContainer, trgContainer) {
    // delegate events
    container.addEventListener('input', () => syncHidden(form, ctxContainer, sigContainer, trgContainer));
    container.addEventListener('change', () => syncHidden(form, ctxContainer, sigContainer, trgContainer));
    container.addEventListener('click', (ev) => {
      if (!ev.target) return;
      if (ev.target.classList && ev.target.classList.contains('rule-remove')) {
        const row = ev.target.closest('.rule-row'); if (row) { row.remove(); syncHidden(form, ctxContainer, sigContainer, trgContainer); }
      }
    });
  }

  function init() {
    const form = document.querySelector('form[action="/create-strategy-guided/step4"]') || document.getElementById('guided_step4_form');
    if (!form) return;

    const ctxContainer = document.getElementById('context-rules');
    const sigContainer = document.getElementById('signal-rules');
    const trgContainer = document.getElementById('trigger-rules');

    const ctxHidden = ensureHidden(form, 'context_rules_json');
    const sigHidden = ensureHidden(form, 'signal_rules_json');
    const trgHidden = ensureHidden(form, 'trigger_rules_json');

    // load initial rules from hidden inputs
    const ctx = parseJsonSafe(ctxHidden.value);
    const sig = parseJsonSafe(sigHidden.value);
    const trg = parseJsonSafe(trgHidden.value);

    renderRules(ctxContainer, ctx, 'context');
    renderRules(sigContainer, sig, 'signal');
    renderRules(trgContainer, trg, 'trigger');

    // wire add buttons
    const addCtx = document.getElementById('add-context');
    const addSig = document.getElementById('add-signal');
    const addTrg = document.getElementById('add-trigger');

    if (addCtx && ctxContainer) addCtx.addEventListener('click', (ev) => { ev.preventDefault(); ctxContainer.appendChild(createRuleRow('context', {})); syncHidden(form, ctxContainer, sigContainer, trgContainer); });
    if (addSig && sigContainer) addSig.addEventListener('click', (ev) => { ev.preventDefault(); sigContainer.appendChild(createRuleRow('signal', {})); syncHidden(form, ctxContainer, sigContainer, trgContainer); });
    if (addTrg && trgContainer) addTrg.addEventListener('click', (ev) => { ev.preventDefault(); trgContainer.appendChild(createRuleRow('trigger', {})); syncHidden(form, ctxContainer, sigContainer, trgContainer); });

    // delegate wiring for edits/removes
    wire(ctxContainer || document, form, ctxContainer, sigContainer, trgContainer);
    wire(sigContainer || document, form, ctxContainer, sigContainer, trgContainer);
    wire(trgContainer || document, form, ctxContainer, sigContainer, trgContainer);

    // ensure we sync initially
    syncHidden(form, ctxContainer, sigContainer, trgContainer);

    // on submit, make sure hidden inputs reflect current state
    form.addEventListener('submit', () => syncHidden(form, ctxContainer, sigContainer, trgContainer));
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();

})();
/* Guided Builder v2 — Modular entrypoint (clean rewrite start)
   This file replaces the previous monolithic script. It bootstraps the smaller
   modules added in the rewrite: model, state, and DOM bindings. Full rule-row
   renderer will be implemented in Phase 4; this entrypoint keeps the page
   functional and free of syntax/runtime parse errors.
*/
(function () {
  function safeWarn(msg) { try { console.warn(msg); } catch (e) {} }

  // Ensure model and state objects exist (modules created in earlier phases)
  if (!window.GuidedBuilderV2Model) {
    safeWarn('GuidedBuilderV2Model missing; guided builder limited functionality');
    window.GuidedBuilderV2Model = {
      defaults: () => ({ _version: 1, context_rules: [], signal_rules: [], trigger_rules: [] })
    };
  }

  if (!window.GuidedBuilderV2State) {
    safeWarn('GuidedBuilderV2State missing; creating minimal fallback');
    window.GuidedBuilderV2State = (function () {
      let _raw = GuidedBuilderV2Model.defaults();
      const subs = [];
      return {
        load: (x) => { _raw = Object.assign({}, GuidedBuilderV2Model.defaults(), x || {}); subs.forEach(s=>s(_raw)); },
        get: (k) => _raw[k],
        set: (k,v) => { _raw[k]=v; subs.forEach(s=>s(_raw)); },
        raw: () => _raw,
        subscribe: (cb) => { subs.push(cb); return () => { const i=subs.indexOf(cb); if(i>=0) subs.splice(i,1); }; },
        flush: () => {},
      };
    })();
  }

  // If DOM binder exists, it will initialize itself on DOMContentLoaded.
  if (!window.GuidedBuilderV2DomInitialized) {
    // Minimal init that ensures hidden rule JSONs are present and consistent
    document.addEventListener('DOMContentLoaded', function () {
      try {
        const form = document.querySelector('form[action="/create-strategy-guided/step4"]') || document.getElementById('guided_step4_form');
        if (!form) return;

        // Ensure hidden inputs exist so server endpoints receive payloads
        const ensureHidden = (id) => {
          let el = form.querySelector('#' + id);
          if (!el) {
            el = document.createElement('input');
            el.type = 'hidden';
            el.id = id;
            el.name = id;
            el.value = '[]';
            form.appendChild(el);
          }
        };
        ensureHidden('context_rules_json');
        ensureHidden('signal_rules_json');
        ensureHidden('trigger_rules_json');

        // If the DOM bindings module is present it will subscribe and populate
        if (window.GuidedBuilderV2State && window.GuidedBuilderV2Model) {
          try {
            // attempt to load current form values into state via DOM binder
            if (window.GuidedBuilderV2Dom) {
              // Dom module exposes init() if present; otherwise it's self-initializing
              if (typeof window.GuidedBuilderV2Dom.init === 'function') window.GuidedBuilderV2Dom.init();
            }
          } catch (e) { console.error('guided_builder_v2 dom init error', e); }
        }

      } catch (e) { console.error('guided_builder_v2 init error', e); }
    });
    window.GuidedBuilderV2DomInitialized = true;
  }
})();
    (function () {
      const form = document.querySelector('form[action="/create-strategy-guided/step4"]');
      // Feature flags (toggle experimental features safely)
      const FEATURE_FLAG = {
        volume: false,
      };
      const body = document.getElementById('effective-preview-body');
      if (!form || !body) return;

      const contextVisualEl = document.getElementById('context-visual');
      const contextVisualMsgEl = document.getElementById('context-visual-message');

      const setupVisualEl = document.getElementById('setup-visual');
      const setupVisualMsgEl = document.getElementById('setup-visual-message');

      const ctxHidden = document.getElementById('context_rules_json');
      const sigHidden = document.getElementById('signal_rules_json');
      const trgHidden = document.getElementById('trigger_rules_json');
      const ctxContainer = document.getElementById('context-rules');
      const sigContainer = document.getElementById('signal-rules');
      const trgContainer = document.getElementById('trigger-rules');
      const primaryContextSel = document.getElementById('primary_context_type');
      const maStackBlock = document.getElementById('context_ma_stack_block');

      function val(name) {
        const el = form.querySelector(`[name="${name}"]`);
        if (!el) return null;
        if (el.type === 'checkbox') return !!el.checked;
        return el.value;
      }

      function num(name) {
        const v = val(name);
        if (v === null || v === '') return null;
        const n = Number(v);
        return Number.isFinite(n) ? n : null;
      }

      function escapeHtml(s) {
        return String(s)
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;')
          .replaceAll('"', '&quot;')
          .replaceAll("'", '&#39;');
      }

      function renderSide(title, side) {
        if (!side || !side.enabled) return '';
        let html = `<h4 style="margin-top:12px;">${escapeHtml(title)} (${escapeHtml(side.direction)})</h4>`;

        if (side.context && side.context.length) {
          const groups = new Map();
          for (const item of side.context) {
            const g = item.group || 'Context';
            if (!groups.has(g)) groups.set(g, []);
            groups.get(g).push(item);
          }

          html += `<div class="muted">Context</div>`;
          for (const [g, items] of groups.entries()) {
            html += `<div style="margin-top:6px;"><strong>${escapeHtml(g)}</strong></div>`;
            html += `<ul>`;
            for (const item of items) {
              html += `<li>${escapeHtml(item.label)}<br /><code style="display:block;white-space:pre-wrap;">${escapeHtml(item.expr)}</code></li>`;
            }
            html += `</ul>`;
          }
        } else {
          html += `<div class="muted">Context: none</div>`;
        }

        if (side.signals && side.signals.length) {
          html += `<div class="muted">Signals</div>`;
          html += `<ul>`;
          for (const item of side.signals) {
            html += `<li>${escapeHtml(item.label)}<br /><code style="display:block;white-space:pre-wrap;">${escapeHtml(item.expr)}</code></li>`;
          }
          html += `</ul>`;
        } else {
          html += `<div class="muted">Signals: none</div>`;
        }

        if (side.triggers && side.triggers.length) {
          html += `<div class="muted">Triggers</div>`;
          html += `<ul>`;
          for (const item of side.triggers) {
            html += `<li>${escapeHtml(item.label)}<br /><code style="display:block;white-space:pre-wrap;">${escapeHtml(item.expr)}</code></li>`;
          }
          html += `</ul>`;
        } else {
          html += `<div class="muted">Triggers: none</div>`;
        }
        return html;
      }

      function safeParseJson(s) {
        try {
          const v = JSON.parse(String(s || ''));
          return Array.isArray(v) ? v : [];
        } catch (e) {
          return [];
        }
      }

      function el(tag, attrs) {
        const node = document.createElement(tag);
        if (attrs) {
          for (const [k, v] of Object.entries(attrs)) {
            if (k === 'text') node.textContent = String(v);
            else if (k === 'html') node.innerHTML = String(v);
            else node.setAttribute(k, String(v));
          }
        }
        return node;
      }

      function ruleRow(section, rule) {
        const row = el('div', { class: 'card rule-row', 'data-section': section });

        const header = el('div', { style: 'display:flex; gap:10px; align-items:center; justify-content:space-between;' });
        const left = el('div');
        const right = el('div');

        const typeSel = el('select', { 'data-field': 'type' });

        const tfSel = el('select', { 'data-field': 'tf', style: 'margin-left:10px;' });
        function addTfOpt(value, label) {
          const o = el('option');
          o.value = value;
          o.textContent = label;
          tfSel.appendChild(o);
        }
        // Per-rule TF applies to signals/triggers only (context uses the single Context TF).
        if (section === 'signal' || section === 'trigger') {
          addTfOpt('default', 'TF: Default');
          addTfOpt('1m', 'TF: 1m');
          addTfOpt('5m', 'TF: 5m');
          addTfOpt('15m', 'TF: 15m');
          addTfOpt('1h', 'TF: 1h');
          addTfOpt('4h', 'TF: 4h');
          addTfOpt('1d', 'TF: 1d');
        }

        const validWrap = el('span', { style: 'display:inline-flex; gap:6px; align-items:center; margin-left:10px;' });
        const validLabel = el('span', { class: 'muted', text: 'Valid bars:' });
        const validInp = el('input', { type: 'number', min: '0', step: '1', 'data-field': 'valid_for_bars', style: 'width:90px;' });
        validInp.value = (rule && (rule.valid_for_bars != null)) ? String(rule.valid_for_bars) : '';
        validWrap.appendChild(validLabel);
        validWrap.appendChild(validInp);

        function addOpt(value, label) {
          const o = el('option');
          o.value = value;
          o.textContent = label;
          typeSel.appendChild(o);
        }

        const removeBtn = el('button', { type: 'button', class: 'danger', text: 'Remove' });
        removeBtn.addEventListener('click', () => {
          row.remove();
          syncHidden();
          schedule();
        });

        left.appendChild(typeSel);
        if (section === 'signal' || section === 'trigger') {
          left.appendChild(tfSel);
          left.appendChild(validWrap);
        }
        right.appendChild(removeBtn);
        header.appendChild(left);
        header.appendChild(right);
        row.appendChild(header);

        const fields = el('div', { style: 'margin-top:10px;' });

        function block(type, build) {
          const b = el('div', { 'data-type': type, style: 'display:none;' });
          build(b);
          fields.appendChild(b);
        }

        function inputNumber(parent, label, field, value, opts) {
          parent.appendChild(el('label', { text: label }));
          const inp = el('input');
          inp.type = 'number';
          inp.setAttribute('data-field', field);
          if (opts && typeof opts.min !== 'undefined') inp.min = String(opts.min);
          if (opts && typeof opts.step !== 'undefined') inp.step = String(opts.step);
          inp.value = value != null ? String(value) : '';
          parent.appendChild(inp);
        }

        function inputText(parent, label, field, value, placeholder) {
          parent.appendChild(el('label', { text: label }));
          const inp = el('input');
          inp.type = 'text';
          inp.setAttribute('data-field', field);
          if (placeholder) inp.placeholder = placeholder;
          inp.value = value != null ? String(value) : '';
          parent.appendChild(inp);
        }

        function inputTextArea(parent, label, field, value, placeholder) {
          parent.appendChild(el('label', { text: label }));
          const ta = el('textarea', { rows: '2' });
          ta.setAttribute('data-field', field);
          if (placeholder) ta.placeholder = placeholder;
          ta.value = value != null ? String(value) : '';
          parent.appendChild(ta);
        }

        function inputCheckbox(parent, label, field, checked) {
          const wrap = el('label');
          const inp = el('input');
          inp.type = 'checkbox';
          inp.setAttribute('data-field', field);
          inp.checked = !!checked;
          wrap.appendChild(inp);
          wrap.appendChild(el('span', { text: label }));
          parent.appendChild(wrap);
        }

        function select(parent, label, field, options, value) {
          parent.appendChild(el('label', { text: label }));
          const sel = el('select');
          sel.setAttribute('data-field', field);
          for (const opt of options) {
            const o = el('option');
            o.value = opt[0];
            o.textContent = opt[1];
            sel.appendChild(o);
          }
          if (value != null) sel.value = value;
          parent.appendChild(sel);
        }
          block('ma_cross_state', (b) => {
            select(b, 'MA type', 'ma_type', [['ema', 'EMA'], ['sma', 'SMA']], rule?.ma_type || 'ema');
            inputNumber(b, 'Fast length', 'fast', rule?.fast ?? 20, { min: 1 });
            inputNumber(b, 'Slow length', 'slow', rule?.slow ?? 50, { min: 1 });

            inputNumber(b, 'Min MA spread (fraction, optional)', 'min_spread_pct', rule?.min_spread_pct ?? 0, { step: 0.001, min: 0 });
            b.appendChild(el('div', { class: 'muted', html: 'Optional separation filter: require fast/slow distance ≥ threshold (directional).' }));

            inputCheckbox(b, 'Require MA slope filter (directional)', 'slope_enabled', rule?.slope_enabled);
            select(b, 'Slope on', 'slope_on', [['slow', 'Slow MA'], ['fast', 'Fast MA']], rule?.slope_on || 'slow');
            inputNumber(b, 'Slope lookback bars', 'slope_lookback', rule?.slope_lookback ?? 10, { min: 1 });
          });
          block('atr_pct', (b) => {
            inputNumber(b, 'ATR length', 'atr_len', rule?.atr_len ?? 14, { min: 1 });
            select(b, 'Operator', 'op', [['>', '>'], ['<', '<'], ['>=', '>='], ['<=', '<=']], rule?.op || '>');
            inputNumber(b, 'Threshold (fraction)', 'threshold', rule?.threshold ?? 0.005, { step: 0.0001, min: 0 });
          });
          block('structure_breakout_state', (b) => {
            inputNumber(b, 'Lookback length', 'length', rule?.length ?? 20, { min: 1 });
          });
          block('ma_spread_pct', (b) => {
            select(b, 'MA type', 'ma_type', [['ema', 'EMA'], ['sma', 'SMA']], rule?.ma_type || 'ema');
            inputNumber(b, 'Fast length', 'fast', rule?.fast ?? 20, { min: 1 });
            inputNumber(b, 'Slow length', 'slow', rule?.slow ?? 200, { min: 1 });
            inputNumber(b, 'Min spread (fraction)', 'threshold', rule?.threshold ?? 0.01, { step: 0.0001, min: 0 });

            inputCheckbox(b, 'Require MA slope filter (directional)', 'slope_enabled', rule?.slope_enabled);
            select(b, 'Slope on', 'slope_on', [['slow', 'Slow MA'], ['fast', 'Fast MA']], rule?.slope_on || 'slow');
            inputNumber(b, 'Slope lookback bars', 'slope_lookback', rule?.slope_lookback ?? 10, { min: 1 });
          });
          block('custom', (b) => {
            b.appendChild(el('div', { class: 'muted', text: 'Bull/bear expressions. If you omit @tf, the builder will apply the Context TF automatically.' }));
            inputTextArea(b, 'Bull expr', 'bull_expr', rule?.bull_expr || '', 'e.g. close > sma(close, 200)');
            inputTextArea(b, 'Bear expr', 'bear_expr', rule?.bear_expr || '', 'e.g. close < sma(close, 200)');
          });

        // Signal blocks
        if (section === 'signal') {
          block('ma_cross', (b) => {
            select(b, 'MA type', 'ma_type', [['ema', 'EMA'], ['sma', 'SMA']], rule?.ma_type || 'ema');
            inputNumber(b, 'Fast length', 'fast', rule?.fast ?? 20, { min: 1 });
            inputNumber(b, 'Slow length', 'slow', rule?.slow ?? 50, { min: 1 });
            b.appendChild(el('div', { class: 'muted', text: 'Bull = crosses above; Bear = crosses below (auto by side).' }));
          });
          block('rsi_threshold', (b) => {
            inputNumber(b, 'RSI length', 'length', rule?.length ?? 14, { min: 1 });
            inputNumber(b, 'Bull level (RSI < level)', 'bull_level', rule?.bull_level ?? 30, { step: 0.1 });
            inputNumber(b, 'Bear level (RSI > level)', 'bear_level', rule?.bear_level ?? 70, { step: 0.1 });
          });
          block('rsi_cross_back', (b) => {
            inputNumber(b, 'RSI length', 'length', rule?.length ?? 14, { min: 1 });
            inputNumber(b, 'Bull cross above', 'bull_level', rule?.bull_level ?? 30, { step: 0.1 });
            inputNumber(b, 'Bear cross below', 'bear_level', rule?.bear_level ?? 70, { step: 0.1 });
          });
          block('donchian_breakout', (b) => {
            inputNumber(b, 'Length', 'length', rule?.length ?? 20, { min: 1 });
          });
          block('new_high_low_breakout', (b) => {
            inputNumber(b, 'Length', 'length', rule?.length ?? 20, { min: 1 });
          });
          block('pullback_to_ma', (b) => {
            select(b, 'MA type', 'ma_type', [['ema', 'EMA'], ['sma', 'SMA']], rule?.ma_type || 'ema');
            inputNumber(b, 'MA length', 'length', rule?.length ?? 20, { min: 1 });
          });
          block('custom', (b) => {
            b.appendChild(el('div', { class: 'muted', text: 'Bull/bear expressions. If you omit @tf, the builder will apply the Entry TF automatically.' }));
            inputTextArea(b, 'Bull expr', 'bull_expr', rule?.bull_expr || '', 'e.g. rsi(close, 14) < 30');
            inputTextArea(b, 'Bear expr', 'bear_expr', rule?.bear_expr || '', 'e.g. rsi(close, 14) > 70');
          });
        }

        // Trigger blocks
        if (section === 'trigger') {
          block('pin_bar', (b) => {
            inputNumber(b, 'Wick/body ratio', 'pin_wick_body', rule?.pin_wick_body ?? 2.0, { step: 0.1, min: 0.1 });
            inputNumber(b, 'Opposite wick/body max', 'pin_opp_wick_body_max', rule?.pin_opp_wick_body_max ?? 1.0, { step: 0.1, min: 0 });
            inputNumber(b, 'Min body % of range', 'pin_min_body_pct', rule?.pin_min_body_pct ?? 0.2, { step: 0.05, min: 0, max: 1 });
          });
          block('ma_reclaim', (b) => {
            select(b, 'MA type', 'ma_type', [['ema', 'EMA'], ['sma', 'SMA']], rule?.ma_type || 'ema');
            inputNumber(b, 'MA length', 'length', rule?.length ?? 20, { min: 1 });
          });
          block('donchian_breakout', (b) => {
            inputNumber(b, 'Length', 'length', rule?.length ?? 20, { min: 1 });
          });
          block('range_breakout', (b) => {
            inputNumber(b, 'Length', 'length', rule?.length ?? 20, { min: 1 });
          });
          block('wide_range_candle', (b) => {
            inputNumber(b, 'ATR length', 'atr_len', rule?.atr_len ?? 14, { min: 1 });
            inputNumber(b, 'Multiplier', 'mult', rule?.mult ?? 2.0, { step: 0.1, min: 0.1 });
          });
          block('custom', (b) => {
            b.appendChild(el('div', { class: 'muted', text: 'Bull/bear expressions. If you omit @tf, the builder will apply the Entry TF automatically.' }));
            inputTextArea(b, 'Bull expr', 'bull_expr', rule?.bull_expr || '', 'e.g. close > shift(high, 1)');
            inputTextArea(b, 'Bear expr', 'bear_expr', rule?.bear_expr || '', 'e.g. close < shift(low, 1)');
          });

          // No-params types still need a block for visibility switching.
          block('inside_bar_breakout', (b) => {
            b.appendChild(el('div', { class: 'muted', text: 'No extra parameters.' }));
          });
          block('engulfing', (b) => {
            b.appendChild(el('div', { class: 'muted', text: 'No extra parameters.' }));
          });
          block('prior_bar_break', (b) => {
            b.appendChild(el('div', { class: 'muted', text: 'No extra parameters.' }));
          });
        }

        row.appendChild(fields);

        if (section === 'context') {
          addOpt('price_vs_ma', 'Trend: Price vs MA');
          addOpt('ma_cross_state', 'Trend: MA state (fast vs slow)');
          addOpt('atr_pct', 'Volatility: ATR% filter');
          addOpt('structure_breakout_state', 'Structure: Breakout state (close vs prior high/low)');
          addOpt('ma_spread_pct', 'Trend strength: MA spread %');
          // addOpt('relative_volume', 'Relative Volume (RVOL)');
          // addOpt('volume_osc_increase', 'Volume Oscillator Increase');
          // addOpt('volume_above_ma', 'Volume Above MA');
          addOpt('custom', 'Custom (DSL)');
        } else if (section === 'signal') {
          addOpt('ma_cross', 'MA cross (fast vs slow)');
          addOpt('rsi_threshold', 'RSI threshold');
          addOpt('rsi_cross_back', 'RSI cross back');
          addOpt('donchian_breakout', 'Donchian breakout/breakdown');
          addOpt('new_high_low_breakout', 'New high/low breakout');
          addOpt('pullback_to_ma', 'Pullback to MA');
          // addOpt('relative_volume', 'Relative Volume (RVOL)');
          // addOpt('volume_osc_increase', 'Volume Oscillator Increase');
          // addOpt('volume_above_ma', 'Volume Above MA');
          addOpt('custom', 'Custom (DSL)');
        } else {
          addOpt('pin_bar', 'Pin bar');
          addOpt('inside_bar_breakout', 'Inside bar breakout');
          addOpt('engulfing', 'Engulfing');
          addOpt('ma_reclaim', 'MA reclaim');
          addOpt('prior_bar_break', 'Break prior bar high/low');
          addOpt('donchian_breakout', 'Donchian breakout/breakdown');
          addOpt('range_breakout', 'Range breakout (highest/lowest)');
          addOpt('wide_range_candle', 'Wide-range candle (vs ATR)');
          addOpt('custom', 'Custom (DSL)');
        }

        // Volume rule blocks (Context & Signal)
        // if (section === 'context' || section === 'signal') {
        //   block('relative_volume', (b) => {
        //     select(b, 'MA type', 'ma_type', [['sma', 'SMA'], ['ema', 'EMA']], (rule && rule.ma_type != null) ? rule.ma_type : 'sma');
        //     inputNumber(b, 'MA length', 'length', (rule && rule.length != null) ? rule.length : 20, { min: 1 });
        //     select(b, 'Operator', 'op', [['>=', '≥'], ['<=', '≤'], ['>', '>'], ['<', '<']], (rule && rule.op != null) ? rule.op : '>=');
        //     inputNumber(b, 'Threshold', 'threshold', (rule && rule.threshold != null) ? rule.threshold : 1.5, { step: 0.01, min: 0 });
        //     b.appendChild(el('div', { class: 'muted', html: 'RVOL = volume / MA(volume, length). E.g. RVOL ≥ 1.5 means volume is 50% above normal.' }));
        //   });
        //   block('volume_osc_increase', (b) => {
        //     inputNumber(b, 'Fast length', 'fast', (rule && rule.fast != null) ? rule.fast : 12, { min: 1 });
        //     inputNumber(b, 'Slow length', 'slow', (rule && rule.slow != null) ? rule.slow : 26, { min: 1 });
        //     inputNumber(b, 'Min % increase', 'min_pct', (rule && rule.min_pct != null) ? rule.min_pct : 0.1, { step: 0.01, min: 0 });
        //     inputNumber(b, 'Lookback bars (N)', 'lookback', (rule && rule.lookback != null) ? rule.lookback : 3, { min: 1 });
        //     b.appendChild(el('div', { class: 'muted', html: 'True if (fast - slow) / slow increases by at least min % over N bars.' }));
        //   });
        //   block('volume_above_ma', (b) => {
        //     select(b, 'MA type', 'ma_type', [['ema', 'EMA'], ['sma', 'SMA']], (rule && rule.ma_type != null) ? rule.ma_type : 'ema');
        //     inputNumber(b, 'MA length', 'length', (rule && rule.length != null) ? rule.length : 20, { min: 1 });
        //     inputNumber(b, 'Min % above MA', 'min_pct', (rule && rule.min_pct != null) ? rule.min_pct : 0.1, { step: 0.01, min: 0 });
        //     b.appendChild(el('div', { class: 'muted', html: 'True if volume is at least min % above the MA.' }));
        //   });
        // }

        // Initialize row UI (set selected type, show param block, wire inputs)
        (function initRow() {
          try {
            // prefer explicitly provided rule type, otherwise use the first option
            const initialType = (rule && rule.type) ? String(rule.type) : (typeSel.options && typeSel.options.length ? typeSel.options[0].value : '');
            if (initialType) typeSel.value = initialType;
          } catch (e) {}

          try {
            if ((section === 'signal' || section === 'trigger') && tfSel) {
              tfSel.value = (rule && rule.tf) ? String(rule.tf) : (tfSel.querySelector('option[value="default"]') ? 'default' : (tfSel.options[0] ? tfSel.options[0].value : ''));
            }
          } catch (e) {}

          function updateTypeVisibility() {
            const sel = String(typeSel.value || '');
            for (const b of fields.querySelectorAll('[data-type]')) {
              b.style.display = (b.getAttribute('data-type') === sel) ? '' : 'none';
            }
          }

          typeSel.addEventListener('change', () => {
            updateTypeVisibility();
            try { syncHidden(); } catch (e) {}
            try { schedule(); } catch (e) {}
          });

          // wire inner controls to update preview/state
          for (const inp of fields.querySelectorAll('[data-field]')) {
            inp.addEventListener('input', () => { try { syncHidden(); schedule(); } catch (e) {} });
            inp.addEventListener('change', () => { try { syncHidden(); schedule(); } catch (e) {} });
          }

          updateTypeVisibility();
        })();

        return row;
      }

      const serialize = (container) => {
        const rules = [];
        if (!container) return rules;
        const rows = container.querySelectorAll('.rule-row');
        for (const row of rows) {
          const rule = {};
          const typeEl = row.querySelector('[data-field="type"]');
          if (!typeEl) continue;
          rule.type = String(typeEl.value || '').trim();

          const tfEl = row.querySelector('[data-field="tf"]');
          if (tfEl && String(tfEl.value || '') !== 'default') rule.tf = String(tfEl.value || '');

          const validEl = row.querySelector('[data-field="valid_for_bars"]');
          if (validEl) {
            const v = String(validEl.value || '').trim();
            const n = Number(v);
            if (v !== '' && Number.isFinite(n)) rule.valid_for_bars = n;
          }

          const block = row.querySelector(`[data-type="${rule.type}"]`);
          if (block) {
            const inputs = block.querySelectorAll('[data-field]');
            for (const inp of inputs) {
              const key = inp.getAttribute('data-field');
              if (!key) continue;
              const raw = inp.value != null ? String(inp.value).trim() : '';
              if (raw === '') continue;
              const num = Number(raw);
              rule[key] = (Number.isFinite(num) && raw === String(num)) ? num : raw;
            }
          }

          rules.push(rule);
        }
        return rules;
      }

      function syncHidden() {
        if (ctxHidden) ctxHidden.value = JSON.stringify(serialize(ctxContainer));
        if (sigHidden) sigHidden.value = JSON.stringify(serialize(sigContainer));
        if (trgHidden) trgHidden.value = JSON.stringify(serialize(trgContainer));
      }

      function loadInitial() {
        const ctx = safeParseJson(ctxHidden?.value);
        const sig = safeParseJson(sigHidden?.value);
        const trg = safeParseJson(trgHidden?.value);

        if (ctxContainer) {
          ctxContainer.innerHTML = '';
          for (const r of ctx) ctxContainer.appendChild(ruleRow('context', r));
        }
        if (sigContainer) {
          sigContainer.innerHTML = '';
          for (const r of sig) sigContainer.appendChild(ruleRow('signal', r));
        }
        if (trgContainer) {
          trgContainer.innerHTML = '';
          for (const r of trg) trgContainer.appendChild(ruleRow('trigger', r));
        }
        syncHidden();
      }

      function defaultPrimaryContextRule(type) {
        const t = String(type || '').trim();
        if (t === 'price_vs_ma') return { type: 'price_vs_ma', ma_type: 'ema', length: 200 };
        if (t === 'ma_cross_state') return { type: 'ma_cross_state', ma_type: 'ema', fast: 20, slow: 50 };
        if (t === 'atr_pct') return { type: 'atr_pct', atr_len: 14, op: '>', threshold: 0.005 };
        if (t === 'structure_breakout_state') return { type: 'structure_breakout_state', length: 20 };
        if (t === 'ma_spread_pct') return { type: 'ma_spread_pct', ma_type: 'ema', fast: 20, slow: 200, threshold: 0.01 };
        if (t === 'custom') return { type: 'custom', bull_expr: '', bear_expr: '' };
        return { type: t || 'price_vs_ma' };
      }

      function applyPrimaryContextType() {
        const t = String(primaryContextSel?.value || 'ma_stack');
        if (maStackBlock) {
          maStackBlock.style.display = (t === 'ma_stack') ? '' : 'none';
        }

        // For non-MA-stack primary contexts, seed (or retarget) the first context rule.
        if (t !== 'ma_stack' && t !== 'none') {
          if (ctxContainer) {
            const firstRow = ctxContainer.querySelector('.rule-row');
            if (!firstRow) {
              ctxContainer.prepend(ruleRow('context', defaultPrimaryContextRule(t)));
            } else {
              const typeEl = firstRow.querySelector('[data-field="type"]');
              if (typeEl && typeEl.value !== t) {
                typeEl.value = t;
                typeEl.dispatchEvent(new Event('change', { bubbles: true }));
              }
            }
          }
        }

        syncHidden();
      }

      const addContextBtn = document.getElementById('add-context');
      if (addContextBtn) {
        addContextBtn.addEventListener('click', () => {
          if (ctxContainer) ctxContainer.appendChild(ruleRow('context', { type: 'price_vs_ma' }));
          syncHidden();
          schedule();
        });
      }
      const addSignalBtn = document.getElementById('add-signal');
      if (addSignalBtn) {
        addSignalBtn.addEventListener('click', () => {
          if (sigContainer) sigContainer.appendChild(ruleRow('signal', { type: 'rsi_threshold' }));
          syncHidden();
          schedule();
        });
      }
      const addTriggerBtn = document.getElementById('add-trigger');
      if (addTriggerBtn) {
        addTriggerBtn.addEventListener('click', () => {
          if (trgContainer) trgContainer.appendChild(ruleRow('trigger', { type: 'prior_bar_break' }));
          syncHidden();
          schedule();
        });
      }

      function updateTriggerParamVisibility() {
        const t = String(val('trigger_type') || 'pin_bar');
        const show = (id, on) => {
          const el = document.getElementById(id);
          if (!el) return;
          el.style.display = on ? '' : 'none';
        };
        show('trigger_pin_bar', t === 'pin_bar');
        show('trigger_ma_reclaim', t === 'ma_reclaim');
        show('trigger_donchian', t === 'donchian_breakout');
        show('trigger_range', t === 'range_breakout');
        show('trigger_wide_range', t === 'wide_range_candle');
        show('trigger_custom', t === 'custom');
      }

      let ctxVisTimer = null;
      async function updateContextVisual(payload, opts) {
        if (!contextVisualEl || !contextVisualMsgEl) return;
        if (!window.Plotly) {
          contextVisualMsgEl.textContent = 'Plotly is not available; context preview disabled.';
          return;
        }

        const force = !!(opts && opts.force);

        // Debounce to avoid spamming the server on every keystroke.
        if (ctxVisTimer) window.clearTimeout(ctxVisTimer);
        const delayMs = force ? 0 : 350;
        ctxVisTimer = window.setTimeout(async () => {
          try {
            contextVisualMsgEl.textContent = 'Loading context preview…';
            const res = await fetch('/api/guided/builder_v2/context_visual', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              cache: 'no-store',
              body: JSON.stringify(payload)
            });
            const out = await res.json();
            if (!out || !out.ok) {
              contextVisualMsgEl.textContent = (out && out.message) ? String(out.message) : 'Context preview unavailable.';
              if (contextVisualEl) contextVisualEl.innerHTML = '';
              return;
            }
            contextVisualMsgEl.textContent = out.message ? String(out.message) : '';
            if (out.fig && out.fig.data && out.fig.layout) {
              await window.Plotly.react(contextVisualEl, out.fig.data, out.fig.layout, {
                displayModeBar: false,
                responsive: true,
              });
            }
          } catch (e) {
            contextVisualMsgEl.textContent = 'Context preview unavailable.';
            if (contextVisualEl) contextVisualEl.innerHTML = '';
          }
        }, delayMs);
      }

      let setupVisTimer = null;
      async function updateSetupVisual(payload, opts) {
        if (!setupVisualEl || !setupVisualMsgEl) return;
        if (!window.Plotly) {
          setupVisualMsgEl.textContent = 'Plotly is not available; setup preview disabled.';
          return;
        }

        const force = !!(opts && opts.force);

        if (setupVisTimer) window.clearTimeout(setupVisTimer);
        const delayMs = force ? 0 : 450;
        setupVisTimer = window.setTimeout(async () => {
          try {
            setupVisualMsgEl.textContent = 'Loading setup preview…';
            const res = await fetch('/api/guided/builder_v2/setup_visual', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              cache: 'no-store',
              body: JSON.stringify(payload)
            });
            const out = await res.json();
            if (!out || !out.ok) {
              setupVisualMsgEl.textContent = (out && out.message) ? String(out.message) : 'Setup preview unavailable.';
              if (setupVisualEl) setupVisualEl.innerHTML = '';
              return;
            }
            setupVisualMsgEl.textContent = out.message ? String(out.message) : '';
            if (out.fig && out.fig.data && out.fig.layout) {
              await window.Plotly.react(setupVisualEl, out.fig.data, out.fig.layout, {
                displayModeBar: false,
                responsive: true,
              });
            }
          } catch (e) {
            setupVisualMsgEl.textContent = 'Setup preview unavailable.';
            if (setupVisualEl) setupVisualEl.innerHTML = '';
          }
        }, delayMs);
      }

      function buildPayload() {
        syncHidden();
        updateTriggerParamVisibility();
        applyPrimaryContextType();
        return {
          entry_tf: val('entry_tf') || '1h',
          context_tf: val('context_tf') || '',
          signal_tf: val('signal_tf') || '',
          trigger_tf: val('trigger_tf') || '',
          primary_context_type: val('primary_context_type') || 'ma_stack',
          align_with_context: !!val('align_with_context'),
          long_enabled: !!val('long_enabled'),
          short_enabled: !!val('short_enabled'),
          ma_type: val('ma_type') || 'ema',
          ma_fast: num('ma_fast'),
          ma_mid: num('ma_mid'),
          ma_slow: num('ma_slow'),
          stack_mode: val('stack_mode') || 'none',
          slope_mode: val('slope_mode') || 'none',
          slope_lookback: num('slope_lookback'),
          min_ma_dist_pct: num('min_ma_dist_pct'),
          trigger_type: val('trigger_type') || 'pin_bar',
          trigger_valid_for_bars: num('trigger_valid_for_bars'),
          pin_wick_body: num('pin_wick_body'),
          pin_opp_wick_body_max: num('pin_opp_wick_body_max'),
          pin_min_body_pct: num('pin_min_body_pct'),
          trigger_ma_type: val('trigger_ma_type') || 'ema',
          trigger_ma_len: num('trigger_ma_len'),
          trigger_don_len: num('trigger_don_len'),
          trigger_range_len: num('trigger_range_len'),
          trigger_atr_len: num('trigger_atr_len'),
          trigger_atr_mult: num('trigger_atr_mult'),
          trigger_custom_bull_expr: val('trigger_custom_bull_expr') || '',
          trigger_custom_bear_expr: val('trigger_custom_bear_expr') || '',
          context_rules: safeParseJson(ctxHidden?.value),
          signal_rules: safeParseJson(sigHidden?.value),
          trigger_rules: safeParseJson(trgHidden?.value)
        };
      }

      async function updatePreview() {
        const payload = buildPayload();

        // Update the context chart in parallel with the textual preview.
        updateContextVisual(payload);
        updateSetupVisual(payload);

        try {
          const res = await fetch('/api/guided/builder_v2/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            cache: 'no-store',
            body: JSON.stringify(payload)
          });
          if (!res.ok) throw new Error('preview request failed');
          const preview = await res.json();

          let html = `<div class="muted" style="margin-bottom:10px;">Context TF: <strong>${escapeHtml(preview.context_tf)}</strong>. Entry TF: <strong>${escapeHtml(preview.entry_tf)}</strong>. Signals TF: <strong>${escapeHtml(preview.signal_tf)}</strong>. Triggers TF: <strong>${escapeHtml(preview.trigger_tf)}</strong>.</div>`;
          if (preview.aligned_dual) {
            html += `<div class="muted" style="margin-bottom:10px;">Aligned mode: LONG=bull, SHORT=bear</div>`;
          }
          html += renderSide('LONG', preview.long);
          html += renderSide('SHORT', preview.short);
          body.innerHTML = html;
        } catch (e) {
          body.innerHTML = `<div class="muted">Preview unavailable.</div>`;
        }
      }

      let t = null;
      function schedule() {
        if (t) window.clearTimeout(t);
        t = window.setTimeout(updatePreview, 150);
      }

      form.addEventListener('input', schedule);
      form.addEventListener('change', schedule);
      if (primaryContextSel) {
        primaryContextSel.addEventListener('change', () => {
          applyPrimaryContextType();
          schedule();
        });
      }
      form.addEventListener('submit', () => {
        syncHidden();
      });

      const reloadCtxBtn = document.getElementById('reload-context-visual');
      if (reloadCtxBtn) {
        reloadCtxBtn.addEventListener('click', () => {
          const payload = buildPayload();
          updateContextVisual(payload, { force: true });
        });
      }

      const reloadSetupBtn = document.getElementById('reload-setup-visual');
      if (reloadSetupBtn) {
        reloadSetupBtn.addEventListener('click', () => {
          const payload = buildPayload();
          updateSetupVisual(payload, { force: true });
        });
      }

      // If the user navigates back to this page, some browsers restore from bfcache.
      // Force-refresh the previews/charts on pageshow.
      window.addEventListener('pageshow', () => {
        updatePreview();
      });

      loadInitial();
      applyPrimaryContextType();
      updatePreview();
    })();
  