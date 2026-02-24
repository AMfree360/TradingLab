/* Guided Builder v2 — State-integrated rule renderer (Phase 4)
   - Renders rule rows with per-type parameter blocks
   - Two-way sync: DOM <-> `GuidedBuilderV2State` and hidden JSON inputs
   - Keeps volume features behind FEATURE_FLAG.volume (not rendered)
*/
(function () {
  // If a richer/full renderer is preferred (e.g. reopening a spec with
  // detailed `x_guided_ui`), skip this lightweight/clean-rewrite module to
  // avoid it overwriting per-type controls. The full Phase-4 renderer
  // appears earlier in the file and will run instead.
  try {
    if (window && window.guidedBuilderV2PreferFullRenderer) return;
  } catch (e) {}
  const FEATURE_FLAG = { volume: true };
  // Runtime debug gate: set `window.__GUIDED_V2_DEBUG = true` in the console
  // or via test harness to enable debug persistence (localStorage / window globals).
  const GUIDED_V2_DEBUG = (typeof window !== 'undefined' && window.__GUIDED_V2_DEBUG === true) || (typeof process !== 'undefined' && process && process.env && process.env.GUIDED_V2_DEBUG === '1');
  const GUIDED_V2_DEBUG_LOG_DIR = (typeof window !== 'undefined' && window.__GUIDED_V2_DEBUG_LOG_DIR) || null;
  try { if (GUIDED_V2_DEBUG) { console.log('[guided-debug] GUIDED_V2_DEBUG enabled; server debug log dir:', GUIDED_V2_DEBUG_LOG_DIR); } } catch (e) {}
  try { console.log('[guided-debug] guided_builder_v2.js loaded (workspace edition)'); } catch (e) {}

  // Global ensure-submit helper: synchronous DOM/state -> hidden sync.
  try {
    window.__guided_v2_ensure_submit = function (form) {
      try {
        if (!form) return true;
        // flush state then sync hidden from state
        try { if (window.GuidedBuilderV2State) { window.GuidedBuilderV2State.flush && window.GuidedBuilderV2State.flush(); syncHiddenFromState(form, window.GuidedBuilderV2State.raw ? window.GuidedBuilderV2State.raw() : {}); } } catch (e) {}
        // fallback: sync from DOM containers
        try { syncHidden(form, document.getElementById('context-rules'), document.getElementById('signal-rules'), document.getElementById('trigger-rules')); } catch (e) {}
      } catch (e) { /* best-effort */ }
      return true;
    };
  } catch (e) { /* ignore */ }

  const contextTypes = [
    ['price_vs_ma', 'Trend: Price vs MA'],
    ['ma_cross_state', 'Trend: MA state (fast vs slow)'],
    ['atr_pct', 'Volatility: ATR% filter'],
    ['atr_spike', 'Volatility: ATR spike'],
    ['structure_breakout_state', 'Structure: Breakout state'],
    ['ma_spread_pct', 'Trend strength: MA spread %'],
    ['bollinger_context', 'Bollinger (anchor TF)'],
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

  // Instrumented write: stamp hidden inputs with source and timestamp for debugging
  function writeHiddenStamped(form, id, value, source) {
    try {
      // If we're in the middle of a final submit, ignore non-final writers
      try {
        if (window.__guided_v2_in_submit && (source || '') !== 'final-pre-submit') {
          try { console.log('[guided-debug] hidden-write SKIP during submit', id, source); } catch (e) {}
          return;
        }
      } catch (ee) {}
      const elH = ensureHidden(form, id);
      const stamp = (new Date()).toISOString();
      // Always write the hidden input value (functional). Only perform
      // debug persistence (window globals / localStorage / extra logging)
      // when runtime debug is explicitly enabled.
      elH.value = value;
      if (!GUIDED_V2_DEBUG) return;
      // capture a trimmed stack to help identify the caller path
      let stack = null;
      try {
        const e = new Error();
        if (e && e.stack) {
          // split lines and drop the first line which is the Error message
          const lines = e.stack.split('\n').slice(1).map(l => l.trim());
          // keep only the top 6 stack frames for brevity
          stack = lines.slice(0, 6).join(' | ');
        }
      } catch (ee) { stack = null; }
      try { console.log('[guided-debug] hidden-write', stamp, id, source || 'unknown', value, stack ? ('stack=' + stack) : ''); } catch (e) {}
      try {
        const rec = { id: id, ts: stamp, source: source || 'unknown', value: value, stack: stack };
        window.__guided_v2_last_write = rec;
        try {
          if (!Array.isArray(window.__guided_v2_write_history)) window.__guided_v2_write_history = [];
          window.__guided_v2_write_history.push(rec);
          if (window.__guided_v2_write_history.length > 200) window.__guided_v2_write_history.shift();
        } catch (ee) {}
        try {
          // Persist debug write history so it survives navigation (useful for test harnesses)
          try {
            if (typeof localStorage !== 'undefined') {
              try { localStorage.setItem('__guided_v2_write_history', JSON.stringify(window.__guided_v2_write_history || [])); } catch (e) {}
              try { localStorage.setItem('__guided_v2_last_write', JSON.stringify(window.__guided_v2_last_write || null)); } catch (e) {}
            }
          } catch (ee) {}
        } catch (ee) {}
      } catch (e) {}
    } catch (e) { /* ignore */ }
  }

  function parseJsonSafe(txt) {
    try { const p = JSON.parse(String(txt || '[]')); return Array.isArray(p) ? p : []; } catch (e) { return []; }
  }

  // Wait until the next animation frames have run (double rAF) to give
  // the renderer a chance to apply DOM updates. Falls back to setTimeout
  // when rAF isn't available.
  function afterNextFrame(fn) {
    try {
      if (typeof requestAnimationFrame !== 'function') return setTimeout(fn, 0);
      requestAnimationFrame(() => {
        try { requestAnimationFrame(fn); } catch (e) { setTimeout(fn, 0); }
      });
    } catch (e) { try { setTimeout(fn, 0); } catch (ee) {} }
  }

  // Force a synchronous final-pre-submit write of the three rule hidden inputs.
  // Exposed globally as a small helper so render/subscribe code can call it
  // when a submit is pending to deterministically persist the latest DOM.
  function invokeFinalPreSubmitWrites(f) {
    try {
      if (!f) return;
      const ctxContainer = f.querySelector ? f.querySelector('#context-rules') : document.getElementById('context-rules');
      const sigContainer = f.querySelector ? f.querySelector('#signal-rules') : document.getElementById('signal-rules');
      const trgContainer = f.querySelector ? f.querySelector('#trigger-rules') : document.getElementById('trigger-rules');
      // Snapshot the DOM containers (innerHTML + visible field values) to localStorage
      try {
        function snapshotContainer(container) {
          if (!container) return null;
          const fields = [];
          try {
            const els = Array.from(container.querySelectorAll('select,input,textarea,[data-field],[data-key]'));
            for (const e of els) {
              const df = e.getAttribute('data-field') || e.getAttribute('data-key') || null;
              let val = null;
              try {
                if (e.type === 'checkbox') val = !!e.checked;
                else val = e.value;
              } catch (ee) { val = null; }
              fields.push({ tag: e.tagName, dataField: df, value: val });
            }
          } catch (ee) {}
          return { innerHTML: container.innerHTML, fields: fields };
        }
        const snap = {
          ts: (new Date()).toISOString(),
          form: { id: f.id || null, name: f.name || null, action: f.action || null },
          ctx: snapshotContainer(ctxContainer),
          sig: snapshotContainer(sigContainer),
          trg: snapshotContainer(trgContainer)
        };
        if (GUIDED_V2_DEBUG) {
          try { window.__guided_v2_last_dom_snapshot = snap; } catch (ee) {}
          try {
            if (!Array.isArray(window.__guided_v2_dom_snapshots)) window.__guided_v2_dom_snapshots = [];
            window.__guided_v2_dom_snapshots.push(snap);
            if (window.__guided_v2_dom_snapshots.length > 20) window.__guided_v2_dom_snapshots.shift();
          } catch (ee) {}
          try { if (typeof localStorage !== 'undefined') localStorage.setItem('__guided_v2_dom_snapshot', JSON.stringify(snap)); } catch (ee) {}
          try { if (typeof localStorage !== 'undefined') localStorage.setItem('__guided_v2_dom_snapshots', JSON.stringify(window.__guided_v2_dom_snapshots || [])); } catch (ee) {}
          try { console.log('[guided-debug] dom-snapshot saved', snap.ts, f.id || f.name || f.action); } catch (ee) {}
        }
      } catch (ee) {}

      try { writeHiddenStamped(f, 'context_rules_json', JSON.stringify(canonicalSerialize(ctxContainer)), 'final-pre-submit'); } catch (e) {}
      try { writeHiddenStamped(f, 'signal_rules_json', JSON.stringify(canonicalSerialize(sigContainer)), 'final-pre-submit'); } catch (e) {}
      try { writeHiddenStamped(f, 'trigger_rules_json', JSON.stringify(canonicalSerialize(trgContainer)), 'final-pre-submit'); } catch (e) {}
    } catch (e) { /* best-effort */ }
  }
  try { window.__guided_v2_invoke_final_pre_write = invokeFinalPreSubmitWrites; } catch (e) {}

  // Prefer a DOM-appropriate serializer. Some renderers use class-based
  // markup (serializeRulesFromDOM) while others (advanced renderer) use a
  // minimal data-* schema (serializeRulesFromDOMMinimal). Use the minimal
  // serializer when present, otherwise fall back to the class-based one.
  function canonicalSerialize(container) {
    try {
      if (typeof serializeRulesFromDOMMinimal === 'function') {
        try { const m = serializeRulesFromDOMMinimal(container); if (Array.isArray(m)) return m; } catch (e) {}
      }
      if (typeof serializeRulesFromDOM === 'function') {
        try { const d = serializeRulesFromDOM(container); if (Array.isArray(d) && d.length > 0) return d; } catch (e) {}
      }
      // If prior serializers returned nothing but rows exist, attempt the
      // tolerant inline fallback to capture advanced renderer markup.
      try {
        const hasRows = container && container.querySelector && container.querySelectorAll && container.querySelectorAll('.rule-row') && container.querySelectorAll('.rule-row').length > 0;
        if (hasRows) {
          const fb = canonicalSerializeFallback(container);
          if (Array.isArray(fb)) return fb;
        }
      } catch (e) {}
    } catch (e) {}
    return [];
  }

  // Best-effort inline minimal serializer used when the dedicated minimal
  // serializer isn't present. This tolerantly reads rows and supports
  // controls annotated with `data-field` or `data-key` so advanced renderers
  // which don't populate `.rule-type` selects still serialize correctly.
  function canonicalSerializeFallback(container) {
    const out = [];
    try {
      if (!container) return out;
      const rows = Array.from(container.querySelectorAll('.rule-row'));
      for (const r of rows) {
        try {
          const typeEl = r.querySelector('.rule-type') || r.querySelector('[data-type]') || r.querySelector('[data-field="type"]') || r.querySelector('select[name="type"]');
          const type = typeEl ? (typeEl.value || typeEl.getAttribute && (typeEl.getAttribute('data-type') || typeEl.getAttribute('data-field')) || '') : '';
          const rule = { type: String(type || '') };
          const tfEl = r.querySelector('.rule-tf') || r.querySelector('[data-tf]') || r.querySelector('select[name="tf"]'); if (tfEl && tfEl.value && tfEl.value !== 'default') rule.tf = tfEl.value;
          const validEl = r.querySelector('.rule-valid') || r.querySelector('[data-valid]') || r.querySelector('input[name="valid_for_bars"]'); if (validEl && validEl.value !== '') { const n = Number(validEl.value); if (Number.isFinite(n)) rule.valid_for_bars = n; }
          // collect params from .param, data-field or data-key
          const params = r.querySelectorAll('.param, [data-field], [data-key]');
          for (const p of Array.from(params)) {
            try {
              const k = p.getAttribute && (p.getAttribute('data-key') || p.getAttribute('data-field') || p.getAttribute('name'));
              if (!k) continue;
              let v = null;
              if (p.type === 'checkbox') v = !!p.checked;
              else v = p.value;
              if (p.type === 'number') { const n = Number(v); rule[k] = Number.isFinite(n) ? n : v; } else { rule[k] = v; }
            } catch (ee) { /* ignore per-param errors */ }
          }
          const sideEl = r.querySelector('.rule-side') || r.querySelector('[data-side]') || r.querySelector('[data-field="side"]') || r.querySelector('select[name="side"]'); if (sideEl && sideEl.value) rule.side = String(sideEl.value).trim().toLowerCase();
          out.push(rule);
        } catch (e) { /* per-row best-effort */ }
      }
    } catch (e) { /* ignore */ }
    return out;
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
    // per-rule side selector (both | long | short) for all sections
    const sideLabel = el('label', { class: 'rule-side-label', text: 'Side: ' });
    const sideSel = el('select', { class: 'rule-side' });
    sideSel.title = 'Apply rule to both (default), long-only, or short-only.';
    sideSel.setAttribute('aria-label', 'Rule side selector');
    sideSel.appendChild(el('option', { value: 'both', text: 'both' }));
    sideSel.appendChild(el('option', { value: 'long', text: 'long' }));
    sideSel.appendChild(el('option', { value: 'short', text: 'short' }));
    if (rule && rule.side) sideSel.value = String(rule.side);
    sideLabel.appendChild(sideSel);
    left.querySelector('.rule-row-top').appendChild(sideLabel);

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
      const sideEl = r.querySelector('.rule-side'); if (sideEl && sideEl.value) rule.side = String(sideEl.value).trim().toLowerCase();
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
      const rules = canonicalSerialize(container);
      window.GuidedBuilderV2State.set(section + '_rules', rules);
    } catch (e) { console.error('commitSectionFromDOM', e); }
  }

  function syncHiddenFromState(form, state) {
    try {
      const ctx = ensureHidden(form, 'context_rules_json');
      const sig = ensureHidden(form, 'signal_rules_json');
      const trg = ensureHidden(form, 'trigger_rules_json');
      try {
        writeHiddenStamped(form, 'context_rules_json', JSON.stringify((state && state.context_rules) ? state.context_rules : []), 'state-sync');
      } catch (e) { writeHiddenStamped(form, 'context_rules_json', '[]', 'state-sync-fallback'); }
      try {
        writeHiddenStamped(form, 'signal_rules_json', JSON.stringify((state && state.signal_rules) ? state.signal_rules : []), 'state-sync');
      } catch (e) { writeHiddenStamped(form, 'signal_rules_json', '[]', 'state-sync-fallback'); }
      try {
        writeHiddenStamped(form, 'trigger_rules_json', JSON.stringify((state && state.trigger_rules) ? state.trigger_rules : []), 'state-sync');
      } catch (e) { writeHiddenStamped(form, 'trigger_rules_json', '[]', 'state-sync-fallback'); }
    } catch (e) { console.error('syncHiddenFromState', e); }
  }

  // Fallback DOM -> hidden synchronizer used by the minimal/advanced preview path.
  function syncHidden(form, ctxContainer, sigContainer, trgContainer) {
    try {
      if (!form) return;
      const ctxH = ensureHidden(form, 'context_rules_json');
      const sigH = ensureHidden(form, 'signal_rules_json');
      const trgH = ensureHidden(form, 'trigger_rules_json');
      try {
        writeHiddenStamped(form, 'context_rules_json', JSON.stringify(canonicalSerialize(ctxContainer)), 'dom-sync');
      } catch (e) { writeHiddenStamped(form, 'context_rules_json', '[]', 'dom-sync-fallback'); }
      try {
        writeHiddenStamped(form, 'signal_rules_json', JSON.stringify(canonicalSerialize(sigContainer)), 'dom-sync');
      } catch (e) { writeHiddenStamped(form, 'signal_rules_json', '[]', 'dom-sync-fallback'); }
      try {
        writeHiddenStamped(form, 'trigger_rules_json', JSON.stringify(canonicalSerialize(trgContainer)), 'dom-sync');
      } catch (e) { writeHiddenStamped(form, 'trigger_rules_json', '[]', 'dom-sync-fallback'); }
    } catch (e) { /* best-effort */ }
  }

  function init() {
    const form = document.querySelector('form[action="/create-strategy-guided/step4"]') || document.getElementById('guided_step4_form');
    if (!form) return;

    // Attach inline onsubmit attribute to cover any submit pathway
    try {
      const existingOn = form.onsubmit;
      form.onsubmit = function (ev) {
        try { window.__guided_v2_ensure_submit && window.__guided_v2_ensure_submit(form); } catch (e) {}
        if (typeof existingOn === 'function') return existingOn.call(this, ev);
        return true;
      };
      // also set attribute for non-JS consumers/other hooks
      try { form.setAttribute('onsubmit', 'return (window.__guided_v2_ensure_submit && window.__guided_v2_ensure_submit(this))'); } catch (e) {}
    } catch (e) { /* ignore */ }

    // Ensure programmatic submits also serialize hidden inputs.
    // Wrap `form.submit` and `form.requestSubmit` (if present) to perform
    // a synchronous DOM/state -> hidden sync before calling the original.
    (function bindProgrammaticSubmit(f) {
      try {
        if (!f) return;
        const origSubmit = f.submit && f.submit.bind(f);
        const origRequestSubmit = f.requestSubmit && f.requestSubmit.bind(f);

        function doSync() {
          try {
            // flush state first
            if (window.GuidedBuilderV2State) { window.GuidedBuilderV2State.flush && window.GuidedBuilderV2State.flush(); syncHiddenFromState(f, window.GuidedBuilderV2State.raw ? window.GuidedBuilderV2State.raw() : {}); }
          } catch (e) {}
          try {
            const ctx = document.getElementById('context-rules');
            const sig = document.getElementById('signal-rules');
            const trg = document.getElementById('trigger-rules');
            syncHidden(f, ctx, sig, trg);
          } catch (e) {}
          try { console.log('[guided-debug]', new Date().toISOString(), 'programmatic-sync ->', { ctx: (document.getElementById('context_rules_json')||{}).value, sig: (document.getElementById('signal_rules_json')||{}).value, trg: (document.getElementById('trigger_rules_json')||{}).value }); } catch (e) {}
        }

        if (origRequestSubmit) {
          f.requestSubmit = function (submitter) {
            try { doSync(); } catch (e) {}
            return origRequestSubmit(submitter);
          };
        }

        if (origSubmit) {
          f.submit = function () {
            try { doSync(); } catch (e) {}
            return origSubmit();
          };
        }
      } catch (e) { /* best-effort */ }
    })(form);

    // Authoritative global submit hook: ensure any call to `form.submit()` or
    // `form.requestSubmit()` performs a final synchronous serialization of
    // DOM -> hidden inputs and a state flush before calling the native submit.
    try {
      (function installGlobalSubmitHook() {
        try {
          const proto = HTMLFormElement && HTMLFormElement.prototype;
          if (!proto) return;
          // Avoid installing multiple times
          if (proto.__guided_v2_submit_hook_installed) return;

          const origSubmit = proto.submit;
          const origRequestSubmit = proto.requestSubmit;

          function finalSync(f, cb) {
            try {
              // best-effort: commit pending DOM sections
              try { commitSectionFromDOM('context'); commitSectionFromDOM('signal'); commitSectionFromDOM('trigger'); } catch (e) {}
              // flush state manager if present
              try { if (window.GuidedBuilderV2State) { window.GuidedBuilderV2State.flush && window.GuidedBuilderV2State.flush(); } } catch (e) {}
              // write authoritative stamped values synchronously
              try {
                const ctxContainer = f.querySelector ? f.querySelector('#context-rules') : document.getElementById('context-rules');
                const sigContainer = f.querySelector ? f.querySelector('#signal-rules') : document.getElementById('signal-rules');
                const trgContainer = f.querySelector ? f.querySelector('#trigger-rules') : document.getElementById('trigger-rules');
                // Attempt to send the payload via sendBeacon as a guaranteed-before-unload delivery.
                // Defer beacon/XHR and final hidden-writes until the next animation frames
                // so the renderer has a chance to apply any pending DOM updates.
                try {
                  afterNextFrame(function () {
                    // Another microtask tick after rAF to allow pending microtask DOM updates
                    Promise.resolve().then(function () {
                      // Schedule a macrotask so layout/painting and any late microtasks
                      // have a final chance to run before we serialize and write.
                      setTimeout(function () {
                        try {
                          let beaconOk = false;
                          const url = (window.__guided_v2_debug_beacon_url || (f.action || window.location.href));
                          if (typeof navigator !== 'undefined' && typeof navigator.sendBeacon === 'function') {
                            try {
                              const draftEl = f.querySelector && (f.querySelector('input[name="draft_id"]') || f.querySelector('#draft_id'));
                              const draftId = draftEl ? draftEl.value : (window.__guided_v2_draft_id || '');
                              const ctxHidden = f.querySelector && f.querySelector('#context_rules_json');
                              const sigHidden = f.querySelector && f.querySelector('#signal_rules_json');
                              const trgHidden = f.querySelector && f.querySelector('#trigger_rules_json');
                              const ctxVal = ctxHidden && ctxHidden.value !== undefined ? parseJsonSafe(ctxHidden.value) : canonicalSerialize(ctxContainer);
                              const sigVal = sigHidden && sigHidden.value !== undefined ? parseJsonSafe(sigHidden.value) : canonicalSerialize(sigContainer);
                              const trgVal = trgHidden && trgHidden.value !== undefined ? parseJsonSafe(trgHidden.value) : canonicalSerialize(trgContainer);
                              const payload = { draft_id: draftId, context_rules: ctxVal, signal_rules: sigVal, trigger_rules: trgVal };
                              const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
                              beaconOk = !!navigator.sendBeacon(url, blob);
                              try { console.log('[guided-debug] sendBeacon ->', beaconOk, url, { ctxLen: JSON.stringify(ctxVal).length, sigLen: JSON.stringify(sigVal).length, trgLen: JSON.stringify(trgVal).length }); } catch (e) {}
                            } catch (be) { try { console.log('[guided-debug] sendBeacon failed', be && be.message); } catch (e) {} }
                          }
                          if (!beaconOk) {
                            try {
                              const params = new URLSearchParams();
                              const draftEl2 = f.querySelector && (f.querySelector('input[name="draft_id"]') || f.querySelector('#draft_id'));
                              const draftId2 = draftEl2 ? draftEl2.value : (window.__guided_v2_draft_id || '');
                              const ctxHidden2 = f.querySelector && f.querySelector('#context_rules_json');
                              const sigHidden2 = f.querySelector && f.querySelector('#signal_rules_json');
                              const trgHidden2 = f.querySelector && f.querySelector('#trigger_rules_json');
                              const ctxVal2 = ctxHidden2 && ctxHidden2.value !== undefined ? ctxHidden2.value : JSON.stringify(canonicalSerialize(ctxContainer));
                              const sigVal2 = sigHidden2 && sigHidden2.value !== undefined ? sigHidden2.value : JSON.stringify(canonicalSerialize(sigContainer));
                              const trgVal2 = trgHidden2 && trgHidden2.value !== undefined ? trgHidden2.value : JSON.stringify(canonicalSerialize(trgContainer));
                              params.append('draft_id', draftId2);
                              params.append('context_rules_json', ctxVal2);
                              params.append('signal_rules_json', sigVal2);
                              params.append('trigger_rules_json', trgVal2);
                              try { console.log('[guided-debug] final-pre-submit syncXHR ->', url, params.toString().slice(0,200)); } catch (e) {}
                              const xhr = new XMLHttpRequest(); xhr.open('POST', url, false); xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded; charset=UTF-8'); xhr.send(params.toString());
                              try { console.log('[guided-debug] final-pre-submit syncXHR status ->', xhr.status); } catch (e) {}
                            } catch (sx) { try { console.log('[guided-debug] final-pre-submit syncXHR failed', sx && sx.message); } catch (e) {} }
                          }
                          try { console.log('[guided-debug] SERIALIZE-DOM (finalSync - pre-final-writes) ' + JSON.stringify({ ctxSerialized: JSON.stringify(canonicalSerialize(ctxContainer)), ctxInner: (ctxContainer||{}).innerHTML, sigSerialized: JSON.stringify(canonicalSerialize(sigContainer)), sigInner: (sigContainer||{}).innerHTML, trgSerialized: JSON.stringify(canonicalSerialize(trgContainer)), trgInner: (trgContainer||{}).innerHTML })); } catch (e) {}
                          // Capture a DOM snapshot at this exact moment for debugging (fields + innerHTML)
                          try {
                            function _snapshotContainer(container) {
                              if (!container) return null;
                              const fields = [];
                              try {
                                const els = Array.from(container.querySelectorAll('select,input,textarea,[data-field],[data-key]'));
                                for (const e of els) {
                                  const df = e.getAttribute('data-field') || e.getAttribute('data-key') || null;
                                  let val = null;
                                  try { if (e.type === 'checkbox') val = !!e.checked; else val = e.value; } catch (ee) { val = null; }
                                  fields.push({ tag: e.tagName, dataField: df, value: val });
                                }
                              } catch (ee) {}
                              return { innerHTML: container.innerHTML, fields: fields };
                            }
                            const snap2 = { ts: (new Date()).toISOString(), form: { id: f.id || null, name: f.name || null, action: f.action || null }, ctx: _snapshotContainer(ctxContainer), sig: _snapshotContainer(sigContainer), trg: _snapshotContainer(trgContainer) };
                            if (GUIDED_V2_DEBUG) {
                              try { window.__guided_v2_last_dom_snapshot = snap2; } catch (ee) {}
                              try { if (!Array.isArray(window.__guided_v2_dom_snapshots)) window.__guided_v2_dom_snapshots = []; window.__guided_v2_dom_snapshots.push(snap2); if (window.__guided_v2_dom_snapshots.length > 20) window.__guided_v2_dom_snapshots.shift(); } catch (ee) {}
                              try { if (typeof localStorage !== 'undefined') localStorage.setItem('__guided_v2_dom_snapshot', JSON.stringify(snap2)); } catch (ee) {}
                              try { if (typeof localStorage !== 'undefined') localStorage.setItem('__guided_v2_dom_snapshots', JSON.stringify(window.__guided_v2_dom_snapshots || [])); } catch (ee) {}
                              try { console.log('[guided-debug] dom-snapshot saved (finalSync)', snap2.ts); } catch (ee) {}
                            }
                          } catch (ee) {}
                          writeHiddenStamped(f, 'context_rules_json', JSON.stringify(canonicalSerialize(ctxContainer)), 'final-pre-submit');
                          writeHiddenStamped(f, 'signal_rules_json', JSON.stringify(canonicalSerialize(sigContainer)), 'final-pre-submit');
                          writeHiddenStamped(f, 'trigger_rules_json', JSON.stringify(canonicalSerialize(trgContainer)), 'final-pre-submit');
                          try { if (typeof cb === 'function') cb(); } catch (e) {}
                        } catch (enne) { try { console.log('[guided-debug] final-pre-submit beacon/xhr error', enne && enne.message); } catch (e) {} }
                      }, 0);
                    }).catch(function (err) { try { console.log('[guided-debug] finalSync microtask error', err && err.message); } catch (e) {} });
                  });
                } catch (e) {
                  try { syncHiddenFromState(f, window.GuidedBuilderV2State && window.GuidedBuilderV2State.raw ? window.GuidedBuilderV2State.raw() : {}); } catch (e2) {}
                  try { if (typeof cb === 'function') cb(); } catch (e) {}
                }
              } catch (e) {
                try { syncHiddenFromState(f, window.GuidedBuilderV2State && window.GuidedBuilderV2State.raw ? window.GuidedBuilderV2State.raw() : {}); } catch (e2) {}
                try { if (typeof cb === 'function') cb(); } catch (e) {}
              }
            } catch (e) { /* ignore */ }
          }

          proto.submit = function () {
            try {
              finalSync(this, () => {
                try { window.__guided_v2_in_submit = true; } catch (e) {}
                try {
                  const res = origSubmit.apply(this, arguments);
                  setTimeout(() => { try { window.__guided_v2_in_submit = false; } catch (e) {} }, 2000);
                  return res;
                } catch (e) {
                  try { window.__guided_v2_in_submit = false; } catch (ee) {}
                  throw e;
                }
              });
            } catch (e) {}
          };

          if (origRequestSubmit) {
            proto.requestSubmit = function (submitter) {
              try {
                finalSync(this, () => {
                  try { window.__guided_v2_in_submit = true; } catch (e) {}
                  try {
                    const res = origRequestSubmit.apply(this, arguments);
                    setTimeout(() => { try { window.__guided_v2_in_submit = false; } catch (e) {} }, 2000);
                    return res;
                  } catch (e) {
                    try { window.__guided_v2_in_submit = false; } catch (ee) {}
                    throw e;
                  }
                });
              } catch (e) {}
            };
          }

          proto.__guided_v2_submit_hook_installed = true;
        } catch (e) { /* best-effort */ }
      })();
    } catch (e) { /* ignore */ }

    // Watch for forms being replaced dynamically and rebind our wrappers
    try {
      const mo = new MutationObserver((mutations) => {
        for (const m of mutations) {
          for (const n of Array.from(m.addedNodes || [])) {
            try {
              if (!(n instanceof HTMLElement)) continue;
              const newForm = n.querySelector && n.querySelector('form[action="/create-strategy-guided/step4"]') || (n.id === 'guided_step4_form' ? n : null);
              if (newForm) {
                (function bindAgain(f) {
                  try {
                    const origSubmit = f.submit && f.submit.bind(f);
                    const origRequestSubmit = f.requestSubmit && f.requestSubmit.bind(f);
                    if (!origSubmit && !origRequestSubmit) return;
                    // reuse same doSync logic
                    const doSync = function () {
                      try { if (window.GuidedBuilderV2State) { window.GuidedBuilderV2State.flush && window.GuidedBuilderV2State.flush(); syncHiddenFromState(f, window.GuidedBuilderV2State.raw ? window.GuidedBuilderV2State.raw() : {}); } } catch (e) {}
                      try { syncHidden(f, document.getElementById('context-rules'), document.getElementById('signal-rules'), document.getElementById('trigger-rules')); } catch (e) {}
                    };
                    if (origRequestSubmit) f.requestSubmit = function (submitter) { try { doSync(); } catch (e) {} ; return origRequestSubmit(submitter); };
                    if (origSubmit) f.submit = function () { try { doSync(); } catch (e) {} ; return origSubmit(); };
                  } catch (e) {}
                })(newForm);
              }
            } catch (e) {}
          }
        }
      });
      mo.observe(document.documentElement || document.body, { childList: true, subtree: true });
    } catch (e) { /* ignore */ }

    // Ensure we always serialize DOM -> hidden inputs before any submit.
    // Use a capture-phase listener so this runs before other submit handlers
    // (prevents races where requestSubmit or other handlers cause submission
    // before hidden inputs are populated).
    try {
      form.addEventListener('submit', function (ev) {
        try {
          const ctxContainer2 = document.getElementById('context-rules');
          const sigContainer2 = document.getElementById('signal-rules');
          const trgContainer2 = document.getElementById('trigger-rules');
          let ctxH = document.getElementById('context_rules_json'); if (!ctxH) ctxH = ensureHidden(form, 'context_rules_json');
          let sigH = document.getElementById('signal_rules_json'); if (!sigH) sigH = ensureHidden(form, 'signal_rules_json');
          let trgH = document.getElementById('trigger_rules_json'); if (!trgH) trgH = ensureHidden(form, 'trigger_rules_json');
              try {
                // use stamped writes so we can trace who last changed the inputs
                writeHiddenStamped(form, 'context_rules_json', JSON.stringify(canonicalSerialize(ctxContainer2)), 'capture-submit-dom');
                writeHiddenStamped(form, 'signal_rules_json', JSON.stringify(canonicalSerialize(sigContainer2)), 'capture-submit-dom');
                writeHiddenStamped(form, 'trigger_rules_json', JSON.stringify(canonicalSerialize(trgContainer2)), 'capture-submit-dom');
              } catch (e) {
                // best-effort: fall back to state if available
                try { if (window.GuidedBuilderV2State && window.GuidedBuilderV2State.raw) { syncHiddenFromState(form, window.GuidedBuilderV2State.raw()); } } catch (e2) {}
              }
        } catch (e) { /* best-effort */ }
      }, true);
    } catch (e) { /* ignore */ }

    function performSubmitWithEvents(f) {
      try {
        if (typeof f.requestSubmit === 'function') { f.requestSubmit(); return; }
        const evt = new Event('submit', { bubbles: true, cancelable: true });
        const notCanceled = f.dispatchEvent(evt);
        if (notCanceled) {
          // call native submit
          HTMLFormElement.prototype.submit.call(f);
        }
      } catch (e) {
        try { HTMLFormElement.prototype.submit.call(f); } catch (e2) { /* ignore */ }
      }
    }

    // If the advanced guided builder (preview body) is present, prefer its
    // renderer to avoid duplicate/wrong wiring from this legacy module.
    // When the advanced renderer is present we should NOT run the full
    // legacy wiring (it will cause duplicate renders). However we still
    // must ensure the form submit handler syncs hidden JSON inputs/draft
    // id. Attach a minimal submit handler and return early.
    const advancedPreviewBody = document.getElementById('effective-preview-body');
    // If the advanced preview body exists we normally prefer the advanced
    // renderer and skip the full wiring. However, allow an override via
    // `window.guidedBuilderV2PreferFullRenderer` so reopen flows that carry
    // detailed UI metadata can force the full renderer to initialize.
    if (advancedPreviewBody && !window.guidedBuilderV2PreferFullRenderer) {
      // ensure hidden inputs exist
      ensureHidden(form, 'context_rules_json'); ensureHidden(form, 'signal_rules_json'); ensureHidden(form, 'trigger_rules_json');
      // minimal submit sync to preserve draft/hidden JSON when submitting from advanced UI
      form.addEventListener('submit', () => {
        try {
          if (window.GuidedBuilderV2State) {
            window.GuidedBuilderV2State.flush && window.GuidedBuilderV2State.flush();
            const s = window.GuidedBuilderV2State.raw ? window.GuidedBuilderV2State.raw() : {};
            syncHiddenFromState(form, s);
          } else {
            // fallback: sync directly from DOM containers if present
            const ctxContainer = document.getElementById('context-rules');
            const sigContainer = document.getElementById('signal-rules');
            const trgContainer = document.getElementById('trigger-rules');
            syncHidden(form, ctxContainer, sigContainer, trgContainer);
          }
          // Additionally, always attempt to sync from DOM rule containers
          // to cover cases where rules exist in the DOM but weren't yet
          // reflected in the state manager.
          try {
            const ctxContainer2 = document.getElementById('context-rules');
            const sigContainer2 = document.getElementById('signal-rules');
            const trgContainer2 = document.getElementById('trigger-rules');
            syncHidden(form, ctxContainer2, sigContainer2, trgContainer2);
            try { console.log('[guided-debug] minimal submit sync ->', { ctx: (document.getElementById('context_rules_json')||{}).value, sig: (document.getElementById('signal_rules_json')||{}).value, trg: (document.getElementById('trigger_rules_json')||{}).value }); } catch (e) {}
          } catch (e) { /* best-effort */ }
        } catch (e) { console.error('submit sync (minimal) failed', e); }
      });
      // Also intercept the Review button click to force immediate DOM->hidden sync
      try {
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn && !submitBtn.dataset.guidedV2ClickWired) {
          // Ensure hidden inputs are populated as early as possible across input types
          const ensureHiddenPopulated = () => {
              try {
              commitSectionFromDOM('context'); commitSectionFromDOM('signal'); commitSectionFromDOM('trigger');
              if (window.GuidedBuilderV2State) { window.GuidedBuilderV2State.flush && window.GuidedBuilderV2State.flush(); syncHiddenFromState(form, window.GuidedBuilderV2State.raw ? window.GuidedBuilderV2State.raw() : {}); }
              // fallback DOM serialization (stamped)
              try {
                writeHiddenStamped(form, 'context_rules_json', JSON.stringify(canonicalSerialize(document.getElementById('context-rules'))), 'submit-btn-fallback');
                writeHiddenStamped(form, 'signal_rules_json', JSON.stringify(canonicalSerialize(document.getElementById('signal-rules'))), 'submit-btn-fallback');
                writeHiddenStamped(form, 'trigger_rules_json', JSON.stringify(canonicalSerialize(document.getElementById('trigger-rules'))), 'submit-btn-fallback');
              } catch (e) { /* best-effort */ }
            } catch (e) { console.error('ensureHiddenPopulated failed', e); }
          };

          // Best-effort beacon/xhr to ensure server receives serialized payload
          const tryBeaconFromEnsure = () => {
            try {
              const ctxC = document.getElementById('context-rules');
              const sigC = document.getElementById('signal-rules');
              const trgC = document.getElementById('trigger-rules');
              const url = (window.__guided_v2_debug_beacon_url || (form.action || window.location.href));
              // Wait for next frames and flush state so DOM/state have settled
              try {
                afterNextFrame(function () {
                  // One more microtask tick after rAF to allow any microtask DOM updates
                  Promise.resolve().then(function () {
                    try { if (window.GuidedBuilderV2State) { window.GuidedBuilderV2State.flush && window.GuidedBuilderV2State.flush(); } } catch (e) {}
                    try {
                      let beaconOk = false;
                      if (typeof navigator !== 'undefined' && typeof navigator.sendBeacon === 'function') {
                        try {
                          const fd = new FormData();
                          const draftEl = form.querySelector && (form.querySelector('input[name="draft_id"]') || form.querySelector('#draft_id'));
                          const did = draftEl ? draftEl.value : (window.__guided_v2_draft_id || '');
                          fd.append('draft_id', did);
                          fd.append('context_rules_json', JSON.stringify(canonicalSerialize(ctxC)));
                          fd.append('signal_rules_json', JSON.stringify(canonicalSerialize(sigC)));
                          fd.append('trigger_rules_json', JSON.stringify(canonicalSerialize(trgC)));
                          beaconOk = !!navigator.sendBeacon(url, fd);
                          try { console.log('[guided-debug] sendBeacon(ensure) ->', beaconOk, url); } catch (e) {}
                        } catch (be) { try { console.log('[guided-debug] sendBeacon(ensure) failed', be && be.message); } catch (e) {} }
                      }
                      if (!beaconOk) {
                        try {
                          const params = new URLSearchParams();
                          const draftEl2 = form.querySelector && (form.querySelector('input[name="draft_id"]') || form.querySelector('#draft_id'));
                          const did2 = draftEl2 ? draftEl2.value : (window.__guided_v2_draft_id || '');
                          params.append('draft_id', did2);
                          params.append('context_rules_json', JSON.stringify(canonicalSerialize(ctxC)));
                          params.append('signal_rules_json', JSON.stringify(canonicalSerialize(sigC)));
                          params.append('trigger_rules_json', JSON.stringify(canonicalSerialize(trgC)));
                          try { console.log('[guided-debug] ensure-syncXHR ->', url); } catch (e) {}
                          const xhr = new XMLHttpRequest(); xhr.open('POST', url, false); xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded; charset=UTF-8'); xhr.send(params.toString());
                          try { console.log('[guided-debug] ensure-syncXHR status ->', xhr.status); } catch (e) {}
                        } catch (sx) { try { console.log('[guided-debug] ensure-syncXHR failed', sx && sx.message); } catch (e) {} }
                      }
                    } catch (eee) { try { console.log('[guided-debug] tryBeaconFromEnsure error', eee && eee.message); } catch (e) {} }
                  }).catch(function (err) { try { console.log('[guided-debug] tryBeaconFromEnsure microtask error', err && err.message); } catch (e) {} });
                });
              } catch (e) { try { console.log('[guided-debug] tryBeaconFromEnsure scheduling error', e && e.message); } catch (e2) {} }
            } catch (eee) { try { console.log('[guided-debug] tryBeaconFromEnsure error', eee && eee.message); } catch (e) {} }
          };

          // Wire pointer/touch/click to populate as early as possible
          // On pointer/touch start, populate hidden inputs and attempt an early
          // beacon, but do NOT mark `__guided_v2_in_submit` yet — that flag is
          // authoritative for blocking non-final writers and must be set only
          // once final write is imminent.
          submitBtn.addEventListener('pointerdown', (ev) => { try { ensureHiddenPopulated(); try { tryBeaconFromEnsure(); } catch (e) {} } catch (e) {} });
          submitBtn.addEventListener('touchstart', (ev) => { try { ensureHiddenPopulated(); try { tryBeaconFromEnsure(); } catch (e) {} } catch (e) {} }, { passive: true });
          submitBtn.addEventListener('click', (ev) => {
            // prevent duplicate handling if another handler already started submission
            if (form.dataset.guidedV2Submitting) return;
            form.dataset.guidedV2Submitting = '1';
            ev.preventDefault();
            try {
              // populate hidden inputs (try fast path first) and mark submit-in-progress
              try { ensureHiddenPopulated(); try { tryBeaconFromEnsure(); } catch (e) {} } catch (e) { /* ignore */ }
            } catch (e) { console.error('submit-button click sync failed', e); }
            try { console.log('[guided-debug] submit-button click -> hidden', { ctx: (document.getElementById('context_rules_json')||{}).value, sig: (document.getElementById('signal_rules_json')||{}).value, trg: (document.getElementById('trigger_rules_json')||{}).value }); } catch (e) {}
            // proceed with submit via prototype.submit so the overridden
            // submit (which runs finalSync) executes before native submit
            try { HTMLFormElement.prototype.submit.call(form); } catch (e) { console.error('form.submit failed', e); }
            setTimeout(() => { try { delete form.dataset.guidedV2Submitting; } catch (e) {} }, 2000);
          });
              submitBtn.dataset.guidedV2ClickWired = '1';
            }
          } catch (e) { /* ignore */ }
          // NOTE: do not return here — allow the full renderer wiring to run as
          // well so the page initializes the richer per-type renderer while
          // retaining the minimal submit-sync handlers above. This prevents
          // lost-settings bugs when the minimal path fails to hydrate complex
          // rule parameters before submit.
        }

    // Capture-phase submit handler: ensure we serialize DOM -> hidden inputs
    try {
      // Robust capture-phase submit: prevent default, populate hidden inputs,
      // then perform native submit to avoid other handlers overwriting values.
      form.addEventListener(
        'submit',
        function (ev) {
          try {
            ev.preventDefault();
            // populate hidden inputs synchronously
            try {
              // prefer state flush
              if (window.GuidedBuilderV2State) { window.GuidedBuilderV2State.flush && window.GuidedBuilderV2State.flush(); syncHiddenFromState(form, window.GuidedBuilderV2State.raw ? window.GuidedBuilderV2State.raw() : {}); }
            } catch (e) {}
            try {
              const ctxContainer2 = document.getElementById('context-rules');
              const sigContainer2 = document.getElementById('signal-rules');
              const trgContainer2 = document.getElementById('trigger-rules');
              const ctxH = document.getElementById('context_rules_json');
              const sigH = document.getElementById('signal_rules_json');
              const trgH = document.getElementById('trigger_rules_json');
              if (ctxH) ctxH.value = JSON.stringify(canonicalSerialize(ctxContainer2));
              if (sigH) sigH.value = JSON.stringify(canonicalSerialize(sigContainer2));
              if (trgH) trgH.value = JSON.stringify(canonicalSerialize(trgContainer2));
            } catch (e) {}
            try { console.log('[guided-debug] capture-submit -> hidden', { ctx: (document.getElementById('context_rules_json')||{}).value, sig: (document.getElementById('signal_rules_json')||{}).value, trg: (document.getElementById('trigger_rules_json')||{}).value }); } catch (e) {}
            // perform native submit (bypass other submit handlers)
            setTimeout(function () { try { HTMLFormElement.prototype.submit.call(form); } catch (e) { try { performSubmitWithEvents(form); } catch (e2) {} } }, 0);
          } catch (e) { /* best-effort */ }
        },
        true
      );
    } catch (e) { /* ignore */ }

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
          // Avoid re-rendering a section while the user is actively focused
          // inside that section (prevents focus loss while typing).
          const active = document.activeElement;
          if (!(active && ctxContainer && ctxContainer.contains(active))) {
            renderRulesInto(ctxContainer, s.context_rules || [], 'context');
          }
          if (!(active && sigContainer && sigContainer.contains(active))) {
            renderRulesInto(sigContainer, s.signal_rules || [], 'signal');
          }
          if (!(active && trgContainer && trgContainer.contains(active))) {
            renderRulesInto(trgContainer, s.trigger_rules || [], 'trigger');
          }
          syncHiddenFromState(form, s);
          // If a submit button click has started (pending submit), ensure the
          // authoritative final-pre-submit writes happen immediately so the
          // server receives the latest DOM state rather than an earlier stale
          // value.
          try {
            if (form && form.dataset && form.dataset.guidedV2Submitting) {
              try { window.__guided_v2_invoke_final_pre_write && window.__guided_v2_invoke_final_pre_write(form); } catch (e) {}
            }
          } catch (e) {}
        } catch (e) { console.error('state subscribe render error', e); }
      });
    }

    // wire add buttons
    const addCtx = document.getElementById('add-context');
    const addSig = document.getElementById('add-signal');
    const addTrg = document.getElementById('add-trigger');
    if (addCtx && ctxContainer && !addCtx.dataset.guidedV2Wired) {
      addCtx.addEventListener('click', (ev) => { ev.preventDefault(); ctxContainer.appendChild(createRuleRow('context', {})); commitSectionFromDOM('context'); });
      addCtx.dataset.guidedV2Wired = '1';
    }
    if (addSig && sigContainer && !addSig.dataset.guidedV2Wired) {
      addSig.addEventListener('click', (ev) => { ev.preventDefault(); sigContainer.appendChild(createRuleRow('signal', {})); commitSectionFromDOM('signal'); });
      addSig.dataset.guidedV2Wired = '1';
    }
    if (addTrg && trgContainer && !addTrg.dataset.guidedV2Wired) {
      addTrg.addEventListener('click', (ev) => { ev.preventDefault(); trgContainer.appendChild(createRuleRow('trigger', {})); commitSectionFromDOM('trigger'); });
      addTrg.dataset.guidedV2Wired = '1';
    }

    // edits/removes: delegate to container and commit on change
    const delegateEvents = (container, section) => {
      if (!container) return;
      container.addEventListener('input', () => commitSectionFromDOM(section));
      container.addEventListener('change', () => commitSectionFromDOM(section));
      container.addEventListener('click', (ev) => { if (ev.target && ev.target.classList && ev.target.classList.contains('rule-remove')) commitSectionFromDOM(section); });
      // On focusout (blur from any element within the container), ensure
      // the latest edits are committed to the state immediately so a
      // subsequent re-render doesn't overwrite them.
      container.addEventListener('focusout', (ev) => {
        try { commitSectionFromDOM(section); } catch (e) { console.error('focusout commit failed', e); }
      });
    };
    delegateEvents(ctxContainer, 'context'); delegateEvents(sigContainer, 'signal'); delegateEvents(trgContainer, 'trigger');

    // ensure hidden sync on submit
    form.addEventListener('submit', () => { if (window.GuidedBuilderV2State) { window.GuidedBuilderV2State.flush && window.GuidedBuilderV2State.flush(); const s = window.GuidedBuilderV2State.raw ? window.GuidedBuilderV2State.raw() : {}; syncHiddenFromState(form, s); } else { syncHidden(form, ctxContainer, sigContainer, trgContainer); } });
    // Also ensure the submit button click forces an immediate sync (best-effort)
    try {
      const submitBtn2 = form.querySelector('button[type="submit"]');
      if (submitBtn2 && !submitBtn2.dataset.guidedV2ClickWired) {
        submitBtn2.addEventListener('click', (ev) => {
          if (form.dataset.guidedV2Submitting) return;
          form.dataset.guidedV2Submitting = '1';
          ev.preventDefault();
          try {
            commitSectionFromDOM('context'); commitSectionFromDOM('signal'); commitSectionFromDOM('trigger');
            if (window.GuidedBuilderV2State) { window.GuidedBuilderV2State.flush && window.GuidedBuilderV2State.flush(); syncHiddenFromState(form, window.GuidedBuilderV2State.raw ? window.GuidedBuilderV2State.raw() : {}); }
            try {
              const ctxH = document.getElementById('context_rules_json'); if (ctxH) ctxH.value = JSON.stringify(canonicalSerialize(document.getElementById('context-rules')));
              const sigH = document.getElementById('signal_rules_json'); if (sigH) sigH.value = JSON.stringify(canonicalSerialize(document.getElementById('signal-rules')));
              const trgH = document.getElementById('trigger_rules_json'); if (trgH) trgH.value = JSON.stringify(canonicalSerialize(document.getElementById('trigger-rules')));
            } catch (e) { /* best-effort */ }
          } catch (e) { console.error('submit-button click sync failed', e); }
          try { HTMLFormElement.prototype.submit.call(form); } catch (e) { console.error('form.submit failed', e); }
          setTimeout(() => { try { delete form.dataset.guidedV2Submitting; } catch (e) {} }, 2000);
        });
        submitBtn2.dataset.guidedV2ClickWired = '1';
      }
    } catch (e) { /* ignore */ }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();

})();
    (function () {
      const form = document.querySelector('form[action="/create-strategy-guided/step4"]');
      // Feature flags (toggle experimental features safely)
      const FEATURE_FLAG = {
        volume: false,
      };
      const body = document.getElementById('effective-preview-body');
      if (!form || !body) return;

      // If the server-side template didn't populate `#draft_id`, try the
      // URL query parameter and populate the hidden input so POSTs include it.
      try {
        const draftEl = form.querySelector('#draft_id') || form.querySelector('input[name="draft_id"]');
        if (draftEl && (!draftEl.value || String(draftEl.value).trim() === '')) {
          const sp = new URLSearchParams(window.location.search || '');
          const q = sp.get('draft_id') || '';
          if (q) draftEl.value = q;
        }
      } catch (e) { console.error('populate draft_id from URL failed', e); }

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

      // Ensure the main primary context dropdown includes all per-row context block options.
      // This keeps the global primary selector in sync with available context blocks (including volume blocks).
      (function populatePrimaryContextOptions() {
        if (!primaryContextSel) return;
        const opts = [
          ['none', 'None'],
          ['ma_stack', 'MA Stack'],
          ['price_vs_ma', 'Trend: Price vs MA'],
          ['ma_cross_state', 'Trend: MA state (fast vs slow)'],
          ['atr_pct', 'Volatility: ATR% filter'],
          ['atr_spike', 'Volatility: ATR spike'],
          ['structure_breakout_state', 'Structure: Breakout state'],
          ['ma_spread_pct', 'Trend strength: MA spread %'],
          ['bollinger_context', 'Bollinger (anchor TF)'],
          ['relative_volume', 'Relative Volume (RVOL)'],
          ['volume_osc_increase', 'Volume Oscillator Increase'],
          ['volume_above_ma', 'Volume Above MA'],
          ['custom', 'Custom (DSL)']
        ];
        // Preserve current value if present
        const cur = String(primaryContextSel.value || '').trim();
        primaryContextSel.innerHTML = '';
        for (const [v, label] of opts) {
          const o = document.createElement('option'); o.value = v; o.textContent = label; primaryContextSel.appendChild(o);
        }
        if (cur) primaryContextSel.value = cur;
        else primaryContextSel.value = 'none';
      })();

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
        // per-rule side selector (both | long | short)
        const sideSel = el('select', { 'data-field': 'side', style: 'margin-left:10px;' });
        sideSel.title = 'Apply rule to both (default), long-only, or short-only.';
        sideSel.setAttribute('aria-label', 'Rule side selector');
        function addSideOpt(value, label) {
          const o = el('option'); o.value = value; o.textContent = label; sideSel.appendChild(o);
        }
        addSideOpt('both', 'Side: Both');
        addSideOpt('long', 'Side: Long');
        addSideOpt('short', 'Side: Short');
        sideSel.value = (rule && rule.side) ? String(rule.side) : 'both';
        left.appendChild(sideSel);
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
          block('price_vs_ma', (b) => {
            select(b, 'MA type', 'ma_type', [['ema', 'EMA'], ['sma', 'SMA']], rule?.ma_type || 'ema');
            inputNumber(b, 'MA length', 'length', rule?.length ?? 200, { min: 1 });
            select(b, 'Op', 'op', [['>=', '>='], ['<=', '<='], ['>', '>'], ['<', '<']], rule?.op || '>=');
            inputNumber(b, 'Threshold', 'threshold', rule?.threshold ?? 0.01, { step: 0.0001, min: 0 });
            b.appendChild(el('div', { class: 'muted', text: 'Price vs MA: compare price to a single MA.' }));
          });
          block('ma_stack', (b) => {
            select(b, 'MA type', 'ma_type', [['ema', 'EMA'], ['sma', 'SMA']], rule?.ma_type || 'ema');
            inputNumber(b, 'Fast MA', 'ma_fast', rule?.ma_fast ?? 20, { min: 1 });
            inputNumber(b, 'Mid MA', 'ma_mid', rule?.ma_mid ?? 50, { min: 1 });
            inputNumber(b, 'Slow MA', 'ma_slow', rule?.ma_slow ?? 200, { min: 1 });
            select(b, 'Stack mode', 'stack_mode', [['none', 'None'], ['long_only', 'Long only'], ['short_only', 'Short only']], rule?.stack_mode || 'none');
            inputNumber(b, 'Slope lookback', 'slope_lookback', rule?.slope_lookback ?? 10, { min: 1 });
            inputNumber(b, 'Min MA dist %', 'min_ma_dist_pct', rule?.min_ma_dist_pct ?? 0.01, { step: 0.0001, min: 0 });
            b.appendChild(el('div', { class: 'muted', text: 'MA Stack: configure multiple moving averages for trend detection.' }));
          });
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
          block('atr_spike', (b) => {
            inputNumber(b, 'ATR length', 'atr_len', rule?.atr_len ?? 14, { min: 1 });
            inputNumber(b, 'Lookback / rolling ATR lag', 'lookback', rule?.lookback ?? 1, { min: 1 });
            inputNumber(b, 'Multiplier', 'mult', rule?.mult ?? 1.3, { step: 0.01, min: 0 });
            b.appendChild(el('div', { class: 'muted', text: 'ATR Spike: true when current ATR > multiplier × rolling-average ATR.' }));
          });
          block('bollinger_context', (b) => {
            inputNumber(b, 'Period', 'length', rule?.length ?? 20, { min: 1 });
            inputNumber(b, 'Std dev (×mult)', 'mult', rule?.mult ?? 2.5, { step: 0.1, min: 0 });
            select(b, 'Mode', 'mode', [['inside', 'Inside band'], ['outside', 'Outside band'], ['near', 'Near band']], rule?.mode || 'inside');
            b.appendChild(el('div', { class: 'muted', text: 'Bollinger Context: require anchor TF price membership relative to Bollinger bands.' }));
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
          block('z_score', (b) => {
            inputNumber(b, 'Length (mean & std)', 'length', rule?.length ?? 20, { min: 1 });
            select(b, 'Operator', 'op', [['>', '>'], ['<', '<'], ['>=', '>='], ['<=', '<=']], rule?.op || '>');
            inputNumber(b, 'Threshold (abs)', 'threshold', rule?.threshold ?? 2.0, { step: 0.1, min: 0 });
            b.appendChild(el('div', { class: 'muted', text: 'Z-score = (close - mean)/std. Use next_open to avoid same-bar lookahead.' }));
          });
          block('bollinger_outlier', (b) => {
            inputNumber(b, 'Period', 'length', rule?.length ?? 20, { min: 1 });
            inputNumber(b, 'Std dev (×mult)', 'mult', rule?.mult ?? 2.5, { step: 0.1, min: 0 });
            select(b, 'Direction', 'direction', [['either', 'Either'], ['below', 'Below only'], ['above', 'Above only']], rule?.direction || 'either');
            inputCheckbox(b, 'Require close outside band', 'require_close_outside', rule?.require_close_outside);
            b.appendChild(el('div', { class: 'muted', text: 'Detects price closing outside Bollinger bands.' }));
          });
          block('atr_deviation', (b) => {
            inputNumber(b, 'Mean length', 'length', rule?.length ?? 20, { min: 1 });
            inputNumber(b, 'ATR length', 'atr_len', rule?.atr_len ?? 14, { min: 1 });
            inputNumber(b, 'Deviation k (×ATR)', 'threshold', rule?.threshold ?? 2.5, { step: 0.1, min: 0 });
            inputCheckbox(b, 'Use range-based ATR', 'use_range', (rule && rule.use_range) ? rule.use_range : true);
            b.appendChild(el('div', { class: 'muted', text: 'Price distance from MA or mean expressed in ATR multiples.' }));
          });
          block('delta_divergence', (b) => {
            inputNumber(b, 'Lookback bars', 'lookback', rule?.lookback ?? 5, { min: 1 });
            inputNumber(b, 'Threshold', 'threshold', rule?.threshold ?? 0.0, { step: 0.01, min: 0 });
            b.appendChild(el('div', { class: 'muted', text: 'Futures-only: price extreme without supporting delta move (absorption/divergence).' }));
          });
          block('volume_rejection', (b) => {
            inputNumber(b, 'Volume MA length', 'vol_len', rule?.vol_len ?? 20, { min: 1 });
            inputNumber(b, 'Multiplier', 'mult', rule?.mult ?? 1.5, { step: 0.1, min: 0 });
            inputNumber(b, 'Close-in-range pct', 'close_pct', rule?.close_pct ?? 0.3, { step: 0.01, min: 0, max: 1 });
            b.appendChild(el('div', { class: 'muted', text: 'Volume spike + rejection candle: high-volume extreme followed by a rejection close.' }));
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
          block('volume_rejection', (b) => {
            inputNumber(b, 'Volume MA length', 'vol_ma_len', rule?.vol_ma_len ?? 20, { min: 1 });
            inputNumber(b, 'Multiplier', 'multiplier', rule?.multiplier ?? 1.5, { step: 0.1, min: 0 });
            inputNumber(b, 'Close-in-range pct', 'close_pct', rule?.close_pct ?? 0.3, { step: 0.01, min: 0, max: 1 });
            b.appendChild(el('div', { class: 'muted', text: 'Trigger when extreme + volume spike + rejection-close occurs.' }));
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
          addOpt('ma_stack', 'MA Stack');
          addOpt('price_vs_ma', 'Trend: Price vs MA');
          addOpt('ma_cross_state', 'Trend: MA state (fast vs slow)');
          addOpt('atr_pct', 'Volatility: ATR% filter');
          addOpt('atr_spike', 'Volatility: ATR spike');
          addOpt('structure_breakout_state', 'Structure: Breakout state (close vs prior high/low)');
          addOpt('ma_spread_pct', 'Trend strength: MA spread %');
          addOpt('bollinger_context', 'Bollinger (anchor TF)');
          addOpt('relative_volume', 'Relative Volume (RVOL)');
          addOpt('volume_osc_increase', 'Volume Oscillator Increase');
          addOpt('volume_above_ma', 'Volume Above MA');
          addOpt('custom', 'Custom (DSL)');
        } else if (section === 'signal') {
          addOpt('ma_cross', 'MA cross (fast vs slow)');
          addOpt('rsi_threshold', 'RSI threshold');
          addOpt('rsi_cross_back', 'RSI cross back');
          addOpt('donchian_breakout', 'Donchian breakout/breakdown');
          addOpt('z_score', 'Z-score (statistical)');
          addOpt('bollinger_outlier', 'Bollinger outlier');
          addOpt('atr_deviation', 'ATR deviation');
          addOpt('delta_divergence', 'Delta divergence (futures)');
          addOpt('new_high_low_breakout', 'New high/low breakout');
          addOpt('pullback_to_ma', 'Pullback to MA');
          addOpt('relative_volume', 'Relative Volume (RVOL)');
          addOpt('volume_osc_increase', 'Volume Oscillator Increase');
          addOpt('volume_above_ma', 'Volume Above MA');
          addOpt('volume_rejection', 'Volume rejection (spike+reject)');
          addOpt('custom', 'Custom (DSL)');
        } else {
          addOpt('pin_bar', 'Pin bar');
          addOpt('inside_bar_breakout', 'Inside bar breakout');
          addOpt('engulfing', 'Engulfing');
          addOpt('ma_reclaim', 'MA reclaim');
          addOpt('volume_rejection', 'Volume rejection (spike+reject)');
          addOpt('prior_bar_break', 'Break prior bar high/low');
          addOpt('donchian_breakout', 'Donchian breakout/breakdown');
          addOpt('range_breakout', 'Range breakout (highest/lowest)');
          addOpt('wide_range_candle', 'Wide-range candle (vs ATR)');
          addOpt('custom', 'Custom (DSL)');
        }

        // Volume rule blocks (Context & Signal)
        if (section === 'context' || section === 'signal') {
          block('relative_volume', (b) => {
            select(b, 'MA type', 'ma_type', [['sma', 'SMA'], ['ema', 'EMA']], (rule && rule.ma_type != null) ? rule.ma_type : 'sma');
            inputNumber(b, 'MA length', 'length', (rule && rule.length != null) ? rule.length : 20, { min: 1 });
            select(b, 'Operator', 'op', [['>=', '≥'], ['<=', '≤'], ['>', '>'], ['<', '<']], (rule && rule.op != null) ? rule.op : '>=');
            inputNumber(b, 'Threshold', 'threshold', (rule && rule.threshold != null) ? rule.threshold : 1.5, { step: 0.01, min: 0 });
            b.appendChild(el('div', { class: 'muted', html: 'RVOL = volume / MA(volume, length). E.g. RVOL ≥ 1.5 means volume is 50% above normal.' }));
          });
          block('volume_osc_increase', (b) => {
            inputNumber(b, 'Fast length', 'fast', (rule && rule.fast != null) ? rule.fast : 12, { min: 1 });
            inputNumber(b, 'Slow length', 'slow', (rule && rule.slow != null) ? rule.slow : 26, { min: 1 });
            inputNumber(b, 'Min % increase', 'min_pct', (rule && rule.min_pct != null) ? rule.min_pct : 0.1, { step: 0.01, min: 0 });
            inputNumber(b, 'Lookback bars (N)', 'lookback', (rule && rule.lookback != null) ? rule.lookback : 3, { min: 1 });
            b.appendChild(el('div', { class: 'muted', html: 'True if (fast - slow) / slow increases by at least min % over N bars.' }));
          });
          block('volume_above_ma', (b) => {
            select(b, 'MA type', 'ma_type', [['ema', 'EMA'], ['sma', 'SMA']], (rule && rule.ma_type != null) ? rule.ma_type : 'ema');
            inputNumber(b, 'MA length', 'length', (rule && rule.length != null) ? rule.length : 20, { min: 1 });
            inputNumber(b, 'Min % above MA', 'min_pct', (rule && rule.min_pct != null) ? rule.min_pct : 0.1, { step: 0.01, min: 0 });
            b.appendChild(el('div', { class: 'muted', html: 'True if volume is at least min % above the MA.' }));
          });
        }

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

          // wire side selector to update preview/state
          try {
            sideSel.addEventListener('change', () => { try { syncHidden(); schedule(); } catch (e) {} });
          } catch (e) {}

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

          const sideEl = row.querySelector('[data-field="side"]');
          if (sideEl && sideEl.value) rule.side = String(sideEl.value || 'both').trim();

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
        const t = String(primaryContextSel?.value || 'none');
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
      if (addContextBtn && !addContextBtn.dataset.guidedV2Wired) {
        addContextBtn.addEventListener('click', () => {
          const primary = String(primaryContextSel?.value || 'none');
          if (primary === 'ma_stack') {
            if (maStackBlock) {
              maStackBlock.style.display = '';
              syncHidden();
              schedule();
              return;
            }
            // Fallback: if no global MA-stack block is present, insert a per-row ma_stack rule.
            if (ctxContainer) ctxContainer.appendChild(ruleRow('context', { type: 'ma_stack' }));
            syncHidden();
            schedule();
            return;
          }
          if (ctxContainer) ctxContainer.appendChild(ruleRow('context', defaultPrimaryContextRule(primary)));
          syncHidden();
          schedule();
        });
        addContextBtn.dataset.guidedV2Wired = '1';
      }
      const addSignalBtn = document.getElementById('add-signal');
      if (addSignalBtn && !addSignalBtn.dataset.guidedV2Wired) {
        addSignalBtn.addEventListener('click', () => {
          if (sigContainer) sigContainer.appendChild(ruleRow('signal', { type: 'rsi_threshold' }));
          syncHidden();
          schedule();
        });
        addSignalBtn.dataset.guidedV2Wired = '1';
      }
      const addTriggerBtn = document.getElementById('add-trigger');
      if (addTriggerBtn && !addTriggerBtn.dataset.guidedV2Wired) {
        addTriggerBtn.addEventListener('click', () => {
          if (trgContainer) trgContainer.appendChild(ruleRow('trigger', { type: 'prior_bar_break' }));
          syncHidden();
          schedule();
        });
        addTriggerBtn.dataset.guidedV2Wired = '1';
      }

      function updateTriggerParamVisibility() {
        const t = String(val('trigger_type') || '');
        const show = (id, on) => {
          const el = document.getElementById(id);
          if (!el) return;
          el.style.display = on ? '' : 'none';
        };
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

        // If the user is actively editing a context row/input, avoid auto-refreshing
        // to prevent the preview constantly reloading while typing. Honor `force`.
        try {
          const active = document.activeElement;
          if (!force && active && ctxContainer && ctxContainer.contains(active)) {
            // postpone refresh until user finishes editing
            return;
          }
        } catch (e) {
          // ignore and continue
        }

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

        // If user is focused inside any signal/trigger/context inputs, avoid
        // auto-refreshing the setup visual unless forced. This prevents reload
        // loops while configuring rules.
        try {
          const active = document.activeElement;
          if (!force && active && (ctxContainer?.contains(active) || sigContainer?.contains(active) || trgContainer?.contains(active))) {
            return;
          }
        } catch (e) {
          // ignore
        }

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
          trigger_type: val('trigger_type') || '',
          trigger_valid_for_bars: num('trigger_valid_for_bars'),
          // Global pin-bar trigger params removed; per-row trigger params used instead.
          trigger_ma_type: val('trigger_ma_type') || 'ema',
          trigger_ma_len: num('trigger_ma_len'),
          trigger_don_len: num('trigger_don_len'),
          trigger_range_len: num('trigger_range_len'),
          trigger_atr_len: num('trigger_atr_len'),
          trigger_atr_mult: num('trigger_atr_mult'),
          trigger_custom_bull_expr: val('trigger_custom_bull_expr') || '',
          trigger_custom_bear_expr: val('trigger_custom_bear_expr') || '',
          // Visual dataset hints
          asset: val('visual_symbol') || val('symbol') || '',
          asset_class: val('visual_asset_class') || '',
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
      form.addEventListener('submit', (ev) => {
        try {
          syncHidden();
          // Ensure `draft_id` hidden input exists and preserves its value so
          // server receives it on form POST (some flows expect JSON but the
          // normal form submit should include this field).
          let d = form.querySelector('#draft_id');
          if (!d) {
            const fallback = document.querySelector('input[name="draft_id"]')?.value || '';
            d = el('input', { type: 'hidden', id: 'draft_id', name: 'draft_id' });
            d.value = fallback;
            form.appendChild(d);
          } else if (!d.value) {
            const fallback = document.querySelector('input[name="draft_id"]')?.value || '';
            if (fallback) d.value = fallback;
          }
        } catch (e) { console.error('guided_builder_v2 submit sync error', e); }
      }, true);

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
  