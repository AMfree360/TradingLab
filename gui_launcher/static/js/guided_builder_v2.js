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
  // Historically this module would early-return when
  // `window.guidedBuilderV2PreferFullRenderer` was set to avoid
  // duplicating a richer full renderer that runs from another bundle.
  // In practice that flag may be present while the full renderer bundle
  // is not actually loaded (race or missing script). To ensure the
  // page always has a functioning renderer we no longer bail out here.
  // The module will initialize and subscribe to `GuidedBuilderV2State`.
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
        try { syncHidden(form, document.getElementById('context-rules'), document.getElementById('signal-rules'), document.getElementById('trigger-rules'), document.getElementById('trade-filters')); } catch (e) {}
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
    ['bollinger_context', 'Bollinger'],
    ['custom', 'Custom (DSL)']
  ];

  const WEEKDAYS = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const TIMEZONES = [
    'UTC','Etc/UTC','Europe/London','Europe/Berlin','Europe/Paris','Europe/Madrid','Europe/Rome','Europe/Stockholm','Europe/Warsaw',
    'America/New_York','America/Detroit','America/Toronto','America/Chicago','America/Winnipeg','America/Denver','America/Edmonton',
    'America/Los_Angeles','America/Vancouver','America/Sao_Paulo','America/Argentina/Buenos_Aires','America/Mexico_City',
    'Asia/Tokyo','Asia/Seoul','Asia/Shanghai','Asia/Hong_Kong','Asia/Singapore','Asia/Kolkata','Asia/Dubai','Asia/Jakarta',
    'Australia/Sydney','Australia/Melbourne','Pacific/Auckland','Africa/Johannesburg','Africa/Nairobi','Asia/Manila','Asia/Kuala_Lumpur',
    'Etc/GMT+12','Etc/GMT+11','Etc/GMT+10','Etc/GMT+9','Etc/GMT+8','Etc/GMT+7','Etc/GMT+6','Etc/GMT+5','Etc/GMT+4','Etc/GMT+3','Etc/GMT+2','Etc/GMT+1'
  ];

  // Ensure a shared datalist element exists for timezone suggestions. This creates
  // a lightweight, searchable autocomplete experience across all timezone inputs.
  function ensureTimezoneDatalist() {
    try {
      if (document.getElementById('guided-v2-timezone-datalist')) return;
      const dl = document.createElement('datalist');
      dl.id = 'guided-v2-timezone-datalist';
      // include a helpful empty option representing default/exchange
      const emptyOpt = document.createElement('option'); emptyOpt.value = ''; dl.appendChild(emptyOpt);
      for (const z of TIMEZONES) {
        try { const o = document.createElement('option'); o.value = z; dl.appendChild(o); } catch (e) {}
      }
      try {
        const head = document.head || document.getElementsByTagName('head')[0] || document.body || document.documentElement;
        head.appendChild(dl);
      } catch (e) {
        // fallback: append to body when available
        try { document.body.appendChild(dl); } catch (ee) {}
      }
    } catch (e) {}
  }
  try { if (typeof document !== 'undefined') { ensureTimezoneDatalist(); document.addEventListener && document.addEventListener('DOMContentLoaded', ensureTimezoneDatalist); } } catch (e) {}

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

  const tradeFilterTypes = [
    ['session_window', 'Session window (allow entries during session)'],
    ['blackout_window', 'Blackout window (disallow entries)'],
    ['day_month', 'Day / Month filter'],
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
      const tradeContainer = f.querySelector ? f.querySelector('#trade-filters') : document.getElementById('trade-filters');
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
      try { writeHiddenStamped(f, 'trade_filters_json', JSON.stringify(canonicalSerialize(tradeContainer)), 'final-pre-submit'); } catch (e) {}
    } catch (e) { /* best-effort */ }
  }
  try { window.__guided_v2_invoke_final_pre_write = invokeFinalPreSubmitWrites; } catch (e) {}

  // Prefer a DOM-appropriate serializer. Some renderers use class-based
  // markup (serializeRulesFromDOM) while others (advanced renderer) use a
  // minimal data-* schema (serializeRulesFromDOMMinimal). Use the minimal
  // serializer when present, otherwise fall back to the class-based one.
  function canonicalSerialize(container) {
    try {
      const hasRows = !!(container && container.querySelectorAll && container.querySelectorAll('.rule-row') && container.querySelectorAll('.rule-row').length > 0);
      if (typeof serializeRulesFromDOMMinimal === 'function') {
        try {
          const m = serializeRulesFromDOMMinimal(container);
          // Only trust an empty minimal-serializer result when there are
          // no rows. When rows exist, fall through so our tolerant fallback
          // can capture `data-field` based rows.
          if (Array.isArray(m) && (m.length > 0 || !hasRows)) return m;
        } catch (e) {}
      }
      if (typeof serializeRulesFromDOM === 'function') {
        try { const d = serializeRulesFromDOM(container); if (Array.isArray(d) && d.length > 0) return d; } catch (e) {}
      }
      // If prior serializers returned nothing but rows exist, attempt the
      // tolerant inline fallback to capture advanced renderer markup.
      try {
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
          // Determine selected rule type. Prefer the explicit type select.
          // Some renderers momentarily leave the select unselected
          // (selectedIndex=-1) even though options exist; default to the
          // first option in that case so we don't serialize type:"".
          const typeSelect = r.querySelector('select[data-field="type"]') || r.querySelector('.rule-type') || r.querySelector('select[name="type"]');
          let selectedType = '';
          try {
            if (typeSelect) {
              selectedType = String(typeSelect.value || '');
              if (!selectedType && typeof typeSelect.selectedIndex === 'number' && typeSelect.selectedIndex < 0) {
                const opt0 = (typeSelect.options && typeSelect.options.length) ? typeSelect.options[0] : null;
                if (opt0 && opt0.value !== undefined) selectedType = String(opt0.value);
              }
            }
          } catch (e) { selectedType = ''; }
          // Fallback: some advanced rows only expose per-type blocks.
          if (!selectedType) {
            try {
              const block = r.querySelector('[data-type]');
              if (block && block.getAttribute) selectedType = String(block.getAttribute('data-type') || '');
            } catch (e) {}
          }

          const rule = { type: String(selectedType || '') };
          const tfEl = r.querySelector('.rule-tf') || r.querySelector('[data-tf]') || r.querySelector('select[name="tf"]'); if (tfEl && tfEl.value && tfEl.value !== 'default') rule.tf = tfEl.value;
          const validEl = r.querySelector('.rule-valid') || r.querySelector('[data-valid]') || r.querySelector('input[name="valid_for_bars"]'); if (validEl && validEl.value !== '') { const n = Number(validEl.value); if (Number.isFinite(n)) rule.valid_for_bars = n; }

          // Only serialize params from the active per-type block to avoid
          // polluting the rule object with defaults from hidden blocks.
          // Always include base fields (type/side/tf/valid_for_bars).
          let activeBlock = null;
          try {
            if (selectedType) {
              const safe = String(selectedType).replace(/\\/g, '\\\\').replace(/"/g, '\\"');
              activeBlock = r.querySelector('[data-type="' + safe + '"]');
            }
          } catch (e) { activeBlock = null; }
          const baseKeys = { type: true, side: true, tf: true, valid_for_bars: true };

          const params = r.querySelectorAll('input, select, textarea');
          for (const p of Array.from(params)) {
            try {
              const k = p.getAttribute && (p.getAttribute('data-key') || p.getAttribute('data-field') || p.getAttribute('name'));
              if (!k) continue;
              if (k === 'type') continue; // type is handled explicitly above
              const inActiveBlock = !!(activeBlock && activeBlock.contains && activeBlock.contains(p));
              if (!inActiveBlock && !baseKeys[k]) continue;
              let v = null;
              // checkboxes -> boolean
              if (p.type === 'checkbox') v = !!p.checked;
              // multi-selects -> array of selected values
              else if (p.tagName === 'SELECT' && p.multiple) {
                const vals = [];
                for (const o of Array.from(p.options)) { if (o.selected) vals.push(o.value); }
                v = vals;
              } else {
                v = p.value;
                // If a select is temporarily unselected (selectedIndex=-1),
                // default to its first option rather than serializing "".
                try {
                  if (p.tagName === 'SELECT' && (v === '' || v === null || v === undefined) && typeof p.selectedIndex === 'number' && p.selectedIndex < 0) {
                    const opt0 = (p.options && p.options.length) ? p.options[0] : null;
                    if (opt0 && opt0.value !== undefined) v = opt0.value;
                  }
                } catch (ee) {}
              }
              // coerce numeric inputs to numbers when possible
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

  // --- Validation helpers for trade_filters ---
  function isTimeStringValid(s) {
    if (!s) return false;
    return /^([01]\d|2[0-3]):[0-5]\d$/.test(String(s));
  }

  function showValidationErrors(form, errors) {
    try {
      if (!form) return;
      let box = form.querySelector('#guided-v2-validation-errors');
      if (!box) {
        box = el('div', { id: 'guided-v2-validation-errors', class: 'validation-errors' });
        if (form.firstChild) form.insertBefore(box, form.firstChild);
        else form.appendChild(box);
      }
      box.innerHTML = '';
      const title = el('div', { class: 'validation-title', text: 'Validation errors:' }); box.appendChild(title);
      const ul = el('ul');
      for (const e of (errors || [])) { const li = el('li', { text: e }); ul.appendChild(li); }
      box.appendChild(ul);
      try { box.scrollIntoView && box.scrollIntoView({ behavior: 'smooth', block: 'center' }); } catch (e) {}
    } catch (e) { console.error('showValidationErrors', e); }
  }

  function removeValidationDisplay(form) {
    try {
      if (!form) return;
      const box = form.querySelector('#guided-v2-validation-errors'); if (box) box.remove();
    } catch (e) {}
  }

  function validateAllRules(form) {
    try {
      const errs = [];
      const tradeContainer = document.getElementById('trade-filters');
      const rules = canonicalSerialize(tradeContainer) || [];
      for (let i = 0; i < rules.length; i++) {
        const r = rules[i] || {};
        const which = `Filter #${i+1} (${r.type || 'unknown'})`;
        if (!r.type) { errs.push(which + ': missing type'); continue; }
        switch (r.type) {
          case 'session_window':
          case 'blackout_window':
            if (!isTimeStringValid(r.start_time)) errs.push(which + ': invalid or missing start_time (HH:MM)');
            if (!isTimeStringValid(r.end_time)) errs.push(which + ': invalid or missing end_time (HH:MM)');
            if (r.days) {
              const ds = Array.isArray(r.days) ? r.days : String(r.days).split(',').map(s => s.trim()).filter(Boolean);
              for (const d of ds) { if (WEEKDAYS.indexOf(d) === -1) errs.push(which + `: invalid weekday '${d}'`); }
            }
            if (r.timezone && TIMEZONES.indexOf(r.timezone) === -1) errs.push(which + `: unknown timezone '${r.timezone}'`);
            break;
          case 'day_month':
            if (r.days) {
              const ds2 = Array.isArray(r.days) ? r.days : String(r.days).split(',').map(s => s.trim()).filter(Boolean);
              for (const d of ds2) { if (WEEKDAYS.indexOf(d) === -1) errs.push(which + `: invalid weekday '${d}'`); }
            }
            if (r.months) {
              const ms = Array.isArray(r.months) ? r.months : String(r.months).split(',').map(s => s.trim()).filter(Boolean);
              for (const m of ms) { if (MONTHS.indexOf(m) === -1) errs.push(which + `: invalid month '${m}'`); }
            }
            break;
          default:
            // custom or unknown types have no validation here
            break;
        }
      }
      return errs;
    } catch (e) { return ['validation: internal error']; }
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

    // Trade filter parameter UIs
    if (type === 'session_window') {
      const start = el('input', { type: 'time', class: 'param', 'data-key': 'start_time', value: (rule && rule.start_time) ? rule.start_time : '00:00' });
      const end = el('input', { type: 'time', class: 'param', 'data-key': 'end_time', value: (rule && rule.end_time) ? rule.end_time : '23:59' });
      const days = el('select', { class: 'param', 'data-key': 'days', multiple: 'multiple', size: 4 });
      for (const d of WEEKDAYS) { const o = el('option', { value: d, text: d }); days.appendChild(o); }
      // populate selected days if present (accept array or csv)
      try {
        const selDays = (rule && rule.days) ? (Array.isArray(rule.days) ? rule.days : String(rule.days).split(',').map(s => s.trim())) : [];
        for (const opt of Array.from(days.options)) { if (selDays.includes(opt.value)) opt.selected = true; }
      } catch (e) {}
      const tz = el('input', { type: 'text', class: 'param', 'data-key': 'timezone', list: 'guided-v2-timezone-datalist', placeholder: '(default / exchange)' });
      if (rule && rule.timezone) tz.value = rule.timezone;
      container.appendChild(el('label', { text: 'Start (HH:MM): ' })); container.appendChild(start);
      container.appendChild(el('label', { text: ' End (HH:MM): ' })); container.appendChild(end);
      container.appendChild(el('label', { text: ' Days: ' })); container.appendChild(days);
      container.appendChild(el('label', { text: ' Timezone: ' })); container.appendChild(tz);
      container.appendChild(el('div', { class: 'muted', html: 'Pick weekdays (multi-select) and optional timezone to interpret times.' }));
      return;
    }

    if (type === 'blackout_window') {
      const start = el('input', { type: 'time', class: 'param', 'data-key': 'start_time', value: (rule && rule.start_time) ? rule.start_time : '00:00' });
      const end = el('input', { type: 'time', class: 'param', 'data-key': 'end_time', value: (rule && rule.end_time) ? rule.end_time : '00:00' });
      const days = el('select', { class: 'param', 'data-key': 'days', multiple: 'multiple', size: 4 });
      for (const d of WEEKDAYS) { const o = el('option', { value: d, text: d }); days.appendChild(o); }
      try {
        const selDays = (rule && rule.days) ? (Array.isArray(rule.days) ? rule.days : String(rule.days).split(',').map(s => s.trim())) : [];
        for (const opt of Array.from(days.options)) { if (selDays.includes(opt.value)) opt.selected = true; }
      } catch (e) {}
      const tz = el('input', { type: 'text', class: 'param', 'data-key': 'timezone', list: 'guided-v2-timezone-datalist', placeholder: '(default / exchange)' });
      if (rule && rule.timezone) tz.value = rule.timezone;
      container.appendChild(el('label', { text: 'Start (HH:MM): ' })); container.appendChild(start);
      container.appendChild(el('label', { text: ' End (HH:MM): ' })); container.appendChild(end);
      container.appendChild(el('label', { text: ' Days: ' })); container.appendChild(days);
      container.appendChild(el('label', { text: ' Timezone: ' })); container.appendChild(tz);
      container.appendChild(el('div', { class: 'muted', html: 'Blackout windows disallow entries during the specified times.' }));
      return;
    }

    if (type === 'day_month') {
      const days = el('input', { type: 'text', class: 'param', 'data-key': 'days', placeholder: 'Mon,Tue,Wed', value: (rule && rule.days) ? rule.days : '' });
      const months = el('input', { type: 'text', class: 'param', 'data-key': 'months', placeholder: 'Jan,Feb', value: (rule && rule.months) ? rule.months : '' });
      container.appendChild(el('label', { text: 'Days (e.g. Mon,Tue): ' })); container.appendChild(days);
      container.appendChild(el('label', { text: ' Months (e.g. Jan,Feb): ' })); container.appendChild(months);
      container.appendChild(el('div', { class: 'muted', html: 'Filter entries by day-of-week and month.' }));
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
      if (p.tagName === 'SELECT') {
        const sel = p;
        if (sel.multiple) {
          const vals = Array.from(sel.selectedOptions).map(o => o.value);
          params[key] = vals;
        } else {
          params[key] = sel.value;
        }
      } else if (p.tagName === 'INPUT' || p.tagName === 'TEXTAREA') {
        if (p.type === 'number') {
          const n = Number(p.value);
          params[key] = Number.isFinite(n) ? n : p.value;
        } else if (p.type === 'checkbox') {
          params[key] = !!p.checked;
        } else {
          params[key] = p.value;
        }
      }
    }
    return params;
  }

  // Legacy `createRuleRow` removed — require the full `ruleRow` renderer.
  // Intentionally do not provide a fallback here so missing renderer
  // will surface errors during testing/deployment.

  function renderRulesInto(container, rules, section) {
    if (!container) return;
    container.innerHTML = '';
    // Use the canonical `ruleRow` renderer when available. Attempt to
    // resolve from the local binding first, then `window.ruleRow` as a
    // fallback. If neither is available, skip inserting rows so we do
    // not render empty placeholder DIVs that confuse the UI.
    const renderer = (typeof ruleRow === 'function') ? ruleRow : (typeof window !== 'undefined' && typeof window.ruleRow === 'function' ? window.ruleRow : null);
    if (!renderer) return;
    for (const r of (rules || [])) {
      try {
        const row = renderer(section, r);
        if (row) container.appendChild(row);
      } catch (e) { /* skip faulty render */ }
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
      const sec = String(section || '');
      const stateKey = (sec === 'trade_filters') ? 'trade_filters' : (sec + '_rules');
      const hiddenId = (sec === 'trade_filters') ? 'trade_filters_json' : (stateKey + '_json');
      // Support container ids using either underscore or hyphen (e.g.
      // `trade_filters` -> `trade-filters`) to remain tolerant to
      // template changes where the DOM id may use dashes.
      // Resolve the container deterministically. Historically there was
      // inconsistency between `trade_filters` (internal section name)
      // and the DOM id `trade-filters`. Prefer explicit mapping for the
      // `trade_filters` section to avoid races where the wrong id is
      // probed and an empty state causes a subsequent re-render to
      // wipe user-added rows.
      let container = null;
      try {
        if (String(section) === 'trade_filters') {
          container = document.getElementById('trade-filters') || document.getElementById('trade_filters') || document.getElementById('trade-filters-rules') || document.getElementById('trade_filters-rules');
        } else {
          const candidates = [section + '-rules', String(section).replace(/_/g, '-') + '-rules', String(section).replace(/_/g, '-'), section];
          for (const id of candidates) {
            try { const el = document.getElementById(id); if (el) { container = el; break; } } catch (e) {}
          }
        }
      } catch (e) { container = null; }
      if (!container) return;
      const rules = canonicalSerialize(container);
      try {
        if (window.GuidedBuilderV2State) window.GuidedBuilderV2State.set(stateKey, rules);
      } catch (e) { console.error('commitSectionFromDOM state set failed', e); }
      try { if (window.GuidedBuilderV2State && typeof window.GuidedBuilderV2State.flush === 'function') { window.GuidedBuilderV2State.flush(); } } catch (e) {}
      // Always update the corresponding hidden input as a fallback so
      // non-state code paths (or tests) observe the latest DOM immediately.
      try {
        const formEl = document.querySelector('form[action="/create-strategy-guided/step3"]') || document.querySelector('form[action="/create-strategy-guided/step4"]') || document.getElementById('guided_step4_form');
        if (formEl) {
          ensureHidden(formEl, hiddenId);
          writeHiddenStamped(formEl, hiddenId, JSON.stringify(rules), 'commit-dom');
        }
      } catch (e) { /* best-effort */ }
    } catch (e) { console.error('commitSectionFromDOM', e); }
  }

  // Expose a narrow commit helper for other renderer variants in this file.
  // Some modules run in separate IIFE scopes and cannot reference this
  // function directly, but they still need to commit deletes/adds to the
  // state manager so rows don't get re-rendered back.
  try {
    if (typeof window !== 'undefined') {
      window.__guided_v2_commitSectionFromDOM = commitSectionFromDOM;
    }
  } catch (e) { /* ignore */ }

  // Global capture-phase remove wiring.
  // Why: the advanced renderer attaches per-row Remove handlers, but a
  // focus/blur-triggered re-render can replace the row synchronously during
  // pointer/click interactions. That can cause the Remove handler to act on a
  // stale node reference and then commit a DOM that still contains the rule.
  // Installing a capture-phase handler at document level ensures we remove the
  // *currently clicked* row and commit before blur/rerender handlers run.
  function wireGlobalRemoveCapture() {
    try {
      if (typeof document === 'undefined' || !document || !document.addEventListener) return;
      const root = document.documentElement || document.body;
      if (!root || !root.dataset) return;
      if (root.dataset.guidedV2GlobalRemoveCapture === '1') return;
      root.dataset.guidedV2GlobalRemoveCapture = '1';

      const allowedContainerSelector = '#context-rules, #signal-rules, #trigger-rules, #trade-filters, #trade_filters, #trade-filters-rules, #trade_filters-rules';

      function inferSection(row, container) {
        try {
          const ds = row && row.getAttribute ? row.getAttribute('data-section') : '';
          if (ds) return String(ds);
        } catch (e) {}
        try {
          const id = container && container.id ? String(container.id) : '';
          if (id === 'context-rules' || id === 'context_rules') return 'context';
          if (id === 'signal-rules' || id === 'signal_rules') return 'signal';
          if (id === 'trigger-rules' || id === 'trigger_rules') return 'trigger';
          if (id.indexOf('trade') !== -1) return 'trade_filters';
        } catch (e) {}
        return '';
      }

      function isRemoveControl(btn) {
        try {
          if (!btn) return false;
          const cls = btn.classList;
          const isRemoveClass = !!(cls && cls.contains && cls.contains('rule-remove'));
          const isDangerClass = !!(cls && cls.contains && cls.contains('danger'));
          const txt = ((btn.textContent || '').trim().toLowerCase());
          const aria = (((btn.getAttribute && btn.getAttribute('aria-label')) || '').trim().toLowerCase());
          const dataAction = (((btn.getAttribute && btn.getAttribute('data-action')) || '').trim().toLowerCase());
          const isRemoveText = txt === 'remove';
          const isRemoveAria = aria === 'remove';
          const isRemoveAction = dataAction === 'remove' || dataAction === 'delete';
          // Require either an explicit remove marker or the exact word
          // "remove" to avoid catching other danger buttons.
          return (isRemoveClass || isRemoveAction || isRemoveText || isRemoveAria) && (isDangerClass || isRemoveText || isRemoveAria || isRemoveAction);
        } catch (e) {
          return false;
        }
      }

      function handleRemove(ev) {
        try {
          const tgt = ev && ev.target && ev.target.closest ? ev.target.closest('button, a, [role="button"]') : null;
          if (!tgt) return;
          if (!isRemoveControl(tgt)) return;

          const row = tgt.closest ? tgt.closest('.rule-row') : null;
          if (!row) return;

          const container = row.closest ? row.closest(allowedContainerSelector) : null;
          if (!container) return;

          // Capture a stable address for the row before any blur-triggered
          // handlers can replace/move DOM nodes. We'll attempt immediate
          // removal, and also a deferred removal that re-queries the
          // container to handle cases where the row node was replaced.
          const section = inferSection(row, container);
          let rowIndex = -1;
          try {
            const rows = Array.from(container.querySelectorAll('.rule-row'));
            rowIndex = rows.indexOf(row);
          } catch (e) { rowIndex = -1; }

          // Prevent blur/click handlers from racing and re-rendering the row
          // before we commit the removal.
          try { ev.preventDefault && ev.preventDefault(); } catch (e) {}
          try { ev.stopImmediatePropagation && ev.stopImmediatePropagation(); } catch (e) { try { ev.stopPropagation && ev.stopPropagation(); } catch (ee) {} }

          const commitNow = () => {
            try {
              const commitFn = (typeof window !== 'undefined' && typeof window.__guided_v2_commitSectionFromDOM === 'function') ? window.__guided_v2_commitSectionFromDOM : null;
              if (commitFn && section) commitFn(section);
            } catch (e) {}
          };

          const removeRowNode = (node) => {
            try { if (node && node.remove) { node.remove(); return true; } } catch (e) {}
            try { if (node && node.parentNode && node.parentNode.removeChild) { node.parentNode.removeChild(node); return true; } } catch (e) {}
            return false;
          };

          // Immediate attempt (works when the row wasn't replaced).
          try { removeRowNode(row); } catch (e) {}
          commitNow();

          // Deferred attempt: after blur/focusout handlers have run, the row
          // may have been replaced. Remove by index from the current DOM.
          const deferred = () => {
            try {
              if (!section) return;
              let container2 = null;
              if (section === 'trade_filters') {
                container2 = document.getElementById('trade-filters') || document.getElementById('trade_filters') || document.getElementById('trade-filters-rules') || document.getElementById('trade_filters-rules');
              } else {
                container2 = document.getElementById(section + '-rules') || document.getElementById(String(section).replace(/_/g, '-') + '-rules') || document.getElementById(String(section).replace(/_/g, '-')) || document.getElementById(section);
              }
              if (!container2) return;
              const rows2 = Array.from(container2.querySelectorAll('.rule-row'));
              if (rows2.length === 0) return;
              const idx = (typeof rowIndex === 'number' && rowIndex >= 0) ? rowIndex : 0;
              const candidate = rows2[idx] || rows2[0];
              removeRowNode(candidate);
              commitNow();
            } catch (e) {}
          };
          try { setTimeout(deferred, 0); } catch (e) {}
          try { if (typeof requestAnimationFrame === 'function') requestAnimationFrame(() => { try { deferred(); } catch (e) {} }); } catch (e) {}
        } catch (e) {}
      }

      // pointerdown/mousedown happens before focus shifts, reducing blur races.
      document.addEventListener('pointerdown', handleRemove, true);
      document.addEventListener('mousedown', handleRemove, true);
      document.addEventListener('click', handleRemove, true);
    } catch (e) {
      // best-effort
    }
  }

  try { wireGlobalRemoveCapture(); } catch (e) {}

  function syncHiddenFromState(form, state) {
    try {
      const ctx = ensureHidden(form, 'context_rules_json');
      const sig = ensureHidden(form, 'signal_rules_json');
      const trg = ensureHidden(form, 'trigger_rules_json');
      const trade = ensureHidden(form, 'trade_filters_json');
      try {
        writeHiddenStamped(form, 'context_rules_json', JSON.stringify((state && state.context_rules) ? state.context_rules : []), 'state-sync');
      } catch (e) { writeHiddenStamped(form, 'context_rules_json', '[]', 'state-sync-fallback'); }
      try {
        writeHiddenStamped(form, 'signal_rules_json', JSON.stringify((state && state.signal_rules) ? state.signal_rules : []), 'state-sync');
      } catch (e) { writeHiddenStamped(form, 'signal_rules_json', '[]', 'state-sync-fallback'); }
      try {
        writeHiddenStamped(form, 'trigger_rules_json', JSON.stringify((state && state.trigger_rules) ? state.trigger_rules : []), 'state-sync');
      } catch (e) { writeHiddenStamped(form, 'trigger_rules_json', '[]', 'state-sync-fallback'); }
      try {
        writeHiddenStamped(form, 'trade_filters_json', JSON.stringify((state && state.trade_filters) ? state.trade_filters : []), 'state-sync');
      } catch (e) { writeHiddenStamped(form, 'trade_filters_json', '[]', 'state-sync-fallback'); }
    } catch (e) { console.error('syncHiddenFromState', e); }
  }

  // Fallback DOM -> hidden synchronizer used by the minimal/advanced preview path.
  function syncHidden(form, ctxContainer, sigContainer, trgContainer, tradeContainer) {
    try {
      if (!form) return;
      const ctxH = ensureHidden(form, 'context_rules_json');
      const sigH = ensureHidden(form, 'signal_rules_json');
      const trgH = ensureHidden(form, 'trigger_rules_json');
      const tradeH = ensureHidden(form, 'trade_filters_json');
      try {
        writeHiddenStamped(form, 'context_rules_json', JSON.stringify(canonicalSerialize(ctxContainer)), 'dom-sync');
      } catch (e) { writeHiddenStamped(form, 'context_rules_json', '[]', 'dom-sync-fallback'); }
      try {
        writeHiddenStamped(form, 'signal_rules_json', JSON.stringify(canonicalSerialize(sigContainer)), 'dom-sync');
      } catch (e) { writeHiddenStamped(form, 'signal_rules_json', '[]', 'dom-sync-fallback'); }
      try {
        writeHiddenStamped(form, 'trigger_rules_json', JSON.stringify(canonicalSerialize(trgContainer)), 'dom-sync');
      } catch (e) { writeHiddenStamped(form, 'trigger_rules_json', '[]', 'dom-sync-fallback'); }
      try {
        writeHiddenStamped(form, 'trade_filters_json', JSON.stringify(canonicalSerialize(tradeContainer)), 'dom-sync');
      } catch (e) { writeHiddenStamped(form, 'trade_filters_json', '[]', 'dom-sync-fallback'); }
    } catch (e) { /* best-effort */ }
  }

  function init() {
    const form = document.querySelector('form[action="/create-strategy-guided/step3"]') || document.querySelector('form[action="/create-strategy-guided/step4"]') || document.getElementById('guided_step4_form');
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
            syncHidden(f, ctx, sig, trg, document.getElementById('trade-filters'));
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
          // Debug-only: this hook can generate extra pre-submit activity.
          // Keep it disabled in normal operation to avoid duplicate requests.
          if (!GUIDED_V2_DEBUG) return;
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
                  const tradeContainer = f.querySelector ? f.querySelector('#trade-filters') : document.getElementById('trade-filters');
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
                              const tradeHidden = f.querySelector && f.querySelector('#trade_filters_json');
                              const tradeVal = tradeHidden && tradeHidden.value !== undefined ? parseJsonSafe(tradeHidden.value) : canonicalSerialize(tradeContainer);
                              const payload = { draft_id: draftId, context_rules: ctxVal, signal_rules: sigVal, trigger_rules: trgVal, trade_filters: tradeVal };
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
                          try { writeHiddenStamped(f, 'trade_filters_json', JSON.stringify(canonicalSerialize(document.getElementById('trade-filters'))), 'final-pre-submit'); } catch (e) {}
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
              const newForm = (n.querySelector && (n.querySelector('form[action="/create-strategy-guided/step3"]') || n.querySelector('form[action="/create-strategy-guided/step4"]'))) || (n.id === 'guided_step4_form' ? n : null);
              if (newForm) {
                (function bindAgain(f) {
                  try {
                    const origSubmit = f.submit && f.submit.bind(f);
                    const origRequestSubmit = f.requestSubmit && f.requestSubmit.bind(f);
                    if (!origSubmit && !origRequestSubmit) return;
                    // reuse same doSync logic
                    const doSync = function () {
                      try { if (window.GuidedBuilderV2State) { window.GuidedBuilderV2State.flush && window.GuidedBuilderV2State.flush(); syncHiddenFromState(f, window.GuidedBuilderV2State.raw ? window.GuidedBuilderV2State.raw() : {}); } } catch (e) {}
                      try { syncHidden(f, document.getElementById('context-rules'), document.getElementById('signal-rules'), document.getElementById('trigger-rules'), document.getElementById('trade-filters')); } catch (e) {}
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
          const tradeContainer2 = document.getElementById('trade-filters');
          let ctxH = document.getElementById('context_rules_json'); if (!ctxH) ctxH = ensureHidden(form, 'context_rules_json');
          let sigH = document.getElementById('signal_rules_json'); if (!sigH) sigH = ensureHidden(form, 'signal_rules_json');
          let trgH = document.getElementById('trigger_rules_json'); if (!trgH) trgH = ensureHidden(form, 'trigger_rules_json');
          let tradeH = document.getElementById('trade_filters_json'); if (!tradeH) tradeH = ensureHidden(form, 'trade_filters_json');
              try {
                // use stamped writes so we can trace who last changed the inputs
                writeHiddenStamped(form, 'context_rules_json', JSON.stringify(canonicalSerialize(ctxContainer2)), 'capture-submit-dom');
                writeHiddenStamped(form, 'signal_rules_json', JSON.stringify(canonicalSerialize(sigContainer2)), 'capture-submit-dom');
                writeHiddenStamped(form, 'trigger_rules_json', JSON.stringify(canonicalSerialize(trgContainer2)), 'capture-submit-dom');
                writeHiddenStamped(form, 'trade_filters_json', JSON.stringify(canonicalSerialize(tradeContainer2)), 'capture-submit-dom');
                // validate rules and halt submission if issues found
                const validationErrors = validateAllRules(form);
                if (validationErrors && validationErrors.length > 0) {
                  try { writeHiddenStamped(form, 'guided_v2_validation_errors', JSON.stringify(validationErrors), 'capture-submit-validate'); } catch (e) {}
                  try { showValidationErrors(form, validationErrors); } catch (e) {}
                  try { ev.preventDefault(); ev.stopImmediatePropagation && ev.stopImmediatePropagation(); } catch (e) {}
                  return;
                } else {
                  try { removeValidationDisplay(form); } catch (e) {}
                }
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
          try { window.__guided_v2_ensure_submit && window.__guided_v2_ensure_submit(f); } catch (e) {}
          try { f.submit(); } catch (e) {}
        }
      } catch (e) {
        try { window.__guided_v2_ensure_submit && window.__guided_v2_ensure_submit(f); } catch (e2) {}
        try { f.submit(); } catch (e2) { /* ignore */ }
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
      ensureHidden(form, 'context_rules_json'); ensureHidden(form, 'signal_rules_json'); ensureHidden(form, 'trigger_rules_json'); ensureHidden(form, 'trade_filters_json');
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
            syncHidden(form, ctxContainer, sigContainer, trgContainer, document.getElementById('trade-filters'));
          }
          // Additionally, always attempt to sync from DOM rule containers
          // to cover cases where rules exist in the DOM but weren't yet
          // reflected in the state manager.
          try {
            const ctxContainer2 = document.getElementById('context-rules');
            const sigContainer2 = document.getElementById('signal-rules');
            const trgContainer2 = document.getElementById('trigger-rules');
            syncHidden(form, ctxContainer2, sigContainer2, trgContainer2, document.getElementById('trade-filters'));
            try { console.log('[guided-debug] minimal submit sync ->', { ctx: (document.getElementById('context_rules_json')||{}).value, sig: (document.getElementById('signal_rules_json')||{}).value, trg: (document.getElementById('trigger_rules_json')||{}).value }); } catch (e) {}
          } catch (e) { /* best-effort */ }
        } catch (e) { console.error('submit sync (minimal) failed', e); }
      });
      // NOTE: do not intercept submit clicks here. The capture-phase submit
      // handler plus `__guided_v2_ensure_submit` cover DOM->hidden sync.

    }

    // NOTE: removed a second capture-phase submit handler which prevented
    // default and performed a native submit. That handler caused duplicate
    // POSTs and could surface 422s in the console.

    const ctxContainer = document.getElementById('context-rules');
    const sigContainer = document.getElementById('signal-rules');
    const trgContainer = document.getElementById('trigger-rules');
    const tradeContainer = document.getElementById('trade-filters') || document.getElementById('trade_filters') || document.getElementById('trade-filters-rules') || document.getElementById('trade_filters-rules');

    ensureHidden(form, 'context_rules_json'); ensureHidden(form, 'signal_rules_json'); ensureHidden(form, 'trigger_rules_json'); ensureHidden(form, 'trade_filters_json');

    // initial load: prefer state if present, else hidden inputs
    const initialState = (window.GuidedBuilderV2State && typeof window.GuidedBuilderV2State.get === 'function') ? window.GuidedBuilderV2State.get() : null;
    const initial = initialState || {};

    // if state has no rules, read hidden inputs
    if (!initial || !initial.context_rules || !initial.signal_rules || !initial.trigger_rules) {
      initial.context_rules = parseJsonSafe(form.querySelector('#context_rules_json')?.value);
      initial.signal_rules = parseJsonSafe(form.querySelector('#signal_rules_json')?.value);
      initial.trigger_rules = parseJsonSafe(form.querySelector('#trigger_rules_json')?.value);
      initial.trade_filters = parseJsonSafe(form.querySelector('#trade_filters_json')?.value);
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
          if (!(active && tradeContainer && tradeContainer.contains(active))) {
            renderRulesInto(tradeContainer, s.trade_filters || [], 'trade_filters');
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

    // wire add buttons. Use a small retry strategy when the canonical
    // `ruleRow` renderer may not yet be bound in the execution scope to
    // avoid inserting empty placeholder DIVs.
    function getRenderer() {
      if (typeof ruleRow === 'function') return ruleRow;
      if (typeof window !== 'undefined' && typeof window.ruleRow === 'function') return window.ruleRow;
      return null;
    }

    function appendRowWithRetry(section, container, maxAttempts = 8) {
      let attempts = 0;
      const tryAppend = () => {
        attempts += 1;
        const renderer = getRenderer();
        if (renderer) {
          try {
            const row = renderer(section, {});
            if (row && container) container.appendChild(row);
            // Some renderers populate default select values (notably
            // `select[data-field="type"]`) on the next frame. Commit after
            // a rAF to avoid serializing transient empty values.
            try { afterNextFrame(() => { try { commitSectionFromDOM(section); } catch (e) {} }); } catch (e) { try { commitSectionFromDOM(section); } catch (ee) {} }
            return;
          } catch (e) { /* fallthrough to retry */ }
        }
        if (attempts < maxAttempts) {
          try { setTimeout(tryAppend, 60 * attempts); } catch (e) {}
        } else {
          // give up silently to avoid UI noise; do not append empty nodes
        }
      };
      tryAppend();
    }

    // If another script has already wired the add buttons, our per-button
    // wiring below may intentionally skip (dataset guard) to avoid
    // duplicating inserts. In that case, the other handler may append DOM
    // rows without committing them into `GuidedBuilderV2State`, so the next
    // state-driven re-render can wipe the DOM. Install a small shim that
    // commits from DOM after any add click, regardless of who inserted.
    try {
      if (form && form.dataset && !form.dataset.guidedV2AddCommitShim) {
        form.dataset.guidedV2AddCommitShim = '1';
        form.addEventListener('click', (ev) => {
          try {
            const t = ev && ev.target && ev.target.closest ? ev.target.closest('#add-context, #add-signal, #add-trigger, #add-trade-filter') : null;
            if (!t || !t.id) return;
            const section = (t.id === 'add-context') ? 'context' : (t.id === 'add-signal') ? 'signal' : (t.id === 'add-trigger') ? 'trigger' : (t.id === 'add-trade-filter') ? 'trade_filters' : null;
            if (!section) return;
            // Defer to allow the actual add handler (whoever owns it) to
            // append the row first and populate default values.
            try { afterNextFrame(() => { try { commitSectionFromDOM(section); } catch (e) {} }); } catch (e) { setTimeout(() => { try { commitSectionFromDOM(section); } catch (ee) {} }, 0); }
          } catch (e) {}
        }, false);
      }
    } catch (e) {}

    const addCtx = document.getElementById('add-context');
    const addSig = document.getElementById('add-signal');
    const addTrg = document.getElementById('add-trigger');
    if (addCtx && ctxContainer && !addCtx.dataset.guidedV2Wired) {
      addCtx.addEventListener('click', (ev) => { try { ev.preventDefault(); } catch (e) {} ; appendRowWithRetry('context', ctxContainer); });
      addCtx.dataset.guidedV2Wired = '1';
    }
    if (addSig && sigContainer && !addSig.dataset.guidedV2Wired) {
      addSig.addEventListener('click', (ev) => { try { ev.preventDefault(); } catch (e) {} ; appendRowWithRetry('signal', sigContainer); });
      addSig.dataset.guidedV2Wired = '1';
    }
    if (addTrg && trgContainer && !addTrg.dataset.guidedV2Wired) {
      addTrg.addEventListener('click', (ev) => { try { ev.preventDefault(); } catch (e) {} ; appendRowWithRetry('trigger', trgContainer); });
      addTrg.dataset.guidedV2Wired = '1';
    }
    const addTrade = document.getElementById('add-trade-filter');
    if (addTrade && tradeContainer && !addTrade.dataset.guidedV2Wired) {
      addTrade.addEventListener('click', (ev) => { try { ev.preventDefault(); } catch (e) {} ; appendRowWithRetry('trade_filters', tradeContainer); });
      addTrade.dataset.guidedV2Wired = '1';
    }

    // edits/removes: delegate to container and commit on change
    const delegateEvents = (container, section) => {
      if (!container) return;
      container.addEventListener('input', () => commitSectionFromDOM(section));
      container.addEventListener('change', () => commitSectionFromDOM(section));
      container.addEventListener('click', (ev) => {
        try {
          // Normalize target to the nearest button in case inner elements
          // (icons/spans) are clicked. Accept either explicit
          // `.rule-remove` marker or buttons whose visible text is
          // "Remove" to handle alternate renderer variants.
          const tgt = ev.target && ev.target.closest ? ev.target.closest('button, a, [role="button"]') : ev.target;
          if (!tgt) return;
          const isRemoveClass = tgt.classList && tgt.classList.contains && tgt.classList.contains('rule-remove');
          const isDangerClass = tgt.classList && tgt.classList.contains && tgt.classList.contains('danger');
          const isRemoveText = (tgt.textContent || '').trim().toLowerCase() === 'remove';
          const isRemoveAria = ((tgt.getAttribute && tgt.getAttribute('aria-label')) || '').trim().toLowerCase() === 'remove';
          if (isRemoveClass || isDangerClass || isRemoveText || isRemoveAria) {
            try { ev && ev.preventDefault && ev.preventDefault(); } catch (e) {}
            try {
              const row = (tgt && tgt.closest) ? (tgt.closest('.rule-row') || tgt.closest('.card')) : null;
              if (row) {
                try { row.remove(); } catch (e) { try { row.parentNode && row.parentNode.removeChild && row.parentNode.removeChild(row); } catch (ee) {} }
              }
            } catch (e) {}
            commitSectionFromDOM(section);
          }
        } catch (e) {}
      });
      // On focusout (blur from any element within the container), ensure
      // the latest edits are committed to the state immediately so a
      // subsequent re-render doesn't overwrite them.
      container.addEventListener('focusout', (ev) => {
        try { commitSectionFromDOM(section); } catch (e) { console.error('focusout commit failed', e); }
      });
    };
    delegateEvents(ctxContainer, 'context'); delegateEvents(sigContainer, 'signal'); delegateEvents(trgContainer, 'trigger'); delegateEvents(tradeContainer, 'trade_filters');

    // ensure hidden sync on submit
    form.addEventListener('submit', () => { if (window.GuidedBuilderV2State) { window.GuidedBuilderV2State.flush && window.GuidedBuilderV2State.flush(); const s = window.GuidedBuilderV2State.raw ? window.GuidedBuilderV2State.raw() : {}; syncHiddenFromState(form, s); } else { syncHidden(form, ctxContainer, sigContainer, trgContainer, document.getElementById('trade-filters')); } });
    // NOTE: removed custom submit-button click handler. Rely on the submit
    // event and `__guided_v2_ensure_submit` for serialization.
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();

  // NOTE: Removed test-only global API exposure to avoid accidental
  // reliance on a secondary renderer. Tests should prefer the public
  // `createRuleRow` function when needed.

})();

// NOTE: Global fallback wiring for the Add Trade Filter button intentionally
// removed. Trade-filter add buttons are wired within the main `init()`
// path alongside context/signal/trigger (see the per-section `addTrade`
// wiring in `init`) so that behavior is consistent and not globally
// visible across IIFE scopes.
    (function () {
      const form = document.querySelector('form[action="/create-strategy-guided/step3"]') || document.querySelector('form[action="/create-strategy-guided/step4"]') || document.getElementById('guided_step4_form');
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
      const tradeHidden = document.getElementById('trade_filters_json');
      const ctxContainer = document.getElementById('context-rules');
      const sigContainer = document.getElementById('signal-rules');
      const trgContainer = document.getElementById('trigger-rules');
      const tradeContainer = document.getElementById('trade-filters');
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
          ['bollinger_context', 'Bollinger'],
          ['relative_volume', 'Relative Volume (RVOL)'],
          ['volume_osc_increase', 'Volume Oscillator Increase'],
          ['volume_above_ma', 'Volume Above MA'],
          ['custom', 'Custom (DSL)']
        ];
        // Preserve current value if present
        const cur = String(primaryContextSel.value || '').trim();
        primaryContextSel.innerHTML = '';
        for (const [v, label] of opts) {
          const o = document.createElement('option'); o.value = v; o.textContent = label; 
          if (v === 'bollinger_context') {
            o.title = 'Uses the Context TF (selected above) for band calculation; this is a higher-timeframe/context filter.';
          }
          primaryContextSel.appendChild(o);
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
        // Per-rule TF applies to all sections (signal/trigger/context).
        if (section === 'signal' || section === 'trigger' || section === 'context') {
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
          if (value === 'bollinger_context') {
            o.title = 'Uses the Context TF (selected in the form) for band calculation; acts as a higher-timeframe/context filter.';
          }
          typeSel.appendChild(o);
        }

        const removeBtn = el('button', { type: 'button', class: 'danger', text: 'Remove' });
        removeBtn.addEventListener('click', (ev) => {
          try { ev && ev.stopPropagation && ev.stopPropagation(); ev && ev.preventDefault && ev.preventDefault(); } catch (e) {}
          try {
            if (row && (typeof row.isConnected === 'boolean' ? row.isConnected : !!row.parentNode)) {
              try { row.remove(); } catch (e) { try { row.parentNode && row.parentNode.removeChild && row.parentNode.removeChild(row); } catch (ee) {} }
            }
          } catch (e) {}
          // Commit synchronously after removal to ensure the state manager
          // reflects the deletion before any subscriber re-renders. This
          // prevents races where an old state snapshot might re-insert the
          // removed row. Keep best-effort fallbacks for environments without
          // `commitSectionFromDOM`.
          try {
            const commitFn = (typeof window !== 'undefined' && typeof window.__guided_v2_commitSectionFromDOM === 'function') ? window.__guided_v2_commitSectionFromDOM : null;
            if (commitFn) {
              try { commitFn(section); } catch (e) {}
            } else if (typeof syncHidden === 'function') {
              // In this renderer scope, `syncHidden` is a local no-arg helper.
              // Do not call it with the other module's signature.
              try { syncHidden(); } catch (e) {}
            }
            // Best-effort cleanup: remove any empty `.rule-row` elements
            // that may have been inserted by a transient fallback or
            // renderer race. Keep this conservative to avoid removing
            // legitimate rows.
            try {
              const parent = row && row.parentNode ? row.parentNode : null;
              const container = parent || (section === 'trade_filters' ? document.getElementById('trade-filters') : document.getElementById(section + '-rules'));
              if (container) {
                const children = Array.from(container.querySelectorAll('.rule-row'));
                for (const ch of children) {
                  try {
                    if (ch && (!ch.innerHTML || String(ch.innerHTML).trim() === '')) {
                      ch.parentNode && ch.parentNode.removeChild && ch.parentNode.removeChild(ch);
                    }
                  } catch (e) {}
                }
              }
            } catch (e) {}
          } catch (e) {}
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
        // Show per-row TF and "Valid bars" for all sections
        if (section === 'signal' || section === 'trigger' || section === 'context') {
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

        function inputTime(parent, label, field, value) {
          parent.appendChild(el('label', { text: label }));
          const inp = el('input');
          inp.type = 'time';
          inp.setAttribute('data-field', field);
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
            b.appendChild(el('div', { class: 'muted', html: 'Bollinger Context: require the <strong>Context TF</strong> price membership relative to Bollinger bands. Uses the Context TF configured above.' }));
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

        // Trade filter blocks
        if (section === 'trade_filters') {
          block('session_window', (b) => {
            inputTime(b, 'Start (HH:MM)', 'start_time', rule?.start_time ?? '00:00');
            inputTime(b, 'End (HH:MM)', 'end_time', rule?.end_time ?? '23:59');
            inputText(b, 'Days (csv; optional)', 'days', rule?.days ?? '', 'Mon,Tue,Wed,Thu,Fri');
            // keep timezone as text + datalist (shared datalist is created earlier in the file)
            b.appendChild(el('label', { text: 'Timezone (optional)' }));
            const tz = el('input', { type: 'text', placeholder: '(default / exchange)' });
            tz.setAttribute('data-field', 'timezone');
            try { tz.setAttribute('list', 'guided-v2-timezone-datalist'); } catch (e) {}
            tz.value = rule?.timezone ?? '';
            b.appendChild(tz);
          });

          block('blackout_window', (b) => {
            inputTime(b, 'Start (HH:MM)', 'start_time', rule?.start_time ?? '00:00');
            inputTime(b, 'End (HH:MM)', 'end_time', rule?.end_time ?? '00:00');
            inputText(b, 'Days (csv; optional)', 'days', rule?.days ?? '', 'Mon,Tue,Wed,Thu,Fri');
            b.appendChild(el('label', { text: 'Timezone (optional)' }));
            const tz = el('input', { type: 'text', placeholder: '(default / exchange)' });
            tz.setAttribute('data-field', 'timezone');
            try { tz.setAttribute('list', 'guided-v2-timezone-datalist'); } catch (e) {}
            tz.value = rule?.timezone ?? '';
            b.appendChild(tz);
          });

          block('max_trades_per_day', (b) => {
            inputNumber(b, 'Max trades per day', 'limit', rule?.limit ?? 1, { min: 0 });
          });

          block('max_exposure', (b) => {
            inputNumber(b, 'Max exposure', 'amount', rule?.amount ?? 1000, { min: 0, step: 0.01 });
            inputText(b, 'Currency', 'currency', rule?.currency ?? 'USD', 'USD');
          });

          block('news_buffer', (b) => {
            inputNumber(b, 'Buffer minutes', 'minutes', rule?.minutes ?? 60, { min: 0, step: 1 });
            inputText(b, 'Sources (optional)', 'sources', rule?.sources ?? '', 'economic,earnings');
          });

          block('session_buffer', (b) => {
            inputNumber(b, 'Before minutes', 'before_minutes', rule?.before_minutes ?? 15, { min: 0, step: 1 });
            inputNumber(b, 'After minutes', 'after_minutes', rule?.after_minutes ?? 15, { min: 0, step: 1 });
          });

          block('min_trade_interval', (b) => {
            inputNumber(b, 'Minutes', 'minutes', rule?.minutes ?? 30, { min: 0, step: 1 });
          });

          block('max_position_pct', (b) => {
            inputNumber(b, 'Max position %', 'pct', rule?.pct ?? 5.0, { min: 0, step: 0.1 });
          });

          block('entry_delay', (b) => {
            inputNumber(b, 'Delay minutes', 'minutes', rule?.minutes ?? 0, { min: 0, step: 1 });
          });

          block('timezone_restriction', (b) => {
            b.appendChild(el('label', { text: 'Timezone' }));
            const tz = el('input', { type: 'text', placeholder: 'e.g. UTC, America/New_York' });
            tz.setAttribute('data-field', 'timezone');
            try { tz.setAttribute('list', 'guided-v2-timezone-datalist'); } catch (e) {}
            tz.value = rule?.timezone ?? '';
            b.appendChild(tz);
          });

          block('context_invalidation', (b) => {
            inputCheckbox(b, 'Invalidate when context changes', 'invalidate_on_change', rule?.invalidate_on_change);
            inputNumber(b, 'Grace period (bars)', 'grace_period_bars', rule?.grace_period_bars ?? 0, { min: 0, step: 1 });
          });

          block('day_month', (b) => {
            inputText(b, 'Days (csv; optional)', 'days', rule?.days ?? '', 'Mon,Tue,Wed');
            inputText(b, 'Months (csv; optional)', 'months', rule?.months ?? '', 'Jan,Feb');
          });

          block('custom', (b) => {
            inputTextArea(b, 'Custom (DSL)', 'custom', rule?.custom ?? '', 'e.g. my_custom_filter(...)');
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
          addOpt('bollinger_context', 'Bollinger');
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
        } else if (section === 'trade_filters') {
          addOpt('session_window', 'Session window (allow entries during session)');
          addOpt('blackout_window', 'Blackout window (disallow entries)');
          addOpt('max_trades_per_day', 'Max trades per day');
          addOpt('max_exposure', 'Max exposure (USD)');
          addOpt('news_buffer', 'News buffer (minutes)');
          addOpt('session_buffer', 'Session buffer (pre/post minutes)');
          addOpt('min_trade_interval', 'Min interval between trades (minutes)');
          addOpt('max_position_pct', 'Max position (% of account)');
          addOpt('entry_delay', 'Entry delay (minutes)');
          addOpt('timezone_restriction', 'Timezone restriction');
          addOpt('context_invalidation', 'Context invalidation policy');
          addOpt('day_month', 'Day / Month filter');
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
            if ((section === 'signal' || section === 'trigger' || section === 'context') && tfSel) {
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

      // Make the canonical row renderer discoverable to the state-driven
      // init/wiring code above (it looks for `window.ruleRow`).
      try {
        if (typeof window !== 'undefined' && typeof window.ruleRow !== 'function') {
          window.ruleRow = ruleRow;
        }
      } catch (e) {}

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
        if (typeof tradeContainer !== 'undefined' && tradeContainer && tradeHidden) tradeHidden.value = JSON.stringify(serialize(tradeContainer));
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
          trigger_rules: safeParseJson(trgHidden?.value),
          trade_filters: safeParseJson(tradeHidden?.value)
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
  