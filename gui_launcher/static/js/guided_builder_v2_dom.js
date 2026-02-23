/* archived copy of guided_builder_v2_dom.js
   This file has been archived to:
     gui_launcher/static/js/archived_builder_duplicates/guided_builder_v2_dom.js
   The active Step4 implementation is provided by `guided_builder_v2.js`.
   Keeping this stub to avoid accidental inclusion; do not re-enable.
*/

console.warn('guided_builder_v2_dom.js is archived; use guided_builder_v2.js');

// no-op stub (archived)

  function readFormPrimitives(form, modelDefaults) {
    const out = {};
    if (!form) return out;
    const keys = Object.keys(modelDefaults || {});
    for (const k of keys) {
      // skip underscored or rule arrays (we handle rule arrays separately)
      if (k === '_version') continue;
      if (k === 'context_rules' || k === 'signal_rules' || k === 'trigger_rules') continue;
      const def = modelDefaults[k];
      const el = form.elements ? form.elements[k] : null;
      if (!el) {
        out[k] = def;
        continue;
      }
      if (el.type === 'checkbox') {
        out[k] = !!el.checked;
      } else if (el.tagName === 'SELECT' || el.tagName === 'TEXTAREA' || el.type === 'text' || el.type === 'number' || el.type === 'hidden') {
        const raw = el.value;
        out[k] = parseValueByDefault(raw, def);
      } else {
        out[k] = parseValueByDefault(el.value, def);
      }
    }
    return out;
  }

  function syncHiddenRules(form, state) {
    try {
      const ctxHidden = form.querySelector('#context_rules_json');
      const sigHidden = form.querySelector('#signal_rules_json');
      const trgHidden = form.querySelector('#trigger_rules_json');
      if (ctxHidden) ctxHidden.value = JSON.stringify(state.context_rules || []);
      if (sigHidden) sigHidden.value = JSON.stringify(state.signal_rules || []);
      if (trgHidden) trgHidden.value = JSON.stringify(state.trigger_rules || []);
    } catch (e) {
      console.error('syncHiddenRules error', e);
    }
  }

  function serializeRulesFromDOMMinimal(container) {
    const out = [];
    if (!container) return out;
    const rows = Array.from(container.querySelectorAll('.rule-row'));
    for (const r of rows) {
      try {
        // be tolerant: support multiple renderer variants
        const typeEl = r.querySelector('.rule-type') || r.querySelector('[data-type]') || r.querySelector('select[name="type"]') || r.querySelector('input[data-type]') || r.querySelector('select.rule-type');
        const type = typeEl ? (typeEl.value || typeEl.textContent || '') : '';
        if (!type) continue;
        const rule = { type };
        const tfEl = r.querySelector('.rule-tf') || r.querySelector('[data-tf]') || r.querySelector('select[name="tf"]'); if (tfEl && tfEl.value && tfEl.value !== 'default') rule.tf = tfEl.value;
        const validEl = r.querySelector('.rule-valid') || r.querySelector('[data-valid]') || r.querySelector('input[name="valid_for_bars"]'); if (validEl && validEl.value !== '') { const n = Number(validEl.value); if (Number.isFinite(n)) rule.valid_for_bars = n; }
        // collect simple params (inputs/selects/textareas with data-key)
        const params = r.querySelectorAll('.param');
        for (const p of Array.from(params)) {
          const k = p.getAttribute && p.getAttribute('data-key');
          if (!k) continue;
          const v = p.value;
          if (p.type === 'number') { const n = Number(v); rule[k] = Number.isFinite(n) ? n : v; } else { rule[k] = v; }
        }
        // per-rule side control (both|long|short) if present
        try {
          const sideEl = r.querySelector('.rule-side') || r.querySelector('[data-side]') || r.querySelector('select[name="side"]');
          if (sideEl && sideEl.value) {
            rule.side = String(sideEl.value).trim().toLowerCase();
          }
        } catch (e) { /* ignore */ }
        try { console.log('[guided-debug] serializeRow ->', { type: rule.type, tf: rule.tf || null, valid: rule.valid_for_bars || null, side: rule.side || null }); } catch (e) {}
        out.push(rule);
      } catch (e) { /* best-effort */ }
    }
    return out;
  }

  function applyStateToForm(form, state, modelDefaults) {
    if (!form || !state) return;
    const keys = Object.keys(modelDefaults || {});
    for (const k of keys) {
      if (k === '_version') continue;
      if (k === 'context_rules' || k === 'signal_rules' || k === 'trigger_rules') continue;
      const def = modelDefaults[k];
      const el = form.elements ? form.elements[k] : null;
      if (!el) continue;
      try {
        if (el.type === 'checkbox') {
          el.checked = !!state[k];
        } else if (el.tagName === 'SELECT' || el.tagName === 'TEXTAREA' || el.type === 'text' || el.type === 'number' || el.type === 'hidden') {
          if (state[k] === null || typeof state[k] === 'undefined') el.value = '';
          else el.value = String(state[k]);
        } else {
          el.value = state[k] == null ? '' : String(state[k]);
        }
      } catch (e) {
        // ignore per-control errors
      }
    }
    // sync hidden rule JSONs
    syncHiddenRules(form, state);
  }

  function wireFormToState(form, stateManager, modelDefaults) {
    if (!form || !stateManager) return;
    let applying = false;

    // subscribe to state updates and reflect to form
    stateManager.subscribe((s) => {
      if (applying) return;
      applying = true;
      try { applyStateToForm(form, s, modelDefaults); } catch (e) { console.error(e); }
      applying = false;
    });

    // on user input, update state
    const inputs = $all('input[name], select[name], textarea[name]', form);
    for (const inp of inputs) {
      const name = inp.getAttribute('name');
      if (!name) continue;
      // skip rule JSON hidden inputs — they are driven by state
      if (name === 'context_rules_json' || name === 'signal_rules_json' || name === 'trigger_rules_json') continue;

      const handler = () => {
        if (applying) return;
        let val;
        if (inp.type === 'checkbox') val = !!inp.checked;
        else if (inp.type === 'number') {
          const n = Number(inp.value);
          val = Number.isFinite(n) ? n : inp.value;
        } else {
          val = inp.value;
        }
        // set into state (flat path)
        try { stateManager.set(name, val); } catch (e) { console.error('state set error', e); }
      };

      inp.addEventListener('change', handler);
      inp.addEventListener('input', handler);
    }

    // ensure hidden rules reflect current state immediately
    stateManager.flush();
  }

  function init() {
    const form = document.querySelector('form[action="/create-strategy-guided/step4"]') || document.getElementById('guided_step4_form');
    if (!form) return;
    if (!window.GuidedBuilderV2State || !window.GuidedBuilderV2Model) {
      console.warn('GuidedBuilderV2Model or GuidedBuilderV2State missing; DOM bindings skipped');
      return;
    }

    // Build initial state from model defaults and current form values (including hidden JSONs)
    const modelDefaults = GuidedBuilderV2Model.defaults();
    const primitives = readFormPrimitives(form, modelDefaults);

    // parse rule JSONs if present
    let ctxRules = [];
    let sigRules = [];
    let trgRules = [];
    try { ctxRules = JSON.parse(String((form.querySelector('#context_rules_json') || {}).value || '[]')); } catch (e) { ctxRules = []; }
    try { sigRules = JSON.parse(String((form.querySelector('#signal_rules_json') || {}).value || '[]')); } catch (e) { sigRules = []; }
    try { trgRules = JSON.parse(String((form.querySelector('#trigger_rules_json') || {}).value || '[]')); } catch (e) { trgRules = []; }

    const initial = Object.assign({}, primitives, {
      context_rules: Array.isArray(ctxRules) ? ctxRules : [],
      signal_rules: Array.isArray(sigRules) ? sigRules : [],
      trigger_rules: Array.isArray(trgRules) ? trgRules : [],
    });

    // load into state manager
    try {
      window.GuidedBuilderV2State.load(initial);
    } catch (e) {
      console.error('Failed to load initial state', e);
    }

    // wire form <-> state sync
    wireFormToState(form, window.GuidedBuilderV2State, modelDefaults);

    // capture-phase submit handler: prefer state.raw() to populate hidden JSONs
    try {
      form.addEventListener('submit', (ev) => {
        try {
          // ensure state flushed into its internal store first
          try { window.GuidedBuilderV2State && window.GuidedBuilderV2State.flush && window.GuidedBuilderV2State.flush(); } catch (e) {}

          // prepare hidden inputs
          const ctxHidden = form.querySelector('#context_rules_json');
          const sigHidden = form.querySelector('#signal_rules_json');
          const trgHidden = form.querySelector('#trigger_rules_json');
          let usedState = false;

          // prefer explicit preview builder payload if available
          if (window._guided_builder_v2_test && typeof window._guided_builder_v2_test.buildPayload === 'function') {
            try {
              const pb = window._guided_builder_v2_test.buildPayload() || {};
              if (pb.context_rules && Array.isArray(pb.context_rules) && pb.context_rules.length > 0) { if (ctxHidden) ctxHidden.value = JSON.stringify(pb.context_rules); usedState = true; }
              if (pb.signal_rules && Array.isArray(pb.signal_rules) && pb.signal_rules.length > 0) { if (sigHidden) sigHidden.value = JSON.stringify(pb.signal_rules); usedState = true; }
              if (pb.trigger_rules && Array.isArray(pb.trigger_rules) && pb.trigger_rules.length > 0) { if (trgHidden) trgHidden.value = JSON.stringify(pb.trigger_rules); usedState = true; }
            } catch (e) { /* ignore preview builder error */ }
          }

          // prefer state if it contains rules
          try {
            if (window.GuidedBuilderV2State && typeof window.GuidedBuilderV2State.raw === 'function') {
              const s = window.GuidedBuilderV2State.raw();
              if (s && Array.isArray(s.context_rules) && s.context_rules.length > 0) {
                if (ctxHidden) ctxHidden.value = JSON.stringify(s.context_rules);
                usedState = true;
              }
              if (s && Array.isArray(s.signal_rules) && s.signal_rules.length > 0) {
                if (sigHidden) sigHidden.value = JSON.stringify(s.signal_rules);
                usedState = true;
              }
              if (s && Array.isArray(s.trigger_rules) && s.trigger_rules.length > 0) {
                if (trgHidden) trgHidden.value = JSON.stringify(s.trigger_rules);
                usedState = true;
              }
            }
          } catch (e) { /* ignore */ }

          // fallback to DOM serialization if state had no rules
          if (!usedState) {
            try {
              const ctxContainer = document.getElementById('context-rules');
              const sigContainer = document.getElementById('signal-rules');
              const trgContainer = document.getElementById('trigger-rules');
              const ctxRows = ctxContainer ? Array.from(ctxContainer.querySelectorAll('.rule-row')) : [];
              const sigRows = sigContainer ? Array.from(sigContainer.querySelectorAll('.rule-row')) : [];
              const trgRows = trgContainer ? Array.from(trgContainer.querySelectorAll('.rule-row')) : [];
              console.log('[guided-debug] submit-dom-serialize: row counts', { ctx: ctxRows.length, sig: sigRows.length, trg: trgRows.length });
              const ctxSer = serializeRulesFromDOMMinimal(ctxContainer) || [];
              const sigSer = serializeRulesFromDOMMinimal(sigContainer) || [];
              const trgSer = serializeRulesFromDOMMinimal(trgContainer) || [];
              console.log('[guided-debug] submit-dom-serialize: serialized', { ctx: ctxSer, sig: sigSer, trg: trgSer });
              if (ctxHidden) ctxHidden.value = JSON.stringify(ctxSer);
              if (sigHidden) sigHidden.value = JSON.stringify(sigSer);
              if (trgHidden) trgHidden.value = JSON.stringify(trgSer);
            } catch (e) { console.error('submit-dom-serialize inner failed', e); }
          }

          // Debug: expose hidden JSONs in console and debug box when enabled
          try {
            const dbg = document.getElementById('guided-debug');
            const cval = (ctxHidden || {}).value || '[]';
            const sval = (sigHidden || {}).value || '[]';
            const tval = (trgHidden || {}).value || '[]';
            console.log('[guided-debug] submit hidden ->', { context: cval, signals: sval, triggers: tval });
            if (window.guidedBuilderV2DebugEnabled && dbg) {
              dbg.textContent = 'SUBMIT HIDDEN JSONS:\n' + 'context_rules_json=' + cval + '\n' + 'signal_rules_json=' + sval + '\n' + 'trigger_rules_json=' + tval + '\n' + (dbg.textContent || '');
            }
          } catch (e) { /* ignore */ }

        } catch (e) { /* best-effort */ }
      }, true);
    } catch (e) { /* ignore */ }

    // ensure on submit we flush and sync hidden (legacy/bubble handler)
    form.addEventListener('submit', (ev) => {
      try { window.GuidedBuilderV2State.flush(); } catch (e) {}
      // Always attempt to serialize rule rows directly from DOM as a last-resort
      try {
        const ctxContainer = document.getElementById('context-rules');
        const sigContainer = document.getElementById('signal-rules');
        const trgContainer = document.getElementById('trigger-rules');
        const ctxHidden = form.querySelector('#context_rules_json');
        const sigHidden = form.querySelector('#signal_rules_json');
        const trgHidden = form.querySelector('#trigger_rules_json');
        if (ctxHidden) ctxHidden.value = JSON.stringify(serializeRulesFromDOMMinimal(ctxContainer) || []);
        if (sigHidden) sigHidden.value = JSON.stringify(serializeRulesFromDOMMinimal(sigContainer) || []);
        if (trgHidden) trgHidden.value = JSON.stringify(serializeRulesFromDOMMinimal(trgContainer) || []);
      } catch (e) { console.error('submit-dom-serialize failed', e); }

      // Defensive check: if LONG enabled but no entry conditions provided, prevent submit and show message
      try {
        const longEnabled = !!(form.querySelector('[name="long_enabled"]') && form.querySelector('[name="long_enabled"]').checked);
        const sigVal = String((form.querySelector('#signal_rules_json') || {}).value || '[]');
        const trgVal = String((form.querySelector('#trigger_rules_json') || {}).value || '[]');
        let sigArr = [];
        let trgArr = [];
        try { sigArr = JSON.parse(sigVal); } catch (e) { sigArr = []; }
        try { trgArr = JSON.parse(trgVal); } catch (e) { trgArr = []; }
        const hasEntry = (Array.isArray(sigArr) && sigArr.length > 0) || (Array.isArray(trgArr) && trgArr.length > 0);
        if (longEnabled && !hasEntry) {
          ev.preventDefault();
          alert('Cannot Review: no LONG entry condition defined. Add a Signal or Trigger row.');
          return;
        }
      } catch (e) { /* ignore */ }
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
