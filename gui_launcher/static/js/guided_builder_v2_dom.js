/* Guided Builder v2 — DOM bindings and minimal renderer (Phase 3)
   Binds primitive form controls to the central state manager (GuidedBuilderV2State)
   and keeps hidden JSON rule inputs in sync. Intended as a lightweight layer
   before full rule-row renderer is implemented.
*/
(function () {
  function $(sel, root) { return (root || document).querySelector(sel); }
  function $all(sel, root) { return Array.from((root || document).querySelectorAll(sel)); }

  function parseValueByDefault(raw, def) {
    if (def === null || def === undefined) return raw;
    if (typeof def === 'boolean') return !!raw;
    if (typeof def === 'number') {
      const n = Number(raw);
      return Number.isFinite(n) ? n : def;
    }
    if (Array.isArray(def)) {
      try { const p = JSON.parse(raw); return Array.isArray(p) ? p : def; } catch (e) { return def; }
    }
    return raw == null ? '' : String(raw);
  }

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

    // ensure on submit we flush and sync hidden
    form.addEventListener('submit', (ev) => {
      try { window.GuidedBuilderV2State.flush(); } catch (e) {}
      try { syncHiddenRules(form, window.GuidedBuilderV2State.raw()); } catch (e) {}
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
