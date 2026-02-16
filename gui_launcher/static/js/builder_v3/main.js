/* Builder V3 — frontend entry skeleton (no reuse of v2 JS)

This file is a minimal skeleton to be expanded by frontend devs.
It intentionally contains no v2 code and only outlines module structure.

Implementations should:
- Provide a StateStore module for the canonical form model
- Provide ApiClient module to call /api/builder_v3/* endpoints
- Implement components as modular JS classes/functions and mount into #builder-v3-root
- Use PAYLOAD_SCHEMA.json for client-side validation
*/

import ApiClient from './modules/api_client.js';
import StateStore from './modules/state_store.js';
import ValidationEngine from './modules/validation_engine.js';
import mountContextBuilder from './modules/components/context_builder.js';
import mountSignalsBuilder from './modules/components/signals_builder.js';
import mountTriggersBuilder from './modules/components/triggers_builder.js';
import mountStopsBuilder from './modules/components/stops_builder.js';
import mountTradeManagement from './modules/components/trade_management_builder.js';
import mountPresetsBuilder from './modules/components/presets_builder.js';
import BuilderModal from './modules/modal.js';
import saveFlow from './modules/save_flow.js';
// Component modules to be implemented under modules/

async function bootstrap() {
  const root = document.getElementById('builder-v3-root');
  if (!root) return;

  // CSRF token: read from hidden input rendered by server
  const csrfEl = document.getElementById('csrf_token');
  const csrf = csrfEl ? csrfEl.value : null;

  const api = new ApiClient({ csrfToken: csrf });
  const store = new StateStore();

  // Load metadata from server (advanced defaults, presets, instruments)
  try {
    const meta = await api.metadata();
    store.setMetadata(meta);
  } catch (e) {
    root.innerHTML = '<div class="card card-error">Failed to load metadata for Builder V3.</div>';
    console.error('metadata load failed', e);
    return;
  }

  // Render initial skeleton UI — frontend dev to replace with components
  root.innerHTML = `
    <div class="v3-grid">
      <section id="v3-left" class="v3-col"></section>
      <aside id="v3-right" class="v3-col preview"></aside>
    </div>
  `;

  const left = document.getElementById('v3-left');
  const right = document.getElementById('v3-right');
  // Render top-level form fields (grouped to match Guided Builder V2 flow)
  function makeLabelInput(labelText, inputEl) {
    const wrap = document.createElement('div');
    wrap.style.marginBottom = '8px';
    const lbl = document.createElement('label');
    lbl.textContent = labelText + ': ';
    lbl.style.display = 'block';
    wrap.appendChild(lbl);
    wrap.appendChild(inputEl);
    return wrap;
  }

  function renderTopLevelForm(container, store) {
    const state = store.getState() || {};
    const form = document.createElement('div');
    form.className = 'v3-top-form';

    // Timeframes section
    const tfSec = document.createElement('fieldset');
    tfSec.innerHTML = '<legend>Timeframes</legend>';
    const makeTfSel = (id, val) => {
      const sel = document.createElement('select');
      sel.id = id;
      ['','1m','5m','15m','1h','4h','1d'].forEach((o) => { const opt = document.createElement('option'); opt.value = o; opt.textContent = o || 'default'; sel.appendChild(opt); });
      sel.value = val || '';
      sel.addEventListener('change', () => store.setState({ [id]: sel.value }));
      return sel;
    };
    tfSec.appendChild(makeLabelInput('Entry TF', makeTfSel('entry_tf', state.entry_tf)));
    tfSec.appendChild(makeLabelInput('Context TF', makeTfSel('context_tf', state.context_tf)));
    tfSec.appendChild(makeLabelInput('Signal TF', makeTfSel('signal_tf', state.signal_tf)));
    tfSec.appendChild(makeLabelInput('Trigger TF', makeTfSel('trigger_tf', state.trigger_tf)));
    form.appendChild(tfSec);

    // Sides & alignment
    const sidesSec = document.createElement('fieldset'); sidesSec.innerHTML = '<legend>Sides</legend>';
    const chk = (id, label, checked) => {
      const input = document.createElement('input'); input.type = 'checkbox'; input.id = id; input.checked = !!checked; input.addEventListener('change', () => store.setState({ [id]: input.checked })); return makeLabelInput(label, input);
    };
    sidesSec.appendChild(chk('long_enabled', 'Enable Long', state.long_enabled));
    sidesSec.appendChild(chk('short_enabled', 'Enable Short', state.short_enabled));
    sidesSec.appendChild(chk('align_with_context', 'Align with context', state.align_with_context));
    form.appendChild(sidesSec);

    // Primary Context & MA settings
    const ctxSec = document.createElement('fieldset'); ctxSec.innerHTML = '<legend>Context / MA</legend>';
    const primarySel = document.createElement('select'); primarySel.id = 'primary_context_type'; ['ma_stack','price_vs_ma','ma_cross_state','atr_pct','structure_breakout_state','ma_spread_pct','custom','none'].forEach((v) => { const o = document.createElement('option'); o.value = v; o.textContent = v; primarySel.appendChild(o); });
    primarySel.value = state.primary_context_type || 'ma_stack'; primarySel.addEventListener('change', () => store.setState({ primary_context_type: primarySel.value }));
    ctxSec.appendChild(makeLabelInput('Primary Context Type', primarySel));

    const maTypeSel = document.createElement('select'); maTypeSel.id = 'ma_type'; ['ema','sma'].forEach((v) => { const o = document.createElement('option'); o.value = v; o.textContent = v; maTypeSel.appendChild(o); }); maTypeSel.value = state.ma_type || 'ema'; maTypeSel.addEventListener('change', () => store.setState({ ma_type: maTypeSel.value })); ctxSec.appendChild(makeLabelInput('MA Type', maTypeSel));
    const makeNum = (id, val, placeholder) => { const n = document.createElement('input'); n.type = 'number'; n.id = id; n.value = typeof val !== 'undefined' && val !== null ? val : ''; n.placeholder = placeholder || ''; n.addEventListener('input', () => { const v = n.value === '' ? null : Number(n.value); store.setState({ [id]: v }); }); return n; };
    ctxSec.appendChild(makeLabelInput('MA Fast', makeNum('ma_fast', state.ma_fast, 'fast')));
    ctxSec.appendChild(makeLabelInput('MA Mid', makeNum('ma_mid', state.ma_mid, 'mid')));
    ctxSec.appendChild(makeLabelInput('MA Slow', makeNum('ma_slow', state.ma_slow, 'slow')));
    ctxSec.appendChild(makeLabelInput('Min MA dist %', makeNum('min_ma_dist_pct', state.min_ma_dist_pct, '0.01')));
    ctxSec.appendChild(makeLabelInput('Slope lookback', makeNum('slope_lookback', state.slope_lookback, 'bars')));
    form.appendChild(ctxSec);

    // Trigger section
    const trgSec = document.createElement('fieldset'); trgSec.innerHTML = '<legend>Trigger</legend>';
    const trigSel = document.createElement('select'); trigSel.id = 'trigger_type'; ['pin_bar','inside_bar_breakout','engulfing','ma_reclaim','prior_bar_break','donchian_breakout','range_breakout','wide_range_candle','custom'].forEach((v) => { const o = document.createElement('option'); o.value = v; o.textContent = v; trigSel.appendChild(o); }); trigSel.value = state.trigger_type || 'pin_bar'; trigSel.addEventListener('change', () => store.setState({ trigger_type: trigSel.value })); trgSec.appendChild(makeLabelInput('Trigger Type', trigSel));
    trgSec.appendChild(makeLabelInput('Trigger valid for bars', makeNum('trigger_valid_for_bars', state.trigger_valid_for_bars, 'bars')));
    trgSec.appendChild(makeLabelInput('Pin wick/body', makeNum('pin_wick_body', state.pin_wick_body, 'ratio')));
    trgSec.appendChild(makeLabelInput('Pin opp wick/body max', makeNum('pin_opp_wick_body_max', state.pin_opp_wick_body_max, 'ratio')));
    trgSec.appendChild(makeLabelInput('Pin min body %', makeNum('pin_min_body_pct', state.pin_min_body_pct, '0.2')));
    trgSec.appendChild(makeLabelInput('Trigger MA type', (function(){ const s=document.createElement('select'); s.id='trigger_ma_type'; ['ema','sma'].forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;s.appendChild(o)}); s.value=state.trigger_ma_type||'ema'; s.addEventListener('change',()=>store.setState({ trigger_ma_type: s.value })); return s; })()));
    trgSec.appendChild(makeLabelInput('Trigger MA len', makeNum('trigger_ma_len', state.trigger_ma_len, 'len')));
    trgSec.appendChild(makeLabelInput('Trigger donchian len', makeNum('trigger_don_len', state.trigger_don_len, 'len')));
    trgSec.appendChild(makeLabelInput('Trigger range len', makeNum('trigger_range_len', state.trigger_range_len, 'len')));
    trgSec.appendChild(makeLabelInput('Trigger ATR len', makeNum('trigger_atr_len', state.trigger_atr_len, 'len')));
    trgSec.appendChild(makeLabelInput('Trigger ATR mult', makeNum('trigger_atr_mult', state.trigger_atr_mult, 'mult')));
    trgSec.appendChild(makeLabelInput('Trigger custom bull expr', (function(){ const ta=document.createElement('textarea'); ta.id='trigger_custom_bull_expr'; ta.rows=2; ta.value=state.trigger_custom_bull_expr||''; ta.addEventListener('input',()=>store.setState({ trigger_custom_bull_expr: ta.value })); return ta; })()));
    trgSec.appendChild(makeLabelInput('Trigger custom bear expr', (function(){ const ta=document.createElement('textarea'); ta.id='trigger_custom_bear_expr'; ta.rows=2; ta.value=state.trigger_custom_bear_expr||''; ta.addEventListener('input',()=>store.setState({ trigger_custom_bear_expr: ta.value })); return ta; })()));
    form.appendChild(trgSec);

    // Risk / sizing
    const rmSec = document.createElement('fieldset'); rmSec.innerHTML = '<legend>Risk & Sizing</legend>';
    rmSec.appendChild(makeLabelInput('Risk per trade %', makeNum('risk_per_trade_pct', state.risk_per_trade_pct, '0.01')));
    const sizingSel = document.createElement('select'); sizingSel.id='sizing_mode'; ['fixed_size','percent_risk','percent_account'].forEach(v=>{const o=document.createElement('option'); o.value=v; o.textContent=v; sizingSel.appendChild(o)}); sizingSel.value=state.sizing_mode||''; sizingSel.addEventListener('change',()=>store.setState({ sizing_mode: sizingSel.value })); rmSec.appendChild(makeLabelInput('Sizing mode', sizingSel));
    rmSec.appendChild(makeLabelInput('Account size', makeNum('account_size', state.account_size, 'account')));
    rmSec.appendChild(makeLabelInput('Max trades per day', makeNum('max_trades_per_day', state.max_trades_per_day, '0')));
    rmSec.appendChild(makeLabelInput('Max daily loss %', makeNum('max_daily_loss_pct', state.max_daily_loss_pct, '0.1')));
    form.appendChild(rmSec);

    container.appendChild(form);
  }

  left.innerHTML = '';
  renderTopLevelForm(left, store);
  right.innerHTML = `
    <div id="v3-preview-json" class="v3-preview-json muted">Preview JSON will appear here.</div>
    <div id="v3-preview-fig" class="v3-preview-fig" style="margin-top:12px"></div>
    <div id="v3-error-summary" class="v3-error-summary" style="margin-top:12px"></div>
  `;

  // Simple debounce helper
  function debounce(fn, wait) {
    let t = null;
    return (...args) => {
      if (t) clearTimeout(t);
      t = setTimeout(() => fn(...args), wait);
    };
  }

  // Preview JSON renderer
  function renderPreview(container, data) {
    try {
      const el = document.getElementById('v3-preview-json');
      if (!el) return;
      el.textContent = JSON.stringify(data, null, 2);
    } catch (e) {
      console.error('renderPreview error', e);
    }
  }

  // Plotly loader + renderer
  const PLOTLY_CDN = 'https://cdn.plot.ly/plotly-2.29.1.min.js';
  function loadPlotly() {
    return new Promise((resolve, reject) => {
      if (window.Plotly) return resolve(window.Plotly);
      // In test environments (Cypress) avoid loading external CDN — provide a stub.
      try {
        if (window.Cypress) {
          window.Plotly = { react: function() {} };
          return resolve(window.Plotly);
        }
      } catch (e) {}
      const s = document.createElement('script');
      s.src = PLOTLY_CDN;
      s.async = true;
      s.onload = () => (window.Plotly ? resolve(window.Plotly) : reject(new Error('Plotly failed to load')));
      s.onerror = (e) => reject(e || new Error('Plotly load error'));
      document.head.appendChild(s);
    });
  }

  async function renderPlotly(container, fig) {
    try {
      const figContainerId = 'v3-plot';
      let c = document.getElementById(figContainerId);
      if (!c) {
        c = document.createElement('div');
        c.id = figContainerId;
        c.style.width = '100%';
        c.style.height = '380px';
        const parent = document.getElementById('v3-preview-fig') || container;
        parent.innerHTML = '';
        parent.appendChild(c);
      }
      await loadPlotly();
      const data = fig && fig.data ? fig.data : (fig || {}).data || [];
      const layout = fig && fig.layout ? fig.layout : (fig || {}).layout || {};
      window.Plotly.react(c, data, layout, { responsive: true });
    } catch (e) {
      console.error('renderPlotly error', e);
      const parent = document.getElementById('v3-preview-fig') || container;
      parent.innerHTML = '<div class="card card-error">Failed to render chart</div>';
    }
  }

  // Load payload schema (use fetch to avoid import-assertion syntax which
  // is not supported in some browsers / test runners).
  let schema = {};
  try {
    const resp = await fetch('/static/js/builder_v3/modules/payload_schema.json', { cache: 'no-cache' });
    if (resp.ok) schema = await resp.json();
    else console.warn('Could not load payload schema:', resp.status);
  } catch (e) {
    console.error('Failed to fetch payload schema', e);
  }

  // Debounced preview call wiring: on any store change, call preview (debounced)
  const validator = new ValidationEngine(schema);

  // Mount simple components into the left column
  // build a richer left column with multiple components for Phase 2 parity
  left.innerHTML = '';
  const sections = ['ctx-builder','triggers-builder','stops-builder','tm-builder','presets-builder','sig-builder'];
  for (const id of sections) {
    const d = document.createElement('div');
    d.id = id;
    left.appendChild(d);
  }

  mountContextBuilder(document.getElementById('ctx-builder'), store);
  mountTriggersBuilder(document.getElementById('triggers-builder'), store);
  mountStopsBuilder(document.getElementById('stops-builder'), store);
  mountTradeManagement(document.getElementById('tm-builder'), store);
  mountPresetsBuilder(document.getElementById('presets-builder'), store);
  mountSignalsBuilder(document.getElementById('sig-builder'), store);

  // Add Save UI and validation error area
  const saveBar = document.createElement('div');
  saveBar.className = 'v3-save-bar';
  saveBar.innerHTML = `
    <button id="v3-save-btn" class="btn btn-primary">Save</button>
    <span id="v3-save-status" style="margin-left:12px"></span>
  `;
  left.appendChild(saveBar);

  const errorArea = document.createElement('pre');
  errorArea.id = 'v3-validation-errors';
  errorArea.style.cssText = 'color:#b22222; background:#fff6f6; padding:8px; display:none; white-space:pre-wrap;';
  left.appendChild(errorArea);

  const doPreview = debounce(async (state) => {
    try {
      const payload = state.payload || {};
      // client-side validation: quick feedback
      const v = validator.validate(payload);
      if (!v.valid) {
        const el = document.getElementById('v3-preview-json');
        if (el) el.textContent = 'Validation errors:\n' + JSON.stringify(v.errors, null, 2);
        return;
      }
      const [previewRes, ctxRes, setupRes] = await Promise.allSettled([
        api.preview(payload),
        api.contextVisual(payload),
        api.setupVisual(payload),
      ]);

      // Normalize preview response (v2 helper returns direct object)
      let previewObj = null;
      if (previewRes.status === 'fulfilled') {
        const v = previewRes.value;
        previewObj = (v && v.ok === true && v.data) ? v.data : v;
      }
      if (previewObj) renderPreview(right, previewObj);

      // Prefer context visual; fall back to setup visual
      let fig = null;
      if (ctxRes.status === 'fulfilled') {
        const v = ctxRes.value;
        fig = (v && v.ok === true && v.data && v.data.fig) ? v.data.fig : (v && v.fig) ? v.fig : null;
      }
      if (!fig && setupRes.status === 'fulfilled') {
        const v = setupRes.value;
        fig = (v && v.ok === true && v.data && v.data.fig) ? v.data.fig : (v && v.fig) ? v.fig : null;
      }
      if (fig) await renderPlotly(right, fig);

    } catch (err) {
      console.error('preview error', err);
      const el = document.getElementById('v3-preview-json');
      if (el) el.textContent = 'Preview failed.';
    }
  }, 400);

  // Subscribe to store changes
  store.subscribe((s) => {
    doPreview(s);
  });

  // Render a clickable summary of server validation errors
  function renderErrorSummary(errors) {
    const container = document.getElementById('v3-error-summary');
    if (!container) return;
    if (!errors || !errors.length) {
      container.innerHTML = '';
      return;
    }
    const list = document.createElement('div');
    list.setAttribute('role', 'region');
    list.setAttribute('aria-live', 'polite');
    list.innerHTML = `<strong>Validation Errors (${errors.length}):</strong>`;
    const ul = document.createElement('ul');
    ul.style.paddingLeft = '18px';
    ul.setAttribute('role', 'list');
    for (const [i, err] of errors.entries()) {
      const li = document.createElement('li');
      li.className = 'v3-error-item';
      li.setAttribute('role', 'listitem');
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'link-like';
      btn.setAttribute('aria-label', err.path ? `Error ${i+1} at ${err.path}` : `Error ${i+1}`);
      btn.setAttribute('data-error-path', err.path || '');
      btn.tabIndex = 0;
      const path = err.path || '';
      btn.textContent = path ? `${path}: ${err.message}` : `${err.message}`;
      btn.addEventListener('click', () => {
        // try to find a matching element and focus/scroll to it
        const el = store.findElementForPath(path);
        if (el) {
          el.scrollIntoView({ behavior: 'smooth', block: 'center' });
          try { el.focus(); } catch (e) {}
        } else {
          // fallback: focus the preview pane
          const p = document.getElementById('v3-preview-json');
          if (p) p.focus();
        }
      });
      li.appendChild(btn);
      ul.appendChild(li);
    }
    list.appendChild(ul);
    container.innerHTML = '';
    container.appendChild(list);
  }

  // subscribe to store to update summary panel
  store.subscribe((fullState) => {
    const errs = (fullState && fullState.validationErrors) ? fullState.validationErrors : [];
    renderErrorSummary(errs);
  });

  // Keyboard shortcuts for accessibility / power users
  // - 'p' focuses preview pane
  // - 's' focuses Save button
  // - 'e' focuses first error in the summary
  document.addEventListener('keydown', (ev) => {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return; // ignore modifiers
    const key = ev.key.toLowerCase();
    if (key === 'p') {
      const p = document.getElementById('v3-preview-json');
      if (p) { p.tabIndex = -1; p.focus(); ev.preventDefault(); }
    }
    if (key === 's') {
      const sb = document.getElementById('v3-save-btn');
      if (sb) { sb.focus(); ev.preventDefault(); }
    }
    if (key === 'e') {
      const firstErr = document.querySelector('#v3-error-summary button');
      if (firstErr) { firstErr.focus(); ev.preventDefault(); }
    }
  });

  // Save handler: validate on server, then save if valid
  const modal = new BuilderModal();

  async function handleSave() {
    const status = document.getElementById('v3-save-status');
    const errEl = document.getElementById('v3-validation-errors');
    const payload = store.serializeForSubmit();
    await saveFlow({ api, modal, store, payload, statusEl: status, errEl });
  }

  const saveBtn = document.getElementById('v3-save-btn');
  if (saveBtn) saveBtn.addEventListener('click', handleSave);
}

  // Signal test runners (Cypress) that the app finished bootstrapping by
  // dispatching a `load` event when running under Cypress. This helps E2E
  // tests that may otherwise wait on external resource load for the native
  // `load` event.
  bootstrap().then(() => {
    try {
      if (window && (window.Cypress || window.__FORCE_DISPATCH_LOAD)) {
        setTimeout(() => {
          try { window.dispatchEvent(new Event('load')); } catch (e) {}
        }, 0);
      }
    } catch (e) {}
  });
