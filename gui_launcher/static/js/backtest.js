(function(){
  const fileEl = document.getElementById('data_file');
  const infoEl = document.getElementById('dataset_info');
  const startEl = document.getElementById('start_date');
  const endEl = document.getElementById('end_date');

  const marketEl = document.getElementById('market_symbol');
  const dlSymEl = document.getElementsByName('download_symbol')[0];
  const dlEnabledEl = document.getElementsByName('download_data')[0];
  const slipEnabledEl = document.getElementById('slippage_enabled');
  const slipValEl = document.getElementById('slippage_value');
  const slipUnitEl = document.getElementById('slippage_unit_label');
  const slipHintEl = document.getElementById('slippage_hint');
  const slipDefaultsEl = document.getElementById('slippageDefaults');

  let slippageDirty = false;
  let lastAutoValue = (slipDefaultsEl && slipDefaultsEl.dataset) ? (slipDefaultsEl.dataset.initialValue || '') : '';
  let inspectedSymbolHint = '';

  if (slipValEl) {
    slipValEl.addEventListener('input', () => {
      slippageDirty = true;
    });
  }

  function chosenSymbol() {
    const m = marketEl ? String(marketEl.value || '').trim() : '';
    if (m) return m;

    const dlEnabled = dlEnabledEl ? !!dlEnabledEl.checked : false;
    if (dlEnabled && dlSymEl) {
      const s = String(dlSymEl.value || '').trim();
      if (s) return s;
    }

    if (inspectedSymbolHint) return inspectedSymbolHint;

    try {
      const f = fileEl && fileEl.files && fileEl.files[0];
      const name = f ? String(f.name || '') : '';
      if (name) {
        const base = name.split('/').pop();
        const stem = base.includes('.') ? base.slice(0, base.lastIndexOf('.')) : base;
        const parts = stem.split('_').filter(Boolean);
        for (let i = 0; i < parts.length; i++) {
          const tok = String(parts[i] || '').trim();
          if (!tok) continue;
          if (tok.toLowerCase() === 'dataset') continue;
          if (/^(m|h|d|w)\d+$/i.test(tok)) continue;
          if (/^\d+(m|min|h|d|w)$/i.test(tok)) continue;
          if (/^20\d{2}$/.test(tok)) continue;
          if (/^[A-Za-z0-9]{3,}$/.test(tok)) {
            return tok;
          }
        }
      }
    } catch (e) {}

    return '';
  }

  function canAutofillSlippage() {
    if (!slipValEl) return false;
    if (slippageDirty) return false;
    const cur = String(slipValEl.value || '').trim();
    return (!cur) || (String(lastAutoValue).trim() === cur);
  }

  let slippageTimer = null;
  async function refreshSlippageDefaults() {
    const sym = chosenSymbol();
    if (!sym) return;

    const enabled = slipEnabledEl ? !!slipEnabledEl.checked : true;
    if (!enabled && !canAutofillSlippage()) return;

    try {
      const res = await fetch('/api/advanced-defaults?symbol=' + encodeURIComponent(sym));
      const data = await res.json();
      if (!res.ok || !data.ok) {
        if (slipHintEl) slipHintEl.textContent = '';
        return;
      }

      const unit = (data.labels && data.labels.slippage_unit) ? String(data.labels.slippage_unit) : '';
      const bv = data.backtest_validation || {};

      let val = null;
      if (bv.slippage_ticks !== undefined && bv.slippage_ticks !== null) {
        val = bv.slippage_ticks;
      } else if (bv.slippage_default !== undefined && bv.slippage_default !== null) {
        val = bv.slippage_default;
      }

      if (unit && slipUnitEl) slipUnitEl.textContent = unit;

      if (val !== null && val !== undefined && canAutofillSlippage()) {
        const v = String(Number(val));
        slipValEl.value = v;
        lastAutoValue = v;
        if (slipHintEl) slipHintEl.textContent = `Autofilled from defaults for ${String(data.symbol || sym).toUpperCase()}.`;
      } else {
        if (slipHintEl) slipHintEl.textContent = '';
      }
    } catch (e) {
      if (slipHintEl) slipHintEl.textContent = '';
    }
  }

  function scheduleRefreshSlippage() {
    if (slippageTimer) window.clearTimeout(slippageTimer);
    slippageTimer = window.setTimeout(refreshSlippageDefaults, 250);
  }

  if (marketEl) marketEl.addEventListener('input', scheduleRefreshSlippage);
  if (dlSymEl) dlSymEl.addEventListener('input', scheduleRefreshSlippage);
  if (dlEnabledEl) dlEnabledEl.addEventListener('change', scheduleRefreshSlippage);
  if (slipEnabledEl) slipEnabledEl.addEventListener('change', scheduleRefreshSlippage);

  async function inspectDataset(file) {
    if (!infoEl) return;
    infoEl.textContent = 'Inspecting dataset…';

    const fd = new FormData();
    fd.append('data_file', file);

    try {
      const res = await fetch('/backtest/inspect', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        infoEl.textContent = `Could not inspect dataset: ${data.error || res.status}`;
        return;
      }

      const tf = data.timeframe ? ` · TF: ${data.timeframe}` : '';
      const sym = data.symbol_hint ? ` · Symbol: ${String(data.symbol_hint).toUpperCase()}` : '';
      infoEl.textContent = `Detected range: ${data.start_date} → ${data.end_date} (${data.rows} rows)${tf}${sym}`;

      if (data.symbol_hint) {
        inspectedSymbolHint = String(data.symbol_hint || '').trim().toUpperCase();
      } else {
        inspectedSymbolHint = '';
      }

      if (startEl) startEl.value = data.start_date;
      if (endEl) endEl.value = data.end_date;

      refreshSlippageDefaults();
    } catch (e) {
      infoEl.textContent = `Could not inspect dataset: ${e}`;
    }
  }

  if (fileEl) {
    fileEl.addEventListener('change', () => {
      const f = fileEl.files && fileEl.files[0];
      if (!f) {
        if (infoEl) infoEl.textContent = '';
        return;
      }
      inspectDataset(f);
      scheduleRefreshSlippage();
    });
  }

  scheduleRefreshSlippage();
})();