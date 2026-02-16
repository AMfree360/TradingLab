(function(){
  const submitStatusEl = document.getElementById('submit_status');
  function setStatus(msg) { if (!submitStatusEl) return; submitStatusEl.textContent = 'Status: ' + msg; }

  let formEl = null; let reviewBtn = null; let clientErrorEl = null; let inlineErrorEl = null;
  try {
    formEl = document.getElementById('guided_step4_form');
    reviewBtn = document.getElementById('review_btn');
    clientErrorEl = document.getElementById('client_error');
    inlineErrorEl = document.getElementById('inline_error');
    setStatus('js loaded');
  } catch (e) { setStatus('js error (init)'); }

  function showClientError(msg) { if (!clientErrorEl) return; clientErrorEl.textContent = msg; clientErrorEl.style.display = 'block'; clientErrorEl.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  function clearClientError() { if (!clientErrorEl) return; clientErrorEl.textContent = ''; clientErrorEl.style.display = 'none'; }
  function isChecked(name) { if (!formEl) return false; const el = formEl.querySelector(`input[name="${name}"]`); return !!(el && el.checked); }
  function getSelectValue(name) { if (!formEl) return ''; const el = formEl.querySelector(`select[name="${name}"]`); return el ? String(el.value || '') : ''; }
  function getInputValue(name) { if (!formEl) return ''; const el = formEl.querySelector(`input[name="${name}"]`); return el ? String(el.value || '').trim() : ''; }

  function validateOneCondition(prefix) {
    const t = getSelectValue(prefix + '_type');
    if (!t || t === 'none') return null;
    const tf = getSelectValue(prefix + '_tf');
    if (!tf) return `${prefix}: timeframe is required`;
    if (t === 'rsi') { const level = getInputValue(prefix + '_level'); if (!level) return 'RSI level is required.'; return null; }
    if (t === 'ema_cross') { const fast = getInputValue(prefix + '_fast'); const slow = getInputValue(prefix + '_slow'); if (!fast || !slow) return 'EMA fast/slow lengths are required.'; return null; }
    if (t === 'close_vs_ma') { const maLen = getInputValue(prefix + '_ma_len'); if (!maLen) return 'MA length is required.'; return null; }
    if (t === 'donchian') { const donLen = getInputValue(prefix + '_don_len'); if (!donLen) return 'Donchian length is required.'; return null; }
    return null;
  }

  function validateForm() {
    clearClientError();
    if (!formEl) return 'Internal error: form not found in page.';
    const customSymbolEl = formEl.querySelector('input[name="symbol"]');
    if (customSymbolEl && !String(customSymbolEl.value || '').trim()) return 'Instrument symbol is required.';
    const longEnabled = isChecked('long_enabled'); const shortEnabled = isChecked('short_enabled');
    if (!longEnabled && !shortEnabled) return 'Enable LONG and/or SHORT.';
    if (longEnabled) { const e1 = validateOneCondition('long1'); if (e1) return `LONG Condition 1: ${e1}`; if (getSelectValue('long1_type') === 'none') return 'LONG Condition 1 type is required.'; const e2 = validateOneCondition('long2'); if (e2) return `LONG Condition 2: ${e2}`; }
    if (shortEnabled) { const e1 = validateOneCondition('short1'); if (e1) return `SHORT Condition 1: ${e1}`; if (getSelectValue('short1_type') === 'none') return 'SHORT Condition 1 type is required.'; const e2 = validateOneCondition('short2'); if (e2) return `SHORT Condition 2: ${e2}`; }
    return null;
  }

  const enableLong2 = document.getElementById('enable_long2');
  const enableShort2 = document.getElementById('enable_short2');
  const long2Card = document.getElementById('long2_card');
  const short2Card = document.getElementById('short2_card');

  function initOptionalSecondConditions() {
    enableLong2.checked = (long2Card && long2Card.style.display !== 'none');
    enableShort2.checked = (short2Card && short2Card.style.display !== 'none');

    enableLong2.addEventListener('change', () => {
      if (!long2Card) return; long2Card.style.display = enableLong2.checked ? 'block' : 'none';
      if (!enableLong2.checked) { const sel = long2Card.querySelector('select[name="long2_type"]'); if (sel) sel.value = 'none'; updateConditionVisibility('long2', 'none'); }
    });

    enableShort2.addEventListener('change', () => {
      if (!short2Card) return; short2Card.style.display = enableShort2.checked ? 'block' : 'none';
      if (!enableShort2.checked) { const sel = short2Card.querySelector('select[name="short2_type"]'); if (sel) sel.value = 'none'; updateConditionVisibility('short2', 'none'); }
    });
  }

  function updateConditionVisibility(prefix, typeVal) {
    const types = ['rsi','ema_cross','close_vs_ma','donchian'];
    for (const t of types) { const el = document.getElementById(prefix + '_' + t); if (el) el.style.display = (t === typeVal) ? 'block' : 'none'; }
  }

  function updateStopVisibility() {
    const v = document.getElementById('stop_type').value;
    document.getElementById('stop_atr').style.display = (v === 'atr') ? 'block' : 'none';
    document.getElementById('stop_percent').style.display = (v === 'percent') ? 'block' : 'none';
  }

  for (const sel of document.querySelectorAll('.cond-type')) {
    const prefix = sel.getAttribute('data-prefix');
    updateConditionVisibility(prefix, sel.value);
    sel.addEventListener('change', (e) => updateConditionVisibility(prefix, e.target.value));
  }

  document.getElementById('stop_type').addEventListener('change', updateStopVisibility);
  updateStopVisibility();

  initOptionalSecondConditions();

  if (reviewBtn) {
    reviewBtn.addEventListener('click', (e) => {
      setStatus('clicked');
      try {
        const err = validateForm();
        if (err) { e.preventDefault(); showClientError(err); setStatus('blocked: ' + err); }
      } catch (ex) { setStatus('js error (click validation)'); }
    });
  }

  if (formEl) {
    formEl.addEventListener('invalid', (e) => { try { const t = e.target; const name = (t && (t.getAttribute('name') || t.getAttribute('id'))) || 'field'; const msg = (t && t.validationMessage) ? t.validationMessage : 'Invalid input.'; showClientError(`${name}: ${msg}`); setStatus('blocked: ' + msg); } catch (ex) { setStatus('blocked'); } }, true);

    formEl.addEventListener('submit', (e) => {
      const err = validateForm();
      if (err) { e.preventDefault(); showClientError(err); setStatus('blocked: ' + err); return; }
      if (inlineErrorEl) inlineErrorEl.style.display = 'none'; clearClientError(); if (reviewBtn) { reviewBtn.disabled = true; reviewBtn.textContent = 'Loading…'; } setStatus('submitting');
    });

    for (const el of formEl.querySelectorAll('input, select')) {
      el.addEventListener('input', () => { if (inlineErrorEl) inlineErrorEl.style.display = 'none'; clearClientError(); if (reviewBtn) { reviewBtn.disabled = false; reviewBtn.textContent = 'Review'; } setStatus('ready'); });
      el.addEventListener('change', () => { if (inlineErrorEl) inlineErrorEl.style.display = 'none'; clearClientError(); if (reviewBtn) { reviewBtn.disabled = false; reviewBtn.textContent = 'Review'; } setStatus('ready'); });
    }
  }
})();