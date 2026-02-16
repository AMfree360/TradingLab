export default function mountStopsBuilder(container, store) {
  container.innerHTML = `
    <div class="v3-component" role="group" aria-labelledby="stops-title">
      <h3 id="stops-title">Stops</h3>
      <div class="v3-field">
        <label for="stop_mode">Stop mode:</label>
        <select id="stop_mode" aria-describedby="stop_mode_msg">
          <option value="fixed">Fixed</option>
          <option value="atr">ATR</option>
          <option value="trailing">Trailing</option>
        </select>
        <span id="stop_mode_msg" class="v3-field-msg" aria-live="polite"></span>
      </div>
      <div style="margin-top:8px">
        <label for="stop_fixed_pips">Fixed distance (pips):</label>
        <input id="stop_fixed_pips" value="20" aria-describedby="stop_fixed_pips_msg"/>
        <span id="stop_fixed_pips_msg" class="v3-field-msg" aria-live="polite"></span>
        <label style="margin-left:8px" for="stop_atr_len">ATR length:</label>
        <input id="stop_atr_len" value="14" aria-describedby="stop_atr_len_msg"/>
        <span id="stop_atr_len_msg" class="v3-field-msg" aria-live="polite"></span>
      </div>
    </div>
  `;

  store.registerField('stop_mode', 'stop_mode');
  store.registerField('stop_fixed_pips', 'stop_fixed_pips');
  store.registerField('stop_atr_len', 'stop_atr_len');

  const mode = container.querySelector('#stop_mode');
  const fixed = container.querySelector('#stop_fixed_pips');
  const atr = container.querySelector('#stop_atr_len');

  if (mode) mode.addEventListener('change', (e) => store.setState({ stop_mode: e.target.value }));
  if (fixed) fixed.addEventListener('change', (e) => {
    let v = parseFloat(e.target.value) || 0;
    const min = 0, max = 10000;
    if (v < min) v = min;
    if (v > max) v = max;
    e.target.value = String(v);
    store.setState({ stop_fixed_pips: v });
  });
  if (atr) atr.addEventListener('change', (e) => {
    let v = parseInt(e.target.value, 10) || 0;
    const min = 1, max = 500;
    if (v < min) v = min;
    if (v > max) v = max;
    e.target.value = String(v);
    store.setState({ stop_atr_len: v });
  });

  store.subscribe((s) => {
    const errs = s && s.validationErrors ? s.validationErrors : [];
    [mode, fixed, atr].forEach((inp) => inp && inp.classList && inp.classList.remove('field-error'));
    for (const err of errs) {
      if (!err.path) continue;
      if (err.path.startsWith('stop_')) {
        const el = store.findElementForPath(err.path);
        if (el && el.classList) el.classList.add('field-error');
      }
    }
  });

  // initialize values from store
  try {
    const s = (typeof store.getState === 'function') ? store.getState() : {};
    if (s && s.stop_mode && mode) mode.value = s.stop_mode;
    if (s && s.stop_fixed_pips !== undefined && fixed) fixed.value = String(s.stop_fixed_pips);
    if (s && s.stop_atr_len !== undefined && atr) atr.value = String(s.stop_atr_len);
  } catch (e) {}

  // show/hide inputs depending on mode
  function refreshVisibility() {
    const m = (mode && mode.value) ? mode.value : (store.getState && store.getState().stop_mode) || 'fixed';
    if (fixed) fixed.closest && fixed.closest('.signal-rule') ? null : null;
    if (m === 'fixed') {
      if (fixed && fixed.style) fixed.style.display = '';
      if (atr && atr.style) atr.style.display = 'none';
    } else if (m === 'atr') {
      if (fixed && fixed.style) fixed.style.display = 'none';
      if (atr && atr.style) atr.style.display = '';
    } else {
      if (fixed && fixed.style) fixed.style.display = '';
      if (atr && atr.style) atr.style.display = '';
    }
  }

  // inline messages
  function renderMessages() {
    const errs = (typeof store.getValidationErrors === 'function') ? store.getValidationErrors() : (store._state && store._state.validationErrors) || [];
    // ensure message spans exist
    let fixedMsg = container.querySelector('#stop_fixed_pips_msg');
    let atrMsg = container.querySelector('#stop_atr_len_msg');
    if (!fixedMsg && fixed && fixed.parentNode) {
      fixedMsg = document.createElement('span'); fixedMsg.id = 'stop_fixed_pips_msg'; fixedMsg.className = 'v3-field-msg'; fixed.parentNode.appendChild(fixedMsg);
    }
    if (!atrMsg && atr && atr.parentNode) {
      atrMsg = document.createElement('span'); atrMsg.id = 'stop_atr_len_msg'; atrMsg.className = 'v3-field-msg'; atr.parentNode.appendChild(atrMsg);
    }
    if (fixedMsg) fixedMsg.textContent = '';
    if (atrMsg) atrMsg.textContent = '';
    for (const er of errs) {
      if (!er.path) continue;
      if (er.path === 'stop_fixed_pips' && fixedMsg) fixedMsg.textContent = er.message;
      if (er.path === 'stop_atr_len' && atrMsg) atrMsg.textContent = er.message;
    }
  }
  store.subscribe(renderMessages);
  renderMessages();

  if (mode) mode.addEventListener('change', refreshVisibility);
  refreshVisibility();
}
