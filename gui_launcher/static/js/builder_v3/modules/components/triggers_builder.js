export default function mountTriggersBuilder(container, store) {
  container.innerHTML = `
    <div class="v3-component" role="group" aria-labelledby="trig-title">
      <h3 id="trig-title">Triggers</h3>
      <div class="v3-field">
        <label for="trigger_type">Trigger type:</label>
        <select id="trigger_type" aria-describedby="trigger_type_msg">
          <option value="pin_bar">Pin Bar</option>
          <option value="breakout">Breakout</option>
          <option value="custom">Custom</option>
        </select>
        <span id="trigger_type_msg" class="v3-field-msg" aria-live="polite"></span>
      </div>
      <div style="margin-top:8px">
        <label for="trigger_atr_len">ATR length:</label>
        <input id="trigger_atr_len" value="14" aria-describedby="trigger_atr_len_msg"/>
        <span id="trigger_atr_len_msg" class="v3-field-msg" aria-live="polite"></span>
        <label style="margin-left:8px" for="trigger_atr_mult">ATR mult:</label>
        <input id="trigger_atr_mult" value="2.0" aria-describedby="trigger_atr_mult_msg"/>
        <span id="trigger_atr_mult_msg" class="v3-field-msg" aria-live="polite"></span>
      </div>
    </div>
  `;

  // Register fields for validation mapping
  store.registerField('trigger_type', 'trigger_type');
  store.registerField('trigger_atr_len', 'trigger_atr_len');
  store.registerField('trigger_atr_mult', 'trigger_atr_mult');

  const sel = container.querySelector('#trigger_type');
  const atrLen = container.querySelector('#trigger_atr_len');
  const atrMult = container.querySelector('#trigger_atr_mult');

  if (sel) sel.addEventListener('change', (e) => store.setState({ trigger_type: e.target.value }));
  if (atrLen) atrLen.addEventListener('change', (e) => {
    let v = parseInt(e.target.value, 10) || 0;
    const min = 1, max = 500;
    if (v < min) v = min;
    if (v > max) v = max;
    e.target.value = String(v);
    store.setState({ trigger_atr_len: v });
  });
  if (atrMult) atrMult.addEventListener('change', (e) => {
    let v = parseFloat(e.target.value) || 0;
    const min = 0.1, max = 50;
    if (v < min) v = min;
    if (v > max) v = max;
    e.target.value = String(v);
    store.setState({ trigger_atr_mult: v });
  });

  // initialize from store state
  try {
    const s = (typeof store.getState === 'function') ? store.getState() : {};
    if (s && s.trigger_type && sel) sel.value = s.trigger_type;
    if (s && (s.trigger_atr_len !== undefined) && atrLen) atrLen.value = String(s.trigger_atr_len);
    if (s && (s.trigger_atr_mult !== undefined) && atrMult) atrMult.value = String(s.trigger_atr_mult);
  } catch (e) {}

  // reflect external state changes into inputs
  store.subscribe((state) => {
    try {
      const s = state && state.payload ? state.payload : state;
      if (!s) return;
      if (sel && s.trigger_type !== undefined && sel.value !== String(s.trigger_type)) sel.value = s.trigger_type;
      if (atrLen && s.trigger_atr_len !== undefined && atrLen.value !== String(s.trigger_atr_len)) atrLen.value = String(s.trigger_atr_len);
      if (atrMult && s.trigger_atr_mult !== undefined && atrMult.value !== String(s.trigger_atr_mult)) atrMult.value = String(s.trigger_atr_mult);
    } catch (e) {}
  });

  // highlight on validation errors
  store.subscribe((s) => {
    const errs = s && s.validationErrors ? s.validationErrors : [];
    const inputs = [sel, atrLen, atrMult];
    inputs.forEach((inp) => inp && inp.classList && inp.classList.remove('field-error'));
    for (const err of errs) {
      if (!err.path) continue;
      if (err.path === 'trigger_type' || err.path.startsWith('trigger_')) {
        const el = store.findElementForPath(err.path);
        if (el && el.classList) el.classList.add('field-error');
      }
    }
  });

  // inline error messages and simple validation feedback
  function renderFieldMessages() {
    const errs = (typeof store.getValidationErrors === 'function') ? store.getValidationErrors() : (store._state && store._state.validationErrors) || [];
    const lenMsg = container.querySelector('#trigger_atr_len_msg');
    const multMsg = container.querySelector('#trigger_atr_mult_msg');
    if (lenMsg) lenMsg.textContent = '';
    if (multMsg) multMsg.textContent = '';
    for (const er of errs) {
      if (!er.path) continue;
      if (er.path === 'trigger_atr_len' && lenMsg) lenMsg.textContent = er.message;
      if (er.path === 'trigger_atr_mult' && multMsg) multMsg.textContent = er.message;
    }
  }
  store.subscribe(renderFieldMessages);
  renderFieldMessages();
}
