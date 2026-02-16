export default function mountTradeManagement(container, store) {
  container.innerHTML = `
    <div class="v3-component" role="group" aria-labelledby="tm-title">
      <h3 id="tm-title">Trade Management</h3>
      <div>
        <label for="tm_tp_enabled"><input type="checkbox" id="tm_tp_enabled"/> Take Profit enabled</label>
      </div>
      <div style="margin-top:6px">
        <label for="tm_tp_target_r">Target R:</label>
        <input id="tm_tp_target_r" value="3.0" aria-describedby="tm_tp_target_r_msg"/>
        <span id="tm_tp_target_r_msg" class="v3-field-msg" aria-live="polite"></span>
      </div>
      <div style="margin-top:8px">
        <label for="tm_partial_enabled"><input type="checkbox" id="tm_partial_enabled"/> Partial exit enabled</label>
        <label style="margin-left:8px" for="tm_partial_level_r">Level R:</label>
        <input id="tm_partial_level_r" value="1.5" aria-describedby="tm_partial_level_r_msg"/>
        <span id="tm_partial_level_r_msg" class="v3-field-msg" aria-live="polite"></span>
      </div>
      <div style="margin-top:8px">
        <label for="tm_trailing_enabled"><input type="checkbox" id="tm_trailing_enabled"/> Trailing stop enabled</label>
        <label style="margin-left:8px" for="tm_trailing_length">Length:</label>
        <input id="tm_trailing_length" value="21" aria-describedby="tm_trailing_length_msg"/>
        <span id="tm_trailing_length_msg" class="v3-field-msg" aria-live="polite"></span>
      </div>
    </div>
  `;

  const ids = ['tm_tp_enabled','tm_tp_target_r','tm_partial_enabled','tm_partial_level_r','tm_trailing_enabled','tm_trailing_length'];
  ids.forEach((id) => store.registerField(id, id));

  const el = (id) => container.querySelector(`#${id}`);
  const wireBool = (id, key) => {
    const e = el(id);
    if (e) e.addEventListener('change', (ev) => store.setState({ [key]: !!ev.target.checked }));
  };
  const wireVal = (id, key, parse) => {
    const e = el(id);
    if (e) e.addEventListener('change', (ev) => store.setState({ [key]: parse(ev.target.value) }));
  };

  wireBool('tm_tp_enabled', 'take_profit_enabled');
  wireVal('tm_tp_target_r', 'take_profit_target_r', (v) => parseFloat(v) || 3.0);
  wireBool('tm_partial_enabled', 'partial_enabled');
  wireVal('tm_partial_level_r', 'partial_level_r', (v) => parseFloat(v) || 1.5);
  wireBool('tm_trailing_enabled', 'trailing_enabled');
  wireVal('tm_trailing_length', 'trailing_length', (v) => parseInt(v, 10) || 21);

  store.subscribe((s) => {
    const errs = s && s.validationErrors ? s.validationErrors : [];
    ids.forEach((id) => {
      const i = container.querySelector(`#${id}`);
      if (i && i.classList) i.classList.remove('field-error');
    });
    for (const err of errs) {
      if (!err.path) continue;
      const field = store.findElementForPath(err.path);
      if (field && field.classList) field.classList.add('field-error');
    }
  });
  // initialize values from store
  try {
    const s = (typeof store.getState === 'function') ? store.getState() : {};
    if (s) {
      const map = {
        tm_tp_enabled: 'take_profit_enabled',
        tm_tp_target_r: 'take_profit_target_r',
        tm_partial_enabled: 'partial_enabled',
        tm_partial_level_r: 'partial_level_r',
        tm_trailing_enabled: 'trailing_enabled',
        tm_trailing_length: 'trailing_length',
      };
      ids.forEach((id) => {
        const el = container.querySelector(`#${id}`);
        const key = map[id];
        if (!el || !key) return;
        const v = s[key];
        if (el.type === 'checkbox') el.checked = !!v;
        else if (v !== undefined) el.value = String(v);
      });
    }
  } catch (e) {}

  // simple inter-field guardrail: ensure TP target > partial level when both enabled
  function guardrails() {
    try {
      const s = (typeof store.getState === 'function') ? store.getState() : {};
      const tpEn = !!(s && s.take_profit_enabled);
      const partEn = !!(s && s.partial_enabled);
      const tp = s && s.take_profit_target_r ? parseFloat(s.take_profit_target_r) : null;
      const pr = s && s.partial_level_r ? parseFloat(s.partial_level_r) : null;
      const tpEl = container.querySelector('#tm_tp_target_r');
      const partEl = container.querySelector('#tm_partial_level_r');
      const msgId = 'tm_guard_msg';
      let msgEl = container.querySelector('#' + msgId);
      if (!msgEl) { msgEl = document.createElement('div'); msgEl.id = msgId; msgEl.className = 'v3-field-msg'; container.appendChild(msgEl); }
      msgEl.textContent = '';
      if (tpEn && partEn && tp !== null && pr !== null && tp <= pr) {
        // bump tp to pr + 0.1
        const newTp = Math.max(pr + 0.1, (tp || 0) + 0.1);
        if (tpEl) tpEl.value = String(newTp);
        store.setState({ take_profit_target_r: newTp });
        msgEl.textContent = 'Take Profit target increased to be above Partial level.';
      }
    } catch (e) {}
  }
  store.subscribe(guardrails);
  guardrails();
}
