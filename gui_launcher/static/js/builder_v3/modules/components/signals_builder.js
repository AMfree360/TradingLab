export default function mountSignalsBuilder(container, store) {
  container.innerHTML = `
    <div class="v3-component" role="group" aria-labelledby="sig-title">
      <h3 id="sig-title">Signals</h3>
      <div class="v3-field">
        <label for="signal-rsi">Enable simple RSI signal:</label>
        <input id="signal-rsi" type="checkbox" />
      </div>
      <div style="margin-top:8px"><button id="sig-add" aria-label="Add signal rule">Add Rule</button></div>
      <div id="signal-rules-list" style="margin-top:8px"></div>
    </div>
  `;

  const chk = container.querySelector('#signal-rsi');
  const addBtn = container.querySelector('#sig-add');
  const rulesList = container.querySelector('#signal-rules-list');

  function createRuleElement(rule, idx) {
    const ruleEl = document.createElement('div');
    ruleEl.className = 'signal-rule';
    ruleEl.id = `signal_rules.${idx}.container`;
    ruleEl.innerHTML = `
      <label for="signal_rules.${idx}.length">Rule ${idx} length:</label>
      <input id="signal_rules.${idx}.length" value="${rule.length}" aria-describedby="signal_rules.${idx}.length_msg" />
      <span id="signal_rules.${idx}.length_msg" class="v3-field-msg" aria-live="polite"></span>
      <button type="button" class="sig-remove" style="margin-left:8px" aria-label="Remove rule ${idx}">Remove</button>
    `;
    const lengthInput = ruleEl.querySelector('[id="' + `signal_rules.${idx}.length` + '"]');
    const removeBtn = ruleEl.querySelector('.sig-remove');

    // register input for validation mapping
    store.registerField('signal_rules.[].length', `signal_rules.${idx}.length`);
    // ensure keyboard focus order
    const liInputs = rulesList.querySelectorAll('input,button');
    liInputs.forEach((e, i) => e.tabIndex = 0);

    lengthInput.addEventListener('change', (ev) => {
      let v = parseInt(ev.target.value, 10) || 0;
      const min = 1, max = 1000;
      if (v < min) v = min;
      if (v > max) v = max;
      ev.target.value = String(v);
      const st2 = store.getState() || {};
      const rs = Array.isArray(st2.signal_rules) ? st2.signal_rules.slice() : [];
      if (rs[idx]) rs[idx].length = v;
      store.setState({ signal_rules: rs });
      // inline message clear
      const msg = ruleEl.querySelector('[id="' + `signal_rules.${idx}.length_msg` + '"]');
      if (msg) msg.textContent = '';
    });

    removeBtn.addEventListener('click', () => {
      // unregister this field before DOM/state update
      store.unregisterField(`signal_rules.${idx}.length`);
      const st3 = store.getState() || {};
      const rs3 = Array.isArray(st3.signal_rules) ? st3.signal_rules.slice() : [];
      rs3.splice(idx, 1);
      store.setState({ signal_rules: rs3 });
      // Re-render rules to ensure ids/listeners are consistent
      renderRules();
    });

    return ruleEl;
  }

  function clearRegisteredRuleFields() {
    // Unregister any previously-registered rule input ids present in the DOM
    const children = Array.from(rulesList.children || []);
    for (const ch of children) {
      const input = ch.querySelector('input');
      if (input && input.id) store.unregisterField(input.id);
    }
  }

  function renderRules() {
    // cleanup existing registrations
    clearRegisteredRuleFields();
    // clear DOM
    rulesList.innerHTML = '';
    const st = store.getState() || {};
    const rules = Array.isArray(st.signal_rules) ? st.signal_rules : [];
    for (let i = 0; i < rules.length; i++) {
      const el = createRuleElement(rules[i], i);
      rulesList.appendChild(el);
    }
  }

  addBtn.addEventListener('click', () => {
    const st = store.getState() || {};
    const rules = Array.isArray(st.signal_rules) ? st.signal_rules.slice() : [];
    const newRule = { type: 'rsi_threshold', length: 14, bull_level: 30, bear_level: 70 };
    rules.push(newRule);
    store.setState({ signal_rules: rules });
    renderRules();
  });

  const s = store.getState() || {};
  if (s.rsi_signal) chk.checked = !!s.rsi_signal;

  // subscribe to validation errors and highlight relevant fields
  store.subscribe((fullState) => {
    const errs = (fullState && fullState.validationErrors) ? fullState.validationErrors : [];
    const hasRsiError = errs.some((e) => (e.path === 'rsi_signal' || e.path.startsWith('rsi_signal')));
    if (hasRsiError) chk.classList.add('field-error'); else chk.classList.remove('field-error');
    // highlight per-rule fields
    const rulesErrs = errs.filter((e) => e.path && e.path.startsWith('signal_rules'));
    // clear all rule highlights first
    const inputs = rulesList.querySelectorAll('input');
    inputs.forEach((inp) => inp.classList.remove('field-error'));
    for (const re of rulesErrs) {
      const el = store.findElementForPath(re.path);
      if (el) el.classList.add('field-error');
    }
  });

  // initial render of existing rules
  renderRules();
}
