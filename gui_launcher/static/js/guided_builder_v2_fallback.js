// Minimal robust fallback to ensure add-buttons work if primary renderer fails
(function () {
  function el(tag, attrs) {
    const node = document.createElement(tag);
    if (!attrs) return node;
    for (const k of Object.keys(attrs)) {
      if (k === 'text') node.textContent = attrs[k];
      else if (k === 'html') node.innerHTML = attrs[k];
      else node.setAttribute(k, String(attrs[k]));
    }
    return node;
  }

  function serialize(container) {
    const out = [];
    if (!container) return out;
    for (const r of container.querySelectorAll('.rule-row')) {
      const typeEl = r.querySelector('[data-type]');
      const tfEl = r.querySelector('[data-tf]');
      const validEl = r.querySelector('[data-valid]');
      const rule = {};
      if (typeEl) rule.type = String(typeEl.value || typeEl.textContent || '').trim();
      if (tfEl && tfEl.value && tfEl.value !== 'default') rule.tf = String(tfEl.value);
      if (validEl && validEl.value !== '') { const n = Number(validEl.value); if (Number.isFinite(n)) rule.valid_for_bars = n; }
      out.push(rule);
    }
    return out;
  }

  function createRow(section, label) {
    const row = el('div', { class: 'card rule-row' });
    row.setAttribute('data-section', section);
    const left = el('div');
    const type = el('select'); type.setAttribute('data-type','');
    const opt = el('option', { value: label }); opt.textContent = label; type.appendChild(opt);
    left.appendChild(type);
    if (section === 'signal' || section === 'trigger') {
      const tf = el('select'); tf.setAttribute('data-tf',''); tf.appendChild(el('option',{value:'default',text:'default'}));
      left.appendChild(tf);
    }
    const right = el('div');
    const valid = el('input'); valid.type = 'number'; valid.setAttribute('data-valid',''); valid.placeholder = 'valid_for_bars';
    right.appendChild(valid);
    const remove = el('button'); remove.type = 'button'; remove.textContent = 'Remove';
    remove.addEventListener('click', () => { row.remove(); syncAll(); });
    right.appendChild(remove);
    row.appendChild(left); row.appendChild(right);
    return row;
  }

  function syncAll() {
    const ctxHidden = document.getElementById('context_rules_json');
    const sigHidden = document.getElementById('signal_rules_json');
    const trgHidden = document.getElementById('trigger_rules_json');
    const ctx = document.getElementById('context-rules');
    const sig = document.getElementById('signal-rules');
    const trg = document.getElementById('trigger-rules');
    if (ctxHidden && ctx) ctxHidden.value = JSON.stringify(serialize(ctx));
    if (sigHidden && sig) sigHidden.value = JSON.stringify(serialize(sig));
    if (trgHidden && trg) trgHidden.value = JSON.stringify(serialize(trg));
  }

  function init() {
    const addCtx = document.getElementById('add-context');
    const addSig = document.getElementById('add-signal');
    const addTrg = document.getElementById('add-trigger');
    const ctx = document.getElementById('context-rules');
    const sig = document.getElementById('signal-rules');
    const trg = document.getElementById('trigger-rules');
    if (addCtx && ctx) addCtx.addEventListener('click', (ev) => { ev.preventDefault(); ctx.appendChild(createRow('context','price_vs_ma')); syncAll(); });
    if (addSig && sig) addSig.addEventListener('click', (ev) => { ev.preventDefault(); sig.appendChild(createRow('signal','rsi_threshold')); syncAll(); });
    if (addTrg && trg) addTrg.addEventListener('click', (ev) => { ev.preventDefault(); trg.appendChild(createRow('trigger','prior_bar_break')); syncAll(); });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
