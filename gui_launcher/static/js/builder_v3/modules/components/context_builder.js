export default function mountContextBuilder(container, store) {
  container.innerHTML = `
    <div class="v3-component" role="group" aria-labelledby="ctx-title">
      <h3 id="ctx-title">Context</h3>
      <div class="v3-field">
        <label for="ctx-type">Primary context type:</label>
        <select id="ctx-type" aria-describedby="ctx-type-msg">
          <option value="ma_stack">MA Stack</option>
          <option value="price_vs_ma">Price vs MA</option>
        </select>
      </div>
      <div style="margin-top:8px"><button id="ctx-refresh" aria-label="Refresh preview">Refresh preview</button></div>
    </div>
  `;

  const typeEl = container.querySelector('#ctx-type');
  const btn = container.querySelector('#ctx-refresh');
  typeEl.addEventListener('change', (e) => {
    store.setState({ primary_context_type: e.target.value });
  });
  btn.addEventListener('click', () => {
    // trigger a store update to cause preview
    store.setState({ ctx_refresh: Date.now() });
  });

  // initialize from store
  const s = store.getState() || {};
  if (s.primary_context_type) typeEl.value = s.primary_context_type;
  // register field for validation mapping
  store.registerField('primary_context_type', 'ctx-type');

  // inline message for context field
  const ctxMsgId = 'ctx-type-msg';
  let ctxMsg = container.querySelector('#' + ctxMsgId);
  if (!ctxMsg && typeEl && typeEl.parentNode) {
    ctxMsg = document.createElement('span'); ctxMsg.id = ctxMsgId; ctxMsg.className = 'v3-field-msg'; ctxMsg.setAttribute('aria-live', 'polite'); typeEl.parentNode.appendChild(ctxMsg);
    // ensure select references the message for screen readers
    typeEl.setAttribute('aria-describedby', ctxMsgId);
  }

  // subscribe to validation errors and highlight relevant fields
  store.subscribe((fullState) => {
    const errs = (fullState && fullState.validationErrors) ? fullState.validationErrors : [];
    const hasError = errs.some((e) => (e.path === 'primary_context_type' || e.path.startsWith('primary_context_type')));
    if (hasError) typeEl.classList.add('field-error'); else typeEl.classList.remove('field-error');
    // auto-focus first matching error
    if (ctxMsg) ctxMsg.textContent = '';
    if (errs && errs.length) {
      const first = errs[0];
      const el = store.findElementForPath(first.path);
      if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'center' }); el.focus(); }
      // show first matching message for this field
      for (const er of errs) {
        if (er.path === 'primary_context_type' && ctxMsg) { ctxMsg.textContent = er.message; break; }
      }
    }
  });
}
