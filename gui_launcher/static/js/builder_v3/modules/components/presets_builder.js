export default function mountPresetsBuilder(container, store) {
  container.innerHTML = `
    <div class="v3-component">
      <h3>Presets</h3>
      <div>
        <label>TP presets: <select id="preset_tp"></select></label>
        <button id="copy_tp" style="margin-left:8px">Copy master defaults</button>
      </div>
      <div style="margin-top:8px">
        <label>Partial presets: <select id="preset_partial"></select></label>
        <button id="copy_partial" style="margin-left:8px">Copy master defaults</button>
      </div>
    </div>
  `;

  const tpSel = container.querySelector('#preset_tp');
  const partSel = container.querySelector('#preset_partial');
  const copyTp = container.querySelector('#copy_tp');
  const copyPart = container.querySelector('#copy_partial');

  store.registerField('preset_tp', 'preset_tp');
  store.registerField('preset_partial', 'preset_partial');

  // populate from metadata if available
  const meta = store._state && store._state.metadata ? store._state.metadata : {};
  const tp = (meta.tp_ladders && Object.keys(meta.tp_ladders)) || [];
  const parts = (meta.partial_ladders && Object.keys(meta.partial_ladders)) || [];
  if (tpSel) tp.forEach((k) => tpSel.appendChild(new Option(k, k)));
  if (partSel) parts.forEach((k) => partSel.appendChild(new Option(k, k)));

  if (tpSel) tpSel.addEventListener('change', (e) => store.setState({ preset_tp: e.target.value }));
  if (partSel) partSel.addEventListener('change', (e) => store.setState({ preset_partial: e.target.value }));

  // initialize selections from store
  try {
    const s = (typeof store.getState === 'function') ? store.getState() : {};
    if (s && s.preset_tp && tpSel) tpSel.value = s.preset_tp;
    if (s && s.preset_partial && partSel) partSel.value = s.preset_partial;
  } catch (e) {}

  if (copyTp) copyTp.addEventListener('click', () => {
    const meta2 = store._state && store._state.metadata ? store._state.metadata : {};
    const first = meta2.tp_ladders ? Object.values(meta2.tp_ladders)[0] : null;
    if (first) store.setState({ take_profit: first });
  });
  if (copyPart) copyPart.addEventListener('click', () => {
    const meta2 = store._state && store._state.metadata ? store._state.metadata : {};
    const first = meta2.partial_ladders ? Object.values(meta2.partial_ladders)[0] : null;
    if (first) store.setState({ partial_exit: first });
  });

  store.subscribe((s) => {
    const errs = s && s.validationErrors ? s.validationErrors : [];
    [tpSel, partSel].forEach((el) => el && el.classList && el.classList.remove('field-error'));
    for (const err of errs) {
      if (!err.path) continue;
      const el = store.findElementForPath(err.path);
      if (el && el.classList) el.classList.add('field-error');
    }
  });
}
