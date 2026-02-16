(function(){
  function showHideByRadio(name, value, el) {
    const checked = document.querySelector('input[name="' + name + '"]:checked');
    if (!checked) return; el.style.display = (checked.value === value) ? 'block' : 'none';
  }
  function showHideByCheckbox(id, el) { const cb = document.getElementsByName(id)[0]; if (!cb) return; el.style.display = cb.checked ? 'block' : 'none'; }

  const calCustom = document.getElementById('calendar_custom');
  const tpCustom = document.getElementById('tp_custom');
  const partialFields = document.getElementById('partial_fields');
  const trailingFields = document.getElementById('trailing_fields');
  const exitCtxFields = document.getElementById('exitCtxFields');
  const maxDailyLossEnabledEl = document.getElementById('max_daily_loss_enabled');
  const maxDailyLossPctEl = document.getElementById('max_daily_loss_pct');
  const overrideCommEl = document.getElementById('override_commissions');
  const commissionsEl = document.getElementById('commissions');
  const overrideSlipEl = document.getElementById('override_slippage');
  const slippageEl = document.getElementById('slippage_ticks');
  const stopSeparateCb = document.getElementsByName('stop_separate')[0];
  const stopSharedCard = document.getElementById('stopSharedCard');
  const stopSeparateCard = document.getElementById('stopSeparateCard');
  const sharedTypeSel = document.getElementById('stop_type_shared');

  function setEnabled(root, enabled) { if (!root) return; root.querySelectorAll('input,select,textarea').forEach(el => { el.disabled = !enabled; }); }

  function refresh() {
    if (calCustom) showHideByRadio('calendar_mode', 'custom', calCustom);
    if (tpCustom) showHideByRadio('tp_mode', 'custom', tpCustom);
    if (partialFields) { const cb = document.getElementsByName('partial_enabled')[0]; partialFields.style.display = (cb && cb.checked) ? 'block' : 'none'; }
    if (trailingFields) { const cb = document.getElementsByName('trailing_enabled')[0]; trailingFields.style.display = (cb && cb.checked) ? 'block' : 'none'; }
    if (exitCtxFields) { const cb = document.getElementsByName('exit_ctx_enabled')[0]; exitCtxFields.style.display = (cb && cb.checked) ? 'block' : 'none'; }

    if (maxDailyLossPctEl && maxDailyLossEnabledEl) maxDailyLossPctEl.disabled = !maxDailyLossEnabledEl.checked;
    if (commissionsEl && overrideCommEl) commissionsEl.disabled = !overrideCommEl.checked;
    if (slippageEl && overrideSlipEl) slippageEl.disabled = !overrideSlipEl.checked;

    const separateOn = !!(stopSeparateCb && stopSeparateCb.checked);
    if (stopSharedCard) stopSharedCard.style.display = separateOn ? 'none' : 'block';
    if (stopSeparateCard) stopSeparateCard.style.display = separateOn ? 'block' : 'none';

    setEnabled(stopSharedCard, !separateOn);
    setEnabled(stopSeparateCard, separateOn);

    function showStopFields(which, t) {
      document.querySelectorAll('.stopFields[data-stop-for="' + which + '"]').forEach(el => {
        const want = el.getAttribute('data-stop-type');
        const show = (want === t);
        el.style.display = show ? 'block' : 'none';
        el.querySelectorAll('input,select,textarea').forEach(c => { c.disabled = !show; });
      });
    }

    if (!separateOn) {
      const t = String(sharedTypeSel?.value || 'atr');
      showStopFields('shared', t);
    } else {
      document.querySelectorAll('.stopTypeSel').forEach(sel => {
        const which = sel.getAttribute('data-stop-for');
        const t = String(sel.value || 'atr');
        if (which) showStopFields(which, t);
      });
    }
  }

  document.addEventListener('change', refresh);

  function parseCsvInts(s) {
    if (!s) return [];
    return String(s).split(',').map(x => parseInt(x.trim(), 10)).filter(x => Number.isFinite(x));
  }

  function copyMasterCalendar() {
    const md = document.getElementById('masterDefaults');
    if (!md || !md.dataset) return;
    const custom = document.querySelector('input[name="calendar_mode"][value="custom"]'); if (custom) custom.checked = true;
    const allowed = new Set(parseCsvInts(md.dataset.allowedDays));
    document.querySelectorAll('input[name="dow"]').forEach(cb => { const v = parseInt(cb.value, 10); cb.checked = allowed.has(v); });
    const sa = document.getElementsByName('session_Asia')[0]; const sl = document.getElementsByName('session_London')[0]; const sn = document.getElementsByName('session_NewYork')[0];
    if (sa) sa.checked = md.dataset.sessionAsiaEnabled === '1'; if (sl) sl.checked = md.dataset.sessionLondonEnabled === '1'; if (sn) sn.checked = md.dataset.sessionNewyorkEnabled === '1';
    const asa = document.getElementsByName('session_Asia_start')[0]; const aea = document.getElementsByName('session_Asia_end')[0]; const asl = document.getElementsByName('session_London_start')[0]; const ael = document.getElementsByName('session_London_end')[0]; const asn = document.getElementsByName('session_NewYork_start')[0]; const aen = document.getElementsByName('session_NewYork_end')[0];
    if (asa) asa.value = md.dataset.sessionAsiaStart || asa.value; if (aea) aea.value = md.dataset.sessionAsiaEnd || aea.value; if (asl) asl.value = md.dataset.sessionLondonStart || asl.value; if (ael) ael.value = md.dataset.sessionLondonEnd || ael.value; if (asn) asn.value = md.dataset.sessionNewyorkStart || asn.value; if (aen) aen.value = md.dataset.sessionNewyorkEnd || aen.value;
    refresh();
  }

  function copyMasterTp() {
    const md = document.getElementById('masterDefaults'); if (!md || !md.dataset) return; const custom = document.querySelector('input[name="tp_mode"][value="custom"]'); if (custom) custom.checked = true; const tr = document.getElementsByName('tp1_target_r')[0]; const ep = document.getElementsByName('tp1_exit_pct')[0]; const en = document.getElementsByName('tp1_enabled')[0]; if (en) en.checked = true; if (tr) tr.value = md.dataset.tpTargetR || tr.value; if (ep) ep.value = md.dataset.tpExitPct || ep.value; refresh(); }

  function copyMasterPartial() { const md = document.getElementById('masterDefaults'); if (!md || !md.dataset) return; const en = document.getElementsByName('partial_enabled')[0]; if (en) en.checked = md.dataset.partialEnabled === '1'; const l1en = document.getElementsByName('partial1_enabled')[0]; const lr = document.getElementsByName('partial1_level_r')[0]; const ep = document.getElementsByName('partial1_exit_pct')[0]; if (l1en) l1en.checked = true; if (lr) lr.value = md.dataset.partialLevelR || lr.value; if (ep) ep.value = md.dataset.partialExitPct || ep.value; refresh(); }

  function copyMasterTrailing() { const md = document.getElementById('masterDefaults'); if (!md || !md.dataset) return; const en = document.getElementsByName('trailing_enabled')[0]; if (en) en.checked = md.dataset.trailingEnabled === '1'; const len = document.getElementsByName('trailing_length')[0]; const ar = document.getElementsByName('trailing_activation_r')[0]; const st = document.getElementsByName('trailing_stepped')[0]; const mm = document.getElementsByName('trailing_min_move_pips')[0]; if (len) len.value = md.dataset.trailingLength || len.value; if (ar) ar.value = md.dataset.trailingActivationR || ar.value; if (st) st.checked = md.dataset.trailingStepped === '1'; if (mm) mm.value = md.dataset.trailingMinMovePips || mm.value; refresh(); }

  const calBtn = document.getElementById('copyMasterCalendarBtn'); if (calBtn) calBtn.addEventListener('click', copyMasterCalendar);
  const tpBtn = document.getElementById('copyMasterTpBtn'); if (tpBtn) tpBtn.addEventListener('click', copyMasterTp);
  const peBtn = document.getElementById('copyMasterPartialBtn'); if (peBtn) peBtn.addEventListener('click', copyMasterPartial);
  const tsBtn = document.getElementById('copyMasterTrailingBtn'); if (tsBtn) tsBtn.addEventListener('click', copyMasterTrailing);

  refresh();
})();