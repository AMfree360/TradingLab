export default async function saveFlow({ api, modal, store, payload, statusEl, errEl }) {
  if (statusEl) statusEl.textContent = 'Validating...';
  if (errEl) { errEl.style.display = 'none'; errEl.textContent = ''; }

  try {
    await api.validate(payload);
    if (statusEl) statusEl.textContent = 'Saving...';
    try {
      const res = await api.save('save', payload.draft_id || null, payload, {});
      if (statusEl) statusEl.textContent = 'Saved';
      if (res && res.data && res.data.next) window.location.href = res.data.next;
      return res;
    } catch (saveErr) {
      if (saveErr && saveErr.status === 409) {
        const suggested = (saveErr.body && (saveErr.body.detail && saveErr.body.detail.suggested_name)) || (saveErr.body && saveErr.body.suggested_name) || '';
        const choice = modal ? await modal.showConflict(suggested, (payload && payload.name) ? payload.name : '') : null;
        if (choice && choice.action === 'overwrite') {
          const r2 = await api.save('save', payload.draft_id || null, payload, { force: true });
          if (statusEl) statusEl.textContent = 'Saved (overwritten)';
          if (r2 && r2.data && r2.data.next) window.location.href = r2.data.next;
          return r2;
        }
        if (!choice || choice.action === 'cancel') {
          if (statusEl) statusEl.textContent = 'Save cancelled';
          return null;
        }
        // rename flow
        const renameRes = modal ? await modal.promptRename(suggested, (payload && payload.name) ? payload.name : '') : null;
        if (renameRes && renameRes.action === 'rename' && renameRes.name) {
          const r3 = await api.save('save', payload.draft_id || null, payload, { rename_to: renameRes.name });
          if (statusEl) statusEl.textContent = 'Saved (renamed)';
          if (r3 && r3.data && r3.data.next) window.location.href = r3.data.next;
          return r3;
        }
        if (statusEl) statusEl.textContent = 'Save cancelled';
        return null;
      }
      throw saveErr;
    }
  } catch (e) {
    if (statusEl) statusEl.textContent = 'Validation failed';
    const payloadErrs = (e && e.body && e.body.errors) ? e.body.errors : (e && e.errors) ? e.errors : (e && e.detail && e.detail.errors) ? e.detail.errors : null;
    if (payloadErrs && payloadErrs.length) {
      store.setValidationErrors(payloadErrs);
      if (errEl) { errEl.style.display = 'block'; errEl.textContent = JSON.stringify(payloadErrs, null, 2); }
    } else {
      store.setValidationErrors([]);
      if (errEl) { errEl.style.display = 'block'; errEl.textContent = JSON.stringify(e, null, 2); }
    }
    return null;
  }
}
