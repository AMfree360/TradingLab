/* ARCHIVE: light harness for guided builder (moved to archive)
*/
(function () {
  function dbg(...args) { try { console.debug('guided_builder_v2_light:', ...args); } catch (e) {} }

  function init() {
    const form = document.querySelector('form[action="/create-strategy-guided/step4"]') || document.getElementById('guided_step4_form');
    if (!form) return dbg('form not found; light harness not attached');
    dbg('attaching light harness to form');

    const controls = Array.from(form.querySelectorAll('input[name], select[name], textarea[name], button'));
    for (const c of controls) {
      try {
        if (c.tagName === 'BUTTON' || c.type === 'button' || c.type === 'submit') {
          c.addEventListener('click', (ev) => {
            dbg('button click', c.id || c.name || c.textContent.trim());
            c.classList.toggle('debug-active');
          });
        } else {
          c.addEventListener('change', () => dbg('control change', c.name || c.id, c.type || c.tagName, c.value));
          c.addEventListener('input', () => dbg('control input', c.name || c.id, c.type || c.tagName, c.value));
        }
      } catch (e) { dbg('attach error', e); }
    }

    if (!document.getElementById('guided_builder_v2_light_status')) {
      const status = document.createElement('div');
      status.id = 'guided_builder_v2_light_status';
      status.className = 'card';
      status.style.marginTop = '12px';
      status.innerHTML = '<strong>Light JS harness active</strong><div class="muted">Interactions logged to console.</div>';
      form.parentNode.insertBefore(status, form.nextSibling);
    }

    form.addEventListener('submit', (ev) => dbg('form submit (native)'));
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
