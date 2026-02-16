/* Guided Builder v2 — Light test harness
   Purpose: attach minimal, non-invasive listeners to form controls so we can
   verify basic interactions without the full state/model/renderer running.
*/
(function () {
  function dbg(...args) { try { console.debug('guided_builder_v2_light:', ...args); } catch (e) {} }

  function init() {
    const form = document.querySelector('form[action="/create-strategy-guided/step4"]') || document.getElementById('guided_step4_form');
    if (!form) return dbg('form not found; light harness not attached');
    dbg('attaching light harness to form');

    // Attach change listeners to selects/inputs/textarea and click to buttons
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

    // Add a visible debugging card below the form so testers know the harness is active
    if (!document.getElementById('guided_builder_v2_light_status')) {
      const status = document.createElement('div');
      status.id = 'guided_builder_v2_light_status';
      status.className = 'card';
      status.style.marginTop = '12px';
      status.innerHTML = '<strong>Light JS harness active</strong><div class="muted">Interactions logged to console.</div>';
      form.parentNode.insertBefore(status, form.nextSibling);
    }

    // Keep the harness passive: do not prevent default form submit
    form.addEventListener('submit', (ev) => dbg('form submit (native)'));
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
