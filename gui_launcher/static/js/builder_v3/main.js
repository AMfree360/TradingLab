// Archived builder_v3 entry — no-op stub to avoid runtime/parse errors.
(function () {
  try { console.warn('[archived] builder_v3/main.js disabled (archived)'); } catch (e) {}
})();
  // tests that may otherwise wait on external resource load for the native
  // `load` event.
  bootstrap().then(() => {
    try {
      if (window && (window.Cypress || window.__FORCE_DISPATCH_LOAD)) {
        setTimeout(() => {
          try { window.dispatchEvent(new Event('load')); } catch (e) {}
        }, 0);
      }
    } catch (e) {}
  });
