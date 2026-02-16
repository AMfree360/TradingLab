// Minimal Plotly stub for tests — provides the `react` method used by the app.
(function (global) {
  try {
    global.Plotly = global.Plotly || {};
    global.Plotly.react = function (el, data, layout, opts) {
      // create a lightweight placeholder so the DOM gets updated for tests
      try {
        if (typeof el === 'string') el = document.getElementById(el);
        if (!el) return;
        el.innerHTML = '<div class="plotly-stub">[Plot]</div>';
      } catch (e) {
        /* swallow */
      }
    };
  } catch (e) {
    // noop
  }
})(window);
