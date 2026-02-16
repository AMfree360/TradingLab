(function() {
  function preferredTheme() {
    try {
      var saved = localStorage.getItem('tl_theme');
      if (saved === 'light' || saved === 'dark') return saved;
    } catch (e) {}
    try {
      return (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light';
    } catch (e) {
      return 'light';
    }
  }

  function applyTheme(theme) {
    var t = (theme === 'dark') ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', t);
    try { localStorage.setItem('tl_theme', t); } catch (e) {}
    try { window.dispatchEvent(new Event('tl-theme-change')); } catch (e) {}
    var btn = document.getElementById('themeToggle');
    if (btn) btn.textContent = (t === 'dark') ? 'Dark' : 'Light';
  }

  applyTheme(preferredTheme());

  var btn = document.getElementById('themeToggle');
  if (btn) {
    btn.addEventListener('click', function() {
      var cur = document.documentElement.getAttribute('data-theme') || 'light';
      applyTheme(cur === 'dark' ? 'light' : 'dark');
    });
  }
})();
