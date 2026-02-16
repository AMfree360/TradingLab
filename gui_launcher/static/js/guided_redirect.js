(function(){
  const el = document.getElementById('redirectData');
  const url = el ? el.getAttribute('data-url') : null;
  try {
    if (url) window.location.href = url;
  } catch (e) {
    // fallback: nothing (meta refresh + link already present)
  }
})();