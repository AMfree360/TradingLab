(function(){
  const hsEl = document.getElementById('holding_style');
  const flattenBlock = document.getElementById('flatten_block');
  function refresh() {
    const v = hsEl.value;
    flattenBlock.style.opacity = (v === 'swing') ? '0.9' : '1.0';
  }
  if (hsEl) {
    hsEl.addEventListener('change', refresh);
    refresh();
  }
})();