(function(){
  const dataEl = document.getElementById('guided-step2-data');
  let instruments = {};
  let initialSymbol = '';
  try {
    if (dataEl) {
      const json = JSON.parse(dataEl.textContent || '{}');
      instruments = json.instruments || {};
      initialSymbol = json.initialSymbol || '';
    }
  } catch (e) { instruments = {}; initialSymbol = ''; }

  const assetEl = document.getElementById('asset_class');
  const symbolEl = document.getElementById('symbol');
  const marketTypeEl = document.getElementById('market_type');
  const exchangeEl = document.getElementById('exchange');
  const marketTypeHintEl = document.getElementById('market_type_hint');

  function refreshSymbols() {
    const asset = assetEl.value;
    const list = instruments[asset] || [];
    symbolEl.innerHTML = '';

    const optCustom = document.createElement('option');
    optCustom.value = 'CUSTOM';
    optCustom.textContent = 'Custom…';
    symbolEl.appendChild(optCustom);

    for (const s of list) {
      const o = document.createElement('option');
      o.value = s;
      o.textContent = s;
      symbolEl.appendChild(o);
    }

    if (initialSymbol && Array.from(symbolEl.options).some(o => o.value === initialSymbol)) {
      symbolEl.value = initialSymbol;
    } else {
      symbolEl.value = list.includes('BTCUSDT') ? 'BTCUSDT' : (list[0] || 'CUSTOM');
    }
  }

  function refreshMarketDefaults() {
    const asset = assetEl.value;
    if (asset === 'futures') {
      marketTypeEl.value = 'futures';
      marketTypeEl.disabled = true;
      marketTypeHintEl.textContent = 'Futures market type is fixed.';
      if (!exchangeEl.value || exchangeEl.value.toLowerCase() === 'binance') exchangeEl.value = 'cme';
    } else if (asset === 'fx') {
      marketTypeEl.value = 'spot';
      marketTypeEl.disabled = true;
      marketTypeHintEl.textContent = 'FX market type is spot.';
      if (!exchangeEl.value || exchangeEl.value.toLowerCase() === 'binance') exchangeEl.value = 'oanda';
    } else {
      marketTypeEl.disabled = false;
      marketTypeHintEl.textContent = 'Crypto can be spot or futures.';
      if (!exchangeEl.value) exchangeEl.value = 'binance';
    }
  }

  assetEl.addEventListener('change', () => {
    refreshMarketDefaults();
    refreshSymbols();
  });
  refreshMarketDefaults();
  refreshSymbols();
})();