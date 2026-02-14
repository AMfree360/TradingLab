(function () {
  // Guided Builder v2 — State model, serializer, deserializer, and validator
  const VERSION = 1;

  function defaults() {
    return {
      _version: VERSION,
      entry_tf: '1h',
      context_tf: '',
      signal_tf: '',
      trigger_tf: '',
      primary_context_type: 'ma_stack',
      align_with_context: false,
      long_enabled: true,
      short_enabled: false,
      ma_type: 'ema',
      ma_fast: 20,
      ma_mid: 50,
      ma_slow: 200,
      stack_mode: 'none',
      slope_mode: 'none',
      slope_lookback: 10,
      min_ma_dist_pct: 0,
      trigger_type: 'pin_bar',
      trigger_valid_for_bars: 0,
      pin_wick_body: 2.0,
      pin_opp_wick_body_max: 1.0,
      pin_min_body_pct: 0.2,
      trigger_ma_type: 'ema',
      trigger_ma_len: 20,
      trigger_don_len: 20,
      trigger_range_len: 20,
      trigger_atr_len: 14,
      trigger_atr_mult: 2.0,
      trigger_custom_bull_expr: '',
      trigger_custom_bear_expr: '',
      context_rules: [],
      signal_rules: [],
      trigger_rules: [],
      risk_per_trade_pct: 1.0,
      stop_type: 'atr',
      atr_length: 14,
      atr_multiplier: 3.0,
      stop_percent: 0.02,
    };
  }

  function clone(obj) {
    return JSON.parse(JSON.stringify(obj));
  }

  function normalize(raw) {
    const out = Object.assign({}, defaults());
    if (!raw || typeof raw !== 'object') return out;

    // copy allowed keys, with basic coercion
    const keys = Object.keys(out);
    for (const k of keys) {
      if (Object.prototype.hasOwnProperty.call(raw, k)) {
        const v = raw[k];
        if (v === null || typeof v === 'undefined') continue;
        try {
          if (typeof out[k] === 'number') {
            const n = Number(v);
            if (!Number.isNaN(n)) out[k] = n;
          } else if (typeof out[k] === 'boolean') {
            out[k] = !!v;
          } else if (Array.isArray(out[k])) {
            out[k] = Array.isArray(v) ? clone(v) : out[k];
          } else {
            out[k] = String(v);
          }
        } catch (e) {
          // ignore and keep default
        }
      }
    }

    out._version = VERSION;
    return out;
  }

  function validate(state) {
    const errors = [];
    if (!state) return ['state missing'];
    const okTfs = ['1m','5m','15m','1h','4h','1d',''];
    if (!okTfs.includes(String(state.entry_tf || '')).toString()) {
      // noop: keep lightweight
    }

    if (state.ma_fast <= 0) errors.push('ma_fast must be > 0');
    if (state.ma_mid <= 0) errors.push('ma_mid must be > 0');
    if (state.ma_slow <= 0) errors.push('ma_slow must be > 0');
    if (state.risk_per_trade_pct <= 0) errors.push('risk_per_trade_pct must be > 0');
    return { ok: errors.length === 0, errors };
  }

  function serialize(state) {
    const s = normalize(state);
    return JSON.stringify({ version: VERSION, state: s });
  }

  function deserialize(raw) {
    if (!raw) return normalize(null);
    try {
      const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
      if (parsed && typeof parsed === 'object') {
        // support both {version, state} and raw-state formats
        const candidate = parsed.state && typeof parsed.state === 'object' ? parsed.state : parsed;
        return normalize(candidate);
      }
    } catch (e) {
      // fallthrough
    }
    return normalize(null);
  }

  // Expose as global
  window.GuidedBuilderV2Model = {
    VERSION,
    defaults,
    normalize,
    validate,
    serialize,
    deserialize,
  };
})();
