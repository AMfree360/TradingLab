const { loadComponent } = require('./component_helper');

describe('Trade Management component', () => {
  const dirname = require('path').resolve(__dirname, '..');
  let mount;
  beforeAll(() => {
    mount = loadComponent('modules/components/trade_management_builder.js', dirname);
  });

  test('initializes and enforces TP > partial guardrail', () => {
    const container = document.createElement('div');
    const store = {
      _state: { payload: { take_profit_enabled: true, take_profit_target_r: 0.05, partial_enabled: true, partial_level_r: 0.04 }, validationErrors: [] },
      _listeners: [],
      registerField: jest.fn(),
      getState() { return this._state.payload; },
      setState(p) { Object.assign(this._state.payload, p); this._listeners.forEach((l) => l(this._state)); },
      subscribe(fn) { this._listeners.push(fn); },
      findElementForPath() { return null; },
      getValidationErrors() { return this._state.validationErrors; },
    };

    mount(container, store);
    const tp = container.querySelector('#tm_tp_target_r');
    const partial = container.querySelector('#tm_partial_level_r');
    expect(tp.value).toBe('0.05');
    expect(partial.value).toBe('0.04');

    // set partial above TP -> guardrail reduces it (or increases TP)
    partial.value = '0.1';
    partial.dispatchEvent(new Event('change'));
    expect(parseFloat(store.getState().partial_level_r)).toBeLessThanOrEqual(parseFloat(store.getState().take_profit_target_r));
  });
});
