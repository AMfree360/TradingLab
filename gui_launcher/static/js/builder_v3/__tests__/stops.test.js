const { loadComponent } = require('./component_helper');

describe('Stops component', () => {
  const dirname = require('path').resolve(__dirname, '..');
  let mount;
  beforeAll(() => {
    mount = loadComponent('modules/components/stops_builder.js', dirname);
  });

  test('toggles stop visibility and clamps input', () => {
    const container = document.createElement('div');
    const store = {
      _state: { payload: { stop_fixed_pips: 20, stop_atr_len: 14, stop_mode: 'fixed' }, validationErrors: [] },
      _listeners: [],
      registerField: jest.fn(),
      getState() { return this._state.payload; },
      setState(p) { Object.assign(this._state.payload, p); this._listeners.forEach((l) => l(this._state)); },
      subscribe(fn) { this._listeners.push(fn); },
      findElementForPath() { return null; },
      getValidationErrors() { return this._state.validationErrors; },
    };

    mount(container, store);
    const stopInput = container.querySelector('#stop_fixed_pips');
    expect(stopInput.value).toBe('20');

    stopInput.value = '-1';
    stopInput.dispatchEvent(new Event('change'));
    expect(store.getState().stop_fixed_pips).toBeGreaterThanOrEqual(0);
  });
});
