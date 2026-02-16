const { loadComponent } = require('./component_helper');

describe('Triggers component', () => {
  const dirname = require('path').resolve(__dirname, '..');
  let mount;
  beforeAll(() => {
    mount = loadComponent('modules/components/triggers_builder.js', dirname);
  });

  test('initializes inputs from store and clamps values', () => {
    const container = document.createElement('div');
    const store = {
      _state: { payload: { trigger_atr_len: 20, trigger_atr_mult: 3.0, trigger_type: 'pin_bar' }, validationErrors: [] },
      _listeners: [],
      registerField: jest.fn(),
      getState() { return this._state.payload; },
      setState(p) { Object.assign(this._state.payload, p); this._listeners.forEach((l) => l(this._state)); },
      subscribe(fn) { this._listeners.push(fn); },
      findElementForPath() { return null; },
      getValidationErrors() { return this._state.validationErrors; },
    };

    mount(container, store);
    const len = container.querySelector('#trigger_atr_len');
    const mult = container.querySelector('#trigger_atr_mult');
    expect(len.value).toBe('20');
    expect(mult.value).toBe('3');

    // set a too-small value -> clamped to min 1
    len.value = '0';
    len.dispatchEvent(new Event('change'));
    expect(store.getState().trigger_atr_len).toBeGreaterThanOrEqual(1);

    // set a too-large mult -> clamped
    mult.value = '9999';
    mult.dispatchEvent(new Event('change'));
    expect(store.getState().trigger_atr_mult).toBeLessThanOrEqual(50);
  });
});
