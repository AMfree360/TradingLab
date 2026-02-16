const { loadComponent } = require('./component_helper');

describe('Context component', () => {
  const dirname = require('path').resolve(__dirname, '..');
  let mount;
  beforeAll(() => {
    mount = loadComponent('modules/components/context_builder.js', dirname);
  });

  test('initializes and updates primary_context_type, shows inline messages on error', () => {
    const container = document.createElement('div');
    const store = {
      _state: { payload: { primary_context_type: 'ma_stack' }, validationErrors: [] },
      _listeners: [],
      registerField: jest.fn(),
      getState() { return this._state.payload; },
      setState(p) { Object.assign(this._state.payload, p); this._listeners.forEach((l) => l(this._state)); },
      subscribe(fn) { this._listeners.push(fn); },
      findElementForPath() { return null; },
      getValidationErrors() { return this._state.validationErrors; },
    };

    mount(container, store);
    const sel = container.querySelector('#ctx-type');
    expect(sel.value).toBe('ma_stack');

    sel.value = 'price_vs_ma';
    sel.dispatchEvent(new Event('change'));
    expect(store.getState().primary_context_type).toBe('price_vs_ma');

    // trigger validation error and ensure inline message shows
    store._state.validationErrors = [{ path: 'primary_context_type', message: 'invalid' }];
    store._listeners.forEach((l) => l(store._state));
    const msg = container.querySelector('#ctx-type-msg');
    expect(msg && msg.textContent).toBe('invalid');
  });
});
