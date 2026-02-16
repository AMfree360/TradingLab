const { loadComponent } = require('./component_helper');

describe('Signals component', () => {
  const dirname = require('path').resolve(__dirname, '..');
  let mount;
  beforeAll(() => {
    mount = loadComponent('modules/components/signals_builder.js', dirname);
  });

  test('adds a rule and clamps length input', () => {
    const container = document.createElement('div');
    const store = {
      _state: { payload: { signal_rules: [] }, validationErrors: [] },
      _listeners: [],
      registerField: jest.fn(),
      unregisterField: jest.fn(),
      getState() { return this._state.payload; },
      setState(p) { Object.assign(this._state.payload, p); this._listeners.forEach((l) => l(this._state)); },
      subscribe(fn) { this._listeners.push(fn); },
      findElementForPath() { return null; },
      getValidationErrors() { return this._state.validationErrors; },
    };

    mount(container, store);
    const addBtn = container.querySelector('#sig-add');
    addBtn.click();
    const input = container.querySelector('[id="signal_rules.0.length"]');
    expect(input).not.toBeNull();

    input.value = '0';
    input.dispatchEvent(new Event('change'));
    expect(store.getState().signal_rules[0].length).toBeGreaterThanOrEqual(1);
  });
});
