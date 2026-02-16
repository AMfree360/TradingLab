const { loadComponent } = require('./component_helper');

describe('Presets component', () => {
  const dirname = require('path').resolve(__dirname, '..');
  let mount;
  beforeAll(() => {
    mount = loadComponent('modules/components/presets_builder.js', dirname);
  });

  test('populates selects from metadata and copy master defaults', () => {
    const container = document.createElement('div');
    const store = {
      _state: { payload: {}, metadata: { tp_ladders: { fast: [0.01,0.02], slow: [0.05] }, partial_ladders: { p1: [0.02] } }, validationErrors: [] },
      _listeners: [],
      registerField: jest.fn(),
      getState() { return this._state.payload; },
      setState(p) { Object.assign(this._state.payload, p); this._listeners.forEach((l) => l(this._state)); },
      subscribe(fn) { this._listeners.push(fn); },
      findElementForPath() { return null; },
      getValidationErrors() { return this._state.validationErrors; },
    };

    mount(container, store);
    const tpSel = container.querySelector('#preset_tp');
    const partSel = container.querySelector('#preset_partial');
    expect(tpSel.options.length).toBeGreaterThan(0);
    expect(partSel.options.length).toBeGreaterThan(0);

    const copyBtn = container.querySelector('#copy_tp');
    copyBtn.click();
    expect(store.getState().take_profit).toEqual(store._state.metadata.tp_ladders.fast);
  });
});
