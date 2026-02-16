// Simple Node harness to exercise component mounting and ensure they register fields
function assert(cond, msg) { if (!cond) throw new Error(msg || 'Assertion failed'); }

// Minimal mock container that supports innerHTML and querySelector
function makeContainer() {
  const elMap = new Map();
  return {
    innerHTML: '',
    appendChild() {},
    querySelector: (sel) => {
      const id = sel && sel.startsWith('#') ? sel.slice(1) : sel;
      return elMap.get(id) || null;
    },
    __setEl: (id, obj) => elMap.set(id, obj),
  };
}

// Minimal fake store that records registerField calls
class FakeStore {
  constructor() { this._regs = []; this._listeners = []; this._state = { payload: {}, metadata: {}, validationErrors: [] }; }
  registerField(p, id) { this._regs.push({ pattern: p, elementId: id }); }
  setState(s) { Object.assign(this._state.payload, s); this._listeners.forEach((l) => l(this._state)); }
  subscribe(fn) { this._listeners.push(fn); }
  findElementForPath(path) { return null; }
  getState() { return this._state.payload; }
}

// Load components from the filesystem by requiring their files as modules.
const path = require('path');
const root = path.resolve(__dirname, '..', '..');
const compDir = path.join(root, 'gui_launcher', 'static', 'js', 'builder_v3', 'modules', 'components');

const fs = require('fs');
const vm = require('vm');
function loadModule(name) {
  const p = path.join(compDir, name);
  const src = fs.readFileSync(p, 'utf8');
  // Convert a simple `export default` to CommonJS assignment for test harness
  const wrapped = src.replace(/^export default\s+/, 'module.exports = ');
  const module = { exports: {} };
  const script = new vm.Script(wrapped, { filename: p });
  const ctx = vm.createContext({ module, exports: module.exports, require, console, setTimeout, clearTimeout });
  script.runInContext(ctx);
  return module.exports;
}

(function run() {
  const store = new FakeStore();

  const comps = ['context_builder.js','triggers_builder.js','stops_builder.js','trade_management_builder.js','presets_builder.js','signals_builder.js'];
  for (const c of comps) {
    const mod = loadModule(c);
    const cont = makeContainer();
    // pre-populate expected elements so components can query them
    // We'll provide minimal element objects with addEventListener and classList
    const ids = ['ctx-type','ctx-refresh','trigger_type','trigger_atr_len','trigger_atr_mult','stop_mode','stop_fixed_pips','stop_atr_len','tm_tp_enabled','tm_tp_target_r','tm_partial_enabled','tm_partial_level_r','tm_trailing_enabled','tm_trailing_length','preset_tp','preset_partial','copy_tp','copy_partial','signal-rsi','sig-add','signal-rules-list'];
    for (const id of ids) cont.__setEl(id, { id, addEventListener: () => {}, classList: { add: () => {}, remove: () => {} } , value: '' });

    // mount (support both CommonJS-wrapped and ESM default shapes)
    const fn = (typeof mod === 'function') ? mod : (mod && mod.default) ? mod.default : null;
    if (!fn) throw new Error('Component module did not export a default function: ' + c);
    fn(cont, store);
  }

  // basic assertions: store should have registered some fields
  assert(store._regs.length > 0, 'Components did not register any fields');
  console.log('OK');
})();
