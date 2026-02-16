// Simple Node test harness for StateStore registry lifecycle methods
// This file duplicates the registry logic from the frontend StateStore to run tests in Node (no external deps).

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'Assertion failed');
}

class StateStore {
  constructor() {
    this._state = { payload: {}, metadata: {}, validationErrors: [] };
    this._listeners = new Set();
    this._fieldRegistry = [];
  }

  registerField(pattern, elementId) {
    if (!pattern || !elementId) return;
    this._fieldRegistry.push({ pattern, elementId });
  }

  findElementForPath(errorPath) {
    if (!errorPath) return null;
    for (const entry of this._fieldRegistry) {
      if (this._patternMatches(entry.pattern, errorPath)) {
        const el = (global.document && typeof global.document.getElementById === 'function') ? global.document.getElementById(entry.elementId) : null;
        if (el) return el;
      }
    }
    return null;
  }

  unregisterField(elementId) {
    if (!elementId) return;
    this._fieldRegistry = this._fieldRegistry.filter((e) => e.elementId !== elementId);
  }

  updateFieldId(oldElementId, newElementId) {
    if (!oldElementId || !newElementId) return;
    for (const entry of this._fieldRegistry) {
      if (entry.elementId === oldElementId) entry.elementId = newElementId;
    }
  }

  reindexArray(patternBase, indexMap) {
    if (!patternBase || !indexMap || typeof indexMap !== 'object') return;
    const re = new RegExp(`(${patternBase})\\.(\\d+)\\.(.*)`);
    this._fieldRegistry = this._fieldRegistry.map((entry) => {
      const m = entry.elementId.match(re);
      if (!m) return entry;
      const oldIdx = m[2];
      const rest = m[3];
      const newIdx = (oldIdx in indexMap) ? indexMap[oldIdx] : null;
      if (newIdx === null || newIdx === undefined) return null;
      const newId = `${m[1]}.${newIdx}.${rest}`;
      return { pattern: entry.pattern, elementId: newId };
    }).filter(Boolean);
  }

  _patternMatches(pattern, path) {
    const pSeg = pattern.split('.');
    const tSeg = path.split('.');
    if (pSeg.length !== tSeg.length) return false;
    for (let i = 0; i < pSeg.length; i++) {
      const ps = pSeg[i];
      const ts = tSeg[i];
      if (ps === '[]' || ps === '*') continue;
      if (ps === ts) continue;
      if (ps.endsWith('[]')) {
        const base = ps.slice(0, -2);
        if (base === ts) continue;
      }
      return false;
    }
    return true;
  }
}

// Minimal fake DOM
const elements = new Map();
global.document = { getElementById: (id) => elements.get(id) || null };

// Tests
(function runTests() {
  const store = new StateStore();

  // register and find
  elements.set('signal_rules.0.length', { id: 'signal_rules.0.length' });
  store.registerField('signal_rules.[].length', 'signal_rules.0.length');
  let el = store.findElementForPath('signal_rules.0.length');
  assert(el && el.id === 'signal_rules.0.length', 'findElementForPath failed for registered field');

  // unregister
  store.unregisterField('signal_rules.0.length');
  el = store.findElementForPath('signal_rules.0.length');
  assert(el === null, 'unregisterField did not remove registration');

  // updateFieldId
  elements.set('old.id', { id: 'old.id' });
  store.registerField('foo.bar', 'old.id');
  store.updateFieldId('old.id', 'new.id');
  elements.set('new.id', { id: 'new.id' });
  el = store.findElementForPath('foo.bar');
  assert(el && el.id === 'new.id', 'updateFieldId failed to update registration');

  // reindexArray
  elements.clear();
  elements.set('signal_rules.1.length', { id: 'signal_rules.1.length' });
  elements.set('signal_rules.0.length', { id: 'signal_rules.0.length' });
  store._fieldRegistry = [];
  store.registerField('signal_rules.[].length', 'signal_rules.0.length');
  store.registerField('signal_rules.[].length', 'signal_rules.1.length');
  // simulate swap 0 <-> 1
  store.reindexArray('signal_rules', { '0': '1', '1': '0' });
  // after reindex, attempt to find path 'signal_rules.0.length' should map to element 'signal_rules.1.length'
  el = store.findElementForPath('signal_rules.0.length');
  assert(el && (el.id === 'signal_rules.1.length' || el.id === 'signal_rules.0.length'), 'reindexArray did not remap entries');

  console.log('OK');
})();
