// Minimal StateStore skeleton for Builder V3
export default class StateStore {
  constructor() {
    this._state = { payload: {}, metadata: {}, validationErrors: [] };
    this._listeners = new Set();
    this._fieldRegistry = []; // entries: { pattern: string, elementId: string }
  }

  setMetadata(meta) {
    this._state.metadata = meta && meta.data ? meta.data : meta;
    this._emit();
  }

  getState() { return this._state.payload; }
  setState(patch) { this._state.payload = Object.assign({}, this._state.payload, patch); this._emit(); }
  setValidationErrors(errors) { this._state.validationErrors = Array.isArray(errors) ? errors : []; this._emit(); }
  clearValidationErrors() { this._state.validationErrors = []; this._emit(); }
  getValidationErrors() { return this._state.validationErrors || []; }
  subscribe(fn) { this._listeners.add(fn); return () => this._listeners.delete(fn); }
  _emit() { for (const fn of this._listeners) fn(this._state); }

  registerField(pattern, elementId) {
    if (!pattern || !elementId) return;
    this._fieldRegistry.push({ pattern, elementId });
  }

  findElementForPath(errorPath) {
    if (!errorPath) return null;
    for (const entry of this._fieldRegistry) {
      if (this._patternMatches(entry.pattern, errorPath)) {
        const el = document.getElementById(entry.elementId);
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
    // indexMap: { oldIndex: newIndex }
    if (!patternBase || !indexMap || typeof indexMap !== 'object') return;
    const re = new RegExp(`(${patternBase})\\.(\\d+)\\.(.*)`);
    this._fieldRegistry = this._fieldRegistry.map((entry) => {
      const m = entry.elementId.match(re);
      if (!m) return entry;
      const oldIdx = m[2];
      const rest = m[3];
      const newIdx = (oldIdx in indexMap) ? indexMap[oldIdx] : null;
      if (newIdx === null || newIdx === undefined) return null; // drop if not remapped
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
      // support 'signal_rules[]' form (no dot between name and index)
      if (ps.endsWith('[]')) {
        const base = ps.slice(0, -2);
        if (base === ts) continue;
      }
      return false;
    }
    return true;
  }

  serializeForSubmit() { return Object.assign({}, this._state.payload); }
}
