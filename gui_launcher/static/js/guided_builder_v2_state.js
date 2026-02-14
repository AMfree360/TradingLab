(function () {
  // Guided Builder v2 — State manager
  // Depends on: GuidedBuilderV2Model (window.GuidedBuilderV2Model)

  function deepClone(v) {
    return JSON.parse(JSON.stringify(v));
  }

  function deepGet(obj, path) {
    if (!path) return obj;
    const parts = String(path).split('.');
    let cur = obj;
    for (const p of parts) {
      if (cur == null) return undefined;
      cur = cur[p];
    }
    return cur;
  }

  function deepSet(obj, path, value) {
    if (!path) return;
    const parts = String(path).split('.');
    let cur = obj;
    for (let i = 0; i < parts.length - 1; i++) {
      const p = parts[i];
      if (typeof cur[p] !== 'object' || cur[p] === null) cur[p] = {};
      cur = cur[p];
    }
    cur[parts[parts.length - 1]] = value;
  }

  function mergeDeep(target, src) {
    if (!src) return target;
    for (const k of Object.keys(src)) {
      const sv = src[k];
      const tv = target[k];
      if (Array.isArray(sv)) {
        target[k] = deepClone(sv);
      } else if (sv && typeof sv === 'object') {
        if (!tv || typeof tv !== 'object') target[k] = {};
        mergeDeep(target[k], sv);
      } else {
        target[k] = sv;
      }
    }
    return target;
  }

  function createStateManager(opts) {
    const debounceMs = (opts && opts.debounceMs) || 120;

    let state = GuidedBuilderV2Model ? GuidedBuilderV2Model.defaults() : {};
    let listeners = new Set();
    let dirty = false;
    let timer = null;

    function notify() {
      const snapshot = deepClone(state);
      for (const cb of Array.from(listeners)) {
        try {
          cb(snapshot);
        } catch (e) {
          console.error('state listener error', e);
        }
      }
    }

    function scheduleNotify() {
      dirty = true;
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        dirty = false;
        timer = null;
        notify();
      }, debounceMs);
    }

    return {
      load(initial) {
        if (initial == null) {
          state = GuidedBuilderV2Model ? GuidedBuilderV2Model.defaults() : {};
        } else {
          state = GuidedBuilderV2Model ? GuidedBuilderV2Model.normalize(initial) : deepClone(initial);
        }
        scheduleNotify();
      },
      get(path) {
        if (!path) return deepClone(state);
        return deepClone(deepGet(state, path));
      },
      patch(partial) {
        if (!partial || typeof partial !== 'object') return;
        mergeDeep(state, partial);
        scheduleNotify();
      },
      set(path, value) {
        if (!path) return;
        deepSet(state, path, value);
        scheduleNotify();
      },
      subscribe(cb) {
        if (typeof cb !== 'function') return () => {};
        listeners.add(cb);
        // immediate call with snapshot
        try { cb(deepClone(state)); } catch (e) {}
        return () => listeners.delete(cb);
      },
      flush() {
        if (timer) { clearTimeout(timer); timer = null; }
        if (dirty) { dirty = false; notify(); }
      },
      serialize() {
        return GuidedBuilderV2Model ? GuidedBuilderV2Model.serialize(state) : JSON.stringify(state);
      },
      raw() { return deepClone(state); }
    };
  }

  window.GuidedBuilderV2State = createStateManager({ debounceMs: 120 });
})();
