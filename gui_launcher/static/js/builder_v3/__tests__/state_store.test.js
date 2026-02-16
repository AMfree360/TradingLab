const { loadComponent } = require('./component_helper');
const path = require('path');

describe('StateStore module (light)', () => {
  test('register/unregister and findElementForPath mapping', () => {
    const dirname = path.resolve(__dirname, '..');
    const StoreClass = loadComponent('modules/state_store.js', dirname);
    const store = new StoreClass();
    // ensure API exists
    expect(typeof store.registerField).toBe('function');
    expect(typeof store.unregisterField).toBe('function');
    // create fake element in DOM to be found
    const elId = 'foo.bar.0.baz';
    const el = document.createElement('div'); el.id = elId; document.body.appendChild(el);
    store.registerField('foo.bar.[].baz', elId);
    const found = store.findElementForPath('foo.bar.0.baz');
    // should return the DOM element
    expect(found).toBe(el);
    store.unregisterField(elId);
    const notFound = store.findElementForPath('foo.bar.0.baz');
    expect(notFound).toBeNull();
    el.remove();
  });
});
