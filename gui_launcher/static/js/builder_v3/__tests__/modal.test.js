const fs = require('fs');
const path = require('path');

// Load modal.js into CommonJS for testing
function loadModal() {
  const p = path.join(__dirname, '..', 'modules', 'modal.js');
  let src = fs.readFileSync(p, 'utf8');
  // replace ES module export with CommonJS
  src = src.replace(/export default/, 'module.exports =');
  const module = { exports: {} };
  const fn = new Function('module', 'exports', 'require', src);
  fn(module, module.exports, require);
  return module.exports;
}

describe('BuilderModal', () => {
  let BuilderModal;
  beforeAll(() => { BuilderModal = loadModal(); });

  afterEach(() => {
    // clean up DOM
    const el = document.getElementById('v3-modal-root');
    if (el && el.parentNode) el.parentNode.removeChild(el);
  });

  test('showConflict resolves overwrite when overwrite clicked', async () => {
    const m = new BuilderModal();
    // call showConflict and simulate click after a tick
    const p = m.showConflict('suggested_name', 'current');
    // wait a tick so modal is visible
    await new Promise((r) => setTimeout(r, 10));
    const ov = document.getElementById('v3-modal-overwrite');
    expect(ov).toBeTruthy();
    ov.click();
    const res = await p;
    expect(res).toEqual(expect.objectContaining({ action: 'overwrite' }));
  });

  test('promptRename returns rename action with provided name', async () => {
    const m = new BuilderModal();
    const p = m.promptRename('sug', 'cur');
    await new Promise((r) => setTimeout(r, 10));
    const input = document.getElementById('v3-modal-rename-input');
    expect(input).toBeTruthy();
    input.value = 'my-new-name';
    const renameBtn = document.getElementById('v3-modal-rename-btn');
    // clicking rename button resolves with {action: 'rename', name: ...}
    renameBtn.click();
    const res = await p;
    expect(res).toEqual(expect.objectContaining({ action: 'rename', name: 'my-new-name' }));
  });
});
