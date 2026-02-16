const fs = require('fs');
const path = require('path');

function loadSaveFlow() {
  const p = path.join(__dirname, '..', 'modules', 'save_flow.js');
  let src = fs.readFileSync(p, 'utf8');
  src = src.replace(/export default/, 'module.exports =');
  const module = { exports: {} };
  const fn = new Function('module', 'exports', 'require', src);
  fn(module, module.exports, require);
  return module.exports;
}

describe('saveFlow', () => {
  let saveFlow;
  beforeAll(() => { saveFlow = loadSaveFlow(); });

  function makeEl() {
    const el = document.createElement('span');
    document.body.appendChild(el);
    return el;
  }

  test('handles 409 then overwrite path', async () => {
    const api = {
      validate: jest.fn(() => Promise.resolve()),
      save: jest.fn()
        .mockRejectedValueOnce({ status: 409, body: { suggested_name: 'sugg' } })
        .mockResolvedValueOnce({ ok: true, data: { status: 'ok' } }),
    };
    const modal = {
      showConflict: jest.fn(() => Promise.resolve({ action: 'overwrite' })),
      promptRename: jest.fn(),
    };
    const store = { setValidationErrors: jest.fn() };
    const statusEl = makeEl();
    statusEl.id = 'test-status-ow';
    const res = await saveFlow({ api, modal, store, payload: { draft_id: null, name: 'old' }, statusEl });
    expect(api.validate).toHaveBeenCalled();
    expect(api.save).toHaveBeenCalledTimes(2);
    expect(modal.showConflict).toHaveBeenCalledWith('sugg', 'old');
    expect(statusEl.textContent).toMatch(/Saved/);
    expect(res).toEqual(expect.objectContaining({ ok: true }));
  });

  test('handles 409 then rename path', async () => {
    const api = {
      validate: jest.fn(() => Promise.resolve()),
      save: jest.fn()
        .mockRejectedValueOnce({ status: 409, body: { suggested_name: 'sugg' } })
        .mockResolvedValueOnce({ ok: true, data: { status: 'ok' } }),
    };
    const modal = {
      showConflict: jest.fn(() => Promise.resolve({ action: 'rename' })),
      promptRename: jest.fn(() => Promise.resolve({ action: 'rename', name: 'new-name' })),
    };
    const store = { setValidationErrors: jest.fn() };
    const statusEl = makeEl();
    statusEl.id = 'test-status-rn';
    const res = await saveFlow({ api, modal, store, payload: { draft_id: null, name: 'old' }, statusEl });
    expect(api.validate).toHaveBeenCalled();
    expect(modal.showConflict).toHaveBeenCalledWith('sugg', 'old');
    expect(modal.promptRename).toHaveBeenCalled();
    expect(api.save).toHaveBeenCalledTimes(2);
    expect(statusEl.textContent).toMatch(/Saved/);
    expect(res).toEqual(expect.objectContaining({ ok: true }));
  });
});
