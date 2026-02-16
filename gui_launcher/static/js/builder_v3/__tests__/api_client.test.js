const fs = require('fs');
const path = require('path');

// Load the ApiClient source and evaluate it under CommonJS for testing
function loadApiClient() {
  const p = path.join(__dirname, '..', 'modules', 'api_client.js');
  let src = fs.readFileSync(p, 'utf8');
  src = src.replace(/export default\s+class\s+ApiClient/, 'class ApiClient');
  src = src + '\nmodule.exports = ApiClient;';
  const module = { exports: {} };
  const fn = new Function('module', 'exports', 'require', src);
  fn(module, module.exports, require);
  return module.exports;
}

describe('ApiClient.save', () => {
  let ApiClient;
  beforeAll(() => { ApiClient = loadApiClient(); });

  beforeEach(() => {
    global.fetch = jest.fn();
  });
  afterEach(() => {
    delete global.fetch;
  });

  test('throws object with status and body on 409', async () => {
    const api = new ApiClient({ csrfToken: 'tok' });
    global.fetch.mockResolvedValueOnce({ ok: false, status: 409, json: async () => ({ suggested_name: 'name_v2' }) });
    await expect(api.save('save', null, { name: 'x' })).rejects.toEqual(expect.objectContaining({ status: 409 }));
    expect(global.fetch).toHaveBeenCalled();
  });

  test('sends force and rename_to in body when opts provided', async () => {
    const api = new ApiClient({ csrfToken: 'tok' });
    global.fetch.mockImplementationOnce((url, opts) => {
      const body = JSON.parse(opts.body);
      expect(body.force).toBe(true);
      expect(body.rename_to).toBe('newname');
      return Promise.resolve({ ok: true, status: 200, json: async () => ({ ok: true }) });
    });
    const res = await api.save('save', null, { name: 'x' }, { force: true, rename_to: 'newname' });
    expect(res).toEqual({ ok: true });
  });
});
