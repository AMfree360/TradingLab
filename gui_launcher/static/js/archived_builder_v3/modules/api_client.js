export default class ApiClient {
  constructor(opts = {}) {
    this.csrfToken = opts.csrfToken || null;
    this.base = '/api/builder_v3';
  }

  async _post(path, body) {
    const headers = { 'Content-Type': 'application/json' };
    if (this.csrfToken) headers['X-CSRF-Token'] = this.csrfToken;
    const res = await fetch(this.base + path, { method: 'POST', headers, body: JSON.stringify(body) });
    const json = await res.json().catch(() => null);
    if (!res.ok) {
      const err = { status: res.status, body: json };
      throw err;
    }
    return json;
  }

  async metadata() {
    const res = await fetch(this.base + '/metadata');
    if (!res.ok) throw new Error('metadata fetch failed');
    return res.json();
  }

  async preview(payload) { return this._post('/preview', payload); }
  async contextVisual(payload) { return this._post('/context_visual', payload); }
  async setupVisual(payload) { return this._post('/setup_visual', payload); }
  async validate(payload) { return this._post('/validate', payload); }
  // Save uses a slightly different contract: it returns the parsed body for 2xx,
  // and throws an object `{status, body}` for non-2xx which callers can inspect
  // (notably 409 conflict responses).
  async save(action, draftId, payload, opts = {}) {
    const headers = { 'Content-Type': 'application/json' };
    if (this.csrfToken) headers['X-CSRF-Token'] = this.csrfToken;
    const body = { action, draft_id: draftId || null, payload };
    if (opts.force) body.force = true;
    if (opts.rename_to) body.rename_to = opts.rename_to;
    const res = await fetch(this.base + '/save', { method: 'POST', headers, body: JSON.stringify(body) });
    const json = await res.json().catch(() => null);
    if (!res.ok) throw { status: res.status, body: json };
    return json;
  }
}
