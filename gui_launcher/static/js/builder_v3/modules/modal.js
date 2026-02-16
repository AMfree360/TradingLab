/* Modal: improved styling and accessibility */
export default class BuilderModal {
  constructor() {
    this._ensure();
    this._visible = false;
  }

  _ensure() {
    if (document.getElementById('v3-modal-root')) return;
    const root = document.createElement('div');
    root.id = 'v3-modal-root';
    root.className = 'v3-modal-root';
    root.innerHTML = `
      <div class="v3-modal-backdrop" tabindex="-1"></div>
      <div class="v3-modal-card" role="dialog" aria-modal="true" aria-labelledby="v3-modal-title">
        <div class="v3-modal-body">
          <h3 id="v3-modal-title" class="v3-modal-title"></h3>
          <div id="v3-modal-msg" class="v3-modal-msg"></div>
          <div id="v3-modal-suggest" class="v3-modal-suggest"></div>
          <div id="v3-modal-rename" class="v3-modal-rename"><label class="v3-modal-label">New name: <input id="v3-modal-rename-input" class="v3-modal-input" /></label></div>
        </div>
        <div class="v3-modal-actions">
          <button id="v3-modal-cancel" class="v3-modal-btn">Cancel</button>
          <button id="v3-modal-rename-btn" class="v3-modal-btn">Rename</button>
          <button id="v3-modal-overwrite" class="v3-modal-btn v3-modal-primary">Overwrite</button>
        </div>
      </div>
    `;
    document.body.appendChild(root);

    this.root = root;
    this.backdrop = root.querySelector('.v3-modal-backdrop');
    this.card = root.querySelector('.v3-modal-card');
    this.titleEl = root.querySelector('#v3-modal-title');
    this.msg = root.querySelector('#v3-modal-msg');
    this.suggest = root.querySelector('#v3-modal-suggest');
    this.renameWrapper = root.querySelector('#v3-modal-rename');
    this.renameInput = root.querySelector('#v3-modal-rename-input');
    this.btnCancel = root.querySelector('#v3-modal-cancel');
    this.btnRename = root.querySelector('#v3-modal-rename-btn');
    this.btnOverwrite = root.querySelector('#v3-modal-overwrite');

    // keyboard handlers
    this._onKeyDown = (e) => {
      if (!this._visible) return;
      if (e.key === 'Escape') {
        e.preventDefault();
        this._resolve({ action: 'cancel' });
      } else if (e.key === 'Enter') {
        // if rename visible, confirm rename
        if (this.renameWrapper.style.display !== 'none') {
          e.preventDefault();
          this._resolve({ action: 'rename', name: (this.renameInput.value || '').trim() });
        } else {
          e.preventDefault();
          this._resolve({ action: 'overwrite' });
        }
      } else if (e.key === 'Tab') {
        this._trapFocus(e);
      }
    };
  }

  _show() {
    this.root.classList.add('v3-modal-visible');
    this._visible = true;
    document.body.classList.add('v3-modal-open');
    window.addEventListener('keydown', this._onKeyDown);
    // focus first actionable element
    setTimeout(() => { this.btnOverwrite.focus(); }, 0);
  }

  _hide() {
    this.root.classList.remove('v3-modal-visible');
    this._visible = false;
    document.body.classList.remove('v3-modal-open');
    window.removeEventListener('keydown', this._onKeyDown);
  }

  _trapFocus(e) {
    const focusable = Array.from(this.root.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')).filter((el) => !el.disabled && el.offsetParent !== null);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }

  async showConflict(suggested = '', current = '') {
    this._ensure();
    this.titleEl.textContent = 'Name conflict';
    this.msg.textContent = 'A strategy with this name already exists.';
    this.suggest.innerHTML = suggested ? `<div class="v3-modal-suggest-text">Suggested: <strong>${suggested}</strong></div>` : '';
    this.renameWrapper.style.display = 'none';
    this.renameInput.value = suggested || current || '';
    this._show();

    return new Promise((resolve) => {
      this._resolve = (val) => { cleanup(); resolve(val); };
      const cleanup = () => {
        this.btnCancel.removeEventListener('click', onCancel);
        this.btnRename.removeEventListener('click', onRename);
        this.btnOverwrite.removeEventListener('click', onOverwrite);
        this._hide();
      };
      const onCancel = () => this._resolve({ action: 'cancel' });
      const onRename = () => { this.renameWrapper.style.display = 'block'; this.renameInput.focus(); };
      const onOverwrite = () => this._resolve({ action: 'overwrite' });
      this.btnCancel.addEventListener('click', onCancel);
      this.btnRename.addEventListener('click', onRename);
      this.btnOverwrite.addEventListener('click', onOverwrite);
    });
  }

  async promptRename(suggested = '', current = '') {
    this._ensure();
    this.titleEl.textContent = 'Rename strategy';
    this.msg.textContent = 'Choose a new strategy name';
    this.suggest.innerHTML = '';
    this.renameWrapper.style.display = 'block';
    this.renameInput.value = suggested || current || '';
    this._show();

    return new Promise((resolve) => {
      this._resolve = (val) => { cleanup(); resolve(val); };
      const cleanup = () => {
        this.btnCancel.removeEventListener('click', onCancel);
        this.btnRename.removeEventListener('click', onRename);
        this.btnOverwrite.removeEventListener('click', onOverwrite);
        this._hide();
      };
      const onCancel = () => this._resolve({ action: 'cancel' });
      const onRename = () => this._resolve({ action: 'rename', name: (this.renameInput.value || '').trim() });
      const onOverwrite = () => this._resolve({ action: 'overwrite' });
      this.btnCancel.addEventListener('click', onCancel);
      this.btnRename.addEventListener('click', onRename);
      this.btnOverwrite.addEventListener('click', onOverwrite);
    });
  }
}
