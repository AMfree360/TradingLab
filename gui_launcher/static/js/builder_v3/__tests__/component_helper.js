const fs = require('fs');
const path = require('path');

function loadComponent(filename, dirname) {
  const p = path.join(dirname, filename);
  let src = fs.readFileSync(p, 'utf8');
  // simple transform: export default -> module.exports =
  src = src.replace(/export\s+default\s+/, 'module.exports = ');
  const module = { exports: {} };
  const fn = new Function('module', 'exports', 'require', src);
  fn(module, module.exports, require);
  return module.exports;
}

module.exports = { loadComponent };
