// Minimal static file server for the Zend web assets (one dir up).
// Used by playwright.config.js as the Phase-1 webServer so the mock-backed UI
// can be driven headless with no daemon. No dependencies (Node core only).
const http = require('http');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', 'web');
// What the host serves every UI at `/common` (`host::ui::COMMON`): the shared
// styles and the estate switcher the page imports — and, at `/brand`, its
// `brand/` folder, where the switcher finds the estate's marks.
const COMMON = path.resolve(__dirname, '..', '..', 'web', 'content', 'common');
const PORT = process.env.ZEND_WEB_PORT ? Number(process.env.ZEND_WEB_PORT) : 4321;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.json': 'application/json',
};

http.createServer((req, res) => {
  let urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
  if (urlPath === '/') urlPath = '/index.html';
  const [root, rel] = urlPath.startsWith('/common/')
    ? [COMMON, urlPath.slice('/common'.length)]
    : urlPath.startsWith('/brand/')
      ? [COMMON, urlPath]
      : [ROOT, urlPath];
  const filePath = path.join(root, rel);
  // contain to its root
  if (!filePath.startsWith(root)) { res.writeHead(403); res.end('forbidden'); return; }
  fs.readFile(filePath, (err, buf) => {
    if (err) {
      console.error('not found: ' + urlPath);
      res.writeHead(404); res.end('not found'); return;
    }
    res.writeHead(200, { 'content-type': MIME[path.extname(filePath)] || 'application/octet-stream' });
    res.end(buf);
  });
}).listen(PORT, () => console.log('zend web test server on http://localhost:' + PORT));
