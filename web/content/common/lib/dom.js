/* DOM helpers. `h` builds elements; `mount` swaps children. Deliberately tiny —
 * pages that stream (act feed, token feed) append rather than re-render. */

export function h(tag, props = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style') el.setAttribute('style', v);
    else if (k === 'html') el.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  add(el, kids);
  return el;
}

function add(el, kids) {
  for (const k of kids.flat(9)) {
    if (k == null || k === false) continue;
    el.appendChild(k instanceof Node ? k : document.createTextNode(String(k)));
  }
}

export const frag = (...kids) => { const f = document.createDocumentFragment(); add(f, kids); return f; };
export const mount = (host, ...kids) => { host.replaceChildren(); add(host, kids); return host; };
export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* Formatting used across pages. */
export const fmtNum = (n) => (n == null ? '—' : Number(n).toLocaleString());
export const fmtK = (n) => (n == null ? '—' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n));
export const fmtPct = (n) => (n == null ? '—' : Math.round(n * 100) + '%');

export function ago(ms) {
  if (!ms) return '—';
  const s = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (s < 60) return s + 's ago';
  if (s < 3600) return Math.floor(s / 60) + 'm ago';
  if (s < 86400) return Math.floor(s / 3600) + 'h ago';
  return Math.floor(s / 86400) + 'd ago';
}

/* Narrative time reads as day + clock, never as a wall date (§4). */
export function worldTime(ms) {
  if (ms == null) return '—';
  const day = Math.floor(ms / 86400000);
  const rem = ms % 86400000;
  const hh = String(Math.floor(rem / 3600000)).padStart(2, '0');
  const mm = String(Math.floor((rem % 3600000) / 60000)).padStart(2, '0');
  return `day ${day}, ${hh}:${mm}`;
}

export function truncId(id) {
  const s = String(id ?? '');
  return s.length > 12 ? s.slice(0, 4) + '…' + s.slice(-4) : s;
}
