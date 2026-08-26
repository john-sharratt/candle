/* Hash router + page registry (§23).
 *
 * Pages self-register; nav and breadcrumbs are DERIVED from the registry rather
 * than maintained beside it. `load` is a dynamic import, so page count does not
 * become bundle size. Hash routing needs no server rewrite — the daemon's
 * `embedded_asset` fallback keeps working untouched. */

const PAGES = [];
let outlet = null;
let current = null;

export function definePage(def) { PAGES.push(def); }
export function pages() { return PAGES.slice(); }

/** Nav sections, derived. A page with no `nav` never appears in navigation. */
export function navFor(section) {
  return PAGES.filter((p) => p.nav && p.nav.section === section)
    .sort((a, b) => (a.nav.order || 0) - (b.nav.order || 0));
}

function compile(pattern) {
  const names = [];
  const rx = new RegExp('^' + pattern.replace(/:[A-Za-z_]\w*/g, (m) => {
    names.push(m.slice(1));
    return '([^/]+)';
  }).replace(/\//g, '\\/') + '$');
  return { rx, names };
}

function match(path) {
  for (const p of PAGES) {
    const { rx, names } = p._c || (p._c = compile(p.path));
    const m = rx.exec(path);
    if (m) {
      const params = {};
      names.forEach((n, i) => { params[n] = decodeURIComponent(m[i + 1]); });
      return { page: p, params };
    }
  }
  return null;
}

export const path = () => (location.hash || '#/').slice(1).split('?')[0] || '/';
export const query = () => {
  const q = (location.hash || '').split('?')[1] || '';
  return Object.fromEntries(new URLSearchParams(q));
};
export const go = (to) => { location.hash = to.startsWith('#') ? to : '#' + to; };
export const replace = (to) => location.replace('#' + to.replace(/^#/, ''));

export function link(to, props, ...kids) {
  const a = document.createElement('a');
  a.href = '#' + to.replace(/^#/, '');
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') a.className = v;
    else if (k.startsWith('on') && typeof v === 'function') a.addEventListener(k.slice(2).toLowerCase(), v);
    else a.setAttribute(k, v === true ? '' : v);   // hyphenated attrs (aria-current) must not go through Object.assign
  }
  kids.flat(9).forEach((k) => a.appendChild(k instanceof Node ? k : document.createTextNode(String(k))));
  return a;
}

export function start(host, onRoute) {
  outlet = host;
  const render = async () => {
    const p = path();
    const hit = match(p);
    if (!hit) {
      outlet.replaceChildren(Object.assign(document.createElement('div'), {
        className: 'page',
        innerHTML: `<div class="empty"><div class="big">404</div>
          <div>No page at <code class="mono">${p}</code></div></div>`,
      }));
      if (onRoute) onRoute(null, {});
      return;
    }
    // Guards run before load, so authorization cannot be forgotten (§23).
    if (hit.page.guard) {
      const verdict = await hit.page.guard(hit.params);
      if (verdict && verdict.redirect) return replace(verdict.redirect);
    }
    if (current && current.teardown) { try { current.teardown(); } catch (_) {} }
    // A page that throws must never leave a blank screen with no explanation —
    // the shell chrome still renders and the failure is shown in place.
    try {
      const mod = await hit.page.load();
      const view = await mod.render(hit.params, query());
      current = view && view.teardown ? view : null;
      outlet.replaceChildren(view && view.el ? view.el : view);
    } catch (err) {
      current = null;
      const box = document.createElement('div');
      box.className = 'page-error';
      const msg = (err && (err.stack || err.message)) || String(err);
      box.innerHTML = '<h3 style="color:var(--crit);margin:0">This page failed to render</h3>'
        + '<div class="tiny dim" style="margin-top:6px">' + p + '</div>'
        + '<pre></pre>';
      box.querySelector('pre').textContent = msg;
      outlet.replaceChildren(box);
      console.error('[npcd] page render failed:', p, err);
    }
    outlet.scrollTop = 0;
    if (onRoute) onRoute(hit.page, hit.params);
  };
  window.addEventListener('hashchange', render);
  render();
}
