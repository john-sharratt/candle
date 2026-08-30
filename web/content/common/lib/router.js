/* Hash router + page registry (§23).
 *
 * Pages self-register; nav and breadcrumbs are DERIVED from the registry rather
 * than maintained beside it. `load` is a dynamic import, so page count does not
 * become bundle size. Hash routing needs no server rewrite — the daemon's
 * `embedded_asset` fallback keeps working untouched. */

const PAGES = [];
let outlet = null;
let current = null;

/* The roles a page may be declared at, weakest first — the same three the
 * daemon uses, and the same ordering, so `rank(have) >= rank(need)` is the
 * whole check on this side too. */
const RANK = { unauthenticated: 0, user: 1, admin: 2 };

/* Who is looking, as the SERVER decided. Set once at boot from `/v1/me`; a
 * page never computes it. `viewerRole` is a function rather than a value
 * because boot resolves the account after the registry is built. */
let viewerRole = () => 'unauthenticated';
export function setViewerRole(fn) { viewerRole = fn; }
export function can(need) { return RANK[viewerRole()] >= RANK[need]; }

/**
 * Register a page.
 *
 * `role` is **required**, for the reason the daemon's `guard::Api` requires one:
 * a page whose access nobody stated is a page that is open by accident, and an
 * omission is the hardest thing to notice in a diff. There is no default —
 * defaulting either way is wrong. Open is a silent hole; locked is a silently
 * broken console. So it throws, at import time, naming the page.
 *
 * The value must match what the API will actually allow. A page shown to a user
 * whose every request 403s is worse than no page at all.
 *
 * `under` names the nav entry this page sits beneath, for the pages that are not
 * nav entries themselves — see [`navOwner`].
 */
export function definePage(def) {
  if (!def || !(def.role in RANK)) {
    throw new Error(
      `definePage(${(def && def.path) || '?'}): a page must declare role: `
      + Object.keys(RANK).map((r) => `'${r}'`).join(' | '),
    );
  }
  PAGES.push(def);
}

export function pages() { return PAGES.slice(); }

/** Every page and the role it needs — the table, for the console's own audit. */
export function roleTable() {
  return PAGES.map((p) => ({ path: p.path, role: p.role }));
}

/**
 * Nav sections, derived. A page with no `nav` never appears in navigation, and
 * neither does one the viewer could not open — a link that leads to a refusal
 * is a worse experience than no link.
 */
export function navFor(section) {
  return PAGES.filter((p) => p.nav && p.nav.section === section && can(p.role))
    .sort((a, b) => (a.nav.order || 0) - (b.nav.order || 0));
}

/**
 * Which nav entry a page belongs to — the one to mark as current.
 *
 * A nav entry is a page, but most pages are not nav entries: `/world/:wid` is
 * one world, `/npc/:id` is one character. Matching the current path against the
 * nav list alone therefore finds nothing the moment anybody opens something,
 * and the bar goes blank exactly when it is most useful — you are two levels
 * into a world with nothing on screen saying so.
 *
 * So a page may declare `under: '/worlds'`, naming the entry it sits beneath.
 * Declared rather than derived: `/world/:wid` under `/worlds` and `/npc/:id`
 * under `/` are not a prefix rule, and a rule that got them right by accident
 * would get the next one wrong.
 */
export function navOwner(page, section) {
  if (!page) return null;
  const items = navFor(section);
  return items.find((p) => p.path === page.path)
    || items.find((p) => p.path === page.under)
    || null;
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
      // `p` is `location.hash`, so it is whatever a link says it is. Written as
      // text rather than interpolated into the markup: a crafted `#/<img src=x
      // onerror=...>` would otherwise execute in this origin, where the session
      // cookie lives, and the 404 page is the one page guaranteed to be
      // reachable with an arbitrary path.
      const page = Object.assign(document.createElement('div'), {
        className: 'page',
        innerHTML: `<div class="empty"><div class="big">404</div>
          <div>No page at <code class="mono"></code></div></div>`,
      });
      page.querySelector('code').textContent = p;
      outlet.replaceChildren(page);
      if (onRoute) onRoute(null, {});
      return;
    }
    // The role gate, before anything loads. Hiding a nav link is presentation;
    // this is what answers a typed URL, a bookmark, or a link somebody pasted.
    // Neither is a security control — the daemon refuses the requests whatever
    // happens here — but a page that renders and then fails every fetch tells
    // its reader the product is broken, when the truth is that it is not
    // theirs to open.
    if (!can(hit.page.role)) {
      const page = Object.assign(document.createElement('div'), {
        className: 'page',
        innerHTML: `<div class="empty"><div class="big">⊘</div>
          <div style="font-weight:700;color:var(--ink-dim);margin-bottom:6px"></div>
          <div class="tiny"></div></div>`,
      });
      const [title, detail] = page.querySelectorAll('div.empty > div:not(.big)');
      title.textContent = hit.page.role === 'admin' ? 'Admins only' : 'Sign in to see this';
      detail.textContent = hit.page.role === 'admin'
        ? 'This page needs the admin role. You are signed in as ' + viewerRole() + '.'
        : 'This page needs you to be signed in.';
      outlet.replaceChildren(page);
      if (onRoute) onRoute(hit.page, hit.params);
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
        + '<div class="tiny dim" style="margin-top:6px"></div>'
        + '<pre></pre>';
      // Both dynamic values as text, for the reason the 404 branch gives: `p`
      // is the hash, and a page that throws is a state an attacker can steer.
      box.querySelector('.tiny').textContent = p;
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
