/* npcd shell — boot, theme, nav, routing. */

import { API, BACKEND } from './lib/api.js';
import { definePage, navFor, navOwner, start, go, path, link, setViewerRole, can } from './lib/router.js';
import { h, mount } from './lib/dom.js';
import { toast } from './lib/ui.js';
import { checkBuild, takeReloadState } from './lib/build.js';
import { state as vp, onBreakpoint } from './lib/viewport.js';
import { estateSwitcher } from './lib/estate.js';

// ── page registry ───────────────────────────────────────────────────────────
// Adding a page touches exactly one entry here plus one file. Nav is derived.

/* Each page is a path, how to load it, and — for the ones that appear in
 * navigation — where it sits there. No `title`: the tab keeps the name
 * `index.html` gave it, so a per-page title had no reader and was quietly
 * becoming a second, unmaintained set of labels beside `nav.label`. */
/* The page table, and the role each page needs.
 *
 * `definePage` throws without a `role`, for the same reason the daemon's
 * `guard::Api` will not compile without one: a page whose access nobody stated
 * is a page that is open by accident.
 *
 * **Each role here mirrors the API route the page actually calls.** That is the
 * property worth protecting — a page shown to somebody whose every request 403s
 * is worse than no page, and a page hidden from somebody the API would have
 * served is a feature quietly withdrawn. The daemon's table is in `api::api`
 * and `ops::api`; when one moves, this moves.
 *
 *   /            → GET  /v1/npc                     user
 *   /npc/*       → GET  /v1/npc/:nid                user
 *   /worlds      → GET  /v1/world                   unauthenticated (writes are admin)
 *   /personalities → GET /v1/personality            unauthenticated (writes are admin)
 *   /performance → GET  /v1/telemetry, /v1/memory   user
 *   /substrate   → GET  /v1/substrate/storage       admin
 *   /logs        → WS   /ws/logs                    admin
 */
definePage({ path: '/', role: 'user', nav: { section: 'main', order: 10, label: 'My NPCs' },
  load: () => import('./pages/roster.js') });
definePage({ path: '/npc/new', role: 'user', under: '/',
  load: () => import('./pages/create.js') });
definePage({ path: '/npc/:id', role: 'user', keepsRail: true, under: '/',
  load: () => import('./pages/npc.js') });
definePage({ path: '/npc/:id/:tab', role: 'user', keepsRail: true, under: '/',
  load: () => import('./pages/npc.js') });
definePage({ path: '/interaction/:ix', role: 'user', under: '/',
  load: () => import('./pages/console.js') });
// Worlds and personalities are READABLE by anyone — the daemon serves their
// GETs unauthenticated — and the pages render read-only below `admin`.
definePage({ path: '/worlds', role: 'unauthenticated', nav: { section: 'main', order: 20, label: 'Worlds' },
  load: () => import('./pages/worlds.js') });
definePage({ path: '/world/:wid', role: 'unauthenticated', under: '/worlds',
  load: () => import('./pages/worlds.js') });
definePage({ path: '/personalities', role: 'unauthenticated', nav: { section: 'main', order: 30, label: 'Personalities' },
  load: () => import('./pages/personalities.js') });
/* The authored corpus as files. `user` rather than `unauthenticated` unlike the
 * two pages above, and the daemon agrees: those answer somebody who already
 * knows an id, while this one ENUMERATES — it hands out the mind a directory at
 * a time, which is the browsing the `hidden` flag exists to prevent. Writing
 * still needs `admin`; the page renders read-only below it. */
definePage({ path: '/mind', role: 'user', nav: { section: 'main', order: 35, label: 'Mind' },
  load: () => import('./pages/mind.js') });
definePage({ path: '/tools', role: 'user', nav: { section: 'main', order: 40, label: 'Tools' },
  load: () => import('./pages/tools.js') });
// Names the redo log's absolute path, so it matches `/v1/substrate/storage`.
definePage({ path: '/substrate', role: 'admin', nav: { section: 'main', order: 50, label: 'Substrate' },
  load: () => import('./pages/substrate.js') });
definePage({ path: '/performance', role: 'user', nav: { section: 'main', order: 60, label: 'Performance' },
  load: () => import('./pages/system.js') });
definePage({ path: '/probe', role: 'user', nav: { section: 'main', order: 55, label: 'Probe' },
  load: () => import('./pages/probe.js') });
// Carries every save's full path and the account ids — matches `/ws/logs`.
definePage({ path: '/logs', role: 'admin', nav: { section: 'main', order: 70, label: 'Logs' },
  load: () => import('./pages/logs.js') });
// The same page as `/performance`, under an older address, so it belongs to the
// same nav entry.
definePage({ path: '/system', role: 'user', under: '/performance',
  load: () => import('./pages/system.js') });
definePage({ path: '/me', role: 'user', load: () => import('./pages/profile.js') });
// The signed-out landing page. Necessarily reachable by nobody in particular.
definePage({ path: '/welcome', role: 'unauthenticated', load: () => import('./pages/landing.js') });

// ── theme ───────────────────────────────────────────────────────────────────

const THEMES = [
  ['dark', 'Dark', '#1c1917'],
  ['light', 'Light', '#e1d9c9'],
  ['vivid', 'Vivid', '#ffa23c'],
];
const readTheme = () => { try { return localStorage.getItem('npcd.theme') || 'dark'; } catch (_) { return 'dark'; } };
function setTheme(t) {
  try { localStorage.setItem('npcd.theme', t); } catch (_) {}
  document.documentElement.setAttribute('data-theme', t);
  renderChrome();
}

let menuOpen = false;

/* ── one popover at a time ──────────────────────────────────────────────────
 *
 * The topbar holds three: the estate switcher on the left, the collapsed nav
 * beside it, and the appearance menu on the right. Only one of them may be
 * open, and opening one closes the rest.
 *
 * Each used to dismiss only *itself*, with a one-shot document click listener —
 * and each trigger calls `stopPropagation`, so that the click which opens a
 * popover does not immediately reach the document and close it again. Those two
 * together meant a trigger's click never reached the document at all, so the
 * listener that would have closed the *other* popover never ran: opening the
 * appearance menu while the nav menu was open left both on screen, overlapping.
 *
 * Closing them here, at the moment of opening, needs no event to travel
 * anywhere and so cannot be stopped by anything. */
function closeOtherPopovers(keep) {
  if (keep !== 'nav' && navMenuOpen) {
    navMenuOpen = false;
    renderNav();
  }
  if (keep !== 'theme' && menuOpen) {
    menuOpen = false;
    renderChrome();
  }
  if (keep !== 'estate') {
    // A `<details>`: closing it is setting the attribute, and it owns its own
    // open state rather than a flag here.
    for (const d of document.querySelectorAll('details.estate[open]')) d.open = false;
  }
}

function themeButton() {
  const t = readTheme();
  const meta = THEMES.find((x) => x[0] === t) || THEMES[0];
  const btn = h('button', {
    class: 'btn sm', title: 'Appearance',
    onClick: (e) => {
      e.stopPropagation();
      closeOtherPopovers('theme');
      menuOpen = !menuOpen;
      renderChrome();
    },
  }, h('span', {
    class: 'sw',
    style: 'width:12px;height:12px;border-radius:4px;background:' + meta[2]
      + ';box-shadow:inset 0 0 0 1px rgba(128,128,128,.4)',
  }), meta[1]);
  return btn;
}

function themeMenu() {
  if (!menuOpen) return null;
  const cur = readTheme();
  return h('div', { class: 'menu' }, THEMES.map(([k, label, sw]) =>
    h('button', { class: k === cur ? 'on' : '', onClick: () => { menuOpen = false; setTheme(k); } },
      h('span', { class: 'sw', style: `background:${sw}` }), label)));
}

// ── chrome (right side of the topbar) ───────────────────────────────────────

let ME = null;

/* The signed-in person's face, or their initial when there is no picture.
 *
 * `referrerpolicy` because the src is the identity provider's CDN: without it
 * every avatar load tells Google which page of the console is being read.
 * `onError` because an avatar URL outlives the image behind it — a provider
 * rotates the path and the chrome would otherwise show a broken-image glyph
 * where a person's face was, which looks like a bug in the sign-in. */
export function faceOf(me, px) {
  // Every candidate can be present-but-blank: a provider may send a name that
  // is only whitespace, and `''.trim()[0]` is `undefined`, which would throw on
  // `.toUpperCase()` and take the whole top bar down with it.
  const initial = ([me.display, me.unique_name, '?']
    .map((s) => (s || '').trim()).find((s) => s.length)[0]).toUpperCase();
  const ring = `width:${px}px;height:${px}px;border-radius:50%;flex:none`;
  if (me.avatar_url) {
    const img = h('img', {
      src: me.avatar_url, alt: '', referrerpolicy: 'no-referrer',
      style: ring + ';object-fit:cover',
    });
    img.addEventListener('error', () => img.replaceWith(letter(initial, px, ring)));
    return img;
  }
  return letter(initial, px, ring);
}

function letter(initial, px, ring) {
  return h('span', {
    style: ring + ';background:var(--accent);color:var(--accent-ink);display:grid;place-items:center;'
      + `font-size:${Math.round(px * 0.44)}px;font-weight:800`,
  }, initial);
}

/* Hover text: the display name is already visible, so the tooltip carries what
 * is not — the account it belongs to, and the handle characters address. */
function whoTitle() {
  return [ME.email, ME.unique_name && '@' + ME.unique_name].filter(Boolean).join(' · ');
}

function renderChrome() {
  const host = document.getElementById('chrome');
  if (!host) return;
  mount(host,
    themeButton(),
    ME
      /* Your own name and face, not your handle. `unique_name` is the address
       * an NPC uses for you — a lowercased, punctuation-stripped thing derived
       * from an email — and showing it here reads as somebody else's account.
       * The provider's `display` and `avatar_url` are what a person recognises
       * as themselves, so they are what the chrome shows; the handle belongs on
       * the profile page, where it is being edited. */
      ? h('button', {
        class: 'btn sm ghost', title: whoTitle(),
        onClick: () => go('/me'),
      }, faceOf(ME, 20), (ME.display || '').trim() || ME.unique_name || 'Signed in')
      /* Straight to the provider, not to `#/welcome` — a link to the page you
       * are already on is a control that visibly does nothing, and the welcome
       * page is where this button is most likely to be pressed. When sign-in is
       * unconfigured there is nowhere to send anyone, so it says so instead. */
      : AUTH_UNAVAILABLE
        ? h('span', { class: 'tiny dim', title: 'This deployment has no sign-in configured' }, 'sign-in off')
        : h('button', { class: 'btn sm primary', onClick: () => window.__npcdSignIn() }, 'Sign in'));
  const old = document.querySelector('.menu');
  if (old) old.remove();
  const m = themeMenu();
  if (m) {
    document.querySelector('.topbar').appendChild(m);
    setTimeout(() => document.addEventListener('click', function close() {
      menuOpen = false; renderChrome(); document.removeEventListener('click', close);
    }, { once: true }), 0);
  }
}

let navMenuOpen = false;
let activeNavPage = null;

function renderNav(activePage) {
  if (activePage !== undefined) activeNavPage = activePage;
  const host = document.getElementById('nav');
  if (!host) return;
  if (!ME) { navMenuOpen = false; return mount(host); }

  const items = navFor('main');
  // Not an exact path match: a detail page is not a nav entry, so opening one
  // used to blank the bar — you were two levels into a world with nothing on
  // screen saying which section you were in. `navOwner` follows the page's
  // `under` to the entry it belongs to.
  const current = navOwner(activeNavPage, 'main');

  // Wide enough for the full tab strip.
  if (!vp.narrow) {
    navMenuOpen = false;
    return mount(host, items.map((p) =>
      link(p.path, {
        class: 'navlink',
        ...(current === p ? { 'aria-current': 'page' } : {}),
      }, p.nav.label)));
  }

  // Narrow: seven tabs will not fit, so collapse them behind one button that
  // names where you are. Same decision zend makes for its top-right actions.
  const btn = h('button', {
    class: 'navlink nav-collapsed' + (navMenuOpen ? ' on' : ''),
    onClick: (e) => {
      e.stopPropagation();
      closeOtherPopovers('nav');
      navMenuOpen = !navMenuOpen;
      renderNav();
    },
  }, current ? current.nav.label : 'Menu', h('span', { class: 'caret' }, '▾'));

  mount(host, btn);

  const stale = document.querySelector('.nav-menu');
  if (stale) stale.remove();
  if (!navMenuOpen) return;

  const menu = h('div', { class: 'nav-menu' }, items.map((p) =>
    link(p.path, {
      class: 'nav-menu-item' + (current === p ? ' on' : ''),
      onClick: () => { navMenuOpen = false; renderNav(); },
    }, p.nav.label)));
  document.querySelector('.topbar').appendChild(menu);
  setTimeout(() => document.addEventListener('click', function close() {
    navMenuOpen = false; renderNav(); document.removeEventListener('click', close);
  }, { once: true }), 0);
}

/* ── rail drawer (narrow) ──────────────────────────────────────────────────
 * The rail is docked on wide screens and becomes a fixed overlay drawer on
 * narrow ones. The toggle only appears when the current page actually put
 * something in the rail — a button that opens an empty drawer is worse than no
 * button. A MutationObserver watches the rail so pages need no extra call. */

let drawerOpen = false;

function setDrawer(open) {
  drawerOpen = open && vp.narrow;
  const rail = document.getElementById('rail');
  const scrim = document.getElementById('scrim');
  if (rail) rail.classList.toggle('open', drawerOpen);
  if (scrim) scrim.hidden = !drawerOpen;
  document.documentElement.classList.toggle('drawer-open', drawerOpen);
}

function syncRailButton() {
  const rail = document.getElementById('rail');
  const btn = document.getElementById('rail-btn');
  if (!rail || !btn) return;
  const hasRail = rail.childElementCount > 0;
  btn.hidden = !(vp.narrow && hasRail);
  if (!hasRail || !vp.narrow) setDrawer(false);
}

// ── boot ────────────────────────────────────────────────────────────────────

/* Sign-in is real. `/v1/me` answers only for a caller the gateway named, so
 * being signed in is not something this page can decide — it is something it
 * discovers.
 *
 * The gateway owns the flow. `/auth/login` is served on every hostname ahead of
 * site routing, and the cookie it issues is on `.tokera.com`, which is what
 * makes one sign-in carry to code. and bot. without either daemon taking part.
 * So signing in is a navigation, not a fetch.
 *
 * # `next` is absolute, and has to be
 *
 * The provider's registered redirect URI is `https://tokera.com/auth/callback` —
 * one host for the whole estate, because that is what a provider registration
 * is. So the browser always comes back to tokera.com, and a relative `next` like
 * `/#/welcome` resolves *there*: sign in from this console and you land on the
 * home page of a different site, having asked to come back here.
 *
 * Sending the full URL survives the hop. `safe_next` accepts it because the
 * host is under the cookie domain, and refuses anything that is not — so this
 * cannot be turned into an open redirect by handing it somebody else's URL. */
window.__npcdSignIn = () => {
  location.href = '/auth/login?next=' + encodeURIComponent(location.href);
};
window.__npcdSignOut = () => {
  /* Logout needs no round trip through the provider, so a relative target would
   * work — but the rule is worth keeping uniform: whoever is reading this next
   * should not have to work out which of the two hops loses the host. */
  location.href =
    '/auth/logout?next=' + encodeURIComponent(location.origin + '/#/welcome');
};

/* Distinct from being signed out, and the difference decides whether to offer a
 * sign-in control at all: a button that cannot work is worse than none.
 *
 * The gateway is the only authority on it, because it owns the whole flow. With
 * `auth:` off it does not serve `/auth/login` at all — the navigation then lands
 * on site routing, gets `index.html` back, and reads to the visitor as a button
 * that does nothing. `/auth/me` says `configured: false` in that state, which is
 * the one reliable way to know before offering the control. */
export let AUTH_UNAVAILABLE = false;

async function gatewayHasSignIn() {
  try {
    const r = await fetch('/auth/me', { credentials: 'same-origin' });
    if (!r.ok) return false;
    return (await r.json()).configured !== false;
  } catch (_) {
    // Unreachable gateway. Not a claim that sign-in is broken forever, but it
    // is a claim that this page cannot start a sign-in right now.
    return false;
  }
}

async function boot() {
  document.documentElement.setAttribute('data-theme', readTheme());
  const detail = document.getElementById('boot-detail');

  /* Wait for the daemon — but only for the thing worth waiting for.
   *
   * A *refusal* is an answer. A 401 or a 403 means the daemon is up and has
   * decided something about this caller, and no amount of retrying changes it;
   * looping on one spent sixteen seconds saying "waiting for daemon…" before
   * landing on the welcome page, which reads as an outage and is a permission.
   *
   * Only a network failure or a not-yet-ready state is worth another go. */
  let status = null;
  for (let i = 0; i < 40; i++) {
    try {
      status = await API.getStatus();
      if (status.state === 'ready') break;
      if (detail) detail.textContent = status.detail || (status.loading && status.loading.current) || 'loading…';
    } catch (e) {
      if (e && (e.status === 401 || e.status === 403)) {
        // Up, and not answering this to us. Carry on to sign-in rather than
        // pretending to wait for something that has already replied.
        status = null;
        break;
      }
      if (detail) detail.textContent = 'waiting for daemon…';
    }
    await new Promise((r) => setTimeout(r, 400));
  }

  // A 401 here means signed out, which is the only thing this daemon can say
  // about identity — it does not run sign-in and has no configuration of its
  // own that could be missing.
  try {
    ME = await API.getMe();
  } catch (e) {
    ME = null;
  }
  // Hand the router the server's answer, once. Everything role-shaped — which
  // nav links appear, which pages open, which controls are writable — reads
  // through this one function, so there is a single place that can be wrong
  // and no page that can disagree with the router about who is looking.
  setViewerRole(() => (ME && ME.role) || 'unauthenticated');
  // Whether anyone *could* sign in is the gateway's answer, and worth asking
  // only when nobody is: a live session is itself proof that it is configured.
  if (!ME && !(await gatewayHasSignIn())) AUTH_UNAVAILABLE = true;

  const bootEl = document.getElementById('boot');
  if (bootEl) bootEl.remove();
  const shell = document.getElementById('shell');
  shell.hidden = false;

  const st = (id, text, title) => {
    const el = document.getElementById(id);
    if (el) { el.textContent = text; if (title) el.title = title; }
  };
  st('st-state', (status && status.state) === 'ready' ? '● ready' : '○ ' + ((status && status.state) || 'offline'));
  st('st-mode', (status && status.mode) || '');
  st('st-backend', 'backend: ' + BACKEND, 'add ?mock=1 to run without a daemon');
  st('st-build', (status && status.build) || '');

  /* The brand corner is the switcher. Its "you are here" row goes to this
   * console's own front page rather than to `https://bot.tokera.com/`, which
   * would be a full page load to arrive where you already are. */
  const estate = document.getElementById('estate');
  if (estate) {
    const switcher = estateSwitcher('npcd', { homeHref: '#/welcome' });
    mount(estate, switcher);
    /* The switcher is a `<details>`, so it opens itself and there is no click
     * handler to hang this on. `toggle` does not bubble, hence the listener on
     * the element rather than on the document. */
    switcher.addEventListener('toggle', () => {
      if (switcher.open) closeOtherPopovers('estate');
    });
  }

  renderChrome();
  renderNav(null);   // tabs appear immediately, even if the first page fails to render

  // ── responsive wiring ────────────────────────────────────────────────────
  const railEl = document.getElementById('rail');
  const scrimEl = document.getElementById('scrim');
  const railBtn = document.getElementById('rail-btn');

  if (railBtn) railBtn.addEventListener('click', (e) => { e.stopPropagation(); setDrawer(!drawerOpen); });
  if (scrimEl) scrimEl.addEventListener('click', () => setDrawer(false));
  // Choosing something inside the drawer should dismiss it — otherwise the
  // destination renders behind an overlay the user has to close by hand.
  if (railEl) railEl.addEventListener('click', (e) => {
    if (drawerOpen && e.target.closest('button, a')) setDrawer(false);
  });
  // Pages fill the rail whenever they like; watch it rather than making every
  // page remember to announce itself.
  if (railEl) new MutationObserver(syncRailButton).observe(railEl, { childList: true });

  onBreakpoint(() => { renderNav(); renderChrome(); syncRailButton(); });
  syncRailButton();

  // Escape closes whatever is over the page: the drawer first, since it is the
  // one that covers everything, then any open popover. It used to close the nav
  // menu alone, so the appearance menu and the switcher had to be dismissed by
  // clicking away from them.
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (drawerOpen) return setDrawer(false);
    closeOtherPopovers(null);
  });

  // Not signed in → the landing page owns the viewport.
  if (!ME && path() !== '/welcome') {
    location.replace('#/welcome');
  }

  start(document.getElementById('outlet'), (page, params) => {
    renderNav(page);
    /* The tab keeps the name it was served with. It used to be rewritten on
     * every route — `Roster · npcd`, `Worlds · npcd` — which is restless when
     * the page already says where you are, twice, in the nav and in its own
     * heading. `index.html` sets it once and nothing here touches it. */
    // Pages own the rail; clear it on every route so a stale one never lingers.
    const rail = document.getElementById('rail');
    if (rail && !(page && page.keepsRail)) rail.replaceChildren();
    document.body.classList.toggle('logged-out', !ME);
  });

  // The web assets are embedded in the binary, so a hot rebuild would leave this
  // page running against a newer API — they disagree on the surface and break
  // silently. Watch the build id and reload, carrying the route across.
  const restored = takeReloadState();
  if (restored && restored.hash) location.replace(restored.hash);
  setInterval(async () => {
    try { checkBuild(await API.getStatus(), { hash: location.hash }); } catch (_) {}
  }, 5000);

  // Keyboard: g-then-key jumps, "/" focuses search unless a composer has it.
  let g = false;
  document.addEventListener('keydown', (e) => {
    const inField = /^(INPUT|TEXTAREA)$/.test(document.activeElement?.tagName || '');
    if (inField) return;
    if (e.key === 'g') { g = true; setTimeout(() => { g = false; }, 700); return; }
    if (g) {
      g = false;
      const to = { r: '/', w: '/worlds', p: '/personalities', t: '/tools', s: '/system' }[e.key];
      if (to) { e.preventDefault(); go(to); }
      return;
    }
    if (e.key === '/') {
      const s = document.querySelector('[data-search]');
      if (s) { e.preventDefault(); s.focus(); }
    }
  });
}

window.addEventListener('error', (e) => toast(String(e.message || e), 'err'));
window.__npcd = { API, BACKEND, get me() { return ME; } };

/* The caller's role, as the SERVER decided it.
 *
 * Read from `/v1/me`, never inferred and never stored — a page that decided for
 * itself would be one more place the answer can be wrong, and the wrong
 * direction is showing somebody a Save that will 403.
 *
 * This is presentation only. It hides controls the server would refuse; it is
 * not the check. The check is `require(…, Role::Admin)` in the daemon, on the
 * far side of a network hop where a browser cannot reach it.
 *
 * `can` is re-exported so a page asks the same question the router asked when
 * it decided to show the page at all — one predicate, not two that can differ. */
export const role = () => (ME && ME.role) || 'unauthenticated';
export const isAdmin = () => can('admin');
export { can };

boot().catch((err) => {
  const b = document.getElementById('boot');
  if (b) b.remove();
  const s = document.getElementById('shell');
  if (s) s.hidden = false;
  const o = document.getElementById('outlet');
  if (o) {
    const box = document.createElement('div');
    box.className = 'page-error';
    box.innerHTML = '<h3 style="color:var(--crit);margin:0">Startup failed</h3><pre></pre>';
    box.querySelector('pre').textContent = (err && (err.stack || err.message)) || String(err);
    o.replaceChildren(box);
  }
  console.error('[npcd] boot failed:', err);
});
