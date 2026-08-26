/* npcd shell — boot, theme, nav, routing. */

import { API, BACKEND } from './lib/api.js';
import { definePage, navFor, start, go, path, link } from './lib/router.js';
import { h, mount } from './lib/dom.js';
import { toast } from './lib/ui.js';
import { checkBuild, takeReloadState } from './lib/build.js';
import { state as vp, onBreakpoint } from './lib/viewport.js';

// ── page registry ───────────────────────────────────────────────────────────
// Adding a page touches exactly one entry here plus one file. Nav is derived.

definePage({ path: '/', title: () => 'My NPCs', nav: { section: 'main', order: 10, label: 'My NPCs' },
  load: () => import('./pages/roster.js') });
definePage({ path: '/npc/new', title: () => 'New character',
  load: () => import('./pages/create.js') });
definePage({ path: '/npc/:id', title: () => 'Character', keepsRail: true,
  load: () => import('./pages/npc.js') });
definePage({ path: '/npc/:id/:tab', title: () => 'Character', keepsRail: true,
  load: () => import('./pages/npc.js') });
definePage({ path: '/interaction/:ix', title: () => 'Interaction',
  load: () => import('./pages/console.js') });
definePage({ path: '/worlds', title: () => 'Worlds', nav: { section: 'main', order: 20, label: 'Worlds' },
  load: () => import('./pages/worlds.js') });
definePage({ path: '/world/:wid', title: () => 'World',
  load: () => import('./pages/worlds.js') });
definePage({ path: '/archetypes', title: () => 'Archetypes', nav: { section: 'main', order: 30, label: 'Archetypes' },
  load: () => import('./pages/archetypes.js') });
definePage({ path: '/tools', title: () => 'Tools', nav: { section: 'main', order: 40, label: 'Tools' },
  load: () => import('./pages/tools.js') });
definePage({ path: '/substrate', title: () => 'Substrate', nav: { section: 'main', order: 50, label: 'Substrate' },
  load: () => import('./pages/substrate.js') });
definePage({ path: '/performance', title: () => 'Performance', nav: { section: 'main', order: 60, label: 'Performance' },
  load: () => import('./pages/system.js') });
definePage({ path: '/probe', title: () => 'Probe', nav: { section: 'main', order: 55, label: 'Probe' },
  load: () => import('./pages/probe.js') });
definePage({ path: '/logs', title: () => 'Logs', nav: { section: 'main', order: 70, label: 'Logs' },
  load: () => import('./pages/logs.js') });
definePage({ path: '/system', title: () => 'System', load: () => import('./pages/system.js') });
definePage({ path: '/me', title: () => 'Profile', load: () => import('./pages/profile.js') });
definePage({ path: '/welcome', title: () => 'npcd', load: () => import('./pages/landing.js') });

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

function themeButton() {
  const t = readTheme();
  const meta = THEMES.find((x) => x[0] === t) || THEMES[0];
  const btn = h('button', {
    class: 'btn sm', title: 'Appearance',
    onClick: (e) => { e.stopPropagation(); menuOpen = !menuOpen; renderChrome(); },
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
  const current = items.find((p) => activeNavPage && activeNavPage.path === p.path);

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
    onClick: (e) => { e.stopPropagation(); navMenuOpen = !navMenuOpen; renderNav(); },
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

/* Sign-in is real. `/v1/me` answers only for a caller whose session assertion
 * the daemon could verify against the estate's shared key, so being signed in
 * is not something this page can decide — it is something it discovers.
 *
 * The gateway owns the flow. `/auth/login` is served on every hostname ahead of
 * site routing, and the cookie it issues is on `.tokera.com`, which is what
 * makes one sign-in carry to code. and bot. without either daemon taking part.
 * So signing in is a navigation, not a fetch. */
const here = () => location.pathname + location.search + location.hash;
window.__npcdSignIn = () => {
  location.href = '/auth/login?next=' + encodeURIComponent(here());
};
window.__npcdSignOut = () => {
  location.href = '/auth/logout?next=' + encodeURIComponent('/#/welcome');
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

  let status = null;
  for (let i = 0; i < 40; i++) {
    try {
      status = await API.getStatus();
      if (status.state === 'ready') break;
      if (detail) detail.textContent = status.detail || (status.loading && status.loading.current) || 'loading…';
    } catch (e) {
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

  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (drawerOpen) return setDrawer(false);
    if (navMenuOpen) { navMenuOpen = false; renderNav(); }
  });

  // Not signed in → the landing page owns the viewport.
  if (!ME && path() !== '/welcome') {
    location.replace('#/welcome');
  }

  start(document.getElementById('outlet'), (page, params) => {
    renderNav(page);
    document.title = page ? (typeof page.title === 'function' ? page.title(params) : page.title) + ' · npcd' : 'npcd';
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
      const to = { r: '/', w: '/worlds', a: '/archetypes', t: '/tools', s: '/system' }[e.key];
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
