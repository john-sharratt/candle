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

function renderChrome() {
  const host = document.getElementById('chrome');
  if (!host) return;
  mount(host,
    themeButton(),
    ME
      ? h('button', {
        class: 'btn sm ghost', title: ME.display + ' · ' + (ME.email || ''),
        onClick: () => go('/me'),
      }, h('span', {
        style: 'width:20px;height:20px;border-radius:50%;background:var(--accent);color:var(--accent-ink);display:grid;place-items:center;font-size:.66rem;font-weight:800',
      }, (ME.unique_name || '?')[0]), ME.unique_name)
      : h('a', { class: 'btn sm primary', href: '#/welcome' }, 'Sign in'));
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

/* There is no real auth yet, so `/v1/me` always answers. A local flag lets the
 * signed-out half of the product — the landing page, the provider list — be
 * reached and exercised. It disappears the day OAuth lands. */
const signedOut = () => { try { return localStorage.getItem('npcd.signedout') === '1'; } catch (_) { return false; } };
export const setSignedOut = (v) => {
  try { v ? localStorage.setItem('npcd.signedout', '1') : localStorage.removeItem('npcd.signedout'); } catch (_) {}
};
window.__npcdSignIn = () => { setSignedOut(false); location.hash = '#/'; location.reload(); };

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

  if (signedOut()) ME = null;
  else { try { ME = await API.getMe(); } catch (_) { ME = null; } }

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
