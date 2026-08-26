/* The estate switcher — one control that names every site and says which one
 * you are on.
 *
 * There are four hosts and no obvious way between them. A row of "other sites"
 * links says where you can go but never where you are, and it grows a new entry
 * every time a product ships. Folding the whole set into the brand — the one
 * element every page already has in the same corner — makes the answer to
 * "where am I" and "where else is there" the same click.
 *
 * # This list exists three times
 *
 * Rust renders it for tokera.com, this module renders it for the npcd console,
 * and `zend/web/lib/estate.js` is a copy for zend, which embeds its own assets
 * and cannot read this directory. Three copies of one list is exactly the thing
 * that drifts, so two tests pin it: `web/src/site/tokera/page.rs` compares the
 * Rust constant against this file, and `zend`'s copy is compared byte for byte.
 * Edit this file first and let the tests tell you what else to change.
 *
 * # The icons are each site's own favicon, by absolute URL
 *
 * Not copies. A copied icon is a second file to update when a brand changes,
 * and the favicon is the one asset guaranteed to exist and to be current on
 * every one of these hosts. The cost is that an icon 404s if a site is down —
 * which is why each entry also carries a colour, painted behind the image, so a
 * failed load degrades to a coloured chip rather than a broken-image glyph.
 */

export const SITES = [
  { id: 'tokera',       name: 'Tokera',        url: 'https://tokera.com/',
    icon: 'https://tokera.com/favicon.png',            tint: '#a80c0c' },
  { id: 'zend',         name: 'Zend',          url: 'https://code.tokera.com/',
    icon: 'https://code.tokera.com/favicon.svg',       tint: '#c98a3e' },
  { id: 'npcd',         name: 'NPCs',          url: 'https://bot.tokera.com/',
    icon: 'https://bot.tokera.com/favicon.svg',        tint: '#c98a3e' },
  { id: 'battlecities', name: 'Battle Cities', url: 'https://battlecities.net/',
    icon: 'https://battlecities.net/favicon-32x32.png', tint: '#3a2a24' },
];

/** The site this page belongs to, or `null` if it is not one of them. */
export const siteById = (id) => SITES.find((s) => s.id === id) || null;

/* Where a row goes. The site you are already on links to its own root rather
 * than to itself-with-a-fragment: the switcher is also the way home, which is
 * what the brand did before it grew a menu and what a reader expects of it. */
export const hrefFor = (site, currentId) => (site.id === currentId ? '/' : site.url);

/**
 * Build the switcher.
 *
 * `<details>` rather than a scripted popover, because the whole control then
 * works with no JavaScript at all: open, close, keyboard, and Escape are the
 * element's own behaviour, and a reader with scripts off still gets the menu
 * instead of a dead brand. The only script here is the one that closes it on an
 * outside click, which is the single thing `<details>` does not do itself.
 *
 * `currentId` is the id of the site this page is; pass null if it is none.
 */
export function estateSwitcher(currentId, { homeHref = '/' } = {}) {
  const current = siteById(currentId);

  const el = document.createElement('details');
  el.className = 'estate';

  const summary = document.createElement('summary');
  summary.className = 'estate-current';
  summary.setAttribute('aria-label', 'Switch site');
  summary.append(chip(current), label(current ? current.name : 'Tokera'), caret());
  el.appendChild(summary);

  const menu = document.createElement('nav');
  menu.className = 'estate-menu';
  menu.setAttribute('aria-label', 'Sites');
  for (const site of SITES) {
    const a = document.createElement('a');
    a.href = site.id === currentId ? homeHref : site.url;
    a.className = 'estate-row' + (site.id === currentId ? ' is-current' : '');
    if (site.id === currentId) a.setAttribute('aria-current', 'true');
    a.append(chip(site), label(site.name));
    if (site.id === currentId) {
      const here = document.createElement('span');
      here.className = 'estate-here';
      here.textContent = 'you are here';
      a.appendChild(here);
    }
    menu.appendChild(a);
  }
  el.appendChild(menu);
  armDismiss();
  return el;
}

/* `<details>` closes on Escape and on toggling its summary, but stays open when
 * you click past it, which reads as stuck.
 *
 * One delegated listener for the whole document, installed at most once, rather
 * than one per switcher: zend rebuilds its sidebar on every render — nine call
 * sites plus a five-second sync — and a per-instance listener would accumulate
 * without bound, each one holding a detached `<details>` alive. Closing "every
 * open switcher not containing the click" needs no reference to any particular
 * one, so nothing has to be unregistered when a switcher is thrown away. */
let dismissArmed = false;
function armDismiss() {
  if (dismissArmed || typeof document === 'undefined') return;
  dismissArmed = true;
  document.addEventListener('click', (e) => {
    for (const open of document.querySelectorAll('details.estate[open]')) {
      if (!open.contains(e.target)) open.open = false;
    }
  });
}

function chip(site) {
  const s = document.createElement('span');
  s.className = 'estate-chip';
  if (site) {
    s.style.backgroundColor = site.tint;
    s.style.backgroundImage = `url("${site.icon}")`;
  }
  return s;
}

function label(text) {
  const s = document.createElement('span');
  s.className = 'estate-name';
  s.textContent = text;
  return s;
}

function caret() {
  const s = document.createElement('span');
  s.className = 'estate-caret';
  s.setAttribute('aria-hidden', 'true');
  s.textContent = '▾';
  return s;
}
