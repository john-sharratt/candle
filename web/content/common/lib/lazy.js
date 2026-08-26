/* Fetch-on-expand disclosure, ported from zend's substrate.html.
 *
 * Three rules that page arrived at, and the reasons they exist:
 *
 *   1. A collapsed node holds NO DOM. The body is emptied on close, so a page
 *      with a thousand collapsible turns costs nothing until they are opened.
 *   2. The loader runs on expand, not on render — and its result is cached, so
 *      re-opening rebuilds from memory with no second round-trip.
 *   3. Every async body re-checks that it is still open after the await. A fetch
 *      that resolves *after* the user collapsed the card would otherwise render
 *      into — and leave stale DOM inside — a closed node.
 */

import { h, mount } from './dom.js';

/** Still expanded? Also reads false for a detached node (teardown mid-fetch). */
export function stillOpen(node) {
  const c = node && node.closest && node.closest('[data-disclosure]');
  return !!c && c.hasAttribute('data-open');
}

export function spinner(text) {
  return h('div', { class: 'lazy-wait' },
    h('span', { class: 'lazy-spin' }), text || 'loading…');
}

/**
 * A disclosure whose body is built on each expand and discarded on collapse.
 *
 *   disclosure({
 *     head: [ …nodes… ],            // rendered inside the clickable header
 *     accent: 'var(--l-beliefs)',   // left rail colour
 *     open: false,
 *     body: async (host) => { … },  // must check stillOpen(host) after awaits
 *   })
 */
export function disclosure({ head, body, accent, open = false, dense = false }) {
  const bodyEl = h('div', { class: 'disc-body' });
  const chev = h('span', { class: 'disc-chev' }, '▸');

  const card = h('div', {
    class: 'disc' + (dense ? ' dense' : ''),
    'data-disclosure': '',
    style: accent ? `border-left-color:${accent}` : '',
  });

  const toggle = () => {
    const isOpen = card.hasAttribute('data-open');
    if (isOpen) {
      card.removeAttribute('data-open');
      bodyEl.replaceChildren();          // collapsed nodes hold no DOM
    } else {
      card.setAttribute('data-open', '');
      const r = body(bodyEl);
      if (r && typeof r.catch === 'function') {
        r.catch((e) => {
          if (stillOpen(bodyEl)) mount(bodyEl, h('div', { class: 'lazy-err' }, 'Failed to load: ' + (e.message || e)));
        });
      }
    }
  };

  card.appendChild(h('div', { class: 'disc-hd', onClick: toggle }, chev, head));
  card.appendChild(bodyEl);
  if (open) toggle();
  return card;
}

/**
 * A keyed cache for lazy loaders. `get(key, loader)` fetches once and reuses
 * thereafter; `invalidate()` drops everything (used when the daemon reports
 * structurally new data and the user asks to reload).
 */
export function cache() {
  const m = new Map();
  return {
    async get(key, loader) {
      if (!m.has(key)) m.set(key, await loader());
      return m.get(key);
    },
    has: (k) => m.has(k),
    invalidate() { m.clear(); },
  };
}
