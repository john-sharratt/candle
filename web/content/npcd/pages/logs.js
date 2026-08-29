/* Logs — structured lines, never parsed from a formatted string.
 *
 * Uses the ring-buffer discipline from zend: append ONE row per arriving line
 * and trim the head, rather than rebuilding ~600 rows per line — that per-line
 * full rebuild is what made a live stream sluggish with the pane open. The tail
 * is followed only while the reader is already at the bottom. */

import { API } from '../lib/api.js';
import { h, mount } from '../lib/dom.js';
import { ringView } from '../lib/live.js';
import { copyText } from '../lib/clip.js';

const LEVELS = ['TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR'];
const COLOR = {
  TRACE: 'var(--violet)', DEBUG: 'var(--info)', INFO: 'var(--ok)',
  WARN: 'var(--warn)', ERROR: 'var(--crit)',
};
const MAX = 600;

export async function render() {
  const el = h('div', { class: 'page wide' });

  /* No role check here: the page is declared `role: 'admin'` in `app.js`, so
   * the router refused it before this ran and the nav link was never shown.
   *
   * That matters for this page specifically. The daemon refuses the upgrade
   * with a 401 or 403, which is right on the wire — but a browser's
   * `WebSocket` does not expose the status of a failed handshake, so the socket
   * would simply not open and the page would sit on "reconnecting…" forever,
   * looking like an outage rather than a permission. The declaration is what
   * stops a non-admin ever reaching the socket. */
  let minLevel = 'DEBUG';
  let filter = '';

  const list = h('div', { class: 'logs-list mono' });
  const scroller = h('div', { class: 'logs-scroll' }, list);

  const levelSel = h('select', {
    class: 'select', style: 'width:auto',
    onChange: (e) => { minLevel = e.target.value; repaint(); },
  }, LEVELS.map((l) => h('option', { value: l, selected: l === minLevel }, l + '+')));

  const search = h('input', {
    class: 'input', placeholder: 'filter…', 'data-search': '', style: 'width:220px',
    onInput: (e) => { filter = e.target.value.toLowerCase(); repaint(); },
  });

  const copyBtn = h('button', { class: 'btn sm ghost' }, 'Copy');
  copyBtn.addEventListener('click', () => copyText(
    visible().map((l) => `${l.ts} ${l.level} ${l.target} ${l.msg}`).join('\n'), copyBtn));

  const live = h('span', { class: 'chip ok' }, '● live');
  const tally = h('span', { class: 'tiny dim' });
  const pauseBtn = h('button', { class: 'btn sm' }, '⏸ pause');

  /* Paused holds arriving lines rather than dropping them. Pausing is for
   * reading something that just went past, not for missing what came next —
   * a pane that discards while you read is worse than one that never stopped.
   * The hold is capped like the buffer, so a pause left running overnight
   * cannot grow without limit. */
  let paused = false;
  const held = [];

  pauseBtn.addEventListener('click', () => {
    paused = !paused;
    pauseBtn.textContent = paused ? '▶ resume' : '⏸ pause';
    pauseBtn.classList.toggle('primary', paused);
    if (!paused) {
      while (held.length) accept(held.shift());   // in arrival order
      repaint();
    }
    count();
  });

  const count = () => {
    tally.textContent = paused
      ? `paused · ${held.length} held`
      : `${ring.items.length} buffered`;
  };

  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, 'Logs'),
      h('div', { class: 'sub' },
        'Structured lines from the daemon — filtering is a property test, not a regex over formatted text.')),
    h('div', { class: 'row' }, live, tally, search, levelSel, pauseBtn, copyBtn,
      h('button', { class: 'btn sm ghost', onClick: () => { held.length = 0; ring.clear(); count(); } }, 'Clear'))));
  el.appendChild(scroller);

  const row = (l) => h('div', { class: 'log-row' },
    h('span', { class: 'ts' }, l.ts),
    h('span', { class: 'lvl', style: 'color:' + (COLOR[l.level] || 'var(--ink-faint)') }, l.level),
    h('span', { class: 'tgt' }, l.target),
    h('span', { class: 'msg' }, l.msg));

  const ring = ringView({ max: MAX, row, host: list, scroller });

  const passes = (l) => LEVELS.indexOf(l.level) >= LEVELS.indexOf(minLevel)
    && (!filter || (l.msg + ' ' + l.target).toLowerCase().includes(filter));
  const visible = () => ring.items.filter(passes);

  // A filter change is the ONE case that legitimately rebuilds — the set itself
  // changed. Arriving lines never do.
  function repaint() {
    const shown = visible();
    if (!shown.length) return mount(list, h('div', { class: 'lazy-err' }, 'Nothing matches'));
    mount(list, shown.map(row));
    scroller.scrollTop = scroller.scrollHeight;
  }

  /* Lines the filter excludes still enter `items` — they are what a widened
   * filter has to be able to show without re-fetching anything.
   *
   * Trimmed here as well as in `ring.push`, because pushing straight to
   * `ring.items` bypasses the ring's own cap: with a filter active, every
   * excluded line took that path and the buffer grew without bound for as long
   * as the pane stayed open. Only the *rendered* rows were ever capped, so
   * nothing on screen showed it happening. */
  function accept(l) {
    if (passes(l)) {
      ring.push(l);
    } else {
      ring.items.push(l);
      if (ring.items.length > MAX) ring.items.splice(0, ring.items.length - MAX);
    }
  }

  // The backlog arrives on the same socket as the tail, so there is no window
  // in which a line can be both "already seeded" and "not yet subscribed".
  const sub = API.subscribeLogs(
    (l) => {
      if (paused) {
        held.push(l);
        if (held.length > MAX) held.splice(0, held.length - MAX);
      } else {
        accept(l);
      }
      count();
    },
    (state) => {
      const ok = state === 'live';
      live.className = 'chip ' + (ok ? 'ok' : 'warn');
      live.textContent = ok ? '● live' : '○ reconnecting…';
    },
  );

  repaint();
  count();

  return { el, teardown: () => sub.close() };
}
