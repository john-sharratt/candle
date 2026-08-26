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

  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, 'Logs'),
      h('div', { class: 'sub' },
        'Structured lines from the daemon — filtering is a property test, not a regex over formatted text.')),
    h('div', { class: 'row' }, live, search, levelSel, copyBtn,
      h('button', { class: 'btn sm ghost', onClick: () => { ring.clear(); } }, 'Clear'))));
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

  // The backlog arrives on the same socket as the tail, so there is no window
  // in which a line can be both "already seeded" and "not yet subscribed".
  // Lines the filter excludes still enter `items` — they are what a widened
  // filter has to be able to show without re-fetching anything.
  const sub = API.subscribeLogs(
    (l) => { if (passes(l)) ring.push(l); else ring.items.push(l); },
    (state) => {
      const ok = state === 'live';
      live.className = 'chip ' + (ok ? 'ok' : 'warn');
      live.textContent = ok ? '● live' : '○ reconnecting…';
    },
  );

  repaint();

  return { el, teardown: () => sub.close() };
}
