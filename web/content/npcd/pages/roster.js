/* My NPCs (§27) — the landing page once signed in.
 *
 * The tag field is an ordinary filter beside world and state. Nothing on this
 * page indicates that hidden characters exist: no count, no lock, no hint.
 * A hidden NPC surfaced by a tag filter renders identically to any other.
 *
 * The page never polls. `/ws/events` pushes what changed and only the affected
 * row's mutable nodes are swapped — a re-list per event would rebuild every row
 * on the page to move one badge, and it would do it under the reader's cursor
 * while they are trying to click something. */

import { API } from '../lib/api.js';
import { h, mount, ago } from '../lib/dom.js';
import { go, link } from '../lib/router.js';
import { avatar, stateDot, bandChip, pending, empty, STATE_LABEL } from '../lib/ui.js';

const VIEW_KEY = 'npcd.roster.view';

export async function render() {
  const el = h('div', { class: 'page' });
  let view = (() => { try { return localStorage.getItem(VIEW_KEY) || 'list'; } catch (_) { return 'list'; } })();
  const filters = { tag: '', state: 'any', world_id: '', q: '' };

  const results = h('div', {});
  const worlds = (await API.listWorlds().catch(() => ({ worlds: [] }))).worlds || [];

  const tagInput = h('input', {
    class: 'input', placeholder: 'tag', 'data-search': '', style: 'width:150px',
    onInput: (e) => { filters.tag = e.target.value; refresh(); },
  });
  const nameInput = h('input', {
    class: 'input', placeholder: 'search names', style: 'width:170px',
    onInput: (e) => { filters.q = e.target.value; refresh(); },
  });
  const worldSel = h('select', { class: 'select', onChange: (e) => { filters.world_id = e.target.value; refresh(); } },
    h('option', { value: '' }, 'all worlds'),
    worlds.map((w) => h('option', { value: w.world_id }, w.name)));
  const stateSel = h('select', { class: 'select', onChange: (e) => { filters.state = e.target.value; refresh(); } },
    ['any', 'active', 'ticking', 'idle', 'asleep', 'suspended'].map((s) => h('option', { value: s }, s)));

  const viewBtn = (mode, glyph, title) => h('button', {
    class: 'btn sm' + (view === mode ? ' primary' : ' ghost'), title,
    onClick: () => { view = mode; try { localStorage.setItem(VIEW_KEY, mode); } catch (_) {} refresh(); },
  }, glyph);

  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, 'My NPCs'),
      h('div', { class: 'sub' }, 'Characters you own or have been given access to.')),
    h('div', { class: 'row' },
      link('/npc/new', { class: 'btn primary' }, '+ New NPC'))));

  el.appendChild(h('div', { class: 'filters' },
    tagInput, nameInput, worldSel, stateSel,
    h('span', { style: 'flex:1' }),
    viewBtn('cards', '▦', 'card view'), viewBtn('list', '☰', 'list view')));

  el.appendChild(results);

  /* npc_id → the nodes on its row that an event can change. Rebuilt with the
   * list, because a node from a discarded render is detached and updating it
   * writes into nothing. */
  const live = new Map();

  function track(n, nodes) {
    live.set(n.npc_id, {
      n,
      apply(ev) {
        if (ev.state) n.state = ev.state;
        if (ev.pending_events != null) n.tick = { ...(n.tick || {}), pending_events: ev.pending_events };
        if (ev.band || ev.overlap != null) {
          n.monitor = { overlap: ev.overlap ?? n.monitor?.overlap, band: ev.band || n.monitor?.band };
        }
        swap(nodes, 'dot', stateDot(n.state));
        swap(nodes, 'pend', pending(n.tick?.pending_events ?? null));
        swap(nodes, 'band', bandChip(n.monitor?.band ?? null));
        if (nodes.meta) nodes.meta.textContent = metaLine(n);
      },
    });
    return nodes;
  }

  function swap(nodes, key, next) {
    const cur = nodes[key];
    if (!cur || !cur.isConnected) return;
    cur.replaceWith(next);
    nodes[key] = next;
  }

  const metaLine = (n) => [
    `tick ${Math.round((n.tick?.heartbeat_ms || 0) / 1000)}s`,
    `last ${ago(n.tick?.last_tick_ms)}`,
    n.live_interactions ? `${n.live_interactions} live` : STATE_LABEL[n.state] || n.state,
  ].join(' · ');

  async function refresh() {
    live.clear();
    const { items } = await API.listNpcs(filters);
    if (!items.length) {
      mount(results, filters.tag || filters.q
        ? empty('◌', 'No characters match', 'Try a different tag or name.')
        : empty('◈', 'No characters yet',
          'Create one and it will start living here.',
          link('/npc/new', { class: 'btn primary' }, '+ New NPC')));
      return;
    }
    mount(results, view === 'cards' ? cards(items) : rows(items));
  }

  function rows(items) {
    // One surface with hairline dividers — not a stack of bordered boxes.
    return h('div', { class: 'list' }, items.map((n) => {
      const p = track(n, {
        dot: stateDot(n.state),
        // No `|| 0` / `|| 'healthy'`: both are engine measurements, and a
        // character the engine has never run has neither. The helpers render
        // absence as absence.
        pend: pending(n.tick?.pending_events ?? null),
        band: bandChip(n.monitor?.band ?? null),
        meta: h('div', { class: 'npc-meta' }, metaLine(n)),
      });
      return h('div', { class: 'npc-row', onClick: () => go('/npc/' + n.npc_id) },
        avatar(n),
        h('div', { style: 'min-width:0' },
          h('div', { class: 'row', style: 'gap:7px' },
            p.dot,
            h('span', { class: 'npc-name' }, n.name),
            h('span', { class: 'chip' }, n.personality_name || n.personality_id),
            n.access && n.access !== 'owner' ? h('span', { class: 'chip' }, n.access) : null),
          p.meta),
        h('div', { class: 'row', style: 'gap:10px' }, p.pend, p.band));
    }));
  }

  function cards(items) {
    return h('div', { class: 'cards' }, items.map((n) => {
      const p = track(n, {
        dot: stateDot(n.state),
        // No `|| 0` / `|| 'healthy'`: both are engine measurements, and a
        // character the engine has never run has neither. The helpers render
        // absence as absence.
        pend: pending(n.tick?.pending_events ?? null),
        band: bandChip(n.monitor?.band ?? null),
      });
      return h('div', { class: 'npc-card', onClick: () => go('/npc/' + n.npc_id) },
        h('div', { class: 'art' }, (n.name || '?')[0]),
        h('div', { class: 'meta' },
          h('div', { class: 'row', style: 'gap:6px' }, p.dot, h('span', { class: 'npc-name' }, n.name)),
          h('div', { class: 'npc-meta' }, n.personality_name || ''),
          h('div', { class: 'row', style: 'gap:8px;margin-top:8px' }, p.pend, p.band)));
    }));
  }

  await refresh();

  // An event for an NPC the current filter excludes is dropped: it is not on
  // screen, and re-listing to find out whether it now matches is the polling
  // this replaced.
  const sub = API.subscribeEvents((ev) => {
    const row = live.get(ev.npc_id);
    if (row) row.apply(ev);
  });

  return { el, teardown: () => sub.close() };
}
