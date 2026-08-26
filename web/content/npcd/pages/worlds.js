/* Worlds (§38).
 *
 * A world is not a name and a blurb — it is the projection schema its characters
 * think through. Two things live here and nothing else matters as much:
 * the SUBSTRATE LAYERS (window, budget, selection rule, masking) and the
 * SECTION COLLECTIONS the system-prompt lens is assembled from.
 *
 * Everything on this page states its blast radius, because editing any of it
 * changes how every character in the world thinks. */

import { API } from '../lib/api.js';
import { h, mount, fmtK, worldTime } from '../lib/dom.js';
import { go } from '../lib/router.js';
import { empty, toast, confirmDialog, layerColor, bar, modal } from '../lib/ui.js';

export async function render(params, q) {
  const worlds = (await API.listWorlds().catch(() => ({ worlds: [] }))).worlds || [];
  const wid = params.wid || (worlds[0] && worlds[0].world_id);
  const w = worlds.find((x) => x.world_id === wid);
  const tab = q.tab || 'layers';

  const el = h('div', { class: 'page wide' });

  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, w ? w.name : 'Worlds'),
      h('div', { class: 'sub' },
        'A world is the substrate schema its characters run under: the layers they think through, and the ' +
        'authored sections their lens is built from.')),
    h('div', { class: 'row' },
      worlds.length > 1
        ? h('select', { class: 'select', style: 'width:auto', onChange: (e) => go('/world/' + e.target.value) },
          worlds.map((x) => h('option', { value: x.world_id, selected: x.world_id === wid }, x.name)))
        : null,
      h('button', { class: 'btn primary', onClick: () => toast('creating a world — engine required', 'err') }, '+ New world'))));

  if (!w) {
    el.appendChild(empty('◍', 'No worlds yet',
      'A character needs a world to live in — and a world is the schema that character thinks through.',
      h('button', { class: 'btn primary' }, '+ New world')));
    return { el };
  }

  const TABS = [
    ['layers', 'Substrate layers'],
    ['collections', 'Section collections'],
    ['setting', 'Setting & world knowledge'],
    ['clock', 'Narrative clock'],
  ];
  el.appendChild(h('div', { class: 'row', style: 'gap:4px;margin-bottom:18px' },
    TABS.map(([k, label]) => h('button', {
      class: 'btn sm' + (tab === k ? ' primary' : ' ghost'),
      onClick: () => go(`/world/${wid}?tab=${k}`),
    }, label))));

  const host = h('div', {});
  el.appendChild(host);

  ({ layers, collections, setting, clock }[tab] || layers)();

  // ── substrate layers ──────────────────────────────────────────────────────

  async function layers() {
    const s = await API.getLayerSchema().catch(() => ({ layers: [] }));
    const maxWindow = Math.max(...(s.layers || []).map((l) => l.window || 0), 1);

    mount(host,
      h('div', { class: 'panel', style: 'margin-bottom:12px' },
        h('div', { class: 'tiny dim' },
          'Layer geometry is the calibration surface: window, budget priority and floor, score threshold, ' +
          'and the selection rule. It is data rather than code precisely so it can be moved without a rebuild — ' +
          `and every change here reaches all ${w.npc_count} characters in ${w.name}.`)),

      h('div', { class: 'list' }, (s.layers || []).map((l) => h('div', {
        style: 'padding:14px 18px',
      },
        h('div', { class: 'row', style: 'gap:10px;margin-bottom:8px' },
          h('span', { style: `width:3px;height:16px;border-radius:2px;background:${layerColor(l.layer)}` }),
          h('strong', { style: 'font-size:.9rem' }, l.layer),
          l.masking === 'cross-timeline'
            ? h('span', { class: 'chip warn', title: 'shared across characters' }, 'cross-timeline')
            : h('span', { class: 'chip' }, 'self-local'),
          l.summarize ? h('span', { class: 'chip' }, 'summarised') : null,
          h('span', { style: 'flex:1' }),
          h('span', { class: 'chip accent' }, l.decode_priority + ' priority')),

        h('div', { class: 'tiny dim', style: 'margin-bottom:10px;max-width:74ch' }, l.description),

        h('div', { class: 'grid g4' },
          field('window', fmtK(l.window), bar((l.window || 0) / maxWindow, layerColor(l.layer))),
          field('budget priority', String(l.budget?.priority ?? '—')),
          field('budget floor', (l.budget?.min_percent ?? 0) + '%'),
          field('score threshold', String(l.score_threshold ?? 0))),

        h('div', { class: 'row', style: 'gap:9px;margin-top:10px' },
          h('span', { class: 'tiny dim' }, 'selection'),
          h('code', { class: 'mono tiny', style: 'color:var(--ink-dim)' }, l.selection),
          h('span', { style: 'flex:1' }),
          h('button', { class: 'btn sm ghost', onClick: () => editLayer(l) }, 'Edit'))))));
  }

  function field(label, value, extra) {
    return h('div', {},
      h('div', { class: 'tiny dim' }, label),
      h('div', { class: 'mono', style: 'font-size:.9rem;font-weight:700' }, value),
      extra || null);
  }

  function editLayer(l) {
    modal({
      title: 'Layer · ' + l.layer,
      body: h('div', {},
        h('div', { class: 'grid g2' },
          h('label', { class: 'field' }, h('span', {}, 'Window (tokens)'), h('input', { class: 'input', value: l.window })),
          h('label', { class: 'field' }, h('span', {}, 'Score threshold'), h('input', { class: 'input', value: l.score_threshold }))),
        h('div', { class: 'grid g2' },
          h('label', { class: 'field' }, h('span', {}, 'Budget priority'), h('input', { class: 'input', value: l.budget?.priority })),
          h('label', { class: 'field' }, h('span', {}, 'Budget floor %'), h('input', { class: 'input', value: l.budget?.min_percent }))),
        h('label', { class: 'field' }, h('span', {}, 'Selection rule'), h('input', { class: 'input', value: l.selection })),
        h('label', { class: 'field' }, h('span', {}, 'Masking'),
          h('select', { class: 'select' },
            ['self-local', 'cross-timeline'].map((m) => h('option', { selected: m === l.masking }, m)))),
        h('div', { class: 'tiny dim' },
          'Cross-timeline masking lets this layer be read across characters. Only the world layer should ever use it — ' +
          'on a private layer it is the scope leak the isolation test exists to catch.')),
      footer: [h('button', { class: 'btn primary', onClick: () => toast('layer schema edit — engine required', 'err') }, 'Save')],
    });
  }

  // ── section collections ───────────────────────────────────────────────────

  async function collections() {
    const c = await API.getWorldCollections(wid).catch(() => ({ collections: [] }));
    mount(host,
      h('div', { class: 'panel', style: 'margin-bottom:12px' },
        h('div', { class: 'tiny dim' },
          'The system prompt is the immutable core, surfaced. Each collection is a folder of authored sections; ' +
          'a selection rule decides which of them reach the frame this turn. Per-turn variation is selection ' +
          'over a fixed substrate, never mutation of it.')),
      (c.collections || []).map(collectionCard));
  }

  function collectionCard(col) {
    const body = h('div', { hidden: true });
    const toggle = h('button', { class: 'btn sm ghost', onClick: () => {
      body.hidden = !body.hidden;
      toggle.textContent = body.hidden ? '▸ ' + col.sections.length + ' sections' : '▾ hide';
    } }, '▸ ' + col.sections.length + ' sections');

    mount(body, h('div', { class: 'list', style: 'margin-top:10px' },
      col.sections.map((s) => h('div', { style: 'padding:11px 15px' },
        h('div', { class: 'row', style: 'gap:9px' },
          h('code', { class: 'mono', style: 'color:var(--accent);font-size:.8rem' }, s.id),
          h('span', { class: 'chip' }, s.category),
          h('span', { style: 'flex:1' }),
          h('span', { class: 'tiny dim mono' }, s.tokens + ' tok'),
          s.examples
            ? h('span', { class: 'chip ok', title: 'calibration lead-ins' }, s.examples + ' examples')
            : h('span', { class: 'chip warn' }, 'uncalibrated'),
          col.locked ? null : h('button', { class: 'btn sm ghost', onClick: () => editSection(col, s) }, 'Edit')),
        h('div', { class: 'tiny', style: 'color:var(--ink-soft);margin-top:5px;max-width:88ch' }, s.template)))));

    return h('div', { class: 'panel' },
      h('div', { class: 'row', style: 'gap:9px' },
        h('strong', { style: 'font-size:.9rem' }, col.name),
        h('code', { class: 'mono tiny dim' }, col.folder),
        col.locked ? h('span', { class: 'chip', title: 'read-only by construction' }, 'immutable') : null,
        h('span', { class: 'chip accent' }, col.source),
        h('span', { style: 'flex:1' }),
        h('span', { class: 'chip' }, col.rule),
        toggle),
      h('div', { class: 'tiny dim', style: 'margin-top:7px;max-width:80ch' }, col.description),
      body);
  }

  function editSection(col, s) {
    modal({
      title: col.name + ' · ' + s.id, wide: true,
      body: h('div', {},
        h('label', { class: 'field' }, h('span', {}, 'Template — installed as this section’s content'),
          h('textarea', { class: 'textarea', rows: 5 }, s.template)),
        h('div', { class: 'tiny dim' },
          `Selected by: ${col.rule}. ` + (s.examples
            ? `${s.examples} calibration lead-ins train this section’s selection.`
            : 'No calibration examples — this section will be selected worse than its neighbours.'))),
      footer: [h('button', { class: 'btn primary', onClick: () => toast('section edit — engine required', 'err') }, 'Save')],
    });
  }

  // ── setting / clock ───────────────────────────────────────────────────────

  function setting() {
    mount(host,
      h('div', { class: 'panel' },
        h('div', { class: 'grid g2' },
          h('label', { class: 'field' }, h('span', {}, 'Name'), h('input', { class: 'input', value: w.name })),
          h('label', { class: 'row', style: 'gap:9px;margin-top:22px;cursor:pointer' },
            h('input', { type: 'checkbox', checked: w.public }),
            h('div', {}, h('div', { style: 'font-size:.86rem;font-weight:600' }, 'Public'),
              h('div', { class: 'tiny dim' }, 'anyone may spawn characters here')))),
        h('label', { class: 'field' },
          h('span', {}, 'World knowledge — the shared immutable core'),
          h('textarea', { class: 'textarea', rows: 6 }, w.setting)),
        h('div', { class: 'tiny dim' },
          `Read by every character in ${w.name}. Editing it changes what all ${w.npc_count} of them know.`)),

      h('h2', {}, 'Map zoom bands'),
      h('div', { class: 'panel' },
        h('div', { class: 'row wrap', style: 'gap:6px' },
          (w.zoom_bands || []).map((b) => h('span', { class: 'chip accent' }, b)),
          h('input', { class: 'input', placeholder: 'add band…', style: 'width:130px' })),
        h('div', { class: 'tiny dim', style: 'margin-top:9px' },
          'The `zoom` values perception maps may declare. Declared per world, because a city game and a ' +
          'campaign game want different granularities.')));
  }

  function clock() {
    const t = w.time || {};
    mount(host, h('div', { class: 'panel' },
      h('div', { class: 'row wrap', style: 'gap:26px' },
        h('div', {}, h('div', { class: 'tiny dim' }, 'now'),
          h('div', { class: 'mono', style: 'font-size:1.3rem;font-weight:700' }, worldTime(t.world_ms))),
        h('div', {}, h('div', { class: 'tiny dim' }, 'scale'),
          h('select', { class: 'select', style: 'width:auto' },
            [0, 1, 10, 60, 360, 1440].map((s) => h('option', { value: s, selected: s === t.scale },
              s === 0 ? 'paused' : s + '× real time')))),
        h('div', {}, h('div', { class: 'tiny dim' }, ' '),
          h('button', { class: 'btn', onClick: () => confirmDialog({
            title: 'Jump the world clock',
            message: `This affects every character in ${w.name}. They will experience the gap, and the ` +
              'consolidation folds will have it to reconcile.',
            confirmText: 'Jump',
            onConfirm: () => toast('clock jumped', 'ok'),
          }) }, 'Jump to…'))),
      h('div', { class: 'tiny dim', style: 'margin-top:14px' },
        'Scale is world-seconds per real second; 0 pauses. Narrative time is what characters date their ' +
        'memories by — wall time is only ever a diagnostic.')));
  }

  return { el };
}
