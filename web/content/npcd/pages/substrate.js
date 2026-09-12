/* Substrate inspector — a lazy tree over the whole substrate.
 *
 * Ported in shape from zend's substrate.html: the overview is deliberately
 * light (storage + counts + layer metadata), and everything below it loads on
 * expand and caches. A collapsed node holds no DOM, so a character with four
 * thousand memory turns costs nothing until you open it.
 *
 * The poll refreshes the cheap header in place. When the structure changes it
 * OFFERS a reload rather than performing one — rebuilding would collapse
 * whatever the operator has open and throw away its fetched sub-tree. */

import { API } from '../lib/api.js';
import { h, mount, fmtNum, fmtK, worldTime } from '../lib/dom.js';
import { go } from '../lib/router.js';
import { layerColor, bar, empty, toast } from '../lib/ui.js';
import { disclosure, stillOpen, spinner, cache } from '../lib/lazy.js';
import { copyText } from '../lib/clip.js';

const KIND_COLOR = {
  user: 'var(--l-beliefs)', assistant: 'var(--l-action)',
  thinking: 'var(--l-agency)', glue: 'var(--ink-ghost)',
};

export async function render(_params, q) {
  // `subs` scopes this page's styling — see the block in `app.css`.
  const el = h('div', { class: 'page wide subs' });
  const npcs = (await API.listNpcs({}).catch(() => ({ items: [] }))).items || [];
  let npcId = q.npc || (npcs[0] && npcs[0].npc_id);

  const layerCache = cache();
  const turnCache = cache();

  const kpiHost = h('div', { class: 'grid g4', style: 'margin-bottom:16px' });
  const storageHost = h('div', { style: 'margin-bottom:16px' });
  const treeHost = h('div', {});
  const liveBtn = h('button', { class: 'chip', title: 'polling every 6s' }, '● live');

  const sel = h('select', {
    class: 'select', style: 'width:auto',
    onChange: (e) => { npcId = e.target.value; layerCache.invalidate(); turnCache.invalidate(); paintAll(); },
  }, npcs.map((n) => h('option', { value: n.npc_id, selected: n.npc_id === npcId }, n.name)));

  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, 'Substrate'),
      h('div', { class: 'sub' },
        'Layer occupancy, then conversations, then turns, then the K/V segment vector. Each level loads when you open it.')),
    h('div', { class: 'row' }, liveBtn, h('span', { class: 'tiny dim' }, 'character'), sel)));
  el.appendChild(kpiHost);
  el.appendChild(storageHost);
  el.appendChild(treeHost);

  let lastShape = null;

  // ── header (cheap, polled) ────────────────────────────────────────────────

  function paintKpis(sub, schema) {
    const layers = sub.layers || [];
    const totTok = layers.reduce((a, l) => a + (l.tokens || 0), 0);
    const totTurns = layers.reduce((a, l) => a + (l.turns || 0), 0);
    const resident = layers.length
      ? Math.round(layers.reduce((a, l) => a + (l.resident || 0), 0) / layers.length) : 0;
    mount(kpiHost,
      // `gather_scope`, the schema's own word. This read `l.masking ===
      // 'cross-timeline'`, which was the fixture's vocabulary and matched
      // nothing in `projection.yaml` — so the count was silently always zero
      // once the route became real.
      stat('layers', layers.length,
        (schema.layers || []).filter((l) => l.gather_scope === 'shared').length + ' cross-timeline'),
      stat('turns', fmtNum(totTurns)),
      stat('tokens', fmtK(totTok)),
      stat('mean resident', resident + '%', resident < 50 ? 'paged out' : 'warm'));
  }

  function stat(lbl, val, note) {
    return h('div', { class: 'panel stat' },
      h('div', { class: 'lbl' }, lbl),
      h('div', { class: 'val' }, String(val)),
      note ? h('div', { class: 'tiny dim' }, note) : null);
  }

  // ── storage ───────────────────────────────────────────────────────────────

  /* The redo log every character's memory is written into.
   *
   * Daemon-scoped rather than per-character, and the one part of this page that
   * does not wait on the engine: a segmented append-only log is a directory of
   * files, so `/v1/substrate/storage` reads it straight off disk. The layer
   * tree below is still the console's fixture until an engine opens a
   * substrate; this is not. */
  async function paintStorage() {
    let s;
    try {
      s = await API.getSubstrateStorage();
    } catch (_) {
      // The whole page is declared `role: 'admin'` in `app.js` — it names the
      // redo log's absolute path — so a refusal here is not a permission
      // problem, it is the daemon being unreachable. Nothing to explain that
      // the page's other panels will not already be showing.
      mount(storageHost);
      return;
    }
    if (!s || !s.open) {
      mount(storageHost, h('div', { class: 'panel' },
        h('div', { class: 'row', style: 'justify-content:space-between;align-items:baseline' },
          h('h3', { style: 'margin:0' }, 'Storage'),
          h('span', { class: 'chip' }, 'not opened')),
        h('div', { class: 'tiny dim', style: 'margin-top:8px' },
          'No substrate has been written yet — nothing has run an engine against this daemon. '
          + 'It would live at ', h('code', { class: 'mono' }, (s && s.path) || '.substrate'), '.')));
      return;
    }

    const segs = s.segments || [];
    const total = s.total_bytes || 0;
    /* Width by share of the log, so the strip reads as the file it describes:
     * a long tail of sealed segments and one growing head. */
    const strip = segs.map((g) => h('i', {
      class: g.active ? 'seg active' : 'seg',
      style: `flex:${Math.max(1, g.bytes || 1)}`,
      title: `seg-${g.id} · ${fmtBytes(g.bytes)}${g.active ? ' · active' : ''}`,
    }));

    mount(storageHost, h('div', { class: 'panel' },
      h('div', { class: 'row', style: 'justify-content:space-between;align-items:baseline' },
        h('h3', { style: 'margin:0' }, 'Storage ',
          h('span', { class: 'tiny dim' }, '· the redo log on disk')),
        h('span', { class: 'mono tiny' }, fmtBytes(total))),
      h('div', { class: 'segstrip', style: 'margin-top:10px' }, ...strip),
      h('div', { class: 'row wrap', style: 'gap:6px;margin-top:10px' },
        h('span', { class: 'chip' }, fmtNum(s.segment_count || segs.length) + ' segments'),
        s.listed === false
          // Say so rather than showing 512 of 900 as though it were all of them.
          ? h('span', { class: 'chip warn' }, 'showing newest ' + segs.length) : null,
        h('span', { class: 'chip' }, 'live chunks '
          + (s.live_chunks == null ? '—' : fmtNum(s.live_chunks))),
        h('span', { class: 'chip' + (s.dead_ratio > 0.5 ? ' warn' : '') }, 'dead '
          + (s.dead_ratio == null ? '—' : Math.round(s.dead_ratio * 100) + '%')),
        h('span', { class: 'tiny dim mono', style: 'margin-left:auto' }, s.path || '')),
      s.live_chunks == null
        ? h('div', { class: 'tiny st-na', style: 'margin-top:8px' },
          'Live chunk count and reclaimable fraction live in the substrate’s in-memory '
          + 'index, so they appear when an engine holds one open.')
        : null));
  }

  function fmtBytes(b) {
    if (b == null) return '—';
    const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
    let i = 0, v = b;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i += 1; }
    return v.toFixed(i >= 2 ? 1 : 0) + ' ' + u[i];
  }

  // ── tree ──────────────────────────────────────────────────────────────────

  async function paintAll() {
    // Storage is daemon-scoped, so it paints whether or not a character is
    // selected — and it is the only real thing on the page today.
    paintStorage();
    if (!npcId) return mount(treeHost, empty('◌', 'No characters', 'Create one to see its substrate.'));
    const [sub, schema] = await Promise.all([
      API.getSubstrate(npcId).catch(() => ({ layers: [] })),
      API.getLayerSchema().catch(() => ({ layers: [] })),
    ]);
    lastShape = JSON.stringify((sub.layers || []).map((l) => [l.layer, l.turns]));
    paintKpis(sub, schema);
    // Keyed by `name` — the schema names a layer that way; the occupancy rows
    // from the substrate call the same thing `layer`.
    const byName = Object.fromEntries((schema.layers || []).map((l) => [l.name, l]));
    mount(treeHost, (sub.layers || []).map((l) => layerCard(l, byName[l.layer] || {})));
  }

  function layerCard(l, s) {
    const frac = Math.min(1, (l.tokens || 0) / (l.window || 1));
    const head = [
      h('span', { class: 'disc-swatch', style: `background:${layerColor(l.layer)}` }),
      h('span', { class: 'disc-title mono' }, l.layer),
      h('div', { class: 'disc-meta' },
        s.gather_scope === 'shared'
          ? h('span', { class: 'chip warn' }, 'cross-timeline')
          : h('span', { class: 'chip' }, 'self-local'),
        h('span', { class: 'chip' }, fmtNum(l.turns) + ' turns'),
        h('span', { class: 'chip' }, fmtK(l.tokens) + ' tok'),
        h('span', { class: 'chip' }, Math.round(frac * 100) + '% of window'),
        h('span', { class: 'chip' }, (l.resident ?? '—') + '% resident')),
    ];
    return disclosure({
      accent: layerColor(l.layer),
      head,
      body: async (host) => {
        mount(host, spinner('loading turns…'));
        const data = await layerCache.get(npcId + '::' + l.layer, () => API.getLayer(npcId, l.layer));
        if (!stillOpen(host)) return;                 // collapsed while fetching
        mount(host,
          s.description ? h('div', { class: 'disc-desc' }, s.description) : null,
          h('div', { class: 'row wrap', style: 'gap:6px;margin:8px 0 4px' },
            s.selection ? h('span', { class: 'chip' }, 'selection · ' + s.selection) : null,
            s.window ? h('span', { class: 'chip' }, 'window ' + fmtK(s.window)) : null,
            s.score_threshold != null ? h('span', { class: 'chip' }, 'threshold ' + s.score_threshold) : null,
            s.decode_priority ? h('span', { class: 'chip accent' }, s.decode_priority + ' priority') : null),
          bar(frac, layerColor(l.layer)),
          (data.items || []).length
            ? (data.items || []).map((t) => turnCard(l.layer, t))
            : h('div', { class: 'lazy-err' }, 'No turns in this layer yet.'),
          h('div', { class: 'row', style: 'margin-top:10px' },
            h('button', {
              class: 'btn sm ghost',
              onClick: () => go(`/npc/${npcId}/${l.layer}`),
            }, 'Open as a stream →')));
      },
    });
  }

  function turnCard(layer, t) {
    const head = [
      h('span', { class: 'disc-idx mono' }, '#' + t.turn),
      h('span', { class: 'disc-title' }, t.kind === 'act' ? 'act' : 'turn'),
      h('div', { class: 'disc-meta' },
        h('span', { class: 'tiny dim mono' }, worldTime(t.world_ms)),
        h('span', { class: 'chip' }, 'score ' + (t.score ?? 0).toFixed(2)),
        h('span', { class: 'chip' }, t.tokens + ' tok')),
    ];
    return disclosure({
      dense: true,
      accent: layerColor(layer),
      head,
      body: async (host) => {
        mount(host, spinner());
        const full = await turnCache.get(`${npcId}::${layer}::${t.turn}`,
          () => API.getTurn(npcId, layer, t.turn).catch(() => ({ text: t.preview, layout: null })));
        if (!stillOpen(host)) return;
        const copyBtn = h('button', { class: 'btn sm ghost' }, 'Copy');
        copyBtn.addEventListener('click', () => copyText(full.text || t.preview || '', copyBtn));
        mount(host,
          h('div', { class: 'row', style: 'justify-content:flex-end;margin-bottom:6px' }, copyBtn),
          h('pre', { class: 'disc-pre' }, full.text || t.preview || '(no body)'),
          full.layout && full.layout.segments && full.layout.segments.length
            ? kvLayout(full.layout)
            : null);
      },
    });
  }

  /* The turn's K/V segment vector, rendered verbatim. A segment with `kv == null`
   * is ETHEREAL — recorded, but not part of this turn's own K/V grid (the spine
   * materialised it, or a reasoning block was dropped). Dimmed and italicised so
   * that distinction reads at a glance rather than being invisible. */
  function kvLayout(layout) {
    const segs = layout.segments;
    const real = segs.filter((s) => s.kv != null).length;
    return disclosure({
      dense: true,
      accent: 'var(--ink-ghost)',
      head: [
        h('span', { class: 'disc-title' }, 'K/V layout'),
        h('div', { class: 'disc-meta' },
          h('span', { class: 'chip' }, segs.length + ' segments'),
          h('span', { class: 'chip' }, real + ' with K/V'),
          h('span', { class: 'chip' }, (segs.length - real) + ' ethereal')),
      ],
      body: (host) => {
        mount(host, h('pre', { class: 'disc-pre' }, segs.map((s) => {
          const kind = (s.kind || 'glue').toLowerCase();
          const ethereal = s.kv == null;
          const text = s.text != null ? s.text
            : (typeof s.marker === 'string' ? '⟐ ' + s.marker : '[' + kind + ']');
          return h('span', {
            style: `color:${KIND_COLOR[kind] || 'var(--ink-ghost)'}`
              + (ethereal ? ';opacity:.5;font-style:italic' : ''),
          }, text + '\n');
        })));
      },
    });
  }

  // ── poll: refresh the header, OFFER a reload on structural change ─────────

  await paintAll();

  const timer = setInterval(async () => {
    if (!npcId) return;
    try {
      const [sub, schema] = await Promise.all([API.getSubstrate(npcId), API.getLayerSchema()]);
      paintKpis(sub, schema);
      const shape = JSON.stringify((sub.layers || []).map((l) => [l.layer, l.turns]));
      if (shape !== lastShape) {
        // Do NOT rebuild: it would collapse everything the operator has open and
        // discard its fetched sub-tree. Offer the reload instead.
        liveBtn.className = 'chip accent';
        liveBtn.textContent = '↻ new data · click to reload';
        liveBtn.onclick = () => {
          layerCache.invalidate(); turnCache.invalidate();
          liveBtn.onclick = null;
          liveBtn.className = 'chip'; liveBtn.textContent = '● live';
          paintAll();
        };
      }
    } catch (_) {
      liveBtn.className = 'chip warn';
      liveBtn.textContent = '○ reconnecting…';
    }
  }, 6000);

  return { el, teardown: () => clearInterval(timer) };
}
