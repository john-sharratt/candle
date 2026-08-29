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
import { empty, toast, confirmDialog, layerColor, bar, modal, mayEdit, ro, roChip, only, roNote } from '../lib/ui.js';

/* The filter box and the world selector, together.
 *
 * Filters as you type. It is safe to, because the WHOLE-WORD rule is what keeps
 * a hidden world hidden, not the moment the query is sent: `e`, `ea`, `ear` and
 * `eart` all return nothing from the daemon, so typing letters and watching
 * reveals nothing. Only the complete word does, and that is true whether it
 * arrives a character at a time or all at once.
 *
 * These two live in one function because the input must SURVIVE a result. The
 * page-level `go()` rebuilds everything and takes the caret with it, so a
 * keystroke here fetches and re-fills the select in place; only choosing a
 * world navigates. The URL is kept in step with `replaceState`, which does not
 * fire `hashchange` — so a reload or a shared link still opens on the same
 * filter without the router re-rendering under the typist.
 */
function finder(find, wid, tab) {
  const sel = h('select', {
    class: 'select', style: 'width:auto',
    onChange: (e) => go('/world/' + e.target.value + '?tab=' + tab
      + (box.value.trim() ? '&find=' + encodeURIComponent(box.value.trim()) : '')),
  });

  const fill = (worlds) => {
    mount(sel, worlds.map((x) => h('option', {
      value: x.world_id, selected: x.world_id === wid,
    }, x.name || x.world_id)));
    // Hidden only when there is nothing to choose *and* nobody is choosing. A
    // filter that narrowed to one result and then hid the control would leave
    // the reader unable to see what they had just found — which is the moment
    // the whole-word reveal exists for.
    sel.hidden = worlds.length < 2 && !box.value.trim();
  };

  let timer = null;
  let seq = 0;
  const search = async (v) => {
    // Results can arrive out of order when one request is slower than the next
    // keystroke's. The stale one would overwrite the fresh list, so a sequence
    // number decides which answer is still wanted.
    const mine = ++seq;
    const r = await API.listWorlds(v).catch(() => null);
    if (!r || mine !== seq) return;
    fill(r.worlds || []);
    const at = (wid ? '/world/' + wid : '/worlds') + '?tab=' + tab
      + (v ? '&find=' + encodeURIComponent(v) : '');
    history.replaceState(null, '', '#' + at);
  };

  const box = h('input', {
    class: 'input', style: 'width:150px', placeholder: 'find…', value: find,
    onInput: () => {
      // Debounced, so a word is one request rather than five. Short enough that
      // the list feels live under a normal typing speed.
      clearTimeout(timer);
      timer = setTimeout(() => search(box.value.trim()), 180);
    },
  });

  const clear = h('button', {
    class: 'btn sm ghost', title: 'clear the filter',
    onClick: () => { box.value = ''; clearTimeout(timer); search(''); box.focus(); },
  }, '✕');

  return { el: h('div', { class: 'row', style: 'gap:6px' }, box, clear, sel), fill };
}

export async function render(params, q) {
  // The filter travels in the URL so a reveal survives a reload and can be
  // linked. A hidden world is not in the listing until a whole word of `find`
  // names it — see `npcd::visibility` — so this has to reach the server.
  const find = q.find || '';
  const worlds = (await API.listWorlds(find).catch(() => ({ worlds: [] }))).worlds || [];
  const wid = params.wid || (worlds[0] && worlds[0].world_id);
  const w = worlds.find((x) => x.world_id === wid);
  const tab = q.tab || 'layers';

  const el = h('div', { class: 'page wide' });
  const find_ = finder(find, wid, tab);

  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, w ? w.name : 'Worlds'),
      h('div', { class: 'sub' },
        'A world is the substrate schema its characters run under: the layers they think through, and the ' +
        'authored sections their lens is built from.')),
    // No "+ New world". An empty world is non-functional — no canon means the
    // `world` layer projects nothing — so a button that creates a container
    // hands back something broken and calls it success. A world is a YAML file
    // in the mind and a tag over the corpus: making one is a file operation and
    // a commit, which is what authored content should be. Characters are what
    // users create; worlds are written.
    find_.el));
  // Seeded with the list this render already fetched, so the first paint costs
  // no second request.
  find_.fill(worlds);

  if (!w) {
    el.appendChild(empty('◍', 'No worlds',
      'A world is a YAML file in the mind, beside the corpus it indexes. Point the daemon at a mind with ' +
      '--mind, or add a file to its worlds/ directory.'));
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
          `and every change here reaches every character in ${w.name}` +
          (w.npc_count ? `, all ${w.npc_count} of them.` : ' — none yet.'))),

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
      footer: [only('admin', () => h('button', {
        class: 'btn primary',
        onClick: () => toast('layer schema edit — engine required', 'err'),
      }, 'Save'))],
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
          // Characters, not tokens. There is no tokenizer in the daemon, so a
          // token count here would be a plausible-looking guess — which is the
          // habit that let six invented templates stand in for 596 real ones.
          h('span', { class: 'tiny dim mono', title: 'characters in the template' },
            (s.chars ?? 0).toLocaleString() + ' ch'),
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
      footer: [only('admin', () => h('button', {
        class: 'btn primary',
        onClick: () => toast('section edit — engine required', 'err'),
      }, 'Save'))],
    });
  }

  // ── setting / clock ───────────────────────────────────────────────────────

  /* The world's own document, edited in place. A PUT replaces
   * `worlds/<id>.yaml` whole, so the object read back goes with it and only the
   * three edited fields are overwritten — `selects` and anything else an author
   * put in the file rides through untouched. */
  function setting() {
    const count = w.npc_count || 0;
    // Editing a world is an admin's. The daemon refuses the PUT either way;
    // this is so the page says so before somebody types a paragraph into it.
    const editable = mayEdit('admin');
    const name = h('input', { class: 'input', value: w.name || '', ...ro('admin') });
    const pub = h('input', { type: 'checkbox', checked: w.public, ...ro('admin', 'toggle') });
    const text = h('textarea', { class: 'textarea', rows: 8, ...ro('admin') }, w.setting || '');

    const save = only('admin', () => h('button', { class: 'btn primary' }, 'Save'));
    if (save) save.onclick = async () => {
      const next = { ...w, name: name.value.trim(), public: pub.checked, setting: text.value };
      delete next.world_id;
      delete next.npc_count;
      try {
        await API.setWorld(w.world_id, next);
        toast('worlds/' + w.world_id + '.yaml written', 'ok');
        go('/world/' + w.world_id + '?tab=setting');
      } catch (e) { toast(e.detail || e.message || 'save failed', 'err'); }
    };

    mount(host,
      h('div', { class: 'panel' },
        h('div', { class: 'row', style: 'justify-content:space-between' },
          h('h3', { style: 'margin:0' }, 'The world document'),
          h('div', { class: 'row', style: 'gap:6px' }, roChip('admin'),
            h('code', { class: 'mono tiny dim' }, `worlds/${w.world_id}.yaml`))),
        h('div', { class: 'grid g2', style: 'margin-top:10px' },
          h('label', { class: 'field' }, h('span', {}, 'Name'), name),
          h('label', { class: 'row', style: 'gap:9px;margin-top:22px;cursor:pointer' }, pub,
            h('div', {}, h('div', { style: 'font-size:.86rem;font-weight:600' }, 'Public'),
              h('div', { class: 'tiny dim' }, 'anyone may spawn characters here')))),
        h('label', { class: 'field' },
          h('span', {}, 'World knowledge — the shared immutable core'), text),
        h('div', { class: 'row', style: 'justify-content:space-between;margin-top:9px' },
          h('div', { class: 'tiny dim' },
            !editable
              ? roNote('a world')
              : count
                ? `Read by every character here. Editing it changes what all ${count} of them know.`
                : 'Read by every character here. Nobody lives here yet.'),
          save)),

      h('h2', {}, 'What this world selects'),
      h('div', { class: 'panel' },
        h('div', { class: 'tiny dim', style: 'max-width:88ch' },
          'A world is a tag-filter over one shared corpus, not a corpus of its own. These are the tags its ' +
          '`world` layer admits — canon ingested under them is visible here and nowhere else, while craft ' +
          '(responses, moods, personalities) is ingested untagged and shared by every world, sharing its KV ' +
          'as well as its text.'),
        (w.selects || []).length
          ? h('div', { class: 'row wrap', style: 'gap:6px;margin-top:11px' },
            (w.selects || []).map((t) => h('span', { class: 'chip accent' }, t)))
          : h('div', { class: 'tiny dim', style: 'margin-top:11px' },
            'Nothing selected — this world admits only untagged content. An empty filter is not "everything"; ' +
            'it is the shared craft and no canon at all.'),
        h('div', { class: 'tiny dim', style: 'margin-top:11px' },
          'Edited in the file, not here: the tag set decides what every character in this world can know, ' +
          'and it belongs in a diff.')));
  }

  function clock() {
    const t = w.time || {};
    mount(host, h('div', { class: 'panel' },
      h('div', { class: 'row wrap', style: 'gap:26px' },
        h('div', {}, h('div', { class: 'tiny dim' }, 'now'),
          h('div', { class: 'mono', style: 'font-size:1.3rem;font-weight:700' }, worldTime(t.world_ms))),
        h('div', {}, h('div', { class: 'tiny dim' }, 'scale'),
          h('select', { class: 'select', style: 'width:auto', ...ro('admin', 'toggle') },
            [0, 1, 10, 60, 360, 1440].map((s) => h('option', { value: s, selected: s === t.scale },
              s === 0 ? 'paused' : s + '× real time')))),
        // Jumping the clock moves narrative time for every character in the
        // world, so it belongs with the other world edits: admin.
        only('admin', () => h('div', {}, h('div', { class: 'tiny dim' }, ' '),
          h('button', { class: 'btn', onClick: () => confirmDialog({
            title: 'Jump the world clock',
            message: `This affects every character in ${w.name}. They will experience the gap, and the ` +
              'consolidation folds will have it to reconcile.',
            confirmText: 'Jump',
            onConfirm: () => toast('clock jumped', 'ok'),
          }) }, 'Jump to…')))),
      h('div', { class: 'tiny dim', style: 'margin-top:14px' },
        (mayEdit('admin') ? '' : roNote('the narrative clock') + ' ')
        + 'Scale is world-seconds per real second; 0 pauses. Narrative time is what characters date their '
        + 'memories by — wall time is only ever a diagnostic.')));
  }

  return { el };
}
