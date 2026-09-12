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
import { empty, toast, confirmDialog, layerColor, bar, mayEdit, ro, roChip, only, roNote } from '../lib/ui.js';

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
  // Opens on the sections, which is what a world is mostly looked at for.
  const tab = q.tab || 'collections';

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

  /* Ordered by how often somebody comes here to do it.
   *
   * `layers` used to be first and was also the default, so the page opened on
   * the one thing nobody edits: layer geometry is calibration data, re-derived
   * by measurement rather than typed into a form. It is now last, and reached
   * through a quieter control below rather than sitting level with the three
   * things people actually came for. */
  const TABS = [
    ['collections', 'Sections'],
    ['setting', 'Setting & knowledge'],
    ['clock', 'Narrative clock'],
  ];
  el.appendChild(h('div', { class: 'row', style: 'gap:4px;margin-bottom:18px;flex-wrap:wrap' },
    TABS.map(([k, label]) => h('button', {
      class: 'btn' + (tab === k ? ' primary' : ' ghost'),
      onClick: () => go(`/world/${wid}?tab=${k}`),
    }, label)),
    h('span', { class: 'spacer', style: 'flex:1' }),
    /* The corpus behind all of it, as one editable tree.
     *
     * Not in `TABS`, because it leaves this page: the tabs are views of one
     * world's settings, and this is the whole mind with that world held over it
     * as a lens. Carrying `wid` opens it already filtered to what this world
     * admits. */
    wid
      ? h('button', {
        class: 'btn sm ghost',
        title: 'Browse and edit everything this world draws on',
        onClick: () => go(`/mind?world=${wid}`),
      }, 'Edit the corpus →')
      : null,
    /* Rarely wanted, so a quiet link rather than a peer of the tabs — but it
     * reaches a real editor now, not a reading of a fixture. */
    h('button', {
      class: 'btn sm ghost' + (tab === 'layers' ? ' primary' : ''),
      title: 'The nine layers this mind projects through',
      onClick: () => go(`/world/${wid}?tab=layers`),
    }, 'Layer geometry')));

  const host = h('div', {});
  el.appendChild(host);

  // An unknown tab lands on the sections, the same place a bare URL does.
  ({ layers, collections, setting, clock }[tab] || collections)();

  // ── substrate layers ──────────────────────────────────────────────────────

  /* The nine layers, read from the projection schema itself.
   *
   * From the mind, not from a `/v1/schema/layers` of its own: the schema is an
   * authored document like the rest of the corpus, and a second endpoint
   * serving the same nine layers is a second set of numbers to drift. It did —
   * the fixture this page used to read had `action` at priority 95 while the
   * schema said 100, and nothing could have noticed.
   *
   * Each layer is a *part* of `settings/projection` (`npcd/src/mind/parts.rs`),
   * so it lists and opens like any other entry. Which is also why the Edit
   * button below is real. */
  async function layers() {
    mount(host, h('div', { class: 'tiny dim' }, 'reading the schema…'));
    let found = [];
    try {
      const listing = await API.mindList('settings/projection');
      found = await Promise.all((listing.children || []).map(async (c) => ({
        id: c.id,
        name: c.title,
        fields: (await API.mindFields(c.id)).fields || [],
      })));
    } catch (e) {
      mount(host, h('div', { class: 'panel' },
        empty('◌', 'No projection schema',
          e.detail || e.message || 'this mind declares no layers')));
      return;
    }

    /* A value out of a field list, following nested groups: `at(f, 'budget',
     * 'adaptive', 'gain')`. The form and this view read the same shape, so
     * there is nowhere for the two to disagree about what a layer says. */
    const at = (fields, ...path) => {
      let list = fields;
      let value;
      for (const key of path) {
        const f = (list || []).find((x) => x.key === key);
        if (!f) return undefined;
        value = f.value;
        list = f.fields;
      }
      return value;
    };
    const rowsOf = (fields, key) =>
      ((fields || []).find((f) => f.key === key) || {}).rows || [];

    const maxWindow = Math.max(...found.map((l) => at(l.fields, 'window') || 0), 1);

    mount(host,
      h('div', { class: 'panel', style: 'margin-bottom:12px' },
        h('div', { class: 'tiny dim' },
          'Layer geometry is the calibration surface: window, budget, score threshold, and the '
          + 'selection rule. It is data rather than code precisely so it can be moved without a '
          + `rebuild — and every change here reaches every character in ${w.name}`
          + (w.npc_count ? `, all ${w.npc_count} of them.` : ' — none yet.'))),

      h('div', { class: 'list' }, found.map((l) => {
        const window = at(l.fields, 'window');
        const shared = at(l.fields, 'gather_scope') === 'shared';
        const summarised = (l.fields || []).some((f) => f.key === 'summary');
        return h('div', { style: 'padding:14px 18px' },
          h('div', { class: 'row', style: 'gap:10px;margin-bottom:8px' },
            h('span', { style: `width:3px;height:16px;border-radius:2px;background:${layerColor(l.name.toLowerCase())}` }),
            h('strong', { style: 'font-size:.9rem' }, l.name),
            shared
              ? h('span', { class: 'chip warn', title: 'one tree across every conversation' }, 'cross-timeline')
              : h('span', { class: 'chip' }, 'self-local'),
            summarised ? h('span', { class: 'chip' }, 'summarised') : null,
            h('span', { style: 'flex:1' }),
            h('span', { class: 'chip accent' }, (at(l.fields, 'decode_priority') || '—') + ' priority'),
            h('button', {
              class: 'btn sm ghost',
              title: 'Edit this layer',
              onClick: () => go(`/mind?id=${encodeURIComponent(l.id)}`),
            }, 'Edit')),

          h('div', { class: 'tiny dim', style: 'margin-bottom:10px;max-width:74ch;white-space:pre-line' },
            (at(l.fields, 'description') || '').trim()),

          h('div', { class: 'grid g4' },
            field('window', fmtK(window), bar((window || 0) / maxWindow, layerColor(l.name.toLowerCase()))),
            field('budget priority', String(at(l.fields, 'budget', 'priority') ?? '—')),
            field('budget ceiling', (at(l.fields, 'budget', 'max_percent')
              ?? at(l.fields, 'budget', 'min_percent') ?? 0) + '%'),
            field('score threshold', String(at(l.fields, 'score_threshold') ?? 0))),

          // The selection groups, as the document states them rather than as a
          // pre-rendered sentence somebody has to keep in step with it.
          h('div', { class: 'row', style: 'gap:9px;margin-top:10px;flex-wrap:wrap' },
            h('span', { class: 'tiny dim' }, 'selection'),
            rowsOf(l.fields, 'groups').map((row) => {
              const cell = (k) => (row.find((f) => f.key === k) || {}).value;
              const sel = (row.find((f) => f.key === 'selection') || {}).fields || [];
              const part = (k) => (sel.find((f) => f.key === k) || {}).value;
              const rule = part('kind') === 'top_k'
                ? `top-k ${part('k')}`
                : [part('recent') != null ? `recent ${part('recent')}` : null,
                  part('history_top_k') != null ? `top-k ${part('history_top_k')}` : null]
                  .filter(Boolean).join(', ') || part('kind') || '';
              return h('code', { class: 'mono tiny', style: 'color:var(--ink-dim)' },
                `${cell('id')}(${rule})`);
            })));
      })));
  }

  function field(label, value, extra) {
    return h('div', {},
      h('div', { class: 'tiny dim' }, label),
      h('div', { class: 'mono', style: 'font-size:.9rem;font-weight:700' }, value),
      extra || null);
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

  /* A section's address in the corpus.
   *
   * `col.folder` is `responses/` or `moods/`, which is the section slug plus a
   * separator — the one place this page touches the corpus's vocabulary, and it
   * is a slice rather than a mapping table so it cannot drift. */
  const addressOf = (col, s) => `${col.folder.replace(/\/$/, '')}/${s.id}`;

  /* `blush_then_own` → `Blush then own`. Display only; the id is untouched. */
  const readable = (id) => {
    const s = String(id).replace(/[_-]/g, ' ');
    return s.charAt(0).toUpperCase() + s.slice(1);
  };

  function collectionCard(col) {
    const body = h('div', { hidden: true });
    const n = col.sections.length;
    const caret = h('span', { class: 'caret' }, '▸');
    const label = h('span', {}, `Show all ${n} ${col.name} sections`);
    const toggle = h('button', {
      class: 'sec-open',
      onClick: () => {
        body.hidden = !body.hidden;
        caret.textContent = body.hidden ? '▸' : '▾';
        label.textContent = body.hidden
          ? `Show all ${n} ${col.name} sections`
          : `Hide the ${n} ${col.name} sections`;
        // Built on first open. Six hundred tiles is not work to do for a card
        // nobody expanded.
        if (!body.hidden && !body.dataset.built) {
          body.dataset.built = '1';
          mount(body, h('div', { class: 'sec-tiles', style: 'margin-top:11px' },
            col.sections.map((s) => tile(col, s))));
        }
      },
    }, caret, label, h('span', { style: 'flex:1' }),
      h('span', { class: 'tiny dim' }, 'each opens for editing'));

    return h('div', { class: 'panel' },
      h('div', { class: 'row', style: 'gap:9px;flex-wrap:wrap' },
        h('strong', { style: 'font-size:.95rem' }, readable(col.name) + ' sections'),
        h('span', { class: 'chip' }, col.rule),
        h('span', { style: 'flex:1' }),
        h('span', { class: 'tiny dim' }, col.source)),
      h('div', { class: 'tiny dim', style: 'margin-top:7px;max-width:80ch' }, col.description),
      toggle,
      body);
  }

  /* One section, as a tile.
   *
   * The whole tile is the control. An `Edit` button in the corner was the old
   * shape and it was both hard to find and never shown — collections are marked
   * `locked`, so the button was never rendered at all and the editor behind it
   * was unreachable code whose Save only ever raised an error.
   *
   * `locked` was right about one thing and wrong about another: these files are
   * shared by every world, so editing them *from a world* would be editing
   * every other world's copy from a page that names only this one. But they are
   * not immutable, and the place they are edited now exists — so the tile goes
   * there, carrying the world so the corpus opens through the same lens. */
  function tile(col, s) {
    const id = addressOf(col, s);
    return h('button', {
      class: 'sec-tile',
      title: 'Open ' + id,
      onClick: () => go(`/mind?id=${encodeURIComponent(id)}&world=${wid}`),
    },
      h('div', { class: 'nm' }, readable(s.id)),
      h('div', { class: 'why' }, s.description || s.template),
      h('div', { class: 'row', style: 'gap:6px;flex-wrap:wrap;align-items:center' },
        h('span', { class: 'chip' }, s.category),
        h('span', { style: 'flex:1' }),
        // Characters, not tokens. There is no tokenizer in the daemon, so a
        // token count here would be a plausible-looking guess — which is the
        // habit that let six invented templates stand in for 596 real ones.
        h('span', { class: 'tiny dim mono', title: 'characters in the template' },
          (s.chars ?? 0).toLocaleString() + ' ch'),
        s.examples
          ? h('span', {
            class: 'chip ok',
            title: 'authored lead-ins that train this section’s selection — editable with it',
          }, s.examples + ' examples')
          : h('span', { class: 'chip warn', title: 'nothing trains this section’s selection' },
            'uncalibrated')));
  }

  // ── setting / clock ───────────────────────────────────────────────────────

  /* The world's own document, edited in place. A PUT replaces
   * `worlds/<id>.yaml` whole, so the object read back goes with it and only the
   * three edited fields are overwritten — `selects` and anything else an author
   * put in the file rides through untouched. */
  async function setting() {
    const count = w.npc_count || 0;
    // Editing a world is an admin's. The daemon refuses the PUT either way;
    // this is so the page says so before somebody types a paragraph into it.
    const editable = mayEdit('admin');
    const name = h('input', { class: 'input', value: w.name || '', ...ro('admin') });
    const pub = h('input', { type: 'checkbox', checked: w.public, ...ro('admin', 'toggle') });
    const text = h('textarea', { class: 'textarea', rows: 8, ...ro('admin') }, w.setting || '');

    // Filled in below, after the corpus answers. Mounted empty first so the
    // world document is editable immediately rather than waiting on a listing.
    const knowledge = h('div', { class: 'panel' },
      h('div', { class: 'tiny dim' }, 'reading the corpus…'));

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
        /* "Setting", not "World knowledge".
         *
         * It was the second, which named the same thing as the 1,267-page canon
         * corpus below — so this eight-line box looked like the place the
         * world's knowledge lived, and the history and the technology tree
         * looked like they were missing. They were not; they were one label
         * away. */
        h('label', { class: 'field' },
          h('span', {}, 'Setting — the paragraph every character here opens with'), text),
        h('div', { class: 'row', style: 'justify-content:space-between;margin-top:9px' },
          h('div', { class: 'tiny dim' },
            !editable
              ? roNote('a world')
              : count
                ? `Read by every character here. Editing it changes what all ${count} of them know.`
                : 'Read by every character here. Nobody lives here yet.'),
          save)),

      h('h2', {}, 'What this world knows'),
      knowledge);

    /* The knowledge itself, and the way in to editing it.
     *
     * This panel used to be a row of grey chips — the raw `selects` tags, with
     * a note saying they are edited in the file. True, and useless: it named
     * `history` and `technology` without saying that each is a folder of pages,
     * how many, or where to change one. The corpus was reachable only through a
     * generic "Edit the corpus" button that gave no reason to think the game's
     * history was behind it.
     *
     * So the tags are asked of the daemon instead, which answers with the
     * topics this world actually admits — its own filter applied — and each one
     * is a link into the editor with the world already held over it. */
    const admitted = await API.getWorldKnowledge(w.world_id).catch(() => null);
    mount(knowledge,
      h('div', { class: 'tiny dim', style: 'max-width:88ch' },
        'A world is a tag-filter over one shared corpus, not a corpus of its own. These are the topics its '
        + '`world` layer admits — canon tagged under them is visible here and nowhere else, while craft '
        + '(responses, moods, personalities) is ingested untagged and shared by every world, sharing its KV '
        + 'as well as its text.'),
      !admitted
        ? h('div', { class: 'tiny dim', style: 'margin-top:11px' }, 'could not read the corpus')
        : !admitted.length
          ? h('div', { class: 'tiny dim', style: 'margin-top:11px' },
            'Nothing selected — this world admits only untagged content. An empty filter is not "everything"; '
            + 'it is the shared craft and no canon at all.')
          : h('div', { class: 'topic-grid' }, admitted.map((t) => h('button', {
            class: 'topic',
            title: `Edit ${t.title}`,
            onClick: () => go(`/mind?id=${encodeURIComponent(t.id)}&world=${w.world_id}`),
          },
            h('div', { class: 'nm' }, t.title),
            h('div', { class: 'ct' },
              t.count ? `${t.count} page${t.count === 1 ? '' : 's'}` : 'one page')))),
      h('div', { class: 'tiny dim', style: 'margin-top:12px' },
        'Which topics are admitted is edited in the world file, not here: the tag set decides what every '
        + 'character in this world can know, and it belongs in a diff. What is written *inside* each topic '
        + 'is edited by opening it.'));
  }

  /* The narrative clock, which now writes.
   *
   * Both controls here used to be scenery: the `<select>` had no `onChange` at
   * all, and Jump ended in `toast('clock jumped', 'ok')` under a dialog warning
   * that it affected every character in the world. It claimed a write it never
   * made, which is the worst thing a control can do. */
  function clock() {
    const t = w.time || {};
    const now = h('div', { class: 'mono', style: 'font-size:1.3rem;font-weight:700' },
      worldTime(t.world_ms));

    // Re-read after every write, because the answer is computed from an anchor
    // and the daemon is the only thing that knows what time it is there.
    const settle = async (body, said) => {
      try {
        const next = await API.setWorldTime(w.world_id, body);
        w.time = next;
        mount(now, worldTime(next.world_ms));
        toast(said, 'ok');
      } catch (e) {
        toast(e.detail || e.message || 'the clock did not move', 'err');
      }
    };

    const scale = h('select', { class: 'select', style: 'width:auto', ...ro('admin', 'toggle'),
      onChange: (e) => {
        const s = Number(e.target.value);
        // `0` is how the console spells paused, and the daemon keeps the pace
        // a paused world was running at — so this sends both.
        settle(s === 0 ? { paused: true } : { scale: s, paused: false },
          s === 0 ? 'clock paused' : `running at ${s}× real time`);
      } },
      [0, 1, 10, 60, 360, 1440].map((s) => h('option', {
        value: s,
        selected: t.paused ? s === 0 : s === t.scale,
      }, s === 0 ? 'paused' : s + '× real time')));

    mount(host, h('div', { class: 'panel' },
      h('div', { class: 'row wrap', style: 'gap:26px' },
        h('div', {}, h('div', { class: 'tiny dim' }, 'now'), now),
        h('div', {}, h('div', { class: 'tiny dim' }, 'scale'), scale),
        // Jumping the clock moves narrative time for every character in the
        // world, so it belongs with the other world edits: admin.
        only('admin', () => {
          const to = h('input', { class: 'input', type: 'datetime-local', style: 'width:auto' });
          return h('div', {}, h('div', { class: 'tiny dim' }, 'jump to'),
            h('div', { class: 'row', style: 'gap:6px' }, to,
              h('button', { class: 'btn', onClick: () => {
                const at = Date.parse(to.value);
                if (!Number.isFinite(at)) return toast('pick a date and time first', 'err');
                confirmDialog({
                  title: 'Jump the world clock',
                  message: `This affects every character in ${w.name}. They will experience the gap, and the `
                    + 'consolidation folds will have it to reconcile.',
                  confirmText: 'Jump',
                  onConfirm: () => settle({ world_ms: at }, 'clock jumped'),
                });
              } }, 'Jump')));
        })),
      h('div', { class: 'tiny dim', style: 'margin-top:14px' },
        (mayEdit('admin') ? '' : roNote('the narrative clock') + ' ')
        + 'Scale is world-seconds per real second; 0 pauses. Narrative time is what characters date their '
        + 'memories by — wall time is only ever a diagnostic.')));
  }

  return { el };
}
