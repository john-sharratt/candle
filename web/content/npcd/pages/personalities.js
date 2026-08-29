/* Personalities (§38).
 *
 * A personality is what a character IS before it has lived anything: an anchor
 * and a set of constant traits, plus the doctrine that steers how it acts. It
 * was called an "archetype" — the same object under a name that described its
 * role in a type system rather than the thing itself.
 *
 * On disk it is ONE file, `personalities/<id>.yaml` in the mind, and this page
 * renders that document rather than a description of it. There was a "section
 * collections" panel here listing `identity_anchor`, `identity` and `doctrine`
 * as three folders of templates; after the restructure those are three fields
 * of the same file, and a panel describing them as collections was narrating a
 * structure that had stopped existing.
 *
 * Anchor and traits are read-only on purpose: they are the shared CoW prefix,
 * and an operator who can edit identity in a text box will come to believe
 * identity is editable, which is the invariant the prefix exists to protect.
 * Doctrine is the one part designed to change — it reaches every living
 * character of this type at their next spawn, which is why it carries a version
 * and the anchor does not.
 *
 * There is no "+ New personality" button, for the same reason the worlds page
 * has none: these are files an author writes, and a button that creates an
 * empty one in a directory the author is not looking at makes the console and
 * the mind disagree about what exists. */

import { API } from '../lib/api.js';
import { h, mount } from '../lib/dom.js';
import { go } from '../lib/router.js';
import { empty, confirmDialog, toast, mayEdit, ro, roChip, only, roNote } from '../lib/ui.js';

/* A trait key is `under_pressure` in the file and "Under pressure" on the page.
 * The file is what an author edits, so the file's spelling wins there. */
const label = (k) => k.replace(/[_-]+/g, ' ').replace(/^./, (c) => c.toUpperCase());

/* Most personalities in a mind carry no `name:` — the slug is the whole of
 * their identity. Titling it beats printing `anchor-the-protector` as a page
 * heading, and it is derived rather than stored, so an author who does want a
 * particular spelling adds `name:` and it wins. */
const title = (id) => String(id || '').split('-').map(label).join(' ');
const named = (a) => a.name || title(a.personality_id);

export async function render(params, q) {
  const list = (await API.listPersonalities().catch(() => ({ personalities: [] }))).personalities || [];
  const aid = params.aid || q.a || (list[0] && list[0].personality_id);
  const a = list.find((x) => x.personality_id === aid);

  const el = h('div', { class: 'page wide' });

  el.appendChild(h('div', { class: 'hd' },
    h('div', {},
      h('h1', {}, a ? named(a) : 'Personalities'),
      h('div', { class: 'sub' },
        'What a character is before it has lived anything. Every character of this type shares it as a ' +
        'read-only prefix, so it costs one copy however many of them exist — which is also why it cannot drift.')),
    list.length > 1 ? picker(list, aid) : null));

  if (!a) {
    el.appendChild(empty('◈', 'No personalities',
      'Personalities are YAML files in the mind. Point the daemon at one with --mind, or add a file to ' +
      'its personalities/ directory.'));
    return { el };
  }

  const count = a.npc_count || 0;
  const traits = (a.personality && typeof a.personality === 'object') ? Object.entries(a.personality) : [];

  el.appendChild(h('div', { class: 'panel' },
    h('div', { class: 'row', style: 'gap:8px' },
      h('h3', { style: 'margin:0' }, 'Anchor'),
      h('span', { class: 'chip' }, 'always resident'),
      h('span', { style: 'flex:1' }),
      h('code', { class: 'mono tiny dim' }, `personalities/${a.personality_id}.yaml`)),
    h('pre', {
      class: 'mono',
      style: 'margin:9px 0 0;padding:12px 14px;background:var(--bg-deep);border-radius:8px;' +
        'font-size:.82rem;color:var(--ink-mid);white-space:pre-wrap;overflow-x:auto',
    }, (a.anchor || '').trim() || '— this file has no anchor —'),
    h('div', { class: 'tiny dim', style: 'margin-top:8px;max-width:88ch' },
      'The floor the whole substrate is read through. It never competes for the gather budget, because it ' +
      'is the prefix the budget is measured inside. ' +
      (count ? `Shared by ${count} living character${count === 1 ? '' : 's'}.` : 'Nothing has been made from it yet.'))));

  el.appendChild(h('div', { class: 'panel' },
    h('div', { class: 'row', style: 'gap:8px' },
      h('h3', { style: 'margin:0' }, 'Traits'),
      h('span', { class: 'chip' }, 'always resident'),
      h('span', { style: 'flex:1' }),
      h('span', { class: 'tiny dim' }, traits.length + ' constant' + (traits.length === 1 ? '' : 's'))),
    h('div', { class: 'tiny dim', style: 'margin:7px 0 0;max-width:88ch' },
      'Constant properties of the same self, resident alongside the anchor rather than selected. Choosing ' +
      'three per turn would make a character partly itself, differently each turn. Biography — the part ' +
      'that genuinely is situational — lives in the memory layer, where provenance retrieves it.'),
    traits.length
      ? h('div', { class: 'list', style: 'margin-top:10px' },
        traits.map(([k, v]) => h('div', { style: 'padding:11px 15px' },
          h('div', { class: 'row', style: 'gap:9px' },
            // The key as the file spells it. A prettified label beside it said
            // the same thing twice, and the file's spelling is the one an
            // author has to type.
            h('code', { class: 'mono', style: 'color:var(--accent);font-size:.8rem' }, k)),
          h('div', {
            class: 'tiny',
            style: 'color:var(--ink-soft);margin-top:5px;max-width:92ch;white-space:pre-wrap',
          }, String(v).trim()))))
      : h('div', { class: 'tiny dim', style: 'margin-top:10px' },
        'None declared. The anchor carries this character on its own.')));

  el.appendChild(doctrinePanel(a, count));
  return { el };
}

/* A search box and a select, not a select alone.
 *
 * A mind holds every character it has ever described — this one holds 74 — and
 * a 74-option dropdown is a list you scroll past what you wanted.
 *
 * The filter goes to the SERVER rather than narrowing the list already in hand.
 * That looks like the long way round for 74 rows, and it is the only way that
 * works: a personality with `hidden: true` is never sent, so a client-side
 * filter would have nothing to find however completely you typed its name.
 * Filtering where the hiding happens is what makes the two agree.
 *
 * It filters as you type, and the whole-word rule is what keeps that safe — the
 * daemon answers nothing to `pro`, `prot` or `protec`, so watching the list
 * while typing reveals nothing that was hidden. Only a complete word does.
 *
 * Typing never navigates: `go()` would rebuild the page and take the caret with
 * it, so results refill the select in place and only a selection moves. */
function picker(list, aid) {
  const sel = h('select', {
    class: 'select', style: 'width:auto;max-width:280px',
    onChange: (e) => go('/personalities?a=' + e.target.value),
  });

  const fill = (rows) => {
    mount(sel, rows.map((x) => h('option', {
      value: x.personality_id, selected: x.personality_id === aid,
    }, `${named(x)} · ${x.npc_count || 0}`)));
    sel.hidden = rows.length < 2;
  };
  fill(list);

  let timer = null;
  let seq = 0;
  const search = async (v) => {
    // A slower request must not overwrite a newer one's answer.
    const mine = ++seq;
    const r = await API.listPersonalities(v).catch(() => null);
    if (r && mine === seq) fill(r.personalities || []);
  };

  const box = h('input', {
    class: 'input', style: 'width:150px', placeholder: `filter ${list.length}…`,
    onInput: () => {
      clearTimeout(timer);
      timer = setTimeout(() => search(box.value.trim()), 180);
    },
  });

  return h('div', { class: 'row' }, box, sel);
}

/* The one editable part. A publish is a real write: PUT replaces the document,
 * so the whole object goes back with the doctrine and its version advanced —
 * sending only the two changed fields would blank the anchor and every trait. */
function doctrinePanel(a, count) {
  const version = Number(a.doctrine_version || 0);
  // Read-only for anyone but an admin. Presentation, not the check — the daemon
  // refuses the PUT regardless — but a box you can type into and a button that
  // then 403s teaches its user the product is broken, when the truth is only
  // that this is not their job.
  const editable = mayEdit('admin');
  const box = h('textarea', {
    class: 'textarea', rows: 4, style: 'margin-top:8px', ...ro('admin'),
  }, (a.doctrine || '').trim());
  const status = h('div', { class: 'tiny dim' },
    !editable ? roNote('doctrine')
      : count ? `reaches ${count} living character${count === 1 ? '' : 's'} at next spawn`
        : 'nothing is living under it yet');

  const publish = only('admin', () => h('button', { class: 'btn sm' }, 'Publish'));
  if (publish) publish.onclick = () => confirmDialog({
    title: 'Publish doctrine v' + (version + 1),
    message: `This reaches every ${named(a)} in every world at their next spawn or fork ` +
      'refresh. Lived experience is untouched — only how they act changes. The file on disk is rewritten.',
    confirmText: 'Publish',
    onConfirm: async () => {
      const next = { ...a, doctrine: box.value.trim(), doctrine_version: version + 1 };
      // The id is the URL, not the body — the daemon discards a body's own id
      // rather than let a document name its own file.
      delete next.personality_id;
      delete next.npc_count;
      try {
        await API.setPersonality(a.personality_id, next);
        a.doctrine_version = version + 1;
        a.doctrine = next.doctrine;
        toast('doctrine v' + (version + 1) + ' written', 'ok');
        go('/personalities?a=' + a.personality_id);
      } catch (e) {
        toast(e.detail || e.message || 'publish failed', 'err');
      }
    },
  });

  return h('div', { class: 'panel' },
    h('div', { class: 'row', style: 'justify-content:space-between' },
      h('h3', { style: 'margin:0' }, 'Doctrine'),
      h('div', { class: 'row', style: 'gap:6px' }, roChip('admin'),
        h('span', { class: 'chip accent' }, 'v' + version))),
    h('div', { class: 'tiny dim', style: 'margin-top:6px;max-width:88ch' },
      'How this character acts, as opposed to what it is. The one part of the shared prefix designed to ' +
      'change, which is why it carries a version and the anchor does not.'),
    box,
    h('div', { class: 'row', style: 'justify-content:space-between;margin-top:9px' }, status, publish));
}
