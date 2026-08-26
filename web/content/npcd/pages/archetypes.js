/* Archetypes (§38).
 *
 * An archetype is a description AND its own section collections — the identity
 * anchor and its detail facets. Those render read-only on purpose: they are the
 * shared CoW prefix, and an operator who can edit identity in a text box will
 * come to believe identity is editable, which is the invariant the prefix
 * exists to protect. Doctrine is the one part designed to change. */

import { API } from '../lib/api.js';
import { h, mount } from '../lib/dom.js';
import { go } from '../lib/router.js';
import { empty, confirmDialog, toast, modal } from '../lib/ui.js';

export async function render(params, q) {
  const list = (await API.listArchetypes().catch(() => ({ archetypes: [] }))).archetypes || [];
  const aid = params.aid || q.a || (list[0] && list[0].archetype_id);
  const a = list.find((x) => x.archetype_id === aid);

  const el = h('div', { class: 'page wide' });

  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, a ? a.name : 'Archetypes'),
      h('div', { class: 'sub' },
        'A shared read-only prefix per character type: identity, voice and processing rules, plus the section ' +
        'collections they are surfaced from. Identity never changes; doctrine is the one part that evolves.')),
    h('div', { class: 'row' },
      list.length > 1
        ? h('select', { class: 'select', style: 'width:auto', onChange: (e) => go('/archetypes?a=' + e.target.value) },
          list.map((x) => h('option', { value: x.archetype_id, selected: x.archetype_id === aid },
            `${x.name} · ${x.npc_count}`)))
        : null,
      h('button', { class: 'btn primary' }, '+ New archetype'))));

  if (!a) { el.appendChild(empty('◈', 'No archetypes')); return { el }; }

  el.appendChild(h('div', { class: 'grid g2' },
    h('div', { class: 'panel' },
      h('div', { class: 'row', style: 'gap:8px' },
        h('h3', { style: 'margin:0' }, 'Core identity'),
        h('span', { class: 'chip' }, 'immutable')),
      h('div', {
        style: 'margin-top:8px;padding:11px 13px;background:var(--bg-deep);border-radius:8px;' +
          'font-size:.86rem;color:var(--ink-mid)',
      }, a.core_identity),
      h('div', { class: 'tiny dim', style: 'margin-top:8px' },
        `Shared by all ${a.npc_count} characters of this type as a copy-on-write prefix. It cannot drift, ` +
        'because drifting would break the sharing that makes it free.')),

    h('div', { class: 'panel' },
      h('div', { class: 'row', style: 'justify-content:space-between' },
        h('h3', { style: 'margin:0' }, 'Doctrine'),
        h('span', { class: 'chip accent' }, 'v' + a.doctrine_version)),
      h('textarea', { class: 'textarea', rows: 3, style: 'margin-top:8px' }, a.doctrine),
      h('div', { class: 'row', style: 'justify-content:space-between;margin-top:9px' },
        h('div', { class: 'tiny dim' }, `reaches all ${a.npc_count} characters worldwide`),
        h('button', {
          class: 'btn sm',
          onClick: () => confirmDialog({
            title: 'Publish doctrine v' + (a.doctrine_version + 1),
            message: `This reaches every ${a.name} in every world at their next spawn or fork refresh. ` +
              'Lived experience is untouched — only how they act changes.',
            confirmText: 'Publish',
            onConfirm: () => toast('doctrine published', 'ok'),
          }),
        }, 'Publish')))));

  el.appendChild(h('h2', {}, 'Section collections'));
  const colHost = h('div', {});
  el.appendChild(colHost);

  const c = await API.getArchetypeCollections(aid).catch(() => ({ collections: [] }));
  mount(colHost,
    h('div', { class: 'panel', style: 'margin-bottom:12px' },
      h('div', { class: 'tiny dim' },
        'These are the archetype’s own collections — the lens half of the system prompt. The anchor is ' +
        'structurally always-present and never competes for the gather budget; the facets surface only when ' +
        'relevant to the exchange.')),
    (c.collections || []).map(collectionCard));

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
          s.examples ? h('span', { class: 'chip ok' }, s.examples + ' examples') : null,
          col.locked
            ? h('span', { class: 'chip', title: 'read-only by construction' }, 'locked')
            : h('button', { class: 'btn sm ghost', onClick: () => edit(col, s) }, 'Edit')),
        h('div', { class: 'tiny', style: 'color:var(--ink-soft);margin-top:5px;max-width:88ch' }, s.template)))));

    return h('div', { class: 'panel' },
      h('div', { class: 'row', style: 'gap:9px' },
        h('strong', { style: 'font-size:.9rem' }, col.name),
        h('code', { class: 'mono tiny dim' }, col.folder),
        col.locked ? h('span', { class: 'chip' }, 'immutable') : h('span', { class: 'chip accent' }, 'evolves'),
        h('span', { style: 'flex:1' }),
        h('span', { class: 'chip' }, col.rule),
        toggle),
      h('div', { class: 'tiny dim', style: 'margin-top:7px;max-width:80ch' }, col.description),
      body);
  }

  function edit(col, s) {
    modal({
      title: col.name + ' · ' + s.id, wide: true,
      body: h('div', {},
        h('label', { class: 'field' }, h('span', {}, 'Template'),
          h('textarea', { class: 'textarea', rows: 5 }, s.template)),
        h('div', { class: 'tiny dim' }, 'Selected by: ' + col.rule)),
      footer: [h('button', { class: 'btn primary', onClick: () => toast('section edit — engine required', 'err') }, 'Save')],
    });
  }

  return { el };
}
