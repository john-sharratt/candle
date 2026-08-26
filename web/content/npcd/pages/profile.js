/* Profile (§12) — the self an NPC reads.
 *
 * Editing appends a new profile turn and tombstones the previous one; it never
 * rewrites. An NPC's memory of who you were when you spoke stays true. */

import { API } from '../lib/api.js';
import { h, mount } from '../lib/dom.js';
import { toast, empty } from '../lib/ui.js';
import { go } from '../lib/router.js';

export async function render() {
  const me = await API.getMe().catch(() => null);
  const el = h('div', { class: 'page', style: 'max-width:820px' });
  if (!me) { el.appendChild(empty('◌', 'Not signed in')); return { el }; }

  const p = me.profile || {};
  const f = {};
  const field = (key, label, hint, rows) => {
    const ctl = rows
      ? h('textarea', { class: 'textarea', rows }, p[key] || '')
      : h('input', { class: 'input', value: p[key] || '' });
    f[key] = ctl;
    return h('label', { class: 'field' }, h('span', {}, label), ctl,
      hint ? h('div', { class: 'tiny dim', style: 'margin-top:4px' }, hint) : null);
  };

  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, 'Your profile'),
      h('div', { class: 'sub' }, 'This is what characters read about you. It lives in the substrate like anything else.')),
    h('div', { class: 'row' },
      h('span', { class: 'chip' }, 'revision ' + (p.revision ?? 1)),
      h('button', {
        class: 'btn sm',
        onClick: async () => { await API.logout(); try { localStorage.setItem('npcd.signedout', '1'); } catch (_) {} location.hash = '#/welcome'; location.reload(); },
      }, 'Sign out'))));

  el.appendChild(h('div', { class: 'panel' },
    h('div', { class: 'grid g2' },
      h('label', { class: 'field' }, h('span', {}, 'Unique name'),
        h('input', { class: 'input', value: me.unique_name || '', id: 'uname' }),
        h('div', { class: 'tiny dim', style: 'margin-top:4px' },
          'The only identifier a character ever sees. Tools take it as a target — when an NPC sends you an image, it sends it to this name.')),
      h('div', {},
        h('label', { class: 'field' }, h('span', {}, 'Account'),
          h('input', { class: 'input', value: me.email || '', disabled: true })),
        h('div', { class: 'tiny dim' }, 'From ' + (me.provider || 'your provider') + '. Never shown to a character.'))),

    field('pronouns', 'Pronouns'),
    field('gender', 'Gender'),
    field('description', 'Description — how a character perceives you', null, 4),
    field('history', 'History', 'Background a character can come to know about you over time.', 3),

    h('div', { class: 'row', style: 'justify-content:space-between;margin-top:6px' },
      h('div', { class: 'tiny dim', style: 'max-width:560px' },
        'Saving appends a new profile turn and tombstones the previous one — it never rewrites. ' +
        'A character’s memory of who you were during past conversations stays true.'),
      h('button', {
        class: 'btn primary',
        onClick: async () => {
          const body = Object.fromEntries(Object.entries(f).map(([k, ctl]) => [k, ctl.value]));
          await API.putProfile(body);
          toast('profile updated — previous revision tombstoned', 'ok');
        },
      }, 'Save'))));

  el.appendChild(h('h2', {}, 'Characters'));
  el.appendChild(h('div', { class: 'panel' },
    h('div', { class: 'row', style: 'justify-content:space-between' },
      h('div', {}, h('div', { class: 'mono', style: 'font-size:1.3rem;font-weight:700' }, String(me.npc_count ?? 0)),
        h('div', { class: 'tiny dim' }, 'characters you own')),
      h('button', { class: 'btn', onClick: () => go('/') }, 'View roster'))));

  return { el };
}
