/* Profile (§12) — the self an NPC reads.
 *
 * Editing appends a new profile turn and tombstones the previous one; it never
 * rewrites. An NPC's memory of who you were when you spoke stays true. */

import { API } from '../lib/api.js';
import { h, mount, ago } from '../lib/dom.js';
import { toast, empty } from '../lib/ui.js';
import { go } from '../lib/router.js';
import { AUTH_UNAVAILABLE, faceOf } from '../app.js';

export async function render() {
  const me = await API.getMe().catch(() => null);

  const el = h('div', { class: 'page', style: 'max-width:820px' });
  if (!me) {
    /* Two dead ends, and only one of them has a way out. */
    el.appendChild(AUTH_UNAVAILABLE
      ? empty('◌', 'Sign-in is not configured',
        'This deployment has no identity provider, so nobody can sign in. That is an operator setting, not ' +
        'something you can fix from here.')
      : empty('◌', 'Not signed in', 'Your profile is what characters read about you.',
        h('button', { class: 'btn primary', onClick: () => window.__npcdSignIn() }, 'Sign in')));
    return { el };
  }

  const p = me.profile || {};
  const uname = h('input', { class: 'input', value: me.unique_name || '' });
  const rev = h('span', { class: 'chip' }, 'revision ' + (p.revision ?? 0));
  const f = {};
  const field = (key, label, hint, rows) => {
    const ctl = rows
      ? h('textarea', { class: 'textarea', rows }, p[key] || '')
      : h('input', { class: 'input', value: p[key] || '' });
    f[key] = ctl;
    return h('label', { class: 'field' }, h('span', {}, label), ctl,
      hint ? h('div', { class: 'tiny dim', style: 'margin-top:4px' }, hint) : null);
  };

  /* A closed set, so a `<select>` rather than a text box. The blank option is
   * what a new account starts on — signing in creates the record before anyone
   * has been asked anything — and it stays available, because a field nobody
   * has filled in should not silently claim the first value in the list. */
  const choice = (key, label, options, hint) => {
    const ctl = h('select', { class: 'select' },
      h('option', { value: '' }, '—'),
      ...options.map((o) => h('option', p[key] === o ? { value: o, selected: true } : { value: o }, o)));
    f[key] = ctl;
    return h('label', { class: 'field' }, h('span', {}, label), ctl,
      hint ? h('div', { class: 'tiny dim', style: 'margin-top:4px' }, hint) : null);
  };

  el.appendChild(h('div', { class: 'hd' },
    h('div', { class: 'row', style: 'gap:14px' },
      faceOf(me, 46),
      h('div', {}, h('h1', {}, me.display || 'Your profile'),
        h('div', { class: 'sub' },
          'What characters read about you — and what they never see: your name, picture and email ' +
          'stay between you and ' + (me.provider || 'your provider') + '.'))),
    h('div', { class: 'row' }, rev,
      h('button', {
        class: 'btn sm',
        /* The gateway owns the session, so signing out is its business: it
         * clears the `.tokera.com` cookie, which signs you out of code. and
         * bot. at the same time. Clearing something locally would leave the
         * cookie live and the other two sites still signed in. */
        onClick: () => window.__npcdSignOut(),
      }, 'Sign out'))));

  el.appendChild(h('div', { class: 'panel' },
    h('div', { class: 'grid g2' },
      h('label', { class: 'field' }, h('span', {}, 'Unique name'), uname,
        h('div', { class: 'tiny dim', style: 'margin-top:4px' },
          'The only identifier a character ever sees. Tools take it as a target — when an NPC sends you an image, ' +
          'it sends it to this name. Letters, digits, hyphen and underscore; unique across all authors.')),
      h('div', {},
        h('label', { class: 'field' }, h('span', {}, 'Account'),
          h('input', { class: 'input', value: me.email || '', disabled: true })),
        h('div', { class: 'tiny dim' }, 'Set by ' + (me.provider || 'your provider') + ', not editable here.'))),

    choice('gender', 'Gender', ['Male', 'Female'],
      'A character writes about you in prose, so it needs this rather than inferring it.'),
    field('description', 'Description — how a character perceives you', null, 4),
    field('history', 'History', 'Background a character can come to know about you over time.', 3),

    h('div', { class: 'row', style: 'justify-content:space-between;margin-top:6px' },
      h('div', { class: 'tiny dim', style: 'max-width:560px' },
        'Saving appends a new profile turn and tombstones the previous one — it never rewrites. ' +
        'A character’s memory of who you were during past conversations stays true.'),
      h('button', {
        class: 'btn primary',
        onClick: async () => {
          /* The name goes first and on its own. It is the only field here that
           * can be refused — wrong shape, or already somebody else's — and
           * saving the prose first would leave the author looking at an error
           * about a name while the rest had silently succeeded. */
          const wanted = uname.value.trim();
          if (wanted && wanted !== (me.unique_name || '')) {
            try {
              await API.putUniqueName(wanted);
            } catch (e) {
              uname.focus();
              toast(e.error === 'name_taken'
                ? 'that name is already taken — pick another'
                : 'name rejected: ' + (e.detail || e.message), 'err');
              return;
            }
            me.unique_name = wanted;
          }
          const body = Object.fromEntries(Object.entries(f).map(([k, ctl]) => [k, ctl.value]));
          const saved = await API.putProfile(body);
          rev.textContent = 'revision ' + (saved.revision ?? '?');
          paintHistory();
          toast('profile updated — previous revision tombstoned', 'ok');
        },
      }, 'Save'))));

  /* The claim above is only worth making if you can see it happen. Every
   * superseded revision stays readable, because a character that gathered your
   * profile last month attended over what it said then. */
  const history = h('div', {});
  const paintHistory = async () => {
    let revisions;
    try {
      revisions = (await API.getProfileHistory()).revisions || [];
    } catch (e) {
      mount(history, h('div', { class: 'tiny dim' }, 'history unavailable: ' + (e.detail || e.message)));
      return;
    }
    mount(history, ...revisions.map((r) => h('div', {
      class: 'panel',
      style: r.live ? '' : 'opacity:.62',
    },
      h('div', { class: 'row', style: 'justify-content:space-between;margin-bottom:6px' },
        h('span', { class: 'chip' + (r.live ? ' accent' : '') },
          'revision ' + (r.revision ?? 0) + (r.live ? ' · live' : '')),
        r.tombstoned_ms
          ? h('span', { class: 'tiny dim' }, 'superseded ' + ago(r.tombstoned_ms))
          : null),
      h('div', { class: 'tiny' }, r.description || h('span', { class: 'dim' }, '(no description)')))));
  };
  el.appendChild(h('h2', {}, 'Revisions'));
  el.appendChild(history);
  paintHistory();

  el.appendChild(h('h2', {}, 'Characters'));
  el.appendChild(h('div', { class: 'panel' },
    h('div', { class: 'row', style: 'justify-content:space-between' },
      h('div', {}, h('div', { class: 'mono', style: 'font-size:1.3rem;font-weight:700' }, String(me.npc_count ?? 0)),
        h('div', { class: 'tiny dim' }, 'characters you own')),
      h('button', { class: 'btn', onClick: () => go('/') }, 'View roster'))));

  return { el };
}
