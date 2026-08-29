/* npcd UI vocabulary. The generic pieces come from the shared
 * lib/ui-base.js, re-exported here so pages import a single module.
 *
 * `./ui-base.js` resolves through the layered content roots: it is not in
 * this site, so the server finds it in `common/`. The URL space is one
 * tree even though the files live in two directories. */

import { h } from './dom.js';
import { can } from './router.js';
export * from './ui-base.js';

/* ── Access, in the shape a page needs it ──────────────────────────────────
 *
 * One vocabulary for "this section is not yours to change", so a page does not
 * invent its own and so a reader meets the same thing everywhere.
 *
 * None of this is a security control. The daemon refuses the request whatever
 * the browser renders — `guard::Api` puts every route behind a role and there
 * is no way past it from here. What these do is stop the console *offering*
 * what the server will refuse, because a Save that 403s teaches its user the
 * product is broken when the truth is only that this is somebody else's job.
 */

/** Whether the viewer may edit something gated at `need`. */
export const mayEdit = (need = 'admin') => can(need);

/**
 * Props that make an input, textarea or checkbox read-only when the viewer
 * cannot edit at `need`. Spread into `h`.
 *
 * `readonly` for text, `disabled` for checkboxes and selects — a `readonly`
 * checkbox is still clickable, which is exactly the trap this exists to avoid.
 */
export function ro(need = 'admin', kind = 'text') {
  if (mayEdit(need)) return {};
  return kind === 'text' ? { readonly: true } : { disabled: true };
}

/** The chip that says a panel is read-only, or nothing when it is not. */
export const roChip = (need = 'admin') =>
  (mayEdit(need) ? null : h('span', { class: 'chip', title: `needs the ${need} role` }, 'read-only'));

/**
 * A control the viewer may not use, replaced by nothing.
 *
 * `only('admin', () => h('button', …))` builds the button for an admin and
 * yields `null` otherwise, which `h` drops. Takes a thunk so the control is not
 * constructed — and its handlers not wired — for somebody who will never see
 * it.
 */
export const only = (need, build) => (mayEdit(need) ? build() : null);

/** The sentence under a read-only panel, explaining rather than just refusing. */
export const roNote = (what, need = 'admin') =>
  `Read-only — changing ${what} is an ${need}’s to do. Edit the file in the mind, or ask one.`;

export const LAYERS = [
  'perception', 'action', 'agency', 'relationships', 'beliefs',
  'memory', 'interaction', 'environment', 'world',
];

export const layerColor = (l) => `var(--l-${l})`;

export const STATE_LABEL = {
  active: 'active', ticking: 'ticking now', idle: 'idle',
  asleep: 'asleep', suspended: 'suspended', tombstoned: 'deleted',
};

export const MODE_LABEL = {
  physical: 'physical encounter', video_call: 'video call',
  voice_call: 'voice call', instant_message: 'instant message',
};

export const MODE_ICON = { physical: '◍', video_call: '▣', voice_call: '◎', instant_message: '▤' };

export function stateDot(state) { return h('span', { class: 'dot ' + state, title: STATE_LABEL[state] || state }); }

/* The metacognition band, or the fact that nobody measured it.
 *
 * `null` is not `healthy`. The monitor is an engine measurement, and a
 * character the engine has never run has no band — rendering one as a green
 * "♥ healthy" states that it was checked and found well, which is the most
 * confident possible way to be wrong. */
export function bandChip(band) {
  if (!band) {
    return h('span', { class: 'chip', title: 'the monitor has not run for this character' },
      '· not measured');
  }
  const cls = band === 'healthy' ? 'ok' : band === 'fixated' ? 'warn' : 'crit';
  const glyph = band === 'healthy' ? '♥' : band === 'fixated' ? '⚠' : '⚡';
  return h('span', { class: 'chip ' + cls, title: 'metacognition monitor band' }, glyph + ' ' + band);
}

/* Pending events as four pips, or an empty rail when the figure is absent.
 *
 * Same rule: an unlit rail means "measured, and nothing is waiting". A
 * character with no engine behind it gets the muted variant, which is visibly
 * not the same thing. */
export function pending(n) {
  if (n == null) {
    return h('span', { class: 'pend na', title: 'no engine has reported for this character' },
      [0, 1, 2, 3].map(() => h('i', {})));
  }
  return h('span', { class: 'pend', title: `${n} pending event${n === 1 ? '' : 's'}` },
    [0, 1, 2, 3].map((i) => h('i', { class: i < Math.min(n, 4) ? 'on' : '' })));
}

export function avatar(npc, big) {
  const initial = (npc.name || '?').trim()[0] || '?';
  const kids = npc.portrait && npc.portrait.image_id && npc.portrait.url
    ? [h('img', { src: npc.portrait.url, alt: '' })]
    : [initial];
  return h('div', { class: 'avatar' + (big ? ' lg' : '') }, kids);
}

