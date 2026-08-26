/* npcd UI vocabulary. The generic pieces come from the shared
 * lib/ui-base.js, re-exported here so pages import a single module.
 *
 * `./ui-base.js` resolves through the layered content roots: it is not in
 * this site, so the server finds it in `common/`. The URL space is one
 * tree even though the files live in two directories. */

import { h } from './dom.js';
export * from './ui-base.js';

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

export function bandChip(band) {
  const cls = band === 'healthy' ? 'ok' : band === 'fixated' ? 'warn' : 'crit';
  const glyph = band === 'healthy' ? '♥' : band === 'fixated' ? '⚠' : '⚡';
  return h('span', { class: 'chip ' + cls, title: 'metacognition monitor band' }, glyph + ' ' + band);
}

export function pending(n) {
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

