/* Substrate — the inspector over npcd's substrate.
 *
 * The shared inspector (`common/lib/substrate.js`), the one zend's console
 * mounts over its own: npcd's line in the host's `/v1/substrate`, and its views
 * under `/v1/substrate/npcd/`. Every class it writes is `sx-`-prefixed and
 * styled under `.subs` in `app.css`, so this console's own `.row` and `.chip`
 * mean what they mean everywhere else. */

import { BACKEND } from '../lib/api.js';
import { h } from '/common/lib/dom.js';
import { mountSubstrate } from '/common/lib/substrate.js';

/* The inspector's three states, as this console's chips. */
const CHIP = { live: 'chip ok', stale: 'chip warn', changed: 'chip accent' };

export async function render() {
  // `subs` scopes this page's styling — see the block in `app.css`.
  const el = h('div', { class: 'page wide subs' });
  const liveChip = h('button', { class: 'chip', title: 'polling every 6s' }, 'connecting…');
  const kpis = h('div', { class: 'sx-kpis' });
  const sysprompt = h('div', {});
  const layers = h('div', {});

  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, 'Substrate'),
      h('div', { class: 'sub' },
        'The redo log every character’s memory is written into, the layers the mind declares, the conversations in each, and their turns. Each level loads when you open it.')),
    h('div', { class: 'row' }, liveChip)));
  el.appendChild(kpis);
  el.appendChild(h('h2', { class: 'sx-head' }, 'System prompt'));
  el.appendChild(sysprompt);
  el.appendChild(h('h2', { class: 'sx-head' }, 'Layers & conversations'));
  el.appendChild(layers);

  const inspector = mountSubstrate({
    tenant: 'npcd',
    prefix: 'sx-',
    mock: BACKEND === 'mock',
    hosts: { kpis, sysprompt, layers },
    live: {
      set: (state, text) => { liveChip.className = CHIP[state]; liveChip.textContent = text; },
      onClick: (fn) => { liveChip.onclick = fn; },
    },
  });

  return { el, teardown: inspector.stop };
}
