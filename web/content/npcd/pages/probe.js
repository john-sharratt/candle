/* Retrieval probe — "what would this character reach for?"
 *
 * Ported in behaviour from zend's project.html. Type a hypothetical message; it
 * is prefilled under the current lens and projected against the substrate, and
 * the tiles are what the gather would actually select, scored.
 *
 * The concurrency model is SINGLE-FLIGHT WITH A TRAILING RE-RUN, deliberately
 * not a debounce: continuous typing keeps producing results against the current
 * text, one request at a time, instead of stalling until you pause. Single
 * flight is the throttle.
 *
 * This is the instrument the mind document's open questions need — every one of
 * them is a calibration question answerable only by watching real selection. */

import { API } from '../lib/api.js';
import { h, mount, fmtNum, fmtK } from '../lib/dom.js';
import { layerColor, empty, modal, toast } from '../lib/ui.js';
import { singleFlight } from '../lib/live.js';
import { copyText } from '../lib/clip.js';

const KIND_LABEL = {
  turn: 'turn', summary: 'summary', belief: 'belief',
  relationship: 'relationship', section: 'section',
};

export async function render(_params, q) {
  const el = h('div', { class: 'page' });
  const npcs = (await API.listNpcs({}).catch(() => ({ items: [] }))).items || [];
  let npcId = q.npc || (npcs[0] && npcs[0].npc_id);

  const status = h('span', { class: 'chip' }, 'idle');
  const meta = h('div', { class: 'tiny dim mono', style: 'margin:10px 2px 12px;min-height:16px' });
  const tiles = h('div', { class: 'probe-tiles' });

  const sel = h('select', {
    class: 'select', style: 'width:auto',
    onChange: (e) => { npcId = e.target.value; flight.kick(); },
  }, npcs.map((n) => h('option', { value: n.npc_id, selected: n.npc_id === npcId }, n.name)));

  const input = h('textarea', {
    class: 'textarea mono', rows: 3,
    placeholder: 'e.g. what do you see on the eastern line?',
    onInput: () => flight.kick(),
  });

  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, 'Retrieval probe'),
      h('div', { class: 'sub' },
        'Type a hypothetical message. It is projected against this character’s substrate as you type — the tiles ' +
        'below are what the gather would select for it, scored. Click one to read the whole thing.')),
    h('div', { class: 'row' }, status, h('span', { class: 'tiny dim' }, 'character'), sel)));

  el.appendChild(input);
  el.appendChild(meta);
  el.appendChild(tiles);

  if (!npcs.length) {
    mount(tiles, empty('◌', 'No characters', 'Create one to probe its substrate.'));
    return { el };
  }

  const setStatus = (cls, text) => { status.className = 'chip ' + cls; status.textContent = text; };

  const flight = singleFlight(async () => {
    const text = input.value;
    if (!text.trim()) {
      mount(tiles); meta.textContent = ''; setStatus('', 'idle');
      return;
    }
    setStatus('accent', 'projecting…');
    try {
      const res = await API.probe(npcId, text);
      paint(res);
      setStatus('ok', 'ready');
    } catch (e) {
      setStatus('crit', (e.error === 'model_loading') ? 'model loading…' : 'error');
      meta.textContent = 'projection failed: ' + (e.detail || e.message || e);
    }
  });

  function paint(res) {
    const list = (res.tiles || []).slice().sort((a, b) => b.score - a.score);
    const picked = list.filter((t) => t.selected).length;
    meta.textContent = [
      res.query_tokens != null ? res.query_tokens + ' query tokens' : null,
      list.length + ' candidate' + (list.length === 1 ? '' : 's'),
      picked + ' selected',
      res.budget ? `${fmtNum(res.budget.would_use)} / ${fmtNum(res.budget.total)} tok` : null,
    ].filter(Boolean).join(' · ');

    if (!list.length) return mount(tiles, empty('◌', 'Nothing retrieved for this query'));
    const max = list.reduce((m, t) => Math.max(m, t.score), 0) || 1;
    mount(tiles, list.map((t) => tile(t, max)));
  }

  function tile(t, max) {
    const ratio = t.score / max;
    const cls = ratio >= 0.66 ? 'hi' : ratio >= 0.33 ? 'mid' : '';
    return h('div', {
      class: 'probe-tile' + (t.selected ? '' : ' unsel'),
      style: `border-left-color:${layerColor(t.layer) || 'var(--line-2)'}`,
      onClick: () => open(t),
    },
      h('div', { class: 'row', style: 'gap:8px' },
        h('span', { class: 'probe-score ' + cls }, String(Math.round(t.score))),
        h('span', { style: 'flex:1' }),
        t.selected ? null : h('span', { class: 'chip' }, 'skipped'),
        h('span', { class: 'chip' }, KIND_LABEL[t.kind] || t.kind),
        h('span', { class: 'chip' }, t.tokens + ' tok')),
      h('div', { class: 'probe-nm mono' }, t.label || '(unnamed)'),
      t.layer ? h('div', { class: 'tiny dim mono' }, t.layer) : null,
      t.text ? h('div', { class: 'probe-prev' }, t.text) : null);
  }

  function open(t) {
    const copyBtn = h('button', { class: 'btn sm ghost' }, 'Copy');
    copyBtn.addEventListener('click', () => copyText(t.text || '', copyBtn));
    modal({
      title: t.label || '(unnamed)', wide: true,
      body: h('div', {},
        h('div', { class: 'row wrap', style: 'gap:6px;margin-bottom:12px' },
          h('span', { class: 'chip accent' }, 'score ' + Math.round(t.score)),
          h('span', { class: 'chip' }, KIND_LABEL[t.kind] || t.kind),
          h('span', { class: 'chip' }, t.tokens + ' tok'),
          t.layer ? h('span', { class: 'chip' }, t.layer) : null,
          t.selected ? h('span', { class: 'chip ok' }, 'selected') : h('span', { class: 'chip' }, 'skipped'),
          h('span', { style: 'flex:1' }), copyBtn),
        h('pre', { class: 'disc-pre' }, t.text || '(no content resolved)')),
    });
  }

  requestAnimationFrame(() => input.focus());
  return { el };
}
