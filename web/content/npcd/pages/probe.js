/* Retrieval probe — "what would this character reach for?"
 *
 * The port of zend's `project.html`, with §36 of `docs/npc_api_gui_design.md`
 * added underneath it. Two instruments on one page, answering the same question
 * from opposite ends — and the order between them is not arbitrary:
 *
 *   THE PROBE    what a hypothetical message *would* pull, projected live as
 *                you type. This is the page. It is the only control here that
 *                does anything, and it is why the page gets opened.
 *   THE GATHER   what a real tick actually projected — the lens it was read
 *                through, the per-layer occupancy it selected, the budget it
 *                spent, and what it turned away. Context for reading the
 *                probe's output, stepped through history.
 *
 * The gather was briefly first, which put a read-only summary above the input
 * box and made the instrument look like a footnote to its own telemetry.
 *
 * ── Dropped turns are first-class ────────────────────────────────────────────
 *
 * The version this replaces showed only what was retrieved. The interesting
 * question is almost never what was gathered — it is what nearly was, and why
 * it missed: a turn that lost to the budget is a sizing problem, one that lost
 * to the threshold is a scoring problem, and they need opposite fixes. A tool
 * that shows only winners cannot tell you which you have.
 *
 * The probe's concurrency model is SINGLE-FLIGHT WITH A TRAILING RE-RUN,
 * deliberately not a debounce: continuous typing keeps producing results
 * against the current text, one request at a time, instead of stalling until
 * you pause. Single flight is the throttle. */

import { API } from '../lib/api.js';
import { h, mount, fmtNum, fmtK } from '../lib/dom.js';
import { link } from '../lib/router.js';
import { layerColor, empty, modal, toast } from '../lib/ui.js';
import { singleFlight } from '../lib/live.js';
import { copyText } from '../lib/clip.js';

const KIND_LABEL = {
  turn: 'turn', summary: 'summary', belief: 'belief',
  relationship: 'relationship', section: 'section',
};

/* Why a turn did not make it. Named rather than shown raw, because the reason
 * is the actionable half: budget means the window is too small for the gather,
 * threshold means the scorer did not rate it, and the fixes are opposite. */
const DROP_REASON = {
  budget: { label: 'lost to budget', hint: 'Scored well enough, but the window filled first.' },
  threshold: { label: 'below threshold', hint: 'Scored too low to be considered.' },
  masked: { label: 'masked', hint: 'Excluded by this layer’s visibility rule.' },
};

const pct = (a, b) => (b > 0 ? Math.round((a / b) * 100) : 0);

export async function render(_params, q) {
  const el = h('div', { class: 'page wide proj' });
  const npcs = (await API.listNpcs({}).catch(() => ({ items: [] }))).items || [];
  let npcId = q.npc || (npcs[0] && npcs[0].npc_id);

  // ── the gather ────────────────────────────────────────────────────────────

  let proj = null;          // the projection currently shown
  let tick = null;          // null = follow the latest rather than pinning one
  const gatherHost = h('div', {});

  const tickLabel = h('span', { class: 'tick-n mono' }, '—');
  const stepBtn = (glyph, delta, title) => h('button', {
    class: 'btn sm ghost', title,
    onClick: () => {
      if (proj == null || proj.tick == null) return;
      // Stepping pins the view. Following the live tick while trying to read
      // one would move the thing being read out from under the reader.
      tick = Math.max(0, (tick == null ? proj.tick : tick) + delta);
      loadGather();
    },
  }, glyph);
  const liveBtn = h('button', {
    class: 'btn sm', title: 'follow the newest tick',
    onClick: () => { tick = null; loadGather(); },
  }, '⟲ latest');

  async function loadGather() {
    try {
      proj = await API.getProjection(npcId, tick == null ? null : tick);
      paintGather();
    } catch (e) {
      mount(gatherHost, empty('◌', 'No projection for this tick',
        e && (e.detail || e.message) ? String(e.detail || e.message) : ''));
      tickLabel.textContent = tick == null ? '—' : String(tick);
    }
  }

  function paintGather() {
    const p = proj || {};
    tickLabel.textContent = p.tick != null ? 't' + p.tick : '—';
    liveBtn.classList.toggle('primary', tick == null);

    const b = p.budget || {};
    const used = b.used || 0, total = b.total || 0;
    const bpct = pct(used, total);
    const bstate = bpct >= 98 ? 'crit' : bpct >= 90 ? 'warn' : 'ok';

    const layers = (p.layers || []).slice().sort((x, y) => (y.tokens || 0) - (x.tokens || 0));
    const maxTok = layers.reduce((m, l) => Math.max(m, l.tokens || 0), 0) || 1;

    mount(gatherHost,
      /* Budget first. Every layer row below is a claim on it, and the number
       * that decides whether the dropped list is a sizing problem. */
      h('div', { class: 'panel' },
        h('div', { class: 'row', style: 'justify-content:space-between;align-items:baseline' },
          h('h3', { style: 'margin:0' }, 'Budget'),
          h('span', { class: 'mono st-' + bstate },
            `${fmtNum(used)} / ${fmtNum(total)} tok · ${bpct}%`)),
        h('div', { class: 'kpi-meter', style: 'height:6px;margin-top:10px' },
          h('i', { class: 'bg-' + bstate, style: `width:${Math.min(100, bpct)}%` }))),

      /* The lens. Which system-prompt branch was active decides what every
       * score below even means, so it belongs above them, not in a footnote. */
      p.system_prompt ? h('div', { class: 'panel' },
        h('h3', { style: 'margin-top:0' }, 'System prompt ',
          h('span', { class: 'tiny dim' }, '· the lens this was read through')),
        h('div', { class: 'row wrap', style: 'gap:6px' },
          p.system_prompt.template
            ? h('span', { class: 'chip accent' }, 'template · ' + p.system_prompt.template) : null,
          p.system_prompt.mood
            ? h('span', { class: 'chip warn' }, 'mood · ' + p.system_prompt.mood
              + (p.system_prompt.mood_spiked_at != null ? ` (spiked t${p.system_prompt.mood_spiked_at})` : '')) : null,
          ...(p.system_prompt.sections || []).map((s) => h('span', { class: 'chip mono' }, s)))) : null,

      /* Gathered. Ordered by tokens rather than by the layer schema: the layer
       * eating the budget is the one you came to find. */
      h('div', { class: 'panel' },
        h('div', { class: 'row', style: 'justify-content:space-between;align-items:baseline' },
          h('h3', { style: 'margin:0' }, 'Gathered'),
          h('span', { class: 'tiny dim' }, 'selected / available · top score')),
        layers.length
          ? h('div', { class: 'gather', style: 'margin-top:10px' },
            ...layers.map((l) => gatherRow(l, maxTok)))
          : empty('◌', 'Nothing gathered', 'This tick projected an empty window.')),

      droppedPanel(p.dropped || []));
  }

  function gatherRow(l, maxTok) {
    const col = layerColor(l.layer) || 'var(--line-3)';
    const w = Math.max(1.5, ((l.tokens || 0) / maxTok) * 100);
    /* A layer that selected 8 of 41 is doing something different from one that
     * selected 8 of 4,412 — the second is a needle-in-a-haystack retrieval and
     * the ratio is the only thing that says so. */
    const ratio = pct(l.gathered || 0, l.available || 0);
    return h('div', { class: 'gather-row' },
      h('span', { class: 'gather-nm mono', style: `color:${col}` }, l.layer),
      h('span', { class: 'gather-bar' },
        h('i', { style: `width:${w}%;background:${col}` })),
      h('span', { class: 'gather-tok mono' }, fmtNum(l.tokens || 0)),
      h('span', { class: 'gather-sel mono' },
        `${fmtNum(l.gathered || 0)} / ${fmtK(l.available || 0)}`,
        h('span', { class: 'dim' }, ` ${ratio}%`)),
      h('span', { class: 'gather-top mono' },
        l.top_score != null ? l.top_score.toFixed(2) : '—'));
  }

  function droppedPanel(dropped) {
    const total = dropped.reduce((a, d) => a + (d.turns || 0), 0);
    return h('div', { class: 'panel' },
      h('div', { class: 'row', style: 'justify-content:space-between;align-items:baseline' },
        h('h3', { style: 'margin:0' }, 'Dropped'),
        h('span', { class: 'tiny dim' },
          total ? `${fmtNum(total)} turns did not make it` : 'nothing was turned away')),
      h('div', { class: 'tiny dim', style: 'margin:6px 0 10px' },
        'What nearly loaded. A turn that lost to the budget is a sizing problem; '
        + 'one that fell below the threshold is a scoring problem.'),
      dropped.length
        ? h('div', {}, ...dropped.map((d) => {
          const r = DROP_REASON[d.reason] || { label: d.reason || 'dropped', hint: '' };
          const col = layerColor(d.layer) || 'var(--line-3)';
          return h('div', { class: 'drop-row', title: r.hint },
            h('span', { class: 'disc-swatch', style: `background:${col}` }),
            h('span', { class: 'mono', style: 'min-width:112px' }, d.layer),
            h('span', { class: 'mono' }, fmtNum(d.turns || 0) + ' turns'),
            h('span', { style: 'flex:1' }),
            h('span', { class: 'chip ' + (d.reason === 'budget' ? 'warn' : '') }, r.label));
        }))
        : empty('✓', 'Nothing dropped',
          'Every candidate that scored above the threshold fitted in the budget.'));
  }

  // ── the probe ─────────────────────────────────────────────────────────────

  const status = h('span', { class: 'chip' }, 'idle');
  const meta = h('div', { class: 'tiny dim mono', style: 'margin:10px 2px 12px;min-height:16px' });
  const tiles = h('div', { class: 'probe-tiles' });

  const sel = h('select', {
    class: 'select', style: 'width:auto',
    onChange: (e) => {
      npcId = e.target.value;
      tick = null;              // ticks belong to a character; do not carry one across
      loadGather();
      flight.kick();
    },
  }, npcs.map((n) => h('option', { value: n.npc_id, selected: n.npc_id === npcId }, n.name)));

  const input = h('textarea', {
    class: 'textarea mono', rows: 3,
    placeholder: 'e.g. what do you see on the eastern line?',
    onInput: () => flight.kick(),
  });

  /* Nothing to probe.
   *
   * The page stops here rather than rendering itself and apologising inside it.
   * The earlier version drew the whole instrument — a textarea inviting you to
   * type, a tick stepper with its arrows, a section heading — and then put a
   * small "No characters" box in the middle of it, twice. Every one of those
   * controls was inert, and a disabled instrument surrounded by live-looking
   * chrome reads as broken rather than as empty.
   *
   * So: the title, one composed panel that says what the page is *for* — the
   * reader has probably never seen it working — and the one action that leads
   * anywhere from here. */
  if (!npcs.length) {
    el.appendChild(h('div', { class: 'hd' },
      h('div', {}, h('h1', {}, 'Retrieval probe'),
        h('div', { class: 'sub' },
          'What a character would reach for, before you ask it anything.'))));
    el.appendChild(h('div', { class: 'blank' },
      h('div', { class: 'blank-mark' }, '◉'),
      h('div', { class: 'blank-title' }, 'No characters to probe'),
      h('div', { class: 'blank-body' },
        'Type a hypothetical message and this projects it against a character’s substrate as '
        + 'you type — showing what its mind would load to answer, scored, and what it would '
        + 'leave behind. It needs a character first.'),
      h('div', { style: 'margin-top:20px' },
        link('/npc/new', { class: 'btn primary' }, '+ New character'))));
    return { el };
  }

  /* The probe is the page. Typing is the first thing you can do here and the
   * reason to open it at all, so the box is under the title with nothing
   * between — no panel, no stepper, no summary to scroll past. The gather below
   * is context for what the probe returns; it was briefly on top, which put a
   * read-only summary in front of the one control that does anything. */
  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, 'Retrieval probe'),
      h('div', { class: 'sub' },
        'Type a hypothetical message. It is projected against this character’s substrate as '
        + 'you type — the tiles below are what the gather would select for it, scored. Click '
        + 'one to read the whole thing.')),
    h('div', { class: 'row' }, status, h('span', { class: 'tiny dim' }, 'character'), sel)));

  el.appendChild(input);
  el.appendChild(meta);
  el.appendChild(tiles);

  /* Below the fold, deliberately: what a real tick actually loaded. The probe
   * answers "what would it reach for"; this answers "what did it", and the
   * second is only interesting once you have asked the first. */
  el.appendChild(h('h2', {}, 'The last gather'));
  el.appendChild(h('div', { class: 'tiny dim', style: 'margin:-4px 0 12px' },
    'What this character actually loaded on a real tick — the lens it was read through, the '
    + 'layers it drew from, and what it turned away. Step back through history to trace a '
    + 'strange act to the gather that produced it.'));
  el.appendChild(h('div', { class: 'panel tickbar' },
    h('span', { class: 'tiny dim' }, 'tick'),
    tickLabel,
    stepBtn('◀', -1, 'previous tick'),
    stepBtn('▶', 1, 'next tick'),
    liveBtn));
  el.appendChild(gatherHost);

  mount(gatherHost, empty('◷', 'Loading the gather…', ''));
  loadGather();

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
