/* Pulse — watching the cast think.
 *
 * The hard thing about an NPC engine is that the interesting part is
 * asynchronous and invisible. Events arrive from the world, each character's
 * loop ticks when its own salience says so, and acts come out somewhere else. A
 * conversation transcript tells you what happened and never why it happened
 * *then* — which is where the real bugs are: a character that ticked too often,
 * or never, or drained a batch it should have preempted on.
 *
 * ── the interlacing is the design ──
 *
 * One stream, every character woven together in the order things actually
 * happened. That ordering is the whole point: a cast is not a set of
 * independent transcripts, it is a world in which several minds are running at
 * once, and the questions worth asking are about the relationship between them —
 * who woke while who else was mid-batch, whether an event reached two characters
 * at the same instant.
 *
 * Interlacing only reads if you can tell the characters apart at a glance, so
 * each gets a stable hue derived from its id and carries it on a spine down the
 * left of every tick it owns. Colour is the identity; the name is confirmation.
 *
 * ── what you see ──
 *
 * Your characters, because a tick record carries the literal prose a character
 * perceived and somebody else's perceptions are somebody else's world. An admin
 * can ask for the whole cast; the toggle only appears for someone who could
 * actually use it, because a control that appears and then refuses is worse than
 * one that never appears.
 */

import { API } from '../lib/api.js';
import { h, mount, fmtNum } from '../lib/dom.js';
import { toast } from '../lib/ui.js';

/* Faster than the shortest heartbeat, so a preempted character's tick lands in
 * the feed while it still feels like a consequence of what you typed. */
const POLL_MS = 1800;
const FEED_LIMIT = 60;

/* Why a character woke, and how that reads. The colours are the point of the
 * column: a run of quiet grey with one amber preempt in it is legible at a
 * glance in a way a text column never is. */
const CAUSE = {
  blocked:   { label: 'quiet',   hint: 'the heartbeat fired on an empty inbox' },
  pending:   { label: 'batch',   hint: 'drained the events that were waiting' },
  preempted: { label: 'preempt', hint: 'a high-salience event forced this tick now' },
};

const READY = {
  blocked:   { label: 'blocked',   hint: 'inbox empty — burns no decode, not in the batch' },
  pending:   { label: 'pending',   hint: 'events waiting for the next tick' },
  preempted: { label: 'preempted', hint: 'ticking now' },
};

/* A stable hue per character, so the same character is the same colour on every
 * reload and across the census and the feed. Hashed from the id rather than
 * assigned by position — position changes when a character is added, and a cast
 * whose colours shuffle underneath you is worse than no colour at all. */
function hueOf(id) {
  const s = String(id);
  let acc = 0;
  for (let i = 0; i < s.length; i++) acc = (acc * 31 + s.charCodeAt(i)) >>> 0;
  /* Steps of 47° around the wheel: coprime with 360, so successive characters
   * land far apart instead of clustering. */
  return (acc % 360 + 47 * (acc % 7)) % 360;
}

/* A heartbeat is the character's idle metabolism. Rendered as a duration
 * because "4s" and "2m" say something a raw millisecond count does not. */
function beat(ms) {
  if (ms == null) return '—';
  if (ms >= 60000) return Math.round(ms / 60000) + 'm';
  if (ms >= 1000) return Math.round(ms / 1000) + 's';
  return ms + 'ms';
}

/* An act arrives as "tool — intent; intent". Split so the tool can be set in
 * mono and the intent in the body face, italic — the same treatment the landing
 * page's demo gives an act, because it is the same thing. */
function splitAct(text) {
  const i = text.indexOf(' — ');
  return i === -1
    ? { tool: text, intent: '' }
    : { tool: text.slice(0, i), intent: text.slice(i + 3) };
}

/* Where a character is in its day, as a word. A phase of 0.53 means nothing to
 * read; "afternoon" is the same fact in the register the character lives in. */
function timeOfDay(phase) {
  if (phase == null) return '';
  const h24 = phase * 24;
  if (h24 < 5) return 'night';
  if (h24 < 8) return 'dawn';
  if (h24 < 12) return 'morning';
  if (h24 < 14) return 'midday';
  if (h24 < 18) return 'afternoon';
  if (h24 < 21) return 'evening';
  return 'night';
}

export async function render() {
  const el = h('div', { class: 'pulse' });

  let focus = null;        // npc_id (string) or null for the interlaced stream
  let showAll = false;
  let maySeeAll = false;
  let names = {};
  let timer = null;
  let stopped = false;
  let lastTick = -1;       // so only genuinely new rows animate in

  const censusHost = h('div', { class: 'pulse-cast' });
  const feedHost = h('div', { class: 'pulse-stream' });
  const statusEl = h('span', { class: 'pulse-status' });
  const toggleHost = h('span', {});

  const nameOf = (id) => names[String(id)] || ('character ' + String(id).slice(0, 6));

  // ── the cast strip ──────────────────────────────────────────────────────
  function renderCast(data) {
    const cast = data.characters || [];
    if (!cast.length) {
      mount(censusHost, h('div', { class: 'pulse-empty' },
        h('p', {}, data.ready
          ? 'No characters are in the scheduler yet.'
          : 'The engine is still loading. The cast wakes when it is ready.')));
      return;
    }
    mount(censusHost, ...cast.map((c) => {
      const r = READY[c.readiness] || READY.blocked;
      const id = String(c.npc_id);
      const on = focus === id;
      const hue = hueOf(id);
      return h('button', {
        class: 'cast-card' + (on ? ' on' : '') + ' is-' + c.readiness,
        style: `--hue:${hue}`,
        title: r.hint,
        'aria-pressed': on ? 'true' : 'false',
        onClick: () => { focus = on ? null : id; lastTick = -1; refresh(); },
      },
        h('span', { class: 'cast-spine' }),
        h('span', { class: 'cast-name' }, nameOf(id)),
        h('span', { class: 'cast-state' },
          h('span', { class: 'cast-dot' }), r.label),
        h('div', { class: 'cast-figs' },
          h('span', {}, fmtNum(c.ticks) + ' ticks'),
          c.inbox_depth ? h('span', { class: 'cast-inbox' }, c.inbox_depth + ' waiting') : null,
          h('span', { title: 'the idle metabolism — salience sets it' }, '♥ ' + beat(c.heartbeat_ms)),
        ),
        /* The window's occupancy. `faded` is not a loss — those turns are in
         * the substrate and the gather can pull them back — but a full bar with
         * a high fade count is the sign that continuity is coming from
         * retrieval, which is what it is supposed to be doing. */
        h('div', {
          class: 'cast-win',
          title: `${c.window_turns} of ${c.window_cap} turns held verbatim · ` +
                 `${fmtNum(c.faded)} faded into the substrate`,
        }, h('div', {
          class: 'cast-win-fill',
          style: 'width:' + Math.round(100 * (c.window_turns / (c.window_cap || 1))) + '%',
        })),
        h('div', { class: 'cast-day' },
          c.day == null ? '' : 'day ' + fmtNum(c.day),
          c.phase == null ? null : h('span', { class: 'cast-phase' }, timeOfDay(c.phase))),
      );
    }));
  }

  // ── the interlaced stream ───────────────────────────────────────────────
  function renderStream(data) {
    const ticks = (data.ticks || []).slice().reverse();   // newest first
    if (!ticks.length) {
      mount(feedHost, h('div', { class: 'pulse-empty' },
        h('p', {}, focus
          ? 'This character has not thought yet.'
          : 'Nothing has ticked yet. A quiet character still wakes on its heartbeat — ' +
            'give it a moment, or send it something below.')));
      return;
    }
    const newest = ticks[0].tick;
    mount(feedHost, ...ticks.map((t) => {
      const c = CAUSE[t.cause] || CAUSE.blocked;
      const id = String(t.npc_id);
      const fresh = t.tick > lastTick;
      return h('article', {
        class: 'tick is-' + t.cause + (fresh ? ' is-new' : ''),
        style: `--hue:${hueOf(id)}`,
      },
        h('div', { class: 'tick-spine' }),
        h('header', { class: 'tick-head' },
          h('button', {
            class: 'tick-who',
            title: 'show only this character',
            onClick: () => { focus = id; lastTick = -1; refresh(); },
          }, nameOf(id)),
          h('span', { class: 'tick-cause', title: c.hint }, c.label),
          h('span', { class: 'tick-spacer' }),
          h('span', { class: 'tick-beat', title: 'metabolism after this tick' },
            '♥ ' + beat(t.heartbeat_ms)),
        ),
        /* The literal prose the model received. Not a summary of it — a summary
         * would hide exactly the wording bugs this view exists to catch. */
        h('div', { class: 'tick-perceived' },
          ...t.perceived.map((p) => h('p', { class: 'perceive' }, p))),
        t.acts && t.acts.length
          ? h('div', { class: 'tick-acts' }, ...t.acts.map((a) => {
              const { tool, intent } = splitAct(a);
              return h('div', { class: 'act' },
                h('span', { class: 'act-tool' }, tool),
                intent ? h('span', { class: 'act-intent' }, intent) : null);
            }))
          /* Absent, not "no acts". A character choosing to do nothing is a
           * different and much more interesting fact than a decode that did not
           * run, and drawing them alike would hide the one behind the other. */
          : h('div', { class: 'tick-silent' }, 'nothing came of it'),
      );
    }));
    lastTick = newest;
  }

  async function refresh() {
    if (stopped) return;
    const q = { limit: FEED_LIMIT, npc_id: focus || undefined, all: showAll ? 1 : undefined };
    try {
      const [feed, census] = await Promise.all([API.pulse(q), API.pulseCensus(q)]);
      maySeeAll = !!feed.may_see_all;
      names = Object.assign({}, census.names, feed.names);
      renderCast(census);
      renderStream(feed);

      statusEl.textContent = feed.ready
        ? `${fmtNum(census.characters.length)} awake` +
          (census.thinking != null ? ` · ${fmtNum(census.thinking)} thinking` : '')
        : 'engine loading';
      statusEl.className = 'pulse-status' + (feed.ready ? ' is-live' : '');
      renderToggle();
    } catch (e) {
      /* A 503 during startup is the ordinary state, not a fault. Saying so is
       * the truth; a red banner would send somebody looking for a problem that
       * is about to resolve itself. */
      const loading = e.error === 'no_engine' || e.status === 503;
      mount(feedHost, h('div', { class: 'pulse-empty' },
        h('p', {}, loading
          ? 'The engine is still loading. Ticks appear here when the cast wakes.'
          : 'Could not read the pulse — ' + (e.detail || e.message))));
      statusEl.textContent = loading ? 'loading' : 'unavailable';
      statusEl.className = 'pulse-status';
    }
  }

  function renderToggle() {
    if (!maySeeAll) { mount(toggleHost); return; }
    mount(toggleHost, h('label', { class: 'pulse-toggle' },
      h('input', {
        type: 'checkbox', checked: showAll,
        onChange: (e) => { showAll = e.target.checked; focus = null; lastTick = -1; refresh(); },
      }),
      h('span', {}, 'Show all NPCs'),
    ));
  }

  // ── the composer ────────────────────────────────────────────────────────
  const input = h('input', {
    class: 'input mono', placeholder: 'say something, or /hurt a bolt through the shoulder',
    onKeydown: (e) => { if (e.key === 'Enter') send(); },
  });

  async function send() {
    const line = input.value.trim();
    if (!line) return;
    if (!focus) { toast('Pick a character first — an event goes to one inbox.'); return; }
    try {
      const r = await API.pulseInject(focus, line);
      input.value = '';
      /* Show the prose it became, not just "sent". The rendering is what decides
       * how the event lands, and seeing it is half the debugging. */
      toast((r.preempts ? 'Preempt · ' : 'Delivered · ') + r.prose, 'ok');
      refresh();
    } catch (e) {
      toast(e.detail || e.message || 'refused', 'err');
    }
  }

  let helpEl = null;
  async function toggleHelp() {
    if (helpEl) { helpEl.remove(); helpEl = null; return; }
    /* The daemon's own list, never a copy. A console holding its own would offer
     * a command that gets rejected, and an operator reads a rejection as the
     * character ignoring them. */
    const r = await API.listCommands();
    helpEl = h('div', { class: 'pulse-help' },
      ...(r.commands || []).map((c) => h('div', { class: 'help-row' },
        h('code', { class: 'help-cmd' }, '/' + c.name),
        h('div', {},
          h('div', { class: 'help-desc' }, c.description || ''),
          h('code', { class: 'help-eg' }, c.example || '')))));
    composer.after(helpEl);
  }

  const composer = h('div', { class: 'pulse-composer' },
    input,
    h('button', { class: 'btn primary', onClick: send }, 'Send'),
    h('button', { class: 'btn ghost', onClick: toggleHelp, title: 'the / vocabulary' }, '?'),
  );

  mount(el,
    h('header', { class: 'pulse-hd' },
      h('div', { class: 'pulse-hd-main' },
        h('h1', {}, 'Pulse'),
        statusEl),
      h('p', { class: 'pulse-lede' },
        'Every character’s loop, interlaced in the order things happened. A character with an ',
        'empty inbox is ', h('em', {}, 'blocked'), ' — it burns no decode and is not in the batch, ',
        'which is what lets a hundred of them run at once.'),
      toggleHost,
    ),
    censusHost,
    composer,
    feedHost,
  );

  refresh();
  timer = setInterval(refresh, POLL_MS);
  return { el, teardown: () => { stopped = true; clearInterval(timer); } };
}
