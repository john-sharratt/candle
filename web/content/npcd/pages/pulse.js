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
 * perceived and somebody else's perceptions are somebody else's world. The page
 * only watches; nothing on it speaks into the world, because a character that
 * heard from an observer would be reacting to the instrument.
 */

import { API } from '../lib/api.js';
import { h, mount, fmtNum } from '../lib/dom.js';

/* Faster than the shortest heartbeat, so a preempted character's tick lands in
 * the feed while it is still fresh. */
const POLL_MS = 1800;

/* **How much each poll carries, and how much the page remembers — two numbers,
 * because they answer different questions.**
 *
 * A tick is immutable and its number only goes up, so a poll never needs to
 * re-send what the page already has: the window only has to be wide enough that
 * a busy cast cannot outrun one interval. The page then accumulates, and how far
 * back you can scroll is `FEED_KEEP` rather than whatever fits in a request.
 *
 * Asking for the whole history every 1.8 s instead would put ~70 KB on the wire
 * per poll for a page somebody leaves open all afternoon, and re-render several
 * hundred tiles to show the one that changed. */
const FEED_LIMIT = 60;
/* Ticks the page holds. The scheduler's own ring is 512, so this asks for a
 * little less than it can actually serve. */
const FEED_KEEP = 400;

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
  let names = {};
  let timer = null;
  let stopped = false;
  let lastTick = -1;       // so only genuinely new rows animate in

  const censusHost = h('div', { class: 'pulse-cast' });
  const worldHost = h('div', { class: 'pulse-world' });
  const feedHost = h('div', { class: 'pulse-stream' });
  const statusEl = h('span', { class: 'pulse-status' });

  const nameOf = (id) => names[String(id)] || ('character ' + String(id).slice(0, 6));

  /* ── where everybody is ──────────────────────────────────────────────────
   *
   * The feed says what a character did and the window says what it is holding.
   * Neither says where anybody *is*, and without that the most important thing
   * about a cast in one building is invisible: two characters talking and two
   * characters two floors apart produce the same shape of feed.
   *
   * Only rooms with somebody in them are drawn. A building of seventy-eight
   * places listed in full is a wall of empty rows with the two that matter
   * somewhere inside it — and the empty ones say nothing that the level
   * headings do not.
   */
  function renderWorld(data) {
    const worlds = (data && data.worlds) || [];
    const occupied = [];
    for (const w of worlds) {
      for (const r of w.rooms || []) {
        if ((r.who || []).length) occupied.push(r);
      }
    }
    if (!occupied.length) {
      mount(worldHost, h('div', { class: 'pulse-empty' },
        h('p', {}, 'No character has a body in a world yet.')));
      return;
    }
    /* Grouped by level, because a building is read by floor and because two
     * characters on the same floor is the thing worth seeing at a glance. */
    const levels = new Map();
    for (const r of occupied) {
      if (!levels.has(r.area)) levels.set(r.area, { name: r.area_name || r.area, rooms: [] });
      levels.get(r.area).rooms.push(r);
    }
    const bodies = worlds.reduce((n, w) => n + (w.bodies || 0), 0);
    mount(worldHost,
      h('div', { class: 'pulse-world-hd' },
        h('span', { class: 'pulse-world-title' }, 'Where everybody is'),
        h('span', { class: 'tiny dim' },
          `${fmtNum(bodies)} in the world · ${fmtNum(occupied.length)} room${occupied.length === 1 ? '' : 's'} occupied`)),
      ...[...levels.values()].map((lv) => h('div', { class: 'pulse-level' },
        h('div', { class: 'pulse-level-name' }, lv.name),
        ...lv.rooms.map((r) => h('div', { class: 'pulse-room' },
          h('span', { class: 'pulse-room-name' }, r.name),
          h('span', { class: 'pulse-room-who' },
            ...r.who.map((p) => {
              /* Coloured by the same hue the feed uses, so a person here and
               * their ticks over there are recognisably one character. */
              const npcId = (data.bound || {})[p.body];
              const hue = npcId != null ? hueOf(String(npcId)) : null;
              return h('span', {
                class: 'pulse-who' + (r.who.length > 1 ? ' is-together' : ''),
                style: hue != null ? `--hue:${hue}` : '',
                title: p.going_to ? 'on its way to ' + p.going_to : 'here',
              },
              p.name,
              p.going_to ? h('span', { class: 'pulse-going' }, ' →') : null,
              p.holding ? h('span', { class: 'pulse-holding' }, ' · ' + p.holding) : null);
            })))))),
    );
  }

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
        onClick: () => { focus = on ? null : id; lastTick = -1; refetch(); },
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

  /* ── what the page is holding ─────────────────────────────────────────────
   *
   * Keyed by tick number, which is monotonic across the whole cast, so a poll
   * that overlaps what is already here merges instead of duplicating and the
   * page keeps history further back than any one request carries.
   *
   * Cleared on a focus change, because the accumulated set belongs to the query
   * that produced it — keeping it would leave one character's tiles standing
   * under another character's filter. */
  let held = new Map();

  /* Merge a poll's ticks in. Returns whether anything was actually new: most
   * polls of a quiet cast add nothing, and re-mounting several hundred tiles to
   * show no change is the cost that makes a long feed feel broken.
   *
   * **A tick number is only monotonic within one run of the daemon.** The
   * scheduler's counter starts at zero every time the process does, so a page
   * left open across a restart holds a set of high numbers that the new run
   * will take hours to reach — and since the trim keeps the highest and the
   * sort puts them first, every genuinely new tick sorted to the bottom and was
   * then discarded. The feed froze on the previous run's last minutes while the
   * cast strip, which is a different call and does not accumulate, carried on
   * updating. Numbering that goes backwards is a restart, and the only sound
   * thing to do with what came before is drop it. */
  function absorb(ticks) {
    if (!ticks.length) return false;
    const incoming = Math.max(...ticks.map((t) => t.tick));
    const newest = held.size ? Math.max(...held.keys()) : -1;
    if (incoming < newest) {
      held = new Map();
      lastTick = -1;
    }
    let added = false;
    for (const t of ticks) {
      if (!held.has(t.tick)) added = true;
      held.set(t.tick, t);
    }
    if (!added) return false;
    if (held.size > FEED_KEEP) {
      /* Oldest out. Insertion order is the arrival order rather than tick
       * order — a merge can fill a gap behind the newest — so the numbers
       * decide what goes, not the Map's own order. */
      const keep = [...held.keys()].sort((a, b) => b - a).slice(0, FEED_KEEP);
      held = new Map(keep.map((k) => [k, held.get(k)]));
    }
    return true;
  }

  // ── the interlaced stream ───────────────────────────────────────────────
  function renderStream(data) {
    if (!absorb(data.ticks || []) && held.size) return;
    const ticks = [...held.values()].sort((a, b) => b.tick - a.tick);  // newest first
    if (!ticks.length) {
      mount(feedHost, h('div', { class: 'pulse-empty' },
        h('p', {}, focus
          ? 'This character has not thought yet.'
          : 'Nothing has ticked yet. A quiet character still wakes on its ' +
            'heartbeat — give it a moment.')));
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
            onClick: () => { focus = id; lastTick = -1; refetch(); },
          }, nameOf(id)),
          h('span', { class: 'tick-cause', title: c.hint }, c.label),
          h('span', { class: 'tick-spacer' }),
          h('span', { class: 'tick-beat', title: 'metabolism after this tick' },
            '♥ ' + beat(t.heartbeat_ms)),
        ),
        /* The literal prose the model received. Not a summary of it — a summary
         * would hide exactly the wording bugs this view exists to catch. The
         * tile clamps it visually; the `title` keeps the whole thing reachable,
         * so what is on screen is a crop rather than an edit. */
        h('div', { class: 'tick-perceived' },
          ...t.perceived.map((p) => h('p', { class: 'perceive', title: p }, p))),
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

  /* **One refresh at a time, and the newest wins.**
   *
   * The poll and a focus change both call this, so a slow read overlaps the next
   * tick's and the two land in whatever order the network returns them — which
   * on a page whose whole job is *when things happened* renders an older feed
   * over a newer one and leaves it there until the tick after. The guard drops
   * the overlapping call rather than queueing it: the next tick is 1–2 s away
   * and carries fresher data than the one being dropped.
   *
   * `focus` changes the query, so that call must not be dropped — it bumps
   * `generation`, and an in-flight read whose generation is stale discards its
   * own result instead of rendering it. */
  let generation = 0;
  /* Generation of the read currently running, or -1 for none. Keyed by
   * generation rather than a bare boolean so a control's forced read starts
   * immediately — its generation differs — while a poll tick landing on top of
   * the same generation is dropped. A boolean could not tell the two apart, and
   * clearing it to let the control through meant the older read's own cleanup
   * then cleared the newer one's claim. */
  let inFlightGen = -1;
  async function refresh() {
    if (stopped) return;
    if (inFlightGen === generation) return;
    const mine = generation;
    inFlightGen = mine;
    try {
      await read(mine);
    } finally {
      if (inFlightGen === mine) inFlightGen = -1;
    }
  }

  /* Force a refresh that a stale in-flight read cannot overwrite. For a focus
   * change, whose whole point is that the query changed — so what the page has
   * accumulated under the old query goes with it. */
  function refetch() {
    generation += 1;
    held = new Map();
    refresh();
  }

  async function read(mine) {
    const q = { limit: FEED_LIMIT, npc_id: focus || undefined };
    try {
      /* The world comes back with the other two rather than on its own timer,
       * so the room somebody is in and the tick they took in it are read from
       * the same instant. Two pollers would show a character acting in a room
       * it had already left. */
      const [feed, census, world] = await Promise.all([
        API.pulse(q),
        API.pulseCensus(q),
        // A daemon hosting no world is the ordinary case, not a failure, so
        // this one may come back empty without taking the page with it.
        API.pulseWorld().catch(() => ({ worlds: [] })),
      ]);
      /* The query moved while this was in flight — a focus change, a toggle.
       * Rendering now would put the previous filter's data on screen under the
       * new filter's controls. */
      if (stopped || mine !== generation) return;
      names = Object.assign({}, census.names, feed.names);
      renderCast(census);
      renderWorld(world);
      renderStream(feed);

      statusEl.textContent = feed.ready
        ? `${fmtNum(census.characters.length)} awake` +
          (census.thinking != null ? ` · ${fmtNum(census.thinking)} thinking` : '')
        : 'engine loading';
      statusEl.className = 'pulse-status' + (feed.ready ? ' is-live' : '');
    } catch (e) {
      /* A 503 during startup is the ordinary state, not a fault. Saying so is
       * the truth; a red banner would send somebody looking for a problem that
       * is about to resolve itself. */
      if (stopped || mine !== generation) return;
      const loading = e.error === 'no_engine' || e.status === 503;
      mount(feedHost, h('div', { class: 'pulse-empty' },
        h('p', {}, loading
          ? 'The engine is still loading. Ticks appear here when the cast wakes.'
          : 'Could not read the pulse — ' + (e.detail || e.message))));
      statusEl.textContent = loading ? 'loading' : 'unavailable';
      statusEl.className = 'pulse-status';
    }
  }


  mount(el,
    h('header', { class: 'pulse-hd' },
      h('div', { class: 'pulse-hd-main' },
        h('h1', {}, 'Pulse'),
        statusEl),
      h('p', { class: 'pulse-lede' },
        'Every character’s loop, interlaced in the order things happened. A character with an ',
        'empty inbox is ', h('em', {}, 'blocked'), ' — it burns no decode and is not in the batch, ',
        'which is what lets a hundred of them run at once.'),
    ),
    censusHost,
    worldHost,
    feedHost,
  );

  refresh();
  timer = setInterval(refresh, POLL_MS);
  return { el, teardown: () => { stopped = true; clearInterval(timer); } };
}
