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
  quiet:     { label: 'quiet',   hint: 'came back on its own from a pause, with nothing waiting' },
  pending:   { label: 'batch',   hint: 'drained the events that were waiting' },
  preempted: { label: 'preempt', hint: 'a high-salience event forced this tick now' },
};

/* The same three states, read as "what is it doing now" rather than "why did it
 * wake". `quiet` was `blocked` on both maps and rendered under two different
 * words — and "blocked" is wrong twice over: it reads as stuck, and since idle
 * ticks were removed it is the resting state of every character with nothing
 * happening to it rather than a rare one. */
const READY = {
  quiet:     { label: 'quiet',     hint: 'inbox empty — burns no decode, waiting on the world' },
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

/* How long ago, for a card that is re-rendered every poll.
 *
 * The server sends an age rather than a timestamp, so nothing here has to
 * reconcile this browser's clock with the daemon's — see `Census::acted_ms_ago`.
 * `null` is a character that has not acted yet, which is a different fact from
 * a long silence and reads as one. */
function ago(ms) {
  if (ms == null) return 'no acts yet';
  if (ms < 1000) return 'just now';
  if (ms < 60000) return Math.round(ms / 1000) + 's ago';
  if (ms < 3600000) return Math.round(ms / 60000) + 'm ago';
  return Math.round(ms / 3600000) + 'h ago';
}

/* An act arrives as "tool — what was asked → what came of it", with `✗` in
 * place of the arrow when the world refused it. Three parts, set in three
 * faces: the tool in mono, what it asked for in the body face, and what came
 * back in a quieter one — green-ish when it landed, warn when it did not.
 *
 * Split on the marks rather than parsed, because they are single characters
 * chosen for exactly this (`runtime::LANDED` / `runtime::REFUSED`) and cannot
 * appear inside either half's prose. An act with nothing to report — one that
 * happens in a head — has no arrow and simply has no outcome part.
 *
 * The intent half is searched for the mark rather than the whole string: an
 * outcome can quote a room name containing an em-dash, and splitting the tool
 * off first means only the tail is ever scanned for the arrow. */
function splitAct(text) {
  const i = text.indexOf(' — ');
  let tool = text;
  let intent = '';
  if (i !== -1) {
    tool = text.slice(0, i);
    intent = text.slice(i + 3);
  }
  for (const [mark, ok] of [[' → ', true], [' ✗ ', false]]) {
    const j = intent.indexOf(mark);
    if (j !== -1) {
      return { tool, intent: intent.slice(0, j), said: intent.slice(j + mark.length), ok };
    }
    /* A no-argument act — `recall`, `reflect` with nothing — puts the mark
     * straight after the tool, so the whole tail is the outcome. */
    if (i === -1 && tool.indexOf(mark.trimEnd()) !== -1) {
      const k = tool.indexOf(mark.trimEnd());
      return { tool: tool.slice(0, k), intent: '', said: tool.slice(k + mark.length - 1), ok };
    }
  }
  return { tool, intent, said: '', ok: true };
}

/* The arguments half of an act, as one node per argument.
 *
 * A multi-argument act arrives as `name: value; name: value` (`Act::summary`),
 * so each part gets its label set apart from its value — "to" and "about" in a
 * small dim mono, the values in the body face. A single-argument act arrives
 * bare and stays bare: a label in front of the only thing worth reading is
 * noise.
 *
 * The label is taken only when the part *starts* with a short bare word and a
 * colon-space. A value that happens to contain ": " later on — a quoted line, a
 * ratio — cannot be mistaken for one, and a value that begins with prose keeps
 * its whole self.
 */
function actArgs(intent) {
  if (!intent) return [];
  return intent.split('; ').flatMap((part, n) => {
    const m = /^([a-z_]{1,20}): ([\s\S]+)$/.exec(part);
    const sep = n ? h('span', { class: 'act-sep' }, '·') : null;
    if (!m) return [sep, h('span', { class: 'act-intent' }, part)].filter(Boolean);
    return [
      sep,
      h('span', { class: 'act-key' }, m[1]),
      h('span', { class: 'act-val' }, m[2]),
    ].filter(Boolean);
  });
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
      const r = READY[c.readiness] || READY.quiet;
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
        /* **How long since it last did anything**, in the corner, because it is
         * the one figure on this card that says whether the cast is alive at
         * all. It replaced the idle metabolism — a heartbeat that has scheduled
         * nothing since idle ticks were removed, so an operator could watch its
         * number while the world did not move. */
        h('span', {
          class: 'cast-age',
          title: 'how long since this character last acted',
        }, ago(c.acted_ms_ago)),
        h('span', { class: 'cast-name' }, nameOf(id)),
        h('span', { class: 'cast-state' },
          h('span', { class: 'cast-dot' }), r.label),
        h('div', { class: 'cast-figs' },
          h('span', {}, fmtNum(c.ticks) + ' ticks'),
          c.inbox_depth ? h('span', { class: 'cast-inbox' }, c.inbox_depth + ' waiting') : null,
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
          /* Not "wakes on its heartbeat", which it has not done since idle
           * ticks were removed. A character thinks when something reaches it —
           * somebody speaking, a message, or the room it is standing in doing
           * something — and at no other time. */
          : 'Nothing has ticked yet. A character thinks when something reaches ' +
            'it — give the room a moment.')));
      return;
    }
    const newest = ticks[0].tick;
    mount(feedHost, ...ticks.map((t) => {
      const c = CAUSE[t.cause] || CAUSE.quiet;
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
          /* When this happened, not the metabolism after it. A heartbeat has
           * scheduled nothing since idle ticks were removed, so `♥ 4s` was the
           * same figure on every row of the feed whatever the world was doing —
           * where the age tells you at a glance whether you are looking at a
           * live world or the last thing it did an hour ago. */
          h('span', { class: 'tick-beat', title: 'when this tick ran' },
            ago(t.ms_ago)),
        ),
        /* The literal prose the model received. Not a summary of it — a summary
         * would hide exactly the wording bugs this view exists to catch. The
         * tile clamps it visually; the `title` keeps the whole thing reachable,
         * so what is on screen is a crop rather than an edit. */
        h('div', { class: 'tick-perceived' },
          ...t.perceived.map((p) => h('p', { class: 'perceive', title: p }, p))),
        t.acts && t.acts.length
          ? h('div', { class: 'tick-acts' }, ...t.acts.map((a) => {
              const { tool, intent, said, ok } = splitAct(a);
              /* The tool, each argument as its own labelled part, then what
               * came of it — and the whole line in `title`, so what is on
               * screen is a crop rather than an edit. */
              return h('div', { class: ok ? 'act' : 'act refused', title: a },
                h('span', { class: 'act-tool' }, tool),
                ...actArgs(intent),
                said ? h('span', { class: 'act-mark' }, ok ? '→' : '✗') : null,
                said ? h('span', { class: 'act-said' }, said) : null);
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
    ),
    censusHost,
    worldHost,
    feedHost,
  );

  refresh();
  timer = setInterval(refresh, POLL_MS);
  return { el, teardown: () => { stopped = true; clearInterval(timer); } };
}
