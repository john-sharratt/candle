/* The conversations you are in — held by the app, not by the page (§32).
 *
 * # Why this is not the console's own state
 *
 * A conversation is not a view. Opening one puts a real body in a real room and
 * a character starts being told it has company; the console is only where you
 * watch that from. So it cannot live in `render()`, where the router discards it
 * the moment you look at anything else — going to Pulse for ten seconds used to
 * take you out of the world and hand the character an empty room, and coming
 * back was a *new* conversation with none of what had been said in it.
 *
 * This module is where a conversation lives instead. It outlives every route,
 * because it is module state in a single-page app, and it holds the transcript
 * so returning to a console shows the exchange you left rather than a blank one.
 *
 * # What still ends a conversation
 *
 * Two things, and only two: pressing **End**, and going quiet for long enough
 * that the daemon reaps it (`idle_timeout_secs`, five minutes for standing in a
 * room). Navigating around the console is neither. A console with the stream
 * open counts as being there — the daemon's `Interactions::attended` — so the
 * timeout is measuring what it is meant to: nobody watching, and nothing said.
 *
 * # The transcript
 *
 * A log of plain entries, not DOM. The console paints from it on mount and
 * appends to it as frames land, which is what makes leaving and returning
 * lossless. Capped, because a conversation left open for an afternoon is
 * otherwise a memory leak with a nice name.
 */

import { API } from './api.js';

const KEY = 'npcd.sessions';

/* Enough to scroll back through a long exchange, bounded so an afternoon does
 * not accumulate. The daemon is the record; this is the window on it. */
const CAP = 500;

/** interaction_id → session */
const LIVE = new Map();
const watchers = new Set();

/* Which ids to look for again after a reload. The ids only — the daemon owns
 * whether they are still live, and a transcript restored from storage would be
 * a conversation this browser had rather than one that happened. */
function persist() {
  try { localStorage.setItem(KEY, JSON.stringify([...LIVE.keys()])); } catch (_) {}
}

function announce() {
  for (const fn of [...watchers]) { try { fn(list()); } catch (_) {} }
}

/** Told whenever the set of open conversations changes. Returns an unsubscribe. */
export function subscribe(fn) {
  watchers.add(fn);
  return () => watchers.delete(fn);
}

export function list() { return [...LIVE.values()]; }
export function get(ix) { return LIVE.get(ix) || null; }
export function countFor(npcId) {
  return list().filter((s) => s.npcId === String(npcId)).length;
}

/**
 * Take an interaction the daemon has confirmed into the set, or return the one
 * already there. `info` is a `/v1/interaction/:ix` body; `npc` the character
 * record, for the rail and the header.
 */
export function adopt(info, npc) {
  const ix = info.interaction_id;
  const known = LIVE.get(ix);
  if (known) {
    if (npc) known.npc = npc;
    known.mode = info.mode;
    return known;
  }
  const s = {
    ix,
    npcId: String(info.npc_id),
    npc: npc || { npc_id: String(info.npc_id), name: 'NPC' },
    mode: info.mode,
    interlocutor: info.interlocutor || null,
    openedAt: Date.now(),
    /* The exchange, as data. See the module note — this is what makes leaving
     * and coming back lossless. */
    log: [],
    /* act_id of everything already logged. Re-attaching asks the daemon to
     * resume one tick behind the last one read, so the tail of that tick is
     * handed back and has to be recognised rather than painted twice. */
    seen: new Set(),
    /* The tick to resume the stream from, so what the character did while you
     * were looking at another page is still there when you come back. */
    lastTick: 0,
  };
  LIVE.set(ix, s);
  persist();
  announce();
  return s;
}

/** Add one entry to a session's transcript, oldest dropped past the cap. */
export function append(s, entry) {
  s.log.push(entry);
  if (s.log.length > CAP) s.log.splice(0, s.log.length - CAP);
  return entry;
}

/**
 * Start talking to a character, or rejoin the conversation already open with
 * them in this mode — the daemon's `open` is idempotent per person and mode, so
 * asking twice continues rather than forking.
 */
export async function open(npcId, mode) {
  const info = await API.openInteraction(npcId, { mode });
  const npc = await API.getNpc(npcId).catch(() => null);
  return adopt(info, npc);
}

/**
 * Whether you are standing in a room with this character, **as the daemon sees
 * it**.
 *
 * Asked of the server rather than answered from the set above, because being in
 * a room is world state and not a fact about this browser: the session may have
 * gone quiet while the page was closed, another tab may have walked you out,
 * and either way there is a body in a room or there is not. What is held here
 * is only the transcript.
 *
 * `null` when you are away. The character's own record is joined back on when
 * this browser has one, so the roster band can name who you are with.
 */
export async function where(npcId, npc) {
  const r = await API.listInteractions(npcId).catch(() => null);
  const found = (r?.interactions || []).find(
    (ix) => ix.mode === 'physical' && ix.state === 'live',
  );
  if (!found) {
    // Gone. Drop any transcript still held for it, so the roster does not go on
    // offering a room nobody is in.
    for (const s of list()) {
      if (s.npcId === String(npcId) && s.mode === 'physical') forget(s.ix);
    }
    return null;
  }
  return adopt(found, npc);
}

/**
 * End one for real: the body leaves the room and the character stops being told
 * it has company.
 *
 * Dropped here first and awaited after, so the rail and the roster update on
 * the press rather than on the round trip. The daemon is the authority on
 * whether it existed; there is nothing useful to do with a refusal for a
 * conversation this side has already let go of.
 */
export async function close(ix) {
  LIVE.delete(ix);
  persist();
  announce();
  await API.endInteraction(ix).catch(() => {});
}

/** Drop one the daemon says is over, without asking it to end what already has. */
export function forget(ix) {
  if (!LIVE.delete(ix)) return;
  persist();
  announce();
}

/* One restore per page load, however many pages ask for it. Two pages mounting
 * together would otherwise each issue the whole set of lookups. */
let restoring = null;

/**
 * Pick the conversations back up after a reload.
 *
 * The console's assets are embedded in the binary, so a rebuild reloads this
 * page out from under an open conversation — and the conversation is still
 * running on the daemon, which is the thing that matters. Ids come from storage;
 * whether each is still live is the daemon's answer, and one that is not is
 * simply dropped.
 */
export function restore() {
  if (restoring) return restoring;
  restoring = (async () => {
    let ids = [];
    try { ids = JSON.parse(localStorage.getItem(KEY) || '[]'); } catch (_) {}
    if (!Array.isArray(ids)) ids = [];
    await Promise.all(ids.map(async (ix) => {
      if (typeof ix !== 'string' || LIVE.has(ix)) return;
      const info = await API.getInteraction(ix).catch(() => null);
      if (!info || info.state !== 'live') return;
      const npc = await API.getNpc(info.npc_id).catch(() => null);
      adopt(info, npc);
    }));
    // Rewrite the list, so ids the daemon has forgotten stop being looked up on
    // every load for ever.
    persist();
    announce();
    return list();
  })();
  return restoring;
}
