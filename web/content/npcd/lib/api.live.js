/* Live implementation — talks to the daemon's /v1 surface, and to the two
 * push streams under /ws. */

import { subscribe } from './ws.js';

async function j(path, opts) {
  const r = await fetch(path, {
    headers: { 'content-type': 'application/json' },
    ...opts,
    body: opts && opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!r.ok) {
    let e = { error: 'http_' + r.status, detail: r.statusText };
    try { e = await r.json(); } catch (_) {}
    // Carry the status as well as the body. Some failures are told apart by
    // code (`name_taken`) and some only by status, and a caller that has to
    // re-fetch to learn which is a caller that will not bother.
    throw Object.assign(new Error(e.detail || e.error), e, { status: r.status });
  }
  return r.status === 204 ? null : r.json();
}

const qs = (o) => {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(o || {})) if (v != null && v !== '') p.set(k, v);
  const s = p.toString();
  return s ? '?' + s : '';
};

export const LiveAPI = {
  getStatus:    () => j('/v1/status'),
  getTelemetry: () => j('/v1/telemetry'),
  /* Full memory accounting. Separate from telemetry because it is a fresh OS
   * read rather than a slice of the retained series, and the performance page
   * wants it at a slower cadence than the charts. */
  /* `getMemoryDump`, not `getMemory` — that name is taken further down by a
   * character's memory layer, and an object literal silently keeps the LAST
   * definition of a duplicated key. The daemon's memory accounting would have
   * been shadowed by a character's memory turns, with no error anywhere. */
  getMemoryDump: () => j('/v1/memory'),
  /* The redo log on disk. Daemon-scoped, not per-character: it is the storage
   * every character's memory is written into, and it is real today because a
   * segmented log is a directory of files. */
  getSubstrateStorage: () => j('/v1/substrate/storage'),

  /* No sign-in or sign-out here. The gateway owns both — its `/auth/*` is
   * served ahead of site routing on every hostname, and the cookie it issues is
   * on `.tokera.com`, which is what makes one sign-in carry to code. and bot.
   * The daemon only ever reports who the gateway says you are. */
  getMe:          () => j('/v1/me'),
  getProfile:     () => j('/v1/me/profile'),
  putProfile:     (b) => j('/v1/me/profile', { method: 'PUT', body: b }),
  getProfileHistory: () => j('/v1/me/profile/history'),
  getProfileRevision: (n) => j(`/v1/me/profile/history/${n}`),
  restoreProfile: (n) => j(`/v1/me/profile/restore/${n}`, { method: 'POST' }),
  putUniqueName:  (n) => j('/v1/me/unique-name', { method: 'PUT', body: { unique_name: n } }),

  listNpcs:  (f) => j('/v1/npc' + qs(f)),
  getNpc:    (id) => j(`/v1/npc/${id}`),
  createNpc: (b) => j('/v1/npc', { method: 'POST', body: b }),
  patchNpc:  (id, b) => j(`/v1/npc/${id}`, { method: 'PATCH', body: b }),
  deleteNpc: (id) => j(`/v1/npc/${id}`, { method: 'DELETE' }),
  setTags:   (id, tags) => j(`/v1/npc/${id}/tags`, { method: 'PUT', body: { tags } }),
  setHidden: (id, hidden) => j(`/v1/npc/${id}/hidden`, { method: 'PUT', body: { hidden } }),

  perceive:  (id, events) => j(`/v1/npc/${id}/perceive`, { method: 'POST', body: { events } }),

  getBeliefs:       (id) => j(`/v1/npc/${id}/beliefs`),
  authorBelief:     (id, b) => j(`/v1/npc/${id}/beliefs/${b.belief_id}`, { method: 'PUT', body: b }),
  deleteBelief:     (id, bid) => j(`/v1/npc/${id}/beliefs/${bid}`, { method: 'DELETE' }),
  getRelationships: (id) => j(`/v1/npc/${id}/relationships`),
  setRelationship:  (id, r) => j(`/v1/npc/${id}/relationships/${r.entity_id}`, { method: 'PUT', body: r }),
  getAgency:        (id) => j(`/v1/npc/${id}/agency`),
  getMemory:        (id, q) => j(`/v1/npc/${id}/memory` + qs(q)),
  getModulation:    (id) => j(`/v1/npc/${id}/modulation`),
  setModulation:    (id, m) => j(`/v1/npc/${id}/modulation`, { method: 'PUT', body: m }),

  getSubstrate: (id) => j(`/v1/npc/${id}/substrate`),
  getLayer:     (id, layer, q) => j(`/v1/npc/${id}/substrate/layer/${layer}` + qs(q)),
  getProjection:(id, tick) => j(`/v1/npc/${id}/projection` + (tick ? '/' + tick : '')),
  getMonitor:   (id, w) => j(`/v1/npc/${id}/monitor` + qs({ window: w })),

  getEnvironment:    (id) => j(`/v1/npc/${id}/environment`),
  setEnvironment:    (id, c) => j(`/v1/npc/${id}/environment`, { method: 'PUT', body: c }),
  injectEnvironment: (id, e) => j(`/v1/npc/${id}/environment/inject`, { method: 'POST', body: e }),

  listInteractions: (id) => j(`/v1/npc/${id}/interaction`),
  openInteraction:  (id, spec) => j(`/v1/npc/${id}/interaction`, { method: 'POST', body: spec }),
  getInteraction:   (ix) => j(`/v1/interaction/${ix}`),
  inject:           (ix, p) => j(`/v1/interaction/${ix}/inject`, { method: 'POST', body: p }),
  endInteraction:   (ix) => j(`/v1/interaction/${ix}`, { method: 'DELETE' }),

  streamInteraction(ix, handlers) {
    const es = new EventSource(`/v1/interaction/${ix}/stream`);
    const bind = (n, fn) => es.addEventListener(n, (e) => { try { fn(JSON.parse(e.data)); } catch (_) {} });
    bind('open', handlers.onOpen || (() => {}));
    bind('act', handlers.onAct || (() => {}));
    bind('act_rendered', handlers.onActRendered || (() => {}));
    bind('tick', handlers.onTick || (() => {}));
    bind('narration', handlers.onNarration || (() => {}));
    bind('state', handlers.onState || (() => {}));
    es.onerror = () => { if (handlers.onError) handlers.onError({ error: 'stream_closed' }); };
    return { cancel: () => es.close() };
  },

  /* `q` is the filter box, and it goes to the SERVER rather than narrowing a
   * list the browser already holds. A hidden world is not sent at all until a
   * whole word of `q` names it, so filtering here would have nothing to reveal
   * — and a list the client narrows is a list the client was first sent whole. */
  /* `reveal` asks the daemon to include hidden documents. It is a request, not
   * a grant: the daemon honours it only for an admin, so sending it from
   * anywhere else changes nothing. */
  listWorlds:    (q, reveal) => j('/v1/world' + qs({ q, reveal: reveal ? 1 : '' })),
  getWorld:      (w) => j(`/v1/world/${w}`),
  setWorld:      (w, c) => j(`/v1/world/${w}`, { method: 'PUT', body: c }),
  setWorldTime:  (w, t) => j(`/v1/world/${w}/time`, { method: 'PUT', body: t }),
  /* `q` as for `listWorlds`: a hidden personality is not sent until a whole
   * word of it names one, so the filter has to reach the server. */
  listPersonalities: (q, reveal) =>
    j('/v1/personality' + qs({ q, reveal: reveal ? 1 : '' })),
  getPersonality:    (a) => j(`/v1/personality/${a}`),
  /* A PUT replaces the whole document — the daemon rewrites
   * `personalities/<a>.yaml` from the body. Send the object you read back, not
   * the fields you changed. */
  setPersonality:    (a, c) => j(`/v1/personality/${a}`, { method: 'PUT', body: c }),

  getLayerSchema:          () => j('/v1/schema/layers'),
  getTurn:     (id, layer, turn) => j('/v1/npc/' + id + '/substrate/turn/' + layer + '/' + turn),
  probe:       (id, text) => j('/v1/npc/' + id + '/project', { method: 'POST', body: { text } }),
  getWorldCollections:     (w) => j('/v1/world/' + w + '/collections'),
  getPersonalityCollections: (a) => j('/v1/personality/' + a + '/collections'),
  /* Push streams. Both hand back `{close}`; a caller that forgets to call it
   * leaks a socket that keeps reconnecting after its page is gone. */
  subscribeLogs:   (onLine, onState) => subscribe('/ws/logs', { onMessage: onLine, onState }),
  subscribeEvents: (onEvent, onState) => subscribe('/ws/events', { onMessage: onEvent, onState }),

  listTools:      () => j('/v1/tools'),
  calibrateTools: () => j('/v1/tools/calibrate', { method: 'POST' }),
  listCommands:   () => j('/v1/commands'),

  /* The authored corpus.
   *
   * `id` is an ADDRESS — `canon/ammo/bolt` — not a path. Nothing here knows
   * where the mind keeps its files, which extension a section uses, or that
   * there is a filesystem at all; the daemon owns all of it. An empty id is the
   * corpus itself, which lists its sections.
   *
   * `world` is optional and narrows by that world's own filters — `selects` for
   * the canon topics, `excludes` for the section categories, its cast for the
   * characters. The daemon applies it, so anything a world excludes is never
   * sent and there is nothing here to leak by forgetting to filter. */
  mindList:  (id, world) => j('/v1/mind/list' + qs({ id, world })),
  /* The canon topics a world admits — history, technology, factions — with how
   * many pages each holds.
   *
   * A listing of `canon` under that world's lens, not a route of its own: the
   * daemon already applies `selects` on the way down, so this is the same
   * answer the mind browser gets and there is no second filter to disagree with
   * it. It exists as a named method because "what does this world know" is a
   * question the world page asks, and spelling it out at the call site there
   * would put the address `canon` in a file that should not know one. */
  getWorldKnowledge: async (world) =>
    (await j('/v1/mind/list' + qs({ id: 'canon', world }))).children || [],
  mindEntry: (id, world) => j('/v1/mind/entry' + qs({ id, world })),
  /* The same entry as fields rather than as text, so it can be edited without
   * knowing YAML. Answers `not_fields` for a document that is not a mapping —
   * a canon page is prose, and prose has no fields — and the console falls back
   * to the text editor for those.
   *
   * A save patches the values into the document already on disk, so the
   * authoring comments above each key survive. That is the whole reason this
   * is not "parse, edit, write out again": 701 of the 712 section files carry
   * comments, and a re-serialise would delete every one. */
  mindFields: (id, world) => j('/v1/mind/fields' + qs({ id, world })),
  saveMindFields: (id, values, world) =>
    j('/v1/mind/fields' + qs({ id, world }), { method: 'PUT', body: { values } }),
  /* `isNew` refuses to land on something that already exists, which is what
   * separates "add" from "save" — a create that overwrote somebody's work would
   * do it with no error to say so. */
  saveMindEntry: (id, text, world, isNew) =>
    j('/v1/mind/entry' + qs({ id, world, new: isNew ? 1 : '' }), { method: 'PUT', body: { text } }),
  deleteMindEntry: (id, world) =>
    j('/v1/mind/entry' + qs({ id, world }), { method: 'DELETE' }),

  /* A portrait: the raw image as the body, not a multipart form. There is one
   * file and no other fields, so an envelope would be ceremony around a byte
   * string — and the daemon decides the format from the bytes anyway, so the
   * `Content-Type` here is a courtesy rather than a claim it trusts. */
  async putPortrait(id, file) {
    const r = await fetch(`/v1/npc/${id}/portrait`, {
      method: 'PUT',
      headers: { 'content-type': file.type || 'application/octet-stream' },
      body: file,
    });
    if (!r.ok) {
      let e = { error: 'http_' + r.status, detail: r.statusText };
      try { e = await r.json(); } catch (_) {}
      throw Object.assign(new Error(e.detail || e.error), e, { status: r.status });
    }
    return r.json();
  },

  generateDescription: (b) => j('/v1/generate/description', { method: 'POST', body: b || {} }),
  generateImage:       (b) => j('/v1/image/generate', { method: 'POST', body: b || {} }),
  listImageModels:     () => j('/v1/image/models'),
  getImageQueue:       () => j('/v1/image/queue'),
  imageUrl:            (id) => `/v1/image/${id}`,
};
