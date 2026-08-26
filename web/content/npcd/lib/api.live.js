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
    // code (`auth_unconfigured`) and some only by status, and a caller that has
    // to re-fetch to learn which is a caller that will not bother.
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

  /* No sign-in or sign-out here. The gateway owns both — its `/auth/*` is
   * served ahead of site routing on every hostname, and the cookie it issues is
   * on `.tokera.com`, which is what makes one sign-in carry to code. and bot.
   * The daemon only ever reports who the gateway says you are. */
  getMe:          () => j('/v1/me'),
  getProfile:     () => j('/v1/me/profile'),
  putProfile:     (b) => j('/v1/me/profile', { method: 'PUT', body: b }),
  getProfileHistory: () => j('/v1/me/profile/history'),
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

  listWorlds:    () => j('/v1/world'),
  getWorld:      (w) => j(`/v1/world/${w}`),
  setWorld:      (w, c) => j(`/v1/world/${w}`, { method: 'PUT', body: c }),
  setWorldTime:  (w, t) => j(`/v1/world/${w}/time`, { method: 'PUT', body: t }),
  listArchetypes:() => j('/v1/archetype'),
  getArchetype:  (a) => j(`/v1/archetype/${a}`),

  getLayerSchema:          () => j('/v1/schema/layers'),
  getTurn:     (id, layer, turn) => j('/v1/npc/' + id + '/substrate/turn/' + layer + '/' + turn),
  probe:       (id, text) => j('/v1/npc/' + id + '/project', { method: 'POST', body: { text } }),
  getWorldCollections:     (w) => j('/v1/world/' + w + '/collections'),
  getArchetypeCollections: (a) => j('/v1/archetype/' + a + '/collections'),
  /* Push streams. Both hand back `{close}`; a caller that forgets to call it
   * leaks a socket that keeps reconnecting after its page is gone. */
  subscribeLogs:   (onLine, onState) => subscribe('/ws/logs', { onMessage: onLine, onState }),
  subscribeEvents: (onEvent, onState) => subscribe('/ws/events', { onMessage: onEvent, onState }),

  listTools:      () => j('/v1/tools'),
  calibrateTools: () => j('/v1/tools/calibrate', { method: 'POST' }),
  listCommands:   () => j('/v1/commands'),

  generateDescription: (b) => j('/v1/generate/description', { method: 'POST', body: b || {} }),
  generateAttributes:  (b) => j('/v1/generate/attributes', { method: 'POST', body: b || {} }),
  generateImage:       (b) => j('/v1/image/generate', { method: 'POST', body: b || {} }),
  listImageModels:     () => j('/v1/image/models'),
  getImageQueue:       () => j('/v1/image/queue'),
  imageUrl:            (id) => `/v1/image/${id}`,
};
