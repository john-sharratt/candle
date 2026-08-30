/* Pure client-side mock. `?mock=1` runs the whole GUI with no daemon.
 *
 * It must be able to reach the states that are easy to design wrong (§41):
 * a hidden NPC found only by tag, an act whose narration lags, an image job
 * stalled on VRAM, a description edit regenerating a portrait, a user with no
 * worlds. `?empty=1` gives the zero-NPC first-run state; `?loggedout=1` the
 * landing page. */

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const flag = (k) => { try { return new URLSearchParams(location.search).has(k); } catch (_) { return false; } };
const EMPTY = flag('empty');

const pad2 = (n) => String(n).padStart(2, '0');
/** Seconds-since-midnight to the HH:MM:SS the log pane renders. */
const clock = (s) => `${pad2(Math.floor(s / 3600) % 24)}:${pad2(Math.floor(s / 60) % 60)}:${pad2(s % 60)}`;

const WORLD_EPOCH = 412 * 86400000 + 6 * 3600000 + 14 * 60000;
const worldMs = () => WORLD_EPOCH + (Date.now() % 3600000) * 60;

/* Narrow a listing by the console's filter box.
 *
 * A substring match, which is looser than the daemon's whole-word rule — and
 * that difference is deliberate rather than an oversight. The daemon's rule
 * exists to stop a HIDDEN document being discovered by typing letters and
 * watching; nothing in this fixture is hidden, so there is nothing here for it
 * to protect and a substring is what a person expects a filter to do. */
const narrow = (rows, q, fields) => {
  const t = (q || '').trim().toLowerCase();
  if (!t) return rows;
  return rows.filter((r) => fields(r).some((f) => String(f || '').toLowerCase().includes(t)));
};

const mk = (id, name, arch, archName, state, pending, band, overlap, hb, hidden, tags, desc) => ({
  npc_id: id, name, world_id: 'ardh', personality_id: arch, personality_name: archName, state,
  tick: { heartbeat_ms: hb, last_tick_ms: Date.now() - pending * 900 - 400, pending_events: pending, salience_gate: 0.42 },
  environment_enabled: true, monitor: { overlap, band }, owner_id: 'u_8812', access: 'owner',
  hidden, tags, portrait: null, persona: { description: desc, origin: 'generated' },
  live_interactions: state === 'active' ? 2 : 0,
  created_ms: Date.now() - 86400000 * 12, updated_ms: Date.now() - 4000,
});

const NPCS = EMPTY ? [] : [
  mk('10237749914772934281', 'Varek', 'loyal-soldier', 'Loyal Soldier', 'active', 3, 'healthy', 0.19, 30000, false, ['campaign-2', 'north'],
    'Fifty-three, a former staff sergeant who now runs the night shift on a loading dock. Precise about time to the point of rudeness. Comfortable giving orders, uneasy in conversations with no clear purpose.'),
  mk('10237749914772934282', 'Ilse', 'merchant', 'Merchant', 'active', 0, 'healthy', 0.11, 120000, false, ['campaign-2', 'market'],
    'Late thirties, runs a stall she inherited and has quietly doubled. Friendly in a way that is also a negotiation. Remembers every price anyone ever quoted her.'),
  mk('10237749914772934283', 'Hess', 'commander', 'Commander', 'ticking', 11, 'fixated', 0.38, 5000, false, ['campaign-2', 'north', 'command'],
    'Sixty, career officer, recently passed over. Speaks in complete paragraphs. Has started reading disloyalty into ordinary delays.'),
  mk('10237749914772934284', 'Bramble', 'gardener', 'Gardener', 'asleep', 0, 'healthy', 0.08, 300000, false, ['ambient'],
    'Seventy-one, keeps the allotment behind the church. Cheerful, digressive, and occasionally stops mid-sentence when something reminds him of the campaign.'),
  mk('10237749914772934285', 'Sable', 'drifter', 'Drifter', 'idle', 0, 'healthy', 0.14, 90000, true, ['moonlight'],
    'Thirties, no fixed trade, arrives places slightly before she is expected. Answers questions with questions.'),
  mk('10237749914772934286', 'Toll-keeper', 'drifter', 'Drifter', 'suspended', 0, 'healthy', 0.05, 600000, false, ['north'],
    'Ageless in the way of people who sit in booths. Has opinions about everyone who crosses, and shares them for a fee.'),
];

const ALL_MODES = ['physical', 'video_call', 'voice_call', 'instant_message'];
const VIS = ['physical', 'video_call'];

const act = (id, tick, tool, intent, args, obs) =>
  ({ t: 'act', d: { act_id: id, tick, tool, intent, args, observable_in: obs, committed: true, rendered: null } });
const rend = (id, text) => ({ t: 'act_rendered', d: { act_id: id, rendered: { text } } });

const SCRIPT = [
  act('a_88210', 411, 'face', 'check the eastern line', { dir: 'east' }, VIS),
  act('a_88211', 411, 'speak', 'acknowledge Wren, stay watchful', { to: 'Wren' }, ALL_MODES),
  rend('a_88210', 'He glances east.'),
  rend('a_88211', 'Quiet, so far.'),
  { t: 'tick', d: { tick: 411, acts: 2 } },
  { t: 'narration', d: { narration_id: 'n_5511', tick: 411, covers_acts: ['a_88210', 'a_88211'],
    text: 'He straightens as you approach, shears still in hand. "Quiet, so far," he says, and glances east.' } },
  act('a_88212', 412, 'observe', 'read the eastern line', { target: 'eastern_line' }, VIS),
  act('a_88213', 412, 'speak', 'break off, the line is buckling', { to: 'Wren' }, ALL_MODES),
  act('a_88214', 412, 'move_to', 'get to the ridge', { to: 'ridge_east' }, VIS),
  act('a_88215', 412, 'broadcast_strategy', 'the northern road matters more than the ridge', {}, []),
  rend('a_88212', 'He squints east.'),
  rend('a_88213', '…hold on.'),
  rend('a_88214', 'He starts moving.'),
  { t: 'scene_image', d: { image_id: 'scene_1', prompt: 'the ridge at dusk, from where you are standing' } },
  { t: 'tick', d: { tick: 412, acts: 4 } },
  { t: 'narration', d: { narration_id: 'n_5512', tick: 412, covers_acts: ['a_88212', 'a_88213', 'a_88214'],
    text: 'You ask what he sees; before he can answer he is already moving as the eastern line buckles. Somewhere below, a horn.' } },
];

const PERSONAS = [
  'Fifty-three, a former staff sergeant who now runs the night shift on a loading dock. Precise about time to the point of rudeness. Comfortable giving orders, uneasy in conversations with no clear purpose.',
  'Early forties, teaches secondary maths and referees on weekends. Explains things in numbered steps whether or not you asked. Cannot let a wrong claim stand, which has cost him two friendships.',
  'Sixty-eight, retired from a job she will not name. Watches the street from a first-floor window and knows the delivery schedule of every van on it. Kind to strangers, guarded with neighbours.',
  'Twenty-nine, works nights in a hospital laundry and paints in the mornings. Speaks quietly and rarely first. Notices what people do with their hands.',
];
let personaIdx = 0;

const MEM = [
  'The mill road washed out; the crossing moved a mile north and nobody told the garrison.',
  'Hess countermanded the rotation twice in one week, then denied the second order.',
  'Ilse would not take coin for the tobacco, which meant she wanted something later.',
  'A horn from the eastern slope, twice — the signal for ground given, not for contact.',
  'The recruit from the coast asked why the fallback was never written down.',
];

const T = (name, category, description, source, calibrated, modes) =>
  ({ name, category, description, source, calibrated, modes: modes || ALL_MODES, writes_layers: ['action'] });
const C = (name, group, summary, emits, properties, required) =>
  ({ name, group, summary, emits, aliases: [], parameters: { type: 'object', properties }, required });

/* ── the mind, in miniature ───────────────────────────────────────────────────
 *
 * Keyed by address. A node with `doc` is a mapping and opens as fields; one
 * with `text` is prose and opens as text; one with neither is a place you walk
 * into. `notes` are the author's comments — the thing the field form shows
 * beside each input — and `stubborn` marks the one document that refuses a
 * field-by-field save, so the console's `cannot_patch` path is reachable.
 *
 * The root's children are the sections, which is why they have no parent in
 * their address. */
const MIND = {
  canon: { title: 'Canon' },
  'canon/ammo': {
    title: 'Ammo',
    blurb: 'What the guns eat, and what it costs',
    text: '# Ammo\n\nEvery round is machined, and nobody in the cities machines rounds any more.\n',
  },
  'canon/ammo/bolt': {
    title: 'Bolt',
    text: '# Bolt\n\nA hand-loaded slug. Cheap, loud, and it will jam a rail gun if you are\ndesperate enough to try it.\n',
  },
  'canon/ammo/slug': {
    title: 'Slug',
    text: '# Slug\n\nThe standard round. Scarce enough that a full magazine is a statement.\n',
  },
  responses: { title: 'Responses' },
  'responses/accept_then_move_on': {
    title: 'Accept then move on',
    blurb: 'accept',
    notes: {
      template: 'The frozen structural mode — its KV is loaded; the model decodes the NEXT turn into this once the section is selected.',
      examples: 'Provenance lead-ins — the context that PRODUCES the next (accepting) reply. FIXED SHAPE: 4 turns, user → assistant → user → assistant. Final assistant turn is the decode point.',
    },
    doc: {
      id: 'accept_then_move_on',
      category: 'accept',
      description: 'Accepting what the interlocutor offered, admitted, or refused, then letting the moment close without extracting more.',
      template: 'Accept what the interlocutor just offered, admitted, refused, or decided, and let the\nmoment close without demanding more.\n\nTone: settled, unhurried, generous, unbothered.\n',
      examples: [
        {
          note: 'Late apology, no toll charged for it.',
          turns: [
            { role: 'user', content: '"I\'m late — sorry, the train—" I stop myself, exhaling.\n' },
            { role: 'assistant', content: 'A small tip of their head, already unbothered — the seat beside them is patted once, an answer in itself.\n' },
            { role: 'user', content: '"Thank you for not making me grovel through the whole excuse."\n' },
            { role: 'assistant', thinking: 'They will take the thanks lightly and steer straight into the evening, so lateness never becomes a debt you owe.\n' },
          ],
        },
        {
          note: 'A boundary named plainly, respected without probing.',
          turns: [
            { role: 'user', content: '"Can we just— not talk about my ex tonight."\n' },
            { role: 'assistant', content: "The subject drops the instant it's named; their attention simply resettles on you.\n" },
          ],
        },
      ],
    },
  },
  'responses/admit_then_explain': {
    title: 'Admit then explain',
    blurb: 'admit',
    // The one that cannot be patched, so the console's refusal has something to
    // refuse. A real one is a document whose edit failed its own read-back.
    stubborn: true,
    doc: {
      id: 'admit_then_explain',
      category: 'admit',
      description: 'Owning the thing first, without the explanation doing the owning.',
      tags: ['fault', 'repair'],
      template: 'Say the thing you did. Then, and only then, say why.\n',
      examples: [],
    },
  },
  settings: { title: 'Settings' },
  // The projection schema is a collection *and* an entry: it reads whole, and
  // its layers are addressable underneath it. Two of the nine, with the shape
  // the real ones have — a nested budget, a summarisation prompt, and a list of
  // selection groups.
  'settings/projection': {
    title: 'Projection schema',
    text: 'default_policy:\n  preset: high_recall_scope\n\nlayers:\n  # ── World ──\n  - name: world\n    window: 8000\n',
  },
  'settings/projection/world': {
    title: 'World',
    blurb: 'the only cross-timeline layer',
    notes: { description: 'Ingested from the canon topics a world admits.' },
    doc: {
      name: 'world',
      description: 'Shared knowledge about the setting — places, factions, history.\n\nTHE ONLY CROSS-TIMELINE LAYER: one tree across every conversation.\n',
      window: 8000,
      score_threshold: 0.3,
      gather_scope: 'shared',
      decode_priority: 'low',
      ingest_unit: 'documents',
      budget: { priority: 70, max_percent: 20, adaptive: { gain: 2.0, max_percent: 40 } },
      summary: {
        turns: {
          max_tokens: 384,
          scope: 'union',
          assistant: {
            system_prompt: 'You compress documents about a world into one faithful digest.\n',
            user_prompt: 'Digest the documents above, keeping every name exactly.\n',
          },
        },
      },
      groups: [{ id: 'canon', selection: { kind: 'top_k', k: 6 }, budget: { priority: 100 } }],
    },
  },
  'settings/projection/beliefs': {
    title: 'Beliefs',
    doc: {
      name: 'beliefs',
      description: 'What the character holds to be true about the world and itself.\n',
      window: 4000,
      score_threshold: 0.4,
      gather_scope: 'conversation',
      decode_priority: 'normal',
      ingest_unit: 'beliefs',
      budget: { priority: 90, max_percent: 15 },
      groups: [{ id: 'held', selection: { kind: 'top_k', k: 5 }, budget: { priority: 100 } }],
    },
  },
};

/* The keys with a fixed vocabulary, mirroring the daemon's — and, like it,
 * offered only where the value is already one of them. */
const MIND_CHOICES = {
  gather_scope: ['conversation', 'shared'],
  decode_priority: ['low', 'normal', 'high'],
  on_corrupt_turn: ['drop_turn', 'drop_conversation'],
  kind: ['conversation', 'top_k'],
};

const mindNode = (id) => (id ? MIND[id] : { title: 'The mind' });
const mindChildren = (id) => {
  const prefix = id ? id + '/' : '';
  return Object.keys(MIND).filter(
    (k) => k.startsWith(prefix) && !k.slice(prefix.length).includes('/') && k !== id,
  );
};
/* A document as the text it would be stored as. Rough on purpose: the mock's
 * job is to give the text editor something real-shaped to open, not to be a
 * second YAML writer that can disagree with the daemon's. */
const mindText = (node) => {
  if (!node) return '';
  if (node.text != null) return node.text;
  if (!node.doc) return '';
  return Object.entries(node.doc)
    .map(([k, v]) => (typeof v === 'string' && v.includes('\n')
      ? `${k}: |\n${v.replace(/\n$/, '').split('\n').map((l) => '  ' + l).join('\n')}\n`
      : `${k}: ${JSON.stringify(v)}\n`))
    .join('');
};
const mindLabel = (key) => {
  const s = key.replace(/[_-]/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
};
const mindKind = (key, v) => {
  if (typeof v === 'string') {
    if ((MIND_CHOICES[key] || []).includes(v)) return 'choice';
    return v.includes('\n') || v.length > 90 ? 'text' : 'line';
  }
  if (typeof v === 'number') return 'number';
  if (typeof v === 'boolean') return 'bool';
  if (Array.isArray(v)) {
    if (!v.length || v.every((i) => typeof i === 'string')) return 'list';
    if (v.every((i) => i && Array.isArray(i.turns))) return 'conversations';
    if (v.every((i) => i && typeof i === 'object')) return 'rows';
  }
  if (v && typeof v === 'object' && Object.keys(v).length) return 'group';
  return 'raw';
};

/* One field, the same shape the daemon sends — including the nesting, so the
 * recursive controls are reachable with no daemon. */
const mindField = (key, value, notes) => {
  const kind = mindKind(key, value);
  return {
    key,
    label: mindLabel(key),
    kind,
    value,
    note: (notes || {})[key] || null,
    readonly: key === 'id' || key === 'name',
    yaml: kind === 'raw' ? JSON.stringify(value, null, 2) : null,
    choices: kind === 'choice' ? MIND_CHOICES[key] : null,
    fields: kind === 'group'
      ? Object.entries(value).map(([k, v]) => mindField(k, v))
      : null,
    rows: kind === 'rows'
      ? value.map((row) => Object.entries(row).map(([k, v]) => mindField(k, v)))
      : null,
  };
};
/* The same shape the live client throws, so a caller branching on `e.error`
 * cannot tell the two apart. */
const mindErr = (error, detail, status) =>
  Object.assign(new Error(detail), { error, detail, status: status || 404 });

export const MockAPI = {
  async getStatus() {
    await sleep(60);
    return { state: 'ready', detail: 'client-side mock', started_at_ms: Date.now() - 3600000,
      build: 'npcd-mock', mode: 'server-headless',
      loading: { current: 'Ready', progress: 1, completed: ['Mock store'] } };
  },
  /* A synthetic hour, in the column shape the daemon serves. This one DOES
   * fabricate — that is what `?mock` is for, and the header chip says "backend:
   * mock" while it is on. It exists so the performance page can be built
   * against a full engine without one running; the live path reports absence
   * instead, and the two must not be confused. */
  async getTelemetry() {
    const n = 900, period = 2;                    // 30 minutes at the real cadence
    const t = Array.from({ length: n }, (_, i) => i * period);
    const wave = (i, a, b, f) => a + b * Math.sin(i / f);
    const used = t.map((_, i) => Math.round(wave(i, 9000, 1800, 90)));
    const weights = t.map(() => 5400);
    const kv = t.map((_, i) => Math.round(wave(i, 2200, 900, 70)));
    const image = t.map((_, i) => (i % 130 < 40 ? 640 : 0));
    return {
      gpu: { name: 'mock device', compute_cap: '8.6', pcie_gen: 3, pcie_width: 16 },
      model: {
        name: 'Qwen3-30B-A3B', quant: 'Q6_K', params_total: '30B', params_active: '3B',
        repo: 'unsloth/Qwen3-30B-A3B-GGUF', filename: 'Qwen3-30B-A3B-Q6_K.gguf',
        bytes: 25092532800,
      },
      host: { total_mib: 65457, free_mib: 41000, rss_mib: 820 },
      sample_period_s: period,
      engine_connected: true,
      image_queue_state: 'waiting_for_vram',
      uptime_s: 4820,
      series: {
        t,
        vram_total_mib: t.map(() => 24576),
        vram_used_mib: used,
        vram_free_mib: used.map((u) => 24576 - u),
        host_total_mib: t.map(() => 65457),
        host_used_mib: t.map((_, i) => Math.round(wave(i, 24000, 2600, 140))),
        rss_mib: t.map(() => 820),
        weights_mib: weights,
        kv_mib: kv,
        image_mib: image,
        decode_tps: t.map((_, i) => Math.round(wave(i, 430, 90, 55))),
        prefill_tps: t.map((_, i) => Math.round(wave(i, 1100, 320, 33))),
        mean_npcs_per_decode: t.map((_, i) => +wave(i, 3.1, 1.1, 61).toFixed(1)),
        max_batch: t.map(() => 6),
        npcs_active: t.map((_, i) => Math.round(wave(i, 7, 4, 77))),
        ticks_per_sec: t.map((_, i) => +wave(i, 0.45, 0.2, 48).toFixed(2)),
        inbox_depth_p50: t.map((_, i) => Math.round(Math.abs(wave(i, 1, 2, 40)))),
        inbox_depth_p99: t.map((_, i) => Math.round(Math.abs(wave(i, 11, 9, 95)))),
        image_queue_depth: image.map((v) => (v ? 2 : 0)),
      },
    };
  },
  async getSubstrateStorage() {
    const seg = (id, bytes, active) => ({ id, bytes, active: !!active });
    return {
      open: true,
      path: '.substrate',
      listed: true,
      segment_count: 4,
      segments: [seg(1, 67108864), seg(2, 67108864), seg(3, 67108864), seg(4, 21402112, true)],
      total_bytes: 222728704,
      live_chunks: 41882,
      dead_ratio: 0.18,
    };
  },
  /* `getMemoryDump`, not `getMemory`: a character's memory layer claims that
   * name below, and a duplicate key in an object literal silently keeps the
   * last one. */
  async getMemoryDump() {
    const mib = 1024 * 1024;
    return {
      report: null,
      report_age_ms: null,
      host_now: { total_bytes: 65457 * mib, available_bytes: 41000 * mib, free_bytes: 38200 * mib },
      process: { working_set_bytes: 820 * mib, virtual_bytes: 41000 * mib },
    };
  },
  async getMe() {
    if (flag('loggedout')) return null;
    return { user_id: 'u_8812', unique_name: 'Wren', display: 'Johnathan', email: 'you@example.com',
      // No `npc_count` — see §8.3. A total of everything you own is the figure
      // that gives your hidden characters away.
      provider: 'google',
      profile: { description: 'Reads people quickly, talks slowly. Ex-surveyor, so tends to describe places by their edges.',
        gender: 'Male', history: 'Grew up on the coast. Came inland for work and stayed.',
        turn_index: 7, revision: 3 } };
  },
  async getProfile() { return (await this.getMe()).profile; },
  async putProfile(b) { return { ...(await this.getProfile()), ...b, revision: 4 }; },
  async putUniqueName(n) { return { ...(await this.getMe()), unique_name: n }; },
  /* Enough of them to exercise the chooser rather than a list of two: with
   * hundreds, anything that renders every revision at once is the wrong shape. */
  async getProfileHistory() {
    const live = await this.getProfile();
    const revs = [{ revision: live.revision, live: true, tombstoned_ms: null,
      preview: live.description.slice(0, 90) }];
    for (let r = live.revision - 1; r >= 0; r--) {
      revs.push({ revision: r, live: false,
        tombstoned_ms: Date.now() - (live.revision - r) * 36e5,
        preview: `Earlier wording, revision ${r}. Ex-surveyor; talks slowly.` });
    }
    return { revisions: revs };
  },
  async getProfileRevision(n) {
    const live = await this.getProfile();
    if (n === live.revision) return live;
    return { ...live, revision: n, tombstoned_ms: Date.now() - 36e5,
      description: `Earlier wording, revision ${n}. Ex-surveyor; talks slowly.` };
  },
  async restoreProfile(n) {
    const old = await this.getProfileRevision(n);
    return { ...old, revision: (await this.getProfile()).revision + 1, tombstoned_ms: null };
  },

  async listNpcs(f = {}) {
    await sleep(40);
    const tag = (f.tag || '').trim().toLowerCase();
    const items = NPCS.filter((n) => {
      // §8.3 — the entire discretion rule. Without a tag filter hidden NPCs are
      // omitted; with one they match like anything else.
      if (tag) { if (!n.tags.some((t) => t.toLowerCase().includes(tag))) return false; }
      else if (n.hidden) return false;
      if (f.state && f.state !== 'any' && n.state !== f.state) return false;
      if (f.world_id && n.world_id !== f.world_id) return false;
      if (f.q && !n.name.toLowerCase().includes(String(f.q).toLowerCase())) return false;
      return true;
    });
    return { items, next_cursor: null, has_more: false };
  },
  async getNpc(id) {
    const n = NPCS.find((x) => x.npc_id === id);
    if (!n) throw Object.assign(new Error('no such NPC'), { error: 'npc_not_found' });
    return n;
  },
  async createNpc(b) {
    const n = mk(String(Date.now()), b.name || 'New character', b.personality_id || 'loyal-soldier', 'Loyal Soldier',
      'idle', 0, 'healthy', 0.1, 60000, false, [], b.persona_description || '');
    NPCS.push(n); return n;
  },
  async patchNpc(id, b) { const n = await this.getNpc(id); Object.assign(n, b); return n; },
  async deleteNpc(id) { const i = NPCS.findIndex((x) => x.npc_id === id); if (i >= 0) NPCS.splice(i, 1); return null; },
  async setTags(id, tags) { (await this.getNpc(id)).tags = tags; return { ok: true }; },
  async setHidden(id, hidden) { (await this.getNpc(id)).hidden = hidden; return { ok: true }; },
  async perceive(_id, events) { return { accepted: events.length, tick_scheduled: true, preempted: false }; },

  async getBeliefs() {
    return { beliefs: [
      { belief_id: 'hess_word', statement: 'Hess is a man of his word', confidence: 0.72, threshold: 0.85,
        disconfirmation: 0.30, origin: 'authored', under_pressure: true,
        history: [
          { at_world_ms: worldMs() - 86400000 * 40, confidence: 0.95 },
          { at_world_ms: worldMs() - 86400000 * 20, confidence: 0.93 },
          { at_world_ms: worldMs() - 86400000 * 5, confidence: 0.81 },
          { at_world_ms: worldMs(), confidence: 0.72 }] },
      { belief_id: 'north_road', statement: 'The northern road is passable in winter', confidence: 0.95,
        threshold: 0.60, disconfirmation: 0, origin: 'evidence', under_pressure: false, history: [] },
      { belief_id: 'orders', statement: 'An order given badly is still an order', confidence: 0.90,
        threshold: 0.95, disconfirmation: 0.05, origin: 'generated', under_pressure: false, history: [] },
    ] };
  },
  async authorBelief() { return { ok: true }; },
  async deleteBelief() { return null; },
  async getRelationships() {
    return { relationships: [
      { entity_id: 'hess', display: 'Commander Hess', trust: 0.6, affect: 0.2, familiarity: 0.9,
        last_contact_world_ms: worldMs() - 3600000, notes: 'Chain of command.' },
      { entity_id: 'ilse', display: 'Ilse', trust: 0.1, affect: 0.4, familiarity: 0.35,
        last_contact_world_ms: worldMs() - 86400000, notes: 'Sells him tobacco. Overcharges; both know it.' },
      { entity_id: 'wren', display: 'Wren', trust: 0.0, affect: 0.05, familiarity: 0.1,
        last_contact_world_ms: worldMs() - 600000, notes: 'New. Asks direct questions, which he respects.' },
    ] };
  },
  async setRelationship() { return { ok: true }; },
  async getAgency() {
    return { agency: [
      { strategy_id: 'hold_ridge', statement: 'Hold the eastern ridge until relieved', state: 'active',
        parent_id: null, children: ['watch_rotation', 'fallback_named'], salience: 0.88,
        progress_notes: ['Rotation set', 'Fallback to the mill agreed'] },
      { strategy_id: 'watch_rotation', statement: 'Keep a two-hour watch rotation', state: 'active',
        parent_id: 'hold_ridge', children: [], salience: 0.51, progress_notes: [] },
      { strategy_id: 'fallback_named', statement: 'Name a fallback before dark', state: 'finished',
        parent_id: 'hold_ridge', children: [], salience: 0.12, progress_notes: ['The mill'] },
    ] };
  },
  async getMemory() {
    return { items: Array.from({ length: 24 }, (_, i) => ({
      turn: 4412 - i, world_ms: worldMs() - i * 3600000, text: MEM[i % MEM.length], tokens: 120 + (i % 7) * 30 })),
      has_more: true };
  },
  async getModulation() { return { affect: -0.2, threat: 0.66, curiosity: 0.3 }; },
  async setModulation() { return { ok: true }; },

  async getSubstrate() {
    return { layers: [
      { layer: 'perception', turns: 41, tokens: 12400, window: 16000, resident: 88 },
      { layer: 'action', turns: 212, tokens: 31900, window: 16000, resident: 62 },
      { layer: 'agency', turns: 6, tokens: 2100, window: 4000, resident: 100 },
      { layer: 'relationships', turns: 14, tokens: 3800, window: 4000, resident: 100 },
      { layer: 'beliefs', turns: 9, tokens: 2400, window: 4000, resident: 100 },
      { layer: 'memory', turns: 4412, tokens: 918233, window: 8000, resident: 61 },
      { layer: 'interaction', turns: 88, tokens: 19100, window: 16000, resident: 74 },
      { layer: 'environment', turns: 24, tokens: 5200, window: 6000, resident: 100 },
      { layer: 'world', turns: 88, tokens: 21000, window: 8000, resident: 47 },
    ] };
  },
  async getLayer(_id, layer) {
    const P = {
      perception: ['A horn, twice, from the eastern slope.', 'Wind off the ridge; the light going amber.',
        'The line east of the mill gives ground.', 'Movement in the treeline — two, maybe three.'],
      action: ['speak → "Quiet, so far."', 'face → east', 'move_to → ridge_east', 'observe → eastern_line'],
      memory: MEM, world: ['The crown courier has not come in eleven days.', 'Tolls on the north road doubled after the thaw.'],
      environment: ['The light goes amber and the wind drops.', 'Rain starts, fine and cold, from the west.'],
    }[layer] || ['…'];
    return { layer, has_more: false, items: Array.from({ length: 14 }, (_, i) => ({
      turn: 200 - i, world_ms: worldMs() - i * 600000, score: Math.max(0.05, 0.95 - i * 0.05),
      tokens: 90 + (i % 5) * 40, preview: P[i % P.length] })) };
  },
  async getProjection(_id, tick = 412) {
    return { tick: Number(tick), budget: { total: 16000, used: 15214 },
      system_prompt: { mood: 'tense', mood_spiked_at: Number(tick) - 3, template: 'battlefield_urgency',
        sections: ['identity_anchor', 'situation', 'concerns'] },
      layers: [
        { layer: 'perception', gathered: 8, available: 41, tokens: 4120, top_score: 0.94 },
        { layer: 'action', gathered: 5, available: 212, tokens: 2010, top_score: 0.81 },
        { layer: 'beliefs', gathered: 3, available: 9, tokens: 812, top_score: 0.88 },
        { layer: 'relationships', gathered: 2, available: 14, tokens: 540, top_score: 0.77 },
        { layer: 'agency', gathered: 1, available: 6, tokens: 260, top_score: 0.69 },
        { layer: 'memory', gathered: 11, available: 4412, tokens: 3180, top_score: 0.72 },
        { layer: 'world', gathered: 4, available: 88, tokens: 1290, top_score: 0.66 }],
      dropped: [{ layer: 'memory', turns: 6, reason: 'budget' }, { layer: 'world', turns: 9, reason: 'threshold' }] };
  },
  async getMonitor(_id, w = 100) {
    return { band: 'healthy', thresholds: { fixated: 0.35, runaway: 0.55 },
      overlap: Array.from({ length: w }, (_, i) => {
        const t = i / w;
        return { tick: 312 + i, value: +(0.12 + 0.1 * Math.abs(Math.sin(t * 6)) + 0.06 * t).toFixed(3) };
      }) };
  },

  async getEnvironment() {
    return { enabled: true, window_turns: 24,
      system_prompt: 'You describe what happens around a character in Ardh: a northern frontier three years after an inconclusive war. Keep to what could be perceived from where they stand. Never narrate their thoughts or decide their actions.',
      recent: [
        { world_ms: worldMs() - 600000, text: 'Wind off the ridge; the light going amber.' },
        { world_ms: worldMs() - 300000, text: 'A horn, twice, from below the eastern slope.' },
        { world_ms: worldMs() - 60000, text: 'The line east of the mill gives ground.' }] };
  },
  async setEnvironment() { return { ok: true }; },
  async injectEnvironment() { return { ok: true }; },

  async listInteractions(id) {
    const base = { npc_id: id, interlocutor: { kind: 'operator', id: 'u_8812', display: 'Wren' }, state: 'live' };
    return { interactions: [
      { ...base, interaction_id: '4471028855119', mode: 'physical', idle_timeout_secs: 300,
        idle_remaining_secs: 252, act_count: 14, narration_count: 5, opened_world_ms: worldMs() - 900000 },
      { ...base, interaction_id: '4471028855120', mode: 'instant_message', idle_timeout_secs: 86400,
        idle_remaining_secs: 84200, act_count: 6, narration_count: 3, opened_world_ms: worldMs() - 3600000 },
    ] };
  },
  async openInteraction(id, spec) {
    return { interaction_id: String(Date.now()), npc_id: id, mode: spec.mode, state: 'live',
      interlocutor: spec.interlocutor || { kind: 'operator', id: 'u_8812', display: 'Wren' },
      idle_timeout_secs: 300, idle_remaining_secs: 300, act_count: 0, narration_count: 0,
      opened_world_ms: worldMs() };
  },
  async getInteraction(ix) {
    return { interaction_id: ix, npc_id: NPCS[0] ? NPCS[0].npc_id : '1', mode: 'physical', state: 'live',
      interlocutor: { kind: 'operator', id: 'u_8812', display: 'Wren' },
      idle_timeout_secs: 300, idle_remaining_secs: 252, act_count: 14, narration_count: 5,
      opened_world_ms: worldMs() - 900000 };
  },
  async inject() { return { ok: true }; },
  async endInteraction() { return null; },

  streamInteraction(ix, hh) {
    let i = 0, stop = false;
    if (hh.onOpen) hh.onOpen({ interaction_id: ix, mode: 'physical' });
    (async () => {
      while (!stop && i < SCRIPT.length) {
        const f = SCRIPT[i++];
        const fn = { act: hh.onAct, act_rendered: hh.onActRendered, tick: hh.onTick,
          narration: hh.onNarration, scene_image: hh.onSceneImage }[f.t];
        if (fn) fn({ ...f.d, world_ms: worldMs(), at_ms: Date.now() });
        await sleep(f.t === 'narration' ? 1000 : 620);
      }
    })();
    return { cancel: () => { stop = true; } };
  },

  // `q` narrows the listing, as the daemon's does. Nothing in this fixture is
  // `hidden`, so the whole-word reveal has nothing to reveal here — what the
  // argument buys is that the console's filter box is not dead against the
  // mock, which is the kind of difference that gets mistaken for a bug.
  async listWorlds(q) {
    const all = EMPTY ? [] : [{ world_id: 'ardh', name: 'Ardh', public: false,
      setting: 'A kingdom of hill villages on a northern frontier, three years after a war nobody won. Roads are unsafe after dark. The crown is distant and the garrisons are underpaid.',
      npc_count: NPCS.length, time: { world_ms: worldMs(), scale: 60, paused: false },
      zoom_bands: ['strategic', 'regional', 'tactical', 'local'],
      templates: { responses: 'override', moods: 'default' } }];
    return { worlds: narrow(all, q, (x) => [x.world_id, x.name]) };
  },
  async getWorld(w) { return (await this.listWorlds()).worlds.find((x) => x.world_id === w); },
  async setWorld() { return { ok: true }; },
  async setWorldTime() { return { ok: true }; },
  async listPersonalities(q) {
    // Ids are slugs and the anchor is `anchor`, matching
    // `personalities/<id>.yaml`. Both were invented shapes — numbered rows and
    // a `core_identity` field that existed only on the wire — and a console
    // exercised against them is a console tested against nothing real.
    const P = (id, name, anchor, npc_count, doctrine_version, doctrine) => ({
      personality_id: id, name, anchor, npc_count, doctrine_version, doctrine,
      personality: {
        voice: 'Short sentences. Rank and role before names. Silence rather than a guess.',
        processing: 'Weight direct observation over second-hand intel. Distrust a plan with no named fallback.',
        under_pressure: 'Get narrower, not louder. Reduce the problem until one action is obviously next.',
      },
    });
    const all = [
      P('loyal-soldier', 'Loyal Soldier', 'Betrayal is unforgivable. Orders are a contract, not a request.', 3, 4,
        'Flank at 2:1 or not at all. Cross open ground only with a fallback named.'),
      P('merchant', 'Merchant', 'Every exchange is a relationship. Price is memory made numeric.', 1, 2,
        'Never the first number.'),
      P('commander', 'Commander', 'Position is read before people are. Loyalty is assessed, not assumed.', 1, 3,
        'Name a fallback before committing.'),
      P('gardener', 'Gardener', 'Things grow at their own rate. Patience is not passivity.', 1, 1,
        'Prune in the cold.'),
      P('drifter', 'Drifter', 'Attachment is a cost. Observation is free.', 2, 1,
        'Leave before asked.'),
    ];
    return { personalities: narrow(all, q, (x) => [x.personality_id, x.name]) };
  },
  async getPersonality(a) { return (await this.listPersonalities()).personalities.find((x) => x.personality_id === a); },
  async setPersonality() { return { ok: true }; },

  async getLayerSchema() {
    const L = (layer, window, priority, min_percent, selection, masking, score_threshold,
      decode_priority, summarize, description) =>
      ({ layer, window, budget: { priority, min_percent }, selection, masking,
        score_threshold, decode_priority, summarize, description });
    return { layers: [
      L('perception', 16000, 100, 40, 'sequence(recent 12, top-k 8)', 'self-local', 0, 'high', false,
        'API-fed. Maps supersede at the same zoom band; descriptions accumulate.'),
      L('action', 16000, 95, 30, 'sequence(recent 16, top-k 6)', 'self-local', 0, 'high', true,
        'The act stream. Ground truth — everything narrated is read from here.'),
      L('agency', 4000, 80, 20, 'top-k 4', 'self-local', 0.35, 'normal', false,
        'Missions, strategies and sub-goals.'),
      L('relationships', 4000, 75, 20, 'top-k 6', 'self-local', 0.30, 'normal', false,
        'Per-entity calibration. Writable on both planes.'),
      L('beliefs', 4000, 90, 25, 'top-k 5', 'self-local', 0.40, 'normal', false,
        'Write-protected against the action plane. Evidence threshold only.'),
      L('memory', 8000, 60, 15, 'sequence(recent 4, top-k 12)', 'self-local', 0.25, 'low', true,
        'Unbounded. The consolidation target for daydream and sleep folds.'),
      L('interaction', 16000, 100, 50, 'sequence(recent 16, top-k 8)', 'self-local', 0, 'high', true,
        "One timeline per interaction, forked from the NPC's sealed prefix."),
      L('environment', 6000, 50, 10, 'sequence(recent 24)', 'self-local', 0, 'low', false,
        'Sliding window only — continuity of the scene, not recall.'),
      L('world', 8000, 70, 10, 'top-k 6', 'cross-timeline', 0.30, 'low', true,
        'The only unmasked layer: shared facts are retrievable across NPCs.'),
    ] };
  },
  async getTurn(_id, layer, turn) {
    const US = '<|im_start|>user\n', IE = '<|im_end|>\n', AS = '<|im_start|>assistant\n';
    const bodies = {
      perception: ['A horn, twice, from the eastern slope.', 'Wind off the ridge; the light going amber.',
        'The line east of the mill gives ground.', 'Movement in the treeline — two, maybe three.'],
      action: ['speak → "Quiet, so far."', 'face → east', 'move_to → ridge_east', 'observe → eastern_line'],
      beliefs: ['Hess countermanded the rotation twice, then denied the second order. Disconfirmation on "a man of his word" now 0.30 of a 0.85 threshold.',
        'The northern road held through the thaw. Confidence unchanged at 0.95.',
        'An order given badly is still an order — nothing this week tested it.',
        'No new evidence bearing on any standing belief.'],
      memory: MEM, world: ['The crown courier has not come in eleven days.', 'Tolls on the north road doubled after the thaw.'],
    }[layer] || ['The light goes amber and the wind drops.'];
    const user = layer === 'action' ? 'tick 412' : 'Scope';
    const assistant = bodies[turn % bodies.length];
    const ul = Math.max(1, Math.round(user.length / 4)), al = Math.max(1, Math.round(assistant.length / 4));
    // `kv: null` = ethereal — recorded, but not in this turn's own K/V grid.
    const segments = [
      { kind: 'glue', marker: 'user_start', kv: null },
      { kind: 'user', text: user, kv: { offset: 0, len: ul } },
      { kind: 'glue', marker: 'im_end', kv: { offset: ul, len: 2 } },
      { kind: 'glue', marker: 'assistant_start', kv: { offset: ul + 2, len: 3 } },
    ];
    if (layer === 'action') {
      segments.push({ kind: 'thinking', kv: null,
        text: 'The line is giving. Say the reassuring thing, then move — she can follow or not.' });
    }
    segments.push({ kind: 'assistant', text: assistant, kv: { offset: ul + 5, len: al } });
    segments.push({ kind: 'glue', marker: 'im_end', kv: null });
    return { layer, turn, user, assistant, tokens: ul + al,
      text: US + user + IE + AS + assistant, layout: { segments } };
  },
  async probe(_id, text) {
    const q = (text || '').trim();
    const qn = q.length;
    let seed = 1; for (let i = 0; i < q.length; i++) seed += q.charCodeAt(i);
    const jit = (k) => ((seed * k) % 97) / 97;
    const T = (kind, layer, label, base, k, tokens, body) => {
      const score = Math.round((base + jit(k) * 380) * Math.min(1.6, 1 + qn / 220));
      return { kind, layer, label, score, tokens, selected: score > 520, text: body };
    };
    const tiles = [
      T('turn', 'perception', 'A horn, twice, from the eastern slope.', 940, 3, 96,
        'The signal for ground given, not for contact. Second in an hour.'),
      T('belief', 'beliefs', 'Hess is a man of his word', 880, 5, 128,
        'confidence 0.72 · disconfirmation 0.30 / 0.85 — under pressure'),
      T('summary', 'memory', 'summary #412 (compresses turns 388–411)', 720, 7, 180,
        'The week the rotation was countermanded twice; the fallback was never written down.'),
      T('relationship', 'relationships', 'Commander Hess', 610, 11, 84,
        'trust +0.60 · affect +0.20 · familiarity 0.90 — chain of command'),
      T('section', 'system', 'mood · tense', 540, 13, 104,
        'Clipped. Attention divided — part of you is elsewhere, and it shows in what you leave unfinished.'),
      T('turn', 'action', 'move_to → ridge_east', 430, 17, 64, 'intent: get to the ridge before the line folds'),
      T('section', 'system', 'response · battlefield_urgency', 380, 19, 128,
        'Answer in short, load-bearing sentences. Lead with the thing that changes what they do next.'),
      T('turn', 'world', 'The crown courier has not come in eleven days.', 210, 23, 72,
        'World fact, shared across every character in Ardh.'),
    ].sort((a, b) => b.score - a.score);
    await sleep(90);
    return { query_tokens: Math.ceil(qn / 4), budget: { total: 16000, would_use: 15214 }, tiles };
  },

  async getWorldCollections() {
    const S = (id, category, tokens, examples, template) => ({ id, category, tokens, examples, template });
    return { collections: [
      { name: 'response', folder: 'responses/', rule: 'named(selector: response) · locked',
        locked: false, source: 'world override',
        description: 'The structural mode of a reply. Selected once at interaction start by top-k provenance match, then frozen for the entire decode.',
        sections: [
          S('battlefield_urgency', 'combat', 128, 3, 'Answer in short, load-bearing sentences. Lead with the thing that changes what they do next. No preamble.'),
          S('military_briefing', 'combat', 142, 3, 'Situation, then assessment, then recommendation, in that order. Name uncertainties explicitly.'),
          S('merchant_negotiation', 'social', 156, 4, 'Never name the first number. Acknowledge what they want before saying what it costs.'),
          S('casual_conversation', 'social', 118, 3, "Follow the other person's thread. Volunteer detail only when asked or genuinely surprising."),
          S('whispered_conspiracy', 'social', 134, 2, 'Short clauses. Assume you may be overheard. Say the dangerous part last and least directly.'),
          S('storytelling', 'social', 148, 3, 'Set the scene before the event. One concrete sensory anchor per beat, never the same one twice.'),
        ] },
      { name: 'mood', folder: 'moods/', rule: 'named(selector: mood) · spiking',
        locked: false, source: 'defaults',
        description: 'Event-driven and threshold-gated, not drifting. Holds its register until provenance scores a different one above the spike threshold at a barrier, then snaps.',
        sections: [
          S('confident', 'affect', 96, 2, 'Certain, unhurried. Declaratives. No hedging.'),
          S('tense', 'affect', 104, 2, 'Clipped. Attention divided — part of you is elsewhere, and it shows in what you leave unfinished.'),
          S('grieving', 'affect', 112, 2, 'Slower. Ordinary things take effort to name.'),
          S('analytical', 'affect', 98, 2, 'Structure first. Enumerate before you conclude.'),
          S('guarded', 'affect', 101, 2, 'Answer the question asked and no more.'),
        ] },
      { name: 'situation', folder: 'situations/', rule: 'top-k 2', locked: false, source: 'defaults',
        description: 'The current mission, strategy and perception state, surfaced only when relevant.',
        sections: [
          S('under_orders', 'framing', 88, 1, 'You have a standing order and a named fallback.'),
          S('unsupervised', 'framing', 84, 1, 'Nobody is watching. What you do now is yours.'),
        ] },
    ] };
  },
  async getPersonalityCollections() {
    const S = (id, category, tokens, examples, template) => ({ id, category, tokens, examples, template });
    return { collections: [
      { name: 'identity_anchor', folder: 'identities/<name>/anchor.yaml', rule: 'always-visible',
        locked: true, source: 'personality',
        description: 'The always-on compressed self. Structurally resident — it never competes for the gather budget, because it is the prefix the budget is read inside.',
        sections: [S('anchor', 'identity', 186, 0,
          'You are a soldier before you are anything else. An order is a contract. Betrayal is not a setback, it is a category.')] },
      { name: 'identity', folder: 'identities/<name>/*.yaml', rule: 'top-k 3',
        locked: true, source: 'personality',
        description: 'Detail facets of the same self, surfaced only when relevant to the exchange.',
        sections: [
          S('voice', 'identity', 132, 2, 'Short sentences. Rank and role before names. Silence rather than a guess.'),
          S('processing', 'identity', 148, 2, 'Weight direct observation over second-hand intel. Distrust a plan with no named fallback.'),
          S('history', 'identity', 164, 1, 'Twenty years in, three campaigns, one of them the kind nobody writes down.'),
          S('under_pressure', 'identity', 121, 2, 'Get narrower, not louder. Reduce the problem until one action is obviously next.'),
        ] },
      { name: 'doctrine', folder: 'doctrine.yaml', rule: 'always-visible',
        locked: false, source: 'personality · evolves',
        description: 'The one part of the shared layer designed to change. Aggregated from strategic learning across every NPC of this type, then published as a version.',
        sections: [S('current', 'doctrine', 142, 0, 'Flank at 2:1 or not at all. Cross open ground only with a fallback named.')] },
    ] };
  },
  /* ── the authored corpus ──────────────────────────────────────────────────
   *
   * A miniature mind: two sections, a topic with an overview and entries under
   * it, and one response section with the shape the real ones have. Enough that
   * the browser, the text editor, the field form and the conversation editor
   * are all reachable with no daemon — including the two refusals the console
   * branches on, `not_fields` and `cannot_patch`, which are the paths a mock
   * that only ever succeeds would leave untested.
   *
   * Addresses, never paths, exactly as the daemon has it: nothing here knows
   * where a file would live or which extension it would take. */
  async mindList(id) {
    await sleep(60);
    const node = mindNode(id);
    if (!node) throw mindErr('not_found', 'no such place');
    return {
      id: id || '',
      title: node.title,
      has_text: node.doc != null || node.text != null,
      scoped: false,
      children: mindChildren(id).map((cid) => {
        const c = MIND[cid];
        const count = mindChildren(cid).length;
        return {
          id: cid,
          title: c.title,
          // Anything holding something is a collection, whether or not it also
          // has text of its own — a canon topic has both, and so does the
          // projection schema.
          kind: count ? 'collection' : 'entry',
          count,
          chars: mindText(c).length,
          has_text: c.doc != null || c.text != null,
          blurb: c.blurb || null,
        };
      }),
    };
  },

  /* A portrait upload, accepted and forgotten. The fixture has no store, so it
   * gives back an id shaped like a real one — enough for the create flow to
   * finish without a daemon, which is what `?mock=1` is for. */
  async putPortrait(id, file) {
    await sleep(120);
    if (!file || !/^image\//.test(file.type || '')) {
      throw Object.assign(new Error('that is not an image'), { error: 'not_an_image' });
    }
    return { npc_id: id, portrait: { image_id: 'img_0011223344556677.png', origin: 'uploaded' } };
  },

  async getWorldKnowledge() {
    await sleep(60);
    return (await this.mindList('canon')).children;
  },

  async mindEntry(id) {
    await sleep(60);
    const node = mindNode(id);
    if (!node || (node.doc == null && node.text == null)) {
      throw mindErr('not_found', 'nothing written here');
    }
    const text = mindText(node);
    return { id, title: node.title, text, chars: text.length };
  },

  async saveMindEntry(id, text, world, isNew) {
    await sleep(90);
    let node = mindNode(id);
    // `isNew` refuses to land on something that already exists, which is what
    // separates an add from a save.
    if (isNew) {
      if (node) throw mindErr('name_taken', 'something is already called that', 409);
      const name = id.split('/').pop();
      node = MIND[id] = { title: name.charAt(0).toUpperCase() + name.slice(1).replace(/[_-]/g, ' ') };
    }
    if (!node) throw mindErr('not_found', 'no such place');
    // Saving text over a document that has fields drops the field view, which
    // is what editing the file itself means.
    delete node.doc;
    node.text = text;
    return { id, title: node.title };
  },

  async deleteMindEntry(id) {
    await sleep(90);
    const node = mindNode(id);
    if (!node) throw mindErr('not_found', 'no such place');
    delete node.doc;
    delete node.text;
    return null;
  },

  async mindFields(id) {
    await sleep(60);
    const node = mindNode(id);
    if (!node) throw mindErr('not_found', 'no such place');
    if (!node.doc) {
      throw mindErr('not_fields', 'this document is not a set of fields, so it opens as text', 422);
    }
    return {
      id,
      title: node.title,
      fields: Object.entries(node.doc).map(([key, value]) =>
        mindField(key, value, node.notes)),
    };
  },

  async saveMindFields(id, values) {
    await sleep(90);
    const node = mindNode(id);
    if (!node || !node.doc) throw mindErr('not_found', 'no such place');
    if (node.stubborn) {
      throw mindErr(
        'cannot_patch',
        'this document could not be edited field by field without rewriting it',
        409,
      );
    }
    node.doc = { ...values };
    return { id, title: node.title };
  },

  /* Push streams, on a timer. Same contract as the live socket — backlog
   * first, then one frame at a time, and a handle that stops it — so the pane
   * has no idea which side it is talking to. */
  subscribeLogs(onLine, onState) {
    const l = (ts, level, target, msg) => ({ ts, level, target, msg });
    const backlog = [
      l('06:14:02', 'INFO', 'npcd', 'npcd ready — mock backend, no engine loaded'),
      l('06:14:02', 'INFO', 'npcd::api', 'router mounted: 45 routes'),
      l('06:14:03', 'DEBUG', 'substrate', '9 layers declared, 3 collections resolved'),
      l('06:14:09', 'INFO', 'scheduler', 'wave slice 2000ms · admission window 4'),
      l('06:14:11', 'DEBUG', 'tick', 'npc …4281 gate 0.42 → tick scheduled'),
      l('06:14:11', 'TRACE', 'projection', 'gather: 34 turns, 15214/16000 tok, 6 dropped (budget)'),
      l('06:14:12', 'INFO', 'narrator', 'rendered a_88211 in 214ms'),
      l('06:14:14', 'WARN', 'image', 'queue depth 2 — waiting for VRAM headroom'),
      l('06:14:19', 'DEBUG', 'tick', 'npc …4283 preempted (salience 0.91)'),
      l('06:14:20', 'WARN', 'monitor', 'npc …4283 overlap 0.38 → band=fixated'),
      l('06:14:22', 'INFO', 'scheduler', 'batch composition: 3 npcs / decode'),
      l('06:14:26', 'ERROR', 'image', 'job_img_1 abandoned: relief exhausted before slot claim'),
      l('06:14:31', 'DEBUG', 'persistence', 'checkpoint written · 41 records · 812 KiB'),
    ];
    const more = [
      ['DEBUG', 'tick', 'npc …4281 gate 0.42 → tick scheduled'],
      ['TRACE', 'projection', 'gather: 31 turns, 14880/16000 tok, 4 dropped (budget)'],
      ['INFO', 'narrator', 'rendered a_88211 in 197ms'],
      ['DEBUG', 'persistence', 'checkpoint written · 38 records · 764 KiB'],
      ['WARN', 'monitor', 'npc …4283 overlap 0.39 → band=fixated'],
      ['INFO', 'scheduler', 'batch composition: 3 npcs / decode'],
      ['DEBUG', 'tick', 'npc …4285 idle → heartbeat deferred 90s'],
      ['ERROR', 'image', 'job_img_2 abandoned: relief exhausted before slot claim'],
    ];
    backlog.forEach(onLine);
    if (onState) onState('live');
    let n = 0;
    const t = setInterval(() => {
      const [level, target, msg] = more[n % more.length];
      const s = 6 * 3600 + 14 * 60 + 34 + (++n) * 3;
      onLine(l(clock(s), level, target, msg));
    }, 1400);
    return { close: () => clearInterval(t) };
  },

  subscribeEvents(onEvent, onState) {
    const ids = ['1', '3', '4', '6'].map((k) => '1023774991477293428' + k);
    if (onState) onState('live');
    let n = 0;
    const t = setInterval(() => {
      const id = ids[++n % ids.length];
      onEvent(n % 3 === 0
        ? { type: 'npc.tick', npc_id: id, pending_events: n % 7, state: 'ticking' }
        : { type: 'npc.monitor', npc_id: id, overlap: 0.11 + (n % 5) * 0.04,
            band: n % 5 === 4 ? 'fixated' : 'healthy' });
    }, 2600);
    return { close: () => clearInterval(t) };
  },

  async listTools() {
    return { uncalibrated: 1, tools: [
      T('speak', 'speech', 'Say something. Carries intent, not words — the narrator renders it.', 'generic', true),
      T('send_image', 'messaging', 'Send a picture to a named interlocutor. Messaging modes only.', 'generic', true, ['video_call', 'instant_message']),
      T('move_to', 'movement', 'Move to a named location.', 'generic', true),
      T('face', 'movement', 'Turn to face a direction or entity.', 'generic', true),
      T('follow', 'movement', 'Follow an entity.', 'generic', true),
      T('flee', 'movement', 'Break contact and withdraw.', 'generic', true),
      T('gesture', 'gesture', 'Perform a visible gesture.', 'generic', true),
      T('express', 'gesture', 'Show an expression.', 'generic', true),
      T('observe', 'attention', 'Direct attention at something.', 'generic', true),
      T('listen', 'attention', 'Attend to sound.', 'generic', true),
      T('inspect', 'attention', 'Examine an object closely.', 'generic', true),
      T('greet', 'social', 'Acknowledge someone.', 'generic', true),
      T('offer', 'social', 'Offer something.', 'generic', true),
      T('refuse', 'social', 'Decline.', 'generic', true),
      T('threaten', 'social', 'Make a threat.', 'generic', true),
      T('note_concern', 'internal', 'Record a concern. No observable trace.', 'generic', true),
      T('set_intent', 'internal', 'Set a standing intent.', 'generic', true),
      T('broadcast_strategy', 'internal', 'Write upward to the strategy layer.', 'generic', true),
      T('wait', 'meta', 'Do nothing this tick.', 'generic', true),
      T('end_interaction', 'meta', 'Close the interaction.', 'generic', true),
      T('open_gate', 'extension', 'Open a named gate in the world.', 'extension', false),
    ] };
  },
  async calibrateTools() { return { job_id: 'job_cal_1', tools: ['open_gate'] }; },
  async listCommands() {
    return { commands: [
      C('say', 'narration', 'Speak as yourself', 'interaction_event', { text: { type: 'string', description: 'What you say' } }, ['text']),
      C('act', 'narration', 'Perform a physical action', 'interaction_event', { action: { type: 'string', description: 'What you do' } }, ['action']),
      C('scene', 'narration', 'Describe the environment', 'environment_event', { description: { type: 'string' } }, ['description']),
      C('cue', 'narration', 'Force the NPC to act (it does not deliberate)', 'interaction_event', { character: { type: 'string' }, action: { type: 'string' } }, ['action']),
      C('beat', 'narration', 'Steer the narration. Operator-only; never shown to a participant.', 'interaction_event', { description: { type: 'string' } }, ['description']),
      C('damage', 'combat', 'Apply damage to the NPC', 'perception', {
        amount: { type: 'integer', minimum: 1, maximum: 100, description: 'Hit points' },
        source: { type: 'string', description: 'What caused it' },
        location: { type: 'string', enum: ['head', 'torso', 'left_arm', 'right_arm', 'leg'] },
        severity: { type: 'number', minimum: 0, maximum: 1, default: 0.5 } }, ['amount']),
      C('danger', 'combat', 'Raise the perceived threat level', 'perception', { level: { type: 'number', minimum: 0, maximum: 1 } }, ['level']),
      C('daybreak', 'world', 'Advance the world clock to dawn', 'environment_event', {}, []),
      C('weather', 'world', 'Change the weather', 'environment_event', {
        kind: { type: 'string', enum: ['clear', 'rain', 'fog', 'snow', 'storm'] },
        intensity: { type: 'number', minimum: 0, maximum: 1, default: 0.5 } }, ['kind']),
      C('enter', 'world', 'Someone enters the scene', 'perception', { who: { type: 'string' }, from: { type: 'string' } }, ['who']),
      C('give', 'social', 'Hand the NPC an object', 'perception', { item: { type: 'string' }, from: { type: 'string' } }, ['item']),
      C('open_gate', 'extension', 'Open a named gate (registered by the game)', 'environment_event', { gate_id: { type: 'string' } }, ['gate_id']),
    ] };
  },

  async generateDescription() { await sleep(650); return { description: PERSONAS[personaIdx++ % PERSONAS.length], seed: 88213 + personaIdx }; },
  async generateImage() { return { job_id: 'job_img_1', kind: 'image', state: 'queued', progress: 0, queue_position: 2, eta_secs: null }; },
  async listImageModels() {
    return { models: [
      { id: 'sdxl-turbo', display: 'SDXL Turbo', vram_gib: 8, loaded: false, default: true },
      { id: 'sd15', display: 'Stable Diffusion 1.5', vram_gib: 2.8, loaded: false },
      { id: 'wuerstchen', display: 'Würstchen', vram_gib: 3.6, loaded: false }] };
  },
  async getImageQueue() { return { depth: 2, position: 1, state: 'waiting_for_vram', next_run_eta: null }; },
  imageUrl: () => null,
};
