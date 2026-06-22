/* ============================================================================
 * zend-api.mock.js — in-memory mock backend (window.ZendMockAPI)
 * ----------------------------------------------------------------------------
 * Deterministic, daemon-free implementation of the ZendAPI contract
 * (docs/zend_ui_redesign.md §4). Phase-1 builds the whole GUI against this; the
 * Playwright suite drives it with ?mock=1. The live adapter (zend-api.live.js)
 * implements the identical surface against the daemon.
 *
 * Contract notes baked in here (per the §9 decisions):
 *  - Conversation ids are STRINGS.
 *  - seedConversations returns metadata; only the active conv is hydrated.
 *    getConversation(id) lazily hydrates the rest (mirrors GET /:id).
 *  - Projection spans carry only the RAW core {id,region,metric,detail,step,
 *    from,to,total,window}; the UI derives barLeft/winK/etc.
 *  - uploadFiles is PROGRESSIVE (carve + prefill in parts) and runs every
 *    dropped file in PARALLEL; returns { cancel }.
 *  - getFileContent reconstructs file text; binaries would arrive hex-encoded on
 *    the live layer (here the mock just returns inline content).
 * ========================================================================== */
(function () {
  'use strict';

  const J = (a) => a.join('\n');
  const delay = (ms) => new Promise((r) => setTimeout(r, ms));

  const ZendMockAPI = {
    _fileSeq: 1000,
    _spanSeq: 0,
    _fileContent: {},   // fileId -> reconstructed content (getFileContent source)

    // ── Conversations ──────────────────────────────────────────────────────
    // The list the daemon recovered from its redo log. Only the active one is
    // hydrated (history + projection spans + files); the rest carry turn_count
    // metadata and hydrate lazily via getConversation().
    seedConversations(now) {
      const convs = [
        {
          id: '1', title: 'Trace the substrate redo log replay', archived: false,
          updated_ms: now - 60000, turn_count: 2,
          history: [
            { role: 'user', content: 'Trace how the substrate redo log gets replayed on daemon boot.' },
            { role: 'assistant', content: J([
              '<think>',
              'Locate the entry point first — Substrate::recover is what opens the log on boot.',
              '',
              'Each frame is idempotent, so a torn write at the tail is safe to skip on the next boot.',
              '',
              "I'll show the replay loop itself and cite the function rather than just describing it.",
              '</think>', '',
              'On startup the daemon opens the redo log and replays each committed turn before it starts serving requests.', '',
              '<tool_call>{"name":"read_file","arguments":{"path":"crates/substrate/src/redo.rs","range":"1-40"}}</tool_call>', '',
              'The replay happens in `Substrate::recover`:', '',
              '```rust',
              'pub fn recover(path: &Path) -> Result<Self> {',
              '    let mut s = Substrate::open(path)?;',
              '    for frame in s.log.iter()? {',
              '        s.apply(frame?)?;   // idempotent',
              '    }',
              '    Ok(s)',
              '}',
              '```', '',
              'Each frame is **idempotent**, so a torn write at the tail is simply skipped on the next boot — there is no separate fsck pass.',
            ]) },
          ],
        },
        { id: '2', title: 'Why is decode latency spiking under load?', archived: false, updated_ms: now - 9e5, turn_count: 4, history: [] },
        { id: '3', title: 'Add archive endpoint to conversations API', archived: false, updated_ms: now - 36e5, turn_count: 6, history: [] },
        { id: '4', title: 'Explain the tokenizer ChatML decoder', archived: false, updated_ms: now - 72e5, turn_count: 2, history: [] },
        { id: '5', title: 'Scratch notes on WS reconnect backoff', archived: true, updated_ms: now - 18e6, turn_count: 3, history: [] },
      ];
      this.seedProjections(convs[0]);
      convs[0].files = this._seedFiles();
      this._convs = convs;
      return convs;
    },

    // GET /v1/conversations/{id} — lazily hydrate a conversation's history.
    // The active conv (id '1') is already hydrated; others get a canned exchange
    // synthesized from the title so lazy-hydrate-on-select is demonstrable.
    async getConversation(id) {
      await delay(110);
      const panel = this._panelData();
      const seed = (this._convs || []).find((c) => c.id === String(id));
      if (seed && seed.history && seed.history.length) return Object.assign({}, seed, panel);
      const title = seed ? seed.title : 'Conversation';
      return Object.assign({
        id: String(id), title,
        archived: seed ? seed.archived : false,
        updated_ms: seed ? seed.updated_ms : Date.now(),
        turn_count: seed ? seed.turn_count : 2,
        history: [
          { role: 'user', content: title },
          { role: 'assistant', content: J([
            'Here is where that stands.', '',
            'This is a hydrated view of **' + title + '** — recovered from the substrate redo log and split back into role bubbles on the server.',
          ]) },
        ],
      }, panel);
    },

    archiveConversation(id) { return delay(60); },
    unarchiveConversation(id) { return delay(60); },

    // GET /v1/status — the mock daemon is always ready (no model to load).
    getStatus() {
      return Promise.resolve({ state: 'ready', started_at_ms: 0, detail: '', loading: null, build: 'mock' });
    },

    // ── Files (conversation-scoped) ────────────────────────────────────────
    _seedFiles() {
      const files = [
        { id: 1, name: 'redo.rs', ext: 'RS', kind: 'code', size: '18.4 KB', added: '2h ago', content: J([
          '// crates/substrate/src/redo.rs',
          'use std::path::Path;',
          '',
          'pub struct RedoLog { path: PathBuf, file: File }',
          '',
          'impl RedoLog {',
          '    pub fn append(&mut self, frame: &Frame) -> Result<u64> {',
          '        let bytes = frame.encode();',
          '        self.file.write_all(&(bytes.len() as u32).to_le_bytes())?;',
          '        self.file.write_all(&bytes)?;',
          '        self.file.sync_data()?;          // durability barrier',
          '        Ok(self.offset())',
          '    }',
          '',
          '    pub fn iter(&self) -> Result<FrameIter<\'_>> {',
          '        FrameIter::open(&self.path)',
          '    }',
          '}',
        ]) },
        { id: 2, name: 'boot-trace.log', ext: 'LOG', kind: 'log', size: '242 KB', added: '2h ago', content: J([
          '2026-06-21T02:14:09.118Z  INFO zend::daemon: opening substrate at /var/lib/zend/store',
          '2026-06-21T02:14:09.214Z  INFO zend::substrate: redo log: 4128 frames, 18.7 MiB',
          '2026-06-21T02:14:09.402Z DEBUG zend::substrate: replaying frame 1/4128',
          '2026-06-21T02:14:09.880Z  WARN zend::substrate: torn tail frame skipped at offset 18_612_004',
          '2026-06-21T02:14:10.011Z  INFO zend::substrate: recovered 5 conversations, 4127 turns',
          '2026-06-21T02:14:10.115Z  INFO zend::model: weights mmapped (4.31 GiB) in 212ms',
          '2026-06-21T02:14:10.640Z  INFO zend::http: listening on 127.0.0.1:8731',
        ]) },
        { id: 3, name: 'substrate-schema.md', ext: 'MD', kind: 'doc', size: '6.1 KB', added: '1h ago', content: J([
          '# Substrate redo-log format',
          '',
          'Each frame is length-prefixed and idempotent:',
          '',
          '    [u32 len][payload …]',
          '',
          '## Frame kinds',
          '- `0x01` TurnAppend — one ChatML turn',
          '- `0x02` Label      — conversation title',
          '- `0x03` Archive    — soft-delete marker',
          '',
          'Recovery replays frames in order; a short tail frame is treated as EOF.',
        ]) },
        { id: 4, name: 'panic-backtrace.txt', ext: 'TXT', kind: 'text', size: '3.8 KB', added: '47m ago', content: J([
          'thread \'decode-0\' panicked at crates/model/src/kv.rs:218:13:',
          'kv-cache block index out of range: 4097 >= 4096',
          'note: run with `RUST_BACKTRACE=full` for a verbose backtrace',
          '',
          '   0: zend::model::kv::Cache::block',
          '   1: zend::model::decode::step',
          '   2: zend::http::completions::stream',
          '   3: tokio::runtime::task::harness::poll',
        ]) },
      ];
      for (const f of files) this._fileContent[f.id] = f.content;
      return files;
    },

    fmtBytes(b) {
      if (b == null) return '—';
      if (b < 1024) return b + ' B';
      if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
      return (b / 1048576).toFixed(1) + ' MB';
    },
    _kindFor(name) {
      const e = (name.split('.').pop() || '').toLowerCase();
      const code = ['rs', 'js', 'ts', 'tsx', 'jsx', 'py', 'go', 'rb', 'c', 'cpp', 'h', 'java', 'json', 'toml', 'yaml', 'yml', 'sh'];
      const img = ['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp', 'bmp'];
      if (code.includes(e)) return 'code';
      if (e === 'log') return 'log';
      if (['md', 'markdown', 'rst'].includes(e)) return 'doc';
      if (img.includes(e)) return 'img';
      return 'text';
    },
    _mkFile(name, bytes) {
      const ext = (name.split('.').pop() || '').toUpperCase().slice(0, 4);
      const kind = this._kindFor(name);
      const content = kind === 'img' ? ''
        : '// ' + name + '\n(uploaded · ' + this.fmtBytes(bytes) + ' · reconstructed from token-string ranges)';
      const id = (this._fileSeq += 1);
      this._fileContent[id] = content;
      return { id, name, ext: ext || '·', kind, size: this.fmtBytes(bytes), added: 'just now', content };
    },

    // POST /v1/conversations/{id}/files — progressive carve + prefill, one bar
    // per file, all files in parallel. Returns a cancel handle.
    uploadFiles(convId, descriptors, handlers) {
      handlers = handlers || {};
      const timers = [];
      const results = [];
      let remaining = descriptors.length;
      let cancelled = false;
      if (!remaining) { Promise.resolve().then(() => handlers.onAllDone && handlers.onAllDone([])); return { cancel() {} }; }

      descriptors.forEach((d) => {
        const sizeKB = (d.size || 4096) / 1024;
        const totalParts = Math.max(1, Math.min(24, Math.round(sizeKB / 8) + 1));
        const meta = this._mkFile(d.name, d.size);
        if (handlers.onFileStart) handlers.onFileStart(meta.id, d.name, totalParts);
        let part = 0;
        const timer = setInterval(() => {
          if (cancelled) { clearInterval(timer); return; }
          if (part >= totalParts) {
            clearInterval(timer);
            if (handlers.onFileDone) handlers.onFileDone(meta.id, meta);
            results.push(meta);
            remaining -= 1;
            if (remaining <= 0 && handlers.onAllDone) handlers.onAllDone(results);
            return;
          }
          if (handlers.onPart) handlers.onPart(meta.id, part, totalParts);
          part += 1;
        }, 110);
        timers.push(timer);
      });
      return { cancel: () => { cancelled = true; timers.forEach(clearInterval); } };
    },

    // GET /v1/conversations/{id}/files/{fileId} — reconstructed content.
    async getFileContent(convId, fileId) {
      await delay(80);
      return this._fileContent[fileId] != null ? this._fileContent[fileId] : '';
    },
    deleteFile(convId, fileId) { return delay(60); },

    // ── Projection events ──────────────────────────────────────────────────
    // One event per decode: the decode-span throughput plus the composition of
    // the materialized context provenance selected, bucketed by category. Shape
    // mirrors candle-conversation's ProjectionEvent (system → section groups →
    // turns); the UI derives the bar map / legend / readouts from it.
    mkProjEvent(conv, region) {
      const rnd = (a, b) => a + Math.floor(Math.random() * (b - a));
      conv._substrate = (conv._substrate || 42000) + rnd(900, 2100);
      const buckets = [
        { label: 'system', kind: 'system', tokens: 320 },
        { label: 'code_read', kind: 'section', tokens: rnd(1800, 9200) },
        { label: 'repo_map', kind: 'section', tokens: rnd(400, 2600) },
        { label: 'conversation', kind: 'turns', tokens: rnd(600, 4200) },
      ].filter((b) => b.tokens > 0);
      const materialized = buckets.reduce((a, b) => a + b.tokens, 0);
      const start = conv._projTok || 0;
      const gen = rnd(48, 280);
      const end = start + gen;
      conv._projTok = end;
      const tps = 36 + Math.random() * 18;
      return {
        id: (this._spanSeq += 1),
        region: region || 'answer',
        step: 't=' + end,
        start_token: start,
        end_token: end,
        seconds: gen / tps,
        tokens_per_second: tps,
        materialized_tokens: materialized,
        substrate_tokens: Math.max(conv._substrate, materialized),
        buckets,
        // The system prompt in materialized order: structural glue (system +
        // tool-block envelopes, including the CLOSING markers) interleaved with
        // content sections and the code_read collection (most files skipped),
        // then the selected turns.
        selection: {
          system: [
            { kind: 'glue', name: 'system_open', content: '<|im_start|>system\n', tokens: 3 },
            { kind: 'section', name: 'agent_identity', tokens: 220 },
            { kind: 'section', name: 'tool_protocol', tokens: 100 },
            { kind: 'glue', name: 'tools_open', content: '\n<tools>\n', tokens: 4 },
            { kind: 'collection', name: 'code_read', sections: [
              { name: 'src/lib.rs', tokens: rnd(400, 1200), selected: true },
              { name: 'src/tensor.rs', tokens: rnd(400, 1200), selected: true },
              { name: 'src/backend.rs', tokens: rnd(300, 900), selected: false },
              { name: 'src/ops/add.rs', tokens: rnd(200, 700), selected: false },
              { name: 'src/ops/matmul.rs', tokens: rnd(200, 700), selected: false },
            ] },
            { kind: 'glue', name: 'tools_close', content: '</tools>\n', tokens: 3 },
            { kind: 'glue', name: 'system_close', content: '<|im_end|>\n', tokens: 2 },
          ],
          turns: [
            // Lower-layer memory tiers projected into the window (no bodies in GUI).
            { layer: 'repo_map', group: 'files', index: 0, role: 'assistant', tokens: rnd(180, 360) },
            { layer: 'repo_map', group: 'files', index: 1, role: 'assistant', tokens: rnd(180, 360) },
            { layer: 'code_reading', group: 'scopes', index: 0, role: 'assistant', tokens: rnd(300, 600) },
            { layer: 'code_reading', group: 'scopes', index: 1, role: 'assistant', tokens: rnd(300, 600) },
            { layer: 'code_reading', group: 'scopes', index: 2, role: 'assistant', tokens: rnd(300, 600) },
            // Dialogue turns (rendered from the GUI message history, not these).
            { layer: 'dialogue', group: 'primary_conversation', index: 0, role: 'user', tokens: rnd(40, 120) },
            { layer: 'dialogue', group: 'primary_conversation', index: 0, role: 'assistant', tokens: rnd(300, 900) },
          ],
        },
      };
    },
    seedProjections(conv) {
      const asst = conv.history.find((m) => m.role === 'assistant');
      if (!asst) return;
      // One event per recovered decode (the hydrated turn had reasoning, so
      // seed a think-phase and an answer-phase dot).
      asst.spans = [this.mkProjEvent(conv, 'think'), this.mkProjEvent(conv, 'answer')];
    },

    // ── Chat completion (SSE-like stream) ──────────────────────────────────
    streamChatCompletion(conv, text, opts, handlers) {
      const reply = this.cannedReply(text, opts);
      const tokens = reply.match(/\S+\s*|\s+/g) || [reply];
      let i = -1, started = false, acc = '', thinkEmitted = false;
      const timer = setInterval(() => {
        if (!started) { started = true; handlers.onStatus(''); return; }
        i++;
        if (i >= tokens.length) {
          clearInterval(timer);
          handlers.onProjection(this.mkProjEvent(conv, 'answer')); // decode-end event (final t/s)
          handlers.onDone();
          return;
        }
        acc += tokens[i];
        handlers.onToken(tokens[i]);
        // A think-phase event lands on the timeline the moment reasoning closes.
        if (!thinkEmitted && acc.includes('</think>')) { thinkEmitted = true; handlers.onProjection(this.mkProjEvent(conv, 'think')); }
        if (i % 7 === 0 && handlers.onLog) handlers.onLog();
      }, 34);
      return { cancel: () => clearInterval(timer) };
    },

    thinkFor(userText, effort) {
      const t = (userText || '').toLowerCase();
      let lines;
      if (t.includes('weak') || t.includes('fragile') || t.includes('risk'))
        lines = ['They want the risky areas, not a feature tour.',
          "The parts I'm least sure about: the WS reconnect backoff with no jitter, the redo-log tail invariant, and the titler racing decode.",
          "I'll frame each as something concretely fixable rather than a vague worry."];
      else if (t.includes('tokenizer') || t.includes('chatml'))
        lines = ['This is about how stored turns decode back out.',
          'The detail that matters: the decoder strips the special tokens and leaves bare role markers on their own lines.',
          "So I'll show the decoded shape, then the split-on-role-line step we use when hydrating."];
      else
        lines = ['I should walk the request lifecycle end to end.',
          'Start from the redo log giving durability, then the per-request path: look up by conv_id, append the user turn, stream over SSE.',
          'Persist the assistant turn on [DONE]. Keep it tight and offer to go deeper on any step.'];
      const n = Math.max(1, Math.min(lines.length, (effort == null ? 2 : effort)));
      return '<think>\n' + lines.slice(0, n).join('\n\n') + '\n</think>\n\n';
    },

    cannedReply(text, opts) {
      opts = opts || {};
      const A = (a) => a.join('\n');
      const t = (text || '').toLowerCase();
      const think = (opts.think === false || opts.effort === 0) ? '' : this.thinkFor(text, opts.effort);
      const v = (opts.verbosity == null) ? 2 : opts.verbosity;
      let body;
      if (t.includes('weak') || t.includes('fragile') || t.includes('risk')) {
        body = [
          'A few areas stand out as fragile right now:', '',
          '- **WS reconnect backoff** doubles without a jitter window, so every client retries on the same beat after a daemon restart.',
          '- **Redo log tail** assumes the last frame is either whole or absent — a half-flushed `len` prefix would currently panic in `iter()`.',
          '- **Titler races decode**: if `[DONE]` lands before the label is written, the sidebar keeps the placeholder title until the next list poll.',
        ];
        if (v >= 3) body.push('', 'Want me to open the relevant files and propose fixes for any of these?');
      } else if (t.includes('tokenizer') || t.includes('chatml')) {
        body = [
          'The decoder strips the special tokens and leaves literal role markers on their own lines:', '',
          '```text', '<|im_start|>user', 'hello<|im_end|>', '<|im_start|>assistant', 'hi there<|im_end|>', '```', '',
          'So when we hydrate a stored turn we split on bare `user` / `assistant` lines to recover one bubble per role.',
        ];
      } else {
        body = [
          'Here is the short version.', '',
          'The daemon stores each conversation as an append-only sequence of turns in the substrate redo log, so nothing is lost across a restart — on boot it replays the log and rehydrates the in-memory index.', '',
          'When a request arrives it:', '',
          '- looks up (or creates) the conversation by `conv_id`',
          '- appends the user turn, then streams the model output back over SSE',
          '- writes the assistant turn once `[DONE]` is reached',
        ];
        if (v >= 3) body.push('', 'I can point you at the exact functions for any of those steps.');
        if (v >= 4) body.push('', 'There is also a background titler that labels the conversation in parallel with decode, and a compaction pass that rolls old turns into a memory summary once the window fills.');
      }
      if (v <= 0) body = body.slice(0, 1);
      else if (v === 1) body = body.slice(0, Math.min(body.length, 5));
      return think + A(body);
    },

    // ── Windowed substrate ─────────────────────────────────────────────────
    // { content: name→authored text, glue: dialect framing markers }. Section
    // names match mkProjEvent's selection; the panel shows a section's text when
    // expanded and renders the glue between rows. Returned as part of
    // getConversation (first-class), so the panel needs no extra fetch.
    _panelData() {
      const content = {
        agent_identity: J([
          'You are Zen-Code, the coding agent embedded in the Zend daemon.',
          'You operate over a Rust workspace and answer questions about it precisely.', '',
          'Guidelines:',
          '- Cite exact files, functions, and line ranges.',
          '- Keep answers tight; expand only when asked.',
          '- Never fabricate APIs — if unsure, say so and offer to look.',
        ]),
        tool_protocol: J([
          'Tools are invoked with <tool_call> blocks and return <tool_result> blocks.', '',
          'Available: read_file(path, range?), grep(pattern, glob?), run_tests(crate?).',
          'One tool call per block; wait for the result before the next.',
        ]),
        'src/lib.rs': J([
          '// crates/substrate/src/lib.rs  (Substrate::recover, retrieved)',
          'pub fn recover(path: &Path) -> Result<Self> {',
          '    let mut s = Substrate::open(path)?;',
          '    for frame in s.log.iter()? { s.apply(frame?)?; }   // idempotent',
          '    s.index.rebuild();',
          '    Ok(s)',
          '}',
        ]),
        'src/tensor.rs': J([
          '// crates/substrate/src/tensor.rs  (retrieved)',
          'impl Tensor {',
          '    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {',
          '        self.backend.matmul(self, rhs)',
          '    }',
          '}',
        ]),
        'src/backend.rs': J(['// src/backend.rs — not selected for this projection']),
        'src/ops/add.rs': J(['// src/ops/add.rs — not selected for this projection']),
        'src/ops/matmul.rs': J(['// src/ops/matmul.rs — not selected for this projection']),
      };
      const glue = {
        system_start: '<|im_start|>system\n', system_end: '<|im_end|>\n',
        user_start: '<|im_start|>user\n', user_end: '<|im_end|>\n',
        assistant_start: '<|im_start|>assistant\n', assistant_end: '<|im_end|>\n',
      };
      // Memory-tier turn bodies, keyed by `group::index` (matches mkProjEvent),
      // each split into its user + assistant halves (the GUI frames them).
      const turnContent = {
        'files::0': { user: 'Repository index — `crates/`:', assistant: J(['crates/candle-core/src/  →  lib.rs, tensor.rs, device.rs, ops/', 'crates/candle-nn/src/   →  kv_cache/, layers.rs']) },
        'files::1': { user: 'Repository index — `crates/` (cont.):', assistant: J(['crates/candle-transformers/src/  →  batched_inference.rs, models/', 'crates/candle-kernels/src/       →  paged-decode/, quantized/']) },
        'scopes::0': { user: 'Scope: substrate/src/lib.rs :: Substrate::recover', assistant: J(['pub fn recover(path: &Path) -> Result<Self> {', '    let mut s = Substrate::open(path)?;', '    for frame in s.log.iter()? { s.apply(frame?)?; }', '    Ok(s)', '}']) },
        'scopes::1': { user: 'Scope: substrate/src/redo.rs :: FrameIter::next', assistant: J(['fn next(&mut self) -> Option<Result<Frame>> {', '    let len = self.read_len().ok()??;', '    if self.remaining() < len { return None; }', '    Some(self.read_frame(len))', '}']) },
        'scopes::2': { user: 'Scope: substrate/src/tensor.rs :: Tensor::matmul', assistant: J(['pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {', '    self.backend.matmul(self, rhs)', '}']) },
      };
      return { sectionContent: content, glue, turnContent, targetLayer: 'dialogue' };
    },

    // ── Live logs ──────────────────────────────────────────────────────────
    mkLog(level, target, msg, offsetMs) {
      const d = new Date(Date.now() + (offsetMs || 0));
      const ts = [d.getHours(), d.getMinutes(), d.getSeconds()].map((n) => String(n).padStart(2, '0')).join(':');
      return { ts, level, target, msg };
    },
    seedLogs() {
      const pool = [
        ['INFO', 'zend::daemon', 'substrate recovered: 5 conversations, 4128 turns'],
        ['INFO', 'zend::http', 'listening on 127.0.0.1:8731'],
        ['DEBUG', 'zend::model', 'weights mmapped (4.31 GiB) in 212ms'],
        ['INFO', 'zend::model', 'kv-cache warm: 4096 ctx, 32 layers'],
        ['TRACE', 'zend::tokenizer', 'encode bpe merges=151643'],
        ['DEBUG', 'zend::http', 'GET /v1/conversations?include_archived=true'],
        ['INFO', 'zend::http', 'POST /v1/chat/completions conv_id=1'],
        ['TRACE', 'zend::decode', 'sampler temp=0.7 top_p=0.95 top_k=40'],
        ['DEBUG', 'zend::substrate', 'append turn conv=1 bytes=2184'],
        ['WARN', 'zend::decode', 'prefill batch truncated: ctx pressure 96%'],
        ['INFO', 'zend::titler', 'labelled conv=1 "Trace the substrate redo log replay"'],
        ['TRACE', 'zend::ws', 'logs client connected (1 active)'],
      ];
      return pool.map((p, i) => this.mkLog(p[0], p[1], p[2], -((pool.length - i) * 1300)));
    },
    _logSamples: [
      ['DEBUG', 'zend::http', 'GET /v1/status'],
      ['TRACE', 'zend::decode', 'tok/s={n} batch=1'],
      ['DEBUG', 'zend::substrate', 'flush redo log fsync=ok'],
      ['INFO', 'zend::http', 'GET /v1/conversations 200 {m}ms'],
      ['TRACE', 'zend::ws', 'heartbeat ping/pong'],
      ['DEBUG', 'zend::model', 'kv-cache evict 1 block (lru)'],
    ],
    nextLogLine() {
      const s = this._logSamples[Math.floor(Math.random() * this._logSamples.length)];
      const msg = s[2].replace('{n}', 38 + Math.floor(Math.random() * 9)).replace('{m}', 3 + Math.floor(Math.random() * 6));
      return this.mkLog(s[0], s[1], msg, 0);
    },
    subscribeLogs(onLine, everyMs) {
      const id = setInterval(() => onLine(this.nextLogLine()), everyMs || 2600);
      return () => clearInterval(id);
    },
  };

  window.ZendMockAPI = ZendMockAPI;
})();
