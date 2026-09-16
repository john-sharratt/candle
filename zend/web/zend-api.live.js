/* ============================================================================
 * zend-api.live.js — live daemon adapter (window.ZendLiveAPI)
 * ----------------------------------------------------------------------------
 * Implements the ZendAPI contract (docs/zend_ui_redesign.md §4) against the
 * daemon's HTTP/SSE/WebSocket endpoints.
 *
 * Implemented (Phase 2.1 — endpoints that exist today, validated by the
 * gui_api_harness integration test):
 *   - seedConversations / getConversation   GET /v1/conversations[/{id}]
 *   - archiveConversation (one-way)          POST …/archive
 *   - getProjectionDetail                    GET …/{id}/projections/{turn}/{event}
 *   - getProjectionContext                   POST …/{id}/projection-context
 *   - streamChatCompletion (token + status + think + prefill)  POST /v1/chat/completions (SSE)
 *   - subscribeLogs / seedLogs               WS /ws/logs (structured JSON frames)
 *   - getToolSchemas                         GET /v1/substrate/tools
 *
 * A conversation's history carries its projection points light (the fields
 * the timeline draws, plus each point's `turn`/`event` address). The projection
 * panel fetches what it shows when it opens: a point in full with its context
 * (getProjectionDetail), or — for a point streamed live, which arrives in full
 * but unaddressed — the context alone (getProjectionContext).
 * ========================================================================== */
(function () {
  'use strict';

  const ni = (name) => () => { throw new Error('ZendLiveAPI.' + name + ' not implemented yet (Phase 2)'); };

  async function getJSON(path) {
    const r = await fetch(path, { headers: { accept: 'application/json' } });
    if (!r.ok) throw new Error('GET ' + path + ' -> ' + r.status);
    return r.json();
  }
  async function postVoid(path) {
    const r = await fetch(path, { method: 'POST' });
    if (!r.ok && r.status !== 204) throw new Error('POST ' + path + ' -> ' + r.status);
  }
  const enc = (id) => encodeURIComponent(id);

  const ZendLiveAPI = {
    // ── conversations ──────────────────────────────────────────────────────
    async seedConversations() {
      const body = await getJSON('/v1/conversations?include_archived=true');
      return (body.conversations || []).map((e) => ({
        id: String(e.id),
        title: e.label || 'Conversation',
        archived: !!e.archived,
        // Server-supplied creation-order rank (monotonic, not a clock). Conv
        // ids are random u64s, so this is the only reliable sort key.
        updated_ms: Number(e.updated_ms) || 0,
        turn_count: e.turn_count || 0,
        history: [],
      }));
    },
    async getConversation(id) {
      // The daemon returns role-split, /no_think-stripped bubbles (decision 9),
      // each assistant bubble's projection points light.
      const body = await getJSON('/v1/conversations/' + enc(id));
      return {
        id: String(id),
        title: body.title,
        // `thinking` ({tokens, exact}) is the turn's reasoning length; the UI
        // keeps one per think block, in order.
        history: (body.messages || []).map((m) => ({ role: m.role, content: m.content, no_think: !!m.no_think, thinking: m.thinking ? [m.thinking] : [], tool_tokens: m.tool_tokens || [], spans: m.spans || [], files: m.files || [] })),
        uploads: body.uploads || [],
        // The composer dials this conversation last ran at, as levels. Absent
        // for one that has never taken a turn — the composer keeps its own.
        dials: body.dials || null,
      };
    },
    archiveConversation(id) { return postVoid('/v1/conversations/' + enc(id) + '/archive'); },

    // One recorded projection point in full — its selection and materialized
    // spine — with the panel context beside it.
    async getProjectionDetail(convId, turn, event) {
      const body = await getJSON('/v1/conversations/' + enc(convId) + '/projections/' + Number(turn) + '/' + Number(event));
      return Object.assign({ span: body.span }, panelContext(body));
    },
    // The panel context alone, for a point already held in full: `turns` is
    // its `selection.turns`, sent as they are.
    async getProjectionContext(convId, turns) {
      const r = await fetch('/v1/conversations/' + enc(convId) + '/projection-context', {
        method: 'POST',
        headers: { 'content-type': 'application/json', accept: 'application/json' },
        body: JSON.stringify({ turns: turns || [] }),
      });
      if (!r.ok) throw new Error('POST projection-context -> ' + r.status);
      return panelContext(await r.json());
    },

    // GET /v1/status — daemon loading state (drives the startup overlay). If the
    // daemon isn't reachable yet, report a synthetic "connecting" loading state.
    getStatus() {
      return getJSON('/v1/status').catch(() => ({
        state: 'loading',
        started_at_ms: 0,
        detail: 'connecting to daemon…',
        loading: { current: 'Connecting', progress: 0, completed: [] },
      }));
    },

    // GET /v1/substrate/tools — each tool's argument JSON Schema, by name. The
    // tool-call cards list every parameter from it, including the ones a call
    // left at their defaults.
    async getToolSchemas() {
      const body = await getJSON('/v1/substrate/tools');
      const out = {};
      (body.tools || []).forEach((t) => { out[t.name] = t.parameters || null; });
      return out;
    },

    // ── chat completion (SSE: status events + OpenAI chunk deltas) ──────────
    streamChatCompletion(conv, text, opts, handlers) {
      // Only real chat turns go to the daemon. The history also holds non-chat
      // events — notably the inline `upload` tile ({role:'upload'}) startUpload
      // drops in — whose role isn't a valid `Role` (system|user|assistant). Left
      // in, the daemon's JSON extractor rejects the whole request (422) before
      // the handler runs, so the send silently no-ops: the exact "type a message
      // after uploading and nothing happens" failure.
      const CHAT_ROLES = { user: 1, assistant: 1, system: 1 };
      const messages = (conv.history || [])
        .filter((m) => !m.streaming && CHAT_ROLES[m.role])
        .map((m) => ({ role: m.role, content: m.content }));
      const controller = new AbortController();
      const payload = {
        model: 'zen-code',
        stream: true,
        messages,
        conv_id: String(conv.id),
        effort: opts ? opts.effort : undefined,
        verbosity: opts ? opts.verbosity : undefined,
        think: opts ? opts.think : undefined,
        tools: opts ? opts.tools : undefined,
      };
      fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      }).then((resp) => {
        // **A stream that failed is not a stream that finished.** Every one of
        // these used to call `onDone`, so an HTTP 500, a body-less response and
        // a mid-stream socket drop all rendered as a completed answer — an
        // empty bubble and no way to tell whether the model had nothing to say
        // or the daemon had died. The caller gets `onError` and decides;
        // `onDone` still runs after it, so the composer always unlocks.
        // Any status but 200 means no turn started: every request that reaches
        // the daemon's handler is answered 200 and streamed. A 408 from the edge
        // on a slow uplink is one of these.
        if (!resp.ok) {
          fail(handlers, 'The request failed with HTTP ' + resp.status + ' before the turn started.', false);
          return;
        }
        if (!resp.body) {
          fail(handlers, 'The response arrived without a body.', true);
          return;
        }
        const reader = resp.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        let sawFrame = false;
        const pump = () => reader.read().then(({ done, value }) => {
          if (done) {
            // A stream that closed without ever sending a frame produced no
            // answer at all. The daemon logs why; the user needs to know it
            // happened.
            if (!sawFrame) {
              fail(handlers, 'The response ended before it started. The daemon logged the reason.', true);
              return;
            }
            handlers.onDone();
            return;
          }
          buf += dec.decode(value, { stream: true });
          let nl;
          while ((nl = buf.indexOf('\n\n')) !== -1) {
            const frame = buf.slice(0, nl);
            buf = buf.slice(nl + 2);
            sawFrame = true;
            handleFrame(frame, handlers);
          }
          return pump();
        }).catch((e) => {
          // An abort is the user pressing stop, not a failure.
          if (e && e.name === 'AbortError') { handlers.onDone(); return; }
          fail(handlers, 'The response stream broke: ' + errText(e), true);
        });
        pump();
      }).catch((e) => {
        if (e && e.name === 'AbortError') { handlers.onDone(); return; }
        fail(handlers, 'Could not reach the daemon: ' + errText(e), false);
      });
      return { cancel: () => controller.abort() };
    },

    // ── live logs (structured JSON frames over WS) ─────────────────────────
    seedLogs() { return []; },   // backlog arrives on WS connect
    subscribeLogs(onLine) {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      let ws = null, retry = 1000, closed = false, timer = null;
      const setStatus = (s) => { try { window.__ZEND_LOG_WS__ = s; } catch (_) {} };
      const schedule = () => {
        if (closed) return;
        clearTimeout(timer);
        timer = setTimeout(connect, retry);
        retry = Math.min(retry * 2, 16000);
      };
      const connect = () => {
        if (closed) return;
        setStatus('wait');
        try { ws = new WebSocket(proto + '//' + location.host + '/ws/logs'); } catch (_) { schedule(); return; }
        ws.onopen = () => { retry = 1000; setStatus('ok'); };
        ws.onmessage = (ev) => { try { onLine(JSON.parse(ev.data)); } catch (_) {} };
        ws.onclose = () => { setStatus('err'); schedule(); };  // reconnect with backoff
        ws.onerror = () => { setStatus('err'); };
      };
      connect();
      return () => { closed = true; clearTimeout(timer); try { if (ws) ws.close(); } catch (_) {} };
    },
    mkLog(level, target, msg) {
      const d = new Date();
      const ts = [d.getHours(), d.getMinutes(), d.getSeconds()].map((n) => String(n).padStart(2, '0')).join(':');
      return { ts, level, target, msg };
    },
    nextLogLine() { return null; },   // live logs come from the daemon via WS

    // ── conversation files (§2.5) ──────────────────────────────────────────
    uploadFiles(convId, files, handlers) {
      handlers = handlers || {};
      const form = new FormData();
      [].slice.call(files).forEach((f) => form.append('file', f, f.name));
      const controller = new AbortController();
      const metas = [];
      fetch('/v1/conversations/' + enc(convId) + '/files', { method: 'POST', body: form, signal: controller.signal })
        .then((resp) => {
          if (!resp.ok || !resp.body) { if (handlers.onAllDone) handlers.onAllDone([]); return; }
          const reader = resp.body.getReader();
          const dec = new TextDecoder();
          let buf = '';
          const pump = () => reader.read().then(({ done, value }) => {
            if (done) { if (handlers.onAllDone) handlers.onAllDone(metas); return; }
            buf += dec.decode(value, { stream: true });
            let nl;
            while ((nl = buf.indexOf('\n\n')) !== -1) {
              const frame = buf.slice(0, nl);
              buf = buf.slice(nl + 2);
              handleUploadFrame(frame, handlers, metas);
            }
            return pump();
          }).catch(() => { if (handlers.onAllDone) handlers.onAllDone(metas); });
          pump();
        }).catch(() => { if (handlers.onAllDone) handlers.onAllDone([]); });
      return { cancel: () => controller.abort() };
    },
    async getFileContent(convId, fileId) {
      const r = await fetch('/v1/conversations/' + enc(convId) + '/files/' + enc(fileId));
      if (!r.ok) return '';
      return r.text();
    },
    deleteFile(convId, fileId) {
      return fetch('/v1/conversations/' + enc(convId) + '/files/' + enc(fileId), { method: 'DELETE' }).then(() => {});
    },

    // ── not yet implemented (per-phase) ────────────────────────────────────
    mkProjEvent: ni('mkProjEvent'),
  };

  // The panel context as the UI keeps it: section text by name, and turn bodies
  // keyed by group::timeline::index — one group holds many conversations
  // (code_read: one per file) and turn indices repeat across them, so the
  // timeline is a load-bearing part of the key. `text` is absent for turns
  // whose Tokens record was lost; consumers fall back to the halves.
  function panelContext(body) {
    const sectionContent = {};
    (body.section_content || []).forEach((s) => { sectionContent[s.name] = s.content; });
    const turnContent = {};
    (body.turn_content || []).forEach((t) => { turnContent[t.group + '::' + t.timeline + '::' + t.index] = { text: t.text, user: t.user, assistant: t.assistant, layout: t.layout }; });
    return { glue: body.glue || null, sectionContent, turnContent, targetLayer: body.target_layer || '' };
  }

  // Parse one upload SSE frame -> the upload handlers.
  function handleUploadFrame(frame, handlers, metas) {
    let event = null;
    const dataLines = [];
    frame.split('\n').forEach((line) => {
      if (line.indexOf('event:') === 0) event = line.slice(6).trim();
      else if (line.indexOf('data:') === 0) dataLines.push(line.slice(5).trim());
    });
    if (!dataLines.length) return;
    const data = dataLines.join('\n');
    if (data === '[DONE]') return; // onAllDone fires when the stream ends
    let obj;
    try { obj = JSON.parse(data); } catch (_) { return; }
    if (event === 'file_start' && handlers.onFileStart) handlers.onFileStart(obj.fileId, obj.name, obj.totalParts);
    else if (event === 'part' && handlers.onPart) handlers.onPart(obj.fileId, obj.partIndex, obj.totalParts);
    else if (event === 'file_done') { if (obj.meta) metas.push(obj.meta); if (handlers.onFileDone) handlers.onFileDone(obj.fileId, obj.meta); }
    else if (event === 'file_rejected' && handlers.onFileRejected) handlers.onFileRejected(obj.name, obj.reason);
    else if (event === 'phase' && handlers.onPhase) handlers.onPhase(obj.phase, obj.state, obj);
    else if (event === 'stats' && handlers.onStats) handlers.onStats(obj);
  }

  // A stream ended badly. Tell the caller what happened, then end the stream
  // normally so the composer unlocks whether or not it handles `onError`.
  // `reached` says whether the daemon may have started the turn. False when the
  // request failed before any response: the turn never ran, so the same message
  // can be sent again. True once a response began: the daemon cancels a turn
  // whose client drops and may already have stored part of it.
  function fail(handlers, message, reached) {
    if (handlers.onError) handlers.onError(message, { reached });
    handlers.onDone();
  }

  function errText(e) {
    if (!e) return 'unknown error';
    return (e && e.message) ? e.message : String(e);
  }

  // Parse one SSE frame: a named `status` event, or an OpenAI chunk / [DONE].
  function handleFrame(frame, handlers) {
    let event = null;
    const dataLines = [];
    frame.split('\n').forEach((line) => {
      if (line.indexOf('event:') === 0) event = line.slice(6).trim();
      else if (line.indexOf('data:') === 0) dataLines.push(line.slice(5).trim());
    });
    if (!dataLines.length) return;
    const data = dataLines.join('\n');
    if (event === 'status') {
      try { handlers.onStatus(JSON.parse(data).text || ''); } catch (_) {}
      return;
    }
    if (event === 'projection') {
      try { if (handlers.onProjection) handlers.onProjection(JSON.parse(data)); } catch (_) {}
      return;
    }
    if (event === 'tool') {
      try { if (handlers.onTool) handlers.onTool(JSON.parse(data)); } catch (_) {}
      return;
    }
    if (event === 'think') {
      try { if (handlers.onThink) handlers.onThink(JSON.parse(data)); } catch (_) {}
      return;
    }
    if (event === 'prefill') {
      try { if (handlers.onPrefill) handlers.onPrefill(JSON.parse(data)); } catch (_) {}
      return;
    }
    if (data === '[DONE]') { handlers.onDone(); return; }
    try {
      const chunk = JSON.parse(data);
      const delta = chunk.choices && chunk.choices[0] && chunk.choices[0].delta;
      if (delta && typeof delta.content === 'string') handlers.onToken(delta.content);
    } catch (_) { /* ignore keepalive / partial */ }
  }

  window.ZendLiveAPI = ZendLiveAPI;
})();
