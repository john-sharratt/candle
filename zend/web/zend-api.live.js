/* ============================================================================
 * zend-api.live.js — live daemon adapter (window.ZendLiveAPI)
 * ----------------------------------------------------------------------------
 * Implements the ZendAPI contract (docs/zend_ui_redesign.md §4) against the
 * daemon's HTTP/SSE/WebSocket endpoints.
 *
 * Implemented (Phase 2.1 — endpoints that exist today, validated by the
 * gui_api_harness integration test):
 *   - seedConversations / getConversation   GET /v1/conversations[/{id}]
 *   - archiveConversation / unarchiveConversation  POST …/archive|unarchive
 *   - streamChatCompletion (token + status)  POST /v1/chat/completions (SSE)
 *   - subscribeLogs / seedLogs               WS /ws/logs (structured JSON frames)
 *
 * Projection glue + section content ride along on getConversation (first-class
 * fields), so the projection panel renders the framing and expands sections
 * with no extra round-trip.
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
      // plus the workspace-wide projection glue + section content (first-class,
      // so the projection panel needs no extra round-trip).
      const body = await getJSON('/v1/conversations/' + enc(id));
      const sectionContent = {};
      (body.section_content || []).forEach((s) => { sectionContent[s.name] = s.content; });
      const turnContent = {};
      (body.turn_content || []).forEach((t) => { turnContent[t.group + '::' + t.index] = { text: t.text, user: t.user, assistant: t.assistant, layout: t.layout }; });
      return {
        id: String(id),
        history: (body.messages || []).map((m) => ({ role: m.role, content: m.content, no_think: !!m.no_think, spans: m.spans || [], files: m.files || [] })),
        glue: body.glue || null,
        sectionContent,
        turnContent,
        targetLayer: body.target_layer || '',
        uploads: body.uploads || [],
      };
    },
    archiveConversation(id) { return postVoid('/v1/conversations/' + enc(id) + '/archive'); },
    unarchiveConversation(id) { return postVoid('/v1/conversations/' + enc(id) + '/unarchive'); },

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
        if (!resp.ok || !resp.body) { handlers.onDone(); return; }
        const reader = resp.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        const pump = () => reader.read().then(({ done, value }) => {
          if (done) { handlers.onDone(); return; }
          buf += dec.decode(value, { stream: true });
          let nl;
          while ((nl = buf.indexOf('\n\n')) !== -1) {
            const frame = buf.slice(0, nl);
            buf = buf.slice(nl + 2);
            handleFrame(frame, handlers);
          }
          return pump();
        }).catch(() => handlers.onDone());
        pump();
      }).catch(() => handlers.onDone());
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
    if (data === '[DONE]') { handlers.onDone(); return; }
    try {
      const chunk = JSON.parse(data);
      const delta = chunk.choices && chunk.choices[0] && chunk.choices[0].delta;
      if (delta && typeof delta.content === 'string') handlers.onToken(delta.content);
    } catch (_) { /* ignore keepalive / partial */ }
  }

  window.ZendLiveAPI = ZendLiveAPI;
})();
