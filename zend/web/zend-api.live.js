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
 *   - followConversation (the reply in flight, same frames)    GET …/{id}/live (SSE)
 *   - subscribeLogs / seedLogs               WS /ws/logs (structured JSON frames)
 *   - getToolSchemas                         GET /v1/tools (zend's entries)
 *   - getMe                                  GET /v1/me (the role, and zend's share)
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
  // How long one `/v1/status` poll may wait for an answer (see `getStatus`).
  // A live daemon answers it without the engine lock, in milliseconds.
  const STATUS_TIMEOUT_MS = 8000;

  // The aggregate module is the same for the life of the page. Importing it
  // on every status poll is another request, and a tab waking up already has
  // its streams to reopen.
  let aggregateMod = null;
  function loadAggregate() {
    if (!aggregateMod) {
      aggregateMod = import('/common/lib/aggregate.js').catch((e) => {
        aggregateMod = null;
        throw e;
      });
    }
    return aggregateMod;
  }

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
        // `error`: why a failed turn stopped, on its last assistant bubble.
        history: (body.messages || []).map((m) => ({ role: m.role, content: m.content, no_think: !!m.no_think, thinking: m.thinking ? [m.thinking] : [], tool_tokens: m.tool_tokens || [], spans: m.spans || [], files: m.files || [], error: m.error || null })),
        uploads: body.uploads || [],
        // The composer dials this conversation last ran at, as levels. Absent
        // for one that has never taken a turn — the composer keeps its own.
        dials: body.dials || null,
        // A reply is in flight: `history` ends at its question, and
        // `followConversation` streams the reply.
        live: !!body.live,
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

    // GET /v1/status — the host's status: the process's, with zend's own line
    // under `tenants.zend` (drives the startup overlay). Through a clustered
    // gateway it is every member's, and the ready member's speaks for zend
    // (`common/lib/aggregate.js`). A failed fetch rejects. So does a gateway
    // envelope nobody answered: that used to come back as a synthetic
    // "connecting" line, and the page treated it as zend reloading.
    //
    // So does a fetch still unanswered after `STATUS_TIMEOUT_MS`. A gateway
    // whose route drops packets never refuses the connection, and the
    // browser's own limit is minutes — every poll waits on the one before
    // it, so without this the page sits on whatever it last drew. The
    // caller's `signal` (the page being hidden) still rejects as an
    // `AbortError`; the timeout rejects as an ordinary error, a miss.
    getStatus(signal) {
      const ctl = new AbortController();
      const forward = () => ctl.abort();
      if (signal) {
        if (signal.aborted) ctl.abort();
        else signal.addEventListener('abort', forward, { once: true });
      }
      let timer = null;
      const timeout = new Promise((_, reject) => {
        timer = setTimeout(() => {
          reject(new Error('/v1/status did not answer within ' + (STATUS_TIMEOUT_MS / 1000) + 's.'));
          ctl.abort();
        }, STATUS_TIMEOUT_MS);
      });
      const answer = Promise.all([
        fetch('/v1/status', { headers: { accept: 'application/json' }, signal: ctl.signal }).then((r) => {
          if (!r.ok) throw new Error('GET /v1/status -> ' + r.status);
          return r.json();
        }),
        loadAggregate(),
      ]).then(([body, aggregate]) => {
        const st = aggregate.memberStatus(body, 'zend');
        if (aggregate.isEnvelope(body) && !(st && st.started_at_ms)) {
          throw new Error('No zend daemon answered behind the gateway.');
        }
        return st;
      });
      return Promise.race([answer, timeout]).finally(() => {
        clearTimeout(timer);
        if (signal) signal.removeEventListener('abort', forward);
      });
    },

    // GET /v1/me — the caller's role, and zend's share for it: the tools modes
    // it may choose and the one it starts at. The gateway's sign-in decides the
    // role; a caller the daemon cannot place gets the least.
    getMe() {
      return getJSON('/v1/me')
        .then((me) => Object.assign({ role: me.role }, (me.tenants || {}).zend || {}))
        .catch(() => ({
          role: 'unauthenticated',
          tool_modes: ['none', 'restricted'],
          default_tools: 'restricted',
        }));
    },

    // GET /v1/tools — every tenant's catalog, each tool naming its tenant;
    // zend's tools' argument JSON Schemas, by name. The tool-call cards list
    // every parameter from them, including the ones a call left at their
    // defaults.
    async getToolSchemas() {
      const body = await getJSON('/v1/tools');
      const out = {};
      (body.tools || []).filter((t) => t.tenant === 'zend').forEach((t) => { out[t.name] = t.parameters || null; });
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
      // **A 503 that says when to come back is a wait, not a failure.** The
      // gateway answers one, with `Retry-After`, while the member holding this
      // conversation is reachable only through another gateway, and while every
      // member is restarting; no turn started, so the same request is simply
      // sent again. The bubble says what it waits on in the meantime, and only a
      // wait past `RETRY_FOR_MS` is reported as an error.
      const RETRY_FOR_MS = 120000;
      const started = Date.now();
      let retryTimer = null;
      const send = () => fetch('/v1/chat/completions', {
        method: 'POST',
        // The conversation, named where a cluster's gateway reads it without
        // buffering the body (docs/replicated_substrate.md §5.2).
        headers: { 'content-type': 'application/json', 'x-tokera-conversation': String(conv.id) },
        body: JSON.stringify(payload),
        signal: controller.signal,
      }).then((resp) => {
        const retryAfter = Number(resp.headers.get('retry-after'));
        if (resp.status === 503 && retryAfter > 0 && Date.now() - started < RETRY_FOR_MS) {
          resp
            .json()
            .then((body) => (body && body.error && body.error.message) || '')
            .catch(() => '')
            .then((detail) => {
              if (controller.signal.aborted) { handlers.onDone(); return; }
              const waitS = Math.min(Math.max(retryAfter, 1), 10);
              if (handlers.onStatus) {
                handlers.onStatus((detail || 'Waiting for the cluster') + ' — retrying in ' + waitS + ' s');
              }
              retryTimer = setTimeout(() => { retryTimer = null; send(); }, waitS * 1000);
            });
          return;
        }
        // **A stream that failed is not a stream that finished.** Every one of
        // these used to call `onDone`, so an HTTP 500, a body-less response and
        // a mid-stream socket drop all rendered as a completed answer — an
        // empty bubble and no way to tell whether the model had nothing to say
        // or the daemon had died. The caller gets `onError` and decides;
        // `onDone` still runs after it, so the composer always unlocks.
        // Any status but 200 means no turn started. The daemon streams every
        // request it accepts, and rejects the rest before submitting anything —
        // a malformed turn (`400`, e.g. a missing `conv_id`) or a 408 from the
        // edge on a slow uplink are both of these. A rejection carries an
        // OpenAI-shaped `{ error: { message } }` body saying what was wrong, and
        // that reason is worth more to the user than the bare status, so it is
        // read before reporting. Parsing may itself fail — the edge's 408 is
        // HTML — in which case the status stands on its own.
        if (!resp.ok) {
          const status = 'HTTP ' + resp.status;
          resp
            .json()
            .then((body) => (body && body.error && body.error.message) || '')
            .catch(() => '')
            .then((detail) => {
              fail(
                handlers,
                detail
                  ? status + ' before the turn started: ' + detail
                  : 'The request failed with ' + status + ' before the turn started.',
                false,
              );
            });
          return;
        }
        readStream(resp, handlers);
      }).catch((e) => {
        if (e && e.name === 'AbortError') { handlers.onDone(); return; }
        fail(handlers, 'Could not reach the daemon: ' + errText(e), false);
      });
      send();
      return {
        cancel: () => {
          controller.abort();
          // A wait between sends has no request for the abort to end, so the
          // stream is ended here instead.
          if (retryTimer !== null) {
            clearTimeout(retryTimer);
            retryTimer = null;
            handlers.onDone();
          }
        },
      };
    },

    // GET /v1/conversations/{id}/live — the reply in flight, from its first
    // frame, then live until it ends. For a page that opened (or reloaded, or
    // lost its connection) while the daemon was still answering: the turn runs
    // to completion regardless, and this is how the page catches up with it.
    // `handlers.onGone` runs instead of anything else when nothing is in flight
    // any more — the stored conversation is then complete.
    followConversation(id, handlers) {
      const controller = new AbortController();
      fetch('/v1/conversations/' + enc(id) + '/live', { signal: controller.signal }).then((resp) => {
        if (resp.status === 404) { if (handlers.onGone) handlers.onGone(); return; }
        if (!resp.ok) {
          fail(handlers, 'Following the reply failed with HTTP ' + resp.status + '.', true);
          return;
        }
        readStream(resp, handlers);
      }).catch((e) => {
        if (e && e.name === 'AbortError') { handlers.onDone(); return; }
        fail(handlers, 'Could not reach the daemon: ' + errText(e), true);
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
        // The host's one stream carries every tenant's lines, each naming its
        // tenant; zend's pane shows zend's and the engine's, which name none.
        ws.onmessage = (ev) => {
          try {
            const line = JSON.parse(ev.data);
            if (!line.tenant || line.tenant === 'zend') onLine(line);
          } catch (_) {}
        };
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

    // ── the conversation's working journal (docs/zend_journal.md §8) ───────
    // Read-only from here: every line in it is one the daemon cited and
    // checked, so an entry can be forgotten but not edited.
    getJournal(convId) {
      return getJSON('/v1/conversations/' + enc(convId) + '/journal');
    },
    // Resolves to the journal as it now reads.
    async forgetJournalEntry(convId, entryId) {
      const r = await fetch('/v1/conversations/' + enc(convId) + '/journal/' + Number(entryId), { method: 'DELETE' });
      if (!r.ok) throw new Error('DELETE journal entry -> ' + r.status);
      return r.json();
    },
    // Resolves to the turns the draft covers, or rejects with the daemon's
    // reason (a draft already running, nothing new, the conversation not open).
    async draftJournal(convId) {
      const r = await fetch('/v1/conversations/' + enc(convId) + '/journal/draft', { method: 'POST' });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.message || ('POST journal draft -> ' + r.status));
      return body;
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

  // Read a chat stream's SSE frames into `handlers` until it ends.
  function readStream(resp, handlers) {
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
    // The turn failed. Not the stream: the daemon keeps what the turn wrote
    // and why it stopped, and the stream still ends with `[DONE]`.
    if (event === 'turn_error') {
      try { if (handlers.onTurnError) handlers.onTurnError(JSON.parse(data).message || ''); } catch (_) {}
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
