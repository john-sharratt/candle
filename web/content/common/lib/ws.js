/* A websocket that reconnects, for panes that must not be polled.
 *
 * The backoff mirrors the proxy's (250ms doubling to 10s) on purpose: when the
 * daemon behind the proxy goes down, both sides are waiting the same amount of
 * time, so the page recovers on the same beat the gateway does instead of
 * hammering a socket the gateway is refusing anyway.
 *
 * Every frame is one JSON object. A frame that does not parse is dropped with a
 * console warning rather than killing the stream — one malformed line must not
 * cost the reader everything after it.
 *
 *   const sub = subscribe('/ws/logs', {
 *     onMessage: (line) => ring.push(line),
 *     onState:   (s) => badge(s),        // 'live' | 'reconnecting' | 'closed'
 *   });
 *   sub.close();                          // on teardown — always call this
 */

const INITIAL_MS = 250;
const MAX_MS = 10_000;

/** ws:// or wss:// for a path on this origin, matching the page's scheme. */
export function wsUrl(path) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${location.host}${path}`;
}

export function subscribe(path, { onMessage, onState, onOpen } = {}) {
  let sock = null;
  let timer = null;
  let failures = 0;
  let closed = false;

  const state = (s) => { if (onState) onState(s); };

  const open = () => {
    if (closed) return;
    let s;
    try {
      s = new WebSocket(wsUrl(path));
    } catch (_) {
      // Construction itself throws on a malformed URL or a blocked scheme —
      // treat it as a failed attempt so the retry ladder still applies.
      return retry();
    }
    sock = s;

    s.onopen = () => {
      // Reset only on a connection that actually opened. Resetting on the
      // attempt would turn the ladder into a fixed 250ms retry against a
      // daemon that is down, which is the behaviour the ladder exists to stop.
      failures = 0;
      state('live');
      if (onOpen) onOpen();
    };

    s.onmessage = (ev) => {
      let obj;
      try { obj = JSON.parse(ev.data); }
      catch (_) { console.warn('ws: dropping unparseable frame from', path); return; }
      if (onMessage) onMessage(obj);
    };

    // `onclose` fires after `onerror` for a failed connection, so retrying
    // here alone avoids scheduling two reconnects for one failure.
    s.onerror = () => {};
    s.onclose = () => { sock = null; if (!closed) retry(); };
  };

  const retry = () => {
    failures += 1;
    state('reconnecting');
    const wait = Math.min(INITIAL_MS * 2 ** (failures - 1), MAX_MS);
    clearTimeout(timer);
    timer = setTimeout(open, wait);
  };

  open();

  return {
    close() {
      closed = true;
      clearTimeout(timer);
      state('closed');
      if (sock) { const s = sock; sock = null; try { s.close(); } catch (_) {} }
    },
    get connected() { return !!sock && sock.readyState === WebSocket.OPEN; },
  };
}
