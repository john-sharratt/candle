/* What one `/v1/status` poll may do to a console that already has the chat up.
 *
 * Leaving the tab — alt-tab, or the browser suspending a background page —
 * aborts the in-flight status fetch, and a gateway envelope nobody answered
 * looks like a daemon that is loading. Neither is a restart. Showing the
 * loading card and dropping every hydrated conversation on them is what made
 * coming back to the console freeze or sit on "Reconnecting" while zend was
 * ready the whole time.
 *
 * `misses` is how many unanswered polls are already being ignored.
 * The result:
 *   ignore    — the page is hidden, or this poll was aborted
 *   miss      — unanswered; the chat stays
 *   reconnect — unanswered often enough, while the page is visible, to say so
 *   reload    — a daemon answered, and zend is not ready
 *   restart   — a daemon answered ready, and the process start changed
 *   stay      — answered ready, same process
 *
 * A poll that hangs is a miss too: `getStatus` rejects once it has waited
 * `STATUS_TIMEOUT_MS` (zend-api.live.js) instead of holding the chain open.
 */

export const MISS_LIMIT = 2;

export function decideStatus(p) {
  if (p.hidden || p.aborted) return 'ignore';
  const state = p.zendState || '';
  const started = p.startedAt || 0;
  // No answer, or the invented loading line a silent gateway envelope
  // becomes (`started_at_ms` 0). A live daemon's start is a unix millis.
  if (state !== 'ready') {
    if (!(started > 0)) return (p.misses || 0) + 1 >= MISS_LIMIT ? 'reconnect' : 'miss';
    return 'reload';
  }
  if (p.lastStartedAt != null && started !== p.lastStartedAt) return 'restart';
  return 'stay';
}

/* The boot poll — the one that owns the screen before the chat is up, and
 * after a reconnect — when nothing answers it. A gateway that is down, or a
 * route to it that drops packets, never answers at all, so the splash cannot
 * wait for an answer to change what it says.
 *
 * `failures` is how many boot polls in a row went unanswered, this one
 * included. The first few are the daemon or the network blinking and keep
 * the splash; from `UNREACHABLE_AFTER` on the page says it cannot reach the
 * server, and the retry backs off to `RETRY_MAX_MS` so a tab left open on a
 * dead gateway is not a request a second forever.
 */
export const UNREACHABLE_AFTER = 3;
export const RETRY_MIN_MS = 1000;
export const RETRY_MAX_MS = 10000;

export function bootRetry(failures) {
  const n = Math.max(1, failures | 0);
  return {
    unreachable: n >= UNREACHABLE_AFTER,
    delay: Math.min(RETRY_MAX_MS, RETRY_MIN_MS * 2 ** (n - 1)),
  };
}

/* What the unreachable card says went wrong, from the error the status fetch
 * rejected with (`zend-api.live.js` `getStatus`). A fetch that never reached
 * anything rejects with a `TypeError` whose message is the browser's own
 * ("Failed to fetch", "NetworkError when attempting to fetch resource.",
 * "Load failed"), which says nothing to a reader; the others carry their own.
 */
export function unreachableReason(name, message) {
  const msg = String(message || '');
  if (name === 'TypeError') return 'The server could not be reached.';
  const status = /-> (\d{3})$/.exec(msg);
  if (status) return 'The gateway answered ' + status[1] + '.';
  return msg || 'The server did not answer.';
}
