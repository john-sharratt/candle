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
