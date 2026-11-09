/* Polling that stops while nobody can see the page.
 *
 * Every console page that refreshes on a timer used to keep doing it in a tab
 * left open in the background — a request every second or two, for nobody. One
 * helper for the whole estate, so each page's poll is the same thing: it skips
 * while the page is hidden, polls once at once on becoming visible again, and
 * stops completely — its timer cleared, its listener removed — when the page
 * asks it to, which a poll may do from inside its own callback.
 */

/**
 * Call `fn` every `ms` milliseconds while the page is visible, and once more
 * each time it becomes visible again. Returns `stop`, which ends it for good.
 */
export function pollWhileVisible(ms, fn) {
  let stopped = false;
  const hidden = () => typeof document !== 'undefined' && document.hidden;
  const tick = () => {
    if (stopped || hidden()) return;
    fn();
  };
  const timer = setInterval(tick, ms);
  const onVisibility = () => {
    if (!hidden()) tick();
  };
  if (typeof document !== 'undefined') {
    document.addEventListener('visibilitychange', onVisibility);
  }
  return function stop() {
    if (stopped) return;
    stopped = true;
    clearInterval(timer);
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', onVisibility);
    }
  };
}
