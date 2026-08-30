/* Hold RIGHT ALT to see the hidden documents — if you are an admin.
 *
 * Some authored content is flagged `hidden`: it stays out of listings so a
 * dropdown on a screen share, or a screenshot, does not contain it. The filter
 * box already reveals one by name — typing the whole word `earth` finds that
 * world — but naming a thing you are trying to look through the whole set for
 * is no help, so this is the gesture for "show me everything, briefly".
 *
 * **It is discretion, not access control, and the daemon is the authority.**
 * Holding the key makes the console *ask* (`?reveal=1`); the daemon honours the
 * request only for an admin, reading the role from the gateway's headers on
 * that request. Nothing here grants anything — `can('admin')` below decides
 * whether to bother asking, and a client that lies about it gets the same
 * listing as one that does not.
 *
 * Right Alt specifically, by `code` rather than by `key`: on several keyboard
 * layouts that key is AltGr and reports as `Alt` with `ctrlKey` set, so `code`
 * is the only spelling that means the physical key on every layout. Left Alt is
 * deliberately not it — it opens the browser's menu bar on Windows.
 */

import { can } from './router.js';

let held = false;
const listeners = new Set();

function set(next) {
  if (next === held) return;
  held = next;
  for (const fn of listeners) {
    try {
      fn(revealing());
    } catch (_) {
      /* One listener throwing must not strand the others in the wrong state. */
    }
  }
}

addEventListener('keydown', (e) => {
  if (e.code === 'AltRight') set(true);
});
addEventListener('keyup', (e) => {
  if (e.code === 'AltRight') set(false);
});
/* A key held while the window loses focus never delivers its `keyup` — alt-tab
 * away and the console would stay revealed with nobody holding anything. */
addEventListener('blur', () => set(false));

/* Whether hidden documents should be shown right now: the key is down AND the
 * viewer is an admin. Both halves are re-read on every call, so signing out
 * with the key held stops revealing without needing an event. */
export function revealing() {
  return held && can('admin');
}

/* Run `fn(revealing())` whenever the key goes down or up. Returns an unsubscribe
 * — a page that forgets to call it keeps repainting after it has been replaced. */
export function onReveal(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
