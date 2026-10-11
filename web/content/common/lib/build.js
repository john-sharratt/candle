/* Force a reload when the UI on disk or the daemon changes.
 *
 * `/v1/status`'s `build` hashes every UI file on disk and the running binary,
 * so a saved edit or a rebuild moves it. A page left running would otherwise
 * talk to a new API with old code — they disagree on the surface and break
 * silently (a method the new HTML calls is absent in the old JS). Capture the
 * build id on first sighting and reload when it changes.
 *
 * The scroll position and current route ride across the reload in
 * sessionStorage, so a rebuild during work is a blink rather than a reset.
 */

const KEY = 'npcd.reload';

export function saveForReload(extra) {
  try {
    sessionStorage.setItem(KEY, JSON.stringify({ hash: location.hash, ...extra }));
  } catch (_) {}
}

export function takeReloadState() {
  try {
    const v = sessionStorage.getItem(KEY);
    sessionStorage.removeItem(KEY);
    return v ? JSON.parse(v) : null;
  } catch (_) { return null; }
}

let seen = null;

/** Returns true when it triggered a reload (caller should stop). */
export function checkBuild(status, extra) {
  if (!status || !status.build) return false;
  if (seen == null) { seen = status.build; return false; }
  if (status.build !== seen) { saveForReload(extra); location.reload(); return true; }
  return false;
}

export const currentBuild = () => seen;
