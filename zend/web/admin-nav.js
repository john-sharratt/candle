/* The tool pages' admin-only links.
 *
 * A link an admin follows is in the page from the start, `hidden`; it is shown
 * once `/v1/me` says the reader is an admin. Hiding it is courtesy, not the
 * guard: the daemon refuses the page's routes to anyone else, and the page
 * says so if it is reached anyway. */
const ADMIN_ROLES = { admin: 1, creator: 1 };

export function revealAdminLinks() {
  fetch('/v1/me', { headers: { accept: 'application/json' } })
    .then((res) => (res.ok ? res.json() : null))
    .then((me) => {
      if (!me || !ADMIN_ROLES[me.role]) return;
      document.querySelectorAll('[data-admin]').forEach((el) => { el.hidden = false; });
    })
    .catch(() => {});
}
