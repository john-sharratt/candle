/* Sign-in state in the top bar.
 *
 * The only script on a tokera.com page, and the page is complete without it —
 * everything else is server-rendered, so a reader with JavaScript off loses the
 * sign-in control and nothing else. That is why the slot ships `hidden`: an
 * empty box is better than a "Sign in" link that a broken script would leave
 * showing to someone who is already signed in.
 *
 * The session cookie is HttpOnly, so this cannot read it — which is the point.
 * `/auth/me` is the only way to ask, and the gateway is the only thing that can
 * answer.
 */

const slot = document.getElementById('site-auth');

/** Come back *here* after signing in — the whole URL, host included.
 *
 * The provider's registered redirect URI names one host for the estate, so the
 * browser always returns to it and a relative `next` resolves against that host
 * rather than the one you left. This file only ever runs on the callback host
 * today, which is the sort of accident that holds until someone loads it on a
 * second site and cannot see why sign-in moves them. `safe_next` accepts an
 * absolute URL under the cookie domain and refuses everything else. */
const here = () => location.href;

function el(tag, attrs, text) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  return n;
}

function signedOut() {
  slot.replaceChildren(
    el('a', { href: '/auth/login?next=' + encodeURIComponent(here()), class: 'signin' }, 'Sign in'),
  );
}

function signedIn(me) {
  const bits = [];
  if (me.picture) {
    // referrerpolicy so the provider's CDN is not told which page is being read.
    bits.push(el('img', { src: me.picture, alt: '', referrerpolicy: 'no-referrer' }));
  }
  bits.push(el('span', { class: 'who' }, me.name || me.email || 'Signed in'));
  bits.push(el('a', { href: '/auth/logout?next=' + encodeURIComponent(here()) }, 'Sign out'));
  slot.replaceChildren(...bits);
}

async function paint() {
  if (!slot) return;
  let me;
  try {
    const r = await fetch('/auth/me', { credentials: 'same-origin' });
    if (!r.ok) throw new Error(r.status);
    me = await r.json();
  } catch (_) {
    // The gateway is unreachable or sign-in is not configured. Offering a
    // button that cannot work is worse than offering none.
    return;
  }
  // `configured: false` means this deployment has no sign-in at all — distinct
  // from being signed out, and the difference is whether a button makes sense.
  if (me.configured === false) return;
  if (me.authenticated) signedIn(me);
  else signedOut();
  slot.hidden = false;
}

paint();
