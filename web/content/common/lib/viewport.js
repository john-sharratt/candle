/* Breakpoint state, shared by the shell and any page that needs to change
 * behaviour rather than just layout.
 *
 * The CSS media queries fire at these exact numbers. zend's note is worth
 * repeating: if the JS breakpoint and the CSS breakpoint disagree you get a
 * dead-band where the layout has gone narrow but the drawers and modals are
 * still rendering desktop-style. One source of truth, both sides.
 */

/** Below this the rail becomes an overlay drawer and the nav collapses. */
export const NARROW_W = 820;
/** At or above this there is room for side panels alongside the main column. */
export const WIDE_W = 1100;
/** Below this, labels give way to icons. */
export const TINY_W = 560;

const width = () => (typeof window === 'undefined' ? 1280 : window.innerWidth);

export const state = {
  narrow: width() < NARROW_W,
  wide: width() >= WIDE_W,
  tiny: width() < TINY_W,
};

const subs = new Set();

function recompute() {
  const next = { narrow: width() < NARROW_W, wide: width() >= WIDE_W, tiny: width() < TINY_W };
  if (next.narrow === state.narrow && next.wide === state.wide && next.tiny === state.tiny) return;
  Object.assign(state, next);
  // Mirror onto the root so CSS can key off the *same* decision the JS made,
  // for the handful of rules that need to follow behaviour rather than width.
  const r = document.documentElement;
  r.toggleAttribute('data-narrow', state.narrow);
  r.toggleAttribute('data-wide', state.wide);
  r.toggleAttribute('data-tiny', state.tiny);
  subs.forEach((fn) => { try { fn(state); } catch (_) {} });
}

/** Subscribe to breakpoint crossings. Returns an unsubscribe. */
export function onBreakpoint(fn) {
  subs.add(fn);
  return () => subs.delete(fn);
}

if (typeof window !== 'undefined') {
  // Coalesce resize storms to one recompute per frame.
  let raf = 0;
  window.addEventListener('resize', () => {
    if (raf) return;
    raf = requestAnimationFrame(() => { raf = 0; recompute(); });
  }, { passive: true });
  // Orientation change on mobile fires before the new size settles.
  window.addEventListener('orientationchange', () => setTimeout(recompute, 120));
  recompute();
  const r = document.documentElement;
  r.toggleAttribute('data-narrow', state.narrow);
  r.toggleAttribute('data-wide', state.wide);
  r.toggleAttribute('data-tiny', state.tiny);
}
