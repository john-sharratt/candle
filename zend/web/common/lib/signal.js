/* Minimal reactivity — the whole of it. No dependency, no build step.
 * §23: this is the ~50-line signal helper the no-build decision leans on. */

let ACTIVE = null;

export function signal(value) {
  const subs = new Set();
  const s = {
    get value() { if (ACTIVE) subs.add(ACTIVE); return value; },
    set value(v) {
      if (Object.is(v, value)) return;
      value = v;
      [...subs].forEach((fn) => fn());
    },
    peek: () => value,
    subscribe(fn) { subs.add(fn); return () => subs.delete(fn); },
  };
  return s;
}

export function effect(fn) {
  const run = () => { const prev = ACTIVE; ACTIVE = run; try { fn(); } finally { ACTIVE = prev; } };
  run();
  return () => { ACTIVE = null; };
}

export function computed(fn) {
  const out = signal(undefined);
  effect(() => { out.value = fn(); });
  return out;
}

/* A tiny event bus for cross-page notifications (toasts, stream frames). */
const bus = new Map();
export const on = (evt, fn) => {
  if (!bus.has(evt)) bus.set(evt, new Set());
  bus.get(evt).add(fn);
  return () => bus.get(evt).delete(fn);
};
export const emit = (evt, payload) => (bus.get(evt) || []).forEach((fn) => fn(payload));
