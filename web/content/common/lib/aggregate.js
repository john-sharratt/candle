/* One status from `/v1/status`, whoever answered it.
 *
 * Straight from a daemon, `/v1/status` is that process's status: its `state`,
 * `build`, and each tenant's line under `tenants`. Through a gateway with the
 * app clustered it is an aggregate (`docs/replicated_substrate.md` §5.5) —
 * `{ members: [{ node, host, build, status, body }], group }` — every ready
 * member's answer side by side, and nothing at the top level a console reads.
 * A console that took the envelope for a status saw no `tenants`, and held its
 * loading screen forever while every member behind it was ready.
 *
 * `memberStatus` turns either into the one status a console drives its loading
 * screen and its build check from. Through the gateway the app is usable as
 * soon as one member is ready for it — the gateway sends the console's requests
 * to ready members — so that member's status speaks for the app; until one is,
 * the first member that answered shows its load. An envelope no member answered
 * reads as connecting.
 */

/** Whether `body` is a gateway's aggregate rather than one process's answer. */
export const isEnvelope = (body) => !!body && Array.isArray(body.members);

/* The members whose answer is a `2xx` object, in the envelope's order. */
const answered = (body) =>
  body.members.filter(
    (m) => m.status >= 200 && m.status < 300 && m.body && typeof m.body === 'object',
  );

/* `tenant`'s line in one process's answer, or `null`. */
const lineOf = (body, tenant) => (body.tenants || {})[tenant] || null;

/** The status a console for `tenant` reads from a `/v1/status` body. */
export function memberStatus(body, tenant) {
  if (!isEnvelope(body)) return body;
  const members = answered(body);
  const ready = members.find((m) => (lineOf(m.body, tenant) || {}).state === 'ready');
  const chosen = ready || members[0];
  if (chosen) return chosen.body;
  return {
    state: 'loading',
    started_at_ms: 0,
    tenants: {
      [tenant]: {
        state: 'loading',
        loading: {
          label: 'Connecting',
          detail: `no ${tenant} member answered`,
          progress: 0,
          completed: [],
        },
      },
    },
  };
}

/** A per-member page's view of an answer: the members that answered, as
 * `{ node, host }` in the envelope's order, and the body of the one named
 * `node` — the first, when `node` is not among them. One process's answer has
 * no members, and is the body; an envelope no member answered has no body. */
export function pick(body, node) {
  if (!isEnvelope(body)) return { members: [], node: null, body };
  const members = answered(body);
  const chosen = members.find((m) => m.node === node) || members[0] || null;
  return {
    members: members.map((m) => ({ node: m.node, host: m.host })),
    node: chosen ? chosen.node : null,
    body: chosen ? chosen.body : null,
  };
}

/** For an answer that is per tenant but carries no state — `/v1/substrate`'s
 * overview — the first member's that holds `tenant`'s line; `{ tenants: {} }`,
 * which holds none, when no member's does. One process's answer is its own. */
export function memberWith(body, tenant) {
  if (!isEnvelope(body)) return body;
  const holding = answered(body).find((m) => lineOf(m.body, tenant) !== null);
  return holding ? holding.body : { tenants: {} };
}
