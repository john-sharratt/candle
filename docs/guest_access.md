# Guests: an allowlist, a role below User, and a queue

Who may use the estate, and how much of the card a stranger gets. Written
2026-10-28 after an inventory found that every Google account that signs in is
a `User` (`web/src/auth/role.rs:210-226`), that `/mcp` consent asks only that
the caller be signed in (`web/src/auth/mcp/authorize.rs:180`), and that zend's
`POST /v1/chat/completions` runs the model for a caller who has not signed in
at all (`zend/src/api/mod.rs:81-82`; reached from the internet through
`code.tokera.com`, answered `422` for a body missing `messages` rather than
`401`).

## 1. The role table is an allowlist with a role per entry

`host/roles.yaml` — the one table the gateway reads and every daemon compiles
in (`host::access::ROLES`) — becomes a single list:

```yaml
people:
  - email: owner@example.com
    role: creator
  - email: friend@example.com
    role: user
```

Each entry names one of `sub`, `email` or `email_sha256`, and a `role` of
`user`, `admin` or `creator`. A person listed twice is refused at load, not
resolved by order — an address written out and the same address hashed are
one person — and someone named in forms the load cannot compare (a `sub` and
an email) gets the highest role among them. `admins:` and `creators:` are
replaced, not kept beside it.

**The shipped table names people by `email_sha256`.** The repository is
public, and an address committed there is published permanently. The hash is
SHA-256 of the address trimmed and lowercased, in lowercase hex
(`printf '%s' 'someone@example.com' | sha256sum`). It keeps the address from a
reader, not from someone who already suspects it and hashes their guess
(the author's choice, 2026-10-28).

## 2. `Guest`: signed in, not listed

`Role` gains a level between `Unauthenticated` and `User`:

| Role | Who |
|---|---|
| `Unauthenticated` | the gateway named nobody |
| **`Guest`** | signed in, not in `people` |
| `User` | listed as `user` |
| `Admin` | listed as `admin` |
| `Creator` | listed as `creator` |

`Ord` follows declaration order, so every existing `at_least(Role::User)` and
every route whose minimum is `User` refuses a guest with no edit. That is the
point: today's `User` capabilities become the listed people's, and a guest gets
only what is opened to `Guest` on purpose (§3). The wire spelling is `guest`.

## 3. What a guest may do

From the inventory (every route at `User` today, by daemon). The lockdown
(§8, step 1) gives a guest their own account and nothing else; the inference
below opens with the queue (step 2).

- **Inference, through the guest queue (§4):**
  - npcd `POST /v1/generate/description`, `/description/stream`, `/name`;
  - npcd `POST /v1/image/generate`, `/v1/image/cutout`;
  - zend `POST /v1/chat/completions` (§5).
- **Their own account:** `/v1/me/profile` and its history, `/v1/me/unique-name`.
- **Nothing else.** In particular:
  - no characters. A character keeps ticking in the background once created
    (`npcd/src/api.rs:1318-1330`), which is load no queue gates;
  - no `/mcp` (consent requires `User`, §6);
  - nothing from the operational and corpus reads: host `/v1/telemetry`,
    `/v1/memory`, `/v1/phases`, `/v1/projections`, `/v1/layers`, `/v1/logs`,
    `/ws/logs`; npcd `/v1/mind/*`, `/v1/pulse/world`, `/v1/world/:wid/channel`,
    `/v1/image/:iid`. These stay at `User` and above.

## 4. The guest queue

Guests run inference only while the listed people are not. It is a gate in
front of the engine, per member, in the host crate so every tenant shares it
(`host/src/guest_gate/`), not an engine band: a band says how
latency-sensitive work is, and a guest's chat is as latency-sensitive as
anyone's — what differs is whose turn it is.

**Person work.** A request from a caller of `User` or above to a route marked
`Cost::Compute` (below) is person work for as long as its handler runs,
counted by a guard dropped when the response ends — a streamed response
counts until its stream closes. Background work — NPC ticks, ingest,
calibration — is nobody's request and is not counted: it shares the card by
band as it does today.

**The gate.** A guest's request to a `Cost::Compute` route waits in a FIFO
until all of these hold, then runs:

- no person work in flight on this member;
- none for `QUIET` (5 s): one person's consecutive turns are not split by a
  guest slipping in between them;
- fewer than `GUEST_SLOTS` (1) guest jobs running on this member;
- this guest has no other job running or queued (one at a time per `sub`).

**No preemption.** A guest job running when a person arrives finishes; the
person's work is not held behind it. What bounds the person's wait is the
job's size:

- every guest turn is submitted at band `Low` (`DecodePriority::METADATA_KEY`
  on its conversation), so a person's prompt outranks it in admission and
  pauses its creep (`docs/continuous_fair_waves.md` §7);
- `max_tokens` is clamped to `GUEST_MAX_TOKENS` (512) and a prompt over
  `GUEST_MAX_PROMPT_TOKENS` (4,096) is refused;
- an image draw is one job and the drain's own bounds apply (≈7.5 s on
  BlackWorld).

**What the guest sees.**

- A streamed route opens at once and says `{"event":"queued","position":N}`
  each time the position changes, then runs.
- A non-streamed route waits.
- More than `GUEST_QUEUE_DEPTH` (32) waiting: `429 {"error":"busy",
  "retry":true}` at once.
- Waited `GUEST_MAX_WAIT` (120 s): `503 {"error":"busy","retry":true}`.
- The caller hanging up leaves the queue.

**Marking routes.** The host route table (`host/src/guard.rs`) gains a cost
beside each route's minimum role, `Cost::Free` or `Cost::Compute`, and the
guard applies the gate centrally: a handler never sees a guest it has not
been admitted for. Each tenant marks its inference routes: the five npcd
routes and zend's chat above, and every `User` route that runs the model, so a
listed person's use of them counts as person work.

**MCP tools.** npcd's tools call route handlers in-process and so skip the
route table (`npcd/src/mcp/mod.rs:7-14`): `image_generate` reaches `post_image`,
whose route is `Admin`, at `User`. Each tool takes its route's minimum role
and cost from the table instead of restating it, so a tool can never be wider
than its route.

**Across members.** Each member's status says whether it has person work in
flight (`people_busy`). The gateway places a guest's request on a member that
is not busy when one exists, and on the least loaded otherwise
(`cluster/src/gateway/registry.rs`), so guests use idle members instead of
queueing behind busy ones. It is a field the member already sends in its
status, so it rides the next protocol bump.

## 5. Callers who have not signed in

zend's routes were open to anyone: `POST /v1/chat/completions` ran the model
for a caller with no identity, and the conversation routes listed, read and
deleted what people wrote. They are the table's people's now — chat, the repo
map and every `/v1/conversations/*` route at `User` (`zend/src/api/mod.rs`).
`/v1/models` runs nothing and stays open; `/v1/mcp` names its own caller.

A conversation that names no owner — the dialogues zend's own UI makes, and
passthrough clients' — is an Admin's (`zend/src/mcp/ownership.rs`): nobody's
name is on it, and with more than one person on the table no user can be
shown to be its owner. A user's zend work is the conversations `/mcp` starts
for them, which name them.

OpenAI-style clients that send no identity — the Cline passthrough
(`cline_passthrough_wiring`), `zen-vscode`, `zend-load`, scripts on a
member — are refused until they present one; run against a member directly,
`--local-signin` with a listed email is that identity. Opening chat to guests through the queue, and giving
those clients a way to present an identity, is step 2 (§8), decided by the
author: "we can open up for cline and friends later".

## 6. `/mcp`

Consent requires `User`: `authorize::ask` and `decide` refuse a guest with
"this account is not on the list of people who may use Zen; ask the estate's
owner to add it". Every token exchange — a refresh included — re-reads the
caller's role from the table, so a person removed from `people` loses `/mcp`
within an access token's hour, not a refresh token's 30 days. zend's and
npcd's MCP servers answer a guest's call `403` with the same words, with no
sign-in challenge (`mcp_server::NoCaller::NotAllowed`): a challenge would send
the client round a sign-in that the consent page then refuses.

## 7. Logging

At `INFO` on `web::auth::mcp`, one line each — the account, client id,
redirect host and outcome, never a code or a token:

- consent shown, allowed, denied, refused for role;
- code exchanged, refresh, refused (with the RFC 6749 error);
- client metadata document fetched (its URL, and the error when it failed).

The guest gate logs on `host::guest_gate` when a guest is queued, admitted,
refused for depth, and timed out, with the queue depth and wait.

## 8. Rollout

1. **The lockdown** (the author, 2026-10-28: "we need to lock down the roles
   and whitelist"): §1, §2, §6, §7's `web::auth::mcp` lines, and zend's routes
   at `User` (§5). A guest owns their account and runs nothing. Released as
   one motherd build — the table is compiled in, and a daemon that does not
   know `Guest` would read every unlisted person as a `User` — and a gateway
   restart, for its copy of the table, the consent check and the logging.
2. **Guests run inference:** the queue (§4), §3's inference routes opened to
   `Guest`, zend chat at `Guest` through it, the `people_busy` placement, MCP
   tools taking their route's role and cost, and the clients of §5
   presenting an identity.

## 9. Tests

- The table: the `people` form parses; a role names one of three; a person
  listed twice, an entry naming both `sub` and `email`, an unknown field all
  refused; an unlisted signed-in caller is `Guest`; nobody is anything without
  an identity.
- `Role`: `Guest` sits between `Unauthenticated` and `User`, spelled `guest`;
  every `User` route refuses a guest.
- The gate, against a fake clock:
  - a guest runs at once on an idle member;
  - waits while person work is in flight and for `QUIET` after;
  - runs in FIFO order, one per member;
  - a second request from the same guest waits for the first;
  - depth → `429`, wait → `503`;
  - a hang-up leaves the queue;
  - a person arriving mid-job is not held.
- Clamps: `max_tokens` above the cap is clamped; a prompt over the cap is
  refused; a guest turn's conversation records band `low`.
- MCP: a tool is refused exactly where its route is; consent refuses a guest;
  a refresh after removal from the table is refused.
- Placement: a guest request goes to a member without person work when there
  is one.
