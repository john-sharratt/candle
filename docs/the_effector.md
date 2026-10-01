# The Effector Device and `http://local`

*How an NPC interacts with a world that can grow without bound, through a tool
surface that never grows. The body keeps its hands; it gains an **effector
device**; the device speaks HTTP to a real router the world stands up; the schema
that router hands back is the same schema that constrains the model's call; and the
whole world layer is attachable, so an embedder can bring their own.*

This is the fourth surface in the NPC estate. The other three are named in
[`npc_api_gui_design.md`](npc_api_gui_design.md): what an NPC *is*
([`npc_mind_design.md`](npc_mind_design.md)), what we *build*
([`npc_engine_design.md`](npc_engine_design.md)), and what it *looks like from
outside* over the operator wire (`npc_api_gui_design.md` itself). This document is
the surface from *inside the fiction* — the API an NPC's own effector device calls,
at the host `local`.

Two APIs share this daemon's transport and must never share a meaning:

| | Operator API | Effector (device) API |
|---|---|---|
| Host | `bot.tokera.com` / `npcd.localhost` | `local` (in fiction) + a mount on the hosted domain (external) |
| Prefix | `/v1/*`, `/ws/*` | `http://local/...` in fiction; `/v1/local/...` externally (§8.2) |
| Caller | a person, a script, the console | an NPC in the fiction, or an external client bearing an NPC's token |
| Identity | gateway `X-Tokera-*` headers | the NPC's own auth token (§8.3) |
| Purpose | run and observe the daemon | *be* a character acting in the world |

The operator API is out of the fiction and already built (`npcd/src/api.rs`,
`ops.rs`, `engine/mod.rs`). The effector API is in the fiction and is what this
document specifies.

---

# Part A — Why

## 1. The tool surface does not scale, and cannot be made to

[`tool_surface_audit.md`](tool_surface_audit.md) counted it exactly. Nine body
tools are implemented and armed. **One hundred and fifteen** more are authored into
the vault with no parameters, no handlers, no plane, and no presence in the
grammar. The audit's own conclusion: *"Building the foundation is building all 79,
plus what this audit adds."* That is the road we are not going to walk.

*(Correction, from a later code audit — Appendix D: the **handlers were since
implemented**, in `engine/work.rs` (`work::perform`); what the audit called missing is
now the *routing surface*, not the handler bodies. That makes the migration mostly a
re-wiring, and it does not weaken the argument here — the compiled per-verb cost is
what fails to scale, whether or not a handler already exists behind each verb.)*

Each of those tools, to become real, is the same five-part cost, paid once per
verb, forever:

1. A `Tool` static (`tools.rs:186`) with an `Availability` variant and `params`.
2. A `Choices` variant and a `LIVE` row for every world-enumerated argument
   (`tools.rs:1196`, `1443`).
3. A field on `Within` — the per-turn world snapshot that is *also the grammar
   cache key* (`tools.rs:1509`) — so the new fact is hashable and gate-visible.
4. A handler in `enact.rs`.
5. Its share of the compiled turn grammar.

The fifth cost is the wall. The turn grammar is the reachable catalogue expanded to
`ACTS_PER_TURN = 2` (`tools.rs:1162`), budgeted against `MAX_TURN_PATHS = 2_000_000`
(`tools.rs:1955`). Because the tree is the catalogue *squared*, and an unbounded
value type makes it non-terminating, **every parameter is restricted to `string` or
`boolean`** — an `integer` is "any JSON number," measured running away at 390 MB/s.
A world that wants a machine with a numeric dial, a list argument, or a nested
payload cannot have one. The surface is not just expensive to grow; it is *shaped*
to stay small.

So "the world can expand almost infinitely" is unreachable through this door. The
door has to change.

## 2. The body keeps its hands; it gains an effector device

The NPC does not stop being embodied. It still thinks it is real, and everything
that makes it act alive stays where it is: the body catalogue — `tell`, `whisper`,
`shout`, `ask`, `gesture`, `move_to`, `follow`, `reflect` (`tools.rs:876`) — is
fixed, always present, and unchanged. Speech is a body act. Walking is a body act.
Those are hands, and hands do not need a URL.

What the NPC gains is a **second thing in its hands: an effector device.** A
diegetic handheld that talks to whatever is around it, shows what is reachable from
where the body stands, and lets the character read the world's affordances and act
on them. The device is one object in the fiction — the character holds it and reads
its screen — not a system-level catalogue handed to a model. That distinction is
the whole immersion argument, and §6 makes it concrete. "Effector" is the term the
character-facing copy uses: hands sense-and-move the body; the effector device
sense-and-acts on the world.

The split is the point:

- **The body** — a fixed set of embodied acts. Never grows. Makes it alive.
- **The effector device** — a fixed set of *verbs* (§5) over a *dynamic* set of
  addressable things. The one extensible port.
- **The world** — places and machines and subsystems, each advertising what it
  affords as data the device reads — and *attachable*, so the world behind the
  device can be npcd's vault or an embedder's own (§9.1).

A new place, machine, or entire subsystem shows up as new addresses under
`http://local/...`, discovered and called through the device, with **zero new tools
and zero recompile.** The tool surface stays fixed while the world behind it grows
without bound.

## 3. The wager: an LLM is already an API client

Models are extraordinarily good at calling well-defined JSON APIs, because the
corpus is saturated with exactly that. So we do not simulate an API — we run a real
one, and let the model do what it is best at. Concretely, each defended later:

- Real URLs at a real host, `http://local/...`, each carrying the world item's id
  (§7).
- Real JSON, only JSON, serialized fast, with field-level prescriptive errors
  (§8.4, §12).
- Real HTTP methods — `GET` to read, `OPTIONS` to fetch a schema, `POST`/`PUT`/
  `DELETE` to act (§5).
- The schema the world advertises for a call is the *same* schema that constrains
  the model's decode of that call (§11).

The effector device is therefore not a dispatch table dressed up as HTTP. It is an
HTTP router that answers, reachable both by a fast in-process path and over a real
socket, tested like one. Everything else falls out of taking that literally.

---

# Part B — The effector device (the fixed surface)

## 4. What is a body act and what is a device call

The seam: **the body is what you do with yourself; the device is what you do to the
world through an interface.** Speech and movement are the body reaching people and
places directly. Working a terminal, calling a lift, reading a ledger, redrawing a
map — those are the body operating a *thing*, and every thing is reached through the
effector device.

| Stays a body act | Becomes a device route |
|---|---|
| `tell` `whisper` `shout` `ask` `gesture` | every `STATION_ACTS` verb (~53, `station.rs:996`) |
| `move_to` `follow` | `lift_call` `lift_use` (`acts.rs:682`, `705`) |
| `reflect` | the world-state half of `WORLD_ACTS` (`acts.rs:1311`) |
| `act` `sleep` `promise` `remind` (human interactions) | the phone acts (`message` `invite` `open_group` `send_image` `reach_out`) → `http://local/phone` |
| — | `bench_*` (11), reused `file_*` (5), all 115 audited part tools |

Speech does not go behind the device, and not from sentiment: if `tell` were `POST
http://local/say`, a character could only speak where a `/say` route was mounted —
asserting a body can talk near a terminal and not in a corridor, the exact falsehood
the audit found when `room.talk` was hung on furniture (`tool_surface_audit.md`
§3.1). Speaking is `Availability::Always`. The device is for what a body does *not*
always have — which is everything the world supplies.

The full migration — every tool that stays and every tool that becomes a route,
with its proposed path — is **Appendix A**, for your review.

## 5. The verbs: real HTTP methods behind two device tools

The effector device adds exactly two tools to the compiled catalogue. They are the
only new `Tool` statics this design introduces, and they never change as the world
grows. Behind them are ordinary HTTP methods, so the model's prior applies directly.

```
query(url: string)
    Read the address without changing anything. Issues GET for a listing or a
    readable resource's data, and OPTIONS for a resource's schema (§5.1). Safe and
    idempotent, so the character — and the engine (§6, §11) — may look freely.

invoke(url: string, body: object)
    Act at the address. Issues the resource's effectful method (POST / PUT /
    DELETE) with a JSON body. Returns the result, or a prescriptive error (§12).
```

`query` versus `invoke` is GET/OPTIONS versus everything-else — the line the model
already draws between finding out and acting. Two named tools (rather than a single
`request(method, url, body)`) match the discover-then-act mental model and keep the
safe path unmistakably safe. **The model never picks the method:** the resource
advertises its one effectful method in its schema and `invoke` uses it, so a
wrong-method 405 is not a mistake available to the model.

### 5.1 `OPTIONS` completes the abstraction

A resource's *shape* is fetched with a real method, not a bespoke call. `OPTIONS
http://local/table/<id>` returns the methods the resource accepts and the JSON
schema of each effectful call's body:

```
OPTIONS http://local/table/<id>
→ 200 {
    "id": "table/command-3",
    "summary": "The command table. Turn it on to bring the level's boards live.",
    "methods": {
      "GET":  { "returns": { "powered": "boolean", "claimed_by": "string|null" } },
      "POST": { "summary": "Set power.",
                "body": { "type": "object",
                          "properties": { "powered": { "type": "boolean" } },
                          "required": ["powered"] } }
    }
  }
```

`OPTIONS` is chosen over `HEAD` because it is the standard "describe this resource /
these are its methods" verb, it can carry a schema *body*, and it maps one-to-one
onto the discovery step the design needs; `HEAD` (schema in headers) was the
alternative floated in review, and `OPTIONS` is the decision. The schema it returns is
what the stencil is compiled from (§11), so `query`-for-schema is not just for the
character to read.

## 5.2 Verbs are resources

A thing in the world is addressed by its id — `http://local/chronicle/<id>` — and
**each verb it affords is itself an address beneath it**:
`http://local/chronicle/<id>/add_entry`. That verb path is a full resource: `query`
it for *its* schema, `invoke` it to act. The parent (`.../chronicle/<id>`) answers
what the thing is and lists the verbs beneath it; a verb answers exactly one body.

This is the decision that makes typed `invoke` total rather than partial. A resource
that afforded several verbs could not be typed by a single focus — the schema the
character queried could not know which verb the next `invoke` would pick (the
multi-verb limit §11 first hit). Addressing the **verb** removes the ambiguity by
construction: every invokable address has exactly one body, so the schema a `query`
returns is exactly the schema the following `invoke` needs. There is nothing left to
disambiguate, and no per-branch decode-time machinery is required (§11).

So the grammar of an address is:

```
http://local/<ns>/<id>            a thing — GET its state, OPTIONS lists its verbs
http://local/<ns>/<id>/<verb>     a verb — GET/OPTIONS its one body, POST to act
```

The personal surfaces follow the same shape: `http://local/phone/message` is a verb
resource; `http://local/self/plan` is a readable one.

*As-built delta: the shipped station router mounts each verb as a `POST`-only
sub-path of the thing and types the body by a compile-time splice, which leaves a
multi-verb thing's body free-JSON (Step 4's honest gap). Closing it is this section:
give each verb path its own `GET`/`OPTIONS`, and arm the focus (§11) when a character
queries a **verb** — then every focus is single-body and every `invoke` is typed. It
is a small, additive change to the station route and the focus, not a rewrite.*

## 6. The near-you index is a route, shown as a superseding device percept

The list of what is reachable from here is a route — `GET http://local/`, served by
the same handler machinery as everything else, consistent and externally
inspectable. Its answer is the **top level and only the top level**, recomputed from
where the body stands, each entry carrying its world item id (§7):

```
GET http://local/
→ 200 { "routes": [
    { "url": "http://local/lift/command-shaft",   "summary": "the lift" },
    { "url": "http://local/table/command-3",       "summary": "the command table" },
    { "url": "http://local/chronicle/ct-command-1","summary": "a world-history terminal" },
    { "url": "http://local/history",               "summary": "what you have done and seen" },
    { "url": "http://local/phone",                 "summary": "your phone" }
  ] }
```

**Where this is shown to the model is the point.** It is only ever true *now*, so it
is delivered each turn as a **superseding percept** — an `EventKind::Reachable` keyed
so the pending inbox holds exactly one current device screen, never a growing pile of
lists as the body moves (`replaces()`, the same discipline the situation band already
follows, `delta.rs:191`). It is rendered verbatim, at `Salience::IDLE` so it never
preempts.

*As-built correction (Appendix D territory): the design first called this "a dynamic
section of the system prompt, like the mission and task sections." That is how it
reads to the character, but not how it is built — npcd has no per-turn-fresh
system-prompt-section mechanism (schema sections are cast-shared and sealed; a
per-turn re-seal is model-dependent and would mean touching `candle-*`). And the
mission/task content it was compared to does not live in a projected section either:
missions reach the character as a **superseding percept** too (the `Nudge`, rendered
verbatim in `narrator.rs`). So "like the missions and tasks" is honoured literally —
the near-you list is the same kind of superseding percept. The one honest cost: like
every point-in-time percept, a turn's line stays in that turn's window history — but it
is a single top-level line, not schemas (those arrive via `query`, inline), and the
window is bounded, so the conversation does not carry the world's whole API.*

It reads, in the character's register:

```
YOUR EFFECTOR DEVICE
Reachable from here: the lift · the command table · a world-history terminal · world history · your phone
```

and, for the leaves the body is standing at, it carries their schema inline (§11), so
the character can act in one turn.

**The queries and interactions stay inline.** Only the *ambient list* is the
superseding percept. A `query` and its schema response, an `invoke` and its result or
error, are actions and their outcomes, and they belong in the turn stream as
`<tool_call>`/`<tool_response>` exactly like any act (§5, §12). The split is the whole
of it: *what is reachable* is a superseding, point-in-time percept; *what I did about
it* is the conversation.

## 6.1 The system prompt explains the device

The character is told, once, in the system prompt, what the effector device is and
how to use it — the same place the body's acts and the world are explained
(`prompt.rs:614`, the `WHAT YOU CAN DO` frame). The copy establishes, in the
character's own register:

- **What it is** — a handheld effector: it senses what is around you and lets you
  act on it. Its screen lists addresses; the world answers at them.
- **How to look** — `query` an address to see what lives beneath it or what a
  thing will accept. Looking never changes anything.
- **How to act** — `invoke` an address with the fields it asked for. The world
  answers, or tells you plainly what was wrong so you can fix it.
- **What the addresses are** — the screen shows what is reachable from where you
  stand; walking changes it. Each address names one thing in the world.

It does *not* mention the auth token (§8.3): that is carried for the character, not
by it. The exact wording is drafting, not architecture; the requirement is that no
turn is the first time the character learns the device exists.

---

# Part C — The world API (the dynamic half)

## 7. `http://local`, and every URL carries an id

The host is `local`. In the fiction it is the only host the device can reach: no
DNS, no egress. `http://local/...` is the entire universe the device can address —
the immersion boundary (the character's world is exactly what the device reaches)
and, for the model's output, a safety boundary (a hallucinated `http://evil.example`
resolves to nothing). Externally the same routes are mounted on the hosted domain
(§8.2); `local` is the in-fiction spelling of that mount.

**Every addressable thing carries its world item id in the URL.** Not
`http://local/table` but `http://local/table/command-3`. Three reasons, all from
review:

- **Uniqueness.** A level has several terminals; a building has several lifts. The
  id is what tells `chronicle/ct-command-1` from `chronicle/ct-portraits-2`.
- **External addressability.** An external client (§8.2) turns *the command table
  on floor three* on or off by naming it — `POST http://local/table/command-3` —
  without standing anywhere.
- **Stable reference.** An id in a URL is a durable handle the world, the console,
  and a test can all share.

`local` means **local to the caller**: the *near-you index* (§6) is filtered to the
caller's standpoint, so what a character *discovers* is only what is around it.
Whether a fully-qualified id can be reached *directly* depends on the caller's token
scope (§8.3): an **as-NPC** token (every in-fiction call, and any external client
acting *as* the character) is proximity-gated and 404s on an id not near the body; a
**direct** token (an operator or embedder driving the world from outside) addresses
any item by id regardless of standpoint — which is what makes external control by id
work.

Two kinds of route live under `local`:

- **Situated routes** — reachable because of *where you are*: the lift, the room's
  machines, the stations within reach. They come and go as the body moves.
- **Personal routes** — reachable because they are *yours*, wherever you stand:
  `http://local/history`, `http://local/phone` (§16), the character's own memory
  surface. Always in the near-you index; the exception to "reachable from here."

## 7.1 When a route is available — the model we must be able to simulate

A route's availability is a product of three independent facts, and simulating the
world means answering all three from state:

1. **Reachability** — is the thing within reach of where the body stands? This is
   the map: `within_reach` (`perceive.rs:280`) over the node's placed parts,
   `at_landing` (`world.rs:545`) for the lift. Frozen topology, live standpoint.
2. **Condition** — is the thing in a state that affords the call? A powered-down
   command table affords `OPTIONS` and `POST {powered:true}` but none of its
   working verbs; a claimed station affords reading but not a second body's write; a
   part in one `mode` affords a different verb set than in another (`Part.modes`,
   `part.rs`; `binds`/`holder_of`, `world.rs`).
3. **Standing** — is it a personal route, always yours (history, phone)?

Reachability and standing exist in the map and world today. **Condition is the
gap.** Per-item mutable state — the command table's on/off, a terminal's mode, which
body holds a claim — is the same "dynamic half" the world already keeps for
actors/holds/lift, but *per placed item* it does not exist yet: a `Part` is a shared
catalogue entry placed by reference, with no per-instance state (`part.rs:133`). So
the world-map design this document depends on adds one thing — a **per-item state
record**, keyed by the item id of §7, living in the world's mutable half beside
`actors`/`riders`/`lift` (`world.rs:458`), that route handlers read and write. This
is the smallest addition that makes "turn the command table off" real, and it is
what the simulation tests of Part F drive.

The near-you index (`GET http://local/`, §6) is the composition of all three: the
reachable items, filtered to those whose condition affords at least one call, plus
the personal routes. That composition is a pure function of world state — so it is
simulable and testable without a model, and mapping it out fully (every part, its
conditions, the verbs each condition affords) is a prerequisite deliverable, not an
afterthought. **Appendix C is that map** for the vault.

## 7.2 The personal namespace

The personal routes are reachable wherever the body stands (§7) and are the same for
every world, so npcd owns them even when an embedder has replaced the vault (§9.1):

- `http://local/phone` — the messaging surface, where the migrated phone acts live:
  `phone/message` (POST, to a `unique_name`), `phone/invite`, `phone/open_group`,
  `phone/send_image`, `phone/reach_out`. The addressee is the recipient's unique name
  (`npc_api_gui_design.md` §12) — an enum where the world can enumerate contacts, a
  free string otherwise.
- `http://local/history` — read-only (GET) onto what this body has done and seen,
  cursor-paginated like the operator substrate routes; the effector's window onto the
  character's own past.
- `http://local/self` — the character's own maintained state as a read view: its
  plan, its orders, its beliefs, its memory. The *writes* to these happen at the
  situated stations that own them (the planning board, the order table — §9.2) and
  project into the system prompt; `self` is where the character reads them back on
  demand.

Everything else under `local` is situated (§7.1) and supplied by the active world.

## 8. One API, two entry paths, one identity

The world stands up a genuine `axum` router for host `local`, built beside the three
that already exist and **nested into the npcd router at `/v1/local`** (`main.rs`) —
*not* a second `web` `local_api` site. (As-built: `web` dispatches sites by the `Host`
header, `server.rs:311`, so a `local` *host* would need its own hostname, whereas a
`/v1/local` *path* already resolves to the npcd site — the nest is the clean
equivalent, no `sites:` change.) The router runs behind its **own** device-auth
middleware (§8.3), *not* `guard::Api`: that wrapper hardcodes the operator `X-Tokera`
role check with no escape hatch (`guard.rs:91`), which is the whole reason the device
carries its own.

### 8.1 The fast path skips TCP, never the API

An NPC's call must not pay for a socket. But it must still be a real API call —
same routing, same handlers, same validation, same error shape. So there is **one
router with two entry paths:**

- **In-fiction (fast).** The engine builds a real `http::Request`, stamps the auth
  header (§8.3), and invokes the router *in-process* — the `tower::ServiceExt::
  oneshot` path the existing routers are already tested through (`api.rs:448`). No
  TCP, no serialize-to-socket, no syscall. This is the path every `query`/`invoke`
  takes, at the world's cadence (`EVERY = 500ms`, `driver.rs:44`) times the cast.
- **External (real socket).** The identical routes answer on the bound port
  (§8.2), for an external client.

Both hit the same handler code. The fast path is not a second implementation; it is
the same service invoked without a network in front of it. Nothing about the
contract — URLs, methods, JSON, schemas, errors — differs between the two.

Three implementation realities the review surfaced — none fatal, all to be designed
for rather than discovered:

- **The sync/async seam.** `within` and the act path (`record_act` → `act_on_world`
  → `body::perform`) are synchronous and run inside the async `character_loop`
  (`runtime.rs:1038`, `1481`, `3564`). A `tower` router call is a `Future`, and
  `block_on` from a worker thread panics — so the fast path either makes that slice
  `async` (viral through a deep sync tree) or hops through `spawn_blocking` / a
  scoped current-thread runtime. Cheap in bytes, not free in plumbing; this seam is
  the real cost, chosen deliberately.
- **No re-entry while holding the world lock.** `Hosted.state` is a non-reentrant
  `std::sync::Mutex` (`world/mod.rs:41`). A handler that holds `hosted.with(...)` and
  then invokes the router again in-process (a sub-resource, an `OPTIONS`, the near-you
  index) deadlocks silently. **Invariant: never invoke the router while holding the
  world lock** — handlers acquire, act, release, as the lift handlers already do
  (`enact.rs:535`) — and it is tested.
- **The production Router handle.** Today the merged router is handed to `web` and no
  handle is kept; the `oneshot` path is `#[cfg(test)]` (`api.rs:448`, `main.rs:636`).
  **As-built (Step 1):** `Runtime` retains the router (`set_effector_router`) and
  drives it via `effector_query`, an `async` method — the sync/async seam above is
  handled by making the effector call at the async layer, never `block_on`.

### 8.2 The external mount

The effector routes are also reachable from outside, on the hosted domain, under a
proper path. The in-fiction `http://local/table/command-3` is mounted externally as
`/v1/local/table/command-3` (the same router, nested — §8), authenticated by the
NPC's token (§8.3) rather than by an operator role. This is what lets an external client —
a game embedder, a test, a tool — reach exactly what a character reaches, and it is
why the id-in-URL of §7 matters: external control addresses things by id.

*As-built: the near-you index — the in-fiction `http://local/` — is reached externally
at the bare prefix `/v1/local` (no trailing slash: axum serves a nested router's `/`
route at the prefix itself, and `/v1/local/` 404s). Sub-resources are
`/v1/local/<ns>/<id>...` as expected.*

### 8.3 Identity, unified: one `Principal`, two shapes

Per your note, collapse the two auth models into one `Principal`, resolved at one
point:

- `Principal::Human(Identity)` — a person, from the gateway `X-Tokera-*` headers,
  with roles/ownership as today (`identity.rs`, `role.rs`).
- `Principal::Npc(token)` — a character, from the effector token, resolving to a
  body.

The operator API resolves a `Human`; the effector API resolves an `Npc`; "an
external client acting as NPC X" is then a first-class `Npc(token)`, not a special
case. This is where "who is acting" is decided, once.

Every NPC is seeded at startup with an opaque token, carried as an auth header on
every effector call and **invisible to the LLM** — the model never sees it, never
types it, cannot leak it into prose. The engine stamps it on the in-process request
(§8.1); an external client sends it directly.

- **Seeding & lookup.** The token is minted when the NPC starts and recorded in a
  quick-lookup table token → body. It must **not** be derived from the deterministic
  `body_id` (that would be guessable) — it is a real secret. Whether it is persisted
  so an external token survives a restart is **Q6** (recommended: persisted, since
  the world otherwise is not — `runtime.rs:794`).
- **The v1 check, whole.** Authorized iff the request carries the header, the header
  names a token, and the token is in the table (one lookup). No roles, no ownership
  on this surface — the token *is* the identity, resolving to a body whose standpoint
  the handlers read from world state (`body_of`; `at_landing`, `world.rs:545`;
  `within_reach`, `perceive.rs:280`).
- **Scope decides proximity.** A token carries a scope. An **as-NPC** token — the
  character's own — is proximity-gated: it reaches only what is within the body's
  standpoint, so an id not near the body 404s, in the fiction or from outside. A
  **direct** token — for an operator or embedder driving the world externally —
  addresses any item by id regardless of standpoint. The scope is a property of the
  token, checked after the one lookup; the in-fiction fast path always uses the
  as-NPC scope, so a character can never reach past its own standpoint (§7).

**The device surface needs its own middleware — it cannot reuse `guard::Api`.** The
one route-registration primitive, `Api::route`, hardcodes `require(headers, roles,
min)` reading `X-Tokera-*`, with no escape hatch by design (`guard.rs:91`). The
effector routes register through a *parallel* wrapper (or a `Principal::Npc` path in
`require`) that **ignores `X-Tokera-*` entirely** and resolves the token. This is a
build blocker for Step 1, not a detail, and two hazards make it sharp:

- **The fast path** builds an `http::Request` and stamps the token (§8.1). Because
  npcd runs `behind_gateway()` and therefore *believes* inbound `X-Tokera-*`
  (`identity.rs:1`), that request must never carry one, or the operator middleware
  would authenticate a character as a human — pinned by a test, both directions.
- **The external mount** (§8.2) is on the same gateway-fronted domain, so a client
  can send *both* a token and `X-Tokera-*`; the effector middleware must read only
  the token and never fall through to the role check.

One token, one lookup, one meaning of "who is acting." Richer auth (scopes,
revocation) is later; §16 keeps it out of v1.

### 8.4 The JSON must be fast, and wrong calls must say why

Two hard requirements on the serializer, because both the fast path's throughput and
the wager of §3 depend on them:

- **Fast.** `serde` + `serde_json` (the estate's serializer, `Cargo.toml`) on the
  in-process path, with request/response values built and parsed without an
  intermediate socket encode. Handlers deserialize into typed structs (not
  `Value`) wherever the schema is fixed, so parsing is a single pass into the shape
  the handler wants. Performance is a build-time concern to measure, not assume —
  the device call is on the world's hot loop.
- **Prescriptive.** A malformed or invalid body returns a structured error with a
  `field` naming the offending key — `400 missing_field {field:"powered"}`, `400
  bad_range {field:"count", detail:"1..10"}`. Two honest costs the review named: the
  estate's `err()` helper emits `{error, detail}` only today (`api.rs:1637`), so
  `field` is *added* — either by widening that shared helper (which touches the
  operator surface) or a device-only helper; and mapping a deserialize failure to a
  precise `field` is real work (serde's error path is clean for a missing scalar but
  blurs across `flatten`, enums, and nested objects — exactly the rich bodies §11
  wants). This is first-order, not plumbing: per §11 the handler's validation *is*
  the enforcement for every type the decoder only shapes, so the `field` it returns
  is the correction signal the whole wager (§3) rests on.

## 9. Who registers a route — the join, and the attachable world

A route is four things: a **path** (carrying an id, §7), a **method**, a **JSON
schema** for its body, and a **handler** that reads or changes the world. This is
the seam where the two halves meet: the world declares routes as data, and both the
device (for the character) and the engine (for the stencil, §11) read the same
declaration.

The map already has the right shape. A `Part` in `npc-map` is a catalogue entry —
terminal, board, chair — placed into rooms by reference, and it deliberately does
*not* name the acts it offers (`part.rs:75`): that vocabulary lives in npcd's
engine. Under this design it becomes *routes addressed by the part instance's id*,
mounted wherever that instance is within reach. The chronicle terminal's `binds`,
`modes`, and the audit's verbs become `http://local/chronicle/<id>/...` routes. One
declaration, mounted per standpoint, is what makes the situated address space of §7
fall out of the map — *once placed instances have ids, which today they do not.*

**Minting the ids is the biggest hidden cost, and it must come from the map.** Today
a `Placement` is anonymous and count-based (`Bare` or `Counted{part,count}`,
`part.rs:131`); `part_ids_at` de-duplicates by catalogue `part.id` within a node
(`load.rs:268`), so two world-history terminals in one room collapse to a single
entry — `chronicle/ct-command-1` versus `-2` in the same room is *not expressible*
today; and world state keys on `place` (`area/node`) plus display name, never an
instance (`sim/mod.rs`; `within_reach` returns catalogue ids, `perceive.rs:280`).
Because the world is not persisted (bodies re-enter deterministically,
`runtime.rs:788`), an id must be a **pure function of the map** to stay stable across
restart and be shareable by console and test. **As-built:** the id is
`<part-id>~<ordinal>` (tilde is URL-unreserved and appears in no kebab-case part id, so
the split on the last tilde is reversible; ordinal is **0-based and world-wide**,
counting placements of that catalogue part across the whole map in a deterministic walk
— areas by id, nodes in file order, placements in file order) —
`npc-map/src/instance.rs`, `MapSet::{instances_at, resolve_instance}`.

The id was once `<area>~<node>~<part-id>~<ordinal>`, which was self-describing but long
(`vault-command~command-room~order-table~0`) — many tokens per arm once the url became a
grammar-forced enum (§5.1's stencil, below). Dropping the area and node names to a
world-wide ordinal (`order-table~0`) keeps it unique and a pure function of the map — the
walk is deterministic and each node's starting offset is precomputed once at load
(`MapSet::part_offset`), so `instances_at` stays O(node) — while cutting the id to a
fraction of the length. The within-node catalogue-id de-dup is kept as `part_ids_at`
(for `within_reach`), with the per-instance enumeration beside it; the sim keys devices
by this instance id (`sim/seed.rs`, `Sim::station`), while record *custody* stays global
(an era held anywhere is unavailable everywhere — correctly not per-instance).

**The url is a stencil, not free text (the anti-hallucination cut).** `query`'s `url` and
`invoke`'s `url` are grammar-forced enums, not free strings: `query.url` is bound to the
near-you set (`Choices::QueryUrl` ← `Within::reachable`), and `invoke.url` to each
reachable resource's verb-paths (`Choices::InvokeUrl` ← `Within::invokable`,
`<resource-url>/<verb>` from `station::verbs_of`). So the decoder is forced through a real
address the device actually lists — a character can never `query http://local/command-table`
(a hallucinated guess that 404s) nor `invoke` a bare resource (a 405); it reads and acts
on exactly what stands within reach. The near-you percept renders each route as
`- <summary> — <url>` so the address it must name is on the screen to copy
(`prompt::near_you_section`).

Handlers reach the world through the one lock: `Hosted`, a single mutex over
`{ world, attention, sim, rooms }` (`world/mod.rs:41`), with `with` the sole write
path, `read` for queries, `with_both` for world+sim (`world/mod.rs:117`). A device
handler is an `enact.rs` handler with an HTTP envelope; the lift handlers
(`enact.rs:519`, `535`) are the template, and their typed refusals (`Refused`,
`world.rs:351`) become the route's error bodies (§12).

### 9.1 Two worlds: the simulated world and the attachable world API

npcd is two things that this design must keep separable:

1. **The NPC engine** — minds, embodiment, the effector device, the schema→stencil
   machinery, the token auth, the near-you index. This is npcd's, and it is
   world-agnostic.
2. **A world** — the concrete set of `http://local/...` routes and the state behind
   them. npcd *ships one* (the vault), but a project that adds npcd as a dependency
   must be able to **attach its own** — its own affordances, its own state model —
   and have the NPCs reach it through the same effector device.

So the route layer is an extension point, not a hard-coded vault. The vault is one
implementation of a world-routes interface (a trait the host implements, and/or a
runtime registration API — **Q4**); an embedder like Battle Cities supplies another.
npcd brings the brain, the body, and the device; the embedder brings the world the
device reaches. This mirrors the estate's existing posture — "the HTTP layer is a
transport over the core, never where behaviour lives" (`npc_api_gui_design.md` §1) —
extended one level: the *world* is a provider over the core, not baked into it.

An embedder's world **replaces** the vault — one active world per daemon — while
npcd keeps the personal routes (`/history`, `/phone`, and the character's own
surface); the embedder brings its own state, behind npcd's one lock (below). What is
settled is the seam: the device and its machinery do not know which world they are
addressing.

**One constraint on the seam, from the one-lock invariant.** `Hosted` unifies
`{world, attention, sim, rooms}` under a single mutex precisely so no situation
computation crosses two locks (`world/mod.rs:13`). An attachable world must not
reintroduce that hazard: the extension point is a provider of **route handlers that
run inside `Hosted`'s lock** (a `dyn WorldRoutes` invoked under `with`/`with_both`),
with the embedder's state living behind that same mutex exactly as `Sim` does today —
not a second, independently-locked world object. The embedder owns its state; it does
not own a second lock. That is the version of "attachable" that keeps the invariant,
and it also bounds *how deep* the reach can be: as deep as `Sim`'s, no deeper.

### 9.1.1 The `WorldRoutes` trait (Step 7)

The device machinery is world-agnostic: the token auth (§8.3), the near-you framing
(§6), the `query`/`invoke` verbs (§5), the focus and the schema→stencil (§11), and the
personal roots (`/history`, `/phone`, `/self`, §7.2) are all npcd's and know nothing of
what world sits behind them. Everything world-specific is one trait:

```rust
/// A world the effector device can address. npcd ships the vault as the default
/// implementation; a project embedding npcd supplies another. The provider is held
/// behind npcd's ONE world lock (for the vault that lock is `Hosted`; an embedder's
/// state lives behind the same lock, like `Sim`), so no second lock is introduced
/// (§9.1) — npcd acquires the lock and then calls these, never the reverse.
pub trait WorldRoutes: Send + Sync {
    /// The situated top-level addresses reachable from a caller's standpoint — the
    /// world's half of the near-you index (§6). npcd adds the personal roots around
    /// this; the provider returns only what its own world affords from here.
    fn reachable(&self, caller: Caller) -> Vec<RouteEntry>;

    /// A thing's state — `GET http://local/<ns>/<id>` — or `None` if the id names
    /// nothing the caller can reach (→ 404). A pure read.
    fn read(&self, caller: Caller, resource: &str) -> Option<Value>;

    /// A verb's one body schema — `OPTIONS http://local/<ns>/<id>/<verb>` (§5.1,
    /// §5.2) — the schema the `invoke` stencil is compiled from; `None` if the verb
    /// is not afforded here. Reading a thing (no verb) lists the verbs beneath it.
    fn schema(&self, caller: Caller, resource: &str, verb: Option<&str>) -> Option<Value>;

    /// Enact a verb — `POST http://local/<ns>/<id>/<verb>` — returning the world's
    /// verdict (`Did`/`Refused`), which npcd maps to `200`/`409` (§12). Runs under
    /// the write lock; proximity, custody and mode are the provider's to enforce and
    /// to phrase, exactly as the vault's act handlers do.
    fn invoke(&self, caller: Caller, resource: &str, verb: &str, args: &Map<String, Value>)
        -> RouteOutcome;
}

/// Who is acting, world-neutrally: the body the token resolved to and how far it
/// reaches (§8.3). The provider maps `npc_id` to its own notion of standpoint.
pub struct Caller { pub npc_id: u64, pub scope: Scope }
```

`RouteEntry` is `{ url, summary }` (§6); `RouteOutcome` is the `Did`/`Departed`/
`Refused` the envelope already maps (`effector/enact_route.rs`). Interior mutability is
not the provider's concern — npcd holds the single lock around every call, so the
methods take `&self` and mutate the state that lock guards.

**The vault is the reference implementation, not a special case.** npcd's own
`WorldRoutes` impl is exactly the wiring built this session, read through the trait:
`reachable` = `within_reach` + the lift landing + `instances_at` mapped through
`namespace_of` (§C.0); `read`/`schema` = the station and lift `GET`/`OPTIONS` handlers;
`invoke` = synthesise the `Act` and run `body::perform` (§C, D.5). Factoring it behind
the trait is a lift-and-name of code that already exists and is green — the change is
that the router calls `provider.reachable(...)`/`read`/`schema`/`invoke` instead of
naming the vault's functions directly, and `Runtime` holds a `Box<dyn WorldRoutes>`
(installed after construction, defaulting to the vault) beside the world lock.

**Replace, not compose (Q4).** One active `WorldRoutes` per daemon: an embedder's
provider *replaces* the vault's; npcd keeps only the personal roots around it. A world
that wants both the vault and its own affordances composes them inside its own impl,
not in npcd — which keeps npcd with exactly one answer to "what is reachable here."

*Not built this session (design only): the trait itself. The vault's routes ship as
concrete modules; Step 7 is the refactor that names the seam. It is deferred to when a
second world (an embedder) is real, because a trait with one implementation is a shape
guessed rather than a seam proven — the interface above is the target that refactor
lands on.*

## 9.2 Some routes write prompt-visible state, not just world state

Most effector calls change the world and are perceived over ticks (§10). A few
change the character's *own maintained state* — its plan, its orders, its beliefs —
which is not perceived as an event but *projected into the dynamic system prompt* and
carried every turn. `plan_*` and `orders_*` are the clearest case: they are the
mission machinery, they are edited through the effector device like anything else,
**and the plan must stay visible in the part of the system prompt that maintains
it** — a character with a plan reads its plan each turn; it does not re-discover it
from an event feed.

So a route handler's write target is one of two stores, and the route declares
which:

- **World state** — `Hosted`'s `world`/`sim`/`rooms` (`world/mod.rs:41`); the
  effect is perceived over ticks (§10). The lift, the command table, a posted
  notice.
- **Projected state** — the character's agency/belief/memory layers in the
  substrate, the same authoring plane the operator API writes
  (`npc_api_gui_design.md` §10; `/v1/npc/{id}/agency|beliefs`). The effect is
  re-projected into the next turn's system prompt, not narrated. The plan, the
  orders held, a revised belief.

This is not a new mechanism — it is the existing authoring plane and projection,
reached through the effector device instead of the operator API. `plan_*`/`orders_*`
write the agency layer; the projection resurfaces it; the character reads its plan in
the dynamic system prompt exactly as it does today. Appendix A must route these to
the projected store, not the world store — flagged there.

*As-built correction (Appendix E "The bridge"): this is now **met**. The bridge is
built — the cast is shared as `Arc<tokio::sync::RwLock<Npcs>>` and installed on
`Runtime` (`runtime.rs::set_npcs`/`npcs`), reusing the one `Npcs`/substrate handle,
never a second. `plan_*`/`orders_*` route through `effector/plan.rs` (mounted at
`/plan` and `/orders`, and excluded from the generic station nest so they never reach
`Sim.ledger`) to `Npcs::put_strategy_self` (`npcs.rs`), an owner-blind write of the
character's own `agency` layer that projects on the next turn via the `agency`
collection and `persona::intent`. The write target is `Npcs::put_strategy`, not
`engine/authoring.rs` (which is only a life-document parser).* (The one hard invariant
the authoring plane already carries, and which this must not breach: a belief-write
is the character's own, never a tool's silent edit — `npc_mind_design.md`; the
effector's belief routes are the character revising itself, on the record.)

## 9.3 The four stores a handler touches

The effector API is an envelope over existing state, not a new store — but "existing
state" is **four** stores, not the two (world / projected) the sections above imply.
Appendix D is the grounded account; this is the taxonomy every handler is written
against:

1. **The npc-map `World` (RAM)** — who is where, what each body holds, the lift,
   witnessed events. Not persisted; bodies re-enter deterministically on restart
   (`runtime.rs:776`). Written by the world acts (lift, claim/release/operate/…,
   `enact.rs`). Claim = proximity + `Actor.hold`.
2. **`Sim` (RAM)** — everything about the world that is not its map: the `record`
   authored-content state machine (`sim/record.rs`), the `bench` per-body working
   copy (`sim/bench.rs`), the `ledger` (orders/verdicts), missions, phone. **This is
   what most authoring namespaces actually write** — Appendix C's `Store = World` for
   them means `Sim` here. Runtime-only, so anything left in `Sim.record`/`ledger` and
   not committed through `bench` does **not** survive a restart (D.6 #1). Claim =
   `Record` custody + bench isolation.
3. **The mind folder (disk)** — the authored lore: canon, eras, stories, the cast,
   the craft libraries, the map source (`D:/prog/mind`, §D.1). Atomic writes. Two
   doors that never conflate: the **console** editor (`npcd::mind`, admin-gated,
   §D.3) and the **effector** working copy (`Sim.bench`, claim+commit gated, §D.4).
   Only a subset of authoring verbs reach it (chronicle-era, story, portrait-prompt,
   craft-library, and the raw `bench_`/`file_` surface); the rest stay in `Sim` RAM
   (D.5).
4. **The substrate (projected, persisted)** — agency, belief, memory; redo-logged;
   the belief-write invariant (§9.2). This is §9.2's "projected store." **No station
   handler writes it yet** — the bridge from `work.rs` to the authoring plane
   (`engine/authoring.rs`) is to-build (D.6 #2).

The rule stands: a handler reads and writes the store that already owns that fact,
never a parallel one. An **attachable** world (§9.1) brings its own state behind the
one lock, exactly as `Sim` does. Appendix D.5 maps every authoring namespace to its
real store; D.6 lists the durability and projection gaps a full migration must close.

## 10. The consequence loop — invoke acts, the world answers over time

`invoke` returns immediately; what it sets in motion unfolds over ticks and returns
through perception, not the return value. `invoke("http://local/lift/command-shaft",
{level:"command"})` returns `{ ok:true, detail:"The lift is on its way." }` that
turn. The car then travels on the world clock — `step_lift` inside `tick`
(`world.rs:928`, `world/mod.rs:172`) voices each `Moment` as a `Stirred` event on
the right landing — and the character learns it arrived the way it learns anything
that happened while it was not looking: through `witness`/`stir` →
`perceived::digest` → the narrated perception at the head of its next input
(`witness.rs:151`, `perceived.rs:84`, `mind.rs:1473`), and through the near-you
index, which at the landing now shows the car boardable.

The loop: **`invoke` → acknowledgement now → world ticks → effects perceived +
near-you index changes → the character queries/invokes again.** The return value is
the receipt; the world is the answer. The tick ordering — journeys, then the
building's say, then perception (`environment.rs:261`) — is unchanged; the device is
another writer under the same lock.

---

# Part D — The stencil (why the model calls it well)

## 11. The query drives the stencil — the mechanism that makes this cheap

This is the load-bearing idea, and it needs stating precisely, because a review of
this design found the first draft overclaimed it.

The stencil turns a JSON tool description into a constrained-decode tree
(`compile_tool_call_tree`, `stencil_tree.md` §8.3). What it *enforces* exactly: a
**string enum** decodes to one of its arms; a **boolean** to `true`/`false`. What it
*shapes but does not type-enforce*: `integer`, `number`, `array`, `object` decode as
any structurally-valid JSON of that shape, lookahead-terminated (`tool_call.rs:13`),
not range- or element-checked. So:

- `level: enum[command, casting, …]` is grammar-guaranteed correct.
- `count: integer` is guaranteed to be *a number*, not that it is `1..10`.
- `points: array<string>` is guaranteed *an array of strings*, not their contents.

This is still the right escape from §1's wall — the cost is *per-resource*, one
lookahead node, never the catalogue squared — so rich types are affordable where they
were not. But it means **the world's validation is load-bearing, not a backstop**:
the schema's enums are enforced by the decoder; its ranges, lengths, and cross-field
rules are enforced by the handler, which returns a prescriptive error (§8.4, §12) the
character corrects against. The schema mixes strong and weak typing per field; the
decoder enforces the strong half it can, the handler enforces the rest.

**How the schema reaches the stencil without breaking the grammar cache.** The
whole-turn grammar is cached on `(Deliberation, Within)` and hits almost always
*because `Within` is a room fact shared across the cast* (`mind.rs:174`). A
per-character, per-resource `invoke` body must **not** enter that key, or the hit
rate collapses and a fresh whole-turn tree compiles on the tick path every focused
turn (`compile_action_loop` cannot memoise across a spliced tree, `tools.rs:1919`).
So:

1. The turn grammar is the **fixed frame** — body acts, `query`, `invoke` — keyed on
   `Within` alone, cached as today (and, post-§15, with a much smaller `Within`).
2. The **invoke body** is a *separate* small sub-stencil, compiled from one
   resource's schema and cached by `(resource-id, schema-fingerprint)`, composed with
   the fixed frame at decode time. The composition reuses the as-built stencil
   driver's tree-swap machinery — the `TriggerRegistry` / `with_trigger` mechanism
   that already swaps a per-turn `<think>` tree onto the tool-call base
   (`stencil_tree.md`, as-built deltas) — armed on the `invoke` branch for the focus
   resource. This is §13's caching rule applied to the stencil: it keeps the
   near-perfect hit rate on the frame while giving `invoke` its typed body.

*As-built (Step 4): the **compile-time-splice fallback**, not the decode-time swap.*
*A true decode-time tree-swap on the `invoke` branch is not reachable without editing*
*the turn driver (`candle-conversation/src/conversation.rs`, a reserved WIP file): a*
*character turn is entered directly inside a single prefilled `turn_grammar` tree, not*
*driven through the `TriggerRegistry` the assistant/think path uses, so there is no*
*live branch to arm a swap on. So the acceptable fallback of the build order is what*
*shipped: the focus's typed body is a `{ … }` sub-stencil compiled by front-end B*
*(`compile_invoke_body_tree`, `ToolSpec::from_json_schema` over the OPTIONS body*
*schema) and cached by `(resource-id, schema-fingerprint)`; a focused turn's grammar is*
*the ordinary frame with that sub-stencil **spliced** onto the `invoke` body at compile*
*time (`compile_action_loop_with_body`), cached in its own map keyed by the frame key*
*plus `(resource-id, fingerprint)`. Blocker 1 holds exactly: the `(Deliberation, Within)`*
*frame cache is never keyed by focus, so two characters with different focuses still*
*share the frame; a no-focus turn is byte-identical to the plain frame. Only the JSON*
*call styles get the typed body; a function-block body keeps its free span. Focus lives*
*on `Minds` beside the frame cache (`engine/effector_focus.rs`), set by a resource*
*`query`'s in-process `OPTIONS` in `Runtime::enact_device`, whose schema also rides back*
*inline. A multi-verb station arms nothing (the focus arms one body and cannot know the*
*verb), so its body stays free JSON. **Resolved by verb-as-resource (§5.2), not the*
*decode-time swap:** once each verb is its own queryable address, every focus is one*
*verb's single body and the compile-time splice types it — editing the reserved*
*`conversation.rs` is not needed at all. The remaining work is the §5.2 as-built delta*
*(each verb path gets its own `GET`/`OPTIONS`; the focus arms on a verb query), small*
*and additive; single-body things (the lift) and single-verb things are already typed.*

**How the character reaches a schema without spending a turn on it.** The naive path
— discover, `query` for schema, then `invoke` — is three turns to do one thing, and
at 2 Hz that is slow. Two changes collapse it, both keeping the fixed surface:

- The near-you index (§6) carries each reachable leaf's schema *inline* — a `GET`
  returning JSON, still ordinary HTTP — and it is re-projected into the dynamic
  system-prompt section *this* turn (§6), so the schema the `invoke` stencil needs is
  already in front of the model. No separate schema-query turn, and because it is a
  point-in-time section it never accumulates in the window.
- The engine **pre-arms the focus** for the leaf the body is standing at, so
  "operate the thing in front of me" is one `invoke`, no explicit `OPTIONS`. Bounded
  to the reachable leaves, so it does not reopen the cache-key explosion.

The `url` of `query`/`invoke` is itself constrained: an enum of the currently-
reachable ids with a free-string fallback for deeper paths (post-migration the frame
is tiny, so there is ample path budget to spend on it, `tools.rs:1946`). A
hallucinated address is thus unlikely, and a wrong one 404s cheaply (§12). With no
focus at all, `invoke`'s `body` decodes as `FreeText{Balanced '{','}'}` — free,
well-formed JSON (`stencil_tree.md` §6.1) — and validation plus prescriptive errors
catch the rest.

**The schema is live, so it is re-read each focused turn.** Between the turn a schema
is seen and the turn `invoke` fires, the world ticks and other bodies act under the
one lock, so an `OPTIONS` enum (`floor_names()`, `claimable_at`, `modes_here`) can go
stale — the car arrives, a floor opens, someone claims the terminal, the body walks
and the route 404s. The focus's schema is recomputed at the start of each focused
turn, not reused from when it was first seen; §13's "pure function of world state" is
what makes that safe and cheap. A schema that went stale mid-flight fails benignly —
the world refuses and says why (§12) — but the design does not pretend the first read
is durable.

The API stays stateless (§13); the per-character *focus* that arms the sub-stencil
lives in the engine, beside the grammar cache — the same split the engine already
keeps between the world read-cursor (`Actor.looked`) and what was last shown
(`Attention`) (`delta.rs:28`).

## 12. Prescriptive errors close the loop the model already knows

A wrong `invoke` comes back as a real API error (§8.4): a status and `{ error,
detail, field }`. The character reads it next turn as a `<tool_response>` — the
channel act outcomes already ride back on (`record_act`, `runtime.rs:1481`;
`compose_narrated`, `mind.rs:1473`) — and corrects, because correcting a JSON API
against its error body is a thing the model has done a million times. The rule: **a
device error is what a well-behaved REST API would return, and nothing more
clever** — an invented format would forfeit the whole prior the wager rests on.

## 13. Statelessness — the engine calls the API whenever it needs the answer

The API is a pure function of `(world state, caller token → standpoint, method,
path, body)`. No session, no handshake, no cursor inside it. The engine calls it for
three reasons and none may leave residue: to fill the **near-you index** (`GET
http://local/`), to build the **invoke stencil** (the focus resource's schema, via
`OPTIONS`), and to render a **world event** referring to a capability. Because the
answer is a function of world state, asking repeatedly and from more than one place
is safe — two callers get the same answer, the property `perceive.rs` guarantees for
percepts (`perceive.rs:9`). Any caching is the *engine's*, keyed on
`(id, schema fingerprint, standpoint)`, never the API's.

---

# Part E — What this changes in the engine

## 14. What stays, what is new, what migrates

**Stays.** The body catalogue (`tools.rs:876`). The stencil machinery
(`candle-conversation/src/stencil/`). The two clocks and turn loop (`driver.rs`,
`character_loop`, `runtime.rs:3443`). Witness, stir, narration. The one-lock world
model. The per-character `ConversationLock`, projection/window/prompt structure.

**New.** `query` and `invoke` (§5). The `local` router + its external mount (§8).
`OPTIONS` schema responses (§5.1). The near-you index route (§6). The token auth and
quick-lookup table (§8.3). The in-process fast path (§8.1). The per-character focus
that arms the invoke stencil (§11). The world-routes extension point (§9.1). Item
ids on placed instances and the per-item state record (§7, §7.1; Q2). Routing of
`plan_*`/`orders_*` to the projected store so the plan stays in the system prompt
(§9.2).

**Migrates.** The world-touching half of `WORLD_ACTS` (handlers in `enact.rs`), all
`STATION_ACTS`/`bench`/`mission` (handlers in `engine/work.rs`, `work::perform` —
*shipped*, not the audit's "0-of-79"; Appendix D), and the reused `file_*` stop being
compiled `Tool` statics reached through the grammar and become routes reached through
`query`/`invoke`. For most verbs this is **wrapping an existing handler in a route
envelope**, not writing it — the audit's "build 115 tools" (`tool_surface_audit.md`
§6) becomes "declare 152 routes" over handlers that largely exist, at no grammar and
no `Within` cost. The genuinely-absent handlers (`plan_*`, `roster_*`, `room_sit`,
`character_write_beliefs`) and the durability/projection gaps are D.6. **Appendix A**
is the inventory, **Appendix C** the route map, **Appendix D** the store map.

## 15. Availability becomes routing; Choices becomes schema

- **Availability becomes the router's own answer.** A route not mounted at your
  standpoint 404s. `AtLift` is "the `/lift/<id>` routes are mounted where a landing
  is"; `AtPart` is "the part's routes are mounted within reach"; `Nearby` is "the
  handler checks company and refuses when alone." The empty-set rule
  (`tools.rs:1762`) becomes "the route is absent or its enum is empty." Same
  decision, expressed as HTTP.
- **Choices becomes handler-computed schema.** `Company`, `Floors`, `Reachable`,
  `Claimable` (`tools.rs:1196`) are enums a handler fills from world state at
  `OPTIONS` time — `http://local/lift/<id>/call`'s `level` enum *is*
  `world.floor_names()` (`world.rs:613`) at that moment. The world enumerates; the
  schema carries; the stencil enforces.

`Within` does not vanish — the body acts still gate through it and it is still the
grammar cache key for the fixed frame — but it stops growing a field per world
capability, the unbounded cost §1 named.

**And it shrinks — the migration's free win, and the review's top recommendation.**
Post-migration the body acts need only a few `Within` fields (`company`, `places`,
`feelings`, `cooling`, `me` — tell/whisper/ask/gesture→`Company`, move_to→
`Reachable`, reflect→`Feelings`, `tools.rs:1444`); every sim-derived field
(`operable`, `readable`, `claimable`, `device_modes`, `hostiles`, `makeable`,
`queues`, `threads`, …, `tools.rs:1529`) leaves `Within` because those acts are now
routes. The cache key shrinks, the hit rate rises, and the exponential path budget
(`tools.rs:1946`) stops being a live concern — which is exactly what makes the
fixed-frame grammar cheap enough to compose a per-schema `invoke` sub-stencil onto
(§11). The empty-set rule still applies to `move_to` (nowhere to walk) and the
`Nearby` speech acts (alone), so routing-becomes-availability does not remove
`Within`; it only stops it *growing*. This turns the grammar-cache and path-budget
risks from live problems into non-issues, and it is required by the migration
anyway — so it is done first.

## 16. Decisions made, and questions open

Made and defended above: two verbs, the model never picking the method, `OPTIONS` for
schema (§5, §5.1); one host `local`, ids in every URL, closed in fiction (§7); one
router, fast in-process path plus external socket mount (§8.1–8.2); one unified
`Principal`, token auth carried for the LLM, v1 = header+token match, external control
by **token scope** — as-NPC (proximity-gated) vs direct by-id (§8.3); `act`, `sleep`,
`promise`, `remind` stay body/human interactions, the phone acts and every
world/station verb become routes (§4); an embedder world **replaces** the vault, one
per daemon, under npcd's one lock (§9.1); fast JSON with field-level errors (§8.4);
the query's schema is the invoke's stencil, cached separately from the turn grammar
(§11); stateless API, focus in the engine (§11, §13).

Open — none. The product questions (external scope, act placement, embedder
replacement, schema verb) are decided above; the engineering calls are now settled for
the migration:

- **Route declaration** — handler in code, its path and schema declared beside it,
  mounted by the part's presence at a standpoint. The map says a room *has* a
  chronicle terminal; the code says a chronicle terminal *affords* these routes.
- **Sync/async** — the effector router is awaited only from the async turn loop
  (`character_loop`); handlers are `async fn` with *synchronous* world bodies that
  take the world lock, act, and release without awaiting. No `block_on`, no
  reentrancy — the no-reentry-under-lock invariant (§8.1) is what keeps it true.
- **Token** — persisted per NPC, so an external token survives a restart (§8.3); a
  real secret, never derived from `body_id`.
- **Route paths** — RPC-flavoured, one path per existing verb, for a faithful 1:1
  migration that keeps the audit's naming; REST-ify per route later if wanted.
- **Topology mutation** — deferred to its own cut. v1 is interaction-dynamic (routes
  over the world's mutable half plus the new per-item state, §7.1); topology-dynamic
  (the `MapSet` mutation surface) is Step 8, since the map is frozen after load today.

---

# Part F — Testing and build order

House conventions hold (`CLAUDE.md`): TDD with raw expected values, one concern per
file, no stubs, no env-flags, file tools only. Three properties make this testable:

- **The `local` router is `oneshot`-testable on CPU**, no GPU, no server — the path
  the existing routers use (`api.rs:448`). Every route's happy path, every error,
  every per-standpoint mount/404, every `OPTIONS` schema is a request/response
  assertion. The token check (§8.3) is pinned both ways (valid token passes; absent
  or mismatched fails; `X-Tokera-*` never read).
- **Schema→stencil is CPU-testable with the stencil simulator** (`stencil_tree.md`
  §14): a route's `OPTIONS` schema compiled to an invoke grammar, run across every
  field, asserting the emitted JSON parses and validates — the regression net that
  the device never emits an ill-formed call.
- **The consequence loop is an integration test** against the live world under its
  one lock.

**The test suite is also the latency guard.** Every device call on the hot path
takes the fast in-process route (§8.1), so a `oneshot` test *is* a call on that path
minus the decode. The suite must run **very fast** — the same bar `/fast-test` holds
(a binary over 20 s is a defect) — because a slow test is a slow tool call. Speed is
not a nicety here; it is the standing measurement that the 2 Hz world can afford a
character reaching for its device, and a regression in it is a regression in the
game's felt responsiveness.

**Port what exists, then go wide.** The world already has unit tests in `npc-map`
(lift, movement, witness, bumps) and `npcd` (tools, enact). They port onto the
effector abstraction — drive the world by `query`/`invoke` against the router and
assert the response *and* the resulting state — which is both a migration check and
the start of an **extensive** suite: every route, every error, every
availability-by-condition transition (§7.1), every schema→stencil path (§11). The
abstraction is what makes the world broadly unit-testable: a room, a machine, a
mode change is now a request and an assertion.

**Implement the missing world — but most of it exists.** Contrary to the audit's
"0 of 79" (`tool_surface_audit.md` §1), the handlers are shipped in `engine/work.rs`
(Appendix D); the migration wraps them in route envelopes. What is *genuinely* missing
is scoped in Appendix D.6: durable persistence for the RAM-only `Sim.record`/`ledger`
writes (character, place, record, orders — they do not survive a restart today), the
bridge from effector handlers to the substrate agency plane so `plan_*`/`orders_*`
actually project (§9.2), the few absent handlers (`plan_*`, `roster_*`, `room_sit`,
`character_write_beliefs`), and — if custody/history is to be real — git plumbing for
`bench_commit`/`blame`/`log`. Each lands with its tests, or it does not land.

Build order, dependency-first, each step green before the next:

1. The `local` router beside the three, with `GET http://local/` (near-you index)
   and the **device-only middleware** of §8.3 (token → body; ignores `X-Tokera-*`;
   *not* `guard::Api`); the `sites:` entry, the external mount, the in-process fast
   path with the `Router` handle retained on `Runtime`, and the no-reentry-under-lock
   invariant; `oneshot`-tested including the `X-Tokera-*` refusal both directions.
2. Item ids on placed instances (Q2), the per-item state record (§7.1), and
   standpoint-filtered mounting — with the availability model of §7.1 mapped out in
   full (every part, its conditions, the verbs each condition affords).
3. `query`/`invoke` as the fixed-frame tools; `OPTIONS` schema; the "Your effector
   device shows:" band and the system-prompt copy (§6.1).
4. Schema→stencil for the focus resource: the `invoke` sub-stencil cached by
   `(resource-id, schema-fingerprint)` and composed with the `Within`-keyed frame
   (§11) — *not* folded into the grammar cache key; inline schema on the near-you
   index and focus pre-armed for the standing leaf (so the common call is one turn);
   the simulator suite. This rides on Step 6's `Within` shrink (§15), so the two
   proceed together.
5. The lift, migrated end-to-end (`http://local/lift/<id>/{status,call}`, enum from
   `world.floor_names()`, consequence loop tested) — the smallest complete example.
6. Bulk migration of `STATION_ACTS` and the world-half of `WORLD_ACTS` to routes,
   **implementing each handler for real** (the audit's 0-of-79 gap closes here) with
   its tests, and deleting each corresponding `Tool`, `Choices`, `LIVE` row, and
   `Within` field as it moves. `plan_*`/`orders_*` route to the projected store, not
   the world store (§9.2). Appendix A tracks it.
7. The world-routes extension point (§9.1), with the vault as the first
   implementation and a second, trivial implementation as the test that it is truly
   attachable.
8. (Later, its own cut) the `MapSet` mutation surface and topology-dynamic routes.

---

# Part G — Sharp edges (from adversarial review), and where they are handled

The design survived an adversarial review against the code; the core (compiled body
acts + a fixed two-tool device over an in-process real router, per-resource
schema→stencil, attachable world) held. These are the edges it found, each with
where this document now handles it. "Blocker" = must be solved before or during
Step 1–2; "sharp" = scoped but real; "minor" = watch.

| # | Sev | Edge | Handled in |
|---|---|---|---|
| 1 | blocker | Focus must not enter the `(Deliberation, Within)` grammar-cache key, or the hit rate collapses and a whole-turn tree recompiles on the tick path | §11 (separate `invoke` sub-stencil cached by `(resource-id, schema-fingerprint)`) + §15 (shrink `Within`) |
| 2 | blocker | The device surface cannot reuse `guard::Api` — it hardcodes the `X-Tokera-*` role check; and the fast path sits behind a header-trusting gateway | §8.3 (device-only middleware; `X-Tokera-*` never read; token both-ways test) |
| 3 | blocker | Stable per-instance ids do not exist: `Placement` is anonymous, `part_ids_at` de-dups by catalogue id, world state keys on place+name | §7, §9 (id = `(place, part-id, ordinal)`; remove de-dup at `load.rs:271`; widen sim key) — Step 2 |
| 4 | sharp | Sync/async seam: `within`/act path are sync inside an async loop; a router call is a `Future`, `block_on` panics | §8.1 (async slice or `spawn_blocking`) |
| 5 | sharp | Re-entering the non-reentrant world mutex from a handler that calls the router deadlocks silently | §8.1 (no-reentry-under-lock invariant, tested) |
| 6 | sharp | "Strong typing" only truly binds string enums and booleans; `integer`/`array`/`object` are shaped, not enforced | §11 (world validation is load-bearing, not a backstop) |
| 7 | sharp | The `{error,detail,field}` shape does not exist yet (`err()` has no `field`), and path→field mapping is real work | §8.4 (add `field`; it is the correction signal, not plumbing) |
| 8 | sharp | The focus schema can go stale between the `OPTIONS` turn and the `invoke` turn | §11 (re-read schema each focused turn; stale fails benignly) |
| 9 | sharp | An embedder world with its own lock reintroduces the two-lock bug the design avoids | §9.1 (provider runs under `Hosted`'s one lock, like `Sim`) |
| 10 | minor | `within` is already non-atomic (~8 lock acquisitions); the device adds reads | §13 note; fold into a `with`-scoped builder if it grows |
| 11 | minor | An external token dies across restart unless persisted; must not be derived from `body_id` | §8.3 (real secret; persist — Q6) |
| 12 | minor | The `oneshot` router is `#[cfg(test)]`; production keeps no handle | §8.1 (plumb the handle onto `Runtime`) — Step 1 |

The review's own bottom line: solve **#1 via the `Within` shrink (§15) first** — it
is required by the migration anyway and turns the two hardest problems (grammar cache,
path budget) into non-issues — then the device middleware (#2) and the id minting
(#3). Everything else is sharp-but-scoped.

# Appendix A — The migration inventory (for review)

The tools that **stay** (the body — always present, never a URL):

| Tool | Why it stays |
|---|---|
| `tell` `whisper` `shout` `ask` `gesture` | speech is a body act, `Availability::Always`-class |
| `move_to` `follow` | movement is a body act |
| `reflect` | an internal act, not a world interface |

Placement (decided): `act`, `sleep`, `promise`, and `remind` stay **body acts** —
human interactions, always available, never a URL. The phone acts (`send_image`
`message` `invite` `open_group` `reach_out`) become `http://local/phone/...` routes.

The tools that **become routes** (removed as `Tool` statics; each row is a route
`METHOD http://local/<ns>/<id>/<verb>`, GET/OPTIONS for reads, POST for effects):

| Namespace | n | Becomes | From |
|---|---|---|---|
| `lift` | 2 | `.../lift/<id>/{status,call,use}` | `LIFT_CALL`, `LIFT_USE` (`acts.rs:682`) |
| world-state acts | ~18 | `.../<thing>/<id>/<verb>` | `GIVE READ POST_NOTICE CLAIM RELEASE EQUIP USE GATHER ENGAGE OPERATE RECALL SCAN COMMAND_TOWER PRODUCE RECORD_VERDICT SIGN_OFF` (`acts.rs:1311`) |
| `chronicle` | 8 | `.../chronicle/<id>/{read_era,rewrite_page,add_entry,…}` | audit §2.2 |
| `record` | 16 | `.../record/<id>/{appraise,let_go,accession,…,cross_reference,leave_note,tidy_index,hand_on}` | audit §2.2, §6.2 |
| `story` | 9 | `.../story/<id>/{read_ledger,draft,file,read_aloud,…}` | audit §2.2 |
| `portrait` | 9 | `.../portrait/<id>/{draw,redraw,read_hung,settle_likeness,…}` | audit §2.2 |
| `character` | 6 | `.../character/<id>/{read,write_identity,write_wants,…}` | audit §2.2 |
| `place` | 6 | `.../place/<id>/{read_index,settle_route,write_entry,…}` | audit §2.2 |
| `map` | 5 | `.../map/<id>/{read,move_border,add_place,drown_place,redraw_coast}` | audit §2.2 *(topology-mutating — needs §16/Q; Step 8)* |
| `orders` | 6 | `.../orders/<id>/{read,take,give_back,report_done,set,hand_to}` | audit §2.2, §6.2 |
| `enquiry` | 5 | `.../enquiry/<id>/{take_question,answer_from_record,name_the_gap,read_history,raise_work}` | audit §2.2, §6.2 |
| `dispatch` | 4 | `.../dispatch/<id>/{read,read_holds,post_wake,read_wake}` | audit §2.2, §6.2 |
| `plant` | 3 | `.../plant/<id>/{read_panel,note_drift,raise_fault}` | audit §2.2 |
| `stores` | 3 | `.../stores/<id>/{put_back,take_out,walk_the_racks}` | audit §2.2 |
| `structure` | 3 | `.../structure/<id>/{lay_out_scenes,test_the_want,find_the_slack}` | audit §2.2 |
| `roster` | 2 | `.../roster/<id>/{read,take_unheld}` | audit §2.2 |
| `cast` | 2 | `.../cast/<id>/{read_all,report_disagreement}` | audit §2.2 |
| `room` | 1 | `.../room/<id>/sit` (`room.talk` already removed, audit §3.1) | audit §2.2 |
| `creator` | 1 | `.../creator/<id>/present` | audit §2.2 |
| `plan` | 5 | `.../plan/<id>/{break_down,order,reorder,scope,read}` | audit §6.2 |
| `trial` | 3 | `.../trial/<id>/{keep,read_failures,compare}` | audit §6.2 |
| `standard` | 3 | `.../standard/<id>/{read,propose,settle}` | audit §6.2 |
| `gather` | 2 | `.../gather/<id>/{call,read_standing}` | audit §6.2 |
| `bench` | 11 | `.../bench/<id>/{branch,stash,diff,restore,stage,commit,status,blame,log,…}` | audit §5.2 |
| `file` | 5 | `.../file/<id>/{read,edit,write,list,delete}` (reused from `zend-tools`) | audit §5.3 |

Counts are the audit's; the exact per-verb method (GET vs POST) and any RPC→REST
reshaping is settled per route as it migrates (**Q7**). This table is the checklist
Step 6 works down.

**`plan` and `orders` write projected state, not world state (§9.2).** They are the
mission machinery: editable through the effector device, but their handlers write the
agency layer so the plan and the held orders are re-projected into the dynamic system
prompt each turn rather than perceived as events. Their routes read/write the
substrate authoring plane, not `World`/`Sim`.

**Appendix C is the executable form of this table** — every route with its method,
condition, body schema, store, and `file:line` source, grounded in the actual vault
map (~152 v1 routes across 27 namespaces). Building it surfaced one namespace missing
from this inventory *and* from the audit: the shipped `library`/craft acts
(`LIBRARY_READ`/`WRITE`, `station.rs:333`/`345`), whose writes change how every
character in the world feels or answers — added there as C.7.

---

# Appendix B — Documentation to align with this design

This design changes a mechanism that a dozen docs describe. The rule for the whole
sweep: **the mechanism is superseded, the analysis of *what the world must afford*
is retained** — the clusters, verbs, currencies, and store-mappings become the
specification for the routes' *contents*, only the way a character reaches them
changes. Ordered by how load-bearing the contradiction is.

**Tier 1 — central mechanism superseded (banner + section rewrite).**

| Doc | What is now stale | What survives |
|---|---|---|
| [`tool_surface_audit.md`](tool_surface_audit.md) | the whole "build all 115 as compiled tools" plan (§1, §6, §6.4); the `namespace_verb` naming/tokenisation analysis (§5A) is moot for URL path segments; the counts (§6.3); and its "Implemented: 0" claim is itself stale — the handlers ship in `engine/work.rs` (Appendix D) | §2 (what exists), §3 (removals), §4 (the 8 missing clusters), the craft-word naming rule for verbs |
| [`tool_world_acts.md`](tool_world_acts.md) | world-acting verbs as compiled body tools; the string/boolean workaround (§0) — lifted by per-resource schema (§11); the "seven new live sets" (§4) become handler-computed `OPTIONS` schema | the intent-vs-typed field rule; §6 "to simulate this in the vault" |
| [`tool_decisions.md`](tool_decisions.md) | "vocabulary moved onto the argument axis" — it moves onto the **route axis**; `read`/`claim`/`release` as body tools (§2, §3) reopen (read → `query`) | §1 belief-write prohibition (reinforced, §9.2); tool-shape/argument-binding sections |
| [`tool_interaction_reconciliation.md`](tool_interaction_reconciliation.md) | "25 `*_read_*` tools vs one `observe`" (§4.2) — reads are now `query`; the 25-reads open question dissolves | interaction modes (§1, §3), interlocutor resolution; note phone acts' placement is Q5 |
| [`tool_world_state.md`](tool_world_state.md) | the "129 tools" framing; `bench_`/`file_` as tools | most of it — the four stores and the namespace→store table become the **route-handler** spec (§9) |

**Tier 2 — engine & wire contract (section rewrites + additions).**

- [`npc_engine_design.md`](npc_engine_design.md) Part VII — "Availability is a
  product of location" keeps its conclusion (a not-mounted route 404s) but its
  mechanism becomes routing (§15); the `trait NpcTool` / `ToolRegistry::register`
  extension story is re-scoped to the attachable world-routes provider (§9.1).
  "Tools carry intent, not output" stays.
- [`npc_api_gui_design.md`](npc_api_gui_design.md) — `/v1/tools` and `ToolInfo`
  (§9, §10) now describe only the compiled frame (body + `query`/`invoke`); add a
  new operator route exposing the `local` route map for console inspection; document
  the external effector mount `/v1/local/...` and its **token auth** (§8.2–8.3) as a
  new auth path beside the gateway headers. It already draws "Battle Cities
  (embedder)" and states "the HTTP layer is a transport over the core" — this design
  extends that to the world itself (§9.1). Error shape and intent/rendered split are
  adopted unchanged.

**Tier 3 — vault & repertoire (correction notes; content survives as route spec).**

- `vault_world.md` — "the tools hang on the part… `MapSet::tools_at(node)`" becomes
  routes addressed by the part instance's id, mounted per standpoint (§9); the near-
  you index replaces `tools_at`; `binds`/`modes` become mounting + methods; the
  "appear in reach and refuse" line is stale. Grows the stable-instance-id the
  design needs (Q2). The building/graph/memory design is untouched.
- [`maker_repertoire.md`](maker_repertoire.md) — invariant 8's "reach and refuse"
  gloss and the dotted identifiers are stale; the 480 tasks, the currencies-as-types,
  and the invariants remain the spec for what routes must afford.
- [`asynchronous_mind_hierarchy.md`](asynchronous_mind_hierarchy.md) — "tools fixed
  at the boundary" still holds; the bottom-level fixed catalogue is now the effector
  device, and §5's git work-surface is now `.../bench/<id>/...` and `.../file/<id>/...`
  routes. Core argument unaffected; reconcile §1.1 and §5.

**Tier 4 — index & cross-reference.**

- `docs/README.md` — add `local_api.md` (the fourth NPC surface) and the `tool_*`
  series, which it does not currently list.
- [`web_gateway_design.md`](web_gateway_design.md) — cross-reference note that npcd
  now registers a `local` effector router plus the `/v1/local/...` external mount.

**Agrees, no change (a reader might expect otherwise):**
[`stencil_tree.md`](stencil_tree.md) is the reused machinery §11 depends on (the
string/boolean limit lives in `tools.rs`, not here); `reflect`
([`reflection_and_dreams.md`](reflection_and_dreams.md)) is preserved as a body act;
[`narrative_engine.md`](narrative_engine.md) is unchanged (device acts return JSON
receipts, body acts still carry intent); [`npc_mind_design.md`](npc_mind_design.md)'s
belief-write protection is reinforced (§9.2), not contradicted.

One decision touches several docs: `act`, `sleep`, `promise`, and `remind` stay body
acts (human interactions), while the phone/messaging acts become `http://local/phone`
routes (§4). `tool_world_acts.md`, `tool_interaction_reconciliation.md`, and Appendix
A must be made to agree with that split.

---

# Appendix C — The vault route map

*The concrete route map the design references at §7.1 ("**Appendix C is that map** for the
vault"). It turns Appendix A's checklist into an executable specification: every part that
becomes routes, the routes each affords, what gates them, the body schema each `invoke`
takes, and the store the handler reads or writes. It is grounded in the shipped vault
(`npc-map/maps/*.yaml`), the shipped act tables (`npcd/src/engine/{acts,station}.rs`), the
availability/choices machinery (`npcd/src/engine/tools.rs`), and the current handlers
(`npcd/src/engine/enact.rs`). Where a namespace is authored in the audit but has no handler
yet (the audit's 0-of-79 finding, `tool_surface_audit.md` §1), the row cites the audit and
is flagged **[no handler yet]** — that route is the work Step 6 lands, per Part F.*

Conventions used in every table below:

- **Method** — `GET` = read (issued by `query`; `OPTIONS` is implicit on every route and
  returns the schema, §5.1). `POST` = effect (issued by `invoke`; the model never picks the
  method, §5). RPC-flavoured one-path-per-verb, per the settled decision (§16 "Route paths").
- **Condition** beyond reachability (§7.1): `always` (reading affords it once within reach),
  `claimed` (requires holding the part's `binds` — its working mode, entered by `bench_branch`),
  `paired` (a settling table: needs the other holder present, the `Nearby`-class two-body
  gate), `affordable` (the `OPTIONS` enum is non-empty — the economy is enforced in the schema).
- **Body schema** field tags: `enum(src)` = grammar-enforced closed set filled from the named
  live-set at `OPTIONS` time (§15, `tools.rs:1443` LIVE / `1421` FIXED); `typed(kind)` =
  integer/number/array/object — **shaped, not enforced**, so the handler validates (§11);
  `free` = open string. `req`/`opt` = required/optional.
- **Store** — the column below records the *intended* target and is **superseded by
  Appendix D.5**, which maps each namespace to the store the shipped handler actually
  touches: almost every authoring row marked `World` is really `Sim` (its `record`/
  `ledger`), and a subset also reaches the **mind folder** (the authored lore) through
  the `bench` working copy (§D.4). `projected` = the substrate authoring plane (§9.2),
  which no station handler writes yet (D.6). The dispatcher for these handlers is
  `engine/work.rs` (`work::perform`), not `station.rs` (declaration only) — so the
  `[no handler yet]` flags mark *reads and genuinely-absent verbs*, not the shipped
  writing handlers (see D.5).
- **id pattern** — `<part-id>~<ordinal>` (§7). The ordinal is 0-based and world-wide (across
  the whole map in a deterministic walk), so a singleton is `<part>~0` and same-part
  placements take consecutive numbers without the place in the string. The `<id>` placeholder
  in each table below is the old long `<area>~<node>~<part>~<ordinal>` form; the shipped id is
  the short one (`order-table~0`, not `vault-command~command-room~order-table~0`).

---

## C.0 The availability matrix — which parts stand where in the vault

From the six vault level maps and `creators-vault.yaml`. "Ordinal?" = does any node place
more than one, so a route id must carry the ordinal to be unique (§9, the removed
`load.rs:271` de-dup). Parts marked **(war world)** carry world-acts but are placed only in
`tower-redoubt.yaml`, never in the vault — listed because Appendix A routes their verbs.

*(As-built, Step 2: the "id pattern" column below is shorthand. A route's full address
is `http://local/<ns>/<instance-id>`, where `<instance-id>` is the concrete
`<area>~<node>~<part-id>~<ordinal>` of §9 — e.g. a band-one character terminal is
`http://local/character/vault-casting~band-one~character-terminal~0`. The `<node>-<n>`
here just names which node and which ordinal.)*

| Part | kind | binds | modes | Vault level → node(s) | Ordinal? | id pattern (ns/…) |
|---|---|---|---|---|---|---|
| chronicle-terminal | station | one era | reading/working/offered | chronicle → early-range ×6, middle-range ×6, late-range ×4, catalogue-room ×2 | **yes** | `chronicle/<node>-<n>` |
| character-terminal | station | one character | reading/working/offered | casting → band-one ×6, band-two ×6, band-three ×4 | **yes** | `character/<node>-<n>` |
| story-desk | station | one gap | reading/working/offered | story → first-room ×6, second-room ×6, third-room ×4 | **yes** | `story/<node>-<n>` |
| easel | station | one character | reading/working/offered | portraits → north-studio ×6, middle-studio ×6, south-studio ×4 | **yes** | `portrait/<node>-<n>` |
| survey-desk | station | one place | reading/working/offered | cartography → north-survey ×6, middle-survey ×6, south-survey ×4 | **yes** | `place/<node>-<n>` |
| map-table | station | whole geography | reading/working/offered | cartography → map-room ×1 | no | `map/map-room-1` |
| accession-desk | station | the intake | reading/working/offered | command → receiving ×1 | no | `record/receiving-1` |
| catalogue | fixture | — | reading/working/offered | chronicle → catalogue-room ×1 | no | `record/catalogue-1` |
| appraisal-bench | fixture | — | reading/working/offered | chronicle → sorting-room ×1 | no | `record/sorting-1` |
| mending-bench | fixture | — | reading/working/offered | chronicle → mending-room ×1 | no | `record/mending-1` |
| concordance-table | fixture | — | — | chronicle → concordance ×1 | no | `chronicle/concordance-1` |
| archive | fixture | — | — | chronicle → stacks ×1 | no | `chronicle/stacks-archive` |
| timeline-wall | fixture | — | — | chronicle → stacks ×1 | no | `chronicle/stacks-timeline` |
| enquiry-desk | station | an open enquiry | — | command → enquiry ×1 | no | `enquiry/enquiry-1` |
| order-table | fixture | — | — | command → command-room ×1 | no | `command/command-room-1` (Step 6 split it from `orders`; see §C.11) |
| creators-chair | fixture | — | — | command → command-room ×1 | no | `creator/command-room-1` |
| dispatch-board | fixture | — | — | command → dispatch ×1 | no | `dispatch/dispatch-1` |
| stores | fixture | — | — | command → receiving ×1 | no | `stores/receiving-1` |
| plant-panel | fixture | — | — | command → plant ×1 | no | `plant/plant-1` |
| gap-ledger | fixture | — | — | story → ledger-room ×1 | no | `story/ledger-1` |
| filed-stories | fixture | — | — | story → ledger-room ×1 | no | `story/ledger-filed` |
| structure-board | fixture | — | — | story → board-room ×1 | no | `structure/board-1` |
| reading-table | fixture | — | — | story → long-table ×1 | no | `gather/long-table-1` |
| watch-desk | station | (reads, claims nobody) | — | casting → watch ×1 | no | `cast/watch-1` |
| roster | fixture | — | — | casting → roster-room ×1 | no | `roster/roster-1` |
| relations-table | fixture | — | — | casting → relations ×1 | no | `character/relations-1` |
| plate-rack | fixture | — | — | portraits → plate-room ×1 | no | `portrait/plate-1` |
| house-palette | fixture | — | — | portraits → palette-room ×1 | no | `standard/palette-1` |
| hung-faces | fixture | — | — | portraits → gallery ×1 | no | `portrait/gallery-1` |
| likeness-table | fixture | — | — | portraits → likeness ×1 | no | `portrait/likeness-1` |
| place-index | fixture | — | — | cartography → index-room ×1 | no | `place/index-1` |
| gallery-rail | fixture | — | — | cartography → gallery ×1 | no | `map/gallery-rail-1` |
| road-table | fixture | — | — | cartography → road-table ×1 | no | `place/road-1` |
| seat | seat | — | — | every social/work node (counts 2–16) | n/a | `room/<node>-<n>` |
| muster-board **(war world)** | fixture | — | — | tower-redoubt → muster ×1 | no | `orders/muster-1` |
| bridge-console **(war world)** | station | the tower | — | tower-redoubt → bridge ×1 | no | `tower/bridge-1` |
| fabricator **(war world)** | station | one run | idle/running/paused/purging | tower-redoubt → fab-bay ×8 | yes | `stores/fab-<n>` |
| sensor-panel **(war world)** | fixture | — | — | tower-redoubt → bridge ×1 | no | (scan target) |

**New parts (audit §6.2) not yet in any map** — planning board (`plan_*`), trials shelf
(`trial_*`), standards board (`standard_*`). Their placement is a Step 2/6 map edit ("placed
in rooms that already exist", audit §6.2); routes below assume the standards board is reachable
from every level (audit §3.4), the planning board sits where planning happens, and the trials
shelf where experiments are kept. Ordinal `1` until a map places more than one.

**Standpoint → mount.** A route's routes are mounted wherever its part is within reach
(`within_reach`, `perceive.rs:280`); a not-mounted route 404s (§15). Because the vault is one
building with a single lift shaft (`creators-vault.yaml` portals, all `kind: a lift`), the lift
routes (C.1) mount at every level's `core` node.

---

## C.1 The lift — `http://local/lift/<id>`

id pattern: one shaft, id `lift/command-shaft` (the vault has a single car serving all levels,
`vault-command.yaml` core "one car serves every level"). Mounted at the `core` node of each
level.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../lift/<id>` | GET | at a landing (`at_landing`, `world.rs:545`) | — (returns `{here, level, moving}`) | World | new (status), design §5.1/§14 |
| `POST .../lift/<id>/call` | POST | at landing **and car elsewhere** (`AtLift`, `tools.rs:1750`) | — | World | `LIFT_CALL` acts.rs:682; enact `lift_call` enact.rs:85 |
| `POST .../lift/<id>/use` | POST | at landing **and car here** (`InLift`, `tools.rs:1751`) | `floor` enum(`world.floor_names()`, `world.rs:613`) req | World | `LIFT_USE` acts.rs:705; enact `lift_use` enact.rs:86 |

---

## C.2 The world-state half of `WORLD_ACTS` — `acts.rs:1311`

These are the sixteen Appendix A names ("world-state acts | ~18", `acts.rs:1311`). `read` and
`scan` are **reads → GET** (dissolved into `query`, per Tier-1 `tool_world_acts.md`); the rest
are `POST` effects. Several are not part-bound (`Availability` other than `AtPart`): their verb
is mounted on the target thing named by their live-set (`claim`/`release`/`post_notice`/
`record_verdict` on the room's claimable/postable/judgeable things; `give`/`equip`/`use` on the
body's own inventory as personal-ish world routes). Handler dispatch table: `enact.rs:71–98`.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../<thing>/<id>` (read) | GET | always (`Readable` non-empty) | — (returns contents) | Sim (`postings`/`ledger`) | `READ` acts.rs:289; enact.rs:236 |
| `GET .../scan/<place>` | GET | always (`Reachable` non-empty) | `at` enum(`Reachable`) req | Sim (`field`) | `SCAN` acts.rs:730; enact.rs:87 |
| `POST .../<part>/<id>/claim` | POST | always (`Claimable`) | — (target is the id) | World / Sim `ledger` | `CLAIM` acts.rs:386; enact.rs:311 |
| `POST .../<part>/<id>/release` | POST | holding it | — | World | `RELEASE` acts.rs:418; enact.rs:336 |
| `POST .../<board>/<id>/post_notice` | POST | always (`Postable`) | `what` free req | Sim `postings` | `POST_NOTICE` acts.rs:336; enact.rs:289 |
| `POST .../<thing>/<id>/record_verdict` | POST | always | `judgement` free req; `what_would_change_it` free opt | World | `RECORD_VERDICT` acts.rs:983; enact.rs:92 |
| `POST .../give` | POST | `Nearby` (someone here) | `what` enum(`Carried`) req; `to` enum(`Company`) req; `count` typed(int) opt | Sim (inventory) | `GIVE` acts.rs:226; enact.rs:205 |
| `POST .../equip` | POST | `Embodied` | `what` enum(`Equippable`) req | Sim | `EQUIP` acts.rs:436; enact.rs:79 |
| `POST .../use` | POST | `Embodied` | `what` enum(`Usable`) req; `on` enum(`Company`) opt | Sim | `USE` acts.rs:459; enact.rs:362 |
| `POST .../gather` (field) | POST | `Embodied` (`Extractable`) | `what` enum(`Extractable`) req | Sim `field` | `GATHER` (field) acts.rs:499; enact.rs:394 — **distinct from `gather_*` (C.24)** |
| `POST .../operate/<id>` | POST | always (`Operable`) | `mode` enum(`DeviceModes` — dependent on the chosen thing) req | World / Sim | `OPERATE` acts.rs:604; enact.rs:83 |
| `POST .../recall` | POST | `AwayFromHome` (`tools.rs:1743`) | — | World | `RECALL` acts.rs:644; enact.rs:84 |
| `POST .../engage` **(war world)** | POST | `Embodied` + `Postures` non-empty | `posture` enum(`Postures`) req; `target` enum(`Hostiles`) opt; `priority` enum(FIXED `sim::field::PRIORITIES`) opt | Sim `field` | `ENGAGE` acts.rs:529; enact.rs:82 |
| `POST .../tower/<id>/command` **(war world)** | POST | `affordable` (`TowerActions`) | `action` enum(`TowerActions`) req; `target` free opt; `x`,`y`,`depth` typed(number) opt | Sim (tower) | `COMMAND_TOWER` acts.rs:791; enact.rs:88 |
| `POST .../stores/fab-<n>/produce` **(war world)** | POST | `affordable` (`Makeable`) + fabricator `idle` | `what` enum(`Makeable`) req; `count` typed(int) opt; `queue` enum(`Queues`) opt | Sim | `PRODUCE` acts.rs:855; enact.rs:89 |
| `POST .../phone/<thread>/sign_off` | POST | on a leavable thread (`Leavable`) | `intent` free req | Sim phone / **projected** | `SIGN_OFF` acts.rs:1027; enact.rs:96 — **see C.28.2** |

`act`, `sleep`, `promise`, `remind` stay **body acts** and gain no route (§4, App A;
`ACT` acts.rs:95, `SLEEP` acts.rs:186, `PROMISE` acts.rs:901, `REMIND` acts.rs:940).

---

## C.3 `chronicle` — `http://local/chronicle/<id>`

Part: chronicle-terminal (station, `binds: one era`, modes reading/working/offered) — plus the
concordance-table (`chronicle_settle_boundary`) and read-only fixtures archive + timeline-wall.
id: `chronicle/<node>-<n>` (ordinal required — up to 6 per range).

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../chronicle/<id>` (read_era / read_any_page) | GET | always (reading mode) | — | World | `chronicle_read_era`,`read_any_page` audit §2.2 **[no handler yet]** |
| `POST .../chronicle/<id>/add_entry` | POST | claimed (working) | `to` free req; `what` free req | World | `CHRONICLE_ADD_ENTRY` station.rs:184 |
| `POST .../chronicle/<id>/rewrite_page` | POST | claimed (working) | `in` free req; `what` free req | World | `CHRONICLE_REWRITE_PAGE` station.rs:200 |
| `POST .../chronicle/<id>/retire_entry` | POST | claimed (working) | `what` free req | World | `CHRONICLE_RETIRE_ENTRY` station.rs:218 |
| `POST .../chronicle/concordance-1/settle_boundary` | POST | **paired** (both era holders present) | `between` free req; `and` free req | World | `CHRONICLE_SETTLE_BOUNDARY` station.rs:231 |
| `GET .../chronicle/stacks-archive` (read_any_page) | GET | always | — | World | archive.yaml; `chronicle_read_any_page` audit §2.2 **[no handler yet]** |
| `GET .../chronicle/stacks-timeline` (read_density / read_conflicts) | GET | always | — | World | timeline-wall.yaml; audit §2.2 **[no handler yet]** |

---

## C.4 `record` — `http://local/record/<id>`

Parts: accession-desk (station, `binds: the intake`), appraisal-bench, catalogue, mending-bench
(fixtures, all modes reading/working/offered — §5.4 gives them working modes so a mutation takes
the claim, closing audit §3.3). id per node (all singletons: ordinal 1).

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../record/<id>` (read_description / read_condition) | GET | always | — | World | `record_read_description`,`read_condition` audit §2.2 **[no handler yet]** |
| `POST .../record/receiving-1/accession` | POST | claimed | `what` free req; `from` free req | World | `RECORD_ACCESSION` station.rs:542 |
| `POST .../record/receiving-1/write_provenance` | POST | claimed | `of` free req; `from` free req | World | `RECORD_WRITE_PROVENANCE` station.rs:558 |
| `POST .../record/receiving-1/hand_on` | POST | claimed | `what` free req; `to` free req | World | `RECORD_HAND_ON` station.rs:574 (Custody, audit §6.2) |
| `POST .../record/sorting-1/appraise` | POST | claimed | `what` free req; `verdict` free req | World | `RECORD_APPRAISE` station.rs:586 |
| `POST .../record/sorting-1/write_reason` | POST | claimed | `about` free req; `why` free req | World | `RECORD_WRITE_REASON` station.rs:602 |
| `POST .../record/sorting-1/let_go` | POST | **claimed** (working mode, entered deliberately — audit §3.3; destroys) | `what` free req; `because` free req | World | `RECORD_LET_GO` station.rs:618 |
| `POST .../record/catalogue-1/describe` | POST | claimed | `what` free req; `how` free req | World | `RECORD_DESCRIBE` station.rs:634 |
| `POST .../record/catalogue-1/arrange` | POST | claimed | `what` free req; `under` free req | World | `RECORD_ARRANGE` station.rs:651 |
| `POST .../record/catalogue-1/cross_reference` | POST | claimed | `from` free req; `to` free req | World | `RECORD_CROSS_REFERENCE` station.rs:667 (audit §6.2) |
| `POST .../record/catalogue-1/leave_note` | POST | claimed | `on` free req; `what` free req | World | `RECORD_LEAVE_NOTE` station.rs:683 (audit §6.2) |
| `POST .../record/catalogue-1/tidy_index` | POST | claimed | `what` free req | World | `RECORD_TIDY_INDEX` station.rs:699 (audit §6.2) |
| `POST .../record/mending-1/mend` | POST | claimed | `what` free req; `how` free req | World | `RECORD_MEND` station.rs:712 |
| `POST .../record/mending-1/mark_repair` | POST | claimed | `what` free req | World | `RECORD_MARK_REPAIR` station.rs:728 |

---

## C.5 `story` — `http://local/story/<id>`

Parts: story-desk (station, `binds: one gap`, modes) ×16; ledger-room fixtures gap-ledger +
filed-stories (reads). id: `story/<node>-<n>` (ordinal required for desks).

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../story/ledger-1` (read_ledger / read_around_gap) | GET | always | — | World | audit §2.2 **[no handler yet]** |
| `GET .../story/ledger-filed` (read_filed) | GET | always | — | World | filed-stories.yaml; audit §2.2 **[no handler yet]** |
| `POST .../story/ledger-1/take_next_silence` | POST | always (`Claimable`) | `what` free req | World / Sim `ledger` | audit §2.2 (→ `claim`, C.2) **[no handler yet]** |
| `POST .../story/<id>/draft` | POST | claimed (working) | `for` free req; `what` free req | World | `STORY_DRAFT` station.rs:245 |
| `POST .../story/<id>/file` | POST | claimed | `what` free req | World | `STORY_FILE` station.rs:262 |
| `GET .../story/<id>/read_aloud` (at reading table) | GET | always | — | World | audit §2.2 → see `gather` (C.24) **[no handler yet]** |

---

## C.6 `portrait` — `http://local/portrait/<id>`

Parts: easel (station, `binds: one character`, modes) ×16; likeness-table (`settle_likeness`,
paired); plate-rack (`file_plate`); hung-faces + house-palette (reads — house-palette's read is
re-homed to `standard`, audit §3.4, C.22). id: `portrait/<node>-<n>` for easels (ordinal req).

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../portrait/gallery-1` (read_hung) | GET | always | — | World | hung-faces.yaml; audit §2.2 **[no handler yet]** |
| `GET .../portrait/<id>/prompt_read` | GET | always | `of` enum(character id) req | World | `PORTRAIT_PROMPT_READ` station.rs:292 |
| `POST .../portrait/<id>/draw` | POST | claimed (working) | `of` free req; `carrying` free req | World | `PORTRAIT_DRAW` station.rs:276 |
| `POST .../portrait/<id>/prompt_edit` (redraw) | POST | claimed | `of` free req; `carrying` free req | World | `PORTRAIT_PROMPT_EDIT` station.rs:306 |
| `POST .../portrait/likeness-1/settle_likeness` | POST | **paired** (drawer + person-holder) | `of` free req; `with` free req | World | `PORTRAIT_SETTLE_LIKENESS` station.rs:368 |
| `POST .../portrait/plate-1/file_plate` | POST | always (holding the plate) | `what` free req | World | `PORTRAIT_FILE_PLATE` station.rs:380 |
| `POST .../portrait/plate-1/take_faceless` | POST | always (`Claimable`) | `what` free req | World / Sim | audit §2.2 (→ `claim`) **[no handler yet]** |

---

## C.7 `library` (craft) — `http://local/library/<id>`

**A real `STATION_ACTS` namespace not listed in Appendix A but shipped in code** — the craft
libraries (mood / response), mixed onto the story-desk and character-terminal (`CRAFT`,
station.rs:156). A write here changes how *every* character in the world feels or answers.
id: mounted on the station being worked (`library/<story-or-character-id>`).

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../library/<id>/read` | GET | always | `kind` enum(`mood`,`response`) req; `id` free req | World | `LIBRARY_READ` station.rs:333 |
| `POST .../library/<id>/write` | POST | claimed (working) | `kind` enum(`mood`,`response`) req; `id` free req; `field` enum(`description`,`template`) req; `text` free req | World | `LIBRARY_WRITE` station.rs:345 |

---

## C.8 `character` — `http://local/character/<id>`

Parts: character-terminal (station, `binds: one character`, modes) ×16; relations-table
(`settle_relation`, paired). id: `character/<node>-<n>` (ordinal req).

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../character/<id>` (read) | GET | always | — | World | `character_read` audit §2.2 (→ `query`) **[no handler yet]** |
| `POST .../character/<id>/write_identity` | POST | claimed (working) | `of` free req; `what` free req | World | `CHARACTER_WRITE_IDENTITY` station.rs:393 |
| `POST .../character/<id>/write_wants` | POST | claimed | `of` free req; `what` free req | World | `CHARACTER_WRITE_WANTS` station.rs:410 |
| `POST .../character/<id>/write_memories` | POST | claimed | `of` free req; `what` free req | World | `CHARACTER_WRITE_MEMORIES` station.rs:426 |
| `POST .../character/relations-1/settle_relation` | POST | **paired** (both character holders) | `between` free req; `and` free req | World | `CHARACTER_SETTLE_RELATION` station.rs:442 |

**Belief-write invariant (§9.2):** `write_beliefs`/`write_memories` are the character revising
itself, on the record — never a tool's silent edit (`npc_mind_design.md`). Kept in the `World`
store here (the founding sheet a Maker authors), distinct from the character's *own* lived
beliefs written on the sleep clock.

---

## C.9 `place` — `http://local/place/<id>`

Parts: survey-desk (station, `binds: one place`, modes) ×16; road-table (`settle_route`,
paired); place-index (reads/claims). id: `place/<node>-<n>` (ordinal req for desks).

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../place/index-1` (read_index) | GET | always | — | World | place-index.yaml; audit §2.2 **[no handler yet]** |
| `POST .../place/index-1/take_next_unwritten` | POST | always (`Claimable`) | `what` free req | World / Sim | audit §2.2 (→ `claim`) **[no handler yet]** |
| `GET .../place/<id>` (read_entry) | GET | always | — | World | `place_read_entry` audit §2.2 **[no handler yet]** |
| `POST .../place/<id>/write_entry` | POST | claimed (working) | `of` free req; `what` free req | World | `PLACE_WRITE_ENTRY` station.rs:456 |
| `POST .../place/<id>/write_local_history` | POST | claimed | `of` free req; `what` free req | World | `PLACE_WRITE_LOCAL_HISTORY` station.rs:472 |
| `POST .../place/road-1/settle_route` | POST | **paired** (both end-holders) | `from` free req; `to` free req | World | `PLACE_SETTLE_ROUTE` station.rs:487 |

---

## C.10 `map` — `http://local/map/<id>` — **DEFERRED to Step 8 (topology-mutating)**

Part: map-table (station, `binds: the whole geography`, modes) ×1; gallery-rail (read-only
overlook). These mutate frozen topology (§16 "Topology mutation… Step 8"; App A flags `map` as
"topology-mutating — needs §16/Q; Step 8"). Listed for completeness; **not landed in v1.**

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../map/map-room-1` (read) | GET | always | — | World | `map_read` audit §2.2 **[Step 8]** |
| `GET .../map/gallery-rail-1` (read) | GET | always | — | World | gallery-rail.yaml **[Step 8]** |
| `POST .../map/map-room-1/add_place` | POST | claimed (working) | `called` free req; `where` free req | World (topology) | `MAP_ADD_PLACE` station.rs:502 **[Step 8]** |
| `POST .../map/map-room-1/settle_border` (move_border) | POST | **paired** | `between` free req; `and` free req | World (topology) | `MAP_SETTLE_BORDER` station.rs:517 **[Step 8]** |
| `POST .../map/map-room-1/remove_place` (drown_place) | POST | claimed | `what` free req | World (topology) | `MAP_REMOVE_PLACE` station.rs:528 **[Step 8]** |
| `POST .../map/map-room-1/redraw_coast` | POST | claimed | `what` free req | World (topology) | `map_redraw_coast` audit §2.2 **[Step 8]** |

---

## C.11 `orders` — `http://local/orders/<id>` — **PROJECTED store (§9.2)**

Part: muster-board (fixture, tower-redoubt — the `set`/`hand_to` end). Per §9.2 / App A,
`orders_*` write the **agency layer** and re-project into the system prompt, not the world.

**order-table routes under `command`, not `orders`.** The plan below to keep both parts on
one `orders` prefix was superseded during Step 6: `report_done` at the command desk closes a
mission against `Sim.ledger` (`ORDERS_REPORT_DONE`, `work.rs`'s `mission` dispatch), which is
a different store and a different concern from muster-board's projected-agency `set`/`hand_to`
— so `namespace_of` mounts order-table's acts at `command/<id>` instead
(`npcd/src/effector/namespace.rs`). `command` is otherwise a plain generic station namespace
like any other (§9), not a second projected-store surface.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../command/command-room-1` (read) | GET | always | — | World (generic station read) | `station.rs::state` |
| `POST .../command/command-room-1/collect_mission` | POST | at the table, no open mission | — | Sim `missions` | `MISSION_ACTS`/`mission_acts.rs`, dispatched in `work.rs::mission` |
| `POST .../command/command-room-1/report_done` | POST | holding an order | `what` free req | Sim `ledger` | `ORDERS_REPORT_DONE` station.rs:830, `work.rs::perform` |
| `POST .../orders/muster-1/set` | POST | always | `what` free req; `for` free opt | projected (agency) | `ORDERS_SET` station.rs:804 (audit §3.2/§6.2) |
| `POST .../orders/muster-1/hand_to` | POST | always | `what` free req; `to` free req | projected (agency) | `ORDERS_HAND_TO` station.rs:818 (audit §6.2) |

---

## C.12 `enquiry` — `http://local/enquiry/<id>`

Part: enquiry-desk (station, `binds: an open enquiry`, command → enquiry). id `enquiry/enquiry-1`.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../enquiry/enquiry-1/read_history` | GET | always | — | World | `enquiry_read_history` audit §6.2 **[no handler yet]** |
| `POST .../enquiry/enquiry-1/take_question` | POST | always (claims the enquiry) | `what` free req | World | `ENQUIRY_TAKE_QUESTION` station.rs:744 |
| `POST .../enquiry/enquiry-1/answer_from_record` | POST | holding the enquiry | `what` free req; `answer` free req | World | `ENQUIRY_ANSWER_FROM_RECORD` station.rs:755 |
| `POST .../enquiry/enquiry-1/name_the_gap` | POST | holding the enquiry | `what` free req; `missing` free req | World | `ENQUIRY_NAME_THE_GAP` station.rs:771 |
| `POST .../enquiry/enquiry-1/raise_work` | POST | holding the enquiry | `from` free req; `work` free req | projected + Sim `ledger` (produces an order, audit §3.2) | `ENQUIRY_RAISE_WORK` station.rs:787 (audit §6.2) |

---

## C.13 `dispatch` — `http://local/dispatch/<id>`

Part: dispatch-board (fixture, command → dispatch). Reads the whole building; `post_wake` is the
change-and-its-wake producer. id `dispatch/dispatch-1`.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../dispatch/dispatch-1` (read) | GET | always | — | World | dispatch-board.yaml; audit §2.2 **[no handler yet]** |
| `GET .../dispatch/dispatch-1/read_holds` | GET | always | — | World | `dispatch_read_holds` audit §6.2 **[no handler yet]** |
| `GET .../dispatch/dispatch-1/read_wake` | GET | always | — | World | `dispatch_read_wake` audit §6.2 **[no handler yet]** |
| `POST .../dispatch/dispatch-1/post_wake` | POST | always | `from` free req; `breaks` free req | World | `DISPATCH_POST_WAKE` station.rs:843 (audit §6.2) |

---

## C.14 `plant` — `http://local/plant/<id>`

Part: plant-panel (fixture, command → plant). Read, not driven. id `plant/plant-1`.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../plant/plant-1/read_panel` | GET | always | — | World | plant-panel.yaml; audit §2.2 **[no handler yet]** |
| `POST .../plant/plant-1/note_drift` | POST | always | `what` free req; `drift` free req | World | `PLANT_NOTE_DRIFT` station.rs:887 |
| `POST .../plant/plant-1/raise_fault` | POST | always | `what` free req; `why` free req | World | `PLANT_RAISE_FAULT` station.rs:903 |

---

## C.15 `stores` — `http://local/stores/<id>`

Part: stores (fixture, command → receiving). id `stores/receiving-1`.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../stores/receiving-1/walk_the_racks` | GET | always | — | Sim | stores.yaml; audit §2.2 **[no handler yet]** |
| `POST .../stores/receiving-1/put_back` | POST | always | `what` free req | Sim | `STORES_PUT_BACK` station.rs:918 |
| `POST .../stores/receiving-1/take_out` | POST | always | `what` free req | Sim | `STORES_TAKE_OUT` station.rs:931 |

---

## C.16 `structure` — `http://local/structure/<id>`

Part: structure-board (fixture, story → board-room). id `structure/board-1`.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `POST .../structure/board-1/lay_out_scenes` | POST | always | `what` free req | World | `STRUCTURE_LAY_OUT_SCENES` station.rs:942 |
| `POST .../structure/board-1/test_the_want` | POST | always | `in` free req; `scene` free req | World | `STRUCTURE_TEST_THE_WANT` station.rs:953 |
| `POST .../structure/board-1/find_the_slack` | POST | always | `what` free req | World | `STRUCTURE_FIND_THE_SLACK` station.rs:965 |

---

## C.17 `roster` — `http://local/roster/<id>`

Part: roster (fixture, casting → roster-room). id `roster/roster-1`. **[no handlers — audit §2.2]**

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../roster/roster-1` (read) | GET | always | — | World | roster.yaml; audit §2.2 **[no handler yet]** |
| `POST .../roster/roster-1/take_unheld` | POST | always (`Claimable`) | `what` free req | World / Sim | audit §2.2 (→ `claim`) **[no handler yet]** |

---

## C.18 `cast` — `http://local/cast/<id>`

Part: watch-desk (station "reads and never writes, claims nobody", casting → watch). id `cast/watch-1`.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../cast/watch-1/read_all` | GET | always | — | World | watch-desk.yaml; audit §2.2 **[no handler yet]** |
| `POST .../cast/watch-1/report_disagreement` | POST | always | `about` free req; `what` free req | World | `CAST_REPORT_DISAGREEMENT` station.rs:859 |

---

## C.19 `room` — `http://local/room/<id>`

Part: seat (kind `seat`, every social/work node). `room.talk` **removed** (audit §3.1). id `room/<node>-<n>`.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `POST .../room/<id>/sit` | POST | always (a seat here) | — | World (standpoint) | seat.yaml; `room_sit` audit §2.2/§6.1 **[no handler yet]** |

---

## C.20 `creator` — `http://local/creator/<id>`

Part: creators-chair (fixture, command → command-room). id `creator/command-room-1`.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `POST .../creator/command-room-1/present` | POST | always | `what` free req | World | `CREATOR_PRESENT` station.rs:875 |

---

## C.21 `plan` — `http://local/plan/<id>` — **PROJECTED store (§9.2)** — new part (audit §6.2)

Part: **planning board** (new fixture, audit §6.2 — not yet placed in a map). `plan_*` write the
agency layer and stay visible in the dynamic system prompt (§9.2). id `plan/<node>-1` once placed.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../plan/<id>/read` | GET | always | — | projected (agency) | `plan_read` audit §6.2 **[no handler yet]** |
| `POST .../plan/<id>/break_down` | POST | always | `what` free req | projected (agency) | `plan_break_down` audit §6.2 **[no handler yet]** |
| `POST .../plan/<id>/order` | POST | always | `pieces` typed(array<string>) req | projected (agency) | `plan_order` audit §6.2 **[no handler yet]** |
| `POST .../plan/<id>/reorder` | POST | always | `pieces` typed(array<string>) req | projected (agency) | `plan_reorder` audit §6.2 **[no handler yet]** |
| `POST .../plan/<id>/scope` | POST | always | `what` free req; `in`/`out` free opt | projected (agency) | `plan_scope` audit §6.2 (renamed from `plan_set_scope`, §5A.3) **[no handler yet]** |

---

## C.22 `standard` — `http://local/standard/<id>` — new part (audit §6.2 / §3.4)

Parts: **standards board** (new, reachable from every level — audit §3.4) and house-palette's
read (`standard_read`, re-homed from `portrait.read_house_style`, audit §3.4).

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../standard/palette-1/read` | GET | always | — | World | house-palette.yaml; `standard_read` audit §3.4/§6.2 **[no handler yet]** |
| `POST .../standard/<id>/propose` | POST | always | `what` free req; `why` free req | World | `standard_propose` audit §6.2 **[no handler yet]** |
| `POST .../standard/<id>/settle` | POST | **paired** (a norm is settled, not decreed) | `what` free req | World | `standard_settle` audit §6.2 **[no handler yet]** |

---

## C.23 `trial` — `http://local/trial/<id>` — new part (audit §6.2)

Part: **trials shelf** (new fixture, audit §6.2). id `trial/<node>-1` once placed.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `POST .../trial/<id>/keep` | POST | always | `what` free req; `why_it_failed` free req | World | `trial_keep` audit §6.2 **[no handler yet]** |
| `GET .../trial/<id>/read_failures` | GET | always | — | World | `trial_read_failures` audit §6.2 **[no handler yet]** |
| `POST .../trial/<id>/compare` | POST | always | `a` free req; `b` free req | World | `trial_compare` audit §6.2 (renamed from `trial_set_side_by_side`, §5A.3) **[no handler yet]** |

---

## C.24 `gather` — `http://local/gather/<id>` — reading-table (audit §6.2)

Part: reading-table (fixture, story → long-table). **Name collision with the field `gather`
(C.2)** — resolved by the id: the field act is `POST .../gather` (no id), the gathering namespace
is `POST .../gather/long-table-1/…`. id `gather/long-table-1`.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `POST .../gather/long-table-1/call` | POST | always | `about` free req; `who` free req | World | `GATHER_CALL` station.rs:978 (audit §6.2) |
| `GET .../gather/long-table-1/read_standing` | GET | always | — | World | `gather_read_standing` audit §6.2 **[no handler yet]** |

---

## C.25 `bench_*` — the mode surface — mounted on every part declaring `modes:`

The 11 git-named verbs (audit §5.2) are mixed onto every part with a `modes:` block, so one
implementation serves every editing station (§5.1). In the vault that is the ten mode-bearing
parts: chronicle-terminal, character-terminal, story-desk, easel, survey-desk, map-table,
accession-desk, catalogue, appraisal-bench, mending-bench (§5.4). Mounted at
`http://local/bench/<id>` where `<id>` is the station instance being worked. The mode is
**per body** (§5.1), held as engine state. **These are shipped** (`work.rs:795–1026`) —
the `[no handler yet]` flags in the table below are stale (Appendix D.5); `bench_commit`'s
collision check already names the other party (`bench.rs:559`), so the Settling trigger
is real today.

| Route | Method | Condition (mode transition) | Body schema | Store | From |
|---|---|---|---|---|---|
| `POST .../bench/<id>/branch` | POST | reading → working (takes the claim) | — | World + engine mode state | `bench_branch` audit §5.2 — shipped (`work.rs:795–1026`) |
| `POST .../bench/<id>/stash` | POST | working → reading | — | World + mode | `bench_stash` audit §5.2 — shipped (`work.rs:795–1026`) |
| `POST .../bench/<id>/stash_pop` | POST | reading → working | — | World + mode | `bench_stash_pop` audit §5.2 — shipped (`work.rs:795–1026`) |
| `GET .../bench/<id>/diff` | GET | working, offered | — | World | `bench_diff` audit §5.2 — shipped (`work.rs:795–1026`) |
| `POST .../bench/<id>/restore` | POST | working → reading (discards) | — | World + mode | `bench_restore` audit §5.2 — shipped (`work.rs:795–1026`) |
| `POST .../bench/<id>/stage` | POST | working → offered | — | World + mode | `bench_stage` audit §5.2 — shipped (`work.rs:795–1026`) |
| `POST .../bench/<id>/unstage` | POST | offered → working | — | World + mode | `bench_unstage` audit §5.2 — shipped (`work.rs:795–1026`) |
| `POST .../bench/<id>/commit` | POST | offered → reading (**may fail with a conflict naming the other party**) | — | World + mode | `bench_commit` audit §5.2 (manufactures the Settling trigger) — shipped (`work.rs:935`) |
| `GET .../bench/<id>/status` | GET | any | — | World | `bench_status` audit §5.2 — shipped (`work.rs:795–1026`) |
| `GET .../bench/<id>/blame` | GET | any | — | World | `bench_blame` audit §5.2 (the custody chain) **[no handler yet]** |
| `GET .../bench/<id>/log` | GET | any | — | World | `bench_log` audit §5.2 — shipped (`work.rs:795–1026`) |

---

## C.26 `file_*` — reused from `zend-tools` (audit §5.3)

Five file verbs over the session overlay, mounted on the same mode-bearing editing stations as
`bench_*`. `http://local/file/<id>`. These are the only migrated routes with an **implemented**
handler already (`zend-tools`).

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../file/<id>/read` | GET | working (copy open) | `path` free req | World (session overlay) | `file_read` zend-tools, audit §5.3 |
| `GET .../file/<id>/list` | GET | working | `path` free opt | World (overlay) | `file_list` zend-tools, audit §5.3 |
| `POST .../file/<id>/edit` | POST | claimed (working) | `path` free req; `old_str` free req; `new_str` free req | World (overlay) | `file_edit` zend-tools (single-site `str_replace`), audit §5.3 |
| `POST .../file/<id>/write` | POST | claimed (working) | `path` free req; `content` free req | World (overlay) | `file_write` zend-tools, audit §5.3 |
| `POST .../file/<id>/delete` | POST | claimed (working) | `path` free req | World (overlay, whiteout) | `file_delete` zend-tools, audit §5.3 |

---

## C.27 The personal namespace — `/phone`, `/history`, `/self` (§7.2)

Reachable wherever the body stands (§7); npcd owns them even when an embedder replaces the vault
(§9.1). Always in the near-you index (§6). The phone acts migrate off `WORLD_ACTS` /
`send_image` (§4, App A).

### `/phone` — the messaging surface

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../phone` | GET | always (has a handset) | — | Sim phone / projected | new (thread listing), §7.2 |
| `POST .../phone/message` | POST | on a thread (`Threads`) | `to` enum(`Threads`) req; `intent` free req | Sim phone / projected | `MESSAGE` acts.rs:1140; enact.rs:93 |
| `POST .../phone/reach_out` | POST | `Contacts` non-empty | `to` enum(`Contacts`) req; `intent` free req | Sim phone / projected | `REACH_OUT` acts.rs:1061; enact.rs:97 |
| `POST .../phone/invite` | POST | `Invitable` non-empty | `to` enum(`Invitable`) req; `who` enum(`Invitees`) req; `intent` free opt | Sim phone / projected | `INVITE` acts.rs:1230; enact.rs:94 |
| `POST .../phone/open_group` | POST | `Contacts` non-empty | `called` free req; `with` enum(`Contacts`) req; `intent` free opt | Sim phone / projected | `OPEN_GROUP` acts.rs:1270; enact.rs:95 |
| `POST .../phone/send_image` | POST | *(deferred — see below)* | `to` enum(`Threads`) req | Sim phone / projected | `send_image` (body, `tools.rs`); audit §2.1 |
| `POST .../phone/<thread>/sign_off` | POST | `Leavable` | `intent` free req | Sim phone / projected | `SIGN_OFF` acts.rs:1027; enact.rs:96 |

*As-built: `/phone/send_image` is **deferred** and is **not** mounted or advertised. `send_image` is
not a body act — it is absent from `body::is_of_the_body` and `enact::is_mine`, so `enact::perform`
returns `NotOfTheBody`, which `enact_response` maps to `500`. The engine performs `send_image` above
the body layer (the image-guest / interaction path), which the `/phone` route cannot reach; wiring
that path is a later cut. Until then `send_image` is left out of `phone.rs` `VERBS` and the `OPTIONS`
schema rather than mounted as a route that only ever errors. The other four verbs
(`message`/`invite`/`open_group`/`reach_out`) and `sign_off` are routed as the table shows.*

### `/history` — read-only onto what this body has done and seen

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../history` | GET | always (personal, §7) | `cursor` free opt (cursor-paginated) | projected (memory, read) | §7.2; new |

### `/self` — the character's own maintained state as a read view

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../self/plan` | GET | always | — | projected (agency, read) | §7.2; writes at C.21 |
| `GET .../self/orders` | GET | always | — | projected (agency, read) | §7.2; writes at C.11 |
| `GET .../self/beliefs` | GET | always | — | projected (belief, read) | §7.2, §9.2 |
| `GET .../self/memory` | GET | always | — | projected (memory, read) | §7.2 |

`/self` is **read-only** — the writes happen at the situated stations that own them (C.11, C.21,
C.8) and project forward (§7.2).

*As-built: the four reads are **built** (`effector/selfsurface.rs`, mounted at `/self`), served
from the shared cast (`Runtime::npcs`). `plan` and `orders` both render the one `agency` vector on
the record — there is no separate ledger-orders layer on `NpcPayload`. `memory` returns an empty
projection here: `NpcPayload` carries no memory layer (a character's memory lives in the
substrate's own records and `layers/memory/`, not on the record); the body's witnessed recent past
is `/history`. Each read degrades to an empty layer if the cast is not installed, never a failure.*

---

## C.28 Decisions & ambiguities

1. **`read` and `scan` → `GET` (`query`), not their own POST routes.** Both are reads; per
   Tier-1 `tool_world_acts.md` reads dissolve into `query`, so `read` is the `GET` on the target
   resource and `scan` is `GET .../scan/<place>`.
2. **`sign_off` — store and namespace.** Appendix A groups it with *world-state* acts
   (`acts.rs:1311`), but it acts on a conversation thread. **Call:** routed under
   `/phone/<thread>/sign_off` (C.27), store = Sim phone / projected; cross-referenced in C.2.
3. **Two `gather`s.** Field extraction (`Extractable`, acts.rs:499) vs the reading-table
   `gather_*` (station.rs:978). **Call:** disambiguated by URL shape — `POST .../gather` (no id)
   vs `POST .../gather/long-table-1/…`.
4. **Not-part-bound world acts.** `claim`/`release`/`post_notice`/`record_verdict` mount on the
   target thing's id; `give`/`equip`/`use`/`recall`/`engage` are body-inventory/movement routes
   with no id. `engage`/`command_tower`/`produce` are **war-world**, never mounting in the vault.
5. **`library` (craft) namespace.** `LIBRARY_READ`/`WRITE` (station.rs:333/345) are shipped
   `STATION_ACTS` absent from Appendix A and audit §2.2. **Call:** given subsection C.7.
6. **Code verb names vs Appendix A.** Tables use the **code** names for shipped verbs and the
   **audit** names for authored-but-unimplemented verbs, citing both (e.g. code
   `map_settle_border` vs App A `move_border`).
7. **`claimed` vs `mode:working`.** For the ten mode-bearing parts, "claimed (working)" — holding
   the `binds` *is* being in the working mode (audit §5.1). `bench_*` (C.25) are the transitions.
8. **`plan_*` / `orders_*` → projected store.** Per §9.2; `orders_take`/`give_back` and
   `enquiry_raise_work` also touch the Sim `ledger` (the claim/order is world-visible).
9. **`map_*` deferred (C.10).** Step 8 per §16; not counted in the v1 total.
10. **Typed (shaped-not-enforced) fields.** Only `give.count`, `produce.count` (int),
    `command_tower.x/y/depth` (number), `plan.order`/`reorder.pieces` (array). Handler validates
    (§11).

### Route count (v1, excluding the Step-8 `map` cut)

| Namespace | routes | | Namespace | routes |
|---|---|---|---|---|
| lift (C.1) | 3 | | plant (C.14) | 3 |
| world-state WORLD_ACTS (C.2) | 16 | | stores (C.15) | 3 |
| chronicle (C.3) | 7 | | structure (C.16) | 3 |
| record (C.4) | 14 | | roster (C.17) | 2 |
| story (C.5) | 6 | | cast (C.18) | 2 |
| portrait (C.6) | 7 | | room (C.19) | 1 |
| library/craft (C.7) | 2 | | creator (C.20) | 1 |
| character (C.8) | 5 | | plan (C.21, projected) | 5 |
| place (C.9) | 6 | | standard (C.22) | 3 |
| orders (C.11, projected) | 6 | | trial (C.23) | 3 |
| enquiry (C.12) | 5 | | gather (C.24) | 2 |
| dispatch (C.13) | 4 | | bench_* (C.25) | 11 |
| file_* (C.26) | 5 | | personal /phone·/history·/self (C.27) | 12 |

**v1 total ≈ 152 routes** across 27 namespaces (map's 6 Step-8 routes excluded).
**Correction (Appendix D):** the "Store" column above and any "0-of-79 / [no handler
yet]" framing are stale. The writing handlers are *shipped* — in `engine/work.rs`
(`work::perform`), not `station.rs` (declaration only) — and the store for the
authoring namespaces is `Sim` (its `record`/`ledger`), not the npc-map `World`; a
subset reaches the mind folder on disk through the `bench` working copy. Appendix D.5
is the authoritative per-namespace store map, and D.6 the genuinely-missing work. So
the migration mostly *wraps existing `work.rs` handlers in route envelopes*; the
handlers truly absent are the short list (`plan_*`, `roster_*`, `room_sit`,
`character_write_beliefs`).

---

# Appendix D — Handlers and the mind folder

*Companion to §9/§9.2/§9.3 and Appendix C. The main text once divided a route handler's
write target into two stores — **world state** and **projected state**. That was a
simplification: the writing handlers (`chronicle_`, `character_`, `story_`, `place_`,
`portrait_`, `record_`, `orders_`, and the `bench_`/`file_` surface) touch a **third**
store the two-store split hid — the **mind folder**, the authored lore on disk, reached
not directly but through a per-body working copy that commits to the tree. This appendix
names every store a handler can touch, the mechanism by which a write reaches the mind
folder, and — grounded in the shipped code — what each Appendix-C writing namespace
actually does today versus what it must do. It corrects two now-stale claims: the audit's
"0-of-79 handlers" and Appendix C's uniform `Store = World` (both wrong — the handlers are
shipped, in `engine/work.rs`), and `tool_world_state.md §2`'s "the mind folder is not a git
repository" (it is one now). Grounded in `--mind D:/prog/mind` and `npcd/src/{mind,sim,engine}`.*

## D.1 The mind folder on disk

`D:/prog/mind` is the authored half of the product — text a person wrote, none generated
(`npcd/src/mind/mod.rs:1`). Its inspected layout:

| On disk | Holds | Console `Section` (address) | Bench area |
|---|---|---|---|
| `layers/world/` | **1,268** canon `.md` — nested topics+entries (`history/`, `geography/`, `locations/`, `factions/`, `events/`, `creatures/`, `identity/`, `combat/`, …) | `canon` → `layers/world` (`address.rs:174`) | `layers` |
| `layers/eras/` | 13 era `.md` — the shared storyline, one per era | `eras` → `layers/eras` (`address.rs:175`) | `layers` |
| `layers/stories/` | backstory `.md` — accounts filling the gaps between eras | `stories` → `layers/stories` (`address.rs:176`) | `layers` |
| `layers/memory/<char>/` | per-character remembered episodes `.md` | `memory` → `layers/memory` (`address.rs:177`) | `layers` |
| `layers/life/<char>/` | dated life episodes — lifegen input, executed by the substrate authoring plane (`engine/authoring.rs`) | — | `layers` |
| `personalities/*.yaml` | **character sheets** — 24 casts; who a character is before living anything | `characters` → `personalities` (`address.rs:181`) | `personalities` |
| `personalities/portraits/*.png` | portrait **plates** (30) shipped beside the cast | — (binary) | — |
| `worlds/*.yaml` | settings + their `selects`/`excludes`/`personalities` filters | `worlds` → `worlds` (`address.rs:182`) | `worlds` |
| `responses/*.yaml`, `moods/*.yaml` | craft libraries — reply shapes and registers | `responses`, `moods` (`address.rs:179-180`) | `responses`, `moods` |
| `map/battle-cities/*.yaml` | the **npc-map source** — level maps + `parts/` catalogue; where the C.0 parts are placed | — | `map` |
| `images/*.png` | generated-image store — the guest pipeline's output | — | — |
| `projection.yaml`, `mind.yaml`, `CLAUDE.md` | settings — schema, mind config, authoring guidance | `settings/{projection,mind,guidance}` (`address.rs:219`) | — (bench allow-list excludes) |
| `.substrate/`, `accounts/`, `schema/`, `.git/` | redo-log persistence; operator accounts; JS validators; git metadata | — | — |

Two facts the rest turns on:

- **The mind folder is now a git repository** (`layers/` tracked; `.substrate/`
  git-ignored). This resolves the *precondition* of `tool_world_state.md` dispute A. **But
  the daemon never invokes git** — no `Command::new("git")`, `git2`, or `gix` in
  `npcd/src`. The repo is a human's safety net, not a runtime store.
- **The canon under `layers/world/` (1,268 files) is not writable by any station act.** The
  station/bench surface reaches only `layers/eras`, `layers/stories`, `personalities`,
  `moods`, `responses` (and `map`). Canon is editable only through the console admin editor
  (D.3).

## D.2 The store taxonomy — every store a handler can touch

There are **four** stores, not two. The design's "world state vs projected state" names the
first and last and folds the middle two away.

**1. The in-RAM npc-map `World`** — *who is where, holding what, and what happened.* Owns
placement/standpoint, `Actor.hold` claims, the lift, witnessed events. No persistence;
deterministic re-entry (`runtime.rs:776`); reached under the one lock (`Hosted::with`/`read`,
`world/mod.rs:41`). Claim = proximity (`perceive.rs:280`) + `Actor.hold`. Handlers: the
world-state half of `WORLD_ACTS` (`enact.rs:71–98`; C.1–C.2).

**2. The in-RAM `Sim`** (`sim/mod.rs:61`) — *everything about the world that is not its map.*
Behind the same lock (`with_sim`/`with_both`); **the store Appendix C mislabels "World" for
every authoring namespace.** Sub-stores:
- `record: Record` (`sim/record.rs:236`) — the authored-content state machine. One `Item`
  per named thing (era, entry, gap, story, character, portrait, place, enquiry, structure,
  accession) with `state` (Unwritten→Held→Draft→Offered→Filed→Retired), `holder`,
  `condition`, and an optional `path` (`record.rs:190`) naming the mind-folder file when the
  thing is a document. `Item.body` is an in-RAM draft buffer for non-documents.
- `bench: Benches` (`sim/bench.rs:108`) — the per-body working copy over the mind folder
  (D.4); the only sub-store that reads/writes disk.
- `ledger: Ledger` — orders, verdicts, claims-as-orders. Plus missions, phone threads,
  postings, field, devices, tower, packs.
- **Persistence:** in-RAM runtime state; not the substrate. Anything left in `Record.body`/
  `Ledger` and never pushed through `bench` **does not survive a restart** (`record.rs:363`).
- **Claim:** `Record` custody (`held_by_other`, `record.rs:229`) + `Benches` isolation and
  commit-time collision check.

**3. The mind folder (authored lore on disk)** — *canon, cast, craft, storyline.* Owns the
D.1 tree. Persistence: the filesystem, atomic (temp + rename, `doc.rs:177`, `bench.rs:748`).
**Two doors, never conflated:** the console editor `npcd::mind` (D.3, addressed
`canon/ammo/bolt`, gated read `User` / write `Admin`) and the effector working copy
`Sim.bench` (D.4, addressed by `MindPath`, gated by claim + commit, invisible until commit).
Distinct from World (RAM placement) and from the substrate (per-character projected layers).

**4. The substrate** — *projected state: agency, belief, memory.* Owns `AuthoredBelief`,
`AuthoredRelationship`, `AuthoredStrategy`, modulations, the per-character layers.
Persistence: mandatory redo-log (`.substrate/substrate.log`). Claim: the belief-write
invariant — a belief is the character's own, never a tool's silent edit (`npc_mind_design.md`;
§9.2); the authoring plane that may write it is `engine/authoring.rs`, in a *separate catalog*
from the action plane (`authoring.rs:394`). **This is §9.2's "projected store." No station
handler writes it today** — `orders_`/`plan_` are supposed to (C.11/C.21) but currently write
`Sim.ledger` or are unimplemented (D.5/D.6).

> **The correction in one line:** §9's "world-state vs projected-state" should read **World
> (RAM) · Sim (RAM, incl. the record/bench state machine) · the mind folder (disk) · the
> substrate (projected, persisted)**. Appendix C's `Store = World` for the authoring
> namespaces means, in almost every case, **`Sim.record` (+ the mind folder via `Sim.bench`
> for the document kinds)** — not the npc-map World.

## D.3 The mind authoring layer (`npcd::mind`) — the console door

For an implementor (`mind/mod.rs` is authoritative):
- **Addressing.** A wire address is `section/name/name…` (`canon/ammo/bolt`), never a path.
  `Address::parse` (`address.rs:302`) is the only place that knows the on-disk mapping; the
  section supplies directory *and* extension (`.md` for canon/eras/stories/memory, `.yaml`
  otherwise). Nine sections; a client cannot invent a tenth (`address.rs:98`). Names are
  validated as `MindPath` (`..`, drive letters, NUL, symlinks refused before touching disk).
- **Granularity.** `doc` (whole document), `section` (a YAML doc as form `Field`s, comments
  preserved), `parts` (one item inside a settings document). YAML edits go through
  `registry::yaml_edit::splice` so author comments survive.
- **Mechanism.** `doc::read` (`doc.rs:88`), `doc::write` with `must_be_new` (`doc.rs:124`),
  `doc::remove` (`doc.rs:162`); write = temp + `fsync` + rename (`doc.rs:177`); ≤512 KB UTF-8.
- **Permission.** Read signed-in, write admin — `GET /v1/mind/entry` `Role::User`,
  `PUT/DELETE` `Role::Admin` (`api.rs:331–336`). Scope filtering (`scope.rs`) applies a
  world's `selects`/`excludes`/`personalities`.

An NPC never uses this door — it cannot present an admin role, and would not write disk
directly (D.4).

## D.4 How an effector handler writes the mind folder

The shipped mechanism is `engine/work.rs` (the dispatcher for `STATION_ACTS` + `BENCH_ACTS`
+ `MISSION_ACTS`, `work.rs:37`, entry `work::perform` `work.rs:173`) — **not** `enact.rs`
(world acts) and **not** `station.rs` (which only declares the `Tool` statics).

**The two-store bench act.** A body at a station holds both a `Record` item (the thing it has
taken on) and a `Benches` working set (documents it has changed, uncommitted). `Benches`
(`sim/bench.rs`) is a git-*like* working copy in memory:
- `open` maps body→`Working` (`Change{base, now}` per path); a read falls through to disk, a
  write touches disk **only at commit**.
- **collision is a comparison, not a lock:** commit re-reads each file and requires it to
  still equal `base`; the second committer is refused and told whose work it hit
  (`bench.rs:559`). This is what "manufactures the Settling trigger" (C.25) — for real.
- commit writes every changed doc atomically (`bench.rs:748` `put`) and records
  `last[path]=body` for `bench_blame`.
- allow-list of editable areas (`bench.rs:666`): `layers`, `map`, `moods`, `personalities`,
  `responses`, `worlds` — deliberately not `projection.yaml`/`mind.yaml`/`schema/`.

**The write path.** For the prose verbs (`chronicle_add_entry`/`rewrite_page`, `story_draft`,
`character_write_*`, `place_write_*`) dispatch funnels into `write_into` (`work.rs:700`):
1. Ask `Record::settle_path(name)` (`record.rs:338`) — returns a path **only for `Era`/
   `Story`/`Gap` kinds** (`layers/eras/…`, `layers/stories/…`), minting on first write; all
   other kinds get `None`.
2. **No path →** `Record::write` appends to the in-RAM `Item.body` (`record.rs:512`). *Never
   reaches disk.*
3. **Path →** `Record::claim_for_write` (custody) then `Benches::append(body, name, path,
   text)` — the text goes to the working set, invisible until commit (`work.rs:718–726`).

The join: `Record::index_canon(root)` (`record.rs:379`) reads `layers/eras` and
`layers/stories` at host time, adopting each doc with its `path` and `State::Filed`, so an
era *is* the file it is. Portrait and craft writes take a **field** path instead:
`portrait_draw`/`prompt_edit` → `bench.write_field(personalities/<who>.yaml,
["portrait","prompt"], …)` (`work.rs:239`); `library_write` → `bench.write_field(moods|
responses/<id>.yaml, [field], …)` (`work.rs:634`); `file_write`/`edit`/`delete` →
`bench.write`/`edit`/`remove` over any editable-area doc (`work.rs:1051–1094`).

**The commit is the only disk write, and it reconciles "admin writes disk."** `bench_commit`
(`work.rs:935`) requires a one-line `why`, runs `Benches::commit` (collision check + atomic
writes), then settles the record item. The disk write is plain `std::fs` under the world
lock, on behalf of the body's token — not through the admin-gated `/v1/mind` API. So the two
rules do not conflict — they are **two doors** to store #3: the **console door** (a person's
edit, `Role::Admin`) and the **effector door** (the character's edit, gated in-fiction by
claim + working copy + commit-collision, never by an operator role — the token resolves to a
body, the body's standing is the authorization, commit lands it).

**Recipe for a new writing handler.**
1. If it is a **document** (prose in a file), give its `Record::Kind` a `settle_path` arm
   (today only Era/Story/Gap, `record.rs:343`) and `index_canon` adoption; then route through
   `write_into` for working-copy isolation + commit-collision + disk for free.
2. If it is a **judgement** (provenance, condition, verdict, order), write `Record` fields or
   `Sim.ledger` — but it is RAM-only until a durability decision (D.6).
3. If it is **projected** (a plan, a belief, an order the character holds), it must reach the
   substrate authoring plane (store #4) — no station handler does this yet (D.6).
4. Gate with `Record` custody (`take`/`claim_for_write`/`held_by_other`), not an operator
   role; return second-person `Outcome::Refused`, which the route envelope (§12) wraps.

## D.5 Per-namespace store table (reconciling Appendix C)

Store the App-C column claims vs the store the shipped handler touches (`work.rs`), whether it
reaches the mind folder, and the mechanism. Schemas/verbs are in Appendix C.

| Namespace (App C) | App C "Store" | **Actual store today** (`work.rs`) | Mind folder? | Mechanism |
|---|---|---|---|---|
| `chronicle_add_entry`/`rewrite_page` (C.3) | World | `Sim.record` + **mind folder** via `Sim.bench` | **Yes** (`layers/eras/*.md`) | `write_into`→`settle_path`→`bench.append` (`work.rs:198,700`) |
| `chronicle_retire_entry` (C.3) | World | `Sim.record` (→Retired) | No | `set_state` (`work.rs:261`) |
| `chronicle_settle_boundary` (C.3) | World | `Sim.record` xref + `Sim.ledger` | No | `settle` (`work.rs:264`) |
| `story_draft` (C.5) | World | `Sim.record` + **mind folder** via `Sim.bench` | **Yes** (`layers/stories/*.md`) | `write_into`→`bench.append` |
| `story_file` (C.5) | World | `Sim.record` (→Filed) | No | `set_state` (`work.rs:260`) |
| `character_write_identity`/`wants`/`memories` (C.8) | World | **mind folder** via `Sim.bench` *(as-built: D.6 #1 done)* | **Yes** (`personalities/<who>.yaml` `anchor`/`wants`; `layers/memory/<who>/memories.md`) | `bench.write_field`/`bench.append` (the `portrait_draw` pattern) — RAM `Record.write` only when there is no mind folder |
| `character_settle_relation` (C.8) | World | `Sim.record` xref + `Sim.ledger` | No | `settle` (`work.rs:264`) |
| `place_write_entry`/`write_local_history` (C.9) | World | `Sim.record` + **mind folder** via `Sim.bench` *(as-built: D.6 #1 done)* | **Yes** (`layers/world/locations/<slug>.md`; `layers/world/geography/<slug>.md`) | entry: `write_into`→`settle_path`(Place→locations)→`bench.append`; local history: `history_path`(geography)→`bench.append` |
| `place_settle_route` (C.9) | World | `Sim.record` xref + `Sim.ledger` | No | `settle` |
| `portrait_draw`/`prompt_edit` (C.6) | World | **mind folder** via `Sim.bench` | **Yes** (`personalities/<who>.yaml` `portrait.prompt`) | `bench.write_field` (`work.rs:239`) |
| `portrait_file_plate` (C.6) | World | `Sim.record` (→Filed) | No (no image store) | `set_state` |
| `portrait_settle_likeness` (C.6) | World | `Sim.record` xref + `Sim.ledger` | No | `settle` |
| `library_read`/`write` (C.7) | World | **mind folder** via `Sim.bench` | **Yes** (`moods`/`responses/*.yaml`) | `bench.read_field`/`write_field` (`work.rs:612`) |
| `record_*` (accession/provenance/hand_on/describe/arrange/cross_reference/leave_note/tidy_index/mend/mark_repair/let_go) (C.4) | World | `Sim.record` fields (RAM) | No | field setters (`work.rs:310–440`) |
| `record_appraise`/`write_reason` (C.4) | World | `Sim.ledger` verdicts (RAM) | No | `ledger.record_verdict` (`work.rs:357,366`) |
| `map_add_place`/`remove_place` (C.10) | World (topology) | `Sim.record` (RAM); **no topology mutation** | No | `Record.put`/`let_go` (`work.rs:271,299`) |
| `map_settle_border` (C.10) | World (topology) | `Sim.record` xref + `Sim.ledger` | No | `settle` |
| `orders_set`/`hand_to`/`report_done` (C.11) | **projected (agency)** | **`Sim.ledger` (RAM)** — *not the substrate* | No | `ledger.set_order`/`hand_to`/`finish` (`work.rs:468–490`) |
| `plan_*` (C.21) | **projected (agency)** | **unimplemented** — no `work.rs` arm | No | — (must reach substrate `AuthoredStrategy`) |
| `enquiry_*` (C.12) | World | `Sim.record` + `Sim.ledger` | No | `work.rs:443–465` |
| `dispatch_post_wake`/`cast_report_disagreement`/`plant_*`/`gather_call` (C.13/18/14/24) | World | `Sim.ledger` (RAM) | No | `ledger.set_order` (`work.rs:491–599`) |
| `structure_*` (C.16) | World | `Sim.ledger` verdicts (RAM) | No | `ledger.record_verdict` |
| `stores_put_back`/`take_out` (C.15) | Sim | `Sim.record` give_back/take | No | `work.rs:528,532` |
| `creator_present` (C.20) | World | `Sim.record` (read) | No | `work.rs:499` |
| `bench_*` (C.25) | World + mode | `Sim.bench` + `Sim.record` (commit → **mind folder**) | **commit: Yes** | `work.rs:795–1026` (**shipped**, contra App C flags) |
| `file_*` (C.26) | World overlay | `Sim.bench` (commit → **mind folder**) | **commit: Yes** | `work.rs:1037–1094` |

**Verbs Appendix C lists that do not exist as station acts** (`station.rs:996`):
`character_write_beliefs` (deliberately absent — beliefs are earned on the sleep clock, not
authored, `station.rs:406`), `plan_*` (none), `roster_*` (none), `room_sit` (none); and
`portrait_redraw` is really `portrait_prompt_edit` (`station.rs:306`). Building the C.8-belief,
C.19, C.21 routes is writing *new* handlers, not migrating.

**A subtlety:** `settle_path` mints a path per record *kind*, and it now mints one for `Place`
too (`layers/world/locations`), so `place_write_entry` reaches the mind folder through the same
`write_into` path as an era. `character_write_*` do *not* go through `write_into` — an identity
or a want is a field on the personality sheet (`bench.write_field`, the `portrait_draw`
pattern), and a memory is an append to `layers/memory/<who>/`. A place's local history is a
second document in `layers/world/geography/` (`Record::history_path`), distinct from the
place's own entry, because a place has one entry and any number of histories and `Item.path`
can name only one. *(As-built after D.6 #1; the earlier "always fall to the RAM branch" was the
pre-implementation state.)*

**Net correction.** Of the authoring namespaces, only **chronicle-era, story, portrait-prompt,
craft-library, and the raw `bench_`/`file_` surface** reach the mind folder — through
`Sim.bench`, not "World." Everything else marked `World` is `Sim.record`/`Sim.ledger` in RAM.

## D.6 What's missing to implement (ordered)

1. **Durable persistence of `Sim.record`/`Sim.ledger`, or a disk write for the non-era/story
   authoring verbs.** `character_write_*`, `place_write_*`, and every `record_`/`orders_`/
   `structure_` verdict live only in RAM (`work.rs:710`, `record.rs:512`); `Sim` is not in the
   substrate redo log, so they do not survive a restart — the failure `record.rs:363` fixed for
   eras/stories but which still holds elsewhere. Fix: give Character/Place kinds a `settle_path`
   (`personalities/…`, `layers/world/locations|geography/…`) + `index_canon` adoption so they
   route through `bench`; or persist `Sim`.

   *As-built: **done for character and place.** `character_write_identity`/`wants` splice the
   `anchor`/`wants` fields of `personalities/<who>.yaml` (the `portrait_draw` pattern);
   `character_write_memories` appends `layers/memory/<who>/memories.md`; `place_write_entry`
   settles into `layers/world/locations/<slug>.md` through `write_into`; `place_write_local_history`
   appends `layers/world/geography/<slug>.md`. `settle_path` gained a `Place → locations` arm,
   `Record::history_path` names the geography document, and `index_canon` now adopts both world
   layers as `Filed` `Place`s, so a survey survives a restart; a commit of a freshly-branched
   document also lands it `Filed` (the new `Held → Filed` transition). The `record_*`/`orders_`/
   `structure_` verdicts that live in `Sim.ledger` remain RAM-only.*
2. **The projected store is never written by a station handler.** §9.2/C.11/C.21 require
   `orders_`/`plan_` to write the substrate **agency** layer (`AuthoredStrategy`) and re-project
   into the system prompt. Today `orders_*` write `Sim.ledger` and `plan_*` has no handler. **The
   write API is `Npcs::put_strategy` (`npcs.rs:745`), not `engine/authoring.rs`** (which is only a
   life-document parser with no substrate side effect). A bridge from the turn loop to the shared
   `Npcs` handle must be built (Appendix E). Once it writes, the change projects automatically via
   the `agency` collection (`projection.yaml:437`) and `persona::intent` (`persona.rs:98`).

   *As-built: **done.** The shared cast is installed on `Runtime` (`set_npcs`/`npcs`), and
   `effector/plan.rs` (mounted at `/plan`, `/orders`, and kept out of the generic station nest)
   writes the agency layer through the new owner-blind `Npcs::put_strategy_self`. `/self`
   (`effector/selfsurface.rs`) reads the same layers back. The change projects with no extra
   wiring, exactly as this item predicted.*
3. **Git is a repository but not a runtime store.** The mind folder is a git repo, yet the
   daemon never runs git. `bench_commit` writes with `std::fs::rename`; `bench_blame`/`log`
   answer from in-RAM state that resets on restart. **v1 keeps this in-RAM (per-run) custody;
   real git is a later cut** (Appendix E): custody and blame/log are correct within a run — the
   collision check that manufactures the Settling trigger is real (§C.25) — and only their
   *persistence across a restart* waits on git plumbing (`git add`/`commit` as the acting body at
   `bench_commit`, `git blame`/`log` back). It is deferred because it is durability-only, adds a
   git runtime dependency, and the world it records is itself not persisted (bodies re-enter, so a
   run is the natural custody horizon).

Also missing, lower-stakes: the **effector router/token surface itself** (all of Appendix C's
routing is future work — the handlers are still reached through the old `Tool`/`enact`/`work`
path, §14 "New"); an **image store** for portrait plates; **canon (`layers/world/`)
writability from a station** (only the console reaches it); and **document metadata as front
matter** (provenance/condition live in `Sim.record`, not on the `.md` files).

---

# Appendix E — Resolved implementation decisions

*Every decision needed to implement the migration without further input, settled either with
the author (lore/product) or by the pre-implementation code investigation (engineering). The
build follows Part F's order; this is the decision record it is built against.*

## Lore / product (settled with the author)

- **Character authoring.** `character_write_identity` → the `anchor` field of
  `personalities/<who>.yaml` via `bench.write_field(["anchor"], …)` (the `portrait_draw`
  pattern, `work.rs:239`); `character_write_wants` → a **new top-level `wants` field** on the
  sheet; `character_write_memories` → **append to `layers/memory/<who>/*.md`**.
  `character_write_beliefs` is **not exposed** — beliefs form on the sleep clock and
  belief-writes stay operator-only (the §9.2 invariant). So C.8 adds no belief route.
- **Place authoring.** `place_write_entry` → `layers/world/locations/<slug>.md`;
  `place_write_local_history` → `layers/world/geography/*.md`. Both are already in the bench
  allow-list (`bench.rs:666`).
- **New-part placement** (map YAML edits — Step 2). **standards board** in every level's `core`
  node (reachable everywhere, audit §3.4); **planning board** in command → command-room (beside
  the order table); **trials shelf** in chronicle → sorting-room (beside the appraisal bench).
- **Git custody is deferred; v1 is in-RAM (per-run).** `bench_commit` writes the files with
  `std::fs`, and `bench_blame`/`bench_log` answer from in-RAM state — correct within a run (the
  commit-collision Settling trigger is real), reset on a restart. Real git (`git add`/`commit` as
  the acting body, `git blame`/`log` back) is a later cut: it is durability-only, the world it
  records is not itself persisted (a run is the natural custody horizon), and it adds a git runtime
  dependency best introduced deliberately. (When it lands: shelling `git` is preferred over the
  `git2`/libgit2 native dependency, since the mind folder is already a repo.)

## Engineering (settled by investigation)

- **Agency writer = `Npcs::put_strategy`** (`npcs.rs:745`), not `engine/authoring.rs`.
  `plan_break_down`/`order`/`reorder`/`scope` and `orders_*` write `AuthoredStrategy` (a tree via
  `parent_id`) into the `agency` layer, which projects automatically (`projection.yaml:437`,
  `persona::intent` `persona.rs:98`) — no extra projection wiring.
- **The bridge.** Wrap the cast as `Arc<tokio::RwLock<Npcs>>` and install it on `Runtime` (the
  installed-after-construction pattern, `runtime.rs:291-319`), reusing the *existing* `Npcs`/
  substrate handle — never a second (the one-writable-handle guard, `npcs.rs:16-23`). Projected
  writes (`plan_*`/`orders_*`) run at the **async turn-loop layer**, split out of the sync
  `&Hosted`-only `work::perform`; `npc_id` parses from `body` (`"npc-{id}"`, `runtime.rs:763`); a
  new **owner-blind self-write** method resolves `owner` from `Npcs::payload(npc_id).owner`.

  *As-built: **built.** `Authored.npcs` is `Arc<tokio::sync::RwLock<Npcs>>` (only its
  construction changed; every call site reaches it through the `Arc`'s `Deref`), installed on
  `Runtime` via `set_npcs`/`npcs` beside `set_substrate`. The self-write is
  `Npcs::put_strategy_self(npc_id, strategy_id, body, now_ms)` — it resolves `owner` from
  `payload(npc_id).owner_id` and calls the existing `put_strategy`. The projected verbs live in
  `effector/plan.rs`, an async router mounted at `/plan` and `/orders` (not the sync
  `body::perform`); the caller's `npc_id` comes straight from the device token's `DeviceCaller`,
  so no `body`-string parse is needed on this path. `/self` reads are `effector/selfsurface.rs`.
  `give_back` maps to `abandoned` (not "dormant"): the agency states are exactly
  `active`/`finished`/`abandoned`.*
- **Token store.** A git-ignored `tokens/` `Registry` under the **data directory**
  (`data/tokens`, parallel to `accounts/` — as-built; the mind and data roots coincide
  in the current deployment) keyed `npc_id → {token,
  scope}` (parallel to `accounts/`), a real secret never derived from `body_id`; a `token →
  npc_id → body` lookup built at startup. Scope ∈ {`as-npc` (default, proximity-gated), `direct`
  (by-id, §7/§8.3)}.
- **Character/place persistence.** Extend `Record::settle_path` (`record.rs:343`) with a
  `Kind::Place` arm (locations|geography per the write) and `index_canon` (`record.rs:379`)
  adoption of those dirs, so place writes route through `bench`→commit→disk like eras/stories.
  Character writes use `bench.write_field` on the sheet (persisting through commit). This closes
  D.6 #1 for character and place; `record_*`/`structure_*` verdicts that remain RAM-only get a
  durability pass as their namespaces migrate.
- **Grammar migration.** `query`/`invoke` join the fixed frame as `Availability::Always` tools
  with a **free-string `url`** (`FreeText{Balanced}`), coexisting with unmigrated world/station
  tools; migrate namespace-by-namespace, deleting each namespace's `Tool`/`Choices`/`LIVE`/
  `Within` field as it moves (the `Within` shrink trails each move, §15). The `url` enum arrives
  only after the shrink.

With these settled, implementation proceeds per Part F with no further design input required.

---

# Appendix F — Runtime topology mutation (Step 8)

*Specced now, built later. This is the design for the last thing that makes the world
"fully dynamic": adding, removing and reshaping rooms while the daemon runs. Everything
above makes the world dynamic for what characters *do*; this makes it dynamic for what
the map *is*. It is a real cut of its own because the map is frozen today (`MapSet`
exposes only an immutable borrow; only actors/holds/lift/events are mutable —
world-model survey), and unfreezing it safely is the whole of the work.*

## F.1 The mutation surface

`MapSet` (and `World`, which owns it) gains a small, transactional write surface — the
mirror of the reads it already has:

```rust
impl World {
    fn add_area(&mut self, area: Area) -> Done;          // a new level/region/room
    fn add_node(&mut self, at: &Where, node: Node) -> Done;
    fn add_portal(&mut self, portal: Portal) -> Done;    // a cross-area link (a lift stop, a gate)
    fn place_part(&mut self, at: &Where, placement: Placement) -> Done;
    fn remove_node(&mut self, at: &Where) -> Done;       // and remove_area / remove_portal / unplace_part
    fn retitle(&mut self, at: &Where, name: String) -> Done;   // and the other in-place edits
}
```

Each returns the same typed `Done`/`Refused` (`world.rs:351`) the movement and claim
surface uses — a mutation that would break the map is refused, in second person, not
half-applied.

## F.2 Every mutation re-runs the derived passes, transactionally

The map has derived state that is never authored — `Node::exits` and `Node::visible`
(woven from one-ended doors and sightlines, `load.rs:302/348`), the portal graph
`MapSet.ways`, and the lift's `shaft` (one `Core` per `Level`, ordered by `ordinal`,
`world.rs:512`). A mutation is therefore **apply-to-a-copy, re-derive, validate, swap**:

1. build the candidate map with the delta applied;
2. re-run `weave` (doors mutual, exits filled), `ways` (portals both-way), the
   visibility pass, and `validate::check` — the same passes `assemble` runs;
3. if `validate` fails, discard the candidate and return `Refused` — nothing changes;
4. re-derive `build_shaft` and rebuild/adjust the `Lift` **only when the set of `Core`
   nodes per `Level` changed** (a new floor, a floor drowned), preserving the car's
   position where the floor it is on survives;
5. swap the candidate in under the one lock and log the change as a world event, so it
   is witnessed (§10) — a room appearing or vanishing is a thing bodies notice.

## F.3 The invariants a mutation may not break

- **Doors stay mutual and derived.** A one-ended `off`/`sees` is authored; `exits`/
  `visible` are always re-woven, never hand-set — so a one-way door remains impossible
  to write (`schema.rs:24`).
- **No stranded hold.** Removing a node or a part must first release any hold on it, as
  `leave` does for a departing actor (`world.rs:837`) — a station that vanishes with a
  claim on it would strand that claim forever.
- **No body left nowhere.** Removing an occupied node relocates its bodies to the
  area's `arrival`/a `Core` (and logs it), or the removal is refused while occupied —
  the design's choice is *relocate and tell them*, since a Maker drowning a place should
  not be blocked by someone standing in it, and the body feeling the ground go is good
  fiction.
- **The shaft stays one `Core` per `Level`, ordinal-ordered.** A level added without a
  `Core`, or a second `Core` on a level, is a `validate` failure (refused).
- **Instance ids shift, and that is stated.** An instance id is `(area, node, part-id,
  ordinal)` (§7), so removing the 2nd of three terminals renumbers the 3rd. Topology
  mutation therefore may invalidate outstanding URLs — acceptable because it is rare,
  authored, and witnessed (a character re-reads its near-you index the next turn), but
  it is the reason ids are re-derived, never cached across a mutation.

## F.4 How it is reached, and how it persists

An earlier draft named the Makers' `map_*` verbs (`map_add_place`, `map_settle_border`,
`map_remove_place`, …) as the effector face of this surface. Building it made plain that
those verbs are a *different map*, and this records the correction (design docs are
authoritative; a draft the code disproves is fixed, §CLAUDE.md):

- The `map_*` verbs author **cartography lore** — `sim.record` items of `Kind::Place`,
  the fictional world-map the Makers draw as content (`work.rs:342`, `map_add_place`
  puts a `Place` item; `map_remove_place` lets one go). They do not touch the walkable
  `npc-map` topology at all, and they should not: a Maker drawing a coastline is
  authoring the game's world, not adding a room to the vault it is standing in.
- Runtime topology mutation is the **engine primitive** `MapSet::apply` /
  `World::reshape` (`npc-map/src/mutate.rs`, `world.rs`), which reshapes the *walkable*
  map — the rooms and ways bodies actually move through. Its trigger is not a Maker's
  in-fiction act; it is the **operator / embedder surface** (direct scope, §8.3): an
  embedder growing the world it attached (through the public `World::reshape`), or an
  operator editing the running vault (over the wire).

The operator wire surface is a single effector route, `POST http://local/reshape/:world`
(`npcd/src/effector/reshape.rs`), **gated on `Scope::Direct`** — an as-npc token is
`403`, so a Maker cannot reshape the vault it stands in by reaching for its own device.
It is not in the near-you index (a character is never shown a way to reshape its world),
and it is its own prefix, distinct from the Makers' `/map` cartography station (which
authors lore). The body is a `MapEdit` in its adjacently tagged wire form
(`{"op":"add_node","with":{…}}`); a malformed one is a prescriptive `400`, an edit the
world refuses a `409` in its own words, and a success reports who it relocated and whether
it is durable.

Persistence is `npcd/src/world/mapstore.rs`, invoked by `Hosted::reshape` after the in-RAM
swap: it writes the one area the edit touched back to `<mind>/map/<world>/<area-id>.yaml`
(atomic temp-then-rename), or removes that file for a drowned area. Only the touched file
is rewritten — every other authored file is left byte-for-byte, comments and all — and the
derived `exits`/`visible` are `#[serde(skip)]`, so what lands on disk is the authored form
a fresh load re-weaves. The write is off the world lock and best-effort: the swap is the
source of truth, so a disk failure is *reported* (`durable: "failed"`) rather than
un-happening a reshape that already took. A world with no authored directory (a generated
or test world) is `durable: "ephemeral"` — the change holds for the run only.

*As-built: implemented and tested end to end — `MapSet::apply` (the transactional
mutation), `World::reshape` (swap, re-derive the shaft, relocate the stranded), the
direct-scope `/reshape/:world` route, and the `mapstore` YAML writeback.*

---

# Appendix G — World-state acts as routes

The generic station mechanism (§C, Step 6) mounts only the acts an authored `Part`
names in its `at`. The world-state half of `WORLD_ACTS` (`acts.rs:1311`) is not
`at`-bound — its availability is `Always`/`Nearby`/`Embodied`/`AwayFromHome`, not "at
this part" — so those verbs need their own mounting.

## G.1 One surface, name-addressed — not instance-addressed

An earlier draft of this appendix split these acts by target and mounted the
target-facing ones (`claim`, `operate`, `record_verdict`, …) on a target's *instance id*
(`POST http://local/<ns>/<id>/claim`). Building it made plain that the shape does not fit
the world model, so the design changed and this records the change (design docs are
authoritative; a draft the code disproves is corrected, §CLAUDE.md):

- These acts name their targets **by the name the world writes down**, resolved against
  the live `Choices` sets — not by a map instance id. "The blast door" is a
  `crate::sim` device keyed by *place*, not a placed map part with an id
  (`Within::operable` ← `sim.operable(place)`); "close the longest silence" is a
  `Claimable` subject with no placement at all. Instance-id addressing (§7) is for placed
  parts; it cannot name a sim device or an abstract subject, which is most of what these
  acts work on.
- So they mount on **one personal surface, `http://local/here`** — reachable wherever the
  body stands (like `/phone` and `/self`, §7.2), carrying no instance id. The target is a
  *field in the body*, drawn from an enumerated set the schema advertises.

The verbs `/here` owns are the non-part-bound world-state acts, less the ones another
surface already holds and the ones this design keeps as embodied body acts: `read`,
`scan`, `claim`, `release`, `operate`, `post_notice`, `record_verdict`, `give`, `equip`,
`use`, `gather`, `engage`, `recall`. Speech, movement, `act`, `sleep`, `promise` and
`remind` stay compiled body acts; the lift keeps `/lift`, the phone `/phone`, and the
tower's `command_tower`/`produce` are `AtPart`, so the station mechanism mounts them
where their fixtures stand.

## G.2 OPTIONS is the live `Choices` set

The one computation that decides, for a body standing here, which acts are reachable and
— per enumerated argument — exactly which values it may take is
`tools::specs_within(mode, &within)`, over the `Within` that `Runtime::within` assembles.
`/here` renders that computation directly, so the device and the grammar can never
disagree:

- `GET http://local/here` — the world-state acts `specs_within` admits this moment, each
  with its one-line description. An act with nothing to work on (nothing claimable, no
  fight to `engage`) is absent rather than advertised-and-refused — the grammar's own
  discipline.
- `OPTIONS http://local/here` — one `POST` verb per available act, each body the JSON
  schema of its arguments: the old `Choices` live sets (`Company`, `Reachable`,
  `Carried`, `Equippable`, `Usable`, `DeviceModes`, `Claimable`, `Postable`, …) become
  the `enum` on the matching property, filled from world state at the moment it is asked,
  exactly as the lift's `floor` enum is; free arguments are plain strings.
- `POST http://local/here/<verb>` — the act's declared params are read from the body and
  run through the real dispatch (synthesise the `Act`, `body::perform`, map with
  `enact_response`). The act's own gating is the route's, `409` on a refusal; and because
  every verb here is a body act, none `500`s. Availability is not pre-checked — an act the
  room offers nothing to work on refuses itself in the world's own words, the prescriptive
  error a character corrects against (§12).

`read` and `scan` mount as `POST` verbs here rather than as `query`/GET: both route
through `body::perform` and `read` mutates the reader's read-cursor (which is why
`Choices::Readable` excludes what a body has already read), so they are effectful reads,
not the pure GETs the earlier draft assumed. `engage` is `Embodied` and joins the set
when a fight gives it a posture to take.

## G.3 It does not remove the compiled acts, and that is not a dual path

The same act reaches `body::perform` two ways — as a compiled body act in a turn's
grammar, and as a `/here` route — exactly as an `AtPart` act reaches it both as a
compiled act and as a station route. One implementation (the world's own dispatch), two
front doors: the no-dual-path rule (§CLAUDE.md) is kept.

*As-built: mounted. `npcd/src/effector/here.rs` is the `/here` surface; the near-you
index advertises `http://local/here`.*
