# The Effector Device and `http://local`

*How an NPC interacts with a world that can grow without bound, through a tool
surface that never grows. The body keeps its hands; it gains an **effector
device**; the device speaks HTTP to a real router the world stands up; the schema
that router hands back and the schema that constrains the model's call are both
rendered from the act's own parameters, so they cannot differ; and the world layer is
designed to be attachable, so an embedder can bring their own (**Not built**, §9.1.1).*

This is the fourth surface in the NPC estate. The other three are named in
[`npc_api_gui_design.md`](archived/npc_api_gui_design.md) (archived): what an NPC *is*
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

The operator API is out of the fiction (`npcd/src/api/mod.rs` and its route
modules, `npcd/src/api/route_table.rs`, `npcd/src/engine/mod.rs`). The effector
API is in the fiction (`npcd/src/effector/`) and is what this document
specifies. Everything in it is built unless it is marked **Not built**.

---

# Part A — Why

## 1. The tool surface does not scale, and cannot be made to

The archived audit, [`tool_surface_audit.md`](archived/tool_surface_audit.md), counted it exactly
when it was written: nine body tools implemented and armed, and **one hundred and
fifteen** more authored into the vault with no parameters, no handlers, no plane,
and no presence in the grammar. Its own conclusion: *"Building the foundation is
building all 79, plus what this audit adds."* That is the road this design does
not walk: a world verb is reached through the effector device, not compiled into
the body's catalogue as a tool of its own.

*(Correction, from a later code audit — Appendix D: the **handlers were since
implemented**, in `engine/work/dispatch.rs` (`work::perform`); what the audit called missing is
now the *routing surface*, not the handler bodies. The compiled body catalogue now
holds ten acts (`BODY_ACTS` in `engine/tools/mod.rs`: the eight body acts, `send_image`
and `invoke`), and the station acts still exist as compiled `Tool`s beside their routes
(Appendix G.3). The argument here is unchanged: the compiled per-verb cost is what fails
to scale, whether or not a handler already exists behind each verb.)*

Each of those tools, to become real, is the same five-part cost, paid once per
verb, forever:

1. A `Tool` static (`Tool` in `engine/tools/mod.rs`; the acts are in `engine/tools/act_tools.rs`, `engine/acts/`) with an `Availability` variant and `params`.
2. A `Choices` variant and a `LIVE` row for every world-enumerated argument
   (`Choices` and `LIVE` in `engine/tools/choices.rs`).
3. A field on `Within` — the per-turn world snapshot that is *also the grammar
   cache key* (`engine/tools/within_tools.rs`) — so the new fact is hashable and gate-visible.
4. A handler in `engine/enact/`.
5. Its share of the compiled turn grammar.

The fifth cost is the wall. The turn grammar is the reachable catalogue expanded to
`ACTS_PER_TURN = 2`, budgeted against `MAX_TURN_PATHS = 2_000_000` (both in
`engine/tools/compile.rs`). Because the tree is the catalogue *squared*, the paths per
act are what it is made of, and every enumerated argument multiplies them: measured
on the shipped catalogue in a busy room, about 360 paths an act, so four acts would
be 1.7 × 10¹⁰ paths — a compile that allocated about 390 MB a second without bound
and took the machine down twice. Two acts is about 130,000, which compiles in
milliseconds. A world that grows by a verb with an enumerated argument per machine
therefore grows the grammar of every turn that can reach it. The surface is not just expensive to grow; it is *shaped* to stay
small.

So "the world can expand almost infinitely" is unreachable through this door. The
door has to change.

## 2. The body keeps its hands; it gains an effector device

The NPC does not stop being embodied. It still thinks it is real, and everything
that makes it act alive stays where it is: the body catalogue — `tell`, `whisper`,
`shout`, `ask`, `gesture`, `move_to`, `follow`, `reflect` (`engine/tools/act_tools.rs`) — is
fixed and does not grow with the world. Speech is a body act. Walking is a body act.
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
- Real JSON, only JSON, serialized fast, with prescriptive errors (§8.4, §12).
- Real HTTP methods — `GET` to read, `OPTIONS` to fetch a schema, `POST`/`PUT`/
  `DELETE` to act (§5).
- The schema the world advertises for a call and the schema that constrains the
  model's decode of that call are one description of the act's parameters (§11).

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

| Stays a body act | Also a device route |
|---|---|
| `tell` `whisper` `shout` `ask` `gesture` | every `STATION_ACTS` verb (58, `engine/station.rs`) |
| `move_to` `follow` | `lift_call` `lift_use` (`LIFT_CALL`, `LIFT_USE` in `engine/acts/combat_and_machines.rs`) |
| `reflect` | the world-state half of `WORLD_ACTS` (`engine/acts/mod.rs`), under `/here` (Appendix G) |
| `act` `sleep` `promise` `remind` (human interactions) | the phone acts (`message` `reply` `invite` `open_group` `reach_out`) → `http://local/phone`; `send_image` is not routed (Appendix C.27) |
| — | `bench_*` (11) and `file_*` (5) (`engine/bench.rs`), on the six editing stations; the audited part tools that were built |

Speech does not go behind the device, and not from sentiment: if `tell` were `POST
http://local/say`, a character could only speak where a `/say` route was mounted —
asserting a body can talk near a terminal and not in a corridor, the exact falsehood
the archived audit found when `room.talk` was hung on furniture (`archived/tool_surface_audit.md`
§3.1; the reason is kept here: speech belongs to the body, so it is gated by who is in
earshot — `PersonNearby`, `PersonAmongOthers`, `PersonInEarshot`, `docs/npc_conversations.md`
§12 — never by a station). The device is for what a body does *not* carry with it, which
is everything the world supplies.

The full migration — every tool that stays and every tool that becomes a route,
with its proposed path — is **Appendix A**, for your review.

## 5. The verbs: one device tool, and the body act that finds addresses

The effector device adds exactly one tool to the compiled catalogue, and it never
changes as the world grows. Behind it is an ordinary HTTP method, so the model's
prior applies directly. The model-facing frame is `invoke` plus the body act `scan`:

```
scan()
    With no place named, a prose survey of what the body can work where it
    stands: one line per station, with its state in a few words and the exact
    address and fields to `invoke` (§6). A body act, not a device call.

invoke(url: string, body: object)
    Act at the address. Issues the resource's effectful method (POST) with a
    JSON body. Returns the result, or a prescriptive error (§12).
```

There is no model-facing read tool. Finding out is `scan`, and the survey is built
from the same source the grammar's `invoke` enum is (`address_of`, `verbs_at`,
`Sim::reading`), so every address it names is one `invoke` accepts. **The model
never picks the method:** every address is a single verb, `invoke` uses its
effectful method, so a wrong-method 405 is not a mistake available to the model.

The router keeps `GET` and `OPTIONS` handlers on every resource. They are for external
callers and the tests, which drive them over the mounted router (§8.2); no tool the
model can call issues them, and the engine does not call them either. The engine
compiles the stencil from the same act parameters `OPTIONS` is rendered from (§11), so
the two cannot differ.

### 5.1 `OPTIONS` completes the abstraction

A resource's *shape* is fetched with a real method, not a bespoke call. `OPTIONS
http://local/<ns>/<id>` returns the verbs the resource accepts and the JSON schema of
each one's body, built from the act's own `params` (`schema` in
`npcd/src/effector/station.rs`):

```
OPTIONS http://local/chronicle/chronicle-terminal~0
→ 200 {
    "id": "chronicle/chronicle-terminal~0",
    "summary": "…the part's own summary…",
    "methods": {
      "POST": { "verbs": {
        "add_entry": { "body": { "type": "object",
                                 "properties": { "to":   { "type": "string" },
                                                 "what": { "type": "string" } },
                                 "required": ["to", "what"] } },
        "rewrite_page": { "body": { … } }
      } }
    }
  }
```

`OPTIONS` is chosen over `HEAD` because it is the standard "describe this resource /
these are its methods" verb, it can carry a schema *body*, and it maps one-to-one
onto the discovery step the design needs; `HEAD` (schema in headers) was the
alternative floated in review, and `OPTIONS` is the decision. Types are passed through
as the catalogue declares them (`string`, `boolean`, …) and no enum is invented there:
the live enums of the world are the grammar's (§11) and, for the `/here` surface, the
schema's own (Appendix G.2). The schema is read by external callers, not by the model.

## 5.2 Verbs are resources

A thing in the world is addressed by its id — `http://local/chronicle/<id>` — and
**each verb it affords is itself an address beneath it**:
`http://local/chronicle/<id>/add_entry`. That verb path is what `invoke` acts on
(`POST`), and it takes exactly one body. The parent (`.../chronicle/<id>`) answers
what the thing is and lists the verbs beneath it — as `verbs` and as `invoke`, the
whole `http://local/<ns>/<id>/<verb>` addresses to act on — and its `OPTIONS` carries
each verb's body schema (§5.1). A verb path itself answers only `POST`.

The parent read also carries the thing's live `state`, drawn from the sim
(`Sim::reading`): a machine's mode and whether it answers, and — for a station whose
acts draw on the tower — the stockpile, what it can make from it, which of the eight
queues are free and what is queued. A question about the world ("which fabricators are
running, what is there to make from?") therefore has an answer at the machine, rather
than only a refusal or a neighbour who knows no more. A station that commands the tower
reads the tower's own state besides: posture, depth and shields, what it draws per
minute (`energy_per_minute`) and how long the energy lasts at that draw
(`minutes_of_energy`), any contact closing on it (`contact`: id, kind, `minutes_out`),
and the open `decisions` with who, by name, is deciding each (`held_by`, or null).

This is the decision that makes typed `invoke` total rather than partial. A resource
that afforded several verbs could not be typed by a single body — the address could
not say which verb the `invoke` would pick. Addressing the **verb** removes the
ambiguity by construction: every invokable address has exactly one body, so the
grammar can pair each address the `invoke` enum offers with the body it takes
(§11), and the schema `OPTIONS` returns is exactly the schema `invoke` needs.

So the grammar of an address is:

```
http://local/<ns>/<id>            a thing — GET its state, OPTIONS lists its verbs and bodies
http://local/<ns>/<id>/<verb>     a verb — POST to act, with its one body
```

The personal surfaces follow the same shape: `http://local/phone/message` is a verb
resource; `http://local/self/plan` is a readable one.

The station router mounts each verb as a sub-path of the thing, and the `invoke`
grammar types its body from the act's own parameters (§11), so every address the
character is offered is typed, single-verb or not.

## 5.3 The tower has a clock, and what it asks has an owner

A stockpile that never moved would give a character no reason to choose anything, and
a question put to a cast of equals — "what should the tower do next?" — has no answer
anywhere in the world, so it goes between them and becomes the loop. Two parts fix that
at the source rather than breaking the loop after it forms.

**The tower's state moves** (`sim/upkeep.rs`, driven by `engine/watch.rs`). While anybody
is in the world, the tower is drained each beat at a rate per real minute: a base draw,
more with shields up, a siege or a batch running, half of it when dug in. A contact is
sighted after a quiet spell, counts down, and on arrival is met by whatever the tower
is doing: dug in, it passes over; shielded, it costs energy to hold; otherwise it takes
energy and metal. Folding away leaves it behind. Shields fail when the spendable energy is gone. A beat is
capped at five seconds, so a stalled daemon is not charged for the gap, and an empty
world does not run its clock. What happens is stirred into every occupied room of the
tower's area — outcomes as notes, a sighting or a falling reserve as a wake — so the crew
learns of it together.

**Strategic decisions have one owner** (`sim/decisions.rs`). What presses on the tower
(`Tower::pressing`: a contact, the reserve under a floor) is put on the order board as
an order set by "the tower", once per cause and announced once. A character takes it
from the order table; `take_order` is exclusive, so a second one is told who has it.
While somebody holds a decision the tower is theirs to command: `command_tower` by
anybody else is refused with the holder's name and the decision. Nobody is in the way
until somebody has taken one. A command that lands settles what its maker holds, and a
settled cause is not put up again while it stands. A cause that passes — the contact
folded away from, the reserve restored — comes off the board, held or not, and is a new
decision if it returns. The klaxon that announces a decision names it and says how to
take it on (`claim "<the decision>"`); `claim` offers every unheld order from wherever
the character stands, so hearing the klaxon is enough to act on.

**A tower can recover.** Standing, a held contact and the shields spend only the energy
above a reserve (`RESERVE`, 200), so a tower run down is not dead: it can still afford to
dig in. A buried tower is paid by the ground (`TAP_YIELD` a minute) and, with its draw
halved, gains energy; its reading says what the ground gives and reports
`minutes_of_energy: null` when the ground out-gives the draw. The cost is what dug-in
already costs: it cannot fold or lay siege until it surfaces. The grammar keeps this
honest: `raise shields` is offered only on enough spare energy to hold them a minute,
`drill down` only to a tower that can pay for a metre of it, and a drill deeper than the
stockpile covers is refused with how deep it can go.

## 6. The near-you index is an internal route; the character surveys with `scan`

The list of what is reachable from here is a route — `GET http://local/`, served by
the same handler machinery as everything else, consistent and externally
inspectable. It is not a tool the model calls and it is not placed in the prompt:
the character's view of the same ground is the `scan` survey. Its answer is the
**top level and only the top level**, recomputed from where the body stands, each
entry carrying its world item id (§7):

```
GET http://local/
→ 200 { "routes": [
    { "url": "http://local/here",     "summary": "what you can do where you stand" },
    { "url": "http://local/history",  "summary": "what you have done and seen" },
    { "url": "http://local/phone",    "summary": "your phone" },
    { "url": "http://local/self",     "summary": "your own plan, orders and memory" },
    { "url": "http://local/lift/command-shaft", "summary": "the lift" },
    { "url": "http://local/chronicle/chronicle-terminal~0", "summary": "…the part's name…" }
  ] }
```

**Where the model sees this ground is the point.** It is only ever true *now*, so
it is not carried in the prompt or as a percept: the character asks for it with
`scan`, and the survey is recomputed from the room each time. An earlier design
delivered the list each turn as a superseding percept; the survey replaces it
because it names the exact addresses `invoke` takes, where the list named bare
resources `invoke` cannot.

The survey reads, in the character's register:

```
Here you can work:
- Bridge console: besieging the eastern ledger, shields down. You can `invoke` `http://local/tower/bridge-console~0/command_tower` (needs action).
To act on one, `invoke` its address.
On your phone:
- Pax: 1 unread.
To read them, `read`.
Lately: Pax came in.
```

The survey also carries what the body holds, wherever it stands, each compressed to
one line that still reads as a sentence (`engine/on_you.rs`): the open phone threads
(who it is with and how much is unread — never what was said, which is for `read` —
and no instruction on how to send) and the recent witnessed past as the narrator
writes it. A station's line never lists what it can make: a fabricator's catalogue is
long, so the line carries its mode and queues only. Each section is left out when empty, and the
phone section when the body carries no handset. The character's identity, beliefs and
plan are in the system prompt, so the survey does not repeat them.

**The survey is also served.** `GET /v1/npc/:nid/scan` (owner or admin) returns
`{ prose, survey }`: `prose` is the survey word for word, and `survey` is the data it
is written from (`engine/survey.rs`, `Survey::of`): each station's name, address,
state, verbs with every parameter's type, whether it is required and what it means,
the tower's options and the raw reading, plus the threads and the recent past. The
console's Scan tab on a character renders it, with hover text on stations, addresses
and parameters, so a scan can be checked without making the character act.

**The interactions stay inline.** An `invoke` and its result or error are actions
and their outcomes, and they belong in the turn stream as
`<tool_call>`/`<tool_response>` exactly like any act (§5, §12). *What is here* is
a survey the character asks for; *what I did about it* is the conversation.

## 6.1 The system prompt explains the device

The character is told, once, in the system prompt, what the effector device is and
how to use it — the same place the body's acts and the world are explained
(`engine/prompt.rs`, the `WHAT YOU CAN DO` frame). The copy establishes, in the
character's own register:

- **What it is** — a handheld effector: it senses what is around you and lets you
  act on it. Its screen lists addresses; the world answers at them.
- **How to look** — `scan` with no place named lists what you can work where you
  stand, one prose line per station: its state in a few words and the exact
  address and fields to `invoke` (`engine/survey.rs`, built from the same
  `address_of` / `verbs_at` / `Sim::reading` the grammar's `invoke` set uses).
  Looking never changes anything.
- **How to act** — `invoke` an address with the fields it asked for. The world
  answers, or tells you plainly what was wrong so you can fix it.
- **What the addresses are** — `http://local/...`, each naming one thing in the
  world; walking changes which ones the survey lists.

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
`http://local/chronicle` but `http://local/chronicle/chronicle-terminal~3`. Three
reasons, all from review:

- **Uniqueness.** A level has several terminals; a building has several lifts. The
  id is what tells `chronicle/chronicle-terminal~0` from `chronicle/chronicle-terminal~1`.
- **External addressability.** An external client (§8.2) works *the third chronicle
  terminal* by naming it — `POST http://local/chronicle/chronicle-terminal~2/add_entry` —
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
   the map: `within_reach` (`npc-map/src/perceive.rs`) over the node's placed parts,
   `at_landing` (`npc-map/src/world/lifts.rs`) for the lift. Frozen topology, live standpoint.
2. **Condition** — is the thing in a state that affords the call? A powered-down
   command table affords `OPTIONS` and `POST {powered:true}` but none of its
   working verbs; a claimed station affords reading but not a second body's write; a
   part in one `mode` affords a different verb set than in another (`Part.modes`,
   `part.rs`; `binds`/`holder_of`, `world.rs`).
3. **Standing** — is it a personal route, always yours (history, phone)?

Reachability and standing live in the map and world. **Condition needed its own
state.** Per-item mutable state — a machine's mode, which body holds a claim — is the same
"dynamic half" the world keeps for actors/holds/lift, but *per placed item*: a `Part` is a
shared catalogue entry placed by reference, with no per-instance state
(`npc-map/src/part.rs`). The per-item state record is `Sim.devices` (`Devices` in
`npcd/src/sim/device.rs`), keyed by the item id of §7 (`Sim::station` and `station_mut` in
`npcd/src/sim/mod.rs` resolve an instance id, and refuse a stale or forged one rather than
resolving to something nearby), living in npcd's `Sim` beside the record and ledger — not in
`npc-map`, whose `State` (`npc-map/src/state.rs`) is only what two copies of one world must
agree on. Route handlers read and write it, and the simulation tests of Part F drive it.

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
that already exist and **nested into the npcd router at `/v1/local`**
(`npcd/src/tenant/routes_and_sinks.rs`) —
*not* a second `web` `local_api` site. (As-built: `web` dispatches sites by the `Host`
header (`web/src/server.rs`), so a `local` *host* would need its own hostname, whereas a
`/v1/local` *path* already resolves to the npcd site — the nest is the clean
equivalent, no `sites:` change.) The router runs behind its **own** device-auth
middleware (§8.3), *not* `guard::Api`: that wrapper hardcodes the operator `X-Tokera`
role check with no escape hatch (`host::guard::Api`), which is the whole reason the device
carries its own.

### 8.1 The fast path skips TCP, never the API

An NPC's call must not pay for a socket. But it must still be a real API call —
same routing, same handlers, same validation, same error shape. So there is **one
router with two entry paths:**

- **In-fiction (fast).** The engine builds a real `http::Request`, stamps the auth
  header (§8.3), and invokes the router *in-process* — the `tower::ServiceExt::
  oneshot` path the existing routers are already tested through (`npcd/src/api/`). No
  TCP, no serialize-to-socket, no syscall. This is the path every `invoke`
  takes, at the world's cadence (`EVERY = 500ms`, `npcd/src/engine/driver.rs`) times the cast.
- **External (real socket).** The identical routes answer on the bound port
  (§8.2), for an external client.

Both hit the same handler code. The fast path is not a second implementation; it is
the same service invoked without a network in front of it. Nothing about the
contract — URLs, methods, JSON, schemas, errors — differs between the two.

Three implementation realities the review surfaced — none fatal, all to be designed
for rather than discovered:

- **The sync/async seam.** `within` and the act path (`record_act` → `act_on_world`
  → `body::perform`) are synchronous and run inside the async `character_loop`
  (`engine/runtime/turn.rs`, `engine/runtime/acting.rs`). A `tower` router call is a `Future`, and
  `block_on` from a worker thread panics — so the fast path either makes that slice
  `async` (viral through a deep sync tree) or hops through `spawn_blocking` / a
  scoped current-thread runtime. Cheap in bytes, not free in plumbing; this seam is
  the real cost, chosen deliberately.
- **No re-entry while holding the world lock.** `Hosted.state` is a non-reentrant
  `std::sync::Mutex` (`npcd/src/world/mod.rs`). A handler that holds `hosted.with(...)` and
  then invokes the router again in-process (a sub-resource, an `OPTIONS`, the near-you
  index) deadlocks silently. **Invariant: never invoke the router while holding the
  world lock** — handlers acquire, act, release, as the lift handlers already do
  (`engine/enact/movement.rs`) — and it is tested.
- **The production Router handle.** The merged router is handed to `web`, and
  `Runtime` also retains it (`set_effector_router`) and
  drives it via `effector_invoke` (`engine/runtime/device.rs`), an `async` method — the sync/async seam above is
  handled by making the effector call at the async layer, never `block_on`.

### 8.2 The external mount

The effector routes are also reachable from outside, on the hosted domain, under a
proper path. The in-fiction `http://local/chronicle/chronicle-terminal~0` is mounted externally as
`/v1/local/chronicle/chronicle-terminal~0` (the same router, nested — §8), authenticated by the
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
case. This is where "who is acting" is decided, once. (As built, the two shapes are
two middlewares rather than one enum: `host::guard::Api` resolves the human, and
`device_auth` in `npcd/src/effector/auth.rs` resolves the token into a `DeviceCaller
{ npc_id, scope }`.)

Every NPC is seeded at startup with an opaque token, carried as an auth header on
every effector call and **invisible to the LLM** — the model never sees it, never
types it, cannot leak it into prose. The engine stamps it on the in-process request
(§8.1); an external client sends it directly.

- **Seeding & lookup.** The token is minted when the NPC starts and recorded in a
  quick-lookup table token → body. It must **not** be derived from the deterministic
  `body_id` (that would be guessable) — it is a real secret. It is persisted, one
  file per character in the `tokens/` registry under the data directory
  (`Tokens` in `npcd/src/effector/token.rs`, opened in `npcd/src/tenant/tenant_start.rs`),
  so an external token survives a restart although the world itself does not.
- **The v1 check, whole.** Authorized iff the request carries the header, the header
  names a token, and the token is in the table (one lookup). No roles, no ownership
  on this surface — the token *is* the identity, resolving to a body whose standpoint
  the handlers read from world state (`body_of`; `at_landing`, `npc-map/src/world/lifts.rs`;
  `within_reach`, `npc-map/src/perceive.rs`).
- **Scope decides proximity.** A token carries a scope. An **as-NPC** token — the
  character's own — is proximity-gated: it reaches only what is within the body's
  standpoint, so an id not near the body 404s, in the fiction or from outside. A
  **direct** token — for an operator or embedder driving the world externally —
  addresses any item by id regardless of standpoint. The scope is a property of the
  token, checked after the one lookup; the in-fiction fast path always uses the
  as-NPC scope, so a character can never reach past its own standpoint (§7).

**The device surface needs its own middleware — it cannot reuse `guard::Api`.** The
one route-registration primitive, `Api::route`, hardcodes `require(headers, roles,
min)` reading `X-Tokera-*`, with no escape hatch by design (`host::guard::Api`). The
effector routes register through a *parallel* wrapper (or a `Principal::Npc` path in
`require`) that **ignores `X-Tokera-*` entirely** and resolves the token (`device_auth`
in `npcd/src/effector/auth.rs`). It is not a detail, and two hazards make it sharp:

- **The fast path** builds an `http::Request` and stamps the token (§8.1). Because
  npcd *believes* inbound `X-Tokera-*` from the peers it trusts
  (`web::Builder::trust`), that request must never carry one, or the
  operator middleware would authenticate a character as a human — pinned by a test,
  both directions.
- **The external mount** (§8.2) is on the same gateway-fronted domain, so a client
  can send *both* a token and `X-Tokera-*`; the effector middleware must read only
  the token and never fall through to the role check.

One token, one lookup, one meaning of "who is acting." The two scopes are the whole
of the authorization, and a re-mint is the only revocation (`Tokens::mint` rotates
a character's token and the old one stops resolving).

### 8.4 The JSON must be fast, and wrong calls must say why

Two hard requirements on the serializer, because both the fast path's throughput and
the wager of §3 depend on them:

- **Fast.** `serde` + `serde_json` (the estate's serializer, `Cargo.toml`) on the
  in-process path, with request/response values built and parsed without an
  intermediate socket encode. Handlers deserialize into typed structs (not
  `Value`) wherever the schema is fixed, so parsing is a single pass into the shape
  the handler wants. Performance is a build-time concern to measure, not assume —
  the device call is on the world's hot loop.
- **Prescriptive.** A refused call returns a structured error the character can
  correct against: the estate's `err()` helper (`npcd/src/api/errors.rs`) emits
  `{error, detail}` and the effector routes use it unchanged, so `409
  {error:"refused", detail:"<the world's own sentence>"}`. The `detail` is the
  correction signal the whole wager (§3) rests on: per §11 the handler's
  validation *is* the enforcement for every type the decoder only shapes, and it
  phrases its refusal in second person, as the world would say it. **Replaced
  design:** a `field` naming the offending key on every error is not built; §12
  gives the reason and the one case, `bad_json`, that carries one.

## 9. Who registers a route — the join, and the attachable world

A route is four things: a **path** (carrying an id, §7), a **method**, a **JSON
schema** for its body, and a **handler** that reads or changes the world. This is
the seam where the two halves meet: the world declares routes as data, and both the
device (for the character) and the engine (for the stencil, §11) read the same
declaration.

The map already has the right shape. A `Part` in `npc-map` is a catalogue entry —
terminal, board, chair — placed into rooms by reference, and it deliberately does
*not* name the acts it offers (`npc-map/src/part.rs`; the act names its stations, `Tool::at`): that vocabulary lives in npcd's
engine. Under this design it becomes *routes addressed by the part instance's id*,
mounted wherever that instance is within reach. The chronicle terminal's `binds`,
`modes`, and the audit's verbs become `http://local/chronicle/<id>/...` routes. One
declaration, mounted per standpoint, is what makes the situated address space of §7
fall out of the map, because placed instances have ids.

**Minting the ids was the biggest hidden cost, and it comes from the map.** A
`Placement` is anonymous and count-based (`Bare` or `Counted{part,count}`,
`npc-map/src/part.rs`); `part_ids_at` de-duplicates by catalogue `part.id` within a node
(`npc-map/src/load.rs`), so two chronicle terminals in one room collapse to a single
entry there, and `within_reach` returns catalogue ids (`npc-map/src/perceive.rs`). Two
terminals of one part in one room therefore needed an enumeration that does not
de-duplicate, and an id that tells them apart. Because the world's map is not persisted
(bodies re-enter deterministically, `engine/runtime/`), an id must be a **pure function
of the map** to stay stable across restart and be shareable by console and test.
**As-built:** the id is `<part-id>~<ordinal>` (tilde is URL-unreserved and appears in no
kebab-case part id, so the split on the last tilde is reversible; the ordinal is
**0-based, world-wide and written in base 36** (`0-9a-z`), counting placements of that
catalogue part across the whole map in a deterministic walk — areas by id, nodes in file
order, placements in file order) — `npc-map/src/instance.rs`: `PartInstance::id`,
`base36`, `MapSet::{instances_at, resolve_instance}`. Base 36 keeps a filling room's
ordinals to one or two tokens, which matters because these ids are spent in the model's
context again and again.

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

**The url is a stencil, not free text (the anti-hallucination cut).** `invoke`'s `url`
is a grammar-forced enum, not a free string: it is bound to each reachable resource's
verb-paths (`Choices::InvokeUrl` ← `Within::invokable`, `<resource-url>/<verb>` from
`effector::station::verbs_at`). So the decoder is forced through a real address the device
actually lists — a character can never `invoke` a hallucinated guess that 404s
(`http://local/command-table`) nor a bare resource (a 405); it acts on exactly what
stands within reach. The `scan` survey renders each station with the whole verb-path
to copy (`engine/survey.rs`).

**Only an address a route answers for is offered.** A station namespace is mounted only
when some catalogue act attaches to a part under it, and `map` is held back from the
generic routes, so an instance of a part with no verbs (a seat, a blast door, a wall
turret) or of a `map` part has nothing behind its url: `GET /room/seat~0` is a `404`.
`router::address_of` is the single place an instance's url is composed, and it returns
`None` for such a part (`router::serves`: its namespace is mounted, or is `orders`, which
the plan bridge serves). The near-you index, the `scan` survey and the grammar's `invoke`
enum all read it, so what a character is shown, what it may name and what the router
serves are one set; `every_address_a_body_is_offered_is_served` stands a body at every
place in the vault and reads the resource behind every offered address. A part that later gains a verb is served the
moment its act names it, with no change here.

Handlers reach the world through the one lock: `Hosted`, a single mutex over
`{ world, attention, sim, rooms }` (`npcd/src/world/mod.rs`), with `with` the sole write
path, `read` for queries, `with_both` for world+sim. A device
handler is an `engine/enact/` handler with an HTTP envelope; the lift handlers
(`engine/enact/movement.rs`) are the template, and their typed refusals (`Refused`,
`npc-map/src/world/events.rs`) become the route's error bodies (§12).

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
computation crosses two locks (`npcd/src/world/mod.rs`). An attachable world must not
reintroduce that hazard: the extension point is a provider of **route handlers that
run inside `Hosted`'s lock** (a `dyn WorldRoutes` invoked under `with`/`with_both`),
with the embedder's state living behind that same mutex exactly as `Sim` does today —
not a second, independently-locked world object. The embedder owns its state; it does
not own a second lock. That is the version of "attachable" that keeps the invariant,
and it also bounds *how deep* the reach can be: as deep as `Sim`'s, no deeper.

### 9.1.1 The `WorldRoutes` trait (Step 7) — Not built

This section is the plan, kept because it is still the target: no `WorldRoutes` exists in
the code. The vault's routes ship as concrete modules (`npcd/src/effector/`) reached through
`Hosted`, and `Runtime` holds the effector `Router` (`set_effector_router`), not a
provider. The device machinery is world-agnostic: the token auth (§8.3), the near-you framing
(§6), the `invoke` verb (§5), the per-address body stencil (§11), and the
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

**The vault would be the reference implementation, not a special case.** npcd's own
`WorldRoutes` impl is the wiring that exists, read through the trait: `reachable` =
`within_reach` + the lift landing + `instances_at` mapped through `namespace_of` (§C.0);
`read`/`schema` = the station and lift `GET`/`OPTIONS` handlers; `invoke` = synthesise
the `Act` and run `body::perform` (§C, D.5). Factoring it behind the trait is a
lift-and-name of code that already exists and is green — the change is that the router
would call `provider.reachable(...)`/`read`/`schema`/`invoke` instead of naming the
vault's functions directly, and `Runtime` would hold a `Box<dyn WorldRoutes>` (installed
after construction, defaulting to the vault) beside the world lock.

**Replace, not compose (Q4).** One active `WorldRoutes` per daemon: an embedder's
provider *replaces* the vault's; npcd keeps only the personal roots around it. A world
that wants both the vault and its own affordances composes them inside its own impl,
not in npcd — which keeps npcd with exactly one answer to "what is reachable here."

*Not built: the trait itself. Step 7 is the refactor that names the seam. It waits for a
second world (an embedder) to be real, because a trait with one implementation is a shape
guessed rather than a seam proven — the interface above is the target that refactor lands
on.*

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

- **World state** — `Hosted`'s `world`/`sim`/`rooms` (`npcd/src/world/mod.rs`); the
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
built — the cast is shared as `Arc<Npcs>`, which takes no lock
(`docs/npcd_cast_lock_free.md`), and installed on
`Runtime` (`set_npcs`/`npcs` in `npcd/src/engine/runtime/mind_hold.rs`), reusing the one
`Npcs`/substrate handle, never a second. `plan_*`/`orders_*` route through
`effector/plan.rs` (mounted at `/plan` and `/orders`, and excluded from the generic
station nest so they never reach `Sim.ledger`) to `Npcs::put_strategy_self`
(`npcd/src/npcs/methods.rs`), an owner-blind write of the character's own `agency` layer
that projects on the next turn through `persona::intent` (`npcd/src/engine/persona.rs`,
which reads the first `active` strategy). The bundled `npcd/projection.yaml` also projects
an `agency` layer of its own. The write target is `Npcs::put_strategy`, not `engine/authoring.rs`
(which is only a life-document parser).* (The one hard invariant
the authoring plane already carries, and which this must not breach: a belief-write
is the character's own, never a tool's silent edit — `npc_mind_design.md`; the
effector's belief routes are the character revising itself, on the record.)

## 9.3 The four stores a handler touches

The effector API is an envelope over existing state, not a new store — but "existing
state" is **four** stores, not the two (world / projected) the sections above imply.
Appendix D is the grounded account; this is the taxonomy every handler is written
against:

1. **The npc-map `World` (RAM)** — who is where, what each body holds, the lift,
   witnessed events. Held in RAM and snapshotted into the substrate's entity records
   (`body/<body>` and `lift`, `npcd/src/world_state/snapshot.rs`), so it survives a
   restart; bodies also re-enter deterministically (`engine/runtime/`). Written by the
   world acts (lift, claim/release/operate/…, `engine/enact/`). Claim = proximity +
   `Actor.hold`.
2. **`Sim` (RAM, snapshotted)** — everything about the world that is not its map: the
   `record` authored-content state machine (`sim/record.rs`), the `bench` per-body
   working copy (`sim/bench.rs`), the `ledger` (orders/verdicts), missions, phone.
   **This is what most authoring namespaces actually write** — Appendix C's `Store =
   World` for them means `Sim` here. Its entity maps are persisted as `sim/<map
   path>/<key>` records and the rest as `sim/meta` (`SIM_MAPS` in
   `world_state/snapshot.rs`), so a restart keeps the record items, ledger and open
   working copies; the benches' *documents* live on this machine's shelf, not in the
   snapshot. Claim = `Record` custody + bench isolation.
3. **The mind folder (disk, or git)** — the authored lore: canon, eras, stories, the cast,
   the craft libraries, the map source (§D.1). Two doors that never conflate: the
   **console** editor (`npcd::mind`, admin-gated, §D.3) and the **effector** working copy
   (`Sim.bench`, claim+commit gated, §D.4). Only a subset of authoring verbs reach it
   (chronicle-era, story, character, place, portrait-prompt, craft-library, and the raw
   `bench_`/`file_` surface); the rest stay in `Sim` (D.5).
4. **The substrate (projected, persisted)** — agency, belief, memory; redo-logged; the
   belief-write invariant (§9.2). This is §9.2's "projected store." The `plan_*`/`orders_*`
   routes write it (`effector/plan.rs`, D.6 #2); no station act handler does.

The rule stands: a handler reads and writes the store that already owns that fact,
never a parallel one. An **attachable** world (§9.1) would bring its own state behind
the one lock, exactly as `Sim` does. Appendix D.5 maps every authoring namespace to its
real store; D.6 lists what is still open.

## 10. The consequence loop — invoke acts, the world answers over time

`invoke` returns immediately; what it sets in motion unfolds over ticks and returns
through perception, not the return value. `invoke("http://local/lift/command-shaft/call",
{})` returns `{ ok:true, detail:"The lift is on its way." }` that
turn. The car then travels on the world clock — `step_lift` (`npc-map/src/world/lifts.rs`),
run by `World::tick` (`npc-map/src/world/movement.rs`), voices each `Moment`
(`npc-map/src/lift.rs`) as a `Stirred` event on the right landing — and the character learns it
arrived the way it learns anything that happened while it was not looking: through
`witness`/`stir` (`since` in `npc-map/src/witness.rs`, `World::stir` in
`npc-map/src/world/speech.rs`) → `perceived::digest` (`npcd/src/engine/perceived.rs`)
→ the narrated perception at the head of its next input (`compose_narrated` in
`npcd/src/engine/mind/perception.rs`), and through the `scan` survey, which at the landing
now shows the car boardable.

The loop: **`invoke` → acknowledgement now → world ticks → effects perceived +
near-you index changes → the character `scan`s or invokes again.** The return value is
the receipt; the world is the answer. The tick ordering — journeys, then the
building's say, then perception (`advance` and `perceive` in `npcd/src/engine/environment.rs`)
— is unchanged; the device is another writer under the same lock.

---

# Part D — The stencil (why the model calls it well)

## 11. The schema drives the stencil — the mechanism that makes this cheap

This is the load-bearing idea, and it needs stating precisely, because a review of
this design found the first draft overclaimed it.

The stencil turns a JSON tool description into a constrained-decode tree
(`compile_tool_call_tree`, `stencil_tree.md` §8.3). What it *enforces* exactly: a
**string enum** decodes to one of its arms; a **boolean** to `true`/`false`. What it
*shapes but does not type-enforce*: `integer`, `number`, `array`, `object` decode as
any structurally-valid JSON of that shape, lookahead-terminated (`ParamType` in
`candle-conversation/src/stencil/tool_call/param.rs`, built in `builder.rs` beside it),
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

**How the body reaches the stencil: one tree, every address carries its own body.**
The whole-turn grammar is cached on `(Deliberation, Within)` and hits almost always
*because `Within` is a room fact shared across the cast*. `Within` carries
`invokable: Vec<Invokable>` — each `(url, act)` the body can reach from where it
stands (`Runtime::invokable_urls`, the verbs `station::verbs_at` lists under each
part) — so the address set is part of the key and two characters in the same room
share one tree. There is no per-character focus, no second cache and no spliced
tree.

`invoke.url` is a closed enum over those addresses, and each value brings the body
it takes. The stencil expresses that as `Param.shapes` (`candle-conversation/src/stencil/tool_call/param.rs`):
a list of `(enum value, sibling fields)`, where a value's fields are the *required* siblings
that follow it and replace any same-named sibling (`build_discriminated` in
`candle-conversation/src/stencil/tool_call/builder.rs`). `engine/invoke_body.rs` builds the shapes: an address's body
is its act's own parameters, typed exactly as the compiled act types them, so a
station verb's body is the body the same act would carry as a native call, an
act with no parameters takes `{}`, and an address nobody could perform here (a
required parameter with no live value, or the act cooling down) is not offered at
all — the same `tools::performable` gate that decides whether the act is compiled.
Equal shapes share one tail `SpecId`, and the compile memoises on
`(SpecId, left context)`, so the tree grows with the number of *distinct* bodies and
the cost of one more address is its url, not its body. The lift's `invoke` is a
compiled act with its floor enum and is unaffected.

The measured cost is in `stencil_tree.md` (as-built deltas, "A value can bring its
own fields"); `invoke_body`'s tests hold it: a room with two of every station part
and everything a body could name in reach compiles in milliseconds, and the
per-address cost is bounded by test.

**How the character reaches a schema without spending a turn on it.** The naive path
— discover, read the schema, then `invoke` — is three turns to do one thing, and
at 2 Hz that is slow. Two things collapse it:

- The `scan` survey (§6) names each station's whole verb-path and the fields it needs,
  so the body the character is about to write is already in front of it.
- Every address the `invoke` enum offers is already typed, so "operate the thing in
  front of me" is one `invoke` with no preceding `OPTIONS`.

A hallucinated address is impossible (the enum is closed) and a body that does not
fit its address is impossible (the shape is that address's own). The remaining
failures are the ones the decoder cannot see: a free-text field left empty, and the
world refusing for a reason only the world knows (§12).

**The schema is live, so `Within` is rebuilt every turn.** Between the turn a schema
is seen and the turn `invoke` fires, the world ticks and other bodies act under the
one lock, so an enum (`floor_names()`, `claimable_at`, `modes_here`) can go stale —
the car arrives, a floor opens, someone claims the terminal, the body walks and the
route 404s. The address set and every live enum are recomputed from the world at the
start of each turn; §13's "pure function of world state" is what makes that safe and
cheap. A schema that went stale mid-flight fails benignly — the world refuses and
says why (§12) — but the design does not pretend the first read is durable.

The API stays stateless (§13); the engine keeps no per-character state for the
stencil.

## 12. Prescriptive errors close the loop the model already knows

A wrong `invoke` comes back as a real API error (§8.4): a status and `{ error,
detail }` — `409 refused` when the world declines, `404 not_found` for an id out of
reach, `404 unknown_verb` for a verb the part does not take, `401 unauthorized` for a
bad token. The character does not read the envelope: `Runtime::enact_device`
(`npcd/src/engine/runtime/device.rs`) hands it the world's own sentence, the `detail`
("You are already carrying a mission."), as its `<tool_response>` next turn — the
channel act outcomes already ride back on (`record_act` in
`npcd/src/engine/runtime/acting.rs`; `compose_narrated` in
`npcd/src/engine/mind/perception.rs`) — and the status goes on the feed line for the
operator. It corrects, because correcting a JSON API against an error body is a thing
the model has done a million times. The rule: **a device error is what a
well-behaved REST API would return, and nothing more clever** — an invented format
would forfeit the whole prior the wager rests on.

**Replaced design.** An earlier draft added a `field` to every error, naming the
offending key (`400 missing_field {field:"powered"}`). It is not built, deliberately:
the grammar types each address's body (§11), so a required field cannot be left out
and an enum cannot take a value outside its set, and what is left — an empty free-text
field, a refusal only the world can make — is refused by the act's own handler in the
world's own words. Station routes copy the declared parameters from the JSON object and
ignore the rest; they validate no body of their own, and the projected-store routes
(`effector/plan.rs`) name the offending field in the `detail` sentence ("`<field>` is
missing or out of range") under `409 refused`. The one `field` that exists is in
the answer to a body string that does not parse, `{error:"bad_json", detail,
field:"body"}`, which `enact_device` returns for a caller that is not the grammar.

## 13. Statelessness — the engine calls the API whenever it needs the answer

The API is a pure function of `(world state, caller token → standpoint, method,
path, body)`. No session, no handshake, no cursor inside it. The engine calls the
router in-process for every `invoke` (`Runtime::effector_invoke`), and builds what the
character is shown — the `scan` survey and the grammar's address set — from the same
functions the routes answer from (`address_of`, `verbs_at`, `Sim::reading`), without a
round trip; the near-you index (`GET http://local/`) and `OPTIONS` are read by
external callers. Nothing a call does leaves residue in the engine. Because the answer
is a function of world state, asking repeatedly and from more than one place is safe —
two callers get the same answer, the property `npc-map/src/perceive.rs` guarantees for
percepts. The grammar cache is the engine's, keyed on `(Deliberation, Within)` where
`Within.invokable` carries the address set (§11), never the API's.

---

# Part E — What this changes in the engine

## 14. What stays, what is new, what migrates

**Stays.** The body catalogue (`BODY_ACTS` in `npcd/src/engine/tools/mod.rs`). The
stencil machinery (`candle-conversation/src/stencil/`). The two clocks and turn loop
(`driver.rs`, `character_loop` in `npcd/src/engine/runtime/turn.rs`). Witness, stir,
narration. The one-lock world model. The per-character `ConversationLock`
(`npcd/src/engine/mind/mod.rs`), projection/window/prompt structure.

**New, and built.** `invoke` (§5). The `local` router + its external mount (§8). `OPTIONS`
schema responses (§5.1). The near-you index route (§6). The token auth and its persisted
registry (§8.3). The in-process fast path (§8.1). The per-address body stencil on
`invoke` (§11). Item ids on placed instances (§7). Routing of `plan_*`/`orders_*` to
the projected store so the plan stays in the system prompt (§9.2). The `/here` surface for
the world-state acts (Appendix G) and the operator `/reshape` route (Appendix F).

**New, and Not built.** The world-routes extension point (§9.1.1).

**Does not migrate.** The audit's "build 115 tools" was not turned into "delete the compiled
`Tool`s and declare routes". The station acts, the bench and file verbs and the world-state
half of `WORLD_ACTS` are still compiled `Tool` statics reached through the grammar, **and** are
routes reached through `invoke`: one handler (`body::perform`, with the handlers in
`engine/enact/` and `engine/work/`), two front doors (Appendix G.3). The handlers were
shipped before the routes, so the routes wrap them in an HTTP envelope at no cost to the
handlers; what the routes did not buy is a smaller grammar. **Replaced design:** the plan
to delete each `Tool`, `Choices` row and `Within` field as its act became a route (the
earlier §15 and Build order step 6) is not what was built. The genuinely-absent handlers
(`plan_*` as station acts, `roster_*`, `room_sit`, `character_write_beliefs`) and the
durability and projection gaps are D.6. **Appendix A** is the inventory, **Appendix C**
the route map, **Appendix D** the store map.

## 15. Availability and Choices serve both doors

- **Availability is the router's answer, and the grammar's.** A route not mounted at your
  standpoint 404s: `AtLift` and `AtLanding` are the `/lift/<id>` verbs where a landing is;
  `AtPart` is the part's routes within reach (`address_of` and `serves` in
  `npcd/src/effector/router.rs`, `station::resolve`); the speech availabilities
  (`PersonNearby`, `PersonAmongOthers`, `PersonInEarshot`) gate the body acts. The same
  `Availability` decides whether a compiled act is offered (`performable`, `specs_within`
  in `npcd/src/engine/tools/compile.rs`). The empty-set rule — an act whose required
  world-bound argument has nothing to name is absent — is `performable` and `live_empty`
  there, and the `invokable` address set reads it, so a route is offered exactly when the
  compiled act would be.
- **Choices are the schema's enums.** `Company`, `Floors`, `Reachable`, `Claimable`
  (`Choices` in `npcd/src/engine/tools/choices.rs`) are filled from world state each turn.
  The lift's `floor` is `world.floor_names()` (`npc-map/src/world/lifts.rs`) at that moment,
  in the grammar and in `/here`'s `OPTIONS` (Appendix G.2). The world enumerates; the
  schema carries; the stencil enforces.

**`Within` did not shrink, and §1's wall is held by other means.** The earlier plan was that
as acts became routes every sim-derived `Within` field (`operable`, `readable`, `claimable`,
`device_modes`, `hostiles`, `makeable`, `queues`, `threads`, …) would leave `Within`, and the
grammar cache key with them. **Replaced design.** The acts stayed compiled (§14), so those
fields stay, and `Within` gained `invokable` (the address set, §11), `at_lift`,
`lift_here`, `floors` and `away_from_home` (`Within` in
`npcd/src/engine/tools/within_tools.rs`). The wall is held by `ACTS_PER_TURN` = 2 and
`MAX_TURN_PATHS` (§1), by the per-address body being one shared shape per distinct act
(§11), and by the cache hitting because `Within` is a room fact shared across the cast.

## 16. Decisions made, and questions open

Made and defended above: two verbs, the model never picking the method, `OPTIONS` for
schema (§5, §5.1); one host `local`, ids in every URL, closed in fiction (§7); one
router, fast in-process path plus external socket mount (§8.1–8.2); one unified
`Principal`, token auth carried for the LLM, v1 = header+token match, external control
by **token scope** — as-NPC (proximity-gated) vs direct by-id (§8.3); `act`, `sleep`,
`promise`, `remind` stay body/human interactions, the phone acts and every
world/station verb become routes (§4); an embedder world **replaces** the vault, one
per daemon, under npcd's one lock (§9.1 — the decision stands, the trait is **Not
built**); fast JSON with the world's own sentence as the correction (§8.4, §12);
each invokable address brings its own body into the one turn grammar (§11); stateless
API, no per-character state in the engine (§11, §13).

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
- **Topology mutation** — its own cut, built: routes are interaction-dynamic over the
  world's mutable half (§7.1), and the walkable map is reshaped by the engine primitive
  `MapSet::apply` / `World::reshape` behind the direct-scope `/reshape/:world` route
  (Appendix F).

---

# Part F — Testing and build order

House conventions hold (`AGENTS.md`): TDD with raw expected values, one concern per
file, no stubs, no env-flags, file tools only. Three properties make this testable:

- **The `local` router is `oneshot`-testable on CPU**, no GPU, no server — the path
  the existing routers use, and the path production takes
  (`Runtime::effector_invoke` in `npcd/src/engine/runtime/device.rs`). Every route's
  happy path, every error, every per-standpoint mount/404, every `OPTIONS` schema is a
  request/response assertion (the tests of each `npcd/src/effector/` module and
  `npcd/tests/effector.rs`). The token check (§8.3) is pinned both ways (valid token
  passes; absent or mismatched fails; `X-Tokera-*` never read: `npcd/src/effector/auth.rs`).
- **Schema→stencil is CPU-testable with the stencil simulator** (`stencil_tree.md`
  §14): the address set and each address's body compiled to the `invoke` grammar
  (`npcd/src/engine/invoke_body.rs`), run across every field, asserting the emitted
  JSON parses and validates — the regression net that the device never emits an
  ill-formed call.
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
effector abstraction — drive the world by `invoke` (and `GET` for reads) against the router and
assert the response *and* the resulting state — which is both a migration check and
the start of an **extensive** suite: every route, every error, every
availability-by-condition transition (§7.1), every schema→stencil path (§11). The
abstraction is what makes the world broadly unit-testable: a room, a machine, a
mode change is now a request and an assertion.

**The missing world was mostly not missing.** Contrary to the audit's "0 of 79"
(`archived/tool_surface_audit.md` §1), the handlers are shipped in `engine/work/`
(Appendix D), and the routes wrap them in an envelope. What was genuinely missing is
scoped in Appendix D.6: durability for the `Sim.record`/`ledger` writes (character and
place writes are now durable; a snapshot persists the rest), the bridge from effector
handlers to the substrate agency plane so `plan_*`/`orders_*` project (§9.2; built), and
the absent handlers (`roster_*`, `room_sit`, `character_write_beliefs`, which stay
absent). Each landed with its tests.

The build order, dependency-first, each step green before the next, and where each ended
up:

1. The `local` router beside the three, with `GET http://local/` (near-you index) and the
   **device-only middleware** of §8.3 (token → body; ignores `X-Tokera-*`; *not*
   `guard::Api`); the external mount, the in-process fast path with the `Router` handle
   retained on `Runtime` (`set_effector_router`), and the no-reentry-under-lock
   invariant; `oneshot`-tested including the `X-Tokera-*` refusal both directions.
   **Built.** The mount is a code route nested at `/v1/local`
   (`npcd/src/tenant/routes_and_sinks.rs`), not a `sites:` entry (§8).
2. Item ids on placed instances, the per-item state record (§7.1), and
   standpoint-filtered mounting — with the availability model of §7.1 mapped out in full
   (every part, its conditions, the verbs each condition affords; Appendix C). **Built.**
3. `invoke` as the fixed-frame tool beside the body act `scan`; `OPTIONS` schema; the
   system-prompt copy (§6.1). **Built.**
4. The per-address body stencil: each `invoke` address pairs with its act's body in the
   `Within`-keyed grammar (§11); the `scan` survey naming each address (so the common call
   is one turn); the simulator suite. **Built**, without the `Within` shrink it was to ride
   on (§15).
5. The lift, end-to-end (`http://local/lift/<id>/{call,use}` and `GET .../lift/<id>`, enum
   from `world.floor_names()`, consequence loop tested). **Built.**
6. The station acts, the bench and file verbs and the world-state half of `WORLD_ACTS` as
   routes, over the handlers that already existed, with `plan_*`/`orders_*` routed to the
   projected store (§9.2). **Built as routes beside the compiled acts**, not as a
   replacement for them: no `Tool`, `Choices` row or `Within` field was deleted (§14).
   Appendix A tracks it.
7. The world-routes extension point (§9.1.1), with the vault as the first implementation
   and a second, trivial implementation as the test that it is truly attachable.
   **Not built.**
8. The `MapSet` mutation surface and topology-dynamic routes. **Built** (Appendix F).

---

# Part G — Sharp edges (from adversarial review), and where they are handled

The design survived an adversarial review against the code; the core (compiled body
acts + a fixed two-tool device over an in-process real router, per-resource
schema→stencil, attachable world) held. These are the edges it found, each with
where this document now handles it. "Blocker" = must be solved before or during
Step 1–2; "sharp" = scoped but real; "minor" = watch.

| # | Sev | Edge | Handled in |
|---|---|---|---|
| 1 | blocker | Focus must not enter the `(Deliberation, Within)` grammar-cache key, or the hit rate collapses and a whole-turn tree recompiles on the tick path | §11: one tree cached on `(Deliberation, Within)`; the address set is in `Within.invokable` and each address brings its body through `Param.shapes`, so there is no per-character focus and no second cache. **Replaced design:** the separate sub-stencil keyed by `(resource-id, schema-fingerprint)` and the `Within` shrink (§15) were not built |
| 2 | blocker | The device surface cannot reuse `guard::Api` — it hardcodes the `X-Tokera-*` role check; and the fast path sits behind a header-trusting gateway | §8.3 (device-only middleware; `X-Tokera-*` never read; token both-ways test) |
| 3 | blocker | Stable per-instance ids needed minting: `Placement` is anonymous, `part_ids_at` de-dups by catalogue id, world state keyed on place+name | §7, §9: id = `<part-id>~<ordinal>` (base 36) from `instances_at`; `part_ids_at` is kept for `within_reach`; the sim keys devices by instance id. **Replaced design:** `(place, part-id, ordinal)` and removing the de-dup |
| 4 | sharp | Sync/async seam: `within`/act path are sync inside an async loop; a router call is a `Future`, `block_on` panics | §8.1 (async slice or `spawn_blocking`) |
| 5 | sharp | Re-entering the non-reentrant world mutex from a handler that calls the router deadlocks silently | §8.1 (no-reentry-under-lock invariant, tested) |
| 6 | sharp | "Strong typing" only truly binds string enums and booleans; `integer`/`array`/`object` are shaped, not enforced | §11 (world validation is load-bearing, not a backstop) |
| 7 | sharp | A `{error,detail,field}` shape would need `err()` to grow a `field`, and path→field mapping is real work | §8.4, §12. **Replaced design:** `{error, detail}` with the world's own sentence as the correction; `field` only in `bad_json` |
| 8 | sharp | An enum can go stale between the turn it is seen and the turn `invoke` fires | §11 (`Within` is rebuilt every turn; stale fails benignly) |
| 9 | sharp | An embedder world with its own lock reintroduces the two-lock bug the design avoids | §9.1 (provider runs under `Hosted`'s one lock, like `Sim`) — the constraint on the **Not built** trait |
| 10 | minor | `Runtime::within` is non-atomic (about ten `hosted.read`/`hosted.sim` acquisitions in `engine/runtime/perceiving.rs`); the device adds reads | §13 note; fold into a `with`-scoped builder if it grows |
| 11 | minor | An external token dies across restart unless persisted; must not be derived from `body_id` | §8.3 (real secret; persist — Q6) |
| 12 | minor | The `oneshot` router was `#[cfg(test)]`; production kept no handle | §8.1: built — `Runtime` holds the router (`set_effector_router`) and calls it through `effector_invoke` |

The review's bottom line was to solve **#1 first**, then the device middleware (#2) and
the id minting (#3); everything else was sharp-but-scoped. #1 was solved by keying the
one tree on `Within.invokable` with per-address bodies (§11), not by shrinking `Within`
(§15).

# Appendix A — The migration inventory (for review)

The tools that **stay** (the body — always present, never a URL):

| Tool | Why it stays |
|---|---|
| `tell` `whisper` `shout` `ask` `gesture` | speech is a body act, gated by who is in earshot (§4), never by a station |
| `move_to` `follow` | movement is a body act |
| `reflect` | an internal act, not a world interface |

Placement (decided): `act`, `sleep`, `promise`, and `remind` stay **body acts** —
human interactions, never a URL. The phone acts (`message` `reply` `invite` `open_group`
`reach_out`) are `http://local/phone/...` routes; `send_image` is not routed (C.27).

The acts that are **routes**. Each row gives the audit's name and count for the namespace,
what was built, and where it is mounted. Every one of them is *also* still a compiled `Tool`
(§14): the route is a second door onto the same handler. Every verb is `POST
http://local/<ns>/<id>/<short verb>`; the short verb is the act's name with its `<ns>_`
prefix removed (Appendix C):

| Namespace | Audit n | Built | Mounted as |
|---|---|---|---|
| `lift` | 2 | `LIFT_CALL`, `LIFT_USE` | `/lift/<id>/{call,use}`, `GET /lift/<id>` |
| world-state acts | ~18 | 13 of the 28 `WORLD_ACTS` | `POST /here/<verb>` (Appendix G); `command_tower` at `tower/<id>`, `produce` at `stores/<id>`, `sign_off` at `phone/<thread>` |
| `chronicle` | 8 | 4: `add_entry`, `rewrite_page`, `retire_entry`, `settle_boundary` | `/chronicle/<id>/…` |
| `record` | 16 | 13: `accession`, `write_provenance`, `hand_on`, `appraise`, `write_reason`, `let_go`, `describe`, `arrange`, `cross_reference`, `leave_note`, `tidy_index`, `mend`, `mark_repair` | `/record/<id>/…` |
| `story` | 9 | 2: `draft`, `file` | `/story/<id>/…` |
| `portrait` | 9 | 5: `draw`, `prompt_read`, `prompt_edit`, `settle_likeness`, `file_plate` | `/portrait/<id>/…` |
| `library` | — | 2: `library_read`, `library_write` (not in the audit) | on `story-desk` and `character-terminal`, under `/story` and `/character`, verb names unstripped |
| `character` | 6 | 4: `write_identity`, `write_wants`, `write_memories`, `settle_relation` | `/character/<id>/…` |
| `place` | 6 | 3: `write_entry`, `write_local_history`, `settle_route` | `/place/<id>/…` |
| `map` | 5 | 3: `add_place`, `settle_border`, `remove_place` — cartography lore, not topology (F.4) | compiled acts only; the `map` namespace is not mounted |
| `orders` | 6 | 3 station acts: `set`, `hand_to`, `report_done` | `/orders/<id>/…` and `/plan/<id>/…`, the projected-store bridge (C.11, C.21), and `/command/<id>/…` for the order table |
| `enquiry` | 5 | 4: `take_question`, `answer_from_record`, `name_the_gap`, `raise_work` | `/enquiry/<id>/…` |
| `dispatch` | 4 | 1: `post_wake` | `/dispatch/<id>/post_wake` |
| `plant` | 3 | 2: `note_drift`, `raise_fault` | `/plant/<id>/…` |
| `stores` | 3 | 2: `put_back`, `take_out` | `/stores/<id>/…` |
| `structure` | 3 | 3 | `/structure/<id>/…` |
| `cast` | 2 | 1: `report_disagreement` | `/cast/<id>/report_disagreement` |
| `creator` | 1 | 1: `present` | `/creator/<id>/present` |
| `gather` | 2 | 1: `call` | `/gather/<id>/call` |
| `plan` | 5 | no station acts; `break_down`, `order`, `reorder`, `scope` and the read | `/plan/<id>/…` (the projected-store bridge) |
| `bench` | 11 | 11 (`bench_*`) | on the six editing stations |
| `file` | 5 | 5 (`file_*`, reused from `zend-tools`) | on the six editing stations; `file_delete` on `story-desk` only |
| `archive`, `time` | — | `merge_status`, `merge_publish`, `merge_sync`; `time_travel` (not in the audit) | `/archive/<id>/…`, `/time/<id>/travel` |
| `roster`, `room`, `trial`, `standard` | 2, 1, 3, 3 | none | not built |
| `plan_*` as station acts | — | none | the bridge routes stand in |

Counts in the audit column are the audit's; the built column is `STATION_ACTS` and `BENCH_ACTS`
read from the code. The 58 station acts are these rows less `bench` and `file`.

The audit's read verbs (`read_era`, `read_ledger`, `read_hung`, …) are not routes: reading is
the body's `read` and `scan` (the station headers in `engine/station.rs` record the collapse of
twenty-two `read_*` and six `take_*` acts into `read` and `claim`), served as `POST /here/read`
and `/here/scan`, and a resource's own `GET` returns its descriptor and live state.

**`plan` and `orders` write projected state, not world state (§9.2).** They are the
mission machinery: editable through the effector device, but their handlers write the
agency layer so the plan and the held orders are re-projected into the dynamic system
prompt each turn rather than perceived as events. Their routes read/write the
substrate authoring plane, not `World`/`Sim`.

**Appendix C is the route map of this table** — each namespace's routes, their gating,
their body, and the store each handler touches, read from the shipped vault map and the act
tables. It carries the `library`/craft acts (`LIBRARY_READ`/`WRITE`) that the audit lacks, as C.7.

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
| [`tool_surface_audit.md`](archived/tool_surface_audit.md) (archived) | the whole "build all 115 as compiled tools" plan (§1, §6, §6.4); the `namespace_verb` naming/tokenisation analysis (§5A) is moot for URL path segments; the counts (§6.3); and its "Implemented: 0" claim is itself stale — the handlers ship in `engine/work/` (Appendix D) | §2 (what exists), §3 (removals), §4 (the 8 missing clusters), the craft-word naming rule for verbs |
| [`tool_world_acts.md`](tool_world_acts.md) | world-acting verbs as compiled body tools; the string/boolean workaround (§0) — lifted by per-resource schema (§11); the "seven new live sets" (§4) become handler-computed `OPTIONS` schema | the intent-vs-typed field rule; §6 "to simulate this in the vault" |
| `tool_decisions.md` (no longer in `docs/`) | "vocabulary moved onto the argument axis" — it moves onto the **route axis**; `read`/`claim`/`release` as body tools (§2, §3) reopen (read → the body's `read` and `scan`, `POST /here/read`) | §1 belief-write prohibition (reinforced, §9.2); tool-shape/argument-binding sections |
| `tool_interaction_reconciliation.md` (no longer in `docs/`) | "25 `*_read_*` tools vs one `observe`" (§4.2) — reads are now the body's `read` and `scan`; the 25-reads open question dissolves | interaction modes (§1, §3), interlocutor resolution; note phone acts' placement is Q5 |
| [`tool_world_state.md`](archived/tool_world_state.md) (archived) | the "129 tools" framing; `bench_`/`file_` as tools | most of it — the four stores and the namespace→store table become the **route-handler** spec (§9) |

**Tier 2 — engine & wire contract (section rewrites + additions).**

- [`npc_engine_design.md`](npc_engine_design.md) Part VII — "Availability is a
  product of location" keeps its conclusion (a not-mounted route 404s) but its
  mechanism becomes routing (§15); the `trait NpcTool` / `ToolRegistry::register`
  extension story is re-scoped to the attachable world-routes provider (§9.1).
  "Tools carry intent, not output" stays.
- [`npc_api_gui_design.md`](archived/npc_api_gui_design.md) (archived) — `/v1/tools` and `ToolInfo`
  (§9, §10) now describe only the compiled frame (body + `invoke`); add a
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
- [`maker_repertoire.md`](archived/maker_repertoire.md) (archived) — invariant 8's "reach and refuse"
  gloss and the dotted identifiers are stale; the 480 tasks, the currencies-as-types,
  and the invariants remain the spec for what routes must afford.
- [`asynchronous_mind_hierarchy.md`](asynchronous_mind_hierarchy.md) — "tools fixed
  at the boundary" still holds; the bottom-level fixed catalogue is now the effector
  device, and §5's git work-surface is now `.../bench/<id>/...` and `.../file/<id>/...`
  routes. Core argument unaffected; reconcile §1.1 and §5.

**Tier 4 — index & cross-reference.**

- `docs/README.md` — lists this document (the fourth NPC surface; there is no
  `local_api.md`) and `tool_world_acts.md`.
- [`web_gateway_design.md`](web_gateway_design.md) — does not yet mention that npcd
  registers a `local` effector router plus the `/v1/local/...` external mount; the note
  belongs there.

**Agrees, no change (a reader might expect otherwise):**
[`stencil_tree.md`](stencil_tree.md) is the reused machinery §11 depends on (the
string/boolean limit lives in `engine/tools/`, not here); `reflect`
([`reflection_and_dreams.md`](reflection_and_dreams.md)) is preserved as a body act;
[`narrative_engine.md`](narrative_engine.md) is unchanged (device acts return JSON
receipts, body acts still carry intent); [`npc_mind_design.md`](npc_mind_design.md)'s
belief-write protection is reinforced (§9.2), not contradicted.

One decision touches several docs: `act`, `sleep`, `promise`, and `remind` stay body
acts (human interactions), while the phone/messaging acts become `http://local/phone`
routes (§4). `tool_world_acts.md` and Appendix
A must be made to agree with that split.

---

# Appendix C — The vault route map

*The concrete route map the design references at §7.1 ("**Appendix C is that map** for the
vault"). It turns Appendix A's checklist into an executable specification: every part that
becomes routes, the routes each affords, what gates them, the body schema each `invoke`
takes, and the store the handler reads or writes. It is grounded in the shipped vault
(`npc-map/maps/*.yaml`), the shipped act tables (`npcd/src/engine/acts/`, `npcd/src/engine/station.rs`), the
availability/choices machinery (`npcd/src/engine/tools/`), and the current handlers
(`npcd/src/engine/enact/`, `npcd/src/engine/work/`). Where a namespace is authored in the
audit (`archived/tool_surface_audit.md`) but has no act in the code, the row cites the audit and
is flagged **[no handler yet]**: the verb was not built as a verb of its own.*

Conventions used in every table below:

- **Where an act lives** — the constant in the From column (`LIFT_CALL`, `CHRONICLE_ADD_ENTRY`,
  …) is declared in `engine/station.rs` for a station act, in `engine/bench.rs` for the
  `bench_*` and `file_*` verbs, and in `engine/acts/` for the world acts; its handler is in
  `engine/work/` (station and bench acts) or `engine/enact/` (world acts). Dispatch is
  `work::perform` (`engine/work/dispatch.rs`) and `enact::perform` (`engine/enact/mod.rs`).
- **Method** — every verb is `POST`, issued by `invoke`; the model never picks the method (§5).
  The only `GET`s are a resource's descriptor, `GET /<ns>/<id>` → `{id, part, name, verbs,
  invoke, state}` (`state` is `Sim::reading`: a machine's mode, and for the tower and the
  fabricators the stockpile, what can be made and the queues), and `OPTIONS /<ns>/<id>`, which
  returns the verbs' body schemas (§5.1). A `GET` row below is that descriptor, never a content
  read: era text, the ledger and the holds are read with the body's `read` and `scan`
  (`POST /here/read`, `/here/scan`, Appendix G), which is why **[no handler yet]** marks the
  audit's `read_*` and `take_*` verbs. RPC-flavoured one-path-per-verb, per the settled decision
  (§16 "Route paths"); the verb in the path is the act's name with its `<ns>_` prefix removed
  (`short_verb` in `effector/station.rs`), or the whole name where the act does not begin with
  it (`library_read`, `merge_status`, `orders_report_done`).
- **Condition** beyond reachability (§7.1): `always` (reading affords it once within reach),
  `claimed` (requires holding the part's `binds` — its working mode; on the six editing
  stations entered by `bench_branch`), `paired` (a settling table: needs the other holder present,
  the `Nearby`-class two-body gate), `affordable` (the live enum is non-empty — the economy is
  enforced in the schema). The act's own handler enforces the condition and refuses in the world's
  words, mapped to `409` (§12).
- **Body schema** field tags: `enum(src)` = grammar-enforced closed set filled from the named
  live-set when the turn's grammar is built (and, for `/here`, at `OPTIONS` time — §15, `LIVE` /
  `FIXED` in `engine/tools/choices.rs`); `typed(kind)` = integer/number/array/object —
  **shaped, not enforced**, so the handler validates (§11); `free` = open string. `req`/`opt` =
  required/optional. The station routes' `OPTIONS` schemas carry the raw declared types only (§5.1).
- **Store** — the column below records the *intended* target and is **superseded by
  Appendix D.5**, which maps each namespace to the store the shipped handler actually
  touches: almost every authoring row marked `World` is really `Sim` (its `record`/
  `ledger`), and a subset also reaches the **mind folder** (the authored lore) through
  the `bench` working copy (§D.4). `projected` = the substrate authoring plane (§9.2), which
  the `plan_*`/`orders_*` bridge routes write (`effector/plan.rs`, D.6 #2) and no station act
  handler does. The dispatcher for the station handlers is `engine/work/dispatch.rs`
  (`work::perform`), not `engine/station.rs` (declaration only).
- **id pattern** — `<part-id>~<ordinal>` (§7). The ordinal is 0-based, world-wide and written in
  base 36 (across the whole map in a deterministic walk), so a singleton is `<part>~0` and
  same-part placements take consecutive numbers without the place in the string
  (`order-table~0`, `fabricator~7`). The `<id>` placeholder and the `ns/<node>-<n>` shorthand in
  the tables below name which placement is meant; the address is always
  `http://local/<ns>/<part-id>~<ordinal>`, e.g. `http://local/record/accession-desk~0`.

---

## C.0 The availability matrix — which parts stand where in the vault

From the eight vault level maps in `npc-map/maps/` (command, chronicle, casting, story,
portraits, cartography, `vault-time` and `vault-archive`, plus `creators-vault.yaml`).
"Ordinal?" = does any node place more than one, so a route id must carry the ordinal to be
unique (§9). Parts marked **(war world)** carry world-acts but are placed only in
`tower-redoubt.yaml`, never in the vault — listed because Appendix A routes their verbs.

*(As-built: the "id pattern" column below is shorthand for where a part stands. A route's
address is `http://local/<ns>/<instance-id>`, where `<instance-id>` is `<part-id>~<ordinal>`
(§9) — e.g. the first chronicle terminal is `http://local/chronicle/chronicle-terminal~0`. The
`<node>-<n>` here just names which node and which placement.)*

| Part | kind | binds | modes | Vault level → node(s) | Ordinal? | id pattern (ns/…) |
|---|---|---|---|---|---|---|
| chronicle-terminal | station | one era | reading/working/offered | chronicle → early-range ×6, middle-range ×6, late-range ×4, catalogue-room ×2 | **yes** | `chronicle/<node>-<n>` |
| character-terminal | station | one character | reading/working/offered | casting → band-one ×6, band-two ×6, band-three ×4 | **yes** | `character/<node>-<n>` |
| story-desk | station | one gap | reading/working/offered | story → first-room ×6, second-room ×6, third-room ×4; vault-archive → merge-room ×1 | **yes** | `story/<node>-<n>` |
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
| order-table | fixture | — | — | command → command-room ×1, tower-redoubt muster-hall ×1 | no | `command/command-room-1` (`namespace_of` mounts it at `command`, not `orders`; see §C.11); the Redoubt's is `order-table~0`, the vault's `order-table~1` |
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
| light-ring | fixture | — | steady/failing | the first work room of each of the five working levels, command → command-room, tower-redoubt → muster-hall; vault-archive → merge-room; vault-time → first-time-room | no | (no route; operated through `operate`) |
| breaker-panel | fixture | — | closed/tripped | the second work room of each of the five working levels, command → plant, tower-redoubt → foundry | no | (no route; operated through `operate`) |
| coolant-valve | fixture | — | tight/weeping | beside every breaker-panel | no | (no route; operated through `operate`) |
| pressure-door | fixture | — | seated/hissing | the social room of each working level, command → enquiry, tower-redoubt → gatehouse | no | (no route; operated through `operate`) |
| air-handler | fixture | — | — | the first work room of each of the five working levels, command → plant, tower-redoubt → foundry; vault-time → second-time-room | no | (read only) |
| compute-rack | fixture | — | — | the store or watch room of each working level, command → dispatch, tower-redoubt → bridge | no | (read only) |
| status-board | fixture | — | — | beside every vault compute-rack, and tower-redoubt → muster-hall | no | (a posting; `read`) |
| seat | seat | — | — | every social/work node (counts 2–16) | n/a | `room/<node>-<n>` |
| muster-board **(war world)** | fixture | — | — | tower-redoubt → muster-hall ×1 | no | `orders/muster-1` |
| merge-table | station | one operation | — | vault-archive → merge-room ×1 | no | `archive/merge-table~0` (C.29) |
| time-machine | station | one year | — | vault-time → first-time-room ×8, second-time-room ×8 | **yes** | `time/time-machine~<n>` (C.29) |
| blast-door **(war world)** | fixture | — | open/closed | tower-redoubt → gatehouse | no | (no route; operated through `operate`) |
| wall-turret **(war world)** | fixture | — | hold fire/free fire | tower-redoubt → rampart ×4 | yes | (no route; operated through `operate`) |
| carrier **(war world)** | fixture | — | parked/driving | tower-redoubt → motor-pool | no | (no route; operated through `operate`) |
| bridge-console **(war world)** | station | the tower | — | tower-redoubt → bridge ×1 | no | `tower/bridge-1` |
| fabricator **(war world)** | station | one run | idle/running/paused/purging | tower-redoubt → fab-bay ×8 | yes | `stores/fab-<n>` |
| sensor-panel **(war world)** | fixture | — | — | tower-redoubt → bridge ×1 | no | (scan target) |

**New parts (audit §6.2) in no map** — planning board (`plan_*`), trials shelf
(`trial_*`), standards board (`standard_*`) — **Not built**: no map places them and no act
attaches to them. The audit intended them "placed in rooms that already exist" (§6.2), the
standards board reachable from every level (§3.4), the planning board where planning happens,
and the trials shelf where experiments are kept; C.21–C.23 say what exists instead.

**Standpoint → mount.** A route's routes are mounted wherever its part is within reach
(`within_reach`, `npc-map/src/perceive.rs`); a not-mounted route 404s (§15). Because the vault is one
building with a single lift shaft (`creators-vault.yaml` portals, all `kind: a lift`), the lift
routes (C.1) mount at every level's `core` node.

---

## C.1 The lift — `http://local/lift/<id>`

id pattern: one shaft, id `lift/command-shaft` (the vault has a single car serving all levels,
`vault-command.yaml` core "one car serves every level"). Mounted at the `core` node of each
level.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../lift/<id>` | GET | any standpoint; off a landing `here` is false (`status` in `effector/lift.rs`) | — (returns `{here, level, moving}`) | World | `effector/lift.rs` (status) |
| `POST .../lift/<id>/call` | POST | at landing **and car elsewhere** (`Availability::AtLift`: `at_lift && !lift_here`, `engine/tools/compile.rs`) | — | World | `LIFT_CALL`; enact `lift_call` (`engine/enact/movement.rs`) |
| `POST .../lift/<id>/use` | POST | at a landing (`Availability::AtLanding`); with the car elsewhere it is called and the body waits | `floor` enum(`world.floor_names()`, `npc-map/src/world/lifts.rs`) req | World | `LIFT_USE`; enact `lift_use` (`engine/enact/movement.rs`) |

---

## C.2 The world-state half of `WORLD_ACTS` — `/here`, and three part-bound acts

These are the sixteen Appendix A names ("world-state acts | ~18"); `WORLD_ACTS` (`engine/acts/mod.rs`)
holds 28 acts in all, the rest being the body and phone acts of Appendix A and `reply` and `focus`,
which Appendix C never listed. Thirteen of the sixteen are **name-addressed** and mounted on the one
personal surface `/here` (Appendix G): `POST /here/<verb>`, the target a field in the body drawn from the
live `Choices` set, never an instance id. The other three are `AtPart` acts mounted by the station
mechanism where their fixture stands. `read` and `scan` are `POST` verbs, not `GET`s: both run through
`body::perform`, and `read` moves the reader's read-cursor (G.2). Handler dispatch table: `perform_said`
in `engine/enact/mod.rs`, whose `is_mine` lists the same acts.

| Route | Condition | Body schema | Store | From |
|---|---|---|---|---|
| `POST /here/read` | `Always` (`Readable` non-empty) | `what` enum(`Readable`) | Sim (`postings`/`ledger`) | `READ` (`engine/acts/carrying.rs`); `read` (`engine/enact/reading.rs`) |
| `POST /here/scan` | `Always`; `at` empty surveys | `at` enum(`Reachable`) opt | Sim (`field`) | `SCAN` (`engine/acts/combat_and_machines.rs`); `scan` (`engine/enact/scan.rs`) |
| `POST /here/claim` | `Always` (`Claimable`) | `what` enum(`Claimable`) req | World / Sim `ledger` | `CLAIM` (`engine/acts/carrying.rs`); `claim` (`engine/enact/notices.rs`) |
| `POST /here/release` | `Holding`: only while holding an order or a station | — | World | `RELEASE`; `release` (`engine/enact/carry.rs`) |
| `POST /here/post_notice` | `Always` (`Postable`) | `on` enum(`Postable`) req; `what` free req | Sim `postings` | `POST_NOTICE`; `post_notice` (`engine/enact/notices.rs`) |
| `POST /here/record_verdict` | `Always` | `on` req; `judgement` free req; `what_would_change_it` free opt | World | `RECORD_VERDICT` (`engine/acts/tower_and_work.rs`); `record_verdict` (`engine/enact/command.rs`) |
| `POST /here/give` | `Nearby` (someone here) | `what` enum(`Carried`) req; `to` enum(`Company`) req; `count` opt | Sim (inventory) | `GIVE`; `give` (`engine/enact/carry.rs`) |
| `POST /here/equip` | `Embodied` | `what` enum(`Equippable`) req | Sim | `EQUIP`; `equip` (`engine/enact/carry.rs`) |
| `POST /here/use` | `Embodied` | `what` enum(`Usable`) req; `on` enum(`Company`) opt | Sim | `USE`; `use_it` (`engine/enact/carry.rs`) |
| `POST /here/gather` (field) | `Embodied` (`Extractable`) | `what` enum(`Extractable`) req | Sim `field` | `GATHER` (field); `gather` (`engine/enact/carry.rs`) — **distinct from `gather_*` (C.24)** |
| `POST /here/operate` | `Always` (`Operable`) | the thing, and `mode` enum(`DeviceModes` — dependent on the chosen thing) | World / Sim | `OPERATE`; `operate` (`engine/enact/engage.rs`) |
| `POST /here/recall` | `AwayFromHome` (`engine/tools/types.rs`) | — | World | `RECALL`; `recall` (`engine/enact/reading.rs`) |
| `POST /here/engage` **(war world)** | `Embodied` + `Postures` non-empty | `posture` enum(`Postures`) req; `target` enum(`Hostiles`) opt; `priority` enum(FIXED `sim::field::PRIORITIES`) opt | Sim `field` | `ENGAGE`; `engage` (`engine/enact/engage.rs`) |
| `POST /tower/bridge-console~0/command_tower` **(war world)** | `AtPart` at the bridge console; `affordable` (`TowerActions`) | `action` enum(`TowerActions`) req; `target` free opt; `x`, `y`, `depth` `integer` opt | Sim (tower) | `COMMAND_TOWER`; `command_tower` (`engine/enact/command.rs`) |
| `POST /stores/fabricator~<n>/produce` **(war world)** | `AtPart` at a fabricator; `affordable` (`Makeable`) | `what` enum(`Makeable`) req; `count` opt; `queue` enum(`Queues`) opt | Sim | `PRODUCE`; `produce` (`engine/enact/command.rs`) |
| `POST /phone/<thread>/sign_off` | on a leavable thread (`Leavable`) | `intent` free req | Sim phone / **projected** | `SIGN_OFF`; `sign_off` (`engine/enact/messaging.rs`) — **see C.28.2** |

`act`, `sleep`, `promise`, `remind` stay **body acts** and gain no route (§4, App A; `ACT` in
`engine/acts/mod.rs`, `SLEEP` in `engine/acts/carrying.rs`, `PROMISE` and `REMIND` in
`engine/acts/tower_and_work.rs`).

---

## C.3 `chronicle` — `http://local/chronicle/<id>`

Part: chronicle-terminal (station, `binds: one era`, modes reading/working/offered) — plus the
concordance-table (`chronicle_settle_boundary`) and read-only fixtures archive + timeline-wall.
id: `chronicle/<node>-<n>` (ordinal required — up to 6 per range).

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../chronicle/<id>` (read_era / read_any_page) | GET | always (reading mode) | — | World | `chronicle_read_era`,`read_any_page` audit §2.2 **[no handler yet]** |
| `POST .../chronicle/<id>/add_entry` | POST | claimed (working) | `to` free req; `what` free req | World | `CHRONICLE_ADD_ENTRY` |
| `POST .../chronicle/<id>/rewrite_page` | POST | claimed (working) | `in` free req; `what` free req | World | `CHRONICLE_REWRITE_PAGE` |
| `POST .../chronicle/<id>/retire_entry` | POST | claimed (working) | `what` free req | World | `CHRONICLE_RETIRE_ENTRY` |
| `POST .../chronicle/concordance-1/settle_boundary` | POST | **paired** (both era holders present) | `between` free req; `and` free req | World | `CHRONICLE_SETTLE_BOUNDARY` |
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
| `POST .../record/receiving-1/accession` | POST | claimed | `what` free req; `from` free req | World | `RECORD_ACCESSION` |
| `POST .../record/receiving-1/write_provenance` | POST | claimed | `of` free req; `from` free req | World | `RECORD_WRITE_PROVENANCE` |
| `POST .../record/receiving-1/hand_on` | POST | claimed | `what` free req; `to` free req | World | `RECORD_HAND_ON` |
| `POST .../record/sorting-1/appraise` | POST | claimed | `what` free req; `verdict` free req | World | `RECORD_APPRAISE` |
| `POST .../record/sorting-1/write_reason` | POST | claimed | `about` free req; `why` free req | World | `RECORD_WRITE_REASON` |
| `POST .../record/sorting-1/let_go` | POST | **claimed** (working mode, entered deliberately: it destroys, so it takes the claim rather than being offered to anyone passing) | `what` free req; `because` free req | World | `RECORD_LET_GO` |
| `POST .../record/catalogue-1/describe` | POST | claimed | `what` free req; `how` free req | World | `RECORD_DESCRIBE` |
| `POST .../record/catalogue-1/arrange` | POST | claimed | `what` free req; `under` free req | World | `RECORD_ARRANGE` |
| `POST .../record/catalogue-1/cross_reference` | POST | claimed | `from` free req; `to` free req | World | `RECORD_CROSS_REFERENCE` |
| `POST .../record/catalogue-1/leave_note` | POST | claimed | `on` free req; `what` free req | World | `RECORD_LEAVE_NOTE` |
| `POST .../record/catalogue-1/tidy_index` | POST | claimed | `what` free req | World | `RECORD_TIDY_INDEX` |
| `POST .../record/mending-1/mend` | POST | claimed | `what` free req; `how` free req | World | `RECORD_MEND` |
| `POST .../record/mending-1/mark_repair` | POST | claimed | `what` free req | World | `RECORD_MARK_REPAIR` |

---

## C.5 `story` — `http://local/story/<id>`

Parts: story-desk (station, `binds: one gap`, modes) ×16; ledger-room fixtures gap-ledger +
filed-stories (reads). id: `story/<node>-<n>` (ordinal required for desks).

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../story/ledger-1` (read_ledger / read_around_gap) | GET | always | — | World | audit §2.2 **[no handler yet]** |
| `GET .../story/ledger-filed` (read_filed) | GET | always | — | World | filed-stories.yaml; audit §2.2 **[no handler yet]** |
| `POST .../story/ledger-1/take_next_silence` | POST | always (`Claimable`) | `what` free req | World / Sim `ledger` | audit §2.2 (→ `claim`, C.2) **[no handler yet]** |
| `POST .../story/<id>/draft` | POST | claimed (working) | `for` free req; `what` free req | World | `STORY_DRAFT` |
| `POST .../story/<id>/file` | POST | claimed | `what` free req | World | `STORY_FILE` |
| `GET .../story/<id>/read_aloud` (at reading table) | GET | always | — | World | audit §2.2 → see `gather` (C.24) **[no handler yet]** |

---

## C.6 `portrait` — `http://local/portrait/<id>`

Parts: easel (station, `binds: one character`, modes) ×16; likeness-table (`settle_likeness`,
paired); plate-rack (`file_plate`); hung-faces + house-palette (reads — house-palette's read is
re-homed to `standard`, audit §3.4, C.22). id: `portrait/<node>-<n>` for easels (ordinal req).

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../portrait/gallery-1` (read_hung) | GET | always | — | World | hung-faces.yaml; audit §2.2 **[no handler yet]** |
| `GET .../portrait/<id>/prompt_read` | GET | always | `of` enum(character id) req | World | `PORTRAIT_PROMPT_READ` (a `POST`: every verb is) |
| `POST .../portrait/<id>/draw` | POST | claimed (working) | `of` free req; `carrying` free req | World | `PORTRAIT_DRAW` |
| `POST .../portrait/<id>/prompt_edit` (redraw) | POST | claimed | `of` free req; `carrying` free req | World | `PORTRAIT_PROMPT_EDIT` |
| `POST .../portrait/likeness-1/settle_likeness` | POST | **paired** (drawer + person-holder) | `of` free req; `with` free req | World | `PORTRAIT_SETTLE_LIKENESS` |
| `POST .../portrait/plate-1/file_plate` | POST | always (holding the plate) | `what` free req | World | `PORTRAIT_FILE_PLATE` |
| `POST .../portrait/plate-1/take_faceless` | POST | always (`Claimable`) | `what` free req | World / Sim | audit §2.2 (→ `claim`) **[no handler yet]** |

---

## C.7 `library` (craft) — on the story desk and the character terminal

**A real `STATION_ACTS` pair not listed in the audit but shipped in code** — the craft
libraries (mood / response), attached to the story-desk and character-terminal (`CRAFT` in
`engine/station.rs`). A write here changes how *every* character in the world feels or answers.
There is no `library` namespace: `namespace_of` has no mapping for it, so the acts mount on the
station being worked under that station's own prefix, with their full names as the verbs
(`library_read`, `library_write`: the act does not begin with its namespace).

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `POST .../story/story-desk~<n>/library_read` (and under `character/`) | POST | always | `kind` enum(`mood`,`response`) req; `id` free req | mind folder via `Sim.bench` | `LIBRARY_READ` |
| `POST .../story/story-desk~<n>/library_write` (and under `character/`) | POST | claimed (working) | `kind` enum(`mood`,`response`) req; `id` free req; `field` enum(`description`,`template`) req; `text` free req | mind folder via `Sim.bench` | `LIBRARY_WRITE` |

---

## C.8 `character` — `http://local/character/<id>`

Parts: character-terminal (station, `binds: one character`, modes) ×16; relations-table
(`settle_relation`, paired). id: `character/<node>-<n>` (ordinal req).

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../character/<id>` (read) | GET | always | — | World | `character_read` audit §2.2 (→ `scan`) **[no handler yet]** |
| `POST .../character/<id>/write_identity` | POST | claimed (working) | `of` free req; `what` free req | World | `CHARACTER_WRITE_IDENTITY` |
| `POST .../character/<id>/write_wants` | POST | claimed | `of` free req; `what` free req | World | `CHARACTER_WRITE_WANTS` |
| `POST .../character/<id>/write_memories` | POST | claimed | `of` free req; `what` free req | World | `CHARACTER_WRITE_MEMORIES` |
| `POST .../character/relations-1/settle_relation` | POST | **paired** (both character holders) | `between` free req; `and` free req | World | `CHARACTER_SETTLE_RELATION` |

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
| `POST .../place/<id>/write_entry` | POST | claimed (working) | `of` free req; `what` free req | World | `PLACE_WRITE_ENTRY` |
| `POST .../place/<id>/write_local_history` | POST | claimed | `of` free req; `what` free req | World | `PLACE_WRITE_LOCAL_HISTORY` |
| `POST .../place/road-1/settle_route` | POST | **paired** (both end-holders) | `from` free req; `to` free req | World | `PLACE_SETTLE_ROUTE` |

---

## C.10 `map` — the cartography acts — **compiled acts only; the `map` namespace is not mounted**

Part: map-table (station, `binds: the whole geography`, modes) ×1; gallery-rail (read-only
overlook). The audit expected these verbs to mutate the walkable topology. They do not: they
author cartography *lore* (F.4), a `Kind::Place` item in `Sim.record`, and the walkable map is
reshaped by the separate `/reshape` primitive. `map` is held back from the generic station
mount (`station_namespaces` in `effector/router.rs`), so none of these is routable; the table
lists the acts the catalogue carries, as compiled `Tool`s at the map-table.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../map/map-room-1` (read) | GET | always | — | World | `map_read` audit §2.2 **[no handler yet]** |
| `GET .../map/gallery-rail-1` (read) | GET | always | — | World | gallery-rail.yaml **[no handler yet]** |
| (not mounted) `add_place` | POST | claimed (working) | `called` free req; `where` free req | `Sim.record` (lore) | `MAP_ADD_PLACE` |
| (not mounted) `settle_border` (move_border) | POST | **paired** | `between` free req; `and` free req | `Sim.record` + `Sim.ledger` | `MAP_SETTLE_BORDER` |
| (not mounted) `remove_place` (drown_place) | POST | claimed | `what` free req | `Sim.record` (lore) | `MAP_REMOVE_PLACE` |
| (not mounted) `redraw_coast` | POST | claimed | `what` free req | — | `map_redraw_coast` audit §2.2 **[no handler yet]** |

---

## C.11 `orders` — `http://local/orders/<id>` — **PROJECTED store (§9.2)**

Part: muster-board (fixture, tower-redoubt — the `set`/`hand_to` end). Per §9.2 / App A,
`orders_*` write the **agency layer** and re-project into the system prompt, not the world.

**The projected-store routes are not keyed by a part instance.** `/orders/<id>` and `/plan/<id>`
are the one sub-router `effector/plan.rs`, nested under both prefixes and excluded from the
generic station nest, so they never reach `Sim.ledger`. The `<id>` is a **strategy id** — the
character's own goal keyed by name, so a second `scope` of the same id revises it — not a map
instance, and there is no reach check: a character's plan is its own (§7.2). The verbs are
`set`, `take`, `hand_to`, `report_done`, `give_back`, `scope`, `order`, `reorder` and
`break_down` on either prefix, each an async write of the agency layer through
`Npcs::put_strategy_self` (`report_done` moves the strategy to `finished`, `give_back` to
`abandoned`, and a verb naming a strategy the character never had is refused); `GET /<ns>/<id>`
returns the whole agency layer.

**order-table routes under `command`, not `orders`.** The plan to keep both parts on one `orders`
prefix was superseded: the command desk closes a mission against `Sim.ledger`, which is a
different store and a different concern from the projected-agency verbs — so `namespace_of`
mounts the order-table's acts at `command/<id>` (`npcd/src/effector/namespace.rs`). `command`
is otherwise a plain generic station namespace like any other (§9), carrying seven verbs:
`report_done`, `report_stuck`, `report_rejected` (the mission acts of `engine/mission_acts.rs`,
`report_done` taking `found`), `order_post`, `order_results`, `mission_write`, and
`orders_report_done` (the station act, taking `what`).

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../command/command-room-1` (descriptor) | GET | always | — | World (generic station read) | `state` in `effector/station.rs` |
| `POST .../command/command-room-1/orders_report_done` | POST | holding an order | `what` free req | Sim `ledger` | `ORDERS_REPORT_DONE` (`engine/work/dispatch.rs`) |
| `POST .../command/command-room-1/report_done` | POST | the mission's work is in | `found` free req | Sim `ledger` / `missions` | `REPORT_DONE` (`engine/mission_acts.rs`; handler `mission` in `engine/work/mission_acts.rs`) |
| `POST .../orders/<strategy id>/set` | POST | always | `what` free req; `for` free opt | projected (agency) | `ORDERS_SET`; `effector/plan.rs` |
| `POST .../orders/<strategy id>/hand_to` | POST | always | `what` free req; `to` free req | projected (agency) | `ORDERS_HAND_TO`; `effector/plan.rs` |

---

## C.12 `enquiry` — `http://local/enquiry/<id>`

Part: enquiry-desk (station, `binds: an open enquiry`, command → enquiry). id `enquiry/enquiry-1`.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../enquiry/enquiry-1/read_history` | GET | always | — | World | `enquiry_read_history` audit §6.2 **[no handler yet]** |
| `POST .../enquiry/enquiry-1/take_question` | POST | always (claims the enquiry) | `what` free req | World | `ENQUIRY_TAKE_QUESTION` |
| `POST .../enquiry/enquiry-1/answer_from_record` | POST | holding the enquiry | `what` free req; `answer` free req | World | `ENQUIRY_ANSWER_FROM_RECORD` |
| `POST .../enquiry/enquiry-1/name_the_gap` | POST | holding the enquiry | `what` free req; `missing` free req | World | `ENQUIRY_NAME_THE_GAP` |
| `POST .../enquiry/enquiry-1/raise_work` | POST | holding the enquiry | `from` free req; `work` free req | projected + Sim `ledger` (produces an order, audit §3.2) | `ENQUIRY_RAISE_WORK` |

---

## C.13 `dispatch` — `http://local/dispatch/<id>`

Part: dispatch-board (fixture, command → dispatch). Reads the whole building; `post_wake` is the
change-and-its-wake producer. id `dispatch/dispatch-1`.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../dispatch/dispatch-1` (read) | GET | always | — | World | dispatch-board.yaml; audit §2.2 **[no handler yet]** |
| `GET .../dispatch/dispatch-1/read_holds` | GET | always | — | World | `dispatch_read_holds` audit §6.2 **[no handler yet]** |
| `GET .../dispatch/dispatch-1/read_wake` | GET | always | — | World | `dispatch_read_wake` audit §6.2 **[no handler yet]** |
| `POST .../dispatch/dispatch-1/post_wake` | POST | always | `from` free req; `breaks` free req | World | `DISPATCH_POST_WAKE` |

---

## C.14 `plant` — `http://local/plant/<id>`

Part: plant-panel (fixture, command → plant). Read, not driven. id `plant/plant-1`.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../plant/plant-1/read_panel` | GET | always | — | World | plant-panel.yaml; audit §2.2 **[no handler yet]** |
| `POST .../plant/plant-1/note_drift` | POST | always | `what` free req; `drift` free req | World | `PLANT_NOTE_DRIFT` |
| `POST .../plant/plant-1/raise_fault` | POST | always | `what` free req; `why` free req | World | `PLANT_RAISE_FAULT` |

---

## C.15 `stores` — `http://local/stores/<id>`

Part: stores (fixture, command → receiving). id `stores/receiving-1`.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../stores/receiving-1/walk_the_racks` | GET | always | — | Sim | stores.yaml; audit §2.2 **[no handler yet]** |
| `POST .../stores/receiving-1/put_back` | POST | always | `what` free req | Sim | `STORES_PUT_BACK` |
| `POST .../stores/receiving-1/take_out` | POST | always | `what` free req | Sim | `STORES_TAKE_OUT` |

---

## C.16 `structure` — `http://local/structure/<id>`

Part: structure-board (fixture, story → board-room). id `structure/board-1`.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `POST .../structure/board-1/lay_out_scenes` | POST | always | `what` free req | World | `STRUCTURE_LAY_OUT_SCENES` |
| `POST .../structure/board-1/test_the_want` | POST | always | `in` free req; `scene` free req | World | `STRUCTURE_TEST_THE_WANT` |
| `POST .../structure/board-1/find_the_slack` | POST | always | `what` free req | World | `STRUCTURE_FIND_THE_SLACK` |

---

## C.17 `roster` — `http://local/roster/<id>`

Part: roster (fixture, casting → roster-room). id `roster/roster-1`. **Not built:** no act attaches to
the roster, so the `roster` namespace is not mounted (`namespace_of("roster")` exists and nothing uses
it); `take_unheld` collapsed into `claim` (`/here/claim`).

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
| `POST .../cast/watch-1/report_disagreement` | POST | always | `about` free req; `what` free req | World | `CAST_REPORT_DISAGREEMENT` |

---

## C.19 `room` — `http://local/room/<id>`

Part: seat (kind `seat`, every social/work node). `room.talk` **removed** (audit §3.1). **Not built:**
a seat has no act, so `serves` is false for it and its instance has no address (§9); `room_sit` does not exist.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `POST .../room/<id>/sit` | POST | always (a seat here) | — | World (standpoint) | seat.yaml; `room_sit` audit §2.2/§6.1 **[no handler yet]** |

---

## C.20 `creator` — `http://local/creator/<id>`

Part: creators-chair (fixture, command → command-room). id `creator/command-room-1`.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `POST .../creator/command-room-1/present` | POST | always | `what` free req | World | `CREATOR_PRESENT` |

---

## C.21 `plan` — `http://local/plan/<strategy id>` — **PROJECTED store (§9.2)**

There is no planning-board part in any map and no `plan_*` station act. The routes are the
projected-store bridge of C.11 (`effector/plan.rs`), reachable wherever the body stands: `plan_*`
write the agency layer and stay visible in the dynamic system prompt (§9.2). The `<id>` is the
strategy id, as in C.11.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../plan/<id>` | GET | always | — (returns the whole agency layer) | projected (agency) | `read` in `effector/plan.rs` |
| `POST .../plan/<id>/break_down` | POST | always | `what` free req | projected (agency) | a child strategy of `<id>` (`StrategyTarget::ChildOf`) |
| `POST .../plan/<id>/order` | POST | always | `pieces` array of strings, read by the handler | projected (agency) | `write` in `effector/plan.rs` |
| `POST .../plan/<id>/reorder` | POST | always | `pieces` array of strings, read by the handler | projected (agency) | `write` in `effector/plan.rs` |
| `POST .../plan/<id>/scope` | POST | always | `what` free req; `in`/`out` free opt, kept as scope notes | projected (agency) | `write` in `effector/plan.rs` (named `plan_scope` in the audit) |

---

## C.22 `standard` — `http://local/standard/<id>` — **Not built**

Parts: **standards board** (new, to be reachable from every level — audit §3.4) and house-palette's
read (`standard_read`, re-homed from `portrait.read_house_style`, audit §3.4). No act, part or map
placement exists; `namespace_of` maps `house-palette` to `standard`, but nothing attaches.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../standard/palette-1/read` | GET | always | — | World | house-palette.yaml; `standard_read` audit §3.4/§6.2 **[no handler yet]** |
| `POST .../standard/<id>/propose` | POST | always | `what` free req; `why` free req | World | `standard_propose` audit §6.2 **[no handler yet]** |
| `POST .../standard/<id>/settle` | POST | **paired** (a norm is settled, not decreed) | `what` free req | World | `standard_settle` audit §6.2 **[no handler yet]** |

---

## C.23 `trial` — `http://local/trial/<id>` — **Not built**

Part: **trials shelf** (new fixture, audit §6.2). No act, part or map placement exists.

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
| `POST .../gather/long-table-1/call` | POST | always | `about` free req; `who` free req | World | `GATHER_CALL` |
| `GET .../gather/long-table-1/read_standing` | GET | always | — | World | `gather_read_standing` audit §6.2 **[no handler yet]** |

---

## C.25 `bench_*` — the mode surface — mounted on the six editing stations

The 11 git-named verbs are declared in `engine/bench.rs` (`BENCH_ACTS`) and attach to a fixed list
of six stations, `BENCHES`: chronicle-terminal, character-terminal, story-desk, easel, survey-desk
and map-table, so one implementation serves every editing station. The accession-desk,
catalogue, appraisal-bench and mending-bench have `modes:` but no bench verbs. There is no
`/bench` namespace: each verb is a `POST` under the station's own namespace,
`http://local/<ns>/<id>/bench_<verb>` — chronicle, character, story, portrait or place; the
`map-table`'s are under the unmounted `map` namespace and so unroutable. The verbs are named after
git because the model has seen them millions of times (the module doc of `engine/bench.rs`): a
coinage recruits nothing from the prior. The mode is **per body** (§5.1), held as engine state.
All eleven are shipped (the `bench` dispatch in `engine/work/bench_acts.rs`), `bench_blame`
included; `bench_commit` is the only act that can be refused by somebody else's work, and its
collision check names the other party (`Benches::commit_where` in `sim/bench.rs`), so the Settling
trigger is real. Every verb is a `POST`, including the reads.

| Route | Method | Condition (mode transition) | Body schema | Store | From |
|---|---|---|---|---|---|
| `POST .../<ns>/<id>/bench_branch` | POST | reading → working (takes the claim) | `what` free req | `Sim.bench` + mode | `bench_branch` |
| `POST .../<ns>/<id>/bench_stash` | POST | working → reading | — | `Sim.bench` + mode | `bench_stash` |
| `POST .../<ns>/<id>/bench_stash_pop` | POST | reading → working | — | `Sim.bench` + mode | `bench_stash_pop` |
| `POST .../<ns>/<id>/bench_diff` | POST | working, offered | — | `Sim.bench` | `bench_diff` |
| `POST .../<ns>/<id>/bench_restore` | POST | working → reading (discards) | — | `Sim.bench` + mode | `bench_restore` |
| `POST .../<ns>/<id>/bench_stage` | POST | working → offered | — | `Sim.bench` + mode | `bench_stage` |
| `POST .../<ns>/<id>/bench_unstage` | POST | offered → working | — | `Sim.bench` + mode | `bench_unstage` |
| `POST .../<ns>/<id>/bench_commit` | POST | offered → reading (**may fail with a conflict naming the other party**) | `why` free req | `Sim.bench` → mind folder | `bench_commit` (manufactures the Settling trigger) |
| `POST .../<ns>/<id>/bench_status` | POST | any | — | `Sim.bench` | `bench_status` |
| `POST .../<ns>/<id>/bench_blame` | POST | any | `what` free req | `Sim.bench` (`last`) | `bench_blame` (the custody chain) |
| `POST .../<ns>/<id>/bench_log` | POST | any | `what` free req | `Sim.bench` | `bench_log` |

`compose` (no parameters) is declared beside them (`COMPOSE` in `engine/bench.rs`) and offered to a
Maker whose mission writes a document.

---

## C.26 `file_*` — the working-copy file verbs

Five file verbs over a body's `Sim.bench` working copy, declared in `engine/bench.rs` beside
`bench_*` and attached to the same six stations (`BENCHES`); `file_delete` attaches to the
`story-desk` only (`SCRATCH`: a story desk accumulates scratch nobody else has to see, and the
other benches retire a document with a reason, `record_let_go`, rather than deleting it). There is
no `/file` namespace: each is `POST http://local/<ns>/<id>/file_<verb>`. Their handlers are in
`engine/work/bench_acts.rs`.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `POST .../<ns>/<id>/file_read` | POST | working (copy open) | `path` free req; `page` free opt | `Sim.bench` | `FILE_READ` |
| `POST .../<ns>/<id>/file_list` | POST | working | `path` free req | `Sim.bench` | `FILE_LIST` |
| `POST .../<ns>/<id>/file_edit` | POST | claimed (working) | `path` free req; `at` free req (the anchor); `do` enum req (`replace`, `insert_before`, `insert_after`, `delete`, `move`; `EDIT_ACTIONS`), each value bringing its fields (`text`; `dest`, `where` enum `before`/`after`); `old`, `to` free opt | `Sim.bench` | `FILE_EDIT`, by anchor (`docs/npcd_anchored_edits.md`) |
| `POST .../<ns>/<id>/file_write` | POST | claimed (working) | `path` free req; `content` free req | `Sim.bench` | `FILE_WRITE` |
| `POST .../<ns>/<id>/file_delete` | POST | claimed (working); `story-desk` only | `path` free req | `Sim.bench` (whiteout) | `FILE_DELETE` |

---

## C.27 The personal namespace — `/phone`, `/history`, `/self` (§7.2)

Reachable wherever the body stands (§7); npcd owns them even when an embedder replaces the vault
(§9.1). Always in the near-you index (§6). The phone acts are `WORLD_ACTS` entries routed
through `effector/phone.rs`; `send_image` is not (below).

### `/phone` — the messaging surface

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../phone` | GET | always (has a handset) | — (the thread listing, `phone::threads`) | Sim phone / projected | §7.2 |
| `POST .../phone/message` | POST | `Always` | `via` enum(`Ways`) req; `to` opt; `intent` free req | Sim phone / projected | `MESSAGE` (`engine/acts/conversation_acts.rs`); `message` (`engine/enact/messaging.rs`) |
| `POST .../phone/reply` | POST | `Always` | `to` enum(`Awaiting`) req; `intent` free req | Sim phone / projected | `REPLY` (`engine/acts/conversation_acts.rs`) |
| `POST .../phone/reach_out` | POST | `Contacts` non-empty | `to` enum(`Contacts`) req; `intent` free req | Sim phone / projected | `REACH_OUT`; `reach_out` (`engine/enact/messaging.rs`) |
| `POST .../phone/invite` | POST | `Invitable` non-empty | `to` enum(`Invitable`) req; `who` enum(`Invitees`) req; `intent` free opt | Sim phone / projected | `INVITE`; `invite` (`engine/enact/messaging.rs`) |
| `POST .../phone/open_group` | POST | `Contacts` non-empty | `called` free req; `with` enum(`Contacts`) req; `intent` free opt | Sim phone / projected | `OPEN_GROUP`; `open_group` (`engine/enact/messaging.rs`) |
| `POST .../phone/<thread>/sign_off` | POST | `Leavable` (`Always` act) | `intent` free req | Sim phone / projected | `SIGN_OFF` (`engine/acts/tower_and_work.rs`); `sign_off` (`engine/enact/messaging.rs`) |

*As-built: `/phone/send_image` is **not built**: it is not mounted or advertised. `send_image` is
not a body act — it is absent from `body::is_of_the_body` and `enact::is_mine`, so `enact::perform`
returns `NotOfTheBody`, which `enact_response` maps to `500`. The engine performs `send_image` above
the body layer (the image-guest / interaction path), which the `/phone` route cannot reach, so it is
left out of `phone.rs` `VERBS` and the `OPTIONS` schema rather than mounted as a route that only
ever errors. The five verbs (`message`, `reply`, `invite`, `open_group`, `reach_out`) and `sign_off`
are routed as the table shows.*

### `/history` — read-only onto what this body has done and seen

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `GET .../history` | GET | always (personal, §7) | query `cursor` (a position) and `limit` (default 50, at most 200) | projected (memory, read) | `history` in `effector/history.rs` |

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

1. **`read` and `scan` → `POST /here/…`, not `GET`.** The earlier decision dissolved reads into
   the router's internal `GET`. Both acts route through `body::perform`, and `read` moves the
   reader's read-cursor, so they are effectful reads and are `POST` verbs on `/here` (G.2); a
   resource's `GET` is its descriptor.
2. **`sign_off` — store and namespace.** Appendix A groups it with *world-state* acts, but it acts on
   a conversation thread. **Call:** routed under `/phone/<thread>/sign_off` (C.27), store = Sim phone
   / projected; cross-referenced in C.2.
3. **Two `gather`s.** Field extraction (`GATHER`, `Extractable`, in `engine/acts/carrying.rs`) vs the
   reading-table `gather_*` (`GATHER_CALL` in `engine/station.rs`). **Call:** disambiguated by
   surface — `POST /here/gather` vs `POST /gather/reading-table~0/call`.
4. **Not-part-bound world acts.** Thirteen are name-addressed on `/here` (C.2, Appendix G): the target
   is a body field, not an instance id. `engage`/`command_tower`/`produce` are **war-world**, never
   mounting in the vault; `command_tower` and `produce` are part-bound and mount at their fixtures.
5. **`library` (craft) acts.** `LIBRARY_READ`/`WRITE` are shipped `STATION_ACTS` absent from the audit
   and from Appendix A, given subsection C.7. They mount under the station's own prefix.
6. **Code verb names vs Appendix A.** Tables use the **code** names for shipped verbs and the
   **audit** names for authored-but-unbuilt verbs, citing both (e.g. code `map_settle_border` vs
   audit `move_border`).
7. **`claimed` vs `mode:working`.** For the six editing stations, "claimed (working)" — holding the
   `binds` *is* being in the working mode. `bench_*` (C.25) are the transitions.
8. **`plan_*` / `orders_*` → projected store.** Per §9.2; the bridge routes (C.11, C.21) write the
   agency layer; the station acts `orders_set`/`hand_to`/`report_done` write `Sim.ledger` (D.5).
9. **`map_*`.** The `map_*` acts are cartography lore and the `map` namespace is not mounted (C.10);
   runtime topology mutation is the `/reshape` primitive (Appendix F).
10. **Typed (shaped-not-enforced) fields.** The only declared non-string parameters in the catalogue
    are `command_tower.x/y/depth` (`integer`). `give.count` and `produce.count` are strings the
    handler parses, and the plan bridge reads `pieces` from the JSON as an array. The handler
    validates (§11).

### Route count (as built)

Counted from the act `at:` attachments and `short_verb`, not from the audit. Station verbs are all `POST`:

| Namespace | verbs | | Namespace | verbs |
|---|---|---|---|---|
| archive (C.29) | 3 | | record (C.4) | 13 |
| time (C.29) | 1 | | enquiry (C.12) | 4 |
| chronicle (C.3, C.25, C.26) | 20 | | command (C.11) | 7 |
| character (C.7, C.8, C.25, C.26) | 22 | | dispatch, cast, creator, gather (C.13, C.18, C.20, C.24) | 1 each |
| story (C.5, C.7, C.25, C.26) | 21 | | plant (C.14) | 2 |
| portrait (C.6, C.25, C.26) | 21 | | stores (C.15, with `produce`) | 3 |
| place (C.9, C.25, C.26) | 19 | | structure (C.16) | 3 |
| tower (C.2) | 1 | | | |

That is 144 station verbs, plus a `GET` descriptor and an `OPTIONS` on each of the 18 mounted
namespaces. The fixed surfaces add 52 routes: `/lift` 3, `/phone` 7, `/here` 14 (the index and 13
verbs), `/history` 1, `/self` 5, `/plan` 10, `/orders` 10, `/reshape` 1, and the near-you index.
**About 214 routes**, excluding `OPTIONS`, across 18 station namespaces and eight fixed prefixes. The
namespaces `library`, `roster`, `room`, `standard`, `trial`, `bench` and `file` the audit and this
appendix once counted do not exist as mounts.

**Correction (Appendix D):** the "Store" column above and any "0-of-79 / [no handler yet]" framing
are stale. The writing handlers are *shipped* — in `engine/work/` (`work::perform`), not
`engine/station.rs` (declaration only) — and the store for the authoring namespaces is `Sim` (its
`record`/`ledger`), not the npc-map `World`; a subset reaches the mind folder through the `bench`
working copy. Appendix D.5 is the authoritative per-namespace store map, and D.6 what is still open.
The handlers truly absent are the short list (`roster_*`, `room_sit`, `character_write_beliefs`).

## C.29 `archive` and `time` — the merge table and the time machine

Two station namespaces the audit and Appendix A never listed, both in the vault: `vault-archive`
holds the merge table, `vault-time` the time machines.

| Route | Method | Condition | Body schema | Store | From |
|---|---|---|---|---|---|
| `POST .../archive/merge-table~0/merge_status` | POST | the body carries an operation whose publish step is current | — | the operation store | `MERGE_STATUS` (`engine/station.rs`); `perform` in `engine/archive.rs` |
| `POST .../archive/merge-table~0/merge_publish` | POST | as above; refused, naming the documents, when the record has moved under the work at the same lines | — | the operation store | `MERGE_PUBLISH` |
| `POST .../archive/merge-table~0/merge_sync` | POST | as above | — | the operation store | `MERGE_SYNC` |
| `POST .../time/time-machine~<n>/travel` | POST | at a time machine | `year` free req (a number as text) | `Sim` | `TIME_TRAVEL`; `time_travel` in `engine/work/dispatch.rs` |

The merge acts' verbs keep their full names (`archive_` is not a prefix of them) and are the only
way work reaches the record (`docs/replicated_substrate.md` §10.3). The time machine sets the year
of the world's history a Maker works in; reporting the work returns it to the present.

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
shipped, in `engine/work/`), and `archived/tool_world_state.md §2`'s "the mind folder is not a git
repository" (it is one now). Grounded in the mind folder (`D:\prog\npcd-data\mind\battle-mind` on
the machine this was checked on) and `npcd/src/{mind,sim,engine}`.*

## D.1 The mind folder on disk

The mind folder is the authored half of the product — text a person wrote, none generated
(`npcd/src/mind/mod.rs`, module doc). Its inspected layout:

| On disk | Holds | Console `Section` (address) | Bench area |
|---|---|---|---|
| `layers/world/` | **1,268** canon `.md` — nested topics+entries (`history/`, `geography/`, `locations/`, `factions/`, `events/`, `creatures/`, `identity/`, `combat/`, …) | `canon` → `layers/world` (`Section::dir` in `mind/address.rs`) | `layers` |
| `layers/eras/` | 13 era `.md` — the shared storyline, one per era | `eras` → `layers/eras` | `layers` |
| `layers/stories/` | backstory `.md` — accounts filling the gaps between eras | `stories` → `layers/stories` | `layers` |
| `layers/memory/<char>/` | per-character remembered episodes `.md` | `memory` → `layers/memory` | `layers` |
| `layers/life/<char>/` | dated life episodes — lifegen input, executed by the substrate authoring plane (`engine/authoring.rs`) | — | `layers` |
| `layers/dreams/` | dream seeds and the dream prompt (`maker/seed-dreams.md`) | — | `layers` |
| `personalities/*.yaml` | **character sheets** — 27 of them (the cast, five zenlings, the keeper and the maker); who a character is before living anything | `characters` → `personalities` | `personalities` |
| `personalities/portraits/*.png` | portrait **plates** (27) shipped beside the cast | — (binary) | — |
| `worlds/*.yaml` | settings + their `selects`/`excludes`/`personalities` filters | `worlds` → `worlds` | `worlds` |
| `responses/*.yaml`, `moods/*.yaml` | craft libraries — reply shapes and registers | `responses`, `moods` | `responses`, `moods` |
| `map/battle-cities/*.yaml` | the **npc-map source** — level maps + `parts/` catalogue; where the C.0 parts are placed | — | `map` |
| `projection.yaml`, `mind.yaml`, `CLAUDE.md` | settings — schema, mind config, authoring guidance (`SETTINGS` in `mind/address.rs`) | `settings/{projection,mind,guidance}` | — (bench allow-list excludes) |
| `bearings.yaml`, `guardian.yaml`, `missions.yaml` | configuration the engine reads (bearings, the guardian, the mission templates) | — | — (`missions`: its template library only) |
| `.git/` | git metadata | — | — |

The tree does not hold `images/`, `accounts/` or `schema/`: a deployment whose data and mind
roots coincide keeps its redo-log persistence (`.substrate/`), operator accounts and token
registry in the data directory beside it, and none of those is in the checkout inspected.

Two facts the rest turns on:

- **The mind folder is a git repository** (`layers/` tracked). This resolves the *precondition* of
  `archived/tool_world_state.md` dispute A. **The daemon reads it in process and writes through a
  shelf, not through `git` commands:** `Command::new("git")` appears in `npcd/src` only in test
  modules, and `git2` and `gix` not at all. The mind is read from a folder or a commit snapshot
  (`MindFiles` in `mind/files.rs`, over `zend_vfs`; `mind_store.rs` loads it from the substrate
  log), and a commit lands through a `Shelf` (D.4).
- **The canon under `layers/world/` (1,268 files) is not writable by any station act.** The
  station/bench surface reaches only `layers/eras`, `layers/stories`, `personalities`,
  `moods`, `responses` (and `map`). Canon is editable only through the console admin editor
  (D.3).

## D.2 The store taxonomy — every store a handler can touch

There are **four** stores, not two. The design's "world state vs projected state" names the
first and last and folds the middle two away.

**1. The in-RAM npc-map `World`** — *who is where, holding what, and what happened.* Owns
placement/standpoint, `Actor.hold` claims, the lift, witnessed events. Held in RAM and
snapshotted into the substrate's entity records (`body/<body>` and `lift`,
`npcd/src/world_state/snapshot.rs`); bodies also re-enter deterministically
(`engine/runtime/`). Reached under the one lock (`Hosted::with`/`read`,
`npcd/src/world/mod.rs`). Claim = proximity (`npc-map/src/perceive.rs`) + `Actor.hold`. Handlers: the
world-state half of `WORLD_ACTS` (`perform_said` in `engine/enact/mod.rs`; C.1–C.2).

**2. The `Sim`** (`Sim` in `sim/mod.rs`) — *everything about the world that is not its map.*
Behind the same lock (`with_sim`/`with_both`); **the store Appendix C mislabels "World" for
every authoring namespace.** Sub-stores:
- `record: Record` (`sim/record.rs`) — the authored-content state machine. One `Item`
  per named thing (era, entry, gap, story, character, portrait, place, enquiry, structure,
  accession) with `state` (Unwritten→Held→Draft→Offered→Filed→Retired), `holder`,
  `condition`, and an optional `path` (`Item` in `sim/record.rs`) naming the mind-folder file
  when the thing is a document. `Item.body` is an in-RAM draft buffer for non-documents.
- `bench: Benches` (`sim/bench.rs`) — the per-body working copy over the mind folder (D.4),
  reading and committing through a `Shelf` (`sim/shelf.rs`).
- `ledger: Ledger` — orders, verdicts, claims-as-orders. Plus missions, phone threads,
  postings, field, devices, tower, packs.
- **Persistence:** the world snapshot writes the entity maps as `sim/<map path>/<key>` records
  (`SIM_MAPS` in `world_state/snapshot.rs`: record items, the benches' open, stashed and last
  state, the ledger's lists, missions, …) and the rest as `sim/meta`, so a restart keeps them.
  The benches' *documents* stand on a shelf, not in the snapshot.
- **Claim:** `Record` custody (`Item::held_by_other`) + `Benches` isolation and commit-time
  collision check.

**3. The mind folder (authored lore on disk, or in git)** — *canon, cast, craft, storyline.*
Owns the D.1 tree. Reached through a `Shelf`. **Two doors, never conflated:** the console
editor `npcd::mind` (D.3, addressed `canon/ammo/bolt`, gated read `User` / write `Admin`) and
the effector working copy `Sim.bench` (D.4, addressed by `MindPath`, gated by claim + commit,
invisible until commit). Distinct from World (placement) and from the substrate
(per-character projected layers).

**4. The substrate** — *projected state: agency, belief, memory.* Owns `AuthoredBelief`,
`AuthoredRelationship`, `AuthoredStrategy`, modulations, the per-character layers.
Persistence: mandatory redo-log (`substrate/substrate.log`). Claim: the belief-write
invariant — a belief is the character's own, never a tool's silent edit (`npc_mind_design.md`;
§9.2); the authoring plane that may write it is `engine/authoring.rs`, in a *separate catalog*
from the action plane (`CATALOG` in `engine/authoring.rs`, kept disjoint by a test). **This is
§9.2's "projected store."** The `plan_*`/`orders_*` bridge routes write it through
`Npcs::put_strategy_self` (`effector/plan.rs`, D.6 #2); no station act handler does: the
station acts `orders_*` write `Sim.ledger` (D.5).

> **The correction in one line:** §9's "world-state vs projected-state" should read **World
> (RAM, snapshotted) · Sim (RAM, snapshotted, incl. the record/bench state machine) · the mind
> folder (disk or git) · the substrate (projected, persisted)**. Appendix C's `Store = World`
> for the authoring namespaces means, in almost every case, **`Sim.record` (+ the mind folder
> via `Sim.bench` for the document kinds)** — not the npc-map World.

## D.3 The mind authoring layer (`npcd::mind`) — the console door

For an implementor (`mind/mod.rs` is authoritative):
- **Addressing.** A wire address is `section/name/name…` (`canon/ammo/bolt`), never a path.
  `Address::parse` (`mind/address.rs`) is the only place that knows the on-disk mapping; the
  section supplies directory *and* extension (`.md` for canon/eras/stories/memory, `.yaml`
  otherwise). Nine sections (`SECTIONS` in `mind/address.rs`); a client cannot invent a tenth.
  Names are validated as `MindPath` (`..`, drive letters, NUL, symlinks refused before touching
  disk).
- **Granularity.** `doc` (whole document), `section` (a YAML doc as form `Field`s, comments
  preserved), `parts` (one item inside a settings document). YAML edits go through
  `registry::yaml_edit::splice` so author comments survive.
- **Mechanism.** `doc::read`, `doc::write` with `must_be_new`, `doc::remove` (`mind/doc.rs`),
  each over a `&dyn Shelf`: a write is `shelf.commit` of the one document, so it lands wherever
  that shelf lands commits; ≤512 KB UTF-8 (`MAX_BYTES`).
- **Permission.** Read signed-in, write admin — `GET /v1/mind/entry` `Role::User`,
  `PUT/DELETE` `Role::Admin` (`npcd/src/api/route_table.rs`). Scope filtering (`scope.rs`)
  applies a world's `selects`/`excludes`/`personalities`.

An NPC never uses this door — it cannot present an admin role, and would not write disk
directly (D.4).

## D.4 How an effector handler writes the mind folder

The shipped mechanism is `engine/work/dispatch.rs` (the dispatcher for `STATION_ACTS` + `BENCH_ACTS`
+ `MISSION_ACTS`: `is_mine` and `perform` in `engine/work/dispatch.rs`) — **not** `engine/enact/`
(world acts) and **not** `engine/station.rs` (which only declares the `Tool` statics).

**The two-store bench act.** A body at a station holds both a `Record` item (the thing it has
taken on) and a `Benches` working set (documents it has changed, uncommitted). `Benches`
(`sim/bench.rs`) is a git-*like* working copy in memory, over a `Shelf` (`sim/shelf.rs`):
- `open` maps body→`Working` (`Change{base, now}` per path); a read falls through to the shelf,
  a write touches the shelf **only at commit**.
- **collision is a comparison, not a lock:** commit re-reads each file and requires it to
  still equal `base`; the second committer is refused and told whose work it hit
  (`Benches::commit_where` in `sim/bench.rs`). This is what "manufactures the Settling trigger"
  (C.25) — for real.
- commit hands every changed doc to the shelf as one all-or-nothing `Shelf::commit`, and
  records `last[path]=body` for `bench_blame`.
- allow-list of editable areas (`EDITABLE_AREAS` in `sim/bench.rs`): `layers`, `map`, `missions`
  (its template library only), `moods`, `personalities`, `responses` — deliberately not
  `projection.yaml`/`mind.yaml`/`schema/`, nor `worlds/`, a world's own configuration.
- a mission whose step writes one document commits that document alone (`commit_only`);
  anything else in the working set stays open.

**The write path.** For the prose verbs (`chronicle_add_entry`/`rewrite_page`, `story_draft`,
`character_write_*`, `place_write_*`) dispatch funnels into `write_into` (`engine/work/bench_acts.rs`):
1. Ask `Record::settle_path(name)` (`sim/record.rs`) — returns a path for `Era`, `Story`,
   `Gap` (`layers/eras/…`, `layers/stories/…`) and `Place` (`layers/world/locations/…`) kinds,
   minting on first write; all other kinds get `None`.
2. **No path →** `Record::write` appends to the in-RAM `Item.body`. *Never reaches a document.*
3. **Path →** `Record::claim_for_write` (custody) then `Benches::append(body, name, path,
   text)` — the text goes to the working set, invisible until commit.

The join: `Record::index_canon(root)` (`sim/record.rs`) reads `layers/eras`, `layers/stories`
and the two world layers at host time, adopting each doc with its `path` and `State::Filed`, so
an era *is* the file it is. Portrait and craft writes take a **field** path instead:
`portrait_draw`/`prompt_edit` → `bench.write_field(personalities/<who>.yaml,
["portrait","prompt"], …)` (the `portrait_draw` arm of `perform` in `engine/work/dispatch.rs`);
`library_write` → `bench.write_field(moods|responses/<id>.yaml, [field], …)` (`library` in
`engine/work/craft.rs`); `file_write`/`edit`/`delete` → `bench.write`/`edit`/`remove` over any
editable-area doc (the `file_*` arms of `bench` in `engine/work/bench_acts.rs`).

**The commit does not write the mind, and that replaces the admin-writes-disk reconciliation.**
`bench_commit` requires a one-line `why`, runs `Benches::commit` (collision check, then
`Shelf::commit`), then settles the record item. **Replaced design:** the first plan wrote the
changed files with plain `std::fs` under the world lock on behalf of the body's token. What was
built is that the shelf a body commits to is chosen by `Hosted::shelf_of(body)` (`world/mod.rs`): a
body on a step of an operation commits into that operation's store (`StoreShelf` in
`opstore/shelf.rs`), and a body on no such step reads the published mind (`Published`), whose
`commit` always refuses with `NO_OPERATION` ("There is nowhere for this to go: work reaches what
stands only through an operation"). Work reaches the record only through `merge_publish`
(`MERGE_PUBLISH`, C.29; `engine/archive.rs`). The two doors to store #3 remain: the **console door**
(a person's edit, `Role::Admin`, through the same `Shelf` abstraction) and the **effector door**
(the character's edit, gated in-fiction by claim + working copy + commit-collision, never by an
operator role — the token resolves to a body, the body's standing is the authorization, and
publishing lands it).

**Recipe for a new writing handler.**
1. If it is a **document** (prose in a file), give its `Record::Kind` a `settle_path` arm
   (today Era, Story, Gap and Place, in `sim/record.rs`) and `index_canon` adoption; then route
   through `write_into` for working-copy isolation + commit-collision + the shelf for free.
2. If it is a **judgement** (provenance, condition, verdict, order), write `Record` fields or
   `Sim.ledger` — kept by the world snapshot (D.2), and not a document on any shelf.
3. If it is **projected** (a plan, a belief, an order the character holds), it must reach the
   substrate authoring plane (store #4) through the async bridge (`effector/plan.rs`), not through
   the synchronous `work::perform`.
4. Gate with `Record` custody (`take`/`claim_for_write`/`held_by_other`), not an operator
   role; return second-person `Outcome::Refused`, which the route envelope (§12) wraps.

## D.5 Per-namespace store table (reconciling Appendix C)

Store the App-C column claims vs the store the shipped handler touches (`engine/work/`), whether it
reaches the mind folder, and the mechanism. Schemas/verbs are in Appendix C.

| Namespace (App C) | App C "Store" | **Actual store today** (`engine/work/`) | Mind folder? | Mechanism |
|---|---|---|---|---|
| `chronicle_add_entry`/`rewrite_page` (C.3) | World | `Sim.record` + **mind folder** via `Sim.bench` | **Yes** (`layers/eras/*.md`) | `write_into`→`settle_path`→`bench.append` (`write_into` in `engine/work/bench_acts.rs`) |
| `chronicle_retire_entry` (C.3) | World | `Sim.record` (→Retired) | No | `set_state` (`engine/work/bench_acts.rs`) |
| `chronicle_settle_boundary` (C.3) | World | `Sim.record` xref + `Sim.ledger` | No | `settle` (`engine/work/bench_acts.rs`) |
| `story_draft` (C.5) | World | `Sim.record` + **mind folder** via `Sim.bench` | **Yes** (`layers/stories/*.md`) | `write_into`→`bench.append` |
| `story_file` (C.5) | World | `Sim.record` (→Filed) | No | `set_state` (`engine/work/bench_acts.rs`) |
| `character_write_identity`/`wants`/`memories` (C.8) | World | **mind folder** via `Sim.bench` *(as-built: D.6 #1 done)* | **Yes** (`personalities/<who>.yaml` `anchor`/`wants`; `layers/memory/<who>/memories.md`) | `bench.write_field`/`bench.append` (the `portrait_draw` pattern) — RAM `Record.write` only when there is no mind folder |
| `character_settle_relation` (C.8) | World | `Sim.record` xref + `Sim.ledger` | No | `settle` (`engine/work/bench_acts.rs`) |
| `place_write_entry`/`write_local_history` (C.9) | World | `Sim.record` + **mind folder** via `Sim.bench` *(as-built: D.6 #1 done)* | **Yes** (`layers/world/locations/<slug>.md`; `layers/world/geography/<slug>.md`) | entry: `write_into`→`settle_path`(Place→locations)→`bench.append`; local history: `history_path`(geography)→`bench.append` |
| `place_settle_route` (C.9) | World | `Sim.record` xref + `Sim.ledger` | No | `settle` |
| `portrait_draw`/`prompt_edit` (C.6) | World | **mind folder** via `Sim.bench` | **Yes** (`personalities/<who>.yaml` `portrait.prompt`) | `bench.write_field` (the `portrait_draw` arm of `perform` in `engine/work/dispatch.rs`) |
| `portrait_file_plate` (C.6) | World | `Sim.record` (→Filed) | No (no image store) | `set_state` |
| `portrait_settle_likeness` (C.6) | World | `Sim.record` xref + `Sim.ledger` | No | `settle` |
| `library_read`/`write` (C.7) | World | **mind folder** via `Sim.bench` | **Yes** (`moods`/`responses/*.yaml`) | `bench.read_field`/`write_field` (`library` in `engine/work/craft.rs`) |
| `record_*` (accession/provenance/hand_on/describe/arrange/cross_reference/leave_note/tidy_index/mend/mark_repair/let_go) (C.4) | World | `Sim.record` fields | No | field setters (the `record_*` arms of `perform` in `engine/work/dispatch.rs`) |
| `record_appraise`/`write_reason` (C.4) | World | `Sim.ledger` verdicts | No | `ledger.record_verdict` (the `record_appraise` and `record_write_reason` arms of `perform`) |
| `map_add_place`/`remove_place` (C.10) | World (topology) | `Sim.record`; **no topology mutation** | No | `Record.put`/`let_go` (the `map_*` arms of `perform`) |
| `map_settle_border` (C.10) | World (topology) | `Sim.record` xref + `Sim.ledger` | No | `settle` |
| `orders_set`/`hand_to`/`report_done` (C.11) | **projected (agency)** | **`Sim.ledger`** (kept by the world snapshot) — *not the substrate*; the bridge routes of C.11 write the substrate | No | `ledger.set_order`/`hand_to`/`finish` (the `orders_*` arms of `perform` in `engine/work/dispatch.rs`) |
| `plan_*` (C.21) | **projected (agency)** | no station act; the bridge routes of `effector/plan.rs` | No | `Npcs::put_strategy_self` → substrate `AuthoredStrategy` (D.6 #2) |
| `enquiry_*` (C.12) | World | `Sim.record` + `Sim.ledger` | No | the `enquiry_*` arms of `perform` in `engine/work/dispatch.rs` |
| `dispatch_post_wake`/`cast_report_disagreement`/`plant_*`/`gather_call` (C.13/18/14/24) | World | `Sim.ledger` | No | `ledger.set_order` (the matching arms of `perform`) |
| `structure_*` (C.16) | World | `Sim.ledger` verdicts | No | `ledger.record_verdict` |
| `stores_put_back`/`take_out` (C.15) | Sim | `Sim.record` give_back/take | No | the `stores_*` arms of `perform` |
| `creator_present` (C.20) | World | `Sim.record` (read) | No | the `creator_present` arm of `perform` |
| `bench_*` (C.25) | World + mode | `Sim.bench` + `Sim.record` (commit → the operation's store; published through `merge_publish`) | **on publish: Yes** | the `bench` dispatch in `engine/work/bench_acts.rs` (**shipped**, contra App C flags) |
| `file_*` (C.26) | World overlay | `Sim.bench` (commit → the operation's store; published through `merge_publish`) | **on publish: Yes** | the `file_*` arms of `bench` in `engine/work/bench_acts.rs` |
| `merge_*` (C.29) | — | the operation store (`opstore/`) | **Yes**: the only way work reaches the record | `perform` in `engine/archive.rs` |

**Verbs Appendix C lists that do not exist as station acts** (`STATION_ACTS` in `engine/station.rs`):
`character_write_beliefs` (deliberately absent — beliefs are earned on the sleep clock, not
authored; the example of `CHARACTER_WRITE_IDENTITY` in `engine/station.rs` says so), `plan_*` (none as
station acts; the bridge routes stand in), `roster_*` (none), `room_sit` (none); and
`portrait_redraw` is really `portrait_prompt_edit` (`PORTRAIT_PROMPT_EDIT`). Building the C.8-belief
and C.19 routes is writing *new* handlers, not migrating.

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
craft-library, character, place, and the raw `bench_`/`file_` surface** reach the mind folder —
through `Sim.bench`, not "World." Everything else marked `World` is `Sim.record`/`Sim.ledger`,
kept by the world snapshot.

## D.6 What was missing, and what still is

1. **Durable persistence of `Sim.record`/`Sim.ledger`, or a disk write for the non-era/story
   authoring verbs.** `character_write_*`, `place_write_*`, and every `record_`/`orders_`/
   `structure_` verdict lived only in RAM (the no-path branch of `write_into` in
   `engine/work/bench_acts.rs`, `Record::write`); `Sim` was not in the substrate redo log, so they
   did not survive a restart. Fix: give Character/Place kinds a `settle_path`
   (`personalities/…`, `layers/world/locations|geography/…`) + `index_canon` adoption so they
   route through `bench`; or persist `Sim`.

   *As-built: **done, both ways.** `Sim` is persisted by the world snapshot (`SIM_MAPS` and
   `sim/meta` in `world_state/snapshot.rs`), so the `record_*`/`orders_`/`structure_` verdicts
   survive a restart too; and character and place also reach a document.
   `character_write_identity`/`wants` splice the
   `anchor`/`wants` fields of `personalities/<who>.yaml` (the `portrait_draw` pattern);
   `character_write_memories` appends `layers/memory/<who>/memories.md`; `place_write_entry`
   settles into `layers/world/locations/<slug>.md` through `write_into`; `place_write_local_history`
   appends `layers/world/geography/<slug>.md`. `settle_path` gained a `Place → locations` arm,
   `Record::history_path` names the geography document, and `index_canon` now adopts both world
   layers as `Filed` `Place`s, so a survey survives a restart; a commit of a freshly-branched
   document also lands it `Filed` (the `Held → Filed` transition in `Record::set_state`).*
2. **The projected store was never written by a station handler.** §9.2/C.11/C.21 require
   `orders_`/`plan_` to write the substrate **agency** layer (`AuthoredStrategy`) and re-project
   into the system prompt. The station acts `orders_*` write `Sim.ledger` and there is no `plan_*`
   station act. **The write API is `Npcs::put_strategy` (`npcd/src/npcs/methods.rs`), not
   `engine/authoring.rs`** (which is only a life-document parser with no substrate side effect). A
   bridge from the turn loop to the shared `Npcs` handle was needed (Appendix E). Once it writes,
   the change projects through `persona::intent` (`npcd/src/engine/persona.rs`) and, in a mind
   whose projection declares one, the `agency` layer (`npcd/projection.yaml`).

   *As-built: **done.** The shared cast is installed on `Runtime` (`set_npcs`/`npcs`), and
   `effector/plan.rs` (mounted at `/plan`, `/orders`, and kept out of the generic station nest)
   writes the agency layer through the new owner-blind `Npcs::put_strategy_self`. `/self`
   (`effector/selfsurface.rs`) reads the same layers back. The change projects with no extra
   wiring, exactly as this item predicted.*
3. **Git is a repository, and the commit path is not git plumbing.** The daemon never runs the
   `git` command outside tests. `bench_commit` commits through a `Shelf` (D.4): into an operation's
   store, published to the record by `merge_publish`, and the collision check that manufactures the
   Settling trigger is real (§C.25). `bench_blame`/`log` answer from `Sim.bench`'s `last` and `open`
   state, which the world snapshot keeps. **Replaced design:** the earlier plan (`git add`/`commit` as
   the acting body at `bench_commit`, `git blame`/`log` back) was not built; the operation store and
   the merge table (C.29) are the custody and history that was built.

Also missing, lower-stakes: an **image store** for portrait plates; **canon (`layers/world/`)
writability from a station** (only the console reaches it); and **document metadata as front
matter** (provenance/condition live in `Sim.record`, not on the `.md` files). The effector
router/token surface is built (§8, Appendix C and E).

---

# Appendix E — Resolved implementation decisions

*The decisions the build was made against, settled either with the author (lore/product) or by
code investigation (engineering). Where the code chose differently the bullet says what was built
and names the replaced design.*

## Lore / product (settled with the author)

- **Character authoring.** `character_write_identity` → the `anchor` field of
  `personalities/<who>.yaml` via `bench.write_field(["anchor"], …)` (the `portrait_draw` pattern,
  `author_sheet` in `engine/work/craft.rs`); `character_write_wants` → a **new top-level `wants`
  field** on the sheet; `character_write_memories` → **append to `layers/memory/<who>/*.md`**.
  `character_write_beliefs` is **not exposed** — beliefs form on the sleep clock and
  belief-writes stay operator-only (the §9.2 invariant). So C.8 adds no belief route.
- **Place authoring.** `place_write_entry` → `layers/world/locations/<slug>.md`;
  `place_write_local_history` → `layers/world/geography/*.md`. Both are in the bench allow-list
  (`EDITABLE_AREAS` in `sim/bench.rs`).
- **New-part placement** — **Not built.** The plan was a **standards board** in every level's `core`
  node (reachable everywhere, audit §3.4), a **planning board** in command → command-room (beside the
  order table) and a **trials shelf** in chronicle → sorting-room (beside the appraisal bench). No
  map places them and no act attaches to them (C.21–C.23); the `/plan` bridge needs no part.
- **Git custody.** **Replaced design.** The decision was in-RAM, per-run custody with `bench_commit`
  writing files with `std::fs` and real git (`git add`/`commit` as the acting body, `git blame`/`log`
  back) a later cut. What was built is a `Shelf` (`sim/shelf.rs`): `bench_commit` records into the
  operation's store and work reaches the record through `merge_publish`; `bench_blame` and `bench_log`
  answer from `Sim.bench`'s state, which the world snapshot keeps (D.4, D.6 #3).

## Engineering (settled by investigation)

- **Agency writer = `Npcs::put_strategy`** (`npcd/src/npcs/methods.rs`), not `engine/authoring.rs`.
  `plan_break_down`/`order`/`reorder`/`scope` and `orders_*` write `AuthoredStrategy` (a tree via
  `parent_id`) into the `agency` layer, which projects through `persona::intent`
  (`npcd/src/engine/persona.rs`) with no extra wiring.
- **The bridge.** The cast is `Arc<Npcs>` (`Authored.npcs` in `npcd/src/api/mod.rs`) — no lock: a
  read loads the published `Cast`, a write runs on the persistence owner's thread
  (`docs/npcd_cast_lock_free.md`) — installed on `Runtime` via `set_npcs`/`npcs`
  (`engine/runtime/mind_hold.rs`) beside `set_substrate`, reusing the *existing* `Npcs`/substrate
  handle, never a second (the one-writable-handle guard, the `Npcs::substrate` doc in
  `npcs/mod.rs`). **Replaced design:** wrapping the cast as `Arc<tokio::RwLock<Npcs>>` and parsing
  `npc_id` from the `"npc-{id}"` body. The self-write is `Npcs::put_strategy_self(npc_id, target,
  body, now_ms)` — it resolves `owner` from the record, and a `StrategyTarget::ChildOf`
  (`break_down`) mints the child's id inside the same write. The projected verbs live in
  `effector/plan.rs`, an async router mounted at `/plan` and `/orders` (not the sync
  `work::perform`); the caller's `npc_id` comes straight from the device token's `DeviceCaller`.
  `/self` reads are `effector/selfsurface.rs`. `give_back` maps to `abandoned` (not "dormant"): the
  agency states are exactly `active`/`finished`/`abandoned`.
- **Token store.** A `tokens/` `Registry` under the **data directory** (`data/tokens`, opened in
  `tenant/tenant_start.rs`; `Tokens` in `effector/token.rs`) keyed `npc_id → {token, scope}`, one
  file per character, a real secret never derived from `body_id`; a `token → npc_id → body` lookup
  built at startup. Scope ∈ {`as-npc` (default, proximity-gated), `direct` (by-id, §7/§8.3)}; a
  re-mint rotates the token and revokes the old one.
- **Character/place persistence.** `Record::settle_path` has a `Kind::Place` arm
  (locations|geography per the write) and `index_canon` adopts those dirs, so place writes route
  through `bench`→commit like eras/stories. Character writes use `bench.write_field` on the sheet.
  This closed D.6 #1 for character and place; the `record_*`/`structure_*` verdicts are kept by the
  world snapshot (D.2).
- **Grammar.** **Replaced design.** The decision was that `invoke` joins the fixed frame as an
  `Availability::Always` tool with a **free-string `url`** (`FreeText{Balanced}`), coexisting with
  unmigrated tools while each namespace's `Tool`/`Choices`/`LIVE`/`Within` field was deleted as it
  moved, the `url` enum arriving only after the `Within` shrink. What was built is the enum from the
  start: `Choices::InvokeUrl` (`engine/tools/choices.rs`, bound as `("invoke", "url", InvokeUrl)`),
  resolved from `invoke_body::urls(within)` and compiled with each address's body shape (§11), and no
  namespace's compiled acts were deleted (§14).

---

# Appendix F — Runtime topology mutation

*Built. This is the last thing that makes the world "fully dynamic": adding, removing
and reshaping rooms while the daemon runs. Everything above makes the world dynamic for
what characters *do*; this makes it dynamic for what the map *is*. It is a real cut of
its own because the map is otherwise frozen once loaded (only actors/holds/lift/events are
mutable), and unfreezing it safely is the whole of the work.*

## F.1 The mutation surface

**Replaced design.** The first plan gave `World` a method per edit (`add_area`, `add_node`,
`add_portal`, `place_part`, `remove_node`, `retitle`, …), each returning a `Done`/`Refused`.
What was built is one value and one transaction: a `MapEdit` (`npc-map/src/mutate.rs`),
adjacently tagged on the wire (`{"op":"add_node","with":{…}}`), applied by `MapSet::apply`
and swapped into a running world by `World::reshape` (`npc-map/src/world/bodies.rs`):

```rust
pub enum MapEdit {
    AddArea(Box<Area>),                       // a new level/region/room
    RemoveArea(String),                       // and everything in it
    AddNode { area: String, node: Box<Node> },
    RemoveNode(Where),
    Retitle { at: Where, name: String },      // in place, without moving anything
    AddPortal { area: String, portal: Portal },        // a cross-area link (a lift stop, a gate)
    RemovePortal { area: String, between: [String; 2] },
    PlacePart { at: Where, placement: Placement },
    UnplacePart { at: Where, part: String },
    DefinePart(Box<Part>),                    // add a kind of thing to the catalogue
    LinkPart { part: String, link: PartLink },
    Describe { at: Where, character: String },// rewrite what a node is like
}
```

`MapSet::apply(&self, &MapEdit) -> Result<MapSet>` returns the rebuilt set or `Err`, leaving
the original untouched; `World::reshape(&mut self, &MapEdit) -> Result<Vec<String>, String>`
returns the bodies it relocated, or the reason it refused. A mutation that would break the
map is refused, in words, not half-applied.

## F.2 Every mutation re-runs the derived passes, transactionally

The map has derived state that is never authored — `Node::exits` and `Node::visible`
(woven from one-ended doors and sightlines, `npc-map/src/load.rs`), the portal graph
`MapSet.ways`, and the lift's `shaft` (one `Core` per `Level`, ordered by `ordinal`,
`World::build_shaft` in `npc-map/src/world/lifts.rs`). A mutation is therefore
**apply-to-a-copy, re-derive, validate, swap**:

1. clone the authored data (`MapSet::authored`) and apply the edit to the clone, refusing
   an edit that names nothing real (`MapEdit::apply_to`);
2. hand the result back through `MapSet::assemble_from` — the same pipeline a load runs:
   `weave` (doors mutual, exits filled), `ways` (portals both-way), the visibility pass, and
   `validate::check`;
3. if any of that fails, `apply` returns `Err` and nothing changes;
4. `World::reshape` resolves every body left standing where a node used to be against the new
   map (refusing the edit if there is nowhere to put one), swaps the set in, re-derives the
   shaft (`reshaft`), and releases the holds on what vanished before relocating the bodies,
   emitting `Left` and `Arrived` events;
5. the swap happens under the one lock, and the events are witnessed (§10) — a room appearing
   or vanishing is a thing bodies notice.

## F.3 The invariants a mutation may not break

- **Doors stay mutual and derived.** A one-ended `off`/`sees` is authored; `exits`/
  `visible` are always re-woven, never hand-set — so a one-way door remains impossible
  to write (the module doc of `npc-map/src/schema.rs`; `exits` and `visible` are
  `#[serde(skip)]`).
- **No stranded hold.** Removing a node or a part first releases any hold on it
  (`release_at`, as `World::leave` in `npc-map/src/world/bodies.rs` does for a departing
  actor) — a station that vanishes with a claim on it would strand that claim forever.
- **No body left nowhere.** Removing an occupied node relocates its bodies to the
  area's `arrival`/a `Core` (and logs it), or the removal is refused while occupied —
  the design's choice is *relocate and tell them*, since a Maker drowning a place should
  not be blocked by someone standing in it, and the body feeling the ground go is good
  fiction.
- **The shaft stays one `Core` per `Level`, ordinal-ordered.** A level added without a
  `Core`, or a second `Core` on a level, is a `validate` failure (refused).
- **Instance ids do not shift.** **Replaced design:** the plan accepted that removing the 2nd of
  three terminals renumbers the 3rd and that outstanding URLs could go stale. What was built
  carries every `(node, part)` offset of the old set forward (`MapSet::offsets`, `part_offsets`
  in `npc-map/src/load.rs`), so an edit can only add new instance ids and never renumber one
  already handed out: an id a character followed yesterday still resolves today, or `404`s
  because the thing is gone.

## F.4 How it is reached, and how it persists

An earlier draft named the Makers' `map_*` verbs (`map_add_place`, `map_settle_border`,
`map_remove_place`, …) as the effector face of this surface. Building it made plain that
those verbs are a *different map*, and this records the correction (design docs are
authoritative; a draft the code disproves is fixed, §AGENTS.md):

- The `map_*` verbs author **cartography lore** — `sim.record` items of `Kind::Place`,
  the fictional world-map the Makers draw as content (the `map_add_place` arm of `perform` in
  `engine/work/dispatch.rs` puts a `Place` item; `map_remove_place` lets one go). They do not touch the walkable
  `npc-map` topology at all, and they should not: a Maker drawing a coastline is
  authoring the game's world, not adding a room to the vault it is standing in.
- Runtime topology mutation is the **engine primitive** `MapSet::apply` /
  `World::reshape` (`MapSet::apply` in `npc-map/src/mutate.rs`, `World::reshape` in `npc-map/src/world/bodies.rs`), which reshapes the *walkable*
  map — the rooms and ways bodies actually move through. Its trigger is not a Maker's
  in-fiction act; it is the **operator / embedder surface** (direct scope, §8.3): an
  embedder growing the world it attached (through the public `World::reshape`), or an
  operator editing the running vault (over the wire).

The operator wire surface is a single effector route, `POST http://local/reshape/:world`
(`npcd/src/effector/reshape.rs`), **gated on `Scope::Direct`** — an as-npc token is
`403`, so a Maker cannot reshape the vault it stands in by reaching for its own device.
The edit is applied by the world's authority as `WorldOp::Reshape` (`reshape_here` in
`effector/reshape.rs`), so it is journaled once (`docs/npc_conversations.md` §5.1.1).
It is not in the near-you index (a character is never shown a way to reshape its world),
and it is its own prefix, distinct from the Makers' `/map` cartography station (which
authors lore). The body is a `MapEdit` in its adjacently tagged wire form
(`{"op":"add_node","with":{…}}`); a malformed one is a prescriptive `400`, an edit the
world refuses a `409` in its own words, and a success reports who it relocated and whether
it is durable.

Persistence is `npcd/src/world/mapstore.rs`, invoked by `Hosted::reshape` (`npcd/src/world/mod.rs`)
after the in-RAM swap: it keeps the one area the edit touched as `<mind>/map/<world>/<area-id>.yaml`,
or removes that file for a drowned area, through a `Shelf` (`sim/shelf.rs`). A map in the mind is
kept as a write of the character's owner whose token asked for it (`docs/replicated_substrate.md`
§10.3): their draft over a mind under git (`durable: "drafted"`, the authored map once they
publish it), the file itself over a mind folder that is not one (`durable: "written"`). Only the
touched file is rewritten — every other authored file is left
byte-for-byte, comments and all — and the derived `exits`/`visible` are `#[serde(skip)]`,
so what is kept is the authored form a fresh load re-weaves. Keeping it is off the world
lock and best-effort: the swap is the source of truth, so a failure is *reported*
(`durable: "failed"`) rather than un-happening a reshape that already took. A world with no
authored directory (a generated or test world), or a map in the mind with no owner to keep
it for, is `durable: "ephemeral"` — the change holds for the run only.

*As-built: implemented and tested end to end — `MapSet::apply` (the transactional
mutation), `World::reshape` (swap, re-derive the shaft, relocate the stranded), the
direct-scope `/reshape/:world` route, and the `mapstore` YAML writeback.*

---

# Appendix G — World-state acts as routes

The generic station mechanism (Appendix C) mounts only the acts an authored `Part`
names in its `at`. The world-state half of `WORLD_ACTS` (`engine/acts/mod.rs`) is not
`at`-bound — its availability is `Always`/`Nearby`/`Embodied`/`AwayFromHome`/`Holding`,
not "at this part" — so those verbs need their own mounting. `Holding` is the
availability of `release`: it is offered only while the body holds an order it took
off a board or a station it claimed (`Within.held`), so the index never lists a verb
that would be refused with "You are not holding anything to give back".

## G.1 One surface, name-addressed — not instance-addressed

An earlier draft of this appendix split these acts by target and mounted the
target-facing ones (`claim`, `operate`, `record_verdict`, …) on a target's *instance id*
(`POST http://local/<ns>/<id>/claim`). Building it made plain that the shape does not fit
the world model, so the design changed and this records the change (design docs are
authoritative; a draft the code disproves is corrected, §AGENTS.md):

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
  `enact_response`). The act's own gating is the route's, `409` on a refusal; an unknown verb is
  `404 unknown_verb`; and because every verb here is a body act, `enact_response`'s `500
  not_of_the_body` is not reached. Availability is not pre-checked — an act the
  room offers nothing to work on refuses itself in the world's own words, the prescriptive
  error a character corrects against (§12).

`read` and `scan` mount as `POST` verbs here rather than as `GET`: both route
through `body::perform` and `read` mutates the reader's read-cursor (which is why
`Choices::Readable` excludes what a body has already read), so they are effectful reads,
not the pure GETs the earlier draft assumed. `engage` is `Embodied` and joins the set
when a fight gives it a posture to take.

## G.3 It does not remove the compiled acts, and that is not a dual path

The same act reaches `body::perform` two ways — as a compiled body act in a turn's
grammar, and as a `/here` route — exactly as an `AtPart` act reaches it both as a
compiled act and as a station route. One implementation (the world's own dispatch), two
front doors: the no-dual-path rule (§AGENTS.md) is kept.

*As-built: mounted. `npcd/src/effector/here.rs` is the `/here` surface; the near-you
index advertises `http://local/here`.*
