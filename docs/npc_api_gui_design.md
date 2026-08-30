# The NPC Engine — API and GUI

## The wire contract and the operator surface

This document is the **normative contract** for `npcd`. It specifies every endpoint, every
object shape, every stream frame, every error, and every GUI page. It is the third of three:
[`npc_mind_design.md`](npc_mind_design.md) says what an NPC *is*,
[`npc_engine_design.md`](npc_engine_design.md) says what we *build*, and this one says what
it *looks like from outside* — over HTTP, over a socket, from Rust, and on screen.

It follows the pattern [`zend_ui_redesign.md`](zend_ui_redesign.md) established: the wire
contract is the source of truth, both a live and a mock implementation satisfy it, and the
GUI is built against the mock before the backend exists.

---

# Part A — Principles and conventions

## 1. Four entry points, one capability set

```
   ┌──────────┐      ┌──────────┐      ┌──────────────┐      ┌──────────────┐
   │   GUI    │      │   API    │      │ TEST HARNESS │      │ Battle Cities│
   │ NpcAPI.js│─────▶│  /v1/*   │      │ tests/       │      │  (embedder,  │
   │          │ HTTP │          │      │  harness.rs  │      │   later ver.)│
   └──────────┘      └────┬─────┘      └──────┬───────┘      └──────┬───────┘
                          │                    │                     │
                          ▼                    │                     │
               ┌─────────────────────┐         │                     │
               │  AsyncEngine        │ the only│                     │
               │  (HTTP adapter)     │ async   │                     │
               └──────────┬──────────┘         │                     │
                          ▼                    ▼                     ▼
               ┌──────────────────────────────────────────────────────────┐
               │  trait NpcEngine — SYNCHRONOUS core                      │
               │  submit · poll · drain · snapshot · pump                 │
               │  no runtime · no allocation on the hot path              │
               └──────────────────────────────────────────────────────────┘
```

**Four entry points, one core.** The test harness is deliberately drawn as a peer of the others
rather than as tooling hanging off the side, because treating it that way enforces something
useful: **the harness may only use what a real consumer can use.** A test that needs a back door
into engine internals is evidence of a capability the API is missing, and the fix is to add it
to the core rather than to widen a test-only seam. Part XIV of `npc_engine_design.md` specifies
the harness; this diagram is why it can exist without a parallel surface.

The core is deliberately **synchronous**. A consumer with its own threads and its own tick loop —
a game later, the harness today — should not have a runtime forced on it, every call boxed, or a
`block_on` invited that stalls behind a ~100 ms decode. Async exists only where a socket does.
See §22.

Two rules follow, and they are the ones most likely to be violated by accident:

**No capability exists at only one entry point.** If the GUI can do it, the API can do it,
and so can an embedder — and so can the harness. The HTTP layer is a *transport over*
`NpcEngine`, never a place where behaviour lives. Any logic that ends up in an axum handler is
logic the crate embedder silently does not get, and that the harness therefore cannot test.

That last clause is the reason the harness earns a place in the diagram: it is a standing check
on this rule. A capability reachable only through HTTP is one the harness would have to spin up
a server to reach, and the moment that feels necessary, the rule has already been broken.

**The GUI has no privileged path.** It authenticates like any other client and calls the same
routes. When a GUI feature needs something the API cannot express, the fix is a new endpoint,
not a back door.

## 2. API-first, web second

zend states the posture by ordering: one axum `Router` of `/v1/*` routes plus websockets, with
the UI mounted as `.fallback(embedded_asset)` — structurally the last resort after every API
route misses. `npcd` states it more strongly, by **separation**: the daemon's router has no
fallback to HTML at all, because it has no HTML. A path that looks like an API route and isn't
returns a JSON 404 with no arrangement needed to make that true.

### The `web` crate

The files live in their own crate, `web/` — the static host and API gateway for the whole
estate, documented in **[`web_gateway_design.md`](web_gateway_design.md)**, which is
authoritative for everything in this section. What matters here is what it means for `npcd`:

- **The console is not this daemon's *router's* problem.** `npcd`'s API router has no HTML in it;
  the console is files, served by the `web` server `npcd` embeds — compiled in, or from disk with
  `--content <dir>`. Either way it is served from the same box and the same checkout as the API,
  which is the point: they are one program and must deploy as one.
- **`npcd` does not authenticate anyone** — see §8.1. The gateway owns sign-in for
  tokera.com, code.tokera.com and bot.tokera.com, and states the identity on `X-Tokera-*`
  headers it strips from every inbound request first.
- **bot.tokera.com** is the console's production hostname.
- **Running the console with no engine** is `npcd` itself: it layers its real routes over
  `web::mock::npcd`, so every surface that is not yet built still answers. That is the second of
  the two mock depths described in §41. `web --authoritative` cannot stand in for this site — the
  gateway holds no npcd files to serve, and the mock answers `/v1`, not `/app.js`.

The rest of this section is the part of that design a reader of *this* document needs.

It has three ways to run and one implementation; a route's `upstream` picks between them:

| | Embedded | Gateway | `--authoritative` |
|---|---|---|---|
| Who runs it | `npcd`, `zend` — as a library | the DMZ box — the `web` binary | the `web` binary, one flag |
| Content | compiled in (`include_dir!`), or `--content <dir>` | read from disk | read from disk |
| `/v1`, `/ws` | `upstream: local` → the `Router` this process supplied | `upstream: http://host:port` → forwarded | every upstream forced local → `web::mock` |
| What it is for | one binary that serves the console and its own API, testable alone | top-level domains land here; daemons live elsewhere | the console on a laptop with no daemon anywhere |

The deployment: `web` on the DMZ box holds every file; `npcd` runs on **192.168.0.6:8081** and
`zend` on **192.168.0.5:8081**. `tokera.com` is the default site, served by the gateway itself.

```yaml
sites:
  - name: tokera                               # tokera.com, battlecities.net
    default: true                              # every unmatched Host lands here
    roots: ["content/tokera", "content/common"]

  - name: npcd                                 # no roots: a pure gateway
    hosts: ["bot.tokera.com", "npcd.localhost", "*.npcd.dev"]
    api:
      - {prefix: /, upstream: "http://192.168.0.6:8081"}
```

**`npcd` forwards whole, console included, and that is a correction.** The gateway used to serve
the console from the DMZ box's checkout while forwarding only `/v1` and `/ws` — which made one
program two deployments. A console edit needed a commit on the daemon box and a pull on the
gateway before anyone could see it, and in between the two disagreed silently: the console was
twice found running against an API it did not match, presenting both times as a bug in the
daemon. Forwarding `/` costs a LAN hop per asset and ties the console's availability to the
daemon's; it buys one source of truth and an edit that is live on a refresh.

`/auth/*` is unaffected. Those routes are registered ahead of site routing precisely so a site
proxying `/` cannot swallow them — sign-in stays the gateway's, for every hostname.

`npcd.localhost` resolves to `127.0.0.1` in every current browser, so reaching a specific site
during local development needs no `hosts` file entry.

Five properties are worth naming because each is a decision:

**`roots` is a list, so the framework has no owner.** `content/common` holds the router,
signals, DOM helpers, live-update discipline and base stylesheet; `content/npcd` holds this
product's pages and palette. A request for `/lib/dom.js` falls through to common while
`/pages/roster.js` does not — one URL tree assembled from two directories, and zend gets the
same framework by naming the same second root rather than by either product depending on the
other.

**Promotion is an edit, not a migration.** `upstream: local` and `upstream: <url>` differ by one
line of YAML, and nothing above the config knows which is in force. Iterating on `npcd` as a
single binary and deploying it behind the gateway are the same build.

**A down daemon is a page, not a stack trace.** Each upstream has its own backoff — 250 ms
doubling to 10 s — and requests inside the window fail fast rather than queueing behind another
connect timeout. A browser gets a self-contained error page (no stylesheet, no script: the thing
that would serve them is what is down) carrying `<meta http-equiv="refresh">` set to the retry
delay, so recovery looks automatic. An API caller gets the same failure as the ordinary
`{error, detail, field}` object. One probe is released when the window expires; nobody restarts
anything. Reaching the upstream *at all* counts as success — a 500 from a live daemon is the
daemon's answer to pass through, not a reason to take the route out of service.

**Upgrades are tunnelled.** On a `101` the gateway stops speaking HTTP and joins the two
upgraded connections, so `/ws/logs` and `/ws/events` work identically through the proxy and
direct. `Connection` is hop-by-hop and stripped from every other response, but it is restated on
a `101` — a handshake without it is rejected by the client as malformed.

**`--authoritative` is a flag, never a fallback.** It rewrites every upstream to `local` and
registers the mock in `web::mock` for each site that ships one. Without it, an API route is
forwarded and a missing daemon is an error; there is no path by which a mock quietly substitutes
for a real backend that failed. A site with no mock — `zend`, whose console is not split out of
its binary yet — says so at startup and returns a `502` naming the site, rather than serving an
empty page that looks like a bug in the GUI.

That gives two mocks, at different depths, and both are wanted:

| | Replaces | Exercises | Used by |
|---|---|---|---|
| `?mock=1` (`api.mock.js`) | the network | the GUI only | Playwright, offline UI work |
| `web --authoritative` (`web::mock`) | the daemon | routing, error pages, the ws tunnel, real sockets | console development against real HTTP |

The server-side mock lives in `web/src/mock/npcd/` — beside the files it serves, for the same
reason `api.mock.js` does: a console and its fixtures ship together.

**`npcd` no longer calls it.** It did, as a `fallback_service` under its own routers, so every
path the real ones had not claimed was answered with invented data — for any character id,
including ones that did not exist. `npcd` now answers its whole `/v1` surface itself, in three
routers (`api`, `ops`, `engine`) with no fallback beneath them, and a path none of them claims
is a genuine `404`. The fixture is still built and still serves `web --authoritative`, which is
what it was written for.

That split is worth stating as a rule, because it is what keeps the console honest:

| | |
|---|---|
| `api` | real today — the corpus, the mind, the cast, the authoring plane, accounts, portraits |
| `ops` | real today — status, telemetry, memory, substrate storage, the log stream |
| `engine` | wired, and honest: **empty** where empty is the measurement, **`null`** where nothing has measured, **`503 no_engine`** where the request asks for work |

Nothing in any of the three answers with something it did not measure.

### Extensionless paths fall back; assets do not

`/npc/42` with no matching file serves `index.html`, so the hash router owns navigation and a
deep link survives a refresh. Anything with an extension 404s instead — a missing `.js` served
as HTML surfaces as `Unexpected token '<'` a long way from its cause.

## 3. Identifiers are strings on the wire

`NpcId` and `InteractionId` are `NonZeroU64` in Rust. **JSON numbers cannot carry them
safely** — JavaScript's `Number` loses precision above 2^53, and the GUI is JavaScript. So:

> Every id crosses the wire as a **decimal string**. Never as a JSON number.

```json
{ "npc_id": "10237749914772934281", "interaction_id": "4471028855119" }
```

zend's live adapter already does `id: String(e.id)` for exactly this reason. Server-side, ids
parse from string and reject on overflow with `invalid_id`. A client that round-trips an id
it received is always correct; a client that does arithmetic on one is always wrong.

## 4. Two clocks

An NPC lives in narrative time and runs in wall time, and conflating them makes both
unreadable. Every timestamped object carries both:

| Field | Meaning |
|---|---|
| `at_ms` | wall clock, ms since Unix epoch — when the daemon did it |
| `world_ms` | narrative time, ms since world epoch — when it happened in the fiction |

`world_ms` comes from the world's `TimeSource`, the trait whose own docs cite "narrative-time
applications" and "fictional clocks". A world may run at any ratio to wall time, may pause,
and may jump. Clients sort displays by `world_ms` and diagnostics by `at_ms`.

The `_ms` suffix convention is zend's (`updated_ms`, `started_at_ms`) and is kept.

## 5. Errors

zend's HTTP layer currently returns bare status codes. That is not enough for an API-first
product, so `npcd` adopts the structured shape `zend-tools` already uses internally:

```json
{
  "error":  "npc_not_found",
  "detail": "no NPC with id 10237749914772934281",
  "field":  null
}
```

Codes are **stable across releases** — clients may key off them. `field` is populated for
validation failures. The full catalog is §21.

| Status | Used for |
|---|---|
| 400 | malformed request, validation failure |
| 401 / 403 | auth |
| 404 | resource does not exist |
| 409 | conflict — duplicate creation, stale world-version, mode change on a live interaction |
| 422 | semantically invalid — e.g. an action-plane belief write |
| 429 | tick queue saturated |
| 503 | model still loading; daemon not ready |

**503 is normal, not exceptional.** Model load takes minutes. Every endpoint that needs
inference returns 503 with `detail` describing the load step until ready, and the GUI renders
a loading overlay rather than an error. This mirrors zend, whose handlers return
`SERVICE_UNAVAILABLE` while `inference` is `None`.

## 6. Pagination

Substrate streams are unbounded by design, so every collection over turns is cursor-paginated:

```
GET /v1/npc/{id}/substrate/layer/memory?limit=100&cursor=eyJ0IjoxNzN9
→ { "items": [...], "next_cursor": "eyJ0IjoyNzN9", "has_more": true }
```

Cursors are opaque, forward-only, and stable across appends. `limit` defaults to 50, caps at
500. Never offset-based: an append-only stream with offsets shifts under the reader.

## 7. Idempotency

Creation endpoints accept `Idempotency-Key`. A repeat with the same key returns the original
result rather than creating a second NPC. Keys are retained 24h. Without a key, creation is
not idempotent and a retry after a timeout may create a duplicate — the header exists so
clients can avoid that.

## 8. Auth and ownership

`npcd` is multi-user. That is a larger change than it sounds: zend is a single-operator
loopback daemon with no concept of a user, and every design decision downstream of "who is
asking" is new here.

### 8.1 OAuth / OIDC — run by the gateway, not by this daemon

> **Superseded in part.** This section originally had `npcd` run its own OAuth dance. It does
> not. The gateway owns sign-in for the whole estate, because one account has to reach
> tokera.com, code.tokera.com and bot.tokera.com and only the single ingress can hold one
> session for all three. See [`web_gateway_design.md`](web_gateway_design.md) §5. What remains
> below — the two credential shapes, CSRF, and the local-development rule — is unchanged.

Users sign in with a third-party identity provider — Google first, any OIDC provider by
configuration. Nothing in this estate sees or stores a password.

The flow (authorization code with PKCE, `state` and `nonce` checked, client secret held only by
the gateway) happens at `/auth/*` on **every** hostname, ahead of site routing. The session
cookie is issued on the parent domain, `Domain=.tokera.com`, so the browser presents it to
bot.tokera.com without this daemon being involved.

What `npcd` sees is the result, on headers the gateway sets and strips:

```
X-Tokera-User       the provider's stable subject id — the only field safe to key on
X-Tokera-Email      display only; an address can be reassigned
X-Tokera-Name
X-Tokera-Picture
```

`GET /v1/me` reports that identity. `npcd` has no `/v1/auth/*` routes, no client secret, and no
key of any kind. A request arriving with `X-Tokera-*` set by a client cannot reach it, because
`web` clears those names on ingress before anything is dispatched — on the gateway *and* inside
`npcd` itself, which matters because `npcd` embeds its API as `upstream: local` and would
otherwise be handed the client's own headers untouched.

The one exception is declared, never inferred: `npcd` calls `web::Builder::behind_gateway`, which
says *this bind address is reachable through the gateway and nothing else*, and only then are the
inbound headers left alone for its router to read. Nothing in the code can check that claim — it
is a statement about the network — so it lives at one greppable call site next to the bind, and
`web/tests/roles.rs` pins both directions: forged headers refused without it, forwarded headers
readable with it.

> **Not a shared signing key.** An earlier design had `npcd` verify an `X-Tokera-Assertion`
> against the gateway's session key. Since that assertion *is* the session cookie, it handed every
> daemon both a replayable 30-day token for each user and the means to mint sessions valid across
> the whole estate — a verifying credential that doubles as a minting credential, which is worse
> than the header trust it was meant to replace. One compromised daemon would have been the entire
> estate. Trusting a header from a peer that cannot be anyone else is the smaller claim, and it
> needs no secret distributed to any machine.

Two credential shapes, deliberately not one:

| Credential | For | Carried as |
|---|---|---|
| **Session cookie** | the GUI | `HttpOnly; Secure; SameSite=Lax` |
| **API token** | scripts, world simulations, embedders | `Authorization: Bearer <token>` |

The GUI uses a cookie so no token ever reaches JavaScript — the session cookie is `HttpOnly`,
which is why the sign-in control has to ask `/auth/me` rather than read it, and which removes
the entire class of XSS-token-exfiltration bugs. Programmatic clients use bearer tokens minted at
`POST /v1/me/tokens`, listed and revocable at `/v1/me/tokens`. Because cookies are used, every
mutating endpoint requires either a bearer token or a `X-CSRF-Token` header matching the
session — cookie-only mutation is rejected.

Local development keeps zend's convenience: `--no-auth` binds loopback and injects a fixed
local user. It **refuses to start on a non-loopback interface**, so the convenience cannot
accidentally become the deployment.

### 8.1a Roles

Three levels, and they are how the daemon decides everything. `web/src/auth/role.rs` owns the
type; every route resolves one before it does anything else.

| Role | Who | May |
|---|---|---|
| `unauthenticated` | the gateway named nobody | read worlds, personalities, and the console |
| `user` | signed in | everything above, plus **their own** characters and profile, plus the hardware telemetry |
| `admin` | named in the config | everything above, plus **edit authored content on disk**, plus the substrate footprint and the log stream |

`Role` derives `Ord` from declaration order, so `role >= Role::Admin` is the whole of an access
check and there is no second spelling of the same question.

**Only an admin may change a file on disk.** Worlds and personalities live in a mind that is
not under version control, so a bad write is not a row to restore — it is prose somebody wrote,
gone. Reading them is open, because the console is mostly a reading tool and the content is
fiction.

**Admin is deployment configuration, and there is no API to grant it.** It lives in
`roles.admins:` in the config and is decided at startup. Whoever can edit that file and restart
the process is already the person who can edit the files it grants power over, so a user record
would be a *weaker* way to grant the same thing. An empty list means nobody can edit anything,
which is the right way for a misconfigured deployment to fail — and it is logged at startup, so
it is not discovered from a 403 an hour later.

An entry names `sub` or `email` and says which:

```yaml
roles:
  admins:
    - sub: "108000000000000000000"    # durable: survives an email change
    - email: someone@example.com      # readable: inherits the provider's reassignment risk
```

`sub` is the account key everywhere else in this estate, precisely because an email can be
reassigned. It is therefore the durable answer — and a bare subject id in a config file is not
a thing a human maintains, so both are allowed. Neither is guessed: an entry that matched
"whichever field it looks like" would be a rule nobody can hold in their head, failing silently
in the direction that grants access.

> **A public repository is a reason to prefer `email` here.** A subject id is a stable
> identifier for a real account, and `npcd/.gitignore` excludes `accounts/` on the grounds that
> git history is the one place you cannot quietly remove something from later. Committing a
> `sub:` line puts the same value in the same history under a different file name. `npcd`'s own
> config therefore names an email — one already public as the author address on every commit —
> and accepts that admin follows the address rather than the account. A private deployment, or
> one that keeps its role table outside the tree the way `auth_file` keeps credentials, should
> prefer `sub`.

**401 and 403 are different answers.** Nobody signed in is 401 — signing in fixes it. Somebody
signed in and not enough is 403 — signing in again will not, and telling them to try is how an
operator loses an afternoon to a permissions problem that was never a session problem. The body
names the role required and the role held; neither is a secret.

`GET /v1/me` reports the caller's role so the console can hide a control the server would
refuse. That is presentation only. The check is in the daemon, on the far side of a network hop
a browser cannot reach.

### 8.2 Ownership is authorization, not substrate scope

Every NPC has an owner. This creates a second scoping concept, and conflating it with the
first would be a serious mistake:

> **`OwnerId` decides who may call. The scope chain `(WorldId, PersonalityId, NpcId)` decides
> what the model gathers.** They are different questions and they are answered in different
> layers.

Ownership is an ACL check in the HTTP layer, evaluated before the engine is touched. Substrate
scoping is a projection concern and knows nothing about users. If ownership ever leaks into
the projection builder, an authorization change becomes capable of altering what an NPC
remembers — which is a bug class nobody wants to debug.

```rust
enum Access { Owner, Editor, Viewer, None }
fn access(user: &User, npc: &Npc) -> Access;
```

| Access | May |
|---|---|
| `Owner` | everything, including delete and share |
| `Editor` | authoring-plane writes, interactions, perception |
| `Viewer` | read state, open interactions, no authoring writes |
| `None` | 404 — never 403 |

**Unauthorized reads return 404, not 403.** A 403 confirms the resource exists, which leaks
the existence of other users' characters. The exception is a resource the user can see but not
modify, where 403 is correct and non-leaking.

Worlds and personalities are owned too, and may be marked `public: true` — a shared world any
user can spawn NPCs into, while each NPC stays privately owned.

### 8.3 Tags and hidden characters

Two independent, ordinary properties. Neither is a security feature and the design is careful
not to look like one.

```jsonc
{ "tags": ["campaign-2", "wip", "moonlight"], "hidden": false }
```

**Tags** are free text on any NPC — plain metadata, used for ordinary filtering.

**Hidden** is a boolean. A hidden NPC is omitted from the default listing. That is all it does.

The two combine through one rule, and the rule is the whole feature:

> A tag filter matches **every** NPC bearing that tag — including hidden ones.

So a hidden character is reached by filtering for a tag it carries. There is no reveal mode, no
unlock, no separate flow. From the outside it looks exactly like someone filtering their
roster, because that is precisely what it is.

```
GET /v1/npc?tag=moonlight
→ visible NPCs tagged "moonlight"  +  hidden NPCs tagged "moonlight"

GET /v1/npc
→ visible NPCs only
```

### The one rule that makes it discreet

**Hidden NPCs are never enumerated, counted, or suggested.**

- No hidden count is returned or displayed anywhere. The roster does not say "3 hidden"; it
  says nothing at all.
- Tag autocomplete is built from **visible** NPCs' tags only. A hidden NPC's tags never appear
  in a suggestion list, a facet count, or a tag cloud.
- A tag filter that matches nothing returns an empty list, indistinguishable from a tag that
  was never used.

You surface a hidden character by knowing a tag it carries. Nothing in the interface reveals
that there is anything to look for.

**No "characters you own" total, anywhere.** `/v1/me` deliberately carries no `npc_count`, and
the profile page shows none. It is the figure that defeats the whole rule by arithmetic: anyone
who can see the roster in front of them subtracts it and learns exactly how many characters
there are to go looking for. Counting only the visible ones would be safe but would disagree
with what its owner knows they have, which reads as a fault rather than a policy — and the
roster already says it better. A field carried "for later" is worse than none, because later it
gets honoured by counting everything.

The same arithmetic applies to the per-world and per-personality `npc_count` on those listings.
It counts **every living character, hidden ones included**, across every owner. Global is the
right scope, because the sentence it answers is global — publishing doctrine reaches every
character of that personality, not only the publisher's.

Including hidden characters is the part that looks wrong and is not. *Excluding* them is what
would breach §8.3: the figure would drop the moment one was hidden, so anybody polling it learns
that a character was just hidden and under which personality — a sharper signal than the roster
gives, because the denominator is smaller. Including them makes hiding invisible here, which is
the whole point of hiding. What remains answers "how many of these exist" and never "how many do
*you* have"; the per-owner total is the one §8.3 forbids, and the roster still refuses to
produce it.

### Why the tag is no longer hashed

An earlier draft hashed the tag client-side so the plaintext never reached the server. That is
dropped, for two reasons that are worth recording so it does not get re-added by reflex.

**It bought nothing against the actual threat model.** The substrate is stored in plaintext by
deliberate decision — the daemon must read it to run inference over it. An attacker with
database access already has the character's beliefs, memories and dialogue; withholding the
word "moonlight" from them is not a meaningful defence.

**It broke the owner's own workflow.** Tags are now ordinary metadata the owner edits, sorts and
reuses. If only a hash is stored, the owner cannot see their own tags in an edit form — which
makes the feature actively worse at the job it exists to do.

What remains is honest: hiding is **discretion, not confidentiality.** It defends against a
character appearing in a roster over your shoulder, in a screenshot, or on a stream. It does not
defend against anyone with server access. The UI therefore says "hidden" and never "private",
"secure", or "locked".

---

# Part B — The API

## 9. Resource map

```
  /v1/status                                  daemon readiness, load progress
  /v1/telemetry                               throughput, VRAM, tick stats

  ─ sign-in lives on the gateway, not here (§8.1) ─
  /auth/login?next=…                          begin OAuth (PKCE), any hostname
  /auth/callback                              complete it, set the estate cookie
  /auth/logout                                end the session
  /auth/me                                    who the browser is

  /v1/me                                      current user, from X-Tokera-*
  /v1/me/tokens                               API tokens: list, mint, revoke
  /v1/me/profile                              self an NPC reads: get, append-revise, history
  /v1/me/unique-name                          the name NPCs address you by

  /v1/generate/description                    random modern-human persona
  /v1/generate/attributes                     random beliefs, relationships, agency
  /v1/generate/npc                            whole NPC, one call
  /v1/generate/{job_id}                       poll a generation job

  /v1/image/generate                          text-to-image job
  /v1/image/{image_id}                        fetch bytes
  /v1/image/models                            available image models, load state
  /v1/image/queue                             drain queue: depth, position, next run

  /v1/commands                                slash-command catalog (schema-described)

  /v1/world                                   list, create
  /v1/world/{wid}                             get, update, delete
  /v1/world/{wid}/time                        narrative clock: get, set, scale, pause

  /v1/personality                             list
  /v1/personality/{aid}                       get, update, delete
  /v1/personality/{aid}/doctrine              the one evolving part of the shared layer

  /v1/npc                                     list, create
  /v1/npc/{id}                                get, patch, delete
  /v1/npc/{id}/tags                           set tags
  /v1/npc/{id}/hidden                         toggle hidden
  /v1/npc/{id}/perceive                       batched perception ingest
  /v1/npc/{id}/relationships[/{entity}]       authoring plane
  /v1/npc/{id}/beliefs[/{belief}]             authoring plane
  /v1/npc/{id}/agency[/{strategy}]            missions, strategies, sub-goals
  /v1/npc/{id}/memory                         consolidated memory
  /v1/npc/{id}/modulation                     affect, threat, curiosity
  /v1/npc/{id}/tick                           force a tick; read tick config
  /v1/npc/{id}/environment                    simulator state, toggle, system prompt
  /v1/npc/{id}/substrate[/layer/{name}]       introspection
  /v1/npc/{id}/projection[/{tick}]            what the gather actually selected
  /v1/npc/{id}/monitor                        metacognition health

  /v1/npc/{id}/interaction                    open one; list live ones
  /v1/interaction/{ix}                        get, end
  /v1/interaction/{ix}/inject                 event into the NPC's inbox
  /v1/interaction/{ix}/stream                 SSE: acts live, narration at tick close

  /v1/tools                                   catalog
  /v1/tools/calibrate                         calibration pass

  /ws/logs                                    structured log stream
  /ws/events                                  daemon-wide event stream (roster live-updates)
```

## 10. Core objects

Normative. Both implementations conform.

```jsonc
Npc {
  "npc_id":       "10237749914772934281",
  "name":         "Varek",
  // Slugs — the names of `worlds/<id>.yaml` and `personalities/<id>.yaml` in
  // the mind. A file's identity is its name; a number beside it would be a
  // second identity for the same document, free to disagree with the first.
  "world_id":       "battle-cities",
  "personality_id": "commander",
  // Joined from the authored document at read time, absent when the slug no
  // longer resolves. Never stored on the record: a copy of the name goes stale
  // the moment the file is retitled.
  "personality_name": "Commander",
  "state":        "active" | "idle" | "asleep" | "suspended" | "tombstoned",
  "tick": {
    "heartbeat_ms": 30000,        // idle metabolism; salience sets this
    "last_tick_ms": 1740300112340,
    "pending_events": 3,
    "salience_gate": 0.42
  },
  "environment_enabled": true,
  "monitor": { "overlap": 0.19, "band": "healthy" | "fixated" | "runaway" },
  "owner_id":   "u_8812",
  "access":     "owner" | "editor" | "viewer",   // the caller's access
  "hidden":     false,                            // omitted from the default listing
  "tags":       ["campaign-2", "moonlight"],
  "portrait":   { "image_id": "img_4471", "origin": "uploaded" | "generated" } | null,
  "persona":    { "description": "…", "origin": "authored" | "generated" },
  "created_ms": 1740200000000,
  "updated_ms": 1740300112340
}

User {
  "user_id":     "u_8812",
  "unique_name": "Wren",              // what NPCs see and address; globally unique
  "display":     "Johnathan",         // account name from the provider
  "email":       "…",
  "avatar_url":  "…",
  "provider":    "google",
  "profile": {                        // the self an NPC reads
    "description": "…",
    "gender":      "Male",            // or "Female", or "" until chosen
    "history":     "…",
    "turn_index":  7,                 // the live profile turn
    "revision":    3
  },
  "created_ms": …                     // no npc_count — see §8.3
}

Act {                                 // one tool call the NPC actually made
  "act_id":   "a_88213",
  "tick":     412,
  "tool":     "speak",
  "intent":   "reassure Wren the eastern line is holding, but hedge — he is not sure",
  "args":     { "to": "Wren" },       // structured, never the final words
  "rendered": {                       // filled by the narrator, may lag the act
    "narrator_input": { "type": "say", "character": "Varek", "text": "…" },
    "text": "The line's holding. For now."
  },
  "observable_in": ["physical", "video_call", "voice_call", "instant_message"],
  "committed": true,
  "world_ms": …, "at_ms": …
}

ImageAsset {
  "image_id": "img_4471",
  "origin":   "uploaded" | "generated",
  "mime":     "image/png",
  "width": 768, "height": 768,
  "prompt":   "…",                  // generated only
  "model":    "flux-schnell-q8",    // generated only
  "seed":     441028,               // generated only
  "created_ms": …
}

GenerationJob {                      // text or image; same lifecycle
  "job_id":  "job_991",
  "kind":    "description" | "attributes" | "npc" | "image",
  "state":   "queued" | "running" | "done" | "failed" | "cancelled",
  "progress": 0.0..1.0,
  "queue_position": 2,
  "eta_secs": 41,
  "result":  { … } | null,
  "error":   { "error": "…", "detail": "…" } | null
}

Command {                            // one slash command, schema-described
  "name":        "damage",
  "group":       "combat",
  "summary":     "Apply damage to the NPC",
  "aliases":     ["hit"],
  "parameters":  { /* JSON Schema — the same shape tools use */ },
  "required":    ["amount"],
  "emits":       "perception" | "interaction_event" | "environment_event"
}

Perception {                        // one element of a perceive batch
  "kind": "description" | "map" | "entity" | "sound" | "internal",
  "salience": 0.0..1.0,             // biases the gather; never gates the write
  "world_ms": 1740300110000,        // optional; defaults to the world clock
  // kind-specific:
  "text":      "…",                 // description, sound
  "zoom":      "tactical",          // map
  "ascii":     "…",                 // map
  "legend":    { "@": "you", "#": "wall" },
  "entity_id": "hess",              // entity
  "observation": "…"                // entity
}

Relationship {
  "entity_id": "hess",
  "display":   "Commander Hess",
  "trust":     -1.0..1.0,
  "affect":    -1.0..1.0,
  "familiarity": 0.0..1.0,
  "last_contact_world_ms": 1740299000000,
  "notes": "…"
}

Belief {
  "belief_id":   "hess_keeps_word",
  "statement":   "Hess is a man of his word",
  "confidence":  0.0..1.0,
  "threshold":   0.85,              // disconfirmation needed to rewrite
  "disconfirmation": 0.30,          // accumulated so far
  "origin":      "authored" | "evidence",
  "under_pressure": true,           // disconfirmation > 0 and rising
  "history": [ { "at_world_ms": …, "confidence": …, "origin": … } ]
}

Strategy {
  "strategy_id": "control_north_road",
  "statement":   "Control the northern trade route",
  "state":       "active" | "finished" | "dormant" | "dead",
  "parent_id":   null,
  "children":    ["secure_bridge", "turn_the_toll_keeper"],
  "progress_notes": [ … ],
  "salience": 0.71
}

Interaction {
  "interaction_id": "4471028855119",
  "npc_id":         "10237749914772934281",
  "mode":           "physical" | "video_call" | "voice_call" | "instant_message",
  "interlocutor":   { "kind": "player"|"npc"|"operator", "id": "…", "display": "…" },
  "state":          "live" | "ended_idle" | "ended_explicit",
  "idle_timeout_secs": 900,
  "idle_remaining_secs": 612,
  "opened_world_ms": …, "opened_ms": …,
  "act_count": 14, "narration_count": 5
}

// (Act is defined above, with the intent/rendered split of §18.)

Narration {                         // the tick-bounded woven summary
  "narration_id": "n_5512",
  "tick": 412,
  "interaction_id": "4471028855119",
  "text": "You ask what he sees; before he can answer he's already moving…",
  "covers_acts": ["a_88211", "a_88213", "a_88214"],
  "world_ms": …, "at_ms": …
}

ProjectionSnapshot {                // what the gather selected on one tick
  "tick": 412,
  "budget": { "total": 16000, "used": 15214 },
  "layers": [
    { "layer": "beliefs", "gathered": 3, "available": 41,
      "tokens": 812, "top_score": 0.88,
      "turns": [ { "turn": 17, "score": 0.88, "tokens": 240, "preview": "…" } ] }
  ],
  "system_prompt": { "mood": "tense", "template": "battlefield_urgency",
                     "sections": ["identity_anchor", "situation", "concerns"] },
  "dropped": [ { "layer": "memory", "turns": 6, "reason": "budget" } ]
}

ToolInfo {
  "name": "speak",
  "description": "…",
  "category": "speech",
  "source": "generic" | "extension",
  "parameters": { /* JSON Schema, from schemars */ },
  "calibrated": true,
  "writes_layers": ["action"]       // never contains "beliefs"
}
```

## 11. NPC lifecycle

```
POST /v1/npc
```
```jsonc
{
  "name": "Varek",
  "world_id": "battle-cities",
  "personality_id": "commander",
  // The record's own field name. A body that says `description` writes a
  // character with an empty persona and no error, because an absent persona is
  // legal — so the two names have to be the same one.
  "persona_description": "Fifty-three, a former staff sergeant.",
  "environment_enabled": null,      // null → default by origin (see below)
  "seed": {
    "relationships": [ … ],
    "beliefs":       [ … ],
    "agency":        [ … ],
    "memory_summary": "…"
  }
}
→ 201 { Npc }
```

`environment_enabled: null` resolves by **origin**: the GUI sends `true`, an API client that
omits it gets `false`. A character created in the GUI has no world attached and would
otherwise perceive nothing; an API caller presumably has its own world simulation and does
not want a second one inventing events underneath it. Clients that care set it explicitly.

```
GET    /v1/npc?world_id=&personality_id=&state=&tag=&q=&limit=&cursor=
GET    /v1/npc/{id}
PATCH  /v1/npc/{id}          { name?, persona_description?, state?, environment_enabled?,
                               heartbeat_ms?, salience_gate?, tags?, hidden? }
PUT    /v1/npc/{id}/tags     { "tags": ["campaign-2", "moonlight"] }
PUT    /v1/npc/{id}/hidden   { "hidden": true }
DELETE /v1/npc/{id}          → 204, tombstones (irreversible)
```

The two single-field routes are `PATCH` underneath — same validator, same record, same
supersession. They exist because the console edits those fields from controls nowhere near the
rest of the form, a tag chip and a checkbox, each saving on the spot; without them the console
would have to send a whole character to add one tag. A route that wrote its own record would be
a second answer to "what may a tag be".

**Every write appends.** One record keyed by `npc_id`, newest wins on replay — an implicit
tombstone, with no delete record to write and none to replay. So an edit does not mutate a
character; it supersedes one, and `revision` advances. Deleting is the same operation with
`state: "tombstoned"`: the record stays, because the id must stay taken and the acts it already
committed still name it.

`heartbeat_ms` and `salience_gate` are **authored settings**, not measurements — the resting
rate an idle character thinks at, and the level below which an event does not wake it. The live
tick figures beside them in the response (`last_tick_ms`, `pending_events`) are the engine's and
are absent rather than zero when nothing has reported. Nothing in the console lets one be typed.

Two behaviours on `GET /v1/npc` carry the whole discretion design (§8.3):

- **Without `tag`**, hidden NPCs are omitted — and nothing in the response hints at them. No
  count, no `total` that includes them, no pagination gap.
- **With `tag`**, the filter matches hidden NPCs too, and they are returned indistinguishably
  from visible ones. `hidden` is present on the object for the owner's own edit form, but the
  listing carries no marker that would let a UI render them differently.

Any tag-enumeration endpoint (autocomplete, facet counts) is built from **visible NPCs only**.
That is a hard rule: a tag suggestion sourced from all NPCs would leak the existence of hidden
ones through the front door the rest of the design closes.

`PATCH` with a changed `description` updates the NPC's identity section **and** enqueues a
portrait regeneration (§29) — unless the current portrait has `origin: "uploaded"`.

`DELETE` tombstones rather than erases: `tombstoned_timelines` already excludes the NPC from
every scan, and keeping the bytes means a mistaken delete is recoverable by an operator with
filesystem access. `?purge=true` performs the irreversible removal and is deliberately awkward.

## 12. The user in the substrate

An NPC talks to *someone*. That someone has a name, a body, a history, and the NPC needs to be
able to read all of it — which means the user is not merely an authenticated caller. **The user
is substrate content.**

### Sign-in writes a profile turn

On first sign-in, and on every profile change, the user's details are appended to the substrate
as a turn on a dedicated `user_profile` group, scoped to that user's timeline.

The trigger is the first request carrying an `X-Tokera-User` this daemon has not seen before —
`npcd` is not part of the sign-in flow, so it cannot hook the callback and does not need to.

```
first request from an unknown sub  → creates the account, writes profile turn #0
GET  /v1/me/profile                 → { description, gender, history, turn_index, revision }
                                      gender is "Male", "Female", or "" — 400 bad_gender otherwise
PUT  /v1/me/profile                 → appends a new turn, tombstones the previous
GET  /v1/me/profile/history         → an index: {revision, live, tombstoned_ms, preview}
GET  /v1/me/profile/history/{rev}   → that revision in full
POST /v1/me/profile/restore/{rev}   → brings it back as a NEW revision — 404 if unknown
```

**The index is summaries, and restoring is an edit.** Two decisions that go together.

An author who edits often has hundreds of revisions, so the index carries a one-line preview
and the prose arrives only when one is picked — otherwise opening the profile page downloads
every paragraph that person has ever written in order to render a list of dates. The GUI is a
single chooser for the same reason: a panel per revision grows without bound and buries the
page under text the reader already knows they wrote, while one control is the same height at
two revisions or five hundred.

Restoring appends rather than rewinds. Moving the counter backwards would leave two different
profiles claiming one revision number, and an NPC citing the earlier would be pointing at text
it never read. So it is `POST` and it is not idempotent: restoring twice yields two revisions,
because the second says *still this* a minute after the first, and both are true.

Kept history is bounded at **200 superseded revisions**, oldest dropped first. The account file
is read into memory whole at start and every entry is a full copy of a profile, so unbounded
history means carrying an author's entire writing history resident forever to support an undo
that reaches back a few steps. Two hundred is far past any real use of it.

### Editing appends and tombstones — it never rewrites

> `PUT /v1/me/profile` **appends a new profile turn and tombstones the previous one.** The old
> turn is never edited and never deleted.

The account file is where this happens, not the substrate. Accounts are authored records kept as
files so they survive a substrate wipe (§8.2), and a profile is part of the account, so its turns
are revisions in that file: `profile` is the live one and `profile_history` holds every
predecessor, each stamped with the `tombstoned_ms` at which it stopped being the answer. A reader
with a turn index can therefore tell which text was live when a given turn was gathered. The
history is not returned by `/v1/me` — it is the larger half of the record and is asked for
rarely — which is why `/v1/me/profile/history` exists as its own route.

For NPC-side timelines the equivalent mechanism is `Substrate::tombstone_turn(timeline,
turn_index)`, which flags a single turn dead replay-order-independently without touching the
timeline it lives on. Same semantics, different store; the three reasons below hold for both.

Three reasons this is append-and-tombstone rather than an update:

**Everything an NPC reads is append-only, and the user is something an NPC reads.** An in-place
edit would be the only mutable turn in a system whose entire design rests on nothing being
rewritten. The projection, persistence and replay paths all assume append-only; a special case
for user details would be a special case everywhere — and the fact that this particular record
happens to live in a file rather than a timeline is a storage decision, not a licence to rewrite.

**KV stays valid.** Once gathered, a profile revision is a sealed turn whose KV is shared by
reference across every conversation that attended it. Editing the source text and re-gathering
under the same identity would leave two timelines citing one turn id with different contents;
appending gives the new text a new revision and leaves those reads intact.

**An NPC's memory of who you were stays true.** If you tell Varek you are a soldier, and later
change your profile to say you are a merchant, the NPC's memory of the conversations where you
were a soldier is not retroactively falsified. The tombstoned turn stops being surfaced as
*current*, but it remains what was true at the time. Rewriting history is exactly the failure
the architecture refuses everywhere else, and the user's own record is no exception.

### The unique name is the address

`unique_name` is globally unique across users and is **the only identifier an NPC ever sees**.
Not the email, not the provider id, not the display name from the OAuth account.

```
PUT /v1/me/unique-name   { "unique_name": "Wren" }   → 409 name_taken if collides
                                                     → 400 bad_unique_name if malformed
```

`PUT`, matching `/v1/me/profile`: the name is a single-valued field being replaced, so setting it
to what it already is must succeed rather than collide with itself. The shape is narrower than a
display name — 2–24 of `[A-Za-z0-9_-]`, no leading or trailing separator — because a person types
it into a tool call, and because confusable spellings of one author's address are impersonation
rather than a typo. Uniqueness is case-insensitive for the same reason.

It matters because the NPC's tools take it as an argument. When Varek sends an image, he sends
it *to Wren* — the tool call carries the name, and the interaction layer resolves the name to a
delivery target. An NPC that does not know who it is talking to cannot address them, and a
system that passes an opaque id gives the model nothing to attend over.

Changing `unique_name` follows the same append-and-tombstone path, and NPCs learn the new name
the way a person would: the next time they gather the profile.

## 13. Generation — personas, attributes, portraits

Creating a believable NPC by hand is slow, and an empty character is a bad starting point.
Everything the creation flow offers is available as API first; the GUI has no generation
capability of its own.

### Random persona

```
POST /v1/generate/description
{ "personality_id": "commander", "world_id": "battle-cities", "hints": { "age_band": "50s", "gender": "any" } }
→ { "description": "…", "seed": 88213 }
```

The persona is written as a **modern-day real human equivalent** — not the fantasy character,
but the person they would be if you met them today. A Loyal Soldier comes back as a
fifty-something former staff sergeant now running a loading dock, precise about time, uneasy
in unstructured conversation.

This is not a stylistic flourish; it is a grounding technique. Models have vastly more
purchase on ordinary contemporary people than on archetypal fantasy roles, and a persona
written in that register produces more specific, less generic behaviour once it is read
through the personality lens. The fantasy framing comes from the immutable core; the human
texture comes from here. Two different jobs, kept apart.

`seed` is returned so a generation can be reproduced or nudged.

### Random attributes

```
POST /v1/generate/attributes
{ "npc_id": "…" | "description": "…", "personality_id": "commander",
  "want": ["beliefs", "relationships", "agency"],
  "counts": { "beliefs": 5, "relationships": 4, "agency": 2 } }
→ { "beliefs": [ Belief ], "relationships": [ Relationship ], "agency": [ Strategy ] }
```

Generated attributes come back **as proposals, not as writes.** Nothing lands on the substrate
until the client confirms with the ordinary authoring endpoints. This keeps one authoring path
rather than two, and it means an operator always reviews what a generator invented before a
character starts believing it.

Every generated attribute carries `origin: "generated"`, distinct from both `"authored"` and
`"evidence"`, so the provenance of a belief is legible forever.

### Whole NPC in one call

```
POST /v1/generate/npc
{ "personality_id": "commander", "world_id": "battle-cities", "with_portrait": true }
→ 202 { GenerationJob }        // kind: "npc"
```

Async, because it fans out into several model calls plus an image. Poll `/v1/generate/{job_id}`
or subscribe over `/ws/events`. The result is a complete `NpcSpec` the client posts to
`/v1/npc` — again, generation proposes and creation commits.

## 14. The image module

Portraits are either uploaded or generated. Generation needs a diffusion model, and that is a
new subsystem with a real constraint attached.

```
POST /v1/image/generate
{ "prompt": "…", "negative": "…", "size": "768x768", "seed": null,
  "model": "flux-schnell-q8" }
→ 202 { GenerationJob }        // kind: "image"

GET  /v1/image/{image_id}                   → image bytes (immutable, cacheable)
GET  /v1/image/models                       → available models + load state
POST /v1/npc/{id}/portrait                  multipart upload
PUT  /v1/npc/{id}/portrait                  { "image_id": "…" }
```

### The models are already in the tree

No external service is required. `candle-transformers` ships `flux` (**including
`quantized_model.rs`**), `stable_diffusion`, `stable_diffusion_3`, and `wuerstchen`, with
working examples for each. The module is a loader, a scheduler, and an API over models that
already exist here.

**Quantized Flux Schnell is the default.** Schnell is a few-step model, so a portrait is
seconds rather than a minute, and the quantized weights are what make it viable at all next to
a 30B MoE.

### It runs as misc work between waves

> **Sizing note.** v1 runs headless (§22), so no renderer competes and the full ~8 GiB slot is
> available — SDXL-class weights are the right default. A later version that runs cognition
> during play will need a smaller alternative (SD 1.5 or Wuerstchen, ~3 GiB); nothing here
> forecloses shipping both and selecting at load.

A diffusion model is not a side process — **it competes for exactly the memory the NPCs are
using.** So it does not run beside the inference loop; it runs *inside* it, at a boundary the
loop already has.

The scheduler's decode quantum is bounded by wall clock (`WAVE_SLICE`, 2 s), so the loop
returns to its top on a predictable cadence no matter how fast steps run. That top-of-loop is
where the creep cohort re-forms, and it is the natural place to hand the card to something
else. The image module registers there as **misc work**:

```rust
/// Extensible between-wave work. Invoked at the top of the scheduler loop,
/// after the quantum closes and before the next wave forms. Returns when it
/// has released everything it claimed.
pub trait MiscWork: Send + Sync {
    fn name(&self) -> &'static str;
    /// Cheap check — is there anything to do? Called every quantum.
    fn has_work(&self) -> bool;
    /// VRAM this run needs. The loop asks the governor for relief first.
    fn want_bytes(&self) -> u64;
    /// Run to completion. `budget` is the reservation actually granted.
    fn run(&self, budget: VramLease) -> MiscOutcome;
}

scheduler.register_misc_work(Arc::new(ImageWorker::new(queue)));
```

This is deliberately a general hook rather than an image-specific one. Consolidation passes,
index rebuilds and calibration sweeps want the same slot: *substantial work that needs the card
to itself, briefly, at a boundary.*

### The cycle: load once, drain the whole queue, evict

The expensive parts of image generation are loading the model and re-warming what was evicted
to make room for it — **not** the generation itself. So the worker never generates one image at
a time. It amortises:

```
   ┌─ quantum closes ────────────────────────────────────────────────┐
   │                                                                  │
   │  1. has_work()?          queue depth > 0                        │
   │  2. relief               governor ladder → Moderate → Critical   │
   │                          experts slot→pinned  ·  KV hot→warm     │
   │  3. load                 host RAM ──PCIe──▶ reserved slot        │
   │                          ~0.6 s for SDXL at 12 GB/s              │
   │  4. DRAIN ENTIRE QUEUE   ████ ████ ████ ████   all pending jobs  │
   │  5. release              slot freed; experts re-warm on demand   │
   │                                                                  │
   └─ next wave forms ────────────────────────────────────────────────┘
```

Ten queued portraits pay one load and one re-warm between them, not ten. This is what makes the
"generate a whole cast" flow viable at all, and it is why the queue is drained rather than
serviced.

Within the drain, jobs run **strictly serialized** — one image at a time. Batching the *queue*
is the win; batching the *diffusion* buys nothing and risks an allocation failure with the
card already at its floor.

### Why the reclaim is affordable

The governor's relief ladder already registers the two classes that matter, and their costs are
very different:

| Class | Relief | Cost to restore |
|---|---|---|
| `Expert` | slot → pinned (`Moderate`), shrink `num_slots` (`Critical`) | **cheap** — already backed by pinned RAM; a DMA already paid on every miss |
| `Kv` | hot → warm evict (`Costly`) | expensive — re-read or recompute |

Experts are the cheap pool, and on a MoE model they are the *large* one: with
`all_resident = num_slots >= total_experts` false — which it is for Qwen3-30B-A3B Q6 on a 24 GB
card — the expert cache is reclaimable at slot-release cost. The ladder's cross-class rule
already allows this: *"only `Critical` reaches into `Expert` (and vice-versa)."*

The consequence is worth stating because it inverts an obvious assumption: **a MoE model is a
better host for this than a dense one.** A dense model has no `Expert` class at all, so the only
reclaimable pool is KV — the expensive one.

### Rules

- **Reclaim completes before the copy starts.** The slot is a managed allocation through the
  governor, so the driver never sees an over-commit. If the ladder returns `Exhausted`, the run
  is abandoned and the queue waits for the next quantum — it never proceeds hoping.
- **No oversubscription, ever.** On Windows, WDDM will silently spill an over-commit to host
  RAM rather than failing; this codebase has measured that outcome at **~3 tok/s**
  (`scheduler/prefill.rs`). Set the daemon's NVIDIA profile to **"Prefer No Sysmem Fallback"**
  so a sizing mistake surfaces as a clean OOM the circuit breaker can act on.
- **Rate-limited, not merely deprioritised.** Each drain costs an expert re-warm — precisely
  the cold-decode penalty `continuous_fair_waves.md` exists to eliminate. Occasional is fine;
  continuous is corrosive. A minimum interval between drains bounds it.
- **Never mid-wave.** Work starts only at a quantum boundary and finishes before the next wave
  forms. NPC ticks are never interrupted by a portrait.

### What the client sees

Start latency is *next quantum boundary + reclaim time*, and reclaim of several GiB exceeds one
slice. So `eta_secs` may be `null` and the state is honestly `queued` with a reason. The UI
shows a progress bar and queue position; it does not promise a deadline it cannot keep.

A deployment with a second GPU sets `--image-device cuda:1`, and none of the above applies —
no relief, no misc-work slot, no re-warm. That remains the recommended production shape.

### Uploads

Multipart, `image/png|jpeg|webp`, 10 MB cap, re-encoded server-side to strip EXIF (which
routinely carries GPS) and normalise to PNG. Uploaded portraits are `origin: "uploaded"` and
never overwritten by a generator.

## 15. Perception

```
POST /v1/npc/{id}/perceive
{ "events": [ Perception, … ] }
→ 202 { "accepted": 12, "tick_scheduled": true, "preempted": false }
```

**Batch is the only shape.** There is no single-event endpoint, and this is load-bearing
rather than an economy: perception is prefill and action is decode, and a batched POST is what
lets fifty world events absorb in one batched prefill across every NPC in a fight while decode
is spent only when an NPC acts. A per-event endpoint would invite clients to destroy that
property one call at a time.

`202`, not `200` — acceptance is not processing. `tick_scheduled` says whether this batch
crossed the salience gate; `preempted` says whether it interrupted an in-flight action.

**Maps replace, descriptions accumulate.** A `map` event supersedes the previous map at the
same `zoom` for that NPC. Twelve stale tactical maps in the gather is twelve chances to act on
a position that no longer exists. Supersession writes a new turn and marks the prior one
distilled — history is not mutated.

Multi-NPC fan-out, for a world event several NPCs witness:

```
POST /v1/perceive
{ "npc_ids": ["…","…"], "events": [ … ] }
→ 202 { "results": [ { "npc_id": "…", "accepted": 3, "tick_scheduled": true } ] }
```

One call, one batched prefill across all of them. This is the endpoint a real world simulation
should use.

## 16. State — the authoring plane

```
GET  /v1/npc/{id}/relationships
PUT  /v1/npc/{id}/relationships/{entity_id}     { trust?, affect?, familiarity?, notes? }
GET  /v1/npc/{id}/beliefs
PUT  /v1/npc/{id}/beliefs/{belief_id}           { statement?, confidence?, threshold? }
DELETE /v1/npc/{id}/beliefs/{belief_id}
GET  /v1/npc/{id}/agency
PUT  /v1/npc/{id}/agency/{strategy_id}
GET  /v1/npc/{id}/memory?limit=&cursor=
GET  /v1/npc/{id}/modulation
PUT  /v1/npc/{id}/modulation                    { affect?, threat?, curiosity? }
```

Every write here is an **authoring** act and is recorded as such — `origin: "authored"` — so
an operator can always tell an authored belief from one the evidence process earned.

> **Where this is stored, and why it is not a record type of its own.** All of it lives on the
> character's own `NpcPayload` and supersedes with it (`npcd/src/npcs.rs`). Three reasons: it is
> operator-scale — tens of entries, not the thousands an engine accumulates; its lifetime *is*
> the character's; and the write path, the supersession rule and the compaction handling are
> the ones already there and already tested. The engine's own belief traffic is a different
> problem with a different volume and will want its own records; this is what somebody types
> when they build a world.
>
> **The engine's half is absent, not zero.** A belief carries the `confidence` and `threshold`
> an operator stated; its `disconfirmation`, whether it is `under_pressure`, and its confidence
> `history` are measurements the evidence process makes, and come back `null` until it has made
> them. Same for a strategy's `salience`. A `disconfirmation: 0` would read as *weighed and
> unshaken*, which is a claim a daemon with no engine cannot make.

> **The invariant these endpoints depend on.** The belief write-protection in the mind
> document is against the *model*, not the *operator*. The action plane — what the NPC's own
> decode can emit — has no path to the belief layer at all. The authoring plane does, because
> someone has to say what a character believes when the world is built.
>
> The engine enforces this at the registry: a tool declaring `beliefs` in `writes_layers` is
> rejected at registration with `tool_writes_protected_layer`, and the generic catalog
> contains none. An attempt to reach these endpoints from a tool context returns **422
> `action_plane_belief_write`**.

Relationships carry no such restriction on either plane — a relationship is a calibration
trajectory that is meant to move easily. That asymmetry is the reason both layers exist.

## 17. Interactions

```
POST /v1/npc/{id}/interaction
{ "mode": "physical", "interlocutor": { "kind": "player", "id": "p_17", "display": "Ilse" },
  "idle_timeout_secs": 900 }
→ 201 { Interaction }
```

Opening one forks the NPC's substrate: `fork_resuming(timeline_of_interaction(npc, ix))`
against the `interaction` layer, inheriting the sealed prefix by reference. The tenth
concurrent interaction with a popular NPC costs a suffix, not a mind.

```
GET    /v1/npc/{id}/interaction              live interactions for this NPC
GET    /v1/interaction/{ix}
DELETE /v1/interaction/{ix}                  → 204, ends explicitly (archives)
POST   /v1/interaction/{ix}/inject           { "text": "…" } | { "event": Perception }
```

**Mode is immutable for the life of an interaction.** A `PATCH` that changes mode returns
**409 `mode_change_forbidden`**; the client ends the interaction and opens another. Structural
mode cannot change mid-decode without breaking coherence — and in the fiction, hanging up and
walking over really is a different encounter.

Mode sets the observability envelope, which is what the narrator filters by:

| Mode | Observable | Not observable | Extra tools |
|---|---|---|---|
| `physical` | speech, movement, gesture, expression, ambient acts | internal broadcasts | — |
| `video_call` | speech, expression, framed gesture | movement out of frame, ambient | `send_image` |
| `voice_call` | speech, audible action | all visual acts | — |
| `instant_message` | speech only, text-shaped | everything else | `send_image` |

Scoping is **by observability, not relevance**. An NPC breaking off to look east reaches a
physical observer as a turn of the head and a voice-call observer as a pause — one act, two
vantages. Stripping it as "irrelevant" would strip out exactly the texture that sells the NPC
as a person with a life rather than a presence that exists only when addressed.

### Mode gates the tool catalog, not just the view

Mode is an input to **which tools exist for this interaction**, resolved when it opens:

- **`send_image` is a messaging-mode tool.** In an instant-message or video-call interaction an
  NPC can send a picture, because that is a thing people do over those channels. In a physical
  encounter it is not offered — a character standing in front of you does not text you a photo.
  The tool is absent from the catalog rather than present-and-refused, so the model is never
  invited to attempt it.
- **Physical mode instead gets narrator-driven scene imagery** (§34), which is generated by the
  surface rather than called for by the NPC.

`send_image` takes the interlocutor's **unique name** as its target:

```jsonc
send_image { "to": "Wren", "intent": "show her the ridge where the fighting is" }
```

The name is required and validated against the interaction's interlocutor. An NPC addressing a
name that is not in this interaction gets `invalid_arguments` — which is also what stops a
character in one conversation from delivering an image into another.

**Ending archives, it does not delete.** `TimelineEntry::archived` already excludes a timeline
from `active_timelines_for_group`. The turns stop being gathered but survive for consolidation
to fold into `memory` on the sleep clock — the mind document's *soft fade by non-selection*,
reversible and cue-resurfaceable, as opposed to the hard forget that belongs only to the sleep
fold.

Idle defaults per mode, because silence means different things:

| Mode | Default idle timeout |
|---|---|
| `physical` | 5 min |
| `voice_call` / `video_call` | 10 min |
| `instant_message` | 24 h |

## 18. Intent and narration — the NPC never writes its own words

This is the sharpest expression of the mind document's rule that the interaction layer
*narrates acts and never fabricates*, and it changes what a tool call contains.

> **The NPC emits intent. The narrator writes the words.**
>
> `speak` does not carry a sentence. It carries what the character means to convey. A separate
> narrator pass renders that intent into actual prose, in the character's voice, in the register
> the current mood and template select.

```
   ACTION LAYER (the tick)                    NARRATOR
   ┌──────────────────────────┐              ┌──────────────────────────────┐
   │ speak {                  │              │  NarratorInput               │
   │   to: "Wren",            │─── intent ──▶│  { "type": "say",            │
   │   intent: "reassure her  │              │    "character": "Varek",     │
   │   the line holds, but    │              │    "text": "The line's       │
   │   hedge — he isn't sure" │              │     holding. For now." }     │
   │ }                        │              └──────────────┬───────────────┘
   └──────────────────────────┘                             │
                                                            ▼
                                              rendered prose → the interaction
```

### It is the module that already exists

`candle-conversation::narrator` is built for exactly this. Its wire format is a tagged enum
carrying a character name and a structured event:

```rust
pub enum NarratorInput {
    Say   { character: String, text: String },
    Act   { character: String, action: String },
    Scene { description: String },
    Cue   { character: String, action: String },
    Beat  { description: String },
}
```

and its documented contract is that the model "responds with narrative prose only."
`SessionConfig` already carries a **protagonist** and **persona** name — the same
unique-name concept §12 makes first-class. The engine wires the NPC's act stream into
`NarratorInput` and lets the existing narrator do what it was written to do.

### Why intent rather than text

**It keeps the can't-lie guarantee under a second model.** If the action layer wrote final prose
and the narrator merely relayed it, the narrator would be decoration. If the narrator invented
content, it could drift from the acts. Intent-in, prose-out means the *substance* is decided by
the mind and the *wording* by the surface, and neither can produce something the other did not
license.

**One voice, many registers.** The same intent renders differently by mode: terse over instant
message, unhurried in a physical scene, clipped on a bad voice line. If the action layer wrote
words, mode would have to be an input to cognition rather than to presentation.

**The act stream stays readable.** An operator watching intents sees what the character *means*,
which is far more diagnostic than watching finished sentences — and it is the thing that
explains a strange reply.

### Images work the same way

```
send_image { to: "Wren", intent: "show her the ridge where the fighting is" }
```

The NPC does not write an image prompt any more than it writes a sentence. The narrator turns
the intent into a scene prompt, and the image module (§14) renders it as queued misc work.

This preserves the same property: the NPC cannot send a picture of something it did not mean to
show, and the image is grounded in a committed act rather than generated beside the
conversation.

### What the client sees

`Act` carries `intent` and `args` immediately; `rendered` is filled when the narrator completes
and **may lag by one beat**. The `act` SSE frame fires on commit, and a later `act_rendered`
frame carries the prose:

```
event: act            data: { act_id, tool: "speak", intent: "…", rendered: null }
event: act_rendered   data: { act_id, rendered: { narrator_input, text } }
```

A messaging client waits for `rendered` before showing a bubble. The operator console shows
intent instantly and prose when it arrives — which is the two-lane property again, one level
down.

## 19. The interaction stream

```
GET /v1/interaction/{ix}/stream          → text/event-stream
```

Named SSE events, following zend's `event: status | projection | tool` convention.

```
event: open
data: { "interaction_id": "…", "mode": "physical", "resume_from": null }

event: act            # commit; carries intent, rendered=null
data: { Act }

event: act_rendered   # narrator finished; carries the prose
data: { "act_id": "…", "rendered": { "narrator_input": …, "text": "…" } }

event: tick
data: { "tick": 412, "window_closed_world_ms": …, "acts": 3 }

event: narration
data: { Narration }

event: status
data: { "text": "gathering" }

event: state
data: { "state": "ended_idle" }

event: error
data: { "error": "…", "detail": "…" }
```

**Two streams at two latencies, on one connection.** `act` frames arrive live as each act
commits; `narration` arrives at tick close, after the `tick` frame that names the window it
covers. This is not an optimisation to be collapsed — it is the property that makes the
interaction read as watching a person act and then explain themselves, rather than as a
chatbot taking turns. A client that buffers acts until narration arrives has thrown away the
feature.

`resume_from` supports reconnection: `?since_act=a_88213` replays what was missed. Streams are
resumable for 5 minutes after disconnect; after that the client re-fetches state and reopens.

Heartbeat comment frames every 15s keep intermediaries from closing an idle stream.

## 20. Environment, tools, introspection, worlds

### Environment simulator

```
GET  /v1/npc/{id}/environment
→ { "enabled": true, "system_prompt": "…", "window_turns": 24,
    "recent": [ { "world_ms": …, "text": "…" } ] }
PUT  /v1/npc/{id}/environment      { "enabled"?, "system_prompt"?, "window_turns"? }
POST /v1/npc/{id}/environment/inject   { "text": "…", "world_ms"? }
```

Its own system prompt and a sliding window — `Sequence { recent: N }`, no historical top-k,
because the environment's job is continuity of the immediate scene rather than recall of
everything that ever happened. Long-run world memory belongs to the `world` layer, which the
simulator writes into.

### Tools

```
GET  /v1/tools?source=&category=
→ { "tools": [ ToolInfo ], "uncalibrated": 2 }
POST /v1/tools/calibrate            → 202 { "job_id": "…", "tools": ["…"] }
GET  /v1/tools/calibrate/{job_id}   → { "state": "running"|"done", "progress": 0.4 }
```

Extension tools are registered **through the crate**, not over HTTP — a tool is a Rust closure
with typed parameters and cannot be posted as JSON. The API surfaces the catalog and drives
calibration.

`calibrated: false` is reported honestly rather than hidden. A tool registered before the
calibration phase is calibrated with the rest; one registered while the engine is live is
usable but selects worse until the next pass, because tool selection quality comes from the
`examples` prefilled into the reserved calibration layer.

### Introspection

```
GET /v1/npc/{id}/substrate
→ { "layers": [ { "layer": "memory", "turns": 4412, "tokens": 918233,
                  "window": 8000, "resident": 61 } ] }
GET /v1/npc/{id}/substrate/layer/{name}?limit=&cursor=
GET /v1/npc/{id}/projection            latest ProjectionSnapshot
GET /v1/npc/{id}/projection/{tick}     a specific one (retained N ticks)
GET /v1/npc/{id}/monitor?window=100
→ { "band": "healthy", "overlap": [ { "tick": 410, "value": 0.19 } ],
    "thresholds": { "fixated": 0.35, "runaway": 0.55 } }
```

Introspection is a **product surface, not a debug afterthought**. Every open question in the
mind document is a calibration question answerable only by watching real runs, and
`/v1/npc/{id}/projection` — *what did this NPC actually gather on that tick* — is the
instrument that makes answering them possible.

### Worlds, personalities, narrative time

> **A world is a tag-filter over one shared corpus**, not a corpus of its own —
> see `docs/npcd_worlds_and_layers.md` for the full design. In short: canon is
> ingested tagged with its world and is visible only to it; craft (responses,
> moods, identities) is ingested untagged and is shared by every world, sharing
> its KV as well as its text. There is therefore **no** create button for either
> in the GUI — a world and a personality are authored YAML files in the mind
> directory beside the corpus they index, and an empty world would project
> nothing. Both listings are read-only in that one respect: they show what the
> mind holds, and gain an entry when an author writes a file.

**A world may name its cast.** Craft being shared by every world is the standing default and
stays it — a world that declares no cast admits every personality, which is what keeps adding
the key to one world from emptying the rest.

```yaml
# worlds/<id>.yaml
personalities: [cindy-tan]        # only these may be created in this world
```

It sits on the world, beside `selects` and `excludes`, for the reason those do: **a world is a
filter, and what it admits is written on the world.** One file answers "who belongs to this
setting", rather than the answer being assembled by reading seventy-four personality files.

This is **not** the `hidden` flag and does not overlap with it. `hidden` answers "should this
appear in a listing" — a question about screen shares, revealed by naming the document in the
filter or by an admin holding RIGHT ALT. A cast answers which world a character is *of*, which
no keypress should be able to change. The two are read at different moments and neither
substitutes for the other.

The daemon is the authority: it refuses a create naming a personality the world does not cast
(`personality_not_of_world`), so the form's filtering exists to keep a refused pairing from
being offered, not to enforce anything.

```
GET        /v1/world                     GET|PUT|DELETE /v1/world/{wid}
GET|PUT    /v1/world/{wid}/time          { "world_ms", "scale", "paused" }
GET        /v1/personality               GET|PUT|DELETE /v1/personality/{aid}
GET|PUT    /v1/personality/{aid}/doctrine
```

Every `GET` here is open; every `PUT` and `DELETE` needs `admin` (§8.1a). A document larger
than 256 KiB is refused — twenty times the biggest real personality, and small enough that the
API is not a way to fill a disk one save at a time. A name already taken by something that is
not a plain file is refused rather than followed, because `write` follows a symlink.

### The corpus — browsing and editing everything that was authored

The documents above are the ones with a *shape*: a world and a personality are parsed,
validated, and patched key-by-key so an author's comments survive a save. That covers two
folders. The mind holds far more that has no schema at all — **1,818 pages** of canon, the 596
responses and 116 moods, and the settings — and until this existed the only way to change any
of it was a text editor on the machine the daemon runs on.

```
GET    /v1/mind/list?world=&id=        what is inside a place
GET    /v1/mind/entry?id=              { id, title, text, chars }
PUT    /v1/mind/entry?id=&new=1        { text }  →  201 created / 200 updated
DELETE /v1/mind/entry?id=              204
GET    /v1/mind/fields?id=             { id, title, fields[] }  —  422 not_fields
PUT    /v1/mind/fields?id=             { values }  —  409 cannot_patch
```

**`id` is an address, not a path.** `canon/ammo/bolt`, never
`layers/world/ammo/bolt.md`. This is the whole design and it is worth being plain about why:
an API that says the second has published its storage as its contract, and every token in it
is a promise — that canon lives under `layers/`, that a topic is a directory, that prose is
markdown. `npcd/src/mind/address.rs` is the only place that knows any of that.

An address is a **section** and a chain of **names**. There are nine sections and a client
cannot invent a tenth:

| Section | Holds | Stored as |
|---|---|---|
| `canon` | the setting itself | Markdown |
| `agency` `beliefs` `memory` | what characters want, hold true, remember | Markdown |
| `responses` `moods` | the shapes and registers of a reply | YAML |
| `characters` `worlds` | who they are, and where | YAML |
| `settings` | how the mind is configured | a named set |

Three things follow, and each removes a class of mistake rather than merely tidying:

- **The section supplies the extension**, so a caller never states a storage question and can
  never get it wrong. `canon/x` becomes Markdown; `characters/x` becomes YAML. There is no
  address that can name an executable, because there is nowhere in an address to put one.
- **Anything that is not a section is unaddressable.** `node_modules`, the daemon's own
  `.substrate`, a scratch folder — these are not filtered out of a listing, they cannot be
  *named*. That is a stronger guarantee than a deny-list, and it does not need maintaining.
- **A topic is one thing.** `ammo.md` is the overview and `ammo/` holds the entries — one idea
  stored as two files — so `canon/ammo` addresses both: listing it gives the entries, reading
  it gives the overview, and `has_text` says whether there is one. Nothing suggests two files,
  because to a reader there are not.

**Reading needs `user`, not `unauthenticated`** — the one place this surface is stricter than
the documents above, and the reason is enumeration. `GET /v1/personality/cindy-tan` answers
somebody who already knows the id; listing hands out the corpus a level at a time, which is
exactly the browsing `hidden` exists to prevent. Writing needs `admin`, like every other change
to something on disk.

**A world is a lens, and the daemon holds it.** `?world=` narrows by that world's own three
fields — `selects` gates the canon topics, `excludes` gates the section categories in
`responses` and `moods`, the cast gates `characters` and the per-character `beliefs` and
`memory`. The filter is applied as each level is read, so anything a world excludes is never
named on the wire, and addressing an excluded topic directly is refused rather than served.
Omitting `?world=` shows the whole corpus, which is what editing it wants.

A canon topic exists twice on disk and `selects` names it once, so the extension comes off
before the comparison. Getting that wrong hid 37 of a real world's 66 topics while leaving
every folder in place.

### A document as fields, not as a file

`/v1/mind/entry` hands over a document's text, and for prose that is the right answer — a canon
page is prose from its first byte to its last. A section is not. `responses/accept_then_move_on`
is five keys, one of which is sixteen four-turn conversations, and showing that as a textarea of
YAML asks the person editing the wording of a reply to also be a serialisation format's
proof-reader: their job becomes indentation, block scalars, and not breaking the `examples:`
list.

So `/v1/mind/fields` answers the same document as a list of fields, in the order the file writes
them, and `422 not_fields` for one that is not a mapping. Each field is a `key`, a `label`, a
`kind`, a `value`, and a `note`:

| Kind | Value | Edited as |
|---|---|---|
| `line` | a short string | an input |
| `text` | prose | a textarea |
| `number` | a number | a numeric input — and a **number** on the way back |
| `bool` | true or false | a checkbox |
| `choice` | one of a fixed vocabulary | a select |
| `list` | short strings | chips, add and remove |
| `conversations` | `[{ note, turns: [{ role, content, thinking }] }]` | the conversation editor |
| `group` | a mapping | these same controls, nested |
| `rows` | a list of mappings | one titled card each |
| `raw` | anything unmodelled | that value's YAML, and only that value |

`group` and `rows` make the form recursive, and that is what carries a value
with structure. A projection layer's `budget` is a priority and a ceiling and
sometimes an `adaptive` pair inside that; its `groups` are a list of mappings
with a `selection` inside each. Flat, they are two YAML boxes. As themselves,
they are a dozen inputs with names on them.

`number` is not cosmetic. A window typed into a text input comes back as the
string `"8000"`, which is a different document that looks identical — and the
splice below would faithfully write the quotes.

`choice` is offered **only where the value is already one of the vocabulary's
words**. `gather_scope` and `decode_priority` are fixed by the engine and a typo
in either is a layer it cannot load, so a select is right; but these are also
ordinary words, and a document elsewhere with its own `kind` must not be told
its value is invalid by a form that has never heard of it.

`raw` is the honest escape hatch. A form that silently dropped the part of a document it did not
understand would be worse than one that admits it, so an unmodelled shape round-trips as its own
YAML text and a malformed edit is refused with its key named — never written out as a quoted
string that changes the value's type.

**The `note` is the author's own comment.** 701 of the 712 section files carry one above a key,
and they are the best documentation the corpus has — *"FIXED SHAPE: 4 turns, user → assistant →
user → assistant. Final assistant turn is the decode point"*. The field carries it, so the
guidance is where the editing happens instead of only in a file nobody opens.

**A save patches; it never re-serialises.** The values go through
`npcd/src/registry/yaml_edit.rs`, which compares the new document to the old one *all the way
down* and emits an op at the deepest node that differs — editing one turn's wording is a change
to `examples[1].turns[0].content`, so the diff is that block scalar and every other byte of the
file is untouched. Where the shape itself changed, and there is no entry-for-entry
correspondence left to walk, that collection is rewritten as **block** YAML in the key order the
file already used, with prose as the literal blocks it was written as. Both halves matter for
the same reason: a save whose diff is the whole file cannot be reviewed by the person whose file
it is.

The result is then parsed and compared to what was asked for, and a mismatch **refuses the save**
with `409 cannot_patch` rather than falling back to writing the document out whole. That fallback
is precisely what would cost the file its comments, which is the thing this path exists to
protect; the console offers the text editor instead.

### A document can have addressable parts — the projection layers

Most of the corpus is a file per thing. The projection schema is not: its **nine
layers** live in one seven-hundred-line document, and each is a window, a score
threshold, a budget, a summarisation prompt and a set of selection groups. As
one document the only way to change a layer's budget is to find it in a
textarea.

So a settings document may declare that one of its keys holds *parts*, and each
part gets an address:

```
settings/projection            the schema — lists its layers, and reads whole
settings/projection/world       one layer, which opens as fields
```

This is the same idea as a canon topic having both entries and a body. Nothing
about the routes is special: `list`, `entry` and `fields` all work through the
address, and `npcd/src/mind/parts.rs` is the only place that knows a layer is an
item in a list rather than a file.

**A save is spliced into the document that holds it.** The whole schema goes
through `yaml_edit` with just that layer replaced, so changing one layer's
`window` is a one-line diff and the other six hundred and ninety lines are the
bytes they were — including the `# ── Environment ──` banner the author wrote
above each layer.

**A part can be edited but not added or removed**, and the address model is what
enforces it: an address names a part that exists, so there is no address for a
tenth layer and none for deleting the ninth. That is deliberate rather than
missing. Adding or removing an item changes the length of the list, which leaves
no entry-for-entry correspondence to walk and so rewrites the list whole —
taking every banner comment with it. Adding a layer is an act for the whole
document, where the author can see the comments they are moving; `settings/projection`
still opens as text for exactly that.

Delete on a part is refused rather than falling through to the document. A layer
is not `projection.yaml`, and deleting the schema because somebody pressed Delete
on a layer would be the worst kind of surprise.

**Three rules bound what can be touched**, enforced in `npcd/src/mind/`:

| Rule | Why |
|---|---|
| Names still have to survive becoming a file name | `..`, `\`, `:`, NUL, control characters and reserved device names are refused, and the resolved path is checked to still be under the mind — an address is a nicer spelling of a path, never a way around one |
| Writes are atomic | a temporary beside the target, `fsync`, then rename — the mind is not under version control, and a truncating write that is interrupted leaves the file *gone* rather than unchanged |
| Deleting takes only what was named | removing a topic's text keeps everything inside it; those have addresses of their own. One button must never become a recursive delete |

### Hidden documents, and the whole word that reveals them

An authored document may carry `hidden: true`. It is then left out of listings and revealed by
typing a **whole word** of its id or name into the filter — `earth` reveals `earth`; `ear` does
not. Both rules are server-side (`npcd/src/visibility.rs`), because a hidden document is never
sent: a client-side filter would have nothing to reveal however completely it was typed.

The listing has two rules on purpose. A **visible** document narrows on an ordinary
case-insensitive substring, which is what a filter box should do. A **hidden** one needs a
complete word, because a substring would reveal it one letter at a time — and that incremental
discovery is the entire thing the flag prevents. It is what makes filtering-as-you-type safe
here.

**This is discretion, not access control**, and the code says so in those words. Anyone who
knows the id can still `GET /v1/world/earth` — which is the same act as typing the word. What it
buys is that the content is never *offered*: not in a dropdown, not in an autocomplete, not in a
screenshot. It is also deliberately role-independent: an admin sees the same listing as anybody
else, because the moment that matters is a demo, and during a demo the person at the keyboard is
an admin.

### A world admits a subset of the shared craft

```
GET  /v1/world/{wid}/collections
```

The response and mood libraries — 596 and 116 files in the mind — are ingested **untagged** and
shared by every world, text and KV both. Which of them reach a given world is therefore a
projection of one library rather than a second copy of it, and a world declares what it does not
admit:

```yaml
selects: [combat, lore, …]     # canon, by tag
excludes: [sexual, intimate]   # craft, by section category
```

A world that names nothing admits everything. The response reports `excluded` alongside the
sections it kept, so a collection that is 546 of 596 says why rather than being quietly short.

> This route used to fall through to the console's fixture, which answered with six invented
> response templates and five invented moods — unlabelled, on the page somebody would open to
> look at their own library. The reasonable conclusion from that screen was that seven hundred
> files had been lost. Nothing had; the daemon never read them. A fixture that is not obviously
> fake, standing where the real thing belongs, is the worst shape one can take.

Sections report `chars`, not `tokens`. There is no tokenizer in this daemon, so a token count
would be a plausible-looking guess — the same habit that produced the paragraph above.

A `PUT` **replaces** the document, so a client sends back what it read rather than the fields it
changed — sending only the doctrine would blank the anchor and every trait. The id comes from the
URL and the body's own is discarded: a document that could name its own file could name somebody
else's. `npc_count` is discarded the same way and for the same reason — it is computed at read
time, and the obvious client sends back everything it was given, so a derived value has to be
kept out of the file by the server rather than by the caller's good manners.

> **Replacing the document does not mean rewriting the file.** The registry *edits*
> (`npcd/src/registry/yaml_edit.rs`): the file on disk is the base, only the values that actually
> differ are re-rendered, and every other byte — comments, blank lines, block scalars, key order,
> the author's line wrapping — is copied through. A save that changes a name changes one line.
>
> This is not cosmetic. An authored world or personality carries its reasoning in its comments,
> and none of that is data, so a `serde_yaml` round-trip silently deletes the half of the document
> a person wrote — a file that still loads perfectly and has lost everything that explained it.
> The base is re-read from disk at save time rather than taken from memory, so a file edited by
> hand while the daemon is up keeps that edit in every field the console did not touch.
>
> The editor verifies its own output before returning it: it parses what it produced and compares
> it to what was asked for, falling back to a whole-document serialisation (with a loud log) if
> they differ. The worst case is losing comments; it is never a file that says something the
> author did not.

Doctrine is the **only** part of the personality that changes — identity never propagates, lived
experience never aggregates. `PUT` bumps a version; characters of that personality pick it up at
next spawn or fork refresh.

`PUT /v1/world/{wid}/time` is the narrative clock: set an instant, set a scale (`0` pauses,
`1.0` is real time, `60.0` is a minute per second), or jump. Every NPC in the world sees it.

## 21. Error catalog

| Code | Status | Meaning |
|---|---|---|
| `invalid_id` | 400 | id not a decimal string, or overflows u64 |
| `invalid_arguments` | 400 | schema or validator failure; `field` set |
| `unauthorized` | 401 | missing/bad bearer token |
| `npc_not_found` | 404 | |
| `interaction_not_found` | 404 | |
| `forbidden` | 403 | signed in, and not the role this needs; carries `required_role` and `role` |
| `world_not_found` / `personality_not_found` | 404 | |
| `unknown_world` / `unknown_personality` | 400 | a character named a document the mind does not hold |
| `write_failed` | 500 | the document could not be written; the reason is in the daemon log and deliberately not in the response |
| `tool_not_found` | 404 | |
| `duplicate_npc` | 409 | idempotency key reused with different body |
| `mode_change_forbidden` | 409 | mode is immutable for a live interaction |
| `stale_world_version` | 409 | arbiter rejected a world-mutating act |
| `interaction_ended` | 409 | inject or stream on an ended interaction |
| `action_plane_belief_write` | 422 | belief write attempted from a tool context |
| `tool_writes_protected_layer` | 422 | registration declared `beliefs` as a write target |
| `npc_tombstoned` | 422 | |
| `tick_queue_saturated` | 429 | shed load; `Retry-After` set |
| `model_loading` | 503 | `detail` names the load step |
| `engine_not_ready` | 503 | |

## 22. The Rust surface

There are **two** embedder surfaces, and the sync one is the foundation.

An earlier draft made `#[async_trait]` the core and treated everything else as a wrapper. That
is the wrong way round. Battle Cities has its own thread architecture and its own tick loop, and
an async-first core imposes three costs a game will not accept:

- **A runtime it did not ask for.** Embedding the engine should not mean embedding tokio.
- **An allocation per call.** `#[async_trait]` boxes every future. In a loop touching hundreds
  of NPCs per frame that is heap traffic for nothing.
- **A blocking hazard on the frame thread.** `block_on` inside a frame stalls it, and a decode
  step is ~100 ms — two orders of magnitude past a frame budget. The one thing a game engine
  must never do is wait for inference on the thread that draws.

So the core is **synchronous, non-blocking, and runtime-free**, and async is a thin adapter over
it for HTTP.

```
        ┌────────────────────────────────────┐
        │  axum / HTTP  (async adapter)      │   ~200 lines, no logic
        └──────────────────┬─────────────────┘
                           │
        ┌──────────────────▼─────────────────┐
        │  NpcEngine — synchronous core      │◀── Battle Cities embeds this
        │  submit · poll · drain · snapshot  │    directly, no runtime
        └────────────────────────────────────┘
```

### The core: submit, drain, snapshot

Every method returns immediately. Nothing blocks on inference. Nothing allocates on the hot
path.

```rust
pub trait NpcEngine: Send + Sync + 'static {
    // ── commands: submit and return, never block ──────────────────────────
    /// Queue a perception batch. Returns at once; the tick consumes it later.
    fn submit_perceive(&self, npc: NpcId, events: &[Perception]) -> Ticket;
    fn submit_perceive_many(&self, npcs: &[NpcId], events: &[Perception]) -> Ticket;
    fn submit_inject(&self, ix: InteractionId, ev: InboxEvent) -> Ticket;
    fn submit_author(&self, npc: NpcId, write: AuthoringWrite) -> Ticket;

    /// Cheap, immediate, no inference — bookkeeping only.
    fn open_interaction(&self, npc: NpcId, spec: &InteractionSpec)
        -> Result<Interaction, EngineError>;
    fn end_interaction(&self, ix: InteractionId) -> Result<(), EngineError>;

    // ── completion: poll, never await ─────────────────────────────────────
    fn poll(&self, t: Ticket) -> Poll<Result<Accepted, EngineError>>;

    // ── events: drain into a buffer the caller owns and reuses ────────────
    /// Appends everything ready since the last call. Zero allocation when
    /// `out` has capacity. Returns the count appended.
    fn drain_events(&self, out: &mut Vec<EngineEvent>) -> usize;
    fn drain_events_for(&self, ix: InteractionId, out: &mut Vec<EngineEvent>) -> usize;

    // ── state: lock-free snapshot read ────────────────────────────────────
    /// An immutable view, swapped at tick boundaries. Reading it never
    /// contends with the inference threads.
    fn snapshot(&self) -> Arc<EngineSnapshot>;

    // ── lifecycle ─────────────────────────────────────────────────────────
    fn create_npc(&self, spec: &NpcSpec) -> Result<Npc, EngineError>;
    fn delete_npc(&self, npc: NpcId, purge: bool) -> Result<(), EngineError>;
    fn tools(&self) -> &ToolRegistry;

    /// Bounded housekeeping, called once per game tick. Drains completions,
    /// retires tickets, services the reaper. Returns within `budget` and
    /// never runs inference.
    fn pump(&self, budget: Duration) -> PumpStats;
}
```

`Ticket` is a `Copy` id, not a handle with a destructor. `EngineEvent` is the union of what a
game cares about — an act committed, a narration ready, an interaction ended, a monitor band
changed.

### The snapshot is the design's own distilled projection

`snapshot()` is not a convenience. The mind document already requires it:

> …this architecture projects a distilled state vector off the gathered cognitive state and
> publishes it to the animation and reflex consumers. That projection is *distilled*, not a full
> KV gather, because those consumers are latency-critical and **must never sit on the decode
> path**.

`EngineSnapshot` is that vector, published for every NPC and swapped atomically at tick
boundaries. A game reads it every frame for free — posture, affect, threat, current intent,
position of attention — with no lock shared with the inference threads and no risk of tearing.
Rich form goes to the log and the substrate; distilled form goes to the frame.

Reading deep state (`beliefs`, `relationships`, the substrate) goes through the snapshot too, or
through an explicit query that is documented as *not frame-safe* and intended for tooling.

### Threading: the engine never runs on your frame thread

```rust
/// Engine owns its threads. The default, and what Battle Cities should use.
let engine = NpcEngine::spawn(config)?;

/// Or: hand work to the game's existing job system.
let engine = NpcEngine::with_executor(config, my_job_system)?;
```

Either way the contract is the same: **inference runs on engine-owned or engine-submitted
threads, never on the caller's.** The game's tick becomes a fixed, bounded shape:

```rust
// once per frame — no allocation, no blocking, no runtime
fn tick(&mut self, dt: Duration) {
    for (npc, sensed) in self.world.sense() {
        self.engine.submit_perceive(npc, sensed);        // returns immediately
    }

    self.events.clear();
    self.engine.drain_events(&mut self.events);          // reused buffer
    for ev in &self.events {
        self.apply(ev);                                   // drive animation, dialogue, AI
    }

    let snap = self.engine.snapshot();                   // Arc clone, lock-free
    self.render_npc_state(&snap);

    self.engine.pump(Duration::from_micros(200));        // bounded housekeeping
}
```

Two tick loops exist and they are deliberately not synchronised: the game's runs at frame rate,
the NPC's at world tempo. The mind document's whole asymmetry depends on that — perception is
cheap and batched, action is expensive and rare — and coupling them would force a decode into a
frame.

### Async is the adapter, not the core

```rust
/// Async facade for the HTTP layer. Not used by embedders.
pub struct AsyncEngine(Arc<dyn NpcEngine>);

impl AsyncEngine {
    pub async fn perceive(&self, npc: NpcId, events: Vec<Perception>)
        -> Result<Accepted, EngineError> {
        let t = self.0.submit_perceive(npc, &events);
        self.completion(t).await          // a notify, not a spin
    }
    pub fn subscribe(&self, ix: InteractionId) -> BoxStream<'static, StreamFrame> { … }
}
```

The adapter owns the only `async` in the system, and it is where SSE and websockets live. If
logic ever appears here rather than in the core, the embedder silently loses it — which is the
same rule as §1, now with teeth.

### One executable, two launch modes — and v1 is headless

**Battle Cities ships the entire system in one executable**, but the AI does not run alongside
the renderer. The binary has two launch modes, and **only one of them loads the engine**:

```
   battlecities.exe                    → the game client. No engine. Zero AI VRAM.
   battlecities.exe --server --headless → the daemon. Engine, API, GUI. No renderer at all.
```

**Version 1 runs AI in headless server mode only.** There is no in-game NPC cognition, no AI
window inside the menu, and therefore no renderer competing for the card while the engine is up.
One EXE, two roles, never both at once in the same process.

This is a much smaller problem than the embedded-alongside-renderer design it replaces, and it
is worth being explicit about how much it removes:

| Concern | Status in v1 |
|---|---|
| Renderer/engine VRAM contention | **gone** — never co-resident |
| Balloon ordering vs renderer init | **gone** — the card is genuinely empty |
| Yield-under-deadline on mode change | **gone** — no mode change |
| Engine load/unload on a menu boundary | **gone** — process lifetime is the boundary |
| A GPU fault killing the game | **gone** — see below |

#### What headless settles

**The sizing analysis in this document is unconditionally correct.** With no renderer, the
engine owns the card: ~22 GiB on the 24 GB box, the SDXL-class image slot is comfortable, and
the misc-work drain has room. Nothing in §14 needs a second, constrained budget.

**The poison contract reverts to what the codebase already does — correctly.**

> `gpu_poison.rs`: a sticky fault "leaves the device context permanently unusable… recreating
> the context in-process is unreliable (especially on WDDM). **The daemon watches this flag and
> exits cleanly for a supervisor restart.**"

That behaviour is right for a server and wrong only for a game process. In headless mode there
is no game to kill, so the existing exit-for-supervisor-restart path is kept unchanged: one root
fault, a clean exit, a fast reboot, and nothing durable lost because the substrate redo log is
crash-safe. **This is a reason to prefer headless for v1, not merely a consequence of it.**

**The game is still the daemon, for tooling.** The API and management GUI are served by the
server-mode process, so an author opens the GUI against a live engine, watches the projection
inspector, edits beliefs, and sees the effect — exactly the workflow that made the
single-executable choice attractive, and it survives intact.

#### What still holds

**No serialization on the hot path.** The JSON shapes in Part B are the HTTP representation only.
The sync core (§22) takes native Rust — `&[Perception]`, not a parsed body — and `snapshot()` is
an `Arc` clone. The sync-first surface is *not* made unnecessary by headless mode: the test
harness and any future in-process client depend on it, and it costs nothing here.

**Panic isolation.** Engine threads wrap their bodies in `catch_unwind` and degrade rather than
abort. Cheaper insurance in a server than in a game, but the same code.

**The build.** CUDA and candle are build dependencies of the shipping binary even though the
client half never calls them. Worth confirming early for any target without a CUDA card — the
engine must compile out cleanly, not merely fail to initialise.

#### Deferred, not solved

Running cognition *during play* is a later version, and when it arrives the analysis that used to
live here comes back with it: two VRAM regimes, yield-before-the-renderer-allocates, a poison
contract that degrades to reflex-only instead of exiting, and the question of whether narrative
time passes while a session runs. None of that is v1's problem, and none of it is invalidated —
it is simply not yet due.

The one piece worth keeping in view now is that **the engine must never assume it owns the
process**. Not calling `exit` from library code, not installing global handlers, not assuming the
CUDA context is its own — those are cheap disciplines today that make the later embedding
possible instead of a rewrite.

#### What hosting implies for substrate confidentiality

Worth stating plainly here, because it compounds two decisions made earlier in this document and
is much cheaper to design for now than to discover later.

Hosting means **an NPC's substrate is processed on a stranger's machine**. To run a tick, the
hosting client needs that NPC's gathered working set in plaintext — the daemon must read the
substrate to run inference over it (§8.3). Combined with the decision not to encrypt the
substrate, and with hidden characters being discretion rather than confidentiality, the
consequence is:

> A hosting player can, in principle, read the content of any NPC their machine processes —
> including its beliefs, memories, dialogue, and **the user-profile turn of whoever that NPC is
> talking to** (§12: description, gender, history).

For ordinary game NPCs this is probably fine: it is game content, much of it authored by the
studio, and no worse than any moddable game's assets. The part that is *not* obviously fine is
the user profile, because that is a real person describing themselves, and it travels with the
conversation.

Three options, none of them free, and the choice belongs to the product rather than this
document:

- **Don't distribute interactive work.** Hosting processes only background cognition — daydream,
  sleep folds, doctrine aggregation, image generation — and never a live interaction with a
  human. This keeps user profiles off other people's machines and costs the distributed system
  its most latency-sensitive workload, which it was probably least suited to anyway.
- **Redact the profile for remote work.** The hosting node receives a pseudonymised interlocutor
  — a unique name and nothing else. Cheap, and it degrades the NPC's ability to be personal with
  a specific player, which is much of the point.
- **Accept and disclose it.** Legitimate for a game, but it must be said out loud in the
  hosting opt-in rather than discovered.

The first option looks strongest: it is a scheduling rule rather than a mechanism, it aligns
with what hosting is naturally good at (batched, latency-tolerant work), and it makes the
disclosure a short and honest one.

### Tool registration is unchanged and still synchronous

```rust
engine.tools_mut().register::<OpenGate>(move |ctx, req: OpenGateRequest| {
    world.gate(req.gate_id).open();
    Ok(OpenGateResponse { opened: true })
})?;
```

`Tool::run` is already synchronous in `zend-tools` — "if a tool needs async I/O the orchestrator
wraps the dispatch call in `spawn_blocking`." A game's tool handler runs on the engine's tool
thread, not the frame thread, and the handler is expected to be quick or to hand off to the
game's own queue. The JSON Schema the model sees is derived from `OpenGateRequest` by schemars,
never hand-written, so the prompt and the parser cannot disagree.

**One caution for game handlers:** a closure capturing game state must not take a lock the frame
thread holds, or inference will stall the game through the back door the rest of this design
closes. The recommended shape is that a tool handler pushes onto a game-owned queue and returns
immediately, exactly as the engine does for its callers.

---

# Part C — The GUI

## 23. The framework — native Web Components, no build step

Two requirements break what zend's GUI does today: **many pages**, and **zend's own GUI ported
onto the same framework**. Today `zend/web` is four hand-written HTML files, the largest 2,804
lines with ~212 inline functions, no router and no components — embedded via `include_dir!` and
served by the `embedded_asset` fallback. That shape is fine for four pages. It does not survive
twenty, and it certainly does not survive two products sharing it.

### Decision

> **Native Web Components with a small hash router. No build step, no node in the build, no
> bundler.**

The alternative considered was Svelte + Vite with committed build output. It wins on ergonomics
and loses on toolchain purity, and the project chose purity: `cargo build` remains the complete
story for producing a shipping binary, and changing the GUI requires no second toolchain at all.

That is a real trade and worth naming honestly. What is given up: reactive bindings, single-file
components, compile-time template checking, and TypeScript. What is kept: one toolchain, no
`node_modules`, no lockfile drift, no build cache, and a GUI that any contributor can edit with
a text editor and reload.

### What must not break, and does not

| Property | How it survives |
|---|---|
| Single-binary deploy | `include_dir!` still embeds plain files — now `web/content/{npcd,common}`, handed to the gateway builder (§2) |
| The mock seam | `window.NpcAPI` selected at load by `?mock=1` — unchanged |
| Playwright suite | drives the mock exactly as it does today |
| `no-store` caching | unchanged; no hashed bundle names to invalidate |

Nothing in the Rust half learns about JavaScript, because there is nothing to learn.

### The architecture

Four native platform features carry what a framework would otherwise provide, and all four are
available without tooling in every browser this ships to.

**ES modules, loaded natively.** `<script type="module">` plus an import map for bare specifiers.
Code-splitting is `await import('./pages/projection.js')` — dynamic import is the platform's own
lazy loading, so twenty pages do not become one file.

**Custom elements as the component model.** One page is one element:

```js
// pages/projection.js
export class NpcProjectionPage extends HTMLElement {
  connectedCallback() { this.render(); this.sub = bus.on('tick', () => this.render()); }
  disconnectedCallback() { this.sub(); }
  render() { /* build DOM, or patch the parts that changed */ }
}
customElements.define('npc-projection-page', NpcProjectionPage);
```

`connectedCallback` / `disconnectedCallback` give lifecycle for free, which is most of what a
page needs — subscribe on enter, unsubscribe on leave, and no leaked listeners when the router
swaps pages.

**Light DOM plus constructable stylesheets.** Shadow DOM is *not* used for pages. It would
isolate them from the shared design system for no benefit, and it complicates the Playwright
selectors that already exist. Instead one `CSSStyleSheet` is built once and attached via
`document.adoptedStyleSheets`, so every page shares tokens and components with no per-page cost.
Shadow DOM stays available for genuinely self-contained widgets where encapsulation earns its
keep.

**`<template>` for markup.** Static structure lives in `<template>` elements cloned per instance,
so pages are still written as HTML rather than as string concatenation.

### The page registry and router

Extensibility to "a large number of pages" is a routing and registration concern, not a framework
one. Pages self-register; navigation and breadcrumbs are *derived* from the registry rather than
maintained beside it.

```js
definePage({
  path:  '/npc/:id/projection',
  tag:   'npc-projection-page',
  title: (p) => `Projection · ${p.npcName}`,
  nav:   { section: 'npc', order: 40, icon: 'layers' },
  guard: requireAccess('viewer'),
  load:  () => import('./pages/projection.js'),   // code-split, native
});
```

The router is a hash router in roughly 200 lines: parse the fragment, match against registered
patterns, extract params, run the guard, `await load()`, create the element, swap it into the
outlet. Hash routing rather than the History API because it needs no server-side rewrite rule —
the `embedded_asset` fallback keeps working untouched.

Four properties follow, and they are what "extendable" has to mean in practice: adding a page
touches exactly one new file; nav is derived, not maintained; every page declares its own access
guard so authorization cannot be forgotten; and `load` is dynamic, so page count does not become
bundle size.

### Reactivity, in about fifty lines

This is the one place the platform gives least, so it is worth being explicit rather than letting
every page invent its own approach.

A tiny signal helper — `signal(value)`, `computed(fn)`, `effect(fn)` — is enough, and it is a
well-understood ~50-line pattern with no dependency. Pages subscribe in `connectedCallback` and
release in `disconnectedCallback`.

For the two genuinely high-frequency surfaces — the live act stream and the token feed — pages
**append** rather than re-render. That is faster than any framework's diff and is the natural
shape for an append-only stream anyway.

The rule that keeps this from degenerating: **no page manipulates another page's DOM.** State
flows through signals and the event bus; DOM ownership stops at the element boundary.

### Porting zend

zend's pages move over as elements, not as a rewrite. `substrate.html`, `perf.html`,
`project.html` and the chat shell become registered pages backed by the **same** api-seam module,
with `ZendAPI` and `NpcAPI` as two implementations of one client convention. Both daemons serve
the same `web/` tree with a different page set enabled.

Because there is no bundle, the migration is genuinely incremental: `embedded_asset` serves
whichever file exists, so a ported page and an unported one coexist with no flag day and no
build-config branch.

### Shared pages

Three zend screens are engine-level rather than product-level and are shared rather than
duplicated:

| Page | Reused as | Change needed |
|---|---|---|
| `substrate.html` | substrate browser | scope by `NpcId` instead of `conv_id` |
| `perf.html` | telemetry / performance | add tick-rate, batch-composition and image-queue panels |
| logs pane | logs | none — `/ws/logs` is identical |
| `project.html` | projection inspector | becomes the per-tick inspector of §36 |

That last row is the useful one: zend's projection view already renders what a projection
selected. The NPC projection inspector is that page with layers renamed and a tick stepper added,
not a new invention.

### Sequencing

Shared pages first (substrate, perf, logs — most reuse, least product logic), then zend's own
screens, then the NPC-specific pages, which are new anyway and can be written directly as
elements.

That order front-loads the risk: if the no-build approach proves too painful, it becomes obvious
while porting three self-contained pages, not after twenty are committed to it. The escape hatch
stays open precisely because there is no bundler to unwind — adding Vite later is additive,
whereas removing it would not be.

## 24. Logged out — the home page

The first screen a stranger sees. It has one job: make someone who has never heard of this
understand what it is and want an account.

```
┌────────────────────────────────────────────────────────────────────────────┐
│  ⬢ npcd                                          Docs   [ Sign in ]        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│              NPCs that remember you.                                       │
│              A mind per character, running on one card.                    │
│                                                                            │
│              [ Sign in with Google ]    [ See a character think ]          │
│                                                                            │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │  LIVE DEMO — Varek, on watch                       ● thinking     │    │
│   │  ┌──────────────┬────────────────────────────────────────────┐   │    │
│   │  │  [portrait]  │ t411  observe  eastern line                │   │    │
│   │  │              │ t411  speak    "Quiet, so far."            │   │    │
│   │  │  Varek       │ t412  face     east                        │   │    │
│   │  │  Loyal       │ t412  move_to  ridge_east                  │   │    │
│   │  │  Soldier     │                                            │   │    │
│   │  └──────────────┴────────────────────────────────────────────┘   │    │
│   │  "You ask what he sees; before he can answer he's already        │    │
│   │   moving as the eastern line buckles."                           │    │
│   │                                        [ say something to him ]  │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│                                                                            │
│   ── what makes it different ───────────────────────────────────────────   │
│   Unbounded memory        Convictions that hold      One card              │
│   Years of history with   Beliefs change only when   16 GB runs a 30B      │
│   bounded error per step  evidence accumulates       model, 64 sessions    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**The demo is the pitch.** A screenshot of a chat window looks like every other product. An
NPC *acting before it explains itself* — the act stream moving while narration is still
assembling — is visible in about four seconds, and it is the shape of the exchange rather than
any particular exchange that makes the point.

**It is a sample, and it says so.** It runs from the mock seam, and the label under it reads *a
sample exchange* rather than the *live — not a recording* it once claimed. That claim committed
the front page to running a real sandboxed NPC for every visitor — a standing operational cost
(a warm model, rate limiting, a world to reset on a schedule) in exchange for something a
stranger cannot verify anyway. Worse, it is the kind of promise that quietly stops being true:
the first time the demo world is down and the block falls back to a replay, the page is lying
to everyone who reads it.

Running a real one later is a strict improvement and needs no change here — the seam is the
same. What must not happen is the label going back before the daemon does.

## 25. Sign-in

Sign-in is one button, not a form — there is no password to collect.

```
┌───────────────────────────────────────────────┐
│              Sign in to npcd                  │
│                                               │
│   ┌───────────────────────────────────────┐   │
│   │             Get started               │   │
│   └───────────────────────────────────────┘   │
│                                               │
│   New here? Signing in creates your account.  │
│   ⓘ We store your name, email and avatar.     │
└───────────────────────────────────────────────┘
```

**One button, not a provider list.** `/auth/login` takes only `next`; the gateway decides which
provider runs the exchange. A row of provider buttons would therefore be several controls that
all navigate to the same URL, one of them naming a provider the gateway may not have configured
— a menu whose choices do not reach the thing that chooses. Adding a provider is gateway
configuration rather than a GUI change here.

Sign-in can also be *unavailable*, which is not the same as being signed out: a deployment with no
`auth:` block has no identity provider, so nobody can sign in at all. The gateway is the only
authority on that — with `auth:` off it does not serve `/auth/login`, and the navigation would
land on site routing and come back as `index.html`, reading to the visitor as a button that does
nothing. `/auth/me` reports `configured: false`, which the console asks before offering the
control. The rest of the landing page — including the live demo — is unaffected.

First sign-in creates the account
with no separate registration step. On return, the user lands on the page they originally asked
for — `next` carries it through the round trip, and is refused unless it points inside this
estate, since an unchecked one is an open redirect.

Signing in at tokera.com is already signed in here: the session cookie is issued on the parent
domain and the browser presents it to this host by itself. A user arriving from the home page
never sees this screen.

## 26. Shell

```
┌────────────────────────────────────────────────────────────────────────────┐
│ ⬢ npcd  My NPCs  Worlds  Personalities  Tools  System   ● ready     (JS)▾  │
├──────────────┬─────────────────────────────────────────────────────────────┤
│              │                                                             │
│  CONTEXT     │   PAGE BODY                                                 │
│  RAIL        │                                                             │
│              │                                                             │
│  (NPC list,  │                                                             │
│   or layer   │                                                             │
│   nav, or    │                                                             │
│   live       │                                                             │
│   interacts) │                                                             │
│              │                                                             │
├──────────────┴─────────────────────────────────────────────────────────────┤
│ ▸ logs                                        tick 412 · 3 npcs · 41 t/s   │
└────────────────────────────────────────────────────────────────────────────┘
```

The status pill mirrors zend's `/v1/status` gating: until the daemon reports ready, a loading
overlay shows the current load step and completed steps. The bottom bar is a collapsible log
pane fed by `/ws/logs`, structured (`LogLine`), not parsed from formatted strings.

One addition for the multi-user build: the **account menu** (avatar, right) holds profile, API
tokens, and sign-out.

There is deliberately **no hidden-character control in the shell**. Hidden NPCs are reached
through ordinary tag filtering on the roster (§28), and a dedicated control would announce that
there is something to find — which is the one thing the feature must not do.

Nav sections come from the page registry, so a new page appears in navigation without the shell
being edited.

## 27. My NPCs — `/`

The landing page once signed in. Every NPC the user owns or has been given access to, with the
two numbers that matter (pending events, monitor band) visible without opening anything.

```
┌────────────────────────────────────────────────────────────────────────────┐
│  My NPCs                                                    [+ New NPC]    │
│  tag:[            ]  world:[Ardh ▾]  state:[any ▾]                    ⊞ ≣ │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ ◯ ● Varek        Loyal Soldier   Ardh   ▮▮▮▯ 3 pend   ♥ healthy   ⋯ │  │
│  │   ▓ tick 30s · last 4s ago · 2 live interactions          owner      │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │ ◯ ● Ilse         Merchant        Ardh   ▮▯▯▯ 0 pend   ♥ healthy   ⋯ │  │
│  │   ▓ tick 120s · last 51s ago · idle                       owner      │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │ ◯ ◐ Hess         Commander       Ardh   ▮▮▮▮ 11 pend  ⚠ fixated   ⋯ │  │
│  │   ▓ tick 5s · last 0s ago · 1 live interaction   ← needs a look      │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │ ◯ ○ Bramble      Gardener        Ardh   ▮▯▯▯ 0 pend   ♥ healthy   ⋯ │  │
│  │   ▓ asleep · consolidating                              editor       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

The tag field is an ordinary filter sitting beside world and state. **Nothing on this page
indicates that hidden characters exist** — no count, no lock, no hint.

`◯` is the portrait thumbnail; `▓` the state glyph strip. State glyphs: `●` active, `◐` ticking
now, `○` asleep, `◌` suspended. Access chips (`owner` / `editor` / `viewer`) appear only when
not owner, so the common case stays uncluttered.

The monitor band is on the roster deliberately — an NPC drifting toward fixation should be
visible here, not discoverable only by opening its own monitor page.

A grid/list toggle (`⊞ ≣`) switches to a portrait-first card grid, which is the better view once
characters have faces and the collection grows past a screenful.

Live-updated over `/ws/events`; no polling.

## 28. Tag filtering, and how hidden characters surface

There is no hidden-characters feature in the UI. There is a tag filter, and hidden characters
happen to match it.

```
   tag:[ moonl        ]                      ← ordinary filter, ordinary autocomplete
        ┌──────────────────────┐
        │ moonlight            │             ← suggestions come from VISIBLE NPCs only
        │ moon-cult            │
        └──────────────────────┘
```

Typing a tag filters the roster the way world and state do. The only difference is invisible
from the outside: the query also matches hidden NPCs carrying that tag, so they appear in the
results alongside everything else, rendered identically.

```
   tag:[ moonlight    ]
   ┌──────────────────────────────────────────────────────────────────────┐
   │ ◯ ● Ilse         Merchant     Ardh   ▮▯▯▯ 0 pend   ♥ healthy   ⋯   │
   │ ◯ ○ Sable        Assassin     Ardh   ▮▯▯▯ 0 pend   ♥ healthy   ⋯   │  ← hidden
   └──────────────────────────────────────────────────────────────────────┘
```

Sable is hidden. Nothing marks it as such in the list — no badge, no lock, no ordering
difference. Its own detail page shows the hidden toggle, but the roster does not.

**Three properties make this discreet, and all three are omissions rather than features:**

- **No count.** The roster never says how many characters are hidden, because saying so is an
  invitation to ask what they are.
- **No autocomplete leak.** Suggestions are built from visible NPCs' tags only. A hidden
  character's tags never appear in a dropdown, a facet count, or a tag cloud.
- **No distinguishable miss.** A tag matching nothing returns an empty list, identical to a tag
  that was never used by anyone.

Clearing the filter returns the default view and the hidden character disappears again. There
is no session state to manage, no reveal to remember, and nothing to re-hide — which is
precisely why it is more discreet than the mode-based design it replaces.

**Vocabulary rule.** The word is **hidden**. "Private", "secure", "locked" and "vault" appear
nowhere in the interface. A user who believes this is encryption will eventually put something
in it that matters, and the interface is the only place that belief gets formed or prevented.

## 29. Creating an NPC — `/npc/new`

A three-step wizard, because a good NPC has more inputs than a dialog holds, and every step has
a working default so the whole thing can be completed by pressing Next three times.

```
┌────────────────────────────────────────────────────────────────────────────┐
│  New character                            ① Identity  ② Face  ③ Inner life │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   Name      ┌──────────────────────┐   World       [ Ardh       ▾ ]        │
│             │ Varek                │   Personality [ Loyal Sold ▾ ]        │
│             └──────────────────────┘                                       │
│                                                                            │
│   Description — who this character is             [ ⟳ Regenerate ]         │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │ Fifty-three, a former staff sergeant who now runs the night      │    │
│   │ shift on a loading dock. Precise about time to the point of      │    │
│   │ rudeness. Comfortable giving orders, uneasy in conversations     │    │
│   │ with no clear purpose. Keeps a folding knife he doesn't use.     │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│   ⓘ This becomes the character's identity in the system prompt, and the    │
│     portrait is generated from it. Written as a present-day person: the    │
│     personality supplies the anchor and the traits, this supplies the      │
│     texture.                                                               │
│                                                                            │
│                                          [ Cancel ]        [ Next → ]      │
└────────────────────────────────────────────────────────────────────────────┘
```

**The description is the character, not a prompt.** It is installed as the NPC's identity
section in the system prompt — the mutable persona sitting above the immutable personality — and
it is *also* the source the portrait is generated from. One field, two consumers, no separate
"image prompt" for the user to keep in sync with the character.

Pre-filled by generation on entry, not left blank beside a button. An empty box is a request for
work; a filled box the user can reject is an offer. Editing by hand flips `origin` to
`authored`.

**World and personality are plain pickers, with no `+`.** Both list files in the mind, and a
button that creates one from here would put a document in a directory the author is not looking
at — the console and the mind would then disagree about what exists. A daemon started without a
mind lists neither, and the picker says so rather than offering to invent one.

Visibility and tags are **not** in the wizard. They are properties of an existing character,
edited later (§30) — putting them here would make every creation a decision about concealment,
which is both noise and, for the one user who cares, a prompt at exactly the wrong moment.

### ② Face

```
┌────────────────────────────────────────────────────────────────────────────┐
│  New character                            ① Identity  ② Face  ③ Inner life │
├────────────────────────────────────────────────────────────────────────────┤
│    ┌─────────────────┐                                                     │
│    │                 │     Generating from the description                 │
│    │   [ portrait ]  │     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░  62%              │
│    │    generating   │                                                     │
│    │                 │     Model [ sdxl-turbo ▾ ]   seed 441028  [ ⟳ ]     │
│    └─────────────────┘                                                     │
│                                                                            │
│    ┌──────────────────────────────────────────────────────────────────┐   │
│    │  or drop an image here / [ Upload a portrait ]                    │   │
│    └──────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│                                  [ ← Back ]  [ Skip ]      [ Next → ]      │
└────────────────────────────────────────────────────────────────────────────┘
```

There is no prompt field. The portrait derives from the description, so a prompt box would be a
second place to say who the character is and a guaranteed source of drift.

**A progress bar, not a warning.** Generation waits for the wave boundary and the reclaim, which
is a real delay, but that is the system working normally and the UI treats it as such. The bar
reflects queue position and generation progress; **Skip** stays available so nobody is blocked,
and the job continues in the background either way.

### Regeneration follows the description

> **Editing the description regenerates the portrait** — queued as misc work, running at the
> next wave boundary — **unless the user has uploaded one.**

An uploaded portrait is a deliberate choice and outranks the generator permanently. It is never
replaced by regeneration; `origin: "uploaded"` is sticky until the user explicitly asks for a
generated one again.

This rule applies wherever the description is edited, including the NPC detail page long after
creation. The character's face tracks the character's identity, which is the behaviour someone
rewriting a description expects and would otherwise have to remember to trigger by hand.

Drag-and-drop anywhere on the panel switches to upload.

### ③ Inner life

```
┌────────────────────────────────────────────────────────────────────────────┐
│  New character                            ① Identity  ② Face  ③ Inner life │
├────────────────────────────────────────────────────────────────────────────┤
│   Generate  [✓] beliefs 5   [✓] relationships 4   [✓] goals 2   [ ⟳ ]      │
│                                                                            │
│   BELIEFS                                                    generated     │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │ [✓] "An order given badly is still an order"      conf 0.90  ✎ ✕ │    │
│   │ [✓] "Hess is a man of his word"                   conf 0.72  ✎ ✕ │    │
│   │ [ ] "The northern road is impassable in winter"   conf 0.55  ✎ ✕ │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│   RELATIONSHIPS                                              generated     │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │ [✓] Hess — commander        trust +0.6  affect +0.2          ✎ ✕ │    │
│   │ [✓] Ilse — merchant         trust +0.1  affect +0.4          ✎ ✕ │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│                                                                            │
│   Environment simulator  [✓] on                                            │
│   ⓘ No world simulation is attached, so this generates what happens        │
│     around the character. Turn it off if your own game drives events.      │
│                                                                            │
│                                  [ ← Back ]            [ Create ]          │
└────────────────────────────────────────────────────────────────────────────┘
```

Generated attributes arrive as **checked proposals the user can uncheck, edit, or delete** —
never as writes. Nothing reaches the substrate until **Create**, and everything that survives
lands with `origin: "generated"` so its provenance stays legible years later.

The environment toggle is on by default with its reason stated inline, matching the API's
origin-based default.

## 30. Managing NPCs

Per-row `⋯` on the roster, and a **Manage** tab on NPC detail. This is where visibility and tags
live, deliberately away from creation:

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Manage · Varek                                                            │
│                                                                            │
│   Description                                       [ ⟳ Regenerate ]       │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │ Fifty-three, a former staff sergeant who now runs the night …    │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│   ⓘ Changing this updates the character's identity and regenerates the     │
│     portrait. (Not while a portrait you uploaded is in use.)               │
│                                                                            │
│   Tags   [campaign-2 ✕] [moonlight ✕]  ┌──────────────┐                    │
│                                        │ add a tag…   │                    │
│                                        └──────────────┘                    │
│                                                                            │
│   Hidden  [ ]  Keep out of the default list                                │
│           ⓘ Still found by filtering for any tag above. Hiding is          │
│             discretion, not encryption.                                    │
└────────────────────────────────────────────────────────────────────────────┘
```

Tags and the hidden toggle sit together because they are only useful together: hiding a
character with no tags makes it unreachable from the roster. The form says so if you try.

- **Rename**, **change portrait**, **edit description** (regenerates the portrait — §29)
- **Duplicate** — a fresh NPC with the same identity and seed state but no lived history. The
  distinction is explicit in the dialog, because "copy this character" means both things to
  different people and only one of them is cheap.
- **Share** — grant another user `editor` or `viewer` by email
- **Export** — the NPC as JSON (identity, persona, authored state, portrait reference)
- **Suspend** — stop ticking without deleting; reversible
- **Delete** — tombstone, requires typing the character's name

Bulk selection on the roster supports suspend, tag, and delete across many characters, since a
user with fifty NPCs will want to put a whole cast to sleep at once.

## 31. NPC detail — `/npc/{id}`

The substrate made browsable. The rail is the layer list; each layer is a stream.

```
┌──────────────┬─────────────────────────────────────────────────────────────┐
│ VAREK        │  Beliefs                                     [+ Author]     │
│ Loyal Soldier│  ┌───────────────────────────────────────────────────────┐  │
│ Ardh · tick30│  │ "Hess is a man of his word"          conf 0.72  ⚠     │  │
│              │  │ authored · threshold 0.85 · disconf 0.30              │  │
│ ─ layers ─   │  │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░  under pressure                 │  │
│ perception 41│  │                                       [history] [edit] │  │
│ action    212│  ├───────────────────────────────────────────────────────┤  │
│ agency      6│  │ "The northern road is passable in winter"  conf 0.95   │  │
│ relations  14│  │ evidence · threshold 0.60 · disconf 0.00               │  │
│▸beliefs     9│  └───────────────────────────────────────────────────────┘  │
│ memory   4412│                                                             │
│ interaction 2│  ── belief history ──────────────────────────────────────   │
│ environment  │   1.0 ┤●───●───●──                                          │
│ world        │   0.5 ┤          ╲●──●   ← disconfirmation accumulating     │
│              │   0.0 ┼──────────────────────────────────────────────       │
│ ─ tools ─    │        T-40d      T-20d      T-5d      now                  │
│ projection   │                                                             │
│ monitor      │                                                             │
│ environment  │                                                             │
└──────────────┴─────────────────────────────────────────────────────────────┘
```

Per-layer bodies:

- **perception** — reverse-chronological feed. Map events render the ascii block monospaced
  with its legend, and superseded maps are collapsed under a "3 superseded" disclosure rather
  than deleted from view.
- **action** — the act stream, ground truth. Each act shows its tool, args, tick, and whether
  the arbiter committed it. Rejected acts are shown struck through with the rejection reason;
  hiding them would hide the most interesting failures.
- **agency** — strategies as a tree (parent → sub-goals), each with state and salience.
- **relationships** — per-entity cards with trust/affect/familiarity sliders, editable inline.
- **beliefs** — as drawn. The disconfirmation bar and `under_pressure` flag are the point:
  they make visible the thing the architecture protects.
- **memory** — consolidated memory, cursor-paginated, searchable.
- **interaction** — live and archived interactions; archived ones open read-only.

**Every editable control here writes on the authoring plane** and is labelled as such. Authored
values render with an `authored` chip so an operator can always distinguish what they set from
what the character earned.

## 32. Interaction console — one page, four modes

`/interaction/{ix}` renders differently per mode, because talking to someone in a room and
messaging them are genuinely different experiences and a single layout serves neither well.

What is shared: the **two-latency stream** (acts live, narration at tick close), the composer
with slash commands (§35), the idle countdown, and the operator's ability to see acts the
interlocutor cannot.

| Mode | Layout | §|
|---|---|---|
| `instant_message`, `video_call` | messaging — bubbles, images inline | §33 |
| `physical` | narrated scene — coloured event log, periodic POV imagery | §34 |
| `voice_call` | messaging layout, audio-only acts, no images | §33 |

**Operator view versus participant view.** Every layout has a toggle. *Participant* shows only
what the interlocutor can observe — what a player would see. *Operator* additionally shows
intents, hidden acts, and tick boundaries. The default is operator, since this is a management
GUI, but the participant view is what gets checked before shipping a character.

## 33. Messaging mode — `instant_message`, `video_call`, `voice_call`

A chat client, with the NPC's own machinery visible beside it when you want it.

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Varek · instant message · as Wren               idle 23:41   [End] [◫ ops]│
├──────────────────────────────────────────────┬─────────────────────────────┤
│                                              │  ACTS            live ●     │
│   ┌────────────────────────────────────┐     │                             │
│   │ what do you see?                   │ You │  t411 speak                 │
│   └────────────────────────────────────┘     │   → "reassure Wren the      │
│                                              │      line holds, but hedge" │
│   ┌────────────────────────────────────┐     │   ✓ rendered                │
│   │ The line's holding. For now.       │     │                             │
│   └────────────────────────────────────┘     │  t412 send_image            │
│    Varek · 06:14                             │   to: Wren                  │
│                                              │   → "show her the ridge     │
│   ┌────────────────────────────────────┐     │      where the fighting is" │
│   │  ┌──────────────────────────────┐  │     │   ▓▓▓▓▓░░░ generating       │
│   │  │                              │  │     │                             │
│   │  │      [ ridge, dusk ]         │  │     │  t412 note_concern          │
│   │  │                              │  │     │   ⊘ not observable          │
│   │  └──────────────────────────────┘  │     │                             │
│   │  "east ridge — that's where"       │     │  ⋯ decoding                 │
│   └────────────────────────────────────┘     │                             │
│    Varek · 06:15                             │                             │
├──────────────────────────────────────────────┴─────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐ [ Send ] │
│  │ /                                                            │          │
│  └──────────────────────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────────────────────┘
```

**Bubbles wait for `rendered`.** The NPC emits intent; the narrator writes the words (§18).
A bubble appears only when the `act_rendered` frame lands — showing a typing indicator until
then, which is both honest and exactly what a messaging client should look like.

The ops panel (`◫`) shows the intent behind every bubble. That pairing — intent on the right,
prose on the left — is the most useful debugging view in the product, because a reply that reads
wrong is almost always either a wrong intent or a wrong rendering, and this tells you which.

**Images arrive as an act, then as a picture.** `send_image` commits immediately with its intent
and target name; the image is queued as misc work and lands at the next drain (§14). The bubble
shows a placeholder with progress, then the image. A caption, if any, is narrated from the same
intent.

Voice-call mode uses this layout with `send_image` absent from the catalog and acts marked
audio-only.

## 34. Physical mode — the narrated scene

A room, not a thread. The centre column is a continuous narration where every event is a typed,
coloured entry, and the scene is periodically illustrated.

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Varek · physical encounter · as Wren            idle 04:12   [End] [◫ ops]│
├────────────────────────────────────────────────────┬───────────────────────┤
│                                                    │  ACTS        live ●   │
│   ┌──────────────────────────────────────────┐     │                       │
│   │                                          │     │  t411 face east       │
│   │        [ the ridge at dusk, from         │     │  t411 speak           │
│   │          where you are standing ]        │     │   → "acknowledge her, │
│   │                                          │     │      stay watchful"   │
│   └──────────────────────────────────────────┘     │                       │
│    the scene · 06:14                               │  t412 observe         │
│                                                    │  t412 speak           │
│  ▎He straightens as you approach, shears still     │  t412 move_to ridge   │
│  ▎in hand.                                   scene │  t412 broadcast_strat │
│                                                    │   ⊘ no observable     │
│  ▎"Quiet, so far."                             say │     trace             │
│                                                    │                       │
│  ▎He glances east, and does not look back.     act │  ⋯ decoding           │
│                                                    │                       │
│  ▎The eastern line buckles. Somewhere below,       │                       │
│  ▎a horn.                                     world│                       │
│                                                    │                       │
│  ▎He's moving before he finishes the sentence.  act│                       │
│                                                    │                       │
│  ── tick 412 ──────────────────────────────────    │                       │
├────────────────────────────────────────────────────┴───────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐ [ Send ] │
│  │ /act reaches for his arm                                     │          │
│  └──────────────────────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────────────────────┘
```

### Event types are turn metadata, not separate storage

Every line in that column is an ordinary conversation turn on the `interaction` layer. What
makes it renderable as a typed, coloured event is **metadata carried on the turn**, matching the
narrator's own input vocabulary:

| Kind | Rail | What it is |
|---|---|---|
| `say` | speech | dialogue rendered from a `speak` intent |
| `act` | action | a physical act the interlocutor can observe |
| `scene` | ambient | environment description, from the simulator or world |
| `world` | event | an injected world event that landed in this window |
| `cue` | forced | an operator-forced act — visually distinct, since it is not the NPC's choice |
| `beat` | steering | operator narrative steering; **operator view only**, never shown to a participant |

These are exactly `NarratorInput`'s variants, which is not a coincidence — the same enum
describes what goes *into* the narrator and what comes *out* as a typed turn. One vocabulary,
both directions.

Storing them as turns with metadata rather than as a parallel event log matters: the scene the
operator reads **is** the conversation the NPC gathers. There is no second record that could
disagree with the first.

Colour and rail are a rendering of that metadata. `beat` being operator-only is the one place
the participant view diverges structurally rather than by observability.

### Periodic scene imagery, from the user's eyes

> Every so often the narrator generates an image of the current scene **from the interlocutor's
> point of view** — what you would see standing there.

The prompt is composed by the narrator from the gathered scene state, framed as the user's
vantage: what is in front of *you*, at the distance *you* are standing, in the light there is
now. It is a `Scene`-derived render rather than a portrait of the NPC, and it uses the same
misc-work queue (§14), so it never competes with a tick.

Cadence is deliberately not per-tick. An image is worth generating when the scene has
**materially changed** — a location move, a lighting or weather shift, a new participant, or a
long interval with none of those. Rendering every tick would be both ruinous on one card and
wrong: a picture per beat reads as a slideshow, not a place.

The images interleave into the narration column as `scene` entries with an image body, dated
like any other turn. Under VRAM pressure they simply do not appear, and the scene reads as text
— degradation the player never notices, which is the correct failure mode for a decoration.

### The composer is narrator input

The physical-mode composer is a `NarratorInput` producer. Plain prose is a `say` from the user;
`/act`, `/scene`, `/cue`, `/beat` produce the corresponding typed events through the same
clap-style parser the narrator already uses (`parse_turn`) and the same schema-driven palette as
§35. There is one command grammar in the product, and this is it.

## 35. Slash commands — creating events

Typing prose into the composer injects speech. Everything else an operator wants to do to a
live scene — damage the NPC, have someone enter, change the weather, hand them an object — is a
**structured event**, and structured events need structured input.

They live behind `/`.

### The palette

```
┌────────────────────────────────────────────────────────────────────────────┐
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ /da                                                              │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │  COMBAT                                                          │      │
│  │  ▸ /damage      Apply damage to the NPC                          │      │
│  │  ▸ /danger      Raise the perceived threat level                 │      │
│  │  WORLD                                                           │      │
│  │  ▸ /daybreak    Advance the world clock to dawn                  │      │
│  └──────────────────────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────────────────────┘
```

Fuzzy match over name, alias and summary, grouped by `Command.group`, `↑↓` to move, `Tab` or
`Enter` to accept. Recently used commands float to the top, because a session tends to reuse
the same four.

### Narrowed to one — the parameter view

Once exactly one command matches, the palette becomes a **parameter form built from the
command's JSON Schema**:

```
┌────────────────────────────────────────────────────────────────────────────┐
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ /damage amount:12 source:"arrow" location:                       │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │  /damage — Apply damage to the NPC                    → perception│      │
│  │                                                                  │      │
│  │   amount    integer  1..100   required   ✓ 12                    │      │
│  │   source    string             optional   ✓ "arrow"              │      │
│  │ ▸ location  enum               optional                          │      │
│  │             head · torso · left_arm · right_arm · leg            │      │
│  │   severity  number   0..1      optional   (default 0.5)          │      │
│  │                                                                  │      │
│  │   ⏎ send    ⇥ next field    esc cancel                           │      │
│  └──────────────────────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────────────────────┘
```

The line stays editable text throughout — a fluent user types
`/damage 12 arrow torso` and presses Enter without ever looking at the panel, while a new user
tabs through fields. Both drive the same parse. Validation is live: satisfied parameters show
`✓`, a violated constraint shows the reason inline, and Enter is refused while a required field
is missing.

### One parser, shared with the tool system

The important design point is that **this is not a new command language.**

> A slash command is described by a JSON Schema, exactly as a tool is. The palette, the
> parameter hints, the completion of enum values, the validation, and the final parse all read
> that one schema.

`GET /v1/commands` returns `Command[]` whose `parameters` field is the same schemars-generated
JSON Schema shape that `ToolInfo.parameters` carries. The consequences are worth spelling out
because they are the reason to do it this way:

- **Adding a command is adding a schema.** No GUI change, no parser change, no grammar to
  extend. It appears in the palette with correct hints the moment the daemon serves it.
- **Extension tools registered by the framework can expose slash commands for free**, since
  they already have a typed `Request` with a derived schema. A game that registers `open_gate`
  gets `/open_gate` in the operator console at no extra cost.
- **The parse cannot drift from the validation.** One schema, one `validate()`, the same
  `invalid_arguments` error shape the tool dispatch pipeline already returns.

The shared parser is a small crate module — tokenize positional and `name:value` arguments,
coerce against the schema's types, apply defaults, validate — used by the GUI (compiled to
WASM, or reimplemented against a shared test corpus) and by the daemon. **The test corpus is
the contract:** a table of input strings and expected parses that both implementations run, so
the browser and the server can never disagree about what `/damage 12 "left arm"` means.

Commands `emit` one of three things, shown in the panel's top-right so the operator knows where
it lands: a `perception` event on the NPC's inbox, an `interaction_event` scoped to this
interaction, or an `environment_event` for the simulator.

Unknown command, or a parse that cannot be repaired: the line stays put with the error inline
and nothing is sent. A malformed event silently becoming speech would be much worse than a
refusal.

## 36. Projection inspector — `/npc/{id}/projection`

The instrument for every calibration question the mind document leaves open.

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Projection · tick [412 ▾]  ◀ ▶            budget 15,214 / 16,000  (95%)   │
├────────────────────────────────────────────────────────────────────────────┤
│  SYSTEM PROMPT (the lens)                                                  │
│   identity_anchor ▮ always   mood ▮ tense (spiked t409)                    │
│   template ▮ battlefield_urgency (locked at open)   situation ▮  concerns ▮ │
├────────────────────────────────────────────────────────────────────────────┤
│  GATHERED                                       selected / available       │
│   perception  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  4,120 tok        8 / 41   top 0.94        │
│   action      ▓▓▓▓▓▓▓▓          2,010 tok        5 / 212  top 0.81        │
│   beliefs     ▓▓▓                  812 tok        3 / 9    top 0.88        │
│   relations   ▓▓                   540 tok        2 / 14   top 0.77        │
│   agency      ▓                    260 tok        1 / 6    top 0.69        │
│   memory      ▓▓▓▓▓▓▓▓▓▓▓        3,180 tok       11 / 4412 top 0.72        │
│   world       ▓▓▓▓               1,290 tok        4 / 88   top 0.66        │
├────────────────────────────────────────────────────────────────────────────┤
│  DROPPED                                                                   │
│   memory   6 turns  budget      ▸ show                                     │
│   world    9 turns  threshold   ▸ show                                     │
└────────────────────────────────────────────────────────────────────────────┘
```

Clicking a layer expands its selected turns with score, tokens and preview. **Dropped turns are
first-class** — the interesting question is usually not what was gathered but what nearly was,
and a tool that only shows winners cannot answer it.

The tick stepper walks history so a strange act can be traced back to the gather that produced
it.

## 37. Monitor — `/npc/{id}/monitor`

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Monitor · Varek                                     band: healthy ♥       │
│                                                                            │
│  0.6 ┤                                          ░░░░░ runaway              │
│  0.5 ┤─────────────────────────────────────────────── 0.55                 │
│  0.4 ┤                                          ▒▒▒▒▒ fixated              │
│  0.35┤───────────────────────────────────────────────                      │
│  0.3 ┤                    ╭─╮                                              │
│  0.2 ┤────────╮      ╭────╯ ╰──╮   ╭──                expressive band      │
│  0.1 ┤   ╰────╯──────╯          ╰──╯                                       │
│  0.0 ┼──────────────────────────────────────────────────────────           │
│       t312            t360            t400          t412                   │
│                                                                            │
│  narration/substrate n-gram overlap · window 100 ticks                      │
│  ⓘ Rising overlap means the NPC is reading its own output as fresh signal.  │
└────────────────────────────────────────────────────────────────────────────┘
```

The bands are labelled with intent rather than as pass/fail: the expressive band is where a
brooding character lives, and the point of the instrument is to let you push an NPC toward a
characterful near-edge *deliberately* while seeing when it is about to tip past character into
incoherence.

## 38. Environment, worlds, personalities, tools

**Environment panel** (`/npc/{id}/environment`) — a toggle with its consequence stated, a
system-prompt editor, the sliding window's recent turns, and a world-event injector.

### Worlds — `/world` and `/world/{wid}`

A world is a first-class editable object, reachable from the nav and from the `+` beside the
world picker in the creation wizard (§29). A first-time user creates one here without leaving
the flow that needed it.

```
┌────────────────────────────────────────────────────────────────────────────┐
│  World · Ardh                                          4 characters   ⋯    │
├────────────────────────────────────────────────────────────────────────────┤
│   Name  ┌────────────────┐        Public [ ]  anyone may spawn here        │
│         │ Ardh           │                                                 │
│         └────────────────┘                                                 │
│                                                                            │
│   Setting — the shared world knowledge every character reads               │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │ A kingdom of hill villages on a northern frontier, three years   │    │
│   │ after a war nobody won. Roads are unsafe after dark. …           │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│   ⓘ This is the immutable core shared by every character here. Editing     │
│     it changes what all 4 of them know.                                    │
│                                                                            │
│   ── narrative clock ────────────────────────────────────────────────      │
│   Now   day 412, 06:14        Scale [ 60× ▾ ]  ⏸ paused [ ]                │
│                               60 world-seconds per real second             │
│                               [ Jump to… ]                                 │
│                                                                            │
│   ── map zoom bands ─────────────────────────────────────────────────      │
│   [strategic ✕] [regional ✕] [tactical ✕] [local ✕]   ┌──────────┐        │
│                                                        │ add band │        │
│                                                        └──────────┘        │
│                                                                            │
│   ── templates ──────────────────────────────────────────────────────      │
│   responses/  ● world override (12 sections)      [ edit ] [ revert ]      │
│   moods/      ○ using defaults                    [ override ]             │
└────────────────────────────────────────────────────────────────────────────┘
```

Four things live here, and each has a blast radius the form states plainly:

- **Setting** — the shared world knowledge in the immutable core. Editing it changes what every
  character in the world knows, and the character count is shown next to the warning for that
  reason.
- **Narrative clock** — current world time, scale (`0` = paused), and a jump. Applies to every
  NPC in the world; a jump confirms before applying.
- **Zoom bands** — the `zoom` values perception maps may declare (§15). Declared per world
  rather than hardcoded, because a city game and a campaign game want different granularities.
- **Template overrides** — per-world `responses/` and `moods/` shadowing the defaults (`npc_engine_design.md` §IX),
  since tone is world-specific.

Deleting a world is refused while characters live in it, naming how many.

**Personalities** (`/personalities?a={aid}`) — the anchor and the constant traits read-only
(they are immutable by construction and the UI should not imply otherwise), and **doctrine**
editable with its version. The read-only rendering is a design statement: an operator who can
edit identity in a text box will eventually believe identity is editable.

The page renders the document, not a description of it. `personalities/<id>.yaml` carries the
anchor and the traits inline, so there is no separate "section collections" view to keep in step
with the file — a panel that described anchor, traits and doctrine as three folders of templates
was narrating a structure that had stopped existing. Traits are **always visible**, not selected
`top-k 3`: a character is not situationally itself, and choosing three traits per turn made it
partly itself, differently each turn. Biography — the part that genuinely is situational — lives
in the `memory` layer, where provenance retrieves it.

**Tools** (`/tools`) — the catalog, grouped by category, showing source (generic/extension),
the JSON Schema the model actually sees, and calibration state. Uncalibrated tools are flagged
with a **Calibrate** action, because a silently-uncalibrated tool is a mysteriously bad NPC.

## 39. Ported and shared pages

Three zend screens are engine-level rather than product-level and are shared by both daemons
rather than reimplemented. They move into `packages/core/` and are enabled per app.

### Substrate browser — from `substrate.html` (709 lines)

zend's substrate view already renders layers, groups, timelines and their turn occupancy. The
NPC version is the same page with the scope changed: **`NpcId` where zend passes `conv_id`**,
and the NPC layer names in place of the coding-assistant ones. It becomes the per-layer detail
view behind §31's rail.

The reason this ports rather than being rebuilt is worth noting: both products are looking at
the *same substrate*, and the interesting questions — which turns are resident, what a layer's
occupancy is against its window, which timelines share a group — are identical.

### Performance — from `perf.html` (976 lines)

Throughput, VRAM, tier residence, decode rates. Kept nearly whole, with three panels added for
`npcd`:

- **Tick rates** — ticks/sec across the population, and the distribution of inbox depth
- **Batch composition** — how many NPCs share each decode batch, which is where the
  popular-NPC-is-cheapest claim is either visible or false
- **Image queue** — depth, current job, whether it is stalled on VRAM

The second is the one to build carefully. It is the direct empirical check on the central
performance thesis, and there is currently no view of it anywhere.

### Logs — from the existing pane

Unchanged. `/ws/logs` carries structured `LogLine` and the pane needs no product knowledge. It
moves into the shell and both apps get it.

### Projection — from `project.html` (318 lines)

Becomes the projection inspector of §36: the existing render of what a projection selected, with
layers renamed, dropped turns surfaced, and a tick stepper added.

### What does not port

zend's chat shell, repo-map completeness, code-reading views and file-upload panes are
coding-assistant specific and stay in `packages/zend/`. They move onto the new framework as part
of the port but are not shared with `npcd`.

### Sequencing the port

The port is incremental and needs no flag day: `embedded_asset` serves whichever file exists, so
a ported page and an unported one coexist. The recommended order is **shared pages first**
(substrate, perf, logs — they have the most reuse and the least product logic), then zend's own
screens, then the NPC-specific pages, which are new anyway and can be written directly on the
new framework.

That order also front-loads the risk: if the framework choice is wrong, it becomes obvious while
porting three self-contained pages, not after twenty are committed to it.

## 40. Cross-cutting behaviours

- **Loading overlay** until `/v1/status` reports ready, showing the current load step and the
  completed list. Long model loads are the normal case.
- **503 is not an error toast.** It re-enters the loading state.
- **Optimistic authoring writes** with rollback on failure; the `authored` chip appears
  immediately.
- **Every id displayed is copyable** and shown truncated (`1023…4281`) with the full value on
  hover.
- **Narrative time is primary in every display**, wall time available on hover. An NPC's
  memory dated in wall time is unreadable.
- **No destructive action without naming the object.** Tombstoning an NPC requires typing its
  name; the doctrine editor warns that a doctrine change reaches every character of that
  personality worldwide.
- **Keyboard**: `g r` roster, `g t` tools, `j/k` list nav, `Esc` close. `/` opens the command
  palette in a composer and search elsewhere — the same key, context-dependent, because in a
  composer `/` can only mean a command.
- **Portraits degrade gracefully.** No portrait yet, queued, generating (with progress),
  regenerating after a description edit, failed, and upload-in-progress are distinct visual
  states — never a broken image icon. A character whose face is still queued is the normal case
  for minutes, and it is shown as progress rather than as a problem.
- **Hidden characters are never rendered differently.** No badge, no dimming, no ordering
  change, no count. A hidden NPC surfaced by a tag filter is pixel-identical to a visible one;
  only its own Manage tab shows the toggle.
- **Session expiry is not a data-loss event.** An expired session surfaces a re-auth prompt over
  the current page and restores it afterwards; an unsent composer draft survives.
- **Generated content is always labelled.** Persona, portrait, beliefs, relationships — each
  carries its origin chip. The user must never have to guess whether they wrote something or a
  model did.

## 41. The mock seam

Inherited directly from zend's `zend-api.js`: one seam the UI talks to, two implementations,
selected at load.

```
window.NpcAPI = live | mock            // ?mock=1 or window.NPC_BACKEND==='mock'
```

Both satisfy the identical contract:

```
getStatus() -> { state, detail, loading?, build }
listNpcs(filter) -> Page<Npc>
getNpc(id) -> Npc
createNpc(spec) -> Npc
patchNpc(id, patch) -> Npc
perceive(id, events) -> Accepted
getLayer(id, layer, {limit, cursor}) -> Page<Turn>
getRelationships(id) / setRelationship(id, rel)
getBeliefs(id) / authorBelief(id, belief) / deleteBelief(id, beliefId)
getAgency(id) / setStrategy(id, s)
getModulation(id) / setModulation(id, m)
openInteraction(id, spec) -> Interaction
inject(ix, payload)
streamInteraction(ix, handlers) -> { cancel() }
endInteraction(ix)
getProjection(id, tick?) -> ProjectionSnapshot
getMonitor(id, window) -> MonitorReport
getEnvironment(id) / setEnvironment(id, cfg) / injectEnvironment(id, ev)
listWorlds() / getWorld(wid) / setWorld(wid, cfg) / setWorldTime(wid, t)
listPersonalities() / getPersonality(aid) / setPersonality(aid, doc)
listTools() / calibrateTools()
subscribeLogs(onLine, onState) -> { close() }      // backlog replays on the socket
subscribeEvents(onEvent, onState) -> { close() }

── auth & ownership ──
getProviders() -> Provider[]
getMe() -> User | null                       // null = logged out → home page
logout()
listTokens() / mintToken(name) / revokeToken(id)
setHidden(npcId, hidden)
setTags(npcId, tags)

── generation & images ──
generateDescription(spec) -> { description, seed }
generateAttributes(spec) -> { beliefs, relationships, agency }
generateNpc(spec) -> GenerationJob
pollJob(jobId) -> GenerationJob
generateImage(spec) -> GenerationJob
getImageQueue() -> { depth, position, next_run_eta }
uploadPortrait(npcId, file, handlers) -> { cancel() }
imageUrl(imageId) -> string

── commands ──
listCommands() -> Command[]
parseCommand(line, commands) -> { command, args, errors }   // the shared parser
```

```
handlers (interaction) = {
  onOpen(info), onAct(act), onTick(tick), onNarration(n),
  onStatus(text), onState(state), onError(e)
}
```

The mock runs a small scripted NPC — ticks on a timer, emits plausible acts, produces narration
at tick close, drifts a belief under pressure, and walks the monitor into the fixated band on
demand. That last one matters: **the failure states must be reachable in the mock**, or the UI
for them gets designed blind and discovered wrong in production.

The mock now also covers the states the new features introduce, for the same reason:

| State | Why the mock must reach it |
|---|---|
| logged out | the home page is a real screen, not a redirect |
| a fresh account with zero NPCs | the empty state is most users' first impression |
| a hidden NPC reachable only by its tag | the discretion rule is invisible by design and so is easy to break silently |
| image queue with several jobs draining as a batch | the load-once-drain-all cycle, and the only way to see it is wrong |
| image job waiting on a wave boundary, then failed | the common case on one card, and the easiest to design wrong |
| description edited → portrait regenerating | the coupling in §29, including the uploaded-portrait exception |
| slash command with every schema type | enum, range, required, default — the parameter view's whole surface |
| a user with no worlds | the wizard's `+` path out of a dead end |
| an act whose `rendered` never arrives | the messaging bubble must not hang forever |
| physical mode with every event kind | say · act · scene · world · cue · beat, and beat operator-only |
| a scene image arriving mid-narration | interleaving into the column, and its absence under pressure |
| profile edited mid-conversation | the NPC still addressing the old name until it re-gathers |

The mock is also the Playwright fixture, as it is in zend.

### The other mock

This one replaces the network, which is what makes it fast and what makes it a fixture. It
therefore cannot catch anything below `fetch`: a broken route prefix, a proxy that drops a
header, a `101` the client rejects, an error page that renders as raw JSON.

`web --authoritative` (§2) is the second mock, one layer down — it replaces the *daemon* and
leaves real sockets, real routing and the real proxy in place. Between them the console is
covered from both directions, and neither is ever selected automatically: `?mock=1` and
`--authoritative` are both things a person types.

---

# Part D — Decisions needed

These are choices the design cannot make for you; each changes the contract above.

**RESOLVED — the player client shares this API behind a scoped token.** Everything specified here is operator-shaped: it exposes hidden acts, intents, and authored-versus-earned provenance. A player must see strictly less, and the filtering happens **server-side by token scope** rather than through a second `/v1/play/*` surface — two surfaces drift, one does not. A player token is denied hidden acts, `intent` fields, `beat` events, the authoring plane, and every introspection route.

**Is `world` a real resource or a label?** The API above treats worlds as first-class with their
own clock. If a deployment only ever has one world, most of §20 is ceremony.

**Who owns the entity namespace?** `Relationship.entity_id` and `Perception.entity_id` reference
things — players, NPCs, places — that the engine does not model. Either the engine grows an
entity registry, or these stay opaque strings the world simulation gives meaning to. The
document assumes opaque strings.

**Should `/v1/npc/{id}/tick` exist in production?** Forcing a tick is invaluable for authoring
and dangerous in a live world, where it lets a client drive an NPC's cognition at whatever rate
it likes. Options: gate it behind a `--dev` flag, rate-limit it, or accept it.

**How much projection history is retained?** `/v1/npc/{id}/projection/{tick}` implies a ring
buffer. Deep history makes the inspector far more useful for exactly the calibration questions
it exists to answer, and costs memory per NPC. A default of 200 ticks is a guess, not a finding.

**RESOLVED — no build step.** §23 specifies native Web Components with a hash router. `cargo build` remains the complete story for a shipping binary and the GUI needs no second toolchain. The cost — no reactive bindings, no single-file components, no TypeScript — is accepted deliberately; a ~50-line signal helper covers the reactivity gap, and high-frequency streams append rather than re-render.

**Do NPCs share a machine across users?** Multi-user plus one GPU means one user's busy
character degrades another's. Nothing in this document addresses per-user quotas, fair
scheduling between users, or what a user sees when the card is saturated by someone else. That
is a scheduling design, and it is genuinely absent.

**~~Is the hosted demo NPC on the home page worth its cost?~~** Answered: not yet. A live demo
consumes real GPU on a machine whose whole premise is that GPU is scarce, and the liveness was
the half a visitor could not verify anyway. The block now runs from the mock seam and is
labelled *a sample exchange* (§24). Hosting a real one later is a strict improvement through the
same seam; the open part is only the policy — how much of the card a stranger may use — which
is a question worth having when there is a card to spare.

**Where does the shared command parser actually live?** §35 requires the browser and the daemon
to agree exactly. Compiling the Rust parser to WASM guarantees it and adds a WASM artifact to
the GUI build; reimplementing in JS against a shared test corpus avoids that and relies on the
corpus being complete. The corpus is specified either way; which implementation ships is not.

**Should generated attributes be reviewable after creation?** §29 reviews them in the wizard,
but `POST /v1/generate/attributes` on an existing NPC has no review surface yet — it would land
in the NPC detail page as a proposals tray, which is designed nowhere.

**RESOLVED — per-interaction.** Each interaction gets its own narrator conversation and sliding window, so it stays in voice for its own thread and cannot bleed another conversation's phrasing into it. The cost is real — three concurrent interactions mean three narrator contexts on one NPC — so phase 5 instruments it, and collapsing to per-NPC stays available if it proves unacceptable.

**What happens when narration lags behind acts?** `rendered` may arrive a beat late, and under
load it could fall further. A messaging client showing a typing indicator indefinitely is worse
than one that shows the raw intent. There should be a timeout after which the surface degrades
to something, and what that something is — plain intent, an ellipsis, a dropped bubble — is not
decided.

**How is scene-image cadence actually decided?** §34 says "when the scene has materially
changed," which is a judgment the narrator would have to make. Options: a model call that costs
a decode, a heuristic over changed scene fields, or an explicit `render_scene` narrator output.
The heuristic is cheapest and probably right, but nothing here specifies it.

**Is `unique_name` claimable or assigned?** §12 makes it globally unique and user-chosen,
which means a namespace to squat, reserve, and moderate. A prefixed or suffixed form
(`wren#4471`) removes the contention entirely at some cost to how it reads in dialogue — and how
it reads in dialogue is the entire reason it exists.

**Do NPCs address each other by unique name too?** The design gives users unique names because
tools need a target. NPC-to-NPC interactions have the same requirement, and nothing here says
whether NPCs share that namespace or have a separate one.

**When cognition-during-play arrives, what does it cost?** v1 sidesteps the renderer entirely by
running AI only in headless server mode (§22), which makes every VRAM number in this document
unconditional. The deferred version brings back two regimes, yield-before-the-renderer-allocates,
and a poison contract that degrades instead of exiting. None of it is due yet; all of it is
cheaper if the engine keeps to the discipline of never assuming it owns the process.

**Which work is safe to distribute?** §22 raises it and recommends restricting hosting to
background cognition so user profiles never reach a stranger's machine, but that is a
recommendation, not a decision. It shapes what the global queue can contain and therefore what
distributed compute is actually worth.

**What stops hosting being farmed?** Paying game currency for idle GPU time creates an incentive
to fake it — a client that claims work, returns plausible-looking output, and collects. Verifying
inference cheaply is genuinely hard (redundant execution, spot-checking against a trusted node,
staking). Nothing here addresses it, and it is an economy question as much as a technical one.

**What is in `EngineSnapshot`, exactly?** §22 says it is the mind document's distilled state
vector, but the contents are unspecified — posture, affect, threat, current intent and attention
target are guesses. The set matters: too little and the game must query deep state on the frame
thread (the thing the snapshot exists to prevent), too much and publishing it becomes a cost on
every tick. It should be derived from what Battle Cities actually renders and reacts to.

**How does the game get backpressure?** `submit_perceive` never blocks, which means an
over-producing game silently grows a queue. The engine needs to tell the caller it is behind —
a `PumpStats` field, a rejected `Ticket`, or a queue-depth reading on the snapshot — and the
game needs a defined response. Right now the API's only failure mode is `429` over HTTP, with no
in-process equivalent.
