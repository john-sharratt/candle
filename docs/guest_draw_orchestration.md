# One call: draw now, authorize on a peer, release on both

A draw today runs in series on one member: the content check (four one-token
questions on the resident model, `npcd/src/compliance.rs`), then the guest
drain (`candle-conversation/src/guest/drain.rs`). Measured on BlackWorld
(2026-10-28), a prompt edit spent 14–28 s in npcd, of which the drain was
6.3–8.2 s (load ~3 s, run 2.6–4.5 s); the rest was the check's four turns
queued behind live traffic. The check cannot overlap the drain on one member —
a drain stops that member's engine — but it can on another.

So the check becomes its own step yielding a **signed authorization** for a
prompt, and a draw is **started at once, held, and released** only once it is
cleared. The member a draw landed on starts it, asks a peer for the
authorization at the same moment, and answers when both are done. The caller
makes one call; a caller that already holds an authorization passes it and no
check runs.

Two releases:

- **(a)** The authorization layer, the one-call orchestration, and one gateway
  routing rule. No wire change, but a gateway (`web`) release: `web` is not
  rolled out by the cluster and is restarted through each machine's keep,
  before the npcd members.
- **(b)** `MemberState::Paused`: while a draw holds a member's card, gateways
  route new work elsewhere. A wire change (protocol 7), §4.

## 1. Prompt authorization (a)

A general layer: anything that must be cleared before it is acted on asks for
an authorization of its content for a purpose.

```
POST /v1/prompt/authorize   {purpose, prompt}
  → 200 {authorization}
  → 403 {error: "prompt_declined", detail}
  → 503 {error: "check_unavailable", retry: true}
```

**Who may call it.** A member, over a cluster socket (`from_member`,
`host/src/cluster/up.rs:146`; the precedent is `npcd/src/authority/routes.rs`),
and an Admin. A member's call runs the check. An Admin's call with
`"mint": true` issues the authorization without the check — the Admin vouches
for content cleared another way (a curated set of portrait prompts, an
operator's own) — and is the one place an authorization is minted outside the
check. Users do not call it: a user-callable route would be a free oracle for
probing the filter (decided 2026-10-28).

**The token.** `{purpose, prompt_sha256, exp_ms, sig}`:

- `sig` is HMAC-SHA256 under `Join::derive_key("npcd prompt authorization")`
  (`host/src/cluster/mod.rs:627`), over exactly the bytes
  `"{purpose}\n{prompt_sha256_hex}\n{exp_ms}"`, verified in constant time —
  as image links are (`npcd/src/image_links.rs:115-146`), but a type of its
  own with its own key. With no cluster the key is minted per process, as
  image links' is (`api.rs:189`); an unclustered daemon checks locally anyway.
- `prompt_sha256` is over the exact `ImageRequest.prompt` handed to the guest.
  The route does no normalisation: a caller trims before it authorizes, as the
  MCP tools already trim (`npcd/src/mcp/args.rs`).
- `exp_ms` is ten minutes after issue; expired when `now >= exp_ms`, the rule
  image links use.
- It covers the **prompt only**, not a prompt edit's reference picture: a
  stored or uploaded picture is not checked today either.
- The key is derived from the group key, so the trust boundary is every holder
  of it: gateways, members, keep, the MCP aggregator, agents.

`compliance::check` becomes the one implementation behind a `ContentCheck`
trait, called with the state per call (no `Arc<Authored>` held in a trait
object stored on `Authored`). Its two callers today — `post_image`
(`guest_routes.rs:226`) and the portrait route (`portrait.rs:262`) — both go
through it. The existing error codes are unified: a check that could not run is
`check_unavailable` everywhere (`post_image` says `prompt_declined` with 503
today, `guest_routes.rs:237`).

## 2. The one call (a)

`POST /v1/image/generate` and `/v1/guest/image` (both `post_image`), and the
MCP tools that call it in-process — `image_generate`, `image_edit` in
`prompt` mode — take an optional `authorization`. The portrait route keeps
check-then-draw, through `ContentCheck` and the cache below: a portrait's
prompt is generated, its latency is not a user's (decided 2026-10-28).

On the member the call landed on (D):

1. **Authorization given:** verified — signature, `purpose == image`, hash of
   this prompt, not expired — before the stream opens. Invalid: refused with
   its status, naming why. Valid: draw and answer; no check.
2. **Restricted LoRA:** the existing rule — an Admin, no check — stands, as a
   `Clearance::AdminWaiver` constructible only from a passed
   `require(Admin)` (`guest_routes.rs:207-227`).
3. **None given, another npcd member routable:** submit the guest job at once
   and, at the same moment, send `POST /v1/prompt/authorize` up (`send_up`,
   `host/src/cluster/up.rs:46`, `read_for` 30 s). The gateway places it by
   `Best`, and **a request a member sends up never goes back to that member
   while another member of its app is routable** (§3). D chooses no peer: the
   gateway has the live capacity, D sees only admission weights (`MemberInfo`,
   `cluster/src/frame/types.rs:118`).
4. **None given, no other member routable:** check locally first, then draw —
   today's order. The gateway rule sends the request back to D when D is the
   only member, so D knows this from the group (`npcd_members`) and does not
   send it.
5. **The join.** The stream opens at once (`loading`, `step`, `waiting` as
   today), and when the draw is done and the verdict is not, an
   `{"event":"authorizing"}` line. Then:
   - **allowed** (a 200 the D re-verifies) → the picture is released: the
     `done` line as today, with the `authorization` in it, so a caller that
     retries can pass it back;
   - **refused** — a body whose `error` is `prompt_declined`, and only that →
     the picture is dropped unseen — never stored, linked, shown or logged by
     content — and the stream's last line is
     `{"event":"error","error":"prompt_declined","retry":false}`;
   - **peer failed** — any other non-200 (a gateway 403/404/503 `Problem`), a
     transport error, or no answer within 30 s → checked locally once the draw
     has finished (D's engine is free then), and joined on that; a local
     `Unavailable` ends the stream `check_unavailable`, `retry: true`;
   - **the draw failed** (`no_room`, `failed`) → the draw's error, as today,
     at once; no local check; a verdict still on its way is not waited for;
   - **D dies** → the stream drops; nothing was stored.
6. **Refused:** answered at once. The job is withdrawn behind the answer —
   cancelled while still queued (below); held by a drain, tried again every
   second, so a job the drain puts back to wait for room is taken then rather
   than drawn later; otherwise drawn and dropped.

Refusals move from a status code before the stream to an in-band error line
under 200, as every other draw failure already is. The MCP tools read it
through `outcome()` (`mcp/image.rs:375-386`); nothing else branches on the
status.

**The cost of drawing first.** The check takes 10–19 s under load
(`compliance.rs:555-558`) and a drain starts on the next scheduler pass, so a
refused prompt is usually drawn before it is refused: one drain (~6–8 s with
every world on that member stopped, and the card's KV evicted) instead of four
one-token turns. Accepted (decided 2026-10-28): refusals are rare, and every
allowed draw saves the check's whole latency. A refused-and-drawn job also joins its drain's
footprint, which can push a batch-mate to `no_room` — the same as any draw.

**Cancelling a queued job.** `GuestQueue::cancel(seq) -> bool`
(`candle-conversation/src/guest/queue.rs`) and `ConversationEngine::cancel_guest`
beside `submit_guest_watched` (`engine.rs:839`): removes the job while it is
still queued — including one requeued to wait for room — answers its receipt
`GuestError::Cancelled`, and **clears `waiting` and `retry_at_ms` when no job
with `waiting_since` remains**: `waiting` closes prefill admission
(`scheduler/prefill.rs:2144`), and a cancel that removed the last waiting job
without clearing it would close admission for good. `GuestError::Cancelled` is
added to the exhaustive matches (`guest_routes.rs:312-319, 491-498`).

**Where release happens.** `Held<T>` wraps the guest's outcome
(`GuestImage.png`, 1–3 MB at the MCP's 1024 side) and opens only with a
`Clearance` — `Authorized(Authorization)` matching its prompt, or
`AdminWaiver`. It opens inside `post_image`'s streaming task before the `done`
line is built (`guest_routes.rs:297-305`). Everything downstream — the MCP
tools' store, seed stamp (`npcd/src/image_seed.rs`), kept notes and edit chain
(`npcd/src/images/`) — sees only released output and is unchanged.

**The cache.** On D, keyed by `(purpose, prompt_sha256)`, bounded, an entry
dropped at its `exp_ms`. A retry that lands on D — a `no_room` retry, a redraw
— asks no one; one that lands elsewhere carries the `authorization` the `done`
line returned. Refusals are kept the same way: a prompt refused within ten
minutes is declined at once, neither drawn nor asked again — what keeps the
cost of drawing first to one drain per refused prompt.

**The stream after a refusal.** The job's progress reaches the stream through
a sender the call takes back when it answers, so a refusal is the stream's
last line and the stream ends with it, even while a drain still holds the
withdrawn job.

**Rollout order.** `ImageBody` denies unknown fields (`guest_routes.rs:52`): a
member that predates `authorization` answers a request carrying it 400. So the
gateway first, then every npcd member, then callers may send `authorization`.
The MCP tools ignore unknown arguments, so a model passing one to an old member
gets today's behaviour. The mock npcd (`web/src/mock/npcd/api.rs`) takes the
field too.

### Timing

One line per call at the join, `draw released` on `npcd::authorize`:
`drawn_ms` and `authorized_ms` from the call's start, its `source` (`peer`,
`local`, `given`, `cache`, `waiver`), and `total_ms` — the moment it was
released.

## 3. The gateway rule (a)

A member that wants a peer says so: the authorize request carries
`x-tokera-prefer-peer` (`cluster::member::PREFER_PEER`), honoured only on a
request that came up from a member — `from_member` set
(`web/src/proxy.rs:253`). Placed by `Best`, it goes to a member that is neither
the sender nor a canary certifying; when none is left, back to the sender
rather than to an unproven canary; and to the canary only when it is all there
is (`Registry::pick_from`, `best_for_peer`; `Hub::pick_from`). Such a request is
never a canary's share and never mirrored to one: certification samples
clients' traffic, and an unproven build is not handed the members' own. Every
other request — a member's without the header included — is placed as a
client's is, so zend's operations tools keep browser-identical routing.
Addressing by `x-tokera-node` is not used: it is Admin-only
(`web/src/cluster/route.rs:99-117`), skips routing and `in_flight`, and answers
404 for a node not on the gateway the stream reached.

## 4. Paused (b)

While a guest holds D's card, D's capacity says `Paused`; gateways route new
work to other members, best-effort: a request already on its way when the
change lands still arrives, and waits for the drain as today.

**When.** Paused for as long as the guest needs the card or is waiting for
it: from the guest job queued until the queue is empty and no job is waiting
for room. That covers the drain's whole stop (`drain_guests`,
`scheduler/guest_room.rs`), the shed ladder before the claim, and a job
requeued for room — which closes admission for up to `ROOM_PATIENCE`
(`drain.rs:157`) — and does not flap between an image drain and a matte drain
back to back. The engine sets a process-wide `GuestOccupancy` static beside
`reach::STATED` (`candle-conversation/src/guest/reach.rs:30`), cleared by an
RAII guard so a panicking guest cannot leave it set. A host task subscribes and
calls `capacity_changed()` on every `AppLink`, on the cluster socket runtime,
so `CAPACITY` goes out on the change. One engine serves both tenants, so zend
and npcd both report it.

**Precedence** in `member_state` (`host/src/cluster/capacity.rs:71`): Fault,
then Loading (tenant not ready), then Paused, then Ready — a zend tenant still
loading does not report Paused because npcd's draw holds the shared engine.
`capacity_for` (`cluster/src/member/app.rs:712`) tests `state.up()`.

**Up, not ready-for-new-work.** `MemberState::up()` is `Ready | Paused`, and
every reader that asks "is this member up" uses it:

| Reader | Where |
|---|---|
| clustered / table / standing-by | `gateway/registry.rs:410-435` |
| joining beside a serving member (sync) | `gateway/hub.rs:449` |
| in-sync judgement | `gateway/relay.rs:162` |
| catch-up answerer | `gateway/registry.rs:466` |
| aggregates | `gateway/registry.rs:609` |
| handover forgetting, `everyone_holds` | `gateway/registry.rs:256, 519-522` |
| any ready, app view | `gateway/hub.rs:1258-1265, 1361` |
| world and character rendezvous | `npcd/src/authority/cast.rs:34` |
| replication's ready count, governor, absent leases | `replicate/consensus.rs:421, 345-355`; `quorum_governor.rs` |
| forced-rollout quorum | `gateway/upgrades/plan.rs:108-113`; `hub.rs:985` |
| certification start | `gateway/upgrades/mod.rs:281` |
| status page, worlds view | `gateway/status.rs:249`, `worlds_view.rs:67`, `cluster/gateway.html:348, 361, 859` |

Live events reach a paused member as ever (`relay.rs:47` skips only
Upgrading and Fault).

**Routing.** One helper, `prefer(pool, |m| !m.paused())`, returning the pool
unchanged when no member qualifies — the shape of `ready_but` and
`within_reach` (`registry.rs:443-462, 711-727`). Applied in `best_but` after
`within_reach` (reach first: a draw only the paused card reaches still goes
there), in `best_for` after the holding filter (with `everyone_holds` on the
unfiltered pool), in `round_robin_but`, and in an unpinned `Ui` pick. Not
applied to a held key, nor to a pinned UI page — moving a user's pin on every
draw would switch their page between members. A paused member counts as
drawing: `draw_of` sorts by `drawing + paused`. A paused canary takes no share
while it is paused.

**Not `Draining`, not `Loading`:** draining is a member leaving and handing its
claims back (`replicated_substrate.md` §12.4); loading is a member not yet able
to serve.

**Observability and failure.** The status row shows `paused` and
`paused_since_ms`; past two minutes the member logs a warning — a drain that
never ends leaves its held keys waiting. A member that crashes while paused
leaves the group and returns Loading; nothing stale stays on a gateway. Voting
and lease ticks run off the scheduler thread and continue; world beats and
character turns on that member wait for the drain, as today.

**The bump.** `MemberState` is a byte on the wire and an unknown value is a
decode error (`frame/types.rs:106-110`), so (b) is protocol 7. A gateway
refuses a different protocol with 4426, which stops the member's socket for
good (`handshake/gateway.rs:33-44`, `close.rs:64, 76`), and gateways peer on
the same check (`peer/dial.rs:29`, `peer/accept.rs:40`). So: every gateway at
once; then every dialler on every machine staged by hand — motherd (zend, npcd,
its in-process agent), keep, mcp, standalone agents — since a motherd at 7
cannot join a gateway at 6 to be certified and rolled. Until a member is
replaced it stays off the cluster.

## 5. Modules and seams

npcd, one concern per file under `npcd/src/authorize/`: `token` (the
authorization, sign and verify), `clearance` (`Clearance`, `Held`), `cache`,
`check` (`ContentCheck`, the compliance implementation), `peer`
(`PeerAuthorizer`, the `send_up` call), `draw` (`GuestDraw`: submit, cancel),
`orchestrate` (the one call), `route` (`/v1/prompt/authorize`). The three
seams — `ContentCheck`, `PeerAuthorizer`, `GuestDraw` — are what the
orchestration's tests fake. The inline `crate::compliance::Denial` paths
(`guest_routes.rs:226-233`, `portrait.rs:262-266`) become imports. The
`image_generate` description stops saying "Prompts are checked first"
(`mcp/image.rs:94`).

## 6. Tests

(a):
- Token: sign and verify against raw expected bytes; another prompt, purpose,
  expiry or key fails; `now == exp_ms` is expired.
- `Held` opens only with a matching `Clearance`; `AdminWaiver` only from an
  Admin.
- `GuestQueue::cancel`: a queued job is removed and answered `Cancelled`; a
  waiting job's cancel clears `waiting` when it was the last; a taken job
  cannot be cancelled.
- The orchestration, with fake `ContentCheck`, `PeerAuthorizer` and
  `GuestDraw`: given-valid draws with no check; given-invalid refuses before
  the stream; peer allows → released with the authorization in `done`; peer
  refuses → dropped, nothing stored, `prompt_declined`; refusal while queued →
  cancelled; a refused job a drain puts back → withdrawn; peer 403 `Problem` /
  503 / timeout → local check after the draw, announced; no peer → local
  first; draw fails → its error; a prompt refused moments ago → declined with
  no draw and no check.
- The cache: authorizations and refusals kept for their lifetime and bounded.
- The gateway rule: a member's `Best` request asking for a peer goes to
  another member, never a canary while one is left, and to the sender when
  none is; a client's is placed as ever.
- The authorize route: member and Admin only; Admin `mint` skips the check.
- Live, two members: draws and prompt edits with the timing line; a refused
  prompt, confirming nothing is stored.

(b):
- `GuestOccupancy` set from queue to empty, cleared by the guard on every
  exit including a panicking fake guest.
- `member_state` precedence.
- Registry: a paused member gets no new work while another is routable, gets
  it when none is, keeps held keys and pinned pages, counts as drawing, is up
  for every reader in §4's table.
- Wire: `Paused` round-trips; protocol 6 refused.
