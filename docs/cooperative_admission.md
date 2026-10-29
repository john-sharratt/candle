# Cooperative admission

How a zend cluster places turns across members so that the sum is better than
each member admitting alone: decodes concentrated where weights are resident,
prefill kept off the decode set's card, and every conversation still getting
its fair turn.

The member's own admission (`candle-conversation` `scheduler::admit`) decides
what one card runs. This document is the layer above it: which card a turn goes
to. Code: `coop/` — the cost model, its fit from each member's wave log, and the
cluster simulator that compares the policies below before any of them is built
into the gateway.

## 1. What the members measure

`coop-lab` reads a member's scheduler `wave` lines (one per 2 s window, per
channel: forwards, rows, tokens, attended KV, ms per forward, the window's
expert hit rate) and reports two things: a non-negative least-squares fit of

```text
ms per forward = base + token·T + attention·MKV + resident·GB_hit + link·GB_missed
```

where the expert bytes a forward reads are the expected distinct experts its
`T` tokens select (`49 layers × 512 × (1 − (1 − 10/512)^T) × 2.61 MB`), split by
the hit rate; and the measured step time bucketed by rows per forward, which is
what the simulator uses where the fit explains a channel poorly.

### BlackWorld (RTX PRO 5000 72 GB, PCIe 5.0), builds 3436–3442, 2,829 windows

Decode, by rows per forward:

| rows | decode alone: ms/step | t/s | beside prefill: ms/step | t/s |
|---|---|---|---|---|
| 1 | 53.0 | 19 | 76.6 | 15 |
| 2–3 | 129.0 | 22 | 110.7 | 24 |
| 4–7 | 119.8 | 48 | 168.8 | 30 |
| 8–15 | 94.0 | 139 | 91.4 | 117 |
| 16–31 (~23) | 75.7 | 304 | 81.5 | 270 |

Prefill forwards average ~200 tokens at 77–145 ms: 0.8–2.5K t/s, against
4.8K t/s at full width in the forward gate.

Fits: prefill (all windows) `13.7 ms + 0.218 ms/token + 12.7 ms/MKV + 2.6 ms/GB
missed`, R² 0.79 — the 0.218 ms/token margin is the gate's 4.6K t/s. Decode
alone `38.1 ms + 104 ms/GB missed`, R² 0.72; decode over all windows R² 0.48,
because a decode beside prefill waits on that prefill's expert traffic, which
its own counts do not show.

What this says for placement:

- **Decode throughput is nearly linear in batch size** on a member up to the
  widths seen (~23 rows ≈ 300 t/s, one row ≈ 19 t/s). Twenty-four decodes on
  one card deliver what two cards of twelve each deliver, and leave the
  second card free.
- **Decode beside prefill is 1.1–1.5× slower per step.** Separating the two
  across cards removes that.
- **A single-row decode costs 53 ms in production** against single-digit ms in
  the gate: per-step overhead outside the forward the gate measures. It is the
  member's to fix, and it makes small decode batches the worst use of a card.

WhiteWorld (RTX 3090 24 GB, PCIe 3.0) is to be measured the same way; its gate
figures are ×1 decode 138, ×16 338–544, prefill 2.3K t/s, and its expert zone
is a third of BlackWorld's.

### The link

The members share one 1 GbE path (~110 MB/s) through the gateway, RTT ~1 ms.
What crosses it when a turn's work moves:

- its new KV — on the order of 12 KB a token (12 full-attention layers),
  ~24 MB for a 2K-token prompt, ~0.2 s;
- its recurrent snapshot — 36 DeltaNet layers × 48 value heads × 128 × 128 at
  f32, **113 MB**, about 1 s.

Both replicate anyway: every sealed turn's records go to every member. So a
turn placed on another member costs nothing the substrate was not already
sending, provided that member already holds the turn before it (§2.2).

## 2. Mechanisms

### 2.1 Advertised load figures

Each member says in `CAPACITY`, as it runs, what more work would cost it, from
its own measurements (§1):

- **decode-heavy and prefill-heavy** — how loaded it is on each: the rows it is
  decoding and the prompt tokens it has queued;
- **the marginal value of one more of each** — what one more decoding turn,
  and one more prompt, would do to its own throughput at its current
  residency, from its own fitted curve.

The gateway compares these numbers and nothing else. It models nothing of a
member's insides; the member that knows its card is the one that says what
the card can do.

How it is built (protocol 4): `CAPACITY` carries `prefill_backlog` (queued
prompt tokens), `prefill_tok_s`, `decode_step_next_us`, `decode_step_wide_us`
and `prefill_tok_s_wide`. The last four are what the member's admission has
learned (`phase_ring::learned_rates`): the prefill rate at the width it runs,
the decode step with one more row, and the decode step and prefill rate with
`PLACEMENT_WIDE` (16) more rows. The gateway's `Best` estimates each
candidate's finish for a turn of the size the members report (protocol 5:
`CAPACITY`'s `prompt_tokens` and `reply_tokens`, the mean prompt of the turns a
member's scheduler begins and the reply such a turn decodes — one size for every
candidate, the mean of what they report, 2,048 and 512 where none has learned
one): the backlog, plus one prompt for each
request handed to it since its last `CAPACITY` and for this one, at the time
per token those requests and this one leave it; then the reply at the step
they make, over as many steps as the reply takes at the tokens a step commits
on that card (`tokens_per_step_milli`, above a token where it speculates; a
token when unlearned) — per card, since what a step commits is the card's
where the turn's size is the turn's. Both are on the line through the one-more and wide figures — time
per token for prefill, so the extension never reaches a rate of zero —
extended past the wide point, and flat when the wide figure is no slower. A
request handed earlier is not charged again: it is queued, and in the
backlog, or decoding, and in the step. A card that copies every routed expert
steepens with width, so a burst priced at its idle step alone lands whole on
it (run 24's shift-decode-2: 48 turns at 41.2 chunks/s, the step 70 ms
advertised and 450–580 ms delivered); and a placed turn decodes beside the
prompts behind it, so prefill slows too (run 35's heavy-read: 24 turns charged
at an idle 1,771 tok/s that then drained 213–463, all on one member). It picks the soonest
(`gateway/registry.rs`, `finish_us`). A candidate that has not learned both
rates — one fresh from a restart — is probed: it takes a turn while it is
answering none, one at a time, which is what it learns from, and every other
turn goes by finish among those that have. Ranked by power instead, run 36's
restarted WhiteWorld took 12 of heavy-decode's 48 turns at 25–45 s each
against BlackWorld's 10–18 s. With none learned, it ranks by power per request.

### 2.2 A lease per turn, placed among the replicated

A conversation's lease (its claim, §5.4 of `replicated_substrate.md`) is held
for **one turn**: its prefill and its decode, the time its state has to be on
a card. At the turn's boundary, once the turn is durable, the lease is
released with `through`, and the next turn is placed afresh.

- **Only members that hold the conversation through its latest turn are
  candidates.** A member that has not yet confirmed replication up to the turn
  just ended is skipped, never waited on. Placement therefore never stalls a
  turn on the link, and a move costs nothing the substrate was not already
  sending.
- Among the candidates, the turn goes by §2.1's figures.
- A turn runs where it is placed, start to finish: no part of it is sent to
  another member mid-turn.

How it is built:

- **The claim follows use.** zend's member holds a conversation's claim (§5.4
  of `replicated_substrate.md`) only while something uses it
  (`AppMember::follow_use`). A request uses it until its response body ends (the
  host's sticky check); a turn uses it from submit until its last record is
  written (`zend::turn_claims`), because a dialogue turn outlives a client that
  goes away. Its last record is its `Commit`, which the persistence writer
  writes when the turn goes cold — not at its seal — so a finished turn holds
  the claim, its reply's stream already closed, until that lands, however long
  it takes. A conversation whose slot stays live between tool rounds pins its
  newest turn hot, and that turn goes cold only when a persistence pass finds
  it unpinned: under load its handover waits seconds to minutes. Released
  sooner, the claim named the previous turn's commit, and a member holding
  only that took the conversation while the newer turn's snapshot had already
  reached it. When the last use ends the claim goes as `UNCLAIM`, its `through`
  the `oseq` of the conversation's own last state record — its newest commit —
  as the app names it (`AppMember::key_through`), or this member's whole
  origin's when the app names none. npcd and the agent keep their claims until
  idle: their work runs on after the request that started it.
- **Current enough is a conversation dimension.** Every member hears every
  release (`AppMember::on_unclaim`) and awaits the key: once it has applied
  every record of that conversation through the release's point and holds
  every turn of it whole — no gap turn (`remote/completeness.rs`) — it reports
  the key in `COMPLETE` (`AppMember::report_complete`). A key stays reported,
  re-sent whole on every admission, until anyone claims it again. A member
  behind on everything else the releaser wrote may take the conversation all
  the same; one missing any of the conversation's own records, K/V included,
  may not — a turn with missing K/V is a gap, never rebuilt from its tokens.
- **The gateway places the next request among those holding the turn.** It
  keeps each key's last release (who, through which `oseq`, and the members
  that reported it complete since) and each member's applied set from its
  `ACK`s. A request for a key nobody holds goes `Best` among the placeable
  members that reported it complete, or that hold the releaser's records
  through that `oseq` — the releaser itself always does — and the canary takes
  it only if it does. With none holding it the request goes among them all,
  which is the most anyone has (§2.4). A release every placeable member already
  holds is forgotten, and a new release forgets the reports made against the
  one before.
- **A member forgets what it no longer holds.** Every way a claim stops being
  this member's — its last use, idleness, a lease handed back, another member's
  later claim — calls the app's release hook (`AppMember::on_release`). zend
  drops the conversation's cached sequence and passthrough transcript there: a
  conversation that moved away and back would otherwise be appended to as it
  stood when it left, missing the turns written elsewhere. The next turn served
  here resumes from the substrate.
- A request the gateway sends after a release reaches a member that applies the
  records relayed ahead of it before serving (`APPLY_WAIT`); placement only ever
  chooses one that already has.
- **A stored picture is placed the same way, with no turn to hold it.** npcd's
  member takes a picture's key and lets it go the moment the picture is stored
  (`Authored::hand_over_image`), so an `image_edit` naming it — its `image`
  argument, an MCP subject — is placed among the members holding that release's
  records, exactly as a conversation's next turn is. A draw names nothing and
  goes `Best`.

### 2.3 Fairness

Every policy is judged on per-conversation slowdown (the turn's time against
the same turn alone on an idle cluster): its p95 and its Jain index across
conversations, and the longest any turn waited for admission. A policy that
raises throughput by starving the conversations it does not concentrate is
rejected.

### 2.4 Failure

A member dropping hands its claims back (§12.4). A turn running on it starts
again, placed among the members that hold its conversation through the turn
before. A conversation whose only up-to-date copy was on the dropped member
waits for nothing: its next turn is placed among the members that hold it
through the latest turn they have, which is the last turn that member
replicated before it went.

## 3. Experiments

The simulator (`coop/src/sim/`) replays a workload — conversations whose
turns are prefill-then-decode segments, sized from the zend-load rungs —
against members whose step time is their measured curve. Each turn's end
replicates its state to every other member over the link above, FIFO, and a
member becomes a candidate for that conversation once its copy has arrived
(§2.2). The policies:

1. **today** — sticky; a new conversation goes `Best` by
   `power / (1 + in_flight)`;
2. **concentrate decodes** — every turn on the strong card while it reports
   room (rows under its maximum, decode hit rate above 0.9), otherwise `Best`;
3. **earliest advertised finish** — each candidate advertises what this turn
   would cost it now (its prefill backlog plus this prompt at its prefill
   rate, then the reply at the step time one more row gives it), and the turn
   goes to the soonest. §2.1's figures used as the only guide.

Reported per policy: aggregate t/s, first-token p50/p95, longest stall, Jain
index, turns per member, lease moves, and the same with one member dropping
mid-run.

### 3.1 Results (`coop-lab sim`, 24 conversations, seed 1)

BlackWorld as measured; **WhiteWorld assumed** from its gate figures (decode
slope 3.2× BlackWorld's, prefill half its rate, a 9 GB expert zone) until its
wave log is fitted, so every WhiteWorld-dependent figure below moves when it is.

| scenario | policy | t/s | first token p95 | Jain | turns BW/WW | moves |
|---|---|---|---|---|---|---|
| chat | today | 141 | 1.8 s | 0.942 | 60/12 | 0 |
| chat | concentrate | 160 | 2.7 s | 0.997 | 72/0 | 0 |
| chat | earliest finish | **161** | 2.5 s | **0.997** | 71/1 | 1 |
| read | today | 85 | **6.8 s** | 0.995 | 40/8 | 0 |
| read | concentrate | 81 | 9.8 s | 0.997 | 48/0 | 0 |
| read | earliest finish | **86** | 8.7 s | 0.992 | 45/3 | 3 |
| chat + read | today | 94 | 3.0 s | 0.969 | 50/10 | 0 |
| chat + read | earliest finish | **106** | 5.0 s | **0.990** | 59/1 | 1 |

With WhiteWorld dropping at 30 s, today's first-token p95 rises to 22–28 s
(its turns restart); earliest finish stays at 2.7–9.0 s.

What it says so far:

- **Concentrating decodes beats today's power split** by 13% on chat and
  evens the slowdown (Jain 0.94 → 1.00): today's split sends a sixth of the
  conversations to the card that serves them several times slower.
- **Earliest advertised finish is the best single policy** on every mix but
  read's first-token tail, and it is the simplest to build: each member
  computes its own figure and the gateway compares numbers. It moves a lease
  at most three times a run, and only to a member already holding the state.
- Every result assumes the member's own scheduler forms full-width prefills;
  production forwards average ~200 tokens (§1), which no placement fixes.

## 4. The placement invariants

Five rules place every unit of work in the cluster. A **unit** is one turn of a
conversation, one turn of a character, one ingest unit, or one draw. The rules
are the gateway's alone: they decide **where** a unit runs and never **how** a
card runs it. A card's admission and its priority bands
(`DecodePriority`) are a black box to them. When two rules disagree, the lower
number wins.

1. **Place only on current state, and only at boundaries.** A unit goes only to
   a member that holds its state through its latest turn, and runs there to its
   end.
2. **The soonest finish wins.** A unit goes to the eligible card that would
   finish it soonest.
3. **Never overfill a card.** Every unit on a card counts against it, whichever
   app placed it. Work that fits on no card waits at the gateway, most urgent
   first.
4. **Latency first, behind a fence.** Latency work takes the cards it needs.
   Movable work — characters, ingest — fills what is left. The NPC floor's
   capacity is fenced off, and latency work never takes it.
5. **Change slowly.** A card's share of movable work moves only after its
   trigger has held for a while, and in coarse steps.

A draw needs no rule of its own: it is a unit only some cards can take (§5.3 of
`replicated_substrate.md`, reach), and a card counts as full while it draws, so
rule 2 sends it to the strongest card not drawing. Its content check is a
latency unit placed away from the card that draws.

### 4.1 Where each rule is enforced

| Rule | Enforced by |
|---|---|
| 1 | zend: a claim per turn, the next turn only among members holding the conversation complete (§2.2). npcd: a character moves only at its turn boundary (`hand_back`), and the plan (§4.4) lends nothing to a member that is syncing or lagging. A picture goes among the members holding it. |
| 2 | `Best` by finish (`cluster/src/gateway/placement/best.rs`, §2.1); a draw by `placement/draw.rs` |
| 3 | the card count and the queue (§4.2, §4.3) |
| 4 | the planner (§4.4) |
| 5 | the planner's hysteresis (§4.5) |

### 4.2 A card, and when it is full

**A card is a process.** Each process draws a random `card` id once
(`cluster::card::this`) and every member it runs sends it in its `HELLO`
(protocol 7). zend's member and npcd's member in one motherd share it, so the
gateway counts them as one card. A member that names none (zero) — a peer
gateway — is a card of its own.

**What a card has placed** is the sum over its members, across every app, of
the requests the gateway has handed them and they have not finished
(`in_flight`), of those handed since their last `CAPACITY` (`placed`), and of
the draws among them (`drawing`).

**A card is full for a unit when:**

- it is drawing, for anything but a draw (a draw holds the card's admission
  while it runs); or
- one more row would take its decode step past the **latency target**,
  `LATENCY_STEP_US` = 500 ms — two tokens a second for every stream on the card.
  Loaded cards run well above an idle gate's step (run 24's BlackWorld at 364 ms
  with 48,918 tokens queued, run 35's WhiteWorld at 208 ms for one row) and still
  serve; a stream slower than two tokens a second serves nobody. The step with
  `n` more rows is the line
  through the card's advertised one-more and sixteen-more steps (§2.1,
  `step_with_us`). The rows the card may take beyond what it decodes now are the
  largest `n` that stays under the target, less the rows the fence holds back
  (§4.4), less the units handed to it since its last `CAPACITY`. A card whose
  wide step is no slower than its one-more step has no width limit from the
  line; its free slots (`slots_free`) bound it as before.

**A card with nothing placed is never full** — except while it draws. Its own
background work can slow its step past the target, and the gateway cannot see
inside the card to tell; refusing it the first unit would leave that unit
waiting on work the planner is about to move away.

**A key's holder is never refused.** A request whose key a member holds goes to
it (rule 1 before rule 3). It counts against the card, and is never queued.

### 4.3 The queue

A unit that fits on no eligible card waits at the gateway in one queue per
gateway:

- **Order.** By class, then by arrival. **Latency** is every zend request — a
  live chat, a passthrough or MCP conversation — and a member's ask for a peer
  (a draw's content check). **Deadline** is every other request.
- **Release.** Whenever a card's count falls (an answer ends) or a `CAPACITY`
  arrives, the gateway walks the queue in order and places each unit that now
  fits. One that still does not fit stays, and does not hold back those behind
  it that fit elsewhere.
- **Bound.** A unit that has waited 30 s — the catch-up hold of §5.4 — is
  answered `503` with `Retry-After`. A client that goes away is dropped from
  the queue; a unit placed for a client already gone is uncounted at once.

### 4.4 The plan

Every 5 s the gateway plans the movable work. For each card it reads:

- **latency load**: the zend requests in flight on it;
- **bulk load**: the ingest zend's member on that card reports outstanding
  (`CAPACITY.bulk_backlog`, protocol 7);
- **power**, as `CAPACITY` says (§5.3 of `replicated_substrate.md`).

A card is **busy** when its latency or bulk load has held (§4.5). Then:

- **The fence** is the cards with the least latency load, taken until their
  power reaches the **NPC floor**, `NPC_FLOOR_PERCENT` = 15% of the cluster's.
  On a fenced card, latency work may take only the share of its width the floor
  does not need: the floor's power over the card's, as rows held back from §4.2.
- **The weights.** A card that is not busy lends characters by its whole power.
  A busy card lends nothing — unless it is fenced, when it lends its fence share,
  or unless every card is busy, when every card lends by its whole power and
  characters run beneath the conversations everywhere. A member syncing or
  lagging lends nothing (rule 1).

The gateway sends the weights to every npcd member as `PLAN { epoch, weights }`
(protocol 7) whenever they change, and to a member as it is admitted. npcd's
authority lends by the plan's weight for each member in place of the power the
member was admitted at (`replicated_substrate.md` §9.5), so rendezvous moves
only the characters whose holder changed, at their next turn boundary (§9.3).
A member keeps the last plan it was sent; with none, it lends by the admitted
power as before.

**Worlds do not follow the plan.** A world's authority stays where rendezvous
over the admitted powers puts it: moving a world reloads it, and a character's
act reaches its authority as one request wherever the world is.

### 4.5 Changing slowly

- **Busy quickly, free slowly.** A card is latency-busy from the first plan that
  sees a zend request in flight on it, and free again once none has been for 10
  minutes; bulk-busy once its ingest has been outstanding for 60 s, and free
  again 5 minutes after it empties. A conversation pauses while its person
  reads and types; ingest ends and restarts unit by unit. Neither pause moves a
  character.
- **Coarse steps.** A weight is a quarter of the card's power — 0, ¼, ½, ¾ or
  all of it — and moves at most one quarter towards its target every 30 s.
- **A plan is sent only when it changes.** A steady cluster keeps its epoch,
  and nothing moves.

### 4.6 What is logged

Every line below is at `INFO` under the target `cluster::gateway::placement`,
inside a `place` span naming the request's id (`req`), its app and its
dispatch, so the lines of one request join on `req`:

| Line | When | Says |
|---|---|---|
| `placed` | a `Best` or draw pick | how (`finish`, `probe`, `power`, `draw`), the turn size, every candidate's figures and card count |
| `routed` | a unit placed by something other than the policy: a key's holder, a canary's share | the member and why |
| `queued` | a unit fits nowhere | its class and the queue's depth |
| `released` | a queued unit is placed | the member and how long it waited |
| `refused` | a queued unit times out | how long it waited |
| `answered` | the answer's body ends, or its client goes away | the member, its status, the time to the first byte and to the end, and its bytes; the same `req`'s `placed` line holds the finish `Best` predicted for it |

and every 5 s, under `cluster::gateway::plan`, one `cards` line — each card's
members, placed count, latency and bulk load, width and fence — and a `plan`
line when the plan's epoch moves, naming each weight that changed and why.

npcd logs each lease `lend()` changes — the character, its old and new holder,
the weights it saw — and how long a new holder waited for the old holder's
writes before it woke the character.
