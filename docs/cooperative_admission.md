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

### 2.1 Where a turn goes: the machine table

A turn goes where the deployment's machine table says (`docs/static_placement.md`
§3): each machine filled to its `min` in rank order, then round to each one's
`max`, then a wait at the gateway. The members still learn and advertise their
rates in `CAPACITY` (`prefill_tok_s`, `decode_step_next_us`, `decode_step_wide_us`,
`prefill_tok_s_wide`, `prefill_backlog`); their own admission uses them, and the
status page shows them. Placement reads none of them.

Placement by advertised finish, which this replaced, was undone by the figures it
read. On 2026-11-02 a live turn went to WhiteWorld on a predicted 28.3 s against
BlackWorld's 31.0 s: BlackWorld had been charged 5,074 tokens of the characters'
work, which the live turn outranks, and WhiteWorld — paging — advertised a prefill
rate it no longer had. The turn took seven minutes.

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
number wins. The policies §3 compared placed by measured figures; what places
now is the machine table (rule 2).

1. **Place only on current state, and only at boundaries.** A unit goes only to
   a member that holds its state through its latest turn, and runs there to its
   end.
2. **The machine table decides.** A unit goes where `host/machines.yaml` puts it
   (`docs/static_placement.md` §3–§4): the fill to each machine's `min` in rank
   order, then round to its `max`; a draw by the picture preference.
3. **Never overfill a card.** Every unit on a card counts against it, whichever
   app placed it. Work that fits on no card waits at the gateway, most urgent
   first.
4. **Conversations first.** A card busy with a live conversation hosts no
   character while a free card can (`docs/static_placement.md` §5).
5. **Move only on what has held.** A member's fitness to host must hold for a
   minute before the cast moves for it; a card stays busy for two minutes after
   its last conversation.

### 4.1 Where each rule is enforced

| Rule | Enforced by |
|---|---|
| 1 | zend: a claim per turn, the next turn only among members holding the conversation complete (§2.2). npcd: a character moves only at its turn boundary (`hand_back`), and the plan lends nothing to a member that is syncing or lagging. A picture goes among the members holding it. |
| 2 | `cluster/src/gateway/placement/{conversation,draw}.rs` |
| 3 | the card count and the queue (§4.2, §4.3) |
| 4 | the planner (`cluster/src/gateway/planner.rs`) |
| 5 | the planner's `SETTLE` and `RETURN_AFTER` |

### 4.2 A card, and when it is full

**A card is a process.** Each process draws a random `card` id once
(`cluster::card::this`) and every member it runs sends it in its `HELLO`
(protocol 7). zend's member and npcd's member in one motherd share it, so the
gateway counts them as one card. A member that names none (zero) — a peer
gateway — is a card of its own.

**What a card has placed** is the sum over its members, across every app, of
the requests the gateway has handed them and they have not finished
(`in_flight`), and of the draws among them (`drawing`).

**A card is full for a unit when** it is drawing (a draw holds the card's
admission while it runs), when it is held for a waiting draw, or when its
machine's units of work — `in_flight` less the draws — have reached the
machine's `max` (`docs/static_placement.md` §3.2).

**A key's holder is never refused.** A request whose key a member holds goes to
it (rule 1 before rule 3). It counts against the card, and is never queued.

### 4.3 The queue

A unit that fits on no eligible card waits at the gateway in one queue per
gateway:

- **Order.** By class, then by arrival. **Latency** is every zend request — a
  live chat, a passthrough or MCP conversation — and a member's ask for a peer
  (a draw's content check). **Deadline** is every other request.
- **Release.** Whenever a card's count falls (an answer ends) or a `CAPACITY`
  arrives, the gateway takes each waiting unit out in order and places it. One
  that still does not fit is put back, and does not hold back those behind it
  that fit elsewhere.
- **Bound.** A conversation and a draw wait as long as their client does. Any
  other unit that has waited 30 s — the catch-up hold of §5.4 — is answered
  `503` with `Retry-After`. A client that goes away is dropped from the queue;
  a unit placed for a client already gone is uncounted at once.

### 4.4 The plan

The gateway plans the cast every 5 s, and at once when a card is handed its
first live conversation. Each npcd member is weighed by its machine's
`characters` share; a card busy with a live conversation lends nothing while a
free card can host, and every card lends by its share when every hosting card is
busy. The full rule is `docs/static_placement.md` §5.

A member syncing or lagging lends nothing (rule 1), and neither does one whose
`CAPACITY` says `process_loading`: another tenant of its process is still
loading — zend priming, say — and a character's turns there would take the
shared engine from the load. It serves, keeps its quorum place and takes its own
leases as ready; the cast runs on the members that have loaded.

The gateway sends the weights to every npcd member as `PLAN { epoch, weights }`
(protocol 7) whenever they change, and to a member as it is admitted. npcd's
authority lends by the plan's weight for each member (`replicated_substrate.md`
§9.5), so rendezvous moves only the characters whose holder changed, at their
next turn boundary (§9.3). A member keeps the last plan it was sent; with none,
it lends by the admitted power.

**Worlds do not follow the plan.** A world stays where its lease is
(`docs/static_placement.md` §6): moving a world reloads it, and a character's act
reaches its authority as one request wherever the world is.

### 4.5 Changing only on what has held

- **Busy at once, free after two minutes.** A card is busy from its first live
  conversation, and free again once it has had none for 120 s (`RETURN_AFTER`),
  the pause while a person reads and types.
- **Fitness settles.** A member's change of fitness to host moves the cast only
  once it has held for 60 s (`SETTLE`); its first fitness, once it has caught up,
  counts at once.
- **A plan is sent only when it changes.** A steady cluster keeps its epoch,
  and nothing moves.

### 4.6 What is logged

Every line below is at `INFO` under the target `cluster::gateway::placement`,
inside a `place` span naming the request's id (`req`), its app and its
dispatch, so the lines of one request join on `req`:

| Line | When | Says |
|---|---|---|
| `routing standing` | the first pick for an app, and each pick after its members' standing changed | every member, `routable` or `passed over` with each reason — loading, upgrading, fault, draining, disabled, lagging, syncing, closing, silent |
| `placed` | a `Best` or draw pick | how (`fill`, `round`, `full`, `draw`), and every candidate's machine, rank, `min`, `max`, units, drawing and reserved |
| `routed` | a request placed by something other than the policy: a key's holder (`holder`), a canary's share (`canary`), an `Any` request or an errand dealt in turn (`turn`) | the member and why |
| `routed among the members declaring the route` | an `Any` request whose route not every ready member declares | the route, the members declaring it, and the pick |
| `queued` | a unit fits nowhere | its class and the queue's depth |
| `released` | a queued unit is placed | the member and how long it waited |
| `refused` | a queued unit times out | how long it waited |
| `answered` | the answer's body ends, or its client goes away | the member, its status, the time to the first byte and to the end, and its bytes |

and under `cluster::gateway::plan` a `cards` line — each card's live
conversations and whether it is busy — on a plan whose cards differ from the
line last written (the plan runs every few seconds, and an unchanged line each
time buried every other), and a `plan` line when the plan's epoch moves,
naming the weights.

Every entry of the coordination log — the one the cluster page and the audit
read: a member admitted, closed, lagging, its lane drained, in sync, silent,
reporting again, a catch-up asked — is also written at `INFO` under
`cluster::gateway::coord`, naming the app and the member, so a member leaving
the routing reads beside the placement lines it changed. A catch-up frame
dropped at a member's lane cap is a `WARN` under `cluster::gateway::relay`, at
the socket's first drop and at each power of two after it.

A member logs the replication it does itself: `answered a catch-up` (the
requester, the ranges, the records listed and the records gone), `a fetch was
answered short` (`WARN`: records asked for that it did not send, by count and
span), `a catch-up finished` (with the span of `oseq`s it wanted, so the same
gap fetched again and again shows as the same span), and a stall of its own
receipts — `WARN` once events in flight have stood for a minute without a
`RECEIPT` covering any (again every ten minutes while it lasts), `INFO` when
they are receipted again.

npcd logs each lease `lend()` changes — the character, its old and new holder,
the weights it saw — and how long a new holder waited for the old holder's
writes before it woke the character.
