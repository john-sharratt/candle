# Static placement

Where every unit of work in the cluster runs — a conversation's turn, a
character, a world, a picture — decided from a table the deployment states, not
from figures the members measure.

**Status: design, not yet built.** When it is built it replaces the placement
and planning sections of `docs/cooperative_admission.md` (§2.1, §4.2–§4.5) and
`docs/replicated_substrate.md` §5.3's `Best`, and the world-lease handover of
§9.2 there; those documents are rewritten in the same change. A member's own
admission — how its scheduler forms waves, prefill and decode — is untouched.

## 1. Why

On 2026-11-02 a live conversation on code.tokera.com ("Codebase Tour",
`9675b8e3`) was placed on WhiteWorld, the 3090 box, while BlackWorld, the RTX
PRO 5000, was free to take it. Its first turn took about seven minutes, at
12 t/s. The gateway's own placement line says why:

| | WhiteWorld | BlackWorld |
|---|---|---|
| backlog charged | 0 | 5,074 tokens |
| prefill rate advertised | 180 t/s | 710 t/s |
| predicted finish | **28.3 s** | 31.0 s |

- **BlackWorld was charged for work the turn would have gone ahead of.** The
  backlog a card advertises counts every prompt at `High` or `Urgent`
  (`DecodePriority::is_live`) and every `Normal` prompt held 30 s. The 5,074
  tokens were the characters' work; a live turn is `Urgent` and outranks all of
  it. Without them BlackWorld finishes in about 24 s.
- **WhiteWorld's figures were stale.** It was paging at ~20,000 pages a second.
  Forty seconds later it advertised 2 t/s of prefill.
- **Once there, the conversation could not leave.** Every later request went to
  WhiteWorld by holder (rule 1): WhiteWorld was behind on replication, so no
  other member ever held the conversation complete.
- **The characters did what the plan says, in the wrong direction.** WhiteWorld
  became latency-busy, so it lent nothing, and the whole cast ran on BlackWorld —
  the reverse of what the deployment wants.

Every one of these is a measurement standing in for a fact the deployment
already knows: BlackWorld is the strong card and WhiteWorld the weak one, and
that does not change from minute to minute. So the placement reads it from a
table.

## 2. The machine table

`host/machines.yaml`, one file, identical on every machine — the same rule as
`web/web.yaml` and `host/roles.yaml`. Each machine is keyed by the name its
members report at admission (the name the gateway's status page shows); a
member whose machine the table does not name is admitted and given no work
(§3.1, §5, §6).

```yaml
BlackWorld:
  conversations: {rank: 1, min: 4, max: 8}
  pictures: 2
  characters: 3
  pinned_experts_gib: 120      # optional
WhiteWorld:
  conversations: {rank: 2, min: 1, max: 2}
  pictures: 1
  characters: 1
AlienWorld: {}                 # an agent only
```

| Key | Means | Absent |
|---|---|---|
| `conversations.rank` | the order the deterministic fill visits machines in (§3.2); lower first | takes no conversations |
| `conversations.min` | how many live conversations it is filled to, in rank order, before any round robin | 0 |
| `conversations.max` | how many it takes at most; beyond every machine's `max`, a conversation queues (§3.4) | equal to `min` |
| `pictures` | its picture preference (§4); lower first | draws nothing |
| `characters` | its share of the cast while it is free of conversations (§5) | hosts no characters |
| `pinned_experts_gib` | how much host RAM its expert cache may page-lock (§7) | derived (§7) |

The gateway reads `conversations`, `pictures` and `characters`; motherd reads
its own machine's `pinned_experts_gib` at startup, before it joins. The table is
read at start, never measured and never rewritten: changing a number is
changing the deployment, and takes a restart.

## 3. Conversations

### 3.1 What counts

A **live conversation** on a machine is a zend conversation whose claim its
member holds: from the request that begins a turn until the turn's last record
is written (`docs/cooperative_admission.md` §2.2, "the claim follows use") —
its tool rounds included. A machine's **count** is how many it holds. The
gateway already knows it: it is the zend requests it has handed the member and
the claims the member has reported.

Only a conversation a person is waiting on — one whose band is `Urgent` —
counts towards `min`, `max` and busy (§5.1). A task delegated to a worker,
decoding at `normal` while nobody watches (`DecodePriority::METADATA_KEY`), is
placed by the same order but counts towards neither: a worker running for an
hour would otherwise hold the strong card's characters off it for that hour.

### 3.2 Placement

A machine is **eligible** for a conversation's turn when it is ready, holds the
conversation's state through its latest turn (rule 1 of
`docs/cooperative_admission.md` §4, unchanged), is not lagging on replication,
and is not drawing (§4). A new conversation holds no state, so every ready,
current, non-drawing machine with a `conversations` entry is eligible for it.

Among the eligible machines:

1. **The deterministic fill.** Visit the machines in `rank` order and give the
   turn to the first whose count is below its `min`. With the table above, the
   first four live conversations go to BlackWorld and the fifth to WhiteWorld.
2. **The round robin.** Once every eligible machine is at its `min`, the turn
   goes to the next machine, from a cursor the gateway keeps, whose count is
   below its `max`. The cursor walks the machines in `rank` order and moves past
   the machine it gave the turn to. With the table above, conversations six to
   ten go WhiteWorld (now full at 2), BlackWorld, BlackWorld, BlackWorld,
   BlackWorld.
3. **Nobody below `max`:** the turn queues (§3.4).

An eligible machine is never skipped for a reason the table does not give. A
machine that is not eligible — drawing, lagging, not holding the state — is
passed over, never waited on, as long as any other machine can take the turn
(§8).

### 3.3 Stickiness and return

- **A turn runs where it is placed, to its end** (rule 1). A key's holder is
  never refused: a request for a conversation its member holds goes there,
  whatever the counts say.
- **At a turn's boundary the conversation is placed afresh**, by §3.2, among the
  machines that hold it complete. A conversation that spilled to WhiteWorld goes
  back to BlackWorld at its next turn once BlackWorld holds it and is below its
  `min`.

### 3.4 The queue

A turn no machine can take now waits in the gateway's **forward queue**: the
request has been accepted and is held, asynchronously, before it is forwarded
to any member. It is not the members' admission: a member's scheduler sees a
turn only once the gateway forwards it, and forms its waves exactly as it does
today.

- **Order.** Arrival, within a class: a conversation a person waits on first,
  then delegated tasks, then other deadline work.
- **Release.** Whenever a machine's count falls, a draw ends or a member
  becomes eligible, the gateway walks the queue in order and forwards each turn
  that §3.2 now places. One that still fits nowhere stays, and does not hold
  back those behind it that fit elsewhere.
- **No time bound.** A queued conversation waits as long as its client does; a
  client that goes away is dropped from the queue. The 30 s `503` of
  `docs/cooperative_admission.md` §4.3 no longer applies to conversations.
  The client is told it is queued, and its position, on the stream it opened.

## 4. Pictures

A draw needs a card that reaches its size (`guest_reach`, which the card's VRAM
fixes) and holds that card's admission while it runs.

- **Placement.** Among the machines that reach the draw and are not drawing,
  the lowest `pictures` value. A card draws one picture at a time.
- **A draw prefers a card with no live conversation.** Among the reaching
  cards not drawing, one whose count is zero wins over one with a turn in
  flight, whatever their `pictures` values — a draw on a card mid-turn stalls
  that turn's stream for the draw's length. With the table above, a picture
  goes to WhiteWorld, and BlackWorld keeps answering conversations.
- **A card drawing takes no new conversation** (§3.2), and its characters pause
  for the draw, as today.
- **A draw only one card reaches.** A picture larger than WhiteWorld reaches
  (above 1024²) can only go to BlackWorld. If BlackWorld has a turn in flight,
  the draw does not wait behind conversations forever: BlackWorld is
  **reserved** for it — it takes no new conversation, the conversations its
  members hold finish their current turns, and the draw starts at the first
  moment none is in flight. Conversations arriving meanwhile go to the other
  machines, or queue (§3.4).
- **Every reaching card drawing:** the draw queues behind the draws, in arrival
  order, and is released when one ends.

## 5. Characters

### 5.1 Busy and free

A machine is **busy** while it has a live conversation (§3.1), from the first
one, and becomes **free** once it has had none for `RETURN_AFTER`, 120 s. The
pause covers a person reading a reply and typing the next message; it is the
only hysteresis in the design.

### 5.2 Where they run

Characters are lent by weighted rendezvous, as today (`replicated_substrate.md`
§9.5), over the eligible machines with a `characters` share, each weighted:

- by its `characters` share while it is free;
- by **zero** while it is busy, when any free eligible machine has a share;
- by its share again when every machine with a share is busy — the cast then
  runs beneath the conversations everywhere, below them in the scheduler's
  bands.

Eligible is ready, current, and not lagging — a lagging that has held for
`SETTLE`, 60 s. A member flipping in and out of lag faster than that does not
move the cast; on 2026-11-02 WhiteWorld's weight flipped between 0 and 245 every
one to two minutes, and Umbra, asked to move and then not, never ran anywhere.

The weights are the gateway's `PLAN`, as today, with the table's shares in
place of measured power, and no quarter steps.

### 5.3 A fast vacate

When a machine becomes busy:

1. **The plan goes out at once.** The gateway sends `PLAN` on the event — the
   first live conversation on a machine — not on its next 5 s tick.
2. **The authority re-lends at once.** On a new plan, every character whose
   rendezvous holder changed is given its `next` in the same beat.
3. **The holder hands each back at its first boundary.** A boundary is the end
   of a turn. A dream, a reflection or a journal pass is cut, not waited for: it
   is work the character does on its own time, and it starts again where the
   character lands. A turn already running finishes — on the strong card that is
   seconds.
4. **The hand-back names the character's own records, not the member's.**
   `HandBack.through` is the `oseq` of the character's last own record
   (`AppMember::key_through`, as zend's `UNCLAIM` names a conversation's), so
   the new holder waits for the character's turns and journal, never for
   everything else its old holder wrote. Today it names the whole origin
   (`npcd/src/authority/cast.rs:218-222`), and a holder behind on replication —
   WhiteWorld, 256 MiB unsent — makes every character it hands over wait for all
   of it.
5. **The new holder wakes it** as soon as those records are applied. The
   character's K/V and recurrent state are already there: every sealed turn
   replicates to every member.

The cast leaves a busy machine within one turn of each character, and the
conversation has the card to itself.

### 5.4 A fast return

When a machine becomes free (§5.1), the plan restores its share, and the
characters rendezvous homes there move back the same way (§5.3). With the
table above, the cast runs 3:1 BlackWorld to WhiteWorld while nothing is live,
leaves BlackWorld within a turn of the first live conversation, and returns two
minutes after the last.

### 5.5 A move, once asked, completes

A lease's `next` is fixed from the moment it is set until the character is
handed back. A plan that would name a third holder, or the old one again,
waits for the hand-back and then moves the character on. The only exception is
a `next` that leaves (drops out of the group, or is ineligible for `SETTLE`):
then the move is withdrawn and the holder runs the character as before. A
character is never left held by a member that has stopped running it and has
not handed it back.

## 6. Worlds

A world's authority is wherever its lease is, and it stays there.

- **Taken once.** At start, the lowest-`rank` eligible machine with a
  `characters` share asks for every world nobody holds. A world held stays
  held: the holder renews it, and no plan, rank or rendezvous takes it away.
- **Moved only by loss.** A holder that leaves the group, or stops renewing, has
  its lease run out, and the next machine in rank order takes the world from
  its records (§9.2's reload, unchanged). A holder still in the group that has
  not renewed within the lease's term loses it the same way — the term, not the
  group, is what says whether it still runs the world. A member that shuts down
  or drains hands its worlds over before it goes.
- **The rendezvous handover is removed** (`npcd/src/authority/world_lease.rs`
  `Step::HandOver` on a changed proposer). Moving a world reloads it, and a
  character's act reaches its authority as one request wherever it is.

## 7. Host memory

`pinned_experts_gib`, when the table gives it, is the expert cache's page-lock
budget, exactly. When it does not, the budget is derived from the machine:

```text
tier_pool     = total − weights_reserved − PAGEABLE_RESERVE − OS_KEEP
                      − KV_WARM_FLOOR − ENGINE_RESERVE
expert_pinned = tier_pool
```

The derivation today (`candle-core/src/vram/host_probe.rs:468-504`) leaves out
the last two terms:

- **`KV_WARM_FLOOR`** is documented as beside the pool and never subtracted
  from it. Pageable RAM the OS must page to keep is RAM the machine does not
  have.
- **`ENGINE_RESERVE`** is the engine's own unpinned heap — the warm K/V arenas
  above their floor, held payloads, the substrate's in-memory state.
  `PAGEABLE_RESERVE`'s 10 GiB was measured on the 31.5 GiB laptop against the OS
  and other applications only. On WhiteWorld, 2026-11-02, the engine held 8.5 GB
  of warm K/V and 10.8 GB of held payloads beside 47.9 GB of pinned experts, on
  a 64 GB machine, and paged at ~20,000 pages a second. `ENGINE_RESERVE` is sized
  from that measurement.

The warm K/V tier also ran at twice its budget there (8.46 GB against 4.29 GB,
`over_budget = true`): `purge_warm_to_budget` did not shed to the floor. That
is fixed with this, since an engine reserve sized for a tier that overruns its
budget is sized for nothing.

## 8. Placement never runs out of places

Each rule that takes a machine out of consideration — drawing, lagging, busy,
reserved for a draw, full — could, combined with the others, leave a unit with
nowhere to go while a card sits idle. The invariants that prevent it:

1. **Skip, never wait, while another machine can take it.** A conversation is
   never held for its preferred machine; it goes to the next eligible one. Only
   when no eligible machine has room does it queue.
2. **A queue is released by every event that could make room** — a count
   falling, a draw ending, a member becoming eligible, a reservation ending —
   not by a timer.
3. **A reservation is for one draw and ends when it starts.** It never holds a
   card idle: it waits only for turns already running there to end.
4. **Characters always have a home.** Busy machines take them back when no free
   one remains (§5.2), and a move whose target goes away is withdrawn (§5.5).
5. **The holder of a key is never refused** (rule 1). A conversation mid-turn on
   a machine that is then reserved or drawing finishes its turn there.
6. **A machine the table does not name gets nothing,** so a mis-named machine
   shows on the gateway's page as admitted and idle rather than taking work at a
   guessed size.

## 9. Scenarios

With the table of §2.

| Scenario | Conversations | Characters | Worlds | Pictures |
|---|---|---|---|---|
| Nothing live | — | 3:1 BW:WW | stay where held | WW |
| One live conversation | BW | leave BW within a turn, all to WW | stay | WW |
| Its person stops typing | stays on BW | stay on WW for 120 s, then 3:1 | stay | WW |
| Five live | 4 BW, 1 WW | both busy: run beneath them on both, 3:1 | stay | WW, or BW if WW draws |
| Ten live | 8 BW, 2 WW | beneath them on both | stay | queued behind conversations only for a draw only BW reaches (§4) |
| Eleven live | the eleventh queues at the gateway | unchanged | stay | — |
| BW draws a 2048² picture | BW reserved: new conversations to WW, then queued | BW's pause for the draw | stay | BW once its turns end |
| WW lagging for 60 s | no new turns on WW | none newly lent to WW; its own stay | stay | BW |
| WW down | all on BW, up to 8 | WW's characters wait for it, as today; the rest on BW | WW's worlds move to BW when their leases run out | BW |
| BW down | WW, up to 2, the rest queue | all on WW | BW's worlds move to WW | WW, up to 1024² |
| A member just restarted | eligible once ready and current — nothing to learn | lent once eligible for `SETTLE` | stay | its preference |

## 10. What changes in the code

Removed:

- `Best`'s finish ranking, its probe of a member that has not learned, and its
  power-per-request ranking — `cluster/src/gateway/placement/{best,finish,turn_size}.rs`.
- The 500 ms step test for a full card — `placement/room.rs` `width` and
  `LATENCY_STEP_US`; a card is full at its `max`.
- The planner's quarter steps, `STEP_EVERY`, and the NPC floor's fence —
  `planner.rs`, `planner/fence.rs`.
- `CAPACITY`'s `power` and learned rates as placement inputs; a member's
  scheduler still learns and logs them, for its own admission.
- The world lease's rendezvous handover — `npcd/src/authority/world_lease.rs`.

Added:

- `host/machines.yaml` and its reader, shared by the gateway and motherd.
- The deterministic fill, the round robin and its cursor — `placement/`.
- The forward queue's unbounded wait for conversations and its position report
  — `gateway/queue.rs`.
- The picture preference, the conversation-free preference and the one-draw
  reservation — `placement/draw.rs`.
- The plan sent on the busy event, `RETURN_AFTER`, `SETTLE` — `planner/`.
- `HandBack.through` by the character's own records; cutting a dream or
  reflection at a hand-back; a fixed `next` — `npcd/src/authority/{cast,npc_lease}.rs`.
- The world lease's take-once and term-based loss — `npcd/src/authority/{driver,world_lease}.rs`.
- `pinned_experts_gib`, `KV_WARM_FLOOR` and `ENGINE_RESERVE` in the host budget,
  and the warm tier's purge to its floor — `candle-core/src/vram/host_probe.rs`
  and the K/V tier.

## 11. Tests

Each building block is a unit test against the table, with no cluster:

- The fill and round robin: the order the first `Σ max` conversations land in,
  and the next queuing, for the table of §2 and for one machine alone.
- Eligibility: a drawing, lagging, or not-current machine passed over; a
  holder never refused.
- The queue: released by each event of §8.2, in order, a later turn that fits
  elsewhere not held behind one that fits nowhere; a client gone dropped.
- Pictures: preference, the conversation-free preference, the reservation and
  its end, a draw no card reaches.
- The plan: busy at the first conversation, free `RETURN_AFTER` after the last;
  a lag under `SETTLE` moves nothing; every machine busy lends by share.
- Leases: a `next` fixed until the hand-back; withdrawn when its target leaves;
  `through` the character's own records; a world never handed over while held,
  and taken when its term runs out with the holder still in the group.
- Memory: the budget given, and derived, for the 31.5 GiB, 64 GB and 194 GB
  machines, as exact byte counts.

And `npcd/tests/worlds.rs`'s two-member runs: the cast leaves a member when a
conversation lands on it and returns after it ends, with every character
running somewhere throughout.
