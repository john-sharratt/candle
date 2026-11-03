# Static placement

Where every unit of work in the cluster runs — a conversation's turn, a
character, a world, a picture — decided from a table the deployment states, not
from figures the members measure.

This replaces the measured placement and planning of
`docs/cooperative_admission.md` (§2.1, §4.2–§4.5) and `Best` in
`docs/replicated_substrate.md` §5.3, and the world-lease handover of §9.2
there. A member's own admission — how its scheduler forms waves, prefill and
decode — is untouched.

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
  backlog a card advertised counted every prompt at `High` or `Urgent`
  (`DecodePriority::is_live`) and every `Normal` prompt held 30 s. The 5,074
  tokens were the characters' work; a live turn is `Urgent` and outranks all of
  it. Without them BlackWorld finishes in about 24 s.
- **WhiteWorld's figures were stale.** It was paging at ~20,000 pages a second.
  Forty seconds later it advertised 2 t/s of prefill.
- **Once there, the conversation could not leave.** Every later request went to
  WhiteWorld by holder (rule 1): WhiteWorld was lagging on replication, so no
  other member ever held the conversation complete.
- **The characters did what the plan said, in the wrong direction.** WhiteWorld
  became latency-busy, so it lent nothing, and the whole cast ran on BlackWorld —
  the reverse of what the deployment wants.

Every one of these is a measurement standing in for a fact the deployment
already knows: BlackWorld is the strong card and WhiteWorld the weak one, and
that does not change from minute to minute. So the placement reads it from a
table.

## 2. The machine table

`host/machines.yaml`, one file, identical on every machine — the same rule as
`web/web.yaml` and `host/roles.yaml`. The daemons compile it in
(`host::machines::MACHINES`); the gateway reads it from `cluster.machines_file`
in `web/web.yaml`, and refuses to start when the file is missing or does not
parse. Each machine is keyed by the host name its members report at admission
(`MemberView::host`), matched in any case; a member whose machine the table
does not name is admitted and given no work.

```yaml
BlackWorld:
  conversations: {rank: 1, min: 4, max: 8}
  pictures: 2
  characters: 3
  builds: 1
WhiteWorld:
  conversations: {rank: 2, min: 1, max: 2}
  pictures: 1
  characters: 1
  pinned_experts_gib: 28
  builds: 2
AlienWorld: {}                 # an agent only, and a standby one
```

| Key | Means | Absent |
|---|---|---|
| `conversations.rank` | the order the deterministic fill visits machines in (§3.2); lower first | takes no conversations |
| `conversations.min` | how many units it is filled to, in rank order, before any round robin | 0 |
| `conversations.max` | how many it takes at most; beyond every machine's `max`, a unit queues (§3.4) | equal to `min` |
| `pictures` | its picture preference (§4); lower first | draws nothing |
| `characters` | its share of the cast while it is free of conversations (§5) | hosts no characters |
| `pinned_experts_gib` | how much host RAM its expert cache may page-lock (§7) | derived (§7) |
| `builds` | its build preference; lower first. A build or test job (`repo_job_start` with no `node`) goes to the agent on the most preferred machine with a free slot, before a dedicated agent elsewhere and before idle cores, and a lineage it holds stays there (`gateway::agents::choose`). A standby agent (`cluster.standby_agents`) still runs only what no other can | after every machine that has one |

The gateway reads `conversations`, `pictures`, `characters` and `builds`; motherd reads
its own machine's `pinned_experts_gib` at startup, before any tenant loads the
model. Changing a number is changing the deployment: the gateway takes it at
its next start, the daemons at their next build.

## 3. Conversations

### 3.1 What counts

The table places the **model's** units: every request of `zend` and `npcd`
that is work for its member (`Best`, `Sticky` and `Owner` dispatch, but an
errand). A machine's **count** is the units in flight on its card — every app's
requests the gateway has handed the card's members and they are still
answering, draws aside (`CardLoad::work`). A chat streams for as long as it
decodes, so a live conversation is counted for the whole of its turn.

A task delegated to a worker runs inside its zend member, turn after turn,
without passing the gateway: it is neither placed nor counted towards `min` and
`max`. It does make its card busy (§5.1).

An app that runs no model — the agents, keep, the MCP aggregator — is not
placed by the table: a unit of one goes to its member with the fewest requests
in flight, then the lower node.

### 3.2 Placement

A machine is **eligible** for a unit when it is ready, holds the unit's state
(rule 1 of `docs/cooperative_admission.md` §4, unchanged), and its card is
neither drawing nor held for a waiting draw (§4). A member **lagging** on
replication is passed over while another with room can take the unit — weighed
among the members with room, so a lagging one still takes what every other is
too full for, and after the holders of a released conversation are found, so a
lagging member that alone holds a conversation still takes it.

A unit goes to no machine the table gives nothing. When none of the members
that may take work now is on a machine that takes conversations, the unit
**waits** while the table names any machine that does — its member still
loading, behind, or not yet joined, it comes to take it. Only when **no machine
of the table takes conversations at all** — none with a `conversations` entry —
is the unit **unplaced**: the gateway refuses it, `502` with no retry, since
there is no room to wait for (`Registry::unplaced`, over
`MachineTable::any_takes`). A narrower pool that comes out unplaced — a
member's ask for a peer without the sender, a released conversation among its
holders — falls through to the wider one, as an empty pool does. A canary of an
app that runs no model, alone and on a drawing card, makes the unit wait for
the draw to end.

Among the eligible machines below their `max`:

1. **The fill.** In `(rank, node)` order, the first whose count is below its
   `min`. With the table above, the first four go to BlackWorld and the fifth to
   WhiteWorld.
2. **The round robin.** Once every one is at its `min`: the next after the
   app's cursor, in `(rank, node)` order, wrapping; the cursor moves to it. Each
   app has its own cursor.
   With the table above, the sixth goes to BlackWorld, the seventh to WhiteWorld
   — full at 2 — and the eighth to tenth to BlackWorld, full at 8.
3. **None below `max`:** the unit queues (§3.4).

### 3.3 Stickiness and return

- **A turn runs where it is placed, to its end** (rule 1). A key's holder is
  never refused: a request for a conversation its member holds goes there,
  whatever the counts say.
- **At a turn's boundary the conversation is placed afresh**, by §3.2, among the
  machines that hold it complete. One that spilled to WhiteWorld goes back to
  BlackWorld at its next turn once BlackWorld holds it and is below its `min`.

### 3.4 The queue

A unit no machine can take now waits in the gateway's **forward queue**
(`cluster/src/gateway/queue.rs`): accepted and held, asynchronously, before it
is forwarded to any member. A member's scheduler sees a unit only once the
gateway forwards it, and forms its waves exactly as before.

- **Order.** A conversation, and a member's ask for a peer, before other work;
  within each, arrival.
- **Release.** Whenever a card's count falls (an answer ends, a draw ends) or a
  `CAPACITY` arrives, the gateway takes each waiting unit out in order and
  places it; one that still fits nowhere is put back, and does not hold back
  those behind it that fit elsewhere. The cards the waiting draws hold (§4) are
  worked out once per walk, not once per unit.
- **A client's conversation turn and a client's draw wait as long as their
  client does** (`Placing::waits_for_its_client`): zend's turn route,
  `POST /v1/chat/completions` (`TURN_ROUTE`), or a draw, that no member sent. A
  client that goes away is dropped. Everything else — a read or an upload under
  a conversation, an MCP tool call, a member's ask for a peer — has no person
  waiting on a turn behind it, and is answered `503` after the 30 s hold
  (`HOLD`).

## 4. Pictures

A draw needs a card that reaches its size (`guest_reach`, which the card's VRAM
fixes) and holds that card's admission while it runs.

- **One at a time.** A card draws one picture at a time: a drawing card takes no
  other draw, and no other unit (§3.2). A burst takes each card that reaches it
  once and waits for the rest.
- **A card with nothing in flight first.** Among the reaching cards that may
  draw — not drawing, on a machine with a `pictures` preference — one with no
  unit in flight wins, whatever the preferences: a draw on a card mid-turn
  stalls that turn's stream for as long as it draws. Then one not lagging —
  weighed after the turn in flight, so an idle card that lags draws before a
  busy one that does not — then the lowest `pictures`, then the lower node.
  With the table above, a picture goes to WhiteWorld while BlackWorld answers
  conversations.
- **When every card that may draw it has a turn in flight, the draw waits, and
  holds its card.** The card it would take is **reserved** (`draw_target`): it
  takes no new unit but a draw, the turns already there run to their end, and
  the draw starts when they have. Units arriving meanwhile go to the other
  machines, or queue. The reservation is derived from the queue, so it ends
  with the draw that made it — placed, or its client gone.
- **A reservation is bounded.** A conversation whose claim the card holds still
  lands there between its tool rounds (rule 1), so the turns there may never all
  end. A draw that has waited `RESERVE_FOR`, 60 s, takes its card as it stands
  at the next release.
- **Every card that may draw it is drawing:** the draw waits for one to finish.
- **No machine draws at all** — none in the table with a `pictures`
  preference: the draw is unplaced, and refused (§3.2). While one does, a draw
  no member that may draw now can take waits for it.

## 5. Characters

### 5.1 Busy and free

A card is **busy** while zend works on it — a live conversation's request in
flight, or a conversation whose claim its zend member holds, a worker's
delegated task among them — from the first, and **free** once it has had none
for `RETURN_AFTER`, 120 s: the pause while a person reads a reply and types the
next message, or between a task's turns. A delegated task runs inside its member
and never passes the gateway, so its claim is how the plan sees it
(`Registry::card_views`); the cast leaves a card running a task exactly as it
leaves one serving a conversation. A member that has left makes no card busy —
the plan counts only members present. **A member that crashed and came back
releases the claims of its previous process itself** (`MemberClaims::of_own_node`,
`docs/replicated_substrate.md` §5.4): its `ADMIT` still names its node the holder,
since a gateway keeps a claim until its holder lets it go, and the new process,
holding none of them, `UNCLAIM`s each. Its card is then free `RETURN_AFTER`
after, rather than busy for good, and the gateway, the restarted member and its
peers agree on every key's holder.

### 5.2 Where they run

Characters are lent by weighted rendezvous, as before
(`docs/replicated_substrate.md` §9.5), over the npcd members each weighed by the
gateway's `PLAN`:

- by its machine's `characters` share while its card is free;
- by **zero** while its card is busy and some free card can host;
- by its share again when every hosting card is busy — the cast then runs
  beneath the conversations everywhere, below them in the scheduler's bands.

A member hosts only while it is fit — ready, in sync, not lagging, nothing in
its process loading (`MemberView::may_host`) — and the plan acts on a change in
that only once it has held for `SETTLE`, 60 s, so a member flipping in and out
of lag moves nothing. A member's first fitness, once it has joined and caught
up, counts at once: there is nothing yet for it to flip back from.

### 5.3 A fast vacate

When a card becomes busy:

1. **The plan goes out at once.** The pick that hands a card its first live
   conversation plans the cast there and then, and sends a changed `PLAN` to
   every npcd member — not on the next 5 s tick.
2. **The authority re-lends at once.** It lends every beat from the newest plan:
   every character whose rendezvous holder changed is given its `next`.
3. **The holder hands each back at its first boundary.** A boundary is the end
   of a turn; a turn already running finishes, which on the strong card is
   seconds. A character between turns is handed back at once — waiting on an
   engine, reflecting, or not running at all: one held here and never woken is
   handed back without being woken. A dream runs as a task of its own and is not
   waited for; putting the character down stops its loop.
4. **The new holder wakes it** once it has applied what the old holder wrote
   (`HandBack.through`, the old holder's own records). The character's K/V and
   recurrent state are already there: every sealed turn replicates to every
   member. A member the plan lends to is never one lagging on replication, so
   that wait is short. **A hand-back adds to what the lease waits for and never
   replaces it** (`npc_lease::hand_back`): the lease's `waits` name the handing
   holder's writes and every earlier holder's still owed, one entry per origin,
   its latest. A holder that applied A's writes but never woke the
   character, or never caught up with them at all, wrote nothing of it; A's
   writes stay owed, and the next holder wakes it only once it holds every
   origin the lease names (`NpcLease::ready`). Hand A → B → C with B moved on
   before it woke the character: C waits for A's writes and B's.

### 5.4 A fast return

When the card becomes free (§5.1), the plan restores its share, and the
characters rendezvous puts there move back the same way. With the table above,
the cast runs 3:1 BlackWorld to WhiteWorld while nothing is live, leaves
BlackWorld within a turn of the first live conversation, and returns two
minutes after the last.

### 5.5 A move, once asked, completes

A lease's `next` stands from the moment it is set until the holder hands the
character back, whatever rendezvous names meanwhile — the holder again, or a
third member (`npc_lease::lend`). It is withdrawn only when the next holder can
no longer host — gone from the group, or weighed nothing by the plan — and then
the holder runs the character as before. On 2026-11-02 a member flipping in and
out of lag set and cleared Umbra's `next` every minute or two, and Umbra ran
nowhere.

## 6. Worlds

A world's authority is wherever its lease is, and it stays there.

- **Asked for while nobody holds it**, by its proposer: the ready member
  admitted at the greatest power, then the lower node (`world_lease::proposer`) —
  BlackWorld, at 734 against WhiteWorld's 245.
- **Kept by its holder.** The holder renews the lease every 10 s for as long as
  it runs. No rank, plan or rendezvous hands it over.
- **Moved only by loss.** A holder that leaves the group, or is cut off, lets
  its lease run out, and the proposer among the rest takes the world from its
  records (§9.2's reload, unchanged). Back, the old holder follows. A member that
  runs no characters hands each world it holds over, as before.

A character's act reaches its world's authority as one request wherever the
world is, so a world on one card serves a cast on both.

## 7. Host memory

`pinned_experts_gib`, when the table gives it, is the expert cache's page-lock
budget (`candle::vram::state_expert_pin`), held to the derived pool: a pin may
lower the budget below what the machine leaves after its fixed obligations,
never raise it above. When the table gives none, the budget is the derived pool
(`host_ram_budget_from`):

```text
tier_pool     = total − weights_reserved − PAGEABLE_RESERVE − OS_KEEP − KV_WARM_FLOOR
expert_pinned = tier_pool
```

- **`KV_WARM_FLOOR` comes out of the machine.** It is pageable, and was left out
  of the sum for that reason; but RAM the OS must page to keep is RAM the
  machine does not have.
- **A machine whose engine holds more says so.** What the engine itself holds
  unpinned — warm K/V past its floor, held payloads, the substrate's in-memory
  state — follows its traffic, not its RAM: on WhiteWorld, 2026-11-02, 8.5 GB of
  warm K/V and 10.8 GB of held payloads beside 47.9 GB of pinned experts on a
  64 GB machine, paging at ~20,000 pages a second; on the 31.5 GiB laptop's gate,
  next to nothing. No reserve derived from the machine is right for both — one
  sized for WhiteWorld leaves the laptop no expert pool — so WhiteWorld states
  its pin, 28 GiB, in the table.

## 8. Placement never runs out of places

Each rule that takes a machine out of consideration — drawing, reserved, lagging,
busy, full — could, combined with the others, leave a unit with nowhere to go
while a card sits idle. The invariants that prevent it:

1. **Pass over, never wait, while another machine can take it.** A unit is never
   held for its preferred machine; it goes to the next eligible one. Only when
   none has room does it queue.
2. **Every lagging member is still a member.** When every candidate lags, the
   unit is placed among them as ever.
3. **A queue is released by every event that could make room** — an answer
   ending, a draw ending, a `CAPACITY` — not by a timer.
4. **A reservation is for one waiting draw**, derived from the queue, and ends
   with it — or after `RESERVE_FOR`, when the draw takes its card as it stands.
   It never holds a card idle: it waits only for turns already running there.
5. **No room and no machine are different outcomes.** A unit waits only when a
   machine that takes it could come to have room — one still loading, or not
   yet joined, included; one that no machine of the table takes at all is
   refused, unplaced.
6. **Characters always have a home.** Busy cards lend again when no free card
   can host (§5.2), and a move whose target cannot host is withdrawn (§5.5).
7. **The holder of a key is never refused** (rule 1). A conversation mid-turn on
   a card that is then reserved or drawing finishes its turn there.
8. **A machine the table does not name gets nothing**, so a mis-named machine
   shows on the gateway's page as admitted and idle rather than taking work at a
   guessed size.

## 9. Scenarios

With the table of §2.

| Scenario | Conversations | Characters | Worlds | Pictures |
|---|---|---|---|---|
| Nothing live | — | 3:1 BW:WW | stay where held | WW |
| One live conversation | BW | leave BW within a turn, all to WW | stay | WW |
| Its person stops typing | stays on BW | stay on WW for 120 s, then 3:1 | stay | WW |
| Five live | 4 BW, 1 WW | both busy: beneath them on both, 3:1 | stay | both mid-turn: waits for WW's turn to end, holding WW |
| Ten live | 8 BW, 2 WW | beneath them on both | stay | waits for a card's turns to end, holding it |
| Eleven live | the eleventh queues at the gateway | unchanged | stay | — |
| BW draws a 2048² picture with turns in flight | BW reserved: new conversations to WW, then queued | BW's pause for the draw | stay | BW once its turns end |
| WW lagging | no new units on WW while BW can take them | none newly lent to WW once it has held 60 s | stay | BW |
| WW down | all on BW, up to 8 | WW's characters wait for it, as before; the rest on BW | WW's worlds go to BW when their leases run out | BW |
| BW down | WW, up to 2, the rest queue | all on WW | BW's worlds go to WW | WW, up to 1024² |
| A member just restarted | eligible once ready — nothing to learn | lent to once caught up | stay | its preference |

## 10. Where it lives

| Piece | Where |
|---|---|
| The table and its reader | `host/machines.yaml`, `cluster/src/machines.rs`, `host/src/machines.rs`, `web/src/config.rs` (`machines_file`) |
| The fill, the round robin, an app that runs no model | `cluster/src/gateway/placement/{conversation,least}.rs` |
| Room: drawing, reserved, `max` | `cluster/src/gateway/placement/room.rs` |
| Pictures and the reservation | `cluster/src/gateway/placement/draw.rs`, `Registry::draw_target`, `Hub::pick_in` |
| Lagging passed over, among those with room | `cluster/src/gateway/placement.rs` (`prefer_current`) |
| The forward queue | `cluster/src/gateway/queue.rs`, `Hub::place`, `Hub::release_waiting` |
| The plan: shares, busy, `RETURN_AFTER`, `SETTLE`, the plan on the busy event | `cluster/src/gateway/planner/`, `Hub::pick_in` |
| A move that completes; a held, unwoken character handed back | `npcd/src/authority/{npc_lease,cast}.rs` |
| Worlds that keep their lease | `npcd/src/authority/{world_lease,driver}.rs` |
| The expert pin | `candle-core/src/vram/host_probe.rs`, `host/src/run.rs` |

`CAPACITY` still carries a member's power and learned rates; the status page
shows them, and a member's own admission uses its rates. Placement reads
neither.

## 11. Deploying it

**motherd first, then the gateways.** No frame changes, so either side
admits the other; what changes is who releases a stale claim. The gateway keeps a
claim until its holder lets it go, and it is a member — `MemberClaims::of_own_node`
— that releases a claim of its node that its process does not hold
(`docs/replicated_substrate.md` §5.4). A member of the earlier build does not: sent
its previous process's claim in its `ADMIT`, it takes it in and serves the key as
its own without ever announcing it. So behind a new gateway, a member of the earlier
build that restarts holding a claim leaves it standing — its card busy, its peers
answering `409` for it — for as long as that gateway runs: nothing else releases it,
and only a gateway restart forgets it. No gateway relays a node its own claims, so
`ADMIT` is the only way one reaches it. With motherd rolled out first, every member
that restarts behind the new gateway releases its own.

The table's entries are checked strictly: an earlier gateway refuses a table
naming a key it does not know (`builds`, at the build that introduced it), so a
gateway is restarted on the new table only with the new `web.exe`.

The new `web.exe` reads `cluster.machines_file` from `web/web.yaml`, which the
earlier one refuses as an unknown field. A machine that restarted its earlier
gateway with that line removed must have it back when the new gateway is installed:
without it the gateway has no table and places no unit.
