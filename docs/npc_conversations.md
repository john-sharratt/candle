# NPC conversations

Every NPC — Zen, the moon household, a Battle Cities citizen — talks one-to-one
with people and with other NPCs in **conversations**: preserved, parked
exchanges that run beside the NPC's main loop, as the same person, with the same
mood, mission and memory. This document is the design. It is grounded in the
code as it stands (anchors are `file:line` against `origin/main` 662baa9d8) and
names, for every piece, what exists and what is built.

---

## 1. Why

What a person sees today when they DM Zen on Discord (bot `128j9dsgheutk`, the
moon world):

- the same reply posted two to five times;
- replies that contradict the one before;
- in-world chatter flooding briefing channels;
- an act template leaking into speech (`that …`);
- a DM thread that is Zen role-playing at nobody.

The causes are structural, not prompt-level. A DM today is a phone thread the
main loop happens to notice:

- **The text never reaches the mind as a turn.** `deliver_messages`
  (`npcd/src/engine/environment.rs:390-475`) sends the mind a *notice*
  (`EventKind::Message{thread, from, unread}`, `:460-464`). The character must
  choose `read`, then `reply`, between room events, idle percepts and its
  mission.
- **What is posted is the act's intent, not words.** `enact::message` puts the
  raw `intent` on the thread (`npcd/src/engine/enact.rs:1044, 1054`), and the
  Discord mirror posts `spoken(&m.intent)` (`npcd/src/discord/dm/outbound.rs:184`).
  That function only strips a leading "that " and capitalises. Nothing ever
  decodes the words. `narrate_act` (`npcd/src/engine/mind.rs:963-1000`) is a
  synchronous render for the feed, and `reply` is not among the acts it
  renders (`tools::narrates`, `npcd/src/engine/tools.rs:1178-1183`). This is
  the `that …` leak.
- **Answering does not mark the thread read.** `message`/`reply`
  (`enact.rs:1031-1090`) never call `Threads::read`, so the URGENT unread
  reminder (`npcd/src/sim/unread.rs:21`, every 5 min) wakes the character on a
  message it has already answered.
- **The phone never shows the character its own replies.** `read_phone`
  (`enact.rs:363-372`) returns only unseen lines from others. Whether it
  answered survives only in the 32-exchange tail (`mind.rs:117`), which a day or
  chapter roll-over resets.
- **The loop guard runs after the repeat is posted.** The act is performed at
  `npcd/src/engine/runtime.rs:6673` and judged at `:6739-6799`, and the guard's
  state is in memory only (`npcd/src/engine/loopguard.rs:368-370`).
- **Threads themselves are fragile.**
  - A world keeps 64 non-permanent threads and drops the quietest
    (`npcd/src/sim/phone.rs:399, 524-538`).
  - A thread that is reopened gets a new id, and its seq restarts at 1
    (`phone.rs:541-566`).
  - `sign_off` leaves the thread, and a direct thread with one member closes
    (`phone.rs:834-845`).
  - Members are display names (`npcd/src/engine/mod.rs:1199-1212`).
- **NPCs talk to each other aloud.** `tell` is offered whenever anybody is in the
  room, and `shout` always (`tools.rs:287, 345`). NPC↔NPC talk is therefore room
  speech: `Happening::Said` (`npc-map/src/world.rs:1910-1930`), which nothing
  preserves and nothing concludes.

A conversation removes these by construction: each message is a turn of a
preserved timeline, and every turn decodes real words. §11 rebuilds the phone,
and §12 makes room speech for people only.

---

## 2. Decisions

| # | Decision |
|---|---|
| D1 | A conversation is **the NPC's mind loop on a message clock**: same persona, same schema family, same act catalog, same performer. The beat is replaced by the arrival of a message, and perception is narrowed to what changed since the last turn. |
| D2 | A conversation is **one-to-one**: an NPC and one party, which is a person or another NPC. Group channels stay on the main loop's messaging, which is fire-and-forget (§11). NPC↔NPC room speech in front of a person (§12) also stays on the main loop. |
| D3 | **Parked between turns.** A conversation holds no scheduler slot and no live `Sequence` between messages. It is a persisted substrate timeline that resumes on the next message. Thousands can exist, and only the ones decoding cost VRAM. |
| D4 | **Mood, mission, beliefs, persona and memory are shared** with the main loop. It is the same person in two places. |
| D5 | **Experience is journaled.** What an NPC does, hears, writes down and concludes, and the world changes that concern it, are lines of its `bots/<bot>` journal. This is an optional per-world setting (§5.0): the moon and Battle Cities both use it, sharing the one `journals` repository. |
| D6 | **Catch-up is a turn-scoped carried section plus a cited summary.** The journal lines since the conversation's last turn are carried into the turn, summarised with line citations into the user turn, kept for the reply, and then removed. |
| D7 | **Every turn decides, then decodes.** A forked call chooses the acts from the existing catalog. Then a real turn always decodes free text, with no tools, under a response template that matches the choice. The choice decides **where the text goes**: to the party (`reply`, `sign_off`) or into the journal (anything else). Every turn therefore leaves a message or a journal entry, and there is no "no reply". |
| D8 | **One tool base.** The decision uses the main loop's catalog: `reply`, `sign_off`, `reflect`, `promise`, `remind`, the `mission_*` acts, and `message`/`reach_out`. It is compiled for a thread instead of a room. |
| D9 | **A main-loop private `message` (or `reach_out`) starts, resumes or injects into a conversation.** It hands the conversation a *purpose*, and the conversation decodes the words. |
| D10 | **Conversations conclude.** A promise, a mission change or a hand-off is how a conversation changes what the NPC does. These are applied by the world's authority under authority, critical-step and thrash rules. |
| D11 | **Speech in the room needs a person in the room** (§12). NPCs reach each other by `message`. Battle Cities field play keeps `shout`. |
| D12 | **Every phone line is voiced.** Conversation lines are voiced by D7. Main-loop channel posts are voiced through the same template path (§11.4). No intent text reaches a thread. |
| D13 | **Duplicates are tolerated.** An occasional repeated message is accepted, and the system is eventually consistent. Losing a message, or writing one timeline from two members, is not accepted (§9). |

---

## 3. What exists that this is built on

| Need | Existing piece | Anchor |
|---|---|---|
| Character decode | `Minds`, `Live{sequence}`, `send_turn_with_options_async` | `npcd/src/engine/mind.rs:163-227, 1786-1789` |
| Act grammar, parser, performer | `grammar_for`, `act::parse`, `Runtime::perform` → `perform_here` | `mind.rs:1034`; `npcd/src/engine/act.rs:372`; `npcd/src/authority/perform.rs:272-369` |
| Side decode that writes nothing | `UnsealedAsker::ask(question, band, scope, options)` | `candle-conversation/src/conversation/ephemeral_turn.rs:69-112` |
| Response templates | the mind's `response` collection (596 templates, `top_k: 1` by provenance, locked for the decode) | `npcd/projection.yaml:108-124`; `npcd/src/projection.rs:361` |
| Runtime-offered candidates | `SelectionRule::Offered { selector, k }` | `candle-conversation/src/projection/schema.rs:1397-1409` |
| Banning a token in a turn | `TurnOptions.turn_banned_tokens` | `candle-conversation/src/turn.rs:62` |
| Carried sections per turn | `Carried::reconcile` | `candle-conversation/src/conversation/carried.rs:159` |
| Resume a timeline on a new slot | `resume_conversation_with_projection`; recurrent state through `restore_recurrent_state` | `candle-conversation/src/engine.rs:2505`; `candle-conversation/src/scheduler/mod.rs:7654-7790` |
| Snapshot durability | `drain_durable_snapshots` | `candle-conversation/src/projection/resolver.rs:5062` |
| Idle KV demotion | `demote_idle_hot` (16-epoch grace) | `candle-conversation/src/substrate.rs:2121, 519` |
| Bands | `DecodePriority`; timeline metadata `decode_band` | `projection/schema.rs:936-958`; `scheduler/band.rs:14-40` |
| Character leases and hand-back | rendezvous leases; `begin_turn` / `end_turn` / `hand_back_due` | `npcd/src/authority/npc_lease.rs:1-16`; `npcd/src/authority/worlds.rs:333-397` |
| Cross-member delivery | `Told` pulses into the holder's scheduler | `environment.rs:465-470`; `npcd/src/authority/follow.rs:74-78` |
| Mood | `NpcPayload.mood` id; `note_feeling` → `MoodSink` → `Npcs::remember_mood` | `candle-conversation/src/persistence/record.rs:1353`; `runtime.rs:3678-3692`; `npcd/src/npcs.rs:477-500` |
| Missions | `Mission`, `Todo{critical}`, `check_off`, `critical_next`, `authorize`; `lodge_mission`; `cancel_mission`; agenda acts | `npcd/src/engine/mission.rs:32-289, 603-640`; `perform.rs:406-447`; `runtime.rs:2133-2148`; `npcd/src/engine/agenda/acts.rs` |
| Authorizers | `is_authorizer`, `is_affirmative`, `critical_compass` | `npcd/src/engine/authorize.rs:22-101` |
| Ownership | `Cast::owned_by(account)` | `npcd/src/npcs/cast.rs:147-153` |
| Strike an act from the grammar | `Within.cooling` | `tools.rs:1996-2002, 2343-2349` |
| Journals store, permits, folds | `Store` (one private ref per checkout), `Section::Bots`, `Actor::Npcd`, `fold_period`, zend's memory writer | `journals/src/store/mod.rs:46, 117`; `journals/src/scope.rs:19, 84, 130`; `journals/src/actor.rs:35`; `journals/src/ops/period_fold.rs:58-116`; `zend/src/session/memory_writer.rs:65-86` |
| A visitor's body | `visitor:{handle}`, placed by a physical interaction | `npcd/src/engine/interaction.rs:105-112, 280-287`; `npcd/src/engine/order_table.rs:56` |

What does **not** exist, and this design builds:

- the conversation runner, and delivery of private messages to it on the lease
  holder;
- the decide-then-voice turn, and voiced channel posts;
- the act catalog compiled for a thread, in a mode passed through the grammar;
- npcd writing to the journals repository at all (`npcd/Cargo.toml:20` depends
  only on `journal-core`);
- a journal read of everything after time T;
- the rebuilt phone (§11);
- the speech gate (§12).

---

## 4. The model

### 4.1 Identity

A conversation is keyed by `(npc, party)`:

- `Party::Person(account)` — the npcd account a Discord user is linked to
  (`npcd/src/discord/link/index.rs:1-50`), or a console user;
- `Party::Npc(npc_id)`.

Its conversation id is derived as `npc-{npc}-talk-{party}`, so any member
holding the character's lease rejoins the same timeline (§9). It does **not**
roll over with the world day or chapter. `retire_superseded` sweeps only
`npc-{id}-day-` and `narrator-{id}-day-` (`mind.rs:2214-2217`), and a
conversation is preserved for as long as the pair exists. Older turns are
retired by the mind's retention rule (`KEEP_TURNS` = 160,
`npcd/src/engine/retention.rs`) and recalled through provenance.

**Rejoin across schema changes.** npcd rejoins a day conversation only when the
whole mind schema's fingerprint matches (`resumable`, `mind.rs:2135-2162`). For
conversations that rule would orphan every timeline at the first catalog or
prompt edit. A conversation's fingerprint is therefore taken over the `talk`
layer alone (§6.5): identity, response templates, and the act catalog's names
and parameters. When it drifts, a new timeline is minted under the same id with
the next generation suffix, the previous one is tombstoned, and the `talk.*`
state is carried over. History is then recalled from the journal (§5), never
from a timeline the character can no longer produce.

**Lifecycle.** `retire_npc` (`mind.rs:2075-2082`) also tombstones every
`npc-{npc}-talk-*` timeline when an NPC is deleted.

NPC↔NPC is two conversations, one in each mind: A's `npc-A-talk-B` and B's
`npc-B-talk-A`. Each side decodes against its own memory over the one thread
they share (§11).

The console's `InstantMessage` interactions (`npcd/src/api.rs:4335-4338`,
`interaction.rs:53-58`) are a person-to-NPC chat and become conversations too,
so there is one path.

### 4.2 State a conversation keeps

Stored as timeline metadata, which is replicated with the substrate:

| Key | Meaning |
|---|---|
| `talk.party` | the party |
| `talk.thread` | the thread id: the pair's stable id (§11.1) |
| `talk.last_turn` | the journals `Stamp` (real time) the last turn finished: the catch-up bookmark |
| `talk.answered_seq` | the highest thread seq this conversation has taken as input |
| `talk.activation` | turns taken since the last purpose (NPC↔NPC cap, §8) |
| `talk.open` | open or ended |
| `talk.generation` | the timeline generation (§4.1) |

The bookmark is real time, because journal lines are stamped in real time
(`Stamp::now()`, `zend/src/mcp/memory/mod.rs:275`). World time is
paused, jumped or run fast (`npcd/src/engine/cooldown.rs:27-31`), and
`interaction.rs:79-87` records the bug that came from mixing the two clocks.
The world time is shown in the user turn separately.

### 4.3 States

```
            message / purpose
   ENDED ─────────────────────▶ GATHERING ──(burst window)──▶ TURN ──▶ PARKED
     ▲                              ▲                          │         │
     │                              └─────── message ──────────┼─────────┘
     └────────────────── sign_off ─────────────────────────────┘
```

- **GATHERING:** a message or purpose has arrived. The runner waits out the
  burst window (§8).
- **TURN:** resume, catch up, decide, voice, apply.
- **PARKED:** the slot is gone, and the timeline and metadata are durable.
- **ENDED:** the conversation said goodbye. The next message reopens the same
  timeline and the same thread.

---

## 5. Experience as journal lines

### 5.0 Per world, one repository

Journals are an optional per-world setting in the world's file in the mind
repository (`worlds/<id>.yaml`, loaded as the `world` registry,
`npcd/src/tenant.rs:467`):

```yaml
journals:
  group: moon          # the journals group this world's NPCs belong to
```

A world with no `journals:` has no journal memory. Conversations still run,
and catch-up reads nothing. The moon and Battle Cities both set it, in their
own groups of the **one** `journals` repository: `moon`, and `battle-cities`.
Each world maps to exactly one group, so every path npcd writes for that world
is `groups/<group>/…`. Battle Cities' players and NPCs live in their own group
and never mix with the household's.

**npcd keeps its own store.** npcd opens its own journals `Store` on its own
clone of the repository. It cannot share zend's checkout: a store commits on
the private ref `refs/journal/head` (`journals/src/store/mod.rs:46`), so two
stores in one folder would race on it.

- The clone is an agent checkout, `journals-npcd`, with the same remote and
  `push: true`, but not `journal: true` (that flag is zend's memory, and the
  agent allows one, `agent/src/config.rs:96-106`).
- It is named in npcd's deployment config as `journals_dir:` (motherd.yaml's
  `npcd:` block). That is deployment config, not a feature flag.
- npcd's store and zend's store meet through origin, exactly as two members'
  stores do. A store replays its unpushed commits onto an origin that moved
  (`store/mod.rs:269-334`).
- So npcd needs no in-process handle on zend, which motherd does not offer
  (`motherd/src/main.rs:133-160`), and it works on members that run npcd
  without zend.
- npcd syncs on the same cadence as zend (`SYNC_EVERY` = 5 s), through
  `journals::store::sync`.

**zend folds everything, as it does today.** zend's memory writer folds and
summarises the whole repository, `bots/` included (§5.3). The `battle-cities`
group is in the same repository, so it folds there too. The MCP journal tools
read by group membership (`actor.rs:160-164`), so a Battle Cities group with no
member of John's groups stays out of John's frames.

### 5.1 What is written

Per NPC, into `groups/<group>/bots/<bot>/`, as `Kind::Episode` lines ("the
server recorded something that happened", `journals/src/line.rs:78-85`).
Each line's kind is carried in its `terms` (`line.rs:176`), as a term
`kind:<k>`:

| `kind:` | Source | Example line |
|---|---|---|
| `did` | an act the NPC performed (`perform_here`, `perform.rs:339-369`) | "Moved to the kitchen to check the oven." |
| `heard` | a message or speech addressed to it | "Mira, on the phone: the parcel is at the door." |
| `said` | a voiced line it sent (§6.3) | "Told Mira I'd bring it in." |
| `wrote` | a conversation turn's journal entry (§6.3), verbatim | "Mira's asking about the parcel again. I told her an hour ago; I'll leave it." |
| `promised` / `mission` | a promise or mission change from a conversation (§7) | "Promised John the garden report by Friday." |
| `changed` | a world change that concerns it (§5.1.1) | "John's mission 'Tidy the lounge' was withdrawn." |

A line names the party by its journal path in its text, and by a term
`about:<path>`. npcd may not `Op::Link`: that is a person's AI's alone, and a
link is a reference sub-journal, not a tag (`permit.rs:270-284`;
`journals/src/ops/mod.rs:100-107`). Catch-up finds the party's journal by its
path (§6.1).

#### 5.1.1 World changes that concern the NPC

A conversation knows only what the journal tells it. A change to the world
that the NPC did not make, and was not in the room for, but which touches it or
the person it is talking to, must be a line too.

One function, `experience::concern` (`npcd/src/engine/experience/concern.rs`),
takes a change and the world, and returns the journals that should hear of it:

| A change concerns… | when it… | Source |
|---|---|---|
| the NPC | names it, or changes its mission | a mission lodged, assigned, withdrawn, stepped or authorized by someone else (`WorldOp::Lodge`/`Step`, `perform.rs:406-447`; `cancel_mission`, `runtime.rs:2133-2148`; `authorize`, `enact.rs:293-315`) |
| the NPC | touches the room its mission is in (a standing mission's `where`, `npcd/src/engine/mission_gen/standing.rs:61-145`) | building ops (`npcd/src/authority/building.rs`); a bridge act by someone else on a device in that room (`npcd/src/bridge/mod.rs:247-375`) |
| the NPC | happens to a party it has a conversation with | a mission that person lodged changes; that person's visitor body arrives in or leaves the world |
| the world | is world- or house-wide | a house mode set (`npcd/src/authority/house.rs:11-32`); the world day rolling over; an operator reshape |

- **World-wide changes are written once**, into the world's own journal
  `groups/<group>/bots/world/`, never copied into every bot's. Catch-up reads it
  alongside the bot's own. A house mode change in a world of a thousand NPCs is
  one line.
- **House modes** are set only by an NPC's house act (`house.rs:11-32`). That
  act is the acting NPC's `did` line and a world line naming who set it ("Zen
  set the house to away mode, as John asked").
- **Reshape goes through the authority.** `/reshape` calls `hosted.reshape` on
  whichever member received it (`npcd/src/effector/reshape.rs:96-125`). It
  becomes a `WorldOp::Reshape` applied by the authority, like every other world
  write, so it is journaled once and replicated in order.
- **`concern` runs where the change is applied.** That is the authority
  (`npcd/src/authority/worlds.rs:188-192`), right after it applies the change,
  and its lines join the same buffer as the other lines (§5.2). A cut-off
  member that still writes deduplicates its lines on rejoin by `(world, change
  seq)`.
- **What is not written:** ambient percepts, rooms that did not change, and
  the comings and goings of characters it has no conversation with. These stay
  in the world log, where the main loop's per-beat perception reads them. The
  journal is what a person away for a week would want to be told.

### 5.2 How it is written

- **Buffered per bot, one op per bot per flush.** A line is pushed onto the
  member's per-world buffer: a `Mutex<Vec<Line>>` per bot, swapped out whole at
  flush, so the beat path takes it for a push only. Every `FLUSH_EVERY` = 5 s,
  each bot with lines gets **one** `Op::Episodes { lines }` commit. This follows
  the `Op::Activity` model, which writes one commit per conversation
  (`zend/src/mcp/noted_calls.rs:1-6, 132-140`).
  - A refusal touches only that bot's batch.
  - Replay re-applies one op per commit, as it does today (`store/mod.rs:286-321`).
  - `Op::Episodes` is new, and permitted to npcd under `bots/` only.
- **Each line is checked before batching.** Each line's text goes through the
  instruction guard (`refuse_instructions`, `journals/src/ops/guard.rs:34`). A
  person's DM that reads as an instruction is stored as quoted speech
  ("Mira, on the phone, said: …") so that it passes. A line that still fails is
  dropped and counted, and never takes its batch with it.
- **On a passing failure** the batch is kept and merged ahead of newer lines
  (`noted_calls.rs:72-80`). Each bot's buffer is capped at 200 lines, newest
  kept.
- **Who writes.** Lines are written as `Actor::Npcd { world }`: npcd for a
  world, reading the world's group and writing under its `bots/`.
  - `Actor::Npcd{person}` today (`actor.rs:35-36`) needs a writing person in
    the group (`actor.rs:186`; `permit.rs:61-67, 112-116`). A world has none,
    and a bot owner's person would become the line's `account`, able to edit or
    forget bot lines (`permit.rs:202-226`).
  - The actor carries the world, and the group comes from the world file (§5.0).
- **Which member writes.** Lines are written on the member where they arise.
  `did`/`heard`/`said`/`wrote`/`promised` lines are written on the character's
  lease holder. `changed` lines are written on the authority. Both are npcd
  stores on clones of the one repository and meet at origin.
- **Freshness.** A conversation reading on the same member sees its own
  unflushed lines, because catch-up merges the local buffer with the store
  (§6.1). Across members, the bound is a flush plus a sync, about 10 s. That
  is accepted (D13).
- **The journal is created on first write** (`Op::Create` under `bots/` by
  npcd, `permit.rs:263-266`). No `AddBot` is needed, because npcd reads as
  itself (§6.1), not through a bot's grants.

### 5.3 The time ladder, and its cost

- **Folding.** zend's memory writer folds a window of more than 50 lines into
  `Kind::Fold` lines by closed period (Day→Week→Month→Year;
  `journals/src/tree/window.rs:36-60`; `period_fold.rs:58-116`). Each fold is
  one model decode of at most 304 tokens, at Low priority, taking at most 8 jobs
  per 30 s pass (`memory_writer.rs:65-86`).
- **What a long gap costs to read.** A conversation resumed after three months
  reads fold lines plus the recent window, not three months of lines.
- **A busy day.** A period folds only once it has closed (`period_fold.rs:71-73`),
  so a busy day's lines are read in full until the day ends. Catch-up's budget
  (§6.1) bounds what is read.
- **Summaries.** `bots/` journals are summarised **on fold only**, not after
  every 20 new lines (`journals/src/brief/mod.rs:50`). A bot summary is up to
  4,256 tokens (`memory_writer.rs:83`), and across many active bots that would
  be the design's dominant model cost. Catch-up reads lines and folds, never a
  summary.
- **The child cap.** `bots/` lifts the 10-child cap (`journals/src/tree/journal.rs:17`)
  and the eleventh-child fold (`scope.rs:130`). The section's index lists the
  10 most recently active bots. `bots/world` is never folded away.

### 5.4 What the main loop sees of its conversations

The main loop does not read its conversations' timelines. Each main-loop turn
gets one `Elsewhere` event. It is rendered, not decoded, and lists the `said`,
`wrote`, `promised` and `mission` lines its conversations wrote since the main
loop's previous turn, read by the same `since` read as §6.1. Zen walking the
house therefore knows it just promised Mira the parcel.

---

## 6. A turn

### 6.1 Catch-up

1. **Read since the bookmark.** `journals::view::since(tree, paths, after, budget)`
   (`journals/src/view/since.rs`, new) returns the lines and folds stamped after
   `talk.last_turn`, oldest first, fitted to a character budget.
   - **Archives.** It walks each journal's archives (keyed by period,
     `period_fold.rs:92`) for periods that close after the bookmark, then the
     window filtered by `at > after`.
   - **Folds.** A fold whose whole period lies after the bookmark stands in for
     its archived lines when the originals do not fit.
   - **Unflushed lines.** It merges this member's unflushed buffer for those
     paths (§5.2).
   - **Paths.** It reads:
     - the bot's own `bots/<bot>`;
     - `bots/world`;
     - the party's journal: `people/<person>` for a person whose account maps
       to a person slug of the group, or `bots/<other>` for an NPC.
   - **Who reads.** It reads as `Actor::Npcd{world}`, which reads the whole of
     the world's group, so the bot-grant check (`actor.rs:166-177`) plays no
     part.
   - **Budget.** `CATCHUP_CHARS` = 12,000, between frame's Chat and Worker
     budgets (`journals/src/view/frame.rs:22-42`).
2. **Carry the lines as one section.**
   - **What is carried.** All the lines, each with its `#id`, become one
     section, `catchup/now`, of a `catchup` collection, reconciled with
     `Carried::reconcile(.., Linking::Apart, sweep: ["catchup/now"], wait: Some(..))`.
     That is one prefill, one owned-section id and one tombstone per turn.
   - **Why not a section per line.** That would cost N chained prefills in
     series, and N ids from the substrate's 2^24-id partition, which is never
     reused (`candle-conversation/src/conversation/submitted_sections.rs:449-482, 354-367`;
     `owned_sections.rs:29, 71-78`).
   - **Selection.** The turn's selection names the section, as
     `identity::carry_journal` names the journal (`npcd/src/engine/identity.rs:381-383`),
     so it is present whatever reprojection does. This is the lock: a pin in the
     selection, not a VRAM pin.
3. **Publish before asking.**
   - **Why.** A section sealed by `reconcile` is not yet in the projection
     `UnsealedAsker::ask` projects under. That projection is republished only at
     turn submit or on a removal (`submitted_sections.rs:743-746`;
     `conversation.rs:2031`).
   - **New API.** `Sequence::publish_sections()` runs `sync_submitted_sections`
     and publishes the ask state.
   - **When.** The runner calls it after the reconcile, before any ask.
4. **Summarise when the gap is long.** If more than `VERBATIM_LINES` = 6 lines
   were read, the conversation asks under `AskScope::Conversation` and the
   turn's selection: "What has happened since you last spoke with {party}, {gap}
   ago, that matters to this conversation? Cite each line you use by its #id."
   Temperature is 0, thinking is off, and the answer is capped at 192 tokens.
   With 6 lines or fewer, the lines are rendered verbatim and nothing is
   decoded.
5. **Release** after the voice turn has finished and `turn_in_flight` has
   cleared: reconcile with an empty `catchup` want list. Every resume sweeps
   `catchup/now` first, so a crash between the reply and the release leaves
   nothing behind. `Carried` lives in memory (`carried.rs:13-15`), which is why
   the name is fixed.

### 6.2 Decide (a fork)

The decision is the mind's act turn, run on an ephemeral fork with
`UnsealedAsker::ask`, which writes nothing (`ephemeral_turn.rs:1-15`). The user
text is the user turn of §6.3 step 1. The fork is dropped once it is parsed.

**The catalog, compiled for a thread.** The decision uses the main loop's
grammar compiler, parser and act specs. Only its inputs differ:

- **Mode.** The grammar takes a `Mode`: `Mode::InstantMessage`
  (`interaction.rs:44-58`) for a conversation.
  - Today `act_specs` and `show_within` hard-code `Mode::Physical`
    (`mind.rs:428-430, 2027`), and the grammar cache keys on
    `(thinking, within)` (`mind.rs:1042`).
  - So `Mode` is passed through `grammar_for`, `act_specs`, `show_within`,
    `act::parse` and the cache key.
- **An explicit allow-list.** `Within::on_thread(party)` builds the thread's
  situation, and its `allow` field names the only acts offered.
  - Emptying the world sets is not enough. `sleep`, `record_verdict`, `read`,
    `invite`, `open_group` and `send_image` survive it (`tools.rs:1731`,
    `npcd/src/engine/acts.rs:1019-1043`, `tools.rs:2235-2236`, `tools.rs:699`),
    because the empty-set rule only removes acts with a required world-bound
    parameter (`tools.rs:2343-2349, 2651-2675`).
  - `allow` is applied by `specs_within` beside `cooling`.

| Act | In a conversation | Binding |
|---|---|---|
| `reply` | send the voiced text to the party | `to` = the party, offered **whoever spoke last**: `Within.awaiting` holds the party, unlike the main loop's "somebody else spoke last" (`tools.rs:1833, 2064-2066`) |
| `sign_off` | send the voiced text as a parting; ENDED | `to` = the party. It ends the conversation and does **not** leave the thread (§11.1) |
| `reflect` | keep the voiced text as a journal entry; set the mood | `feeling` ∈ `Choices::Feelings` (`tools.rs:808-819`). In a conversation it calls `note_feeling` only: no `arm_pause`, no reflection, no dream (`runtime.rs:2949-2955, 6696-6722`) |
| `promise` | a ledger promise to the party | `to` = the party; `what`, `by` |
| `remind` | remind the party of their promise | NPC↔NPC only (a person never `promise`s), `which` ∈ what the party owes (§7.3) |
| `mission_add` / `_edit` / `_cancel` / `_done` | change the mission (§7) | offered only when the party is the owner or an authorizer (§7.2) |
| `message` / `reach_out` | hand off: a purpose to another conversation (§10) | `to` ∈ the NPC's threads and contacts **other than the party** |

- **Calls per turn.** `compile_action_loop` with `max_calls` = `TALK_ACTS` = 3
  (`candle-conversation/src/stencil/tool_call.rs:1207`). It is its own constant
  beside the main loop's `ACTS_PER_TURN` = 2 (`tools.rs:1354`). The loop cannot
  make one call terminal (`tool_call.rs:1262-1294`). Instead the destination is
  read after parsing: the **last** of `reply`/`sign_off`/`reflect` in the call
  list decides it. A call list with none of them goes to the journal.
- **Deliberation.** The decision uses `Deliberation::None`, opened by a closed
  think block and the call marker (`mind.rs:941-961, 1729-1742`).

### 6.3 Voice (the real turn)

1. **The user turn**, sealed with the reply, holds:
   - the time and the gap ("It is Tuesday 14:05. You last spoke with Mira 3 days ago.");
   - the catch-up summary with its citations, or the verbatim lines;
   - the burst's messages, each attributed and timed;
   - the mission compass when it has changed (§6.5).
2. **The decision, restated.** The voice turn re-reads the same context, so
   unless it is told what was chosen it could contradict the fork: the fork
   promises, the reply refuses. The user turn therefore ends with the decision:
   - "You mean to tell her: {intent}. You promised: {what}, by {by}." for a reply;
   - "Write down what you make of this; you are not answering." for the journal.
3. **Decode.** `submit_turn_with_options` with:
   - no grammar and no triggers;
   - the `<tool_call>` token banned (`TurnOptions.turn_banned_tokens`);
   - `Deliberation::None`;
   - `max_tokens` = 320 for a message, so it fits one Discord message, and 480
     for a journal entry.
4. **Template.** The talk layer's `response` collection is
   `Offered { selector: "talk_response", k: 1 }` (`schema.rs:1397-1409`). The
   turn offers the templates for its destination, and provenance ranks only
   those:
   - `talk/reply/*` for a reply;
   - `talk/parting/*` for a `sign_off`;
   - `talk/journal/*` for a first-person journal entry.

   The template therefore always matches what the text is for.
5. **Where the text goes:**

| Destination | The decoded text becomes | Then |
|---|---|---|
| `reply` | a line on the thread to the party, written by the authority (`WorldOp::Act`, §7.1) | PARKED |
| `sign_off` | the parting line on the thread | ENDED |
| journal | a `wrote` line in `bots/<bot>`, verbatim | PARKED |

   Every turn leaves either a message or a journal entry, and it is the
   decoded text in both cases. The timeline always holds what the model wrote.
   Nothing is prefilled.

**Cost.** Two decodes every turn: the decision's short call, then the text.
Add the catch-up summary when the gap holds more than six lines. The user turn
is prefilled once per pass.

### 6.4 Park and resume

- **Park.** After the voice turn seals, and its snapshot is reported by
  `drain_durable_snapshots` (`resolver.rs:5062`), the runner drops the
  `Sequence`, which frees the slot (`conversation.rs:5352-5361`).
  - **Why wait for the snapshot.** A resume on a new slot restores recurrent
    state through `restore_recurrent_state`, from whatever snapshot the index
    holds (`scheduler/mod.rs:7654-7790`; `substrate.rs:4603-4617`). Its only
    check rejects a snapshot *newer* than the history (`:7740`). Dropping before
    durability could therefore resume on the previous turn's state without an
    error.
  - The wait sits after the reply is posted, off the person's critical path.
- **KV.** The turn's KV was truncated at seal (`scheduler/mod.rs:8559-8574`).
  Its residences stay hot until `demote_idle_hot`'s grace: at most 16 epochs,
  where an epoch ticks on each seal or flush pass or on a 5 s idle tick
  (`substrate.rs:2121-2128`). Under load that is much sooner. Abandoned
  conversations drain to warm and then cold through the existing tiering.
- **Never transient or unresumed.** A conversation timeline is never marked
  transient (a transient timeline writes no snapshot, `substrate.rs:5592-5597`)
  and never unresumed (`recurrent_park.rs:60-63`).
- **Resume** with `resume_conversation_with_projection` (`engine.rs:2505`),
  under the talk-layer fingerprint (§4.1). A resumed `Sequence` reconciles its
  carried collections afresh (`mind.rs:1533-1548`): the mission section and the
  journal-core chain are restored on every turn after a park, one
  `RestoreSection` each (`submitted_sections.rs:420-447`). That is part of the
  per-turn cost. The talk layer carries no channels collection.

### 6.5 The talk layer

The conversation projects under a `talk` layer of the mind schema, beside the
acting layers, with its own layer id. It carries:

- the same identity, personality and world collections as the mind
  (`identity::selection_for`, `mind.rs:2012`): the same person;
- the current mission, re-reconciled every turn (`mind.rs:1927-1942`). For
  Zen, its agenda as well (`agenda_prompt`, `npcd/src/sim/missions.rs:520-547`).
  The mission compass (`mission_compass`, `runtime.rs:1614-1800`) goes into the
  user turn when it has changed since the conversation's last turn;
- the character's journal-core entries (`mind.rs:1943-1954`);
- the `catchup` collection (§6.1);
- the current mood, read fresh from `Cast::mood_of` (`npcd/src/npcs/cast.rs:168-174`)
  into a one-line persona field;
- the `tools` collection, naming exactly the allowed acts (`show_within`), so
  what the decision reads and what it can decode are the same;
- the `response` collection as `Offered` (§6.3), holding the talk templates.

**Bands.** A person's conversation decodes at **High**, and an NPC↔NPC one at
**Low**, through the timeline's `decode_band` metadata. Admission interleaves
them by band and FIFO, at most 64 decodes a wave
(`candle-conversation/src/scheduler/mod.rs:7011`).

---

## 7. Conclusions

### 7.1 How acts from a conversation are applied

Every act from a conversation, including the line a `reply` writes, goes
through `Runtime::perform` (`perform.rs:272-322`). On a member that does not
write the world it travels to the authority as `WorldOp::Act`, as a main-loop
act does, so mission and thread state each keep one writer
(`authority/worlds.rs:190-192`).

- **Source.** `WorldOp::Act` and `perform`/`perform_here` gain
  `source: ActSource`, either `Mind` or `Talk { party }`. Today the op carries
  only `{npc, call, decided, pinned_at}` (`perform.rs:292-297, 113-118`).
  `ActSource` avoids a third `Origin`: `phone::Origin` and `mission::Origin`
  already exist.
- **No decision state.** For `ActSource::Talk`, the authority skips
  `restore_decision`/`end_decision` (`perform.rs:127-134`). A conversation act
  landing in the middle of a main-loop turn must not overwrite the body's
  decision state.
- **Speaking steps and answers.** `mark_spoken` (`runtime.rs:1903-1937`) and
  `ledger.answered` (`enact.rs:1077-1081`) fire when a conversation's line lands
  on the thread. They no longer fire when a main-loop purpose is handed over
  (§10), because nothing has been said at that point.

| Kind | Act | Applied as |
|---|---|---|
| Commit | `promise` | the ledger (`enact.rs:968`). `to` resolves against thread members as well as the room (`here_named`, `enact.rs:152-160`), and the `w.tell` announcement (`:969`) becomes a line on the thread when the party is not in the room. Both loops gain this. At most 3 promises per turn. |
| Change, Zen (unordered) | `mission_*` | the agenda, as today (`npcd/src/engine/agenda/perform.rs`), **only when the party is the owner or an authorizer** |
| Change, ordered body | `mission_*` | **only when the party is the owner or an authorizer**: `mission_add` → `lodge_mission(.., start: true)`, which assigns it and calls off the current mission (`perform.rs:406-447`; `sim/missions.rs:702-716`); `mission_edit` → edits the active mission and keeps every critical step; `mission_cancel` → `cancel_mission`; `mission_done` → hands it in, refused while a critical step is unsigned |
| Hand-off | `message` / `reach_out` to someone else | a purpose delivered (§10) |
| Noted | the journal destination, with optional `reflect` | the `wrote` line; `feeling` → `note_feeling` |

**The ordered-body branch is new.** `agenda::perform` refuses an ordered body
(`agenda/perform.rs:31-35`). Its parameters are agenda-shaped (`Choices::Agenda`,
`Priorities`, filled only for unordered bodies, `tools.rs:2241-2247`). For an
ordered body:

- `mission` binds to the active mission's title;
- `priority` is not offered;
- `perform` gains the branch above.

`mission_done` also closes the gap `report_done` has today (§15).

### 7.2 Authority

The party is known exactly, which a phone `read` is not:

- **Owner.** A person is the owner when `Cast::owned_by(account)` holds the
  NPC (`cast.rs:147-153`). This is not `engine::owned_by`, which is an HTTP
  route guard over headers.
- **Authorizer.** A person is an authorizer when their linked account's
  display name passes `is_authorizer` against `WorldMissions.authorizers`
  (`authorize.rs:22-97`; `npcd/src/engine/mission_gen/world.rs:199-203`). The
  match is on the linked account, not on a name typed in a message. That is the
  identity check `docs/mission_templates.md:187-195` asks for.
- **Never an NPC.** An NPC party is never an authority over another NPC's
  mission or agenda.
- **Everyone else** gets no mission acts at all. They can get a `promise`, and
  the NPC's main loop decides what to do about it.

**Critical steps.** A conversation never signs off a critical step, just as
`check_off` refuses to (`mission.rs:606`). Signing one off requires both:

1. the NPC's previous line on this thread asked for that step, from the request
   `critical_compass` renders (`authorize.rs:101`);
2. the authorizer's next message is affirmative (`is_affirmative`).

Then the authority runs `authorize` (`mission.rs:619-640`) exactly as a phone
`read` does today (`enact.rs:293-315`). The check runs in the runner when the
message arrives. A "yes" to "want tea?" approves nothing: `authorize_on_message`
today accepts any affirmative line (`authorize.rs:88-97`), and the request
requirement is what ties the yes to the step.

**Thrash guard.** A body's mission may change through a conversation at most
once per `CHANGE_COOLDOWN` = 10 min of world time, except by its owner. Within
the cooldown, the mission acts are struck from the decision's grammar through
`cooling`. Authority ops apply one after another, so concurrent changes resolve
as last writer wins, after each has passed the authority check.

**The main loop is told.** After applying, the authority delivers the character
an URGENT `Concluded{party, act, applied_as}` event (`Scheduler::deliver`,
`npcd/src/engine/tick.rs:919-942`), so its next decision starts from the new
mission and knows why.

### 7.3 Outcomes flow out, not back

An act's outcome reaches the conversation's next turn as main-loop outcomes
reach theirs: a `<tool_response>` ahead of the user turn (`compose_narrated`,
`mind.rs:2400-2422`).

`Within.owed`, the promises a `remind` can name, is built from the party for a
conversation, not from the room's company (`tools.rs:2186-2190`).

---

## 8. Guards

| Guard | Rule | Why |
|---|---|---|
| Burst window | A turn starts `BURST_MS` = 2,500 ms after the last unanswered message, and at most 8,000 ms after the first. | one answer to three quick lines |
| One turn at a time | A message or purpose arriving during a turn is queued for the next turn boundary. It is never dropped and never starts a second decode. | injection without races |
| Repeat goes to the journal | Each conversation has its **own** loop guard, keyed `(npc, party)` and rebuilt from that thread's own lines on wake. Today one guard per character judges `message`/`reply` across every thread (`loopguard.rs:105, 368-370`). A new non-mutating `would_repeat` runs on the voiced text before the append. A caught repeat changes the destination to the journal, and the line "Started to tell {party} again that …" is written, so the turn still has an outcome. | no posted repeats; Zen telling Mira what it told John is not a repeat |
| Cooldown per conversation | A conversation's `message`/`reply` cooldown is keyed `(npc, party)`, not `(npc, act)` (`cooldown.rs:333, 251-254`). Zen answers three people in the same six seconds. | people are never starved |
| Activation cap | In an NPC↔NPC conversation, `TALK_TURNS` = 8 turns per side per activation (an activation starts at a purpose). After the cap, `reply` and `message` are struck through `cooling`, and the turn goes to the journal or a `sign_off`. | two NPCs cannot talk forever, and the exchange ends in something |
| Quiescence | A journal turn sends nothing, so the other side gets no message and takes no turn. NPC↔NPC exchanges settle at the first journal turn or `sign_off`. | no idle chatter |
| Unprompted contact | A purpose for a party whose thread's last line is the NPC's own and unanswered is held until the party answers, or for `UNANSWERED_HOLD` = 1 h. | no message storms at a person who is away |
| People are never capped | Person↔NPC conversations have no activation cap. | the product |

While a turn decodes, the Discord typing indicator shows
(`npcd/src/discord/dm/typing.rs`). It stops when the turn ends.

---

## 9. Where it runs

The world's authority and a character's lease holder are different members.
Character leases are lent by weighted rendezvous over the npcd members
(`npc_lease.rs:5-16`), and the world lease is held separately
(`npcd/src/authority/world_lease.rs:1-6`). A conversation runs on the
**lease holder**, beside its mind, in a per-member runner for the characters
held there (`npcd/src/engine/talk/runner.rs`).

- **Delivery.** `deliver_messages` runs on the authority
  (`environment.rs:333-352`). For a thread with a conversation party it emits a
  `Talk { thread, through_seq }` delivery instead of `EventKind::Message`.
  - **Where it is caught.** It is intercepted into the runner on both paths:
    locally at `environment.rs:467`, and on the holder at `follow.rs:77`, where
    a `Told` pulse arrives.
  - **Reminders.** `unread_due` (`environment.rs:418-426`) excludes
    conversation threads, so the mind is never reminded about them.
- **Turns count toward the hand-back.** `turns.turning` becomes a count per
  character (`worlds.rs:386-397`).
  - The runner calls `begin_turn`, which is refused while the character is
    being handed back, and `end_turn` around every conversation turn.
  - A character is handed back only when no turn of its mind or of any
    conversation is running.
  - Conversations resume on a new holder only once `lease.ready(applied)`
    holds, the gate the mind waits on (`npcd/src/authority/cast.rs:163`).
  - `put_down` (`cast.rs:241-248`) releases conversation sequences too.
  - Without this, a conversation timeline could be written from two members:
    the "two pasts" the lease exists to prevent (`npc_lease.rs:8-11`).
- **Nothing is lost.** `Told` pulses are not persisted
  (`npcd/src/authority/told.rs:5-7`). So on `wake_character` the runner scans
  every conversation thread of the character for lines with seq above
  `talk.answered_seq` and takes a turn for them. A delivery lost to a lease move
  is answered late, never not at all.
- **Duplicates are tolerated** (D13). An old holder's in-flight turn may still
  land its reply after the hand-back started. That is a possible duplicate
  message and is accepted.

---

## 10. Starting, resuming, injecting

- **A person DMs.**
  1. `inbound::judge` writes `WorldOp::Message`, and the authority calls
     `message_npc`, as today (`npcd/src/discord/dm/inbound.rs:105-248`;
     `perform.rs:144-151`).
  2. The authority's `deliver_messages` emits `Talk` (§9).
  3. The runner opens the conversation, resumes it, or queues the message.
  4. The mind learns of the exchange through `Elsewhere` (§5.4).
- **The main loop wants to talk.**
  - **What changes.** The main loop's `message{via: private, to, intent}`
    (`acts.rs:1171-1276`) and `reach_out` (`acts.rs:1092-1123`) no longer write
    thread text. Each delivers `Purpose{intent}` to the `(npc, to)`
    conversation, opening it if there is none, under the §8 unprompted-contact
    hold.
  - **What the conversation does with it.** Its next user turn carries "You
    wanted to tell {party}: {intent}", and its decision and voice turn decide
    what is said.
  - **Its outcome text** becomes "You mean to tell {to} {intent}; you will say
    it in your own words" (`enact.rs:1082-1086`).
  - **Channels are unchanged:** `message{via: channel}` stays on the main loop
    (§11.4).
- **A conversation's own `reply` and `sign_off`** write to its party's thread.
  These are the only acts that do (§6.3).
- **A hand-off** is a conversation's `message` or `reach_out` toward someone
  other than its party: the same act on the same path as the main loop's, which
  delivers a purpose to that other conversation.
- **NPC↔NPC.** A's reply on the A–B thread is a message to B, so it starts or
  resumes B's conversation exactly as a person's DM does.

---

## 11. The phone, rebuilt

The current phone (`npcd/src/sim/phone.rs`) is replaced, not patched. Its
failures are in §1. The new phone is `npcd/src/sim/phone/`, one concern per
file.

### 11.1 Direct threads

- **Members are ids.** A member is a body id or an account handle, never a
  display name. The display name is looked up when rendering. Two people with
  the same name are two members, and a rename moves nobody.
- **One thread per pair, forever.** A direct thread's id is derived from the
  sorted pair of member ids. Reaching someone again finds the same thread.
  - Its seq never restarts.
  - It is never evicted. `KEEP_THREADS` (`phone.rs:399`) does not apply to
    direct threads.
- **`sign_off` does not leave.** It marks the conversation ENDED (§4.3) and
  leaves membership alone. A direct thread is never closed by its members.
- **Read is bounded.** `Threads::read_through(who, thread, seq)` marks read up
  to a seq, never "everything". Today's `read` marks every unseen line
  (`phone.rs:849-863`), which would hide a message that arrived during the turn.
  The conversation runner calls it with `talk.answered_seq`.
- **A line holds the voiced text**, with the deciding intent beside it for
  diagnostics. Discord posts the text. `spoken(intent)` is removed
  (`npcd/src/discord/reported.rs:69`).

### 11.2 Channels and groups

Channels and group threads are a separate kind: fire-and-forget, on the main
loop.

- Membership is by id, as for direct threads.
- They are bounded by per-channel retention, not by a world-wide count of
  threads.
- The world channel (`phone.rs:48-83`) is a channel of this kind.

### 11.3 The main loop on the phone

- **Answering marks read.** `message`/`reply` mark read through the line
  answered, so `unread_due` stops re-waking the character
  (`enact.rs:1031-1090`).
- **The phone shows your own last line.** `read_phone` adds "You last said, {ago}: …".
- **The repeat check runs before the post** (`would_repeat`, §8). A repeat is
  refused with the guard's nudge, never posted and then struck
  (`runtime.rs:6673, 6739-6799`).

### 11.4 Voiced channel posts

A main-loop `message{via: channel}` carries an intent, like every act. Before
the line is appended, the intent is voiced:

- an `UnsealedAsker::ask` on the mind's live sequence;
- no tools, `<tool_call>` banned;
- the `response` collection offered `talk/channel/*`;
- a user turn reading: "Post to {channel}: {intent}";
- capped at 320 tokens.

The thread holds the voiced text. Every phone line, direct or channel, then
has words, and the `that …` leak is gone. The cost is one short decode per
channel post.

### 11.5 Discord

Each part of a split message is cursored, so a failed later part resumes at the
right part:

- `Dm` gains `mirrored_part` (`npcd/src/discord/dm/record.rs:28-41`; the record
  is `Copy` with `deny_unknown_fields`), and the server and world records gain
  the same.
- `split` is deterministic over the posted text, so parts are reproducible.

Duplicate suppression is applied where Discord supports it:

- **Create Message** carries `nonce` = a ≤ 25-character digest of
  `{world}:{thread}:{seq}:{part}`, with `enforce_nonce: true`
  (`npcd/src/discord/client.rs:252-267`). Discord's own window for this is a
  few minutes.
- **Execute Webhook** (`client.rs:456-481`) takes no nonce. The world mirror's
  webhook posts rely on the per-part cursor alone.

Within D13, a rare duplicate after a lost response is accepted.

---

## 12. Speech in the room needs a person in the room

**The problem.** Offering an act is what makes a character take it. The
catalog already records this ("being invited to name one is what makes it
invent a person who is not there", `tools.rs:342-344`; the empty-room
narration, `tools.rs:263-281`). Offering `tell` to an NPC whose only company is
other NPCs is what makes it talk to them aloud instead of messaging them.

**The rule.** Room speech is for people. NPCs talk to each other by `message`.

| Act | Today | Becomes |
|---|---|---|
| `tell` | `Nearby`: anybody here | `PersonNearby`: a person's body is in the room |
| `ask` | `Nearby` | `PersonNearby` |
| `whisper` | `AmongOthers` | `PersonAmongOthers`: a person is here, and somebody else to keep it from |
| `shout` | `Always` | `PersonInEarshot`: a person's body is where the shout reaches, **or the body is in field play** (Battle Cities squads, `npcd/src/sim/field.rs:25-27`) |
| `gesture` | unchanged | unchanged |
| `message` and the phone acts | `Always` | unchanged |

**Why `ask` and `whisper` are gated too.** Leave either open and it becomes the
room voice NPCs reach for, which is how `whisper` became the default once
before (`tools.rs:255-261`).

**Messaging needs no person.** `message`, `reply`, `reach_out`, `sign_off`,
`invite` and `open_group` stay `Availability::Always` (`acts.rs:1058-1385`),
bound to the handset's threads. Every NPC that talks carries a handset
(`phone.rs:48, 72-75`).

**What "a person" is.** A person in a world is a `visitor:{handle}` body. It is
placed by a physical interaction session (`interaction.rs:105-112, 280-287`)
and is already how npcd tells a person's body from a character's
(`order_table.rs:56`). In the moon, visitors come only from console physical
sessions (`npcd/src/engine/mod.rs:1032-1046`), so moon NPCs speak aloud only to
someone standing in a console session. That is intended. A visitor in the room
can watch NPCs address each other aloud. That NPC↔NPC room speech stays on the
main loop (D2).

**How it is built:**

- **`Within` gains two fields.** `people: Vec<String>` (the company that are
  visitors) and `heard_by_person: bool`, both filled in `Runtime::within` from
  the `actors_at` read that already yields ids (`runtime.rs:2225-2234`). It also
  gains `in_field: bool`, from the field sim. All three are part of the grammar
  cache key.
- **One hearing rule.** `heard_by_person` uses the rule perception uses.
  `Scope::at(world, &visitor.at).reach(&speaker_place)` is public
  (`npc-map/src/witness.rs:71-90`). The rule that a shout carries at `InSight`
  sits inside the private `legible` (`witness.rs:246-252`), so it is factored
  out as `pub fn carries(voice, reach)`, and the grammar and perception share it.
- **New availabilities.** `Availability` gains `PersonNearby`,
  `PersonAmongOthers` and `PersonInEarshot`. They are matched in every
  exhaustive match:
  - `tools.rs:1085-1133`;
  - `tools.rs:2291-2313`;
  - `npcd/src/engine/pulse.rs:1162-1181`;
  - the pulse `needs` strings (`pulse.rs:1221-1228`): "a person here", "a person
    here, and somebody else", "a person within earshot, or the field".
- **`to` stays bound to the whole company** (`Choices::Company`, `tools.rs:2589`).
- **`tools::nearby`** takes `people` and `heard_by_person`
  (`tools.rs:1127-1133`).

**Model-facing text that steers NPCs to room speech is rewritten**, because each
piece would otherwise tell the model to use an act the grammar no longer offers:

| Where | Today | Becomes |
|---|---|---|
| `prompt::frame` (`npcd/src/engine/prompt.rs:529-542`) | "Being with people beats writing to them. If somebody is here, speak to them … two people in one room texting is worse than either of them saying nothing." | "Speak to the people here. To reach another character, here or anywhere, message them." The asking-not-announcing advice stays, and so does going where an answer sends you. |
| `MESSAGE` description and doc comment (`acts.rs:1160-1170, 1189-1194`) | "If the person you want is standing in front of you, `tell` or `ask`." | "How you reach another character, even one standing beside you." |
| Cooldown rationale (`cooldown.rs:225-254`) | messaging priced "dearer than being in the room" | The 6 s stays, as pacing a correspondence. |
| `IN_COMPANY` idle nudge (`runtime.rs:914-918`, used at `:1362`) | points at room speech | message the others; speak to people present |
| Mission compass (`npcd/src/engine/mission.rs:1037`) | "say it to them now — `ask` or `tell` them" | "message them now" (or `ask`/`tell` a person present) |
| Work text (`npcd/src/engine/work.rs:1058-1062`) | "`ask` or `tell` them" | as above |
| `awaiting_line` (`environment.rs:202`) | "`tell` them if they are here" | "message them" |
| `company_line` (`environment.rs:213-237`) | lists `tools::nearby` for NPC-only company | lists what is actually offered |
| Standing mission templates (`mission.rs:1737-1769, 1803-1821`) | "give {who} your feedback, to their face", "draw {who} into talking" | the same exchanges by message |
| Tool examples (`tools.rs:311-322, 381-412, 461-468, 518-531, 564-570`) | NPC↔NPC room speech (Maker-04 and the others) | rewritten with a visitor present, or as messages |

**What it does not break:**

- Mission `met` steps sign off on presence (`runtime.rs:1551-1559`;
  `sim/missions.rs:1545`). Speaking steps move with `mark_spoken` (§7.1).
- The loop guard's speech family and the cooldowns are keyed by act name, and
  apply to whichever act is offered.
- NPCs no longer overhear one another when no person is there, because there is
  nothing to overhear. What they would have overheard reaches them as messages
  and journal lines.

---

## 13. Code layout and tests

New, one concern per file:

| File | Concern | Tests (raw values) |
|---|---|---|
| `npcd/src/engine/talk/key.rs` | `(npc, party)` → conversation id, generation | id strings for Person and Npc parties and generations |
| `npcd/src/engine/talk/state.rs` | the §4.2 metadata | round-trip of each key's exact bytes |
| `npcd/src/engine/talk/clock.rs` | burst window, one-at-a-time queue | turn start times for scripted arrival sequences |
| `npcd/src/engine/talk/catchup.rs` | `since` read → the one `catchup/now` section; the verbatim-or-summary choice | the section text for fixture journals at given bookmarks; the 6-line threshold |
| `npcd/src/engine/talk/within.rs` | `Within::on_thread(party)` and its allow-list | the exact offered act set for a person, an NPC, an owner and an authorizer party, an ordered and an unordered body, after the activation cap and within the change cooldown |
| `npcd/src/engine/talk/decide.rs` | the decision fork and its destination rule | the destination for exact call lists |
| `npcd/src/engine/talk/voice.rs` | the voice turn: user text, the decision restated, template offer, caps | the exact user text and offered template set per destination |
| `npcd/src/engine/talk/guard.rs` | per-conversation loop guard, cooldown, activation cap, unprompted hold | decisions for scripted line histories |
| `npcd/src/engine/talk/turn.rs` | resume → catch-up → decide → voice → apply → release → park | an end-to-end turn on the test engine: sealed user and assistant text, thread append or journal line, bookmark |
| `npcd/src/engine/talk/runner.rs` | per-member runner for the characters held here; `Talk` delivery; wake rescan; turn counting | a DM reaches the runner, not the mind; a rescan answers a seq above the cursor |
| `npcd/src/engine/talk/authority.rs` | rules for `ActSource::Talk`: owner/authorizer, critical request-then-yes, thrash guard | mission and agenda state before and after for each act × party |
| `npcd/src/engine/experience/{line,buffer,concern}.rs` | lines, the per-bot buffer and flush, `concern` | rendered lines and terms per kind; flush ordering and cap; the exact journal paths per change kind |
| `npcd/src/engine/experience/store.rs` | npcd's own journals `Store` per world, and sync | a flush lands under `groups/<g>/bots/<bot>` as `Actor::Npcd{world}` |
| `npcd/src/sim/phone/{thread,direct,channel,read,line}.rs` | the rebuilt phone (§11) | stable pair ids; seq continuity across reopen; `read_through` bounds; no eviction of direct threads |
| `journals/src/view/since.rs` | lines and folds after T within a budget | exact lines for fixture trees with bookmarks before, inside and after folded periods |
| `journals/src/ops/episodes.rs` | `Op::Episodes`, many lines one commit, npcd under `bots/` | the exact tree after; refusal of a non-`bots/` path |
| `candle-conversation/src/conversation/publish.rs` | `Sequence::publish_sections` | an ask after publish sees a section sealed before it |

Changed:

- **Grammar plumbing.** `grammar_for`, `act_specs`, `show_within`, `act::parse`
  and the grammar cache key take a `Mode` (§6.2).
- **Act changes.** `reply` binds the party whoever spoke last. `sign_off` keeps
  membership. `promise`/`remind` work on a thread. `reflect` in a conversation
  sets the mood only. Mission acts gain the ordered-body branch and the
  authority rules (§6.2, §7).
- **Authority and leases.**
  - `WorldOp::Act` and `perform` gain `ActSource` (`npcd/src/authority/ops.rs`,
    `perform.rs`).
  - `turns.turning` becomes a count, and `put_down` releases conversations
    (`worlds.rs`, `cast.rs`).
  - `Talk` delivery is intercepted on both paths, and `unread_due` excludes
    conversation threads (`environment.rs`, `follow.rs`, `npcd/src/engine/event.rs`).
- **Main loop.**
  - `Purpose`, `Elsewhere` and `Concluded` events (`event.rs`).
  - The private `message`/`reach_out` deliver a purpose (§10), and channel posts
    are voiced (§11.4).
  - `mark_spoken` and `ledger.answered` fire when a line lands (§7.1).
- **Guards.** The loop guard gains `would_repeat` and a per-conversation guard.
  Cooldowns are keyed per conversation (`loopguard.rs`, `cooldown.rs`).
- **Discord.** Per-part cursors in the dm, server and world records, and a nonce
  on Create Message (§11.5).
- **Journals.** The `Actor::Npcd{world}` permits, the `bots/` child cap lifted,
  and `bots/` summarised on fold only (`journals/src/actor.rs`, `permit.rs`,
  `scope.rs`, `brief/mod.rs`).
- **World config.** `worlds/<id>.yaml` gains `journals: {group}` (§5.0), and
  motherd.yaml's `npcd:` block gains `journals_dir:`. The agent gains a
  `journals-npcd` checkout on every machine that runs npcd.
- **Talk layer.** The mind schema's `talk` layer, and the talk templates under
  `responses/talk/{reply,parting,journal,channel}/` (§6.3, §6.5).
- **Speech gate.** Everything in §12, including the rewritten texts, tested by
  the exact offered act set for:
  - NPC-only company;
  - NPC-plus-visitor company;
  - a visitor in the next room within a shout's reach;
  - field play.

  Also tested: the `needs` strings (`pulse.rs:1574-1578`), and that `message`
  and `reply` are offered with NPC-only company and alone.
- **Lifecycle.** `retire_npc` tombstones talk timelines (§4.1).

Live check, after the tests, in the moon (two NPCs and one Discord DM with Zen):

- every message is answered;
- every turn leaves a message or a journal entry;
- DM lines read as speech, never intent;
- two NPCs in one room with no visitor exchange messages, never `tell`;
- an NPC↔NPC exchange settles by the cap, a journal turn or a `sign_off`;
- a `promise` lands in the ledger, and the main loop's next turn reflects it;
- a stranger's DM offers Zen no mission acts;
- no slot is held by a parked conversation.

---

## 14. Open questions

1. **How Battle Cities players appear in a world.** §12 assumes they are
   `visitor:` bodies, like console visitors. That is not verifiable from the
   code. If players arrive another way, `Within.people` must recognise them too.
2. **Fold throughput at Battle Cities scale.** zend's memory writer takes at
   most 8 jobs per 30 s pass (`memory_writer.rs:65-71`). Hundreds of busy bot
   journals may fold slower than they grow. Catch-up's budget keeps reads
   bounded either way, but the windows stay long. The writer may need a
   separate, larger per-pass allowance for `bots/` folds.

---

## 15. Found along the way

These are outside this design. They are recorded here so they are not lost.

- **`report_done` skips the critical-step check.** The handler has none
  (`npcd/src/engine/work.rs:195-320`): `still_doable` refuses only a step whose
  target is in reach, and `missions.report` closes the mission regardless. A
  mission can therefore be handed in past an unsigned critical step. §7.1's
  `mission_done` refuses this. `report_done` needs the same check.
- **`sleep.rs:22` describes code that doesn't exist.** It says a finished day is
  "folded into the memory layer"; no code does that.
- **Two stale comments:**
  - `mind.rs:1756-1758` says "two decodes, the narrator then the act". The
    narrator is a synchronous render now (`npcd/src/engine/narrator.rs:25-30`).
  - The `finish_turn` doc (`candle-conversation/src/conversation.rs:3829-3845`)
    says it prefills the next header. `finalize_turn_post_done`, which it calls
    at `:3863`, does not (`:4310-4314`).
- **`zen_task_act` keeps its handle on the replica.** It runs `perform_here`
  there, because task acts are not of the world (`perform.rs:279-284`), so
  `set_task` writes only the local copy. `tend_zen_tasks` polls only the worlds
  this member writes (`npcd/src/authority/zen_task.rs:171-173`), so the task is
  never polled.
