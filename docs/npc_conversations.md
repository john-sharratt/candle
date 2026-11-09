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
- **What is posted is the act's intent, not words.**
  - `enact::message` puts the raw `intent` on the thread
    (`npcd/src/engine/enact.rs:1044, 1054`).
  - Discord posts `spoken(&m.intent)` (`npcd/src/engine/reported.rs:69`;
    callers `npcd/src/discord/dm/outbound.rs:184`, `server/outbound.rs:50`,
    `world/outbound.rs:77`). It only strips a leading "that " and capitalises.
  - Nothing decodes the words. `narrate_act` (`npcd/src/engine/mind.rs:963-1000`)
    is a synchronous render for the feed, and `reply` is not among the acts it
    renders (`tools::narrates`, `npcd/src/engine/tools.rs:1178-1183`).

  This is the `that …` leak.
- **Answering does not mark the thread read.** `message`/`reply`
  (`enact.rs:1031-1090`) never call `Threads::read`, so the URGENT unread
  reminder (`npcd/src/sim/unread.rs:21`, every 5 min) wakes the character on a
  message it has already answered.
- **The phone never shows the character its own replies.** `read_phone`
  (`enact.rs:341-386`) returns only unseen lines from others.
- **The loop guard runs after the repeat is posted.** The act is performed at
  `npcd/src/engine/runtime.rs:6673`, judged at `:6739-6799`, and the guard's
  state is in memory only (`npcd/src/engine/loopguard.rs:368-370`).
- **Threads themselves are fragile.**
  - A world keeps 64 non-permanent threads and drops the quietest
    (`npcd/src/sim/phone.rs:399, 524-538`).
  - A thread that is reopened gets a new id, and its seq restarts at 1
    (`phone.rs:541-566`).
  - `sign_off` leaves the thread, and a direct thread with one member closes
    (`phone.rs:834-845`).
  - Members are display names (`npcd/src/engine/mod.rs:1199-1212`).
- **The party's identity is lost.** `inbound::judge` resolves the Discord
  author's account, then writes `WorldOp::Message{from: speaking_as(..)}`, a
  display name (`npcd/src/discord/dm/inbound.rs:193, 213-218`).
  - Authorizers are matched by display name (`npcd/src/engine/authorize.rs:22-27`).
  - Any affirmative line from one signs off a critical step
    (`authorize.rs:88-97`; `enact.rs:293-315`).
- **NPCs talk to each other aloud.** `tell` is offered whenever anybody is in
  the room, and `shout` always (`tools.rs:287, 345`). NPC↔NPC talk is room
  speech (`npc-map/src/world.rs:1910-1930`), which nothing preserves and
  nothing concludes.

A conversation removes these by construction: each message is a turn of a
preserved timeline, every turn decodes real words, and the party is known by
account. §11 rebuilds the phone, and §12 makes room speech for people only.

---

## 2. Decisions

| # | Decision |
|---|---|
| D1 | A conversation is **the NPC's mind loop on a message clock**: same persona, same shared system prompt, same act catalog, same performer. The beat is replaced by the arrival of a message, and perception is narrowed to what changed since the last turn. |
| D2 | A conversation is **one-to-one**: an NPC and one party, which is a person or another NPC. Group channels stay on the main loop's fire-and-forget messaging (§11.2). So does NPC↔NPC room speech in front of a person (§12). |
| D3 | **Parked between turns.** A conversation holds no scheduler slot and no live `Sequence` between messages. It is a persisted substrate timeline that resumes on the next message. Thousands can exist, and only the ones decoding cost VRAM. |
| D4 | **Mood, mission, beliefs, persona and memory are shared** with the main loop. It is the same person in two places. |
| D5 | **Experience is journaled**, as an optional per-world setting in the one `journals` repository (§5.0). Each NPC's `bots/<bot>` journal gets lines for:<br>- what it hears and says;<br>- what it writes down in a conversation;<br>- its promises and mission changes;<br>- its notable world acts;<br>- the world changes that concern it.<br>The moon and Battle Cities both use it, each in its own group. |
| D6 | **Catch-up is a turn-scoped carried section plus a cited summary.** The journal lines since the conversation's last turn are carried into the turn, summarised with line citations into the user turn, kept for the reply, then removed. |
| D7 | **Every turn decides, then decodes.** A forked call chooses the acts from the existing catalog. Then a real turn always decodes free text, with no tools, under a response template that matches the choice. The choice decides **where the text goes**: to the party (`reply`, `sign_off`) or verbatim into the journal (anything else). Every turn leaves a message or a journal entry. A person's message may be answered by a journal entry; the NPC then marks it read, and Discord's typing indicator stops. |
| D8 | **One tool base.** The decision uses the main loop's catalog, compiled for a thread instead of a room, through an explicit allow-list: `reply`, `sign_off`, `reflect`, `promise`, `remind`, the `mission_*` acts, and `message`/`reach_out`. |
| D9 | **A main-loop private `message` (or `reach_out`) starts, resumes or injects into a conversation.** It hands the conversation a *purpose*, and the conversation decodes the words. |
| D10 | **Conversations conclude.** A promise, a mission change or a hand-off is how a conversation changes what the NPC does. They are applied by the world's authority under authority, critical-step and thrash rules. Mission changes are open to the owner and authorizers only. |
| D11 | **Speech in the room needs a person in the room** (§12). NPCs reach each other by `message`. Battle Cities field play keeps `shout`. |
| D12 | **Every phone line is voiced on the lease holder** before it is sent to the authority. No intent text reaches a thread (§7.1, §11.4). |
| D13 | **Duplicates are tolerated.** An occasional repeated message is accepted, and the system is eventually consistent. A message that is never answered, or one timeline written from two members, is not accepted (§9). |
| D14 | **The party is known by account.** Authorizers, owners and thread members are account handles or body ids, never display names (§7.2, §11.1). |

---

## 3. What exists that this is built on

| Need | Existing piece | Anchor |
|---|---|---|
| Character decode | `Minds`, `Live{sequence}`, `send_turn_with_options_async` | `npcd/src/engine/mind.rs:163-227, 1786-1789` |
| Act grammar, parser, performer | `grammar_for`, `act::parse` (over specs), `performable`, `Runtime::perform` → `perform_here` | `mind.rs:1034`; `npcd/src/engine/act.rs:372`; `tools.rs:2343-2349`; `npcd/src/authority/perform.rs:272-369` |
| Modes | `Mode::{Physical, InstantMessage}` | `tools.rs:1015-1023` |
| Side decode that writes nothing | `UnsealedAsker::ask`, taking full `TurnOptions` (grammar, prefill) | `candle-conversation/src/conversation/ephemeral_turn.rs:69-112`; npcd's use at `npcd/src/engine/mind/ask.rs:178-185` |
| Response templates | the shared prompt's `response` collection (`top_k: 1`, filled from `<mind>/responses/`) | the mind's `projection.yaml`; `npcd/src/projection.rs:361` |
| Runtime-offered candidates | `SelectionRule::Offered { selector, k }`, set per turn through `SelectionState` | `candle-conversation/src/projection/schema.rs:1397-1409`; precedent `tools.rs:1266-1274` |
| Banning tokens in a turn | `TurnOptions.turn_banned_tokens` | `candle-conversation/src/turn.rs:60-62` |
| Carried sections per turn | `Carried::reconcile` | `candle-conversation/src/conversation/carried.rs:159` |
| Republishing the projection | `rebuild_projection` → `publish_ask_state` (private) | `candle-conversation/src/conversation/submitted_sections.rs:733-746` |
| Resume on a new slot | `resume_conversation_with_projection`; `restore_recurrent_state` | `candle-conversation/src/engine.rs:2505`; `candle-conversation/src/scheduler/mod.rs:7654-7790` |
| Idle KV demotion | `demote_idle_hot` | `candle-conversation/src/substrate.rs:2121, 519` |
| Bands | `DecodePriority`; timeline `decode_band` | `projection/schema.rs:936-958`; `scheduler/band.rs:14-40` |
| Leases and hand-back | rendezvous leases; `Standing::begin_turn`/`end_turn`/`hand_back_due`; `lease.ready`; `wake_character`; `put_down` | `npcd/src/authority/npc_lease.rs:1-16`; `npcd/src/authority/worlds.rs:333-397`; `npcd/src/authority/cast.rs:163, 206, 241-248` |
| Cross-member delivery | `Told` pulses; `Runtime::tell` | `environment.rs:465-470`; `npcd/src/authority/follow.rs:74-78`; `runtime.rs:2660-2672` |
| Mood | `NpcPayload.mood`; `note_feeling` → `MoodSink` → `Npcs::remember_mood` | `candle-conversation/src/persistence/record.rs:1353`; `runtime.rs:3678-3692`; `npcd/src/npcs.rs:477-500` |
| Missions | `Mission`, `Todo{critical}`, `check_off`, `critical_next`, `authorize`, `critical_compass`; `lodge_mission`; `cancel_mission`; agenda acts | `npcd/src/engine/mission.rs:32-289, 603-640`; `authorize.rs:101`; `perform.rs:406-447`; `runtime.rs:2133-2148`; `npcd/src/engine/agenda/` |
| Ownership | `NpcPayload.owner_id` (a public handle `u_…`) | `npcd/src/api.rs:1092`; `npcd/src/npcs/cast.rs:147-153` |
| Striking an act | `Within.cooling`, read by `performable` | `tools.rs:1996-2002, 2343-2349` |
| Journals | `Store` (one private ref per checkout); `Section::Bots`; `Actor::Npcd`; `fold_period`; zend's `Memory` wrapper and memory writer | `journals/src/store/mod.rs:46, 117-173, 269-334`; `journals/src/scope.rs:19, 84, 130`; `journals/src/actor.rs:35`; `journals/src/ops/period_fold.rs:58-116`; `zend/src/mcp/memory/mod.rs:121-290`; `zend/src/session/memory_writer.rs:65-86` |
| A visitor's body | `visitor:{handle}` | `npcd/src/engine/interaction.rs:105-112, 280-287`; `npcd/src/engine/order_table.rs:56` |

What does **not** exist, and this design builds:

- the conversation runner on the lease holder, and its delivery;
- the decide-then-voice turn, and voicing on the lease holder;
- the catalog compiled for a thread under a `Mode`;
- npcd writing journals (`npcd/Cargo.toml:20` depends only on `journal-core`);
- typed experience on journal lines;
- a journal read since a point;
- a non-consuming durability query;
- the rebuilt phone (§11);
- accounts carried through messages and authority;
- the speech gate (§12).

---

## 4. The model

### 4.1 Identity and generations

A conversation is keyed by `(npc, party)`:

- `Party::Person(account)` — the npcd account a Discord user is linked to
  (`npcd/src/discord/link/index.rs:1-50`), a console user, or a Battle Cities
  player (§12), by public handle;
- `Party::Npc(npc_id)`.

Its conversation id is `npc-{npc}-talk-{party}-g{n}`, where `n` is the
generation. Any member holding the character's lease rejoins the newest
generation (§9). It does not roll over with the world day or chapter.
`retire_superseded` sweeps only `npc-{id}-day-` and `narrator-{id}-day-`
(`mind.rs:2205-2221`).

- **A conversation is resumed only under the frame it was decoded under.** npcd
  rejoins a timeline only when the rendered frame's fingerprint matches
  (`resumable`, `mind.rs:2135-2170`), because a history the character can no
  longer produce is worse than none (`mind.rs:2099-2105`). A conversation's
  frame is the shared prompt rendered under the `talking` stance (§6.5).
- **When the frame changes**, the next message opens generation `n+1`. The
  `talk.*` state is carried over, and generations below `n+1` are retired by
  prefix, as `retire_superseded` does. History before the change is recalled
  from the journal (§5). This is why every conversation's content also lands in
  the journal.
- **Lifecycle.** `retire_npc` (`mind.rs:2075-2082`) also tombstones every
  `npc-{npc}-talk-*` timeline when an NPC is deleted.
- **NPC↔NPC** is two conversations, one in each mind, over the one direct thread
  the pair shares (§11.1).
- **The console's chat.** The console's `InstantMessage` interactions
  (`npcd/src/api.rs:4335-4338`) become conversations too, so there is one path.

### 4.2 State a conversation keeps

Timeline metadata, replicated with the substrate:

| Key | Meaning |
|---|---|
| `talk.party` | the party |
| `talk.thread` | the pair's stable thread id (§11.1) |
| `talk.last_turn` | the journals `Stamp` (real time) the last turn finished |
| `talk.seen` | the line ids read within the overlap window before `last_turn` (§6.1) |
| `talk.answered_seq` | the highest thread seq taken as input |
| `talk.activation` | turns in the current activation (§8) |
| `talk.pending_critical` | the critical step the NPC's last line asked an authorizer for (§7.2) |
| `talk.compass` | the fingerprint of the mission compass last shown |
| `talk.open` | open or ended |
| `talk.generation` | the current generation |

The bookmark is real time, because journal lines are stamped in real time
(`Stamp::now()`, `zend/src/mcp/memory/mod.rs:275`). World time can be paused,
jumped or run fast (`npcd/src/engine/cooldown.rs:27-31`; the bug that mixing
the two caused is recorded at `interaction.rs:79-87`). World time is shown in
the user turn separately.

### 4.3 States

```
            message / purpose
   ENDED ─────────────────────▶ GATHERING ──(burst window)──▶ TURN ──▶ PARKED
     ▲                              ▲                          │         │
     │                              └─────── message ──────────┼─────────┘
     └────────────────── sign_off ─────────────────────────────┘
```

- **GATHERING:** a message or purpose has arrived, and the burst window runs (§8).
- **TURN:** resume, catch up, decide, voice, apply.
- **PARKED:** the slot is gone; the timeline and metadata are durable.
- **ENDED:** the conversation said goodbye. The next message reopens the same
  timeline and the same thread.

---

## 5. Experience as journal lines

### 5.0 Per world, one repository

Journals are an optional per-world setting in the world's file in the mind
repository (`worlds/<id>.yaml`, the `world` registry loaded at
`npcd/src/tenant.rs:463`):

```yaml
journals:
  group: moon          # the journals group this world's NPCs belong to
```

A world with no `journals:` has no journal memory. Its conversations still
run, and catch-up reads nothing. The moon and Battle Cities both set it, in
their own groups of the **one** `journals` repository: `moon` and
`battle-cities`.

- **One group per world.** Every path npcd writes for a world is under
  `groups/<group>/`.
- **Keeping Battle Cities out of personal frames.** `CreateGroup` makes its
  creator the first owner (`journals/src/ops/mod.rs:313-318`), and a member's
  frame shows every group they belong to. The `battle-cities` group is
  therefore created by the `zen` service person (`zend/src/mcp/memory/who.rs:102-124`),
  so it is in no person's frame unless they join it.

**The live store moves into the journals crate.** zend's `Memory` wrapper (the
store mutex, the published snapshot, and the sync thread that pushes and fetches
every `SYNC_EVERY` = 5 s; `zend/src/mcp/memory/mod.rs:121-290`) becomes
`journals::live::Live`. zend and npcd each open one, so the sync logic exists
once.

**npcd keeps its own clone.**
- **Why.** A store commits on the private ref `refs/journal/head`
  (`journals/src/store/mod.rs:46`), so two stores cannot share zend's checkout.
- **The clone.** npcd's is an agent checkout, `journals-npcd`, with the same
  remote and `push: true`, but not `journal: true`. That flag is zend's memory,
  and the agent allows one (`agent/src/config.rs:96-106`). It is named in
  motherd.yaml's `npcd:` block as `journals_dir:`, which is deployment config.
- **Two stores, one origin.** The two stores meet through origin as two
  members' stores do: a lease push, and op-level replay onto an origin that
  moved (`store/mod.rs:269-334`; `journals/src/store/sync.rs:78-167`).
- **What this avoids.** npcd needs no in-process handle on zend, which motherd
  does not offer (`motherd/src/main.rs:133-160`), and it works on members that
  run npcd without zend.

**zend folds the repository**, `bots/` included, with its own budget for
`bots/` (§5.3).

### 5.1 What is written

Per NPC, into `groups/<group>/bots/<bot>/`, as `Kind::Episode` lines
(`journals/src/line.rs:78-85`).

**Experience is a typed field.** `Line.terms` cannot carry it: `Terms` is a
fixed struct, and only a note may state terms (`journals/src/terms.rs:21-33,
69-76`; `line.rs:208-210`). So `Line` gains
`experience: Option<Experience { kind: ExperienceKind, about: Option<JournalPath> }>`,
which is valid on episodes only.
- It is rendered into the line's file and read back, with tests asserting the
  raw bytes.
- It is included in `period_fold::lines_over` (`period_fold.rs:39-55`), so a
  fold conflicts when it changes.
- `about` names the party's journal. npcd may not `Op::Link`, which is a
  person's AI's alone and makes a sub-journal, not a tag (`permit.rs:270-284`).

| `kind` | Written when | Example |
|---|---|---|
| `heard` | a message or speech addressed to the NPC lands | "Mira, on the phone: the parcel is at the door." |
| `said` | a voiced line it sent lands (§6.3, §11.4) | "Told Mira I'd bring it in." |
| `wrote` | a conversation turn's text goes to the journal (§6.3), verbatim | "Mira's asking about the parcel again. I told her an hour ago; I'll leave it." |
| `promised` | a promise lands (§7.1) | "Promised John the garden report by Friday." |
| `mission` | its mission changes through a conversation or an order | "John withdrew 'Tidy the lounge'." |
| `did` | a **notable** world act lands. That means an act whose outcome changed world state (a device, a house mode, a part, an item given, a mission step), not walking, waiting or idle acts. | "Turned the lounge TV off." |
| `changed` | a world change that concerns it (§5.1.1) | "The house was set to away mode." |

**Bounded volume.**
- **The store's cost.** Each line is a file, and a store holds every file of
  the repository in memory and reloads the tree on replay
  (`store/mod.rs:96, 281-282`). With two clones per machine, volume is the
  scaling limit.
- **Daily cap.** Lines are capped at `DAY_LINES` = 200 per bot per day. When
  over, the oldest `did` and `heard` lines of the day are dropped first.
- **Folds absorb the rest** (§5.3).

#### 5.1.1 World changes that concern the NPC

A conversation knows only what the journal tells it. So a world change that the
NPC did not make, and was not in the room for, but which touches it or its
party, is a line too.

One function, `experience::concern` (`npcd/src/engine/experience/concern.rs`),
takes a change and the world, and returns the journals that should hear of it:

| A change concerns… | when it… | Source |
|---|---|---|
| the NPC | names it, or changes its mission | a mission lodged, assigned, withdrawn, stepped or authorized (`WorldOp::Lodge`/`Step`, `perform.rs:406-447`; `cancel_mission`; `authorize`) |
| the NPC | touches the room its mission is in (a standing mission's `where`, `npcd/src/engine/mission_gen/standing.rs:61-145`) | building ops (`npcd/src/authority/building.rs`); a bridge act by someone else on a device in that room (`npcd/src/bridge/mod.rs:247-375`) |
| the NPC | happens to a party it has a conversation with | a mission that person lodged changes; that person's visitor body arrives or leaves |
| the world | is world- or house-wide | a house mode set (`npcd/src/authority/house.rs:11-32`); the day rolling over; an operator reshape |

- **World-wide changes are written once**, into `groups/<group>/bots/world/`.
  Catch-up reads that journal alongside the bot's own.
- **House modes** are set only by an NPC's house act, which is that NPC's `did`
  line plus one world line ("Zen set the house to away mode, as John asked").
- **Reshape goes through the authority.** `/reshape` calls `hosted.reshape` on
  whichever member received it (`npcd/src/effector/reshape.rs:96-125`). It
  becomes `WorldOp::Reshape`, applied by the authority, so it is journaled once.
- **Where `concern` runs.** It runs where the change is applied (the
  authority, `worlds.rs:188-192`), right after the change. A cut-off member that
  still writes deduplicates its lines on rejoin by `(world, change seq)`.
- **What is never written.** Ambient percepts, rooms that did not change, and
  the comings and goings of characters it has no conversation with. These stay
  in the world log, where the main loop's per-beat perception reads them.

### 5.2 How it is written

**Buffered per bot, one op per bot per flush.**
- **The buffer.** A line is pushed onto the member's per-world buffer, a
  `Mutex<Vec<Line>>` per bot that is swapped out whole at flush. The beat path
  takes the lock only for a push.
- **The flush.** Every `FLUSH_EVERY` = 5 s, each bot with lines gets **one**
  `Op::Episodes { lines }` commit, on the `Op::Activity` model of one commit
  per subject (`zend/src/mcp/noted_calls.rs:1-6, 132-140`). A refusal touches
  only that bot's batch.
- **Each line keeps its own `at`** (stamped when it arose), not the flush
  time. Today `applied_to` takes one `now` per op (`ops/mod.rs:236-246`).
- **Line ids survive replay.** `Applied` gains a list of line ids, with one
  `Journal-Line` trailer per line (`store/message.rs:19-31`). Replay renames
  each line that collides (`store/mod.rs:302-311`), so a catch-up's `#id`
  citations stay correct.
- **Wire codec.** `Op::Episodes` and the new actor get one (`ops/wire.rs:54, 152`).
- **Permits.** `Op::Episodes` is permitted to npcd under `bots/` only.

**The instruction guard.** Each line's text goes through `refuse_instructions`
(`journals/src/ops/guard.rs:34-45`), which matches its phrases anywhere in the
text. A line that fails is dropped and counted, never taking its batch with it.
A person's DM that trips the guard therefore leaves no `heard` line, though the
conversation still answers it.

**Passing failures.** On a passing failure the batch is kept and merged ahead of
newer lines (`noted_calls.rs:72-80`). Each bot's buffer is capped at 200 lines,
newest kept.

**The actor is `Actor::Npcd { group }`.** The journals crate knows groups, not
worlds.
- **Why not today's actor.** `Actor::Npcd{person}` (`actor.rs:35-36`), built
  today only in a test (`ops/create.rs:438`), needs a writing person in the
  group (`actor.rs:186`). A bot owner's person would become the line's
  `account`, able to edit or forget bot lines (`permit.rs:202-226`).
- **Reads** go through its own arm in `need_read`: `bots/` of its group, plus
  `people/<p>` for a person it has a conversation with. It reads nothing else:
  not `conversations/`, not `threads/`, not `topics/`. An NPC must never read a
  person's AI conversations and repeat them to another party.
- **Writes** are `Op::Episodes` and `Op::Create` under `bots/` of its group.
- **Commits** carry no `Journal-Account`, and `who()` renders "npcd"
  (`journals/src/view/who.rs:36`).

**Creating the journal is idempotent.** A bot's first line may arrive on two
members at once (the lease holder's `heard`, the authority's `changed`).
`Op::Create` uses `OnTaken::Refuse`, and "exists already" counts as success, so
`bots/<bot>.2` can never be minted (`ops/create.rs:102-103`). The create sits in
the same flush as the first batch.

**Which member writes.**
- `heard`, `said`, `wrote`, `promised` and `did` lines: the character's lease
  holder.
- `changed` and `mission` lines: the authority.

Both write to npcd stores on clones of the one repository. Within a member,
catch-up merges the unflushed buffer (§6.1). Across members, the bound is a
flush plus a sync (about 10 s), which D13 accepts.

### 5.3 The time ladder, and its cost

- **Folding.** zend's memory writer folds a window of more than 50 lines into
  `Kind::Fold` lines by closed period, Day→Week→Month→Year
  (`journals/src/tree/window.rs:36-60`; `period_fold.rs:58-116`). A fold is one
  model decode of at most 304 tokens at Low priority.
- **`bots/` folds have their own allowance.** They take `BOTS_PER_PASS` = 16 jobs
  per 30 s pass, beside the existing `PER_PASS` = 8 (`memory_writer.rs:65-71`),
  so bot upkeep never starves people's folds and summaries.
- **A busy day.** A period folds only once it has closed
  (`period_fold.rs:71-73`), so a busy day is read in full until it ends.
  Catch-up's budget bounds what is read (§6.1).
- **`bots/` summaries happen on fold only.** `bots/` is exempt from every
  summary trigger: quiet for 30 s, quiet for 10 min as a parent, and 20 new
  lines (`QUIET_SECS`, `PARENT_QUIET_SECS`, `NEW_LINES`;
  `journals/src/brief/mod.rs:34-50`). A bot summary is refreshed when a period
  folds. Catch-up reads lines and folds, never a summary.
- **The child cap.** `bots/` lifts the 10-child cap and the eleventh-child fold
  (`journals/src/tree/journal.rs:17`; `scope.rs:130`; the check at
  `journals/src/tree/check.rs:300`). The section's index lists the 10 most
  recently active bots, and `bots/world` is never folded away.

### 5.4 What the main loop sees of its conversations

The main loop does not read its conversations' timelines. Each main-loop turn
gets one `Elsewhere` event, rendered rather than decoded. It lists the `said`,
`wrote`, `promised` and `mission` lines its conversations wrote since the main
loop's previous turn, read by the same `since` read as §6.1. Zen walking the
house knows it just promised Mira the parcel.

---

## 6. A turn

### 6.1 Catch-up

1. **Read since the bookmark.** `journals::view::since(tree, paths, after, seen, budget)`
   (`journals/src/view/since.rs`, new) returns lines and folds, oldest first,
   fitted to a character budget.
   - **The overlap window.** It reads from `after − OVERLAP`, where
     `OVERLAP` = 60 s, and drops line ids in `seen`. A line stamped before the
     bookmark but arriving after it, from another member's flush and sync, is
     therefore still read once. The ids read within the window become the next
     `talk.seen`.
   - **Archives and folds.** It walks each journal's archives (keyed by period,
     `period_fold.rs:92`) for periods that close after the window starts, then
     the window filtered by `at`. A fold whose whole period lies inside stands
     in for its archived lines when the originals do not fit.
   - **Unflushed lines.** It merges this member's unflushed buffer for those
     paths.
   - **Priority within the budget.** It fills in this order:
     1. `said`, `wrote`, `promised`, `mission` and `changed`;
     2. `heard`;
     3. `did` and folds.

     So a busy NPC's acts never push out what matters to the conversation.
   - **Paths:**
     - `bots/<bot>`;
     - `bots/world`;
     - the party's journal: `people/<person>` where the party's account maps to
       a person of the group (§5.2), or `bots/<other>` for an NPC.
   - **Actor and budget.** It reads as `Actor::Npcd{group}`. `CATCHUP_CHARS` =
     12,000, which sits between the frame's Chat and Worker budgets
     (`journals/src/view/frame.rs:22-42`).
2. **Carry the lines as one section.**
   - **What is carried.** All lines, each with its `#id`, become one section,
     `catchup/now`, of a `catchup` collection. It is reconciled with
     `Carried::reconcile(.., Linking::Apart, sweep: ["catchup/now"], wait: Some(..))`:
     one prefill, one owned-section id and one tombstone per turn.
   - **Why not a section per line.** That would cost N chained prefills in
     series and N ids from a 2^24 partition that is never reused
     (`submitted_sections.rs:449-482, 354-367, 586-591`; `owned_sections.rs:29, 71-78`).
   - **Declared in the shared prompt.** `catchup` is declared there with a named
     selector, unset for the main loop, because `submit_section` refuses a
     collection the schema does not declare (`submitted_sections.rs:569-578`).
   - **Selection.** The turn's selection names it, as `identity::carry_journal`
     names the journal (`npcd/src/engine/identity.rs:381-383`).
   - **Sweeping on resume.** Every resume builds a fresh `Carried::new(false)`,
     so its first reconcile sweeps `catchup/now` (`carried.rs:186-188`). A crash
     between the reply and the release leaves nothing behind.
3. **Publish before asking.** A section sealed by `reconcile` is not yet in the
   projection `UnsealedAsker::ask` uses. That projection is republished only at
   submit or on a removal (`submitted_sections.rs:743-746`;
   `candle-conversation/src/conversation.rs:2031, 2039`).
   - **The new method.** `Sequence::publish_sections()` runs
     `sync_submitted_sections` and publishes the ask state
     (`submitted_sections.rs:733-746`; `conversation.rs:2106-2111`).
   - **When it runs.** The runner calls it after the reconcile, before any ask.
   - **The mind gets it too.** The mind's own asks have the same gap after
     `prepare` reconciles (`mind.rs:1931-1954`), and the mind calls it as well.
4. **Summarise when the gap is long.** If more than `VERBATIM_LINES` = 6 lines
   were read, an ask runs under `AskScope::Conversation` at the conversation's
   band: "What has happened since you last spoke with {party}, {gap} ago, that
   matters to this conversation? Cite each line you use by its #id."
   Temperature is 0, no thinking, and the answer is capped at 192 tokens. With 6
   lines or fewer, the lines are rendered verbatim and nothing is decoded.
5. **Release** after the voice turn has finished and `turn_in_flight` has
   cleared: reconcile with an empty `catchup` want list.

### 6.2 Decide (a fork)

The decision is the mind's act turn, run on an ephemeral fork with
`UnsealedAsker::ask`, which writes nothing (`ephemeral_turn.rs:1-15`).
- **Scope and band.** It uses `AskScope::Whole`, at the conversation's band
  (High for a person, Low for an NPC).
- **Its user text** is the user turn of §6.3 step 1. That is the text the voice
  turn will seal, without step 2's restatement.
- **Afterwards.** The fork is parsed and dropped.

**The catalog, compiled for a thread.**

- **`Mode` is a grammar input.** Today `act_specs` and `show_within` hard-code
  `Mode::Physical` (`mind.rs:428-430, 2027`), as do `mind.rs:861`,
  `runtime.rs:7925` and `npcd/src/engine/invoke_body.rs:368`, and the grammar
  cache keys on `(thinking, within)` (`mind.rs:1042`).
  - `Mode` is therefore passed through `grammar_for`, `act_specs`, `show_within`
    and the cache key.
  - `act::parse` takes specs, so it follows without change.
- **An explicit allow-list.** `Within::on_thread(party)` sets `allow`, the only
  acts offered.
  - **Why emptying the world sets is not enough.** `sleep`, `record_verdict`,
    `read`, `invite`, `open_group` and `send_image` survive it (`tools.rs:1731`;
    `npcd/src/engine/acts.rs:1019-1043`; `tools.rs:2235-2236, 699`). The
    empty-set rule removes only acts with a required world-bound parameter
    (`tools.rs:2343-2349, 2651-2675`).
  - **Where it is checked.** `allow` is checked in `performable` beside
    `cooling`, which both `specs_within` and `invoke_body` share, so both
    grammars honour it.
- **The path budget.** `estimated_paths` raises the per-act total to a
  hard-coded `ACTS_PER_TURN` (`tools.rs:2700-2703`) under `MAX_TURN_PATHS` =
  2,000,000 (`tools.rs:2737`). It takes the call count as a parameter, and a
  budget test covers the full thread catalog at `TALK_ACTS`.

| Act | In a conversation | Binding |
|---|---|---|
| `reply` | the voiced text goes to the party | `to` = the party, **whoever spoke last** (the talk `Within.awaiting` holds the party; the main loop's needs somebody else to have spoken last, `tools.rs:1833, 2064-2066`) |
| `sign_off` | the voiced text is the parting; ENDED | `to` = the party; it ends the conversation and keeps membership (§11.1) |
| `reflect` | the voiced text goes to the journal; the mood is set | `feeling` ∈ `Choices::Feelings` (`tools.rs:808-819`). In a conversation the performer calls `note_feeling` only: no `arm_pause`, no reflection, no dream (`runtime.rs:2949-2955, 6696-6722`). |
| `promise` | a ledger promise to the party | `to` = the party; `what`, `by` |
| `remind` | remind the party of their promise | NPC↔NPC only, because a person never `promise`s. `which` ∈ what the party owes. |
| `mission_*` | change the mission (§7.1) | offered only when the party is the owner or an authorizer, outside the thrash cooldown |
| `message` / `reach_out` | hand off: a purpose to another conversation (§10) | `to` ∈ the NPC's threads and contacts **other than the party** |

- **Calls per turn.** `compile_action_loop` with `max_calls` = `TALK_ACTS` = 3
  (`candle-conversation/src/stencil/tool_call.rs:1207`). This is its own
  constant beside the main loop's `ACTS_PER_TURN` = 2 (`tools.rs:1354`).
- **The destination.** The loop cannot make one call terminal
  (`tool_call.rs:1262-1294`), so the destination is read after parsing: the
  **last** of `reply`, `sign_off` or `reflect` in the list decides. A list with
  none of them goes to the journal.
- **Deliberation.** The decision uses `Deliberation::None`, opened by a closed
  think block and the call marker (`mind.rs:941-961, 1729-1742`).

### 6.3 Voice (the real turn)

1. **The user turn**, sealed with the voiced text, holds:
   - the time and the gap ("It is Tuesday 14:05. You last spoke with Mira 3 days ago.");
   - the summary with citations, or the verbatim lines;
   - the burst's messages, attributed and timed;
   - the mission compass when its fingerprint differs from `talk.compass`.
2. **The decision, restated.** The voice turn re-reads the same context, so
   unless it is told what was chosen it could contradict the fork. The user
   turn therefore ends with the decision:
   - for a reply: "You mean to tell her: {intent}. You promised: {what}, by {by}.";
   - for the journal: "Write down what you make of this; you are not answering."
3. **Decode.** `submit_turn_with_options` with no grammar and no triggers.
   - Thinking is off: the no-think selector is set, and both `<think>` and
     `<tool_call>` are banned (`turn_banned_tokens`). With no grammar,
     `opening()` prefills nothing (`mind.rs:949-951`), so the bans are what
     hold.
   - `max_tokens` is 320 for a message, which fits one Discord message, and 480
     for a journal entry.
4. **Template.**
   - **One collection for both loops.** The shared prompt's `response`
     collection becomes `Offered { selector: "response", k: 1 }`
     (`schema.rs:1397-1409`). `Offered` selects explicit member names, so each
     loop lists its own set:
     - the main loop offers every non-talk template, which with provenance
       ranking gives exactly today's `top_k: 1` over the library;
     - a talk turn offers only its destination's templates: `talk/reply/*`,
       `talk/parting/*` or `talk/journal/*`.
   - **Why one collection.** Whichever template is shown always matches what
     the text is for, and no second collection has to be kept off the main
     loop.
5. **Where the text goes:**

| Destination | The decoded text becomes | Then |
|---|---|---|
| `reply` | a line on the thread to the party (`said`), carried to the authority in `WorldOp::Act.said` (§7.1) | PARKED |
| `sign_off` | the parting line on the thread | ENDED |
| journal | a `wrote` line in `bots/<bot>`, verbatim. The thread is marked read through the input line, so Discord's typing indicator stops (§8). | PARKED |

Every turn leaves a message or a journal entry, and it is the decoded text in
both cases. Nothing is prefilled.

**Cost.** Two decodes every turn: the decision's short call, then the text.
Add the catch-up summary when the gap holds more than six lines. After a park,
the slot was just resumed, so each pass may prefill the whole projected context,
not just the user turn. The KV that stays hot from the previous turn (§6.4)
covers a quick back-and-forth. The resumed-turn latency on a person's DM is
measured in the live check (§13).

### 6.4 Park and resume

- **Wait for durability, then drop.** After the voice turn seals, the runner
  waits for that turn's recurrent snapshot to be durable, then drops the
  `Sequence`, which frees the slot (`conversation.rs:5352-5361`).
  - **The new query.** The wait is a new non-consuming per-timeline query,
    `Substrate::snapshot_durable(timeline, turn)`, built from the snapshot's
    location and `sealed.is_durable` (`candle-conversation/src/projection/resolver.rs:5062-5069`).
  - **Why not the existing call.** `drain_durable_snapshots` is a global
    read-once queue whose only consumer is the scheduler's own idle parking
    (`candle-conversation/src/scheduler/recurrent_park.rs:24-28`). Draining it
    elsewhere would stop slots from parking.
  - **Why wait.** A resume on a new slot restores recurrent state through
    `restore_recurrent_state` from whatever snapshot the index holds, rejecting
    only one that is newer than the history (`scheduler/mod.rs:7739-7751`).
    Dropping before durability could silently resume on the previous turn's
    state.
  - **Latency.** The wait comes after the reply is posted, so it is off the
    person's critical path.
- **KV.** The turn's KV was truncated at seal (`scheduler/mod.rs:8559-8574`).
  Its residences stay hot until `demote_idle_hot`'s grace: at most 16 epochs,
  ticking on seal or flush passes or a 5 s idle tick (`substrate.rs:2121-2128`).
  Abandoned conversations drain to warm and cold through the existing tiering.
- **Never transient, never unresumed.** A conversation timeline is never marked
  transient (`substrate.rs:5592-5597`) or unresumed (`recurrent_park.rs:60-63`).
- **Resume.** `resume_conversation_with_projection` (`engine.rs:2505`), under
  the frame rule of §4.1. A resumed `Sequence` reconciles its carried
  collections afresh (`mind.rs:1533-1548`): the mission section and the
  journal-core chain are restored on every turn after a park, one
  `RestoreSection` each (`submitted_sections.rs:420-447`). That is part of the
  per-turn cost.

### 6.5 The talking stance

Collections belong to the one shared system prompt, and every layer frames on
it. A conversation is therefore a layer of the mind schema, `talk`, with its own
layer id and dials, framing on the same prompt. Its differences are selections
and a `talking` stance branch:

- **Unchanged:** identity, personality and world collections
  (`identity::selection_for`, `mind.rs:2012`). It is the same person.
- **Mission.** The current mission, re-reconciled every turn
  (`mind.rs:1927-1942`). For Zen, its agenda as well (`agenda_prompt`,
  `npcd/src/sim/missions.rs:520-547`).
- **Journal-core.** The character's journal-core entries (`mind.rs:1943-1954`).
- **Catch-up.** The `catchup` collection (§6.1).
- **Mood.** The current mood, read fresh from `Cast::mood_of`
  (`npcd/src/npcs/cast.rs:168-174`).
- **Tools.** The `tools` collection offering exactly the allowed acts
  (`show_within`), so what the decision reads is what it can decode.
- **Response.** The `response` offer of §6.3.
- **The `talking` stance branch:** the few lines of talk-only framing ("You are
  on the phone with {party}…"). Talk-only text lives here, never in shared
  sections, so an edit to it changes only the talk frame (§4.1).

**Bands.** A person's conversation decodes at **High**, and an NPC↔NPC one at
**Low**, through the timeline's `decode_band` (`scheduler/band.rs:14-40`), at
most 64 decodes a wave (`candle-conversation/src/scheduler/mod.rs:7011`). The
catch-up section seals at the layer's band (`submitted_sections.rs:593-600`).

---

## 7. Conclusions

### 7.1 How acts from a conversation reach the world

Every act from a conversation, including the line a `reply` writes, goes
through `Runtime::perform` (`perform.rs:272-322`). On a member that does not
write the world it travels to the authority as `WorldOp::Act` (`perform.rs:279-297`),
so mission and thread state each keep one writer (`worlds.rs:190-192`).

**Voiced text travels with the act.** The authority holds no live `Sequence`
for a character leased elsewhere, so it cannot voice anything. Voicing happens
on the lease holder, before the op is built:
- for a conversation, the voice turn (§6.3);
- for a main-loop channel post, §11.4.

`WorldOp::Act` and `perform_here` gain `said: Option<String>`, and
`enact::message`/`reply`/`sign_off` append `said` in place of the intent.
Today the op carries only `{npc, call, decided, pinned_at}`
(`npcd/src/authority/ops.rs:38-44`), and the call is re-parsed against
`tools::specs()` (`perform.rs:124`). Voicing runs after the mind's turn has
cleared `turn_in_flight` (`conversation.rs:3790-3798, 3855`; perform at
`runtime.rs:6596-6673`), so the voicing ask never waits on the mind's own
turn.

**Source.** `WorldOp::Act` and `perform`/`perform_here` gain
`source: ActSource`, with variants `Mind` and `Talk { party }`. The name avoids
a third `Origin` beside `phone::Origin` and `mission::Origin`.
- **No decision state.** For `ActSource::Talk` the authority skips
  `restore_decision`/`end_decision` (`perform.rs:127-134`), and the talk path
  reads nothing from the body's deliberation.
- **What remains concurrent.**
  - A talk op landing while a main-loop op sits between its restore and end
    sees nothing of that state, because it reads none.
  - A talk mission change landing during a main-loop act shows up in that act's
    before/after `moved` diff (`perform.rs:345-361`). The main loop then reads
    it as news, which is correct.

**Speaking steps and answers fire when a line lands.** `mark_spoken`
(`runtime.rs:1903-1937`) and `ledger.answered` (`enact.rs:1077-1081`) fire when
a conversation's line lands on the thread, not when a main-loop purpose is
handed over (§10).

| Kind | Act | Applied as |
|---|---|---|
| Commit | `promise` | the ledger (`enact.rs:952-973`). `to` resolves against thread members as well as the room (`here_named`, `enact.rs:152-160`). The `w.tell` announcement (`:969`) becomes a thread line when the party is not in the room. At most 3 per turn. |
| Remind | `remind` | as today, with `who` resolved against thread members (`enact.rs:975-990`) |
| Change, Zen (unordered) | `mission_*` | the agenda (`npcd/src/engine/agenda/perform.rs`), with critical steps protected (below) |
| Change, ordered body | `mission_*` (ordered specs) | below |
| Hand-off | `message`/`reach_out` to someone else | a purpose delivered (§10) |
| Noted | journal destination, with optional `reflect` | the `wrote` line; `feeling` → `note_feeling` |

**Ordered bodies get their own mission specs.**
- **Why the agenda acts don't fit.** They refuse an ordered body
  (`agenda/perform.rs:31-35`). Their required `priority` binds
  `Choices::Priorities`, which is empty for ordered bodies, so the empty-set
  rule would remove the whole act (`agenda/acts.rs:46`; `tools.rs:1854,
  2241-2247, 2343-2349`). An ordered mission has no title either: titles live
  on `agendum` (`mission.rs:284`).
- **So.** The same names (`mission_add`, `mission_edit`, `mission_cancel`,
  `mission_done`) get ordered-body specs with their own descriptions. `mission`
  binds the active mission's prompt, and there is no `priority`.
  - `mission_add` → `lodge_mission(.., start: true)` with
    `Origin::Lodged { by: account }`, which assigns it, and calls off the
    current mission and its operation (`perform.rs:406-447`;
    `sim/missions.rs:702-716`). Not `Origin::Own`, which closes as a routine
    (`missions.rs:659-669`).
  - `mission_edit` → a new `Missions::edit_active`. It keeps every done and
    every critical step, and refuses a mission with an operation, whose steps
    are the operation's.
  - `mission_cancel` → `cancel_mission`.
  - `mission_done` → the `report_done` handler (`npcd/src/engine/work.rs:195-320`),
    so every existing gate applies (`still_doable`, `written_up`,
    `unread_step`, `report_block`). That handler gains the critical-step check
    it lacks today (§15).

**Critical steps are protected on the agenda path too.**
- `agenda_edit` rebuilds `todo` and would drop an unsigned critical step, or
  re-create it as non-critical (`sim/missions.rs:388-399`). It keeps critical
  and done steps.
- `agenda_done` (`missions.rs:415-447`) is refused while a critical step is
  unsigned.

### 7.2 Authority

The party is known by account (D14):

- **Owner.** `payload(npc).owner_id == public_id(account)` (`npcd/src/api.rs:1092`).
- **Authorizer.** `WorldMissions.authorizers` (`npcd/src/engine/mission_gen/world.rs:199-203`)
  holds **account handles**, and `is_authorizer` takes the party's account.
  - **Why not names.** Display names come from the identity provider, can
    change, and collide. The creator's carries a " (The Creator)" mark
    (`npcd/src/engine/mod.rs:1199-1212`).
  - Names are kept only to render `critical_compass` (`authorize.rs:101`).
  - This is the identity check `docs/mission_templates.md:187-195` asks for.
- **Getting the account to the authority.** `WorldOp::Message` gains
  `account: Option<String>`, and `inbound::judge` fills it
  (`npcd/src/discord/dm/inbound.rs:193, 213-218`).
- **An NPC party** is never an authority over another NPC's mission or agenda.
- **Everyone else** gets no mission acts, for Zen's agenda as for an ordered
  mission. They can get a `promise`, and the NPC's main loop decides.

**Critical steps: a request, then a yes.** A conversation signs off a critical
step only when both hold:

1. **It asked.** When a purpose reaches a conversation whose party is an
   authorizer while `critical_next()` is `Some(step)` (`mission.rs:619-640`),
   the runner records `talk.pending_critical = step`. The posted line carries
   `requests: step` (a `Message` field, §11.1).
2. **The authorizer answered yes.** Their next message is affirmative
   (`is_affirmative`), and `step` is still `critical_next`, re-checked on the
   authority.

Then the authority runs `authorize`. A "yes" to "want tea?" approves nothing.

**The weaker path is removed.** The phone `read` path no longer authorizes
(`enact.rs:293-315`), so conversations are the only way a critical step is
signed off.

**Thrash guard.** A body's mission may change through a conversation at most
once per `CHANGE_COOLDOWN` = 10 min of world time, except by its owner. Within
the cooldown, the mission acts are struck through `cooling`. Authority ops
apply one after another, so concurrent changes resolve as last writer wins,
after each passes the authority check.

**The main loop is told.** After applying, the authority delivers an URGENT
`Concluded{party, act, applied_as}` through `Runtime::tell`
(`runtime.rs:2660-2672`), which reaches the character wherever it is held. The
next decision starts from the new mission and knows why.

### 7.3 Outcomes flow out, not back

An act's outcome reaches the conversation's next turn the way main-loop
outcomes reach theirs: a `<tool_response>` ahead of the user turn
(`compose_narrated`, `mind.rs:2400-2422`). `Within.owed` is built from the
party for a conversation, not from the room (`tools.rs:2186-2190`).

---

## 8. Guards

Conversation guards are **per conversation** and never touch the character's.

- **Today all strikes land on the character.** `Within.cooling` is the
  per-character union of `cooldowns.cooling(npc)` and `loop_guards.cooling(npc)`
  (`runtime.rs:2284-2290`). A remote `reply` stamps the character's
  `(npc, "reply")` cooldown (`perform.rs:306-308`).
- **A talk `Within` builds `cooling` only from its own state.** Acts with
  `ActSource::Talk` never call `cooldowns.took_act` or `loop_guards.record` for
  the character.

| Guard | Rule | Why |
|---|---|---|
| Burst window | A turn starts `BURST_MS` = 2,500 ms after the last unanswered message, and at most 8,000 ms after the first. | one answer to three quick lines |
| One turn at a time | A message or purpose arriving during a turn is queued for the next turn boundary. | injection without races |
| Repeats | A new non-mutating `LoopGuard::would_repeat(lines, text)` runs on the voiced text before it is sent, against this thread's own last 4 lines (the guard's 0.45 overlap, `loopguard.rs:112-118`). A repeat changes the destination to the journal, and the line "Started to tell {party} again that …" is written. There is **no carried-forward strike**: today's `record` strikes the speech family for up to 16 turns (`loopguard.rs:344-352`), which would silence a person's next 16 messages. | no posted repeats; Zen telling Mira what it told John is not a repeat |
| No reply cooldown with people | Person conversations have no `reply` cooldown. A 6 s cooldown (`cooldown.rs:251-254`) would strike a reply whenever a person answers within about 3 s. NPC↔NPC conversations keep 6 s, keyed `(npc, party)` (today `(npc, act)`, `cooldown.rs:333`). | people are never made to wait |
| Activation cap | NPC↔NPC only. `TALK_TURNS` = 8 turns per side per activation. An activation starts at a purpose, **or** at the first incoming line after the conversation went quiet (a journal turn or `sign_off`), so the answering side resets too. After the cap, `reply` and `message` are struck, and the turn goes to the journal or a `sign_off`. | two NPCs cannot talk forever, and the exchange ends in something |
| Quiescence | A journal turn sends nothing, so the other side takes no turn. | no idle chatter |
| Unprompted contact | A purpose for a party whose thread's last line is the NPC's own, still unanswered, is held until the party answers or `UNANSWERED_HOLD` = 1 h passes. | no message storms at someone away |

**Typing.** Discord's indicator derives "waiting" from the thread
(`npcd/src/discord/dm/typing.rs:42-51`), on the authority's bot
(`npcd/src/discord/dm/record.rs:8-12`). In the new phone (§11.1) "waiting"
means the person's last line is above the NPC's read cursor. A turn that ends in
the journal marks read through its input, so typing stops without the authority
knowing the holder's turn state.

---

## 9. Where it runs

The world's authority and a character's lease holder are different members.
Character leases are lent by weighted rendezvous over the npcd members
(`npc_lease.rs:5-16`), and the world lease is held separately
(`npcd/src/authority/world_lease.rs:1-6`). A conversation runs on the **lease
holder**, in a per-member runner for the characters held there
(`npcd/src/engine/talk/runner.rs`).

- **Delivery carries the lines.** `deliver_messages` runs on the authority
  (`environment.rs:333-352`). For a conversation thread it emits
  `Talk { thread, lines }`, the new lines themselves (seq, from, text, account),
  instead of `EventKind::Message`.
  - **Why carry the lines.** A `Told` pulse travels on the beat's pulse channel
    (`npcd/src/authority/beat.rs:182-195`), and nothing orders it against the
    thread's world record (`world_state/snapshot.rs:36-57`). A runner reading
    its local copy could find nothing new.
  - **Where it is intercepted:** at `environment.rs:467` locally, and at
    `follow.rs:77` on the holder.
  - **No reminders.** `unread_due` (`environment.rs:418-426`) excludes
    conversation threads, so the mind is never reminded about them.
- **Turns count toward the hand-back.**
  - **A count, not a set.** `Standing.turns.turning` becomes a count per
    character (`worlds.rs:386-397`).
  - **The runner's calls.** The runner calls `Standing::begin_turn`, which is
    refused while the character is being handed back, and `Standing::end_turn`
    around every conversation turn. These are not `Runtime::begin_turn`
    (`perform.rs:374`), which sends `WorldOp::Begin` and takes the mind's
    arrival and compass news (`runtime.rs:1885-1899`).
  - **When a hand-back happens.** A character is handed back only when no turn
    of its mind or any conversation is running.
  - **On the new holder.** Conversations resume only once `lease.ready(applied)`
    holds (`cast.rs:163`).
  - **`put_down`** (`cast.rs:241-248`), which can run without a hand-back when an
    admin or the authority moves a lease (`cast.rs:191-196`), cancels in-flight
    conversation turns before releasing them.
  - **What this prevents.** A conversation timeline written from two members.
- **A message is never left unanswered.** `Told` pulses are not persisted
  (`npcd/src/authority/told.rs:5-7`). The runner rescans whenever it starts a
  character: `spawn_character` (`runtime.rs:6265`; `api.rs:1455`) and
  `wake_character` (`cast.rs:206`). The rescan covers every conversation thread
  and takes a turn for lines with a seq above `talk.answered_seq`.
  - **A thread with no conversation yet** initialises the bookmark to the
    mind's delivered cursor (`phone.rs:879-901`), so it does not answer the
    whole history.
- **Duplicates are tolerated** (D13). An old holder's reply in flight may still
  land after a hand-back began.

---

## 10. Starting, resuming, injecting

- **A person DMs.**
  1. `inbound::judge` writes `WorldOp::Message{account, ..}`, and the authority
     calls `message_npc`, which writes the person as a member by account
     (§11.1).
  2. `deliver_messages` emits `Talk` (§9).
  3. The runner opens the conversation, resumes it, or queues the message.
  4. The mind learns of it through `Elsewhere` (§5.4).
- **The main loop wants to talk.**
  - **No thread text.** The main loop's `message{via: private, to, intent}`
    (`acts.rs:1171-1276`) and `reach_out` (`acts.rs:1092-1123`) no longer write
    thread text. Each delivers `Purpose{intent}` to the `(npc, to)`
    conversation, opening it if needed, under the §8 hold.
  - **What the conversation sees.** Its next user turn carries "You wanted to
    tell {party}: {intent}".
  - **The act's outcome text** becomes "You mean to tell {to} {intent}; you
    will say it in your own words" (`enact.rs:1082-1086`).
  - **Channels are unchanged.** `message{via: channel}` stays on the main loop
    (§11.4).
- **The main loop keeps off conversation threads.**
  - **`reply`.** It binds `Choices::Awaiting` (`tools.rs:1833`), which would
    include direct threads where the party spoke last. The main loop's
    `awaiting`, `leavable` and `invitable` sets (`sim/mod.rs:443-614`) exclude
    conversation threads, so its `reply` and `sign_off` never touch them.
  - **`invite`** is refused on a direct thread (`phone.rs:819-831`). A group
    starts only with `open_group`.
- **A conversation's own `reply` and `sign_off`** write to its party's thread.
  They are the only acts that do.
- **A hand-off** is a conversation's `message` or `reach_out` toward someone
  other than its party. It is the main loop's act on the main loop's path, and
  delivers a purpose to that other conversation.
- **NPC↔NPC.** A's reply on the A–B thread is delivered to B's conversation as
  `Talk`, exactly like a person's DM.

---

## 11. The phone, rebuilt

The current phone (`npcd/src/sim/phone.rs`) is replaced, not patched. The new
phone is `npcd/src/sim/phone/`, one concern per file.

### 11.1 Direct threads

- **Members are ids.** A member is a body id or an account handle, never a
  display name. The display name is looked up when rendering. The ledger keys
  promises by id too (today by name, `enact.rs:960-968`).
- **One thread per pair, forever.** A direct thread's id is derived from the
  sorted pair of member ids, so reaching someone again finds the same thread.
  - Its seq never restarts.
  - It is never evicted; `KEEP_THREADS` (`phone.rs:399`) does not apply to
    direct threads.
- **`sign_off` does not leave.** It marks the conversation ENDED (§4.3) and
  leaves membership alone. A direct thread is never closed by its members.
- **Read is bounded.** `read_through(who, thread, seq)` marks read up to a seq,
  never "everything". Today `read` marks every unseen line (`phone.rs:849-863`),
  which would hide a message that arrived during the turn.
- **A line** holds:
  - the voiced text;
  - the deciding intent, for diagnostics;
  - `requests: Option<step>` (§7.2);
  - the sender's account.

  Discord posts the text, so `spoken(intent)` (`npcd/src/engine/reported.rs:69`)
  is removed.

### 11.2 Channels and groups

Channels and group threads are a separate kind: fire-and-forget, on the main
loop.

- Members are ids.
- They are bounded by per-channel retention, not a world-wide count.
- The world channel (`phone.rs:53-90`) is one of them.
- The roster functions cover both kinds by id (`sim/mod.rs:443-614`): `enrol`,
  `contacts_for`, `invitees_for`, `leavable_for`, `awaiting_for`.

### 11.3 The main loop on the phone

- **Answering marks read** through the line answered.
- **The phone shows your own last line.** `read_phone` adds "You last said, {ago}: …".
- **Repeats are checked before posting.** `would_repeat` runs before the post,
  and a repeat is refused with the guard's nudge, never posted and then struck
  (`runtime.rs:6673, 6739-6799`).

### 11.4 Voiced channel posts

A main-loop `message{via: channel}` carries an intent. Before `perform` builds
the op, on the lease holder, the intent is voiced:
- an `UnsealedAsker::ask` on the mind's live sequence, with no tools;
- `<think>` and `<tool_call>` banned;
- the `response` offer `talk/channel/*`;
- the user turn "Post to {channel}: {intent}";
- a 320-token cap.

The text rides in `WorldOp::Act.said` (§7.1). The cost is one short decode per
channel post.

### 11.5 Discord

**Per-part cursors.** Each part of a split message is cursored:
- `Dm` gains `mirrored_part` (`npcd/src/discord/dm/record.rs:28-41`, a record
  that is `Copy` and `deny_unknown_fields`), and the server and world records
  gain the same.
- `split` is deterministic over the posted text, so a failed later part resumes
  at the right part.

**Duplicate suppression where Discord supports it:**
- **Create Message** carries `nonce` = a ≤ 25-character digest of
  `{world}:{thread}:{seq}:{part}`, with `enforce_nonce: true`
  (`npcd/src/discord/client.rs:252-267`). Discord's own window for this is a few
  minutes.
- **Execute Webhook** (`client.rs:456-481`) takes no nonce. The world mirror's
  webhook posts rely on the per-part cursor.

Within D13, a rare duplicate is accepted.

### 11.6 What the rebuild must carry

Everything that reads or writes `Threads` today moves to the new API:

- **World state.** The thread records (`npcd/src/world_state/snapshot.rs:55`).
- **Discord DM:** `npcd/src/discord/dm/{inbound, outbound, typing, record}`.
  These look threads up today by `direct_between(display name, character)`
  (`inbound.rs:204, 232`; `outbound.rs:144-147`; `typing.rs:43`).
- **Discord world mirror:**
  `npcd/src/discord/world/{dms, groups, channels, gap, rooms, record, outbound, apply, inbound}`.
- **Discord server:** `npcd/src/discord/server/{thread, guild, inbound, outbound, apply, record}`.
- **Discord trace:** `npcd/src/discord/trace/report.rs`.
- **MCP tools.**
  - `npcd/src/mcp/{message_send, message_read, thread_read, channel_post}`,
    which back the `bot_message_*`, `bot_thread_read` and `bot_channel_post`
    tools.
  - `zend/src/mcp/npcd_tools.rs`.
  - `zend-tools/src/tools/ops/bots.rs`.
- **Effector phone:** `npcd/src/effector/phone.rs` and its routes in `router.rs`.
- **Console API:** `message_npc`, `messages_with`, `say_on_channel_from` and
  `channel` (`runtime.rs:2777-2894`).
- **Roster and unread:** the roster in `npcd/src/sim/mod.rs:443-614`, and
  `npcd/src/sim/unread.rs`.
- **Engine:** `npcd/src/engine/{on_you, channels, survey, pulse, mod}.rs`.
- **Other sim users:** `npcd/src/sim/{posting, item, seed}.rs`. `seed::issue_handset`
  (`seed.rs:355-359`) gives each NPC its handset.
- **Zen tasks:** `npcd/src/authority/zen_task.rs:213`, which already reaches by
  body id.

---

## 12. Speech in the room needs a person in the room

**The problem.** Offering an act is what makes a character take it
(`tools.rs:342-344`, `263-281`). Offering `tell` to an NPC whose only company
is other NPCs is what makes it talk to them aloud instead of messaging them.

**The rule.** Room speech is for people. NPCs talk to each other by `message`.

| Act | Today | Becomes |
|---|---|---|
| `tell`, `ask` | `Nearby` | `PersonNearby`: a person's body is in the room |
| `whisper` | `AmongOthers` | `PersonAmongOthers`: a person is here, and somebody else to keep it from |
| `promise`, `remind` | `Nearby` (`acts.rs:937, 976`); both announce with `w.tell` (`enact.rs:969, 989`) | in the room, `PersonNearby`; on a thread, the party (§7.1) |
| `shout` | `Always` | `PersonInEarshot`: a person's body is where the shout reaches, **or the body is in field play** |
| `gesture` | unchanged | unchanged |
| `message` and the phone acts | `Always` | unchanged |

**Why the related acts are gated too.** Leave any of them open and it becomes
the room voice NPCs reach for, which is how `whisper` became the default once
before (`tools.rs:255-261`).

**Messaging needs no person.**
- **Always offered.** `message`, `reply`, `reach_out`, `sign_off`, `invite` and
  `open_group` stay `Availability::Always` (`acts.rs:1058-1385`), bound to the
  handset's threads.
- **The handset.** Every NPC gets one from `seed::issue_handset`
  (`npcd/src/sim/seed.rs:355-359`).
- **Without a handset.** A handset can be taken (`acts.rs:1135-1139`), and an NPC
  without one can talk to nobody unless a person is present. That is the
  existing rule, and it is intended.

**What "a person" is.** In every world a person is a `visitor:{handle}` body,
where `handle` is the person's **account handle** (`interaction.rs:105-112`;
already how npcd tells a person's body from a character's, `order_table.rs:56`).
- **One rule, both uses.** The handle is the same identity that authorizers,
  owners and thread members use (D14). `Within.people` lists these bodies, and a
  conversation with one is `Party::Person(account)`, with the account read
  straight off the prefix. So no world needs an `is_player` mark, and no second lookup
  turns a body into an account.
- **How a body is placed.** Three existing world ops:
  - `WorldOp::EnterBeside` places it (`npcd/src/authority/ops.rs:55-59`);
  - `WorldOp::Place` moves it (`ops.rs:63`);
  - `WorldOp::Leave` removes it (`ops.rs:61`).
- **In the moon** visitors come only from console physical sessions
  (`npcd/src/engine/mod.rs:1032-1046`), so moon NPCs speak aloud only to someone
  standing in a console session. That is intended.
- **In Battle Cities a player's body is an avatar, not a visitor.** The paths
  below are in the Battle Cities repository, `crates/sim/src/`.
  - **Bodies are born with a tower.** An avatar is born with its tower when
    the tower is created. `MessageCreateTower` (`exports/region/create_tower.rs`)
    spawns one `AvatarType::SuperSoldier` per tower, carrying the tower's name.
  - **Ownership.** The tower records it as `main: AvatarId`
    (`model/tower/mod.rs`), and the account records the tower
    (`model/account.rs`).
  - **Respawn keeps the id.** A dead avatar comes back with the same id at its
    tower (`model/avatar/respawn.rs`).
  - **So the rule is:** an avatar that is some tower's `main` is a person.
    In npcd's world it is the body `visitor:{handle}`, where `handle` is the
    owning account's handle.
  - **Not people:**
    - companions (`tower.companions`), which are AI-driven bodies with their own
      minds;
    - zenlings and critters, which are object types of their own;
    - spectators, who enter with no body at all.
  - **Field play.** A spectator therefore never unlocks room speech, and a
    player's avatar in a fight is also in field play.
  - **The bridge is the game's side of the contract.** npcd's `battle-cities`
    world is its own seeded tower (`npcd/src/sim/seed.rs:254-288`), and today
    nothing carries a game avatar into it. The Battle Cities server owns the
    bridge:
    - When an avatar that is a tower's `main` enters an npcd character's
      surroundings, the server places it with `EnterBeside` as
      `visitor:{handle}`, named for the avatar.
    - It sends `Place` as the avatar moves, and `Leave` when the avatar dies
      or the player disconnects. The respawn keeps the id, so the same
      visitor re-enters.
    - Companions, zenlings, critters and spectators are never sent.

    npcd needs nothing beyond the three ops, which take the handle as given.
    Until the bridge sends a body, a Battle Cities NPC simply has no person in
    the room, and messaging works as it does everywhere else.
- **NPC↔NPC room speech in front of a person** stays on the main loop (D2).

**What "field play" is.** There are no squads in the field sim today
(`npcd/src/sim/field.rs:244-341`; `:25-27` is a module comment). Field play is
defined as: the body holds a field stance, or a hostile is within its reach.
- `Within` gains `in_field: bool`, computed from that.
- It is tested against fixture fields.

**How it is built:**
- **New `Within` fields.** `people: Vec<String>` (the company that are
  visitors), `heard_by_person: bool` and `in_field: bool`, filled in
  `Runtime::within` from the `actors_at` read, which yields ids
  (`runtime.rs:2225-2234`). All are part of the grammar cache key.
- **One hearing rule.**
  - `heard_by_person` uses perception's rule.
  - `Scope::at(world, &at).reach(&place)` is already public
    (`npc-map/src/witness.rs:71-90`).
  - The voice-carries rule inside the private `legible` (`witness.rs:217-270`)
    is factored out as `pub fn carries(voice, reach, aimed_at_reader)`. It
    keeps the "aimed at the reader" cases (`witness.rs:228-231, 257`), so the
    grammar and perception never diverge.
- **New availabilities.** `Availability` gains `PersonNearby`,
  `PersonAmongOthers` and `PersonInEarshot`. They are matched in every
  exhaustive match:
  - `tools.rs:1085-1133`;
  - `tools.rs:2291-2313`;
  - `npcd/src/engine/pulse.rs:1162-1181`;
  - the `needs` strings (`pulse.rs:1220-1241`): "a person here", "a person here,
    and somebody else", "a person within earshot, or the field".
- **`to`** stays bound to the whole company (`tools.rs:2589`).
- **`tools::nearby`** takes `people` and `heard_by_person`
  (`tools.rs:1127-1133`).

**Model-facing text that steers NPCs to room speech is rewritten.** Each piece
below would otherwise tell the model to use an act the grammar no longer
offers:

| Where | Today | Becomes |
|---|---|---|
| `prompt::frame` (`npcd/src/engine/prompt.rs:529-542`) | "If somebody is here, speak to them … two people in one room texting is worse than either of them saying nothing." | "Speak to the people here. To reach another character, here or anywhere, message them." The advice to ask rather than announce, and to go where an answer sends you, stays. |
| `MESSAGE` description and doc comment (`acts.rs:1160-1170, 1189-1194`) | "If the person you want is standing in front of you, `tell` or `ask`." | "How you reach another character, even one standing beside you." |
| `TELL`/`ASK`/`SHOUT` descriptions (`tools.rs:288-291, 346-353`) | "how you talk to a person", "Not for talking to somebody who is with you — that is `tell`" | "how you talk to the people here" |
| `GESTURE` description (`tools.rs:632`) | refers to speech in company | refers to people present |
| `act` description (`acts.rs:115`) | "promise or order somebody, use `tell`, `ask`, `whisper` or `shout`" | names `message` for characters |
| `post_notice` description (`acts.rs:346`) | "`tell` reaches whoever is standing here" | "speak to the people here; message the others" |
| `promise` example (`acts.rs:966`) | room speech | a thread |
| Cooldown rationale (`cooldown.rs:225-254`) | messaging priced "dearer than being in the room" | pacing a correspondence |
| `IN_COMPANY` idle nudge (`runtime.rs:914-918`, used at `:1362`) | room speech | message the others; speak to people present |
| `TO_THE_TABLE` (`runtime.rs:787, 798`) | "do not answer the room" | names messaging |
| Mission compass (`npcd/src/engine/mission.rs:1037`) | "`ask` or `tell` them" | "message them" (or speak to a person present) |
| Work text (`npcd/src/engine/work.rs:1058-1062`) | "`move_to` {room} and `ask` or `tell` them" | "message them" |
| `awaiting_line` (`environment.rs:202`) | "`tell` them if they are here" | "message them" |
| `company_line` (`environment.rs:213-237`) | lists `tools::nearby` for NPC-only company | lists what is offered |
| Standing mission templates (`mission.rs:1737-1769, 1803-1829`, including trace-a-thread and visit-and-review) | face-to-face NPC talk | the same exchanges by message |
| Tool examples (`tools.rs:311-322, 381-412, 461-468, 518-531, 564-570`) | NPC↔NPC room speech | rewritten with a visitor present, or as messages |

**What it does not break:**
- Mission `met` steps sign off on presence (`runtime.rs:1551-1559`;
  `sim/missions.rs:1545`). Speaking steps move with `mark_spoken` (§7.1).
- The loop guard's speech family and the cooldowns are keyed by act name.
- NPCs no longer overhear one another when no person is there. What they would
  have overheard reaches them as messages and journal lines.

---

## 13. Code layout and tests

New, one concern per file:

| File | Concern | Tests (raw values) |
|---|---|---|
| `npcd/src/engine/talk/key.rs` | `(npc, party)` → conversation id and generation; prefix retirement | id strings per party kind and generation |
| `npcd/src/engine/talk/state.rs` | the §4.2 metadata | round-trip of each key's exact bytes |
| `npcd/src/engine/talk/clock.rs` | burst window, one-at-a-time queue, activation start | turn start times and activation resets for scripted arrivals |
| `npcd/src/engine/talk/catchup.rs` | `since` → the `catchup/now` section; verbatim or summary; kind priority | the section text for fixture journals and bookmarks, including late arrivals in the overlap |
| `npcd/src/engine/talk/within.rs` | `Within::on_thread(party)`, allow-list, own cooling | the exact offered set per party kind, authority, ordered or unordered body, cap and cooldown |
| `npcd/src/engine/talk/decide.rs` | the fork and the destination rule | the destination for exact call lists |
| `npcd/src/engine/talk/voice.rs` | user text, restated decision, template offer, bans, caps | exact user text and offered template set per destination |
| `npcd/src/engine/talk/guard.rs` | `would_repeat` use, NPC↔NPC cooldown, cap, unprompted hold | decisions for scripted line histories |
| `npcd/src/engine/talk/turn.rs` | resume → catch-up → publish → decide → voice → apply → release → wait → park | an end-to-end turn on the test engine: sealed text, `said` or `wrote`, bookmark |
| `npcd/src/engine/talk/runner.rs` | per-member runner; `Talk` delivery; rescan on spawn and wake; turn counting; cancel on `put_down` | a DM reaches the runner; a rescan answers a seq above the cursor; a hand-back waits for a running conversation turn |
| `npcd/src/engine/talk/authority.rs` | `ActSource::Talk` rules: owner and authorizer by account, request-then-yes, thrash guard | mission and agenda state before and after for each act × party |
| `npcd/src/engine/talk/critical.rs` | `pending_critical` recording and the affirmative match | authorize, or not, for scripted line pairs |
| `npcd/src/engine/experience/{line,buffer,concern,store}.rs` | lines, per-bot buffer and flush, `concern`, npcd's `journals::live::Live` per world | rendered lines and experience fields; flush order and caps; the exact journal paths per change; a flush lands under `groups/<g>/bots/<bot>` as `Actor::Npcd{group}` |
| `npcd/src/sim/phone/{thread,direct,channel,read,line,roster}.rs` | the rebuilt phone (§11) | stable pair ids; seq continuity; `read_through` bounds; no eviction of direct threads; `invite` refused on direct threads |
| `journals/src/live.rs` | the live store moved from zend (`zend/src/mcp/memory/mod.rs:121-290`) | as zend's existing memory tests |
| `journals/src/view/since.rs` | lines and folds after a point, with overlap and seen ids | exact lines for fixture trees around folded periods |
| `journals/src/ops/episodes.rs` | `Op::Episodes`: one commit, per-line `at` and id, per-line trailers | exact tree and commit message; replay renames each colliding id; refusal outside `bots/` |
| `journals/src/experience.rs` | the `Experience` line field | raw rendered bytes; fold conflict on change |
| `candle-conversation/src/conversation/publish.rs` | `Sequence::publish_sections` | an ask after publish sees a section sealed before it |
| `candle-conversation/src/substrate/durable.rs` | `snapshot_durable(timeline, turn)` | false before, true after a turn's snapshot is durable; the scheduler's drain is unaffected |

Changed:

- **Grammar.** `Mode` through `grammar_for`, `act_specs`, `show_within` and the
  cache key, plus the other `Mode::Physical` sites (§6.2). `allow` in
  `performable`. `estimated_paths` takes a call count.
- **Acts.**
  - `reply` binds the party whoever spoke last.
  - `sign_off` keeps membership.
  - `promise`/`remind` work on threads and are gated in the room.
  - `reflect` sets the mood only, in a conversation.
  - Ordered-body mission specs, `Missions::edit_active`, critical protection in
    the agenda and `report_done`.
  - `read` no longer authorizes (§6.2, §7).
- **Authority and leases.**
  - `WorldOp::Act` gains `said` and `source`; `WorldOp::Message` gains
    `account`; `WorldOp::Reshape` (`npcd/src/authority/ops.rs`, `perform.rs`).
  - `turning` becomes a count, and `put_down` cancels (`worlds.rs`, `cast.rs`).
  - `Talk` delivery with lines, and `unread_due` exclusion (`environment.rs`,
    `follow.rs`, `npcd/src/engine/event.rs`).
  - Authorizers as account handles (`mission_gen/world.rs`, `authorize.rs`).
- **Main loop.**
  - `Purpose`, `Elsewhere` and `Concluded` events.
  - The private `message`/`reach_out` deliver a purpose.
  - Channel posts are voiced.
  - `mark_spoken` and `ledger.answered` fire on landing.
  - Conversation threads are excluded from the main loop's `awaiting`,
    `leavable` and `invitable`.
- **Guards.** `would_repeat`; talk cooling isolated; cooldowns keyed per
  conversation (`loopguard.rs`, `cooldown.rs`, `runtime.rs:2284-2290`).
- **Discord.** Per-part cursors in the dm, server and world records; a nonce on
  Create Message.
- **Journals.**
  - `Actor::Npcd{group}` with its own read arm and permits.
  - The `bots/` child cap lifted, including the check.
  - `bots/` exempt from summary triggers, and folds on `BOTS_PER_PASS`.
  - Files: `journals/src/actor.rs`, `permit.rs`, `scope.rs`, `tree/check.rs`,
    `brief/mod.rs`, `ops/wire.rs`, `store/message.rs`,
    `zend/src/session/memory_writer.rs`.
- **Config.**
  - `worlds/<id>.yaml` gains `journals: {group}`.
  - motherd.yaml's `npcd:` block gains `journals_dir:`.
  - The agent gains a `journals-npcd` checkout on every machine that runs npcd.
  - The `battle-cities` group is created by the `zen` service person.
- **Shared prompt.**
  - `response` becomes `Offered`, with the main loop offering its set.
  - `catchup` is declared.
  - The `talk` layer and the `talking` stance branch are added.
  - The talk templates go under `responses/talk/{reply,parting,journal,channel}/`.
- **Speech gate.** Everything in §12, tested by the exact offered act set for:
  - NPC-only company;
  - NPC-plus-visitor company;
  - a visitor within a shout's reach;
  - field play.

  Also tested: the `needs` strings, and that `message` and `reply` are offered
  with NPC-only company and alone.
- **Phone.** Every dependant in §11.6.
- **Lifecycle.** `retire_npc` tombstones talk timelines.

Live check, after the tests, in the moon (two NPCs and one Discord DM with Zen):

- every message is answered by a message or a journal entry, and typing stops;
- DM lines read as speech, never intent;
- two NPCs in one room with no visitor exchange messages, never `tell`;
- an NPC↔NPC exchange settles by the cap, a journal turn or a `sign_off`;
- a `promise` lands in the ledger, and the main loop's next turn reflects it;
- a stranger's DM offers Zen no mission acts;
- a critical step is signed only after Zen asked and the authorizer said yes;
- no slot is held by a parked conversation;
- the time from a DM to the reply after a park is recorded.

---

## 14. Open questions

None. The last one, how a Battle Cities player is known to npcd, is settled
in §12: the player's body is `visitor:{account handle}`, and the game server
places it through the existing visitor ops. A Battle Cities companion whose
mind is npcd's is an NPC party like any other.

---

## 15. Found along the way

These are recorded here so they are not lost.

- **`report_done` skips the critical-step check** (`npcd/src/engine/work.rs:195-320`).
  `still_doable` refuses only a step whose target is in reach, and
  `missions.report` closes the mission regardless. §7.1 routes `mission_done`
  through this handler and adds the check, which fixes it for the main loop as
  well.
- **`sleep.rs:22` describes code that doesn't exist.** It says a finished day is
  "folded into the memory layer"; no code does that.
- **Two stale comments:**
  - `mind.rs:1756-1758` says "two decodes, the narrator then the act". The
    narrator is a synchronous render (`npcd/src/engine/narrator.rs:25-30`).
  - The `finish_turn` doc (`candle-conversation/src/conversation.rs:3829-3845`)
    says it prefills the next header. `finalize_turn_post_done`, which it calls
    at `:3863`, does not (`:4310-4314`).
- **`zen_task_act` keeps its handle on the replica.** It runs `perform_here`
  there, because task acts are not of the world (`perform.rs:279-284`), so
  `set_task` writes only the local copy. `tend_zen_tasks` polls only the worlds
  this member writes (`npcd/src/authority/zen_task.rs:171-173`), so the task is
  never polled.
