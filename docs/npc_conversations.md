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
  (`EventKind::Message{thread, from, unread}`, `:460-470`); the character must
  choose `read`, then `reply`, between room events, idle percepts and its
  mission.
- **Answering does not mark the thread read.** `message`/`reply`
  (`npcd/src/engine/enact.rs:1031-1090`) never call `threads.read`, so the
  URGENT unread reminder (`npcd/src/sim/unread.rs:21`, every 5 min) wakes the
  character on a message it has already answered.
- **The phone never shows the character its own replies.** `read_phone`
  (`enact.rs:363-372`) returns only unseen lines from others; whether it
  answered survives only in the 32-exchange tail (`npcd/src/engine/mind.rs:117`),
  which a day or chapter roll-over resets.
- **The loop guard fires after the repeat is posted.** It checks the act after
  it is appended (`npcd/src/engine/runtime.rs:6737` then `:6783`), and its state is in
  memory only (`npcd/src/engine/loopguard.rs:368-370`).
- **Discord posts are not idempotent.** A failed later part of a split message
  re-sends the earlier parts (`npcd/src/discord/dm/outbound.rs:193-207`), and a
  post Discord accepted but whose response was lost is posted again
  (`npcd/src/discord/post/mod.rs:140-145`; no nonce, `client.rs:263`).

A conversation removes the first three by construction — each message *is* a
turn of a timeline that holds every line both sides said — and §11 fixes the
last two for every path.

---

## 2. Decisions

| # | Decision |
|---|---|
| D1 | A conversation is **the NPC's mind loop on a message clock**: same persona, same schema family, same decode path, with the beat replaced by the arrival of a message and perception narrowed to what changed since the last turn. Common code, not a second agent. |
| D2 | A conversation is **one-to-one**: an NPC and one party, the party a person or another NPC. Group channels stay on the main loop's existing messaging (`message via: channel`), which is fire-and-forget and needs no turn structure (§11). |
| D3 | **Parked between turns.** A conversation holds no scheduler slot and no live `Sequence` between messages. It is a persisted substrate timeline that resumes on the next message. Thousands can exist; only the ones decoding cost VRAM. |
| D4 | **Mood, mission, beliefs, persona and memory are shared** with the main loop. It is the same person in two places. A conversation reads them each turn and writes back only through its acts' outcomes (§7.3). |
| D5 | **Experience is journaled.** What an NPC does, hears and concludes is written as lines of its `bots/<bot>` journal, batched off the beat path. The world's in-memory event log stays the per-beat perception of the main loop. |
| D6 | **Catch-up is a turn-scoped lock plus a cited summary.** The journal lines written since the conversation's last turn are carried into the turn as a section collection, a short decode summarises them with line citations into the user turn, the reply decodes with the lines still carried, and the collection is removed. |
| D7 | **One tool base. A conversation turn is the mind's act turn, using the existing catalog, compiled for a thread instead of a room.** It offers `reply`, `sign_off`, `reflect`, `promise`, `remind`, the `mission_*` acts, and `message`/`reach_out`, with no conversation-only tools. **Every turn has an outcome**: the turn must open with a call, and every act offered changes something. Choosing not to answer is `reflect`, whose line goes in the journal. There is no silent no-op. |
| D11 | **Speech in the room needs a person in the room.** `tell`, `ask`, `whisper` and `shout` are offered only when a human's body (`visitor:*`) can hear them. NPCs reach each other by `message`, which is a conversation (§12). |
| D8 | **A main-loop private `message` starts, resumes or injects into a conversation.** The main loop hands the conversation a *purpose*; the conversation decodes the words. In-room `tell` stays speech (`npcd/src/engine/body.rs:421-461`). |
| D9 | **Conversations conclude.** A conclusion — noted (`reflect`), commit (`promise`), change mission (`mission_*`), hand-off (`message`/`reach_out`) — is the one way a conversation changes what the main loop does. It is applied by the world's authority under authority, critical-step and thrash rules, never by the conversation itself. |
| D10 | **All NPCs, both directions.** Person↔NPC and NPC↔NPC use the same module. Zen and the moon household first; Battle Cities players are the scale target. |

---

## 3. What exists that this is built on

| Need | Existing piece | Anchor |
|---|---|---|
| Threads, messages, delivery cursor | `Threads`, `Thread`, `Message{seq, from, to, seen_by, origin}`; `undelivered_for` / `mark_delivered_for` | `npcd/src/sim/phone.rs:108-251, 403-414` |
| Discord DM in | `inbound::judge` → `WorldOp::Message{origin: Discord}` → `message_npc` | `npcd/src/discord/dm/inbound.rs:105-248`; `npcd/src/authority/perform.rs:144-151` |
| Discord DM out | thread append with no origin → `dm::outbound::post` on the bot beat | `npcd/src/discord/dm/outbound.rs:153` |
| World ops routed to the authority | `Runtime::perform` → `WorldOp::*` | `npcd/src/authority/perform.rs:272-322` |
| Character decode | `Minds`, `Live{sequence}`, `send_turn_with_options_async` | `npcd/src/engine/mind.rs:163-227, 1786-1789` |
| Carried sections per turn | `Carried::reconcile(..., Linking::Chain, ...)` | `candle-conversation/src/conversation/carried.rs:159` |
| Side decode that writes nothing | `UnsealedAsker::ask(question, band, scope, options)` | `candle-conversation/src/conversation/ephemeral_turn.rs:69-112` |
| Forced one-call turn | `TurnOptions.turn_grammar` + `assistant_prefill`; npcd's `compile_calls` | `candle-conversation/src/turn.rs:68-92`; `mind.rs:437-470` |
| Resume a parked timeline | `resume_conversation_with_projection` | `candle-conversation/src/engine.rs:2505` |
| Recurrent state parking | `park_durable_states`, `unpark_from_record` | `candle-conversation/src/scheduler/recurrent_park.rs:20-58, 153-224` |
| Idle KV demotion | `demote_idle_hot` (16-epoch grace) | `candle-conversation/src/substrate.rs:2121, 519` |
| Bands | `DecodePriority`; timeline metadata `decode_band` | `candle-conversation/src/projection/schema.rs:936-958`; `scheduler/band.rs:3-40` |
| Mood | `NpcPayload.mood` id; `MoodSink` → `Npcs::remember_mood`; vocabulary `Choices::Feelings` | `persistence/record.rs:1353`; `npcd/src/tenant.rs:744-764, 987-996`; `npcd/src/npcs.rs:477-500` |
| Missions | `Mission`, `Todo{critical}`, `add_todo`, `check_off`, `authorize`; `WorldOp::Lodge`; agenda acts | `npcd/src/engine/mission.rs:32-289, 603-661`; `perform.rs:406-447`; `npcd/src/engine/agenda/acts.rs:19-143` |
| Authorizers | `is_authorizer`, `is_affirmative` | `npcd/src/engine/authorize.rs:22-97` |
| Repeat measure | loop-guard overlap ≥ 0.45 over the last 4 utterances | `loopguard.rs:105-124` |
| Journals: bots section, npcd actor, permits | `Section::Bots`; `Actor::Npcd{person}`; npcd may note, episode, create, close under `bots` | `journals/src/scope.rs:19-20, 84, 110`; `journals/src/actor.rs:35-36`; `journals/src/ops/permit.rs:176-190, 263-266` |
| Journals: time ladder | Day→Week→Month→Year folds at `WINDOW` = 50; zend's memory writer folds and summarises | `journals/src/period.rs:19-37`; `tree/window.rs:36-60`; `zend/src/session/memory_writer.rs:65-86` |
| Journals: budgeted assembly | `frame.rs` `Filler` | `journals/src/view/frame.rs:53-139` |
| Batched-write precedent | `Op::Activity` buffer flushed per 60 s | `zend/src/mcp/noted_calls.rs:23-140` |

What does **not** exist, and §4–§12 build:

- a conversation runner beside the mind loop, and routing of private-thread messages to it;
- npcd writing to the journals repository at all (it depends only on `journal-core`, `npcd/Cargo.toml:20`);
- a single-commit multi-op journal write;
- a journal read of "everything after time T";
- releasing a conversation's slot at the end of every turn;
- the act catalog compiled for a thread (`Within::on_thread`), the four
  catalog changes in §6.2, and the authority rules for acts that come from a
  conversation;
- speech-act gates that tell a person in the room from an NPC;
- Discord post idempotency.

---

## 4. The model

### 4.1 Identity

A conversation is keyed by `(npc, party)`:

- `Party::Person(account_key)` — the npcd account a Discord user is linked to (`npcd/src/discord/link/index.rs:1-50`), or an in-game player;
- `Party::Npc(npc_id)`.

Its timeline id is derived, `npc-{npc}-talk-{party}`, so any member holding
the character's lease opens the same timeline (§9). It does **not** roll over
with the world day or chapter, unlike the mind's `npc-{id}-day-{day}-chapter-{chapter}`
(`npcd/src/engine/sleep.rs:128-130`). A conversation is preserved for as long as
the pair exists — like a zend conversation — with older turns retired by the
same retention rule as the mind (`KEEP_TURNS` = 160, `npcd/src/engine/retention.rs`)
and recalled through provenance.

NPC↔NPC is two conversations, one in each mind: A's `npc-A-talk-B` and B's
`npc-B-talk-A`. Each side decodes its own turns against its own memory. The
thread between them is the one `Thread` both already share.

### 4.2 State a conversation keeps

Stored as timeline metadata (replicated with the substrate), never in memory alone:

| Key | Meaning |
|---|---|
| `talk.party` | the party |
| `talk.thread` | the phone thread id |
| `talk.last_turn_ms` | the world-clock time the last turn finished — the catch-up bookmark |
| `talk.answered_seq` | the highest thread `seq` this conversation has taken as input |
| `talk.activation` | turns taken since the last main-loop purpose (NPC↔NPC cap, §8) |
| `talk.open` | open or ended |

The bookmark is **per conversation, not per NPC**. A thousand conversations
for one NPC hold a thousand timestamps and nothing else. The events between
turns live once, in the journal (§5).

### 4.3 States

```
            message / purpose
   ENDED ─────────────────────▶ GATHERING ──(burst window)──▶ TURN ──▶ PARKED
     ▲                              ▲                          │         │
     │                              └─────── message ──────────┼─────────┘
     └────────────────── sign_off ─────────────────────────────┘
```

- **GATHERING** — a message or purpose has arrived; wait for the burst window
  (§8) so a person typing three lines gets one answer.
- **TURN** — resume, catch up, decode one call, act on it.
- **PARKED** — the slot is gone; the timeline and metadata are durable.
- **ENDED** — the conversation said goodbye. The next message reopens it, as
  the same timeline; ending is a social act, not a deletion.

---

## 5. Experience as journal lines

### 5.1 What is written

Per NPC, into `groups/<g>/bots/<bot>/`, one line per experience that is the
character's own — never ambient percepts or unchanged rooms, which stay in the
world log:

| Kind | Source | Example line |
|---|---|---|
| Did | an act performed with intent (`perform_here`, `perform.rs:339-369`) | "Moved to the kitchen to check the oven." |
| Heard | speech or a message addressed to it (`event.rs:490-495`; delivered thread lines) | "Mira, on the phone: the parcel is at the door." |
| Said | a conversation reply | "Told Mira I'd bring it in." |
| Observed | a mission event, an arrival or departure of someone it knows | "Step 2 of 'Tidy the lounge' signed off." |
| Concluded | a conversation conclusion (§7) | "Agreed with John to pause the garden mission until Friday." |

The kinds reuse `journal_core::Claim` kinds (`journal-core/src/entry.rs:72-85`)
plus `Said`, `Concluded` and `Changed` (§5.1.1). Lines name the party by its
journal path where one exists, so a person's `people/<person>` journal and the
bot's line are linked (`Op::Link`, `journals/src/ops/mod.rs:100-107`).

#### 5.1.1 World changes that concern the NPC

A conversation knows only what the journal tells it. A change to the world that
the NPC did not make and was not in the room for, but which touches it or the
person it is talking to, must be a line too. Otherwise, a conversation resumed
after the house went into away mode, or after the mission it was discussing
was withdrawn, talks as if neither happened.

**What counts as concerning an NPC.** One function, `experience::concern`
(`npcd/src/engine/experience/concern.rs`), decides it for each world change.
It takes the change and the world, and returns the journals that should hear
of it:

| A change concerns… | when it… | Sources |
|---|---|---|
| the NPC | names it, or changes its body, phone or mission | mission lodged, assigned, withdrawn, stepped or authorized by someone else (`WorldOp::Lodge` / `Step`, `perform.rs:406-447`; `Runtime::cancel_mission`, `runtime.rs:2133-2148`; `authorize`, `enact.rs:293-315`) |
| the NPC | touches what it is responsible for: its mission's room (a standing mission's `where`, `mission_gen/standing.rs:61-145`), its building, or the devices it linked or onboarded | building ops (`npcd/src/authority/building.rs`); a bridge act by anyone else on one of those devices (`npcd/src/bridge/mod.rs:247-375`) |
| the NPC | happens to someone it has a conversation with | a mission that person lodged changes; that person arrives at or leaves the NPC's home world (a `visitor:*` body, `interaction.rs:280-287`) |
| the world | is world- or house-wide | a house mode set (`npcd/src/authority/house.rs`; `docs/house_modes.md`); an operator reshape (`npcd/src/effector/router.rs`, `/reshape`); the world day rolling over |

**World-wide changes are written once.** They go into the world's own journal,
`bots/world-<world>`, never copied into every bot's. Catch-up reads that
journal alongside the bot's own (§6.1). A house mode change in a world of a
thousand Battle Cities NPCs is then one line, not a thousand.

**Where it is hooked.** The world's authority already applies every change, as
the single writer of world records (`npcd/src/authority/worlds.rs:190-192`).
`concern` runs there, after a change is applied, and pushes each line onto the
same lock-free queue as the other experience lines (§5.2). It never runs inside
a character's turn and never waits on the journal.

**Lines say what changed and for whom.** For example: "John set the house to
away mode." "The garden mission was withdrawn by John." "The lounge TV was
turned off from the bridge by Mira." A change that touches a conversation's
person links to their `people/<person>` journal. The person's own journal is
not written to, because npcd may write only under `bots` (`permit.rs:176-190`).
So a later catch-up for that person, and the summary's "what matters to this
conversation", finds it.

**What is not written.** These stay in the world log, where the main loop's
per-beat perception already reads them:

- ambient percepts and rooms that did not change;
- arrivals and departures of characters the NPC has no conversation with;
- every happening in a room it merely passes through.

The journal is for what a person away for a week would want to be told, not a
transcript of the world.

### 5.2 How it is written

- **A sink, not an HTTP call.** npcd and zend run in one process (motherd). npcd
  gets a `JournalSink` trait (`npcd/src/engine/experience/sink.rs`), injected
  by motherd like `MoodSink` (`tenant.rs:987-996`). motherd's implementation
  writes into zend's `Memory` store as `Actor::Npcd{person: <bot owner>}` — the
  actor the permits already allow under `bots`. This avoids `/v1/journals`,
  which maps every caller to a Person and would be refused under `bots`
  (`permit.rs:157-165`).
- **Batched, one commit per flush across all bots.** Lines are buffered per bot
  in memory, then flushed every `FLUSH_EVERY` = 5 s, aligned with the store's
  `SYNC_EVERY` (`zend/src/mcp/memory/mod.rs:48`). A flush is **one commit** for
  every bot's lines. This needs `Store::write_all(ops)`, which checks every op
  against one tree copy and commits once (`journals/src/store/write_all.rs`).
  It is new: today one op is one commit (`store/mod.rs:195-208`), and the
  HTTP batch commits op by op (`zend/src/api/journals/write.rs:176-184`).
- **On a passing failure** the buffer is kept and merged ahead of newer lines,
  as `noted_calls` does (`noted_calls.rs:72-80`). The buffer is capped per bot,
  newest kept, at 200 lines.
- **Off the beat path.** Recording an experience is a push onto a lock-free
  per-world queue (`crossbeam::queue::SegQueue`). The flush runs on the
  persistence side, never inside `moment`.
- **The journal is created on first write** (`Op::Create` under `bots` by npcd,
  `permit.rs:263-266`). `AddBot` grants the bot read of its own `bots/<bot>`
  (`journals/src/ops/group.rs:372-406` appends it to `reads`).

### 5.3 The time ladder bounds long gaps

The journal folds itself. A window over 50 lines folds its oldest closed period
into a `Kind::Fold` line (`ops/period_fold.rs:58-116`); folds gather into
weeks, months, years. A conversation resumed after three months therefore
reads a handful of fold lines plus the recent window, not three months of
lines. This is why no per-conversation event store is needed.

### 5.4 What the main loop sees of its conversations

The main loop does not read its own conversations' timelines. Each main-loop
turn gets one `Elsewhere` event (rendered, not decoded) listing the `Said` and
`Concluded` lines its conversations wrote since the main loop's previous turn,
read from the journal buffer by the same `since` read as §6.1. So Zen walking
the house knows it just promised Mira the parcel.

---

## 6. A turn

### 6.1 Catch-up (the lock)

1. **Read since the bookmark.** `journals::view::since(tree, paths, after_ms, budget)`
   (`journals/src/view/since.rs`, new) returns the lines and folds written
   after `talk.last_turn_ms`, oldest first, fitted to a character budget.
   - Archives are ranged from `Period::of(level, after_ms)` (`period.rs:94-100`),
     and the window is filtered by `at > after_ms`.
   - A fold whose whole period lies after the bookmark stands in for its
     archived lines when the originals do not fit. When they do fit, the
     originals are used.
   - Paths read:
     - the bot's own `bots/<bot>`;
     - its world's `bots/world-<world>` (§5.1.1);
     - the party's journal (`people/<person>` or `bots/<other>`);
     - the journals those link to, while the budget lasts.
   - Each path is read only if `bot_may_read` allows it
     (`journals/src/actor.rs:166-177`). An ungranted path is skipped, not
     refused.
   - Budget: `CATCHUP_CHARS` = 12,000, the conversation's high journal budget.
     It sits between frame's Chat (6,000) and Worker (32,000) budgets
     (`frame.rs:22-42`).
   - The read is from the store, so a line written a second ago is there. There
     is no ingest lag; provenance retrieval of journal files is only for recall
     later.
2. **Carry the lines.** Each line becomes one section of a `catchup` collection,
   reconciled onto the conversation with `Carried::reconcile(.., Linking::Chain, ..)`,
   the same machinery that carries the mind's journal (`mind.rs:1943-1954`).
   The turn's selection names them, so they are present for the whole turn
   whatever reprojection does. This is the lock: a pin in the selection,
   never a VRAM-resident pin.
3. **Summarise.** If more than `VERBATIM_LINES` = 6 lines were read, the
   conversation's `UnsealedAsker` asks:
   - Scope is `AskScope::Conversation`, under the turn's selection, at the
     conversation's band.
   - The question: "What has happened since you last spoke with {party}, {gap}
     ago, that matters to this conversation? Cite each line you use by its #id."
   - Temperature is 0, thinking is off, and the answer is capped at
     `SUMMARY_TOKENS` = 192.

   With 6 lines or fewer, the lines are rendered verbatim and nothing is
   decoded. The ask writes nothing to the timeline (`ephemeral_turn.rs:1-15`).
4. **Write the user turn**, which contains:
   - the time and gap ("It is Tuesday 14:05. You last spoke with Mira 3 days ago.");
   - the summary with its citations;
   - the messages of the burst, each attributed and timed.

   Only this text is sealed. The carried lines are not.
5. **Reply under the lock** (§6.2). The reply attends over the carried lines, so a
   cited `#id` can be read in full.
6. **Release.** The next reconcile is called with an empty `catchup` want list,
   which removes the sections. Then `talk.last_turn_ms` is set to the turn's end.

The cost is one extra prefill of the user half: `ask` and the real turn each
prefill it (`ephemeral_turn.rs:94-110`). It is paid only when the gap holds more
than six lines.

### 6.2 The acts: the catalog the main loop already has

A conversation has **no tools of its own**. Its turn is the mind's act turn,
with the same catalog (`npcd/src/engine/tools.rs`, `acts.rs`, `agenda/acts.rs`),
the same grammar compiler (`grammar_for`, `mind.rs:1034`), the same parser
(`act::parse`, `act.rs:372`) and the same performer (`Runtime::perform`,
`perform.rs:272-322`). The only thing that differs is the situation it is
compiled for:

- **The mode is `Mode::InstantMessage`.** That is the existing mode for a party
  reached down a line (`npcd/src/engine/interaction.rs:44-58`).
  `Availability::PhysicalOnly`, `Embodied`, `Nearby` and the rest already strike
  the world acts in that mode (`tools.rs:1085-1108`).
- **`Within` is the thread's, not the room's.** A conversation builds
  `Within::on_thread(party)`, in which:
  - every world-binding set is empty (places, rooms, parts, links, devices,
    stations, carried, operable), so the empty-set rule takes the world acts
    out of the grammar the way it already does for a character with nowhere to
    go (`tools.rs:1952-1954`);
  - the thread sets hold only this conversation's thread, so every `to`
    (`Choices::Threads` / `Awaiting`) binds to exactly one value: the party.
  - The talk itself cannot touch the body, so the runner never takes the
    character's `Resident` mutex (`mind.rs:302, 1747`). What the conversation
    decides about the world goes through the authority as conclusions (§7).

What that leaves offered, every act an existing one:

| Act | Exists at | In a conversation it… |
|---|---|---|
| `reply` | `acts.rs:1290-1333` | answers the party on this thread |
| `sign_off` | `acts.rs:1058-1089` | ends the conversation, with a parting |
| `reflect` | `tools.rs:766-860` | says nothing back. Its `situation` and `inner_thoughts` become the journal line, and `feeling` becomes the mood. This is the act for choosing not to answer. |
| `promise` | `acts.rs:932-968` | commits to the party, held in the ledger (`enact.rs:952-973`) |
| `remind` | `acts.rs:971-1007` | reminds the party of a promise they made |
| `mission_add` / `_edit` / `_cancel` / `_done` | `agenda/acts.rs:19-138` | changes the mission (§7) |
| `message` / `reach_out` | `acts.rs:1092-1276` | hands off to someone else: starts or adds to *that* conversation (§10) |

Four catalog changes make these work on a thread. Each is a change to the
existing act, made once for both loops. There is no conversation-only variant.

- **`promise` and `remind` are offered on a private thread as well as in the
  room.** Their availability becomes `Nearby` *or* a private thread with the
  person, and `to`/`who` bind to the company plus the private-thread parties.
  `enact::promise` resolves `to` against those parties as well as the room
  (`here_named`, `enact.rs:957`). Its `w.tell(...)` announcement in the room
  becomes a line on the thread when the party is not here (`enact.rs:969`).
  Both loops gain this, because a promise made by text is as binding as one
  made in person.
- **`reflect` writes its journal line in both loops** (`Did`/`Inferred`, §5.1).
  It already sets the mood through `note_feeling` (`runtime.rs:2949-2955`).
  It still starts the reflection exactly as it does today
  (`docs/reflection_and_dreams.md`).
- **The mission acts are offered to an ordered body too, when the party has the
  authority to change its mission** (§7.2). Today they are offered only to a
  body that keeps an agenda (`agenda/acts.rs:1-5`). The act names and
  parameters are unchanged.
- **The phone acts mark the thread read** (§11). Again, this applies in both
  loops.

**Every turn has an outcome.** A turn that changes nothing anywhere might as
well not have happened. Every act above has a consequence somewhere:

- a line sent, or a parting;
- a journal line and a mood;
- a promise in the ledger;
- a mission changed;
- another conversation started.

The turn is forced to open with a call (`assistant_prefill` with a closed think
block and the call marker, as `mind.rs:1729-1742` does) and cannot close
without one. So the do-nothing turn cannot be decoded. Choosing not to answer
is `reflect`, and its required `inner_thoughts` must say what the NPC took from
the message and why it is not answering ("Mira asked again about the parcel. I
already told her it's in, and I won't say it twice."). That line is in the
NPC's journal, where every later catch-up, every other conversation and the
main loop see it (§5.4).

**Up to three calls a turn.** The grammar is `compile_action_loop` with
`max_calls` = 3 and `close_turn` (`candle-conversation/src/stencil/tool_call.rs:1207`).
That allows "Sure, I'll bring it in" to be `promise` followed by `reply` in one
turn. A `reply` or `sign_off` closes the turn, because what is said is said
last.

**The words.** `reply`/`sign_off` carry `intent`, not wording, exactly as they
do in the main loop. The narration pass voices them (`narrate_act`,
`mind.rs:972`; `runtime.rs:6685-6694`). That is the same path that writes
the words of every DM today, so a conversation sounds like the character
in the room. The voiced line is capped at 1,200 characters, below Discord's
2,000-character split point, so a conversation line is one Discord message and
never a split. Thinking is allowed, up to `THINK_TOKENS` = 256.

After the acts:

- a `reply` is appended to the thread as the NPC, and Discord mirrors it on the
  next bot beat (`dm/outbound.rs:153`);
- `sign_off` also marks the conversation ENDED;
- every turn marks the thread read through the last line it took as input
  (`Threads::read`, `phone.rs:849-863`).

### 6.3 Park

After the call is acted on:

- the `Sequence` is dropped, which frees the slot (`conversation.rs:5352-5361`);
- the turn's KV was already truncated at seal (`scheduler/mod.rs:8559-8574`);
- the recurrent state was snapshotted at seal and parks;
- the turn's residences stay hot until `demote_idle_hot`'s grace
  (`substrate.rs:2121`, 16 epochs, about 80 s).

So a quick back-and-forth re-projects from hot, and an abandoned conversation
drains to warm and then cold through the existing tiering. No new eviction
policy is needed.

The next message resumes with `resume_conversation_with_projection`
(`engine.rs:2505`), and `unpark_from_record` stages the snapshot
(`recurrent_park.rs:153-224`). A conversation timeline is never marked
transient, because transient timelines write no snapshot and never park
(`substrate.rs:5592-5597`; `recurrent_park.rs:60-63`).

### 6.4 Prompt and projection

A conversation turn projects under a **`talk` layer** of the character's mind
schema, served from the mind store (`npcd/src/mind_store.rs:113`) beside the
acting layers. It carries:

- the same identity, personality and world collections as the mind
  (`identity::selection_for`, `mind.rs:2012`): the same person;
- the current mission, the same views the main loop reads: it knows what it is
  doing, and a Commit or Change is weighed against the mission it is actually on.
  - The `mission` section, re-reconciled each turn (`mind.rs:1927-1942`), so a
    mission the main loop changed between messages is the one the conversation
    sees.
  - For an unordered body (Zen), its agenda (`agenda_prompt`,
    `npcd/src/sim/missions.rs:520-547`).
  - The mission compass (`mission_compass`, `npcd/src/engine/runtime.rs:1614-1800`),
    rendered into the user turn when it has changed since the conversation's
    last turn.
- the character's own journal-core entries (`mind.rs:1943-1954`): its sense of
  its day;
- the `catchup` collection (§6.1);
- the current mood, read from `Cast::mood_of` (`npcd/src/npcs/cast.rs:169-174`)
  into a one-line persona field. In the main loop, mood is delivered as an IDLE
  event (`runtime.rs:5510-5532`). In a conversation it is read fresh each turn,
  because the main loop may have changed it since.

The `tools` collection is the same one the mind installs. A conversation's
selection names the acts its grammar offers (`tools::show_within`,
`mind.rs:2027`), so the list it reads and the list it can decode are the same,
exactly as in the main loop.

Bands: a person's conversation decodes at **High**, and an NPC↔NPC one at
**Low**. The band is set through the timeline's `decode_band` metadata
(`scheduler/band.rs:3-40`). Admission then interleaves thousands of small
turns by band and FIFO (`scheduler/admit/order.rs:44-84`), at most 64 decodes
a wave (`scheduler/mod.rs:7011`).

---

## 7. Conclusions

### 7.1 Kinds

A conclusion is the outcome of an act from §6.2, not a separate structure. Every
one of them is recorded as a `Concluded` journal line, naming the party.

| Kind | Act | Effect |
|---|---|---|
| Noted | `reflect` | a journal line and a mood; nothing else changes |
| Commit | `promise` | a ledger promise to the party (`s.ledger.promise`, `enact.rs:968`). It outlives the conversation and is reminded, whether or not the NPC wants it to be (`acts.rs:927-940`). |
| Change | `mission_add` / `mission_edit` / `mission_cancel` / `mission_done` | the mission changes (§7.2) |
| HandOff | `message` / `reach_out` to another party | a purpose delivered to that conversation (§10) |
| Ended | `sign_off` | the conversation is ENDED; its last conclusion is what it came to |

### 7.2 Application

Every act from a conversation goes through `Runtime::perform`'s routing
(`perform.rs:272-322`), so on a member that does not write the world it travels
to the authority as `WorldOp::Act`, as a main-loop act does. Mission state
therefore has one writer, as it already does (`npcd/src/authority/worlds.rs:190-192`).
The act carries `Origin::Talk{party}`, which the authority uses for the rules
below.

| Act | Applied as | Rules |
|---|---|---|
| `promise` | the ledger, as today | at most 3 promises per conversation turn |
| `mission_*` on an unordered body (Zen) | the agenda, as today (`agenda/acts.rs`) | thrash guard |
| `mission_*` on an ordered body | **party is the owner or an authorizer**: applied to the active mission. `mission_add` lodges and assigns with `Origin::Own`, replacing the current mission (`WorldOp::Lodge`, `perform.rs:406-447`). `_edit` edits it, `_cancel` withdraws it (`Runtime::cancel_mission`, `runtime.rs:2133-2148`), and `_done` hands it in. **Party is anyone else**: the act is not offered, because the binding sets are empty (§6.2). The NPC can still `promise` to raise it, and its main loop decides. | thrash guard |
| `message` / `reach_out` | a purpose delivered (§10) | the target's §8 guards |

**Authority.** The party is known exactly, which a phone `read` is not:

- a person's party is the linked npcd account, and its owner match is
  `owned_by` (`npcd/src/engine/mod.rs:708-738`);
- an authorizer match compares the account's display name against
  `WorldMissions.authorizers` with `is_authorizer` (`authorize.rs:22-97`).
- The match is on the linked account, not on a name typed in a message. That
  is the identity check `docs/mission_templates.md:187-195` asks for.
- An NPC party is never an authority over another NPC's mission, so the
  mission acts are never offered to an ordered body talking with an NPC.

**Critical steps.** A conversation never signs off a critical step, just as
`check_off` refuses to (`mission.rs:606`). If an authorizer's message in a
conversation is affirmative for the active mission's critical next step
(`critical_next`, `mission.rs:619-640`; `is_affirmative`), the authority runs
`authorize` exactly as a phone `read` of that message does today
(`enact.rs:293-315`). The conversation's model output plays no part in that
decision; it rests on the authorizer's own words.

**Thrash guard.** A body's mission may change through a conversation at most
once per `CHANGE_COOLDOWN` = 10 min of world time. An owner or authorizer is
exempt. Within the cooldown, the mission acts are struck from the conversation's
grammar, as `cooling` strikes an act taken too recently (`tools.rs:1996-2002`).
When two changes conflict in one moment, the higher authority wins (owner, then
authorizer, then everyone else), and between equals the later one wins.

**The main loop is told.** After applying, the authority delivers the character
an URGENT `Concluded{party, conclusion, applied_as}` event through
`Scheduler::deliver` (`tick.rs:919-942`). The next main-loop decision therefore
starts from the new mission and knows why.

### 7.3 Outcomes flow out, not back

An act's outcome reaches the conversation's next turn as the main loop's
outcomes reach its own: a `<tool_response>` ahead of the user turn
(`compose_narrated`, `mind.rs:2400-2422`). Nothing re-enters the turn that
produced it. Mood moves through the existing `MoodSink` →
`Npcs::remember_mood` (`npcs.rs:477-500`; it writes only on change), shared
by both loops.

---

## 8. Guards

| Guard | Rule | Why |
|---|---|---|
| Burst window | A turn starts `BURST_MS` = 2,500 ms after the last unanswered message, and at most 8,000 ms after the first. | one answer to three quick lines |
| One turn at a time | A message or purpose arriving during a turn is queued for the next turn boundary; it is never dropped and never starts a second decode. | injection without races |
| A repeat is refused, and still leaves a line | The loop guard runs before the append, in both loops (§11). A `reply`/`sign_off` it catches (overlap ≥ 0.45 with the last 4 utterances, `loopguard.rs:112-118`) is refused, not sent. The refusal writes "Started to tell {party} again that …; had already said it." to the journal, so the turn still has an outcome. | no posted repeats, and still an outcome |
| Activation cap | In an NPC↔NPC conversation, `TALK_TURNS` = 8 turns per side per activation (an activation starts at a main-loop purpose). After the cap, `reply` and `message` are struck from the grammar, as `cooling` strikes acts (`tools.rs:1996-2002`). What is left (`sign_off`, `reflect`, `promise`, the mission acts) all conclude. | two NPCs cannot talk forever, and the exchange ends in something |
| Quiescence | `reflect` sends nothing, so the other side gets no message and takes no turn. NPC↔NPC exchanges settle at the first `reflect` or `sign_off`, each leaving its line. | no idle chatter |
| Unprompted contact | A purpose for a party whose thread's last line is the NPC's own and unanswered is held until the party answers, or for `UNANSWERED_HOLD` = 1 h. | no message storms at a person who is away |
| People are never capped | Person↔NPC conversations have no activation cap. A person may always get an answer. | the product |

A turn without a `reply` sends the party nothing. On Discord, the typing
indicator (`npcd/src/discord/dm/typing`) shows while a turn decodes and stops
when the turn ends. Whatever the turn changed (a journal line, a promise, a
mission) shows wherever that change shows.

---

## 9. Where it runs

A conversation runs on the member holding the character's lease, beside its
mind. When the lease moves, parked conversations need no handover: the
timeline, its metadata and the thread are all replicated state, and the next
message resumes on the new holder. A conversation in TURN when the lease moves
is lost for that turn. Its input messages are still unanswered
(`talk.answered_seq` did not move), so the new holder takes the turn again.

---

## 10. Starting, resuming, injecting

- **A person DMs.** `inbound::judge` → `WorldOp::Message` → `message_npc`, as
  today. Then `deliver_messages` routes a private thread with a conversation
  party to the conversation runner instead of the mind. It opens the
  conversation if there is none, resumes it if parked, or queues the message
  if a turn is running. The mind gets no `Message` notice for that thread. It
  learns of the exchange through `Elsewhere` (§5.4).
- **The main loop wants to talk.** The existing `message{via: private, to, intent}`
  act (`npcd/src/engine/acts.rs:1171-1276`) no longer writes thread text for a
  private thread, and neither does `reach_out` (`acts.rs:1092-1123`). Each
  delivers `Purpose{intent}` to the `(npc, to)` conversation, opening it if
  there is none. The conversation's next user turn carries "You wanted to tell
  {party}: {intent}", and its acts decide what is said. `message{via: channel}`
  is unchanged (§11).
- **A hand-off** is a conversation performing `message` or `reach_out` toward
  someone other than its party. It is the same act on the same path as the
  main loop's, so it delivers a purpose to that other conversation.
- **NPC↔NPC.** A's reply appended to the A–B thread is a message to B, so it
  starts or resumes B's conversation exactly as a person's DM does.

---

## 11. Channels stay on the main loop

Group channels are fire-and-forget, so they stay with the main loop's messaging.
They get the repetition fixes the conversation gets by construction:

1. **Answering marks read.** `message`/`reply` on any thread calls
   `Threads::read` for the sender through the line answered (`enact.rs:1031-1090`),
   so `unread_due` stops re-waking the character on what it answered.
2. **The phone shows your own last line.** `read_phone` adds "You last said, {ago}: …"
   for the thread (`enact.rs:363-372`).
3. **The repeat check runs before the post.** The loop guard's overlap test
   runs on a speech act's candidate text against the last 4 utterances
   **before** `perform` appends it. A repeat is refused with the guard's nudge,
   never posted and then struck (`runtime.rs:6737-6799`). The guard's history
   is rebuilt from the character's own thread lines on load, not held only in
   memory (`loopguard.rs:368-370`).
4. **Discord posts are idempotent** on every path (DM, server, world):
   - each part of a message is sent with `nonce` = `{thread}:{seq}:{part}` and
     `enforce_nonce: true` (`client.rs:252-267`);
   - the mirrored cursor records the part posted, not only the message
     (`dm/outbound.rs:193-210`, `server/outbound.rs:59-77`,
     `world/outbound.rs:80-120`).

   A lost response or a failed later part then resumes at the right part and
   never duplicates.

---

## 12. Speech in the room needs a person in the room

**The problem.** Conversations only work if NPCs use them. Today, an NPC that
wants to reach another NPC standing nearby has `tell` offered and takes it.
`tell` is described as "your ordinary speaking voice and how you talk to a
person" (`npcd/src/engine/tools.rs:346-353`), and it is offered whenever anybody
at all is in the room (`Availability::Nearby`, `tools.rs:345`; `company >= 1`,
`tools.rs:1132`, `:2299`). `shout` is offered always (`tools.rs:287`). Room
speech is a `Happening::Said` in the world log (`npc-map/src/world.rs:1910-1930`):

- nothing preserves it as a conversation;
- nothing concludes it;
- it lives only in each listener's 32-exchange tail.

That is the ring of characters announcing things to each other that §1 starts
from. The catalog already knows the lesson: offering an act is what makes a
character take it ("being invited to name one is what makes it invent a person
who is not there", `tools.rs:342-344`; the empty-room narration,
`tools.rs:263-281`). Offering `tell` to an NPC whose only company is other NPCs
is what makes it talk to them aloud instead of messaging them.

**The rule.** Room speech is for people. NPCs talk to each other by `message`.

| Act | Today | Becomes |
|---|---|---|
| `tell` | `Nearby`: anybody here | `PersonNearby`: a human's body is in the room |
| `ask` | `Nearby` | `PersonNearby` |
| `whisper` | `AmongOthers`: two or more here | `PersonNearby` plus `AmongOthers`: a human is here, and somebody else to keep it from |
| `shout` | `Always` | `PersonInEarshot`: a human's body is where the shout reaches |
| `gesture` | unchanged | unchanged. It is not speech, and two NPCs nodding at each other is still a room. |
| `message` | `Always` | unchanged. It is how an NPC reaches anyone, including someone standing beside it. |

Gating `ask` and `whisper` along with `tell` and `shout` matters. Leave either
open and it becomes the room voice that NPCs reach for, which is exactly how
`whisper` became the default once before (`tools.rs:255-261`).

**What "a human" is.** A person in a world is a `visitor:{handle}` body. It is
placed by a physical interaction session (`npcd/src/engine/interaction.rs:105-112`,
`visiting`, `:280-287`) and derived from the account, so it is stable. The
prefix is already how npcd tells a person's body from a character's
(`npcd/src/engine/order_table.rs:56`). Battle Cities players are visitors, so
NPCs speak aloud to players, and in front of them.

**How it is built:**

- `Within` gains two fields, `people: Vec<String>` (the company that are
  visitors, by name) and `heard_by_person: bool`. `Runtime::within`
  (`runtime.rs:2217-2234`) fills them from the same `actors_at` read that fills
  `company`.
- For `heard_by_person`, the shout's reach is the hearing rule
  `npc-map/src/witness.rs` already applies to a `Voice::Shouted` event. The
  gate asks the same question the listeners' perception does, so the grammar and
  the world cannot disagree about who would hear it.
- Both fields are part of the grammar's cache key (`tools.rs:1944-1948`), as
  `company` is.
- `Availability` gains `PersonNearby` and `PersonInEarshot`. They are matched
  in both offer functions (`tools.rs:1085-1133`, `:2291-2313`) and in the
  `needs` text pulse shows (`npcd/src/engine/pulse.rs:1221-1228`): "a person
  here" and "a person within earshot".
- `to` stays bound to the whole company (`Choices::Company`, `tools.rs:2589`).
  With a person in the room, an NPC may address another NPC aloud: the person
  sees the scene, which is what being in a room with characters is for.
- The descriptions change to match:
  - `tell`: "how you talk to the people here";
  - `message` (`acts.rs:1171-1276`) gains: "how you reach another character,
    even one standing beside you".
- The NPC-to-NPC examples (Maker-04 and the others, `tools.rs:381-412`,
  `:461-468`) are rewritten with a visitor present, because an example of an
  act the grammar would not offer is an example of nothing.

**Messaging needs no person.** The gate covers room speech only. `message`,
`reply`, `reach_out`, `sign_off`, `invite` and `open_group` stay
`Availability::Always` (`acts.rs:1058-1385`). They bind to the threads on the
character's handset, so an NPC alone with other NPCs, or alone entirely,
always has them. The one precondition is the existing one: a character is on
the threads from the moment it carries a handset (`npcd/src/sim/phone.rs:48,
72-75`). Every NPC that talks carries one.

**The guidance that pushed the other way is rewritten.** It tells NPCs to
prefer the room to the phone, which was right while room speech was the
conversation and is now the reverse:

- `prompt::frame` (`npcd/src/engine/prompt.rs:529-542`): "Being with people
  beats writing to them. If somebody is here, speak to them … two people in one
  room texting is worse than either of them saying nothing." It becomes: "Speak
  to the people here. To reach another character, here or anywhere, message
  them." The advice to *ask* rather than announce, and to go where the answer
  sends you for anything physical, stays.
- `MESSAGE`'s description and doc comment (`acts.rs:1160-1170`, `:1189-1194`):
  "If the person you want is standing in front of you, `tell` or `ask`." This
  is replaced by the reach-anyone wording above.
- The `message`/`reply` cooldown's rationale (`npcd/src/engine/cooldown.rs:240-254`)
  says messaging is priced "dearer than being in the room". The six seconds
  stays, because ten sends a minute is still a conversation. Its reason becomes
  pacing a correspondence, not steering NPCs towards the room. NPC↔NPC
  exchanges are bounded by the conversation's own clock and cap (§8).

**What it does not break:**

- Mission `met` steps sign off on presence (`runtime.rs:1551-1559`;
  `sim/missions.rs:1545`), not on speech.
- The loop guard's speech family (`loopguard.rs:105`) and the cooldowns
  (`npcd/src/engine/cooldown.rs:430-453`) are keyed by act name and still apply
  to whichever act is offered.
- NPCs stop overhearing one another when no person is there, because there is
  nothing to overhear. What they would have overheard now reaches them as
  messages and journal lines (§5).

---

## 13. Code layout and tests

New, one concern per file:

| File | Concern | Tests (raw values) |
|---|---|---|
| `npcd/src/engine/talk/key.rs` | `(npc, party)` → timeline id, metadata keys | id strings for Person and Npc parties |
| `npcd/src/engine/talk/state.rs` | the §4.2 metadata, read/write | round-trip of each key's exact bytes |
| `npcd/src/engine/talk/clock.rs` | burst window, one-at-a-time queue | turn start times for scripted arrival sequences |
| `npcd/src/engine/talk/catchup.rs` | `since` read → `catchup` wants, verbatim-or-summary choice | wants for a fixture journal at given bookmarks; the 6-line threshold |
| `npcd/src/engine/talk/within.rs` | `Within::on_thread(party)`: the thread's situation for the existing grammar | the exact offered act set for a person party, an NPC party, an owner and an authorizer party, an ordered and an unordered body, and after the activation cap or within the change cooldown |
| `npcd/src/engine/talk/guard.rs` | activation cap, unprompted hold | decisions for scripted line histories |
| `npcd/src/engine/talk/turn.rs` | resume → catch-up → ask → turn → act → release → park | an end-to-end turn on the test engine: sealed user text, thread append, bookmark |
| `npcd/src/engine/talk/runner.rs` | per-world runner, routing from `deliver_messages` | a private DM reaches the runner, not the mind |
| `npcd/src/engine/talk/authority.rs` | the authority's rules for an act carrying `Origin::Talk{party}`: owner/authorizer match on the linked account, thrash guard, conflict order | mission state before and after for each mission act × party authority |
| `npcd/src/engine/experience/{line,sink,buffer}.rs` | experience → journal line, batching | rendered line text per kind; flush ordering and cap |
| `npcd/src/engine/experience/concern.rs` | which journals hear of a world change | for each change kind in §5.1.1, the exact journal paths returned, including none for an unconcerned NPC and one `bots/world-<world>` for a house mode |
| `journals/src/store/write_all.rs` | multi-op single commit | one commit, exact tree after; whole batch refused on one refusal |
| `journals/src/view/since.rs` | lines and folds after T within a budget | exact lines returned for fixture trees at bookmarks before, inside, after folded periods |
| `motherd/src/journal_sink.rs` | `JournalSink` over zend's `Memory` as `Actor::Npcd` | a flush lands under `bots/<bot>` with npcd as writer |

Changed:

- `deliver_messages`, routing (§10);
- the private `message` (§10);
- `enact::message`/`reply` mark read, and `read_phone` shows the last own line (§11);
- the loop guard runs before perform and is rebuilt on load (§11);
- Discord nonces and per-part cursor (§11);
- `AddBot` grants `bots/<bot>` (§5.2);
- the mind schema's `talk` layer (§6.4);
- main-loop `Elsewhere` and `Concluded` events (§5.4, §7.2);
- `Availability::PersonNearby` / `PersonInEarshot`, `Within.people` /
  `heard_by_person`, the speech acts' gates, descriptions and examples (§12).
  Tests: the exact offered act set for NPC-only company, NPC-plus-visitor
  company, and a visitor in the next room within a shout's reach. Also the
  `needs` strings pulse shows (`pulse.rs:1574-1578`), and that `message` and
  `reply` are offered with NPC-only company and alone;
- the `promise`/`remind` thread availability, `reflect`'s journal line, the
  mission acts for an ordered body under an authorized party, and the phone
  acts marking read (§6.2). These apply in both loops, and are tested in each
  act's existing test module;
- `prompt::frame`, `MESSAGE` and the cooldown rationale, rewritten (§12).

Live check, after the tests: two moon NPCs and one Discord DM with Zen, with
these outcomes:

- every message is answered once;
- every turn leaves an outcome: a message, a journal line or a mission change;
- a `reflect` posts nothing and writes its line;
- an NPC↔NPC exchange settles by the cap, a `reflect` or a `sign_off`, each leaving a line;
- two NPCs in one room with no visitor exchange messages, never `tell`;
- a Commit lands on the mission and the main loop's next turn reflects it;
- `nvidia-smi` shows no slot held by a parked conversation.

---

## 14. Open questions

1. **The `bots` child cap.** A journal holds at most 10 live children
   (`journals/src/tree/journal.rs:17`), and an eleventh under `bots` folds the
   idlest bot's journal (`journals/src/scope.rs:130`). The moon household fits.
   A Battle Cities group with thousands of NPCs does not. Choose one:
   - lift the cap for `bots`, so the section's index lists only the most recent
     10;
   - nest bots by world, which still needs a cap change at scale.

   Recommendation: lift it for `bots`.
2. **Summary cadence for bots journals.** zend's memory writer summarises a
   journal after 20 new lines (`journals/src/brief/mod.rs:50`), at up to 4,256
   tokens each (`memory_writer.rs:83`). Across thousands of active bots that is
   the dominant model cost of this design. Recommendation: `bots` journals
   summarise on fold only. Folds stay as they are (≤ 304 tokens).
3. **Mood granularity.** Mood is one id, changed only by `reflect` or the
   conversation tail, with no decay (`record.rs:1353`). A numeric
   `Modulation{affect, threat, curiosity}` exists (`record.rs:1458-1480`), but
   nothing renders it into a prompt. Should conversations drive it?

---

## 15. Found along the way

These are outside this design. They are recorded here so they are not lost.

- **`report_done` skips the critical-step check.** The `report_done` handler has
  no critical-step check (`npcd/src/engine/work.rs:195-320`), so it may hand in
  a mission past an unsigned critical step. Found by reading, not tested.
- **`sleep.rs:22` describes code that doesn't exist.** It says a finished day is
  "folded into the memory layer"; no code does that.
- **Two stale comments:**
  - `mind.rs:1756-1758` says "two decodes, the narrator then the act". The
    narrator is a synchronous render now (`narrator.rs:25-30`).
  - The `finish_turn` doc (`candle-conversation/src/conversation.rs:3836-3845`)
    says it prefills the next header. It doesn't (`:4310-4314`).
- **`zen_task_act` keeps its handle on the replica.** On a replica it keeps the
  task handle local, so the authority never polls the task. (Known from the
  performance-patch review.)
