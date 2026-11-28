# NPC conversations

Every NPC — Zen, the moon household, a Battle Cities citizen — talks one-to-one
with people and with other NPCs in **conversations**: preserved, parked
exchanges that run beside the NPC's main loop, as the same person, with the same
mood, mission and memory. This document is the design, and all of it is built.
Code is cited by file and function, constant or type name, never by line
number: a name is found where it lives. Paths are under `npcd/src/` unless they
begin with a crate name (`journals/`, `candle-conversation/`, `zend/`, `agent/`,
`motherd/`, `npc-map/`).

---

## 1. Why

A DM to Zen on Discord (bot `128j9dsgheutk`, the moon world) once produced:

- the same reply posted two to five times;
- replies that contradict the one before;
- in-world chatter flooding briefing channels;
- an act template leaking into speech (`that …`);
- a DM thread that is Zen role-playing at nobody.

The causes were structural, not prompt-level. A DM was a phone thread the main
loop happened to notice. Each cause is removed by a mechanism below, and the
mechanism is what the rest of this document describes:

- **The text reaches the mind as a turn.** The main loop used to get a *notice*
  and had to choose `read`, then `reply`, between room events, idle percepts and
  its mission. `deliver_messages` (`engine/environment.rs`) now emits
  `EventKind::Talk{thread, lines}` at `Salience::URGENT` for lines from a person,
  carrying the lines themselves, and the conversation runner takes a turn on
  them (§9). `EventKind::Message{thread, from, unread}` remains for group and
  channel notices only.
- **What is posted is words, not the act's intent.** A direct-thread `message`
  hands a `Purpose` to the conversation (`engine/enact/messaging.rs`: `message`,
  `purpose`; `Sim::purposes`), and the conversation decodes the words. A line
  carries the voiced text, and Discord posts `m.text` (`discord/dm/outbound.rs`,
  `discord/server/outbound.rs`, `discord/world/outbound.rs`). `spoken` in
  `engine/reported.rs` has no callers. The `that …` leak is gone with it:
  `tools::narrates` (`engine/tools/show.rs`) still renders only `tell`, `ask`,
  `whisper`, `shout`, `gesture` and `act` for the feed, and `reply` is not among
  them.
- **Answering marks the thread read.** A conversation's answer sends
  `WorldOp::ReadThrough` (`authority/ops.rs`, handled in `authority/perform.rs`),
  which calls `Threads::read_through` (`sim/phone/mod.rs`). The URGENT unread
  reminder (`sim/unread.rs`: `Threads::unread_due`, every `REMIND_EVERY_MS` =
  5 min) covers group and channel threads only; a direct thread is a
  conversation's, never the mind's.
- **The phone shows the character its own replies.** `read_phone`
  (`engine/enact/reading.rs`) appends "(you last said, {ago}: …)" per thread.
- **The repeat guard runs before the post.** `talk::voice::post`
  (`engine/talk/voice.rs`) checks `would_repeat` (`engine/loopguard.rs`) on the
  voiced text against the thread's own last lines, and a conversation's guard
  state is derived from the thread (`engine/talk/guard.rs`), not held in memory
  that a restart loses.
- **Threads are stable.**
  - A direct thread's id is `dm:{lo}:{hi}` over the sorted pair of member ids
    (`sim/phone/direct.rs`: `direct_id`), so reaching someone again finds the
    same thread and its seq never restarts.
  - Direct threads are never evicted. `KEEP_GROUPS` = 64 (`sim/phone/channel.rs`)
    bounds the open groups characters make, and nothing else.
  - `sign_off` ends the conversation and leaves membership alone
    (`engine/enact/messaging.rs`: `sign_off`; `Threads::leave` in
    `sim/phone/channel.rs` refuses a direct thread).
  - Members are ids (`Who { id, name }`, `sim/phone/mod.rs`), and the display
    name is looked up to render.
- **The party's identity is kept.** `inbound::judge` (`discord/dm/inbound.rs`)
  writes `WorldOp::Message{from: Who::new(&handle, &speaking_as(..)), ..}`, the
  account's public handle plus a display name (`engine/voices.rs`: `speaking_as`).
  - Authorizers are `Authorizer{account, name}` (`engine/authorize.rs`), and
    `is_authorizer` matches on the account.
  - A critical step is signed off only through `engine/talk/critical.rs` and
    `engine/talk/authority.rs` (§7.2), never by an affirmative line read off the
    phone.
- **NPCs do not talk to each other aloud.** `tell` and `ask` are
  `Availability::PersonNearby`, `whisper` is `PersonAmongOthers`, `shout` is
  `PersonInEarshot` (`engine/tools/act_tools.rs`; `Availability` in
  `engine/tools/types.rs`). NPC↔NPC room speech (`World::tell` and `World::shout`
  in `npc-map/src/world/speech.rs`) is something nothing preserves and nothing
  concludes, so characters reach each other by `message` (§12).

A conversation removes these by construction: each message is a turn of a
preserved timeline, every turn decodes real words, and the party is known by
account. §11 describes the phone, and §12 makes room speech for people only.

---

## 2. Decisions

| # | Decision |
|---|---|
| D1 | A conversation is **the NPC's mind loop on a message clock**: same persona, same shared system prompt, same act catalog, same performer. The beat is replaced by the arrival of a message, and perception is narrowed to what changed since the last turn. |
| D2 | A conversation is **one-to-one**: an NPC and one party, which is a person or another NPC. Only a person's lines open a turn: a line from another character is told to the mind as something that happened (`EventKind::Description`), and the mind answers it or does not. A conversation with an NPC party is therefore started by a `Purpose` from the NPC's own mind, never by an incoming line. *(Replaced design: an earlier draft had a line from an NPC open a turn on the receiving side too. Two conversations answering each other never ended: Zen and a Drone traded the same two lines every forty minutes for a day, four decodes a round.)* Group channels stay on the main loop's fire-and-forget messaging (§11.2). So does NPC↔NPC room speech in front of a person (§12). |
| D3 | **Parked between turns.** A conversation holds no scheduler slot and no live `Sequence` between messages. It is a persisted substrate timeline that resumes on the next message. Thousands can exist, and only the ones decoding cost VRAM. |
| D4 | **Mood, mission, beliefs, persona and memory are shared** with the main loop. It is the same person in two places. |
| D5 | **Experience is journaled**, as an optional per-world setting in the one `journals` repository (§5.0). Each NPC's `bots/<bot>` journal gets lines for:<br>- what it hears and says;<br>- what it writes down in a conversation;<br>- its promises and mission changes;<br>- its notable world acts;<br>- the world changes that concern it.<br>The moon and Battle Cities both use it, each in its own group. |
| D6 | **Catch-up is a turn-scoped carried section plus a cited summary.** The journal lines since the conversation's last turn are carried into the turn, summarised with line citations into the user turn, kept for the reply, then removed. |
| D7 | **Every turn decides, then decodes.** A forked call chooses the acts from the existing catalog. Then a real turn always decodes free text, with no tools, under a response template that matches the choice. The choice decides **where the text goes**: to the party (`reply`, `sign_off`) or verbatim into the journal (anything else). Every turn leaves a message or a journal entry. A person's message may be answered by a journal entry; the NPC then marks it read, and Discord's typing indicator stops. |
| D8 | **One tool base.** The decision uses the main loop's catalog, compiled for a thread instead of a room (`Catalog::THREAD`), through an explicit allow-list (`ACTS` in `engine/talk/within.rs`): `reply`, `sign_off`, `reflect`, `promise`, `remind`, the `mission_*` acts, `message_someone_else` and `reach_out`. |
| D9 | **A main-loop private `message` (or `reach_out`) starts, resumes or injects into a conversation.** It hands the conversation a *purpose* (`engine/enact/messaging.rs`: `purpose`; `Sim::purposes`), and the conversation decodes the words. |
| D10 | **Conversations conclude.** A promise, a mission change or a hand-off is how a conversation changes what the NPC does. They are applied by the world's authority under authority, critical-step and thrash rules. Mission changes are open to the owner and authorizers only. |
| D11 | **Speech in the room needs a person in the room** (§12). NPCs reach each other by `message`. Battle Cities field play keeps `shout`. |
| D12 | **Every phone line is voiced on the lease holder** before it is sent to the authority. No intent text reaches a thread (§7.1, §11.4). |
| D13 | **Duplicates are tolerated.** An occasional repeated message is accepted, and the system is eventually consistent. A message that is never answered, or one timeline written from two members, is not accepted (§9). |
| D14 | **The party is known by account.** Authorizers, owners and thread members are account handles or body ids, never display names (§7.2, §11.1). |

---

## 3. What exists that this is built on

| Need | Existing piece | Anchor |
|---|---|---|
| Character decode | `Minds`, `Live{sequence}`, `send_turn_with_options_async` | `engine/mind/mod.rs` (`Minds`, `Live`); the sends are in `engine/mind/mind_schema.rs`, `think.rs` and `talk.rs` |
| Act grammar, parser, performer | `grammar_for`, `act::parse` (over specs), `performable`, `Runtime::perform` → `perform_said` → `perform_here` | `engine/mind/mind_schema.rs`; `engine/act.rs`; `engine/tools/compile.rs`; `authority/perform.rs` |
| Modes | `Mode::{Physical, InstantMessage}`, carried with the call count by `Catalog{mode, calls}` (`Catalog::ROOM`, `Catalog::THREAD`) | `engine/tools/types.rs` |
| Side decode that writes nothing | `UnsealedAsker::ask`, taking full `TurnOptions` (grammar, prefill) | `candle-conversation/src/conversation/ephemeral_turn.rs`; npcd's use in `engine/mind/ask.rs`, `talk.rs` and `voice.rs` |
| Response templates | the shared prompt's `response` collection (filled from `<mind>/responses/`) | the mind's `projection.yaml`; `projection.rs`; `engine/talk/templates.rs` |
| Runtime-offered candidates | `SelectionRule::Offered { selector, k }`, set per turn through `SelectionState` | `candle-conversation/src/projection/schema/selection.rs`; precedent `engine/tools/show.rs` (`show`, `show_within`) |
| Banning tokens in a turn | `TurnOptions.turn_banned_tokens` | `candle-conversation/src/turn.rs`; npcd's `Minds::banned_in_words` in `engine/mind/words.rs` |
| Carried sections per turn | `Carried::reconcile` | `candle-conversation/src/conversation/carried.rs` |
| Republishing the projection | `rebuild_projection` → `publish_ask_state` (crate-private) | `candle-conversation/src/conversation/submitted_sections.rs` (`rebuild_projection`, `publish_sections`); `conversation/turn_submitter.rs` (`publish_ask_state`) |
| Resume on a new slot | `resume_conversation_with_projection`; `restore_recurrent_state` | `candle-conversation/src/engine/creation.rs`; `candle-conversation/src/scheduler/recurrent.rs` |
| Idle KV demotion | `demote_idle_hot` | `candle-conversation/src/substrate/tiers.rs` (`Substrate::demote_idle_hot`); `scheduler/gap_residence.rs` (the pass) |
| Bands | `DecodePriority`; the timeline's `decode_band` metadata | `candle-conversation/src/projection/schema/layer.rs` (`DecodePriority`, `METADATA_KEY`); `scheduler/band.rs` (`rank`, `recorded_band`) |
| Leases and hand-back | rendezvous leases; `Standing::begin_turn`/`end_turn`/`hand_back_due`; `NpcLease::ready`; `wake_character`; `put_down` | `authority/npc_lease.rs`; `authority/worlds.rs`; `authority/cast.rs` |
| Cross-member delivery | `Told` pulses; `Runtime::tell` | `authority/told.rs`; `authority/follow.rs`; `engine/runtime/moments.rs` |
| Mood | `NpcPayload.mood`; `note_feeling` → `MoodSink` → `Npcs::remember_mood` | `candle-conversation/src/persistence/record/npc.rs`; `engine/runtime/acting.rs` (`note_feeling`); `engine/runtime/mod.rs` (`MoodSink`); `npcs/methods.rs` (`remember_mood`) |
| Missions | `Mission`, `Todo{critical}`, `check_off`, `critical_next`, `authorize`, `critical_compass`; `lodge_mission`; `cancel_mission`; agenda acts | `engine/mission/mod.rs` and `lifecycle.rs`; `engine/authorize.rs` (`critical_compass`); `authority/perform.rs` (`lodge_mission`); `engine/runtime/missions.rs` (`cancel_mission`); `engine/agenda/` |
| Ownership | `NpcPayload.owner_id` (a public handle `u_…`) | `candle-conversation/src/persistence/record/npc.rs`; `npcs/cast.rs` (`owned_by`) |
| Striking an act | `Within.cooling`, read by `performable` | `engine/tools/within_tools.rs`; `engine/tools/compile.rs` |
| Journals | `Store` (one private ref per checkout); `Section::Bots`; `Actor::Npcd`; `fold_period`; `Live`; zend's `Memory` wrapper and memory writer | `journals/src/store/mod.rs` (`Store`, `JOURNAL_REF`); `journals/src/scope.rs` (`Section`); `journals/src/actor.rs` (`Actor`); `journals/src/ops/period_fold.rs` (`fold_period`); `journals/src/live.rs` (`Live`); `zend/src/mcp/memory/mod.rs` (`Memory`, a wrapper over `Live`); `zend/src/session/memory_writer.rs` |
| A visitor's body | `visitor:{handle}` | `engine/interaction.rs` (`body_of`); `engine/order_table.rs`; `engine/talk/key.rs` (`VISITOR`) |

What this design adds on top of those pieces, all built:

- the conversation runner on the lease holder, and its delivery
  (`engine/talk/runner.rs`: `Talks`);
- the decide-then-voice turn (`engine/talk/decide.rs`, `engine/talk/turn.rs`,
  `Minds::talk_turn` in `engine/mind/talk.rs`), and the voicing of group and
  channel posts on the lease holder (`engine/talk/voice.rs`);
- the catalog compiled for a thread (`Within::on_thread` in
  `engine/talk/within.rs`, under `Catalog::THREAD`);
- the rest of `engine/talk/`: `key`, `state`, `clock`, `catchup`, `guard`,
  `authority`, `critical`, `orders`, `hand_off`, `journal` and `templates`;
- npcd writing journals (`npcd/Cargo.toml` depends on `journals` as well as
  `journal-core`; `engine/experience/`);
- typed experience on journal lines (`journals/src/experience.rs`);
- a journal read since a point (`journals/src/view/since.rs`);
- a non-consuming durability query (`snapshot_durable` on the resolver's
  `Conversation`, `candle-conversation/src/projection/resolver/recurrent_state.rs`,
  reached through `Sequence::sealed_state_durable` in
  `conversation/sequence_state.rs`);
- the rebuilt phone (§11);
- accounts carried through messages and authority;
- the speech gate (§12).

---

## 4. The model

### 4.1 Identity and generations

A conversation is keyed by `(npc, party)`:

- `Party::Person(account)` — the npcd account a Discord user is linked to
  (`discord/link/index.rs`), a console user, or a Battle Cities
  player (§12), by public handle;
- `Party::Npc(npc_id)`.

Both are in `engine/talk/key.rs`. Its conversation id is
`npc-{npc}-talk-{party}-g{n}` (`key::timeline`), where `n` is the generation,
counted from `g0`, and `{party}` is a slug: a person's public handle with `-`
replaced by `_` (`npc-7-talk-u_8812-g0`), or `npc{id}` for an NPC party. Any
member holding the character's lease rejoins the newest generation (§9). It
does not roll over with the world day or chapter. `retire_superseded`
(`engine/mind/conversation.rs`) sweeps only `npc-{id}-day-` and
`narrator-{id}-day-`.

- **A conversation is resumed only under the frame it was decoded under.** npcd
  rejoins a timeline only when the rendered frame's fingerprint matches
  (`resumable` in `engine/mind/conversation.rs`, matching `META_FRAME` against
  `frame_fingerprint`), because a history the character can no longer produce is
  worse than none. A conversation's frame is the shared prompt rendered under
  the `talking` stance (§6.5).
- **When the frame changes**, the next message opens generation `n+1`. The
  `talk.*` state is carried over, and generations below `n+1` are retired by
  prefix, as `retire_superseded` does. History before the change is recalled
  from the journal (§5). This is why every conversation's content also lands in
  the journal.
- **Lifecycle.** `retire_npc` (`engine/mind/conversation.rs`, through
  `retire_talks` in `engine/mind/talk.rs`) also tombstones every
  `npc-{npc}-talk-*` timeline when an NPC is deleted.
- **NPC↔NPC** is a conversation in the mind of the NPC whose purpose started
  it, over the one direct thread the pair shares (§11.1). The other side's lines
  reach its mind as something that happened (D2), not as a turn of a
  conversation of its own.
- **The console's chat.** The console's `InstantMessage` interactions become
  conversations too, so there is one path: a line typed in one is a
  `WorldOp::Message` on the person's thread by their account handle, and a
  slash command stays the operator's (`engine/interactions.rs`: `inject`).

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
| `talk.journal` | the conversation's own journal record (§5.5) |

The keys are the constants of `engine/talk/state.rs` (`talk.journal` is
`META_JOURNAL` in `engine/talk/journal.rs`). The bookmark is real time, because
journal lines are stamped in real time (`Stamp::now()` in `Live::write`,
`journals/src/live.rs`). World time can be paused, jumped or run fast (the
"Real time" section of `engine/cooldown.rs`; the bug that mixing the two caused
is recorded on `last_ms` in `engine/interaction.rs`). World time is shown in the
user turn separately.

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
repository (`worlds/<id>.yaml`, the `world` registry; the key is read by
`journals_group` in `tenant/worlds_and_identities.rs` and applied in
`host_worlds`, which calls `runtime.journals.set_group`):

```yaml
journals:
  group: moon          # the journals group this world's NPCs belong to
```

A world with no `journals:` has no journal memory. Its conversations still
run, and catch-up reads nothing. Worlds join a group of the **one** `journals`
repository, named for the world by convention: `moon` and `battle-cities`.

- **One group per world.** Every path npcd writes for a world is under
  `groups/<group>/`.
- **Keeping Battle Cities out of personal frames.** `Op::CreateGroup` makes its
  creator the first owner (`create_group` in `journals/src/ops/group.rs`), and a
  member's frame shows every group they belong to. The `battle-cities` group is
  therefore created by the `zen` service person (`service:zen` resolves to the
  person `zen` in `zend/src/mcp/memory/who.rs`), so it is in no person's frame
  unless they join it.

**The live store is in the journals crate.** `journals::live::Live`
(`journals/src/live.rs`) holds the store mutex, the published snapshot, and the
sync thread that pushes and fetches every `SYNC_EVERY` = 5 s. zend's `Memory`
(`zend/src/mcp/memory/mod.rs`) and npcd's `Journals`
(`engine/experience/store.rs`) each hold one `Arc<Live>`, so the sync logic
exists once.

**npcd keeps its own clone.**
- **Why.** A store commits on the private ref `refs/journal/head`
  (`JOURNAL_REF` in `journals/src/store/mod.rs`), so two stores cannot share
  zend's checkout.
- **The clone.** npcd's is an agent checkout, `journals-npcd`
  (`motherd/agent.yaml`), with the same remote and `push: true`, but not
  `journal: true`. That flag is zend's memory, and the agent allows one
  (`CheckoutConfig.journal`, enforced in `AgentConfig::parse` in
  `agent/src/config.rs`). It is named in motherd.yaml's `npcd:` block as
  `journals_dir:` (`NpcdArgs.journals_dir` in `tenant/mod.rs`, started in
  `tenant/tenant_start.rs`), which is deployment config.
- **Two stores, one origin.** The two stores meet through origin as two
  members' stores do: a lease push, and op-level replay onto an origin that
  moved (`Store::sync` and `replay_onto` in `journals/src/store/mod.rs`;
  `Transfer::run` in `journals/src/store/sync.rs`).
- **What this avoids.** npcd needs no in-process handle on zend, which motherd
  does not offer (`motherd/src/main.rs`: the `Seat` runs and asks zend and npcd
  as separate services), and it works on members that run npcd without zend.

**zend folds the repository**, `bots/` included, with its own budget for
`bots/` (§5.3).

### 5.1 What is written

Per NPC, into `groups/<group>/bots/<bot>/`, as `Kind::Episode` lines
(`Kind` in `journals/src/line.rs`).

**Experience is a typed field.** `Line.terms` cannot carry it: `Terms`
(`journals/src/terms.rs`) is a fixed struct, and only a note may state terms
(`Line::check` in `journals/src/line.rs`). So `Line` carries
`experience: Option<Experience { kind: ExperienceKind, about: Option<JournalPath> }>`
(`journals/src/experience.rs`), which is valid on episodes only: `Line::check`
refuses it on any other kind.
- It is rendered into the line's file and read back, with tests asserting the
  raw bytes (`an_experience_is_written_exactly_and_read_back` in `line.rs`).
- It is included in `lines_over` in `journals/src/ops/period_fold.rs`, so a
  fold conflicts when it changes.
- `about` names the party's journal. npcd may not `Op::Link`, which is a
  person's AI's alone and makes a sub-journal, not a tag
  (`journals/src/ops/permit.rs`).

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
  the repository in memory (`Store.files`) and reloads the tree on replay
  (`replay_onto`), both in `journals/src/store/mod.rs`. With two clones per
  machine, volume is the scaling limit.
- **Daily cap.** Lines are capped at `DAY_LINES` = 200 per bot per UTC day
  (`engine/experience/buffer.rs`). When over, the oldest `did` and `heard`
  lines of the day are dropped first, and a new `did` or `heard` line with none
  of the day's to displace is itself dropped.
- **Folds absorb the rest** (§5.3).

#### 5.1.1 World changes that concern the NPC

A conversation knows only what the journal tells it. So a world change that the
NPC did not make, and was not in the room for, but which touches it or its
party, is a line too.

One function, `experience::concern` (`engine/experience/concern.rs`), takes a
change (`Change::{Mission, Room, Party, World}`) and the sim, and returns the
journals that should hear of it (`Hears::{Bot, World}`):

| A change concerns… | when it… | Source |
|---|---|---|
| the NPC | names it, or changes its mission | a mission lodged, assigned, withdrawn, stepped or authorized (`WorldOp::{Lodge, Step, Cancel, Authorize}` in `authority/ops.rs`, applied in `authority/perform.rs` through `lodge_mission` and `Runtime::changed`; `cancel_mission` in `engine/runtime/missions.rs`; `Mission::authorize`) |
| the NPC | touches the room its mission is in (a standing mission's `where`, `Standing` in `engine/mission_gen/standing.rs`) | building ops (`build_act` in `authority/building.rs`); a bridge act by someone else on a device in that room (`bridge_act` in `authority/building.rs`; `enact` in `bridge/mod.rs`) |
| the NPC | happens to a party it has a conversation with | a mission that person lodged changes; that person's visitor body arrives or leaves |
| the world | is world- or house-wide | a house mode set (`house_act` in `authority/house.rs`); the day rolling over; an operator reshape |

- **World-wide changes are written once**, into `groups/<group>/bots/world/`.
  Catch-up reads that journal alongside the bot's own.
- **House modes** are set only by an NPC's house act, which is that NPC's `did`
  line plus one world line ("Zen set the house to away mode, as John asked").
- **Reshape goes through the authority.** `/reshape` called `hosted.reshape` on
  whichever member received it. It is `WorldOp::Reshape { npc, edit: MapEdit }`
  (`authority/ops.rs`), applied by the authority (`reshape_here` in
  `effector/reshape.rs`, which emits the `Change::World`), so it is journaled
  once, and answered as `OpReply::Reshaped`.
- **The day rolling over** is told once by the member that writes the world,
  on the first moment of a new world day (`Days::rolled` and `day_line` in
  `engine/experience/days.rs`); the first sight of a world only marks its day.
- **Acts that are not a body's own** — a device, a bridge, the house,
  building — write their `did` line where they land, beside a room line for
  whoever's standing mission is for that room, or the world line for a house
  mode (`Runtime::landed_elsewhere` in `authority/perform.rs`).
- **Where `concern` runs.** It runs where the change is applied (the
  authority: the `WorldOp` arms of `authority/perform.rs`, through
  `Runtime::changed` in `engine/runtime/moments.rs`), right after the change. No
  line is deduplicated across members, and D13 accepts an occasional repeat.
- **What is never written.** Ambient percepts, rooms that did not change, and
  the comings and goings of characters it has no conversation with. These stay
  in the world log, where the main loop's per-beat perception reads them.

### 5.2 How it is written

**Buffered per bot, one op per bot per flush.**
- **The buffer.** A line is pushed onto the member's per-world `Buffer`
  (`engine/experience/buffer.rs`): one mutex over a map from bot path to that
  bot's pending lines, swapped out whole (`take`) at flush. The beat path takes
  the lock only for a push.
- **The flush.** Every `FLUSH_EVERY` = 5 s (`engine/experience/store.rs`), each
  bot with lines gets **one** `Op::Episodes { lines }` commit, on the
  `Op::Activity` model of one commit per subject (`write_one` in
  `zend/src/mcp/noted_calls.rs`). A refusal touches only that bot's batch.
- **Each line keeps its own `at`** (stamped when it arose), not the flush
  time: `Lived { at, text, experience }` in `journals/src/ops/episodes.rs`, and
  `episodes()` writes `at: lived.at`.
- **Line ids survive replay.** `Applied` carries the list of line ids
  (`Applied.lines`, `journals/src/ops/mod.rs`), with one `Journal-Line`
  trailer per line (`message` in `journals/src/store/message.rs`). Replay
  renames each line that collides (`replay_onto` in `journals/src/store/mod.rs`,
  with `remap` in `journals/src/ops/remap.rs`), so a catch-up's `#id` citations
  stay correct.
- **Wire codec.** `Op::Episodes` and the `npcd` actor have one
  (`journals/src/ops/wire.rs`).
- **Permits.** `Op::Episodes` is permitted only to `Actor::Npcd`, and only on a
  bot journal directly under `bots/` (`journals/src/ops/permit.rs`).

**The instruction guard.** Each line's text goes through `refuse_instructions`
(`journals/src/ops/guard.rs`), which matches its phrases anywhere in the text.
A line that fails is dropped and counted in `Journals::flush`
(`engine/experience/store.rs`), never taking its batch with it. A person's DM
that trips the guard therefore leaves no `heard` line, though the conversation
still answers it.

**Passing failures.** On a passing failure the batch is kept and merged ahead of
newer lines (`Buffer::restore`, as `NotedCalls::keep` does in zend). A refusal
drops it. Each bot's buffer is capped at `MAX_EPISODES` = 200 lines
(`journals/src/ops/episodes.rs`), newest kept.

**The actor is `Actor::Npcd { group }`** (`journals/src/actor.rs`). The
journals crate knows groups, not worlds.
- **Why no person.** An actor keyed by a person would need a writing person in
  the group, and a bot owner's person would become the line's `account`, able to
  edit or forget bot lines.
- **Reads** go through its own arm in `need_read` (`npcd_may_read`): `bots/` and
  `people/` of its group. It reads nothing else: not `conversations/`, not
  `threads/`, not `topics/`. An NPC must never read a person's AI conversations
  and repeat them to another party. The catch-up reads only the party's own
  journal among `people/` (§6.1).
- **Writes** are `Op::Episodes` and `Op::Create` under `bots/` of its group.
- **Commits** carry no `Journal-Account`. They carry `Journal-Writer: npcd`
  and `Journal-Group: <group>` (`Actor::trailers`), and `who()`
  (`journals/src/view/who.rs`) renders "npcd" for a line with no account.

**Creating the journal is idempotent.** A bot's first line may arrive on two
members at once (the lease holder's `heard`, the authority's `changed`).
`Op::Create` uses `OnTaken::Refuse`, and "exists already" counts as success
(`Journals::write` in `engine/experience/store.rs` re-checks), so `bots/<bot>.2`
can never be minted (`create` in `journals/src/ops/create.rs`). The create is
its own op, issued in the same flush just before the first batch.

**Which member writes.**
- `heard`, `said`, `wrote`, `promised` and `did` lines: the character's lease
  holder.
- `changed` and `mission` lines: the authority.

Both write to npcd stores on clones of the one repository. Within a member,
catch-up merges the unflushed buffer (§6.1). Across members, the bound is a
flush plus a sync (about 10 s), which D13 accepts.

### 5.3 The time ladder, and its cost

- **Folding.** zend's memory writer folds a window of more than `WINDOW` = 50
  lines (`journals/src/tree/journal.rs`) into `Kind::Fold` lines by closed
  period, Day→Week→Month→Year (`next_fold` in `journals/src/tree/window.rs`,
  with `Level` in `journals/src/period.rs`; `fold_period` in
  `journals/src/ops/period_fold.rs`). A fold is one model decode of at most
  `FOLD_TOKENS` = 304 tokens (`FIELD_TOKENS` 256 plus `FRAME_TOKENS` 48) at Low
  priority (`zend/src/session/memory_writer.rs`).
- **`bots/` folds have their own allowance.** They take `BOTS_PER_PASS` = 16 jobs
  per 30 s pass (`EVERY`), beside the existing `PER_PASS` = 8 (`this_pass` in
  `zend/src/session/memory_writer.rs` partitions the jobs by section), so bot
  upkeep never starves people's folds and summaries.
- **A busy day.** A period folds only once it has closed (`fold_period` refuses
  an open period), so a busy day is read in full until it ends. Catch-up's
  budget bounds what is read (§6.1).
- **`bots/` summaries happen on fold only.** `bots/` is exempt from every
  summary trigger: quiet for 30 s, quiet for 10 min as a parent, and 20 new
  lines (`QUIET_SECS`, `PARENT_QUIET_SECS`, `NEW_LINES`; `due` in
  `journals/src/brief/mod.rs`). A bot summary is refreshed when its window
  folds, or when a child's summary is newer. Catch-up reads lines and folds,
  never a summary.
- **The child cap.** `bots/` lifts the 10-child cap (`MAX_CHILDREN` in
  `journals/src/tree/journal.rs`) and the eleventh-child fold for the journals
  directly under it: `Section::eleventh` (`journals/src/scope.rs`) answers
  `Unbounded` for `bots/` at depth 0, and the check in `journals/src/tree/check.rs`
  skips a journal for which `caps_children` is false. Journals nested below a
  bot keep the cap. The section's index lists the 10 most recently active bots
  (`Journal::listed`), and `bots/world` is never folded away because nothing
  under an unbounded section is.

### 5.4 What the main loop sees of its conversations

The main loop does not read its conversations' timelines. Each main-loop turn
gets one `Elsewhere` event, rendered rather than decoded. It lists the `said`,
`wrote`, `promised` and `mission` lines its conversations wrote since the main
loop's previous turn, read by the same `since` read as §6.1. Zen walking the
house knows it just promised Mira the parcel.

`Elsewhere` wakes nothing: a conversation that woke the mind on every line would
have it answering a call it is not on. **A turn that settles something — a
promise, or a mission change — tells it as `Concluded` instead, which does**
(D10): that is how a conversation changes what the character does, and only the
mind does it. Told quietly, it waited for whatever woke the mind next, and for
a character standing in a quiet room that was nothing.

### 5.5 A conversation's own journal

A conversation with a person keeps a journal of its own, as a zend dialogue
does (`docs/zend_journal.md`), written by the same workflow
(`journal_core::workflow::run`) — `npcd/src/engine/talk/journal.rs`,
`Minds::talk_journal`.

- **When.** After a turn a person spoke on, once the reply has gone: the party
  never waits on it. A turn the character began itself is not journaled — it
  is the mind's own doing, and the mind already knows it.
- **The stretch.** Each line the person said, citable as *heard*, then what
  the character replied, citable as its own act (`journal::stretch`). The
  numbers a claim cites are the thread's seqs, so they mean the same thing
  after a restart.
- **The draft.** Asked of the conversation on an ephemeral fork of itself
  under `AskScope::Conversation`, writing nothing back, at the conversation's
  own band: it is what carries the person's request to the mind, and the
  conversation is held until it is written. Asked at `Normal`, where other
  journal drafts run, it waited ten minutes behind the character's own
  reflection.
  One answer writes an entry — a one-sentence summary, claims that cite, what
  the character means to do next, what it left open — or, once the journal
  holds an entry to weigh the stretch against, skips it.
- **No facts.** What somebody said is nothing the world can read back, so the
  conversation's style offers no fact slot (`journal::TALK`), as zend's does
  not.
- **Where it is kept.** The record is in the conversation's own metadata
  (`talk.journal`), beside the rest of its state: it follows the conversation
  and is nobody else's.
- **A kept entry is told to the mind, and wakes it.** It goes as `Concluded`,
  with the entry in full and nothing beside it — the entry already says what
  was asked, what was answered and what is meant to be done, so the line said
  and the acts taken are not repeated next to it. That is how what was said on the phone becomes
  something the character does: "John asked whether the cluster is up; I said
  I would look" reaches the mind with what it now means to do and what is
  open. A skipped stretch tells nothing more than §5.4's quiet `Elsewhere`.
- **A telling wakes the mind as soon as it is said.** When the turn spoke for
  the mind (the main loop's `message` to this party) rather than answered
  anybody, the mind is told `Concluded` before the journal is drafted: the
  words, that saying them again tells the party nothing, and that what it said
  it would do is still its to do. A telling hears nothing, so its stretch often
  keeps no entry, and without this a mind that said "I will survey the devices
  now" was woken by nothing and never did. An entry that is kept for a telling
  follows quietly, read on the turn that wake starts.

---

## 6. A turn

### 6.1 Catch-up

1. **Read since the bookmark.** `journals::view::since(tree, actor, paths, after, seen, budget)`
   (`journals/src/view/since.rs`) returns `Since { lines, left_out }`: lines and
   folds, oldest first, fitted to a character budget.
   - **The overlap window.** It reads from `after − OVERLAP`, where
     `OVERLAP` = 60 s, and drops line ids in `seen`. A line stamped before the
     bookmark but arriving after it, from another member's flush and sync, is
     therefore still read once. The ids read within the window become the next
     `talk.seen`.
   - **Archives and folds.** It walks each journal's archives
     (`Journal.archives`, keyed by period) for periods that close after the
     window starts, then the window filtered by `at`. A fold whose whole period
     lies inside stands in for its archived lines when the originals do not
     fit.
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
     12,000 (`engine/talk/catchup.rs`), which sits between the frame's
     `Budget::Chat` (6,000) and `Budget::Worker` (32,000) in
     `journals/src/view/frame.rs`.
2. **Carry the lines as one section.**
   - **What is carried.** All lines, each with its `#id`, become one section,
     `catchup/now`, of a `catchup` collection. It is reconciled with
     `Carried::reconcile(.., Linking::Apart, sweep: ["catchup/now"], wait: Some(..))`:
     one prefill, one owned-section id and one tombstone per turn.
   - **Why not a section per line.** That would cost N chained prefills in
     series and N ids from a 2^24 partition that is never reused
     (`submit_section` and the `RestoreSection` send in
     `candle-conversation/src/conversation/submitted_sections.rs`; `SLOTS` =
     `1 << 24` in `candle-conversation/src/projection/owned_sections.rs`).
   - **Declared in the shared prompt.** `catchup` is declared in the mind's
     `projection.yaml` with a named selector, unset for the main loop, because
     `submit_section` refuses a collection the schema does not declare. The
     name is `catchup::COLLECTION` (`engine/talk/catchup.rs`).
   - **Selection.** The turn's selection names it, as `identity::carry_journal`
     names the journal (`engine/identity.rs`; `Minds::talk_selection` in
     `engine/mind/talk.rs` passes `Linking::Apart`, the sweep
     `[catchup::SECTION]` and `Some(SEAL_WAIT)`).
   - **Sweeping on resume.** Every resume builds a fresh `Carried::new(false)`
     for the catch-up, so its first reconcile sweeps `catchup/now`
     (`Carried::reconcile` in `candle-conversation/src/conversation/carried.rs`).
     A crash between the reply and the release leaves nothing behind.
3. **Publish before asking.** A section sealed by `reconcile` is not yet in the
   projection `UnsealedAsker::ask` uses. That projection is republished only at
   submit or on a removal (`sync_submitted_sections` and `rebuild_projection` in
   `submitted_sections.rs`).
   - **The method.** `Sequence::publish_sections()` runs
     `sync_submitted_sections` and publishes the ask state
     (`publish_sections` in `submitted_sections.rs`; `publish_ask_state` in
     `conversation/turn_submitter.rs`).
   - **When it runs.** The runner calls it after the reconcile, before any ask
     (`Minds::talk_turn` in `engine/mind/talk.rs`, right after
     `talk_selection`).
4. **Summarise when the gap is long.** If more than `VERBATIM_LINES` = 6 lines
   were read, an ask runs under `AskScope::Conversation` at the conversation's
   band: "What has happened since you last spoke with {party}, {gap} ago, that
   matters to this conversation? Cite each line you use by its #id"
   (`catchup::summary_question`). It uses the checkpoint's no-think sampling
   (`sampling_for(Deliberation::None)`), no thinking, and the answer is capped
   at `SUMMARY_TOKENS` = 192. With 6 lines or fewer, the lines are rendered
   verbatim and nothing is decoded.
5. **Release** after the voice turn has finished and `turn_in_flight` has
   cleared: reconcile with an empty `catchup` want list.

### 6.2 Decide (a fork)

The decision is the mind's act turn, run on an ephemeral fork with
`UnsealedAsker::ask`, which writes nothing (`candle-conversation/src/conversation/ephemeral_turn.rs`).
- **Scope and band.** It uses `AskScope::Whole`, at the conversation's band
  (`Urgent` for a person, `Low` for an NPC: `talk_band` in `engine/talk/turn.rs`).
- **Its user text** is the user turn of §6.3 step 1. That is the text the voice
  turn will seal, without step 2's restatement.
- **Afterwards.** The fork is parsed and dropped.
- **A telling is not decided.** When the mind itself began the turn (the main
  loop's `message` to this party, handed over as a `Purpose`), the fork is
  skipped and a `reply` carrying the purpose's intent is the decision
  (`telling` in `Minds::talk_turn`, `engine/mind/talk.rs`).

**The catalog, compiled for a thread.**

- **`Catalog` is a grammar input.** `Catalog { mode, calls }`
  (`engine/tools/types.rs`) is `Catalog::ROOM` (`Mode::Physical`, `ACTS_PER_TURN`
  calls) for the main loop and `Catalog::THREAD` (`Mode::InstantMessage`,
  `TALK_ACTS` calls) for a conversation. It is passed through `grammar_for` and
  `act_specs` (`engine/mind/mind_schema.rs`), `show_within`
  (`engine/tools/show.rs`) and the grammar cache key `(thinking, catalog,
  within)`.
  - `act::parse` takes specs, so it follows without change.
- **An explicit allow-list.** `Within::on_thread(&OnThread)`
  (`engine/talk/within.rs`) sets `allow` to `ACTS`, the only acts offered.
  `OnThread` carries the party, its name, what is owed, whether the party may
  change the mission, the carried state, the activation cap and the cooling.
  - **Why emptying the world sets is not enough.** `sleep`, `record_verdict`,
    `read`, `invite`, `open_group` and `send_image` survive it (their
    `Availability` in `engine/tools/types.rs`; their declarations in
    `engine/acts/` and `engine/tools/act_tools.rs`). The empty-set rule removes
    only acts with a required world-bound parameter (`performable` and
    `live_empty` in `engine/tools/compile.rs`).
  - **Where it is checked.** `allow` is checked in `performable` beside
    `cooling`, which both `specs_within` and `invoke_body` share, so both
    grammars honour it. `Availability::OnThread` offers an act only when `allow`
    is set.
- **The path budget.** `estimated_paths(specs, calls)` (`engine/tools/compile.rs`)
  raises the per-act total to the catalog's call count (`ACTS_PER_TURN` for a
  room, `TALK_ACTS` for a thread) under `MAX_TURN_PATHS` = 2,000,000, and a
  budget test covers the full thread catalog at `TALK_ACTS`.

| Act | In a conversation | Binding |
|---|---|---|
| `reply` | the voiced text goes to the party | `to` = the party, **whoever spoke last** (`Choices::Awaiting` in `engine/tools/choices.rs`, resolved from `within.awaiting` in `engine/tools/offers.rs`: the talk `Within` holds the party, where the main loop's needs somebody else to have spoken last) |
| `sign_off` | the voiced text is the parting; ENDED | `to` = the party; it ends the conversation and keeps membership (§11.1) |
| `reflect` | the voiced text goes to the journal; the mood is set | `feeling` ∈ `Choices::Feelings` (`engine/tools/choices.rs`). In a conversation the performer calls `note_feeling` only (`engine/runtime/acting.rs`): `arm_pause`, which starts a reflection and a dream, is called by the main loop's turn (`engine/runtime/turn.rs`) alone. |
| `promise` | a ledger promise to the party | `to` = the party; `what`, `by` |
| `remind` | remind the party of their promise | offered only when the party is an NPC (`talk/within.rs` drops it for a person, who never `promise`s). `which` ∈ what the party owes. |
| `mission_*` | change the mission (§7.1) | `mission_add`, `mission_edit`, `mission_cancel`, `mission_done`: offered only when the party `may_change_mission` (the owner or an authorizer, outside the thrash cooldown) |
| `invoke` at `order_post` | post work on the order table, as the character's own (§7.1) | offered only to a character that gives orders and can reach the table, when the party `may_post` (its owner or an authorizer), and at the table's `order_post` address alone, with the world's standing jobs and what each brings (`talk/within.rs`, `POSTING`). Checked again where it lands (`Runtime::talk_post_refused`, `authority/perform.rs`), and told to the main loop as the table answered it, so the mind oversees what its conversation posted. The post names the party (`AskedBy`), and its report comes back to this conversation to tell them (`docs/zen_moon_mind.md` §8.2). A post the table refuses is told to the mind in the table's words and wakes it (`talk/turn.rs` `refused_post`): the conversation has decided by then and never reads the refusal, so without it the request was dropped with a word to nobody. A conversation that could only talk once told the Creator "I have instructed the Drones"; the mind that could post took the work as its own and ran it on the cluster's terminal while nothing was ever on the table. |
| `message_someone_else` / `reach_out` | hand off: a purpose to another conversation (§10) — not the answer | `to` ∈ the NPC's threads, channels and contacts **other than the party**. `message_someone_else` is the conversation's own act (`Availability::OnThread`, `talk/hand_off.rs`), performed as the world's `message`; the main loop keeps `message` and is never offered it. |

- **Calls per turn.** `compile_action_loop` (`candle-conversation/src/stencil/tool_call/call_tree.rs`)
  with `max_calls` = the catalog's `calls`, which is `TALK_ACTS` = 2 for a thread
  (`engine/tools/compile.rs`): a promise or a mission change, and the reply
  beside it. This is its own constant beside the main loop's `ACTS_PER_TURN` =
  2. It is two, not three, because of the path budget: `reflect` binds the
  whole mood library, so with the shipped mind's 116 moods and an agenda to
  change one act is about 300 paths. Three acts admit 2.6 × 10⁷ decodes,
  twelve times `MAX_TURN_PATHS`; two admit under 10⁵
  (`tools::tests_grammar::a_full_thread_stays_inside_the_path_budget`).
- **A hand-off does not end the decision.** A decision whose acts include a
  hand-off and none of `reply`, `sign_off` or `reflect` is asked again on a
  fresh fork, with what it handed off struck from the offered acts and told
  what it sent and that the party still waits (`decide::handed_off_only`,
  `decide::after_hand_off`). `reflect` is struck with it
  (`hand_off::spent_with`): the party is waiting, so the round answers them —
  left offered, Zen asked about the weather posted "I need to check the
  weather" to the channel, reflected, and the person who asked heard nothing.
  Each round strikes at least one hand-off, so the rounds end once they are
  spent. The acts of every round are performed, so the character messages
  somebody else and then answers the party. A hand-off is for what somebody
  else must hear or do; what the character can do itself it says it will do,
  and does once the call is over (§6.3). A character that keeps its own
  agenda — one that takes no orders, as Zen — is offered no hand-off at all
  on a person's thread (`talk::within`, `passes_on`): its mind tells whoever
  else must hear once the call is over. From the phone Zen handed every
  request to "the channel" ("I need to calculate…, check the time…, get the
  weather"), where it reached the Drones as noise.
- **The destination.** The loop cannot make one call terminal
  (`compile_action_loop`), so the destination is read after parsing
  (`decide::destination`): the **last** of `reply`, `sign_off` or `reflect` in
  the list decides. A list with none of them goes to the journal. A fourth
  destination, `Destination::Channel`, is a post on a channel or group from the
  main loop's own turn (§11.4).
- **Deliberation.** The decision uses `Deliberation::None`, opened by a closed
  think block and the call marker (`opening` in `engine/mind/mind_schema.rs`,
  which prefills only when a grammar was compiled).

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
   - for a sign-off: "You are saying goodbye to {party}: {intent}.";
   - for the journal: "Write down what you make of this; you are not answering."
     (`decide::restated`).
3. **Decode.** `send_turn_with_options_async` with no grammar and no triggers
   (`Minds::talk_turn`, `engine/mind/talk.rs`).
   - Thinking is off: the no-think selector is set, and both `<think>` and
     `<tool_call>` are banned (`turn_banned_tokens`, from
     `Minds::banned_in_words`). With no grammar, `opening` (`engine/mind/mind_schema.rs`)
     prefills nothing, so the bans are what hold.
   - `max_tokens` is `LINE_TOKENS` = 320 for a message, which fits one Discord
     message, and `JOURNAL_TOKENS` = 480 for a journal entry.
4. **Template.**
   - **One collection for both loops.** The shared prompt's `response`
     collection is `Offered { selector: "response", k: 1 }` (`install` in
     `engine/talk/templates.rs`; `SelectionRule::Offered` in
     `candle-conversation/src/projection/schema/selection.rs`). `Offered`
     selects explicit member names, so each loop lists its own set:
     - the main loop offers every non-talk template, which with provenance
       ranking gives exactly the former `top_k: 1` over the library;
     - a talk turn offers only its destination's templates
       (`Installed::offer`): `talk/reply/*`, `talk/parting/*`, `talk/journal/*`
       or `talk/channel/*`, from files under `responses/talk/<destination>/`.
   - **Why one collection.** Whichever template is shown always matches what
     the text is for, and no second collection has to be kept off the main
     loop.
5. **Where the text goes:**

| Destination | The decoded text becomes | Then |
|---|---|---|
| `reply` | a line on the thread to the party (`said`), carried to the authority in `WorldOp::Act.said` (§7.1) | PARKED |
| `sign_off` | the parting line on the thread | ENDED |
| journal | a `wrote` line in `bots/<bot>`, verbatim. The thread is marked read through the input line on the authority (`WorldOp::ReadThrough`), so Discord's typing indicator stops (§8). | PARKED |

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
  `Sequence`, which frees the slot (`Minds::park_talk` in `engine/mind/talk.rs`:
  it polls `sealed_state_durable` every 50 ms, up to `DURABLE_WAIT` = 30 s, and
  then parks regardless, with a warning).
  - **The query.** The wait is a non-consuming per-timeline query,
    `snapshot_durable(timeline, turn)`, built from the snapshot's location and
    `sealed.is_durable` (`candle-conversation/src/projection/resolver/recurrent_state.rs`,
    on the resolver's `Conversation`), reached through
    `Sequence::sealed_state_durable` in `conversation/sequence_state.rs`. It is
    true at once for a model with no recurrent state or no sealed turn.
  - **Why not the existing call.** `drain_durable_snapshots` is a global
    read-once queue whose only consumer is the scheduler's own idle parking
    (`candle-conversation/src/scheduler/recurrent_park.rs`). Draining it
    elsewhere would stop slots from parking.
  - **Why wait.** A resume on a new slot restores recurrent state through
    `restore_recurrent_state` (`candle-conversation/src/scheduler/recurrent.rs`)
    from whatever snapshot the index holds, rejecting only one that is newer than
    the history. Dropping before durability could silently resume on the
    previous turn's state.
  - **Latency.** The wait comes after the reply is posted, so it is off the
    person's critical path.
- **KV.** The turn's KV was truncated at seal (`truncate_sequence_to_blocks` in
  `candle-conversation/src/scheduler/seal_turn.rs`). Its residences stay hot
  until `demote_idle_hot`'s grace: `IDLE_DEMOTE_GRACE_EPOCHS` = 16 epochs
  (`candle-conversation/src/substrate/mod.rs`; `Substrate::demote_idle_hot` in
  `substrate/tiers.rs`; the pass in `scheduler/gap_residence.rs`), ticking on
  seal or flush passes or a 5 s idle tick (`DEFAULT_TICK` in
  `candle-conversation/src/persistence/thread/mod.rs`). Abandoned conversations
  drain to warm and cold through the existing tiering.
- **Never transient, never unresumed.** A conversation timeline is never marked
  transient (`mark_timeline_transient` in
  `candle-conversation/src/projection/resolver/lifecycle.rs`) or unresumed
  (`candle-conversation/src/scheduler/recurrent_park.rs`).
- **Resume.** `resume_conversation_with_projection`
  (`candle-conversation/src/engine/creation.rs`), under the frame rule of §4.1.
  A resumed `Sequence` reconciles its carried collections afresh (the `Talk`
  open in `engine/mind/talk.rs`: `Carried::new(fresh)` for the mission and
  journal sections, `Carried::new(false)` for the catch-up): the mission section
  and the journal-core chain are restored on every turn after a park, one
  `RestoreSection` each (`submitted_sections.rs`). That is part of the per-turn
  cost.

### 6.5 The talking stance

Collections belong to the one shared system prompt, and every layer frames on
it. A conversation is therefore a layer of the mind schema, `talk`, with its own
layer id and dials, framing on the same prompt. Its differences are selections
and a `talking` stance branch:

- **Unchanged:** identity, personality and world collections
  (`identity::selection_for`, `engine/identity.rs`). It is the same person.
- **Mission.** The current mission, re-reconciled every turn
  (`Minds::talk_selection`, `engine/mind/talk.rs`). For Zen, its agenda as well
  (`agenda_prompt`, `sim/missions/agendas.rs`).
- **Journal-core.** The character's journal-core entries (`talk_selection`).
- **Catch-up.** The `catchup` collection (§6.1).
- **Mood.** The current mood, read fresh from `Cast::mood_of`
  (`npcs/cast.rs`).
- **Tools.** The `tools` collection offering exactly the allowed acts
  (`show_within`), so what the decision reads is what it can decode.
- **Response.** The `response` offer of §6.3.
- **The `talking` stance branch:** the few lines of talk-only framing ("You are
  on the phone with {party}…"). Talk-only text lives here, never in shared
  sections, so an edit to it changes only the talk frame (§4.1).

**Bands.** A person's conversation decodes at **Urgent**, and an NPC↔NPC one
at **Low** (`talk_band` in `engine/talk/turn.rs`), through the timeline's
`decode_band` metadata (`DecodePriority::METADATA_KEY` in
`candle-conversation/src/projection/schema/layer.rs`; `scheduler/band.rs`), at
most `MAX_DECODE_WIDTH` = 64 decodes a wave (`candle-conversation/src/scheduler/mod.rs`).

**What a turn waits on runs at the turn's band.** A conversation's frame —
its system prompt, sealed as it opens — and every section submitted to it
(mission, journal, catch-up) seal at the conversation's own band
(`Sequence::band`: the `decode_band` it records, else its layer's). The frame
was sealed at `Low` whatever the conversation was for, and the submitted
sections at the layer's band, so work asked at `Urgent` stood behind the
cast before its turn could be submitted: an image prompt's content check
took eighty-five seconds to ask four one-token questions, past the sixty its
caller waits.

The content check's four questions are asked once as the engine comes up
(`tenant::warm_content_check`), so their frames are sealed before a draw.

**A mind carrying out a person's request is waited on too.** While a request
stands (`Scheduler::asked`, npcd's tick scheduler in
`engine/tick/scheduler_events.rs`), the character's own turns decode at
`Urgent` (`Minds::set_band` in `engine/mind/mod.rs`, called from
`engine/runtime/turn.rs`; it reaches the scheduler as
`SchedulerRequest::SetBand`, `candle-conversation/src/scheduler/requests.rs`),
and at the conversation's own band again once the person has been told.

---

## 7. Conclusions

### 7.1 How acts from a conversation reach the world

Every act from a conversation, including the line a `reply` writes, goes
through `Runtime::perform` (`authority/perform.rs`: `perform` →
`perform_said` → `perform_routed`). On a member that does not write the world
it travels to the authority as `WorldOp::Act` (`standing.writes(&world)` in
`perform_routed` decides local or authority), so mission and thread state each
keep one writer.

**Voiced text travels with the act.** The authority holds no live `Sequence`
for a character leased elsewhere, so it cannot voice anything. Voicing happens
on the lease holder, before the op is built:
- for a conversation, the voice turn (§6.3);
- for a main-loop channel post, §11.4.

`WorldOp::Act` is `{npc, call, decided: Option<Decided>, pinned_at, said:
Option<Voiced>, source: ActSource}` (`authority/ops.rs`), and `perform_here`
takes `said` and `source`. `Voiced { text, requests: Option<String> }` is the
voiced line and the critical step it asks an authorizer for (§7.2).
`enact::message`/`reply`/`sign_off` (`engine/enact/messaging.rs`: `words(said,
intent)`) append `said` in place of the intent, with its `requests` on the
line. The call is re-parsed on the authority against `tools::specs()`, or
against `tools::specs_ordered()` for an `ActSource::Talk` on an ordered body
(`make`, the `WorldOp::Act` arm of `authority/perform.rs`). Voicing runs after
the mind's turn has cleared `turn_in_flight` (the `Sequence` flag, cleared by
`finish_turn` in `candle-conversation/src/conversation/sequence_lifecycle.rs`;
the main loop's perform is `perform_said` in `engine/runtime/turn.rs`), so the
voicing ask never waits on the mind's own turn.

**Source.** `WorldOp::Act`, `perform_said` and `perform_here` carry
`source: ActSource`, with variants `Mind` (the default) and `Talk { party }`.
The name avoids a third `Origin` beside `phone::Origin` and `mission::Origin`.
- **No decision state.** For `ActSource::Talk` the authority skips
  `restore_decision`/`end_decision` (the `body` in the `WorldOp::Act` arm of
  `make` is `None`), and the talk path reads nothing from the body's
  deliberation.
- **What remains concurrent.**
  - A talk op landing while a main-loop op sits between its restore and end
    sees nothing of that state, because it reads none.
  - A talk mission change landing during a main-loop act shows up in that act's
    before/after `moved` diff (`perform_here`: `mission_mark` before and after).
    The main loop then reads it as news, which is correct.

**Speaking steps and answers fire when a line lands.** `mark_spoken`
(`engine/runtime/acting.rs`, called from `record_act_said`) and
`ledger.answered` (`engine/enact/messaging.rs`; `sim/ledger.rs`) fire when a
conversation's line lands on the thread, not when a main-loop purpose is handed
over (§10).

| Kind | Act | Applied as |
|---|---|---|
| Commit | `promise` | the ledger (`promise` in `engine/enact/notices.rs`). `to` resolves against thread members as well as the room (`here_named` and `party_named` in `engine/enact/people.rs`). The `w.tell` announcement is made only to a party who is here, and becomes a thread line when the party is not in the room. At most one per turn, beside the reply (`TALK_ACTS`). |
| Remind | `remind` | as in the main loop, with `who` resolved against thread members (`remind` in `engine/enact/notices.rs`) |
| Change, Zen (unordered) | `mission_*` | the agenda (`engine/agenda/perform.rs`), with critical steps protected (below) |
| Change, ordered body | `mission_*` (ordered specs) | below |
| Hand-off | `message`/`reach_out` to someone else | a purpose delivered (§10) |
| Noted | journal destination, with optional `reflect` | the `wrote` line; `feeling` → `note_feeling` |

**Checked again where they land.** The lease holder offers the mission acts
when the party had standing as the turn began; the authority asks again —
owner, authorizers and thrash cooldown — before applying any of them, Zen's
agenda acts included (`npcd/src/engine/talk/orders.rs`).

**Ordered bodies get their own mission specs.**
- **Why the agenda acts don't fit.** They refuse an ordered body (`perform` in
  `engine/agenda/perform.rs`). Their required `priority` binds
  `Choices::Priorities` (`engine/tools/choices.rs`, resolved in
  `engine/tools/offers.rs`), which is empty for ordered bodies, so the
  empty-set rule would remove the whole act (`engine/agenda/acts.rs`;
  `performable` in `engine/tools/compile.rs`). An ordered mission has no title
  either: titles live on `agendum` (`Mission::on_agenda` in `engine/mission/`).
- **So.** The same names (`mission_add`, `mission_edit`, `mission_cancel`,
  `mission_done`) get ordered-body specs with their own descriptions
  (`npcd/src/engine/agenda/ordered.rs`). `mission` binds the active mission's
  prompt, and there is no `priority`.
  - **How a turn takes them.** They are not in `CATALOG`, which is keyed by
    name. A talk `Within` with `orders` set (an ordered body, a party with
    standing) takes each in place of the agenda's act of its name
    (`tools::variant`), and they are installed in the prompt under members of
    their own (`tools/ordered/<name>`).
  - **How they are read.** The parser asks a parameter for only when the
    catalog the decode was drawn from takes it, and the authority reads a
    conversation's call for an ordered body against `tools::specs_ordered()`.
  - `mission_add` → `lodge_mission(.., start: true)` with
    `Origin::Lodged { by: account }`, which assigns it, and calls off the
    current mission and its operation (`Runtime::lodge_mission` in
    `authority/perform.rs`, called from `ordered_act` in `engine/talk/orders.rs`;
    `Missions::assign` in `sim/missions/active.rs`). Not `Origin::Own`, which
    closes as a routine (`Missions::close` in `sim/missions/close.rs`).
  - `mission_edit` → `Missions::edit_active` (`sim/missions/active.rs`). It
    keeps every done and every critical step (`kept_steps`), and refuses a
    mission with an operation, whose steps are the operation's.
  - `mission_cancel` → `cancel_mission` (`engine/runtime/missions.rs`).
  - `mission_done` → the `report_done` handler (the `report_done` arm of
    `mission` in `engine/work/mission_acts.rs`), so every existing gate applies
    (`still_doable` in that file, `written_up` and `unread_step` on `Mission`
    in `engine/mission/lifecycle.rs`, `report_block` in `engine/work/report.rs`).
    That handler also refuses while a critical step is unsigned (§15).

**Critical steps are protected on the agenda path too.**
- `agenda_edit` (`sim/missions/agendas.rs`) rebuilds `todo` through
  `kept_steps`, which keeps critical and done steps, so an unsigned critical
  step is neither dropped nor re-created as non-critical.
- `agenda_done` (`sim/missions/agendas.rs`) is refused while a critical step is
  unsigned.

### 7.2 Authority

The party is known by account (D14):

- **Owner.** `payload(npc).owner_id` is the party's public handle
  (`talk_mission_act` in `engine/talk/orders.rs` reads it and passes it to
  `talk::authority::standing`, which yields `Standing::Owner`). That
  `Standing` (Owner, Authorizer, Nobody) is a different type from
  `authority::worlds::Standing`, the lease bookkeeping of §9.
- **Authorizer.** `WorldMissions.authorizers` (`engine/mission_gen/world.rs`) is a
  list of `Authorizer { account, name }`, and `is_authorizer`
  (`engine/authorize.rs`) matches the party's account handle.
  - **Why not names.** Display names come from the identity provider, can
    change, and collide. The creator's carries a " (The Creator)" mark
    (`CREATOR_MARK` in `engine/voices.rs`).
  - Names are kept only to render `critical_compass` (`engine/authorize.rs`).
  - This is the identity check `docs/mission_templates.md` asks for: the
    authorizers are "each by Discord identity" (its critical-step section).
- **Getting the account to the authority.** The account rides in the message's
  sender: `WorldOp::Message { npc, from: Who, text, origin }`, where
  `Who { id, name }` (`sim/phone/mod.rs`) carries the account handle as `id`.
  `inbound::judge` (`discord/dm/inbound.rs`) builds it from
  `public_id(&account)`, and `message_npc` (`engine/runtime/messages.rs`)
  stores it as `Said.account`. *(Replaced design: a separate
  `account: Option<String>` field on the op. The member id of a thread is
  already the handle, so the sender carries it.)*
- **An NPC party** is never an authority over another NPC's mission or agenda.
- **Everyone else** gets no mission acts, for Zen's agenda as for an ordered
  mission. They can get a `promise`, and the NPC's main loop decides.

**Critical steps: a request, then a yes.** A conversation signs off a critical
step only when both hold:

1. **It asked.** When a purpose reaches a conversation whose party is an
   authorizer while `critical_next()` is `Some(step)` (`Mission::critical_next`,
   `engine/mission/lifecycle.rs`),
   the runner records `talk.pending_critical = step` (`asks` in
   `engine/talk/critical.rs`). The posted line carries `requests: step`
   (`Voiced.requests`, a `Message` field, §11.1).
2. **The authorizer answered yes.** Their next message is affirmative
   (`is_affirmative` in `engine/authorize.rs`), and `step` is still
   `critical_next`, re-checked on the authority.

Then the authority runs `authorize` (`Runtime::authorize_step` in
`engine/runtime/missions.rs`, from `WorldOp::Authorize { npc, step, by, name }`).
A "yes" to "want tea?" approves nothing.

**The weaker path is removed.** The phone `read` path no longer authorizes
(`read_phone` in `engine/enact/reading.rs` calls neither `authorize` nor
`is_affirmative`), so conversations are the only way a critical step is
signed off.

**Thrash guard.** A body's mission may change through a conversation at most
once per `CHANGE_COOLDOWN_MS` = 600,000 (10 min of world time) for an
authorizer; the owner is always allowed (`may_change` in
`engine/talk/authority.rs`; the last change is kept in `Runtime.talk_changes`).
Within the cooldown, the mission acts are struck through `cooling`. Authority
ops apply one after another, so concurrent changes resolve as last writer wins,
after each passes the authority check.

**The main loop is told.** After applying, the authority delivers an URGENT
`Concluded { party, applied }` (`engine/event.rs`; `party` is the party's name)
through `Runtime::tell` (`engine/runtime/moments.rs`), which reaches the
character wherever it is held. It is emitted by `Runtime::authorize_step` for a
signed-off step and by `engine/talk/turn.rs` for the acts a conversation
applied. The next decision starts from the new mission and knows why.

### 7.3 Outcomes flow out, not back

An act's outcome reaches the conversation's next turn the way main-loop
outcomes reach theirs: a `<tool_response>` ahead of the user turn
(`compose_narrated` in `engine/mind/perception.rs`). `Within.owed` is built from
the party for a conversation, not from the room (`Within::on_thread` in
`engine/talk/within.rs` takes it from `OnThread.owed`, which `talk/turn.rs`
builds).

---

## 8. Guards

Conversation guards are **per conversation** and never touch the character's.

- **The main loop's strikes land on the character.** Its `Within.cooling` is
  the per-character union of `cooldowns.cooling(npc)` and
  `loop_guards.cooling(npc)`, plus `reflect` when the mind is bodiless or has no
  dream available, sorted and deduplicated (`Runtime::within` in
  `engine/runtime/perceiving.rs`). `perform_routed` (`authority/perform.rs`) and
  `act_on_world_said` (`engine/runtime/acting.rs`) stamp `cooldowns.took_act`
  only for `ActSource::Mind`.
- **A talk `Within` builds `cooling` only from its own state**
  (`guard::cooling(&party, last_said_ms, now)` in `engine/talk/guard.rs`, taken
  by `OnThread.cooling`). Acts with `ActSource::Talk` never call
  `cooldowns.took_act` or `loop_guards.record` for the character.

| Guard | Rule | Why |
|---|---|---|
| Burst window | A turn starts `BURST_MS` = 2,500 ms after the last unanswered message, and at most `BURST_MAX_MS` = 8,000 ms after the first (`engine/talk/clock.rs`). | one answer to three quick lines |
| One turn at a time | A message or purpose arriving during a turn is queued for the next turn boundary. | injection without races |
| A turn that could not start | A turn writes nothing until its model has decided: what it heard is noted once it has, and only a critical step signed off writes the world sooner. A turn that fails before its first write — the model could not run, the card had no room — puts what it took back, and the conversation tries again after `RETRY_FIRST_MS` = 5 s, doubling for each failure in a row up to `RETRY_MAX_MS` = 5 min (`retry_delay` in `engine/talk/clock.rs`; `Talks::put_back` in `engine/talk/runner.rs`, called from `talk_turn` in `engine/talk/turn.rs`). One that fails after writing is finished as before: running it again would say or note things twice. | a message is never lost to a mind that could not answer yet; on 2026-11-11 one to Zen was, to a turn with no schema |
| Repeats | A non-mutating `would_repeat(lines, text)` (a free function in `engine/loopguard.rs`) runs on the voiced text before it is sent, against this thread's own last `SPEECH_LOOKBACK` = 4 lines, at the guard's `SPEECH_CLOSE` = 0.45 overlap (`words_for` in `engine/talk/guard.rs`). A repeat changes the destination to the journal, and the line "Started to tell {party} again that …" is written. There is **no carried-forward strike**: the main loop's `Guard::record` strikes the speech family for up to `COOL_CAP` = 16 turns, which would silence a person's next 16 messages. | no posted repeats; Zen telling Mira what it told John is not a repeat |
| No reply cooldown with people | Person conversations have no `reply` cooldown. The main loop's 6 s `reply` cooldown (`("reply", Cost::cools(6))` in `engine/cooldown.rs`) would strike a reply whenever a person answers within about 3 s. A conversation with an NPC party keeps 6 s (`REPLY_COOLS_MS` in `engine/talk/guard.rs`), derived from the thread's last line said to that party rather than from a `(npc, act)` entry. | people are never made to wait |
| Activation cap | NPC party only. `TALK_TURNS` = 8 turns per activation (`capped` in `engine/talk/guard.rs`). An activation starts at a purpose. After the cap, `reply` and `message_someone_else` are struck (`TALKING_ACTS` in `engine/talk/within.rs`), and the turn goes to the journal or a `sign_off`. | two NPCs cannot talk forever, and the exchange ends in something |
| Quiescence | A journal turn sends nothing, so the other side hears nothing. A line from an NPC never opens a turn on the other side in any case (D2). | no idle chatter |
| Unprompted contact | A purpose for a party whose thread ends in `UNANSWERED_LINES` = 2 or more of the NPC's own lines, the newest said less than `UNANSWERED_HOLD_MS` = 3,600,000 (1 h) ago, is held (`holds_purpose` in `engine/talk/guard.rs`, checked in `talk/turn.rs` before a telling): it is written to the journal, and the mind is told it waits for the party to write again. Two, not one: the phone's "I'll look" and the mind's answer are both owed — held at one, every answer after a promise to look was held. Past two the NPC is talking at someone away: Zen sent John four lines in twenty-four seconds, the last two "I'm here. What do you need?". | no message storms at someone away |

**Typing.** Discord's indicator derives "waiting" from the thread
(`show` in `discord/dm/typing.rs`), on the authority's bot
(`discord/dm/record.rs`). In the phone (§11.1) "waiting"
means the person's last line is above the NPC's read cursor. A turn that ends in
the journal marks read through its input, so typing stops without the authority
knowing the holder's turn state.

---

## 9. Where it runs

The world's authority and a character's lease holder are different members.
Character leases are lent by weighted rendezvous over the npcd members
(`authority/npc_lease.rs`: `NpcLease`), and the world lease is held separately
(`authority/world_lease.rs`). A conversation runs on the **lease holder**, in a
per-member runner for the characters held there (`engine/talk/runner.rs`:
`Talks`).

- **Delivery carries the lines.** `deliver_messages` runs on the authority
  (`engine/environment.rs`). For a direct thread whose new lines come from a
  person it emits `EventKind::Talk { thread, lines }` at `Salience::URGENT`, the
  new lines themselves (`TalkLine`: name, seq, from, text, account, at), instead
  of `EventKind::Message`. When every new line comes from a character it emits
  an ordinary `EventKind::Description` ("{name}, on your phone: …") for the
  mind instead, and no conversation turn opens (D2).
  - **Why carry the lines.** A `Told` pulse travels on the beat's pulse channel
    (`authority/beat.rs`), and nothing orders it against the thread's world
    record (`world_state/snapshot.rs`). A runner reading its local copy could
    find nothing new.
  - **Where it is intercepted:** by `Scheduler::deliver`
    (`engine/tick/scheduler_events.rs`) locally, and in the `TOLD` arm of
    `authority/follow.rs` on the holder.
  - **No reminders.** `Threads::unread_due` (`sim/unread.rs`) reads groups and
    channels only, so the mind is never reminded about a conversation thread.
- **Turns count toward the hand-back.**
  - **A count, not a set.** `Standing` keeps `turning: HashMap<u64, usize>`, a
    count per character (`authority/worlds.rs`).
  - **The runner's calls.** The runner calls `Standing::begin_turn`, which is
    refused while the character is being handed back, and `Standing::end_turn`
    around every conversation turn (`talk_turn` in `engine/talk/turn.rs`). These
    are not `Runtime::begin_turn` (`authority/perform.rs`), which sends
    `WorldOp::Begin` and takes the mind's arrival and compass news.
  - **When a hand-back happens.** A character is handed back only when no turn
    of its mind or any conversation is running (`hand_back_due`).
  - **On the new holder.** Conversations resume only once
    `NpcLease::ready(applied)` holds (`authority/npc_lease.rs`, checked in the
    reconcile loop of `authority/cast.rs`).
  - **`Runtime::put_down`** (`authority/cast.rs`), which can run without a
    hand-back when an admin or the authority moves a lease, cancels in-flight
    conversation turns (`talks.forget`) before releasing them.
  - **What this prevents.** A conversation timeline written from two members.
- **A message is never left unanswered.** `Told` pulses are not persisted
  (`authority/told.rs`). The runner rescans whenever it starts a character:
  `spawn_character` (`engine/runtime/supervise.rs`, which calls `rescan_talks`;
  its API caller is in `api/npc_routes.rs`) and `Runtime::wake_character`
  (`authority/cast.rs`). The rescan (`rescan_talks` in `engine/talk/turn.rs`,
  over `Threads::unanswered_for` in `sim/phone/read.rs`) covers every
  conversation thread and takes a turn for lines with a seq above
  `talk.answered_seq`.
  - **A thread with no conversation yet** initialises the bookmark to just
    below the first line it was handed (`TalkState::begin` in
    `engine/talk/state.rs`), so it does not answer the whole history.
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
    (`MESSAGE` in `engine/acts/conversation_acts.rs`, performed by `message` in
    `engine/enact/messaging.rs`) and `reach_out` (`REACH_OUT` in the same file)
    no longer write thread text. Each delivers `Purpose{intent}` to the `(npc, to)`
    conversation, opening it if needed, under the §8 hold.
  - **What the conversation sees.** Its next user turn carries "You wanted to
    tell {party}: {intent}".
  - **The act's outcome text** is "You mean to tell {to} {intent}; you will say
    it in your own words" (`message` in `engine/enact/messaging.rs`).
  - **Channels are unchanged.** `message{via: channel}` stays on the main loop
    (§11.4).
- **The main loop keeps off conversation threads.**
  - **`reply`.** It binds `Choices::Awaiting` (`engine/tools/choices.rs`), which
    would include direct threads where the party spoke last. The main loop's
    `awaiting`, `leavable` and `invitable` sets (`awaiting_for`, `leavable_for`
    and `invitable_for` in `sim/mod.rs`, over `sim/phone/read.rs`) exclude
    conversation threads, so its `reply` and `sign_off` never touch them.
  - **`invite`** is refused on a direct thread (`Threads::invite` in
    `sim/phone/channel.rs`). A group starts only with `open_group`.
- **A conversation's own `reply` and `sign_off`** write to its party's thread.
  They are the only acts that do.
- **A hand-off** is a conversation's `message` or `reach_out` toward someone
  other than its party. It is the main loop's act on the main loop's path, and
  delivers a purpose to that other conversation.
- **NPC↔NPC.** A's line on the A–B thread is delivered to B's mind as a
  `Description` of what happened (§9), not as a `Talk` turn: a person's DM
  opens a turn, a character's line does not (D2). B answers by its own choice,
  through its main loop's `message`, which hands B's own conversation with A a
  purpose.

---

## 11. The phone, rebuilt

The phone is `npcd/src/sim/phone/`, one concern per file (`line.rs`, `thread.rs`,
`direct.rs`, `channel.rs`, `read.rs`, and `mod.rs` as the index). It replaced the
single-file phone rather than patching it.

### 11.1 Direct threads

- **Members are ids.** A member is a body id or an account handle, never a
  display name. The display name is looked up when rendering. The ledger still
  keeps promises by display name (`promise` and `remind` in
  `engine/enact/notices.rs`; `Ledger::promise(by, to, ..)` in `sim/ledger.rs`);
  only thread membership is by id. *(Replaced design: keying the ledger by id
  as well.)*
- **One thread per pair, forever.** A direct thread's id is derived from the
  sorted pair of member ids (`direct_id`, `dm:{lo}:{hi}`, in
  `sim/phone/direct.rs`), so reaching someone again finds the same thread.
  - Its seq never restarts.
  - It is never evicted; `KEEP_GROUPS` (`sim/phone/channel.rs`, 64) bounds
    group threads only. Each thread also caps its own stored lines
    (`KEEP_DIRECT` = 64 and `KEEP_GROUP` = 128 in `sim/phone/thread.rs`).
- **`sign_off` does not leave.** It marks the conversation ENDED (§4.3) and
  leaves membership alone. A direct thread is never closed by its members.
- **Read is bounded.** `Threads::read_through(who, thread, seq)`
  (`sim/phone/mod.rs`, over `Thread::read_through` in `sim/phone/thread.rs`)
  marks read up to a seq, never "everything", so a message that arrived during
  the turn stays unread. The plain `read` (`sim/phone/read.rs`) marks through
  the newest line as it stood when it was called.
- **A line** (`Message` in `sim/phone/line.rs`) holds:
  - the voiced text;
  - the deciding intent, for diagnostics;
  - `requests: Option<String>`, the step it asks an authorizer for (§7.2);
  - the sender's account.

  Discord posts the text. `spoken(intent)` (`engine/reported.rs`) has no
  callers outside its own tests.

### 11.2 Channels and groups

Channels and group threads are a separate kind: fire-and-forget, on the main
loop.

- Members are ids.
- They are bounded by per-thread line caps (`KEEP_GROUP`) and, for the groups
  characters open, by `KEEP_GROUPS` (`forget_quiet_groups` drops the quietest);
  permanent and standing groups are never candidates.
- The world channel (`phone::CHANNEL` in `sim/phone/channel.rs`) is one of them.
- The roster functions cover both kinds by id (`sim/mod.rs`): `enrol`,
  `contacts_for`, `invitees_for`, `leavable_for`, `awaiting_for`.

### 11.3 The main loop on the phone

- **Answering marks read** through the line answered.
- **The phone shows your own last line.** `read_phone`
  (`engine/enact/reading.rs`) adds "(you last said, {ago}: …)".
- **Repeats are checked before posting.** `would_repeat` runs on the voiced
  words against the character's own lines on that thread, before the post
  (`talk::voice::post`), and a repeat is refused with the guard's nudge, never
  posted and then struck.

### 11.4 Voiced channel posts

A main-loop `message{via: channel}` carries an intent, as do a group `reply`,
`open_group` and `sign_off`. Before `perform` builds the op, on the lease
holder, the intent is voiced (`Runtime::voice_post`, `post_of` and
`post_prompt` in `engine/talk/voice.rs`):
- an `UnsealedAsker::ask` on the mind's live sequence, with no tools;
- `<think>` and `<tool_call>` banned;
- the `response` offer `talk/channel/*`;
- the user turn "Post to {to}, in your own words, as you would type it:
  {intent}" ("Say goodbye on {to}, …" for a sign-off);
- a `POST_TOKENS` = 320 cap, at `POST_BAND` = `High`.

The text rides in `WorldOp::Act.said` (§7.1). The cost is one short decode per
channel post.

### 11.5 Discord

**Per-part cursors.** Each part of a split message is cursored:
- `Dm` carries `mirrored_part` (`discord/dm/record.rs`, a record that is `Copy`
  and `deny_unknown_fields`), and the server and world records (`discord/server/record.rs`,
  `discord/world/record.rs`) carry the same.
- `split` (`discord/post/split.rs`) is deterministic over the posted text, so a
  failed later part resumes at the right part.

**Duplicate suppression where Discord supports it:**
- **Create Message** carries `nonce` = a 24-character digest (the first 12
  bytes of the SHA-256, in hex) of `{world}:{thread}:{seq}:{part}`
  (`Target::nonce` in `discord/post/mod.rs`), with `enforce_nonce: true`
  (`create_message_once` in `discord/client.rs`). Discord's own window for this
  is a few minutes.
- **Execute Webhook** (`execute_webhook` in `discord/client.rs`) takes no nonce.
  The world mirror's webhook posts rely on the per-part cursor.

Within D13, a rare duplicate is accepted.

### 11.6 What the rebuild must carry

Everything that reads or writes `Threads` uses the new API:

- **World state.** The thread records (`npcd/src/world_state/snapshot.rs`, the
  `threads/threads` entity).
- **Discord DM:** `npcd/src/discord/dm/{inbound, outbound, typing, record}`.
  These look threads up by `direct_between` over member ids (`inbound.rs`,
  `outbound.rs`, `typing.rs`).
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
  `channel` (`npcd/src/engine/runtime/messages.rs`).
- **Roster and unread:** the roster in `npcd/src/sim/mod.rs`, and
  `npcd/src/sim/unread.rs`.
- **Engine:** `npcd/src/engine/{on_you, channels, survey, mod}.rs` and the
  `npcd/src/engine/pulse/` folder.
- **Other sim users:** `npcd/src/sim/{posting, item, seed}.rs`. `seed::issue_handset`
  (`sim/seed.rs`) gives each NPC its handset.
- **Zen tasks:** `npcd/src/authority/zen_task.rs`, which already reaches by
  body id.

---

## 12. Speech in the room needs a person in the room

**The problem.** Offering an act is what makes a character take it (the module
doc of `engine/tools/act_tools.rs` records how `whisper` became the default
voice, and how three characters alone in three rooms narrated to nobody).
Offering `tell` to an NPC whose only company is other NPCs is what makes it
talk to them aloud instead of messaging them.

**The rule.** Room speech is for people. NPCs talk to each other by `message`.

| Act | Before | Now |
|---|---|---|
| `tell`, `ask` | `Nearby` | `PersonNearby`: a person's body is in the room |
| `whisper` | `AmongOthers` | `PersonAmongOthers`: a person is here, and somebody else to keep it from |
| `promise`, `remind` | `Nearby`; both announce with `w.tell` (`engine/acts/tower_and_work.rs`: `PROMISE`, `REMIND`; `engine/enact/notices.rs`) | in the room, `PersonNearby`; on a thread, the party (§7.1) |
| `shout` | `Always` | `PersonInEarshot`: a person's body is where the shout reaches, **or the body is in field play** |
| `gesture` | `Nearby` | unchanged |
| `message` and the phone acts | `Always` | unchanged |

The thread's own allow-list adds a fourth availability, `Availability::OnThread`
(`engine/tools/types.rs`), for the acts only a conversation offers.

**Why the related acts are gated too.** Leave any of them open and it becomes
the room voice NPCs reach for, which is how `whisper` became the default once
before (the "`whisper` is the exception" section of the module doc of
`engine/tools/act_tools.rs`).

**Messaging needs no person.**
- **Always offered.** `message`, `reply`, `reach_out`, `invite` and
  `open_group` (`engine/acts/conversation_acts.rs`) and `sign_off`
  (`engine/acts/tower_and_work.rs`) are `Availability::Always`, bound to the
  handset's threads.
- **The handset.** Every NPC gets one from `seed::issue_handset`
  (`sim/seed.rs`).
- **Without a handset.** A handset can be taken, and an NPC without one can
  talk to nobody unless a person is present. That is the existing rule, and it
  is intended.

**What "a person" is.** In every world a person is a `visitor:{handle}` body,
where `handle` is the person's **account handle** (`body_of` in
`engine/interaction.rs`; already how npcd tells a person's body from a
character's, `engine/order_table.rs`; `is_person` in `engine/people.rs`).
- **One rule, both uses.** The handle is the same identity that authorizers,
  owners and thread members use (D14). `Within.people` lists these bodies, and a
  conversation with one is `Party::Person(account)`, with the account read
  straight off the prefix. So no world needs an `is_player` mark, and no second lookup
  turns a body into an account.
- **How a body is placed.** Three existing world ops:
  - `WorldOp::EnterBeside { npc, visitor, name }` places it (`authority/ops.rs`);
  - `WorldOp::Place { visitor, at }` moves it;
  - `WorldOp::Leave { visitor }` removes it.
- **In the moon** visitors come only from console physical sessions (the
  `Mode::Physical` branch of `engine/interactions.rs`, which sends
  `WorldOp::EnterBeside`), so moon NPCs speak aloud only to someone standing
  in a console session. That is intended.
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
    world is its own seeded tower (`battle_cities` in `npcd/src/sim/seed.rs`), and
    nothing carries a game avatar into it from npcd's side. The Battle Cities server owns the
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

**What "field play" is.** There are no squads in the field sim
(`npcd/src/sim/field.rs`; its module doc mentions a `shout` to the squad). Field
play is defined as: the body holds a field stance, or a hostile is within its
reach.
- `Within` carries `in_field: bool`, computed from that in `Within::from_sim`
  (`engine/tools/within_tools.rs`): `sim.field.stance(body).is_some()` or a
  hostile in reach.
- It is tested against fixture fields.

**How it is built:**
- **`Within` fields.** `people: Vec<String>` (the company that are visitors),
  `heard_by_person: bool` and `in_field: bool` (`engine/tools/within_tools.rs`).
  `people` and `heard_by_person` are filled in `Runtime::within`
  (`engine/runtime/perceiving.rs`) from the `actors_at` read, which yields ids.
  All are part of the grammar cache key.
- **One hearing rule.**
  - `heard_by_person` uses perception's rule (`hear_a_shout` in
    `engine/people.rs`).
  - `Scope::at(world, &at).reach(&place)` is public (`npc-map/src/witness.rs`).
  - The voice-carries rule that the private `legible` in `npc-map/src/witness.rs`
    applies is factored out as `pub fn carries(voice, reach, aimed_at_reader)`.
    It keeps the "aimed at the reader" cases, so the grammar and perception never
    diverge.
- **Availabilities.** `Availability` (`engine/tools/types.rs`) has
  `PersonNearby`, `PersonAmongOthers` and `PersonInEarshot`. They are matched in
  every exhaustive match:
  - `engine/tools/show.rs` (the `Availability` match and `nearby`);
  - `specs_within` in `engine/tools/compile.rs`;
  - `engine/pulse/vocabulary.rs` (the modes filter, and the `needs` strings: "a
    person here", "a person here, and somebody else", "a person within earshot,
    or the field").
- **`to`** stays bound to the whole company (`Choices::Company` in
  `engine/tools/offers.rs`).
- **`tools::nearby`** takes `people` and `heard_by_person`
  (`nearby(company, people, heard_by_person)` in `engine/tools/show.rs`).

**Model-facing text that steers NPCs to room speech is rewritten.** Each piece
below would otherwise tell the model to use an act the grammar no longer
offers:

| Where | Before | Now |
|---|---|---|
| `frame` in `engine/prompt.rs` ("HOW YOU ACT") | "If somebody is here, speak to them … two people in one room texting is worse than either of them saying nothing." | "Speak to the people here — `tell` is how you speak to a person … To reach another character, here or anywhere, message them: you carry a handset …" The advice to ask rather than announce, and to go where an answer sends you, stays. |
| `MESSAGE` description and doc comment (`engine/acts/conversation_acts.rs`) | "If the person you want is standing in front of you, `tell` or `ask`." | "How you reach another character, even one standing beside you. Speak aloud only to the people here." |
| `TELL`/`ASK`/`SHOUT` descriptions (`engine/tools/act_tools.rs`) | "how you talk to a person", "Not for talking to somebody who is with you — that is `tell`" | "how you talk to the people here"; `SHOUT`: "How you talk to the people within earshot all at once" |
| `GESTURE` description (`engine/tools/act_tools.rs`) | refers to speech in company | unchanged: "Do something without speaking …"; it stays `Nearby` |
| `act` description (`engine/acts/mod.rs`) | "promise or order somebody, use `tell`, `ask`, `whisper` or `shout`" | "To tell, ask, warn, promise or order a person here, use `tell`, `ask`, `whisper` or `shout`, and `message` another character" |
| `post_notice` description (`engine/acts/carrying.rs`) | "`tell` reaches whoever is standing here" | "speak to the people here, message the others" |
| `promise` example (`engine/acts/tower_and_work.rs`) | room speech | a thread ("… on the phone to you …") |
| Cooldown rationale (`engine/cooldown.rs`, the `message` entry) | messaging priced "dearer than being in the room" | "The rate paces a correspondence" |
| `IN_COMPANY` idle nudge (`engine/runtime/instructions.rs`, used in `engine/runtime/moments.rs`) | room speech | "speak to a person here, `message` another character" |
| Mission compass (`compass_line` in `engine/mission/compass.rs`) | "`ask` or `tell` them" | "say it to them now — `message` them", or "`message` them, which reaches them wherever they are" |
| Work text (the reach text of `engine/work/mission_acts.rs`) | "`move_to` {room} and `ask` or `tell` them" | person-gated: a person here gets "`ask` or `tell` them", a person elsewhere "`ask` or `tell` them — or `message` them", a character "`message` them" |
| `awaiting_line` (`engine/environment.rs`) | "`tell` them if they are here" | "Answer them: `message` them, or speak to them if they are a person here." |
| `company_line` (`engine/environment.rs`) | lists `tools::nearby` for NPC-only company | lists `tools::nearby(others.len(), people, heard)`, what is offered |
| Standing mission templates (`mod routines` in `engine/mission/mod.rs`) | face-to-face NPC talk | `visit-and-review` and `trace-a-thread` use "message {who} …"; `sit-with-a-story` ("find someone and tell them") and `take-stock` ("tell the next person you meet, properly") still say `tell`, which now reaches a person present |
| Tool examples (`engine/tools/act_tools.rs`) | NPC↔NPC room speech | rewritten with a visitor present ("Wren, a visitor, is here"; "Two visitors …") |

**What it does not break:**
- Mission `met` steps sign off on presence (`engine/runtime/missions.rs`: whoever
  is in the room with the character meets a step to find them). Speaking steps
  move with `mark_spoken` (§7.1).
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
| `npcd/src/engine/talk/orders.rs` | a conversation's mission acts on the authority: standing re-checked, an ordered body's applied | refusals for a stranger and inside the cooldown |
| `npcd/src/engine/talk/hand_off.rs` | `message_someone_else`, the thread's own hand-off act, and the acts struck once a hand-off is spent | the hand-off set offered per party, and what is struck |
| `npcd/src/engine/talk/journal.rs` | the conversation's own journal (§5.5): `stretch`, `TALK`, `TalkJournal` | the stretch's citable lines; the style with no fact slot |
| `npcd/src/engine/talk/templates.rs` | installing the `response` offer and offering a destination's templates | the offered member names per destination |
| `npcd/src/engine/agenda/ordered.rs` | the ordered-body mission specs and the order each call makes | the specs' names and params; each order read off exact arguments |
| `npcd/src/engine/experience/days.rs` | the world's day rolling over, once | a roll told once, a first sight told nothing |
| `npcd/src/engine/experience/{line,buffer,concern,store}.rs` | lines, per-bot buffer and flush, `concern`, npcd's `journals::live::Live` per world | rendered lines and experience fields; flush order and caps; the exact journal paths per change; a flush lands under `groups/<g>/bots/<bot>` as `Actor::Npcd{group}` |
| `npcd/src/sim/phone/{thread,direct,channel,read,line}.rs` (the roster is in `npcd/src/sim/mod.rs`) | the rebuilt phone (§11) | stable pair ids; seq continuity; `read_through` bounds; no eviction of direct threads; `invite` refused on direct threads |
| `journals/src/live.rs` | the live store (the store mutex, published snapshot and sync thread), shared by zend's `Memory` (`zend/src/mcp/memory/mod.rs`) and npcd's `Journals` | as zend's existing memory tests |
| `journals/src/view/since.rs` | lines and folds after a point, with overlap and seen ids | exact lines for fixture trees around folded periods |
| `journals/src/ops/episodes.rs` | `Op::Episodes`: one commit, per-line `at` and id, per-line trailers | exact tree and commit message; replay renames each colliding id; refusal outside `bots/` |
| `journals/src/experience.rs` | the `Experience` line field | raw rendered bytes; fold conflict on change |
| `candle-conversation/src/conversation/submitted_sections.rs` | `Sequence::publish_sections` | an ask after publish sees a section sealed before it |
| `candle-conversation/src/projection/resolver/recurrent_state.rs` | `snapshot_durable(timeline, turn)` | false before, true after a turn's snapshot is durable; the scheduler's drain is unaffected |

Changed:

- **Grammar.** `Catalog { mode, calls }` through `grammar_for`, `act_specs`,
  `show_within` and the cache key (§6.2). `allow` in `performable`.
  `estimated_paths` takes a call count.
- **Acts.**
  - `reply` binds the party whoever spoke last.
  - `sign_off` keeps membership.
  - `promise`/`remind` work on threads and are gated in the room.
  - `reflect` sets the mood only, in a conversation.
  - Ordered-body mission specs, `Missions::edit_active`, critical protection in
    the agenda and `report_done`.
  - `read` no longer authorizes (§6.2, §7).
- **Authority and leases.**
  - `WorldOp::Act` carries `said` and `source`; `WorldOp::Message` carries the
    sender as `Who` (the account handle is its `id`); `WorldOp::Reshape`
    (`npcd/src/authority/ops.rs`, `perform.rs`).
  - `turning` is a count, and `put_down` cancels (`worlds.rs`, `cast.rs`).
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
- **Guards.** `would_repeat`; talk cooling isolated; cooling derived per
  conversation (`loopguard.rs`, `cooldown.rs`, `runtime/perceiving.rs`,
  `talk/guard.rs`).
- **Discord.** Per-part cursors in the dm, server and world records; a nonce on
  Create Message.
- **Journals.**
  - `Actor::Npcd{group}` with its own read arm and permits.
  - The `bots/` child cap lifted, including the check.
  - `bots/` exempt from summary triggers, and folds on `BOTS_PER_PASS`.
  - Files: `journals/src/actor.rs`, `ops/permit.rs`, `scope.rs`, `tree/check.rs`,
    `brief/mod.rs`, `ops/wire.rs`, `store/message.rs`,
    `zend/src/session/memory_writer.rs`.
- **Config.**
  - `worlds/<id>.yaml` takes `journals: {group}`.
  - motherd.yaml's `npcd:` block takes `journals_dir:`.
  - The agent carries a `journals-npcd` checkout on every machine that runs npcd.
  - The `battle-cities` group is created by the `zen` service person.
- **Shared prompt.**
  - `response` is `Offered`, with the main loop offering its set.
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

- **`report_done` used to skip the critical-step check; it no longer does.** The
  `report_done` arm of `mission` in `engine/work/mission_acts.rs` refuses while a
  todo is `critical` and not `done` ("This mission waits on an authorizer's yes
  to …"), where `still_doable` alone refused only a step whose target is in
  reach. §7.1 routes `mission_done` through this handler, so the check holds for
  the main loop as well. The test is "A mission with an unsigned critical step is
  not reported done" in `engine/work/tests_mission.rs`.
- **`engine/sleep.rs` describes code that does not exist.** Its module doc (item
  3 of "What sleeping does") says a finished day is "folded into the memory
  layer". The roll-over (`Scheduler::roll_day` in `engine/tick/scheduler_ticks.rs`)
  strands the window into the journal and rolls the window over; no code folds
  the day into a memory layer.
- **Two stale comments:**
  - `engine/mind/think.rs` says "two decodes, the narrator then the act" (and
    "one decode's worth of narrator work followed by the act decode"). The
    narrator is a synchronous render (`narrator::build_turn` and
    `narration::render`; the module doc of `engine/narrator.rs`, "The shape").
  - The `finish_turn` doc (`candle-conversation/src/conversation/sequence_lifecycle.rs`)
    says it prefills the next header and returns the boundary text that was
    prefilled. `finalize_turn_post_done`, which it calls, does not: its body
    notes "No inter-turn header prefill is needed" and returns an empty string.
- **`zen_task_act` keeps its handle on the replica.** It runs `perform_here`
  there, because task acts are not of the world (`of_the_world` in
  `perform_routed`, `authority/perform.rs`, leaves them out), so `set_task`
  writes only the local copy. `tend_zen_tasks` polls only the worlds this member
  writes (`carried_tasks` in `authority/zen_task.rs` skips a world for which
  `standing.writes` is false), so the task is never polled.
