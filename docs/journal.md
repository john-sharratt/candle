# The journal

A character's conversation keeps its last 32 completed exchanges and the
perception window keeps 64 turns. Whatever falls off either is gone from the
character's own line of thought, and what stays is read back at face value —
including a claim the character made once, never checked, and has been
re-reading ever since (a "fabricator 5 lost" that persisted fourteen minutes
across thirteen of the thirty-two exchanges). The journal gives a character
something longer than the tail and more careful than the tail: a short, dated,
checked record of what it saw, heard, did and left open, written off the main
line and read back in the system prompt.

This document is authoritative over the code in `npcd/src/engine/journal/`.

| File | Concern |
|---|---|
| `entry.rs` | `Entry`, `Claim`, `Kind`, `Item`; the prose an entry renders to |
| `verify.rs` | the server's half: citation check, kind, perishability, typed-claim check, open-item rules |
| `tools.rs` | the `journal_write` call and the grammar it is held to; reading an answer back |
| `ask.rs` | what the character is asked to write about a stretch |
| `workflow.rs` | `run`: the write, the check, the keep, against a `Desk` |
| `desk.rs` | `EngineDesk`: the question put to the live conversation and the real keep |
| `state.rs` | per-character `JournalState`: cadence, coverage, open items |
| `section.rs` | the journal as the sections of a system prompt, and the metadata record a restart rebuilds from |

The question of whether a stretch needs an entry at all is not here: it is the
guardian's `journal` module (`guardian/modules/journal.rs`,
`docs/npcd_guardian.md`), one of the background questions it puts to a character.
| `world.rs` | `Snapshot`: what the sim can say about a claim |

## 1. Shape

A journal is a sequence of **entries**. An entry covers a span of the character's
window — `from_turn ..= to_turn`, and the world instants they fall at — and holds:

| Field | What it holds |
|---|---|
| `claims` | one flat list; each claim is text, citations and a server-derived `kind` |
| `intend` | what the character means to do next, at most `MAX_INTEND` (3) lines |
| `opened` | items left unresolved, each with an id the server assigns |
| `resolved` | ids of earlier open items this entry settled |

A claim has four kinds, and the **kind is not written by the model**:

| `Kind` | Meaning | Rendered under |
|---|---|---|
| `Observed` | a tool result or a world event the character itself received | `Saw:` |
| `Heard` | somebody else said so — a peer, the narrator, the operator | `Heard:` |
| `Did` | the character's own act, recorded: a fact about what it did, not about the world | `Did:` |
| `Inferred` | nothing in the world supports it; the character concluded it | `Concluded, not seen or heard:` |

The list is flat in storage and **grouped on render** (`Entry::render`), so the
character reads an entry sorted by how each thing came to be known.

### 1.1 What the model supplies, and nothing else

The model supplies claim text, citations, optional intentions and open items with
a `relates` link. It never supplies the `kind`, never grades its own certainty,
and has no `unsure` option: the journal is deterministic.

* **Citations** are an enum over the turn ids of the covered span, compiled per
  call into the tool's grammar. A turn that is not citable
  (`window::Origin::citable()` — ambient idle lines, `nudge`, `reachable`,
  `heartbeat`, `sleep`, `wake`, the character's own `reflect`) is not in the enum,
  and a citation to one that slipped through is refused (`NotCitable`).
* **`cite` is one enum and `also_cite` a second**, not a list. The grammar guides
  an object inside an array and one array level; an array inside an array element
  decodes unconstrained, so a list of citations would be the one place the
  model writes freely.
* **`relates`** on an open item is `new`, `restates N` or `resolves N`, an enum
  over the live open-item ids.

### 1.2 What the server derives

* **`kind`**: read off the cited turns' `Origin`, never off wording. An event
  tagged `speech`, `message`, `announcement` or `operator` is `Heard`; any other
  event, and an act that is a body answer (a `scan`, say), is `Observed`; any other
  act of the character's is `Did`. A claim with several citations takes the
  strongest support (`Observed` over `Heard` over `Did`); a claim citing nothing
  the world gave is `Inferred`. The model cannot promote a rumour to an
  observation by calling it one.
* **`perishable`**: true for every kind except `Did`. What the character did does
  not stop having been done; anything else about the world may stop being so
  without the character noticing. A perishable line renders with `(re-check)`.
* **Typed claims** `{subject, attribute, value}` are checked at commit against
  what the world reads now (§4), compared without regard to case. A match is
  kept. A mismatch is replaced by what the world says and marked `corrected`. A
  claim the world cannot answer (`None`) is demoted to prose.
* **Dedup**: typed claims are unique per `(subject, attribute)`; the newest wins.
* **Open items**: `restates` and `resolves` must name an item that is open, and an
  item may not be both in one entry. At most `MAX_OPEN` (6) items are open; past
  that the oldest is dropped.
* **Limits**: at most `MAX_CLAIMS` (12) claims and `MAX_INTEND` (3) intentions.
  An entry with nothing in it is refused (`Empty`).

Every verdict (kept, corrected, demoted, superseded, refused) is reported on the
`Outcome`. The same output against the same span, open items and world always
draws the same verdict; nothing promises bit-identical replay of the model.

## 2. Drafting

Drafting runs **on the character's own live conversation**, asked the way the
guardian asks (`Minds::ask`, through `Sequence::ask_unsealed`). The character reads
its own system prompt and the same selected history it reads when it acts, so the
stretch being written about is already in its KV cache and a question costs only
its own tokens. There is no drafting conversation: nothing is opened, nothing is
tombstoned, and no stance or tool selection is swapped. Neither the question nor
its answer is written back to the live conversation; the entry reaches the
character only through the journal section collection once it has been kept (§6).

Whether to draft is the guardian's call. On each scan the `journal` module asks
the character whether the stretch waiting (`JournalState::waiting`) needs an entry
given what happened and what its journal already says (the journal is in its
system prompt). `Runtime::decide_journal` acts on the answer: a yes spawns the
draft, a no closes the stretch (`nothing_to_write`) and records the character's
reason as a `nothing` draft. The manual route (§7) and the pulse's forced draft
skip the question.

Each question is put under a stencilled one-call grammar. The turn opens inside
the grammar, so a reasoning block is unrepresentable and the answer is exactly one
call. The write's grammar depends on the stretch (its `cite` enum holds that
stretch's turns), so it is compiled the first time the write is asked and reused
for the rest of the draft.

A draft:

* is **single-flight per character** — `JournalState::pending` is set by `begin`
  and a second trigger while one is running is not given a span;
* cannot reach the world: `journal_write` is not in the act catalog, `act::parse`
  never accepts it on a live turn, and its call exists only as an answer grammar;
* **holds the character's conversation lock for each decode**, as the guardian
  does, so a draft is put between the character's turns and never inside one. The
  character can wait on a draft for the length of one decode; the draft takes the
  journal state's lock only to read it and to record how it ended, never across a
  question to the model;
* is told the stretch in the write question as numbered text, because a claim cites
  a turn by number and the live history is not numbered. The live history is the
  provenance-selected history, so it is not exactly the span; the write question's
  numbering is what a citation is checked against.

The drafts' queries are not distinguished from the character's own in the pulse.

## 3. The workflow

`workflow::run(desk, state, span, world) -> (Outcome, Timing)`. Plain Rust; each
model decision is a closed question held to a grammar. It is written against the
`Desk` trait (`gate`, `write`, `keep`), so the whole workflow runs in tests against
a script and `EngineDesk` is only plumbing.

**Step 0.** A span with no citable turns is a mechanical `Nothing`; the model is
not asked.

**Step 1 — the gate.** An `answer` call with `value` in `yes` / `no` and a required
`reason`. The question names the stretch ("your last *n* turns, from … to …") and
does not repeat it: the character reads the turns where it already holds them. It
is relative to the journal already in the system prompt. One attempt: the grammar
leaves nothing to retry.

* The gate is always asked and its answer is the model's own: nothing forces a
  `yes`, however many `no`s have come before.
* A `no` is final for its stretch: `covered_to` moves to the end of the span, so
  the next draft is asked only about what has happened since. A draft starts once
  `EVERY_TURNS` (16) new turns have landed since the last look, never on each turn.
* The reason is kept: `Outcome::Nothing { why }` and the draft record's `detail`.

**Step 2 — the write.** If the gate said yes, a second question carries the
numbered stretch and describes `journal_write` part by part (the call is not shown
anywhere else), and is held to the per-draft `write_spec` grammar: exactly one
`journal_write` call. At most `MAX_ROUNDS` (4) answers.

* `journal_write` is the entry, verified per §1. **Write is terminal**: a write
  that passes is kept and the draft ends; the model is never asked whether it is
  finished.
* A write that is refused, or a decode that does not read, is asked again with the
  previous answer and the refusal reason appended, because each question is put
  afresh and the character otherwise would not know what to change.
* There is no read and no decline: the character's open items and newest entries
  are in its prompt, and a gate that said yes has committed to an entry.

**Outcome.** `Wrote{entry, notes, rounds}`, `Nothing{why}`, or `Abandoned(Abandon)`
where `Abandon` is `Asking` (the model could not be asked), `Refused` (refused
until the rounds ran out) or `Keeping` (it passed and could not be kept).
Abandoning leaves `covered_to` unchanged, so the next trigger drafts the same
turns again. `abandon()` is idempotent, so a dropped future that never reached the
end of `run` is settled by the runtime's own `settle` and cannot leave `pending`
set.

**Timing.** `run` also returns `Timing{gate_ms, write_ms}`: the time asking the
gate and the time on every answer to the write, refused ones included. The draft
record carries both beside `took_ms`.

## 4. What the world can say

Typed claims are verified against a `World`, answering
`read(subject, attribute) -> Option<String>`. The runtime supplies a `Snapshot`
(`world.rs`) taken from the sim **before** the draft starts, so the draft reads a
still picture however long it takes and the world's lock is never held across a
decode. A subject or attribute it does not hold reads as `None`.

What the snapshot holds is the tower and its stock only:

| Subject | Attribute |
|---|---|
| `tower` | `posture`, `depth`, `shields`, `energy_per_minute`; `besieging` only while there is a siege; `minutes_of_energy` only when the energy runs out |
| `stockpile` | one attribute per resource, zeros included |

It does not read `Sim::reading` (the station readings) or the ledger. A claim
about anything else stays prose. Widening the snapshot is the way to put more of
the world under the check.

## 5. State

Per character, on the `Inbox` beside the window and the day
(`JournalState`, `state.rs`):

| Field | Meaning |
|---|---|
| `entries` | the newest `IN_PROMPT` (5) kept entries, newest last: the ones the prompt carries |
| `next_entry`, `next_item` | id counters; `written()` is `next_entry - 1`, every entry ever kept, including those aged out |
| `open` | live unresolved items, at most `MAX_OPEN` (6) |
| `covered_to` | the last window turn id settled by a kept entry or a gate `no`; an abandoned draft leaves it |
| `looked_to` | the last turn id a draft in flight was handed; set by `begin` |
| `pending` | single-flight flag |

The substrate is authoritative: the record in the character's day conversation's
metadata (§6) is what the in-memory state is rebuilt from (`Minds::restore_journal`,
`JournalState::restore`) when a character's loop first runs with an engine.
**`restore` resets `covered_to` to 0**: the entries remember which window turns
they covered, but the window's turn ids restart at 1 on every run of the daemon,
so an old run's ids would otherwise point past the end of the new window and
nothing would be drafted until the ids caught up.

### 5.1 Cadence

A stretch is waiting (`JournalState::waiting`, read by the guardian's scan) when
either holds:

* `EVERY_TURNS` (16) turns have landed since `looked_to`, the newest turn the
  last draft was handed;
* the window is within `EVICTION_MARGIN` (4) turns of full and its oldest turn is
  one no draft has been handed yet, so a turn is not lost before it was looked at.

Neither holds while a draft is pending. The first draft of a run waits
`EVERY_TURNS + npc_id % EVERY_TURNS` turns, so the cast does not draft in the
same tick.

A draft's span is every turn after `covered_to`, not after `looked_to`. A gate
`no` moves `covered_to` past the stretch; an abandoned draft leaves it behind, is
not retried until new turns land, and the next span is then the longer stretch
that includes the one not written.

**Day rollover.** `Scheduler::roll_day` strands the unwritten turns
(`JournalState::strand`) **before** it clears the window, so a day's last stretch
is still waiting afterwards and the guardian asks about it like any other.

## 6. Persistence and injection

The journal is a **section collection of the character's own conversation**,
built the way the mission is. `projection.yaml` declares `journal_intro` (the
framing: this is your own record of an earlier time), a `journal` collection
(`always_visible`, `sections: []`) and `journal_none` (`depends_on_absent` on the
collection), so a character with no entries reads "journal is empty". At install
(`identity::install`) the collection gets a Named selector and a stand-in member;
`carry_journal` selects the stand-in while nothing is sealed under it and every
member once something is.

* **One section per entry** (`journal/<id>`), the newest `IN_PROMPT` (5) of them,
  and one for the open items (`journal/open`, absent when nothing is open).
  `Minds::prepare` reconciles them before each turn through the same `Carried`
  reconcile the mission uses (`mind/carried.rs`): what the prompt wants and is not
  yet held is submitted and sealed; a section the prompt no longer wants is
  removed (`remove_section_named`). An entry that falls off the newest five is
  therefore removed from the conversation.
* **Sections are the conversation's.** They are sealed and persisted with it and
  released when it is tombstoned (`tombstone_timeline`), so a journal is gone
  exactly when its conversation is. A day roll-over or `retire_superseded`
  tombstones the old day conversation; `prepare` writes the journal into the new
  one on its first turn, so the journal is carried across days.
* **Sealed KV cannot be re-rendered at gather time**, so an entry's time is baked
  in as an absolute stamp (`[14 Jun 2187, 14:20–14:41]`, or both ends in full
  across midnight), never "twenty minutes ago". The date is the world's own
  calendar, year offset included (`clock::WorldTime`), so a world set 217 years
  ahead stamps 2187, not 1970. Anything marked `(re-check)` may have
  changed, and a line under "Concluded, not seen or heard" was worked out, not
  witnessed.
* **Keeping an entry** (`Minds::keep_journal`, the `Desk`'s keep) reconciles the
  live conversation without waiting for a seal and then writes the record. A
  failure ends the draft `Abandon::Keeping` and the entry is not kept.

### 6.1 The record a restart rebuilds from

A submitted section's text cannot be read back from the substrate, so the same
entries are kept as JSON in the day conversation's metadata: `journal.of`
(the character's id) and `journal.kept` (entries, open items, `next_item`).
`Minds::record_journal` rewrites it when it changed. The record lives in the
conversation's metadata, so tombstoning the conversation drops it.

On first run with an engine, `Minds::restore_journal` finds the conversations
with `journal.of = <npc id>` (tombstoned ones are excluded), takes the newest
record and rebuilds `JournalState`. A rejoin reuses the same day conversation and
resubmits the same names and text, which restores the sealed streams without
prefilling them again.

### 6.2 Private

A character's journal sections are in its own conversation and no other's; there
is no cross-character scope to get wrong.

### 6.3 `--forget-journal`

`npcd --forget-journal` empties the record (`journal.kept`) of every conversation
holding a character's journal at load (`Minds::forget_journal`) and logs the
count. The conversation is not tombstoned, so sections already sealed in it stay
until it is, but nothing selects them once the journal is empty. Without the flag a
journal is never discarded.

## 7. Observability

`Census` (the scheduler's per-tick view, read by the pulse and the console)
carries `journal_written` (`written()`), `journal_open` (open items) and
`journal_drafting` (`pending`). A kept entry logs
`npc <id>: journal entry <n> kept in <t> after <r> round(s)…`; an abandoned draft
logs why.

Two routes, both `user` with the ownership check (the GET also takes `?all=true`
for an admin), expose a character's journal; a third shows its system prompt:

- `GET /v1/npc/:nid/journal` returns `written`, `in_prompt` (the
  entry ids the next prompt carries), `covered_to`, `looked_to`, `drafting`,
  `open` (the open items), `entries` (each entry's stored
  form plus its `span` and the `text` it reads as in the prompt) and `drafts`.
  503 `not_awake` for a character not in the scheduler. An admin may add
  `?all=true` to read a journal they do not own, as `/v1/pulse?all=true` does;
  for anyone else the flag changes nothing and the ownership check applies.
  404 `npc_not_found` for an id that is not a living character.
- `POST /v1/npc/:nid/journal/draft` drafts now, over every turn after
  `covered_to`. 202 with `{from_turn, to_turn, turns}`; the outcome appears in
  `drafts`. 409 `nothing_to_draft` while a draft is running or when nothing is new.
- `DELETE /v1/npc/:nid/journal/:eid` forgets one entry (owner only). The entry
  leaves the journal state, its section is removed from the character's
  conversation, and the metadata record a restart rebuilds from is rewritten
  without it. Entry ids are not reused and open items are kept. Answers the
  journal as it now reads; 404 `no_such_entry` when the entry is not one of those
  held (only the newest few are).

`GET /v1/npc/:nid/system-prompt` (same ownership rule and `?all=true`) returns the
system prompt the character's latest turn was made under, as `pieces` (name, kind
`glue`/`section`/`member`, text) and the `text` they make end to end. It is laid out
from that turn's own projection, so it shows which members were selected, the
`mission/carrying` one included. 404 `no_system_prompt` until the character has had
a turn since the daemon started.

`GET /v1/npc/:nid/prompt-tokens` (same ownership rule and `?all=true`) returns the
tokens the character would be given if it decoded at the moment of the call. The
projection is recomputed then, not read back from the last turn, so it is what the
next turn starts from. `pieces` are in injection order — each a `section` (named
`collection/section`, with its `section_id`), a sealed `turn` (with `timeline`,
`index`, `role`), live template `glue`, or the pending `user_message` — carrying its
decoded `text`, `token_count`, and the `score` and `qualified` flag of
the belief that selected it. Token ids are not serialized; the text is the decode of
exactly those tokens. `token_count` and `text` at the top level are the pieces joined. Two sections of the same name under different `section_id`s
show as two pieces. 404 `no_conversation` for a character with no live conversation.

`drafts` holds the last 16 (`KEPT_DRAFTS`) as `DraftRecord`s: the turns covered,
`took_ms` split into `gate_ms` and `write_ms`, `result` (`wrote`, `nothing`,
`abandoned` or `not_started`), a `detail` saying why when there is no entry (the
character's own reason for a gate `no`), the `entry` id, `rounds`, and `checks` — each
claim's verdict (`kept`, `corrected`, `demoted`, `superseded`) and `kind`. The
log is in memory only and starts empty on each run.

## 8. Failure and privacy

* A mind without a `journal` collection in its projection journals nothing, and is
  not an error: a draft against such a mind is abandoned with a warning.
* Without a `guardian.yaml` carrying `kind: journal`, or with `enabled: false`, no
  one asks whether a stretch needs an entry, so nothing is journaled except by the
  manual route.
* A draft is asked on the character's own conversation, which reads the journal of
  that character and of no one else.
* A crash after a day roll-over opens the new conversation but before its first
  record write loses the journal the old conversation held.
* No failure of the journal stops the character acting: every failure above ends
  in an `Abandon` that is logged, and the character's loop does not wait on a
  draft.

## 9. Tests, and what only a live run can say

Covered by unit and scripted-`Desk` tests: window ids and origins; cadence, the
eviction trigger and the stagger; the `Carried` reconcile plan (submit, remove,
resubmit after a failed seal, sweep); the journal record round trip; the guardian
journal module's question and judgement; the citation enum and refusal; kind derivation; perishability; typed-claim
correction, demotion and dedup; the open-item cap and `resolves`; the gate
answered `no` any number of times; `covered_to` on a keep, a `no` and an abandonment; restore
resetting `covered_to`; `written()` surviving eviction from memory; single-flight
through `begin`; stranding at day roll-over; the census; the `Snapshot` reads;
`journal_write` absent from the world catalog and refused as unknown by
`act::parse`; the write question's wording; a refused attempt coming back
with its reason.

npcd has no engine-level test harness, so `Minds::prepare`, `keep_journal`,
`restore_journal` and `Runtime::decide_journal` need a live run. Only a live run
can say:

* the KV cost of submitting and removing journal sections between turns;
* what the gate costs: it is asked on the live conversation under
  `ask_unsealed`, and its latency is measured in seconds, not tokens;
* how long a decode holds the character's lock, and whether the character
  noticeably waits on a draft;
* whether the character answers the gate with `no` too often or too rarely;
* the order equal-score groups are emitted in;
* the real stencil compile cost of the per-draft grammar.
