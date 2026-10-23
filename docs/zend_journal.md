# zend's journal

**Status: built.** It adapts npcd's journal (`docs/journal.md`, built for
Battle Cities characters) to zend dialogues, and it closes the `TODO.md` item
"Implement the journal tool". This document is authoritative over
`zend/src/journal/`, `journal-core/` and the journal's wiring in
`zend/src/session.rs`.

## 1. Why

A coding session runs longer than the dialogue projection can hold whole.
Provenance selection keeps the turns the current probe scores well and drops
the rest from the prompt. A decision made two hours ago, a file found to be the
real culprit, or a test that was failing before the last edit is gone from the
model's line of thought unless the probe happens to retrieve it again.

zend's summary tree was meant to bridge that gap. It is switched off
(`candle-conversation/src/summary_tree/mod.rs`) because it made **false
memories**: measured on a 16-turn conversation, 5 of 9 summary nodes were
unfaithful. They were written in the first person, as if they were the reply,
so a later projection read a wrong one as something the model had actually
said.

The journal fills the same gap and avoids each cause of that failure:

| What went wrong with summaries | What the journal does instead |
|---|---|
| Written in the reply's voice, so they read as history | Framed as the model's own **dated notes**, in a section of the system prompt, never as turns |
| Nothing tied a sentence to where it came from | Every claim **cites turn parts**; the kind of claim (saw, did, heard, concluded) is derived by the server from what it cites, never from the wording |
| Nothing checked a sentence against the world | Typed claims about the **workspace** are checked against it when the entry is kept, and corrected or demoted when they are wrong |
| Compressed every exchange, needed or not | A **gate** asks whether a stretch needs an entry at all; a `no` is final for that stretch |

The real turns stay in the substrate, and provenance still retrieves them. The
journal adds what selection cannot: a short, always-present record of what the
session has established and what is still open.

## 2. One journal, two products

The journal's logic is one crate, **`journal-core`**, shared with npcd. It has
no engine dependency, so it builds and tests in seconds:

| File | Concern |
|---|---|
| `entry.rs` | `Entry`, `Claim`, `Kind`, `Item`; the prose an entry renders to |
| `verify.rs` | the server's check: citations, kind, perishability, typed claims against a `World`, open-item rules |
| `tools.rs` | the `journal_write` call as flat `{name, params}` JSON a stencil `ToolSpec` reads, and reading an answer back |
| `repeats.rs` | what the journal already says, dropped before a keep (70% word overlap) |
| `ask.rs` | the question a stretch is put as, and a refused answer sent back with its reason |
| `workflow.rs` | `run`: the write (at most `MAX_ROUNDS`, 4, answers), the check, the keep, against a `Desk` |
| `journal.rs` | `Journal`: entries, open items, coverage, ids; the `Holds` trait a caller's state implements |
| `section.rs` | the journal as system-prompt sections, and the metadata record a restart rebuilds from |
| `record.rs` | what each draft did, and the API's view |
| `style.rs` | `Style`: every word an entry and a question render with |

The section reconcile — submit what is wanted and not held, remove what is held
and not wanted — is `candle_conversation::Carried`, beside the `Sequence` it
works on. Each product supplies only what is its own: how a turn's origin maps
to a claim kind, the `World` typed claims are checked against, its `Style`, its
`Desk`, and when a draft is due.

## 3. An entry, for coding

An entry covers a stretch of the dialogue. It holds:

| Field | What it holds | Limit |
|---|---|---|
| `claims` | what the session established, each with citations and a server-derived kind | `MAX_CLAIMS` 12 |
| `intend` | what the model means to do next | `MAX_INTEND` 3 |
| `opened` | items left unresolved, each given an id by the server: a failing test, an unanswered question, a step not yet done | `MAX_OPEN` 6 live |
| `resolved` | ids of earlier open items this entry settles | — |

### 3.1 Citations are parts

A substrate turn holds both halves of one exchange — the user side and the
assistant's reply — and they support a claim differently, so each half is a
**part**, numbered from the turn's own index (`zend/src/journal/origin.rs`):
`2·turn + 1` for the user side, `2·turn + 2` for the reply. The numbering is the
conversation's, stable across restarts and members, so a journal restored from
its record carries on from the last part its entries covered. Part 0 is no part:
a journal that covers nothing covers part 0.

The write question shows the stretch as numbered parts, each clipped to 400
characters on one line, because the live history is not numbered; the draft's
fork already holds every turn in full. A reply's reasoning block is left out.

### 3.2 Kind, derived from what is cited

| Kind | Cited part | Rendered under |
|---|---|---|
| `Observed` | a user side carrying `<tool_response>` blocks: a file read, a grep, a command's output | `Saw:` |
| `Heard` | a user side carrying a person's message | `User said:` |
| `Did` | a reply: the call it made, or what it told the user | `Did:` |
| `Inferred` | nothing the session saw supports it | `Concluded, not seen:` |

A claim with several citations takes the strongest support (`Observed` over
`Heard` over `Did`). The model cannot promote a guess to an observation by
calling it one.

`Did` lines are not perishable: an edit, once made, was made. Every other line
renders with `(re-check)`, because the file, the test result or the branch may
have changed since.

### 3.3 Typed claims, checked against the workspace

A claim may carry `{subject, attribute, value}`. It is checked when the entry is
checked, against the conversation's own files — its branch in each repository
and the changes its tool rounds made there (`zend/src/journal/world.rs`):

| Subject | Attribute | Read as |
|---|---|---|
| `file:<repo>/<path>` | `exists` | `true` or `false` |
| `file:<repo>/<path>` | `blob` | the git blob id, while the file is the branch's |
| `branch:<repo>` | `head` | the commit the conversation's branch is at |
| `branch:<repo>` | `name` | the branch the conversation reads the repository through |

- A match is kept.
- A mismatch is replaced by what the workspace says, and marked `corrected`.
- A claim the workspace cannot answer becomes prose. So does a file's `blob`
  once the session has changed the file: its bytes are no blob yet.

Test results and exit codes are deliberately **not** typed claims. They are
facts about a past run, cited as `Observed` from the tool response that
reported them, and they render with `(re-check)`.

## 4. The journal in the prompt

The shared system prompt in `zend/src/prompts/projection.yaml` declares, last
before `system_close`:

- `journal_intro`: the framing, `depends_on: journal`.
- `journal`: a collection with `sections: []`.

`journal::carry::install` scopes the collection to the members a conversation
names (`SelectionRule::Named`). It has no member of its own: the sections are
the conversation's — one per entry (`journal/<id>`) for the newest `IN_PROMPT`
(5) entries, plus `journal/open` while any item is open — submitted to its own
timeline.

- **A round pins exactly the sections that have sealed.** A conversation that
  has written nothing, or whose sections are all still sealing, pins none, so
  neither the collection nor its intro emits, and every other layer's prompt
  reads exactly as if no journal were declared. There is no `journal_none` and
  no placeholder: the prompt never shows a journal heading without the entries
  under it. (A placeholder member was tried and dropped: a reply that read it
  instead of the entries, asked what its journal held, made them up.)
- **At the head of every round** the conversation's sections are reconciled with
  its journal (`carry_journal`). Nothing waits for a seal: a new entry joins the
  prompt the first round after its section seals, and until then the turns it
  was written from are still in the conversation.
- **Last before the close**, so a new entry changes nothing above it.
- The collection is not a section sink: `ingest::section_sinks` never fills it
  from a `journals/` folder.

**The framing is what keeps it from reading as history.** It names the sections
as notes the model wrote, each line under how it came to be known, and says
that `(re-check)` lines may have changed. Sealed K/V cannot be re-rendered at
gather time, so each entry carries an absolute time
(`[2026-10-23 14:20–14:41 UTC]`), never "twenty minutes ago".

**Budget.** The system prompt has none; the journal is bounded by its shape —
five entries of at most twelve short claims, three intentions and six open
items.

## 5. Drafting

**Drafts run on the dialogue's own conversation**, through `UnsealedAsker`
(`zend/src/journal/desk.rs`). Each question runs on an ephemeral fork of the
live slot:

- it reuses the K/V the dialogue already holds, so the stretch being asked
  about costs nothing to prefill;
- it reads under the selection the conversation's last round ran under, journal
  included;
- nothing is written back, neither the question nor the answer;
- the scheduler parks the fork until any turn in flight ends, so a draft never
  runs inside a turn.

Each answer is held to one call of the dialect's own shape, the reasoning block
written closed, decoded greedily (`zend/src/journal/draft.rs`):

1. **Nothing citable** is a mechanical `Nothing`; the model is not asked.
2. **The gate:** one `journal_gate` call — `answer` `yes` or `no`, and a
   `reason` — judged against the journal already in the prompt. A journal that
   holds no entry is not asked: there is nothing to weigh the stretch against.
   A `no` covers the stretch.
3. **The write:** one `journal_write` call under the per-draft grammar, whose
   `cite` enum holds the stretch's citable parts, at most four answers; a
   refused answer is asked again with the refusal appended. What the journal
   already says is dropped, and an entry of nothing else covers the stretch as
   a `no` would.
4. **The keep:** ids are staged, the record is written to the conversation's
   metadata, and only then is the entry in memory — a record that fails to
   write leaves the journal as it was. The sections follow at the next round.

`journal_write` and `journal_gate` are never in the dialogue's tool catalog.
They exist only as answer grammars, so the model cannot write an entry on a live
turn.

### 5.1 Priority: drafts never compete with the user

A dialogue decodes at `High` and the background layers at `Low`. A journal
draft runs at **`Normal`**, below the user's turns and above the layers, so
launched coding tasks run first, drafts follow them, and the background
analysis comes last.

`ForkEphemeralSequence` and `UnsealedAsker::ask` take the fork's
`DecodePriority`, and the scheduler resolves a fork — and any turn view on it —
to that band in place of its parent layer's (`scheduler/ephemeral_fork.rs`).
Every asker names its band: the journal `Normal`, npcd's guardian and journal
questions `Normal`, zend's `--self-check` `Low`.

## 6. Cadence

zend counts sealed dialogue turns (each tool round seals one)
(`zend/src/journal/state.rs`, `pressure.rs`). After each round's turn seals, a
draft starts when either holds:

- **`EVERY_TURNS` (16)** turns have sealed since the last draft was handed a
  stretch.
- **Selection pressure:** the round's projection left out a turn of this
  conversation that no entry and no `no` covers. The projection event a round
  records already names every turn it selected, so the signal costs nothing. A
  turn is looked at before the session loses sight of it.

At most one draft runs per conversation (single flight). A draft's stretch is
every part after `covered_to`. A `no` or a kept entry moves `covered_to` to the
end of the stretch. An abandoned draft leaves it where it was, so the next draft
retries those turns.

There is no idle trigger. A quiet conversation's uncovered turns are still in
the substrate, whole, and the next trigger after the session resumes covers
them.

## 7. Persistence, replication and privacy

- **Sections are the conversation's.** They are sealed and persisted with it,
  replicated with it, and released when it is tombstoned.
- **A submitted section's text cannot be read back from the substrate**, so the
  entries are also kept as JSON in the conversation's metadata (`journal.kept`).
  This is the record a restart, or the member a conversation is handed to,
  rebuilds from; `covered_to` is restored to the last part the newest entries
  cover. A member that lets a conversation go (`turn_claims`) drops its journal
  and reads the record again when it next holds the conversation.
- **A forgotten entry** leaves the record at once and its section at the
  conversation's next round. One a restart left sealed is never selected — the
  collection is read by name — and goes with the conversation.
- **Private.** A journal is read by its own conversation and nothing else: no
  other conversation retrieves its sections, and no layer (`daily_history`
  included) reads them.
- **Only a dialogue journals.** zend has no dialogue fork; the titler,
  passthrough conversations, calibration and the ingest layers never journal.

## 8. Observability

- `GET /v1/conversations/:id/journal` returns the entries (stored form, span and
  rendered text), the open items, `covered_to` and `looked_to` (parts), whether
  a draft is running, and the last 16 draft records. Each record carries
  `gate_ms`, `write_ms`, the result, and each claim's verdict and kind.
- `POST /v1/conversations/:id/journal/draft` drafts now over whatever is not
  covered: 202 with the turns it covers, or 409 — `nothing_to_draft` while a
  draft runs or when nothing is new, `not_open` when the conversation is not
  open on this member (a draft forks its live slot).
- `DELETE /v1/conversations/:id/journal/:eid` forgets one entry: the journal as
  it now reads, or 404 `no_such_entry`.
- Log target `zend::journal`: each draft's start, and its end with stretch,
  outcome, timings and verdicts.
- **GUI:** the Journal pane on the conversation lists the entries, each claim
  with its kind and whether the workspace corrected it, the open items, and the
  recent drafts, with a Forget on each entry and a Draft now. Entries are not
  editable: every line in the journal is one the server cited and checked.

## 9. Code layout

| Where | What |
|---|---|
| `journal-core/` | the journal's logic (§2) |
| `candle-conversation` `conversation/carried.rs` | `Carried`, the section reconcile |
| `candle-conversation` `scheduler/ephemeral_fork.rs` | a fork's band |
| `zend/src/journal/origin.rs` | the dialogue's turns as citable parts |
| `zend/src/journal/state.rs` | `SessionJournal`: a journal and its cadence |
| `zend/src/journal/pressure.rs` | selection pressure from the projection event |
| `zend/src/journal/style.rs` | `CODING`: UTC dates and a coding session's words |
| `zend/src/journal/gate.rs` | the `journal_gate` call |
| `zend/src/journal/world.rs` | `ConversationFiles`: the workspace typed claims are checked against |
| `zend/src/journal/desk.rs` | `ZendDesk`: questions on forks at `Normal`, the record write |
| `zend/src/journal/draft.rs` | one draft: the gate, then the write |
| `zend/src/journal/carry.rs` | the schema install, the wanted sections, the round's selection |
| `zend/src/journal/record.rs` | the metadata record |
| `zend/src/journal/registry.rs` | `Journals`: every dialogue's journal, and what a draft forks from |
| `zend/src/api/journal.rs` | the three routes |
| `zend/web/index.html` | the Journal pane |
| `npcd/src/engine/journal/` | npcd's origin mapping, window cadence, voice, sim `World` and `EngineDesk` |

## 10. Failure

- No failure of the journal stops the dialogue. Every failure ends in a logged
  abandon or a recorded draft, and the user's turn never waits on a draft beyond
  the band sharing the device.
- A schema with no `journal` collection journals nothing; that is not an error.
- A record write that fails abandons the draft: the journal stays as it was and
  the same turns are covered again.

## 11. Tests

Built alongside the code, each building block testable alone:

- **`journal-core`**: the scripted-`Desk` workflow, the check, the grammar, the
  record, repeats; npcd keeps passing through the move.
- **Origin:** each half of a turn maps to its part, citability and kind;
  reasoning is left out; long parts are clipped.
- **Workspace `World`:** against a temporary git repository, a file's existence
  and blob, a changed file, a branch's head and name, and every question it
  cannot answer.
- **Cadence:** every-16, selection pressure, single flight, and `covered_to`
  after a keep, a `no` and an abandon; a restored journal carries on.
- **Draft:** gate skipped on an empty journal, a yes, a no, an unreadable or
  unaskable gate, a write refused out of rounds.
- **Band:** a fork made at each band decodes at it, a turn view on it too, a
  parked request keeps its band, and the band goes with the fork.
- **Prompt:** the embedded schema declares the journal, the install scopes it,
  an empty journal pins nothing, and the journal is never a section sink.
- **Routes:** the table and the dispatch of the three routes.

Only a live coding session can say:

- how often the gate says `no`;
- how much a draft slows the user's next round;
- whether five entries are enough for a long task;
- whether the model actually uses the open items and `(re-check)` marks.

## 12. Relation to the summary tree

Once the journal is live and has carried real long sessions, the dormant
summary tree (`candle-conversation/src/summary_tree/`) and the `summary:`
blocks in `projection.yaml` have no remaining purpose, and they are removed
then — not before, so the journal is proven first.

## 13. Decisions

Settled with the user on 2026-10-23:

1. **One shared crate.** npcd's core moved into `journal-core` (§2); npcd and
   zend each keep only their own origin mapping, `World`, `Desk` and style.
2. **The summary tree goes once the journal is proven** in real long sessions
   (§12).
3. **Drafts run at `Normal`**, below the user's coding tasks and above the
   background layers (§5.1).
4. **The gate stays.** A `no` is cheap, and it stops the journal filling with
   "nothing new".
5. **No idle trigger** (§6).
6. **Private to its conversation** and tombstoned with it; no layer reads it
   (§7).
7. **A read-only GUI pane with delete** (§8).
8. **Only the parent dialogue journals**, not sub-runs or agent jobs (§7).

The starting numbers, `EVERY_TURNS` 16 and `IN_PROMPT` 5, are guesses to tune
in live sessions.
