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
| Nothing checked a sentence against the world | Where the world can read what a claim is about — npcd's game state — a claim's **typed fact** is checked against it when the entry is kept, and corrected or demoted when wrong. A coding journal offers no facts (§3.3) |
| Compressed every exchange, needed or not | Once the journal holds an entry, the draft may answer **`journal_skip`** instead of writing; a skip is final for that stretch |

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
| `tools.rs` | the `journal_write` and `journal_skip` calls as flat `{name, params}` JSON a stencil `ToolSpec` reads, and reading an answer back |
| `repeats.rs` | what the journal already says, dropped before a keep (70% word overlap) |
| `ask.rs` | the question a stretch is put as, the skip offered beside an entry, and a refused answer sent back with its reason |
| `workflow.rs` | `run`: write or skip in one answer (at most `Desk::rounds` answers — `MAX_ROUNDS`, 4, unless the desk asks fewer), the check, the keep, against a `Desk` |
| `cut.rs` | `close_cut`: an answer its token budget cut off, closed so what it said is read |
| `journal.rs` | `Journal`: entries, open items, coverage, ids; the `Holds` trait a caller's state implements |
| `section.rs` | the journal as system-prompt sections, and the metadata record a restart rebuilds from |
| `record.rs` | what each draft did, and the API's view |
| `style.rs` | `Style`: every word an entry and a question render with |

The section reconcile — submit what is wanted and not held, remove what is held
and not wanted — is `candle_conversation::Carried`, beside the `Sequence` it
works on. Each product supplies only what is its own: how a turn's origin maps
to a claim kind, the `World` typed claims are checked against and the entries
above a claim may rest on, its `Style` (whether claims carry facts), its
`Desk`, and when a draft is due.

## 3. An entry, for coding

An entry is one **step** of the session — a task taken on, or a major step in
it — and covers the stretch of the dialogue that step took, which grows while
the session stays on it (§5). It reads two ways: in the prompt, one line — its
number and its `summary` (`Entry::render`); asked for, everything below
(`Entry::detail`). It holds:

| Field | What it holds | Limit |
|---|---|---|
| `summary` | the task or step, in one sentence — the line the prompt shows | — |
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
| `Heard` | a user side carrying a person's message | `User said:` — `Claude said:` in a task delegated through `/mcp`, `Its worker said:` in a part (`style::style_for`) |
| `Did` | a reply: the call it made, or what it told the user | `Did:` |
| `Inferred` | nothing the session saw supports it | `Concluded, not seen:` |

A claim with several citations takes the strongest support (`Observed` over
`Heard` over `Did`). The model cannot promote a guess to an observation by
calling it one.

`Did` lines are not perishable: an edit, once made, was made. Every other line
renders with `(re-check)`, because the file, the test result or the branch may
have changed since.

### 3.3 No typed facts; entries above, by `per`

A coding journal's claims carry **no typed facts** (`Style::facts` is off for
`CODING`, `DELEGATED` and `PART`): the call offers no `typed` slot, and the
write question never mentions one. What a coding claim is about — what a file
says, why a test failed, what was decided — is nothing a check can read, and
the workspace questions that could be read (does a file exist, which blob,
which commit) never earned their place: live drafts filled the slot with
facts about something else — every claim typed `file:… exists` with a line
number for its value, and a web-research claim typed with an unrelated
`README.md` — which the check, reading only the fact, confirmed. A claim rests
on the turns it cites; a test result or an exit code is `Observed` from the
tool response that reported it, and renders with `(re-check)`.

A claim of a conversation with journals above it — owned through `/mcp` — may
name the entry it rests on in **`per`** (`T:review#4`). The call offers exactly
the entries those journals hold (`World::entries`, `JournalsAbove` in
`zend/src/journal/world.rs`), as `cite` offers exactly the stretch's turns, so
an entry that is not there cannot be written; one that arrives by any other
path is dropped and the claim stands on its turns (`Demoted`), never costing
the entry. The `per` sentence of the write question is said only when the
call offers `per` (`Style::per_question`), so a writer with no journals above
is never told of a slot it does not have. The list costs the grammar nothing
as it grows: `per` is one choice whose values are a table (1,169 nodes at 10
entries above or 1,600). The claim's line reads `… (per T:review#4)`.

npcd keeps typed facts (`Style::facts` on): a character's claims are about the
world's state — energy, stockpiles, what a tower is doing — which its world
reads, and a fact the world reads differently is corrected.

## 4. The journal in the prompt

The shared system prompt in `zend/src/prompts/projection.yaml` declares, last
before `system_close`:

- `journal_intro`: the framing, `depends_on: journal`.
- `journal`: a collection with `sections: []`.

`journal::carry::install` scopes the collection to the members a conversation
names (`SelectionRule::Named`). It has no member of its own: the sections are
the conversation's — one per entry the journal holds (`journal/<id>`), plus
`journal/open` while any item is open — submitted to its own timeline.

- **A round pins exactly the sections that have sealed.** A conversation that
  has written nothing, or whose sections are all still sealing, pins none, so
  neither the collection nor its intro emits, and every other layer's prompt
  reads exactly as if no journal were declared. There is no `journal_none` and
  no placeholder: the prompt never shows a journal heading without the entries
  under it. (A placeholder member was tried and dropped: a reply that read it
  instead of the entries, asked what its journal held, made them up.)
- **The sections are a chain.** Each is sealed as the continuation of the ones
  before it (`Sequence::submit_section`'s `after`): its K/V attends to them,
  and its recurrence runs on from the state they leave — itself run on from
  the state the prompt before the collection leaves, kept once for every
  conversation on the prompt. Read straight after them, an entry is computed
  as it is read. A new entry seals the entry and the closing section; a
  fold, or a forgotten entry, reseals everything after it
  (`Linking::Chain`), and its address names what it continues, so a restart
  restores the right one. A round pins only the chain's sealed beginning.
- **At the head of every round** the conversation's sections are reconciled with
  its journal (`carry_journal`). Nothing waits for a seal: a new entry joins the
  prompt the first round after its section seals, and until then the turns it
  was written from are still in the conversation.
- **Last before the close**, so a new entry changes nothing above it.
- **A task inherits its lineage's journals.** A conversation owned through
  `/mcp` carries, in the same collection and the same chain, its ancestors'
  journals ahead of its own — the project's standing rules, its thread's notes,
  and for a part each worker's above it — as sections `lineage/<n>`, and a
  reader's line, `lineage/reader`, after its own (`mcp::lineage`,
  `docs/claude_zend_gateway.md` §5.1).
- The collection is not a section sink: `ingest::section_sinks` never fills it
  from a `journals/` folder.

**The framing is what keeps it from reading as history.** It names the sections
as notes the model wrote, each line under how it came to be known, and says
that `(re-check)` lines may have changed. Sealed K/V cannot be re-rendered at
gather time, so time is absolute, never "twenty minutes ago". The journal reads
as one running list, dated only where the date moves: `2026-10-23, 14:00 UTC`
above the first entry and above one that starts a new day, `15:00 UTC` above one
that starts a new hour, nothing within the hour. Each entry is one line,
`Entry N — <summary>`; its claims, in the order the parts they cite happened and
each labelled with how it was known (`- Saw: …`, `- User said: …`), are its
detail, kept behind the line for the inspector. The last section is always
`Still open:` — `Still open: nothing.` when nothing is — followed by a line
saying the journal ends there and that anything else known of the conversation
comes from its turns, so neither silence nor the rest of the context is taken
for something written down.

**Budget.** The system prompt has none; the journal is bounded by its shape —
one line per entry, six open items, and the calendar folding it (§4.2): today's
steps, this week's days, this month's weeks, this year's months, and a line per
year before that.

### 4.2 Folding on the calendar

The journal compresses itself as the UTC calendar turns
(`journal_core::period`, `fold`, `folding`). Once a day is over, its steps —
each entry belongs to the day it *ended* in, so a step a skip ran on into
today is today's — fold into one **day** entry; once a week (Monday to
Sunday) is over, its days fold into a **week**; then a month's weeks into a
**month**, and a year's months into a **year**, which never folds. The lowest
level folds first and the earliest period first, one fold at a time, so a
journal that slept through a month catches up period by period.

The model folds them itself, on a focused fork of the dialogue at `Normal`
like a draft: every entry of the period is put in front of it whole — a
step's claims, a folded entry's digest — and the answer is one
`journal_fold` call, a one-line `summary` the journal keeps for the whole
period and a `digest`, the detailed account behind it. The folded entry takes
the next id and the place the first entry it folded held. **It is lossy on
purpose:** the entries folded leave the journal, their sections are removed and
their text is ceased (§4.1); the folded entry's digest is filed as the text
behind it, so `journal_read` on a day entry reads its digest. Open items are the
journal's, not any entry's, and are untouched.

A conversation owned through `/mcp` is the exception: the conversations below it
cite its entries, so its fold files the folded entries' turns under the fold,
beside the digest, and notes where each went (`JournalText::refile`). Reading a
folded entry reads the fold's turns and says where the entry went, through a
later fold too (`docs/claude_zend_gateway.md` §5.2).

A fold shares the draft's single flight. It is tried after every round's seal
when no draft starts, after every draft, and every five minutes over the open
conversations (`FOLD_SWEEP_EVERY`), so a day that turns over while a
conversation sits open and quiet folds too. A skip runs on only over a step:
after a fold the latest entry is a day, and a skipped stretch is only covered.

npcd's characters fold the same way on their world's calendar, after each
draft, with the newest turn the draft covered as their now.

### 4.1 The turns behind each entry

One line per entry summarises; the turns it was written from are kept, whole,
for the dialogue to read back when the line is not enough — the exact request,
a file's contents, an error from back then. Two dialogue tools read them:

- **`journal_read {journal, entry?, offset?, limit?}`** lists a journal's
  entries, or with `entry` pages one entry's turns the way `file_read` pages a
  file: each turn opens with its number and time (`#7 14:20 User: …`), then
  runs as it was, 200 lines a page by default and 400 at most.
- **`journal_grep {journal, pattern, entry?, max_results?}`** finds the lines
  that hold some text, ignoring case, in every entry's turns or one's: each hit
  names its entry, turn and line, to read around with `journal_read`.

`journal` is mandatory: `"self"` for the dialogue's own, or — in a
conversation owned through `/mcp` — a journal above it, by its bare name or
full path (`docs/claude_zend_gateway.md` §5.3). A thread's or project's
journal is Claude's notes, with no turns behind them, so its entry reads as
the note itself.

Both are **mandatory** (`ToolDef::mandatory`), projected on every round the way
the file tools are, and the journal intro names them, so the model knows the
line is not all there is and always holds the call that reads behind it.
Reading and searching are `journal_core::transcript::Book`'s, shared with npcd;
the tools (`zend-tools/src/tools/journal/`) reach the journals through
`ToolContext::journals`, a shelf (`zend/src/journal/shelf.rs`) zend builds on
every dialogue round.

**Each stretch is filed under the entry it went into** (`transcript::Filed`):
the entry a draft wrote, or — when the draft skipped the stretch as more of the
same — the latest entry, whose span runs on over it. A stretch skipped before
the first entry exists is filed nowhere. Each is one substrate custom object,
`journal/turns/<timeline>/<entry>/<first part>`, zero-padded so key order is
entry order and then stretch order, and **written once**: extending an entry
adds an object beside the ones it holds and rewrites none. The text is not the
conversation's turns — those leave the conversation's window and are tombstoned
with it — but a copy of their words, with no K/V, that lives as long as the
entry does:

- **Forgetting an entry**, or its folding into a day, week, month or year
  (§4.2), ceases its objects.
- **Tombstoning the conversation** ceases all of them.
- **Archiving the conversation** ceases all of them and keeps the entries: the
  journal can still be read, and nothing runs a tool on an archived
  conversation again. A worker `/mcp` started is the exception
  (`archive_worker`): its turns are the evidence behind its task's outcome,
  which Claude reads later, so they are kept.

npcd's characters keep no text behind their entries: none of their tools would
read it.

## 5. Drafting

**Drafts run on the dialogue's own conversation**, through `UnsealedAsker`
(`zend/src/journal/desk.rs`). Each question runs on an ephemeral fork of the
live slot:

- it is projected under **`AskScope::Conversation`**
  (`projection::Schema::focus`): the system prompt with its journal, the
  dialogue's own turns, and from other layers only the working-set members the
  dialogue was served and still holds locked. What its rounds retrieved from
  other layers — project files, priming chains, history — is left out, so the
  entry is written from the session and not from material it only had beside
  it, and the fork carries fewer tokens;
- it reuses the K/V the dialogue already holds for the prefix the two share,
  so the stretch being asked about costs nothing to prefill;
- nothing is written back, neither the question nor the answer;
- the scheduler parks the fork until any turn in flight ends, so a draft never
  runs inside a turn.

The answer is held to one call of the dialect's own shape, the reasoning block
written closed, decoded greedily (`zend/src/journal/draft.rs`, which runs
`journal_core::workflow::run`, the same workflow npcd runs):

1. **Nothing citable** is a mechanical `Nothing`; the model is not asked.
2. **Write or skip, in one answer:** one call under the per-draft grammar —
   `journal_write`, whose `cite` enum holds the stretch's citable parts and
   whose `claims` hold at least one per two citable parts (at most six are
   insisted on: decoded greedily, a writer offered "another claim or close"
   closed after one, the request, however the question asked), or,
   once the journal holds an entry to weigh the stretch against,
   `journal_skip` with a one-sentence `reason`. **A new entry is a step
   change**: the user asking for something new, a new task begun, or a major
   step done (a fix made, a cause found, a test passing or failing, a
   decision). More reading, searching or more of the same step is skipped, and
   **a skip files the stretch under the latest entry**: that entry's span runs
   on over it (`Journal::nothing_to_write`), so its turns are found under the
   step they belong to, and its line does not change, so nothing is resealed.
   The check runs as often as before — the cadence gives the record its
   resolution — and the gate decides what is a step. **zend asks once**
   (`Desk::rounds`, 1; npcd keeps the default four): a refused write is not
   asked again, because an answer asked again runs as long as the first and a
   delegated task's caller waits on the draft. An answer its 2,048-token
   budget (`WRITE_TOKENS`) cuts off is read as far as it got
   (`journal_core::cut`): the string it was writing ends in `...`, the call is
   closed, and a last claim or open item the cut left incomplete is dropped.
   Where more than one answer is allowed, a refused write is asked again with
   the refusal appended, and may still be answered with a skip. What the
   journal already says is dropped, and an entry of nothing else is filed as a
   skip is.
3. **The keep:** ids are staged, the stretch's turns are filed under their
   entry (§4.1), the record is written to the conversation's metadata, and only
   then is the entry in memory — a record that fails to write leaves the
   journal as it was. A skip keeps the same way, filing its stretch under the
   latest entry. The sections follow at the next round.

Deciding in the write's own answer means the model weighs the stretch with its
parts numbered in front of it, and a stretch worth writing costs one answer on
one fork rather than a question on one fork and an answer on another.

`journal_write` and `journal_skip` are never in the dialogue's tool catalog.
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
  conversation that no entry and no skip covers. The projection event a round
  records already names every turn it selected, so the signal costs nothing. A
  turn is looked at before the session loses sight of it.

At most one draft runs per conversation (single flight). A draft's stretch is
every part after `covered_to`. A skip or a kept entry moves `covered_to` to the
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
  rebuilds from. It carries `covered_to`, so a stretch a draft skipped — or
  found only repeats in — after the newest entry stays covered across a
  restart: a draft that ends with nothing to keep writes the record too, and
  one whose record cannot be written is abandoned like a failed keep. A member that lets a conversation go (`turn_claims`) drops its journal
  and reads the record again when it next holds the conversation.
- **A forgotten entry** leaves the record at once and its section at the
  conversation's next round. One a restart left sealed is never selected — the
  collection is read by name — and goes with the conversation.
- **Private, lineage-scoped.** A journal is read by its own conversation, by
  the conversations below it in an `/mcp` lineage (§4), and through the
  `/mcp` tools by its owner — nothing else: no other conversation retrieves its
  sections, and no layer (`daily_history` included) reads them.
- **Only a dialogue journals.** zend has no dialogue fork; the titler,
  passthrough conversations, calibration and the ingest layers never journal.

## 8. Observability

- `GET /v1/conversations/:id/journal` returns the entries (stored form, span and
  rendered text), the open items, `covered_to` and `looked_to` (parts), whether
  a draft is running, and the last 16 draft records. Each record carries
  `took_ms`, `write_ms`, the result (a skip's reason as its `detail`), and each
  claim's verdict and kind.
- `POST /v1/conversations/:id/journal/draft` drafts now over whatever is not
  covered: 202 with the turns it covers, or 409 — `draft_running` while a
  draft runs (it may not cover the latest turns: ask again once `drafting`
  clears), `nothing_to_draft` when nothing is new, `not_open` when the
  conversation is not open on this member (a draft forks its live slot).
- `DELETE /v1/conversations/:id/journal/:eid` forgets one entry: the journal as
  it now reads, or 404 `no_such_entry`.
- Log target `zend::journal`: each draft's start, and its end with stretch,
  outcome, timings and verdicts.
- `GET /v1/conversations/:id/journal/:eid/turns?offset=&limit=` returns the
  turns behind one entry exactly as `journal_read` pages them — a folded
  entry's digest for a folded one — or 404 under the tool's own code
  (`no_entry_text`) when none is kept.
- **GUI:** the journal docks at the bottom right of an open conversation, so it
  costs the chat as little room as it can. Collapsed it is a pill — the
  journal's icon, how many entries it holds, a dot while a draft or fold runs.
  Open it is a short card (300 px, at most 46% of the height) that reads as the
  prompt does: one line per entry, oldest first and scrolled to the newest,
  dated where the day or hour moves, a tag on a folded day, week, month or
  year, what is still open below; with Drafts (how recent drafts ended), Draft
  (draft now), refresh and collapse in its header. Which of the two it is, is
  the browser's choice and kept. The dock reads the journal on opening a
  conversation and after each reply, and follows a draft or fold until it ends.
  Clicking an entry opens the journal modal: its line, everything behind it
  (its claims in order, or a folded entry's digest), and the turns it was
  written from as `journal_read` reads them, paged 200 lines at a time, with a
  Forget. Entries are not editable: every line in the journal is one the server
  cited and checked.

## 9. Code layout

| Where | What |
|---|---|
| `journal-core/` | the journal's logic (§2) |
| `candle-conversation` `conversation/carried.rs` | `Carried`, the section reconcile |
| `candle-conversation` `scheduler/ephemeral_fork.rs` | a fork's band |
| `candle-conversation` `conversation/ephemeral_turn.rs` | `AskScope`: a fork projected whole or focused on its conversation |
| `candle-conversation` `projection/` (`Schema::focus`) | the focused projection: own turns and locked working-set members only |
| `zend/src/journal/origin.rs` | the dialogue's turns as citable parts |
| `zend/src/journal/state.rs` | `SessionJournal`: a journal and its cadence |
| `zend/src/journal/pressure.rs` | selection pressure from the projection event |
| `zend/src/journal/style.rs` | `CODING`: UTC dates and a coding session's words |
| `zend/src/journal/world.rs` | `JournalsAbove`: the entries above a conversation its claims may rest on (`per`) |
| `zend/src/journal/desk.rs` | `ZendDesk`: the draft's question on a focused fork at `Normal`, the record write |
| `zend/src/journal/draft.rs` | one draft and its record |
| `zend/src/journal/carry.rs` | the schema install, the wanted sections, the round's selection |
| `zend/src/journal/record.rs` | the metadata record |
| `zend/src/journal/text.rs` | `JournalText`: the turns behind each entry as custom objects |
| `zend/src/journal/shelf.rs` | the tools' `JournalShelf`: a dialogue's own journal and those above it, each a `JournalBook` |
| `journal-core/src/transcript.rs` | `Filed`, the keys, and `Book`: paging and searching the kept turns |
| `zend-tools/src/tools/journal/` | `journal_read`, `journal_grep`, and the `JournalShelf` and `JournalBook` they read through |
| `zend/src/journal/registry.rs` | `Journals`: every dialogue's journal, and what a draft forks from |
| `zend/src/api/journal.rs` | the four routes |
| `zend/web/index.html` | the journal dock and its entry modal |
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
  reasoning is left out; long parts are clipped for the draft and kept whole
  for the text.
- **Turn text:** keys order by entry then stretch; a read pages lines with each
  turn's number and time; a search ignores case, counts every hit and returns
  the first few; a write files under the new entry and a skip under the latest;
  a filed stretch reads back under its own conversation alone; forgetting an
  entry ceases its text and ceasing a conversation's leaves nothing; the tools
  pass their arguments, defaults and bounds through, and name each miss.
- **Workspace `World`:** against a temporary git repository, a file's existence
  and blob, a changed file, a branch's head and name, and every question it
  cannot answer.
- **Cadence:** every-16, selection pressure, single flight, and `covered_to`
  after a keep, a skip and an abandon; a restored journal carries on.
- **Draft:** an empty journal offered the write alone; a journal with an entry
  offered the skip in the same answer; a skip covering the stretch with its
  reason; an unaskable draft; a write refused out of rounds.
- **Focused projection:** the locked working-set members kept and the
  momentum-held ones dropped; no group taken from another layer.
- **Band:** a fork made at each band decodes at it, a turn view on it too, a
  parked request keeps its band, and the band goes with the fork.
- **Prompt:** the embedded schema declares the journal, the install scopes it,
  an empty journal pins nothing, and the journal is never a section sink.
- **Routes:** the table and the dispatch of the four routes; a turns page's
  defaults and bounds, and a miss under the tool's code.

Only a live coding session can say:

- how often the model skips;
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
4. **A draft may decline.** Once the journal holds an entry, a draft may skip
   the stretch, which stops the journal filling with "nothing new". Settled on
   2026-10-24: the decline is offered in the write's own answer, on one focused
   fork, rather than as a separate question first.
5. **No idle trigger** (§6).
6. **Private to its conversation** and tombstoned with it; no layer reads it
   (§7).
7. **A read-only GUI pane with delete** (§8).
8. **Only the parent dialogue journals**, not sub-runs or agent jobs (§7).
9. **The journal folds on the calendar, not by count** (§4.2), settled on
   2026-10-24: day, week, month and year folds in UTC, each a one-line entry
   and a detailed digest the tools read, lossy over time; folding replaced
   keeping the newest five.
10. **The turns behind each entry are kept as text** for `journal_read` and
    `journal_grep` (§4.1), settled on 2026-10-24: not the conversation's turns,
    which leave its window, but a copy living as long as the entry, ceased when
    it folds or the conversation is tombstoned or archived.

The starting number `EVERY_TURNS` 16 is a guess to tune in live sessions.
