# zend's journal

**Status: proposed.** Nothing here is built yet. It adapts npcd's journal
(`docs/journal.md`, built for Battle Cities characters) to zend dialogues, and
it closes the `TODO.md` item "Implement the journal tool".

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
| Nothing tied a sentence to where it came from | Every claim **cites turn numbers**; the kind of claim (saw, did, heard, concluded) is derived by the server from what it cites, never from the wording |
| Nothing checked a sentence against the world | Typed claims about the **workspace** are checked against it when the entry is kept, and corrected or demoted when they are wrong |
| Compressed every exchange, needed or not | A **gate** asks whether a stretch needs an entry at all; a `no` is final for that stretch |

The real turns stay in the substrate, and provenance still retrieves them. The
journal adds what selection cannot: a short, always-present record of what the
session has established and what is still open.

## 2. What carries over from npcd

npcd's journal (`npcd/src/engine/journal/`) is built and tested against a
scripted `Desk`. These parts carry over unchanged in behaviour:

- **Entry shape**: claims with citations, `intend` (next steps), opened and
  resolved items (§3).
- **Server-derived kind and perishability**: the model never grades its own
  certainty (§3.2).
- **The workflow**: step 0 (nothing citable means nothing to ask), the gate, the
  write with up to `MAX_ROUNDS` (4) answers, and refused writes asked again with
  the refusal appended (§5).
- **One-call grammars** compiled per draft, with the `cite` enum holding the
  stretch's turn numbers.
- **Repeat suppression** (`repeats.rs`): a claim sharing at least 70% of its
  distinct words with one already held is dropped.
- **Section reconcile** (`mind/carried.rs`): what the prompt wants and is not
  yet held is submitted; a section no longer wanted is removed.

**The shared core moves into one crate, `journal-core`.** It holds `entry`,
`verify`, `tools` (the grammar), `workflow` (with the `Desk` and `World`
traits), `repeats`, the generic cadence state, and the `carried` reconcile.
npcd and zend then each supply only what is genuinely their own:

- how a turn's origin maps to a claim kind;
- the `World` that typed claims are checked against;
- section and metadata names;
- the `Desk` that puts the questions to their engine.

Copying npcd's code into zend instead would leave two journals that drift
(§13).

## 3. An entry, for coding

An entry covers a stretch of the dialogue, from turn `a` to turn `b`. It holds:

| Field | What it holds | Limit |
|---|---|---|
| `claims` | what the session established, each with citations and a server-derived kind | `MAX_CLAIMS` 12 |
| `intend` | what the model means to do next | `MAX_INTEND` 3 |
| `opened` | items left unresolved, each given an id by the server: a failing test, an unanswered question, a step not yet done | `MAX_OPEN` 6 live |
| `resolved` | ids of earlier open items this entry settles | — |

### 3.1 Citations

A citation is the dialogue's turn index, within the conversation's timeline.
The write question shows the stretch as numbered text, because the live history
is not numbered, and a citation is checked against that numbering.

**Citable:** the user's messages, the assistant's tool-call turns, and the tool
responses, including a fast-path reply served from turns already in context.

**Not citable:** injected or placed turns (ingest reading, the priming chain),
glue, and the journal's own questions, which are never sealed.

### 3.2 Kind, derived from what is cited

| Kind | Cited turn | Rendered under |
|---|---|---|
| `Observed` | a tool response: a file read, a grep, a command's output, `git status` | `Saw:` |
| `Heard` | a user message | `User said:` |
| `Did` | the assistant's own acting call: `file_edit`, `write`, `file_delete`, `git_commit`, `run_command` (the call, not its output) | `Did:` |
| `Inferred` | nothing the session saw supports it | `Concluded, not seen:` |

A claim with several citations takes the strongest support (`Observed` over
`Heard` over `Did`). The model cannot promote a guess to an observation by
calling it one.

`Did` lines are not perishable: an edit, once made, was made. Every other line
renders with `(re-check)`, because the file, the test result or the branch may
have changed since.

### 3.3 Typed claims, checked against the workspace

A claim may carry `{subject, attribute, value}`. When the entry is kept, the
claim is checked against a **workspace snapshot** taken before the draft
starts, so the check reads one consistent picture however long the draft takes.

| Subject | Attributes |
|---|---|
| `file:<repo>/<path>` | `exists`, `blob` (the git blob id at the conversation's checkout) |
| `branch:<repo>` | `head` (the commit the conversation's checkout is on) |

- A match is kept.
- A mismatch is replaced by what the workspace says, and marked `corrected`.
- A claim the snapshot cannot answer becomes prose.

Test results and exit codes are deliberately **not** typed claims. They are
facts about a past run, cited as `Observed` from the tool response that
reported them, and they render with `(re-check)`.

## 4. The journal in the prompt

The dialogue layer in `zend/src/prompts/projection.yaml` gains three entries,
the way npcd's character prompt has them:

- `journal_intro`: the framing.
- `journal`: an `always_visible` collection with `sections: []`.
- `journal_none`: shown when the collection is empty (`depends_on_absent`).

There is one section per entry (`journal/<id>`) for the newest `IN_PROMPT` (5)
entries, plus `journal/open` while any item is open. Before each turn, the
section reconcile submits what is wanted and removes what is not, so an entry
that ages out of the newest five leaves the prompt.

**The framing is what keeps it from reading as history.** It names the
sections as notes the model wrote earlier, each line with how it knows it, and
says that `(re-check)` lines may have changed. Sealed K/V cannot be re-rendered
at gather time, so each entry carries an absolute time
(`[23 Oct 2026, 14:20–14:41 UTC]`), never "twenty minutes ago".

**Budget.** The collection takes a fixed share of the dialogue budget,
`journal_tokens`, proposed at 1,500 tokens. A full journal is five entries of
up to 12 short lines each, plus the open items, so it should fit in that.

## 5. Drafting

**Drafts run on the dialogue's own conversation**, through `UnsealedAsker`.
zend already uses it for `--self-check` (`zend/src/self_check/ask.rs`). Each
question runs on an ephemeral fork of the live slot:

- it reuses the K/V the dialogue already holds, so the stretch being asked
  about costs nothing to prefill;
- nothing is written back, neither the question nor the answer;
- the scheduler parks the fork until any turn in flight ends, so a draft never
  runs inside a turn.

The workflow is npcd's, unchanged (§2):

1. **Step 0:** a stretch with nothing citable is a mechanical `Nothing`; the
   model is not asked.
2. **The gate:** an `answer` call, `yes` or `no`, with a required reason. It
   names the stretch ("your last *n* turns, from … to …") and is judged against
   the journal already in the prompt. A `no` is final for that stretch.
3. **The write:** one `journal_write` call under the per-draft grammar, at most
   `MAX_ROUNDS` (4) answers. A refused write is asked again with the previous
   answer and the reason appended.

`journal_write` is never in the dialogue's tool catalog. Its call exists only
as an answer grammar, so the model cannot write an entry on a live turn.

### 5.1 Priority: drafts never compete with the user

A dialogue is the `High` band, and the background layers are `Low`. A journal
draft runs at **`Normal`**, below the user's turns and above the layers. That
also fits the coding-task priority you want: launched tasks run at `High`,
drafts follow them, and the background analysis comes last.

**This needs an engine change.** Today `ForkEphemeralSequence` takes no band,
so a fork inherits its parent's `High`. The request gains a required
`DecodePriority` for the fork. Callers name it: the journal passes `Normal`,
and self-check passes what it means. It is a real parameter, not an optional
flag that keeps the old behaviour alive.

## 6. Cadence

zend counts sealed dialogue turns (each tool round seals one), not npcd's
window turns. A stretch is waiting when either of these holds:

- **`EVERY_TURNS` (16)** turns have landed since the last draft was handed a
  stretch.
- **Selection pressure:** the dialogue's projection left out a turn that no
  draft has covered yet. This is the analogue of npcd's eviction margin: a turn
  is looked at before the session loses sight of it. The projection already
  knows which sealed turns it left out, so the signal costs nothing extra.

At most one draft runs per conversation, as in npcd (single flight). A draft's
stretch is every turn after `covered_to`. A `no` or a kept entry moves
`covered_to` to the end of the stretch. An abandoned draft leaves it where it
was, so the next draft retries those turns.

There is no idle trigger. A quiet conversation's uncovered turns are still in
the substrate, whole, and the next trigger after the session resumes covers
them.

## 7. Persistence, replication and forks

- **Sections are the conversation's.** They are sealed and persisted with it,
  replicated with it, and released when it is tombstoned, like the `response`
  collection's.
- **A submitted section's text cannot be read back from the substrate**, so the
  entries are also kept as JSON in the conversation's metadata (`journal.kept`).
  This is the record a restart, or the member a conversation is handed to,
  rebuilds from. On a rebuild, `covered_to` resets to the last turn the newest
  entry covers. Unlike npcd's window, zend's turn indices are stable across
  runs, so there is no need to reset it to 0.
- **A forked conversation** (`conv_branches`) inherits the parent's record at
  the fork point. Entries citing turns after the fork are not inherited.
- **Private.** A journal is read by its own conversation and nothing else: no
  other conversation retrieves it, and no layer (`daily_history` included)
  reads it. It is tombstoned with its conversation.
- **Only the dialogue journals.** A coding task's sub-conversations (`sub_run`,
  agent jobs) do not; their results land in the parent's turns, which the
  parent's journal cites.

## 8. Observability

- `GET /v1/conversations/:id/journal` returns the entries (stored form, span,
  and rendered text), the open items, `covered_to`, whether a draft is running,
  and the last 16 draft records. Each record carries `gate_ms`, `write_ms`, the
  result, and each claim's verdict and kind.
- `POST /v1/conversations/:id/journal/draft` drafts now: 202 with the stretch,
  or 409 if a draft is running or nothing is new.
- `DELETE /v1/conversations/:id/journal/:eid` forgets one entry. Its section is
  removed and the metadata record rewritten.
- Log target `zend::journal`: each draft's stretch, outcome, timings and
  verdicts.
- **GUI:** a read-only journal pane on the conversation lists the entries as
  the prompt reads them, the open items, and each claim's verdict and kind,
  with a delete on each entry (the `DELETE` route). Entries are not editable:
  every line in the journal is one the server cited and checked.

## 9. Code layout

One concern per file.

| Where | What |
|---|---|
| `journal-core/` (new crate) | `entry`, `verify`, `tools`, `workflow` (`Desk`, `World`), `repeats`, `state` (cadence), `carried` (section reconcile), lifted from npcd |
| `zend/src/journal/origin.rs` | a zend turn to its citability and claim kind |
| `zend/src/journal/world.rs` | the workspace snapshot: file existence and blob ids, branch heads |
| `zend/src/journal/desk.rs` | `ZendDesk`: the gate and write through `UnsealedAsker` at `Normal`; the keep (reconcile plus metadata record) |
| `zend/src/journal/trigger.rs` | cadence: every-16, selection pressure |
| `zend/src/journal/section.rs` | section names, rendering, the metadata record |
| `zend/src/api/journal.rs` | the three routes |
| the zend GUI | the read-only journal pane, with delete |
| `npcd/src/engine/journal/` | shrinks to npcd's origin mapping, sim `World`, `EngineDesk` and sections |
| `candle-conversation` scheduler | `ForkEphemeralSequence` gains the fork's band |

## 10. Failure

- No failure of the journal stops the dialogue. Every failure ends in a logged
  abandon, and the user's turn never waits on a draft beyond the band sharing
  the device.
- A dialogue layer with no `journal` collection journals nothing; that is not
  an error.
- A crash between a kept entry's reconcile and its metadata write loses that
  entry from the record. Its section is still sealed and is dropped on rebuild.

## 11. Tests

Built alongside the code, each building block testable alone:

- **`journal-core`** keeps npcd's scripted-`Desk` tests. npcd keeps passing
  through the move.
- **Origin mapping:** each zend turn kind maps to its expected citability and
  claim kind.
- **Workspace `World`:** against a temporary git repository, a typed claim
  matches, is corrected, or is demoted to prose.
- **Cadence:** every-16, selection pressure, single flight, and
  `covered_to` after a keep, a `no` and an abandon.
- **Band:** a fork made at `Normal` decodes at `Normal` beside a `High` turn.
- **Restore:** a record round-trips, and a fork inherits only entries that cite
  turns before the fork.

Only a live coding session can say:

- how often the gate says `no`;
- how much a draft slows the user's next round;
- whether five entries are enough for a long task, and whether 1,500 tokens is
  the right share;
- whether the model actually uses the open items and `(re-check)` marks.

## 12. Relation to the summary tree

Once the journal is live and has carried real long sessions, the dormant
summary tree (`candle-conversation/src/summary_tree/`) and the `summary:`
blocks in `projection.yaml` have no remaining purpose, and they are removed
then — not before, so the journal is proven first.

## 13. Decisions

Settled with the user on 2026-10-23:

1. **One shared crate.** npcd's core moves into `journal-core` (§2); npcd and
   zend each keep only their own origin mapping, `World`, `Desk` and sections.
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

The starting numbers, `EVERY_TURNS` 16, `IN_PROMPT` 5 and `journal_tokens`
1,500, are guesses to tune in live sessions.
