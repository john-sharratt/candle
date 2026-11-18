# A request's steps, and the request turn that anchors them

**Status: designed, not built.** Settled with the user on 2026-11-19. This
document is authoritative over the steps section of a dialogue's journal and
over the projection's rule that a request's own turn is never windowed out. When
it is built, `docs/zend_journal.md` §4 and §5.2 take its summary.

## 1. Why

A long agentic request is one user message followed by tens of tool rounds.
Two things anchor the model in it: **what it was asked**, and **what it has
already established**. Under selection pressure both can fall out of view:

- **Notes leave with their turns.** A `journal_note` is the `text` of a call in
  the reply that wrote it (`docs/zend_journal.md` §5.2), so it is visible only
  while that turn is projected. The fill keeps the current exchange's first
  turn, its newest tool turn and its last, and sheds the older tool turns first
  (`projection/fill.rs`, floors). In a 27-round loop the notes of rounds 2–25
  are therefore the first thing to go, and after that they can be reached only
  through `journal_read` / `journal_grep`. Measured on 2026-11-18
  (`yenbot-employee-lifecycle/anonymous-demo-mode`): the loop noted the same
  finding six times and re-derived its plan each round. The reasoning window
  (`REASONING_ROUNDS` = 8, `docs/thinking_span_projection.md`) restored the
  plan within eight rounds. It did not restore what was concluded before them.
- **Entries are too coarse and arrive late.** An entry is a step of the
  session. Within a request one is written only every 16 rounds or under
  selection pressure, and the request's closing entry only after it finishes
  (`docs/zend_journal.md` §6). An entry is the right record of a finished
  request and the wrong one for the request in progress.

The journal has two levels. **Entries** are the high-level record, carried in
the system prompt for as long as they live. **Notes** are the fine lines
written as the work goes. This design shows the notes of the request in
progress as that request's **steps**, in the system prompt, for exactly as long
as the request runs. When the request finishes, its entry replaces them.

## 2. The steps

### 2.1 What they are

A **step** is one `journal_note` call that this conversation's replies made
within the current request, written to the conversation's own journal
(`journal` omitted or `"self"`). These do not count as steps:

- a note to a memory journal (`~/…`, `@group/…`), which is a memory write and
  lands elsewhere;
- a note written by another conversation, including another worker on the same
  thread or a conversation above this one in an `/mcp` lineage. Steps are this
  conversation's own;
- a note of an earlier request.

The **current request** is the run of turns the next turn continues:
`summary_tree::exchange::continued_exchange` over the timeline's couplings,
read exactly as `Substrate::reasoning_turns` reads it. While a tool round's
output is in flight, the newest exchange is the request's rounds so far. When
the newest turn is uncoupled, the request has answered and the next turn opens
a new one, which has no steps yet.

### 2.2 Where they render

The steps are part of the journal collection in `zend/src/prompts/projection.yaml`.
They come after the conversation's own entries and before `journal/close`, so
the journal reads as the record so far, then the work in progress, then what is
still open:

```
Entry 7 — Began making demo mode anonymous across the employee lifecycle.
Steps of the current request, as noted so far:
Step 1 — The demo flag is read in three places; only session.rs gates on it.
Step 2 — EmployeeCard renders the real name before the demo check runs.
…
Still open: …
```

- One section per step, `journal/step/<n>`, chained after the entries like
  every journal section (`Sequence::submit_section`'s `after`). A new step
  seals its own line and the closing section, nothing before it. The heading
  is `journal/steps`: a fixed line sealed when the first step arrives, with the
  steps after it.
- `n` is the step's number within its request, counted from 1. Numbers stay
  fixed as steps age out (§2.4), so "Step 3" means the same step on every round.
- **The text, verbatim, clipped to 400 characters** with `…`. These are the same
  bounds a draft's parts are shown at (`docs/zend_journal.md` §3.1). The note
  grammar sets no length bound, and one runaway note must not take the room.
- The journal intro gains one sentence naming the steps as the notes the model
  has written so far in the request it is working on. It says nothing about
  where the notes came from beyond that.

### 2.3 When they change

At the head of every round, `carry_journal` reconciles the steps with the
current request, exactly as it reconciles entries (`Carried`):

- **A new note** joins as the next step. The round that wrote it does not wait
  for its section to seal. The turn holding the note is the current exchange's
  newest tool turn, which the fill keeps (§3), so the note is in view either
  way. By the round after that, the section has sealed and is pinned.
- **A request that finishes** loses its steps at the next round's reconcile.
  Every `journal/step/*` section is removed, and the closing section reseals.
  The request's closing entry, drafted as soon as it finishes (§6 there; may
  not skip), is the high-level line that replaces them.
- **A progress entry within the request** (every 16 rounds, or selection
  pressure) changes nothing about the steps. The request is not finished, and
  both stay in the prompt.

### 2.4 The cap: 64 steps

A request shows its **newest 64 steps**. Past that, each new step drops the
oldest, and the steps after the dropped one reseal. That is at most 64 short
lines, about 3–6K tokens, prefilled in one go, and only in a request that has
noted more than 64 times. Steps that age out are still in their turns, still
filed under the request's entries, and still found by `journal_grep`. What the
request was **for** is never among what ages out: the request turn is held
separately (§3).

### 2.5 State, restart and handover

Following the journal's rule that **nothing new is stored** (§5.2 there), a
step has no record of its own. Its words are the note call in its turn:

- `SessionJournal` keeps the current request's steps: its first turn, and each
  step's text and number. When a round seals, the steps grow from the note
  calls in that round's reply (`journal::notes`). When a new request opens,
  they reset.
- **A restart, a member that takes the conversation over, or a resumed turn**
  (`docs/turn_handover.md`) rebuilds the steps once. It reads the note calls
  from the turns of `continued_exchange`, when the journal is loaded and before
  the first round.
- Sections are the conversation's (§7 there). A steps section a restart left
  sealed is reconciled away like a forgotten entry's.

### 2.6 Cost

The steps are part of the system prompt, so they are charged before the fill's
room (`segment_tokens(&system_prompt_segments, …)` in `project.rs`). At the cap
that is ~3–6K tokens, which the dialogue's other shares give up while a long
request runs and get back when it finishes. Per round, a step costs its own
line's prefill plus the closing section's.

## 3. The request turn stays in

**Rule:** however long a request runs and however hard the window presses,
the request's own turn — the one whose user side is the person's message (or a
worker's task) — is in every projection the request decodes under. Without it
the model loses what it is doing, whatever the steps say.

**This holds in the code today, by two rules:**

1. **Selection** (`projection/selection.rs`, `select_conversation`). The
   candidates are exchanges, not turns, and a tool loop is one exchange. The
   `recent` window keeps the newest exchange "whatever it costs: it is the
   exchange the conversation is in". It is never trimmed by budget.
2. **The fill** (`projection/fill.rs`, `dialogue_fill.rs`). The current
   exchange's first turn, its newest tool turn and its last are floors, placed
   first. When the floors pass the room they give way from the bottom: the
   recent exchanges, then the current exchange's *older* tool rounds. The
   request turn is never shed. What is left past the room carries the slot past
   its window (`over_room`), and nothing downstream trims it.

**What is missing, and what is built:**

- **No test pins it end to end.** The fill tests check the floors on
  hand-built units, and the selection tests check the recent window. Neither
  checks the whole projection on a long loop. One test is built: a dialogue
  whose request runs 64 tool rounds of ~6K-token tool outputs, projected under
  a window that holds a fraction of them, still emits the request turn, the
  newest tool turn and the newest turn, and emits every step of §2.
- **The summary tree's path does not hold it.** When a summary tree exists,
  the §8 score-density pick replaces the rule-based selection and is only closed
  under exchange (`project.rs`, `summary_tree_select`). Nothing in it keeps the
  newest exchange. The tree is switched off and goes once the journal is proven
  (`docs/zend_journal.md` §12). Until then, the end-to-end test above runs on
  the path production runs, and the tree is not re-enabled without passing it.
- **The steps take room from the fill, never from the floors.** Because the
  steps are in the system prompt, a long request's steps shrink the room. A
  smaller room presses on the recent exchanges and the older tool rounds first,
  and never on the request turn. The test runs with the steps at their cap.

## 4. Decisions

Settled with the user on 2026-11-19:

1. **The forced note stays.** `journal_note` after 200 tokens of reasoning
   (`NOTE_AFTER_REASONING`) is what writes the steps, and journaling depends on
   it.
2. **Only this conversation's notes** are steps (§2.1).
3. **Capped at 64**, newest kept (§2.4).
4. **They last as long as the request.** A finished request's steps go, and its
   entry replaces them (§2.3).
5. **The request turn is never windowed out** (§3).

Made in writing this, to be confirmed or reversed before it is built:

- **Placement** after the entries, before `journal/close` (§2.2). The steps
  read as the newest part of the record, and "Still open" stays last.
- **One section per step**, so a new step seals one line (§2.2). The
  alternative, one section for all steps, reseals every step on every note.
- **Clipping at 400 characters** (§2.2).
- **Dropped at the round after the request finishes, not when its entry
  seals.** If the next request starts before the closing entry seals, there is
  a gap: its first round shows neither the steps nor the entry. It still has the
  finished exchange as its newest recent exchange, a floor. Holding the steps
  until the entry is pinned would close the gap, at the cost of a second label
  ("steps of the request before this one").

## 5. Code

| Where | What |
|---|---|
| `zend/src/journal/steps.rs` | `Steps`: a request's first turn and its steps; grow from a sealed reply, reset on a new request, rebuild from the request's turns, the newest 64 |
| `zend/src/journal/notes.rs` | reading a reply's note calls with their `journal`, so a memory-journal note is told apart |
| `journal-core/src/section.rs` | the `journal/steps` heading and `journal/step/<n>` names; a step's rendered line |
| `zend/src/journal/carry.rs` | the steps in the wanted sections, between the entries and `journal/close` |
| `zend/src/journal/state.rs` | `SessionJournal` holds the `Steps` |
| `zend/src/prompts/projection.yaml` | the intro's sentence on steps |
| `candle-conversation/src/projection/` tests | the end-to-end request-turn test (§3) |

## 6. Tests

- **Steps:** a reply's self-journal notes become steps in order, and a memory
  journal's do not; a new request resets them; the 65th step drops the first
  and keeps the numbers; a note longer than 400 characters is clipped with `…`;
  a rebuild from a request's turns equals the steps grown round by round.
- **Sections** (raw expected text): the heading and step lines render exactly;
  the wanted sections order is entries, steps, close; a finished request wants
  no step section.
- **Carry:** a new step seals one section and the close; a finished request
  removes every step section; a progress entry leaves the steps standing.
- **Handover:** a conversation taken up mid-request (`resume_conversation`)
  shows its steps on its first round.
- **The request turn** (§3): the long-loop projection test, with the steps at
  their cap.
- **Live, measured:** on long worker loops before and after, count tool calls
  repeated with the same name and arguments, re-reads of a range already read,
  notes repeating an earlier note, and rounds to the final answer
  (`docs/zend_journal.md` §5.2's footprint).
