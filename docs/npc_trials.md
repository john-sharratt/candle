# Trial turns: a suspended character, one turn at a time

A trial is one turn put to a character that takes no turns of its own, and
answered as one call under a grammar the caller supplies. It writes nothing
into the character: not the turn, not the answer. Two trials with the same
inputs read the same context and decode greedily, so the same trial asked
twice gets the same answer.

It exists so that what a character is shown can be **measured**. A search
over ways of presenting a world (Battle Cities'
`docs/project/perception-search.md`) sends thousands of trials, each a scene
rendered one way and a question whose answer the game already knows, and
scores the calls that come back. It runs through npcd because what is being
tuned is npcd's own mind: its model, its sampling, its frame and its prompt.
A stand-in runner beside npcd would tune a copy that drifts from the real
thing.

## 1. The route

`POST /v1/npc/:nid/trial`, `User` with the ownership check (an admin passes
for any character), sent to the member holding the character as every
character route is.

```json
{
  "turn": "what the character reads this turn",
  "tools": [{"name": "answer", "params": [
      {"name": "value", "type": "string", "required": true, "enum": ["north", "south"]},
      {"name": "reason", "type": "string", "required": true}]}],
  "frame": {"system": "a system prompt", "history": [{"user": "…", "assistant": "…"}]},
  "max_tokens": 640,
  "temperature": 0.0,
  "seed": 0,
  "thinking": "off"
}
```

- `turn`: the user message, verbatim. Required and non-empty.
- `tools`: one or more `ToolSpec`s (`candle_conversation::stencil`). The
  grammar holds the turn to exactly one call of one of them. A forced answer is
  an `answer` spec; an act is the spec of the act. Names must be distinct.
- `thinking`: how hard the turn thinks before its call — `off` (the default),
  `quick`, `balanced`, `deep` or `exhaustive`, as a mind's turn would at that
  level. Off, the turn opens with the dialect's closed think block and
  `<tool_call>`; any other level opens `<think>`, and one grammar steers the
  block shut at that level's budget and holds the call after it. The sampling
  is the character's own row for that level.
- `frame`: what the turn is read under. **Absent: the character's own** (§2).
  Present: the given `system` prompt, with `history` laid down before the
  turn as verbatim exchanges (§3). `system` must be non-empty, and `history`
  holds at most 16 exchanges.
- Every field is known: a misspelt one, at any level, is refused rather than
  ignored, since a misspelt `frame` or `history` would silently run a
  different trial. The turn, system prompt and history together are at most
  256 KiB.
- `max_tokens`: the decode's cap, `1..=4096`, default `640`.
- `temperature`: default `0.0`, which is argmax. A trial is a measurement,
  so it decodes greedily unless the caller asks for spread on purpose.
- `seed`: the sampler's seed, default `0`. It matters only above zero
  temperature.

The answer:

```json
{
  "raw": "the decode as it came back",
  "call": {"name": "answer", "arguments": {"value": "north", "reason": "…"}},
  "prompt_tokens": 5120,
  "tokens": 23,
  "ms": 1840
}
```

`call` is `null` when the decode holds no call of one of `tools`. That can
happen only when a decode is cut off at `max_tokens`, since the grammar
admits nothing else. `prompt_tokens` is the context the turn was read in,
and `ms` is the wall time from the route taking the request to its answer.

**Both frames decode alike**: one call of the trial's tools under the same
grammar, the reply opened as the trial's `thinking` opens it, and the
character's own sampling for that thinking at the trial's `temperature` and
`seed`. So a prompt measured on a given frame measures what it does as the
mind's own.

### Refusals

| Status | `error` | When |
|---|---|---|
| 400 | `bad_trial` | An empty `turn`, no `tools` or more than 16, two tools of one name, tools that make no grammar, an empty `system`, more than 16 exchanges, more than 256 KiB of text (tools counted), `max_tokens` out of range, `temperature` outside 0 to 2, a `thinking` that is no level, an unknown field |
| 404 | `npc_not_found` | No such character, or not the caller's |
| 409 | `npc_not_suspended` | The character is not suspended |
| 503 | `no_engine` | The daemon runs no engine, or its model has not loaded yet |
| 409 | `not_answered` | The engine refused or failed the decode |

**Only a suspended character takes trials.** That is what makes trials
comparable. A suspended character takes no turn of any kind (`crate::suspend`),
so between two trials nothing it reads moves: no turn lands in its
conversation, no journal entry is kept, no mission advances. A running
character's context changes under the trials, and the second of two
identical trials would be answered from a different history.

The world's own clock still runs. A test world whose trials must not cross a
day boundary has its clock paused (`PUT /v1/world/:wid/time`), because the
day names the character's conversation (§2).

## 2. The character's own frame

With no `frame`, the trial reads what the character's next turn would read:
its system prompt under the mind's projection, its pinned identity, mission
and journal sections, and the selected history of its day's conversation.
It then answers on an **ephemeral fork** of that conversation's slot
(`Sequence::unsealed_asker`, as `Minds::ask` does). The fork is never sealed,
so the turn and the answer are gone when it ends.

A suspended character has let its conversation go (`Minds::let_go`). The
first trial opens it the way the character's own next turn would, through the
one function that turn opens it by (`Minds::resident_for`): for the world's
day and the character's chapter, rejoining the day's conversation from the
log or minting it, and rolling a conversation of another day or chapter over.
That function holds a lock per character while it opens, so a trial on a
character resumed as it opens never adopts a second conversation over the
first's slot. The trial reads the chapter from the sim and does not close
one: closing a chapter writes the journal, which is the character's own
turn's business. The conversation stays open for the trials after it, and
for the character's first turn once it is resumed, which carries on in it.

**Every trial reads one context.** The selection is prepared once for the
conversation, under its lock — the mission, journal and channel sections
brought in line, waiting for journal entries still sealing — and kept against
the conversation's timeline; every trial's fork reads it, and each fork is
taken on the conversation its selection was prepared on.

**A suspended character holds its conversation only while it is tried.** The
last trial in flight on it lets the conversation go again as it ends, so a
suspended cast holds nothing on the card between runs, as suspension
promises. A run that pauses between trials opens the conversation again on
its next, rejoining it from the log.

Opening is the one thing a trial changes, and it is the change the
character's own next turn would have made.

## 3. A given frame

With `frame`, the trial reads `frame.system` as its system prompt, then each
of `frame.history` as a completed exchange, then the turn. All of it is in a
conversation opened for the one trial and thrown away after it
(`engine::prose`'s shape, `prose::decode_framed`: a transient frame, a closed
reserved group, the timeline tombstoned on every path out). The history is
prefilled (`Sequence::insert_turn`), not decoded: turns of the conversation's
own, which its closed group admits, read as its projection selects them —
which is why a history is bounded to 16 exchanges and the whole trial to
256 KiB. Nothing of the character is in it.
The character is named only because a trial is the character's: its route,
its owner, its suspension.

This is how a system prompt is tuned. A search sends many arms, each a
different prompt, at once, and none of them has to be installed in the mind
first. An arm that wins becomes the mind's own prompt (`frame` absent), and
a trial run on the character's own frame confirms it there.

## 4. Many at once

Trials are independent requests, so a caller sends them concurrently and the
scheduler batches their prefills and decodes into the same waves as any other
work. On the own frame every trial is a fork of one slot and shares its
resident K/V. On a given frame, trials whose `system` is the same text share
one transient frame section.

**Bit-identical answers hold only at a fixed batch.** Greedy decoding fixes
the token chosen for given logits. The logits themselves come from batched
kernels, whose sums can round differently when the batch around a row
changes. A search measures how often a repeated trial changes its answer
before it compares arms, and treats a difference smaller than that as no
difference.

## 5. Where it lives

| File | What |
|---|---|
| `npcd/src/engine/trial.rs` | The body and its checks, `TrialRefused`, the answer, and reading the call back |
| `npcd/src/engine/trial_api.rs` | The route and its refusals |
| `npcd/src/suspend.rs` | `not_suspended`, the 409 for a character that takes turns |
| `npcd/src/engine/runtime/trial.rs` | `Runtime::trial`: the character's persona, surroundings, day and chapter, then the decode |
| `npcd/src/engine/mind/trial.rs` | `Minds::trial_own` (the fork, its selection prepared once, the conversation let go after the last trial) and `Minds::trial_given` |
| `npcd/src/engine/mind/conversation.rs` | `Minds::resident_for`: opening a character's conversation, one opening a character at a time |
| `npcd/src/engine/mind/think.rs` | `Minds::prepare_waiting`: a selection prepared waiting for journal entries still sealing |
| `npcd/src/engine/prose.rs` | `decode_framed`: a throwaway turn after a history, under its caller's grammar, opening and sampling |
