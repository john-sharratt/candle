# tree_gen — Design Document

## Concept

`tree_gen` generates a deep sequential transcript of a character's life by running a
multi-stage guide pipeline against a single character LLM. The output is a structured YAML redo
log that can be replayed by `chat --resume` or inspected offline.

The architecture separates *planning* from *delivery*. A series of guide inference steps
construct scaffolding — character blocks, daily waypoints, per-turn reflections, narrated
events — before anything is sent to the character. The character only ever receives polished,
contextually grounded narration.

---

## Input File Format

The input is a Markdown document divided into named **periods** by level-1 headers. Within each
period, each non-empty line is a **timeline entry** with a date and a description.

```markdown
# Early Childhood

1987-03-04: Bramble is born in a small market town. His father works at the mill.
1987-12-25: First Christmas. He is fascinated by the wrapped boxes under the tree.
1988-06-10: Starts nursery. Cries on the first day; stops by the third.

# Primary School Years

1991-09-03: First day at St. Cuthbert's. He is placed next to a girl named Fen.
1992-02-14: Receives his first Valentine's card. He does not know who sent it.
```

**Period headers** (`# Name`) divide the timeline into named phases of life. The Rust parser
splits on these headers, producing a `Period { name, entries: Vec<(NaiveDate, String)> }`.

---

## Pipeline Overview

```
Input file
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Parse                                          │
│  Split Markdown into Period { name, entries }   │
└────────────────────────┬────────────────────────┘
                         │ for each period
                         ▼
┌─────────────────────────────────────────────────┐
│  guide_summarize_period                         │
│  In:  period entries                            │
│  Out: CharacterBlock { summary, beliefs[] }     │
└────────────────────────┬────────────────────────┘
                         │ for each day in period
                         ▼
┌─────────────────────────────────────────────────┐
│  guide_today                                    │
│  In:  CharacterBlock (system), date,            │
│       description, yesterday, last month        │
│  Out: Vec<Waypoint> (5–15 per day)              │
└────────────────────────┬────────────────────────┘
                         │ for each waypoint
                         ▼
┌─────────────────────────────────────────────────┐
│  guide_reflect          (skipped on turn 1)     │
│  In:  day waypoints so far,                     │
│       character's last response                 │
│  Out: Reflection (2–4 sentences)                │
└────────────────────────┬────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│  guide_narrate                                  │
│  System: CharacterBlock + Reflection            │
│  User:   next Waypoint                          │
│  History: all prior narrations this day         │
│  Out: narrated event text (1–4 sentences)       │
└────────────────────────┬────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│  character_conv.submit_turn(narration)          │
│  Out: character's response                      │
└────────────────────────┬────────────────────────┘
                         │
                         ▼
                  LogRecord::Turn  ──►  YAML redo log (flushed)
```

---

## Guide Prompts

All prompts live in `candle-conversation/src/prompts/` and are embedded at compile time.

### `guide_summarize_period`

**Role**: One-shot inference. Takes the already-sliced period entries and synthesises a
character block for use throughout that period.

**Output format**:
- A 2–4 sentence third-person summary paragraph (appearance, temperament, circumstances)
- A `Character Beliefs` section with bullet lines beginning `- Believes …`

**When**: Once per period, before any day is planned.

---

### `guide_today`

**Role**: One-shot inference. Plans the story of a single day as an ordered waypoint list.

**System prompt includes**: CharacterBlock for the current period.

**User message includes**:
- `DATE`: calendar date
- `DESCRIPTION`: the source timeline entry for this day
- `YESTERDAY`: summary of the previous day (empty at start of period)
- `LAST MONTH`: summary of the preceding month (empty early in period)

**Output**: Numbered list of 5–15 waypoints, concrete present-tense events.

**When**: Once per day, after `guide_summarize_period`, before the first narration.

---

### `guide_reflect`

**Role**: One-shot inference. Observes the character's last response against the events
of the day so far, producing a brief internal compass for the next narration.

**Input**:
- The waypoints completed so far today
- The character's most recent response (thinking stripped)

**Output**: 2–4 sentences in third-person past tense. Emotional/inner state, unresolved
tension, beliefs touched or tested.

**When**: Before each narration except the very first of the day (no prior response yet).

---

### `guide_narrate`

**Role**: Stateful multi-turn inference — a single `guide_narrate` conversation persists
across all waypoints in one day.

**System prompt includes**:
- CharacterBlock (static for the period)
- Latest Reflection (updated after each character response)

**User message**: Next waypoint (one sentence).

**Conversation history**: All narrations already delivered today — the guide has full
context of how the day has unfolded before composing the next event.

**Output**: 1–4 sentences in second-person present tense — the event delivered to the
character LLM. Does not tell the character what to feel or do.

**When**: Once per waypoint.

---

## Log Format

Multi-document YAML redo log, shared with `chat --resume` via
`candle_conversation::conversation_log`.

```yaml
---
kind: header
character_system_prompt: |
  Bramble is a boy of about nine...
guide_system_prompt: |
  (period summary prompt)
started_at: epoch+20507d 14:22:01Z
---
kind: turn
seq: 1
guide_message: A letter is pushed under your door before anyone else is awake.
character_response: I hear the soft thud before I am properly out of sleep...
character_token_count: 203
---
kind: done
total_turns: 47
elapsed_secs: 1840.2
```

---

## Sampling

Guide inference steps (summarize, today, reflect, narrate) and character inference share a
default `ConversationConfig`. The same CLI sampling flags apply to all conversations; per-role
overrides are a future extension.

| Flag | Description |
|---|---|
| `--temperature` | Sampling temperature |
| `--top-p` | Nucleus sampling cutoff |
| `--top-k` | Top-k cutoff |
| `--repeat-penalty` | Penalty on recently generated tokens |
| `--presence-penalty` | Penalty on any previously seen token |
| `--seed` | RNG seed |
| `--sampler` | Named preset (`relaxed`, `creative`, `precise`, `antirep`) |
| `--thinking` / `--no-thinking` | Enable or suppress `<think>` blocks |
| `--max-tokens` | Max tokens per response |

---

## Open Questions / Next Steps

1. **Period boundary handling** — When moving from one period to the next, should the character
   conversation be forked, closed and reopened, or continued? The character's KV cache holds
   everything they have lived through; a new period may span years.
2. **Yesterday / last-month summaries** — These need to be generated automatically from the
   previous day's log turns, not written by hand. A summarization pass over completed days is
   the likely approach.
3. **Per-role sampling** — Guide steps (especially `guide_narrate`) may benefit from different
   temperature settings than the character. The builder currently applies one config to all.
4. **Resume** — A killed run should be continuable. The redo log has enough information to
   reconstruct character KV state via `insert_turn`, but the guide pipeline state (which day,
   which waypoint, the current `guide_narrate` conversation) needs to be checkpointed too.
5. **Branching** — The `ConversationTree` API supports forks. A future extension could branch
   the character at decision waypoints, exploring multiple paths from one day.

```
waypoints[i]  ──►  guide_conv.submit_turn()
                        │
                   guide assistant response
                        │  (thinking blocks stripped)
                        ▼
               character_conv.submit_turn()
                        │
                   character assistant response
                        │  (thinking blocks stripped)
                        ▼
                  LogRecord::Turn  ──►  YAML redo log (flushed immediately)
                        │
                   last_character_response  ──►  next guide input
                                                  (once waypoints exhausted)
```

---

## Inputs

| Source | Flag | Description |
|---|---|---|
| ChaCML file | `--input` | Guide's system prompt (`system` block) + ordered waypoints (`user` blocks) |
| Inline string | `--character-prompt` | Character system prompt (takes precedence over file) |
| Text file | `--character-prompt-file` | Character system prompt loaded from file |

### ChaCML format

ChatML markup (`<|im_start|>` / `<|im_end|>`).

- `system` block → guide's system prompt (operating instructions, tone, rules)
- `user` blocks → waypoints, consumed in order as user messages to the guide
- `assistant` blocks → silently ignored

Example:

```
<|im_start|>system
You are guiding a character through a journey of quiet revelation.
Speak as a participant in her world — a voice, a letter, an encounter.
<|im_end|>
<|im_start|>user
Begin: she receives a letter from someone she thought had forgotten her.
<|im_end|>
<|im_start|>user
Move on: the letter leads her outside for the first time in weeks.
<|im_end|>
```

---

## Output

Multi-document YAML redo log (`---` separated). Written and flushed to disk after each turn — the file is always valid up to the last completed exchange even if the process is killed.

Three record types (defined in `candle_conversation::conversation_log`):

```yaml
---
kind: header
character_system_prompt: |
  You are Mira...
guide_system_prompt: |
  You are guiding a character...
started_at: epoch+20507d 14:22:01Z
---
kind: turn
seq: 1
guide_message: A letter arrives on your desk.
character_response: I hold the envelope for a long moment before opening it.
character_token_count: 187
---
kind: done
total_turns: 30
elapsed_secs: 412.7
```

The same types are used by `chat --resume` to replay the log and rebuild the KV cache.

---

## Guide Behavior

1. While the waypoint list has entries, the next waypoint is sent to the guide as a user message.
2. Once waypoints are exhausted, `last_character_response` is fed to the guide instead — free-generating the next event organically.
3. Falls back to the literal string `"Continue."` if the character's last response was empty.

---

## Sampling

Both conversations share the same `ConversationConfig`, constructed from CLI flags. Full parity with `chat.rs`:

| Flag | Description |
|---|---|
| `--temperature` | Sampling temperature |
| `--top-p` | Nucleus sampling cutoff |
| `--top-k` | Top-k cutoff |
| `--repeat-penalty` | Penalty on recently generated tokens |
| `--presence-penalty` | Penalty on any previously seen token |
| `--seed` | RNG seed for deterministic output |
| `--sampler` | Named preset (`relaxed`, `creative`, `precise`, `antirep`) |
| `--thinking` | Enable `<think>` reasoning blocks |
| `--no-thinking` | Suppress thinking; inject `/no_think` |
| `--max-tokens` | Max tokens per response (both conversations) |

---

## Post-processing

`trim_thinking_blocks()` strips `<think>…</think>` content before anything crosses the conversation boundary in either direction. The full un-stripped `TurnResponse` is used locally for token counting; only the cleaned text enters the log and the inter-LLM channel.

---

## Key Limitations (current state)

1. **Single shared model** — guide and character use the same model weights, same `ConversationConfig`. No way to assign different models, temperatures, or token budgets per role at the CLI level.
2. **No resume** — a killed run cannot be continued from where it stopped. The log is complete up to the last flush, but there is no `--resume` flag to reload it and continue generation.
3. **Waypoints are one-shot and linear** — each waypoint is consumed once in strict order. No repetition, weighting, or conditional branching based on character responses.
4. **No actual branching** — despite the name "tree generator", all turns form a single linear path. The `ConversationTree` API in the library is unused.
5. **Both conversations share one builder** — character and guide cannot be independently tuned (e.g. a more constrained guide with a freer character).
6. **No mid-run injection** — there is no mechanism to push new waypoints or override guide behavior while a run is in progress.
