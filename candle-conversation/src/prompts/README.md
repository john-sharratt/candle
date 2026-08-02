# candle-conversation/src/prompts/

Fourteen Markdown prompt templates for the NPC cognitive-lifecycle and
narrator machinery, embedded at compile time via `include_str!`.

## What it does

Every file here is a system prompt (or a template with `{placeholder}`
tokens filled in by the caller before use). They are treated as code, not
content: `mod.rs` warns that changing any `.md` file invalidates downstream
KV caches, because the system prompt is prefilled once and pinned in BF16 for
the lifetime of a conversation — a changed prompt string produces a different
KV fingerprint.

Two files load their prompts here (`mod.rs` and
`candle-conversation/src/narrator/converter.rs`), each re-declaring its own
`include_str!` constants rather than importing from the other — there is
duplication for `narrator_waypoint.md` and `narrator_author.md`, which are
embedded by both modules independently.

`guide_reflect.md` exists on disk and is described in the archived
`docs/archived/tree_gen_design.md`, but no `include_str!` in the current
source tree loads it — it is not wired into any live pipeline.

## The templates

**Cognitive lifecycle** (`mod.rs`, used by the character's background
processes — see `docs/npc_mind_design.md` / `docs/theory_of_the_mind.md`):

| File | Constant | Fires when |
|---|---|---|
| `daydream.md` | `DAYDREAM_PROMPT` | A cold substrate node crosses the resonance threshold during Reality decode; produces a short (50–100 word) unbidden associative thought. |
| `sleep.md` | `SLEEP_PROMPT` | The end-of-day prospective sleep batch; elaborates a memory seed into a 100–150 word dream. |
| `reason.md` | `REASON_PROMPT` | The executive self-dialogue turn; produces a plan (max 150 words) that gets injected into future Reality system prompts. |
| `summarize.md` | `SUMMARIZE_PROMPT` | A temporary scheduler slot compresses a window of turns into a `ConversationSegment` (max 200 words, past tense, no editorializing). |
| `temporal_marker.md` | `TEMPORAL_MARKER_POSTFIX` | Appended to the system prompt whenever `ConversationTreeConfig::temporal_markers_enabled` is set; teaches the model the `[T-{days}.{seq}]` marker format. |

**`tree_gen` life-timeline pipeline** (`mod.rs`, driven by
`candle-conversation/examples/tree_gen.rs` — see `docs/archived/tree_gen_design.md`):

| File | Constant | Fires when |
|---|---|---|
| `guide_summarize_period.md` | `GUIDE_SUMMARIZE_PERIOD_PROMPT` | Once per named timeline period; synthesises a Life Story + Cast section from dated timeline entries, building on any prior period's background. |
| `guide_today.md` | `GUIDE_TODAY_PROMPT` | Once per story day; plans a single day as 5–15 ordered concrete waypoints (DATE/DESCRIPTION/YESTERDAY/LAST_MONTH in, waypoint list out). |
| `director.md` | `DIRECTOR_PROMPT` | Once per waypoint (fresh, single-shot); writes SETTING/PRESENT/ACTION stage directions the character will react to, matching the waypoint's emotional temperature. |
| `guide_reflect.md` | *(not currently wired)* | Documented in `tree_gen_design.md` as a calibration step comparing where the character's last response landed against what the waypoint actually was; no current code path loads it. |

**Narrator / text-to-input conversion** (`candle-conversation/src/narrator/converter.rs`
— structured `Input` JSON in and out of narrative prose; see `docs/narrative_engine.md`):

| File | Constant | Fires when |
|---|---|---|
| `narrator_waypoint.md` | `WAYPOINT_SYSTEM_PROMPT` / `NARRATOR_WAYPOINT_SYSTEM_PROMPT` | `text_to_inputs` is called with `author: None`; converts third-person narrative prose into a JSON array of `Input` objects (say/act/scene/cue). |
| `narrator_author.md` | `AUTHOR_SYSTEM_PROMPT_TEMPLATE` / `NARRATOR_AUTHOR_SYSTEM_PROMPT_TEMPLATE` | `text_to_inputs` is called with `author: Some(name)`; same conversion, but resolves first-person references ("I", "me") to the named author character. Contains a literal `{author}` placeholder. |
| `narrator_response.md` | `RESPONSE_SYSTEM_PROMPT_TEMPLATE` | Extracting a character's in-character reply down to only its externally-observable say/act events, discarding internal narration. Contains a `{character}` placeholder. |
| `narrator_character.md` | `CHARACTER_SYSTEM_PROMPT_TEMPLATE` | Driving the actual in-character response generation (`character_system_prompt`); constrains the model to a maximum 2-sentence, first-person, no-other-character reply. Contains `{persona}` and `{protagonist}` placeholders. |
| `guide_narrate.md` | `NARRATOR_SYSTEM_PROMPT` (`narrate_system_prompt`) | Rendering a turn's `Vec<Input>` (say/act/scene/cue/beat) as second-person-present narrative prose addressed to the protagonist. |

## Related docs

`docs/theory_of_the_mind.md`, `docs/npc_mind_design.md`,
`docs/narrative_engine.md`, `docs/archived/tree_gen_design.md`,
`docs/immutable_summary_forest.md` (the summary structure that
`summarize.md`-produced segments feed into).
