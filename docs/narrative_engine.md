# Roleplay Engine (RPE) — Design Document
**Status:** Draft v0.7  
**Model Target:** Qwen3-30B-MoE (the converter is recommended non-thinking; the narrator's mode is the caller's `SequenceConfig`)  
**Crate:** Internal Candle crate — not a standalone binary (the `narrator` module of `candle-conversation`, `candle-conversation/src/narrator/`)

---

## 1. Overview

The Roleplay Engine (RPE) is an internal Candle crate that translates structured player inputs into narrated story turns. Input arrives from a chat application as multi-line text strings. Clap parses each line via `RpeCommand::try_parse_from()` on a whitespace-split iterator — full flag and subcommand semantics without a binary entrypoint.

The narrator is a **conversation** — the full message history is maintained in the KV cache. The model remembers everything that has happened because it is all in prior turns. There is no state block injected per turn, and no state delta expected in the output. The model's output is narrative prose only.

Each turn, the caller serialises `Vec<NarratorInput>` as a JSON array and appends it as the next user message in the conversation. The model responds with narrative. The crate exposes three things: `parse_turn(input: &str, config: &SessionConfig) -> Result<Vec<NarratorInput>, ParseError>`, `text_to_inputs(text: &str, mode: ConverterMode<'_>, max_attempts: usize, engine: &ConversationEngine, config: SequenceConfig) -> Result<Vec<NarratorInput>, ConvertError>`, and `SessionConfig` for initialisation.

---

## 2. Architecture Summary

```
Player Input (chat app)          NPC Character LLM (prose output)
     │  "act takes your hand          │  "Voss hesitates, then slaps Bob..."
      \  say we have to go"           │
       \                              │
        ▼                             ▼
┌─────────────────┐         ┌───────────────────────┐
│  parse_turn()   │         │  text_to_inputs()      │  ConverterMode::Response,
│  + SessionConfig│         │  (response mode)       │  NPC prose → 1–3 say/act entries
└────────┬────────┘         └──────────┬────────────┘
         │ Vec<NarratorInput>                  │ Vec<NarratorInput>
         │                            │
         ▼                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Conversation (KV cache)                                         │
│  [system]      prompts/guide_narrate.md                          │
│  [user]      turn 1 player inputs JSON          ┐ real turns    │
│  [assistant] narrative 1                        ┘               │
│  [user]      NPC response as Vec<NarratorInput> JSON ┐ real turn │
│  [assistant] narrator's own prose of it              ┘          │
│  [user]      turn 2 player inputs JSON          ┐ real turns    │
│  [assistant] narrative 2                        ┘               │
│  [user]      turn N inputs JSON   ← appended here               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Inference Stack  │  Qwen3-30B-MoE
                    └────────┬─────────┘
                             │ narrative prose
                             ▼
                         Chat App Output
```

---

## 3. Input Pipeline

### 3.1 Design Principle

`parse_turn(input: &str, config: &SessionConfig) -> Result<Vec<NarratorInput>, ParseError>` is the crate's entry point. The host chat application passes the raw multi-line string from the user's input box. `parse_turn` splits on newlines, calls `RpeCommand::try_parse_from(line.split_whitespace())` for each non-empty line, and immediately resolves default characters using `config`. By the time `Vec<NarratorInput>` is produced, every `character` field is a concrete name — `null` never appears in the output.

`no_binary_name = true` is set on the root command so no dummy argv[0] is required.

The narrator engine's input contract is `Vec<NarratorInput>` serialised as a JSON array, appended as a user message in the conversation. Test inputs can be authored as raw JSON and fed directly, bypassing `parse_turn` entirely.

### 3.2 Command Set

Five commands cover all input. Natural language in the free-text field carries semantic meaning — no typed sub-categories are needed.

| Command | `--character` default | Meaning |
|---|---|---|
| `say` | session PC | A character speaks dialogue |
| `act` | session PC | A character performs an action |
| `scene` | n/a | Environment or world stage direction |
| `cue` | session LLM character | Forces a character to do something regardless of disposition |
| `beat` | n/a | Optional single-turn narrative steering hint |

`say` and `act` default to the **PC** when `--character` is absent.  
`cue` defaults to the **LLM character** — whoever the model is voicing this scene — when `--character` is absent.  
`scene` and `beat` have no character field.  
`beat` is optional and single-turn — whether to include it, and what it says, is entirely the caller's decision. It may be the same every turn, change each turn, or be omitted entirely.

### 3.3 Argument Types

All free-text fields use `trailing_var_arg = true` with `Vec<String>`. Clap consumes all named flags first, then collects every remaining token. No quotes needed for free-text content. Multi-word `--character` values must be quoted: `--character "Old Man"`.

| Field | Command(s) |
|---|---|
| `text` | `say` |
| `action` | `act`, `cue` |
| `description` | `scene`, `beat` |

### 3.4 Clap Command Definitions (Rust)

```rust
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(no_binary_name = true)]
struct RpeCommand {
    #[command(subcommand)]
    command: RpeSubcommand,
}

#[derive(Subcommand, Debug)]
enum RpeSubcommand {
    /// Speak dialogue as a character (defaults to session PC)
    Say {
        #[arg(short, long)]
        character: Option<String>,
        #[arg(trailing_var_arg = true, num_args = 1..)]
        text: Vec<String>,
    },

    /// Perform a physical action as a character (defaults to session PC)
    Act {
        #[arg(short, long)]
        character: Option<String>,
        #[arg(trailing_var_arg = true, num_args = 1..)]
        action: Vec<String>,
    },

    /// Describe what is physically happening in the environment right now
    Scene {
        #[arg(trailing_var_arg = true, num_args = 1..)]
        description: Vec<String>,
    },

    /// Force a character to do something regardless of disposition
    /// (defaults to session LLM character)
    Cue {
        #[arg(short, long)]
        character: Option<String>,
        #[arg(trailing_var_arg = true, num_args = 1..)]
        action: Vec<String>,
    },

    /// Optional single-turn narrative steering hint
    Beat {
        #[arg(trailing_var_arg = true, num_args = 1..)]
        description: Vec<String>,
    },
}
```

### 3.5 Serialisation Layer

`RpeSubcommand` is not serialised directly. `NarratorInput` (`candle-conversation/src/narrator/types.rs`) is the crate's wire format. All `character` fields are `String` — never `Option`. Default resolution happens during conversion, using the `SessionConfig` passed into `parse_turn`. `null` is not a valid value in the `Input` JSON schema.

```rust
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NarratorInput {
    Say   { character: String, text: String },
    Act   { character: String, action: String },
    Scene { description: String },
    Cue   { character: String, action: String },
    Beat  { description: String },
}

/// Internal conversion — requires session config to resolve defaults.
/// Not exposed publicly; called only from parse_turn.
fn command_to_input(cmd: RpeSubcommand, config: &SessionConfig) -> NarratorInput {
    match cmd {
        RpeSubcommand::Say { character, text } =>
            NarratorInput::Say {
                character: character.unwrap_or_else(|| config.protagonist.clone()),
                text: text.join(" "),
            },
        RpeSubcommand::Act { character, action } =>
            NarratorInput::Act {
                character: character.unwrap_or_else(|| config.protagonist.clone()),
                action: action.join(" "),
            },
        RpeSubcommand::Scene { description } =>
            NarratorInput::Scene { description: description.join(" ") },
        RpeSubcommand::Cue { character, action } =>
            NarratorInput::Cue {
                character: character.unwrap_or_else(|| config.persona.clone()),
                action: action.join(" "),
            },
        RpeSubcommand::Beat { description } =>
            NarratorInput::Beat { description: description.join(" ") },
    }
}
```

### 3.6 parse_turn Entry Point

```rust
#[derive(Debug)]
pub enum ParseError {
    InvalidCommand { line: String, error: clap::Error },
    EmptyInput,
    /// A valid command whose content field is empty.
    EmptyContent { line: String, field: &'static str },
}

pub fn parse_turn(
    input: &str,
    config: &SessionConfig,
) -> Result<Vec<NarratorInput>, ParseError> {
    let lines: Vec<&str> = input.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();

    if lines.is_empty() {
        return Err(ParseError::EmptyInput);
    }

    // `shell_split` is a whitespace split that keeps a quoted span as one
    // token, so `--character "Old Man"` works without a shell.
    lines.iter()
        .map(|line| {
            let input = RpeCommand::try_parse_from(shell_split(line))
                .map(|parsed| command_to_input(parsed.command, config))
                .map_err(|e| ParseError::InvalidCommand {
                    line: line.to_string(),
                    error: e,
                })?;
            if let Some(field) = empty_content_field(&input) {
                return Err(ParseError::EmptyContent { line: line.to_string(), field });
            }
            Ok(input)
        })
        .collect()
}
```

### 3.7 Canonical Input JSON Schema

`character` is always a concrete name — `null` is not valid in the schema. Resolution of defaults happens inside `parse_turn` using `SessionConfig` before `Input` objects are produced.

```json
{ "type": "say",   "character": "Kael Dorn", "text": "we have to get out of here" }
{ "type": "say",   "character": "Voss",      "text": "That's the third time tonight." }
{ "type": "act",   "character": "Kael Dorn", "action": "takes your hand" }
{ "type": "act",   "character": "Voss",      "action": "reaches for the door" }
{ "type": "scene", "description": "the roof begins to collapse" }
{ "type": "cue",   "character": "Voss",      "action": "draws her sidearm" }
{ "type": "cue",   "character": "Voss",      "action": "steps in front of the exit" }
{ "type": "beat",  "description": "Voss is close to agreeing but needs one more push" }
```

### 3.8 Parse Examples

Assuming `SessionConfig { protagonist: "Kael Dorn", persona: "Voss", .. }`:

```
scene the roof begins to collapse
→ { "type": "scene", "description": "the roof begins to collapse" }

act takes your hand
→ { "type": "act", "character": "Kael Dorn", "action": "takes your hand" }

say we have to get out of here
→ { "type": "say", "character": "Kael Dorn", "text": "we have to get out of here" }

act --character Voss moves towards the door with haste
→ { "type": "act", "character": "Voss", "action": "moves towards the door with haste" }

cue draws her sidearm
→ { "type": "cue", "character": "Voss", "action": "draws her sidearm" }

cue --character Voss steps in front of the exit
→ { "type": "cue", "character": "Voss", "action": "steps in front of the exit" }

beat Voss is close to agreeing but needs one more push
→ { "type": "beat", "description": "Voss is close to agreeing but needs one more push" }
```

### 3.9 Full Turn Input Example

```
scene the roof begins to collapse
act takes your hand
cue draws her sidearm
say we have to get out of here
beat steer toward escape through the east corridor
```

User message appended to conversation:

```json
[
  { "type": "scene", "description": "the roof begins to collapse" },
  { "type": "act",   "character": "Kael Dorn", "action": "takes your hand" },
  { "type": "cue",   "character": "Voss",      "action": "draws her sidearm" },
  { "type": "say",   "character": "Kael Dorn", "text": "we have to get out of here" },
  { "type": "beat",  "description": "steer toward escape through the east corridor" }
]
```

All `character` fields are resolved to concrete names by `parse_turn`. The JSON that reaches the model contains no null values.

---

## 4. NPC LLM Handoff

The narrator does not directly observe or interact with the NPC character LLM. The engine reduces the NPC's response to compact inputs and the narrator narrates them itself, as an ordinary turn.

### 4.1 Response Handoff (as built)

`NarratorEngine::insert_character_response_streaming` (`candle-conversation/src/narrator/engine.rs`) takes the NPC's prose and:

1. Converts it with `text_to_inputs_streaming(char_text, ConverterMode::Response(persona), 3, ...)`. The response prompt (`prompts/narrator_response.md`) keeps only externally observable speech and action, as `say` and `act` entries, one to three in all, with first-person references resolved to the character's name.
2. Runs the narrator on those inputs through `NarratorEngine::next` — a real inference turn. The user message is the `Vec<NarratorInput>` JSON; the assistant message is the narrator's own prose, in second person.
3. Returns that prose to the caller, which may show or discard it.

```
[user]      [{ "type": "act", "character": "Voss", "action": "slaps Bob" }]
[assistant] Voss's hand moves faster than Bob can react. The slap lands
            hard — the sound of it fills the room.     ← narrator-generated
```

Every assistant turn in the history is therefore narrator output in the narrator's voice, and every user turn is a structured `Vec<NarratorInput>`. The converter does the structural work (what mechanically occurred); the narrator does the prose.

> **Replaced design.** The first design appended a *synthetic exchange*: the converted inputs as the user message and the NPC's own prose, verbatim, as the assistant message, with no narrator inference. It was dropped because storing raw first-person character prose as the narrator's assistant output taught the narrator that it generates first-person dialogue, and after a few turns the context was contaminated and the output corrupted (the comment on `insert_character_response_streaming` records this).

### 4.2 Serialisation Order

Player inputs, NPC response inputs, and `cue` overrides are all serialised in event order — no distinction is made by input source. Ordering within a turn is the caller's job. A typical turn sequence:

```
1. NPC LLM produces prose response to prior player action
2. insert_character_response_streaming converts it and runs a narrator turn
3. The caller collects the player's next input from the chat app
4. The caller applies any cue overrides
5. All player inputs serialised as Vec<NarratorInput> JSON → user turn; narrator inference runs
```

---

## 5. Turn Serialisation and Window

There is no separate Turn Assembler component. `NarratorEngine::next` (`narrator/engine.rs`) serialises the inputs with `serde_json::to_string(&inputs)` and sends the result as the next user turn. No prose construction, no merging, no templates, no character resolution: by then every field is already concrete.

**Sliding window.** `NarratorEngine::new` fixes the window at **4 turns**: it sets `context_window_turns = 4` on the sequence config and `set_max_turns(4)` on the conversation tree. The value is hard-coded, not read from `SessionConfig.max_turns` (§7), which is a caller-side setting the engine does not consult.

> **Replaced design.** The first design had a Turn Assembler that dropped the oldest user/assistant pair once the conversation exceeded `SessionConfig.max_turns` (default 7), never dropping the system prompt or the world/scene message. The engine instead owns a fixed 4-turn window.

---

## 6. System Prompt

The system prompt is the file `candle-conversation/src/prompts/guide_narrate.md`, returned by `narrate_system_prompt()` (`narrator/converter.rs`) and handed to `engine.new_conversation` by `NarratorEngine::new`. It is set once and never changes. It is generic and world-agnostic.

### 6.1 Narrator Role (as built)

The prompt tells the narrator it is the narrator of a collaborative story that receives a JSON array of Input objects each turn and renders them as prose **addressed to the protagonist in second person present tense**. It defines the five types (`say`, `act`, `scene`, `cue`, `beat`) as in §3.2, and its rules are:

- **Length:** no more than 3 sentences; stop at the third sentence-ending mark.
- **Fidelity:** every person, object and action named across the inputs must appear; nothing is replaced, omitted or abstracted.
- **Name clarity:** full character names on first mention; pronouns only after the name has appeared.
- **Tone:** match the tone the inputs imply; add no tension or drama they do not contain.
- Do not tell the protagonist what to feel, think or do.
- **Dialogue** is quoted verbatim but always framed by a speech tag, never emitted as a bare sentence.
- No meta-reference to the input format; no headings, timestamps or labels.
- The prompt ends with one worked good/bad example (fidelity).

> **Replaced design.** The first prompt was third person, 2–6 sentences, and told the narrator to narrate only what the inputs give and to leave absent characters out. The built prompt is second person, at most 3 sentences, and its fidelity rule requires every named element to appear.

### 6.2 World and Scene Initialisation

A caller may open the conversation with a world and scene setup message; `NarratorEngine::new` does not send one, so any such message is the caller's. It is plain prose or structured description authored by the caller:

```
World: [name], [tone], [era].
World rules: [rule 1]. [rule 2].

Scene: [location]. [atmosphere].

Characters present:
- [PC name]: [brief description]
- [NPC name]: [role], [disposition toward PC], [relevant history]

The scene opens: [opening narration or situation]
```

The model's first assistant message acknowledges the setup and opens the scene. All subsequent turns are `Vec<NarratorInput>` JSON.

---

## 7. Session Initialisation

```rust
pub struct SessionConfig {
    /// Default speaker for `say` and `act` with no --character
    pub protagonist: String,
    /// Default target for `cue` with no --character.
    /// Update when the model switches primary characters across a scene transition.
    pub persona: String,
    /// Maximum number of turns a caller retains in its window. Enforced by
    /// the caller; `NarratorEngine` uses its own fixed window of 4 (§5).
    /// Default: 7
    pub max_turns: usize,
}

/// Sensible default for `SessionConfig::max_turns`.
pub const DEFAULT_MAX_TURNS: usize = 7;
```

`SessionConfig` (`candle-conversation/src/narrator/types.rs`) has no `Default`
impl; a caller names `protagonist` and `persona` and takes `DEFAULT_MAX_TURNS`
for `max_turns` if it has no preference.

`SessionConfig` is intentionally minimal. World and scene data live in the conversation's first message, not in the config.

---

## 8. Response Contract

The model returns narrative prose only. No JSON delta, no structured output. The conversation history is the record of what happened.

If the caller needs to extract mechanical information (disposition changes, item transfers, position) that is not implicit in the narrative, it can make a separate lightweight extraction call after the narrative turn — this is outside the RPE's scope.

---

## 9. Text-to-Input Converter

The Text-to-Input Converter transforms free-form prose or stage directions into a valid `Vec<NarratorInput>` JSON array. It is a separate single-shot call — not part of the narrator conversation — and uses its own system prompt and correction loop.

The converter operates in two modes selected by the `author` parameter:

**Waypoint mode** (`author: None`) — the source text is third-person narrative describing what characters do. No perspective assumed. Used for scripted waypoints, scene setups, or GM-authored beats: `"Berch enters the room and confronts George about the stolen money."`

**Author mode** (`author: Some("Kael Dorn")`) — the source text is written in first person from the named character’s perspective. First-person pronouns resolve to that character. Used for real-time player input in the chat app: `"\"What are you doing?\" then i slap you."`

Each mode has its own system prompt. The correction loop is identical in both.

### 9.1 Public Interface

```rust
#[derive(Debug)]
pub enum ConvertError {
    /// JSON never parsed successfully within the attempt limit
    MaxAttemptsExceeded { last_error: String, last_output: String },
    /// Inference call failed
    InferenceError(String),
}

/// Convert free-form text into a Vec<NarratorInput>.
///
/// `mode` — controls the conversion:
///   ConverterMode::Waypoint     → third-person narrative, no perspective assumed
///   ConverterMode::Author(name) → first-person text written as the named character.
///                                 Pass `session_config.protagonist` for live player input.
///   ConverterMode::Response(c)  → a character's response text reduced to a compact
///                                 summary (say and act only, one to three entries),
///                                 before it is fed back to the narrator.
///
/// Retries up to `max_attempts` times, feeding parse errors back
/// to the model for self-correction on each failure.
/// Default max_attempts: 3 (a `max_attempts` of 0 means 3)
pub fn text_to_inputs(
    text: &str,
    mode: ConverterMode<'_>,
    max_attempts: usize,
    engine: &ConversationEngine,
    config: SequenceConfig,
) -> Result<Vec<NarratorInput>, ConvertError>
```

### 9.2 System Prompts

Three converter prompts are built: waypoint (a constant), author (template, `{author}`) and response (template, `{character}`; §4.1). The files under `candle-conversation/src/prompts/` are the source of truth; the texts below are the shared core. The built files add to it: an explicit list of unnamed-referent phrases that map to `"unknown"`; a fixed list of cue signal words ("reluctantly", "has no choice", "forced to", "compelled to", "can't resist", ...) that always mean `cue`, with `cue` required to carry an `action`; names preserved exactly as written; every array element must be an object (never a bare string); and double quotes inside string values must be escaped. The author prompt also forces first-person physical actions to `act` for the author, gives each third-party actor its own entry, and treats bare spoken words as a `say` by the author.

#### Waypoint System Prompt (author: None)

```
You are a JSON converter. Your only output is a valid JSON array.
No explanation, no markdown, no code fences. Only the raw JSON array.

You convert third-person narrative text into a JSON array of Input objects.
The text describes what characters do, say, and experience.

Each Input object has a "type" field. The valid types and their fields are:

  { "type": "say",   "character": "<n>", "text": "<dialogue>" }
  { "type": "act",   "character": "<n>", "action": "<action>" }
  { "type": "scene", "description": "<environment description>" }
  { "type": "cue",   "character": "<n>", "action": "<forced action>" }
  { "type": "beat",  "description": "<narrative steering hint>" }

Rules:
- If a character is genuinely unnamed or unidentifiable, use the string
  "unknown" for "character" rather than omitting the field.
- "say" is for spoken dialogue only.
- "act" is for physical actions performed by a character.
- "scene" is for environment, atmosphere, or world state descriptions
  not tied to any character.
- "cue" is for actions a character is forced to perform regardless of
  will or disposition. Signals: "reluctantly", "has no choice",
  "doesn't want to but". Default to "act" when ambiguous.
- "beat" is for authorial steering intent only: "this should lead to...",
  "the goal here is...", "push toward...".
- Preserve the order of events as they appear in the source text.
- One Input object per distinct action, line of dialogue, or scene
  description. Do not merge unrelated events.
- Output only the JSON array. Nothing else.
```

#### Author System Prompt (author: Some(name))

Built at call time with `{author}` replaced by the character name.

```
You are a JSON converter. Your only output is a valid JSON array.
No explanation, no markdown, no code fences. Only the raw JSON array.

You convert first-person text into a JSON array of Input objects.
The text is written from the perspective of a character named "{author}".
All first-person references ("I", "me", "my", "we", "us") refer to {author}.
Resolve them to "{author}" in the "character" field.
Never use "unknown" for {author}’s own actions or dialogue.

Each Input object has a "type" field. The valid types and their fields are:

  { "type": "say",   "character": "<n>", "text": "<dialogue>" }
  { "type": "act",   "character": "<n>", "action": "<action>" }
  { "type": "scene", "description": "<environment description>" }
  { "type": "cue",   "character": "<n>", "action": "<forced action>" }
  { "type": "beat",  "description": "<narrative steering hint>" }

Rules:
- If a third-party character is genuinely unnamed or unidentifiable,
  use the string "unknown" for "character" rather than omitting the field.
- "say" is for spoken dialogue only. Quoted speech is always "say".
- "act" is for physical actions performed by a character.
- "scene" is for environment, atmosphere, or world state descriptions
  not tied to any character.
- "cue" is for actions a character is forced to perform regardless of
  will or disposition. Signals: "reluctantly", "has no choice",
  "doesn't want to but". Default to "act" when ambiguous.
- "beat" is for authorial steering intent only: "this should lead to...",
  "the goal here is...", "push toward...".
- Preserve the order of events as they appear in the source text.
- One Input object per distinct action, line of dialogue, or scene
  description. Do not merge unrelated events.
- Convert first-person verb forms to third person:
  "I take her hand" → "takes her hand".
  "I say" → dialogue goes in "say" with {author} as character.
- Output only the JSON array. Nothing else.
```


### 9.3 Correction Loop

On each attempt, the converter runs a short conversation:

- **Message 1 (user):** the raw input text
- **Message 2 (assistant):** the model's JSON output
- Parse with `serde_json::from_str::<Vec<NarratorInput>>()`
- If parse succeeds → return `Ok(inputs)`
- If parse fails → append a correction message and retry

The correction message feeds the exact error and prior output back to the model:

```
The JSON you produced failed to parse. Fix it and return only the
corrected JSON array — no explanation.

Parse error: {error}

Your output was:
{last_output}
```

The correction appends to the same conversation so the model sees exactly what it wrote and what went wrong — more reliable than restarting fresh.

The loop is `text_to_inputs` in `candle-conversation/src/narrator/converter.rs`.
It opens one single-use conversation on the engine under
`converter_system_prompt(&mode)` and sends the raw text as its first turn. Each
reply is cleaned of code fences (`sanitize_json_output`) and parsed with
`serde_json::from_str::<Vec<NarratorInput>>`; on a failure the exact error and
prior output are sent back as the next turn (`correction_message`), up to
`max_attempts` times, and the last failure is returned as
`ConvertError::MaxAttemptsExceeded`. `text_to_inputs_streaming` is the same loop
streaming each attempt to stderr. The system prompts are the files
`candle-conversation/src/prompts/narrator_waypoint.md`, `narrator_author.md`
(`{author}` substituted by `build_author_system_prompt`) and
`narrator_response.md`.

### 9.4 Conversion Examples

#### Waypoint mode (author: None)

**Input text:**
```
Berch enters the room and confronts George about the stolen money.
```

**Output:**
```json
[
  { "type": "act",  "character": "Berch",  "action": "enters the room" },
  { "type": "act",  "character": "Berch",  "action": "confronts George about the stolen money" }
]
```

**Input text:**
```
Voss doesn't want to but she lowers her weapon. The room goes quiet.
George makes a run for the door.
```

**Output:**
```json
[
  { "type": "cue",   "character": "Voss",   "action": "lowers her weapon" },
  { "type": "scene", "description": "The room goes quiet." },
  { "type": "act",   "character": "George", "action": "makes a run for the door" }
]
```

#### Author mode (author: Some("Kael Dorn"))

**Input text:**
```
I take her hand. The roof is coming down.
We have to get out of here, I tell her.
```

**Output:**
```json
[
  { "type": "act",   "character": "Kael Dorn", "action": "takes her hand" },
  { "type": "scene", "description": "The roof is coming down." },
  { "type": "say",   "character": "Kael Dorn", "text": "We have to get out of here." }
]
```

**Input text:**
```
"What are you doing?", then i slap you
```

**Output:**
```json
[
  { "type": "say", "character": "Kael Dorn", "text": "What are you doing?" },
  { "type": "act", "character": "Kael Dorn", "action": "slaps you" }
]
```

### 9.5 Notes

- The system prompt is selected at call time from `ConverterMode`; the author and response prompts have their placeholder interpolated before the call. This keeps schema and context in the same high-attention position.
- The converter's `SequenceConfig` is the caller's; the doc comment on `text_to_inputs` recommends non-thinking mode for production and `SamplingConfig::argmax()` in tests. This is a structured extraction task — reasoning tokens add latency with no benefit.
- Third-person conversion ("I take" → "takes") is instructed in the system prompt. The model handles this reliably for simple first-person prose. Complex constructions ("I had been thinking about whether to...") may need the caller to pre-normalise the text if precision matters.
- The converter never produces `character: null`. Unnamed or unidentifiable characters are represented as `"unknown"`. The `parse_turn` path resolves defaults at parse time via `SessionConfig`, so null never appears anywhere in the `Input` JSON schema.