# Roleplay Engine (RPE) — Design Document
**Status:** Draft v0.7  
**Model Target:** Qwen3-30B-MoE (non-thinking mode, narrative turns)  
**Crate:** Internal Candle crate — not a standalone binary

---

## 1. Overview

The Roleplay Engine (RPE) is an internal Candle crate that translates structured player inputs into narrated story turns. Input arrives from a chat application as multi-line text strings. Clap parses each line via `Command::try_parse_from()` on a whitespace-split iterator — full flag and subcommand semantics without a binary entrypoint.

The narrator is a **conversation** — the full message history is maintained in the KV cache. The model remembers everything that has happened because it is all in prior turns. There is no state block injected per turn, and no state delta expected in the output. The model's output is narrative prose only.

Each turn, the caller serialises `Vec<Input>` as a JSON array and appends it as the next user message in the conversation. The model responds with narrative. The crate exposes three things: `parse_turn(input: &str, config: &SessionConfig) -> Result<Vec<Input>, ParseError>`, `text_to_inputs(text: &str, author: Option<&str>, max_attempts: usize) -> Result<Vec<Input>, ConvertError>`, and `SessionConfig` for initialisation.

---

## 2. Architecture Summary

```
Player Input (chat app)          NPC Character LLM (prose output)
     │  "act takes your hand          │  "Voss hesitates, then slaps Bob..."
      \  say we have to go"           │
       \                              │
        ▼                             ▼
┌─────────────────┐         ┌───────────────────────┐
│  parse_turn()   │         │  text_to_inputs()      │  waypoint mode,
│  + SessionConfig│         │  author: None          │  NPC prose → Vec<Input>
└────────┬────────┘         └──────────┬────────────┘
         │ Vec<Input>                  │ Vec<Input>
         │                            │ + NPC prose verbatim
         ▼                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Conversation (KV cache)                                         │
│  [system]                                                        │
│  [user]      turn 1 player inputs JSON          ┐ real turns    │
│  [assistant] narrative 1                        ┘               │
│  [user]      NPC LLM output as Vec<Input> JSON  ┐ synthetic     │
│  [assistant] NPC LLM prose verbatim             ┘ exchange      │
│  [user]      turn 2 player inputs JSON          ┐ real turns    │
│  [assistant] narrative 2                        ┘               │
│  [user]      turn N inputs JSON   ← appended here               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Inference Stack  │  Qwen3-30B-MoE, non-thinking mode
                    └────────┬─────────┘
                             │ narrative prose
                             ▼
                         Chat App Output
```

---

## 3. Input Pipeline

### 3.1 Design Principle

`parse_turn(input: &str, config: &SessionConfig) -> Result<Vec<Input>, ParseError>` is the crate's entry point. The host chat application passes the raw multi-line string from the user's input box. `parse_turn` splits on newlines, calls `Command::try_parse_from(line.split_whitespace())` for each non-empty line, and immediately resolves default characters using `config`. By the time `Vec<Input>` is produced, every `character` field is a concrete name — `null` never appears in the output.

`no_binary_name = true` is set on the root command so no dummy argv[0] is required.

The narrator engine's input contract is `Vec<Input>` serialised as a JSON array, appended as a user message in the conversation. Test inputs can be authored as raw JSON and fed directly, bypassing `parse_turn` entirely.

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
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
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

`Command` is not serialised directly. `Input` is the crate's wire format. All `character` fields are `String` — never `Option`. Default resolution happens during conversion, using the `SessionConfig` passed into `parse_turn`. `null` is not a valid value in the `Input` JSON schema.

```rust
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Input {
    Say   { character: String, text: String },
    Act   { character: String, action: String },
    Scene { description: String },
    Cue   { character: String, action: String },
    Beat  { description: String },
}

/// Internal conversion — requires session config to resolve defaults.
/// Not exposed publicly; called only from parse_turn.
fn command_to_input(cmd: Command, config: &SessionConfig) -> Input {
    match cmd {
        Command::Say { character, text } =>
            Input::Say {
                character: character.unwrap_or_else(|| config.protagonist.clone()),
                text: text.join(" "),
            },
        Command::Act { character, action } =>
            Input::Act {
                character: character.unwrap_or_else(|| config.protagonist.clone()),
                action: action.join(" "),
            },
        Command::Scene { description } =>
            Input::Scene { description: description.join(" ") },
        Command::Cue { character, action } =>
            Input::Cue {
                character: character.unwrap_or_else(|| config.persona.clone()),
                action: action.join(" "),
            },
        Command::Beat { description } =>
            Input::Beat { description: description.join(" ") },
    }
}
```

### 3.6 parse_turn Entry Point

```rust
#[derive(Debug)]
pub enum ParseError {
    InvalidCommand { line: String, error: clap::Error },
    EmptyInput,
}

pub fn parse_turn(
    input: &str,
    config: &SessionConfig,
) -> Result<Vec<Input>, ParseError> {
    let lines: Vec<&str> = input.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();

    if lines.is_empty() {
        return Err(ParseError::EmptyInput);
    }

    lines.iter()
        .map(|line| {
            RpeCommand::try_parse_from(line.split_whitespace())
                .map(|parsed| command_to_input(parsed.command, config))
                .map_err(|e| ParseError::InvalidCommand {
                    line: line.to_string(),
                    error: e,
                })
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

The narrator does not directly observe or interact with the NPC character LLM. Instead, the engine serialises NPC LLM output into the narrator's conversation history as a **synthetic exchange** — a fabricated user/assistant pair that the narrator treats as a completed prior turn.

### 4.1 Synthetic Exchange Pattern

When the NPC LLM produces a prose response, the engine:

1. Passes the prose to `text_to_inputs(prose, None, max_attempts)` — waypoint mode, no author.
2. Serialises the resulting `Vec<Input>` as a JSON array → the **synthetic user message**.
3. Takes the NPC LLM's original prose verbatim → the **synthetic assistant message**.
4. Appends both to the narrator conversation in order before the next real player turn.

```
[user]      [{ "type": "act", "character": "Voss", "action": "slaps Bob" }]
[assistant] Voss's hand moves faster than Bob can react. The slap lands
            hard — the sound of it fills the room.
```

The narrator sees a completed exchange. It knows Voss slapped Bob because it is in the conversation history as something it already narrated. When the next player turn arrives, the narrator continues naturally without re-narrating that event.

### 4.2 Serialisation Order

Player inputs, NPC LLM outputs, and engine `cue` overrides are all serialised in event order — no distinction is made by input source. The engine is responsible for ordering them correctly within a turn before they reach the narrator. A typical turn sequence:

```
1. NPC LLM produces prose response to prior player action
2. Engine converts NPC prose → synthetic exchange, appends to conversation  
3. Engine collects player's next input from chat app
4. Engine applies any cue overrides
5. All player inputs serialised as Vec<Input> JSON → real user turn appended
6. Narrator inference runs
```

### 4.3 Why This Works

The narrator never double-narrates. NPC actions are injected as already-narrated history. The narrator's next generation picks up from that established context and continues forward. The conversation history is always consistent — every assistant turn is either genuine narrator output or verbatim NPC LLM prose, and the corresponding user turn is always the structured `Vec<Input>` that describes what happened.

The `text_to_inputs` converter is doing real structural work here — it extracts what mechanically occurred from the NPC's prose so the narrator has a clean JSON description of events, while the NPC's own prose becomes the canonical narration of those events.

---

## 5. Turn Assembler

The Turn Assembler is the only component between `parse_turn` and the conversation. It has two responsibilities:

**Sliding window management** — before appending the new user turn, checks the conversation length against `SessionConfig.max_turns`. If the cap is exceeded, the oldest turn pair (user + assistant) is dropped. The system prompt and the world/scene initialisation message are never dropped.

**Serialisation** — `serde_json::to_string(&inputs)` produces the user message content. No prose construction, no merging, no templates, no character resolution. By the time `Vec<Input>` reaches the Turn Assembler, all fields are already concrete.

---

## 6. System Prompt

The system prompt is set once at session initialisation and never changes. It is generic and world-agnostic — world and scene details are given in the first user message at conversation start.

### 6.1 Narrator Role

```
You are a Narrative Engine for a collaborative story. Your role is to
narrate — not to author. You do not decide what characters do or say.
You only narrate what you are explicitly given.

Each turn you receive a JSON array of inputs. Each input has a "type":

  "say"   — a character speaks. Render as dialogue in your narrative.
  "act"   — a character performs an action. Narrate it.
  "scene" — a stage direction describing the environment. Incorporate it.
  "cue"   — a forced character action. Narrate it as happening. The character
            does not resist, deliberate, or refuse. It happens.
  "beat"  — a steering hint for this turn only. Use it to guide the direction
            of your narrative without stating it explicitly. Do not narrate
            the beat itself.

Narrate only what is described in the inputs for this turn. Do not invent
actions, dialogue, or decisions for any character beyond what is given.
Do not have any character speak, act, or react unless they appear in the
inputs. Characters not present in the inputs are absent from your narration.

Narrate all inputs together as a single cohesive scene continuation.
Write in present tense, third person, 2–6 sentences unless a major
transition warrants more.

You have memory of recent turns in this conversation.
Do not contradict anything established in prior turns.
```

### 6.2 World and Scene Initialisation

The first user message in the conversation is not a turn — it is the world and scene setup. It is plain prose or structured description authored by the caller:

```
World: [name], [tone], [era].
World rules: [rule 1]. [rule 2].

Scene: [location]. [atmosphere].

Characters present:
- [PC name]: [brief description]
- [NPC name]: [role], [disposition toward PC], [relevant history]

The scene opens: [opening narration or situation]
```

The model's first assistant message acknowledges the setup and opens the scene. All subsequent turns are `Vec<Input>` JSON.

---

## 7. Session Initialisation

```rust
pub struct SessionConfig {
    /// Default speaker for `say` and `act` with no --character
    pub protagonist: String,
    /// Default target for `cue` with no --character.
    /// Update when the model switches primary characters across a scene transition.
    pub persona: String,
    /// Maximum number of turns retained in the conversation window.
    /// When exceeded, the oldest turn pair (user + assistant) is dropped.
    /// The system prompt and scene init message are never dropped.
    /// Default: 7
    pub max_turns: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            protagonist: String::new(),
            persona: String::new(),
            max_turns: 7,
        }
    }
}
```

`SessionConfig` is intentionally minimal. World and scene data live in the conversation's first message, not in the config.

---

## 8. Response Contract

The model returns narrative prose only. No JSON delta, no structured output. The conversation history is the record of what happened.

If the caller needs to extract mechanical information (disposition changes, item transfers, position) that is not implicit in the narrative, it can make a separate lightweight extraction call after the narrative turn — this is outside the RPE's scope.

---

## 9. Text-to-Input Converter

The Text-to-Input Converter transforms free-form prose or stage directions into a valid `Vec<Input>` JSON array. It is a separate single-shot call — not part of the narrator conversation — and uses its own system prompt and correction loop.

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

/// Convert free-form text into a Vec<Input>.
///
/// `author` — controls the conversion mode:
///   None        → waypoint mode: third-person narrative, no perspective assumed
///   Some(name)  → author mode: first-person text written as the named character.
///                 Pass `session_config.protagonist` for live player input.
///
/// Retries up to `max_attempts` times, feeding parse errors back
/// to the model for self-correction on each failure.
/// Default max_attempts: 3
pub async fn text_to_inputs(
    text: &str,
    author: Option<&str>,
    max_attempts: usize,
) -> Result<Vec<Input>, ConvertError>
```

### 9.2 System Prompts

Two system prompts are defined. The waypoint prompt is a constant. The author prompt is a template with `{author}` interpolated at call time.

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
- Parse with `serde_json::from_str::<Vec<Input>>()`
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

```rust
pub async fn text_to_inputs(
    text: &str,
    author: Option<&str>,
    max_attempts: usize,
) -> Result<Vec<Input>, ConvertError> {
    let max_attempts = if max_attempts == 0 { 3 } else { max_attempts };
    let system = match author {
        Some(name) => build_author_system_prompt(name),
        None       => WAYPOINT_SYSTEM_PROMPT.to_string(),
    };
    let mut messages: Vec<Message> = vec![
        Message::system(&system),
        Message::user(text),
    ];

    for attempt in 0..max_attempts {
        let output = infer(&messages).await
            .map_err(|e| ConvertError::InferenceError(e.to_string()))?;

        messages.push(Message::assistant(&output));

        match serde_json::from_str::<Vec<Input>>(&output) {
            Ok(inputs) => return Ok(inputs),
            Err(e) if attempt + 1 < max_attempts => {
                let correction = format!(
                    "The JSON you produced failed to parse. Fix it and                      return only the corrected JSON array — no explanation.                     \n\nParse error: {e}\n\nYour output was:\n{output}"
                );
                messages.push(Message::user(&correction));
            }
            Err(e) => {
                return Err(ConvertError::MaxAttemptsExceeded {
                    last_error: e.to_string(),
                    last_output: output,
                });
            }
        }
    }
    unreachable!()
}

fn build_author_system_prompt(author: &str) -> String {
    AUTHOR_SYSTEM_PROMPT_TEMPLATE
        .replace("{author}", author)
}
```

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

- Two system prompts are selected at call time based on `author`. Both are static except the author prompt which has `{author}` interpolated before the call. This keeps schema and context in the same high-attention position.
- The converter uses non-thinking mode. This is a structured extraction task — reasoning tokens add latency with no benefit.
- Third-person conversion ("I take" → "takes") is instructed in the system prompt. The model handles this reliably for simple first-person prose. Complex constructions ("I had been thinking about whether to...") may need the caller to pre-normalise the text if precision matters.
- The converter never produces `character: null`. Unnamed or unidentifiable characters are represented as `"unknown"`. The `parse_turn` path resolves defaults at parse time via `SessionConfig`, so null never appears anywhere in the `Input` JSON schema.