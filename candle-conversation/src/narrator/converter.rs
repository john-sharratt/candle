//! Text-to-inputs converter for the narrator module.
//!
//! Wraps a single-use [`Sequence`] with a correction loop that retries
//! JSON parsing failures up to `max_attempts` times, feeding the exact error
//! and prior output back to the model on each retry.

use std::io::Write;

use crate::{
    handle::{TokenDecoder, TurnEvent},
    ConversationEngine, ConversationError, Sequence, SequenceConfig,
};

use super::error::ConvertError;
use super::types::NarratorInput;

// ── Embedded system prompts ───────────────────────────────────────────────────

const WAYPOINT_SYSTEM_PROMPT: &str = include_str!("../prompts/narrator_waypoint.md");
const AUTHOR_SYSTEM_PROMPT_TEMPLATE: &str = include_str!("../prompts/narrator_author.md");
const RESPONSE_SYSTEM_PROMPT_TEMPLATE: &str = include_str!("../prompts/narrator_response.md");
const CHARACTER_SYSTEM_PROMPT_TEMPLATE: &str = include_str!("../prompts/narrator_character.md");
const NARRATOR_SYSTEM_PROMPT: &str = include_str!("../prompts/guide_narrate.md");

// ── Helpers ───────────────────────────────────────────────────────────────────

fn build_author_system_prompt(author: &str) -> String {
    AUTHOR_SYSTEM_PROMPT_TEMPLATE.replace("{author}", author)
}

fn build_response_system_prompt(character: &str) -> String {
    RESPONSE_SYSTEM_PROMPT_TEMPLATE.replace("{character}", character)
}

pub fn character_system_prompt(persona: &str, protagonist: &str) -> String {
    CHARACTER_SYSTEM_PROMPT_TEMPLATE
        .replace("{persona}", persona)
        .replace("{protagonist}", protagonist)
}

pub fn narrate_system_prompt() -> String {
    NARRATOR_SYSTEM_PROMPT.to_string()
}

fn send_to_converter(conv: &mut Sequence, message: &str) -> Result<String, ConvertError> {
    conv.send_turn(message)
        .map(|r| r.text)
        .map_err(|e: ConversationError| ConvertError::InferenceError(e.to_string()))
}

/// Like [`send_to_converter`] but streams decoded tokens to stderr in real time.
fn send_to_converter_streaming(
    conv: &mut Sequence,
    message: &str,
    decoder: &TokenDecoder,
) -> Result<String, ConvertError> {
    let handle = conv
        .submit_turn(message)
        .map_err(|e: ConversationError| ConvertError::InferenceError(e.to_string()))?;

    let mut all_tokens: Vec<u32> = Vec::new();
    let mut response = None;

    for event in handle.stream() {
        match event {
            TurnEvent::Token(id) => {
                let prev_len = decoder.decode(&all_tokens).len();
                all_tokens.push(id);
                let full = decoder.decode(&all_tokens);
                // Append only the newly decoded suffix — avoids \r line-wrap mess.
                eprint!("{}", &full[prev_len..]);
                std::io::stderr().flush().ok();
            }
            TurnEvent::Done(resp) => {
                response = Some(resp);
            }
            TurnEvent::Error(e) => {
                return Err(ConvertError::InferenceError(e.to_string()));
            }
            _ => {}
        }
    }
    eprintln!(); // newline after streamed output

    let resp = response
        .ok_or_else(|| ConvertError::InferenceError("no response from converter".to_string()))?;

    conv.finish_turn(handle, &resp)
        .map_err(|e: ConversationError| ConvertError::InferenceError(e.to_string()))?;

    Ok(resp.text)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Strip common model output slop before JSON parsing:
/// - markdown code fences (```json ... ``` or ``` ... ```)
/// - any text after the final closing `]`
fn sanitize_json_output(raw: &str) -> std::borrow::Cow<'_, str> {
    let s = raw.trim();
    // Strip code fences.
    let s = if let Some(inner) = s.strip_prefix("```json").or_else(|| s.strip_prefix("```")) {
        inner.trim_start()
    } else {
        s
    };
    let s = if let Some(inner) = s.strip_suffix("```") {
        inner.trim_end()
    } else {
        s
    };
    // Truncate anything after the last `]`.
    if let Some(end) = s.rfind(']') {
        std::borrow::Cow::Owned(s[..=end].to_string())
    } else {
        std::borrow::Cow::Borrowed(s)
    }
}

/// Build the correction message appended when the model's JSON fails to parse.
///
/// Exposed publicly so callers can inspect or test the exact message format.
pub fn correction_message(error: &str, last_output: &str) -> String {
    format!(
        "The JSON you produced failed to parse. Fix it and return only the \
corrected JSON array — no explanation.\n\nParse error: {error}\n\nYour output was:\n{last_output}"
    )
}

/// Return the system prompt appropriate for the given `author` mode.
///
/// - `None` → waypoint mode (third-person narrative, no perspective assumed).
/// - `Some(name)` → author mode (first-person text from `name`'s perspective).
/// Selects the system prompt and conversion strategy for [`text_to_inputs`].
#[derive(Debug, Clone)]
pub enum ConverterMode<'a> {
    /// Third-person narrative → full waypoints (scene, say, act, cue, beat).
    Waypoint,
    /// First-person text from `name`'s perspective → full waypoints.
    Author(&'a str),
    /// Character response text → compact summary (say + act only, 1–3 entries).
    ///
    /// Used to extract only the externally-observable actions and spoken words
    /// from a character's response before feeding it back to the narrator.
    Response(&'a str),
}

///
/// Exposed so callers can create a [`Sequence`] with the correct system
/// prompt before passing it to [`text_to_inputs`].
pub fn converter_system_prompt(mode: &ConverterMode<'_>) -> String {
    match mode {
        ConverterMode::Waypoint => WAYPOINT_SYSTEM_PROMPT.to_string(),
        ConverterMode::Author(name) => build_author_system_prompt(name),
        ConverterMode::Response(character) => build_response_system_prompt(character),
    }
}

/// Convert free-form prose into a [`Vec<Input>`] using a model correction loop.
///
/// Creates a fresh single-use [`Sequence`] internally with the appropriate
/// system prompt, sends the source text as the first user message, and tries
/// to parse the output as a JSON array of [`Input`] objects. If parsing fails,
/// a correction message is appended and the model retries — up to
/// `max_attempts` times total.
///
/// # Parameters
///
/// - `text` — the source prose to convert.
/// - `mode` — conversion mode; see [`ConverterMode`].
/// - `max_attempts` — maximum inference attempts (0 → treated as 3).
/// - `engine` — the [`ConversationEngine`] used to create the single-use
///   converter conversation.
/// - `config` — [`SequenceConfig`] for the converter conversation.
///   Use `SamplingConfig::argmax()` in tests and non-thinking mode for
///   production.
///
/// # Errors
///
/// - [`ConvertError::MaxAttemptsExceeded`] — never produced valid JSON.
/// - [`ConvertError::InferenceError`] — a [`ConversationError`] from the
///   scheduler.
pub fn text_to_inputs(
    text: &str,
    mode: ConverterMode<'_>,
    max_attempts: usize,
    engine: &ConversationEngine,
    config: SequenceConfig,
) -> Result<Vec<NarratorInput>, ConvertError> {
    let max_attempts = if max_attempts == 0 { 3 } else { max_attempts };
    let system = converter_system_prompt(&mode);

    let mut conv = engine
        .new_conversation(&system, config)
        .map_err(|e| ConvertError::InferenceError(e.to_string()))?;

    // First turn: raw text → model JSON output.
    let raw = send_to_converter(&mut conv, text)?;
    let mut last_output = sanitize_json_output(&raw).into_owned();

    match serde_json::from_str::<Vec<NarratorInput>>(&last_output) {
        Ok(inputs) => return Ok(inputs),
        Err(_) if max_attempts == 1 => {
            // No retries allowed; fall through to the attempt-exceeded error.
        }
        Err(_) => {
            // Correction loop: feed parse errors back to the model.
            for attempt in 1..max_attempts {
                let parse_err = match serde_json::from_str::<Vec<NarratorInput>>(&last_output) {
                    Ok(inputs) => return Ok(inputs),
                    Err(e) => e,
                };

                let correction = correction_message(&parse_err.to_string(), &last_output);
                let raw = send_to_converter(&mut conv, &correction)?;
                last_output = sanitize_json_output(&raw).into_owned();

                if attempt + 1 == max_attempts {
                    break;
                }
            }
        }
    }

    // Final parse attempt after correction loop.
    match serde_json::from_str::<Vec<NarratorInput>>(&last_output) {
        Ok(inputs) => Ok(inputs),
        Err(e) => Err(ConvertError::MaxAttemptsExceeded {
            last_error: e.to_string(),
            last_output,
        }),
    }
}

/// Like [`text_to_inputs`] but streams each converter attempt to stderr so
/// the caller can see what the model is producing in real time.
pub fn text_to_inputs_streaming(
    text: &str,
    mode: ConverterMode<'_>,
    max_attempts: usize,
    engine: &ConversationEngine,
    config: SequenceConfig,
    decoder: &TokenDecoder,
) -> Result<Vec<NarratorInput>, ConvertError> {
    let max_attempts = if max_attempts == 0 { 3 } else { max_attempts };
    let system = converter_system_prompt(&mode);

    let mut conv = engine
        .new_conversation(&system, config)
        .map_err(|e| ConvertError::InferenceError(e.to_string()))?;

    let raw = send_to_converter_streaming(&mut conv, text, decoder)?;
    let mut last_output = sanitize_json_output(&raw).into_owned();

    match serde_json::from_str::<Vec<NarratorInput>>(&last_output) {
        Ok(inputs) => return Ok(inputs),
        Err(_) if max_attempts == 1 => {}
        Err(_) => {
            for attempt in 1..max_attempts {
                let parse_err = match serde_json::from_str::<Vec<NarratorInput>>(&last_output) {
                    Ok(inputs) => return Ok(inputs),
                    Err(e) => e,
                };
                eprintln!("  [converter retry {attempt}: {parse_err}]");
                let correction = correction_message(&parse_err.to_string(), &last_output);
                let raw = send_to_converter_streaming(&mut conv, &correction, decoder)?;
                last_output = sanitize_json_output(&raw).into_owned();

                if attempt + 1 == max_attempts {
                    break;
                }
            }
        }
    }

    match serde_json::from_str::<Vec<NarratorInput>>(&last_output) {
        Ok(inputs) => Ok(inputs),
        Err(e) => Err(ConvertError::MaxAttemptsExceeded {
            last_error: e.to_string(),
            last_output,
        }),
    }
}
