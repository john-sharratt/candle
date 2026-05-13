//! Clap-based command parser for the narrator module.
//!
//! Defines the five RPE commands (`say`, `act`, `scene`, `cue`, `beat`),
//! a shell tokeniser that respects quoted spans, and the public
//! [`parse_turn`] entry point.

use clap::{Parser, Subcommand};

use super::error::ParseError;
use super::types::{NarratorInput, SessionConfig};

// ── Clap command definitions ──────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(no_binary_name = true)]
struct RpeCommand {
    #[command(subcommand)]
    command: RpeSubcommand,
}

#[derive(Subcommand, Debug)]
enum RpeSubcommand {
    /// Speak dialogue as a character (defaults to session protagonist).
    Say {
        #[arg(short, long)]
        character: Option<String>,
        #[arg(trailing_var_arg = true, num_args = 1..)]
        text: Vec<String>,
    },
    /// Perform a physical action as a character (defaults to session protagonist).
    Act {
        #[arg(short, long)]
        character: Option<String>,
        #[arg(trailing_var_arg = true, num_args = 1..)]
        action: Vec<String>,
    },
    /// Describe what is physically happening in the environment right now.
    Scene {
        #[arg(trailing_var_arg = true, num_args = 1..)]
        description: Vec<String>,
    },
    /// Force a character to do something regardless of disposition
    /// (defaults to session persona).
    Cue {
        #[arg(short, long)]
        character: Option<String>,
        #[arg(trailing_var_arg = true, num_args = 1..)]
        action: Vec<String>,
    },
    /// Optional single-turn narrative steering hint.
    Beat {
        #[arg(trailing_var_arg = true, num_args = 1..)]
        description: Vec<String>,
    },
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Tokenise a line respecting single- and double-quoted spans.
///
/// `"Old Man"` produces the single token `Old Man`; unquoted words are split
/// by whitespace. This allows multi-word `--character` values without
/// requiring a shell: `say --character "Old Man" you are dismissed`.
fn shell_split(s: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_quote: Option<char> = None;

    for c in s.chars() {
        match in_quote {
            Some(q) if c == q => in_quote = None,
            Some(_) => current.push(c),
            None if c == '"' || c == '\'' => in_quote = Some(c),
            None if c.is_whitespace() => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            None => current.push(c),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Return `Some(field_name)` if the input's required content field is empty.
fn empty_content_field(input: &NarratorInput) -> Option<&'static str> {
    match input {
        NarratorInput::Say { text, .. } if text.trim().is_empty() => Some("text"),
        NarratorInput::Act { action, .. } if action.trim().is_empty() => Some("action"),
        NarratorInput::Scene { description } if description.trim().is_empty() => Some("description"),
        NarratorInput::Cue { action, .. } if action.trim().is_empty() => Some("action"),
        NarratorInput::Beat { description } if description.trim().is_empty() => Some("description"),
        _ => None,
    }
}

/// Resolve a parsed clap command into a concrete [`Input`] by filling in
/// absent `--character` values from `config`.
fn command_to_input(cmd: RpeSubcommand, config: &SessionConfig) -> NarratorInput {
    match cmd {
        RpeSubcommand::Say { character, text } => NarratorInput::Say {
            character: character.unwrap_or_else(|| config.protagonist.clone()),
            text: text.join(" "),
        },
        RpeSubcommand::Act { character, action } => NarratorInput::Act {
            character: character.unwrap_or_else(|| config.protagonist.clone()),
            action: action.join(" "),
        },
        RpeSubcommand::Scene { description } => NarratorInput::Scene {
            description: description.join(" "),
        },
        RpeSubcommand::Cue { character, action } => NarratorInput::Cue {
            character: character.unwrap_or_else(|| config.persona.clone()),
            action: action.join(" "),
        },
        RpeSubcommand::Beat { description } => NarratorInput::Beat {
            description: description.join(" "),
        },
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse multi-line clap-style player input into a [`Vec<Input>`].
///
/// Each non-empty line is parsed independently as an RPE command (`say`,
/// `act`, `scene`, `cue`, `beat`). Missing `--character` flags are resolved
/// against `config` before any `Input` object is produced — no null characters
/// ever appear in the result.
///
/// Quoted spans in `--character` values are respected:
/// `say --character "Old Man" you need to leave` produces a `Say` with
/// `character = "Old Man"`.
///
/// # Errors
///
/// - [`ParseError::EmptyInput`] — all lines were blank.
/// - [`ParseError::InvalidCommand`] — a line contained an unrecognised
///   command or invalid flags.
/// - [`ParseError::EmptyContent`] — a valid command was given with no
///   content (e.g. `say` with no trailing words).
///
/// # Example
///
/// ```rust
/// use candle_conversation::narrator::{parse_turn, SessionConfig};
///
/// let config = SessionConfig {
///     protagonist: "Kael Dorn".into(),
///     persona: "Voss".into(),
///     max_turns: 7,
/// };
/// let inputs = parse_turn("say we have to get out\nact takes her hand", &config).unwrap();
/// assert_eq!(inputs.len(), 2);
/// ```
pub fn parse_turn(input: &str, config: &SessionConfig) -> Result<Vec<NarratorInput>, ParseError> {
    let lines: Vec<&str> = input
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();

    if lines.is_empty() {
        return Err(ParseError::EmptyInput);
    }

    lines
        .iter()
        .map(|line| {
            let tokens = shell_split(line);
            let input = RpeCommand::try_parse_from(tokens)
                .map(|parsed| command_to_input(parsed.command, config))
                .map_err(|e| ParseError::InvalidCommand {
                    line: line.to_string(),
                    error: e,
                })?;
            if let Some(field) = empty_content_field(&input) {
                return Err(ParseError::EmptyContent {
                    line: line.to_string(),
                    field,
                });
            }
            Ok(input)
        })
        .collect()
}
