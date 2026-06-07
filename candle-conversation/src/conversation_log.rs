//! YAML redo-log format shared by `tree_gen` (writer) and `chat --resume` (reader).
//!
//! The output file is a multi-document YAML stream. Each `---`-delimited document
//! is one [`LogRecord`], appended and flushed after every completed exchange.
//! The file is always valid and parseable up to the last flush — killing the
//! process mid-run leaves a complete, readable log of everything that happened.
//!
//! Replay order: `header` → `turn` × N → `done`
//!
//! ```text
//! ---
//! kind: header
//! character_system_prompt: |
//!   You are Mira, a thoughtful archivist...
//! guide_system_prompt: |
//!   You are guiding a character through a journey of quiet revelation.
//! started_at: epoch+20507d 14:22:01Z
//! ---
//! kind: turn
//! seq: 1
//! guide_message: A letter arrives on your desk. It is from her.
//! character_response: I hold the envelope for a long moment before opening it.
//! character_token_count: 187
//! ---
//! kind: done
//! total_turns: 30
//! elapsed_secs: 412.7
//! ```

use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use crate::Result;

// ── Log record ────────────────────────────────────────────────────────────────

/// One document in a tree-gen YAML redo log.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LogRecord {
    /// Written once at the start of a run, before any turns.
    Header {
        character_system_prompt: String,
        guide_system_prompt: String,
        started_at: String,
    },
    /// Written after each completed character exchange.
    Turn {
        /// 1-based index within this run.
        seq: usize,
        /// The event the guide delivered to the character.
        guide_message: String,
        /// The character's response (thinking blocks stripped).
        character_response: String,
        character_token_count: usize,
    },
    /// Written once when the run completes normally.
    Done {
        total_turns: usize,
        elapsed_secs: f64,
    },
}

// ── Log writer ────────────────────────────────────────────────────────────────

/// Appends [`LogRecord`] documents to a YAML redo-log file, flushing after each.
pub struct LogWriter {
    file: File,
}

impl LogWriter {
    /// Create (or truncate) a redo-log file at `path`.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::create(path)?;
        Ok(Self { file })
    }

    /// Serialize `record` as a YAML document and flush immediately.
    ///
    /// Multi-line string fields are written with YAML literal block style (`|`)
    /// so the log file is human-readable.
    pub fn append(&mut self, record: &LogRecord) -> Result<()> {
        writeln!(self.file, "---")?;
        match record {
            LogRecord::Header {
                character_system_prompt,
                guide_system_prompt,
                started_at,
            } => {
                writeln!(self.file, "kind: header")?;
                write_block_scalar(&mut self.file, "character_system_prompt", character_system_prompt)?;
                write_block_scalar(&mut self.file, "guide_system_prompt", guide_system_prompt)?;
                writeln!(self.file, "started_at: {started_at}")?;
            }
            LogRecord::Turn {
                seq,
                guide_message,
                character_response,
                character_token_count,
            } => {
                writeln!(self.file, "kind: turn")?;
                writeln!(self.file, "seq: {seq}")?;
                write_block_scalar(&mut self.file, "guide_message", guide_message)?;
                write_block_scalar(&mut self.file, "character_response", character_response)?;
                writeln!(self.file, "character_token_count: {character_token_count}")?;
            }
            LogRecord::Done {
                total_turns,
                elapsed_secs,
            } => {
                writeln!(self.file, "kind: done")?;
                writeln!(self.file, "total_turns: {total_turns}")?;
                writeln!(self.file, "elapsed_secs: {elapsed_secs:.1}")?;
            }
        }
        self.file.flush()?;
        Ok(())
    }
}

/// Write a string field as a YAML literal block scalar (`|`).
/// Single-line values are written inline; multi-line values use `|` with
/// 2-space indentation.
fn write_block_scalar(w: &mut impl Write, key: &str, value: &str) -> std::io::Result<()> {
    // Normalise \r\n → \n
    let value = value.replace("\r\n", "\n");
    let value = value.trim_end_matches('\n');
    if value.contains('\n') {
        writeln!(w, "{key}: |")?;
        for line in value.split('\n') {
            if line.is_empty() {
                writeln!(w)?;
            } else {
                writeln!(w, "  {line}")?;
            }
        }
    } else if value.contains(':') || value.contains('#') || value.contains('\'') || value.contains('"') {
        // Single-line: inline, but quote if it contains special YAML characters.
        writeln!(w, "{key}: {value:?}")?;
    } else {
        writeln!(w, "{key}: {value}")?;
    }
    Ok(())
}

// ── Resume helpers ────────────────────────────────────────────────────────────

/// A single turn extracted from a resume log.
pub struct ResumeTurn {
    /// The message the guide delivered to the character.
    pub user_message: String,
    /// The character's reply.
    pub character_response: String,
}

/// Parsed contents of a tree-gen redo log, ready for replay.
pub struct ResumeLog {
    /// System prompt for the character conversation.
    pub character_system_prompt: String,
    /// All turns in chronological order.
    pub turns: Vec<ResumeTurn>,
}

/// Parse a multi-document YAML redo log written by `tree_gen`.
///
/// Returns the character system prompt from the `header` record and all turn
/// exchanges in chronological order. Unknown fields are silently ignored.
pub fn load_resume_log(path: impl AsRef<Path>) -> Result<ResumeLog> {
    use serde::Deserialize as _;

    let mut text = String::new();
    File::open(path)?.read_to_string(&mut text)?;

    let mut system_prompt = String::new();
    let mut turns = Vec::new();

    for document in serde_yaml::Deserializer::from_str(&text) {
        match LogRecord::deserialize(document)? {
            LogRecord::Header { character_system_prompt, .. } => {
                system_prompt = character_system_prompt;
            }
            LogRecord::Turn { guide_message, character_response, .. } => {
                turns.push(ResumeTurn {
                    user_message: guide_message,
                    character_response,
                });
            }
            LogRecord::Done { .. } => {}
        }
    }

    Ok(ResumeLog { character_system_prompt: system_prompt, turns })
}

// ── Utilities ─────────────────────────────────────────────────────────────────

/// Truncate `text` to `max_chars` characters (Unicode scalar values), appending
/// `…` if it was cut.
pub fn truncate_for_display(text: &str, max_chars: usize) -> String {
    let text = text.trim();
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let cut: String = text.chars().take(max_chars).collect();
        format!("{cut}…")
    }
}

/// Simple ISO-8601-ish UTC timestamp without pulling in `chrono`.
pub fn now_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let days = secs / 86400;
    format!("epoch+{}d {:02}:{:02}:{:02}Z", days, h, m, s)
}
