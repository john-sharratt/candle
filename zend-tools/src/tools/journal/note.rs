//! journal_note tool: a note of what the turn's thinking concluded, kept in
//! the conversation's journal (`docs/zend_journal.md` §5.2).
//!
//! The note's words are the call's own `text`, which the reply already holds,
//! so the turn is where the note is kept and nothing is written here. The
//! journal reads it from the turn (`zend::journal::notes`), and the answer is
//! one word: it is in the projection after every note, and says only that the
//! note is kept.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize, Serializer};
use validator::Validate;

use super::JournalError;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct NoteRequest {
    /// The note: a short conclusion from the thinking. Required.
    #[validate(length(min = 1))]
    pub text: String,
}

/// What a kept note answers: `noted`.
pub struct Noted;

impl Serialize for Noted {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str("noted")
    }
}

pub struct JournalNote;

impl Tool for JournalNote {
    const NAME: &'static str = "journal_note";
    const DESCRIPTION: &'static str =
        "Note a short conclusion from your thinking — what you found, decided or mean to do — so \
         it is kept in your journal's current entry when the thinking itself is gone.";

    type Request = NoteRequest;
    type Response = Noted;
    type Error = JournalError;

    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(_ctx: &ToolContext, _req: NoteRequest) -> Result<Noted, JournalError> {
        Ok(Noted)
    }
}

pub const JOURNAL_NOTE: RegisteredTool = RegisteredTool::new::<JournalNote>();

#[cfg(test)]
mod tests {
    use super::*;

    /// **A note answers one word**, the same bytes every time.
    #[test]
    fn a_note_answers_noted() {
        assert_eq!(serde_json::to_string(&Noted).unwrap(), "\"noted\"");
        let ctx = ToolContext::new();
        let req = NoteRequest {
            text: "scan.rs holds its lock across the await".into(),
        };
        assert!(JournalNote::run(&ctx, req).is_ok());
    }
}
