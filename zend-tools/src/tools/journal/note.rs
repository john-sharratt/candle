//! journal_note tool: a note of what the turn's thinking concluded, kept in
//! the conversation's journal (`docs/zend_journal.md` §5.2) — or, naming a
//! memory journal, a line written there (`docs/zen_tool_reach.md` §6.2).
//!
//! A note on the conversation's own journal is the call's own `text`, which
//! the reply already holds, so the turn is where it is kept and nothing is
//! written here: the journal reads it from the turn (`zend::journal::notes`),
//! and the answer is one word, `noted`. A note naming a memory journal goes
//! to the daemon's runner, which writes it as the person the conversation is
//! with — or holds it for that person's approval when the turn has read
//! content from outside.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use validator::Validate;

use crate::tools::journal::SELF;
use crate::tools::ops::{forward, OpsError};
use crate::{RegisteredTool, Replay, Tool, ToolContext};

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct NoteRequest {
    /// The note: a short conclusion from the thinking. Required.
    #[validate(length(min = 1, max = 2000))]
    pub text: String,
    /// A memory journal to write it on; left out, your own journal. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    #[validate(length(min = 1, max = 200))]
    pub journal: Option<String>,
}

/// What a note on the conversation's own journal answers.
pub const NOTED: &str = "noted";

pub struct JournalNote;

impl Tool for JournalNote {
    const NAME: &'static str = "journal_note";
    const DESCRIPTION: &'static str =
        "Note a short conclusion from your thinking — what you found, decided or mean to do — so \
         it is kept in your journal's current entry when the thinking itself is gone; or, naming \
         a memory journal, keep a line there.";

    type Request = NoteRequest;
    type Response = Value;
    type Error = OpsError;

    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: NoteRequest) -> Result<Value, OpsError> {
        match req.journal.as_deref() {
            None | Some(SELF) => Ok(json!(NOTED)),
            Some(_) => forward(ctx, Self::NAME, &req).map(|r| json!({"answer": r.answer})),
        }
    }
}

pub const JOURNAL_NOTE: RegisteredTool = RegisteredTool::new::<JournalNote>();

#[cfg(test)]
mod tests {
    use super::*;

    /// **A note on your own journal answers one word**, the same bytes every
    /// time, and writes nothing; one naming a memory journal goes to the
    /// runner, which a context without one refuses.
    #[test]
    fn a_note_answers_noted_and_a_memory_note_goes_to_the_runner() {
        let ctx = ToolContext::new();
        let own = NoteRequest {
            text: "scan.rs holds its lock across the await".into(),
            journal: None,
        };
        assert_eq!(
            serde_json::to_string(&JournalNote::run(&ctx, own).unwrap()).unwrap(),
            "\"noted\""
        );
        let named_self = NoteRequest {
            text: "the same".into(),
            journal: Some(SELF.into()),
        };
        assert_eq!(JournalNote::run(&ctx, named_self).unwrap(), json!(NOTED));
        let memory = NoteRequest {
            text: "Yen prefers the morning flight".into(),
            journal: Some("people/yen".into()),
        };
        assert!(matches!(
            JournalNote::run(&ctx, memory),
            Err(OpsError::NotConfigured)
        ));
    }
}
