//! Indirection between the workspace-ingestion paths and the
//! underlying [`candle_conversation::Sequence`].
//!
//! One operation is abstracted: **`insert_prefill_turn(user, assistant)`** —
//! prefill a complete user/assistant exchange with no decode. The prefilled
//! halves of a tool round-trip (the user-side request or `<tool_response>`, the
//! assistant-side `<tool_call>` echo) flow through this.
//!
//! **Nothing here decodes.** Both layers that decode — `code_reading` per file
//! (`crate::code_read::run_file_conversation`) and `repo_map` per folder
//! (`crate::repo_scan::converse::run_folder_conversation`) — are real agentic
//! tool loops, which need to read each round's decoded text and couple the turn
//! it sealed, so they drive the `Sequence` directly. A sink method that decoded
//! one round could not carry a loop, and a sink that grew `couple_turn` beside it
//! would only be mirroring `Sequence` behind a trait.
//!
//! Integration tests wire a [`RecordingTurnSink`] that captures every call
//! into memory, so the conversation shape can be verified without loading a
//! model.

use candle_conversation::{Sequence, TurnText};

/// Accepts a structured `(user, assistant)` turn stream from the
/// workspace-ingestion paths.
pub trait InsertTurnSink {
    /// Prefill a complete user/assistant exchange with no model decode. Returns
    /// the number of tokens prefilled — the ingest path sums it into the upload's
    /// "tokens ingested" stat. `tags` (e.g. `["code", <path>]`) are persisted on
    /// the TurnDecl so tag-scoped provenance galleries admit the turn, alongside
    /// the staged projection events the production sink records for it.
    fn insert_prefill_turn(
        &mut self,
        user: &TurnText,
        assistant: &str,
        tags: Vec<String>,
    ) -> anyhow::Result<usize>;
}

/// Sink that drives a live [`Sequence`] — the daemon's production
/// path.  Holds a mutable borrow for the lifetime of the ingestion
/// pass.
pub struct SequenceTurnSink<'a> {
    inner: &'a mut Sequence,
}

impl<'a> SequenceTurnSink<'a> {
    pub fn new(inner: &'a mut Sequence) -> Self {
        Self { inner }
    }
}

impl<'a> InsertTurnSink for SequenceTurnSink<'a> {
    fn insert_prefill_turn(
        &mut self,
        user: &TurnText,
        assistant: &str,
        tags: Vec<String>,
    ) -> anyhow::Result<usize> {
        let start = std::time::Instant::now();
        tracing::debug!(
            target: "zend::turn_sink",
            user_bytes = user.text().len(),
            assistant_bytes = assistant.len(),
            "insert_prefill_turn: calling Sequence::insert_turn_staged",
        );
        let result = self
            .inner
            .insert_turn_staged(user.clone(), assistant, tags)
            .map_err(|e| anyhow::anyhow!("insert_turn_staged: {e}"));
        tracing::debug!(
            target: "zend::turn_sink",
            ms = start.elapsed().as_millis() as u64,
            ok = result.is_ok(),
            "insert_prefill_turn: returned",
        );
        result
    }
}

/// Recording sink for integration tests.  Stores every
/// `(user, assistant, tags)` prefill entry the ingestion path emits, in order,
/// so test cases can verify the conversation shape (and its provenance tags)
/// without loading a model.
#[allow(dead_code)]
#[derive(Default)]
pub struct RecordingTurnSink {
    pub turns: Vec<(String, String, Vec<String>)>,
}

#[allow(dead_code)]
impl RecordingTurnSink {
    pub fn new() -> Self {
        Self { turns: Vec::new() }
    }
}

impl InsertTurnSink for RecordingTurnSink {
    fn insert_prefill_turn(
        &mut self,
        user: &TurnText,
        assistant: &str,
        tags: Vec<String>,
    ) -> anyhow::Result<usize> {
        let user = user.text();
        // No tokenizer in the recording sink — approximate the prefilled token
        // count by whitespace words so callers that surface a stat see a
        // plausible non-zero value in model-less tests.
        let words = user.split_whitespace().count() + assistant.split_whitespace().count();
        self.turns.push((user, assistant.to_string(), tags));
        Ok(words)
    }
}
