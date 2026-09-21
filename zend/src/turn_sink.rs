//! Indirection between the workspace-ingestion paths and the
//! underlying [`candle_conversation::Sequence`].
//!
//! One operation is abstracted:
//!
//! * **`insert_prefill_turn(user, assistant)`** — prefill a complete
//!   user/assistant exchange with no decode. The prefilled halves of a tool
//!   round-trip (the user-side request or `<tool_response>`, the assistant-side
//!   `<tool_call>` echo) flow through this.
//! * **`ingest_chain`** — a tool round-trip whose LAST assistant turn is
//!   DECODED: the `repo_map` layer's per-folder summary. `code_reading` no
//!   longer prefills anything — each file is a real hidden conversation (see
//!   `crate::code_read::run_file_conversation`), driven directly through
//!   `Sequence::submit_turn_with_options` rather than through this sink.
//!
//! Integration tests wire a [`RecordingTurnSink`] that captures every call
//! into memory, so the conversation shape can be verified without loading a
//! model.

use candle_conversation::stencil::TriggerRegistry;
use candle_conversation::{Sequence, TurnText};
use std::sync::Arc;

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

    /// Ingest an N-turn tool round-trip chain whose LAST assistant turn is
    /// DECODED — the `repo_map` folder shape. `prefilled` holds the verbatim
    /// `(user, assistant)` pairs (a request or `<tool_response>` paired with the
    /// `<tool_call>` it provokes); `decode_user` is the final tool response, whose
    /// assistant half the model writes. `force_tools` names every tool the
    /// prefilled calls refer to, so the projection carries their definitions.
    /// Returns the tokens ingested.
    ///
    /// Default (model-less sinks, e.g. tests): record every turn with an empty
    /// final assistant half — no engine to decode it.
    fn ingest_chain(
        &mut self,
        prefilled: &[(TurnText, String)],
        decode_user: &TurnText,
        tags: Vec<String>,
        _max_summary_tokens: usize,
        _force_tools: &[String],
    ) -> anyhow::Result<usize> {
        let mut total = 0usize;
        for (user, assistant) in prefilled {
            total += self.insert_prefill_turn(user, assistant, tags.clone())?;
        }
        total += self.insert_prefill_turn(decode_user, "", tags)?;
        Ok(total)
    }
}

/// Sink that drives a live [`Sequence`] — the daemon's production
/// path.  Holds a mutable borrow for the lifetime of the ingestion
/// pass.
pub struct SequenceTurnSink<'a> {
    inner: &'a mut Sequence,
    /// Decode steering for the summary turn: the `<think>` trigger bound to
    /// [`ThinkMode::Off`]'s tree, so the block closes the token after it opens.
    ///
    /// Carried on the sink rather than added to the [`InsertTurnSink`] methods
    /// because the model-less sinks below never decode — they have nothing to
    /// steer, and a parameter they all had to ignore would say otherwise.
    triggers: Arc<TriggerRegistry>,
}

impl<'a> SequenceTurnSink<'a> {
    pub fn new(inner: &'a mut Sequence, triggers: Arc<TriggerRegistry>) -> Self {
        Self { inner, triggers }
    }
}

impl<'a> InsertTurnSink for SequenceTurnSink<'a> {
    fn ingest_chain(
        &mut self,
        prefilled: &[(TurnText, String)],
        decode_user: &TurnText,
        tags: Vec<String>,
        max_summary_tokens: usize,
        force_tools: &[String],
    ) -> anyhow::Result<usize> {
        self.inner
            .ingest_roundtrip_chain(
                prefilled,
                decode_user.clone(),
                tags,
                max_summary_tokens,
                force_tools,
                Arc::clone(&self.triggers),
            )
            .map_err(|e| anyhow::anyhow!("ingest_roundtrip_chain: {e}"))
    }

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
