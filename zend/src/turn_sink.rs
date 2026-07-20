//! Indirection between the workspace-ingestion paths and the
//! underlying [`candle_conversation::Sequence`].
//!
//! One operation is abstracted: **`insert_prefill_turn(user, assistant)`**
//! — prefill a complete user/assistant exchange with no decode.  The
//! repo_map layer's cluster listings and the per-part prefilled halves of
//! the code_reading layer's per-file tool-call conversation (the user-side
//! "Read X lines A-B." prompt + the assistant-side `<tool_call>` echo, and
//! the user-side `<tool_response>` carrying the file content) all flow
//! through this method. Per-file summaries are NOT decoded here — they are
//! the async summary tree's rollup, built by the summariser over the
//! recorded scope turns.
//!
//! Integration tests wire a [`RecordingTurnSink`] that captures every call
//! into memory, so the conversation shape can be verified without loading a
//! model.

use candle_conversation::Sequence;

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
        user: &str,
        assistant: &str,
        tags: Vec<String>,
    ) -> anyhow::Result<usize>;

    /// Like [`Self::insert_prefill_turn`], but `assistant` carries `seam_marker`s
    /// at structural boundaries (e.g. subdirectory headers). Each becomes a
    /// self-referencing projection event so the listing's regions are
    /// independently retrievable (see
    /// `Conversation::insert_turn_staged_windowed`). Default: strip the markers and
    /// fall back to a plain prefill — a sink without projection-event storage keeps
    /// the turn, just not the sub-window seams.
    fn insert_prefill_turn_windowed(
        &mut self,
        user: &str,
        assistant_with_seams: &str,
        seam_marker: &str,
        tags: Vec<String>,
    ) -> anyhow::Result<()> {
        self.insert_prefill_turn(user, &assistant_with_seams.replace(seam_marker, ""), tags)?;
        Ok(())
    }

    /// Ingest one code scope as a TOOL ROUND-TRIP of two coupled turns — the
    /// call (`user(request)` → `assistant(<tool_call>)`) and the response
    /// (`user(<tool_response>)` → `assistant(summary)`). Recording it as two
    /// coupled turns (not one baked exchange) keeps the inter-turn seams as
    /// regenerated glue and the `/no_think` / `<think>` handling correct — see
    /// [`candle_conversation::Sequence::ingest_scope_roundtrip`]. Returns the
    /// tokens ingested.
    ///
    /// Default (model-less sinks, e.g. tests): record the two turns with an empty
    /// response summary — no engine to decode it. Real engines override to decode
    /// the summary under `/no_think` and couple the pair.
    fn ingest_scope_roundtrip(
        &mut self,
        call_user: &str,
        call_assistant: &str,
        response_user: &str,
        tags: Vec<String>,
        _max_summary_tokens: usize,
    ) -> anyhow::Result<usize> {
        let a = self.insert_prefill_turn(call_user, call_assistant, tags.clone())?;
        let b = self.insert_prefill_turn(response_user, "", tags)?;
        Ok(a + b)
    }

    /// Restart-resume cache probe: whether some conversation in the
    /// substrate already carries `key == value` in its `custom` metadata
    /// (i.e. this unit was ingested in a prior run and reloaded from the
    /// redo log). Default `false` — non-substrate sinks never cache-hit.
    fn unit_cached(&self, _key: &str, _value: &str) -> bool {
        false
    }

    /// Tag the underlying conversation with `tags` (content hash + rich
    /// descriptive fields) so a later run's [`Self::unit_cached`] finds it.
    /// Default no-op for sinks without a backing conversation.
    fn tag_unit(&self, _tags: &std::collections::BTreeMap<String, String>) {}
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
    fn insert_prefill_turn_windowed(
        &mut self,
        user: &str,
        assistant_with_seams: &str,
        seam_marker: &str,
        tags: Vec<String>,
    ) -> anyhow::Result<()> {
        self.inner
            .insert_turn_staged_windowed(user, assistant_with_seams, seam_marker, tags)
            .map_err(|e| anyhow::anyhow!("insert_turn_staged_windowed: {e}"))
    }

    fn insert_prefill_turn(
        &mut self,
        user: &str,
        assistant: &str,
        tags: Vec<String>,
    ) -> anyhow::Result<usize> {
        let start = std::time::Instant::now();
        tracing::debug!(
            target: "zend::turn_sink",
            user_bytes = user.len(),
            assistant_bytes = assistant.len(),
            "insert_prefill_turn: calling Sequence::insert_turn_staged",
        );
        let result = self
            .inner
            .insert_turn_staged(user, assistant, tags)
            .map_err(|e| anyhow::anyhow!("insert_turn_staged: {e}"));
        tracing::debug!(
            target: "zend::turn_sink",
            ms = start.elapsed().as_millis() as u64,
            ok = result.is_ok(),
            "insert_prefill_turn: returned",
        );
        result
    }

    fn ingest_scope_roundtrip(
        &mut self,
        call_user: &str,
        call_assistant: &str,
        response_user: &str,
        tags: Vec<String>,
        max_summary_tokens: usize,
    ) -> anyhow::Result<usize> {
        self.inner
            .ingest_scope_roundtrip(
                call_user,
                call_assistant,
                response_user,
                tags,
                max_summary_tokens,
            )
            .map_err(|e| anyhow::anyhow!("ingest_scope_roundtrip: {e}"))
    }

    fn unit_cached(&self, key: &str, value: &str) -> bool {
        !self
            .inner
            .find_conversations_by_metadata(key, value)
            .is_empty()
    }

    fn tag_unit(&self, tags: &std::collections::BTreeMap<String, String>) {
        if let Err(e) = self.inner.set_metadata_many(tags) {
            tracing::warn!(
                target: "zend::turn_sink",
                "failed to tag conversation metadata (resume cache): {e:#}",
            );
        }
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
        user: &str,
        assistant: &str,
        tags: Vec<String>,
    ) -> anyhow::Result<usize> {
        self.turns
            .push((user.to_string(), assistant.to_string(), tags));
        // No tokenizer in the recording sink — approximate the prefilled token
        // count by whitespace words so callers that surface a stat see a
        // plausible non-zero value in model-less tests.
        Ok(user.split_whitespace().count() + assistant.split_whitespace().count())
    }
}
