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

use candle_conversation::{ScopeTurn, Sequence};

/// The dialect role boundaries the code-read layer splices into a scope's
/// assistant string to reconstruct the full tool-exchange alternation
/// `user(excerpt) → assistant(<tool_call>) → user(<tool_response>) →
/// assistant(close)`. A tool response returns in a *user* turn (Hermes/Qwen
/// convention), and the exchange is closed by an assistant turn so the sequence
/// never ends hanging on a user turn (which would collide with the next turn's
/// user opener).
pub struct ToolExchangeBoundaries {
    /// Closes the assistant `<tool_call>` and opens the user `<tool_response>`:
    /// `assistant_end` + `user_start`.
    pub call_to_response: String,
    /// Closes the user `<tool_response>` and opens the closing assistant
    /// segment: `user_end` + `assistant_start`.
    pub response_to_close: String,
}

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

    /// Prefill every `(user, assistant)` scope of one file **in parallel** — each
    /// on its own scratch slot so the file's scopes (and scopes across
    /// concurrently-ingesting files) batch into large amortising forwards instead
    /// of prefilling one-at-a-time — then record them into the conversation's
    /// timeline in scope order. Returns the per-scope prefilled token counts, in
    /// order, so the ingest bar can advance one unit per scope. `tags` apply to
    /// every scope turn (provenance scoping), exactly as [`Self::insert_prefill_turn`].
    ///
    /// `on_prefilled(tokens)` fires as each scope's prefill lands, so the caller
    /// can advance a live per-scope progress bar + token count instead of only
    /// updating once the whole file's batch completes.
    ///
    /// Default: the serial fallback (one [`Self::insert_prefill_turn`] per scope),
    /// for sinks without a batched engine behind them.
    fn insert_prefill_turns_parallel(
        &mut self,
        scopes: &[(String, String)],
        tags: Vec<String>,
        on_prefilled: candle_conversation::ScopeProgressFn,
    ) -> anyhow::Result<Vec<usize>> {
        scopes
            .iter()
            .map(|(user, assistant)| {
                let tokens = self.insert_prefill_turn(user, assistant, tags.clone())?;
                on_prefilled(tokens);
                Ok(tokens)
            })
            .collect()
    }

    /// The two dialect role boundaries a prefilled code-read tool exchange
    /// splices in, so a scope reconstructs as the full
    /// `user → assistant → user → assistant` alternation rather than one blob.
    /// See [`ToolExchangeBoundaries`].
    ///
    /// Default is the ChatML form (the daemon's dialect), so model-less sinks
    /// still produce a structurally valid turn.
    fn tool_exchange_boundaries(&self) -> ToolExchangeBoundaries {
        ToolExchangeBoundaries {
            call_to_response: "<|im_end|>\n<|im_start|>user\n".to_string(),
            response_to_close: "<|im_end|>\n<|im_start|>assistant\n".to_string(),
        }
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

    fn insert_prefill_turns_parallel(
        &mut self,
        scopes: &[(String, String)],
        tags: Vec<String>,
        on_prefilled: candle_conversation::ScopeProgressFn,
    ) -> anyhow::Result<Vec<usize>> {
        let start = std::time::Instant::now();
        let turns: Vec<ScopeTurn> = scopes
            .iter()
            .map(|(user, assistant)| ScopeTurn {
                user: user.clone(),
                assistant: assistant.clone(),
            })
            .collect();
        tracing::debug!(
            target: "zend::turn_sink",
            n_scopes = turns.len(),
            "insert_prefill_turns_parallel: calling Sequence::ingest_scopes_parallel",
        );
        let result = self
            .inner
            .ingest_scopes_parallel(&turns, tags, on_prefilled)
            .map_err(|e| anyhow::anyhow!("ingest_scopes_parallel: {e}"));
        tracing::debug!(
            target: "zend::turn_sink",
            ms = start.elapsed().as_millis() as u64,
            ok = result.is_ok(),
            "insert_prefill_turns_parallel: returned",
        );
        result
    }

    fn tool_exchange_boundaries(&self) -> ToolExchangeBoundaries {
        // Pull the real markers from the live dialect rather than assuming
        // ChatML, so the boundaries match the model's chat template exactly.
        let g = self.inner.glue_markers();
        ToolExchangeBoundaries {
            call_to_response: format!("{}{}", g.assistant_end, g.user_start),
            response_to_close: format!("{}{}", g.user_end, g.assistant_start),
        }
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
