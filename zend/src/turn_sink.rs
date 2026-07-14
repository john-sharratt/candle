//! Indirection between the workspace-ingestion paths and the
//! underlying [`candle_conversation::Sequence`].
//!
//! Two operations are abstracted:
//!
//! 1. **`insert_prefill_turn(user, assistant)`** — prefill a complete
//!    user/assistant exchange with no decode.  The repo_map layer's
//!    cluster listings and the per-part prefilled halves of the
//!    code_reading layer's per-file tool-call conversation (the
//!    user-side "Read X lines A-B." prompt + the assistant-side
//!    `<tool_call>` echo, and the user-side `<tool_response>`
//!    carrying the file content) all flow through this method.
//!
//! 2. **`decode_summary_turn(user)`** — submit a user turn and run
//!    the model to decode the assistant response.  The code_reading
//!    layer uses this for the final whole-file summary (≤200 words)
//!    that closes each per-file conversation, so the resulting K/V —
//!    and the summary text itself — lands in the trunk under genuine
//!    model reasoning, not a prefilled placeholder.
//!
//! Integration tests wire a [`RecordingTurnSink`] that captures
//! every call into memory and returns a deterministic placeholder
//! for `decode_summary_turn`, so the conversation shape can be
//! verified without loading a model.

use candle_conversation::{SamplingConfig, Sequence, TurnEvent, TurnOptions};

/// Accepts a structured `(user, assistant)` turn stream from the
/// workspace-ingestion paths.
pub trait InsertTurnSink {
    /// Prefill a complete user/assistant exchange with no model decode.
    /// `tags` are the turn's gather-scope tags (e.g. `["code", <path>]`) —
    /// persisted on the TurnDecl so tag-scoped provenance galleries admit
    /// the turn, alongside the staged projection events the production
    /// sink records for it.
    fn insert_prefill_turn(
        &mut self,
        user: &str,
        assistant: &str,
        tags: Vec<String>,
    ) -> anyhow::Result<()>;

    /// Submit `user` and decode the assistant response with the
    /// supplied `max_tokens` cap and gather-scope `tags`.  Returns the
    /// decoded text so the caller can inspect it; the summary's K/V lands
    /// on the sequence automatically as part of the staged finish.
    fn decode_summary_turn(
        &mut self,
        user: &str,
        max_tokens: usize,
        tags: Vec<String>,
    ) -> anyhow::Result<String>;

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
    fn insert_prefill_turn(
        &mut self,
        user: &str,
        assistant: &str,
        tags: Vec<String>,
    ) -> anyhow::Result<()> {
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

    fn decode_summary_turn(
        &mut self,
        user: &str,
        max_tokens: usize,
        tags: Vec<String>,
    ) -> anyhow::Result<String> {
        let options = TurnOptions {
            max_tokens: Some(max_tokens),
            sampling: Some(SamplingConfig::argmax()),
            tags,
            ..Default::default()
        };

        let submit_start = std::time::Instant::now();
        tracing::debug!(
            target: "zend::turn_sink",
            user_bytes = user.len(),
            max_tokens = max_tokens,
            "decode_summary_turn: calling submit_turn_with_options",
        );
        let handle = self
            .inner
            .submit_turn_with_options(user, options)
            .map_err(|e| anyhow::anyhow!("submit_turn_with_options: {e}"))?;
        tracing::debug!(
            target: "zend::turn_sink",
            submit_ms = submit_start.elapsed().as_millis() as u64,
            "decode_summary_turn: submit returned handle, waiting on stream",
        );

        let stream_start = std::time::Instant::now();
        let mut done = None;
        let mut first_event_logged = false;
        let mut token_count: usize = 0;
        let mut last_event_at = std::time::Instant::now();
        for event in handle.stream() {
            if !first_event_logged {
                tracing::debug!(
                    target: "zend::turn_sink",
                    wait_ms = stream_start.elapsed().as_millis() as u64,
                    "decode_summary_turn: first stream event received",
                );
                first_event_logged = true;
            }
            let gap_ms = last_event_at.elapsed().as_millis() as u64;
            last_event_at = std::time::Instant::now();
            match event {
                TurnEvent::Prefill(_) => {
                    tracing::debug!(
                        target: "zend::turn_sink",
                        gap_ms = gap_ms,
                        "decode_summary_turn: Prefill event",
                    );
                }
                TurnEvent::PrefillProgress {
                    tokens_done,
                    tokens_total,
                } => {
                    tracing::debug!(
                        target: "zend::turn_sink",
                        gap_ms = gap_ms,
                        tokens_done = tokens_done,
                        tokens_total = tokens_total,
                        "decode_summary_turn: PrefillProgress",
                    );
                }
                TurnEvent::Token(_) => {
                    token_count += 1;
                    if token_count == 1 {
                        tracing::debug!(
                            target: "zend::turn_sink",
                            first_token_ms = stream_start.elapsed().as_millis() as u64,
                            "decode_summary_turn: first decoded token",
                        );
                    }
                }
                TurnEvent::Done(resp) => {
                    tracing::debug!(
                        target: "zend::turn_sink",
                        total_ms = stream_start.elapsed().as_millis() as u64,
                        token_count = token_count,
                        text_bytes = resp.text.len(),
                        "decode_summary_turn: Done",
                    );
                    done = Some(resp);
                    break;
                }
                TurnEvent::Error(e) => {
                    return Err(anyhow::anyhow!("decode_summary_turn scheduler error: {e}"));
                }
                _ => {}
            }
        }
        let resp = done
            .ok_or_else(|| anyhow::anyhow!("decode_summary_turn: scheduler closed without Done"))?;
        let text = resp.text.clone();

        let finish_start = std::time::Instant::now();
        tracing::debug!(
            target: "zend::turn_sink",
            "decode_summary_turn: calling finish_turn_staged",
        );
        self.inner
            .finish_turn_staged(handle, &resp)
            .map_err(|e| anyhow::anyhow!("finish_turn_staged: {e}"))?;
        tracing::debug!(
            target: "zend::turn_sink",
            ms = finish_start.elapsed().as_millis() as u64,
            "decode_summary_turn: finish_turn_staged returned",
        );
        Ok(text)
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
/// `(user, assistant, decoded, tags)` entry the ingestion path emits,
/// in order, so test cases can verify the conversation shape
/// without loading a model.  `decoded == true` flags the entries
/// produced by [`InsertTurnSink::decode_summary_turn`]; the
/// assistant slot for those entries carries [`Self::summary_stub`]
/// in place of a real model output.
#[allow(dead_code)]
#[derive(Default)]
pub struct RecordingTurnSink {
    pub turns: Vec<(String, String, bool, Vec<String>)>,
    pub summary_stub: String,
}

#[allow(dead_code)]
impl RecordingTurnSink {
    pub fn new() -> Self {
        Self {
            turns: Vec::new(),
            summary_stub: String::from("[fake summary]"),
        }
    }
}

impl InsertTurnSink for RecordingTurnSink {
    fn insert_prefill_turn(
        &mut self,
        user: &str,
        assistant: &str,
        tags: Vec<String>,
    ) -> anyhow::Result<()> {
        self.turns
            .push((user.to_string(), assistant.to_string(), false, tags));
        Ok(())
    }

    fn decode_summary_turn(
        &mut self,
        user: &str,
        _max_tokens: usize,
        tags: Vec<String>,
    ) -> anyhow::Result<String> {
        self.turns
            .push((user.to_string(), self.summary_stub.clone(), true, tags));
        Ok(self.summary_stub.clone())
    }
}
