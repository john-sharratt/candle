//! Indirection between the workspace-ingestion paths and the
//! underlying [`candle_conversation::Sequence`].
//!
//! Two operations are abstracted:
//!
//! * **`insert_prefill_turn(user, assistant)`** — prefill a complete
//!   user/assistant exchange with no decode. The prefilled halves of a tool
//!   round-trip (the user-side request or `<tool_response>`, the assistant-side
//!   `<tool_call>` echo) flow through this.
//! * **`ingest_chain` / `ingest_scope_roundtrip`** — a tool round-trip whose
//!   LAST assistant turn is DECODED: the `code_reading` layer's per-scope
//!   summary (two turns) and the `repo_map` layer's per-folder summary (three).
//!   Per-file summaries are NOT decoded here — they are the async summary
//!   tree's rollup, built by the summariser over the recorded scope turns.
//!
//! Integration tests wire a [`RecordingTurnSink`] that captures every call
//! into memory, so the conversation shape can be verified without loading a
//! model.

use candle_conversation::Sequence;
use std::sync::Arc;

/// Per-scope progress callback, invoked with a scope's ingested token count as it
/// lands so the upload/code-read path can climb its progress bar per scope rather
/// than only when the whole file completes. `Arc<dyn Fn>` so it's `'static` +
/// `Send` and cheap to clone across a file's scopes.
pub type ScopeProgressFn = Arc<dyn Fn(usize) + Send + Sync>;

/// Concurrent scopes per chunk in the parallel per-file ingest
/// ([`SequenceTurnSink::ingest_scopes`]). Chunks run sequentially, so this bounds
/// both the concurrent fork/slot count (with [`crate::code_read::CODE_READ_PARALLELISM`]
/// files in flight → `files × SCOPE_PARALLELISM` forks) and the window a fork's
/// K/V must stay hot for the splice.
const SCOPE_PARALLELISM: usize = 4;

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
        prefilled: &[(String, String)],
        decode_user: &str,
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

    /// Ingest one file's scopes, each as a two-turn tool round-trip
    /// ([`Self::ingest_scope_roundtrip`]). `prepared` holds the rendered
    /// `(call_user, call_assistant, response_user)` per scope IN FILE ORDER;
    /// `on_prefilled` fires per scope with its token count.
    ///
    /// Default: **serial** — the correct fallback for model-less sinks (tests),
    /// where the round-trip has no engine to parallelise across. The production
    /// [`SequenceTurnSink`] overrides this to fork each scope onto its own
    /// timeline, run the round-trips CONCURRENTLY (co-batched on the wave engine),
    /// and splice the sealed pairs back onto the file timeline in order.
    fn ingest_scopes(
        &mut self,
        prepared: Vec<(String, String, String)>,
        tags: Vec<String>,
        max_summary_tokens: usize,
        on_prefilled: &crate::turn_sink::ScopeProgressFn,
    ) -> anyhow::Result<()> {
        for (call_user, call_assistant, response_user) in prepared {
            let tokens = self.ingest_scope_roundtrip(
                &call_user,
                &call_assistant,
                &response_user,
                tags.clone(),
                max_summary_tokens,
            )?;
            on_prefilled(tokens);
        }
        Ok(())
    }
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
    fn ingest_chain(
        &mut self,
        prefilled: &[(String, String)],
        decode_user: &str,
        tags: Vec<String>,
        max_summary_tokens: usize,
        force_tools: &[String],
    ) -> anyhow::Result<usize> {
        self.inner
            .ingest_roundtrip_chain(
                prefilled,
                decode_user,
                tags,
                max_summary_tokens,
                force_tools,
            )
            .map_err(|e| anyhow::anyhow!("ingest_roundtrip_chain: {e}"))
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

    /// Parallel per-file scope ingest: fork each scope onto its own timeline, run
    /// the proven two-turn round-trips CONCURRENTLY in bounded chunks (co-batched
    /// on the wave engine), then splice each fork's coupled pair onto the file
    /// timeline IN ORDER (`splice_scope_turns`, which couples + tombstones the
    /// fork). Chunks run sequentially so the concurrent fork count — and the
    /// window in which a fork's K/V must stay HOT for the splice — stays bounded.
    fn ingest_scopes(
        &mut self,
        prepared: Vec<(String, String, String)>,
        tags: Vec<String>,
        max_summary_tokens: usize,
        on_prefilled: &crate::turn_sink::ScopeProgressFn,
    ) -> anyhow::Result<()> {
        for chunk in prepared.chunks(SCOPE_PARALLELISM) {
            // Fork one throwaway timeline per scope in this chunk.
            let mut forks: Vec<Sequence> = Vec::with_capacity(chunk.len());
            for _ in chunk {
                forks.push(
                    self.inner
                        .fork_scope()
                        .map_err(|e| anyhow::anyhow!("fork_scope: {e}"))?,
                );
            }
            // Run each fork's two-turn round-trip concurrently; the scheduler
            // co-batches their prefills + summary decodes on the shared wave.
            let results: Vec<candle_conversation::Result<(u32, u32, usize)>> =
                std::thread::scope(|s| {
                    let handles: Vec<_> = forks
                        .iter_mut()
                        .zip(chunk.iter())
                        .map(|(fork, (call_user, call_assistant, response_user))| {
                            let tags = tags.clone();
                            s.spawn(move || {
                                fork.ingest_scope_roundtrip_indices(
                                    call_user,
                                    call_assistant,
                                    response_user,
                                    tags,
                                    max_summary_tokens,
                                )
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| h.join().expect("scope ingest thread panicked"))
                        .collect()
                });
            // Splice the sealed pairs onto the file timeline in scope order. On the
            // first failure (a scope round-trip that errored, or a splice that
            // failed) STOP and tombstone every fork from that point on. The forks
            // before it were adopted + tombstoned by `splice_scope_turns`, but the
            // failing fork and every later one ran their round-trip WITHOUT being
            // spliced — and `Sequence`'s `Drop` frees only the slot, leaving their
            // registered timeline + sealed turns behind as orphaned, path-less
            // "(untitled)" scope conversations. Tombstoning them means a failed
            // chunk leaves nothing behind before the file is retried whole.
            let mut spliced = 0usize;
            let mut chunk_err: Option<anyhow::Error> = None;
            for (fork, res) in forks.iter().zip(results) {
                match res {
                    Ok((call_idx, resp_idx, tokens)) => {
                        match self.inner.splice_scope_turns(
                            fork.timeline_id(),
                            call_idx,
                            resp_idx,
                            tags.clone(),
                        ) {
                            Ok(_) => {
                                spliced += 1;
                                on_prefilled(tokens);
                            }
                            Err(e) => {
                                chunk_err = Some(anyhow::anyhow!("splice_scope_turns: {e}"));
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        chunk_err = Some(anyhow::anyhow!("scope round-trip: {e}"));
                        break;
                    }
                }
            }
            if let Some(e) = chunk_err {
                // Every fork at or after `spliced` was never adopted onto the file
                // timeline — tombstone so no orphan scope timeline survives.
                for fork in forks.iter().skip(spliced) {
                    self.inner.tombstone_fork(fork.timeline_id());
                }
                drop(forks);
                return Err(e);
            }
            // All spliced → their timelines are already tombstoned by
            // `splice_scope_turns`; dropping frees the scheduler slots.
            drop(forks);
        }
        Ok(())
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
