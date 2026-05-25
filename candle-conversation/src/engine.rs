//! The conversation engine: entry point, spawns the scheduler thread.

use crate::config::{EngineConfig, SequenceConfig};
use crate::conversation::Sequence;
use crate::error::ConversationError;
use crate::projection::{Builder, Conversation, ProjectionTarget};
use crate::provenance::ProvenanceFile;
use crate::scheduler::{Scheduler, SchedulerRequest};
use crate::substrate_cache::SubstrateCache;
use crate::token_buffer::TokenBuffer;

use candle_nn::CHUNK_SIZE;
use candle_transformers::models::batched_inference::{ManagedBatchedModel, ModelCoreProperties};
use crossbeam::channel;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// The entry point for the conversation engine.
///
/// Owns the scheduler thread and provides factory methods for creating
/// conversations. All GPU resources live on the scheduler thread; the
/// engine itself is a lightweight handle.
///
/// # Example
///
/// ```ignore
/// let engine = ConversationEngine::new(model, tokenizer, config)?;
/// let mut conv = engine.new_conversation("You are helpful.", Default::default())?;
/// let response = conv.send("Hello!")?;
/// println!("{}", response.text);
/// ```
pub struct ConversationEngine {
    /// Channel to submit work to the scheduler thread.
    scheduler_tx: channel::Sender<SchedulerRequest>,

    /// Handle to the scheduler thread (joined on drop or on explicit shutdown).
    /// Wrapped in `Mutex<Option>` so `shutdown()` can be called via `&self`,
    /// enabling clean teardown from a `&'static ConversationEngine` reference
    /// (e.g. from a thread-local drop guard in tests).
    scheduler_handle: Mutex<Option<JoinHandle<()>>>,

    /// Tokenizer (shared, immutable, safe to clone into conversations).
    tokenizer: Arc<tokenizers::Tokenizer>,

    /// Engine-wide configuration.
    #[allow(dead_code)]
    config: EngineConfig,

    /// Workspace-shared `Conversation` handle: holds per-turn
    /// metadata (token counts, scores, sig entries, sealed-sequence
    /// handles) across every `Sequence` allocated from this engine.
    /// Each `Sequence` receives a clone.
    conversation: Conversation,

    /// Shared hot-tier KV cache — one instance per engine, cloned into every
    /// [`Substrate`](crate::substrate::Substrate).  VRAM byte accounting and
    /// the eviction budget are therefore global across all sessions.
    ///
    /// Budget is set via [`ConversationEngine::init_hot_budget`] after model
    /// weights are fully resident and a post-load CUDA free-memory query is
    /// available.
    substrate_cache: SubstrateCache,

    /// Shared provenance signature file (anonymous temp, deleted on process exit).
    ///
    /// One file per engine, shared via `Arc` across all conversations so all
    /// turns are indexed in the same mmap-backed store.
    provenance: Arc<ProvenanceFile>,

    /// Static model properties captured before the model moves to the scheduler thread.
    model_core: ModelCoreProperties,
}

impl ConversationEngine {
    /// Create a new engine with a pre-built model and tokenizer.
    ///
    /// Spawns the scheduler thread. The model must be `Send + 'static`
    /// because it is moved to the scheduler thread.
    ///
    /// # Arguments
    ///
    /// * `model` — Any type implementing [`ManagedBatchedModel`]. Typically
    ///   `BatchedInference<M>` for some `M: BatchedModelCore`.
    /// * `tokenizer` — HuggingFace tokenizer for encoding/decoding text.
    /// * `config` — Engine-wide configuration (VRAM budgets, EOS token, etc.).
    pub fn new(
        model: Box<dyn ManagedBatchedModel + Send>,
        tokenizer: tokenizers::Tokenizer,
        config: EngineConfig,
    ) -> crate::Result<Self> {
        // Create the batched inference session on this thread, then move
        // it to the scheduler thread. Session creation touches the GPU
        // (arena allocation) but is a one-time cost.
        let session = model
            .create_batched_session(config.batched_config.clone())
            .map_err(ConversationError::Model)?;

        // Capture model metadata before the model moves to the scheduler thread.
        let model_core = model.model_core_properties();

        let eos_tokens = config.eos_tokens.clone();
        let vocab_size = config.vocab_size;
        let max_recent_len = config.max_recent_len;
        let show_special_tokens = config.show_special_tokens;
        let tokenizer_for_scheduler = tokenizer.clone();

        // Create the scheduler channel (unbounded — backpressure is per-conversation
        // via the turn_in_flight guard, not at the channel level).
        let (tx, rx) = channel::unbounded();

        // Workspace-shared `Conversation`: holds per-turn metadata
        // (the substrate handle).  Every `Sequence` we hand out gets a
        // clone of this handle, so they all attach to the same shared
        // substrate.
        //
        // Mandatory substrate persistence — the redo log under the
        // workspace's `.substrate/` directory (or the process CWD).
        let mut persistence = match config.workspace_path.as_ref() {
            Some(dir) => crate::persistence::SubstratePersistence::open_in(dir.as_ref()),
            None => crate::persistence::SubstratePersistence::open(),
        }
        .map_err(|e| {
            ConversationError::from(candle::Error::Msg(format!("substrate persistence: {e}")))
        })?;
        // Persist the model identity into the substrate's `ModelSpec` record —
        // compare-and-insert, so it only appends when the model differs from
        // what the log already records. Makes the log a self-contained image.
        if let Some(spec) = &config.model_spec {
            let wrote = persistence.set_model_spec(spec).map_err(|e| {
                ConversationError::from(candle::Error::Msg(format!("persist model spec: {e}")))
            })?;
            if wrote {
                persistence.commit().map_err(|e| {
                    ConversationError::from(candle::Error::Msg(format!("commit model spec: {e}")))
                })?;
            }
        }
        // Embed the tokenizer.json (compare-and-insert) so the log can
        // detokenize offline. ~11 MB for Qwen3, but written at most once per
        // distinct model since identical bytes are a no-op.
        if let Some(tok) = &config.tokenizer {
            let wrote = persistence.set_tokenizer(tok).map_err(|e| {
                ConversationError::from(candle::Error::Msg(format!("persist tokenizer: {e}")))
            })?;
            if wrote {
                persistence.commit().map_err(|e| {
                    ConversationError::from(candle::Error::Msg(format!("commit tokenizer: {e}")))
                })?;
            }
        }
        // Shared hot-tier cache.  Budget is derived from `config` if the caller
        // provided a post-load free-VRAM figure; otherwise unlimited.
        let substrate_cache = match config.hot_cache_free_vram_bytes {
            Some(free_vram) => SubstrateCache::new(
                free_vram,
                config.hot_cache_abs_reserve_bytes,
                config.hot_cache_rel_reserve_frac,
            ),
            None => SubstrateCache::unbounded(),
        };
        let conversation = Conversation::with_cache(substrate_cache.clone(), persistence);

        // Workspace-shared `ProvenanceFile`: created up-front so the
        // scheduler can append to it inline during `cleanup_finished`'s
        // post-Done seal step.
        let provenance = Arc::new(ProvenanceFile::new().map_err(ConversationError::from)?);
        let scheduler_provenance = Arc::clone(&provenance);

        // Spawn the scheduler thread.
        let penalty_log = config.penalty_log_path.clone();
        let health_config = config.health.clone();
        // A clone of the workspace conversation for the scheduler thread —
        // used on startup to rebuild the substrate from the redo log.
        let scheduler_conversation = conversation.clone();
        let handle = std::thread::Builder::new()
            .name("conversation-scheduler".into())
            .spawn(move || {
                let mut scheduler = Scheduler::new(
                    rx,
                    model,
                    session,
                    tokenizer_for_scheduler,
                    eos_tokens,
                    vocab_size,
                    max_recent_len,
                    show_special_tokens,
                    penalty_log,
                    health_config,
                    config.scheduler.large_prefill_max_tokens,
                    scheduler_provenance,
                    model_core,
                );
                // §16.12 — reload any persisted turns into the substrate
                // before serving requests.
                #[cfg(feature = "cuda")]
                scheduler.reconstruct_substrate(&scheduler_conversation);
                #[cfg(not(feature = "cuda"))]
                let _ = scheduler_conversation;
                scheduler.run();
            })
            .map_err(|e| {
                ConversationError::Channel(format!("failed to spawn scheduler thread: {e}"))
            })?;

        Ok(Self {
            scheduler_tx: tx,
            scheduler_handle: Mutex::new(Some(handle)),
            tokenizer: Arc::new(tokenizer),
            config,
            provenance,
            model_core,
            conversation,
            substrate_cache,
        })
    }

    /// Activate the shared hot-tier VRAM budget.
    ///
    /// Call this after model weights are fully resident.  Pass the free-VRAM
    /// figure from a post-load CUDA memory query (`cuMemGetInfo` or equivalent)
    /// so model weight consumption is automatically excluded from the budget.
    ///
    /// `abs_reserve_bytes` is a fixed floor held back for decode activations
    /// and attention scratch space.  `rel_reserve_frac` (e.g. `0.05`) is an
    /// additional fractional reserve.  Both are subtracted from
    /// `free_vram_bytes`; the result becomes the cap shared across all sessions.
    pub fn init_hot_budget(
        &self,
        free_vram_bytes: u64,
        abs_reserve_bytes: u64,
        rel_reserve_frac: f64,
    ) {
        self.substrate_cache
            .activate_budget(free_vram_bytes, abs_reserve_bytes, rel_reserve_frac);
    }

    /// Create a new conversation.
    ///
    /// Creates a sequence slot on the scheduler. The system prompt is NOT
    /// prefilled here — call [`Sequence::initial_handle`] to prefill
    /// the system prompt + user header and get a [`TurnHandle`] confirming
    /// the tokens are in the KV cache.
    ///
    /// # Arguments
    ///
    /// * `system_prompt` — The formatted system prompt text. Pass `""` for none.
    /// * `config` — Per-conversation configuration (role markers, sampling, etc.).
    pub fn new_conversation(
        &self,
        system_prompt: &str,
        config: SequenceConfig,
    ) -> crate::Result<Sequence> {
        // Build a synthetic single-layer / single-group projection from the
        // raw prompt and delegate to the projection-aware constructor,
        // which mints a fresh `TimelineId` internally.
        //
        // `for_plain_prompt` stores the text as the schema section content,
        // and `new_with_projection` wraps it with dialect markers at ingest
        // time — so we strip the markers here before passing the inner text.
        let inner_prompt = {
            let s = system_prompt
                .strip_prefix(config.dialect.system_start)
                .unwrap_or(system_prompt);
            s.strip_suffix(config.dialect.system_end).unwrap_or(s)
        };
        let builder = Builder::for_plain_prompt(inner_prompt);
        let (layer_id, group_id) = {
            let layer = &builder.schema().layers[0];
            (layer.id, layer.groups[0].id)
        };
        self.new_conversation_with_projection(system_prompt, builder, layer_id, group_id, config)
    }

    /// Create a new conversation backed by a full projection [`Builder`].
    ///
    /// Identical to [`Self::new_conversation`] except the given `builder`
    /// replaces the synthetic schema constructed from the prompt string,
    /// and `target` names the `(layer, group)` this conversation is for —
    /// turns are appended into `target.group`, and `target` is what gets
    /// passed to `projection.project()`.
    ///
    /// `system_prompt` must be the ChatML-formatted text (same as you would
    /// pass to `new_conversation`).
    pub fn new_conversation_with_projection(
        &self,
        system_prompt: &str,
        builder: Builder,
        layer: crate::projection::LayerId,
        group: crate::projection::GroupId,
        config: SequenceConfig,
    ) -> crate::Result<Sequence> {
        // Persist the projection schema as the substrate's `Template` record
        // (compare-and-insert) so the log carries the projection it was built
        // with. Programmatic schemas (no source YAML) are skipped.
        if let Some(yaml) = builder.source_yaml() {
            if let Err(e) = self.conversation.set_template(yaml.as_bytes()) {
                tracing::warn!("persist projection template failed: {e}");
            }
        }

        // Mint a fresh `TimelineId` for this conversation before
        // allocating a slot — the substrate registers it against
        // `(layer, group)` so the seal path can write turns into the
        // right timeline without consulting the schema.
        let timeline = self.conversation.mint_timeline(layer, group);
        let target = ProjectionTarget {
            layer,
            group,
            timeline,
        };

        let (response_tx, response_rx) = channel::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::NewSequence {
                conversation: self.conversation.clone(),
                target: Some(target),
                response_tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;

        let sequence_id = response_rx
            .recv()
            .map_err(|_| ConversationError::SchedulerGone)??;

        let provenance = Arc::clone(&self.provenance);
        let conv = Sequence::new_with_projection(
            self.scheduler_tx.clone(),
            sequence_id,
            Arc::clone(&self.tokenizer),
            system_prompt,
            builder,
            target,
            config,
            CHUNK_SIZE,
            provenance,
            self.model_core,
            self.conversation.clone(),
        )?;

        Ok(conv)
    }

    /// Get the shared tokenizer.
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Get a clone of the shared provenance file.
    ///
    /// Allows external code (e.g. data-generation tools) to read back
    /// the signatures written during turn seals.  The Arc keeps the
    /// backing file alive as long as any clone exists.
    pub fn provenance_file(&self) -> Arc<ProvenanceFile> {
        Arc::clone(&self.provenance)
    }

    /// Static model properties captured at engine construction.
    pub fn model_core_properties(&self) -> ModelCoreProperties {
        self.model_core
    }

    /// Low-level helper used by benchmarks (e.g. RULER): create a fresh
    /// sequence, prefill the supplied token IDs, decode argmax until EOS or
    /// `max_decode_tokens`, and return the decoded text.
    ///
    /// All parallelism comes from many threads calling this concurrently;
    /// the scheduler batches their prefills and decodes together.
    pub fn infer_raw_tokens(
        &self,
        tokens: &[u32],
        max_decode_tokens: usize,
    ) -> crate::Result<String> {
        use crate::config::SamplingConfig;
        use crate::handle::TurnEvent;

        // 1. Allocate a sequence.
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::NewSequence {
                conversation: self.conversation.clone(),
                // Raw RULER eval path: no projection, no substrate
                // write, so no target binding either.
                target: None,
                response_tx: resp_tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        let sequence_id = resp_rx
            .recv()
            .map_err(|_| ConversationError::SchedulerGone)??;

        // 2. Submit the turn with raw tokens.  The scheduler carves a
        //    view over the parent's full block range and auto-finalizes
        //    on Done — for a fresh parent with no blocks the view
        //    borrows nothing and decoded blocks transfer back on
        //    finalize.
        let (event_tx, event_rx) = channel::unbounded();
        self.scheduler_tx
            .send(SchedulerRequest::SubmitTurn {
                sequence_id: sequence_id,
                projection_inputs: None,
                prefill_tokens: TokenBuffer::from(tokens.to_vec()),
                prefill_text: String::new(),
                post_decode_tokens: TokenBuffer::new(),
                max_decode_tokens,
                sampling: SamplingConfig::argmax(),
                event_tx,
                reprojection: None,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;

        // 3. Drain events until Done / Error.
        let mut collected: Vec<u32> = Vec::with_capacity(max_decode_tokens);
        let mut text_from_done: Option<String> = None;
        let mut last_error: Option<ConversationError> = None;
        while let Ok(ev) = event_rx.recv() {
            match ev {
                TurnEvent::Token(t) => collected.push(t),
                TurnEvent::Done(resp) => {
                    text_from_done = Some(resp.text);
                    break;
                }
                TurnEvent::Error(e) => {
                    last_error = Some(e);
                    break;
                }
                _ => {}
            }
        }

        // 4. Always release the sequence slot — RULER spawns one sequence per
        //    sample and never reuses it, so without this the KV pool leaks
        //    until OOM.
        let _ = self
            .scheduler_tx
            .send(SchedulerRequest::FreeSequence { sequence_id });

        if let Some(e) = last_error {
            return Err(e);
        }
        if let Some(t) = text_from_done {
            return Ok(t);
        }

        // Fallback: decode whatever tokens we collected.
        self.tokenizer
            .decode(&collected, true)
            .map_err(|e| ConversationError::Channel(format!("decode failed: {e}")))
    }

    /// Get a `TokenDecoder` for decoding token IDs into text.
    ///
    /// The decoder is cheap to clone and can be used across threads.
    pub fn token_decoder(&self) -> crate::handle::TokenDecoder {
        crate::handle::TokenDecoder::new(Arc::clone(&self.tokenizer))
    }

    /// Durably flush the substrate redo log — the group-commit point.
    /// Call after a turn completes so an in-flight turn survives a crash.
    pub fn commit_persistence(&self) -> crate::Result<()> {
        Ok(self.conversation.commit_persistence()?)
    }

    /// Flush, optionally compact, and checkpoint the substrate redo log —
    /// the fast-recovery snapshot. Call on the daemon's checkpoint cadence
    /// and as part of graceful shutdown.
    pub fn checkpoint_persistence(&self) -> crate::Result<()> {
        Ok(self.conversation.checkpoint_persistence()?)
    }

    /// Shut down the scheduler, releasing all GPU resources.
    ///
    /// Safe to call multiple times (idempotent). Takes `&self` so it can be
    /// called from a `&'static ConversationEngine` reference (e.g. a TLS drop
    /// guard in tests), which is necessary because Rust statics are never
    /// dropped and we must ensure the CUDA scheduler thread exits before the
    /// CUDA driver's atexit handler fires.
    pub fn shutdown(&self) -> crate::Result<()> {
        let _ = self.scheduler_tx.send(SchedulerRequest::Shutdown);
        let handle = self
            .scheduler_handle
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(h) = handle {
            h.join()
                .map_err(|_| ConversationError::Channel("scheduler thread panicked".into()))?;
        }
        Ok(())
    }
}

impl Drop for ConversationEngine {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
