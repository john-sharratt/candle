//! The conversation engine: entry point, spawns the scheduler thread.

use crate::config::{EngineConfig, SamplingConfig, SequenceConfig};
use crate::conversation::Sequence;
use crate::error::ConversationError;
use crate::handle::{TokenDecoder, TurnEvent};
use crate::persistence::thread::PersistenceThread;
use crate::persistence::SubstratePersistence;
use crate::projection::{
    Builder, Conversation, GroupId, LayerId, ProjectionTarget, Reserved, TimelineId,
};
use crate::provenance::ProvenanceFile;
use crate::scheduler::{Scheduler, SchedulerRequest};
use crate::substrate::Substrate;
use crate::summary_tree::{ChannelProbeRunner, SelectionDiagnostics, SummariserThread};
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

    /// Substrate persistence thread — owns the redo-log write path.
    /// Wakes on a 5-second tick or on triggers from the scheduler;
    /// joined on engine drop (or via [`Self::shutdown`]) after a final
    /// drain pass. `PersistenceThread::shutdown` is `&self`-callable —
    /// no `Option`/`Mutex` shuffle at this layer.
    persist_thread: PersistenceThread,

    /// Async summariser thread — drains the per-turn pending queue,
    /// runs §6 probes, builds the per-timeline AVL summary tree.
    /// Mirrors [`PersistenceThread`]'s lifecycle (trigger / tick /
    /// shutdown).  Spawned alongside the scheduler at engine startup;
    /// [`Self::shutdown`] joins it after the persistence thread has
    /// drained, and `Drop` falls through to the same path.
    summariser_thread: SummariserThread,

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
        mut config: EngineConfig,
    ) -> crate::Result<Self> {
        // Capture model metadata before the model moves to the scheduler thread.
        let model_core = model.model_core_properties();

        // Plumb the model's tuned K/V error threshold factors into the
        // batched config so the persistence thread's `compression_policy()`
        // (built from `config.batched_config`) uses them. Without this the
        // policy falls back to identity factors (1.0) regardless of which
        // model is loaded — the 24-iter Qwen3 tuning would never reach the
        // selection kernel. The model impl is the single source of truth;
        // `BatchedModelCore::*_error_threshold_factor()` returns the per-model
        // constant (e.g. `QWEN3_MOE_KV_FACTORS`).
        config.batched_config.k_hi_error_threshold_factor =
            model_core.k_hi_error_threshold_factor;
        config.batched_config.k_low_error_threshold_factor =
            model_core.k_low_error_threshold_factor;
        config.batched_config.v_hi_error_threshold_factor =
            model_core.v_hi_error_threshold_factor;
        config.batched_config.v_low_error_threshold_factor =
            model_core.v_low_error_threshold_factor;

        // Create the batched inference session on this thread, then move
        // it to the scheduler thread. Session creation touches the GPU
        // (arena allocation) but is a one-time cost.
        let session = model
            .create_batched_session(config.batched_config.clone())
            .map_err(ConversationError::Model)?;

        let eos_tokens = config.eos_tokens.clone();
        let vocab_size = config.vocab_size;
        let max_recent_len = config.max_recent_len;
        let show_special_tokens = config.show_special_tokens;
        let tokenizer_for_scheduler = tokenizer.clone();

        // Pre-tokenise the dialect's `user_start` and `assistant_end`
        // strings once at engine construction.  The scheduler hands a
        // borrow of this to every `ApplyContext` so the assembler can
        // wrap every `Sealed::Turn` in live-prefilled boundary
        // markers without re-tokenising on each projection.
        let boundary_markers = crate::scheduler::projection_assembler::BoundaryMarkers::from_dialect(
            &config.dialect,
            |s| {
                let encoded = tokenizer.encode(s, false).map_err(|e| {
                    ConversationError::Channel(format!("boundary marker tokenise: {e}"))
                })?;
                Ok::<_, ConversationError>(encoded.get_ids().to_vec())
            },
        )?;

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
        // Open persistence and drive every record straight into the
        // substrate's in-RAM state in one walker pass — no manifest
        // mirror, no `reconstruct → collected_*` second pass.
        let mut substrate = Substrate::new();
        let workspace_dir: std::path::PathBuf = match config.workspace_path.as_ref() {
            Some(p) => AsRef::<std::path::Path>::as_ref(p).to_path_buf(),
            None => std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
        };
        let mut persistence = SubstratePersistence::open_in_with_substrate(
            &workspace_dir,
            &mut substrate,
        )
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
        let conversation = Conversation::from_parts(substrate, persistence);

        // Workspace-shared `ProvenanceFile`: created up-front so the
        // scheduler can append to it inline during `cleanup_finished`'s
        // post-Done seal step.
        let provenance = Arc::new(ProvenanceFile::new()?);
        let scheduler_provenance = Arc::clone(&provenance);

        // Spawn the substrate persistence thread (§5s heartbeat + per-
        // seal trigger). Owns the redo-log write path; needs backings +
        // device for hot→warm migration and warm→cold gather. Spawn it
        // **before** the scheduler thread so the trigger handle can be
        // handed in.
        let backings: Arc<Vec<candle_nn::kv_cache::ChunkedKvBacking>> =
            Arc::new(session.backings().to_vec());
        let persist_thread = crate::persistence::thread::PersistenceThread::spawn(
            conversation.clone(),
            Arc::clone(&backings),
            session.device().clone(),
            config.batched_config.compression_policy(),
        );
        let persist_trigger = persist_thread.trigger_handle();

        // Spawn the async summariser thread (`docs/infinite_conversations.md`
        // §3.3 / §7).  Drains the per-timeline pending queue every
        // 250 ms (or on trigger), runs §6 probes via the
        // scheduler-backed [`ChannelProbeRunner`], extends the
        // per-timeline summary tree, and persists the resulting
        // [`TreeMetadata`] records to the redo log.  Spawned after the
        // persistence thread so its writes flow through the same
        // workspace handle.
        let summariser_runner = Arc::new(ChannelProbeRunner::new(tx.clone()));
        let summariser_thread =
            SummariserThread::spawn(conversation.clone(), summariser_runner);
        // Hand the trigger to the scheduler so every assistant-turn
        // seal wakes the summariser immediately — design §4 step ③.
        let summariser_trigger = summariser_thread.trigger_handle();

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
                    persist_trigger,
                    summariser_trigger,
                    boundary_markers,
                );
                // §16.12 — reload any persisted turns into the substrate
                // before serving requests.
                scheduler.reconstruct_substrate(&scheduler_conversation);
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
            persist_thread,
            summariser_thread,
        })
    }

    /// Clone the workspace `Conversation` handle.
    ///
    /// The handle wraps the substrate behind its `RwLock`, so callers
    /// can take `.read()` / `.write()` views to inspect or mutate
    /// section / turn / timeline state directly.  Most production
    /// callers should instead go through `new_conversation` /
    /// `new_conversation_with_projection` / `Sequence::submit_turn`
    /// — this accessor is for tooling that needs the raw substrate
    /// (integration tests, diagnostics, the workspace inspector).
    pub fn conversation(&self) -> Conversation {
        self.conversation.clone()
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
    /// Persist a substrate-side resume key (`debug_id`) for
    /// `timeline`.  Used by the debug-id-resumable grow-conversation
    /// harness (`docs/infinite_conversations.md` §10.4): a test can
    /// re-open the workspace, call [`Self::lookup_by_debug_id`] to
    /// find a previously-built timeline, and continue growing.
    ///
    /// Last-write-wins on replay.  Idempotent: the redo-log writer
    /// skips the append when the substrate already records the same
    /// value.
    pub fn set_conversation_debug_id(
        &self,
        timeline: TimelineId,
        debug_id: &str,
    ) -> crate::Result<()> {
        self.conversation
            .set_conversation_debug_id(timeline, debug_id)
            .map_err(ConversationError::Model)
    }

    /// Look up a timeline by its previously-set `debug_id`.  O(1).
    /// Returns `None` when no timeline carries that key.
    pub fn lookup_by_debug_id(&self, debug_id: &str) -> Option<TimelineId> {
        self.conversation.lookup_by_debug_id(debug_id)
    }

    /// Backpressure metric — turns awaiting summariser absorption for
    /// `timeline`.  Zero in steady state.
    pub fn pending_summary_len(&self, timeline: TimelineId) -> usize {
        self.conversation.pending_summary_len(timeline)
    }

    /// Backpressure metric — summary nodes currently dirty for
    /// `timeline`.  Zero when the dirty sweep is caught up.
    pub fn dirty_summary_len(&self, timeline: TimelineId) -> usize {
        self.conversation.dirty_summary_len(timeline)
    }

    /// Test-harness diagnostic — the most recent score-density
    /// [`SelectionDiagnostics`] for `timeline`, or `None` if no
    /// projection has run yet (or projection used the rule-based
    /// path).  Last-write-wins across reprojections within a turn.
    pub fn last_selection_diagnostics(
        &self,
        timeline: TimelineId,
    ) -> Option<SelectionDiagnostics> {
        self.conversation.last_selection_diagnostics(timeline)
    }

    /// Persist a sidebar label for `timeline` to the workspace substrate.
    /// Last-write-wins; preserves whatever `conv_id` is already known
    /// for this timeline. The daemon's titler is the typical caller.
    pub fn set_conversation_label(
        &self,
        timeline: TimelineId,
        label: &str,
    ) -> crate::Result<()> {
        self.conversation
            .set_conversation_label(timeline, label)
            .map_err(ConversationError::Model)
    }

    /// Persist the client-supplied `conv_id` for `timeline`. Idempotent;
    /// callers can invoke on every submit. Preserves any existing label.
    /// This is the "substrate-as-single-source-of-truth" replacement for
    /// the old daemon-side `conv_labels.json` sidecar — the conv_id ↔
    /// timeline mapping now lives in the redo log.
    pub fn set_conversation_conv_id(
        &self,
        timeline: TimelineId,
        conv_id: &str,
    ) -> crate::Result<()> {
        self.conversation
            .set_conversation_conv_id(timeline, conv_id)
            .map_err(ConversationError::Model)
    }

    /// Read the workspace substrate's sidebar label for `timeline`, or
    /// `None` if none has been recorded. Useful for "should we still run
    /// the titler?" checks at submit time.
    pub fn conversation_label_of(&self, timeline: TimelineId) -> Option<String> {
        self.conversation.label_of(timeline)
    }

    /// Every conversation the workspace substrate knows about —
    /// `(timeline, conv_id, label, archived)` quads. Drives the
    /// daemon's `GET /v1/conversations` sidebar listing directly.
    pub fn known_conversations(
        &self,
    ) -> Vec<(TimelineId, String, String, bool)> {
        self.conversation.known_conversations()
    }

    /// Toggle the archived lifecycle flag for a conversation. Persists
    /// to the redo log as `RecordType::ConvState` (last-writer-wins)
    /// and updates the in-RAM substrate. Drives the daemon's
    /// `POST /v1/conversations/{id}/archive` and `/unarchive`.
    pub fn set_conversation_archived(
        &self,
        timeline: TimelineId,
        archived: bool,
    ) -> crate::Result<()> {
        self.conversation
            .set_conversation_archived(timeline, archived)
            .map_err(ConversationError::Model)
    }

    /// Tombstone `timeline` — see
    /// [`crate::projection::Conversation::tombstone_timeline`].
    pub fn tombstone_timeline(&self, timeline: TimelineId) -> crate::Result<()> {
        self.conversation
            .tombstone_timeline(timeline)
            .map_err(ConversationError::Model)
    }

    /// Build an **engine-internal** conversation that lives on the reserved
    /// id range for `kind` — disjoint from any YAML-allocated user schema.
    ///
    /// This is the right entry point for synthetic helper conversations
    /// (the daemon's titler, future label-summarisers, etc.) that share
    /// the same workspace substrate as user conversations but must never
    /// have their turns enter user-projection retrieval.
    ///
    /// `system_prompt` may be either pre-formatted with dialect markers
    /// or raw text — `new_conversation_with_projection` handles wrapping
    /// the same way it does for user prompts.
    pub fn new_reserved_conversation(
        &self,
        system_prompt: &str,
        kind: Reserved,
        config: SequenceConfig,
    ) -> crate::Result<Sequence> {
        let builder = Builder::for_plain_prompt_reserved(system_prompt, kind);
        let layer_id = LayerId::reserved(kind);
        let group_id = GroupId::reserved(kind);
        self.new_conversation_with_projection(system_prompt, builder, layer_id, group_id, config)
    }

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
        layer: LayerId,
        group: GroupId,
        config: SequenceConfig,
    ) -> crate::Result<Sequence> {
        self.new_conversation_with_projection_progress(
            system_prompt,
            builder,
            layer,
            group,
            config,
            None,
        )
    }

    /// Same as [`Self::new_conversation_with_projection`] but accepts an
    /// optional progress callback fired as the schema's pinned sections
    /// are prefilled. The callback receives
    /// `(chars_done, total_chars)` — total content-bytes across every
    /// schema-declared section (including collection members). Used by
    /// the daemon's loading overlay; library callers pass `None`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_conversation_with_projection_progress(
        &self,
        system_prompt: &str,
        builder: Builder,
        layer: LayerId,
        group: GroupId,
        config: SequenceConfig,
        section_progress: Option<&dyn Fn(u64, u64)>,
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
            section_progress,
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
                sequence_id,
                projection_inputs: None,
                prefill_tokens: TokenBuffer::from(tokens.to_vec()),
                prefill_text: String::new(),
                user_text: String::new(),
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
    pub fn token_decoder(&self) -> TokenDecoder {
        TokenDecoder::new(Arc::clone(&self.tokenizer))
    }

    /// Durably flush the substrate redo log — the group-commit point.
    /// Call after a turn completes so an in-flight turn survives a crash.
    pub fn commit_persistence(&self) -> crate::Result<()> {
        Ok(self.conversation.commit_persistence()?)
    }

    /// Like [`Self::commit_persistence`] but skipped when nothing is staged.
    /// Returns `Ok(true)` when an `fsync` actually happened. The daemon's
    /// periodic flush task uses this so an idle workspace doesn't issue
    /// pointless syscalls — and so writes produced asynchronously by the
    /// bg-quantizer's persist callback aren't left stranded between turns.
    pub fn commit_persistence_if_pending(&self) -> crate::Result<bool> {
        Ok(self.conversation.commit_persistence_if_pending()?)
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
        // Tear down the persistence thread after the scheduler has
        // joined — by then no more turn-seals will fire triggers, so
        // the thread's final drain pass captures the last work. The
        // call is idempotent, so a redundant invocation from `Drop`
        // after this is a no-op.
        self.persist_thread.shutdown();
        // Tear down the summariser thread last: it depends on the
        // scheduler for §6 probes (via `ChannelProbeRunner` over the
        // scheduler request channel).  Once the scheduler has joined,
        // any in-flight probe request hangs forever — shutdown
        // signals the loop to exit on its next select rather than
        // wait for a probe response that will never arrive.
        self.summariser_thread.shutdown();
        Ok(())
    }
}

impl Drop for ConversationEngine {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
