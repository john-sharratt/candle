//! The conversation engine: entry point, spawns the scheduler thread.

use crate::config::{EngineConfig, SamplingConfig, SequenceConfig};
use crate::conversation::Sequence;
use crate::error::ConversationError;
use crate::handle::{TokenDecoder, TurnEvent};
use crate::persistence::record::DistillMode;
use crate::persistence::thread::PersistenceThread;
use crate::persistence::SubstratePersistence;
use crate::projection::{
    Builder, Conversation, GroupId, LayerId, ProjectionTarget, Reserved, TimelineId,
};
use crate::scheduler::{Scheduler, SchedulerRequest};
use crate::sequence_handle::SequenceId;
use crate::stencil::{
    compile, compile_think_tree, compile_tool_call_tree, HfVocab, StencilTree, ThinkMode,
    ThinkSteerEnvelope, TokenId, ToolCallEnvelope, ToolSpec, TriggerRegistry,
};
use crate::substrate::{ConvCompression, Substrate};
use crate::summary_tree::{ChannelProbeRunner, SelectionDiagnostics, SummariserThread};
use crate::token_buffer::TokenBuffer;

use candle::Device;
use candle_nn::CHUNK_SIZE;
use candle_transformers::models::batched_inference::{ManagedBatchedModel, ModelCoreProperties};
use crossbeam::channel;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// The compiled thinking-block steering trees, one per non-`Off` effort dial,
/// built once at engine init (parallel to the tool-call registry).  Each turn
/// derives its trigger registry by replacing the `<think>` trigger with the
/// dial's tree — atomic and idempotent via [`TriggerRegistry::with_trigger`].
pub struct ThinkSteering {
    /// `<think>` id — the trigger the dial's tree is bound to.
    think_open: TokenId,
    quick: Arc<StencilTree>,
    balanced: Arc<StencilTree>,
    deep: Arc<StencilTree>,
    exhaustive: Arc<StencilTree>,
}

impl ThinkSteering {
    /// Derive a per-turn registry from `base` (e.g. the tool-call catalog) for
    /// `mode`: bind the `<think>` trigger to that dial's steering tree, or clear
    /// it for [`ThinkMode::Off`] (the `/no_think` glue yields the empty block, so
    /// no tree steers it).  The base is untouched; the result is a fresh registry.
    pub fn registry_for(&self, base: &TriggerRegistry, mode: ThinkMode) -> Arc<TriggerRegistry> {
        let tree = match mode {
            ThinkMode::Off => return Arc::new(base.without_trigger(self.think_open)),
            ThinkMode::Quick => &self.quick,
            ThinkMode::Balanced => &self.balanced,
            ThinkMode::Deep => &self.deep,
            ThinkMode::Exhaustive => &self.exhaustive,
        };
        Arc::new(base.with_trigger(self.think_open, Arc::clone(tree)))
    }
}

/// How many summary probes the summariser submits per batch, chosen by total
/// VRAM at engine init. Their decodes batch in the scheduler's wave loop, so a
/// bigger card can keep more summaries in flight: 16 above 32 GB, 4 at or below
/// (and on CPU, where there's no device VRAM to read).
fn summary_probe_concurrency(device: &Device) -> usize {
    const VRAM_32_GIB: usize = 32 * 1024 * 1024 * 1024;
    let total_vram = match device {
        Device::Cuda(d) => d.mem_get_info().map(|(_free, total)| total).unwrap_or(0),
        _ => 0,
    };
    if total_vram > VRAM_32_GIB {
        16
    } else {
        4
    }
}

/// Live progress of the startup substrate reload (redo-log replay), shared
/// between the scheduler thread (writer) and the daemon's load-state machine
/// (reader). As the substrate grows this replay stops being instantaneous, so
/// the GUI needs a real progress signal instead of a stalled loading bar.
#[derive(Debug, Default)]
pub struct SubstrateReloadStatus {
    /// Turns restored so far.
    done: AtomicUsize,
    /// Total turns to restore — `0` until the redo-log decl list is known.
    total: AtomicUsize,
    /// Set once the reload pass has finished (success *or* error). Readers must
    /// key completion off this, not `done == total` — corrupt turns are skipped
    /// so `done` may never reach `total`.
    finished: AtomicBool,
}

impl SubstrateReloadStatus {
    /// Writer (scheduler thread): record turns-restored / total-to-restore.
    pub fn record(&self, done: usize, total: usize) {
        self.total.store(total, Ordering::Relaxed);
        self.done.store(done, Ordering::Relaxed);
    }

    /// Writer (scheduler thread): mark the reload pass complete. Always called,
    /// even on the no-op / error paths, so a reader never waits forever.
    pub fn finish(&self) {
        self.finished.store(true, Ordering::Release);
    }

    /// Reader (load-state machine): `(done, total, finished)` snapshot.
    pub fn snapshot(&self) -> (usize, usize, bool) {
        let finished = self.finished.load(Ordering::Acquire);
        (
            self.done.load(Ordering::Relaxed),
            self.total.load(Ordering::Relaxed),
            finished,
        )
    }
}

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

    /// Static model properties captured before the model moves to the scheduler thread.
    model_core: ModelCoreProperties,

    /// Progress of the startup substrate reload, written by the scheduler
    /// thread and polled by the daemon's load-state machine for the GUI.
    substrate_reload_status: Arc<SubstrateReloadStatus>,
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
        config.batched_config.k_hi_error_threshold_factor = model_core.k_hi_error_threshold_factor;
        config.batched_config.k_low_error_threshold_factor =
            model_core.k_low_error_threshold_factor;
        config.batched_config.v_hi_error_threshold_factor = model_core.v_hi_error_threshold_factor;
        config.batched_config.v_low_error_threshold_factor =
            model_core.v_low_error_threshold_factor;

        // Create the batched inference session on this thread, then move
        // it to the scheduler thread. Session creation touches the GPU
        // (arena allocation) but is a one-time cost.
        let session_start = std::time::Instant::now();
        let session = model
            .create_batched_session(config.batched_config.clone())
            .map_err(ConversationError::Model)?;
        tracing::info!(
            session_init_ms = session_start.elapsed().as_millis() as u64,
            "batched session created (KV arenas allocated)"
        );

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
        let boundary_markers =
            crate::scheduler::projection_assembler::BoundaryMarkers::from_dialect(
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
        let open_start = std::time::Instant::now();
        let mut persistence = SubstratePersistence::open_in_with_substrate(
            &workspace_dir,
            &mut substrate,
        )
        .map_err(|e| {
            ConversationError::from(candle::Error::Msg(format!("substrate persistence: {e}")))
        })?;
        tracing::info!(
            open_ms = open_start.elapsed().as_millis() as u64,
            log_bytes = persistence.write_offset(),
            records = persistence.recovered_record_count(),
            indexed = persistence.last_index().is_some(),
            streams = substrate.all_streams().count(),
            "substrate persistence opened"
        );
        // Persist the model identity into the substrate's `ModelSpec` record —
        // compare-and-insert, so it only appends when the model differs from
        // what the log already records. Makes the log a self-contained image.
        let singletons_start = std::time::Instant::now();
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
        tracing::info!(
            singletons_ms = singletons_start.elapsed().as_millis() as u64,
            "model spec + tokenizer records reconciled"
        );
        let conversation = Conversation::from_parts(substrate, persistence);

        // Register per-layer corrupt-turn policies (from the projection schema)
        // BEFORE the reload thread is spawned, so the startup reconstruct applies
        // the right policy per layer (drop the whole conversation for ingest
        // layers, only the turn for dialogue). Empty ⇒ every layer defaults to
        // `DropConversation`.
        for (&layer, &policy) in &config.layer_corrupt_turn {
            conversation.set_layer_corrupt_turn_policy(layer, policy);
        }

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

        // Spawn the async summariser thread (`docs/immutable_summary_forest.md`
        // — *Two queues*).  Drains the per-timeline pending queue every
        // 250 ms (or on trigger), runs probes
        // (`docs/archived/infinite_conversations.md` §6) via the
        // scheduler-backed [`ChannelProbeRunner`], extends the
        // per-timeline summary tree, and persists the resulting
        // [`TreeMetadata`] records to the redo log.  Spawned after the
        // persistence thread so its writes flow through the same
        // workspace handle.
        let summariser_runner = Arc::new(ChannelProbeRunner::new(tx.clone()));
        let summary_concurrency = summary_probe_concurrency(session.device());
        tracing::info!(
            summary_concurrency,
            "summariser probe-batch concurrency set from total VRAM"
        );
        let summariser_thread = if config.disable_summariser {
            tracing::info!("disable_summariser: summariser thread not spawned");
            SummariserThread::disabled()
        } else {
            SummariserThread::spawn(conversation.clone(), summariser_runner, summary_concurrency)
        };
        // Hand the trigger to the scheduler so every assistant-turn
        // seal wakes the summariser immediately — design §4 step ③.
        let summariser_trigger = summariser_thread.trigger_handle();

        // Spawn the scheduler thread.
        let penalty_log = config.penalty_log_path.clone();
        let health_config = config.health.clone();
        // A clone of the workspace conversation for the scheduler thread —
        // used on startup to rebuild the substrate from the redo log.
        let scheduler_conversation = conversation.clone();
        // Shared reload-progress handle: the scheduler thread updates it during
        // the redo-log replay; the daemon polls it for the loading screen.
        let substrate_reload_status = Arc::new(SubstrateReloadStatus::default());
        let reload_status_for_thread = Arc::clone(&substrate_reload_status);
        let handle = std::thread::Builder::new()
            .name("conversation-scheduler".into())
            .spawn(move || {
                let t_init = std::time::Instant::now();
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
                    persist_trigger,
                    summariser_trigger,
                    boundary_markers,
                );
                // Localizes the startup gap between "model loaded" and the
                // substrate progress bar moving: this is the scheduler thread
                // binding its CUDA context + allocating its buffers, before the
                // redo-log replay (which reports its own progress) begins.
                tracing::info!(
                    init_ms = t_init.elapsed().as_millis() as u64,
                    "scheduler thread: init complete, starting substrate reconstruct"
                );
                // §16.12 — reload any persisted turns into the substrate
                // before serving requests, reporting progress to the daemon.
                let t_recon = std::time::Instant::now();
                scheduler.reconstruct_substrate(&scheduler_conversation, &reload_status_for_thread);
                tracing::info!(
                    reconstruct_ms = t_recon.elapsed().as_millis() as u64,
                    "scheduler thread: substrate reconstruct complete"
                );
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
            model_core,
            conversation,
            persist_thread,
            summariser_thread,
            substrate_reload_status,
        })
    }

    /// Shared handle to the startup substrate-reload progress. The daemon's
    /// load-state machine polls [`SubstrateReloadStatus::snapshot`] to drive
    /// the GUI's "Loading substrate" step while the redo log replays.
    pub fn substrate_reload_status(&self) -> Arc<SubstrateReloadStatus> {
        Arc::clone(&self.substrate_reload_status)
    }

    /// Re-reconstruct the substrate on the scheduler thread (needs the model
    /// backings for KV residence) — call after a compaction rewrites the redo log
    /// so all offsets / KV pointers are rebuilt from the new log. Returns a fresh
    /// status handle; poll [`SubstrateReloadStatus::snapshot`] until `finished`.
    pub fn reload_substrate(&self) -> Arc<SubstrateReloadStatus> {
        let status = Arc::new(SubstrateReloadStatus::default());
        let _ = self
            .scheduler_tx
            .send(SchedulerRequest::ReconstructSubstrate {
                conversation: self.conversation.clone(),
                status: Arc::clone(&status),
            });
        status
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
    /// How many sequences the model currently holds recurrent memory for.
    ///
    /// The leak gauge. Slot ids are recycled pool indices, so memory that
    /// outlives its conversation is not merely wasted VRAM — the next
    /// conversation on that id inherits a stranger's memory, fluently.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn live_memory_count(&self) -> usize {
        let (tx, rx) = crossbeam::channel::bounded(1);
        if self
            .scheduler_tx
            .send(crate::scheduler::SchedulerRequest::CountRecurrentMemories { response_tx: tx })
            .is_err()
        {
            return 0;
        }
        rx.recv().unwrap_or(0)
    }

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
    /// harness (`docs/archived/infinite_conversations.md` §10.4): a test can
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

    /// Wake the summariser thread now instead of waiting for its next
    /// tick — used to kick off summarisation of freshly-ingested turns promptly.
    pub fn trigger_summariser(&self) {
        self.summariser_thread.trigger();
    }

    /// Test-harness diagnostic — the most recent score-density
    /// [`SelectionDiagnostics`] for `timeline`, or `None` if no
    /// projection has run yet (or projection used the rule-based
    /// path).  Last-write-wins across reprojections within a turn.
    pub fn last_selection_diagnostics(&self, timeline: TimelineId) -> Option<SelectionDiagnostics> {
        self.conversation.last_selection_diagnostics(timeline)
    }

    /// Persist a sidebar label for `timeline` to the workspace substrate.
    /// Last-write-wins; preserves whatever `conv_id` is already known
    /// for this timeline. The daemon's titler is the typical caller.
    pub fn set_conversation_label(&self, timeline: TimelineId, label: &str) -> crate::Result<()> {
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

    /// Set (or clear) the per-conversation KV-compression override for
    /// `timeline` at runtime — used by the daemon to flag a forked capture
    /// conversation as lossless (native R16/F16, no quantize) before its first
    /// turn migrates hot→warm. See [`crate::substrate::ConvCompression`].
    pub fn set_timeline_compression(
        &self,
        timeline: TimelineId,
        compression: Option<ConvCompression>,
    ) {
        self.conversation
            .set_timeline_compression(timeline, compression);
    }

    /// Enable or disable AVL summarisation for `timeline`. Conversations default
    /// to `true`; scratch/scaffolding timelines (e.g. the tool-summary
    /// categorize/assign passes) set `false` before their first turn seals so
    /// the wave-driven summariser never spends a compression decode on work that
    /// is about to be tombstoned. See [`crate::summary_tree`].
    pub fn set_timeline_summarize(&self, timeline: TimelineId, summarize: bool) {
        self.conversation
            .set_timeline_summarize(timeline, summarize);
    }

    /// Mark `layer` as an append-only ingest layer (code_reading/repo_map): a
    /// projection targeting it is scored/selected self-local (belief groups masked
    /// to the target timeline), so an ingest scope-summary is grounded in its own
    /// scope rather than cross-file retrieval. Called once per ingest layer at
    /// setup. See [`crate::substrate::Substrate::mark_layer_append_only`].
    pub fn mark_layer_append_only(&self, layer: LayerId) {
        self.conversation.mark_layer_append_only(layer);
    }

    /// Re-arm the score-normalization warm-up so the next projection re-learns
    /// per-file hit levels — call after an ingest reconcile mints fresh timelines
    /// the prior warm never scanned. See
    /// [`crate::projection::Conversation::reset_normalization_warm`].
    pub fn reset_normalization_warm(&self) {
        self.conversation.reset_normalization_warm();
    }

    /// Warm the ingest layers' per-file hit levels from their own turns. Call
    /// AFTER an ingest pass / reconcile finishes (never concurrently — it would
    /// starve the ingest writer). See
    /// [`crate::projection::Conversation::warm_ingest_normalization`].
    pub fn warm_ingest_normalization(&self, schema: &crate::projection::Schema) {
        self.conversation.warm_ingest_normalization(schema);
    }

    /// Merge a `(key, value)` into `timeline`'s free-form `custom`
    /// metadata bag and persist it. Used by utility ingests to tag each
    /// conversation with a content hash + descriptive fields for the
    /// restart-resume cache.
    pub fn set_conversation_metadata(
        &self,
        timeline: TimelineId,
        key: &str,
        value: &str,
    ) -> crate::Result<()> {
        self.conversation
            .set_conversation_metadata(timeline, key, value)
            .map_err(ConversationError::Model)
    }

    /// `timeline`'s `custom` metadata bag, or `None` if unregistered.
    pub fn conversation_metadata(
        &self,
        timeline: TimelineId,
    ) -> Option<std::collections::BTreeMap<String, String>> {
        self.conversation.conversation_metadata(timeline)
    }

    /// Every live conversation whose `custom` metadata contains `key == value`.
    /// The content-addressed lookup utility ingests use after substrate
    /// load to skip rebuilding units already present (tombstoned excluded).
    pub fn find_conversations_by_metadata(&self, key: &str, value: &str) -> Vec<TimelineId> {
        self.conversation.find_timelines_by_metadata(key, value)
    }

    /// [`Self::find_conversations_by_metadata`] plus tombstoned conversations
    /// that carry a distillation mode — the provenance corpus, whose designed
    /// end state is archived + distilled + tombstoned. Ordinary tombstones stay
    /// excluded. Used by the calibration resume filter.
    pub fn find_conversations_by_metadata_including_distilled(
        &self,
        key: &str,
        value: &str,
    ) -> Vec<TimelineId> {
        self.conversation
            .find_timelines_by_metadata_including_distilled(key, value)
    }

    /// One-pass snapshot of the distinct `custom[key]` values across live
    /// conversations — for O(1) resume-cache membership probing.
    pub fn conversation_metadata_values(&self, key: &str) -> std::collections::HashSet<String> {
        self.conversation.metadata_values_for_key(key)
    }

    /// Live conversations carrying `key`, paired with its value. Drives
    /// ingest reconciliation (tombstone units whose source file is gone).
    pub fn conversations_with_metadata_key(&self, key: &str) -> Vec<(TimelineId, String)> {
        self.conversation.timelines_with_metadata_key(key)
    }

    /// Every conversation the workspace substrate knows about —
    /// `(timeline, conv_id, label, archived, order)` tuples, where `order`
    /// is the creation-order rank ([`crate::substrate::TimelineEntry::order`]).
    /// Drives the daemon's `GET /v1/conversations` sidebar listing directly.
    pub fn known_conversations(&self) -> Vec<(TimelineId, String, String, bool, u64)> {
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

    /// Whether `timeline` is archived. Unlike [`Self::known_conversations`]
    /// — which omits internal conversations that never set a `conv_id` — this
    /// reads the flag directly, so it works for reserved/utility timelines too.
    pub fn is_conversation_archived(&self, timeline: TimelineId) -> bool {
        self.conversation.is_conversation_archived(timeline)
    }

    /// Tombstone `timeline` — see
    /// [`crate::projection::Conversation::tombstone_timeline`].
    pub fn tombstone_timeline(&self, timeline: TimelineId) -> crate::Result<()> {
        self.conversation
            .tombstone_timeline(timeline)
            .map_err(ConversationError::Model)
    }

    /// Mark `timeline` for distillation at `mode` (shed content at compaction) —
    /// see [`crate::projection::Conversation::distill_timeline`]. A later call may
    /// upgrade the mode; gate on [`Self::is_timeline_distilled`] only to avoid
    /// re-marking at the same mode.
    pub fn distill_timeline(&self, timeline: TimelineId, mode: DistillMode) -> crate::Result<()> {
        self.conversation
            .distill_timeline(timeline, mode)
            .map_err(ConversationError::Model)
    }

    /// Whether `timeline` is already marked for distillation.
    pub fn is_timeline_distilled(&self, timeline: TimelineId) -> bool {
        self.conversation.is_timeline_distilled(timeline)
    }

    /// Demote the hot K/V of `timelines` to the warm (RAM) tier, keeping the
    /// warm copy — the VRAM the hot copies held returns to the pool. The demote
    /// itself runs on the scheduler thread (single-owner GPU-pool mutation).
    /// Used by the loader's calibration phase to keep VRAM flat: reclaim each
    /// throwaway case's K/V as it retires rather than letting it accumulate hot.
    /// Idempotent — a turn already demoted (or not yet warm) is skipped.
    ///
    /// `flush` selects the mode:
    /// - `true` (boundary sweep): first drain the hot→warm migration **on this
    ///   thread** so the whole tail is warm-backed (hence demotable), then issue
    ///   the demote and **block** until it completes — the caller needs the VRAM
    ///   reclaimed before the next phase prefills. The flush is done here rather
    ///   than inside the scheduler handler so its (≤30 s) wait can't stall the
    ///   scheduler's decode/prefill loop. Returns the number of residences
    ///   demoted.
    /// - `false` (incremental sweep): **fire-and-forget** — issue the demote and
    ///   return immediately without blocking the caller (case submission must not
    ///   stall). Only already-warm-backed turns are dropped; any not-yet-warm
    ///   tail is caught by the next sweep or the boundary flush. Returns `0`.
    pub fn demote_timelines_hot(
        &self,
        timelines: &[TimelineId],
        flush: bool,
    ) -> crate::Result<usize> {
        if flush {
            self.persist_thread
                .trigger_handle()
                .flush_blocking(std::time::Duration::from_secs(30));
        }
        if timelines.is_empty() {
            return Ok(0);
        }
        let (response_tx, response_rx) = channel::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::DemoteTimelinesHot {
                conversation: self.conversation.clone(),
                timelines: timelines.to_vec(),
                response_tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        if flush {
            // Boundary: wait for the demote to complete (VRAM must be reclaimed
            // before the next phase).
            let demoted = response_rx
                .recv()
                .map_err(|_| ConversationError::SchedulerGone)??;
            Ok(demoted)
        } else {
            // Incremental: fire-and-forget. Dropping `response_rx` makes the
            // handler's reply a silent no-op; the demote still runs.
            drop(response_rx);
            Ok(0)
        }
    }

    /// Whether `timeline` still has KV content (not yet reclaimed by a distill
    /// compaction). Gate distill-marking on this to keep it idempotent and avoid
    /// looping compaction.
    pub fn timeline_has_kv(&self, timeline: TimelineId) -> bool {
        self.conversation.timeline_has_kv(timeline)
    }

    /// Fully evict a **completed ingest** `timeline`'s KV from VRAM + RAM once
    /// it is durable on disk: flag every turn residence `evict_when_cold` and
    /// wake the persistence thread so the hot→warm→cold pipeline runs promptly.
    /// As each turn migrates, its VRAM is freed at warm-land and its RAM copy at
    /// cold-land, leaving it cold-only on NVMe — `elevate_to_hot` pulls it back
    /// on demand if a later projection re-selects it. Returns the number of turn
    /// residences flagged.
    ///
    /// Unlike [`Self::demote_timelines_hot`] (hot→warm, keeps the warm RAM
    /// copy), this reclaims **both** resident tiers: a completed code_read file
    /// is not attended again until retrieval, so keeping it warm only wastes RAM
    /// and PCIe migration bandwidth. Fire-and-forget — the actual frees happen
    /// on the persistence thread as durability lands.
    pub fn evict_ingest_timeline(&self, timeline: TimelineId) -> usize {
        let flagged = self.conversation.mark_timeline_evict_when_cold(timeline);
        if flagged > 0 {
            // Wake the persistence thread so the flagged turns migrate → persist
            // → evict now, rather than waiting for the next periodic tick.
            self.persist_thread.trigger();
        }
        flagged
    }

    /// The segmented redo log's maintenance state — `(segment_count, last_op)`,
    /// where `last_op` is `(label, unix_secs)` — for the daemon status / GUI
    /// compaction indicator.
    pub fn substrate_maintenance_status(&self) -> (usize, Option<(String, u64)>, bool) {
        self.conversation.maintenance_status()
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

    /// Compile a tool catalog into a [`TriggerRegistry`] for constrained
    /// tool-call decoding.  Pass the returned registry to a turn via
    /// [`TurnOptions::triggers`](crate::TurnOptions::triggers); a turn without it
    /// (or with an empty registry) free-decodes as usual.
    ///
    /// The model emits the `<tool_call>` trigger token freely; the stencil then
    /// forces the catalog's exact shape — name ∈ catalog, required params in
    /// order, enum values exact, structurally-valid JSON — for the rest of the
    /// call.  Compile once and reuse the registry across turns.
    ///
    /// If the tokenizer has no single `<tool_call>` token (the marker tokenizes
    /// to several pieces), an **empty** registry is returned — constrained
    /// decoding is simply inactive and the model free-decodes tool calls as
    /// before.  This never fails startup over a tokenizer mismatch.
    pub fn compile_tool_stencil(&self, tools: &[ToolSpec]) -> crate::Result<Arc<TriggerRegistry>> {
        let Some(trigger) = self.tokenizer.token_to_id("<tool_call>") else {
            tracing::warn!(
                "tokenizer has no single <tool_call> token — tool-call stencils are inactive \
                 (the model will free-decode tool calls)"
            );
            return Ok(Arc::new(TriggerRegistry::new()));
        };
        // The model emits the `<tool_call>` trigger itself, so the tree resumes
        // *after* that marker: its `open` is the envelope minus the marker.
        //
        // The close ends with the assistant-turn EOS (`<|im_end|>`): a tool call
        // is the entire assistant turn, so once it is emitted the turn must end.
        // Without this the stencil releases control after `</tool_call>` and the
        // model free-decodes a hallucinated answer past the call. The decode
        // loop detects the EOS in the injected close run and seals the turn.
        let envelope = ToolCallEnvelope {
            open: "\n{\"name\": \"".to_string(),
            args_open: ", \"arguments\": {".to_string(),
            close: "}}\n</tool_call><|im_end|>".to_string(),
        };
        let spec = compile_tool_call_tree(tools, &envelope).map_err(|e| {
            ConversationError::from(candle::Error::Msg(format!("tool stencil: {e}")))
        })?;
        let eos = self.config.eos_tokens.iter().next().copied().unwrap_or(0);
        let vocab = HfVocab::new(
            (*self.tokenizer).clone(),
            eos,
            self.config.vocab_size as u64,
        );
        let tree = compile(&spec, &vocab).map_err(|e| {
            ConversationError::from(candle::Error::Msg(format!("tool stencil: {e}")))
        })?;
        let mut registry = TriggerRegistry::new();
        registry.register(trigger, Arc::new(tree));
        Ok(Arc::new(registry))
    }

    /// Compile the thinking-block steering trees (one per non-`Off` effort dial)
    /// once, for reuse across turns via [`ThinkSteering::registry_for`].  Like the
    /// tool stencil, this is inactive — `Ok(None)` — when the tokenizer lacks a
    /// single `<think>`/`</think>` token, so the model free-decodes its reasoning.
    pub fn compile_think_steering(&self) -> crate::Result<Option<Arc<ThinkSteering>>> {
        let (Some(think_open), Some(think_close)) = (
            self.tokenizer.token_to_id("<think>"),
            self.tokenizer.token_to_id("</think>"),
        ) else {
            tracing::warn!(
                "tokenizer has no single <think>/</think> token — think steering is inactive \
                 (the model will free-decode its reasoning)"
            );
            return Ok(None);
        };
        let eos = self.config.eos_tokens.iter().next().copied().unwrap_or(0);
        let env = ThinkSteerEnvelope {
            think_open,
            think_close,
            eos,
        };
        let vocab = HfVocab::new(
            (*self.tokenizer).clone(),
            eos,
            self.config.vocab_size as u64,
        );
        let compile_mode = |mode: ThinkMode| -> crate::Result<Arc<StencilTree>> {
            let spec = compile_think_tree(mode, &env)
                .expect("a non-Off mode always yields a steering spec");
            let tree = compile(&spec, &vocab).map_err(|e| {
                ConversationError::from(candle::Error::Msg(format!("think stencil: {e}")))
            })?;
            Ok(Arc::new(tree))
        };
        Ok(Some(Arc::new(ThinkSteering {
            think_open,
            quick: compile_mode(ThinkMode::Quick)?,
            balanced: compile_mode(ThinkMode::Balanced)?,
            deep: compile_mode(ThinkMode::Deep)?,
            exhaustive: compile_mode(ThinkMode::Exhaustive)?,
        })))
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
        // Register this conversation's per-conversation KV-compression
        // override (if any) before the first turn seals, so each turn
        // residence inherits it at alloc time. Utility layers set a
        // compression level (and may drop the K override or pin forced K/V
        // formats) via their SequenceConfig.
        let compression = if config.kv_compression_level.is_some()
            || config.kv_force_k_format.is_some()
            || config.kv_force_v_format.is_some()
            || config.kv_lossless
        {
            Some(ConvCompression {
                lossless: config.kv_lossless,
                level: config.kv_compression_level,
                disable_k_override: config.kv_disable_k_override,
                force_k: config.kv_force_k_format,
                force_v: config.kv_force_v_format,
            })
        } else {
            None
        };
        self.conversation
            .set_timeline_compression(timeline, compression);
        // Every layer summarises into its AVL summary tree; provenance scans then
        // expand the compressed nodes on retrieval. This is independent of
        // `disable_reprojection` — that flag only gates the per-turn reprojection
        // for append-only utility layers. The AVL summariser runs on its own
        // thread (wave-driven compression) and never blocks ingest, so even
        // high-turn-count utility layers can summarise.
        self.conversation
            .set_timeline_summarize(timeline, !self.config.disable_summariser);
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
                // A fresh conversation: any state comes from the timeline's own
                // snapshot, which `create_sequence` reads.
                parent: None,
                response_tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;

        let sequence_id = response_rx
            .recv()
            .map_err(|_| ConversationError::SchedulerGone)??;

        let conv = Sequence::new_with_projection(
            self.scheduler_tx.clone(),
            sequence_id,
            Arc::clone(&self.tokenizer),
            system_prompt,
            builder,
            target,
            config,
            CHUNK_SIZE,
            self.model_core,
            self.conversation.clone(),
            section_progress,
            // Single-create path keeps the first-turn priming optimization.
            true,
        )?;

        Ok(conv)
    }

    /// Batch-create `n` conversations that share one projection (system prompt,
    /// builder, layer/group, config), **pipelining** the `NewSequence` slot
    /// allocations: every request is fired before any response is awaited, so
    /// the scheduler drains the whole batch in a single cycle and it costs ~one
    /// round-trip instead of `n` serial ones.
    ///
    /// [`Self::new_conversation_with_projection`] blocks on its slot-alloc reply,
    /// and the scheduler interleaves those replies between forward waves — so
    /// creating a window of cases one at a time pays ~one wave-latency per case.
    /// During calibration that starved the wave-batched prefill to 2–4 sequences
    /// wide (poor MoE expert amortization). Firing the window's allocations up
    /// front lets the cases prefill together in one wide forward instead.
    ///
    /// Each conversation gets a fresh timeline; per-conversation compression and
    /// summariser settings mirror the single-create path. Returns one `Result`
    /// per requested conversation, in submission order.
    pub fn new_conversations_with_projection_batch(
        &self,
        n: usize,
        system_prompt: &str,
        builder: &Builder,
        layer: LayerId,
        group: GroupId,
        config: &SequenceConfig,
    ) -> Vec<crate::Result<Sequence>> {
        if n == 0 {
            return Vec::new();
        }
        if let Some(yaml) = builder.source_yaml() {
            if let Err(e) = self.conversation.set_template(yaml.as_bytes()) {
                tracing::warn!("persist projection template failed: {e}");
            }
        }
        let compression = if config.kv_compression_level.is_some()
            || config.kv_force_k_format.is_some()
            || config.kv_force_v_format.is_some()
            || config.kv_lossless
        {
            Some(ConvCompression {
                lossless: config.kv_lossless,
                level: config.kv_compression_level,
                disable_k_override: config.kv_disable_k_override,
                force_k: config.kv_force_k_format,
                force_v: config.kv_force_v_format,
            })
        } else {
            None
        };

        // Phase 1 — mint a timeline and fire `NewSequence` for every case
        // WITHOUT awaiting, so all `n` requests sit in the scheduler queue
        // together and one drain cycle allocates every slot.
        struct Fired {
            target: ProjectionTarget,
            rx: channel::Receiver<crate::Result<SequenceId>>,
        }
        let mut fired: Vec<crate::Result<Fired>> = Vec::with_capacity(n);
        for _ in 0..n {
            let timeline = self.conversation.mint_timeline(layer, group);
            let target = ProjectionTarget {
                layer,
                group,
                timeline,
            };
            let (response_tx, rx) = channel::bounded(1);
            match self.scheduler_tx.send(SchedulerRequest::NewSequence {
                conversation: self.conversation.clone(),
                target: Some(target),
                // Resume by timeline: the snapshot read in `create_sequence` is
                // the whole of the state recovery here — there is no live
                // parent to copy from.
                parent: None,
                response_tx,
            }) {
                Ok(()) => {
                    // Only register per-timeline metadata for a conversation that
                    // actually exists — set it after the send succeeds (still
                    // before the scheduler drains the request or any turn seals,
                    // so residence compression inheritance is unaffected). A
                    // failed send leaves only the bare timeline mint, no metadata.
                    self.conversation
                        .set_timeline_compression(timeline, compression);
                    self.conversation
                        .set_timeline_summarize(timeline, !self.config.disable_summariser);
                    fired.push(Ok(Fired { target, rx }));
                }
                Err(_) => fired.push(Err(ConversationError::SchedulerGone)),
            }
        }

        // Phase 2 — collect each slot id (already queued, so no extra wave wait)
        // and build its Sequence. After the warm-up case has pinned the shared
        // sections, `new_with_projection` only re-references already-hot sections
        // here, so it issues no further scheduler round-trip.
        fired
            .into_iter()
            .map(|f| {
                let f = f?;
                let sequence_id =
                    f.rx.recv()
                        .map_err(|_| ConversationError::SchedulerGone)??;
                Sequence::new_with_projection(
                    self.scheduler_tx.clone(),
                    sequence_id,
                    Arc::clone(&self.tokenizer),
                    system_prompt,
                    builder.clone(),
                    f.target,
                    config.clone(),
                    CHUNK_SIZE,
                    self.model_core,
                    self.conversation.clone(),
                    None,
                    // Pipelined batch: skip the per-sequence priming round-trip
                    // that would otherwise serialise the burst — `apply_projection`
                    // at first submit materialises the projection instead.
                    false,
                )
            })
            .collect()
    }

    /// Get the shared tokenizer.
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
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
                parent: None,
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
                user_content_start: 0,
                user_content_end: 0,
                assistant_content_start: 0,
                no_think: false,
                tags: Vec::new(),
                projection_offsets: Vec::new(),
                prefill_assistant_text: String::new(),
                post_decode_tokens: TokenBuffer::new(),
                max_decode_tokens,
                sampling: SamplingConfig::argmax(),
                event_tx,
                reprojection: None,
                disable_reprojection: false,
                // Raw eval/summarisation path: no tools, no constrained decode.
                triggers: Arc::new(TriggerRegistry::new()),
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

    /// Force a full redo-log compaction (operator opt-in via the startup
    /// flag). Rewrites the log to just the live record set, reclaiming the
    /// dead weight that accrues from superseded turns and tombstoned
    /// timelines. The persistence thread also compacts automatically when
    /// the dead-byte ratio crosses the threshold. `progress` reports coarse
    /// phase progress (0..=5) for the loading screen.
    pub fn compact_substrate(&self, progress: Option<&dyn Fn(usize, usize)>) -> crate::Result<()> {
        Ok(self.conversation.compact_substrate(progress)?)
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
        // Terminal durability step: every enqueuer has now stopped (the
        // scheduler's seal path, the persistence thread's final hot→warm→cold
        // drain, and the summariser), so drain the off-thread writer's queue to
        // the redo log, fsync, and join it. This MUST run here because the daemon
        // force-exits (`std::process::exit`), which skips the writer's `Drop` —
        // without this flush, every warm→cold KV / tokens / sig append still in
        // the writer's queue at exit would be silently lost.
        self.conversation.flush_writer();
        Ok(())
    }
}

impl Drop for ConversationEngine {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
