//! The conversation engine: entry point, spawns the scheduler thread.

use crate::config::{EngineConfig, SamplingConfig, SequenceConfig};
use crate::conversation::{install_branch_states, PendingBranchState, Sequence};
use crate::error::ConversationError;
use crate::handle::{TokenDecoder, TurnEvent};
use crate::persistence::record::DistillMode;
use crate::persistence::thread::PersistenceThread;
use crate::persistence::SharedSubstrate;
use crate::projection::{
    Builder, CollectionWarm, Conversation, GroupId, LayerId, PlainPromptFrames, ProjectionTarget,
    Reserved, Schema, SectionId, TimelineId, TurnIndex,
};
use crate::scheduler::{Scheduler, SchedulerRequest};
use crate::sequence_handle::SequenceId;
use crate::stencil::{
    compile, compile_think_tree, compile_tool_call_loop, HfVocab, StencilTree, ThinkMode,
    ThinkSteerEnvelope, TokenId, ToolCallEnvelope, ToolSpec, TriggerRegistry,
    MAX_TOOL_CALLS_PER_TURN,
};
// `ChannelProbeRunner` is deliberately not imported: the summariser is
// disconnected, so nothing constructs a runner. `Substrate` comes from our side.
use crate::substrate::ConvCompression;
use crate::summary_tree::{SelectionDiagnostics, SummariserThread};
use crate::token_buffer::TokenBuffer;
use crate::turn_text::literal_tokenizer;

use candle_nn::CHUNK_SIZE;
use candle_transformers::models::batched_inference::{ManagedBatchedModel, ModelCoreProperties};
use flume::{Receiver, Sender};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// The compiled thinking-block steering trees, one per effort dial, built once
/// at engine init (parallel to the tool-call registry).  Each turn derives its
/// trigger registry by binding the `<think>` trigger to the dial's tree, once —
/// see [`TriggerRegistry::with_once_trigger`].
pub struct ThinkSteering {
    /// `<think>` id — the trigger the dial's tree is bound to.
    think_open: TokenId,
    /// Empties the block the moment it opens — see [`crate::stencil::think`].
    off: Arc<StencilTree>,
    quick: Arc<StencilTree>,
    balanced: Arc<StencilTree>,
    deep: Arc<StencilTree>,
    exhaustive: Arc<StencilTree>,
}

impl ThinkSteering {
    /// Derive a per-turn registry from `base` (e.g. the tool-call catalog) for
    /// `mode`: bind the `<think>` trigger to that dial's steering tree.
    /// The base is untouched; the result is a fresh registry.
    ///
    /// **[`ThinkMode::Off`] binds a tree too**, and this is the correction that
    /// matters. It used to *clear* the trigger on the reasoning that the
    /// `/no_think` glue would yield an empty block — true for Qwen3, false for
    /// Qwen3.5 and Qwen3.8, which have no such marker at all. On those families
    /// clearing the trigger left the model free to reason with nothing steering
    /// it: the block ran to the token ceiling and the whole decode was discarded,
    /// while the dial reported itself off.
    ///
    /// Binding `off` instead makes suppression a property of the grammar rather
    /// than of the family's chat template, so it holds for every checkpoint and
    /// a caller does not have to know which mechanism its model happens to use.
    ///
    /// **A registry has to actually be bound for any of that to hold.** A caller
    /// that passes `TurnOptions::default()` gets an empty one and no steering at
    /// all — which is how the ingest summariser came to reason unchecked despite
    /// its dial reading `Off`. See `Sequence::no_think_triggers`.
    ///
    /// **The tree fires once per turn.** It steers the block that opens the
    /// turn; a `<think>` the model writes after that is text and decodes freely
    /// (see [`TriggerRegistry::with_once_trigger`]). The base's triggers — the
    /// tool call — stay armed for every call the turn makes.
    pub fn registry_for(&self, base: &TriggerRegistry, mode: ThinkMode) -> Arc<TriggerRegistry> {
        let tree = match mode {
            ThinkMode::Off => &self.off,
            ThinkMode::Quick => &self.quick,
            ThinkMode::Balanced => &self.balanced,
            ThinkMode::Deep => &self.deep,
            ThinkMode::Exhaustive => &self.exhaustive,
        };
        Arc::new(base.with_once_trigger(self.think_open, Arc::clone(tree)))
    }
}

#[cfg(test)]
mod think_steering_tests {
    use super::*;
    use crate::stencil::{TestVocab, Vocab};

    const THINK: TokenId = 151667;
    const THINK_CLOSE: TokenId = 151668;
    const TOOL_CALL: TokenId = 151657;

    fn steering() -> ThinkSteering {
        let v = TestVocab::new()
            .with_special("<think>", THINK)
            .with_special("</think>", THINK_CLOSE);
        let env = ThinkSteerEnvelope {
            think_open: THINK,
            think_close: THINK_CLOSE,
            eos: v.eos(),
            after_close: "",
        };
        let tree =
            |mode| Arc::new(compile(&compile_think_tree(mode, &env), &v).expect("think tree"));
        ThinkSteering {
            think_open: THINK,
            off: tree(ThinkMode::Off),
            quick: tree(ThinkMode::Quick),
            balanced: tree(ThinkMode::Balanced),
            deep: tree(ThinkMode::Deep),
            exhaustive: tree(ThinkMode::Exhaustive),
        }
    }

    /// The turn's think block is steered once; after it, `<think>` is text and
    /// the tool-call trigger is still armed.
    #[test]
    fn the_think_block_fires_once_and_the_call_trigger_stays() {
        let s = steering();
        let tools = TriggerRegistry::new().with_trigger(TOOL_CALL, Arc::clone(&s.off));
        for mode in [ThinkMode::Off, ThinkMode::Balanced, ThinkMode::Exhaustive] {
            let turn = s.registry_for(&tools, mode);
            assert!(
                turn.driver_for(THINK).is_some(),
                "{mode:?}: the opening block"
            );
            let rest = turn
                .after_firing(THINK)
                .unwrap_or_else(|| panic!("{mode:?}: the think trigger was not spent"));
            assert!(
                rest.driver_for(THINK).is_none(),
                "{mode:?}: a later <think> is text"
            );
            assert!(
                rest.driver_for(TOOL_CALL).is_some(),
                "{mode:?}: calls stay armed"
            );
        }
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

/// Whether a slot is being opened onto a conversation that already has turns.
///
/// A `bool` would read as `open(…, true)` at the two call sites that pick it,
/// and the two cases differ in what they *mean* rather than in a setting: one
/// mints a timeline and one continues one. See
/// [`ConversationEngine::resume_conversation_with_projection`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Resumed {
    No,
    Yes,
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
    scheduler_tx: Sender<SchedulerRequest>,

    /// Handle to the scheduler thread (joined on drop or on explicit shutdown).
    /// Wrapped in `Mutex<Option>` so `shutdown()` can be called via `&self`,
    /// enabling clean teardown from a `&'static ConversationEngine` reference
    /// (e.g. from a thread-local drop guard in tests).
    scheduler_handle: Mutex<Option<JoinHandle<()>>>,

    /// Tokenizer (shared, immutable, safe to clone into conversations).
    tokenizer: Arc<tokenizers::Tokenizer>,

    /// The frame sections handed out to plain-prompt conversations, and the
    /// claim on each — see [`Self::plain_prompt_section`].
    plain_frames: Mutex<PlainPromptFrames>,
    /// [`Self::tokenizer`] reading every chat tag as plain characters, for the
    /// literal pieces of a turn — see [`crate::turn_text`]. Built once here: it
    /// is a copy of the whole vocabulary.
    literal_tokenizer: Arc<tokenizers::Tokenizer>,

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

    /// Async summariser thread — **always the disabled handle.** It would drain
    /// the per-turn pending queue, run §6 probes and build the per-timeline AVL
    /// summary tree; none of that happens, because the summariser is
    /// disconnected (see [`Self::new`]) and no timeline enqueues. Kept as a field
    /// so the lifecycle calls below stay honest no-ops rather than disappearing.
    /// Mirrors [`PersistenceThread`]'s lifecycle (trigger / tick /
    /// shutdown).  Spawned alongside the scheduler at engine startup;
    /// [`Self::shutdown`] joins it after the persistence thread has
    /// drained, and `Drop` falls through to the same path.
    summariser_thread: SummariserThread,

    /// The co-resident models that borrow the card between waves.
    ///
    /// Shared with the scheduler, which polls the queue once per pass and
    /// drains it between forwards. Held here because the *registry* is a
    /// deployment decision made before the engine starts, and the *queue* is
    /// what callers submit to from any thread — see [`Self::submit_guest`] and
    /// [`crate::guest`].
    guests: Arc<crate::guest::Guests>,

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
    /// * `guests` — The co-resident models this deployment has configured, if
    ///   any. [`GuestRegistry::new`] is the ordinary answer and costs one atomic
    ///   load per scheduler pass; see [`crate::guest`] for what a populated one
    ///   buys. A parameter rather than a field on `EngineConfig` because the
    ///   registry holds constructors — it is not `Clone`, and the config is.
    pub fn new(
        model: Box<dyn ManagedBatchedModel + Send>,
        tokenizer: tokenizers::Tokenizer,
        mut config: EngineConfig,
        guests: crate::guest::GuestRegistry,
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

        let guests = Arc::new(crate::guest::Guests {
            queue: Arc::new(crate::guest::GuestQueue::new()),
            registry: guests,
        });
        let guests_for_scheduler = Arc::clone(&guests);

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
        let (tx, rx) = flume::unbounded();

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
        //
        // A host that writes its own record classes into this same log opens it
        // first and hands the open pair over ([`SharedSubstrate`]) — one
        // `.substrate/` admits exactly one writable handle per process. Everyone
        // else names a directory and the engine opens it here. Resolved once,
        // into the same pair either way, so nothing downstream knows which.
        let open_start = std::time::Instant::now();
        let adopted = config.substrate.is_some();
        let shared = match config.substrate.clone() {
            Some(shared) => shared,
            None => {
                let workspace_dir: std::path::PathBuf = match config.workspace_path.as_ref() {
                    Some(p) => AsRef::<std::path::Path>::as_ref(p).to_path_buf(),
                    None => {
                        std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
                    }
                };
                // A read-only open writes nothing under `.substrate/` and
                // requires the store to exist — see
                // `EngineConfig::read_only_substrate`.
                let opened = if config.read_only_substrate {
                    SharedSubstrate::open_in_read_only(&workspace_dir)
                } else {
                    SharedSubstrate::open_in(&workspace_dir)
                };
                opened.map_err(|e| {
                    ConversationError::from(candle::Error::Msg(format!(
                        "substrate persistence: {e}"
                    )))
                })?
            }
        };
        {
            let persistence = shared.persistence.lock().unwrap_or_else(|e| e.into_inner());
            tracing::info!(
                open_ms = open_start.elapsed().as_millis() as u64,
                adopted,
                read_only = persistence.is_read_only(),
                log_bytes = persistence.write_offset(),
                records = persistence.recovered_record_count(),
                indexed = persistence.last_index().is_some(),
                streams = shared
                    .substrate
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .all_streams()
                    .count(),
                "substrate persistence opened"
            );
        }
        // Persist the model identity into the substrate's `ModelSpec` record —
        // compare-and-insert, so it only appends when the model differs from
        // what the log already records. Makes the log a self-contained image.
        // A read-only substrate appends neither record: `set_model_spec`
        // reports nothing written, and `set_tokenizer` still refuses a
        // tokenizer other than the one the log records.
        let singletons_start = std::time::Instant::now();
        let mut persistence = shared.persistence.lock().unwrap_or_else(|e| e.into_inner());
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
        // Bind the substrate to this model's tokenizer.json, so the log can
        // detokenize offline and so no second vocabulary can ever be written
        // over turns sealed under the first. ~11 MB for Qwen3, written at most
        // once per substrate; identical bytes are a no-op, and different bytes
        // are refused rather than appended.
        if let Some(tok) = &config.tokenizer {
            let wrote = persistence.set_tokenizer(tok).map_err(|e| {
                ConversationError::Tokenizer(format!(
                    "{e}\n\nThe substrate at this working directory belongs to a different \
                     model. Start this one against its own working directory (--working-dir), \
                     or discard this substrate with --wipe-substrate."
                ))
            })?;
            if wrote {
                persistence.commit().map_err(|e| {
                    ConversationError::from(candle::Error::Msg(format!("commit tokenizer: {e}")))
                })?;
            }
        }
        drop(persistence);
        tracing::info!(
            singletons_ms = singletons_start.elapsed().as_millis() as u64,
            "model spec + tokenizer records reconciled"
        );
        let conversation = Conversation::from_shared(shared);

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

        // **The AVL summariser is disconnected.** It is never spawned, and no
        // timeline enqueues turns for it (`Timeline::summarize` is false for
        // every timeline) — so nothing in this engine compresses a conversation
        // or a layer into summary nodes.
        //
        // The decision, deliberately: compression was a persistent source of bad
        // memory rather than a saving. Measured on a 16-turn conversation, 5 of 9
        // summary nodes were unfaithful — two echoed the user's question back,
        // one echoed the compressor's own instruction, and the merge node that
        // stands for the WHOLE conversation read "I am an AI assistant." Those
        // nodes are written in the first person, as if they were the reply, and
        // are what a later projection reads as history: a wrong one is not a
        // missing summary but a false memory the model cannot distinguish from
        // something it actually said. Retrieval quality is being pursued through
        // provenance selection instead, which ranks real turns rather than
        // manufacturing new text.
        //
        // `summary_tree` stays compiled and tested so the machinery — the AVL
        // shape, the probe protocol, the seal path — is here to build on when
        // that work resumes. Nothing calls into it.
        let summariser_thread = SummariserThread::disabled();
        // The scheduler still holds a trigger and still fires it on every
        // assistant-turn seal (design §4 step ③). Against the disabled handle
        // the send has no receiver and fails silently, which is why the seal
        // path needs no knowledge of whether a summariser exists — and why
        // re-enabling is a change in `Engine::new` alone.
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
                    guests_for_scheduler,
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
            literal_tokenizer: Arc::new(literal_tokenizer(&tokenizer)),
            tokenizer: Arc::new(tokenizer),
            plain_frames: Mutex::new(PlainPromptFrames::default()),
            config,
            model_core,
            conversation,
            persist_thread,
            summariser_thread,
            guests,
            substrate_reload_status,
        })
    }

    /// Queue work for a co-resident model and get a handle on its answer.
    ///
    /// The scheduler picks it up between two of its waves, evicts what it needs
    /// to make room, serves the whole backlog for that guest, and hands the
    /// ground back. Normal inference is blocked for the length of that drain —
    /// see [`crate::guest`] for why that is the design rather than a cost.
    ///
    /// Refuses synchronously — before anything is queued and before anything is
    /// evicted — when the request itself is not servable. A caller can block on
    /// [`GuestReceipt::wait`] or poll `try_take`.
    pub fn submit_guest(
        &self,
        request: crate::guest::GuestRequest,
    ) -> Result<crate::guest::GuestReceipt, crate::guest::GuestError> {
        self.submit_guest_watched(request, crate::guest::GuestSink::none())
    }

    /// The same, reporting progress to `sink` while the job runs.
    ///
    /// The sink is called on the scheduler thread with normal inference blocked,
    /// so it must not block — see [`crate::guest::progress`].
    pub fn submit_guest_watched(
        &self,
        request: crate::guest::GuestRequest,
        sink: crate::guest::GuestSink,
    ) -> Result<crate::guest::GuestReceipt, crate::guest::GuestError> {
        let receipt = self.guests.queue.submit_watched(request, sink)?;
        // **Wake the scheduler.** It parks in `rx.recv()` when there is nothing
        // to do, and the guest queue is the one producer that does not arrive
        // through that channel — so on an idle daemon a submitted job sat in
        // the queue indefinitely, below a loop that was never going to come
        // back round to look at it.
        //
        // Sent *after* the job is queued, so the loop either sees the work on
        // its own next pass or is woken to find it. There is no ordering in
        // which both miss: the channel buffers, so a wake that arrives while
        // the loop is still running is waiting for it when it parks.
        let _ = self.scheduler_tx.send(SchedulerRequest::Wake);
        Ok(receipt)
    }

    /// The guests this deployment configured, for a status view.
    pub fn configured_guests(&self) -> Vec<crate::guest::Guest> {
        self.guests.registry.configured()
    }

    /// How many guest jobs are waiting.
    pub fn guest_backlog(&self) -> usize {
        self.guests.queue.depth()
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
        let (tx, rx) = flume::bounded(1);
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
    /// `timeline`. **Always zero**: the summariser is disconnected, so nothing
    /// enqueues (see [`Self::new`]). It was zero in steady state before, too, so
    /// this reads the same either way.
    pub fn pending_summary_len(&self, timeline: TimelineId) -> usize {
        self.conversation.pending_summary_len(timeline)
    }

    /// Wake the summariser thread now instead of waiting for its next tick.
    ///
    /// **A no-op while the summariser is disconnected** — the disabled handle has
    /// no receiver, so the send fails silently. Callers (the ingest pipeline
    /// kicks it after a scope lands) need no knowledge of that.
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

    /// Enable or disable AVL summarisation for `timeline`.
    ///
    /// **Every timeline now defaults to `false`** — the summariser is
    /// disconnected (see [`Self::new`]), so setting `true` here would queue turns
    /// onto `pending_summary_queue` that nothing drains. Nothing in the engine or
    /// `zend` calls it with `true`; the remaining production callers pass `false`
    /// on ingest timelines, which is redundant against the default but states the
    /// intent at the site.
    ///
    /// It exists as the single re-enabling point: deciding *which* timelines opt
    /// in is this call plus spawning the thread in [`Self::new`]. See
    /// [`crate::summary_tree`].
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
    pub fn warm_ingest_normalization(&self, schema: &Schema) {
        self.conversation.warm_ingest_normalization(schema);
    }

    /// Warm one belief group's per-timeline hit levels by self-match — for a
    /// group nothing else teaches, such as a tag-scoped turn group, which
    /// learns only from probes inside its scope. Returns how many timelines
    /// were warmed. See
    /// [`crate::projection::Conversation::warm_group_normalization`].
    pub fn warm_group_normalization(
        &self,
        schema: &crate::projection::Schema,
        group: GroupId,
    ) -> usize {
        self.conversation.warm_group_normalization(schema, group)
    }

    /// Warm one timeline's hit levels by self-match — a conversation written
    /// onto a belief group after the group was warmed. See
    /// [`crate::projection::Conversation::warm_timeline_normalization`].
    pub fn warm_timeline_normalization(
        &self,
        schema: &crate::projection::Schema,
        timeline: TimelineId,
    ) -> bool {
        self.conversation
            .warm_timeline_normalization(schema, timeline)
    }

    /// Warm the belief-driven section collections' per-member hit levels from
    /// their own tag-scoped corpus. Call after load, once the corpus is stable —
    /// without it a collection's levels are cold on every process start but the
    /// one that built the corpus, which changes both the scale and the RANKING of
    /// its scores. Runs on the scheduler thread, where the GPU gallery arena
    /// scores the corpus in batch, and blocks until it is done. See
    /// [`Conversation::warm_collection_normalization`].
    pub fn warm_collection_normalization(&self, schema: &Schema) -> CollectionWarm {
        let (tx, rx) = flume::bounded(1);
        if self
            .scheduler_tx
            .send(SchedulerRequest::WarmCollectionNormalization {
                conversation: self.conversation.clone(),
                schema: Box::new(schema.clone()),
                response_tx: tx,
            })
            .is_err()
        {
            return CollectionWarm::default();
        }
        rx.recv().unwrap_or_default()
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

    /// Merge several `(key, value)` pairs into `timeline`'s `custom` metadata in
    /// one persisted record, so values that describe one decision — a turn's
    /// composer dials — are never recovered half-written.
    pub fn set_conversation_metadata_many(
        &self,
        timeline: TimelineId,
        kv: &std::collections::BTreeMap<String, String>,
    ) -> crate::Result<()> {
        self.conversation
            .set_conversation_metadata_many(timeline, kv)
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

    /// Every live conversation, whether or not it carries a `conv_id` —
    /// tombstoned ones excluded.
    ///
    /// [`Self::known_conversations`] lists only the named ones, which is what a
    /// sidebar wants and not what a caller retiring everything wants: an
    /// ingested document is a conversation with no `conv_id` at all.
    pub fn live_conversations(&self) -> Vec<TimelineId> {
        self.conversation.live_timeline_ids()
    }

    /// Live conversations whose `conv_id` starts with `prefix`, as
    /// `(timeline, conv_id)` — tombstoned ones excluded.
    ///
    /// For a caller that names its conversations by a scheme rather than
    /// listing them, and wants one scheme's members without paying for the
    /// whole registry: [`Self::known_conversations`] materialises every
    /// conversation the workspace has ever held, and that grows for the life of
    /// the log while the live set stays bounded. `npcd` finds one character's
    /// `npc-<id>-day-*` conversations this way.
    pub fn conversations_with_conv_id_prefix(&self, prefix: &str) -> Vec<(TimelineId, String)> {
        self.conversation.conversations_with_conv_id_prefix(prefix)
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

    /// Mark `timeline` as scratch: its turns never reach cold storage.
    ///
    /// Sets `no_cold_persist` on every residence the timeline holds, so a
    /// conversation opened for one question and thrown away costs GPU and RAM
    /// for as long as it runs and nothing on disk afterwards. The alternative —
    /// writing it and tombstoning it — pays the whole write and then asks
    /// compaction to take it back.
    ///
    /// Call it immediately after minting, before the first turn seals: a turn
    /// that has already been written is already on the cold path and this does
    /// not retract it.
    pub fn mark_timeline_transient(&self, timeline: TimelineId) {
        self.conversation.mark_timeline_transient(timeline);
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

    /// Tombstone one turn of a live timeline — see
    /// [`crate::projection::Conversation::tombstone_turn`].
    pub fn tombstone_turn(&self, timeline: TimelineId, turn_index: u32) -> crate::Result<()> {
        self.conversation
            .tombstone_turn(timeline, turn_index)
            .map_err(ConversationError::Model)
    }

    /// Whether `(timeline, turn)` was already dropped by a turn-scoped
    /// tombstone.
    pub fn is_turn_tombstoned(&self, timeline: TimelineId, turn_index: u32) -> bool {
        self.conversation.is_turn_tombstoned(timeline, turn_index)
    }

    /// Whether the whole of `timeline` has been tombstoned.
    ///
    /// The companion to [`Self::is_turn_tombstoned`], and what a caller about
    /// to retire a conversation consults so an already-dead one costs a lookup
    /// rather than another record.
    pub fn is_timeline_tombstoned(&self, timeline: TimelineId) -> bool {
        self.conversation.is_timeline_tombstoned(timeline)
    }

    /// How many turns `timeline` actually holds — the substrate's own count.
    ///
    /// **The number a retention sweep must reckon against.** `Sequence::turn_count`
    /// is a different quantity: it is a per-sequence diagnostic that counts a
    /// user and an assistant message separately (`+= 2` an exchange) and resets
    /// whenever a process opens a fresh sequence. Turn *indices* advance once
    /// per exchange, so a horizon computed from that counter runs at twice the
    /// rate of the turns it indexes — past ~65 exchanges it exceeds the last
    /// real turn and retires the entire conversation, which is `keep_turns`
    /// meaning its own opposite.
    ///
    /// This counts turns the way the index does, so a horizon derived from it
    /// names a turn that exists.
    pub fn timeline_turn_count(&self, timeline: TimelineId) -> u64 {
        self.conversation.read().turn_count(timeline) as u64
    }

    /// Every live conversation written to `group` — the set selection reads,
    /// so archived and tombstoned conversations are not in it.
    pub fn group_conversations(&self, group: GroupId) -> Vec<TimelineId> {
        self.conversation
            .read()
            .active_timelines_for_group(group)
            .collect()
    }

    /// Both halves of every turn in `timeline`, in order — `(user, assistant)`,
    /// verbatim as stored.
    pub fn conversation_texts(&self, timeline: TimelineId) -> Vec<(String, String)> {
        let read = self.conversation.read();
        (0..read.turn_count(timeline))
            .map(|i| {
                (
                    read.user_text_of(timeline, TurnIndex(i)),
                    read.assistant_text_of(timeline, TurnIndex(i)),
                )
            })
            .collect()
    }

    /// Whether any turn of `timeline` carries one of `tags` — the test a
    /// tag-scoped group applies to decide whether a conversation is in scope.
    pub fn conversation_carries(&self, timeline: TimelineId, tags: &[String]) -> bool {
        let read = self.conversation.read();
        (0..read.turn_count(timeline)).any(|i| {
            read.turn_tags(timeline, TurnIndex(i))
                .iter()
                .any(|t| tags.contains(t))
        })
    }

    /// The name a conversation was written under (its `conv_id`), if it was
    /// given one.
    pub fn conversation_conv_id(&self, timeline: TimelineId) -> Option<String> {
        self.conversation.conv_id_of(timeline)
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
        let (response_tx, response_rx) = flume::bounded(1);
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
        let frame = self.plain_prompt_section(system_prompt)?;
        let builder = Builder::for_plain_prompt_reserved(system_prompt, kind, frame);
        let layer_id = LayerId::reserved(kind);
        let group_id = GroupId::reserved(kind);
        self.new_conversation_with_projection(system_prompt, builder, layer_id, group_id, config)
    }

    /// The section a plain prompt's frame is sealed under.
    ///
    /// Chosen from the prompt's own tokens and checked against what the
    /// substrate already holds — see [`PlainPromptFrames`] for why a fixed id
    /// hands a conversation the prompt some other conversation was opened with.
    /// Give it to [`Builder::for_plain_prompt`] or
    /// [`Builder::for_plain_prompt_reserved`] with this same text, since that
    /// text is what the section seals.
    ///
    /// The id is claimed as it is returned, so a conversation opening a
    /// different prompt at the same moment is not handed it before this one
    /// has sealed. A conversation opened for one job and discarded takes
    /// [`Self::transient_prompt_section`] instead, which leaves nothing behind.
    pub fn plain_prompt_section(&self, prompt_text: &str) -> crate::Result<SectionId> {
        let encoding = self
            .tokenizer
            .encode(prompt_text, false)
            .map_err(|e| ConversationError::Tokenizer(e.to_string()))?;
        let mut frames = self.plain_frames.lock().unwrap();
        let view = self.conversation.read();
        frames
            .resolve(encoding.get_ids(), |id| {
                view.section_exists(id).then(|| view.section_tokens_of(id))
            })
            .ok_or_else(|| {
                ConversationError::Other(format!(
                    "no frame section is free for this prompt within {} probes of its slot",
                    PlainPromptFrames::MAX_PROBES
                ))
            })
    }

    /// The section a throwaway conversation's frame is sealed under, held until
    /// [`Self::release_prompt_section`].
    ///
    /// [`Self::plain_prompt_section`] for a conversation opened for one job and
    /// discarded. The frame is **transient**: never written to disk, and retired
    /// from the substrate when the last conversation holding it releases it, so
    /// a caller whose every prompt is different leaves nothing behind. Two jobs
    /// on the same prompt share the section while both hold it.
    ///
    /// Every successful call must be matched by one release, made after the
    /// conversation that used the frame has been dropped.
    pub fn transient_prompt_section(&self, prompt_text: &str) -> crate::Result<SectionId> {
        let encoding = self
            .tokenizer
            .encode(prompt_text, false)
            .map_err(|e| ConversationError::Tokenizer(e.to_string()))?;
        let id = {
            let mut frames = self.plain_frames.lock().unwrap();
            let view = self.conversation.read();
            frames.acquire(encoding.get_ids(), |id| {
                view.section_exists(id).then(|| view.section_tokens_of(id))
            })
        }
        .ok_or_else(|| {
            ConversationError::Other(format!(
                "no transient frame section is free for this prompt within {} probes of its \
                 slot",
                PlainPromptFrames::MAX_PROBES
            ))
        })?;
        // Before the conversation that uses it opens: the seal reads this to
        // skip the disk, and a section that sealed first has already written.
        self.conversation.mark_section_transient(id);
        Ok(id)
    }

    /// Let go of a frame taken with [`Self::transient_prompt_section`].
    ///
    /// The last release retires the section, on the wave thread between waves —
    /// see [`SchedulerRequest::RetireSections`]. Sent after the conversation's
    /// own slot was freed, on the same queue, so the retirement never runs ahead
    /// of the slot that was reading the frame.
    pub fn release_prompt_section(&self, section: SectionId) {
        let last = self.plain_frames.lock().unwrap().release(section);
        if last {
            let _ = self.scheduler_tx.send(SchedulerRequest::RetireSections {
                conversation: self.conversation.clone(),
                sections: vec![section],
            });
        }
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
    /// Compile a stencil spec against **this engine's** vocabulary.
    ///
    /// The vocabulary is the tokenizer, the EOS id and the vocab size together,
    /// and all three belong to the loaded checkpoint — so a caller that built
    /// its own would be compiling a grammar against a model it is not running.
    /// Exposed for callers with a tree of their own: `npcd` compiles an action
    /// loop rather than the single-call assistant shape
    /// [`Self::compile_tool_stencil`] builds.
    pub fn compile_stencil(&self, spec: &crate::stencil::TreeSpec) -> crate::Result<StencilTree> {
        let vocab = HfVocab::new(
            (*self.tokenizer).clone(),
            &self.config.eos_tokens,
            self.config.vocab_size as u64,
        );
        compile(spec, &vocab)
            .map_err(|e| ConversationError::from(candle::Error::Msg(format!("stencil: {e}"))))
    }

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
        // **A turn may make up to [`MAX_TOOL_CALLS_PER_TURN`] calls.** After each
        // one the loop offers exactly two continuations — the marker, which opens
        // another call, or the assistant-turn terminator, which ends the turn — so
        // the decoder is never free between a call and whatever follows it. That
        // is the same guarantee the single-call tree gave by baking the EOS into
        // its close (without which the model free-decodes a hallucinated answer
        // past the call), now expressed as the loop's second arm rather than as an
        // unconditional ending. The decode loop still detects the EOS in the
        // injected close run and seals the turn.
        //
        // The batching is what this buys: a model that already knows it wants
        // three files says so in one turn instead of paying a reasoning block, a
        // prefill and a belief scan for each.
        //
        // **Taken from the dialect, not written here.** The shape of a call is
        // decided by the template the weights were trained against — Qwen3.5
        // writes a nested function element, ChatML writes a JSON object — and a
        // literal in this function is a second opinion about that, free to
        // disagree with the checkpoint actually loaded.
        // The marker off the front and the turn terminator on the close, with
        // the close ending exactly ON that terminator — see
        // [`ToolCallEnvelope::for_assistant_turn`] for why the trailing newline
        // in `assistant_end` cannot be allowed to ride along.
        let envelope = ToolCallEnvelope::for_assistant_calls(&self.config.dialect);
        let close_turn = ToolCallEnvelope::turn_close(&self.config.dialect);
        let spec = compile_tool_call_loop(tools, &envelope, MAX_TOOL_CALLS_PER_TURN, &close_turn)
            .map_err(|e| {
            ConversationError::from(candle::Error::Msg(format!("tool stencil: {e}")))
        })?;
        let vocab = HfVocab::new(
            (*self.tokenizer).clone(),
            &self.config.eos_tokens,
            self.config.vocab_size as u64,
        );
        let tree = compile(&spec, &vocab).map_err(|e| {
            ConversationError::from(candle::Error::Msg(format!("tool stencil: {e}")))
        })?;
        let mut registry = TriggerRegistry::new();
        registry.register(trigger, Arc::new(tree));
        Ok(Arc::new(registry))
    }

    /// Compile the thinking-block steering trees (one per effort dial, `Off`
    /// included) once, for reuse across turns via
    /// [`ThinkSteering::registry_for`].  Like the tool
    /// stencil, this is inactive — `Ok(None)` — when the tokenizer lacks a
    /// single `<think>`/`</think>` token, so the model free-decodes its reasoning.
    ///
    /// **The tree hands control back to the decoder at `</think>`, and cannot do
    /// otherwise.** Emitting anything after the closing tag — a tool-call marker,
    /// to put a character that has finished thinking straight into a call — reads
    /// as a natural extension and breaks the index page cut. A static run's last
    /// token is deliberately held back from the forward and rides the next decode
    /// step, which commits it through `push_committed` and arms the cut; a marker
    /// in any earlier slot goes through `push_forwarded`, whose cut flag is
    /// dropped, and `run_prefill`'s `reasoning_split` declines to split when the
    /// break token is last in the pass. The block still records its
    /// `think_close_at`, so the turn seals with a reasoning span that is not a
    /// union of whole pages and every LATER turn fails
    /// `Substrate::turn_sealed_without_thinking`. See the `Off` arm in
    /// `stencil::think` for the measured case.
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
            // The assistant's reasoning is followed by prose, so control returns
            // to the decoder the moment the block closes.
            after_close: "",
        };
        // Every end token, so a think span intercepts whichever one the model
        // samples — the canonical `eos` above is only what the tree writes.
        let vocab = HfVocab::new(
            (*self.tokenizer).clone(),
            &self.config.eos_tokens,
            self.config.vocab_size as u64,
        );
        let compile_mode = |mode: ThinkMode| -> crate::Result<Arc<StencilTree>> {
            let spec = compile_think_tree(mode, &env);
            let tree = compile(&spec, &vocab).map_err(|e| {
                ConversationError::from(candle::Error::Msg(format!("think stencil: {e}")))
            })?;
            Ok(Arc::new(tree))
        };
        Ok(Some(Arc::new(ThinkSteering {
            think_open,
            off: compile_mode(ThinkMode::Off)?,
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
        // The frame's id comes from its text — see [`Self::plain_prompt_section`].
        let frame = self.plain_prompt_section(inner_prompt)?;
        let builder = Builder::for_plain_prompt(inner_prompt, frame);
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
        self.bind_slot_to_timeline(
            timeline,
            Resumed::No,
            system_prompt,
            builder,
            layer,
            group,
            config,
            section_progress,
        )
    }

    /// Open a **new GPU slot onto a conversation that already exists** in the
    /// substrate, and carry on appending to it.
    ///
    /// # What this is for
    ///
    /// Every other constructor mints a fresh `TimelineId`, so a process restart
    /// abandons whatever conversation the last one was using — the turns stay in
    /// the log, reachable by the gather, but nothing appends to them again. For
    /// an assistant that is right: a session ends. For a mind that never stops
    /// living it is not. Its conversation is *the* conversation, and a restart
    /// should rejoin it rather than start the day over with a stranger's history
    /// filed beside it.
    ///
    /// # What actually resumes
    ///
    /// The turns, the recurrent state and the carried selection belief — the
    /// three durable things a timeline owns. The first two are restored by
    /// [`crate::scheduler`]'s slot-creation funnel, which reads them from the
    /// timeline; the third is refolded from the persisted per-turn projection
    /// events. What does not resume is GPU residence: the slot starts empty and
    /// the next turn's projection materialises what it needs out of the
    /// substrate, exactly as it would after any eviction.
    ///
    /// # What the caller still owes
    ///
    /// **That this timeline is the right one to continue.** Nothing here can
    /// tell whether the schema, the prompt or the dialect has changed underneath
    /// it since it was written, and resuming into a conversation shaped by a
    /// different frame is worse than starting a new one — the model reads a
    /// history it would never have produced. Tag the conversation with a
    /// fingerprint of the frame it was opened under and check it before calling
    /// this; `npcd`'s character loop and zend's `content_sha256` ingest cache
    /// are both that pattern.
    ///
    /// Refuses an unregistered timeline, and refuses a tombstoned one — a
    /// tombstone is the record that something decided this conversation was
    /// finished, and silently reviving it would make that decision meaningless.
    pub fn resume_conversation_with_projection(
        &self,
        timeline: TimelineId,
        system_prompt: &str,
        builder: Builder,
        config: SequenceConfig,
    ) -> crate::Result<Sequence> {
        if self.is_timeline_tombstoned(timeline) {
            return Err(ConversationError::Channel(format!(
                "cannot resume conversation on timeline {timeline}: it is tombstoned",
            )));
        }
        let Some((layer, group)) = self.conversation.timeline_target(timeline) else {
            return Err(ConversationError::Channel(format!(
                "cannot resume conversation on timeline {timeline}: not registered in the substrate",
            )));
        };
        if let Some(yaml) = builder.source_yaml() {
            if let Err(e) = self.conversation.set_template(yaml.as_bytes()) {
                tracing::warn!("persist projection template failed: {e}");
            }
        }
        self.bind_slot_to_timeline(
            timeline,
            Resumed::Yes,
            system_prompt,
            builder,
            layer,
            group,
            config,
            None,
        )
    }

    /// [`Self::resume_conversation_with_projection`] for a conversation that
    /// was opened from a raw prompt string.
    ///
    /// The synthetic single-layer schema is rebuilt from `system_prompt` the
    /// same way [`Self::new_conversation`] builds it, so the frame the resumed
    /// conversation projects under is the frame it was created under — provided
    /// the caller hands over the same prompt. It cannot check that for you; see
    /// the fingerprint note on [`Self::resume_conversation_with_projection`].
    pub fn resume_conversation(
        &self,
        timeline: TimelineId,
        system_prompt: &str,
        config: SequenceConfig,
    ) -> crate::Result<Sequence> {
        let inner_prompt = {
            let s = system_prompt
                .strip_prefix(config.dialect.system_start)
                .unwrap_or(system_prompt);
            s.strip_suffix(config.dialect.system_end).unwrap_or(s)
        };
        // The frame's id comes from its text — see [`Self::plain_prompt_section`].
        let frame = self.plain_prompt_section(inner_prompt)?;
        let builder = Builder::for_plain_prompt(inner_prompt, frame);
        self.resume_conversation_with_projection(timeline, system_prompt, builder, config)
    }

    /// Allocate a slot for `timeline` and build the [`Sequence`] over it.
    ///
    /// The shared tail of [`Self::new_conversation_with_projection_progress`]
    /// and [`Self::resume_conversation_with_projection`] — everything from
    /// "the timeline exists" onwards is identical, and `resumed` chooses only
    /// which scheduler entry point allocates the slot.
    #[allow(clippy::too_many_arguments)]
    fn bind_slot_to_timeline(
        &self,
        timeline: TimelineId,
        resumed: Resumed,
        system_prompt: &str,
        builder: Builder,
        layer: LayerId,
        group: GroupId,
        config: SequenceConfig,
        section_progress: Option<&dyn Fn(u64, u64)>,
    ) -> crate::Result<Sequence> {
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
        // No summarisation is registered for the timeline: the AVL summariser is
        // disconnected (see the `SummariserThread::disabled()` note in `new`), so
        // a timeline keeps its turns whole and retrieval ranks them by provenance
        // rather than reading a compressed stand-in.
        let target = ProjectionTarget {
            layer,
            group,
            timeline,
        };

        let (response_tx, response_rx) = flume::bounded(1);
        let request = match resumed {
            // A fresh conversation: any state comes from the timeline's own
            // snapshot, which `create_sequence` reads.
            Resumed::No => SchedulerRequest::NewSequence {
                conversation: self.conversation.clone(),
                target: Some(target),
                parent: None,
                response_tx,
            },
            // The same funnel, entered by the door that says so. Both end in
            // `create_sequence(conversation, Some(target))` and restore the
            // timeline's recurrent state and carried belief; what differs is
            // that this one re-derives `(layer, group)` from the substrate
            // registry and fails loudly if the timeline was never registered.
            Resumed::Yes => SchedulerRequest::ResumeSequence {
                conversation: self.conversation.clone(),
                timeline,
                response_tx,
            },
        };
        self.scheduler_tx
            .send(request)
            .map_err(|_| ConversationError::SchedulerGone)?;

        let sequence_id = response_rx
            .recv()
            .map_err(|_| ConversationError::SchedulerGone)??;

        // **Everything past the slot allocation has to hand the slot back.**
        //
        // A built [`Sequence`] frees its slot on drop, so the only window where
        // one can be stranded is between the scheduler handing out `sequence_id`
        // and a `Sequence` existing to own it. A `?` here would return through
        // that window and leave a GPU slot allocated, its sampling state
        // registered and its conversation handle held, with nothing left that
        // knows the id — leaked until the process ends.
        //
        // Rare, and rarer still to notice: the surviving symptom is a slot count
        // that never comes back down, which reads as a busy engine rather than
        // as an error. Resume widens the window — a conversation whose tree
        // rebuild fails is a new way in — so it is closed rather than reasoned
        // about.
        let free_slot = || {
            let _ = self
                .scheduler_tx
                .send(SchedulerRequest::FreeSequence { sequence_id });
        };
        let (conv, pending) = Sequence::new_with_projection(
            self.scheduler_tx.clone(),
            sequence_id,
            Arc::clone(&self.tokenizer),
            Arc::clone(&self.literal_tokenizer),
            system_prompt,
            builder,
            target,
            config,
            CHUNK_SIZE,
            self.model_core,
            self.conversation.clone(),
            section_progress,
            // **Primed on both paths, resume included.** Priming injects the
            // schema's prelude *sections* and explicitly no turns, and the
            // per-turn `apply_projection` at submit materialises the real
            // projection regardless — so a resumed slot is warmed by this and
            // its history still arrives the ordinary way.
            true,
        )
        .inspect_err(|_| free_slot())?;
        // A group of one — the same install path the batch create takes.
        let pending: Vec<_> = pending.into_iter().collect();
        if let Err(e) = install_branch_states(&self.scheduler_tx, &pending, &self.conversation) {
            // `conv` owns the slot by now and would free it on drop, but it is
            // dropped here without being returned, so say what happened rather
            // than letting a silent drop stand in for an error.
            drop(conv);
            return Err(e);
        }

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
            rx: Receiver<crate::Result<SequenceId>>,
        }
        // Split the call's wall time three ways, because the three parts have
        // very different characters and a caller that batches creations needs to
        // know which one it is paying for. `fire` and `build` are this thread's
        // own work; `wait` is time blocked on the scheduler, which only drains
        // its request queue between waves — so a `wait` of order one wave
        // latency means the caller is stalling on a wave boundary rather than on
        // the cost of creating anything.
        let t_call = std::time::Instant::now();
        let mut fire = std::time::Duration::ZERO;
        let mut wait = std::time::Duration::ZERO;
        let mut fired: Vec<crate::Result<Fired>> = Vec::with_capacity(n);
        let t_fire = std::time::Instant::now();
        // All `n` timelines under one write-lock span, before the fire loop —
        // minting per iteration took the substrate's exclusive lock once per
        // conversation, right where conversations are opened in bursts.
        let minted = self.conversation.mint_timelines(layer, group, n);
        for timeline in minted {
            let target = ProjectionTarget {
                layer,
                group,
                timeline,
            };
            let (response_tx, rx) = flume::bounded(1);
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
                    fired.push(Ok(Fired { target, rx }));
                }
                Err(_) => fired.push(Err(ConversationError::SchedulerGone)),
            }
        }

        fire += t_fire.elapsed();

        // Phase 2 — collect each slot id (already queued, so no extra wave wait)
        // and build its Sequence. After the warm-up case has pinned the shared
        // sections, `new_with_projection` only re-references already-hot sections
        // here, so it issues no further scheduler round-trip.
        // Collect the whole batch's pending branch states as the sequences are
        // built, then install them in ONE request below. The install is
        // overwhelmingly the wait for the scheduler to drain it, and that wait is
        // per-request — installing per sequence here would put a queue wait
        // between every pair of conversations in the burst.
        let mut pending: Vec<PendingBranchState> = Vec::with_capacity(n);
        let out: Vec<crate::Result<Sequence>> = fired
            .into_iter()
            .map(|f| {
                let f = f?;
                let t_wait = std::time::Instant::now();
                let sequence_id =
                    f.rx.recv()
                        .map_err(|_| ConversationError::SchedulerGone)??;
                wait += t_wait.elapsed();
                let (conv, p) = Sequence::new_with_projection(
                    self.scheduler_tx.clone(),
                    sequence_id,
                    Arc::clone(&self.tokenizer),
                    Arc::clone(&self.literal_tokenizer),
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
                )?;
                pending.extend(p);
                Ok(conv)
            })
            .collect();
        // NOT downgraded to a warning. `install_branch_states` already handles
        // every recoverable degradation internally (missing checkpoint, unreadable
        // record, refused install) by warning and carrying on — so the only error
        // it can return is `SchedulerGone`. Swallowing it would hand back a full
        // batch of `Ok(Sequence)` values bound to a dead scheduler; this function
        // reports per entry, so every entry fails.
        if let Err(e) = install_branch_states(&self.scheduler_tx, &pending, &self.conversation) {
            let msg = format!("batch branch-state install failed: {e}");
            return (0..n)
                .map(|_| Err(ConversationError::Channel(msg.clone())))
                .collect();
        }

        let call = t_call.elapsed();
        tracing::debug!(
            target: "candle_conversation::batch_creation",
            n,
            call_ms = call.as_millis() as u64,
            fire_ms = fire.as_millis() as u64,
            wait_ms = wait.as_millis() as u64,
            build_ms = call.saturating_sub(fire + wait).as_millis() as u64,
            "batch conversation creation"
        );
        out
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
        let (resp_tx, resp_rx) = flume::bounded(1);
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
        let (event_tx, event_rx) = flume::unbounded();
        self.scheduler_tx
            .send(SchedulerRequest::SubmitTurn {
                // One turn, and it is the slot's tail.
                seal_group: None,
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
                turn_grammar: None,
                free_tool_calls_from_penalties: false,
                recorded_reply: None,
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
        // **Before the scheduler is asked to stop.** A caller blocked in
        // `GuestReceipt::wait` is blocked on the scheduler thread; once that
        // thread joins there is nobody left to answer, and the caller waits for
        // the life of the process with its HTTP request still open. Closing
        // first also refuses anything submitted during the teardown, so no job
        // lands in a queue nothing will drain.
        self.guests.queue.close();
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
