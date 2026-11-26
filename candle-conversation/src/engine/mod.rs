//! The conversation engine: the handle that owns the scheduler thread and creates
//! conversations on it.
//!
//! `ConversationEngine` holds the scheduler, the model and the tokenizer in a
//! shared core. Each substrate attached to that core gets its own handle and its
//! own persistence thread. Conversation methods are split across files by concern,
//! and all GPU work runs on the scheduler thread. That thread ends the process if
//! it panics, so the supervisor restarts the daemon.
//!
//! - `creation`: opening new, resumed, batched and named conversations.
//! - `metadata`: labels, conv_ids, custom metadata and fork lineage.
//! - `retrieval`: working sets, scopes and retrieval diagnostics.
//! - `tiering`: movement between the GPU, warm and cold tiers.
//! - `timelines`: listings, tombstones and conversation state.
//! - `ingest`: append-only layers, normalisation warm-up and gallery preload.
//! - `services`: substrate reload, guest jobs and shutdown.
//! - `compile`: tool-call and thinking grammars.
//!
//! Test files:
//!
//! - `tests_steering`: the thinking steer fires once per turn.

use crate::cancel::CancelToken;
use crate::config::{EngineConfig, SamplingConfig, SequenceConfig};
use crate::conversation::{install_branch_states, PendingBranchState, Sequence};
use crate::engine_attachments::Attachments;
use crate::error::ConversationError;
use crate::files::{FileKey, FilesPin};
use crate::group_scanner::GroupScanner;
use crate::guest::reach;
use crate::guest::Guest;
use crate::handle::{TokenDecoder, TurnEvent};
use crate::ingest_warmer::IngestWarmer;
use crate::models::identity::ModelKey;
use crate::models::Dialect;
use crate::persistence::manifest::ConvState;
use crate::persistence::record::DistillMode;
use crate::persistence::thread::{BacklogWaker, PersistenceThread};
use crate::persistence::triggers::FIRST_ATTACHMENT;
use crate::persistence::vfs::{VfsEventId, VfsEventPayload, VfsWrite};
use crate::persistence::warm_ledger::WarmLedger;
use crate::persistence::BoundSubstrate;
use crate::projection::CorruptTurnPolicy;
use crate::projection::{
    Builder, CollectionWarm, Conversation, DecodePriority, GalleryPreload, GroupId, LayerId,
    PlainPromptFrames, ProjectionTarget, Reserved, Schema, SectionId, TimelineId, TurnIndex,
    WorkingSetShare,
};
use crate::scheduler::{Scheduler, SchedulerRequest, SettleReport};
use crate::sequence_handle::SequenceId;
use crate::stencil::{
    compile, compile_steps_tail, compile_think_tree, compile_tool_call_loop, held_branches,
    held_branches_first, HeldRules, HeldValues, HfVocab, NeverTwice, OpeningCall, StencilTree,
    ThinkMode, ThinkSteerEnvelope, TokenBytes, TokenId, ToolCallEnvelope, ToolSpec,
    TriggerRegistry, ValueHolds,
};
use crate::tenant_rank::TenantRank;
#[cfg(feature = "memory-census")]
use heap_census::{set_thread, Scope};
// `ChannelProbeRunner` is deliberately not imported: the summariser is
// disconnected, so nothing constructs a runner.
use crate::substrate::pcoll::POrdMap;
use crate::substrate::{ConvCompression, Substrate};
use crate::summary_tree::{SelectionDiagnostics, SummariserThread};
use crate::token_buffer::TokenBuffer;
use crate::turn_text::literal_tokenizer;
use crate::working_set::{TypeShares, WorkingSet};

use candle::quantized::pinned_staging::release_recycled_wc;
use candle::Device;
use candle_nn::kv_cache::{ChunkedKvBacking, CompressionPolicy};
use candle_nn::CHUNK_SIZE;
use candle_transformers::models::batched_inference::{ManagedBatchedModel, ModelCoreProperties};
use candle_transformers::models::delta_net::release_recurrent_host_buffers;
use candle_transformers::models::qwen_vision::VisionTowerSource;
use flume::{Receiver, Sender};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::process::exit;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

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
    /// What this handle shares with every other substrate attached to the
    /// same engine: the scheduler and the model it owns.
    core: Arc<EngineCore>,

    /// This attachment's id — how the scheduler knows its persistence trigger.
    attachment: u64,

    /// The frame sections handed out to plain-prompt conversations, and the
    /// claim on each — see [`Self::plain_prompt_section`]. Sections live in a
    /// substrate, so the frames are this attachment's.
    plain_frames: Mutex<PlainPromptFrames>,

    /// This attachment's `Conversation` handle on its substrate: holds per-turn
    /// metadata (token counts, scores, sig entries, sealed-sequence handles)
    /// across every `Sequence` allocated from this engine handle. Each
    /// `Sequence` receives a clone.
    conversation: Conversation,

    /// The substrate's persistence thread — owns its redo-log write path.
    /// Wakes on a 5-second tick or on triggers from the scheduler; joined on
    /// [`Self::shutdown`] (or drop) after a final drain pass.
    persist_thread: PersistenceThread,

    /// Progress of this substrate's reload, written by the scheduler thread
    /// and polled by the daemon's load-state machine for the GUI.
    substrate_reload_status: Arc<SubstrateReloadStatus>,

    /// Set once [`Self::shutdown`] has run, so it runs once.
    detached: AtomicBool,
}

/// What every substrate attached to one engine shares
/// (`docs/shared_model_host.md` §3.2): the scheduler thread and the model it
/// owns, the tokenizer, the co-resident guests, the model's identity, and what
/// a later attachment needs to start a persistence thread of its own.
struct EngineCore {
    /// Channel to submit work to the scheduler thread.
    scheduler_tx: Sender<SchedulerRequest>,

    /// Handle to the scheduler thread, joined when the last attachment shuts
    /// down. `Mutex<Option>` so shutdown can run through `&self`.
    scheduler_handle: Mutex<Option<JoinHandle<()>>>,

    /// Tokenizer (shared, immutable, safe to clone into conversations).
    tokenizer: Arc<tokenizers::Tokenizer>,

    /// [`Self::tokenizer`] reading every chat tag as plain characters, for the
    /// literal pieces of a turn — see [`crate::turn_text`]. Built once here: it
    /// is a copy of the whole vocabulary.
    literal_tokenizer: Arc<tokenizers::Tokenizer>,

    /// The model's end-of-sequence tokens, vocabulary size and dialect — what
    /// the stencil compilers read.
    eos_tokens: TokenBuffer,
    vocab_size: usize,
    dialect: Dialect,

    /// Every token's bytes, sorted — what a stencil compile reads to find the
    /// tokens that run on past an arm (`stencil::trie`). Built on the first
    /// compile, from the whole vocabulary, and shared by every compile after.
    token_bytes: OnceLock<Arc<TokenBytes>>,

    /// Async summariser thread — **always the disabled handle.** It would drain
    /// the per-turn pending queue, run §6 probes and build the per-timeline AVL
    /// summary tree; none of that happens, because the summariser is
    /// disconnected (see [`ConversationEngine::new`]) and no timeline enqueues.
    /// Kept so the lifecycle calls stay honest no-ops rather than disappearing.
    summariser_thread: SummariserThread,

    /// The co-resident models that borrow the card between waves.
    ///
    /// Shared with the scheduler, which polls the queue once per pass and
    /// drains it between forwards. Held here because the *registry* is a
    /// deployment decision made before the engine starts, and the *queue* is
    /// what callers submit to from any thread — see
    /// [`ConversationEngine::submit_guest`] and [`crate::guest`].
    guests: Arc<crate::guest::Guests>,

    /// Static model properties captured before the model moves to the scheduler thread.
    model_core: ModelCoreProperties,

    /// The model every attached substrate must be bound to.
    model_key: ModelKey,

    /// How many layers hold K/V — what says when a turn another member wrote
    /// has all its chunks (`docs/replicated_substrate.md` §6.5).
    kv_layers: usize,

    /// What [`ConversationEngine::attach`] hands a new persistence thread
    /// beside the backings ([`Attachments::persist_with`]): the device, the
    /// compression policy, and a share of the one warm budget every attached
    /// substrate's purge divides.
    device: Device,
    compression_policy: Option<CompressionPolicy>,
    warm_ledger: Arc<WarmLedger>,

    /// Every attached substrate, and the backings while the core runs.
    attachments: Mutex<Attachments>,

    /// The id the next attachment takes.
    next_attachment: AtomicU64,
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
        // How large a draw this card serves, stated once from its size: a draw
        // past it is refused at submission rather than evicting the engine for
        // ground the card does not have. A device with no memory report (the
        // CPU) keeps the request's own per-side limit.
        if let Ok((_, card)) = session.device().mem_get_info() {
            let side = reach::side_for_card(card as u64);
            guests.queue.set_reach(side);
            tracing::info!(
                card_mib = card >> 20,
                side,
                "guest draws on this node reach {side}x{side}"
            );
        }
        // And to the cluster, whose gateways place a draw where it is reached —
        // nowhere here when this node has no image guest to draw with.
        reach::state(reach::advertised(
            guests.queue.reach(),
            guests.registry.has(Guest::Image),
        ));
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
        // substrate — the one the builder opened and bound to this model
        // before loading it.
        let shared = config.substrate.shared().clone();
        {
            let (log_bytes, records, indexed) = shared.persistence.call(|p| {
                (
                    p.write_offset(),
                    p.recovered_record_count(),
                    p.last_index().is_some(),
                )
            });
            tracing::info!(
                read_only = shared.persistence.is_read_only(),
                log_bytes,
                records,
                indexed,
                streams = shared.substrate.read().all_streams().count(),
                "substrate attached"
            );
        }
        let conversation = Conversation::from_shared(shared)
            .with_cancel(config.cancel.clone())
            .with_tenant_rank(config.tenant_rank);

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
        let backings: Arc<Vec<ChunkedKvBacking>> = Arc::new(session.backings().to_vec());
        let device = session.device().clone();
        let compression_policy = config.batched_config.compression_policy();
        let warm_ledger = WarmLedger::new();
        let persist_thread = PersistenceThread::spawn(
            conversation.clone(),
            Arc::clone(&backings),
            device.clone(),
            compression_policy,
            warm_ledger.join(),
        );
        let persist_trigger = persist_thread.trigger_handle();
        persist_trigger.on_backlog_fall(scheduler_waker(&tx));

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
                #[cfg(feature = "memory-census")]
                set_thread(Scope::Scheduler);
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
                scheduler.register_substrate(&scheduler_conversation);
                scheduler.reconstruct_substrate(&scheduler_conversation, &reload_status_for_thread);
                tracing::info!(
                    reconstruct_ms = t_recon.elapsed().as_millis() as u64,
                    "scheduler thread: substrate reconstruct complete"
                );
                // **A scheduler that panics ends the process.** Every model
                // forward runs on this thread, and nothing restarts it: a panic
                // here left the daemon answering every request with "scheduler
                // has shut down" until someone restarted it by hand. The panic
                // hook has already logged the panic and its backtrace; exiting
                // hands the restart to the supervisor (keep).
                if catch_unwind(AssertUnwindSafe(|| scheduler.run())).is_err() {
                    tracing::error!(
                        "the scheduler thread panicked; ending the process so its \
                         supervisor starts it again"
                    );
                    exit(SCHEDULER_PANIC_EXIT);
                }
            })
            .map_err(|e| {
                ConversationError::Channel(format!("failed to spawn scheduler thread: {e}"))
            })?;

        let kv_layers = backings.len();
        let core = EngineCore {
            scheduler_tx: tx,
            scheduler_handle: Mutex::new(Some(handle)),
            literal_tokenizer: Arc::new(literal_tokenizer(&tokenizer)),
            tokenizer: Arc::new(tokenizer),
            eos_tokens: config.eos_tokens.clone(),
            vocab_size: config.vocab_size,
            dialect: config.dialect.clone(),
            token_bytes: OnceLock::new(),
            summariser_thread,
            guests,
            model_core,
            model_key: config.substrate.model_key().clone(),
            kv_layers,
            device,
            compression_policy,
            warm_ledger,
            attachments: Mutex::new(Attachments::first(
                FIRST_ATTACHMENT,
                conversation.clone(),
                backings,
            )),
            next_attachment: AtomicU64::new(FIRST_ATTACHMENT + 1),
        };
        Ok(Self {
            core: Arc::new(core),
            attachment: FIRST_ATTACHMENT,
            plain_frames: Mutex::new(PlainPromptFrames::default()),
            conversation,
            persist_thread,
            substrate_reload_status,
            detached: AtomicBool::new(false),
        })
    }

    /// Attach another substrate to this engine — a second tenant on the same
    /// loaded model and scheduler (`docs/shared_model_host.md` §3.2).
    ///
    /// The substrate must be bound to the model this engine runs; one already
    /// attached, or one holding a timeline id another attached substrate holds,
    /// is refused, because the scheduler keys its maps by timeline and two
    /// substrates naming one timeline would share its state. The attachment
    /// gets its own persistence thread and is reloaded from its redo log on
    /// the scheduler thread; poll [`Self::substrate_reload_status`] on the
    /// returned handle. Conversations created through the returned handle live
    /// on the attached substrate.
    pub fn attach(
        &self,
        substrate: BoundSubstrate,
        cancel: CancelToken,
        tenant_rank: TenantRank,
        layer_corrupt_turn: HashMap<LayerId, CorruptTurnPolicy>,
    ) -> crate::Result<ConversationEngine> {
        self.attacher()
            .attach(substrate, cancel, tenant_rank, layer_corrupt_turn)
    }

    /// A handle that attaches further substrates to this engine without
    /// holding this one — what a tenant on another thread is given
    /// (`crate::engine_slot`).
    pub fn attacher(&self) -> EngineAttacher {
        EngineAttacher {
            core: Arc::clone(&self.core),
        }
    }

    /// How many substrates are attached to this engine.
    pub fn attached_count(&self) -> usize {
        self.core
            .attachments
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .list
            .len()
    }

    /// The model this engine runs.
    pub fn model_key(&self) -> &ModelKey {
        &self.core.model_key
    }

    /// How many layers hold K/V.
    pub fn kv_layers(&self) -> usize {
        self.core.kv_layers
    }

    /// The model's chat dialect.
    pub fn dialect(&self) -> &Dialect {
        &self.core.dialect
    }
}

/// The process exit status when the scheduler thread panicked. `EX_SOFTWARE`:
/// an internal fault, not a usage error.
const SCHEDULER_PANIC_EXIT: i32 = 70;

/// A persistence thread's wake for the scheduler, called when its hot→warm
/// backlog falls ([`PersistenceTrigger::on_backlog_fall`]): with nothing
/// admissible the scheduler blocks on its request channel, and this is what
/// tells it the backlog that shut admission has moved. `try_send` on an
/// unbounded channel, so the persistence thread never waits on it.
///
/// [`PersistenceTrigger::on_backlog_fall`]: crate::persistence::thread::PersistenceTrigger::on_backlog_fall
fn scheduler_waker(tx: &Sender<SchedulerRequest>) -> BacklogWaker {
    let tx = tx.clone();
    Box::new(move || {
        let _ = tx.try_send(SchedulerRequest::Wake);
    })
}

/// Attaches substrates to an engine — a cheap clone of its shared core, so the
/// engine's own handle can stay with the tenant that built it.
#[derive(Clone)]
pub struct EngineAttacher {
    core: Arc<EngineCore>,
}

impl EngineAttacher {
    /// The model the engine runs — what a substrate attached to it must be
    /// bound to.
    pub fn model_key(&self) -> &ModelKey {
        &self.core.model_key
    }

    /// See [`ConversationEngine::attach`].
    pub fn attach(
        &self,
        substrate: BoundSubstrate,
        cancel: CancelToken,
        tenant_rank: TenantRank,
        layer_corrupt_turn: HashMap<LayerId, CorruptTurnPolicy>,
    ) -> crate::Result<ConversationEngine> {
        if !substrate.model_key().same_model(&self.core.model_key) {
            return Err(ConversationError::Other(format!(
                "the substrate at {} is bound to {}, and this engine runs {}",
                substrate.dir().display(),
                substrate.model_key().describe(),
                self.core.model_key.describe()
            )));
        }
        let conversation = Conversation::from_shared(substrate.shared().clone())
            .with_cancel(cancel)
            .with_tenant_rank(tenant_rank);
        // Registered before the reload, as `new` does, so the reload applies
        // each layer's policy.
        for (&layer, &policy) in &layer_corrupt_turn {
            conversation.set_layer_corrupt_turn_policy(layer, policy);
        }
        // The checks, the scheduler's request and the registration under one
        // hold of the lock: two attaches racing cannot both pass the checks,
        // and none lands on a core whose last attachment has stopped it.
        let mut attached = self
            .core
            .attachments
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let backings = attached.admit(&conversation, &substrate.dir())?;
        let persist_thread = PersistenceThread::spawn(
            conversation.clone(),
            backings,
            self.core.device.clone(),
            self.core.compression_policy,
            self.core.warm_ledger.join(),
        );
        let id = self.core.next_attachment.fetch_add(1, Ordering::Relaxed);
        let status = Arc::new(SubstrateReloadStatus::default());
        let trigger = persist_thread.trigger_handle();
        trigger.on_backlog_fall(scheduler_waker(&self.core.scheduler_tx));
        self.core
            .scheduler_tx
            .send(SchedulerRequest::AttachSubstrate {
                id,
                trigger,
                conversation: conversation.clone(),
                status: Arc::clone(&status),
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        attached.list.push((id, conversation.clone()));
        drop(attached);
        tracing::info!(
            attachment = id,
            dir = %substrate.dir().display(),
            "substrate attached to the engine"
        );
        Ok(ConversationEngine {
            core: Arc::clone(&self.core),
            attachment: id,
            plain_frames: Mutex::new(PlainPromptFrames::default()),
            conversation,
            persist_thread,
            substrate_reload_status: status,
            detached: AtomicBool::new(false),
        })
    }
}

impl Drop for ConversationEngine {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

mod compile;
mod creation;
mod ingest;
mod metadata;
mod retrieval;
mod services;
#[cfg(test)]
mod tests_steering;
mod tiering;
mod timelines;

pub use compile::*;
pub use services::*;
