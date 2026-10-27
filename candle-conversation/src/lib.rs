// Several clippy lints are systemic in this crate by deliberate
// design and silenced here at the crate root with rationale:
//
// * `doc_lazy_continuation` — Markdown list continuations are
//   formatted to read well at the source level; the rendered
//   docs.rs output is fine either way.
// * `too_many_arguments` — scheduler / model / engine entry points
//   carry many distinct concerns by necessity (model weights,
//   tokenizer, projection schema, persistence config, progress
//   sinks, etc.); bundling them into context structs at the API
//   boundary would hide intent without saving complexity.
// * `type_complexity` — projection / scheduler types form nested
//   tuples by design (turns × layers × tiers).  Aliases would just
//   rename them, not simplify.
// * `large_enum_variant` — `SchedulerRequest` and `RefreshOutcome`
//   carry one heavy variant alongside lightweight ones; the
//   payloads are consumed at the receiving end and never queued,
//   so the size disparity is academic.
// * `chunks_exact_to_as_chunks` — a style lint that became default in a recent
//   clippy and fires on correct, unchanged code that reads packed bytes in
//   fixed strides. Rewriting those call sites would change `&[T]` to `&[T; N]`
//   through the quantization and KV paths of a fork that has to stay mergeable
//   with upstream candle, for no behavioural gain. Forty-two of them drowned
//   the `-D warnings` gate, which is worse than the lint: a gate nobody can get
//   to zero is a gate that hides the next real warning.
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::large_enum_variant)]
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

//! Turn-based conversation engine for the candle inference stack.
//!
//! Provides a high-level conversation API that manages multi-turn dialogue
//! with streaming token generation, backed by batched paged attention.
//!
//! # Architecture
//!
//! A single **scheduler thread** owns all GPU resources (model weights, KV cache
//! arenas, inference session). Caller threads hold lightweight [`Sequence`]
//! objects that submit work via channels and get back [`TurnHandle`]s for
//! streaming or blocking reception of generated tokens.
//!
//! # Phase 1
//!
//! - Single scheduler thread, single-mode prefill (no small/large split)
//! - All turns HOT (no tiering, no tree rotations, no rematerialization)
//! - Fork supported from day 1 (CoW pages via `fork_sequence`)
//! - Append-only cold store for persistence

mod banned_rows;
mod batched_sampler;
mod cancel;
/// Whether a turn chain finished what it started — the ingest-recovery signal.
pub mod chain_health;
mod config;
mod conversation;
pub mod conversation_log;
mod decode_health;
pub mod decoded_text;
mod dry_reference;
mod engine;
mod engine_attachments;
mod error;
/// The KV-fragmentation probe: the specification for the compaction pass, and the gate
/// that keeps it honest. Reachable from both the integration tests and the
/// `kv_fragmentation` example, which is why it is here rather than under `tests/`.
pub mod fragmentation_probe;
mod group_scanner;
pub mod guest;
mod handle;
pub mod index_pages;
mod ingest_warmer;
mod line_ends;
pub mod models;
pub mod narrator;
pub mod normalization;
mod penalty_counts;
pub mod persistence;
mod prefill_width;
pub mod projection;
/// The process-wide hot-path span profiler, for a daemon that serves its own
/// timings.
///
/// Re-exported rather than reimplemented: the spans are recorded across three
/// crates — this one's scheduler loop, the model's host stages, and the GPU
/// spans in the kernels — into a single table, and a second accessor would be a
/// second table by another name. Snapshotting resets, so two reads bracket a
/// window exactly.
pub mod profile {
    /// How the elastic VRAM boundary answered: the pool's growth answers and
    /// the expert zone's outcomes, process-wide since boot — so a window's
    /// share is the difference of two readings.
    pub use candle_nn::kv_cache::spare_tally;
    pub use candle_transformers::models::expert_lre::grow_tally;
    pub use candle_transformers::models::profile::{
        pipeline_snapshot_and_reset, set_gpu_span_period, ProfileSnapshot,
    };
}
pub mod engine_slot;
pub mod prompts;
pub mod provenance;
mod recorded_reply;
mod recovered_message;
mod sampler_args;
pub(crate) mod scheduler;
/// A throwaway substrate directory that removes itself on drop and sweeps the
/// corpses of crashed runs — so a harness never attaches to the live store.
pub mod scratch_substrate;
mod sealed_turn;
mod sequence_handle;
mod stats;
pub mod stencil;
pub mod store;
pub mod substrate;
pub mod summary_tree;
pub mod tenant_rank;
pub mod think_strip;
mod time_source;
pub mod token_bias;
pub mod token_buffer;
pub mod tree;
pub mod turn;
pub mod turn_group;
pub mod turn_layout;
pub mod turn_text;
pub mod working_set;

pub use cancel::{CancelToken, CancelWait};
pub use config::{
    pick_max_hot_turns, DecodeHealthConfig, DryConfig, EngineConfig, ModeSampling, SamplingConfig,
    SchedulerConfig, SequenceConfig, REPROJECT_EVERY_N_TOKENS, REPROJECT_MAX_PROBE_TOKENS,
    REPROJECT_TRIGGER_TEXTS,
};
pub use conversation::{
    AskScope, Carried, ChainRound, GlueMarkers, Linking, PieceKind, PromptDump, PromptPiece,
    SectionRef, SectionState, Sequence, UnsealedAsker, Want,
};
pub use engine::{
    ConversationEngine, EngineAttacher, SubstrateReloadStatus, SubstrateReloader, ThinkSteering,
};
pub use engine_slot::{Capabilities, EngineSlot, EngineSource, EngineTicket};
pub use error::ConversationError;
pub use group_scanner::{GroupScanner, ScanRefused};
pub use handle::{TokenDecoder, TurnEvent, TurnHandle, TurnResponse};
pub use ingest_warmer::IngestWarmer;
pub use projection::{
    BucketKind, OptionalState, ProjectionBucket, ProjectionEvent, ProjectionPoint, SelectionState,
    NO_THINK_SELECTOR, TOOL_ROUND_SELECTOR,
};
pub use recovered_message::RecoveredMessage;
pub use scheduler::branch_checkpoint_counts;
pub use scheduler::memory_report;
pub use scheduler::phase_ring;
pub use scheduler::provenance_capture_path_counts;
pub use scheduler::recurrent_state_cost;
pub use scheduler::relief_trace;
pub use scheduler::SettleReport;
pub use sealed_turn::{SealedPages, SealedTurn};
pub use sequence_handle::SequenceId;
pub use stats::{FinishReason, TurnStats};
pub use tenant_rank::TenantRank;
pub use token_buffer::TokenBuffer;
pub use tree::TokenizedText;
pub use turn::{Role, Turn, TurnId, TurnOptions};
pub use turn_text::{TextPiece, TurnText};

// Phase 1 tree types — available under their own names.
// `TurnId` from `tree` is re-exported as `TreeTurnId` to avoid shadowing the
// existing `turn::TurnId` (u64) used by `Turn` records.
pub use tree::{
    ConversationNode, ConversationSegment, ConversationTree, ConversationTreeConfig,
    ConversationTreeFork, ConversationTurn, FixedTimeSource, NodeId, SegmentId, StorageTier,
    TreeMetadataDelta, TreePatch, TurnId as TreeTurnId, TurnType, TEMPORAL_MARKER_POSTFIX,
};

// Re-export the batched sampler for advanced use cases.
pub use batched_sampler::{BatchedSampler, SequenceSamplingState};

// Re-export types callers need from our dependencies.
pub use candle_nn::kv_cache::{KvFormat, QuantFormat};
pub use candle_transformers::generation::Sampling;
pub use candle_transformers::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ManagedBatchedModel,
};
/// The CPU expert split's knobs and measured costs, for a daemon's config and
/// its admin route (`docs/cpu_expert_compute.md` §5.3.1).
pub use candle_transformers::models::expert_lre::cpu_pool::params as expert_split;
/// Layers the CPU expert pool has served in this process.
pub use candle_transformers::models::expert_lre::cpu_pool::service::served_total as expert_split_served;
/// The last forward's walk through the routed layers, for a daemon's telemetry
/// page — recorded by the expert pipeline thread, read here with no engine lock.
pub use candle_transformers::models::expert_lre::layer_trace;
/// What a model load reports while it runs — the phase and how far through —
/// for a daemon's loading screen.
pub use candle_transformers::models::load_phase::{LoadPhase, LoadProgress};

/// Refuse every model download from now on, or allow them again: what the
/// caches hold still loads. A host turns it with its network switch. Nothing to
/// refuse in a build without the `hub` feature.
pub fn set_hub_offline(offline: bool) {
    #[cfg(feature = "hub")]
    candle_transformers::models::hub_download::set_offline(offline);
    #[cfg(not(feature = "hub"))]
    let _ = offline;
}

/// Convenience type alias for Results in this crate.
pub type Result<T> = std::result::Result<T, ConversationError>;
