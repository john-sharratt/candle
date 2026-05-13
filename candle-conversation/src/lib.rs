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

mod batched_sampler;
pub mod projection;
mod config;
pub mod provenance;
mod conversation;
pub mod conversation_log;
mod decode_health;
mod engine;
mod error;
mod handle;
pub mod models;
pub mod narrator;
pub mod prompts;
mod scheduler;
mod sequence_handle;
pub mod substrate;
pub mod substrate_cache;
mod stats;
pub mod store;
pub mod think_strip;
mod time_source;
pub mod token_buffer;
pub mod tree;
pub mod turn;

pub use config::{
    pick_max_hot_turns, SequenceConfig, DecodeHealthConfig, DryConfig,
    EngineConfig, SamplingConfig, SchedulerConfig,
};
pub use conversation::Sequence;
pub use engine::ConversationEngine;
pub use error::ConversationError;
pub use handle::{TokenDecoder, TurnEvent, TurnHandle, TurnResponse};
pub use sequence_handle::SequenceId;
pub use stats::TurnStats;
pub use provenance::{
    ProbeSignatures, ProvenanceFile, SigEntry, TokenSignature, TurnChunkRank, TurnSignatures,
};
pub use token_buffer::TokenBuffer;
pub use tree::TokenizedText;
pub use turn::{Role, Turn, TurnId, TurnOptions};

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
pub use candle_transformers::generation::Sampling;
pub use candle_transformers::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ManagedBatchedModel,
};

/// Convenience type alias for Results in this crate.
pub type Result<T> = std::result::Result<T, ConversationError>;
