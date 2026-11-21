//! Batched inference over paged attention, where many sequences share one KV
//! cache and a session runs them through forward passes together.
//!
//! `BatchedInferenceSession` allocates sequence slots, owns the chunked KV
//! backing that every layer shares, and tracks each sequence's offset and state.
//! Each layer has its own backing over a shared arena pool, so layers agree on
//! block structure by construction, and the helpers here reconcile, snapshot and
//! roll back that structure around forwards. Models declare their behaviour
//! through `ManagedBatchedModel` and `ModelCoreProperties`, and speculative
//! decoding, prefill slab packing, provenance capture and seal checks all run on
//! the same session.
//!
//! - `accessors`: read-only views of layout, dtype and layer count.
//! - `blanket_impl`: the uniform-transformer model trait and wave sweep.
//! - `block_tail`: block counts kept in agreement across layers.
//! - `config`: session settings and their defaults.
//! - `construction`: session creation, shared backings and the rung schedule.
//! - `decode_metadata`: device slot headers for one layer group.
//! - `diagnostics`: palette and format grids for debugging.
//! - `glue`: gap-fill glue metadata.
//! - `inference_mode`: modes mapping to dtypes and KV formats.
//! - `injection`: placing sealed and borrowed pieces on a sequence.
//! - `kv_arenas`: GPU arena release, compaction and fragmentation.
//! - `kv_layers`: division into stream and draft layers.
//! - `managed_model`: the scheduler-facing model trait.
//! - `model_core`: static model properties for planning.
//! - `prefill_slabs`: slab packing for pure prefills.
//! - `provenance`: provenance gathering and sign packing.
//! - `seal_guard`: refusal of sealed per-layer skew.
//! - `sequences`: sequence slot lifecycle.
//! - `snapshot`: sealed per-layer snapshots.
//! - `stats`: token, block and compression figures.
//! - `views`: zero-copy view sequences.
//! - `wave_result`: forward outputs held with their span.
//! - `tests_seal`: seal guard tests.
//! - `tests_slab`: prefill slab packer tests.

use super::expert_lre::PipelineStats;
use super::expert_lre::ProfileSnapshot;
use super::expert_lre::WeightPlanning;
use crate::models::delta_net::ExportedLayerState;
use crate::models::delta_net::RecurrentCompaction;
use crate::models::delta_net::TIMELINE_ROWS;
use crate::models::delta_net::{ArmKind, ReplyRows};
use crate::models::hit_layers::{HitPercentiles, PieceHits, PooledRound};
use crate::models::kv_cache_utils::{new_kv_caches, KvCaches};
use crate::models::layer_parallel;
use crate::models::piece_key::PieceKey;
use crate::models::positional_piece::PositionalPiece;
use crate::models::residual_rows::CreepResidual;
use crate::models::rope_schedule::RungSelect;
#[cfg(feature = "cuda")]
use crate::models::sealed_injection::{reconcile_sealed_prefix, SealedInjection};
use crate::models::selection_strata::StrataTokens;
use crate::models::slot_header::{SlotHeaderHost, SLOT_HEADER_BYTES};
use candle::quantized::pinned_staging::Generation;
#[cfg(feature = "cuda")]
use candle::quantized::pinned_staging::GpuBuf;
use candle::quantized::GgmlDType;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{
    fragmentation, launch_scratch_reserve, plan_pool, region_stats, ArenaKey, ArenaLocation,
    ChunkedKvBacking, CompressionPolicy, Fragmentation, GpuArenaClassStats, GroundLost, HeadGids,
    KvCache, KvFormat, ModelGeometry, PrefillKvStageLayout, PrefillLaunchBounds, ProvSource,
    QuantFormat, SizeClass, WavePlan, WaveWidth, CHUNK_SIZE, REGION_BYTES, WAVE_FFN_BYTES,
};
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use super::batched_layer::GlueMeta;
use super::batched_model::{
    activation_dtype, BatchedInference, BatchedModelCore, WaveGuard, WavePhase,
};
use super::held_tensor::held_empty;
#[cfg(feature = "cuda")]
use super::wave_buffers::wave_from_vec_ticketed;
use super::wave_driver::{drive_step, drive_wave, StepOutcome, StepSpec, WaveGroups, WaveSweep};
#[cfg(feature = "cuda")]
use crate::models::profile::{gpu_span, pipeline_record_duration, span};
use crate::models::speculative_choice::{
    all_rows, AcceptWalk, SpeculativeStep, TokenChooser, TypicalAcceptance,
};
use crate::models::verify_wave::{issue_verify_wave, upload_plan_rows, VerifyPlan};
#[cfg(feature = "cuda")]
use candle::wave_provenance::WaveTicket;
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::GapCycle;

#[cfg(feature = "cuda")]
mod gap;

mod gap_seal;
pub use gap_seal::GapSeal;

#[cfg(test)]
mod prov_signs_tests;

/// Batched inference session that manages KV cache state for multiple sequences.
///
/// This is the main interface for batched inference. It:
/// - Manages the chunked KV cache backing for paged attention
/// - Owns KvCaches for each sequence
/// - Tracks per-sequence state (offsets, active status)
/// - Provides direct access for models implementing [`ManagedBatchedModel::forward_batched`]
pub struct BatchedInferenceSession {
    /// Shared KV cache backing for all layers.
    /// One backing per layer.
    backings: Vec<ChunkedKvBacking>,
    /// Per-sequence state, indexed by sequence index.
    sequences: Vec<Option<SequenceState>>,
    /// Slot ids held out of the free pool with no state — see
    /// [`Self::reserve_sequence`].
    reserved: HashSet<usize>,
    /// Configuration used to create this session.
    config: BatchedConfig,
    /// Number of layers in the model.
    num_layers: usize,
    /// Leading layers a wave steps together — see [`KvLayers`]. Every
    /// stream-wide aggregate is taken over `0..stream_layers`, never over all
    /// `num_layers`, so a draft head's layer cannot poison one.
    stream_layers: usize,
    /// Per layer, a ceiling on the compression level its seals use. `None`
    /// takes the session's. See [`Self::cap_layer_seal_level`].
    layer_seal_cap: Vec<Option<u8>>,
    /// Device the session is on.
    device: Device,
    /// Pending reprojection-glue descriptors, set by the wave immediately before
    /// its gap-fill `forward_batched`. One entry per sequence (in the forward's
    /// `seq_indices` order). Taken + cleared inside `forward_batched`, which
    /// routes HD128 glue to the paged-glue kernel.
    pending_glue: Option<Vec<PendingGlue>>,
    /// How the model picks a sequence's RoPE rung — its ceilings and floor —
    /// what every header writer picks `SlotHeader.rope_rung` by, from its reach
    /// (`docs/progressive_yarn.md` §8). One unbounded rung until the model sets
    /// its own: the latent kernels read no rung.
    rope: RungSelect,
    /// Whether the wave being issued hands its logits to a caller that reads
    /// them **in place** on the head's span — the accept walk of
    /// [`ManagedBatchedModel::speculative_decode_step_batch`], which selects
    /// rows there. A forward prices `WaveBuffer::AcceptRows` for its scored
    /// rows only when this is set; a caller that owns the logits
    /// first (the scheduler) never sets it, so its waves reserve nothing for a
    /// walk that does not happen on the span.
    accept_in_place: bool,
    /// Rows of the wave being issued that the sampler holds to a grammar's
    /// bitmap: the forward prices `WaveBuffer::StencilBitmaps` for them.
    stencil_rows: usize,
    /// The draft walk's token buffer — see [`Self::walk_tokens`].
    walk_tokens: Option<Tensor>,
    /// The gap quantize's state from one gap to the next, made with a session
    /// that quantizes (and, for one built over shared backings, at its first
    /// gap) — see `gap`.
    #[cfg(feature = "cuda")]
    gap: Option<GapCycle>,
}

mod accessors;
mod blanket_impl;
mod block_tail;
mod config;
mod construction;
mod decode_metadata;
mod diagnostics;
mod glue;
mod inference_mode;
mod injection;
mod kv_arenas;
mod kv_layers;
mod managed_model;
mod model_core;
mod prefill_slabs;
mod provenance;
mod seal_guard;
mod sequences;
mod snapshot;
mod stats;
#[cfg(test)]
mod tests_seal;
#[cfg(test)]
mod tests_slab;
mod views;
mod wave_result;

pub use config::*;
pub use glue::*;
pub use inference_mode::*;
pub use kv_layers::*;
pub use managed_model::*;
pub use model_core::*;
pub use prefill_slabs::*;
pub use provenance::*;
pub use seal_guard::*;
use sequences::*;
pub use stats::*;
pub use views::*;
pub use wave_result::*;
