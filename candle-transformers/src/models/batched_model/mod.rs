//! Batched transformer inference for continuous batching: the model trait, and
//! the wrapper that runs a model's waves.
//!
//! A model implements the small `BatchedModelCore` trait of accessors.
//! `BatchedInference` wraps it and owns the RoPE cache and the forward, so that
//! caching is written once for every model. The attention metadata of a wave is
//! built inside the layer sweep, because a model may move its arena before its
//! layers read it. The types here describe the outcome of a sweep over a layer
//! range, and the constant that sets how many free KV regions a wave keeps back
//! before it asks the weight side for more ground.
//!
//! - `inference_wrapper`: the wrapper, its RoPE tables and the co-batched wave forward.
//! - `model_core`: the model trait, the wave shapes and the boundary hooks.

use std::sync::{Arc, Mutex, RwLock};

#[cfg(not(feature = "cuda"))]
use candle::wave_provenance::WaveTicket;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::KvCache;
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{
    begin_forward, begin_wave, end_wave_transient, plan_wave_transient, LayerPhase, ModelGeometry,
    PrefillKvStageLayout, PrefillLaunchBounds, WavePlan, WaveWidth,
};
use candle_nn::Module;

#[cfg(feature = "cuda")]
use super::batched_inference::build_glue_meta;
use super::batched_inference::BatchedInferenceSession;
use super::batched_layer::{
    forward_layer_batched_mixed, BatchedAttentionLayer, BatchedAttentionParams, BatchedPrefillMeta,
    DecodeHeaders, WaveAttnGroup,
};
use super::delta_net::RecurrentCompaction;
use super::expert_lre::PipelineStats;
use super::expert_lre::ProfileSnapshot;
use super::expert_lre::{WeightPlan, WeightPlanning};
use super::head_rows::select_head_rows;
use super::lazy_rope::LazyRope;
use super::prefill_utils::SharedPm;
use super::quantized_matmul::QMatMul;
use super::residency_rows::residency_decode_rows;
use super::rope_schedule::{RopeRungs, RopeSchedule};
use super::rope_tables::CisPrecomputations;
use super::tensor_cat::TensorCat;
use super::wave_admit::admit_wave_kv;
use super::wave_driver::{assemble_wave_contexts, SweepMode, WaveGroups};
#[cfg(feature = "cuda")]
use crate::models::prefill_utils::paged_decode_q8_head_dim;
use crate::models::profile::{pipeline_record, profile_now};
use crate::models::wave_buffers::{wave_empty_ticketed, wave_root, wave_root_ticketed};
use crate::models::wave_token_ids::device_token_ids;
use crate::quantized_nn::RmsNorm;
use candle_nn::Embedding;

/// The forward-scoped generation a wave's head outputs were carved from.
///
/// Returned alongside the phase so the caller can keep the span open for as long
/// as it holds the values sitting in it. There is no wave domain without CUDA,
/// so off-CUDA this is a unit and every guard is `None`.
#[cfg(feature = "cuda")]
pub type WaveGuard = candle_nn::kv_cache::WaveGeneration;

#[cfg(not(feature = "cuda"))]
pub type WaveGuard = ();

/// Outcome of one sweep over a contiguous layer range.
///
/// Stopped short of the final layer it yields the inter-layer residual stream,
/// still on its forward span; reached the last layer it ran the head and yields
/// per-sequence logits; a segment being prepared for a step yields its price.
pub enum WavePhase {
    /// Layers `[start, end)` ran with `end < num_layers`; the residual stream, on
    /// the forward span, for the driver to move where it goes next.
    Residual(TensorCat),
    /// The range reached the final layer; the head ran. One logits row per input
    /// sequence, packed as the current forward's `TensorCat`.
    Logits(TensorCat),
    /// [`SweepMode::Prepare`]: the segment's per-phase transient price —
    /// attention, FFN, forward span — with nothing placed and nothing swept.
    Priced([usize; 3]),
}

// Batched Inference Wrapper

/// Chunk size for extending RoPE tables.
const ROPE_EXTEND_CHUNK: usize = 1024;

/// Free KV regions a wave wants beyond one per sequence, before phase 0 stops
/// asking the weight side for ground.
///
/// The mirror of the weight side's own `KV_REGION_SLACK`, and the same number for
/// the same reason: a partition trimmed to exactly what the last wave used has
/// nothing for the next one to grow into, and every arena that then has to be
/// created is a region claim mid-wave — which is the claim the transient tier's
/// ceiling refuses.
///
/// The per-sequence term beside it is what makes the target a statement about
/// *this* wave rather than a constant: ground is spare only if nobody needs it,
/// and the sequence count is the best cheap estimate of who does.
#[cfg(feature = "cuda")]
const KV_FREE_SLACK_REGIONS: usize = 32;

mod inference_wrapper;
mod model_core;

pub use inference_wrapper::*;
pub use model_core::*;
