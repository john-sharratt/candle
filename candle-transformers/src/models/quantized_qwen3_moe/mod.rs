//! The Qwen3 MoE model: quantised GGUF weights with an expert cache, implementing
//! the batched-model traits the conversation engine drives.
//!
//! Non-expert weights live in VRAM. Expert weights sit in an LRU cache in VRAM,
//! with cold storage in the memory-mapped file. The model reads both the merged
//! 3D expert tensors and the 2D per-expert layout, the second as a fallback. The
//! batched core and attention traits are implemented here for the conversation
//! engine.
//!
//! - `layer`: attention and feed-forward halves of a layer.
//! - `loader`: loading from the model pack.
//! - `model`: the forward ladder and the weight footprint.
//! - `moe`: the routed MoE block.
//! - `rope`: the rotary embedding.
//! - `tests_gates`: model-level gate tests.
//! - `tests_kv_dump`: KV cache dump.
//! - `tests_profile`: decode profiling and routing capture.
//! - `tests_wave`: wave-engine equivalence tests.

#[cfg(feature = "cuda")]
use super::batched_layer::{add_ffn_residual, BatchedAttentionLayer, QkvProjection};
#[cfg(feature = "cuda")]
use super::batched_model::{BatchedModelCore, WaveShapes};
use super::dense_span;
#[cfg(feature = "cuda")]
use super::expert_lre::section::{geometries_of, open_section};
#[cfg(feature = "cuda")]
use super::expert_lre::ExpertCacheSetup;
#[cfg(feature = "cuda")]
use super::expert_lre::{minimum_resident_slots, slot_bytes_for};
use super::expert_lre::{ExpertCache, LookAhead, PipelineStats, ProfileSnapshot};
use super::kv_cache_utils::{new_kv_caches, KvCaches};
use super::model_pack::ModelPack;
#[cfg(feature = "cuda")]
use super::profile::{pipeline_record, profile_now};
use super::quantized_matmul::QMatMul;
use super::quantized_mlp::QuantizedMlp;
use super::rope_schedule::DeclaredScaling;
use super::rope_tables::CisPrecomputations;
// `batch_test` is itself gated on `cuda` plus `test`-or-`ruler-bench`, so the ladder
// expressed in its types carries the same gate. `candle-conversation` depends on this
// crate with `ruler-bench` on, which is how it reaches the ladder.
#[cfg(all(feature = "cuda", any(test, feature = "ruler-bench")))]
use crate::models::batch_test::utils::{TestConfig, TestMode};
#[cfg(all(feature = "cuda", any(test, feature = "ruler-bench")))]
use crate::models::batched_inference::InferenceMode;
use crate::models::batched_layer::WaveRef;
use crate::models::load_phase::{LoadPhase, LoadProgress};
use crate::models::routing_capture;
use crate::models::wave_buffers::wave_root;
use crate::quantized_nn::RmsNorm;
#[cfg(feature = "cuda")]
use candle::quantized::cuda::{moe_route, to_dynamic, DynamicActs};
use candle::quantized::decode_rows::DecodeRows;
#[cfg(feature = "cuda")]
use candle::quantized::get_vram_info;
use candle::quantized::{gguf_file, Int8Mode, QTensor, SumScale};
use candle::LiveTensor;
use candle::{DType, Device, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::WaveGeneration;
use candle_nn::kv_cache::WeightZone;
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{
    initial_weight_bytes, set_weight_floor, span_end, weight_capacity_bytes,
};
use candle_nn::{kv_cache::KvCache, Embedding, Module};
#[cfg(feature = "cuda")]
use std::collections::BTreeMap;
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex, RwLock};

// Model Weights

pub struct ModelWeights {
    /// Resident embedding table. `None` when [`Self::host_embedding`] serves it
    /// from pinned host memory instead — exactly one of the two is populated.
    embeddings: Option<Embedding>,
    /// Embedding table copied into pinned host memory and gathered per forward, chosen at
    /// load when the table is large relative to the card (see
    /// [`crate::models::host_embedding`]).
    #[cfg(feature = "cuda")]
    host_embedding: Option<crate::models::host_embedding::HostEmbedding>,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    #[allow(dead_code)]
    expert_cache: Option<Arc<ExpertCache>>,
    device: Device,
    /// Inference numeric mode for the dense (non-expert) projections. Baked into each dense
    /// `QMatMul` at load; retained here for introspection. Experts are unaffected (always FP16).
    #[allow(dead_code)]
    int8mode: Int8Mode,
    /// Fixed VRAM bytes held by the **base** (non-expert) weights — embeddings,
    /// attention projections, norms, router, lm_head. Measured at load as the
    /// driver-used delta across weight loading, minus the resident-expert
    /// footprint. Added to the live resident-expert gauge to report the model's
    /// total (time-varying) weight VRAM for the whole-card decomposition. `0` on
    /// non-CUDA.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    base_weight_bytes: usize,
    /// Shapes the transient tier is priced from, read from the checkpoint config.
    wave_shapes: WaveShapes,
}

// ============================================================================
// Tests
// ============================================================================

mod layer;
mod loader;
mod model;
mod moe;
mod rope;
#[cfg(test)]
mod tests_gates;
#[cfg(test)]
mod tests_kv_dump;
#[cfg(test)]
mod tests_profile;
#[cfg(test)]
mod tests_wave;

pub use layer::*;
#[cfg(all(feature = "cuda", any(test, feature = "ruler-bench")))]
pub use model::*;
pub(crate) use moe::*;
pub use rope::*;
