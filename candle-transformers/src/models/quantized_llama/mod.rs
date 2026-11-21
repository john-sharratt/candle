//! The quantized Llama model: its layers, weights, RoPE tables and tests.
//!
//! The model is a quantized Llama decoder that supports 2-, 3-, 4- and 8-bit
//! weights and the Llama 2 and Llama 3 rope schemes. Its RoPE tables start empty
//! and grow in chunks on demand, so no up-front allocation is made. The frequencies
//! come from `rope_freqs.weight` first, because llama.cpp writes Llama 3 scaling
//! only as that tensor, and from the scaling keys only when the file has no such
//! tensor.
//!
//! - `layer`: the per-layer weights and forward paths.
//! - `model`: GGUF loading and the batched and sequential forward passes.
//! - `tests_forward`: clone independence and flash-attention tests.
//! - `tests_gates`: the Llama acceptance gates.
//! - `tests_kv_dump`: KV-cache dump tooling and the R16 versus F16 comparison.

use std::sync::{Arc, RwLock};

#[cfg(feature = "cuda")]
use super::batched_layer::{add_ffn_residual, BatchedAttentionLayer, QkvProjection};
#[cfg(feature = "cuda")]
use super::batched_model::{BatchedModelCore, WaveShapes};
use super::dense_span;
use super::kv_cache_utils::{new_kv_caches, KvCaches, SequenceContext};
use super::profile::gpu_span_phase;
use super::quantized_mlp::QuantizedMlp;
use super::rope_schedule::{from_rope_freqs, llama_inv_freq};
use super::rope_tables::CisPrecomputations;
use super::{decode_utils, quantized_matmul::QMatMul};
use crate::models::batched_layer::WaveRef;
use crate::models::llama::{Llama3RopeConfig, Llama3RopeType};
#[allow(unused_imports)]
use crate::models::mmdit::blocks::SelfAttnDiTBlock;
use crate::models::wave_buffers::wave_root;
use crate::quantized_nn::RmsNorm;
#[cfg(feature = "cuda")]
use candle::quantized::cuda::DynamicActs;
#[cfg(feature = "cuda")]
use candle::quantized::decode_rows::DecodeRows;
#[cfg(feature = "cuda")]
use candle::quantized::register_mmap_cuda;
use candle::quantized::QTensor;
use candle::quantized::{ggml_file, gguf_file, Int8Mode};
use candle::{DType, Device, IndexOp, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{WaveGeneration, LLAMA2_KV_FACTOR, LLAMA3_KV_FACTOR, LLAMA_KV_FACTORS};
use candle_nn::{kv_cache::KvCache, Embedding, Module};

/// Initial number of RoPE positions to precompute for quantized llama models.
///
/// The tables can be extended on demand in chunks (see `ROPE_EXTEND_CHUNK`) to
/// keep RoPE lookup fast (index_select/narrow into a contiguous table) while
/// avoiding a hard cap.
///
/// We default to 0 to avoid any up-front RoPE allocation; the tables will grow
/// on demand.
pub const MAX_ROPE_SEQ_LEN: usize = 0;

/// When extending RoPE tables, grow them in this many positions at a time.
pub const ROPE_EXTEND_CHUNK: usize = 1024;

type SharedCis = Arc<RwLock<CisPrecomputations>>;

/// The tensor in which llama.cpp's converter writes Llama3 RoPE scaling: one
/// divisor per rotary pair.
const ROPE_FREQS: &str = "rope_freqs.weight";

/// The RoPE frequencies this file states.
///
/// **`rope_freqs.weight` first.** llama.cpp writes Llama3 scaling only as that
/// tensor, never as metadata keys, so a loader reading the keys alone runs a
/// Llama3-scaled model on plain RoPE — wrong at every position, since the
/// scaling divides the low-frequency pairs everywhere. Without the tensor, the
/// scaling keys, when all four are present, and plain RoPE otherwise.
fn stated_inv_freq(
    rope_dim: usize,
    freq_base: f32,
    scaling_keys: Option<Llama3RopeConfig>,
    rope_freqs: Option<QTensor>,
) -> Result<Vec<f32>> {
    match rope_freqs {
        Some(t) => {
            let factors = t
                .dequantize(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            from_rope_freqs(rope_dim, freq_base, &factors)
        }
        None => Ok(llama_inv_freq(rope_dim, freq_base, scaling_keys)),
    }
}

mod layer;
mod model;
#[cfg(test)]
mod tests_forward;
#[cfg(test)]
mod tests_gates;
#[cfg(test)]
mod tests_kv_dump;

pub use layer::*;
pub use model::*;
