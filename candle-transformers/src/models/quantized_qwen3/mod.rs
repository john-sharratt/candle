//! Qwen3 with quantized weights, loaded from GGUF and run through the batched
//! inference path.
//!
//! The implementation follows the Qwen3 architecture and reads its RoPE schedule from
//! the file's own declaration only. A declared context beyond the trained window does
//! not by itself imply a scaling factor, since Qwen3's released files declare a longer
//! window than they were trained on, and a scaling inferred from that ratio would
//! compress every position. Rotary tables start empty and grow in fixed steps as
//! positions are reached.
//!
//! - `layers`: one decoder layer, with separate query, key and value projections.
//! - `model_weights`: GGUF loading, the resident footprint, and the forward passes.
//!
//! Tests:
//!
//! - `tests_gate`: checkpoint benchmark, fidelity and ruler gates.

// OLD: use super::batched_inference::{BatchedInferenceSession, ManagedBatchedModel as BatchableModel};
#[cfg(feature = "cuda")]
use super::batched_layer::{add_ffn_residual, BatchedAttentionLayer, QkvProjection};
#[cfg(feature = "cuda")]
use super::batched_model::{BatchedModelCore, WaveShapes};
use super::dense_span;
use super::kv_cache_utils::{new_kv_caches, KvCaches, SequenceContext};
use super::quantized_matmul::QMatMul;
use super::quantized_mlp::QuantizedMlp;
use super::rope_schedule::DeclaredScaling;
use super::rope_tables::CisPrecomputations;
use crate::models::batched_layer::WaveRef;
use crate::models::wave_buffers::wave_root;
use crate::{quantized_nn::RmsNorm, utils::repeat_kv};
#[cfg(feature = "cuda")]
use candle::quantized::cuda::DynamicActs;
#[cfg(feature = "cuda")]
use candle::quantized::decode_rows::DecodeRows;
#[cfg(feature = "cuda")]
use candle::quantized::register_mmap_cuda;
use candle::quantized::{gguf_file, Int8Mode, QTensor};
use candle::LiveTensor;
use candle::{DType, Device, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::WaveGeneration;
use candle_nn::{kv_cache::KvCache, Embedding, Module};
use std::io::{Read, Seek};
use std::sync::{Arc, RwLock};

/// Initial number of RoPE positions to precompute for quantized Qwen3 models.
///
/// Defaulting to 0 avoids up-front RoPE allocation; tables are extended on demand.
pub const MAX_ROPE_SEQ_LEN: usize = 0;

/// When extending RoPE tables, grow them in this many positions at a time.
pub const ROPE_EXTEND_CHUNK: usize = 1024;

type SharedCis = Arc<RwLock<CisPrecomputations>>;

struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        QMatMul::from_weights(ws.into())
    }

    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cis: SharedCis,
}

impl RotaryEmbedding {
    /// The file's own RoPE: only a scaling it declares, never one inferred —
    /// a `context_length` beyond the trained window is not a factor. Qwen3's
    /// released files declare 40,960 while training at 32,768, and a factor
    /// inferred from that ratio compressed every position 1.25× at every
    /// length. Qwen3 extends by YaRN, per slot, from its preset's schedule
    /// (`docs/progressive_yarn.md` §3.1).
    fn new(
        _dtype: DType,
        head_dim: usize,
        _max_position_embeddings: usize,
        rope_theta: f64,
        declared: DeclaredScaling,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq = declared.inv_freq(head_dim, rope_theta as f32)?;
        Ok(Self {
            cis: Arc::new(RwLock::new(CisPrecomputations::new_growable_with_inv_freq(
                inv_freq,
                MAX_ROPE_SEQ_LEN,
                ROPE_EXTEND_CHUNK,
                dev,
            )?)),
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let (cos, sin) = {
            let mut cis = self
                .cis
                .write()
                .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
            cis.narrow_growable(0, offset, seq_len, q.dtype())?
        };
        let q = if q.is_contiguous() {
            q.clone()
        } else {
            q.contiguous()?
        };
        let k = if k.is_contiguous() {
            k.clone()
        } else {
            k.contiguous()?
        };
        let q_embed = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

mod layers;
mod model_weights;
#[cfg(test)]
mod tests_gate;

pub use layers::*;
pub use model_weights::*;
