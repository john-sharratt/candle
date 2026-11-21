//! Builds the rotary position embedding that the model applies to queries and
//! keys.
//!
//! The tables are precomputed for an initial number of positions and grow in
//! fixed chunks as longer sequences arrive. `RotaryEmbedding::new` applies only a
//! scaling that the model file declares, so the frequencies are the file's own.

use super::*;

/// Initial number of RoPE positions to precompute.
pub const MAX_ROPE_SEQ_LEN: usize = 0;

/// When extending RoPE tables, grow them in this many positions at a time.
pub const ROPE_EXTEND_CHUNK: usize = 1024;

pub(super) type SharedCis = Arc<RwLock<CisPrecomputations>>;

// Rotary Embeddings (copied from quantized_qwen3)

#[derive(Debug, Clone)]
pub(super) struct RotaryEmbedding {
    pub(super) cis: SharedCis,
}

impl RotaryEmbedding {
    /// The file's own RoPE: only a scaling it declares
    /// (`docs/progressive_yarn.md` §2).
    pub(super) fn new(
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
}
