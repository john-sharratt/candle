//! Hyperparameters of a sparse-latent MoE model.
//!
//! The fields are the family's, shared by every model in it. The *values* are a
//! particular model's, and so is every name they are stored under on disk —
//! both come from the [`Arch`] the config carries. See
//! [`models::deepseek4`](crate::models::deepseek4) for the V4-Flash values and
//! `docs/deepseek_v4_flash.md` for the design.

use super::arch::Arch;

/// The per-layer attention kind, decided by `compress_ratios[layer]`:
/// * `0`   → sliding-window-only attention (no compression, no indexer),
/// * `4`   → Compressed Sparse Attention (overlapping 4:1 compressor + indexer top-k),
/// * `128` → Heavily Compressed Attention (128:1 compressor, attends to all entries).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Sliding window only. `compress_ratio == 0`.
    SlidingWindow,
    /// Compressed Sparse Attention. `compress_ratio == 4` (overlap pooling + indexer).
    Csa,
    /// Heavily Compressed Attention. `compress_ratio == 128` (no indexer).
    Hca,
}

impl LayerKind {
    pub fn from_ratio(ratio: usize) -> Self {
        match ratio {
            0 => Self::SlidingWindow,
            4 => Self::Csa,
            _ => Self::Hca,
        }
    }

    pub fn compresses(self) -> bool {
        !matches!(self, Self::SlidingWindow)
    }

    pub fn has_indexer(self) -> bool {
        matches!(self, Self::Csa)
    }
}

/// Model hyperparameters.
///
/// Built by starting from [`Arch::defaults`] and overriding whatever the GGUF
/// metadata carries — see [`loader::config_from_gguf`](super::loader::config_from_gguf).
/// There is no `Default`: a config without an architecture could not resolve a
/// single tensor name.
#[derive(Debug, Clone)]
pub struct Config {
    /// The model this config describes. Carried here rather than passed
    /// alongside so that any code holding a `&Config` can resolve tensor names
    /// and metadata keys, and so a config can never be read with the wrong
    /// model's naming.
    pub arch: &'static dyn Arch,
    pub vocab_size: usize,
    pub dim: usize,
    pub moe_inter_dim: usize,
    pub n_layers: usize,
    pub n_hash_layers: usize,
    pub n_heads: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub n_activated_experts: usize,
    pub score_func: String,
    pub route_scale: f64,
    pub swiglu_limit: f64,
    // Attention (MLA-style single latent KV).
    pub q_lora_rank: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub norm_eps: f64,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    pub compress_ratios: Vec<usize>,
    // YaRN.
    pub compress_rope_theta: f64,
    pub original_seq_len: usize,
    pub rope_theta: f64,
    pub rope_factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    // Indexer.
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    // Hyper-connections.
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f64,
}

impl Config {
    /// Non-RoPE portion of each head (the FP8-quantized latent dims).
    pub fn nope_head_dim(&self) -> usize {
        self.head_dim - self.rope_head_dim
    }

    /// The attention kind for `layer`, from `compress_ratios`.
    pub fn layer_kind(&self, layer: usize) -> LayerKind {
        LayerKind::from_ratio(self.compress_ratio(layer))
    }

    /// The raw compression ratio for `layer` (0 when unspecified / SWA).
    pub fn compress_ratio(&self, layer: usize) -> usize {
        self.compress_ratios.get(layer).copied().unwrap_or(0)
    }

    /// True when `layer` routes experts by token id (`tid2eid`) instead of top-k scores.
    pub fn is_hash_layer(&self, layer: usize) -> bool {
        layer < self.n_hash_layers
    }

    /// The RoPE `(theta, original_seq_len)` for `layer`: compression layers use the
    /// long-context theta with YaRN; sliding-window layers use the base theta with YaRN
    /// disabled (a 128-token window never extrapolates).
    pub fn rope_params(&self, layer: usize) -> (f64, usize) {
        if self.layer_kind(layer).compresses() {
            (self.compress_rope_theta, self.original_seq_len)
        } else {
            (self.rope_theta, 0)
        }
    }

    /// A minimal synthetic config for the engine's unit tests: a handful of tiny
    /// layers exercising every layer kind, with small dims so tensors stay cheap.
    /// Model-independent by construction — see
    /// [`test_arch`](super::arch::test_arch).
    #[cfg(test)]
    pub fn tiny() -> Self {
        use super::arch::test_arch::TEST_ARCH;
        TEST_ARCH.defaults()
    }
}
