//! DeepSeek-V4-Flash model configuration.
//!
//! Field names and defaults mirror the reference `inference/config.json` and
//! `ModelArgs` in `inference/model.py`. See `docs/deepseek_v4_flash.md`.

use serde::Deserialize;

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

/// Model hyperparameters. Defaults are the tiny debug config from `model.py`'s
/// `ModelArgs`; the real model overrides every field from GGUF metadata / config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    #[serde(default = "d_dim", alias = "hidden_size")]
    pub dim: usize,
    #[serde(default = "d_moe_inter", alias = "moe_intermediate_size")]
    pub moe_inter_dim: usize,
    #[serde(default = "d_layers", alias = "num_hidden_layers")]
    pub n_layers: usize,
    #[serde(default, alias = "num_hash_layers")]
    pub n_hash_layers: usize,
    #[serde(default = "d_heads", alias = "num_attention_heads")]
    pub n_heads: usize,
    #[serde(default = "d_routed", alias = "n_routed_experts")]
    pub n_routed_experts: usize,
    #[serde(default = "d_shared", alias = "n_shared_experts")]
    pub n_shared_experts: usize,
    #[serde(default = "d_activated", alias = "num_experts_per_tok")]
    pub n_activated_experts: usize,
    #[serde(default = "d_score_func", alias = "scoring_func")]
    pub score_func: String,
    #[serde(default = "d_route_scale", alias = "routed_scaling_factor")]
    pub route_scale: f64,
    #[serde(default = "d_swiglu_limit")]
    pub swiglu_limit: f64,
    // Attention (MLA-style single latent KV).
    #[serde(default = "d_q_lora")]
    pub q_lora_rank: usize,
    #[serde(default = "d_head_dim")]
    pub head_dim: usize,
    #[serde(default = "d_rope_head_dim", alias = "qk_rope_head_dim")]
    pub rope_head_dim: usize,
    #[serde(default = "d_eps", alias = "rms_norm_eps")]
    pub norm_eps: f64,
    #[serde(default = "d_o_groups")]
    pub o_groups: usize,
    #[serde(default = "d_o_lora")]
    pub o_lora_rank: usize,
    #[serde(default = "d_window", alias = "sliding_window")]
    pub window_size: usize,
    #[serde(default)]
    pub compress_ratios: Vec<usize>,
    // YaRN.
    #[serde(default = "d_compress_theta", alias = "compress_rope_theta")]
    pub compress_rope_theta: f64,
    #[serde(default, alias = "original_max_position_embeddings")]
    pub original_seq_len: usize,
    #[serde(default = "d_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "d_rope_factor")]
    pub rope_factor: f64,
    #[serde(default = "d_beta_fast")]
    pub beta_fast: f64,
    #[serde(default = "d_beta_slow")]
    pub beta_slow: f64,
    // Indexer.
    #[serde(default = "d_index_heads", alias = "index_n_heads")]
    pub index_n_heads: usize,
    #[serde(default = "d_index_head_dim", alias = "index_head_dim")]
    pub index_head_dim: usize,
    #[serde(default = "d_index_topk", alias = "index_topk")]
    pub index_topk: usize,
    // Hyper-connections.
    #[serde(default = "d_hc_mult")]
    pub hc_mult: usize,
    #[serde(default = "d_hc_sinkhorn")]
    pub hc_sinkhorn_iters: usize,
    #[serde(default = "d_hc_eps")]
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

    /// A minimal synthetic config for unit tests: a handful of tiny layers exercising
    /// every layer kind, with small dims so tensors stay cheap.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 128,
            dim: 64,
            moe_inter_dim: 48,
            n_layers: 6,
            n_hash_layers: 1,
            n_heads: 4,
            n_routed_experts: 8,
            n_shared_experts: 1,
            n_activated_experts: 2,
            score_func: "sqrtsoftplus".to_string(),
            route_scale: 1.5,
            swiglu_limit: 10.0,
            q_lora_rank: 32,
            head_dim: 32,
            rope_head_dim: 8,
            norm_eps: 1e-6,
            o_groups: 2,
            o_lora_rank: 24,
            window_size: 8,
            // SWA, SWA, CSA, HCA, CSA, HCA
            compress_ratios: vec![0, 0, 4, 128, 4, 128],
            compress_rope_theta: 160000.0,
            original_seq_len: 64,
            rope_theta: 10000.0,
            rope_factor: 16.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            index_n_heads: 4,
            index_head_dim: 16,
            index_topk: 8,
            hc_mult: 4,
            hc_sinkhorn_iters: 20,
            hc_eps: 1e-6,
        }
    }
}

fn d_vocab() -> usize {
    129280
}
fn d_dim() -> usize {
    4096
}
fn d_moe_inter() -> usize {
    2048
}
fn d_layers() -> usize {
    43
}
fn d_heads() -> usize {
    64
}
fn d_routed() -> usize {
    256
}
fn d_shared() -> usize {
    1
}
fn d_activated() -> usize {
    6
}
fn d_score_func() -> String {
    "sqrtsoftplus".to_string()
}
fn d_route_scale() -> f64 {
    1.5
}
fn d_swiglu_limit() -> f64 {
    10.0
}
fn d_q_lora() -> usize {
    1024
}
fn d_head_dim() -> usize {
    512
}
fn d_rope_head_dim() -> usize {
    64
}
fn d_eps() -> f64 {
    1e-6
}
fn d_o_groups() -> usize {
    8
}
fn d_o_lora() -> usize {
    1024
}
fn d_window() -> usize {
    128
}
fn d_compress_theta() -> f64 {
    160000.0
}
fn d_rope_theta() -> f64 {
    10000.0
}
fn d_rope_factor() -> f64 {
    16.0
}
fn d_beta_fast() -> f64 {
    32.0
}
fn d_beta_slow() -> f64 {
    1.0
}
fn d_index_heads() -> usize {
    64
}
fn d_index_head_dim() -> usize {
    128
}
fn d_index_topk() -> usize {
    512
}
fn d_hc_mult() -> usize {
    4
}
fn d_hc_sinkhorn() -> usize {
    20
}
fn d_hc_eps() -> f64 {
    1e-6
}
