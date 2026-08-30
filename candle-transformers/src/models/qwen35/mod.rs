//! The `qwen35` arch lineage — hybrid Gated DeltaNet + gated-attention models.
//!
//! llama.cpp names this architecture family by its first release: the
//! `qwen35` / `qwen35moe` arch strings cover Qwen3.5, the Qwen3.6 point
//! release, and Qwen3.8 (the way `llama` covers every Llama generation).
//! This module is the lineage's shared implementation; the per-model files —
//! `models/quantized_qwen35.rs`, `models/quantized_qwen35_moe.rs`, and their
//! 3.6/3.8 siblings — pin checkpoints and hold the gates.
//!
//! Two implementations of one model (`docs/qwen35_qwen38_models.md` §7):
//!
//! * the **reference** — configuration, GGUF schema, and a
//!   numerically-transparent F32 stack (`attention`, `moe`, `model`,
//!   `loader`), validated token-for-token against llama.cpp;
//! * the **production** path — quantized projections on the GPU
//!   (`quantized_weights` and the per-layer-kind drivers), which reuses the
//!   generic DeltaNet subsystem (`crate::models::delta_net`) rather than
//!   restating its algebra.
//!
//! The reference is the oracle the production path is tested against, and
//! llama.cpp is the oracle the reference is tested against.

pub mod attention;
#[cfg(feature = "cuda")]
pub mod batched;
pub mod config;
#[cfg(feature = "cuda")]
pub mod draft;
pub mod embedding;
pub mod engine;
pub mod expert_loader;
#[cfg(feature = "cuda")]
pub mod forward;
/// Standing up the layer cache for a dense checkpoint larger than the card.
#[cfg(feature = "cuda")]
pub mod layer_loader;
pub mod layer_store;
pub mod loader;
pub mod model;
pub mod moe;
pub mod mtp;
#[cfg(feature = "cuda")]
pub mod quantized_attention;
pub mod quantized_delta_net;
#[cfg(feature = "cuda")]
pub mod quantized_loader;
#[cfg(feature = "cuda")]
pub mod quantized_moe;
pub mod quantized_weights;
pub mod spec;

/// Tier-2 gates for the recurrent-state hooks — inside the lib because they
/// drive `batch_test`, which is `cfg(test)` on this crate.
#[cfg(all(test, feature = "cuda"))]
mod recurrent_gates;
#[cfg(feature = "cuda")]
pub mod wave;

pub use config::{MoeConfig, Qwen35Config};
pub use quantized_weights::{load_quantized_model, QuantLayerMix, QuantModel};

#[cfg(feature = "cuda")]
pub use batched::HybridBatched;
#[cfg(feature = "cuda")]
pub use quantized_loader::{load_hybrid_gguf, Qwen35LoadOptions};
