//! The `qwen4exp` architecture — Qwen3.8-Flash-Next, Qwen's preview of the
//! Qwen4 generation.
//!
//! A hybrid 3:1 Gated-DeltaNet / sparse-attention stack over a 4-stream gated
//! residual (hyper-connections in place of layer norms), with a 512-expert
//! MoE on every layer, an n-gram hash embedding injected at layer 1 (PLE),
//! and block-sparse attention (QSA) on the 12 full-attention layers. Design
//! doc: `docs/qwen38_flash_next.md`; the frozen schema and reference algebra
//! are its §12, taken from the actual GGUF and llama.cpp's `qwen4exp.cpp`.
//!
//! This module is the **reference implementation** — the `forward_batched`
//! oracle every later path (paged, int8, wave-batched) is diffed against. It
//! reuses the lineage's shared subsystems rather than restating them: the
//! GDN mixer from `delta_net` (with the sigmoid z-gate this generation
//! switched to), the gated-attention core and rope tables from `qwen35`, the
//! MoE routing reference from `qwen35::moe`, and the split-GGUF reader from
//! `latent_moe`. What is genuinely new — the hyper-connection algebra, PLE,
//! and QSA selection — lives in its own file each.
//!
//! The model file (checkpoint pins, the gates) is
//! `models/quantized_qwen38_moe.rs`, named beside its `_moe` siblings —
//! `quantized_qwen38.rs` is the *other* Qwen3.8, the 27B dense.

#[cfg(feature = "cuda")]
pub mod batched_attention;
pub mod config;
pub mod convert;
pub mod convert_bench;
pub mod draft;
#[cfg(feature = "cuda")]
pub mod engine;
pub mod hyper;
#[cfg(feature = "cuda")]
pub mod indexer;
pub mod loader;
pub mod model;
/// The NextN / MTP draft head — the block past the trunk, and the input
/// assembly that feeds it.
pub mod mtp;
pub mod ple;
pub mod ple_cache;
pub mod qsa;
pub mod qsa_select;
pub mod spec;
#[cfg(feature = "cuda")]
pub mod wave;

pub use config::Qwen4ExpConfig;
#[cfg(feature = "cuda")]
pub use engine::Qwen4ExpGpu;
pub use loader::load_oracle_model;
pub use model::{Qwen4ExpModel, SessionState};
#[cfg(feature = "cuda")]
pub use wave::Qwen4ExpBatched;
