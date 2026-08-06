//! DeepSeek-V4-Flash — reference inference implementation.
//!
//! A faithful port of DeepSeek's `inference/model.py` expressed in Candle tensor ops.
//! The architecture is unusual — see `docs/deepseek_v4_flash.md` for the full design —
//! combining:
//!
//! * **Latent single-KV attention**: 64 query heads read one shared 512-dim KV vector
//!   per token (K and V are the same rows), with learned per-head attention sinks and
//!   output de-rotation of the RoPE dims.
//! * **Compressed Sparse / Heavily Compressed Attention (CSA/HCA)**: a learned
//!   `Compressor` pools consecutive tokens into compressed KV entries; CSA layers use an
//!   `Indexer` to pick the top-k compressed entries to attend to, HCA layers attend to
//!   all of them. Every layer additionally keeps a raw sliding window.
//! * **Manifold-Constrained Hyper-Connections (mHC)**: the residual stream carries
//!   `hc_mult` copies mixed by a Sinkhorn-normalized combination matrix.
//! * **MoE** with `sqrtsoftplus`/`noaux_tc` routing, hash-routed early layers, clamped
//!   SwiGLU experts, and MXFP4 routed-expert weights.
//!
//! This module is the numerically-correct reference; the batched/paged/kernel-optimized
//! path is layered on top separately.

mod attention;
mod batched;
mod compressor;
mod config;
#[cfg(feature = "cuda")]
mod engine;
mod hyper;
mod indexer;
mod linear;
mod loader;
mod moe;
mod rope;
mod streaming;
mod transformer;

pub use attention::{Attention, AttentionParams};
pub use compressor::Compressor;
pub use config::{Config, LayerKind};
pub use hyper::{HyperConnection, HyperParams};
pub use indexer::Indexer;
pub use linear::QLinear;
pub use loader::{config_from_gguf, GgufModel};
pub use moe::{Expert, Gate, MoE, ScoreFunc};
pub use rope::RotaryCache;
pub use streaming::StreamingModel;
pub use transformer::{Block, Transformer};
