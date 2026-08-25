//! Sparse-latent MoE inference engine — the architecture family shared by
//! DeepSeek-V4-Flash and any successor built on the same shape.
//!
//! This module is the **machinery**: the layer implementations, the paged/batched
//! kernel path, the provenance gallery, and the wave engine. It carries no model
//! version. A concrete model supplies its geometry, config defaults, GGUF metadata
//! keys, and tensor names through the [`Arch`](arch::Arch) trait — see
//! [`models::deepseek4`](crate::models::deepseek4) for the V4-Flash instantiation
//! and `docs/deepseek_v4_flash.md` for the design it was derived from.
//!
//! The family is defined by four properties:
//!
//! * **Latent single-KV attention**: every query head reads one shared KV vector per
//!   token (K and V are the same rows), with learned per-head attention sinks and
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
//! Within the module, `transformer` is the numerically-correct eager reference and
//! `wave`/`engine` are the batched/paged/kernel-optimized path layered on top.

pub mod arch;
mod attention;
#[cfg(feature = "cuda")]
pub mod bench;
#[cfg(feature = "cuda")]
mod comp_idx;
mod compressor;
mod config;
#[cfg(feature = "cuda")]
mod desc;
mod dspark;
mod dspark_experts;
#[cfg(feature = "cuda")]
mod engine;
mod footprint;
mod gallery;
pub mod geometry;
mod hyper;
mod indexer;
#[cfg(feature = "cuda")]
mod kernel_attention;
mod linear;
mod loader;
mod moe;
mod paged;
pub mod readback;
mod rope;
#[cfg(feature = "cuda")]
mod scatter;
#[cfg(feature = "cuda")]
pub mod select_bench;
mod streaming;
mod transformer;
#[cfg(feature = "cuda")]
mod wave;

#[cfg(feature = "cuda")]
pub use engine::Engine;

pub use arch::{Arch, Global, Meta, Weight};
pub use attention::{Attention, AttentionParams};
pub use compressor::Compressor;
pub use config::{Config, LayerKind};
#[cfg(feature = "cuda")]
pub use dspark::DsparkDrafter;
pub use dspark::{ConfidenceHead, DsparkConfig, MarkovHead};
pub use footprint::{
    deepseek_kv_footprint, fp16_linear_baseline_bytes, ratio_vs_fp16_linear, KvFootprint,
    CORPUS_DTYPE, WINDOW_KV_DTYPE,
};
#[cfg(feature = "cuda")]
pub use gallery::{bdp_recall, sign_pack};
pub use gallery::{CorpusSnapshot, FloatGallery};
pub use geometry::LatentGeometry;
pub use hyper::{HyperConnection, HyperParams};
pub use indexer::Indexer;
pub use linear::QLinear;
pub use loader::{config_from_gguf, GgufModel};
pub use moe::{Expert, Gate, MoE, ScoreFunc};
#[cfg(feature = "cuda")]
pub use paged::{paged_latent_decode, paged_latent_decode_raw, CorpusCache, SyntheticSlots};
pub use rope::yarn_freqs;
pub use rope::RotaryCache;
pub use streaming::StreamingModel;
pub use transformer::{Block, Transformer};
#[cfg(feature = "cuda")]
pub use wave::{BatchedEngine, WindowRingLayer, WindowRingSnapshot};
