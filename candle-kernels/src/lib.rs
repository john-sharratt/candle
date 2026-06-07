//! candle-kernels: AOT-compiled CUDA kernels for candle
//!
//! This crate provides FFI bindings to precompiled CUDA kernels.

/// Chunk size for paged attention kernels.
/// Must match CHUNK_SIZE in arena_table.cuh (compile-time constant = 32).
/// This value is used for GGML quantization alignment and fast bit-shift division.
pub const CHUNK_SIZE: i32 = 32;

pub mod simple;

#[path = "quantized/api.rs"]
pub mod quantized;

#[path = "sampling/api.rs"]
pub mod sampling;

#[path = "paged-decode/api.rs"]
pub mod paged_decode;

#[path = "paged-prefill/api.rs"]
pub mod paged_prefill;

#[path = "fused-attn-v1/api.rs"]
pub mod fused_attn_v1;
