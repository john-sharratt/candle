//! AOT-compiled CUDA kernels for the fork's inference engine, plus their FFI
//! bindings.
//!
//! `build.rs` invokes NVCC to compile each `.cu` source into PTX, keyed by a
//! SHA256 of the source tree so unchanged kernels are not recompiled; the
//! resulting PTX is embedded into the binary at compile time (no NVCC or
//! `.ptx` file is needed at runtime — `cargo build --features cuda` is
//! sufficient, see `make clean-ptx` to force a full rebuild). Each `pub mod`
//! below corresponds to a `src/<subdir>/` of `.cu` kernels plus an `api.rs`
//! Rust wrapper: `simple` (generic elementwise/reduce/indexing/conv ops used
//! by `candle-core::cuda_backend`), `quantized` (GGML-format quantized
//! matmul), `sampling` (batched logit-processing/sampling), `provenance`
//! (Binary Directional Provenance scan kernels for KV-chunk retrieval), and
//! `paged_decode`/`paged_prefill`/`paged_glue` (the paged, per-block-quantized
//! attention kernels backing the three-tier KV cache). `CHUNK_SIZE = 32` is
//! the shared Rust/CUDA block-size constant used throughout the paged and
//! quantized kernels.

/// Chunk size for paged attention kernels.
/// Must match CHUNK_SIZE in arena_table.cuh (compile-time constant = 32).
/// This value is used for GGML quantization alignment and fast bit-shift division.
pub const CHUNK_SIZE: i32 = 32;

pub mod simple;

#[path = "quantized/api.rs"]
pub mod quantized;

#[path = "sampling/api.rs"]
pub mod sampling;

#[path = "provenance/api.rs"]
pub mod provenance;

#[path = "paged-decode/api.rs"]
pub mod paged_decode;

#[path = "paged-prefill/api.rs"]
pub mod paged_prefill;

#[path = "paged-glue/api.rs"]
pub mod paged_glue;

#[path = "paged-latent/api.rs"]
pub mod paged_latent;
