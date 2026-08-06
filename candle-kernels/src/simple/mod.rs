//! FFI bindings for simple CUDA kernels.
//!
//! Each submodule provides dispatcher functions with enum-based type selection,
//! reducing 100s of FFI bindings to clean single-function APIs.
//!
//! ## Example
//! ```rust,ignore
//! use candle_kernels::simple::binary::{run_binary_op, BinaryOp, BinaryDType};
//! run_binary_op(BinaryOp::Add as i32, BinaryDType::F32 as i32, ...);
//! ```

// Core operations
pub mod affine;
pub mod binary;
pub mod cast;
pub mod conv;
pub mod fast_exp;
pub mod fill;
pub mod indexing;
pub mod reduce;
pub mod sort;
pub mod ternary;
pub mod unary;

// Fused activation kernels
pub mod fused_silu_mul;

// Sampling and misc operations
pub mod multinomial;
pub mod quantized;
pub mod repeat_penalty;

// Index mutation ops
pub mod add_at_indices;
pub mod div_at_indices;
pub mod mul_at_indices;
pub mod sub_at_indices;
pub mod sub_at_indices_with_values;

// Unified scatter operation dispatcher (combines add/sub/mul/div_at_indices)
pub mod scatter_op;

// Fused MoE gather and weighted scatter-add kernels
pub mod moe_bucketize;
pub mod moe_scatter;

// R16 KV gather: single-kernel replacement for per-chunk memcpy_dtov
pub mod gather_r16_kv;

// Provenance sign(Q) bit-pack: GPU read+sign+pack, one launch for all layers
pub mod prov_sign_pack;

// KV tier-migration scatter/gather (kv_pack / kv_unpack primitive)
pub mod kv_migrate;

// Fletcher-32 KV-chunk golden checksum: GPU-side integrity hash, one launch for
// a whole plan of quantized chunks, leaving the KV data on the device
pub mod fletcher32;

// Fused Sinkhorn (doubly-stochastic) normalization of the mHC combine matrix:
// one launch replaces the ~120 tiny host-orchestrated ops per sub-block per layer
pub mod deepseek_bdp;
pub mod sinkhorn;
