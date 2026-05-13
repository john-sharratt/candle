//! Tests for quantized matmul kernels.
//!
//! These tests verify that the GEMX tensor-core kernels produce results
//! matching the baseline GGML dequantize+matmul path.

pub mod common;

// Simple quants (block size 32)
mod q4_0;
mod q4_1;
mod q5_0;
mod q5_1;
mod q8_0;
mod q8_1;

// K-quants (block size 256)
mod q2_k;
mod q3_k;
mod q4_k;
mod q5_k;
mod q6_k;
mod q8_k;

// AWQ (Activation-aware Weight Quantization)
mod q_awq;
