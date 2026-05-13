//! Quantized matmul integration tests.
//!
//! These tests verify that the GEMX tensor-core kernels produce results
//! matching the baseline GGML dequantize+matmul path.

mod quantized;
