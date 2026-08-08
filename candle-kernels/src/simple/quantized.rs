//! FFI bindings for CUDA quantization kernels.
//!
//! Kernels adapted from llama.cpp ggml-cuda.cu
//! https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda.cu
//!
//! Note: These symbols are linked via libkernels.a (see build.rs), not a separate library.

use std::ffi::c_int;
use std::ffi::c_void;

// Extern declarations for dequantize block kernels (K-quants without k parameter)
extern "C" {
    // -------------------------------------------------------------------------
    // Dequantize block kernels (K-quants) - no k parameter
    // -------------------------------------------------------------------------

    /// Dequantize Q2_K block to f32
    pub fn dequantize_block_q2_K_f32(vx: *const c_void, y: *mut f32);

    /// Dequantize Q2_K block to f16
    pub fn dequantize_block_q2_K_f16(vx: *const c_void, y: *mut c_void);

    /// Dequantize Q3_K block to f32
    pub fn dequantize_block_q3_K_f32(vx: *const c_void, y: *mut f32);

    /// Dequantize Q3_K block to f16
    pub fn dequantize_block_q3_K_f16(vx: *const c_void, y: *mut c_void);

    /// Dequantize Q4_K block to f32
    pub fn dequantize_block_q4_K_f32(vx: *const c_void, y: *mut f32);

    /// Dequantize Q4_K block to f16
    pub fn dequantize_block_q4_K_f16(vx: *const c_void, y: *mut c_void);

    /// Dequantize Q5_K block to f32
    pub fn dequantize_block_q5_K_f32(vx: *const c_void, y: *mut f32);

    /// Dequantize Q5_K block to f16
    pub fn dequantize_block_q5_K_f16(vx: *const c_void, y: *mut c_void);

    /// Dequantize Q6_K block to f32
    pub fn dequantize_block_q6_K_f32(vx: *const c_void, y: *mut f32);

    /// Dequantize Q6_K block to f16
    pub fn dequantize_block_q6_K_f16(vx: *const c_void, y: *mut c_void);

    /// Dequantize Q2_K block to bf16
    pub fn dequantize_block_q2_K_bf16(vx: *const c_void, y: *mut c_void);

    /// Dequantize Q3_K block to bf16
    pub fn dequantize_block_q3_K_bf16(vx: *const c_void, y: *mut c_void);

    /// Dequantize Q4_K block to bf16
    pub fn dequantize_block_q4_K_bf16(vx: *const c_void, y: *mut c_void);

    /// Dequantize Q5_K block to bf16
    pub fn dequantize_block_q5_K_bf16(vx: *const c_void, y: *mut c_void);

    /// Dequantize Q6_K block to bf16
    pub fn dequantize_block_q6_K_bf16(vx: *const c_void, y: *mut c_void);

    // -------------------------------------------------------------------------
    // Dequantize block kernels (basic quants) - with k parameter
    // -------------------------------------------------------------------------

    /// Dequantize Q4_0 block to f32
    pub fn dequantize_block_q4_0_f32(vx: *const c_void, y: *mut f32, k: c_int);

    /// Dequantize Q4_0 block to f16
    pub fn dequantize_block_q4_0_f16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q4_0 block to bf16
    pub fn dequantize_block_q4_0_bf16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q4_1 block to f32
    pub fn dequantize_block_q4_1_f32(vx: *const c_void, y: *mut f32, k: c_int);

    /// Dequantize Q4_1 block to f16
    pub fn dequantize_block_q4_1_f16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q4_1 block to bf16
    pub fn dequantize_block_q4_1_bf16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q5_0 block to f32
    pub fn dequantize_block_q5_0_f32(vx: *const c_void, y: *mut f32, k: c_int);

    /// Dequantize Q5_0 block to f16
    pub fn dequantize_block_q5_0_f16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q5_0 block to bf16
    pub fn dequantize_block_q5_0_bf16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q5_1 block to f32
    pub fn dequantize_block_q5_1_f32(vx: *const c_void, y: *mut f32, k: c_int);

    /// Dequantize Q5_1 block to f16
    pub fn dequantize_block_q5_1_f16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q5_1 block to bf16
    pub fn dequantize_block_q5_1_bf16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q8_0 block to f32
    pub fn dequantize_block_q8_0_f32(vx: *const c_void, y: *mut f32, k: c_int);

    /// Dequantize Q8_0 block to f16
    pub fn dequantize_block_q8_0_f16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q8_0 block to bf16
    pub fn dequantize_block_q8_0_bf16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q8_1 block to f32
    pub fn dequantize_block_q8_1_f32(vx: *const c_void, y: *mut f32, k: c_int);

    /// Dequantize Q8_1 block to f16
    pub fn dequantize_block_q8_1_f16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q8_1 block to bf16
    pub fn dequantize_block_q8_1_bf16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q4_KS block (4-bit with attention-sink sub-block scaling) to f32
    pub fn dequantize_block_q4_ks_f32(vx: *const c_void, y: *mut f32, k: c_int);

    /// Dequantize Q4_KS block to f16
    pub fn dequantize_block_q4_ks_f16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q4_KS block to bf16
    pub fn dequantize_block_q4_ks_bf16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q8_KS block (8-bit with attention-sink sub-block scaling) to f32
    pub fn dequantize_block_q8_ks_f32(vx: *const c_void, y: *mut f32, k: c_int);

    /// Dequantize Q8_KS block to f16
    pub fn dequantize_block_q8_ks_f16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q8_KS block to bf16
    pub fn dequantize_block_q8_ks_bf16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q2_0 block (2-bit symmetric) to f32
    pub fn dequantize_block_q2_0_f32(vx: *const c_void, y: *mut f32, k: c_int);

    /// Dequantize Q2_0 block to f16
    pub fn dequantize_block_q2_0_f16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q2_0 block to bf16
    pub fn dequantize_block_q2_0_bf16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q3_0 block (3-bit symmetric) to f32
    pub fn dequantize_block_q3_0_f32(vx: *const c_void, y: *mut f32, k: c_int);

    /// Dequantize Q3_0 block to f16
    pub fn dequantize_block_q3_0_f16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize Q3_0 block to bf16
    pub fn dequantize_block_q3_0_bf16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize R16 block (extract K values from block_r16::d[]) to f32
    pub fn dequantize_block_r16_f32(vx: *const c_void, y: *mut f32, k: c_int);

    /// Dequantize R16 block to f16
    pub fn dequantize_block_r16_f16(vx: *const c_void, y: *mut c_void, k: c_int);

    /// Dequantize R16 block to bf16
    pub fn dequantize_block_r16_bf16(vx: *const c_void, y: *mut c_void, k: c_int);

    // -------------------------------------------------------------------------
    // Dequantize block kernels (K/128 AWQ types) - no k parameter
    // -------------------------------------------------------------------------

    /// Dequantize AWQ block (g128) to f32
    pub fn dequantize_block_q_awq_f32(vx: *const c_void, y: *mut f32);

    /// Dequantize AWQ block (g128) to f16
    pub fn dequantize_block_q_awq_f16(vx: *const c_void, y: *mut c_void);

    /// Dequantize AWQ block (g128) to bf16
    pub fn dequantize_block_q_awq_bf16(vx: *const c_void, y: *mut c_void);

    /// Dequantize AWQ block (g64) to f32
    pub fn dequantize_block_q_awq_g64_f32(vx: *const c_void, y: *mut f32);

    /// Dequantize AWQ block (g64) to f16
    pub fn dequantize_block_q_awq_g64_f16(vx: *const c_void, y: *mut c_void);

    /// Dequantize AWQ block (g64) to bf16
    pub fn dequantize_block_q_awq_g64_bf16(vx: *const c_void, y: *mut c_void);

    // -------------------------------------------------------------------------
    // Dequantize mul mat vec kernels (basic quants)
    // -------------------------------------------------------------------------

    /// Dequantize and multiply matrix-vector for Q4_0
    pub fn dequantize_mul_mat_vec_q4_0_cuda(
        vx: *const c_void,
        y: *const f32,
        dst: *mut f32,
        ncols: c_int,
        nrows: c_int,
    );

    /// Dequantize and multiply matrix-vector for Q4_1
    pub fn dequantize_mul_mat_vec_q4_1_cuda(
        vx: *const c_void,
        y: *const f32,
        dst: *mut f32,
        ncols: c_int,
        nrows: c_int,
    );

    /// Dequantize and multiply matrix-vector for Q5_0
    pub fn dequantize_mul_mat_vec_q5_0_cuda(
        vx: *const c_void,
        y: *const f32,
        dst: *mut f32,
        ncols: c_int,
        nrows: c_int,
    );

    /// Dequantize and multiply matrix-vector for Q5_1
    pub fn dequantize_mul_mat_vec_q5_1_cuda(
        vx: *const c_void,
        y: *const f32,
        dst: *mut f32,
        ncols: c_int,
        nrows: c_int,
    );

    /// Dequantize and multiply matrix-vector for Q8_0
    pub fn dequantize_mul_mat_vec_q8_0_cuda(
        vx: *const c_void,
        y: *const f32,
        dst: *mut f32,
        ncols: c_int,
        nrows: c_int,
    );

    // -------------------------------------------------------------------------
    // Dequantize mul mat vec kernels (K-quants)
    // -------------------------------------------------------------------------

    /// Dequantize and multiply matrix-vector for Q2_K
    pub fn dequantize_mul_mat_vec_q2_k(
        vx: *const c_void,
        yy: *const f32,
        dst: *mut f32,
        ncols: c_int,
        nrows: c_int,
    );

    /// Dequantize and multiply matrix-vector for Q3_K
    pub fn dequantize_mul_mat_vec_q3_k(
        vx: *const c_void,
        yy: *const f32,
        dst: *mut f32,
        ncols: c_int,
        nrows: c_int,
    );

    /// Dequantize and multiply matrix-vector for Q4_K
    pub fn dequantize_mul_mat_vec_q4_k(
        vx: *const c_void,
        yy: *const f32,
        dst: *mut f32,
        ncols: c_int,
        nrows: c_int,
    );

    /// Dequantize and multiply matrix-vector for Q5_K (no nrows parameter)
    pub fn dequantize_mul_mat_vec_q5_k(
        vx: *const c_void,
        yy: *const f32,
        dst: *mut f32,
        ncols: c_int,
    );

    /// Dequantize and multiply matrix-vector for Q6_K
    pub fn dequantize_mul_mat_vec_q6_k(
        vx: *const c_void,
        yy: *const f32,
        dst: *mut f32,
        ncols: c_int,
        nrows: c_int,
    );

    // -------------------------------------------------------------------------
    // Quantize kernels
    // -------------------------------------------------------------------------

    /// Quantize f32 to Q8_1
    pub fn quantize_q8_1(x: *const f32, vy: *mut c_void, kx: c_int, kx_padded: c_int);

    // -------------------------------------------------------------------------
    // Mul mat vec kernels - batch size = 1
    // -------------------------------------------------------------------------

    /// Matrix-vector multiply for Q4_0 x Q8_1, batch size 1
    pub fn mul_mat_vec_q4_0_q8_1_cuda1(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q4_1 x Q8_1, batch size 1
    pub fn mul_mat_vec_q4_1_q8_1_cuda1(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_0 x Q8_1, batch size 1
    pub fn mul_mat_vec_q5_0_q8_1_cuda1(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_1 x Q8_1, batch size 1
    pub fn mul_mat_vec_q5_1_q8_1_cuda1(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q8_0 x Q8_1, batch size 1
    pub fn mul_mat_vec_q8_0_q8_1_cuda1(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q2_K x Q8_1, batch size 1
    pub fn mul_mat_vec_q2_K_q8_1_cuda1(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q3_K x Q8_1, batch size 1
    pub fn mul_mat_vec_q3_K_q8_1_cuda1(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q4_K x Q8_1, batch size 1
    pub fn mul_mat_vec_q4_K_q8_1_cuda1(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_K x Q8_1, batch size 1
    pub fn mul_mat_vec_q5_K_q8_1_cuda1(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q6_K x Q8_1, batch size 1
    pub fn mul_mat_vec_q6_K_q8_1_cuda1(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    // -------------------------------------------------------------------------
    // Mul mat vec kernels - batch size = 2
    // -------------------------------------------------------------------------

    /// Matrix-vector multiply for Q4_0 x Q8_1, batch size 2
    pub fn mul_mat_vec_q4_0_q8_1_cuda2(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q4_1 x Q8_1, batch size 2
    pub fn mul_mat_vec_q4_1_q8_1_cuda2(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_0 x Q8_1, batch size 2
    pub fn mul_mat_vec_q5_0_q8_1_cuda2(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_1 x Q8_1, batch size 2
    pub fn mul_mat_vec_q5_1_q8_1_cuda2(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q8_0 x Q8_1, batch size 2
    pub fn mul_mat_vec_q8_0_q8_1_cuda2(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q2_K x Q8_1, batch size 2
    pub fn mul_mat_vec_q2_K_q8_1_cuda2(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q3_K x Q8_1, batch size 2
    pub fn mul_mat_vec_q3_K_q8_1_cuda2(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q4_K x Q8_1, batch size 2
    pub fn mul_mat_vec_q4_K_q8_1_cuda2(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_K x Q8_1, batch size 2
    pub fn mul_mat_vec_q5_K_q8_1_cuda2(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q6_K x Q8_1, batch size 2
    pub fn mul_mat_vec_q6_K_q8_1_cuda2(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    // -------------------------------------------------------------------------
    // Mul mat vec kernels - batch size = 3
    // -------------------------------------------------------------------------

    /// Matrix-vector multiply for Q4_0 x Q8_1, batch size 3
    pub fn mul_mat_vec_q4_0_q8_1_cuda3(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q4_1 x Q8_1, batch size 3
    pub fn mul_mat_vec_q4_1_q8_1_cuda3(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_0 x Q8_1, batch size 3
    pub fn mul_mat_vec_q5_0_q8_1_cuda3(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_1 x Q8_1, batch size 3
    pub fn mul_mat_vec_q5_1_q8_1_cuda3(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q8_0 x Q8_1, batch size 3
    pub fn mul_mat_vec_q8_0_q8_1_cuda3(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q2_K x Q8_1, batch size 3
    pub fn mul_mat_vec_q2_K_q8_1_cuda3(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q3_K x Q8_1, batch size 3
    pub fn mul_mat_vec_q3_K_q8_1_cuda3(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q4_K x Q8_1, batch size 3
    pub fn mul_mat_vec_q4_K_q8_1_cuda3(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_K x Q8_1, batch size 3
    pub fn mul_mat_vec_q5_K_q8_1_cuda3(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q6_K x Q8_1, batch size 3
    pub fn mul_mat_vec_q6_K_q8_1_cuda3(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    // -------------------------------------------------------------------------
    // Mul mat vec kernels - batch size = 4
    // -------------------------------------------------------------------------

    /// Matrix-vector multiply for Q4_0 x Q8_1, batch size 4
    pub fn mul_mat_vec_q4_0_q8_1_cuda4(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q4_1 x Q8_1, batch size 4
    pub fn mul_mat_vec_q4_1_q8_1_cuda4(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_0 x Q8_1, batch size 4
    pub fn mul_mat_vec_q5_0_q8_1_cuda4(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_1 x Q8_1, batch size 4
    pub fn mul_mat_vec_q5_1_q8_1_cuda4(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q8_0 x Q8_1, batch size 4
    pub fn mul_mat_vec_q8_0_q8_1_cuda4(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q2_K x Q8_1, batch size 4
    pub fn mul_mat_vec_q2_K_q8_1_cuda4(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q3_K x Q8_1, batch size 4
    pub fn mul_mat_vec_q3_K_q8_1_cuda4(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q4_K x Q8_1, batch size 4
    pub fn mul_mat_vec_q4_K_q8_1_cuda4(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_K x Q8_1, batch size 4
    pub fn mul_mat_vec_q5_K_q8_1_cuda4(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q6_K x Q8_1, batch size 4
    pub fn mul_mat_vec_q6_K_q8_1_cuda4(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    // -------------------------------------------------------------------------
    // Mul mat vec kernels - batch size = 5
    // -------------------------------------------------------------------------

    /// Matrix-vector multiply for Q4_0 x Q8_1, batch size 5
    pub fn mul_mat_vec_q4_0_q8_1_cuda5(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q4_1 x Q8_1, batch size 5
    pub fn mul_mat_vec_q4_1_q8_1_cuda5(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_0 x Q8_1, batch size 5
    pub fn mul_mat_vec_q5_0_q8_1_cuda5(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_1 x Q8_1, batch size 5
    pub fn mul_mat_vec_q5_1_q8_1_cuda5(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q8_0 x Q8_1, batch size 5
    pub fn mul_mat_vec_q8_0_q8_1_cuda5(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q2_K x Q8_1, batch size 5
    pub fn mul_mat_vec_q2_K_q8_1_cuda5(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q3_K x Q8_1, batch size 5
    pub fn mul_mat_vec_q3_K_q8_1_cuda5(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q4_K x Q8_1, batch size 5
    pub fn mul_mat_vec_q4_K_q8_1_cuda5(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_K x Q8_1, batch size 5
    pub fn mul_mat_vec_q5_K_q8_1_cuda5(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q6_K x Q8_1, batch size 5
    pub fn mul_mat_vec_q6_K_q8_1_cuda5(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    // -------------------------------------------------------------------------
    // Mul mat vec kernels - batch size = 6
    // -------------------------------------------------------------------------

    /// Matrix-vector multiply for Q4_0 x Q8_1, batch size 6
    pub fn mul_mat_vec_q4_0_q8_1_cuda6(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q4_1 x Q8_1, batch size 6
    pub fn mul_mat_vec_q4_1_q8_1_cuda6(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_0 x Q8_1, batch size 6
    pub fn mul_mat_vec_q5_0_q8_1_cuda6(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_1 x Q8_1, batch size 6
    pub fn mul_mat_vec_q5_1_q8_1_cuda6(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q8_0 x Q8_1, batch size 6
    pub fn mul_mat_vec_q8_0_q8_1_cuda6(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q2_K x Q8_1, batch size 6
    pub fn mul_mat_vec_q2_K_q8_1_cuda6(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q3_K x Q8_1, batch size 6
    pub fn mul_mat_vec_q3_K_q8_1_cuda6(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q4_K x Q8_1, batch size 6
    pub fn mul_mat_vec_q4_K_q8_1_cuda6(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_K x Q8_1, batch size 6
    pub fn mul_mat_vec_q5_K_q8_1_cuda6(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q6_K x Q8_1, batch size 6
    pub fn mul_mat_vec_q6_K_q8_1_cuda6(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    // -------------------------------------------------------------------------
    // Mul mat vec kernels - batch size = 7
    // -------------------------------------------------------------------------

    /// Matrix-vector multiply for Q4_0 x Q8_1, batch size 7
    pub fn mul_mat_vec_q4_0_q8_1_cuda7(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q4_1 x Q8_1, batch size 7
    pub fn mul_mat_vec_q4_1_q8_1_cuda7(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_0 x Q8_1, batch size 7
    pub fn mul_mat_vec_q5_0_q8_1_cuda7(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_1 x Q8_1, batch size 7
    pub fn mul_mat_vec_q5_1_q8_1_cuda7(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q8_0 x Q8_1, batch size 7
    pub fn mul_mat_vec_q8_0_q8_1_cuda7(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q2_K x Q8_1, batch size 7
    pub fn mul_mat_vec_q2_K_q8_1_cuda7(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q3_K x Q8_1, batch size 7
    pub fn mul_mat_vec_q3_K_q8_1_cuda7(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q4_K x Q8_1, batch size 7
    pub fn mul_mat_vec_q4_K_q8_1_cuda7(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_K x Q8_1, batch size 7
    pub fn mul_mat_vec_q5_K_q8_1_cuda7(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q6_K x Q8_1, batch size 7
    pub fn mul_mat_vec_q6_K_q8_1_cuda7(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    // -------------------------------------------------------------------------
    // Mul mat vec kernels - batch size = 8
    // -------------------------------------------------------------------------

    /// Matrix-vector multiply for Q4_0 x Q8_1, batch size 8
    pub fn mul_mat_vec_q4_0_q8_1_cuda8(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q4_1 x Q8_1, batch size 8
    pub fn mul_mat_vec_q4_1_q8_1_cuda8(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_0 x Q8_1, batch size 8
    pub fn mul_mat_vec_q5_0_q8_1_cuda8(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_1 x Q8_1, batch size 8
    pub fn mul_mat_vec_q5_1_q8_1_cuda8(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q8_0 x Q8_1, batch size 8
    pub fn mul_mat_vec_q8_0_q8_1_cuda8(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q2_K x Q8_1, batch size 8
    pub fn mul_mat_vec_q2_K_q8_1_cuda8(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q3_K x Q8_1, batch size 8
    pub fn mul_mat_vec_q3_K_q8_1_cuda8(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q4_K x Q8_1, batch size 8
    pub fn mul_mat_vec_q4_K_q8_1_cuda8(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q5_K x Q8_1, batch size 8
    pub fn mul_mat_vec_q5_K_q8_1_cuda8(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q6_K x Q8_1, batch size 8
    pub fn mul_mat_vec_q6_K_q8_1_cuda8(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    // -------------------------------------------------------------------------
    // Mul mat vec kernels - Q8_1 (batch sizes 1-8)
    // -------------------------------------------------------------------------

    /// Matrix-vector multiply for Q8_1 x Q8_1, batch size 1
    pub fn mul_mat_vec_q8_1_q8_1_cuda1(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q8_1 x Q8_1, batch size 2
    pub fn mul_mat_vec_q8_1_q8_1_cuda2(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q8_1 x Q8_1, batch size 3
    pub fn mul_mat_vec_q8_1_q8_1_cuda3(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q8_1 x Q8_1, batch size 4
    pub fn mul_mat_vec_q8_1_q8_1_cuda4(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q8_1 x Q8_1, batch size 5
    pub fn mul_mat_vec_q8_1_q8_1_cuda5(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q8_1 x Q8_1, batch size 6
    pub fn mul_mat_vec_q8_1_q8_1_cuda6(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q8_1 x Q8_1, batch size 7
    pub fn mul_mat_vec_q8_1_q8_1_cuda7(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-vector multiply for Q8_1 x Q8_1, batch size 8
    pub fn mul_mat_vec_q8_1_q8_1_cuda8(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    // -------------------------------------------------------------------------
    // Matrix-matrix multiply (MMQ) kernels
    // -------------------------------------------------------------------------

    /// Matrix-matrix multiply for Q4_0
    pub fn mul_mat_q4_0(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        ncols_y: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-matrix multiply for Q4_1
    pub fn mul_mat_q4_1(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        ncols_y: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-matrix multiply for Q5_0
    pub fn mul_mat_q5_0(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        ncols_y: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-matrix multiply for Q5_1
    pub fn mul_mat_q5_1(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        ncols_y: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-matrix multiply for Q8_0
    pub fn mul_mat_q8_0(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        ncols_y: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-matrix multiply for Q8_1
    pub fn mul_mat_q8_1(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        ncols_y: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-matrix multiply for Q_AWQ (group size 128)
    pub fn mul_mat_q_awq(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        ncols_y: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-matrix multiply for Q_AWQ_G64 (group size 64)
    pub fn mul_mat_q_awq_g64(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        ncols_y: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-matrix multiply for Q2_K
    pub fn mul_mat_q2_K(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        ncols_y: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-matrix multiply for Q3_K
    pub fn mul_mat_q3_K(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        ncols_y: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-matrix multiply for Q4_K
    pub fn mul_mat_q4_K(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        ncols_y: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-matrix multiply for Q5_K
    pub fn mul_mat_q5_K(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        ncols_y: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    /// Matrix-matrix multiply for Q6_K
    pub fn mul_mat_q6_K(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        ncols_y: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
    );

    // =========================================================================
    // DISPATCHER FUNCTIONS
    // =========================================================================
    // These handle kernel launch configuration internally, making the API simpler.

    /// Quantize f32 to Q8_1 (legacy interface with ky rows)
    ///
    /// Handles grid/block dimensions internally based on elem_count.
    pub fn run_quantize_q8_1(src: *const f32, dst: *mut c_void, elem_count: c_int, ky: c_int);

    /// Quantize act[rows][cols] (dtype 0=F16,1=BF16,2=F32) → block_q8a128
    /// [rows][cols/128] (the contiguous q8 activation block, 144 B each).
    pub fn run_quantize_q8a128(
        act: *const c_void,
        out: *mut c_void,
        rows: c_int,
        cols: c_int,
        dtype: c_int,
    );

    /// Dequantize block_q8a128[rows][cols/128] → out[rows][cols]
    /// (dtype 0=F16,1=BF16,2=F32).
    pub fn run_dequantize_q8a128(
        inp: *const c_void,
        out: *mut c_void,
        rows: c_int,
        cols: c_int,
        dtype: c_int,
    );

    /// Quantize f32 weights `[nrows × ncols]` (row-major) → the lane-major per-128 KO
    /// chunk tensor. `qtype` = QTYPE_Q{4,5,6,8}_KO (45..48); `nrows` a multiple of 8,
    /// `ncols` a multiple of 128. Byte-identical to the CPU `ko_quant::quantize_ko`.
    pub fn run_quantize_ko(
        w: *const f32,
        out: *mut c_void,
        nrows: c_int,
        ncols: c_int,
        qtype: c_int,
    );

    /// Dequantize a lane-major KO chunk tensor → f32 `[nrows × ncols]` (row-major).
    /// `qtype` = QTYPE_Q{4,5,6,8}_KO. Inverse of [`run_quantize_ko`].
    pub fn run_dequantize_ko(
        inp: *const c_void,
        out: *mut f32,
        nrows: c_int,
        ncols: c_int,
        qtype: c_int,
    );

    /// Quantize f32 to any supported quantized format
    ///
    /// # Parameters
    /// - `elem_count`: Total number of f32 elements to quantize
    /// - `qtype`: 0=Q4_0, 1=Q4_1, 2=Q5_0, 3=Q5_1, 4=Q8_0, 5=Q2_K, 6=Q3_K, 7=Q4_K, 8=Q5_K, 9=Q6_K, 10=Q8_1, 11=Q8_K, 12=Q_AWQ, 13=Q_AWQ_G64
    ///
    /// The destination buffer must be large enough to hold the quantized data.
    /// Number of quantized blocks = ceil(elem_count / block_size) where block_size
    /// depends on qtype (32 for standard, 256 for K-quants, 128/64 for AWQ).
    pub fn run_quantize_block(src: *const f32, dst: *mut c_void, elem_count: c_int, qtype: c_int);

    /// Quantize f32 with transpose from [H, T, D] to [H, D, T] layout
    ///
    /// Fuses the memory layout transformation with quantization to avoid
    /// intermediate allocations. Used for KV cache quantization where:
    /// - Input layout: [n_head, chunk_size, head_dim] - channel-oriented float
    /// - Output layout: [n_head, head_dim, chunk_size] - token-oriented quant
    ///
    /// # Parameters
    /// - `n_head`: Number of KV heads
    /// Batched quantize f32 with transpose for multiple chunks
    ///
    /// Processes multiple chunks in a single kernel launch for efficient KV cache migration.
    /// Fuses the memory layout transformation with quantization:
    /// - Input layout: [num_chunks, n_head, chunk_size, head_dim] - channel-oriented float
    /// - Output layout: [num_chunks, n_head, head_dim] Q blocks - token-oriented quant
    ///
    /// # Parameters
    /// - `src`: Source float data (can be non-contiguous if src_offsets provided)
    /// - `dst`: Destination quantized data (can be non-contiguous if dst_offsets provided)
    /// - `src_offsets`: Per-chunk element offsets into src (null for contiguous)
    /// - `dst_offsets`: Per-chunk byte offsets into dst (null for contiguous)
    /// - `num_chunks`: Number of chunks to process
    /// - `n_head`: Number of KV heads per chunk
    /// - `chunk_size`: Number of tokens per chunk (must be 32 for standard quants)
    /// - `head_dim`: Dimension per head
    /// - `qtype`: 0=Q4_0, 1=Q4_1, 2=Q5_0, 3=Q5_1, 4=Q8_0, 5=Q8_1
    pub fn run_quantize_transposed_batched(
        src: *const f32,
        dst: *mut c_void,
        src_offsets: *const c_int,
        dst_offsets: *const c_int,
        num_chunks: c_int,
        n_head: c_int,
        chunk_size: c_int,
        head_dim: c_int,
        qtype: c_int,
    );

    /// Batched fused transpose + quantize with multi-dtype support.
    ///
    /// Transforms [num_chunks, n_head, chunk_size, head_dim] input to
    /// [num_chunks, n_head, head_dim] quantized blocks. Supports multiple
    /// input dtypes (F32, F16, BF16, FP8) with inline conversion.
    ///
    /// # Parameters
    /// - `src`: Source data pointer (typed based on src_dtype)
    /// - `dst`: Destination quantized block pointer
    /// - `src_offsets`: Per-chunk element offsets into src (null for contiguous)
    /// - `dst_offsets`: Per-chunk byte offsets into dst (null for contiguous)
    /// - `num_chunks`: Number of chunks to process
    /// - `n_head`: Number of KV heads per chunk
    /// - `chunk_size`: Number of tokens per chunk (must be 32 for standard quants)
    /// - `head_dim`: Dimension per head
    /// - `qtype`: 0=Q4_0, 1=Q4_1, 2=Q5_0, 3=Q5_1, 4=Q8_0, 5=Q8_1
    /// - `src_dtype`: 0=F32, 1=F16, 2=BF16, 3=F8E4M3
    pub fn run_quantize_transposed_batched_typed(
        src: *const c_void,
        dst: *mut c_void,
        src_offsets: *const c_int,
        dst_offsets: *const c_int,
        num_chunks: c_int,
        n_head: c_int,
        chunk_size: c_int,
        head_dim: c_int,
        qtype: c_int,
        src_dtype: c_int,
    );

    /// Fused paged format selection + palette-4 grouping kernel.
    ///
    /// Single-pass: one CUDA block per (chunk, head), one warp.
    /// blocks_per_head must equal 128 (FUSED_HEAD_BLOCKS).
    /// Outputs 4 palette slots per (chunk, head) with format tags, scale indices,
    /// and per-block slot assignments — no intermediate per-block format arrays.
    pub fn run_select_kv_format_palette4_paged(
        per_head_table_raw: *const i64,
        head_gids: *const i64,
        q_relevance_median: *mut f32,
        q_relevance_spread: *mut f32,
        k_head_amax: *mut f32,
        v_head_amax: *mut f32,
        k_head_p95: *mut f32,
        v_head_p95: *mut f32,
        k_candidates: *const c_int,
        v_candidates: *const c_int,
        num_k_candidates: c_int,
        num_v_candidates: c_int,
        k_threshold_hi: f32,
        k_threshold_lo: f32,
        v_threshold_hi: f32,
        v_threshold_lo: f32,
        total_heads: c_int,
        blocks_per_head: c_int,
        n_kv_head: c_int,
        arena_chunks: c_int,
        valid_ranges: *const c_int,
        k_palette_tags: *mut c_int,
        v_palette_tags: *mut c_int,
        k_palette_scale: *mut f32,
        v_palette_scale: *mut f32,
        k_palette_map: *mut c_int,
        v_palette_map: *mut c_int,
        k_effective_block_tags: *mut c_int,
        v_effective_block_tags: *mut c_int,
        k_head_tags: *mut c_int,
        v_head_tags: *mut c_int,
        q_relevance_out: *mut f32,
        stream: *mut std::ffi::c_void,
    );

    /// Paged batched sampled-error kernel.
    ///
    /// Emits one scalar error per sampled head dimension, keeping batch as the
    /// outer slice and using the logical order:
    /// [batch_item][head_dim][quant_index][head].
    pub fn run_sample_quant_errors_paged(
        per_head_table_raw: *const i64,
        head_gids: *const i64,
        candidates: *const c_int,
        num_candidates: c_int,
        error_out: *mut f32,
        q_relevance_out: *mut f32,
        sample_token: c_int,
        side_is_k: c_int,
        num_chunks: c_int,
        n_kv_head: c_int,
        head_dim: c_int,
        arena_chunks: c_int,
    );

    /// Fused KV sampled-error kernel.
    ///
    /// Processes K and V in a single launch, sharing the per-head table
    /// lookup.  Q relevance (cosine K·Q) computed from the K side is used
    /// to weight V errors — important V positions (high Q·K) are penalised
    /// more aggressively, tightening V compression thresholds there.
    ///
    /// Requires K and V to share the same candidate list (`candidates`).
    ///
    /// Output layout (both `k_error_out` and `v_error_out`):
    /// [batch_item][head_dim][quant_index][head]
    pub fn run_sample_quant_errors_kv_paged(
        per_head_table_raw: *const i64,
        head_gids: *const i64,
        candidates: *const c_int,
        num_candidates: c_int,
        k_error_out: *mut f32,
        v_error_out: *mut f32,
        sample_token: c_int,
        num_chunks: c_int,
        n_kv_head: c_int,
        head_dim: c_int,
        arena_chunks: c_int,
    );

    /// GPU winner selection kernel — takes K and V error surfaces already on device
    /// and selects the winner candidate index for each (chunk, head, dim) cell across
    /// all threshold levels.  Output is `uint8_t` winner indices, one per cell per
    /// threshold, with layout `[n_thresholds × n_cells]` where
    /// `cell = (chunk * n_kv_head + head) * head_dim + dim`.
    ///
    /// Replaces the large D→H error surface download and CPU selection scan with
    /// a tiny download of u8 winners, saving ~(n_quant × 4)× bandwidth.
    pub fn run_select_winners_kv_paged(
        k_errors: *const f32,
        v_errors: *const f32,
        k_thresholds: *const f32,
        v_thresholds: *const f32,
        k_winners: *mut u8,
        v_winners: *mut u8,
        n_k_thresholds: c_int,
        n_v_thresholds: c_int,
        n_cells: c_int,
        n_quant: c_int,
        n_kv_head: c_int,
        head_dim: c_int,
    );

    /// Per-chunk worst-case reduction of per-block format tags.
    ///
    /// Reduces per-block format tags to one format tag per chunk by selecting
    /// the most conservative (highest fidelity) format any block requires.
    /// This preserves the selection kernel's quality guarantees.
    ///
    /// # Parameters
    /// - `k_block_tags`: per-block K format tags [num_chunks × blocks_per_chunk]
    /// - `v_block_tags`: per-block V format tags [num_chunks × blocks_per_chunk]
    /// - `k_chunk_tags`: output K format per chunk [num_chunks]
    /// - `v_chunk_tags`: output V format per chunk [num_chunks]
    /// - `k_candidates`: ordered candidate tags, high→low fidelity [num_k_candidates]
    /// - `v_candidates`: ordered candidate tags, high→low fidelity [num_v_candidates]
    /// - `num_k_candidates`: number of K candidates
    /// - `num_v_candidates`: number of V candidates
    /// - `blocks_per_chunk`: uniform block count per chunk
    /// - `num_chunks`: number of chunks
    pub fn run_reduce_chunk_format(
        k_block_tags: *const c_int,
        v_block_tags: *const c_int,
        k_chunk_tags: *mut c_int,
        v_chunk_tags: *mut c_int,
        k_candidates: *const c_int,
        v_candidates: *const c_int,
        num_k_candidates: c_int,
        num_v_candidates: c_int,
        blocks_per_chunk: c_int,
        num_chunks: c_int,
    );

    /// Per-head worst-case reduction: reduces per-block format tags to
    /// per-(chunk, head) tags.
    ///
    /// Output is `[num_chunks * n_kv_head]` per side.
    pub fn run_reduce_head_format(
        k_block_tags: *const c_int,
        v_block_tags: *const c_int,
        k_head_tags: *mut c_int,
        v_head_tags: *mut c_int,
        k_candidates: *const c_int,
        v_candidates: *const c_int,
        num_k_candidates: c_int,
        num_v_candidates: c_int,
        blocks_per_head: c_int,
        n_kv_head: c_int,
        num_chunks: c_int,
    );

    /// Per-head reduction that also expands the effective block tags after
    /// worst-case selection, allowing GPU/CPU A-B comparison on the reduced output.
    pub fn run_reduce_head_stats_format(
        k_block_tags: *const c_int,
        v_block_tags: *const c_int,
        k_head_tags: *mut c_int,
        v_head_tags: *mut c_int,
        k_effective_block_tags: *mut c_int,
        v_effective_block_tags: *mut c_int,
        blocks_per_head: c_int,
        n_kv_head: c_int,
        num_chunks: c_int,
    );

    /// Dequantize a block to f32/f16/bf16
    ///
    /// # Parameters
    /// - `qtype`: 0=Q4_0, 1=Q4_1, 2=Q5_0, 3=Q5_1, 4=Q8_0, 5=Q2_K, 6=Q3_K, 7=Q4_K, 8=Q5_K, 9=Q6_K, 10=Q8_1, 11=Q_AWQ, 12=Q_AWQ_G64
    /// - `out_dtype`: 0=F32, 1=F16, 2=BF16
    pub fn run_dequantize_block(
        src: *const c_void,
        dst: *mut c_void,
        elem_count: c_int,
        qtype: c_int,
        out_dtype: c_int,
    );

    /// Q0_V dequantize test entrypoint — wraps `BlockConverter<block_q0_v, float>::load`
    /// so unit tests can exercise the exact production GPU decode path used by
    /// attention/prefill kernels. Writes `num_blocks * 32` f32 elements to `dst`.
    pub fn run_dequantize_block_q0_v_f32(
        src: *const c_void,
        dst: *mut c_void,
        num_blocks: c_int,
        scale: f32,
    );

    /// Q0_V round-trip test entrypoint — quantizes then dequantizes each
    /// 32-element block via the production CUDA encoder/decoder pair under
    /// the IS_K compile-time selector. `is_k = 1` uses K-side calibrated
    /// tables, `is_k = 0` uses V-side. Caller passes f32 input pre-scaled
    /// by `outer` (typically 1.0 for in-normalised work) and gets the
    /// reconstruction in original units back. Used by the offline modelling
    /// test to measure real round-trip error against the format-selector
    /// pass_metric formulas, isolating quant/dequant correctness from the
    /// selection-kernel wiring.
    pub fn run_roundtrip_q0_v(
        src: *const c_void,
        recon: *mut c_void,
        num_blocks: c_int,
        outer: f32,
        is_k: c_int,
    );

    /// Q0_V runtime-table round-trip — same encoder/decoder pair as the
    /// production path, but with codebook tables supplied at launch time
    /// instead of read from `__constant__` memory. Used by the iterative
    /// curve-selection diagnostic to swap codebooks between iterations
    /// without recompiling. All table pointers are device-side and must
    /// match the production layout (256 × 32 i8 curves, 32 f16 scales,
    /// 32 × 8 f16 centroids, 256-entry peak permutation, 33-entry peak
    /// bin offsets).
    pub fn run_roundtrip_q0_v_runtime(
        src: *const c_void,
        recon: *mut c_void,
        num_blocks: c_int,
        outer: f32,
        curve_table_flat: *const c_void,
        scale_table_bits: *const c_void,
        centroid_table_bits_flat: *const c_void,
        peak_curve_indices: *const c_void,
        peak_bin_offsets: *const c_void,
    );

    /// Dequantize and multiply with vector (legacy path)
    ///
    /// # Parameters
    /// - `qtype`: 0=Q4_0, 1=Q4_1, 2=Q5_0, 3=Q5_1, 4=Q8_0, 5=Q2_K, 6=Q3_K, 7=Q4_K, 8=Q5_K, 9=Q6_K
    pub fn run_dequantize_mul_mat_vec(
        vx: *const c_void,
        y: *const f32,
        dst: *mut f32,
        ncols: c_int,
        nrows: c_int,
        qtype: c_int,
    );

    /// Matrix-vector multiply via Q8_1 quantization (batched)
    ///
    /// # Parameters
    /// - `b_size`: Batch size (1-8)
    /// - `qtype`: 0=Q4_0, 1=Q4_1, 2=Q5_0, 3=Q5_1, 4=Q8_0, 5=Q2_K, 6=Q3_K, 7=Q4_K, 8=Q5_K, 9=Q6_K, 10=Q8_1
    pub fn run_mul_mat_vec_q8_1(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
        b_size: c_int,
        qtype: c_int,
    );

    /// Full matrix multiply (tensor core / MMQ)
    ///
    /// # Parameters
    /// - `qtype`: 0=Q4_0, 1=Q4_1, 2=Q5_0, 3=Q5_1, 4=Q8_0, 5=Q2_K, 6=Q3_K, 7=Q4_K, 8=Q5_K, 9=Q6_K
    pub fn run_mul_mat(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: c_int,
        nrows_x: c_int,
        ncols_y: c_int,
        nrows_y: c_int,
        nrows_dst: c_int,
        qtype: c_int,
    );

    /// Palette4 KV-cache format conversion kernel.
    ///
    /// Converts K or V data between arbitrary arena formats using 4-palette
    /// metadata from KvHead structs. Each launch handles one of K or V
    /// (caller issues two launches).
    ///
    /// # Parameters
    /// - `src_kvhead_ptrs`: [num_layers × num_kv_heads] device ptrs to src KvHead structs, row-major
    /// - `dst_kvhead_ptrs`: [num_layers × num_kv_heads] device ptrs to dst KvHead structs, row-major
    /// - `num_kv_heads`: Number of KV heads per layer
    /// - `num_layers`: Number of layers
    /// - `num_chunks`: Number of 32-token chunks per head to convert
    /// - `is_k`: 1 = convert K, 0 = convert V
    /// - `head_dim`: Number of dimensions per head (block size)
    pub fn run_quantize_palette4_convert(
        heads_base: *const u8,
        num_heads: c_int,
        num_kv_heads: c_int,
        num_layers: c_int,
        num_chunks: c_int,
        is_k: c_int,
        head_dim: c_int,
        stream: *mut std::ffi::c_void,
    );

    /// GPU winner summarization kernel — reduces a full `[n_thresholds × n_cells]` u8 winner
    /// array (output of `run_select_winners_kv_paged`) to `[n_thresholds × 3]` f32 accumulators
    /// (ideal_bits, head_bits, pal4_bits) without a round-trip to host.
    ///
    /// Process one side (K or V) per call.  `out` must be zero-initialised before launch.
    ///
    /// # Parameters
    /// - `winners`: device pointer to `[n_thresholds × n_cells]` u8 winner indices
    /// - `cand_bpe`: device pointer to `[n_quant]` f32 bits-per-element values
    /// - `out`: zero-initialised device pointer to `[n_thresholds × 3]` f32 accumulators
    /// - `n_thresholds`: number of threshold levels
    /// - `n_cells`: total cells = n_bh × n_dim
    /// - `n_bh`: number of (batch, head) pairs = n_chunks × n_kv_head
    /// - `n_dim`: head dimension
    /// - `n_quant`: number of quantization candidates
    /// - `chunk_size`: tokens per chunk (32)
    /// - `pal_overhead`: palette metadata bits per head = n_dim * 2 + 4 * 8
    pub fn run_summarize_winners_side_paged(
        winners: *const u8,
        cand_bpe: *const f32,
        out: *mut f32,
        n_thresholds: c_int,
        n_cells: c_int,
        n_bh: c_int,
        n_dim: c_int,
        n_quant: c_int,
        chunk_size: c_int,
        pal_overhead: f32,
    );
}

// =============================================================================
// Quant type enum for dispatcher functions
// =============================================================================

/// Quantization type enum for the KV-quant / palette dispatcher.
///
/// Values MUST match:
///   - The C++ `QType` enum in `candle-kernels/src/quantized/block_compact.cuh`
///     (locked in via `static_assert` in that file).
///   - `GgmlDType` in `candle-core/src/quantized/mod.rs` (`#[repr(u32)]`).
///   - `ArenaFormat::*` in `candle-kernels/src/arena_table.cuh`.
///   - `SELECT_FMT_*` in `candle-kernels/src/quantize/select_kv_format.cuh`.
///
/// ⚠ Distinct from `crate::quantized::QType` (in `quantized/api.rs`),
/// which uses a smaller 0..13 numbering for the `run_quantized_matmul`
/// dispatcher. The two enums are intentionally separate because the
/// underlying CUDA dispatchers use different kernel lookup tables — see
/// the top-level QType audit comment in `api.rs` for the reasoning.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum QType {
    R16 = 3,
    P2 = 4,
    QAWQ = 5,
    QAWQ_G64 = 6,
    Q8_0 = 7,
    Q8_1 = 8,
    Q8_K = 9,
    Q8_KS = 10,
    Q6_K = 11,
    Q5_0 = 12,
    Q5_1 = 13,
    Q5_K = 14,
    Q4_0 = 15,
    Q4_1 = 16,
    Q4_K = 17,
    Q4_KS = 18,
    Q3_0 = 19,
    Q3_1 = 20,
    Q3_K = 21,
    Q2_0 = 22,
    Q2_1 = 23,
    Q2_K = 24,
    Q2_S = 25,
    Q2_A = 26,
    Q1_S = 27,
    Q0_V = 28,
    Q1_A = 29,
    Q0_X = 30,
    Q0_M2 = 31,
    Q0_M4 = 32,
    Q0 = 33,
    F8E4M3 = 34,
    F8E5M2 = 35,
    // Kernel-only activation QType past the GgmlDType-aligned range (no GgmlDType
    // counterpart, so it is absent from the GgmlDType lock). q8a128 is the contiguous
    // q8 activation block; mirrors QTYPE_Q8A128V/X in block_compact.cuh. Two modes,
    // same block, (eventually) different layouts: V = mode-1, X = mode-2 (weight-reuse).
    Q8A128V = 36,
    Q8A128X = 37,
    // Lane-major per-128 affine ("ordered") weight twins of the K-quant blocks for the
    // q8a128 int8 path. GPU-only weight layouts; mirror QTYPE_Q*_KO in block_compact.cuh.
    // All four have standalone quantize/dequant kernels (run_quantize_ko/run_dequantize_ko).
    Q4_KO = 45,
    Q5_KO = 46,
    Q6_KO = 47,
    Q8_KO = 48,
}

#[cfg(test)]
mod kv_qtype_lock_tests {
    //! Pin the integer value of every variant in this KV-side `QType`.
    //! See the C++ `static_assert` block in `block_compact.cuh` and the
    //! Rust test `ggml_dtype_values_are_stable` in
    //! `candle-core/src/quantized/mod.rs` — together these form a three-
    //! way lock between C++ / Rust-KV / Rust-GgmlDType.
    use super::QType;

    #[test]
    fn kv_qtype_values_are_stable() {
        assert_eq!(QType::R16 as i32, 3);
        assert_eq!(QType::P2 as i32, 4);
        assert_eq!(QType::QAWQ as i32, 5);
        assert_eq!(QType::QAWQ_G64 as i32, 6);
        assert_eq!(QType::Q8_0 as i32, 7);
        assert_eq!(QType::Q8_1 as i32, 8);
        assert_eq!(QType::Q8_K as i32, 9);
        assert_eq!(QType::Q8_KS as i32, 10);
        assert_eq!(QType::Q6_K as i32, 11);
        assert_eq!(QType::Q5_0 as i32, 12);
        assert_eq!(QType::Q5_1 as i32, 13);
        assert_eq!(QType::Q5_K as i32, 14);
        assert_eq!(QType::Q4_0 as i32, 15);
        assert_eq!(QType::Q4_1 as i32, 16);
        assert_eq!(QType::Q4_K as i32, 17);
        assert_eq!(QType::Q4_KS as i32, 18);
        assert_eq!(QType::Q3_0 as i32, 19);
        assert_eq!(QType::Q3_1 as i32, 20);
        assert_eq!(QType::Q3_K as i32, 21);
        assert_eq!(QType::Q2_0 as i32, 22);
        assert_eq!(QType::Q2_1 as i32, 23);
        assert_eq!(QType::Q2_K as i32, 24);
        assert_eq!(QType::Q2_S as i32, 25);
        assert_eq!(QType::Q2_A as i32, 26);
        assert_eq!(QType::Q1_S as i32, 27);
        assert_eq!(QType::Q0_V as i32, 28);
        assert_eq!(QType::Q1_A as i32, 29);
        assert_eq!(QType::Q0_X as i32, 30);
        assert_eq!(QType::Q0_M2 as i32, 31);
        assert_eq!(QType::Q0_M4 as i32, 32);
        assert_eq!(QType::Q0 as i32, 33);
        assert_eq!(QType::F8E4M3 as i32, 34);
        assert_eq!(QType::F8E5M2 as i32, 35);
        // Kernel-only activation QType (no GgmlDType counterpart).
        assert_eq!(QType::Q8A128V as i32, 36);
        assert_eq!(QType::Q8A128X as i32, 37);
        // KO byte-permuted twins — mirror QTYPE_Q*_KO / GgmlDType::Q*_KO.
        assert_eq!(QType::Q4_KO as i32, 45);
        assert_eq!(QType::Q5_KO as i32, 46);
        assert_eq!(QType::Q6_KO as i32, 47);
        assert_eq!(QType::Q8_KO as i32, 48);
    }
}

/// Output dtype enum for dequantize dispatcher
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DequantOutDType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
}
