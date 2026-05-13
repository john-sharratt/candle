// SPDX-License-Identifier: MIT
// Quantize Kernels - CUDA kernels for float -> quantized conversions
//
// This compilation unit includes all quantize kernels and their entry points.
// It's separate from quantized.cu to keep compilation clean.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// =============================================================================
// Block type definitions (must match quantized.cu exactly)
// =============================================================================

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef QK_K
#define QK_K 256
#endif

#define QK4_0 32
#define QK4_1 32
#define QK5_0 32
#define QK5_1 32
#define QK8_0 32
#define QK8_1 32

typedef uint16_t ggml_fp16_t;

typedef struct {
    half    d;              // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;

typedef struct {
    half2   dm;             // dm.x = delta, dm.y = min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;

typedef struct {
    half d;                 // delta
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants
} block_q5_0;

typedef struct {
    half2 dm;               // dm.x = delta, dm.y = min
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_1 / 2];  // nibbles / quants
} block_q5_1;

typedef struct {
    half    d;              // delta
    int8_t  qs[QK8_0];      // quants
} block_q8_0;

typedef struct {
    half2   ds;             // ds.x = delta, ds.y = sum
    int8_t  qs[QK8_0];      // quants
} block_q8_1;

// K-quants
#define K_SCALE_SIZE 12

typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    half2 dm;                // super-block scale for quantized scales/mins
} block_q2_K;

typedef struct {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
    uint8_t scales[K_SCALE_SIZE]; // scales, quantized with 6 bits
    half d;             // super-block scale
} block_q3_K;

typedef struct {
    half2 dm;                  // super-block scale for quantized scales/mins
    uint8_t scales[3*QK_K/64]; // scales, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;

typedef struct {
    half2 dm;                     // super-block scale for quantized scales/mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K/2];   // quants, lower 4 bits
    uint8_t qh[QK_K/4];   // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales
    half    d;         // delta
} block_q6_K;

typedef struct {
    float d;              // delta (scale) - note: f32 not f16!
    int8_t qs[QK_K];      // quants (256 int8 values)
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;

// AWQ types - must match Rust BlockQAWQ/BlockQAWQG64 (80 bytes each)
#define QK_Q_AWQ 128
#define QK_Q_AWQ_G64 128  // Block still contains 128 elements but uses G64 grouping

// BlockQAWQ: 4-bit AWQ with group size 128 (80 bytes, 16-byte aligned)
// Layout matches Rust: qs[16] (u32), scale (f16), zero (f16), _pad[3] (u32)
typedef struct __align__(16) {
    uint32_t qs[16];     // 64 bytes: 128 × 4-bit nibbles packed as u32 (8 per u32)
    half scale;          // 2 bytes: scale factor
    half zero;           // 2 bytes: zero point
    uint32_t _pad[3];    // 12 bytes: padding to 80 bytes total
} block_q_awq;
static_assert(sizeof(block_q_awq) == 80, "block_q_awq must be 80 bytes");

// BlockQAWQG64: 4-bit AWQ with group size 64 (80 bytes, 16-byte aligned)
// Layout matches Rust: qs[16] (u32), scales[2] (f16), zeros[2] (f16), _pad (u32) + alignment
typedef struct __align__(16) {
    uint32_t qs[16];     // 64 bytes: 128 × 4-bit nibbles packed as u32 (8 per u32)
    half scales[2];      // 4 bytes: scale factors (one per 64 elements)
    half zeros[2];       // 4 bytes: zero points (one per 64 elements)
    uint32_t _pad;       // 4 bytes: explicit padding
    // alignment padding adds 4 more bytes to reach 80
} block_q_awq_g64;
static_assert(sizeof(block_q_awq_g64) == 80, "block_q_awq_g64 must be 80 bytes");

// =============================================================================
// Include AWQ quantize headers only (other kernels are in quantized_dispatcher.cu)
// =============================================================================

// Enable AWQ quantize since we have 80-byte padded structs
#define CANDLE_AWQ_QUANTIZE_PADDED

// Include only the AWQ quantize headers, not the full quantize.cuh
// (to avoid duplicate definitions of non-AWQ kernels)
#include "quantize_q_awq.cuh"
#include "quantize_q_awq_g64.cuh"

// =============================================================================
// AWQ KERNEL ENTRY POINTS
// =============================================================================

extern "C" __global__ void quantize_tensor_q_awq(
    const float* __restrict__ src,
    block_q_awq* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q_awq<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q_awq_g64(
    const float* __restrict__ src,
    block_q_awq_g64* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q_awq_g64<1>(src, dst, num_blocks);
}
