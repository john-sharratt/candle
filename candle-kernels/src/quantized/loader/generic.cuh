#pragma once

// =============================================================================
// GENERIC LOADER INFRASTRUCTURE
// =============================================================================
// This file provides the generic loader type trait and backward-compatible
// wrapper functions for quantized matrix-vector multiplication.
//
// vec_dot_loader_for<block_q_t, vdr>::type maps a quantization type to its
// specialized loader struct, enabling compile-time dispatch based on the
// weight format (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K).
//
// Each loader struct implements:
//   - load_x(vbq, iqs): Decode quantized X weights into registers
//   - dot_y<act_t>(y): Templated dispatcher for different activation types
//     * block_q8_1: Optimized __dp4a path
//     * float: Vectorized float4 loads and FMA
//     * __half: Vectorized __half2 loads and FMA
//     * __nv_bfloat16: Vectorized __nv_bfloat162 loads and FMA
//     * __nv_fp8_e4m3: Vectorized __nv_fp8x2_e4m3 loads and FMA
//
// The specialization pattern ensures optimal layout and register usage per
// quantization scheme (e.g., Q2_K vdr=1, Q4_K vdr=2, etc.).
//
// LOADER TYPES:
// -------------
// 1. INLINE LOADERS (_inline suffix):
//    - Stores packed quantized values + scale
//    - Dequantizes on-the-fly during dot_y()
//    - Lower register pressure
//    - Best for: _s1 to _s8 kernels (small batch, each X used once)
//
// 2. PREDEQUANT LOADERS (_predequant suffix):
//    - Pre-dequantizes X during load_x()
//    - Stores dequantized values in Y-type optimal format (half2/bfloat162)
//    - Uses vectorized __hfma2 for 2 FMAs per instruction
//    - Best for: _s16, _s32 kernels (X reused across multiple Y vectors)
//
// =============================================================================

// =============================================================================
// Primary template declaration for loader type trait
// Specializations defined in individual loader headers (q4_0.cuh, q4_1.cuh, etc.)
// NOTE: Default arguments declared in loaders.cuh to avoid redefinition errors
// =============================================================================
template <typename block_q_t, int vdr, typename act_t> struct vec_dot_loader_for;



// =============================================================================
// LOADER QR TRAIT - Number of split-load phases per iteration
// =============================================================================
// K-quants use split-load design where load_x is called multiple times:
//   QR=1: Simple loaders (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q6_K) - single load_x + dot_y
//   QR=2: Q4_K, Q5_K - load_x_first + dot_y<0> + load_x_second + dot_y<1>
//   QR=4: Q2_K, Q3_K - load_x_first + dot_y<0> + load_x_nth<1,2,3> + dot_y<1,2,3>
//
// Primary template: defaults to QR=1 for simple loaders
// =============================================================================
template <typename block_q_t> struct loader_qr { static constexpr int value = 1; };

// K-quant QR specializations (original block types map to their QR values)
template <> struct loader_qr<block_q2_K> { static constexpr int value = 4; };
template <> struct loader_qr<block_q3_K> { static constexpr int value = 4; };
template <> struct loader_qr<block_q4_K> { static constexpr int value = 2; };
template <> struct loader_qr<block_q5_K> { static constexpr int value = 2; };
template <> struct loader_qr<block_q6_K> { static constexpr int value = 2; };

// Helper variable template for convenient access
template <typename block_q_t>
constexpr int loader_qr_v = loader_qr<block_q_t>::value;
