// SPDX-License-Identifier: MIT
//
// Register round trip for the selection search: one lane's value of a 32-element
// block, quantized and dequantized without passing through the block's bytes.
//
// The selection kernel measures a format by quantizing a block and decoding it
// again. Done through the bytes, that is a shared-memory write, a warp barrier,
// the encoder's bit packing, a second barrier and a byte read — per block, per
// format, per scale. Every lane already holds what the bytes would give back to
// it: its own code and the block's stored parameters. These functions compute
// exactly that, so the round trip is the encoder's reductions plus a few
// arithmetic operations, all in registers.
//
// **They must reproduce the bytes path bit for bit.** Each one repeats its
// encoder's reductions in the encoder's order, rounds its stored parameters to
// the type they are stored in (an INT8 scale, an F16 `d`), and evaluates the
// decoder's expression from `BlockConverter<…, float>::load_element` as written.
// The encoders store lane 0's parameters; every lane computes the same ones,
// because an XOR butterfly pairs lanes that add the same two values in swapped
// order. Where an encoder broadcasts lane 0's value explicitly (Q5_0's signed
// maximum, whose strict tie rule leaves lanes holding different winners), so
// does its register path.
// The selection then measures exactly what the attention kernel will read. A
// format without a register path here keeps the bytes path; `lane_roundtrip`
// says which.
//
// Every lane of the warp must call these together: the reductions are full-warp
// shuffles.

#pragma once

// Is there a register round trip for this format?
template <int FMT> struct lane_roundtrip {
    static constexpr bool value = false;
};
template <> struct lane_roundtrip<SELECT_FMT_Q1_S> { static constexpr bool value = true; };
template <> struct lane_roundtrip<SELECT_FMT_Q2_A> { static constexpr bool value = true; };
template <> struct lane_roundtrip<SELECT_FMT_Q2_S> { static constexpr bool value = true; };
template <> struct lane_roundtrip<SELECT_FMT_Q3_0> { static constexpr bool value = true; };
template <> struct lane_roundtrip<SELECT_FMT_Q3_1> { static constexpr bool value = true; };
template <> struct lane_roundtrip<SELECT_FMT_Q4_0> { static constexpr bool value = true; };
template <> struct lane_roundtrip<SELECT_FMT_Q4_1> { static constexpr bool value = true; };
template <> struct lane_roundtrip<SELECT_FMT_Q5_0> { static constexpr bool value = true; };
template <> struct lane_roundtrip<SELECT_FMT_Q5_1> { static constexpr bool value = true; };
template <> struct lane_roundtrip<SELECT_FMT_Q8_0> { static constexpr bool value = true; };
template <> struct lane_roundtrip<SELECT_FMT_Q8_1> { static constexpr bool value = true; };
template <> struct lane_roundtrip<SELECT_FMT_Q0>    { static constexpr bool value = true; };
template <> struct lane_roundtrip<SELECT_FMT_Q0_X>  { static constexpr bool value = true; };
template <> struct lane_roundtrip<SELECT_FMT_Q0_M2> { static constexpr bool value = true; };
template <> struct lane_roundtrip<SELECT_FMT_Q0_M4> { static constexpr bool value = true; };
template <> struct lane_roundtrip<SELECT_FMT_Q1_A>  { static constexpr bool value = true; };
template <> struct lane_roundtrip<SELECT_FMT_Q0_V>  { static constexpr bool value = true; };

// `quantize_block_q1_s` then `BlockConverter<block_q1_s, float>::load_element`:
// the mean |x| as an INT8 scale, the sign as the code.
__device__ __forceinline__ float lane_roundtrip_q1_s(float xi, float outer) {
    float sum_abs = fabsf(xi);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_abs += __shfl_xor_sync(0xffffffff, sum_abs, offset, 32);
    const float mean_abs = sum_abs / 32.0f;
    const int8_t scale = (int8_t)__float2int_rn(fminf(127.0f, mean_abs * 127.0f));

    const float blk_scale = (float)scale * (1.0f / 127.0f) / outer;
    return (xi >= 0.0f) ? blk_scale : -blk_scale;
}

// `quantize_block_q2_a` then its `load_element`: INT8 scale and bias from the
// block's range, a 2-bit code above the bias.
__device__ __forceinline__ float lane_roundtrip_q2_a(float xi, float outer) {
    float vmax = xi, vmin = xi;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, offset, 32));
        vmin = fminf(vmin, __shfl_xor_sync(0xffffffff, vmin, offset, 32));
    }
    const int8_t scale_i8 = (int8_t)__float2int_rn(fminf(127.0f, ((vmax - vmin) * (1.0f / 3.0f)) * 127.0f));
    const int8_t bias_i8  = (int8_t)__float2int_rn(fmaxf(-127.0f, fminf(127.0f, vmin * 127.0f)));
    const float d  = (float)scale_i8 * (1.0f / 127.0f);
    const float m  = (float)bias_i8  * (1.0f / 127.0f);
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    const uint8_t q2 = (uint8_t)fminf(3.0f, fmaxf(0.0f, roundf((xi - m) * id)));

    const float dd = (float)scale_i8 * (1.0f / 127.0f) / outer;
    const float mm = (float)bias_i8  * (1.0f / 127.0f) / outer;
    return dd * (float)(q2 & 3) + mm;
}

// `quantize_block_q2_s` then its `load_element`: an INT8 scale from |x|max, a
// 2-bit code centred on 1.5.
__device__ __forceinline__ float lane_roundtrip_q2_s(float xi, float outer) {
    const float amax = quantize_warp_reduce_max(fabsf(xi));
    const int8_t scale = (int8_t)__float2int_rn(fminf(127.0f, (amax * (1.0f / 1.5f)) * 127.0f));
    const float d  = (float)scale * (1.0f / 127.0f);
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    const uint8_t q2 = (uint8_t)fminf(3.0f, fmaxf(0.0f, roundf(xi * id + 1.5f)));

    const float dd = (float)scale * (1.0f / 127.0f) / outer;
    return dd * ((float)(q2 & 3) - 1.5f);
}

// `quantize_block_q3_0` then its `load_element`: an F16 `d` from |x|max, a
// 3-bit code centred on 3.5.
__device__ __forceinline__ float lane_roundtrip_q3_0(float xi, float outer) {
    const float amax = quantize_warp_reduce_max(fabsf(xi));
    const float d  = amax * (1.0f / 3.5f);
    const float id = (amax != 0.0f) ? 3.5f / amax : 0.0f;
    const uint8_t q3 = (uint8_t)fminf(7.0f, fmaxf(0.0f, roundf(xi * id + 3.5f)));

    return __half2float(__float2half_rn(d)) * ((float)(q3 & 7) - 3.5f) / outer;
}

// `quantize_block_q3_1` then its `load_element`: F16 `d` and min from the
// block's range, a 3-bit code above the min.
__device__ __forceinline__ float lane_roundtrip_q3_1(float xi, float outer) {
    float vmax = xi, vmin = xi;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, offset, 32));
        vmin = fminf(vmin, __shfl_xor_sync(0xffffffff, vmin, offset, 32));
    }
    const float d  = (vmax - vmin) * (1.0f / 7.0f);
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    const uint8_t q = (uint8_t)fminf(7.0f, fmaxf(0.0f, roundf((xi - vmin) * id)));

    const float dd = __half2float(__float2half_rn(d))    / outer;
    const float mm = __half2float(__float2half_rn(vmin)) / outer;
    return dd * (float)(q & 7) + mm;
}

// `quantize_block_q4_0_vec` then its `load_element`: F16 `d` from the signed
// value of largest magnitude (lowest index on a tie; a zero or NaN never wins,
// leaving 0), a 4-bit code centred on 8.
__device__ __forceinline__ float lane_roundtrip_q4_0(float xi, float outer) {
    const int lane = threadIdx.x % WARP_SIZE;
    const float a0 = fabsf(xi);
    float amax    = (a0 > 0.0f) ? a0   : 0.0f;
    float max_val = (a0 > 0.0f) ? xi   : 0.0f;
    int   max_idx = (a0 > 0.0f) ? lane : 0;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_amax = __shfl_xor_sync(0xffffffff, amax, offset, 32);
        const float other_val  = __shfl_xor_sync(0xffffffff, max_val, offset, 32);
        const int   other_idx  = __shfl_xor_sync(0xffffffff, max_idx, offset, 32);
        if (other_amax > amax || (other_amax == amax && other_idx < max_idx)) {
            amax    = other_amax;
            max_val = other_val;
            max_idx = other_idx;
        }
    }
    const float d  = max_val / -8.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    const uint8_t q = (uint8_t)fminf(15.0f, fmaxf(0.0f, xi * id + 8.5f));

    const float dd = __half2float(__float2half_rn(d)) / outer;
    return dd * ((float)(q & 0xF) - 8.f);
}

// `quantize_block_q4_1` then its `load_element`: F16 `d` and min from the
// block's range, a 4-bit code above the min.
__device__ __forceinline__ float lane_roundtrip_q4_1(float xi, float outer) {
    const float vmax = quantize_warp_reduce_max(xi);
    const float vmin = quantize_warp_reduce_min(xi);
    const float d  = (vmax - vmin) * (1.0f / 15.0f);
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    const uint8_t q4 = (uint8_t)fminf(15.0f, fmaxf(0.0f, (xi - vmin) * id + 0.5f));

    const float dd = __half2float(__float2half_rn(d))    / outer;
    const float mm = __half2float(__float2half_rn(vmin)) / outer;
    return dd * (float)(q4 & 0xF) + mm;
}

// `quantize_block_q5_0` then its `load_element`: F16 `d` from the signed value
// of largest magnitude as lane 0 ends the encoder's butterfly holding it, a
// 5-bit code centred on 16.
__device__ __forceinline__ float lane_roundtrip_q5_0(float xi, float outer) {
    float amax = fabsf(xi);
    float max_val = xi;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_amax = __shfl_xor_sync(0xffffffff, amax, offset, 32);
        const float other_val  = __shfl_xor_sync(0xffffffff, max_val, offset, 32);
        if (other_amax > amax) { amax = other_amax; max_val = other_val; }
    }
    max_val = __shfl_sync(0xffffffff, max_val, 0, 32);
    const float d  = max_val / -16.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    const int q5 = (int)fminf(31.0f, fmaxf(0.0f, xi * id + 16.5f));

    const float dd = __half2float(__float2half_rn(d)) / outer;
    return dd * ((float)(q5 & 0x1F) - 16.f);
}

// `quantize_block_q5_1` then its `load_element`: F16 `d` and min from the
// block's range, a 5-bit code above the min.
__device__ __forceinline__ float lane_roundtrip_q5_1(float xi, float outer) {
    const float vmax = quantize_warp_reduce_max(xi);
    const float vmin = quantize_warp_reduce_min(xi);
    const float d  = (vmax - vmin) / 31.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    const int q5 = (int)fminf(31.0f, fmaxf(0.0f, (xi - vmin) * id + 0.5f));

    const float dd = __half2float(__float2half_rn(d))    / outer;
    const float mm = __half2float(__float2half_rn(vmin)) / outer;
    return dd * (float)(q5 & 0x1F) + mm;
}

// `quantize_block_q8_0_vec` then its `load_element`: F16 `d` from |x|max (a
// NaN element never sets it), an INT8 code.
__device__ __forceinline__ float lane_roundtrip_q8_0(float xi, float outer) {
    const float amax = quantize_warp_reduce_max(fmaxf(0.0f, fabsf(xi)));
    const float id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
    const int8_t q = (int8_t)__float2int_rn(xi * id);

    return __half2float(__float2half_rn(amax * (1.0f / 127.0f))) * (float)q / outer;
}

// `quantize_block_q8_1` then its `load_element`: as Q8_0 with the block sum
// stored beside `d`, which the decode does not read.
__device__ __forceinline__ float lane_roundtrip_q8_1(float xi, float outer) {
    const float amax = quantize_warp_reduce_max(fabsf(xi));
    const float id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
    const int8_t q = (int8_t)__float2int_rn(xi * id);

    return __half2float(__float2half_rn(amax * (1.0f / 127.0f))) * (float)q / outer;
}

// `quantize_block_q0` then its `load_element`: the block mean as one INT8
// centroid.
__device__ __forceinline__ float lane_roundtrip_q0(float xi, float outer) {
    const int8_t centroid = q0_encode_centroid(q0_warp_sum(xi) * (1.0f / 32.0f));
    return (float)centroid * (1.0f / 127.0f) / outer;
}

// `quantize_block_q0_x` then its `load_element`: the bulk anchor, plus the
// coarse delta on the outlier lane.
__device__ __forceinline__ float lane_roundtrip_q0_x(float xi, float outer) {
    const int lane = threadIdx.x % WARP_SIZE;
    const Q0XFit fit = q0_x_fit(xi);
    const int bulk_anchor = (int)(int8_t)fit.bulk_anchor;
    const int delta_u = (fit.outlier_delta & 0x07);
    const int outlier_delta = delta_u < 4 ? delta_u : delta_u - 8;
    const int delta_scaled = (lane == (fit.outlier_idx & 0x1F)) ? outlier_delta * Q0_X_S_OUTLIER : 0;
    const int v_i8 = max(-127, min(127, bulk_anchor + delta_scaled));
    return (float)v_i8 * (1.0f / 127.0f) / outer;
}

// `quantize_block_q0_m2` then its `load_element`: the centroid this lane's
// quartet is assigned to.
__device__ __forceinline__ float lane_roundtrip_q0_m2(float xi, float outer) {
    const int lane = threadIdx.x % WARP_SIZE;
    const Q0M2Fit fit = q0_m2_fit(xi);
    const bool hi = (((fit.qmask & 0xFF) >> (lane / 4)) & 1) != 0;
    const int8_t c = q0_encode_centroid(hi ? fit.c1 : fit.c0);
    return (float)c * (1.0f / 127.0f) / outer;
}

// `quantize_block_q0_m4` then its `load_element`: the centroid this lane's pair
// is assigned to.
__device__ __forceinline__ float lane_roundtrip_q0_m4(float xi, float outer) {
    const int lane = threadIdx.x % WARP_SIZE;
    const Q0M4Fit fit = q0_m4_fit(xi);
    const int k = (int)((fit.qmask >> (2 * (lane / 2))) & 3);
    float c = fit.c[0];
    #pragma unroll
    for (int j = 1; j < 4; j++) c = (k == j) ? fit.c[j] : c;
    return (float)q0_encode_centroid(c) * (1.0f / 127.0f) / outer;
}

// `quantize_block_q1_a` then its `load_element`: the mean of each sign's
// magnitudes as an INT8 amplitude, the sign as the code.
__device__ __forceinline__ float lane_roundtrip_q1_a(float xi, float outer) {
    const bool is_pos = (xi >= 0.0f);
    const uint32_t qmask = __ballot_sync(0xffffffff, is_pos);
    const float sum_pos = q0_warp_sum(is_pos ? xi  : 0.0f);
    const float sum_neg = q0_warp_sum(is_pos ? 0.0f : -xi);
    const int n_pos = __popc(qmask);
    const int n_neg = 32 - n_pos;
    const float mean_pos = (n_pos > 0) ? (sum_pos / (float)n_pos) : 0.0f;
    const float mean_neg = (n_neg > 0) ? (sum_neg / (float)n_neg) : 0.0f;
    const int8_t scale_pos = (int8_t)max(0, min(127, __float2int_rn(mean_pos * 127.0f)));
    const int8_t scale_neg = (int8_t)max(0, min(127, __float2int_rn(mean_neg * 127.0f)));

    const int scale_int = is_pos ? (int)scale_pos : (int)scale_neg;
    const float magnitude = (float)scale_int * (1.0f / 127.0f);
    return (is_pos ? magnitude : -magnitude) / outer;
}

// `quantize_block_q0_v<IS_K>` then `q0_v_load_element_f32<IS_K>`: the block's
// (scale, centroid) entries and best curve, decoded from the packed indexes.
// The curve search reads the 32 normalised values from the warp's scratch row
// (`find_best_curve_exhaustive_with`) instead of gathering them into 32 more
// registers, which the search kernel's register budget would spill.
template <bool IS_K>
__device__ __forceinline__ float lane_roundtrip_q0_v(float xi, float outer, float* warp_scratch) {
    const int lane = threadIdx.x % WARP_SIZE;
    q0_v_detail::Q0VTablesStatic<IS_K> tbl;
    float target_scaled;
    int scale_idx = 0, centroid_idx = 0;
    q0_v_detail::compute_target_and_indices(xi, tbl, target_scaled, scale_idx, centroid_idx);
    warp_scratch[lane] = target_scaled;
    __syncwarp();
    const int curve_idx = q0_v_detail::find_best_curve_exhaustive_with<IS_K>(
        [&](int e) { return warp_scratch[e]; }, lane);
    // Every lane has read the row; the next round trip may overwrite it.
    __syncwarp();
    block_q0_v packed;
    q0_v_detail::q0_v_pack_block(&packed, curve_idx, scale_idx, centroid_idx);
    return q0_v_load_element_f32<IS_K>(&packed, lane, outer);
}

// The round trip of lane value `xi` (already multiplied by `outer`) for a format
// with `lane_roundtrip<FMT>::value`. `warp_scratch` is the warp's 32-float
// shared-memory row, used by Q0_V's curve search; `is_k` picks Q0_V's K-side or
// V-side tables. Every other format ignores both.
template <int FMT>
__device__ __forceinline__ float lane_roundtrip_for_fmt(float xi, float outer, float* warp_scratch, bool is_k) {
    if      constexpr (FMT == SELECT_FMT_Q1_S) return lane_roundtrip_q1_s(xi, outer);
    else if constexpr (FMT == SELECT_FMT_Q2_A) return lane_roundtrip_q2_a(xi, outer);
    else if constexpr (FMT == SELECT_FMT_Q2_S) return lane_roundtrip_q2_s(xi, outer);
    else if constexpr (FMT == SELECT_FMT_Q3_0) return lane_roundtrip_q3_0(xi, outer);
    else if constexpr (FMT == SELECT_FMT_Q3_1) return lane_roundtrip_q3_1(xi, outer);
    else if constexpr (FMT == SELECT_FMT_Q4_0) return lane_roundtrip_q4_0(xi, outer);
    else if constexpr (FMT == SELECT_FMT_Q4_1) return lane_roundtrip_q4_1(xi, outer);
    else if constexpr (FMT == SELECT_FMT_Q5_0) return lane_roundtrip_q5_0(xi, outer);
    else if constexpr (FMT == SELECT_FMT_Q5_1) return lane_roundtrip_q5_1(xi, outer);
    else if constexpr (FMT == SELECT_FMT_Q8_0) return lane_roundtrip_q8_0(xi, outer);
    else if constexpr (FMT == SELECT_FMT_Q8_1) return lane_roundtrip_q8_1(xi, outer);
    else if constexpr (FMT == SELECT_FMT_Q0)    return lane_roundtrip_q0(xi, outer);
    else if constexpr (FMT == SELECT_FMT_Q0_X)  return lane_roundtrip_q0_x(xi, outer);
    else if constexpr (FMT == SELECT_FMT_Q0_M2) return lane_roundtrip_q0_m2(xi, outer);
    else if constexpr (FMT == SELECT_FMT_Q0_M4) return lane_roundtrip_q0_m4(xi, outer);
    else if constexpr (FMT == SELECT_FMT_Q1_A)  return lane_roundtrip_q1_a(xi, outer);
    else if constexpr (FMT == SELECT_FMT_Q0_V)
        return is_k ? lane_roundtrip_q0_v<true>(xi, outer, warp_scratch)
                    : lane_roundtrip_q0_v<false>(xi, outer, warp_scratch);
    else return 0.0f;
}
