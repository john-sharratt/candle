#pragma once
// Q0_V: Parametric-curve quantization — per-element decoder.
//
// Reconstruction for element `e` in block (new 7/5/4 bit layout):
//   1. Unpack 16-bit field = lo | (hi << 8):
//        curve_idx     = bits[0..6]   (7 bits, 128 entries)
//        scale_idx     = bits[7..11]  (5 bits, 32 entries)
//        centroid_idx  = bits[12..15] (4 bits, 16 entries)
//   2. curve_e   = (float)q0_v_curve_table_<side>[curve_idx][e]                       ∈ [-127, +127]
//   3. scale     = __half2float(q0_v_scale_table_bits_<side>[scale_idx])              ∈ [ 0, 1/127]
//   4. centroid  = __half2float(q0_v_centroid_table_bits_<side>[scale_idx][centroid_idx]) ∈ [-1,  +1]
//   5. x[e]      = __fmaf_rn(scale, curve_e, centroid)                                ∈ [-1,  +1]
//
// All three normalisation constants (1/127 for curve, 1/65535 for scale,
// 1/32767 for centroid) have been pre-baked into the table values offline,
// so the runtime hot path is a single FMA with ZERO constant multiplications.
//
// `q0_v_elem` returns the outer-normalised reconstruction in [-1, +1]. The
// BlockConverter then divides by `outer_scale` (the per-(chunk, head, side)
// head-amax chosen by the format selector — same convention as Q0_X / Q0 /
// Q0_M*); this recovers the un-scaled value `orig` because the encoder
// operated on `orig / head_amax`.
//
// IS_K selects between the K-side and V-side calibrated table sets. K and V
// are calibrated separately from real Qwen3/Llama dumps because they have
// different statistical properties (K is sensitive by channel, V by token).

#include "convert.cuh"
#include "../quantize/q0_v_tables.cuh"

// Reconstruct a single element of a Q0_V block (outer-normalised), generic
// over the table source. `Tables` exposes `curve(slot)`, `scale_bits(i)`,
// `centroid_bits(scale_idx, cent_idx)` — implemented by both
// `q0_v_detail::Q0VTablesStatic<IS_K>` (production, zero-sized) and
// `q0_v_detail::Q0VTablesRuntime` (diagnostic, runtime pointers).
//
// Hot path: ZERO constant multiplications, single FMA. The /127 normalisation
// of the curve is pre-baked into the scale (stored as f16: scale_norm / 127),
// so `scale * curve_i8` directly produces the scaled-and-normalised
// contribution with no extra constants in the kernel.
template <typename Tables>
static __device__ __forceinline__ float q0_v_elem_generic(
    const block_q0_v* s, int e, const Tables& tbl)
{
    const unsigned bits     = (unsigned)(s->lo) | ((unsigned)(s->hi) << 8);
    const int curve_idx     = (int)( bits        & 0x7Fu);
    const int scale_idx     = (int)((bits >> 7)  & 0x1Fu);
    const int centroid_idx  = (int)((bits >> 12) & 0x0Fu);
    const float curve_e  = (float)tbl.curve(curve_idx)[e];
    const float scale    = __half2float(__ushort_as_half(tbl.scale_bits(scale_idx)));
    const float centroid = __half2float(__ushort_as_half(tbl.centroid_bits(scale_idx, centroid_idx)));
    return __fmaf_rn(scale, curve_e, centroid);
}

// Backward-compat wrapper: IS_K-templated decoder used by the production
// dispatch path. Empty `Q0VTablesStatic<IS_K>` instance optimises away.
template <bool IS_K>
static __device__ __forceinline__ float q0_v_elem(const block_q0_v* s, int e) {
    q0_v_detail::Q0VTablesStatic<IS_K> tbl;
    return q0_v_elem_generic(s, e, tbl);
}

// Runtime-tables variant: same decoder, codebook supplied at launch time.
static __device__ __forceinline__ float q0_v_elem_runtime(
    const block_q0_v* s, int e, const q0_v_detail::Q0VTablesRuntime& tbl)
{
    return q0_v_elem_generic(s, e, tbl);
}

// BlockConverter primary specialisations default to V-side (IS_K=false). K-side
// callers invoke the explicit `_k` helpers below. The ArenaFormat dispatch in
// convert_all.cuh routes K vs V by selecting the appropriate helper.
template <> struct BlockConverter<block_q0_v, float> {
    static constexpr int BLOCK_SIZE = QK_Q0_V;
    static __device__ __forceinline__ int load(float* dst, const block_q0_v* src, int lane, float scale)
    { dst[lane] = q0_v_elem<false>(src, lane) / scale; return BLOCK_SIZE; }
    static __device__ __forceinline__ float load_element(const block_q0_v* src, int e, float scale)
    { return q0_v_elem<false>(src, e) / scale; }
};
template <> struct BlockConverter<block_q0_v, __half> {
    static constexpr int BLOCK_SIZE = QK_Q0_V;
    static __device__ __forceinline__ int load(__half* dst, const block_q0_v* src, int lane, float scale)
    { dst[lane] = __float2half_rn(q0_v_elem<false>(src, lane) / scale); return BLOCK_SIZE; }
    static __device__ __forceinline__ __half load_element(const block_q0_v* src, int e, float scale)
    { return __float2half_rn(q0_v_elem<false>(src, e) / scale); }
};
template <> struct BlockConverter<block_q0_v, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK_Q0_V;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q0_v* src, int lane, float scale)
    { dst[lane] = __float2bfloat16_rn(q0_v_elem<false>(src, lane) / scale); return BLOCK_SIZE; }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q0_v* src, int e, float scale)
    { return __float2bfloat16_rn(q0_v_elem<false>(src, e) / scale); }
};
template <> struct BlockConverter<block_q0_v, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK_Q0_V;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q0_v* src, int lane, float scale)
    { dst[lane] = from_f32<__nv_fp8_e4m3>(q0_v_elem<false>(src, lane) / scale); return BLOCK_SIZE; }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q0_v* src, int e, float scale)
    { return from_f32<__nv_fp8_e4m3>(q0_v_elem<false>(src, e) / scale); }
};

// IS_K-aware variants. Call sites that know the side at compile time use these
// directly instead of going through BlockConverter; the V-side BlockConverter
// specialisations above remain as the back-compat default.
template <bool IS_K>
static __device__ __forceinline__ float q0_v_load_element_f32(
    const block_q0_v* src, int e, float scale)
{ return q0_v_elem<IS_K>(src, e) / scale; }

template <bool IS_K>
static __device__ __forceinline__ __half q0_v_load_element_f16(
    const block_q0_v* src, int e, float scale)
{ return __float2half_rn(q0_v_elem<IS_K>(src, e) / scale); }

template <bool IS_K>
static __device__ __forceinline__ __nv_bfloat16 q0_v_load_element_bf16(
    const block_q0_v* src, int e, float scale)
{ return __float2bfloat16_rn(q0_v_elem<IS_K>(src, e) / scale); }

template <bool IS_K>
static __device__ __forceinline__ __nv_fp8_e4m3 q0_v_load_element_fp8(
    const block_q0_v* src, int e, float scale)
{ return from_f32<__nv_fp8_e4m3>(q0_v_elem<IS_K>(src, e) / scale); }

// Type-generic IS_K-aware element loader. Dispatches to the typed helpers
// above via tag overloading. Used by the format-runtime dispatch in
// convert_all.cuh when the caller wants to choose K vs V at compile time
// without specialising the entire dispatch tree.
namespace q0_v_load_dispatch_detail {
    template <bool IS_K> __device__ __forceinline__ float
    load_one(float, const block_q0_v* s, int e, float scale)
    { return q0_v_load_element_f32<IS_K>(s, e, scale); }
    template <bool IS_K> __device__ __forceinline__ __half
    load_one(__half, const block_q0_v* s, int e, float scale)
    { return q0_v_load_element_f16<IS_K>(s, e, scale); }
    template <bool IS_K> __device__ __forceinline__ __nv_bfloat16
    load_one(__nv_bfloat16, const block_q0_v* s, int e, float scale)
    { return q0_v_load_element_bf16<IS_K>(s, e, scale); }
    template <bool IS_K> __device__ __forceinline__ __nv_fp8_e4m3
    load_one(__nv_fp8_e4m3, const block_q0_v* s, int e, float scale)
    { return q0_v_load_element_fp8<IS_K>(s, e, scale); }
}

template <typename T, bool IS_K>
static __device__ __forceinline__ T q0_v_load_element_typed(
    const block_q0_v* src, int e, float scale)
{
    return q0_v_load_dispatch_detail::load_one<IS_K>(T{}, src, e, scale);
}

template <typename T, bool IS_K>
static __device__ __forceinline__ void q0_v_load_block_typed(
    T* dst, const block_q0_v* src, int lane, float scale)
{
    dst[lane] = q0_v_load_element_typed<T, IS_K>(src, lane, scale);
}
