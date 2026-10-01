#pragma once
// Q0_V: Parametric-curve quantization — per-element decoder.
//
// Reconstruction for element `e` in block (new 7/5/4 bit layout):
//   1. Unpack 16-bit field = lo | (hi << 8):
//        curve_idx     = bits[0..6]   (7 bits, 128 entries)
//        scale_idx     = bits[7..11]  (5 bits, 32 entries)
//        centroid_idx  = bits[12..15] (4 bits, 16 entries)
//   2. curve_e   = ±q0_v_curve_base2_<side>[(curve_idx >> 4) & 3][e + 2·(curve_idx & 15)] ∈ [-127, +127]
//                  (negated for curve_idx ≥ 64; see q0_v_tables.cuh)
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

// Reconstruct a single element of a Q0_V block (outer-normalised) from a
// codebook supplied at launch time. `Tables` exposes `curve(slot)`,
// `scale_bits(i)` and `centroid_bits(scale_idx, cent_idx)`; the diagnostic
// round-trip paths pass `q0_v_detail::Q0VTablesRuntime` to swap the codebook
// without recompiling.
//
// Single FMA, no constant multiplications: the curve's /127 normalisation is
// pre-baked into the scale (stored as f16: scale_norm / 127).
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

// The production decoder, in two steps: a block's HEADER — everything that
// does not depend on the element — and then each element from it.
//
// A read at palette scale `scale` returns element / scale. The division is
// folded into the header: with r = 1 / scale (correctly rounded), the header
// holds s·r and c·r, and element e is ONE FMA against the base curve byte:
//
//   x[e] = fma(±s·r, base[e + 2p], c·r)        − for buckets 4–7
//
// Putting the bucket's sign on the scale rather than the byte is exact:
// fma(−a, b, c) and fma(a, −b, c) are the same operation. At r = 1 the header
// is the codebook's own (s, c), so `q0_v_elem` is the unscaled decode. The
// reference is `k_quants::q0_v_elem_scaled`; the decode-oracle test holds
// every GPU path to it, bit for bit, at several scales.
struct Q0VHeader {
    const int8_t* row;  // the doubled base row, offset by the phase's 2p
    float s;            // ±scale · r
    float c;            // centroid · r
};

template <bool IS_K>
static __device__ __forceinline__ Q0VHeader q0_v_header(const block_q0_v* blk, float r) {
    using namespace q0_v_detail;
    const uint32_t bits = (uint32_t)blk->lo | ((uint32_t)blk->hi << 8);
    const int curve = q0_v_curve_idx(bits);
    const float s = __fmul_rn(q0_v_scale<IS_K>(bits), r);
    return Q0VHeader{ q0_v_curve_row<IS_K>(curve),
                      q0_v_curve_negated(curve) ? -s : s,
                      __fmul_rn(q0_v_centroid<IS_K>(bits), r) };
}

static __device__ __forceinline__ float q0_v_header_elem(const Q0VHeader& h, int e) {
    return __fmaf_rn(h.s, (float)(int)__ldg(h.row + e), h.c);
}

template <bool IS_K>
static __device__ __forceinline__ float q0_v_elem(const block_q0_v* s, int e) {
    return q0_v_header_elem(q0_v_header<IS_K>(s, 1.f), e);
}

// Runtime-tables variant: same arithmetic, codebook supplied at launch time.
static __device__ __forceinline__ float q0_v_elem_runtime(
    const block_q0_v* s, int e, const q0_v_detail::Q0VTablesRuntime& tbl)
{
    return q0_v_elem_generic(s, e, tbl);
}

// Q0_V has NO BlockConverter. Its codebook depends on the side — K and V are
// calibrated separately — and BlockConverter's interface has no side, so a
// specialisation would have to pick one; a default of V decoded every K block
// read through a side-less path (the tile and INT8 prefill kernels) against the
// wrong codebook. Without one, a side-less read of a Q0_V block fails to
// compile (the primary template's static_assert), and every caller names the
// side through the helpers below. Each reads element e at palette scale
// `scale` — element / scale, through the folded header.
template <bool IS_K>
static __device__ __forceinline__ float q0_v_load_element_f32(
    const block_q0_v* src, int e, float scale)
{ return q0_v_header_elem(q0_v_header<IS_K>(src, __frcp_rn(scale)), e); }

// Type-generic IS_K-aware element loader, in the element type T, given the
// reciprocal palette scale `r` — so a caller reading many elements at one
// scale computes the reciprocal once. Used by the format-runtime dispatch in
// convert_all.cuh when the caller wants to choose K vs V at compile time
// without specialising the entire dispatch tree.
namespace q0_v_load_dispatch_detail {
    __device__ __forceinline__ float         narrow(float, float x)         { return x; }
    __device__ __forceinline__ __half        narrow(__half, float x)        { return __float2half_rn(x); }
    __device__ __forceinline__ __nv_bfloat16 narrow(__nv_bfloat16, float x) { return __float2bfloat16_rn(x); }
    __device__ __forceinline__ __nv_fp8_e4m3 narrow(__nv_fp8_e4m3, float x) { return from_f32<__nv_fp8_e4m3>(x); }
}

template <typename T, bool IS_K>
static __device__ __forceinline__ T q0_v_load_element_rcp(
    const block_q0_v* src, int e, float r)
{
    return q0_v_load_dispatch_detail::narrow(T{}, q0_v_header_elem(q0_v_header<IS_K>(src, r), e));
}

template <typename T, bool IS_K>
static __device__ __forceinline__ T q0_v_load_element_typed(
    const block_q0_v* src, int e, float scale)
{
    return q0_v_load_element_rcp<T, IS_K>(src, e, __frcp_rn(scale));
}

template <typename T, bool IS_K>
static __device__ __forceinline__ void q0_v_load_block_typed(
    T* dst, const block_q0_v* src, int lane, float scale)
{
    dst[lane] = q0_v_load_element_typed<T, IS_K>(src, lane, scale);
}
