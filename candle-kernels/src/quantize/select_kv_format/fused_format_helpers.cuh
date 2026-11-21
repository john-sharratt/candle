// Fused-kernel dimensions, scale candidates, bits-per-element table, warp error
// primitives and compile-time format dispatch.
//
// The launch dimensions of the fused selection kernel, the scale-candidate and
// bits-per-element tables used to rank formats, and the warp-wide error metrics
// the search compares against its thresholds. Quantize and dequantize are
// templated on the format at compile time, so each candidate runs one
// specialised path.

// =============================================================================
// FUSED-KERNEL DIMENSIONS
// =============================================================================
// 4 warps × 32 lanes = 128 threads per (chunk, head), regardless of head_dim.
// The kernel processes one (chunk, head) per CUDA block. The per-(chunk,head)
// block count (= chunk_size · head_dim / WARP_SIZE = head_dim) is the template
// parameter HB ∈ {128, 256} on the fused kernel and its search/claim helpers;
// the launcher dispatches on the runtime `blocks_per_head`. Thread count does
// NOT scale with HB — at HB = 256 each thread owns 2 entries where it owned 1.
//
// Work distribution across the 4 warps:
//   Phase 1 (load):   warps stride blocks 4-wise (warp_id, +4, +8, …)
//   Phase 2.5 (sink): warp 0 runs the per-token sink stats; warps 1–3 idle
//   Phase 2 (sort):   warp 0 sorts K, warp 1 sorts V (concurrent); 2/3 idle
//   Phase 3 (kthresh):threads stride entries 128-wise (b = tid, +128, …)
//   Phase 4+5 (search): all 4 warps stride live blocks 4-wise inside the
//                       per-(fmt, scale) round-trip; cross-warp reduction
//                       merges per-warp pass counts and pass masks
//
// Hoisted to file scope so the templated search/claim helpers below can
// refer to them; the kernel body re-uses the same constants for shared-
// memory sizing.
#define FUSED_WARP_SIZE            32
#define FUSED_WARPS_PER_BLOCK       4
#define FUSED_THREADS_PER_BLOCK    (FUSED_WARPS_PER_BLOCK * FUSED_WARP_SIZE)

// Occupancy target for __launch_bounds__ on the fused kernel. The HB = 128
// layout (~12.6 KB smem/block) sustains 8 blocks/SM inside the ~102.4 KB
// MaxShared carveout; the HB = 256 layout (~23.8 KB) fits 4.
template <int HB>
struct FusedOccupancy {
    static constexpr int min_blocks_per_sm = (HB == 128) ? 8 : 4;
};

// Compile-time HB tag for the launcher's blocks_per_head dispatch (same
// pattern as FmtTag / with_select_fmt further down).
template <int V>
struct HbTag {
    static constexpr int value = V;
};

// =============================================================================
// SCALE CANDIDATES
// =============================================================================
// Each block is quantized at `value · outer`, then dequantized through
// `/ outer`. The "outer" scale is searched per (slot, fmt): six candidates
// per format, derived from the slot's per-block amax distribution at
// compaction time (amax, p95, p80, mean, p25 are computed by tid 0 in a
// single alive-walk over the HB sort positions).
//
// The candidates form a monotone ladder from zero clipping to aggressive:
//
//   idx 0   1.0              — identity; the only candidate that does not
//                              scale up; correct when slot amax ≤ 1
//   idx 1   1 / amax         — normalise slot max to ±1; universal safe
//                              baseline (no block saturates INT8 scale,
//                              no Q0 block exceeds ±1 after scaling)
//   idx 2   1 / p95          — clips top 5% of blocks; p95 is the amax
//                              value exceeded by exactly 5% of alive blocks
//   idx 3   1 / p80          — clips top 20%; concentrates precision on
//                              the 80% majority at the cost of outliers
//   idx 4   1 / mean         — normalises to the mean amax; roughly clips
//                              the top half of a heavy-tailed distribution
//   idx 5   1 / p25          — clips top 75%; only the bottom quartile
//                              (by block amax) fits without saturation;
//                              most aggressive clipping, smallest step size
//                              for that quartile
//
// For FP16-scale formats (Q8_0, Q4_0, Q4_1, Q2_1, Q3_1, Q4_KS, Q8_KS,
// etc.) outer cancels algebraically in the round-trip, so all six candidates
// produce identical reconstruction error — the search still runs them to
// pick the winner used as the stored palette_scale.
//
// For INT8-scale formats (Q2_S, Q2_A, Q1_S) candidate 1 (1/amax) guarantees
// no saturation for any block. Candidates 2–5 saturate blocks whose amax
// exceeds the chosen quantile, causing them to fail the metric; this is
// intentional — the scale optimises for the surviving majority.
//
// For Q0-family formats (no internal scale; outer IS the scale) candidate 1
// maps the slot max to ±1, keeping all scaled values in the valid encode
// range. Candidates 2–5 push outlier blocks outside ±1; those blocks are
// clipped by q0_encode_centroid and fail. Candidate 0 (1.0) is uniquely
// useful when slot amax < 1 (no scaling needed; scale-up from 1/x would
// overflow the encode range).
//
// The search picks the (fmt, scale) pair with ≥ HB/4 passing blocks and the
// lowest `pass_metric` across all 6 candidates — see `search_scales_for_fmt`.
// Outer scale matters most for fixed-outer formats (Q4_KS, Q8_KS) where
// `outer` is the actual quantization scale; for block-internal-scale
// formats (Q4_0 etc.) `outer` cancels out of the round-trip and is
// effectively metadata.
#define NUM_SCALE_CANDIDATES 6
__device__ __forceinline__ float preferred_range(int idx, float amax, float safe_p95, float safe_p80, float mean, float safe_p25) {
    switch (idx) {
        case 0: return 1.0f;
        case 1: return 1.0f / amax;
        case 2: return 1.0f / safe_p95;
        case 3: return 1.0f / safe_p80;
        case 4: return 1.0f / mean;
        case 5: return 1.0f / safe_p25;
        default: return 1.0f;
    }
}

// =============================================================================
// WARP UTILITIES
// =============================================================================
// Reductions over the 32 lanes of a warp. All assume a fully-active warp
// (mask 0xffffffff). Results are convergent — every lane returns the same
// value after the reduction, so the caller can branch on it without
// further synchronization. The other warp-wide error reductions
// (`max_abs_error_warp`, `weighted_max_abs_error_warp`,
// `mean_top4_abs_error_warp`) live further down with the error primitives.

__device__ __forceinline__ float select_warp_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset, 32));
    return val;
}

// =============================================================================
// LOW-VALUE ERROR DEAD-ZONE
// =============================================================================
// Optional dead-zone smoothing for callers that need to suppress sub-noise
// errors. Subtracts a fixed margin from the absolute difference, clamped
// at zero, so differences smaller than the margin become zero. Not used
// inside the production pass/fail path — that path uses the raw absolute
// error throughout — but kept available for offline analysis paths that
// want noise-suppressed metrics.
//
// A fixed absolute dead-zone (rather than a relative one) gives uniform
// noise suppression across blocks of all magnitudes; relative thresholds
// would over-suppress small-amax blocks.

#define ERROR_MARGIN_ABS 0.001f   // absolute dead-zone (same for every block)

__device__ __forceinline__ float apply_error_margin(float x, float x_rt, float margin) {
    float diff = x - x_rt;
    float abs_diff = fabsf(diff);
    float adj_diff = fmaxf(0.0f, abs_diff - margin);
    return x - copysignf(adj_diff, diff);
}

// =============================================================================
// BITS-PER-ELEMENT LOOKUP
// =============================================================================
// Returns 4× the true bits-per-element for each format. The factor of 4
// keeps every value as an exact integer — sub-bit formats like Q0
// (0.25 bpe) and Q0_M2 (0.75 bpe) would otherwise round to the same
// number as their ×2 neighbours, breaking BPE-ascending tie-breaking.
// Lower = better compression. The returned int is suitable for direct
// integer comparison (no float-equality concerns).
//
// Equivalent identity: `bpe × 4 = block_bytes` for our 32-element blocks
// (bpe × 4 = bytes × 8/32 × 4 = bytes), so this can be read as
// "block bytes" wherever that's clearer.
__device__ __forceinline__ int format_bpe_x4(int fmt) {
    switch (fmt) {
        case SELECT_FMT_F16:     return 64;  // 16.00 bpe × 4
        case SELECT_FMT_BF16:    return 64;  // 16.00 bpe × 4
        case SELECT_FMT_Q8_KS:   return 36;  //  9.00 bpe × 4
        case SELECT_FMT_Q8_1:    return 36;  //  9.00 bpe × 4
        case SELECT_FMT_Q8_0:    return 34;  //  8.50 bpe × 4
        case SELECT_FMT_Q5_1:    return 24;  //  6.00 bpe × 4
        case SELECT_FMT_Q5_0:    return 22;  //  5.50 bpe × 4
        case SELECT_FMT_Q4_KS:   return 20;  //  5.00 bpe × 4
        case SELECT_FMT_Q4_1:    return 20;  //  5.00 bpe × 4
        case SELECT_FMT_Q4_0:    return 18;  //  4.50 bpe × 4
        case SELECT_FMT_Q3_1:    return 16;  //  4.00 bpe × 4
        case SELECT_FMT_Q3_0:    return 14;  //  3.50 bpe × 4
        case SELECT_FMT_Q2_1:    return 12;  //  3.00 bpe × 4
        case SELECT_FMT_Q2_A: return 10;  //  2.50 bpe × 4
        case SELECT_FMT_Q2_0:    return 10;  //  2.50 bpe × 4
        case SELECT_FMT_Q2_S: return  9;  //  2.25 bpe × 4
        case SELECT_FMT_Q0_M4:   return  8;  //  2.00 bpe × 4
        case SELECT_FMT_Q1_S: return  5;  //  1.25 bpe × 4
        case SELECT_FMT_Q0_M2:   return  3;  //  0.75 bpe × 4
        case SELECT_FMT_Q0_V:    return  2;  //  0.50 bpe × 4
        case SELECT_FMT_Q1_A:    return  6;  //  1.50 bpe × 4
        case SELECT_FMT_Q0_X:    return  2;  //  0.50 bpe × 4
        case SELECT_FMT_Q0:      return  1;  //  0.25 bpe × 4
        default:                 return 256; // unknown → worst
    }
}

// Fallback when the candidate set is empty — return the user's "max
// fidelity" candidate if one exists, otherwise F16 (no quantization).
__device__ __forceinline__ int first_candidate_or_f16(const int* candidates, int num_candidates) {
    return (num_candidates > 0) ? candidates[0] : SELECT_FMT_F16;
}


// =============================================================================
// ERROR PRIMITIVES  (real quant/dequant + head-normalised metrics)
// =============================================================================
// All reductions assume a fully-active warp (mask 0xffffffff) and broadcast
// their result to every lane. The pass/fail predicate the search uses is
// warp-uniform after these reductions, which is what makes the per-lane
// "if (passes) lane 0 records bit" pattern in `search_scales_for_fmt`
// safe — every lane sees the same value and branches the same way.

// CAS-loop atomicMax for float (CUDA lacks native float atomicMax).
// Required where multiple blocks contend on a head-amax accumulator.
__device__ __forceinline__
void atomicMax_f32(float* addr, float val) {
    unsigned int* addr_u = (unsigned int*)addr;
    unsigned int  old_u  = *addr_u, assumed;
    do {
        assumed = old_u;
        if (__uint_as_float(assumed) >= val) break;
        old_u = atomicCAS(addr_u, assumed, __float_as_uint(val));
    } while (assumed != old_u);
}

// max(|orig − recon|) across the warp's 32 lanes.
__device__ __forceinline__ float max_abs_error_warp(float orig, float recon) {
    float err = fabsf(orig - recon);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        err = fmaxf(err, __shfl_xor_sync(0xffffffff, err, off, 32));
    return err;
}

// Same, with a per-lane non-negative weight applied before the reduction.
// Used by callers where lanes that dominate q·k get amplified errors.
// When w_lane is uniform 1.0 this reduces to max_abs_error_warp.
__device__ __forceinline__ float weighted_max_abs_error_warp(
    float orig, float recon, float w_lane
) {
    float err = fabsf(orig - recon) * w_lane;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        err = fmaxf(err, __shfl_xor_sync(0xffffffff, err, off, 32));
    return err;
}

// Mean of the four largest weighted absolute errors across the 32 lanes.
// This is the K-side `pass_metric` (with w_lane = 1.0). It's the right
// metric for K because attention scores depend on the per-element
// products — outliers dominate, but the absolute max alone over-penalises
// a single-element spike that won't survive softmax. Top-4 mean gives a
// stable proxy that tracks the worst few errors without being all-or-
// nothing on a single lane.
//
// Tie handling: at each pass the first lane holding the current max
// value is masked out (replaced with -FLT_MAX) so subsequent passes find
// the next-largest. Degenerate case (all 32 lanes equal): all four
// passes return that value, mean = that value — fixes a prior bug where
// an all-equal warp returned 0.
__device__ __forceinline__ float mean_top4_abs_error_warp(
    float orig, float recon, float w_lane
) {
    const int lane = threadIdx.x & 31;
    // Errors are non-negative (`fmaxf` also turns a NaN into 0, as the shuffle
    // reduction's `fmaxf` ignored it), so a removed lane is marked with 0.0f
    // rather than -FLT_MAX: the next maximum is the same either way, and a
    // non-negative float orders as its bit pattern, which lets sm_80+ take each
    // warp maximum with one `REDUX.MAX` instead of five shuffles.
    float e = fmaxf(fabsf(orig - recon) * w_lane, 0.0f);
    float sum = 0.0f;
    #pragma unroll
    for (int pass = 0; pass < 4; pass++) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        const float m = __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(e)));
#else
        float m = e;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off, 32));
#endif
        sum += m;
        const unsigned ballot = __ballot_sync(0xffffffff, e == m);
        const int first_lane = __ffs(ballot) - 1;
        if (lane == first_lane) e = 0.0f;
    }
    return __fmul_rn(sum, 0.25f);
}

// Normalise an absolute warp error by per-head amax. `head_scale` must be
// strictly positive — the caller guarantees this via an epsilon guard
// (`fmaxf(amax_in, 1.0e-8f)`) on the inputs to `process_side`.
__device__ __forceinline__ float normalise_error(float abs_err, float head_scale) {
    return abs_err / head_scale;
}

// =============================================================================
// FORMAT CONVERT / QUANTIZE DISPATCH (runtime tag)
// =============================================================================
// `SELECT_FMT_*` is this file's tag space; `ArenaFormat::*` is the
// per-arena tag space that `convert_all.cuh` understands. They are
// numerically distinct (SELECT codes are dense 0..35 in the order
// formats were added to this file; ArenaFormat codes group by storage
// layout). All cross-file calls go through this glue layer.
//
// `quantize_to_smem` is warp-cooperative: all 32 lanes of the calling
// warp must invoke simultaneously with `src` populated by a
// `warp_f32[lane] = ...; __syncwarp();` sequence. The block_q* writers
// shuffle internally on mask 0xffffffff. BF16 and F16 are NOT quantized
// through this layer — they have no block struct; callers handle them
// inline via `float_fmt_roundtrip`, which applies the same `outer`
// scaling so float-format candidates still get caught when `outer`
// pushes values outside the format's representable range.
//
// The runtime switch in `quantize_to_smem` and `select_fmt_to_arena_fmt`
// is acceptable for the older non-fused / sampling kernels. The fused
// selection kernel uses the templated `quantize_block_for_fmt<FMT>` /
// `dequant_element_for_fmt<FMT>` paths below, which constant-fold the
// fmt tag into the call so the inner search loop has no per-block
// runtime branch.

// Maximum quantized block size across all candidates (Q8_KS / Q8_1 = 36 bytes).
// Used to size per-warp scratch buffers in callers.
#define MAX_QUANT_BLOCK_BYTES 36

// Map a runtime SELECT_FMT_* code to its ArenaFormat equivalent. The two
// spaces are different numberings (SELECT_FMT_Q4_0 = 15 vs.
// ArenaFormat::Q4_0 = 18, etc). BF16/F16 are handled inline by callers
// and never reach `dequant_element`, so they map to ArenaFormat::Invalid
// as a safety sentinel — any path that mistakenly falls through to here
// will produce a recognisable invalid-format failure rather than silent
// data corruption.
__device__ __forceinline__ int select_fmt_to_arena_fmt(int sfmt) {
    switch (sfmt) {
        case SELECT_FMT_Q8_KS:   return ArenaFormat::Q8_KS;
        case SELECT_FMT_Q8_0:    return ArenaFormat::Q8_0;
        case SELECT_FMT_Q8_1:    return ArenaFormat::Q8_1;
        case SELECT_FMT_Q5_1:    return ArenaFormat::Q5_1;
        case SELECT_FMT_Q5_0:    return ArenaFormat::Q5_0;
        case SELECT_FMT_Q4_KS:   return ArenaFormat::Q4_KS;
        case SELECT_FMT_Q4_1:    return ArenaFormat::Q4_1;
        case SELECT_FMT_Q4_0:    return ArenaFormat::Q4_0;
        case SELECT_FMT_Q3_1:    return ArenaFormat::Q3_1;
        case SELECT_FMT_Q3_0:    return ArenaFormat::Q3_0;
        case SELECT_FMT_Q2_1:    return ArenaFormat::Q2_1;
        case SELECT_FMT_Q2_0:    return ArenaFormat::Q2_0;
        case SELECT_FMT_Q2_A:    return ArenaFormat::Q2_A;
        case SELECT_FMT_Q2_S:    return ArenaFormat::Q2_S;
        case SELECT_FMT_Q1_S:    return ArenaFormat::Q1_S;
        case SELECT_FMT_Q0:      return ArenaFormat::Q0;
        case SELECT_FMT_Q0_V:    return ArenaFormat::Q0_V;
        case SELECT_FMT_Q1_A:    return ArenaFormat::Q1_A;
        case SELECT_FMT_Q0_X:    return ArenaFormat::Q0_X;
        case SELECT_FMT_Q0_M2:   return ArenaFormat::Q0_M2;
        case SELECT_FMT_Q0_M4:   return ArenaFormat::Q0_M4;
        default:                 return ArenaFormat::Invalid;
    }
}

// Lossy round-trip for a single float through F16 or BF16. Pass-through
// for any other format tag (caller invokes the quant path instead).
// Lets the BF16/F16 candidates participate in the same error-measurement
// loop as quant formats — catches the case where `outer` scales a value
// past the float format's representable range.
__device__ __forceinline__
float float_fmt_roundtrip(float x, int fmt) {
    if (fmt == SELECT_FMT_F16)  return __half2float(__float2half(x));
    if (fmt == SELECT_FMT_BF16) return __bfloat162float(__float2bfloat16(x));
    return x;
}

// Warp-cooperative quantize. All 32 lanes of the calling warp must invoke
// simultaneously, with `src` populated by a `warp_f32[lane] = ...` +
// `__syncwarp()` sequence. Internal shuffles use 0xffffffff masks. BF16/F16
// sentinels fall through (caller handles those separately).
//
// `is_k` is a runtime argument so that this switch — which carries every
// format's encoder — exists once. Templated on the side it existed twice, to
// serve the one arm (Q0_V) that calibrates K and V separately; every other
// format ignores it. Q0_V keeps its compile-time table choice inside its own
// arm, which is two small encoders rather than two of everything.
__device__ __forceinline__
void quantize_to_smem(const float* __restrict__ src, uint8_t* __restrict__ dst, int fmt,
                      bool is_k) {
    switch (fmt) {
        case SELECT_FMT_Q8_KS:   quantize_block_q8_ks  (src, (block_q8_ks*)  dst); break;
        case SELECT_FMT_Q8_0:    quantize_block_q8_0   (src, (block_q8_0*)   dst); break;
        case SELECT_FMT_Q8_1:    quantize_block_q8_1   (src, (block_q8_1*)   dst); break;
        case SELECT_FMT_Q5_1:    quantize_block_q5_1   (src, (block_q5_1*)   dst); break;
        case SELECT_FMT_Q5_0:    quantize_block_q5_0   (src, (block_q5_0*)   dst); break;
        case SELECT_FMT_Q4_KS:   quantize_block_q4_ks  (src, (block_q4_ks*)  dst); break;
        case SELECT_FMT_Q4_1:    quantize_block_q4_1   (src, (block_q4_1*)   dst); break;
        case SELECT_FMT_Q4_0:    quantize_block_q4_0   (src, (block_q4_0*)   dst); break;
        case SELECT_FMT_Q3_1:    quantize_block_q3_1   (src, (block_q3_1*)   dst); break;
        case SELECT_FMT_Q3_0:    quantize_block_q3_0   (src, (block_q3_0*)   dst); break;
        case SELECT_FMT_Q2_1:    quantize_block_q2_1   (src, (block_q2_1*)   dst); break;
        case SELECT_FMT_Q2_0:    quantize_block_q2_0   (src, (block_q2_0*)   dst); break;
        case SELECT_FMT_Q2_A: quantize_block_q2_a(src, (block_q2_a*)dst); break;
        case SELECT_FMT_Q2_S: quantize_block_q2_s(src, (block_q2_s*)dst); break;
        case SELECT_FMT_Q1_S: quantize_block_q1_s(src, (block_q1_s*)dst); break;
        case SELECT_FMT_Q0:      quantize_block_q0     (src, (block_q0*)     dst); break;
        // The one arm that reads `is_k`.
        case SELECT_FMT_Q0_V:
            if (is_k) quantize_block_q0_v<true>(src, (block_q0_v*)dst);
            else      quantize_block_q0_v<false>(src, (block_q0_v*)dst);
            break;
        case SELECT_FMT_Q1_A:    quantize_block_q1_a   (src, (block_q1_a*)   dst); break;
        case SELECT_FMT_Q0_X:    quantize_block_q0_x   (src, (block_q0_x*)   dst); break;
        case SELECT_FMT_Q0_M2:   quantize_block_q0_m2  (src, (block_q0_m2*)  dst); break;
        case SELECT_FMT_Q0_M4:   quantize_block_q0_m4  (src, (block_q0_m4*)  dst); break;
        default: break;  // BF16/F16/unknown: caller handles inline
    }
}

// =============================================================================
// COMPILE-TIME FORMAT DISPATCH (templated quantize / dequant)
// =============================================================================
// `quantize_to_smem` and `dequant_element` both switch over the format
// tag at runtime. Inside the selection kernel's hot loop that would cost
// ~21 case-arms of i-cache walk per block, plus a `__noinline__` call
// into `dequant_element_slow` for any format outside the 5 hot-path
// cases — well over an order of magnitude more expensive than the
// useful work. The fused selection kernel pays this cost once per
// candidate (in `with_select_fmt`), which materialises a compile-time
// `FmtTag<FMT>` constant the inner loop then uses to pick the
// specialisations below.
//
// Inside `search_scales_for_fmt`, both the quantize and dequant paths
// fully inline. The round-trip body collapses to roughly:
//
//     warp_f32[lane] = orig * outer;
//     __syncwarp();
//     quantize_block_q<FMT>(warp_f32, warp_quant);   // direct call
//     __syncwarp();
//     return BlockConverter<block_q<FMT>, float>::load_element(...);
//
// — no per-block fmt branching, no slow-path fallback, no virtual call.
//
// Coverage matches the format set the selection kernel can ever
// encounter (the union of `quantize_to_smem`'s arms). F16/BF16 are
// intentionally absent: they have no block struct, and callers handle
// them via `float_fmt_roundtrip` before reaching the quant path.

// `is_k` picks Q0_V's K-side or V-side calibrated table set. **Every other
// format ignores it**, which is why it is a runtime argument rather than a
// template parameter: as a template parameter it doubled every instantiation
// below it — `search_scales_for_fmt<FMT, IS_K, HB>` and everything it inlines —
// to serve one arm of a twenty-one-arm switch. Two full copies of twenty codecs
// so that the twenty-first could choose a table.
//
// Q0_V keeps its compile-time choice, one `if constexpr` deeper, where it costs
// two small table-indexed encoders instead of two of everything.
//
// Single function template + `if constexpr` because C++ does not allow partial
// specialisation of function templates.
template <int FMT>
__device__ __forceinline__ void quantize_block_for_fmt(
    const float* src, uint8_t* dst, bool is_k)
{
    if      constexpr (FMT == SELECT_FMT_Q8_KS)  quantize_block_q8_ks_vec(src, (block_q8_ks*) dst);
    else if constexpr (FMT == SELECT_FMT_Q8_0)   quantize_block_q8_0_vec(src, (block_q8_0*)  dst);
    else if constexpr (FMT == SELECT_FMT_Q8_1)   quantize_block_q8_1   (src, (block_q8_1*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q5_1)   quantize_block_q5_1   (src, (block_q5_1*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q5_0)   quantize_block_q5_0   (src, (block_q5_0*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q4_KS)  quantize_block_q4_ks_vec(src, (block_q4_ks*) dst);
    else if constexpr (FMT == SELECT_FMT_Q4_1)   quantize_block_q4_1   (src, (block_q4_1*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q4_0)   quantize_block_q4_0_vec(src, (block_q4_0*)  dst);
    else if constexpr (FMT == SELECT_FMT_Q3_1)   quantize_block_q3_1   (src, (block_q3_1*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q3_0)   quantize_block_q3_0   (src, (block_q3_0*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q2_1)   quantize_block_q2_1   (src, (block_q2_1*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q2_0)   quantize_block_q2_0_vec(src, (block_q2_0*)  dst);
    else if constexpr (FMT == SELECT_FMT_Q2_A)   quantize_block_q2_a   (src, (block_q2_a*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q2_S)   quantize_block_q2_s   (src, (block_q2_s*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q1_S)   quantize_block_q1_s   (src, (block_q1_s*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q0)     quantize_block_q0     (src, (block_q0*)     dst);
    else if constexpr (FMT == SELECT_FMT_Q0_V) {
        // The one format that reads `is_k`; both table sets live here rather
        // than duplicating every other codec above.
        if (is_k) quantize_block_q0_v<true>(src, (block_q0_v*) dst);
        else      quantize_block_q0_v<false>(src, (block_q0_v*) dst);
    }
    else if constexpr (FMT == SELECT_FMT_Q1_A)   quantize_block_q1_a   (src, (block_q1_a*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q0_X)   quantize_block_q0_x   (src, (block_q0_x*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q0_M2)  quantize_block_q0_m2  (src, (block_q0_m2*)  dst);
    else if constexpr (FMT == SELECT_FMT_Q0_M4)  quantize_block_q0_m4  (src, (block_q0_m4*)  dst);
}

// =============================================================================
// OUTER-SCALE CANCELLATION TRAIT
// =============================================================================
// Formats that store their per-block scale with FP16 (or wider) precision
// recompute that scale from the encoder input. When the input is `orig*outer`
// the encoder's internal scale absorbs `outer`; the decoder then divides by
// `outer` again, so the round-trip is identical regardless of the outer value
// chosen. For these formats the search only needs to trial outer = 1.0 — the
// other five candidates (1/amax, 1/p95, 1/p80, 1/mean, 1/p25) all produce
// bit-equivalent reconstructions modulo half-precision noise.
//
// INT8-scale formats (Q1_S, Q1_A, Q2_S, Q2_A) and Q0-family formats (no
// per-block scale; outer IS the scale) keep the full 6-candidate ladder.
template <int FMT> struct outer_cancels_in_roundtrip {
    static constexpr bool value = false;
};
template <> struct outer_cancels_in_roundtrip<SELECT_FMT_Q4_0>  { static constexpr bool value = true; };
template <> struct outer_cancels_in_roundtrip<SELECT_FMT_Q4_1>  { static constexpr bool value = true; };
template <> struct outer_cancels_in_roundtrip<SELECT_FMT_Q5_0>  { static constexpr bool value = true; };
template <> struct outer_cancels_in_roundtrip<SELECT_FMT_Q5_1>  { static constexpr bool value = true; };
template <> struct outer_cancels_in_roundtrip<SELECT_FMT_Q8_0>  { static constexpr bool value = true; };
template <> struct outer_cancels_in_roundtrip<SELECT_FMT_Q8_1>  { static constexpr bool value = true; };
template <> struct outer_cancels_in_roundtrip<SELECT_FMT_Q4_KS> { static constexpr bool value = true; };
template <> struct outer_cancels_in_roundtrip<SELECT_FMT_Q8_KS> { static constexpr bool value = true; };
template <> struct outer_cancels_in_roundtrip<SELECT_FMT_Q2_0>  { static constexpr bool value = true; };
template <> struct outer_cancels_in_roundtrip<SELECT_FMT_Q2_1>  { static constexpr bool value = true; };
template <> struct outer_cancels_in_roundtrip<SELECT_FMT_Q3_0>  { static constexpr bool value = true; };
template <> struct outer_cancels_in_roundtrip<SELECT_FMT_Q3_1>  { static constexpr bool value = true; };
template <> struct outer_cancels_in_roundtrip<SELECT_FMT_R16>   { static constexpr bool value = true; };

// =============================================================================
// FLAT-ONLY TRAIT
// =============================================================================
// Q0 stores a block's mean and nothing else, so it keeps none of the variation
// between the block's 32 tokens. The error metrics compare an error against the
// HEAD's scale, so a dim whose tokens vary by a small fraction of the head's
// range passes them even when that variation is the whole of what the dim
// contributes to a score (K) or an output (V). Q0 is therefore offered only the
// blocks that have no variation to keep: those flat to within one INT8 step of
// the head scale (`block_var_is_flat`). Flatness is a property of the block, not of
// the format or the outer scale, so it is decided once per head in Phase 1 and
// the search skips Q0's quantize-and-measure round trip for every other block.
// No other format is touched: each keeps at least a few levels per block. A
// candidate list is never Q0 alone: every list in use ends in a format with
// per-block levels, which is what a slot Q0 cannot take falls back to.
template <int FMT> struct flat_only {
    static constexpr bool value = false;
};
template <> struct flat_only<SELECT_FMT_Q0> { static constexpr bool value = true; };
// A block whose standard deviation is below head scale / FLAT_BLOCK_STEPS carries
// less than one INT8 step of variation: nothing there to keep.
#define FLAT_BLOCK_STEPS 127.0f
// The head scale the error metrics divide by is capped at this many times the
// head's 95th-percentile |x|, so a few sink-token elements cannot loosen every
// threshold in the chunk that holds them.
#define HEAD_SCALE_P95_CAP 8.0f

// Dequant a single element. `outer` is the same scale value the caller
// passed to `quantize_block_for_fmt`; the BlockConverter divides by it
// internally, matching the runtime `dequant_element` path. Returns
// float because that's the only destination type the selection kernel
// uses. (The `BlockConverter<block_q<FMT>, float>::load_element` path
// is the SAME path the attention kernel uses to read the block at
// inference time, so a successful round-trip here is a guarantee that
// the attention kernel will see the same bytes we measured against.)
// `is_k` picks Q0_V's calibrated table set; every other format ignores it, so it
// is a runtime argument for the reason `quantize_block_for_fmt` gives above.
// Single function template + `if constexpr` (no partial specialisation of
// function templates).
template <int FMT>
__device__ __forceinline__ float dequant_element_for_fmt(
    const uint8_t* blk, int lane, float outer, bool is_k)
{
    if      constexpr (FMT == SELECT_FMT_Q8_KS)
        return BlockConverter<block_q8_ks, float>::load_element(reinterpret_cast<const block_q8_ks*>(blk), lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q8_0)
        return BlockConverter<block_q8_0,  float>::load_element(reinterpret_cast<const block_q8_0*>(blk),  lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q8_1)
        return BlockConverter<block_q8_1,  float>::load_element(reinterpret_cast<const block_q8_1*>(blk),  lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q5_1)
        return BlockConverter<block_q5_1,  float>::load_element(reinterpret_cast<const block_q5_1*>(blk),  lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q5_0)
        return BlockConverter<block_q5_0,  float>::load_element(reinterpret_cast<const block_q5_0*>(blk),  lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q4_KS)
        return BlockConverter<block_q4_ks, float>::load_element(reinterpret_cast<const block_q4_ks*>(blk), lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q4_1)
        return BlockConverter<block_q4_1,  float>::load_element(reinterpret_cast<const block_q4_1*>(blk),  lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q4_0)
        return BlockConverter<block_q4_0,  float>::load_element(reinterpret_cast<const block_q4_0*>(blk),  lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q3_1)
        return BlockConverter<block_q3_1,  float>::load_element(reinterpret_cast<const block_q3_1*>(blk),  lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q3_0)
        return BlockConverter<block_q3_0,  float>::load_element(reinterpret_cast<const block_q3_0*>(blk),  lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q2_1)
        return BlockConverter<block_q2_1,  float>::load_element(reinterpret_cast<const block_q2_1*>(blk),  lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q2_0)
        return BlockConverter<block_q2_0,  float>::load_element(reinterpret_cast<const block_q2_0*>(blk),  lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q2_A)
        return BlockConverter<block_q2_a,  float>::load_element(reinterpret_cast<const block_q2_a*>(blk),  lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q2_S)
        return BlockConverter<block_q2_s,  float>::load_element(reinterpret_cast<const block_q2_s*>(blk),  lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q1_S)
        return BlockConverter<block_q1_s,  float>::load_element(reinterpret_cast<const block_q1_s*>(blk),  lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q0)
        return BlockConverter<block_q0,    float>::load_element(reinterpret_cast<const block_q0*>(blk),    lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q0_V)
        // The one format that reads `is_k`; see `quantize_block_for_fmt`.
        return is_k ? q0_v_load_element_f32<true>(reinterpret_cast<const block_q0_v*>(blk), lane, outer)
                    : q0_v_load_element_f32<false>(reinterpret_cast<const block_q0_v*>(blk), lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q1_A)
        return BlockConverter<block_q1_a,  float>::load_element(reinterpret_cast<const block_q1_a*>(blk),  lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q0_X)
        return BlockConverter<block_q0_x,  float>::load_element(reinterpret_cast<const block_q0_x*>(blk),  lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q0_M2)
        return BlockConverter<block_q0_m2, float>::load_element(reinterpret_cast<const block_q0_m2*>(blk), lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q0_M4)
        return BlockConverter<block_q0_m4, float>::load_element(reinterpret_cast<const block_q0_m4*>(blk), lane, outer);
    else
        return 0.0f;
}
