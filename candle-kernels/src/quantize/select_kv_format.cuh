// SPDX-License-Identifier: MIT
//
// =============================================================================
// Adaptive per-block KV-cache format selection (palette-4, paged, sink-aware)
// =============================================================================
//
// What this file produces
// -----------------------
// Given a paged KV-cache where K and V are stored as 32-element blocks at
// (chunk, head, block-in-head) positions, this file's host entry point —
// `run_select_kv_format_palette4_paged` — assigns each block a quantization
// format such that the per-block reconstruction error stays within a
// normalised threshold.
//
// Within each (chunk, head), the 128 blocks are partitioned into 4 palette
// slots of 32 blocks each. Blocks in the same slot share a (format, outer
// scale) pair. Per-head metadata is therefore just 4 (fmt, scale) pairs plus
// a 2-bit-per-block palette index — small enough to keep in cache, and
// uniform enough that the attention kernel's hot path can dispatch a single
// dequant routine per slot rather than per block.
//
// Outputs (indexed by `head_id = chunk_id * n_kv_head + head_idx`):
//
//   palette_tags          [4]   winning format per slot
//   palette_scale         [4]   winning outer scale per slot
//   palette_map           [128] which slot each block belongs to (0..3)
//   effective_block_tags  [128] per-block format (= palette_tags[slot])
//   head_tag                    most conservative slot's format
//   q_relevance_out       [128] optional, per-block Σ(q²k²)/Σ(k²)
//
// How it works
// ------------
// Two kernels run back-to-back per launch:
//
//   1. `approximate_q_relevance_quantiles` summarises the per-(chunk, head)
//      distributions: head amax (max |x|) for K and V, p95 amax for K and
//      V, and the median + spread of the q-relevance signal (the latter
//      only when the source K format carries Q activations — i.e. R16).
//      Three independent 64-bin histograms.
//
//   2. `select_kv_format_palette4_paged` does the actual selection. For
//      each (chunk, head): load all 128 blocks; detect attention-sink
//      tokens; sort by amax desc; iterate 4 slots, each picking the most
//      aggressive (lowest-BPE) format whose normalised reconstruction
//      error stays within threshold for ≥ 32 of the still-unclaimed
//      blocks; assign those 32 to the slot. Slot 0 sees the largest-amax
//      blocks first (most conservative format wins); slot 3 sees the
//      smallest (most aggressive).
//
// Phase pipeline (one CUDA block per (chunk, head), 128 threads)
// --------------------------------------------------------------
//
//                Inputs from arena and quantile pass
//                                │
//                                ▼
//          ┌──────────────────────────────────────────┐
//   1      │ Load K, V, Q from arena  →  smem         │
//          │ Per-block amax (K, V) and q-relevance     │
//          └──────────────────────────────────────────┘
//                                │
//                                ▼
//          ┌──────────────────────────────────────────┐
//   2.5    │ Attention-sink detection                 │
//          │   (a) per-head_dim mean Q                │
//          │   (b) per-token Q·K alignment score      │
//          │   (c) chunk-local μ, σ                   │
//          │   (d) sink_weight[t] = max(0, tanh(z))   │
//          │ Hoist v_thr_eff = lo + max·(hi − lo)     │
//          │      v_thr_sq  = v_thr_eff²              │
//          └──────────────────────────────────────────┘
//                                │
//                                ▼
//          ┌──────────────────────────────────────────┐
//   2      │ Bitonic sort 128 entries by amax desc.   │
//          │ warp 0 sorts K, warp 1 sorts V (parallel)│
//          └──────────────────────────────────────────┘
//                                │
//                                ▼
//          ┌──────────────────────────────────────────┐
//   3      │ Per-block K threshold from               │
//          │   q-relevance, q_median, q_spread        │
//          └──────────────────────────────────────────┘
//                                │
//                                ▼
//          ┌──────────────────────────────────────────┐
//   4+5    │ For each of 4 palette slots:             │
//          │   (a) Compact unclaimed list             │
//          │       (alive bitmask + ballot popcount)  │
//          │   (b) Search BPE-ascending × 6 candidates:│
//          │       find (fmt, scale) with ≥ 32 pass   │
//          │       track fallback (lowest max-err)    │
//          │   (c) Claim 32 (passing first, then fill)│
//          └──────────────────────────────────────────┘
//                                │
//                                ▼
//                      Per-(chunk, head) outputs
//
// Block invariant
// ---------------
// "Block" here means a 32-element strip of one (head_dim row) at a fixed
// (chunk, head, block-in-head) position. blocks_per_head is fixed at 128
// (= chunk_size 32 × head_dim 128 / WARP_SIZE 32). One CUDA block ↔ one
// (chunk, head); per-block round-trips are warp-cooperative — all 32 lanes
// of the active warp hold the block's 32 elements simultaneously.
//
// Numerical contract
// ------------------
// Per-block error metrics are side-asymmetric:
//
//   K side: pass_metric = mean_top4(|orig − recon|) · (1 / head_amax)
//           threshold   = kthresh[b]    (q-relevance scaled, per block)
//
//   V side: pass_metric = mean_{32 lanes}(orig − recon)² · (1 / head_amax²)
//           threshold   = v_thr_sq      (sink-aware, constant per head)
//
// The K choice is top-4 mean because K errors enter the softmax directly
// and outliers dominate the score perturbation. The V choice is MSE
// because V's contribution to attention output is the L2-weighted sum
// Σ_t a_t · v_t; mean-squared error is the structurally correct choice
// for an L2 budget. Top-4 max would over-penalise outlier elements that
// don't move the L2 of the actual computation V participates in.
//
// Sink protection: tokens whose K vector aligns with the chunk's mean Q
// direction receive disproportionate attention mass. Their V errors are
// amplified in the output, so the V threshold is interpolated between
// `v_threshold_lo` (lenient, non-sink) and `v_threshold_hi` (strict,
// peak-sink) using the maximum sink_weight in the chunk. One strong sink
// token forces stricter quality on every V block of the chunk —
// conservative, but correct: any block could end up being attended to
// jointly with the sink.
//
// Output threshold convention: `*_threshold_lo > *_threshold_hi`
// numerically. lo = lenient, hi = strict. Pass `lo == hi` to disable the
// per-side scaling (sink lerp on V; q-relevance scaling on K).
//
// O(1) error contract
// -------------------
// Every metric the search measures is normalised (by head_amax for K,
// head_amax² for V) so the threshold values are dimensionless and
// transferable across heads, layers, and models. Combined with the
// candidate set's BPE ladder, this gives the per-block error budget
// the O(1) bound required by the unbounded-context attention design.
//
// Include via `quantized_dispatcher.cu`, which provides the `block_q*`
// struct definitions referenced by the per-format quantize/dequant paths.

#pragma once

#include "fp8_e4m3_utils.cuh"
#include "../arena_table.cuh"
#include "../convert/convert_all.cuh"
#include "quantize.cuh"


// =============================================================================
// FORMAT TAG CONSTANTS (must match Rust QuantFormat / GgmlDType QType mapping)
// =============================================================================
// `SELECT_FMT_*` is this file's tag space. It must match the Rust
// `QuantFormat`/`GgmlDType` ordering bit-for-bit — these constants are the
// wire ABI between the Rust selection code and these CUDA kernels.
//
// The numbering here is the historical order formats were added to the
// codebase, NOT the BPE ordering. For BPE-ascending iteration use
// `format_bpe_x4`; for the global cross-format ranking (which orders all
// formats including F16/BF16) use `format_table_index_cuda`.

// QType codes (same as transpose_batch.cuh)
#define SELECT_FMT_F32      0
#define SELECT_FMT_F16      1
#define SELECT_FMT_BF16     2
#define SELECT_FMT_R16      3
#define SELECT_FMT_P2       4
#define SELECT_FMT_QAWQ     5
#define SELECT_FMT_QAWQ_G64 6
#define SELECT_FMT_Q8_0     7
#define SELECT_FMT_Q8_1     8
#define SELECT_FMT_Q8_K     9
#define SELECT_FMT_Q8_KS    10
#define SELECT_FMT_Q6_K     11
#define SELECT_FMT_Q5_0     12
#define SELECT_FMT_Q5_1     13
#define SELECT_FMT_Q5_K     14
#define SELECT_FMT_Q4_0     15
#define SELECT_FMT_Q4_1     16
#define SELECT_FMT_Q4_K     17
#define SELECT_FMT_Q4_KS    18
#define SELECT_FMT_Q3_0     19
#define SELECT_FMT_Q3_1     20
#define SELECT_FMT_Q3_K     21
#define SELECT_FMT_Q2_0     22
#define SELECT_FMT_Q2_1     23
#define SELECT_FMT_Q2_K     24
#define SELECT_FMT_Q2_S  25
#define SELECT_FMT_Q2_A  26
#define SELECT_FMT_Q1_S  27
#define SELECT_FMT_Q0_V     28
#define SELECT_FMT_Q1_A     29
#define SELECT_FMT_Q0_X     30
#define SELECT_FMT_Q0_M2    31
#define SELECT_FMT_Q0_M4    32
#define SELECT_FMT_Q0       33
#define SELECT_FMT_F8E4M3   34
#define SELECT_FMT_F8E5M2   35

// =============================================================================
// FUSED-KERNEL DIMENSIONS
// =============================================================================
// 4 warps × 32 lanes = 128 threads per (chunk, head). The kernel processes
// one (chunk, head) per CUDA block; FUSED_HEAD_BLOCKS is the per-(chunk,head)
// block count and is invariant across this codebase (= chunk_size · head_dim
// / WARP_SIZE = 32 · 128 / 32 = 128).
//
// Work distribution across the 4 warps:
//   Phase 1 (load):   warps stride blocks 4-wise (warp_id, +4, +8, …)
//   Phase 2.5 (sink): warp 0 runs the per-token sink stats; warps 1–3 idle
//   Phase 2 (sort):   warp 0 sorts K, warp 1 sorts V (concurrent); 2/3 idle
//   Phase 3 (kthresh):tid < 128 each handles one block (full parallelism)
//   Phase 4+5 (search): all 4 warps stride live blocks 4-wise inside the
//                       per-(fmt, scale) round-trip; cross-warp reduction
//                       merges per-warp pass counts and pass masks
//
// Hoisted to file scope so the templated search/claim helpers below can
// refer to them; the kernel body re-uses the same constants for shared-
// memory sizing.
#define FUSED_HEAD_BLOCKS         128
#define FUSED_WARP_SIZE            32
#define FUSED_WARPS_PER_BLOCK       4
#define FUSED_THREADS_PER_BLOCK    (FUSED_WARPS_PER_BLOCK * FUSED_WARP_SIZE)

// =============================================================================
// SCALE CANDIDATES
// =============================================================================
// Each block is quantized at `value · outer`, then dequantized through
// `/ outer`. The "outer" scale is searched per (slot, fmt): six candidates
// per format, derived from the slot's per-block amax distribution at
// compaction time (amax, p95, p80, mean, p25 are computed by tid 0 in a
// single alive-walk over the 128 sort positions).
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
// The search picks the (fmt, scale) pair with ≥ 32 passing blocks and the
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
    float e = fabsf(orig - recon) * w_lane;
    float sum = 0.0f;
    #pragma unroll
    for (int pass = 0; pass < 4; pass++) {
        float m = e;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off, 32));
        sum += m;
        const unsigned ballot = __ballot_sync(0xffffffff, e == m);
        const int first_lane = __ffs(ballot) - 1;
        if (lane == first_lane) e = -FLT_MAX;
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
// IS_K is threaded through so format encoders that calibrate K and V
// separately (Q0_V) pick the right table set. Other formats ignore it.
template <bool IS_K>
__device__ __forceinline__
void quantize_to_smem(const float* __restrict__ src, uint8_t* __restrict__ dst, int fmt) {
    switch (fmt) {
        case SELECT_FMT_Q8_KS:   quantize_block_q8_ks  (src, (block_q8_ks*)  dst); break;
        case SELECT_FMT_Q8_0:    quantize_block_q8_0   (src, (block_q8_0*)   dst); break;
        case SELECT_FMT_Q8_1:    quantize_block_q8_1   (src, (block_q8_1*)   dst); break;
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
        case SELECT_FMT_Q0_V:    quantize_block_q0_v<IS_K>(src, (block_q0_v*)dst); break;
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
// ~19 case-arms of i-cache walk per block, plus a `__noinline__` call
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

// IS_K is threaded through so Q0_V can pick the K-side or V-side calibrated
// table set at compile time. Other formats ignore IS_K. Single function
// template + `if constexpr` because C++ does not allow partial specialisation
// of function templates.
template <int FMT, bool IS_K>
__device__ __forceinline__ void quantize_block_for_fmt(
    const float* src, uint8_t* dst)
{
    if      constexpr (FMT == SELECT_FMT_Q8_KS)  quantize_block_q8_ks  (src, (block_q8_ks*)  dst);
    else if constexpr (FMT == SELECT_FMT_Q8_0)   quantize_block_q8_0   (src, (block_q8_0*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q8_1)   quantize_block_q8_1   (src, (block_q8_1*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q4_KS)  quantize_block_q4_ks  (src, (block_q4_ks*)  dst);
    else if constexpr (FMT == SELECT_FMT_Q4_1)   quantize_block_q4_1   (src, (block_q4_1*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q4_0)   quantize_block_q4_0   (src, (block_q4_0*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q3_1)   quantize_block_q3_1   (src, (block_q3_1*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q3_0)   quantize_block_q3_0   (src, (block_q3_0*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q2_1)   quantize_block_q2_1   (src, (block_q2_1*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q2_0)   quantize_block_q2_0   (src, (block_q2_0*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q2_A)   quantize_block_q2_a   (src, (block_q2_a*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q2_S)   quantize_block_q2_s   (src, (block_q2_s*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q1_S)   quantize_block_q1_s   (src, (block_q1_s*)   dst);
    else if constexpr (FMT == SELECT_FMT_Q0)     quantize_block_q0     (src, (block_q0*)     dst);
    else if constexpr (FMT == SELECT_FMT_Q0_V)   quantize_block_q0_v<IS_K>(src, (block_q0_v*) dst);
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

// Dequant a single element. `outer` is the same scale value the caller
// passed to `quantize_block_for_fmt`; the BlockConverter divides by it
// internally, matching the runtime `dequant_element` path. Returns
// float because that's the only destination type the selection kernel
// uses. (The `BlockConverter<block_q<FMT>, float>::load_element` path
// is the SAME path the attention kernel uses to read the block at
// inference time, so a successful round-trip here is a guarantee that
// the attention kernel will see the same bytes we measured against.)
// IS_K threaded through so Q0_V dequant picks the K-side or V-side calibrated
// table set at compile time. Other formats ignore IS_K. Single function
// template + `if constexpr` (no partial specialisation of function templates).
template <int FMT, bool IS_K>
__device__ __forceinline__ float dequant_element_for_fmt(
    const uint8_t* blk, int lane, float outer)
{
    if      constexpr (FMT == SELECT_FMT_Q8_KS)
        return BlockConverter<block_q8_ks, float>::load_element(reinterpret_cast<const block_q8_ks*>(blk), lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q8_0)
        return BlockConverter<block_q8_0,  float>::load_element(reinterpret_cast<const block_q8_0*>(blk),  lane, outer);
    else if constexpr (FMT == SELECT_FMT_Q8_1)
        return BlockConverter<block_q8_1,  float>::load_element(reinterpret_cast<const block_q8_1*>(blk),  lane, outer);
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
        return q0_v_load_element_f32<IS_K>(reinterpret_cast<const block_q0_v*>(blk), lane, outer);
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

// =============================================================================
// ALIVE-BITMASK HELPERS  —  per-slot tombstoning
// =============================================================================
// Each slot's claim phase tombstones the 32 blocks it just selected so the
// next slot's compaction skips them. We use a 128-bit alive mask split
// into two u64 halves (lo = blocks 0..63, hi = blocks 64..127) — bit b
// set means block b is unclaimed.
//
// Why a bitmask instead of `idx_sorted[i] = -1` tombstoning:
//
//   - O(1) live count via `__popcll(lo) + __popcll(hi)`,
//     vs. O(N) scan of an int array.
//   - "Find first alive in sort order" becomes a 32-lane chunked
//     `__ballot_sync` scan, vs. a lane-0 serial walk.
//   - The kidx/vidx arrays stay read-only after the bitonic sort, so
//     they fit comfortably in uint16_t (block IDs are 0..127). Halves
//     the smem footprint over int.
//
// `alive_clear` uses atomicAnd because the second-pass fill in
// `process_side` can target distinct bits of the same u64 word from
// multiple lanes in the same warp; a plain RMW would race and drop
// updates.

__device__ __forceinline__ bool alive_get(uint64_t lo, uint64_t hi, int b) {
    return (b < 64) ? (((lo >> b) & 1ULL) != 0ULL)
                    : (((hi >> (b - 64)) & 1ULL) != 0ULL);
}
__device__ __forceinline__ void alive_clear(uint64_t* lo, uint64_t* hi, int b) {
    if (b < 64) atomicAnd((unsigned long long*)lo, ~(1ULL << b));
    else        atomicAnd((unsigned long long*)hi, ~(1ULL << (b - 64)));
}
__device__ __forceinline__ int alive_count(uint64_t lo, uint64_t hi) {
    return __popcll(lo) + __popcll(hi);
}

// =============================================================================
// SEARCH / CLAIM HELPERS  (compile-time fmt dispatch)
// =============================================================================
// These are the building blocks the inner search and claim loops are
// built from. Each is templated on FMT (and IS_K where the metric
// branches), so a single per-candidate switch in the caller dispatches
// to a fully-inlined body — no per-block runtime fmt branching, no
// __noinline__ slow-path call for the cold quant formats.
//
// Cooperative warp model: every helper assumes a fully-active warp (all
// 32 lanes call simultaneously). `warp_f32_smem` and `warp_quant_smem`
// are per-warp shared-memory scratch buffers owned by the caller — each
// of the 4 warps has its own row of `warp_f32` / `warp_quant` so the
// round-trips run concurrently without contention.
//
// Building blocks below:
//
//   roundtrip_block_for_fmt   one block × outer → quantize → dequant
//   compute_pass_metric       (orig, recon) → (pass_metric, threshold)
//   search_scales_for_fmt     for one fmt: 6 candidates × live blocks →
//                             update best/fallback in shared smem
//   claim_passing_blocks…     mask-driven claim (no round-trip)

// One-block warp-cooperative round-trip: scale up by `outer`, quantize
// into the warp's scratch buffer, dequantize element `lane`. Returns
// the per-lane reconstructed float.
//
// The two `__syncwarp()` calls ensure all 32 lanes have written their
// scaled values before the (collectively-called) `quantize_block_for_fmt`
// reads them, and that the quantized bytes are visible before the
// per-lane `dequant_element_for_fmt` load. Since the scratch buffers
// are warp-private smem, no broader barrier is needed.
template <int FMT, bool IS_K>
__device__ __forceinline__ float roundtrip_block_for_fmt(
    float orig,
    float outer,
    int lane,
    float* warp_f32_smem,
    uint8_t* warp_quant_smem
) {
    warp_f32_smem[lane] = orig * outer;
    __syncwarp();
    quantize_block_for_fmt<FMT, IS_K>(warp_f32_smem, warp_quant_smem);
    __syncwarp();
    return dequant_element_for_fmt<FMT, IS_K>(warp_quant_smem, lane, outer);
}

// Compute the (pass_metric, threshold) pair for a single (orig, recon)
// block. The two sides use structurally different metrics:
//
//   K side  pass_metric    = mean_top4(|orig − recon|) · (1 / head_amax)
//           thr_to_compare = kthresh[b]   (q-relevance scaled, per block)
//
//   V side  pass_metric    = mean_{32 lanes}(orig − recon)² · (1 / head_amax²)
//           thr_to_compare = v_thr_sq     (sink-aware, constant per head)
//
// Both pass_metric and thr_to_compare are warp-uniform after the
// reductions inside this function: K's `mean_top4_abs_error_warp`
// broadcasts via __shfl_xor_sync, and V's MSE reduction is a sum over
// the 32 lanes followed by a constant divide. That uniformity is what
// makes the lane-0-only mask accumulation in `search_scales_for_fmt`
// safe — every lane sees the same predicate value.
//
// Templated on IS_K so the side-specific reductions vanish under
// inlining; the compiler emits two distinct functions and the call
// sites pick the right one based on a constexpr.
//
// Hoisted constants (`inv_head_amax`, `inv_head_amax_sq`, `v_thr_sq`)
// are computed once per side in `process_side`. The V-side hoist
// matters most: the prior implementation re-derived `thr_eff²` inside
// this function, which forced a warp-max reduction over
// `sink_weight[lane]` on every (block, fmt, scale) iteration even
// though sink_weight is fixed after Phase 2.5. With ~7,680 search
// iterations per side, that hoist eliminates ~15K redundant warp-max
// reductions per head.
template <bool IS_K>
__device__ __forceinline__ void compute_pass_metric(
    float orig,
    float recon,
    int b,
    float inv_head_amax,
    float inv_head_amax_sq,
    float v_thr_sq,
    const __half* kthresh,
    float& pass_metric,
    float& thr_to_compare
) {
    if (IS_K) {
        const float err = mean_top4_abs_error_warp(orig, recon, 1.0f);
        pass_metric    = err * inv_head_amax;
        thr_to_compare = __half2float(kthresh[b]);
    } else {
        // Per-lane squared error → warp-mean MSE.
        float mse = (orig - recon) * (orig - recon);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            mse += __shfl_xor_sync(0xffffffff, mse, off, 32);
        mse *= (1.0f / 32.0f);

        pass_metric    = mse * inv_head_amax_sq;
        thr_to_compare = v_thr_sq;  // precomputed once after Phase 2.5
    }
}

// Per-fmt scale × block search.
//
// For one format, walks NUM_SCALE_CANDIDATES outer scales; per scale,
// counts how many live blocks pass the threshold from
// `compute_pass_metric`. Updates the caller's best_* / fallback_*
// shared state, and sets *s_search_done = 1 if any scale reaches the
// 32-block slot quota. The caller's ci-loop reads s_search_done at the
// top of each iteration to decide whether to stop climbing the BPE
// ladder.
//
// Multi-warp work distribution
// ----------------------------
//
//   live blocks (compacted, amax-desc):
//       [b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ ...]
//
//                       ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
//          warp 0 →     ●           ●           ●     positions 0, 4, 8, …
//          warp 1 →        ●           ●           ●  positions 1, 5, 9, …
//          warp 2 →           ●           ●           positions 2, 6, …
//          warp 3 →              ●           ●        positions 3, 7, …
//
// Per position: all 32 lanes of the warp cooperate on one block's 32
// elements (one round-trip via `roundtrip_block_for_fmt`; per-lane
// pass_metric, all warp-reduced; broadcast).
//
// Per warp: tracks `my_count` (# blocks passing this warp's slice) and
// a per-warp pass mask `my_pass_lo/hi` covering the blocks it processed.
//
// Cross-warp: tid 0 sums the per-warp counts and OR-merges the pass
// masks (disjoint by construction — different warps process different
// block IDs). It then takes max over warp_amax_err to produce the
// (fmt, scale) summary, and updates the shared best/fallback fields.
//
// Best vs. fallback
// -----------------
// Best is tracked only over (fmt, scale) combos with total ≥ 32 (slot
// quota satisfied). Among winners, the lowest `aerr` (max error among
// passing blocks) wins. Fallback is tracked over ALL (fmt, scale) combos
// seen — it remembers the combo with the lowest max error across all
// alive blocks (passing AND failing). If no candidate hits 32 across the
// entire search, the caller falls back to that combo so the forced-claim
// path uses the format whose worst-case block error is smallest.
//
// Pass-mask cache
// ---------------
// When a (fmt, scale) becomes the new best (count ≥ 32 and lowest aerr)
// or the new fallback (lowest max-err across all blocks), its merged pass
// mask is saved alongside fmt/scale/err. The claim phase reads the
// saved mask and skips re-running the round-trip — search already
// determined which blocks pass at the winning combo, so recomputing
// would just produce the same answer:
//
//     bit b in s_best_pass_*  ←  block b passed at (s_best_fmt, s_best_scale)
//
// Round-trip scratch (`warp_f32_warp`, `warp_quant_warp`) is the
// warp's own slice; warps own disjoint scratch so the round-trips run
// concurrently without interference.
//
// Important: do NOT add an `if (*s_search_done) return;` at the top of
// the si loop — see the comment inside the function. Doing so commits
// the chosen format to whatever scale happened to satisfy 32 first,
// instead of the lowest-err scale across all six candidates. This
// degrades fixed-outer-scale formats (Q4_KS / Q8_KS) where `outer` is
// the actual quantization scale.
//
// Slot-stat parameters (slot_amax, safe_p95, safe_p80, slot_mean, safe_p25):
// These are the per-slot amax distribution statistics computed by tid 0
// in the compaction phase for this slot's alive set. They are warp-uniform
// (written to shared memory and loaded after __syncthreads) and feed
// `preferred_range` to produce the six outer-scale candidates.

template <int FMT, bool IS_K>
__device__ __noinline__ void search_scales_for_fmt(
    float slot_amax,   // max amax of the alive set
    float safe_p95,    // amax exceeded by 5% of alive blocks
    float safe_p80,    // amax exceeded by 20% of alive blocks
    float slot_mean,   // mean amax of the alive set
    float safe_p25,    // amax exceeded by 75% of alive blocks
    const __half* smem_data,
    const uint16_t* idx_compact,    // [live_count] alive entries in amax-desc order
    int live_count,                  // number of valid entries in idx_compact
    int tid, int warp_id, int lane,
    float* warp_f32_warp,            // warp_id's row of warp_f32 [32]
    uint8_t* warp_quant_warp,        // warp_id's row of warp_quant [MAX_QUANT_BLOCK_BYTES]
    const __half* kthresh,
    float inv_head_amax,
    float inv_head_amax_sq,
    float v_thr_sq,
    // Cross-warp aggregation scratch (one slot per warp)
    int*      warp_count,
    uint64_t* warp_pass_lo,
    uint64_t* warp_pass_hi,
    float*    warp_amax_err,
    // Cross-thread best/fallback state, written by tid 0 only
    int*      s_best_fmt,
    float*    s_best_scale,
    float*    s_best_err,
    uint64_t* s_best_pass_lo,
    uint64_t* s_best_pass_hi,
    int*      s_search_done,
    int*      s_fallback_fmt,
    float*    s_fallback_scale,
    float*    s_fallback_err,
    uint64_t* s_fallback_pass_lo,
    uint64_t* s_fallback_pass_hi
) {
    // Per-format candidate count. FP16-scale formats (Q4_0/1, Q5_0/1, Q8_0/1,
    // Q4_KS/Q8_KS, Q2_0/1, Q3_0/1, R16) have the property that the outer
    // scale algebraically cancels in the round-trip — the encoder recomputes
    // its internal scale from the (orig*outer) input, the decoder divides by
    // outer, and the two cancel modulo half-precision noise. Trialling six
    // outer candidates produces near-identical results, so we collapse to
    // a single si=0 (outer=1.0) trial. INT8-scale (Q1_S/Q1_A/Q2_S/Q2_A) and
    // Q0-family formats (no per-block scale) keep the full ladder.
    constexpr int kNumScales =
        outer_cancels_in_roundtrip<FMT>::value ? 1 : NUM_SCALE_CANDIDATES;

    // Run all kNumScales scales for this format. We must NOT early-return on
    // s_search_done set by an earlier scale of the same format: the contract
    // of search_scales_for_fmt is to find the lowest-amax_err scale among
    // the winners. Returning on the first hit commits s_best_scale to si=0
    // (=1.0f) most of the time, which is fine for block-internal-scale
    // formats (Q4_0 etc, outer is metadata) but degrades fixed-outer-scale
    // formats (Q4_KS / Q8_KS) where `outer` is the actual quantization scale.
    // The outer ci-loop in process_side already checks s_search_done to stop
    // climbing to higher BPE.
    //
    // Batched sync strategy: all warps write their per-(warp,si) accumulators
    // into distinct slots [warp_id * NUM_SCALE_CANDIDATES + si] without any
    // intermediate __syncthreads. A single __syncthreads after the loop makes
    // all slots visible; tid 0 reduces all si slices in one pass. The slot
    // stride uses NUM_SCALE_CANDIDATES (the smem allocation) regardless of
    // kNumScales — unused slots are simply not written or read.
    #pragma unroll
    for (int si = 0; si < kNumScales; si++) {
        const float outer = preferred_range(si, slot_amax, safe_p95, safe_p80, slot_mean, safe_p25);

        // Per-warp accumulators. idx_compact contains only alive entries —
        // no per-block alive_get probe needed. Each warp processes
        // live_count / FUSED_WARPS_PER_BLOCK blocks on average.
        float    my_amax_err = 0.0f;
        int      my_count    = 0;
        uint64_t my_pass_lo  = 0;
        uint64_t my_pass_hi  = 0;
        for (int i = warp_id; i < live_count; i += FUSED_WARPS_PER_BLOCK) {
            const int b = idx_compact[i];

            const float orig  = __half2float(smem_data[b * FUSED_WARP_SIZE + lane]);
            const float recon = roundtrip_block_for_fmt<FMT, IS_K>(
                orig, outer, lane, warp_f32_warp, warp_quant_warp);

            float pass_metric, thr_to_compare;
            compute_pass_metric<IS_K>(
                orig, recon, b,
                inv_head_amax, inv_head_amax_sq, v_thr_sq,
                kthresh,
                pass_metric, thr_to_compare);

            // pass_metric and thr_to_compare are warp-uniform after the
            // metric reductions, so all 32 lanes evaluate the same predicate.
            // Track max pass_metric across ALL blocks (passing and failing)
            // so the fallback can pick the candidate with the lowest worst-case
            // error rather than the highest BPE.
            my_amax_err = fmaxf(my_amax_err, pass_metric);
            if (pass_metric <= thr_to_compare) {
                if (lane == 0) {
                    if (b < 64) my_pass_lo |= (1ULL << b);
                    else        my_pass_hi |= (1ULL << (b - 64));
                    my_count++;
                }
            }
            // No per-warp early-exit on count — we don't know the cross-warp
            // total until the reduction below. The per-(fmt,scale) overshoot
            // cost is small relative to the cross-warp synchronisation we'd
            // otherwise need every block iteration.
        }

        // Write per-warp accumulators into the exclusive (warp, si) slot.
        // No __syncthreads here: each warp writes only to its own row
        // [warp_id * NUM_SCALE_CANDIDATES + si]; there is no cross-warp
        // aliasing, so no synchronisation is needed between si iterations.
        // Visibility to tid 0 is established by the __syncthreads below.
        if (lane == 0) {
            const int base = warp_id * NUM_SCALE_CANDIDATES + si;
            warp_count   [base] = my_count;
            warp_pass_lo [base] = my_pass_lo;
            warp_pass_hi [base] = my_pass_hi;
            warp_amax_err[base] = my_amax_err;
        }
    }

    // One sync makes all (warp, si) slots visible to tid 0.
    __syncthreads();

    // Cross-warp reduction by tid 0 across all si slices in one sequential pass.
    // Counts sum (warps process disjoint blocks); masks OR (disjoint →
    // OR == add for bit-presence); amax_err takes the max.
    if (tid == 0) {
        #pragma unroll
        for (int si = 0; si < kNumScales; si++) {
            const float outer = preferred_range(si, slot_amax, safe_p95, safe_p80, slot_mean, safe_p25);
            int      total = 0;
            uint64_t lo    = 0;
            uint64_t hi    = 0;
            float    aerr  = 0.0f;
            #pragma unroll
            for (int w = 0; w < FUSED_WARPS_PER_BLOCK; w++) {
                const int base = w * NUM_SCALE_CANDIDATES + si;
                total += warp_count   [base];
                lo    |= warp_pass_lo [base];
                hi    |= warp_pass_hi [base];
                aerr   = fmaxf(aerr, warp_amax_err[base]);
            }

            // Fallback: candidate with the lowest max error across all blocks.
            // amax_err here is taken over every alive block (passing and
            // failing) — see the inner loop. Picking the lowest steers the
            // forced-claim path toward the format that fits the worst block
            // best, rather than the highest-BPE / most-conservative option.
            if (aerr < *s_fallback_err) {
                *s_fallback_fmt     = FMT;
                *s_fallback_scale   = outer;
                *s_fallback_err     = aerr;
                *s_fallback_pass_lo = lo;
                *s_fallback_pass_hi = hi;
            }

            // Best: count >= 32 (slot quota) and lowest amax_err.
            if (total >= 32) {
                *s_search_done = 1;
                if (aerr < *s_best_err) {
                    *s_best_fmt     = FMT;
                    *s_best_scale   = outer;
                    *s_best_err     = aerr;
                    *s_best_pass_lo = lo;
                    *s_best_pass_hi = hi;
                }
            }
        }
    }
    __syncthreads();
}

// Mask-driven claim pass, single-threaded by tid 0.
//
// Walks idx_compact (alive entries in sort order) and claims any block
// whose bit is set in the cached pass mask, up to 32 blocks total. The
// search phase already determined which blocks pass at the winning
// (fmt, scale), so re-running the round-trip here would just recompute
// the same answer.
//
// Single-threaded because the work is bookkeeping only — smem reads,
// 2 global writes per claim, 1 atomicAnd on the alive mask. Parallelising
// would require a ballot + prefix-popcount dance similar to the
// second-pass fill in the caller for very little win — claim handles
// ≤ 32 blocks per slot, and on most slots the search hits the quota
// well before that.
//
// `alive_lo/alive_hi` is updated via `alive_clear` (atomicAnd) so the
// caller's second-pass fill loop observes the cleared bits without
// needing an explicit __threadfence_block. The closing __syncthreads
// orders the smem `*s_claimed_out` write so all threads see the
// returned count.
__device__ __forceinline__ int claim_passing_blocks_from_mask(
    int s,
    int head_id,
    int best_fmt,
    int tid,
    const uint16_t* idx_compact,
    int live_count,
    uint64_t pass_mask_lo,
    uint64_t pass_mask_hi,
    uint64_t* alive_lo,
    uint64_t* alive_hi,
    int* out_pal_map,
    int* out_eff_tags,
    int* s_claimed_out
) {
    if (tid == 0) {
        int claimed = 0;
        for (int i = 0; i < live_count && claimed < 32; i++) {
            const int b = idx_compact[i];
            const bool passes = (b < 64)
                ? (((pass_mask_lo >> b) & 1ULL) != 0ULL)
                : (((pass_mask_hi >> (b - 64)) & 1ULL) != 0ULL);
            if (passes) {
                out_pal_map [head_id * FUSED_HEAD_BLOCKS + b] = s;
                out_eff_tags[head_id * FUSED_HEAD_BLOCKS + b] = best_fmt;
                alive_clear(alive_lo, alive_hi, b);
                claimed++;
            }
        }
        *s_claimed_out = claimed;
    }
    __syncthreads();
    return *s_claimed_out;
}

// =============================================================================
// RUNTIME-FMT → COMPILE-TIME-FMT DISPATCH
// =============================================================================
// `with_select_fmt` is the only place the 19-arm fmt switch lives in
// the fused selection kernel. Given a runtime fmt value and a callable
// `f`, it invokes `f(FmtTag<FMT>{})` where FMT is the matching
// SELECT_FMT_* constant. Inside the callable, the tag's `::value` is
// a constexpr, so any call like `search_scales_for_fmt<FMT, IS_K>(...)`
// resolves at compile time and the body inlines fully.
//
// Usage in the kernel:
//
//     with_select_fmt(fmt, [&](auto tag) {
//         constexpr int FMT = decltype(tag)::value;
//         search_scales_for_fmt<FMT, /*IS_K=*/true>(...);
//     });
//
// Replaces the older macro-based dispatch: the cases are written once
// here, and the call sites read as normal C++ lambdas with no
// preprocessor games. The 19-arm switch costs i-cache once per
// candidate (≤ 6 candidates per slot × 4 slots × 2 sides = ≤ 48 calls
// per kernel block), vs. once per (block, fmt, scale) iteration if the
// switch lived in the inner loop.
template <int V>
struct FmtTag {
    static constexpr int value = V;
};

template <typename F>
__device__ __forceinline__ void with_select_fmt(int fmt, F&& f) {
    switch (fmt) {
        case SELECT_FMT_Q8_KS: f(FmtTag<SELECT_FMT_Q8_KS>{}); return;
        case SELECT_FMT_Q8_0:  f(FmtTag<SELECT_FMT_Q8_0>{});  return;
        case SELECT_FMT_Q8_1:  f(FmtTag<SELECT_FMT_Q8_1>{});  return;
        case SELECT_FMT_Q4_KS: f(FmtTag<SELECT_FMT_Q4_KS>{}); return;
        case SELECT_FMT_Q4_1:  f(FmtTag<SELECT_FMT_Q4_1>{});  return;
        case SELECT_FMT_Q4_0:  f(FmtTag<SELECT_FMT_Q4_0>{});  return;
        case SELECT_FMT_Q3_1:  f(FmtTag<SELECT_FMT_Q3_1>{});  return;
        case SELECT_FMT_Q3_0:  f(FmtTag<SELECT_FMT_Q3_0>{});  return;
        case SELECT_FMT_Q2_1:  f(FmtTag<SELECT_FMT_Q2_1>{});  return;
        case SELECT_FMT_Q2_0:  f(FmtTag<SELECT_FMT_Q2_0>{});  return;
        case SELECT_FMT_Q2_A:  f(FmtTag<SELECT_FMT_Q2_A>{});  return;
        case SELECT_FMT_Q2_S:  f(FmtTag<SELECT_FMT_Q2_S>{});  return;
        case SELECT_FMT_Q1_S:  f(FmtTag<SELECT_FMT_Q1_S>{});  return;
        case SELECT_FMT_Q0:    f(FmtTag<SELECT_FMT_Q0>{});    return;
        case SELECT_FMT_Q0_V:  f(FmtTag<SELECT_FMT_Q0_V>{});  return;
        case SELECT_FMT_Q1_A:  f(FmtTag<SELECT_FMT_Q1_A>{});  return;
        case SELECT_FMT_Q0_X:  f(FmtTag<SELECT_FMT_Q0_X>{});  return;
        case SELECT_FMT_Q0_M2: f(FmtTag<SELECT_FMT_Q0_M2>{}); return;
        case SELECT_FMT_Q0_M4: f(FmtTag<SELECT_FMT_Q0_M4>{}); return;
        default: return;  // BF16/F16/unknown: not in any candidate ladder
    }
}

// =============================================================================
// CANDIDATE-SET CONTRACT
// =============================================================================
// The host passes per-side `k_candidates` / `v_candidates` arrays —
// SELECT_FMT_* tags ordered by ascending BPE (best compression first).
// `search_scales_for_fmt` walks them in that order and stops at the
// first format that hits the 32-block slot quota; the result is the
// most aggressive format the slot can sustain. Within an equal-BPE tier
// the search picks the lowest-aerr format. If no candidate hits 32, the
// fallback path uses the highest-BPE candidate seen, with its best
// scale and (partial) pass mask.
//
// Kept here for ABI parity with the auxiliary sample/winner/reduce
// kernels below, which were already 4-warp.
#define SELECT_WARPS_PER_BLOCK 4

// =============================================================================
// INPUT DTYPE LOAD HELPER
// =============================================================================
// Loads element `idx` from a float-typed arena (F32, F16, or BF16). The
// dtype code is per arena and resolved by the caller from the per-head
// table entry. R16 is a separate block-structured format that carries
// Q activations alongside K values and is handled via `dequant_element`
// + `dequant_q_element`, not this helper.
#define SELECT_INPUT_F32  0
#define SELECT_INPUT_F16  1
#define SELECT_INPUT_BF16 2
#define SELECT_INPUT_R16  3   // K stored as block_r16 (d[32]=K, q[32]=Q); V still float

__device__ __forceinline__ float load_as_float(const void* __restrict__ data, int idx, int input_dtype) {
    if (input_dtype == SELECT_INPUT_F16) {
        return __half2float(((const __half*)data)[idx]);
    } else if (input_dtype == SELECT_INPUT_BF16) {
        return __bfloat162float(((const __nv_bfloat16*)data)[idx]);
    } else {
        return ((const float*)data)[idx];
    }
}

// =============================================================================
// BLOCK RELEVANCE  —  Σ(q²k²) / Σ(k²)  per block
// =============================================================================
// Coarse proxy for how much the query attends to a given K block.
//
// Derivation: the attention logit for an element pair is q·k. Squaring
// and summing across the 32 lanes of a block gives a non-negative
// magnitude signal; normalising by Σ k² removes the trivial dependence
// on the block's own amax, leaving a (block, query)-conditioned weight
// that's:
//
//   - large when q and k are correlated and aligned in the dominant lanes,
//   - small when q is approximately orthogonal to k or its energy lives
//     in lanes where k is small.
//
// Used in two places:
//
//   - K side: per-block `kthresh[b]` is z-scaled by q-relevance via
//     `k_threshold_scaled`. High relevance → tighter threshold (preserve
//     precision where it matters); low relevance → looser threshold.
//   - Phase 2.5: per-token sink detection uses the same per-element
//     q·k pattern, but at the per-token (not per-block) level.
//
// Q activations are only available when the source K block is in the
// R16 format; for other source formats `q_val == 0` everywhere, has_q
// is false, and the q-relevance path collapses (qrel = 1.0, kthresh
// falls back to sqrt(lo·hi)).

__device__ __forceinline__ float per_lane_qk2(float k, float q) {
    return __fmul_rn(__fmul_rn(q, q), __fmul_rn(k, k));
}

// Warp-reduce per-lane qk² (and k²) into the block relevance scalar.
// Returns Σ(q²k²) / Σ(k²) on every lane after the broadcast.
__device__ __forceinline__ float block_relevance_from_qk2(float qk2_lane, float k) {
    float qk2 = qk2_lane;
    float k2  = __fmul_rn(k, k);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        qk2 += __shfl_xor_sync(0xffffffff, qk2, offset, 32);
        k2  += __shfl_xor_sync(0xffffffff, k2,  offset, 32);
    }
    if (k2 == 0.0f) return 0.0f;
    return __fdiv_rn(qk2, k2);
}

// Convenience wrapper. Identical to qk2 → block_relevance_from_qk2,
// but retains the (k, q) signature where lanes pass their raw values.
__device__ __forceinline__ float block_relevance(float k, float q) {
    return block_relevance_from_qk2(per_lane_qk2(k, q), k);
}

// =============================================================================
// K THRESHOLD SCALING  —  per-block threshold from q-relevance z-score
// =============================================================================
// Anchor at sqrt(lo·hi) (the geometric mean), then scale by a
// z-conditioned multiplier:
//
//     z          = (q_relevance − q_median) / max(q_spread, 1e-8)
//     multiplier = exp(−z)
//     scaled     = sqrt(lo · hi) · multiplier
//     kthresh[b] = clamp(scaled, threshold_hi, threshold_lo)
//
// High q-relevance (z > 0) → smaller threshold (stricter on blocks the
// query attends to); low q-relevance (z < 0) → larger threshold (more
// permissive). The exp(−z) shape gives smooth, monotonic interpolation
// without the saturation kinks of a piecewise lerp.
//
// Convention: `threshold_lo > threshold_hi` numerically. lo = lenient
// (applied at low relevance), hi = strict (applied at high relevance).
// The clamp is anti-symmetric in lo/hi to make the bounds explicit.
//
// q_median and q_spread are precomputed per (chunk, head) by Pass 1
// (`approximate_q_relevance_quantiles`); spread is IQR-scaled so the
// z-score is approximately standard-normal regardless of the block's
// q-relevance distribution.
__device__ __forceinline__ float k_threshold_scaled(
    float threshold_lo,
    float threshold_hi,
    float q_relevance,
    float q_median,
    float q_spread
) {
    const float safe_spread = fmaxf(q_spread, 1.0e-8f);
    const float z           = __fdiv_rn(q_relevance - q_median, safe_spread);
    const float multiplier  = __expf(-z);
    const float base        = sqrtf(__fmul_rn(threshold_lo, threshold_hi));
    const float scaled      = __fmul_rn(base, multiplier);
    return fmaxf(threshold_hi, fminf(threshold_lo, scaled));
}


// =============================================================================
// ARENA HELPERS  —  paged KV-cache addressing
// =============================================================================
// The KV cache is paged: each (chunk, head) pair has its data stored
// inside an "arena" (a large per-arena contiguous buffer), addressed
// via per-head table entries that resolve to byte offsets and strides.
// These helpers are the small primitives used to walk that addressing
// scheme — invoked in tight loops over (chunk, head) inside every
// kernel in this file. Layout matches `paged_decode_kernel` exactly so
// the selection kernel sees the same bytes the attention kernel will:
//
//   - Palette4PerHeadEntry (28 × i64): per (arena, kv_head) with 4
//     sub-entries; compatibility lookup returns the palette-0 sub-entry.
//   - PerHeadTableEntry (7 × i64): per (arena, kv_head) with pre-
//     resolved byte offsets and chunk byte strides.
//   - head_gids: per-chunk per-head K/V global chunk IDs (interleaved).
//   - GID decomposition:
//        arena_idx = gid / arena_chunks
//        chunk_idx = gid % arena_chunks
//   - per_head_lookup → per_head_k_ptr / per_head_v_ptr for byte-level
//     addressing.
//   - Format tags come from per-head metadata, not a global dtype param.

// Float arena formats in `ArenaFormat::*` map directly onto the
// `SELECT_INPUT_*` codes used by `load_as_float`: F32=0, F16=1, BF16=2.
// Quantized formats are handled separately via `dequant_element_inline`.
__device__ __forceinline__ int arena_fmt_to_dtype_code(int fmt) {
    return fmt;  // ArenaFormat::F32=0, F16=1, BF16=2
}

// Byte size of a single 32-element block for the given ArenaFormat.
// Used to step through a chunk's quantized data block-by-block.
__device__ __forceinline__ int quant_block_bytes(int fmt) {
    switch (fmt) {
        case ArenaFormat::F16:  return 64;
        case ArenaFormat::R16:  return 128;
        case ArenaFormat::Q8_KS: return 36;
        case ArenaFormat::Q8_1: return 36;
        case ArenaFormat::Q8_0: return 34;
        case ArenaFormat::Q5_1: return 24;
        case ArenaFormat::Q5_0: return 22;
        case ArenaFormat::Q4_KS: return 20;
        case ArenaFormat::Q4_1: return 20;
        case ArenaFormat::Q4_0: return 18;
        case ArenaFormat::Q3_1: return 16;
        case ArenaFormat::Q3_0: return 14;
        case ArenaFormat::Q2_0: return 10;
        case ArenaFormat::Q2_1: return 12;
        case ArenaFormat::Q2_A: return 10;
        case ArenaFormat::Q2_S: return 9;
        case ArenaFormat::Q0_M4: return 8;
        case ArenaFormat::Q1_S: return 5;
        case ArenaFormat::Q0_M2: return 3;
        case ArenaFormat::Q0_V: return 2;
        case ArenaFormat::Q1_A: return 6;
        case ArenaFormat::Q0_X: return 2;
        case ArenaFormat::Q0: return 1;
        default: return 32;
    }
}

// Extract the Q-capture value at element index `idx` from a quantized
// block. Only R16 carries Q (in the `q[]` field); for any other format
// the compiler folds this to a constant 0.0f, so the call is zero-cost
// on non-R16 paths. Pass 1 uses this to decide whether q-relevance is
// usable for the head; the main kernel uses it to populate q_vals_half
// (Phase 1) which is then mean-reduced into `q_mean` for Phase 2.5
// sink detection.
__device__ __forceinline__ float dequant_q_element(const void* block_ptr, int idx, int fmt) {
    if (fmt == ArenaFormat::R16) {
        const block_r16* blk = reinterpret_cast<const block_r16*>(block_ptr);
        return __half2float(*reinterpret_cast<const __half*>(&blk->q[idx]));
    }
    return 0.0f;
}

// Look up the per-head table entry for (arena, head). Wraps
// `per_head_lookup` to bridge the raw int64 buffer the host passes in
// to the typed `Palette4PerHeadEntry` view.
__device__ __forceinline__ PerHeadTableEntry load_per_head_entry(
    const int64_t* __restrict__ per_head_table_raw,
    int arena_idx,
    int head_idx,
    int n_kv_head
) {
    const Palette4PerHeadEntry* per_head_table =
        reinterpret_cast<const Palette4PerHeadEntry*>(per_head_table_raw);
    return per_head_lookup(per_head_table, arena_idx, head_idx, n_kv_head);
}

// =============================================================================
// PER-(CHUNK, HEAD) QUANTILE KERNEL
// =============================================================================
// Pass 1 of the selection pipeline. One CUDA block per (chunk, head),
// 4 warps × 32 lanes = 128 threads. For each (chunk, head), emits:
//
//   k_head_amax_out [head_id]   max |K| across the 128 blocks
//   v_head_amax_out [head_id]   max |V| across the 128 blocks
//   k_head_p95_out  [head_id]   95th percentile of |K|, head-wide
//   v_head_p95_out  [head_id]   95th percentile of |V|, head-wide
//   q_relevance_median_out      median of per-block q-relevance
//   q_relevance_spread_out      IQR-scaled spread (≈ 2·1.4427 · IQR),
//                               used as the effective sigma in the
//                               z-scoring of q-relevance for K-side
//                               threshold scaling
//
// All quantiles are approximated via 64-bin histograms over the head's
// 4096 elements (128 blocks × 32 lanes). Bin width is amax / 63, so
// resolution is roughly 1.6% of amax — adequate for the downstream
// threshold/scale scaling, much cheaper than a true sort.
//
// Three histogram passes, run sequentially because they reuse the same
// shared memory:
//
//   (1)  K |x| histogram          →  k_head_p95_out
//   (2)  V |x| histogram          →  v_head_p95_out
//   (3)  q-relevance histogram (only when at least one block has Q),
//        bounded by [head_min, head_max] of the per-block q-relevance
//        scalars                  →  q1, median, q3  →  spread
//
// q-relevance is computed only for blocks whose source K is R16 (the
// format that carries Q); for other source formats q_val is uniformly
// 0, has_q is false, and only the amax + p95 statistics are emitted.
// A degenerate-distribution guard (sample_count ≤ 1 or no spread)
// returns median = spread = 0, which the selection kernel treats as
// "skip the z-scaling and use the geometric-mean threshold directly".

#define QREL_WARPS_PER_BLOCK SELECT_WARPS_PER_BLOCK
#define QREL_QUANTILE_THREADS (QREL_WARPS_PER_BLOCK * WARP_SIZE)
#define QREL_HIST_BINS 64

__global__ __launch_bounds__(QREL_QUANTILE_THREADS, 8) void approximate_q_relevance_quantiles(
    const int64_t* __restrict__ per_head_table_raw,
    const int64_t* __restrict__ head_gids,
    float* __restrict__ q_relevance_median_out,
    float* __restrict__ q_relevance_spread_out,
    float* __restrict__ k_head_amax_out,  // [total_heads] max(|K|) per (chunk, head)
    float* __restrict__ v_head_amax_out,  // [total_heads] max(|V|) per (chunk, head)
    float* __restrict__ k_head_p95_out,   // [total_heads] 95th pct of |K| per (chunk, head)
    float* __restrict__ v_head_p95_out,   // [total_heads] 95th pct of |V| per (chunk, head)
    int blocks_per_head,
    int total_heads,
    int n_kv_head,
    int arena_chunks
) {
    const int head_id       = blockIdx.x;
    const int tid           = threadIdx.x;
    const int warp_in_block = tid / WARP_SIZE;
    const int lane          = tid % WARP_SIZE;
    if (head_id >= total_heads) return;

    const int chunk_idx = head_id / n_kv_head;
    const int head_idx  = head_id % n_kv_head;
    const int gid_base  = chunk_idx * n_kv_head * 2;

    // K setup
    const int64_t k_gid      = __ldg(&head_gids[gid_base + head_idx * 2]);
    const int k_arena_idx     = (int)(k_gid / (int64_t)arena_chunks);
    const int k_chunk_idx     = (int)(k_gid - (int64_t)k_arena_idx * (int64_t)arena_chunks);
    PerHeadTableEntry ph_k    = load_per_head_entry(per_head_table_raw, k_arena_idx, head_idx, n_kv_head);
    const int   k_fmt         = per_head_get_k_format(ph_k);
    const char* k_chunk_data  = per_head_k_ptr(ph_k) + (int64_t)k_chunk_idx * ph_k.k_chunk_byte_stride;

    // V setup (for v_head_amax)
    const int64_t v_gid      = __ldg(&head_gids[gid_base + head_idx * 2 + 1]);
    const int v_arena_idx     = (int)(v_gid / (int64_t)arena_chunks);
    const int v_chunk_idx     = (int)(v_gid - (int64_t)v_arena_idx * (int64_t)arena_chunks);
    PerHeadTableEntry ph_v    = load_per_head_entry(per_head_table_raw, v_arena_idx, head_idx, n_kv_head);
    const int   v_fmt         = per_head_get_v_format(ph_v);
    const char* v_chunk_data  = per_head_v_ptr(ph_v) + (int64_t)v_chunk_idx * ph_v.v_chunk_byte_stride;

    __shared__ float warp_min[QREL_WARPS_PER_BLOCK];
    __shared__ float warp_max[QREL_WARPS_PER_BLOCK];
    __shared__ int   warp_count[QREL_WARPS_PER_BLOCK];
    __shared__ float warp_k_amax[QREL_WARPS_PER_BLOCK];
    __shared__ float warp_v_amax[QREL_WARPS_PER_BLOCK];
    // hist[0..QREL_HIST_BINS)            — K absolute-value histogram
    // hist[QREL_HIST_BINS..2*QREL_HIST_BINS) — V absolute-value histogram
    // Both are populated in a single data pass (4 passes → 3).
    __shared__ int   hist[2 * QREL_HIST_BINS];
    __shared__ float head_min;
    __shared__ float head_max;
    __shared__ int   sample_count;
    __shared__ float smem_head_k_amax;
    __shared__ float smem_head_v_amax;

    float local_min    = 0.0f;
    float local_max    = 0.0f;
    int   local_count  = 0;
    float local_k_amax = 0.0f;
    float local_v_amax = 0.0f;

    const float k_src_outer = per_head_get_k_scale(ph_k);
    const float v_src_outer = per_head_get_v_scale(ph_v);
    for (int block_in_head = warp_in_block; block_in_head < blocks_per_head; block_in_head += QREL_WARPS_PER_BLOCK) {
        float k_val, q_val;
        if (ArenaFormat::is_quantized(k_fmt)) {
            const int   k_blk_bytes = quant_block_bytes(k_fmt);
            const char* k_blk_ptr   = k_chunk_data + (int64_t)block_in_head * k_blk_bytes;
            k_val = dequant_element_inline<float, true>(k_blk_ptr, lane, k_fmt, k_src_outer);
            q_val = dequant_q_element(k_blk_ptr, lane, k_fmt);
        } else {
            k_val = load_as_float(k_chunk_data, block_in_head * 32 + lane, arena_fmt_to_dtype_code(k_fmt));
            q_val = 0.0f;
        }

        float v_val;
        if (ArenaFormat::is_quantized(v_fmt)) {
            const int   v_blk_bytes = quant_block_bytes(v_fmt);
            const char* v_blk_ptr   = v_chunk_data + (int64_t)block_in_head * v_blk_bytes;
            v_val = dequant_element_inline<float>(v_blk_ptr, lane, v_fmt, v_src_outer);
        } else {
            v_val = load_as_float(v_chunk_data, block_in_head * 32 + lane, arena_fmt_to_dtype_code(v_fmt));
        }

        // All 32 lanes participate in the warp reduce; only lane 0 accumulates.
        float k_block_amax = select_warp_max(fabsf(k_val));
        float v_block_amax = select_warp_max(fabsf(v_val));
        if (lane == 0) {
            local_k_amax = fmaxf(local_k_amax, k_block_amax);
            local_v_amax = fmaxf(local_v_amax, v_block_amax);
        }

        const int has_q = __any_sync(0xffffffff, q_val != 0.0f);
        if (has_q) {
            float q_relevance = block_relevance(k_val, q_val);
            q_relevance = __shfl_sync(0xffffffff, q_relevance, 0, 32);
            if (lane == 0) {
                if (local_count == 0) {
                    local_min = q_relevance;
                    local_max = q_relevance;
                } else {
                    local_min = fminf(local_min, q_relevance);
                    local_max = fmaxf(local_max, q_relevance);
                }
                local_count += 1;
            }
        }
    }

    if (lane == 0) {
        warp_min[warp_in_block]    = local_min;
        warp_max[warp_in_block]    = local_max;
        warp_count[warp_in_block]  = local_count;
        warp_k_amax[warp_in_block] = local_k_amax;
        warp_v_amax[warp_in_block] = local_v_amax;
    }
    __syncthreads();

    if (tid == 0) {
        // Accumulate head amax from all warps
        float head_k_amax = warp_k_amax[0];
        float head_v_amax = warp_v_amax[0];
        head_min     = warp_min[0];
        head_max     = warp_max[0];
        sample_count = warp_count[0];
        for (int w = 1; w < QREL_WARPS_PER_BLOCK; ++w) {
            head_k_amax = fmaxf(head_k_amax, warp_k_amax[w]);
            head_v_amax = fmaxf(head_v_amax, warp_v_amax[w]);
            if (warp_count[w] <= 0) continue;
            if (sample_count <= 0) {
                head_min     = warp_min[w];
                head_max     = warp_max[w];
                sample_count = warp_count[w];
                continue;
            }
            head_min      = fminf(head_min, warp_min[w]);
            head_max      = fmaxf(head_max, warp_max[w]);
            sample_count += warp_count[w];
        }

        k_head_amax_out[head_id] = head_k_amax;
        v_head_amax_out[head_id] = head_v_amax;
        smem_head_k_amax = head_k_amax;
        smem_head_v_amax = head_v_amax;

        if (sample_count <= 1 || !(head_max > head_min) || head_max <= 0.0f) {
            q_relevance_median_out[head_id] = 0.0f;
            q_relevance_spread_out[head_id] = 0.0f;
        }
    }
    __syncthreads();

    // === K + V absolute-value histograms (combined pass) → 95th percentiles ===
    // hist[0..QREL_HIST_BINS)            accumulates K |value| frequencies.
    // hist[QREL_HIST_BINS..2*QREL_HIST_BINS) accumulates V |value| frequencies.
    // One data pass replaces the prior two separate passes (4 passes → 3).
    for (int i = tid; i < 2 * QREL_HIST_BINS; i += QREL_QUANTILE_THREADS) hist[i] = 0;
    __syncthreads();
    {
        const float k_amax_safe     = (smem_head_k_amax > 1.0e-8f) ? smem_head_k_amax : 1.0f;
        const float v_amax_safe     = (smem_head_v_amax > 1.0e-8f) ? smem_head_v_amax : 1.0f;
        const float k_abs_inv_range = __fdiv_rn((float)(QREL_HIST_BINS - 1), k_amax_safe);
        const float v_abs_inv_range = __fdiv_rn((float)(QREL_HIST_BINS - 1), v_amax_safe);
        for (int block_in_head = warp_in_block; block_in_head < blocks_per_head; block_in_head += QREL_WARPS_PER_BLOCK) {
            float k_val;
            if (ArenaFormat::is_quantized(k_fmt)) {
                const int   k_blk_bytes = quant_block_bytes(k_fmt);
                const char* k_blk_ptr   = k_chunk_data + (int64_t)block_in_head * k_blk_bytes;
                k_val = dequant_element_inline<float, true>(k_blk_ptr, lane, k_fmt, k_src_outer);
            } else {
                k_val = load_as_float(k_chunk_data, block_in_head * 32 + lane, arena_fmt_to_dtype_code(k_fmt));
            }
            float v_val;
            if (ArenaFormat::is_quantized(v_fmt)) {
                const int   v_blk_bytes = quant_block_bytes(v_fmt);
                const char* v_blk_ptr   = v_chunk_data + (int64_t)block_in_head * v_blk_bytes;
                v_val = dequant_element_inline<float>(v_blk_ptr, lane, v_fmt, v_src_outer);
            } else {
                v_val = load_as_float(v_chunk_data, block_in_head * 32 + lane, arena_fmt_to_dtype_code(v_fmt));
            }
            int k_bin = (int)floorf(__fmul_rn(fabsf(k_val), k_abs_inv_range));
            k_bin = max(0, min(QREL_HIST_BINS - 1, k_bin));
            atomicAdd(&hist[k_bin], 1);
            int v_bin = (int)floorf(__fmul_rn(fabsf(v_val), v_abs_inv_range));
            v_bin = max(0, min(QREL_HIST_BINS - 1, v_bin));
            atomicAdd(&hist[QREL_HIST_BINS + v_bin], 1);
        }
        __syncthreads();
        if (tid == 0) {
            const int total_elems = blocks_per_head * 32;
            const int target_p95  = (int)floorf(0.95f * (float)(total_elems - 1));
            {
                int accum = 0, p95_bin = QREL_HIST_BINS - 1;
                for (int b = 0; b < QREL_HIST_BINS; ++b) {
                    accum += hist[b];
                    if (accum > target_p95) { p95_bin = b; break; }
                }
                float k_p95 = __fmul_rn((float)p95_bin, __fdiv_rn(k_amax_safe, (float)(QREL_HIST_BINS - 1)));
                k_head_p95_out[head_id] = (k_p95 > 1.0e-8f) ? k_p95 : k_amax_safe;
            }
            {
                int accum = 0, p95_bin = QREL_HIST_BINS - 1;
                for (int b = 0; b < QREL_HIST_BINS; ++b) {
                    accum += hist[QREL_HIST_BINS + b];
                    if (accum > target_p95) { p95_bin = b; break; }
                }
                float v_p95 = __fmul_rn((float)p95_bin, __fdiv_rn(v_amax_safe, (float)(QREL_HIST_BINS - 1)));
                v_head_p95_out[head_id] = (v_p95 > 1.0e-8f) ? v_p95 : v_amax_safe;
            }
        }
    }
    __syncthreads();

    if (sample_count <= 1 || !(head_max > head_min) || head_max <= 0.0f) {
        return;
    }

    for (int i = tid; i < QREL_HIST_BINS; i += QREL_QUANTILE_THREADS) {
        hist[i] = 0;
    }
    __syncthreads();

    const float inv_range = __fdiv_rn((float)(QREL_HIST_BINS - 1), head_max - head_min);
    for (int block_in_head = warp_in_block; block_in_head < blocks_per_head; block_in_head += QREL_WARPS_PER_BLOCK) {
        float k_val;
        float q_val;
        if (ArenaFormat::is_quantized(k_fmt)) {
            const int k_blk_bytes = quant_block_bytes(k_fmt);
            const char* k_blk_ptr = k_chunk_data + (int64_t)block_in_head * k_blk_bytes;
            k_val = dequant_element_inline<float, true>(k_blk_ptr, lane, k_fmt, k_src_outer);
            q_val = dequant_q_element(k_blk_ptr, lane, k_fmt);
        } else {
            const int k_elem_in_chunk = block_in_head * 32 + lane;
            k_val = load_as_float(k_chunk_data, k_elem_in_chunk, arena_fmt_to_dtype_code(k_fmt));
            q_val = 0.0f;
        }

        const int has_q = __any_sync(0xffffffff, q_val != 0.0f);
        if (has_q) {
            float q_relevance = block_relevance(k_val, q_val);
            q_relevance = __shfl_sync(0xffffffff, q_relevance, 0, 32);
            if (lane == 0) {
                int bin = (int)floorf(__fmul_rn(q_relevance - head_min, inv_range));
                bin = max(0, min(QREL_HIST_BINS - 1, bin));
                atomicAdd(&hist[bin], 1);
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        const int target_q1     = (int)floorf(0.25f * (float)(sample_count - 1));
        const int target_median = (int)floorf(0.50f * (float)(sample_count - 1));
        const int target_q3     = (int)floorf(0.75f * (float)(sample_count - 1));

        int accum = 0;
        int q1_bin = 0, median_bin = 0, q3_bin = QREL_HIST_BINS - 1;
        bool found_q1 = false, found_median = false, found_q3 = false;
        for (int b = 0; b < QREL_HIST_BINS; ++b) {
            accum += hist[b];
            if (!found_q1 && accum > target_q1)         { q1_bin = b;     found_q1 = true; }
            if (!found_median && accum > target_median) { median_bin = b; found_median = true; }
            if (!found_q3 && accum > target_q3)         { q3_bin = b;     found_q3 = true; break; }
        }

        const float scale = __fdiv_rn(head_max - head_min, (float)(QREL_HIST_BINS - 1));
        const float q1_v     = head_min + __fmul_rn((float)q1_bin, scale);
        const float median_v = head_min + __fmul_rn((float)median_bin, scale);
        const float q3_v     = head_min + __fmul_rn((float)q3_bin, scale);

        const float iqr = fmaxf(q3_v - q1_v, 1.0e-8f);
        const float z_scale = 2.0f;
        const float spread = __fmul_rn(iqr, __fmul_rn(z_scale, 1.4426950408889634f));

        q_relevance_median_out[head_id] = median_v;
        q_relevance_spread_out[head_id] = spread;
    }
}

// =============================================================================
// FORMAT TABLE INDEX  —  global BPE ranking
// =============================================================================
// Maps every supported format to a position in a global BPE-ascending
// ranking. Q0 (1 bit, most aggressive) is index 0; BF16 (16 bits, no
// quantization) is index 22.
//
// Used where we need to pick the "most conservative" tag from a set
// without referring to a candidate ordering — specifically:
//
//   - the head_tag computation in the main kernel (worst tag across
//     the 4 slots)
//   - `reduce_head_stats_format` (worst tag across all blocks of a head)
//
// Distinct from `format_bpe_x4`, which returns the actual BPE × 4 (used
// for BPE-ascending search ordering inside a candidate set). The table
// index includes F16/BF16 — useful for the head_tag write-out, where
// any slot's tag could be a float format.
//
// Must be defined before the fused palette4 kernel which uses it
// inside the process_side lambda.
__device__ __forceinline__ int format_table_index_cuda(int fmt) {
    switch (fmt) {
        case SELECT_FMT_Q0:         return 0;
        case SELECT_FMT_Q0_X:       return 1;
        case SELECT_FMT_Q0_V:       return 2;
        case SELECT_FMT_Q0_M2:      return 3;
        case SELECT_FMT_Q1_S:       return 4;
        case SELECT_FMT_Q1_A:       return 5;
        case SELECT_FMT_Q0_M4:      return 6;
        case SELECT_FMT_Q2_S:       return 7;
        case SELECT_FMT_Q2_0:       return 8;
        case SELECT_FMT_Q2_A:       return 9;
        case SELECT_FMT_Q2_1:       return 10;
        case SELECT_FMT_Q3_0:       return 11;
        case SELECT_FMT_Q3_1:       return 12;
        case SELECT_FMT_Q4_0:       return 13;
        case SELECT_FMT_Q4_1:       return 14;
        case SELECT_FMT_Q4_KS:      return 15;
        case SELECT_FMT_Q5_0:       return 16;
        case SELECT_FMT_Q5_1:       return 17;
        case SELECT_FMT_Q8_0:       return 18;
        case SELECT_FMT_Q8_1:       return 19;
        case SELECT_FMT_Q8_KS:      return 20;
        case SELECT_FMT_F16:        return 21;
        case SELECT_FMT_BF16:       return 22;
        default:                    return 20;
    }
}

// =============================================================================
// BITONIC SORT  —  128 entries by amax desc, single warp
// =============================================================================
// Sort phase of the main kernel. After Phase 1 each block has an
// (amax, idx) pair; this routine reorders both arrays so that
// idx_sorted[0] is the block with the largest amax,
// idx_sorted[127] the smallest. Sort positions are deterministic given
// input order.
//
// One warp owns the entire sort: 32 lanes × 4 elements per lane = 128.
// The main kernel uses two of these in parallel — warp 0 sorts K's
// (amax, idx) pair while warp 1 sorts V's. Warps 2 and 3 sit idle until
// the post-sort `__syncthreads()`. The bitonic network is ~50 stages of
// O(1) work plus a `__syncwarp()`, so sort is negligible in the
// kernel's overall time.

__device__ __forceinline__ void bitonic_sort_amax_desc(
    float*    __restrict__ s_amax,
    uint16_t* __restrict__ s_idx,
    int lane
) {
    // Each thread owns 4 elements (lanes 0..31, 4 elements each = 128 total).
    // Standard bitonic sort over 128 elements with want_desc ordering.
    for (int k = 2; k <= 128; k <<= 1) {
        for (int j = k >> 1; j >= 1; j >>= 1) {
            #pragma unroll
            for (int li = 0; li < 4; li++) {
                const int i = lane * 4 + li;
                const int p = i ^ j;
                if (p > i) {
                    const bool want_desc = ((i & k) == 0);
                    const bool swap = want_desc
                        ? (s_amax[i] < s_amax[p])
                        : (s_amax[i] > s_amax[p]);
                    if (swap) {
                        float    tmp_a = s_amax[i]; s_amax[i] = s_amax[p]; s_amax[p] = tmp_a;
                        uint16_t tmp_i = s_idx [i]; s_idx [i] = s_idx [p]; s_idx [p] = tmp_i;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// =============================================================================
// FUSED SELECTION + PALETTE4 KERNEL
// =============================================================================
//
// Pass 2 of the selection pipeline. One CUDA block per (chunk, head):
//
//     grid  = (total_heads, 1, 1)             // total_heads = chunks · n_kv_head
//     block = (FUSED_THREADS_PER_BLOCK, 1, 1) // 4 warps × 32 lanes = 128
//
// Each block reads 128 blocks of K and V, performs sink detection,
// sorts by amax, runs four BPE-ascending searches (one per palette
// slot, twice for K and V), and writes 4 (fmt, scale) palette slots
// plus a per-block slot index.
//
// blocks_per_head MUST equal exactly FUSED_HEAD_BLOCKS = 128 — the
// kernel returns early if that contract is violated.
//
// Algorithm summary
// -----------------
// Phases (all comments inside the kernel are tagged "Phase N: …"):
//
//   1.    Parallel load + per-block amax + per-block q-relevance.
//   2.5.  Per-token attention-sink detection. Compute the chunk's mean
//         Q vector; per-token sink score = q_mean · K_token / √d;
//         z-score against chunk-local μ/σ; sink_weight[t] = max(0, tanh(z)).
//         Hoist the V-side threshold² out of the search loop.
//   2.    Bitonic sort 128 (amax, idx) pairs descending by amax.
//   3.    Per-block K threshold from q-relevance z-score.
//   4+5.  Iterative slot search — see process_side lambda.
//
// Slot evolution
// --------------
// idx_sorted is amax-descending after Phase 2. Each slot iteration
// compacts the alive bitmask, claims 32 entries (passing-first then
// fill), and tombstones them in the bitmask for the next slot to skip.
//
//   slot 0:  ████████████████████████████████████████████████████████████  (live = 128)
//                                                                ┘ claims 32 (highest amax)
//
//   slot 1:  ████████████████████████████████████████████████              (live =  96)
//                                                            ┘ claims 32
//
//   slot 2:  ████████████████████████████████                              (live =  64)
//                                            ┘ claims 32
//
//   slot 3:  ████████████████                                              (live =  32)
//                            ┘ claims 32 (lowest amax)
//
//   Slot 0 ends up with the most conservative format (highest BPE) and
//   slot 3 with the most aggressive (lowest BPE) — the BPE-ascending
//   search exits at the first format where 32 blocks pass, and
//   high-amax blocks need more bits to stay within threshold.
//
// Why MSE on V and top-4 mean on K
// --------------------------------
// V's contribution to attention output is Σ_t a_t · v_t — an
// attention-weighted sum whose error budget is L2 (sum-squared), so
// mean-squared error is the structurally correct choice. Top-4 mean
// would over-penalise outlier elements that don't move the L2 of the
// actual computation V participates in.
//
// K errors enter the softmax via the per-element products q·k. Outliers
// in K dominate the score perturbation, but a single-lane spike often
// gets washed out by softmax — using max alone over-rejects benign
// candidates. The mean of the four largest weighted errors is a
// stable proxy that tracks the worst few errors without being all-or-
// nothing on a single lane.
//
// Sink protection (V-side)
// ------------------------
// Sink tokens carry disproportionate attention mass and dominate output
// error if their V is quantised loosely. The V threshold is interpolated
// between `v_threshold_lo` (lenient, non-sink) and `v_threshold_hi`
// (strict, peak-sink) using the chunk-wide max sink_weight:
//
//     v_thr_eff = v_threshold_lo + max_sink · (v_threshold_hi − v_threshold_lo)
//     v_thr_sq  = v_thr_eff²
//
// One strong sink token in the chunk forces stricter quality on every
// V block of the chunk. Conservative, but correct: any block could end
// up being attended to jointly with the sink during decode, so its
// error budget needs to account for that.
//
// FUSED_HEAD_BLOCKS / FUSED_WARP_SIZE are defined near the top of the
// file alongside SELECT_FMT_* so the templated search/claim helpers
// can reach them.

// Shared memory layout (~12.2 KB total — MaxShared carveout enables 8 blocks/SM)
// -------------------------------------------------------------------------------
//
//   smem_kv       [128 × 32]   f16    8,192 B  K then V values (aliased); f16
//                                              halves the prior 16 KB f32 buffer.
//   amax_k        [128]        f32      512 B  K block amax, sorted desc
//   amax_v        [128]        f32      512 B  V block amax, sorted desc
//   kidx, vidx    [128]        u16    2×256 B  sorted block IDs (read-only after sort)
//   qrel_k        [128]        f16      256 B  per-block q-relevance (f16 saves 256 B)
//   kthresh       [128]        f16      256 B  per-block K threshold  (f16 saves 256 B)
//   q_mean        [128]        f32      512 B  per-head_dim mean Q for Phase 2.5
//   sink_score    [32]         f32      128 B  per-token raw Q·K (2.5)
//   sink_weight   [32]         f32      128 B  per-token weight ∈ [0,1] (2.5)
//   warp_f32      [4][32]      f32      512 B  per-warp round-trip scratch
//   warp_quant    [4][36]      u8       144 B  per-warp quantized block scratch
//   slot_tags     [4]          i32       16 B  winning fmt per slot
//   slot_scales   [4]          f32       16 B  winning outer scale per slot
//   k/v_alive_lo,k/v_alive_hi  u64       32 B  K/V alive bitmask (1 bit/block)
//   idx_compact   [128]        u16      256 B  compacted live entries per slot
//   warp_count    [4×6]        i32       96 B  search reduction scratch [warp][si]
//   warp_pass_lo  [4×6]        u64      192 B  pass mask, low half      [warp][si]
//   warp_pass_hi  [4×6]        u64      192 B  pass mask, high half     [warp][si]
//   warp_amax_err [4×6]        f32       96 B  max passing error        [warp][si]
//   s_best_*, s_fallback_*               ~96 B  search winner / fallback state
//   s_warp_pop[4], s_first_alive[4]      ~32 B  compaction scratch
//   s_live_count, s_slot_{amax,p95,p80,mean,p25}  ~24 B  cross-warp scalars
//   s_claim_count                          ~4 B  pass-1 claim count
//   spill_v_chunk_ptr              u64      8 B  } V-reload register spill slots —
//   spill_v_src_fmt                i32      4 B  } written end of Phase 1 by tid 0,
//   spill_v_src_outer              f32      4 B  } read before V reload.  Keeps 4
//                                               } regs free across K process_side.
//
// Total: ~12.7 KB.  8 × 12.7 KB = 101.6 KB < 102.4 KB (MaxShared budget on Ada).
// launch_bounds(128, 8) caps REG at 64; search_scales_for_fmt is __noinline__ so
// its frame is separate, preventing the inlined loop body from inflating the
// kernel's combined register count.
//
// Tombstoning lives in `*_alive_*` (no -1 sentinel in idx arrays);
// per-slot `idx_compact` rebuild lets the search inner loop walk only
// live entries without an alive_get probe per iteration.
//
// Pass-mask packing
// -----------------
// Each block has an ID b ∈ [0, 128). A passing block sets bit b in a
// 128-bit mask carried as two u64 words (lo: b<64, hi: b≥64). The mask
// is set DURING the search; when a (fmt, scale) wins, its mask becomes
// the new s_best_pass_*. The claim phase reads this mask and skips the
// round-trip — search already determined which blocks pass at the
// winning combo.

extern "C" __global__ __launch_bounds__(FUSED_THREADS_PER_BLOCK, 8) void select_kv_format_palette4_paged(
    const int64_t* __restrict__ per_head_table_raw,
    const int64_t* __restrict__ head_gids,
    const float*   __restrict__ q_relevance_median,   // [total_heads]
    const float*   __restrict__ q_relevance_spread,   // [total_heads]
    const float*   __restrict__ k_head_amax_in,       // [total_heads]
    const float*   __restrict__ v_head_amax_in,       // [total_heads]
    const float*   __restrict__ k_head_p95_in,        // [total_heads] (unused; kept for ABI)
    const float*   __restrict__ v_head_p95_in,        // [total_heads] (unused; kept for ABI)
    const int*     __restrict__ k_candidates,
    const int*     __restrict__ v_candidates,
    int num_k_candidates,
    int num_v_candidates,
    float k_threshold_hi,
    float k_threshold_lo,
    float v_threshold_hi,   // strict — applied at peak sink (w_max=1)
    float v_threshold_lo,   // lenient — applied at non-sink (w_max=0). Convention: lo > hi numerically.
    int total_heads,
    int blocks_per_head,    // must equal FUSED_HEAD_BLOCKS = 128
    int n_kv_head,
    int arena_chunks,
    int*   __restrict__ k_palette_tags,            // [total_heads * 4]
    int*   __restrict__ v_palette_tags,            // [total_heads * 4]
    float* __restrict__ k_palette_scale,           // [total_heads * 4] outer scale per slot
    float* __restrict__ v_palette_scale,           // [total_heads * 4] outer scale per slot
    int*   __restrict__ k_palette_map,             // [total_heads * FUSED_HEAD_BLOCKS]
    int*   __restrict__ v_palette_map,             // [total_heads * FUSED_HEAD_BLOCKS]
    int*   __restrict__ k_effective_block_tags,    // [total_heads * FUSED_HEAD_BLOCKS]
    int*   __restrict__ v_effective_block_tags,    // [total_heads * FUSED_HEAD_BLOCKS]
    int*   __restrict__ k_head_tags,               // [total_heads]
    int*   __restrict__ v_head_tags,               // [total_heads]
    float* __restrict__ q_relevance_out            // [total_heads * FUSED_HEAD_BLOCKS] or nullptr
) {
    // Single aliased buffer (f16): K data lives here through Phase 2.5 + K
    // process_side, then V data is reloaded before V process_side.  f16 halves
    // the 16 KB of the prior f32 buffer; the remaining precision is sufficient
    // for both the sink-score dot product and the roundtrip-error comparison.
    // Combined with MaxShared this is the binding smem constraint for 5 blocks/SM.
    __shared__ __half   smem_kv    [FUSED_HEAD_BLOCKS * FUSED_WARP_SIZE];
    // q_vals_half removed: q_mean is now computed inline during Phase 1 via
    // warp reduce, saving 8 KB and eliminating the Phase 2.5(a) readback loop.
    __shared__ float    amax_k     [FUSED_HEAD_BLOCKS];
    __shared__ float    amax_v     [FUSED_HEAD_BLOCKS];
    // Block ids fit in 8 bits (0..127) — use uint16_t to halve smem footprint
    // and free us from the prior int-tombstone (-1) sentinel; tombstoning now
    // lives in the alive bitmasks below.
    __shared__ uint16_t kidx       [FUSED_HEAD_BLOCKS];
    __shared__ uint16_t vidx       [FUSED_HEAD_BLOCKS];
    // f16 saves 256 B each vs f32; precision is adequate for threshold comparisons.
    __shared__ __half   qrel_k     [FUSED_HEAD_BLOCKS];
    __shared__ __half   kthresh    [FUSED_HEAD_BLOCKS];
    __shared__ float    q_mean     [FUSED_HEAD_BLOCKS];   // Phase 2.5 — mean Q per head_dim slot
    __shared__ float    sink_score [FUSED_WARP_SIZE];     // Phase 2.5 — raw per-token Q·K alignment
    __shared__ float    sink_weight[FUSED_WARP_SIZE];     // Phase 2.5 — per-token weight ∈ [0, 1]
    // Per-warp round-trip scratch — each warp owns one row, so all 4 warps
    // can run independent (fmt, scale, block) round-trips in parallel
    // without contending for buffer space.
    __shared__ float    warp_f32   [FUSED_WARPS_PER_BLOCK][FUSED_WARP_SIZE];
    __shared__ uint8_t  warp_quant [FUSED_WARPS_PER_BLOCK][MAX_QUANT_BLOCK_BYTES];
    __shared__ int      slot_tags  [4];
    __shared__ float    slot_scales[4];
    // 128-bit alive mask per side; bit b set iff K/V block b is unclaimed.
    // Replaces the prior idx_sorted[i] = -1 tombstoning. Updated atomically
    // by `alive_clear` since multiple lanes can target distinct bits during
    // the parallel second-pass fill.
    __shared__ uint64_t k_alive_lo, k_alive_hi;
    __shared__ uint64_t v_alive_lo, v_alive_hi;
    // Compacted live-index scratch, rebuilt at the top of each slot from
    // `kidx`/`vidx` + the alive bitmask. The search inner loop walks
    // idx_compact[0..live_count) instead of idx_sorted with a per-block
    // alive probe, so the hot path becomes branch-free over the iteration
    // count. Reused across K and V because the two process_side calls run
    // sequentially.
    __shared__ uint16_t idx_compact[FUSED_HEAD_BLOCKS];

    // Cross-warp search aggregation — one slot per (warp, si) pair.
    // All NUM_SCALE_CANDIDATES scales are accumulated in parallel across the
    // si loop without intermediate __syncthreads; tid 0 reduces all slots in
    // one pass after a single __syncthreads at the end of the si loop.
    // Layout: [warp_id * NUM_SCALE_CANDIDATES + si].
    __shared__ int      warp_count   [FUSED_WARPS_PER_BLOCK * NUM_SCALE_CANDIDATES];
    __shared__ uint64_t warp_pass_lo [FUSED_WARPS_PER_BLOCK * NUM_SCALE_CANDIDATES];
    __shared__ uint64_t warp_pass_hi [FUSED_WARPS_PER_BLOCK * NUM_SCALE_CANDIDATES];
    __shared__ float    warp_amax_err[FUSED_WARPS_PER_BLOCK * NUM_SCALE_CANDIDATES];

    // Cross-thread best/fallback state for the current slot's search.
    // Lives in shared because the search is now multi-warp: every warp
    // reads `s_search_done` to decide whether to advance to the next
    // candidate, and tid 0 updates the best/fallback fields after the
    // cross-warp reduction.
    __shared__ int      s_best_fmt;
    __shared__ float    s_best_scale;
    __shared__ float    s_best_err;
    __shared__ uint64_t s_best_pass_lo;
    __shared__ uint64_t s_best_pass_hi;
    __shared__ int      s_search_done;
    __shared__ int      s_fallback_fmt;
    __shared__ float    s_fallback_scale;
    __shared__ float    s_fallback_err;
    __shared__ uint64_t s_fallback_pass_lo;
    __shared__ uint64_t s_fallback_pass_hi;

    // Per-slot scratch shared with all threads: live count and slot-level
    // amax statistics (max, p95, p80, mean) of the unclaimed set, populated
    // during compaction by tid 0's alive-walk and broadcast to all threads.
    __shared__ int      s_warp_pop[FUSED_WARPS_PER_BLOCK];
    __shared__ int      s_first_alive[FUSED_WARPS_PER_BLOCK];
    __shared__ int      s_live_count;
    __shared__ float    s_slot_amax;   // max amax of unclaimed set
    __shared__ float    s_slot_p95;    // 95th-percentile amax (5% of blocks exceed this)
    __shared__ float    s_slot_p80;    // 80th-percentile amax (20% of blocks exceed this)
    __shared__ float    s_slot_mean;   // mean amax of unclaimed set
    __shared__ float    s_slot_p25;    // 25th-percentile amax (75% of blocks exceed this)

    // Single-int scratch used by `claim_passing_blocks_from_mask` to return
    // the pass-1 claim count from tid 0 to the rest of the block.
    __shared__ int      s_claim_count;

    // Register-spill slots for V-reload data.  Written by tid 0 at the end of
    // the Phase 1 scope (alongside alive-mask init), made visible by the
    // __syncthreads that follows.  Read back before V reload.  Keeping these 4
    // regs (v_chunk_data×2, v_src_fmt, v_src_outer) out of the frame across
    // Phase 2.5 / Phase 2 / Phase 3 / K process_side is worth 16 B of smem.
    __shared__ unsigned long long spill_v_chunk_ptr;
    __shared__ int                spill_v_src_fmt;
    __shared__ float              spill_v_src_outer;

    const int head_id = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid / FUSED_WARP_SIZE;
    const int lane    = tid % FUSED_WARP_SIZE;

    if (head_id >= total_heads || blocks_per_head != FUSED_HEAD_BLOCKS) return;

    const int chunk_id = head_id / n_kv_head;
    const int head_idx = head_id % n_kv_head;

    const int     gid_base        = chunk_id * n_kv_head * 2;
    const int64_t k_gid           = __ldg(&head_gids[gid_base + head_idx * 2]);
    const int64_t v_gid           = __ldg(&head_gids[gid_base + head_idx * 2 + 1]);
    const int     k_arena_idx     = (int)(k_gid / (int64_t)arena_chunks);
    const int     k_chunk_in_arena= (int)(k_gid - (int64_t)k_arena_idx * (int64_t)arena_chunks);
    const int     v_arena_idx     = (int)(v_gid / (int64_t)arena_chunks);
    const int     v_chunk_in_arena= (int)(v_gid - (int64_t)v_arena_idx * (int64_t)arena_chunks);

    const float q_med    = __ldg(&q_relevance_median[head_id]);
    const float q_spread = __ldg(&q_relevance_spread[head_id]);

    // ── Phase 1 scope ─────────────────────────────────────────────────────────
    // k_ph / v_ph struct lifetime (≈14 regs each) is bounded to this block so
    // the compiler can reclaim them once the derived pointers/scalars are
    // extracted.  v_chunk_data / v_src_fmt / v_src_outer are also scoped here:
    // they are spilled to smem at the end and reloaded just before V reload,
    // keeping them out of the register frame across Phase 2.5, Phase 2, Phase 3,
    // and the entire K process_side — the largest contiguous live range.
    {
    PerHeadTableEntry k_ph = load_per_head_entry(per_head_table_raw, k_arena_idx, head_idx, n_kv_head);
    PerHeadTableEntry v_ph = load_per_head_entry(per_head_table_raw, v_arena_idx, head_idx, n_kv_head);

    const int   k_src_fmt    = per_head_get_k_format(k_ph);
    const int   v_src_fmt    = per_head_get_v_format(v_ph);
    const char* k_chunk_data = per_head_k_ptr(k_ph) + (int64_t)k_chunk_in_arena * k_ph.k_chunk_byte_stride;
    const char* v_chunk_data = per_head_v_ptr(v_ph) + (int64_t)v_chunk_in_arena * v_ph.v_chunk_byte_stride;
    const float k_src_outer  = per_head_get_k_scale(k_ph);
    const float v_src_outer  = per_head_get_v_scale(v_ph);
    // k_ph / v_ph are dead after this point; the compiler reclaims their regs.

    // ── Phase 1: Load all 128 blocks; compute per-block amax and q-relevance ──
    // Multi-warp stride: each of the 4 warps owns 32 blocks (warp_id, +4, +8 …).
    // Lanes within a warp cooperate to load one block's 32 elements; warp-
    // local reductions (`__shfl_xor_sync`) compute amax / q-relevance per
    // block. No cross-warp dependencies in this phase — the closing
    // __syncthreads makes Phase 2.5/2 see consistent smem state.
    for (int blk = warp_id; blk < FUSED_HEAD_BLOCKS; blk += FUSED_WARPS_PER_BLOCK) {
        float k_val = 0.0f, q_val = 0.0f, v_val = 0.0f;

        if (ArenaFormat::is_quantized(k_src_fmt)) {
            const char* k_blk = k_chunk_data + (int64_t)blk * quant_block_bytes(k_src_fmt);
            k_val = dequant_element_inline<float, true>(k_blk, lane, k_src_fmt, k_src_outer);
            q_val = dequant_q_element(k_blk, lane, k_src_fmt);
        } else {
            k_val = load_as_float(k_chunk_data, blk * 32 + lane, arena_fmt_to_dtype_code(k_src_fmt));
        }
        if (ArenaFormat::is_quantized(v_src_fmt)) {
            const char* v_blk = v_chunk_data + (int64_t)blk * quant_block_bytes(v_src_fmt);
            v_val = dequant_element_inline<float>(v_blk, lane, v_src_fmt, v_src_outer);
        } else {
            v_val = load_as_float(v_chunk_data, blk * 32 + lane, arena_fmt_to_dtype_code(v_src_fmt));
        }

        smem_kv   [blk * 32 + lane] = __float2half(k_val);   // K stored as f16; V discarded (reloaded before V process_side)

        float k_abs = fabsf(k_val);
        float v_abs = fabsf(v_val);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            k_abs = fmaxf(k_abs, __shfl_xor_sync(0xffffffff, k_abs, off));
            v_abs = fmaxf(v_abs, __shfl_xor_sync(0xffffffff, v_abs, off));
        }

        // Inline q_mean: all 32 lanes hold q_val for one position in this block,
        // so a warp sum gives the mean without a staging buffer.  Saves 8 KB vs.
        // storing to q_vals_half and reading back in Phase 2.5(a).
        float q_mean_sum = q_val;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            q_mean_sum += __shfl_xor_sync(0xffffffff, q_mean_sum, off);
        if (lane == 0) q_mean[blk] = q_mean_sum * (1.0f / 32.0f);

        float qr = 1.0f;
        const int has_q = __any_sync(0xffffffff, q_val != 0.0f);
        if (has_q) {
            qr = block_relevance(k_val, q_val);
            qr = __shfl_sync(0xffffffff, qr, 0);
        }

        if (lane == 0) {
            // Per-block hash jitter (~6e-8) added to the amax values to
            // break sort ties on tied amax. Partial-tail chunks zero-pad
            // positions past `token_count`, producing many near-equal
            // small amax that — under the bitonic sort's tie behaviour —
            // drift toward near-monotonic block-index order; the claim
            // phase then assigns long contiguous dim ranges to a single
            // palette (tail-chunk clustering). The jitter sits well
            // below any real activation amax (≥ 1e-2 typical), so the
            // format-search and threshold paths on real data are
            // unaffected. Must match the Rust mirror's
            // `amax_tie_jitter` in `cpu_selection.rs` byte-for-byte.
            const unsigned int j_h = ((unsigned int)blk * 2654435761u) ^ 0x9e3779b9u;
            const float j_v = __int_as_float((int)((j_h & 0x007fffffu) | 0x33800000u));
            amax_k[blk] = k_abs + j_v;
            amax_v[blk] = v_abs + j_v;
            qrel_k[blk] = __float2half(qr);
            kidx  [blk] = (uint16_t)blk;
            vidx  [blk] = (uint16_t)blk;
        }

        if (lane == 0 && q_relevance_out != nullptr) {
            q_relevance_out[head_id * FUSED_HEAD_BLOCKS + blk] = qr;
        }
    }

    // Spill V-reload data + initialise alive masks in one tid-0 write.
    // The __syncthreads below makes both visible to all threads.
    // v_chunk_data / v_src_fmt / v_src_outer die at the scope close below.
    if (tid == 0) {
        spill_v_chunk_ptr  = (unsigned long long)(uintptr_t)v_chunk_data;
        spill_v_src_fmt    = v_src_fmt;
        spill_v_src_outer  = v_src_outer;
        // Initialize alive masks: bits 0..127 set, upper bits 128..191 zero.
        k_alive_lo = ~0ULL;
        k_alive_hi = ~0ULL;
        v_alive_lo = ~0ULL;
        v_alive_hi = ~0ULL;
    }
    }  // ── end Phase 1 scope: k_ph, v_ph, k/v_chunk_data, k/v_src_fmt/outer freed ──
    __syncthreads();

    // ── Phase 2.5: per-token attention-sink detection ─────────────────────────
    // Compute Q-K alignment-based sink score per token, then statistically
    // detect sinks via a chunk-local z-score with tanh weighting.
    //
    // Mechanism: tokens whose K vector aligns with the average Q direction
    // of the chunk receive more attention from typical queries. This is a
    // pre-softmax proxy for attention received, capturing both K-norm
    // anomalies (registration sinks at sequence start) and K-direction
    // anomalies (mid-sequence emergent sinks documented in KVSink).
    //
    // Output: sink_weight[t] ∈ [0,1] per token. Above-average tokens get
    // positive weight (saturating at z≈2); below-average tokens get 0.
    // Parameter-free: detection threshold, sharpness, and floor are all
    // derived from chunk-local statistics via tanh(z-score).
    //
    // Used by V-side selection to lerp the per-block error threshold between
    // v_threshold_lo (lenient) and v_threshold_hi (strict) using the maximum
    // sink_weight among the 32 tokens in the block — one strong sink in a
    // block forces tighter quality on the entire block.

    // q_mean is populated per-block during Phase 1 (inline warp reduce).
    // Phase 2.5(a) smem readback loop removed; the existing __syncthreads
    // above already makes q_mean visible to Phase 2.5(b).

    // (b) Per-token sink_score: dot product of q_mean[0..127] with K[blk, token].
    //     Each lane computes one token's score; all 4 warps participate.
    //     warp_f32 is idle here (only used inside search_scales_for_fmt) so
    //     we reuse it as partial-sum scratch without extra smem cost.
    //     Each warp accumulates FUSED_HEAD_BLOCKS/FUSED_WARPS_PER_BLOCK = 32 blocks;
    //     warp 0 then sums the 4 partials and continues with stats + weight.
    {
        float score = 0.0f;
        const int blk_begin = warp_id * (FUSED_HEAD_BLOCKS / FUSED_WARPS_PER_BLOCK);
        const int blk_end   = blk_begin + (FUSED_HEAD_BLOCKS / FUSED_WARPS_PER_BLOCK);
        #pragma unroll 4
        for (int blk = blk_begin; blk < blk_end; blk++) {
            score += q_mean[blk] * __half2float(smem_kv[blk * 32 + lane]);
        }
        warp_f32[warp_id][lane] = score;
    }
    __syncthreads();
    // (c)+(d) μ/σ statistics and sink_weight: warp-wide reductions, warp 0 only.
    if (warp_id == 0) {
        // Sum the 4 partial dot products into the final per-token score.
        float score = warp_f32[0][lane] + warp_f32[1][lane]
                    + warp_f32[2][lane] + warp_f32[3][lane];
        sink_score[lane] = score * rsqrtf((float)FUSED_HEAD_BLOCKS);
        __syncwarp();

        // (c) Chunk-local statistics: mean and std of sink_score across 32 tokens.
        //     Warp reductions; mu and sigma broadcast to all lanes.
        const float s = sink_score[lane];
        float ssum = s;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            ssum += __shfl_xor_sync(0xffffffff, ssum, off, 32);
        const float mu = ssum * (1.0f / 32.0f);

        const float dev = s - mu;
        float dev2 = dev * dev;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dev2 += __shfl_xor_sync(0xffffffff, dev2, off, 32);
        const float sigma = sqrtf(dev2 * (1.0f / 32.0f));

        // (d) Per-token sink_weight via z-score → tanh, clamped at zero.
        //     Below-average tokens (z < 0) get 0 (no protection).
        //     Above-average tokens get tanh(z) ∈ (0, 1), saturating at z≈2.
        const float safe_sigma = fmaxf(sigma, 1.0e-8f);
        const float z = (s - mu) / safe_sigma;
        sink_weight[lane] = fmaxf(0.0f, tanhf(z));
    }
    __syncthreads();

    // ── Phase 2.5 (post): Hoist V-side threshold² out of the search loop ───
    // `v_thr_sq` is the constant V-side `thr_to_compare` used by every block
    // round-trip in process_side. It depends only on `sink_weight` (now
    // finalised) and the two threshold knobs — i.e. it's invariant across
    // (slot, fmt, scale, block). Computing it once here saves ~15K redundant
    // warp-max reductions per head in `compute_pass_metric<false>`.
    //
    // All 4 warps redundantly compute the same value from the shared
    // `sink_weight[0..31]` — cheaper than a smem broadcast for a single
    // warp-max + scalar arithmetic.
    float v_w_max_local = sink_weight[lane];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v_w_max_local = fmaxf(v_w_max_local, __shfl_xor_sync(0xffffffff, v_w_max_local, off, 32));
    const float v_thr_eff = v_threshold_lo + v_w_max_local * (v_threshold_hi - v_threshold_lo);
    const float v_thr_sq  = v_thr_eff * v_thr_eff;

    // ── Phase 2: Sort K and V blocks separately by their respective amax desc ──
    // Warp 0 sorts K, warp 1 sorts V — runs concurrently on different smem
    // arrays. The bitonic network is warp-local (lane-owned 4 elements
    // each, intra-warp `__shfl_xor` substitute for syncthreads); warps 2/3
    // idle through the sort.
    if (warp_id == 0) {
        bitonic_sort_amax_desc(amax_k, kidx, lane);
    } else if (warp_id == 1) {
        bitonic_sort_amax_desc(amax_v, vidx, lane);
    }
    __syncthreads();

    // ── Phase 3: Precompute per-block K error thresholds (parallelised over all 128 threads) ──
    if (tid < FUSED_HEAD_BLOCKS) {
        const int b = tid;
        kthresh[b] = __float2half((q_spread > 1.0e-8f)
            ? k_threshold_scaled(k_threshold_lo, k_threshold_hi, __half2float(qrel_k[b]), q_med, q_spread)
            : sqrtf(k_threshold_lo * k_threshold_hi));
    }

    const float k_head_amax_val = fmaxf(__ldg(&k_head_amax_in[head_id]), 1.0e-8f);
    // v_head_amax_val loaded lazily just before V process_side so it is not live
    // across the K process_side register frame.

    // Defensive: pre-zero palette_map so unclaimed blocks (shouldn't occur under normal
    // operation) always have a valid slot index rather than uninitialized device memory.
    if (tid < FUSED_HEAD_BLOCKS) {
        const int b = tid;
        k_palette_map[head_id * FUSED_HEAD_BLOCKS + b] = 0;
        v_palette_map[head_id * FUSED_HEAD_BLOCKS + b] = 0;
    }
    __syncthreads();

    // ── Phase 4+5: Iterative slot search (process_side lambda) ──────────
    //
    // Runs once for K and once for V. For each of 4 slots:
    //
    //   (a) Compact alive entries from idx_sorted into idx_compact.
    //       Each warp ballots its 32-position chunk of the alive bitmask;
    //       per-warp prefix-popcount packs ranks into a write offset
    //       computed by tid 0 (cross-warp prefix sum). tid 0 also walks
    //       the 128 alive sort positions once to compute slot stats
    //       (amax, p95, p80, mean, p25) broadcast via shared memory to
    //       all threads; these feed preferred_range to produce the six
    //       outer-scale candidates for the format search.
    //
    //   (b) Reset best/fallback state. `s_best_err = FLT_MAX`,
    //       `s_fallback_err = FLT_MAX` (so the first measurement wins).
    //
    //   (c) Search: scan candidates BPE-ascending (most aggressive
    //       first). For each (fmt, scale), `search_scales_for_fmt`
    //       counts how many live blocks pass and accumulates their
    //       pass mask. Once any (fmt, scale) hits the 32-block quota,
    //       `s_search_done` is set and the outer ci-loop breaks. If
    //       no candidate hits 32, the lowest-max-error fallback wins
    //       (with its partial pass mask).
    //
    //   (d) Claim phase: walk idx_compact, claim blocks whose pass-mask
    //       bit is set (no round-trip), up to 32 — this is the work
    //       `claim_passing_blocks_from_mask` does single-threaded by
    //       tid 0. Then a second-pass fill (warp 0) sweeps in
    //       remaining alive blocks in sort order until 32 are claimed.
    //       Both phases tombstone via `alive_clear`, so the next slot's
    //       compaction skips them.
    //
    // What it modifies (per side):
    //   - `*_alive_lo/hi`     — tombstones cleared as blocks are claimed
    //   - `slot_tags[s]`,
    //     `slot_scales[s]`    — overwritten per slot
    //   - `out_pal_map`,
    //     `out_eff_tags`      — written per (head, block)
    //   - `out_pal_tags`,
    //     `out_pal_scale`,
    //     `out_head_tag`      — written per head at the end
    //
    // What it reads (must persist across K → V invocations):
    //   - `smem_data`         — read-only (smem_k for K, smem_v for V)
    //   - `kthresh`           — read-only, K side only
    //   - `sink_weight`       — read-only via the hoisted `v_thr_sq`
    //
    // Tombstoning lives in `alive_lo/alive_hi` (one bit per block);
    // idx_sorted is read-only after the bitonic sort. Live count is
    // `__popcll(lo) + __popcll(hi)`; "find first alive in sort order"
    // uses a 32-lane chunked __ballot_sync scan instead of a serial
    // walk.
    auto process_side = [&](
        const __half* __restrict__ smem_data,  // flat [FUSED_HEAD_BLOCKS * FUSED_WARP_SIZE] f16 values
        float*    __restrict__ amax_sorted, // [FUSED_HEAD_BLOCKS] desc-sorted amax, read-only
        const uint16_t* __restrict__ idx_sorted,  // [FUSED_HEAD_BLOCKS] original indices (read-only after sort)
        uint64_t* alive_lo_ptr,             // → __shared__ alive bitmask (low half: blocks 0..63)
        uint64_t* alive_hi_ptr,             // → __shared__ alive bitmask (high half: blocks 64..127)
        const int* cands,
        int    num_cands,
        float  head_amax,
        bool   is_k,
        int*   __restrict__ out_pal_tags,
        float* __restrict__ out_pal_scale,
        int*   __restrict__ out_pal_map,
        int*   __restrict__ out_eff_tags,
        int*   __restrict__ out_head_tag
    ) {
        // Hoisted per-side constants used by every search/claim metric eval.
        // `inv_head_amax` replaces the prior `err / head_amax` divide; on
        // the V side the squared form is used against pre-squared `v_thr_sq`.
        const float inv_head_amax    = 1.0f / head_amax;
        const float inv_head_amax_sq = inv_head_amax * inv_head_amax;

        for (int s = 0; s < 4; s++) {
            // Snapshot alive masks so search can read consistent state without
            // re-loading from shared on every block. (alive_clear writes happen
            // in the claim phase, after search completes.)
            const uint64_t alive_lo = *alive_lo_ptr;
            const uint64_t alive_hi = *alive_hi_ptr;

            // ── Compact alive entries into idx_compact, all 4 warps in parallel ──
            // Each warp owns positions [warp_id*32, warp_id*32+32). Within a
            // warp: ballot the alive mask, prefix-popcount gives each lane a
            // packed rank inside the chunk. Cross-warp scan combines per-warp
            // counts into a global offset; the warp's first-alive position
            // is also recorded for the first-alive lookup used by tid 0's stat walk.
            const int  pos       = warp_id * FUSED_WARP_SIZE + lane;
            const int  b_at_pos  = idx_sorted[pos];
            const bool is_alive  = alive_get(alive_lo, alive_hi, b_at_pos);
            const unsigned bal   = __ballot_sync(0xffffffff, is_alive);
            const int  rank      = __popc(bal & ((1u << lane) - 1u));
            const int  warp_pop  = __popc(bal);
            if (lane == 0) {
                s_warp_pop[warp_id]    = warp_pop;
                s_first_alive[warp_id] = (bal != 0) ? (warp_id * FUSED_WARP_SIZE + (__ffs(bal) - 1)) : -1;
            }
            __syncthreads();

            // tid 0 computes per-warp prefix sum (4 entries) + total live count
            // + slot stats (amax, p95, p80, mean) by walking alive sort positions.
            // amax_sorted is descending so earlier alive positions have larger values.
            if (tid == 0) {
                int prefix = 0;
                int first_alive = -1;
                #pragma unroll
                for (int w = 0; w < FUSED_WARPS_PER_BLOCK; w++) {
                    if (first_alive < 0 && s_first_alive[w] >= 0) first_alive = s_first_alive[w];
                    const int p = s_warp_pop[w];
                    s_warp_pop[w] = prefix;   // becomes warp's write offset
                    prefix += p;
                }
                s_live_count = prefix;

                const float slot_amax_raw = (first_alive >= 0) ? amax_sorted[first_alive] : 0.0f;
                const int   lc    = prefix;
                float sum_amax    = 0.0f;
                float p95_val     = slot_amax_raw;
                float p80_val     = slot_amax_raw;
                float p25_val     = slot_amax_raw;
                int   cnt         = 0;
                const int p95_tgt = max(1, (lc + 19) / 20);  // ceil(5% of lc)
                const int p80_tgt = max(1, (lc + 4)  /  5);  // ceil(20% of lc)
                const int p25_tgt = max(1, (3 * lc)  /  4);  // floor(75% of lc)
                for (int pos = 0; pos < FUSED_HEAD_BLOCKS; pos++) {
                    if (alive_get(alive_lo, alive_hi, (int)idx_sorted[pos])) {
                        const float av = amax_sorted[pos];
                        sum_amax += av;
                        cnt++;
                        if (cnt == p95_tgt) p95_val = av;
                        if (cnt == p80_tgt) p80_val = av;
                        if (cnt == p25_tgt) p25_val = av;
                        if (cnt == lc) break;
                    }
                }
                s_slot_amax = fmaxf(slot_amax_raw, 1.0e-8f);
                s_slot_p95  = fmaxf(p95_val,       1.0e-8f);
                s_slot_p80  = fmaxf(p80_val,       1.0e-8f);
                s_slot_mean = (lc > 0) ? fmaxf(sum_amax / lc, 1.0e-8f) : 1.0e-8f;
                s_slot_p25  = fmaxf(p25_val,       1.0e-8f);
            }
            __syncthreads();

            const int   live_count = s_live_count;
            const int   warp_offset = s_warp_pop[warp_id];
            const float slot_amax  = s_slot_amax;
            const float safe_p95   = s_slot_p95;
            const float safe_p80   = s_slot_p80;
            const float slot_mean  = s_slot_mean;
            const float safe_p25   = s_slot_p25;

            // Each lane writes its own slot if it's alive.
            if (is_alive) {
                idx_compact[warp_offset + rank] = (uint16_t)b_at_pos;
            }
            __syncthreads();

            if (live_count == 0) {
                if (tid == 0) {
                    slot_tags  [s] = cands[num_cands - 1];
                    // Unclaimed slot: write identity scale (1.0). Encode multiplies and
                    // decode divides — both are no-ops at 1.0, so an unreferenced slot
                    // can't corrupt arena bytes or produce NaN/Inf at decode.
                    slot_scales[s] = 1.0f;
                }
                __syncthreads();
                continue;
            }

            // ── Reset best-of-slot search state ──────────────────────────
            // Initial best: the highest-BPE candidate at scale 1.0 with
            // err = FLT_MAX, so the first real measurement always wins.
            // Initial fallback: same seed; err = FLT_MAX guarantees the first
            // (fmt, scale) measurement wins. Fallback now tracks the candidate
            // with the lowest max error across all alive blocks (passing and
            // failing) — see `search_scales_for_fmt`.
            if (tid == 0) {
                s_best_fmt         = cands[num_cands - 1];
                s_best_scale       = 1.0f;
                s_best_err         = FLT_MAX;
                s_best_pass_lo     = 0;
                s_best_pass_hi     = 0;
                s_search_done      = 0;
                s_fallback_fmt     = cands[num_cands - 1];
                s_fallback_scale   = 1.0f;
                s_fallback_err     = FLT_MAX;
                s_fallback_pass_lo = 0;
                s_fallback_pass_hi = 0;
            }
            __syncthreads();

            // ── Search: BPE-ascending format × NUM_SCALE_CANDIDATES scales ──
            // Per-candidate dispatch: `with_select_fmt` resolves the runtime
            // fmt to a compile-time FmtTag, then the lambda picks the IS_K
            // specialisation. Inside `search_scales_for_fmt`, both quant and
            // dequant paths fully inline — no per-block fmt branching, no
            // __noinline__ slow-path call for cold formats.
            //
            // The 4 warps stride through live_count blocks in parallel; the
            // helper does a cross-warp reduction (per scale) and updates
            // shared best/fallback state. `s_search_done` is written by tid
            // 0 after a __syncthreads, so all threads see it consistently
            // at the top of the next ci iteration — break is convergent
            // across the block.
            for (int ci = 0; ci < num_cands; ci++) {
                if (s_search_done) break;
                const int fmt = cands[ci];
                with_select_fmt(fmt, [&](auto tag) {
                    constexpr int FMT = decltype(tag)::value;
                    if (is_k) {
                        search_scales_for_fmt<FMT, true>(
                            slot_amax, safe_p95, safe_p80, slot_mean, safe_p25,
                            smem_data, idx_compact, live_count,
                            tid, warp_id, lane,
                            warp_f32[warp_id], warp_quant[warp_id], kthresh,
                            inv_head_amax, inv_head_amax_sq, v_thr_sq,
                            warp_count, warp_pass_lo, warp_pass_hi, warp_amax_err,
                            &s_best_fmt, &s_best_scale, &s_best_err,
                            &s_best_pass_lo, &s_best_pass_hi, &s_search_done,
                            &s_fallback_fmt, &s_fallback_scale, &s_fallback_err,
                            &s_fallback_pass_lo, &s_fallback_pass_hi);
                    } else {
                        search_scales_for_fmt<FMT, false>(
                            slot_amax, safe_p95, safe_p80, slot_mean, safe_p25,
                            smem_data, idx_compact, live_count,
                            tid, warp_id, lane,
                            warp_f32[warp_id], warp_quant[warp_id], kthresh,
                            inv_head_amax, inv_head_amax_sq, v_thr_sq,
                            warp_count, warp_pass_lo, warp_pass_hi, warp_amax_err,
                            &s_best_fmt, &s_best_scale, &s_best_err,
                            &s_best_pass_lo, &s_best_pass_hi, &s_search_done,
                            &s_fallback_fmt, &s_fallback_scale, &s_fallback_err,
                            &s_fallback_pass_lo, &s_fallback_pass_hi);
                    }
                });
            }

            // No candidate fit all 32 blocks — fall back to the (fmt, scale)
            // with the lowest max error across all alive blocks (passing and
            // failing), with its best scale and its (partial) pass mask so
            // the claim phase still claims any blocks that did pass at the
            // fallback.
            if (tid == 0 && !s_search_done) {
                s_best_fmt     = s_fallback_fmt;
                s_best_scale   = s_fallback_scale;
                s_best_pass_lo = s_fallback_pass_lo;
                s_best_pass_hi = s_fallback_pass_hi;
            }
            __syncthreads();

            const int      best_fmt     = s_best_fmt;
            const float    best_scale   = s_best_scale;
            const uint64_t best_pass_lo = s_best_pass_lo;
            const uint64_t best_pass_hi = s_best_pass_hi;

            // `best_scale` is the exact `outer` value that won the search.
            // Encoder must multiply by it; decoder must divide by it. This holds
            // uniformly across quant and float formats — the float-format encode
            // path must apply the same scale or the round-trip won't match.
            if (tid == 0) {
                slot_tags  [s] = best_fmt;
                slot_scales[s] = best_scale;
            }
            __syncthreads();

            // Claim phase: walk idx_compact (alive entries in sort order),
            // claim blocks whose bit is set in the cached pass mask. No
            // round-trip; the search already computed the answer at the
            // winning (fmt, scale). Single-threaded by tid 0; broadcasts
            // the count via __syncthreads.
            int claimed = claim_passing_blocks_from_mask(
                s, head_id, best_fmt, tid, idx_compact, live_count,
                best_pass_lo, best_pass_hi,
                alive_lo_ptr, alive_hi_ptr,
                out_pal_map, out_eff_tags,
                &s_claim_count);

            // ── Second pass: fill remaining quota with non-passing blocks ──
            // After pass 1 (mask-driven claim), the slot still needs
            // `32 - claimed` blocks to hit its quota. Pass 2 walks
            // idx_compact in sort order (highest-amax first among the
            // remaining live) and claims them with the same `best_fmt`.
            //
            // Only warp 0 participates: claim ORDER matters because slot
            // s+1 inherits the next-largest amax blocks, so we must
            // walk idx_compact from the top sequentially. Parallelising
            // across warps would race the chunk pointer and reorder the
            // claim. Within warp 0, lanes ballot the alive mask of the
            // current 32-lane chunk; prefix-popcount assigns claim ranks
            // so up to `still_need` lanes commit in parallel within the
            // chunk while preserving sort order.
            if (warp_id == 0) {
                for (int chunk = 0; chunk < live_count && claimed < 32; chunk += FUSED_WARP_SIZE) {
                    const int  i        = chunk + lane;
                    const int  b        = (i < live_count) ? (int)idx_compact[i] : -1;
                    const bool live     = (b >= 0) && alive_get(*alive_lo_ptr, *alive_hi_ptr, b);
                    const unsigned bal2 = __ballot_sync(0xffffffff, live);
                    if (bal2 == 0) continue;

                    const int chunk_alive = __popc(bal2);
                    const int still_need  = 32 - claimed;
                    const int rank2       = __popc(bal2 & ((1u << lane) - 1u));
                    if (live && rank2 < still_need) {
                        out_pal_map [head_id * FUSED_HEAD_BLOCKS + b] = s;
                        out_eff_tags[head_id * FUSED_HEAD_BLOCKS + b] = best_fmt;
                        alive_clear(alive_lo_ptr, alive_hi_ptr, b);
                    }
                    claimed += min(chunk_alive, still_need);
                    __syncwarp();
                }
            }
            __syncthreads();
        }  // for s in 0..4

        // Write palette tags and scale indices + head_tag in parallel: lanes
        // 0..3 of warp 0 each handle one slot. format_table_index_cuda runs
        // once per lane; warp-max reduce picks the worst (highest
        // table-index) slot.
        if (warp_id == 0) {
            if (lane < 4) {
                out_pal_tags [head_id * 4 + lane] = slot_tags  [lane];
                out_pal_scale[head_id * 4 + lane] = slot_scales[lane];
            }

            const int my_tag = (lane < 4) ? slot_tags[lane] : 0;
            const int my_ti  = (lane < 4) ? format_table_index_cuda(my_tag) : -1;
            int worst_ti  = my_ti;
            int worst_tag = my_tag;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const int other_ti  = __shfl_xor_sync(0xffffffff, worst_ti,  off, 32);
                const int other_tag = __shfl_xor_sync(0xffffffff, worst_tag, off, 32);
                if (other_ti > worst_ti) { worst_ti = other_ti; worst_tag = other_tag; }
            }
            if (lane == 0) {
                out_head_tag[head_id] = worst_tag;
            }
        }
        __syncthreads();
    };  // end process_side

    process_side(
        smem_kv, amax_k, kidx,
        &k_alive_lo, &k_alive_hi,
        k_candidates, num_k_candidates,
        k_head_amax_val, true,
        k_palette_tags, k_palette_scale,
        k_palette_map, k_effective_block_tags, k_head_tags
    );

    // Reload V values into smem_kv.  K data is consumed; overwriting is safe.
    // Pointer and format are recovered from smem spill slots written in Phase 1.
    // Scoped so v_chunk_data_r / v_src_fmt_r / v_src_outer_r die at the brace,
    // keeping them out of the V process_side frame.
    {
        const char* v_chunk_data_r = (const char*)(uintptr_t)spill_v_chunk_ptr;
        const int   v_src_fmt_r    = spill_v_src_fmt;
        const float v_src_outer_r  = spill_v_src_outer;
        for (int blk = warp_id; blk < FUSED_HEAD_BLOCKS; blk += FUSED_WARPS_PER_BLOCK) {
            float v_val;
            if (ArenaFormat::is_quantized(v_src_fmt_r)) {
                const char* v_blk = v_chunk_data_r + (int64_t)blk * quant_block_bytes(v_src_fmt_r);
                v_val = dequant_element_inline<float>(v_blk, lane, v_src_fmt_r, v_src_outer_r);
            } else {
                v_val = load_as_float(v_chunk_data_r, blk * 32 + lane, arena_fmt_to_dtype_code(v_src_fmt_r));
            }
            smem_kv[blk * 32 + lane] = __float2half(v_val);
        }
    }  // v_chunk_data_r, v_src_fmt_r, v_src_outer_r freed here
    __syncthreads();

    // Lazy-load v_head_amax_val just before V process_side so it is not live
    // across the K process_side register frame (saved 1 register there).
    const float v_head_amax_val = fmaxf(__ldg(&v_head_amax_in[head_id]), 1.0e-8f);
    process_side(
        smem_kv, amax_v, vidx,
        &v_alive_lo, &v_alive_hi,
        v_candidates, num_v_candidates,
        v_head_amax_val, false,
        v_palette_tags, v_palette_scale,
        v_palette_map, v_effective_block_tags, v_head_tags
    );
}

// =============================================================================
// HOST-SIDE DISPATCHER  —  fused paged palette4
// =============================================================================
//
// Two-pass pipeline. Both kernels are launched with grid.x = total_heads
// (= chunks · n_kv_head). Pass 1 writes the per-(chunk, head) statistics
// pass 2 needs (head amax, p95, q-relevance quantiles). Pass 2 reads
// those statistics, the per-head K and V activations, and the candidate
// format set, and writes the full set of per-head selection outputs.
//
// Both passes run sequentially on the same default stream. There is no
// concurrency benefit from launching them on separate streams because
// pass 2 has a hard data dependency on every output of pass 1.
//
//   pass 1 (1 block × 128 threads per (chunk, head))
//     ├─ k_head_amax, v_head_amax
//     ├─ k_head_p95,  v_head_p95
//     └─ q_relevance_median, q_relevance_spread
//                 │
//                 ▼
//   pass 2 (1 block × 128 threads per (chunk, head))
//     ├─ palette_tags, palette_scale  (per slot)
//     ├─ palette_map, effective_block_tags  (per block)
//     ├─ head_tag  (per head)
//     └─ q_relevance_out  (per block, optional)
//
// Threshold conventions
// ---------------------
// V-side thresholds use the same lo/hi convention as K-side:
// `*_threshold_lo > *_threshold_hi` numerically. lo is the lenient
// threshold (applied at low q-relevance for K, at non-sink tokens for V);
// hi is the stricter threshold (applied at high relevance / peak sink).
//
// Sink detection on V is automatic and parameter-free (z-score + tanh
// on chunk-local statistics — see Phase 2.5 in the kernel). To disable
// sink protection on V, pass v_threshold_lo == v_threshold_hi. To
// disable q-relevance scaling on K, pass k_threshold_lo == k_threshold_hi.

extern "C" void run_select_kv_format_palette4_paged(
    const int64_t* per_head_table_raw,
    const int64_t* head_gids,
    float* q_relevance_median,
    float* q_relevance_spread,
    float* k_head_amax,
    float* v_head_amax,
    float* k_head_p95,
    float* v_head_p95,
    const int* k_candidates,
    const int* v_candidates,
    int num_k_candidates,
    int num_v_candidates,
    float k_threshold_hi,
    float k_threshold_lo,
    float v_threshold_hi,
    float v_threshold_lo,
    int total_heads,
    int blocks_per_head,
    int n_kv_head,
    int arena_chunks,
    int*   k_palette_tags,
    int*   v_palette_tags,
    float* k_palette_scale,
    float* v_palette_scale,
    int*   k_palette_map,
    int*   v_palette_map,
    int*   k_effective_block_tags,
    int*   v_effective_block_tags,
    int*   k_head_tags,
    int*   v_head_tags,
    float* q_relevance_out,
    cudaStream_t stream
) {
    if (total_heads == 0) return;

    // Maximise shared-memory carveout on Ada/Hopper (100 KB available).  The
    // main kernel uses ~28 KB smem after smem_kv aliasing, so MaxShared enables
    // 3 blocks/SM instead of the default 1 (48 KB carveout cap).
    cudaFuncSetAttribute(
        select_kv_format_palette4_paged,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);

    // Pass 1: compute per-(chunk,head) relevance quantiles AND head amaxes.
    approximate_q_relevance_quantiles<<<total_heads, QREL_QUANTILE_THREADS, 0, stream>>>(
        per_head_table_raw,
        head_gids,
        q_relevance_median,
        q_relevance_spread,
        k_head_amax,
        v_head_amax,
        k_head_p95,
        v_head_p95,
        blocks_per_head,
        total_heads,
        n_kv_head,
        arena_chunks
    );

    // Pass 2: fused selection + palette4 grouping, one block per (chunk, head).
    select_kv_format_palette4_paged<<<total_heads, FUSED_THREADS_PER_BLOCK, 0, stream>>>(
        per_head_table_raw,
        head_gids,
        q_relevance_median,
        q_relevance_spread,
        k_head_amax,
        v_head_amax,
        k_head_p95,
        v_head_p95,
        k_candidates,
        v_candidates,
        num_k_candidates,
        num_v_candidates,
        k_threshold_hi,
        k_threshold_lo,
        v_threshold_hi,
        v_threshold_lo,
        total_heads,
        blocks_per_head,
        n_kv_head,
        arena_chunks,
        k_palette_tags,
        v_palette_tags,
        k_palette_scale,
        v_palette_scale,
        k_palette_map,
        v_palette_map,
        k_effective_block_tags,
        v_effective_block_tags,
        k_head_tags,
        v_head_tags,
        q_relevance_out
    );
}

// =============================================================================
// PER-SIDE ERROR SAMPLING KERNEL
// =============================================================================
//
// Samples per-block reconstruction error for one side (K or V) across a
// candidate format set. Used by the older non-fused selection path and
// by the offline threshold-sweep tooling — it does NOT feed into the
// palette4 kernel above (the fused kernel does its own threshold
// evaluation inline).
//
// For each (chunk, dim, candidate, head), the kernel evaluates
// `quantize → dequantize` on a single sampled lane (`sample_token`) of
// each block and records:
//
//   error_out      : max|x − roundtrip(x)| for the sampled element
//   q_relevance_out: per-block Σ(q²k²)/Σ(k²)   (K side only, written
//                    once when quant_index == 0 to avoid clobbering
//                    across candidates)
//
// Grid mapping is intentionally:
//   grid.x = head_idx
//   grid.y = quant_index    (which candidate format to test)
//   grid.z = dim_idx        (block-in-head)
// Chunk is handled inside the block by assigning one warp per chunk and
// looping over the active paged batch.

extern "C" __global__ void sample_quant_errors_paged(
    const int64_t* __restrict__ per_head_table_raw,
    const int64_t* __restrict__ head_gids,
    const int* __restrict__ candidates,
    int num_candidates,
    float* __restrict__ error_out,
    float* __restrict__ q_relevance_out,
    int sample_token,
    int side_is_k,
    int num_chunks,
    int n_kv_head,
    int head_dim,
    int arena_chunks
) {
    __shared__ float   smem_f32  [SELECT_WARPS_PER_BLOCK][32];
    __shared__ uint8_t smem_quant[SELECT_WARPS_PER_BLOCK][MAX_QUANT_BLOCK_BYTES];

    const int head_idx = blockIdx.x;
    const int quant_index = blockIdx.y;
    const int dim_idx = blockIdx.z;
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    if (head_idx >= n_kv_head || quant_index >= num_candidates || dim_idx >= head_dim) return;
    if (sample_token < 0 || sample_token >= 32) return;

    const int fmt = candidates[quant_index];
    float*   warp_f32   = smem_f32  [warp_in_block];
    uint8_t* warp_quant = smem_quant[warp_in_block];

    for (int chunk_idx = warp_in_block; chunk_idx < num_chunks; chunk_idx += SELECT_WARPS_PER_BLOCK) {
        const int gid_base = chunk_idx * n_kv_head * 2;
        const int64_t gid = side_is_k
            ? __ldg(&head_gids[gid_base + head_idx * 2])
            : __ldg(&head_gids[gid_base + head_idx * 2 + 1]);

        const int arena_idx = (int)(gid / (int64_t)arena_chunks);
        const int chunk_idx_in_arena = (int)(gid - (int64_t)arena_idx * (int64_t)arena_chunks);

        PerHeadTableEntry ph = load_per_head_entry(per_head_table_raw, arena_idx, head_idx, n_kv_head);

        const int src_fmt = side_is_k ? per_head_get_k_format(ph) : per_head_get_v_format(ph);
        const char* head_base = side_is_k ? per_head_k_ptr(ph) : per_head_v_ptr(ph);
        const int64_t chunk_stride = side_is_k ? ph.k_chunk_byte_stride : ph.v_chunk_byte_stride;
        const char* chunk_data = head_base + (int64_t)chunk_idx_in_arena * chunk_stride;

        float x_val;
        float k_val = 0.0f;
        float q_val = 0.0f;
        if (ArenaFormat::is_quantized(src_fmt)) {
            const int blk_bytes = quant_block_bytes(src_fmt);
            const char* blk_ptr = chunk_data + (int64_t)dim_idx * blk_bytes;
            x_val = side_is_k
                ? dequant_element_inline<float, true >(blk_ptr, lane, src_fmt, 1.0f)
                : dequant_element_inline<float, false>(blk_ptr, lane, src_fmt, 1.0f);
            if (side_is_k) {
                k_val = x_val;
                q_val = dequant_q_element(blk_ptr, lane, src_fmt);
            }
        } else {
            const int elem_in_chunk = dim_idx * 32 + lane;
            x_val = load_as_float(chunk_data, elem_in_chunk, arena_fmt_to_dtype_code(src_fmt));
            if (side_is_k) {
                k_val = x_val;
            }
        }

        float x_rt;
        if (fmt == SELECT_FMT_BF16 || fmt == SELECT_FMT_F16) {
            x_rt = x_val;
        } else {
            warp_f32[lane] = x_val;
            __syncwarp();
            if (side_is_k) quantize_to_smem<true> (warp_f32, warp_quant, fmt);
            else           quantize_to_smem<false>(warp_f32, warp_quant, fmt);
            __syncwarp();
            x_rt = side_is_k
                ? dequant_element_inline<float, true >((const char*)warp_quant, lane, select_fmt_to_arena_fmt(fmt), 1.0f)
                : dequant_element_inline<float, false>((const char*)warp_quant, lane, select_fmt_to_arena_fmt(fmt), 1.0f);
        }
        const float sample_err = max_abs_error_warp(x_val, x_rt);

        float q_relevance = 1.0f;
        if (side_is_k) {
            const int has_q = __any_sync(0xffffffff, q_val != 0.0f);
            if (has_q) {
                q_relevance = block_relevance(k_val, q_val);
                q_relevance = __shfl_sync(0xffffffff, q_relevance, 0, 32);
            }
        }

        if (lane == sample_token) {
            const int out_idx = (((chunk_idx * head_dim) + dim_idx) * num_candidates + quant_index) * n_kv_head + head_idx;
            error_out[out_idx] = sample_err;
            if (quant_index == 0) {
                const int rel_idx = ((chunk_idx * head_dim) + dim_idx) * n_kv_head + head_idx;
                q_relevance_out[rel_idx] = q_relevance;
            }
        }
    }
}

extern "C" void run_sample_quant_errors_paged(
    const int64_t* per_head_table_raw,
    const int64_t* head_gids,
    const int* candidates,
    int num_candidates,
    float* error_out,
    float* q_relevance_out,
    int sample_token,
    int side_is_k,
    int num_chunks,
    int n_kv_head,
    int head_dim,
    int arena_chunks
) {
    if (num_chunks == 0 || n_kv_head == 0 || head_dim == 0 || num_candidates == 0) return;

    const int warps_per_block = SELECT_WARPS_PER_BLOCK;
    const int threads_per_block = warps_per_block * 32;
    dim3 grid((unsigned int)n_kv_head, (unsigned int)num_candidates, (unsigned int)head_dim);

    sample_quant_errors_paged<<<grid, threads_per_block>>>(
        per_head_table_raw,
        head_gids,
        candidates,
        num_candidates,
        error_out,
        q_relevance_out,
        sample_token,
        side_is_k,
        num_chunks,
        n_kv_head,
        head_dim,
        arena_chunks
    );
}

// =============================================================================
// FUSED K + V ERROR SAMPLING KERNEL
// =============================================================================
//
// Same role as `sample_quant_errors_paged`, but processes K and V
// together in a single launch. The fusion is mainly for table-lookup
// amortisation: each warp reads the per-head table entry once and uses
// it for both K and V samples, halving the global-memory traffic for
// that lookup vs. running the per-side kernel twice.
//
// (Q-relevance is computed from the K side but not used here — it's
// reused in the threshold-sweep tooling, where downstream code weights
// V errors by the K-derived q-relevance to make V compression stricter
// where Q attends to K.)
//
// Grid mapping: same as `sample_quant_errors_paged`:
//   grid.x = head
//   grid.y = quant_index     (shared K/V — must use the same candidate list)
//   grid.z = head_dim
// Chunk is handled inside the block by assigning one warp per chunk.

extern "C" __global__ void sample_quant_errors_kv_paged(
    const int64_t* __restrict__ per_head_table_raw,
    const int64_t* __restrict__ head_gids,
    const int* __restrict__ candidates,
    int num_candidates,
    float* __restrict__ k_error_out,
    float* __restrict__ v_error_out,
    int sample_token,
    int num_chunks,
    int n_kv_head,
    int head_dim,
    int arena_chunks
) {
    __shared__ float   smem_f32  [SELECT_WARPS_PER_BLOCK][32];
    __shared__ uint8_t smem_quant[SELECT_WARPS_PER_BLOCK][MAX_QUANT_BLOCK_BYTES];

    const int head_idx = blockIdx.x;
    const int quant_index = blockIdx.y;
    const int dim_idx = blockIdx.z;
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    if (head_idx >= n_kv_head || quant_index >= num_candidates || dim_idx >= head_dim) return;
    if (sample_token < 0 || sample_token >= 32) return;

    const int fmt = candidates[quant_index];
    float*   warp_f32   = smem_f32  [warp_in_block];
    uint8_t* warp_quant = smem_quant[warp_in_block];

    for (int chunk_idx = warp_in_block; chunk_idx < num_chunks; chunk_idx += SELECT_WARPS_PER_BLOCK) {
        const int gid_base = chunk_idx * n_kv_head * 2;
        const int64_t k_gid = __ldg(&head_gids[gid_base + head_idx * 2]);
        const int64_t v_gid = __ldg(&head_gids[gid_base + head_idx * 2 + 1]);

        // K-side per-head table lookup
        const int k_arena_idx = (int)(k_gid / (int64_t)arena_chunks);
        const int k_chunk_in_arena = (int)(k_gid - (int64_t)k_arena_idx * (int64_t)arena_chunks);
        PerHeadTableEntry k_ph = load_per_head_entry(per_head_table_raw, k_arena_idx, head_idx, n_kv_head);

        // V-side per-head table lookup (may be a different arena)
        const int v_arena_idx = (int)(v_gid / (int64_t)arena_chunks);
        const int v_chunk_in_arena = (int)(v_gid - (int64_t)v_arena_idx * (int64_t)arena_chunks);
        PerHeadTableEntry v_ph = load_per_head_entry(per_head_table_raw, v_arena_idx, head_idx, n_kv_head);

        // ── K pass ────────────────────────────────────────────────────
        const int k_src_fmt = per_head_get_k_format(k_ph);
        const char* k_chunk_data = per_head_k_ptr(k_ph) + (int64_t)k_chunk_in_arena * k_ph.k_chunk_byte_stride;

        float k_val = 0.0f;
        float q_val = 0.0f;
        if (ArenaFormat::is_quantized(k_src_fmt)) {
            const int blk_bytes = quant_block_bytes(k_src_fmt);
            const char* blk_ptr = k_chunk_data + (int64_t)dim_idx * blk_bytes;
            k_val = dequant_element_inline<float, true>(blk_ptr, lane, k_src_fmt, 1.0f);
            q_val = dequant_q_element(blk_ptr, lane, k_src_fmt);
        } else {
            const int elem_in_chunk = dim_idx * 32 + lane;
            k_val = load_as_float(k_chunk_data, elem_in_chunk, arena_fmt_to_dtype_code(k_src_fmt));
        }

        float k_rt;
        if (fmt == SELECT_FMT_BF16 || fmt == SELECT_FMT_F16) {
            k_rt = k_val;
        } else {
            warp_f32[lane] = k_val;
            __syncwarp();
            quantize_to_smem<true>(warp_f32, warp_quant, fmt);  // K side
            __syncwarp();
            k_rt = dequant_element_inline<float, true>((const char*)warp_quant, lane, select_fmt_to_arena_fmt(fmt), 1.0f);
        }
        const float k_err = max_abs_error_warp(k_val, k_rt);

        float q_relevance = 1.0f;
        const int has_q = __any_sync(0xffffffff, q_val != 0.0f);
        if (has_q) {
            q_relevance = block_relevance(k_val, q_val);
            q_relevance = __shfl_sync(0xffffffff, q_relevance, 0, 32);
        }

        // ── V pass ────────────────────────────────────────────────────
        const int v_src_fmt = per_head_get_v_format(v_ph);
        const char* v_chunk_data = per_head_v_ptr(v_ph) + (int64_t)v_chunk_in_arena * v_ph.v_chunk_byte_stride;

        float v_val = 0.0f;
        if (ArenaFormat::is_quantized(v_src_fmt)) {
            const int blk_bytes = quant_block_bytes(v_src_fmt);
            const char* blk_ptr = v_chunk_data + (int64_t)dim_idx * blk_bytes;
            v_val = dequant_element_inline<float>(blk_ptr, lane, v_src_fmt, 1.0f);
        } else {
            const int elem_in_chunk = dim_idx * 32 + lane;
            v_val = load_as_float(v_chunk_data, elem_in_chunk, arena_fmt_to_dtype_code(v_src_fmt));
        }

        float v_rt;
        if (fmt == SELECT_FMT_BF16 || fmt == SELECT_FMT_F16) {
            v_rt = v_val;
        } else {
            warp_f32[lane] = v_val;
            __syncwarp();
            quantize_to_smem<false>(warp_f32, warp_quant, fmt);  // V side
            __syncwarp();
            v_rt = dequant_element_inline<float>((const char*)warp_quant, lane, select_fmt_to_arena_fmt(fmt), 1.0f);
        }
        const float v_err = max_abs_error_warp(v_val, v_rt);

        // ── Write outputs ─────────────────────────────────────────────
        if (lane == sample_token) {
            const int out_idx = (((chunk_idx * head_dim) + dim_idx) * num_candidates + quant_index) * n_kv_head + head_idx;
            k_error_out[out_idx] = k_err;
            v_error_out[out_idx] = v_err;
        }
    }
}

extern "C" void run_sample_quant_errors_kv_paged(
    const int64_t* per_head_table_raw,
    const int64_t* head_gids,
    const int* candidates,
    int num_candidates,
    float* k_error_out,
    float* v_error_out,
    int sample_token,
    int num_chunks,
    int n_kv_head,
    int head_dim,
    int arena_chunks
) {
    if (num_chunks == 0 || n_kv_head == 0 || head_dim == 0 || num_candidates == 0) return;

    const int warps_per_block = SELECT_WARPS_PER_BLOCK;
    const int threads_per_block = warps_per_block * 32;
    dim3 grid((unsigned int)n_kv_head, (unsigned int)num_candidates, (unsigned int)head_dim);

    sample_quant_errors_kv_paged<<<grid, threads_per_block>>>(
        per_head_table_raw,
        head_gids,
        candidates,
        num_candidates,
        k_error_out,
        v_error_out,
        sample_token,
        num_chunks,
        n_kv_head,
        head_dim,
        arena_chunks
    );
}

// =============================================================================
// WINNER SELECTION KERNEL  (threshold-sweep utility)
// =============================================================================
//
// Consumes pre-computed K and V error surfaces (output of
// `sample_quant_errors_kv_paged`) and selects, for every (chunk, head,
// dim) cell and every threshold in `k_thresholds` / `v_thresholds`, the
// most aggressive candidate whose error is ≤ the threshold.
//
// Doing selection on the GPU instead of after a download replaces the
// large float error array with a small u8 winner array — often the
// difference between a multi-MB transfer and a few hundred KB:
//
//   error array  : [n_chunks × head_dim × n_quant × n_kv_head × 2 × 4 B]
//   winner array : [n_thresholds × n_cells × 2 × 1 B]  (n_quant × 4 smaller)
//
// Grid: (ceil(n_cells / SELECT_WINNER_THREADS), 1, 1)
//   n_cells = n_chunks × n_kv_head × head_dim
//   Each thread handles one (chunk, head, dim) cell.
//
// Layout — must match `batch_select_and_summarize` on the host:
//
//   k_winners[t * n_cells + cell_id]
//   cell_id = (chunk * n_kv_head + head) * head_dim + dim  =  bh * n_dim + d
//   error[((chunk * head_dim + dim) * n_quant + q) * n_kv_head + head]

#define SELECT_WINNER_THREADS 256

extern "C" __global__ void select_winners_kv_paged(
    const float* __restrict__ k_errors,       // [n_chunks × head_dim × n_quant × n_kv_head]
    const float* __restrict__ v_errors,       // same layout
    const float* __restrict__ k_thresholds,   // [n_k_thresholds]
    const float* __restrict__ v_thresholds,   // [n_v_thresholds]
    uint8_t* __restrict__ k_winners,          // [n_k_thresholds × n_cells]
    uint8_t* __restrict__ v_winners,          // [n_v_thresholds × n_cells]
    int n_k_thresholds,
    int n_v_thresholds,
    int n_cells,       // = n_chunks × n_kv_head × head_dim
    int n_quant,
    int n_kv_head,
    int head_dim
) {
    const int cell_id = blockIdx.x * SELECT_WINNER_THREADS + threadIdx.x;
    if (cell_id >= n_cells) return;

    // Decompose cell_id using winner layout:
    //   cell_id = (chunk * n_kv_head + head) * head_dim + dim
    const int dim   = cell_id % head_dim;
    const int bh    = cell_id / head_dim;   // = chunk * n_kv_head + head
    const int head  = bh % n_kv_head;
    const int chunk = bh / n_kv_head;

    // Base index into the error surface for this (chunk, dim, head), varying over q:
    //   error_base + q * error_stride  =  ((chunk * head_dim + dim) * n_quant + q) * n_kv_head + head
    const int error_base   = ((chunk * head_dim + dim) * n_quant) * n_kv_head + head;
    const int error_stride = n_kv_head;

    // Cache all candidate errors in registers (n_quant ≤ 32).
    float k_e[32];
    float v_e[32];
    #pragma unroll 1
    for (int q = 0; q < n_quant; q++) {
        k_e[q] = __ldg(&k_errors[error_base + q * error_stride]);
        v_e[q] = __ldg(&v_errors[error_base + q * error_stride]);
    }

    // K: find first candidate whose error ≤ threshold for each threshold level.
    for (int t = 0; t < n_k_thresholds; t++) {
        const float thr = __ldg(&k_thresholds[t]);
        int winner = n_quant - 1;
        #pragma unroll 1
        for (int q = 0; q < n_quant; q++) {
            if (k_e[q] <= thr) { winner = q; break; }
        }
        k_winners[t * n_cells + cell_id] = (uint8_t)winner;
    }

    // V: same.
    for (int t = 0; t < n_v_thresholds; t++) {
        const float thr = __ldg(&v_thresholds[t]);
        int winner = n_quant - 1;
        #pragma unroll 1
        for (int q = 0; q < n_quant; q++) {
            if (v_e[q] <= thr) { winner = q; break; }
        }
        v_winners[t * n_cells + cell_id] = (uint8_t)winner;
    }
}

extern "C" void run_select_winners_kv_paged(
    const float* k_errors,
    const float* v_errors,
    const float* k_thresholds,
    const float* v_thresholds,
    uint8_t* k_winners,
    uint8_t* v_winners,
    int n_k_thresholds,
    int n_v_thresholds,
    int n_cells,
    int n_quant,
    int n_kv_head,
    int head_dim
) {
    if (n_cells == 0) return;
    const int grid = (n_cells + SELECT_WINNER_THREADS - 1) / SELECT_WINNER_THREADS;
    select_winners_kv_paged<<<grid, SELECT_WINNER_THREADS>>>(
        k_errors, v_errors,
        k_thresholds, v_thresholds,
        k_winners, v_winners,
        n_k_thresholds, n_v_thresholds,
        n_cells, n_quant, n_kv_head, head_dim
    );
}

// =============================================================================
// WINNER SUMMARIZATION KERNEL  (threshold-sweep utility)
// =============================================================================
//
// Reduces the full [n_thresholds × n_cells] u8 winner array (output of
// `select_winners_kv_paged`) into [n_thresholds × 3] f32 bit-count
// accumulators, without a round-trip to host. Used by the offline
// threshold sweep to compare three compression strategies head-to-head:
//
//   ideal_bits = Σ_{bh,d} cand_bpe[winners[t,bh,d]] × chunk_size
//                — per-block selection (best possible without grouping)
//
//   head_bits  = Σ_{bh}   cand_bpe[max_d winners[t,bh,d]] × n_dim × chunk_size
//                — per-head selection (one format per head, matches the
//                  worst-case block — no palette grouping)
//
//   pal4_bits  = Σ_{bh}   ( palette4_pass(bh) + palette overhead )
//                — palette4 grouping (the production strategy)
//
// Per-cell winners are histogrammed (32-element register array, since
// n_quant ≤ 32) and the histogram filled into 4 slots from low to
// high to estimate per-block format assignments under palette4.
//
// Grid:  (ceil(n_bh / SUMMARIZE_WINNER_THREADS), n_thresholds)
// Block: (SUMMARIZE_WINNER_THREADS, 1)
//
//   out[t * 3 + 0] = ideal_bits[t]
//   out[t * 3 + 1] = head_bits[t]
//   out[t * 3 + 2] = pal4_bits[t]
//
// The three output accumulators must be zero-initialised by the caller
// before launch (atomicAdd targets).

#define SUMMARIZE_WINNER_THREADS 128

extern "C" __global__ void summarize_winners_side_paged(
    const uint8_t* __restrict__ winners,    // [n_thresholds × n_cells], cell=bh*n_dim+d
    const float*   __restrict__ cand_bpe,   // [n_quant]  bits-per-element per candidate
    float*                      out,        // [n_thresholds × 3]  (atomicAdd target)
    int n_thresholds,
    int n_cells,                            // = n_bh × n_dim
    int n_bh,
    int n_dim,
    int n_quant,
    int chunk_size,
    float pal_overhead                      // (n_dim * 2 + 4 * 8) bits
) {
    const int bh = blockIdx.x * SUMMARIZE_WINNER_THREADS + threadIdx.x;
    const int t  = blockIdx.y;
    if (bh >= n_bh || t >= n_thresholds) return;

    const uint8_t* slice = winners + (size_t)t * n_cells + bh * n_dim;

    // ── Phase 1: freq + head_worst + ideal_bits ───────────────────────────
    uint8_t freq[32];
    #pragma unroll 1
    for (int q = 0; q < n_quant; q++) freq[q] = 0;
    int head_worst = 0;
    float ideal_bits = 0.0f;

    #pragma unroll 1
    for (int d = 0; d < n_dim; d++) {
        int w = (int)slice[d];
        if (w >= n_quant) w = n_quant - 1;
        freq[w]++;
        if (w > head_worst) head_worst = w;
        ideal_bits += cand_bpe[w] * (float)chunk_size;
    }

    float head_bits = cand_bpe[head_worst] * (float)(n_dim * chunk_size);

    // ── Phase 2: fixed 4-slot palette fill from lowest required formats ──
    const int base_slot = n_dim / 4;
    const int extra_slot = n_dim % 4;
    float pal4_bits = pal_overhead;
    int seen = 0;
    int q = 0;
    int filled = 0;

    #pragma unroll 1
    for (int p = 0; p < 4; p++) {
        const int slot_size = base_slot + (p < extra_slot ? 1 : 0);
        if (slot_size <= 0) continue;
        filled += slot_size;
        while (q + 1 < n_quant && seen + (int)freq[q] < filled) {
            seen += (int)freq[q];
            q += 1;
        }
        pal4_bits += cand_bpe[q] * (float)(slot_size * chunk_size);
    }

    // ── Accumulate into per-threshold output ────────────────────────────
    float* t_out = out + t * 3;
    atomicAdd(&t_out[0], ideal_bits);
    atomicAdd(&t_out[1], head_bits);
    atomicAdd(&t_out[2], pal4_bits);
}

extern "C" void run_summarize_winners_side_paged(
    const uint8_t* winners,     // [n_thresholds × n_cells]
    const float*   cand_bpe,    // [n_quant]
    float*         out,         // [n_thresholds × 3] — must be zero-initialised by caller
    int n_thresholds,
    int n_cells,
    int n_bh,
    int n_dim,
    int n_quant,
    int chunk_size,
    float pal_overhead
) {
    if (n_bh == 0 || n_thresholds == 0) return;
    dim3 grid((n_bh + SUMMARIZE_WINNER_THREADS - 1) / SUMMARIZE_WINNER_THREADS,
              n_thresholds);
    summarize_winners_side_paged<<<grid, SUMMARIZE_WINNER_THREADS>>>(
        winners, cand_bpe, out,
        n_thresholds, n_cells, n_bh, n_dim, n_quant, chunk_size, pal_overhead
    );
}

// =============================================================================
// PER-CHUNK FORMAT REDUCTION  (legacy path)
// =============================================================================
//
// Reduces a per-block format-tag array into one tag per chunk. Used by
// the older non-palette path that quantizes whole chunks at a single
// format rather than grouping into 4 palette slots. The fused selection
// kernel above does its own (different) reduction into `head_tag`, so
// this kernel is not on the production path — kept for the legacy code
// that still emits per-block tags.
//
// Reduction policy: most conservative wins. The chunk's format is the
// highest-fidelity (largest candidate index) tag found among any of
// its blocks — downgrading any block below its required precision
// would violate the threshold for that block. F16 is treated as a
// sentinel "no quantization": if any block is F16, the whole chunk is.
//
// Format ordering is by candidate index: candidates[0] is most
// aggressive (lowest BPE), candidates[N−1] is highest fidelity
// (highest BPE).

__global__ void reduce_chunk_format(
    const int* __restrict__ block_tags,   // [num_chunks * blocks_per_chunk]
    int* __restrict__ chunk_tags,         // [num_chunks] output
    const int* __restrict__ candidates,   // [num_candidates] ordered low→high fidelity (ascending BPE)
    int num_candidates,
    int blocks_per_chunk,
    int num_chunks
) {
    const int chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk_idx >= num_chunks) return;

    const int base = chunk_idx * blocks_per_chunk;

    // Find the most conservative (highest candidate index) block in this chunk.
    // Start at -1 (no candidate seen yet).
    int max_cidx = -1;
    bool any_f16 = false;

    for (int i = 0; i < blocks_per_chunk; i++) {
        int t = block_tags[base + i];

        if (t == SELECT_FMT_F16) {
            // F16 sentinel — most conservative, entire chunk stays float
            any_f16 = true;
            break;
        }

        // Map tag to candidate index
        for (int c = 0; c < num_candidates; c++) {
            if (candidates[c] == t) {
                if (c > max_cidx) max_cidx = c;
                break;
            }
        }
    }

    if (any_f16) {
        chunk_tags[chunk_idx] = SELECT_FMT_F16;
    } else if (max_cidx >= 0) {
        chunk_tags[chunk_idx] = candidates[max_cidx];
    } else {
        // No blocks matched any candidate — fall back to F16
        chunk_tags[chunk_idx] = SELECT_FMT_F16;
    }
}

extern "C" void run_reduce_chunk_format(
    const int* k_block_tags,
    const int* v_block_tags,
    int* k_chunk_tags,
    int* v_chunk_tags,
    const int* k_candidates,
    const int* v_candidates,
    int num_k_candidates,
    int num_v_candidates,
    int blocks_per_chunk,
    int num_chunks
) {
    if (num_chunks == 0) return;
    const int threads = 256;
    const int grid = (num_chunks + threads - 1) / threads;
    reduce_chunk_format<<<grid, threads>>>(
        k_block_tags, k_chunk_tags, k_candidates, num_k_candidates,
        blocks_per_chunk, num_chunks);
    reduce_chunk_format<<<grid, threads>>>(
        v_block_tags, v_chunk_tags, v_candidates, num_v_candidates,
        blocks_per_chunk, num_chunks);
}

// =============================================================================
// PER-HEAD FORMAT REDUCTION  (legacy path)
// =============================================================================
//
// Same worst-case-wins policy as `reduce_chunk_format`, but at the head
// level. Output is one tag per (chunk, head) — one entry per CUDA block
// of the per-block selection kernel.
//
// blocks_per_head = head_dim · chunk_size / 32  (e.g. 128·32/32 = 128).
// Block tags are laid out as `[chunk][head][blocks_per_head]` in the
// input. Each thread handles one (chunk, head) pair and scans the
// `blocks_per_head`-sized stripe at that offset.

__global__ void reduce_head_format(
    const int* __restrict__ block_tags,   // [num_chunks * n_kv_head * blocks_per_head]
    int* __restrict__ head_tags,          // [num_chunks * n_kv_head] output
    const int* __restrict__ candidates,   // [num_candidates] ordered low→high fidelity (ascending BPE)
    int num_candidates,
    int blocks_per_head,
    int n_kv_head,
    int num_chunks
) {
    // One thread per (chunk, head) pair
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_chunks * n_kv_head;
    if (idx >= total) return;

    const int chunk_idx = idx / n_kv_head;
    const int head_idx = idx % n_kv_head;

    // Block tags layout: [chunk_idx][head_idx][blocks_per_head]
    const int blocks_per_chunk = n_kv_head * blocks_per_head;
    const int base = chunk_idx * blocks_per_chunk + head_idx * blocks_per_head;

    // Find the most conservative (highest candidate index) block for this head.
    int max_cidx = -1;
    bool any_f16 = false;

    for (int i = 0; i < blocks_per_head; i++) {
        int t = block_tags[base + i];

        if (t == SELECT_FMT_F16) {
            any_f16 = true;
            break;
        }

        for (int c = 0; c < num_candidates; c++) {
            if (candidates[c] == t) {
                if (c > max_cidx) max_cidx = c;
                break;
            }
        }
    }

    if (any_f16) {
        head_tags[idx] = SELECT_FMT_F16;
    } else if (max_cidx >= 0) {
        head_tags[idx] = candidates[max_cidx];
    } else {
        head_tags[idx] = SELECT_FMT_F16;
    }
}

extern "C" void run_reduce_head_format(
    const int* k_block_tags,
    const int* v_block_tags,
    int* k_head_tags,
    int* v_head_tags,
    const int* k_candidates,
    const int* v_candidates,
    int num_k_candidates,
    int num_v_candidates,
    int blocks_per_head,
    int n_kv_head,
    int num_chunks
) {
    if (num_chunks == 0) return;
    const int total = num_chunks * n_kv_head;
    const int threads = 256;
    const int grid = (total + threads - 1) / threads;
    reduce_head_format<<<grid, threads>>>(
        k_block_tags, k_head_tags, k_candidates, num_k_candidates,
        blocks_per_head, n_kv_head, num_chunks);
    reduce_head_format<<<grid, threads>>>(
        v_block_tags, v_head_tags, v_candidates, num_v_candidates,
        blocks_per_head, n_kv_head, num_chunks);
}

// =============================================================================
// PER-HEAD STATS-FORMAT REDUCTION
// =============================================================================
//
// Variant of `reduce_head_format` that uses `format_table_index_cuda`
// (rather than the candidate-index lookup) to rank tags. The table
// index orders all formats globally by BPE — including F16/BF16 — so
// this reduction works without needing a candidate set. Used by stats
// and observability code that needs to summarise heads regardless of
// which candidate set the selection ran with.
//
// Also writes `effective_block_tags`: every block in the head gets the
// head's worst-case tag, so downstream stats code sees the effective
// per-block format under a head-uniform encoding policy. This is the
// metric the threshold-sweep tooling uses to compare grouping
// strategies.

__global__ void reduce_head_stats_format(
    const int* __restrict__ block_tags,
    int* __restrict__ head_tags,
    int* __restrict__ effective_block_tags,
    int blocks_per_head,
    int n_kv_head,
    int num_chunks
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_chunks * n_kv_head;
    if (idx >= total) return;

    const int chunk_idx = idx / n_kv_head;
    const int head_idx = idx % n_kv_head;
    const int blocks_per_chunk = n_kv_head * blocks_per_head;
    const int base = chunk_idx * blocks_per_chunk + head_idx * blocks_per_head;

    int worst_tag = SELECT_FMT_F16;
    int worst_ti = -1;
    for (int i = 0; i < blocks_per_head; ++i) {
        const int tag = block_tags[base + i];
        const int ti = format_table_index_cuda(tag);
        if (ti > worst_ti) {
            worst_ti = ti;
            worst_tag = tag;
        }
    }
    head_tags[idx] = worst_tag;
    for (int i = 0; i < blocks_per_head; ++i) {
        effective_block_tags[base + i] = worst_tag;
    }
}

extern "C" void run_reduce_head_stats_format(
    const int* k_block_tags,
    const int* v_block_tags,
    int* k_head_tags,
    int* v_head_tags,
    int* k_effective_block_tags,
    int* v_effective_block_tags,
    int blocks_per_head,
    int n_kv_head,
    int num_chunks
) {
    if (num_chunks == 0) return;
    const int total = num_chunks * n_kv_head;
    const int threads = 256;
    const int grid = (total + threads - 1) / threads;
    reduce_head_stats_format<<<grid, threads>>>(
        k_block_tags, k_head_tags, k_effective_block_tags,
        blocks_per_head, n_kv_head, num_chunks);
    reduce_head_stats_format<<<grid, threads>>>(
        v_block_tags, v_head_tags, v_effective_block_tags,
        blocks_per_head, n_kv_head, num_chunks);
}

