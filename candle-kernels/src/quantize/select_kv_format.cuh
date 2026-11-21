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
// Within each (chunk, head), the blocks_per_head blocks (= head_dim; 128 or
// 256, template parameter HB) are partitioned into 4 palette slots of HB/4
// blocks each. Blocks in the same slot share a (format, outer scale) pair.
// Per-head metadata is therefore just 4 (fmt, scale) pairs plus a
// 2-bit-per-block palette index — small enough to keep in cache, and
// uniform enough that the attention kernel's hot path can dispatch a single
// dequant routine per slot rather than per block.
//
// Outputs (indexed by `head_id = chunk_id * n_kv_head + head_idx`):
//
//   palette_tags          [4]   winning format per slot
//   palette_scale         [4]   winning outer scale per slot
//   palette_map           [HB]  which slot each block belongs to (0..3)
//   effective_block_tags  [HB]  per-block format (= palette_tags[slot])
//   head_tag                    most conservative slot's format
//   q_relevance_out       [HB]  optional, per-block Σ(q²k²)/Σ(k²)
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
//   2. `select_kv_format_palette4_paged<HB>` does the actual selection. For
//      each (chunk, head): load all HB blocks; detect attention-sink
//      tokens; sort by amax desc; iterate 4 slots, each picking the most
//      aggressive (lowest-BPE) format whose normalised reconstruction
//      error stays within threshold for ≥ HB/4 of the still-unclaimed
//      blocks; assign those HB/4 to the slot. Slot 0 sees the largest-amax
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
//   2      │ Bitonic sort HB entries by amax desc.    │
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
//          │       find (fmt, scale) with ≥ HB/4 pass │
//          │       track fallback (lowest max-err)    │
//          │   (c) Claim HB/4 (passing first, fill)   │
//          └──────────────────────────────────────────┘
//                                │
//                                ▼
//                      Per-(chunk, head) outputs
//
// Block invariant
// ---------------
// "Block" here means a 32-element strip of one (head_dim row) at a fixed
// (chunk, head, block-in-head) position. blocks_per_head equals head_dim
// (= chunk_size 32 × head_dim / WARP_SIZE 32) and is the fused kernel's
// template parameter HB ∈ {128, 256}. One CUDA block ↔ one (chunk, head);
// per-block round-trips are warp-cooperative — all 32 lanes of the active
// warp hold the block's 32 elements simultaneously.
//
// Numerical contract
// ------------------
// Per-block error metrics are side-asymmetric:
//
//   K side: pass_metric = mean_top4(|orig − recon|) · (1 / head_scale)
//           threshold   = kthresh[b]    (q-relevance scaled, per block)
//
//   V side: pass_metric = mean_{32 lanes}(orig − recon)² · (1 / head_scale²)
//           threshold   = v_thr_sq      (sink-aware, constant per head)
//
// `head_scale` is the head's amax capped at HEAD_SCALE_P95_CAP times its 95th
// percentile |x| (`head_scale()`), so a few sink elements cannot stretch it.
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
// Every metric the search measures is normalised (by head_scale for K,
// head_scale² for V) so the threshold values are dimensionless and
// transferable across heads, layers, and models. Combined with the
// candidate set's BPE ladder, this gives the per-block error budget
// the O(1) bound required by the unbounded-context attention design.
//
// Include via `quantized_dispatcher.cu`, which provides the `block_q*`
// struct definitions referenced by the per-format quantize/dequant paths.

#pragma once

#include <cfloat>
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

// Register round trips for the search; reads the SELECT_FMT_* tags above.
#include "select_lane_roundtrip.cuh"

#include "select_kv_format/fused_format_helpers.cuh"

#include "select_kv_format/search_helpers.cuh"

#include "select_kv_format/input_loads_and_relevance.cuh"

#include "select_kv_format/quantile_and_sort.cuh"

#include "select_kv_format/fused_palette4_kernel.cuh"

#include "select_kv_format/host_dispatch_and_sampling.cuh"

#include "select_kv_format/winners_and_reductions.cuh"

