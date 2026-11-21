// Alive-bitmask helpers, the roundtrip and pass metric for one block, head
// scale, and the per-format scale search.
//
// The per-slot alive bitmask that tombstones claimed blocks, the round-trip of
// a block through a candidate format, and the (pass metric, threshold) pair
// that decides whether a candidate passes. `head_scale` caps the head's amax so
// a few sink elements cannot stretch the normalisation, and
// `search_scales_for_fmt` searches the scale for each format.

// =============================================================================
// ALIVE-BITMASK HELPERS  —  per-slot tombstoning
// =============================================================================
// Each slot's claim phase tombstones the HB/4 blocks it just selected so the
// next slot's compaction skips them. We use an HB-bit alive mask carried as
// AW = HB/64 u64 words (word w covers blocks 64·w .. 64·w+63) — bit b of the
// mask set means block b is unclaimed.
//
// Why a bitmask instead of `idx_sorted[i] = -1` tombstoning:
//
//   - O(1) live count via a word-wise `__popcll` sum,
//     vs. O(N) scan of an int array.
//   - "Find first alive in sort order" becomes a 32-lane chunked
//     `__ballot_sync` scan, vs. a lane-0 serial walk.
//   - The kidx/vidx arrays stay read-only after the bitonic sort, so
//     they fit comfortably in uint16_t (block IDs are 0..HB-1). Halves
//     the smem footprint over int.
//
// `alive_clear` uses atomicAnd because the second-pass fill in
// `process_side` can target distinct bits of the same u64 word from
// multiple lanes in the same warp; a plain RMW would race and drop
// updates.
//
// `alive_get` selects the word with an unrolled predicated loop rather than
// `mask[b >> 6]` so that register-resident snapshot arrays (the per-slot
// alive snapshot in process_side) never get demoted to local memory by a
// dynamic index; for shared-memory masks the two forms are equivalent.

template <int AW>
__device__ __forceinline__ bool alive_get(const uint64_t* mask, int b) {
    const int w = b >> 6;
    uint64_t word = 0;
    #pragma unroll
    for (int i = 0; i < AW; i++) word = (i == w) ? mask[i] : word;
    return ((word >> (b & 63)) & 1ULL) != 0ULL;
}
__device__ __forceinline__ void alive_clear(uint64_t* mask, int b) {
    atomicAnd((unsigned long long*)&mask[b >> 6], ~(1ULL << (b & 63)));
}
template <int AW>
__device__ __forceinline__ int alive_count(const uint64_t* mask) {
    int n = 0;
    #pragma unroll
    for (int w = 0; w < AW; w++) n += __popcll(mask[w]);
    return n;
}

// =============================================================================
// SEARCH / CLAIM HELPERS  (compile-time fmt dispatch)
// =============================================================================
// These are the building blocks the inner search and claim loops are
// built from. Each is templated on FMT, so a single per-candidate switch in
// the caller dispatches to a fully-inlined body — no per-block runtime fmt
// branching, no __noinline__ slow-path call for the cold quant formats.
//
// The K/V side is a *runtime* argument, not a second template parameter. Only
// Q0_V's calibrated table set reads it, and templating it doubled every one of
// these — twenty-one codecs, twice — so that one arm could pick a table.
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

// One-block warp-cooperative round-trip: scale up by `outer`, quantize,
// dequantize element `lane`. Returns the per-lane reconstructed float.
//
// A format with a register round trip (`lane_roundtrip`) never touches the
// scratch buffers. Every other format quantizes into the warp's scratch buffer
// and decodes from it: the two `__syncwarp()` calls ensure all 32 lanes have
// written their scaled values before the (collectively-called)
// `quantize_block_for_fmt` reads them, and that the quantized bytes are visible
// before the per-lane `dequant_element_for_fmt` load. Since the scratch buffers
// are warp-private smem, no broader barrier is needed.
template <int FMT>
__device__ __forceinline__ float roundtrip_block_for_fmt(
    float orig,
    float outer,
    int lane,
    float* warp_f32_smem,
    uint8_t* warp_quant_smem,
    bool is_k
) {
    if constexpr (lane_roundtrip<FMT>::value) {
        return lane_roundtrip_for_fmt<FMT>(orig * outer, outer, warp_f32_smem, is_k);
    }
    warp_f32_smem[lane] = orig * outer;
    __syncwarp();
    quantize_block_for_fmt<FMT>(warp_f32_smem, warp_quant_smem, is_k);
    __syncwarp();
    return dequant_element_for_fmt<FMT>(warp_quant_smem, lane, outer, is_k);
}

// Compute the (pass_metric, threshold) pair for a single (orig, recon)
// block. The two sides use structurally different metrics:
//
//   K side  pass_metric    = mean_top4(|orig − recon|) · (1 / head_scale)
//           thr_to_compare = kthresh[b]   (q-relevance scaled, per block)
//
//   V side  pass_metric    = mean_{32 lanes}(orig − recon)² · (1 / head_scale²)
//           thr_to_compare = v_thr_sq     (sink-aware, constant per head)
//
// Both pass_metric and thr_to_compare are warp-uniform after the
// reductions inside this function: K's `mean_top4_abs_error_warp` takes
// each maximum as a warp-wide reduction every lane receives, and V's MSE
// reduction is a sum over the 32 lanes followed by a constant divide. That uniformity is what
// makes the lane-0-only mask accumulation in `search_scales_for_fmt`
// safe — every lane sees the same predicate value.
//
// `is_k` is a runtime argument, not a template parameter. It is uniform across
// the block — the side loop sets it — so the branch is warp-uniform and the two
// collective reductions below stay safe; what it is not worth is a second copy
// of everything that calls this, which is what templating it cost.
//
// The constants (`inv_head_amax` and `inv_head_amax_sq`, the reciprocals of
// the head scale, and `v_thr_sq`) are computed once per side in
// `process_side`. `v_thr_sq` depends only on the sink weights, which are fixed
// after Phase 2.5, so deriving it here would repeat a warp-max reduction over
// `sink_weight[lane]` on every (block, fmt, scale) round trip for nothing.
__device__ __forceinline__ void compute_pass_metric(
    float orig,
    float recon,
    int b,
    float inv_head_amax,
    float inv_head_amax_sq,
    float v_thr_sq,
    const __half* kthresh,
    float& pass_metric,
    float& thr_to_compare,
    bool is_k
) {
    if (is_k) {
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

// The scale the error metrics divide by: the head's amax, capped at
// `HEAD_SCALE_P95_CAP` times its 95th-percentile |x| (`approximate_q_relevance_
// quantiles`). A sink token puts a few elements far above the rest of the head;
// they set the amax, and an amax-relative threshold then admits errors that are
// large against everything else in the chunk. The p95 is not moved by a handful
// of elements, so it bounds how far they can stretch the scale. A head with no
// such elements has amax within the cap and keeps its amax exactly.
__device__ __forceinline__ float head_scale(float amax, float p95) {
    return fmaxf(fminf(amax, HEAD_SCALE_P95_CAP * p95), 1.0e-8f);
}

// Is a block with this variance (across its valid tokens) flat — its standard
// deviation below `step`, one INT8 step of the head scale? The variance is
// E[x^2] - mean^2 from warp-reduced moments; its rounding error is far below the
// step^2 it is compared with, which is (head scale / 127)^2.
__device__ __forceinline__ bool block_var_is_flat(float var, float step) {
    return var <= step * step;
}

// Per-fmt scale × block search.
//
// For one format, walks NUM_SCALE_CANDIDATES outer scales; per scale,
// counts how many live blocks pass the threshold from
// `compute_pass_metric`. Updates the caller's best_* / fallback_*
// shared state, and sets *s_search_done = 1 if any scale reaches the
// HB/N_PALETTE-block slot quota. The caller's ci-loop reads s_search_done
// at the top of each iteration to decide whether to stop climbing the BPE
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
// Per warp and scale: a pass-mask row in shared memory covering the blocks
// the warp processed, and the running max error in a register.
//
// Cross-warp: tid 0 OR-merges the pass masks (disjoint by construction —
// different warps process different block IDs) and counts the passing blocks
// as the merged mask's population. It then takes max over warp_amax_err to
// produce the (fmt, scale) summary, and updates the shared best/fallback fields.
//
// Best vs. fallback
// -----------------
// Best is tracked only over (fmt, scale) combos with total ≥ HB/N_PALETTE
// (slot quota satisfied). Among winners, the lowest `aerr` (max error among
// passing blocks) wins. Fallback is tracked over ALL (fmt, scale) combos
// seen — it remembers the combo with the lowest max error across all
// alive blocks (passing AND failing). If no candidate hits the quota across
// the entire search, the caller falls back to that combo so the forced-claim
// path uses the format whose worst-case block error is smallest.
//
// Pass-mask cache
// ---------------
// When a (fmt, scale) becomes the new best (count ≥ the slot quota and lowest aerr)
// or the new fallback (lowest max-err across all blocks), its merged pass
// mask is saved alongside fmt/scale/err. The claim phase reads the
// saved mask and skips re-running the round-trip — search already
// determined which blocks pass at the winning combo, so recomputing
// would just produce the same answer:
//
//     bit b in s_best_pass  ←  block b passed at (s_best_fmt, s_best_scale)
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

// `is_k` is a runtime argument rather than a template parameter: only Q0_V's
// table choice ever reads it, and templating it emitted a second copy of this
// function — and of every codec it inlines — for all twenty-one formats. See
// `quantize_block_for_fmt`.
template <int FMT, int HB>
__device__ __noinline__ void search_scales_for_fmt(
    bool is_k,
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
    const uint64_t* flat_mask,       // [HB/64] bit b set = block b is flat (this side)
    int*      warp_nonflat,          // [warps] flat-only formats: a non-flat block was skipped
    // Cross-warp aggregation scratch (one row per (warp, scale))
    uint64_t* warp_pass,             // [warps × NUM_SCALE_CANDIDATES × HB/64] pass-mask words
    float*    warp_amax_err,
    // Cross-thread best/fallback state, written by tid 0 only
    int*      s_best_fmt,
    float*    s_best_scale,
    float*    s_best_err,
    uint64_t* s_best_pass,           // [HB/64]
    int*      s_search_done,
    int*      s_fallback_fmt,
    float*    s_fallback_scale,
    float*    s_fallback_err,
    uint64_t* s_fallback_pass,       // [HB/64]
    // Early rejection (see below). Zero on entry; tid 0 re-zeroes them on exit.
    bool      drop_hopeless,         // stop measuring a scale once it cannot fill the slot
    int*      s_fail,                // [NUM_SCALE_CANDIDATES + 1] failures per scale; last = every scale
    int*      s_dropped,             // bit si: some warp stopped measuring scale si
    int*      s_fallback_incomplete  // set: a fallback candidate was dropped, so the fallback is unknown
) {
    // Alive/pass masks are HB bits = AW u64 words; the slot quota is one
    // palette band's worth of blocks.
    constexpr int AW         = HB / 64;
    constexpr int SLOT_QUOTA = HB / N_PALETTE;
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
    // Block-outer, scale-inner: each block is loaded once and measured at every
    // scale, and the scales' round trips are independent of each other, so the
    // scheduler overlaps their shuffle chains instead of waiting out each one.
    //
    // Each warp owns the rows [warp_id * NUM_SCALE_CANDIDATES + si] of the
    // cross-warp scratch. Lane 0 ORs a passing block's bit straight into its
    // warp's pass row (warp-private, so no atomic and no barrier), and the pass
    // count is the population of the merged mask, so nothing per block lives
    // outside registers but the six running maxima. A single __syncthreads after
    // the loop makes every row visible; tid 0 reduces all si slices in one pass.
    // The row stride is NUM_SCALE_CANDIDATES (the smem allocation) regardless of
    // kNumScales — unused rows are neither written nor read.
    //
    // Early rejection. A scale with more than `live_count - SLOT_QUOTA` failing
    // blocks can no longer reach the quota, so it can never be the best. With
    // `drop_hopeless`, lane 0 counts each failure into `s_fail`, and a warp stops
    // as soon as the shared counts say every scale is hopeless. Until then it
    // measures every scale: skipping one inside the unrolled scale loop puts a
    // branch between the scales' shuffle chains, which costs more overlap than
    // the round trip saves. The counts only grow, so a count read while other
    // warps are still adding is never past the limit when the final count is
    // not: dropping is never wrong for the best. The fallback is another
    // matter — it ranks every scale by its error over ALL blocks, which a
    // dropped scale no longer has — so a dropped scale is not ranked, and
    // `s_fallback_incomplete` tells the caller to search again without dropping
    // in the one case the fallback is used: no candidate filled the slot.
    const int fail_budget = live_count - SLOT_QUOTA;
    constexpr int ALL_SCALES = (1 << kNumScales) - 1;
    int live_scales = ALL_SCALES;
    volatile int* fail_count = s_fail;
    float outer[kNumScales];
    float my_amax_err[kNumScales];
    #pragma unroll
    for (int si = 0; si < kNumScales; si++) {
        outer[si]       = preferred_range(si, slot_amax, safe_p95, safe_p80, slot_mean, safe_p25);
        my_amax_err[si] = 0.0f;
    }
    uint64_t* my_pass_rows = warp_pass + (size_t)warp_id * NUM_SCALE_CANDIDATES * AW;
    if (lane < kNumScales * AW) my_pass_rows[lane] = 0ULL;
    __syncwarp();

    int my_nonflat = 0;   // flat-only formats: a non-flat block was skipped
    // idx_compact contains only alive entries — no per-block alive_get probe.
    // Each warp processes live_count / FUSED_WARPS_PER_BLOCK blocks on average.
    for (int i = warp_id; i < live_count; i += FUSED_WARPS_PER_BLOCK) {
        const int b = idx_compact[i];

        if (drop_hopeless) {
            // Lane 0's read decides for the warp, so the drop is warp-uniform.
            int dead = 0;
            if (lane == 0) {
                const int all = fail_count[NUM_SCALE_CANDIDATES];
                #pragma unroll
                for (int si = 0; si < kNumScales; si++)
                    if (fail_count[si] + all > fail_budget) dead |= 1 << si;
            }
            live_scales &= ~__shfl_sync(0xffffffff, dead, 0, 32);
            if (live_scales == 0) break;
        }

        // A flat-only format (Q0) is offered only the blocks flagged flat in
        // Phase 1. For any other block it cannot pass, so the round trip is
        // skipped outright and the slot is marked as holding a block the format
        // cannot take (it then cannot be the fallback). `b` is warp-uniform, so
        // the skip is too. Compiled out for every other format.
        if constexpr (flat_only<FMT>::value) {
            if (((flat_mask[b >> 6] >> (b & 63)) & 1ULL) == 0ULL) {
                my_nonflat = 1;
                if (drop_hopeless && lane == 0) atomicAdd(&s_fail[NUM_SCALE_CANDIDATES], 1);
                continue;
            }
        }

        const float orig = __half2float(smem_data[b * FUSED_WARP_SIZE + lane]);
        #pragma unroll
        for (int si = 0; si < kNumScales; si++) {
            const float recon = roundtrip_block_for_fmt<FMT>(
                orig, outer[si], lane, warp_f32_warp, warp_quant_warp, is_k);

            float pass_metric, thr_to_compare;
            compute_pass_metric(
                orig, recon, b,
                inv_head_amax, inv_head_amax_sq, v_thr_sq,
                kthresh,
                pass_metric, thr_to_compare, is_k);

            // pass_metric and thr_to_compare are warp-uniform after the metric
            // reductions, so all 32 lanes evaluate the same predicate. The max
            // runs over ALL blocks (passing and failing) so the fallback can
            // pick the candidate with the lowest worst-case error rather than
            // the highest BPE.
            my_amax_err[si] = fmaxf(my_amax_err[si], pass_metric);
            if (lane == 0) {
                if (pass_metric <= thr_to_compare)
                    my_pass_rows[si * AW + (b >> 6)] |= (1ULL << (b & 63));
                else if (drop_hopeless)
                    atomicAdd(&s_fail[si], 1);
            }
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int si = 0; si < kNumScales; si++)
            warp_amax_err[warp_id * NUM_SCALE_CANDIDATES + si] = my_amax_err[si];
        if constexpr (flat_only<FMT>::value) warp_nonflat[warp_id] = my_nonflat;
        if (live_scales != ALL_SCALES) atomicOr(s_dropped, ALL_SCALES & ~live_scales);
    }

    // One sync makes all (warp, si) rows visible to tid 0.
    __syncthreads();

    // Cross-warp reduction by tid 0 across all si slices in one sequential pass.
    // Masks OR (warps process disjoint blocks), and the pass count is the merged
    // mask's population; amax_err takes the max.
    if (tid == 0) {
        // A flat-only format that skipped any block cannot take the whole slot,
        // so it is not a fallback candidate: its error covers only the blocks it
        // was offered. It can still win outright on its passing blocks.
        bool skipped_nonflat = false;
        if constexpr (flat_only<FMT>::value) {
            #pragma unroll
            for (int w = 0; w < FUSED_WARPS_PER_BLOCK; w++)
                skipped_nonflat = skipped_nonflat || (warp_nonflat[w] != 0);
        }
        // tid 0 alone, once per format: serial bookkeeping, so the scales are
        // a real loop rather than one inlined copy of the merge each.
        #pragma unroll 1
        for (int si = 0; si < kNumScales; si++) {
            const float outer = preferred_range(si, slot_amax, safe_p95, safe_p80, slot_mean, safe_p25);
            uint64_t mask[AW];
            #pragma unroll
            for (int w = 0; w < AW; w++) mask[w] = 0;
            float    aerr  = 0.0f;
            #pragma unroll
            for (int w = 0; w < FUSED_WARPS_PER_BLOCK; w++) {
                const int base = w * NUM_SCALE_CANDIDATES + si;
                #pragma unroll
                for (int word = 0; word < AW; word++)
                    mask[word] |= warp_pass[base * AW + word];
                aerr   = fmaxf(aerr, warp_amax_err[base]);
            }
            int total = 0;
            #pragma unroll
            for (int w = 0; w < AW; w++) total += __popcll(mask[w]);

            // Fallback: candidate with the lowest max error across all blocks.
            // amax_err here is taken over every alive block (passing and
            // failing) — see the inner loop. Picking the lowest steers the
            // forced-claim path toward the format that fits the worst block
            // best, rather than the highest-BPE / most-conservative option.
            // A dropped scale's error covers only the blocks measured before
            // the drop, so it is not ranked, and the fallback is marked unknown.
            const bool dropped = ((*s_dropped >> si) & 1) != 0;
            if (!skipped_nonflat && dropped) *s_fallback_incomplete = 1;
            if (!skipped_nonflat && !dropped && aerr < *s_fallback_err) {
                *s_fallback_fmt     = FMT;
                *s_fallback_scale   = outer;
                *s_fallback_err     = aerr;
                #pragma unroll
                for (int w = 0; w < AW; w++) s_fallback_pass[w] = mask[w];
            }

            // Best: count >= SLOT_QUOTA and lowest amax_err.
            if (total >= SLOT_QUOTA) {
                *s_search_done = 1;
                if (aerr < *s_best_err) {
                    *s_best_fmt     = FMT;
                    *s_best_scale   = outer;
                    *s_best_err     = aerr;
                    #pragma unroll
                    for (int w = 0; w < AW; w++) s_best_pass[w] = mask[w];
                }
            }
        }
        // Zero the early-rejection state for the next search.
        #pragma unroll
        for (int si = 0; si <= NUM_SCALE_CANDIDATES; si++) s_fail[si] = 0;
        *s_dropped = 0;
    }
    __syncthreads();
}
