// The host launcher for the INT8 decode attention kernels.
//
// `launch_int8_decode_attn` launches the decode attention. It returns 0 on
// success and 1 when the split-KV partial pool could not be set up.

// Returns 0 on success, 1 when the split-KV partial pool could not be
// allocated for a launch that requires it (split accumulation, warp-stripe,
// or a q8 emit — the combine kernel is the only q8 emitter and the stripe
// kernels have no direct-write form). The caller must treat 1 as a failed
// wave: nothing was launched, `out`/`q8_out` hold no result.
template <typename Q_T, typename T, typename O, int HEAD_DIM>
int launch_int8_decode_attn(
    const Q_T* q,
    const uint8_t* headers_ptr,
    O* out,
    int num_active_slots,
    int n_q_head,
    int n_kv_head,
    float softmax_scale,
    const T* k_new,
    const T* v_new,
    const RopeRungs rungs,
    int rope_interleaved,
    cudaStream_t stream,
    uint8_t* q8_out = nullptr,  // non-null → B2 fused q8a128 context (combine path, HEAD_DIM % 128 == 0)
    const O* gate = nullptr,    // non-null → combine applies sigmoid(gate) ⊙ val (requires q8_out)
    int64_t gate_slot_stride = 0, // elements between consecutive slots' gate rows
                                  // (n_q_head·HEAD_DIM when the gate is contiguous;
                                  // the fused [q|gate] projection's row width when
                                  // the gate is a strided view of it)
    QsaSel sel = {nullptr, nullptr, nullptr, nullptr, 0, 1} // QSA: one selection row per slot
) {
    int heads_per_group = (n_kv_head > 0) ? (n_q_head / n_kv_head) : 1;
    if (heads_per_group < 1) heads_per_group = 1;
    const bool use_wide = (HEAD_DIM >= 128) && (heads_per_group > 8);

    // ── Split-KV factor: fan each (slot, kv_head)'s KV-tile loop across multiple
    // blocks so the grid fills the SMs when batch*heads is a small grid. Target
    // ~2 waves at the register-bound ~3 blocks/SM; clamp to MAX_SPLITS. Empty
    // splits (short context) early-out cheaply; S=1 keeps the direct-write path.
    int base_blocks = num_active_slots * n_kv_head;
    int num_splits = 1;
    if (base_blocks > 0) {
        int target_blocks = fused_attn_sm_count() * 3 * 2;
        num_splits = (target_blocks + base_blocks - 1) / base_blocks;
    }
    // **Raised from 32.** The split factor is what fills the SMs when the grid
    // is otherwise `slots × kv_heads` — 2 for a single-sequence decode, which is
    // two blocks on a 110-SM card. The heuristic above asks for ~330 and the old
    // clamp handed back 32, i.e. a 64-block grid: measured by ncu at 5.3% DRAM
    // throughput and 17.8% memory throughput, so the kernel was neither
    // bandwidth- nor compute-bound but simply starved of blocks.
    //
    // Splits past the tile count do no work (their range is empty) and cost only
    // a null partial each, so this is a ceiling rather than a target — the
    // `tiles_per_split` arithmetic below still hands out at most one tile per
    // split once the work runs out.
    if (num_splits < 1) num_splits = 1;
    if (num_splits > MAX_SPLITS) num_splits = MAX_SPLITS;

    // ── Wide heads (HEAD_DIM 256) with a group of ≤ 16 query heads take the
    // INT8 tile kernel: one 32-token slice per tile on the tensor cores, the
    // group's heads as the MMA's M rows, 256 threads per block. It always
    // emits partials (one per split per head) and combines. Under a QSA
    // selection its work is the entry count, so the split factor is derived
    // from that — INT8_TILE_ENTRIES_PER_SPLIT entries per block — rather
    // than from the dense token-walk heuristic.
    if constexpr (HEAD_DIM == 256) {
        if (heads_per_group <= TILE_M_ROWS) {
            // Dense: at most one wave of the tile kernel's real residency
            // (TILE_MIN_BLOCKS per SM, the 128-register cap), the depth split
            // evenly across it. The generic heuristic above sizes for a
            // 3-blocks/SM kernel and hands this one 2.33 waves — the trailing
            // third of a wave ran at a third of the occupancy while the DRAM
            // it was walking sat 15% idle (ncu: SM Active 791K of 929K cycles
            // at 128K depth). The split count rounds DOWN: rounding up put
            // 224 blocks on a 220-block card at 8 slots × 2 heads, and the
            // four that did not fit ran as a second wave by themselves —
            // ncu "Waves Per SM 1.02", the kernel taking two block-times for
            // one wave of work.
            const int wave_blocks = fused_attn_sm_count() * TILE_MIN_BLOCKS;
            int splits = (base_blocks > 0) ? wave_blocks / base_blocks : 1;
            if (splits < 1) splits = 1;
            if (splits > MAX_SPLITS) splits = MAX_SPLITS;
            // **Do not oversubscribe the split past one wave to "bound each
            // block's depth" — it was tried and it is worse.** The reasoning is
            // seductive: at `base_blocks >= wave_blocks` the rounding above
            // hands back 1, every block then walks its slot's whole KV, and the
            // concurrent footprint grows with slot count. Forcing the grid to
            // ~4 waves instead (`splits <= 1` → oversubscribe) was measured on a
            // 110-SM card at ~660-token depth:
            //
            // |  slots | splits 1 (avg ms) | oversubscribed (avg ms) |
            // |--------|-------------------|-------------------------|
            // |     64 | 0.868             | 15.112  (splits 7) ✗    |
            // |    128 | 34.420            | 26.819  (splits 4) ~    |
            //
            // 17× worse at 64 slots for a 1.3× gain at 128, and aggregate decode
            // throughput did not move at either width. Whatever makes the wide
            // grid slow, per-block depth is not it.
            if (sel.entries != nullptr) {
                splits = ((int)sel.stride + INT8_TILE_ENTRIES_PER_SPLIT - 1)
                         / INT8_TILE_ENTRIES_PER_SPLIT;
                if (splits < 1) splits = 1;
                if (splits > MAX_SPLITS) splits = MAX_SPLITS;
            }
            float* pa = nullptr;
            float* pm = nullptr;
            fused_attn_partial_pool((int64_t)num_active_slots * n_q_head, splits,
                                    HEAD_DIM, &pa, &pm);
            if (pa == nullptr || pm == nullptr) {
                return 1;
            }
            // An in-kernel split merge (a pairwise tree over the partial
            // pool, the last-arriving block of each pair merging) was
            // measured against this two-launch form: its serial tail — up
            // to seven levels of device fence, global atomic and dependent
            // loads on the critical path — cost a constant 36 µs at every
            // depth, seven times the launch it saved. The combine kernel
            // stays.
            dim3 grid(num_active_slots, n_kv_head, splits);
            dim3 block(TILE_THREADS);
            if (rope_interleaved) {
                int8_decode_tile_kernel<Q_T, T, HEAD_DIM, true><<<grid, block, 0, stream>>>(
                    q, headers_ptr, num_active_slots, n_q_head, n_kv_head,
                    softmax_scale, k_new, v_new, rungs, pa, pm, sel);
            } else {
                int8_decode_tile_kernel<Q_T, T, HEAD_DIM, false><<<grid, block, 0, stream>>>(
                    q, headers_ptr, num_active_slots, n_q_head, n_kv_head,
                    softmax_scale, k_new, v_new, rungs, pa, pm, sel);
            }
            const int num_rows = num_active_slots * n_q_head;
            const int64_t g_stride =
                (gate_slot_stride != 0) ? gate_slot_stride : (int64_t)n_q_head * HEAD_DIM;
            // The combine also commits the write-slice lengths.
            int8_decode_combine_kernel<O, HEAD_DIM><<<num_rows, HEAD_DIM, 0, stream>>>(
                out, pa, pm, num_rows, splits, q8_out, gate, g_stride, n_q_head,
                headers_ptr, n_kv_head);
            return 0;
        }
    }

    auto launch = [&](auto warps_const, auto rope_const) -> int {
        constexpr int WARPS_PER_BLOCK = decltype(warps_const)::value;
        constexpr bool ROPE_INTERLEAVED = decltype(rope_const)::value;

        // Warp-stripe (1C) when heads_per_group <= WARPS: every warp computes
        // (in pairs, each pair over its own KV stripe), the block merges its
        // warps' flash states and writes ONE partial per head, so it always
        // writes partials + combines. Every kernel emits `num_splits` partials
        // per row. hpg==8 (e.g. Qwen3-MoE, n_q/n_kv=32/4) is included: the
        // batched-M MMA path is gap-aware and the warp=head path is not, so
        // route it here.
        const bool use_stripe = (heads_per_group >= 1 && heads_per_group <= 8
                                 && heads_per_group <= WARPS_PER_BLOCK);
        // Under a QSA selection the stripe's work is the selection's entry
        // count, not the depth, so the split factor is derived from it: one
        // entry (≤4 cells) per pair-stripe. The dense heuristic above sizes
        // for a token walk and would fan a 513-entry row over 256 splits ×
        // 4 pairs — a thousand stripes for five hundred entries, each block
        // paying its Q-staging prologue and block merge for nothing.
        int splits = num_splits;
        if (use_stripe && sel.entries != nullptr) {
            constexpr int PAIRS = WARPS_PER_BLOCK / 2;
            const int stripes = (int)sel.stride;   // entries per row ≥ cnt[slot]
            splits = (stripes + PAIRS - 1) / PAIRS;
            if (splits < 1) splits = 1;
            if (splits > MAX_SPLITS) splits = MAX_SPLITS;
        }
        const int partials_per_row = splits;
        // One predicate decides the whole route: split accumulation, the stripe
        // kernels (which have no direct-write form), and a q8 emit (the combine
        // kernel is the only emitter, and `out` is null on that path) all
        // require the partial pool + combine. If the pool cannot be allocated,
        // launching anything would either write through a null `out` or leave
        // `q8_out` uninitialized while reporting success — so a required pool
        // that fails to allocate fails the launch loudly instead.
        const bool need_pool = use_stripe || (splits > 1) || (q8_out != nullptr);
        float* pa = nullptr;
        float* pm = nullptr;
        if (need_pool) {
            fused_attn_partial_pool((int64_t)num_active_slots * n_q_head, partials_per_row,
                                    HEAD_DIM, &pa, &pm);
            if (pa == nullptr || pm == nullptr) {
                return 1;
            }
        }

        dim3 grid(num_active_slots, n_kv_head, splits);
        dim3 block(WARP_SIZE * WARPS_PER_BLOCK);

        // **The stripe/bmma family only exists below 16 warps.**
        //
        // `use_stripe` requires `heads_per_group <= 8`, and 16 warps is selected
        // only when `heads_per_group > 8` (`use_wide`, above) — so at 16 warps
        // the branch below is unreachable by construction. Without the
        // `if constexpr` it was still *instantiated*: eight `BMMA` plus eight
        // `STRIPE` kernels per (head dim × rope × dtype), compiled into the
        // archive and never launched. Gating the instantiation rather than the
        // launch is what removes them.
        bool launched_stripe = false;
        if constexpr (WARPS_PER_BLOCK <= 8) {
        if (use_stripe) {
            // HPG compile-time so the per-head flash-state arrays stay in registers.
            // hd128 uses the batched-M INT8 tensor-core MMA; other head dims (no
            // 32-wide palette) use the CUDA-core warp-stripe.
            #define BMMA_LAUNCH(H)                                                         \
                int8_decode_bmma_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK,              \
                                        ROPE_INTERLEAVED, H>                               \
                    <<<grid, block, 0, stream>>>(                                          \
                        q, headers_ptr, num_active_slots, n_q_head, n_kv_head,             \
                        softmax_scale, k_new, v_new, rungs, pa, pm, sel)
            #define STRIPE_LAUNCH(H)                                                       \
                int8_decode_stripe_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK,            \
                                          ROPE_INTERLEAVED, H>                             \
                    <<<grid, block, 0, stream>>>(                                          \
                        q, headers_ptr, num_active_slots, n_q_head, n_kv_head,             \
                        softmax_scale, k_new, v_new, rungs, pa, pm, sel)
            if constexpr (HEAD_DIM == 128 && WARPS_PER_BLOCK <= 8) {
                // batched-M's per-warp tile smem fits at WARPS<=8 (~29 KB);
                // WARPS=16 (the hpg>8 wide path) never reaches use_stripe, so it
                // would only be a compiled-never-run instantiation that blows the
                // 48 KB cap — route it to the stripe instead.
                switch (heads_per_group) {
                    case 1: BMMA_LAUNCH(1); break;
                    case 2: BMMA_LAUNCH(2); break;
                    case 3: BMMA_LAUNCH(3); break;
                    case 4: BMMA_LAUNCH(4); break;
                    case 5: BMMA_LAUNCH(5); break;
                    case 6: BMMA_LAUNCH(6); break;
                    case 7: BMMA_LAUNCH(7); break;
                    case 8: BMMA_LAUNCH(8); break;
                    default: break;
                }
            } else {
                switch (heads_per_group) {
                    case 1: STRIPE_LAUNCH(1); break;
                    case 2: STRIPE_LAUNCH(2); break;
                    case 3: STRIPE_LAUNCH(3); break;
                    case 4: STRIPE_LAUNCH(4); break;
                    case 5: STRIPE_LAUNCH(5); break;
                    case 6: STRIPE_LAUNCH(6); break;
                    case 7: STRIPE_LAUNCH(7); break;
                    case 8: STRIPE_LAUNCH(8); break;
                    default: break;
                }
            }
            #undef BMMA_LAUNCH
            #undef STRIPE_LAUNCH
            launched_stripe = true;
        }
        }
        if (!launched_stripe) {
            // **Not at head_dim >= 128 with 8 warps.** There every group of 1–8
            // heads takes the batched-M kernel above, and more than 8 selects 16
            // warps (`use_wide`), so the warp=head kernel at 8 warps cannot be
            // reached. Gating its instantiation rather than its launch is what
            // keeps it out of the archive; arriving here anyway is a routing
            // fault, reported rather than run.
            if constexpr (HEAD_DIM >= 128 && WARPS_PER_BLOCK <= 8) {
                return 2;
            } else {
                // Existing INT8-MMA kernel (warp=head). `pa` is non-null exactly
                // when the route goes through partials + combine (need_pool held
                // and the alloc succeeded — a failed alloc returned above); null
                // `pa` is the single-block direct write.
                int8_decode_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, ROPE_INTERLEAVED>
                    <<<grid, block, 0, stream>>>(
                        q, headers_ptr, out, num_active_slots, n_q_head, n_kv_head,
                        softmax_scale, k_new, v_new, rungs, pa, pm, sel);
            }
        }

        // The write-slice commit rides in the combine when there is one; the
        // direct-write route (no partials) commits with its own launch.
        if (pa != nullptr) {
            int num_rows = num_active_slots * n_q_head;
            const int64_t g_stride =
                (gate_slot_stride != 0) ? gate_slot_stride : (int64_t)n_q_head * HEAD_DIM;
            int8_decode_combine_kernel<O, HEAD_DIM><<<num_rows, HEAD_DIM, 0, stream>>>(
                out, pa, pm, num_rows, partials_per_row, q8_out, gate, g_stride, n_q_head,
                headers_ptr, n_kv_head);
        } else {
            constexpr int COMMIT_THREADS = 128;
            dim3 commit_grid((num_active_slots + COMMIT_THREADS - 1) / COMMIT_THREADS);
            commit_decode_write_len_kernel<HEAD_DIM><<<commit_grid, COMMIT_THREADS, 0, stream>>>(
                headers_ptr, num_active_slots, n_kv_head);
        }
        return 0;
    };

    // 4-way dispatch over (use_wide, rope_interleaved). use_wide selects
    // WARPS_PER_BLOCK=16 for heads_per_group > 8 (e.g. Llama-3 70B class), else 8.
    //
    // The `if constexpr` is the same argument as the stripe gate above, one level
    // out: `use_wide` carries `HEAD_DIM >= 128`, so below that head dim it is
    // always false and the entire 16-warp half of this dispatch — every kernel
    // `launch` instantiates — was compiled for a branch that cannot be taken.
    if constexpr (HEAD_DIM >= 128) {
        if (use_wide) {
            if (rope_interleaved) {
                return launch(std::integral_constant<int, 16>{}, std::true_type{});
            }
            return launch(std::integral_constant<int, 16>{}, std::false_type{});
        }
    }
    // **No 8-warp launch at head_dim 256.** Up to TILE_M_ROWS heads per group
    // the tile kernel above has already run and returned, and more than that
    // is `use_wide` — so the 8-warp half (its stripe kernels for every head
    // count and its warp=head fallback) is unreachable here, and is not
    // instantiated. Arriving here anyway is a routing fault, reported rather
    // than run.
    static_assert(TILE_M_ROWS >= 8, "the tile kernel covers every 8-warp group at head_dim 256");
    if constexpr (HEAD_DIM == 256) {
        return 2;
    } else {
        if (rope_interleaved) {
            return launch(std::integral_constant<int, 8>{}, std::true_type{});
        }
        return launch(std::integral_constant<int, 8>{}, std::false_type{});
    }
}
