// The live MoE worker and the grouped entry that launches it.
//
// `moe_live_ahead_piece` copies one piece of a read-ahead item,
// `moe_live_worker` runs a worker of a live launch over its items, and
// `quantized_matmul_grouped_entry` decodes each (expert, batch-slice) pair from
// the device tables and runs one tile.

// Piece `c` of read-ahead item `a` (`moe_read_ahead.cuh`): the item's bytes
// `[c · piece, min(bytes, (c + 1) · piece))` from its warm-slot image into its
// claimed slot, every thread four 16-byte loads in flight, then their stores.
// Each thread fences its own stores before the block counts the piece done, so
// the worker that counts an item's last piece publishes an expert whose every
// byte is written; it stores the owner tag, up, down, a system fence, then
// gate. Called by every thread of the block.
static __device__ void moe_live_ahead_piece(const MoeLive& live, int a, int c, int tid) {
    const unsigned long long* item = live.ahead + 1 + (size_t)a * AHEAD_ITEM_WORDS;
    const unsigned long long bytes = item[2];
    const unsigned long long piece = ahead_piece_bytes(bytes);
    const unsigned long long lo = (unsigned long long)c * piece;
    const unsigned long long hi = lo + piece < bytes ? lo + piece : bytes;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(item[0]));
    uint8_t* dst = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(item[1]));
    const long long units = lo < hi ? (long long)((hi - lo) / 16ull) : 0ll;
    for (long long b = tid; b < units; b += NUM_THREADS * 4) {
        uint4 v[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const long long idx = b + (long long)u * NUM_THREADS;
            if (idx < units) {
                v[u] = moe_live_ld16(src + lo + (size_t)idx * 16);
            }
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const long long idx = b + (long long)u * NUM_THREADS;
            if (idx < units) {
                *reinterpret_cast<uint4*>(dst + lo + (size_t)idx * 16) = v[u];
            }
        }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0 && atomicAdd(&live.ahead_done[a], 1u) == AHEAD_CHUNKS - 1u) {
        __threadfence();
        if (item[9] != 0ull) {
            *reinterpret_cast<volatile unsigned int*>(static_cast<uintptr_t>(item[9])) =
                (unsigned int)item[10];
        }
        *reinterpret_cast<volatile unsigned long long*>(static_cast<uintptr_t>(item[4])) = item[7];
        *reinterpret_cast<volatile unsigned long long*>(static_cast<uintptr_t>(item[5])) = item[8];
        __threadfence_system();
        *reinterpret_cast<volatile unsigned long long*>(static_cast<uintptr_t>(item[3])) = item[6];
        if (live.stall != nullptr) {
            atomicAdd(&live.stall[5], 1ull);
            atomicAdd(&live.stall[6], bytes);
        }
    }
}

// A worker of a live launch (see "A live expert table" above): items
// `(remote expert r, row tile j)` off the launch's counter until none are left.
// Per item: the expert's entry (waited for while cold); the row tile's slice
// copied into this worker's scratch slot as a 32-row KO matrix —
// `[K block][4 row groups]`, piece `k` from chunk `k·(nrows/8) + 4j` of the
// source; one barrier; then the unmodified impl over the slot (`nrows = 32`,
// row tile 0) for every token tile of `r`, its output at `dst + 32j`. The impl
// indexes chunks `k·(nrows/8) + warp` and stores `dst[token·dst_stride + row]`
// (`store_tile_output`), so each output row gets the arithmetic it gets from
// VRAM, bit for bit.
template <int qk, int qi, typename block_q_t, int vdr, typename output_t, int N_SUB>
static __device__ void moe_live_worker(
    const uint64_t* __restrict__ weight_ptrs,
    const int* __restrict__ tile_b_start,
    const int* __restrict__ tile_b_cnt,
    const block_q8a128* __restrict__ vy,
    output_t* __restrict__ dst,
    int ncols_x, int nrows_x, int y_stride, int dst_stride, int sum_norm,
    const MoeLive& live,
    int worker,
    int8_t smem_A_i8[][N_SUB * 16][KI8_STRIDE],
    half2 smem_A_ds[][N_SUB * 16],
    uint8_t* smem_W_flat)
{
    using block_c_t = block_compact_t<block_q_t>;
    constexpr int CB = int8_chunk_bytes<block_c_t>::value;   // one 8-row chunk
    constexpr int UNITS_PER_PIECE = 4 * CB / 16;                // 16 B units per K block
    static_assert((4 * CB) % 16 == 0, "a row tile's K-block piece is whole 16-byte units");
    __shared__ int s_item;
    __shared__ unsigned long long s_src;
    const int tid = threadIdx.y * WARP_SIZE_TC + threadIdx.x;
    const int row_tiles = nrows_x / N_TILE;
    const int units = (ncols_x / K_TILE) * UNITS_PER_PIECE;
    const int n_items = live.header[3] * row_tiles;
    // The CPU class's result rows (the down launch only): MOE_LIVE_CPU_ROWS a
    // CPU item, after the launch's own items so a worker waiting on the pool
    // never delays a remote expert's copy.
    const int cpu_rows = live.cpu_done != nullptr ? live.header[5] : 0;
    const int n_cpu = (cpu_rows + MOE_LIVE_CPU_ROWS - 1) / MOE_LIVE_CPU_ROWS;
    // The launch's own items, the CPU items, then the read-ahead pieces.
    const int n_all = n_items + n_cpu +
                      (live.ahead != nullptr ? (int)live.ahead[0] * AHEAD_CHUNKS : 0);
    uint8_t* slot = live.scratch + (size_t)worker * live.slot_bytes;
    if (live.stall != nullptr && worker == 0 && tid == 0) {
        atomicAdd(&live.stall[4], 1ull);
    }
    for (;;) {
        if (tid == 0) {
            s_item = atomicAdd(live.counter, 1);
        }
        __syncthreads();
        const int it = s_item;
        if (it >= n_all) {
            return;
        }
        if (it >= n_items + n_cpu) {
            // Read-ahead pieces never wait on a cold expert, so a fault does not
            // touch them: each publishes an expert its own copy completes.
            const int p = it - n_items - n_cpu;
            moe_live_ahead_piece(live, p / AHEAD_CHUNKS, p % AHEAD_CHUNKS, tid);
            continue;
        }
        if (it >= n_items) {
            // A CPU item: wait for the pool, then copy its result rows into the
            // rows bucketize left the CPU experts. The scatter reads them there
            // exactly as it reads a GPU-computed row.
            if constexpr (std::is_same_v<output_t, float>) {
                // `s_src` carries the outcome: the bm128 instantiation sits at
                // the 48 KB static shared-memory cap, so this adds no word.
                if (tid == 0) {
                    s_src = moe_live_cpu_wait(live) ? 1ull : 0ull;
                }
                __syncthreads();
                if (s_src == 0ull) {
                    continue;
                }
                const int c = it - n_items;
                const int r0 = c * MOE_LIVE_CPU_ROWS;
                const int r1 = min(cpu_rows, r0 + MOE_LIVE_CPU_ROWS);
                const int units = nrows_x / 4;  // 16-byte units in one row
                const int total = (r1 - r0) * units;
                for (int b = tid; b < total; b += NUM_THREADS * MOE_LIVE_LOADS) {
                    uint4 v[MOE_LIVE_LOADS];
                    int at[MOE_LIVE_LOADS];
                    #pragma unroll
                    for (int u = 0; u < MOE_LIVE_LOADS; ++u) {
                        at[u] = b + u * NUM_THREADS;
                        if (at[u] < total) {
                            const int rr = r0 + at[u] / units;
                            const int o = at[u] - (rr - r0) * units;
                            v[u] = moe_live_ld16_cv(reinterpret_cast<const uint8_t*>(
                                live.cpu_rows + (size_t)rr * nrows_x + (size_t)o * 4));
                        }
                    }
                    #pragma unroll
                    for (int u = 0; u < MOE_LIVE_LOADS; ++u) {
                        if (at[u] < total) {
                            const int rr = r0 + at[u] / units;
                            const int o = at[u] - (rr - r0) * units;
                            const size_t row = live.cpu_dst_row[rr];
                            *reinterpret_cast<uint4*>(dst + row * (size_t)dst_stride + (size_t)o * 4) = v[u];
                        }
                    }
                }
            }
            continue;
        }
        const int r = it / row_tiles;
        const int j = it - r * row_tiles;
        const int expert = live.remote[4 * r];
        const int first = live.remote[4 * r + 1];
        const int n_tiles = live.remote[4 * r + 2];
        unsigned long long waited = 0;
        if (tid == 0) {
            if (live.remote[4 * r + 3] != 0) {
                waited = moe_live_source(expert, live, &s_src);
            } else {
                s_src = weight_ptrs[expert];
            }
        }
        __syncthreads();
        // The wait gave the expert up (the fault word names it): skip the item.
        if (s_src == 0ull) {
            continue;
        }
        const uint8_t* src = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(s_src));
        // A promotion slot is the copy's ONLY destination: the slice lands at its
        // place in the slot image — the source's own layout — and the tile is
        // computed from there, as a VRAM-resident expert's would be. The expert is
        // in VRAM once the layer is done, for no write beyond the one the compute
        // needed anyway and no second crossing of the link. Without a slot the
        // slice goes to this worker's scratch as a compact 32-row matrix.
        const unsigned long long promo_base = live.remote_dst != nullptr ? live.remote_dst[r] : 0ull;
        uint8_t* promo = promo_base == 0ull
                             ? nullptr
                             : reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(promo_base + live.dst_offset));
        const unsigned long long t_copy = moe_live_now();
        // The mini loop: every thread `MOE_LIVE_LOADS` 16-byte loads in flight, then
        // their stores. A slice is a few PCIe round trips, so loads in flight per
        // round trip are what the worker's rate is.
        for (int b = tid; b < units; b += NUM_THREADS * MOE_LIVE_LOADS) {
            uint4 v[MOE_LIVE_LOADS];
            size_t at[MOE_LIVE_LOADS];
            #pragma unroll
            for (int u = 0; u < MOE_LIVE_LOADS; ++u) {
                const int idx = b + u * NUM_THREADS;
                if (idx < units) {
                    const int k = idx / UNITS_PER_PIECE;
                    const int o = idx - k * UNITS_PER_PIECE;
                    at[u] = ((size_t)k * (nrows_x / 8) + 4 * j) * CB + (size_t)o * 16;
                    v[u] = moe_live_ld16(src + at[u]);
                }
            }
            #pragma unroll
            for (int u = 0; u < MOE_LIVE_LOADS; ++u) {
                const int idx = b + u * NUM_THREADS;
                if (idx < units) {
                    if (promo != nullptr) {
                        *reinterpret_cast<uint4*>(promo + at[u]) = v[u];
                    } else {
                        reinterpret_cast<uint4*>(slot)[idx] = v[u];
                    }
                }
            }
        }
        // The slot's stores are visible to the block's own `cp.async.cg` reads
        // (both through L2) once every thread has passed this barrier.
        __syncthreads();
        if (live.cold_wait != nullptr && tid == 0 && j == 0 && live.remote[4 * r + 3] != 0) {
            atomicAdd(&live.cold_wait[0], waited);
            atomicAdd(&live.cold_wait[1], 1ull);
        }
        if (live.stall != nullptr && tid == 0) {
            atomicAdd(&live.stall[0], waited);
            atomicAdd(&live.stall[1], 1ull);
            atomicAdd(&live.stall[2], (unsigned long long)units * 16ull);
            atomicAdd(&live.stall[3], moe_live_now() - t_copy);
        }
        if constexpr (is_scale_separate<block_c_t>::value) {
            for (int t = 0; t < n_tiles; ++t) {
                const int tile = first + t;
                if (promo != nullptr) {
                    // Row tile `j` of the whole projection, read where it landed —
                    // the chunks `k·(nrows/8) + 4j + warp` this worker just wrote.
                    grouped_matmul_impl_int8<qk, qi, block_q_t, vdr, output_t, N_SUB>(
                        reinterpret_cast<const block_c_t*>(promo), vy, dst,
                        ncols_x, nrows_x, y_stride, dst_stride,
                        tile_b_start[tile], tile_b_cnt[tile], j,
                        smem_A_i8, smem_A_ds, smem_W_flat, sum_norm);
                } else {
                    grouped_matmul_impl_int8<qk, qi, block_q_t, vdr, output_t, N_SUB>(
                        reinterpret_cast<const block_c_t*>(slot), vy, dst + (size_t)j * N_TILE,
                        ncols_x, N_TILE, y_stride, dst_stride,
                        tile_b_start[tile], tile_b_cnt[tile], 0,
                        smem_A_i8, smem_A_ds, smem_W_flat, sum_norm);
                }
                // The next tile's prologue refills the buffers this one read.
                __syncthreads();
            }
        }
    }
}

// Grouped entry: decode (expert, batch-slice) from device tables and run one tile.
//   block = 128 threads (4 warps × 32); the GRID AXIS ORDER is a runtime choice:
//   row_fast != 0 → grid = (row_tiles, total_tiles): consecutively-scheduled blocks
//     are the row-tiles of ONE token tile, sharing that tile's activation slab
//     (~hundreds of KB — L2-resident) and together streaming each expert's weight
//     rows exactly once. The right order when the STACKED activation exceeds L2:
//     with token tiles fast, every row-tile wave re-streamed the whole activation
//     from DRAM — ncu measured 12.6 GB per prefill-scale launch vs a ~2.3 GB
//     minimum.
//   row_fast == 0 → grid = (total_tiles, row_tiles): token tiles fast, weight rows
//     shared between consecutive blocks. The candidate order when the whole
//     activation already fits L2 (residency is then free either way).
// Both orders run the same blocks with the same per-block work — the choice only
// changes schedule order, so outputs are bit-identical. The host picks per launch
// from the activation working set vs L2 (see grouped_grid_row_fast in cuda.rs).
//
// RN (int8 only) sets the row tile to 32·RN rows — the host sizes the row-tile axis to match
// (run_grouped_quantized_matmul). A live launch's workers keep the 32-row slices their scratch
// slots hold, and run the RN = 1 impl over them; the bits are the same either way.
template <int qk, int qi, typename block_q_t, int vdr, typename act_t, typename output_t, int N_SUB = 1,
          int RN = 1>
static __device__ void quantized_matmul_grouped_entry(
    const uint64_t* __restrict__ weight_ptrs,  // [num_experts] device weight pointers
    const int* __restrict__ tile_expert,       // [total_tiles] owning expert
    const int* __restrict__ tile_b_start,      // [total_tiles] stacked batch start
    const int* __restrict__ tile_b_cnt,        // [total_tiles] tokens in tile (1..16·N_SUB)
    const act_t* __restrict__ vy,
    output_t* __restrict__ dst,
    int ncols_x, int nrows_x, int y_stride, int dst_stride, int row_fast,
    // The activation operand's `SumScale::as_code()`. Defaulted because this
    // entry also serves FLOAT activations, which carry no q8a128 header at all —
    // the `if constexpr` below discards it for them, so the FP instantiations
    // and their launchers do not name a value that has no meaning for them.
    int sum_norm = 0,
    // A launch over a live expert table carries a `MoeLive` and the worker grid
    // (see "A live expert table" above); every other launch passes
    // `abort == nullptr`.
    MoeLive live = MoeLive{})
{
    using block_c_t = block_compact_t<block_q_t>;

    // Same decode / grid / store for every activation type; only the smem layout and
    // the per-tile compute differ. q8a128 → INT8 m16n8k32; FP → FP16 m16n8k16. N_SUB
    // scales the int8 token tile (mode-1: 16, mode-2: 32 weight-reuse); the impl sweeps
    // the once-loaded weight over each 16-row sub-tile, partial sub-tiles writing nothing.
    if constexpr (std::is_same_v<act_t, block_q8a128>) {
        constexpr int BATCH_I8 = N_SUB * 16;
        // Activation buffering matches the impl: double for N_SUB ≤ 4, single for N_SUB = 8
        // (the 128-token double buffer capped occupancy at 2 blocks/SM — see the impl note).
        constexpr int ABUF = (N_SUB >= 8) ? 1 : 2;
        __shared__ __align__(16) int8_t smem_A_i8[ABUF][BATCH_I8][KI8_STRIDE];
        __shared__ __align__(16) half2 smem_A_ds[ABUF][BATCH_I8];
        __shared__ uint8_t smem_W_flat[(N_TILE / 8) * RN * RING_I8 * int8_chunk_bytes<block_c_t>::value];
        int tile;
        int row_tile_idx;
        if (live.abort != nullptr) {
            // Live grid: row tiles on x; the first `worker_rows` rows are the
            // workers, laid out row by row, and the rows after them the tiles.
            const int worker_rows = (live.workers + (int)gridDim.x - 1) / (int)gridDim.x;
            if ((int)blockIdx.y < worker_rows) {
                const int worker = (int)(blockIdx.y * gridDim.x + blockIdx.x);
                if (worker < live.workers) {
                    moe_live_worker<qk, qi, block_q_t, vdr, output_t, N_SUB>(
                        weight_ptrs, tile_b_start, tile_b_cnt, vy, dst, ncols_x, nrows_x,
                        y_stride, dst_stride, sum_norm, live, worker, smem_A_i8, smem_A_ds,
                        smem_W_flat);
                }
                return;
            }
            tile = (int)blockIdx.y - worker_rows;
            // A remote expert's tile: a worker computes it.
            if (tile < live.header[4]) {
                return;
            }
            row_tile_idx = blockIdx.x;
        } else {
            tile = row_fast ? blockIdx.y : blockIdx.x;
            row_tile_idx = row_fast ? blockIdx.x : blockIdx.y;
        }
        // A zero-count tile is padding: device-built tile tables (moe_bucketize.cu)
        // are launched at the `n_tokens × k` upper bound so the host never reads a
        // data-dependent tile count back. Exit before touching the pointer table.
        const int b_cnt = tile_b_cnt[tile];
        if (b_cnt == 0) {
            return;
        }
        const block_c_t* weights = reinterpret_cast<const block_c_t*>(
            static_cast<uintptr_t>(weight_ptrs[tile_expert[tile]]));
        const int b_start = tile_b_start[tile];
        // KO-only int8 impl (inline-scale k1024). Non-KO → discarded (no-op kernel).
        if constexpr (is_scale_separate<block_c_t>::value) {
            grouped_matmul_impl_int8<qk, qi, block_q_t, vdr, output_t, N_SUB, 2, false, false, RN>(
                weights, vy, dst, ncols_x, nrows_x, y_stride, dst_stride,
                b_start, b_cnt, row_tile_idx, smem_A_i8, smem_A_ds, smem_W_flat, sum_norm);
        }
    } else {
        const int tile = row_fast ? blockIdx.y : blockIdx.x;
        const int b_cnt = tile_b_cnt[tile];
        if (b_cnt == 0) {
            return;
        }
        const block_c_t* weights = reinterpret_cast<const block_c_t*>(
            static_cast<uintptr_t>(weight_ptrs[tile_expert[tile]]));
        const int b_start = tile_b_start[tile];
        const int row_tile_idx = row_fast ? blockIdx.x : blockIdx.y;
        using compute_t = std::conditional_t<
            std::is_same_v<act_t, float>, half,
            std::conditional_t<
                std::is_same_v<act_t, half>, half,
                std::conditional_t<
                    std::is_same_v<act_t, __nv_bfloat16>, __nv_bfloat16,
                    std::conditional_t<
                        std::is_same_v<act_t, __nv_fp8_e4m3>, __nv_fp8_e4m3, half>>>>;
        __shared__ compute_t smem_A[BATCH_TILE][K_STRIDE];
        __shared__ uint8_t smem_W_flat[N_TILE * sizeof(block_c_t)];
        grouped_matmul_impl<block_c_t, compute_t, act_t, output_t>(
            weights, vy, dst, ncols_x, nrows_x, y_stride, dst_stride,
            b_start, b_cnt, row_tile_idx, smem_A, smem_W_flat);
    }
}
