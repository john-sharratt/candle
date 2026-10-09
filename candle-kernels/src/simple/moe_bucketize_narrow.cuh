// =============================================================================
// MoE BUCKETIZE — the narrow kernel (n_tokens · k ≤ BUCKETIZE_NARROW)
// =============================================================================
// A decode step, a draft step and a verify route a few dozen assignments, and
// at that width the general kernel's cost is its chain of ~22 block barriers
// and a serial per-token tail, not its work. This kernel writes the same bytes
// — every table, the header, the summary, the remote list, the snapshot and
// the promotion ring's effects — in five barriers, one assignment per thread:
//
//   * P0 — thread i loads assignment i; the per-expert counters are zeroed; the
//     assignment's composite key `e << 9 | i` (INVALID_ROW for a sentinel)
//     goes to shared memory and the warp's valid flags to a ballot word.
//     Thread 0 stores the started ticket here, so its fence overlaps the load.
//   * P1 — the first assignment of an expert to reach its counter (a shared
//     atomic) is the expert's REPRESENTATIVE, and issues the live-entry loads
//     at once; while they are in flight every assignment counts the keys below
//     its own. That count IS its row: rows ascend by expert, then by i — the
//     buckets of the general kernel's stable counting sort. Its token-major
//     slot is the valid assignments before its token (the ballot words) plus
//     the keys of its own token below its own — the per-token ascending-row
//     order of the general kernel's insertion sort. The representative then
//     classifies, checks and snapshots its expert.
//   * P2 — one block scan over the expert axis of a packed u64 (thread e holds
//     expert e) gives the bucket offsets, the tile order (pinned, cold, VRAM,
//     each ascending) and the remote list at once; each routed expert writes
//     its tiles and its list entry, and thread e expert e's summary word.
//   * then thread 0 runs the promotion walk (only with a remote expert routed
//     or read-ahead asked for) and publishes the summary's sequence word behind
//     its system fence.
//
// The scratch tables `inv` and `scan` are not written: they exist for the
// general kernel's phases and nothing else reads them.
// =============================================================================
#pragma once

#include "moe_bucketize_common.cuh"
#include "moe_bucketize_live.cuh"

// The widest launch the narrow kernel takes: `n_tokens · k` at most this.
#define BUCKETIZE_NARROW 256
#define NARROW_WORDS (BUCKETIZE_NARROW / 32)
static_assert(BUCKETIZE_NARROW <= BUCKETIZE_THREADS, "one assignment per thread");
static_assert(BUCKETIZE_NARROW <= 512, "the composite key holds i in 9 bits");
static_assert(MAX_EXPERTS <= 512, "the composite key holds e above 9 bits of i");
// The P2 scan packs seven 9-bit lanes into a u64; each lane's total is at most
// the assignment count (offsets, tiles and experts are each bounded by it).
#define NARROW_LANE_BITS 9
#define NARROW_LANE_MASK ((1ull << NARROW_LANE_BITS) - 1ull)
static_assert(BUCKETIZE_NARROW < (1 << NARROW_LANE_BITS), "a lane's total fits its bits");
static_assert(7 * NARROW_LANE_BITS <= 64, "seven lanes fit a u64");
// Lane positions: the bucket offset, the three classes' tiles in tile-order
// rank, the pinned and cold remote experts, and the routed experts.
#define NL_OFF 0
#define NL_TILES 1 // + class rank: 1 pinned, 2 cold, 3 VRAM
#define NL_REMOTE 4 // + 0 pinned, 1 cold
#define NL_ACTIVE 6

__device__ __forceinline__ uint32_t narrow_lane(unsigned long long v, int lane) {
    return (uint32_t)((v >> (NARROW_LANE_BITS * lane)) & NARROW_LANE_MASK);
}

// The valid assignments below index `x`, from the per-warp ballot words.
__device__ __forceinline__ uint32_t narrow_valid_before(const uint32_t* vmask, int x) {
    uint32_t n = 0;
    const int w_end = x >> 5;
    for (int w = 0; w < w_end; w++) {
        n += (uint32_t)__popc(vmask[w]);
    }
    const int b = x & 31;
    if (b != 0) {
        n += (uint32_t)__popc(vmask[w_end] & ((1u << b) - 1u));
    }
    return n;
}

extern "C" __global__ void __launch_bounds__(BUCKETIZE_THREADS) moe_bucketize_narrow_kernel(
    MOE_BUCKETIZE_KERNEL_PARAMS)
{
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int a_ub = n_tokens * k;

    __shared__ __align__(16) uint32_t sh_key[BUCKETIZE_NARROW];
    __shared__ uint32_t sh_vmask[NARROW_WORDS];
    // Per expert: assignments routed (the representative's atomic), whether a
    // decode-scored token routed it, and — written by its representative — its
    // class and in-flight promotion mark.
    __shared__ int32_t sh_counts[MAX_EXPERTS];
    __shared__ uint8_t sh_dec[MAX_EXPERTS];
    __shared__ uint8_t sh_cls[MAX_EXPERTS];
    __shared__ uint8_t sh_marked[MAX_EXPERTS];
    // The remote list's experts, in list order, for the promotion walk.
    __shared__ int32_t sh_remote_e[BUCKETIZE_NARROW];
    __shared__ unsigned long long sh_scan[BUCKETIZE_WARPS + 1];

    const bool live = gate_row != nullptr;

    // ── P0: load, zero, keys ──
    const bool in_list = tid < a_ub;
    const uint32_t e = in_list ? topk_ids[tid] : INVALID_ROW;
    if (live && started_rows != nullptr && tid == 0) {
        store_started_ticket(started_rows, row, ticket);
    }
    if (tid < n_experts) {
        sh_counts[tid] = 0;
        sh_dec[tid] = 0;
    }
    const bool valid = e < (uint32_t)n_experts;
    const uint32_t t = in_list ? (uint32_t)(tid / k) : 0u;
    const uint32_t key = valid ? ((e << 9) | (uint32_t)tid) : INVALID_ROW;
    if (warp < NARROW_WORDS) {
        // Every key slot up to the bound is written, so the P1 walk may read
        // whole 4-key groups past the list: a sentinel key is never below one.
        sh_key[tid] = key;
        const uint32_t vm = __ballot_sync(0xffffffffu, valid);
        if (lane == 0) {
            sh_vmask[warp] = vm;
        }
    }
    __syncthreads();

    // ── P1: representatives' live loads, rows, token-major slots ──
    const uint32_t total_valid = narrow_valid_before(sh_vmask, a_ub);
    if (valid) {
        const bool rep = atomicAdd(&sh_counts[e], 1) == 0;
        LiveEntries entries = {0ull, 0ull, 0ull};
        if (rep && live) {
            entries = load_live_entries(gate_row, table_plane, (int)e);
        }
        if (token_is_decode(t, decode)) {
            // Every writer stores the same 1.
            sh_dec[e] = 1;
        }
        // The keys below this one: the assignment's row.
        uint32_t r = 0;
        const uint4* keys4 = reinterpret_cast<const uint4*>(sh_key);
        const int n4 = (a_ub + 3) >> 2;
        for (int q = 0; q < n4; q++) {
            const uint4 v = keys4[q];
            r += (uint32_t)(v.x < key) + (uint32_t)(v.y < key) + (uint32_t)(v.z < key) +
                 (uint32_t)(v.w < key);
        }
        // This token's keys below this one: its slot among the token's pairs,
        // which are ordered by ascending row.
        const int t0 = (int)t * k;
        uint32_t slot = narrow_valid_before(sh_vmask, t0);
        for (int s = 0; s < k; s++) {
            slot += (uint32_t)(sh_key[t0 + s] < key);
        }
        tok_ids[r] = t;
        weight_ids[r] = (uint32_t)tid;
        perm[slot] = r;
        rw_ids[slot] = (uint32_t)tid;
        if (rep && live) {
            const uint8_t cls = snapshot_expert(
                (int)e, entries, n_experts, snap, pinned0_lo, pinned0_hi, pinned1_lo, pinned1_hi,
                slot_owner, zone_end, zone_slot_bytes, zone_slots, row);
            sh_cls[e] = cls;
            sh_marked[e] = promo_slots != nullptr && cls != CLS_VRAM &&
                           promotion_marked(promo_marks, row, n_experts, (int)e);
        }
    }
    if (in_list) {
        if (tid % k == 0) {
            token_starts[tid / k] = (int32_t)narrow_valid_before(sh_vmask, tid);
        }
        if ((uint32_t)tid >= total_valid) {
            tok_ids[tid] = INVALID_ROW;
            weight_ids[tid] = INVALID_ROW;
            perm[tid] = 0;
            rw_ids[tid] = 0;
        }
    }
    if (tid == 0) {
        token_starts[n_tokens] = (int32_t)total_valid;
        if (counters != nullptr) {
            counters[0] = 0;
            counters[1] = 0;
            counters[2] = 0;
        }
    }
    __syncthreads();

    // ── P2: offsets, tile order, remote list (one packed scan), summary ──
    const bool in = tid < n_experts;
    const int32_t cnt = in ? sh_counts[tid] : 0;
    const bool routed = cnt > 0;
    const uint8_t cls = routed && live ? sh_cls[tid] : (uint8_t)CLS_VRAM;
    const int rank = class_rank(cls);
    const int32_t n_tiles = (cnt + tile_w - 1) / tile_w;
    const bool is_remote = routed && cls != CLS_VRAM;
    const unsigned long long packed =
        ((unsigned long long)cnt << (NARROW_LANE_BITS * NL_OFF)) |
        ((unsigned long long)n_tiles << (NARROW_LANE_BITS * (NL_TILES + rank))) |
        ((unsigned long long)is_remote << (NARROW_LANE_BITS * (NL_REMOTE + (cls == CLS_COLD)))) |
        ((unsigned long long)routed << (NARROW_LANE_BITS * NL_ACTIVE));
    unsigned long long totals = 0ull;
    const unsigned long long before = block_exclusive_scan<unsigned long long>(packed, sh_scan, &totals);
    const int32_t t_pinned = (int32_t)narrow_lane(totals, NL_TILES + 0);
    const int32_t t_cold = (int32_t)narrow_lane(totals, NL_TILES + 1);
    const int32_t t_vram = (int32_t)narrow_lane(totals, NL_TILES + 2);
    const int32_t r_pinned = (int32_t)narrow_lane(totals, NL_REMOTE + 0);
    const int32_t r_cold = (int32_t)narrow_lane(totals, NL_REMOTE + 1);
    const int32_t num_tiles = t_pinned + t_cold + t_vram;
    if (routed) {
        const int32_t start = (int32_t)narrow_lane(before, NL_OFF);
        const int32_t class_base = rank == 0 ? 0 : (rank == 1 ? t_pinned : t_pinned + t_cold);
        const int32_t tile_pref = class_base + (int32_t)narrow_lane(before, NL_TILES + rank);
        for (int32_t q = 0; q < n_tiles; q++) {
            tile_expert[tile_pref + q] = tid;
            tile_b_start[tile_pref + q] = start + q * tile_w;
            const int32_t rem = cnt - q * tile_w;
            tile_b_cnt[tile_pref + q] = rem < tile_w ? rem : tile_w;
        }
        if (is_remote && remote != nullptr) {
            const int32_t at = cls == CLS_PINNED
                ? (int32_t)narrow_lane(before, NL_REMOTE + 0)
                : r_pinned + (int32_t)narrow_lane(before, NL_REMOTE + 1);
            remote[4 * at + 0] = tid;
            remote[4 * at + 1] = tile_pref;
            remote[4 * at + 2] = n_tiles;
            remote[4 * at + 3] = cls == CLS_COLD;
            sh_remote_e[at] = tid;
        }
    }
    if (tid >= num_tiles && tid < a_ub) {
        tile_expert[tid] = 0;
        tile_b_start[tid] = 0;
        tile_b_cnt[tid] = 0;
    }
    // The summary words last of the device's writes: issued ahead of the scan,
    // their mapped writes delayed it (decode-16, live: 6.7 → 7.4 µs).
    if (summary != nullptr && in) {
        summary[tid] = (uint32_t)cnt
                     | ((uint32_t)(cls == CLS_PINNED) << 29)
                     | ((uint32_t)(cls == CLS_COLD) << 30)
                     | ((uint32_t)sh_dec[tid] << 31);
    }
    if (tid == 0) {
        header[0] = (int32_t)narrow_lane(totals, NL_ACTIVE);
        header[1] = (int32_t)total_valid;
        header[2] = num_tiles;
        header[3] = remote != nullptr ? r_pinned + r_cold : 0;
        header[4] = t_pinned + t_cold;
    }

    // ── The promotion walk, then the summary's sequence word ──
    // With no remote expert routed and no read-ahead there is nothing to claim,
    // so the ring is not read and `head` is not republished.
    const int32_t n_remote = r_pinned + r_cold;
    const bool walk = remote != nullptr && remote_dst != nullptr &&
                      (n_remote > 0 || ahead_items != nullptr);
    if (!walk && summary == nullptr) {
        return;
    }
    // The walk reads the list the routed experts wrote; the sequence word must
    // follow every thread's summary word.
    __syncthreads();
    if (tid != 0) {
        return;
    }
    if (walk) {
        // The experts whose claims would evict: the decode-scored ones. A
        // prompt-only expert takes only an empty offer, so it evicts nothing
        // and decides nothing.
        int32_t claiming = 0;
        if (promo_slots != nullptr) {
            for (int32_t r = 0; r < n_remote; r++) {
                const int32_t x = sh_remote_e[r];
                claiming += !sh_marked[x] && sh_dec[x];
            }
        }
        promotion_walk(n_remote, claiming, sh_remote_e, sh_marked, sh_dec, sh_counts, gate_row,
                       table_plane, n_experts, row, summary_seq, promo_slots, promo_log, promo_head,
                       promo_tail, promo_cap, promo_marks, promo_reserve, promo_sweep, promo_victims,
                       promo_retarget, remote_dst,
                       ReadAhead{ahead_window, ahead_depth, ahead_n, ahead_list, ahead_src, ahead_cap,
                                 rows, row_layout, ahead_items, ahead_done},
                       ZoneRanges{pinned0_lo, pinned0_hi, pinned1_lo, pinned1_hi, slot_owner,
                                  zone_end, zone_slot_bytes, zone_slots});
    }
    // The sequence word is the host's signal: it lands only after every summary
    // word, the promotion log and its head are visible system-wide.
    if (summary != nullptr) {
        __threadfence_system();
        *(volatile uint32_t*)&summary[n_experts] = summary_seq;
    }
}
