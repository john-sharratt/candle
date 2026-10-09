// =============================================================================
// GPU MoE EXPERT BUCKETIZE
// =============================================================================
// The expert path's routing step, entirely on the device: the per-layer routing
// indices never leave the GPU. One launch turns `moe_route`'s top-k index tensor
// into every table the downstream GPU pipeline consumes — the expert-grouped
// assignment lists (gather), the tile tables (grouped GEMM), and the token-major
// segment tables (deterministic scatter) — and, over a live expert table, the
// weight snapshot, the remote list and the routing summary the host reads.
//
// The kernel is a SINGLE thread block of BUCKETIZE_THREADS (512) threads — one
// thread per expert up to MAX_EXPERTS (512), which every per-expert phase relies
// on: the offsets, the tile order and the remote list are block-wide scans over
// the expert axis, one value per thread.
//
// Two kernels, one contract. A launch of at most BUCKETIZE_NARROW (256)
// assignments — a decode, draft or verify step — runs
// `moe_bucketize_narrow_kernel` (`moe_bucketize_narrow.cuh`): one assignment
// per thread, five block barriers, the same bytes. At that width the phases
// below cost their ~22 barriers and serial tails, not their work (12 µs a
// launch at 50 assignments over 512 experts, RTX PRO 5000). Wider launches run
// the phases below. The
// host-memory protocol — the started word, the live-entry snapshot, the
// promotion walk — is one set of device functions both kernels call
// (`moe_bucketize_live.cuh`).
//
// Every output is bit-deterministic:
//   * phase 1 — a per-expert histogram: each assignment is read ONCE, and the
//     lanes of a warp routing one expert add their count to its shared bin
//     with one atomicAdd (an id ≥ n_experts is the router's "no expert"
//     sentinel and is skipped). The sums are order-independent, so the counts
//     — and every table derived from them — are identical to a serial scan;
//   * phase 2 — block-wide scans turn the counts into bucket offsets, the
//     per-expert tile counts into the tile order, and the routed remote experts
//     into the remote list; thread 0 then walks only that list for the
//     promotion ring, and writes the device header. A serial walk of all 512
//     experts here — four of them, for the offsets and the three tile classes —
//     was 100 µs of every 116 µs decode call;
//   * phase 3 — a STABLE counting-sort scatter, one contiguous chunk of the
//     list per warp: each warp counts its chunk per expert, a per-expert
//     prefix across the warps gives each its write bases, and each walks its
//     chunk again 32 assignments a step, placing same-expert lanes by
//     `__match_any_sync`. Chunks, steps and lanes all ascend in i, so each
//     bucket is STABLE (ascending i), exactly the buckets of a serial stable
//     counting sort — with every lane of every warp working, where the
//     chunk-per-thread form it replaced kept 16 threads busy at 512 experts,
//     ~1.5 ms a call at 8,192 rows;
//   * phase 4 — each thread emits its expert's GEMM tiles (≤ tile_w tokens
//     per tile); the tail up to the launch bound is padded with `b_cnt = 0`
//     tiles the grouped kernel skips, so the HOST needs no data-dependent
//     value for the GEMM grid — it launches at the `n_tokens × k` bound;
//   * phase 5 — a tiled block scan over the valid flags builds the
//     token-major compaction: `perm` (expert-grouped row of each valid
//     assignment), `reordered_weight_ids`, and `token_starts`, the exact
//     inputs of `deterministic_scatter_*`.
//
// `tile_expert` carries RAW expert ids into the layer's row of the LIVE expert
// table (`expert_lre::live_table`), whose entry for an expert is its address in
// VRAM, its address in pinned host memory (inside one of the two `pinned`
// ranges: the warm tier and the pad), or 0 — cold, still on the NVMe pack.
// Pinned and cold experts are REMOTE: the grouped GEMM's worker blocks copy them
// into VRAM and compute them. The GEMMs read the SNAPSHOT this kernel takes of
// the routed experts' entries (phase 1b), not the live row.
//
// Two orders, deliberately different:
//   * the ROW layout (tok_ids, weight_ids, perm, rw_ids, token_starts) is
//     ascending expert id, always — it decides the scatter's summation order,
//     so it must be a function of the routing alone;
//   * the TILE order puts remote experts' tiles first — pinned, then cold —
//     then the VRAM experts', each class in ascending id. The grouped GEMM's
//     blocks are independent and every output row has one writer, so tile
//     order changes when a row is computed, never its bits.
// The REMOTE LIST (`remote`, `[n_experts][4]`) names each routed remote expert
// in that order as `{expert, first_tile, n_tiles, cold}`; `header[3]` counts
// them and `header[4]` counts their tiles. `counters[0..3]` — the work counters
// of the layer's gate, up and down launches — are zeroed here.
//
// PROMOTION: each remote expert a decode-scored token routed, in list order,
// takes the next VRAM slot image the promotion ring offers while it has one
// (`remote_dst[i]`, else 0). A prompt-only expert takes one only while the ring
// holds more than its `reserve` (a mapped word the host sets; a null address
// means never). Where the zone has room for prompts the host sets it to 0 and a
// prompt takes any stocked slot; where it has not, the host sets it to the
// stock decode's misses want, so a prompt — which sweeps the table about once —
// is served by the workers' copies instead of evicting decode's working set.
// What stands above the reserve is the read-ahead window and the zone's empty
// slots: prompt-only experts and read-ahead (below) share it, and read-ahead
// never takes the stock below the reserve.
//
// A PROMPT-ONLY expert takes only an EMPTY offer — a prompt fills the zone's
// holes and never evicts. When the next offer holds a victim, it stays in its
// workers' scratch.
//
// A SWEEP claims nothing: when the experts whose claims would evict — the
// decode-scored ones — outnumber the mapped `sweep` word (the stock the host
// keeps for the layers ahead), the launch is a prompt passing over the table,
// and every miss stays in its workers' scratch. Decided per launch, before the
// first claim, so a launch either claims as its misses come or not at all.
//
// An offered slot is either EMPTY (`promo_victims[i] == PROMO_EMPTY`) or still
// holds a resident expert, its VICTIM, named by its index `row' · n_experts +
// e'` in the gate plane. A victim stays resident and hittable until a miss
// claims its slot; claiming it is the eviction:
//   * a victim of THIS row that this launch routes is SKIPPED — one of this
//     launch's tiles reads that slot. Its index is logged with the expert
//     `PROMO_SKIP` and the host takes the offer back; the miss tries the next.
//   * otherwise the victim's three live entries are retargeted to its pinned
//     copy (`promo_retarget[i]`: gate, up, down) — up and down, then gate —
//     before this launch's workers write a byte into the slot. Every launch
//     enqueued earlier has completed in stream order, and every later one reads
//     the entries fresh in its own bucketize, so no tile ever reads the slot
//     under the victim's name again. No fence per claim: nothing reads a VRAM
//     victim's entry concurrently, and the host sees the claim only through
//     `head`, published behind a system fence.
// The ring's log records `summary_seq << 32 | row << 16 | expert` for each slot
// taken. The expert's mark (`promo_marks[row][expert]`) is set with it and
// cleared by the host when the promotion lands; a marked expert is not given a
// second slot by a later invocation that finds it still remote. The grouped
// GEMMs' workers write every slice they copy into that slot, so once the layer
// is done the expert is whole in VRAM and the host points its entry there.
//
// READ-AHEAD (`ahead_items`, optional; `moe_read_ahead.cuh` is the item
// contract): after its own misses, a launch that claims at all spends what is
// left of the layer's link window on experts predicted for the rows after the
// next — `row + 2 … row + depth` (wrapping into the next pass), the host's
// prediction lists in `ahead_list[t][..ahead_n[t]]`, each expert with the slot
// image the host vetted as its source in `ahead_src[t][..]`. The budget is the
// mapped `window` word (slot images the link moves in one layer) less this
// launch's remote experts, at most AHEAD_MAX. A predicted expert is read ahead
// only while its entries still point at that vetted image, in pinned memory: a
// VRAM expert needs nothing, a cold one is the stager's to stage, and an image
// the host did not vet may be a pad slot nobody pinned — the stager may evict
// it while this launch reads it, since the pad's reuse rules guard the pad's
// own row's invocations, not this one. The host vets a warm slot as it is (it
// never changes) and a pad slot only once it has pinned it, and keeps the pin
// until every launch that could read the listing has finished. An unmarked
// vetted expert takes the next offer exactly as a miss does (a victim of this row that this
// launch routes is skipped; a claimed victim is retargeted first), is logged
// `summary_seq << 32 | t << 16 | AHEAD_FLAG | expert` and marked, and gets an
// item: the gate launch's workers copy its image into the slot and the last of
// them publishes its entries, so a later row's bucketize finds it in VRAM with
// no host step between. A row's prediction is a hint, never a correctness
// input: a wrong one costs a slot and the link time the budget allowed.
//
// OWNER CHECK (`slot_owner`, optional): the zone's slots, each tagged with the
// expert last installed there as `(row + 1) << 16 | expert`. Every VRAM entry
// snapshotted for a GEMM is checked against its slot's tag, and a mismatch —
// a tile about to read one expert's weights under another's name — traps,
// naming the slot. Null skips the check.
//
// The ROUTING SUMMARY (`summary`, optional) is what the host learns about the
// layer: `count | pinned << 29 | cold << 30 | decode << 31` per expert, where
// `decode` marks an expert some assignment of a decode-scored token routed to
// (one inside a `decode` range — the residency scoring weights such tokens'
// experts as a decode step's), then `summary_seq` in the word after them. The buffer is mapped host
// memory: the host polls the sequence word and reads the counts in place, with
// no copy and no event.
//
// Padding conventions consumed downstream:
//   tok_ids / weight_ids  : 0xFFFFFFFF  (gather skips the row)
//   tile_b_cnt            : 0           (grouped GEMM early-outs the block)
//   perm / rw_ids         : 0           (never referenced — token_starts
//                                        segments only cover valid rows)
// =============================================================================

#include "moe_bucketize_common.cuh"
#include "moe_bucketize_live.cuh"
#include "moe_bucketize_narrow.cuh"

// One thread per expert; phase 3 is one chunk per warp.
//
// The static shared-memory cost is `sh_cc` (32 KB, fixed) plus four int32
// arrays over the expert axis — 512·4·4 ≈ 8 KB — plus ~2 KB of scan and header,
// ≈ 42 KB against the 48 KiB static cap. Raising MAX_EXPERTS again means
// dynamic shared memory and a wider block, not a constant change.
// Phase-3 per-warp count table (ints): one count per warp per expert.
#define SH_CC_INTS 8192
// Block-wide tiles of the assignment list whose loads the walks issue together.
#define BUCKETIZE_PRELOAD 8
// The tile scan packs one 21-bit lane per class rank into a u64.
#define TILE_LANE_BITS 21
#define TILE_LANE_MASK ((1ull << TILE_LANE_BITS) - 1ull)

extern "C" __global__ void moe_bucketize_kernel(
    const uint32_t* __restrict__ topk_ids, // [n_tokens * k] row-major
    const int n_tokens,
    const int k,
    const int n_experts, // ≤ MAX_EXPERTS; id ≥ n_experts = sentinel
    const int tile_w,    // grouped-GEMM tile width (tokens per tile)
    uint32_t* __restrict__ tok_ids,      // [a_ub] expert-grouped token ids
    uint32_t* __restrict__ weight_ids,   // [a_ub] expert-grouped widx (= i)
    int32_t* __restrict__ tile_expert,   // [a_ub] RAW expert id per tile
    int32_t* __restrict__ tile_b_start,  // [a_ub]
    int32_t* __restrict__ tile_b_cnt,    // [a_ub]
    uint32_t* __restrict__ perm,         // [a_ub] token-major → grouped row
    uint32_t* __restrict__ rw_ids,       // [a_ub] token-major widx
    int32_t* __restrict__ token_starts,  // [n_tokens + 1]
    // [5]: n_active, total_valid, num_tiles, remote experts, remote tiles
    int32_t* __restrict__ header,
    uint32_t* __restrict__ inv,          // [a_ub] scratch: i → grouped row
    int32_t* __restrict__ scan,          // [a_ub] scratch: exclusive valid scan
    // This layer's row of the live gate table; null = every expert in VRAM
    // (no remote experts — plain ascending order). The up and down rows are
    // `table_plane` and `2 · table_plane` entries after it.
    const uint64_t* gate_row,
    const long long table_plane,
    // [3][n_experts] snapshot of the routed experts' entries (gate, up, down),
    // the layer's GEMM weight tables; required when `gate_row` is set.
    uint64_t* __restrict__ snap,
    // The two pinned host ranges `[lo, hi)` a remote entry lies in.
    const uint64_t pinned0_lo, const uint64_t pinned0_hi,
    const uint64_t pinned1_lo, const uint64_t pinned1_hi,
    const DecodeRanges decode,           // the tokens scored as decode rows
    uint32_t* __restrict__ summary,      // [n_experts + 1] routing summary, or null
    const uint32_t summary_seq,          // stored at summary[n_experts] last
    int32_t* __restrict__ remote,        // [n_experts][4] remote list, or null
    int32_t* __restrict__ counters,      // [3] launch work counters, or null
    // The promotion ring (mapped host memory), or null: `promo_slots[cap]`
    // VRAM slot images the host has freed, `promo_log[cap]` what each was
    // given to (`summary_seq << 32 | row << 16 | expert` — the sequence word
    // says which invocation must finish before the slot is whole), and the two
    // counters — `tail` the host's, `head` this kernel's.
    const uint64_t* promo_slots,
    uint64_t* promo_log,
    uint32_t* promo_head,
    const uint32_t* promo_tail,
    const uint32_t promo_cap,
    // `[rows][n_experts]` marks (mapped): non-zero while an expert's promotion
    // is in flight — set here when a slot is given, cleared by the host when
    // it lands — so a later invocation does not promote it a second time.
    uint32_t* promo_marks,
    // Mapped `u32`: the stock kept for decode-scored experts — a prompt-only
    // expert takes a slot only while more than this is stocked. Null = never.
    const uint32_t* promo_reserve,
    // Mapped `u32`: a launch with more claiming experts than this is a sweep
    // and claims nothing.
    const uint32_t* promo_sweep,
    // Mapped `[cap]`: the victim behind each offer (`row' · n_experts + e'`), or
    // PROMO_EMPTY; and `[cap][3]` the entries a claimed victim is retargeted to.
    const uint64_t* promo_victims,
    const uint64_t* promo_retarget,
    // Mapped `u32[zone_slots]` slot tags, or null: the owner check. Slot `s`
    // spans `[zone_end - (s + 1) · zone_slot_bytes, zone_end - s · zone_slot_bytes)`.
    const uint32_t* slot_owner,
    const uint64_t zone_end,
    const uint64_t zone_slot_bytes,
    const uint32_t zone_slots,
    const int32_t row,                   // this layer's row, for the log
    uint64_t* __restrict__ remote_dst,   // [n_experts] promotion slot per remote expert, or null
    // `[rows]` started words (mapped), or null: `ticket` is stored into
    // `started_rows[row]` before any live entry is read — the host's reclaim
    // rule pairs its retarget-then-read with this store-then-read.
    uint64_t* started_rows,
    const uint64_t ticket,
    // READ-AHEAD, or null `ahead_items` (requires the ring). Mapped: the
    // `window` and `depth` words and the per-row prediction lists
    // `ahead_n[rows]`, `ahead_list[rows][ahead_cap]` and the vetted source
    // images `ahead_src[rows][ahead_cap]`. Device: `row_layout`
    // `u64[rows][4]` (gate, up, down offset in a slot image, image bytes), the
    // item buffer and its per-item piece counters (`moe_read_ahead.cuh`).
    const uint32_t* ahead_window,
    const uint32_t* ahead_depth,
    const uint32_t* ahead_n,
    const uint32_t* ahead_list,
    const uint64_t* ahead_src,
    const uint32_t ahead_cap,
    const int32_t rows,
    const uint64_t* row_layout,
    uint64_t* ahead_items,
    uint32_t* ahead_done)
{
    const int tid = (int)threadIdx.x;
    const int a_ub = n_tokens * k;

    __shared__ int32_t sh_counts[MAX_EXPERTS];
    __shared__ int32_t sh_offsets[MAX_EXPERTS + 1];
    __shared__ int32_t sh_tile_pref[MAX_EXPERTS + 1];
    __shared__ int32_t sh_header[3]; // n_active, total_valid, num_tiles
    // Per-expert: where the gate entry points (CLS_VRAM / CLS_PINNED / CLS_COLD),
    // and whether a decode row routed here.
    __shared__ uint8_t sh_cls[MAX_EXPERTS];
    __shared__ uint8_t sh_dec[MAX_EXPERTS];
    // Per routed remote expert: a promotion of it is already in flight.
    __shared__ uint8_t sh_marked[MAX_EXPERTS];
    // The remote list's experts, in list order, for the promotion walk.
    __shared__ int32_t sh_remote_e[MAX_EXPERTS];
    // One buffer per block-wide scan, so no scan waits on the one before it.
    __shared__ uint32_t sh_scan_off[BUCKETIZE_WARPS + 1];
    __shared__ unsigned long long sh_scan_tiles[BUCKETIZE_WARPS + 1];
    __shared__ unsigned long long sh_scan_remote[BUCKETIZE_WARPS + 1];
    __shared__ int32_t sh_scan_valid[BUCKETIZE_WARPS + 1];
    // Per-warp per-expert counts for the phase-3 stable scatter, flat
    // `[BUCKETIZE_WARPS][n_experts]` with a runtime `n_experts` stride.
    __shared__ int32_t sh_cc[SH_CC_INTS];

    // ── Phase 1: per-expert histogram (grid-stride, shared-memory atomics) ──
    // Each assignment is read ONCE and atomically bumped into its expert's
    // shared bin — O(a_ub) work instead of the O(n_experts × a_ub) scan where
    // every thread re-read the whole list. Summation is order-independent, so
    // the counts (and therefore every downstream table) are identical to the
    // serial scan: the change is bit-exact, just far faster for large prefill
    // `a_ub` (the single-block kernel's dominant cost).
    //
    for (int e = tid; e < n_experts; e += BUCKETIZE_THREADS) {
        sh_counts[e] = 0;
        sh_dec[e] = 0;
        sh_cls[e] = CLS_VRAM;
        sh_marked[e] = 0;
    }
    __syncthreads();
    // The loop bound is block-uniform so every lane of every warp reaches the
    // match below; a lane past the list carries the invalid key.
    const int lane = tid & 31;
    // BUCKETIZE_PRELOAD tiles' loads are issued together before any is used,
    // so the walk waits on global memory once per group rather than per tile.
    for (int g0 = 0; g0 < a_ub; g0 += BUCKETIZE_THREADS * BUCKETIZE_PRELOAD) {
        uint32_t ids[BUCKETIZE_PRELOAD];
        #pragma unroll
        for (int u = 0; u < BUCKETIZE_PRELOAD; ++u) {
            const int i = g0 + u * BUCKETIZE_THREADS + tid;
            ids[u] = i < a_ub ? topk_ids[i] : INVALID_ROW;
        }
        #pragma unroll
        for (int u = 0; u < BUCKETIZE_PRELOAD; ++u) {
            const int i = g0 + u * BUCKETIZE_THREADS + tid;
            const uint32_t e = ids[u];
            const bool valid = e < (uint32_t)n_experts;
            // Lanes routing the same expert add their count once, from the
            // lowest of them: a wide prefill's thousands of assignments per
            // expert would otherwise all serialise on one shared bin.
            const uint32_t peers = __match_any_sync(0xffffffffu, valid ? e : INVALID_ROW);
            if (!valid) {
                continue;
            }
            if (lane == __ffs(peers) - 1) {
                atomicAdd(&sh_counts[e], __popc(peers));
            }
            // Every writer stores the same 1, so the race is benign. An expert
            // already marked needs no range scan — a stale 0 read only costs
            // one — which keeps a wide prefill's many assignments to the same
            // experts off the per-range loop.
            if (!sh_dec[e]) {
                const uint32_t t = (uint32_t)(i / k);
                for (uint32_t r = 0; r < decode.n; r++) {
                    if (t >= decode.lo[r] && t < decode.hi[r]) {
                        sh_dec[e] = 1;
                        break;
                    }
                }
            }
        }
    }
    __syncthreads();
    // ── Phase 1b: classify every routed expert, and snapshot its entries ──
    // The live entries are read once per routed expert, volatile: the host
    // rewrites them while kernels run. What was read is copied into `snap`
    // (`[3][n_experts]`: gate, up, down), and the layer's GEMMs read their
    // weights from the snapshot, never from the live table — so an entry the
    // host retargets after this point cannot reach a block that was ordered by
    // the old value. (Every address read stays valid until the host has seen a
    // later layer's summary word: `expert_lre`'s reclaim rule.) A cold expert —
    // any of its three entries 0 — is the one exception: its workers wait on
    // the live gate entry and read the live up / down entries, which the host
    // publishes before it and never clears while this layer can read them.
    //
    // Before the first read, this invocation's ticket goes into its row's
    // started word, and the fence orders it ahead of every entry load. The host
    // retargets an entry, fences, then reads the word: so either it sees this
    // ticket and holds the old slot until this invocation is done, or the loads
    // below see the retargeted entry. That is what lets the host key slot reuse
    // on the invocation the device is inside, not on what is queued behind it.
    //
    // Thread 0 alone fences, then the barrier: the store is visible
    // system-wide before any thread's first entry load issues, which is all
    // the pairing needs — sixteen warps each draining a PCIe write behind the
    // same fence bought nothing.
    if (gate_row != nullptr && started_rows != nullptr) {
        if (tid == 0) {
            store_started_ticket(started_rows, row, ticket);
        }
        __syncthreads();
    }
    if (gate_row != nullptr) {
        for (int e = tid; e < n_experts; e += BUCKETIZE_THREADS) {
            if (sh_counts[e] == 0) {
                continue;
            }
            const LiveEntries entries = load_live_entries(gate_row, table_plane, e);
            const uint8_t cls = snapshot_expert(
                e, entries, n_experts, snap, pinned0_lo, pinned0_hi, pinned1_lo, pinned1_hi,
                slot_owner, zone_end, zone_slot_bytes, zone_slots, row);
            sh_cls[e] = cls;
            if (promo_slots != nullptr && cls != CLS_VRAM) {
                sh_marked[e] = promotion_marked(promo_marks, row, n_experts, e);
            }
        }
        __syncthreads();
    }
    if (summary != nullptr) {
        for (int e = tid; e < n_experts; e += BUCKETIZE_THREADS) {
            summary[e] = (uint32_t)sh_counts[e]
                       | ((uint32_t)(sh_cls[e] == CLS_PINNED) << 29)
                       | ((uint32_t)(sh_cls[e] == CLS_COLD) << 30)
                       | ((uint32_t)sh_dec[e] << 31);
        }
    }

    // ── Phase 2: offsets + tile order + remote list (block scans), then the
    // promotion walk and the header (thread 0) ──
    // Offsets ascend by expert id (the row layout); the tile prefix runs over
    // the pinned experts, then the cold, then the VRAM ones (the tile order);
    // the remote list names every routed remote expert in tile order. Each is a
    // block-wide exclusive scan over the expert axis, thread `e` holding expert
    // `e`: the tile order is ONE scan whose three 21-bit lanes are the classes'
    // tile counts, and the remote list one scan whose two 32-bit lanes are the
    // pinned and cold experts' flags.
    {
        const int e = tid;
        const bool in = e < n_experts;
        const int32_t cnt = in ? sh_counts[e] : 0;
        const int32_t n_tiles = (cnt + tile_w - 1) / tile_w;
        const uint8_t cls = in ? sh_cls[e] : (uint8_t)CLS_VRAM;
        const int rank = class_rank(cls);

        uint32_t total_valid_u = 0;
        const uint32_t off = block_exclusive_scan<uint32_t>((uint32_t)cnt, sh_scan_off, &total_valid_u);

        unsigned long long tile_totals = 0ull;
        const unsigned long long tile_before = block_exclusive_scan<unsigned long long>(
            (unsigned long long)n_tiles << (TILE_LANE_BITS * rank), sh_scan_tiles, &tile_totals);
        const int32_t t_pinned = (int32_t)(tile_totals & TILE_LANE_MASK);
        const int32_t t_cold = (int32_t)((tile_totals >> TILE_LANE_BITS) & TILE_LANE_MASK);
        const int32_t t_vram = (int32_t)((tile_totals >> (2 * TILE_LANE_BITS)) & TILE_LANE_MASK);
        const int32_t class_base = rank == 0 ? 0 : (rank == 1 ? t_pinned : t_pinned + t_cold);
        const int32_t tile_pref =
            class_base + (int32_t)((tile_before >> (TILE_LANE_BITS * rank)) & TILE_LANE_MASK);

        const bool is_remote = in && cls != CLS_VRAM && n_tiles > 0;
        unsigned long long remote_totals = 0ull;
        const unsigned long long remote_before = block_exclusive_scan<unsigned long long>(
            is_remote ? (1ull << (32 * (cls == CLS_COLD))) : 0ull, sh_scan_remote, &remote_totals);
        const int32_t r_pinned = (int32_t)(remote_totals & 0xffffffffull);
        const int32_t r_cold = (int32_t)(remote_totals >> 32);

        const int32_t active = __syncthreads_count(in && cnt > 0);
        // The experts whose claims would evict: the decode-scored ones. A
        // prompt-only expert takes only an empty offer, so it evicts nothing
        // and decides nothing — counting it would let a zone's holes turn a
        // short prompt into a sweep and defer its decode rows' claims.
        const int32_t claiming = __syncthreads_count(
            promo_slots != nullptr && in && cnt > 0 && cls != CLS_VRAM && !sh_marked[e] && sh_dec[e]);

        if (in) {
            sh_offsets[e] = (int32_t)off;
            sh_tile_pref[e] = tile_pref;
        }
        if (is_remote && remote != nullptr) {
            const int32_t at = cls == CLS_PINNED
                ? (int32_t)(remote_before & 0xffffffffull)
                : r_pinned + (int32_t)(remote_before >> 32);
            remote[4 * at + 0] = e;
            remote[4 * at + 1] = tile_pref;
            remote[4 * at + 2] = n_tiles;
            remote[4 * at + 3] = cls == CLS_COLD;
            sh_remote_e[at] = e;
        }
        if (tid == 0) {
            const int32_t total_valid = (int32_t)total_valid_u;
            const int32_t tiles = t_pinned + t_cold + t_vram;
            sh_offsets[n_experts] = total_valid;
            sh_tile_pref[n_experts] = tiles;
            sh_header[0] = active;
            sh_header[1] = total_valid;
            sh_header[2] = tiles;
            header[0] = active;
            header[1] = total_valid;
            header[2] = tiles;
            header[3] = remote != nullptr ? r_pinned + r_cold : 0;
            header[4] = t_pinned + t_cold;
            if (counters != nullptr) {
                counters[0] = 0;
                counters[1] = 0;
                counters[2] = 0;
            }
        }
        __syncthreads();
        // The promotion walk: in list order, each remote expert takes the next
        // slot while the ring has one. Serial because each grant moves the head
        // the next one is judged against — but over the remote list alone, which
        // a decode step whose experts are all resident leaves empty. With no
        // remote expert and no read-ahead there is nothing to claim, so the ring
        // is not read and `head` is not republished: its two system fences and
        // the round of mapped loads are a decode step's most common avoidable
        // wait. Read-ahead runs even with every routed expert resident — that
        // is when the layer's link window is free.
        if (tid == 0 && remote != nullptr && remote_dst != nullptr &&
            (r_pinned + r_cold > 0 || ahead_items != nullptr)) {
            promotion_walk(r_pinned + r_cold, claiming, sh_remote_e, sh_marked, sh_dec, sh_counts,
                           gate_row, table_plane, n_experts, row, summary_seq, promo_slots,
                           promo_log, promo_head, promo_tail, promo_cap, promo_marks,
                           promo_reserve, promo_sweep, promo_victims, promo_retarget, remote_dst,
                           ReadAhead{ahead_window, ahead_depth, ahead_n, ahead_list, ahead_src,
                                     ahead_cap, rows, row_layout, ahead_items, ahead_done},
                           ZoneRanges{pinned0_lo, pinned0_hi, pinned1_lo, pinned1_hi, slot_owner,
                                      zone_end, zone_slot_bytes, zone_slots});
        }
    }
    __syncthreads();
    // The sequence word is the host's signal: it polls this word, so it lands
    // only after every summary word, the promotion log and its head are
    // visible system-wide.
    if (summary != nullptr && tid == 0) {
        __threadfence_system();
        *(volatile uint32_t*)&summary[n_experts] = summary_seq;
    }

    const int32_t total_valid = sh_header[1];
    const int32_t num_tiles = sh_header[2];

    // ── Phase 3: STABLE counting-sort scatter, a contiguous chunk per warp ──
    // The list is cut into BUCKETIZE_WARPS contiguous chunks, warp w owning
    // chunk w. Pass 1: each warp counts its chunk's assignments per expert into
    // its row of `sh_cc` (`[BUCKETIZE_WARPS][n_experts]`). A per-expert
    // exclusive prefix across the warps, seeded at the bucket's offset, turns
    // those counts into each warp's write base for each expert. Pass 2: each
    // warp walks its chunk again, 32 assignments a step in ascending i; an
    // assignment's slot is its warp's base plus the same-expert lanes below it
    // in the step (`__match_any_sync`), and the lowest such lane advances the
    // base. Chunks are ordered, steps ascend within a chunk and lanes within a
    // step, so every bucket comes out in ascending i — the buckets of a serial
    // stable counting sort, bit for bit. Each warp's walk needs no block
    // barrier: only its own row of bases moves.
    static_assert(BUCKETIZE_WARPS * MAX_EXPERTS <= SH_CC_INTS,
                  "phase 3 keeps one count per warp per expert in sh_cc");
    for (int idx = tid; idx < BUCKETIZE_WARPS * n_experts; idx += BUCKETIZE_THREADS) {
        sh_cc[idx] = 0;
    }
    __syncthreads();
    {
        const int warp = tid >> 5;
        const uint32_t lanes_below = (1u << lane) - 1u;
        // Whole 32-wide steps per chunk, so every lane of a warp runs every step.
        const int chunk = ((a_ub + BUCKETIZE_WARPS - 1) / BUCKETIZE_WARPS + 31) & ~31;
        const int c_lo = warp * chunk;
        const int c_hi = min(a_ub, c_lo + chunk);
        int32_t* const base = &sh_cc[warp * n_experts];
        constexpr int STEP_LOADS = BUCKETIZE_PRELOAD;
        // Pass 1: the warp's per-expert counts.
        for (int s0 = c_lo; s0 < c_hi; s0 += 32 * STEP_LOADS) {
            uint32_t ids[STEP_LOADS];
            #pragma unroll
            for (int u = 0; u < STEP_LOADS; ++u) {
                const int i = s0 + 32 * u + lane;
                ids[u] = i < c_hi ? topk_ids[i] : INVALID_ROW;
            }
            #pragma unroll
            for (int u = 0; u < STEP_LOADS; ++u) {
                const uint32_t e = ids[u];
                const bool valid = e < (uint32_t)n_experts;
                const uint32_t peers = __match_any_sync(0xffffffffu, valid ? e : INVALID_ROW);
                if (valid && lane == __ffs(peers) - 1) {
                    base[e] += __popc(peers);
                }
                __syncwarp();
            }
        }
        __syncthreads();
        // Per-expert exclusive prefix across the warps, from the bucket offset.
        for (int e = tid; e < n_experts; e += BUCKETIZE_THREADS) {
            int32_t run = sh_offsets[e];
            for (int w = 0; w < BUCKETIZE_WARPS; w++) {
                const int idx = w * n_experts + e;
                const int32_t v = sh_cc[idx];
                sh_cc[idx] = run;
                run += v;
            }
        }
        __syncthreads();
        // Pass 2: the stable scatter, from the warp's own bases.
        for (int s0 = c_lo; s0 < c_hi; s0 += 32 * STEP_LOADS) {
            uint32_t ids[STEP_LOADS];
            #pragma unroll
            for (int u = 0; u < STEP_LOADS; ++u) {
                const int i = s0 + 32 * u + lane;
                ids[u] = i < c_hi ? topk_ids[i] : INVALID_ROW;
            }
            #pragma unroll
            for (int u = 0; u < STEP_LOADS; ++u) {
                const int i = s0 + 32 * u + lane;
                const uint32_t e = ids[u];
                const bool valid = e < (uint32_t)n_experts;
                const uint32_t peers = __match_any_sync(0xffffffffu, valid ? e : INVALID_ROW);
                if (valid) {
                    const int32_t pos = base[e] + __popc(peers & lanes_below);
                    tok_ids[pos] = (uint32_t)(i / k);
                    weight_ids[pos] = (uint32_t)i;
                    inv[i] = (uint32_t)pos;
                }
                // Every lane has read this step's bases before the leaders move them.
                __syncwarp();
                if (valid && lane == __ffs(peers) - 1) {
                    base[e] += __popc(peers);
                }
                __syncwarp();
            }
        }
    }
    __syncthreads();

    // ── Phase 4: tile tables + padding ──
    for (int e = tid; e < n_experts; e += BUCKETIZE_THREADS) {
        const int32_t count = sh_counts[e];
        if (count <= 0) {
            continue;
        }
        const int32_t base = sh_tile_pref[e];
        const int32_t start = sh_offsets[e];
        const int32_t n_my_tiles = (count + tile_w - 1) / tile_w;
        for (int t = 0; t < n_my_tiles; t++) {
            tile_expert[base + t] = e;
            tile_b_start[base + t] = start + t * tile_w;
            const int32_t rem = count - t * tile_w;
            tile_b_cnt[base + t] = rem < tile_w ? rem : tile_w;
        }
    }
    for (int t = num_tiles + tid; t < a_ub; t += BUCKETIZE_THREADS) {
        tile_expert[t] = 0;
        tile_b_start[t] = 0;
        tile_b_cnt[t] = 0;
    }
    for (int i = total_valid + tid; i < a_ub; i += BUCKETIZE_THREADS) {
        tok_ids[i] = INVALID_ROW;
        weight_ids[i] = INVALID_ROW;
    }
    __syncthreads();

    // ── Phase 5: token-major compaction (exclusive scan of valid) ──
    // 5a: the exclusive scan of the valid flags, a block-wide group at a time:
    // each thread takes BUCKETIZE_PRELOAD consecutive assignments, the group's
    // per-thread counts are one block scan, and the running total carries into
    // the next group.
    {
        int32_t carried = 0;
        for (int g0 = 0; g0 < a_ub; g0 += BUCKETIZE_THREADS * BUCKETIZE_PRELOAD) {
            const int i_lo = g0 + tid * BUCKETIZE_PRELOAD;
            uint32_t flags = 0u;
            int32_t local = 0;
            #pragma unroll
            for (int u = 0; u < BUCKETIZE_PRELOAD; ++u) {
                const int i = i_lo + u;
                if (i < a_ub && topk_ids[i] < (uint32_t)n_experts) {
                    flags |= 1u << u;
                    local++;
                }
            }
            int32_t group_total = 0;
            int32_t run = carried + block_exclusive_scan<int32_t>(local, sh_scan_valid, &group_total);
            #pragma unroll
            for (int u = 0; u < BUCKETIZE_PRELOAD; ++u) {
                const int i = i_lo + u;
                if (i < a_ub) {
                    scan[i] = run;
                    run += (int32_t)((flags >> u) & 1u);
                }
            }
            carried += group_total;
            // The next group's scan rewrites the buffer this one's total was read from.
            __syncthreads();
        }
    }
    // 5d: per-token compaction + segment boundaries + padding. Within a token
    // the (perm, rw_ids) pairs are ordered by ASCENDING expert-grouped row —
    // the scatter accumulates each token's contributions sequentially in perm
    // order, which is `sort_by_key((token_id, row))` exactly, so the
    // float-summation order (and therefore every output bit) is a function of
    // the routing alone. k is small (≤ MAX_TOPK), so each
    // token sorts its pairs with an in-register insertion sort — deterministic,
    // one thread per token.
    for (int t = tid; t < n_tokens; t += BUCKETIZE_THREADS) {
        uint32_t rows[MAX_TOPK];
        uint32_t wids[MAX_TOPK];
        int n_valid = 0;
        for (int s = 0; s < k; s++) {
            const int i = t * k + s;
            if (topk_ids[i] < (uint32_t)n_experts) {
                const uint32_t r = inv[i];
                // Insertion sort by grouped row, ascending.
                int p = n_valid;
                while (p > 0 && rows[p - 1] > r) {
                    rows[p] = rows[p - 1];
                    wids[p] = wids[p - 1];
                    p--;
                }
                rows[p] = r;
                wids[p] = (uint32_t)i;
                n_valid++;
            }
        }
        const int32_t base = scan[t * k];
        for (int s = 0; s < n_valid; s++) {
            perm[base + s] = rows[s];
            rw_ids[base + s] = wids[s];
        }
    }
    for (int t = tid; t <= n_tokens; t += BUCKETIZE_THREADS) {
        token_starts[t] = t < n_tokens ? scan[t * k] : total_valid;
    }
    for (int j = total_valid + tid; j < a_ub; j += BUCKETIZE_THREADS) {
        perm[j] = 0;
        rw_ids[j] = 0;
    }
}

// Single-block launch: one BUCKETIZE_THREADS-wide block over ≤ MAX_EXPERTS
// experts. `stream` is the caller's launch stream — the compute stream, or the
// capture stream while the forward is recorded — so the outputs are ordered
// after the router's writes with no host synchronisation.
//
// Returns 0 when the kernel was launched, 1 when an argument guard refused it
// (nothing written: the workspace still holds the previous layer's tables), 2
// when the launch itself returned an error, and 3 when an earlier launch on
// this thread had left an error pending (nothing launched).
extern "C" int32_t run_moe_bucketize(
    const void* topk_ids,
    int32_t n_tokens,
    int32_t k,
    int32_t n_experts,
    int32_t tile_w,
    void* tok_ids,
    void* weight_ids,
    void* tile_expert,
    void* tile_b_start,
    void* tile_b_cnt,
    void* perm,
    void* rw_ids,
    void* token_starts,
    void* header,
    void* inv,
    void* scan,
    const void* gate_row,
    int64_t table_plane,
    void* snap,
    uint64_t pinned0_lo,
    uint64_t pinned0_hi,
    uint64_t pinned1_lo,
    uint64_t pinned1_hi,
    const uint32_t* decode_lo,
    const uint32_t* decode_hi,
    int32_t decode_ranges,
    void* summary,
    uint32_t summary_seq,
    void* remote,
    void* counters,
    const void* promo_slots,
    void* promo_log,
    void* promo_head,
    const void* promo_tail,
    uint32_t promo_cap,
    void* promo_marks,
    const void* promo_reserve,
    const void* promo_sweep,
    const void* promo_victims,
    const void* promo_retarget,
    const void* slot_owner,
    uint64_t zone_end,
    uint64_t zone_slot_bytes,
    uint32_t zone_slots,
    int32_t row,
    void* remote_dst,
    void* started_rows,
    uint64_t ticket,
    const void* ahead_window,
    const void* ahead_depth,
    const void* ahead_n,
    const void* ahead_list,
    const void* ahead_src,
    uint32_t ahead_cap,
    int32_t rows,
    const void* row_layout,
    void* ahead_items,
    void* ahead_done,
    void* stream)
{
    if (ahead_items != nullptr &&
        (promo_slots == nullptr || remote == nullptr || ahead_window == nullptr ||
         ahead_depth == nullptr || ahead_n == nullptr || ahead_list == nullptr ||
         ahead_src == nullptr || ahead_cap == 0 ||
         rows <= 0 || row >= rows || row_layout == nullptr || ahead_done == nullptr)) {
        return 1;
    }
    if (n_tokens <= 0 || k <= 0 || k > MAX_TOPK || n_experts <= 0 ||
        n_experts > MAX_EXPERTS || tile_w <= 0 || (gate_row != nullptr && snap == nullptr) ||
        (promo_slots != nullptr && (promo_cap == 0 || promo_log == nullptr ||
                                    promo_head == nullptr || promo_tail == nullptr ||
                                    promo_marks == nullptr || remote_dst == nullptr ||
                                    promo_sweep == nullptr || promo_victims == nullptr ||
                                    promo_retarget == nullptr || gate_row == nullptr)) ||
        (slot_owner != nullptr && (zone_slot_bytes == 0 || gate_row == nullptr)) ||
        decode_ranges < 0 || decode_ranges > MAX_DECODE_RANGES ||
        (decode_ranges > 0 && (decode_lo == nullptr || decode_hi == nullptr))) {
        return 1;
    }
    DecodeRanges decode;
    decode.n = (uint32_t)decode_ranges;
    for (int r = 0; r < MAX_DECODE_RANGES; r++) {
        decode.lo[r] = r < decode_ranges ? decode_lo[r] : 0u;
        decode.hi[r] = r < decode_ranges ? decode_hi[r] : 0u;
    }
    // An error a PRIOR launch left on this thread is reported, not cleared:
    // most launchers return nothing, so this checked launch is where a node
    // dropped from the segment before it surfaces. Reading it also clears it,
    // so the check below reports this launch and nothing else.
    cudaError_t earlier = cudaGetLastError();
    if (earlier != cudaSuccess) {
        fprintf(stderr, "moe_bucketize: an earlier launch failed: %s\n", cudaGetErrorString(earlier));
        return 3;
    }
    // A narrow launch — a decode, draft or verify step — takes the narrow
    // kernel, which writes the same bytes in a few barriers; anything wider
    // the general one.
    const bool narrow = (int64_t)n_tokens * (int64_t)k <= BUCKETIZE_NARROW;
    void (*const kernel)(MOE_BUCKETIZE_KERNEL_PARAMS) =
        narrow ? moe_bucketize_narrow_kernel : moe_bucketize_kernel;
    kernel<<<1, BUCKETIZE_THREADS, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)topk_ids, n_tokens, k, n_experts, tile_w,
        (uint32_t*)tok_ids, (uint32_t*)weight_ids, (int32_t*)tile_expert,
        (int32_t*)tile_b_start, (int32_t*)tile_b_cnt, (uint32_t*)perm,
        (uint32_t*)rw_ids, (int32_t*)token_starts, (int32_t*)header,
        (uint32_t*)inv, (int32_t*)scan, (const uint64_t*)gate_row,
        (long long)table_plane, (uint64_t*)snap, pinned0_lo, pinned0_hi, pinned1_lo, pinned1_hi,
        decode, (uint32_t*)summary, summary_seq,
        (int32_t*)remote, (int32_t*)counters,
        (const uint64_t*)promo_slots, (uint64_t*)promo_log, (uint32_t*)promo_head,
        (const uint32_t*)promo_tail, promo_cap, (uint32_t*)promo_marks,
        (const uint32_t*)promo_reserve, (const uint32_t*)promo_sweep,
        (const uint64_t*)promo_victims, (const uint64_t*)promo_retarget,
        (const uint32_t*)slot_owner, zone_end, zone_slot_bytes, zone_slots, row,
        (uint64_t*)remote_dst, (uint64_t*)started_rows, ticket,
        (const uint32_t*)ahead_window, (const uint32_t*)ahead_depth, (const uint32_t*)ahead_n,
        (const uint32_t*)ahead_list, (const uint64_t*)ahead_src, ahead_cap, rows,
        (const uint64_t*)row_layout,
        (uint64_t*)ahead_items, (uint32_t*)ahead_done);
    cudaError_t launched = cudaPeekAtLastError();
    if (launched != cudaSuccess) {
        fprintf(stderr, "moe_bucketize: launch failed: %s\n", cudaGetErrorString(launched));
        return 2;
    }
    return 0;
}
