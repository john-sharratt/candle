// =============================================================================
// MoE BUCKETIZE — the live expert table and the promotion ring
// =============================================================================
// The host-memory protocol both bucketize kernels run, so the two cannot
// diverge on it: the started word, the read-classify-snapshot of a routed
// expert's live entries, and the promotion walk. The orderings each step keeps
// are documented in `moe_bucketize.cu`'s header and at each fence below.
// =============================================================================
#pragma once

#include "moe_bucketize_common.cuh"
#include "../moe_read_ahead.cuh"

// Store this invocation's ticket into its row's started word, fenced: the store
// is visible system-wide before the caller's block barrier releases the first
// live-entry load. The host retargets an entry, fences, then reads the word —
// so either it sees this ticket and holds the old slot until this invocation is
// done, or the loads see the retargeted entry. Thread 0 alone calls it.
__device__ __forceinline__ void store_started_ticket(uint64_t* started_rows, int32_t row, uint64_t ticket) {
    ((volatile uint64_t*)started_rows)[row] = ticket;
    __threadfence_system();
}

// The three live entries (gate, up, down) of expert `e`, read volatile: the
// host rewrites them while kernels run.
struct LiveEntries {
    uint64_t gate;
    uint64_t up;
    uint64_t down;
};

__device__ __forceinline__ LiveEntries load_live_entries(const uint64_t* gate_row, long long table_plane, int e) {
    const volatile uint64_t* g = (const volatile uint64_t*)gate_row;
    LiveEntries r;
    r.gate = g[e];
    r.up = g[table_plane + e];
    r.down = g[2 * table_plane + e];
    return r;
}

// Classify expert `e` from the entries read — any of the three 0 is cold, a gate
// entry inside either pinned range is pinned, anything else VRAM — check a VRAM
// entry against its slot's owner tag (a mismatch traps, naming the slot), and
// copy the entries into the snapshot (`[3][n_experts]`). Returns the class.
__device__ __forceinline__ uint8_t snapshot_expert(
    int e, const LiveEntries& x, int n_experts, uint64_t* snap,
    uint64_t pinned0_lo, uint64_t pinned0_hi, uint64_t pinned1_lo, uint64_t pinned1_hi,
    const uint32_t* slot_owner, uint64_t zone_end, uint64_t zone_slot_bytes, uint32_t zone_slots,
    int32_t row)
{
    const uint64_t pg = x.gate;
    uint8_t cls = CLS_VRAM;
    if (pg == 0ull || x.up == 0ull || x.down == 0ull) {
        cls = CLS_COLD;
    } else if ((pg >= pinned0_lo && pg < pinned0_hi) || (pg >= pinned1_lo && pg < pinned1_hi)) {
        cls = CLS_PINNED;
    }
    if (slot_owner != nullptr && cls == CLS_VRAM && pg < zone_end &&
        pg >= zone_end - (uint64_t)zone_slots * zone_slot_bytes) {
        const uint32_t s = (uint32_t)((zone_end - 1ull - pg) / zone_slot_bytes);
        const uint32_t want = ((uint32_t)(row + 1) << 16) | (uint32_t)e;
        const uint32_t got = ((const volatile uint32_t*)slot_owner)[s];
        if (got != want) {
            printf("moe_bucketize: row %d expert %d reads slot %u, whose tenant is "
                   "row %d expert %u\n",
                   row, e, s, (int)(got >> 16) - 1, got & 0xffffu);
            __trap();
        }
    }
    snap[e] = pg;
    snap[n_experts + e] = x.up;
    snap[2 * n_experts + e] = x.down;
    return cls;
}

// Whether a promotion of remote expert `e` of `row` is already in flight.
__device__ __forceinline__ bool promotion_marked(const uint32_t* promo_marks, int32_t row, int n_experts, int e) {
    return ((const volatile uint32_t*)promo_marks)[(size_t)row * n_experts + e] != 0u;
}

// Take the next promotion offer at `*head` for an expert the log will name as
// `log_row` / `log_expert`, or return 0 when there is none it may take: the
// ring is empty, or the next offer holds a victim and `may_evict` is false. A
// victim of this launch's row that this launch routes (`routed[e] > 0`) is
// skipped — logged with PROMO_SKIP and passed over; any other victim's three
// entries are retargeted to its fallback (up and down, then gate) before the
// slot is handed out. No fence here: every later kernel sees these stores by
// stream order, nothing polls a VRAM victim's entry concurrently, and the host
// learns of the claim only through `head`, published behind a system fence by
// the walk.
__device__ __forceinline__ uint64_t take_offer(
    uint32_t* head, const uint32_t tail, const bool may_evict,
    const uint64_t* promo_slots, uint64_t* promo_log, const uint32_t promo_cap,
    const uint64_t* promo_victims, const uint64_t* promo_retarget,
    uint64_t* gate_plane, const long long table_plane, const int n_experts,
    const int32_t row, const int32_t* routed, const uint32_t summary_seq,
    const uint32_t log_row, const uint32_t log_expert)
{
    while (*head != tail) {
        const uint32_t i = *head % promo_cap;
        const uint64_t victim = ((const volatile uint64_t*)promo_victims)[i];
        if (victim != PROMO_EMPTY && !may_evict) {
            return 0ull;
        }
        if (victim != PROMO_EMPTY) {
            const int32_t vr = (int32_t)(victim / (uint64_t)n_experts);
            const int32_t ve = (int32_t)(victim % (uint64_t)n_experts);
            if (vr == row && routed[ve] > 0) {
                ((volatile uint64_t*)promo_log)[i] =
                    ((uint64_t)summary_seq << 32) | ((uint64_t)row << 16) | (uint64_t)PROMO_SKIP;
                (*head)++;
                continue;
            }
            const volatile uint64_t* rt = (const volatile uint64_t*)promo_retarget + 3 * (size_t)i;
            volatile uint64_t* g = (volatile uint64_t*)gate_plane + victim;
            g[table_plane] = rt[1];
            g[2 * table_plane] = rt[2];
            g[0] = rt[0];
        }
        const uint64_t dst = ((const volatile uint64_t*)promo_slots)[i];
        ((volatile uint64_t*)promo_log)[i] =
            ((uint64_t)summary_seq << 32) | ((uint64_t)log_row << 16) | (uint64_t)log_expert;
        (*head)++;
        return dst;
    }
    return 0ull;
}

// The read-ahead walk's inputs (`moe_bucketize_kernel` documents each, and
// `moe_read_ahead.cuh` the items it writes). `items` null = no read-ahead.
struct ReadAhead {
    const uint32_t* window;
    const uint32_t* depth;
    const uint32_t* n;
    const uint32_t* list;
    const uint64_t* src;
    uint32_t cap;
    int32_t rows;
    const uint64_t* row_layout;
    uint64_t* items;
    uint32_t* done;
};

// The pinned ranges a remote entry lies in, and the zone's owner tags — what the
// read-ahead walk vets a predicted expert's source against and tags its slot in.
struct ZoneRanges {
    uint64_t pinned0_lo;
    uint64_t pinned0_hi;
    uint64_t pinned1_lo;
    uint64_t pinned1_hi;
    const uint32_t* slot_owner;
    uint64_t zone_end;
    uint64_t zone_slot_bytes;
    uint32_t zone_slots;
};

// The promotion walk, thread 0 alone. First the `n_remote` routed remote
// experts `remote_e` in list order: each takes the next slot while the ring has
// one. Serial because each grant moves the head the next one is judged
// against. `claiming` counts the experts whose claims would evict (decode-
// scored, unmarked remote experts); a launch with more of them than the mapped
// `sweep` word is a sweep and claims nothing. `marked[e]` / `dec[e]` are the
// expert's in-flight mark and decode bit, and `routed[e] > 0` for every expert
// this launch routes. `remote_dst[r]` receives each expert's slot, or 0.
//
// Then, with `ahead.items` set, the READ-AHEAD walk: what is left of the
// layer's link window, spent on the vetted experts predicted for the rows after
// the next (`moe_bucketize.cu`'s header), and only on the stock above the
// reserve — the offers held back for the next rows' decode misses stay theirs.
__device__ __forceinline__ void promotion_walk(
    int32_t n_remote, int32_t claiming,
    const int32_t* remote_e, const uint8_t* marked, const uint8_t* dec, const int32_t* routed,
    const uint64_t* gate_row, long long table_plane, int n_experts, int32_t row, uint32_t summary_seq,
    const uint64_t* promo_slots, uint64_t* promo_log, uint32_t* promo_head, const uint32_t* promo_tail,
    uint32_t promo_cap, uint32_t* promo_marks, const uint32_t* promo_reserve, const uint32_t* promo_sweep,
    const uint64_t* promo_victims, const uint64_t* promo_retarget, uint64_t* remote_dst,
    const ReadAhead ahead, const ZoneRanges zone)
{
    uint32_t head = 0;
    uint32_t tail = 0;
    uint32_t reserve = 0xffffffffu;
    // Whether this launch claims at all: not a sweep.
    bool claim = false;
    if (promo_slots != nullptr) {
        head = *(volatile const uint32_t*)promo_head;
        tail = *(volatile const uint32_t*)promo_tail;
        if (promo_reserve != nullptr) {
            reserve = *(volatile const uint32_t*)promo_reserve;
        }
        const uint32_t sweep = *(volatile const uint32_t*)promo_sweep;
        // The slots the host published before its tail store.
        __threadfence_system();
        claim = (uint32_t)claiming <= sweep;
    }
    // The live table's gate plane, which a victim's index is into.
    uint64_t* const gate_plane =
        gate_row != nullptr ? (uint64_t*)gate_row - (size_t)row * (size_t)n_experts : nullptr;
    for (int32_t r = 0; r < n_remote; r++) {
        const int32_t x = remote_e[r];
        uint64_t dst = 0ull;
        if (claim && !marked[x] && (dec[x] || tail - head > reserve)) {
            // A prompt-only expert evicts nothing: when the next offer holds a
            // resident expert it stays in scratch.
            dst = take_offer(&head, tail, dec[x] != 0, promo_slots, promo_log, promo_cap,
                             promo_victims, promo_retarget, gate_plane, table_plane, n_experts,
                             row, routed, summary_seq, (uint32_t)row, (uint32_t)x);
            if (dst != 0ull) {
                ((volatile uint32_t*)promo_marks)[(size_t)row * n_experts + x] = summary_seq;
            }
        }
        remote_dst[r] = dst;
    }
    if (ahead.items != nullptr) {
        uint32_t n_ahead = 0;
        const uint32_t window = *(volatile const uint32_t*)ahead.window;
        const uint32_t depth = *(volatile const uint32_t*)ahead.depth;
        uint32_t budget = claim && window > (uint32_t)n_remote ? window - (uint32_t)n_remote : 0u;
        budget = budget < AHEAD_MAX ? budget : AHEAD_MAX;
        const uint32_t keep = promo_reserve != nullptr ? reserve : 0u;
        for (uint32_t hop = 2; hop <= depth && n_ahead < budget && tail - head > keep; hop++) {
            const int32_t t = (int32_t)(((uint32_t)row + hop) % (uint32_t)ahead.rows);
            if (t == row) {
                break;
            }
            const uint32_t listed = ((const volatile uint32_t*)ahead.n)[t];
            const uint32_t n = listed < ahead.cap ? listed : ahead.cap;
            for (uint32_t q = 0; q < n && n_ahead < budget && tail - head > keep; q++) {
                const uint32_t x = ((const volatile uint32_t*)ahead.list)[(size_t)t * ahead.cap + q];
                if (x >= (uint32_t)n_experts) {
                    continue;
                }
                const size_t at = (size_t)t * (size_t)n_experts + x;
                volatile uint64_t* g = (volatile uint64_t*)gate_plane + at;
                const uint64_t pg = g[0];
                const uint64_t pu = g[table_plane];
                const uint64_t pd = g[2 * table_plane];
                const bool pinned = (pg >= zone.pinned0_lo && pg < zone.pinned0_hi) ||
                                    (pg >= zone.pinned1_lo && pg < zone.pinned1_hi);
                const uint64_t vetted =
                    ((const volatile uint64_t*)ahead.src)[(size_t)t * ahead.cap + q];
                if (pg == 0ull || pu == 0ull || pd == 0ull || !pinned ||
                    pg - ahead.row_layout[4 * (size_t)t] != vetted ||
                    ((const volatile uint32_t*)promo_marks)[at] != 0u) {
                    continue;
                }
                const uint64_t dst = take_offer(
                    &head, tail, true, promo_slots, promo_log, promo_cap, promo_victims,
                    promo_retarget, gate_plane, table_plane, n_experts, row, routed,
                    summary_seq, (uint32_t)t, AHEAD_FLAG | x);
                if (dst == 0ull) {
                    break;
                }
                ((volatile uint32_t*)promo_marks)[at] = summary_seq;
                const uint64_t* lay = ahead.row_layout + 4 * (size_t)t;
                uint64_t* item = ahead.items + 1 + (size_t)n_ahead * AHEAD_ITEM_WORDS;
                item[0] = pg - lay[0];
                item[1] = dst;
                item[2] = lay[3];
                item[3] = (uint64_t)(uintptr_t)&g[0];
                item[4] = (uint64_t)(uintptr_t)&g[table_plane];
                item[5] = (uint64_t)(uintptr_t)&g[2 * table_plane];
                item[6] = dst + lay[0];
                item[7] = dst + lay[1];
                item[8] = dst + lay[2];
                item[9] = 0ull;
                item[10] = ((uint64_t)(t + 1) << 16) | (uint64_t)x;
                if (zone.slot_owner != nullptr && dst < zone.zone_end &&
                    dst >= zone.zone_end - (uint64_t)zone.zone_slots * zone.zone_slot_bytes) {
                    const uint32_t s = (uint32_t)((zone.zone_end - 1ull - dst) / zone.zone_slot_bytes);
                    item[9] = (uint64_t)(uintptr_t)(zone.slot_owner + s);
                }
                ahead.done[n_ahead] = 0u;
                n_ahead++;
            }
        }
        ahead.items[0] = n_ahead;
    }
    if (promo_slots != nullptr) {
        // The log entries before the counter that covers them.
        __threadfence_system();
        *(volatile uint32_t*)promo_head = head;
    }
}
