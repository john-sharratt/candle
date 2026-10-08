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

// The promotion walk, thread 0 alone, over the `n_remote` routed remote experts
// `remote_e` in list order: each takes the next slot while the ring has one.
// Serial because each grant moves the head the next one is judged against.
// `claiming` counts the experts whose claims would evict (decode-scored,
// unmarked remote experts); a launch with more of them than the mapped `sweep`
// word is a sweep and claims nothing. `marked[e]` / `dec[e]` are the expert's
// in-flight mark and decode bit, and `routed[e] > 0` for every expert this
// launch routes. `remote_dst[r]` receives each expert's slot, or 0.
__device__ __forceinline__ void promotion_walk(
    int32_t n_remote, int32_t claiming,
    const int32_t* remote_e, const uint8_t* marked, const uint8_t* dec, const int32_t* routed,
    const uint64_t* gate_row, long long table_plane, int n_experts, int32_t row, uint32_t summary_seq,
    const uint64_t* promo_slots, uint64_t* promo_log, uint32_t* promo_head, const uint32_t* promo_tail,
    uint32_t promo_cap, uint32_t* promo_marks, const uint32_t* promo_reserve, const uint32_t* promo_sweep,
    const uint64_t* promo_victims, const uint64_t* promo_retarget, uint64_t* remote_dst)
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
            while (head != tail) {
                const uint32_t i = head % promo_cap;
                const uint64_t victim = ((const volatile uint64_t*)promo_victims)[i];
                if (victim != PROMO_EMPTY && !dec[x]) {
                    // A prompt-only expert evicts nothing: the next offer holds a
                    // resident expert, so it stays in scratch.
                    break;
                }
                if (victim != PROMO_EMPTY) {
                    const int32_t vr = (int32_t)(victim / (uint64_t)n_experts);
                    const int32_t ve = (int32_t)(victim % (uint64_t)n_experts);
                    if (vr == row && routed[ve] > 0) {
                        ((volatile uint64_t*)promo_log)[i] =
                            ((uint64_t)summary_seq << 32) | ((uint64_t)row << 16) | (uint64_t)PROMO_SKIP;
                        head++;
                        continue;
                    }
                    // No fence per claim: every later kernel sees these stores
                    // by stream order, nothing polls a VRAM victim's entry
                    // concurrently, and the host learns of the claim only
                    // through `head`, which is published behind a system fence
                    // below.
                    const volatile uint64_t* rt = (const volatile uint64_t*)promo_retarget + 3 * (size_t)i;
                    volatile uint64_t* g = (volatile uint64_t*)gate_plane + victim;
                    g[table_plane] = rt[1];
                    g[2 * table_plane] = rt[2];
                    g[0] = rt[0];
                }
                dst = ((const volatile uint64_t*)promo_slots)[i];
                ((volatile uint64_t*)promo_log)[i] =
                    ((uint64_t)summary_seq << 32) | ((uint64_t)row << 16) | (uint64_t)x;
                ((volatile uint32_t*)promo_marks)[(size_t)row * n_experts + x] = summary_seq;
                head++;
                break;
            }
        }
        remote_dst[r] = dst;
    }
    if (promo_slots != nullptr) {
        // The log entries before the counter that covers them.
        __threadfence_system();
        *(volatile uint32_t*)promo_head = head;
    }
}
