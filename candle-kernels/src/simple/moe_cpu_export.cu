// =============================================================================
// MoE CPU EXPORT — the CPU class's input rows, out to the host pool
// =============================================================================
// Right after bucketize, this copies the INPUT ROWS bucketize chose
// (`moe_bucketize_cpu.cuh`, `cpu_inputs`) — each distinct token the layer's
// CPU experts route, once, however many of them it routes to — from the MoE
// input operand (token-major) into the mapped ROW RING, input row r being
// token `in_tok[r]`, and then rings the pool's DOORBELL with the invocation's
// sequence number. It reads only the input and bucketize's tables, so it runs
// before the gather and the pool starts as early as it can.
//
// The ring keeps the operand's own layout (q8a1024: tile `r·tiles + t`), so
// the pool reads it exactly as it reads any q8a128 operand. A launch with no
// input rows (`header[7] == 0`: no CPU expert this layer) copies nothing,
// fences nothing and rings nothing: its blocks only count themselves out.
//
// One warp per 128-wide tile, eight warps a block, striding over the rows: the
// grid is a fixed few blocks, not one per row. The doorbell is the LAST
// block's: every block that copied fences its writes system-wide before
// counting itself out, so the block that takes the count to the grid size
// knows every row is visible to the host before it stores the doorbell — and
// keys the count by the ticket it counts, so no other launch's count completes it.
// =============================================================================

#include <stdint.h>
#include "../blocks.cuh"

#define EXPORT_WARPS 8

extern "C" __global__ void moe_cpu_export_kernel(
    const uint8_t* __restrict__ src,       // the MoE input operand (q8a1024, token-major)
    const int tiles,                       // 128-wide tiles per row
    const uint32_t* __restrict__ in_tok,   // input row → token
    const int32_t* __restrict__ header,    // bucketize's header: [7] = input rows
    uint8_t* ring,                         // the row ring (mapped)
    const uint32_t cap_rows,               // rows the ring holds
    unsigned long long* doorbell,          // the pool's doorbell (mapped)
    const unsigned long long seq,
    unsigned long long* blocks_done)       // device counter, keyed by ticket
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int32_t rows_h = header[7];
    const uint32_t n = rows_h > 0 ? ((uint32_t)rows_h < cap_rows ? (uint32_t)rows_h : cap_rows) : 0u;
    const long long n_tiles = (long long)n * tiles;
    for (long long ft = (long long)blockIdx.x * EXPORT_WARPS + warp; ft < n_tiles;
         ft += (long long)gridDim.x * EXPORT_WARPS) {
        const long long r = ft / tiles;
        const int t = (int)(ft - r * tiles);
        const long long s = (long long)in_tok[r] * tiles + t;
        const uint32_t q = *reinterpret_cast<const uint32_t*>(src + q8a1024_qs_off(s) + lane * 4);
        *reinterpret_cast<uint32_t*>(ring + q8a1024_qs_off(ft) + lane * 4) = q;
        if (lane < 4) {
            const uint32_t d = *reinterpret_cast<const uint32_t*>(src + q8a1024_ds_off(s) + lane * 4);
            *reinterpret_cast<uint32_t*>(ring + q8a1024_ds_off(ft) + lane * 4) = d;
        }
    }
    // `n` is the launch's: either every block copied and fences, or none.
    if (n > 0) {
        __threadfence_system();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // The count is keyed by the invocation: the high word names the ticket
        // it counts, the low word how many of its blocks are out. A count left
        // by another launch — one discarded mid-layer before its blocks all
        // counted out, or one with another grid — names another ticket, and the
        // first block here starts this one's from its own 1 instead of adding
        // to it. Shared and unkeyed, a stale count reached `gridDim` early or
        // never, and a never rang nothing: the down launch waited out its
        // deadline on a doorbell that had not moved.
        const unsigned long long key = (seq & 0xffffffffull) << 32;
        unsigned long long seen = *(volatile unsigned long long*)blocks_done;
        unsigned long long next;
        for (;;) {
            next = ((seen & 0xffffffff00000000ull) == key) ? seen + 1ull : (key | 1ull);
            const unsigned long long was = atomicCAS(blocks_done, seen, next);
            if (was == seen) {
                break;
            }
            seen = was;
        }
        if ((next & 0xffffffffull) == (unsigned long long)gridDim.x && n > 0) {
            __threadfence_system();
            *(volatile unsigned long long*)doorbell = seq;
        }
    }
}

// `grid` blocks of EXPORT_WARPS warps, striding over the rows: the caller
// sizes it to the rows a launch can export at most, capped at a few blocks,
// since the row count is a device value. Returns 0 when launched, 1 when an
// argument guard refused it.
extern "C" int32_t run_moe_cpu_export(
    const void* src,
    int32_t tiles,
    const void* in_tok,
    const void* header,
    void* ring,
    uint32_t cap_rows,
    void* doorbell,
    uint64_t seq,
    void* blocks_done,
    int32_t grid,
    void* stream)
{
    if (src == nullptr || tiles <= 0 || in_tok == nullptr || header == nullptr ||
        ring == nullptr || doorbell == nullptr || blocks_done == nullptr || grid <= 0) {
        return 1;
    }
    moe_cpu_export_kernel<<<grid, EXPORT_WARPS * 32, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)src, tiles, (const uint32_t*)in_tok, (const int32_t*)header,
        (uint8_t*)ring, cap_rows, (unsigned long long*)doorbell, (unsigned long long)seq,
        (unsigned long long*)blocks_done);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 2;
}
