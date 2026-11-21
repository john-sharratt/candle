// The device-side live table reads, fault word and wait helpers of the grouped
// MoE launch.
//
// The `MOE_LIVE_*` and `MOE_FAULT_*` constants, `moe_live_load` and the
// `moe_live_ld16` loads of the live entry, `moe_live_faulted` for the fault
// word, and `moe_live_source` and `moe_live_cpu_wait`, which wait on a cold
// expert and on the host pool's results through mapped words.

// =============================================================================
// GROUPED TENSOR-CORE GEMM — all MoE experts in a SINGLE kernel launch
// =============================================================================
// The per-expert path dispatches N experts as N separate launches (one per
// segment). At small tokens-per-expert that is both launch-bound AND occupancy-
// starved: each expert's matmul fills only ceil(N/32) row-blocks, so a handful
// of blocks idle most of the GPU. This kernel processes ALL experts' tiles in
// one launch — grid = (total_tiles, row_tiles). A device-side table maps
// blockIdx.x → the owning expert's weight pointer and its [b_start,b_start+b_cnt)
// slice of the stacked activations/output (b_cnt ≤ 16, one MMA M-tile). The
// combined expert tiles fill the SMs, so occupancy and launch count improve
// together. Reuses the tc16 MMA inner loop verbatim.
//
// TILE WIDTH: this kernel only emits 16-wide (tc16-style) MMA tiles — an expert
// with M tokens becomes ceil(M/16) tiles. That is optimal for MoE decode, where
// tokens-per-expert is small (1 single-session, ~4-16 wave-batched). For very
// large M-per-expert (>= ~32) the per-weight path would escalate to the 32-wide
// tc32 kernel, which re-reads each weight block half as often; so a single
// expert carrying hundreds of tokens (a regime MoE decode does not hit) would
// run marginally faster through the per-segment path than through this one. A
// 32-wide grouped tile variant would close that gap if such a workload appears.
//
// PRECONDITION: nrows (output features) must be a multiple of N_TILE (32) — the
// row-tile writes a full 32-row block with no partial-row guard, same as the
// tc16 path. All MoE expert dims (768 / 2048) satisfy this; grouped_matmul_gemx
// enforces it host-side.

// ── A live expert table: worker blocks ─────────────────────────────────────
//
// The routed MoE expert path (`expert_lre`) launches the device-table grouped
// GEMM over a LIVE pointer table kept in mapped pinned host memory: an entry is
// an expert's projection in VRAM (a hit), in pinned host memory (a warm-tier or
// pad slot image), or 0 (cold — the host's stager has not yet read it from the
// NVMe pack). `moe_bucketize` orders the tiles of every non-VRAM ("remote")
// expert first, lists those experts, and snapshots the routed experts' entries
// into VRAM; the launch's `weight_ptrs` is that snapshot. Such a launch carries
// a `MoeLive`, and its grid is `(row_tiles, worker_rows + tiles)`, where
// `worker_rows = ⌈workers / row_tiles⌉`:
//
// - The first `worker_rows` grid rows are the WORKERS, numbered row by row (the
//   last row's blocks past `workers` exit). Rows rather than one row as wide as
//   the worker count, because the grid's width is every tile row's width: a
//   128-worker row over a 16-row-tile projection gave each tile row 112 blocks
//   that only exited. Each worker pulls items `(remote expert, 32-row tile)`
//   off the launch's counter, takes the expert's address — from the snapshot,
//   or for a cold expert from the live row, spinning gently while it is 0 —
//   copies the row tile's slice (every K block's 4 chunks) into its
//   VRAM scratch slot with wide loads, syncs once, and runs the unmodified
//   int8 impl over the slot as a 32-row matrix for every token tile of the
//   expert. A remote expert crosses PCIe once per projection, never per tile.
// - The rows after them are the tiles. A remote expert's tile exits (a worker owns it);
//   every other tile is a hit and runs as it always has. Hit blocks never wait.
// - A remote expert bucketize gave a promotion slot (`remote_dst`, a VRAM slot
//   image) also has each slice the workers copy stored there, at the
//   projection's offset (`dst_offset`): once the layer's three launches are
//   done the slot holds the whole expert, and the host points its entry there.
// - The gate launch's workers then READ AHEAD (`ahead`, `moe_read_ahead.cuh`):
//   past the launch's own items come `AHEAD_CHUNKS` pieces of each item
//   bucketize claimed for a later row's predicted expert. A piece is a plain
//   byte copy of the expert's warm-slot image into its claimed slot — no
//   compute, no layer geometry. The worker that finishes an item's last piece
//   publishes it: the slot's owner tag, the up and down entries, a system
//   fence, then gate. A later bucketize then finds the expert in VRAM without
//   the host having done anything; the host books it when it collects the
//   claim from the ring log. Demand items come first, so read-ahead only ever
//   fills time the launch's own experts left over.
//
// Only workers wait, and what they wait on is a host store — never a submitted
// copy or any other driver call, which a thread blocked in the driver (a lazy
// kernel load behind this launch) could hold back. `abort` (a mapped host word)
// and `spin_limit_ns` end a wait without the expert: the worker claims `fault`
// (a mapped host word naming the row, the expert and the wait) and gives the
// item up; any other cold wait that sees the word set gives its expert up at
// once. The host reads it once the forward has synchronised and fails the
// forward, so no result computed from the layer is returned, and the context is
// left usable. A given-up item writes nothing — not its output, and not the
// promotion slot it was copying into, which the host therefore never installs.
//
// `stall` is the profile build's per-row counter block, null otherwise:
//   stall[0] cold_wait_ns     — workers' time spent waiting for a cold expert
//   stall[1] items            — worker items (remote expert × row tile) served
//   stall[2] bytes            — bytes workers copied into scratch
//   stall[3] copy_ns          — workers' time in the copy (mini loop)
//   stall[4] launches         — live launches (one per projection per layer)
//   stall[5] ahead_items      — read-ahead items published
//   stall[6] ahead_bytes      — bytes read ahead
//
// The struct itself is in `moe_live.cuh`, shared with the launcher.
#define MOE_LIVE_STALL_WORDS 7

__device__ __forceinline__ unsigned long long moe_live_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// An entry the host rewrites while kernels run, read with an acquire load at
// system scope — never through the read-only path a `const __restrict__`
// pointer may compile to.
__device__ __forceinline__ unsigned long long moe_live_load(const uint64_t* p) {
#if __CUDA_ARCH__ >= 700
    unsigned long long v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
#else
    return *reinterpret_cast<const volatile unsigned long long*>(p);
#endif
}

// 16 bytes through L2 only — the worker's copy reads each byte once.
__device__ __forceinline__ uint4 moe_live_ld16(const uint8_t* p) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

// 16 bytes of host memory the host REWRITES in place — the CPU pool's result
// ring, the same addresses every layer. `.cv` treats any cached copy of the
// line as stale and fetches it again: `.cg` may be served from an L2 line the
// previous layer's pull left behind, which nothing between two launches of one
// graph invalidates, and returns the previous layer's results.
__device__ __forceinline__ uint4 moe_live_ld16_cv(const uint8_t* p) {
    uint4 v;
    asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

// Every poll is a PCIe read, and a launch can have every worker waiting: back
// off to tens of microseconds so the polls never compete with the copies.
#define MOE_LIVE_MAX_SLEEP_NS 32768u

// Result rows one CPU item of the down launch pulls (`api.rs`'s
// `MOE_LIVE_CPU_ROWS`), and the expert its fault names: the pool's results
// never landed.
#define MOE_LIVE_CPU_ROWS 16
#define MOE_FAULT_CPU_EXPERT 0xffffull

// 16-byte loads each worker thread keeps in flight in a slice copy. Eight measured
// level with four on Flash-Next decode (RTX 4090 Laptop, 32 workers): with that
// many workers, the round trips of one are hidden by the others.
#define MOE_LIVE_LOADS 4

// The fault word's fields (`MoeLive::fault`; `api.rs`'s `MOE_FAULT_*`): bit 63
// set when the wait ended on the abort word, the row in bits 48-62, the expert
// in bits 32-47, the microseconds waited in bits 0-31, saturating.
#define MOE_FAULT_ABORTED (1ull << 63)
#define MOE_FAULT_ROW_SHIFT 48
#define MOE_FAULT_EXPERT_SHIFT 32

// Whether a worker of any launch has claimed the fault word. A forward that
// faulted is failed by the host, so a cold wait that sees the word gives its
// expert up at once instead of waiting out the spin limit. Read only inside a
// wait, which polls the link anyway: an item whose expert is already published
// pays nothing for it.
__device__ __forceinline__ bool moe_live_faulted(const MoeLive& live) {
    return *reinterpret_cast<const volatile unsigned long long*>(live.fault) != 0ull;
}

// Thread 0 of a worker: the live entry of a cold `expert`, waited for while it
// is 0. Returns the time spent waiting. A wait that ends without the expert —
// the abort word raised, or past `spin_limit_ns` — claims the fault word with
// what it waited on and leaves `*src` 0: the worker gives the item up rather
// than trapping, so the context survives and the host fails only the forward.
__device__ __forceinline__ unsigned long long moe_live_source(
    int expert, const MoeLive& live, unsigned long long* src)
{
    const uint64_t* entry = reinterpret_cast<const uint64_t*>(live.live_row) + expert;
    unsigned long long p = moe_live_load(entry);
    if (p != 0) {
        *src = p;
        return 0;
    }
    const unsigned long long t0 = moe_live_now();
    unsigned int ns = 32;
    while (p == 0) {
        const unsigned int aborted = *reinterpret_cast<const volatile unsigned int*>(live.abort);
        const unsigned long long waited = moe_live_now() - t0;
        if (aborted != 0u || waited > live.spin_limit_ns || moe_live_faulted(live)) {
            const unsigned long long us = waited / 1000ull;
            const unsigned long long code =
                (aborted != 0u ? MOE_FAULT_ABORTED : 0ull) |
                ((unsigned long long)(live.row & 0x7fff) << MOE_FAULT_ROW_SHIFT) |
                ((unsigned long long)(expert & 0xffff) << MOE_FAULT_EXPERT_SHIFT) |
                (us < 0xffffffffull ? us : 0xffffffffull);
            atomicCAS(live.fault, 0ull, code);
            __threadfence_system();
            *src = 0;
            return waited;
        }
#if __CUDA_ARCH__ >= 700
        __nanosleep(ns);
#endif
        ns = ns < MOE_LIVE_MAX_SLEEP_NS ? ns * 2u : MOE_LIVE_MAX_SLEEP_NS;
        p = moe_live_load(entry);
    }
    *src = p;
    return moe_live_now() - t0;
}

// Thread 0 of a worker: waits for the host pool's results — its mapped `done`
// word reading `live.cpu_seq` (`moe_bucketize_cpu.cuh`). Polls tightly for
// `cpu_tight_ns` — the pool's predicted finish, when a result is due — and
// backs off past it, where a late pool is waiting on something slow. A wait
// that ends without the results — the abort word, the spin limit, another
// fault, or the pool's liveness word dropped — claims the fault word naming
// MOE_FAULT_CPU_EXPERT and returns false. A wait that ends with the results
// raises `cpu_wait` to its length: the device's lateness on the pool.
__device__ __forceinline__ bool moe_live_cpu_wait(const MoeLive& live)
{
    if (moe_live_load(reinterpret_cast<const uint64_t*>(live.cpu_done)) == live.cpu_seq) {
        return true;
    }
    const unsigned long long t0 = moe_live_now();
    unsigned int ns = 32;
    for (;;) {
        const unsigned int aborted = *reinterpret_cast<const volatile unsigned int*>(live.abort);
        const unsigned int alive = *reinterpret_cast<const volatile unsigned int*>(live.cpu_alive);
        const unsigned long long waited = moe_live_now() - t0;
        if (aborted != 0u || alive == 0u || waited > live.spin_limit_ns || moe_live_faulted(live)) {
            const unsigned long long us = waited / 1000ull;
            const unsigned long long code =
                (aborted != 0u ? MOE_FAULT_ABORTED : 0ull) |
                ((unsigned long long)(live.row & 0x7fff) << MOE_FAULT_ROW_SHIFT) |
                (MOE_FAULT_CPU_EXPERT << MOE_FAULT_EXPERT_SHIFT) |
                (us < 0xffffffffull ? us : 0xffffffffull);
            atomicCAS(live.fault, 0ull, code);
            __threadfence_system();
            return false;
        }
        if (waited > live.cpu_tight_ns) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(ns);
#endif
            ns = ns < MOE_LIVE_MAX_SLEEP_NS ? ns * 2u : MOE_LIVE_MAX_SLEEP_NS;
        }
        if (moe_live_load(reinterpret_cast<const uint64_t*>(live.cpu_done)) == live.cpu_seq) {
            if (live.cpu_wait != nullptr) {
                atomicMax(live.cpu_wait, moe_live_now() - t0);
            }
            return true;
        }
    }
}
