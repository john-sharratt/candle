#pragma once

// A grouped GEMM launched over a LIVE expert table — passed by value from the
// launcher (`dispatcher.cu`) to the kernel entry (`kernel.cuh`, where the worker
// blocks and the counter layout are documented). Its Rust twin is
// `candle::quantized::cuda::MoeLive`; the two must stay field-for-field
// identical.
struct MoeLive {
    // Mapped host word the host sets non-zero when a cold expert can never be
    // published; null for every launch that is not live.
    const unsigned int* abort;
    // Mapped host word the first worker whose wait ends without its expert
    // claims (`kernel.cuh`, "A live expert table", for the bits); the host
    // fails the forward on it.
    unsigned long long* fault;
    // This projection's row of the live table (mapped host memory): where a
    // worker waits for a COLD expert. Every other expert's address comes from
    // the launch's `weight_ptrs`, `moe_bucketize`'s snapshot in VRAM.
    const unsigned long long* live_row;
    // `moe_bucketize`'s remote-expert list, `[n][4]` of
    // `{expert, first_tile, n_tiles, cold}`, its promotion slots (`[n]`, a VRAM
    // slot image per remote expert or 0), and its header (`[3]` remote
    // experts, `[4]` the tiles they own — the first of the tile list).
    const int* remote;
    const unsigned long long* remote_dst;
    const int* header;
    // This launch's work counter, zeroed by `moe_bucketize`.
    int* counter;
    // Worker scratch in VRAM: `workers` slots of `slot_bytes`.
    unsigned char* scratch;
    unsigned long long slot_bytes;
    // This projection's byte offset inside a slot image — where in a promotion
    // slot its slices go.
    unsigned long long dst_offset;
    // Profile build only: this row's counters (see `kernel.cuh`); null otherwise.
    unsigned long long* stall;
    // The gate launch only: `moe_bucketize`'s read-ahead items and their piece
    // counters (`moe_read_ahead.cuh`) — copied and published by the workers
    // after their own items. Null on the up and down launches.
    const unsigned long long* ahead;
    unsigned int* ahead_done;
    // Backstop below the display watchdog: a worker waiting longer gives the
    // expert up and claims the fault word.
    unsigned long long spin_limit_ns;
    // The down launch only — the CPU class's results (`moe_bucketize_cpu.cuh`),
    // pulled by the workers' CPU items into the rows bucketize left the CPU
    // experts: the host pool's mapped `done` word and the value that means this
    // invocation's rows have landed; the mapped result rows (ring order, one F32
    // row of the output width each); the ring row → grouped row map; the pool's
    // mapped liveness word (non-zero while it lives); how long a CPU item
    // polls tightly before it backs off; and the mapped word each CPU item
    // raises to the nanoseconds it waited for the pool (atomicMax), which the
    // pool reads back as the device's lateness for the split's online loop.
    // Null `cpu_done` everywhere else.
    const unsigned long long* cpu_done;
    unsigned long long cpu_seq;
    const float* cpu_rows;
    const unsigned int* cpu_dst_row;
    const unsigned int* cpu_alive;
    unsigned long long cpu_tight_ns;
    unsigned long long* cpu_wait;
    // The gate launch only: a mapped `{ns, count}` pair the worker holding a
    // cold expert's first row tile adds its wait for the publish to — one
    // sample per cold expert per layer. Null on the up and down launches,
    // which find the expert published.
    unsigned long long* cold_wait;
    int workers;
    // The launch's MoE row, which a fault names.
    int row;
};
static_assert(sizeof(MoeLive) == 23 * 8, "MoeLive is api.rs's MoeLive byte for byte");
