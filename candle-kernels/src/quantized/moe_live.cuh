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
    int workers;
    // The launch's MoE row, which a fault names.
    int row;
};
