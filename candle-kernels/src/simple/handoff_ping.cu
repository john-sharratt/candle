// handoff_ping — the device side of the CPU expert pool's handoff, measured
// alone (docs/cpu_expert_compute.md §5.9, §8 measurement 0).
//
// One thread rings `doorbell` with 1..n in turn, and after each ring spins on
// `done` until the host has answered it — the shape the down launch's CPU
// worker item takes: a store to mapped host memory, a system fence, and a
// volatile poll of a mapped word the host writes with no CUDA call. All n
// round trips run inside one launch, so the measurement is the mapped-memory
// path itself, not launch overhead.
//
// A wait that passes `limit_ns` gives up: it records the iteration in
// `gave_up` and returns, so a host that never answers cannot hold the device.

#include <stdint.h>

static __device__ __forceinline__ uint64_t ping_globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void handoff_ping_kernel(
    volatile uint32_t* doorbell,
    const volatile uint32_t* done,
    uint32_t n,
    uint64_t limit_ns,
    uint32_t* gave_up
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (uint32_t i = 1; i <= n; ++i) {
        *doorbell = i;
        __threadfence_system();
        const uint64_t t0 = ping_globaltimer();
        while (*done < i) {
            if (ping_globaltimer() - t0 > limit_ns) {
                *gave_up = i;
                __threadfence_system();
                return;
            }
        }
    }
}

extern "C" void run_handoff_ping(
    uint32_t* doorbell,
    const uint32_t* done,
    uint32_t n,
    uint64_t limit_ns,
    uint32_t* gave_up,
    cudaStream_t stream
) {
    if (n == 0) return;
    handoff_ping_kernel<<<1, 32, 0, stream>>>(doorbell, done, n, limit_ns, gave_up);
}
