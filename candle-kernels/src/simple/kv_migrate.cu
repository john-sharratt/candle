// =============================================================================
// kv_migrate_copy — the KV-cache tier-migration scatter/gather kernel
// =============================================================================
//
// The migration primitive behind kv_pack (evict / gather) and kv_unpack
// (load / scatter) — see docs/kv_tier_migration.md §9.
//
// One CUDA block per migration-plan record. Each block copies byte_lens[r]
// bytes from src_ptrs[r] to dst_ptrs[r]. The plan is built host-side: the
// caller resolves every chunk's device address (so the kernel needs no
// arena-table indexing) and lays out the contiguous staging buffer.
//
//   kv_pack   — src = scattered arena chunks, dst = contiguous staging.
//   kv_unpack — src = contiguous staging,     dst = fresh arena chunks.
//
// Direction is just which side is the scattered set; the kernel body is
// identical, so one kernel serves both.
//
//   Grid:  (n_records, 1, 1)
//   Block: (256, 1, 1)
//
// A 16-byte vectorised copy is used when a record's source, destination,
// and length are all 16-byte aligned; otherwise a correct byte copy runs.

#include <cstdint>
#include <cuda_runtime.h>

__global__ void kv_migrate_copy_kernel(
    const int64_t* __restrict__ src_ptrs,
    const int64_t* __restrict__ dst_ptrs,
    const int64_t* __restrict__ byte_lens,
    int n_records
) {
    const int r = blockIdx.x;
    if (r >= n_records) return;

    const int64_t s = src_ptrs[r];
    const int64_t d = dst_ptrs[r];
    const int64_t len = byte_lens[r];
    const int t = threadIdx.x;
    const int stride = blockDim.x;

    if (((s | d | len) & 15) == 0) {
        // Fast path: fully 16-byte aligned.
        const int4* sv = (const int4*)s;
        int4* dv = (int4*)d;
        const int64_t n = len >> 4;
        for (int64_t i = t; i < n; i += stride) {
            dv[i] = sv[i];
        }
    } else {
        // Correct for any alignment.
        const char* sb = (const char*)s;
        char* db = (char*)d;
        for (int64_t i = t; i < len; i += stride) {
            db[i] = sb[i];
        }
    }
}

extern "C" void run_kv_migrate_copy(
    const int64_t* src_ptrs,
    const int64_t* dst_ptrs,
    const int64_t* byte_lens,
    int            n_records,
    cudaStream_t   stream
) {
    if (n_records <= 0) return;
    const int threads = 256;
    kv_migrate_copy_kernel<<<n_records, threads, 0, stream>>>(
        src_ptrs, dst_ptrs, byte_lens, n_records
    );
}
