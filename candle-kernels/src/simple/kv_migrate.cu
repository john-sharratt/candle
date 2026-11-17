// =============================================================================
// kv_migrate_copy — the KV-cache tier-migration scatter/gather kernel
// =============================================================================
//
// The migration primitive behind kv_pack (evict / gather) and kv_unpack
// (load / scatter) — see docs/archived/kv_tier_migration.md §9.
//
// One WARP per migration-plan record, grid-stride (`scatter_copy.cuh`): a
// record is a KV band, ~1-4 KB, which a whole block left mostly idle and paid
// one round trip for. Each record copies byte_lens[r] bytes from src_ptrs[r] to
// dst_ptrs[r]. The plan is built host-side: the caller resolves every chunk's
// device address (so the kernel needs no arena-table indexing) and lays out the
// contiguous staging buffer.
//
//   kv_pack   — src = scattered arena chunks, dst = contiguous staging.
//   kv_unpack — src = contiguous staging,     dst = fresh arena chunks.
//
// Direction is just which side is the scattered set; the kernel body is
// identical, so one kernel serves both.
//
//   Grid:  (min(warps needed, SCATTER_BLOCKS_PER_SM blocks per SM), 1, 1)
//   Block: (SCATTER_THREADS, 1, 1)
//
// A 16-byte vectorised copy, four loads in flight a lane, is used when a row's
// source, destination and length are all 16-byte aligned; otherwise a correct
// byte copy runs.

#include <cstdint>
#include <cuda_runtime.h>

#include "scatter_copy.cuh"

// A record may copy a **run of equally-spaced rows** rather than a single
// block. `rows`, `src_strides` and `dst_strides` are optional: a null array
// means one row, which is the migration case and reduces the body below to
// exactly the single-block copy it always was. Passing null also means the
// caller uploads nothing extra, so the generalisation costs the migration path
// three null pointers and no memory.
//
// The strided form exists for the prefill KV write, where a band's destination
// is a contiguous run inside an arena chunk but its source is a slice of
// `[tokens, head_dim]` — `head_dim / N_PALETTE` contiguous elements per token,
// `head_dim` apart. Expressed one row at a time that is a record per token; as
// a strided record it is one record per (block, head, band), and the whole
// layer's write becomes a single launch.
__global__ void __launch_bounds__(SCATTER_THREADS, SCATTER_BLOCKS_PER_SM)
kv_migrate_copy_kernel(
    const int64_t* __restrict__ src_ptrs,
    const int64_t* __restrict__ dst_ptrs,
    const int64_t* __restrict__ byte_lens,
    const int64_t* __restrict__ rows,
    const int64_t* __restrict__ src_strides,
    const int64_t* __restrict__ dst_strides,
    int n_records
) {
    const int lane = threadIdx.x & (SCATTER_WARP - 1);
    const int warps = scatter_warp_count();
    for (int r = scatter_warp_id(); r < n_records; r += warps) {
        const uint8_t* s = reinterpret_cast<const uint8_t*>(src_ptrs[r]);
        uint8_t* d = reinterpret_cast<uint8_t*>(dst_ptrs[r]);
        const unsigned long long len = (unsigned long long)byte_lens[r];
        const int n_rows = rows ? (int)rows[r] : 1;
        // A strided record's rows, each a stride on from the last; with no
        // stride array they are back to back, which for one row is unused.
        const int64_t ss = src_strides ? src_strides[r] : (int64_t)len;
        const int64_t ds = dst_strides ? dst_strides[r] : (int64_t)len;
        for (int row = 0; row < n_rows; ++row, s += ss, d += ds) {
            warp_copy<LoadCg>(s, d, len, lane);
        }
    }
}

// The device's SM count, read once per process: the launch is sized by it.
static int kv_migrate_sm_count() {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        int n = 0;
        if (cudaGetDevice(&dev) == cudaSuccess
            && cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess
            && n > 0) {
            sms = n;
        } else {
            sms = 1;
        }
    }
    return sms;
}

extern "C" void run_kv_migrate_copy(
    const int64_t* src_ptrs,
    const int64_t* dst_ptrs,
    const int64_t* byte_lens,
    const int64_t* rows,
    const int64_t* src_strides,
    const int64_t* dst_strides,
    int            n_records,
    cudaStream_t   stream
) {
    if (n_records <= 0) return;
    const int warps_per_block = SCATTER_THREADS / SCATTER_WARP;
    const int needed = (n_records + warps_per_block - 1) / warps_per_block;
    const int fill = kv_migrate_sm_count() * SCATTER_BLOCKS_PER_SM;
    const int grid = needed < fill ? needed : fill;
    kv_migrate_copy_kernel<<<grid, SCATTER_THREADS, 0, stream>>>(
        src_ptrs, dst_ptrs, byte_lens, rows, src_strides, dst_strides, n_records
    );
}
