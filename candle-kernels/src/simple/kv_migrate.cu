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
__global__ void kv_migrate_copy_kernel(
    const int64_t* __restrict__ src_ptrs,
    const int64_t* __restrict__ dst_ptrs,
    const int64_t* __restrict__ byte_lens,
    const int64_t* __restrict__ rows,
    const int64_t* __restrict__ src_strides,
    const int64_t* __restrict__ dst_strides,
    int n_records
) {
    const int r = blockIdx.x;
    if (r >= n_records) return;

    const int64_t s = src_ptrs[r];
    const int64_t d = dst_ptrs[r];
    const int64_t len = byte_lens[r];
    const int64_t n_rows = rows ? rows[r] : 1;
    // With no stride array the rows are back to back, which for `n_rows == 1`
    // makes the stride unreachable anyway.
    const int64_t ss = src_strides ? src_strides[r] : len;
    const int64_t ds = dst_strides ? dst_strides[r] : len;
    const int t = threadIdx.x;
    const int stride = blockDim.x;

    // Hoisted: every row shares the base alignment, and the strides decide
    // whether that survives row to row. For a single unstrided record this is
    // the original `(s | d | len)` test, since `ss == ds == len`.
    const bool aligned = ((s | d | len | ss | ds) & 15) == 0;

    for (int64_t row = 0; row < n_rows; ++row) {
        const int64_t sr = s + row * ss;
        const int64_t dr = d + row * ds;
        if (aligned) {
            const int4* sv = (const int4*)sr;
            int4* dv = (int4*)dr;
            const int64_t n = len >> 4;
            for (int64_t i = t; i < n; i += stride) {
                dv[i] = sv[i];
            }
        } else {
            const char* sb = (const char*)sr;
            char* db = (char*)dr;
            for (int64_t i = t; i < len; i += stride) {
                db[i] = sb[i];
            }
        }
    }
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
    const int threads = 256;
    kv_migrate_copy_kernel<<<n_records, threads, 0, stream>>>(
        src_ptrs, dst_ptrs, byte_lens, rows, src_strides, dst_strides, n_records
    );
}
