// FFI binding for the KV tier-migration scatter/gather kernel.
//
// `run_kv_migrate_copy` launches one thread block per migration-plan record;
// each block copies `byte_lens[r]` bytes from `src_ptrs[r]` to `dst_ptrs[r]`.
// It is the primitive behind both kv_pack (gather: scattered arena chunks →
// contiguous staging) and kv_unpack (scatter: staging → fresh arena chunks);
// see docs/kv_tier_migration.md §9.
//
// A record may also copy a **run of equally-spaced rows**, which is what the
// prefill KV write needs: a band's destination is a contiguous run inside an
// arena chunk, but its source is `head_dim / N_PALETTE` elements per token,
// `head_dim` apart. One strided record per (block, head, band) turns a layer's
// whole write into a single launch.
//
// The plan arrays are device pointers:
//   src_ptrs:    device i64[n_records] — resolved source addresses
//   dst_ptrs:    device i64[n_records] — resolved destination addresses
//   byte_lens:   device i64[n_records] — bytes per row
//   rows:        device i64[n_records] — rows per record, or **null** for 1
//   src_strides: device i64[n_records] — source row pitch, or **null**
//   dst_strides: device i64[n_records] — destination row pitch, or **null**
//   stream:      cudaStream_t
//
// The three optional arrays are null for the migration paths, which copy one
// block per record: the kernel then reduces to exactly the copy it always was
// and the caller uploads nothing extra.

use std::ffi::c_void;

extern "C" {
    pub fn run_kv_migrate_copy(
        src_ptrs: *const i64,
        dst_ptrs: *const i64,
        byte_lens: *const i64,
        rows: *const i64,
        src_strides: *const i64,
        dst_strides: *const i64,
        n_records: i32,
        stream: *mut c_void,
    );
}
