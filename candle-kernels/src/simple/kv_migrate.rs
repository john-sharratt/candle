// FFI binding for the KV tier-migration scatter/gather kernel.
//
// `run_kv_migrate_copy` launches one thread block per migration-plan record;
// each block copies `byte_lens[r]` bytes from `src_ptrs[r]` to `dst_ptrs[r]`.
// It is the primitive behind both kv_pack (gather: scattered arena chunks →
// contiguous staging) and kv_unpack (scatter: staging → fresh arena chunks);
// see docs/kv_tier_migration.md §9.
//
// The plan arrays are device pointers:
//   src_ptrs:  device i64[n_records] — resolved source addresses
//   dst_ptrs:  device i64[n_records] — resolved destination addresses
//   byte_lens: device i64[n_records] — per-record byte counts
//   stream:    cudaStream_t

use std::ffi::c_void;

extern "C" {
    pub fn run_kv_migrate_copy(
        src_ptrs: *const i64,
        dst_ptrs: *const i64,
        byte_lens: *const i64,
        n_records: i32,
        stream: *mut c_void,
    );
}
