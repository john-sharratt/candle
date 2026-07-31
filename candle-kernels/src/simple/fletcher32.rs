// FFI binding for the Fletcher-32 KV-chunk golden checksum kernel.
//
// `run_fletcher32` launches one thread block per plan record; each block
// computes a Fletcher-32 checksum over `byte_lens[r]` bytes at `src_ptrs[r]`
// and writes the packed `u32` result to `out[r]`. It shares the (ptr, len)
// plan model of `run_kv_migrate_copy` — the caller resolves each chunk's
// device address, so the KV bytes are checksummed in place on the GPU and only
// the small `out` array is copied back to host. See `simple/fletcher32.cu`.
//
// The arrays are device pointers:
//   src_ptrs:  device i64[n_records] — resolved chunk base addresses
//   byte_lens: device i64[n_records] — per-chunk byte counts
//   out:       device u32[n_records] — packed (sum2 << 16) | sum1 per chunk
//   stream:    cudaStream_t

use std::ffi::c_void;

extern "C" {
    pub fn run_fletcher32(
        src_ptrs: *const i64,
        byte_lens: *const i64,
        out: *mut u32,
        n_records: i32,
        stream: *mut c_void,
    );
}
