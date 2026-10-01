// FFI bindings for the KV content hash.
//
// One hash per slot over the K/V that slot's block tables actually name, built from
// a host-side descriptor table of band addresses. See `simple/kv_hash.cu` for why
// the fold is commutative and seeded, and `kv_cache::chunked::kv_integrity` for what
// the numbers are compared against.

use std::ffi::c_void;

extern "C" {
    /// Fold each band's bytes into `out[slot_of[b]]`.
    ///
    /// `ptrs`, `lens`, `slot_of` and `seeds` are device-resident and `n_bands` long;
    /// `out` is device-resident, one `u64` per slot, and **must be zeroed by the
    /// caller** — the kernel accumulates into it.
    ///
    /// `seeds[b]` must be **odd**: it multiplies the band's hash before the add, and
    /// an odd multiplier is invertible modulo 2^64, so no band can be folded away.
    /// Its purpose is to make the fold sensitive to *which* band contributed what,
    /// since the add itself is commutative and two bands holding identical bytes
    /// would otherwise be interchangeable — which is precisely the corruption this
    /// exists to detect.
    ///
    /// Safe to call between forwards. It only reads the arenas, so it needs no arena
    /// window; it does need the bands it is given to still be live, which is the
    /// caller's to guarantee.
    pub fn run_kv_hash(
        ptrs: *const i64,
        lens: *const i64,
        slot_of: *const i32,
        seeds: *const u64,
        n_bands: i32,
        out: *mut u64,
        stream: *mut c_void,
    );
}
