// FFI bindings for the KV band-pointer patch.
//
// A compaction relocates chunk slots; the device holds their addresses inside
// `KvHead` records, not gids. These rewrite those words in one launch from two
// sorted arrays the host fills while it rewrites the matching `HeadGids`. See
// `simple/kv_ptr_patch.cu` for why this is a scatter rather than a table search,
// and for the two conditions that make an in-place record write safe.

use std::ffi::c_void;

extern "C" {
    /// Write `vals[i]` to the 8-byte word at device address `addrs[i]`.
    ///
    /// Both arrays are device-resident, `n_words` long. The caller must have
    /// sorted them by address — one head's eight band pointers are 64 contiguous
    /// bytes, so sorted pairs from one record land in one or two sectors instead
    /// of eight scattered ones.
    ///
    /// **Only legal inside the arena window**, with no forward in flight: it
    /// writes records that a running kernel would otherwise be reading.
    pub fn run_kv_ptr_patch(addrs: *const u64, vals: *const u64, n_words: i32, stream: *mut c_void);

    /// Count words whose current contents differ from `vals[i]`, adding into
    /// `mismatches` (device `u32`, zeroed by the caller).
    ///
    /// The patch's own proof. A band pointer left stale does not fault — every
    /// address in the reservation is mapped, so it reads whatever now occupies the
    /// vacated slot and surfaces as a wrong number many layers later. "The patch
    /// ran" is therefore not evidence that it landed; this is.
    pub fn run_kv_ptr_verify(
        addrs: *const u64,
        vals: *const u64,
        n_words: i32,
        mismatches: *mut u32,
        stream: *mut c_void,
    );
}
