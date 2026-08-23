// FFI binding for the compressed-index expansion.
//
// Replaces two pageable uploads plus a five-launch tensor-op chain — the decode
// path's `comp_idx` construction — with ONE launch driven by a staged
// descriptor. See `simple/comp_idx.cu`.

use std::ffi::c_void;

/// u32 words per slot in the descriptor table, mirroring `COMP_IDX_SLOT_WORDS`
/// in `simple/comp_idx.cu`. Struct-of-arrays over `n` slots: the offsets occupy
/// `[0, n)` and the counts `[n, 2n)`.
pub const COMP_IDX_SLOT_WORDS: usize = 2;

extern "C" {
    /// Expand each slot's `{offset, count}` into its row of the compressed-index
    /// matrix, and republish the counts as a device array.
    ///
    ///   `idx[i][k] = k < count[i] ? offset[i] + k : u32::MAX`
    ///   `cnt[i]    = count[i]`
    ///
    ///   desc: device u32 `[COMP_IDX_SLOT_WORDS * n]` (layout above)
    ///   idx:  device u32 `[n * max_sel]`, fully written
    ///   cnt:  device u32 `[n]`, fully written
    pub fn run_comp_idx_build(
        desc: *const u32,
        idx: *mut u32,
        cnt: *mut u32,
        n: i32,
        max_sel: i32,
        stream: *mut c_void,
    );
}
