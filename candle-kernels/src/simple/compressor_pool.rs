// FFI binding for the fused compressor group pool.
//
// Collapses the eight eager launches of `Compressor::pool_and_norm`'s pooling
// half (the `ape` `broadcast_add`, candle's unfused `softmax` over a non-last
// dim — max/sub/exp/sum/div — plus a `broadcast_mul` and a `sum`) into one, and
// takes its pooling rows as a DESCRIPTOR TABLE so the caller never has to `cat`
// them into a dense block first. See `simple/compressor_pool.cu`.

use std::ffi::c_void;

/// i64 words per group in part 1 of the descriptor table, mirroring
/// `POOL_GROUP_WORDS` in `simple/compressor_pool.cu`. Part 1 is
/// struct-of-arrays with stride `groups`: word `w` of group `g` lives at
/// `desc[w * groups + g]`.
///
/// | word | meaning                          |
/// |------|----------------------------------|
/// | 0    | index of the group's first segment |
/// | 1    | how many segments the group has   |
pub const POOL_GROUP_WORDS: usize = 2;

/// i64 words per segment in part 2 of the descriptor table, mirroring
/// `POOL_SEG_WORDS` in `simple/compressor_pool.cu`. Part 2 starts at
/// `desc[POOL_GROUP_WORDS * groups]` and is array-of-structs: segment `s`
/// occupies `POOL_SEG_WORDS` consecutive words.
///
/// | word | meaning                                  |
/// |------|------------------------------------------|
/// | 0    | `kv` base pointer                        |
/// | 1    | `kv` row stride (elements)               |
/// | 2    | `score` base pointer                     |
/// | 3    | `score` row stride                       |
/// | 4    | `ape` base pointer, or 0 for no bias     |
/// | 5    | `ape` row stride                         |
/// | 6    | rows this segment contributes            |
///
/// The channel stride is 1 in every operand.
pub const POOL_SEG_WORDS: usize = 7;

extern "C" {
    /// `out[g, k] = Σ_p softmax_p(score_g[p, k] + ape[p, k]) · kv_g[p, k]`, where
    /// group `g`'s pooling rows are its segments' rows in order, each read in
    /// place through the base pointer and row stride in `desc`.
    ///   desc: device i64, `POOL_GROUP_WORDS * groups` words followed by
    ///         `POOL_SEG_WORDS` per segment (layout above)
    ///   out:  device f32 `[groups * d]`, packed
    pub fn run_compressor_pool(
        desc: *const i64,
        out: *mut f32,
        groups: i32,
        d: i32,
        stream: *mut c_void,
    );
}
