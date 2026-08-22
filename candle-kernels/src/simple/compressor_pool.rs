// FFI binding for the fused compressor group pool.
//
// Collapses the seven eager launches of `Compressor::pool_and_norm`'s pooling
// half (candle's unfused `softmax` over a non-last dim — max/sub/exp/sum/div —
// plus a `broadcast_mul` and a `sum`) into one. Bit-exact to the eager chain;
// see `simple/compressor_pool.cu`.

use std::ffi::c_void;

extern "C" {
    /// `out[g, k] = Σ_p softmax_p(score[g, :, k]) · kv[g, p, k]`.
    ///   pool_kv/pool_score: device f32 `[groups, p, d]`, read through the
    ///     `*_s{g,p,d}` strides (in ELEMENTS) so honest views — a dim-0 narrow of
    ///     a fleet-wide pooled block, say — are consumed in place rather than
    ///     copied into a packed layout first.
    ///   out: device f32[groups * d], packed.
    #[allow(clippy::too_many_arguments)]
    pub fn run_compressor_pool(
        pool_kv: *const f32,
        pool_score: *const f32,
        out: *mut f32,
        groups: i32,
        p: i32,
        d: i32,
        kv_sg: i64,
        kv_sp: i64,
        kv_sd: i64,
        sc_sg: i64,
        sc_sp: i64,
        sc_sd: i64,
        stream: *mut c_void,
    );
}
