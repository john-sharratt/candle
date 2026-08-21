// FFI bindings for the fused mHC (manifold-constrained hyper-connection) kernels.
//
// These collapse the remaining ~50 tiny eager tensor ops per `hc_pre`/`hc_post`
// call (rms-rsqrt, the sigmoid gate split, the weighted residual reductions)
// into three launches, the same way `sinkhorn.cu` fused the combine-matrix
// normalization. Semantics match `hyper.rs::HyperConnection` exactly; see
// `simple/hyper_mhc.cu`.

use std::ffi::c_void;

extern "C" {
    /// `hc_pre` stage 1: rms-rsqrt · sigmoid gate split.
    ///   xf:        device f32[n * hc * d]   (pre-reshape input, for the rms scale)
    ///   mixes_raw: device f32[n * mix_hc]   (fn_w(xf), mix_hc = (2+hc)*hc)
    ///   base:      device f32[mix_hc]; scale: device f32[3] = (s0,s1,s2)
    ///   pre/post:  device f32[n * hc] (out); comb_raw: device f32[n * hc * hc] (out, → sinkhorn)
    pub fn run_mhc_pre_gates(
        xf: *const f32,
        mixes_raw: *const f32,
        base: *const f32,
        scale: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb_raw: *mut f32,
        // `hc_pre` stage 2 is fused in too: y[n, d] = sum_c pre[c] * x[c, k],
        // computed by warps 1+ while warp 0 runs the sinkhorn.
        y: *mut f32,
        n: i32,
        hc: i32,
        d: i32,
        eps: f32,
        // Sinkhorn is folded into this kernel: `comb_raw` comes back NORMALIZED,
        // so no separate `run_sinkhorn_f32` launch is needed. `sink_iters <= 0`
        // leaves the raw affine result in place.
        sink_iters: i32,
        sink_eps: f32,
        stream: *mut c_void,
    );

    /// `hc_post`: `new[j,k] = post[j]·block_out[k] + Σ_i comb[i,j]·residual[i,k]`.
    ///   block_out: device f32[n * d]; residual: device f32[n * hc * d]
    ///   post: device f32[n * hc]; comb: device f32[n * hc * hc]; out: device f32[n * hc * d]
    pub fn run_mhc_post(
        block_out: *const f32,
        residual: *const f32,
        post: *const f32,
        comb: *const f32,
        out: *mut f32,
        n: i32,
        hc: i32,
        d: i32,
        stream: *mut c_void,
    );
}
