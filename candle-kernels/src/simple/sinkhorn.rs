// FFI binding for the fused Sinkhorn (doubly-stochastic) normalization kernel.
//
// `run_sinkhorn_f32` normalizes `n` small `[hc, hc]` matrices in one launch,
// replacing the ~120 host-orchestrated tiny tensor ops the mHC combine-matrix
// normalization ran per call. Semantics match `hyper.rs::sinkhorn` exactly
// (softmax over cols + eps → col-norm → (iters-1)×[row-norm, col-norm]); see
// `simple/sinkhorn.cu`.
//
//   inp/out: device f32[n * hc * hc] row-major (may alias for in-place)
//   n:       number of matrices
//   hc:      matrix side (≤ 16)
//   iters:   Sinkhorn iterations (matches `hc_sinkhorn_iters`)
//   eps:     stabilizing epsilon added at each normalization
//   stream:  cudaStream_t

use std::ffi::c_void;

extern "C" {
    pub fn run_sinkhorn_f32(
        inp: *const f32,
        out: *mut f32,
        n: i32,
        hc: i32,
        iters: i32,
        eps: f32,
        stream: *mut c_void,
    );
}
