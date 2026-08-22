//! Manifold-Constrained Hyper-Connections (mHC).
//!
//! Mirrors `Block.hc_pre` / `hc_post` / `hc_head` and the `hc_split_sinkhorn` kernel in
//! `inference/model.py`. The residual stream carries `hc_mult` copies of the hidden
//! state; around each sub-block a learned mix reduces the copies to one input, and a
//! Sinkhorn-normalized combination matrix re-expands the block output back to `hc_mult`
//! copies. All math is done in f32, matching the reference.

use candle::{DType, Result, Tensor, D};
use candle_nn::ops::sigmoid;

use super::guard::expect_dtype;
use super::linear::QLinear;

/// The learned per-block hyper-connection parameters for one sub-block (attention or
/// FFN). `fn_w` is `[mix_hc, hc_mult*dim]`, `base` is `[mix_hc]`, `scale` is `[3]`,
/// where `mix_hc = (2 + hc_mult) * hc_mult`.
#[derive(Debug, Clone)]
pub struct HyperParams {
    pub fn_w: QLinear, // mixing projection; int8-KO on the engine path (dense if not KO-tileable)
    pub base: Tensor,
    pub scale: Tensor,
}

/// mHC operator: reduces the `hc_mult`-copy residual stream to a single block input
/// (`pre`) and recombines the block output with the residual (`post`).
#[derive(Debug, Clone)]
pub struct HyperConnection {
    hc_mult: usize,
    sinkhorn_iters: usize,
    eps: f64,
}

impl HyperConnection {
    pub fn new(hc_mult: usize, sinkhorn_iters: usize, eps: f64) -> Self {
        Self {
            hc_mult,
            sinkhorn_iters,
            eps,
        }
    }

    /// `mix_hc = (2 + hc_mult) * hc_mult`.
    pub fn mix_hc(&self) -> usize {
        (2 + self.hc_mult) * self.hc_mult
    }

    /// Expand an embedding `[b, s, d]` into the initial `hc_mult`-copy residual stream
    /// `[b, s, hc_mult, d]` (each copy identical), matching
    /// `h.unsqueeze(2).repeat(1, 1, hc_mult, 1)`.
    pub fn expand(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        x.reshape((b, s, 1, d))?
            .broadcast_as((b, s, self.hc_mult, d))?
            .contiguous()
    }

    /// `hc_pre`: reduce `[b, s, hc, d]` to a block input `[b, s, d]` and return the
    /// `post`/`comb` mixing tensors needed by [`Self::post`].
    ///
    /// On CUDA the rms-rsqrt, the sigmoid gate split, the Sinkhorn and the
    /// weighted residual reduction all fuse into ONE `mhc_pre_gates` launch (+
    /// the `fn_w` matmul), bit-exact to the eager path below which is the CPU
    /// reference. This kills ~25 tiny per-call launches that pure
    /// launch-overhead dominated at decode.
    pub fn pre(&self, x: &Tensor, p: &HyperParams) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, hc, d) = x.dims4()?;
        // Validated, not converted — see `Self::post`.
        expect_dtype(x, DType::F32, "mhc pre: residual stream")?;
        let xf = x.reshape((b, s, hc * d))?;

        #[cfg(feature = "cuda")]
        if matches!(x.device(), candle::Device::Cuda(_)) {
            let mixes_raw = p.fn_w.forward(&xf)?; // [b,s,mix_hc] (rsqrt folded in the kernel)
            // The whole of `hc_pre` is ONE launch: gates, sinkhorn and the
            // weighted residual reduction. `Self::sinkhorn` and the eager chain
            // below stay as the CPU reference paths the tests compare against.
            let (y, post, comb, _pre) =
                cuda_fused::pre_gates(&xf, &mixes_raw, p, hc, d, self.eps, self.sinkhorn_iters)?;
            return Ok((y, post, comb));
        }

        let rsqrt = self.rms_rsqrt(&xf)?; // [b,s,1]
        let mixes = p.fn_w.forward(&xf)?.broadcast_mul(&rsqrt)?; // [b,s,mix_hc]

        let (pre, post, comb) = self.split_sinkhorn(&mixes, p)?;
        // y = sum_c pre[c] * x[...,c,:]   -> [b,s,d]
        let y = pre.unsqueeze(D::Minus1)?.broadcast_mul(&x)?.sum(2)?;
        Ok((y, post, comb))
    }

    /// `hc_post`: recombine block output `[b, s, d]` with the residual `[b, s, hc, d]`
    /// via `new[j] = post[j] * out + Σ_i comb[i,j] * residual[i]`.
    pub fn post(
        &self,
        block_out: &Tensor,
        residual: &Tensor,
        post: &Tensor,
        comb: &Tensor,
    ) -> Result<Tensor> {
        // VALIDATED, not converted (invariant 1). The residual stream is F32
        // the whole way round the loop — `Self::post` returns F32, the attention
        // kernels emit F32 and the MoE emits F32 — so casting here only hid the
        // requirement; a producer that ever changed type would have bought a
        // silent full-tensor pass per layer instead of an error.
        expect_dtype(block_out, DType::F32, "mhc post: block output")?;
        expect_dtype(residual, DType::F32, "mhc post: residual stream")?;

        #[cfg(feature = "cuda")]
        if matches!(block_out.device(), candle::Device::Cuda(_)) {
            // Fused: `new[j,k] = post[j]·out[k] + Σ_i comb[i,j]·res[i,k]` in one
            // launch (was ~10 eager broadcast/sum ops). Bit-exact to the path below.
            return cuda_fused::post(block_out, residual, post, comb);
        }

        // post term: post[...,hc,1] * out[...,1,d] -> [b,s,hc,d]
        let post_term = post
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&block_out.unsqueeze(D::Minus2)?)?;
        // comb term: sum_i comb[i,j] * residual[i]  -> [b,s,j=hc,d]
        // comb[b,s,i,j,1] * residual[b,s,i,1,d] -> [b,s,i,j,d], sum over i (dim 2).
        let comb_term = comb
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&residual.unsqueeze(D::Minus2)?)?
            .sum(2)?;
        post_term + comb_term
    }

    /// `hc_head`: final reduction of the residual stream `[b, s, hc, d]` to `[b, s, d]`
    /// before the output norm / LM head. `fn_w` is `[hc_mult, hc_mult*dim]`,
    /// `base` is `[hc_mult]`, `scale` is `[1]`.
    pub fn head_reduce(&self, x: &Tensor, p: &HyperParams) -> Result<Tensor> {
        let (b, s, hc, d) = x.dims4()?;
        // Validated, not converted — see `Self::post`.
        expect_dtype(x, DType::F32, "mhc head_reduce: residual stream")?;
        let xf = x.reshape((b, s, hc * d))?;

        #[cfg(feature = "cuda")]
        if matches!(x.device(), candle::Device::Cuda(_)) {
            // ONE launch for the whole of `hc_head` — rms-rsqrt, the sigmoid
            // gate and the weighted reduction (+ the `fn_w` matmul), matching
            // what `pre`/`post` already do. The eager chain below stays as the
            // CPU reference the parity test compares against.
            let mixes_raw = p.fn_w.forward(&xf)?; // [b,s,hc] (rsqrt folded in the kernel)
            return cuda_fused::head_reduce(&xf, &mixes_raw, p, hc, d, self.eps);
        }

        let rsqrt = self.rms_rsqrt(&xf)?;
        let mixes = p.fn_w.forward(&xf)?.broadcast_mul(&rsqrt)?; // [b,s,hc]
        let scale = p.scale.to_dtype(candle::DType::F32)?;
        let base = p.base.to_dtype(candle::DType::F32)?;
        let pre = (sigmoid(&mixes.broadcast_mul(&scale)?.broadcast_add(&base)?)? + self.eps)?; // [b,s,hc]
        pre.unsqueeze(D::Minus1)?.broadcast_mul(&x)?.sum(2)
    }

    /// `rsqrt(mean(x², -1) + eps)` -> `[.., 1]`.
    fn rms_rsqrt(&self, xf: &Tensor) -> Result<Tensor> {
        let ms = xf.sqr()?.mean_keepdim(D::Minus1)?;
        (ms + self.eps)?.sqrt()?.recip()
    }

    /// Split the mix vector into `pre` (hc), `post` (hc), and the Sinkhorn-normalized
    /// `comb` (hc×hc). Matches `hc_split_sinkhorn`.
    fn split_sinkhorn(&self, mixes: &Tensor, p: &HyperParams) -> Result<(Tensor, Tensor, Tensor)> {
        let hc = self.hc_mult;
        let (b, s, _) = mixes.dims3()?;
        let scale = p.scale.to_dtype(candle::DType::F32)?;
        let base = p.base.to_dtype(candle::DType::F32)?;

        let s0 = scale.narrow(0, 0, 1)?;
        let s1 = scale.narrow(0, 1, 1)?;
        let s2 = scale.narrow(0, 2, 1)?;

        let pre_raw = mixes.narrow(D::Minus1, 0, hc)?;
        let post_raw = mixes.narrow(D::Minus1, hc, hc)?;
        let comb_raw = mixes.narrow(D::Minus1, 2 * hc, hc * hc)?;

        let base_pre = base.narrow(0, 0, hc)?;
        let base_post = base.narrow(0, hc, hc)?;
        let base_comb = base.narrow(0, 2 * hc, hc * hc)?;

        // pre = sigmoid(pre_raw*s0 + base_pre) + eps
        let pre = (sigmoid(&pre_raw.broadcast_mul(&s0)?.broadcast_add(&base_pre)?)? + self.eps)?;
        // post = 2*sigmoid(post_raw*s1 + base_post)
        let post = (sigmoid(&post_raw.broadcast_mul(&s1)?.broadcast_add(&base_post)?)? * 2.0)?;
        // comb = comb_raw*s2 + base_comb, reshaped to [b,s,hc,hc]
        let comb = comb_raw
            .broadcast_mul(&s2)?
            .broadcast_add(&base_comb)?
            .reshape((b, s, hc, hc))?;
        let comb = self.sinkhorn(&comb)?;
        Ok((pre, post, comb))
    }

    /// Sinkhorn normalization of `comb` `[b, s, hc, hc]` (rows = input copy `i`, cols = output copy
    /// `j`): softmax over `j`, then alternating column/row normalization. Runs as ONE fused kernel
    /// launch (`SinkhornOp`) instead of ~120 tiny host-orchestrated tensor ops per call; the op's
    /// `cpu_fwd` is the scalar reference for CPU tensors, `cuda_fwd` is the device kernel.
    fn sinkhorn(&self, comb: &Tensor) -> Result<Tensor> {
        comb.contiguous()?.apply_op1_no_bwd(&SinkhornOp {
            iters: self.sinkhorn_iters,
            eps: self.eps as f32,
        })
    }
}

/// Fused doubly-stochastic (Sinkhorn) normalization of a batch of `[hc, hc]` matrices (the trailing
/// two dims of the input). See `candle-kernels/src/simple/sinkhorn.cu`; the CPU and CUDA paths and
/// the kernel's isolation test all share one op order (softmax-over-cols + eps → col-norm →
/// `(iters-1)`×[row-norm, col-norm]).
struct SinkhornOp {
    iters: usize,
    eps: f32,
}

impl SinkhornOp {
    /// Scalar Sinkhorn over `n` contiguous `[hc, hc]` matrices — the CPU reference and the exact
    /// arithmetic the CUDA kernel mirrors.
    fn scalar(&self, inp: &[f32], n: usize, hc: usize) -> Vec<f32> {
        let (iters, eps) = (self.iters, self.eps);
        let mut out = vec![0f32; n * hc * hc];
        for mtx in 0..n {
            let a = &inp[mtx * hc * hc..(mtx + 1) * hc * hc];
            let c = &mut out[mtx * hc * hc..(mtx + 1) * hc * hc];
            for i in 0..hc {
                let mut m = f32::MIN;
                for j in 0..hc {
                    m = m.max(a[i * hc + j]);
                }
                let mut s = 0f32;
                for j in 0..hc {
                    let e = (a[i * hc + j] - m).exp();
                    c[i * hc + j] = e;
                    s += e;
                }
                for j in 0..hc {
                    c[i * hc + j] = c[i * hc + j] / s + eps;
                }
            }
            for j in 0..hc {
                let mut s = eps;
                for i in 0..hc {
                    s += c[i * hc + j];
                }
                for i in 0..hc {
                    c[i * hc + j] /= s;
                }
            }
            for _ in 0..iters.saturating_sub(1) {
                for i in 0..hc {
                    let mut s = eps;
                    for j in 0..hc {
                        s += c[i * hc + j];
                    }
                    for j in 0..hc {
                        c[i * hc + j] /= s;
                    }
                }
                for j in 0..hc {
                    let mut s = eps;
                    for i in 0..hc {
                        s += c[i * hc + j];
                    }
                    for i in 0..hc {
                        c[i * hc + j] /= s;
                    }
                }
            }
        }
        out
    }
}

impl candle::CustomOp1 for SinkhornOp {
    fn name(&self) -> &'static str {
        "sinkhorn"
    }

    fn cpu_fwd(
        &self,
        storage: &candle::CpuStorage,
        layout: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        let shape = layout.shape();
        let dims = shape.dims();
        let hc = dims[dims.len() - 1];
        let n = shape.elem_count() / (hc * hc);
        let inp = match storage {
            candle::CpuStorage::F32(s) => {
                let (o1, o2) = layout
                    .contiguous_offsets()
                    .ok_or_else(|| candle::Error::RequiresContiguous { op: "sinkhorn" }.bt())?;
                &s[o1..o2]
            }
            _ => candle::bail!("sinkhorn: expected F32, got {:?}", storage.dtype()),
        };
        Ok((
            candle::CpuStorage::F32(self.scalar(inp, n, hc)),
            shape.clone(),
        ))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::{Backing, CudaStorageSlice};
        use candle_kernels::simple::sinkhorn::run_sinkhorn_f32;

        let dev = storage.device().clone();
        let shape = layout.shape().clone();
        let dims = shape.dims();
        let hc = dims[dims.len() - 1];
        let n = shape.elem_count() / (hc * hc);
        let (o1, o2) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle::Error::RequiresContiguous { op: "sinkhorn" }.bt())?;
        let src = match &storage.slice {
            CudaStorageSlice::F32(s) => s,
            _ => candle::bail!("sinkhorn: expected F32 CUDA tensor"),
        };
        let out = unsafe { dev.alloc::<f32>(n * hc * hc)? };
        let stream = dev.cuda_stream();
        {
            let sv = src.slice(o1..o2);
            let (sp, _gs) = sv.device_ptr(&stream);
            let (op, _go) = out.device_ptr(&stream);
            unsafe {
                run_sinkhorn_f32(
                    sp as *const f32,
                    op as *mut f32,
                    n as i32,
                    hc as i32,
                    self.iters as i32,
                    self.eps,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }
        let dst = candle::CudaStorage {
            slice: CudaStorageSlice::F32(out),
            device: dev,
            backing: Backing::Owned,
        };
        Ok((dst, shape))
    }
}

/// Raw-FFI launchers for the fused mHC kernels (`simple/hyper_mhc.cu`). One
/// launch each replaces the tiny eager op chains in `pre`/`post`; the eager
/// paths in [`HyperConnection`] remain the CPU reference and the bit-exact
/// oracle for `fused_pre_post_matches_eager`.
#[cfg(feature = "cuda")]
mod cuda_fused {
    use super::super::guard::expect_dense_dtype;
    use super::HyperParams;
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::{DType, Device, Result, Storage, Tensor};
    use candle_kernels::simple::hyper_mhc::{
        run_mhc_head_reduce, run_mhc_post, run_mhc_pre_gates, MHC_MAX_HC,
    };

    /// Device pointer of a contiguous f32 CUDA tensor (extracted inline so the
    /// storage-guard and pointer-guard both live to the launch — matching the
    /// `gallery::sign_pack` pattern). `$p` binds the `u64` device address.
    macro_rules! cuda_f32_ptr {
        ($t:expr, $stream:expr, $s:ident, $p:ident, $g:ident) => {
            let ($s, _) = $t.storage_and_layout();
            let ($p, $g) = match &*$s {
                Storage::Cuda(c) => c.as_cuda_slice::<f32>()?.device_ptr($stream),
                _ => candle::bail!("mhc fused kernels require CUDA f32 storage"),
            };
        };
    }

    /// `hc_pre` stage 1 (rms-rsqrt · gate split): `(pre, post, comb_raw)`.
    /// The whole of `hc_pre` in ONE launch: returns `(y, post, comb, pre)` with
    /// **`comb` already sinkhorn-normalized** and **`y` already reduced**. Warp 0
    /// runs the sinkhorn while warps 1+ run the reduction, so the two overlap
    /// rather than costing two launches and two passes over the same row.
    /// `pre` is returned for the eager/reference comparison only.
    pub(super) fn pre_gates(
        xf: &Tensor,
        mixes_raw: &Tensor,
        p: &HyperParams,
        hc: usize,
        d: usize,
        eps: f64,
        sink_iters: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (b, s, _) = xf.dims3()?;
        let n = (b * s) as i32;
        let dev = match xf.device() {
            Device::Cuda(dd) => dd.clone(),
            _ => candle::bail!("mhc pre_gates requires CUDA"),
        };
        let stream = dev.cuda_stream();
        // Operands are VALIDATED, not rewritten (invariants 1 and 2). The kernel
        // indexes every one of these as `base + row * row_len`, so it needs a
        // dense f32 buffer — and it already gets one: `xf` is a reshape of the
        // F32 residual stream, `mixes_raw` is the `fn_w` matmul's output, and
        // `base`/`scale` are loaded through `dequant_f32`. Converting here bought
        // nothing and would have quietly absorbed a producer that changed.
        expect_dense_dtype(xf, DType::F32, "mhc pre_gates: xf")?;
        expect_dense_dtype(mixes_raw, DType::F32, "mhc pre_gates: mixes_raw")?;
        expect_dense_dtype(&p.base, DType::F32, "mhc pre_gates: base")?;
        expect_dense_dtype(&p.scale, DType::F32, "mhc pre_gates: scale")?;
        let (base, scale) = (&p.base, &p.scale);
        // All four are pure kernel outputs, so they are allocated uninitialised
        // rather than zeroed (hot-path invariant 6): the kernel writes every
        // element of `pre` and `post` (`for i in 0..hc`), every element of
        // `comb_raw` (`for e in 0..hc*hc`, then again from the sinkhorn), and
        // every element of `y` from the fused reduction. Zeroing them was a
        // second full-width memset on the exact bytes the kernel then stamps —
        // four per call, and this runs once per layer per step.
        let pre = Tensor::empty((b, s, hc), DType::F32, xf.device())?;
        let post = Tensor::empty((b, s, hc), DType::F32, xf.device())?;
        let comb_raw = Tensor::empty((b, s, hc, hc), DType::F32, xf.device())?;
        let y = Tensor::empty((b, s, d), DType::F32, xf.device())?;
        {
            cuda_f32_ptr!(xf, &stream, s_xf, p_xf, _g0);
            cuda_f32_ptr!(mixes_raw, &stream, s_mx, p_mx, _g1);
            cuda_f32_ptr!(base, &stream, s_ba, p_ba, _g2);
            cuda_f32_ptr!(scale, &stream, s_sc, p_sc, _g3);
            cuda_f32_ptr!(pre, &stream, s_pr, p_pr, _g4);
            cuda_f32_ptr!(post, &stream, s_po, p_po, _g5);
            cuda_f32_ptr!(comb_raw, &stream, s_cr, p_cr, _g6);
            cuda_f32_ptr!(y, &stream, s_y, p_y, _g7);
            unsafe {
                run_mhc_pre_gates(
                    p_xf as *const f32,
                    p_mx as *const f32,
                    p_ba as *const f32,
                    p_sc as *const f32,
                    p_pr as *mut f32,
                    p_po as *mut f32,
                    p_cr as *mut f32,
                    p_y as *mut f32,
                    n,
                    hc as i32,
                    d as i32,
                    eps as f32,
                    sink_iters as i32,
                    eps as f32,
                    stream.cu_stream() as *mut core::ffi::c_void,
                );
            }
        }
        Ok((y, post, comb_raw, pre))
    }

    /// `hc_head`: the whole final residual reduction in ONE launch → `[b,s,d]`.
    ///
    /// The eager form is five full passes over `[b,s,hc,d]` — `sqr`, the mean
    /// reduction, a `broadcast_mul` that also materialises a whole temp of that
    /// shape, and `sum(2)`, which reduces the second-to-last axis and therefore
    /// walks the temp with a `d`-element stride. Here the row is streamed once
    /// for the rsqrt and once (warm in L2) for the reduction, with no temp.
    pub(super) fn head_reduce(
        xf: &Tensor,
        mixes_raw: &Tensor,
        p: &HyperParams,
        hc: usize,
        d: usize,
        eps: f64,
    ) -> Result<Tensor> {
        let (b, s, _) = xf.dims3()?;
        let n = (b * s) as i32;
        let dev = match xf.device() {
            Device::Cuda(dd) => dd.clone(),
            _ => candle::bail!("mhc head_reduce requires CUDA"),
        };
        let stream = dev.cuda_stream();
        // The gate is held in a fixed `MHC_MAX_HC` shared array, and the
        // launcher returns `void` so it cannot report the overflow itself.
        if hc > MHC_MAX_HC {
            candle::bail!("mhc head_reduce: hc={hc} exceeds MHC_MAX_HC={MHC_MAX_HC}");
        }
        // Validated, not rewritten (invariants 1 and 2) — same reasoning as
        // `pre_gates`: every one of these is already dense F32 on the engine
        // path, so a conversion here would only have hidden a changed producer.
        expect_dense_dtype(xf, DType::F32, "mhc head_reduce: xf")?;
        expect_dense_dtype(mixes_raw, DType::F32, "mhc head_reduce: mixes_raw")?;
        expect_dense_dtype(&p.base, DType::F32, "mhc head_reduce: base")?;
        expect_dense_dtype(&p.scale, DType::F32, "mhc head_reduce: scale")?;
        let (base, scale) = (&p.base, &p.scale);
        // Pure kernel output — every element of `y` is stamped by the reduction
        // loop, so allocate uninitialised (invariant 6).
        let y = Tensor::empty((b, s, d), DType::F32, xf.device())?;
        {
            cuda_f32_ptr!(xf, &stream, s_xf, p_xf, _g0);
            cuda_f32_ptr!(mixes_raw, &stream, s_mx, p_mx, _g1);
            cuda_f32_ptr!(base, &stream, s_ba, p_ba, _g2);
            cuda_f32_ptr!(scale, &stream, s_sc, p_sc, _g3);
            cuda_f32_ptr!(y, &stream, s_y, p_y, _g4);
            unsafe {
                run_mhc_head_reduce(
                    p_xf as *const f32,
                    p_mx as *const f32,
                    p_ba as *const f32,
                    p_sc as *const f32,
                    p_y as *mut f32,
                    n,
                    hc as i32,
                    d as i32,
                    eps as f32,
                    stream.cu_stream() as *mut core::ffi::c_void,
                );
            }
        }
        Ok(y)
    }

    /// `hc_post` recombination: `new [b,s,hc,d]`.
    pub(super) fn post(
        block_out: &Tensor,
        residual: &Tensor,
        post: &Tensor,
        comb: &Tensor,
    ) -> Result<Tensor> {
        let (b, s, hc, d) = residual.dims4()?;
        let n = (b * s) as i32;
        let dev = match residual.device() {
            Device::Cuda(dd) => dd.clone(),
            _ => candle::bail!("mhc post requires CUDA"),
        };
        let stream = dev.cuda_stream();
        // Validated, not rewritten — `post` and `comb` come straight from
        // `pre_gates`, which produced them dense, and the other two are kernel
        // outputs of the attention/MoE block.
        expect_dense_dtype(block_out, DType::F32, "mhc post: block_out")?;
        expect_dense_dtype(residual, DType::F32, "mhc post: residual")?;
        expect_dense_dtype(post, DType::F32, "mhc post: post gates")?;
        expect_dense_dtype(comb, DType::F32, "mhc post: comb")?;
        // Pure kernel output: the grid is (n, hc) and each block writes its
        // whole `orow_j` row, so every element is stamped (invariant 6).
        let out = Tensor::empty((b, s, hc, d), DType::F32, residual.device())?;
        {
            cuda_f32_ptr!(block_out, &stream, s_bo, p_bo, _g0);
            cuda_f32_ptr!(residual, &stream, s_re, p_re, _g1);
            cuda_f32_ptr!(post, &stream, s_po, p_po, _g2);
            cuda_f32_ptr!(comb, &stream, s_cb, p_cb, _g3);
            cuda_f32_ptr!(out, &stream, s_ou, p_ou, _g4);
            unsafe {
                run_mhc_post(
                    p_bo as *const f32,
                    p_re as *const f32,
                    p_po as *const f32,
                    p_cb as *const f32,
                    p_ou as *mut f32,
                    n,
                    hc as i32,
                    d as i32,
                    stream.cu_stream() as *mut core::ffi::c_void,
                );
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, IndexOp, Tensor};

    /// Scalar transcription of `hc_split_sinkhorn` for one token, used as ground truth.
    fn ref_sinkhorn(comb_raw: &[f32], hc: usize, iters: usize, eps: f32) -> Vec<f32> {
        let mut c = vec![0f32; hc * hc];
        // softmax over j (last), + eps
        for i in 0..hc {
            let row = &comb_raw[i * hc..(i + 1) * hc];
            let m = row.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = row.iter().map(|&v| (v - m).exp()).collect();
            let s: f32 = exps.iter().sum();
            for j in 0..hc {
                c[i * hc + j] = exps[j] / s + eps;
            }
        }
        // col normalize (sum over i)
        let col_norm = |c: &mut [f32]| {
            for j in 0..hc {
                let mut s = 0f32;
                for i in 0..hc {
                    s += c[i * hc + j];
                }
                for i in 0..hc {
                    c[i * hc + j] /= s + eps;
                }
            }
        };
        let row_norm = |c: &mut [f32]| {
            for i in 0..hc {
                let mut s = 0f32;
                for j in 0..hc {
                    s += c[i * hc + j];
                }
                for j in 0..hc {
                    c[i * hc + j] /= s + eps;
                }
            }
        };
        col_norm(&mut c);
        for _ in 0..iters.saturating_sub(1) {
            row_norm(&mut c);
            col_norm(&mut c);
        }
        c
    }

    #[test]
    fn sinkhorn_matches_scalar_reference() -> Result<()> {
        let dev = Device::Cpu;
        let hc = 4;
        let iters = 20;
        let eps = 1e-6f32;
        // Deterministic pseudo-random comb inputs.
        let raw: Vec<f32> = (0..hc * hc)
            .map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.3)
            .collect();
        let comb = Tensor::from_vec(raw.clone(), (1, 1, hc, hc), &dev)?;
        let hcx = HyperConnection::new(hc, iters, eps as f64);
        let got = hcx.sinkhorn(&comb)?.flatten_all()?.to_vec1::<f32>()?;
        let want = ref_sinkhorn(&raw, hc, iters, eps);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "sinkhorn mismatch {g} vs {w}");
        }
        // Doubly-stochastic: rows and columns each sum ~1 (within eps drift).
        for i in 0..hc {
            let rs: f32 = (0..hc).map(|j| got[i * hc + j]).sum();
            assert!((rs - 1.0).abs() < 1e-2, "row {i} sum {rs}");
        }
        Ok(())
    }

    #[test]
    fn expand_and_post_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let hcx = HyperConnection::new(4, 20, 1e-6);
        let x = Tensor::randn(0f32, 1.0, (2, 3, 16), &dev)?;
        let expanded = hcx.expand(&x)?;
        assert_eq!(expanded.dims(), &[2, 3, 4, 16]);
        // Each copy identical to the source.
        let c0 = expanded.i((.., .., 0, ..))?;
        let diff = (c0 - &x)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6);
        Ok(())
    }

    /// The fused CUDA `hc_pre`/`hc_post` kernels reproduce the eager (CPU)
    /// reference within reduction-order tolerance — the correctness oracle for
    /// the launch-collapsing fusion in `cuda_fused`.
    #[cfg(feature = "cuda")]
    #[test]
    fn fused_pre_post_matches_eager() -> Result<()> {
        let cpu = Device::Cpu;
        let cuda = Device::new_cuda(0)?;
        let (hc, d, b, s) = (4usize, 48usize, 1usize, 3usize);
        let mix_hc = (2 + hc) * hc;
        let hcx = HyperConnection::new(hc, 20, 1e-6);

        // Deterministic pseudo-random fixtures (host, then mirrored to device).
        let fn_w = Tensor::randn(0f32, 1.0, (mix_hc, hc * d), &cpu)?;
        let base = Tensor::randn(0f32, 0.5, mix_hc, &cpu)?;
        let scale = Tensor::from_vec(vec![0.7f32, 1.1, 0.9], 3, &cpu)?;
        let x = Tensor::randn(0f32, 1.0, (b, s, hc, d), &cpu)?;
        let block_out = Tensor::randn(0f32, 1.0, (b, s, d), &cpu)?;

        let p_cpu = HyperParams {
            fn_w: fn_w.clone().into(),
            base: base.clone(),
            scale: scale.clone(),
        };
        let p_cuda = HyperParams {
            fn_w: fn_w.to_device(&cuda)?.into(),
            base: base.to_device(&cuda)?,
            scale: scale.to_device(&cuda)?,
        };
        let x_cuda = x.to_device(&cuda)?;

        // pre: eager (CPU) vs fused (CUDA).
        let (y_r, post_r, comb_r) = hcx.pre(&x, &p_cpu)?;
        let (y_c, post_c, comb_c) = hcx.pre(&x_cuda, &p_cuda)?;
        let maxdiff = |a: &Tensor, b: &Tensor| -> Result<f32> {
            let a = a.to_device(&cpu)?.flatten_all()?;
            let b = b.to_device(&cpu)?.flatten_all()?;
            (a - b)?.abs()?.max(0)?.to_scalar::<f32>()
        };
        assert!(maxdiff(&post_r, &post_c)? < 1e-4, "post gate mismatch");
        assert!(maxdiff(&comb_r, &comb_c)? < 1e-4, "comb mismatch");
        assert!(maxdiff(&y_r, &y_c)? < 1e-3, "pre reduce mismatch");

        // post: eager (CPU) vs fused (CUDA), fed each path's own pre outputs.
        let new_r = hcx.post(&block_out, &x, &post_r, &comb_r)?;
        let new_c = hcx.post(&block_out.to_device(&cuda)?, &x_cuda, &post_c, &comb_c)?;
        assert!(maxdiff(&new_r, &new_c)? < 2e-3, "post recombine mismatch");
        Ok(())
    }

    /// The fused CUDA `hc_head` reproduces the eager (CPU) reference within
    /// reduction-order tolerance.
    ///
    /// `hc_head` is `hc_pre` without the split, so its parameter shapes are the
    /// unsplit ones — `fn_w` is `[hc, hc*d]`, `base` is `[hc]` and `scale` is a
    /// single value — which is exactly what makes it a separate kernel rather
    /// than a mode of `mhc_pre_gates_kernel`.
    ///
    /// Tolerance matches the sibling gate at 2e-3: the kernel's rsqrt uses a
    /// float4 + warp-shuffle reduction whose lane→element assignment differs
    /// from candle's `sum`, so the last ULPs may differ. It stays deterministic
    /// (fixed order, no atomics).
    #[cfg(feature = "cuda")]
    #[test]
    fn fused_head_reduce_matches_eager() -> Result<()> {
        let cpu = Device::Cpu;
        let cuda = Device::new_cuda(0)?;
        let (hc, d, b, s) = (4usize, 48usize, 1usize, 3usize);
        let hcx = HyperConnection::new(hc, 20, 1e-6);

        let fn_w = Tensor::randn(0f32, 1.0, (hc, hc * d), &cpu)?;
        let base = Tensor::randn(0f32, 0.5, hc, &cpu)?;
        let scale = Tensor::from_vec(vec![0.7f32], 1, &cpu)?;
        let x = Tensor::randn(0f32, 1.0, (b, s, hc, d), &cpu)?;

        let p_cpu = HyperParams {
            fn_w: fn_w.clone().into(),
            base: base.clone(),
            scale: scale.clone(),
        };
        let p_cuda = HyperParams {
            fn_w: fn_w.to_device(&cuda)?.into(),
            base: base.to_device(&cuda)?,
            scale: scale.to_device(&cuda)?,
        };

        let y_r = hcx.head_reduce(&x, &p_cpu)?; // eager CPU reference
        let y_c = hcx.head_reduce(&x.to_device(&cuda)?, &p_cuda)?; // fused kernel
        assert_eq!(y_r.dims(), &[b, s, d], "eager shape");
        assert_eq!(y_c.dims(), &[b, s, d], "fused shape");

        let a = y_r.flatten_all()?;
        let bb = y_c.to_device(&cpu)?.flatten_all()?;
        let diff = (a - bb)?.abs()?.max(0)?.to_scalar::<f32>()?;
        assert!(diff < 2e-3, "head reduce mismatch: {diff}");
        Ok(())
    }

    /// **Isolation harness for the fused `mhc_pre_gates` (+ sinkhorn).**
    ///
    /// nsys put the mHC chain at 13% of decode GPU time, with `sinkhorn` alone
    /// at 5.4% — a one-thread-per-matrix kernel costing ~30 us for a few dozen
    /// flops, i.e. almost pure launch overhead. That launch is now folded into
    /// this kernel, and this harness is where the result gets profiled and tuned
    /// without a 152 GB model in the way.
    ///
    /// Shapes are the real decode ones: `n = b*s` is the wave width (16
    /// sessions × 1 token) and `hc*d` is the model dim.
    ///
    /// Run under Nsight Compute:
    /// ```text
    /// ncu --set full --kernel-name mhc_pre_gates_kernel \
    ///   target/release/deps/candle_transformers-*.exe \
    ///   --exact models::deepseek4::hyper::tests::bench_fused_pre_gates --ignored --nocapture
    /// ```
    /// **Isolation harness for `mhc_post`.** Same role as
    /// [`bench_fused_pre_gates`]: profile and tune without a 152 GB model in the
    /// way. nsys put this kernel at 24.7 us / 4.5% of decode GPU time, moving
    /// only ~1 MB at n=16 — i.e. ~40 GB/s against ~1.3 TB/s of HBM, so it is
    /// latency- and occupancy-bound rather than bandwidth-bound.
    ///
    /// ```text
    /// ncu --set full --kernel-name mhc_post_kernel \
    ///   target/release/deps/candle_transformers-*.exe bench_mhc_post --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn bench_mhc_post() -> Result<()> {
        const ITERS: usize = 500;
        let Ok(cuda) = Device::new_cuda(0) else {
            eprintln!("[skip] no CUDA device");
            return Ok(());
        };
        let (hc, d, n) = (4usize, 1792usize, 16usize);
        let hcx = HyperConnection::new(hc, 20, 1e-6);
        let block_out = Tensor::randn(0f32, 1.0, (1usize, n, d), &cuda)?;
        let residual = Tensor::randn(0f32, 1.0, (1usize, n, hc, d), &cuda)?;
        let post = Tensor::randn(0f32, 1.0, (1usize, n, hc), &cuda)?;
        let comb = Tensor::randn(0f32, 1.0, (1usize, n, hc, hc), &cuda)?;

        for _ in 0..20 {
            let _ = hcx.post(&block_out, &residual, &post, &comb)?;
        }
        cuda.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            let _ = hcx.post(&block_out, &residual, &post, &comb)?;
        }
        cuda.synchronize()?;
        eprintln!(
            "[mhc] post: {:.1} us/call (n={n}, hc={hc}, d={d})",
            t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64
        );
        Ok(())
    }

    #[test]
    #[ignore]
    fn bench_fused_pre_gates() -> Result<()> {
        const ITERS: usize = 500;
        let Ok(cuda) = Device::new_cuda(0) else {
            eprintln!("[skip] no CUDA device");
            return Ok(());
        };
        // hc*d = 7168 (model dim); n = 16 decode rows; 20 sinkhorn iterations.
        let (hc, d, n) = (4usize, 1792usize, 16usize);
        let mix_hc = (2 + hc) * hc;
        let hcx = HyperConnection::new(hc, 20, 1e-6);

        let p = HyperParams {
            fn_w: Tensor::randn(0f32, 1.0, (mix_hc, hc * d), &cuda)?.into(),
            base: Tensor::randn(0f32, 0.5, mix_hc, &cuda)?,
            scale: Tensor::from_vec(vec![0.7f32, 1.1, 0.9], 3, &cuda)?,
        };
        let x = Tensor::randn(0f32, 1.0, (1usize, n, hc, d), &cuda)?;
        let xf = x.reshape((1usize, n, hc * d))?;
        let mixes_raw = p.fn_w.forward(&xf)?;

        // Warm-up: first launch pays one-off module load and allocation.
        for _ in 0..20 {
            let _ = cuda_fused::pre_gates(&xf, &mixes_raw, &p, hc, d, hcx.eps, hcx.sinkhorn_iters)?;
        }
        cuda.synchronize()?;

        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            let _ = cuda_fused::pre_gates(&xf, &mixes_raw, &p, hc, d, hcx.eps, hcx.sinkhorn_iters)?;
        }
        cuda.synchronize()?;
        let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
        eprintln!(
            "[mhc] fused pre_gates+sinkhorn: {per_call_us:.1} us/call \
             (n={n}, hc={hc}, d={d}, iters={})",
            hcx.sinkhorn_iters
        );
        Ok(())
    }

    /// **Isolation harness for the fused `mhc_head_reduce`.** Same role as
    /// [`bench_fused_pre_gates`] and [`bench_mhc_post`]: profile and tune
    /// without a 152 GB model in the way.
    ///
    /// Shapes are the real ones — `hc*d = 7168` is the model dim, and `n` is
    /// the wave width. Unlike its siblings this kernel runs ONCE PER WAVE
    /// rather than once per layer, so `n` spans a much wider range: 16 at a
    /// decode wave, hundreds-to-thousands of rows on a bulk prefill. Both ends
    /// matter and they are different regimes — at n=16 the grid is 16 blocks on
    /// a ~100-SM card (occupancy-bound), while at large n it should approach
    /// bandwidth.
    ///
    /// ```text
    /// ncu --set full --kernel-name mhc_head_reduce_kernel \
    ///   target/release/deps/candle_transformers-*.exe \
    ///   --exact models::deepseek4::hyper::tests::bench_fused_head_reduce --ignored --nocapture
    /// ```
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn bench_fused_head_reduce() -> Result<()> {
        const ITERS: usize = 500;
        let Ok(cuda) = Device::new_cuda(0) else {
            eprintln!("[skip] no CUDA device");
            return Ok(());
        };
        let (hc, d) = (4usize, 1792usize); // hc*d = 7168 = model dim
        let hcx = HyperConnection::new(hc, 20, 1e-6);

        // `hc_head`'s unsplit parameter shapes: fn_w [hc, hc*d], base [hc], scale [1].
        let p = HyperParams {
            fn_w: Tensor::randn(0f32, 1.0, (hc, hc * d), &cuda)?.into(),
            base: Tensor::randn(0f32, 0.5, hc, &cuda)?,
            scale: Tensor::from_vec(vec![0.7f32], 1, &cuda)?,
        };

        for &n in &[16usize, 256, 2048] {
            let x = Tensor::randn(0f32, 1.0, (1usize, n, hc, d), &cuda)?;
            let xf = x.reshape((1usize, n, hc * d))?;
            let mixes_raw = p.fn_w.forward(&xf)?;

            // Warm-up: the first launch pays one-off module load and allocation.
            for _ in 0..20 {
                let _ = cuda_fused::head_reduce(&xf, &mixes_raw, &p, hc, d, hcx.eps)?;
            }
            cuda.synchronize()?;

            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                let _ = cuda_fused::head_reduce(&xf, &mixes_raw, &p, hc, d, hcx.eps)?;
            }
            cuda.synchronize()?;
            let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
            // Traffic: the row is read twice (rms, then the reduction) and `d`
            // floats are written per row. Reporting the implied bandwidth is
            // what separates "occupancy-bound" from "at the memory wall".
            let bytes = (2.0 * (n * hc * d) as f64 + (n * d) as f64) * 4.0;
            let gbs = bytes / (per_call_us * 1e-6) / 1e9;
            eprintln!(
                "[mhc] fused head_reduce: n={n:<5} {per_call_us:>7.1} us/call  \
                 {gbs:>7.1} GB/s  (hc={hc}, d={d})"
            );
        }
        Ok(())
    }
}
