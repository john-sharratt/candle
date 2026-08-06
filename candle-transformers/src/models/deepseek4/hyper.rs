//! Manifold-Constrained Hyper-Connections (mHC).
//!
//! Mirrors `Block.hc_pre` / `hc_post` / `hc_head` and the `hc_split_sinkhorn` kernel in
//! `inference/model.py`. The residual stream carries `hc_mult` copies of the hidden
//! state; around each sub-block a learned mix reduces the copies to one input, and a
//! Sinkhorn-normalized combination matrix re-expands the block output back to `hc_mult`
//! copies. All math is done in f32, matching the reference.

use candle::{Result, Tensor, D};
use candle_nn::ops::sigmoid;

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
    pub fn pre(&self, x: &Tensor, p: &HyperParams) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, hc, d) = x.dims4()?;
        let x = x.to_dtype(candle::DType::F32)?;
        let xf = x.reshape((b, s, hc * d))?;
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
        let block_out = block_out.to_dtype(candle::DType::F32)?;
        let residual = residual.to_dtype(candle::DType::F32)?;
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
        let x = x.to_dtype(candle::DType::F32)?;
        let xf = x.reshape((b, s, hc * d))?;
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
        use candle::cuda_backend::CudaStorageSlice;
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
        };
        Ok((dst, shape))
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
}
