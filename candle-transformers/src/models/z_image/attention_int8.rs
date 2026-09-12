//! The int8 attention path: quantize, then one fused kernel.
//!
//! Rust side of `candle-kernels/src/dit-attn/dit_attn_int8.cu`, which is where
//! the quantization grid and the reasoning behind it live. This file is the
//! plumbing — allocation, pointer extraction, launch — and the micro harness
//! that measures and checks it against [`super::attention::reference`].
//!
//! # What crosses the boundary
//!
//! Q and K go down as int8 with a f32 scale per row; V goes down mean-centred,
//! scaled per dim, and **transposed** to `[dim, token]`, which is the layout the
//! P·V product's B operand reads. There is no `I8` dtype in candle, so the
//! quantized operands are held in `U8` tensors used as byte buffers — the
//! kernels read them as `int8_t*` and nothing on this side ever interprets
//! them numerically.

use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::{DType, Device, Result, Storage, Tensor};
use candle_kernels::dit_attn::{run_dit_attn_int8_bf16, run_dit_quant_rows_bf16, v_stride};

use super::attention::scale_of;

/// Device pointer to a contiguous tensor's first element.
///
/// Every operand here is freshly allocated or explicitly made contiguous, so the
/// layout's start offset is the only thing that can differ from zero and the
/// stride question does not arise.
macro_rules! dev_ptr {
    ($t:expr, $ty:ty, $stream:expr, $what:expr) => {{
        let (storage, layout) = $t.storage_and_layout();
        let slice = match &*storage {
            Storage::Cuda(c) => c.as_cuda_slice::<$ty>()?,
            _ => candle::bail!(concat!($what, " must be a CUDA tensor")),
        }
        .slice(layout.start_offset()..);
        let (ptr, _guard) = slice.device_ptr($stream);
        ptr
    }};
}

/// Quantized operands, ready for the attention kernel.
///
/// Held together because their lifetimes are: the kernel reads all seven
/// buffers, and a caller that dropped one early would be handing it a pointer
/// into freed memory.
pub struct Quantized {
    q8: Tensor,
    qs: Tensor,
    k8: Tensor,
    ks: Tensor,
    v8: Tensor,
    vs: Tensor,
    vmean: Tensor,
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
}

/// Quantize `[b, h, s, d]` bf16 operands onto the kernel's grid.
///
/// `q` is scaled by `1/√d` here rather than by the caller, so the scale lives
/// with the quantization that has to see it: folding it into `q` before the row
/// amax is what keeps the int8 range centred on the values the matmul will
/// actually produce.
pub fn quantize(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Quantized> {
    let (batch, heads, seq, head_dim) = q.dims4()?;
    let dev = q.device().clone();
    let Device::Cuda(cuda) = &dev else {
        candle::bail!("int8 attention is a CUDA path");
    };
    if q.dtype() != DType::BF16 {
        candle::bail!("int8 attention expects bf16 operands, got {:?}", q.dtype());
    }
    let stream = cuda.cuda_stream();
    let raw = stream.cu_stream() as *mut core::ffi::c_void;

    let q = (q * scale_of(head_dim))?.contiguous()?;
    let k = k.contiguous()?;
    // **V is transposed first, and that is what makes its quantizer coalesced.**
    // The PV MMA needs `[dim, token]` regardless, so the transpose is not a cost
    // this introduces — it is a cost this *moves*. Doing it here turns "reduce
    // down a column with a `head_dim`-strided read", where every lane lands in
    // its own 32-byte sector, into "reduce along a row", and the per-dim scale
    // becomes a per-row one so V shares Q and K's kernel.
    let v = v.transpose(2, 3)?.contiguous()?;
    let rows = batch * heads * seq;
    let bh = batch * heads;

    let vst = v_stride(seq);
    let q8 = Tensor::zeros(rows * head_dim, DType::U8, &dev)?;
    let k8 = Tensor::zeros(rows * head_dim, DType::U8, &dev)?;
    // Padded rows, so the kernel's 16-byte V reads are aligned at any `seq`.
    let v8 = Tensor::zeros(bh * head_dim * vst, DType::U8, &dev)?;
    let qs = Tensor::zeros(rows, DType::F32, &dev)?;
    let ks = Tensor::zeros(rows, DType::F32, &dev)?;
    let vs = Tensor::zeros(bh * head_dim, DType::F32, &dev)?;
    let vmean = Tensor::zeros(bh * head_dim, DType::F32, &dev)?;

    unsafe {
        let (qp, kp, vp) = (
            dev_ptr!(q, half::bf16, &stream, "q"),
            dev_ptr!(k, half::bf16, &stream, "k"),
            dev_ptr!(v, half::bf16, &stream, "v"),
        );
        let (q8p, k8p, v8p) = (
            dev_ptr!(q8, u8, &stream, "q8"),
            dev_ptr!(k8, u8, &stream, "k8"),
            dev_ptr!(v8, u8, &stream, "v8"),
        );
        let (qsp, ksp) = (
            dev_ptr!(qs, f32, &stream, "qs"),
            dev_ptr!(ks, f32, &stream, "ks"),
        );
        let (vsp, vmp) = (
            dev_ptr!(vs, f32, &stream, "vs"),
            dev_ptr!(vmean, f32, &stream, "vmean"),
        );
        candle::set_kernel_breadcrumb("run_dit_quant_rows_bf16", file!(), line!());
        // Q and K: a row is one head's `head_dim` for one token, uncentred.
        for (src, dst, sc) in [(qp, q8p, qsp), (kp, k8p, ksp)] {
            run_dit_quant_rows_bf16(
                src as *const _,
                dst as *mut _,
                sc as *mut _,
                std::ptr::null_mut(),
                rows as i32,
                head_dim as i32,
                head_dim as i32,
                0,
                raw,
            );
        }
        // V, already transposed: a row is one dim across the whole sequence,
        // centred, and padded out to the 16-byte-aligned stride.
        run_dit_quant_rows_bf16(
            vp as *const _,
            v8p as *mut _,
            vsp as *mut _,
            vmp as *mut _,
            (bh * head_dim) as i32,
            seq as i32,
            vst as i32,
            1,
            raw,
        );
    }

    Ok(Quantized {
        q8,
        qs,
        k8,
        ks,
        v8,
        vs,
        vmean,
        batch,
        heads,
        seq,
        head_dim,
    })
}

impl Quantized {
    /// Run the fused attention, returning `[b, h, s, d]` bf16.
    pub fn attend(&self, device: &Device) -> Result<Tensor> {
        let Device::Cuda(cuda) = device else {
            candle::bail!("int8 attention is a CUDA path");
        };
        let stream = cuda.cuda_stream();
        let raw = stream.cu_stream() as *mut core::ffi::c_void;
        let out = Tensor::zeros(
            (self.batch, self.heads, self.seq, self.head_dim),
            DType::BF16,
            device,
        )?;
        unsafe {
            let q8 = dev_ptr!(self.q8, u8, &stream, "q8");
            let qs = dev_ptr!(self.qs, f32, &stream, "qs");
            let k8 = dev_ptr!(self.k8, u8, &stream, "k8");
            let ks = dev_ptr!(self.ks, f32, &stream, "ks");
            let v8 = dev_ptr!(self.v8, u8, &stream, "v8");
            let vs = dev_ptr!(self.vs, f32, &stream, "vs");
            let vm = dev_ptr!(self.vmean, f32, &stream, "vmean");
            let o = dev_ptr!(out, half::bf16, &stream, "out");
            candle::set_kernel_breadcrumb("run_dit_attn_int8_bf16", file!(), line!());
            run_dit_attn_int8_bf16(
                q8 as *const _,
                qs as *const _,
                k8 as *const _,
                ks as *const _,
                v8 as *const _,
                vs as *const _,
                vm as *const _,
                o as *mut _,
                self.batch as i32,
                self.heads as i32,
                self.seq as i32,
                self.head_dim as i32,
                v_stride(self.seq) as i32,
                raw,
            );
        }
        Ok(out)
    }
}

/// Quantize and attend in one call — `[b, h, s, d]` bf16 in and out.
///
/// `q` is **not** pre-scaled: the `1/√d` is applied inside [`quantize`].
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    quantize(q, k, v)?.attend(q.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::z_image::attention::{reference, AttnShape};

    const CAP: usize = 32;

    fn operands(s: AttnShape, dev: &Device) -> Result<(Tensor, Tensor, Tensor)> {
        let dims = (1, s.heads, s.seq, s.head_dim);
        let mk = || Tensor::randn(0f32, 1f32, dims, dev)?.to_dtype(DType::BF16);
        Ok((mk()?, mk()?, mk()?))
    }

    /// Relative L2 of the kernel against the f32 oracle, at one shape.
    fn against_oracle(s: AttnShape, dev: &Device) -> Result<f32> {
        let (q, k, v) = operands(s, dev)?;
        let qs = (q.to_dtype(DType::F32)? * scale_of(s.head_dim))?;
        let want =
            reference(&qs, &k.to_dtype(DType::F32)?, &v.to_dtype(DType::F32)?)?.flatten_all()?;
        let got = attention(&q, &k, &v)?.to_dtype(DType::F32)?.flatten_all()?;
        let num = (&got - &want)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let den = want.sqr()?.sum_all()?.to_scalar::<f32>()?;
        Ok((num / den).sqrt())
    }

    /// **The kernel against the oracle, at the shapes the model runs.**
    ///
    /// The bound is the simulation's measured number with headroom: the grid was
    /// measured at ~2–3% rel_l2 on synthetic operands before any of this was
    /// written, so a kernel landing there is doing the arithmetic it was designed
    /// to. A kernel with a wrong fragment layout does not land near it — it lands
    /// at 100%, because a mismatched tensor-core contract produces noise rather
    /// than a slightly worse answer.
    #[test]
    fn the_kernel_matches_the_oracle() -> Result<()> {
        let dev = Device::cuda_if_available(0)?;
        if !dev.is_cuda() {
            return Ok(());
        }
        for side in [512usize, 1024] {
            let s = AttnShape::turbo(side, CAP);
            let rel = against_oracle(s, &dev)?;
            println!(
                "int8 kernel vs f32 oracle, {side}×{side} (seq {}): rel_l2 = {rel:.4}",
                s.seq
            );
            assert!(
                rel < 0.06,
                "int8 kernel is {rel} off the oracle at seq {}",
                s.seq
            );
        }
        Ok(())
    }

    /// **A sequence that is not a whole number of tiles.** The tail is where an
    /// attention kernel's masking is wrong without being obviously wrong: a key
    /// past the end contributes `exp(-inf) = 0` and a row past the end is never
    /// stored, and getting either subtly wrong shifts one tile's softmax rather
    /// than producing garbage. 4128 is not a multiple of the 64-key tile, and
    /// neither is 1056 — but a deliberately awkward one is worth its own row.
    #[test]
    fn a_ragged_tail_is_handled() -> Result<()> {
        let dev = Device::cuda_if_available(0)?;
        if !dev.is_cuda() {
            return Ok(());
        }
        for seq in [17usize, 65, 100, 1056, 4128] {
            let s = AttnShape {
                seq,
                heads: 4,
                head_dim: 128,
            };
            let rel = against_oracle(s, &dev)?;
            println!("  ragged seq {seq}: rel_l2 = {rel:.4}");
            assert!(rel < 0.06, "seq {seq} is {rel} off the oracle");
        }
        Ok(())
    }

    /// **The micro harness.** What the kernel costs against the bf16 path it
    /// replaces, at the shapes the model runs, with the quantization counted
    /// separately — it is a real cost the bf16 path does not pay, and hiding it
    /// inside the attention number would flatter the comparison.
    ///
    /// Asserts nothing; it exists to be read beside `attention_baseline_at_z_image_size`.
    #[test]
    #[ignore = "GPU benchmark: int8 attention against the bf16 baseline; run with --ignored --nocapture"]
    fn int8_attention_against_the_baseline() -> Result<()> {
        use crate::models::z_image::attention::{banded, ATTN_Q_BAND};

        let dev = Device::cuda_if_available(0)?;
        if !dev.is_cuda() {
            println!("no CUDA device; nothing to measure");
            return Ok(());
        }
        const WARMUP: usize = 3;
        const ITERS: usize = 10;
        const BLOCKS: usize = 32;

        println!(
            "{:<10} {:>6} {:>10} {:>9} {:>9} {:>9} {:>10} {:>9}",
            "image",
            "seq",
            "bf16 ms",
            "quant ms",
            "of which vT",
            "attn ms",
            "int8 TOP/s",
            "speedup"
        );
        for side in [512usize, 1024] {
            let s = AttnShape::turbo(side, CAP);
            let (q, k, v) = operands(s, &dev)?;
            let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
                for _ in 0..WARMUP {
                    f()?;
                }
                dev.synchronize()?;
                let t0 = std::time::Instant::now();
                for _ in 0..ITERS {
                    f()?;
                }
                dev.synchronize()?;
                Ok(t0.elapsed().as_secs_f64() / ITERS as f64)
            };

            let qb = (q.clone() * scale_of(s.head_dim))?;
            let kt = k.transpose(2, 3)?.contiguous()?;
            let bf16 = time(&|| {
                banded(&qb, &kt, &v, ATTN_Q_BAND)?;
                Ok(())
            })?;
            let quant = time(&|| {
                quantize(&q, &k, &v)?;
                Ok(())
            })?;
            // V's transpose on its own. It is inside `quantize`, and it is the
            // part that is candle's generic strided copy rather than one of
            // these kernels — worth knowing separately before optimising either.
            let vt = time(&|| {
                v.transpose(2, 3)?.contiguous()?;
                Ok(())
            })?;
            let qz = quantize(&q, &k, &v)?;
            let attn = time(&|| {
                qz.attend(&dev)?;
                Ok(())
            })?;

            println!(
                "{:<10} {:>6} {:>10.3} {:>9.3} {:>11.3} {:>9.3} {:>10.1} {:>8.2}x",
                format!("{side}×{side}"),
                s.seq,
                bf16 * 1e3,
                quant * 1e3,
                vt * 1e3,
                attn * 1e3,
                s.flops() as f64 / attn / 1e12,
                bf16 / (attn + quant),
            );
            let _ = BLOCKS;
        }
        Ok(())
    }
}
