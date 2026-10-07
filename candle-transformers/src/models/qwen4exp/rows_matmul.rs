//! `x · wᵀ` in F32 for the QSA indexer's projections.
//!
//! At decode and verify width the indexer projects a handful of rows against a
//! `[n, 2560]` F32 weight, and the library GEMM runs that on a small-N kernel of
//! 16 or 64 blocks: 23 µs a call for 1.3–5.2 MB of weights, twice per attention
//! layer on every forward. A wave of up to [`F32_ROWS_MATMUL_MAX_ROWS`] rows runs
//! `simple/f32_rows_matmul.cu` instead — one block per output column, K cut
//! across its threads, every weight row's reads issued at once. A wider wave (a
//! prompt prefill) fills the card on the library GEMM and keeps it.
//!
//! Both are F32 with F32 accumulation; they differ in the order they add K, so
//! they agree to rounding, not to the bit. The kernel's order depends on the
//! launch shape alone, so a given wave's result is the same on every run.

use std::ffi::c_void;

use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::{DType, Device, Result, Storage, Tensor};
use candle_kernels::simple::f32_rows_matmul::{run_f32_rows_matmul, F32_ROWS_MATMUL_MAX_ROWS};

/// `x · wᵀ` for `x: [rows, k]` and `w: [n, k]`, both F32.
pub fn rows_matmul_t(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let (rows, k) = x.dims2()?;
    let (n, wk) = w.dims2()?;
    if wk != k {
        candle::bail!("rows_matmul_t: [{rows}, {k}] against a [{n}, {wk}] weight");
    }
    let Device::Cuda(dev) = x.device() else {
        return x.matmul(&w.t()?);
    };
    if rows == 0 || rows > F32_ROWS_MATMUL_MAX_ROWS {
        return x.matmul(&w.t()?);
    }
    if x.dtype() != DType::F32 || w.dtype() != DType::F32 {
        candle::bail!(
            "rows_matmul_t: F32 operands, got {:?} · {:?}",
            x.dtype(),
            w.dtype()
        );
    }
    if !x.is_contiguous() || !w.is_contiguous() || k % 4 != 0 {
        candle::bail!(
            "rows_matmul_t: dense rows of a multiple of 4 floats, got x {:?} stride {:?}, \
             w {:?} stride {:?}",
            x.dims(),
            x.stride(),
            w.dims(),
            w.stride()
        );
    }
    // Every element is written by the kernel.
    let out = x.empty_beside((rows, n), DType::F32)?;
    let stream = dev.cuda_stream();
    let ptr = |t: &Tensor, what: &str| -> Result<u64> {
        let (s, l) = t.storage_and_layout();
        let slice = match &*s {
            Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("rows_matmul_t: {what} must be CUDA"),
        }
        .slice(l.start_offset()..);
        let (p, _guard) = slice.device_ptr(&stream);
        if p % 16 != 0 {
            candle::bail!("rows_matmul_t: {what} at {p:#x} is not 16-byte aligned");
        }
        Ok(p)
    };
    let (xp, wp, op) = (ptr(x, "x")?, ptr(w, "w")?, ptr(&out, "out")?);
    candle::set_kernel_breadcrumb("run_f32_rows_matmul", file!(), line!());
    // SAFETY: `x` is `[rows, k]`, `w` is `[n, k]` and `out` is `[rows, n]`, all
    // dense F32 on this stream and 16-byte aligned, checked above.
    let launched = unsafe {
        run_f32_rows_matmul(
            xp as *const f32,
            wp as *const f32,
            op as *mut f32,
            rows as i32,
            n as i32,
            k as i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    if launched != 0 {
        candle::bail!("rows_matmul_t: the kernel refused [{rows}, {k}] · [{n}, {k}]ᵀ");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn device() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    /// The indexer's two shapes at every decode/verify width the kernel takes:
    /// the library GEMM's values to rounding, and the same bits on every run.
    #[test]
    fn few_rows_match_the_library_gemm_and_repeat_exactly() -> Result<()> {
        let Some(dev) = device() else {
            return Ok(());
        };
        let k = 2560;
        for n in [128usize, 512] {
            let w = Tensor::from_vec(lcg(n * k, n as u64), (n, k), &dev)?;
            for rows in [1usize, 2, 5, 9, 16] {
                let x = Tensor::from_vec(lcg(rows * k, 7 + rows as u64), (rows, k), &dev)?;
                let got = rows_matmul_t(&x, &w)?;
                let want = x.matmul(&w.t()?)?;
                let diff = (&got - &want)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                let scale = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                assert!(
                    diff <= 1e-5 * scale.max(1.0),
                    "[{rows}, {k}]·[{n}, {k}]ᵀ: off by {diff} at scale {scale}"
                );
                let again = rows_matmul_t(&x, &w)?;
                assert_eq!(
                    got.flatten_all()?.to_vec1::<f32>()?,
                    again.flatten_all()?.to_vec1::<f32>()?,
                    "[{rows}, {k}]·[{n}, {k}]ᵀ differs between two runs"
                );
            }
        }
        Ok(())
    }

    /// A row's result does not depend on the rows beside it: the same row alone
    /// and inside a wider wave come out with the same bits.
    #[test]
    fn a_row_is_independent_of_its_wave() -> Result<()> {
        let Some(dev) = device() else {
            return Ok(());
        };
        let (n, k) = (128usize, 2560usize);
        let w = Tensor::from_vec(lcg(n * k, 3), (n, k), &dev)?;
        let x = Tensor::from_vec(lcg(5 * k, 11), (5, k), &dev)?;
        let wave = rows_matmul_t(&x, &w)?;
        for r in 0..5 {
            let alone = rows_matmul_t(&x.narrow(0, r, 1)?.contiguous()?, &w)?;
            assert_eq!(
                alone.flatten_all()?.to_vec1::<f32>()?,
                wave.narrow(0, r, 1)?.flatten_all()?.to_vec1::<f32>()?,
                "row {r}"
            );
        }
        Ok(())
    }

    /// Past the kernel's row bound the library GEMM runs, bit for bit.
    #[test]
    fn a_wide_wave_is_the_library_gemm() -> Result<()> {
        let Some(dev) = device() else {
            return Ok(());
        };
        let (rows, n, k) = (F32_ROWS_MATMUL_MAX_ROWS + 1, 128usize, 2560usize);
        let w = Tensor::from_vec(lcg(n * k, 5), (n, k), &dev)?;
        let x = Tensor::from_vec(lcg(rows * k, 13), (rows, k), &dev)?;
        assert_eq!(
            rows_matmul_t(&x, &w)?.flatten_all()?.to_vec1::<f32>()?,
            x.matmul(&w.t()?)?.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }
}
