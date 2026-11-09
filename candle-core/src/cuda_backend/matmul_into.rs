//! An F32 matrix product written into a buffer the caller already holds.
//!
//! [`crate::Tensor::matmul`] allocates its output, from the left operand's arena.
//! A caller that computes a product in tiles and reduces each tile before the
//! next has no use for a fresh output per tile: in a bump arena those never
//! come back within the phase, so the tiles add up to the whole product the
//! tiling exists to avoid. This writes each tile into the same buffer instead;
//! work queued on one stream runs in order, so a tile's reader has finished
//! with the buffer before the next tile's product overwrites it.

use cudarc::driver::DevicePtr;

use super::{gemm_config, gemm_strided_batched_f32, CudaStorageSlice, WrapErr};
use crate::{DType, Result, Storage, Tensor};

/// `lhs @ rhs`, `[m, k]` by `[k, n]`, written into `out` — a contiguous F32
/// tensor of exactly `m × n` elements, read as `[m, n]`.
///
/// `lhs` and `rhs` may be any layout [`Tensor::matmul`] takes (a transposed
/// view included); `out` is overwritten, never read.
pub fn matmul_f32_into(lhs: &Tensor, rhs: &Tensor, out: &Tensor) -> Result<()> {
    let (m, k) = lhs.dims2()?;
    let (k2, n) = rhs.dims2()?;
    if k != k2 {
        crate::bail!("matmul into: [{m}, {k}] by [{k2}, {n}]");
    }
    if out.dtype() != DType::F32 || lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 {
        crate::bail!(
            "matmul into: F32 only, got {:?} by {:?} into {:?}",
            lhs.dtype(),
            rhs.dtype(),
            out.dtype()
        );
    }
    if out.elem_count() != m * n || !out.is_contiguous() {
        crate::bail!(
            "matmul into: the output must be a contiguous [{m}, {n}], got {:?}",
            out.dims()
        );
    }
    if m == 0 || n == 0 || k == 0 {
        crate::bail!("matmul into: a degenerate product [{m}, {k}] by [{k}, {n}]");
    }
    let (ls, ll) = lhs.storage_and_layout();
    let (rs, rl) = rhs.storage_and_layout();
    let (os, ol) = out.storage_and_layout();
    let (Storage::Cuda(ls), Storage::Cuda(rs), Storage::Cuda(os)) = (&*ls, &*rs, &*os) else {
        crate::bail!("matmul into: CUDA operands only");
    };
    let (CudaStorageSlice::F32(a), CudaStorageSlice::F32(b), CudaStorageSlice::F32(c)) =
        (&ls.slice, &rs.slice, &os.slice)
    else {
        crate::bail!("matmul into: F32 storage only");
    };
    let dev = &ls.device;
    let stream = dev.cuda_stream();
    let lhs_v = a.slice(ll.start_offset()..);
    let rhs_v = b.slice(rl.start_offset()..);
    let cfg = gemm_config(1f32, 0f32, (1, m, n, k), ll, rl)?;
    let at = c.device_ptr(&stream).0 + (ol.start_offset() * 4) as u64;
    // SAFETY: `at` names the `m × n` F32 elements of `out`, which `out` keeps
    // alive and which this call is documented to overwrite. The view is leaked
    // below rather than dropped, so it frees nothing it does not own.
    let mut view = unsafe { stream.upgrade_device_ptr::<f32>(at, m * n) };
    let known_aligned =
        (ll.start_offset() * 4).is_multiple_of(16) && (rl.start_offset() * 4).is_multiple_of(16);
    // SAFETY: the operands are live views of their storages, the output a view
    // of `out`, all on this device's stream; the configuration is derived from
    // their layouts exactly as `Tensor::matmul`'s is.
    let done = unsafe {
        gemm_strided_batched_f32(
            &*dev.cublas()?,
            cfg,
            &rhs_v,
            &lhs_v,
            &mut view,
            known_aligned,
        )
    };
    view.leak();
    done.w()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    /// The product lands in the buffer exactly as `matmul` computes it — a
    /// transposed right operand included — and a second product into the same
    /// buffer replaces the first.
    #[test]
    fn writes_the_product_into_the_buffer() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let a = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &dev)?;
        let bt = Tensor::new(
            &[[1f32, 0., 1.], [0., 1., 0.], [2., 2., 2.], [1., 1., 1.]],
            &dev,
        )?;
        let out = Tensor::zeros((2, 4), DType::F32, &dev)?;
        matmul_f32_into(&a, &bt.t()?, &out)?;
        assert_eq!(
            out.to_vec2::<f32>()?,
            vec![vec![4., 2., 12., 6.], vec![10., 5., 30., 15.]]
        );
        assert_eq!(out.to_vec2::<f32>()?, a.matmul(&bt.t()?)?.to_vec2::<f32>()?);
        let a2 = Tensor::new(&[[0f32, 0., 1.], [1., 0., 0.]], &dev)?;
        matmul_f32_into(&a2, &bt.t()?, &out)?;
        assert_eq!(
            out.to_vec2::<f32>()?,
            vec![vec![1., 0., 2., 1.], vec![1., 0., 2., 1.]]
        );
        Ok(())
    }

    /// At the scorer's shapes — row tiles of a larger query block at a nonzero
    /// offset, a transposed key block with an odd column count, one buffer for
    /// every tile — each tile is exactly what `matmul` gives.
    #[test]
    fn tiles_of_a_query_block_match_matmul() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let (rows, h, d, cols) = (37usize, 4usize, 128usize, 13_513usize);
        let q = Tensor::randn(0f32, 1., (rows * h, d), &dev)?;
        let keys = Tensor::randn(0f32, 1., (cols, d), &dev)?;
        let keys_t = keys.t()?;
        let per_tile = 10usize;
        let tile = Tensor::zeros((per_tile * h, cols), DType::F32, &dev)?;
        let mut row = 0;
        while row < rows {
            let n = per_tile.min(rows - row);
            let lhs = q.narrow(0, row * h, n * h)?;
            let out = tile.narrow(0, 0, n * h)?;
            matmul_f32_into(&lhs, &keys_t, &out)?;
            let want = lhs.matmul(&keys_t)?;
            let diff = (out - want)?.abs()?.max_all()?.to_vec0::<f32>()?;
            assert_eq!(diff, 0.0, "tile at row {row} differs by {diff}");
            row += n;
        }
        Ok(())
    }

    #[test]
    fn refuses_a_buffer_of_the_wrong_size() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let a = Tensor::ones((2, 3), DType::F32, &dev)?;
        let b = Tensor::ones((3, 4), DType::F32, &dev)?;
        let out = Tensor::zeros((2, 3), DType::F32, &dev)?;
        assert!(matmul_f32_into(&a, &b, &out).is_err());
        Ok(())
    }
}
