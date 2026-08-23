//! Compressed-index expansion — the decode path's `comp_idx` in ONE launch.
//!
//! Every decode slot's corpus selection is the dense range `[offset, offset+k)`
//! of the fleet-wide gathered block, so the index matrix the paged decode kernel
//! reads is fully determined by two small numbers per slot. Building it with
//! tensor ops cost two pageable uploads (the offsets, the counts) plus five
//! launches (`arange`, `broadcast_add`, `broadcast_lt`, `full`, `where_cond`) —
//! every layer, every step, around arithmetic of about thirty input words.
//!
//! [`build`] stages the `{offset, count}` table in the wave's pinned arena and
//! expands it with a single kernel, which also republishes the counts as the
//! device array the decode kernel wants. Seven launches and two uploads become
//! one launch and no upload.

use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::quantized::pinned_staging::Generation;
use candle::{DType, Device, Result, Storage, Tensor};
use candle_kernels::simple::comp_idx::run_comp_idx_build;

use super::desc;

/// Device address of a `u32` tensor's element 0.
fn u32_ptr(t: &Tensor, stream: &candle::cuda_backend::cudarc::driver::CudaStream) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        Storage::Cuda(c) => Ok(c
            .as_cuda_slice::<u32>()?
            .slice(layout.start_offset()..)
            .device_ptr(stream)
            .0),
        _ => candle::bail!("comp_idx: expected CUDA storage"),
    }
}

/// Expand per-slot `{offset, count}` into `(comp_idx [n, max_sel], comp_cnt [n])`.
///
/// `comp_idx[i][k]` is `offsets[i] + k` while `k < cnts[i]` and `u32::MAX`
/// after — the ascending-with-sentinel-padding contract the decode kernel
/// documents. `max_sel` is the widest row; the caller already knows it from the
/// counts, and passing it keeps the allocation the caller's decision.
pub fn build(
    offsets: &[u32],
    cnts: &[u32],
    max_sel: usize,
    device: &Device,
    generation: &Generation,
) -> Result<(Tensor, Tensor)> {
    if offsets.len() != cnts.len() {
        candle::bail!(
            "comp_idx: {} offsets against {} counts",
            offsets.len(),
            cnts.len()
        );
    }
    let n = offsets.len();
    if n == 0 || max_sel == 0 {
        candle::bail!("comp_idx: empty expansion ({n} slots × {max_sel})");
    }
    if let Some(bad) = cnts.iter().position(|&c| c as usize > max_sel) {
        candle::bail!(
            "comp_idx: slot {bad} selects {} entries, past the {max_sel}-wide row",
            cnts[bad]
        );
    }
    let dev = match device {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("comp_idx requires CUDA"),
    };
    let stream = dev.cuda_stream();

    // Struct-of-arrays, matching the kernel: offsets then counts.
    let mut table = Vec::with_capacity(2 * n);
    table.extend_from_slice(offsets);
    table.extend_from_slice(cnts);
    let staged = desc::stage_slice(&table, generation)?;

    // Both outputs are written in full by the kernel — allocate uninitialised
    // rather than paying a memset on the exact bytes it is about to stamp
    // (invariant 6).
    let idx = Tensor::empty((n, max_sel), DType::U32, device)?;
    let cnt = Tensor::empty(n, DType::U32, device)?;
    let idx_p = u32_ptr(&idx, &stream)?;
    let cnt_p = u32_ptr(&cnt, &stream)?;
    unsafe {
        run_comp_idx_build(
            staged.ptr() as *const u32,
            idx_p as *mut u32,
            cnt_p as *mut u32,
            n as i32,
            max_sel as i32,
            stream.cu_stream() as *mut core::ffi::c_void,
        );
    }
    Ok((idx, cnt))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The expansion matches the broadcast chain it replaced, exactly — this is
    /// index arithmetic, so the assertion is on raw values, not a tolerance.
    /// Ragged counts, a zero-count slot, and a full-width slot all in one batch.
    #[test]
    fn expands_every_slot_exactly() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let offsets = [0u32, 3, 3, 11, 12];
        let cnts = [3u32, 0, 8, 1, 5];
        let max_sel = 8usize;
        let (idx, cnt) = build(&offsets, &cnts, max_sel, &dev, &desc::scope(&dev)?)?;

        let mut want = Vec::with_capacity(offsets.len() * max_sel);
        for (i, &c) in cnts.iter().enumerate() {
            for k in 0..max_sel {
                want.push(if k < c as usize {
                    offsets[i] + k as u32
                } else {
                    u32::MAX
                });
            }
        }
        assert_eq!(idx.flatten_all()?.to_vec1::<u32>()?, want);
        assert_eq!(cnt.to_vec1::<u32>()?, cnts.to_vec());
        Ok(())
    }

    /// A single slot at width 1 — the shape a one-session decode step produces,
    /// and the narrowest grid the launcher can be handed.
    #[test]
    fn single_slot_single_entry() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let (idx, cnt) = build(&[7], &[1], 1, &dev, &desc::scope(&dev)?)?;
        assert_eq!(idx.flatten_all()?.to_vec1::<u32>()?, vec![7u32]);
        assert_eq!(cnt.to_vec1::<u32>()?, vec![1u32]);
        Ok(())
    }

    /// A count wider than the row would have the kernel write past the
    /// allocation; it is rejected host-side rather than corrupting memory.
    #[test]
    fn overwide_count_is_rejected() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let scope = desc::scope(&dev)?;
        assert!(build(&[0, 0], &[2, 9], 8, &dev, &scope).is_err());
        assert!(build(&[0, 0], &[2], 8, &dev, &scope).is_err());
        Ok(())
    }
}
