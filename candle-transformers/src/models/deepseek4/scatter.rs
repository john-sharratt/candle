//! Batched row scatter — many `(source run → destination offset)` copies in ONE
//! launch.
//!
//! The corpus-gallery append is a scatter: every session of a wave takes its own
//! slice of one fleet-wide pooled block and writes it at its own gallery's
//! current length, across six arrays (Indexer keys, positions, packed sign bits,
//! and the three regions of the two-region latent cache). Spelled with
//! `slice_set` that is one launch per array per session per compression layer,
//! each moving a few kilobytes — launch overhead almost end to end, and a direct
//! violation of hot-path invariant 2.
//!
//! [`rows_scatter`] takes the whole set as a descriptor table instead, which is
//! what invariant 2b asks for: the destination is named by address, so the copy
//! never has to be split per destination.

use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::quantized::pinned_staging::Generation;
use candle::{DType, Device, Result, Storage, Tensor};
use candle_kernels::simple::rows_scatter::{run_rows_scatter, ROWS_SCATTER_WORDS};

/// One `(rows of `src` → `dst` starting at row `dst_row`)` copy.
///
/// `src` is a `[rows, w]` view (any element type whose row width is a multiple
/// of 4 bytes) and `dst` a `[rows_total, w]` destination with the same element
/// type and row width. Both must have unit stride along the channel axis; the
/// row strides may differ, so a source that is a slice of a wider block is read
/// where it lies.
pub struct RowRun {
    pub src: Tensor,
    pub dst: Tensor,
    pub dst_row: usize,
}

impl RowRun {
    /// `src`'s rows into `dst` starting at `dst_row`. Both tensors are `Arc`
    /// handles, so holding them costs a pointer bump and keeps the storage the
    /// descriptor addresses alive until the launch is enqueued.
    pub fn new(src: Tensor, dst: &Tensor, dst_row: usize) -> Self {
        Self {
            src,
            dst: dst.clone(),
            dst_row,
        }
    }
}

/// Bytes per element, or an error for a type the word-typed kernel cannot carry.
fn elem_size(dt: DType) -> Result<usize> {
    Ok(match dt {
        DType::U8 | DType::F8E4M3 => 1,
        DType::U32 | DType::F32 => 4,
        DType::I64 | DType::F64 => 8,
        DType::BF16 | DType::F16 => 2,
    })
}

/// Base address of a tensor's logical element 0, in bytes.
fn base_ptr(t: &Tensor, stream: &candle::cuda_backend::cudarc::driver::CudaStream) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let off = layout.start_offset();
    let p = match &*storage {
        Storage::Cuda(c) => match t.dtype() {
            // Byte-addressed types share the `u8` slice view — the kernel reads
            // 32-bit words either way, so only the base address matters.
            DType::U8 | DType::F8E4M3 => {
                c.as_cuda_slice::<u8>()?.slice(off..).device_ptr(stream).0
            }
            DType::U32 => c.as_cuda_slice::<u32>()?.slice(off..).device_ptr(stream).0,
            DType::F32 => c.as_cuda_slice::<f32>()?.slice(off..).device_ptr(stream).0,
            DType::BF16 => c
                .as_cuda_slice::<half::bf16>()?
                .slice(off..)
                .device_ptr(stream)
                .0,
            DType::F16 => c
                .as_cuda_slice::<half::f16>()?
                .slice(off..)
                .device_ptr(stream)
                .0,
            DType::I64 => c.as_cuda_slice::<i64>()?.slice(off..).device_ptr(stream).0,
            DType::F64 => c.as_cuda_slice::<f64>()?.slice(off..).device_ptr(stream).0,
        },
        _ => candle::bail!("rows_scatter: expected CUDA storage"),
    };
    Ok(p)
}

/// A tensor's `(rows, row width in bytes, row stride in bytes)`, accepting both
/// `[rows]` (an implicit width of one element) and `[rows, w]`.
fn row_geometry(t: &Tensor) -> Result<(usize, usize, usize)> {
    let es = elem_size(t.dtype())?;
    match t.dims() {
        [n] => Ok((*n, es, t.stride()[0] * es)),
        [n, w] => {
            if t.stride()[1] != 1 {
                candle::bail!("rows_scatter: channel stride {} != 1", t.stride()[1]);
            }
            Ok((*n, w * es, t.stride()[0] * es))
        }
        d => candle::bail!("rows_scatter: expected a 1-D or 2-D operand, got {d:?}"),
    }
}

/// Run every copy in `runs` in ONE launch. A run with no rows is skipped; an
/// empty (or all-empty) `runs` is a no-op.
/// `generation` is the pinned-arena guard the descriptor table is bump-allocated
/// from and read in place — the wave's, or one the caller opened with
/// `desc::scope`. See `desc.rs`.
pub fn rows_scatter(runs: &[RowRun], generation: &Generation) -> Result<()> {
    let live: Vec<&RowRun> = runs
        .iter()
        .filter(|r| r.src.dims().first().copied().unwrap_or(0) > 0)
        .collect();
    if live.is_empty() {
        return Ok(());
    }
    let dev = match live[0].dst.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("rows_scatter requires CUDA"),
    };
    let stream = dev.cuda_stream();

    let mut desc = vec![0i64; ROWS_SCATTER_WORDS * live.len()];
    let mut max_elems = 0usize;
    for (i, run) in live.iter().enumerate() {
        // Element types need NOT match: the kernel copies 32-bit words, so a
        // run is valid whenever the two rows are the same width in BYTES. That
        // is what lets a warm-tier re-heat stage all of its regions in one
        // untyped byte buffer and unpack them into their typed destinations.
        let (rows, w_bytes, src_stride_b) = row_geometry(&run.src)?;
        let (dst_rows, dst_w_bytes, dst_stride_b) = row_geometry(&run.dst)?;
        if w_bytes != dst_w_bytes {
            candle::bail!(
                "rows_scatter: source row {w_bytes} B, destination row {dst_w_bytes} B"
            );
        }
        if w_bytes % 4 != 0 || src_stride_b % 4 != 0 || dst_stride_b % 4 != 0 {
            candle::bail!(
                "rows_scatter: row width {w_bytes} B / strides {src_stride_b},{dst_stride_b} B \
                 are not 32-bit aligned"
            );
        }
        if run.dst_row + rows > dst_rows {
            candle::bail!(
                "rows_scatter: writing rows {}..{} of a {dst_rows}-row destination",
                run.dst_row,
                run.dst_row + rows
            );
        }
        let words = w_bytes / 4;
        let dst_stride_w = dst_stride_b / 4;
        let w = &mut desc[i * ROWS_SCATTER_WORDS..][..ROWS_SCATTER_WORDS];
        w[0] = base_ptr(&run.src, &stream)? as i64;
        w[1] = (src_stride_b / 4) as i64;
        // Bake the destination offset into the address so the kernel only walks
        // rows — the whole point of the descriptor.
        w[2] = (base_ptr(&run.dst, &stream)? + (run.dst_row * dst_stride_b) as u64) as i64;
        w[3] = dst_stride_w as i64;
        w[4] = rows as i64;
        w[5] = words as i64;
        max_elems = max_elems.max(rows * words);
    }

    // Staged into the wave's pinned arena and read in place — for the
    // 64-session shape the upload was 24 us against a 2.9 us kernel (`desc.rs`).
    // Held until the launch below is enqueued.
    let staged = crate::models::deepseek4::desc::stage(&desc, generation)?;
    let p_desc = staged.ptr();
    unsafe {
        run_rows_scatter(
            p_desc as *const i64,
            live.len() as i32,
            max_elems as i32,
            stream.cu_stream() as *mut core::ffi::c_void,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every run lands exactly where its descriptor says, for mixed element
    /// types, mixed row widths, strided sources, and destinations that must be
    /// left untouched outside the written span.
    ///
    /// The assertion is on RAW BYTES against a host-computed expectation, not a
    /// tolerance: this is a copy, so anything other than exact equality is a
    /// bug in the addressing.
    #[test]
    fn scatter_places_every_run_exactly() -> Result<()> {
        let dev = Device::new_cuda(0)?;

        // A wide f32 source the runs slice rows out of — the shape a wave's
        // pooled block has, so the runs read a strided window, not row 0.
        let src_f32: Vec<f32> = (0..12 * 8).map(|i| i as f32 * 0.5).collect();
        let src_f32_t = Tensor::from_vec(src_f32.clone(), (12, 8), &dev)?;
        let src_u32: Vec<u32> = (0..12u32).map(|i| i * 7 + 1).collect();
        let src_u32_t = Tensor::from_vec(src_u32.clone(), 12, &dev)?;
        // 4 bytes per row: the narrowest row the word-typed kernel can carry.
        let src_u8: Vec<u8> = (0..12 * 4).map(|i| (i * 3 % 251) as u8).collect();
        let src_u8_t = Tensor::from_vec(src_u8.clone(), (12, 4), &dev)?;

        let dst_f32 = Tensor::zeros((20, 8), DType::F32, &dev)?;
        let dst_u32 = Tensor::zeros(20, DType::U32, &dev)?;
        let dst_u8 = Tensor::zeros((20, 4), DType::U8, &dev)?;

        // Three galleries' worth: source rows 0..3, 3..7, 7..12 land at three
        // unrelated destination offsets, in all three arrays at once.
        let plan = [(0usize, 3usize, 5usize), (3, 4, 0), (7, 5, 11)];
        let mut runs = Vec::new();
        for &(src_row, n, dst_row) in &plan {
            runs.push(RowRun::new(src_f32_t.narrow(0, src_row, n)?, &dst_f32, dst_row));
            runs.push(RowRun::new(src_u32_t.narrow(0, src_row, n)?, &dst_u32, dst_row));
            runs.push(RowRun::new(src_u8_t.narrow(0, src_row, n)?, &dst_u8, dst_row));
        }
        rows_scatter(&runs, &crate::models::deepseek4::desc::scope(&dev)?)?;

        let mut want_f32 = vec![0f32; 20 * 8];
        let mut want_u32 = vec![0u32; 20];
        let mut want_u8 = vec![0u8; 20 * 4];
        for &(src_row, n, dst_row) in &plan {
            for r in 0..n {
                want_u32[dst_row + r] = src_u32[src_row + r];
                for c in 0..8 {
                    want_f32[(dst_row + r) * 8 + c] = src_f32[(src_row + r) * 8 + c];
                }
                for c in 0..4 {
                    want_u8[(dst_row + r) * 4 + c] = src_u8[(src_row + r) * 4 + c];
                }
            }
        }
        assert_eq!(dst_f32.flatten_all()?.to_vec1::<f32>()?, want_f32);
        assert_eq!(dst_u32.to_vec1::<u32>()?, want_u32);
        assert_eq!(dst_u8.flatten_all()?.to_vec1::<u8>()?, want_u8);
        Ok(())
    }

    /// A run whose source has no rows contributes nothing, and an all-empty
    /// batch is a no-op rather than a launch with a zero-entry table.
    #[test]
    fn empty_runs_are_skipped() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let dst = Tensor::zeros((4, 4), DType::F32, &dev)?;
        let empty = Tensor::zeros((0, 4), DType::F32, &dev)?;
        rows_scatter(&[RowRun::new(empty.clone(), &dst, 0)], &crate::models::deepseek4::desc::scope(&dev)?)?;
        assert_eq!(dst.flatten_all()?.to_vec1::<f32>()?, vec![0f32; 16]);

        let src = Tensor::from_vec((0..8u32).map(|i| i as f32).collect::<Vec<_>>(), (2, 4), &dev)?;
        rows_scatter(&[RowRun::new(empty, &dst, 0), RowRun::new(src, &dst, 2)], &crate::models::deepseek4::desc::scope(&dev)?)?;
        let got = dst.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(&got[..8], &[0f32; 8]);
        assert_eq!(&got[8..], &[0f32, 1., 2., 3., 4., 5., 6., 7.]);
        Ok(())
    }

    /// Isolation bench for the batched scatter at the shapes the gallery append
    /// actually produces: `n_sess` sessions × six arrays, each moving a group's
    /// worth of rows. The interesting axis is SESSION COUNT — the whole point is
    /// that the launch count stops scaling with it — and the interesting
    /// comparison is against the `slice_set` storm it replaced.
    ///
    ///   cargo test -p candle-transformers --features cuda --release --lib \
    ///     bench_rows_scatter -- --ignored --nocapture
    ///   ncu --set full --kernel-name rows_scatter_kernel --launch-count 4 \
    ///     -o scatter_ncu target/release/deps/candle_transformers-<hash>.exe \
    ///     bench_rows_scatter --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_rows_scatter() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        // The six arrays a gallery append writes, by row width in elements.
        let arrays: [(usize, DType); 6] = [
            (1, DType::U32),      // pos
            (4, DType::U32),      // packed signs (index_head_dim 128 → 4 words)
            (128, DType::F32),    // indexer keys
            (448, DType::U8),     // nope int8 band
            (4, DType::F32),      // per-band scale
            (64, DType::BF16),    // rope tail
        ];
        for &(n_sess, rows) in &[(1usize, 1usize), (8, 1), (16, 1), (16, 4), (64, 1)] {
            let total = n_sess * rows;
            let mut srcs = Vec::new();
            let mut dsts = Vec::new();
            for &(w, dt) in &arrays {
                srcs.push(Tensor::zeros((total, w), dt, &dev)?);
                // Destinations are the galleries' capacity buffers, one each.
                dsts.push(
                    (0..n_sess)
                        .map(|_| Tensor::zeros((4096, w), dt, &dev))
                        .collect::<Result<Vec<_>>>()?,
                );
            }
            let mut runs = Vec::with_capacity(6 * n_sess);
            for (src, dst) in srcs.iter().zip(dsts.iter()) {
                for (s, d) in dst.iter().enumerate() {
                    runs.push(RowRun::new(src.narrow(0, s * rows, rows)?, d, 17));
                }
            }

            // One scope for the whole timed loop — the table is bump-allocated
            // from it and read in place over PCIe.
            let gen = crate::models::deepseek4::desc::scope(&dev)?;
            let timed = |g: &Generation| -> Result<f64> {
                for _ in 0..8 {
                    rows_scatter(&runs, g)?;
                }
                let mut best = f64::INFINITY;
                for _ in 0..5 {
                    dev.synchronize()?;
                    let t = std::time::Instant::now();
                    for _ in 0..200 {
                        rows_scatter(&runs, g)?;
                    }
                    dev.synchronize()?;
                    best = best.min(t.elapsed().as_secs_f64() * 1e6 / 200.0);
                }
                Ok(best)
            };
            let best = timed(&gen)?;

            // What it replaced: one `slice_set` per array per session.
            let mut eager = f64::INFINITY;
            let slice_set_all = || -> Result<()> {
                for (src, dst) in srcs.iter().zip(dsts.iter()) {
                    for (s, d) in dst.iter().enumerate() {
                        d.slice_set(&src.narrow(0, s * rows, rows)?, 0, 17)?;
                    }
                }
                Ok(())
            };
            for _ in 0..8 {
                slice_set_all()?;
            }
            for _ in 0..5 {
                dev.synchronize()?;
                let t = std::time::Instant::now();
                for _ in 0..200 {
                    slice_set_all()?;
                }
                dev.synchronize()?;
                eager = eager.min(t.elapsed().as_secs_f64() * 1e6 / 200.0);
            }
            println!(
                "[scatter sess={n_sess:<3} rows={rows}] runs={:<4} arena {best:7.2} us  \
                 slice_set {eager:8.2} us  {:.1}x",
                runs.len(),
                eager / best
            );
        }
        Ok(())
    }

    /// Writing past the destination is caught, not silently corrupting memory
    /// beyond it.
    #[test]
    fn overrun_is_rejected() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let dst = Tensor::zeros((4, 4), DType::F32, &dev)?;
        let src = Tensor::zeros((3, 4), DType::F32, &dev)?;
        let scope = crate::models::deepseek4::desc::scope(&dev)?;
        assert!(rows_scatter(&[RowRun::new(src, &dst, 2)], &scope).is_err());
        Ok(())
    }
}
