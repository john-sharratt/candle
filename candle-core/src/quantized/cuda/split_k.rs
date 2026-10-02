//! The split-K int8 dense matmul: its launch, and the scratch it needs.
//!
//! When the unsplit grid cannot fill the card (see [`super::super::int8_split_k`]), each K slice
//! runs as its own block and stores one F32 partial per K tile; the block that finishes an output
//! tile last sums them in tile order — the chain every unsplit launch folds in, so the bits are
//! the unsplit kernel's. That needs two device buffers per stream:
//!
//! - **partials** — `K_tiles × M × N` F32, every element written before it is read;
//! - **counters** — one u32 per unsplit tile, zero on entry and returned to zero by the kernel.
//!
//! Both are allocated **once per stream, at the fixed caps** ([`SPLIT_SCRATCH_PARTIALS`],
//! [`SPLIT_SCRATCH_COUNTERS`]) the split rule never exceeds, and reused by every split launch on
//! it. Stream order is what makes the reuse sound: a
//! launch's last block has read every partial and reset every counter before the next launch on
//! the same stream starts. A second stream gets its own pair.
//!
//! [`ensure_split_k_scratch`] is called when a KO matmul is built (`QMatMul::from_arc`), so the
//! buffers exist before the first forward: allocating inside one is what the forbidden-allocation
//! detector and the wave arena exist to refuse, and a split launch on a stream without its
//! scratch is an error rather than a lazy allocation.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use candle_kernels::quantized::run_dense_int8_splitk;
use cudarc::driver::{CudaSlice, DevicePtr};

use super::super::int8_split_k::{
    dense_k_split_depth, dense_k_tiles, SPLIT_SCRATCH_COUNTERS, SPLIT_SCRATCH_PARTIALS,
};
use super::super::GgmlDType;
use super::{
    check_matmul_status, dtype_to_qtype, out_dtype_code, resolve_out, tensor_from_owned_out,
    CudaDevice, Q8a128Operand, Result,
};
use crate::{DType, Error, LiveTensor, Shape};

/// One stream's partials and counters.
struct SplitKScratch {
    partials: CudaSlice<f32>,
    counters: CudaSlice<u32>,
}

/// Every stream's scratch, keyed by the stream's address.
fn registry() -> &'static Mutex<HashMap<usize, Arc<SplitKScratch>>> {
    static REG: OnceLock<Mutex<HashMap<usize, Arc<SplitKScratch>>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The registry key of a device handle's stream.
fn stream_key(device: &CudaDevice) -> usize {
    Arc::as_ptr(&device.cuda_stream()) as usize
}

/// This device stream's scratch, which [`ensure_split_k_scratch`] created when a KO matmul was
/// built on it. Absent means a split launch reached a stream no KO weight was built for — an
/// error, never an allocation inside the forward.
fn scratch(device: &CudaDevice) -> Result<Arc<SplitKScratch>> {
    let reg = registry()
        .lock()
        .map_err(|_| Error::Msg("split-K scratch registry poisoned".into()))?;
    reg.get(&stream_key(device)).cloned().ok_or_else(|| {
        Error::Msg(
            "q8a128 split-K matmul: no scratch on this stream — KO matmuls create it when they \
             are built (`QMatMul::from_arc`), so this launch's weight was not built on the \
             stream it runs on"
                .into(),
        )
    })
}

/// Allocate this device stream's split-K scratch now, if it does not exist yet — when a KO
/// matmul is built, so no forward ever allocates it.
pub fn ensure_split_k_scratch(device: &CudaDevice) -> Result<()> {
    let mut reg = registry()
        .lock()
        .map_err(|_| Error::Msg("split-K scratch registry poisoned".into()))?;
    let key = stream_key(device);
    if reg.contains_key(&key) {
        return Ok(());
    }
    let s = Arc::new(SplitKScratch {
        // Fully written by the slices before the last block reads it.
        partials: unsafe { device.alloc::<f32>(SPLIT_SCRATCH_PARTIALS)? },
        // Read before written: the kernel's count-in relies on zero.
        counters: device.alloc_zeros::<u32>(SPLIT_SCRATCH_COUNTERS)?,
    });
    reg.insert(key, s);
    Ok(())
}

/// `op [M, K] × weight [N, K]ᵀ → [M, N]` at `out_dtype`, K cut into `splits` slices of
/// `ceil(K_tiles / splits)` tiles, one block each. `splits` must be a depth with no empty slice
/// ([`dense_k_split_depth`]); every such depth produces the unsplit kernel's bits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn q8a128_dense_matmul_split_k<'w>(
    op: &Q8a128Operand<'w>,
    weight_ptr: u64,
    weight_dtype: GgmlDType,
    nrows: usize,
    splits: usize,
    out_dtype: DType,
    device: &CudaDevice,
) -> Result<LiveTensor<'w>> {
    if !nrows.is_multiple_of(32) {
        crate::bail!("q8a128 split-K matmul: N={nrows} must be a multiple of 32");
    }
    if !op.cols.is_multiple_of(128) {
        crate::bail!(
            "q8a128 split-K matmul: K={} must be a multiple of 128",
            op.cols
        );
    }
    if weight_dtype == GgmlDType::MXFP4_KO {
        crate::bail!(
            "q8a128 split-K matmul: MXFP4 folds each sub straight into the running sum, which \
             per-tile partials cannot reproduce — it runs unsplit"
        );
    }
    if splits < 2 || splits != dense_k_split_depth(op.cols, splits) {
        crate::bail!(
            "q8a128 split-K matmul: {splits} slices of K={} would launch an empty slice; \
             {} covers the same cut",
            op.cols,
            dense_k_split_depth(op.cols, splits)
        );
    }
    let (m, n) = (op.rows, nrows);
    let tiles = m.div_ceil(16) * n.div_ceil(32);
    let s = scratch(device)?;
    let k_tiles = dense_k_tiles(op.cols);
    if k_tiles * m * n > s.partials.len() || tiles > s.counters.len() {
        crate::bail!(
            "q8a128 split-K matmul: {k_tiles} K-tile partials of [{m}, {n}] ({tiles} tiles) exceed the \
             scratch ({} partials, {} counters) — the split rule and its bound disagree",
            s.partials.len(),
            s.counters.len()
        );
    }
    let qtype = dtype_to_qtype(weight_dtype)? as i32;
    let out_code = out_dtype_code(out_dtype)?;
    let (dst_ptr, owned_dst, out_backing) = resolve_out(out_dtype, op.backing(), device, n * m)?;
    let stream = device.cuda_stream();
    let (ws_ptr, _ws_guard) = s.partials.device_ptr(&stream);
    let (ctr_ptr, _ctr_guard) = s.counters.device_ptr(&stream);
    op.with_device_ptr(device, |act_ptr| {
        let status = unsafe {
            run_dense_int8_splitk(
                weight_ptr as *const c_void,
                act_ptr as *const c_void,
                dst_ptr as *mut c_void,
                op.cols as i32,
                n as i32,
                m as i32,
                qtype,
                out_code,
                op.sum_scale.as_code(),
                splits as i32,
                ws_ptr as *mut f32,
                ctr_ptr as *mut u32,
            )
        };
        check_matmul_status(status, "q8a128 split-K matmul")
    })?;
    let mut out_dims = op.lead.clone();
    out_dims.push(n);
    let out_shape: Shape = out_dims.into();
    tensor_from_owned_out(owned_dst, dst_ptr, out_backing, out_shape, device)
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    use super::super::super::int8_split_k::{dense_k_split_depth, q8a128_dense_k_splits};
    use super::super::super::{GgmlDType, Int8Mode, QMatMul, QStorage, QTensor, SumScale};
    use super::super::{dense_qmatmul_with_splits, to_dynamic, CudaDevice};
    use crate::backend::BackendDevice;
    use crate::{DType, Device, Result, Tensor};

    /// A KO twin of a random `[n, k]` weight, from `src`, and an `[m, k]` activation.
    fn operands(
        dev: &CudaDevice,
        src: GgmlDType,
        m: usize,
        n: usize,
        k: usize,
        seed: u64,
    ) -> Result<(QMatMul, Tensor)> {
        let device = Device::Cuda(dev.clone());
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let w: Vec<f32> = (0..n * k).map(|_| rng.random_range(-0.1f32..0.1)).collect();
        let x: Vec<f32> = (0..m * k).map(|_| rng.random_range(-1.0f32..1.0)).collect();
        let q = QMatMul::from_qtensor(QTensor::quantize(
            &Tensor::from_vec(w, (n, k), &device)?,
            src,
        )?)?;
        Ok((
            q.repack_for_optimization(Int8Mode::Performance)?,
            Tensor::from_vec(x, (m, k), &device)?,
        ))
    }

    /// One matmul at a forced slice count, as F32 values.
    fn run(dev: &CudaDevice, w: &QMatMul, x: &Tensor, splits: usize) -> Result<Vec<f32>> {
        run_with(dev, w, x, Some(splits))
    }

    /// One matmul at a forced slice count, or the rule's when `None`, as F32 values.
    fn run_with(
        dev: &CudaDevice,
        w: &QMatMul,
        x: &Tensor,
        splits: Option<usize>,
    ) -> Result<Vec<f32>> {
        let q = w.qtensor().expect("a KO weight");
        let (ptr, len) = match &q.storage {
            QStorage::Cuda(cs) => (cs.data_ptr(), cs.storage_size_in_bytes()),
            _ => unreachable!("a CUDA weight"),
        };
        let acts = to_dynamic(x, Int8Mode::Performance, dev, SumScale::Raw)?;
        let out = dense_qmatmul_with_splits(
            acts.as_dynamic(),
            ptr,
            q.dtype(),
            q.shape().dims()[0],
            len,
            DType::F32,
            splits,
            dev,
        )?;
        out.flatten_all()?.to_vec1::<f32>()
    }

    /// A split launch is the unsplit kernel bit for bit, at every split depth: both fold K one
    /// tile at a time, in tile order — at decode widths, across one and two full token tiles
    /// (32 rows: every reducing thread owns its full share of outputs), a partial tile, a
    /// short last slice (K = 384 is three tiles), one tile per slice and many, and two KO
    /// formats.
    #[test]
    fn a_split_is_the_unsplit_kernel_bit_for_bit() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        for (src, m, n, k) in [
            (GgmlDType::Q8_0, 1, 416, 10_240),
            (GgmlDType::Q8_0, 5, 416, 10_240),
            (GgmlDType::Q8_0, 16, 416, 10_240),
            (GgmlDType::Q8_0, 17, 416, 10_240),
            (GgmlDType::Q8_0, 32, 416, 10_240),
            (GgmlDType::Q8_0, 2, 416, 384),
            (GgmlDType::Q4_K, 3, 512, 2560),
        ] {
            let (w, x) = operands(&dev, src, m, n, k, (m * n + k) as u64)?;
            let unsplit = run(&dev, &w, &x, 1)?;
            let mut depths: Vec<usize> = [2usize, 3, 7, 27, 1_000]
                .iter()
                .map(|&s| dense_k_split_depth(k, s))
                .filter(|&d| d > 1)
                .collect();
            depths.dedup();
            for d in depths {
                assert_eq!(
                    run(&dev, &w, &x, d)?,
                    unsplit,
                    "{src:?} [{m}x{k}]·[{n}x{k}]ᵀ, {d} slices"
                );
            }
        }
        Ok(())
    }

    /// A depth that would launch an empty slice is refused, not run.
    #[test]
    fn a_depth_with_an_empty_slice_is_refused() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (w, x) = operands(&dev, GgmlDType::Q8_0, 1, 416, 10_240, 3)?;
        // 80 tiles in 34 slices of 3: the last 7 slices would be empty.
        assert_eq!(dense_k_split_depth(10_240, 34), 27);
        assert!(run(&dev, &w, &x, 34).is_err());
        Ok(())
    }

    /// A split launch on a stream no KO matmul was built on finds no scratch and is refused —
    /// the forward never allocates it.
    #[test]
    fn a_split_on_a_stream_without_scratch_is_refused() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (w, x) = operands(&dev, GgmlDType::Q8_0, 1, 416, 10_240, 9)?;
        let other = CudaDevice::new_with_stream(0)?;
        let x_other = x.to_device(&Device::Cuda(other.clone()))?;
        let err = run(&other, &w, &x_other, dense_k_split_depth(10_240, 27))
            .expect_err("no scratch on the second stream");
        assert!(
            err.to_string().contains("no scratch on this stream"),
            "{err}"
        );
        Ok(())
    }

    /// MXFP4 folds each sub straight into the running sum, which per-tile partials cannot
    /// reproduce: the rule runs it unsplit at decode width, and a forced split is refused.
    #[test]
    fn mxfp4_never_splits() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (n, k) = (416usize, 2048usize);
        assert!(q8a128_dense_k_splits(1, n, k, dev.multiprocessor_count()?) > 1);
        let (w, x) = operands(&dev, GgmlDType::MXFP4, 1, n, k, 5)?;
        assert_eq!(run_with(&dev, &w, &x, None)?, run(&dev, &w, &x, 1)?);
        assert!(run(&dev, &w, &x, dense_k_split_depth(k, 4)).is_err());
        Ok(())
    }

    /// A row's output does not depend on how many rows share its launch: the same row alone,
    /// in a split decode-width launch and in an unsplit wide one gives the same bits.
    #[test]
    fn a_row_is_the_same_whatever_the_wave_width() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let sm = dev.multiprocessor_count()?;
        let (n, k) = (416usize, 10_240usize);
        let (w, wide) = operands(&dev, GgmlDType::Q8_0, 40, n, k, 11)?;
        assert_eq!(
            q8a128_dense_k_splits(40, n, k, sm),
            1,
            "40 rows run unsplit"
        );
        let wide_out = run(&dev, &w, &wide, 1)?;
        for m in [1usize, 5] {
            let splits = q8a128_dense_k_splits(m, n, k, sm);
            assert!(splits > 1, "{m} rows split");
            let few = wide.narrow(0, 0, m)?.contiguous()?;
            let out = run(&dev, &w, &few, splits)?;
            assert_eq!(out[..], wide_out[..m * n], "the first {m} rows");
        }
        Ok(())
    }

    /// A split launch is bit-repeatable — the last block to finish sums in tile order,
    /// whichever block it is — and back-to-back launches see the counters the previous launch
    /// returned to zero.
    #[test]
    fn a_split_is_bit_repeatable_launch_after_launch() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (w, x) = operands(&dev, GgmlDType::Q8_0, 4, 416, 10_240, 7)?;
        let slices = dense_k_split_depth(10_240, 27);
        let first = run(&dev, &w, &x, slices)?;
        for i in 0..20 {
            assert_eq!(run(&dev, &w, &x, slices)?, first, "launch {i} differs");
        }
        Ok(())
    }
}
