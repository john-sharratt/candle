//! The narrow int8 dense matmul: its launch.
//!
//! At decode width (at most [`NARROW_MAX_ROWS`] rows) a shape whose unsplit grid cannot fill the
//! card runs one block per 8-row output tile, K walked by the block's warps and summed in shared
//! memory in tile order (see [`super::super::int8_split_k`]) — the chain every unsplit launch
//! folds in, so the bits are the unsplit kernel's, with no scratch on any stream.

use std::ffi::c_void;

use candle_kernels::quantized::{run_dense_int8_narrow, SPLITK_MAX_SEGS};

use super::super::int8_split_k::{
    narrow_smem_bound, NARROW_MAX_ROWS, NARROW_MAX_WARPS, NARROW_SMEM_CAP,
};
use super::super::GgmlDType;
use super::{
    check_matmul_status, dtype_to_qtype, out_dtype_code, resolve_out, tensor_from_owned_out,
    CudaDevice, Q8a128Operand, Result,
};
use crate::{DType, LiveTensor, Shape};

/// `op [M, K] × [W₀; W₁; …]ᵀ → [M, ΣNᵢ]` at `out_dtype` in ONE narrow launch at `warps` warps
/// per block: the weights `segments` (`(device pointer, rows)`, all of `weight_dtype`) read the
/// same operand, and each writes its columns after the segments before it. Every output column
/// is the unsplit kernel's on the same weight, bit for bit. At most [`SPLITK_MAX_SEGS`]
/// segments, each a multiple of 32 rows; at most [`NARROW_MAX_ROWS`] rows.
pub(crate) fn q8a128_dense_matmul_narrow_segmented<'w>(
    op: &Q8a128Operand<'w>,
    segments: &[(u64, usize)],
    weight_dtype: GgmlDType,
    warps: usize,
    out_dtype: DType,
    device: &CudaDevice,
) -> Result<LiveTensor<'w>> {
    if segments.is_empty() || segments.len() > SPLITK_MAX_SEGS {
        crate::bail!(
            "q8a128 narrow matmul: {} weight segments, against 1..={SPLITK_MAX_SEGS}",
            segments.len()
        );
    }
    if let Some(&(_, n)) = segments.iter().find(|&&(_, n)| !n.is_multiple_of(32)) {
        crate::bail!("q8a128 narrow matmul: a segment of N={n} must be a multiple of 32");
    }
    if !op.cols.is_multiple_of(128) {
        crate::bail!(
            "q8a128 narrow matmul: K={} must be a multiple of 128",
            op.cols
        );
    }
    if weight_dtype == GgmlDType::MXFP4_KO {
        crate::bail!(
            "q8a128 narrow matmul: MXFP4 folds each sub straight into the running sum, which \
             per-tile folds cannot reproduce — it runs unsplit"
        );
    }
    let m = op.rows;
    if m == 0 || m > NARROW_MAX_ROWS {
        crate::bail!("q8a128 narrow matmul: {m} rows, against 1..={NARROW_MAX_ROWS}");
    }
    if warps == 0 || warps > NARROW_MAX_WARPS {
        crate::bail!("q8a128 narrow matmul: {warps} warps, against 1..={NARROW_MAX_WARPS}");
    }
    if narrow_smem_bound(op.cols, warps) > NARROW_SMEM_CAP {
        crate::bail!(
            "q8a128 narrow matmul: K={} at {warps} warps needs {} bytes of shared memory, past \
             the {NARROW_SMEM_CAP}-byte cap — the plan and its bound disagree",
            op.cols,
            narrow_smem_bound(op.cols, warps)
        );
    }
    let n: usize = segments.iter().map(|&(_, n)| n).sum();
    let qtype = dtype_to_qtype(weight_dtype)? as i32;
    let out_code = out_dtype_code(out_dtype)?;
    let (dst_ptr, owned_dst, out_backing) = resolve_out(out_dtype, op.backing(), device, n * m)?;
    let stream = device.cuda_stream();
    let weights: Vec<*const c_void> = segments.iter().map(|&(p, _)| p as *const c_void).collect();
    let rows: Vec<i32> = segments.iter().map(|&(_, n)| n as i32).collect();
    op.with_device_ptr(device, |act_ptr| {
        // SAFETY: `weights`/`rows` are host arrays of `segments.len()` entries the launcher
        // copies into the kernel parameters; every pointer is a KO weight of its row count.
        let status = unsafe {
            run_dense_int8_narrow(
                weights.as_ptr(),
                rows.as_ptr(),
                segments.len() as i32,
                act_ptr as *const c_void,
                dst_ptr as *mut c_void,
                op.cols as i32,
                m as i32,
                qtype,
                out_code,
                op.sum_scale.as_code(),
                warps as i32,
                stream.cu_stream() as *mut c_void,
            )
        };
        check_matmul_status(status, "q8a128 narrow matmul")
    })?;
    let mut out_dims = op.lead.clone();
    out_dims.push(n);
    let out_shape: Shape = out_dims.into();
    tensor_from_owned_out(owned_dst, dst_ptr, out_backing, out_shape, device)
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use candle_kernels::quantized::{
        run_dense_int8_narrow, run_dense_int8_splitk, run_quantized_matmul, VxSegment, YType,
    };
    use cudarc::driver::DevicePtr;
    use half::bf16;
    use rand::{Rng, SeedableRng};

    use super::super::super::int8_matmul_mode::q8a128_dense_tile;
    use super::super::super::int8_split_k::{
        dense_k_split_depth, dense_k_tiles, q8a128_dense_k_splits, q8a128_dense_plan, DensePlan,
        NARROW_MAX_ROWS,
    };
    use super::super::super::{GgmlDType, Int8Mode, QMatMul, QStorage, QTensor, SumScale};
    use super::super::graph_bench::{time_graph, Rotation};
    use super::super::{
        check_matmul_status, dense_qmatmul_with_plan, dtype_to_qtype, out_dtype_code, to_dynamic,
        CudaDevice, DynamicActs,
    };
    use super::q8a128_dense_matmul_narrow_segmented;
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

    /// The weight's device pointer, byte length, KO dtype and rows.
    fn weight(w: &QMatMul) -> (u64, usize, GgmlDType, usize) {
        let q = w.qtensor().expect("a KO weight");
        match &q.storage {
            QStorage::Cuda(cs) => (
                cs.data_ptr(),
                cs.storage_size_in_bytes(),
                q.dtype(),
                q.shape().dims()[0],
            ),
            _ => unreachable!("a CUDA weight"),
        }
    }

    /// One matmul under `plan`, stored at `out`, as the raw bits of every output element.
    fn bits(
        dev: &CudaDevice,
        w: &QMatMul,
        x: &Tensor,
        sum: SumScale,
        out: DType,
        plan: DensePlan,
    ) -> Result<Vec<u32>> {
        let (ptr, len, dtype, n) = weight(w);
        let acts = to_dynamic(x, Int8Mode::Performance, dev, sum)?;
        let t =
            dense_qmatmul_with_plan(acts.as_dynamic(), ptr, dtype, n, len, out, Some(plan), dev)?
                .flatten_all()?;
        Ok(match out {
            DType::F32 => t.to_vec1::<f32>()?.iter().map(|v| v.to_bits()).collect(),
            DType::BF16 => t
                .to_vec1::<bf16>()?
                .iter()
                .map(|v| v.to_bits() as u32)
                .collect(),
            other => unreachable!("the tests store at F32 or BF16, not {other:?}"),
        })
    }

    /// [`bits`] of a raw-Σx operand stored at F32.
    fn f32_bits(dev: &CudaDevice, w: &QMatMul, x: &Tensor, plan: DensePlan) -> Result<Vec<u32>> {
        bits(dev, w, x, SumScale::Raw, DType::F32, plan)
    }

    /// A narrow launch is the unsplit kernel bit for bit — both fold K one tile at a time, in
    /// tile order — at the Flash-Next decode shapes (the hyper-connection `down`, the DeltaNet
    /// and attention out-projections, the shared expert's `down`) and the edges: one row, a
    /// full eight, K of one, three and five tiles (fewer tiles than warps, empty warp ranges),
    /// a single 64-row output. Two KO formats (8-bit and the 4-bit twin, from Q4_0 so every K
    /// here quantizes), both output
    /// widths, both Σx conventions, and every warp count the plan launches plus two it does
    /// not (one warp: no slots at all; two: half of K slotted).
    #[test]
    fn a_narrow_launch_is_the_unsplit_kernel_bit_for_bit() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        for src in [GgmlDType::Q8_0, GgmlDType::Q4_0] {
            for (m, n, k) in [
                (1usize, 416usize, 10_240usize),
                (5, 416, 10_240),
                (8, 416, 10_240),
                (5, 2560, 6144),
                (5, 2560, 4096),
                (5, 2560, 640),
                (5, 64, 128),
                (2, 416, 384),
            ] {
                let (w, x) = operands(&dev, src, m, n, k, (m * n + k) as u64)?;
                for sum in [SumScale::Raw, SumScale::ByAmax] {
                    for out in [DType::F32, DType::BF16] {
                        let unsplit = bits(&dev, &w, &x, sum, out, DensePlan::Unsplit)?;
                        assert_eq!(unsplit.len(), m * n);
                        for warps in [1usize, 2, 4, 8, 16] {
                            assert_eq!(
                                bits(&dev, &w, &x, sum, out, DensePlan::Narrow { warps })?,
                                unsplit,
                                "{src:?} [{m}x{k}]·[{n}x{k}]ᵀ, {sum:?}, {out:?}, {warps} warps"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The other KO formats the narrow kernel is built for — Q5, Q6, Q2 and Q3 twins — at the
    /// hyper-connection `down` width and five rows.
    #[test]
    fn every_narrow_format_is_the_unsplit_kernel_bit_for_bit() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        for src in [
            GgmlDType::Q5_K,
            GgmlDType::Q6_K,
            GgmlDType::Q2_K,
            GgmlDType::Q3_K,
        ] {
            let (m, n, k) = (5usize, 416usize, 2560usize);
            let (w, x) = operands(&dev, src, m, n, k, 77)?;
            let unsplit = f32_bits(&dev, &w, &x, DensePlan::Unsplit)?;
            for warps in [4usize, 8] {
                assert_eq!(
                    f32_bits(&dev, &w, &x, DensePlan::Narrow { warps })?,
                    unsplit,
                    "{src:?}, {warps} warps"
                );
            }
        }
        Ok(())
    }

    /// Weights that share an operand, launched as segments of one narrow row, are each weight's
    /// own unsplit launch bit for bit, column for column — a MoE layer's router, shared gate_up
    /// and gate at their real widths, at one and five rows.
    #[test]
    fn narrow_segments_are_each_weights_own_unsplit_launch() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let k = 2560usize;
        for m in [1usize, 5] {
            let parts: Vec<(QMatMul, Tensor)> = [512usize, 1280, 32]
                .iter()
                .enumerate()
                .map(|(i, &n)| operands(&dev, GgmlDType::Q8_0, m, n, k, 40 + i as u64))
                .collect::<Result<_>>()?;
            let x = &parts[0].1;
            let acts = to_dynamic(x, Int8Mode::Performance, &dev, SumScale::Raw)?;
            let DynamicActs::Int8(op) = &acts else {
                unreachable!("an int8 mode quantizes")
            };
            let mut segs = Vec::new();
            let mut alone = Vec::new();
            for (w, _) in &parts {
                let (ptr, _, _, n) = weight(w);
                segs.push((ptr, n));
                alone.push(f32_bits(&dev, w, x, DensePlan::Unsplit)?);
            }
            let dtype = weight(&parts[0].0).2;
            let total: usize = segs.iter().map(|&(_, n)| n).sum();
            for warps in [4usize, 8] {
                let out = q8a128_dense_matmul_narrow_segmented(
                    op,
                    &segs,
                    dtype,
                    warps,
                    DType::F32,
                    &dev,
                )?;
                let got: Vec<u32> = out
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                let mut col = 0usize;
                for (s, &(_, n)) in segs.iter().enumerate() {
                    for r in 0..m {
                        assert_eq!(
                            &got[r * total + col..r * total + col + n],
                            &alone[s][r * n..(r + 1) * n],
                            "{m} rows, {warps} warps: segment {s} row {r}"
                        );
                    }
                    col += n;
                }
            }
        }
        Ok(())
    }

    /// A narrow launch is bit-repeatable launch after launch: the slot sum runs in tile order in
    /// one warp, whatever order the warps finish in.
    #[test]
    fn a_narrow_launch_is_bit_repeatable() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (w, x) = operands(&dev, GgmlDType::Q8_0, 5, 416, 10_240, 7)?;
        let plan = DensePlan::Narrow { warps: 8 };
        let first = bits(&dev, &w, &x, SumScale::Raw, DType::F32, plan)?;
        for i in 0..20 {
            assert_eq!(
                bits(&dev, &w, &x, SumScale::Raw, DType::F32, plan)?,
                first,
                "launch {i} differs"
            );
        }
        Ok(())
    }

    /// The rule's own plan at decode width runs narrow and gives the unsplit bits; the same row
    /// in a split launch and in an unsplit wide one agrees too — a row's output does not depend
    /// on the launch its wave width picked.
    #[test]
    fn a_row_is_the_same_under_every_plan() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let sm = dev.multiprocessor_count()?;
        let (n, k) = (416usize, 10_240usize);
        let (w, wide) = operands(&dev, GgmlDType::Q8_0, 40, n, k, 11)?;
        let wide_out = f32_bits(&dev, &w, &wide, DensePlan::Unsplit)?;
        for m in [1usize, 5, 8] {
            let plan = q8a128_dense_plan(m, n, k, sm, true, weight(&w).2);
            assert!(
                matches!(plan, DensePlan::Narrow { .. }),
                "{m} rows run narrow, not {plan:?}"
            );
            let few = wide.narrow(0, 0, m)?.contiguous()?;
            assert_eq!(
                f32_bits(&dev, &w, &few, plan)?[..],
                wide_out[..m * n],
                "{m} rows, narrow"
            );
            let splits = q8a128_dense_k_splits(m, n, k, sm);
            assert_eq!(
                f32_bits(&dev, &w, &few, DensePlan::SplitK(splits))?[..],
                wide_out[..m * n],
                "{m} rows, {splits} slices"
            );
        }
        Ok(())
    }

    /// Launches outside the narrow geometry are refused, not run: nine rows, no warps,
    /// seventeen warps, and MXFP4.
    #[test]
    fn a_narrow_launch_outside_its_geometry_is_refused() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (w, x) = operands(&dev, GgmlDType::Q8_0, 9, 416, 2560, 3)?;
        let four = DensePlan::Narrow { warps: 4 };
        assert!(f32_bits(&dev, &w, &x, four).is_err());
        let five = x.narrow(0, 0, 5)?.contiguous()?;
        for warps in [0usize, 17] {
            assert!(
                f32_bits(&dev, &w, &five, DensePlan::Narrow { warps }).is_err(),
                "{warps} warps"
            );
        }
        let (mx, mx_x) = operands(&dev, GgmlDType::MXFP4, 1, 416, 2048, 5)?;
        assert!(f32_bits(&dev, &mx, &mx_x, four).is_err());
        // MXFP4's own plan is the unsplit kernel.
        assert_eq!(
            q8a128_dense_plan(
                1,
                416,
                2048,
                dev.multiprocessor_count()?,
                false,
                GgmlDType::MXFP4_KO
            ),
            DensePlan::Unsplit
        );
        Ok(())
    }

    /// Narrow against split-K and unsplit on the Flash-Next decode projections, weights rotated
    /// past the L2: device µs per launch (captured and replayed, so no host time is in it),
    /// beside the DRAM floor of the weight bytes at 1.34 TB/s. `*` marks the rule's plan.
    #[test]
    #[ignore = "benchmark — run explicitly with --ignored, card to itself"]
    fn bench_narrow_against_split_on_decode_projections() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let sm = dev.multiprocessor_count()?;
        // (label, source format, N, K).
        let shapes = [
            ("hyper down q8", GgmlDType::Q8_0, 416usize, 10_240usize),
            ("deltanet out q8", GgmlDType::Q8_0, 2560, 6144),
            ("attn out q8", GgmlDType::Q8_0, 2560, 4096),
            ("shexp down q8", GgmlDType::Q8_0, 2560, 640),
            ("stacked trio q8", GgmlDType::Q8_0, 1824, 2560),
            ("deltanet out q4", GgmlDType::Q4_K, 2560, 6144),
            ("hyper down q4", GgmlDType::Q4_K, 416, 10_240),
            // Wider outputs that still split but cannot hold every narrow block in one wave.
            ("qwen3-8b o q8", GgmlDType::Q8_0, 4096, 4096),
            ("qwen3-8b qkv q8", GgmlDType::Q8_0, 6144, 4096),
            // The dense models' own projections in their checkpoint formats: Qwen3-8B (Q6_K),
            // Qwen3-30B-A3B's attention (Q4_K) and Qwen2-0.5B (Q4_0).
            ("qwen3-8b o q6", GgmlDType::Q6_K, 4096, 4096),
            ("qwen3-8b qkv q6", GgmlDType::Q6_K, 6144, 4096),
            ("qwen3-8b down q6", GgmlDType::Q6_K, 4096, 12_288),
            ("qwen3-8b down q8", GgmlDType::Q8_0, 4096, 12_288),
            ("4096 o q5", GgmlDType::Q5_K, 4096, 4096),
            ("4096 down q5", GgmlDType::Q5_K, 4096, 12_288),
            ("4096 o q4", GgmlDType::Q4_K, 4096, 4096),
            ("4096 down q4", GgmlDType::Q4_K, 4096, 12_288),
            ("qwen3-30b o q4", GgmlDType::Q4_K, 2048, 4096),
            ("qwen2-0.5b o q4", GgmlDType::Q4_0, 896, 896),
            ("qwen2-0.5b down q4", GgmlDType::Q4_0, 896, 4864),
        ];
        let warp_counts = [4usize, 5, 6, 8, 10, 12, 16];
        let mut header = format!(
            "{:<16} {:>2} {:>6} {:>8} {:>8} {:>6}",
            "shape", "M", "floor", "unsplit", "split", "depth"
        );
        for w in warp_counts {
            header.push_str(&format!(" {:>8}", format!("n{w}")));
        }
        println!("{header}  (device µs per launch)");
        let out_code = out_dtype_code(DType::BF16)?;
        for (label, src, n, k) in shapes {
            let (w, x_all) = operands(&dev, src, 32, n, k, (n + k) as u64)?;
            let (_, len, dtype, _) = weight(&w);
            let qtype = dtype_to_qtype(dtype)? as i32;
            let (w_ptr, _, _, _) = weight(&w);
            let rot = Rotation::of_device_bytes(&dev, w_ptr, len)?;
            let weights = rot.ptrs(&dev);
            let floor = len as f64 / 1.34e6;
            // One output and one split scratch, allocated outside the timing; the counters
            // start at zero and every split launch returns them there.
            // SAFETY: fully written by every launch before it is read.
            let dst = unsafe { dev.alloc::<u16>(32 * n)? };
            // SAFETY: every partial is written by its slice before the last block reads it.
            let ws = unsafe { dev.alloc::<f32>(dense_k_tiles(k) * 32 * n)? };
            // One counter per (token tile, row tile): two token tiles at 32 rows.
            let counters = dev.alloc_zeros::<u32>(2 * n.div_ceil(32))?;
            let stream = dev.cuda_stream();
            let (dst_p, _dg) = dst.device_ptr(&stream);
            let (ws_p, _wg) = ws.device_ptr(&stream);
            let (ctr_p, _cg) = counters.device_ptr(&stream);
            // Every width the split rule serves: the narrow kernel's (up to eight rows) and the
            // wider decode and verify waves that split without it.
            for m in [1usize, 5, 8, 10, 16, 32] {
                let x = x_all.narrow(0, 0, m)?.contiguous()?;
                let acts = to_dynamic(&x, Int8Mode::Performance, &dev, SumScale::Raw)?;
                let DynamicActs::Int8(op) = &acts else {
                    unreachable!("an int8 mode quantizes")
                };
                let sum_norm = op.sum_scale.as_code();
                let rule = q8a128_dense_plan(m, n, k, sm, true, dtype);
                let splits = q8a128_dense_k_splits(m, n, k, sm).max(dense_k_split_depth(k, 2));
                let tile = q8a128_dense_tile(m, n, k, sm).code();
                let row = op.with_device_ptr(&dev, |act| {
                    let act = act as *const c_void;
                    let dst = dst_p as *mut c_void;
                    let check = |code: i32, what: &str| check_matmul_status(code, what);
                    let unsplit = time_graph(&dev, &weights, |w, cs| {
                        let seg = VxSegment {
                            weights: w as *const c_void,
                            batch_count: m as i32,
                        };
                        // SAFETY: a KO weight of `len` bytes, `n` rows; the operand and `dst`
                        // hold `m` rows.
                        check(
                            unsafe {
                                run_quantized_matmul(
                                    &seg,
                                    1,
                                    act,
                                    dst,
                                    k as i32,
                                    n as i32,
                                    m as i32,
                                    n as i32,
                                    qtype,
                                    YType::Q8A128 as i32,
                                    len,
                                    tile,
                                    out_code,
                                    sum_norm,
                                    cs as *mut c_void,
                                )
                            },
                            "unsplit",
                        )
                    })?;
                    let rows = [n as i32];
                    let split = time_graph(&dev, &weights, |w, cs| {
                        let wp = [w as *const c_void];
                        // SAFETY: as above; the scratch holds K-tiles × m × n partials and a
                        // zeroed counter per row tile.
                        check(
                            unsafe {
                                run_dense_int8_splitk(
                                    wp.as_ptr(),
                                    rows.as_ptr(),
                                    1,
                                    act,
                                    dst,
                                    k as i32,
                                    m as i32,
                                    qtype,
                                    out_code,
                                    sum_norm,
                                    splits as i32,
                                    ws_p as *mut f32,
                                    ctr_p as *mut u32,
                                    cs as *mut c_void,
                                )
                            },
                            "split",
                        )
                    })?;
                    let mut narrow = Vec::new();
                    for warps in warp_counts {
                        if m > NARROW_MAX_ROWS {
                            break;
                        }
                        narrow.push(time_graph(&dev, &weights, |w, cs| {
                            let wp = [w as *const c_void];
                            // SAFETY: as above.
                            check(
                                unsafe {
                                    run_dense_int8_narrow(
                                        wp.as_ptr(),
                                        rows.as_ptr(),
                                        1,
                                        act,
                                        dst,
                                        k as i32,
                                        m as i32,
                                        qtype,
                                        out_code,
                                        sum_norm,
                                        warps as i32,
                                        cs as *mut c_void,
                                    )
                                },
                                "narrow",
                            )
                        })?);
                    }
                    Ok((unsplit, split, narrow))
                })?;
                let (unsplit, split, narrow) = row;
                let mark = |p: DensePlan| if p == rule { "*" } else { " " };
                let mut line = format!(
                    "{label:<16} {m:>2} {floor:>6.2} {unsplit:>7.2}{} {split:>7.2}{} {splits:>6}",
                    mark(DensePlan::Unsplit),
                    mark(DensePlan::SplitK(splits)),
                );
                for (warps, us) in warp_counts.iter().zip(&narrow) {
                    line.push_str(&format!(
                        " {us:>7.2}{}",
                        mark(DensePlan::Narrow { warps: *warps })
                    ));
                }
                println!("{line}");
            }
        }
        Ok(())
    }
}
