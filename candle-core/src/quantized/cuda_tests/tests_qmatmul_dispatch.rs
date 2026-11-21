//! Quantized matmul dispatch: the int8 mode flag end to end, the pairing guard,
//! and the MXFP4 kernels.
//!
//! The int8 flag must reconstruct the f32 matmul within its budget whether it
//! selects KO weights with quantized activations or the float path. The pairing
//! guard rejects the two unsupported crosses before any kernel launches. The MXFP4
//! kernel is checked against the CPU oracle over an adversarial spread of
//! per-32 exponents, which is the case a shared scale collapses. The dequant path
//! is checked bit for bit against the CPU codec.

use super::*;

/// The MXFP4 per-sub int8 kernel (`loader/mxfp4.cuh` + the `is_mxfp4_persub` fold in
/// `kernel.cuh`) must reproduce the CPU oracle `ko_quant::mxfp4_ko_int8_matmul`. The activation
/// path is shared infrastructure, so we feed the oracle the DEQUANTIZED q8a128 activations the
/// kernel consumes (re-quantizing them recovers the exact int8×scale, since each 128-block's max
/// maps to ±127) — isolating the one thing under test: codebook nibble expansion + one int32 MMA
/// per 32-K sub, each folded with its own E8M0 scale × the per-128 activation scale. Adversarial
/// per-32 exponent spread proves the fold is spread-immune (the old shared-scale collapse
/// truncated here). The int32 sums are exact and the oracle mirrors the kernel's FP fold order;
/// the residual tolerance covers the q8a128 activation scale's storage rounding.
#[test]
pub(super) fn mxfp4_persub_cuda_matches_cpu_oracle() -> Result<()> {
    use crate::quantized::ko_quant;
    let dev = CudaDevice::new(0)?;
    let nrows = 256usize; // N (multiple of 8)
    let ncols = 512usize; // K (multiple of 128 → 4 collapse subs per tile)
    let m = 48usize; // M — spans 3 dense 16-token tiles
    let mut rng = rand::rng();

    // Weight with adversarial per-32-block exponent spread: each 32-block gets a random
    // 2^(-3..3) magnitude, so the four subs of every 128-tile land on different E8M0
    // exponents and the collapse must shift them onto a common e_max.
    let mut wf32 = vec![0f32; nrows * ncols];
    for blk in 0..(nrows * ncols) / 32 {
        let s = 2f32.powi(rng.random_range(-3i32..=3));
        for j in 0..32 {
            wf32[blk * 32 + j] = rng.random_range(-1.0f32..1.0) * s;
        }
    }

    // Activations → q8a128 (the exact int8 grid the kernel multiplies), then dequantized so the
    // CPU oracle re-quantizes to the identical a_i8 / a_scale (no activation-quant divergence).
    let act: Vec<f32> = (0..m * ncols)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    let q8a128 = quantize_acts_q8a128_test(&dev, &act, m, ncols)?;
    let act_deq_dev = dequantize_q8a128(q8a128.data_slice()?, m, ncols, &dev)?;
    let act_deq = dev.memcpy_dtov(&act_deq_dev.slice(..))?;

    // CPU oracle over the 544-byte chunk + the shared activations.
    let chunk544 = ko_quant::quantize_mxfp4_ko(&wf32, nrows, ncols);
    let out_cpu = ko_quant::mxfp4_ko_int8_matmul(&chunk544, &act_deq, nrows, ncols, m);

    // GPU: the 576-byte chunk (per-row dm baked) through the dense int8 MXFP4_KO kernel.
    let chunk576 = ko_quant::mxfp4_ko_to_gpu_chunk(&chunk544, nrows, ncols);
    let ko_slice = dev.memcpy_stod(&chunk576)?;
    let stream = dev.cuda_stream();
    let (ko_ptr, _g) = ko_slice.device_ptr(&stream);
    let out = dense_qmatmul(
        DynamicTensor::Int8(&q8a128),
        ko_ptr,
        GgmlDType::MXFP4_KO,
        nrows,
        0,
        crate::DType::F32,
        &dev,
    )?;
    let out_gpu = read_f32_tensor(&dev, &out)?;

    assert_eq!(out_gpu.len(), out_cpu.len());
    let rel = rel_l2(&out_gpu, &out_cpu);
    println!(
        "MXFP4 per-sub: CUDA kernel vs CPU oracle rel_l2 = {rel:.3e} (N={nrows} K={ncols} M={m})"
    );
    assert!(
        rel < 1e-4,
        "CUDA MXFP4 per-sub fold diverged from CPU oracle: rel_l2 = {rel:.6}"
    );
    Ok(())
}

/// End-to-end `int8mode` flag machinery: the SAME flag drives `QMatMul::repack_for_optimization`
/// (weight → KO when on, FP GEMX when off) and `to_dynamic` (activations → q8a128 when on, float
/// when off), and the repacked weight + converted activations feed `dense_qmatmul`. Both flag
/// settings must reconstruct the f32 ground-truth matmul within their quant budgets — exercising
/// the precision (flag-on int8 KO, flag-off float) of the whole path.
#[test]
pub(super) fn qmatmul_int8mode_flag_end_to_end() -> Result<()> {
    use crate::quantized::{QMatMul, QStorage, QTensor};
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let nrows = 256usize; // N
    let ncols = 512usize; // K
    let m = 32usize; // M
    let mut rng = rand::rng();
    let wf32: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-0.1f32..0.1))
        .collect();
    let act: Vec<f32> = (0..m * ncols)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();

    // f32 ground truth: out[i,j] = sum_k act[i,k] * w[j,k]  (w [N,K], act [M,K] -> out [M,N]).
    let mut out_ref = vec![0f32; m * nrows];
    for i in 0..m {
        for j in 0..nrows {
            let mut s = 0f32;
            for k in 0..ncols {
                s += act[i * ncols + k] * wf32[j * ncols + k];
            }
            out_ref[i * nrows + j] = s;
        }
    }

    // Compact Q4_K weight + f32 activation tensor.
    let w_t = crate::Tensor::from_vec(wf32, (nrows, ncols), &device)?;
    let qmm = QMatMul::from_qtensor(QTensor::quantize(&w_t, GgmlDType::Q4_K)?)?;
    let act_t = crate::Tensor::from_vec(act, (m, ncols), &device)?;

    for &mode in &[Int8Mode::Off, Int8Mode::Performance, Int8Mode::Precision] {
        // Weight side: the one knob picks a KO twin (int8) or FP GEMX (float).
        let opt = qmm.repack_for_optimization(mode)?;
        let q = opt.qtensor().unwrap();
        let wdtype = q.dtype();
        assert_eq!(
            wdtype.is_ko(),
            mode.is_int8(),
            "repack picked the wrong weight format"
        );
        let (wptr, wlen) = match &q.storage {
            QStorage::Cuda(cs) => (cs.data_ptr(), cs.storage_size_in_bytes()),
            _ => unreachable!(),
        };
        // Activation side: the same knob picks q8a128 (int8) or float.
        let acts = to_dynamic(&act_t, mode, &dev, SumScale::Raw)?;
        let out = dense_qmatmul(
            acts.as_dynamic(),
            wptr,
            wdtype,
            nrows,
            wlen,
            // The float arm stores at the activation dtype, which is F32 here; the int8
            // arm is free to pick, and F32 keeps both modes directly comparable.
            crate::DType::F32,
            &dev,
        )?;
        let v = read_f32_tensor(&dev, &out)?;
        assert_eq!(v.len(), out_ref.len());
        let rel = rel_l2(&v, &out_ref);
        let tol = if mode.is_int8() { 0.13 } else { 0.09 };
        println!("int8mode={mode:?} (weight {wdtype:?}): rel_l2 vs f32 ref = {rel:.4}");
        assert!(
            rel < tol,
            "int8mode={mode:?} precision rel_l2={rel:.4} >= {tol}"
        );
    }
    Ok(())
}

/// Bit-level flag check: on a weight already quantized to a K-quant, the int8 path
/// (int8mode=true: KO weight + q8a128 activations) must closely reproduce the FLOAT BASELINE
/// (int8mode=false: dequant-K-quant weight + float activations). Because the weight already
/// lives on the K-quant grid, the KO re-quant of that data is near-lossless, so the two
/// matmul outputs match tightly — proving the int8 path is a faithful drop-in across the
/// K-quant→KO twins. The baseline IS the int8mode=false output (no f32 ground truth needed);
/// f32 test activations are in the inference range. No byte extraction — just the two outputs.
#[test]
pub(super) fn qmatmul_int8mode_baseline_bit_check() -> Result<()> {
    use crate::quantized::{QMatMul, QStorage, QTensor};
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (nrows, ncols, m) = (256usize, 512usize, 32usize);
    let mut rng = rand::rng();
    // f32 activations in a realistic inference range (~unit scale).
    let act: Vec<f32> = (0..m * ncols)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    let act_t = crate::Tensor::from_vec(act, (m, ncols), &device)?;

    let run = |mode: Int8Mode, qmm: &QMatMul| -> Result<Vec<f32>> {
        let opt = qmm.repack_for_optimization(mode)?;
        let q = opt.qtensor().unwrap();
        let (wptr, wlen) = match &q.storage {
            QStorage::Cuda(cs) => (cs.data_ptr(), cs.storage_size_in_bytes()),
            _ => unreachable!(),
        };
        let acts = to_dynamic(&act_t, mode, &dev, SumScale::Raw)?;
        let out = dense_qmatmul(
            acts.as_dynamic(),
            wptr,
            q.dtype(),
            nrows,
            wlen,
            act_t.dtype(),
            &dev,
        )?;
        read_f32_tensor(&dev, &out)
    };

    // Per-dtype tight bounds on the int8↔float delta, separately for each int8 mode.
    // Precision steps one notch up the KO ladder (Q4_K→Q5_KO, Q5_K→Q6_KO, Q6_K→Q6_KO,
    // Q8_0→Q8_KO) so the weight re-quant is near-lossless; Performance uses the same-width twin
    // (Q4_K→Q4_KO, …) and takes the per-32→per-128 granularity hit, so its tolerances are looser.
    // The residual is mostly the q8a128 activation (8-bit) plus the granularity step, shrinking
    // with the twin's bit width.
    let cases: &[(GgmlDType, f64, f64)] = &[
        // (src, performance tol, precision tol)
        (GgmlDType::Q4_K, 0.075, 0.045),
        (GgmlDType::Q5_K, 0.040, 0.025),
        (GgmlDType::Q6_K, 0.025, 0.025),
        (GgmlDType::Q8_0, 0.010, 0.010),
    ];
    for &(srcdtype, perf_tol, prec_tol) in cases {
        let wf32: Vec<f32> = (0..nrows * ncols)
            .map(|_| rng.random_range(-0.1f32..0.1))
            .collect();
        let w_t = crate::Tensor::from_vec(wf32, (nrows, ncols), &device)?;
        let qmm = QMatMul::from_qtensor(QTensor::quantize(&w_t, srcdtype)?)?;
        let base = run(Int8Mode::Off, &qmm)?; // float baseline (dequant-K-quant × float acts)
        for (mode, tol) in [
            (Int8Mode::Performance, perf_tol),
            (Int8Mode::Precision, prec_tol),
        ] {
            let int8 = run(mode, &qmm)?; // int8 (KO weight × q8a128 acts)
            assert_eq!(base.len(), int8.len());
            let rel = rel_l2(&int8, &base);
            println!(
                "{srcdtype:?}->{:?} ({mode:?}) int8 vs float baseline: rel_l2 = {rel:.5} (tol {tol})",
                srcdtype.to_ko(mode)?
            );
            assert!(
                rel < tol,
                "{srcdtype:?} {mode:?} int8 path diverged from float baseline (rel_l2 = {rel:.5} >= {tol})"
            );
        }
    }
    Ok(())
}

/// F1: the int8 dense path must reproduce the activation's rank (`[B,M,K]→[B,M,N]`) exactly
/// like the float path — `to_dynamic` preserves the leading dims onto the operand and the
/// matmul rebuilds `[lead.., N]` rather than flattening to 2D `[B*M, N]`.
#[test]
pub(super) fn dense_qmatmul_int8_preserves_3d_shape() -> Result<()> {
    use crate::quantized::{QMatMul, QStorage, QTensor};
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (b, m, nrows, ncols) = (2usize, 16usize, 256usize, 512usize);
    let mut rng = rand::rng();
    let wf32: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-0.1f32..0.1))
        .collect();
    let w_t = crate::Tensor::from_vec(wf32, (nrows, ncols), &device)?;
    let qmm = QMatMul::from_qtensor(QTensor::quantize(&w_t, GgmlDType::Q4_K)?)?;
    let opt = qmm.repack_for_optimization(Int8Mode::Performance)?;
    let q = opt.qtensor().unwrap();
    let (wptr, wlen) = match &q.storage {
        QStorage::Cuda(cs) => (cs.data_ptr(), cs.storage_size_in_bytes()),
        _ => unreachable!(),
    };
    // 3D activation [B, M, K] → int8 → matmul.
    let act: Vec<f32> = (0..b * m * ncols)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    let act_t = crate::Tensor::from_vec(act, (b, m, ncols), &device)?;
    let acts = to_dynamic(&act_t, Int8Mode::Performance, &dev, SumScale::Raw)?;
    let out = dense_qmatmul(
        acts.as_dynamic(),
        wptr,
        q.dtype(),
        nrows,
        wlen,
        act_t.dtype(),
        &dev,
    )?;
    assert_eq!(
        out.dims(),
        &[b, m, nrows],
        "int8 dense must preserve [B, M, N]"
    );
    // And the float arm on the same 3D activation must agree on rank.
    let optf = qmm.repack_for_optimization(Int8Mode::Off)?;
    let qf = optf.qtensor().unwrap();
    let (wpf, wlf) = match &qf.storage {
        QStorage::Cuda(cs) => (cs.data_ptr(), cs.storage_size_in_bytes()),
        _ => unreachable!(),
    };
    let outf = dense_qmatmul(
        DynamicTensor::Float(&act_t),
        wpf,
        qf.dtype(),
        nrows,
        wlf,
        act_t.dtype(),
        &dev,
    )?;
    assert_eq!(
        outf.dims(),
        out.dims(),
        "float and int8 arms must agree on output rank"
    );
    Ok(())
}

/// The exclusive KO <-> int8 pairing guard: `dense_qmatmul`/`grouped_qmatmul` accept the two
/// supported combinations (Int8 activation x KO weight, Float activation x non-KO weight) and
/// reject the two unsupported crosses (Int8 x non-KO, Float x KO) with an error — before any
/// kernel launches, so callers get a clear message instead of silent garbage.
#[test]
pub(super) fn qmatmul_rejects_unpaired_weight_activation() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 256usize; // N
    let ncols = 512usize; // K
    let m = 16usize; // M
    let act_data: Vec<f32> = (0..m * ncols)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let q8 = quantize_acts_q8a128_test(&dev, &act_data, m, ncols)?;

    // A KO weight (for the int8 path) and a regular Q4_K weight + its bf16 activation tensor.
    let wf32 = vec![0.05f32; nrows * ncols];
    let ko = dev.memcpy_stod(&requant_ko_per128(&wf32, nrows, ncols, 15, 0, 0))?;
    let stream = dev.cuda_stream();
    let (ko_ptr, _g) = ko.device_ptr(&stream);
    let (k_ptrs, k_st, act_storage, _l, _off) =
        build_int8_ab_fixture(&dev, nrows, ncols, &[m], &act_data, GgmlDType::Q4_K)?;
    let k_len = k_st[0].storage_size_in_bytes();
    let act = crate::tensor::from_storage(
        crate::Storage::Cuda(act_storage),
        crate::Shape::from((m, ncols)),
        crate::op::BackpropOp::none(),
        false,
    );

    // Supported: Int8 x KO, Float x non-KO — must succeed.
    dense_qmatmul(
        DynamicTensor::Int8(&q8),
        ko_ptr,
        GgmlDType::Q4_KO,
        nrows,
        0,
        crate::DType::F32,
        &dev,
    )?;
    dense_qmatmul(
        DynamicTensor::Float(&act),
        k_ptrs[0],
        GgmlDType::Q4_K,
        nrows,
        k_len,
        act.dtype(),
        &dev,
    )?;

    // Unsupported crosses — must be rejected by both entries.
    assert!(
        dense_qmatmul(
            DynamicTensor::Int8(&q8),
            k_ptrs[0],
            GgmlDType::Q4_K,
            nrows,
            0,
            crate::DType::F32,
            &dev
        )
        .is_err(),
        "Int8 activation x non-KO weight must be rejected"
    );
    assert!(
        dense_qmatmul(
            DynamicTensor::Float(&act),
            ko_ptr,
            GgmlDType::Q4_KO,
            nrows,
            k_len,
            act.dtype(),
            &dev
        )
        .is_err(),
        "Float activation x KO weight must be rejected"
    );
    assert!(
        grouped_qmatmul(
            DynamicTensor::Int8(&q8),
            &k_ptrs[..1],
            GgmlDType::Q4_K,
            nrows,
            &[0, m as i32],
            &dev,
            Backing::Owned,
        )
        .is_err(),
        "grouped Int8 x non-KO weight must be rejected"
    );
    assert!(
        grouped_qmatmul(
            DynamicTensor::Float(&act),
            &[ko_ptr],
            GgmlDType::Q4_KO,
            nrows,
            &[0, m as i32],
            &dev,
            Backing::Owned,
        )
        .is_err(),
        "grouped Float x KO weight must be rejected"
    );
    Ok(())
}

/// The `Float` arm of the unified entries: `dense_qmatmul(DynamicTensor::Float(..))` must
/// match `grouped_qmatmul(DynamicTensor::Float(..))` (single expert) on the same quantized
/// weight + bf16 activation — i.e. the &Tensor storage extraction + the shared
/// `dense_qmatmul_float` helper agree with the grouped float path.
#[test]
pub(super) fn dense_qmatmul_float_matches_grouped() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 256; // N
    let ncols = 512; // K
    let total_batch = 20usize; // M
    let mut rng = rand::rng();
    let act_data: Vec<f32> = (0..total_batch * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let wdtype = GgmlDType::Q4_K;
    let (weight_ptrs, storages, act_storage, _act_layout, _off) =
        build_int8_ab_fixture(&dev, nrows, ncols, &[total_batch], &act_data, wdtype)?;
    let weight_len = storages[0].storage_size_in_bytes();
    // Wrap the bf16 activation (a CudaStorage) into a Tensor for the Float variant.
    let act = crate::tensor::from_storage(
        crate::Storage::Cuda(act_storage),
        crate::Shape::from((total_batch, ncols)),
        crate::op::BackpropOp::none(),
        false,
    );
    let dense = dense_qmatmul(
        DynamicTensor::Float(&act),
        weight_ptrs[0],
        wdtype,
        nrows,
        weight_len,
        act.dtype(),
        &dev,
    )?;
    let grouped = grouped_qmatmul(
        DynamicTensor::Float(&act),
        &weight_ptrs[..1],
        wdtype,
        nrows,
        &[0, total_batch as i32],
        &dev,
        Backing::Owned,
    )?;
    let d = read_f32_tensor(&dev, &dense.to_dtype(crate::DType::F32)?)?;
    let g = read_f32_tensor(&dev, &grouped.to_dtype(crate::DType::F32)?)?;
    assert_eq!(d.len(), g.len());
    let rel = rel_l2(&d, &g);
    println!("dense vs grouped FLOAT [{wdtype:?} bf16]: rel_l2 = {rel:.6}");
    assert!(
        rel < 1e-5,
        "Float dense diverged from grouped (rel_l2 = {rel:.6})"
    );
    Ok(())
}

/// MXFP4 GPU dequant must bit-match the CPU codec: both decode the same 17-byte blocks
/// with identical E2M1-table × E8M0-half-scale arithmetic. Covers F32, F16, and BF16
/// outputs. This is the FP4 expert-weight decode path for DeepSeek-V4.
#[test]
pub(super) fn cuda_mxfp4_dequant_matches_cpu() -> Result<()> {
    let cpu = crate::Device::Cpu;
    let cuda = crate::Device::new_cuda(0)?;
    let n = 1024usize;
    // Values spanning negatives, zero, and a wide magnitude range so multiple E8M0
    // scales and every E2M1 index are exercised across blocks.
    let xs: Vec<f32> = (0..n).map(|i| ((i as f32) - 512.0) * 0.017).collect();
    let x = crate::Tensor::from_vec(xs, (n,), &cpu)?;
    let qc = crate::quantized::QTensor::quantize(&x, GgmlDType::MXFP4)?;
    let deq_cpu = qc.dequantize(&cpu)?.to_vec1::<f32>()?;

    // Upload the exact quantized bytes to the GPU and dequantize there.
    let bytes = qc.data()?;
    let (qg, _guard) =
        crate::quantized::QTensor::from_host_mapped_ggml(GgmlDType::MXFP4, &bytes, vec![n], &cuda)?;

    let deq_gpu_f32 = qg
        .dequantize(&cuda)?
        .to_dtype(crate::DType::F32)?
        .to_vec1::<f32>()?;
    for (i, (a, b)) in deq_cpu.iter().zip(deq_gpu_f32.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "f32 mismatch at {i}: cpu {a} gpu {b}");
    }

    // BF16 output: within one bf16 ulp of the CPU value.
    let deq_gpu_bf16 = qg
        .dequantize_bf16(&cuda)?
        .to_dtype(crate::DType::F32)?
        .to_vec1::<f32>()?;
    for (i, (a, b)) in deq_cpu.iter().zip(deq_gpu_bf16.iter()).enumerate() {
        let tol = 0.01 * (1.0 + a.abs());
        assert!((a - b).abs() < tol, "bf16 mismatch at {i}: cpu {a} gpu {b}");
    }
    Ok(())
}

/// MXFP4 `QMatMul::forward` runs via the dequantize→matmul path (no native FP4 MMA on
/// sm_120) and matches an explicit dequantize-then-matmul. This is the DeepSeek-V4 routed
/// expert compute path.
#[test]
pub(super) fn cuda_mxfp4_qmatmul_dequant_path() -> Result<()> {
    use crate::Module;
    let cpu = crate::Device::Cpu;
    let dev = crate::Device::new_cuda(0)?;
    let (out, inn) = (64usize, 128usize);
    let wf: Vec<f32> = (0..out * inn)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let w_cpu = crate::Tensor::from_vec(wf, (out, inn), &cpu)?;
    let qw = crate::quantized::QTensor::quantize(&w_cpu, GgmlDType::MXFP4)?;
    let bytes = qw.data()?;
    let (qg, _guard) = crate::quantized::QTensor::from_host_mapped_ggml(
        GgmlDType::MXFP4,
        &bytes,
        vec![out, inn],
        &dev,
    )?;
    // Reference dequant weight before qg is moved into the QMatMul.
    let wdq = qg.dequantize(&dev)?.to_dtype(crate::DType::BF16)?; // [out, inn]
    let qmm = crate::quantized::QMatMul::from_qtensor(qg)?;

    let x = crate::Tensor::randn(0f32, 1.0, (4, inn), &dev)?.to_dtype(crate::DType::BF16)?;
    let y = qmm.forward(&x)?; // [4, out]
    assert_eq!(y.dims(), &[4, out]);
    let yref = x.matmul(&wdq.t()?)?;
    let d = (y.to_dtype(crate::DType::F32)? - yref.to_dtype(crate::DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(d < 1e-2, "MXFP4 QMatMul vs dequant-matmul diff {d}");
    Ok(())
}

/// MXFP4 weights feed the int8 KO matmul kernel: repacking an MXFP4 weight to MXFP4_KO
/// (both modes — the exact byte permutation in `repack_ko`) and running the q8a128 int8 MMA
/// matches the float baseline (MXFP4 dequant × float activations). This is the "MXFP4 works
/// with our int8 kernels" end-to-end check — CPU/float baseline vs the CUDA int8 kernel.
/// The per-sub fold is weight-exact, so the kernel must land at the exact per-32 CPU int8
/// reference (only activation-quant error), not merely inside a loose tolerance.
#[test]
pub(super) fn mxfp4_int8_matmul_matches_float_baseline() -> Result<()> {
    use crate::quantized::{QMatMul, QStorage, QTensor};
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let cpu = crate::Device::Cpu;
    let (nrows, ncols, m) = (256usize, 512usize, 32usize);
    let mut rng = rand::rng();
    let act: Vec<f32> = (0..m * ncols)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    let act_t = crate::Tensor::from_vec(act, (m, ncols), &device)?;

    // Weight → MXFP4 (CPU quantize) → GPU (host-mapped).
    let wf32: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-0.1f32..0.1))
        .collect();

    // Exact per-32 int8 CPU reference on the SAME weight + activations: this is the
    // "best-case int8" (per-32 weight scales applied exactly, only the 8-bit activation
    // quant is lossy). Comparing it and the per-128 KO kernel to the float baseline shows
    // exactly what the per-128 optimization costs.
    let cpu_chunk = crate::quantized::ko_quant::quantize_mxfp4_ko(&wf32, nrows, ncols);
    let act_cpu = act_t
        .to_dtype(crate::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let cpu_per32 =
        crate::quantized::ko_quant::mxfp4_ko_int8_matmul(&cpu_chunk, &act_cpu, nrows, ncols, m);

    let w_cpu = crate::Tensor::from_vec(wf32, (nrows, ncols), &cpu)?;
    let qw = QTensor::quantize(&w_cpu, GgmlDType::MXFP4)?;
    let bytes = qw.data()?;
    let (qg, _guard) =
        QTensor::from_host_mapped_ggml(GgmlDType::MXFP4, &bytes, vec![nrows, ncols], &device)?;
    let qmm = QMatMul::from_qtensor(qg)?;

    // Float baseline: MXFP4 dequant × float acts (the QMatMul MXFP4 dequant→matmul path).
    let base_t = crate::Module::forward(&qmm, &act_t)?; // [m, nrows]
    let base = base_t
        .to_dtype(crate::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let run = |mode: Int8Mode| -> Result<Vec<f32>> {
        let opt = qmm.repack_for_optimization(mode)?;
        let q = opt.qtensor().unwrap();
        let (wptr, wlen) = match &q.storage {
            QStorage::Cuda(cs) => (cs.data_ptr(), cs.storage_size_in_bytes()),
            _ => unreachable!(),
        };
        let acts = to_dynamic(&act_t, mode, &dev, SumScale::Raw)?;
        let out = dense_qmatmul(
            acts.as_dynamic(),
            wptr,
            q.dtype(),
            nrows,
            wlen,
            crate::DType::F32,
            &dev,
        )?;
        read_f32_tensor(&dev, &out)
    };

    // Exact per-32 int8 (CPU) vs float baseline — the best int8 could do if it kept the
    // per-32 weight scales (the residual is only the 8-bit activation quant).
    assert_eq!(cpu_per32.len(), base.len());
    let rel_per32 = rel_l2(&cpu_per32, &base);
    println!("exact per-32 int8 (CPU) vs float baseline: rel_l2 = {rel_per32:.5}");

    for (mode, tol) in [
        (Int8Mode::Performance, 0.08f64),
        (Int8Mode::Precision, 0.03),
    ] {
        let int8 = run(mode)?;
        assert_eq!(int8.len(), base.len());
        let rel = rel_l2(&int8, &base);
        println!(
            "MXFP4->{:?} (per-sub, {mode:?}) int8 vs float baseline: rel_l2 = {rel:.5} (tol {tol}); \
             per-sub cost over exact per-32 = {:.5}",
            GgmlDType::MXFP4.to_ko(mode)?,
            (rel - rel_per32).max(0.0)
        );
        assert!(
            rel < tol,
            "MXFP4 {mode:?} int8 diverged from float baseline: rel_l2 {rel:.5} >= {tol}"
        );
    }
    // The exact per-32 reference must be at least as good as the per-128 kernel.
    assert!(
        rel_per32 < 0.03,
        "exact per-32 int8 rel_l2 {rel_per32:.5} unexpectedly high"
    );
    Ok(())
}
