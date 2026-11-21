//! Gates for the fused q, k and v int8 matmul.
//!
//! The segmented kernel must match three separate dense int8 matmuls exactly,
//! with mixed KO formats in the grouped-query shape, and it is faster because one
//! occupied launch replaces three that starve on the small key and value tensors.
//! A fourth segment is refused in Rust, because the kernel would read the third
//! segment's weights past their end. A missing format code leaves a segment's
//! columns unwritten without faulting, so the codes are checked against the
//! kernel's range.

use super::*;

/// The same gate for the fused qkv launcher, whose tables are a dimension larger
/// (`[out_dtype][format][mode]` for the uniform fast path, `[out_dtype][mode]` for
/// the mixed one). Runs both: uniform q/k/v formats take the single-format kernel,
/// mixed formats take the switch kernel, and they are separate table rows.
#[test]
pub(super) fn qkv_segmented_narrow_output_matches_f32_bitwise() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let stream = dev.cuda_stream();
    let k = 1024usize;
    let mut rng = rand::rng();

    let uniform: [(usize, GgmlDType, (i32, usize, usize)); 3] = [
        (1024, GgmlDType::Q4_KO, (15, 0, 0)),
        (256, GgmlDType::Q4_KO, (15, 0, 0)),
        (256, GgmlDType::Q4_KO, (15, 0, 0)),
    ];
    let mixed: [(usize, GgmlDType, (i32, usize, usize)); 3] = [
        (1024, GgmlDType::Q4_KO, (15, 0, 0)),
        (256, GgmlDType::Q6_KO, (63, 256, 0)),
        (256, GgmlDType::Q6_KO, (63, 256, 0)),
    ];
    // The narrow twins, in both dispatch paths. `ko_fmt_code` appended them as codes 4 and 5,
    // so they sit past every index the two configurations above reach — and the mixed-format
    // kernel's `default` arm does not fault on an unknown code, it leaves that segment's output
    // columns **unwritten**. A wrong or missing row here is therefore silence, not a crash,
    // which is precisely what a bitwise gate is for.
    let uniform_narrow: [(usize, GgmlDType, (i32, usize, usize)); 3] = [
        (1024, GgmlDType::Q3_KO, (7, 256, 128)),
        (256, GgmlDType::Q3_KO, (7, 256, 128)),
        (256, GgmlDType::Q3_KO, (7, 256, 128)),
    ];
    let mixed_narrow: [(usize, GgmlDType, (i32, usize, usize)); 3] = [
        (1024, GgmlDType::Q3_KO, (7, 256, 128)),
        (256, GgmlDType::Q2_KO, (3, 256, 0)),
        (256, GgmlDType::Q4_KO, (15, 0, 0)),
    ];

    for dims in [&uniform, &mixed, &uniform_narrow, &mixed_narrow] {
        for &m in &[8usize, 512] {
            let act: Vec<f32> = (0..m * k).map(|_| rng.random_range(-1.0f32..1.0)).collect();
            let op = quantize_acts_q8a128_test(&dev, &act, m, k)?;

            let mut slices = Vec::new();
            let mut segs: Vec<(u64, GgmlDType, usize)> = Vec::new();
            for &(n, dtype, (maxq, crumb, hi)) in dims {
                let wf32: Vec<f32> = (0..n * k).map(|_| rng.random_range(-0.1f32..0.1)).collect();
                slices.push(dev.memcpy_stod(&requant_ko_per128(&wf32, n, k, maxq, crumb, hi))?);
                let (ptr, _g) = slices.last().unwrap().device_ptr(&stream);
                segs.push((ptr, dtype, n));
            }

            let wide = read_f32_tensor(
                &dev,
                &qkv_segmented_matmul(&op, &segs, crate::DType::F32, &dev)?,
            )?;
            for out_dtype in [crate::DType::F16, crate::DType::BF16] {
                let narrow =
                    read_narrow_bits(&dev, &qkv_segmented_matmul(&op, &segs, out_dtype, &dev)?)?;
                assert_eq!(narrow.len(), wide.len());
                for (i, (&got, &w)) in narrow.iter().zip(wide.iter()).enumerate() {
                    let want = match out_dtype {
                        crate::DType::F16 => f16::from_f32(w).to_bits(),
                        _ => bf16::from_f32(w).to_bits(),
                    };
                    assert_eq!(
                        got, want,
                        "M={m} {out_dtype:?} element {i}: kernel stored {got:#06x}, \
                         casting the F32 kernel's {w} gives {want:#06x}"
                    );
                }
            }
        }
    }
    Ok(())
}

/// The segmented kernel carries three segments; a fourth would be computed with the third's
/// weights read past their end, so the launcher refuses it — and the stacked form, which falls
/// back to it whenever it does not split K, refuses it too rather than return garbage.
#[test]
pub(super) fn qkv_segmented_refuses_a_fourth_segment() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let (m, k) = (4usize, 256usize);
    let act: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
    let op = quantize_acts_q8a128_test(&dev, &act, m, k)?;
    let segs = vec![(0u64, GgmlDType::Q8_KO, 32usize); QKV_MAX_SEGS + 1];
    let Err(err) = qkv_segmented_matmul(&op, &segs, crate::DType::F32, &dev) else {
        panic!("four segments refused");
    };
    assert!(err.to_string().contains("at most 3"), "{err}");
    Ok(())
}

/// Fused qkv segmented int8 matmul must be FLOAT-IDENTICAL to running q/k/v as three separate
/// dense int8 matmuls — with MIXED KO formats (q=Q4_KO, k/v=Q6_KO, the real GQA case) over the
/// shared q8a128 activation — and faster (one occupied launch vs three, the tiny k/v no longer
/// starve). Run perf with `--nocapture`.
#[test]
pub(super) fn qkv_segmented_matches_separate() -> Result<()> {
    use std::time::Instant;
    let dev = CudaDevice::new(0)?;
    // The separate references are plain dense launches, which split K at decode
    // width: their scratch is made where a KO weight is built, and these weights
    // are raw device buffers.
    ensure_split_k_scratch(&dev)?;
    let stream = dev.cuda_stream();
    let k = 2048usize; // hidden / contraction

    // (N, KO dtype, requant params) for q, k, v — GQA shape: big q, small k/v; mixed formats.
    let dims: [(usize, GgmlDType, (i32, usize, usize)); 3] = [
        (4096, GgmlDType::Q4_KO, (15, 0, 0)),
        (512, GgmlDType::Q6_KO, (63, 256, 0)),
        (512, GgmlDType::Q6_KO, (63, 256, 0)),
    ];
    let n_total: usize = dims.iter().map(|d| d.0).sum();

    // Deterministic rather than `rng.random_range` per element: this fixture is
    // ~10M weights plus the activations, and a regression test whose input
    // changes every run reports that it broke without reporting what on.
    let fill = |salt: usize, i: usize, scale: f32| -> f32 {
        let h = (salt as u32)
            .wrapping_mul(2_654_435_761)
            .wrapping_add((i as u32).wrapping_mul(40_503));
        ((h >> 20) as f32 / 2048.0 - 1.0) * scale
    };

    // **The weights do not depend on M.** They were built inside the loop below,
    // so all three M values paid for the same 10M-element generation and the same
    // `requant_ko_per128` over it — three times the fixture for one test.
    let mut slices = Vec::new();
    let mut segs: Vec<(u64, GgmlDType, usize)> = Vec::new();
    for (s, &(n, dtype, (maxq, crumb, hi))) in dims.iter().enumerate() {
        let wf32: Vec<f32> = (0..n * k).map(|i| fill(s + 1, i, 0.1)).collect();
        let ob = requant_ko_per128(&wf32, n, k, maxq, crumb, hi);
        let ko = dev.memcpy_stod(&ob)?;
        slices.push(ko);
        let (ptr, _g) = slices.last().unwrap().device_ptr(&stream);
        segs.push((ptr, dtype, n));
    }

    for &m in &[4usize, 64, 256] {
        let act: Vec<f32> = (0..m * k).map(|i| fill(0, i, 1.0)).collect();
        let op = quantize_acts_q8a128_test(&dev, &act, m, k)?;

        let mut sep_refs: Vec<Vec<f32>> = Vec::new();
        for &(ptr, dtype, n) in &segs {
            let r = dense_qmatmul(
                DynamicTensor::Int8(&op),
                ptr,
                dtype,
                n,
                0,
                crate::DType::F32,
                &dev,
            )?;
            sep_refs.push(read_f32_tensor(&dev, &r)?);
        }

        // Fused.
        let fused = qkv_segmented_matmul(&op, &segs, crate::DType::F32, &dev)?;
        let f = read_f32_tensor(&dev, &fused)?;
        assert_eq!(f.len(), m * n_total);

        // Compare each segment's [m, n] block (fused columns [col, col+n)) to the separate result.
        let mut col = 0usize;
        for (i, &(n, dt, _)) in dims.iter().enumerate() {
            let mut seg = vec![0f32; m * n];
            for row in 0..m {
                seg[row * n..row * n + n]
                    .copy_from_slice(&f[row * n_total + col..row * n_total + col + n]);
            }
            let rel = rel_l2(&seg, &sep_refs[i]);
            println!("qkv_segmented M={m} seg{i} {dt:?} N={n}: rel_l2 vs separate = {rel:.6}");
            assert!(
                rel < 1e-5,
                "seg{i} {dt:?} diverged at M={m}: rel_l2={rel:.6}"
            );
            col += n;
        }

        // Perf: fused (1 launch) vs 3 separate, median of repeats.
        const ITERS: usize = 40;
        let time = |run: &dyn Fn() -> Result<()>| -> Result<f64> {
            for _ in 0..3 {
                run()?;
            }
            dev.synchronize()?;
            let mut s = Vec::with_capacity(ITERS);
            for _ in 0..ITERS {
                let t0 = Instant::now();
                run()?;
                dev.synchronize()?;
                s.push(t0.elapsed().as_secs_f64() * 1e6);
            }
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Ok(s[s.len() / 2])
        };
        let us_fused = time(&|| {
            qkv_segmented_matmul(&op, &segs, crate::DType::F32, &dev)?;
            Ok(())
        })?;
        let us_sep = time(&|| {
            for &(ptr, dt, n) in &segs {
                let _ = dense_qmatmul(
                    DynamicTensor::Int8(&op),
                    ptr,
                    dt,
                    n,
                    0,
                    crate::DType::F32,
                    &dev,
                )?;
            }
            Ok(())
        })?;
        println!(
            "qkv_segmented M={m}: fused={us_fused:.1}us  separate={us_sep:.1}us  speedup={:.2}x",
            us_sep / us_fused
        );
    }
    Ok(())
}

/// Decision test: for SAME-dtype q/k/v, is the segmented kernel actually faster than the single
/// concatenated matmul (the `concat_rows_cuda` fusion)? If concat wins, "always segmented" is a
/// regression on uniform-quant models and we should keep concat for same-dtype.
#[test]
#[ignore = "perf decision: run with --ignored --nocapture"]
pub(super) fn qkv_segmented_vs_concat_same_format() -> Result<()> {
    use std::time::Instant;
    let dev = CudaDevice::new(0)?;
    let stream = dev.cuda_stream();
    let k = 2048usize;
    let (nq, nkv) = (4096usize, 512usize);
    let n_total = nq + nkv + nkv;
    let mut rng = rand::rng();

    for &m in &[1usize, 4, 64] {
        let act: Vec<f32> = (0..m * k).map(|_| rng.random_range(-1.0f32..1.0)).collect();
        let op = quantize_acts_q8a128_test(&dev, &act, m, k)?;

        // Three SAME-format (Q4_KO) weights. Concat bytes == the three appended (per-row quant).
        let mut all = Vec::new();
        let mut segs: Vec<(u64, GgmlDType, usize)> = Vec::new();
        let mut slices = Vec::new();
        let mut concat_bytes = Vec::new();
        for &n in &[nq, nkv, nkv] {
            let wf32: Vec<f32> = (0..n * k).map(|_| rng.random_range(-0.1f32..0.1)).collect();
            let ob = requant_ko_per128(&wf32, n, k, 15, 0, 0);
            concat_bytes.extend_from_slice(&ob);
            let ko = dev.memcpy_stod(&ob)?;
            slices.push(ko);
            let (ptr, _g) = slices.last().unwrap().device_ptr(&stream);
            segs.push((ptr, GgmlDType::Q4_KO, n));
            all.push(());
        }
        let concat_ko = dev.memcpy_stod(&concat_bytes)?;
        let (concat_ptr, _gc) = concat_ko.device_ptr(&stream);

        const ITERS: usize = 50;
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            for _ in 0..5 {
                f()?;
            }
            dev.synchronize()?;
            let mut s = Vec::with_capacity(ITERS);
            for _ in 0..ITERS {
                let t0 = Instant::now();
                f()?;
                dev.synchronize()?;
                s.push(t0.elapsed().as_secs_f64() * 1e6);
            }
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Ok(s[s.len() / 2])
        };
        let us_concat = time(&|| {
            let _ = dense_qmatmul(
                DynamicTensor::Int8(&op),
                concat_ptr,
                GgmlDType::Q4_KO,
                n_total,
                0,
                crate::DType::F32,
                &dev,
            )?;
            Ok(())
        })?;
        let us_seg = time(&|| {
            qkv_segmented_matmul(&op, &segs, crate::DType::F32, &dev)?;
            Ok(())
        })?;
        println!(
            "same-fmt M={m}: concat={us_concat:.1}us  segmented={us_seg:.1}us  concat/seg={:.2}",
            us_seg / us_concat
        );
    }
    Ok(())
}
