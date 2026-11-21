//! Correctness gates for the int8 q8a128 matmuls against their float references.
//!
//! The narrowed dense kernels must equal the F32 kernel with the store converted,
//! compared as raw bit patterns, because a tolerance would hide a wrong dispatch
//! row. The grouped and dense int8 paths are checked against the legacy FP16
//! kernel. The 2-bit and 3-bit formats are checked against a CPU reference over
//! the same dequantized weights. The 3-bit test catches a dropped high bit,
//! which produces a smoothly wrong matmul rather than NaN. The wide-tile test
//! requires bit-identity across tile widths, and the outlier test requires
//! graceful degradation.

use super::*;

/// Read a narrow matmul output back as raw bit patterns.
///
/// Bits, not floats: the point of the narrowed dense kernels is that they are the
/// F32 kernel plus the store-time conversion, so the gate compares the exact
/// encoding rather than a tolerance. Widening to f32 first would hide a wrong
/// rounding mode, and a tolerance would hide a whole wrong table row.
#[cfg(test)]
pub(super) fn read_narrow_bits(dev: &CudaDevice, t: &crate::Tensor) -> Result<Vec<u16>> {
    let s = t.storage_and_layout().0;
    let c = match &*s {
        crate::Storage::Cuda(s) => s,
        _ => panic!("expected CUDA storage"),
    };
    match t.dtype() {
        crate::DType::F16 => Ok(dev
            .memcpy_dtov(&c.as_cuda_slice::<f16>()?.slice(..))?
            .iter()
            .map(|v| v.to_bits())
            .collect()),
        crate::DType::BF16 => Ok(dev
            .memcpy_dtov(&c.as_cuda_slice::<bf16>()?.slice(..))?
            .iter()
            .map(|v| v.to_bits())
            .collect()),
        d => panic!("read_narrow_bits: expected a 16-bit float output, got {d:?}"),
    }
}

/// The narrowed dense int8 kernels must be the F32 kernel with the store converted
/// — bit for bit, for every KO format and both tiling modes.
///
/// This is the gate on the dispatch tables. `dense_kernels_int8` is indexed
/// `[out_dtype][format]` and `dense_kernels_int8_m2` `[out_dtype][format - 14]`;
/// a row transposed, shifted, or pointing at the wrong format produces plausible
/// numbers from the wrong weights, which no tolerance test would catch. Comparing
/// against `f16::from_f32` of the F32 kernel's own output pins both the table and
/// the kernel's rounding (`__floats2half2_rn` / `__floats2bfloat162_rn`, both
/// round-to-nearest-even, same as the host cast this replaced).
#[test]
pub(super) fn dense_int8_narrow_output_matches_f32_bitwise() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let stream = dev.cuda_stream();
    let ncols = 1024usize; // K
    let nrows = 512usize; // N
    let mut rng = rand::rng();

    // Every affine KO format the dense table carries, with its requant parameters.
    //
    // **The narrow widths are here because their absence hid a real defect.** This sweep is the
    // only gate on `dense_kernels_int8_m2`, and that table was reached by a literal
    // `kernel_row <= 19` — so when a format landed past 19 it silently fell back to mode-1 and
    // no test noticed, because no test ran a format past 19. Covering every row is what makes
    // this a gate rather than a spot check. MXFP4_KO is absent for a different reason: it is not
    // affine, so `requant_ko_per128` cannot produce it.
    let formats: [(GgmlDType, (i32, usize, usize)); 5] = [
        (GgmlDType::Q2_KO, (3, 256, 0)),
        (GgmlDType::Q3_KO, (7, 256, 128)),
        (GgmlDType::Q4_KO, (15, 0, 0)),
        (GgmlDType::Q5_KO, (31, 0, 128)),
        (GgmlDType::Q6_KO, (63, 256, 0)),
    ];

    // M=8 stays in mode-1 (Bm=16); M=512 crosses into mode-2 (Bm=32 weight-reuse),
    // so both dispatch tables are exercised.
    for &m in &[8usize, 512] {
        let act: Vec<f32> = (0..m * ncols)
            .map(|_| rng.random_range(-1.0f32..1.0))
            .collect();
        let op = quantize_acts_q8a128_test(&dev, &act, m, ncols)?;

        for &(dtype, (maxq, crumb, hi)) in &formats {
            let wf32: Vec<f32> = (0..nrows * ncols)
                .map(|_| rng.random_range(-0.1f32..0.1))
                .collect();
            let ko = dev.memcpy_stod(&requant_ko_per128(&wf32, nrows, ncols, maxq, crumb, hi))?;
            let (ptr, _g) = ko.device_ptr(&stream);

            let wide = read_f32_tensor(
                &dev,
                &dense_qmatmul(
                    DynamicTensor::Int8(&op),
                    ptr,
                    dtype,
                    nrows,
                    0,
                    crate::DType::F32,
                    &dev,
                )?,
            )?;

            for out_dtype in [crate::DType::F16, crate::DType::BF16] {
                let narrow = read_narrow_bits(
                    &dev,
                    &dense_qmatmul(
                        DynamicTensor::Int8(&op),
                        ptr,
                        dtype,
                        nrows,
                        0,
                        out_dtype,
                        &dev,
                    )?,
                )?;
                assert_eq!(narrow.len(), wide.len(), "M={m} {dtype:?} {out_dtype:?}");
                for (i, (&got, &w)) in narrow.iter().zip(wide.iter()).enumerate() {
                    let want = match out_dtype {
                        crate::DType::F16 => f16::from_f32(w).to_bits(),
                        _ => bf16::from_f32(w).to_bits(),
                    };
                    assert_eq!(
                        got, want,
                        "M={m} {dtype:?} {out_dtype:?} element {i}: kernel stored {got:#06x}, \
                         casting the F32 kernel's {w} gives {want:#06x}"
                    );
                }
            }
        }
    }
    Ok(())
}

/// §8.1 correctness gate — the INT8 grouped kernel must match the legacy FP16
/// grouped kernel on the same Q4_K weights (well-conditioned activations). This
/// validates the arithmetic: fragment layouts, the Q4_K min-term correction, and
/// the per-sub scale fold. `randn`-like activations have no outliers, so int8
/// per-32-block quant is near-exact here. Tile sizes 1/8/16 exercise the
/// ceil(m/16) tiling, including a single-token partial tile.
#[test]
#[ignore = "retired: regular-weight int8 is unsupported — the q8a128 path is KO-weight-only \
            (per-128 collapse); see ensure_qmatmul_pairing. Grouped int8 KO is covered by \
            grouped_moe_ko_vs_k_bench."]
pub(super) fn grouped_int8_matches_legacy() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 768; // N = intermediate
    let ncols = 2048; // K = hidden
                      // 24 > 16 forces a second m16 sub-tile (BATCH_TILE_I8 = 32) so the int8 path's
                      // 2-tile loop is validated against the FP16 oracle, not just the ≤16 single-tile case.
    let expert_batches = &[1usize, 8, 16, 24];
    let total_batch: usize = expert_batches.iter().sum();

    let mut rng = rand::rng();
    let act_data: Vec<f32> = (0..total_batch * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let q8a128 = quantize_acts_q8a128_test(&dev, &act_data, total_batch, ncols)?;

    // Each weight format with an int8 unpack is validated against its FP grouped
    // kernel (the oracle). Add formats here as their dequant_to_b_frag_int8 lands.
    for &wdtype in &[
        GgmlDType::Q4_K,
        GgmlDType::Q8_0,
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q5_K,
        GgmlDType::Q6_K,
        GgmlDType::Q3_K,
        GgmlDType::Q2_K,
        GgmlDType::Q8_1,
        GgmlDType::Q8_K,
        GgmlDType::QAWQ,
        GgmlDType::QAWQ_G64,
    ] {
        let (weight_ptrs, _storages, act_storage, act_layout, expert_offsets) =
            build_int8_ab_fixture(&dev, nrows, ncols, expert_batches, &act_data, wdtype)?;
        let int8 = grouped_qmatmul(
            DynamicTensor::Int8(&q8a128),
            &weight_ptrs,
            wdtype,
            nrows,
            &expert_offsets,
            &dev,
            Backing::Owned,
        )?;
        let legacy = grouped_matmul_gemx(
            &weight_ptrs,
            wdtype,
            nrows,
            ncols,
            &act_storage,
            &act_layout,
            &expert_offsets,
            &dev,
        )?;
        let vi = read_f32_tensor(&dev, &int8)?;
        let vl = read_bf16_tensor(&dev, &legacy)?;
        assert_eq!(vi.len(), vl.len());
        let rel = rel_l2(&vi, &vl);
        println!("grouped q8a128 INT8 vs FP16 [{wdtype:?}]: rel_l2 = {rel:.5}");
        assert!(
            rel < 0.03,
            "{wdtype:?} INT8 grouped diverged (rel_l2 = {rel:.5})"
        );
    }
    Ok(())
}

/// Dense (non-MoE) q8a128 INT8 matmul must match the grouped path with a single expert —
/// same INT8 m16n8k32 core, reached through `run_quantized_matmul`'s ytype==3 branch
/// instead of the grouped entry. Exercises every KO weight format the q8a128 path supports
/// after the per-128 collapse (Q4/Q5/Q6/Q8_KO). M < 64 keeps both paths in mode-1, so the
/// int8 accumulation + fold is identical end to end.
#[test]
pub(super) fn dense_int8_matches_grouped() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 256; // N
    let ncols = 512; // K
    let mut rng = rand::rng();
    let wf32: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-0.1f32..0.1))
        .collect();
    let stream = dev.cuda_stream();
    // The dense launch splits K at narrow widths, and its scratch is built with the weight
    // (`QMatMul::from_arc`); these weights are raw device buffers, so build it here.
    ensure_split_k_scratch(&dev)?;

    // (KO weight dtype, maxq, crumb_bytes, hi_bytes) — the int8 weight formats the q8a128
    // path actually supports; Q8_KO takes the symmetric requant path.
    let cases: &[(GgmlDType, i32, usize, usize)] = &[
        (GgmlDType::Q4_KO, 15, 0, 0),
        (GgmlDType::Q5_KO, 31, 0, 128),
        (GgmlDType::Q6_KO, 63, 256, 0),
        (GgmlDType::Q8_KO, 0, 0, 0),
    ];
    // Sweep grouped tile schedules against the dense path they shadow: 20 → one 32-wide tile
    // (16 + partial 4); 40 → two tiles (32 + 8), exercising a 2nd tile at b_start=32; 70 →
    // three tiles (32 + 32 + 6) at b_start 0/32/64, and M>64 also pushes dense into mode-2.
    // This validates the 32-wide tile offsets and the partial trailing sub-tile of the
    // mode-2 weight-reuse kernel — the cases the single-tile check could not reach.
    for total_batch in [20usize, 40, 70] {
        let act_data: Vec<f32> = (0..total_batch * ncols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let q8a128 = quantize_acts_q8a128_test(&dev, &act_data, total_batch, ncols)?;
        for &(kod, maxq, crumb, hi) in cases {
            let ob = if kod == GgmlDType::Q8_KO {
                requant_q8_ko_per128(&wf32, nrows, ncols)
            } else {
                requant_ko_per128(&wf32, nrows, ncols, maxq, crumb, hi)
            };
            let ko_slice = dev.memcpy_stod(&ob)?;
            let (ko_ptr, _g) = ko_slice.device_ptr(&stream);
            let grouped = grouped_qmatmul(
                DynamicTensor::Int8(&q8a128),
                &[ko_ptr],
                kod,
                nrows,
                &[0, total_batch as i32],
                &dev,
                Backing::Owned,
            )?;
            let dense = dense_qmatmul(
                DynamicTensor::Int8(&q8a128),
                ko_ptr,
                kod,
                nrows,
                0,
                crate::DType::F32,
                &dev,
            )?;
            let g = read_f32_tensor(&dev, &grouped)?;
            let d = read_f32_tensor(&dev, &dense)?;
            assert_eq!(g.len(), d.len());
            let rel = rel_l2(&d, &g);
            println!("dense vs grouped INT8 [{kod:?}] M={total_batch}: rel_l2 = {rel:.6}");
            assert!(
                rel < 1e-5,
                "{kod:?} dense diverged from grouped at M={total_batch} (rel_l2 = {rel:.6})"
            );
        }
    }
    Ok(())
}

/// Q2_KO (2-bit affine KO twin) int8 grouped GEMM correctness: the `q2_ko_int8_f32_grouped`
/// kernel + its 2-bit crumb unpack (`loader/q2_KO.cuh`) must reproduce a CPU f32 reference matmul
/// over the SAME weights. The weights are built from random f32 via the CPU codec `quantize_ko`
/// (byte-identical to the GPU `run_quantize_ko`, so exactly what the kernel reads) and the f32
/// reference uses `dequant_ko` of those bytes — so the 2-bit WEIGHT is identical on both sides,
/// isolating the kernel's unpack + per-128 (scale,min) fold. The only divergence is the int8
/// activation quant (well-conditioned uniform activations → ~1%). A wrong crumb unpack, byte
/// order, or fold produces gross error / NaN, not ~1%. Tile widths 1/8/16/24 exercise the
/// ceil(m/16) tiling incl. a partial trailing sub-tile (24 > 16).
#[test]
pub(super) fn q2_ko_int8_grouped_matches_f32_ref() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 256usize; // N (output features, mult of 32)
    let ncols = 512usize; // K (input features, mult of 128)
    let expert_batches = [1usize, 8, 16, 24];
    let total_batch: usize = expert_batches.iter().sum();
    let mut rng = rand::rng();
    let stream = dev.cuda_stream();

    // Per-expert Q2_KO weights (CPU codec == GPU layout) + their exact f32 dequant.
    let mut weight_ptrs: Vec<u64> = Vec::new();
    let mut _storages = Vec::new(); // keep device buffers alive for the launch
    let mut ref_w: Vec<Vec<f32>> = Vec::new();
    for _ in 0..expert_batches.len() {
        let w: Vec<f32> = (0..nrows * ncols)
            .map(|_| rng.random_range(-0.5f32..0.5))
            .collect();
        let q2ko = crate::quantized::ko_quant::quantize_ko(&w, nrows, ncols, GgmlDType::Q2_KO);
        ref_w.push(crate::quantized::ko_quant::dequant_ko(
            &q2ko,
            nrows,
            ncols,
            GgmlDType::Q2_KO,
        ));
        let slice = dev.memcpy_stod(&q2ko)?;
        let p = {
            let (p, _g) = slice.device_ptr(&stream);
            p // guard drops here; the pointer stays valid while `slice` lives in `_storages`
        };
        weight_ptrs.push(p);
        _storages.push(slice);
    }

    let act_data: Vec<f32> = (0..total_batch * ncols)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    let q8a128 = quantize_acts_q8a128_test(&dev, &act_data, total_batch, ncols)?;
    let mut expert_offsets: Vec<i32> = vec![0];
    for &b in &expert_batches {
        expert_offsets.push(expert_offsets.last().unwrap() + b as i32);
    }

    let int8 = grouped_qmatmul(
        DynamicTensor::Int8(&q8a128),
        &weight_ptrs,
        GgmlDType::Q2_KO,
        nrows,
        &expert_offsets,
        &dev,
        Backing::Owned,
    )?;
    let vi = read_f32_tensor(&dev, &int8)?; // [total_batch, nrows] row-major

    // CPU f32 reference over the same dequantized weights (raw f32 activations).
    let mut vref = vec![0f32; total_batch * nrows];
    for (e, _) in expert_batches.iter().enumerate() {
        let (lo, hi) = (expert_offsets[e] as usize, expert_offsets[e + 1] as usize);
        for t in lo..hi {
            for n in 0..nrows {
                let mut acc = 0f32;
                for k in 0..ncols {
                    acc += ref_w[e][n * ncols + k] * act_data[t * ncols + k];
                }
                vref[t * nrows + n] = acc;
            }
        }
    }
    assert_eq!(vi.len(), vref.len());
    assert!(
        vi.iter().all(|x| x.is_finite()),
        "Q2_KO int8 grouped produced non-finite output (broken unpack/fold)"
    );
    let rel = rel_l2(&vi, &vref);
    println!("Q2_KO int8 grouped vs f32 ref: rel_l2 = {rel:.5}");
    assert!(
        rel < 0.03,
        "Q2_KO int8 grouped diverged (rel_l2 = {rel:.5})"
    );
    Ok(())
}

/// Q3_KO (3-bit affine KO twin) int8 grouped GEMM correctness, the same construction as
/// [`q2_ko_int8_grouped_matches_f32_ref`]: the `q3_ko_int8_f32_grouped` kernel and its
/// crumb+hi unpack (`loader/q3_KO.cuh`) must reproduce a CPU f32 reference matmul over the
/// SAME weights, so the 3-bit weight is identical on both sides and only the int8 activation
/// quant can differ.
///
/// **This is the test that catches the one thing Q3_KO could plausibly get wrong.** Its value
/// is split across two planes at two different shifts, and the crumb plane alone reconstructs a
/// *plausible* weight — every value simply mod 4. A dropped or mis-shifted `hi` bit therefore
/// does not produce NaN or garbage; it produces a smoothly wrong matmul that a tolerance check
/// on the output could easily pass. Eight levels against four moves `rel_l2` from ~1% to ~15%,
/// far outside the bound below, which is why the bound is tight rather than generous.
#[test]
pub(super) fn q3_ko_int8_grouped_matches_f32_ref() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 256usize; // N (output features, mult of 32)
    let ncols = 512usize; // K (input features, mult of 128)
    let expert_batches = [1usize, 8, 16, 24];
    let total_batch: usize = expert_batches.iter().sum();
    let mut rng = rand::rng();
    let stream = dev.cuda_stream();

    // Per-expert Q3_KO weights (CPU codec == GPU layout) + their exact f32 dequant.
    let mut weight_ptrs: Vec<u64> = Vec::new();
    let mut _storages = Vec::new(); // keep device buffers alive for the launch
    let mut ref_w: Vec<Vec<f32>> = Vec::new();
    for _ in 0..expert_batches.len() {
        let w: Vec<f32> = (0..nrows * ncols)
            .map(|_| rng.random_range(-0.5f32..0.5))
            .collect();
        let q3ko = crate::quantized::ko_quant::quantize_ko(&w, nrows, ncols, GgmlDType::Q3_KO);
        ref_w.push(crate::quantized::ko_quant::dequant_ko(
            &q3ko,
            nrows,
            ncols,
            GgmlDType::Q3_KO,
        ));
        let slice = dev.memcpy_stod(&q3ko)?;
        let p = {
            let (p, _g) = slice.device_ptr(&stream);
            p // guard drops here; the pointer stays valid while `slice` lives in `_storages`
        };
        weight_ptrs.push(p);
        _storages.push(slice);
    }

    let act_data: Vec<f32> = (0..total_batch * ncols)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    let q8a128 = quantize_acts_q8a128_test(&dev, &act_data, total_batch, ncols)?;
    let mut expert_offsets: Vec<i32> = vec![0];
    for &b in &expert_batches {
        expert_offsets.push(expert_offsets.last().unwrap() + b as i32);
    }

    let int8 = grouped_qmatmul(
        DynamicTensor::Int8(&q8a128),
        &weight_ptrs,
        GgmlDType::Q3_KO,
        nrows,
        &expert_offsets,
        &dev,
        Backing::Owned,
    )?;
    let vi = read_f32_tensor(&dev, &int8)?; // [total_batch, nrows] row-major

    // CPU f32 reference over the same dequantized weights (raw f32 activations).
    let mut vref = vec![0f32; total_batch * nrows];
    for (e, _) in expert_batches.iter().enumerate() {
        let (lo, hi) = (expert_offsets[e] as usize, expert_offsets[e + 1] as usize);
        for t in lo..hi {
            for n in 0..nrows {
                let mut acc = 0f32;
                for k in 0..ncols {
                    acc += ref_w[e][n * ncols + k] * act_data[t * ncols + k];
                }
                vref[t * nrows + n] = acc;
            }
        }
    }
    assert_eq!(vi.len(), vref.len());
    assert!(
        vi.iter().all(|x| x.is_finite()),
        "Q3_KO int8 grouped produced non-finite output (broken unpack/fold)"
    );
    let rel = rel_l2(&vi, &vref);
    println!("Q3_KO int8 grouped vs f32 ref: rel_l2 = {rel:.5}");
    assert!(
        rel < 0.03,
        "Q3_KO int8 grouped diverged (rel_l2 = {rel:.5}) — a dropped `hi` bit reads as ~0.15"
    );
    Ok(())
}

/// The wide-Bm grouped modes (`n_sub` 4 / 8, Bm 64 / 128) are BIT-IDENTICAL to
/// mode-2: the tile width only regroups which tokens share a block — each output
/// row's K-loop int32 accumulation order is unchanged, and the loader zero-pads
/// rows past `b_cnt`, so a partial wide tile computes on zeros and stores
/// nothing for them. Expert batches straddle every regime: below one sub-tile,
/// mid-tile partials at each width, and multi-tile (200 rows → 2×128-wide
/// tiles). Exact f32 equality, not tolerance — the fold multiplies identical
/// int32 sums by identical scales.
#[test]
pub(super) fn grouped_int8_wide_tiles_match_mode2() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 256usize;
    let ncols = 512usize;
    let expert_batches = [1usize, 40, 100, 200];
    let total_batch: usize = expert_batches.iter().sum();
    let mut rng = rand::rng();
    let stream = dev.cuda_stream();

    let mut weight_ptrs: Vec<u64> = Vec::new();
    let mut _storages = Vec::new();
    for _ in 0..expert_batches.len() {
        let w: Vec<f32> = (0..nrows * ncols)
            .map(|_| rng.random_range(-0.5f32..0.5))
            .collect();
        let ko = crate::quantized::ko_quant::quantize_ko(&w, nrows, ncols, GgmlDType::Q2_KO);
        let slice = dev.memcpy_stod(&ko)?;
        let p = {
            let (p, _g) = slice.device_ptr(&stream);
            p
        };
        weight_ptrs.push(p);
        _storages.push(slice);
    }

    let act_data: Vec<f32> = (0..total_batch * ncols)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    let q8a128 = quantize_acts_q8a128_test(&dev, &act_data, total_batch, ncols)?;
    let mut expert_offsets: Vec<i32> = vec![0];
    for &b in &expert_batches {
        expert_offsets.push(expert_offsets.last().unwrap() + b as i32);
    }

    let run = |n_sub: usize, row_fast: bool| -> Result<Vec<f32>> {
        let out = q8a128.with_device_ptr(&dev, |act_ptr| {
            crate::quantized::cuda::grouped_matmul_gemx_q8a128_with_mode(
                act_ptr,
                &weight_ptrs,
                GgmlDType::Q2_KO,
                nrows,
                ncols,
                total_batch,
                &expert_offsets,
                &dev,
                Backing::Owned,
                n_sub,
                row_fast,
                SumScale::Raw,
            )
        })?;
        read_f32_tensor(&dev, &out)
    };
    // Both grid axis orders must be bit-identical too (schedule order only) —
    // every (mode, order) combination lands on the same outputs.
    let m2 = run(2, true)?;
    for n_sub in [2usize, 4, 8] {
        for row_fast in [true, false] {
            if n_sub == 2 && row_fast {
                continue; // the reference itself
            }
            let wide = run(n_sub, row_fast)?;
            assert_eq!(m2.len(), wide.len());
            let diff = m2.iter().zip(&wide).filter(|(a, b)| a != b).count();
            assert_eq!(
                diff,
                0,
                "n_sub={n_sub} row_fast={row_fast} diverged from mode-2 on {diff} of {} outputs",
                m2.len()
            );
        }
    }
    Ok(())
}

/// §8.2 outlier-robustness floor — inject sparse activation outliers (the failure
/// mode `randn` never produces) and require the INT8 path to degrade gracefully,
/// not catastrophically, vs the FP16 oracle (which has the dynamic range to
/// absorb them). This stresses the per-32-block int8 quant headroom.
#[test]
pub(super) fn grouped_int8_outlier_stress() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 768;
    let ncols = 2048;
    let expert_batches = &[8usize, 16];
    let total_batch: usize = expert_batches.iter().sum();

    let mut rng = rand::rng();
    let mut act_data: Vec<f32> = (0..total_batch * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    // Inject ~0.15% outlier channels at 40× magnitude (fixed indices, all tokens).
    for ch in [37usize, 512, 1009, 1900] {
        for t in 0..total_batch {
            act_data[t * ncols + ch] *= 40.0;
        }
    }

    // Two weight sets from the SAME floats: Q4_K feeds the FP16 oracle, Q4_KO the
    // int8 path. They cannot share one pool — `ensure_qmatmul_pairing` pairs int8
    // q8a128 activations exclusively with KO weights and float activations
    // exclusively with non-KO, so the single-pool form this test used to run
    // returns a pairing error rather than a number. (Its sibling
    // `grouped_int8_matches_legacy` was retired for the same reason; this one was
    // missed, and outlier robustness is not covered anywhere else.)
    use crate::Shape;
    let shape = Shape::from((nrows, ncols));
    let stream = dev.cuda_stream();
    let mut k_ptrs: Vec<u64> = Vec::with_capacity(expert_batches.len());
    let mut ko_ptrs: Vec<u64> = Vec::with_capacity(expert_batches.len());
    let mut k_pool = Vec::with_capacity(expert_batches.len());
    let mut ko_pool = Vec::with_capacity(expert_batches.len());
    for _ in 0..expert_batches.len() {
        let w: Vec<f32> = (0..nrows * ncols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let mut xs = QCudaStorage::zeros(&dev, ncols * nrows, GgmlDType::Q4_K)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(
            dev.memcpy_stod(&w)?,
            dev.clone(),
        ))?;
        let xs = xs.repack_gemx(&shape)?;
        k_ptrs.push(xs.data_ptr());
        // Q4_KO per-128 re-quant of the same weights: maxq 15, no crumb/hi bytes.
        let ko_slice = dev.memcpy_stod(&requant_ko_per128(&w, nrows, ncols, 15, 0, 0))?;
        {
            let (p, _g) = ko_slice.device_ptr(&stream);
            ko_ptrs.push(p); // valid while `ko_slice` lives in the pool below
        }
        k_pool.push(xs);
        ko_pool.push(ko_slice);
    }
    let act_bf16: Vec<bf16> = act_data.iter().map(|&v| bf16::from_f32(v)).collect();
    let act_storage = CudaStorage::wrap_cuda_slice(dev.memcpy_stod(&act_bf16)?, dev.clone());
    let act_layout = crate::Layout::contiguous(Shape::from(vec![total_batch, ncols]));
    let mut expert_offsets: Vec<i32> = vec![0];
    for &b in expert_batches {
        expert_offsets.push(expert_offsets.last().unwrap() + b as i32);
    }

    let q8a128 = quantize_acts_q8a128_test(&dev, &act_data, total_batch, ncols)?;
    let int8 = grouped_qmatmul(
        DynamicTensor::Int8(&q8a128),
        &ko_ptrs,
        GgmlDType::Q4_KO,
        nrows,
        &expert_offsets,
        &dev,
        Backing::Owned,
    )?;
    let legacy = grouped_matmul_gemx(
        &k_ptrs,
        GgmlDType::Q4_K,
        nrows,
        ncols,
        &act_storage,
        &act_layout,
        &expert_offsets,
        &dev,
    )?;

    let vi = read_f32_tensor(&dev, &int8)?;
    let vl = read_bf16_tensor(&dev, &legacy)?;
    let rel = rel_l2(&vi, &vl);
    println!("grouped q8a128 INT8 vs legacy FP16 (outlier-injected): rel_l2 = {rel:.5}");
    assert!(
        rel < 0.10,
        "INT8 grouped degraded catastrophically under outliers (rel_l2 = {rel:.5})"
    );
    Ok(())
}
