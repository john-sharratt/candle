//! The sub-major KO layouts, each checked against an independent reference over
//! the same weights.
//!
//! Q4_KO is a pure byte permutation of Q4_K, so its int8 result must equal the
//! Q4_K result bit for bit, which proves the layout, the loader and the dispatch
//! together. The sub-major k1024 layouts for Q4, Q5, Q6 and Q8 are each checked
//! against a CPU reference or the independent fp16 path. The tolerances sit at
//! the int8 activation floor, so a fifth-bit or nibble bug shows as an order-one
//! divergence. The precision model reports the dequant error each K-to-KO rung
//! buys.

use super::*;

/// Q4_KO is a pure byte permutation of the Q4_K compact block — the 16 qs ints made
/// contiguous, the four (scale,-min) pairs grouped at the tail. Reading a permuted
/// Q4_KO weight through the int8 q8a128 matmul must produce the BIT-IDENTICAL result
/// of reading the original Q4_K weight, proving the KO layout + loader + dispatch are
/// correct end to end. This test permutes on the host; the production path repacks on
/// the GPU (`QCudaStorage::repack_ko`).
#[test]
#[ignore = "retired: per-32 sub-major path superseded by the per-128 collapse"]
pub(super) fn q4_ko_matches_q4_k_int8() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 256; // N
    let ncols = 512; // K (ncols/128 = 4 K-blocks per row)
    let total_batch = 20usize; // M — spans two ≤16-token tiles
    let mut rng = rand::rng();
    let act_data: Vec<f32> = (0..total_batch * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let q8a128 = quantize_acts_q8a128_test(&dev, &act_data, total_batch, ncols)?;

    // Build the Q4_K compact (K/128) weight.
    let (k_ptrs, k_storages, _a, _l, _off) = build_int8_ab_fixture(
        &dev,
        nrows,
        ncols,
        &[total_batch],
        &act_data,
        GgmlDType::Q4_K,
    )?;

    // Permute each 80-byte compact block Q4_K -> Q4_KO on the host. Q4_K interleaves
    // the scales between the qs runs ({qs0-3, dm0, dm1, qs4-7, qs8-11, dm2, dm3,
    // qs12-15}); Q4_KO makes qs contiguous (0-63) and groups the scales at the tail
    // (64-79). Same bytes, reordered.
    let k_bytes = k_storages[0].data()?;
    assert_eq!(
        k_bytes.len() % 80,
        0,
        "compact Q4_K weight must be a whole number of 80-byte blocks"
    );
    let mut ko_bytes = k_bytes.clone();
    for (kb, ob) in k_bytes
        .as_chunks::<80>()
        .0
        .iter()
        .zip(ko_bytes.as_chunks_mut::<80>().0.iter_mut())
    {
        // Each sub's 4 qs ints interleaved [I0,I2,I1,I3] (swap I1/I2). K qs bases per
        // sub are {0,24,40,64}; scales group at the tail (64-79).
        for (s, &kb0) in [0usize, 24, 40, 64].iter().enumerate() {
            ob[s * 16..s * 16 + 4].copy_from_slice(&kb[kb0..kb0 + 4]); // I0
            ob[s * 16 + 4..s * 16 + 8].copy_from_slice(&kb[kb0 + 8..kb0 + 12]); // I2
            ob[s * 16 + 8..s * 16 + 12].copy_from_slice(&kb[kb0 + 4..kb0 + 8]); // I1
            ob[s * 16 + 12..s * 16 + 16].copy_from_slice(&kb[kb0 + 12..kb0 + 16]);
            // I3
        }
        ob[64..68].copy_from_slice(&kb[16..20]); // dm0
        ob[68..72].copy_from_slice(&kb[20..24]); // dm1
        ob[72..76].copy_from_slice(&kb[56..60]); // dm2
        ob[76..80].copy_from_slice(&kb[60..64]); // dm3
    }
    let ko_slice = dev.memcpy_stod(&ko_bytes)?;
    let stream = dev.cuda_stream();
    let (ko_ptr, _ko_guard) = ko_slice.device_ptr(&stream);

    // Dense int8 matmul: original Q4_K vs permuted Q4_KO — must be bit-identical.
    let out_k = dense_qmatmul(
        DynamicTensor::Int8(&q8a128),
        k_ptrs[0],
        GgmlDType::Q4_K,
        nrows,
        0,
        crate::DType::F32,
        &dev,
    )?;
    let out_ko = dense_qmatmul(
        DynamicTensor::Int8(&q8a128),
        ko_ptr,
        GgmlDType::Q4_KO,
        nrows,
        0,
        crate::DType::F32,
        &dev,
    )?;
    let vk = read_f32_tensor(&dev, &out_k)?;
    let vko = read_f32_tensor(&dev, &out_ko)?;
    assert_eq!(vk.len(), vko.len());
    let rel = rel_l2(&vk, &vko);
    println!("Q4_KO vs Q4_K dense INT8: rel_l2 = {rel:.6}");
    assert!(rel < 1e-6, "Q4_KO diverged from Q4_K (rel_l2 = {rel:.6})");
    Ok(())
}

/// Gate for the sub-major ("perfect dequant") Q4_KO k1024 layout: build it from compact
/// Q4_K, run the int8 q8a128 matmul (which now reads the sub-major chunk), and compare to a
/// CPU f32 reference that dequantizes the SAME nibbles + fp16 (scale,min) — `W = scale·q + min`
/// — against the original f32 activations. Validates the encoding ↔ GPU dequant + fold end to
/// end; the residual is only q8a128 activation-quant + accumulation order, so rel_l2 sits at
/// the int8 floor (~1e-2), far below the ~O(1) a layout/byte-order bug would produce.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "retired: per-32 sub-major superseded by the per-128 collapse"]
pub(super) fn q4ko_submajor_dequant_correct() -> Result<()> {
    use half::f16;
    let dev = CudaDevice::new(0)?;
    let nrows = 256;
    let ncols = 512;
    let total_batch = 20usize;
    let mut rng = rand::rng();
    let act_data: Vec<f32> = (0..total_batch * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let q8a128 = quantize_acts_q8a128_test(&dev, &act_data, total_batch, ncols)?;

    let (_k_ptrs, k_storages, _a, _l, _o) = build_int8_ab_fixture(
        &dev,
        nrows,
        ncols,
        &[total_batch],
        &act_data,
        GgmlDType::Q4_K,
    )?;
    let kb = k_storages[0].data()?;

    // Sub-major Q4_KO → GPU int8.
    let ob = q4ko_submajor_from_compact(&kb, nrows, ncols);
    let ko_slice = dev.memcpy_stod(&ob)?;
    let stream = dev.cuda_stream();
    let (ko_ptr, _ko_guard) = ko_slice.device_ptr(&stream);
    let out = dense_qmatmul(
        DynamicTensor::Int8(&q8a128),
        ko_ptr,
        GgmlDType::Q4_KO,
        nrows,
        0,
        crate::DType::F32,
        &dev,
    )?;
    let vgpu = read_f32_tensor(&dev, &out)?;

    // CPU reference: W = f16(scale)·nibble + f16(min) from the SAME compact block.
    const QS_BASE: [usize; 4] = [0, 24, 40, 64];
    const DM_OFF: [usize; 4] = [16, 20, 56, 60];
    let mut vref = vec![0f32; total_batch * nrows];
    for row in 0..nrows {
        for k_blk in 0..(ncols / 128) {
            let blk = &kb[(k_blk * nrows + row) * 80..(k_blk * nrows + row) * 80 + 80];
            for sub in 0..4 {
                let kk = q4k_decode_sub(blk, QS_BASE[sub]);
                let o = DM_OFF[sub];
                let scale = f16::from_le_bytes([blk[o], blk[o + 1]]).to_f32();
                let min = f16::from_le_bytes([blk[o + 2], blk[o + 3]]).to_f32();
                for (ki, &kq) in kk.iter().enumerate().take(32) {
                    let kcol = k_blk * 128 + sub * 32 + ki;
                    let w = scale * (kq as f32) + min;
                    for t in 0..total_batch {
                        vref[t * nrows + row] += w * act_data[t * ncols + kcol];
                    }
                }
            }
        }
    }

    let rel = rel_l2(&vgpu, &vref);
    println!("Q4_KO sub-major vs CPU-f32 ref: rel_l2 = {rel:.6}");
    assert!(
        rel < 0.03,
        "Q4_KO sub-major dequant wrong (rel_l2 = {rel:.6})"
    );
    Ok(())
}

/// Gate for the sub-major Q8_KO k1024 layout (full-byte int8, symmetric: W = scale·q8).
/// De-interleave Q8_KO, reorder sub-major, run the int8 matmul, compare to a CPU f32 ref
/// that pulls the int8 values from the same de-interleaved offsets and the per-sub fp16 scale.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "retired: per-32 sub-major superseded by the per-128 collapse"]
pub(super) fn q8ko_submajor_dequant_correct() -> Result<()> {
    use half::f16;
    let dev = CudaDevice::new(0)?;
    let nrows = 256;
    let ncols = 512;
    let total_batch = 20usize;
    let mut rng = rand::rng();
    let act_data: Vec<f32> = (0..total_batch * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let q8a128 = quantize_acts_q8a128_test(&dev, &act_data, total_batch, ncols)?;

    let (_k_ptrs, k_storages, _a, _l, _o) = build_int8_ab_fixture(
        &dev,
        nrows,
        ncols,
        &[total_batch],
        &act_data,
        GgmlDType::Q8_K,
    )?;
    let kb = k_storages[0].data()?;

    // De-interleave Q8_K → Q8_KO (perm mirrors q5q6q8_ko_match_k_int8), then reorder sub-major.
    let q8_perm: fn(&[u8], &mut [u8]) = |kb, ob| {
        const QS: [usize; 16] = [
            0, 8, 16, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 120, 128, 136,
        ];
        const DM: [usize; 4] = [24, 112, 116, 144];
        for m in 0..16 {
            ob[m * 8..m * 8 + 8].copy_from_slice(&kb[QS[m]..QS[m] + 8]);
        }
        for s in 0..4 {
            ob[128 + s * 4..128 + s * 4 + 4].copy_from_slice(&kb[DM[s]..DM[s] + 4]);
        }
    };
    let de = ko_deinterleave(&kb, 160, 128, true, q8_perm);
    let ob = q8ko_submajor_from_de(&de, nrows, ncols);
    let ko_slice = dev.memcpy_stod(&ob)?;
    let stream = dev.cuda_stream();
    let (ko_ptr, _ko_guard) = ko_slice.device_ptr(&stream);
    let out = dense_qmatmul(
        DynamicTensor::Int8(&q8a128),
        ko_ptr,
        GgmlDType::Q8_KO,
        nrows,
        0,
        crate::DType::F32,
        &dev,
    )?;
    let vgpu = read_f32_tensor(&dev, &out)?;

    // CPU ref: int8 values pulled from the de-interleaved Q8_KO at the dequant offsets.
    let n_blocks = (ncols / 128) * nrows;
    let sbase = n_blocks * 128;
    let mut vref = vec![0f32; total_batch * nrows];
    for row in 0..nrows {
        for k_blk in 0..(ncols / 128) {
            let i = k_blk * nrows + row;
            let q = &de[i * 128..i * 128 + 128];
            for sub in 0..4 {
                let so = sbase + i * 16 + sub * 4;
                let scale = f16::from_le_bytes([de[so], de[so + 1]]).to_f32();
                for q3 in 0..4 {
                    let off0 = (sub * 4 + (q3 >> 1)) * 8 + (q3 & 1) * 4;
                    let off1 = (sub * 4 + 2 + (q3 >> 1)) * 8 + (q3 & 1) * 4;
                    for ii in 0..4 {
                        let v0 = q[off0 + ii] as i8 as f32;
                        let v1 = q[off1 + ii] as i8 as f32;
                        let kc0 = k_blk * 128 + sub * 32 + q3 * 4 + ii;
                        let kc1 = k_blk * 128 + sub * 32 + 16 + q3 * 4 + ii;
                        for t in 0..total_batch {
                            vref[t * nrows + row] += scale * v0 * act_data[t * ncols + kc0];
                            vref[t * nrows + row] += scale * v1 * act_data[t * ncols + kc1];
                        }
                    }
                }
            }
        }
    }

    let rel = rel_l2(&vgpu, &vref);
    println!("Q8_KO sub-major vs CPU-f32 ref: rel_l2 = {rel:.6}");
    assert!(
        rel < 0.03,
        "Q8_KO sub-major dequant wrong (rel_l2 = {rel:.6})"
    );
    Ok(())
}

/// Gate for the PER-128 collapse end-to-end (kernel ↔ producer integration), via Q8_KO.
/// Quantize explicit f32 weights per-128, run the per-128 int8 kernel, compare to a CPU ref
/// that re-derives the SAME per-128 symmetric quantization — validates the accumulate-then-
/// scale fold + the dm[8] layout. The activation is now per-128 too (one scale per 128).
#[cfg(feature = "cuda")]
#[test]
pub(super) fn q8ko_per128_dequant_correct() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 256;
    let ncols = 512;
    let total_batch = 20usize;
    let mut rng = rand::rng();
    let w: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let act_data: Vec<f32> = (0..total_batch * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let q8a128 = quantize_acts_q8a128_test(&dev, &act_data, total_batch, ncols)?;

    let ob = requant_q8_ko_per128(&w, nrows, ncols);
    let ko_slice = dev.memcpy_stod(&ob)?;
    let stream = dev.cuda_stream();
    let (ko_ptr, _ko_guard) = ko_slice.device_ptr(&stream);
    let out = dense_qmatmul(
        DynamicTensor::Int8(&q8a128),
        ko_ptr,
        GgmlDType::Q8_KO,
        nrows,
        0,
        crate::DType::F32,
        &dev,
    )?;
    let vgpu = read_f32_tensor(&dev, &out)?;

    // Independent CPU ref: same per-128 symmetric quantization, W = scale·q.
    let mut vref = vec![0f32; total_batch * nrows];
    for row in 0..nrows {
        for k_blk in 0..(ncols / 128) {
            let wbase = row * ncols + k_blk * 128;
            let mut amax = 0f32;
            for kk in 0..128 {
                amax = amax.max(w[wbase + kk].abs());
            }
            let scale = (amax / 127.0).max(1e-12);
            for kk in 0..128 {
                let q = (w[wbase + kk] / scale).round().clamp(-127.0, 127.0);
                let wq = scale * q;
                let kc = k_blk * 128 + kk;
                for t in 0..total_batch {
                    vref[t * nrows + row] += wq * act_data[t * ncols + kc];
                }
            }
        }
    }

    let rel = rel_l2(&vgpu, &vref);
    println!("Q8_KO per-128 vs CPU-f32 ref: rel_l2 = {rel:.6}");
    assert!(
        rel < 0.03,
        "Q8_KO per-128 (collapse) wrong (rel_l2 = {rel:.6})"
    );
    Ok(())
}

/// Per-128 collapse gate for Q5_KO (4+1 bit) and Q6_KO (4+2 bit): quantize f32 per-128,
/// run the per-128 kernel, compare to an independent per-128 affine CPU ref (W = scale·q+min).
#[cfg(feature = "cuda")]
#[test]
pub(super) fn q5q6_ko_per128_dequant_correct() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 256;
    let ncols = 512;
    let total_batch = 20usize;
    let mut rng = rand::rng();
    let w: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let act_data: Vec<f32> = (0..total_batch * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let q8a128 = quantize_acts_q8a128_test(&dev, &act_data, total_batch, ncols)?;
    for &(kod, maxq, crumb, hi) in &[
        (GgmlDType::Q4_KO, 15i32, 0usize, 0usize), // lane-major ql + int4-combine dequant
        (GgmlDType::Q5_KO, 31, 0, 128),
        (GgmlDType::Q6_KO, 63, 256, 0),
    ] {
        let ob = requant_ko_per128(&w, nrows, ncols, maxq, crumb, hi);
        let ko_slice = dev.memcpy_stod(&ob)?;
        let stream = dev.cuda_stream();
        let (ko_ptr, _g) = ko_slice.device_ptr(&stream);
        let out = dense_qmatmul(
            DynamicTensor::Int8(&q8a128),
            ko_ptr,
            kod,
            nrows,
            0,
            crate::DType::F32,
            &dev,
        )?;
        let vgpu = read_f32_tensor(&dev, &out)?;
        let mut vref = vec![0f32; total_batch * nrows];
        for row in 0..nrows {
            for k_blk in 0..(ncols / 128) {
                let wbase = row * ncols + k_blk * 128;
                let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
                for kk in 0..128 {
                    let v = w[wbase + kk];
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                let scale = ((mx - mn) / maxq as f32).max(1e-12);
                for kk in 0..128 {
                    let q = (((w[wbase + kk] - mn) / scale).round() as i32).clamp(0, maxq) as f32;
                    let wq = scale * q + mn;
                    let kc = k_blk * 128 + kk;
                    for t in 0..total_batch {
                        vref[t * nrows + row] += wq * act_data[t * ncols + kc];
                    }
                }
            }
        }
        // Diagnostic: also vs the ORIGINAL f32 weights (ground truth) — should DECREASE with bits.
        let mut vorig = vec![0f32; total_batch * nrows];
        for row in 0..nrows {
            for kc in 0..ncols {
                let wv = w[row * ncols + kc];
                for t in 0..total_batch {
                    vorig[t * nrows + row] += wv * act_data[t * ncols + kc];
                }
            }
        }
        let rel = rel_l2(&vgpu, &vref);
        let rel_f32 = rel_l2(&vgpu, &vorig);
        println!("{kod:?} per-128: vs re-quant ref = {rel:.6}, vs f32 ground truth = {rel_f32:.6}");
        assert!(rel < 0.03, "{kod:?} per-128 wrong (rel_l2 = {rel:.6})");
    }
    Ok(())
}

/// Gate for the sub-major Q6_KO k1024 layout. Quantize explicit f32 weights via
/// requant_q6_ko_affine, reorder sub-major, run the int8 matmul, and compare to a CPU ref
/// that re-derives the SAME per-32 (scale,min,q) quantization directly from the f32 weights —
/// independent of the decode/repack path, so a packing bug can't cancel against the reference.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "retired: per-32 sub-major superseded by the per-128 collapse"]
pub(super) fn q6ko_submajor_dequant_correct() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 256;
    let ncols = 512;
    let total_batch = 20usize;
    let mut rng = rand::rng();
    let w: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let act_data: Vec<f32> = (0..total_batch * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let q8a128 = quantize_acts_q8a128_test(&dev, &act_data, total_batch, ncols)?;

    let de = requant_q6_ko_affine(&w, nrows, ncols);
    let ob = q6ko_submajor_from_de(&de, nrows, ncols);
    let ko_slice = dev.memcpy_stod(&ob)?;
    let stream = dev.cuda_stream();
    let (ko_ptr, _ko_guard) = ko_slice.device_ptr(&stream);
    let out = dense_qmatmul(
        DynamicTensor::Int8(&q8a128),
        ko_ptr,
        GgmlDType::Q6_KO,
        nrows,
        0,
        crate::DType::F32,
        &dev,
    )?;
    let vgpu = read_f32_tensor(&dev, &out)?;

    // Independent CPU ref: same per-32 affine quantization as requant_q6_ko_affine, from w.
    let mut vref = vec![0f32; total_batch * nrows];
    for row in 0..nrows {
        for k_blk in 0..(ncols / 128) {
            for sub in 0..4 {
                let base = k_blk * 128 + sub * 32;
                let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
                for i in 0..32 {
                    let v = w[row * ncols + base + i];
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                let scale = ((mx - mn) / 63.0).max(1e-12);
                for i in 0..32 {
                    let v = w[row * ncols + base + i];
                    let q = (((v - mn) / scale).round() as i32).clamp(0, 63) as f32;
                    let wq = scale * q + mn;
                    let kc = base + i;
                    for t in 0..total_batch {
                        vref[t * nrows + row] += wq * act_data[t * ncols + kc];
                    }
                }
            }
        }
    }

    let rel = rel_l2(&vgpu, &vref);
    println!("Q6_KO sub-major vs CPU-f32 ref: rel_l2 = {rel:.6}");
    assert!(
        rel < 0.03,
        "Q6_KO sub-major dequant wrong (rel_l2 = {rel:.6})"
    );
    Ok(())
}

/// Gate for the sub-major Q5_KO k1024 layout (4-bit + transposed 5th bit — the trickiest).
/// De-interleave Q5_K → Q5_KO, reorder sub-major, run the int8 matmul, and compare to the
/// INDEPENDENT fp16 Q5_K path (`fwd_via_gemx`, which shares neither the decode nor the
/// repack). Same Q5_K weights both sides, so the residual is just q8a128-vs-bf16 activation
/// precision (~int8 floor); a 5th-bit / nibble bug shows up as ~O(1) divergence.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "retired: per-32 sub-major superseded by the per-128 collapse"]
pub(super) fn q5ko_submajor_dequant_correct() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 256;
    let ncols = 512;
    let total_batch = 20usize;
    let mut rng = rand::rng();
    let act_data: Vec<f32> = (0..total_batch * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let q8a128 = quantize_acts_q8a128_test(&dev, &act_data, total_batch, ncols)?;

    let (_k_ptrs, k_storages, _a, _l, _o) = build_int8_ab_fixture(
        &dev,
        nrows,
        ncols,
        &[total_batch],
        &act_data,
        GgmlDType::Q5_K,
    )?;
    let kb = k_storages[0].data()?;
    let q5_perm: fn(&[u8], &mut [u8]) = |kb, ob| {
        const QS: [usize; 16] = [
            8, 12, 16, 20, 24, 28, 32, 36, 56, 60, 64, 68, 72, 76, 80, 84,
        ];
        const QH: [usize; 4] = [4, 40, 52, 88];
        const DM: [usize; 4] = [0, 44, 48, 92];
        for s in 0..4 {
            ob[s * 16..s * 16 + 4].copy_from_slice(&kb[QS[s * 4]..QS[s * 4] + 4]);
            ob[s * 16 + 4..s * 16 + 8].copy_from_slice(&kb[QS[s * 4 + 2]..QS[s * 4 + 2] + 4]);
            ob[s * 16 + 8..s * 16 + 12].copy_from_slice(&kb[QS[s * 4 + 1]..QS[s * 4 + 1] + 4]);
            ob[s * 16 + 12..s * 16 + 16].copy_from_slice(&kb[QS[s * 4 + 3]..QS[s * 4 + 3] + 4]);
        }
        for i in 0..4 {
            ob[64 + i * 4..64 + i * 4 + 4].copy_from_slice(&kb[QH[i]..QH[i] + 4]);
        }
        for s in 0..4 {
            ob[80 + s * 4..80 + s * 4 + 4].copy_from_slice(&kb[DM[s]..DM[s] + 4]);
        }
    };
    let de = ko_deinterleave(&kb, 112, 80, false, q5_perm);
    let ob = q5ko_submajor_from_de(&de, nrows, ncols);
    let ko_slice = dev.memcpy_stod(&ob)?;
    let stream = dev.cuda_stream();
    let (ko_ptr, _ko_guard) = ko_slice.device_ptr(&stream);
    let out = dense_qmatmul(
        DynamicTensor::Int8(&q8a128),
        ko_ptr,
        GgmlDType::Q5_KO,
        nrows,
        0,
        crate::DType::F32,
        &dev,
    )?;
    let vgpu = read_f32_tensor(&dev, &out)?;

    // Independent fp16 Q5_K reference (same weights, no shared decode/repack).
    let act_bf16: Vec<half::bf16> = act_data.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let act_storage = CudaStorage::wrap_cuda_slice(dev.memcpy_stod(&act_bf16)?, dev.clone());
    let act_layout = crate::Layout::contiguous(crate::Shape::from(vec![total_batch, ncols]));
    let wshape = crate::Shape::from((nrows, ncols));
    let (fp_storage, fp_shape) = k_storages[0].fwd_via_gemx(&wshape, &act_storage, &act_layout)?;
    let fp_t = crate::tensor::from_storage(
        crate::Storage::Cuda(fp_storage),
        fp_shape,
        crate::op::BackpropOp::none(),
        false,
    )
    .to_dtype(crate::DType::F32)?;
    let vref = read_f32_tensor(&dev, &fp_t)?;

    let rel = rel_l2(&vgpu, &vref);
    println!("Q5_KO sub-major vs FP16 Q5_K ref: rel_l2 = {rel:.6}");
    assert!(
        rel < 0.05,
        "Q5_KO sub-major dequant wrong (rel_l2 = {rel:.6})"
    );
    Ok(())
}

/// Precision model for the `_K` → `_KO` perf/precision toggle. Each per-32 `_K` source maps
/// to one of two per-128 `_KO` targets: a "perf" rung (same bit width) or a "precision" rung
/// (wider). Quantizes a Gaussian weight set with a per-group affine `(scale,min)` and reports
/// dequant rel_l2 vs f32 so we can SEE the precision each toggle position buys.
///   Q4_K → Q4_KO (perf) | Q5_KO (precision)
///   Q5_K → Q5_KO (perf) | Q6_KO (precision)
///   Q6_K → Q6_KO (perf) | Q8_KO (precision)
///   Q8_K → Q8_KO        (8-bit s8-MMA cap)
#[cfg(feature = "cuda")]
#[test]
pub(super) fn per128_ko_toggle_precision_model() -> Result<()> {
    let mut rng = rand::rng();
    let n = 8192 * 32; // many 32/128 groups
                       // Approx N(0,1) (sum of 12 uniforms) — realistic transformer-weight tails.
    let w: Vec<f32> = (0..n)
        .map(|_| (0..12).map(|_| rng.random_range(-0.5f32..0.5)).sum::<f32>())
        .collect();
    // Per-group affine quant→dequant: scale = (max-min)/maxq, w' = scale·round((w-min)/scale)+min.
    let qd = |group: usize, maxq: i32| -> Vec<f32> {
        let mut out = vec![0f32; n];
        for g in (0..n).step_by(group) {
            let s = &w[g..g + group];
            let mn = s.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let scale = ((mx - mn) / maxq as f32).max(1e-12);
            for i in 0..group {
                let q = (((w[g + i] - mn) / scale).round() as i32).clamp(0, maxq);
                out[g + i] = scale * q as f32 + mn;
            }
        }
        out
    };
    // The perf/precision toggle: each _K (per-32, N-bit) maps to a per-128 _KO at either the
    // same bits (perf) or wider (precision). Show the dequant rel_l2 vs f32 each rung buys, and
    // how each compares to the per-32 _K bar it replaces.
    println!("=== _K (per-32) → _KO (per-128) perf/precision toggle: dequant rel_l2 vs f32 ===");
    println!("  source   K(per-32)   perf KO      precision KO    perf/K   prec/K");
    for &(src, kbits, perf_q, prec_q, perf_l, prec_l) in &[
        ("Q4_K", 4usize, 15i32, 31i32, "Q4_KO", "Q5_KO"),
        ("Q5_K", 5, 31, 63, "Q5_KO", "Q6_KO"),
        ("Q6_K", 6, 63, 255, "Q6_KO", "Q8_KO"),
    ] {
        let bar = rel_l2(&qd(32, (1 << kbits) - 1), &w); // K: per-32, N-bit
        let perf = rel_l2(&qd(128, perf_q), &w); // perf KO: per-128, same bits
        let prec = rel_l2(&qd(128, prec_q), &w); // precision KO: per-128, wider
        println!(
            "  {src}    {bar:.6}    {perf_l} {perf:.6}   {prec_l} {prec:.6}   {:.3}    {:.3}",
            perf / bar,
            prec / bar
        );
    }
    // Q8_K → Q8_KO is the cap (s8 MMA = 8-bit max): per-128, no toggle.
    let q8_32 = rel_l2(&qd(32, 255), &w);
    let q8_128 = rel_l2(&qd(128, 255), &w);
    println!(
        "  Q8_K    {q8_32:.6}    Q8_KO {q8_128:.6}   (cap)            {:.3}",
        q8_128 / q8_32
    );
    Ok(())
}

/// Precision + storage of Q4_K → Q4_KO (same bits) vs Q4_K → Q5_KO (+1 bit), each with an
/// F32 or FP16 `(scale, min)`. Quantizes a Gaussian weight set with a per-group affine
/// (rounding the stored scale/min to the chosen format) and reports dequant rel_l2 vs f32.
#[cfg(feature = "cuda")]
#[test]
pub(super) fn q4_ko_scale_precision() -> Result<()> {
    use half::f16;
    let mut rng = rand::rng();
    let n = 8192 * 32;
    let w: Vec<f32> = (0..n)
        .map(|_| (0..12).map(|_| rng.random_range(-0.5f32..0.5)).sum::<f32>())
        .collect();
    // Affine quant→dequant; optionally store scale+min as fp16 (used for both quant and dequant).
    let qd = |group: usize, maxq: i32, fp16: bool| -> Vec<f32> {
        let mut out = vec![0f32; n];
        for g in (0..n).step_by(group) {
            let s = &w[g..g + group];
            let mn0 = s.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut scale = ((mx - mn0) / maxq as f32).max(1e-12);
            let mut mn = mn0;
            if fp16 {
                scale = f16::from_f32(scale).to_f32();
                mn = f16::from_f32(mn).to_f32();
            }
            for i in 0..group {
                let q = (((w[g + i] - mn) / scale).round() as i32).clamp(0, maxq);
                out[g + i] = scale * q as f32 + mn;
            }
        }
        out
    };
    println!("=== Q4_K → Q4_KO / Q5_KO : precision (rel_l2 vs f32) × scale format ===");
    println!(
        "  Q4_K  (per-32,  4-bit, ref):  {:.6}",
        rel_l2(&qd(32, 15, false), &w)
    );
    println!(
        "  Q4_KO (per-128, 4-bit, F32):  {:.6}",
        rel_l2(&qd(128, 15, false), &w)
    );
    println!(
        "  Q4_KO (per-128, 4-bit, FP16): {:.6}",
        rel_l2(&qd(128, 15, true), &w)
    );
    println!(
        "  Q5_KO (per-128, 5-bit, F32):  {:.6}",
        rel_l2(&qd(128, 31, false), &w)
    );
    println!(
        "  Q5_KO (per-128, 5-bit, FP16): {:.6}",
        rel_l2(&qd(128, 31, true), &w)
    );
    Ok(())
}

/// Q5_KO / Q6_KO / Q8_KO are byte permutations of the Q5_K / Q6_K / Q8_K compact
/// blocks. Each must give the BIT-IDENTICAL int8 q8a128 result of its K twin. The
/// host permutation below mirrors each format's compact layout → KO regularized
/// layout (Q6_KO is the identity — Q6_K's compact block is already ordered).
#[test]
#[ignore = "retired: per-32 sub-major path superseded by the per-128 collapse"]
pub(super) fn q5q6q8_ko_match_k_int8() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 256;
    let ncols = 512;
    let total_batch = 20usize;
    let mut rng = rand::rng();
    let act_data: Vec<f32> = (0..total_batch * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let q8a128 = quantize_acts_q8a128_test(&dev, &act_data, total_batch, ncols)?;

    // (K dtype, KO dtype, compact stride, per-block K→KO byte permutation).
    type Perm = fn(&[u8], &mut [u8]);
    let cases: &[(GgmlDType, GgmlDType, usize, Perm)] = &[
        (GgmlDType::Q5_K, GgmlDType::Q5_KO, 112, |kb, ob| {
            const QS: [usize; 16] = [
                8, 12, 16, 20, 24, 28, 32, 36, 56, 60, 64, 68, 72, 76, 80, 84,
            ];
            const QH: [usize; 4] = [4, 40, 52, 88];
            const DM: [usize; 4] = [0, 44, 48, 92];
            for s in 0..4 {
                // qs interleaved [I0,I2,I1,I3] per sub (swap I1/I2) for the int2 load.
                ob[s * 16..s * 16 + 4].copy_from_slice(&kb[QS[s * 4]..QS[s * 4] + 4]);
                ob[s * 16 + 4..s * 16 + 8].copy_from_slice(&kb[QS[s * 4 + 2]..QS[s * 4 + 2] + 4]);
                ob[s * 16 + 8..s * 16 + 12].copy_from_slice(&kb[QS[s * 4 + 1]..QS[s * 4 + 1] + 4]);
                ob[s * 16 + 12..s * 16 + 16].copy_from_slice(&kb[QS[s * 4 + 3]..QS[s * 4 + 3] + 4]);
            }
            for i in 0..4 {
                ob[64 + i * 4..64 + i * 4 + 4].copy_from_slice(&kb[QH[i]..QH[i] + 4]);
            }
            for s in 0..4 {
                ob[80 + s * 4..80 + s * 4 + 4].copy_from_slice(&kb[DM[s]..DM[s] + 4]);
            }
        }),
        (GgmlDType::Q6_K, GgmlDType::Q6_KO, 112, |kb, ob| {
            ob.copy_from_slice(kb)
        }),
        (GgmlDType::Q8_K, GgmlDType::Q8_KO, 160, |kb, ob| {
            const QS: [usize; 16] = [
                0, 8, 16, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 120, 128, 136,
            ];
            const DM: [usize; 4] = [24, 112, 116, 144];
            for m in 0..16 {
                ob[m * 8..m * 8 + 8].copy_from_slice(&kb[QS[m]..QS[m] + 8]);
            }
            for s in 0..4 {
                ob[128 + s * 4..128 + s * 4 + 4].copy_from_slice(&kb[DM[s]..DM[s] + 4]);
            }
        }),
    ];

    for &(kd, kod, stride, perm) in cases {
        let (k_ptrs, k_storages, _a, _l, _o) =
            build_int8_ab_fixture(&dev, nrows, ncols, &[total_batch], &act_data, kd)?;
        let k_bytes = k_storages[0].data()?;
        assert_eq!(
            k_bytes.len() % stride,
            0,
            "{kd:?} compact not block-aligned"
        );
        let ko_bytes = ko_deinterleave(
            &k_bytes,
            stride,
            ko_quant_bytes(kod),
            kod == GgmlDType::Q8_KO,
            perm,
        );
        let ko_slice = dev.memcpy_stod(&ko_bytes)?;
        let stream = dev.cuda_stream();
        let (ko_ptr, _ko_guard) = ko_slice.device_ptr(&stream);

        let out_k = dense_qmatmul(
            DynamicTensor::Int8(&q8a128),
            k_ptrs[0],
            kd,
            nrows,
            0,
            crate::DType::F32,
            &dev,
        )?;
        let out_ko = dense_qmatmul(
            DynamicTensor::Int8(&q8a128),
            ko_ptr,
            kod,
            nrows,
            0,
            crate::DType::F32,
            &dev,
        )?;
        let vk = read_f32_tensor(&dev, &out_k)?;
        let vko = read_f32_tensor(&dev, &out_ko)?;
        assert_eq!(vk.len(), vko.len());
        let rel = rel_l2(&vk, &vko);
        println!("{kod:?} vs {kd:?} dense INT8: rel_l2 = {rel:.6}");
        assert!(
            rel < 1e-6,
            "{kod:?} diverged from {kd:?} (rel_l2 = {rel:.6})"
        );
    }
    Ok(())
}
