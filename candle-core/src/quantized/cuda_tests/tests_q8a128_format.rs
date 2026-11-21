//! Byte-level tests of the q8a128 activation format and its fused producers.
//!
//! q8a128 is quantized per 128-element tile, with one scale and sum at `ds[0]`.
//! The quantizer is checked against a CPU reference for every byte the format
//! defines, under both sum conventions, because the sum field is the one a
//! matmul reinterprets. A raw sum overflows f16 on large activations, so a test
//! pins the normalised convention holding a matmul past that range. The fused
//! RMSNorm and SwiGLU producers are checked against the unfused oracle. Pad bytes
//! are not compared, since nothing writes them.

use super::*;

/// Build Q4_K experts + stacked bf16 activations for the INT8 A/B tests.
/// Returns (weight_ptrs, kept-alive storages, act_storage, act_layout, expert_offsets).
#[cfg(test)]
pub(super) fn build_int8_ab_fixture(
    dev: &CudaDevice,
    nrows: usize,
    ncols: usize,
    expert_batches: &[usize],
    act_data: &[f32],
    weight_dtype: GgmlDType,
) -> Result<(
    Vec<u64>,
    Vec<QCudaStorage>,
    CudaStorage,
    crate::Layout,
    Vec<i32>,
)> {
    use crate::Shape;
    let shape = Shape::from((nrows, ncols));
    let mut rng = rand::rng();
    let mut weight_ptrs = Vec::new();
    let mut expert_storages = Vec::new();
    for _ in 0..expert_batches.len() {
        let weights: Vec<f32> = (0..nrows * ncols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let mut xs = QCudaStorage::zeros(dev, ncols * nrows, weight_dtype)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(
            dev.memcpy_stod(&weights)?,
            dev.clone(),
        ))?;
        let xs = xs.repack_gemx(&shape)?;
        weight_ptrs.push(xs.data_ptr());
        expert_storages.push(xs);
    }
    let total_batch: usize = expert_batches.iter().sum();
    let act_bf16: Vec<bf16> = act_data.iter().map(|&v| bf16::from_f32(v)).collect();
    let act_storage = CudaStorage::wrap_cuda_slice(dev.memcpy_stod(&act_bf16)?, dev.clone());
    let act_layout = crate::Layout::contiguous(Shape::from(vec![total_batch, ncols]));
    let mut expert_offsets: Vec<i32> = vec![0];
    for &b in expert_batches {
        expert_offsets.push(expert_offsets.last().unwrap() + b as i32);
    }
    Ok((
        weight_ptrs,
        expert_storages,
        act_storage,
        act_layout,
        expert_offsets,
    ))
}

// block_q8a128 — the contiguous q8 activation block. Tests assert RAW BYTES
// (per the repo codec convention), not error thresholds.

/// Round-half-to-even, matching CUDA __float2int_rn (the quantizer's rounding).
#[cfg(test)]
pub(super) fn round_ties_even_i32(x: f32) -> i32 {
    let r = x.floor();
    let frac = x - r;
    if (frac - 0.5).abs() < 1e-4 {
        let lo = r as i32;
        if lo % 2 == 0 {
            lo
        } else {
            lo + 1
        }
    } else {
        x.round() as i32
    }
}

/// Quantize f32 [rows, cols] → q8a1024 blocks and assert EVERY byte of the output
/// (each tile's `ds[4]` f16 scale+sum and all 128 qs) against a CPU reference, for a
/// deterministic input. This pins the exact flat-grouped byte layout + values.
///
/// **Run for BOTH sum conventions**, because the opt-in is a change to the stored
/// bytes and nothing else would catch it going the wrong way: the qs run and the
/// scale field are identical either way, so a producer that ignored the flag
/// would differ from the oracle in exactly two bytes per 128-tile — and those two
/// bytes are the ones a matmul silently reinterprets. [`SumScale::Raw`] must also
/// reproduce the pre-flag bytes exactly, which is what makes the default safe for
/// every model whose thresholds were derived before this existed.
#[test]
pub(super) fn q8a128_quantize_raw_bytes() -> Result<()> {
    for sum_scale in [SumScale::Raw, SumScale::ByAmax] {
        q8a128_quantize_raw_bytes_for(sum_scale)?;
    }
    Ok(())
}

pub(super) fn q8a128_quantize_raw_bytes_for(sum_scale: SumScale) -> Result<()> {
    use half::f16;
    let dev = CudaDevice::new(0)?;
    let rows = 3usize;
    let cols = 256usize; // 2 blocks per row, 4 sub-blocks each
                         // Deterministic, mixed-sign input (exercises negative sum + asymmetry).
    let act: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 11) as f32 - 5.0) * 1.5)
        .collect();

    let f32_dev = dev.memcpy_stod(&act)?;
    let stream = dev.cuda_stream();
    let (ptr, _g) = f32_dev.device_ptr(&stream);
    let op = quantize_acts_q8a128(
        ptr,
        2, /* F32 */
        rows,
        cols,
        &dev,
        Backing::Owned,
        sum_scale,
    )?;
    assert_eq!(
        op.sum_scale, sum_scale,
        "the operand must report the convention its bytes were written in"
    );
    let blocks = op.into_owned_data()?;
    dev.synchronize()?;
    let raw: Vec<u8> = dev.memcpy_dtov(&blocks.slice(..))?;

    let kblocks = cols / 128;
    let total_tiles = rows * kblocks;
    assert_eq!(
        raw.len(),
        total_tiles.div_ceil(8) * 1152,
        "block buffer size"
    );
    // q8a1024 flat-grouped byte offsets (blocks.cuh): 8×128-tiles per 1152B block.
    let qs_off = |flat: usize| (flat / 8) * 1152 + (flat % 8) * 128;
    let ds_off = |flat: usize| (flat / 8) * 1152 + 1024 + (flat % 8) * 16;

    // q8a128 is per-128: ONE {scale, sum} per 128-tile at ds[0] (ds[1..3] are unwritten pad),
    // and the whole 128-element qs run is quantized with that single per-128 scale.
    let mut checked = 0usize;
    for r in 0..rows {
        for t in 0..kblocks {
            let flat = r * kblocks + t;
            let base = r * cols + t * 128;
            let vals = &act[base..base + 128];
            let amax = vals.iter().fold(0f32, |m, &x| m.max(x.abs()));
            let sum: f32 = vals.iter().sum();
            let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
            // The single {scale, sum} lives at ds[0] of the tile's meta slot.
            // `Raw` stores Σx as it is; `ByAmax` stores Σx/amax, which the matmul
            // rebuilds as `ds.y · ds.x · 127` — needed because a raw Σx overflows
            // f16 on any activation whose block sums pass 65504 (blocks.cuh).
            let ds_b = ds_off(flat);
            let exp_scale = f16::from_f32(amax / 127.0);
            let exp_sum = match sum_scale {
                SumScale::Raw => f16::from_f32(sum),
                SumScale::ByAmax => f16::from_f32(sum * id / 127.0),
            };
            let got_scale = f16::from_le_bytes([raw[ds_b], raw[ds_b + 1]]);
            let got_sum = f16::from_le_bytes([raw[ds_b + 2], raw[ds_b + 3]]);
            assert_eq!(
                got_scale.to_bits(),
                exp_scale.to_bits(),
                "blk({r},{t}) scale"
            );
            assert_eq!(
                got_sum.to_bits(),
                exp_sum.to_bits(),
                "blk({r},{t}) sum under {sum_scale:?}"
            );
            // qs: the tile's 128 int8, all quantized with the per-128 id.
            let qs_b = qs_off(flat);
            for (i, &v) in vals.iter().enumerate() {
                let exp_q = round_ties_even_i32(v * id) as i8;
                let got_q = raw[qs_b + i] as i8;
                assert_eq!(got_q, exp_q, "blk({r},{t}) qs[{i}] (v={v})");
            }
            checked += 1;
        }
    }
    println!(
        "q8a128 raw-byte quantize ({sum_scale:?}): {checked} per-128 tiles verified byte-exact"
    );
    Ok(())
}

/// Dequantize must return EXACTLY `scale * qs` (the quantized representation),
/// matched bit-for-bit against a CPU recompute from the same raw block bytes.
/// q8a128 is per-128: ONE {scale, sum} per 128-tile, stored at ds[0]; all four
/// 32-element sub-blocks of the tile share it (ds[1..3] are alignment pad).
#[test]
pub(super) fn q8a128_dequant_exact() -> Result<()> {
    use half::f16;
    let dev = CudaDevice::new(0)?;
    let rows = 2usize;
    let cols = 384usize; // 3 blocks/row
    let mut rng = rand::rng();
    let act: Vec<f32> = (0..rows * cols)
        .map(|_| rng.random_range(-3.0..3.0))
        .collect();

    let f32_dev = dev.memcpy_stod(&act)?;
    let stream = dev.cuda_stream();
    let (ptr, _g) = f32_dev.device_ptr(&stream);
    let blocks = quantize_acts_q8a128(ptr, 2, rows, cols, &dev, Backing::Owned, SumScale::Raw)?
        .into_owned_data()?;
    let deq = dequantize_q8a128(&blocks, rows, cols, &dev)?;
    dev.synchronize()?;

    let raw: Vec<u8> = dev.memcpy_dtov(&blocks.slice(..))?;
    let deq_v: Vec<f32> = dev.memcpy_dtov(&deq.slice(..))?;

    let kblocks = cols / 128;
    let qs_off = |flat: usize| (flat / 8) * 1152 + (flat % 8) * 128;
    let ds_off = |flat: usize| (flat / 8) * 1152 + 1024 + (flat % 8) * 16;
    for r in 0..rows {
        for t in 0..kblocks {
            let flat = r * kblocks + t;
            // per-128: one scale per 128-tile at ds[0], shared by all four subs.
            let ds_b = ds_off(flat);
            let scale = f16::from_le_bytes([raw[ds_b], raw[ds_b + 1]]).to_f32();
            for sub in 0..4 {
                let qs_b = qs_off(flat) + sub * 32;
                for i in 0..32 {
                    let q = raw[qs_b + i] as i8;
                    let exp = scale * q as f32; // CPU recompute of the dequant
                    let idx = r * cols + t * 128 + sub * 32 + i;
                    assert_eq!(
                        deq_v[idx].to_bits(),
                        exp.to_bits(),
                        "dequant blk({r},{t}) sub {sub} [{i}]"
                    );
                }
            }
        }
    }
    // Sanity: the dequantized values track the original within 8-bit budget.
    let rel = rel_l2(&deq_v, &act);
    println!("q8a128 dequant exact + roundtrip rel_l2 = {rel:.5}");
    assert!(rel < 0.01, "q8a128 roundtrip beyond 8-bit budget: {rel}");
    Ok(())
}

/// Edge cases, at the granularity the format actually has: an all-zero TILE
/// (amax=0 → scale 0, sum 0, qs 0) and a mixed tile carrying a spike, a constant
/// run and a mixed-sign ramp that all share the spike's scale.
///
/// q8a128 is per-**128**: one `{scale, sum}` per 128-element tile, at `ds[0]`
/// (see `quantize/quantize_q8a128.cuh`). An earlier revision quantized per
/// 32-element sub-block, and this test still read four scales out of the tile's
/// meta slot — so it asserted the tile scale (100/127, from the spike) against
/// the zeroed sub-block's expected 0, and read `ds[1]` for a "sub 1 scale" that
/// is 16-byte-alignment pad no producer writes.
#[test]
pub(super) fn q8a128_edge_cases() -> Result<()> {
    use half::f16;
    let dev = CudaDevice::new(0)?;
    let rows = 1usize;
    let cols = 256usize; // two 128-element tiles
    let mut act = vec![0.0f32; cols];
    // Tile 0 stays all-zero. Tile 1: spike, then a constant run, then a ramp —
    // one scale covers all three, which is the property under test.
    act[128] = 100.0; // spike
    for i in 0..32 {
        act[160 + i] = 2.0; // constant run
        act[224 + i] = i as f32 - 16.0; // mixed-sign ramp
    }
    let f32_dev = dev.memcpy_stod(&act)?;
    let stream = dev.cuda_stream();
    let (ptr, _g) = f32_dev.device_ptr(&stream);
    let blocks = quantize_acts_q8a128(ptr, 2, rows, cols, &dev, Backing::Owned, SumScale::Raw)?
        .into_owned_data()?;
    dev.synchronize()?;
    let raw: Vec<u8> = dev.memcpy_dtov(&blocks.slice(..))?;

    // Both tiles sit in super-block 0: qs at 0 and 128, meta at 1024 and 1040.
    let scale_at = |off: usize| f16::from_le_bytes([raw[off], raw[off + 1]]);
    let sum_at = |off: usize| f16::from_le_bytes([raw[off + 2], raw[off + 3]]);

    // Tile 0 — all zero: amax 0 takes the `id = 0` branch, so every quant is 0
    // and the scale is exactly 0 rather than an inf/NaN from dividing by amax.
    assert_eq!(scale_at(1024).to_bits(), 0, "zero tile scale");
    assert_eq!(sum_at(1024).to_bits(), 0, "zero tile sum");
    for (i, &q) in raw.iter().enumerate().take(128) {
        assert_eq!(q, 0, "zero tile qs[{i}]");
    }

    // Tile 1 — amax is the spike, so scale = 100/127 and Σx = 100 + 32×2 + Σ(i−16)
    // = 148, stored raw because that is the mode this producer was asked for.
    assert_eq!(
        scale_at(1040).to_bits(),
        f16::from_f32(100.0 / 127.0).to_bits(),
        "mixed tile scale is the spike's",
    );
    assert_eq!(
        sum_at(1040).to_bits(),
        f16::from_f32(148.0).to_bits(),
        "mixed tile sum, raw",
    );

    // The same tile under [`SumScale::ByAmax`] — the opt-in that divides the sum
    // by the tile's amax so it cannot leave f16's range (see
    // `int8_matmul_survives_block_sums_past_f16_range`). Asserted here beside the
    // raw form because the two modes differ in exactly this one field: the scale
    // and every quant are untouched, which is what makes the opt-in free.
    let normed = quantize_acts_q8a128(ptr, 2, rows, cols, &dev, Backing::Owned, SumScale::ByAmax)?
        .into_owned_data()?;
    dev.synchronize()?;
    let n_raw: Vec<u8> = dev.memcpy_dtov(&normed.slice(..))?;
    assert_eq!(
        f16::from_le_bytes([n_raw[1042], n_raw[1043]]).to_bits(),
        f16::from_f32(148.0 / 100.0).to_bits(),
        "mixed tile sum, normalised by amax",
    );
    assert_eq!(
        f16::from_le_bytes([n_raw[1040], n_raw[1041]]).to_bits(),
        f16::from_f32(100.0 / 127.0).to_bits(),
        "the scale is the same in both modes",
    );
    assert_eq!(
        &n_raw[128..256],
        &raw[128..256],
        "and so are the quants — only `ds[0].y` moves"
    );
    assert_eq!(
        f16::from_le_bytes([n_raw[1024], n_raw[1025]]).to_bits(),
        0,
        "zero tile scale, normalised mode"
    );
    assert_eq!(
        f16::from_le_bytes([n_raw[1026], n_raw[1027]]).to_bits(),
        0,
        "zero tile sum stays 0 rather than dividing 0 by amax 0"
    );

    // The spike saturates to 127; everything else is scaled by the SAME id, so
    // the constant run lands on 3 and the ramp spans −20..19.
    let qs = &raw[128..256];
    assert_eq!(qs[0] as i8, 127, "spike quantizes to full scale");
    for (i, &q) in qs.iter().enumerate().take(32).skip(1) {
        assert_eq!(q as i8, 0, "post-spike zeros qs[{i}]");
    }
    for (i, &q) in qs.iter().enumerate().take(64).skip(32) {
        assert_eq!(q as i8, 3, "constant run qs[{i}]");
    }
    for i in 64..96 {
        assert_eq!(qs[i] as i8, 0, "interior zeros qs[{i}]");
    }
    const RAMP: [i8; 32] = [
        -20, -19, -18, -17, -15, -14, -13, -11, -10, -9, -8, -6, -5, -4, -3, -1, 0, 1, 3, 4, 5, 6,
        8, 9, 10, 11, 13, 14, 15, 17, 18, 19,
    ];
    for i in 0..32 {
        assert_eq!(qs[96 + i] as i8, RAMP[i], "ramp qs[{i}]");
    }
    println!("q8a128 edge cases (zero tile / spike / constant / ramp) verified");
    Ok(())
}

/// **An activation whose per-128 block sums exceed f16's range must still
/// matmul.** The block's `ds[0].y` carries the sum feeding the affine weight's
/// min term, and both `ds` fields are f16 — so storing Σx raw put `+inf` in
/// every block whose 128 values sum past 65504, and each dot product touching
/// one became NaN.
///
/// That is not a synthetic bound. An LLM's activations keep block sums under
/// ~10³ and never came near it, which is why the format survived this long; but
/// Z-Image's transformer is sandwich-normed, so its SwiGLU intermediate runs at
/// 10⁴–10⁵ and its block sums reach 2×10⁵. The first symptom was a black image
/// eight steps later, with every weight, shape and dtype checking out.
///
/// [`SumScale::ByAmax`] normalises the field (Σx/amax, bounded by 128 whatever
/// the magnitude — see `blocks.cuh`); this pins the property rather than the
/// encoding, so it holds however the sum is later stored. The activation here is
/// deliberately biased far from zero: a zero-mean one cancels to a small sum
/// however large its elements are, and would pass even with the bug.
///
/// The mode is what is under test, so it is named explicitly. [`SumScale::Raw`]
/// — what the LLM paths ask for, where block sums stay under ~10³ — genuinely
/// does overflow on this fixture; that is the whole reason the other mode
/// exists, not a defect in this one.
#[test]
pub(super) fn int8_matmul_survives_block_sums_past_f16_range() -> Result<()> {
    use crate::quantized::{QMatMul, QTensor};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (n, k, m) = (128usize, 256usize, 16usize);
    let mut rng = StdRng::seed_from_u64(0x2f0e_b10c);

    // Q8_0 → Q8_KO: an *affine* twin, so the min term — and therefore the block
    // sum — actually takes part in the dot product. A centred format would not
    // read the field at all.
    let w: Vec<f32> = (0..n * k).map(|_| rng.random_range(-1.0..1.0)).collect();
    let mut q = QCudaStorage::zeros(&dev, n * k, GgmlDType::Q8_0)?;
    q.quantize(&CudaStorage::wrap_cuda_slice(
        dev.memcpy_stod(&w)?,
        dev.clone(),
    ))?;
    let shape = crate::Shape::from((n, k));
    let w_ref = QTensor::new(
        QStorage::Cuda(q.repack_ko(&shape, GgmlDType::Q8_KO)?),
        shape.clone(),
    )?;
    let src = QTensor::new(QStorage::Cuda(q), shape)?;
    let mm = QMatMul::from_qtensor(w_ref)?;

    // Each 128-element block sums to ~128 × 1000 = 1.3×10⁵, twice f16's 65504,
    // while amax/127 (~8.7) stays comfortably inside it — so the scale field is
    // fine and only the sum overflows, which is exactly the production shape of
    // the bug.
    let act: Vec<f32> = (0..m * k)
        .map(|_| 1000.0 + rng.random_range(-100.0..100.0))
        .collect();
    let xs = crate::Tensor::from_vec(act, (m, k), &device)?;
    let block_sum = xs.narrow(1, 0, 128)?.sum(1)?.min(0)?.to_scalar::<f32>()?;
    assert!(
        block_sum > 65504.0,
        "fixture does not reach the f16 ceiling it exists to cross: {block_sum}"
    );

    let got = mm
        .forward_via_int8(
            &xs,
            Int8Mode::Precision,
            crate::DType::F32,
            SumScale::ByAmax,
        )?
        .to_vec2::<f32>()?;
    // The FP reference: the same weights dequantised, same activations, f32.
    let want = xs
        .matmul(&src.dequantize(&device)?.t()?.contiguous()?)?
        .to_vec2::<f32>()?;

    let mut num = 0f64;
    let mut den = 0f64;
    for (g_row, w_row) in got.iter().zip(want.iter()) {
        for (&g, &w) in g_row.iter().zip(w_row.iter()) {
            assert!(g.is_finite(), "int8 output is {g}, not a number");
            num += ((g - w) as f64).powi(2);
            den += (w as f64).powi(2);
        }
    }
    let rel = (num / den).sqrt();
    assert!(
        rel < 0.01,
        "int8 vs f32 rel_l2 = {rel} beyond the 8-bit budget"
    );
    println!("int8 with block sums {block_sum:.0} (f16 max 65504): rel_l2 = {rel:.5}");
    Ok(())
}

/// Citizenship: q8a128 must be reachable through the *unified* QType dispatch
/// (`run_quantize_block`/`run_dequantize_block` at qtype = QTYPE_Q8A128 = 36),
/// producing byte-identical results to its dedicated typed path. This pins the
/// enum value + the dispatch wiring, not just the standalone kernels.
#[test]
pub(super) fn q8a128_unified_dispatch_matches_typed() -> Result<()> {
    use candle_kernels::simple::quantized::{run_dequantize_block, run_quantize_block};
    const QTYPE_Q8A128: i32 = 36;
    let dev = CudaDevice::new(0)?;
    let rows = 2usize;
    let cols = 256usize;
    let n = rows * cols;
    let act: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect();
    let f32_dev = dev.memcpy_stod(&act)?;
    let stream = dev.cuda_stream();
    let (ptr, _g) = f32_dev.device_ptr(&stream);

    // Typed path (dtype 2 = F32) vs unified run_quantize_block(qtype=36).
    let typed = quantize_acts_q8a128(ptr, 2, rows, cols, &dev, Backing::Owned, SumScale::Raw)?
        .into_owned_data()?;
    let nblocks = n / 128;
    let mut unified = unsafe { dev.alloc::<u8>(nblocks.div_ceil(8) * 1152)? };
    {
        let (dp, _dg) = unified.device_ptr_mut(&stream);
        unsafe {
            run_quantize_block(
                ptr as *const f32,
                dp as *mut std::ffi::c_void,
                n as i32,
                QTYPE_Q8A128,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
    }
    dev.synchronize()?;
    let a: Vec<u8> = dev.memcpy_dtov(&typed.slice(..))?;
    let b: Vec<u8> = dev.memcpy_dtov(&unified.slice(..))?;
    // Tile meta is `ds[0]` plus 12 bytes of alignment pad that neither producer
    // writes; comparing whole buffers compares two allocations' leftovers and
    // fails at random. See `q8a128_f16_bf16_paths_match_f32`.
    let meaningful = |raw: &[u8]| -> Vec<u8> {
        let mut out = Vec::with_capacity(nblocks * 132);
        for tile in 0..nblocks {
            let qs = (tile >> 3) * 1152 + (tile & 7) * 128;
            let ds = (tile >> 3) * 1152 + 1024 + (tile & 7) * 16;
            out.extend_from_slice(&raw[qs..qs + 128]);
            out.extend_from_slice(&raw[ds..ds + 4]);
        }
        out
    };
    assert_eq!(
        meaningful(&a),
        meaningful(&b),
        "run_quantize_block(qtype=36) must equal typed q8a128 quantize byte-for-byte"
    );

    // Unified dequant (out_dtype 0 = F32) vs typed dequantize_q8a128.
    let typed_deq = dequantize_q8a128(&typed, rows, cols, &dev)?;
    let mut unified_deq = unsafe { dev.alloc::<f32>(n)? };
    {
        let (sp, _sg) = unified.device_ptr(&stream);
        let (op, _og) = unified_deq.device_ptr_mut(&stream);
        unsafe {
            run_dequantize_block(
                sp as *const std::ffi::c_void,
                op as *mut std::ffi::c_void,
                n as i32,
                QTYPE_Q8A128,
                0, // unified ordering: 0 = F32
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
    }
    dev.synchronize()?;
    let da: Vec<f32> = dev.memcpy_dtov(&typed_deq.slice(..))?;
    let db: Vec<f32> = dev.memcpy_dtov(&unified_deq.slice(..))?;
    for i in 0..n {
        assert_eq!(da[i].to_bits(), db[i].to_bits(), "dequant mismatch at {i}");
    }
    println!(
        "q8a128 unified dispatch (qtype=36) == typed path: {} quant bytes, {n} deq values",
        a.len()
    );
    Ok(())
}

/// GPU KO weight quantize/dequant must be byte-identical to the CPU reference
/// (`ko_quant`): raw-byte assertion for the quantize, bit-exact f32 for the dequant.
/// This pins the lane-major pack + the de-interleave on the device.
#[test]
pub(super) fn ko_gpu_quantize_dequant_matches_cpu() -> Result<()> {
    use crate::quantized::ko_quant::{dequant_ko, quantize_ko};
    let dev = CudaDevice::new(0)?;
    let (nrows, ncols) = (64usize, 256usize);
    let mut s = 0x2545_F491u32;
    let w: Vec<f32> = (0..nrows * ncols)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    for dtype in [
        GgmlDType::Q2_KO,
        GgmlDType::Q3_KO,
        GgmlDType::Q4_KO,
        GgmlDType::Q5_KO,
        GgmlDType::Q6_KO,
        GgmlDType::Q8_KO,
    ] {
        let cpu_bytes = quantize_ko(&w, nrows, ncols, dtype);
        let gpu = quantize_ko_weights(&w, nrows, ncols, dtype, &dev)?;
        let gpu_bytes: Vec<u8> = dev.memcpy_dtov(&gpu.slice(..))?;
        assert_eq!(
            gpu_bytes, cpu_bytes,
            "{dtype:?}: GPU quantize_ko bytes must equal the CPU reference"
        );
        // Dequant: GPU vs CPU on the same bytes, bit-for-bit.
        let cpu_deq = dequant_ko(&cpu_bytes, nrows, ncols, dtype);
        let gpu_deq_dev = dequant_ko_weights(&gpu, nrows, ncols, dtype, &dev)?;
        let gpu_deq: Vec<f32> = dev.memcpy_dtov(&gpu_deq_dev.slice(..))?;
        assert_eq!(gpu_deq.len(), cpu_deq.len());
        for i in 0..gpu_deq.len() {
            assert_eq!(
                gpu_deq[i].to_bits(),
                cpu_deq[i].to_bits(),
                "{dtype:?}: GPU dequant_ko bit mismatch at element {i}"
            );
        }
    }
    Ok(())
}

/// Verify the F16 and BF16 quantize/dequant paths (the `q8a128_load4`/`store4`
/// specializations the f32 byte-exact tests don't exercise). f16/bf16 → f32 is
/// lossless, so quantizing a typed input MUST yield the exact same block bytes as
/// quantizing the identical values as f32 (already byte-exact-validated); and
/// dequant-to-typed must equal the typed rounding of the f32 dequant. Bit-exact.
#[test]
pub(super) fn q8a128_f16_bf16_paths_match_f32() -> Result<()> {
    use candle_kernels::simple::quantized::run_dequantize_q8a128;
    use std::ffi::c_void;
    let dev = CudaDevice::new(0)?;
    let rows = 4usize;
    let cols = 512usize;
    let n = rows * cols;
    // A q8a1024 buffer is not byte-comparable as a whole: each tile's 16-byte
    // meta slot carries data only in `ds[0]`, and `ds[1..3]` is alignment pad
    // that no producer writes and no consumer reads, so it holds whatever the
    // fresh allocation held. Comparing raw buffers made this test a coin toss on
    // uninitialised device memory. Compare exactly the bytes the format defines.
    let meaningful = |raw: &[u8]| -> Vec<u8> {
        let mut out = Vec::with_capacity((n / 128) * 132);
        for tile in 0..n / 128 {
            let qs = (tile >> 3) * 1152 + (tile & 7) * 128;
            let ds = (tile >> 3) * 1152 + 1024 + (tile & 7) * 16;
            out.extend_from_slice(&raw[qs..qs + 128]);
            out.extend_from_slice(&raw[ds..ds + 4]);
        }
        out
    };
    let base: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.37).collect();
    let stream = dev.cuda_stream();

    // F16: dtype 0. The block bytes must equal the f32-of-same-values block; the
    // f16 dequant must equal f16::from_f32 of the f32 dequant.
    {
        let tv: Vec<half::f16> = base.iter().map(|&x| half::f16::from_f32(x)).collect();
        let as_f32: Vec<f32> = tv.iter().map(|h| h.to_f32()).collect();
        let tdev = dev.memcpy_stod(&tv)?;
        let fdev = dev.memcpy_stod(&as_f32)?;
        let (tp, _a) = tdev.device_ptr(&stream);
        let (fp, _b) = fdev.device_ptr(&stream);
        let blk_t = quantize_acts_q8a128(tp, 0, rows, cols, &dev, Backing::Owned, SumScale::Raw)?
            .into_owned_data()?;
        let blk_f = quantize_acts_q8a128(fp, 2, rows, cols, &dev, Backing::Owned, SumScale::Raw)?
            .into_owned_data()?;
        dev.synchronize()?;
        let bt: Vec<u8> = dev.memcpy_dtov(&blk_t.slice(..))?;
        let bf: Vec<u8> = dev.memcpy_dtov(&blk_f.slice(..))?;
        assert_eq!(
            meaningful(&bt),
            meaningful(&bf),
            "F16 quantize must byte-match the f32 path on identical values"
        );

        let deq_f = dequantize_q8a128(&blk_f, rows, cols, &dev)?;
        let mut deq_t = unsafe { dev.alloc::<half::f16>(n)? };
        {
            let (op, _g) = deq_t.device_ptr_mut(&stream);
            let (sp, _gs) = blk_f.device_ptr(&stream);
            unsafe {
                run_dequantize_q8a128(
                    sp as *const c_void,
                    op as *mut c_void,
                    rows as i32,
                    cols as i32,
                    0,
                    stream.cu_stream() as *mut c_void,
                );
            }
        }
        dev.synchronize()?;
        let df: Vec<f32> = dev.memcpy_dtov(&deq_f.slice(..))?;
        let dt: Vec<half::f16> = dev.memcpy_dtov(&deq_t.slice(..))?;
        for i in 0..n {
            assert_eq!(
                dt[i].to_bits(),
                half::f16::from_f32(df[i]).to_bits(),
                "F16 dequant[{i}]"
            );
        }
    }

    // BF16: dtype 1.
    {
        let tv: Vec<half::bf16> = base.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let as_f32: Vec<f32> = tv.iter().map(|h| h.to_f32()).collect();
        let tdev = dev.memcpy_stod(&tv)?;
        let fdev = dev.memcpy_stod(&as_f32)?;
        let (tp, _a) = tdev.device_ptr(&stream);
        let (fp, _b) = fdev.device_ptr(&stream);
        let blk_t = quantize_acts_q8a128(tp, 1, rows, cols, &dev, Backing::Owned, SumScale::Raw)?
            .into_owned_data()?;
        let blk_f = quantize_acts_q8a128(fp, 2, rows, cols, &dev, Backing::Owned, SumScale::Raw)?
            .into_owned_data()?;
        dev.synchronize()?;
        let bt: Vec<u8> = dev.memcpy_dtov(&blk_t.slice(..))?;
        let bf: Vec<u8> = dev.memcpy_dtov(&blk_f.slice(..))?;
        assert_eq!(
            meaningful(&bt),
            meaningful(&bf),
            "BF16 quantize must byte-match the f32 path on identical values"
        );

        let deq_f = dequantize_q8a128(&blk_f, rows, cols, &dev)?;
        let mut deq_t = unsafe { dev.alloc::<half::bf16>(n)? };
        {
            let (op, _g) = deq_t.device_ptr_mut(&stream);
            let (sp, _gs) = blk_f.device_ptr(&stream);
            unsafe {
                run_dequantize_q8a128(
                    sp as *const c_void,
                    op as *mut c_void,
                    rows as i32,
                    cols as i32,
                    1,
                    stream.cu_stream() as *mut c_void,
                );
            }
        }
        dev.synchronize()?;
        let df: Vec<f32> = dev.memcpy_dtov(&deq_f.slice(..))?;
        let dt: Vec<half::bf16> = dev.memcpy_dtov(&deq_t.slice(..))?;
        for i in 0..n {
            assert_eq!(
                dt[i].to_bits(),
                half::bf16::from_f32(df[i]).to_bits(),
                "BF16 dequant[{i}]"
            );
        }
    }

    println!("q8a128 F16 + BF16 quantize/dequant paths verified bit-exact against f32");
    Ok(())
}

#[cfg(test)]
pub(super) fn read_bf16_tensor(dev: &CudaDevice, t: &crate::Tensor) -> Result<Vec<f32>> {
    let s = t.storage_and_layout().0;
    let c = match &*s {
        crate::Storage::Cuda(s) => s,
        _ => panic!("expected CUDA storage"),
    };
    let b = c.as_cuda_slice::<bf16>()?;
    Ok(dev
        .memcpy_dtov(&b.slice(..))?
        .iter()
        .map(|&v| bf16::to_f32(v))
        .collect())
}

#[cfg(test)]
pub(super) fn read_f32_tensor(dev: &CudaDevice, t: &crate::Tensor) -> Result<Vec<f32>> {
    let s = t.storage_and_layout().0;
    let c = match &*s {
        crate::Storage::Cuda(s) => s,
        _ => panic!("expected CUDA storage"),
    };
    let b = c.as_cuda_slice::<f32>()?;
    Ok(dev.memcpy_dtov(&b.slice(..))?)
}

/// Quantize f32 activations [total_batch, ncols] → `block_q8a128[total_batch][ncols/128]`
/// (the pre-quantize feeding the INT8 grouped kernel). Synchronizes so the f32
/// staging buffer is safe to drop.
#[cfg(test)]
pub(super) fn quantize_acts_q8a128_test(
    dev: &CudaDevice,
    act_data: &[f32],
    total_batch: usize,
    ncols: usize,
) -> Result<Q8a128Operand<'static>> {
    let f32_dev = dev.memcpy_stod(act_data)?;
    let stream = dev.cuda_stream();
    let (ptr, _g) = f32_dev.device_ptr(&stream);
    let out = quantize_acts_q8a128(
        ptr,
        2, /* F32 */
        total_batch,
        ncols,
        dev,
        Backing::Owned,
        SumScale::Raw,
    )?;
    dev.synchronize()?;
    Ok(out)
}

pub(super) fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]) as f64;
        num += d * d;
        den += (b[i] as f64) * (b[i] as f64);
    }
    (num / den.max(1e-12)).sqrt()
}

/// Phase-2a producer fusion: `rms_norm_q8a128` (fused RMSNorm → q8a128 in one kernel) must match
/// the unfused `rms_norm` → `quantize_acts_q8a128` oracle within a tight float margin, for BOTH
/// the f16 and bf16 input variants. Both operands are dequantized and compared by rel-L2 (the
/// quant grid is per-128, so ULP-level reduction differences can flip a quant by ±1; the margin
/// absorbs that while still proving the fused path reproduces the two-call path).
#[test]
pub(super) fn rms_norm_q8a128_matches_reference() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (rows, cols) = (20usize, 2048usize);
    let eps = 1e-6f32;
    let mut rng = rand::rng();

    for dtype in [crate::DType::F16, crate::DType::BF16] {
        let xf: Vec<f32> = (0..rows * cols)
            .map(|_| rng.random_range(-1.0f32..1.0))
            .collect();
        let af: Vec<f32> = (0..cols).map(|_| rng.random_range(0.5f32..1.5)).collect();
        let xs = crate::Tensor::from_vec(xf, (rows, cols), &device)?.to_dtype(dtype)?;
        let alpha = crate::Tensor::from_vec(af, (cols,), &device)?.to_dtype(dtype)?;

        // Oracle: rms_norm in f32, rounded to the store dtype (mirrors the standalone FP store),
        // then quantized via `to_dynamic` — i.e. the unfused two-call path.
        let xf32 = xs.to_dtype(crate::DType::F32)?;
        let ms = xf32.sqr()?.mean_keepdim(1)?;
        let rms = (ms + eps as f64)?.sqrt()?;
        let normed = xf32
            .broadcast_div(&rms)?
            .broadcast_mul(&alpha.to_dtype(crate::DType::F32)?)?
            .to_dtype(dtype)?;
        let oracle_op = match to_dynamic(&normed, Int8Mode::Performance, &dev, SumScale::Raw)? {
            DynamicActs::Int8(op) => op,
            DynamicActs::Float(_) => unreachable!("Performance mode yields Int8"),
        };
        let oracle_deq = dequantize_q8a128(oracle_op.data_slice()?, rows, cols, &dev)?;
        let oracle = dev.memcpy_dtov(&oracle_deq.slice(..))?;

        // Fused: single kernel.
        // Raw, matching the `to_dynamic` oracle this is compared against — the
        // two must agree on the convention or the comparison measures that
        // rather than the fusion.
        let fused_op = rms_norm_q8a128(&xs, &alpha, eps, &dev, Backing::Owned, SumScale::Raw)?;
        let fused_deq = dequantize_q8a128(fused_op.data_slice()?, rows, cols, &dev)?;
        let fused = dev.memcpy_dtov(&fused_deq.slice(..))?;

        let rel = rel_l2(&fused, &oracle);
        println!("rms_norm_q8a128 {dtype:?}: rel_l2 vs unfused oracle = {rel:.5}");
        assert!(
            rel < 0.02,
            "{dtype:?} fused rms_norm_q8a128 diverged from oracle: rel_l2={rel:.5}"
        );
    }
    Ok(())
}

/// The tile gather must hand the experts **the same bytes** whichever order the work is done
/// in: quantizing the tokens once and gathering tiles of the result, or gathering the float
/// rows and quantizing those. Quantization is per 128-element tile, so the two agree exactly —
/// any difference is a wrong row or a wrong tile, not rounding. Repeated and out-of-order ids
/// (a token routed to several experts), at the released 2,560 width, whose rows straddle
/// super-blocks and which the byte-row gather could not serve, and at 2,048, where rows are
/// whole super-blocks and the tile gather must reproduce the row copy.
#[test]
pub(super) fn q8a128_tile_gather_matches_gather_then_quantize() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let tokens = 5usize;
    // Eight output rows, so every row count here is a whole number of super-blocks and every
    // byte of both buffers is written.
    let ids: Vec<u32> = vec![3, 0, 4, 4, 1, 2, 0, 3];
    let ids_dev = dev.memcpy_stod(&ids)?;
    for cols in [2560usize, 2048] {
        let xf: Vec<f32> = (0..tokens * cols)
            .map(|i| ((i * 7919 % 2003) as f32 - 1001.0) * 0.003)
            .collect();
        let xs = crate::Tensor::from_vec(xf, (tokens, cols), &device)?;
        let quantize = |t: &crate::Tensor| -> Result<Q8a128Operand<'static>> {
            match to_dynamic(t, Int8Mode::Precision, &dev, SumScale::Raw)? {
                DynamicActs::Int8(op) => Ok(op),
                DynamicActs::Float(_) => unreachable!("Precision mode yields Int8"),
            }
        };
        let want = quantize(&fused_moe_gather(&xs, &ids_dev, ids.len(), &dev)?)?;
        let got =
            fused_moe_gather_q8a128(&quantize(&xs)?, &ids_dev, ids.len(), &dev, Backing::Owned)?;
        let want_bytes = dev.memcpy_dtov(&want.data_slice()?.slice(..))?;
        let got_bytes = dev.memcpy_dtov(&got.data_slice()?.slice(..))?;
        assert_eq!(got_bytes.len(), want_bytes.len(), "{cols}");
        // Only the bytes a quantize defines: each tile's 128 quants and its ds[0]
        // `{scale, sum}` half2. The other 12 bytes of a tile's ds slot are alignment pad,
        // which the quantize leaves as allocated.
        for tile in 0..ids.len() * cols / 128 {
            let block = (tile / 8) * 1152;
            let qs = block + (tile % 8) * 128;
            let ds = block + 1024 + (tile % 8) * 16;
            assert_eq!(
                got_bytes[qs..qs + 128],
                want_bytes[qs..qs + 128],
                "{cols}: tile {tile} quants differ"
            );
            assert_eq!(
                got_bytes[ds..ds + 4],
                want_bytes[ds..ds + 4],
                "{cols}: tile {tile} scale/sum differ"
            );
        }
    }
    Ok(())
}

/// Phase-2b producer fusion: `silu_mul_q8a128` (fused SwiGLU → q8a128) must match the unfused
/// `silu(gate)·up` → `quantize_acts_q8a128` oracle within float margin, for BOTH f16 and bf16.
/// The reference uses exact sigmoid; the kernel uses the production fast-exp silu, so the margin
/// covers the fast-exp approximation plus the per-128 quant grid.
#[test]
pub(super) fn silu_mul_q8a128_matches_reference() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (rows, cols) = (24usize, 1408usize); // cols % 128 == 0
    let mut rng = rand::rng();

    for dtype in [crate::DType::F16, crate::DType::BF16] {
        let gf: Vec<f32> = (0..rows * cols)
            .map(|_| rng.random_range(-3.0f32..3.0))
            .collect();
        let uf: Vec<f32> = (0..rows * cols)
            .map(|_| rng.random_range(-2.0f32..2.0))
            .collect();
        let gate = crate::Tensor::from_vec(gf, (rows, cols), &device)?.to_dtype(dtype)?;
        let up = crate::Tensor::from_vec(uf, (rows, cols), &device)?.to_dtype(dtype)?;

        // Oracle: silu(gate)*up in f32 (exact sigmoid), rounded to dtype, then quantized.
        let g = gate.to_dtype(crate::DType::F32)?;
        let u = up.to_dtype(crate::DType::F32)?;
        let sig = (g.neg()?.exp()? + 1.0)?.recip()?;
        let outf = (&g * &sig)?.mul(&u)?.to_dtype(dtype)?;
        let oracle_op = match to_dynamic(&outf, Int8Mode::Performance, &dev, SumScale::Raw)? {
            DynamicActs::Int8(op) => op,
            DynamicActs::Float(_) => unreachable!("Performance mode yields Int8"),
        };
        let oracle_deq = dequantize_q8a128(oracle_op.data_slice()?, rows, cols, &dev)?;
        let oracle = dev.memcpy_dtov(&oracle_deq.slice(..))?;

        // Fused: single kernel.
        // Raw, matching the `to_dynamic` oracle — see the sibling rms_norm test.
        let fused_op = silu_mul_q8a128(&gate, &up, &dev, Backing::Owned, SumScale::Raw)?;
        let fused_deq = dequantize_q8a128(fused_op.data_slice()?, rows, cols, &dev)?;
        let fused = dev.memcpy_dtov(&fused_deq.slice(..))?;

        let rel = rel_l2(&fused, &oracle);
        println!("silu_mul_q8a128 {dtype:?}: rel_l2 vs unfused oracle = {rel:.5}");
        assert!(
            rel < 0.03,
            "{dtype:?} fused silu_mul_q8a128 diverged from oracle: rel_l2={rel:.5}"
        );
    }
    Ok(())
}
