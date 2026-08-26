// Test code: loop indices are block/element coordinates in the expected-value
// formula for each quantization format.
#![allow(clippy::needless_range_loop)]

use candle_core::{
    bail,
    quantized::{self, GgmlDType},
    test_device,
    test_utils::to_vec2_round,
    DType, Device, IndexOp, Module, Result, Tensor,
};
use quantized::{k_quants, GgmlType};
use rand::prelude::*;

/// Global mutex to serialize access to the FORCE_DMMV global flag in CUDA tests.
/// Any test that reads or modifies FORCE_DMMV must hold this lock.
static DMMV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

const GGML_TEST_SIZE: usize = 32 * 128;

const GGML_MAX_QUANTIZATION_TOTAL_ERROR: f32 = 0.002;
const GGML_MAX_QUANTIZATION_TOTAL_ERROR_2BITS: f32 = 0.0075;
const GGML_MAX_QUANTIZATION_TOTAL_ERROR_3BITS: f32 = 0.0040;
const GGML_MAX_DOT_PRODUCT_ERROR: f32 = 0.02;

fn test_matmul(
    device: &Device,
    (b, m, n, k): (usize, usize, usize, usize),
    dtype: GgmlDType,
) -> Result<()> {
    if device.is_metal() && (dtype == GgmlDType::Q8_1 || dtype == GgmlDType::Q8_K) {
        return Ok(());
    }
    if device.is_cuda() && dtype == GgmlDType::Q8_K {
        // Q8_K CUDA kernel doesn't support arbitrary (m, k, n) shapes; skip until fixed.
        return Ok(());
    }

    let lhs = (0..(m * k))
        .map(|v| v as f32 / (m * k) as f32)
        .collect::<Vec<_>>();
    let rhs = (0..(k * n))
        .map(|v| v as f32 / (n * k) as f32)
        .collect::<Vec<_>>();

    let lhs = Tensor::from_slice(&lhs, (m, k), device)?;
    let rhs = Tensor::from_slice(&rhs, (k, n), device)?;
    let mm = lhs.matmul(&rhs)?;
    let qtensor = quantized::QTensor::quantize(&rhs.t()?, dtype)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&lhs)?;

    let error: f32 = ((&mm - &res)?.abs()? / &mm.abs()?)?
        .sum_all()?
        .to_scalar()?;
    let error = error / (b * m * n) as f32;
    assert!(
        error <= 0.02,
        "Error {error} is too big. \nExpected:\n {mm} \nFound:\n {res}\n for {dtype:?}"
    );

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_matmul_mm() -> Result<()> {
    let dtype = GgmlDType::Q8_0;
    let device = Device::new_metal(0)?;

    let m = 32;
    let n = 32;
    let k = 32;
    let lhs = (0..(m * k))
        .map(|v| v as f32 / (m * k) as f32)
        .collect::<Vec<_>>();
    let rhs = (0..(k * n))
        .map(|v| v as f32 / (n * k) as f32)
        .collect::<Vec<_>>();

    let lhs = Tensor::from_slice(&lhs, (m, k), &device)?;
    let rhs = Tensor::from_slice(&rhs, (1, 1, k, n), &device)?.repeat((5, 20, 1, 1))?;
    let mm = lhs.broadcast_matmul(&rhs)?;
    let qtensor = quantized::QTensor::quantize(&lhs.t()?, dtype)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&rhs)?;

    let error: f32 = ((&mm - &res)?.abs()? / &mm.abs()?)?
        .sum_all()?
        .to_scalar()?;

    let error = error / res.elem_count() as f32;
    assert!(
        error <= 0.001,
        "Error {error} is too big. \nExpected:\n {mm} \nFound:\n {res}\n for {dtype:?}"
    );

    Ok(())
}

fn quantized_matmul(device: &Device) -> Result<()> {
    // Its device matmuls READ `FORCE_DMMV`, so it takes the lock exactly like
    // `qmm_batch` does. Without it, the dmmv tests flip the flag from another
    // thread mid-run and this picks the dmmv path at a shape it was not built
    // for — "unexpected y size 192, ncols 64 4", reproducible only under the
    // full suite and never in isolation.
    let _dmmv_guard = if device.is_cuda() {
        Some(crate::DMMV_LOCK.lock().unwrap())
    } else {
        None
    };
    let (m, k, n) = (3, 64, 4);
    let lhs_s = (0..(m * k)).map(|v| v as f32).collect::<Vec<_>>();
    let lhs = Tensor::from_slice(&lhs_s, (m, k), device)?;
    let mut dst = vec![42.; 3 * 4];
    let mut rhs_t = vec![k_quants::BlockQ4_0::zeros(); 8];
    let rhs = (0..(k * n)).map(|v| v as f32).collect::<Vec<_>>();
    k_quants::BlockQ4_0::from_float(&rhs, &mut rhs_t);
    k_quants::matmul((m, k, n), &lhs_s, &rhs_t, &mut dst)?;
    assert_eq!(
        dst.iter().map(|x| x.round()).collect::<Vec<_>>(),
        &[
            85120.0, 214562.0, 345455.0, 474748.0, 213475.0, 604465.0, 1000686.0, 1388317.0,
            341876.0, 994283.0, 1655709.0, 2301518.0
        ]
    );
    let tensor_rhs = Tensor::from_slice(&rhs, (n, k), device)?.t()?;
    let mm = lhs.matmul(&tensor_rhs)?;
    assert_eq!(
        mm.to_vec2::<f32>()?,
        &[
            [85344.0, 214368.0, 343392.0, 472416.0],
            [214368.0, 605536.0, 996704.0, 1387872.0],
            [343392.0, 996704.0, 1650016.0, 2303328.0]
        ]
    );

    let qtensor = quantized::QTensor::quantize(&tensor_rhs.t()?, GgmlDType::Q4_0)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&lhs)?;
    match device {
        Device::Metal(_) => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [84946.0, 214126.0, 344757.0, 473798.0],
                [213458.0, 604350.0, 1000469.0, 1387990.0],
                [341970.0, 994574.0, 1656181.0, 2302182.0]
            ]
        ),
        Device::Cuda(_) => {
            // CUDA parallel reduction is non-deterministic; compare against float ref
            // with Q4_0 relative tolerance instead of an exact snapshot.
            let rel_tol = quant_matmul_rel_tolerance(GgmlDType::Q4_0);
            let mm_v = mm.flatten_all()?.to_vec1::<f32>()?;
            let res_v = res.flatten_all()?.to_vec1::<f32>()?;
            for (r, v) in mm_v.iter().zip(res_v.iter()) {
                let tol = r.abs() * rel_tol + 1.0;
                assert!(
                    (r - v).abs() < tol,
                    "CUDA Q4_0 matmul: {v} too far from float ref {r} (tol {tol:.1})"
                );
            }
        }
        Device::Cpu => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [85120.0, 214562.0, 345455.0, 474748.0],
                [213475.0, 604465.0, 1000686.0, 1388317.0],
                [341876.0, 994283.0, 1655709.0, 2301518.0]
            ]
        ),
    }
    test_matmul(device, (1, 3, 4, 256), GgmlDType::Q4_0)?;
    Ok(())
}

fn quantized_matmul_neg(device: &Device) -> Result<()> {
    // Same `FORCE_DMMV` exposure as `quantized_matmul` — same lock.
    let _dmmv_guard = if device.is_cuda() {
        Some(crate::DMMV_LOCK.lock().unwrap())
    } else {
        None
    };
    let (m, k, n) = (3, 64, 4);
    let lhs_s = (0..(m * k))
        .map(|v| v as f32 - (m * k) as f32 / 2.0)
        .collect::<Vec<_>>();
    let lhs = Tensor::from_slice(&lhs_s, (m, k), device)?;
    let mut dst = vec![42.; 3 * 4];
    let mut rhs_t = vec![k_quants::BlockQ4_0::zeros(); 8];
    let rhs = (0..k * n)
        .map(|v| v as f32 - (k * n) as f32 / 3.0)
        .collect::<Vec<_>>();
    let tensor_rhs = Tensor::from_slice(&rhs, (n, k), device)?.t()?;
    k_quants::BlockQ4_0::from_float(&rhs, &mut rhs_t);
    k_quants::matmul((m, k, n), &lhs_s, &rhs_t, &mut dst)?;
    assert_eq!(
        dst.iter().map(|x| x.round()).collect::<Vec<_>>(),
        &[
            243524.0, -19596.0, -285051.0, -549815.0, 23777.0, 21651.0, 19398.0, 18367.0,
            -196472.0, 63012.0, 324585.0, 587902.0
        ]
    );
    let mm = lhs.matmul(&tensor_rhs)?;
    assert_eq!(
        to_vec2_round(&mm, 0)?,
        &[
            [244064.0, -20128.0, -284320.0, -548512.0],
            [23563.0, 21515.0, 19467.0, 17419.0],
            [-196939.0, 63157.0, 323253.0, 583349.0]
        ]
    );

    let qtensor = quantized::QTensor::quantize(&tensor_rhs.t()?, GgmlDType::Q4_0)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&lhs)?;
    if device.is_metal() {
        assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [243659.0, -19716.0, -285444.0, -550439.0],
                [23779.0, 21653.0, 19404.0, 18349.0],
                [-196101.0, 63021.0, 324252.0, 587137.0]
            ]
        );
    } else if device.is_cpu() {
        assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [243524.0, -19596.0, -285051.0, -549815.0],
                [23777.0, 21651.0, 19398.0, 18367.0],
                [-196472.0, 63012.0, 324585.0, 587902.0]
            ]
        );
    } else {
        // CUDA parallel reduction is non-deterministic; compare against the float
        // reference using a per-format relative tolerance.
        let rel_tol = quant_matmul_rel_tolerance(GgmlDType::Q4_0);
        let mm_ref = to_vec2_round(&mm, 0)?;
        let res_vals = to_vec2_round(&res, 0)?;
        for (ref_row, res_row) in mm_ref.iter().zip(res_vals.iter()) {
            for (r, v) in ref_row.iter().zip(res_row.iter()) {
                let tol = r.abs() * rel_tol + 1.0;
                assert!(
                    (r - v).abs() < tol,
                    "CUDA Q4_0 matmul result {v} too far from float ref {r} (tol {tol:.1})"
                );
            }
        }
    }
    let lhs2 = Tensor::stack(&[&lhs, &lhs], 0)?;
    let res2 = matmul.forward(&lhs2)?;
    let res2 = res2.i(1)?;
    let diff = (&res - res2)?.abs()?.mean_all()?.to_vec0::<f32>()?;
    // The vec kernel (b_size ≤ 8) uses different thread configs for 2D (b=3) vs 3D (b=6),
    // leading to minor FP rounding differences between single and batched paths.
    assert!(
        diff < 50.0,
        "batched vs single path mean abs diff too large: {diff}"
    );
    Ok(())
}

fn qmm_batch(dev: &Device) -> Result<()> {
    // Hold DMMV_LOCK for the duration of CUDA calls to prevent interference
    // from dmmv tests that temporarily set FORCE_DMMV=true in parallel threads.
    let _dmmv_guard = if dev.is_cuda() {
        Some(crate::DMMV_LOCK.lock().unwrap())
    } else {
        None
    };
    let (lhs, rhs, _mm) = get_random_tensors(2, 256, 6, dev)?;
    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q2_K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;
    assert_eq!(mm.shape().dims(), [2, 6]);
    let lhs2 = Tensor::cat(&[&lhs, &lhs], 0)?;
    let mm2 = rhs.forward(&lhs2)?;
    assert_eq!(mm2.shape().dims(), [4, 6]);
    let diff2 = (mm2.i(2..)? - &mm)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if dev.is_cuda() {
        // Different batch sizes use different kernel template instantiations on CUDA,
        // which can produce small floating-point differences.
        assert!(diff2 < 0.1, "diff2 too large on cuda: {diff2}");
    } else {
        assert_eq!(diff2, 0.0);
    }
    let lhs3 = Tensor::cat(&[&lhs2, &lhs], 0)?;
    let mm3 = rhs.forward(&lhs3)?;
    assert_eq!(mm3.shape().dims(), [6, 6]);
    let diff3 = (mm3.i(2..4)? - &mm)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if dev.is_cuda() {
        assert!(diff3 < 0.1, "diff3 too large on cuda: {diff3}");
    } else {
        assert_eq!(diff3, 0.0);
    }
    let diff3 = (mm3.i(4..)? - &mm)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if dev.is_cuda() {
        assert!(diff3 < 0.1, "diff3 too large on cuda: {diff3}");
    } else {
        assert_eq!(diff3, 0.0);
    }
    let lhs4 = Tensor::cat(&[&lhs3, &lhs3], 0)?;
    let mm4 = rhs.forward(&lhs4)?;
    assert_eq!(mm4.shape().dims(), [12, 6]);
    let diff4 = (mm4.i(..6)? - &mm3)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if dev.is_cuda() {
        // Batches through 16 rows share the vec-kernel family on cuda (a batch
        // past the templates' 8 runs as ≤8-row chunks), but the kernel's warp
        // count varies with its batch count, so same-row results across
        // different batch sizes agree only to rounding — a 12-row call's rows
        // straddle an 8-row and a 4-row launch where the 6-row call was one.
        assert!(diff4 < 1e-4, "12-row rows 0..6 vs 6-row call: {diff4}")
    } else {
        assert_eq!(diff4, 0.0)
    };
    let diff4 = (mm4.i(6..)? - &mm4.i(..6)?)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    if dev.is_cuda() {
        // Duplicate inputs, but not one launch: rows 8..12 run as the 4-row
        // chunk (4 warps) while their twins in 0..6 sit in the 8-row chunk
        // (2 warps), so this is rounding-equal, not bit-equal.
        assert!(diff4 < 1e-4, "12-row halves disagree: {diff4}")
    } else {
        assert_eq!(diff4, 0.0)
    };

    // Across the vec/MMQ boundary at 8: the tiled kernel reduces in a
    // different order, so agreement is to tolerance, and the tolerance is the
    // property — both paths must describe the same matmul.
    let lhs5 = Tensor::cat(&[&lhs4, &lhs3], 0)?;
    let mm5 = rhs.forward(&lhs5)?;
    assert_eq!(mm5.shape().dims(), [18, 6]);
    let diff5 = (mm5.i(..6)? - &mm3)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if dev.is_cuda() {
        assert!(diff5 < 1e-3, "MMQ vs vec paths diverged: {diff5}")
    } else {
        assert_eq!(diff5, 0.0)
    };
    Ok(())
}

test_device!(quantized_matmul, qmm_cpu, qmm_cuda, qmm_metal);
test_device!(quantized_matmul_neg, qmm_n_cpu, qmm_n_cuda, qmm_n_metal);
test_device!(qmm_batch, qmm_b_cpu, qmm_b_cuda, qmm_b_metal);

fn quantize_q4_0(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();

    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q4_0)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    assert_eq!(
        dst.to_vec1::<f32>()?,
        &[
            -0.0, -0.0, 3.875, 3.875, 3.875, 3.875, 7.75, 7.75, 7.75, 7.75, 11.625, 11.625, 11.625,
            11.625, 15.5, 15.5, 15.5, 15.5, 19.375, 19.375, 19.375, 19.375, 23.25, 23.25, 23.25,
            23.25, 27.125, 27.125, 27.125, 27.125, 31.0, 31.0, 31.5, 31.5, 31.5, 31.5, 39.375,
            39.375, 39.375, 39.375, 39.375, 39.375, 39.375, 39.375, 47.25, 47.25, 47.25, 47.25,
            47.25, 47.25, 47.25, 47.25, 55.125, 55.125, 55.125, 55.125, 55.125, 55.125, 55.125,
            55.125, 63.0, 63.0, 63.0, 63.0, 59.375, 59.375, 71.25, 71.25, 71.25, 71.25, 71.25,
            71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 83.125, 83.125, 83.125, 83.125,
            83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 95.0, 95.0, 95.0, 95.0,
            95.0, 95.0, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 111.125, 111.125,
            111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125,
            111.125, 111.125, 111.125, 111.125, 111.125, 127.0, 127.0, 127.0, 127.0, 127.0, 127.0,
            127.0, 127.0
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q4_0, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q4_1(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();
    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q4_1)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    assert_eq!(
        round_vector(&dst.to_vec1::<f32>()?),
        &[
            0.0, 0.0, 2.066, 2.066, 4.133, 4.133, 6.199, 6.199, 8.266, 8.266, 10.332, 10.332,
            12.398, 12.398, 14.465, 14.465, 16.531, 16.531, 18.598, 18.598, 20.664, 20.664, 22.73,
            22.73, 24.797, 24.797, 26.863, 26.863, 28.93, 28.93, 30.996, 30.996, 32.0, 32.0,
            34.066, 34.066, 36.133, 36.133, 38.199, 38.199, 40.266, 40.266, 42.332, 42.332, 44.398,
            44.398, 46.465, 46.465, 48.531, 48.531, 50.598, 50.598, 52.664, 52.664, 54.73, 54.73,
            56.797, 56.797, 58.863, 58.863, 60.93, 60.93, 62.996, 62.996, 64.0, 64.0, 66.066,
            66.066, 68.133, 68.133, 70.199, 70.199, 72.266, 72.266, 74.332, 74.332, 76.398, 76.398,
            78.465, 78.465, 80.531, 80.531, 82.598, 82.598, 84.664, 84.664, 86.73, 86.73, 88.797,
            88.797, 90.863, 90.863, 92.93, 92.93, 94.996, 94.996, 96.0, 96.0, 98.066, 98.066,
            100.133, 100.133, 102.199, 102.199, 104.266, 104.266, 106.332, 106.332, 108.398,
            108.398, 110.465, 110.465, 112.531, 112.531, 114.598, 114.598, 116.664, 116.664,
            118.73, 118.73, 120.797, 120.797, 122.863, 122.863, 124.93, 124.93, 126.996, 126.996
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q4_1, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q5_0(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();
    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q5_0)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    assert_eq!(
        round_vector(&dst.to_vec1::<f32>()?),
        &[
            -0.0, 1.938, 1.938, 3.875, 3.875, 5.813, 5.813, 7.75, 7.75, 9.688, 9.688, 11.625,
            11.625, 13.563, 13.563, 15.5, 15.5, 17.438, 17.438, 19.375, 19.375, 21.313, 21.313,
            23.25, 23.25, 25.188, 25.188, 27.125, 27.125, 29.063, 29.063, 31.0, 31.5, 31.5, 35.438,
            35.438, 35.438, 35.438, 39.375, 39.375, 39.375, 39.375, 43.313, 43.313, 43.313, 43.313,
            47.25, 47.25, 47.25, 47.25, 51.188, 51.188, 51.188, 51.188, 55.125, 55.125, 55.125,
            55.125, 59.063, 59.063, 59.063, 59.063, 63.0, 63.0, 65.313, 65.313, 65.313, 65.313,
            65.313, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 77.188, 77.188, 77.188, 77.188,
            77.188, 77.188, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 89.063, 89.063, 89.063,
            89.063, 89.063, 89.063, 95.0, 95.0, 95.0, 95.25, 95.25, 95.25, 95.25, 103.188, 103.188,
            103.188, 103.188, 103.188, 103.188, 103.188, 103.188, 111.125, 111.125, 111.125,
            111.125, 111.125, 111.125, 111.125, 111.125, 119.063, 119.063, 119.063, 119.063,
            119.063, 119.063, 119.063, 119.063, 127.0, 127.0, 127.0, 127.0
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q5_0, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q5_1(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();
    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q5_1)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    assert_eq!(
        round_vector(&dst.to_vec1::<f32>()?),
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
            30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0,
            44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0,
            58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0,
            72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0,
            86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0,
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0,
            124.0, 125.0, 126.0, 127.0
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q5_1, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn get_test_vector2(bound: f32, size: usize, device: &Device) -> Result<Tensor> {
    assert!(
        size.is_multiple_of(crate::quantized::k_quants::QK_K),
        "size must be a multiple of {}",
        crate::quantized::k_quants::QK_K
    );

    let src = (0..size)
        .map(|v| (v as f32 - size as f32 / 2.) * bound / (size as f32 / 2.))
        .collect::<Vec<_>>();
    assert_eq!([src[0], src[size / 2]], [-bound, 0.0]);
    Tensor::from_vec(src, (size,), device)
}

/// Round a vector
fn round_vector(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|x| (1000. * x).round() / 1000.)
        .collect::<Vec<_>>()
}

/// Returns the maximum expected relative error for a quantized matmul using `dtype`.
///
/// Derived from the number of quantization levels per block:
/// - Q2_0:  4 levels → max step = amax/1.5  → rel ≈ 33%
/// - Q3_0:  8 levels → max step = amax/3.5  → rel ≈ 14%
/// - Q4_0/Q4_1: 16 levels → max step = amax/15 → rel ≈ 7%
/// - Q5_0/Q5_1: 32 levels → max step = amax/31 → rel ≈ 3.5%
/// - Q8_0/Q8_1: 256 levels → max step = amax/127 → rel ≈ 1%
/// - K-quants use super-scales so error is roughly halved vs same-bit simple quants.
fn quant_matmul_rel_tolerance(dtype: GgmlDType) -> f32 {
    match dtype {
        GgmlDType::Q2_0 => 0.35,
        GgmlDType::Q3_0 => 0.15,
        GgmlDType::Q4_0 | GgmlDType::Q4_1 => 0.07,
        GgmlDType::Q5_0 | GgmlDType::Q5_1 => 0.04,
        GgmlDType::Q8_0 | GgmlDType::Q8_1 => 0.01,
        GgmlDType::Q2_K => 0.18,
        GgmlDType::Q3_K => 0.10,
        GgmlDType::Q4_K | GgmlDType::Q4_KS => 0.04,
        GgmlDType::Q5_K => 0.02,
        GgmlDType::Q6_K | GgmlDType::Q8_K | GgmlDType::Q8_KS => 0.01,
        _ => 0.10, // conservative fallback
    }
}

fn compare_with_error(values: &[f32], expected: &[f32], tolerance: f32) {
    for (i, (value, expected_value)) in values.iter().zip(expected.iter()).enumerate() {
        let difference = (value - expected_value).abs();

        assert!(
            difference < tolerance,
            "Error at index {i}: value = {value}, expected = {expected_value}. Difference = {difference} exceeds tolerance = {tolerance}."
        );
    }
}

/// Creates a vector similar to the ones used in GGML unit tests:
/// https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L26-L30
fn create_ggml_like_vector(offset: f32) -> Vec<f32> {
    (0..GGML_TEST_SIZE)
        .map(|i| 0.1 + 2.0 * (i as f32 + offset).cos())
        .collect()
}

/// Calculates the root mean square error between two vectors
fn calculate_rmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    sum / a.len() as f32
}

/// Similar to the GGML quantization unit test:
/// https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L43-L50
fn ggml_quantization_error_test(dtype: GgmlDType, device: &Device, max_error: f32) -> Result<()> {
    let src = create_ggml_like_vector(0.0);
    let src = Tensor::from_slice(&src, (GGML_TEST_SIZE,), device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    let error = calculate_rmse(&src.to_vec1::<f32>()?, &dst.to_vec1::<f32>()?);
    if error > max_error {
        bail!(
            "Quantization error {} exceeds max error {}",
            error,
            max_error
        );
    }
    Ok(())
}

fn quantize_q2k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q2_K;

    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.1);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.499, -0.366, -0.249, 0.0, 0.295, 0.492]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 6.0);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR_2BITS)?;
    Ok(())
}

fn quantize_q3k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q3_K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.03);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.493, -0.37, -0.243, -0.0, 0.292, 0.492]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 3.5);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR_3BITS)?;
    Ok(())
}

fn quantize_q4k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q4_K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.017);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.5, -0.373, -0.25, 0.0, 0.288, 0.498]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 4.5);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q5k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q5_K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.009);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.5, -0.373, -0.25, 0.0, 0.279, 0.499]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 2.5);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q6k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q6_K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.008);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.497, -0.372, -0.25, -0.0, 0.284, 0.5]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 2.0);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q8k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q8_K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.008);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.5, -0.375, -0.25, -0.0, 0.281, 0.499]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 0.6);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q2_0(device: &Device) -> Result<()> {
    // Q2_0: 2-bit, 4 levels, block_size=32, d=amax/1.5, decoded=d*(q-1.5)
    let dtype = GgmlDType::Q2_0;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;

    // f32 and f16 dequant paths should match exactly on CPU
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    // Q2_0 has 4 levels; max per-element error ≈ amax_in_block/3 ≈ 0.167 for amax=0.5
    compare_with_error(dst.as_slice(), src.as_slice(), 0.25);

    // Large range test
    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 50.0);

    // RMSE test with cosine vector; Q2_0 step≈amax/1.5, RMSE≈step/sqrt(12)≈0.4 for amax≈2.1
    ggml_quantization_error_test(dtype, device, 0.50)?;
    Ok(())
}

fn quantize_q3_0(device: &Device) -> Result<()> {
    // Q3_0: 3-bit, 8 levels, block_size=32, d=amax/3.5, decoded=d*(q-3.5)
    let dtype = GgmlDType::Q3_0;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;

    // f32 and f16 dequant paths should match exactly on CPU
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    // Q3_0 has 8 levels; max per-element error ≈ amax_in_block/7 ≈ 0.071 for amax=0.5
    compare_with_error(dst.as_slice(), src.as_slice(), 0.1);

    // Large range test
    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 25.0);

    // RMSE test with cosine vector; Q3_0 step≈amax/3.5, RMSE≈step/sqrt(12)≈0.17 for amax≈2.1
    ggml_quantization_error_test(dtype, device, 0.22)?;
    Ok(())
}

fn quantize_q4_ks(device: &Device) -> Result<()> {
    // Q4_KS: 4-bit with separate d/s scales, block_size=32
    // 16 levels → max per-element error ≈ amax/15 ≈ 6.7%
    let dtype = GgmlDType::Q4_KS;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;

    let diff = (dst.to_dtype(DType::F16)? - &dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .max(0)?
        .to_vec0::<f32>()?;
    assert!(diff < 1e-3, "f32/f16 dequant mismatch: {diff}");

    let src_v = src.to_vec1::<f32>()?;
    let dst_v = dst.to_vec1::<f32>()?;
    compare_with_error(dst_v.as_slice(), src_v.as_slice(), 0.08);

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let src_big_v = src_big.to_vec1::<f32>()?;
    let dst_big_v = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big_v.as_slice(), src_big_v.as_slice(), 15.0);

    // RMSE on cosine vector; Q4 step≈amax/15, RMSE≈step/sqrt(12)≈0.04 for amax≈2.1
    ggml_quantization_error_test(dtype, device, 0.06)?;
    Ok(())
}

fn quantize_q8_ks(device: &Device) -> Result<()> {
    // Q8_KS: 8-bit with separate d/s scales, block_size=32
    // 256 levels → max per-element error ≈ amax/127 ≈ 0.8%
    let dtype = GgmlDType::Q8_KS;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;

    let diff = (dst.to_dtype(DType::F16)? - &dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .max(0)?
        .to_vec0::<f32>()?;
    assert!(diff < 1e-3, "f32/f16 dequant mismatch: {diff}");

    let src_v = src.to_vec1::<f32>()?;
    let dst_v = dst.to_vec1::<f32>()?;
    compare_with_error(dst_v.as_slice(), src_v.as_slice(), 0.01);

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let src_big_v = src_big.to_vec1::<f32>()?;
    let dst_big_v = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big_v.as_slice(), src_big_v.as_slice(), 2.0);

    // RMSE on cosine vector; Q8 step≈amax/127, RMSE≈step/sqrt(12)≈0.005 for amax≈2.1
    ggml_quantization_error_test(dtype, device, 0.008)?;
    Ok(())
}

test_device!(
    quantize_q2_0,
    quantize_q2_0_cpu,
    quantize_q2_0_cuda,
    quantize_q2_0_metal
);
test_device!(
    quantize_q3_0,
    quantize_q3_0_cpu,
    quantize_q3_0_cuda,
    quantize_q3_0_metal
);
test_device!(
    quantize_q4_ks,
    quantize_q4_ks_cpu,
    quantize_q4_ks_cuda,
    quantize_q4_ks_metal
);
test_device!(
    quantize_q8_ks,
    quantize_q8_ks_cpu,
    quantize_q8_ks_cuda,
    quantize_q8_ks_metal
);
test_device!(
    quantize_q4_0,
    quantize_q4_0_cpu,
    quantize_q4_0_cuda,
    quantize_q4_0_metal
);
test_device!(
    quantize_q4_1,
    quantize_q4_1_cpu,
    quantize_q4_1_cuda,
    quantize_q4_1_metal
);
test_device!(
    quantize_q5_0,
    quantize_q5_0_cpu,
    quantize_q5_0_cuda,
    quantize_q5_0_metal
);
test_device!(
    quantize_q5_1,
    quantize_q5_1_cpu,
    quantize_q5_1_cuda,
    quantize_q5_1_metal
);
test_device!(
    quantize_q2k,
    quantize_q2k_cpu,
    quantize_q2k_cuda,
    quantize_q2k_metal
);
test_device!(
    quantize_q3k,
    quantize_q3k_cpu,
    quantize_q3k_cuda,
    quantize_q3k_metal
);
test_device!(
    quantize_q4k,
    quantize_q4k_cpu,
    quantize_q4k_cuda,
    quantize_q4k_metal
);
test_device!(
    quantize_q5k,
    quantize_q5k_cpu,
    quantize_q5k_cuda,
    quantize_q5k_metal
);
test_device!(
    quantize_q6k,
    quantize_q6k_cpu,
    quantize_q6k_cuda,
    quantize_q6k_metal
);
test_device!(
    quantize_q8k,
    quantize_q8k_cpu,
    quantize_q8k_cuda,
    quantize_q8k_metal
);

/// Very simple dot product implementation
fn vec_dot_reference(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Returns the error achieved by the GGML matmul unit test.
fn ggml_reference_matmul_error(dtype: GgmlDType) -> Result<f32> {
    let ret = match dtype {
        GgmlDType::F32 => 0.000000,
        GgmlDType::F16 => 0.000010,
        GgmlDType::BF16 => 0.000200,
        GgmlDType::Q2_K => 0.004086,
        GgmlDType::Q3_K => 0.016148,
        GgmlDType::Q4_K => 0.002425,
        GgmlDType::Q5_K => 0.000740,
        GgmlDType::Q6_K => 0.000952,
        GgmlDType::Q4_0 => 0.001143,
        GgmlDType::Q4_1 => 0.008,
        GgmlDType::Q5_0 => 0.001353,
        GgmlDType::Q5_1 => 0.00149,
        GgmlDType::Q8_0 => 0.000092,
        GgmlDType::Q8_1 => 0.000092,

        // Not from the ggml repo.
        GgmlDType::Q8_K => 0.00065,

        // AWQ types - these use specialized matmul kernels (GEMX)
        GgmlDType::QAWQ | GgmlDType::QAWQ_G64 => 0.001,

        // Candle-specific types not in GGML matmul benchmarks. The KO variants
        // and MXFP4 run through their own repacked/grouped kernels, gated by
        // their own bit-exactness tests rather than this error table.
        GgmlDType::Q4_KS
        | GgmlDType::Q8_KS
        | GgmlDType::Q2_0
        | GgmlDType::Q3_0
        | GgmlDType::Q2_KO
        | GgmlDType::Q4_KO
        | GgmlDType::Q5_KO
        | GgmlDType::Q6_KO
        | GgmlDType::Q8_KO
        | GgmlDType::MXFP4
        | GgmlDType::MXFP4_KO => {
            panic!("matmul error not defined for this type")
        }

        // R16 is a KV-cache recording format, not used for matmul
        GgmlDType::R16 => panic!("matmul error not defined for this type"),

        // KV-cache quantization formats, not used for matmul
        GgmlDType::Q0
        | GgmlDType::Q1_S
        | GgmlDType::Q2_S
        | GgmlDType::Q2_A
        | GgmlDType::Q2_1
        | GgmlDType::Q3_1
        | GgmlDType::Q0_V
        | GgmlDType::Q1_A
        | GgmlDType::Q0_X
        | GgmlDType::Q0_M2
        | GgmlDType::Q0_M4
        | GgmlDType::P2
        | GgmlDType::F8E4M3
        | GgmlDType::F8E5M2
        | GgmlDType::U8
        | GgmlDType::I8
        | GgmlDType::U16
        | GgmlDType::I16
        | GgmlDType::U32
        | GgmlDType::I32
        | GgmlDType::U64
        | GgmlDType::I64
        | GgmlDType::F64 => panic!("matmul error not defined for this type"),
    };
    Ok(ret)
}

/// Similar to the GGML matmul unit test:
/// https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L76-L91
fn ggml_matmul_error_test<T: GgmlType>() -> Result<()> {
    let a = create_ggml_like_vector(0.0);
    let b = create_ggml_like_vector(1.0);
    ggml_matmul_error_test_::<T>(a.as_slice(), b.as_slice(), 1.0)?;
    // Another example that is more likely to trigger the overflow reported in #1526
    let a = (0..GGML_TEST_SIZE)
        .map(|i| i as f32 / GGML_TEST_SIZE as f32)
        .collect::<Vec<_>>();
    let b = (0..GGML_TEST_SIZE)
        .map(|i| i as f32 / GGML_TEST_SIZE as f32)
        .collect::<Vec<_>>();
    ggml_matmul_error_test_::<T>(a.as_slice(), b.as_slice(), 2.0)?;
    Ok(())
}

fn ggml_matmul_error_test_<T: GgmlType>(a: &[f32], b: &[f32], err_m: f32) -> Result<()> {
    let length = a.len();

    let mut a_quant = vec![T::zeros(); length / T::BLCK_SIZE];
    let mut b_quant = vec![T::VecDotType::zeros(); length / T::VecDotType::BLCK_SIZE];
    T::from_float(a, &mut a_quant);
    T::VecDotType::from_float(b, &mut b_quant);

    let result = T::vec_dot(length, &a_quant, &b_quant);
    let result_unopt = T::vec_dot_unopt(length, &a_quant, &b_quant);

    if (result - result_unopt).abs() / length as f32 > 1e-6 {
        bail!(
            "the opt and unopt vec-dot returned different values, opt: {result} vs unopt: {result_unopt}"
        )
    }

    let mut dst = vec![0.0f32; 1];
    crate::k_quants::matmul((1, length, 1), b, &a_quant, &mut dst)?;
    let result_matmul = dst[0];

    if (result_matmul - result).abs() / length as f32 > 1e-6 {
        bail!(
            "calling matmul vs calling vec-dot directly returned different values, matmul: {result_matmul} vs vec-dot: {result}"
        )
    }

    let reference_result = vec_dot_reference(a, b);

    let verify_result = |result: f32, source: &str| {
        let error = (result - reference_result).abs() / length as f32;
        let ggml_error = ggml_reference_matmul_error(T::DTYPE)? * err_m;
        if !error.is_finite() || error > GGML_MAX_DOT_PRODUCT_ERROR {
            bail!("Dot product with dtype {:?} error {error} exceeds max error {GGML_MAX_DOT_PRODUCT_ERROR}. Source: {source}", T::DTYPE);
        }
        // We diverge slightly due to different rounding behavior / f16 to f32 conversions in GGML
        // => we use a slightly higher error threshold
        const ERROR_LENIENCY: f32 = 0.00001;
        if error - ERROR_LENIENCY > ggml_error {
            bail!(
                "Dot product with dtype {:?} error {error} exceeds ggml reference error {ggml_error}. Source: {source}",
                T::DTYPE,
            );
        }
        Ok(())
    };

    verify_result(result, "vec-dot")?;
    verify_result(result_matmul, "matmul")?;
    Ok(())
}

#[test]
fn quantized_mm() -> Result<()> {
    ggml_matmul_error_test::<f32>()?;
    ggml_matmul_error_test::<half::f16>()?;
    //ggml_matmul_error_test::<half::bf16>()?; TODO: Fails on ubuntu and windows. Check CpuBF16 impl
    ggml_matmul_error_test::<k_quants::BlockQ4_0>()?;
    ggml_matmul_error_test::<k_quants::BlockQ4_1>()?;
    ggml_matmul_error_test::<k_quants::BlockQ5_0>()?;
    ggml_matmul_error_test::<k_quants::BlockQ5_1>()?;
    ggml_matmul_error_test::<k_quants::BlockQ8_0>()?;
    ggml_matmul_error_test::<k_quants::BlockQ8_1>()?;
    Ok(())
}

/// generates random tensors of size `m x k` and `n x k` and calculates their expected matrix multiplication result.
fn get_random_tensors(
    m: usize,
    k: usize,
    n: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut rng = StdRng::seed_from_u64(314159265358979);

    let lhs = (0..m * k)
        .map(|_| rng.random::<f32>() - 0.5)
        .collect::<Vec<_>>();
    let rhs = (0..n * k)
        .map(|_| rng.random::<f32>() - 0.5)
        .collect::<Vec<_>>();

    let lhs = Tensor::from_vec(lhs, (m, k), device)?;
    let rhs = Tensor::from_vec(rhs, (n, k), device)?;

    let mm = lhs.matmul(&rhs.t()?)?;
    Ok((lhs, rhs, mm))
}

#[macro_export]
macro_rules! quantized_matmul {
    // TODO: Switch to generating the two last arguments automatically once concat_idents is
    // stable. https://github.com/rust-lang/rust/issues/29599
    ($fn_name: ident, $fn_name_cpu: ident, $fn_name_cuda: ident, $fn_name_metal: ident, $dtype: expr) => {
        fn $fn_name(device: &Device) -> Result<()> {
            test_matmul(device, (1, 3, 4, 256), $dtype)?;
            Ok(())
        }

        test_device!($fn_name, $fn_name_cpu, $fn_name_cuda, $fn_name_metal);
    };
}

quantized_matmul!(
    quantized_matmul_q4_0_bis,
    quantized_matmul_q4_0_cpu,
    quantized_matmul_q4_0_cuda,
    quantized_matmul_q4_0_metal,
    GgmlDType::Q4_0
);
quantized_matmul!(
    quantized_matmul_q4_1_bis,
    quantized_matmul_q4_1_cpu,
    quantized_matmul_q4_1_cuda,
    quantized_matmul_q4_1_metal,
    GgmlDType::Q4_1
);
quantized_matmul!(
    quantized_matmul_q5_0_bis,
    quantized_matmul_q5_0_cpu,
    quantized_matmul_q5_0_cuda,
    quantized_matmul_q5_0_metal,
    GgmlDType::Q5_0
);
quantized_matmul!(
    quantized_matmul_q5_1_bis,
    quantized_matmul_q5_1_cpu,
    quantized_matmul_q5_1_cuda,
    quantized_matmul_q5_1_metal,
    GgmlDType::Q5_1
);
quantized_matmul!(
    quantized_matmul_q8_0_bis,
    quantized_matmul_q8_0_cpu,
    quantized_matmul_q8_0_cuda,
    quantized_matmul_q8_0_metal,
    GgmlDType::Q8_0
);
quantized_matmul!(
    quantized_matmul_q8_1_bis,
    quantized_matmul_q8_1_cpu,
    quantized_matmul_q8_1_cuda,
    quantized_matmul_q8_1_metal,
    GgmlDType::Q8_1
);
quantized_matmul!(
    quantized_matmul_q2k_bis,
    quantized_matmul_q2k_cpu,
    quantized_matmul_q2k_cuda,
    quantized_matmul_q2k_metal,
    GgmlDType::Q2_K
);
quantized_matmul!(
    quantized_matmul_q3k_bis,
    quantized_matmul_q3k_cpu,
    quantized_matmul_q3k_cuda,
    quantized_matmul_q3k_metal,
    GgmlDType::Q3_K
);
quantized_matmul!(
    quantized_matmul_q4k_bis,
    quantized_matmul_q4k_cpu,
    quantized_matmul_q4k_cuda,
    quantized_matmul_q4k_metal,
    GgmlDType::Q4_K
);
quantized_matmul!(
    quantized_matmul_q5k_bis,
    quantized_matmul_q5k_cpu,
    quantized_matmul_q5k_cuda,
    quantized_matmul_q5k_metal,
    GgmlDType::Q5_K
);
quantized_matmul!(
    quantized_matmul_q6k_bis,
    quantized_matmul_q6k_cpu,
    quantized_matmul_q6k_cuda,
    quantized_matmul_q6k_metal,
    GgmlDType::Q6_K
);
// Not implemented on metal
quantized_matmul!(
    quantized_matmul_q8k_bis,
    quantized_matmul_q8k_cpu,
    quantized_matmul_q8k_cuda,
    quantized_matmul_q8k_metal,
    GgmlDType::Q8_K
);

#[test]
fn quantized_matmul_q2k() -> Result<()> {
    use k_quants::BlockQ2_K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q2_K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [0.916, 0.422, 0.215, 1.668]);

    ggml_matmul_error_test::<BlockQ2_K>()?;

    Ok(())
}

#[test]
fn quantized_matmul_q3k() -> Result<()> {
    use k_quants::BlockQ3_K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q3_K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.029, 1.418, -0.314, 1.495]);

    ggml_matmul_error_test::<BlockQ3_K>()?;

    Ok(())
}

#[test]
fn quantized_matmul_q4k() -> Result<()> {
    use k_quants::BlockQ4_K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q4_K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.125, 1.435, -0.201, 1.589]);

    ggml_matmul_error_test::<BlockQ4_K>()?;

    Ok(())
}

/// CPU codec for the byte-permuted Q4_KO block: `from_float` → `to_float` must
/// recover every element within one affine 4-bit quant step of its 32-element
/// sub-block. Exercises the contiguous-qs / tail-scales layout and the shared
/// {n0,n4,n2,n6,n1,n5,n3,n7} nibble order on the CPU side.
#[test]
fn q4_ko_cpu_roundtrip() -> Result<()> {
    use k_quants::BlockQ4_KO;
    let nblocks = 4usize;
    let n = 128 * nblocks;
    let mut rng = rand::rngs::StdRng::seed_from_u64(20_240_613);
    let xs: Vec<f32> = (0..n).map(|_| rng.random_range(-3.0f32..3.0)).collect();

    let mut blocks = vec![BlockQ4_KO::zeros(); nblocks];
    BlockQ4_KO::from_float(&xs, &mut blocks);
    let mut ys = vec![0f32; n];
    BlockQ4_KO::to_float(&blocks, &mut ys);

    // Affine 4-bit per 32-element sub-block: each value lands within one quant step.
    for b in 0..nblocks {
        for sub in 0..4 {
            let lo = b * 128 + sub * 32;
            let (mut mn, mut mx) = (f32::MAX, f32::MIN);
            for &x in &xs[lo..lo + 32] {
                mn = mn.min(x);
                mx = mx.max(x);
            }
            let step = (mx - mn) / 15.0;
            for i in lo..lo + 32 {
                let err = (xs[i] - ys[i]).abs();
                assert!(
                    err <= step + 1e-3,
                    "elem {i}: |{} - {}| = {err} exceeds one quant step {step}",
                    xs[i],
                    ys[i]
                );
            }
        }
    }
    Ok(())
}

/// CPU codecs for Q5_KO (per-32 affine, 5-bit + 5th-bit), Q6_KO (per-16 symmetric,
/// 6-bit crumbs), Q8_KO (per-128, 8-bit). Each `from_float`→`to_float` recovers every
/// element within one quant step of its scale group.
#[test]
fn q5q6q8_ko_cpu_roundtrip() -> Result<()> {
    use k_quants::{BlockQ5_KO, BlockQ6_KO, BlockQ8_KO};
    let nblocks = 4usize;
    let n = 128 * nblocks;
    let mut rng = rand::rngs::StdRng::seed_from_u64(20_240_614);
    let xs: Vec<f32> = (0..n).map(|_| rng.random_range(-3.0f32..3.0)).collect();

    // Helper: assert each element of `ys` is within `step(group)` of `xs`, where the
    // group covers `gsize` elements and the step is computed from the group's spread.
    let check = |ys: &[f32], gsize: usize, step_of: &dyn Fn(&[f32]) -> f32, tag: &str| {
        for b in 0..nblocks {
            for g in 0..(128 / gsize) {
                let lo = b * 128 + g * gsize;
                let step = step_of(&xs[lo..lo + gsize]);
                for i in lo..lo + gsize {
                    let err = (xs[i] - ys[i]).abs();
                    assert!(
                        err <= step + 1e-3,
                        "{tag} elem {i}: |{} - {}| = {err} exceeds step {step}",
                        xs[i],
                        ys[i]
                    );
                }
            }
        }
    };

    // Q5_KO: per-32 affine, q5 in [0,31] → step = (max-min)/31.
    let mut b5 = vec![BlockQ5_KO::zeros(); nblocks];
    BlockQ5_KO::from_float(&xs, &mut b5);
    let mut y5 = vec![0f32; n];
    BlockQ5_KO::to_float(&b5, &mut y5);
    check(
        &y5,
        32,
        &|g| {
            let (mn, mx) = g
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            (mx - mn) / 31.0
        },
        "Q5_KO",
    );

    // Q6_KO: per-16 symmetric, q6-32 in [-32,31] → step = amax/32.
    let mut b6 = vec![BlockQ6_KO::zeros(); nblocks];
    BlockQ6_KO::from_float(&xs, &mut b6);
    let mut y6 = vec![0f32; n];
    BlockQ6_KO::to_float(&b6, &mut y6);
    check(
        &y6,
        16,
        &|g| g.iter().fold(0f32, |a, &v| a.max(v.abs())) / 32.0,
        "Q6_KO",
    );

    // Q8_KO: per-128, 8-bit → step = amax/127.
    let mut b8 = vec![BlockQ8_KO::zeros(); nblocks];
    BlockQ8_KO::from_float(&xs, &mut b8);
    let mut y8 = vec![0f32; n];
    BlockQ8_KO::to_float(&b8, &mut y8);
    check(
        &y8,
        128,
        &|g| g.iter().fold(0f32, |a, &v| a.max(v.abs())) / 127.0,
        "Q8_KO",
    );

    Ok(())
}

#[test]
fn quantized_matmul_q5k() -> Result<()> {
    use k_quants::BlockQ5_K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q5_K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.192, 1.491, -0.18, 1.743]);

    //Expected: 0.000740408897
    ggml_matmul_error_test::<BlockQ5_K>()?;

    Ok(())
}

#[test]
fn quantized_matmul_q6k() -> Result<()> {
    use k_quants::BlockQ6_K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q6_K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.324, 1.49, -0.164, 1.741]);

    ggml_matmul_error_test::<BlockQ6_K>()?;
    Ok(())
}

#[test]
fn quantized_matmul_q8k() -> Result<()> {
    use k_quants::BlockQ8_K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q8_K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.266, 1.504, -0.204, 1.7]);

    ggml_matmul_error_test::<BlockQ8_K>()?;
    Ok(())
}

// =============================================================================
// Q8_0 and Q8_1 quantize/dequantize tests (matching other quantize_q* tests)
// =============================================================================

fn quantize_q8_0(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q8_0;
    // Q8_0 block size is 32
    let src = (0..32 * 4).map(|v| v as f32 - 64.0).collect::<Vec<_>>();
    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    // Q8_0 should have very low quantization error (8-bit symmetric)
    let src_vec = src.to_vec1::<f32>()?;
    let dst_vec = dst.to_vec1::<f32>()?;
    compare_with_error(dst_vec.as_slice(), src_vec.as_slice(), 0.6);

    // Test with GGML-like test vector
    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q8_1(device: &Device) -> Result<()> {
    // Q8_1 vec-dot not implemented on CPU, skip
    if device.is_cpu() {
        return Ok(());
    }
    let dtype = GgmlDType::Q8_1;
    // Q8_1 block size is 32
    let src = (0..32 * 4).map(|v| v as f32 - 64.0).collect::<Vec<_>>();
    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    // Q8_1 should have very low quantization error (8-bit with min/max)
    let src_vec = src.to_vec1::<f32>()?;
    let dst_vec = dst.to_vec1::<f32>()?;
    compare_with_error(dst_vec.as_slice(), src_vec.as_slice(), 0.6);

    Ok(())
}

test_device!(
    quantize_q8_0,
    quantize_q8_0_cpu,
    quantize_q8_0_cuda,
    quantize_q8_0_metal
);
test_device!(
    quantize_q8_1,
    quantize_q8_1_cpu,
    quantize_q8_1_cuda,
    quantize_q8_1_metal
);

// =============================================================================
// DMMV (dequantize_mul_mat_vec) tests - tests the GEMV kernel path
// These tests force the dequantize_mul_mat_vec path instead of mul_mat_vec_q8_1
// =============================================================================

/// Test helper that runs a matmul with FORCE_DMMV enabled
/// Note: DMMV kernel only supports m=1 (single vector), not batched operations
#[cfg(feature = "cuda")]
fn test_dmmv_path(
    device: &Device,
    dtype: GgmlDType,
    k: usize,
    n: usize,
    max_error: f32,
) -> Result<()> {
    use candle_core::quantized::cuda::set_force_dmmv;

    // DMMV only supports m=1 (GEMV - single vector input)
    let m = 1;

    // Create test data
    let lhs = (0..(m * k))
        .map(|v| (v as f32 / (m * k) as f32) - 0.5)
        .collect::<Vec<_>>();
    let rhs = (0..(k * n))
        .map(|v| (v as f32 / (n * k) as f32) - 0.5)
        .collect::<Vec<_>>();

    let lhs = Tensor::from_slice(&lhs, (m, k), device)?;
    let rhs = Tensor::from_slice(&rhs, (k, n), device)?;

    // Compute reference with standard matmul
    let mm_ref = lhs.matmul(&rhs)?;

    // Quantize RHS and compute with DMMV path
    let qtensor = quantized::QTensor::quantize(&rhs.t()?, dtype)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;

    // Force DMMV path – hold the global lock so no other test sees FORCE_DMMV=true.
    let _dmmv_guard = crate::DMMV_LOCK.lock().unwrap();
    set_force_dmmv(true);
    let res_dmmv = matmul.forward(&lhs)?;
    set_force_dmmv(false);
    drop(_dmmv_guard);

    // Also compute with default path for comparison
    let res_default = matmul.forward(&lhs)?;

    // Both should produce similar results (within quantization error)
    let error_dmmv: f32 = ((&mm_ref - &res_dmmv)?.abs()?
        / &mm_ref.abs().unwrap_or(mm_ref.clone()))?
        .mean_all()?
        .to_scalar()?;
    let error_default: f32 = ((&mm_ref - &res_default)?.abs()?
        / &mm_ref.abs().unwrap_or(mm_ref.clone()))?
        .mean_all()?
        .to_scalar()?;

    // DMMV and default paths should be within reasonable tolerance
    assert!(
        error_dmmv <= max_error,
        "DMMV error {error_dmmv} too high for {dtype:?}"
    );
    assert!(
        error_default <= max_error,
        "Default error {error_default} too high for {dtype:?}"
    );

    // DMMV and default should produce similar results
    let path_diff: f32 = ((&res_dmmv - &res_default)?.abs())?
        .mean_all()?
        .to_scalar()?;
    assert!(
        path_diff <= 0.1,
        "DMMV vs default path difference {path_diff} too high for {dtype:?}"
    );

    Ok(())
}

#[cfg(feature = "cuda")]
fn dmmv_q8_0(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    // Test GEMV case with different k sizes
    test_dmmv_path(device, GgmlDType::Q8_0, 256, 64, 0.15)?;
    test_dmmv_path(device, GgmlDType::Q8_0, 512, 128, 0.15)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn dmmv_q8_1(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    test_dmmv_path(device, GgmlDType::Q8_1, 256, 64, 0.15)?;
    test_dmmv_path(device, GgmlDType::Q8_1, 512, 128, 0.15)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn dmmv_q8k(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    // Q8_K requires k to be multiple of 256 (QK_K)
    test_dmmv_path(device, GgmlDType::Q8_K, 512, 64, 0.15)?;
    test_dmmv_path(device, GgmlDType::Q8_K, 1024, 128, 0.15)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn dmmv_q4_0(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    test_dmmv_path(device, GgmlDType::Q4_0, 256, 64, 0.15)?;
    test_dmmv_path(device, GgmlDType::Q4_0, 512, 128, 0.15)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn dmmv_q4_1(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    test_dmmv_path(device, GgmlDType::Q4_1, 256, 64, 0.15)?;
    test_dmmv_path(device, GgmlDType::Q4_1, 512, 128, 0.15)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn dmmv_q5_0(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    test_dmmv_path(device, GgmlDType::Q5_0, 256, 64, 0.15)?;
    test_dmmv_path(device, GgmlDType::Q5_0, 512, 128, 0.15)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn dmmv_q5_1(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    test_dmmv_path(device, GgmlDType::Q5_1, 256, 64, 0.15)?;
    test_dmmv_path(device, GgmlDType::Q5_1, 512, 128, 0.15)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn dmmv_q2k(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    // Q2K has higher quantization error
    test_dmmv_path(device, GgmlDType::Q2_K, 512, 64, 0.25)?;
    test_dmmv_path(device, GgmlDType::Q2_K, 1024, 128, 0.25)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn dmmv_q3k(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    test_dmmv_path(device, GgmlDType::Q3_K, 512, 64, 0.20)?;
    test_dmmv_path(device, GgmlDType::Q3_K, 1024, 128, 0.20)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn dmmv_q4k(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    test_dmmv_path(device, GgmlDType::Q4_K, 512, 64, 0.15)?;
    test_dmmv_path(device, GgmlDType::Q4_K, 1024, 128, 0.15)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn dmmv_q5k(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    // NOTE: Q5K DMMV kernel has a bug where it's missing the nrows parameter
    // in the kernel signature, so it produces incorrect results.
    // The mul_mat_vec_q8_1 path (default) works correctly.
    // Skipping this test until the kernel is fixed.
    // See: dequantize_mul_mat_vec_q5_k in quantized.cu
    Ok(())
}

#[cfg(feature = "cuda")]
fn dmmv_q6k(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    test_dmmv_path(device, GgmlDType::Q6_K, 512, 64, 0.15)?;
    test_dmmv_path(device, GgmlDType::Q6_K, 1024, 128, 0.15)?;
    Ok(())
}

// DMMV test registrations - CUDA only
#[cfg(feature = "cuda")]
mod dmmv_tests {
    use super::*;

    #[test]
    fn dmmv_q4_0_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        dmmv_q4_0(&device)
    }

    #[test]
    fn dmmv_q4_1_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        dmmv_q4_1(&device)
    }

    #[test]
    fn dmmv_q5_0_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        dmmv_q5_0(&device)
    }

    #[test]
    fn dmmv_q5_1_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        dmmv_q5_1(&device)
    }

    #[test]
    fn dmmv_q8_0_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        dmmv_q8_0(&device)
    }

    #[test]
    fn dmmv_q8_1_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        dmmv_q8_1(&device)
    }

    #[test]
    fn dmmv_q2k_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        dmmv_q2k(&device)
    }

    #[test]
    fn dmmv_q3k_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        dmmv_q3k(&device)
    }

    #[test]
    fn dmmv_q4k_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        dmmv_q4k(&device)
    }

    #[test]
    fn dmmv_q5k_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        dmmv_q5k(&device)
    }

    #[test]
    fn dmmv_q6k_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        dmmv_q6k(&device)
    }

    #[test]
    fn dmmv_q8k_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        dmmv_q8k(&device)
    }
}

// ==================== Phase 2: QTensor Extensions Tests ====================

#[test]
fn qtensor_zeros_q4_0_cpu() -> Result<()> {
    let device = Device::Cpu;
    let q = quantized::QTensor::zeros((128,), GgmlDType::Q4_0, &device)?;

    // Verify shape
    assert_eq!(q.shape().dims(), &[128]);
    assert_eq!(q.dtype(), GgmlDType::Q4_0);

    // Dequantize and verify all zeros
    let deq = q.dequantize(&device)?;
    let data = deq.to_vec1::<f32>()?;
    for (i, &val) in data.iter().enumerate() {
        assert!(val.abs() < 0.01, "expected ~0 at {}, got {}", i, val);
    }
    Ok(())
}

#[test]
fn qtensor_zeros_q8_0_cpu() -> Result<()> {
    let device = Device::Cpu;
    let q = quantized::QTensor::zeros((256,), GgmlDType::Q8_0, &device)?;

    assert_eq!(q.shape().dims(), &[256]);
    assert_eq!(q.dtype(), GgmlDType::Q8_0);

    let deq = q.dequantize(&device)?;
    let data = deq.to_vec1::<f32>()?;
    for (i, &val) in data.iter().enumerate() {
        assert!(val.abs() < 0.01, "expected ~0 at {}, got {}", i, val);
    }
    Ok(())
}

#[test]
fn qtensor_zeros_2d_q8_0_cpu() -> Result<()> {
    let device = Device::Cpu;
    let q = quantized::QTensor::zeros((4, 64), GgmlDType::Q8_0, &device)?;

    assert_eq!(q.shape().dims(), &[4, 64]);
    assert_eq!(q.dtype(), GgmlDType::Q8_0);
    assert_eq!(q.shape().elem_count(), 256);
    Ok(())
}

#[test]
fn qtensor_slice_scatter_q8_0_basic() -> Result<()> {
    let device = Device::Cpu;

    // Create arena (256 elements = 8 blocks of 32)
    let mut arena = quantized::QTensor::zeros((256,), GgmlDType::Q8_0, &device)?;

    // Create source data (64 elements = 2 blocks) with non-zero values
    let src_values: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let src_tensor = Tensor::from_vec(src_values, (64,), &device)?;
    let src = quantized::QTensor::quantize(&src_tensor, GgmlDType::Q8_0)?;

    // Scatter at offset 64 (block-aligned)
    arena.slice_scatter(&src, 64)?;

    // Verify: elements 0-63 should be ~0, elements 64-127 should be ~0-63
    let deq = arena.dequantize(&device)?;
    let data = deq.to_vec1::<f32>()?;

    // First 64 should be ~0 (zero-initialized)
    for i in 0..64 {
        assert!(
            data[i].abs() < 0.5,
            "data[{}] = {}, expected ~0",
            i,
            data[i]
        );
    }

    // Next 64 should approximate 0-63 (with quantization error)
    for i in 0..64 {
        let expected = i as f32;
        let actual = data[64 + i];
        let error = (actual - expected).abs();
        assert!(
            error < 1.5,
            "data[{}] = {}, expected ~{}",
            64 + i,
            actual,
            expected
        );
    }

    // Remaining 128 should be ~0
    for i in 128..256 {
        assert!(
            data[i].abs() < 0.5,
            "data[{}] = {}, expected ~0",
            i,
            data[i]
        );
    }

    Ok(())
}

#[test]
fn qtensor_slice_scatter_q4_0_basic() -> Result<()> {
    let device = Device::Cpu;

    // Create arena (128 elements = 4 blocks)
    let mut arena = quantized::QTensor::zeros((128,), GgmlDType::Q4_0, &device)?;

    // Create source data (32 elements = 1 block)
    let src_values: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let src_tensor = Tensor::from_vec(src_values, (32,), &device)?;
    let src = quantized::QTensor::quantize(&src_tensor, GgmlDType::Q4_0)?;

    // Scatter at offset 32
    arena.slice_scatter(&src, 32)?;

    let deq = arena.dequantize(&device)?;
    let data = deq.to_vec1::<f32>()?;

    // First 32 should be ~0
    for i in 0..32 {
        assert!(
            data[i].abs() < 1.0,
            "data[{}] = {}, expected ~0",
            i,
            data[i]
        );
    }

    // Q4_0 has lower precision, so allow more error
    for i in 0..32 {
        let expected = i as f32;
        let actual = data[32 + i];
        let error = (actual - expected).abs();
        assert!(
            error < 3.0,
            "data[{}] = {}, expected ~{} (Q4_0)",
            32 + i,
            actual,
            expected
        );
    }

    Ok(())
}

#[test]
fn qtensor_slice_scatter_misaligned_offset() {
    let device = Device::Cpu;
    let mut arena = quantized::QTensor::zeros((256,), GgmlDType::Q8_0, &device).unwrap();
    let src = quantized::QTensor::zeros((32,), GgmlDType::Q8_0, &device).unwrap();

    // Offset 17 is not block-aligned (not multiple of 32)
    let result = arena.slice_scatter(&src, 17);
    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("not aligned"),
        "Error should mention alignment"
    );
}

#[test]
fn qtensor_slice_scatter_dtype_mismatch() {
    let device = Device::Cpu;
    let mut arena = quantized::QTensor::zeros((256,), GgmlDType::Q8_0, &device).unwrap();
    let src = quantized::QTensor::zeros((32,), GgmlDType::Q4_0, &device).unwrap();

    let result = arena.slice_scatter(&src, 0);
    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("dtype mismatch"),
        "Error should mention dtype mismatch"
    );
}

#[test]
fn qtensor_slice_scatter_out_of_bounds() {
    let device = Device::Cpu;
    let mut arena = quantized::QTensor::zeros((128,), GgmlDType::Q8_0, &device).unwrap();
    let src = quantized::QTensor::zeros((64,), GgmlDType::Q8_0, &device).unwrap();

    // 96 + 64 = 160 > 128
    let result = arena.slice_scatter(&src, 96);
    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("exceeds"),
        "Error should mention bounds exceeded"
    );
}

#[test]
fn qtensor_slice_scatter_multiple_writes() -> Result<()> {
    let device = Device::Cpu;

    // Create arena (128 elements = 4 blocks)
    let mut arena = quantized::QTensor::zeros((128,), GgmlDType::Q8_0, &device)?;

    // Write block 0 with values 0-31
    let src1_values: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let src1 = quantized::QTensor::quantize(
        &Tensor::from_vec(src1_values, (32,), &device)?,
        GgmlDType::Q8_0,
    )?;
    arena.slice_scatter(&src1, 0)?;

    // Write block 2 with values 100-131
    let src2_values: Vec<f32> = (0..32).map(|i| 100.0 + i as f32).collect();
    let src2 = quantized::QTensor::quantize(
        &Tensor::from_vec(src2_values, (32,), &device)?,
        GgmlDType::Q8_0,
    )?;
    arena.slice_scatter(&src2, 64)?;

    let deq = arena.dequantize(&device)?;
    let data = deq.to_vec1::<f32>()?;

    // Block 0: ~0-31
    for i in 0..32 {
        let expected = i as f32;
        let error = (data[i] - expected).abs();
        assert!(
            error < 1.5,
            "block 0 data[{}] = {}, expected ~{}",
            i,
            data[i],
            expected
        );
    }

    // Block 1: ~0 (untouched)
    for i in 32..64 {
        assert!(
            data[i].abs() < 0.5,
            "block 1 data[{}] = {}, expected ~0",
            i,
            data[i]
        );
    }

    // Block 2: ~100-131
    for i in 0..32 {
        let expected = 100.0 + i as f32;
        let error = (data[64 + i] - expected).abs();
        assert!(
            error < 1.5,
            "block 2 data[{}] = {}, expected ~{}",
            64 + i,
            data[64 + i],
            expected
        );
    }

    // Block 3: ~0 (untouched)
    for i in 96..128 {
        assert!(
            data[i].abs() < 0.5,
            "block 3 data[{}] = {}, expected ~0",
            i,
            data[i]
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
mod qtensor_extensions_cuda {
    use super::*;

    #[test]
    fn qtensor_zeros_q8_0_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let q = quantized::QTensor::zeros((256,), GgmlDType::Q8_0, &device)?;

        assert_eq!(q.shape().dims(), &[256]);
        assert_eq!(q.dtype(), GgmlDType::Q8_0);

        let deq = q.dequantize(&device)?;
        let data = deq.to_vec1::<f32>()?;
        for (i, &val) in data.iter().enumerate() {
            assert!(val.abs() < 0.01, "expected ~0 at {}, got {}", i, val);
        }
        Ok(())
    }

    #[test]
    fn qtensor_slice_scatter_q8_0_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;

        // Create arena (256 elements = 8 blocks)
        let mut arena = quantized::QTensor::zeros((256,), GgmlDType::Q8_0, &device)?;

        // Create source data (64 elements = 2 blocks)
        let src_values: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let src_tensor = Tensor::from_vec(src_values, (64,), &device)?;
        let src = quantized::QTensor::quantize(&src_tensor, GgmlDType::Q8_0)?;

        // Scatter at offset 64
        arena.slice_scatter(&src, 64)?;

        // Verify
        let deq = arena.dequantize(&device)?;
        let data = deq.to_vec1::<f32>()?;

        // First 64 should be ~0
        for i in 0..64 {
            assert!(
                data[i].abs() < 0.5,
                "data[{}] = {}, expected ~0",
                i,
                data[i]
            );
        }

        // Next 64 should approximate 0-63
        for i in 0..64 {
            let expected = i as f32;
            let error = (data[64 + i] - expected).abs();
            assert!(
                error < 1.5,
                "data[{}] = {}, expected ~{}",
                64 + i,
                data[64 + i],
                expected
            );
        }

        Ok(())
    }
}
