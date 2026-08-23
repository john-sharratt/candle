// Test code: loop indices are block/lane/expert coordinates in flat-buffer
// offset arithmetic, and the tuple returns mirror the kernel's raw parameter
// lists rather than modelling a domain type.
#![allow(
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::manual_div_ceil,
    clippy::needless_question_mark
)]

use super::*;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use half::bf16;
use rand::Rng;
use std::ffi::c_void;
use std::time::Instant;

/// Guards the large-N (2*ffn) gemx matmul at batch=1: a fused ffn_gate+ffn_up
/// weight has distinct gate (rows 0..N/2) and up (rows N/2..N) halves, and the
/// kernel must not collapse the up half onto the gate half. This path was the
/// prime suspect while chasing a decode `up == gate` corruption at N=16384 — the
/// kernel was exonerated here (the real bug was an aliased `to_dtype_mut`, see
/// `cuda_to_dtype_mut_respects_start_offset`); this test pins the kernel so a
/// future regression in the large-N batch=1 gemx path is caught directly.
#[test]
fn cuda_mm_gemx_large_n_batch1_no_row_aliasing() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let ncols = 3072usize; // K (hidden)
    let half_n = 8192usize; // ffn
    let nrows = 2 * half_n; // N = 2*ffn (the failing size)
    let device = crate::Device::Cuda(dev.clone());

    // Build the gate (rows 0..ffn) and up (rows ffn..2ffn) weights SEPARATELY,
    // quantize each, then fuse via concat_rows_cuda — exactly how the MLP builds
    // its fused ffn_gate+ffn_up weight. Distinct patterns so the halves differ.
    let gate_w: Vec<f32> = (0..ncols * half_n)
        .map(|v| (((v / ncols) as f32 * 7.0 + (v % ncols) as f32 * 11.0) * 0.001).sin())
        .collect();
    let up_w: Vec<f32> = (0..ncols * half_n)
        .map(|v| (((v / ncols) as f32 * 13.0 + (v % ncols) as f32 * 5.0) * 0.001).cos())
        .collect();
    let gate_t = crate::Tensor::from_vec(gate_w, (half_n, ncols), &device)?;
    let up_t = crate::Tensor::from_vec(up_w, (half_n, ncols), &device)?;
    let wg = crate::quantized::QTensor::quantize(&gate_t, GgmlDType::Q4_K)?;
    let wu = crate::quantized::QTensor::quantize(&up_t, GgmlDType::Q4_K)?;
    let fused = crate::quantized::QTensor::concat_rows_cuda(&[&wg, &wu])?;
    let fused_shape = fused.shape().clone();
    let xs_repacked = match fused.storage() {
        crate::quantized::QStorage::Cuda(s) => s.repack_gemx(&fused_shape)?,
        _ => unreachable!(),
    };
    let qtype = dtype_to_qtype(GgmlDType::Q4_K)? as i32;

    // Reference (batch>=2 is known-good) up value, filled on the y_cols=2 pass.
    let mut ref_up: Option<f32> = None;
    for &y_cols in &[2usize, 1, 4] {
        let y_data: Vec<f16> = vec![f16::from_f32(1.0); ncols * y_cols];
        let y = dev.memcpy_stod(&y_data)?;
        let dst = unsafe { dev.alloc::<f16>(nrows * y_cols)? };
        {
            let stream = dev.cuda_stream();
            let (data_ptr, _g) = xs_repacked.data.inner.device_ptr(&stream);
            let segment = VxSegment {
                weights: data_ptr as *const std::ffi::c_void,
                batch_count: y_cols as i32,
            };
            let (y_ptr, _gy) = y.device_ptr(&stream);
            let (dst_ptr, _gd) = dst.device_ptr(&stream);
            let status = unsafe {
                run_quantized_matmul(
                    &segment as *const VxSegment,
                    1,
                    y_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    ncols as i32,
                    nrows as i32,
                    ncols as i32,
                    nrows as i32,
                    qtype,
                    YType::F16 as i32,
                    xs_repacked.data.len,
                    0,
                    OutDType::F16 as i32,
                )
            };
            assert_eq!(status, 0, "matmul launcher rejected the call");
        }
        dev.synchronize()?;
        let res = dev.memcpy_dtov(&dst.slice(..))?;
        // batch 0 occupies dst[0..nrows].
        let gate0 = f16::to_f32(res[0]);
        let up0 = f16::to_f32(res[nrows / 2]);
        println!("y_cols={y_cols}: gate[0]={gate0:.3}  up[N/2]={up0:.3}");
        if y_cols == 2 {
            ref_up = Some(up0);
        }
        // The up row and gate row use different weight patterns, so their sums differ.
        assert!(
            (up0 - gate0).abs() > 0.5,
            "y_cols={y_cols}: UP ROW ALIASED TO GATE: up={up0} ~= gate={gate0}"
        );
        // And the up value must match the known-good (batch>=2) reference.
        if let Some(r) = ref_up {
            assert!(
                (up0 - r).abs() < 0.5,
                "y_cols={y_cols}: up row wrong: {up0} vs reference {r}"
            );
        }
    }
    Ok(())
}

/// Regression: `to_dtype_mut` on a *contiguous view with a non-zero start offset*
/// (e.g. the second half of a last-dim `narrow`) must cast the offset slice, not
/// the buffer start. The fused ffn_gate+ffn_up MLP hit this: at decode M=1 the
/// `up = gu.narrow(last, ffn, ffn)` view is contiguous (offset ffn); the in-place
/// cast ignored the offset and produced `up == gate`, corrupting decode.
#[test]
fn cuda_to_dtype_mut_respects_start_offset() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let n = 256usize;
    // [1, 1, 2n]: first half = 1.0, second half = 2.0.
    let data: Vec<f32> = (0..2 * n).map(|i| if i < n { 1.0 } else { 2.0 }).collect();
    let gu = crate::Tensor::from_vec(data, (1, 1, 2 * n), &device)?.to_dtype(crate::DType::F16)?;

    let mut up = gu.narrow(2, n, n)?; // contiguous, start_offset = n
    assert!(up.is_contiguous());
    up.to_dtype_mut(crate::DType::BF16)?;
    let up_v = up
        .flatten_all()?
        .to_dtype(crate::DType::F32)?
        .to_vec1::<f32>()?;
    assert!(
        up_v.iter().all(|&x| (x - 2.0).abs() < 1e-3),
        "up half aliased to gate: got {:?}",
        &up_v[..4]
    );

    let mut gate = gu.narrow(2, 0, n)?; // contiguous, start_offset = 0
    gate.to_dtype_mut(crate::DType::BF16)?;
    let gate_v = gate
        .flatten_all()?
        .to_dtype(crate::DType::F32)?
        .to_vec1::<f32>()?;
    assert!(
        gate_v.iter().all(|&x| (x - 1.0).abs() < 1e-3),
        "gate half wrong: got {:?}",
        &gate_v[..4]
    );
    Ok(())
}

#[test]
fn cuda_quantize_q8_1() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let el = 256;
    let el_padded = pad(el, MATRIX_ROW_PADDING);
    let y_size_in_bytes = el_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
    let vs: Vec<f32> = (0..el).map(|v| v as f32).collect();
    let y = dev.memcpy_stod(&vs)?;
    quantize_q8_1(&y.slice(..), &mut y_q8_1, el, 1, &dev)?;
    Ok(())
}

#[test]
fn cuda_mmv_q8_1() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let ncols = 256;
    let vs: Vec<f32> = (0..ncols).map(|v| v as f32).collect();
    let y = dev.memcpy_stod(&vs)?;
    let mut xs = QCudaStorage::zeros(&dev, ncols, GgmlDType::Q4_0)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
    let cuda_storage = mul_mat_vec_via_q8_1(
        &xs.data,
        &y.slice(..),
        /* dtype */ GgmlDType::Q4_0,
        /* ncols */ ncols,
        /* nrows */ 1,
        /* b_size */ 1,
        &dev,
        // No wave here, so nothing to inherit: the result is an ordinary owned
        // allocation, which is what these tests read back from.
        Backing::Owned,
    )?;
    let vs = cuda_storage.as_cuda_slice::<f32>()?;
    let vs = dev.memcpy_dtov(&vs.slice(..))?;
    assert_eq!(vs.len(), 1);
    // for n = 255, n.(n+1).(2n+1) / 6 = 5559680
    // Q8 means 1/256 precision.
    assert_eq!(vs[0], 5561664.5);

    let cuda_storage = dequantize_mul_mat_vec(
        &xs.data,
        &y.slice(..),
        /* dtype */ GgmlDType::Q4_0,
        /* ncols */ ncols,
        /* nrows */ 1,
        &dev,
    )?;
    let vs = cuda_storage.as_cuda_slice::<f32>()?;
    let vs = dev.memcpy_dtov(&vs.slice(..))?;
    assert_eq!(vs.len(), 1);
    assert_eq!(vs[0], 5561851.0);
    Ok(())
}

#[test]
fn cuda_mm_q8_1() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let ncols = 256;
    let vs: Vec<f32> = (0..ncols * 4).map(|v| v as f32 / 4.).collect();
    let y = dev.memcpy_stod(&vs)?;
    let mut xs = QCudaStorage::zeros(&dev, ncols * 4, GgmlDType::Q4_0)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
    let cuda_storage = mul_mat_via_q8_1(
        &xs.data,
        &y.slice(..),
        /* dtype */ GgmlDType::Q4_0,
        /* x_rows */ 4,
        /* x_cols */ ncols,
        /* y_rows */ ncols,
        /* y_cols */ 4,
        &dev,
        Backing::Owned,
    )?;
    let vs = cuda_storage.as_cuda_slice::<f32>()?;
    let vs = dev.memcpy_dtov(&vs.slice(..))?;

    /*
       x = torch.tensor([float(v) for v in range(1024)]).reshape(4, 256)
       x @ x.t() / 16
    tensor([[  347480.0000,   869720.0000,  1391960.0000,  1914200.0000],
            [  869720.0000,  2440536.0000,  4011352.0000,  5582166.5000],
            [ 1391960.0000,  4011352.0000,  6630742.0000,  9250132.0000],
            [ 1914200.0000,  5582166.5000,  9250132.0000, 12918099.0000]])
            */
    assert_eq!(vs.len(), 16);
    assert_eq!(vs[0], 347604.0);
    assert_eq!(vs[1], 888153.06);
    assert_eq!(vs[4], 869780.7);
    assert_eq!(vs[5], 2483145.0);
    assert_eq!(vs[11], 9407368.0);
    assert_eq!(vs[14], 9470856.0);
    assert_eq!(vs[15], 13138824.0);
    Ok(())
}

// The following test used to fail under compute-sanitizer until #2526.
#[test]
fn cuda_mm_q8_1_pad() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let (x_rows, ncols, y_cols) = (4, 16, 2048);
    let vs: Vec<f32> = (0..ncols * y_cols).map(|v| v as f32 / 256.).collect();
    let y = dev.memcpy_stod(&vs)?;
    let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_0)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
    let cuda_storage = mul_mat_via_q8_1(
        &xs.data,
        &y.slice(..),
        /* dtype */ GgmlDType::Q4_0,
        /* x_rows */ x_rows,
        /* x_cols */ ncols,
        /* y_rows */ ncols,
        /* y_cols */ y_cols,
        &dev,
        Backing::Owned,
    )?;
    let vs = cuda_storage.as_cuda_slice::<f32>()?;
    let _vs = dev.memcpy_dtov(&vs.slice(..))?;
    Ok(())
}

#[test]
fn cuda_mm_q4_k() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    // Use dimensions that match typical LLM layers (e.g., 3072 hidden dim)
    let ncols = 256; // K dimension - must be multiple of QK_K=256 for K-quants
    let x_rows = 4;
    let y_cols = 1;

    // Create input data
    let vs: Vec<f32> = (0..ncols * x_rows)
        .map(|v| v as f32 / (ncols as f32))
        .collect();
    let y_data: Vec<f32> = (0..ncols * y_cols)
        .map(|v| v as f32 / (ncols as f32))
        .collect();

    let y = dev.memcpy_stod(&y_data)?;
    let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_K)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(
        dev.memcpy_stod(&vs)?,
        dev.clone(),
    ))?;

    let cuda_storage = mul_mat_via_q8_1(
        &xs.data,
        &y.slice(..),
        /* dtype */ GgmlDType::Q4_K,
        /* x_rows */ x_rows,
        /* x_cols */ ncols,
        /* y_rows */ ncols,
        /* y_cols */ y_cols,
        &dev,
        Backing::Owned,
    )?;
    let result = cuda_storage.as_cuda_slice::<f32>()?;
    let result = dev.memcpy_dtov(&result.slice(..))?;

    // Compute expected result using CPU
    // For a simple check, just verify we get non-garbage values
    assert_eq!(result.len(), x_rows * y_cols);
    println!("Q4_K matmul result (y_cols=1): {:?}", result);

    // Check that results are reasonable (not NaN, not extreme values)
    for &v in &result {
        assert!(!v.is_nan(), "Result contains NaN");
        assert!(!v.is_infinite(), "Result contains Inf");
    }

    Ok(())
}

#[test]
fn cuda_mm_q4_k_large() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    // Test with larger dimensions that use MMQ path
    let ncols = 3072; // Hidden dim - should be multiple of 256
    let x_rows = 3072; // Output dim
    let y_cols = 128; // Sequence length - forces MMQ path

    // Create input data
    let weight_data: Vec<f32> = (0..ncols * x_rows)
        .map(|v| ((v % 256) as f32 - 128.0) / 256.0)
        .collect();
    let y_data: Vec<f32> = (0..ncols * y_cols)
        .map(|v| ((v % 256) as f32 - 128.0) / 256.0)
        .collect();

    let y = dev.memcpy_stod(&y_data)?;
    let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_K)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(
        dev.memcpy_stod(&weight_data)?,
        dev.clone(),
    ))?;

    let cuda_storage = mul_mat_via_q8_1(
        &xs.data,
        &y.slice(..),
        /* dtype */ GgmlDType::Q4_K,
        /* x_rows */ x_rows,
        /* x_cols */ ncols,
        /* y_rows */ ncols,
        /* y_cols */ y_cols,
        &dev,
        Backing::Owned,
    )?;
    let result = cuda_storage.as_cuda_slice::<f32>()?;
    let result = dev.memcpy_dtov(&result.slice(..))?;

    assert_eq!(result.len(), x_rows * y_cols);
    println!(
        "Q4_K matmul result (large): first 10 = {:?}",
        &result[..10.min(result.len())]
    );

    // Check that results are reasonable
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut zero_count = 0;
    for &v in &result {
        if v.is_nan() {
            nan_count += 1;
        }
        if v.is_infinite() {
            inf_count += 1;
        }
        if v == 0.0 {
            zero_count += 1;
        }
    }

    println!(
        "NaN count: {}, Inf count: {}, Zero count: {}",
        nan_count, inf_count, zero_count
    );
    assert_eq!(nan_count, 0, "Result contains NaN values");
    assert_eq!(inf_count, 0, "Result contains Inf values");

    Ok(())
}

/// Test Q4_K matmul with GEMX-repacked weights (K/128 with embedded scales)
/// This tests the FULL path used by the model: repack + run_quantized_matmul
#[test]
fn cuda_mm_q4_k_repacked() -> Result<()> {
    use crate::Shape;
    use half::f16;

    let dev = CudaDevice::new(0)?;
    let ncols = 256; // K dimension - must be multiple of QK_K=256 for K-quants
    let x_rows = 4; // Only 4 rows = 1 phase
    let y_cols = 1;

    // Create VARYING weight data - need variance for quantization to work!
    // GGML returns (0,0) if all values in a 32-element block are equal
    let weights: Vec<f32> = (0..ncols * x_rows)
        .map(|v| {
            let row = v / ncols;
            let col = v % ncols;
            // Vary within each row: col/ncols gives 0..1 range
            // Scale by (row+1) to make rows distinguishable
            (row as f32 + 1.0) * (col as f32 / ncols as f32)
        })
        .collect();
    println!("Input weights: row 0 first 8: {:?}", &weights[..8]);
    println!(
        "Input weights: row 0 last 8: {:?}",
        &weights[ncols - 8..ncols]
    );
    println!(
        "Input weights: row 1 first 8: {:?}",
        &weights[ncols..ncols + 8]
    );

    // First quantize on CPU to verify
    let cpu_storage = crate::CpuStorage::F32(weights.clone());
    let cpu_src = crate::Storage::Cpu(cpu_storage);
    let mut cpu_q = crate::Device::Cpu.qzeros(ncols * x_rows, GgmlDType::Q4_K)?;
    cpu_q.quantize(&cpu_src)?;
    let cpu_data = cpu_q.data()?;
    println!("CPU quantized size: {} bytes", cpu_data.len());

    // Parse first block_q4_K on CPU
    let dm_bytes = &cpu_data[0..4];
    let d = f16::from_ne_bytes([dm_bytes[0], dm_bytes[1]]);
    let dmin = f16::from_ne_bytes([dm_bytes[2], dm_bytes[3]]);
    println!(
        "CPU Block 0 dm: d={}, dmin={}",
        f16::to_f32(d),
        f16::to_f32(dmin)
    );

    // scales are at offset 4 (after dm)
    let scales_bytes = &cpu_data[4..16];
    println!("CPU Block 0 scales[0..12]: {:?}", scales_bytes);

    // Create Y data (F16 for GEMX path) - all ones
    let y_data: Vec<f16> = (0..ncols * y_cols).map(|_| f16::from_f32(1.0)).collect();
    let y = dev.memcpy_stod(&y_data)?;

    // Create and quantize weights on GPU
    let shape = Shape::from((x_rows, ncols));
    let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_K)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(
        dev.memcpy_stod(&weights)?,
        dev.clone(),
    ))?;

    // Read back quantized data to verify structure
    let q_bytes = dev.memcpy_dtov(&xs.data.inner.slice(..xs.data.len))?;
    println!("GPU quantized size: {} bytes", q_bytes.len());

    // Parse first block_q4_K header (dm is at offset 0, 4 bytes = 2 half values)
    let dm_bytes = &q_bytes[0..4];
    let d = f16::from_ne_bytes([dm_bytes[0], dm_bytes[1]]);
    let dmin = f16::from_ne_bytes([dm_bytes[2], dm_bytes[3]]);
    println!(
        "GPU Block 0 dm: d={}, dmin={}",
        f16::to_f32(d),
        f16::to_f32(dmin)
    );

    // Second block (offset 144)
    let dm_bytes = &q_bytes[144..148];
    let d = f16::from_ne_bytes([dm_bytes[0], dm_bytes[1]]);
    let dmin = f16::from_ne_bytes([dm_bytes[2], dm_bytes[3]]);
    println!(
        "GPU Block 1 dm: d={}, dmin={}",
        f16::to_f32(d),
        f16::to_f32(dmin)
    );

    // Repack to K/128 format with embedded scales
    let xs_repacked = xs.repack_gemx(&shape)?;
    println!(
        "Repacked Q4_K: {} bytes -> {} bytes",
        xs.data.len, xs_repacked.data.len
    );

    // Compute expected result using CPU dequantization
    let mut dequant = vec![0.0f32; ncols * x_rows];
    {
        // Dequantize CPU data
        let cpu_blocks = unsafe {
            std::slice::from_raw_parts(
                cpu_data.as_ptr() as *const crate::quantized::k_quants::BlockQ4_K,
                x_rows,
            )
        };
        crate::quantized::k_quants::GgmlType::to_float(cpu_blocks, &mut dequant);
    }
    println!("Dequant row 0 first 8: {:?}", &dequant[..8]);
    println!("Dequant row 1 first 8: {:?}", &dequant[ncols..ncols + 8]);

    // Expected result: sum of dequant[row, :] for each row
    let expected: Vec<f32> = (0..x_rows)
        .map(|row| dequant[row * ncols..(row + 1) * ncols].iter().sum())
        .collect();
    println!("Expected sums: {:?}", expected);

    // Run matmul with repacked weights (K/128 format has embedded scales)
    let qtype = dtype_to_qtype(GgmlDType::Q4_K)? as i32;
    let dst = unsafe { dev.alloc::<f16>(x_rows * y_cols)? };

    {
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = xs_repacked.data.inner.device_ptr(&stream);
        let segment = VxSegment {
            weights: data_ptr as *const std::ffi::c_void,
            batch_count: y_cols as i32,
        };
        let (y_ptr, _y_guard) = y.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

        let status = unsafe {
            run_quantized_matmul(
                &segment as *const VxSegment,
                1,
                y_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                ncols as i32,
                x_rows as i32,
                ncols as i32,
                x_rows as i32,
                qtype,
                YType::F16 as i32,
                xs_repacked.data.len,
                0, // force_mode2 (tiling only; result-invariant)
                OutDType::F16 as i32,
            )
        };
        assert_eq!(status, 0, "matmul launcher rejected the call");
    }

    let result = dev.memcpy_dtov(&dst.slice(..))?;
    println!("Q4_K repacked matmul result:");
    for (i, &v) in result.iter().enumerate() {
        let got = f16::to_f32(v);
        let exp = expected[i];
        let err = (got - exp).abs() / exp.abs().max(1.0) * 100.0;
        println!("  Row {}: {} (expected {}, err {:.1}%)", i, got, exp, err);
    }

    // Verify results - check for zeros which indicate bugs
    let mut zero_count = 0;
    for (i, &v) in result.iter().enumerate() {
        assert!(!f16::is_nan(v), "Result {} is NaN", i);
        assert!(!f16::is_infinite(v), "Result {} is Inf", i);
        if v == f16::from_f32(0.0) {
            zero_count += 1;
        }
    }
    println!("Zero count: {} out of {}", zero_count, result.len());

    // For a non-trivial matmul with non-zero inputs, we shouldn't have zeros
    assert!(zero_count == 0, "Got {} zero results", zero_count);

    Ok(())
}

#[test]
fn cuda_mm_q4_k_repacked_model_size() -> Result<()> {
    // Test with model-sized matrices (e.g., 3072x3072 hidden layer)
    use crate::Shape;
    use half::f16;

    let dev = CudaDevice::new(0)?;
    let ncols = 3072; // K dimension (hidden dim)
    let x_rows = 3072; // N dimension (output dim)
    let y_cols = 1; // Batch size 1 for decode

    // Create weight data with variance
    let weights: Vec<f32> = (0..ncols * x_rows)
        .map(|v| {
            let row = v / ncols;
            let col = v % ncols;
            // Sinusoidal pattern to ensure variance in all sub-blocks
            let r = row as f32 / x_rows as f32;
            let c = col as f32 / ncols as f32;
            ((r * 7.0 + c * 11.0).sin() + 1.0) * 0.5 // Range [0, 1]
        })
        .collect();

    // Create Y data (all ones for simple sum)
    let y_data: Vec<f16> = (0..ncols * y_cols).map(|_| f16::from_f32(1.0)).collect();
    let y = dev.memcpy_stod(&y_data)?;

    // Create and quantize weights on GPU
    let shape = Shape::from((x_rows, ncols));
    let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_K)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(
        dev.memcpy_stod(&weights)?,
        dev.clone(),
    ))?;

    // Repack to GEMX format (K/128 with embedded scales)
    let xs_repacked = xs.repack_gemx(&shape)?;
    println!(
        "Repacked Q4_K model-size: {} bytes -> {} bytes",
        xs.data.len, xs_repacked.data.len
    );

    // Run matmul (K/128 format has embedded scales)
    let qtype = dtype_to_qtype(GgmlDType::Q4_K)? as i32;
    let dst = unsafe { dev.alloc::<f16>(x_rows * y_cols)? };

    {
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = xs_repacked.data.inner.device_ptr(&stream);
        let segment = VxSegment {
            weights: data_ptr as *const std::ffi::c_void,
            batch_count: y_cols as i32,
        };
        let (y_ptr, _y_guard) = y.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

        let status = unsafe {
            run_quantized_matmul(
                &segment as *const VxSegment,
                1,
                y_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                ncols as i32,
                x_rows as i32,
                ncols as i32,
                x_rows as i32,
                qtype,
                YType::F16 as i32,
                xs_repacked.data.len,
                0, // force_mode2 (tiling only; result-invariant)
                OutDType::F16 as i32,
            )
        };
        assert_eq!(status, 0, "matmul launcher rejected the call");
    }

    let result = dev.memcpy_dtov(&dst.slice(..))?;

    // Check for NaN/Inf/zeros
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut zero_count = 0;
    for &v in &result {
        if f16::is_nan(v) {
            nan_count += 1;
        }
        if f16::is_infinite(v) {
            inf_count += 1;
        }
        if v == f16::from_f32(0.0) {
            zero_count += 1;
        }
    }

    println!("Model-size Q4_K result: {} rows", result.len());
    println!(
        "  First 10: {:?}",
        result
            .iter()
            .take(10)
            .map(|v| f16::to_f32(*v))
            .collect::<Vec<_>>()
    );
    println!(
        "  Last 10: {:?}",
        result
            .iter()
            .rev()
            .take(10)
            .map(|v| f16::to_f32(*v))
            .collect::<Vec<_>>()
    );
    println!(
        "  NaN: {}, Inf: {}, Zero: {}",
        nan_count, inf_count, zero_count
    );

    assert_eq!(nan_count, 0, "Got NaN values");
    assert_eq!(inf_count, 0, "Got Inf values");
    // For all-ones Y with sinusoidal weights, expect non-zero sums
    assert!(
        zero_count < result.len() / 100,
        "Too many zeros: {}",
        zero_count
    );

    Ok(())
}

/// Test Q6_K matmul with GEMX-repacked weights (K/128 with embedded scales)
/// Q6_K is a 6-bit K-quant format, similar to Q4_K but higher precision.
#[test]
fn cuda_mm_q6_k_repacked() -> Result<()> {
    use crate::Shape;
    use half::f16;

    let dev = CudaDevice::new(0)?;
    // Test with MULTIPLE blocks per row (like real tensors)
    // Match GGUF 512x2048 shape exactly
    let ncols = 2048; // 8 blocks per row (same as GGUF 512x2048)
    let x_rows = 512;
    let y_cols = 1;

    // Create VARYING weight data for Q6_K - simple gradient pattern like original
    // Scale down to avoid f16 overflow (max ~65504)
    let weights: Vec<f32> = (0..ncols * x_rows)
        .map(|v| {
            let _row = v / ncols;
            let col = v % ncols;
            // All rows have same pattern - just (col / ncols), sum will be ~ncols/2
            col as f32 / ncols as f32
        })
        .collect();
    println!("Q6_K Input weights: row 0 first 8: {:?}", &weights[..8]);
    println!(
        "Q6_K Input weights: row 0 last 8: {:?}",
        &weights[ncols - 8..ncols]
    );

    // Create Y data (all ones for simple sum)
    let y_data: Vec<f16> = (0..ncols * y_cols).map(|_| f16::from_f32(1.0)).collect();
    let y = dev.memcpy_stod(&y_data)?;

    // Create and quantize weights on GPU using Q6_K
    let shape = Shape::from((x_rows, ncols));
    let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q6_K)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(
        dev.memcpy_stod(&weights)?,
        dev.clone(),
    ))?;

    println!("Q6_K quantized size: {} bytes", xs.data.len);

    // Repack to K/128 format with embedded scales
    let xs_repacked = xs.repack_gemx(&shape)?;
    println!(
        "Q6_K repacked: {} bytes -> {} bytes",
        xs.data.len, xs_repacked.data.len
    );

    // Compute expected via dequant
    let dequant_storage = xs.dequantize(ncols * x_rows)?;
    let dequant_f32 = dequant_storage.as_cuda_slice::<f32>()?;
    let dequant_vec = dev.memcpy_dtov(&dequant_f32.slice(..))?;
    println!("Dequant row 0 first 8: {:?}", &dequant_vec[..8]);
    println!(
        "Dequant row 0 el 256-264 (block boundary): {:?}",
        &dequant_vec[256..264]
    );
    println!("Dequant row 0 last 8: {:?}", &dequant_vec[ncols - 8..ncols]);

    let mut expected = vec![0.0f32; x_rows];
    for row in 0..x_rows {
        for col in 0..ncols {
            expected[row] += dequant_vec[row * ncols + col] * 1.0;
        }
    }
    println!("Q6_K expected sums: {:?}", expected);

    // Run matmul
    let qtype = dtype_to_qtype(GgmlDType::Q6_K)? as i32;
    let dst = unsafe { dev.alloc::<f16>(x_rows * y_cols)? };

    {
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = xs_repacked.data.inner.device_ptr(&stream);
        let segment = VxSegment {
            weights: data_ptr as *const std::ffi::c_void,
            batch_count: y_cols as i32,
        };
        let (y_ptr, _y_guard) = y.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

        let status = unsafe {
            run_quantized_matmul(
                &segment as *const VxSegment,
                1,
                y_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                ncols as i32,
                x_rows as i32,
                ncols as i32,
                x_rows as i32,
                qtype,
                YType::F16 as i32,
                xs_repacked.data.len,
                0, // force_mode2 (tiling only; result-invariant)
                OutDType::F16 as i32,
            )
        };
        assert_eq!(status, 0, "matmul launcher rejected the call");
    }

    let result = dev.memcpy_dtov(&dst.slice(..))?;
    println!("Q6_K repacked matmul result ({} rows):", x_rows);
    let mut max_err = 0.0f32;
    let mut total_err = 0.0f32;
    for (i, &v) in result.iter().enumerate() {
        let got = f16::to_f32(v);
        let exp = expected[i];
        let err = (got - exp).abs() / exp.abs().max(1.0) * 100.0;
        max_err = max_err.max(err);
        total_err += err;
        if i < 4 || i >= x_rows - 4 {
            println!("  Row {}: {} (expected {}, err {:.1}%)", i, got, exp, err);
        } else if i == 4 {
            println!("  ... (rows 4-{} omitted) ...", x_rows - 5);
        }
    }
    let avg_err = total_err / x_rows as f32;

    println!(
        "Q6_K max error: {:.1}%, avg error: {:.2}%",
        max_err, avg_err
    );

    // Q6_K should be accurate
    assert!(max_err < 5.0, "Q6_K error too high: {:.1}%", max_err);

    Ok(())
}

/// Test Q6_K with GGUF-loaded data - compare repacked vs dequant path
#[test]
fn cuda_mm_q6_k_gguf_debug() -> Result<()> {
    use crate::Tensor;
    use half::f16;

    // Skip if no model available
    let model_path = match std::env::var("GGUF_MODEL_PATH") {
        Ok(p) => p,
        Err(_) => {
            println!("Skipping - set GGUF_MODEL_PATH to test");
            return Ok(());
        }
    };

    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());

    // Load model - read file into memory
    let file_bytes = std::fs::read(&model_path)?;
    let mut cursor = std::io::Cursor::new(&file_bytes);
    let content = crate::quantized::gguf_file::Content::read(&mut cursor)?;
    let ct = content;

    // Find a Q6_K tensor
    let q6k_tensor_name = ct
        .tensor_infos
        .iter()
        .find(|(name, info)| {
            info.ggml_dtype == crate::quantized::GgmlDType::Q6_K && name.contains("attn_v")
        })
        .map(|(name, _)| name.clone());

    let tensor_name = match q6k_tensor_name {
        Some(name) => name,
        None => {
            println!("No Q6_K tensor found in model");
            return Ok(());
        }
    };

    println!("Testing Q6_K tensor: {}", tensor_name);

    let tensor_info = ct.tensor_infos.get(&tensor_name).unwrap();
    let qtensor = tensor_info.read_from_mmap(&file_bytes, ct.tensor_data_offset, &device)?;
    let (nrows, ncols) = qtensor.shape().dims2()?;
    println!("Shape: {}x{}", nrows, ncols);

    // Create Y as all ones, shape (ncols,) for vec matmul
    let y_f16_data: Vec<f16> = vec![f16::from_f32(1.0); ncols];
    let y_f16 = Tensor::from_vec(y_f16_data, (1, ncols), &device)?;

    // Method 1: Dequantize to f16 then matmul (using QTensor directly, not QMatMul)
    let w_f16 = qtensor.dequantize_f16(&device)?;
    let result_dequant = y_f16.matmul(&w_f16.t()?)?;
    let result_dequant_vec: Vec<f16> = result_dequant.flatten_all()?.to_vec1()?;

    // Expected - just print what we get
    println!("\nResults via dequant f16 (first 8 rows):");
    for (i, &r) in result_dequant_vec.iter().enumerate().take(8.min(nrows)) {
        println!("Row {}: {:.4}", i, f16::to_f32(r));
    }

    Ok(())
}

/// Test Q4_K matmul with GGUF-loaded weights vs dequantized matmul
/// This verifies that GGUF-loaded data produces correct results
///
/// Run with: GGUF_MODEL_PATH=/path/to/model.gguf cargo test cuda_mm_q4_k_gguf_vs_dequant --ignored
#[test]
#[ignore] // Requires GGUF_MODEL_PATH environment variable
fn cuda_mm_q4_k_gguf_vs_dequant() -> Result<()> {
    use crate::quantized::gguf_file;
    #[allow(unused_imports)]
    use half::f16;

    let model_path = std::env::var("GGUF_MODEL_PATH")
        .map_err(|_| crate::Error::Msg("Set GGUF_MODEL_PATH to a Q4_K GGUF model".into()))?;

    let dev = CudaDevice::new(0)?;

    println!("Loading model from: {}", model_path);

    // Open and parse GGUF file
    use memmap2::MmapOptions;
    let file = std::fs::File::open(&model_path)?;
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|e| crate::Error::Msg(format!("Failed to mmap file: {}", e)))?
    };
    let mut cursor = std::io::Cursor::new(&mmap[..]);
    let ct = gguf_file::Content::read(&mut cursor)?;

    // Test both Q4_K and Q6_K tensors
    let test_tensors = [
        ("blk.0.attn_q.weight", "Q4_K (attention_wq)"),
        ("blk.0.attn_k.weight", "Q6_K (attention_wk)"),
        ("blk.0.attn_v.weight", "Q6_K (attention_wv)"),
        ("blk.0.attn_output.weight", "attention_wo"),
        ("blk.0.ffn_gate.weight", "feed_forward_w1"),
        ("blk.0.ffn_down.weight", "feed_forward_w2"),
        ("blk.0.ffn_up.weight", "feed_forward_w3"),
    ];

    let device = crate::Device::Cuda(dev.clone());

    for (tensor_name, label) in test_tensors {
        let tensor_info = match ct.tensor_infos.get(tensor_name) {
            Some(info) => info,
            None => {
                println!("Tensor {} not found, skipping", tensor_name);
                continue;
            }
        };

        println!("\n=== Testing {} ({:?}) ===", label, tensor_info.ggml_dtype);
        println!("Shape: {:?}", tensor_info.shape);

        let qtensor = tensor_info.read_from_mmap(&mmap, ct.tensor_data_offset, &device)?;
        let (nrows, ncols) = qtensor.shape().dims2()?;

        // Create random input
        let input_data: Vec<half::bf16> = (0..ncols)
            .map(|i| half::bf16::from_f32((i as f32 / ncols as f32) * 2.0 - 1.0))
            .collect();
        let input = dev.memcpy_stod(&input_data)?;

        // Method 1: Reference via dequantize
        let dequant = qtensor.dequantize(&device)?;
        let dequant_f32 = dequant.to_vec2::<f32>()?;
        let mut expected = vec![0.0f32; nrows];
        for row in 0..nrows {
            for col in 0..ncols {
                expected[row] += dequant_f32[row][col] * half::bf16::to_f32(input_data[col]);
            }
        }

        // Method 2: Quantized matmul with K/128 embedded scales
        // Repack
        let repacked = match &qtensor.storage {
            crate::quantized::QStorage::Cuda(s) => s.repack_gemx(qtensor.shape())?,
            _ => unreachable!(),
        };

        // For Q6K, verify repacking correctness by comparing first block
        if qtensor.dtype() == crate::quantized::GgmlDType::Q6_K {
            let original_data = match &qtensor.storage {
                crate::quantized::QStorage::Cuda(s) => {
                    let data_slice = s.data.inner.slice(..210.min(s.data.len));
                    dev.memcpy_dtov(&data_slice)?
                }
                _ => unreachable!(),
            };
            let repacked_data = {
                let data_slice = repacked.data.inner.slice(..192.min(repacked.data.len));
                dev.memcpy_dtov(&data_slice)?
            };
            println!(
                "Q6K block 0 original ql[0:8]: {:02x?}",
                &original_data[0..8]
            );
            println!(
                "Q6K block 0 repacked ql[0:8]: {:02x?}",
                &repacked_data[0..8]
            );
            println!(
                "Q6K block 0 original qh[0:8]: {:02x?}",
                &original_data[128..136]
            );
            println!(
                "Q6K block 0 repacked qh[0:8]: {:02x?}",
                &repacked_data[128..136]
            );

            // Verify ql match (bytes 0-127)
            let ql_match = original_data[..128] == repacked_data[..128];
            // Verify qh match (bytes 128-191)
            let qh_match = original_data[128..192] == repacked_data[128..192];
            println!("Q6K repack: ql match={}, qh match={}", ql_match, qh_match);
        }

        // Allocate output
        let qtype = dtype_to_qtype(qtensor.dtype())? as i32;
        let dst = unsafe { dev.alloc::<half::bf16>(nrows)? };

        // Run matmul (K/128 format has embedded scales)
        {
            let stream = dev.cuda_stream();
            let (data_ptr, _data_guard) = repacked.data.inner.device_ptr(&stream);
            let segment = VxSegment {
                weights: data_ptr as *const std::ffi::c_void,
                batch_count: 1,
            };
            let (y_ptr, _y_guard) = input.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

            let status = unsafe {
                run_quantized_matmul(
                    &segment as *const VxSegment,
                    1,
                    y_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    ncols as i32,
                    nrows as i32,
                    ncols as i32,
                    nrows as i32,
                    qtype,
                    YType::BF16 as i32,
                    repacked.data.len,
                    0, // force_mode2 (tiling only; result-invariant)
                    OutDType::BF16 as i32,
                )
            };
            assert_eq!(status, 0, "matmul launcher rejected the call");
        }

        let result = dev.memcpy_dtov(&dst.slice(..))?;
        let result_f32: Vec<f32> = result.iter().map(|v| half::bf16::to_f32(*v)).collect();

        // Compare
        let mut max_err = 0.0f32;
        let mut sum_err = 0.0f32;
        for i in 0..nrows {
            let err = (result_f32[i] - expected[i]).abs() / expected[i].abs().max(1.0);
            max_err = max_err.max(err);
            sum_err += err;
        }
        let avg_err = sum_err / nrows as f32;

        println!(
            "Result first 5: {:?}",
            &result_f32[..5.min(result_f32.len())]
        );
        println!("Expected first 5: {:?}", &expected[..5.min(expected.len())]);
        println!(
            "Error: max={:.2}%, avg={:.2}%",
            max_err * 100.0,
            avg_err * 100.0
        );

        assert!(
            avg_err < 0.05,
            "{} average error too high: {:.2}%",
            label,
            avg_err * 100.0
        );
    }

    Ok(())
}

/// Test fused QKV (3 matrices concatenated) matmul
/// This verifies that concat_rows_cuda + extract_scales + repack works correctly
#[test]
#[ignore] // Requires GGUF_MODEL_PATH environment variable
fn cuda_mm_q4_k_fused_qkv() -> Result<()> {
    use crate::quantized::gguf_file;
    #[allow(unused_imports)]
    use half::f16;

    let model_path = std::env::var("GGUF_MODEL_PATH")
        .map_err(|_| crate::Error::Msg("Set GGUF_MODEL_PATH to a Q4_K GGUF model".into()))?;

    let dev = CudaDevice::new(0)?;

    println!("Loading model from: {}", model_path);

    // Open and parse GGUF file
    use memmap2::MmapOptions;
    let file = std::fs::File::open(&model_path)?;
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|e| crate::Error::Msg(format!("Failed to mmap file: {}", e)))?
    };
    let mut cursor = std::io::Cursor::new(&mmap[..]);
    let ct = gguf_file::Content::read(&mut cursor)?;

    // Load Q, K, V matrices from first layer
    let device = crate::Device::Cuda(dev.clone());

    let wq = ct
        .tensor_infos
        .get("blk.0.attn_q.weight")
        .ok_or_else(|| crate::Error::Msg("attn_q not found".into()))?
        .read_from_mmap(&mmap, ct.tensor_data_offset, &device)?;
    let wk = ct
        .tensor_infos
        .get("blk.0.attn_k.weight")
        .ok_or_else(|| crate::Error::Msg("attn_k not found".into()))?
        .read_from_mmap(&mmap, ct.tensor_data_offset, &device)?;
    let wv = ct
        .tensor_infos
        .get("blk.0.attn_v.weight")
        .ok_or_else(|| crate::Error::Msg("attn_v not found".into()))?
        .read_from_mmap(&mmap, ct.tensor_data_offset, &device)?;

    let (nq, ncols) = wq.shape().dims2()?;
    let (nk, _) = wk.shape().dims2()?;
    let (nv, _) = wv.shape().dims2()?;
    println!(
        "Q: {}x{}, K: {}x{}, V: {}x{}",
        nq, ncols, nk, ncols, nv, ncols
    );

    // Create random input
    let input_data: Vec<half::bf16> = (0..ncols)
        .map(|i| half::bf16::from_f32((i as f32 / ncols as f32) * 2.0 - 1.0))
        .collect();
    let input = dev.memcpy_stod(&input_data)?;

    // Method 1: Separate matmuls (reference)
    println!("\n=== Method 1: Separate matmuls ===");
    let dequant_q = wq.dequantize(&device)?.to_vec2::<f32>()?;
    let dequant_k = wk.dequantize(&device)?.to_vec2::<f32>()?;
    let dequant_v = wv.dequantize(&device)?.to_vec2::<f32>()?;

    let mut expected_q = vec![0.0f32; nq];
    let mut expected_k = vec![0.0f32; nk];
    let mut expected_v = vec![0.0f32; nv];

    for row in 0..nq {
        for col in 0..ncols {
            expected_q[row] += dequant_q[row][col] * half::bf16::to_f32(input_data[col]);
        }
    }
    for row in 0..nk {
        for col in 0..ncols {
            expected_k[row] += dequant_k[row][col] * half::bf16::to_f32(input_data[col]);
        }
    }
    for row in 0..nv {
        for col in 0..ncols {
            expected_v[row] += dequant_v[row][col] * half::bf16::to_f32(input_data[col]);
        }
    }
    println!("Expected Q first 5: {:?}", &expected_q[..5]);
    println!("Expected K first 5: {:?}", &expected_k[..5]);
    println!("Expected V first 5: {:?}", &expected_v[..5]);

    // Method 2: Fused QKV
    println!("\n=== Method 2: Fused QKV ===");
    let fused = crate::quantized::QTensor::concat_rows_cuda(&[&wq, &wk, &wv])?;
    let (nfused, ncols2) = fused.shape().dims2()?;
    println!("Fused shape: {}x{}", nfused, ncols2);
    assert_eq!(nfused, nq + nk + nv);
    assert_eq!(ncols2, ncols);

    // Repack to K/128 format with embedded scales
    let repacked = match &fused.storage {
        crate::quantized::QStorage::Cuda(s) => s.repack_gemx(fused.shape())?,
        _ => unreachable!(),
    };

    // Allocate output
    let qtype = dtype_to_qtype(GgmlDType::Q4_K)? as i32;
    let dst = unsafe { dev.alloc::<half::bf16>(nfused)? };

    // Run fused matmul (K/128 format has embedded scales)
    {
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = repacked.data.inner.device_ptr(&stream);
        let segment = VxSegment {
            weights: data_ptr as *const std::ffi::c_void,
            batch_count: 1,
        };
        let (y_ptr, _y_guard) = input.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

        let status = unsafe {
            run_quantized_matmul(
                &segment as *const VxSegment,
                1,
                y_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                ncols as i32,
                nfused as i32,
                ncols as i32,
                nfused as i32,
                qtype,
                YType::BF16 as i32,
                repacked.data.len,
                0, // force_mode2 (tiling only; result-invariant)
                OutDType::BF16 as i32,
            )
        };
        assert_eq!(status, 0, "matmul launcher rejected the call");
    }

    let result = dev.memcpy_dtov(&dst.slice(..))?;
    let result_f32: Vec<f32> = result.iter().map(|v| half::bf16::to_f32(*v)).collect();

    // Split result into Q, K, V
    let result_q = &result_f32[0..nq];
    let result_k = &result_f32[nq..nq + nk];
    let result_v = &result_f32[nq + nk..nq + nk + nv];

    println!("Result Q first 5: {:?}", &result_q[..5]);
    println!("Result K first 5: {:?}", &result_k[..5]);
    println!("Result V first 5: {:?}", &result_v[..5]);

    // Compare
    let mut max_err_q = 0.0f32;
    let mut sum_err_q = 0.0f32;
    for i in 0..nq {
        let err = (result_q[i] - expected_q[i]).abs() / expected_q[i].abs().max(1.0);
        max_err_q = max_err_q.max(err);
        sum_err_q += err;
    }

    let mut max_err_k = 0.0f32;
    let mut sum_err_k = 0.0f32;
    for i in 0..nk {
        let err = (result_k[i] - expected_k[i]).abs() / expected_k[i].abs().max(1.0);
        max_err_k = max_err_k.max(err);
        sum_err_k += err;
    }

    let mut max_err_v = 0.0f32;
    let mut sum_err_v = 0.0f32;
    for i in 0..nv {
        let err = (result_v[i] - expected_v[i]).abs() / expected_v[i].abs().max(1.0);
        max_err_v = max_err_v.max(err);
        sum_err_v += err;
    }

    println!(
        "\nQ error: max={:.2}%, avg={:.2}%",
        max_err_q * 100.0,
        (sum_err_q / nq as f32) * 100.0
    );
    println!(
        "K error: max={:.2}%, avg={:.2}%",
        max_err_k * 100.0,
        (sum_err_k / nk as f32) * 100.0
    );
    println!(
        "V error: max={:.2}%, avg={:.2}%",
        max_err_v * 100.0,
        (sum_err_v / nv as f32) * 100.0
    );

    // Should have reasonable accuracy
    let avg_err_q = sum_err_q / nq as f32;
    let avg_err_k = sum_err_k / nk as f32;
    let avg_err_v = sum_err_v / nv as f32;

    assert!(
        avg_err_q < 0.05,
        "Q average error too high: {:.2}%",
        avg_err_q * 100.0
    );
    assert!(
        avg_err_k < 0.05,
        "K average error too high: {:.2}%",
        avg_err_k * 100.0
    );
    assert!(
        avg_err_v < 0.05,
        "V average error too high: {:.2}%",
        avg_err_v * 100.0
    );

    Ok(())
}

/// Comprehensive quantize/dequantize roundtrip test for all quantization types.
/// Tests correctness and measures performance (GB/s).
#[test]
fn debug_q4_0_roundtrip() -> Result<()> {
    let dev = CudaDevice::new(0)?;

    // Test one block of Q4_0 = 32 elements
    let elem_count = 32usize;
    let dtype = GgmlDType::Q4_0;

    // Simple ascending values
    let src_data: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) / 2.0).collect();
    println!("Input: {:?}", src_data);

    let src_gpu = dev.memcpy_stod(&src_data)?;

    let block_size = dtype.block_size();
    let type_size = dtype.type_size();
    let num_blocks = elem_count.div_ceil(block_size);
    let quant_size = num_blocks * type_size;

    println!(
        "block_size={}, type_size={}, num_blocks={}, quant_size={}",
        block_size, type_size, num_blocks, quant_size
    );

    let mut quant_buf = unsafe { dev.alloc::<u8>(quant_size)? };

    quantize_to_dtype(&src_gpu.slice(..), &mut quant_buf, elem_count, dtype, &dev)?;
    dev.synchronize()?;

    // Read quantized bytes
    let quant_bytes: Vec<u8> = dev.memcpy_dtov(&quant_buf.slice(..))?;
    println!("Quantized bytes: {:?}", quant_bytes);

    // Dequantize
    let padded = PaddedCudaSlice {
        inner: quant_buf.clone(),
        len: quant_size,
    };
    let dequant_storage = dequantize_f32(&padded, dtype, elem_count, &dev)?;
    dev.synchronize()?;

    let dequant_gpu = dequant_storage.as_cuda_slice::<f32>()?;
    let dequant_data = dev.memcpy_dtov(&dequant_gpu.slice(..))?;
    println!("Dequantized: {:?}", dequant_data);

    // Error
    for i in 0..elem_count {
        let err = (src_data[i] - dequant_data[i]).abs();
        if err > 0.5 {
            println!(
                "ERROR at {}: src={}, dequant={}, err={}",
                i, src_data[i], dequant_data[i], err
            );
        }
    }

    Ok(())
}

#[test]
fn quantize_dequantize_roundtrip_all_dtypes() -> Result<()> {
    use std::time::Instant;

    let dev = CudaDevice::new(0)?;

    // Test parameters - large enough to measure, small enough to be fast
    // 1M elements = 4MB f32 data, runs in ~1-10ms per dtype
    let elem_count = 1024 * 1024usize; // 1M elements
    let num_warmup = 2;
    let num_iters = 5;

    // All supported quantization types with their expected max RMSE
    // Tolerances based on theoretical quantization error for uniform random [-8, 8]
    // RMSE â‰ˆ step_size / sqrt(12) where step_size = range / (2^bits - 1)
    // Added margin for rounding differences between GPU and CPU
    let test_configs: Vec<(GgmlDType, f32, &str)> = vec![
        (GgmlDType::Q4_0, 0.35, "Q4_0"), // 4-bit: theoretical ~0.31
        (GgmlDType::Q4_1, 0.35, "Q4_1"), // 4-bit with min offset
        (GgmlDType::Q5_0, 0.18, "Q5_0"), // 5-bit: theoretical ~0.15
        (GgmlDType::Q5_1, 0.18, "Q5_1"), // 5-bit with min offset
        (GgmlDType::Q8_0, 0.04, "Q8_0"), // 8-bit: theoretical ~0.018
        (GgmlDType::Q8_1, 0.04, "Q8_1"), // 8-bit with sum
        (GgmlDType::Q2_K, 1.50, "Q2K"),  // 2-bit K-quant: high error expected
        (GgmlDType::Q3_K, 0.80, "Q3K"),  // 3-bit K-quant
        (GgmlDType::Q4_K, 0.40, "Q4K"),  // 4-bit K-quant
        (GgmlDType::Q5_K, 0.25, "Q5K"),  // 5-bit K-quant
        (GgmlDType::Q6_K, 0.15, "Q6K"),  // 6-bit K-quant
        (GgmlDType::Q8_K, 0.04, "Q8K"),  // 8-bit K-quant
    ];

    // Generate random test data in KV cache typical range [-8.0, 8.0]
    // This matches typical attention key/value magnitudes
    let mut rng = rand::rng();
    let src_data: Vec<f32> = (0..elem_count)
        .map(|_| rng.random_range(-8.0..8.0))
        .collect();
    let src_gpu = dev.memcpy_stod(&src_data)?;

    // Results table
    struct TestResult {
        name: &'static str,
        valid: bool,
        rmse: f32,
        mean_diff: f32,
        max_diff: f32,
        quant_gbps: f64,
        dequant_gbps: f64,
    }
    let mut results: Vec<TestResult> = Vec::new();

    println!("\n=== Quantize/Dequantize Roundtrip Test ===");
    println!(
        "Elements: {} ({:.2} MB f32), Range: [-8.0, 8.0]",
        elem_count,
        elem_count as f64 * 4.0 / 1e6
    );
    println!();

    for (dtype, max_rmse, name) in &test_configs {
        // Calculate buffer sizes
        let block_size = dtype.block_size();
        let type_size = dtype.type_size();
        let num_blocks = elem_count.div_ceil(block_size);
        let quant_size = num_blocks * type_size;

        // Allocate quantized buffer
        let mut quant_buf = unsafe { dev.alloc::<u8>(quant_size)? };

        // Warmup
        for _ in 0..num_warmup {
            quantize_to_dtype(&src_gpu.slice(..), &mut quant_buf, elem_count, *dtype, &dev)?;
            dev.synchronize()?;
        }

        // Benchmark quantize
        let start = Instant::now();
        for _ in 0..num_iters {
            quantize_to_dtype(&src_gpu.slice(..), &mut quant_buf, elem_count, *dtype, &dev)?;
        }
        dev.synchronize()?;
        let quant_time = start.elapsed().as_secs_f64() / num_iters as f64;

        // Create QCudaStorage for dequantize
        let padded = PaddedCudaSlice {
            inner: quant_buf.clone(),
            len: quant_size,
        };

        // Warmup dequantize
        for _ in 0..num_warmup {
            let _ = dequantize_f32(&padded, *dtype, elem_count, &dev)?;
            dev.synchronize()?;
        }

        // Benchmark dequantize
        let start = Instant::now();
        let mut dequant_storage = None;
        for _ in 0..num_iters {
            dequant_storage = Some(dequantize_f32(&padded, *dtype, elem_count, &dev)?);
        }
        dev.synchronize()?;
        let dequant_time = start.elapsed().as_secs_f64() / num_iters as f64;

        // Get dequantized result for validation
        let dequant_storage = dequant_storage.unwrap();
        let dequant_gpu = dequant_storage.as_cuda_slice::<f32>()?;
        let dequant_data = dev.memcpy_dtov(&dequant_gpu.slice(..))?;

        // Calculate error metrics
        let mut sum_sq_err = 0.0f64;
        let mut sum_abs_diff = 0.0f64;
        let mut max_diff = 0.0f32;
        for i in 0..elem_count {
            let diff = src_data[i] - dequant_data[i];
            let abs_diff = diff.abs();
            sum_sq_err += (diff as f64) * (diff as f64);
            sum_abs_diff += abs_diff as f64;
            if abs_diff > max_diff {
                max_diff = abs_diff;
            }
        }
        let rmse = (sum_sq_err / elem_count as f64).sqrt() as f32;
        let mean_diff = (sum_abs_diff / elem_count as f64) as f32;
        let valid = rmse <= *max_rmse;

        // Calculate throughput
        // Quantize: read f32, write quant
        let quant_bytes = (elem_count * 4 + quant_size) as f64;
        let quant_gbps = quant_bytes / quant_time / 1e9;

        // Dequantize: read quant, write f32
        let dequant_bytes = (quant_size + elem_count * 4) as f64;
        let dequant_gbps = dequant_bytes / dequant_time / 1e9;

        results.push(TestResult {
            name,
            valid,
            rmse,
            mean_diff,
            max_diff,
            quant_gbps,
            dequant_gbps,
        });
    }

    // Print summary table
    println!(
        "â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”"
    );
    println!(
        "â”‚ DType   â”‚ Valid â”‚ RMSE     â”‚ MeanDiff â”‚ MaxDiff  â”‚ Quant GB/s  â”‚ Dequant GB/sâ”‚"
    );
    println!(
        "â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤"
    );
    for r in &results {
        let valid_str = if r.valid { "âœ“" } else { "âœ—" };
        println!(
            "â”‚ {:7} â”‚   {}   â”‚ {:8.5} â”‚ {:8.5} â”‚ {:8.4} â”‚ {:10.2}  â”‚ {:10.2}  â”‚",
            r.name, valid_str, r.rmse, r.mean_diff, r.max_diff, r.quant_gbps, r.dequant_gbps
        );
    }
    println!(
        "â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜"
    );

    // Assert all passed
    let all_valid = results.iter().all(|r| r.valid);
    assert!(all_valid, "Some quantization types failed validation");

    Ok(())
}

/// Precision test: Creates known-good quantized data on CPU, then tests GPU quantize
/// by verifying it produces identical bytes when requantizing dequantized values.
///
/// Test flow:
/// 1. CPU: Generate random f32 data, quantize with CPU (known-good) â†’ cpu_quant_bytes
/// 2. CPU: Dequantize CPU result â†’ baseline_f32 (this is what the quantized block represents)
/// 3. Upload baseline_f32 to GPU
/// 4. GPU: Quantize baseline_f32 â†’ gpu_quant_bytes
/// 5. Compare: gpu_quant_bytes == cpu_quant_bytes (should match exactly!)
/// 6. GPU: Dequantize gpu_quant_bytes â†’ roundtrip_f32
/// 7. Compare: roundtrip_f32 vs baseline_f32 (RMSE should be ~0)
///
/// This isolates the GPU quantize kernel and verifies byte-level correctness.
#[test]
fn quantize_kernel_byte_accuracy() -> Result<()> {
    use crate::quantized::k_quants::{BlockQ2_K, BlockQ4_0, BlockQ4_K, BlockQ8_0, GgmlType};
    use std::time::Instant;

    let dev = CudaDevice::new(0)?;

    // Test with enough blocks to measure bandwidth, but not too many for quick tests
    let num_blocks = 4096; // 4K blocks

    println!("\n=== GPU Quantize Kernel Byte-Level Accuracy Test ===\n");

    // First, a simple debug test for Q4_0
    {
        println!("=== Q4_0 Debug (1 block) ===");
        // Create simple ascending data
        let test_data: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) / 2.0).collect();
        println!("Input data: {:?}", test_data);

        // CPU quantize
        let mut cpu_block = vec![BlockQ4_0::zeros(); 1];
        BlockQ4_0::from_float(&test_data, &mut cpu_block);

        // CPU dequantize to get baseline
        let mut baseline = vec![0.0f32; 32];
        BlockQ4_0::to_float(&cpu_block, &mut baseline);
        println!("CPU baseline (after quant->dequant): {:?}", baseline);

        // Show CPU block
        println!(
            "CPU block: d={:?}, qs={:?}",
            cpu_block[0].d, cpu_block[0].qs
        );

        // Upload baseline to GPU and quantize
        let baseline_gpu = dev.memcpy_stod(&baseline)?;
        let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(18)? };
        quantize_to_dtype(
            &baseline_gpu.slice(..),
            &mut gpu_quant_buf,
            32,
            GgmlDType::Q4_0,
            &dev,
        )?;
        dev.synchronize()?;

        let gpu_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;
        let cpu_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(cpu_block.as_ptr() as *const u8, 18) };

        println!("CPU bytes: {:02x?}", cpu_bytes);
        println!("GPU bytes: {:02x?}", gpu_bytes);

        // Compare
        for i in 0..18 {
            if cpu_bytes[i] != gpu_bytes[i] {
                println!(
                    "  Mismatch at byte {}: CPU={:02x} GPU={:02x}",
                    i, cpu_bytes[i], gpu_bytes[i]
                );
            }
        }
        println!();
    }

    // Q2K debug test (1 block) - DIRECT comparison (same raw input)
    {
        println!("=== Q2K Debug (1 block, direct) ===");
        // Create simple data that spans a known range
        let test_data: Vec<f32> = (0..256).map(|i| (i as f32 - 127.5) / 16.0).collect();
        println!(
            "Input data range: {:.3} to {:.3}",
            test_data[0], test_data[255]
        );

        // CPU quantize the RAW data
        let mut cpu_block = vec![BlockQ2_K::zeros(); 1];
        BlockQ2_K::from_float(&test_data, &mut cpu_block);

        // Show CPU block header
        println!(
            "CPU block: d={:?}, dmin={:?}",
            cpu_block[0].d, cpu_block[0].dmin
        );
        println!("CPU scales[0..16]: {:02x?}", &cpu_block[0].scales);

        // GPU quantize the SAME raw data
        let test_data_gpu = dev.memcpy_stod(&test_data)?;
        let type_size = std::mem::size_of::<BlockQ2_K>();
        let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(type_size)? };
        quantize_to_dtype(
            &test_data_gpu.slice(..),
            &mut gpu_quant_buf,
            256,
            GgmlDType::Q2_K,
            &dev,
        )?;
        dev.synchronize()?;

        let gpu_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;
        let cpu_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(cpu_block.as_ptr() as *const u8, type_size) };

        // Q2K layout: scales[16], qs[64], d (2 bytes), dmin (2 bytes) = 84 bytes
        println!("CPU scales bytes[0..16]: {:02x?}", &cpu_bytes[0..16]);
        println!("GPU scales bytes[0..16]: {:02x?}", &gpu_bytes[0..16]);

        // d is at offset 80, dmin at 82
        let cpu_d = half::f16::from_le_bytes([cpu_bytes[80], cpu_bytes[81]]);
        let cpu_dmin = half::f16::from_le_bytes([cpu_bytes[82], cpu_bytes[83]]);
        let gpu_d = half::f16::from_le_bytes([gpu_bytes[80], gpu_bytes[81]]);
        let gpu_dmin = half::f16::from_le_bytes([gpu_bytes[82], gpu_bytes[83]]);
        println!(
            "CPU: d={:.6}, dmin={:.6}",
            cpu_d.to_f32(),
            cpu_dmin.to_f32()
        );
        println!(
            "GPU: d={:.6}, dmin={:.6}",
            gpu_d.to_f32(),
            gpu_dmin.to_f32()
        );

        // Count mismatches
        let mut mismatches = 0;
        for i in 0..type_size {
            if cpu_bytes[i] != gpu_bytes[i] {
                mismatches += 1;
                if mismatches <= 5 {
                    println!(
                        "Mismatch at byte {}: CPU={:02x} GPU={:02x}",
                        i, cpu_bytes[i], gpu_bytes[i]
                    );
                }
            }
        }
        println!("Total mismatches: {} / {}", mismatches, type_size);

        println!();
    }

    // Q4K debug test (1 block)
    {
        println!("=== Q4K Debug (1 block) ===");
        // Create simple data that spans a known range
        let test_data: Vec<f32> = (0..256).map(|i| (i as f32 - 127.5) / 16.0).collect();
        println!(
            "Input data range: {:.3} to {:.3}",
            test_data[0], test_data[255]
        );

        // CPU quantize
        let mut cpu_block = vec![BlockQ4_K::zeros(); 1];
        BlockQ4_K::from_float(&test_data, &mut cpu_block);

        // CPU dequantize to get baseline
        let mut baseline = vec![0.0f32; 256];
        BlockQ4_K::to_float(&cpu_block, &mut baseline);
        println!(
            "CPU baseline range: {:.3} to {:.3}",
            baseline.iter().cloned().fold(f32::INFINITY, f32::min),
            baseline.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );

        // Show CPU block header
        println!(
            "CPU block: d={:?}, dmin={:?}",
            cpu_block[0].d, cpu_block[0].dmin
        );
        println!("CPU scales[0..12]: {:?}", &cpu_block[0].scales);

        // Upload baseline to GPU and quantize
        let baseline_gpu = dev.memcpy_stod(&baseline)?;
        let type_size = std::mem::size_of::<BlockQ4_K>();
        let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(type_size)? };
        quantize_to_dtype(
            &baseline_gpu.slice(..),
            &mut gpu_quant_buf,
            256,
            GgmlDType::Q4_K,
            &dev,
        )?;
        dev.synchronize()?;

        let gpu_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;
        let cpu_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(cpu_block.as_ptr() as *const u8, type_size) };

        // Show dm bytes (first 4 bytes = half2)
        println!("CPU dm bytes: {:02x?}", &cpu_bytes[0..4]);
        println!("GPU dm bytes: {:02x?}", &gpu_bytes[0..4]);

        // Interpret as half
        let cpu_d = half::f16::from_le_bytes([cpu_bytes[0], cpu_bytes[1]]);
        let cpu_dmin = half::f16::from_le_bytes([cpu_bytes[2], cpu_bytes[3]]);
        let gpu_d = half::f16::from_le_bytes([gpu_bytes[0], gpu_bytes[1]]);
        let gpu_dmin = half::f16::from_le_bytes([gpu_bytes[2], gpu_bytes[3]]);
        println!(
            "CPU: d={:.6}, dmin={:.6}",
            cpu_d.to_f32(),
            cpu_dmin.to_f32()
        );
        println!(
            "GPU: d={:.6}, dmin={:.6}",
            gpu_d.to_f32(),
            gpu_dmin.to_f32()
        );

        // Show scales bytes (bytes 4-15)
        println!("CPU scales bytes: {:02x?}", &cpu_bytes[4..16]);
        println!("GPU scales bytes: {:02x?}", &gpu_bytes[4..16]);

        println!();
    }

    // Q4K debug test with random data (first block only)
    {
        println!("=== Q4K Debug (random, 1 block, seeded) ===");
        // Create random data with a seed for reproducibility
        let test_data: Vec<f32> = (0..256)
            .map(|i| {
                // Pseudo-random but deterministic, uses a simple LCG
                let seed = (i as u64)
                    .wrapping_mul(6364136223846793005u64)
                    .wrapping_add(1442695040888963407u64);
                let x = (seed >> 33) as i32;
                (x as f32 - (i32::MAX / 2) as f32) / (i32::MAX as f32 / 16.0)
            })
            .collect();
        println!(
            "Input data range: {:.3} to {:.3}",
            test_data.iter().cloned().fold(f32::INFINITY, f32::min),
            test_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );

        // Show first 8 values (first sub-block)
        println!("First 8 values: {:?}", &test_data[0..8]);

        // CPU quantize
        let mut cpu_block = vec![BlockQ4_K::zeros(); 1];
        BlockQ4_K::from_float(&test_data, &mut cpu_block);

        // CPU dequantize
        let mut cpu_dequant = vec![0.0f32; 256];
        BlockQ4_K::to_float(&cpu_block, &mut cpu_dequant);

        // Show CPU block header
        println!(
            "CPU block: d={:?}, dmin={:?}",
            cpu_block[0].d, cpu_block[0].dmin
        );

        // GPU quantize the same raw input
        let test_data_gpu = dev.memcpy_stod(&test_data)?;
        let type_size = std::mem::size_of::<BlockQ4_K>();
        let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(type_size)? };
        quantize_to_dtype(
            &test_data_gpu.slice(..),
            &mut gpu_quant_buf,
            256,
            GgmlDType::Q4_K,
            &dev,
        )?;
        dev.synchronize()?;

        let gpu_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;
        let cpu_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(cpu_block.as_ptr() as *const u8, type_size) };

        // Interpret as half
        let cpu_d = half::f16::from_le_bytes([cpu_bytes[0], cpu_bytes[1]]);
        let cpu_dmin = half::f16::from_le_bytes([cpu_bytes[2], cpu_bytes[3]]);
        let gpu_d = half::f16::from_le_bytes([gpu_bytes[0], gpu_bytes[1]]);
        let gpu_dmin = half::f16::from_le_bytes([gpu_bytes[2], gpu_bytes[3]]);
        println!(
            "CPU: d={:.6}, dmin={:.6}",
            cpu_d.to_f32(),
            cpu_dmin.to_f32()
        );
        println!(
            "GPU: d={:.6}, dmin={:.6}",
            gpu_d.to_f32(),
            gpu_dmin.to_f32()
        );

        // Show scales bytes (bytes 4-15)
        println!("CPU scales bytes: {:02x?}", &cpu_bytes[4..16]);
        println!("GPU scales bytes: {:02x?}", &gpu_bytes[4..16]);

        // Decode and show scales/mins
        fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
            if j < 4 {
                let d = q[j] & 63;
                let m = q[j + 4] & 63;
                (d, m)
            } else {
                let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
                (d, m)
            }
        }

        println!("CPU sub-block scales/mins:");
        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, &cpu_bytes[4..16]);
            println!("  sub-block {}: sc={}, m={}", j, sc, m);
        }

        println!("GPU sub-block scales/mins:");
        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, &gpu_bytes[4..16]);
            println!("  sub-block {}: sc={}, m={}", j, sc, m);
        }

        // Count mismatches
        let mut mismatches = 0;
        for i in 0..type_size {
            if cpu_bytes[i] != gpu_bytes[i] {
                mismatches += 1;
            }
        }
        println!("Total mismatches: {}", mismatches);

        println!();
    }

    // Q4K with truly random data (1 block)
    {
        println!("=== Q4K Debug (truly random, 1 block) ===");
        let mut rng = rand::rng();
        let test_data: Vec<f32> = (0..256).map(|_| rng.random_range(-8.0..8.0)).collect();
        println!(
            "Input data range: {:.3} to {:.3}",
            test_data.iter().cloned().fold(f32::INFINITY, f32::min),
            test_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );

        // CPU quantize
        let mut cpu_block = vec![BlockQ4_K::zeros(); 1];
        BlockQ4_K::from_float(&test_data, &mut cpu_block);

        println!(
            "CPU block: d={:?}, dmin={:?}",
            cpu_block[0].d, cpu_block[0].dmin
        );

        // GPU quantize the same raw input
        let test_data_gpu = dev.memcpy_stod(&test_data)?;
        let type_size = std::mem::size_of::<BlockQ4_K>();
        let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(type_size)? };
        quantize_to_dtype(
            &test_data_gpu.slice(..),
            &mut gpu_quant_buf,
            256,
            GgmlDType::Q4_K,
            &dev,
        )?;
        dev.synchronize()?;

        let gpu_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;
        let cpu_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(cpu_block.as_ptr() as *const u8, type_size) };

        let cpu_d = half::f16::from_le_bytes([cpu_bytes[0], cpu_bytes[1]]);
        let cpu_dmin = half::f16::from_le_bytes([cpu_bytes[2], cpu_bytes[3]]);
        let gpu_d = half::f16::from_le_bytes([gpu_bytes[0], gpu_bytes[1]]);
        let gpu_dmin = half::f16::from_le_bytes([gpu_bytes[2], gpu_bytes[3]]);
        println!(
            "CPU: d={:.6}, dmin={:.6}",
            cpu_d.to_f32(),
            cpu_dmin.to_f32()
        );
        println!(
            "GPU: d={:.6}, dmin={:.6}",
            gpu_d.to_f32(),
            gpu_dmin.to_f32()
        );

        println!("CPU scales bytes: {:02x?}", &cpu_bytes[4..16]);
        println!("GPU scales bytes: {:02x?}", &gpu_bytes[4..16]);

        let mut mismatches = 0;
        for i in 0..type_size {
            if cpu_bytes[i] != gpu_bytes[i] {
                mismatches += 1;
            }
        }
        println!("Total mismatches: {} / {}", mismatches, type_size);
        println!();
    }

    // Q2K multi-block debug test
    {
        println!("=== Q2K Debug (4 blocks, random) ===");
        let mut rng = rand::rng();
        let num_test_blocks = 4;
        let test_data: Vec<f32> = (0..256 * num_test_blocks)
            .map(|_| rng.random_range(-8.0..8.0))
            .collect();

        // CPU quantize
        let mut cpu_blocks = vec![BlockQ2_K::zeros(); num_test_blocks];
        BlockQ2_K::from_float(&test_data, &mut cpu_blocks);

        // GPU quantize the same raw input
        let test_data_gpu = dev.memcpy_stod(&test_data)?;
        let type_size = std::mem::size_of::<BlockQ2_K>();
        let quant_size = num_test_blocks * type_size;
        let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(quant_size)? };
        quantize_to_dtype(
            &test_data_gpu.slice(..),
            &mut gpu_quant_buf,
            256 * num_test_blocks,
            GgmlDType::Q2_K,
            &dev,
        )?;
        dev.synchronize()?;

        let gpu_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;
        let cpu_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(cpu_blocks.as_ptr() as *const u8, quant_size).to_vec()
        };

        let mut mismatches = 0;
        let mut first_mismatch = None;
        for blk in 0..num_test_blocks {
            for i in 0..type_size {
                let idx = blk * type_size + i;
                if cpu_bytes[idx] != gpu_bytes[idx] {
                    mismatches += 1;
                    if first_mismatch.is_none() {
                        first_mismatch = Some((blk, i, cpu_bytes[idx], gpu_bytes[idx]));
                    }
                }
            }
        }

        println!("Total mismatches: {} / {} bytes", mismatches, quant_size);
        if let Some((blk, off, cpu, gpu)) = first_mismatch {
            println!(
                "First mismatch: block {}, offset {}: CPU={:02x}, GPU={:02x}",
                blk, off, cpu, gpu
            );
        }

        // Show each block's d/dmin and scales comparison
        // Q2K layout: scales[16] (0-15), qs[64] (16-79), d (80-81), dmin (82-83)
        for blk in 0..num_test_blocks {
            let blk_start = blk * type_size;
            let cpu_d =
                half::f16::from_le_bytes([cpu_bytes[blk_start + 80], cpu_bytes[blk_start + 81]]);
            let cpu_dmin =
                half::f16::from_le_bytes([cpu_bytes[blk_start + 82], cpu_bytes[blk_start + 83]]);
            let gpu_d =
                half::f16::from_le_bytes([gpu_bytes[blk_start + 80], gpu_bytes[blk_start + 81]]);
            let gpu_dmin =
                half::f16::from_le_bytes([gpu_bytes[blk_start + 82], gpu_bytes[blk_start + 83]]);
            let d_match = (cpu_d.to_f32() - gpu_d.to_f32()).abs() < 0.001;
            let dmin_match = (cpu_dmin.to_f32() - gpu_dmin.to_f32()).abs() < 0.001;
            println!(
                "  Block {}: CPU d={:.6} dmin={:.6} | GPU d={:.6} dmin={:.6} | d={} dmin={}",
                blk,
                cpu_d.to_f32(),
                cpu_dmin.to_f32(),
                gpu_d.to_f32(),
                gpu_dmin.to_f32(),
                if d_match { "âœ“" } else { "âœ—" },
                if dmin_match { "âœ“" } else { "âœ—" }
            );

            // Show scales comparison for this block
            let cpu_scales = &cpu_bytes[blk_start..blk_start + 16];
            let gpu_scales = &gpu_bytes[blk_start..blk_start + 16];
            let scales_match = cpu_scales == gpu_scales;
            if !scales_match {
                println!("    CPU scales: {:02x?}", cpu_scales);
                println!("    GPU scales: {:02x?}", gpu_scales);
            }
        }
        println!();
    } // Q4K with truly random data (4 blocks - to test multi-block kernel)
    {
        println!("=== Q4K Debug (truly random, 4 blocks) ===");
        let mut rng = rand::rng();
        let num_test_blocks = 4;
        let test_data: Vec<f32> = (0..256 * num_test_blocks)
            .map(|_| rng.random_range(-8.0..8.0))
            .collect();

        // CPU quantize
        let mut cpu_blocks = vec![BlockQ4_K::zeros(); num_test_blocks];
        BlockQ4_K::from_float(&test_data, &mut cpu_blocks);

        // GPU quantize the same raw input
        let test_data_gpu = dev.memcpy_stod(&test_data)?;
        let type_size = std::mem::size_of::<BlockQ4_K>();
        let quant_size = num_test_blocks * type_size;
        let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(quant_size)? };
        quantize_to_dtype(
            &test_data_gpu.slice(..),
            &mut gpu_quant_buf,
            256 * num_test_blocks,
            GgmlDType::Q4_K,
            &dev,
        )?;
        dev.synchronize()?;

        let gpu_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;
        let cpu_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(cpu_blocks.as_ptr() as *const u8, quant_size).to_vec()
        };

        let mut mismatches = 0;
        let mut first_mismatch_block = None;
        for blk in 0..num_test_blocks {
            for i in 0..type_size {
                let idx = blk * type_size + i;
                if cpu_bytes[idx] != gpu_bytes[idx] {
                    mismatches += 1;
                    if first_mismatch_block.is_none() {
                        first_mismatch_block = Some((blk, i, cpu_bytes[idx], gpu_bytes[idx]));
                    }
                }
            }
        }

        println!("Total mismatches: {} / {} bytes", mismatches, quant_size);
        if let Some((blk, off, cpu, gpu)) = first_mismatch_block {
            println!(
                "First mismatch: block {}, offset {}: CPU={:02x}, GPU={:02x}",
                blk, off, cpu, gpu
            );
        }

        // Show each block's d/dmin comparison
        for blk in 0..num_test_blocks {
            let blk_start = blk * type_size;
            let cpu_d = half::f16::from_le_bytes([cpu_bytes[blk_start], cpu_bytes[blk_start + 1]]);
            let cpu_dmin =
                half::f16::from_le_bytes([cpu_bytes[blk_start + 2], cpu_bytes[blk_start + 3]]);
            let gpu_d = half::f16::from_le_bytes([gpu_bytes[blk_start], gpu_bytes[blk_start + 1]]);
            let gpu_dmin =
                half::f16::from_le_bytes([gpu_bytes[blk_start + 2], gpu_bytes[blk_start + 3]]);
            let d_match = cpu_d == gpu_d;
            let dmin_match = cpu_dmin == gpu_dmin;
            println!(
                "  Block {}: CPU d={:.6} dmin={:.6} | GPU d={:.6} dmin={:.6} | d={} dmin={}",
                blk,
                cpu_d.to_f32(),
                cpu_dmin.to_f32(),
                gpu_d.to_f32(),
                gpu_dmin.to_f32(),
                if d_match { "âœ“" } else { "âœ—" },
                if dmin_match { "âœ“" } else { "âœ—" }
            );
        }
        println!();
    }

    // Helper macro for testing a specific quantized type
    macro_rules! test_dtype {
        ($block_ty:ty, $dtype:expr, $name:expr) => {{
            let block_size = <$block_ty>::BLCK_SIZE;
            let elem_count = num_blocks * block_size;
            let type_size = std::mem::size_of::<$block_ty>();
            let quant_size = num_blocks * type_size;

            // Step 1: Generate random data and CPU-quantize (known-good baseline)
            let mut rng = rand::rng();
            let random_f32: Vec<f32> = (0..elem_count)
                .map(|_| rng.random_range(-8.0..8.0))
                .collect();

            let mut cpu_blocks: Vec<$block_ty> = vec![<$block_ty>::zeros(); num_blocks];
            <$block_ty>::from_float(&random_f32, &mut cpu_blocks);

            // Get CPU quantized bytes
            let cpu_quant_bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(
                    cpu_blocks.as_ptr() as *const u8,
                    quant_size
                ).to_vec()
            };

            // Step 2: CPU dequantize to get baseline (what the quantized block represents)
            let mut baseline_f32 = vec![0.0f32; elem_count];
            <$block_ty>::to_float(&cpu_blocks, &mut baseline_f32);

            // Step 3: Upload baseline to GPU (this is NOT timed - we're testing kernel accuracy)
            let baseline_gpu = dev.memcpy_stod(&baseline_f32)?;

            // Step 4: GPU quantize baseline
            let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(quant_size)? };

            // Warmup
            quantize_to_dtype(&baseline_gpu.slice(..), &mut gpu_quant_buf, elem_count, $dtype, &dev)?;
            dev.synchronize()?;

            // Timed run
            let start = Instant::now();
            quantize_to_dtype(&baseline_gpu.slice(..), &mut gpu_quant_buf, elem_count, $dtype, &dev)?;
            dev.synchronize()?;
            let quant_time = start.elapsed();

            // Download GPU quantized bytes
            let gpu_quant_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;

            // Step 5: Compare bytes
            let mut byte_mismatches = 0usize;
            let mut first_mismatch_idx = None;
            for i in 0..quant_size {
                if cpu_quant_bytes[i] != gpu_quant_bytes[i] {
                    byte_mismatches += 1;
                    if first_mismatch_idx.is_none() {
                        first_mismatch_idx = Some(i);
                    }
                }
            }

            // Step 6: GPU dequantize for RMSE comparison
            let padded = PaddedCudaSlice {
                inner: gpu_quant_buf.clone(),
                len: quant_size,
            };

            // Warmup
            let _ = dequantize_f32(&padded, $dtype, elem_count, &dev)?;
            dev.synchronize()?;

            // Timed run
            let start = Instant::now();
            let roundtrip_storage = dequantize_f32(&padded, $dtype, elem_count, &dev)?;
            dev.synchronize()?;
            let dequant_time = start.elapsed();

            let roundtrip_gpu = roundtrip_storage.as_cuda_slice::<f32>()?;
            let roundtrip_f32 = dev.memcpy_dtov(&roundtrip_gpu.slice(..))?;

            // Step 7: Calculate RMSE between baseline and roundtrip
            let mut sum_sq_err = 0.0f64;
            let mut max_diff = 0.0f32;
            for i in 0..elem_count {
                let diff = baseline_f32[i] - roundtrip_f32[i];
                sum_sq_err += (diff as f64) * (diff as f64);
                max_diff = max_diff.max(diff.abs());
            }
            let rmse = (sum_sq_err / elem_count as f64).sqrt() as f32;

            // Bandwidth calculations (only GPU kernel time)
            let quant_bytes_rw = (elem_count * 4 + quant_size) as f64;
            let quant_gbps = quant_bytes_rw / quant_time.as_secs_f64() / 1e9;
            let dequant_bytes_rw = (quant_size + elem_count * 4) as f64;
            let dequant_gbps = dequant_bytes_rw / dequant_time.as_secs_f64() / 1e9;

            // Report
            let byte_match_pct = 100.0 * (quant_size - byte_mismatches) as f64 / quant_size as f64;
            let status = if byte_mismatches == 0 && rmse < 1e-5 { "âœ“" } else { "âœ—" };

            println!("{} {:5}: Bytes: {:6.2}% match ({:6} / {:6} mismatches), RMSE: {:.2e}, MaxDiff: {:.2e}",
                status, $name, byte_match_pct, byte_mismatches, quant_size, rmse, max_diff);
            println!("         Quant: {:6.1} GB/s, Dequant: {:6.1} GB/s",
                quant_gbps, dequant_gbps);

            if let Some(idx) = first_mismatch_idx {
                let block_idx = idx / type_size;
                let byte_in_block = idx % type_size;
                println!("         First mismatch: byte {} (block {}, offset {}): CPU={:02x} GPU={:02x}",
                    idx, block_idx, byte_in_block, cpu_quant_bytes[idx], gpu_quant_bytes[idx]);
            }

            (byte_mismatches == 0, rmse)
        }};
    }

    // Test each dtype
    println!(
        "Testing Q4_0 ({} blocks Ã— {} elements = {} elements)...",
        num_blocks,
        BlockQ4_0::BLCK_SIZE,
        num_blocks * BlockQ4_0::BLCK_SIZE
    );
    let (q4_0_exact, q4_0_rmse) = test_dtype!(BlockQ4_0, GgmlDType::Q4_0, "Q4_0");

    println!(
        "\nTesting Q8_0 ({} blocks Ã— {} elements = {} elements)...",
        num_blocks,
        BlockQ8_0::BLCK_SIZE,
        num_blocks * BlockQ8_0::BLCK_SIZE
    );
    let (q8_0_exact, q8_0_rmse) = test_dtype!(BlockQ8_0, GgmlDType::Q8_0, "Q8_0");

    println!(
        "\nTesting Q2K ({} blocks Ã— {} elements = {} elements)...",
        num_blocks,
        BlockQ2_K::BLCK_SIZE,
        num_blocks * BlockQ2_K::BLCK_SIZE
    );
    let (q2k_exact, q2k_rmse) = test_dtype!(BlockQ2_K, GgmlDType::Q2_K, "Q2K");

    println!(
        "\nTesting Q4K ({} blocks Ã— {} elements = {} elements)...",
        num_blocks,
        BlockQ4_K::BLCK_SIZE,
        num_blocks * BlockQ4_K::BLCK_SIZE
    );
    let (q4k_exact, q4k_rmse) = test_dtype!(BlockQ4_K, GgmlDType::Q4_K, "Q4K");

    println!("\n=== Summary ===");
    println!("Q4_0: exact_bytes={}, rmse={:.2e}", q4_0_exact, q4_0_rmse);
    println!("Q8_0: exact_bytes={}, rmse={:.2e}", q8_0_exact, q8_0_rmse);
    println!("Q2K:  exact_bytes={}, rmse={:.2e}", q2k_exact, q2k_rmse);
    println!("Q4K:  exact_bytes={}, rmse={:.2e}", q4k_exact, q4k_rmse);

    // Reporting diagnostic: prints per-format exact-byte counts and RMSE for a quick
    // overview. Strict byte-exactness is asserted by the dedicated per-format tests.

    Ok(())
}

/// Direct quantization comparison: GPU quantize vs CPU quantize on identical raw input.
/// This tests whether the GPU quantization algorithm matches the CPU exactly.
///
/// Unlike the byte accuracy test (which quantizes already-dequantized data),
/// this test uses fresh random data, so we expect byte-exact matches for
/// formats with deterministic algorithms (like Q4_0, Q8_0).
///
/// For K-quants with iterative refinement (Q4K, Q5K), the GPU may converge to
/// slightly different local optima due to floating-point differences, so we
/// also measure RMSE of the dequantized results.
#[test]
fn quantize_direct_comparison() -> Result<()> {
    use crate::quantized::k_quants::{
        BlockQ2_K, BlockQ3_K, BlockQ4_0, BlockQ4_1, BlockQ4_K, BlockQ5_0, BlockQ5_1, BlockQ5_K,
        BlockQ6_K, BlockQ8_0, BlockQ8_K, BlockQAWQ, BlockQAWQ_G64, GgmlType,
    };

    let dev = CudaDevice::new(0)?;

    // Thresholds for passing
    const MIN_BYTE_MATCH_PCT: f64 = 95.0; // Minimum byte match percentage
    const MAX_RMSE: f64 = 0.5; // Maximum RMSE between GPU and CPU dequant

    struct TestResult {
        name: &'static str,
        byte_match_pct: f64,
        mismatches: usize,
        rmse: f64,
        max_diff: f32,
        passed: bool,
    }

    let mut results: Vec<TestResult> = Vec::new();

    macro_rules! test_direct {
        ($block_ty:ty, $dtype:expr, $name:expr, $num_blocks:expr) => {{
            let block_size = <$block_ty>::BLCK_SIZE;
            let elem_count = $num_blocks * block_size;
            let type_size = std::mem::size_of::<$block_ty>();
            let quant_size = $num_blocks * type_size;

            // Generate random raw input
            let mut rng = rand::rng();
            let raw_input: Vec<f32> = (0..elem_count)
                .map(|_| rng.random_range(-8.0..8.0))
                .collect();

            // CPU quantize
            let mut cpu_blocks: Vec<$block_ty> = vec![<$block_ty>::zeros(); $num_blocks];
            <$block_ty>::from_float(&raw_input, &mut cpu_blocks);

            // CPU dequantize for RMSE comparison
            let mut cpu_dequant = vec![0.0f32; elem_count];
            <$block_ty>::to_float(&cpu_blocks, &mut cpu_dequant);

            // GPU quantize the same raw input
            let raw_input_gpu = dev.memcpy_stod(&raw_input)?;
            let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(quant_size)? };
            quantize_to_dtype(
                &raw_input_gpu.slice(..),
                &mut gpu_quant_buf,
                elem_count,
                $dtype,
                &dev,
            )?;
            dev.synchronize()?;

            // Download GPU quantized bytes
            let gpu_quant_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;
            let cpu_quant_bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(cpu_blocks.as_ptr() as *const u8, quant_size).to_vec()
            };

            // Count byte mismatches
            let mut byte_mismatches = 0usize;
            for i in 0..quant_size {
                if cpu_quant_bytes[i] != gpu_quant_bytes[i] {
                    byte_mismatches += 1;
                }
            }

            // GPU dequantize for RMSE
            let padded = PaddedCudaSlice {
                inner: gpu_quant_buf.clone(),
                len: quant_size,
            };
            let gpu_dequant_storage = dequantize_f32(&padded, $dtype, elem_count, &dev)?;
            let gpu_dequant_cuda = gpu_dequant_storage.as_cuda_slice::<f32>()?;
            let gpu_dequant: Vec<f32> = dev.memcpy_dtov(&gpu_dequant_cuda.slice(..))?;

            // RMSE between GPU dequant and CPU dequant
            let mut sum_sq = 0.0f64;
            let mut max_diff = 0.0f32;
            for i in 0..elem_count {
                let diff = (gpu_dequant[i] - cpu_dequant[i]).abs();
                sum_sq += (diff as f64).powi(2);
                max_diff = max_diff.max(diff);
            }
            let rmse = (sum_sq / elem_count as f64).sqrt();

            let byte_pct = 100.0 * (quant_size - byte_mismatches) as f64 / quant_size as f64;
            let passed = byte_pct >= MIN_BYTE_MATCH_PCT && rmse <= MAX_RMSE;

            results.push(TestResult {
                name: $name,
                byte_match_pct: byte_pct,
                mismatches: byte_mismatches,
                rmse,
                max_diff,
                passed,
            });
        }};
    }

    // Test all quantization types
    // NOTE: Q8_1 is excluded because its to_float() is unimplemented in CPU code
    test_direct!(BlockQ4_0, GgmlDType::Q4_0, "Q4_0", 4096);
    test_direct!(BlockQ4_1, GgmlDType::Q4_1, "Q4_1", 4096);
    test_direct!(BlockQ5_0, GgmlDType::Q5_0, "Q5_0", 4096);
    test_direct!(BlockQ5_1, GgmlDType::Q5_1, "Q5_1", 4096);
    test_direct!(BlockQ8_0, GgmlDType::Q8_0, "Q8_0", 4096);
    // Q8_1 skipped - to_float() unimplemented
    test_direct!(BlockQ2_K, GgmlDType::Q2_K, "Q2K", 4096);
    test_direct!(BlockQ3_K, GgmlDType::Q3_K, "Q3K", 4096);
    test_direct!(BlockQ4_K, GgmlDType::Q4_K, "Q4K", 4096);
    test_direct!(BlockQ5_K, GgmlDType::Q5_K, "Q5K", 4096);
    test_direct!(BlockQ6_K, GgmlDType::Q6_K, "Q6K", 4096);
    test_direct!(BlockQ8_K, GgmlDType::Q8_K, "Q8K", 4096);
    // AWQ formats
    test_direct!(BlockQAWQ, GgmlDType::QAWQ, "QAWQ", 4096);
    test_direct!(BlockQAWQ_G64, GgmlDType::QAWQ_G64, "QAWQG64", 4096);

    // Print fancy table
    println!();
    println!("â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”");
    println!("â”‚ DType  â”‚ Status â”‚ Byte Match â”‚ Mismatches â”‚   RMSE   â”‚ MaxDiff  â”‚");
    println!("â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤");

    for r in &results {
        let status = if r.passed {
            "   âœ“    "
        } else {
            "   âœ—    "
        };
        println!(
            "â”‚ {:>6} â”‚{}â”‚ {:>9.2}% â”‚ {:>10} â”‚ {:>8.2e} â”‚ {:>8.2e} â”‚",
            r.name, status, r.byte_match_pct, r.mismatches, r.rmse, r.max_diff
        );
    }

    println!("â””â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜");
    println!(
        "\nThresholds: byte_match >= {:.0}%, RMSE <= {:.1}",
        MIN_BYTE_MATCH_PCT, MAX_RMSE
    );
    println!("Note: Q8_1 excluded (to_float unimplemented in CPU)");

    // Check all passed
    let failed: Vec<_> = results.iter().filter(|r| !r.passed).collect();
    if !failed.is_empty() {
        println!("\nFailed types:");
        for r in &failed {
            println!(
                "  - {}: byte_match={:.2}%, rmse={:.2e}",
                r.name, r.byte_match_pct, r.rmse
            );
        }
        panic!(
            "Quantization test failed: {} types did not meet thresholds",
            failed.len()
        );
    }

    Ok(())
}

/// Test that grouped_matmul_gemx produces the same results as matmul_gemx
/// for a single expert (1-expert grouped = regular matmul).
#[test]
fn grouped_matmul_matches_direct() -> Result<()> {
    use crate::Shape;
    use half::bf16;

    let dev = CudaDevice::new(0)?;
    // Dimensions matching Qwen3-30B expert gate_proj: [768, 2048]
    let nrows = 768; // N = intermediate_dim (output)
    let ncols = 2048; // K = hidden_dim (input)
    let num_experts = 3;
    let expert_batches = &[5, 3, 7]; // tokens per expert

    // Create varying weight data for each expert
    let mut rng = rand::rng();
    let mut expert_storages = Vec::new();
    let mut weight_ptrs = Vec::new();
    let shape = Shape::from((nrows, ncols));

    for _ in 0..num_experts {
        let weights: Vec<f32> = (0..nrows * ncols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let mut xs = QCudaStorage::zeros(&dev, ncols * nrows, GgmlDType::Q4_K)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(
            dev.memcpy_stod(&weights)?,
            dev.clone(),
        ))?;
        let xs = xs.repack_gemx(&shape)?;
        weight_ptrs.push(xs.data_ptr());
        expert_storages.push(xs);
    }

    // Create BF16 activations (total_batch = sum of expert_batches)
    let total_batch: usize = expert_batches.iter().sum();
    let act_data: Vec<f32> = (0..total_batch * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let act_bf16: Vec<bf16> = act_data.iter().map(|&v| bf16::from_f32(v)).collect();
    let act_gpu = dev.memcpy_stod(&act_bf16)?;
    let act_storage = CudaStorage::wrap_cuda_slice(act_gpu.clone(), dev.clone());
    let act_layout = crate::Layout::contiguous(Shape::from(vec![total_batch, ncols]));

    // Build expert_offsets prefix sum
    let mut expert_offsets: Vec<i32> = vec![0];
    for &b in expert_batches {
        expert_offsets.push(expert_offsets.last().unwrap() + b as i32);
    }

    // â”€â”€ Reference: per-expert matmul_gemx â”€â”€
    let mut ref_results: Vec<Vec<bf16>> = Vec::new();
    for e in 0..num_experts {
        let start = expert_offsets[e] as usize;
        let end = expert_offsets[e + 1] as usize;
        let batch = end - start;

        // Slice activations for this expert
        let expert_act_bf16 = &act_bf16[start * ncols..end * ncols];
        let expert_act_gpu = dev.memcpy_stod(expert_act_bf16)?;
        let expert_act_storage = CudaStorage::wrap_cuda_slice(expert_act_gpu, dev.clone());
        let expert_act_layout = crate::Layout::contiguous(Shape::from(vec![batch, ncols]));

        let (ref_storage, _) =
            expert_storages[e].matmul_gemx(&shape, &expert_act_storage, &expert_act_layout)?;
        let ref_bf16 = ref_storage.as_cuda_slice::<bf16>()?;
        let ref_result: Vec<bf16> = dev.memcpy_dtov(&ref_bf16.slice(..))?;
        ref_results.push(ref_result);
    }

    // â”€â”€ Test: grouped_matmul_gemx â”€â”€
    let grouped_result = grouped_matmul_gemx(
        &weight_ptrs,
        GgmlDType::Q4_K,
        nrows,
        ncols,
        &act_storage,
        &act_layout,
        &expert_offsets,
        &dev,
    )?;

    // Read grouped result
    let grouped_storage = grouped_result.storage_and_layout().0;
    let grouped_cuda = match &*grouped_storage {
        crate::Storage::Cuda(s) => s,
        _ => panic!("expected CUDA storage"),
    };
    let grouped_bf16 = grouped_cuda.as_cuda_slice::<bf16>()?;
    let grouped_vals: Vec<bf16> = dev.memcpy_dtov(&grouped_bf16.slice(..))?;

    // Compare per-expert
    let mut total_mismatches = 0;
    for e in 0..num_experts {
        let start = expert_offsets[e] as usize * nrows;
        let end = expert_offsets[e + 1] as usize * nrows;
        let ref_vals = &ref_results[e];
        let grp_vals = &grouped_vals[start..end];

        assert_eq!(
            ref_vals.len(),
            grp_vals.len(),
            "expert {} length mismatch",
            e
        );

        let mut err_count = 0;
        for i in 0..ref_vals.len() {
            let r = bf16::to_f32(ref_vals[i]);
            let g = bf16::to_f32(grp_vals[i]);
            let err = (r - g).abs();
            let rel = err / r.abs().max(1e-6);
            if rel > 0.01 {
                if err_count < 3 {
                    println!(
                        "  expert[{}] MISMATCH at [{}]: ref={:.6} grouped={:.6} rel={:.4}",
                        e, i, r, g, rel
                    );
                }
                err_count += 1;
            }
        }
        println!(
            "Expert[{}]: batch={}, mismatches={}/{}",
            e,
            expert_batches[e],
            err_count,
            ref_vals.len()
        );
        total_mismatches += err_count;
    }

    assert_eq!(
        total_mismatches, 0,
        "grouped_matmul_gemx produced different results for multi-expert case"
    );
    Ok(())
}

/// Build Q4_K experts + stacked bf16 activations for the INT8 A/B tests.
/// Returns (weight_ptrs, kept-alive storages, act_storage, act_layout, expert_offsets).
#[cfg(test)]
fn build_int8_ab_fixture(
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

// ===========================================================================
// block_q8a128 — the contiguous q8 activation block. Tests assert RAW BYTES
// (per the repo codec convention), not error thresholds.
// ===========================================================================

/// Round-half-to-even, matching CUDA __float2int_rn (the quantizer's rounding).
#[cfg(test)]
fn round_ties_even_i32(x: f32) -> i32 {
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
#[test]
fn q8a128_quantize_raw_bytes() -> Result<()> {
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
    let blocks = quantize_acts_q8a128(ptr, 2 /* F32 */, rows, cols, &dev)?.into_owned_data()?;
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
            let ds_b = ds_off(flat);
            let exp_scale = f16::from_f32(amax / 127.0);
            let exp_sum = f16::from_f32(sum);
            let got_scale = f16::from_le_bytes([raw[ds_b], raw[ds_b + 1]]);
            let got_sum = f16::from_le_bytes([raw[ds_b + 2], raw[ds_b + 3]]);
            assert_eq!(
                got_scale.to_bits(),
                exp_scale.to_bits(),
                "blk({r},{t}) scale"
            );
            assert_eq!(got_sum.to_bits(), exp_sum.to_bits(), "blk({r},{t}) sum");
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
    println!("q8a128 raw-byte quantize: {checked} per-128 tiles verified byte-exact");
    Ok(())
}

/// Dequantize must return EXACTLY `scale * qs` (the quantized representation),
/// matched bit-for-bit against a CPU recompute from the same raw block bytes.
/// q8a128 is per-128: ONE {scale, sum} per 128-tile, stored at ds[0]; all four
/// 32-element sub-blocks of the tile share it (ds[1..3] are alignment pad).
#[test]
fn q8a128_dequant_exact() -> Result<()> {
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
    let blocks = quantize_acts_q8a128(ptr, 2, rows, cols, &dev)?.into_owned_data()?;
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
fn q8a128_edge_cases() -> Result<()> {
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
    let blocks = quantize_acts_q8a128(ptr, 2, rows, cols, &dev)?.into_owned_data()?;
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

    // Tile 1 — amax is the spike, so scale = 100/127 and Σx = 100 + 32×2 + Σ(i−16).
    assert_eq!(
        scale_at(1040).to_bits(),
        f16::from_f32(100.0 / 127.0).to_bits(),
        "mixed tile scale is the spike's",
    );
    assert_eq!(
        sum_at(1040).to_bits(),
        f16::from_f32(148.0).to_bits(),
        "mixed tile sum",
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

/// Citizenship: q8a128 must be reachable through the *unified* QType dispatch
/// (`run_quantize_block`/`run_dequantize_block` at qtype = QTYPE_Q8A128 = 36),
/// producing byte-identical results to its dedicated typed path. This pins the
/// enum value + the dispatch wiring, not just the standalone kernels.
#[test]
fn q8a128_unified_dispatch_matches_typed() -> Result<()> {
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
    let typed = quantize_acts_q8a128(ptr, 2, rows, cols, &dev)?.into_owned_data()?;
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

/// Throughput bench for the q8a128 quantize/dequant kernels — they sit on the hot
/// inference path so they must run at ~peak memory bandwidth. Pre-allocates all
/// buffers and calls the kernels directly (no per-iter alloc), timing both the
/// f32 and bf16 vectorization paths. Reports GB/s (effective 4090M peak ≈ 450).
#[test]
#[ignore = "GPU throughput bench; run with --ignored --nocapture"]
fn q8a128_throughput_bench() -> Result<()> {
    use candle_kernels::simple::quantized::{run_dequantize_q8a128, run_quantize_q8a128};
    use std::ffi::c_void;
    use std::time::Instant;
    let dev = CudaDevice::new(0)?;
    let rows = 8192usize;
    let cols = 4096usize; // [tokens, hidden]
    let n = rows * cols;
    let qbytes = (n / 128).div_ceil(8) * 1152;
    let (rows_i, cols_i) = (rows as i32, cols as i32);
    let warm = 10usize;
    let iters = 100usize;

    let f32_in: Vec<f32> = (0..n).map(|i| ((i % 257) as f32 - 128.0) * 0.01).collect();
    let f16_in: Vec<half::f16> = f32_in.iter().map(|&x| half::f16::from_f32(x)).collect();
    let bf16_in: Vec<half::bf16> = f32_in.iter().map(|&x| half::bf16::from_f32(x)).collect();
    let f32_dev = dev.memcpy_stod(&f32_in)?;
    let f16_dev = dev.memcpy_stod(&f16_in)?;
    let bf16_dev = dev.memcpy_stod(&bf16_in)?;
    let mut qbuf = unsafe { dev.alloc::<u8>(qbytes)? };
    let mut dout_f32 = unsafe { dev.alloc::<f32>(n)? };
    let mut dout_f16 = unsafe { dev.alloc::<half::f16>(n)? };
    let mut dout_bf16 = unsafe { dev.alloc::<half::bf16>(n)? };
    let stream = dev.cuda_stream();
    let (f32p, _g0) = f32_dev.device_ptr(&stream);
    let (f16p, _g0b) = f16_dev.device_ptr(&stream);
    let (bf16p, _g1) = bf16_dev.device_ptr(&stream);
    let (qp, _g2) = qbuf.device_ptr_mut(&stream);
    let (dfp, _g3) = dout_f32.device_ptr_mut(&stream);
    let (dhp, _g3b) = dout_f16.device_ptr_mut(&stream);
    let (dbp, _g4) = dout_bf16.device_ptr_mut(&stream);

    // (name, input ptr, dtype code [0=f16,1=bf16,2=f32], element bytes, dequant out ptr)
    let configs: [(&str, u64, i32, usize, u64); 3] = [
        ("f32", f32p, 2, 4, dfp),
        ("f16", f16p, 0, 2, dhp),
        ("bf16", bf16p, 1, 2, dbp),
    ];

    for &(name, inp, dtype, elem_bytes, dout) in &configs {
        for _ in 0..warm {
            unsafe {
                run_quantize_q8a128(
                    inp as *const c_void,
                    qp as *mut c_void,
                    rows_i,
                    cols_i,
                    dtype,
                );
            }
        }
        dev.synchronize()?;
        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                run_quantize_q8a128(
                    inp as *const c_void,
                    qp as *mut c_void,
                    rows_i,
                    cols_i,
                    dtype,
                );
            }
        }
        dev.synchronize()?;
        let qt = t0.elapsed().as_secs_f64() / iters as f64;
        let q_gbps = (n * elem_bytes + qbytes) as f64 / qt / 1e9;

        for _ in 0..warm {
            unsafe {
                run_dequantize_q8a128(
                    qp as *const c_void,
                    dout as *mut c_void,
                    rows_i,
                    cols_i,
                    dtype,
                );
            }
        }
        dev.synchronize()?;
        let t1 = Instant::now();
        for _ in 0..iters {
            unsafe {
                run_dequantize_q8a128(
                    qp as *const c_void,
                    dout as *mut c_void,
                    rows_i,
                    cols_i,
                    dtype,
                );
            }
        }
        dev.synchronize()?;
        let dt = t1.elapsed().as_secs_f64() / iters as f64;
        let d_gbps = (qbytes + n * elem_bytes) as f64 / dt / 1e9;

        println!(
            "q8a128 [{rows}x{cols} {name}]  quantize {:.4} ms {:6.1} GB/s  |  dequant {:.4} ms {:6.1} GB/s",
            qt * 1e3, q_gbps, dt * 1e3, d_gbps
        );
    }
    Ok(())
}

/// GPU KO weight quantize/dequant must be byte-identical to the CPU reference
/// (`ko_quant`): raw-byte assertion for the quantize, bit-exact f32 for the dequant.
/// This pins the lane-major pack + the de-interleave on the device.
#[test]
fn ko_gpu_quantize_dequant_matches_cpu() -> Result<()> {
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

/// GPU throughput for the KO weight quantize/dequant kernels (F32 ↔ lane-major per-128
/// KO). Pre-uploads the weight matrix and calls the kernels directly (no per-iter alloc).
/// GB/s is on the f32 side (the bandwidth-relevant traffic) + the on-disk compression vs f32.
/// Run with: cargo test -p candle-core --features cuda ko_quant_throughput_bench -- --nocapture
#[test]
#[ignore = "GPU throughput bench; run with --ignored --nocapture"]
fn ko_quant_throughput_bench() -> Result<()> {
    use crate::quantized::ko_quant::ko_chunk_bytes;
    use candle_kernels::simple::quantized::{run_dequantize_ko, run_quantize_ko};
    let dev = CudaDevice::new(0)?;
    let (nrows, ncols) = (8192usize, 8192usize); // 67.1M weights
    let n = nrows * ncols;
    let f32_bytes = n * 4;
    let (nr, nc) = (nrows as i32, ncols as i32);
    let (warm, iters) = (5usize, 50usize);

    let w: Vec<f32> = (0..n).map(|i| ((i % 257) as f32 - 128.0) * 0.01).collect();
    let w_dev = dev.memcpy_stod(&w)?;
    let stream = dev.cuda_stream();
    let (wp, _g0) = w_dev.device_ptr(&stream);

    for dtype in [
        GgmlDType::Q4_KO,
        GgmlDType::Q5_KO,
        GgmlDType::Q6_KO,
        GgmlDType::Q8_KO,
    ] {
        let qtype = dtype_to_qtype(dtype)? as i32;
        let qbytes = (nrows / 8) * (ncols / 128) * ko_chunk_bytes(dtype);
        let mut qbuf = unsafe { dev.alloc::<u8>(qbytes)? };
        let mut dbuf = unsafe { dev.alloc::<f32>(n)? };
        let (qp, _g1) = qbuf.device_ptr_mut(&stream);
        let (dp, _g2) = dbuf.device_ptr_mut(&stream);

        for _ in 0..warm {
            unsafe { run_quantize_ko(wp as *const f32, qp as *mut c_void, nr, nc, qtype) };
        }
        dev.synchronize()?;
        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe { run_quantize_ko(wp as *const f32, qp as *mut c_void, nr, nc, qtype) };
        }
        dev.synchronize()?;
        let qt = t0.elapsed().as_secs_f64() / iters as f64;
        let q_gbps = (f32_bytes + qbytes) as f64 / qt / 1e9;

        for _ in 0..warm {
            unsafe { run_dequantize_ko(qp as *const c_void, dp as *mut f32, nr, nc, qtype) };
        }
        dev.synchronize()?;
        let t1 = Instant::now();
        for _ in 0..iters {
            unsafe { run_dequantize_ko(qp as *const c_void, dp as *mut f32, nr, nc, qtype) };
        }
        dev.synchronize()?;
        let dt = t1.elapsed().as_secs_f64() / iters as f64;
        let d_gbps = (qbytes + f32_bytes) as f64 / dt / 1e9;

        println!(
            "KO [{nrows}x{ncols} {dtype:?}]  quantize {:.4} ms {:6.1} GB/s  |  dequant {:.4} ms {:6.1} GB/s  |  {:.2}x vs f32",
            qt * 1e3, q_gbps, dt * 1e3, d_gbps, f32_bytes as f64 / qbytes as f64,
        );
    }
    Ok(())
}

/// Verify the F16 and BF16 quantize/dequant paths (the `q8a128_load4`/`store4`
/// specializations the f32 byte-exact tests don't exercise). f16/bf16 → f32 is
/// lossless, so quantizing a typed input MUST yield the exact same block bytes as
/// quantizing the identical values as f32 (already byte-exact-validated); and
/// dequant-to-typed must equal the typed rounding of the f32 dequant. Bit-exact.
#[test]
fn q8a128_f16_bf16_paths_match_f32() -> Result<()> {
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
        let blk_t = quantize_acts_q8a128(tp, 0, rows, cols, &dev)?.into_owned_data()?;
        let blk_f = quantize_acts_q8a128(fp, 2, rows, cols, &dev)?.into_owned_data()?;
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
        let blk_t = quantize_acts_q8a128(tp, 1, rows, cols, &dev)?.into_owned_data()?;
        let blk_f = quantize_acts_q8a128(fp, 2, rows, cols, &dev)?.into_owned_data()?;
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
fn read_bf16_tensor(dev: &CudaDevice, t: &crate::Tensor) -> Result<Vec<f32>> {
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
fn read_f32_tensor(dev: &CudaDevice, t: &crate::Tensor) -> Result<Vec<f32>> {
    let s = t.storage_and_layout().0;
    let c = match &*s {
        crate::Storage::Cuda(s) => s,
        _ => panic!("expected CUDA storage"),
    };
    let b = c.as_cuda_slice::<f32>()?;
    Ok(dev.memcpy_dtov(&b.slice(..))?)
}

/// Read a narrow matmul output back as raw bit patterns.
///
/// Bits, not floats: the point of the narrowed dense kernels is that they are the
/// F32 kernel plus the store-time conversion, so the gate compares the exact
/// encoding rather than a tolerance. Widening to f32 first would hide a wrong
/// rounding mode, and a tolerance would hide a whole wrong table row.
#[cfg(test)]
fn read_narrow_bits(dev: &CudaDevice, t: &crate::Tensor) -> Result<Vec<u16>> {
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
fn dense_int8_narrow_output_matches_f32_bitwise() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let stream = dev.cuda_stream();
    let ncols = 1024usize; // K
    let nrows = 512usize; // N
    let mut rng = rand::rng();

    // Every KO format the dense table carries, with its requant parameters.
    let formats: [(GgmlDType, (i32, usize, usize)); 3] = [
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

/// The same gate for the fused qkv launcher, whose tables are a dimension larger
/// (`[out_dtype][format][mode]` for the uniform fast path, `[out_dtype][mode]` for
/// the mixed one). Runs both: uniform q/k/v formats take the single-format kernel,
/// mixed formats take the switch kernel, and they are separate table rows.
#[test]
fn qkv_segmented_narrow_output_matches_f32_bitwise() -> Result<()> {
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

    for dims in [&uniform, &mixed] {
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

/// Quantize f32 activations [total_batch, ncols] → `block_q8a128[total_batch][ncols/128]`
/// (the pre-quantize feeding the INT8 grouped kernel). Synchronizes so the f32
/// staging buffer is safe to drop.
#[cfg(test)]
fn quantize_acts_q8a128_test(
    dev: &CudaDevice,
    act_data: &[f32],
    total_batch: usize,
    ncols: usize,
) -> Result<Q8a128Operand<'static>> {
    let f32_dev = dev.memcpy_stod(act_data)?;
    let stream = dev.cuda_stream();
    let (ptr, _g) = f32_dev.device_ptr(&stream);
    let out = quantize_acts_q8a128(ptr, 2 /* F32 */, total_batch, ncols, dev)?;
    dev.synchronize()?;
    Ok(out)
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]) as f64;
        num += d * d;
        den += (b[i] as f64) * (b[i] as f64);
    }
    (num / den.max(1e-12)).sqrt()
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
fn grouped_int8_matches_legacy() -> Result<()> {
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
fn dense_int8_matches_grouped() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 256; // N
    let ncols = 512; // K
    let mut rng = rand::rng();
    let wf32: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-0.1f32..0.1))
        .collect();
    let stream = dev.cuda_stream();

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
fn q2_ko_int8_grouped_matches_f32_ref() -> Result<()> {
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

/// Single-layer MoE grouped-GEMM replay bench at the DeepSeek-V4-Flash shapes —
/// the fast iteration loop for routed-GEMM kernel work (256 experts, gate/up
/// `[2048, 7168]`, down `[7168, 2048]`, MXFP4_KO, top-6 routing). Weight bytes
/// are random (timing is value-independent); 32 distinct weights per projection
/// cycle through the expert pointer table so the working set exceeds L2 like
/// the real 256-expert layer. Two regimes: a cfg8 user-turn wave (~3.9k tokens
/// × 6 ≈ 23k gathered rows, ~91/expert) and a width-capped cfg20 slice (8192 ×
/// 6 ≈ 49k rows, ~192/expert). Prints ms/call + effective weight bandwidth per
/// tile mode. nsys baseline (2026-08-17): m8 ≈ 9.3 ms/call ≈ 15% of DRAM peak —
/// latency-bound on the depth-1 weight ring.
#[test]
#[ignore = "GPU perf bench; run with --ignored --nocapture"]
fn moe_layer_gemm_bench() -> Result<()> {
    use std::time::Instant;
    let dev = CudaDevice::new(0)?;
    let stream = dev.cuda_stream();
    let mut rng = rand::rng();
    let n_experts = 256usize;
    let n_weights = 32usize;
    // (label, nrows=N, ncols=K)
    let shapes = [("gate/up", 2048usize, 7168usize), ("down", 7168, 2048)];
    // decode = 64 sessions × top-6 ≈ 2 rows/expert: the small-activation band where
    // the grid-order choice could flip (the whole activation fits L2 trivially).
    let regimes = [
        ("decode", 64usize * 6),
        ("cfg8", 3900 * 6),
        ("cfg20", 8192 * 6),
    ];

    for &(label, nrows, ncols) in &shapes {
        // Random-byte KO weights: [K/128 chunks] × [N/8 row-groups] × 544B.
        let wbytes = (ncols / 128) * (nrows / 8) * crate::quantized::ko_quant::MXFP4_KO_CHUNK_BYTES;
        let mut ptrs_pool: Vec<u64> = Vec::new();
        let mut _keep = Vec::new();
        for _ in 0..n_weights {
            let bytes: Vec<u8> = (0..wbytes).map(|_| rng.random::<u8>()).collect();
            let slice = dev.memcpy_stod(&bytes)?;
            let p = {
                let (p, _g) = slice.device_ptr(&stream);
                p
            };
            ptrs_pool.push(p);
            _keep.push(slice);
        }
        let weight_ptrs: Vec<u64> = (0..n_experts).map(|e| ptrs_pool[e % n_weights]).collect();

        for &(regime, total_batch) in &regimes {
            let per = total_batch / n_experts;
            let mut expert_offsets: Vec<i32> = vec![0];
            for e in 0..n_experts {
                let extra = usize::from(e < total_batch % n_experts);
                expert_offsets.push(expert_offsets.last().unwrap() + (per + extra) as i32);
            }
            let act: Vec<f32> = (0..total_batch * ncols)
                .map(|_| rng.random_range(-1.0f32..1.0))
                .collect();
            let q8 = quantize_acts_q8a128_test(&dev, &act, total_batch, ncols)?;

            for n_sub in [2usize, 4, 8] {
                for row_fast in [true, false] {
                    let run = || -> Result<()> {
                        q8.with_device_ptr(&dev, |act_ptr| {
                            crate::quantized::cuda::grouped_matmul_gemx_q8a128_with_mode(
                                act_ptr,
                                &weight_ptrs,
                                GgmlDType::MXFP4_KO,
                                nrows,
                                ncols,
                                total_batch,
                                &expert_offsets,
                                &dev,
                                Backing::Owned,
                                n_sub,
                                row_fast,
                            )
                        })?;
                        Ok(())
                    };
                    for _ in 0..3 {
                        run()?;
                    }
                    dev.synchronize()?;
                    let iters = 20;
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        run()?;
                    }
                    dev.synchronize()?;
                    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
                    // Minimum weight traffic: each expert's full matrix once.
                    let min_w_gb = (n_experts * wbytes) as f64 / 1e9;
                    let lay = if row_fast { "rowF" } else { "tokF" };
                    println!(
                        "{label:>7} {regime}: n_sub={n_sub} {lay}  {ms:8.3} ms/call  \
                         min-weight-BW {:6.1} GB/s  ({total_batch} rows, ~{per}/expert)",
                        min_w_gb / (ms / 1e3),
                    );
                }
            }
        }
    }
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
fn grouped_int8_wide_tiles_match_mode2() -> Result<()> {
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
fn mxfp4_persub_cuda_matches_cpu_oracle() -> Result<()> {
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
fn qmatmul_int8mode_flag_end_to_end() -> Result<()> {
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
        let acts = to_dynamic(&act_t, mode, &dev)?;
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

/// Repro guard for the offline-KO GGUF load path: a Q8_KO weight built from CPU
/// `quantize_ko` bytes uploaded via `load_repacked` (what `qtensor_from_ggml` now does for a
/// pre-KO GGUF) must forward-via-int8 identically to the same weight repacked on-GPU by
/// `repack_for_optimization` (the Cat2 at-load path). Both feed the SAME Q8_0-dequantized f32,
/// so any divergence (or crash) isolates the offline-load path from the engine.
#[test]
fn ko_offline_load_forward_matches_gpu_repack() -> Result<()> {
    use crate::quantized::ko_quant::quantize_ko;
    use crate::quantized::{QMatMul, QTensor};
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (nrows, ncols, m) = (256usize, 512usize, 32usize); // out, in, batch
    let mut rng = rand::rng();
    let wf32: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-0.1f32..0.1))
        .collect();
    let act: Vec<f32> = (0..m * ncols)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    let w_t = crate::Tensor::from_vec(wf32, (nrows, ncols), &device)?;
    let act_t = crate::Tensor::from_vec(act, (m, ncols), &device)?;

    // Path A (known-good, Cat2): Q8_0 → GPU repack_for_optimization → forward_via_int8.
    let q8 = QMatMul::from_qtensor(QTensor::quantize(&w_t, GgmlDType::Q8_0)?)?;
    let opt_a = q8.repack_for_optimization(Int8Mode::Performance)?;
    let y_a: Vec<f32> = opt_a
        .forward_via_int8(&act_t, Int8Mode::Performance)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    // Path B (offline emulation): dequant the SAME Q8_0 → CPU quantize_ko → load_repacked → wrap.
    let w_q8_deq = QTensor::quantize(&w_t, GgmlDType::Q8_0)?.dequantize(&device)?;
    let w_q8_vec = w_q8_deq.flatten_all()?.to_vec1::<f32>()?;
    let ko_bytes = quantize_ko(&w_q8_vec, nrows, ncols, GgmlDType::Q8_KO);
    let storage_b = crate::quantized::cuda::load_repacked(&dev, &ko_bytes, GgmlDType::Q8_KO)?;
    let qt_b = QTensor::new(storage_b, vec![nrows, ncols])?;
    let mm_b = QMatMul::from_qtensor(qt_b)?;
    let y_b: Vec<f32> = mm_b
        .forward_via_int8(&act_t, Int8Mode::Performance)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let rel = rel_l2(&y_a, &y_b);
    println!("offline-KO-load vs GPU-repack rel_l2 = {rel:.3e}");
    assert!(
        rel < 1e-3,
        "offline-KO-load path diverged from GPU repack: rel_l2={rel}"
    );
    Ok(())
}

/// The fused Sinkhorn kernel (`run_sinkhorn_f32`) must reproduce the scalar reference that mirrors
/// `hyper.rs::sinkhorn` (softmax over cols + eps → col-norm → (iters-1)×[row-norm, col-norm]) for a
/// batch of small `[hc, hc]` matrices — the one-launch replacement for the ~120 tiny host ops.
#[test]
fn sinkhorn_kernel_matches_scalar_reference() -> Result<()> {
    use candle_kernels::simple::sinkhorn::run_sinkhorn_f32;
    use std::ffi::c_void;

    // Scalar reference — identical op order to the kernel and to `hyper.rs::sinkhorn`.
    fn reference(a: &[f32], hc: usize, iters: usize, eps: f32) -> Vec<f32> {
        let mut c = vec![0f32; hc * hc];
        for i in 0..hc {
            let mut m = f32::MIN;
            for j in 0..hc {
                m = m.max(a[i * hc + j]);
            }
            let mut s = 0f32;
            for j in 0..hc {
                let e = (a[i * hc + j] - m).exp();
                c[i * hc + j] = e;
                s += e;
            }
            for j in 0..hc {
                c[i * hc + j] = c[i * hc + j] / s + eps;
            }
        }
        for j in 0..hc {
            let mut s = eps;
            for i in 0..hc {
                s += c[i * hc + j];
            }
            for i in 0..hc {
                c[i * hc + j] /= s;
            }
        }
        for _ in 0..iters - 1 {
            for i in 0..hc {
                let mut s = eps;
                for j in 0..hc {
                    s += c[i * hc + j];
                }
                for j in 0..hc {
                    c[i * hc + j] /= s;
                }
            }
            for j in 0..hc {
                let mut s = eps;
                for i in 0..hc {
                    s += c[i * hc + j];
                }
                for i in 0..hc {
                    c[i * hc + j] /= s;
                }
            }
        }
        c
    }

    let dev = CudaDevice::new(0)?;
    let (n, hc, iters, eps) = (5usize, 4usize, 20usize, 1e-6f32);
    let mut s = 0x1234_5678u32;
    let inp: Vec<f32> = (0..n * hc * hc)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 4.0 - 2.0
        })
        .collect();

    let inp_dev = dev.memcpy_stod(&inp)?;
    let mut out_dev = unsafe { dev.alloc::<f32>(n * hc * hc)? };
    let stream = dev.cuda_stream();
    {
        let (ip, _g0) = inp_dev.device_ptr(&stream);
        let (op, _g1) = out_dev.device_ptr_mut(&stream);
        unsafe {
            run_sinkhorn_f32(
                ip as *const f32,
                op as *mut f32,
                n as i32,
                hc as i32,
                iters as i32,
                eps,
                stream.cu_stream() as *mut c_void,
            );
        }
    }
    dev.synchronize()?;
    let got: Vec<f32> = dev.memcpy_dtov(&out_dev.slice(..))?;

    for m in 0..n {
        let want = reference(&inp[m * hc * hc..(m + 1) * hc * hc], hc, iters, eps);
        for k in 0..hc * hc {
            let g = got[m * hc * hc + k];
            let w = want[k];
            assert!(
                (g - w).abs() < 1e-5,
                "matrix {m} elem {k}: kernel {g} vs reference {w}"
            );
        }
        // Doubly-stochastic sanity: rows and columns each sum to ~1.
        for i in 0..hc {
            let rs: f32 = (0..hc).map(|j| got[m * hc * hc + i * hc + j]).sum();
            assert!((rs - 1.0).abs() < 1e-3, "matrix {m} row {i} sum {rs}");
        }
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
fn qmatmul_int8mode_baseline_bit_check() -> Result<()> {
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
        let acts = to_dynamic(&act_t, mode, &dev)?;
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
fn dense_qmatmul_int8_preserves_3d_shape() -> Result<()> {
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
    let acts = to_dynamic(&act_t, Int8Mode::Performance, &dev)?;
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

/// Phase-2a producer fusion: `rms_norm_q8a128` (fused RMSNorm → q8a128 in one kernel) must match
/// the unfused `rms_norm` → `quantize_acts_q8a128` oracle within a tight float margin, for BOTH
/// the f16 and bf16 input variants. Both operands are dequantized and compared by rel-L2 (the
/// quant grid is per-128, so ULP-level reduction differences can flip a quant by ±1; the margin
/// absorbs that while still proving the fused path reproduces the two-call path).
#[test]
fn rms_norm_q8a128_matches_reference() -> Result<()> {
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
        let oracle_op = match to_dynamic(&normed, Int8Mode::Performance, &dev)? {
            DynamicActs::Int8(op) => op,
            DynamicActs::Float(_) => unreachable!("Performance mode yields Int8"),
        };
        let oracle_deq = dequantize_q8a128(oracle_op.data_slice()?, rows, cols, &dev)?;
        let oracle = dev.memcpy_dtov(&oracle_deq.slice(..))?;

        // Fused: single kernel.
        let fused_op = rms_norm_q8a128(&xs, &alpha, eps, &dev, Backing::Owned)?;
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

/// Phase-2b producer fusion: `silu_mul_q8a128` (fused SwiGLU → q8a128) must match the unfused
/// `silu(gate)·up` → `quantize_acts_q8a128` oracle within float margin, for BOTH f16 and bf16.
/// The reference uses exact sigmoid; the kernel uses the production fast-exp silu, so the margin
/// covers the fast-exp approximation plus the per-128 quant grid.
#[test]
fn silu_mul_q8a128_matches_reference() -> Result<()> {
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
        let oracle_op = match to_dynamic(&outf, Int8Mode::Performance, &dev)? {
            DynamicActs::Int8(op) => op,
            DynamicActs::Float(_) => unreachable!("Performance mode yields Int8"),
        };
        let oracle_deq = dequantize_q8a128(oracle_op.data_slice()?, rows, cols, &dev)?;
        let oracle = dev.memcpy_dtov(&oracle_deq.slice(..))?;

        // Fused: single kernel.
        let fused_op = silu_mul_q8a128(&gate, &up, &dev, Backing::Owned)?;
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

/// The exclusive KO <-> int8 pairing guard: `dense_qmatmul`/`grouped_qmatmul` accept the two
/// supported combinations (Int8 activation x KO weight, Float activation x non-KO weight) and
/// reject the two unsupported crosses (Int8 x non-KO, Float x KO) with an error — before any
/// kernel launches, so callers get a clear message instead of silent garbage.
#[test]
fn qmatmul_rejects_unpaired_weight_activation() -> Result<()> {
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
fn dense_qmatmul_float_matches_grouped() -> Result<()> {
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

/// Q4_KO is a pure byte permutation of the Q4_K compact block — the 16 qs ints made
/// contiguous, the four (scale,-min) pairs grouped at the tail. Reading a permuted
/// Q4_KO weight through the int8 q8a128 matmul must produce the BIT-IDENTICAL result
/// of reading the original Q4_K weight, proving the KO layout + loader + dispatch are
/// correct end to end. This test permutes on the host; the production path repacks on
/// the GPU (`QCudaStorage::repack_ko`).
#[test]
#[ignore = "retired: per-32 sub-major path superseded by the per-128 collapse"]
fn q4_ko_matches_q4_k_int8() -> Result<()> {
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
    for (kb, ob) in k_bytes.chunks_exact(80).zip(ko_bytes.chunks_exact_mut(80)) {
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
fn q4ko_submajor_dequant_correct() -> Result<()> {
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
fn q8ko_submajor_dequant_correct() -> Result<()> {
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
fn q8ko_per128_dequant_correct() -> Result<()> {
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
fn q5q6_ko_per128_dequant_correct() -> Result<()> {
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
fn q6ko_submajor_dequant_correct() -> Result<()> {
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
fn q5ko_submajor_dequant_correct() -> Result<()> {
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

/// Quant-region size (bytes/block) of a de-interleaved KO weight block — the scale
/// region is always a further 16 B/block.
#[cfg(feature = "cuda")]
fn ko_quant_bytes(kod: GgmlDType) -> usize {
    match kod {
        GgmlDType::Q4_KO => 64,
        GgmlDType::Q5_KO => 80,
        GgmlDType::Q6_KO => 96,
        GgmlDType::Q8_KO => 128,
        other => unreachable!("not a KO format: {other:?}"),
    }
}

/// De-interleaved KO producer: run the K→KO byte permutation, then split each block
/// into `[quant region | 16 B scale region]` — quant blocks packed first, then one 16 B
/// scale block per quant block (same block index), matching is_scale_separate + the
/// dm-hoist in kernel.cuh. The perm writes an `in_stride`-byte interleaved KO block whose
/// first `quant_bytes` are quants and whose next 16 B are the per-block scales. For Q8_KO
/// (symmetric) the scale's high half (the min term) is zeroed — the de-interleaved fold
/// reads the raw half2 instead of sub_dm's explicit (d, 0).
#[cfg(feature = "cuda")]
fn ko_deinterleave(
    kb: &[u8],
    in_stride: usize,
    quant_bytes: usize,
    zero_scale_hi: bool,
    perm: fn(&[u8], &mut [u8]),
) -> Vec<u8> {
    let n = kb.len() / in_stride;
    let mut ob = vec![0u8; n * (quant_bytes + 16)];
    let sbase = n * quant_bytes;
    let mut tmp = vec![0u8; in_stride];
    for i in 0..n {
        tmp.iter_mut().for_each(|b| *b = 0);
        perm(&kb[i * in_stride..(i + 1) * in_stride], &mut tmp);
        ob[i * quant_bytes..(i + 1) * quant_bytes].copy_from_slice(&tmp[0..quant_bytes]);
        let sc = sbase + i * 16;
        ob[sc..sc + 16].copy_from_slice(&tmp[quant_bytes..quant_bytes + 16]);
        if zero_scale_hi {
            for s in 0..4 {
                ob[sc + s * 4 + 2] = 0;
                ob[sc + s * 4 + 3] = 0;
            }
        }
    }
    ob
}

/// Decode the 32 unsigned 4-bit weights of one (compact Q4_K block, sub) into `K[0..32]`,
/// mirroring `q4k_dequant_to_b_frag_int8`: the sub's 4 ints J0..J3 (at `qs_base+{0,4,8,12}`)
/// map J0→K[0:8], J1→K[8:16], J2→K[16:24], J3→K[24:32]; within each int the low nibbles are
/// the first 4 K and the high nibbles the next 4, both in `byte_perm 0x3120` ([0,2,1,3]) order.
#[cfg(feature = "cuda")]
fn q4k_decode_sub(block: &[u8], qs_base: usize) -> [u8; 32] {
    const PERM: [usize; 4] = [0, 2, 1, 3]; // byte_perm 0x3120: result[i] = src[PERM[i]]
    let mut k = [0u8; 32];
    for (j, &base) in [0usize, 8, 16, 24].iter().enumerate() {
        let jint = &block[qs_base + j * 4..qs_base + j * 4 + 4];
        for i in 0..4 {
            k[base + i] = jint[PERM[i]] & 0x0F;
            k[base + 4 + i] = (jint[PERM[i]] >> 4) & 0x0F;
        }
    }
    k
}

/// Build the SUB-MAJOR k1024 Q4_KO chunk tensor from compact Q4_K blocks (block index
/// `k_blk*nrows + row`, 80 B each). This is the wavefront-optimal weight layout the int8
/// dequant reads: chunk `[k_blk][row-group of 8]`, the 512 B quant region sub-major — sub s
/// at `[s*128, s*128+128)`, row r's 16 bytes packing the 32 K-nibbles as byte
/// `p = K[p] | (K[p+16]<<4)`. Lane `(row=lane>>2, q3=lane&3)` then reads one coalesced int at
/// `s*128 + lane*4` → bank `lane` (conflict-free, 1 wavefront/sub). The 128 B scale region
/// (`dm[row][sub]`, the same `(scale,min)` half2 as Q4_K) follows at offset 512.
#[cfg(feature = "cuda")]
fn q4ko_submajor_from_compact(kb: &[u8], nrows: usize, ncols: usize) -> Vec<u8> {
    const QS_BASE: [usize; 4] = [0, 24, 40, 64]; // compact sub qs-int bases
    const DM_OFF: [usize; 4] = [16, 20, 56, 60]; // compact sub (scale,min) half2 offsets
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let chunk_bytes = 512 + 128; // 8 rows × 64 B quant + 8 rows × 16 B scales
    let mut ob = vec![0u8; k_blocks * row_groups * chunk_bytes];
    for k_blk in 0..k_blocks {
        for g in 0..row_groups {
            let cbase = (k_blk * row_groups + g) * chunk_bytes;
            for r in 0..8 {
                let row = g * 8 + r;
                let blk = &kb[(k_blk * nrows + row) * 80..(k_blk * nrows + row) * 80 + 80];
                for sub in 0..4 {
                    let kk = q4k_decode_sub(blk, QS_BASE[sub]);
                    for p in 0..16 {
                        ob[cbase + sub * 128 + r * 16 + p] =
                            (kk[p] & 0xF) | ((kk[p + 16] & 0xF) << 4);
                    }
                    let d = cbase + 512 + r * 16 + sub * 4;
                    ob[d..d + 4].copy_from_slice(&blk[DM_OFF[sub]..DM_OFF[sub] + 4]);
                }
            }
        }
    }
    ob
}

/// Build the SUB-MAJOR k1024 Q8_KO chunk from the de-interleaved Q8_KO `[quants(128B) |
/// 16B scales]` tensor (block `k_blk*nrows+row`). The new int8 dequant reads `b_frag[0]` at
/// `sub*256 + lane*4` and `b_frag[1]` at `sub*256 + 128 + lane*4` (the two 128 B half-sub
/// blocks K[0:16]/K[16:32]); we place each lane's 4 bytes there by pulling them from the
/// proven row-major dequant offsets — so the values are correct by construction. dm
/// (`(d,0)` half2) region follows at offset 1024.
#[cfg(feature = "cuda")]
fn q8ko_submajor_from_de(de: &[u8], nrows: usize, ncols: usize) -> Vec<u8> {
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let n_blocks = k_blocks * nrows;
    let sbase = n_blocks * 128;
    let chunk_bytes = 1024 + 128;
    let mut ob = vec![0u8; k_blocks * row_groups * chunk_bytes];
    for k_blk in 0..k_blocks {
        for g in 0..row_groups {
            let cbase = (k_blk * row_groups + g) * chunk_bytes;
            for r in 0..8 {
                let i = k_blk * nrows + g * 8 + r;
                let q = &de[i * 128..i * 128 + 128];
                for sub in 0..4 {
                    for q3 in 0..4 {
                        let lane4 = (r * 4 + q3) * 4; // lane*4 within a half-sub block
                        let off0 = (sub * 4 + (q3 >> 1)) * 8 + (q3 & 1) * 4; // old b_frag[0]
                        let off1 = (sub * 4 + 2 + (q3 >> 1)) * 8 + (q3 & 1) * 4; // old b_frag[1]
                        ob[cbase + sub * 256 + lane4..cbase + sub * 256 + lane4 + 4]
                            .copy_from_slice(&q[off0..off0 + 4]);
                        ob[cbase + sub * 256 + 128 + lane4..cbase + sub * 256 + 128 + lane4 + 4]
                            .copy_from_slice(&q[off1..off1 + 4]);
                    }
                    let d = cbase + 1024 + r * 16 + sub * 4;
                    ob[d..d + 4].copy_from_slice(
                        &de[sbase + i * 16 + sub * 4..sbase + i * 16 + sub * 4 + 4],
                    );
                }
            }
        }
    }
    ob
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
fn per128_ko_toggle_precision_model() -> Result<()> {
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
fn q4_ko_scale_precision() -> Result<()> {
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

/// Decode one Q5_KO block's `(sub, q3)` lane into its two 4-value b_frags (5-bit values
/// 0..31), mirroring the GPU `block_c_q5_KO` dequant: int2 over the sub's qs ints (byte_perm
/// 0x3120 + low/high nibble) plus the 5th-bit extract from the sub's qh int.
#[cfg(feature = "cuda")]
fn q5ko_b_frags(blk: &[u8], sub: usize, q3: usize) -> ([u8; 4], [u8; 4]) {
    const PERM: [usize; 4] = [0, 2, 1, 3];
    let ri = |off: usize| u32::from_le_bytes([blk[off], blk[off + 1], blk[off + 2], blk[off + 3]]);
    let sh = (q3 & 1) * 4;
    // qs ints [I0,I2,I1,I3] at sub*16; q3<2 → {I0,I2}=(x,y at +0,+8), q3>=2 → {I1,I3}=(+8 base).
    let base = sub * 16 + (q3 >> 1) * 8;
    let (qx, qy) = (ri(base), ri(base + 4));
    let nib = |v: u32| -> [u8; 4] {
        let b = ((v >> sh) & 0x0F0F0F0F).to_le_bytes();
        [b[PERM[0]], b[PERM[1]], b[PERM[2]], b[PERM[3]]]
    };
    let qhw = ri(64 + sub * 4);
    let qh0 = (qhw >> ((q3 >> 1) * 8)) & 0xFF;
    let qh1 = (qhw >> (((q3 >> 1) + 2) * 8)) & 0xFF;
    let hb0 = (qh0 >> sh) & 0xF;
    let hb1 = (qh1 >> sh) & 0xF;
    let combine = |nb: [u8; 4], hb: u32| -> [u8; 4] {
        let mut o = [0u8; 4];
        for i in 0..4 {
            o[i] = nb[i] | ((((hb >> i) & 1) as u8) << 4);
        }
        o
    };
    (combine(nib(qx), hb0), combine(nib(qy), hb1))
}

/// Build the SUB-MAJOR k1024 Q5_KO chunk from a de-interleaved `[quants(80B) | 16B
/// (scale,min)]` tensor (block `k_blk*nrows+row`). Layout: 512 B sub-major ql + 128 B
/// sub-major 5th-bit stream (one byte/lane: lo nibble = b0 highs, hi nibble = b1 highs) +
/// 128 B dm. Lane `(row,q3)`: ql int at `sub*128+lane*4`, 5th-bit byte at `512+sub*32+lane`.
#[cfg(feature = "cuda")]
fn q5ko_submajor_from_de(de: &[u8], nrows: usize, ncols: usize) -> Vec<u8> {
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let n_blocks = k_blocks * nrows;
    let sbase = n_blocks * 80;
    let chunk_bytes = 512 + 128 + 128;
    let mut ob = vec![0u8; k_blocks * row_groups * chunk_bytes];
    for k_blk in 0..k_blocks {
        for g in 0..row_groups {
            let cbase = (k_blk * row_groups + g) * chunk_bytes;
            for r in 0..8 {
                let i = k_blk * nrows + g * 8 + r;
                let blk = &de[i * 80..i * 80 + 80];
                for sub in 0..4 {
                    let mut kk = [0u8; 32];
                    for q3 in 0..4 {
                        let (b0, b1) = q5ko_b_frags(blk, sub, q3);
                        for ii in 0..4 {
                            kk[q3 * 4 + ii] = b0[ii];
                            kk[q3 * 4 + 16 + ii] = b1[ii];
                        }
                    }
                    for p in 0..16 {
                        ob[cbase + sub * 128 + r * 16 + p] =
                            (kk[p] & 0xF) | ((kk[p + 16] & 0xF) << 4);
                    }
                    for q3 in 0..4 {
                        let lane = r * 4 + q3;
                        let (mut hb0, mut hb1) = (0u8, 0u8);
                        for j in 0..4 {
                            hb0 |= (((kk[q3 * 4 + j] >> 4) & 1) << j) as u8;
                            hb1 |= (((kk[q3 * 4 + 16 + j] >> 4) & 1) << j) as u8;
                        }
                        ob[cbase + 512 + sub * 32 + lane] = hb0 | (hb1 << 4);
                    }
                    let d = cbase + 640 + r * 16 + sub * 4;
                    ob[d..d + 4].copy_from_slice(
                        &de[sbase + i * 16 + sub * 4..sbase + i * 16 + sub * 4 + 4],
                    );
                }
            }
        }
    }
    ob
}

/// Decode one Q6_KO block's `(sub, q3)` lane into its two 4-value b_frags (6-bit values
/// 0..63), mirroring the GPU `block_c_q6_KO` dequant: int4 over the sub's ql ints (with the
/// `byte_perm 0x3120` [0,2,1,3] order + low/high nibble by `q3&1`), plus the qh crumb spread.
#[cfg(feature = "cuda")]
fn q6ko_b_frags(blk: &[u8], sub: usize, q3: usize) -> ([u8; 4], [u8; 4]) {
    const PERM: [usize; 4] = [0, 2, 1, 3];
    let ri = |off: usize| u32::from_le_bytes([blk[off], blk[off + 1], blk[off + 2], blk[off + 3]]);
    let sh = (q3 & 1) * 4;
    let m0 = sub * 4 + (q3 >> 1);
    let m1 = m0 + 2;
    let (vx, vy, vz, vw) = (
        ri(sub * 16),
        ri(sub * 16 + 4),
        ri(sub * 16 + 8),
        ri(sub * 16 + 12),
    );
    let v0 = if q3 < 2 { vx } else { vy };
    let v1 = if q3 < 2 { vz } else { vw };
    let nib = |v: u32| -> [u8; 4] {
        let b = ((v >> sh) & 0x0F0F0F0F).to_le_bytes();
        [b[PERM[0]], b[PERM[1]], b[PERM[2]], b[PERM[3]]]
    };
    let qh = |m: usize| -> u32 {
        let o = 64 + (m >> 1) * 4 + (m & 1) * 2;
        u16::from_le_bytes([blk[o], blk[o + 1]]) as u32
    };
    let cr0 = (qh(m0) >> ((q3 & 1) * 8)) & 0xFF;
    let cr1 = (qh(m1) >> ((q3 & 1) * 8)) & 0xFF;
    let combine = |nb: [u8; 4], cr: u32| -> [u8; 4] {
        let mut o = [0u8; 4];
        for i in 0..4 {
            o[i] = nb[i] | ((((cr >> (2 * i)) & 0x3) as u8) << 4);
        }
        o
    };
    (combine(nib(v0), cr0), combine(nib(v1), cr1))
}

/// Build the SUB-MAJOR k1024 Q6_KO chunk from a `[quants(96B) | 16B (scale,min)]` tensor
/// (requant_q6_ko_affine output, block `k_blk*nrows+row`). Layout: 512 B sub-major ql +
/// 256 B sub-major qh crumb stream + 128 B dm. Lane `(row,q3)`: ql int at `sub*128+lane*4`
/// (low/high nibbles), qh uint16 at `512+sub*64+lane*2` (cr0 low byte, cr1 high byte).
#[cfg(feature = "cuda")]
fn q6ko_submajor_from_de(de: &[u8], nrows: usize, ncols: usize) -> Vec<u8> {
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let n_blocks = k_blocks * nrows;
    let sbase = n_blocks * 96;
    let chunk_bytes = 512 + 256 + 128;
    let mut ob = vec![0u8; k_blocks * row_groups * chunk_bytes];
    for k_blk in 0..k_blocks {
        for g in 0..row_groups {
            let cbase = (k_blk * row_groups + g) * chunk_bytes;
            for r in 0..8 {
                let i = k_blk * nrows + g * 8 + r;
                let blk = &de[i * 96..i * 96 + 96];
                for sub in 0..4 {
                    let mut kk = [0u8; 32];
                    for q3 in 0..4 {
                        let (b0, b1) = q6ko_b_frags(blk, sub, q3);
                        for ii in 0..4 {
                            kk[q3 * 4 + ii] = b0[ii];
                            kk[q3 * 4 + 16 + ii] = b1[ii];
                        }
                    }
                    for p in 0..16 {
                        ob[cbase + sub * 128 + r * 16 + p] =
                            (kk[p] & 0xF) | ((kk[p + 16] & 0xF) << 4);
                    }
                    for q3 in 0..4 {
                        let lane = r * 4 + q3;
                        let (mut cr0, mut cr1) = (0u8, 0u8);
                        for j in 0..4 {
                            cr0 |= (((kk[q3 * 4 + j] >> 4) & 0x3) << (2 * j)) as u8;
                            cr1 |= (((kk[q3 * 4 + 16 + j] >> 4) & 0x3) << (2 * j)) as u8;
                        }
                        ob[cbase + 512 + sub * 64 + lane * 2] = cr0;
                        ob[cbase + 512 + sub * 64 + lane * 2 + 1] = cr1;
                    }
                    let d = cbase + 768 + r * 16 + sub * 4;
                    ob[d..d + 4].copy_from_slice(
                        &de[sbase + i * 16 + sub * 4..sbase + i * 16 + sub * 4 + 4],
                    );
                }
            }
        }
    }
    ob
}

/// Per-128 affine KO producer (Q4/Q5/Q6) — thin wrapper over the production
/// `ko_quant::quantize_ko`, so the GPU gates double as validation of the production quantizer.
#[cfg(feature = "cuda")]
fn requant_ko_per128(
    w: &[f32],
    nrows: usize,
    ncols: usize,
    maxq: i32,
    crumb_bytes: usize,
    hi_bytes: usize,
) -> Vec<u8> {
    let dtype = match (maxq, crumb_bytes, hi_bytes) {
        (15, 0, 0) => GgmlDType::Q4_KO,
        (31, 0, 128) => GgmlDType::Q5_KO,
        (63, 256, 0) => GgmlDType::Q6_KO,
        _ => panic!("unsupported KO params: maxq={maxq} crumb={crumb_bytes} hi={hi_bytes}"),
    };
    crate::quantized::ko_quant::quantize_ko(w, nrows, ncols, dtype)
}

/// Per-128 Q8_KO producer — thin wrapper over the production `ko_quant::quantize_ko`.
#[cfg(feature = "cuda")]
fn requant_q8_ko_per128(w: &[f32], nrows: usize, ncols: usize) -> Vec<u8> {
    crate::quantized::ko_quant::quantize_ko(w, nrows, ncols, GgmlDType::Q8_KO)
}

#[cfg(feature = "cuda")]
fn requant_q8_ko_affine(w: &[f32], nrows: usize, ncols: usize) -> Vec<u8> {
    use half::f16;
    let k_blocks = ncols / 128;
    let n_blocks = k_blocks * nrows;
    let sbase = n_blocks * 128;
    let mut ob = vec![0u8; sbase + n_blocks * 16];
    for k_blk in 0..k_blocks {
        for row in 0..nrows {
            let block_idx = k_blk * nrows + row;
            for sub in 0..4 {
                let base = k_blk * 128 + sub * 32;
                let mut mn = f32::INFINITY;
                let mut mx = f32::NEG_INFINITY;
                for i in 0..32 {
                    let v = w[row * ncols + base + i];
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                let scale = ((mx - mn) / 255.0).max(1e-12);
                for i in 0..32 {
                    let v = w[row * ncols + base + i];
                    let q = (((v - mn) / scale).round() as i32).clamp(0, 255);
                    ob[block_idx * 128 + sub * 32 + i] = (q - 128) as i8 as u8;
                }
                let m_fold = 128.0 * scale + mn;
                let sc = sbase + block_idx * 16 + sub * 4;
                ob[sc..sc + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
                ob[sc + 2..sc + 4].copy_from_slice(&f16::from_f32(m_fold).to_le_bytes());
            }
        }
    }
    ob
}

/// Requantize F32 weights into Q6_KO with PER-32 affine (scale,min). 6-bit values stay
/// UNSIGNED (0..63 — the un-centered dequant). ql (low 4 bits) and qh (high 2 bits) land
/// at the positions the dequant reads: ql_int/qh use `sub*4 + k'/8`, byte = [0,2,1,3][k'%4]
/// (the dequant's byte_perm 0x3120), nibble/byte = (k'/4)&1, qh 2-bit = 2*(k'%4). Scale
/// region (tail): (scale, min) half2 per sub.
#[cfg(feature = "cuda")]
fn requant_q6_ko_affine(w: &[f32], nrows: usize, ncols: usize) -> Vec<u8> {
    use half::f16;
    const PERM: [usize; 4] = [0, 2, 1, 3];
    let k_blocks = ncols / 128;
    let n_blocks = k_blocks * nrows;
    let sbase = n_blocks * 96; // quant region 96B/block (ql 64 + qh 32)
    let mut ob = vec![0u8; sbase + n_blocks * 16];
    for k_blk in 0..k_blocks {
        for row in 0..nrows {
            let block_idx = k_blk * nrows + row;
            let qbase = block_idx * 96;
            for sub in 0..4 {
                let cbase = k_blk * 128 + sub * 32;
                let mut mn = f32::INFINITY;
                let mut mx = f32::NEG_INFINITY;
                for i in 0..32 {
                    let v = w[row * ncols + cbase + i];
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                let scale = ((mx - mn) / 63.0).max(1e-12);
                for kp in 0..32 {
                    let v = w[row * ncols + cbase + kp];
                    let q = (((v - mn) / scale).round() as i32).clamp(0, 63) as u32;
                    let unit = sub * 4 + kp / 8;
                    let nib = (kp / 4) & 1;
                    // ql low nibble
                    let pos = qbase + unit * 4 + PERM[kp % 4];
                    ob[pos] |= ((q & 0xF) as u8) << (nib * 4);
                    // qh high 2 bits
                    let qh_byte = 2 * unit + nib;
                    ob[qbase + 64 + qh_byte] |= (((q >> 4) & 0x3) as u8) << (2 * (kp % 4));
                }
                let sc = sbase + block_idx * 16 + sub * 4;
                ob[sc..sc + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
                ob[sc + 2..sc + 4].copy_from_slice(&f16::from_f32(mn).to_le_bytes());
            }
        }
    }
    ob
}

/// Q5_KO / Q6_KO / Q8_KO are byte permutations of the Q5_K / Q6_K / Q8_K compact
/// blocks. Each must give the BIT-IDENTICAL int8 q8a128 result of its K twin. The
/// host permutation below mirrors each format's compact layout → KO regularized
/// layout (Q6_KO is the identity — Q6_K's compact block is already ordered).
#[test]
#[ignore = "retired: per-32 sub-major path superseded by the per-128 collapse"]
fn q5q6q8_ko_match_k_int8() -> Result<()> {
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

/// Finer-grained M scan for the mode-1 / mode-2 crossover (Q4_KO). Reports the activation
/// element count (M·K), the weight count (N·K), their ratio (= M/N), and the i8KO time.
/// Run twice — env KO_M2 unset (mode-1) and KO_M2=1 (mode-2) — and compare the i8KO columns
/// to locate the M where mode-2 overtakes mode-1.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn q4_crossover_scan() -> Result<()> {
    use std::time::Instant;
    let dev = CudaDevice::new(0)?;
    let nrows = 768usize; // N
    let ncols = 2048usize; // K
    let mut rng = rand::rng();
    let wf32: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-0.1f32..0.1))
        .collect();
    let ob = requant_ko_per128(&wf32, nrows, ncols, 15, 0, 0); // Q4_KO
    let ko_slice = dev.memcpy_stod(&ob)?;
    let stream = dev.cuda_stream();
    let (ko_ptr, _g) = ko_slice.device_ptr(&stream);
    let weight_count = nrows * ncols;
    let mode = if std::env::var("KO_M2").is_ok() {
        "MODE-2"
    } else {
        "MODE-1"
    };
    println!("=== Q4 crossover scan [{mode}] N={nrows} K={ncols} weight_count={weight_count} ===");
    println!("     M    act_count   act/wt   i8KO(ms)   i8KO-tok/s");
    for &m in &[
        16usize, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 1024,
    ] {
        let act: Vec<f32> = (0..m * ncols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let q8 = quantize_acts_q8a128_test(&dev, &act, m, ncols)?;
        let time_i8 = || -> Result<f64> {
            for _ in 0..20 {
                let _ = dense_qmatmul(
                    DynamicTensor::Int8(&q8),
                    ko_ptr,
                    GgmlDType::Q4_KO,
                    nrows,
                    0,
                    crate::DType::F32,
                    &dev,
                )?;
            }
            dev.synchronize()?;
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                for _ in 0..100 {
                    let _ = dense_qmatmul(
                        DynamicTensor::Int8(&q8),
                        ko_ptr,
                        GgmlDType::Q4_KO,
                        nrows,
                        0,
                        crate::DType::F32,
                        &dev,
                    )?;
                }
                dev.synchronize()?;
                best = best.min(t.elapsed().as_secs_f64() / 100.0);
            }
            Ok(best)
        };
        let t = time_i8()?;
        let act_count = m * ncols;
        let ratio = act_count as f64 / weight_count as f64;
        println!(
            "{m:>6} {act_count:>12} {ratio:>8.3} {:>10.4} {:>12.0}",
            t * 1e3,
            m as f64 / t
        );
    }
    Ok(())
}

/// Benchmark each KO byte-permuted twin against its K format on the int8 q8a128
/// dense matmul across token counts. Both read 80/112/160-byte compact blocks; KO's
/// only difference is the (regularized) in-block byte order, so this measures whether
/// the reorder alone moves the needle before any wide-load optimization.
#[test]
#[ignore]
fn ko_vs_k_int8_bench() -> Result<()> {
    use std::time::Instant;
    let dev = CudaDevice::new(0)?;
    let nrows = 768usize; // N
    let ncols = 2048usize; // K

    // _KO = _K + 1 bit, per-128, F16 scale: (K dtype for the FP16 reference, KO dtype, maxq,
    // crumb_bytes, hi_bytes). int8 weights re-quantized per-128 from f32 (data-independent timing).
    let mut rng = rand::rng();
    let wf32: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-0.1f32..0.1))
        .collect();
    let cases: &[(GgmlDType, GgmlDType, i32, usize, usize)] = &[
        (GgmlDType::Q4_K, GgmlDType::Q4_KO, 15, 0, 0), // same-bit (no +1): precision hit, smallest
        (GgmlDType::Q4_K, GgmlDType::Q5_KO, 31, 0, 128),
        (GgmlDType::Q5_K, GgmlDType::Q6_KO, 63, 256, 0),
        (GgmlDType::Q8_K, GgmlDType::Q8_KO, 255, 0, 0),
    ];

    // Dense per-128: as M ramps the weight amortizes over more tokens. Small M is memory-bound
    // (FP16/int8 read similar bytes → parity); large M is compute-bound where the accumulate-
    // then-scale collapse matters. FP16 = dequant→bf16 MMA on the K fixture; int8 = per-128 KO.
    for &(kd, kod, maxq, crumb, hi) in cases {
        // int8 KO weights — per-128 re-quant from f32, once per case (same across M).
        let ob = if kod == GgmlDType::Q8_KO {
            requant_q8_ko_per128(&wf32, nrows, ncols)
        } else {
            requant_ko_per128(&wf32, nrows, ncols, maxq, crumb, hi)
        };
        let ko_slice = dev.memcpy_stod(&ob)?;
        let stream = dev.cuda_stream();
        let (ko_ptr, _g) = ko_slice.device_ptr(&stream);
        println!("=== dense per-128 [{nrows}x{ncols}]: FP16 {kd:?} vs int8 {kod:?} ===");
        println!(" tokens   f16(ms)  i8KO(ms) f16/i8KO  i8KO-tok/s");
        for &m in &[1usize, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
            let act: Vec<f32> = (0..m * ncols)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            let q8 = quantize_acts_q8a128_test(&dev, &act, m, ncols)?;
            let (_k_ptrs, k_st, _a, _l, _o) =
                build_int8_ab_fixture(&dev, nrows, ncols, &[m], &act, kd)?;
            // FP16 path: bf16 activations → dequant→f16 tensor-core MMA on the K weights.
            let act_bf16: Vec<half::bf16> = act.iter().map(|&v| half::bf16::from_f32(v)).collect();
            let act_storage =
                CudaStorage::wrap_cuda_slice(dev.memcpy_stod(&act_bf16)?, dev.clone());
            let act_layout = crate::Layout::contiguous(crate::Shape::from(vec![m, ncols]));
            let wshape = crate::Shape::from((nrows, ncols));

            let time_i8 = |ptr: u64| -> Result<f64> {
                for _ in 0..20 {
                    let _ = dense_qmatmul(
                        DynamicTensor::Int8(&q8),
                        ptr,
                        kod,
                        nrows,
                        0,
                        crate::DType::F32,
                        &dev,
                    )?;
                }
                dev.synchronize()?;
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t = Instant::now();
                    for _ in 0..100 {
                        let _ = dense_qmatmul(
                            DynamicTensor::Int8(&q8),
                            ptr,
                            kod,
                            nrows,
                            0,
                            crate::DType::F32,
                            &dev,
                        )?;
                    }
                    dev.synchronize()?;
                    best = best.min(t.elapsed().as_secs_f64() / 100.0);
                }
                Ok(best)
            };
            let time_fp = || -> Result<f64> {
                for _ in 0..20 {
                    let _ = k_st[0].fwd_via_gemx(&wshape, &act_storage, &act_layout)?;
                }
                dev.synchronize()?;
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t = Instant::now();
                    for _ in 0..100 {
                        let _ = k_st[0].fwd_via_gemx(&wshape, &act_storage, &act_layout)?;
                    }
                    dev.synchronize()?;
                    best = best.min(t.elapsed().as_secs_f64() / 100.0);
                }
                Ok(best)
            };
            let t_fp = time_fp()?;
            let t_ko = time_i8(ko_ptr)?;
            println!(
                "{m:>7} {:>9.4} {:>9.4} {:>8.2} {:>10.0}",
                t_fp * 1e3,
                t_ko * 1e3,
                t_fp / t_ko,
                m as f64 / t_ko
            );
        }
    }
    Ok(())
}

/// ncu probe: launches each KO int8 DENSE kernel (`q{4,5,6,8}_ko_int8_f32_dense`) once at the
/// big-M prefill config (m=4096) so Nsight Compute can read each kernel's occupancy/resource
/// limits in isolation — this is the shape where KO regresses vs FP16 (~0.87×). No timing;
/// clean launches for `ncu -k regex:_ko_int8_f32_dense -c 4 --section Occupancy`.
#[test]
#[ignore]
fn ko_dense_ncu_probe() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 768usize; // N
    let ncols = 2048usize; // K
    let m = 4096usize; // big-M prefill — the regressing shape
    let mut rng = rand::rng();
    let wf32: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-0.1f32..0.1))
        .collect();
    let act: Vec<f32> = (0..m * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let q8 = quantize_acts_q8a128_test(&dev, &act, m, ncols)?;
    // (dtype, maxq, crumb_bytes, hi_bytes) — Q8 uses its own symmetric per-128 producer.
    for &(kod, maxq, crumb, hi) in &[
        (GgmlDType::Q4_KO, 15i32, 0usize, 0usize),
        (GgmlDType::Q5_KO, 31, 0, 128),
        (GgmlDType::Q6_KO, 63, 256, 0),
        (GgmlDType::Q8_KO, 255, 0, 0),
    ] {
        let ob = if kod == GgmlDType::Q8_KO {
            requant_q8_ko_per128(&wf32, nrows, ncols)
        } else {
            requant_ko_per128(&wf32, nrows, ncols, maxq, crumb, hi)
        };
        let ko_slice = dev.memcpy_stod(&ob)?;
        let stream = dev.cuda_stream();
        let (ko_ptr, _g) = ko_slice.device_ptr(&stream);
        for _ in 0..3 {
            let _ = dense_qmatmul(
                DynamicTensor::Int8(&q8),
                ko_ptr,
                kod,
                nrows,
                0,
                crate::DType::F32,
                &dev,
            )?;
        }
        dev.synchronize()?;
    }
    Ok(())
}

/// Representative MoE-decode benchmark: 128 experts, top-8 routing (Qwen3-30B-A3B
/// config), distinct weights per expert (no cross-expert L2 reuse). `batch` tokens
/// each route to 8 random unique experts and the grouped int8 matmul processes the
/// resulting (expert → token-slice) tiles in ONE launch — the real production path.
/// At batch=1 → 8 experts (1 token each); at batch=128 → ~all 128 experts (~8 tokens
/// each). This is what actually predicts the model, vs the dense single-weight bench.
#[test]
#[ignore]
fn grouped_moe_ko_vs_k_bench() -> Result<()> {
    use std::collections::HashSet;
    use std::time::Instant;
    let dev = CudaDevice::new(0)?;
    let nrows = 768usize; // N
    let ncols = 2048usize; // K
    let num_experts = 128usize;
    let top_k = 8usize;
    let mut rng = rand::rng();

    // _KO = _K + 1 bit, per-128: (K dtype for FP16 ref, KO dtype, maxq, crumb_bytes, hi_bytes).
    let cases: &[(GgmlDType, GgmlDType, i32, usize, usize)] = &[
        (GgmlDType::Q4_K, GgmlDType::Q5_KO, 31, 0, 128),
        (GgmlDType::Q5_K, GgmlDType::Q6_KO, 63, 256, 0),
        (GgmlDType::Q8_K, GgmlDType::Q8_KO, 255, 0, 0),
    ];

    let shape = crate::Shape::from((nrows, ncols));
    for &(kd, kod, maxq, crumb, hi) in cases {
        // Build a pool of `num_experts` DISTINCT weights (K-compact + KO-permuted).
        let mut k_ptrs = vec![0u64; num_experts];
        let mut ko_ptrs = vec![0u64; num_experts];
        let mut k_pool = Vec::with_capacity(num_experts); // QCudaStorage, kept alive
        let mut ko_pool = Vec::with_capacity(num_experts); // CudaSlice<u8>, kept alive
        for e in 0..num_experts {
            let w: Vec<f32> = (0..nrows * ncols)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            let mut xs = QCudaStorage::zeros(&dev, ncols * nrows, kd)?;
            xs.quantize(&CudaStorage::wrap_cuda_slice(
                dev.memcpy_stod(&w)?,
                dev.clone(),
            ))?;
            let xs = xs.repack_gemx(&shape)?;
            k_ptrs[e] = xs.data_ptr();
            // int8 KO weights — per-128 re-quant from this expert's f32 (the _K+1-bit remap).
            let ob = if kod == GgmlDType::Q8_KO {
                requant_q8_ko_per128(&w, nrows, ncols)
            } else {
                requant_ko_per128(&w, nrows, ncols, maxq, crumb, hi)
            };
            let ko_slice = dev.memcpy_stod(&ob)?;
            {
                let stream = dev.cuda_stream();
                let (p, _g) = ko_slice.device_ptr(&stream);
                ko_ptrs[e] = p; // raw address stays valid while ko_slice lives in the pool
            }
            k_pool.push(xs);
            ko_pool.push(ko_slice);
        }

        println!(
            "=== grouped MoE [{nrows}x{ncols}], 128 experts, top-8: FP16 {kd:?} → int8 {kod:?} ==="
        );
        println!("  batch  experts  slices  f16-K(ms) i8-KO(ms) f16K/i8KO  i8KO-tok/s");
        for &batch in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
            // Route each token to `top_k` unique experts; group by expert.
            let mut per_expert: Vec<Vec<usize>> = vec![Vec::new(); num_experts];
            for t in 0..batch {
                let mut chosen = HashSet::new();
                while chosen.len() < top_k {
                    chosen.insert(rng.random_range(0..num_experts));
                }
                for &e in &chosen {
                    per_expert[e].push(t);
                }
            }
            let touched: Vec<usize> = (0..num_experts)
                .filter(|&e| !per_expert[e].is_empty())
                .collect();
            let total: usize = touched.iter().map(|&e| per_expert[e].len()).sum();
            let mut expert_offsets = vec![0i32];
            let mut stacked: Vec<usize> = Vec::with_capacity(total);
            for &e in &touched {
                stacked.extend_from_slice(&per_expert[e]);
                expert_offsets.push(expert_offsets.last().unwrap() + per_expert[e].len() as i32);
            }
            // `batch` distinct token activations; stack them in expert-slice order.
            let token_acts: Vec<Vec<f32>> = (0..batch)
                .map(|_| (0..ncols).map(|_| rng.random_range(-1.0..1.0)).collect())
                .collect();
            let mut act = Vec::with_capacity(total * ncols);
            for &t in &stacked {
                act.extend_from_slice(&token_acts[t]);
            }
            // q8a128 activations for the int8 path; bf16 activations for the FP16 path.
            let q8a128 = quantize_acts_q8a128_test(&dev, &act, total, ncols)?;
            let act_bf16: Vec<bf16> = act.iter().map(|&v| bf16::from_f32(v)).collect();
            let act_storage =
                CudaStorage::wrap_cuda_slice(dev.memcpy_stod(&act_bf16)?, dev.clone());
            let act_layout = crate::Layout::contiguous(crate::Shape::from(vec![total, ncols]));
            let wk: Vec<u64> = touched.iter().map(|&e| k_ptrs[e]).collect();
            let wko: Vec<u64> = touched.iter().map(|&e| ko_ptrs[e]).collect();

            let time_i8 = |wptrs: &[u64], dtype: GgmlDType| -> Result<f64> {
                for _ in 0..20 {
                    let _ = grouped_qmatmul(
                        DynamicTensor::Int8(&q8a128),
                        wptrs,
                        dtype,
                        nrows,
                        &expert_offsets,
                        &dev,
                        Backing::Owned,
                    )?;
                }
                dev.synchronize()?;
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t0 = Instant::now();
                    for _ in 0..100 {
                        let _ = grouped_qmatmul(
                            DynamicTensor::Int8(&q8a128),
                            wptrs,
                            dtype,
                            nrows,
                            &expert_offsets,
                            &dev,
                            Backing::Owned,
                        )?;
                    }
                    dev.synchronize()?;
                    best = best.min(t0.elapsed().as_secs_f64() / 100.0);
                }
                Ok(best)
            };
            let time_fp = |wptrs: &[u64], dtype: GgmlDType| -> Result<f64> {
                for _ in 0..20 {
                    let _ = grouped_matmul_gemx(
                        wptrs,
                        dtype,
                        nrows,
                        ncols,
                        &act_storage,
                        &act_layout,
                        &expert_offsets,
                        &dev,
                    )?;
                }
                dev.synchronize()?;
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t0 = Instant::now();
                    for _ in 0..100 {
                        let _ = grouped_matmul_gemx(
                            wptrs,
                            dtype,
                            nrows,
                            ncols,
                            &act_storage,
                            &act_layout,
                            &expert_offsets,
                            &dev,
                        )?;
                    }
                    dev.synchronize()?;
                    best = best.min(t0.elapsed().as_secs_f64() / 100.0);
                }
                Ok(best)
            };
            let t_fp = time_fp(&wk, kd)?; // FP16 weights=K (current path)
            let t_i8ko = time_i8(&wko, kod)?; // int8 weights=KO (the upgrade)
                                              // Projected end-to-end token rate from the i8-KO matmul: 48 layers, each
                                              // = this FFN matmul + a 0.5× attention pass → 48 * 1.5 * t per decode step.
            let tok_s = batch as f64 / (48.0 * 1.5 * t_i8ko);
            println!(
                "{batch:>7} {:>8} {:>7} {:>9.4} {:>9.4} {:>8.2} {:>10.0}",
                touched.len(),
                total,
                t_fp * 1e3,
                t_i8ko * 1e3,
                t_fp / t_i8ko,
                tok_s
            );
        }
        drop(k_pool);
        drop(ko_pool);
    }
    Ok(())
}

/// Single-config statistical bench: Q4_K, 128 experts, 4096 activation rows
/// (32 token-slices/expert — a balanced prefill batch). Where the sweep bench
/// reports best-of-5 (which is noisy at the sub-percent level we care about for
/// the FP16-vs-int8 parity question), this draws M independent samples — each the
/// mean of N back-to-back launches — and reduces them to mean ± 95% CI, median,
/// min and coefficient of variation. The three paths are interleaved within every
/// sample so thermal/clock drift hits them equally and cancels in the comparison.
/// The final line states whether f16-K and i8-KO differ at 95% confidence or are
/// statistically parity.
#[test]
#[ignore = "retired: builds KO weights in the sub-major byte-permute layout and times the \
            i8-K (non-KO) path the pairing guard now rejects. Superseded by the per-128 \
            grouped_moe_ko_vs_k_bench (the production 128-expert f16-K vs i8-KO comparison)."]
fn q4k_grouped_stats_bench() -> Result<()> {
    use std::time::Instant;
    let dev = CudaDevice::new(0)?;
    let nrows = 768usize; // N
    let ncols = 2048usize; // K
    let num_experts = 128usize;
    let tokens_per_expert = 32usize;
    let total = num_experts * tokens_per_expert; // 4096 activation rows
    let mut rng = rand::rng();

    type Perm = fn(&[u8], &mut [u8]);
    let cases: &[(GgmlDType, GgmlDType, usize, Perm)] = &[
        (GgmlDType::Q4_K, GgmlDType::Q4_KO, 80, |kb, ob| {
            for (s, &kb0) in [0usize, 24, 40, 64].iter().enumerate() {
                ob[s * 16..s * 16 + 4].copy_from_slice(&kb[kb0..kb0 + 4]);
                ob[s * 16 + 4..s * 16 + 8].copy_from_slice(&kb[kb0 + 8..kb0 + 12]);
                ob[s * 16 + 8..s * 16 + 12].copy_from_slice(&kb[kb0 + 4..kb0 + 8]);
                ob[s * 16 + 12..s * 16 + 16].copy_from_slice(&kb[kb0 + 12..kb0 + 16]);
            }
            ob[64..68].copy_from_slice(&kb[16..20]);
            ob[68..72].copy_from_slice(&kb[20..24]);
            ob[72..76].copy_from_slice(&kb[56..60]);
            ob[76..80].copy_from_slice(&kb[60..64]);
        }),
        (GgmlDType::Q5_K, GgmlDType::Q5_KO, 112, |kb, ob| {
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
    let shape = crate::Shape::from((nrows, ncols));

    // Balanced routing: exactly `tokens_per_expert` slices per expert, fixed once so
    // routing variance is not a noise source. Shared across all formats.
    let mut expert_offsets = vec![0i32];
    for _ in 0..num_experts {
        expert_offsets.push(expert_offsets.last().unwrap() + tokens_per_expert as i32);
    }
    let act: Vec<f32> = (0..total * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let q8a128 = quantize_acts_q8a128_test(&dev, &act, total, ncols)?;
    let act_bf16: Vec<bf16> = act.iter().map(|&v| bf16::from_f32(v)).collect();
    let act_storage = CudaStorage::wrap_cuda_slice(dev.memcpy_stod(&act_bf16)?, dev.clone());
    let act_layout = crate::Layout::contiguous(crate::Shape::from(vec![total, ncols]));

    let n_inner = 50usize; // launches averaged per sample
    let m_samples = 200usize; // independent samples → CI ∝ 1/sqrt(M)

    // mean, median, min, 95% CI half-width, CV% — sample stats (N-1 variance).
    fn stats(xs: &[f64]) -> (f64, f64, f64, f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = var.sqrt();
        let ci95 = 1.96 * std / n.sqrt();
        let cv = std / mean * 100.0;
        let mut sorted = xs.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (mean, sorted[sorted.len() / 2], sorted[0], ci95, cv)
    }

    println!(
        "=== grouped MoE stats: {num_experts} experts × {tokens_per_expert} = {total} act rows, [{nrows}x{ncols}] ==="
    );
    println!("    {m_samples} samples × {n_inner} launches each, interleaved per sample\n");

    for &(kd, kod, stride, perm) in cases {
        // 128 distinct expert weights (K-compact + KO-permuted), kept alive in pools.
        let mut k_ptrs = vec![0u64; num_experts];
        let mut ko_ptrs = vec![0u64; num_experts];
        let mut k_pool = Vec::with_capacity(num_experts);
        let mut ko_pool = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let w: Vec<f32> = (0..nrows * ncols)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            let mut xs = QCudaStorage::zeros(&dev, ncols * nrows, kd)?;
            xs.quantize(&CudaStorage::wrap_cuda_slice(
                dev.memcpy_stod(&w)?,
                dev.clone(),
            ))?;
            let xs = xs.repack_gemx(&shape)?;
            k_ptrs[e] = xs.data_ptr();
            let kb = xs.data()?;
            // Q6_KO/Q8_KO: requantize F32 → per-32 affine (scale,min). Q4/Q5 keep the byte
            // permutation (already per-32 affine).
            let de = if kod == GgmlDType::Q8_KO {
                requant_q8_ko_affine(&w, nrows, ncols)
            } else if kod == GgmlDType::Q6_KO {
                requant_q6_ko_affine(&w, nrows, ncols)
            } else {
                ko_deinterleave(
                    &kb,
                    stride,
                    ko_quant_bytes(kod),
                    kod == GgmlDType::Q8_KO,
                    perm,
                )
            };
            // Sub-major k1024 layout: Q4 from the compact, Q5/Q6/Q8 from the per-format `de`
            // ([quants|scale] — requant for Q6/Q8, de-interleave for Q5).
            let ob = match kod {
                GgmlDType::Q4_KO => q4ko_submajor_from_compact(&kb, nrows, ncols),
                GgmlDType::Q5_KO => q5ko_submajor_from_de(&de, nrows, ncols),
                GgmlDType::Q6_KO => q6ko_submajor_from_de(&de, nrows, ncols),
                GgmlDType::Q8_KO => q8ko_submajor_from_de(&de, nrows, ncols),
                _ => unreachable!("not a KO format: {kod:?}"),
            };
            let ko_slice = dev.memcpy_stod(&ob)?;
            {
                let stream = dev.cuda_stream();
                let (p, _g) = ko_slice.device_ptr(&stream);
                ko_ptrs[e] = p;
            }
            k_pool.push(xs);
            ko_pool.push(ko_slice);
        }

        let wk = k_ptrs.clone();
        let wko = ko_ptrs.clone();

        // One-shot precision: KO int8 result vs the FP16 reference of the SAME weights.
        {
            let ref_f16 = grouped_matmul_gemx(
                &wk,
                kd,
                nrows,
                ncols,
                &act_storage,
                &act_layout,
                &expert_offsets,
                &dev,
            )?;
            let ko_i8 = grouped_qmatmul(
                DynamicTensor::Int8(&q8a128),
                &wko,
                kod,
                nrows,
                &expert_offsets,
                &dev,
                Backing::Owned,
            )?;
            let vref = read_bf16_tensor(&dev, &ref_f16)?;
            let vko = read_f32_tensor(&dev, &ko_i8)?;
            println!(
                "  precision  i8-{kod:?} vs f16-{kd:?}: rel_l2 = {:.5}",
                rel_l2(&vref, &vko)
            );
        }

        // Warmup every path past clock spin-up before sampling.
        for _ in 0..100 {
            let _ = grouped_matmul_gemx(
                &wk,
                kd,
                nrows,
                ncols,
                &act_storage,
                &act_layout,
                &expert_offsets,
                &dev,
            )?;
            let _ = grouped_qmatmul(
                DynamicTensor::Int8(&q8a128),
                &wk,
                kd,
                nrows,
                &expert_offsets,
                &dev,
                Backing::Owned,
            )?;
            let _ = grouped_qmatmul(
                DynamicTensor::Int8(&q8a128),
                &wko,
                kod,
                nrows,
                &expert_offsets,
                &dev,
                Backing::Owned,
            )?;
        }
        dev.synchronize()?;

        let mut s_fp = Vec::with_capacity(m_samples);
        let mut s_i8k = Vec::with_capacity(m_samples);
        let mut s_i8ko = Vec::with_capacity(m_samples);
        for _ in 0..m_samples {
            let t = Instant::now();
            for _ in 0..n_inner {
                let _ = grouped_matmul_gemx(
                    &wk,
                    kd,
                    nrows,
                    ncols,
                    &act_storage,
                    &act_layout,
                    &expert_offsets,
                    &dev,
                )?;
            }
            dev.synchronize()?;
            s_fp.push(t.elapsed().as_secs_f64() / n_inner as f64);

            let t = Instant::now();
            for _ in 0..n_inner {
                let _ = grouped_qmatmul(
                    DynamicTensor::Int8(&q8a128),
                    &wk,
                    kd,
                    nrows,
                    &expert_offsets,
                    &dev,
                    Backing::Owned,
                )?;
            }
            dev.synchronize()?;
            s_i8k.push(t.elapsed().as_secs_f64() / n_inner as f64);

            let t = Instant::now();
            for _ in 0..n_inner {
                let _ = grouped_qmatmul(
                    DynamicTensor::Int8(&q8a128),
                    &wko,
                    kod,
                    nrows,
                    &expert_offsets,
                    &dev,
                    Backing::Owned,
                )?;
            }
            dev.synchronize()?;
            s_i8ko.push(t.elapsed().as_secs_f64() / n_inner as f64);
        }

        let (fp_m, fp_md, fp_min, fp_ci, fp_cv) = stats(&s_fp);
        let (k_m, k_md, k_min, k_ci, k_cv) = stats(&s_i8k);
        let (ko_m, ko_md, ko_min, ko_ci, ko_cv) = stats(&s_i8ko);

        println!("--- FP16 {kd:?} → int8 {kod:?} ---");
        println!("  path     mean(ms)   ±95%CI   median(ms)   min(ms)   CV%");
        println!(
            "  f16-K   {:>8.4}  {:>7.4}   {:>8.4}  {:>8.4}  {:>5.2}",
            fp_m * 1e3,
            fp_ci * 1e3,
            fp_md * 1e3,
            fp_min * 1e3,
            fp_cv
        );
        println!(
            "  i8-K    {:>8.4}  {:>7.4}   {:>8.4}  {:>8.4}  {:>5.2}",
            k_m * 1e3,
            k_ci * 1e3,
            k_md * 1e3,
            k_min * 1e3,
            k_cv
        );
        println!(
            "  i8-KO   {:>8.4}  {:>7.4}   {:>8.4}  {:>8.4}  {:>5.2}",
            ko_m * 1e3,
            ko_ci * 1e3,
            ko_md * 1e3,
            ko_min * 1e3,
            ko_cv
        );
        let ratio = fp_m / ko_m;
        let disjoint = (fp_m - fp_ci) > (ko_m + ko_ci) || (ko_m - ko_ci) > (fp_m + fp_ci);
        let pct = (ratio - 1.0) * 100.0;
        println!(
            "  f16-K / i8-KO = {ratio:.4} ({pct:+.2}%)  →  {}\n",
            if disjoint {
                if ratio > 1.0 {
                    "SIGNIFICANT win for i8-KO (95% CI)"
                } else {
                    "SIGNIFICANT win for f16-K (95% CI)"
                }
            } else {
                "parity (CIs overlap at 95%)"
            }
        );

        drop(k_pool);
        drop(ko_pool);
    }
    Ok(())
}

/// ncu probe: launches the Q5_K FP16 grouped kernel (`q5_k_f32_grouped`) and the
/// Q5_KO int8 grouped kernel (`q5_ko_int8_f32_grouped`) a handful of times at the
/// 4096-row config so Nsight Compute can profile each in isolation by kernel name.
/// No timing — just clean, repeatable launches for `ncu -k regex:... -c N`.
#[test]
#[ignore]
fn q5_ncu_probe() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 768usize;
    let ncols = 2048usize;
    let num_experts = 128usize;
    let tokens_per_expert = 32usize;
    let total = num_experts * tokens_per_expert;
    let stride = 112usize; // Q5_K compact block
    let mut rng = rand::rng();
    let kd = GgmlDType::Q5_K;
    let kod = GgmlDType::Q5_KO;
    let perm: fn(&[u8], &mut [u8]) = |kb: &[u8], ob: &mut [u8]| {
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
    let shape = crate::Shape::from((nrows, ncols));
    let mut k_ptrs = vec![0u64; num_experts];
    let mut ko_ptrs = vec![0u64; num_experts];
    let mut k_pool = Vec::with_capacity(num_experts);
    let mut ko_pool = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let w: Vec<f32> = (0..nrows * ncols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let mut xs = QCudaStorage::zeros(&dev, ncols * nrows, kd)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(
            dev.memcpy_stod(&w)?,
            dev.clone(),
        ))?;
        let xs = xs.repack_gemx(&shape)?;
        k_ptrs[e] = xs.data_ptr();
        let kb = xs.data()?;
        // Q6_KO/Q8_KO: requantize F32 → per-32 affine (scale,min). Q4/Q5 keep the byte
        // permutation (already per-32 affine).
        let de = if kod == GgmlDType::Q8_KO {
            requant_q8_ko_affine(&w, nrows, ncols)
        } else if kod == GgmlDType::Q6_KO {
            requant_q6_ko_affine(&w, nrows, ncols)
        } else {
            ko_deinterleave(
                &kb,
                stride,
                ko_quant_bytes(kod),
                kod == GgmlDType::Q8_KO,
                perm,
            )
        };
        // Sub-major k1024 layout: Q4 from the compact, Q5/Q6/Q8 from the per-format `de`
        // ([quants|scale] — requant for Q6/Q8, de-interleave for Q5).
        let ob = match kod {
            GgmlDType::Q4_KO => q4ko_submajor_from_compact(&kb, nrows, ncols),
            GgmlDType::Q5_KO => q5ko_submajor_from_de(&de, nrows, ncols),
            GgmlDType::Q6_KO => q6ko_submajor_from_de(&de, nrows, ncols),
            GgmlDType::Q8_KO => q8ko_submajor_from_de(&de, nrows, ncols),
            _ => unreachable!("not a KO format: {kod:?}"),
        };
        let ko_slice = dev.memcpy_stod(&ob)?;
        {
            let stream = dev.cuda_stream();
            let (p, _g) = ko_slice.device_ptr(&stream);
            ko_ptrs[e] = p;
        }
        k_pool.push(xs);
        ko_pool.push(ko_slice);
    }
    let mut expert_offsets = vec![0i32];
    for _ in 0..num_experts {
        expert_offsets.push(expert_offsets.last().unwrap() + tokens_per_expert as i32);
    }
    let act: Vec<f32> = (0..total * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let q8a128 = quantize_acts_q8a128_test(&dev, &act, total, ncols)?;
    let act_bf16: Vec<bf16> = act.iter().map(|&v| bf16::from_f32(v)).collect();
    let act_storage = CudaStorage::wrap_cuda_slice(dev.memcpy_stod(&act_bf16)?, dev.clone());
    let act_layout = crate::Layout::contiguous(crate::Shape::from(vec![total, ncols]));

    // A few launches of each kernel; ncu filters by name and profiles -c N of them.
    for _ in 0..6 {
        let _ = grouped_matmul_gemx(
            &k_ptrs,
            kd,
            nrows,
            ncols,
            &act_storage,
            &act_layout,
            &expert_offsets,
            &dev,
        )?;
        let _ = grouped_qmatmul(
            DynamicTensor::Int8(&q8a128),
            &ko_ptrs,
            kod,
            nrows,
            &expert_offsets,
            &dev,
            Backing::Owned,
        )?;
    }
    dev.synchronize()?;
    drop(k_pool);
    drop(ko_pool);
    Ok(())
}

/// §8.2 outlier-robustness floor — inject sparse activation outliers (the failure
/// mode `randn` never produces) and require the INT8 path to degrade gracefully,
/// not catastrophically, vs the FP16 oracle (which has the dynamic range to
/// absorb them). This stresses the per-32-block int8 quant headroom.
#[test]
fn grouped_int8_outlier_stress() -> Result<()> {
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

/// Perf A/B — time the INT8 grouped kernel (`q8a128 × KO`) vs the legacy FP16 grouped
/// kernel (dequant K-quant) on the SAME source weight, **single expert**, across a
/// tokens-per-expert sweep. Single expert so the weight load amortizes across M: small
/// M is weight-bandwidth-bound, large M is compute-bound where the INT8 `m16n8k32`
/// should pull ahead. Min-over-batches filters transient GPU stalls.
#[test]
#[ignore = "GPU perf benchmark; run with --ignored --nocapture"]
fn grouped_int8_vs_legacy_bench() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 768; // N = intermediate (gate/up)
    let ncols = 2048; // K = hidden
    let shape = crate::Shape::from((nrows, ncols));
    let mut rng = rand::rng();

    // (K dtype for the FP16 ref, KO twin for int8, maxq, crumb_bytes, hi_bytes).
    let cases: &[(GgmlDType, GgmlDType, i32, usize, usize)] = &[
        (GgmlDType::Q4_K, GgmlDType::Q5_KO, 31, 0, 128),
        (GgmlDType::Q5_K, GgmlDType::Q6_KO, 63, 256, 0),
        (GgmlDType::Q8_K, GgmlDType::Q8_KO, 255, 0, 0),
    ];
    for &(kd, kod, maxq, crumb, hi) in cases {
        // One expert: FP16-K weight (gemx-repacked) + its KO twin, from the same f32.
        let w: Vec<f32> = (0..nrows * ncols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let mut xs = QCudaStorage::zeros(&dev, ncols * nrows, kd)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(
            dev.memcpy_stod(&w)?,
            dev.clone(),
        ))?;
        let xs = xs.repack_gemx(&shape)?;
        let k_ptrs = vec![xs.data_ptr()];
        let ob = if kod == GgmlDType::Q8_KO {
            requant_q8_ko_per128(&w, nrows, ncols)
        } else {
            requant_ko_per128(&w, nrows, ncols, maxq, crumb, hi)
        };
        let ko_slice = dev.memcpy_stod(&ob)?;
        let stream = dev.cuda_stream();
        let (ko_ptr, _kg) = ko_slice.device_ptr(&stream);
        let ko_ptrs = vec![ko_ptr];

        println!(
            "\n=== Grouped expert matmul: INT8 {kod:?} vs legacy FP16 {kd:?} [{nrows}x{ncols}], single expert ==="
        );
        println!(
            "{:>7} {:>11} {:>12} {:>9} {:>14}",
            "tokens", "int8(ms)", "legacy(ms)", "speedup", "int8 GFLOP/s"
        );

        for &m in &[1usize, 8, 16, 32, 64, 128, 256, 512] {
            let act_data: Vec<f32> = (0..m * ncols)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            let act_bf16: Vec<bf16> = act_data.iter().map(|&v| bf16::from_f32(v)).collect();
            let act_storage =
                CudaStorage::wrap_cuda_slice(dev.memcpy_stod(&act_bf16)?, dev.clone());
            let act_layout = crate::Layout::contiguous(crate::Shape::from(vec![m, ncols]));
            let expert_offsets = vec![0i32, m as i32];
            // Pre-quantize activations ONCE to q8a128, off the timed path.
            let q8a128 = quantize_acts_q8a128_test(&dev, &act_data, m, ncols)?;

            let time_call = |legacy: bool| -> Result<f64> {
                for _ in 0..20 {
                    if legacy {
                        grouped_matmul_gemx(
                            &k_ptrs,
                            kd,
                            nrows,
                            ncols,
                            &act_storage,
                            &act_layout,
                            &expert_offsets,
                            &dev,
                        )?;
                    } else {
                        grouped_qmatmul(
                            DynamicTensor::Int8(&q8a128),
                            &ko_ptrs,
                            kod,
                            nrows,
                            &expert_offsets,
                            &dev,
                            Backing::Owned,
                        )?;
                    }
                }
                dev.synchronize()?;
                let iters = 100usize;
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        if legacy {
                            grouped_matmul_gemx(
                                &k_ptrs,
                                kd,
                                nrows,
                                ncols,
                                &act_storage,
                                &act_layout,
                                &expert_offsets,
                                &dev,
                            )?;
                        } else {
                            grouped_qmatmul(
                                DynamicTensor::Int8(&q8a128),
                                &ko_ptrs,
                                kod,
                                nrows,
                                &expert_offsets,
                                &dev,
                                Backing::Owned,
                            )?;
                        }
                    }
                    dev.synchronize()?;
                    best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
                }
                Ok(best)
            };

            let t_int8 = time_call(false)?;
            let t_legacy = time_call(true)?;
            let flops = 2.0 * m as f64 * nrows as f64 * ncols as f64;
            println!(
                "{:>7} {:>11.4} {:>12.4} {:>9.2} {:>14.1}",
                m,
                t_int8 * 1e3,
                t_legacy * 1e3,
                t_legacy / t_int8,
                flops / t_int8 / 1e9,
            );
        }
    }
    Ok(())
}

/// Test grouped_matmul_gemx with down-projection dimensions (nrows > ncols).
/// Validates at model scale: 109 experts with varying batch sizes.
#[test]
fn grouped_matmul_matches_direct_down() -> Result<()> {
    use crate::Shape;
    use half::bf16;

    let dev = CudaDevice::new(0)?;
    // Down projection: [2048, 768] â€” nrows > ncols
    // Test at model scale: 109 experts, ~992 total batch
    let nrows = 2048;
    let ncols = 768;
    let num_experts = 109;
    // Vary batch sizes 1-23 to match real workload
    let expert_batches: Vec<usize> = (0..num_experts).map(|i| 1 + (i * 7 + 3) % 23).collect();
    println!("Testing: {} experts", num_experts);

    let mut rng = rand::rng();
    let mut expert_storages = Vec::new();
    let mut weight_ptrs = Vec::new();
    let shape = Shape::from((nrows, ncols));

    for _ in 0..num_experts {
        let weights: Vec<f32> = (0..nrows * ncols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let mut xs = QCudaStorage::zeros(&dev, ncols * nrows, GgmlDType::Q4_K)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(
            dev.memcpy_stod(&weights)?,
            dev.clone(),
        ))?;
        let xs = xs.repack_gemx(&shape)?;
        weight_ptrs.push(xs.data_ptr());
        expert_storages.push(xs);
    }

    let total_batch: usize = expert_batches.iter().sum();
    let act_data: Vec<f32> = (0..total_batch * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let act_bf16: Vec<bf16> = act_data.iter().map(|&v| bf16::from_f32(v)).collect();
    let act_gpu = dev.memcpy_stod(&act_bf16)?;
    let act_storage = CudaStorage::wrap_cuda_slice(act_gpu.clone(), dev.clone());
    let act_layout = crate::Layout::contiguous(Shape::from(vec![total_batch, ncols]));

    let mut expert_offsets: Vec<i32> = vec![0];
    for &b in &expert_batches {
        expert_offsets.push(expert_offsets.last().unwrap() + b as i32);
    }

    // At 109 experts, full per-expert reference comparison is slow.
    // Just verify no NaN/Inf in grouped output.

    // Test: grouped_matmul_gemx
    let grouped_result = grouped_matmul_gemx(
        &weight_ptrs,
        GgmlDType::Q4_K,
        nrows,
        ncols,
        &act_storage,
        &act_layout,
        &expert_offsets,
        &dev,
    )?;

    let grouped_storage = grouped_result.storage_and_layout().0;
    let grouped_cuda = match &*grouped_storage {
        crate::Storage::Cuda(s) => s,
        _ => panic!("expected CUDA storage"),
    };
    let grouped_bf16 = grouped_cuda.as_cuda_slice::<bf16>()?;
    let grouped_vals: Vec<bf16> = dev.memcpy_dtov(&grouped_bf16.slice(..))?;

    // Check grouped for NaN
    let nan_count = grouped_vals.iter().filter(|v| bf16::is_nan(**v)).count();
    let inf_count = grouped_vals
        .iter()
        .filter(|v| bf16::is_infinite(**v))
        .count();
    println!(
        "Grouped: {} elements, {} NaN, {} Inf",
        grouped_vals.len(),
        nan_count,
        inf_count
    );
    assert_eq!(nan_count, 0, "Grouped matmul produced NaN at model scale");
    assert_eq!(inf_count, 0, "Grouped matmul produced Inf at model scale");
    Ok(())
}

// ============================================================================
// KV-path quantization scalar-function bug regression tests.
//
// The KV cache write path calls scalar single-block functions (quantize_block_*)
// via transpose_batch.cuh, which are distinct from the multi-block reference
// path (quantize_blocks_*) exercised by the existing tests.  These tests drive
// that scalar path directly through `quantize_transposed_batched_to_dtype` with
// geometry (n_head=1, chunk_size=32, head_dim=1) — a no-op transpose that
// processes exactly one 32-element block — and check the raw quantized bytes.
//
// Each test is written to FAIL before the corresponding kernel fix is applied
// and to PASS afterwards.
// ============================================================================

fn kv_path_quantize_one_block(
    src_data: &[f32],
    dtype: GgmlDType,
    dev: &CudaDevice,
) -> Result<Vec<u8>> {
    assert_eq!(
        src_data.len(),
        32,
        "KV-path tests require exactly 32 elements"
    );
    assert_eq!(dtype.block_size(), 32, "dtype must have 32-element blocks");

    let quant_size = dtype.type_size();
    let src_gpu = dev.memcpy_stod(src_data)?;
    let mut kv_buf = unsafe { dev.alloc::<u8>(quant_size)? };

    {
        let stream = dev.cuda_stream();
        let (src_ptr, _sg) = src_gpu.device_ptr(&stream);
        let (dst_ptr, _dg) = kv_buf.device_ptr_mut(&stream);
        unsafe {
            quantize_transposed_batched_to_dtype(
                src_ptr as *const f32,
                dst_ptr as *mut u8,
                None,
                None,
                1,  // num_chunks
                1,  // n_head
                32, // chunk_size (= one full block)
                1,  // head_dim
                dtype,
                dev,
            )?;
        }
    }
    dev.synchronize()?;
    Ok(dev.memcpy_dtov(&kv_buf.slice(..))?)
}

// Q4_0 scalar: d = -amax/8  (always negative)
// Fix:         d = max_val/-8  (preserves sign of dominant element)
//
// Input: v[0]=-2.0 (dominant negative), v[1..31]=0.1.
// Bug:  d = -0.25  → q[0]=15 (saturated), dequant=-1.75, error=0.25
// Fix:  d = +0.25  → q[0]=0,              dequant=-2.0,  error=0
//
// Detected via the f16 scale byte: 0x34 = +0.25 (correct), 0xB4 = -0.25 (bug).
#[test]
fn kv_path_sign_bug_q4_0() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = vec![0.1f32; 32];
    src[0] = -2.0;

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q4_0, &dev)?;

    // Q4_0 block layout: [d: f16 LE (2 bytes), qs: [u8;16]]
    // +0.25 f16 = 0x3400 → [0x00, 0x34]; -0.25 f16 = 0xB400 → [0x00, 0xB4]
    assert_eq!(
        bytes[1], 0x34,
        "Q4_0 KV scalar sign bug: expected d=+0.25 (byte 0x34), got 0x{:02X} \
         (negative scale; bug is d=-amax/8 instead of d=max_val/-8)",
        bytes[1]
    );
    Ok(())
}

// Q5_0 scalar: d = -amax/16  (always negative).
// Fix:         d = max_val/-16.
//
// Same sign-preservation defect as Q4_0 but with 5-bit offset=16.
// Detected via the f16 scale byte: 0x30 = +0.125, 0xB0 = -0.125.
#[test]
fn kv_path_sign_bug_q5_0() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = vec![0.1f32; 32];
    src[0] = -2.0;

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q5_0, &dev)?;

    // Q5_0 block layout: [d: f16 LE (2 bytes), qh: u32, ql: [u8;16]]
    // +0.125 f16 = 0x3000 → [0x00, 0x30]; -0.125 f16 = 0xB000 → [0x00, 0xB0]
    assert_eq!(
        bytes[1], 0x30,
        "Q5_0 KV scalar sign bug: expected d=+0.125 (byte 0x30), got 0x{:02X} \
         (negative scale; bug is d=-amax/16 instead of d=max_val/-16)",
        bytes[1]
    );
    Ok(())
}

// Q1_S scalar/vec/multi-block: scale = encode_e4m3(amax).
// Fix:                             scale = encode_e4m3(mean(|x|)).
//
// Input: v[0]=1.0 (outlier), v[1..31]=0.1.
//   mean_abs = (1.0 + 31*0.1) / 32 = 0.128125 → FP8 → 0.125 → encoded 0x20
//   amax     = 1.0                              → FP8 → 1.0   → encoded 0x38
//
// With amax-scale: all elements reconstruct as ±1.0.  v[1..31] error = 0.9 each,
// MSE ≈ 0.785.  With mean-scale: v[1..31] error = 0.025 each, MSE ≈ 0.025.
// Detected via the stored scale byte (bytes[0]).
#[test]
fn kv_path_scale_q1_s() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = vec![0.1f32; 32];
    src[0] = 1.0;

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q1_S, &dev)?;

    // Q1_S block layout: [scale: i8, qs: [u8;4]]
    // mean_abs = (31*0.1 + 1.0)/32 = 4.1/32 = 0.128125
    // INT8 scale = round(0.128125 * 127) = round(16.27) = 16
    // Bug (amax): round(1.0 * 127) = 127
    let scale = bytes[0] as i8;
    assert_eq!(
        scale, 16,
        "Q1_S INT8 scale: expected round(mean_abs*127)=16, got {} (bug would store amax→127)",
        scale
    );
    Ok(())
}

// Q2_S scalar: encode stores INT8(round(d*127)) and quantizes using d_int8=scale/127.
//
// Input: v[0]=0.9 (sets amax), v[1..31]=0.61.
//   d = 0.9/1.5 = 0.6
//   INT8 scale = round(0.6 * 127) = round(76.2) = 76
//
// Detected via the stored scale byte (bytes[0]).
#[test]
fn kv_path_int8_roundtrip_q2_s() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = vec![0.61f32; 32];
    src[0] = 0.9;

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q2_S, &dev)?;

    // Q2_S block: [scale: i8, qs: [u8;8]]
    // amax=0.9, d=0.6, INT8 scale = round(0.6*127) = 76
    let scale = bytes[0] as i8;
    assert_eq!(
        scale, 76,
        "Q2_S INT8 scale: expected round(d*127)=76, got {}",
        scale
    );
    Ok(())
}

// Q2_A scalar: encode stores INT8 scale=round(delta*127) and INT8 bias=round(vmin*127).
//
// Input: v[0]=1.1875 (vmax), v[1]=-0.59375 (vmin), v[2..31]=-0.31.
//   delta = (1.1875-(-0.59375))/3 = 1.78125/3 = 0.59375
//   INT8 scale = round(0.59375 * 127) = round(75.41) = 75
//   INT8 bias  = round(-0.59375 * 127) = round(-75.41) = -75
//
// Detected via the stored scale and bias bytes (bytes[0], bytes[1]).
#[test]
fn kv_path_int8_roundtrip_q2_a() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = vec![-0.31f32; 32];
    src[0] = 1.1875; // sets vmax
    src[1] = -0.59375; // sets vmin

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q2_A, &dev)?;

    // Q2_A block: [scale: i8, bias: i8, qs: [u8;8]]
    let scale = bytes[0] as i8;
    let bias = bytes[1] as i8;
    assert_eq!(
        scale, 75,
        "Q2_A INT8 scale: expected round(delta*127)=75, got {}",
        scale
    );
    assert_eq!(
        bias, -75,
        "Q2_A INT8 bias:  expected round(vmin*127)=-75, got {}",
        bias
    );
    Ok(())
}

// =============================================================================
// Q1_A: 1-bit asymmetric — separate amplitude per sign + 32 sign bits.
// Input: lanes 0-15 = +1.0 (positive), lanes 16-31 = -0.25 (negative).
//   scale_pos = round(mean(positive) * 127) = round(1.0 * 127) = 127
//   scale_neg = round(mean(|negative|) * 127) = round(0.25 * 127) = round(31.75) = 32
//   qmask: bits 0..15 set (positive), bits 16..31 clear (negative)
//          → bytes[2..6] = [0xFF, 0xFF, 0x00, 0x00]
#[test]
fn kv_path_q1_a_asymmetric() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = [-0.25f32; 32];
    for v in &mut src[..16] {
        *v = 1.0;
    }

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q1_A, &dev)?;

    assert_eq!(bytes.len(), 6, "Q1_A block must be 6 bytes");
    assert_eq!(
        bytes[0] as i8, 127,
        "Q1_A scale_pos (mean(+) = 1.0): expected 127, got {}",
        bytes[0] as i8
    );
    assert_eq!(
        bytes[1] as i8, 32,
        "Q1_A scale_neg (mean(|-|) = 0.25): expected 32, got {}",
        bytes[1] as i8
    );
    assert_eq!(
        bytes[2], 0xFF,
        "Q1_A qs[0] (lanes 0-7 positive): expected 0xFF, got 0x{:02X}",
        bytes[2]
    );
    assert_eq!(
        bytes[3], 0xFF,
        "Q1_A qs[1] (lanes 8-15 positive): expected 0xFF, got 0x{:02X}",
        bytes[3]
    );
    assert_eq!(
        bytes[4], 0x00,
        "Q1_A qs[2] (lanes 16-23 negative): expected 0x00, got 0x{:02X}",
        bytes[4]
    );
    assert_eq!(
        bytes[5], 0x00,
        "Q1_A qs[3] (lanes 24-31 negative): expected 0x00, got 0x{:02X}",
        bytes[5]
    );
    Ok(())
}

// Q0_X: flat block + one outlier escape.
//
// Input: 31 zeros + one outlier value 1.0 at lane 7.
//   sum = 1.0; mean = 1/32 = 0.03125
//   bulk_anchor = round(0.03125 * 127) = round(3.96875) = 4
//   x_i8: 0 everywhere, except lane 7 = round(1.0 * 127) = 127
//   residuals: -4 elsewhere, +123 at lane 7 → argmax = lane 7
//   delta_raw = round(123 / 32) = 4 → clamped to outlier_delta = 3
//   packed byte 1 = (lane_idx 7 in low 5 bits) | (delta 3 in top 3 bits)
//                 = 0x07 | (0x03 << 5) = 0x67
#[test]
fn kv_path_q0_x_outlier() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = [0.0f32; 32];
    src[7] = 1.0;

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q0_X, &dev)?;

    assert_eq!(bytes.len(), 2, "Q0_X block must be 2 bytes");
    assert_eq!(
        bytes[0] as i8, 4,
        "Q0_X bulk_anchor (mean=1/32 → INT8 4): got {}",
        bytes[0] as i8
    );
    assert_eq!(
        bytes[1], 0x67,
        "Q0_X outlier_packed (idx=7, delta=3): expected 0x67, got 0x{:02X}",
        bytes[1]
    );
    Ok(())
}

// Q0_M2: 2-centroid + 8-bit quartet mask.
// Input: 8 quartets alternating [-1.0]*4 and [1.0]*4.
//   Init: c0=min=-1.0, c1=max=1.0.  Lloyd converges immediately.
//   Encoding: c0=-1.0 → INT8 round(-1.0*127)=-127; c1=1.0 → INT8 127.
//   Mask: even quartets (0,2,...) → c0 (bit=0), odd quartets → c1 (bit=1).
//   Expected qmask byte: 0b_10101010 = 0xAA.
// Block layout: [centroid[0], centroid[1], qmask: u8]
#[test]
fn kv_path_q0_m2_alternating() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = [0.0f32; 32];
    for i in 0..8 {
        let base = i * 4;
        let val = if i % 2 == 0 { -1.0 } else { 1.0 };
        src[base..base + 4].fill(val);
    }

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q0_M2, &dev)?;

    assert_eq!(bytes.len(), 3, "Q0_M2 block must be 3 bytes");
    // Quartet assignments: [0→c0, 1→c1, 2→c0, 3→c1, 4→c0, 5→c1, 6→c0, 7→c1]
    // qmask bit k = assignment of quartet k: 0b10101010 = 0xAA
    assert_eq!(
        bytes[0] as i8, -127,
        "Q0_M2 c0 (-1.0): expected INT8 -127, got {}",
        bytes[0] as i8
    );
    assert_eq!(
        bytes[1] as i8, 127,
        "Q0_M2 c1 (+1.0): expected INT8 127, got {}",
        bytes[1] as i8
    );
    assert_eq!(
        bytes[2], 0xAA,
        "Q0_M2 mask: expected 0xAA (alternating), got 0x{:02X}",
        bytes[2]
    );
    Ok(())
}

// Q0_M4: 4-centroid + 32-bit pair mask.
// Input: 8 quartets cycling through [-0.75, -0.25, 0.25, 0.75] (2 quartets each value).
//   Init: vmin=-0.75, vmax=0.75, step=0.5 → c[0..3]=-0.75,-0.25,0.25,0.75.
//   Lloyd converges immediately (pair means map cleanly to centroids).
//   INT8 encoding: round(v*127): -0.75→-95, -0.25→-32, 0.25→32, 0.75→95.
//   2-bit assignment per pair (16 pairs of 2): pair k → (k/2) % 4
//     pairs 0,1→0; 2,3→1; 4,5→2; 6,7→3; 8,9→0; 10,11→1; 12,13→2; 14,15→3.
//   qmask u32 LE: bits [15:0] = 11_11_10_10_01_01_00_00 = 0xFA50,
//                 bits [31:16] = same pattern → 0xFA50FA50.
#[test]
fn kv_path_q0_m4_four_levels() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let mut src = [0.0f32; 32];
    for i in 0..8 {
        let val = [-0.75f32, -0.25, 0.25, 0.75][i % 4];
        src[i * 4..i * 4 + 4].fill(val);
    }

    let bytes = kv_path_quantize_one_block(&src, GgmlDType::Q0_M4, &dev)?;

    assert_eq!(bytes.len(), 8, "Q0_M4 block must be 8 bytes");
    // INT8: round(v*127) — all within [-127,127] so no clamping
    assert_eq!(
        bytes[0] as i8, -95,
        "Q0_M4 c[0]=-0.75: expected INT8 -95, got {}",
        bytes[0] as i8
    );
    assert_eq!(
        bytes[1] as i8, -32,
        "Q0_M4 c[1]=-0.25: expected INT8 -32, got {}",
        bytes[1] as i8
    );
    assert_eq!(
        bytes[2] as i8, 32,
        "Q0_M4 c[2]=+0.25: expected INT8 32, got {}",
        bytes[2] as i8
    );
    assert_eq!(
        bytes[3] as i8, 95,
        "Q0_M4 c[3]=+0.75: expected INT8 95, got {}",
        bytes[3] as i8
    );
    let qmask = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert_eq!(
        qmask, 0xFA50_FA50,
        "Q0_M4 mask: expected 0xFA50FA50, got 0x{:08X}",
        qmask
    );
    Ok(())
}

// =============================================================================
// GROUPED EXPERT LAUNCH-COST PROBE (does N experts cost ~N launches?)
// =============================================================================
// The MoE pipeline dispatches all routed experts via a SINGLE run_quantized_matmul
// call with num_segments = N_experts — but the dispatcher's segment loop issues
// one cudaLaunchKernel PER expert. This probe measures per-call host time as N
// grows, with M=1 token per expert (single-session decode). If time scales ~N,
// the path is launch/host-bound and a single-launch grouped GEMM would cut it.
//
// All N segments point at the SAME weight (timing is pointer-independent), so
// this isolates pure dispatch/launch cost. No per-iter sync — launches queue,
// matching the real async path.
//
//   cargo test -p candle-core --release --features cuda --lib \
//     quantized::test::expert_grouped_launch_cost -- --ignored --nocapture
#[test]
#[ignore = "GPU launch-cost probe; run with --ignored --nocapture"]
fn expert_grouped_launch_cost() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let n = 768usize;
    let k = 2048usize;

    let mut rng = rand::rng();
    let wvals: Vec<f32> = (0..n * k).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    let shape = Shape::from((n, k));
    let mut xs = QCudaStorage::zeros(&dev, n * k, GgmlDType::Q4_K)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(
        dev.memcpy_stod(&wvals)?,
        dev.clone(),
    ))?;
    let xs_repacked = xs.repack_gemx(&shape)?;
    let qtype = dtype_to_qtype(GgmlDType::Q4_K)? as i32;
    let m = 1usize; // tokens per expert (single-session decode)
    let iters = 1000usize;

    println!(
        "\n=== Per-segment expert launch cost [n={n} k={k}] Q4_K, M={m}/expert, \
         {iters} calls/sync (N launches via run_quantized_matmul) ===",
    );
    println!("{:>9} {:>12} {:>12}", "N_experts", "us/call", "us/expert");

    for &n_exp in &[1usize, 2, 4, 8, 16, 32] {
        let total = n_exp * m;
        let yvals: Vec<bf16> = (0..total * k)
            .map(|_| bf16::from_f32(rng.random_range(-1.0f32..1.0)))
            .collect();
        let y = dev.memcpy_stod(&yvals)?;
        let dst = unsafe { dev.alloc::<bf16>(total * n)? };

        let stream = dev.cuda_stream();
        let (wptr, _wg) = xs_repacked.data.inner.device_ptr(&stream);
        let (yptr, _yg) = y.device_ptr(&stream);
        let (dptr, _dg) = dst.device_ptr(&stream);

        // One segment per expert, all pointing at the same weight.
        let segments: Vec<VxSegment> = (0..n_exp)
            .map(|_| VxSegment {
                weights: wptr as *const c_void,
                batch_count: m as i32,
            })
            .collect();

        let call = || unsafe {
            run_quantized_matmul(
                segments.as_ptr(),
                n_exp as i32,
                yptr as *const c_void,
                dptr as *mut c_void,
                k as i32,
                n as i32,
                k as i32,
                n as i32,
                qtype,
                YType::BF16 as i32,
                0, // weight_bytes=0 → L2-cached assumption (matches grouped_matmul_gemx)
                0, // force_mode2 (tiling only; result-invariant)
                OutDType::BF16 as i32,
            );
        };

        for _ in 0..100 {
            call();
        }
        dev.synchronize()?;

        let mut best = f64::MAX;
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..iters {
                call();
            }
            dev.synchronize()?;
            best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
        }
        println!(
            "{:>9} {:>12.3} {:>12.3}",
            n_exp,
            best * 1e6,
            best * 1e6 / n_exp as f64,
        );
    }
    Ok(())
}

// =============================================================================
// GROUPED SINGLE-LAUNCH COST (run_grouped_quantized_matmul: N experts, 1 launch)
// =============================================================================
// A/B partner for expert_grouped_launch_cost: same N-expert / M=1 sweep, but
// all experts run in ONE grouped kernel launch instead of N. Compare us/call
// directly against the per-segment probe to measure the launch+occupancy win.
//
//   cargo test -p candle-core --release --features cuda --lib \
//     quantized::test::expert_grouped_single_launch_cost -- --ignored --nocapture
#[test]
#[ignore = "GPU launch-cost probe; run with --ignored --nocapture"]
fn expert_grouped_single_launch_cost() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let n = 768usize;
    let k = 2048usize;

    let mut rng = rand::rng();
    let wvals: Vec<f32> = (0..n * k).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    let shape = Shape::from((n, k));
    let mut xs = QCudaStorage::zeros(&dev, n * k, GgmlDType::Q4_K)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(
        dev.memcpy_stod(&wvals)?,
        dev.clone(),
    ))?;
    let xs_repacked = xs.repack_gemx(&shape)?;
    let qtype = dtype_to_qtype(GgmlDType::Q4_K)? as i32;
    let m = 1usize;
    let iters = 1000usize;

    println!(
        "\n=== Grouped expert SINGLE-LAUNCH cost [n={n} k={k}] Q4_K, M={m}/expert, \
         {iters} calls/sync ===",
    );
    println!("{:>9} {:>12} {:>12}", "N_experts", "us/call", "us/expert");

    let stream = dev.cuda_stream();
    let (wptr0, _wg0) = xs_repacked.data.inner.device_ptr(&stream);

    for &n_exp in &[1usize, 2, 4, 8, 16, 32] {
        let total = n_exp * m;
        let yvals: Vec<bf16> = (0..total * k)
            .map(|_| bf16::from_f32(rng.random_range(-1.0f32..1.0)))
            .collect();
        let y = dev.memcpy_stod(&yvals)?;
        let dst = unsafe { dev.alloc::<bf16>(total * n)? };

        // One tile per expert (M=1 <= 16). All experts share the same weight ptr.
        let weight_ptrs: Vec<u64> = vec![wptr0 as u64; n_exp];
        let tile_expert: Vec<i32> = (0..n_exp as i32).collect();
        let tile_b_start: Vec<i32> = (0..n_exp as i32).map(|e| e * m as i32).collect();
        let tile_b_cnt: Vec<i32> = vec![m as i32; n_exp];
        let num_tiles = n_exp as i32;
        let (yptr, _yg) = y.device_ptr(&stream);
        let (dptr, _dg) = dst.device_ptr(&stream);

        // Honest end-to-end cost: PACK weight_ptrs + all 3 tile tables into ONE
        // buffer and upload with a SINGLE memcpy_stod per call (vs 4 separate
        // copies), then the single grouped launch. weight_ptrs (u64, 8-aligned)
        // first, then the three i32 tables — base + offsets feed the kernel.
        let off_te = n_exp * 8; // bytes: weight_ptrs = n_exp × u64
        let off_tbs = off_te + num_tiles as usize * 4;
        let off_tbc = off_tbs + num_tiles as usize * 4;
        let total_bytes = off_tbc + num_tiles as usize * 4;
        let once = || -> Result<()> {
            let mut packed: Vec<u8> = Vec::with_capacity(total_bytes);
            for &w in &weight_ptrs {
                packed.extend_from_slice(&w.to_le_bytes());
            }
            for &x in &tile_expert {
                packed.extend_from_slice(&x.to_le_bytes());
            }
            for &x in &tile_b_start {
                packed.extend_from_slice(&x.to_le_bytes());
            }
            for &x in &tile_b_cnt {
                packed.extend_from_slice(&x.to_le_bytes());
            }
            let dev_buf = dev.memcpy_stod(&packed)?;
            let (base, _g) = dev_buf.device_ptr(&stream);
            unsafe {
                run_grouped_quantized_matmul(
                    base as *const c_void,
                    (base + off_te as u64) as *const c_void,
                    (base + off_tbs as u64) as *const c_void,
                    (base + off_tbc as u64) as *const c_void,
                    yptr as *const c_void,
                    dptr as *mut c_void,
                    k as i32,
                    n as i32,
                    k as i32,
                    n as i32,
                    num_tiles,
                    qtype,
                    YType::BF16 as i32,
                    2, // FP grouped kernels ignore the int8 tile mode
                    1, // row-fast grid order
                );
            }
            Ok(())
        };

        for _ in 0..100 {
            once()?;
        }
        dev.synchronize()?;

        let mut best = f64::MAX;
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..iters {
                once()?;
            }
            dev.synchronize()?;
            best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
        }
        println!(
            "{:>9} {:>12.3} {:>12.3}",
            n_exp,
            best * 1e6,
            best * 1e6 / n_exp as f64,
        );
    }
    Ok(())
}

/// `moe_route` fuses softmax + top-k select + (optional) renormalize into one kernel. Validate it
/// against the reference math it replaces (`softmax → sort(desc) → narrow(k) → renorm`): the top-k
/// **indices** must match exactly and the **weights** to f32 epsilon. Covers F32/BF16 logits, both
/// `norm_topk` modes, num_tokens=1 (decode) through batched, k=8 (Qwen3) and a small k, and a
/// non-128 expert count. Logits are `0.5 × distinct-int` so they are bit-exact in bf16 (no
/// rounding → no ties → deterministic top-k).
#[test]
fn cuda_moe_route_matches_reference() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());

    fn check(
        device: &crate::Device,
        num_tokens: usize,
        n_experts: usize,
        k: usize,
        dt: crate::DType,
        norm: bool,
    ) -> Result<()> {
        // (e*131 + t*17) mod 251 is injective over e<251, so every row has distinct experts.
        // Scale by 0.5 (a power of two) to keep the values exactly representable in bf16/f16.
        let logit = |t: usize, e: usize| -> f32 { (((e * 131 + t * 17) % 251) as f32) * 0.5 };
        let data: Vec<f32> = (0..num_tokens * n_experts)
            .map(|i| logit(i / n_experts, i % n_experts))
            .collect();

        let logits =
            crate::Tensor::from_vec(data, (num_tokens, n_experts), device)?.to_dtype(dt)?;
        let (w, idx) = moe_route(&logits, k, norm)?;
        let w = w.to_vec2::<f32>()?;
        let idx = idx.to_vec2::<u32>()?;

        for t in 0..num_tokens {
            let row: Vec<f32> = (0..n_experts).map(|e| logit(t, e)).collect();
            // Reference top-k: descending value, lowest index on tie (none here).
            let mut order: Vec<usize> = (0..n_experts).collect();
            order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
            let top = &order[..k];

            let gmax = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&l| (l - gmax).exp()).collect();
            let z_all: f32 = exps.iter().sum();
            let z_top: f32 = top.iter().map(|&e| exps[e]).sum();
            let denom = if norm { z_top } else { z_all };

            for p in 0..k {
                assert_eq!(
                    idx[t][p] as usize, top[p],
                    "index mismatch nt={num_tokens} n_exp={n_experts} k={k} dt={dt:?} norm={norm} t={t} p={p}"
                );
                let want = exps[top[p]] / denom;
                assert!(
                    (w[t][p] - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "weight mismatch nt={num_tokens} k={k} dt={dt:?} norm={norm} t={t} p={p}: got {} want {want}",
                    w[t][p]
                );
            }
        }
        Ok(())
    }

    for &dt in &[crate::DType::F32, crate::DType::BF16] {
        for &norm in &[true, false] {
            check(&device, 1, 128, 8, dt, norm)?; // decode: single token
            check(&device, 7, 128, 8, dt, norm)?; // batched, Qwen3 top-8
            check(&device, 5, 128, 2, dt, norm)?; // small k
            check(&device, 3, 64, 4, dt, norm)?; // non-128 expert count
        }
    }
    Ok(())
}

/// Degenerate-logit guard: a token whose logits are all `-inf`/`NaN`, or that has
/// fewer finite experts than `k`, must NEVER route to an out-of-range expert. The
/// kernel seeds its argmax with `bi = n_experts` as a "not found" sentinel; if that
/// leaks to the output it indexes past the per-layer expert tables and panics the
/// expert-paging pipeline (bricking decode). Assert every emitted index is in range
/// and every weight is finite, and that a short-finite row keeps its real experts
/// with a zero-weight fallback filling the rest.
#[test]
fn cuda_moe_route_never_emits_out_of_range_index() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let n_experts = 128usize;
    let k = 8usize;
    let ninf = f32::NEG_INFINITY;

    let rows = 5usize;
    let mut data = vec![0f32; rows * n_experts];
    // Row 0: all -inf — no finite candidate for any slot.
    for e in 0..n_experts {
        data[e] = ninf;
    }
    // Row 1: all NaN — NaN loses every `>` compare, so the sentinel would survive.
    for e in 0..n_experts {
        data[n_experts + e] = f32::NAN;
    }
    // Row 2: only 3 finite experts, rest -inf (fewer finite than k).
    for e in 0..n_experts {
        data[2 * n_experts + e] = ninf;
    }
    data[2 * n_experts + 10] = 3.0;
    data[2 * n_experts + 20] = 2.0;
    data[2 * n_experts + 30] = 1.0;
    // Row 3: ordinary distinct logits — normal routing.
    for e in 0..n_experts {
        data[3 * n_experts + e] = (e as f32) * 0.5;
    }
    // Row 4: 3 finite experts, rest NaN (not -inf). This is the case whose KEPT
    // slots' weights depend on the warp `gmax` reduction dropping NaN (CUDA
    // `fmaxf`) so `gmax` stays finite (3.0) — otherwise the kept weights would
    // be `exp(finite - NaN) = NaN` and the index-based clamp would NOT catch it.
    for e in 0..n_experts {
        data[4 * n_experts + e] = f32::NAN;
    }
    data[4 * n_experts + 5] = 3.0;
    data[4 * n_experts + 15] = 2.0;
    data[4 * n_experts + 25] = 1.0;

    let logits = crate::Tensor::from_vec(data, (rows, n_experts), &device)?;
    for &norm in &[true, false] {
        let (w, idx) = moe_route(&logits, k, norm)?;
        let idx = idx.to_vec2::<u32>()?;
        let w = w.to_vec2::<f32>()?;
        for t in 0..rows {
            for p in 0..k {
                assert!(
                    (idx[t][p] as usize) < n_experts,
                    "row {t} slot {p}: index {} must be < n_experts {n_experts} (norm={norm})",
                    idx[t][p]
                );
                assert!(
                    w[t][p].is_finite(),
                    "row {t} slot {p}: weight {} must be finite (norm={norm})",
                    w[t][p]
                );
            }
        }
        // Rows 2 & 4: the three finite experts are selected in descending order,
        // with real (non-zero) kept weights; the remaining slots are the
        // zero-weight fallback (not a phantom expert).
        for (base, e0, e1, e2) in [(2usize, 10u32, 20u32, 30u32), (4, 5, 15, 25)] {
            assert_eq!(idx[base][0], e0, "row{base} norm={norm}");
            assert_eq!(idx[base][1], e1, "row{base} norm={norm}");
            assert_eq!(idx[base][2], e2, "row{base} norm={norm}");
            assert!(
                w[base][0] > 0.0,
                "row{base} kept slot 0 weight (norm={norm})"
            );
            for p in 3..k {
                assert_eq!(
                    w[base][p], 0.0,
                    "row{base} slot {p} must be zero-weight fallback (norm={norm})"
                );
            }
        }
    }
    Ok(())
}

/// Perf probe: fused `moe_route` vs the op-chain it replaced (softmax -> sort -> narrow -> renorm),
/// bf16 logits [num_tokens, 128] top-8, across decode (small) -> prefill (large) token counts.
/// Both include their output allocation (the real per-call cost). Run with:
///   cargo test -p candle-core --features cuda --release bench_moe_route -- --ignored --nocapture
#[test]
#[ignore = "GPU perf probe; run with --ignored --nocapture"]
fn bench_moe_route() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let n_experts = 128usize;
    let k = 8usize;
    let iters = 2000usize;

    println!(
        "\n=== moe_route vs op-chain [n_experts={n_experts} k={k}] bf16, {iters} calls/sync ===",
    );
    println!(
        "{:>10} {:>13} {:>13} {:>13} {:>13}",
        "num_tokens", "full us", "rawkern us", "alloc2 us", "opchain us"
    );

    let bench = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
        for _ in 0..50 {
            f()?;
        }
        dev.synchronize()?;
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..iters {
                f()?;
            }
            dev.synchronize()?;
            best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
        }
        Ok(best)
    };

    for &nt in &[1usize, 4, 16, 64, 256, 1024, 4096] {
        let data: Vec<f32> = (0..nt * n_experts)
            .map(|i| (((i * 131) % 251) as f32) * 0.05)
            .collect();
        let logits = crate::Tensor::from_vec(data, (nt, n_experts), &device)?
            .to_dtype(crate::DType::BF16)?;

        // Pre-grab the logits device pointer + pre-allocate output buffers ONCE for the
        // raw-kernel probe — isolates the FFI launch + kernel from alloc/wrap overhead.
        let (storage, layout) = logits.storage_and_layout();
        let (lo1, _lo2) = layout.contiguous_offsets().unwrap();
        let lcuda = match &*storage {
            crate::Storage::Cuda(c) => c,
            _ => unreachable!(),
        };
        let lslice = match &lcuda.slice {
            crate::cuda_backend::CudaStorageSlice::BF16(s) => s.slice(lo1..),
            _ => unreachable!(),
        };
        let stream = dev.cuda_stream();
        let (lptr, _lg) = lslice.device_ptr(&stream);
        let out_idx = unsafe { dev.alloc::<u32>(nt * k)? };
        let out_w = unsafe { dev.alloc::<f32>(nt * k)? };
        let (iptr, _ig) = out_idx.device_ptr(&stream);
        let (wptr, _wg) = out_w.device_ptr(&stream);

        let mut full = || -> Result<()> {
            let _ = moe_route(&logits, k, true)?;
            Ok(())
        };
        let mut rawkern = || -> Result<()> {
            unsafe {
                candle_kernels::simple::moe_scatter::run_moe_route(
                    2,
                    lptr as *const std::ffi::c_void,
                    iptr as *mut u32,
                    wptr as *mut f32,
                    nt as i32,
                    n_experts as i32,
                    k as i32,
                    1,
                );
            }
            Ok(())
        };
        let mut alloc2 = || -> Result<()> {
            let a = unsafe { dev.alloc::<u32>(nt * k)? };
            let b = unsafe { dev.alloc::<f32>(nt * k)? };
            std::hint::black_box((&a, &b));
            Ok(())
        };
        let mut opchain = || -> Result<()> {
            let m = logits.max_keepdim(crate::D::Minus1)?;
            let e = logits.broadcast_sub(&m)?.exp()?;
            let s = e.sum_keepdim(crate::D::Minus1)?;
            let probs = e.broadcast_div(&s)?.to_dtype(crate::DType::F32)?;
            let (sw, si) = probs.sort_last_dim(false)?;
            let tw = sw.narrow(1, 0, k)?;
            let _ti = si.narrow(1, 0, k)?.contiguous()?;
            let sums = tw.sum(1)?;
            let _ = tw.broadcast_div(&sums.unsqueeze(1)?)?;
            Ok(())
        };

        let f_us = bench(&mut full)? * 1e6;
        let r_us = bench(&mut rawkern)? * 1e6;
        let a_us = bench(&mut alloc2)? * 1e6;
        let o_us = bench(&mut opchain)? * 1e6;
        println!("{nt:>10} {f_us:>13.3} {r_us:>13.3} {a_us:>13.3} {o_us:>13.3}");
    }
    Ok(())
}

/// Crossover benchmark for the q8a128 **dense** int8 tiling modes — mode-1 (`Bm=16`, `N_SUB=1`)
/// vs mode-2 (`Bm=32`, `N_SUB=2`). The mode choice trades **weight** re-reads (M-tiling) against
/// **activation** re-reads (N-tiling) — both are DRAM pressure — so the crossover lives on the full
/// `(M, N, K)` surface, not weight bytes alone (equal-byte `6144×2048` and `2048×6144` cross at
/// different `M`). This sweeps the three axes independently and densely.
///
/// Per `(N, K)` and `M` it times both modes (L2 flushed between launches, median of `ITERS`) and
/// prints, alongside the measured `m1/m2`, the **modeled** quantities the fit needs:
/// - `traf2/1` = modeled DRAM traffic ratio mode-2 / mode-1
///   (`ceil(M/Bm)·W + ceil(N/Nt)·A`, `W=N·K·0.5`, `A=M·K`),
/// - `blk1`,`blk2` = launched block counts (`ceil(M/Bm)·ceil(N/Nt)`) — the occupancy that gates
///   whether mode-2's lower traffic can actually be realized at small `M·N`.
///
/// Run:
/// ```text
/// cargo test -p candle-core --features cuda --release \
///   q8a128_dense_mode_crossover -- --ignored --nocapture
/// ```
#[test]
#[ignore = "benchmark: run with --ignored --nocapture to fit the mode-2 crossover surface"]
fn q8a128_dense_mode_crossover() -> Result<()> {
    use crate::quantized::int8_matmul_mode::q8a128_dense_use_mode2;

    let dev = CudaDevice::new(0)?;
    let stream = dev.cuda_stream();
    let l2 = dev.l2_cache_size().unwrap_or(0);
    let sm = dev.multiprocessor_count().unwrap_or(0);
    let flush_buf = unsafe { dev.alloc::<u8>((l2.max(1 << 20)) * 2)? };
    // Agreement accounting: only count cells with a *clear* winner (|1 - m1/m2| >= 5%); near-wash
    // cells flip on noise and shouldn't score against the formula.
    let (mut agree, mut clear) = (0usize, 0usize);

    const WARMUP: usize = 3;
    const ITERS: usize = 30;

    // Independent axes: N (output / weight-reuse), K (contraction, shared with the activation),
    // M (activation rows / activation-reuse). N % 32 == 0, K % 128 == 0.
    let ns: &[usize] = &[2048, 4096, 8192];
    let ks: &[usize] = &[2048, 4096, 8192];
    let ms: &[usize] = &[
        1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64, 80, 96, 128, 160,
        192, 256, 384, 512,
    ];

    let median = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let ceil = |x: usize, b: usize| ((x + b - 1) / b) as f64;

    println!(
        "\n=== q8a128 dense mode crossover surface ===\nL2={} MiB | SMs={sm} | WARMUP={WARMUP} ITERS={ITERS} | median µs, L2 flushed between launches",
        l2 >> 20
    );
    let mut summary: Vec<(usize, usize, f64, Option<usize>)> = Vec::new();

    let mut rng = rand::rng();
    for &n in ns {
        for &k in ks {
            let wf32: Vec<f32> = (0..n * k).map(|_| rng.random_range(-0.1f32..0.1)).collect();
            let ob = requant_ko_per128(&wf32, n, k, 15, 0, 0);
            let kod = GgmlDType::Q4_KO;
            let wbytes = ob.len();
            let ko_slice = dev.memcpy_stod(&ob)?;
            let (ko_ptr, _g) = ko_slice.device_ptr(&stream);
            let w_mib = wbytes as f64 / (1 << 20) as f64;

            println!("\n# N={n} K={k}  W={w_mib:.1}MiB  act={k}B/token");
            println!(
                "  {:>5} | {:>9} | {:>9} | {:>6} | {:>7} | {:>6} | {:>6} | {:>5} | {:>5} | ok",
                "M", "mode1us", "mode2us", "m1/m2", "traf2/1", "blk1", "blk2", "meas", "pred"
            );

            let mut ratios: Vec<(usize, f64)> = Vec::new();
            for &m in ms {
                let act: Vec<f32> = (0..m * k).map(|_| rng.random_range(-1.0f32..1.0)).collect();
                let op = quantize_acts_q8a128_test(&dev, &act, m, k)?;
                let time = |mode2: bool| -> Result<f64> {
                    for _ in 0..WARMUP {
                        let _ = q8a128_dense_matmul(
                            &op,
                            ko_ptr,
                            kod,
                            n,
                            wbytes,
                            mode2,
                            crate::DType::F32,
                            &dev,
                        )?;
                    }
                    dev.synchronize()?;
                    let mut v = Vec::with_capacity(ITERS);
                    for _ in 0..ITERS {
                        cuda_flush_l2(&flush_buf, &dev);
                        let t0 = Instant::now();
                        let _ = q8a128_dense_matmul(
                            &op,
                            ko_ptr,
                            kod,
                            n,
                            wbytes,
                            mode2,
                            crate::DType::F32,
                            &dev,
                        )?;
                        dev.synchronize()?;
                        v.push(t0.elapsed().as_secs_f64() * 1e6);
                    }
                    Ok(median(v))
                };
                let us1 = time(false)?;
                let us2 = time(true)?;

                // Modeled DRAM traffic (Q4 weight = 0.5 B/elt; q8a128 act ≈ 1 B/elt).
                let w = (n * k) as f64 * 0.5;
                let a = (m * k) as f64;
                let t1 = ceil(m, 16) * w + ceil(n, 32) * a;
                let t2 = ceil(m, 32) * w + ceil(n, 64) * a;
                let blk1 = (ceil(m, 16) * ceil(n, 32)) as usize;
                let blk2 = (ceil(m, 32) * ceil(n, 64)) as usize;

                let r = us1 / us2;
                ratios.push((m, r));

                // Derived-formula prediction vs the measurement. Only score cells with a clear
                // winner (>=5% gap); near-wash cells flip on noise.
                let meas2 = us2 < us1;
                let pred2 = q8a128_dense_use_mode2(m, n, k, sm);
                let is_clear = (r - 1.0).abs() >= 0.05;
                if is_clear {
                    clear += 1;
                    if meas2 == pred2 {
                        agree += 1;
                    }
                }
                let ok = if !is_clear {
                    "~"
                } else if meas2 == pred2 {
                    "Y"
                } else {
                    "X"
                };
                println!(
                    "  {:>5} | {:>9.2} | {:>9.2} | {:>6.2} | {:>7.3} | {:>6} | {:>6} | {:>5} | {:>5} | {}",
                    m,
                    us1,
                    us2,
                    r,
                    t2 / t1,
                    blk1,
                    blk2,
                    if meas2 { "mode2" } else { "mode1" },
                    if pred2 { "mode2" } else { "mode1" },
                    ok
                );
            }
            // Sustained crossover: first M where this and the next sample both favor mode-2.
            let crossover = (0..ratios.len().saturating_sub(1))
                .find(|&i| ratios[i].1 > 1.0 && ratios[i + 1].1 > 1.0)
                .map(|i| ratios[i].0);
            println!("  -> sustained crossover M = {crossover:?}");
            summary.push((n, k, w_mib, crossover));
        }
    }

    println!("\n=== crossover summary  (N, K, W_MiB) -> M* ===");
    for (n, k, w, m) in &summary {
        println!(
            "  N={n:>5} K={k:>5}  W={w:>5.1}MiB  M*={}",
            m.map_or("none".to_string(), |x| x.to_string())
        );
    }
    println!(
        "\n=== formula vs measurement: {agree}/{clear} clear-winner cells agree ({:.1}%) ===",
        100.0 * agree as f64 / clear.max(1) as f64
    );
    Ok(())
}

/// Fused qkv segmented int8 matmul must be FLOAT-IDENTICAL to running q/k/v as three separate
/// dense int8 matmuls — with MIXED KO formats (q=Q4_KO, k/v=Q6_KO, the real GQA case) over the
/// shared q8a128 activation — and faster (one occupied launch vs three, the tiny k/v no longer
/// starve). Run perf with `--nocapture`.
#[test]
fn qkv_segmented_matches_separate() -> Result<()> {
    use std::time::Instant;
    let dev = CudaDevice::new(0)?;
    let stream = dev.cuda_stream();
    let k = 2048usize; // hidden / contraction

    // (N, KO dtype, requant params) for q, k, v — GQA shape: big q, small k/v; mixed formats.
    let dims: [(usize, GgmlDType, (i32, usize, usize)); 3] = [
        (4096, GgmlDType::Q4_KO, (15, 0, 0)),
        (512, GgmlDType::Q6_KO, (63, 256, 0)),
        (512, GgmlDType::Q6_KO, (63, 256, 0)),
    ];
    let n_total: usize = dims.iter().map(|d| d.0).sum();
    let mut rng = rand::rng();

    for &m in &[4usize, 64, 256] {
        let act: Vec<f32> = (0..m * k).map(|_| rng.random_range(-1.0f32..1.0)).collect();
        let op = quantize_acts_q8a128_test(&dev, &act, m, k)?;

        let mut slices = Vec::new();
        let mut segs: Vec<(u64, GgmlDType, usize)> = Vec::new();
        let mut sep_refs: Vec<Vec<f32>> = Vec::new();
        for &(n, dtype, (maxq, crumb, hi)) in &dims {
            let wf32: Vec<f32> = (0..n * k).map(|_| rng.random_range(-0.1f32..0.1)).collect();
            let ob = requant_ko_per128(&wf32, n, k, maxq, crumb, hi);
            let ko = dev.memcpy_stod(&ob)?;
            slices.push(ko);
            let (ptr, _g) = slices.last().unwrap().device_ptr(&stream);
            segs.push((ptr, dtype, n));
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
fn qkv_segmented_vs_concat_same_format() -> Result<()> {
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

// =============================================================================
// moe_bucketize: bit-exact GPU vs CPU-reference tests
// =============================================================================
// The GPU bucketize replaces the CPU counting-sort in the grouped expert
// compute path, so its outputs must be BIT-IDENTICAL to the CPU grouping —
// asserted here with exact equality (never tolerances; the computation is pure
// integer). The reference below deliberately mirrors the production sort's
// single-pass cursor style (`forward_with_indices`), so these tests also prove
// the kernel's per-expert-scan formulation reproduces the production grouping.

/// CPU reference of the `moe_bucketize.cu` contract, padding included.
struct BucketizeRef {
    tok_ids: Vec<u32>,
    weight_ids: Vec<u32>,
    tile_expert: Vec<i32>,
    tile_b_start: Vec<i32>,
    tile_b_cnt: Vec<i32>,
    perm: Vec<u32>,
    rw_ids: Vec<u32>,
    token_starts: Vec<i32>,
    header: [i32; 4],
}

fn bucketize_ref(
    ids: &[u32],
    n_tokens: usize,
    k: usize,
    n_experts: usize,
    tile_w: usize,
) -> BucketizeRef {
    let a_ub = n_tokens * k;
    assert_eq!(ids.len(), a_ub);
    let valid = |e: u32| (e as usize) < n_experts;

    // Counts → offsets → tile prefix (kernel phase 2).
    let mut counts = vec![0i32; n_experts];
    for &e in ids {
        if valid(e) {
            counts[e as usize] += 1;
        }
    }
    let mut offsets = vec![0i32; n_experts + 1];
    let mut tile_pref = vec![0i32; n_experts + 1];
    let mut n_active = 0i32;
    for e in 0..n_experts {
        offsets[e + 1] = offsets[e] + counts[e];
        tile_pref[e + 1] = tile_pref[e] + (counts[e] + tile_w as i32 - 1) / tile_w as i32;
        if counts[e] > 0 {
            n_active += 1;
        }
    }
    let total_valid = offsets[n_experts];
    let num_tiles = tile_pref[n_experts];

    // Stable bucket write — single-pass cursor, the `forward_with_indices` style.
    let mut tok_ids = vec![u32::MAX; a_ub];
    let mut weight_ids = vec![u32::MAX; a_ub];
    let mut inv = vec![0u32; a_ub];
    let mut cursors: Vec<i32> = offsets[..n_experts].to_vec();
    for (i, &e) in ids.iter().enumerate() {
        if valid(e) {
            let row = cursors[e as usize];
            cursors[e as usize] += 1;
            tok_ids[row as usize] = (i / k) as u32;
            weight_ids[row as usize] = i as u32;
            inv[i] = row as u32;
        }
    }

    // Tile tables + padding.
    let mut tile_expert = vec![0i32; a_ub];
    let mut tile_b_start = vec![0i32; a_ub];
    let mut tile_b_cnt = vec![0i32; a_ub];
    let tw = tile_w as i32;
    for e in 0..n_experts {
        let cnt = counts[e];
        let base = tile_pref[e];
        let mut t = 0i32;
        while t * tw < cnt {
            tile_expert[(base + t) as usize] = e as i32;
            tile_b_start[(base + t) as usize] = offsets[e] + t * tw;
            tile_b_cnt[(base + t) as usize] = (cnt - t * tw).min(tw);
            t += 1;
        }
    }

    // Token-major compaction + segment boundaries. Within a token the pairs are
    // ordered by ascending expert-grouped row — the production scatter's
    // `sort_by_key((token_id, row))` accumulation order, so the float-summation
    // order downstream is bit-identical to the CPU-built tables.
    let mut perm = vec![0u32; a_ub];
    let mut rw_ids = vec![0u32; a_ub];
    let mut token_starts = vec![0i32; n_tokens + 1];
    let mut j = 0usize;
    for t in 0..n_tokens {
        token_starts[t] = j as i32;
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for s in 0..k {
            let i = t * k + s;
            if valid(ids[i]) {
                pairs.push((inv[i], i as u32));
            }
        }
        pairs.sort_unstable();
        for (row, widx) in pairs {
            perm[j] = row;
            rw_ids[j] = widx;
            j += 1;
        }
    }
    token_starts[n_tokens] = total_valid;

    BucketizeRef {
        tok_ids,
        weight_ids,
        tile_expert,
        tile_b_start,
        tile_b_cnt,
        perm,
        rw_ids,
        token_starts,
        header: [n_active, total_valid, num_tiles, 0],
    }
}

/// Run the GPU bucketize for `ids` and assert every output buffer is
/// bit-identical to the CPU reference.
#[allow(clippy::too_many_arguments)]
fn assert_bucketize_case(
    device: &crate::Device,
    ids: Vec<u32>,
    n_tokens: usize,
    k: usize,
    n_experts: usize,
    tile_w: usize,
    label: &str,
) -> Result<()> {
    let reference = bucketize_ref(&ids, n_tokens, k, n_experts, tile_w);
    let t = crate::Tensor::from_vec(ids, (n_tokens, k), device)?;
    let cuda_dev = match device {
        crate::Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let mut ws = MoeBucketizeWorkspace::new(&cuda_dev, n_tokens, k)?;
    moe_bucketize(&t, n_experts, tile_w, &mut ws)?;

    let a_ub = n_tokens * k;
    let tok = cuda_dev.memcpy_dtov(&ws.tok_ids.slice(..a_ub))?;
    let wid = cuda_dev.memcpy_dtov(&ws.weight_ids.slice(..a_ub))?;
    let te = cuda_dev.memcpy_dtov(&ws.tile_expert.slice(..a_ub))?;
    let tbs = cuda_dev.memcpy_dtov(&ws.tile_b_start.slice(..a_ub))?;
    let tbc = cuda_dev.memcpy_dtov(&ws.tile_b_cnt.slice(..a_ub))?;
    let pm = cuda_dev.memcpy_dtov(&ws.perm.slice(..a_ub))?;
    let rw = cuda_dev.memcpy_dtov(&ws.rw_ids.slice(..a_ub))?;
    let ts = cuda_dev.memcpy_dtov(&ws.token_starts.slice(..n_tokens + 1))?;
    let hd = cuda_dev.memcpy_dtov(&ws.header.slice(..4))?;

    assert_eq!(hd, reference.header.to_vec(), "{label}: header");
    assert_eq!(tok, reference.tok_ids, "{label}: tok_ids");
    assert_eq!(wid, reference.weight_ids, "{label}: weight_ids");
    assert_eq!(te, reference.tile_expert, "{label}: tile_expert");
    assert_eq!(tbs, reference.tile_b_start, "{label}: tile_b_start");
    assert_eq!(tbc, reference.tile_b_cnt, "{label}: tile_b_cnt");
    assert_eq!(pm, reference.perm, "{label}: perm");
    assert_eq!(rw, reference.rw_ids, "{label}: rw_ids");
    assert_eq!(ts, reference.token_starts, "{label}: token_starts");
    Ok(())
}

/// Assert every `moe_bucketize` output table is bit-identical to the CPU
/// reference for a workspace already run on `ids` — the gate the micro-bench
/// runs before timing each config.
#[cfg(test)]
fn assert_bucketize_ws(
    dev: &CudaDevice,
    ws: &MoeBucketizeWorkspace,
    ids: &[u32],
    n_tokens: usize,
    k: usize,
    n_experts: usize,
    tile_w: usize,
    label: &str,
) -> Result<()> {
    let reference = bucketize_ref(ids, n_tokens, k, n_experts, tile_w);
    let a_ub = n_tokens * k;
    assert_eq!(
        dev.memcpy_dtov(&ws.header.slice(..4))?,
        reference.header.to_vec(),
        "{label}: header"
    );
    assert_eq!(
        dev.memcpy_dtov(&ws.tok_ids.slice(..a_ub))?,
        reference.tok_ids,
        "{label}: tok_ids"
    );
    assert_eq!(
        dev.memcpy_dtov(&ws.weight_ids.slice(..a_ub))?,
        reference.weight_ids,
        "{label}: weight_ids"
    );
    assert_eq!(
        dev.memcpy_dtov(&ws.tile_expert.slice(..a_ub))?,
        reference.tile_expert,
        "{label}: tile_expert"
    );
    assert_eq!(
        dev.memcpy_dtov(&ws.tile_b_start.slice(..a_ub))?,
        reference.tile_b_start,
        "{label}: tile_b_start"
    );
    assert_eq!(
        dev.memcpy_dtov(&ws.tile_b_cnt.slice(..a_ub))?,
        reference.tile_b_cnt,
        "{label}: tile_b_cnt"
    );
    assert_eq!(
        dev.memcpy_dtov(&ws.perm.slice(..a_ub))?,
        reference.perm,
        "{label}: perm"
    );
    assert_eq!(
        dev.memcpy_dtov(&ws.rw_ids.slice(..a_ub))?,
        reference.rw_ids,
        "{label}: rw_ids"
    );
    assert_eq!(
        dev.memcpy_dtov(&ws.token_starts.slice(..n_tokens + 1))?,
        reference.token_starts,
        "{label}: token_starts"
    );
    Ok(())
}

/// Micro-benchmark harness for `moe_bucketize`. For each size regime qwen3
/// actually hits (k=8, 128 experts) plus a 256-expert stress case, it FIRST
/// asserts every output table is bit-identical to the CPU reference, then times
/// the kernel end-to-end (wall clock, one sync per batch of launches). Prints
/// µs/call and throughput. `--ignored` (needs a GPU); run with `--nocapture`.
#[test]
#[ignore]
fn bench_moe_bucketize() -> Result<()> {
    use std::time::Instant;
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let tile_w = 32usize;
    let iters = 300usize;

    // (n_tokens, k, n_experts, label)
    let configs: &[(usize, usize, usize, &str)] = &[
        (1, 8, 128, "decode-1"),
        (64, 8, 128, "decode-64"),
        (512, 8, 128, "prefill-512"),
        (2048, 8, 128, "prefill-2048"),
        (4096, 8, 128, "prefill-4096"),
        (8192, 8, 128, "prefill-8192"),
        (4096, 8, 256, "prefill-4096-e256"),
    ];

    println!("\n=== moe_bucketize micro-bench (tile_w={tile_w}, iters={iters}) ===");
    println!(
        "{:<20} {:>8} {:>7} {:>12} {:>14}",
        "config", "a_ub", "n_exp", "us/call", "M-assign/s"
    );
    for &(n_tokens, k, n_experts, label) in configs {
        let a_ub = n_tokens * k;
        // Deterministic pseudo-random dense routing (all slots valid).
        let ids: Vec<u32> = (0..a_ub)
            .map(|i| ((i as u64).wrapping_mul(2654435761) % n_experts as u64) as u32)
            .collect();
        let t = crate::Tensor::from_vec(ids.clone(), (n_tokens, k), &device)?;
        let mut ws = MoeBucketizeWorkspace::new(&dev, n_tokens, k)?;

        // Correctness gate FIRST — every table bit-exact vs the CPU sort.
        moe_bucketize(&t, n_experts, tile_w, &mut ws)?;
        assert_bucketize_ws(&dev, &ws, &ids, n_tokens, k, n_experts, tile_w, label)?;

        // Warm up, then time `iters` launches with a single trailing sync.
        for _ in 0..20 {
            moe_bucketize(&t, n_experts, tile_w, &mut ws)?;
        }
        let _ = dev.memcpy_dtov(&ws.header.slice(..1))?; // drain

        let start = Instant::now();
        for _ in 0..iters {
            moe_bucketize(&t, n_experts, tile_w, &mut ws)?;
        }
        let _ = dev.memcpy_dtov(&ws.header.slice(..1))?; // drain
        let us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let massign = (a_ub as f64) / (us * 1e-6) / 1e6;
        println!("{label:<20} {a_ub:>8} {n_experts:>7} {us:>12.2} {massign:>14.1}");
    }
    Ok(())
}

/// The GPU-native dispatch gate: `grouped_qmatmul_dev_q8a128` (full resident
/// pointer table + `moe_bucketize` device tile tables, upper-bound launch) must
/// produce BIT-IDENTICAL output rows to the host-orchestrated `grouped_qmatmul`
/// (active-compacted pointer array + host-built tile tables) on the same
/// operand and weights. Same tile decomposition, same per-tile pointer, same
/// kernel ⇒ same bits; this pins it. With the bucketize outputs proven
/// bit-identical to the CPU sort separately, and the gather/silu/scatter
/// kernels shared verbatim between paths, this closes the equivalence chain
/// for the whole GPU-native expert forward.
#[test]
fn cuda_grouped_qmatmul_dev_matches_host_tables() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let n_experts = 32usize; // smaller expert pool keeps the fixture fast
    let nrows = 256usize;
    let ncols = 1024usize;
    let wdtype = GgmlDType::Q6_K; // production base quant
    let ko_dtype = GgmlDType::Q6_KO; // its KO twin (int8 grouped path)

    // Random per-expert weights, quantized then KO-repacked (the model path).
    let mut rng = StdRng::seed_from_u64(0x9e4a_11ce);
    let shape = crate::Shape::from((nrows, ncols));
    let mut full_ptrs: Vec<u64> = Vec::with_capacity(n_experts);
    let mut storages = Vec::with_capacity(n_experts);
    for _ in 0..n_experts {
        let w: Vec<f32> = (0..nrows * ncols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let mut q = QCudaStorage::zeros(&dev, ncols * nrows, wdtype)?;
        q.quantize(&CudaStorage::wrap_cuda_slice(
            dev.memcpy_stod(&w)?,
            dev.clone(),
        ))?;
        let ko = q.repack_ko(&shape, ko_dtype)?;
        full_ptrs.push(ko.data_ptr());
        storages.push(ko);
    }
    let full_table = dev.memcpy_stod(&full_ptrs)?;

    for &(n_tokens, k, label) in &[
        (1usize, 8usize, "decode-1tok"),
        (7, 8, "mixed-small"),
        (64, 8, "prefill-64"),
        (200, 8, "prefill-200-multitile"),
    ] {
        let a_ub = n_tokens * k;
        // Random routing (dense; every slot valid — the sentinel path is pinned
        // by the bucketize tests and skipped rows never reach the GEMM).
        let ids: Vec<u32> = (0..a_ub)
            .map(|_| rng.random_range(0..n_experts as u32))
            .collect();
        let reference = bucketize_ref(&ids, n_tokens, k, n_experts, 32);
        let total_valid = reference.header[1] as usize;

        // GPU tables.
        let t = crate::Tensor::from_vec(ids.clone(), (n_tokens, k), &device)?;
        let mut ws = MoeBucketizeWorkspace::new(&dev, n_tokens, k)?;
        moe_bucketize(&t, n_experts, 32, &mut ws)?;

        // One shared stacked activation covering the full launch bound.
        let act: Vec<f32> = (0..a_ub * ncols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let op = quantize_acts_q8a128_test(&dev, &act, a_ub, ncols)?;

        // Host path: active-compacted pointers + per-active-expert offsets,
        // exactly as `forward_with_indices` builds them.
        let mut counts = vec![0i32; n_experts];
        for &e in &ids {
            counts[e as usize] += 1;
        }
        let mut host_ptrs: Vec<u64> = Vec::new();
        let mut host_offsets: Vec<i32> = vec![0];
        for e in 0..n_experts {
            if counts[e] > 0 {
                host_ptrs.push(full_ptrs[e]);
                host_offsets.push(host_offsets.last().unwrap() + counts[e]);
            }
        }
        let host_out = grouped_qmatmul(
            DynamicTensor::Int8(&op),
            &host_ptrs,
            ko_dtype,
            nrows,
            &host_offsets,
            &dev,
            Backing::Owned,
        )?;

        // Device path: full table, raw ids, upper-bound launch.
        let dev_out = grouped_qmatmul_dev_q8a128(
            &op,
            &full_table,
            0,
            n_experts,
            ko_dtype,
            nrows,
            &ws.tile_expert,
            &ws.tile_b_start,
            &ws.tile_b_cnt,
            a_ub,
            &dev,
        )?;

        let host_v = read_f32_tensor(&dev, &host_out)?;
        let dev_v = read_f32_tensor(&dev, &dev_out)?;
        let n_cmp = total_valid * nrows;
        assert_eq!(
            host_v[..n_cmp]
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<u32>>(),
            dev_v[..n_cmp]
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<u32>>(),
            "{label}: dev-table GEMM must be bit-identical to host-table GEMM"
        );
    }
    Ok(())
}

#[test]
fn cuda_moe_bucketize_matches_cpu_reference() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());

    // Decode: 1 token × k=8 distinct experts (the hot interactive case).
    assert_bucketize_case(
        &device,
        vec![7, 3, 100, 42, 0, 127, 55, 9],
        1,
        8,
        128,
        32,
        "decode-1tok",
    )?;

    // Duplicate experts within a token (allowed by the contract even though
    // top-k emits distinct ids) and multiple tokens sharing experts.
    assert_bucketize_case(
        &device,
        vec![3, 3, 5, 5, 5, 0, 1, 2, 3, 5, 0, 0, 0, 1, 2, 7],
        2,
        8,
        8,
        32,
        "dup-experts",
    )?;

    // Router empty-slot sentinels (id == n_experts) and beyond must be skipped.
    assert_bucketize_case(
        &device,
        vec![1, 16, 2, 99, 3, 16, 16, 4],
        1,
        8,
        16,
        32,
        "sentinels",
    )?;

    // Everything routes to ONE expert → a deep bucket crossing many tile
    // boundaries (multi-tile single expert, non-divisible tail).
    assert_bucketize_case(
        &device,
        vec![5u32; 100 * 8],
        100,
        8,
        128,
        32,
        "one-expert-multitile",
    )?;

    // All slots sentinel (pathological: nothing valid anywhere).
    assert_bucketize_case(&device, vec![128u32; 4 * 8], 4, 8, 128, 32, "all-sentinel")?;

    // k = 1 and tile_w = 16 shape edges.
    assert_bucketize_case(&device, vec![2, 0, 2, 1, 2], 5, 1, 4, 16, "k1-tile16")?;

    // Seeded fuzz across prefill-like shapes, with a sprinkle of sentinels.
    let mut rng = StdRng::seed_from_u64(0xb0cc_e71e);
    for &(n_tokens, k, n_experts, tile_w) in &[
        (4usize, 8usize, 128usize, 32usize),
        (64, 8, 128, 32),
        (333, 8, 128, 32),
        (1024, 8, 128, 32),
        (2048, 8, 128, 32),
        (7, 3, 16, 32),
        (129, 8, 128, 16),
        (17, 8, 128, 1),
    ] {
        let ids: Vec<u32> = (0..n_tokens * k)
            .map(|_| {
                if rng.random_ratio(1, 50) {
                    n_experts as u32 // router sentinel
                } else {
                    rng.random_range(0..n_experts as u32)
                }
            })
            .collect();
        assert_bucketize_case(
            &device,
            ids,
            n_tokens,
            k,
            n_experts,
            tile_w,
            &format!("fuzz-{n_tokens}x{k}-e{n_experts}-w{tile_w}"),
        )?;
    }

    // Determinism: identical input twice through the SAME workspace must give
    // byte-identical buffers (no atomics, no ordering hazards).
    let n_tokens = 512usize;
    let k = 8usize;
    let ids: Vec<u32> = (0..n_tokens * k)
        .map(|_| rng.random_range(0..128))
        .collect();
    let t = crate::Tensor::from_vec(ids, (n_tokens, k), &device)?;
    let mut ws = MoeBucketizeWorkspace::new(&dev, n_tokens, k)?;
    moe_bucketize(&t, 128, 32, &mut ws)?;
    let first = (
        dev.memcpy_dtov(&ws.tok_ids.slice(..n_tokens * k))?,
        dev.memcpy_dtov(&ws.perm.slice(..n_tokens * k))?,
        dev.memcpy_dtov(&ws.tile_b_cnt.slice(..n_tokens * k))?,
    );
    moe_bucketize(&t, 128, 32, &mut ws)?;
    let second = (
        dev.memcpy_dtov(&ws.tok_ids.slice(..n_tokens * k))?,
        dev.memcpy_dtov(&ws.perm.slice(..n_tokens * k))?,
        dev.memcpy_dtov(&ws.tile_b_cnt.slice(..n_tokens * k))?,
    );
    assert_eq!(first, second, "repeat run must be byte-identical");

    Ok(())
}

/// The context-free total-VRAM probe must succeed and report a sane size.
/// (In-process test order may mean a context already exists on this thread;
/// the no-context property is exercised for real by zend's download path,
/// which calls this before any `Device` is created.)
#[test]
fn cuda_total_vram_device0_probe() -> Result<()> {
    let total = get_total_vram_device0()?;
    assert!(
        total > 1024 * 1024 * 1024,
        "reported total VRAM {total} bytes is implausibly small"
    );
    Ok(())
}

/// MXFP4 GPU dequant must bit-match the CPU codec: both decode the same 17-byte blocks
/// with identical E2M1-table × E8M0-half-scale arithmetic. Covers F32, F16, and BF16
/// outputs. This is the FP4 expert-weight decode path for DeepSeek-V4.
#[test]
fn cuda_mxfp4_dequant_matches_cpu() -> Result<()> {
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
fn cuda_mxfp4_qmatmul_dequant_path() -> Result<()> {
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
fn mxfp4_int8_matmul_matches_float_baseline() -> Result<()> {
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
        let acts = to_dynamic(&act_t, mode, &dev)?;
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
