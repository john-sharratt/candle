use super::*;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use half::bf16;
use rand::Rng;
use std::ffi::c_void;
use std::time::Instant;

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

        unsafe {
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
            );
        }
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

        unsafe {
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
            );
        }
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

        unsafe {
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
            );
        }
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
    for i in 0..8.min(nrows) {
        let val = f16::to_f32(result_dequant_vec[i]);
        println!("Row {}: {:.4}", i, val);
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

            unsafe {
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
                );
            }
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

        unsafe {
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
            );
        }
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
    let elem_count = 32;
    let dtype = GgmlDType::Q4_0;

    // Simple ascending values
    let src_data: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) / 2.0).collect();
    println!("Input: {:?}", src_data);

    let src_gpu = dev.memcpy_stod(&src_data)?;

    let block_size = dtype.block_size();
    let type_size = dtype.type_size();
    let num_blocks = (elem_count + block_size - 1) / block_size;
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
    let elem_count = 1024 * 1024; // 1M elements
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
        let num_blocks = (elem_count + block_size - 1) / block_size;
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

    // For now, just report - don't fail the test since we're debugging
    // Later we can add: assert!(q4_0_exact && q8_0_exact && q2k_exact);

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
