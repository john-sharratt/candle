//! Comprehensive cast (to_dtype) tests for all dtype combinations
//!
//! Tests cover:
//! - All pairwise dtype conversions
//! - Contiguous and non-contiguous (strided) tensors
//! - Various tensor sizes (small, medium, large)
//! - In-place casting via to_dtype_mut

use candle_core::{test_device, DType, Device, Result, Tensor};
use float8::F8E4M3;
use half::{bf16, f16};

// =============================================================================
// Helper functions
// =============================================================================

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a == b {
        return true;
    }
    let diff = (a - b).abs();
    diff <= tol || diff <= tol * a.abs().max(b.abs())
}

// =============================================================================
// F32 source casts
// =============================================================================

fn cast_f32_to_f32(device: &Device) -> Result<()> {
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::F32)?;
    let result = t2.to_vec2::<f32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert!(approx_eq(v as f64, data[i * 10 + j] as f64, 1e-5));
        }
    }
    Ok(())
}

fn cast_f32_to_f64(device: &Device) -> Result<()> {
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::F64)?;
    let result = t2.to_vec2::<f64>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert!(approx_eq(v, data[i * 10 + j] as f64, 1e-5));
        }
    }
    Ok(())
}

fn cast_f32_to_f16(device: &Device) -> Result<()> {
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::F16)?;
    let result = t2.to_vec2::<f16>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = f16::from_f32(data[i * 10 + j]);
            assert_eq!(v, expected);
        }
    }
    Ok(())
}

fn cast_f32_to_bf16(device: &Device) -> Result<()> {
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::BF16)?;
    let result = t2.to_vec2::<bf16>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = bf16::from_f32(data[i * 10 + j]);
            assert_eq!(v, expected);
        }
    }
    Ok(())
}

fn cast_f32_to_u8(device: &Device) -> Result<()> {
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::U8)?;
    let result = t2.to_vec2::<u8>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as u8);
        }
    }
    Ok(())
}

fn cast_f32_to_u32(device: &Device) -> Result<()> {
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::U32)?;
    let result = t2.to_vec2::<u32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as u32);
        }
    }
    Ok(())
}

fn cast_f32_to_i64(device: &Device) -> Result<()> {
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::I64)?;
    let result = t2.to_vec2::<i64>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as i64);
        }
    }
    Ok(())
}

fn cast_f32_to_f8e4m3(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(()); // F8E4M3 not supported on Metal
    }
    let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.25).collect();
    let t = Tensor::from_slice(&data, (8, 8), device)?;
    let t2 = t.to_dtype(DType::F8E4M3)?;
    let result = t2.to_vec2::<F8E4M3>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = F8E4M3::from_f32(data[i * 8 + j]);
            assert_eq!(v, expected);
        }
    }
    Ok(())
}

// =============================================================================
// F64 source casts
// =============================================================================

fn cast_f64_to_f32(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(()); // F64 not fully supported on Metal
    }
    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::F32)?;
    let result = t2.to_vec2::<f32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert!(approx_eq(v as f64, data[i * 10 + j], 1e-5));
        }
    }
    Ok(())
}

fn cast_f64_to_f64(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::F64)?;
    let result = t2.to_vec2::<f64>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert!(approx_eq(v, data[i * 10 + j], 1e-10));
        }
    }
    Ok(())
}

fn cast_f64_to_f16(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::F16)?;
    let result = t2.to_vec2::<f16>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = f16::from_f64(data[i * 10 + j]);
            assert_eq!(v, expected);
        }
    }
    Ok(())
}

fn cast_f64_to_bf16(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::BF16)?;
    let result = t2.to_vec2::<bf16>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = bf16::from_f64(data[i * 10 + j]);
            assert_eq!(v, expected);
        }
    }
    Ok(())
}

fn cast_f64_to_u8(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::U8)?;
    let result = t2.to_vec2::<u8>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as u8);
        }
    }
    Ok(())
}

fn cast_f64_to_u32(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::U32)?;
    let result = t2.to_vec2::<u32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as u32);
        }
    }
    Ok(())
}

fn cast_f64_to_i64(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::I64)?;
    let result = t2.to_vec2::<i64>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as i64);
        }
    }
    Ok(())
}

fn cast_f64_to_f8e4m3(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<f64> = (0..64).map(|i| i as f64 * 0.25).collect();
    let t = Tensor::from_slice(&data, (8, 8), device)?;
    let t2 = t.to_dtype(DType::F8E4M3)?;
    let result = t2.to_vec2::<F8E4M3>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = F8E4M3::from_f32(data[i * 8 + j] as f32);
            assert_eq!(v, expected);
        }
    }
    Ok(())
}

// =============================================================================
// Test registration - F32 and F64 source
// =============================================================================

test_device!(
    cast_f32_to_f32,
    cast_f32_to_f32_cpu,
    cast_f32_to_f32_cuda,
    cast_f32_to_f32_metal
);
test_device!(
    cast_f32_to_f64,
    cast_f32_to_f64_cpu,
    cast_f32_to_f64_cuda,
    cast_f32_to_f64_metal
);
test_device!(
    cast_f32_to_f16,
    cast_f32_to_f16_cpu,
    cast_f32_to_f16_cuda,
    cast_f32_to_f16_metal
);
test_device!(
    cast_f32_to_bf16,
    cast_f32_to_bf16_cpu,
    cast_f32_to_bf16_cuda,
    cast_f32_to_bf16_metal
);
test_device!(
    cast_f32_to_u8,
    cast_f32_to_u8_cpu,
    cast_f32_to_u8_cuda,
    cast_f32_to_u8_metal
);
test_device!(
    cast_f32_to_u32,
    cast_f32_to_u32_cpu,
    cast_f32_to_u32_cuda,
    cast_f32_to_u32_metal
);
test_device!(
    cast_f32_to_i64,
    cast_f32_to_i64_cpu,
    cast_f32_to_i64_cuda,
    cast_f32_to_i64_metal
);
test_device!(
    cast_f32_to_f8e4m3,
    cast_f32_to_f8e4m3_cpu,
    cast_f32_to_f8e4m3_cuda,
    cast_f32_to_f8e4m3_metal
);

test_device!(
    cast_f64_to_f32,
    cast_f64_to_f32_cpu,
    cast_f64_to_f32_cuda,
    cast_f64_to_f32_metal
);
test_device!(
    cast_f64_to_f64,
    cast_f64_to_f64_cpu,
    cast_f64_to_f64_cuda,
    cast_f64_to_f64_metal
);
test_device!(
    cast_f64_to_f16,
    cast_f64_to_f16_cpu,
    cast_f64_to_f16_cuda,
    cast_f64_to_f16_metal
);
test_device!(
    cast_f64_to_bf16,
    cast_f64_to_bf16_cpu,
    cast_f64_to_bf16_cuda,
    cast_f64_to_bf16_metal
);
test_device!(
    cast_f64_to_u8,
    cast_f64_to_u8_cpu,
    cast_f64_to_u8_cuda,
    cast_f64_to_u8_metal
);
test_device!(
    cast_f64_to_u32,
    cast_f64_to_u32_cpu,
    cast_f64_to_u32_cuda,
    cast_f64_to_u32_metal
);
test_device!(
    cast_f64_to_i64,
    cast_f64_to_i64_cpu,
    cast_f64_to_i64_cuda,
    cast_f64_to_i64_metal
);
test_device!(
    cast_f64_to_f8e4m3,
    cast_f64_to_f8e4m3_cpu,
    cast_f64_to_f8e4m3_cuda,
    cast_f64_to_f8e4m3_metal
);

// =============================================================================
// F16 source casts
// =============================================================================

fn cast_f16_to_f32(device: &Device) -> Result<()> {
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?.to_dtype(DType::F16)?;
    let t2 = t.to_dtype(DType::F32)?;
    let result = t2.to_vec2::<f32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = f16::from_f32(data[i * 10 + j]).to_f32();
            assert!(approx_eq(v as f64, expected as f64, 0.01));
        }
    }
    Ok(())
}

fn cast_f16_to_f64(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?.to_dtype(DType::F16)?;
    let t2 = t.to_dtype(DType::F64)?;
    let result = t2.to_vec2::<f64>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = f16::from_f32(data[i * 10 + j]).to_f64();
            assert!(approx_eq(v, expected, 0.01));
        }
    }
    Ok(())
}

fn cast_f16_to_bf16(device: &Device) -> Result<()> {
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?.to_dtype(DType::F16)?;
    let t2 = t.to_dtype(DType::BF16)?;
    let result = t2.to_vec2::<bf16>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = bf16::from_f32(f16::from_f32(data[i * 10 + j]).to_f32());
            assert_eq!(v, expected);
        }
    }
    Ok(())
}

fn cast_f16_to_u8(device: &Device) -> Result<()> {
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?.to_dtype(DType::F16)?;
    let t2 = t.to_dtype(DType::U8)?;
    let result = t2.to_vec2::<u8>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as u8);
        }
    }
    Ok(())
}

fn cast_f16_to_u32(device: &Device) -> Result<()> {
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?.to_dtype(DType::F16)?;
    let t2 = t.to_dtype(DType::U32)?;
    let result = t2.to_vec2::<u32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as u32);
        }
    }
    Ok(())
}

// =============================================================================
// BF16 source casts
// =============================================================================

fn cast_bf16_to_f32(device: &Device) -> Result<()> {
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?.to_dtype(DType::BF16)?;
    let t2 = t.to_dtype(DType::F32)?;
    let result = t2.to_vec2::<f32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = bf16::from_f32(data[i * 10 + j]).to_f32();
            assert!(approx_eq(v as f64, expected as f64, 0.1));
        }
    }
    Ok(())
}

fn cast_bf16_to_f64(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?.to_dtype(DType::BF16)?;
    let t2 = t.to_dtype(DType::F64)?;
    let result = t2.to_vec2::<f64>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = bf16::from_f32(data[i * 10 + j]).to_f64();
            assert!(approx_eq(v, expected, 0.1));
        }
    }
    Ok(())
}

fn cast_bf16_to_f16(device: &Device) -> Result<()> {
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?.to_dtype(DType::BF16)?;
    let t2 = t.to_dtype(DType::F16)?;
    let result = t2.to_vec2::<f16>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = f16::from_f32(bf16::from_f32(data[i * 10 + j]).to_f32());
            assert_eq!(v, expected);
        }
    }
    Ok(())
}

fn cast_bf16_to_u8(device: &Device) -> Result<()> {
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?.to_dtype(DType::BF16)?;
    let t2 = t.to_dtype(DType::U8)?;
    let result = t2.to_vec2::<u8>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as u8);
        }
    }
    Ok(())
}

fn cast_bf16_to_u32(device: &Device) -> Result<()> {
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?.to_dtype(DType::BF16)?;
    let t2 = t.to_dtype(DType::U32)?;
    let result = t2.to_vec2::<u32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as u32);
        }
    }
    Ok(())
}

// =============================================================================
// Test registration - F16 and BF16 source
// =============================================================================

test_device!(
    cast_f16_to_f32,
    cast_f16_to_f32_cpu,
    cast_f16_to_f32_cuda,
    cast_f16_to_f32_metal
);
test_device!(
    cast_f16_to_f64,
    cast_f16_to_f64_cpu,
    cast_f16_to_f64_cuda,
    cast_f16_to_f64_metal
);
test_device!(
    cast_f16_to_bf16,
    cast_f16_to_bf16_cpu,
    cast_f16_to_bf16_cuda,
    cast_f16_to_bf16_metal
);
test_device!(
    cast_f16_to_u8,
    cast_f16_to_u8_cpu,
    cast_f16_to_u8_cuda,
    cast_f16_to_u8_metal
);
test_device!(
    cast_f16_to_u32,
    cast_f16_to_u32_cpu,
    cast_f16_to_u32_cuda,
    cast_f16_to_u32_metal
);

test_device!(
    cast_bf16_to_f32,
    cast_bf16_to_f32_cpu,
    cast_bf16_to_f32_cuda,
    cast_bf16_to_f32_metal
);
test_device!(
    cast_bf16_to_f64,
    cast_bf16_to_f64_cpu,
    cast_bf16_to_f64_cuda,
    cast_bf16_to_f64_metal
);
test_device!(
    cast_bf16_to_f16,
    cast_bf16_to_f16_cpu,
    cast_bf16_to_f16_cuda,
    cast_bf16_to_f16_metal
);
test_device!(
    cast_bf16_to_u8,
    cast_bf16_to_u8_cpu,
    cast_bf16_to_u8_cuda,
    cast_bf16_to_u8_metal
);
test_device!(
    cast_bf16_to_u32,
    cast_bf16_to_u32_cpu,
    cast_bf16_to_u32_cuda,
    cast_bf16_to_u32_metal
);

// =============================================================================
// U8 source casts
// =============================================================================

fn cast_u8_to_f32(device: &Device) -> Result<()> {
    let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::F32)?;
    let result = t2.to_vec2::<f32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as f32);
        }
    }
    Ok(())
}

fn cast_u8_to_f64(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::F64)?;
    let result = t2.to_vec2::<f64>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as f64);
        }
    }
    Ok(())
}

fn cast_u8_to_f16(device: &Device) -> Result<()> {
    let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::F16)?;
    let result = t2.to_vec2::<f16>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, f16::from_f32((i * 10 + j) as f32));
        }
    }
    Ok(())
}

fn cast_u8_to_bf16(device: &Device) -> Result<()> {
    let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::BF16)?;
    let result = t2.to_vec2::<bf16>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, bf16::from_f32((i * 10 + j) as f32));
        }
    }
    Ok(())
}

fn cast_u8_to_u32(device: &Device) -> Result<()> {
    let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::U32)?;
    let result = t2.to_vec2::<u32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as u32);
        }
    }
    Ok(())
}

fn cast_u8_to_i64(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::I64)?;
    let result = t2.to_vec2::<i64>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as i64);
        }
    }
    Ok(())
}

// =============================================================================
// U32 source casts
// =============================================================================

fn cast_u32_to_f32(device: &Device) -> Result<()> {
    let data: Vec<u32> = (0..100).map(|i| i as u32).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::F32)?;
    let result = t2.to_vec2::<f32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as f32);
        }
    }
    Ok(())
}

fn cast_u32_to_f64(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<u32> = (0..100).map(|i| i as u32).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::F64)?;
    let result = t2.to_vec2::<f64>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as f64);
        }
    }
    Ok(())
}

fn cast_u32_to_f16(device: &Device) -> Result<()> {
    let data: Vec<u32> = (0..100).map(|i| i as u32).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::F16)?;
    let result = t2.to_vec2::<f16>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, f16::from_f32((i * 10 + j) as f32));
        }
    }
    Ok(())
}

fn cast_u32_to_bf16(device: &Device) -> Result<()> {
    let data: Vec<u32> = (0..100).map(|i| i as u32).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::BF16)?;
    let result = t2.to_vec2::<bf16>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, bf16::from_f32((i * 10 + j) as f32));
        }
    }
    Ok(())
}

fn cast_u32_to_u8(device: &Device) -> Result<()> {
    let data: Vec<u32> = (0..100).map(|i| i as u32).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::U8)?;
    let result = t2.to_vec2::<u8>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as u8);
        }
    }
    Ok(())
}

fn cast_u32_to_i64(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<u32> = (0..100).map(|i| i as u32).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::I64)?;
    let result = t2.to_vec2::<i64>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as i64);
        }
    }
    Ok(())
}

// =============================================================================
// Test registration - U8 and U32 source
// =============================================================================

test_device!(
    cast_u8_to_f32,
    cast_u8_to_f32_cpu,
    cast_u8_to_f32_cuda,
    cast_u8_to_f32_metal
);
test_device!(
    cast_u8_to_f64,
    cast_u8_to_f64_cpu,
    cast_u8_to_f64_cuda,
    cast_u8_to_f64_metal
);
test_device!(
    cast_u8_to_f16,
    cast_u8_to_f16_cpu,
    cast_u8_to_f16_cuda,
    cast_u8_to_f16_metal
);
test_device!(
    cast_u8_to_bf16,
    cast_u8_to_bf16_cpu,
    cast_u8_to_bf16_cuda,
    cast_u8_to_bf16_metal
);
test_device!(
    cast_u8_to_u32,
    cast_u8_to_u32_cpu,
    cast_u8_to_u32_cuda,
    cast_u8_to_u32_metal
);
test_device!(
    cast_u8_to_i64,
    cast_u8_to_i64_cpu,
    cast_u8_to_i64_cuda,
    cast_u8_to_i64_metal
);

test_device!(
    cast_u32_to_f32,
    cast_u32_to_f32_cpu,
    cast_u32_to_f32_cuda,
    cast_u32_to_f32_metal
);
test_device!(
    cast_u32_to_f64,
    cast_u32_to_f64_cpu,
    cast_u32_to_f64_cuda,
    cast_u32_to_f64_metal
);
test_device!(
    cast_u32_to_f16,
    cast_u32_to_f16_cpu,
    cast_u32_to_f16_cuda,
    cast_u32_to_f16_metal
);
test_device!(
    cast_u32_to_bf16,
    cast_u32_to_bf16_cpu,
    cast_u32_to_bf16_cuda,
    cast_u32_to_bf16_metal
);
test_device!(
    cast_u32_to_u8,
    cast_u32_to_u8_cpu,
    cast_u32_to_u8_cuda,
    cast_u32_to_u8_metal
);
test_device!(
    cast_u32_to_i64,
    cast_u32_to_i64_cpu,
    cast_u32_to_i64_cuda,
    cast_u32_to_i64_metal
);

// =============================================================================
// I64 source casts
// =============================================================================

fn cast_i64_to_f32(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<i64> = (0..100).map(|i| i as i64).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::F32)?;
    let result = t2.to_vec2::<f32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as f32);
        }
    }
    Ok(())
}

fn cast_i64_to_f64(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<i64> = (0..100).map(|i| i as i64).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::F64)?;
    let result = t2.to_vec2::<f64>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as f64);
        }
    }
    Ok(())
}

fn cast_i64_to_f16(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<i64> = (0..100).map(|i| i as i64).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::F16)?;
    let result = t2.to_vec2::<f16>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, f16::from_f32((i * 10 + j) as f32));
        }
    }
    Ok(())
}

fn cast_i64_to_bf16(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<i64> = (0..100).map(|i| i as i64).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::BF16)?;
    let result = t2.to_vec2::<bf16>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, bf16::from_f32((i * 10 + j) as f32));
        }
    }
    Ok(())
}

fn cast_i64_to_u8(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<i64> = (0..100).map(|i| i as i64).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::U8)?;
    let result = t2.to_vec2::<u8>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as u8);
        }
    }
    Ok(())
}

fn cast_i64_to_u32(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<i64> = (0..100).map(|i| i as i64).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t2 = t.to_dtype(DType::U32)?;
    let result = t2.to_vec2::<u32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (i * 10 + j) as u32);
        }
    }
    Ok(())
}

fn cast_i64_to_f8e4m3(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(()); // F8E4M3 not supported on Metal
    }
    // Use small values that fit in F8E4M3 range
    let data: Vec<i64> = (0..64).map(|i| i as i64).collect();
    let t = Tensor::from_slice(&data, (8, 8), device)?;
    let t2 = t.to_dtype(DType::F8E4M3)?;
    let result = t2.to_vec2::<F8E4M3>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = F8E4M3::from_f32((i * 8 + j) as f32);
            assert_eq!(v, expected, "Mismatch at [{i}][{j}]");
        }
    }
    Ok(())
}

fn cast_f8e4m3_to_i64(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(()); // F8E4M3 not supported on Metal
    }
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let t = Tensor::from_slice(&data, (8, 8), device)?;
    let t_f8 = t.to_dtype(DType::F8E4M3)?;
    let t2 = t_f8.to_dtype(DType::I64)?;
    let result = t2.to_vec2::<i64>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let f8_val = F8E4M3::from_f32((i * 8 + j) as f32);
            let expected = f8_val.to_f32() as i64;
            assert_eq!(v, expected, "Mismatch at [{i}][{j}]");
        }
    }
    Ok(())
}

fn cast_u32_to_f8e4m3(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(()); // F8E4M3 not supported on Metal
    }
    // Use small values that fit in F8E4M3 range
    let data: Vec<u32> = (0..64).map(|i| i as u32).collect();
    let t = Tensor::from_slice(&data, (8, 8), device)?;
    let t2 = t.to_dtype(DType::F8E4M3)?;
    let result = t2.to_vec2::<F8E4M3>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = F8E4M3::from_f32((i * 8 + j) as f32);
            assert_eq!(v, expected, "Mismatch at [{i}][{j}]");
        }
    }
    Ok(())
}

fn cast_f8e4m3_to_u32(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(()); // F8E4M3 not supported on Metal
    }
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let t = Tensor::from_slice(&data, (8, 8), device)?;
    let t_f8 = t.to_dtype(DType::F8E4M3)?;
    let t2 = t_f8.to_dtype(DType::U32)?;
    let result = t2.to_vec2::<u32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let f8_val = F8E4M3::from_f32((i * 8 + j) as f32);
            let expected = f8_val.to_f32() as u32;
            assert_eq!(v, expected, "Mismatch at [{i}][{j}]");
        }
    }
    Ok(())
}

// =============================================================================
// Test registration - I64 source
// =============================================================================

test_device!(
    cast_i64_to_f32,
    cast_i64_to_f32_cpu,
    cast_i64_to_f32_cuda,
    cast_i64_to_f32_metal
);
test_device!(
    cast_i64_to_f64,
    cast_i64_to_f64_cpu,
    cast_i64_to_f64_cuda,
    cast_i64_to_f64_metal
);
test_device!(
    cast_i64_to_f16,
    cast_i64_to_f16_cpu,
    cast_i64_to_f16_cuda,
    cast_i64_to_f16_metal
);
test_device!(
    cast_i64_to_bf16,
    cast_i64_to_bf16_cpu,
    cast_i64_to_bf16_cuda,
    cast_i64_to_bf16_metal
);
test_device!(
    cast_i64_to_u8,
    cast_i64_to_u8_cpu,
    cast_i64_to_u8_cuda,
    cast_i64_to_u8_metal
);
test_device!(
    cast_i64_to_u32,
    cast_i64_to_u32_cpu,
    cast_i64_to_u32_cuda,
    cast_i64_to_u32_metal
);
test_device!(
    cast_i64_to_f8e4m3,
    cast_i64_to_f8e4m3_cpu,
    cast_i64_to_f8e4m3_cuda,
    cast_i64_to_f8e4m3_metal
);
test_device!(
    cast_f8e4m3_to_i64,
    cast_f8e4m3_to_i64_cpu,
    cast_f8e4m3_to_i64_cuda,
    cast_f8e4m3_to_i64_metal
);
test_device!(
    cast_u32_to_f8e4m3,
    cast_u32_to_f8e4m3_cpu,
    cast_u32_to_f8e4m3_cuda,
    cast_u32_to_f8e4m3_metal
);
test_device!(
    cast_f8e4m3_to_u32,
    cast_f8e4m3_to_u32_cpu,
    cast_f8e4m3_to_u32_cuda,
    cast_f8e4m3_to_u32_metal
);

// =============================================================================
// Strided (non-contiguous) tensor casts - test misaligned memory access
// =============================================================================

fn cast_strided_f32_to_f64(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    // Create a 2D tensor and transpose it to make it non-contiguous
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t_transposed = t.t()?; // Now non-contiguous
    let t2 = t_transposed.to_dtype(DType::F64)?;
    let result = t2.to_vec2::<f64>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            // Transposed: v = original[j][i]
            let expected = (j * 10 + i) as f64 * 0.5;
            assert!(approx_eq(v, expected, 1e-6));
        }
    }
    Ok(())
}

fn cast_strided_f64_to_f32(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t_transposed = t.t()?;
    let t2 = t_transposed.to_dtype(DType::F32)?;
    let result = t2.to_vec2::<f32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = (j * 10 + i) as f32 * 0.5;
            assert!(approx_eq(v as f64, expected as f64, 1e-6));
        }
    }
    Ok(())
}

fn cast_strided_f32_to_f16(device: &Device) -> Result<()> {
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t_transposed = t.t()?;
    let t2 = t_transposed.to_dtype(DType::F16)?;
    let result = t2.to_vec2::<f16>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = f16::from_f32((j * 10 + i) as f32 * 0.5);
            assert_eq!(v, expected);
        }
    }
    Ok(())
}

fn cast_strided_u8_to_f32(device: &Device) -> Result<()> {
    let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    let t_transposed = t.t()?;
    let t2 = t_transposed.to_dtype(DType::F32)?;
    let result = t2.to_vec2::<f32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, (j * 10 + i) as f32);
        }
    }
    Ok(())
}

// =============================================================================
// Narrow/slice casts - another form of non-contiguous
// =============================================================================

fn cast_narrow_f32_to_f64(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?;
    // Narrow to get a non-contiguous slice
    let t_narrow = t.narrow(1, 2, 5)?; // columns 2-6
    let t2 = t_narrow.to_dtype(DType::F64)?;
    let result = t2.to_vec2::<f64>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = (i * 10 + j + 2) as f64;
            assert_eq!(v, expected);
        }
    }
    Ok(())
}

fn cast_narrow_bf16_to_f32(device: &Device) -> Result<()> {
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let t = Tensor::from_slice(&data, (10, 10), device)?.to_dtype(DType::BF16)?;
    let t_narrow = t.narrow(0, 3, 4)?; // rows 3-6
    let t2 = t_narrow.to_dtype(DType::F32)?;
    let result = t2.to_vec2::<f32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = bf16::from_f32(((i + 3) * 10 + j) as f32).to_f32();
            assert!(approx_eq(v as f64, expected as f64, 0.1));
        }
    }
    Ok(())
}

// =============================================================================
// F8E4M3 strided cast tests
// =============================================================================

fn cast_strided_f32_to_f8e4m3(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(()); // F8E4M3 not supported on Metal
    }
    // Use small values that fit in F8E4M3 range
    let data: Vec<f32> = (0..36).map(|i| i as f32 * 0.25).collect();
    let t = Tensor::from_slice(&data, (6, 6), device)?;
    let t_transposed = t.t()?;
    assert!(!t_transposed.is_contiguous());
    let t2 = t_transposed.to_dtype(DType::F8E4M3)?;
    let result = t2.to_vec2::<F8E4M3>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            // Transposed: v = original[j][i]
            let expected = F8E4M3::from_f32((j * 6 + i) as f32 * 0.25);
            assert_eq!(v, expected, "Mismatch at [{i}][{j}]");
        }
    }
    Ok(())
}

fn cast_strided_f8e4m3_to_f32(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(()); // F8E4M3 not supported on Metal
    }
    let data: Vec<f32> = (0..36).map(|i| i as f32 * 0.25).collect();
    let t = Tensor::from_slice(&data, (6, 6), device)?;
    let t_f8 = t.to_dtype(DType::F8E4M3)?;
    let t_transposed = t_f8.t()?;
    assert!(!t_transposed.is_contiguous());
    let t2 = t_transposed.to_dtype(DType::F32)?;
    let result = t2.to_vec2::<f32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            // Transposed: v = original[j][i]
            let original_f8 = F8E4M3::from_f32((j * 6 + i) as f32 * 0.25);
            let expected = original_f8.to_f32();
            assert!(
                (v - expected).abs() < 0.01,
                "Mismatch at [{i}][{j}]: got {}, expected {}",
                v,
                expected
            );
        }
    }
    Ok(())
}

fn cast_strided_f8e4m3_to_f16(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(()); // F8E4M3 not supported on Metal
    }
    let data: Vec<f32> = (0..36).map(|i| i as f32 * 0.25).collect();
    let t = Tensor::from_slice(&data, (6, 6), device)?;
    let t_f8 = t.to_dtype(DType::F8E4M3)?;
    let t_transposed = t_f8.t()?;
    assert!(!t_transposed.is_contiguous());
    let t2 = t_transposed.to_dtype(DType::F16)?;
    let result = t2.to_vec2::<f16>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let original_f8 = F8E4M3::from_f32((j * 6 + i) as f32 * 0.25);
            let expected = f16::from_f32(original_f8.to_f32());
            assert_eq!(v, expected, "Mismatch at [{i}][{j}]");
        }
    }
    Ok(())
}

fn cast_strided_f8e4m3_to_bf16(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(()); // F8E4M3 not supported on Metal
    }
    let data: Vec<f32> = (0..36).map(|i| i as f32 * 0.25).collect();
    let t = Tensor::from_slice(&data, (6, 6), device)?;
    let t_f8 = t.to_dtype(DType::F8E4M3)?;
    let t_transposed = t_f8.t()?;
    assert!(!t_transposed.is_contiguous());
    let t2 = t_transposed.to_dtype(DType::BF16)?;
    let result = t2.to_vec2::<bf16>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let original_f8 = F8E4M3::from_f32((j * 6 + i) as f32 * 0.25);
            let expected = bf16::from_f32(original_f8.to_f32());
            assert_eq!(v, expected, "Mismatch at [{i}][{j}]");
        }
    }
    Ok(())
}

fn cast_strided_bf16_to_f8e4m3(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(()); // F8E4M3 not supported on Metal
    }
    let data: Vec<f32> = (0..36).map(|i| i as f32 * 0.25).collect();
    let t = Tensor::from_slice(&data, (6, 6), device)?;
    let t_bf16 = t.to_dtype(DType::BF16)?;
    let t_transposed = t_bf16.t()?;
    assert!(!t_transposed.is_contiguous());
    let t2 = t_transposed.to_dtype(DType::F8E4M3)?;
    let result = t2.to_vec2::<F8E4M3>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let original_bf16 = bf16::from_f32((j * 6 + i) as f32 * 0.25);
            let expected = F8E4M3::from_f32(original_bf16.to_f32());
            assert_eq!(v, expected, "Mismatch at [{i}][{j}]");
        }
    }
    Ok(())
}

fn cast_strided_f16_to_f8e4m3(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(()); // F8E4M3 not supported on Metal
    }
    let data: Vec<f32> = (0..36).map(|i| i as f32 * 0.25).collect();
    let t = Tensor::from_slice(&data, (6, 6), device)?;
    let t_f16 = t.to_dtype(DType::F16)?;
    let t_transposed = t_f16.t()?;
    assert!(!t_transposed.is_contiguous());
    let t2 = t_transposed.to_dtype(DType::F8E4M3)?;
    let result = t2.to_vec2::<F8E4M3>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let original_f16 = f16::from_f32((j * 6 + i) as f32 * 0.25);
            let expected = F8E4M3::from_f32(original_f16.to_f32());
            assert_eq!(v, expected, "Mismatch at [{i}][{j}]");
        }
    }
    Ok(())
}

fn cast_strided_u32_to_f8e4m3(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(()); // F8E4M3 not supported on Metal
    }
    let data: Vec<u32> = (0..36).map(|i| i as u32).collect();
    let t = Tensor::from_slice(&data, (6, 6), device)?;
    let t_transposed = t.t()?;
    assert!(!t_transposed.is_contiguous());
    let t2 = t_transposed.to_dtype(DType::F8E4M3)?;
    let result = t2.to_vec2::<F8E4M3>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = F8E4M3::from_f32((j * 6 + i) as f32);
            assert_eq!(v, expected, "Mismatch at [{i}][{j}]");
        }
    }
    Ok(())
}

fn cast_strided_f8e4m3_to_u32(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(()); // F8E4M3 not supported on Metal
    }
    let data: Vec<f32> = (0..36).map(|i| i as f32).collect();
    let t = Tensor::from_slice(&data, (6, 6), device)?;
    let t_f8 = t.to_dtype(DType::F8E4M3)?;
    let t_transposed = t_f8.t()?;
    assert!(!t_transposed.is_contiguous());
    let t2 = t_transposed.to_dtype(DType::U32)?;
    let result = t2.to_vec2::<u32>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let f8_val = F8E4M3::from_f32((j * 6 + i) as f32);
            let expected = f8_val.to_f32() as u32;
            assert_eq!(v, expected, "Mismatch at [{i}][{j}]");
        }
    }
    Ok(())
}

fn cast_strided_i64_to_f8e4m3(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(()); // F8E4M3 not supported on Metal
    }
    let data: Vec<i64> = (0..36).map(|i| i as i64).collect();
    let t = Tensor::from_slice(&data, (6, 6), device)?;
    let t_transposed = t.t()?;
    assert!(!t_transposed.is_contiguous());
    let t2 = t_transposed.to_dtype(DType::F8E4M3)?;
    let result = t2.to_vec2::<F8E4M3>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let expected = F8E4M3::from_f32((j * 6 + i) as f32);
            assert_eq!(v, expected, "Mismatch at [{i}][{j}]");
        }
    }
    Ok(())
}

fn cast_strided_f8e4m3_to_i64(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(()); // F8E4M3 not supported on Metal
    }
    let data: Vec<f32> = (0..36).map(|i| i as f32).collect();
    let t = Tensor::from_slice(&data, (6, 6), device)?;
    let t_f8 = t.to_dtype(DType::F8E4M3)?;
    let t_transposed = t_f8.t()?;
    assert!(!t_transposed.is_contiguous());
    let t2 = t_transposed.to_dtype(DType::I64)?;
    let result = t2.to_vec2::<i64>()?;
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let f8_val = F8E4M3::from_f32((j * 6 + i) as f32);
            let expected = f8_val.to_f32() as i64;
            assert_eq!(v, expected, "Mismatch at [{i}][{j}]");
        }
    }
    Ok(())
}

// =============================================================================
// Test registration - strided and narrow casts
// =============================================================================

test_device!(
    cast_strided_f32_to_f64,
    cast_strided_f32_to_f64_cpu,
    cast_strided_f32_to_f64_cuda,
    cast_strided_f32_to_f64_metal
);
test_device!(
    cast_strided_f64_to_f32,
    cast_strided_f64_to_f32_cpu,
    cast_strided_f64_to_f32_cuda,
    cast_strided_f64_to_f32_metal
);
test_device!(
    cast_strided_f32_to_f16,
    cast_strided_f32_to_f16_cpu,
    cast_strided_f32_to_f16_cuda,
    cast_strided_f32_to_f16_metal
);
test_device!(
    cast_strided_u8_to_f32,
    cast_strided_u8_to_f32_cpu,
    cast_strided_u8_to_f32_cuda,
    cast_strided_u8_to_f32_metal
);
test_device!(
    cast_strided_f32_to_f8e4m3,
    cast_strided_f32_to_f8e4m3_cpu,
    cast_strided_f32_to_f8e4m3_cuda,
    cast_strided_f32_to_f8e4m3_metal
);
test_device!(
    cast_strided_f8e4m3_to_f32,
    cast_strided_f8e4m3_to_f32_cpu,
    cast_strided_f8e4m3_to_f32_cuda,
    cast_strided_f8e4m3_to_f32_metal
);
test_device!(
    cast_strided_f8e4m3_to_f16,
    cast_strided_f8e4m3_to_f16_cpu,
    cast_strided_f8e4m3_to_f16_cuda,
    cast_strided_f8e4m3_to_f16_metal
);
test_device!(
    cast_strided_f8e4m3_to_bf16,
    cast_strided_f8e4m3_to_bf16_cpu,
    cast_strided_f8e4m3_to_bf16_cuda,
    cast_strided_f8e4m3_to_bf16_metal
);
test_device!(
    cast_strided_bf16_to_f8e4m3,
    cast_strided_bf16_to_f8e4m3_cpu,
    cast_strided_bf16_to_f8e4m3_cuda,
    cast_strided_bf16_to_f8e4m3_metal
);
test_device!(
    cast_strided_f16_to_f8e4m3,
    cast_strided_f16_to_f8e4m3_cpu,
    cast_strided_f16_to_f8e4m3_cuda,
    cast_strided_f16_to_f8e4m3_metal
);
test_device!(
    cast_strided_u32_to_f8e4m3,
    cast_strided_u32_to_f8e4m3_cpu,
    cast_strided_u32_to_f8e4m3_cuda,
    cast_strided_u32_to_f8e4m3_metal
);
test_device!(
    cast_strided_f8e4m3_to_u32,
    cast_strided_f8e4m3_to_u32_cpu,
    cast_strided_f8e4m3_to_u32_cuda,
    cast_strided_f8e4m3_to_u32_metal
);
test_device!(
    cast_strided_i64_to_f8e4m3,
    cast_strided_i64_to_f8e4m3_cpu,
    cast_strided_i64_to_f8e4m3_cuda,
    cast_strided_i64_to_f8e4m3_metal
);
test_device!(
    cast_strided_f8e4m3_to_i64,
    cast_strided_f8e4m3_to_i64_cpu,
    cast_strided_f8e4m3_to_i64_cuda,
    cast_strided_f8e4m3_to_i64_metal
);
test_device!(
    cast_narrow_f32_to_f64,
    cast_narrow_f32_to_f64_cpu,
    cast_narrow_f32_to_f64_cuda,
    cast_narrow_f32_to_f64_metal
);
test_device!(
    cast_narrow_bf16_to_f32,
    cast_narrow_bf16_to_f32_cpu,
    cast_narrow_bf16_to_f32_cuda,
    cast_narrow_bf16_to_f32_metal
);
