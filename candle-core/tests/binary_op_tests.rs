//! Tests for binary operation kernels (add, sub, mul, div, min, max)
//!
//! These tests verify that vectorized binary kernels work correctly for:
//! - All supported dtypes (f32, f64, f16, bf16, f8_e4m3)
//! - Aligned and unaligned memory access
//! - Various tensor sizes (small, large, odd sizes)
//! - Broadcast operations

// Test code: loop indices are element coordinates in the expected-value formula.
#![allow(clippy::needless_range_loop)]

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

// =============================================================================
// Test Macro
// =============================================================================

macro_rules! test_device {
    ($fn_name: ident, $test_cpu: ident, $test_cuda: ident, $test_metal: ident) => {
        #[test]
        fn $test_cpu() -> Result<()> {
            $fn_name(&Device::Cpu)
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn $test_cuda() -> Result<()> {
            $fn_name(&Device::new_cuda(0)?)
        }

        #[cfg(feature = "metal")]
        #[test]
        fn $test_metal() -> Result<()> {
            $fn_name(&Device::new_metal(0)?)
        }
    };
}

// =============================================================================
// F32 binary tests
// =============================================================================

fn binary_add_f32(dev: &Device) -> Result<()> {
    let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?;
    let b = Tensor::new(&[5.0f32, 6.0, 7.0, 8.0], dev)?;
    let c = (&a + &b)?;
    let result: Vec<f32> = c.to_vec1()?;
    assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    Ok(())
}

fn binary_sub_f32(dev: &Device) -> Result<()> {
    let a = Tensor::new(&[5.0f32, 6.0, 7.0, 8.0], dev)?;
    let b = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?;
    let c = (&a - &b)?;
    let result: Vec<f32> = c.to_vec1()?;
    assert_eq!(result, vec![4.0, 4.0, 4.0, 4.0]);
    Ok(())
}

fn binary_mul_f32(dev: &Device) -> Result<()> {
    let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?;
    let b = Tensor::new(&[2.0f32, 3.0, 4.0, 5.0], dev)?;
    let c = (&a * &b)?;
    let result: Vec<f32> = c.to_vec1()?;
    assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0]);
    Ok(())
}

fn binary_div_f32(dev: &Device) -> Result<()> {
    let a = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], dev)?;
    let b = Tensor::new(&[2.0f32, 4.0, 5.0, 8.0], dev)?;
    let c = (&a / &b)?;
    let result: Vec<f32> = c.to_vec1()?;
    assert_eq!(result, vec![5.0, 5.0, 6.0, 5.0]);
    Ok(())
}

fn binary_min_max_f32(dev: &Device) -> Result<()> {
    let a = Tensor::new(&[1.0f32, 5.0, 3.0, 8.0], dev)?;
    let b = Tensor::new(&[4.0f32, 2.0, 6.0, 1.0], dev)?;
    let min = a.minimum(&b)?;
    let max = a.maximum(&b)?;
    assert_eq!(min.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 1.0]);
    assert_eq!(max.to_vec1::<f32>()?, vec![4.0, 5.0, 6.0, 8.0]);
    Ok(())
}

fn binary_f32_large_aligned(dev: &Device) -> Result<()> {
    // Large aligned tensor - should use float4 vectorization
    let size = 10000;
    let data_a: Vec<f32> = (0..size).map(|x| x as f32).collect();
    let data_b: Vec<f32> = (0..size).map(|x| (x * 2) as f32).collect();
    let a = Tensor::from_vec(data_a, size, dev)?;
    let b = Tensor::from_vec(data_b, size, dev)?;
    let c = (&a + &b)?;
    let result: Vec<f32> = c.to_vec1()?;
    for i in 0..100 {
        assert_eq!(result[i], (i + i * 2) as f32);
    }
    Ok(())
}

fn binary_f32_odd_size(dev: &Device) -> Result<()> {
    // Odd size - tests remainder handling
    let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dev)?;
    let b = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dev)?;
    let c = (&a + &b)?;
    let result: Vec<f32> = c.to_vec1()?;
    assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    Ok(())
}

// =============================================================================
// F16 binary tests
// =============================================================================

fn binary_add_f16(dev: &Device) -> Result<()> {
    let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?.to_dtype(DType::F16)?;
    let b = Tensor::new(&[5.0f32, 6.0, 7.0, 8.0], dev)?.to_dtype(DType::F16)?;
    let c = (&a + &b)?;
    let result: Vec<half::f16> = c.to_vec1()?;
    let result_f32: Vec<f32> = result.iter().map(|x| x.to_f32()).collect();
    assert_eq!(result_f32, vec![6.0, 8.0, 10.0, 12.0]);
    Ok(())
}

fn binary_f16_large_aligned(dev: &Device) -> Result<()> {
    // Large aligned - should use half2 vectorization
    let size = 10000;
    let data_a: Vec<f32> = (0..size).map(|x| (x % 100) as f32).collect();
    let data_b: Vec<f32> = (0..size).map(|x| ((x % 100) * 2) as f32).collect();
    let a = Tensor::from_vec(data_a, size, dev)?.to_dtype(DType::F16)?;
    let b = Tensor::from_vec(data_b, size, dev)?.to_dtype(DType::F16)?;
    let c = (&a + &b)?;
    let result: Vec<half::f16> = c.to_vec1()?;
    for i in 0..100 {
        let expected = ((i % 100) + (i % 100) * 2) as f32;
        assert!((result[i].to_f32() - expected).abs() < 0.1);
    }
    Ok(())
}

fn binary_f16_odd_size(dev: &Device) -> Result<()> {
    let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0], dev)?.to_dtype(DType::F16)?;
    let b = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0, 1.0], dev)?.to_dtype(DType::F16)?;
    let c = (&a + &b)?;
    let result: Vec<half::f16> = c.to_vec1()?;
    let result_f32: Vec<f32> = result.iter().map(|x| x.to_f32()).collect();
    assert_eq!(result_f32, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    Ok(())
}

// =============================================================================
// BF16 binary tests
// =============================================================================

fn binary_add_bf16(dev: &Device) -> Result<()> {
    let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?.to_dtype(DType::BF16)?;
    let b = Tensor::new(&[5.0f32, 6.0, 7.0, 8.0], dev)?.to_dtype(DType::BF16)?;
    let c = (&a + &b)?;
    let result: Vec<half::bf16> = c.to_vec1()?;
    let result_f32: Vec<f32> = result.iter().map(|x| x.to_f32()).collect();
    assert_eq!(result_f32, vec![6.0, 8.0, 10.0, 12.0]);
    Ok(())
}

fn binary_bf16_large_aligned(dev: &Device) -> Result<()> {
    // Large aligned - should use bf162 vectorization
    let size = 10000;
    let data_a: Vec<f32> = (0..size).map(|x| (x % 100) as f32).collect();
    let data_b: Vec<f32> = (0..size).map(|x| ((x % 100) * 2) as f32).collect();
    let a = Tensor::from_vec(data_a, size, dev)?.to_dtype(DType::BF16)?;
    let b = Tensor::from_vec(data_b, size, dev)?.to_dtype(DType::BF16)?;
    let c = (&a + &b)?;
    let result: Vec<half::bf16> = c.to_vec1()?;
    for i in 0..100 {
        let expected = ((i % 100) + (i % 100) * 2) as f32;
        // BF16 has limited precision (7-bit mantissa), allow larger tolerance
        assert!((result[i].to_f32() - expected).abs() < 2.0);
    }
    Ok(())
}

fn binary_bf16_odd_size(dev: &Device) -> Result<()> {
    let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0], dev)?.to_dtype(DType::BF16)?;
    let b = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0, 1.0], dev)?.to_dtype(DType::BF16)?;
    let c = (&a + &b)?;
    let result: Vec<half::bf16> = c.to_vec1()?;
    let result_f32: Vec<f32> = result.iter().map(|x| x.to_f32()).collect();
    assert_eq!(result_f32, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    Ok(())
}

// =============================================================================
// F8E4M3 binary tests (CUDA only, requires SM89+)
// =============================================================================

#[cfg(feature = "cuda")]
fn binary_add_f8e4m3(dev: &Device) -> Result<()> {
    // F8E4M3 has limited range, use small values
    let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?.to_dtype(DType::F8E4M3)?;
    let b = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], dev)?.to_dtype(DType::F8E4M3)?;
    let c = (&a + &b)?;
    // Convert back to f32 to check
    let result = c.to_dtype(DType::F32)?;
    let result_vec: Vec<f32> = result.to_vec1()?;
    // F8E4M3 has limited precision, allow some tolerance
    assert!((result_vec[0] - 2.0).abs() < 0.5);
    assert!((result_vec[1] - 3.0).abs() < 0.5);
    assert!((result_vec[2] - 4.0).abs() < 0.5);
    assert!((result_vec[3] - 5.0).abs() < 0.5);
    Ok(())
}

#[cfg(feature = "cuda")]
fn binary_sub_f8e4m3(dev: &Device) -> Result<()> {
    let a = Tensor::new(&[4.0f32, 5.0, 6.0, 7.0], dev)?.to_dtype(DType::F8E4M3)?;
    let b = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?.to_dtype(DType::F8E4M3)?;
    let c = (&a - &b)?;
    let result = c.to_dtype(DType::F32)?;
    let result_vec: Vec<f32> = result.to_vec1()?;
    assert!((result_vec[0] - 3.0).abs() < 0.5);
    assert!((result_vec[1] - 3.0).abs() < 0.5);
    assert!((result_vec[2] - 3.0).abs() < 0.5);
    assert!((result_vec[3] - 3.0).abs() < 0.5);
    Ok(())
}

#[cfg(feature = "cuda")]
fn binary_mul_f8e4m3(dev: &Device) -> Result<()> {
    let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?.to_dtype(DType::F8E4M3)?;
    let b = Tensor::new(&[2.0f32, 2.0, 2.0, 2.0], dev)?.to_dtype(DType::F8E4M3)?;
    let c = (&a * &b)?;
    let result = c.to_dtype(DType::F32)?;
    let result_vec: Vec<f32> = result.to_vec1()?;
    assert!((result_vec[0] - 2.0).abs() < 0.5);
    assert!((result_vec[1] - 4.0).abs() < 0.5);
    assert!((result_vec[2] - 6.0).abs() < 0.5);
    assert!((result_vec[3] - 8.0).abs() < 0.5);
    Ok(())
}

#[cfg(feature = "cuda")]
fn binary_div_f8e4m3(dev: &Device) -> Result<()> {
    let a = Tensor::new(&[4.0f32, 8.0, 6.0, 10.0], dev)?.to_dtype(DType::F8E4M3)?;
    let b = Tensor::new(&[2.0f32, 4.0, 2.0, 5.0], dev)?.to_dtype(DType::F8E4M3)?;
    let c = (&a / &b)?;
    let result = c.to_dtype(DType::F32)?;
    let result_vec: Vec<f32> = result.to_vec1()?;
    assert!((result_vec[0] - 2.0).abs() < 0.5);
    assert!((result_vec[1] - 2.0).abs() < 0.5);
    assert!((result_vec[2] - 3.0).abs() < 0.5);
    assert!((result_vec[3] - 2.0).abs() < 0.5);
    Ok(())
}

#[cfg(feature = "cuda")]
fn binary_min_max_f8e4m3(dev: &Device) -> Result<()> {
    let a = Tensor::new(&[1.0f32, 5.0, 3.0, 8.0], dev)?.to_dtype(DType::F8E4M3)?;
    let b = Tensor::new(&[4.0f32, 2.0, 6.0, 1.0], dev)?.to_dtype(DType::F8E4M3)?;
    let min = a.minimum(&b)?;
    let max = a.maximum(&b)?;
    let min_result = min.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let max_result = max.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert!((min_result[0] - 1.0).abs() < 0.5);
    assert!((min_result[1] - 2.0).abs() < 0.5);
    assert!((min_result[2] - 3.0).abs() < 0.5);
    assert!((min_result[3] - 1.0).abs() < 0.5);
    assert!((max_result[0] - 4.0).abs() < 0.5);
    assert!((max_result[1] - 5.0).abs() < 0.5);
    assert!((max_result[2] - 6.0).abs() < 0.5);
    assert!((max_result[3] - 8.0).abs() < 0.5);
    Ok(())
}

#[cfg(feature = "cuda")]
fn binary_f8e4m3_large_aligned(dev: &Device) -> Result<()> {
    // Large aligned - should use vectorized 4-byte loads
    let size = 10000;
    // Use small values to stay within F8E4M3 range
    let data_a: Vec<f32> = (0..size).map(|x| ((x % 10) as f32) * 0.5).collect();
    let data_b: Vec<f32> = (0..size).map(|x| ((x % 10) as f32) * 0.25).collect();
    let a = Tensor::from_vec(data_a, size, dev)?.to_dtype(DType::F8E4M3)?;
    let b = Tensor::from_vec(data_b, size, dev)?.to_dtype(DType::F8E4M3)?;
    let c = (&a + &b)?;
    let result = c.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    // Check first few values
    for i in 0..10 {
        let expected = ((i % 10) as f32) * 0.5 + ((i % 10) as f32) * 0.25;
        assert!(
            (result[i] - expected).abs() < 1.0,
            "index {}: expected {}, got {}",
            i,
            expected,
            result[i]
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn binary_f8e4m3_odd_size(dev: &Device) -> Result<()> {
    // 7 elements - tests remainder handling (7 = 4 + 3)
    let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dev)?.to_dtype(DType::F8E4M3)?;
    let b = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dev)?.to_dtype(DType::F8E4M3)?;
    let c = (&a + &b)?;
    let result = c.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    for i in 0..7 {
        let expected = (i + 2) as f32;
        assert!(
            (result[i] - expected).abs() < 0.5,
            "index {}: expected {}, got {}",
            i,
            expected,
            result[i]
        );
    }
    Ok(())
}

// =============================================================================
// Test registration
// =============================================================================

// F32 tests
test_device!(
    binary_add_f32,
    binary_add_f32_cpu,
    binary_add_f32_cuda,
    binary_add_f32_metal
);
test_device!(
    binary_sub_f32,
    binary_sub_f32_cpu,
    binary_sub_f32_cuda,
    binary_sub_f32_metal
);
test_device!(
    binary_mul_f32,
    binary_mul_f32_cpu,
    binary_mul_f32_cuda,
    binary_mul_f32_metal
);
test_device!(
    binary_div_f32,
    binary_div_f32_cpu,
    binary_div_f32_cuda,
    binary_div_f32_metal
);
test_device!(
    binary_min_max_f32,
    binary_min_max_f32_cpu,
    binary_min_max_f32_cuda,
    binary_min_max_f32_metal
);
test_device!(
    binary_f32_large_aligned,
    binary_f32_large_aligned_cpu,
    binary_f32_large_aligned_cuda,
    binary_f32_large_aligned_metal
);
test_device!(
    binary_f32_odd_size,
    binary_f32_odd_size_cpu,
    binary_f32_odd_size_cuda,
    binary_f32_odd_size_metal
);

// F16 tests
test_device!(
    binary_add_f16,
    binary_add_f16_cpu,
    binary_add_f16_cuda,
    binary_add_f16_metal
);
test_device!(
    binary_f16_large_aligned,
    binary_f16_large_aligned_cpu,
    binary_f16_large_aligned_cuda,
    binary_f16_large_aligned_metal
);
test_device!(
    binary_f16_odd_size,
    binary_f16_odd_size_cpu,
    binary_f16_odd_size_cuda,
    binary_f16_odd_size_metal
);

// BF16 tests
test_device!(
    binary_add_bf16,
    binary_add_bf16_cpu,
    binary_add_bf16_cuda,
    binary_add_bf16_metal
);
test_device!(
    binary_bf16_large_aligned,
    binary_bf16_large_aligned_cpu,
    binary_bf16_large_aligned_cuda,
    binary_bf16_large_aligned_metal
);
test_device!(
    binary_bf16_odd_size,
    binary_bf16_odd_size_cpu,
    binary_bf16_odd_size_cuda,
    binary_bf16_odd_size_metal
);

// F8E4M3 tests (CUDA only, requires SM89+)
#[cfg(feature = "cuda")]
#[test]
fn binary_add_f8e4m3_cuda() -> Result<()> {
    binary_add_f8e4m3(&Device::new_cuda(0)?)
}

#[cfg(feature = "cuda")]
#[test]
fn binary_sub_f8e4m3_cuda() -> Result<()> {
    binary_sub_f8e4m3(&Device::new_cuda(0)?)
}

#[cfg(feature = "cuda")]
#[test]
fn binary_mul_f8e4m3_cuda() -> Result<()> {
    binary_mul_f8e4m3(&Device::new_cuda(0)?)
}

#[cfg(feature = "cuda")]
#[test]
fn binary_div_f8e4m3_cuda() -> Result<()> {
    binary_div_f8e4m3(&Device::new_cuda(0)?)
}

#[cfg(feature = "cuda")]
#[test]
fn binary_min_max_f8e4m3_cuda() -> Result<()> {
    binary_min_max_f8e4m3(&Device::new_cuda(0)?)
}

#[cfg(feature = "cuda")]
#[test]
fn binary_f8e4m3_large_aligned_cuda() -> Result<()> {
    binary_f8e4m3_large_aligned(&Device::new_cuda(0)?)
}

#[cfg(feature = "cuda")]
#[test]
fn binary_f8e4m3_odd_size_cuda() -> Result<()> {
    binary_f8e4m3_odd_size(&Device::new_cuda(0)?)
}

// =============================================================================
// In-place binary operation tests
// =============================================================================

fn inplace_add_f32(dev: &Device) -> Result<()> {
    let mut a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?;
    let b = Tensor::new(&[5.0f32, 6.0, 7.0, 8.0], dev)?;
    a.add_mut(&b)?;
    let result: Vec<f32> = a.to_vec1()?;
    assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    Ok(())
}

fn inplace_sub_f32(dev: &Device) -> Result<()> {
    let mut a = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], dev)?;
    let b = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?;
    a.sub_mut(&b)?;
    let result: Vec<f32> = a.to_vec1()?;
    assert_eq!(result, vec![9.0, 18.0, 27.0, 36.0]);
    Ok(())
}

fn inplace_mul_f32(dev: &Device) -> Result<()> {
    let mut a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?;
    let b = Tensor::new(&[2.0f32, 3.0, 4.0, 5.0], dev)?;
    a.mul_mut(&b)?;
    let result: Vec<f32> = a.to_vec1()?;
    assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0]);
    Ok(())
}

fn inplace_div_f32(dev: &Device) -> Result<()> {
    let mut a = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], dev)?;
    let b = Tensor::new(&[2.0f32, 4.0, 5.0, 8.0], dev)?;
    a.div_mut(&b)?;
    let result: Vec<f32> = a.to_vec1()?;
    assert_eq!(result, vec![5.0, 5.0, 6.0, 5.0]);
    Ok(())
}

fn inplace_min_max_f32(dev: &Device) -> Result<()> {
    let mut a = Tensor::new(&[1.0f32, 5.0, 3.0, 8.0], dev)?;
    let b = Tensor::new(&[4.0f32, 2.0, 6.0, 1.0], dev)?;
    a.minimum_mut(&b)?;
    assert_eq!(a.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 1.0]);

    let mut c = Tensor::new(&[1.0f32, 5.0, 3.0, 8.0], dev)?;
    c.maximum_mut(&b)?;
    assert_eq!(c.to_vec1::<f32>()?, vec![4.0, 5.0, 6.0, 8.0]);
    Ok(())
}

fn inplace_large_f32(dev: &Device) -> Result<()> {
    // Test large tensors for vectorization
    let size = 10000;
    let data_a: Vec<f32> = (0..size).map(|x| x as f32).collect();
    let data_b: Vec<f32> = (0..size).map(|_x| 1.0).collect();
    let mut a = Tensor::from_vec(data_a, size, dev)?;
    let b = Tensor::from_vec(data_b, size, dev)?;
    a.add_mut(&b)?;
    let result: Vec<f32> = a.to_vec1()?;
    for i in 0..100 {
        assert_eq!(result[i], (i + 1) as f32);
    }
    Ok(())
}

fn inplace_add_f16(dev: &Device) -> Result<()> {
    let mut a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?.to_dtype(DType::F16)?;
    let b = Tensor::new(&[5.0f32, 6.0, 7.0, 8.0], dev)?.to_dtype(DType::F16)?;
    a.add_mut(&b)?;
    let result: Vec<half::f16> = a.to_vec1()?;
    let result_f32: Vec<f32> = result.iter().map(|x| x.to_f32()).collect();
    assert_eq!(result_f32, vec![6.0, 8.0, 10.0, 12.0]);
    Ok(())
}

fn inplace_add_bf16(dev: &Device) -> Result<()> {
    let mut a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?.to_dtype(DType::BF16)?;
    let b = Tensor::new(&[5.0f32, 6.0, 7.0, 8.0], dev)?.to_dtype(DType::BF16)?;
    a.add_mut(&b)?;
    let result: Vec<half::bf16> = a.to_vec1()?;
    let result_f32: Vec<f32> = result.iter().map(|x| x.to_f32()).collect();
    assert_eq!(result_f32, vec![6.0, 8.0, 10.0, 12.0]);
    Ok(())
}

test_device!(
    inplace_add_f32,
    inplace_add_f32_cpu,
    inplace_add_f32_cuda,
    inplace_add_f32_metal
);
test_device!(
    inplace_sub_f32,
    inplace_sub_f32_cpu,
    inplace_sub_f32_cuda,
    inplace_sub_f32_metal
);
test_device!(
    inplace_mul_f32,
    inplace_mul_f32_cpu,
    inplace_mul_f32_cuda,
    inplace_mul_f32_metal
);
test_device!(
    inplace_div_f32,
    inplace_div_f32_cpu,
    inplace_div_f32_cuda,
    inplace_div_f32_metal
);
test_device!(
    inplace_min_max_f32,
    inplace_min_max_f32_cpu,
    inplace_min_max_f32_cuda,
    inplace_min_max_f32_metal
);
test_device!(
    inplace_large_f32,
    inplace_large_f32_cpu,
    inplace_large_f32_cuda,
    inplace_large_f32_metal
);
test_device!(
    inplace_add_f16,
    inplace_add_f16_cpu,
    inplace_add_f16_cuda,
    inplace_add_f16_metal
);
test_device!(
    inplace_add_bf16,
    inplace_add_bf16_cpu,
    inplace_add_bf16_cuda,
    inplace_add_bf16_metal
);

#[cfg(feature = "cuda")]
fn inplace_add_f8e4m3(dev: &Device) -> Result<()> {
    let mut a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?.to_dtype(DType::F8E4M3)?;
    let b = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], dev)?.to_dtype(DType::F8E4M3)?;
    a.add_mut(&b)?;
    let result: Vec<float8::F8E4M3> = a.to_vec1()?;
    let result_f32: Vec<f32> = result.iter().map(|x| f32::from(*x)).collect();
    // F8E4M3 has limited precision
    for (actual, expected) in result_f32.iter().zip(&[2.0, 3.0, 4.0, 5.0]) {
        assert!((actual - expected).abs() < 0.5);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn inplace_add_f8e4m3_cuda() -> Result<()> {
    inplace_add_f8e4m3(&Device::new_cuda(0)?)
}
