use anyhow::Result;
use candle_core::{DType, Device, Tensor};

// =============================================================================
// Basic to_dtype_mut Tests
// =============================================================================

fn to_dtype_mut_same_type(dev: &Device) -> Result<()> {
    let mut t = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?;
    let original_data: Vec<f32> = t.to_vec1()?;
    t.to_dtype_mut(DType::F32)?;
    assert_eq!(t.dtype(), DType::F32);
    let result: Vec<f32> = t.to_vec1()?;
    assert_eq!(original_data, result);
    Ok(())
}

fn to_dtype_mut_f32_to_f64(dev: &Device) -> Result<()> {
    let mut t = Tensor::new(&[1.0f32, 2.5, 3.75, -4.125], dev)?;
    t.to_dtype_mut(DType::F64)?;
    assert_eq!(t.dtype(), DType::F64);
    let result: Vec<f64> = t.to_vec1()?;
    // f32 -> f64 should preserve values exactly
    assert_eq!(result, vec![1.0f64, 2.5, 3.75, -4.125]);
    Ok(())
}

fn to_dtype_mut_f64_to_f32(dev: &Device) -> Result<()> {
    let mut t = Tensor::new(&[1.0f64, 2.5, 3.75, -4.125], dev)?;
    t.to_dtype_mut(DType::F32)?;
    assert_eq!(t.dtype(), DType::F32);
    let result: Vec<f32> = t.to_vec1()?;
    assert_eq!(result, vec![1.0f32, 2.5, 3.75, -4.125]);
    Ok(())
}

fn to_dtype_mut_f32_to_f16(dev: &Device) -> Result<()> {
    let mut t = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?;
    t.to_dtype_mut(DType::F16)?;
    assert_eq!(t.dtype(), DType::F16);
    // Convert back to f32 to check values
    let result = t.to_dtype(DType::F32)?;
    let result: Vec<f32> = result.to_vec1()?;
    assert_eq!(result, vec![1.0f32, 2.0, 3.0, 4.0]);
    Ok(())
}

fn to_dtype_mut_f32_to_bf16(dev: &Device) -> Result<()> {
    let mut t = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?;
    t.to_dtype_mut(DType::BF16)?;
    assert_eq!(t.dtype(), DType::BF16);
    // Convert back to f32 to check values
    let result = t.to_dtype(DType::F32)?;
    let result: Vec<f32> = result.to_vec1()?;
    assert_eq!(result, vec![1.0f32, 2.0, 3.0, 4.0]);
    Ok(())
}

fn to_dtype_mut_u8_to_f32(dev: &Device) -> Result<()> {
    let mut t = Tensor::new(&[1u8, 2, 127, 255], dev)?;
    t.to_dtype_mut(DType::F32)?;
    assert_eq!(t.dtype(), DType::F32);
    let result: Vec<f32> = t.to_vec1()?;
    assert_eq!(result, vec![1.0f32, 2.0, 127.0, 255.0]);
    Ok(())
}

fn to_dtype_mut_f32_to_u8(dev: &Device) -> Result<()> {
    let mut t = Tensor::new(&[1.0f32, 2.0, 127.0, 200.0], dev)?;
    t.to_dtype_mut(DType::U8)?;
    assert_eq!(t.dtype(), DType::U8);
    let result: Vec<u8> = t.to_vec1()?;
    assert_eq!(result, vec![1u8, 2, 127, 200]);
    Ok(())
}

fn to_dtype_mut_u32_to_i64(dev: &Device) -> Result<()> {
    let mut t = Tensor::new(&[1u32, 1000, 1000000, u32::MAX], dev)?;
    t.to_dtype_mut(DType::I64)?;
    assert_eq!(t.dtype(), DType::I64);
    let result: Vec<i64> = t.to_vec1()?;
    assert_eq!(result, vec![1i64, 1000, 1000000, u32::MAX as i64]);
    Ok(())
}

fn to_dtype_mut_i64_to_f32(dev: &Device) -> Result<()> {
    let mut t = Tensor::new(&[1i64, -1, 1000, -1000], dev)?;
    t.to_dtype_mut(DType::F32)?;
    assert_eq!(t.dtype(), DType::F32);
    let result: Vec<f32> = t.to_vec1()?;
    assert_eq!(result, vec![1.0f32, -1.0, 1000.0, -1000.0]);
    Ok(())
}

// =============================================================================
// Multi-dimensional Tests
// =============================================================================

fn to_dtype_mut_2d(dev: &Device) -> Result<()> {
    let mut t = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], dev)?;
    t.to_dtype_mut(DType::F64)?;
    assert_eq!(t.dtype(), DType::F64);
    assert_eq!(t.dims(), &[2, 2]);
    let result: Vec<Vec<f64>> = t.to_vec2()?;
    assert_eq!(result, vec![vec![1.0f64, 2.0], vec![3.0, 4.0]]);
    Ok(())
}

fn to_dtype_mut_3d(dev: &Device) -> Result<()> {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let mut t = Tensor::from_vec(data, (2, 3, 4), dev)?;
    t.to_dtype_mut(DType::F16)?;
    assert_eq!(t.dtype(), DType::F16);
    assert_eq!(t.dims(), &[2, 3, 4]);
    // Verify values preserved
    let result = t.to_dtype(DType::F32)?;
    let result: Vec<f32> = result.flatten_all()?.to_vec1()?;
    let expected: Vec<f32> = (0..24).map(|x| x as f32).collect();
    assert_eq!(result, expected);
    Ok(())
}

// =============================================================================
// Chain conversion Tests
// =============================================================================

fn to_dtype_mut_chain(dev: &Device) -> Result<()> {
    let mut t = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?;
    
    // f32 -> f64
    t.to_dtype_mut(DType::F64)?;
    assert_eq!(t.dtype(), DType::F64);
    
    // f64 -> f16
    t.to_dtype_mut(DType::F16)?;
    assert_eq!(t.dtype(), DType::F16);
    
    // f16 -> bf16
    t.to_dtype_mut(DType::BF16)?;
    assert_eq!(t.dtype(), DType::BF16);
    
    // bf16 -> f32
    t.to_dtype_mut(DType::F32)?;
    assert_eq!(t.dtype(), DType::F32);
    
    let result: Vec<f32> = t.to_vec1()?;
    assert_eq!(result, vec![1.0f32, 2.0, 3.0, 4.0]);
    Ok(())
}

// =============================================================================
// Large tensor Tests
// =============================================================================

fn to_dtype_mut_large(dev: &Device) -> Result<()> {
    let size = 100_000;
    let data: Vec<f32> = (0..size).map(|x| x as f32).collect();
    let mut t = Tensor::from_vec(data.clone(), size, dev)?;
    
    t.to_dtype_mut(DType::F64)?;
    assert_eq!(t.dtype(), DType::F64);
    
    // Verify first and last few values
    let result: Vec<f64> = t.to_vec1()?;
    assert_eq!(result[0], 0.0f64);
    assert_eq!(result[1], 1.0f64);
    assert_eq!(result[size - 1], (size - 1) as f64);
    Ok(())
}

// =============================================================================
// Non-contiguous tensor Tests
// =============================================================================

fn to_dtype_mut_non_contiguous_transpose(dev: &Device) -> Result<()> {
    // Create a 2D tensor and transpose it (making it non-contiguous)
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::from_vec(data, (2, 3), dev)?;
    // Transpose makes it non-contiguous
    let mut t = t.t()?;
    assert!(!t.is_contiguous());
    
    // to_dtype_mut should make it contiguous first, then convert
    t.to_dtype_mut(DType::F64)?;
    assert_eq!(t.dtype(), DType::F64);
    assert!(t.is_contiguous());
    
    // Verify the transposed values are correct
    // Original: [[1, 2, 3], [4, 5, 6]]
    // Transposed: [[1, 4], [2, 5], [3, 6]]
    let result: Vec<Vec<f64>> = t.to_vec2()?;
    assert_eq!(result, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
    Ok(())
}

fn to_dtype_mut_non_contiguous_narrow(dev: &Device) -> Result<()> {
    // Create a tensor and narrow it (making it non-contiguous)
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let t = Tensor::from_vec(data, (3, 3), dev)?;
    // Narrow the middle column
    let mut t = t.narrow(1, 1, 1)?;
    assert!(!t.is_contiguous());
    
    t.to_dtype_mut(DType::F64)?;
    assert_eq!(t.dtype(), DType::F64);
    assert!(t.is_contiguous());
    
    // Middle column: [2, 5, 8]
    let result: Vec<Vec<f64>> = t.to_vec2()?;
    assert_eq!(result, vec![vec![2.0], vec![5.0], vec![8.0]]);
    Ok(())
}

// =============================================================================
// Test Macros
// =============================================================================

macro_rules! test_device {
    ($fn_name: ident, $test_cpu: ident, $test_cuda: ident, $test_metal: ident) => {
        #[test]
        fn $test_cpu() -> Result<()> {
            $fn_name(&Device::Cpu)
        }

        #[cfg(feature = "cuda")]
        #[test]
        #[serial_test::serial(cuda)]
        fn $test_cuda() -> Result<()> {
            $fn_name(&Device::new_cuda(0)?)
        }

        #[cfg(feature = "metal")]
        #[test]
        #[serial_test::serial(metal)]
        fn $test_metal() -> Result<()> {
            $fn_name(&Device::new_metal(0)?)
        }
    };
}

test_device!(to_dtype_mut_same_type, to_dtype_mut_same_type_cpu, to_dtype_mut_same_type_cuda, to_dtype_mut_same_type_metal);
test_device!(to_dtype_mut_f32_to_f64, to_dtype_mut_f32_to_f64_cpu, to_dtype_mut_f32_to_f64_cuda, to_dtype_mut_f32_to_f64_metal);
test_device!(to_dtype_mut_f64_to_f32, to_dtype_mut_f64_to_f32_cpu, to_dtype_mut_f64_to_f32_cuda, to_dtype_mut_f64_to_f32_metal);
test_device!(to_dtype_mut_f32_to_f16, to_dtype_mut_f32_to_f16_cpu, to_dtype_mut_f32_to_f16_cuda, to_dtype_mut_f32_to_f16_metal);
test_device!(to_dtype_mut_f32_to_bf16, to_dtype_mut_f32_to_bf16_cpu, to_dtype_mut_f32_to_bf16_cuda, to_dtype_mut_f32_to_bf16_metal);
test_device!(to_dtype_mut_u8_to_f32, to_dtype_mut_u8_to_f32_cpu, to_dtype_mut_u8_to_f32_cuda, to_dtype_mut_u8_to_f32_metal);
test_device!(to_dtype_mut_f32_to_u8, to_dtype_mut_f32_to_u8_cpu, to_dtype_mut_f32_to_u8_cuda, to_dtype_mut_f32_to_u8_metal);
test_device!(to_dtype_mut_u32_to_i64, to_dtype_mut_u32_to_i64_cpu, to_dtype_mut_u32_to_i64_cuda, to_dtype_mut_u32_to_i64_metal);
test_device!(to_dtype_mut_i64_to_f32, to_dtype_mut_i64_to_f32_cpu, to_dtype_mut_i64_to_f32_cuda, to_dtype_mut_i64_to_f32_metal);
test_device!(to_dtype_mut_2d, to_dtype_mut_2d_cpu, to_dtype_mut_2d_cuda, to_dtype_mut_2d_metal);
test_device!(to_dtype_mut_3d, to_dtype_mut_3d_cpu, to_dtype_mut_3d_cuda, to_dtype_mut_3d_metal);
test_device!(to_dtype_mut_chain, to_dtype_mut_chain_cpu, to_dtype_mut_chain_cuda, to_dtype_mut_chain_metal);
test_device!(to_dtype_mut_large, to_dtype_mut_large_cpu, to_dtype_mut_large_cuda, to_dtype_mut_large_metal);
test_device!(to_dtype_mut_non_contiguous_transpose, to_dtype_mut_non_contiguous_transpose_cpu, to_dtype_mut_non_contiguous_transpose_cuda, to_dtype_mut_non_contiguous_transpose_metal);
test_device!(to_dtype_mut_non_contiguous_narrow, to_dtype_mut_non_contiguous_narrow_cpu, to_dtype_mut_non_contiguous_narrow_cuda, to_dtype_mut_non_contiguous_narrow_metal);
