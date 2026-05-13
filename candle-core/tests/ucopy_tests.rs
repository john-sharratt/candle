//! Tests for ucopy kernels (strided copy operations used by .contiguous())
//!
//! These tests verify that the vectorized ucopy kernels work correctly for:
//! - All supported dtypes (f32, f64, f16, bf16, u8, u32, i64)
//! - Aligned and unaligned memory access
//! - Various tensor sizes (small, large, odd sizes)
//! - Strided (non-contiguous) tensors

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

// =============================================================================
// F32 ucopy tests
// =============================================================================

fn ucopy_f32_contiguous(dev: &Device) -> Result<()> {
    // Already contiguous - should return clone
    let t = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?;
    assert!(t.is_contiguous());
    let t_contig = t.contiguous()?;
    assert!(t_contig.is_contiguous());
    let result: Vec<f32> = t_contig.to_vec1()?;
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

fn ucopy_f32_transpose(dev: &Device) -> Result<()> {
    // Transposed tensor requires actual copy
    let t = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], dev)?;
    let t_t = t.t()?;
    assert!(!t_t.is_contiguous());
    let t_contig = t_t.contiguous()?;
    assert!(t_contig.is_contiguous());
    let result: Vec<Vec<f32>> = t_contig.to_vec2()?;
    assert_eq!(result, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
    Ok(())
}

fn ucopy_f32_large_aligned(dev: &Device) -> Result<()> {
    // Large aligned tensor - should use float4 vectorization
    let size = 10000;
    let data: Vec<f32> = (0..size).map(|x| x as f32).collect();
    let t = Tensor::from_vec(data.clone(), (100, 100), dev)?;
    let t_t = t.t()?;
    assert!(!t_t.is_contiguous());
    let t_contig = t_t.contiguous()?;
    assert!(t_contig.is_contiguous());

    // Check first row of result (should be first column of original)
    let result: Vec<Vec<f32>> = t_contig.to_vec2()?;
    for i in 0..100 {
        assert_eq!(result[0][i], (i * 100) as f32);
    }
    Ok(())
}

fn ucopy_f32_odd_size(dev: &Device) -> Result<()> {
    // Odd size - tests remainder handling
    let t = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dev)?;
    let t_t = t.t()?;
    let t_contig = t_t.contiguous()?;
    let result: Vec<Vec<f32>> = t_contig.to_vec2()?;
    assert_eq!(
        result,
        vec![
            vec![1.0, 4.0, 7.0],
            vec![2.0, 5.0, 8.0],
            vec![3.0, 6.0, 9.0]
        ]
    );
    Ok(())
}

// =============================================================================
// F64 ucopy tests
// =============================================================================

fn ucopy_f64_transpose(dev: &Device) -> Result<()> {
    let t = Tensor::new(&[[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]], dev)?;
    let t_t = t.t()?;
    assert!(!t_t.is_contiguous());
    let t_contig = t_t.contiguous()?;
    assert!(t_contig.is_contiguous());
    let result: Vec<Vec<f64>> = t_contig.to_vec2()?;
    assert_eq!(result, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
    Ok(())
}

fn ucopy_f64_large(dev: &Device) -> Result<()> {
    let size = 5000;
    let data: Vec<f64> = (0..size).map(|x| x as f64).collect();
    let t = Tensor::from_vec(data.clone(), (50, 100), dev)?;
    let t_t = t.t()?;
    let t_contig = t_t.contiguous()?;

    let result: Vec<Vec<f64>> = t_contig.to_vec2()?;
    for i in 0..50 {
        assert_eq!(result[0][i], (i * 100) as f64);
    }
    Ok(())
}

// =============================================================================
// U8 ucopy tests
// =============================================================================

fn ucopy_u8_transpose(dev: &Device) -> Result<()> {
    let t = Tensor::new(&[[1u8, 2, 3, 4], [5, 6, 7, 8]], dev)?;
    let t_t = t.t()?;
    assert!(!t_t.is_contiguous());
    let t_contig = t_t.contiguous()?;
    assert!(t_contig.is_contiguous());
    let result: Vec<Vec<u8>> = t_contig.to_vec2()?;
    assert_eq!(result, vec![vec![1, 5], vec![2, 6], vec![3, 7], vec![4, 8]]);
    Ok(())
}

fn ucopy_u8_large_aligned(dev: &Device) -> Result<()> {
    // Large aligned - should use uchar4 vectorization
    let size = 10000;
    let data: Vec<u8> = (0..size).map(|x| (x % 256) as u8).collect();
    let t = Tensor::from_vec(data.clone(), (100, 100), dev)?;
    let t_t = t.t()?;
    let t_contig = t_t.contiguous()?;

    let result: Vec<Vec<u8>> = t_contig.to_vec2()?;
    for i in 0..100 {
        assert_eq!(result[0][i], ((i * 100) % 256) as u8);
    }
    Ok(())
}

fn ucopy_u8_odd_size(dev: &Device) -> Result<()> {
    // 5 elements - tests remainder after uchar4 (1 vector + 1 remainder)
    let t = Tensor::new(&[[1u8, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dev)?;
    let t_t = t.t()?;
    let t_contig = t_t.contiguous()?;
    let result: Vec<Vec<u8>> = t_contig.to_vec2()?;
    assert_eq!(
        result,
        vec![vec![1, 6], vec![2, 7], vec![3, 8], vec![4, 9], vec![5, 10]]
    );
    Ok(())
}

// =============================================================================
// U32 ucopy tests
// =============================================================================

fn ucopy_u32_transpose(dev: &Device) -> Result<()> {
    let t = Tensor::new(&[[1u32, 2, 3, 4], [5, 6, 7, 8]], dev)?;
    let t_t = t.t()?;
    assert!(!t_t.is_contiguous());
    let t_contig = t_t.contiguous()?;
    assert!(t_contig.is_contiguous());
    let result: Vec<Vec<u32>> = t_contig.to_vec2()?;
    assert_eq!(result, vec![vec![1, 5], vec![2, 6], vec![3, 7], vec![4, 8]]);
    Ok(())
}

fn ucopy_u32_large(dev: &Device) -> Result<()> {
    let size = 10000;
    let data: Vec<u32> = (0..size).map(|x| x as u32).collect();
    let t = Tensor::from_vec(data.clone(), (100, 100), dev)?;
    let t_t = t.t()?;
    let t_contig = t_t.contiguous()?;

    let result: Vec<Vec<u32>> = t_contig.to_vec2()?;
    for i in 0..100 {
        assert_eq!(result[0][i], (i * 100) as u32);
    }
    Ok(())
}

fn ucopy_u32_odd_size(dev: &Device) -> Result<()> {
    // 5 elements - tests remainder after uint4
    let t = Tensor::new(&[[1u32, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dev)?;
    let t_t = t.t()?;
    let t_contig = t_t.contiguous()?;
    let result: Vec<Vec<u32>> = t_contig.to_vec2()?;
    assert_eq!(
        result,
        vec![vec![1, 6], vec![2, 7], vec![3, 8], vec![4, 9], vec![5, 10]]
    );
    Ok(())
}

// =============================================================================
// I64 ucopy tests
// =============================================================================

fn ucopy_i64_transpose(dev: &Device) -> Result<()> {
    let t = Tensor::new(&[[1i64, 2, 3], [4, 5, 6]], dev)?;
    let t_t = t.t()?;
    assert!(!t_t.is_contiguous());
    let t_contig = t_t.contiguous()?;
    assert!(t_contig.is_contiguous());
    let result: Vec<Vec<i64>> = t_contig.to_vec2()?;
    assert_eq!(result, vec![vec![1, 4], vec![2, 5], vec![3, 6]]);
    Ok(())
}

fn ucopy_i64_large(dev: &Device) -> Result<()> {
    let size = 5000;
    let data: Vec<i64> = (0..size).map(|x| x as i64).collect();
    let t = Tensor::from_vec(data.clone(), (50, 100), dev)?;
    let t_t = t.t()?;
    let t_contig = t_t.contiguous()?;

    let result: Vec<Vec<i64>> = t_contig.to_vec2()?;
    for i in 0..50 {
        assert_eq!(result[0][i], (i * 100) as i64);
    }
    Ok(())
}

fn ucopy_i64_odd_size(dev: &Device) -> Result<()> {
    // 3 elements - tests remainder after longlong2
    let t = Tensor::new(&[[1i64, 2, 3], [4, 5, 6], [7, 8, 9]], dev)?;
    let t_t = t.t()?;
    let t_contig = t_t.contiguous()?;
    let result: Vec<Vec<i64>> = t_contig.to_vec2()?;
    assert_eq!(result, vec![vec![1, 4, 7], vec![2, 5, 8], vec![3, 6, 9]]);
    Ok(())
}

// =============================================================================
// F16 ucopy tests
// =============================================================================

fn ucopy_f16_transpose(dev: &Device) -> Result<()> {
    let t = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], dev)?.to_dtype(DType::F16)?;
    let t_t = t.t()?;
    assert!(!t_t.is_contiguous());
    let t_contig = t_t.contiguous()?;
    assert!(t_contig.is_contiguous());

    // Convert back to f32 for comparison
    let result = t_contig.to_dtype(DType::F32)?;
    let result: Vec<Vec<f32>> = result.to_vec2()?;
    assert_eq!(result, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
    Ok(())
}

fn ucopy_f16_large(dev: &Device) -> Result<()> {
    let size = 10000;
    let data: Vec<f32> = (0..size).map(|x| x as f32).collect();
    let t = Tensor::from_vec(data.clone(), (100, 100), dev)?.to_dtype(DType::F16)?;
    let t_t = t.t()?;
    let t_contig = t_t.contiguous()?;

    let result = t_contig.to_dtype(DType::F32)?;
    let result: Vec<Vec<f32>> = result.to_vec2()?;
    // Check structure is correct (first column of original should be first row of transposed)
    // Use approximate check due to f16 precision
    for i in 0..10 {
        let expected = (i * 100) as f32;
        let actual = result[0][i];
        assert!(
            (actual - expected).abs() < 10.0,
            "mismatch at [0][{}]: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
    Ok(())
}

fn ucopy_f16_odd_size(dev: &Device) -> Result<()> {
    // 5 elements - tests remainder after half2
    let t = Tensor::new(
        &[[1.0f32, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]],
        dev,
    )?
    .to_dtype(DType::F16)?;
    let t_t = t.t()?;
    let t_contig = t_t.contiguous()?;

    let result = t_contig.to_dtype(DType::F32)?;
    let result: Vec<Vec<f32>> = result.to_vec2()?;
    assert_eq!(
        result,
        vec![
            vec![1.0, 6.0],
            vec![2.0, 7.0],
            vec![3.0, 8.0],
            vec![4.0, 9.0],
            vec![5.0, 10.0]
        ]
    );
    Ok(())
}

// =============================================================================
// BF16 ucopy tests
// =============================================================================

fn ucopy_bf16_transpose(dev: &Device) -> Result<()> {
    let t = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], dev)?.to_dtype(DType::BF16)?;
    let t_t = t.t()?;
    assert!(!t_t.is_contiguous());
    let t_contig = t_t.contiguous()?;
    assert!(t_contig.is_contiguous());

    let result = t_contig.to_dtype(DType::F32)?;
    let result: Vec<Vec<f32>> = result.to_vec2()?;
    assert_eq!(result, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
    Ok(())
}

fn ucopy_bf16_large(dev: &Device) -> Result<()> {
    let size = 10000;
    let data: Vec<f32> = (0..size).map(|x| x as f32).collect();
    let t = Tensor::from_vec(data.clone(), (100, 100), dev)?.to_dtype(DType::BF16)?;
    let t_t = t.t()?;
    let t_contig = t_t.contiguous()?;

    let result = t_contig.to_dtype(DType::F32)?;
    let result: Vec<Vec<f32>> = result.to_vec2()?;
    // Check structure is correct (first column of original should be first row of transposed)
    // Use approximate check due to bf16 precision (even less precise than f16)
    for i in 0..10 {
        let expected = (i * 100) as f32;
        let actual = result[0][i];
        assert!(
            (actual - expected).abs() < 50.0,
            "mismatch at [0][{}]: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
    Ok(())
}

fn ucopy_bf16_odd_size(dev: &Device) -> Result<()> {
    // 5 elements - tests remainder after bf16x2
    let t = Tensor::new(
        &[[1.0f32, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]],
        dev,
    )?
    .to_dtype(DType::BF16)?;
    let t_t = t.t()?;
    let t_contig = t_t.contiguous()?;

    let result = t_contig.to_dtype(DType::F32)?;
    let result: Vec<Vec<f32>> = result.to_vec2()?;
    assert_eq!(
        result,
        vec![
            vec![1.0, 6.0],
            vec![2.0, 7.0],
            vec![3.0, 8.0],
            vec![4.0, 9.0],
            vec![5.0, 10.0]
        ]
    );
    Ok(())
}

// =============================================================================
// Narrow/slice tests (different non-contiguous pattern)
// =============================================================================

fn ucopy_narrow_f32(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dev,
    )?;
    let t_narrow = t.narrow(1, 1, 2)?; // Middle 2 columns
    assert!(!t_narrow.is_contiguous());
    let t_contig = t_narrow.contiguous()?;
    assert!(t_contig.is_contiguous());
    let result: Vec<Vec<f32>> = t_contig.to_vec2()?;
    assert_eq!(
        result,
        vec![vec![2.0, 3.0], vec![6.0, 7.0], vec![10.0, 11.0]]
    );
    Ok(())
}

fn ucopy_narrow_u8(dev: &Device) -> Result<()> {
    let t = Tensor::new(&[[1u8, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dev)?;
    let t_narrow = t.narrow(1, 1, 2)?;
    assert!(!t_narrow.is_contiguous());
    let t_contig = t_narrow.contiguous()?;
    assert!(t_contig.is_contiguous());
    let result: Vec<Vec<u8>> = t_contig.to_vec2()?;
    assert_eq!(result, vec![vec![2, 3], vec![6, 7], vec![10, 11]]);
    Ok(())
}

fn ucopy_narrow_i64(dev: &Device) -> Result<()> {
    let t = Tensor::new(&[[1i64, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dev)?;
    let t_narrow = t.narrow(1, 1, 2)?;
    assert!(!t_narrow.is_contiguous());
    let t_contig = t_narrow.contiguous()?;
    assert!(t_contig.is_contiguous());
    let result: Vec<Vec<i64>> = t_contig.to_vec2()?;
    assert_eq!(result, vec![vec![2, 3], vec![6, 7], vec![10, 11]]);
    Ok(())
}

// =============================================================================
// 3D tensor tests
// =============================================================================

fn ucopy_3d_transpose_f32(dev: &Device) -> Result<()> {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let t = Tensor::from_vec(data, (2, 3, 4), dev)?;
    let t_t = t.transpose(0, 2)?;
    assert!(!t_t.is_contiguous());
    let t_contig = t_t.contiguous()?;
    assert!(t_contig.is_contiguous());
    assert_eq!(t_contig.dims(), &[4, 3, 2]);

    // Verify some values
    let result: Vec<Vec<Vec<f32>>> = t_contig.to_vec3()?;
    assert_eq!(result[0][0], vec![0.0, 12.0]);
    assert_eq!(result[3][2], vec![11.0, 23.0]);
    Ok(())
}

fn ucopy_3d_transpose_u32(dev: &Device) -> Result<()> {
    let data: Vec<u32> = (0..24).collect();
    let t = Tensor::from_vec(data, (2, 3, 4), dev)?;
    let t_t = t.transpose(0, 2)?;
    assert!(!t_t.is_contiguous());
    let t_contig = t_t.contiguous()?;
    assert!(t_contig.is_contiguous());

    let result: Vec<Vec<Vec<u32>>> = t_contig.to_vec3()?;
    assert_eq!(result[0][0], vec![0, 12]);
    assert_eq!(result[3][2], vec![11, 23]);
    Ok(())
}

// =============================================================================
// F8E4M3 ucopy tests
// =============================================================================

#[cfg(feature = "cuda")]
fn ucopy_f8e4m3_transpose(dev: &Device) -> Result<()> {
    // F8E4M3 is CUDA-only and requires SM89+
    if !dev.is_cuda() {
        return Ok(());
    }
    let t = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], dev)?.to_dtype(DType::F8E4M3)?;
    let t_t = t.t()?;
    assert!(!t_t.is_contiguous());
    let t_contig = t_t.contiguous()?;
    assert!(t_contig.is_contiguous());

    let result = t_contig.to_dtype(DType::F32)?;
    let result: Vec<Vec<f32>> = result.to_vec2()?;
    assert_eq!(result, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
    Ok(())
}

#[cfg(feature = "cuda")]
fn ucopy_f8e4m3_large(dev: &Device) -> Result<()> {
    if !dev.is_cuda() {
        return Ok(());
    }
    // Use smaller values that fit well in F8E4M3 range
    let size = 10000;
    let data: Vec<f32> = (0..size).map(|x| (x % 100) as f32 * 0.1).collect();
    let t = Tensor::from_vec(data.clone(), (100, 100), dev)?.to_dtype(DType::F8E4M3)?;
    let t_t = t.t()?;
    let t_contig = t_t.contiguous()?;

    let result = t_contig.to_dtype(DType::F32)?;
    let result: Vec<Vec<f32>> = result.to_vec2()?;
    // Check structure - first row of transposed should be first column of original
    // Values may have some precision loss due to F8E4M3
    for i in 0..10 {
        let expected = ((i * 100) % 100) as f32 * 0.1;
        let actual = result[0][i];
        assert!(
            (actual - expected).abs() < 0.5,
            "mismatch at [0][{}]: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn ucopy_f8e4m3_odd_size(dev: &Device) -> Result<()> {
    if !dev.is_cuda() {
        return Ok(());
    }
    // 5 elements - tests remainder after 4-element vector
    let t = Tensor::new(
        &[[1.0f32, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]],
        dev,
    )?
    .to_dtype(DType::F8E4M3)?;
    let t_t = t.t()?;
    let t_contig = t_t.contiguous()?;

    let result = t_contig.to_dtype(DType::F32)?;
    let result: Vec<Vec<f32>> = result.to_vec2()?;
    assert_eq!(
        result,
        vec![
            vec![1.0, 6.0],
            vec![2.0, 7.0],
            vec![3.0, 8.0],
            vec![4.0, 9.0],
            vec![5.0, 10.0]
        ]
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn ucopy_f8e4m3_narrow(dev: &Device) -> Result<()> {
    if !dev.is_cuda() {
        return Ok(());
    }
    let t = Tensor::new(
        &[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dev,
    )?
    .to_dtype(DType::F8E4M3)?;
    let t_narrow = t.narrow(1, 1, 2)?;
    assert!(!t_narrow.is_contiguous());
    let t_contig = t_narrow.contiguous()?;
    assert!(t_contig.is_contiguous());

    let result = t_contig.to_dtype(DType::F32)?;
    let result: Vec<Vec<f32>> = result.to_vec2()?;
    assert_eq!(
        result,
        vec![vec![2.0, 3.0], vec![6.0, 7.0], vec![10.0, 11.0]]
    );
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

// F32 tests
test_device!(
    ucopy_f32_contiguous,
    ucopy_f32_contiguous_cpu,
    ucopy_f32_contiguous_cuda,
    ucopy_f32_contiguous_metal
);
test_device!(
    ucopy_f32_transpose,
    ucopy_f32_transpose_cpu,
    ucopy_f32_transpose_cuda,
    ucopy_f32_transpose_metal
);
test_device!(
    ucopy_f32_large_aligned,
    ucopy_f32_large_aligned_cpu,
    ucopy_f32_large_aligned_cuda,
    ucopy_f32_large_aligned_metal
);
test_device!(
    ucopy_f32_odd_size,
    ucopy_f32_odd_size_cpu,
    ucopy_f32_odd_size_cuda,
    ucopy_f32_odd_size_metal
);

// F64 tests
test_device!(
    ucopy_f64_transpose,
    ucopy_f64_transpose_cpu,
    ucopy_f64_transpose_cuda,
    ucopy_f64_transpose_metal
);
test_device!(
    ucopy_f64_large,
    ucopy_f64_large_cpu,
    ucopy_f64_large_cuda,
    ucopy_f64_large_metal
);

// U8 tests
test_device!(
    ucopy_u8_transpose,
    ucopy_u8_transpose_cpu,
    ucopy_u8_transpose_cuda,
    ucopy_u8_transpose_metal
);
test_device!(
    ucopy_u8_large_aligned,
    ucopy_u8_large_aligned_cpu,
    ucopy_u8_large_aligned_cuda,
    ucopy_u8_large_aligned_metal
);
test_device!(
    ucopy_u8_odd_size,
    ucopy_u8_odd_size_cpu,
    ucopy_u8_odd_size_cuda,
    ucopy_u8_odd_size_metal
);

// U32 tests
test_device!(
    ucopy_u32_transpose,
    ucopy_u32_transpose_cpu,
    ucopy_u32_transpose_cuda,
    ucopy_u32_transpose_metal
);
test_device!(
    ucopy_u32_large,
    ucopy_u32_large_cpu,
    ucopy_u32_large_cuda,
    ucopy_u32_large_metal
);
test_device!(
    ucopy_u32_odd_size,
    ucopy_u32_odd_size_cpu,
    ucopy_u32_odd_size_cuda,
    ucopy_u32_odd_size_metal
);

// I64 tests
test_device!(
    ucopy_i64_transpose,
    ucopy_i64_transpose_cpu,
    ucopy_i64_transpose_cuda,
    ucopy_i64_transpose_metal
);
test_device!(
    ucopy_i64_large,
    ucopy_i64_large_cpu,
    ucopy_i64_large_cuda,
    ucopy_i64_large_metal
);
test_device!(
    ucopy_i64_odd_size,
    ucopy_i64_odd_size_cpu,
    ucopy_i64_odd_size_cuda,
    ucopy_i64_odd_size_metal
);

// F16 tests
test_device!(
    ucopy_f16_transpose,
    ucopy_f16_transpose_cpu,
    ucopy_f16_transpose_cuda,
    ucopy_f16_transpose_metal
);
test_device!(
    ucopy_f16_large,
    ucopy_f16_large_cpu,
    ucopy_f16_large_cuda,
    ucopy_f16_large_metal
);
test_device!(
    ucopy_f16_odd_size,
    ucopy_f16_odd_size_cpu,
    ucopy_f16_odd_size_cuda,
    ucopy_f16_odd_size_metal
);

// BF16 tests
test_device!(
    ucopy_bf16_transpose,
    ucopy_bf16_transpose_cpu,
    ucopy_bf16_transpose_cuda,
    ucopy_bf16_transpose_metal
);
test_device!(
    ucopy_bf16_large,
    ucopy_bf16_large_cpu,
    ucopy_bf16_large_cuda,
    ucopy_bf16_large_metal
);
test_device!(
    ucopy_bf16_odd_size,
    ucopy_bf16_odd_size_cpu,
    ucopy_bf16_odd_size_cuda,
    ucopy_bf16_odd_size_metal
);

// Narrow tests
test_device!(
    ucopy_narrow_f32,
    ucopy_narrow_f32_cpu,
    ucopy_narrow_f32_cuda,
    ucopy_narrow_f32_metal
);
test_device!(
    ucopy_narrow_u8,
    ucopy_narrow_u8_cpu,
    ucopy_narrow_u8_cuda,
    ucopy_narrow_u8_metal
);
test_device!(
    ucopy_narrow_i64,
    ucopy_narrow_i64_cpu,
    ucopy_narrow_i64_cuda,
    ucopy_narrow_i64_metal
);

// 3D tests
test_device!(
    ucopy_3d_transpose_f32,
    ucopy_3d_transpose_f32_cpu,
    ucopy_3d_transpose_f32_cuda,
    ucopy_3d_transpose_f32_metal
);
test_device!(
    ucopy_3d_transpose_u32,
    ucopy_3d_transpose_u32_cpu,
    ucopy_3d_transpose_u32_cuda,
    ucopy_3d_transpose_u32_metal
);

// F8E4M3 tests (CUDA only, requires SM89+)
#[cfg(feature = "cuda")]
#[test]
#[serial_test::serial(cuda)]
fn ucopy_f8e4m3_transpose_cuda() -> Result<()> {
    ucopy_f8e4m3_transpose(&Device::new_cuda(0)?)
}

#[cfg(feature = "cuda")]
#[test]
#[serial_test::serial(cuda)]
fn ucopy_f8e4m3_large_cuda() -> Result<()> {
    ucopy_f8e4m3_large(&Device::new_cuda(0)?)
}

#[cfg(feature = "cuda")]
#[test]
#[serial_test::serial(cuda)]
fn ucopy_f8e4m3_odd_size_cuda() -> Result<()> {
    ucopy_f8e4m3_odd_size(&Device::new_cuda(0)?)
}

#[cfg(feature = "cuda")]
#[test]
#[serial_test::serial(cuda)]
fn ucopy_f8e4m3_narrow_cuda() -> Result<()> {
    ucopy_f8e4m3_narrow(&Device::new_cuda(0)?)
}
