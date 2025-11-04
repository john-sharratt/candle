/// Test script to verify mul_at_indices_mut works on both CPU and CUDA
use candle_core::{DType, Device, Result, Tensor};

fn test_cpu() -> Result<()> {
    println!("Testing CPU implementation...");
    let device = Device::Cpu;

    // Test basic multiplication
    let mut t = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], &device)?;
    let indices = [0u32, 2u32];
    t.mul_at_indices_mut(&indices, 2.0)?;
    let result = t.to_vec1::<f32>()?;
    println!("  Basic test: {:?}", result);
    assert_eq!(result, vec![20.0, 20.0, 60.0, 40.0]);

    // Test with repeated indices
    let mut t = Tensor::new(&[2.0f32, 2.0, 2.0, 2.0], &device)?;
    let indices = [1u32, 1u32, 2u32];
    t.mul_at_indices_mut(&indices, 3.0)?;
    let result = t.to_vec1::<f32>()?;
    println!("  Repeated indices test: {:?}", result);
    assert_eq!(result, vec![2.0, 18.0, 6.0, 2.0]);

    // Test with f16
    let mut t_f16 = Tensor::new(&[4.0f32, 8.0, 12.0], &device)?.to_dtype(DType::F16)?;
    let indices = [0u32, 2u32];
    t_f16.mul_at_indices_mut(&indices, 2.5)?;
    let result = t_f16.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    println!("  F16 test: {:?}", result);
    assert_eq!(result, vec![10.0, 8.0, 30.0]);

    println!("✓ CPU tests passed!");
    Ok(())
}

#[cfg(feature = "cuda")]
fn test_cuda() -> Result<()> {
    println!("\nTesting CUDA implementation...");
    let device = Device::new_cuda(0)?;

    // Test basic multiplication
    let mut t = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], &device)?;
    let indices = [0u32, 2u32];
    t.mul_at_indices_mut(&indices, 2.0)?;
    let result = t.to_vec1::<f32>()?;
    println!("  Basic test: {:?}", result);
    assert_eq!(result, vec![20.0, 20.0, 60.0, 40.0]);

    // Test with repeated indices
    let mut t = Tensor::new(&[2.0f32, 2.0, 2.0, 2.0], &device)?;
    let indices = [1u32, 1u32, 2u32];
    t.mul_at_indices_mut(&indices, 3.0)?;
    let result = t.to_vec1::<f32>()?;
    println!("  Repeated indices test: {:?}", result);
    assert_eq!(result, vec![2.0, 18.0, 6.0, 2.0]);

    // Test with f16
    let mut t_f16 = Tensor::new(&[4.0f32, 8.0, 12.0], &device)?.to_dtype(DType::F16)?;
    let indices = [0u32, 2u32];
    t_f16.mul_at_indices_mut(&indices, 2.5)?;
    let result = t_f16.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    println!("  F16 test: {:?}", result);
    assert_eq!(result, vec![10.0, 8.0, 30.0]);

    // Test with bf16
    let mut t_bf16 = Tensor::new(&[3.0f32, 6.0, 9.0, 12.0], &device)?.to_dtype(DType::BF16)?;
    let indices = [1u32, 2u32];
    t_bf16.mul_at_indices_mut(&indices, 2.0)?;
    let result = t_bf16.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    println!("  BF16 test: {:?}", result);
    assert_eq!(result, vec![3.0, 12.0, 18.0, 12.0]);

    // Test large tensor
    let mut t = Tensor::ones((1000,), DType::F32, &device)?.affine(2.0, 0.0)?;
    let indices: Vec<u32> = (0..500).map(|i| i * 2).collect();
    t.mul_at_indices_mut(&indices, 3.0)?;
    let result_vec = t.to_vec1::<f32>()?;
    // Even indices should be 6.0, odd indices should be 2.0
    for (i, &val) in result_vec.iter().enumerate() {
        if i % 2 == 0 {
            assert!((val - 6.0).abs() < 1e-6, "Index {}: expected 6.0, got {}", i, val);
        } else {
            assert!((val - 2.0).abs() < 1e-6, "Index {}: expected 2.0, got {}", i, val);
        }
    }
    println!("  Large tensor test passed (1000 elements)");

    println!("✓ CUDA tests passed!");
    Ok(())
}

fn main() -> Result<()> {
    test_cpu()?;

    #[cfg(feature = "cuda")]
    test_cuda()?;

    #[cfg(not(feature = "cuda"))]
    println!("\nCUDA tests skipped (CUDA feature not enabled)");

    println!("\n✓ All mul_at_indices_mut tests passed!");
    Ok(())
}
