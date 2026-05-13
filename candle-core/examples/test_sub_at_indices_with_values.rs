/// Example demonstrating sub_at_indices_mut_with_values for frequency-based penalties
use candle_core::{DType, Device, Result, Tensor};

fn test_cpu() -> Result<()> {
    println!("Testing CPU implementation...");
    let device = Device::Cpu;

    // Test basic usage with different values per index
    let mut logits = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0, 50.0], &device)?;
    let indices = [0u32, 2u32, 4u32];
    let values = [5.0f32, 10.0, 15.0];

    println!("Before: {:?}", logits.to_vec1::<f32>()?);
    logits.sub_at_indices_mut_with_values(&indices, &values)?;
    println!("After:  {:?}", logits.to_vec1::<f32>()?);
    assert_eq!(logits.to_vec1::<f32>()?, &[5.0, 20.0, 20.0, 40.0, 35.0]);

    // Test with repeated indices (frequency-based penalty)
    let mut logits = Tensor::ones((5,), DType::F32, &device)?.affine(100.0, 0.0)?;
    let token_history = vec![1u32, 2u32, 1u32, 2u32, 1u32]; // Token 1 appeared 3x, token 2 appeared 2x

    // Calculate frequency-based penalties
    let mut penalties = vec![0.0f32; token_history.len()];
    let mut counts = std::collections::HashMap::new();
    for (i, &token) in token_history.iter().enumerate() {
        let count = counts.entry(token).or_insert(0);
        *count += 1;
        penalties[i] = (*count as f32) * 5.0; // 5.0 penalty per occurrence
    }

    println!("\n--- Frequency-based penalty ---");
    println!("Token history: {:?}", token_history);
    println!("Penalties: {:?}", penalties);
    println!("Before: {:?}", logits.to_vec1::<f32>()?);

    logits.sub_at_indices_mut_with_values(&token_history, &penalties)?;
    let result = logits.to_vec1::<f32>()?;
    println!("After:  {:?}", result);

    // Token 1 at index 1: 100 - 5 - 10 - 15 = 70
    // Token 2 at index 2: 100 - 5 - 10 = 85
    assert_eq!(result[1], 70.0);
    assert_eq!(result[2], 85.0);

    // Test with f16
    let mut logits_f16 =
        Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], &device)?.to_dtype(DType::F16)?;
    let indices = [0u32, 3u32];
    let values = [2.0f32, 8.0];
    logits_f16.sub_at_indices_mut_with_values(&indices, &values)?;
    let result = logits_f16.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    println!("\nF16 test: {:?}", result);
    assert_eq!(result, vec![8.0, 20.0, 30.0, 32.0]);

    println!("\n✓ CPU tests passed!");
    Ok(())
}

#[cfg(feature = "cuda")]
fn test_cuda() -> Result<()> {
    println!("\nTesting CUDA implementation...");
    let device = Device::new_cuda(0)?;

    // Test basic usage
    let mut logits = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0, 50.0], &device)?;
    let indices = [0u32, 2u32, 4u32];
    let values = [5.0f32, 10.0, 15.0];

    logits.sub_at_indices_mut_with_values(&indices, &values)?;
    assert_eq!(logits.to_vec1::<f32>()?, &[5.0, 20.0, 20.0, 40.0, 35.0]);

    // Test with repeated indices
    let mut logits = Tensor::ones((5,), DType::F32, &device)?.affine(100.0, 0.0)?;
    let indices = [1u32, 1u32, 2u32];
    let values = [10.0f32, 20.0, 30.0];
    logits.sub_at_indices_mut_with_values(&indices, &values)?;
    let result = logits.to_vec1::<f32>()?;
    assert_eq!(result[1], 70.0);
    assert_eq!(result[2], 70.0);

    // Test with f16
    let mut logits_f16 =
        Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], &device)?.to_dtype(DType::F16)?;
    let indices = [0u32, 3u32];
    let values = [2.0f32, 8.0];
    logits_f16.sub_at_indices_mut_with_values(&indices, &values)?;
    let result = logits_f16.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert_eq!(result, vec![8.0, 20.0, 30.0, 32.0]);

    // Test with bf16
    let mut logits_bf16 =
        Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], &device)?.to_dtype(DType::BF16)?;
    let indices = [1u32, 2u32];
    let values = [3.0f32, 7.0];
    logits_bf16.sub_at_indices_mut_with_values(&indices, &values)?;
    let result = logits_bf16.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert_eq!(result, vec![10.0, 17.0, 23.0, 40.0]);

    // Test large tensor
    let mut logits = Tensor::ones((1000,), DType::F32, &device)?.affine(10.0, 0.0)?;
    let indices: Vec<u32> = (0..100).map(|i| i * 10).collect();
    let values: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    logits.sub_at_indices_mut_with_values(&indices, &values)?;

    let result_vec = logits.to_vec1::<f32>()?;
    for (i, &idx) in indices.iter().enumerate() {
        let expected = 10.0 - values[i];
        let actual = result_vec[idx as usize];
        assert!(
            (actual - expected).abs() < 1e-5,
            "Index {}: expected {}, got {}",
            idx,
            expected,
            actual
        );
    }
    println!("  Large tensor test passed (1000 elements, 100 updates)");

    println!("✓ CUDA tests passed!");
    Ok(())
}

fn main() -> Result<()> {
    println!("=== Testing sub_at_indices_mut_with_values ===\n");

    test_cpu()?;

    #[cfg(feature = "cuda")]
    test_cuda()?;

    #[cfg(not(feature = "cuda"))]
    println!("\nCUDA tests skipped (CUDA feature not enabled)");

    println!("\n✓ All sub_at_indices_mut_with_values tests passed!");
    Ok(())
}
