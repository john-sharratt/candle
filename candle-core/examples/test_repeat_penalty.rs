/// Test script to verify repeat_penalty_mut works on both CPU and CUDA
///
/// The repeat_penalty operation applies penalty differently based on logit sign:
/// - Positive logits: divided by penalty (reduces probability)
/// - Negative/zero logits: multiplied by penalty (reduces probability)
use candle_core::{DType, Device, Result, Tensor};

fn test_cpu() -> Result<()> {
    println!("Testing CPU implementation...");
    let device = Device::Cpu;

    // Test 1: Mixed positive and negative values
    println!("\n  Test 1: Mixed positive and negative values");
    let mut t = Tensor::new(&[10.0f32, -20.0, 30.0, -40.0], &device)?;
    let indices = [0u32, 1u32, 2u32, 3u32];
    t.repeat_penalty_mut(&indices, 2.0)?;
    let result = t.to_vec1::<f32>()?;
    println!("    Input:  [10.0, -20.0, 30.0, -40.0]");
    println!("    Output: {:?}", result);
    println!("    Expected: [5.0, -40.0, 15.0, -80.0]");
    assert!(
        (result[0] - 5.0).abs() < 1e-5,
        "Positive value should be divided"
    );
    assert!(
        (result[1] - (-40.0)).abs() < 1e-5,
        "Negative value should be multiplied"
    );
    assert!(
        (result[2] - 15.0).abs() < 1e-5,
        "Positive value should be divided"
    );
    assert!(
        (result[3] - (-80.0)).abs() < 1e-5,
        "Negative value should be multiplied"
    );

    // Test 2: Only positive values
    println!("\n  Test 2: Only positive values (all divided)");
    let mut t = Tensor::new(&[6.0f32, 9.0, 12.0, 15.0], &device)?;
    let indices = [0u32, 2u32];
    t.repeat_penalty_mut(&indices, 3.0)?;
    let result = t.to_vec1::<f32>()?;
    println!("    Input:  [6.0, 9.0, 12.0, 15.0]");
    println!("    Output: {:?}", result);
    println!("    Expected: [2.0, 9.0, 4.0, 15.0]");
    assert!((result[0] - 2.0).abs() < 1e-5);
    assert!((result[1] - 9.0).abs() < 1e-5);
    assert!((result[2] - 4.0).abs() < 1e-5);
    assert!((result[3] - 15.0).abs() < 1e-5);

    // Test 3: Only negative values
    println!("\n  Test 3: Only negative values (all multiplied)");
    let mut t = Tensor::new(&[-2.0f32, -4.0, -6.0, -8.0], &device)?;
    let indices = [1u32, 3u32];
    t.repeat_penalty_mut(&indices, 2.0)?;
    let result = t.to_vec1::<f32>()?;
    println!("    Input:  [-2.0, -4.0, -6.0, -8.0]");
    println!("    Output: {:?}", result);
    println!("    Expected: [-2.0, -8.0, -6.0, -16.0]");
    assert!((result[0] - (-2.0)).abs() < 1e-5);
    assert!((result[1] - (-8.0)).abs() < 1e-5);
    assert!((result[2] - (-6.0)).abs() < 1e-5);
    assert!((result[3] - (-16.0)).abs() < 1e-5);

    // Test 4: Zero values (treated as negative)
    println!("\n  Test 4: Zero values (treated as negative, multiplied)");
    let mut t = Tensor::new(&[0.0f32, 5.0, 0.0, -5.0], &device)?;
    let indices = [0u32, 2u32];
    t.repeat_penalty_mut(&indices, 2.0)?;
    let result = t.to_vec1::<f32>()?;
    println!("    Input:  [0.0, 5.0, 0.0, -5.0]");
    println!("    Output: {:?}", result);
    println!("    Expected: [0.0, 5.0, 0.0, -5.0]");
    assert!((result[0] - 0.0).abs() < 1e-5);
    assert!((result[1] - 5.0).abs() < 1e-5);
    assert!((result[2] - 0.0).abs() < 1e-5);
    assert!((result[3] - (-5.0)).abs() < 1e-5);

    // Test 5: Repeated indices (atomic operations)
    println!("\n  Test 5: Repeated indices");
    let mut t = Tensor::new(&[4.0f32, -4.0, 8.0, -8.0], &device)?;
    let indices = [0u32, 0u32, 1u32, 1u32];
    t.repeat_penalty_mut(&indices, 2.0)?;
    let result = t.to_vec1::<f32>()?;
    println!("    Input:  [4.0, -4.0, 8.0, -8.0]");
    println!("    Output: {:?}", result);
    println!("    Expected: [1.0, -16.0, 8.0, -8.0] (4.0/2/2, -4.0*2*2)");
    assert!((result[0] - 1.0).abs() < 1e-5, "Index 0 divided twice");
    assert!(
        (result[1] - (-16.0)).abs() < 1e-5,
        "Index 1 multiplied twice"
    );
    assert!((result[2] - 8.0).abs() < 1e-5);
    assert!((result[3] - (-8.0)).abs() < 1e-5);

    // Test 6: Penalty of 1.0 (no-op)
    println!("\n  Test 6: Penalty of 1.0 (no-op)");
    let mut t = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
    let indices = [0u32, 1u32, 2u32, 3u32];
    t.repeat_penalty_mut(&indices, 1.0)?;
    let result = t.to_vec1::<f32>()?;
    println!("    Input:  [1.0, 2.0, 3.0, 4.0]");
    println!("    Output: {:?}", result);
    assert_eq!(
        result,
        vec![1.0, 2.0, 3.0, 4.0],
        "Values should be unchanged"
    );

    // Test 7: F64 support
    println!("\n  Test 7: F64 data type");
    let mut t_f64 = Tensor::new(&[8.0f64, -16.0, 24.0, -32.0], &device)?;
    let indices = [0u32, 1u32];
    t_f64.repeat_penalty_mut(&indices, 2.0)?;
    let result = t_f64.to_vec1::<f64>()?;
    println!("    Input:  [8.0, -16.0, 24.0, -32.0]");
    println!("    Output: {:?}", result);
    assert!((result[0] - 4.0).abs() < 1e-9);
    assert!((result[1] - (-32.0)).abs() < 1e-9);
    assert!((result[2] - 24.0).abs() < 1e-9);
    assert!((result[3] - (-32.0)).abs() < 1e-9);

    println!("\n✓ CPU tests passed!");
    Ok(())
}

#[cfg(feature = "cuda")]
fn test_cuda() -> Result<()> {
    println!("\n\nTesting CUDA implementation...");
    let device = Device::new_cuda(0)?;

    // Test 1: Mixed positive and negative values
    println!("\n  Test 1: Mixed positive and negative values");
    let mut t = Tensor::new(&[10.0f32, -20.0, 30.0, -40.0], &device)?;
    let indices = [0u32, 1u32, 2u32, 3u32];
    t.repeat_penalty_mut(&indices, 2.0)?;
    let result = t.to_vec1::<f32>()?;
    println!("    Input:  [10.0, -20.0, 30.0, -40.0]");
    println!("    Output: {:?}", result);
    println!("    Expected: [5.0, -40.0, 15.0, -80.0]");
    assert!(
        (result[0] - 5.0).abs() < 1e-5,
        "Positive value should be divided"
    );
    assert!(
        (result[1] - (-40.0)).abs() < 1e-5,
        "Negative value should be multiplied"
    );
    assert!(
        (result[2] - 15.0).abs() < 1e-5,
        "Positive value should be divided"
    );
    assert!(
        (result[3] - (-80.0)).abs() < 1e-5,
        "Negative value should be multiplied"
    );

    // Test 2: Only positive values
    println!("\n  Test 2: Only positive values (all divided)");
    let mut t = Tensor::new(&[6.0f32, 9.0, 12.0, 15.0], &device)?;
    let indices = [0u32, 2u32];
    t.repeat_penalty_mut(&indices, 3.0)?;
    let result = t.to_vec1::<f32>()?;
    println!("    Input:  [6.0, 9.0, 12.0, 15.0]");
    println!("    Output: {:?}", result);
    println!("    Expected: [2.0, 9.0, 4.0, 15.0]");
    assert!((result[0] - 2.0).abs() < 1e-5);
    assert!((result[1] - 9.0).abs() < 1e-5);
    assert!((result[2] - 4.0).abs() < 1e-5);
    assert!((result[3] - 15.0).abs() < 1e-5);

    // Test 3: Only negative values
    println!("\n  Test 3: Only negative values (all multiplied)");
    let mut t = Tensor::new(&[-2.0f32, -4.0, -6.0, -8.0], &device)?;
    let indices = [1u32, 3u32];
    t.repeat_penalty_mut(&indices, 2.0)?;
    let result = t.to_vec1::<f32>()?;
    println!("    Input:  [-2.0, -4.0, -6.0, -8.0]");
    println!("    Output: {:?}", result);
    println!("    Expected: [-2.0, -8.0, -6.0, -16.0]");
    assert!((result[0] - (-2.0)).abs() < 1e-5);
    assert!((result[1] - (-8.0)).abs() < 1e-5);
    assert!((result[2] - (-6.0)).abs() < 1e-5);
    assert!((result[3] - (-16.0)).abs() < 1e-5);

    // Test 4: Repeated indices (atomic operations)
    println!("\n  Test 4: Repeated indices (tests atomics)");
    let mut t = Tensor::new(&[4.0f32, -4.0, 8.0, -8.0], &device)?;
    let indices = [0u32, 0u32, 1u32, 1u32];
    t.repeat_penalty_mut(&indices, 2.0)?;
    let result = t.to_vec1::<f32>()?;
    println!("    Input:  [4.0, -4.0, 8.0, -8.0]");
    println!("    Output: {:?}", result);
    println!("    Expected: [1.0, -16.0, 8.0, -8.0] (4.0/2/2, -4.0*2*2)");
    assert!((result[0] - 1.0).abs() < 1e-5, "Index 0 divided twice");
    assert!(
        (result[1] - (-16.0)).abs() < 1e-5,
        "Index 1 multiplied twice"
    );

    // Test 5: F16 support
    println!("\n  Test 5: F16 data type");
    let mut t_f16 = Tensor::new(&[8.0f32, -16.0, 24.0, -32.0], &device)?.to_dtype(DType::F16)?;
    let indices = [0u32, 1u32];
    t_f16.repeat_penalty_mut(&indices, 2.0)?;
    let result = t_f16.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    println!("    Input:  [8.0, -16.0, 24.0, -32.0]");
    println!("    Output: {:?}", result);
    assert!((result[0] - 4.0).abs() < 0.1, "F16 precision");
    assert!((result[1] - (-32.0)).abs() < 0.1, "F16 precision");

    // Test 6: BF16 support
    println!("\n  Test 6: BF16 data type");
    let mut t_bf16 = Tensor::new(&[12.0f32, -24.0, 36.0, -48.0], &device)?.to_dtype(DType::BF16)?;
    let indices = [0u32, 1u32];
    t_bf16.repeat_penalty_mut(&indices, 2.0)?;
    let result = t_bf16.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    println!("    Input:  [12.0, -24.0, 36.0, -48.0]");
    println!("    Output: {:?}", result);
    assert!((result[0] - 6.0).abs() < 0.1, "BF16 precision");
    assert!((result[1] - (-48.0)).abs() < 0.1, "BF16 precision");

    // Test 7: Large tensor (simulates real logits)
    println!("\n  Test 7: Large tensor (50K vocab, 100 penalized tokens)");
    let vocab_size = 50000;
    let num_penalized = 100;

    // Create logits: half positive (0..25000), half negative (25000..50000)
    let mut logits = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size {
        if i < vocab_size / 2 {
            logits.push((i as f32) * 0.01); // Positive values 0..250
        } else {
            logits.push(-((i - vocab_size / 2) as f32) * 0.01); // Negative values -250..0
        }
    }

    let mut t = Tensor::from_vec(logits.clone(), (vocab_size,), &device)?;

    // Penalize some tokens from both positive and negative ranges
    let indices: Vec<u32> = (0..num_penalized)
        .map(|i| {
            if i % 2 == 0 {
                i * 100 // Positive range
            } else {
                25000 + i * 100 // Negative range
            }
        })
        .collect();

    t.repeat_penalty_mut(&indices, 1.1)?;
    let result = t.to_vec1::<f32>()?;

    // Verify a few specific indices
    let idx0 = 0usize; // Positive
    let expected0 = logits[idx0] / 1.1;
    assert!(
        (result[idx0] - expected0).abs() < 1e-4,
        "Positive logit at {}: expected {}, got {}",
        idx0,
        expected0,
        result[idx0]
    );

    let idx1 = 25100usize; // Negative
    let expected1 = logits[idx1] * 1.1;
    assert!(
        (result[idx1] - expected1).abs() < 1e-4,
        "Negative logit at {}: expected {}, got {}",
        idx1,
        expected1,
        result[idx1]
    );

    // Verify unpenalized tokens are unchanged
    let idx_unchanged = 12345usize;
    assert!(
        (result[idx_unchanged] - logits[idx_unchanged]).abs() < 1e-6,
        "Unpenalized token should be unchanged"
    );

    println!(
        "    ✓ Large tensor test passed ({} elements, {} penalized)",
        vocab_size, num_penalized
    );

    // Test 8: Performance comparison
    println!("\n  Test 8: Performance comparison (CPU approach vs GPU kernel)");
    let vocab_size = 50000;
    let num_repeated = 150;

    let mut logits1 = Tensor::randn(0f32, 10f32, (vocab_size,), &device)?;
    let mut logits2 = logits1.clone();

    let indices: Vec<u32> = (0..num_repeated).map(|i| i * 100).collect();

    // Method 1: Old approach (read to CPU, separate, two kernels)
    let start = std::time::Instant::now();
    let logit_values = logits1.to_vec1::<f32>()?;
    let mut positive_tokens = Vec::new();
    let mut negative_tokens = Vec::new();
    for &token_id in &indices {
        let idx = token_id as usize;
        if logit_values[idx] > 0.0 {
            positive_tokens.push(token_id);
        } else {
            negative_tokens.push(token_id);
        }
    }
    if !positive_tokens.is_empty() {
        logits1.div_at_indices_mut(&positive_tokens, 1.1)?;
    }
    if !negative_tokens.is_empty() {
        logits1.mul_at_indices_mut(&negative_tokens, 1.1)?;
    }
    let time_old = start.elapsed();

    // Method 2: New approach (single kernel)
    let start = std::time::Instant::now();
    logits2.repeat_penalty_mut(&indices, 1.1)?;
    let time_new = start.elapsed();

    println!("    Old approach (CPU + 2 kernels): {:?}", time_old);
    println!("    New approach (1 kernel):         {:?}", time_new);
    println!(
        "    Speedup: {:.2}x",
        time_old.as_secs_f64() / time_new.as_secs_f64()
    );

    // Verify results match
    let result1 = logits1.to_vec1::<f32>()?;
    let result2 = logits2.to_vec1::<f32>()?;
    for i in 0..vocab_size {
        let diff = (result1[i] - result2[i]).abs();
        assert!(
            diff < 1e-4,
            "Results differ at index {}: {} vs {}",
            i,
            result1[i],
            result2[i]
        );
    }
    println!("    ✓ Results match exactly");

    println!("\n✓ CUDA tests passed!");
    Ok(())
}

fn main() -> Result<()> {
    println!("{}", "=".repeat(70));
    println!("Testing repeat_penalty_mut operation");
    println!("{}", "=".repeat(70));

    test_cpu()?;

    #[cfg(feature = "cuda")]
    test_cuda()?;

    #[cfg(not(feature = "cuda"))]
    println!("\n\nCUDA tests skipped (CUDA feature not enabled)");

    println!("\n{}", "=".repeat(70));
    println!("✓ All repeat_penalty_mut tests passed!");
    println!("{}", "=".repeat(70));
    Ok(())
}
