use candle_core::{Device, Result, Tensor};

/// Example usage of the new GPU-native multinomial sampling functionality
fn main() -> Result<()> {
    println!("🚀 GPU-Native Multinomial Sampling Example\n");

    // Create logits for token sampling (simulating language model output)
    let logits = vec![
        1.0f32, // token 0 - low probability
        3.2,    // token 1 - high probability
        0.5,    // token 2 - very low probability
        2.8,    // token 3 - high probability
        -1.0,   // token 4 - very low probability
        2.1,    // token 5 - medium probability
    ];

    let device = Device::Cpu; // In real usage, this would be Device::new_cuda(0)? or Device::new_metal(0)?
    let logits_tensor = Tensor::from_vec(logits.clone(), logits.len(), &device)?;

    println!("📊 Original logits: {:?}", logits);
    println!("🎯 Tensor device: {:?}", logits_tensor.device());
    println!("📐 Tensor shape: {:?}", logits_tensor.shape());

    // Example 1: Basic sampling with temperature
    println!("\n🌡️  Example 1: Temperature sampling");
    let token1 = logits_tensor.sample_multinomial(0.8, None, None, 42)?;
    println!(
        "   Temperature 0.8, seed 42: token {}",
        token1
    );

    let token2 = logits_tensor.sample_multinomial(0.1, None, None, 42)?;
    println!(
        "   Temperature 0.1, seed 42: token {} (more deterministic)",
        token2
    );

    // Example 2: Top-k sampling
    println!("\n🔝 Example 2: Top-k sampling");
    let token3 = logits_tensor.sample_multinomial(1.0, Some(3), None, 42)?;
    println!(
        "   Top-k=3, seed 42: token {} (from top 3 tokens only)",
        token3.to_vec1::<u32>()?[0]
    );

    // Example 3: Nucleus (Top-p) sampling
    println!("\n☢️  Example 3: Nucleus (Top-p) sampling");
    let token4 = logits_tensor.sample_multinomial(1.0, None, Some(0.9), 42)?;
    println!(
        "   Top-p=0.9, seed 42: token {} (nucleus sampling)",
        token4.to_vec1::<u32>()?[0]
    );

    // Example 4: Combined top-k + top-p + temperature
    println!("\n🎛️  Example 4: Combined sampling");
    let token5 = logits_tensor.sample_multinomial(0.8, Some(4), Some(0.95), 42)?;
    println!(
        "   Temperature=0.8, top-k=4, top-p=0.95, seed 42: token {}",
        token5.to_vec1::<u32>()?[0]
    );

    // Example 5: Reproducibility with same seed
    println!("\n🔄 Example 5: Reproducible sampling");
    let token6a = logits_tensor.sample_multinomial(0.7, Some(3), Some(0.8), 123)?;
    let token6b = logits_tensor.sample_multinomial(0.7, Some(3), Some(0.8), 123)?;
    println!(
        "   Same parameters, same seed: {} and {} (should be identical)",
        token6a.to_vec1::<u32>()?[0],
        token6b.to_vec1::<u32>()?[0]
    );

    // Example 6: Different seeds
    let token7a = logits_tensor.sample_multinomial(0.7, Some(3), Some(0.8), 456)?;
    let token7b = logits_tensor.sample_multinomial(0.7, Some(3), Some(0.8), 789)?;
    println!(
        "   Same parameters, different seeds: {} and {} (likely different)",
        token7a.to_vec1::<u32>()?[0],
        token7b.to_vec1::<u32>()?[0]
    );

    println!("\n✨ Key Benefits:");
    println!("   • Avoids GPU→CPU transfer bottleneck in LogitsProcessor::sample()");
    println!("   • Supports all major sampling strategies (temperature, top-k, top-p)");
    println!("   • Works on CPU, CUDA, and Metal devices");
    println!("   • Deterministic with seed for reproducible generation");
    println!("   • Foundation for future GPU kernel optimizations");

    println!("\n🔧 Usage in your code:");
    println!("   // Replace: logits_processor.sample(&logits)");
    println!("   // With:    logits.sample_multinomial(temperature, top_k, top_p, seed)");

    Ok(())
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_full_sampling_workflow() -> Result<()> {
        // Simulate a complete inference workflow
        let device = Device::Cpu;

        // Step 1: Create model logits (simulating transformer output)
        let vocab_size = 1000;
        let logits_data: Vec<f32> = (0..vocab_size)
            .map(|i| (i as f32).sin() * 2.0 + (i as f32 * 0.01).cos())
            .collect();
        let logits = Tensor::from_vec(logits_data, vocab_size, &device)?;

        // Step 2: Apply sampling with realistic parameters
        let temperature = 0.7;
        let top_k = Some(50);
        let top_p = Some(0.9);
        let seed = 12345;

        let sampled_token = logits.sample_multinomial(temperature, top_k, top_p, seed)?;

        // Step 3: Verify result is valid
        let token_id = sampled_token.to_vec1::<u32>()?[0];
        assert!(
            token_id < vocab_size as u32,
            "Token ID should be within vocabulary size"
        );
        assert!(
            sampled_token.device().same_device(&device),
            "Output should be on same device as input"
        );
        assert_eq!(
            sampled_token.dtype(),
            DType::U32,
            "Output should be U32 token ID"
        );
        assert_eq!(
            sampled_token.shape().dims(),
            &[1],
            "Output should be single token"
        );

        println!("✅ Full workflow test passed: sampled token {}", token_id);
        Ok(())
    }

    #[test]
    fn test_batch_sampling_simulation() -> Result<()> {
        let device = Device::Cpu;

        // Simulate sampling multiple tokens in sequence (like text generation)
        let vocab_size = 100;
        let mut generated_tokens = Vec::new();

        for step in 0..5 {
            // Create different logits for each generation step
            let logits_data: Vec<f32> = (0..vocab_size)
                .map(|i| ((i + step * 13) as f32).sin() + (step as f32 * 0.1))
                .collect();

            let logits = Tensor::from_vec(logits_data, vocab_size, &device)?;
            let token = logits.sample_multinomial(0.8, Some(10), Some(0.9), step as u64 + 42)?;
            let token_id = token.to_vec1::<u32>()?[0];

            generated_tokens.push(token_id);
        }

        println!("🔄 Generated sequence: {:?}", generated_tokens);
        assert_eq!(generated_tokens.len(), 5);

        // All tokens should be valid
        for &token_id in &generated_tokens {
            assert!(token_id < vocab_size as u32);
        }

        Ok(())
    }
}
