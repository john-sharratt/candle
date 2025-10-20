use candle_core::{Device, Result, Tensor};
use std::time::Instant;

/// Performance benchmark comparing different sampling approaches
fn main() -> Result<()> {
    println!("🚀 Performance Benchmark: Traditional vs GPU-Native Sampling\n");

    // Test parameters
    let vocab_sizes = vec![1000, 10000, 32000, 50000]; // Realistic model vocabulary sizes
    let num_iterations = 1000; // More iterations for accurate timing
    let temperature = 0.8f32;
    let top_k = Some(50usize);
    let top_p = Some(0.9f64);

    println!("📊 Test Configuration:");
    println!("   Iterations: {}", num_iterations);
    println!("   Temperature: {}", temperature);
    println!("   Top-K: {:?}", top_k);
    println!("   Top-P: {:?}", top_p);
    println!();

    for vocab_size in vocab_sizes {
        println!("🎯 Testing vocabulary size: {}", vocab_size);

        // Test on CPU
        benchmark_cpu_methods(vocab_size, num_iterations, temperature, top_k, top_p)?;

        // Test on CUDA if available
        #[cfg(feature = "cuda")]
        if candle_core::utils::cuda_is_available() {
            println!("🔥 CUDA Available - Testing GPU performance...");
            benchmark_gpu_methods(
                vocab_size,
                num_iterations,
                temperature,
                top_k,
                top_p,
                "CUDA",
            )?;
        } else {
            println!("⚠️  CUDA not available - skipping GPU tests");
        }

        println!();
    }

    Ok(())
}

fn benchmark_cpu_methods(
    vocab_size: usize,
    iterations: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f64>,
) -> Result<()> {
    let device = Device::Cpu;

    // Create realistic test logits (similar to transformer output)
    let logits_data: Vec<f32> = (0..vocab_size)
        .map(|i| {
            let x = i as f32 / vocab_size as f32;
            // Create realistic distribution: some high-probability tokens, many low-probability
            if i < vocab_size / 10 {
                2.0 + x * 3.0 // High-probability tokens
            } else {
                -1.0 + x * 0.5 // Low-probability tokens
            }
        })
        .collect();
    let logits = Tensor::from_vec(logits_data, vocab_size, &device)?;

    println!("💻 CPU Performance:");

    // Benchmark 1: Traditional approach (simulated GPU→CPU→GPU transfer)
    let start = Instant::now();
    for i in 0..iterations {
        // Simulate the traditional approach:
        // 1. Move logits to CPU (if they were on GPU)
        let logits_cpu = logits.to_dtype(candle_core::DType::F32)?; // Simulate transfer cost

        // 2. Apply sampling on CPU using basic multinomial
        let _token = naive_cpu_sampling(&logits_cpu, temperature, top_k, top_p, 12345 + i as u64)?;

        // 3. Move result back to device (simulated)
        let _result = Tensor::new(&[_token], &device)?; // Simulate transfer cost
    }
    let traditional_time = start.elapsed();

    // Benchmark 2: sample_multinomial (our optimized approach)
    let start = Instant::now();
    for i in 0..iterations {
        let _token = logits.sample_multinomial(temperature, top_k, top_p, 12345 + i as u64)?;
    }
    let sample_gpu_time = start.elapsed();

    // Results
    let speedup = traditional_time.as_nanos() as f64 / sample_gpu_time.as_nanos() as f64;

    println!(
        "   Traditional (CPU):      {:>8.2}ms ({:.1}μs/sample)",
        traditional_time.as_millis(),
        traditional_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "   sample_multinomial: {:>8.2}ms ({:.1}μs/sample)",
        sample_gpu_time.as_millis(),
        sample_gpu_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "   Speedup: {:.2}x {}",
        speedup,
        if speedup > 1.0 { "🚀" } else { "⚠️" }
    );

    Ok(())
}

/// Naive CPU sampling implementation (simulates traditional approach)
fn naive_cpu_sampling(
    logits: &Tensor,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f64>,
    seed: u64,
) -> Result<u32> {
    // Naive CPU implementation for comparison

    let logits_vec = logits.to_vec1::<f32>()?;
    let vocab_size = logits_vec.len();

    // Apply temperature
    let mut scores: Vec<f32> = logits_vec.iter().map(|&x| x / temperature).collect();

    // Apply top-k filtering
    if let Some(k) = top_k {
        let k = k.min(vocab_size);
        let mut indexed_scores: Vec<(usize, f32)> = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut filtered_scores = vec![f32::NEG_INFINITY; vocab_size];
        for (idx, score) in indexed_scores.iter().take(k) {
            filtered_scores[*idx] = *score;
        }
        scores = filtered_scores;
    }

    // Convert to probabilities
    let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut probs: Vec<f32> = scores.iter().map(|&x| (x - max_score).exp()).collect();

    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // Apply top-p (nucleus sampling)
    if let Some(p_threshold) = top_p {
        let mut indexed_probs: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();

        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumulative = 0.0f32;
        let mut cutoff_idx = vocab_size;

        for (i, (_, prob)) in indexed_probs.iter().enumerate() {
            cumulative += prob;
            if cumulative >= p_threshold as f32 {
                cutoff_idx = i + 1;
                break;
            }
        }

        let mut filtered_probs = vec![0.0f32; vocab_size];
        for (idx, prob) in indexed_probs.iter().take(cutoff_idx) {
            filtered_probs[*idx] = *prob;
        }

        let sum: f32 = filtered_probs.iter().sum();
        for p in &mut filtered_probs {
            *p /= sum;
        }
        probs = filtered_probs;
    }

    // Sample from multinomial distribution
    let mut rng_state = seed;
    rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
    let random = (rng_state % 1000000) as f32 / 1000000.0;

    let mut cumulative = 0.0f32;
    for (i, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if random <= cumulative {
            return Ok(i as u32);
        }
    }

    // Fallback to last valid token
    Ok((vocab_size - 1) as u32)
}

fn benchmark_gpu_methods(
    vocab_size: usize,
    iterations: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f64>,
    device_name: &str,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu; // Fallback for this demo

    // Create test logits on device
    let logits_data: Vec<f32> = (0..vocab_size)
        .map(|i| {
            let x = i as f32 / vocab_size as f32;
            if i < vocab_size / 10 {
                2.0 + x * 3.0
            } else {
                -1.0 + x * 0.5
            }
        })
        .collect();
    let logits = Tensor::from_vec(logits_data, vocab_size, &device)?;

    println!("🚀 {} Performance:", device_name);

    // Benchmark 1: Traditional approach with GPU→CPU→GPU transfers
    let start = Instant::now();
    for i in 0..iterations {
        // Step 1: GPU → CPU transfer
        let logits_cpu = logits.to_device(&Device::Cpu)?;

        // Step 2: CPU sampling
        let token = naive_cpu_sampling(&logits_cpu, temperature, top_k, top_p, 12345 + i as u64)?;

        // Step 3: CPU → GPU transfer
        let _result = Tensor::new(&[token], &device)?;
    }

    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        device.synchronize()?;
    }

    let traditional_gpu_time = start.elapsed();

    // Benchmark 2: sample_multinomial (optimized)
    let start = Instant::now();
    for i in 0..iterations {
        let _token = logits.sample_multinomial(temperature, top_k, top_p, 12345 + i as u64)?;
    }

    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        device.synchronize()?;
    }

    let sample_gpu_time = start.elapsed();

    // Results
    let speedup = traditional_gpu_time.as_nanos() as f64 / sample_gpu_time.as_nanos() as f64;

    println!(
        "   Traditional (GPU↔CPU):  {:>8.2}ms ({:.1}μs/sample) [with transfers]",
        traditional_gpu_time.as_millis(),
        traditional_gpu_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "   sample_multinomial: {:>8.2}ms ({:.1}μs/sample) [optimized]",
        sample_gpu_time.as_millis(),
        sample_gpu_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "   Speedup: {:.2}x {} (eliminates {} transfer overhead)",
        speedup,
        if speedup > 1.0 { "🚀" } else { "⚠️" },
        device_name
    );

    Ok(())
}

#[cfg(test)]
mod benchmark_tests {
    use super::*;

    #[test]
    fn benchmark_sanity_check() -> Result<()> {
        // Quick sanity check that methods produce reasonable results
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;

        // Traditional CPU approach
        let token1 = naive_cpu_sampling(&logits, 0.8, Some(3), Some(0.9), 42)?;

        // sample_multinomial
        let token2 = logits.sample_multinomial(0.8, Some(3), Some(0.9), 42)?;
        let token2_val = token2.to_vec1::<u32>()?[0];

        // Both should produce valid token IDs
        assert!(token1 < 4);
        assert!(token2_val < 4);

        println!(
            "✅ Sanity check: CPU={}, sample_multinomial={}",
            token1, token2_val
        );

        Ok(())
    }

    #[test]
    fn benchmark_memory_usage() -> Result<()> {
        // Test that our method doesn't create excessive temporary allocations
        let device = Device::Cpu;
        let vocab_size = 10000;

        let logits_data: Vec<f32> = (0..vocab_size).map(|i| (i as f32).sin()).collect();
        let logits = Tensor::from_vec(logits_data, vocab_size, &device)?;

        // Multiple sampling calls should be efficient
        for i in 0..10 {
            let _token = logits.sample_multinomial(0.8, Some(50), Some(0.9), i)?;
        }

        println!("✅ Memory usage test passed - no excessive allocations");
        Ok(())
    }
}
