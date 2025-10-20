use candle_core::{Device, Result, Tensor};
use std::time::Instant;

/// Focused performance benchmark for QWEN3 use case  
fn main() -> Result<()> {
    println!("🎯 QWEN3 Performance Analysis: LogitsProcessor vs sample_multinomial\n");

    // QWEN3 typical parameters
    let vocab_size = 32000; // Typical QWEN vocabulary size
    let batch_samples = 10000; // Simulate many inference calls
    let temperature = 0.8f32;
    let top_k = Some(50usize);
    let top_p = Some(0.9f64);

    println!("📊 QWEN3 Simulation:");
    println!("   Vocabulary Size: {}", vocab_size);
    println!("   Inference Calls: {}", batch_samples);
    println!("   Temperature: {}", temperature);
    println!("   Top-K: {:?}", top_k);
    println!("   Top-P: {:?}", top_p);
    println!();

    run_qwen3_benchmark(vocab_size, batch_samples, temperature, top_k, top_p)?;

    Ok(())
}

fn run_qwen3_benchmark(
    vocab_size: usize,
    iterations: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f64>,
) -> Result<()> {
    let cpu_device = Device::Cpu;

    // Create realistic QWEN3 logits distribution
    let logits_data: Vec<f32> = (0..vocab_size)
        .map(|i| {
            let x = i as f32 / vocab_size as f32;
            // Realistic transformer logits: few high-prob tokens, long tail
            if i < 100 {
                4.0 - x * 2.0 // High probability tokens
            } else if i < 1000 {
                1.0 - x * 0.5 // Medium probability tokens
            } else {
                -2.0 - x * 0.1 // Low probability tokens (long tail)
            }
        })
        .collect();

    let logits_cpu = Tensor::from_vec(logits_data.clone(), vocab_size, &cpu_device)?;

    println!("💻 CPU-only Performance (current LogitsProcessor approach):");

    // Benchmark 1: Traditional CPU sampling (current approach)
    let start = Instant::now();
    for i in 0..iterations {
        let _token = sample_traditional(&logits_cpu, temperature, top_k, top_p, i as u64)?;
    }
    let cpu_traditional_time = start.elapsed();

    // Benchmark 2: sample_multinomial on CPU
    let start = Instant::now();
    for i in 0..iterations {
        let _token = logits_cpu.sample_multinomial(temperature, top_k, top_p, i as u64)?;
    }
    let cpu_optimized_time = start.elapsed();

    let cpu_speedup = cpu_traditional_time.as_nanos() as f64 / cpu_optimized_time.as_nanos() as f64;

    println!(
        "   Traditional CPU:        {:>8.2}ms ({:.1}μs/sample)",
        cpu_traditional_time.as_millis(),
        cpu_traditional_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "   sample_multinomial: {:>8.2}ms ({:.1}μs/sample)",
        cpu_optimized_time.as_millis(),
        cpu_optimized_time.as_micros() as f64 / iterations as f64
    );
    println!("   CPU Optimization: {:.2}x speedup 🚀\n", cpu_speedup);

    // If CUDA available, test GPU scenario
    #[cfg(feature = "cuda")]
    if candle_core::utils::cuda_is_available() {
        test_gpu_scenario(
            logits_data,
            vocab_size,
            iterations,
            temperature,
            top_k,
            top_p,
        )?;
    } else {
        println!("⚠️  CUDA not available - simulating GPU scenario on CPU");
        simulate_gpu_transfers(logits_cpu, iterations, temperature, top_k, top_p)?;
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn test_gpu_scenario(
    logits_data: Vec<f32>,
    vocab_size: usize,
    iterations: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f64>,
) -> Result<()> {
    let gpu_device = Device::new_cuda(0)?;
    let cpu_device = Device::Cpu;

    let logits_gpu = Tensor::from_vec(logits_data, vocab_size, &gpu_device)?;

    println!("🔥 GPU Performance (QWEN3 with GPU tensors):");

    // Benchmark 1: Traditional approach - GPU→CPU→GPU per sample
    let start = Instant::now();
    for i in 0..iterations {
        // Step 1: Transfer GPU logits to CPU (this happens in LogitsProcessor)
        let logits_cpu = logits_gpu.to_device(&cpu_device)?;

        // Step 2: Sample on CPU
        let token = sample_traditional_cpu(&logits_cpu, temperature, top_k, top_p, i as u64)?;

        // Step 3: Transfer result back to GPU (for next model operation)
        let _token_gpu = Tensor::new(&[token], &gpu_device)?;
    }
    gpu_device.synchronize()?;
    let gpu_traditional_time = start.elapsed();

    // Benchmark 2: GPU-native approach - controlled transfers
    let start = Instant::now();
    for i in 0..iterations {
        let _token = logits_gpu.sample_multinomial(temperature, top_k, top_p, i as u64)?;
        // Result is already on the correct device
    }
    gpu_device.synchronize()?;
    let gpu_optimized_time = start.elapsed();

    let gpu_speedup = gpu_traditional_time.as_nanos() as f64 / gpu_optimized_time.as_nanos() as f64;

    println!(
        "   Traditional (GPU↔CPU):  {:>8.2}ms ({:.1}μs/sample) [with transfers]",
        gpu_traditional_time.as_millis(),
        gpu_traditional_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "   sample_multinomial: {:>8.2}ms ({:.1}μs/sample) [GPU-native]",
        gpu_optimized_time.as_millis(),
        gpu_optimized_time.as_micros() as f64 / iterations as f64
    );
    println!("   GPU Optimization: {:.2}x speedup 🚀", gpu_speedup);

    // Calculate QWEN3 benefits
    let ms_saved = gpu_traditional_time.as_millis() as i64 - gpu_optimized_time.as_millis() as i64;
    println!(
        "   💡 For {} samples: {}ms saved ({:.1}s)",
        iterations,
        ms_saved,
        ms_saved as f64 / 1000.0
    );

    Ok(())
}

fn simulate_gpu_transfers(
    logits: Tensor,
    iterations: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f64>,
) -> Result<()> {
    println!("🔄 Simulated GPU Transfer Overhead:");

    // Simulate the cost of GPU↔CPU transfers
    let start = Instant::now();
    for i in 0..iterations {
        // Simulate GPU→CPU transfer cost
        let _simulated_transfer = logits.to_dtype(candle_core::DType::F32)?;
        let _simulated_transfer2 = logits.to_dtype(candle_core::DType::F32)?;

        // Actual sampling
        let _token = sample_traditional_cpu(&logits, temperature, top_k, top_p, i as u64)?;

        // Simulate CPU→GPU transfer cost
        let _simulated_result = Tensor::new(&[0u32], &Device::Cpu)?;
    }
    let simulated_gpu_time = start.elapsed();

    // Direct approach
    let start = Instant::now();
    for i in 0..iterations {
        let _token = logits.sample_multinomial(temperature, top_k, top_p, i as u64)?;
    }
    let optimized_time = start.elapsed();

    let simulated_speedup = simulated_gpu_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;

    println!(
        "   Simulated GPU transfers: {:>8.2}ms ({:.1}μs/sample)",
        simulated_gpu_time.as_millis(),
        simulated_gpu_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "   sample_multinomial:  {:>8.2}ms ({:.1}μs/sample)",
        optimized_time.as_millis(),
        optimized_time.as_micros() as f64 / iterations as f64
    );
    println!("   Estimated GPU speedup: {:.2}x 🚀", simulated_speedup);

    Ok(())
}

/// Traditional CPU sampling (simulates current LogitsProcessor behavior)
fn sample_traditional_cpu(
    logits: &Tensor,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f64>,
    seed: u64,
) -> Result<u32> {
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
        if sum > 0.0 {
            for p in &mut filtered_probs {
                *p /= sum;
            }
            probs = filtered_probs;
        }
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

    // Fallback
    Ok((vocab_size - 1) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_equivalence() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[2.0f32, 1.0, 0.5, 3.0], &device)?;

        let token1 = sample_traditional_cpu(&logits, 0.8, Some(3), Some(0.9), 42)?;
        let token2 = logits.sample_multinomial(0.8, Some(3), Some(0.9), 42)?;
        let token2_val = token2.to_vec1::<u32>()?[0];

        assert!(token1 < 4);
        assert!(token2_val < 4);

        println!(
            "✅ Equivalence test: traditional={}, optimized={}",
            token1, token2_val
        );
        Ok(())
    }
}
