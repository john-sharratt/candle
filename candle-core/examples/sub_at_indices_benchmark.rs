/// Benchmark comparing sub_at_indices() vs sub_at_indices_mut() performance
///
/// Run with: cargo run --example sub_at_indices_benchmark --features cuda --release
use candle_core::{DType, Device, Result, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    println!("sub_at_indices Performance Benchmark");
    println!("=====================================\n");

    // Simulate LLM vocab size (e.g., LLaMA: 32000, GPT-2: 50257, larger models: 150K+)
    let vocab_sizes = vec![32000, 50257, 151936];
    let batch_size = 4;
    let num_penalty_tokens = 50; // Typical repetition penalty scenario

    for &vocab_size in &vocab_sizes {
        println!("Vocab size: {}", vocab_size);
        println!("{}", "-".repeat(40));

        // Test on CPU
        println!("\nCPU:");
        benchmark_device(&Device::Cpu, batch_size, vocab_size, num_penalty_tokens)?;

        // Test on CUDA if available
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                println!("\nCUDA:");
                benchmark_device(&device, batch_size, vocab_size, num_penalty_tokens)?;
            }
        }

        println!();
    }

    Ok(())
}

fn benchmark_device(
    device: &Device,
    batch_size: usize,
    vocab_size: usize,
    num_penalty_tokens: usize,
) -> Result<()> {
    // Create logits tensor (batch x vocab)
    let logits = Tensor::zeros((batch_size, vocab_size), DType::F32, device)?;

    // Generate random penalty tokens
    let penalty_tokens: Vec<u32> = (0..num_penalty_tokens)
        .map(|i| ((i * 173) % vocab_size) as u32) // Pseudo-random
        .collect();

    let penalty_value = 5.0;

    // Warmup
    for _ in 0..3 {
        let _ = logits.sub_at_indices(&penalty_tokens, penalty_value)?;
    }

    // Benchmark immutable API (with clone)
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = logits.sub_at_indices(&penalty_tokens, penalty_value)?;
    }
    let immutable_time = start.elapsed() / iterations;

    // Benchmark mutable API (in-place)
    let mut mutable_logits = logits.clone();
    let start = Instant::now();
    for _ in 0..iterations {
        mutable_logits.sub_at_indices_mut(&penalty_tokens, penalty_value)?;
    }
    let mutable_time = start.elapsed() / iterations;

    // Calculate speedup
    let speedup = immutable_time.as_secs_f64() / mutable_time.as_secs_f64();

    println!(
        "  sub_at_indices():     {:>8.2?} (clone + kernel)",
        immutable_time
    );
    println!(
        "  sub_at_indices_mut(): {:>8.2?} (kernel only)",
        mutable_time
    );
    println!("  Speedup:              {:>8.1}x", speedup);
    println!(
        "  Tensor size:          {:>8.1} KB",
        (batch_size * vocab_size * 4) as f64 / 1024.0
    );

    Ok(())
}
