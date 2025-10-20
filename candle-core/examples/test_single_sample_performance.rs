use candle_core::{Device, Result, Tensor};
use std::time::Instant;

/// Test single sample performance (more realistic for inference)
fn main() -> Result<()> {
    println!("🎯 Single Sample Performance Test\n");

    #[cfg(feature = "cuda")]
    {
        if !candle_core::utils::cuda_is_available() {
            println!("⚠️  CUDA not available, skipping test");
            return Ok(());
        }

        let device = Device::new_cuda(0)?;
        let vocab_size = 32000;

        // Create logits on GPU
        let logits_data: Vec<f32> = (0..vocab_size).map(|i| i as f32 / vocab_size as f32).collect();
        let logits = Tensor::from_vec(logits_data, vocab_size, &device)?;

        println!("📊 Test: Single sample operation");
        println!("   Vocabulary: 32K");
        println!("   Temperature: 0.8");
        println!("   Top-K: 50");
        println!("   Top-P: 0.9");
        println!();

        // Warm-up (first call always slower)
        let _ = logits.sample_multinomial(0.8, Some(50), Some(0.9), 0)?;
        device.synchronize()?;

        // Test 1: Sample only (result stays on GPU)
        println!("🚀 Test 1: Sample only (result stays on GPU)");
        let start = Instant::now();
        let token_tensor = logits.sample_multinomial(0.8, Some(50), Some(0.9), 42)?;
        device.synchronize()?;
        let sample_only = start.elapsed();
        println!("   Time: {:.1}μs", sample_only.as_micros());
        println!();

        // Test 2: Sample + extract (includes GPU→CPU transfer)
        println!("📥 Test 2: Sample + extract to_scalar (includes GPU→CPU transfer)");
        let start = Instant::now();
        let token_tensor = logits.sample_multinomial(0.8, Some(50), Some(0.9), 42)?;
        let token_id = token_tensor.to_scalar::<u32>()?;
        device.synchronize()?;
        let sample_extract = start.elapsed();
        println!("   Token: {}", token_id);
        println!("   Time: {:.1}μs", sample_extract.as_micros());
        println!();

        // Test 3: Multiple samples (amortized cost)
        println!("🔁 Test 3: 100 samples (amortized cost)");
        let iterations = 100;
        let start = Instant::now();
        for i in 0..iterations {
            let token_tensor = logits.sample_multinomial(0.8, Some(50), Some(0.9), i)?;
            let _token_id = token_tensor.to_scalar::<u32>()?;
        }
        device.synchronize()?;
        let total = start.elapsed();
        let per_sample = total.as_micros() as f64 / iterations as f64;
        println!("   Total: {}ms", total.as_millis());
        println!("   Per sample: {:.1}μs", per_sample);
        println!();

        // Analysis
        println!("📈 Analysis:");
        println!("   Sample (GPU only):      {:.1}μs", sample_only.as_micros());
        println!("   Sample + extract:       {:.1}μs", sample_extract.as_micros());
        println!("   Transfer overhead:      {:.1}μs", sample_extract.as_micros() - sample_only.as_micros());
        println!("   Amortized per sample:   {:.1}μs", per_sample);
        println!();

        if per_sample < 1000.0 {
            println!("   ✅ SUCCESS: Amortized cost < 1ms (good for inference!)");
        } else {
            println!("   ⚠️  WARNING: Amortized cost > 1ms");
        }

        // Calculate tokens/second
        let tokens_per_sec = 1_000_000.0 / per_sample;
        println!("   Estimated throughput: {:.1} tokens/second", tokens_per_sec);
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("⚠️  This test requires CUDA support");
    }

    Ok(())
}
