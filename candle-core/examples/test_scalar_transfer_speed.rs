use candle_core::{Device, Result, Tensor};
use std::time::Instant;

/// Test that .to_scalar() is fast and doesn't transfer the entire tensor
fn main() -> Result<()> {
    println!("🧪 Testing Scalar Transfer Optimization\n");

    #[cfg(feature = "cuda")]
    {
        if !candle_core::utils::cuda_is_available() {
            println!("⚠️  CUDA not available, skipping test");
            return Ok(());
        }

        let device = Device::new_cuda(0)?;
        let vocab_size = 32000;

        // Create a large tensor on GPU (32K floats = 128KB)
        let logits_data: Vec<f32> = (0..vocab_size).map(|i| i as f32).collect();
        let logits = Tensor::from_vec(logits_data, vocab_size, &device)?;

        println!("📊 Test Setup:");
        println!("   Tensor size: {} elements ({} KB)", vocab_size, vocab_size * 4 / 1024);
        println!("   Device: GPU (CUDA)");
        println!();

        // Test 1: Full tensor transfer (baseline - should be slow)
        println!("🔄 Test 1: Full tensor transfer (baseline)");
        let iterations = 1000;
        let start = Instant::now();
        for _ in 0..iterations {
            let _vec = logits.to_vec1::<f32>()?;
        }
        let full_time = start.elapsed();
        println!("   Time: {}ms for {} iterations", full_time.as_millis(), iterations);
        println!("   Per call: {:.1}μs", full_time.as_micros() as f64 / iterations as f64);
        println!();

        // Test 2: Scalar transfer from last element (should be fast with optimization)
        println!("🚀 Test 2: Scalar transfer (optimized path)");
        
        // Create a rank-0 tensor (scalar) on GPU
        let scalar_tensor = Tensor::new(&[42u32], &device)?.reshape(())?;
        
        let start = Instant::now();
        for _ in 0..iterations {
            let _val = scalar_tensor.to_scalar::<u32>()?;
        }
        let scalar_time = start.elapsed();
        println!("   Time: {}ms for {} iterations", scalar_time.as_millis(), iterations);
        println!("   Per call: {:.1}μs", scalar_time.as_micros() as f64 / iterations as f64);
        println!();

        // Test 3: Sample and extract token (real-world use case)
        println!("🎯 Test 3: Sample + scalar extract (real-world)");
        let start = Instant::now();
        for i in 0..iterations {
            let token = logits.sample_multinomial(0.8, Some(50), Some(0.9), i as u64)?;
            let _token_id = token.to_scalar::<u32>()?;
        }
        let sample_time = start.elapsed();
        println!("   Time: {}ms for {} iterations", sample_time.as_millis(), iterations);
        println!("   Per call: {:.1}μs", sample_time.as_micros() as f64 / iterations as f64);
        println!();

        // Analysis
        println!("📈 Performance Analysis:");
        let full_us = full_time.as_micros() as f64 / iterations as f64;
        let scalar_us = scalar_time.as_micros() as f64 / iterations as f64;
        let sample_us = sample_time.as_micros() as f64 / iterations as f64;
        
        println!("   Full transfer:      {:.1}μs", full_us);
        println!("   Scalar transfer:    {:.1}μs", scalar_us);
        println!("   Sample + extract:   {:.1}μs", sample_us);
        println!();
        
        let speedup = full_us / scalar_us;
        println!("   Scalar speedup: {:.1}x faster than full transfer ✅", speedup);
        
        if scalar_us < 100.0 {
            println!("   ✅ SUCCESS: Scalar transfer is < 100μs (optimized path working!)");
        } else {
            println!("   ⚠️  WARNING: Scalar transfer is {:.1}μs (expected < 100μs)", scalar_us);
            println!("   This suggests the optimization may not be working correctly.");
        }
        
        if sample_us < 1000.0 {
            println!("   ✅ SUCCESS: Sample + extract is < 1ms (fast enough for real-time inference!)");
        } else {
            println!("   ⚠️  Sample + extract is {:.1}ms (may be too slow for real-time)", sample_us / 1000.0);
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("⚠️  This test requires CUDA support");
        println!("   Rebuild with: cargo run --release --example test_scalar_transfer_speed --features cuda");
    }

    Ok(())
}
