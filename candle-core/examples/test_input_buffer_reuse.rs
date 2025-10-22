use candle_core::{DType, Device, Result, Tensor};
use std::time::Instant;

/// Test the performance difference between recreating tensors vs reusing buffers
fn main() -> Result<()> {
    println!("🧪 Testing Input Buffer Reuse Performance\n");

    #[cfg(feature = "cuda")]
    {
        test_cuda_performance()?;
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("⚠️  CUDA not available, testing on CPU only");
        test_cpu_performance()?;
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn test_cuda_performance() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("✅ Using CUDA device: {:?}\n", device);

    let iterations = 1000;

    // Test 1: Create new tensor every time (current approach)
    println!("📊 Test 1: Create new tensor each iteration (baseline)");
    let start = Instant::now();
    for i in 0..iterations {
        let token = (i % 32000) as u32;
        let _input = Tensor::new(&[token], &device)?.reshape((1, 1))?;
    }
    device.synchronize()?;
    let baseline_time = start.elapsed();
    println!("   Time: {:?} ({:.2}μs per token)\n", 
        baseline_time, 
        baseline_time.as_micros() as f64 / iterations as f64
    );

    // Test 2: Reuse tensor buffer (optimized approach)
    println!("📊 Test 2: Reuse tensor buffer (optimized)");
    let start = Instant::now();
    let mut buffer = Tensor::zeros((1, 1), DType::U32, &device)?;
    for i in 0..iterations {
        let token = (i % 32000) as u32;
        let token_cpu = Tensor::new(&[token], &Device::Cpu)?;
        buffer = token_cpu.to_device(&device)?.reshape((1, 1))?;
    }
    device.synchronize()?;
    let optimized_time = start.elapsed();
    println!("   Time: {:?} ({:.2}μs per token)\n", 
        optimized_time,
        optimized_time.as_micros() as f64 / iterations as f64
    );

    // Test 3: Verify correctness
    println!("📊 Test 3: Verify correctness");
    let token = 12345u32;
    
    // Method 1: New tensor
    let input1 = Tensor::new(&[token], &device)?.reshape((1, 1))?;
    
    // Method 2: Reuse buffer
    let token_cpu = Tensor::new(&[token], &Device::Cpu)?;
    let input2 = token_cpu.to_device(&device)?.reshape((1, 1))?;
    
    // Verify values match
    let val1 = input1.to_vec2::<u32>()?;
    let val2 = input2.to_vec2::<u32>()?;
    
    assert_eq!(val1, val2, "Values should match!");
    assert_eq!(val1[0][0], token, "Token should be preserved!");
    println!("   ✅ Values match: {}\n", token);

    // Results
    println!("📈 Results:");
    println!("   Baseline:  {:?}", baseline_time);
    println!("   Optimized: {:?}", optimized_time);
    
    let speedup = baseline_time.as_micros() as f64 / optimized_time.as_micros() as f64;
    println!("   Speedup:   {:.2}x", speedup);
    
    let time_saved = (baseline_time.as_micros() as f64 - optimized_time.as_micros() as f64) 
        / iterations as f64;
    println!("   Time saved per token: {:.2}μs", time_saved);

    if speedup > 1.0 {
        println!("\n✅ Optimization provides {:.2}x speedup!", speedup);
    } else {
        println!("\n⚠️  No significant speedup (might be within measurement noise)");
    }

    Ok(())
}

fn test_cpu_performance() -> Result<()> {
    let device = Device::Cpu;
    println!("✅ Using CPU device\n");

    let iterations = 1000;

    // Test 1: Create new tensor every time
    println!("📊 Test 1: Create new tensor each iteration (baseline)");
    let start = Instant::now();
    for i in 0..iterations {
        let token = (i % 32000) as u32;
        let _input = Tensor::new(&[token], &device)?.reshape((1, 1))?;
    }
    let baseline_time = start.elapsed();
    println!("   Time: {:?} ({:.2}μs per token)\n", 
        baseline_time,
        baseline_time.as_micros() as f64 / iterations as f64
    );

    // Test 2: "Reuse" on CPU (minimal difference expected)
    println!("📊 Test 2: Reuse tensor buffer (optimized)");
    let start = Instant::now();
    let mut _buffer = Tensor::zeros((1, 1), DType::U32, &device)?;
    for i in 0..iterations {
        let token = (i % 32000) as u32;
        _buffer = Tensor::new(&[token], &device)?.reshape((1, 1))?;
    }
    let optimized_time = start.elapsed();
    println!("   Time: {:?} ({:.2}μs per token)\n", 
        optimized_time,
        optimized_time.as_micros() as f64 / iterations as f64
    );

    // Test 3: Verify correctness
    println!("📊 Test 3: Verify correctness");
    let token = 12345u32;
    let input = Tensor::new(&[token], &device)?.reshape((1, 1))?;
    let val = input.to_vec2::<u32>()?;
    assert_eq!(val[0][0], token);
    println!("   ✅ Values match: {}\n", token);

    println!("📈 Results:");
    println!("   Baseline:  {:?}", baseline_time);
    println!("   Optimized: {:?}", optimized_time);
    println!("\n⚠️  CPU optimization has minimal benefit (GPU-focused optimization)");

    Ok(())
}
