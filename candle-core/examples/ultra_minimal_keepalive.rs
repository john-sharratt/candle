use candle_core::{Device, DType, Result, Tensor, CudaDevice};
use std::time::Instant;

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    
    println!("=== Ultra-Minimal GPU Keepalive Strategies ===\n");
    
    let iterations = 1000;
    
    // Baseline
    println!("--- Baseline (no keepalive) ---");
    let start = Instant::now();
    for _ in 0..iterations {
        let work = Tensor::zeros((4096, 4096), DType::F32, &device)?;
        let _ = work.matmul(&work)?;
    }
    device.synchronize()?;
    let baseline_time = start.elapsed();
    let baseline_us = baseline_time.as_micros() as f64 / iterations as f64;
    println!("Per iteration: {:.2} µs\n", baseline_us);
    
    // Strategy 1: Smallest possible tensor operation
    println!("--- Strategy 1: Minimal scalar operation (1 element) ---");
    let scalar = Tensor::zeros(1, DType::F32, &device)?;
    let start = Instant::now();
    for i in 0..iterations {
        let work = Tensor::zeros((4096, 4096), DType::F32, &device)?;
        let _ = work.matmul(&work)?;
        let _ = (&scalar + (i as f64))?;
    }
    device.synchronize()?;
    let strategy1_time = start.elapsed();
    let strategy1_us = strategy1_time.as_micros() as f64 / iterations as f64;
    println!("Per iteration: {:.2} µs", strategy1_us);
    println!("Overhead: {:.2} µs ({:.3}%)\n", strategy1_us - baseline_us, 
             ((strategy1_us - baseline_us) / baseline_us) * 100.0);
    
    // Strategy 2: Single element reduction
    println!("--- Strategy 2: 64 element sum (minimal reduction) ---");
    let tiny = Tensor::zeros(64, DType::F32, &device)?;
    let start = Instant::now();
    for i in 0..iterations {
        let work = Tensor::zeros((4096, 4096), DType::F32, &device)?;
        let _ = work.matmul(&work)?;
        let _ = (&tiny + (i as f64))?.sum_all()?;
    }
    device.synchronize()?;
    let strategy2_time = start.elapsed();
    let strategy2_us = strategy2_time.as_micros() as f64 / iterations as f64;
    println!("Per iteration: {:.2} µs", strategy2_us);
    println!("Overhead: {:.2} µs ({:.3}%)\n", strategy2_us - baseline_us,
             ((strategy2_us - baseline_us) / baseline_us) * 100.0);
    
    // Strategy 3: Current approach (256 elements)
    println!("--- Strategy 3: 256 element sum (current) ---");
    let small = Tensor::zeros(256, DType::F32, &device)?;
    let start = Instant::now();
    for i in 0..iterations {
        let work = Tensor::zeros((4096, 4096), DType::F32, &device)?;
        let _ = work.matmul(&work)?;
        let _ = (&small + (i as f64))?.sum_all()?;
    }
    device.synchronize()?;
    let strategy3_time = start.elapsed();
    let strategy3_us = strategy3_time.as_micros() as f64 / iterations as f64;
    println!("Per iteration: {:.2} µs", strategy3_us);
    println!("Overhead: {:.2} µs ({:.3}%)\n", strategy3_us - baseline_us,
             ((strategy3_us - baseline_us) / baseline_us) * 100.0);
    
    // Strategy 4: Reuse persistent tensor (no allocation)
    println!("--- Strategy 4: Persistent tensor, just increment ---");
    let mut persistent = Tensor::zeros(64, DType::F32, &device)?;
    let start = Instant::now();
    for _ in 0..iterations {
        let work = Tensor::zeros((4096, 4096), DType::F32, &device)?;
        let _ = work.matmul(&work)?;
        persistent = (&persistent + 1.0)?;
    }
    device.synchronize()?;
    let strategy4_time = start.elapsed();
    let strategy4_us = strategy4_time.as_micros() as f64 / iterations as f64;
    println!("Per iteration: {:.2} µs", strategy4_us);
    println!("Overhead: {:.2} µs ({:.3}%)\n", strategy4_us - baseline_us,
             ((strategy4_us - baseline_us) / baseline_us) * 100.0);
    
    // Strategy 5: Super tiny matmul (2x2)
    println!("--- Strategy 5: 2x2 matmul (minimal compute) ---");
    let tiny_mat = Tensor::zeros((2, 2), DType::F32, &device)?;
    let start = Instant::now();
    for _ in 0..iterations {
        let work = Tensor::zeros((4096, 4096), DType::F32, &device)?;
        let _ = work.matmul(&work)?;
        let _ = tiny_mat.matmul(&tiny_mat)?;
    }
    device.synchronize()?;
    let strategy5_time = start.elapsed();
    let strategy5_us = strategy5_time.as_micros() as f64 / iterations as f64;
    println!("Per iteration: {:.2} µs", strategy5_us);
    println!("Overhead: {:.2} µs ({:.3}%)\n", strategy5_us - baseline_us,
             ((strategy5_us - baseline_us) / baseline_us) * 100.0);
    
    // Strategy 6: Just synchronize (no-op)
    println!("--- Strategy 6: Device synchronize only ---");
    let start = Instant::now();
    for _ in 0..iterations {
        let work = Tensor::zeros((4096, 4096), DType::F32, &device)?;
        let _ = work.matmul(&work)?;
        device.synchronize()?;
    }
    device.synchronize()?;
    let strategy6_time = start.elapsed();
    let strategy6_us = strategy6_time.as_micros() as f64 / iterations as f64;
    println!("Per iteration: {:.2} µs", strategy6_us);
    println!("Overhead: {:.2} µs ({:.3}%)\n", strategy6_us - baseline_us,
             ((strategy6_us - baseline_us) / baseline_us) * 100.0);
    
    // Strategy 7: CUDA event recording (if available)
    #[cfg(feature = "cuda")]
    {
        println!("--- Strategy 7: CUDA stream keepalive ---");
        if let Device::Cuda(cuda_device) = &device {
            test_cuda_stream_keepalive(cuda_device, iterations, baseline_us)?;
        }
    }
    
    // Standalone tests
    println!("\n--- Standalone Operation Costs ---");
    
    // Test each in isolation
    let test_iters = 10000;
    
    let start = Instant::now();
    for i in 0..test_iters {
        let _ = (&scalar + (i as f64))?;
    }
    device.synchronize()?;
    println!("Scalar add:        {:.3} µs", start.elapsed().as_micros() as f64 / test_iters as f64);
    
    let start = Instant::now();
    for i in 0..test_iters {
        let _ = (&tiny + (i as f64))?.sum_all()?;
    }
    device.synchronize()?;
    println!("64-elem sum:       {:.3} µs", start.elapsed().as_micros() as f64 / test_iters as f64);
    
    let start = Instant::now();
    for _ in 0..test_iters {
        let _ = tiny_mat.matmul(&tiny_mat)?;
    }
    device.synchronize()?;
    println!("2x2 matmul:        {:.3} µs", start.elapsed().as_micros() as f64 / test_iters as f64);
    
    let start = Instant::now();
    for _ in 0..test_iters {
        let _ = (&persistent + 1.0)?;
    }
    device.synchronize()?;
    println!("Persistent add:    {:.3} µs", start.elapsed().as_micros() as f64 / test_iters as f64);
    
    // Summary
    println!("\n=== RESULTS ===");
    let strategies = vec![
        ("Baseline (no keepalive)", baseline_us, 0.0),
        ("Strategy 1: Scalar (1 elem)", strategy1_us, strategy1_us - baseline_us),
        ("Strategy 2: 64-elem sum", strategy2_us, strategy2_us - baseline_us),
        ("Strategy 3: 256-elem sum", strategy3_us, strategy3_us - baseline_us),
        ("Strategy 4: Persistent add", strategy4_us, strategy4_us - baseline_us),
        ("Strategy 5: 2x2 matmul", strategy5_us, strategy5_us - baseline_us),
        ("Strategy 6: Sync only", strategy6_us, strategy6_us - baseline_us),
    ];
    
    let mut best = &strategies[0];
    for strategy in &strategies[1..] {
        if strategy.2.abs() < best.2.abs() {
            best = strategy;
        }
    }
    
    println!("\nAll strategies:");
    for (name, time, overhead) in &strategies {
        let marker = if name == &best.0 { " ✅ BEST" } else { "" };
        println!("  {:30} {:8.2} µs  ({:+7.2} µs){}", name, time, overhead, marker);
    }
    
    println!("\n💡 Best Strategy: {}", best.0);
    println!("   Overhead: {:+.2} µs ({:+.3}% change)", best.2, (best.2 / baseline_us) * 100.0);
    
    Ok(())
}

#[cfg(feature = "cuda")]
fn test_cuda_stream_keepalive(cuda_device: &CudaDevice, iterations: usize, baseline_us: f64) -> Result<()> {
    let device = Device::Cuda(cuda_device.clone());
    
    // Try using a persistent tiny buffer that stays in GPU memory
    let keepalive_buffer = Tensor::zeros(32, DType::F32, &device)?;
    
    let start = Instant::now();
    for i in 0..iterations {
        let work = Tensor::zeros((4096, 4096), DType::F32, &device)?;
        let _ = work.matmul(&work)?;
        // Ultra minimal - just touch the buffer
        let _ = (&keepalive_buffer + ((i % 2) as f64))?;
    }
    device.synchronize()?;
    
    let elapsed = start.elapsed();
    let per_iter = elapsed.as_micros() as f64 / iterations as f64;
    
    println!("Per iteration: {:.2} µs", per_iter);
    println!("Overhead: {:.2} µs ({:.3}%)\n", per_iter - baseline_us,
             ((per_iter - baseline_us) / baseline_us) * 100.0);
    
    Ok(())
}
