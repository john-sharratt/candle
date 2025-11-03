use candle_core::{Device, DType, Result, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    
    println!("=== GPU Keepalive Overhead Benchmark ===\n");
    
    // Test 1: Baseline - no keepalive
    println!("--- Test 1: Baseline (no keepalive) ---");
    let iterations = 1000;
    
    let start = Instant::now();
    for _ in 0..iterations {
        // Simulate inference work
        let work = Tensor::zeros((4096, 4096), DType::F32, &device)?;
        let _ = work.matmul(&work)?;
    }
    device.synchronize()?;
    let baseline_time = start.elapsed();
    let baseline_per_iter = baseline_time.as_micros() as f64 / iterations as f64;
    
    println!("Total time: {:?}", baseline_time);
    println!("Per iteration: {:.2} µs\n", baseline_per_iter);
    
    // Test 2: With lightweight ping()
    println!("--- Test 2: With keepalive.ping() ---");
    let dummy = Tensor::zeros(256, DType::F32, &device)?;
    
    let start = Instant::now();
    for i in 0..iterations {
        // Simulate inference work
        let work = Tensor::zeros((4096, 4096), DType::F32, &device)?;
        let _ = work.matmul(&work)?;
        
        // Keepalive ping
        let _ = (&dummy + (i as f64))?.sum_all()?;
    }
    device.synchronize()?;
    let ping_time = start.elapsed();
    let ping_per_iter = ping_time.as_micros() as f64 / iterations as f64;
    
    println!("Total time: {:?}", ping_time);
    println!("Per iteration: {:.2} µs", ping_per_iter);
    println!("Overhead: {:.2} µs ({:.2}%)\n", 
             ping_per_iter - baseline_per_iter,
             ((ping_per_iter - baseline_per_iter) / baseline_per_iter) * 100.0);
    
    // Test 3: With aggressive ping
    println!("--- Test 3: With keepalive.ping_aggressive() ---");
    
    let start = Instant::now();
    for _ in 0..iterations {
        // Simulate inference work
        let work = Tensor::zeros((4096, 4096), DType::F32, &device)?;
        let _ = work.matmul(&work)?;
        
        // Aggressive keepalive
        let small = Tensor::zeros((128, 128), DType::F32, &device)?;
        let _ = small.matmul(&small)?;
    }
    device.synchronize()?;
    let aggressive_time = start.elapsed();
    let aggressive_per_iter = aggressive_time.as_micros() as f64 / iterations as f64;
    
    println!("Total time: {:?}", aggressive_time);
    println!("Per iteration: {:.2} µs", aggressive_per_iter);
    println!("Overhead: {:.2} µs ({:.2}%)\n", 
             aggressive_per_iter - baseline_per_iter,
             ((aggressive_per_iter - baseline_per_iter) / baseline_per_iter) * 100.0);
    
    // Test 4: Just the keepalive operations alone (no inference)
    println!("--- Test 4: Keepalive operations only (no inference work) ---");
    
    // Ping alone
    let start = Instant::now();
    for i in 0..iterations {
        let _ = (&dummy + (i as f64))?.sum_all()?;
    }
    device.synchronize()?;
    let ping_only_time = start.elapsed();
    let ping_only_per_iter = ping_only_time.as_micros() as f64 / iterations as f64;
    
    println!("ping() alone: {:.2} µs per call", ping_only_per_iter);
    
    // Aggressive alone
    let start = Instant::now();
    for _ in 0..iterations {
        let small = Tensor::zeros((128, 128), DType::F32, &device)?;
        let _ = small.matmul(&small)?;
    }
    device.synchronize()?;
    let aggressive_only_time = start.elapsed();
    let aggressive_only_per_iter = aggressive_only_time.as_micros() as f64 / iterations as f64;
    
    println!("ping_aggressive() alone: {:.2} µs per call\n", aggressive_only_per_iter);
    
    // Summary
    println!("=== SUMMARY ===");
    println!("Baseline inference:       {:.2} µs per token", baseline_per_iter);
    println!("With ping():              {:.2} µs per token (+{:.2} µs, +{:.2}%)", 
             ping_per_iter,
             ping_per_iter - baseline_per_iter,
             ((ping_per_iter - baseline_per_iter) / baseline_per_iter) * 100.0);
    println!("With ping_aggressive():   {:.2} µs per token (+{:.2} µs, +{:.2}%)", 
             aggressive_per_iter,
             aggressive_per_iter - baseline_per_iter,
             ((aggressive_per_iter - baseline_per_iter) / baseline_per_iter) * 100.0);
    
    println!("\n💡 Recommendation:");
    if ping_per_iter - baseline_per_iter < 10.0 {
        println!("   Use ping() - overhead is negligible ({:.2} µs)", ping_per_iter - baseline_per_iter);
    } else if aggressive_per_iter - baseline_per_iter < 50.0 {
        println!("   Use ping_aggressive() if ping() doesn't keep GPU active");
    } else {
        println!("   Overhead is significant - may need optimization");
    }
    
    println!("\n📊 Context:");
    println!("   Your current inference: ~40,000 µs per token (with GPU sleeping)");
    println!("   Expected after GPU fix: ~12,000-15,000 µs per token");
    println!("   Keepalive overhead: {:.2} µs ({:.3}% of fixed performance)", 
             ping_per_iter - baseline_per_iter,
             ((ping_per_iter - baseline_per_iter) / 12000.0) * 100.0);
    
    Ok(())
}
