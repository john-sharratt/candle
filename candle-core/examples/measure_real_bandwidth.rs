use candle_core::{Device, DType, Result, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    
    println!("=== RTX 4090 Memory Bandwidth Test ===\n");
    println!("Expected bandwidth for RTX 4090:");
    println!("  Desktop:   ~1000 GB/s (GDDR6X, 384-bit)");
    println!("  Mobile:    ~800 GB/s (same memory, lower clocks)\n");
    
    // Test 1: Simple copy/add operations
    test_simple_bandwidth(&device)?;
    
    // Test 2: Large sequential reads
    test_sequential_read(&device)?;
    
    // Test 3: Actual model weight loading pattern
    test_weight_loading_pattern(&device)?;
    
    Ok(())
}

fn test_simple_bandwidth(device: &Device) -> Result<()> {
    println!("--- Test 1: Simple Element-wise Operations ---");
    
    let sizes = vec![
        (32 * 1024 * 1024, "128 MB"),    // 32M F32 elements = 128 MB
        (64 * 1024 * 1024, "256 MB"),    // 256 MB
        (128 * 1024 * 1024, "512 MB"),   // 512 MB
        (256 * 1024 * 1024, "1 GB"),     // 1 GB
    ];
    
    for (num_elements, label) in sizes {
        let a = Tensor::ones(num_elements, DType::F32, device)?;
        let b = Tensor::ones(num_elements, DType::F32, device)?;
        
        // Warmup
        device.synchronize()?;
        for _ in 0..5 {
            let _ = a.add(&b)?;
        }
        device.synchronize()?;
        
        // Benchmark
        let iterations = 50;
        device.synchronize()?;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _ = a.add(&b)?;
        }
        device.synchronize()?;
        
        let elapsed = start.elapsed();
        let time_per_op = elapsed.as_secs_f64() / iterations as f64;
        
        // Bandwidth calculation:
        // Element-wise add reads 2 arrays and writes 1 array = 3x data movement
        let bytes_moved = num_elements * 4 * 3; // F32 = 4 bytes, 3 arrays
        let bandwidth_gbs = (bytes_moved as f64 / 1e9) / time_per_op;
        
        println!("  {}: {:.3} ms/op, {:.1} GB/s", 
                 label, time_per_op * 1000.0, bandwidth_gbs);
    }
    
    println!();
    Ok(())
}

fn test_sequential_read(device: &Device) -> Result<()> {
    println!("--- Test 2: Large Sequential Memory Reads ---");
    
    let sizes = vec![
        (128 * 1024 * 1024, "512 MB"),
        (256 * 1024 * 1024, "1 GB"),
        (512 * 1024 * 1024, "2 GB"),
        (1024 * 1024 * 1024, "4 GB"),
    ];
    
    for (num_elements, label) in sizes {
        let data = Tensor::ones(num_elements, DType::F32, device)?;
        
        // Warmup
        device.synchronize()?;
        for _ in 0..3 {
            let _ = data.sum_all()?;
        }
        device.synchronize()?;
        
        // Benchmark - sum_all forces reading entire array
        let iterations = 20;
        device.synchronize()?;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _ = data.sum_all()?;
        }
        device.synchronize()?;
        
        let elapsed = start.elapsed();
        let time_per_op = elapsed.as_secs_f64() / iterations as f64;
        
        // Only reading data once
        let bytes_read = num_elements * 4;
        let bandwidth_gbs = (bytes_read as f64 / 1e9) / time_per_op;
        
        println!("  {}: {:.3} ms/op, {:.1} GB/s", 
                 label, time_per_op * 1000.0, bandwidth_gbs);
    }
    
    println!();
    Ok(())
}

fn test_weight_loading_pattern(device: &Device) -> Result<()> {
    println!("--- Test 3: Model Weight Loading Pattern ---");
    println!("Simulating how weights are loaded during inference\n");
    
    // Qwen-8B Q5_K layer sizes (approximate)
    let test_cases = vec![
        (4096, 4096, "Attention Q/K/V weight (64 MB)"),
        (4096, 14336, "FFN up weight (224 MB)"),
        (14336, 4096, "FFN down weight (224 MB)"),
    ];
    
    for (rows, cols, desc) in test_cases {
        // F32 for baseline measurement
        let weight = Tensor::ones((rows, cols), DType::F32, device)?;
        let input = Tensor::ones(cols, DType::F32, device)?;
        
        // Warmup
        device.synchronize()?;
        for _ in 0..5 {
            let _ = weight.matmul(&input.unsqueeze(1)?)?;
        }
        device.synchronize()?;
        
        // Benchmark
        let iterations = 100;
        device.synchronize()?;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _ = weight.matmul(&input.unsqueeze(1)?)?;
        }
        device.synchronize()?;
        
        let elapsed = start.elapsed();
        let time_per_op = elapsed.as_secs_f64() / iterations as f64;
        
        // Memory read: weight matrix + input vector
        let bytes_read = (rows * cols + cols) * 4;
        let bandwidth_gbs = (bytes_read as f64 / 1e9) / time_per_op;
        
        // FLOPs
        let flops = 2.0 * rows as f64 * cols as f64;
        let tflops = (flops / 1e12) / time_per_op;
        
        println!("  {}", desc);
        println!("    Time: {:.3} ms", time_per_op * 1000.0);
        println!("    Bandwidth: {:.1} GB/s", bandwidth_gbs);
        println!("    Compute: {:.2} TFLOPS", tflops);
        println!("    Arithmetic Intensity: {:.2} FLOPS/byte\n", 
                 flops / bytes_read as f64);
    }
    
    println!("Analysis:");
    println!("  If bandwidth < 400 GB/s: Memory bandwidth limited");
    println!("  If bandwidth 400-800 GB/s: Normal for RTX 4090");
    println!("  If compute < 5 TFLOPS: Memory-bound workload (expected for single-token)");
    println!("  If compute > 40 TFLOPS: Compute-bound workload");
    
    Ok(())
}
