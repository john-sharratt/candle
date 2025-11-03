use candle_core::{DType, Device, Result, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== GPU Memory Bandwidth Test ===\n");
    
    let device = Device::new_cuda(0)?;
    
    // RTX 4090 theoretical specs:
    // Memory Bandwidth: ~1008 GB/s (desktop), ~800 GB/s (laptop mobile version)
    // Compute: 82.6 TFLOPS (FP32), 165 TFLOPS (TF32)
    
    println!("Testing memory bandwidth vs compute utilization...\n");
    
    // Test 1: Pure memory bandwidth (large copy operations)
    test_memory_bandwidth(&device)?;
    
    // Test 2: Compute-intensive operations (GEMM)
    test_compute_bound(&device)?;
    
    // Test 3: Typical model layer simulation (Q5_K quantized matmul)
    test_model_layer_simulation(&device)?;
    
    Ok(())
}

fn test_memory_bandwidth(device: &Device) -> Result<()> {
    println!("--- Test 1: Pure Memory Bandwidth ---");
    
    // Create large tensors to measure raw memory throughput
    let sizes = vec![
        (1024 * 1024, "1M elements (4 MB)"),
        (16 * 1024 * 1024, "16M elements (64 MB)"),
        (64 * 1024 * 1024, "64M elements (256 MB)"),
        (256 * 1024 * 1024, "256M elements (1 GB)"),
    ];
    
    for (size, label) in sizes {
        let src = Tensor::zeros(size, DType::F32, device)?;
        
        // Warmup
        for _ in 0..5 {
            let _ = &src + 1.0;
        }
        
        // Measure copy throughput
        let iterations = 100;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _result = &src + 1.0; // Simple operation that reads and writes
        }
        
        let elapsed = start.elapsed();
        let avg_time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        
        // Calculate bandwidth (read + write = 2x data movement)
        let bytes_per_iter = size * 4 * 2; // F32 = 4 bytes, read + write
        let bandwidth_gb_s = (bytes_per_iter as f64 / 1e9) / (avg_time_ms / 1000.0);
        
        println!("{}: {:.2} ms, {:.1} GB/s", label, avg_time_ms, bandwidth_gb_s);
    }
    
    println!();
    Ok(())
}

fn test_compute_bound(device: &Device) -> Result<()> {
    println!("--- Test 2: Compute-Intensive Operations ---");
    
    // Matrix multiplication is compute-bound for large matrices
    let sizes = vec![
        (512, 512, 512, "Small GEMM (512x512x512)"),
        (2048, 2048, 2048, "Medium GEMM (2K x 2K x 2K)"),
        (4096, 4096, 4096, "Large GEMM (4K x 4K x 4K)"),
    ];
    
    for (m, n, k, label) in sizes {
        let a = Tensor::randn(0f32, 1.0, (m, k), device)?;
        let b = Tensor::randn(0f32, 1.0, (k, n), device)?;
        
        // Warmup
        for _ in 0..3 {
            let _ = a.matmul(&b)?;
        }
        
        let iterations = 20;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _result = a.matmul(&b)?;
        }
        
        let elapsed = start.elapsed();
        let avg_time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        
        // Calculate FLOPS
        let flops = 2.0 * m as f64 * n as f64 * k as f64; // 2*M*N*K for matmul
        let tflops = (flops / 1e12) / (avg_time_ms / 1000.0);
        
        // Calculate memory bandwidth
        let bytes = (m * k + k * n + m * n) * 4; // F32 = 4 bytes
        let bandwidth_gb_s = (bytes as f64 / 1e9) / (avg_time_ms / 1000.0);
        
        // Arithmetic intensity (FLOPS per byte)
        let arithmetic_intensity = flops / bytes as f64;
        
        println!("{}: {:.2} ms, {:.2} TFLOPS, {:.1} GB/s, AI={:.1}", 
                 label, avg_time_ms, tflops, bandwidth_gb_s, arithmetic_intensity);
    }
    
    println!();
    Ok(())
}

fn test_model_layer_simulation(device: &Device) -> Result<()> {
    println!("--- Test 3: Model Layer Simulation (Qwen-8B scale) ---");
    
    // Qwen-8B has 4096 hidden size, 14336 intermediate size
    // With Q5_K quantization, weights are ~5.625 bits per weight
    
    // Simulate a typical forward pass through one layer:
    // 1. Input: [1, 4096] (single token)
    // 2. Attention Q/K/V projections: 3 x (4096 x 4096) matmuls
    // 3. Attention output projection: (4096 x 4096) matmul
    // 4. FFN gate/up projections: 2 x (4096 x 14336) matmuls
    // 5. FFN down projection: (14336 x 4096) matmul
    
    let hidden_size = 4096;
    let intermediate_size = 14336;
    let batch_size = 1;
    
    println!("Simulating single token through one transformer layer:");
    println!("Hidden size: {}, Intermediate size: {}\n", hidden_size, intermediate_size);
    
    // Create activation tensor (F32)
    let input = Tensor::randn(0f32, 1.0, (batch_size, hidden_size), device)?;
    
    // Create weight matrices (F32 - in reality Q5_K but we use F32 for baseline)
    let w_qkv = Tensor::randn(0f32, 1.0, (hidden_size * 3, hidden_size), device)?;
    let w_o = Tensor::randn(0f32, 1.0, (hidden_size, hidden_size), device)?;
    let w_gate_up = Tensor::randn(0f32, 1.0, (intermediate_size * 2, hidden_size), device)?;
    let w_down = Tensor::randn(0f32, 1.0, (hidden_size, intermediate_size), device)?;
    
    // Warmup
    for _ in 0..5 {
        let qkv = input.matmul(&w_qkv.t()?)?;
        let attn_out = qkv.narrow(1, 0, hidden_size)?.matmul(&w_o.t()?)?;
        let gate_up = input.matmul(&w_gate_up.t()?)?;
        let _ffn_out = gate_up.narrow(1, 0, intermediate_size)?.matmul(&w_down.t()?)?;
    }
    
    // Measure individual operations
    let iterations = 100;
    
    // QKV projection
    let start = Instant::now();
    for _ in 0..iterations {
        let _qkv = input.matmul(&w_qkv.t()?)?;
    }
    let qkv_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    
    // Attention output
    let qkv = input.matmul(&w_qkv.t()?)?;
    let q = qkv.narrow(1, 0, hidden_size)?;
    let start = Instant::now();
    for _ in 0..iterations {
        let _attn_out = q.matmul(&w_o.t()?)?;
    }
    let attn_out_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    
    // FFN gate/up
    let start = Instant::now();
    for _ in 0..iterations {
        let _gate_up = input.matmul(&w_gate_up.t()?)?;
    }
    let gate_up_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    
    // FFN down
    let gate_up = input.matmul(&w_gate_up.t()?)?;
    let gate = gate_up.narrow(1, 0, intermediate_size)?;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ffn_out = gate.matmul(&w_down.t()?)?;
    }
    let ffn_down_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    
    let total_time = qkv_time + attn_out_time + gate_up_time + ffn_down_time;
    
    println!("Per-operation timings (F32, single token):");
    println!("  QKV projection (4096→12288):     {:.3} ms", qkv_time);
    println!("  Attention output (4096→4096):    {:.3} ms", attn_out_time);
    println!("  FFN gate/up (4096→28672):        {:.3} ms", gate_up_time);
    println!("  FFN down (14336→4096):           {:.3} ms", ffn_down_time);
    println!("  Total per layer:                 {:.3} ms", total_time);
    println!("  Estimated 28 layers:             {:.1} ms", total_time * 28.0);
    
    // Calculate theoretical performance
    let total_flops = (
        2 * batch_size * hidden_size * (hidden_size * 3) + // QKV
        2 * batch_size * hidden_size * hidden_size +       // Attn out
        2 * batch_size * hidden_size * (intermediate_size * 2) + // Gate/up
        2 * batch_size * intermediate_size * hidden_size   // Down
    ) as f64;
    
    let total_bytes = (
        batch_size * hidden_size * 4 +                    // Input
        hidden_size * (hidden_size * 3) * 4 +             // QKV weights
        hidden_size * hidden_size * 4 +                   // Attn weights
        hidden_size * (intermediate_size * 2) * 4 +       // Gate/up weights
        intermediate_size * hidden_size * 4               // Down weights
    ) as f64;
    
    let arithmetic_intensity = total_flops / total_bytes;
    let achieved_tflops = (total_flops / 1e12) / (total_time / 1000.0);
    let achieved_bandwidth = (total_bytes / 1e9) / (total_time / 1000.0);
    
    println!("\nPerformance Analysis:");
    println!("  Arithmetic Intensity: {:.2} FLOPS/byte", arithmetic_intensity);
    println!("  Achieved TFLOPS:      {:.2}", achieved_tflops);
    println!("  Achieved Bandwidth:   {:.1} GB/s", achieved_bandwidth);
    
    // Roofline analysis
    let peak_bandwidth_laptop = 800.0; // GB/s for mobile 4090
    let peak_compute = 82.6; // TFLOPS FP32
    
    let bandwidth_bound_flops = arithmetic_intensity * peak_bandwidth_laptop / 1000.0;
    let is_memory_bound = bandwidth_bound_flops < peak_compute;
    
    println!("\nRoofline Analysis (RTX 4090 Laptop):");
    println!("  Peak Bandwidth: {} GB/s", peak_bandwidth_laptop);
    println!("  Peak Compute:   {:.1} TFLOPS", peak_compute);
    println!("  Bandwidth-bound achievable: {:.2} TFLOPS", bandwidth_bound_flops);
    
    if is_memory_bound {
        println!("  ⚠️  MEMORY BANDWIDTH BOUND");
        println!("      Your operations are limited by memory bandwidth, not compute.");
        println!("      With AI={:.2}, you can achieve at most {:.2} TFLOPS.", 
                 arithmetic_intensity, bandwidth_bound_flops);
    } else {
        println!("  ✓  COMPUTE BOUND");
        println!("      Your operations are limited by compute, not memory bandwidth.");
    }
    
    println!("\nQ5_K Quantization Impact:");
    println!("  Q5_K reduces weight memory by ~5.7x (32 bits → 5.625 bits)");
    println!("  This reduces memory bandwidth requirements significantly.");
    println!("  However, dequantization adds compute overhead.");
    
    // With Q5_K, weight bytes are reduced
    let q5k_weight_bytes = total_bytes * 0.176; // 5.625/32 = 0.176x
    let q5k_bandwidth = (q5k_weight_bytes / 1e9) / (total_time / 1000.0);
    
    println!("  Q5_K weight bandwidth: {:.1} GB/s (vs {:.1} GB/s F32)", 
             q5k_bandwidth, achieved_bandwidth);
    
    Ok(())
}
