use candle_core::{DType, Device, Result, Tensor};

fn main() -> Result<()> {
    println!("Testing max_abs_in_range on GPU vs CPU\n");

    // Test 1: Normal values
    println!("Test 1: Normal values");
    {
        let data: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1) - 50.0).collect();
        let cpu_tensor = Tensor::from_vec(data.clone(), 1000, &Device::Cpu)?;
        
        let cpu_max = cpu_tensor.max_abs_in_range(0, 1000)?;
        println!("  CPU result: {:.2}", cpu_max);
        
        if let Ok(cuda_device) = Device::new_cuda(0) {
            let gpu_tensor = Tensor::from_vec(data, 1000, &cuda_device)?;
            let gpu_max = gpu_tensor.max_abs_in_range(0, 1000)?;
            println!("  GPU result: {:.2}", gpu_max);
            
            let diff = (cpu_max - gpu_max).abs();
            if diff < 0.01 {
                println!("  ✓ CPU and GPU match (diff: {:.6})\n", diff);
            } else {
                println!("  ✗ MISMATCH! CPU={:.2}, GPU={:.2}, diff={:.2}\n", cpu_max, gpu_max, diff);
            }
        } else {
            println!("  GPU not available, skipping GPU test\n");
        }
    }

    // Test 2: Range selection (last 100 elements only)
    println!("Test 2: Range selection (last 100 elements)");
    {
        let mut data = vec![1.0f32; 1000];
        // Put a spike in the last 100 elements
        data[950] = 99.5;
        
        let cpu_tensor = Tensor::from_vec(data.clone(), 1000, &Device::Cpu)?;
        let cpu_max = cpu_tensor.max_abs_in_range(900, 1000)?;
        println!("  CPU result (last 100): {:.2}", cpu_max);
        
        if let Ok(cuda_device) = Device::new_cuda(0) {
            let gpu_tensor = Tensor::from_vec(data, 1000, &cuda_device)?;
            let gpu_max = gpu_tensor.max_abs_in_range(900, 1000)?;
            println!("  GPU result (last 100): {:.2}", gpu_max);
            
            let diff = (cpu_max - gpu_max).abs();
            if diff < 0.01 {
                println!("  ✓ CPU and GPU match (diff: {:.6})\n", diff);
            } else {
                println!("  ✗ MISMATCH! CPU={:.2}, GPU={:.2}, diff={:.2}\n", cpu_max, gpu_max, diff);
            }
        } else {
            println!("  GPU not available, skipping GPU test\n");
        }
    }

    // Test 3: KV cache simulation - check only last token
    println!("Test 3: KV cache simulation (128 head_dim, 10 tokens)");
    {
        let head_dim = 128;
        let num_tokens = 10;
        let total_size = head_dim * num_tokens;
        
        // Simulate KV cache with normal values
        let mut data = vec![0.5f32; total_size];
        // Add an outlier in the last token position
        data[total_size - 64] = 150.0;  // Corruption!
        
        let cpu_tensor = Tensor::from_vec(data.clone(), total_size, &Device::Cpu)?;
        
        // Check only the last token (most recently added)
        let start = (num_tokens - 1) * head_dim;
        let end = num_tokens * head_dim;
        
        let cpu_max = cpu_tensor.max_abs_in_range(start, end)?;
        println!("  CPU max in last token: {:.2}", cpu_max);
        
        if let Ok(cuda_device) = Device::new_cuda(0) {
            let gpu_tensor = Tensor::from_vec(data, total_size, &cuda_device)?;
            let gpu_max = gpu_tensor.max_abs_in_range(start, end)?;
            println!("  GPU max in last token: {:.2}", gpu_max);
            
            let threshold = 100.0;
            if cpu_max > threshold {
                println!("  ⚠ CPU detected corruption: {:.2} > {:.2}", cpu_max, threshold);
            }
            if gpu_max > threshold {
                println!("  ⚠ GPU detected corruption: {:.2} > {:.2}", gpu_max, threshold);
            }
            
            let diff = (cpu_max - gpu_max).abs();
            if diff < 0.01 {
                println!("  ✓ CPU and GPU match (diff: {:.6})\n", diff);
            } else {
                println!("  ✗ MISMATCH! CPU={:.2}, GPU={:.2}, diff={:.2}\n", cpu_max, gpu_max, diff);
            }
        } else {
            println!("  GPU not available, skipping GPU test\n");
        }
    }

    // Test 4: Different data types
    println!("Test 4: Different data types");
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        println!("  Testing {:?}:", dtype);
        
        let data_f32: Vec<f32> = vec![1.0, -2.5, 3.7, -4.2, 5.1];
        let cpu_tensor = Tensor::from_vec(data_f32.clone(), 5, &Device::Cpu)?.to_dtype(dtype)?;
        let cpu_max = cpu_tensor.max_abs_in_range(0, 5)?;
        println!("    CPU result: {:.2}", cpu_max);
        
        if let Ok(cuda_device) = Device::new_cuda(0) {
            let gpu_tensor = Tensor::from_vec(data_f32, 5, &cuda_device)?.to_dtype(dtype)?;
            let gpu_max = gpu_tensor.max_abs_in_range(0, 5)?;
            println!("    GPU result: {:.2}", gpu_max);
            
            let diff = (cpu_max - gpu_max).abs();
            if diff < 0.1 {  // Allow more tolerance for fp16
                println!("    ✓ Match (diff: {:.6})", diff);
            } else {
                println!("    ✗ MISMATCH! diff={:.2}", diff);
            }
        }
    }
    println!();

    // Test 5: Large tensor performance comparison
    println!("Test 5: Large tensor (simulating 4096 tokens × 128 head_dim)");
    {
        let head_dim = 128;
        let num_tokens = 4096;
        let total_size = head_dim * num_tokens;
        
        let data: Vec<f32> = (0..total_size).map(|i| (i as f32 * 0.001).sin()).collect();
        
        let cpu_tensor = Tensor::from_vec(data.clone(), total_size, &Device::Cpu)?;
        
        let start_time = std::time::Instant::now();
        let cpu_max = cpu_tensor.max_abs_in_range(total_size - head_dim, total_size)?;
        let cpu_time = start_time.elapsed();
        
        println!("  CPU: max={:.4}, time={:.3}ms", cpu_max, cpu_time.as_secs_f64() * 1000.0);
        
        if let Ok(cuda_device) = Device::new_cuda(0) {
            let gpu_tensor = Tensor::from_vec(data, total_size, &cuda_device)?;
            
            // Warmup
            let _ = gpu_tensor.max_abs_in_range(total_size - head_dim, total_size)?;
            
            let start_time = std::time::Instant::now();
            let gpu_max = gpu_tensor.max_abs_in_range(total_size - head_dim, total_size)?;
            let gpu_time = start_time.elapsed();
            
            println!("  GPU: max={:.4}, time={:.3}ms", gpu_max, gpu_time.as_secs_f64() * 1000.0);
            println!("  Speedup: {:.1}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
            
            let diff = (cpu_max - gpu_max).abs();
            if diff < 0.01 {
                println!("  ✓ Results match (diff: {:.6})", diff);
            }
        }
    }

    println!("\nAll tests completed!");
    Ok(())
}
