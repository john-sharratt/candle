use candle_core::{Device, Result, Tensor};

/// Demonstrates how to use max_abs_in_range for efficient KV cache health monitoring
/// in LLM inference on GPU.
fn main() -> Result<()> {
    println!("KV Cache Health Monitoring - GPU-Optimized Implementation\n");
    println!("This example shows how to check only the most recently added layer");
    println!("without expensive GPU->CPU transfers of the entire cache.\n");

    // Setup: Simulate LLM KV cache parameters
    let head_dim = 128;        // Hidden dimension per attention head
    let max_seq_len = 2048;    // Maximum sequence length
    let threshold = 100.0;     // Outlier detection threshold
    
    // Use GPU if available, fallback to CPU
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Using device: {:?}", device);
    
    // Initialize KV cache on GPU (normally this would be your actual KV cache)
    let mut kv_cache_data = vec![0.0f32; max_seq_len * head_dim];
    
    // Simulate token generation loop
    println!("\nSimulating token generation with health checks:\n");
    
    for token_idx in 0..10 {
        // ========================================
        // Step 1: Add new token to KV cache
        // ========================================
        let start_idx = token_idx * head_dim;
        let end_idx = (token_idx + 1) * head_dim;
        
        // Simulate new token values (normally from transformer output)
        for i in start_idx..end_idx {
            if token_idx == 7 {
                // Inject corruption at token 7 to demonstrate detection
                kv_cache_data[i] = if i == start_idx + 50 { 150.0 } else { 0.5 };
            } else {
                kv_cache_data[i] = (i as f32 * 0.01).sin() * 2.0;
            }
        }
        
        // Transfer to GPU (in real LLM, new values are already on GPU)
        let kv_cache = Tensor::from_vec(
            kv_cache_data.clone(),
            (token_idx + 1) * head_dim,
            &device,
        )?;
        
        // ========================================
        // Step 2: Health check on MOST RECENT token only
        // ========================================
        // Key optimization: Only check the range [start_idx..end_idx]
        // This is MUCH faster than checking the entire cache!
        let max_val = kv_cache.max_abs_in_range(start_idx, end_idx)?;
        
        print!("Token {:2} | range [{:4}..{:4}] | ", 
               token_idx, start_idx, end_idx);
        
        // ========================================
        // Step 3: Decide whether to continue generation
        // ========================================
        if max_val > threshold {
            println!("max = {:6.2} | ⚠ UNHEALTHY - Terminating generation!", max_val);
            println!("\n🛑 Corruption detected at token {}!", token_idx);
            println!("   Max value {:.2} exceeds threshold {:.2}", max_val, threshold);
            println!("   In production, you would:");
            println!("     - Stop generation");
            println!("     - Return partial result");
            println!("     - Log the error for debugging");
            break;
        } else {
            println!("max = {:6.2} | ✓ healthy", max_val);
        }
    }
    
    println!("\n{}", "=".repeat(60));
    println!("Why this approach is efficient:");
    println!("{}", "=".repeat(60));
    println!("1. GPU Reduction: Max is computed entirely on GPU");
    println!("2. Minimal Transfer: Only 1 scalar (4 bytes) copied to CPU");
    println!("3. Incremental Check: Only newest layer checked, not entire history");
    println!("4. Low Latency: Sub-millisecond overhead per token");
    println!("\n{}", "=".repeat(60));
    println!("Alternative approaches (SLOWER):");
    println!("{}", "=".repeat(60));
    println!("❌ Transfer entire KV cache to CPU: {}MB per check",
             max_seq_len * head_dim * 4 / 1_000_000);
    println!("❌ Check entire cache on GPU: Wastes compute on old tokens");
    println!("✅ This approach: Check only new layer, stay on GPU");
    
    Ok(())
}
