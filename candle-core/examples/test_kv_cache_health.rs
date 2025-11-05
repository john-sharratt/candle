use candle_core::{DType, Device, Result, Tensor};

fn main() -> Result<()> {
    println!("Testing max_abs_in_range for KV cache health monitoring...\n");

    let device = Device::Cpu;

    // Simulate a KV cache with multiple sequence positions
    // Each position has 128 values (head_dim)
    let head_dim = 128;
    let seq_len = 10;
    let total_size = seq_len * head_dim;

    // Create a tensor with normal values
    let mut data = vec![0.0f32; total_size];
    
    // Fill with small values (normal case)
    for i in 0..total_size {
        data[i] = (i as f32 % 10.0) - 5.0; // Values from -5 to 4
    }

    let kv_cache = Tensor::from_vec(data.clone(), (total_size,), &device)?;
    
    // Test 1: Check the entire cache
    println!("Test 1: Normal values in entire cache");
    let max_val = kv_cache.max_abs_in_range(0, total_size)?;
    println!("Max abs value in full cache: {:.2}", max_val);
    assert!((max_val - 5.0).abs() < 0.01);
    println!("✓ Test 1 passed\n");

    // Test 2: Check only the last layer (most recent token)
    println!("Test 2: Check only last token added");
    let last_token_start = (seq_len - 1) * head_dim;
    let last_token_end = seq_len * head_dim;
    let max_val_last = kv_cache.max_abs_in_range(last_token_start, last_token_end)?;
    println!("Max abs value in last token: {:.2}", max_val_last);
    assert!((max_val_last - 5.0).abs() < 0.01);
    println!("✓ Test 2 passed\n");

    // Test 3: Simulate an outlier in the last layer
    println!("Test 3: Detect outlier in last token");
    data[(seq_len - 1) * head_dim + 50] = 150.0; // Add outlier
    let kv_cache_outlier = Tensor::from_vec(data.clone(), (total_size,), &device)?;
    
    let max_val_outlier = kv_cache_outlier.max_abs_in_range(last_token_start, last_token_end)?;
    println!("Max abs value with outlier: {:.2}", max_val_outlier);
    assert!((max_val_outlier - 150.0).abs() < 0.01);
    
    const OUTLIER_THRESHOLD: f32 = 100.0;
    if max_val_outlier > OUTLIER_THRESHOLD {
        println!("⚠️ KV cache outlier detected! Value: {:.2}", max_val_outlier);
        println!("✓ Test 3 passed - outlier detected\n");
    }

    // Test 4: Check with different dtypes
    println!("Test 4: Different dtypes");
    
    // F16
    let kv_f16 = Tensor::from_vec(vec![1.0f32, -2.0, 3.0, -4.0, 5.0], (5,), &device)?
        .to_dtype(DType::F16)?;
    let max_f16 = kv_f16.max_abs_in_range(0, 5)?;
    println!("F16 max abs: {:.2}", max_f16);
    assert!((max_f16 - 5.0).abs() < 0.1);
    
    // F64
    let kv_f64 = Tensor::from_vec(vec![1.0f64, -2.0, 3.0, -4.0, 5.0], (5,), &device)?
        .to_dtype(DType::F64)?;
    let max_f64 = kv_f64.max_abs_in_range(0, 5)?;
    println!("F64 max abs: {:.2}", max_f64);
    assert!((max_f64 - 5.0).abs() < 0.01);
    
    println!("✓ Test 4 passed\n");

    // Test 5: Empty range
    println!("Test 5: Empty range");
    let max_empty = kv_cache.max_abs_in_range(10, 10)?;
    println!("Empty range max: {:.2}", max_empty);
    assert_eq!(max_empty, 0.0);
    println!("✓ Test 5 passed\n");

    // Test 6: Typical usage pattern - iterative checking
    println!("Test 6: Iterative KV cache health monitoring");
    println!("Simulating token generation with health checks...");
    
    let mut healthy = true;
    for token_idx in 0..seq_len {
        let token_start = token_idx * head_dim;
        let token_end = (token_idx + 1) * head_dim;
        
        let max_val = kv_cache_outlier.max_abs_in_range(token_start, token_end)?;
        
        if max_val > OUTLIER_THRESHOLD {
            println!("  Token {}: ⚠️ UNHEALTHY (max={:.2})", token_idx, max_val);
            healthy = false;
        } else {
            println!("  Token {}: ✓ healthy (max={:.2})", token_idx, max_val);
        }
    }
    
    if !healthy {
        println!("🛑 Would terminate: KV cache corruption detected");
    }
    println!("✓ Test 6 passed\n");

    println!("✓ All max_abs_in_range tests passed!");
    println!("\n📝 Usage example:");
    println!("   let start = (seq_len - 1) * head_dim;");
    println!("   let end = seq_len * head_dim;");
    println!("   let max_val = kv_cache.max_abs_in_range(start, end)?;");
    println!("   if max_val > 100.0 {{ /* handle outlier */ }}");

    Ok(())
}
