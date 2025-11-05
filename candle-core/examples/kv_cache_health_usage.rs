// KV Cache Health Monitoring Example
// 
// This shows how to use max_abs_in_range() to detect corruption in KV cache
// by checking only the most recently added layer instead of the entire cache.

use candle_core::{Result, Tensor};

// In your token generation loop:
fn check_last_token_health(kv_cache: &Tensor, seq_len: usize, head_dim: usize) -> Result<bool> {
    // Calculate range for the most recently added token
    let start = (seq_len - 1) * head_dim;
    let end = seq_len * head_dim;
    
    // Check max absolute value in just that range
    let max_val = kv_cache.max_abs_in_range(start, end)?;
    
    const OUTLIER_THRESHOLD: f32 = 100.0;
    
    if max_val > OUTLIER_THRESHOLD {
        eprintln!("⚠️ KV cache outlier detected at seq_len={}: max={:.2}", seq_len, max_val);
        return Ok(false);  // Unhealthy
    }
    
    Ok(true)  // Healthy
}

// Usage in generation loop:
fn generate_with_health_check() -> Result<()> {
    // ... your setup code ...
    
    for seq_len in 1..=max_tokens {
        // ... forward pass and cache update ...
        
        // Check health every token (or every N tokens)
        if !check_last_token_health(&kv_cache.keys, seq_len, head_dim)? {
            eprintln!("🛑 TERMINATED: KV cache corruption detected");
            break;
        }
        
        // ... continue generation ...
    }
    
    Ok(())
}

fn main() -> Result<()> {
    // This is just documentation - see test_kv_cache_health.rs for working examples
    Ok(())
}
