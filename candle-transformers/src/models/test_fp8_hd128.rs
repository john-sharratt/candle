// Minimal test for FP8 HD=128 paged prefill
// This module provides a minimal test to help diagnose FP8 HD=128 issues

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use candle::{DType, Device, Result, Tensor};
    use crate::models::prefill_utils::{KvCache, try_paged_prefill_batched};

    #[test]
    fn test_fp8_hd128_paged_prefill_minimal() -> Result<()> {
        println!("\n=== Testing FP8 HD=128 Paged Prefill ===\n");
        
        let device = Device::new_cuda(0)?;
        println!("Device: {:?}", device);
        
        // Test parameters matching Llama 3.2
        let b_sz = 1usize;
        let seq_len = 4usize;  // Small for debugging
        let n_head = 32usize;
        let n_kv_head = 8usize;
        let head_dim = 128usize;
        
        println!("Parameters:");
        println!("  batch_size: {}", b_sz);
        println!("  seq_len: {}", seq_len);
        println!("  n_head: {}", n_head);
        println!("  n_kv_head: {}", n_kv_head);
        println!("  head_dim: {}", head_dim);
        
        // Create Q/K/V tensors 
        // Start with F32 random, then convert to FP8
        let q = Tensor::randn(0f32, 0.1f32, (b_sz, n_head, seq_len, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let k = Tensor::randn(0f32, 0.1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let v = Tensor::randn(0f32, 0.1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        
        // Create KV cache with FP8 dtype
        let mut cache0 = KvCache::new(2, 512);
        cache0.force_dtype(DType::F8E4M3);
        let offsets = [0usize];
        let mut caches: [&mut KvCache; 1] = [&mut cache0];
        
        // Enable trace for debugging
        std::env::set_var("CANDLE_TRACE_PAGED_PREFILL", "1");
        
        println!("\nRunning BF16 Q with FP8 KV cache...");
        println!("Q dtype: {:?}", q.dtype());
        
        let result = try_paged_prefill_batched(
            &mut caches,
            &offsets,
            &q,
            &k,
            &v,
            b_sz,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
        );
        
        match result {
            Ok(out) => {
                println!("Success! Output count: {}", out.len());
                let y = &out[0];
                println!("Output shape: {:?}", y.dims());
                println!("Output dtype: {:?}", y.dtype());
                
                // Check for NaN/Inf
                let y_f32 = y.to_dtype(DType::F32)?;
                let max_abs = y_f32.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?;
                println!("Max abs value: {}", max_abs);
                
                if !max_abs.is_finite() || max_abs > 100.0 {
                    println!("WARNING: Output values look suspicious!");
                    
                    // Print some values
                    let vals = y_f32.flatten_all()?.to_vec1::<f32>()?;
                    println!("First 10 values: {:?}", &vals[..10.min(vals.len())]);
                }
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
        
        // Now test full FP8 path (FP8 Q as well)
        println!("\n=== Testing Full FP8 (FP8 Q + FP8 KV) ===\n");
        
        let q_fp8 = Tensor::randn(0f32, 0.1f32, (b_sz, n_head, seq_len, head_dim), &device)?
            .to_dtype(DType::F8E4M3)?;
        let k_fp8 = Tensor::randn(0f32, 0.1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(DType::F8E4M3)?
            .contiguous()?;
        let v_fp8 = Tensor::randn(0f32, 0.1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(DType::F8E4M3)?
            .contiguous()?;
        
        let mut cache1 = KvCache::new(2, 512);
        cache1.force_dtype(DType::F8E4M3);
        let mut caches2: [&mut KvCache; 1] = [&mut cache1];
        
        println!("Q dtype: {:?}", q_fp8.dtype());
        
        let result2 = try_paged_prefill_batched(
            &mut caches2,
            &offsets,
            &q_fp8,
            &k_fp8,
            &v_fp8,
            b_sz,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
        );
        
        match result2 {
            Ok(out) => {
                println!("Success! Output count: {}", out.len());
                let y = &out[0];
                println!("Output shape: {:?}", y.dims());
                println!("Output dtype: {:?}", y.dtype());
                
                // Check for NaN/Inf
                let y_f32 = y.to_dtype(DType::F32)?;
                let max_abs = y_f32.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?;
                println!("Max abs value: {}", max_abs);
                
                if !max_abs.is_finite() || max_abs > 100.0 {
                    println!("WARNING: Output values look suspicious!");
                    
                    // Print some values
                    let vals = y_f32.flatten_all()?.to_vec1::<f32>()?;
                    println!("First 10 values: {:?}", &vals[..10.min(vals.len())]);
                }
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
        
        Ok(())
    }
    
    #[test]
    fn test_fp8_hd64_paged_prefill_comparison() -> Result<()> {
        println!("\n=== Testing FP8 HD=64 Paged Prefill (Reference) ===\n");
        
        let device = Device::new_cuda(0)?;
        
        // Qwen2 parameters (HD=64)
        let b_sz = 1usize;
        let seq_len = 4usize;
        let n_head = 14usize;
        let n_kv_head = 2usize;
        let head_dim = 64usize;  // HD=64 works
        
        println!("Parameters (HD=64):");
        println!("  batch_size: {}", b_sz);
        println!("  seq_len: {}", seq_len);
        println!("  n_head: {}", n_head);
        println!("  n_kv_head: {}", n_kv_head);
        println!("  head_dim: {}", head_dim);
        
        let q_fp8 = Tensor::randn(0f32, 0.1f32, (b_sz, n_head, seq_len, head_dim), &device)?
            .to_dtype(DType::F8E4M3)?;
        let k_fp8 = Tensor::randn(0f32, 0.1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(DType::F8E4M3)?
            .contiguous()?;
        let v_fp8 = Tensor::randn(0f32, 0.1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(DType::F8E4M3)?
            .contiguous()?;
        
        let mut cache0 = KvCache::new(2, 512);
        cache0.force_dtype(DType::F8E4M3);
        let mut caches: [&mut KvCache; 1] = [&mut cache0];
        let offsets = [0usize];
        
        std::env::set_var("CANDLE_TRACE_PAGED_PREFILL", "1");
        
        let result = try_paged_prefill_batched(
            &mut caches,
            &offsets,
            &q_fp8,
            &k_fp8,
            &v_fp8,
            b_sz,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
        );
        
        match result {
            Ok(out) => {
                println!("Success! Output count: {}", out.len());
                let y = &out[0];
                println!("Output shape: {:?}", y.dims());
                println!("Output dtype: {:?}", y.dtype());
                
                let y_f32 = y.to_dtype(DType::F32)?;
                let max_abs = y_f32.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?;
                println!("Max abs value: {}", max_abs);
                
                if max_abs.is_finite() && max_abs < 100.0 {
                    println!("HD=64 FP8 looks good!");
                } else {
                    println!("WARNING: HD=64 FP8 also has issues!");
                }
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
        
        Ok(())
    }
}
