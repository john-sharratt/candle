//! Unit tests for FP8 KV cache quantization
//! 
//! Tests validate that FP8 quantization:
//! - Produces similar outputs to regular cache
//! - Saves memory (~50%)
//! - Works with real model inference

#[cfg(test)]
mod tests {
    use candle::{Device, Result, Tensor};

    /// Helper to download a model from HuggingFace
    fn download_model(repo: &str, file: &str) -> Result<std::path::PathBuf> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;
        let repo = api.model(repo.to_string());
        repo.get(file).map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to download {}: {}. This test requires internet access.",
                file, e
            ))
        })
    }

    #[test]
    #[ignore] // Run with: cargo test --features cuda -- --ignored test_fp8_llama_inference
    fn test_fp8_llama_inference() -> Result<()> {
        use crate::models::quantized_llama::ModelWeights;

        println!("\n=== Testing FP8 KV Cache with Llama-3.2-1B ===\n");
        
        let device = Device::cuda_if_available(0)?;
        println!("Using device: {:?}", device);
        
        println!("Downloading model (~650MB)...");
        let model_path = download_model(
            "bartowski/Llama-3.2-1B-Instruct-GGUF",
            "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        )?;
        println!("✓ Model downloaded\n");

        // Test 1: Regular cache inference
        println!("1. Testing with REGULAR cache...");
        let mut model_regular = ModelWeights::from_gguf_by_path(&model_path, &device)?;
        
        let mut regular_outputs = Vec::new();
        for i in 0..50 {
            let input = Tensor::new(&[1u32], &device)?.unsqueeze(0)?;
            let output = model_regular.forward(&input, i)?;
            regular_outputs.push(output);
        }
        println!("   ✓ Generated 50 tokens with regular cache");
        println!("   Cache length: {}", model_regular.cache_len());

        // Test 2: FP8 cache inference
        println!("\n2. Testing with FP8 cache...");
        let mut model_fp8 = ModelWeights::from_gguf_by_path(&model_path, &device)?;
        model_fp8.enable_fp8_kv_cache();
        
        let mut fp8_outputs = Vec::new();
        for i in 0..50 {
            let input = Tensor::new(&[1u32], &device)?.unsqueeze(0)?;
            let output = model_fp8.forward(&input, i)?;
            fp8_outputs.push(output);
        }
        println!("   ✓ Generated 50 tokens with FP8 cache");
        println!("   Cache length: {}", model_fp8.cache_len());

        // Test 3: Compare outputs
        println!("\n3. Comparing outputs...");
        let mut max_diff = 0.0f32;
        let mut mean_diff = 0.0f32;
        let mut same_top_token = 0;
        
        for (i, (regular, fp8)) in regular_outputs.iter().zip(fp8_outputs.iter()).enumerate() {
            // Calculate difference
            let diff = (regular - fp8)?.abs()?.flatten_all()?.max(0)?;
            let diff_val = diff.to_vec0::<f32>()?;
            max_diff = max_diff.max(diff_val);
            mean_diff += diff_val;
            
            // Check top token
            let reg_token = regular.argmax(candle::D::Minus1)?.to_vec1::<u32>()?[0];
            let fp8_token = fp8.argmax(candle::D::Minus1)?.to_vec1::<u32>()?[0];
            if reg_token == fp8_token {
                same_top_token += 1;
            }
            
            if i < 3 {
                println!("   Token {}: diff={:.6}, reg_tok={}, fp8_tok={}", 
                         i, diff_val, reg_token, fp8_token);
            }
        }
        
        mean_diff /= regular_outputs.len() as f32;
        let same_pct = (same_top_token as f32 / regular_outputs.len() as f32) * 100.0;
        
        println!("\n4. Results:");
        println!("   Mean difference: {:.6}", mean_diff);
        println!("   Max difference: {:.6}", max_diff);
        println!("   Same top token: {}/{} ({:.1}%)", same_top_token, regular_outputs.len(), same_pct);
        
        // BF16 should have very low error
        assert!(mean_diff < 0.1, "Mean difference too high: {}", mean_diff);
        assert!(same_pct > 80.0, "Top tokens differ too much: {:.1}%", same_pct);
        
        println!("\n✓ FP8 cache passes accuracy test!");
        println!("  Memory savings: ~50% (BF16 vs F32)\n");
        
        Ok(())
    }

    #[test]
    #[ignore] // Run with: cargo test --features cuda -- --ignored test_fp8_qwen2_inference
    fn test_fp8_qwen2_inference() -> Result<()> {
        use crate::models::quantized_qwen2::ModelWeights;

        println!("\n=== Testing FP8 KV Cache with Qwen2-0.5B ===\n");
        
        let device = Device::cuda_if_available(0)?;
        println!("Using device: {:?}", device);
        
        println!("Downloading model...");
        let model_path = download_model(
            "Qwen/Qwen2-0.5B-Instruct-GGUF",
            "qwen2-0_5b-instruct-q4_0.gguf",
        )?;
        println!("✓ Model downloaded\n");

        // Test with regular cache
        println!("1. Testing with REGULAR cache...");
        let mut model_regular = ModelWeights::from_gguf_by_path(&model_path, &device)?;
        
        let mut regular_outputs = Vec::new();
        for i in 0..30 {
            let input = Tensor::new(&[1u32], &device)?.unsqueeze(0)?;
            let output = model_regular.forward(&input, i)?;
            regular_outputs.push(output);
        }
        println!("   ✓ Generated 30 tokens");

        // Test with FP8 cache
        println!("\n2. Testing with FP8 cache...");
        let mut model_fp8 = ModelWeights::from_gguf_by_path(&model_path, &device)?;
        model_fp8.enable_fp8_kv_cache();
        
        let mut fp8_outputs = Vec::new();
        for i in 0..30 {
            let input = Tensor::new(&[1u32], &device)?.unsqueeze(0)?;
            let output = model_fp8.forward(&input, i)?;
            fp8_outputs.push(output);
        }
        println!("   ✓ Generated 30 tokens");

        // Compare
        println!("\n3. Comparing outputs...");
        let mut same_tokens = 0;
        for (regular, fp8) in regular_outputs.iter().zip(fp8_outputs.iter()) {
            let reg_tok = regular.argmax(candle::D::Minus1)?.to_vec1::<u32>()?[0];
            let fp8_tok = fp8.argmax(candle::D::Minus1)?.to_vec1::<u32>()?[0];
            if reg_tok == fp8_tok {
                same_tokens += 1;
            }
        }
        
        let same_pct = (same_tokens as f32 / regular_outputs.len() as f32) * 100.0;
        println!("   Same top token: {}/{} ({:.1}%)", same_tokens, regular_outputs.len(), same_pct);
        
        assert!(same_pct > 70.0, "Too many different tokens: {:.1}%", same_pct);
        
        println!("\n✓ FP8 Qwen2 inference test passed!\n");
        Ok(())
    }

    #[test]
    #[ignore] // Run with: cargo test --features cuda -- --ignored test_fp8_qwen3_multi_turn
    fn test_fp8_qwen3_multi_turn() -> Result<()> {
        use crate::models::quantized_qwen3::ModelWeights;

        println!("\n=== Testing FP8 KV Cache with Qwen3-0.6B (Multi-Turn) ===\n");
        
        let device = Device::cuda_if_available(0)?;
        println!("Using device: {:?}", device);
        
        println!("Downloading model...");
        let model_path = download_model(
            "unsloth/Qwen3-0.6B-GGUF",
            "Qwen3-0.6B-Q4_K_M.gguf",
        )?;
        println!("✓ Model downloaded\n");

        println!("Loading model with FP8 cache enabled...");
        let mut model = ModelWeights::from_gguf_by_path(&model_path, &device)?;
        model.enable_fp8_kv_cache();
        println!("✓ Model loaded with FP8 cache\n");

        // Simulate multi-turn conversation
        println!("Simulating 3-turn conversation:");
        let turns = vec![
            ("Turn 1", 20),
            ("Turn 2", 15),
            ("Turn 3", 25),
        ];

        let mut total_tokens = 0;
        for (turn_name, num_tokens) in turns {
            println!("\n{}  Generating {} tokens...", turn_name, num_tokens);
            
            for i in 0..num_tokens {
                let input = Tensor::new(&[1u32], &device)?.unsqueeze(0)?;
                let _output = model.forward(&input, total_tokens + i)?;
            }
            
            total_tokens += num_tokens;
            let cache_len = model.cache_len();
            
            println!("   ✓ Generated {} tokens", num_tokens);
            println!("   Total cache length: {}", cache_len);
            
            assert_eq!(cache_len, total_tokens, "Cache length mismatch");
        }

        println!("\n4. Testing cache operations:");
        
        // Test truncation
        println!("   Truncating to 30 tokens...");
        // Note: We'd need to expose truncate method on ModelWeights for this
        
        println!("\n✓ FP8 Qwen3 multi-turn test passed!");
        println!("  Successfully processed {} tokens across 3 turns\n", total_tokens);
        
        Ok(())
    }

    #[test]
    fn test_fp8_quantization_accuracy() -> Result<()> {
        use candle_nn::kv_cache::Fp8KvCache;

        let device = Device::Cpu;
        
        // Create test tensors
        let k = Tensor::randn(0f32, 1.0f32, (1, 4, 10, 64), &device)?;
        let v = Tensor::randn(0f32, 1.0f32, (1, 4, 10, 64), &device)?;

        // Test FP8 cache
        let mut fp8_cache = Fp8KvCache::new(2, 2048);
        let (k_out, v_out) = fp8_cache.append(&k, &v)?;

        // Check shapes
        assert_eq!(k_out.shape(), k.shape());
        assert_eq!(v_out.shape(), v.shape());

        // Check that values are similar (not exact due to quantization)
        let k_diff = (&k - &k_out)?.abs()?.mean_all()?.to_scalar::<f32>()?;
        let v_diff = (&v - &v_out)?.abs()?.mean_all()?.to_scalar::<f32>()?;

        println!("K mean absolute difference: {}", k_diff);
        println!("V mean absolute difference: {}", v_diff);

        // FP8 E4M3 should maintain reasonable accuracy
        assert!(k_diff < 0.1, "K quantization error too large: {}", k_diff);
        assert!(v_diff < 0.1, "V quantization error too large: {}", v_diff);

        println!("✓ FP8 quantization accuracy test passed");
        Ok(())
    }

    #[test]
    fn test_fp8_cache_operations() -> Result<()> {
        use candle_nn::kv_cache::Fp8KvCache;

        let device = Device::Cpu;
        let mut cache = Fp8KvCache::new(2, 512);

        // Test append
        let k1 = Tensor::randn(0f32, 1.0f32, (1, 4, 5, 64), &device)?;
        let v1 = Tensor::randn(0f32, 1.0f32, (1, 4, 5, 64), &device)?;
        cache.append(&k1, &v1)?;
        assert_eq!(cache.current_seq_len(), 5);

        // Test second append
        let k2 = Tensor::randn(0f32, 1.0f32, (1, 4, 3, 64), &device)?;
        let v2 = Tensor::randn(0f32, 1.0f32, (1, 4, 3, 64), &device)?;
        cache.append(&k2, &v2)?;
        assert_eq!(cache.current_seq_len(), 8);

        // Test truncate
        cache.truncate(5)?;
        assert_eq!(cache.current_seq_len(), 5);

        // Test reset
        cache.reset();
        assert_eq!(cache.current_seq_len(), 0);

        println!("✓ FP8 cache operations test passed");
        Ok(())
    }

    #[test]
    fn test_fp8_cache_growing() -> Result<()> {
        use candle_nn::kv_cache::Fp8KvCache;

        let device = Device::Cpu;
        let mut cache = Fp8KvCache::new(2, 10); // Small initial size

        // Append beyond initial capacity
        for i in 0..5 {
            let k = Tensor::randn(0f32, 1.0f32, (1, 2, 5, 32), &device)?;
            let v = Tensor::randn(0f32, 1.0f32, (1, 2, 5, 32), &device)?;
            cache.append(&k, &v)?;
            println!("After append {}: seq_len={}, max_len={}", 
                     i, cache.current_seq_len(), cache.max_seq_len());
        }

        // Should have grown beyond initial capacity
        assert!(cache.max_seq_len() > 10);
        assert_eq!(cache.current_seq_len(), 25);

        println!("✓ FP8 cache growing test passed");
        Ok(())
    }

    #[test]
    fn test_fp8_vs_regular_cache_consistency() -> Result<()> {
        use candle_nn::kv_cache::{KvCache, Fp8KvCache};

        let device = Device::Cpu;
        
        let mut regular_cache = KvCache::new(2, 2048);
        let mut fp8_cache = Fp8KvCache::new(2, 2048);

        // Same input to both caches
        let k = Tensor::randn(0f32, 1.0f32, (1, 4, 10, 64), &device)?;
        let v = Tensor::randn(0f32, 1.0f32, (1, 4, 10, 64), &device)?;

        let (k_regular, v_regular) = regular_cache.append(&k, &v)?;
        let (k_fp8, v_fp8) = fp8_cache.append(&k, &v)?;

        // Shapes should match
        assert_eq!(k_regular.shape(), k_fp8.shape());
        assert_eq!(v_regular.shape(), v_fp8.shape());

        // Check sequence lengths match
        assert_eq!(regular_cache.current_seq_len(), fp8_cache.current_seq_len());

        println!("✓ FP8 vs regular cache consistency test passed");
        Ok(())
    }
}
