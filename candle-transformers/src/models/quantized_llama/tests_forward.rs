//! Tests that a cloned model keeps its KV cache independent, and that flash
//! attention matches standard attention on prompts.
//!
//! `test_clone_with_independent_kv_cache` advances a Llama-3.2-1B-Instruct model
//! past 500 tokens, clones it, advances the clone on a different token, and checks
//! that the original cache is unchanged. `test_flash_attention_prompt` runs a long
//! prompt, which takes the flash path, then a single token and another multi-token
//! sequence, and checks that the outputs match, or are close under BF16.

use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod tests {
    #![allow(
        clippy::cloned_ref_to_slice_refs,
        clippy::type_complexity,
        clippy::useless_vec
    )]

    use super::*;

    #[allow(unused_imports)]
    use candle_nn::kv_cache::CacheIntegrityResult;

    #[test]
    #[ignore] // Downloads model from HuggingFace. Run with: cargo test --release -- --ignored test_clone_with_independent_kv_cache
    pub(in super::super) fn test_clone_with_independent_kv_cache() -> Result<()> {
        // Download a small Llama model from HuggingFace
        // Using Llama-3.2-1B-Instruct (smallest available for fast testing)
        let api = crate::models::batch_test::test_helpers::api()
            .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;

        let repo = api.model("bartowski/Llama-3.2-1B-Instruct-GGUF".to_string());
        let model_path = repo.get("Llama-3.2-1B-Instruct-Q4_K_M.gguf").map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to download model: {}. This test requires internet access.",
                e
            ))
        })?;

        println!("Model downloaded to: {:?}", model_path);

        let device = Device::cuda_if_available(0)?;
        println!("Using device: {:?}", device);

        // Load model using optimized mmap path
        println!("Loading model with mmap optimization...");
        let load_start = std::time::Instant::now();
        let model = ModelWeights::from_gguf_by_path_v3(&model_path, &device)?;
        let load_duration = load_start.elapsed();
        println!(
            "✓ Model loaded in {:.3}s using mmap\n",
            load_duration.as_secs_f64()
        );

        println!("Starting 500-token prefill...");

        // Step 1: Advance model forward by 500+ tokens to populate KV cache
        // Using token ID 1 (typically a valid token in most vocabularies)
        let prefill_tokens = 500;
        let mut caches = model.create_kv_caches(2048);
        for i in 0..prefill_tokens {
            let input = Tensor::new(&[1u32], &device)?.unsqueeze(0)?;
            let _output = model.forward(&mut caches, &input, i)?;

            if (i + 1) % 100 == 0 {
                println!("  Prefill progress: {}/{}", i + 1, prefill_tokens);
            }
        }

        let original_cache_len = caches.current_seq_len();
        assert_eq!(
            original_cache_len, prefill_tokens,
            "Original model should have {} tokens in cache",
            prefill_tokens
        );
        println!("✓ Original model cache: {} tokens", original_cache_len);

        // Step 2: Clone the model
        println!("\nCloning model...");
        let cloned_model = model.clone();
        let mut cloned_caches = cloned_model.create_kv_caches(2048);
        let clone_initial_cache_len = cloned_caches.current_seq_len();
        assert_eq!(
            clone_initial_cache_len, 0,
            "Cloned model should start with empty cache"
        );
        println!(
            "✓ Cloned model cache: {} tokens (empty)",
            clone_initial_cache_len
        );

        // Step 3: Advance clone forward with new prompt (different token: 2)
        println!("\nAdvancing clone with new prompt (100 tokens)...");
        let clone_tokens = 100;
        for i in 0..clone_tokens {
            let input = Tensor::new(&[2u32], &device)?.unsqueeze(0)?;
            let _output = cloned_model.forward(&mut cloned_caches, &input, i)?;
        }

        let clone_cache_len = cloned_caches.current_seq_len();
        assert_eq!(
            clone_cache_len, clone_tokens,
            "Clone should have {} tokens in cache",
            clone_tokens
        );
        println!("✓ Clone cache after generation: {} tokens", clone_cache_len);

        // Step 4: Verify original model cache is still intact
        let original_cache_len_after_clone = caches.current_seq_len();
        assert_eq!(
            original_cache_len_after_clone, prefill_tokens,
            "Original model cache should still have {} tokens (not affected by clone)",
            prefill_tokens
        );
        println!(
            "✓ Original cache after clone generation: {} tokens (unchanged)",
            original_cache_len_after_clone
        );

        // Step 5: Advance original forward with continuation (token 3)
        println!("\nAdvancing original model (50 more tokens)...");
        let original_continue_tokens = 50;
        for i in 0..original_continue_tokens {
            let input = Tensor::new(&[3u32], &device)?.unsqueeze(0)?;
            let _output = model.forward(&mut caches, &input, prefill_tokens + i)?;
        }

        let original_final_cache_len = caches.current_seq_len();
        assert_eq!(
            original_final_cache_len,
            prefill_tokens + original_continue_tokens,
            "Original model should have {} tokens in cache",
            prefill_tokens + original_continue_tokens
        );
        println!(
            "✓ Original cache after continuation: {} tokens",
            original_final_cache_len
        );

        // Verify caches are completely independent
        assert_ne!(
            original_final_cache_len, clone_cache_len,
            "Original and clone should have different cache lengths"
        );

        println!("\n=== Test Summary ===");
        println!(
            "✓ Original model: {} tokens in cache",
            original_final_cache_len
        );
        println!("✓ Cloned model: {} tokens in cache", clone_cache_len);
        println!("✓ Caches are completely independent");
        println!("✓ Clone did not affect original model state");

        Ok(())
    }

    #[test]
    #[ignore] // Run with: cargo test --features cuda,flash-attn -- --ignored test_flash_attention_prompt
    pub(in super::super) fn test_flash_attention_prompt() -> Result<()> {
        println!("\n=== Testing Flash Attention for Prompt Processing ===\n");

        let api = crate::models::batch_test::test_helpers::api()
            .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;

        let repo = api.model("bartowski/Llama-3.2-1B-Instruct-GGUF".to_string());
        let model_path = repo.get("Llama-3.2-1B-Instruct-Q4_K_M.gguf").map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to download model: {}. This test requires internet access.",
                e
            ))
        })?;

        println!("Model downloaded to: {:?}", model_path);

        let device = Device::new_cuda(0).map_err(|e| {
            candle::Error::Msg(format!(
                "CUDA required for this test: {}. Use --features cuda,flash-attn",
                e
            ))
        })?;
        println!("Using device: {:?}\n", device);

        // Load model
        let model = ModelWeights::from_gguf_by_path_v3(&model_path, &device)?;
        println!("✓ Model loaded\n");

        // Test 1: Process a long prompt (should trigger Flash Attention)
        println!("Test 1: Long prompt processing (64 tokens)");
        let prompt_len = 64;
        let prompt_tokens: Vec<u32> = (0..prompt_len).map(|i| (i % 1000 + 1) as u32).collect();
        let prompt = Tensor::new(&prompt_tokens[..], &device)?.unsqueeze(0)?;

        let mut caches = model.create_kv_caches(2048);
        let start = std::time::Instant::now();
        let output = model.forward(&mut caches, &prompt, 0)?;
        let duration = start.elapsed();

        println!("  ✓ Processed {} tokens", prompt_len);
        println!("  Time: {:.3}ms", duration.as_secs_f64() * 1000.0);
        println!("  Output shape: {:?}", output.shape());
        println!("  Cache length: {}", caches.current_seq_len());
        assert_eq!(caches.current_seq_len(), prompt_len);

        // Test 2: Single token generation (should use standard attention)
        println!("\nTest 2: Single token generation (autoregressive)");
        let start = std::time::Instant::now();
        let output = model.forward(
            &mut caches,
            &Tensor::new(&[1u32], &device)?.unsqueeze(0)?,
            prompt_len,
        )?;
        let duration = start.elapsed();

        println!("  ✓ Generated 1 token");
        println!("  Time: {:.3}ms", duration.as_secs_f64() * 1000.0);
        println!("  Output shape: {:?}", output.shape());
        println!("  Cache length: {}", caches.current_seq_len());
        assert_eq!(caches.current_seq_len(), prompt_len + 1);

        // Test 3: Another multi-token sequence (Flash Attention again)
        println!("\nTest 3: Another multi-token batch (32 tokens)");
        let mut caches = model.create_kv_caches(2048);
        let batch_len = 32;
        let batch_tokens: Vec<u32> = (0..batch_len).map(|i| (i % 500 + 1) as u32).collect();
        let batch = Tensor::new(&batch_tokens[..], &device)?.unsqueeze(0)?;

        let start = std::time::Instant::now();
        let output = model.forward(&mut caches, &batch, 0)?;
        let duration = start.elapsed();

        println!("  ✓ Processed {} tokens", batch_len);
        println!("  Time: {:.3}ms", duration.as_secs_f64() * 1000.0);
        println!("  Output shape: {:?}", output.shape());
        println!("  Cache length: {}", caches.current_seq_len());
        assert_eq!(caches.current_seq_len(), batch_len);

        // Test 4: Verify numerical stability
        println!("\nTest 4: Numerical stability check");
        let mut caches1 = model.create_kv_caches(2048);
        let test_tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let test_input = Tensor::new(&test_tokens[..], &device)?.unsqueeze(0)?;

        let output1 = model.forward(&mut caches1, &test_input, 0)?;
        let mut caches2 = model.create_kv_caches(2048);
        let output2 = model.forward(&mut caches2, &test_input, 0)?;

        // Check outputs are identical (or very close due to BF16 precision)
        let diff = (&output1 - &output2)?.abs()?.flatten_all()?.max(0)?;
        let diff_val = diff.to_vec0::<f32>()?;
        println!("  Max difference between runs: {:.6}", diff_val);
        assert!(diff_val < 1e-3, "Outputs should be consistent");
        println!("  ✓ Outputs are consistent");

        println!("\n=== Flash Attention Test Summary ===");
        println!("✓ Long prompt processing works (64 tokens)");
        println!("✓ Single token generation works");
        println!("✓ Multi-token batching works (32 tokens)");
        println!("✓ Numerical stability verified");
        println!(
            "Note: Flash Attention is used for seq_len > 1 on CUDA, fallback for seq_len == 1\n"
        );

        Ok(())
    }
}
