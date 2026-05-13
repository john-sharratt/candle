//! Tests for GGUF quantized model KV cache functionality
//! This tests the new cache management methods added to quantized models

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{Device, Result, Tensor};
use candle_nn::kv_cache::KvCache;

/// Mock attention layer that mimics the behavior of AttentionWeights
struct MockAttentionWeights {
    kv_cache: KvCache,
    head_dim: usize,
}

impl MockAttentionWeights {
    fn new(head_dim: usize) -> Self {
        Self {
            kv_cache: KvCache::new(2, 512), // dim=2, max_seq_len=512
            head_dim,
        }
    }

    /// Mimic the new truncate_cache method added to GGUF AttentionWeights
    pub fn truncate_cache(&mut self, seq_len: usize) -> Result<()> {
        self.kv_cache.truncate(seq_len)
    }

    /// Mimic the new reset_cache method added to GGUF AttentionWeights
    pub fn reset_cache(&mut self) {
        self.kv_cache.reset();
    }

    /// Mimic the new cache_len method added to GGUF AttentionWeights
    pub fn cache_len(&self) -> usize {
        self.kv_cache.k_cache().current_seq_len()
    }

    /// Simulate attention forward pass that appends to cache
    fn forward_step(&mut self, seq_len: usize, offset: usize) -> Result<()> {
        // Reset cache if we're at the first position (mimics GGUF behavior)
        if offset == 0 {
            self.kv_cache.reset();
        }

        // Create dummy k, v tensors for this step
        let k = Tensor::zeros(
            (1, 1, seq_len, self.head_dim),
            candle::DType::F32,
            &Device::Cpu,
        )?;
        let v = Tensor::zeros(
            (1, 1, seq_len, self.head_dim),
            candle::DType::F32,
            &Device::Cpu,
        )?;

        // Append to cache (this is what the real forward() does)
        let (_k_out, _v_out) = self.kv_cache.append(&k, &v)?;

        Ok(())
    }
}

/// Mock layer that contains attention - mimics LayerWeights
struct MockLayerWeights {
    self_attn: MockAttentionWeights,
}

impl MockLayerWeights {
    fn new() -> Self {
        Self {
            self_attn: MockAttentionWeights::new(64), // 64-dim heads
        }
    }

    /// Mimic the new truncate_cache method added to GGUF LayerWeights
    pub fn truncate_cache(&mut self, seq_len: usize) -> Result<()> {
        self.self_attn.truncate_cache(seq_len)
    }

    /// Mimic the new reset_cache method added to GGUF LayerWeights
    pub fn reset_cache(&mut self) {
        self.self_attn.reset_cache();
    }

    /// Mimic the new cache_len method added to GGUF LayerWeights
    pub fn cache_len(&self) -> usize {
        self.self_attn.cache_len()
    }

    fn forward_step(&mut self, seq_len: usize, offset: usize) -> Result<()> {
        self.self_attn.forward_step(seq_len, offset)
    }
}

/// Mock model that contains multiple layers - mimics ModelWeights  
struct MockModelWeights {
    layers: Vec<MockLayerWeights>,
}

impl MockModelWeights {
    fn new(num_layers: usize) -> Self {
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(MockLayerWeights::new());
        }
        Self { layers }
    }

    /// Mimic the new truncate_kv_cache method added to GGUF ModelWeights
    pub fn truncate_kv_cache(&mut self, seq_len: usize) -> Result<()> {
        for layer in &mut self.layers {
            layer.truncate_cache(seq_len)?;
        }
        Ok(())
    }

    /// Mimic the new clear_all_caches method added to GGUF ModelWeights
    pub fn clear_all_caches(&mut self) {
        for layer in &mut self.layers {
            layer.reset_cache();
        }
    }

    /// Mimic the new cache_len method added to GGUF ModelWeights
    pub fn cache_len(&self) -> usize {
        self.layers.first().map(|l| l.cache_len()).unwrap_or(0)
    }

    /// Mimic the new rewind_by_tokens method added to GGUF ModelWeights
    pub fn rewind_by_tokens(&mut self, n_tokens: usize) -> Result<()> {
        let current_len = self.cache_len();
        let target_pos = current_len.saturating_sub(n_tokens);
        self.truncate_kv_cache(target_pos)
    }

    /// Simulate forward pass that processes one token
    fn forward_token(&mut self, offset: usize) -> Result<()> {
        for layer in &mut self.layers {
            layer.forward_step(1, offset)?; // 1 token at a time
        }
        Ok(())
    }
}

#[test]
fn test_gguf_cache_clear() -> Result<()> {
    let mut model = MockModelWeights::new(2); // 2 layers

    // Initially empty
    assert_eq!(model.cache_len(), 0);

    // Process some tokens
    model.forward_token(0)?; // First token, should reset cache
    assert_eq!(model.cache_len(), 1);

    model.forward_token(1)?; // Second token
    assert_eq!(model.cache_len(), 2);

    model.forward_token(2)?; // Third token
    assert_eq!(model.cache_len(), 3);

    // Clear all caches
    model.clear_all_caches();
    assert_eq!(model.cache_len(), 0);

    // Should be able to start again
    model.forward_token(0)?; // offset=0 should reset anyway
    assert_eq!(model.cache_len(), 1);

    Ok(())
}

#[test]
fn test_gguf_cache_reset_on_offset_zero() -> Result<()> {
    let mut model = MockModelWeights::new(1); // 1 layer

    // Build up cache
    model.forward_token(0)?; // offset=0, resets cache
    model.forward_token(1)?;
    model.forward_token(2)?;
    assert_eq!(model.cache_len(), 3);

    // Forward with offset=0 should reset cache (GGUF behavior)
    model.forward_token(0)?;
    assert_eq!(model.cache_len(), 1); // Reset and added 1 token

    Ok(())
}

#[test]
fn test_gguf_cache_truncate_kv_cache() -> Result<()> {
    let mut model = MockModelWeights::new(2); // 2 layers

    // Build up context
    for i in 0..10 {
        model.forward_token(i)?;
    }
    assert_eq!(model.cache_len(), 10);

    // Truncate to position 7
    model.truncate_kv_cache(7)?;
    assert_eq!(model.cache_len(), 7);

    // Truncate to position 3
    model.truncate_kv_cache(3)?;
    assert_eq!(model.cache_len(), 3);

    // Can continue forward from truncated position
    model.forward_token(3)?; // Should append, not reset
    assert_eq!(model.cache_len(), 4);

    // Truncate to 0 (complete clear)
    model.truncate_kv_cache(0)?;
    assert_eq!(model.cache_len(), 0);

    // Should be able to start again after truncate
    model.forward_token(0)?;
    assert_eq!(model.cache_len(), 1);

    Ok(())
}

#[test]
fn test_gguf_cache_truncate_by_tokens() -> Result<()> {
    let mut model = MockModelWeights::new(1);

    // Build up 8 tokens
    for i in 0..8 {
        model.forward_token(i)?;
    }
    assert_eq!(model.cache_len(), 8);

    // Truncate by 3 tokens (rewind by 3)
    model.rewind_by_tokens(3)?;
    assert_eq!(model.cache_len(), 5); // 8 - 3 = 5

    // Truncate by 2 more tokens
    model.rewind_by_tokens(2)?;
    assert_eq!(model.cache_len(), 3); // 5 - 2 = 3

    // Truncate by more tokens than available (should saturate to 0)
    model.rewind_by_tokens(10)?;
    assert_eq!(model.cache_len(), 0);

    Ok(())
}

#[test]
fn test_gguf_cache_truncate_no_op() -> Result<()> {
    let mut model = MockModelWeights::new(1);

    // Build up 5 tokens
    for i in 0..5 {
        model.forward_token(i)?;
    }
    assert_eq!(model.cache_len(), 5);

    // Truncating to same position should be no-op
    model.truncate_kv_cache(5)?;
    assert_eq!(model.cache_len(), 5);

    // Truncating to larger position should be no-op
    model.truncate_kv_cache(10)?;
    assert_eq!(model.cache_len(), 5); // Rewinding by 0 tokens should be no-op
    model.rewind_by_tokens(0)?;
    assert_eq!(model.cache_len(), 5);

    Ok(())
}

#[test]
fn test_gguf_cache_multi_layer_consistency() -> Result<()> {
    let mut model = MockModelWeights::new(4); // 4 layers

    // Build up cache across all layers
    for i in 0..6 {
        model.forward_token(i)?;
    }

    // All layers should have same cache length
    for layer in &model.layers {
        assert_eq!(layer.cache_len(), 6);
    }

    // Truncate all layers
    model.truncate_kv_cache(3)?;

    // All layers should have same truncated length
    for layer in &model.layers {
        assert_eq!(layer.cache_len(), 3);
    }

    // Clear all layers
    model.clear_all_caches();

    // All layers should be empty
    for layer in &model.layers {
        assert_eq!(layer.cache_len(), 0);
    }

    Ok(())
}

#[test]
fn test_gguf_cache_time_travel_workflow() -> Result<()> {
    let mut model = MockModelWeights::new(2);

    // Simulate a conversation workflow
    // 1. System prompt (5 tokens)
    for i in 0..5 {
        model.forward_token(i)?;
    }
    let after_system = model.cache_len();
    assert_eq!(after_system, 5);

    // 2. User question (3 tokens)
    for i in 5..8 {
        model.forward_token(i)?;
    }
    let after_question = model.cache_len();
    assert_eq!(after_question, 8);

    // 3. Assistant thinking (4 tokens)
    for i in 8..12 {
        model.forward_token(i)?;
    }
    let after_thinking = model.cache_len();
    assert_eq!(after_thinking, 12);

    // Try different response approaches:

    // Attempt 1: Truncate to after thinking, try response
    model.truncate_kv_cache(after_thinking)?;
    assert_eq!(model.cache_len(), 12);
    model.forward_token(12)?; // First response token
    assert_eq!(model.cache_len(), 13);

    // Attempt 2: Truncate to after thinking again, different response
    model.truncate_kv_cache(after_thinking)?;
    assert_eq!(model.cache_len(), 12);
    model.forward_token(99)?; // Different response token
    assert_eq!(model.cache_len(), 13);

    // Attempt 3: Go back further, skip thinking entirely
    model.truncate_kv_cache(after_question)?;
    assert_eq!(model.cache_len(), 8);
    model.forward_token(50)?; // Direct answer token
    assert_eq!(model.cache_len(), 9);

    // Reset conversation entirely
    model.clear_all_caches();
    assert_eq!(model.cache_len(), 0);

    Ok(())
}

#[test]
fn test_gguf_cache_forward_consistency() -> Result<()> {
    let mut model1 = MockModelWeights::new(1);
    let mut model2 = MockModelWeights::new(1);

    // Both models should behave identically
    for i in 0..5 {
        model1.forward_token(i)?;
        model2.forward_token(i)?;
        assert_eq!(model1.cache_len(), model2.cache_len());
    }

    // Truncate both to same position
    model1.truncate_kv_cache(3)?;
    model2.truncate_kv_cache(3)?;
    assert_eq!(model1.cache_len(), model2.cache_len());
    assert_eq!(model1.cache_len(), 3);

    // Continue forward identically
    for i in 3..7 {
        model1.forward_token(i)?;
        model2.forward_token(i)?;
        assert_eq!(model1.cache_len(), model2.cache_len());
    }

    Ok(())
}
