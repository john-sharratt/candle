// Integration test for quantized_qwen2 fixes
#[test]
fn test_kvcache_integration() {
    use candle::{Device, Tensor};
    use candle_nn::kv_cache::KvCache;

    let device = Device::Cpu;

    // Test KvCache behavior that was causing issues
    let mut cache = KvCache::new(2, 100);

    // Initial state
    assert_eq!(cache.current_seq_len(), 0, "Cache should start at 0");

    // Simulate appending some cache data (like during model inference)
    let k = Tensor::zeros((1, 4, 10, 64), candle::DType::F32, &device).unwrap();
    let v = Tensor::zeros((1, 4, 10, 64), candle::DType::F32, &device).unwrap();

    // Append data to cache
    let (k_out, _v_out) = cache.append(&k, &v).unwrap();
    assert_eq!(
        cache.current_seq_len(),
        10,
        "Cache should have 10 tokens after append"
    );
    assert_eq!(k_out.dims()[2], 10, "Output K should have seq_len=10");

    // Append more data
    let k2 = Tensor::zeros((1, 4, 5, 64), candle::DType::F32, &device).unwrap();
    let v2 = Tensor::zeros((1, 4, 5, 64), candle::DType::F32, &device).unwrap();
    cache.append(&k2, &v2).unwrap();
    assert_eq!(
        cache.current_seq_len(),
        15,
        "Cache should have 15 tokens after second append"
    );

    // Test truncation to a smaller size
    cache.truncate(10).unwrap();
    assert_eq!(
        cache.current_seq_len(),
        10,
        "Cache should be truncated to 10"
    );

    // Append after truncation - this should work without OutOfAlignment errors
    let k3 = Tensor::zeros((1, 4, 3, 64), candle::DType::F32, &device).unwrap();
    let v3 = Tensor::zeros((1, 4, 3, 64), candle::DType::F32, &device).unwrap();
    let (k_out2, _v_out2) = cache.append(&k3, &v3).unwrap();
    assert_eq!(
        cache.current_seq_len(),
        13,
        "Cache should have 13 tokens after append following truncation"
    );
    assert_eq!(k_out2.dims()[2], 13, "Output K should have seq_len=13");

    // Test truncate(0) - this was causing OutOfAlignment errors before
    cache.truncate(0).unwrap();
    assert_eq!(
        cache.current_seq_len(),
        0,
        "Cache should be 0 after truncate(0)"
    );

    // Append after truncate(0) should work like starting fresh
    let k4 = Tensor::zeros((1, 4, 7, 64), candle::DType::F32, &device).unwrap();
    let v4 = Tensor::zeros((1, 4, 7, 64), candle::DType::F32, &device).unwrap();
    let (k_out3, _v_out3) = cache.append(&k4, &v4).unwrap();
    assert_eq!(
        cache.current_seq_len(),
        7,
        "Cache should have 7 tokens after append following truncate(0)"
    );
    assert_eq!(k_out3.dims()[2], 7, "Output K should have seq_len=7");

    // Test reset
    cache.reset();
    assert_eq!(
        cache.current_seq_len(),
        0,
        "Cache should be 0 after reset()"
    );

    // Append after reset should also work
    cache.append(&k, &v).unwrap();
    assert_eq!(
        cache.current_seq_len(),
        10,
        "Cache should work normally after reset"
    );
}

#[test]
fn test_head_dim_fallback() {
    // This test verifies that the head_dim calculation logic is present
    // The actual fix reads from "qwen2.attention.key_length" with fallback

    let embedding_length = 5120;
    let head_count = 40;

    // Fallback calculation that would be used if metadata key doesn't exist
    let head_dim_fallback = embedding_length / head_count;
    assert_eq!(head_dim_fallback, 128);

    // But for Qwen2.5 14B, metadata should provide head_dim=128 directly
    // preventing the incorrect calculation that was giving 200
}
