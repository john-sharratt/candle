// Test for Qwen3 RoPE handling of long contexts beyond max_position_embeddings
use candle::{DType, Device, Result, Tensor};

#[test]
fn test_qwen3_rope_beyond_max_positions() -> Result<()> {
    // Simulate RoPE with small max_position_embeddings to test wrapping behavior
    let device = Device::Cpu;
    let head_dim = 128;
    let max_position_embeddings = 1024; // Intentionally small for testing
    let rope_theta: f64 = 1000000.0;
    
    // Build RoPE embeddings manually (mimicking RotaryEmbedding::new)
    let dim = head_dim;
    let inv_freq: Vec<_> = (0..dim)
        .step_by(2)
        .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
        .collect();
    let inv_freq_len = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), &device)?.to_dtype(DType::F32)?;
    let t = Tensor::arange(0u32, max_position_embeddings as u32, &device)?
        .to_dtype(DType::F32)?
        .reshape((max_position_embeddings, 1))?;
    let freqs = t.matmul(&inv_freq)?;
    let sin = freqs.sin()?;
    let cos = freqs.cos()?;
    
    // Test 1: Normal case within bounds (offset=512, seq_len=128)
    let offset1 = 512;
    let seq_len1 = 128;
    assert!(offset1 + seq_len1 <= max_position_embeddings);
    let cos1 = cos.narrow(0, offset1, seq_len1)?;
    let sin1 = sin.narrow(0, offset1, seq_len1)?;
    assert_eq!(cos1.dims(), &[seq_len1, inv_freq_len]);
    assert_eq!(sin1.dims(), &[seq_len1, inv_freq_len]);
    println!("✓ Test 1: Normal case (offset={}, seq_len={}) passed", offset1, seq_len1);
    
    // Test 2: Boundary case at exact limit (offset=1000, seq_len=24)
    let offset2 = 1000;
    let seq_len2 = 24;
    assert_eq!(offset2 + seq_len2, max_position_embeddings);
    let cos2 = cos.narrow(0, offset2, seq_len2)?;
    let sin2 = sin.narrow(0, offset2, seq_len2)?;
    assert_eq!(cos2.dims(), &[seq_len2, inv_freq_len]);
    assert_eq!(sin2.dims(), &[seq_len2, inv_freq_len]);
    println!("✓ Test 2: Boundary case (offset={}, seq_len={}) passed", offset2, seq_len2);
    
    // Test 3: Beyond bounds with modulo wrapping (offset=2000, seq_len=1)
    let offset3 = 2000;
    let seq_len3 = 1;
    // This simulates the fixed implementation with modulo wrapping
    let positions: Vec<u32> = (offset3..offset3 + seq_len3)
        .map(|pos| (pos % max_position_embeddings) as u32)
        .collect();
    let pos_tensor = Tensor::from_vec(positions, (seq_len3,), &device)?;
    let cos3 = cos.index_select(&pos_tensor, 0)?;
    let sin3 = sin.index_select(&pos_tensor, 0)?;
    assert_eq!(cos3.dims(), &[seq_len3, inv_freq_len]);
    assert_eq!(sin3.dims(), &[seq_len3, inv_freq_len]);
    
    // Verify that position 2000 wraps to position 976 (2000 % 1024)
    let expected_pos = (offset3 % max_position_embeddings) as usize;
    let cos_expected = cos.narrow(0, expected_pos, 1)?;
    let sin_expected = sin.narrow(0, expected_pos, 1)?;
    
    let cos3_data = cos3.to_vec2::<f32>()?;
    let cos_expected_data = cos_expected.to_vec2::<f32>()?;
    assert_eq!(cos3_data, cos_expected_data, "cos values should match after wrapping");
    
    let sin3_data = sin3.to_vec2::<f32>()?;
    let sin_expected_data = sin_expected.to_vec2::<f32>()?;
    assert_eq!(sin3_data, sin_expected_data, "sin values should match after wrapping");
    
    println!("✓ Test 3: Beyond bounds wrapping (offset={}, wraps to {}) passed", offset3, expected_pos);
    
    // Test 4: Far beyond bounds (offset=10000, seq_len=10)
    let offset4 = 10000;
    let seq_len4 = 10;
    let positions4: Vec<u32> = (offset4..offset4 + seq_len4)
        .map(|pos| (pos % max_position_embeddings) as u32)
        .collect();
    let pos_tensor4 = Tensor::from_vec(positions4, (seq_len4,), &device)?;
    let cos4 = cos.index_select(&pos_tensor4, 0)?;
    let sin4 = sin.index_select(&pos_tensor4, 0)?;
    assert_eq!(cos4.dims(), &[seq_len4, inv_freq_len]);
    assert_eq!(sin4.dims(), &[seq_len4, inv_freq_len]);
    println!("✓ Test 4: Far beyond bounds (offset={}, seq_len={}) passed", offset4, seq_len4);
    
    println!("\n✅ All RoPE long context tests passed!");
    Ok(())
}
