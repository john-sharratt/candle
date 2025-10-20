#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{Device, Result, Tensor};

#[test]
fn kv_cache() -> Result<()> {
    let mut cache = candle_nn::kv_cache::Cache::new(0, 16);
    for _ in [0, 1] {
        assert_eq!(cache.current_seq_len(), 0);
        let data = cache.current_data()?;
        assert!(data.is_none());
        let t = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
        cache.append(&t)?;
        let data = cache.current_data()?.unwrap();
        assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3.]);
        let t = Tensor::new(&[4f32], &Device::Cpu)?;
        cache.append(&t)?;
        let data = cache.current_data()?.unwrap();
        assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3., 4.]);
        let t = Tensor::new(&[0f32, 5., 6., 7.], &Device::Cpu)?;
        cache.append(&t)?;
        let data = cache.current_data()?.unwrap();
        assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3., 4., 0., 5., 6., 7.]);
        assert_eq!(cache.current_seq_len(), 8);
        cache.reset();
    }
    Ok(())
}

#[test]
fn kv_cache_truncate() -> Result<()> {
    let mut cache = candle_nn::kv_cache::Cache::new(0, 16);

    // Build up some cache content
    let t1 = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
    cache.append(&t1)?;
    let t2 = Tensor::new(&[4f32, 5.], &Device::Cpu)?;
    cache.append(&t2)?;
    let t3 = Tensor::new(&[6f32, 7., 8.], &Device::Cpu)?;
    cache.append(&t3)?;

    // Should have [1, 2, 3, 4, 5, 6, 7, 8] with length 8
    assert_eq!(cache.current_seq_len(), 8);
    let data = cache.current_data()?.unwrap();
    assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3., 4., 5., 6., 7., 8.]);

    // Truncate to position 5 - should keep [1, 2, 3, 4, 5]
    cache.truncate(5)?;
    assert_eq!(cache.current_seq_len(), 5);
    let data = cache.current_data()?.unwrap();
    assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3., 4., 5.]);

    // Truncate to position 2 - should keep [1, 2]
    cache.truncate(2)?;
    assert_eq!(cache.current_seq_len(), 2);
    let data = cache.current_data()?.unwrap();
    assert_eq!(data.to_vec1::<f32>()?, [1., 2.]);

    // Can still append after truncate
    let t4 = Tensor::new(&[9f32, 10.], &Device::Cpu)?;
    cache.append(&t4)?;
    assert_eq!(cache.current_seq_len(), 4);
    let data = cache.current_data()?.unwrap();
    assert_eq!(data.to_vec1::<f32>()?, [1., 2., 9., 10.]);

    // Truncate to 0 should clear everything
    cache.truncate(0)?;
    assert_eq!(cache.current_seq_len(), 0);
    let data = cache.current_data()?;
    assert!(data.is_none());

    Ok(())
}

#[test]
fn kv_cache_truncate_no_op() -> Result<()> {
    let mut cache = candle_nn::kv_cache::Cache::new(0, 16);

    // Build up cache content
    let t = Tensor::new(&[1f32, 2., 3., 4., 5.], &Device::Cpu)?;
    cache.append(&t)?;
    assert_eq!(cache.current_seq_len(), 5);

    // Truncating to same length should be no-op
    cache.truncate(5)?;
    assert_eq!(cache.current_seq_len(), 5);
    let data = cache.current_data()?.unwrap();
    assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3., 4., 5.]);

    // Truncating to larger length should be no-op
    cache.truncate(10)?;
    assert_eq!(cache.current_seq_len(), 5);
    let data = cache.current_data()?.unwrap();
    assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3., 4., 5.]);

    Ok(())
}

#[test]
fn kv_cache_full_truncate_workflow() -> Result<()> {
    let mut kv_cache = candle_nn::kv_cache::KvCache::new(0, 16);

    // Simulate attention operations - KvCache concatenates on dim 0 by default
    let k1 = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?; // (2, 2)
    let v1 = Tensor::new(&[[5f32, 6.], [7., 8.]], &Device::Cpu)?; // (2, 2)
    let (k_out, v_out) = kv_cache.append(&k1, &v1)?;
    assert_eq!(k_out.dims(), &[2, 2]);
    assert_eq!(v_out.dims(), &[2, 2]);

    let k2 = Tensor::new(&[[9f32, 10.], [11., 12.]], &Device::Cpu)?; // (2, 2)
    let v2 = Tensor::new(&[[13f32, 14.], [15., 16.]], &Device::Cpu)?; // (2, 2)
    let (k_out, v_out) = kv_cache.append(&k2, &v2)?;
    assert_eq!(k_out.dims(), &[4, 2]); // Now (4, 2) - concatenated on dim 0
    assert_eq!(v_out.dims(), &[4, 2]); // Check cache lengths
    assert_eq!(kv_cache.k_cache().current_seq_len(), 4);
    assert_eq!(kv_cache.v_cache().current_seq_len(), 4);

    // Truncate both caches to position 2
    kv_cache.truncate(2)?;
    assert_eq!(kv_cache.k_cache().current_seq_len(), 2);
    assert_eq!(kv_cache.v_cache().current_seq_len(), 2);

    // Get truncated data
    let k_data = kv_cache.k()?.unwrap();
    let v_data = kv_cache.v()?.unwrap();
    assert_eq!(k_data.dims(), &[2, 2]);
    assert_eq!(v_data.dims(), &[2, 2]);

    // Reset and verify
    kv_cache.reset();
    assert_eq!(kv_cache.k_cache().current_seq_len(), 0);
    assert_eq!(kv_cache.v_cache().current_seq_len(), 0);
    let k_data = kv_cache.k()?;
    let v_data = kv_cache.v()?;
    assert!(k_data.is_none());
    assert!(v_data.is_none());

    Ok(())
}

#[test]
fn rotating_kv_cache() -> Result<()> {
    let mut cache = candle_nn::kv_cache::RotatingCache::new(0, 6);
    for _ in [0, 1] {
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.current_seq_len(), 0);
        let data = cache.current_data()?;
        assert!(data.is_none());
        assert_eq!(cache.positions(1), &[0]);
        assert_eq!(cache.positions(2), &[0, 1]);
        let t = Tensor::new(&[1., 2., 3.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [1., 2., 3.]);
        assert_eq!(cache.positions(0), &[0, 1, 2]);
        assert_eq!(cache.positions(1), &[0, 1, 2, 3]);
        assert_eq!(cache.positions(2), &[0, 1, 2, 3, 4]);
        assert_eq!(cache.positions(3), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(cache.positions(4), &[6, 1, 2, 3, 4, 5]);
        let t = Tensor::new(&[4.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [1., 2., 3., 4.]);
        let t = Tensor::new(&[0., 5., 6., 7.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [6., 7., 3., 4., 0., 5.]);
        assert_eq!(cache.current_seq_len(), 8);
        assert_eq!(cache.offset(), 2);

        let t = Tensor::new(&[8.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [6., 7., 8., 4., 0., 5.]);
        assert_eq!(cache.current_seq_len(), 9);
        assert_eq!(cache.offset(), 3);

        let t = Tensor::new(&[9., 10., 11.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [6., 7., 8., 9., 10., 11.]);
        assert_eq!(cache.current_seq_len(), 12);
        assert_eq!(cache.offset(), 0);

        let t = Tensor::new(&[12.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [12., 7., 8., 9., 10., 11.]);
        assert_eq!(cache.current_seq_len(), 13);
        assert_eq!(cache.offset(), 1);

        let mask = cache.attn_mask(2, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2::<u8>()?,
            &[[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        );
        let mask = cache.attn_mask(3, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2::<u8>()?,
            &[[0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]],
        );
        assert_eq!(cache.positions(0), &[12, 7, 8, 9, 10, 11]);
        assert_eq!(cache.positions(2), &[12, 13, 14, 9, 10, 11]);
        assert_eq!(cache.positions(3), &[12, 13, 14, 15, 10, 11]);
        assert_eq!(cache.positions(8), &[13, 14, 15, 16, 17, 18, 19, 20]);
        let t = Tensor::new(&[0., 1., 2., 3., 4., 5., 6., 7., 8.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [0., 1., 2., 3., 4., 5., 6., 7., 8.]);
        assert_eq!(cache.current_seq_len(), 22);
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.positions(0), &[16, 17, 18, 19, 20, 21]);
        assert_eq!(cache.positions(1), &[22, 17, 18, 19, 20, 21]);

        let mask = cache.attn_mask(1, &Device::Cpu)?;
        assert!(mask.is_none());
        let mask = cache.attn_mask(2, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2::<u8>()?,
            &[[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        );
        let mask = cache.attn_mask(3, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2::<u8>()?,
            &[[0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        );
        let t = Tensor::new(&[42.], &Device::Cpu)?;

        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [42., 4., 5., 6., 7., 8.]);
        assert_eq!(cache.current_seq_len(), 23);
        assert_eq!(cache.offset(), 1);

        cache.reset();
    }
    Ok(())
}
