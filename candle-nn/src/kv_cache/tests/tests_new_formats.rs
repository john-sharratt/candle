//! Tests that the newer quantized formats create, write and read back their blocks
//! within the error each bit width can carry.
//!
//! Each new format is exercised through backing creation, a write and a read. The
//! round-trip checks use tolerances matched to the format's bit width. Q4_KS and
//! Q8_KS are held to the accuracy of their plain counterparts, since their sub-block
//! scales are meant to match them, while Q2_0 and Q3_0 get looser bounds because
//! fewer bits carry more error.

use super::*;

#[test]
pub(super) fn test_quantized_backing_creation_q4_ks() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        4,
        64,
        KvFormat::Quantized(QuantFormat::Q4_KS),
        KvFormat::Quantized(QuantFormat::Q4_KS),
        &device,
        256,
    )?;
    assert_eq!(backing.k_format(), KvFormat::Quantized(QuantFormat::Q4_KS));
    assert!(backing.is_quantized());
    Ok(())
}

#[test]
pub(super) fn test_quantized_backing_creation_q8_ks() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        4,
        64,
        KvFormat::Quantized(QuantFormat::Q8_KS),
        KvFormat::Quantized(QuantFormat::Q8_KS),
        &device,
        256,
    )?;
    assert_eq!(backing.k_format(), KvFormat::Quantized(QuantFormat::Q8_KS));
    assert!(backing.is_quantized());
    Ok(())
}

#[test]
pub(super) fn test_quantized_backing_creation_q2_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        4,
        64,
        KvFormat::Quantized(QuantFormat::Q2_0),
        KvFormat::Quantized(QuantFormat::Q2_0),
        &device,
        256,
    )?;
    assert_eq!(backing.k_format(), KvFormat::Quantized(QuantFormat::Q2_0));
    assert!(backing.is_quantized());
    Ok(())
}

#[test]
pub(super) fn test_quantized_backing_creation_q3_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        4,
        64,
        KvFormat::Quantized(QuantFormat::Q3_0),
        KvFormat::Quantized(QuantFormat::Q3_0),
        &device,
        256,
    )?;
    assert_eq!(backing.k_format(), KvFormat::Quantized(QuantFormat::Q3_0));
    assert!(backing.is_quantized());
    Ok(())
}

#[test]
pub(super) fn test_quantized_write_q4_ks() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q4_KS),
        KvFormat::Quantized(QuantFormat::Q4_KS),
        &device,
        256,
    )?;
    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    Ok(())
}

#[test]
pub(super) fn test_quantized_write_q8_ks() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_KS),
        KvFormat::Quantized(QuantFormat::Q8_KS),
        &device,
        256,
    )?;
    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    Ok(())
}

#[test]
pub(super) fn test_quantized_write_q2_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q2_0),
        KvFormat::Quantized(QuantFormat::Q2_0),
        &device,
        256,
    )?;
    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    Ok(())
}

#[test]
pub(super) fn test_quantized_write_q3_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q3_0),
        KvFormat::Quantized(QuantFormat::Q3_0),
        &device,
        256,
    )?;
    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    Ok(())
}

#[test]
pub(super) fn test_quantized_roundtrip_accuracy_q4_ks() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q4_KS),
        KvFormat::Quantized(QuantFormat::Q4_KS),
        &device,
        256,
    )?;
    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, 1);
    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, 1)?;
    let k_read = k_read.to_dtype(candle::DType::F32)?;
    let v_read = v_read.to_dtype(candle::DType::F32)?;
    // Q4_KS uses sub-block scales for better 4-bit accuracy than plain Q4_0
    let k_diff = (&k - &k_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let v_diff = (&v - &v_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    assert!(k_diff < 0.1, "K MSE too high for Q4_KS: {}", k_diff);
    assert!(v_diff < 0.1, "V MSE too high for Q4_KS: {}", v_diff);
    Ok(())
}

#[test]
pub(super) fn test_quantized_roundtrip_accuracy_q8_ks() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_KS),
        KvFormat::Quantized(QuantFormat::Q8_KS),
        &device,
        256,
    )?;
    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, 1);
    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, 1)?;
    let k_read = k_read.to_dtype(candle::DType::F32)?;
    let v_read = v_read.to_dtype(candle::DType::F32)?;
    // Q8_KS has sub-block scales: similar accuracy to Q8_0
    let k_diff = (&k - &k_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let v_diff = (&v - &v_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    assert!(k_diff < 0.01, "K MSE too high for Q8_KS: {}", k_diff);
    assert!(v_diff < 0.01, "V MSE too high for Q8_KS: {}", v_diff);
    Ok(())
}

#[test]
pub(super) fn test_quantized_roundtrip_accuracy_q2_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q2_0),
        KvFormat::Quantized(QuantFormat::Q2_0),
        &device,
        256,
    )?;
    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, 1);
    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, 1)?;
    let k_read = k_read.to_dtype(candle::DType::F32)?;
    let v_read = v_read.to_dtype(candle::DType::F32)?;
    // Q2_0 has high quantization error (2 bits), use loose tolerance
    let k_diff = (&k - &k_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let v_diff = (&v - &v_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    assert!(k_diff < 1.0, "K MSE too high for Q2_0: {}", k_diff);
    assert!(v_diff < 1.0, "V MSE too high for Q2_0: {}", v_diff);
    Ok(())
}

#[test]
pub(super) fn test_quantized_roundtrip_accuracy_q3_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q3_0),
        KvFormat::Quantized(QuantFormat::Q3_0),
        &device,
        256,
    )?;
    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, 1);
    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, 1)?;
    let k_read = k_read.to_dtype(candle::DType::F32)?;
    let v_read = v_read.to_dtype(candle::DType::F32)?;
    // Q3_0 has moderate quantization error (3 bits)
    let k_diff = (&k - &k_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let v_diff = (&v - &v_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    assert!(k_diff < 0.5, "K MSE too high for Q3_0: {}", k_diff);
    assert!(v_diff < 0.5, "V MSE too high for Q3_0: {}", v_diff);
    Ok(())
}
