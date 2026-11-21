//! Tests writing and reading quantized KV blocks, from single tokens and sequential
//! appends up to the accuracy of each quantized format on known values.
//!
//! The tests cover writes that cross a chunk boundary, writes for several sequences,
//! and head dimensions that are not a multiple of the block size, which are refused.
//! The float path is checked to be unaffected. Round-trip accuracy is checked for the
//! two quantized formats, with the wider 8-bit format expected to carry less error
//! than the 4-bit one.

use super::*;

// ==================== Phase 4: Quantized KV Write Tests ====================

#[test]
pub(super) fn test_quantized_backing_creation_q8_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let backing = ChunkedKvBacking::new_with_format(
        4,  // initial_batch
        4,  // n_kv_head
        64, // head_dim (must be divisible by 32)
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256, // initial_max_seq_len
    )?;

    assert_eq!(backing.k_format(), KvFormat::Quantized(QuantFormat::Q8_0));
    assert!(backing.is_quantized());
    Ok(())
}

#[test]
pub(super) fn test_quantized_backing_creation_q4_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let backing = ChunkedKvBacking::new_with_format(
        4,
        4,
        64,
        KvFormat::Quantized(QuantFormat::Q4_0),
        KvFormat::Quantized(QuantFormat::Q4_0),
        &device,
        256,
    )?;

    assert_eq!(backing.k_format(), KvFormat::Quantized(QuantFormat::Q4_0));
    Ok(())
}

#[test]
pub(super) fn test_quantized_backing_rejects_bad_head_dim() {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    // head_dim=48 is not divisible by 32
    let result = ChunkedKvBacking::new_with_format(
        4,
        4,
        48,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    );

    assert!(result.is_err());
}

#[test]
pub(super) fn test_quantized_write_single_token_q8_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    )?;

    // Allocate sequence
    let batch_idx = backing.alloc_sequence()?;

    // Create K/V tensors (batch=1, n_kv_head, len=1, head_dim)
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;

    // Write at position 0
    backing.write_contiguous(batch_idx, 0, &k, &v)?;

    // If we get here without error, the write succeeded
    Ok(())
}

#[test]
pub(super) fn test_quantized_write_multiple_tokens_q8_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let seq_len = 10;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    )?;

    let batch_idx = backing.alloc_sequence()?;

    // Write multiple tokens at once
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?;

    backing.write_contiguous(batch_idx, 0, &k, &v)?;

    Ok(())
}

#[test]
pub(super) fn test_quantized_write_q4_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q4_0),
        KvFormat::Quantized(QuantFormat::Q4_0),
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
pub(super) fn test_quantized_write_cross_chunk_boundary() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    )?;

    let batch_idx = backing.alloc_sequence()?;

    // Write tokens that span chunk boundary (CHUNK_SIZE=32, so positions 30-35 span chunks 0 and 1)
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 6, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 6, head_dim), &device)?;

    // Start at position 30, write 6 tokens (30-35)
    backing.write_contiguous(batch_idx, 30, &k, &v)?;

    Ok(())
}

#[test]
pub(super) fn test_quantized_write_sequential_appends() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    )?;

    let batch_idx = backing.alloc_sequence()?;

    // Simulate sequential token generation
    for pos in 0..20 {
        let k = Tensor::full(pos as f32, (1, n_kv_head, 1, head_dim), &device)?;
        let v = Tensor::full((pos as f32) * 2.0, (1, n_kv_head, 1, head_dim), &device)?;
        backing.write_contiguous(batch_idx, pos, &k, &v)?;
    }

    Ok(())
}

#[test]
pub(super) fn test_quantized_write_multiple_sequences() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new_with_format(
        4,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    )?;

    // Allocate multiple sequences
    let idx0 = backing.alloc_sequence()?;
    let idx1 = backing.alloc_sequence()?;
    let idx2 = backing.alloc_sequence()?;

    // Write different amounts to each
    let k0 = Tensor::randn(0f32, 1f32, (1, n_kv_head, 10, head_dim), &device)?;
    let v0 = Tensor::randn(0f32, 1f32, (1, n_kv_head, 10, head_dim), &device)?;
    backing.write_contiguous(idx0, 0, &k0, &v0)?;

    let k1 = Tensor::randn(0f32, 1f32, (1, n_kv_head, 50, head_dim), &device)?;
    let v1 = Tensor::randn(0f32, 1f32, (1, n_kv_head, 50, head_dim), &device)?;
    backing.write_contiguous(idx1, 0, &k1, &v1)?;

    let k2 = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    let v2 = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    backing.write_contiguous(idx2, 0, &k2, &v2)?;

    Ok(())
}

#[test]
pub(super) fn test_quantized_write_larger_head_dim() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 8;
    let head_dim = 128; // Larger head_dim (still divisible by 32)

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        512,
    )?;

    let batch_idx = backing.alloc_sequence()?;

    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 20, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 20, head_dim), &device)?;

    backing.write_contiguous(batch_idx, 0, &k, &v)?;

    Ok(())
}

#[test]
pub(super) fn test_quantized_float_regression() -> Result<()> {
    // Ensure float path still works after changes
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new(4, n_kv_head, head_dim, DType::F32, &device, 256)?;

    assert!(!backing.is_quantized());

    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 10, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 10, head_dim), &device)?;

    backing.write_contiguous(batch_idx, 0, &k, &v)?;

    Ok(())
}

// ==================== Phase 5: Quantized KV Read Tests ====================

#[test]
pub(super) fn test_quantized_read_single_token() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    )?;

    let batch_idx = backing.alloc_sequence()?;

    // Write a single token
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, 1);

    // Read it back
    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, 1)?;

    assert_eq!(k_read.dims(), &[1, n_kv_head, 1, head_dim]);
    assert_eq!(v_read.dims(), &[1, n_kv_head, 1, head_dim]);

    Ok(())
}

#[test]
pub(super) fn test_quantized_read_multiple_tokens() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let seq_len = 10;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    )?;

    let batch_idx = backing.alloc_sequence()?;

    // Write multiple tokens
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, seq_len);

    // Read all back
    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, seq_len)?;

    assert_eq!(k_read.dims(), &[1, n_kv_head, seq_len, head_dim]);
    assert_eq!(v_read.dims(), &[1, n_kv_head, seq_len, head_dim]);

    Ok(())
}

#[test]
pub(super) fn test_quantized_roundtrip_accuracy_q8_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    )?;

    let batch_idx = backing.alloc_sequence()?;

    // Write known values
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, 1);

    // Read back and compare (cast to F32 for comparison since backing may use F16)
    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, 1)?;
    let k_read = k_read.to_dtype(candle::DType::F32)?;
    let v_read = v_read.to_dtype(candle::DType::F32)?;

    // Q8_0 should have reasonable accuracy
    let k_diff = (&k - &k_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let v_diff = (&v - &v_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;

    // MSE should be small for Q8_0 (8-bit quantization)
    assert!(k_diff < 0.01, "K MSE too high: {}", k_diff);
    assert!(v_diff < 0.01, "V MSE too high: {}", v_diff);

    Ok(())
}

#[test]
pub(super) fn test_quantized_roundtrip_accuracy_q4_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q4_0),
        KvFormat::Quantized(QuantFormat::Q4_0),
        &device,
        256,
    )?;

    let batch_idx = backing.alloc_sequence()?;

    // Write known values
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, 1);

    // Read back and compare (cast to F32 for comparison since backing may use F16)
    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, 1)?;
    let k_read = k_read.to_dtype(candle::DType::F32)?;
    let v_read = v_read.to_dtype(candle::DType::F32)?;

    // Q4_0 has more quantization error than Q8_0
    let k_diff = (&k - &k_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let v_diff = (&v - &v_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;

    // MSE should be reasonable for Q4_0 (4-bit quantization has more error)
    assert!(k_diff < 0.1, "K MSE too high for Q4_0: {}", k_diff);
    assert!(v_diff < 0.1, "V MSE too high for Q4_0: {}", v_diff);

    Ok(())
}

#[test]
pub(super) fn test_float_read_contiguous() -> Result<()> {
    // Test that float read path also works
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let seq_len = 5;

    let backing = ChunkedKvBacking::new(1, n_kv_head, head_dim, DType::F32, &device, 256)?;

    let batch_idx = backing.alloc_sequence()?;

    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, seq_len);

    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, seq_len)?;

    // Float should be exact
    let k_diff = (&k - &k_read)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let v_diff = (&v - &v_read)?.abs()?.max_all()?.to_scalar::<f32>()?;

    assert!(k_diff < 1e-5, "K diff too high: {}", k_diff);
    assert!(v_diff < 1e-5, "V diff too high: {}", v_diff);

    Ok(())
}

// ==================== Backing configuration accessors ====================

/// The backing reports the geometry and the *configured* formats. Note what it
/// no longer reports: which arenas are float and which are quantized. An arena
/// is a run of fixed-stride byte slots and holds whatever fits, so that
/// question has no answer (`docs/archived/arena_unification.md` principle 8) — a band's
/// format lives on its chunk.
#[test]
pub(super) fn backing_reports_its_geometry_and_configured_formats() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let float = ChunkedKvBacking::new(1, n_kv_head, head_dim, DType::F32, &device, 256)?;
    assert_eq!(float.n_kv_head(), n_kv_head);
    assert_eq!(float.head_dim(), head_dim);
    assert_eq!(float.k_format(), KvFormat::Float(DType::F32));
    assert!(!float.is_quantized());

    let q8 = KvFormat::Quantized(QuantFormat::Q8_0);
    let quant = ChunkedKvBacking::new_with_format(1, n_kv_head, head_dim, q8, q8, &device, 256)?;
    assert_eq!(quant.k_format(), q8);
    assert!(quant.is_quantized());

    // Geometry drives every size-class question, so it is exposed and pinned:
    // head_dim 64 / N_PALETTE 4 = 16, times CHUNK_SIZE 32.
    assert_eq!(quant.elems_per_chunk(), 32 * 16);

    Ok(())
}
