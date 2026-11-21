//! Tests the quantization format and KV format definitions: their names, their
//! equality and conversions, their byte sizes, and their tags.
//!
//! Byte sizes are checked against the exact block layouts, since VRAM accounting
//! rounds to whole blocks and a quantized block does not divide evenly by its element
//! count. The memory ordering across formats is pinned from lowest to highest per
//! element. A format's tag must round-trip through its persisted form, and an unknown
//! tag must decode to nothing rather than to a wrong format.

use super::*;

// ==================== Phase 1: Quantization Format Tests ====================

#[test]
pub(super) fn test_quant_format_q4_0_properties() {
    use crate::kv_cache::QuantFormat;

    let fmt = QuantFormat::Q4_0;
    assert_eq!(fmt.block_size(), 32);
    assert_eq!(fmt.bytes_per_block(), 18); // 16 data + 2 scale
    assert!((fmt.bytes_per_elem() - 0.5625).abs() < 0.001); // 18/32
}

#[test]
pub(super) fn test_quant_format_q8_0_properties() {
    use crate::kv_cache::QuantFormat;

    let fmt = QuantFormat::Q8_0;
    assert_eq!(fmt.block_size(), 32);
    assert_eq!(fmt.bytes_per_block(), 34); // 32 data + 2 scale
    assert!((fmt.bytes_per_elem() - 1.0625).abs() < 0.001); // 34/32
}

#[test]
pub(super) fn test_quant_format_equality() {
    use crate::kv_cache::QuantFormat;

    let q4a = QuantFormat::Q4_0;
    let q4b = QuantFormat::Q4_0;
    let q8 = QuantFormat::Q8_0;

    assert_eq!(q4a, q4b);
    assert_ne!(q4a, q8);
}

#[test]
pub(super) fn test_quant_format_to_ggml_dtype() {
    use crate::kv_cache::QuantFormat;
    use candle::quantized::GgmlDType;

    // Test internal conversion (accessible in tests within crate)
    assert_eq!(QuantFormat::Q4_0.to_ggml_dtype(), GgmlDType::Q4_0);
    assert_eq!(QuantFormat::Q8_0.to_ggml_dtype(), GgmlDType::Q8_0);
}

#[test]
pub(super) fn test_kv_format_float_variants() {
    use crate::kv_cache::KvFormat;
    use candle::DType;

    let f32_fmt = KvFormat::Float(DType::F32);
    let f16_fmt = KvFormat::Float(DType::F16);
    let bf16_fmt = KvFormat::Float(DType::BF16);

    assert!(!f32_fmt.is_quantized());
    assert!(!f16_fmt.is_quantized());
    assert!(!bf16_fmt.is_quantized());

    assert_eq!(f32_fmt.bytes_per_elem(), 4.0);
    assert_eq!(f16_fmt.bytes_per_elem(), 2.0);
    assert_eq!(bf16_fmt.bytes_per_elem(), 2.0);
}

/// `bytes_per_block` is the exact integer counterpart of `bytes_per_elem`, and
/// must agree with the block layouts byte for byte — VRAM accounting rounds to
/// whole blocks, and a quantized block's size does not divide its element count.
#[test]
pub(super) fn test_kv_format_bytes_per_block() {
    use crate::kv_cache::{KvFormat, QuantFormat};
    use candle::DType;

    // Float formats: dtype width across all 32 elements.
    assert_eq!(KvFormat::Float(DType::F32).bytes_per_block(), 128);
    assert_eq!(KvFormat::Float(DType::F16).bytes_per_block(), 64);
    assert_eq!(KvFormat::Float(DType::BF16).bytes_per_block(), 64);

    // Quantized formats delegate to the block layout: d:f16 + 32 nibbles = 18,
    // d:f16 + 32 i8 = 34, and R16's per-element d:f16 + q:u16 = 128.
    assert_eq!(KvFormat::Quantized(QuantFormat::Q4_0).bytes_per_block(), 18);
    assert_eq!(KvFormat::Quantized(QuantFormat::Q8_0).bytes_per_block(), 34);
    assert_eq!(KvFormat::Quantized(QuantFormat::R16).bytes_per_block(), 128);

    // Every format agrees with its own float ratio — the two accessors are the
    // same quantity, so they must not be able to drift apart.
    use strum::IntoEnumIterator;
    let all = [DType::F32, DType::F16, DType::BF16, DType::F8E4M3]
        .into_iter()
        .map(KvFormat::Float)
        .chain(QuantFormat::iter().map(KvFormat::Quantized));
    for fmt in all {
        let ratio = fmt.bytes_per_block() as f32 / CHUNK_SIZE as f32;
        assert!(
            (ratio - fmt.bytes_per_elem()).abs() < 1e-6,
            "{fmt:?}: {ratio} != {}",
            fmt.bytes_per_elem()
        );
    }
}

#[test]
pub(super) fn test_kv_format_quantized_variants() {
    use crate::kv_cache::{KvFormat, QuantFormat};

    let q4_fmt = KvFormat::Quantized(QuantFormat::Q4_0);
    let q8_fmt = KvFormat::Quantized(QuantFormat::Q8_0);

    assert!(q4_fmt.is_quantized());
    assert!(q8_fmt.is_quantized());

    assert!((q4_fmt.bytes_per_elem() - 0.5625).abs() < 0.001);
    assert!((q8_fmt.bytes_per_elem() - 1.0625).abs() < 0.001);
}

#[test]
pub(super) fn test_kv_format_default() {
    use crate::kv_cache::KvFormat;
    use candle::DType;

    let default_fmt = KvFormat::default();
    assert_eq!(default_fmt, KvFormat::Float(DType::BF16));
}

#[test]
pub(super) fn test_kv_format_from_dtype() {
    use crate::kv_cache::KvFormat;
    use candle::DType;

    let fmt: KvFormat = DType::F32.into();
    assert_eq!(fmt, KvFormat::Float(DType::F32));

    let fmt: KvFormat = DType::F16.into();
    assert_eq!(fmt, KvFormat::Float(DType::F16));
}

#[test]
pub(super) fn test_kv_format_from_quant_format() {
    use crate::kv_cache::{KvFormat, QuantFormat};

    let fmt: KvFormat = QuantFormat::Q4_0.into();
    assert_eq!(fmt, KvFormat::Quantized(QuantFormat::Q4_0));

    let fmt: KvFormat = QuantFormat::Q8_0.into();
    assert_eq!(fmt, KvFormat::Quantized(QuantFormat::Q8_0));
}

#[test]
pub(super) fn test_kv_format_equality() {
    use crate::kv_cache::{KvFormat, QuantFormat};
    use candle::DType;

    let f32a = KvFormat::Float(DType::F32);
    let f32b = KvFormat::Float(DType::F32);
    let f16 = KvFormat::Float(DType::F16);
    let q4 = KvFormat::Quantized(QuantFormat::Q4_0);

    assert_eq!(f32a, f32b);
    assert_ne!(f32a, f16);
    assert_ne!(f32a, q4);
}

#[test]
pub(super) fn test_kv_format_memory_comparison() {
    use crate::kv_cache::{KvFormat, QuantFormat};
    use candle::DType;

    // Memory savings comparison
    let f32_bpe = KvFormat::Float(DType::F32).bytes_per_elem();
    let f16_bpe = KvFormat::Float(DType::F16).bytes_per_elem();
    let q8_bpe = KvFormat::Quantized(QuantFormat::Q8_0).bytes_per_elem();
    let q4_bpe = KvFormat::Quantized(QuantFormat::Q4_0).bytes_per_elem();

    // Verify expected savings: Q4 < Q8 < F16 < F32
    assert!(q4_bpe < q8_bpe);
    assert!(q8_bpe < f16_bpe);
    assert!(f16_bpe < f32_bpe);

    // Q4_0 is ~7x smaller than F32
    let q4_savings = f32_bpe / q4_bpe;
    assert!(q4_savings > 7.0);
    assert!(q4_savings < 7.2);

    // Q8_0 is ~3.75x smaller than F32
    let q8_savings = f32_bpe / q8_bpe;
    assert!(q8_savings > 3.7);
    assert!(q8_savings < 3.8);
}

#[test]
pub(super) fn test_kv_format_dtype_method() {
    use crate::kv_cache::{KvFormat, QuantFormat};
    use candle::DType;

    // Float formats return Some(dtype)
    assert_eq!(KvFormat::Float(DType::F32).dtype(), Some(DType::F32));
    assert_eq!(KvFormat::Float(DType::F16).dtype(), Some(DType::F16));
    assert_eq!(KvFormat::Float(DType::BF16).dtype(), Some(DType::BF16));

    // Quantized formats return None
    assert_eq!(KvFormat::Quantized(QuantFormat::Q4_0).dtype(), None);
    assert_eq!(KvFormat::Quantized(QuantFormat::Q8_0).dtype(), None);
}

// ==================== Phase 6: New Quant Format Tests (Q4_KS, Q8_KS, Q2_0, Q3_0) ====================

#[test]
pub(super) fn test_quant_format_q4_ks_properties() {
    use crate::kv_cache::QuantFormat;
    let fmt = QuantFormat::Q4_KS;
    assert_eq!(fmt.block_size(), 32);
    assert_eq!(fmt.bytes_per_block(), 20); // 4-bit nibbles + 2 byte scale + 2 sub-block scales
    assert!((fmt.bytes_per_elem() - 20.0 / 32.0).abs() < 0.001);
}

#[test]
pub(super) fn test_quant_format_q8_ks_properties() {
    use crate::kv_cache::QuantFormat;
    let fmt = QuantFormat::Q8_KS;
    assert_eq!(fmt.block_size(), 32);
    assert_eq!(fmt.bytes_per_block(), 36); // 32 bytes int8 + 2 byte scale + 2 sub-block scales
    assert!((fmt.bytes_per_elem() - 36.0 / 32.0).abs() < 0.001);
}

#[test]
pub(super) fn test_quant_format_q2_0_properties() {
    use crate::kv_cache::QuantFormat;
    let fmt = QuantFormat::Q2_0;
    assert_eq!(fmt.block_size(), 32);
    assert_eq!(fmt.bytes_per_block(), 10); // 2-bit × 32 + 2 byte scale
    assert!((fmt.bytes_per_elem() - 10.0 / 32.0).abs() < 0.001);
}

#[test]
pub(super) fn test_quant_format_q3_0_properties() {
    use crate::kv_cache::QuantFormat;
    let fmt = QuantFormat::Q3_0;
    assert_eq!(fmt.block_size(), 32);
    assert_eq!(fmt.bytes_per_block(), 14); // 3-bit × 32 + 2 byte scale
    assert!((fmt.bytes_per_elem() - 14.0 / 32.0).abs() < 0.001);
}

#[test]
pub(super) fn test_quant_format_to_ggml_dtype_new_types() {
    use crate::kv_cache::QuantFormat;
    use candle::quantized::GgmlDType;

    assert_eq!(QuantFormat::Q4_KS.to_ggml_dtype(), GgmlDType::Q4_KS);
    assert_eq!(QuantFormat::Q8_KS.to_ggml_dtype(), GgmlDType::Q8_KS);
    assert_eq!(QuantFormat::Q2_0.to_ggml_dtype(), GgmlDType::Q2_0);
    assert_eq!(QuantFormat::Q3_0.to_ggml_dtype(), GgmlDType::Q3_0);
}

#[test]
pub(super) fn test_new_quant_formats_memory_ordering() {
    use crate::kv_cache::QuantFormat;

    // Ordering by bytes per element (lowest to highest):
    // Q2_0(10/32) < Q3_0(14/32) < Q4_0(18/32) < Q4_KS(20/32) < Q8_0(34/32) < Q8_KS(36/32)
    // Note: Q4_KS has slightly more bytes than Q4_0 due to sub-block scale overhead
    let q2_bpe = QuantFormat::Q2_0.bytes_per_elem();
    let q3_bpe = QuantFormat::Q3_0.bytes_per_elem();
    let q4_bpe = QuantFormat::Q4_0.bytes_per_elem();
    let q4ks_bpe = QuantFormat::Q4_KS.bytes_per_elem();
    let q8_bpe = QuantFormat::Q8_0.bytes_per_elem();

    assert!(q2_bpe < q3_bpe);
    assert!(q3_bpe < q4_bpe);
    // Q4_KS uses 20 bytes vs Q4_0's 18 bytes (sub-block scales add overhead)
    assert!(q4_bpe < q4ks_bpe);
    assert!(q4ks_bpe < q8_bpe);
}

// ==================== KvFormat tag round-trip ====================

/// `to_tag` / `from_tag` must round-trip for every format
/// `ArenaFormatTag::from_kv_format` accepts — the persistence codec relies on
/// it. `from_tag` derives its answer from `to_tag`, so this also guards the
/// forward mapping against silently producing colliding tags.
#[test]
pub(super) fn kv_format_tag_round_trips() {
    use crate::kv_cache::{KvFormat, QuantFormat};
    use strum::IntoEnumIterator;

    let mut formats: Vec<KvFormat> = [DType::F32, DType::F16, DType::BF16, DType::F8E4M3]
        .into_iter()
        .map(KvFormat::Float)
        .collect();
    formats.extend(QuantFormat::iter().map(KvFormat::Quantized));

    let mut seen_tags = std::collections::HashSet::new();
    for fmt in formats {
        let tag = fmt.to_tag();
        assert!(
            seen_tags.insert(tag),
            "duplicate tag {tag} for {fmt:?} — from_kv_format is not injective"
        );
        assert_eq!(
            KvFormat::from_tag(tag),
            Some(fmt),
            "tag {tag} did not round-trip back to {fmt:?}"
        );
    }
    // An unknown tag decodes to None, not a wrong format.
    assert_eq!(KvFormat::from_tag(254), None);
}
