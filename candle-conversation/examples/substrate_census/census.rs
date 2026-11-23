//! The K/V format census: every live chunk's sub-bands, counted by the format
//! each was sealed in, with the bytes they hold and the bytes the same values
//! would take at BF16.
//!
//! A chunk record holds one 32-token block of every palette sub-band of every
//! KV head, K then V, each in its own format (`ChunkPayload::k_formats`,
//! `v_formats`), packed back to back in `kv_bytes`. Every sub-band of a chunk
//! is the same width, so its blob is `blocks × Σ bytes_per_block(format)`,
//! and `blocks` — the sub-band's 32-element blocks — is read off the blob's
//! length. A chunk whose length does not divide that way is counted apart as
//! unresolved rather than guessed at.

use std::collections::BTreeMap;

use candle_nn::kv_cache::KvFormat;
use serde::Serialize;

/// The bytes one 32-element block takes at BF16.
pub const BF16_BLOCK_BYTES: u64 = 64;

/// One format's share of the census.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct FormatShare {
    /// K sub-bands sealed in it.
    pub k_bands: u64,
    /// V sub-bands sealed in it.
    pub v_bands: u64,
    /// The bytes those sub-bands hold.
    pub bytes: u64,
}

/// The census over every chunk folded in.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
pub struct Census {
    /// Chunks folded in.
    pub chunks: u64,
    /// The tokens they hold.
    pub tokens: u64,
    /// By format name (`Q4_0`, `BF16`, …).
    pub formats: BTreeMap<String, FormatShare>,
    /// The sub-bands' bytes, every format together.
    pub kv_bytes: u64,
    /// The same values at BF16.
    pub bf16_bytes: u64,
    /// Palette maps and outer scales, beside the sub-bands.
    pub side_bytes: u64,
    /// Chunks whose blob does not divide into equal sub-bands, or that name a
    /// format tag no `KvFormat` carries: counted, never sized.
    pub unresolved: u64,
}

/// What the census reads of one chunk record.
pub struct ChunkShape<'a> {
    pub tokens: u64,
    pub k_formats: &'a [u8],
    pub v_formats: &'a [u8],
    pub kv_len: u64,
    pub side_len: u64,
}

impl Census {
    /// Fold one chunk in.
    pub fn add(&mut self, c: &ChunkShape) {
        self.chunks += 1;
        self.tokens += c.tokens;
        self.side_bytes += c.side_len;
        let formats: Option<Vec<KvFormat>> = c
            .k_formats
            .iter()
            .chain(c.v_formats)
            .map(|&t| KvFormat::from_tag(t))
            .collect();
        let Some(formats) = formats else {
            self.unresolved += 1;
            return;
        };
        let per_block: u64 = formats.iter().map(|f| f.bytes_per_block() as u64).sum();
        if per_block == 0 || !c.kv_len.is_multiple_of(per_block) {
            self.unresolved += 1;
            return;
        }
        let blocks = c.kv_len / per_block;
        let n_k = c.k_formats.len();
        for (i, f) in formats.iter().enumerate() {
            let share = self.formats.entry(name(f)).or_default();
            if i < n_k {
                share.k_bands += 1;
            } else {
                share.v_bands += 1;
            }
            share.bytes += blocks * f.bytes_per_block() as u64;
        }
        self.kv_bytes += c.kv_len;
        self.bf16_bytes += blocks * BF16_BLOCK_BYTES * formats.len() as u64;
    }

    /// The BF16 bytes over the bytes stored: the effective compression.
    pub fn ratio(&self) -> f64 {
        self.bf16_bytes as f64 / (self.kv_bytes + self.side_bytes).max(1) as f64
    }
}

/// A format's name as the paper tables it.
pub fn name(f: &KvFormat) -> String {
    match f {
        KvFormat::Float(dt) => format!("{dt:?}").to_uppercase(),
        KvFormat::Quantized(q) => format!("{q:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::DType;
    use candle_nn::kv_cache::QuantFormat;

    fn tag(f: KvFormat) -> u8 {
        f.to_tag()
    }

    /// Two K sub-bands at Q4_0 (18 bytes a block) and two V at Q8_0 (34): 104
    /// bytes a block across the chunk, so a 208-byte blob is two blocks a
    /// sub-band; at BF16 the four would take 4 × 2 × 64 = 512.
    #[test]
    fn a_chunk_is_sized_by_its_formats() {
        let q4 = tag(KvFormat::Quantized(QuantFormat::Q4_0));
        let q8 = tag(KvFormat::Quantized(QuantFormat::Q8_0));
        assert_eq!(KvFormat::Quantized(QuantFormat::Q4_0).bytes_per_block(), 18);
        assert_eq!(KvFormat::Quantized(QuantFormat::Q8_0).bytes_per_block(), 34);
        let mut c = Census::default();
        c.add(&ChunkShape {
            tokens: 32,
            k_formats: &[q4, q4],
            v_formats: &[q8, q8],
            kv_len: 208,
            side_len: 16,
        });
        assert_eq!(
            (c.chunks, c.tokens, c.kv_bytes, c.bf16_bytes),
            (1, 32, 208, 512)
        );
        assert_eq!(
            c.formats["Q4_0"],
            FormatShare {
                k_bands: 2,
                v_bands: 0,
                bytes: 72
            }
        );
        assert_eq!(
            c.formats["Q8_0"],
            FormatShare {
                k_bands: 0,
                v_bands: 2,
                bytes: 136
            }
        );
        assert_eq!(c.side_bytes, 16);
        assert_eq!(c.ratio(), 512.0 / 224.0);
    }

    #[test]
    fn a_blob_that_does_not_divide_is_unresolved() {
        let bf16 = tag(KvFormat::Float(DType::BF16));
        let mut c = Census::default();
        c.add(&ChunkShape {
            tokens: 32,
            k_formats: &[bf16],
            v_formats: &[bf16],
            kv_len: 100,
            side_len: 0,
        });
        c.add(&ChunkShape {
            tokens: 32,
            k_formats: &[250],
            v_formats: &[],
            kv_len: 64,
            side_len: 0,
        });
        assert_eq!((c.chunks, c.unresolved, c.kv_bytes), (2, 2, 0));
        assert!(c.formats.is_empty());
    }

    #[test]
    fn formats_are_named_as_tabled() {
        assert_eq!(name(&KvFormat::Float(DType::BF16)), "BF16");
        assert_eq!(name(&KvFormat::Quantized(QuantFormat::Q5_1)), "Q5_1");
    }
}
