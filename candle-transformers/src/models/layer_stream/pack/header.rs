//! The layer pack's header: what the file claims to be, and the geometry a
//! reader needs to interpret one record without consulting the GGUF.
//!
//! Plain arithmetic over bytes — no CUDA, no device, no model — so the format
//! is pinned down by unit tests that assert against raw expected bytes rather
//! than against a round trip.
//!
//! # Layout
//!
//! ```text
//! 0    8   magic       b"CNDLLYR1"
//! 8    4   version     u32 LE
//! 12   4   num_layers  u32 LE   trunk layers in the model
//! 16   4   slot_bytes  u32 LE   the largest layer image (§5.1 of the design)
//! 20   8   stride      u64 LE   slot_bytes padded to a direct-I/O sector
//! 28   8   source_len  u64 LE   the GGUF's length in bytes
//! 36   4   source_sum  u32 LE   fletcher32 over the GGUF's identity sample
//! 40   4   int8_mode   u32 LE   the numeric mode the repack targeted
//! 44   8   repack_fp   u64 LE   fingerprint of the repack formula itself
//! 52   4   pinned      u32 LE   leading layers with NO record in the file
//! 56  ...  per-layer geometry, variable width
//! ```
//!
//! Each per-layer entry is `kind: u32, count: u32` followed by `count`
//! projections of `role: u32, offset: u32, bytes: u32, dtype: u32`.
//!
//! # Why the table is variable-width, where the expert pack's is fixed
//!
//! An expert record is always three projections, so its table is a fixed 36
//! bytes per layer. A *layer* record is not: a DeltaNet layer carries six
//! projections and an attention layer seven, and a future mixer could carry a
//! different set again. Padding every entry to the widest kind would encode
//! today's maximum into the format — the thing a version number then has to be
//! remembered for. Reading `count` costs one `u32` and removes the question.
//!
//! # `num_layers` counts the whole trunk; `pinned` says what is absent
//!
//! The geometry table describes **every** layer, so the identity check keeps
//! its full strength. `pinned` says how many leading layers the file holds no
//! *records* for: they are permanently VRAM-resident and never reloaded, so
//! storing them would be dead disk. Record `0` is therefore layer `pinned`.

use candle::quantized::GgmlDType;
use candle::Result;

use crate::models::layer_stream::descriptor::{LayerTensor, MixKind};

/// Marks the file as ours and the layout as this one. A change to the record
/// layout changes the last byte rather than adding a compatibility path.
pub(crate) const MAGIC: &[u8; 8] = b"CNDLLYR1";

/// Bumped whenever the record layout or the header's own shape changes. There
/// is no reader for an older version — a mismatch rewrites the pack, which
/// costs one repack and nothing else.
pub(crate) const VERSION: u32 = 1;

/// Bytes before the per-layer table.
const FIXED_BYTES: usize = 56;

/// Bytes per projection in the geometry table.
const PROJECTION_BYTES: usize = 16;

/// Bytes of per-layer preamble: `kind` and `count`.
const LAYER_PREAMBLE: usize = 8;

/// Where one projection sits inside a record, how long it is, and what it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProjectionSpan {
    pub role: LayerTensor,
    /// Byte offset from the start of the record.
    pub offset: u32,
    /// Bytes of repacked payload — what the H2D copies, excluding any padding
    /// to the next projection's alignment.
    pub bytes: u32,
    /// The repacked form's dtype, as the in-workspace discriminant.
    pub dtype: GgmlDType,
}

/// One trunk layer's projections, in image order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LayerSpans {
    pub kind: MixKind,
    pub projections: Vec<ProjectionSpan>,
}

impl LayerSpans {
    fn encoded_len(&self) -> usize {
        LAYER_PREAMBLE + self.projections.len() * PROJECTION_BYTES
    }
}

/// The header, decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PackHeader {
    pub num_layers: u32,
    /// The slot image size: the largest layer's projections at their aligned
    /// offsets. Every record is this wide, whatever kind it holds.
    pub slot_bytes: u32,
    /// Record-to-record distance in the file — `slot_bytes` rounded up to a
    /// direct-I/O sector, so every record offset is legal to `pread` at.
    pub stride: u64,
    pub source_len: u64,
    pub source_sum: u32,
    pub int8_mode: u32,
    /// Fingerprint of the repack formula — what the repack *produces* over a
    /// reference sweep, as opposed to the geometry, which is only where it puts
    /// the results.
    pub repack_fp: u64,
    /// Leading layers the file holds **no records** for. Record `0` is layer
    /// `pinned_layers`.
    pub pinned_layers: u32,
    pub layers: Vec<LayerSpans>,
}

impl PackHeader {
    /// Bytes this header occupies before the first record, unpadded.
    pub(crate) fn encoded_len(&self) -> usize {
        FIXED_BYTES + self.layers.iter().map(|l| l.encoded_len()).sum::<usize>()
    }

    /// Layers the file holds records for — the evictable set.
    pub(crate) fn stored_layers(&self) -> usize {
        (self.num_layers as usize).saturating_sub(self.pinned_layers as usize)
    }

    /// The record index of trunk layer `layer`, or an error when the layer is
    /// pinned and so has no record.
    ///
    /// A `Result` rather than an `Option` because the pinned case is a caller
    /// bug — asking the cold tier for a layer that by construction never leaves
    /// VRAM — and the message should say so at the point it happens.
    pub(crate) fn record_index(&self, layer: usize) -> Result<usize> {
        let pinned = self.pinned_layers as usize;
        if layer < pinned {
            candle::bail!(
                "layer pack: layer {layer} is inside the pinned prefix ({pinned} layers), \
                 which the file holds no record for — it is resident for the life of the \
                 process and is never reloaded"
            );
        }
        if layer >= self.num_layers as usize {
            candle::bail!(
                "layer pack: layer {layer} is past the model's {} layers",
                self.num_layers
            );
        }
        Ok(layer - pinned)
    }

    /// Serialize to exactly [`Self::encoded_len`] bytes.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.encoded_len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.num_layers.to_le_bytes());
        out.extend_from_slice(&self.slot_bytes.to_le_bytes());
        out.extend_from_slice(&self.stride.to_le_bytes());
        out.extend_from_slice(&self.source_len.to_le_bytes());
        out.extend_from_slice(&self.source_sum.to_le_bytes());
        out.extend_from_slice(&self.int8_mode.to_le_bytes());
        out.extend_from_slice(&self.repack_fp.to_le_bytes());
        out.extend_from_slice(&self.pinned_layers.to_le_bytes());
        for l in &self.layers {
            out.extend_from_slice(&l.kind.to_u32().to_le_bytes());
            out.extend_from_slice(&(l.projections.len() as u32).to_le_bytes());
            for p in &l.projections {
                out.extend_from_slice(&p.role.to_u32().to_le_bytes());
                out.extend_from_slice(&p.offset.to_le_bytes());
                out.extend_from_slice(&p.bytes.to_le_bytes());
                out.extend_from_slice(&p.dtype.to_u32().to_le_bytes());
            }
        }
        out
    }

    /// Decode from the head of `buf`.
    ///
    /// Fails rather than truncates on a short or unrecognised buffer: a pack
    /// whose header does not parse is a pack to rewrite, and the caller treats
    /// every error here the same way.
    pub(crate) fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < FIXED_BYTES {
            candle::bail!(
                "layer pack header is {} bytes, needs at least {FIXED_BYTES}",
                buf.len()
            );
        }
        if &buf[..8] != MAGIC {
            candle::bail!("layer pack magic mismatch: {:?}", &buf[..8]);
        }
        let u32_at = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
        let u64_at = |o: usize| {
            u64::from_le_bytes([
                buf[o],
                buf[o + 1],
                buf[o + 2],
                buf[o + 3],
                buf[o + 4],
                buf[o + 5],
                buf[o + 6],
                buf[o + 7],
            ])
        };
        let version = u32_at(8);
        if version != VERSION {
            candle::bail!("layer pack version {version}, this build writes {VERSION}");
        }
        let num_layers = u32_at(12);
        // **Bound the count before reserving against it.** The per-layer walk
        // below checks `buf.len()` at every step, but a `with_capacity` ahead of
        // the loop is reached first — and a corrupt count (bit rot in these four
        // bytes reads as ~4.29e9 layers) asks the allocator for ~137 GiB, which
        // is `handle_alloc_error` and a non-unwinding abort, not the `Err` that
        // makes the caller rebuild the pack. Every layer costs at least
        // `LAYER_PREAMBLE`, so the buffer's own length is the ceiling.
        let max_layers = buf.len().saturating_sub(FIXED_BYTES) / LAYER_PREAMBLE;
        if num_layers as usize > max_layers {
            candle::bail!(
                "layer pack header claims {num_layers} layers, but {} bytes hold at most {max_layers}",
                buf.len()
            );
        }

        let mut layers = Vec::with_capacity(num_layers as usize);
        let mut at = FIXED_BYTES;
        for i in 0..num_layers as usize {
            if buf.len() < at + LAYER_PREAMBLE {
                candle::bail!(
                    "layer pack header claims {num_layers} layers but ends inside layer {i}"
                );
            }
            let kind = MixKind::from_u32(u32_at(at)).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "layer pack: layer {i} names mixer kind {}, which this build does not know",
                    u32_at(at)
                ))
            })?;
            let count = u32_at(at + 4) as usize;
            at += LAYER_PREAMBLE;
            if buf.len() < at + count * PROJECTION_BYTES {
                candle::bail!(
                    "layer pack header claims {count} projections for layer {i} but ends inside them"
                );
            }
            let mut projections = Vec::with_capacity(count);
            for k in 0..count {
                let o = at + k * PROJECTION_BYTES;
                let role = LayerTensor::from_u32(u32_at(o)).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "layer pack: layer {i} projection {k} names role {}, which this build \
                         does not know",
                        u32_at(o)
                    ))
                })?;
                projections.push(ProjectionSpan {
                    role,
                    offset: u32_at(o + 4),
                    bytes: u32_at(o + 8),
                    dtype: GgmlDType::from_u32(u32_at(o + 12))?,
                });
            }
            at += count * PROJECTION_BYTES;
            layers.push(LayerSpans { kind, projections });
        }

        Ok(Self {
            num_layers,
            slot_bytes: u32_at(16),
            stride: u64_at(20),
            source_len: u64_at(28),
            source_sum: u32_at(36),
            int8_mode: u32_at(40),
            repack_fp: u64_at(44),
            pinned_layers: u32_at(52),
            layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dn_layer() -> LayerSpans {
        LayerSpans {
            kind: MixKind::DeltaNet,
            projections: vec![
                ProjectionSpan {
                    role: LayerTensor::Wqkv,
                    offset: 0,
                    bytes: 0x100,
                    dtype: GgmlDType::Q4_K,
                },
                ProjectionSpan {
                    role: LayerTensor::Wz,
                    offset: 0x100,
                    bytes: 0x80,
                    dtype: GgmlDType::Q6_K,
                },
            ],
        }
    }

    fn attn_layer() -> LayerSpans {
        LayerSpans {
            kind: MixKind::Attention,
            projections: vec![ProjectionSpan {
                role: LayerTensor::Wq,
                offset: 0,
                bytes: 0x200,
                dtype: GgmlDType::Q4_K,
            }],
        }
    }

    fn two_layers() -> PackHeader {
        PackHeader {
            num_layers: 2,
            slot_bytes: 0x300,
            stride: 0x1000,
            source_len: 0x0102_0304_0506_0708,
            source_sum: 0xDEAD_BEEF,
            int8_mode: 3,
            repack_fp: 0x1122_3344_5566_7788,
            pinned_layers: 0,
            layers: vec![dn_layer(), attn_layer()],
        }
    }

    #[test]
    fn the_fixed_prefix_is_exact_bytes() {
        let bytes = two_layers().encode();
        // Magic, then version 1, then num_layers 2 — asserted as raw bytes so a
        // field that moves is caught here rather than by a round trip that
        // agrees with itself.
        assert_eq!(&bytes[0..8], b"CNDLLYR1");
        assert_eq!(&bytes[8..12], &1u32.to_le_bytes());
        assert_eq!(&bytes[12..16], &2u32.to_le_bytes());
        assert_eq!(&bytes[16..20], &0x300u32.to_le_bytes());
        assert_eq!(&bytes[20..28], &0x1000u64.to_le_bytes());
        assert_eq!(&bytes[28..36], &0x0102_0304_0506_0708u64.to_le_bytes());
        assert_eq!(&bytes[36..40], &0xDEAD_BEEFu32.to_le_bytes());
        assert_eq!(&bytes[40..44], &3u32.to_le_bytes());
        assert_eq!(&bytes[44..52], &0x1122_3344_5566_7788u64.to_le_bytes());
        assert_eq!(&bytes[52..56], &0u32.to_le_bytes());
    }

    #[test]
    fn a_layer_entry_is_kind_count_then_projections() {
        let bytes = two_layers().encode();
        // Layer 0: DeltaNet (0), two projections.
        assert_eq!(&bytes[56..60], &0u32.to_le_bytes());
        assert_eq!(&bytes[60..64], &2u32.to_le_bytes());
        // First projection: role Wqkv (0), offset 0, bytes 0x100, dtype Q4_K.
        assert_eq!(&bytes[64..68], &0u32.to_le_bytes());
        assert_eq!(&bytes[68..72], &0u32.to_le_bytes());
        assert_eq!(&bytes[72..76], &0x100u32.to_le_bytes());
        assert_eq!(&bytes[76..80], &GgmlDType::Q4_K.to_u32().to_le_bytes());
        // Second projection: role Wz (1).
        assert_eq!(&bytes[80..84], &1u32.to_le_bytes());
        // Layer 1 begins right after: Attention (1), one projection.
        assert_eq!(&bytes[96..100], &1u32.to_le_bytes());
        assert_eq!(&bytes[100..104], &1u32.to_le_bytes());
    }

    #[test]
    fn encoded_len_matches_what_encode_produces() {
        let h = two_layers();
        assert_eq!(h.encode().len(), h.encoded_len());
        // 56 fixed + (8 + 2×16) + (8 + 1×16) = 56 + 40 + 24 = 120
        assert_eq!(h.encoded_len(), 120);
    }

    #[test]
    fn a_header_round_trips() {
        let h = two_layers();
        assert_eq!(PackHeader::decode(&h.encode()).unwrap(), h);
    }

    #[test]
    fn variable_width_layers_decode_at_the_right_offsets() {
        // The point of the variable-width table: a six-projection layer and a
        // seven-projection layer in one file, and the second still parses.
        let mut h = two_layers();
        h.layers[0].projections.extend([ProjectionSpan {
            role: LayerTensor::FfnDown,
            offset: 0x180,
            bytes: 0x40,
            dtype: GgmlDType::Q4_K,
        }]);
        let back = PackHeader::decode(&h.encode()).unwrap();
        assert_eq!(back, h);
        assert_eq!(back.layers[0].projections.len(), 3);
        assert_eq!(back.layers[1].projections.len(), 1);
        assert_eq!(back.layers[1].kind, MixKind::Attention);
    }

    #[test]
    fn a_short_buffer_is_refused_not_truncated() {
        let bytes = two_layers().encode();
        for cut in [0, 8, 40, 55, 60, 90, 119] {
            assert!(
                PackHeader::decode(&bytes[..cut]).is_err(),
                "a {cut}-byte header must be refused"
            );
        }
        assert!(PackHeader::decode(&bytes).is_ok());
    }

    #[test]
    fn a_foreign_magic_is_refused() {
        let mut bytes = two_layers().encode();
        bytes[7] = b'2';
        let err = PackHeader::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("magic mismatch"), "{err}");
    }

    #[test]
    fn an_older_version_is_refused_rather_than_adapted_to() {
        let mut bytes = two_layers().encode();
        bytes[8..12].copy_from_slice(&0u32.to_le_bytes());
        let err = PackHeader::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("version 0"), "{err}");
    }

    #[test]
    fn an_unknown_role_names_itself() {
        let mut bytes = two_layers().encode();
        bytes[64..68].copy_from_slice(&999u32.to_le_bytes());
        let err = PackHeader::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("role 999"), "{err}");
    }

    #[test]
    fn an_absurd_layer_count_is_an_error_not_an_allocation() {
        // The per-layer walk bounds itself against `buf.len()`, but the
        // `with_capacity` ahead of it is reached first: a corrupt count asks the
        // allocator for ~137 GiB, and `handle_alloc_error` aborts the process
        // instead of returning the `Err` that rebuilds the pack. If this
        // regresses the test does not fail — the runner dies.
        let mut bytes = two_layers().encode();
        bytes[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        let err = PackHeader::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("at most"), "{err}");
    }

    #[test]
    fn an_unknown_mixer_kind_names_itself() {
        let mut bytes = two_layers().encode();
        bytes[56..60].copy_from_slice(&7u32.to_le_bytes());
        let err = PackHeader::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("mixer kind 7"), "{err}");
    }

    #[test]
    fn record_indices_skip_the_pinned_prefix() {
        let mut h = two_layers();
        h.num_layers = 6;
        h.pinned_layers = 2;
        assert_eq!(h.stored_layers(), 4);
        assert_eq!(h.record_index(2).unwrap(), 0);
        assert_eq!(h.record_index(5).unwrap(), 3);
    }

    #[test]
    fn a_pinned_layer_has_no_record_and_says_so() {
        let mut h = two_layers();
        h.num_layers = 6;
        h.pinned_layers = 2;
        for pinned in [0, 1] {
            let err = h.record_index(pinned).unwrap_err().to_string();
            assert!(err.contains("inside the pinned prefix"), "{err}");
        }
        let err = h.record_index(6).unwrap_err().to_string();
        assert!(err.contains("past the model's 6 layers"), "{err}");
    }
}
