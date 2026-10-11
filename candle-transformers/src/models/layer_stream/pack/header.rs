//! The layer section's header: what the section claims to be, the geometry a
//! reader rebuilds every layer's image from, and the fingerprint of every
//! repack that produced its bytes.
//!
//! Plain arithmetic over bytes — no CUDA, no device, no model — so the format
//! is pinned down by unit tests that assert against raw expected bytes rather
//! than against a round trip.
//!
//! # Layout
//!
//! ```text
//! 0    8   magic       b"CNDLLYR2"
//! 8    4   version     u32 LE
//! 12   4   num_layers  u32 LE   trunk layers, every one of which has a record
//! 16   4   slot_bytes  u32 LE   the largest layer image (§5.1 of the design)
//! 20   8   stride      u64 LE   slot_bytes padded to a direct-I/O sector
//! 28   4   int8_mode   u32 LE   the numeric mode the repack targeted
//! 32   4   num_pairs   u32 LE   entries in the fingerprint table
//! 36  ...  per-layer geometry, variable width
//! ...  ... fingerprint table, `num_pairs` × 16 bytes
//! ```
//!
//! Each per-layer entry is `kind, ffn, count` (`u32`s) followed by `count`
//! projections of `role, offset, bytes, extent, dtype, src_dtype, rows, cols`
//! (`u32`s). A fingerprint entry is `src_dtype: u32, dtype: u32, fp: u64`, one
//! per distinct pair the projections use.
//!
//! # Why the table is variable-width, where the expert section's is fixed
//!
//! An expert record is always three projections. A *layer* record is not: a
//! DeltaNet layer carries five or six projections and an attention layer six or
//! seven, and a future mixer could carry a different set again. Reading `count`
//! costs one `u32` and removes the question.
//!
//! # Every layer has a record
//!
//! The section is the model's only copy of its streamed projections, so the
//! permanently resident head is stored like the rest and uploaded from here.

use candle::quantized::GgmlDType;
use candle::Result;

use crate::models::layer_stream::descriptor::{FfnForm, LayerTensor, MixKind};
use crate::models::repack_fingerprint::PairPrint;

/// Marks the section as ours and the layout as this one. A change to the record
/// layout changes the last byte rather than adding a compatibility path.
pub(crate) const MAGIC: &[u8; 8] = b"CNDLLYR2";

/// Bumped whenever the record layout or the header's own shape changes. There
/// is no reader for an older version — a mismatch rebuilds the model pack.
pub(crate) const VERSION: u32 = 2;

/// Bytes before the per-layer table.
pub(crate) const FIXED_BYTES: usize = 36;

/// Bytes per projection in the geometry table: eight `u32`s.
const PROJECTION_BYTES: usize = 32;

/// Bytes of per-layer preamble: `kind`, `ffn` and `count`.
const LAYER_PREAMBLE: usize = 12;

/// Bytes per fingerprint entry.
const PAIR_BYTES: usize = 16;

/// Where one projection sits inside a record, what it is, and what it was
/// repacked from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProjectionSpan {
    pub role: LayerTensor,
    /// Byte offset from the start of the record.
    pub offset: u32,
    /// Bytes of payload — what the H2D copies.
    pub bytes: u32,
    /// Bytes the slot reserves from `offset`, which the kernel may address.
    pub extent: u32,
    /// The slot's dtype: a KO twin, or the source quant where none was taken.
    pub dtype: GgmlDType,
    /// The checkpoint's dtype for this projection.
    pub src_dtype: GgmlDType,
    pub rows: u32,
    pub cols: u32,
}

/// One trunk layer's projections, in image order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LayerSpans {
    pub kind: MixKind,
    pub ffn: FfnForm,
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
    /// Record-to-record distance — `slot_bytes` rounded up to a direct-I/O
    /// sector, so every record offset is legal to `pread` at.
    pub stride: u64,
    pub int8_mode: u32,
    pub layers: Vec<LayerSpans>,
    /// One entry per repack pair the projections use, in [`pairs_in`] order.
    pub pairs: Vec<PairPrint>,
}

/// The distinct `(src_dtype, dtype)` pairs `layers` use, sorted — the pairs a
/// fingerprint table must cover, read off the geometry.
pub(crate) fn pairs_in(layers: &[LayerSpans]) -> Vec<(GgmlDType, GgmlDType)> {
    let mut pairs: Vec<(GgmlDType, GgmlDType)> = layers
        .iter()
        .flat_map(|l| l.projections.iter())
        .map(|p| (p.src_dtype, p.dtype))
        .collect();
    pairs.sort_by_key(|&(s, d)| (s.to_u32(), d.to_u32()));
    pairs.dedup();
    pairs
}

impl PackHeader {
    /// Bytes this header occupies before the first record, unpadded.
    pub(crate) fn encoded_len(&self) -> usize {
        FIXED_BYTES
            + self.layers.iter().map(|l| l.encoded_len()).sum::<usize>()
            + self.pairs.len() * PAIR_BYTES
    }

    /// Serialize to exactly [`Self::encoded_len`] bytes.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.encoded_len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.num_layers.to_le_bytes());
        out.extend_from_slice(&self.slot_bytes.to_le_bytes());
        out.extend_from_slice(&self.stride.to_le_bytes());
        out.extend_from_slice(&self.int8_mode.to_le_bytes());
        out.extend_from_slice(&(self.pairs.len() as u32).to_le_bytes());
        for l in &self.layers {
            for v in [l.kind.to_u32(), l.ffn.to_u32(), l.projections.len() as u32] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            for p in &l.projections {
                for v in [
                    p.role.to_u32(),
                    p.offset,
                    p.bytes,
                    p.extent,
                    p.dtype.to_u32(),
                    p.src_dtype.to_u32(),
                    p.rows,
                    p.cols,
                ] {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        for p in &self.pairs {
            out.extend_from_slice(&p.src_dtype.to_u32().to_le_bytes());
            out.extend_from_slice(&p.dtype.to_u32().to_le_bytes());
            out.extend_from_slice(&p.fp.to_le_bytes());
        }
        out
    }

    /// Decode from the head of `buf`.
    ///
    /// Fails rather than truncates on a short or unrecognised buffer, and on a
    /// fingerprint table that does not cover exactly the pairs its own layers
    /// use.
    pub(crate) fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < FIXED_BYTES {
            candle::bail!(
                "layer section header is {} bytes, needs at least {FIXED_BYTES}",
                buf.len()
            );
        }
        if &buf[..8] != MAGIC {
            candle::bail!("layer section magic mismatch: {:?}", &buf[..8]);
        }
        let u32_at = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
        let u64_at = |o: usize| {
            let mut b = [0u8; 8];
            b.copy_from_slice(&buf[o..o + 8]);
            u64::from_le_bytes(b)
        };
        let version = u32_at(8);
        if version != VERSION {
            candle::bail!("layer section version {version}, this build writes {VERSION}");
        }
        let num_layers = u32_at(12) as usize;
        let num_pairs = u32_at(32) as usize;
        // **Bound the counts before reserving against them.** A corrupt count
        // (bit rot in four bytes reads as ~4.29e9 layers) would otherwise ask the
        // allocator for tens of GiB — a non-unwinding abort, not the `Err` that
        // rebuilds the pack. Every layer costs at least its preamble, every pair
        // its entry, so the buffer's own length is the ceiling.
        let room = buf.len() - FIXED_BYTES;
        if num_layers > room / LAYER_PREAMBLE || num_pairs > room / PAIR_BYTES {
            candle::bail!(
                "layer section header claims {num_layers} layers and {num_pairs} fingerprints, \
                 more than {} bytes can hold",
                buf.len()
            );
        }

        let mut layers = Vec::with_capacity(num_layers);
        let mut at = FIXED_BYTES;
        for i in 0..num_layers {
            if buf.len() < at + LAYER_PREAMBLE {
                candle::bail!("layer section header ends inside layer {i}");
            }
            let kind = MixKind::from_u32(u32_at(at)).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "layer section: layer {i} names mixer kind {}, which this build does not know",
                    u32_at(at)
                ))
            })?;
            let ffn = FfnForm::from_u32(u32_at(at + 4)).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "layer section: layer {i} names FFN form {}, which this build does not know",
                    u32_at(at + 4)
                ))
            })?;
            let count = u32_at(at + 8) as usize;
            at += LAYER_PREAMBLE;
            if buf.len() < at + count * PROJECTION_BYTES {
                candle::bail!("layer section header ends inside layer {i}'s projections");
            }
            let mut projections = Vec::with_capacity(count);
            for k in 0..count {
                let o = at + k * PROJECTION_BYTES;
                let role = LayerTensor::from_u32(u32_at(o)).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "layer section: layer {i} projection {k} names role {}, which this \
                         build does not know",
                        u32_at(o)
                    ))
                })?;
                projections.push(ProjectionSpan {
                    role,
                    offset: u32_at(o + 4),
                    bytes: u32_at(o + 8),
                    extent: u32_at(o + 12),
                    dtype: GgmlDType::from_u32(u32_at(o + 16))?,
                    src_dtype: GgmlDType::from_u32(u32_at(o + 20))?,
                    rows: u32_at(o + 24),
                    cols: u32_at(o + 28),
                });
            }
            at += count * PROJECTION_BYTES;
            layers.push(LayerSpans {
                kind,
                ffn,
                projections,
            });
        }
        if buf.len() < at + num_pairs * PAIR_BYTES {
            candle::bail!("layer section header ends inside its fingerprint table");
        }
        let mut pairs = Vec::with_capacity(num_pairs);
        for i in 0..num_pairs {
            let o = at + i * PAIR_BYTES;
            pairs.push(PairPrint {
                src_dtype: GgmlDType::from_u32(u32_at(o))?,
                dtype: GgmlDType::from_u32(u32_at(o + 4))?,
                fp: u64_at(o + 8),
            });
        }
        let covered: Vec<(GgmlDType, GgmlDType)> =
            pairs.iter().map(|p| (p.src_dtype, p.dtype)).collect();
        if covered != pairs_in(&layers) {
            candle::bail!(
                "layer section fingerprints cover {covered:?} but its layers use {:?}",
                pairs_in(&layers)
            );
        }
        Ok(Self {
            num_layers: num_layers as u32,
            slot_bytes: u32_at(16),
            stride: u64_at(20),
            int8_mode: u32_at(28),
            layers,
            pairs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proj(role: LayerTensor, offset: u32, src: GgmlDType) -> ProjectionSpan {
        ProjectionSpan {
            role,
            offset,
            bytes: 0x100,
            extent: 0x100,
            dtype: GgmlDType::Q4_KO,
            src_dtype: src,
            rows: 8,
            cols: 128,
        }
    }

    fn two_layers() -> PackHeader {
        let layers = vec![
            LayerSpans {
                kind: MixKind::DeltaNet,
                ffn: FfnForm::Fused,
                projections: vec![
                    proj(LayerTensor::Wqkv, 0, GgmlDType::Q4_K),
                    proj(LayerTensor::Wz, 0x100, GgmlDType::Q6_K),
                ],
            },
            LayerSpans {
                kind: MixKind::Attention,
                ffn: FfnForm::Split,
                projections: vec![proj(LayerTensor::Wq, 0, GgmlDType::Q4_K)],
            },
        ];
        let pairs = pairs_in(&layers)
            .into_iter()
            .enumerate()
            .map(|(i, (src_dtype, dtype))| PairPrint {
                src_dtype,
                dtype,
                fp: 0x1000 + i as u64,
            })
            .collect();
        PackHeader {
            num_layers: 2,
            slot_bytes: 0x300,
            stride: 0x1000,
            int8_mode: 3,
            layers,
            pairs,
        }
    }

    /// The fixed prefix, as raw bytes — a field that moves is caught here
    /// rather than by a round trip that agrees with itself.
    #[test]
    fn the_fixed_prefix_is_exact_bytes() {
        let bytes = two_layers().encode();
        assert_eq!(&bytes[0..8], b"CNDLLYR2");
        assert_eq!(&bytes[8..12], &2u32.to_le_bytes());
        assert_eq!(&bytes[12..16], &2u32.to_le_bytes());
        assert_eq!(&bytes[16..20], &0x300u32.to_le_bytes());
        assert_eq!(&bytes[20..28], &0x1000u64.to_le_bytes());
        assert_eq!(&bytes[28..32], &3u32.to_le_bytes());
        assert_eq!(&bytes[32..36], &2u32.to_le_bytes());
    }

    #[test]
    fn a_layer_entry_is_kind_ffn_count_then_projections() {
        let bytes = two_layers().encode();
        // Layer 0: DeltaNet (0), fused (0), two projections.
        assert_eq!(&bytes[36..40], &0u32.to_le_bytes());
        assert_eq!(&bytes[40..44], &0u32.to_le_bytes());
        assert_eq!(&bytes[44..48], &2u32.to_le_bytes());
        // First projection: role Wqkv (0), offset 0, bytes, extent, dtype, src.
        assert_eq!(&bytes[48..52], &0u32.to_le_bytes());
        assert_eq!(&bytes[52..56], &0u32.to_le_bytes());
        assert_eq!(&bytes[56..60], &0x100u32.to_le_bytes());
        assert_eq!(&bytes[60..64], &0x100u32.to_le_bytes());
        assert_eq!(&bytes[64..68], &GgmlDType::Q4_KO.to_u32().to_le_bytes());
        assert_eq!(&bytes[68..72], &GgmlDType::Q4_K.to_u32().to_le_bytes());
        assert_eq!(&bytes[72..76], &8u32.to_le_bytes());
        assert_eq!(&bytes[76..80], &128u32.to_le_bytes());
        // Layer 1 begins after two 32-byte projections: Attention (1), split (1).
        assert_eq!(&bytes[112..116], &1u32.to_le_bytes());
        assert_eq!(&bytes[116..120], &1u32.to_le_bytes());
    }

    #[test]
    fn encoded_len_matches_what_encode_produces() {
        let h = two_layers();
        assert_eq!(h.encode().len(), h.encoded_len());
        // 36 fixed + (12 + 2×32) + (12 + 1×32) + 2×16 = 36 + 76 + 44 + 32
        assert_eq!(h.encoded_len(), 188);
    }

    #[test]
    fn a_header_round_trips() {
        let h = two_layers();
        assert_eq!(PackHeader::decode(&h.encode()).unwrap(), h);
    }

    #[test]
    fn a_short_buffer_is_refused_not_truncated() {
        let bytes = two_layers().encode();
        for cut in [0, 8, 35, 40, 60, 120, 187] {
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
        bytes[7] = b'1';
        let err = PackHeader::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("magic mismatch"), "{err}");
    }

    #[test]
    fn another_version_is_refused_rather_than_adapted_to() {
        let mut bytes = two_layers().encode();
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        let err = PackHeader::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("version 1"), "{err}");
    }

    #[test]
    fn an_unknown_role_names_itself() {
        let mut bytes = two_layers().encode();
        bytes[48..52].copy_from_slice(&999u32.to_le_bytes());
        let err = PackHeader::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("role 999"), "{err}");
    }

    #[test]
    fn an_absurd_layer_count_is_an_error_not_an_allocation() {
        let mut bytes = two_layers().encode();
        bytes[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        let err = PackHeader::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("more than"), "{err}");
    }

    /// A table that leaves out a pair its projections use is refused.
    #[test]
    fn a_table_missing_a_pair_is_refused() {
        let mut h = two_layers();
        h.pairs.pop();
        let err = PackHeader::decode(&h.encode()).unwrap_err().to_string();
        assert!(err.contains("fingerprints cover"), "{err}");
    }
}
