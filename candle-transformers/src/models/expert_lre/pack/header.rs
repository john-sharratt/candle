//! The pack file's header: what the file claims to be, and the geometry a
//! reader needs to interpret one record without consulting the GGUF.
//!
//! Everything here is plain arithmetic over bytes — no CUDA, no device, no
//! model — so the format is pinned down by unit tests that assert against raw
//! expected bytes rather than against a round trip.
//!
//! # Layout
//!
//! ```text
//! 0    8   magic       b"CNDLXPK1"
//! 8    4   version     u32 LE
//! 12   4   num_layers  u32 LE
//! 16   4   per_layer   u32 LE   experts in each MoE layer
//! 20   4   slot_bytes  u32 LE   the three projections + interior alignment
//! 24   8   stride      u64 LE   slot_bytes padded to a direct-I/O sector
//! 32   8   source_len  u64 LE   the GGUF's length in bytes
//! 40   4   source_sum  u32 LE   fletcher32 over the GGUF's identity sample
//! 44   4   int8_mode   u32 LE   the numeric mode the repack targeted
//! 48   8   repack_fp   u64 LE   fingerprint of the repack formula itself
//! 56   4   pinned      u32 LE   leading MoE layers with NO record in the file
//! 60  ...  per-layer geometry, `num_layers` × 36 bytes
//! ```
//!
//! `num_layers` counts the model's MoE layers and the geometry table describes
//! all of them, so the identity check keeps its full strength. `pinned` says how
//! many of those leading layers the file holds no *records* for: they are
//! permanently VRAM-resident and never reloaded, so storing them would be dead
//! disk. Record `0` is therefore layer `pinned`, expert `0`.
//!
//! Each per-layer record is three projections in gate, up, down order, and
//! each projection is `offset: u32, bytes: u32, dtype: u32` — the dtype being
//! the in-workspace [`GgmlDType`] discriminant, not the GGUF file code.
//!
//! # Why the geometry is stored rather than recomputed
//!
//! The reader always has the GGUF open as well (the dense weights and the
//! embedding table live there), so it *could* recompute all of this. Storing it
//! makes the check exact instead of assumed: a pack written by a build whose
//! repack layout has since changed is caught by comparing 36 bytes per layer,
//! not by trusting a version number to have been bumped.
//!
//! # Why the geometry is not enough
//!
//! Geometry catches a change to *where* the bytes go. It says nothing about a
//! change to *what they are* — a repack kernel that emits a different
//! permutation, or a quantizer whose rounding moves, at identical sizes,
//! offsets and dtypes. That pack would validate and serve subtly wrong weights
//! for the model's entire expert set, with no error anywhere.
//!
//! `repack_fp` closes it, by running the repack over a reference matrix in every
//! quantisation the engine supports and hashing the results together with the
//! list of what was swept. See [`super::fingerprint`].

use candle::quantized::GgmlDType;
use candle::Result;

/// Marks the file as ours and the layout as this one. A change to the record
/// layout changes the last byte rather than adding a compatibility path.
pub(crate) const MAGIC: &[u8; 8] = b"CNDLXPK1";

/// Bytes before the per-layer table.
const FIXED_BYTES: usize = 60;

/// Bytes per layer in the geometry table: three projections × three u32s.
const LAYER_BYTES: usize = 36;

/// Where one projection sits inside a record, how long it is, and what it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProjectionSpan {
    /// Byte offset from the start of the record.
    pub offset: u32,
    /// Bytes of repacked payload — what the H2D copies, excluding any padding
    /// to the next projection's alignment.
    pub bytes: u32,
    /// The repacked form's dtype, as the in-workspace discriminant.
    pub dtype: GgmlDType,
}

/// One MoE layer's three projections. Within a layer every expert has this
/// same shape, which is what lets a record be a fixed stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayerSpans {
    pub gate: ProjectionSpan,
    pub up: ProjectionSpan,
    pub down: ProjectionSpan,
}

/// The header, decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PackHeader {
    pub num_layers: u32,
    pub experts_per_layer: u32,
    /// The slot image size: the three projections at their aligned offsets.
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
    /// Leading MoE layers the file holds **no records** for, because they are
    /// permanently VRAM-resident and never reloaded. Record `0` is layer
    /// `pinned_layers`, expert `0`.
    pub pinned_layers: u32,
    pub layers: Vec<LayerSpans>,
}

impl PackHeader {
    /// Bytes this header occupies before the first record, unpadded.
    pub(crate) fn encoded_len(&self) -> usize {
        FIXED_BYTES + self.layers.len() * LAYER_BYTES
    }

    /// Layers the file holds records for — the evictable set.
    pub(crate) fn stored_layers(&self) -> usize {
        (self.num_layers as usize).saturating_sub(self.pinned_layers as usize)
    }

    /// Total experts the file holds a record for.
    pub(crate) fn total_experts(&self) -> usize {
        self.stored_layers() * self.experts_per_layer as usize
    }

    /// Serialize to exactly [`Self::encoded_len`] bytes.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.encoded_len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.num_layers.to_le_bytes());
        out.extend_from_slice(&self.experts_per_layer.to_le_bytes());
        out.extend_from_slice(&self.slot_bytes.to_le_bytes());
        out.extend_from_slice(&self.stride.to_le_bytes());
        out.extend_from_slice(&self.source_len.to_le_bytes());
        out.extend_from_slice(&self.source_sum.to_le_bytes());
        out.extend_from_slice(&self.int8_mode.to_le_bytes());
        out.extend_from_slice(&self.repack_fp.to_le_bytes());
        out.extend_from_slice(&self.pinned_layers.to_le_bytes());
        for l in &self.layers {
            for p in [l.gate, l.up, l.down] {
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
                "expert pack header is {} bytes, needs at least {FIXED_BYTES}",
                buf.len()
            );
        }
        if &buf[..8] != MAGIC {
            candle::bail!("expert pack magic mismatch: {:?}", &buf[..8]);
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
            candle::bail!("expert pack version {version}, this build writes {VERSION}");
        }
        let num_layers = u32_at(12);
        let need = FIXED_BYTES + num_layers as usize * LAYER_BYTES;
        if buf.len() < need {
            candle::bail!(
                "expert pack header claims {num_layers} layers ({need} bytes) but only {} are present",
                buf.len()
            );
        }
        let mut layers = Vec::with_capacity(num_layers as usize);
        for i in 0..num_layers as usize {
            let base = FIXED_BYTES + i * LAYER_BYTES;
            let span = |k: usize| -> Result<ProjectionSpan> {
                let o = base + k * 12;
                Ok(ProjectionSpan {
                    offset: u32_at(o),
                    bytes: u32_at(o + 4),
                    dtype: GgmlDType::from_u32(u32_at(o + 8))?,
                })
            };
            layers.push(LayerSpans {
                gate: span(0)?,
                up: span(1)?,
                down: span(2)?,
            });
        }
        Ok(Self {
            num_layers,
            experts_per_layer: u32_at(16),
            slot_bytes: u32_at(20),
            stride: u64_at(24),
            source_len: u64_at(32),
            source_sum: u32_at(40),
            int8_mode: u32_at(44),
            repack_fp: u64_at(48),
            pinned_layers: u32_at(56),
            layers,
        })
    }
}

/// Bumped whenever the record layout or the header's own shape changes. There
/// is no reader for an older version — a mismatch rewrites the pack, which
/// costs one repack and nothing else.
pub(crate) const VERSION: u32 = 3;

#[cfg(test)]
mod tests {
    use super::*;

    fn one_layer() -> PackHeader {
        PackHeader {
            num_layers: 1,
            experts_per_layer: 2,
            slot_bytes: 0x300,
            stride: 0x1000,
            source_len: 0x0102_0304_0506_0708,
            source_sum: 0xDEAD_BEEF,
            int8_mode: 3,
            repack_fp: 0x1122_3344_5566_7788,
            pinned_layers: 0,
            layers: vec![LayerSpans {
                gate: ProjectionSpan {
                    offset: 0,
                    bytes: 0x100,
                    dtype: GgmlDType::Q4_K,
                },
                up: ProjectionSpan {
                    offset: 0x100,
                    bytes: 0x100,
                    dtype: GgmlDType::Q4_K,
                },
                down: ProjectionSpan {
                    offset: 0x200,
                    bytes: 0x100,
                    dtype: GgmlDType::Q6_K,
                },
            }],
        }
    }

    /// The exact bytes, field by field. This is the format — a change that
    /// alters any of these without bumping [`VERSION`] would silently make one
    /// build read another's pack as if it agreed.
    #[test]
    fn the_header_encodes_to_these_exact_bytes() {
        let got = one_layer().encode();
        #[rustfmt::skip]
        let want: Vec<u8> = vec![
            b'C', b'N', b'D', b'L', b'X', b'P', b'K', b'1',
            0x03, 0x00, 0x00, 0x00,                          // version 3
            0x01, 0x00, 0x00, 0x00,                          // num_layers 1
            0x02, 0x00, 0x00, 0x00,                          // experts_per_layer 2
            0x00, 0x03, 0x00, 0x00,                          // slot_bytes 0x300
            0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // stride 0x1000
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,  // source_len
            0xEF, 0xBE, 0xAD, 0xDE,                          // source_sum
            0x03, 0x00, 0x00, 0x00,                          // int8_mode 3
            0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,  // repack_fp
            0x00, 0x00, 0x00, 0x00,                          // pinned_layers 0
            // layer 0: gate
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00,
            // layer 0: up
            0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00,
            // layer 0: down
            0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x0B, 0x00, 0x00, 0x00,
        ];
        assert_eq!(got, want);
        assert_eq!(got.len(), one_layer().encoded_len());
    }

    /// Q4_K is workspace discriminant 17 (0x11) and Q6_K is 11 (0x0B) — *not*
    /// the GGUF file codes 12 and 14. The pack stores the in-workspace form
    /// because that is what the repack produced and what the kernels read.
    #[test]
    fn dtypes_are_stored_as_workspace_discriminants() {
        assert_eq!(GgmlDType::Q4_K.to_u32(), 17);
        assert_eq!(GgmlDType::Q6_K.to_u32(), 11);
        let h = one_layer();
        let bytes = h.encode();
        assert_eq!(bytes[FIXED_BYTES + 8], 0x11);
        assert_eq!(bytes[FIXED_BYTES + 2 * 12 + 8], 0x0B);
    }

    #[test]
    fn decode_inverts_encode() {
        let h = one_layer();
        assert_eq!(PackHeader::decode(&h.encode()).unwrap(), h);
    }

    /// `pinned_layers` lands at offset 56, ahead of the geometry table, and
    /// survives the round trip. Its raw position is asserted because a shift
    /// here reads a projection offset as a layer count.
    #[test]
    fn pinned_layers_encodes_at_offset_56() {
        let mut h = one_layer();
        h.num_layers = 3;
        h.pinned_layers = 2;
        h.layers = vec![h.layers[0]; 3];
        let bytes = h.encode();
        assert_eq!(&bytes[56..60], &[0x02, 0x00, 0x00, 0x00]);
        assert_eq!(PackHeader::decode(&bytes).unwrap(), h);
    }

    /// The record count is over the **stored** layers, not the model's. Getting
    /// this wrong sizes the file for experts it does not hold and every read
    /// past the pinned prefix lands one layer early.
    #[test]
    fn record_counts_exclude_the_pinned_prefix() {
        let mut h = one_layer();
        h.num_layers = 40;
        h.experts_per_layer = 256;
        h.pinned_layers = 2;
        assert_eq!(h.stored_layers(), 38);
        assert_eq!(h.total_experts(), 38 * 256);

        // A pack that pins nothing still holds everything.
        h.pinned_layers = 0;
        assert_eq!(h.total_experts(), 40 * 256);

        // Pinning every layer leaves no records at all.
        h.pinned_layers = 40;
        assert_eq!(h.stored_layers(), 0);
        assert_eq!(h.total_experts(), 0);
    }

    /// Trailing bytes past the header are the records; decoding must ignore
    /// them rather than object.
    #[test]
    fn decode_ignores_what_follows_the_table() {
        let h = one_layer();
        let mut buf = h.encode();
        buf.extend_from_slice(&[0xAB; 4096]);
        assert_eq!(PackHeader::decode(&buf).unwrap(), h);
    }

    #[test]
    fn a_foreign_file_is_rejected_by_its_magic() {
        let mut buf = one_layer().encode();
        buf[3] = b'X';
        let e = PackHeader::decode(&buf).unwrap_err().to_string();
        assert!(e.contains("magic mismatch"), "{e}");
    }

    #[test]
    fn a_future_version_is_rejected_rather_than_guessed_at() {
        let mut buf = one_layer().encode();
        buf[8] = VERSION as u8 + 1;
        let e = PackHeader::decode(&buf).unwrap_err().to_string();
        assert!(e.contains(&format!("version {}", VERSION + 1)), "{e}");
    }

    /// A header whose layer count outruns the bytes present is a truncated
    /// file, not a header with fewer layers.
    #[test]
    fn a_truncated_layer_table_is_an_error_not_a_short_read() {
        let h = one_layer();
        let buf = &h.encode()[..FIXED_BYTES + 12];
        let e = PackHeader::decode(buf).unwrap_err().to_string();
        assert!(e.contains("only"), "{e}");
    }

    #[test]
    fn total_experts_multiplies_the_two_counts() {
        let mut h = one_layer();
        h.num_layers = 48;
        h.experts_per_layer = 128;
        assert_eq!(h.total_experts(), 6144);
    }
}
