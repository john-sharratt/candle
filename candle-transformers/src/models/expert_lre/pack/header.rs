//! The expert section's header: what the section claims to be, the geometry a
//! reader needs to interpret one record, and the fingerprint of every repack
//! that produced its bytes.
//!
//! Everything here is plain arithmetic over bytes — no CUDA, no device, no
//! model — so the format is pinned down by unit tests that assert against raw
//! expected bytes rather than against a round trip.
//!
//! # Layout
//!
//! ```text
//! 0    8   magic       b"CNDLXPK2"
//! 8    4   version     u32 LE
//! 12   4   num_layers  u32 LE   MoE layers, every one of which has records
//! 16   4   per_layer   u32 LE   experts in each MoE layer
//! 20   4   slot_bytes  u32 LE   the three projections + interior alignment
//! 24   8   stride      u64 LE   slot_bytes padded to a direct-I/O sector
//! 32   4   int8_mode   u32 LE   the numeric mode the repack targeted
//! 36   4   num_pairs   u32 LE   entries in the fingerprint table
//! 40  ...  per-layer geometry, `num_layers` × 76 bytes
//! ...  ... fingerprint table, `num_pairs` × 16 bytes
//! ```
//!
//! A layer entry is the layer's block index in the checkpoint (`u32`), then
//! gate, up and down, each `offset, bytes, dtype, src_dtype, rows, cols` as six
//! `u32`s. `dtype` is the repacked form's and `src_dtype` the checkpoint's, both
//! as in-workspace [`GgmlDType`] discriminants; `rows × cols` is one expert's
//! projection. A fingerprint entry is `src_dtype: u32, dtype: u32, fp: u64`.
//!
//! # Every layer has records
//!
//! The section is the model's only copy of its experts — there is no checkpoint
//! beside it to fall back to — so the permanently VRAM-resident leading layers
//! are stored like the rest and filled from here at startup.
//!
//! # Why the geometry is stored rather than recomputed
//!
//! The section is the source of the expert cache's geometry, not a check on it:
//! the shapes and source dtypes here are what the cache sizes its slots from.
//!
//! # Why the geometry is not enough
//!
//! Geometry catches a change to *where* the bytes go. It says nothing about a
//! change to *what they are* — a repack kernel that emits a different
//! permutation, or a quantizer whose rounding moves, at identical sizes,
//! offsets and dtypes. The fingerprint table closes it per format: one entry for
//! every `(src_dtype → dtype)` pair the layers use, each the hash of what this
//! build's repack produces for that pair over a reference matrix. See
//! [`crate::models::repack_fingerprint`].

use crate::models::repack_fingerprint::PairPrint;
use candle::quantized::GgmlDType;
use candle::Result;

/// Marks the section as ours and the layout as this one. A change to the record
/// layout changes the last byte rather than adding a compatibility path.
pub(crate) const MAGIC: &[u8; 8] = b"CNDLXPK2";

/// Bytes before the per-layer table — what a reader takes first, to learn how
/// long the rest of the header is ([`encoded_len_from_fixed`]).
pub(crate) const FIXED_BYTES: usize = 40;

/// Bytes per projection in the geometry table: six `u32`s.
const PROJECTION_BYTES: usize = 24;

/// Bytes per layer in the geometry table: the block index and three projections.
const LAYER_BYTES: usize = 4 + 3 * PROJECTION_BYTES;

/// Bytes per fingerprint entry.
const PAIR_BYTES: usize = 16;

/// Where one projection sits inside a record, how long it is, what it is, and
/// what it was repacked from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProjectionSpan {
    /// Byte offset from the start of the record.
    pub offset: u32,
    /// Bytes of repacked payload — what the H2D copies, excluding any padding
    /// to the next projection's alignment.
    pub bytes: u32,
    /// The repacked form's dtype.
    pub dtype: GgmlDType,
    /// The checkpoint's dtype for this projection — the other half of the
    /// repack pair the fingerprint table covers.
    pub src_dtype: GgmlDType,
    /// One expert's projection: output rows by input columns.
    pub rows: u32,
    pub cols: u32,
}

/// One MoE layer's three projections. Within a layer every expert has this
/// same shape, which is what lets a record be a fixed stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayerSpans {
    /// The layer's block index in the checkpoint (`blk.{n}`) — what a loader
    /// checks its own MoE layer order against.
    pub block: u32,
    pub gate: ProjectionSpan,
    pub up: ProjectionSpan,
    pub down: ProjectionSpan,
}

impl LayerSpans {
    fn projections(&self) -> [ProjectionSpan; 3] {
        [self.gate, self.up, self.down]
    }
}

/// The header, decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PackHeader {
    pub num_layers: u32,
    pub experts_per_layer: u32,
    /// The slot image size: the three projections at their aligned offsets.
    pub slot_bytes: u32,
    /// Record-to-record distance in the section — `slot_bytes` rounded up to a
    /// direct-I/O sector, so every record offset is legal to `pread` at.
    pub stride: u64,
    pub int8_mode: u32,
    pub layers: Vec<LayerSpans>,
    /// One entry per repack pair the layers use, sorted.
    pub pairs: Vec<PairPrint>,
}

/// The whole header's length, from its first [`FIXED_BYTES`].
///
/// Checks the magic and version first, so a foreign or older section is named
/// as such rather than read as a nonsense length. The counts are bounded, so a
/// corrupt one is an error here and not an allocation the caller then makes.
pub(crate) fn encoded_len_from_fixed(fixed: &[u8]) -> Result<usize> {
    if fixed.len() < FIXED_BYTES {
        candle::bail!(
            "expert section header is {} bytes, needs at least {FIXED_BYTES}",
            fixed.len()
        );
    }
    if &fixed[..8] != MAGIC {
        candle::bail!("expert section magic mismatch: {:?}", &fixed[..8]);
    }
    let u32_at =
        |o: usize| u32::from_le_bytes([fixed[o], fixed[o + 1], fixed[o + 2], fixed[o + 3]]);
    let version = u32_at(8);
    if version != VERSION {
        candle::bail!("expert section version {version}, this build writes {VERSION}");
    }
    // Far past any model: 4,096 MoE layers, or a fingerprint for every pair of
    // a thousand formats.
    const MAX_LAYERS: usize = 4096;
    const MAX_PAIRS: usize = 1 << 20;
    let (layers, pairs) = (u32_at(12) as usize, u32_at(36) as usize);
    if layers > MAX_LAYERS || pairs > MAX_PAIRS {
        candle::bail!("expert section header claims {layers} layers and {pairs} fingerprints");
    }
    Ok(FIXED_BYTES + layers * LAYER_BYTES + pairs * PAIR_BYTES)
}

/// The distinct `(src_dtype, dtype)` pairs `layers` use, sorted — the pairs a
/// fingerprint table must cover, read off the geometry rather than maintained
/// beside it.
pub(crate) fn pairs_in(layers: &[LayerSpans]) -> Vec<(GgmlDType, GgmlDType)> {
    let mut pairs: Vec<(GgmlDType, GgmlDType)> = layers
        .iter()
        .flat_map(|l| l.projections())
        .map(|p| (p.src_dtype, p.dtype))
        .collect();
    pairs.sort_by_key(|&(s, d)| (s.to_u32(), d.to_u32()));
    pairs.dedup();
    pairs
}

impl PackHeader {
    /// Bytes this header occupies before the first record, unpadded.
    pub(crate) fn encoded_len(&self) -> usize {
        FIXED_BYTES + self.layers.len() * LAYER_BYTES + self.pairs.len() * PAIR_BYTES
    }

    /// Total records the section holds: every expert of every layer.
    pub(crate) fn total_experts(&self) -> usize {
        self.num_layers as usize * self.experts_per_layer as usize
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
        out.extend_from_slice(&self.int8_mode.to_le_bytes());
        out.extend_from_slice(&(self.pairs.len() as u32).to_le_bytes());
        for l in &self.layers {
            out.extend_from_slice(&l.block.to_le_bytes());
            for p in l.projections() {
                for v in [
                    p.offset,
                    p.bytes,
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
    /// use: a table that omits one would let a changed repack for that format
    /// through unchecked.
    pub(crate) fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < FIXED_BYTES {
            candle::bail!(
                "expert section header is {} bytes, needs at least {FIXED_BYTES}",
                buf.len()
            );
        }
        if &buf[..8] != MAGIC {
            candle::bail!("expert section magic mismatch: {:?}", &buf[..8]);
        }
        let u32_at = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
        let u64_at = |o: usize| {
            let mut b = [0u8; 8];
            b.copy_from_slice(&buf[o..o + 8]);
            u64::from_le_bytes(b)
        };
        let version = u32_at(8);
        if version != VERSION {
            candle::bail!("expert section version {version}, this build writes {VERSION}");
        }
        let num_layers = u32_at(12) as usize;
        let num_pairs = u32_at(36) as usize;
        let need = FIXED_BYTES
            .saturating_add(num_layers.saturating_mul(LAYER_BYTES))
            .saturating_add(num_pairs.saturating_mul(PAIR_BYTES));
        if buf.len() < need {
            candle::bail!(
                "expert section header claims {num_layers} layers and {num_pairs} fingerprints \
                 ({need} bytes) but only {} are present",
                buf.len()
            );
        }
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let base = FIXED_BYTES + i * LAYER_BYTES;
            let span = |k: usize| -> Result<ProjectionSpan> {
                let o = base + 4 + k * PROJECTION_BYTES;
                Ok(ProjectionSpan {
                    offset: u32_at(o),
                    bytes: u32_at(o + 4),
                    dtype: GgmlDType::from_u32(u32_at(o + 8))?,
                    src_dtype: GgmlDType::from_u32(u32_at(o + 12))?,
                    rows: u32_at(o + 16),
                    cols: u32_at(o + 20),
                })
            };
            layers.push(LayerSpans {
                block: u32_at(base),
                gate: span(0)?,
                up: span(1)?,
                down: span(2)?,
            });
        }
        let table = FIXED_BYTES + num_layers * LAYER_BYTES;
        let mut pairs = Vec::with_capacity(num_pairs);
        for i in 0..num_pairs {
            let o = table + i * PAIR_BYTES;
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
                "expert section fingerprints cover {covered:?} but its layers use {:?}",
                pairs_in(&layers)
            );
        }
        Ok(Self {
            num_layers: num_layers as u32,
            experts_per_layer: u32_at(16),
            slot_bytes: u32_at(20),
            stride: u64_at(24),
            int8_mode: u32_at(32),
            layers,
            pairs,
        })
    }
}

/// Bumped whenever the record layout or the header's own shape changes. There
/// is no reader for an older version — a mismatch rebuilds the model pack.
pub(crate) const VERSION: u32 = 4;

#[cfg(test)]
mod tests {
    use super::*;

    fn proj(offset: u32, src: GgmlDType, dtype: GgmlDType) -> ProjectionSpan {
        ProjectionSpan {
            offset,
            bytes: 0x100,
            dtype,
            src_dtype: src,
            rows: 8,
            cols: 128,
        }
    }

    fn one_layer() -> PackHeader {
        PackHeader {
            num_layers: 1,
            experts_per_layer: 2,
            slot_bytes: 0x300,
            stride: 0x1000,
            int8_mode: 3,
            layers: vec![LayerSpans {
                block: 5,
                gate: proj(0, GgmlDType::Q4_K, GgmlDType::Q4_KO),
                up: proj(0x100, GgmlDType::Q4_K, GgmlDType::Q4_KO),
                down: proj(0x200, GgmlDType::Q6_K, GgmlDType::Q6_KO),
            }],
            pairs: vec![
                PairPrint {
                    src_dtype: GgmlDType::Q6_K,
                    dtype: GgmlDType::Q6_KO,
                    fp: 0x1122_3344_5566_7788,
                },
                PairPrint {
                    src_dtype: GgmlDType::Q4_K,
                    dtype: GgmlDType::Q4_KO,
                    fp: 0x0102_0304_0506_0708,
                },
            ],
        }
    }

    /// The header with its fingerprint table ordered as [`pairs_in`] orders it.
    fn sorted() -> PackHeader {
        let mut h = one_layer();
        let order = pairs_in(&h.layers);
        h.pairs
            .sort_by_key(|p| order.iter().position(|&q| q == (p.src_dtype, p.dtype)));
        h
    }

    /// The exact bytes, field by field. This is the format — a change that
    /// alters any of these without bumping [`VERSION`] would silently make one
    /// build read another's section as if it agreed.
    #[test]
    fn the_header_encodes_to_these_exact_bytes() {
        let h = sorted();
        let got = h.encode();
        let le = |v: u32| v.to_le_bytes().to_vec();
        let mut want: Vec<u8> = b"CNDLXPK2".to_vec();
        for v in [4u32, 1, 2, 0x300] {
            want.extend(le(v));
        }
        want.extend(0x1000u64.to_le_bytes());
        want.extend(le(3)); // int8_mode
        want.extend(le(2)); // num_pairs
        want.extend(le(5)); // block
        for (offset, src, dtype) in [
            (0u32, GgmlDType::Q4_K, GgmlDType::Q4_KO),
            (0x100, GgmlDType::Q4_K, GgmlDType::Q4_KO),
            (0x200, GgmlDType::Q6_K, GgmlDType::Q6_KO),
        ] {
            for v in [offset, 0x100, dtype.to_u32(), src.to_u32(), 8, 128] {
                want.extend(le(v));
            }
        }
        for p in &h.pairs {
            want.extend(le(p.src_dtype.to_u32()));
            want.extend(le(p.dtype.to_u32()));
            want.extend(p.fp.to_le_bytes());
        }
        assert_eq!(got, want);
        assert_eq!(got.len(), 40 + 76 + 2 * 16);
        assert_eq!(got.len(), h.encoded_len());
    }

    /// Q4_K is workspace discriminant 17 (0x11) — *not* the GGUF file code 12.
    /// The section stores the in-workspace form because that is what the
    /// repack produced and what the kernels read.
    #[test]
    fn dtypes_are_stored_as_workspace_discriminants() {
        assert_eq!(GgmlDType::Q4_K.to_u32(), 17);
        let bytes = sorted().encode();
        // Layer 0, gate, src_dtype: fixed + block + offset/bytes/dtype.
        assert_eq!(bytes[FIXED_BYTES + 4 + 12], 0x11);
    }

    #[test]
    fn decode_inverts_encode() {
        let h = sorted();
        assert_eq!(PackHeader::decode(&h.encode()).unwrap(), h);
    }

    /// The table is read off the geometry: each distinct pair once, whichever
    /// projections share it.
    #[test]
    fn the_pairs_are_the_layers_distinct_repacks() {
        let h = one_layer();
        assert_eq!(pairs_in(&h.layers).len(), 2);
        let mut two = h.layers.clone();
        two.push(h.layers[0]);
        assert_eq!(pairs_in(&two), pairs_in(&h.layers));
    }

    /// A table that leaves out a pair its layers use is refused: that format's
    /// repack would otherwise go unchecked.
    #[test]
    fn a_table_missing_a_pair_is_refused() {
        let mut h = sorted();
        h.pairs.pop();
        let e = PackHeader::decode(&h.encode()).unwrap_err().to_string();
        assert!(e.contains("fingerprints cover"), "{e}");
    }

    /// Trailing bytes past the header are the records; decoding ignores them.
    #[test]
    fn decode_ignores_what_follows_the_table() {
        let h = sorted();
        let mut buf = h.encode();
        buf.extend_from_slice(&[0xAB; 4096]);
        assert_eq!(PackHeader::decode(&buf).unwrap(), h);
    }

    #[test]
    fn a_foreign_section_is_rejected_by_its_magic() {
        let mut buf = sorted().encode();
        buf[7] = b'1';
        let e = PackHeader::decode(&buf).unwrap_err().to_string();
        assert!(e.contains("magic mismatch"), "{e}");
    }

    #[test]
    fn another_version_is_rejected_rather_than_guessed_at() {
        let mut buf = sorted().encode();
        buf[8] = VERSION as u8 + 1;
        let e = PackHeader::decode(&buf).unwrap_err().to_string();
        assert!(e.contains(&format!("version {}", VERSION + 1)), "{e}");
    }

    /// A header whose counts outrun the bytes present is a truncated section,
    /// not a header with fewer layers — and a corrupt count is an error, not an
    /// allocation.
    #[test]
    fn a_truncated_or_absurd_table_is_an_error() {
        let buf = sorted().encode();
        let e = PackHeader::decode(&buf[..FIXED_BYTES + 12])
            .unwrap_err()
            .to_string();
        assert!(e.contains("only"), "{e}");
        let mut bad = sorted().encode();
        bad[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(PackHeader::decode(&bad).is_err());
    }

    #[test]
    fn total_experts_multiplies_the_two_counts() {
        let mut h = one_layer();
        h.num_layers = 48;
        h.experts_per_layer = 128;
        assert_eq!(h.total_experts(), 6144);
    }
}
