//! Tests that a chunk's per-band format tags are read from the right side and
//! slot, and that they round-trip through the persisted encoding.
//!
//! The K and V tags interleave in band order, so a swap would decode every V
//! band as a K format. The tests pin that order against a chunk whose tags are
//! all distinct. An unrecorded tag reports as invalid rather than as a plausible
//! default. The persisted discriminants are asserted as raw bytes, because
//! changing one would silently reinterpret every chunk already written to disk.

use super::*;

#[cfg(test)]
pub(super) mod band_iteration_tests {
    use super::*;

    use crate::kv_cache::arena_table::{ArenaFormatTag, N_PALETTE};

    use crate::kv_cache::chunked::gid_pool::ChunkGid;

    /// One head, all bands sharing a gid, with the given K/V tags.
    pub(in super::super) fn window(k_tag: u8, v_tag: u8) -> ChunkWindow {
        ChunkWindow {
            gids: HeadGids::uniform(ChunkGid::detached(0), 1),
            usage: 32,
            offset: 0,
            k_pal: Arc::new(Vec::new()),
            v_pal: Arc::new(Vec::new()),
            k_scale: Arc::new(Vec::new()),
            v_scale: Arc::new(Vec::new()),
            k_fmt: Arc::new(vec![k_tag; N_PALETTE]),
            v_fmt: Arc::new(vec![v_tag; N_PALETTE]),
            meta: None,
            q_sign: None,
        }
    }

    /// **The K/V interleave.** `bands()` walks slots in `HeadGids` order —
    /// K,V alternating — and each slot must get the tag from its own side.
    /// Getting this backwards would decode every V band as a K format, which
    /// is exactly the class of bug the per-band tags exist to prevent.
    #[test]
    fn bands_alternate_k_and_v_tags() {
        let w = window(ArenaFormatTag::R16.as_u8(), ArenaFormatTag::F16.as_u8());
        let tags: Vec<ArenaFormatTag> = w.bands().map(|(_, t)| t).collect();

        assert_eq!(tags.len(), N_PALETTE * 2, "one slot per band, K and V");
        for (i, tag) in tags.iter().enumerate() {
            let want = if i % 2 == 0 {
                ArenaFormatTag::R16
            } else {
                ArenaFormatTag::F16
            };
            assert_eq!(*tag, want, "slot {i} took the wrong side's tag");
        }
    }

    /// A chunk that never recorded its tags reports `Invalid` rather than a
    /// plausible default. A guessed format would be decoded as real.
    #[test]
    fn unrecorded_tags_report_invalid() {
        let mut w = window(0, 0);
        w.k_fmt = Arc::new(Vec::new());
        w.v_fmt = Arc::new(Vec::new());

        assert!(
            w.bands().all(|(_, t)| t == ArenaFormatTag::Invalid),
            "absent tags must not decode to a real format"
        );
    }

    /// A live chunk and the sealed chunk it becomes must agree band-for-band —
    /// they share one indexing helper precisely so they cannot drift.
    #[test]
    fn a_window_and_its_sealed_form_agree() {
        let w = window(ArenaFormatTag::Q8_0.as_u8(), ArenaFormatTag::Q4_KS.as_u8());
        let sealed = SealedChunk {
            gids: w.gids.clone(),
            offset: w.offset,
            token_count: w.usage as u16,
            k_pal: w.k_pal.clone(),
            v_pal: w.v_pal.clone(),
            k_scale: w.k_scale.clone(),
            v_scale: w.v_scale.clone(),
            k_fmt: w.k_fmt.clone(),
            v_fmt: w.v_fmt.clone(),
            byte_size: 0,
            meta: None,
            q_sign: None,
        };
        let from_window: Vec<ArenaFormatTag> = w.bands().map(|(_, t)| t).collect();
        let from_sealed: Vec<ArenaFormatTag> = sealed.bands().map(|(_, t)| t).collect();
        assert_eq!(from_window, from_sealed);
    }
}

/// Tests for the chunk-owned format tags as a *source of truth* — the accessors
/// every inverted reader now goes through.
#[cfg(test)]
pub(super) mod band_tag_tests {
    use super::*;

    use crate::kv_cache::arena_table::{ArenaFormatTag, N_PALETTE};

    use crate::kv_cache::chunked::gid_pool::ChunkGid;

    use crate::kv_cache::{KvFormat, QuantFormat};

    use candle::DType;

    /// A two-head chunk whose every band carries a *distinct* tag, so any
    /// index arithmetic error in `bands()` shows up as a mismatched pair
    /// rather than an accidentally-correct uniform answer.
    pub(in super::super) fn distinct_band_chunk(n_kv_head: usize) -> SealedChunk {
        let n = n_kv_head * N_PALETTE;
        let mut sc = SealedChunk::for_test(0, 32);
        sc.gids = HeadGids::uniform(ChunkGid::detached(0), n_kv_head);
        // Tag values are arbitrary but unique per band, and the K and V ranges
        // are disjoint so a K/V swap is detectable.
        sc.k_fmt = Arc::new((0..n).map(|i| i as u8).collect());
        sc.v_fmt = Arc::new((0..n).map(|i| (100 + i) as u8).collect());
        sc
    }

    #[test]
    fn bands_pairs_every_slot_with_its_own_tag() {
        let n_kv_head = 2;
        let sc = distinct_band_chunk(n_kv_head);
        let got: Vec<u8> = sc.bands().map(|(_, tag)| tag.as_u8()).collect();

        // `HeadGids::as_slice` is K,V interleaved per (head, palette); tags are
        // indexed [h * N_PALETTE + p] per side. Rebuild that expectation by
        // hand rather than from the same helper being tested.
        let mut want: Vec<u8> = Vec::new();
        for h in 0..n_kv_head {
            for p in 0..N_PALETTE {
                want.push(ArenaFormatTag::from_u8((h * N_PALETTE + p) as u8).as_u8());
                want.push(ArenaFormatTag::from_u8((100 + h * N_PALETTE + p) as u8).as_u8());
            }
        }
        assert_eq!(
            got, want,
            "bands() must walk K,V interleaved per (head, palette)"
        );
        assert_eq!(got.len(), n_kv_head * N_PALETTE * 2);
    }

    #[test]
    fn bands_yields_the_gid_at_each_slot() {
        let sc = distinct_band_chunk(2);
        let gids: Vec<i64> = sc.bands().map(|(g, _)| g.raw()).collect();
        let want: Vec<i64> = sc.gids.as_slice().iter().map(|g| g.raw()).collect();
        assert_eq!(gids, want);
    }

    /// An unrecorded band reports `Invalid`, which fails every format
    /// predicate — a chunk that cannot say what it holds is never treated as
    /// eligible for a kernel that would misread it.
    #[test]
    fn unrecorded_bands_report_invalid() {
        let sc = SealedChunk::for_test(0, 32); // empty k_fmt / v_fmt
        assert!(sc.bands().all(|(_, t)| t == ArenaFormatTag::Invalid));
        assert!(
            sc.bands().all(|(_, t)| t.is_quantized()),
            "Invalid must not pass as a float"
        );
    }

    #[test]
    fn format_tags_rejects_an_unrecorded_chunk() {
        let sc = SealedChunk::for_test(0, 32);
        let err = sc.format_tags().unwrap_err().to_string();
        assert!(
            err.contains("no recorded format tags"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn format_tags_returns_both_sides_verbatim() {
        let sc = distinct_band_chunk(2);
        let (k, v) = sc.format_tags().unwrap();
        assert_eq!(k, sc.k_fmt.as_slice());
        assert_eq!(v, sc.v_fmt.as_slice());
    }

    /// **The load-bearing assertion for the persist path.** `ChunkPayload`'s
    /// `k_formats`/`v_formats` were built by mapping arena formats through
    /// [`KvFormat::to_tag`]; they are now the chunk's own tag bytes, copied.
    /// That substitution is byte-identical only because the two encodings are
    /// the same one — asserted here against raw byte values, per repo policy on
    /// serialization tests, rather than trusted from the type names.
    #[test]
    fn chunk_tag_bytes_are_the_persisted_format_encoding() {
        for (fmt, tag) in [
            (KvFormat::Float(DType::F32), ArenaFormatTag::F32),
            (KvFormat::Float(DType::F16), ArenaFormatTag::F16),
            (KvFormat::Float(DType::BF16), ArenaFormatTag::BF16),
            (KvFormat::Quantized(QuantFormat::R16), ArenaFormatTag::R16),
            (KvFormat::Quantized(QuantFormat::Q8_0), ArenaFormatTag::Q8_0),
            (
                KvFormat::Quantized(QuantFormat::Q4_KS),
                ArenaFormatTag::Q4_KS,
            ),
            (KvFormat::Quantized(QuantFormat::Q2_0), ArenaFormatTag::Q2_0),
            (KvFormat::Quantized(QuantFormat::Q0), ArenaFormatTag::Q0),
        ] {
            assert_eq!(
                fmt.to_tag(),
                tag.as_u8(),
                "persisted tag for {fmt:?} must equal the chunk's recorded byte"
            );
            assert_eq!(
                KvFormat::from_tag(tag.as_u8()),
                Some(fmt),
                "cold load must decode {tag:?} back to {fmt:?}"
            );
        }
    }

    /// Raw-byte goldens for the discriminants the substrate has already
    /// written to disk. These bytes are on-disk data: changing one silently
    /// re-interprets every persisted chunk.
    #[test]
    fn persisted_tag_discriminants_are_frozen() {
        assert_eq!(ArenaFormatTag::F32.as_u8(), 0);
        assert_eq!(ArenaFormatTag::F16.as_u8(), 1);
        assert_eq!(ArenaFormatTag::BF16.as_u8(), 2);
        assert_eq!(ArenaFormatTag::R16.as_u8(), 3);
        assert_eq!(ArenaFormatTag::Q8_0.as_u8(), 7);
        assert_eq!(ArenaFormatTag::Q8_KS.as_u8(), 10);
        assert_eq!(ArenaFormatTag::Q4_0.as_u8(), 15);
        assert_eq!(ArenaFormatTag::Q4_KS.as_u8(), 18);
        assert_eq!(ArenaFormatTag::Q2_0.as_u8(), 22);
        assert_eq!(ArenaFormatTag::Q0.as_u8(), 33);
        assert_eq!(ArenaFormatTag::Invalid.as_u8(), 255);
    }

    /// `from_u8` is the exact inverse of `as_u8` across the whole tag space,
    /// and unknown bytes degrade to `Invalid` rather than aliasing a format.
    #[test]
    fn tag_byte_round_trip_is_total() {
        for byte in 0u8..=255 {
            let tag = ArenaFormatTag::from_u8(byte);
            if tag == ArenaFormatTag::Invalid && byte != 255 {
                assert!(byte > 35, "byte {byte} should decode to a real tag");
            } else {
                assert_eq!(tag.as_u8(), byte, "round trip broken at byte {byte}");
            }
        }
    }
}
