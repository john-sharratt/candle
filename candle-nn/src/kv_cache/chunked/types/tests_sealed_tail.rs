//! Tests for how a sealed tail is trimmed and cleaned up, which the re-feed and
//! the seal both rely on.
//!
//! A seal snapshots trailing empty blocks as well, and those are dropped, since
//! a restored empty block above the writer boundary would become the chunk the
//! next write lands in. A trim cuts the last tokens across a chunk boundary and
//! keeps the first tokens of the chunk it lands in. A cut into a quantized chunk,
//! or past the tail, is refused, because the re-feed writes those positions
//! again and a sealed format cannot take writes. An empty chunk inside the
//! history is not the tail and stays.

use super::*;

#[cfg(test)]
pub(super) mod sealed_tail_tests {
    use super::*;

    use crate::kv_cache::arena_table::{ArenaFormatTag, ArenaLocation, N_PALETTE};

    use crate::kv_cache::chunked::gid_pool::ChunkGid;

    pub(in super::super) fn sealed_chunk(tokens: u16, tag: u8) -> SealedChunk {
        SealedChunk {
            gids: HeadGids::uniform(ChunkGid::detached(0), 1),
            offset: 0,
            token_count: tokens,
            k_pal: Arc::new(Vec::new()),
            v_pal: Arc::new(Vec::new()),
            k_scale: Arc::new(Vec::new()),
            v_scale: Arc::new(Vec::new()),
            k_fmt: Arc::new(vec![tag; N_PALETTE]),
            v_fmt: Arc::new(vec![tag; N_PALETTE]),
            byte_size: 0,
            meta: None,
            q_sign: None,
        }
    }

    /// A chunk as a live tail writes it: K in `R16`, V in `F16`.
    pub(in super::super) fn native_chunk(tokens: u16) -> SealedChunk {
        SealedChunk {
            v_fmt: Arc::new(vec![ArenaFormatTag::F16.as_u8(); N_PALETTE]),
            ..sealed_chunk(tokens, ArenaFormatTag::R16.as_u8())
        }
    }

    pub(in super::super) fn sequence(chunks: Vec<SealedChunk>) -> SealedSequence {
        let token_count = chunks.iter().map(|c| c.token_count as usize).sum();
        SealedSequence {
            chunks,
            token_count,
            chunk_size: CHUNK_SIZE,
            location: ArenaLocation::Gpu,
        }
    }

    /// A seal snapshots every block, the empty writer and any pad after it
    /// included. Quantized, an empty block becomes a sealed-format chunk with
    /// room left in it, which a later restore can stand where writes go — and
    /// the paged stores skip a sealed format without a word. Dropping them
    /// loses nothing: an empty chunk holds no token.
    #[test]
    fn trailing_empty_chunks_are_dropped_and_nothing_else() {
        let q3 = ArenaFormatTag::Q3_0.as_u8();
        let r16 = ArenaFormatTag::R16.as_u8();
        let mut s = sequence(vec![
            sealed_chunk(32, q3),
            sealed_chunk(7, q3),
            sealed_chunk(0, r16),
            sealed_chunk(0, r16),
        ]);
        s.drop_empty_tail();
        let counts: Vec<u16> = s.chunks.iter().map(|c| c.token_count).collect();
        assert_eq!(counts, vec![32, 7]);
        assert_eq!(s.token_count, 39);
    }

    /// **A re-feed's cut keeps the landing chunk's first tokens and drops the
    /// chunks past it**: 6 off `[32, 3]` leaves `[29]`; 3 off it leaves `[32]`.
    #[test]
    fn a_trim_cuts_the_last_tokens_across_a_chunk_boundary() {
        let mut s = sequence(vec![native_chunk(32), native_chunk(3)]);
        s.trim_tail(6).unwrap();
        let counts: Vec<u16> = s.chunks.iter().map(|c| c.token_count).collect();
        assert_eq!((counts, s.token_count), (vec![29], 29));
        let mut s = sequence(vec![native_chunk(32), native_chunk(3)]);
        s.trim_tail(3).unwrap();
        let counts: Vec<u16> = s.chunks.iter().map(|c| c.token_count).collect();
        assert_eq!((counts, s.token_count), (vec![32], 32));
    }

    /// **A cut into a quantized chunk, or past the tail, is refused**, and a
    /// cut of nothing changes nothing.
    #[test]
    fn a_trim_refuses_a_quantized_chunk_and_an_overlong_cut() {
        let q3 = ArenaFormatTag::Q3_0.as_u8();
        let mut s = sequence(vec![sealed_chunk(32, q3), native_chunk(2)]);
        assert!(s.trim_tail(4).is_err());
        let mut s = sequence(vec![native_chunk(5)]);
        assert!(s.trim_tail(6).is_err());
        s.trim_tail(0).unwrap();
        assert_eq!(s.token_count, 5);
    }

    /// An empty chunk inside the history is not the tail, and stays.
    #[test]
    fn an_empty_chunk_before_a_full_one_is_kept() {
        let f16 = ArenaFormatTag::F16.as_u8();
        let mut s = sequence(vec![
            sealed_chunk(5, f16),
            sealed_chunk(0, f16),
            sealed_chunk(32, f16),
        ]);
        s.drop_empty_tail();
        assert_eq!(s.chunks.len(), 3);
    }
}
