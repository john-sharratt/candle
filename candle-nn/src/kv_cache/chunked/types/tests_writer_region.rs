//! Tests for where the writer boundary sits, and for which chunks compaction and
//! the append path may move.
//!
//! A writer-owned chunk that carries a record stays where it is, since its
//! writes go through its gids while its reads go through the record, and moving
//! it would let the two diverge. A followed record counts as a move, so the
//! cached buffer is discarded. A sealed partial inside the writer region moves
//! the boundary past it, so it cannot be taken for the writer and swallow the
//! next write. An append does the same without any caller having to remember.

#[cfg(feature = "cuda")]
use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod writer_region_tests {
    use super::*;

    use crate::kv_cache::arena_table::{ArenaFormatTag, N_PALETTE};

    use crate::kv_cache::chunked::gid_pool::ChunkGid;

    /// A window of `usage` tokens whose K bands are `tag` — V is F16 beside an
    /// R16 K, as a live writer's is, and `tag` otherwise.
    pub(in super::super) fn window(usage: u32, tag: ArenaFormatTag) -> ChunkWindow {
        let v = if tag == ArenaFormatTag::R16 {
            ArenaFormatTag::F16
        } else {
            tag
        };
        ChunkWindow {
            gids: HeadGids::uniform(ChunkGid::detached(0), 1),
            usage,
            offset: 0,
            k_pal: Arc::new(Vec::new()),
            v_pal: Arc::new(Vec::new()),
            k_scale: Arc::new(Vec::new()),
            v_scale: Arc::new(Vec::new()),
            k_fmt: Arc::new(vec![tag.as_u8(); N_PALETTE]),
            v_fmt: Arc::new(vec![v.as_u8(); N_PALETTE]),
            meta: None,
            q_sign: None,
        }
    }

    pub(in super::super) fn layer(windows: Vec<ChunkWindow>, writer_start: usize) -> SequenceState {
        let mut s = SequenceState::new(None);
        for w in windows {
            s.push_chunk(w);
        }
        s.set_writer_start_idx(writer_start);
        s
    }

    /// **A writer-owned chunk that carries a record is left exactly where it is.**
    ///
    /// The compaction design needs `sealed ⇒ has a record ⇒ immutable` and
    /// `live ⇒ meta: None`, because a writer-owned chunk writes through its **gids** while
    /// anything resolving its record reads through the **record** — so relocating one
    /// would let the two diverge on the next token. Every path that gives a `ChunkWindow`
    /// a record gives it to sealed prefix history, and `set_block_gids` clears `meta` on
    /// any gid mutation, so the combination should not arise; `rewrite_for_compaction`
    /// skips it rather than trusting that, because skipping is safe under every
    /// circumstance and a silent relocation is not.
    ///
    /// Asserted on the boundary `writer_start_idx` draws: the chunk below it is rewritten,
    /// the one at it is not, and both carry a record so the only difference is the side.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_writer_owned_chunk_with_a_record_is_left_alone() {
        use crate::kv_cache::chunked::compact_map::{CompactionMap, Sweep};
        use crate::kv_cache::chunked::gid_pool::ChunkGid;
        use crate::kv_cache::chunked::meta_pool::MetaGid;
        use ArenaFormatTag::*;

        let raw = |arena: usize, chunk: usize| (arena * GID_STRIDE + chunk) as i64;
        let with_record = |usage: u32, gid: i64| {
            let mut w = window(usage, Q8_KS);
            w.gids = HeadGids::uniform(ChunkGid::detached(gid), 1);
            w.meta = Some(MetaGid::from_slot(
                ChunkGid::detached(999),
                w.gids.clone(),
                0xFEED,
            ));
            w
        };

        // Block 0 is sealed prefix history; block 1 is the writer's. Both relocate.
        let mut s = layer(
            vec![with_record(32, raw(5, 2)), with_record(32, raw(6, 3))],
            1,
        );
        let mut map = CompactionMap::new();
        map.insert(raw(5, 2), ChunkGid::detached(raw(0, 1)), 0x1000);
        map.insert(raw(6, 3), ChunkGid::detached(raw(0, 2)), 0x2000);

        let mut sweep = Sweep::new(&map);
        let rewrite = s.rewrite_for_compaction(&mut sweep).unwrap();
        assert!(rewrite.moved);
        assert_eq!(
            rewrite.stranded,
            Some(StrandedRecords {
                first: 1,
                last: 1,
                count: 1,
                writer_start: 1,
            }),
            "the skipped chunk is reported, so the caller can name its slot",
        );

        assert_eq!(
            s.chunk_at(0).unwrap().gids.as_slice()[0].raw(),
            raw(0, 1),
            "the sealed prefix chunk is relocated",
        );
        assert_eq!(
            s.chunk_at(1).unwrap().gids.as_slice()[0].raw(),
            raw(6, 3),
            "the writer-owned chunk with a record keeps its source, so its gids and its \
             record still agree",
        );
    }

    /// **A record followed to its copy counts as a move, with its bands untouched.**
    ///
    /// The slot's cached decode buffer holds the record's address as `kvheads_ptr`, so a
    /// chunk whose record moved while its bands did not must still report a move — that
    /// report is what throws the buffer away. Missing it leaves a header naming the old
    /// record slot, which the pool reissues once the old handle goes.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_followed_record_counts_as_a_move_and_leaves_the_bands_alone() {
        use crate::kv_cache::chunked::compact_map::{CompactionMap, Sweep};
        use crate::kv_cache::chunked::gid_pool::ChunkGid;
        use crate::kv_cache::chunked::meta_pool::MetaGid;
        use ArenaFormatTag::*;

        let raw = |arena: usize, chunk: usize| (arena * GID_STRIDE + chunk) as i64;
        let mut w = window(32, Q8_KS);
        w.gids = HeadGids::uniform(ChunkGid::detached(raw(5, 2)), 1);
        w.meta = Some(MetaGid::from_slot(
            ChunkGid::detached(raw(9, 9)),
            w.gids.clone(),
            0xFEED,
        ));
        let bands = w.gids.clone();
        let mut s = layer(vec![w], 1);

        let mut map = CompactionMap::new();
        map.insert_record(raw(9, 9), ChunkGid::detached(raw(1, 4)), 0xBEEF);
        let mut sweep = Sweep::new(&map);
        assert!(
            s.rewrite_for_compaction(&mut sweep).unwrap().moved,
            "a followed record is a move, or the slot's decode buffer keeps the old address",
        );

        let chunk = s.chunk_at(0).unwrap();
        let meta = chunk.meta.as_ref().expect("the chunk still has a record");
        assert_eq!(meta.raw(), raw(1, 4));
        assert_eq!(meta.device_addr(), 0xBEEF);
        assert!(
            chunk.gids.is_same_alloc(&bands),
            "the bands did not move, so the gids are the same allocation",
        );
        assert!(
            meta.bands().unwrap().is_same_alloc(&bands),
            "and the copy's handle holds the same bands the original did",
        );
    }

    /// The measured shape: a tail restored above the boundary — sealed full
    /// chunks, a sealed partial with room left in it, and the fresh writer the
    /// seal pushed. The partial is the first chunk with room, so it would be
    /// taken for the writer and swallow the next write; the boundary goes past
    /// it, and the fresh writer is the writer again.
    #[test]
    fn a_sealed_partial_in_the_writer_region_moves_the_boundary_past_it() {
        use ArenaFormatTag::*;
        let mut s = layer(
            vec![
                window(32, Q8_KS),
                window(32, Q3_0),
                window(25, Q3_0),
                window(0, R16),
            ],
            0,
        );
        assert_eq!(
            s.decode_write_chunk_idx(),
            2,
            "the sealed partial is taken for the writer"
        );
        assert!(s.seal_unwritable_writer_region());
        assert_eq!(s.writer_start_idx(), 3);
        assert_eq!(s.decode_write_chunk_idx(), 3);
    }

    /// Every path that appends to a table goes through `extend_chunks` — a
    /// restored writer tail, a view folded back into its parent — and any of
    /// them can carry a chunk a mid-turn seal quantized. The append itself
    /// keeps the boundary past it, so no caller has to remember to.
    #[test]
    fn extending_with_a_sealed_partial_moves_the_boundary_past_it() {
        use ArenaFormatTag::*;
        let mut s = layer(vec![window(32, Q8_KS)], 1);
        s.extend_chunks(vec![window(32, Q3_0), window(25, Q3_0), window(0, R16)]);
        assert_eq!(s.writer_start_idx(), 3);
        assert_eq!(s.decode_write_chunk_idx(), 3);
    }

    /// A region the stores can write keeps its boundary; a sealed chunk below
    /// the boundary is not the region's business.
    #[test]
    fn a_writable_region_keeps_its_boundary() {
        use ArenaFormatTag::*;
        let mut s = layer(vec![window(32, Q8_KS), window(32, R16), window(7, R16)], 1);
        assert!(!s.seal_unwritable_writer_region());
        assert_eq!(s.writer_start_idx(), 1);
    }
}
