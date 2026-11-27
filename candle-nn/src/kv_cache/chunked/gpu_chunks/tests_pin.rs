//! Tests that a pin is judged stale when only the record it holds has changed
//! under unchanged band allocations.
//!
//! A compaction moves a chunk onto a copy of its record and leaves its bands
//! where they were. A pin compared on bands alone would keep the old record, so
//! the header written for the new record would be held by nothing in the buffer.
//! The test asserts that the pin is stale in exactly that case.

use super::*;

#[cfg(test)]
pub(super) mod pin_tests {
    use super::*;
    use crate::kv_cache::chunked::gid_pool::ChunkGid;

    fn window(gids: HeadGids, meta: Option<MetaGid>) -> ChunkWindow {
        ChunkWindow {
            gids,
            usage: 32,
            lead: 0,
            offset: 0,
            k_pal: Arc::new(Vec::new()),
            v_pal: Arc::new(Vec::new()),
            k_scale: Arc::new(Vec::new()),
            v_scale: Arc::new(Vec::new()),
            k_fmt: Arc::new(Vec::new()),
            v_fmt: Arc::new(Vec::new()),
            meta,
            q_sign: None,
        }
    }

    /// **A pin no longer describes a chunk whose record changed under unchanged gids.**
    ///
    /// That is what a compaction's record move produces: the chunk is moved onto a copy of
    /// its record and its bands stay put. A pin compared on bands alone would stay on the
    /// OLD record while `update_chunk` writes a header naming the new one, leaving the
    /// record the header dereferences held by nothing in the buffer.
    #[test]
    fn a_pin_is_stale_when_only_the_record_changed() {
        let bands = HeadGids::uniform(ChunkGid::detached(7), 1);
        let old = MetaGid::from_slot(ChunkGid::detached(100), bands.clone(), 0xA000);
        let new = MetaGid::from_slot(ChunkGid::detached(200), bands.clone(), 0xB000);

        let pin = ChunkPin::of(&window(bands.clone(), Some(old.clone())));
        assert!(pin.describes(&window(bands.clone(), Some(old))));
        assert!(
            !pin.describes(&window(bands.clone(), Some(new))),
            "same bands, different record: the pin must be replaced",
        );
        assert!(
            !pin.describes(&window(bands, None)),
            "a record dropped is a change too",
        );
        assert!(
            !pin.describes(&window(HeadGids::uniform(ChunkGid::detached(8), 1), None)),
            "and so is a band change",
        );
    }
}
