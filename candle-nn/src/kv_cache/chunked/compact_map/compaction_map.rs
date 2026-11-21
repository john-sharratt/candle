//! Records where each relocated chunk and record of a compaction pass went, so
//! every holder can be redirected to its replacement.
//!
//! The map is built after the destinations are claimed, so every lookup returns a
//! slot that is already allocated. Each question is answered by one indexed load
//! per arena before any hashing, because a sweep asks it for every chunk in the
//! process and nearly every answer is no. Chunks and records are kept apart,
//! because a record is followed through a different field of a holder than a
//! chunk's gids are.

use super::*;

/// Where every relocated chunk went, published once per pass.
///
/// Built after the destination gids are claimed, so a lookup answers with a gid
/// that is already allocated and whose backing is the destination arena's.
pub struct CompactionMap {
    /// One byte per arena index: non-zero when that arena gave up at least one
    /// chunk. Rejection is a single indexed load, which is the whole point — a
    /// sweep asks this question once per gid across every holder in the process,
    /// and the answer is almost always "no".
    pub(super) touched: Vec<u8>,
    /// `old raw id → replacement gid`, consulted only when `touched` says it is
    /// worth hashing.
    pub(super) moved: AHashMap<i64, ChunkGid>,
    /// Destination addresses by new raw id, so the completeness check can ask
    /// whether a gid a holder names is one of this pass's destinations.
    pub(super) new_addr: AHashMap<i64, u64>,
    /// `old record raw id → (replacement record slot, its device address)` — the
    /// `KvHead` records this pass copied lower.
    ///
    /// Separate from [`Self::moved`] because a record is followed through a different
    /// field of the same holder: `moved` rewrites a chunk's `gids`, this replaces its
    /// `meta`. A record's bytes name band addresses, not its own slot, so a copy of it is
    /// a correct record for as long as its bands have not moved — see `Sweep::follow_record`.
    pub(super) records: AHashMap<i64, (ChunkGid, u64)>,
    /// One byte per arena index: non-zero when a record in that arena was
    /// copied lower — [`Self::touched`]'s counterpart for [`Self::records`].
    pub(super) records_touched: Vec<u8>,
}

impl CompactionMap {
    /// An empty map — nothing moved, so every lookup misses.
    pub fn new() -> Self {
        Self {
            touched: Vec::new(),
            moved: AHashMap::new(),
            new_addr: AHashMap::new(),
            records: AHashMap::new(),
            records_touched: Vec::new(),
        }
    }

    /// Record that the `KvHead` record at `old_raw` has been copied to `replacement`,
    /// whose device address is `addr`.
    pub fn insert_record(&mut self, old_raw: i64, replacement: ChunkGid, addr: u64) {
        debug_assert!(old_raw >= 0, "a sentinel record never moves");
        mark(&mut self.records_touched, old_raw);
        self.records.insert(old_raw, (replacement, addr));
    }

    /// Room for `moves` more relocated slots, so the claims insert without
    /// growing the tables under them.
    pub fn reserve(&mut self, moves: usize) {
        self.moved.reserve(moves);
        self.new_addr.reserve(moves);
    }

    /// Whether this pass can have moved anything of a payload in `set`'s
    /// arenas: a gid in an arena that gave up a chunk, or a record in one that
    /// gave up a record. `false` means every chunk of it would be left as it is.
    pub fn may_touch(&self, set: &ArenaSet) -> bool {
        let hit = |bytes: &[u8], arenas: &[u32]| {
            arenas
                .iter()
                .any(|&a| bytes.get(a as usize).copied().unwrap_or(0) != 0)
        };
        hit(&self.touched, set.chunk_arenas()) || hit(&self.records_touched, set.record_arenas())
    }

    /// The copy of the record at `raw`, or `None` when that record did not move.
    #[inline]
    pub fn record(&self, raw: i64) -> Option<&(ChunkGid, u64)> {
        if raw < 0 {
            return None;
        }
        self.records.get(&raw)
    }

    /// Records relocated in this pass.
    pub fn records_len(&self) -> usize {
        self.records.len()
    }

    /// Record that the chunk at `old_raw` now lives at `replacement`, whose band
    /// address is `addr`.
    pub fn insert(&mut self, old_raw: i64, replacement: ChunkGid, addr: u64) {
        debug_assert!(old_raw >= 0, "a sentinel gid never moves");
        mark(&mut self.touched, old_raw);
        self.new_addr.insert(replacement.raw(), addr);
        self.moved.insert(old_raw, replacement);
    }

    /// Whether nothing moved — no band and no record.
    pub fn is_empty(&self) -> bool {
        self.moved.is_empty() && self.records.is_empty()
    }

    /// Band slots relocated in this pass. Records are counted by [`Self::records_len`].
    pub fn len(&self) -> usize {
        self.moved.len()
    }

    /// Every destination gid this pass claimed, as a raw id.
    ///
    /// Only the compaction pass asks, and there is no host compaction.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn destinations(&self) -> impl Iterator<Item = i64> + '_ {
        self.new_addr.keys().copied()
    }

    /// Whether `raw` is a destination this pass claimed.
    #[inline]
    pub(in super::super) fn is_destination(&self, raw: i64) -> bool {
        self.new_addr.contains_key(&raw)
    }

    /// The replacement for `gid`, or `None` when it did not move.
    ///
    /// Two levels on purpose: the `touched` byte rejects without hashing, which is
    /// what keeps a sweep over every gid in the process cheap when a pass moved a
    /// handful of chunks.
    #[inline]
    pub fn get(&self, gid: &ChunkGid) -> Option<&ChunkGid> {
        let raw = gid.raw();
        if raw < 0 {
            return None;
        }
        let arena = raw as usize / super::super::types::GID_STRIDE;
        if self.touched.get(arena).copied().unwrap_or(0) == 0 {
            return None;
        }
        self.moved.get(&raw)
    }

    /// Whether any of `gids` moved — [`Self::get`] over the slice, asked once per
    /// run of equal ids. A chunk's gids repeat one id across its palettes (`K`
    /// for every palette of a head, then `V`), and a run asks the same question
    /// its first id already answered.
    pub fn names_moved(&self, gids: &[ChunkGid]) -> bool {
        let mut last: Option<i64> = None;
        gids.iter().any(|g| {
            let raw = g.raw();
            if last == Some(raw) {
                return false;
            }
            last = Some(raw);
            self.get(g).is_some()
        })
    }
}

impl Default for CompactionMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Mark `raw`'s arena in a per-arena byte table.
pub(super) fn mark(bytes: &mut Vec<u8>, raw: i64) {
    let arena = raw as usize / super::super::types::GID_STRIDE;
    if bytes.len() <= arena {
        bytes.resize(arena + 1, 0);
    }
    bytes[arena] = 1;
}
