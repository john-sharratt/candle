//! The translation a compaction pass publishes, and the holder rewrite that
//! consumes it.
//!
//! [`compact_plan`](super::compact_plan) decides which chunk goes where; this is
//! what turns those moves into new identities and installs them.
//!
//! # Why a gid cannot simply be renumbered
//!
//! A [`ChunkGid`] is an `i64` **and** a backing pointer: the id encodes
//! `(arena_idx, chunk_idx)`, while the backing is an `Arc` to the refcount table
//! of *the arena it came from*, indexed by `chunk_idx` on every clone and drop.
//! Rewriting the id alone would leave every later clone and drop decrementing the
//! wrong arena's slot — silent refcount corruption that surfaces as a chunk freed
//! under a live holder, arbitrarily later. So the map carries a whole replacement
//! gid, claimed from the destination arena, and the rewrite installs that.
//!
//! # Why the rewrite is keyed on the ALLOCATION, not the holder
//!
//! [`HeadGids`] is `Arc<Vec<ChunkGid>>` with a derived `Clone`, so every sharing
//! path in the cache — a view borrowing its parent's blocks, an injected sealed
//! chunk, a windowed turn half, an adopted turn on another timeline — shares the
//! allocation and clones no gid at all. Two consequences, and both shape this
//! module:
//!
//! * **A slot's refcount is not a holder count.** A chunk held by a residence,
//!   three block tables and a cached glue island can read `strong_count() == 1`.
//!   Any completeness argument built on comparing a gid count against a refcount
//!   is therefore unsound — this is the specific mistake that made the earlier
//!   arena-driven relocation pass corrupt conversations. The argument here is
//!   structural instead: visit every field of gid-holding type reachable from the
//!   process roots, and memoise on [`HeadGids::alloc_id`] so an allocation is
//!   rewritten once and the same replacement lands in every field that held it.
//! * **`Arc::make_mut` is the wrong tool and always will be.** On a shared
//!   allocation it clones the `Vec` and leaves every other holder pointing at the
//!   original, so the holder you edited is fixed and the rest are silently stale.
//!   [`rewrite_sealed`] builds a new `HeadGids` and assigns it; nothing here
//!   mutates one in place.
//!
//! # What the device needs afterwards
//!
//! Nothing on the GPU stores a gid. Every kernel reaches KV through a resident
//! `KvHead` record holding **resolved addresses**, so a moved band leaves one
//! stale 8-byte word per `(head, palette, K/V)` slot that named it. The rewrite
//! emits those words as [`PatchWord`]s — address and new value — for the patch
//! kernel to store in one launch. That is why a chunk keeps its `MetaGid` through
//! a compaction rather than being given a fresh record: the record is *patched*,
//! not rebuilt, which is sound here precisely because a compaction moves each slot
//! once for every holder at once (see `kv_ptr_patch.cu`).

use ahash::{AHashMap, AHashSet};

use super::gid_pool::ChunkGid;
use super::head_gids::HeadGids;
use super::meta_pool::band_ptr_offset;
use super::types::{SealedChunk, SealedSequence};

/// One 8-byte device word a compaction must overwrite: a band pointer inside a
/// resident chunk record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchWord {
    /// Device address of the word itself.
    pub addr: u64,
    /// The band's address after the move.
    pub value: u64,
}

/// Geometry the record layout needs, so the rewrite can locate a band's pointer
/// word without consulting the arena tables again.
#[derive(Clone, Copy, Debug)]
pub struct RecordGeometry {
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub n_palette: usize,
}

/// Where every relocated chunk went, published once per pass.
///
/// Built after the destination gids are claimed, so a lookup answers with a gid
/// that is already allocated and whose backing is the destination arena's.
pub struct CompactionMap {
    /// One byte per arena index: non-zero when that arena gave up at least one
    /// chunk. Rejection is a single indexed load, which is the whole point — a
    /// sweep asks this question once per gid across every holder in the process,
    /// and the answer is almost always "no".
    touched: Vec<u8>,
    /// `old raw id → replacement gid`, consulted only when `touched` says it is
    /// worth hashing.
    moved: AHashMap<i64, ChunkGid>,
    /// Destination addresses by new raw id, for emitting patch words without
    /// re-resolving the arena tables.
    new_addr: AHashMap<i64, u64>,
}

impl CompactionMap {
    /// An empty map — nothing moved, so every lookup misses.
    pub fn new() -> Self {
        Self {
            touched: Vec::new(),
            moved: AHashMap::new(),
            new_addr: AHashMap::new(),
        }
    }

    /// Record that the chunk at `old_raw` now lives at `replacement`, whose band
    /// address is `addr`.
    pub fn insert(&mut self, old_raw: i64, replacement: ChunkGid, addr: u64) {
        debug_assert!(old_raw >= 0, "a sentinel gid never moves");
        let arena = old_raw as usize / super::types::GID_STRIDE;
        if self.touched.len() <= arena {
            self.touched.resize(arena + 1, 0);
        }
        self.touched[arena] = 1;
        self.new_addr.insert(replacement.raw(), addr);
        self.moved.insert(old_raw, replacement);
    }

    /// Whether any chunk moved.
    pub fn is_empty(&self) -> bool {
        self.moved.is_empty()
    }

    /// Chunks relocated in this pass.
    pub fn len(&self) -> usize {
        self.moved.len()
    }

    /// Every destination gid this pass claimed, as a raw id.
    pub(super) fn destinations(&self) -> impl Iterator<Item = i64> + '_ {
        self.new_addr.keys().copied()
    }

    /// Whether `raw` is a destination this pass claimed.
    #[inline]
    pub(super) fn is_destination(&self, raw: i64) -> bool {
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
        let arena = raw as usize / super::types::GID_STRIDE;
        if self.touched.get(arena).copied().unwrap_or(0) == 0 {
            return None;
        }
        self.moved.get(&raw)
    }

    /// Device address of the band a replacement gid names.
    #[inline]
    fn addr_of(&self, new_raw: i64) -> Option<u64> {
        self.new_addr.get(&new_raw).copied()
    }
}

impl Default for CompactionMap {
    fn default() -> Self {
        Self::new()
    }
}

/// State threaded across every holder in one sweep.
///
/// The memo is what makes a shared `HeadGids` allocation rewritten once; the
/// patch words accumulate across holders because a record belongs to a chunk, not
/// to whichever holder happened to be visited first.
pub struct Sweep<'m> {
    map: &'m CompactionMap,
    geometry: RecordGeometry,
    /// `HeadGids::alloc_id` → its replacement.
    memo: AHashMap<usize, HeadGids>,
    /// Records already patched, so a chunk shared by several holders contributes
    /// its words once.
    records_done: AHashSet<u64>,
    /// The words the patch kernel must store, in insertion order.
    patch: Vec<PatchWord>,
    /// Allocations rewritten — the sweep's own progress figure.
    allocations_rewritten: usize,
    /// Why a moved band got no patch word, counted so the deficit is attributable.
    ///
    /// `moves` and `patched_words` disagreed by 44% of a run's relocations and there
    /// was no way to tell which of three reasons was responsible: a holder carrying
    /// no device record (ordinary — a live chunk window's bands are addressed from
    /// its block table), a record already emitted by an earlier holder (ordinary —
    /// one record describes one chunk, however many holders name it), or a moved
    /// band whose record nobody emitted at all (not ordinary). Three causes, one
    /// number, and only the third is a defect.
    no_record: usize,
    dup_record: usize,
    /// Destination gids some visited holder actually named.
    ///
    /// **The pass's own completeness proof, and it cannot be inferred from the
    /// counts.** A pass relocates `map.len()` slots and frees their sources; that is
    /// only sound if every one of them is named by a holder this sweep rewrote. A
    /// holder nobody visits keeps naming the source slot, which is then handed to
    /// the next claim — so the stale holder reads whatever now occupies it, which is
    /// another sequence's KV, finite and plausible and wrong.
    ///
    /// `patch.len()` does not answer it. A live chunk window carries no device
    /// record (`meta: None`) and correctly contributes no patch word, so
    /// `patched < moves` is ordinary. What is not ordinary is a relocated slot that
    /// no holder named at all, and only a set of what was seen can tell the two
    /// apart.
    witnessed: AHashSet<i64>,
}

impl<'m> Sweep<'m> {
    pub fn new(map: &'m CompactionMap, geometry: RecordGeometry) -> Self {
        Self {
            map,
            geometry,
            memo: AHashMap::new(),
            records_done: AHashSet::new(),
            patch: Vec::new(),
            allocations_rewritten: 0,
            no_record: 0,
            dup_record: 0,
            witnessed: AHashSet::new(),
        }
    }

    /// Moved bands that produced no patch word, split by cause: holders with no
    /// device record, and records an earlier holder had already emitted.
    pub(super) fn patch_skips(&self) -> (usize, usize) {
        (self.no_record, self.dup_record)
    }

    /// A rewritten holder that carries no `MetaGid` at all.
    ///
    /// Counted separately from a `MetaGid` whose `device_addr` is zero: the first is
    /// a chunk addressed from its block table, the second one whose record lives on
    /// the host. Both are ordinary and neither needs a patch word, but they are
    /// different shapes and a deficit that cannot be split into them is not
    /// attributable.
    pub(super) fn note_no_meta(&mut self) {
        self.no_record += 1;
    }

    /// Relocated slots no visited holder named — the pass's completeness gap.
    ///
    /// Empty is the only sound answer: every slot the pass moved and is about to
    /// free must be named by a holder it rewrote.
    pub(super) fn unwitnessed(&self, map: &CompactionMap) -> Vec<i64> {
        map.destinations()
            .filter(|raw| !self.witnessed.contains(raw))
            .collect()
    }

    /// The patch words, **sorted by address** — the order the patch kernel wants,
    /// because one head's band pointers are 64 contiguous bytes and a sorted run
    /// lands in one or two sectors rather than eight scattered ones.
    pub fn into_patch_words(mut self) -> Vec<PatchWord> {
        self.patch.sort_unstable_by_key(|w| w.addr);
        self.patch
    }

    pub fn allocations_rewritten(&self) -> usize {
        self.allocations_rewritten
    }

    /// Rewrite one `HeadGids`, or `None` when none of its gids moved.
    ///
    /// Visible to the pass driver so it can walk holders this module knows nothing
    /// about — a live slot's `ChunkWindow` block table as well as a
    /// `SealedSequence` — while still sharing one memo, which is what makes an
    /// allocation held by both come back as one replacement.
    pub(super) fn rewrite_gids(&mut self, gids: &HeadGids) -> candle::Result<Option<HeadGids>> {
        let id = gids.alloc_id();
        if let Some(done) = self.memo.get(&id) {
            return Ok(Some(done.clone()));
        }
        if !gids.as_slice().iter().any(|g| self.map.get(g).is_some()) {
            return Ok(None);
        }
        let map = self.map;
        let next = gids.map_unique(|g| Ok(map.get(g).cloned().unwrap_or_else(|| g.clone())))?;
        // Every destination this holder now names. Recorded here rather than in
        // `emit_patch` because a holder with no device record still names the slot
        // and still keeps it honest — it is the naming that makes freeing the
        // source safe, not the patch.
        for g in next.as_slice() {
            if map.is_destination(g.raw()) {
                self.witnessed.insert(g.raw());
            }
        }
        self.memo.insert(id, next.clone());
        self.allocations_rewritten += 1;
        Ok(Some(next))
    }

    /// Emit the patch words for a chunk whose gids were rewritten.
    ///
    /// Indexes the gid slice at the RECORD's stride (`n_palette * 2` per head),
    /// which is what `serialize_kv_heads` uses and is not the global
    /// `GIDS_PER_HEAD` on a single-latent geometry.
    pub(super) fn emit_patch(&mut self, record_addr: u64, next: &HeadGids) {
        if record_addr == 0 {
            self.no_record += 1;
            return;
        }
        if !self.records_done.insert(record_addr) {
            self.dup_record += 1;
            return;
        }
        let g = self.geometry;
        let stride = g.n_palette * 2;
        let slots = next.as_slice();
        for h in 0..g.n_kv_head {
            for p in 0..g.n_palette {
                for is_value in [false, true] {
                    let idx = h * stride + p * 2 + usize::from(is_value);
                    let Some(gid) = slots.get(idx) else { continue };
                    let Some(addr) = self.map.addr_of(gid.raw()) else {
                        // Not a destination of this pass: the band did not move,
                        // so the record already names it correctly.
                        continue;
                    };
                    let off = band_ptr_offset(h, p, is_value, g.head_dim, g.n_palette) as u64;
                    self.patch.push(PatchWord {
                        addr: record_addr + off,
                        value: addr,
                    });
                }
            }
        }
    }
}

/// Rewrite the gids of `seqs` through the map, returning replacements.
///
/// `None` when nothing in `seqs` moved — the caller then keeps what it has rather
/// than installing an identical copy, which would invalidate device block tables
/// for no reason.
///
/// **Returns replacements; never edits a holder.** That is the shape
/// `quantize_sealed_in_place` and `migrate_sealed_to_cpu_batch_async` already use
/// to move KV thousands of times per run without once corrupting it: the caller
/// owns what it handed in and installs what comes back, so no holder has to be
/// *found*. The earlier relocation pass inverted this — it started from an arena
/// and tried to discover everyone pointing into it — and there is no index from a
/// gid back to its holders, which is why it could not be made correct.
///
/// The chunk keeps its `MetaGid`: its record is patched by the words this
/// accumulates, not rebuilt.
pub fn rewrite_sealed(
    seqs: &[SealedSequence],
    sweep: &mut Sweep<'_>,
) -> candle::Result<Option<Vec<SealedSequence>>> {
    if sweep.map.is_empty() {
        return Ok(None);
    }
    let mut touched_any = false;
    let mut out: Vec<SealedSequence> = Vec::with_capacity(seqs.len());
    for seq in seqs {
        let mut chunks: Vec<SealedChunk> = Vec::with_capacity(seq.chunks.len());
        for chunk in &seq.chunks {
            match sweep.rewrite_gids(&chunk.gids)? {
                None => chunks.push(chunk.clone()),
                Some(next) => {
                    touched_any = true;
                    match chunk.meta.as_ref() {
                        Some(meta) => sweep.emit_patch(meta.device_addr(), &next),
                        None => sweep.note_no_meta(),
                    }
                    let mut c = chunk.clone();
                    c.gids = next;
                    chunks.push(c);
                }
            }
        }
        out.push(SealedSequence {
            chunks,
            token_count: seq.token_count,
            chunk_size: seq.chunk_size,
            location: seq.location,
        });
    }
    Ok(if touched_any { Some(out) } else { None })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::chunked::types::GID_STRIDE;

    fn geometry() -> RecordGeometry {
        RecordGeometry {
            n_kv_head: 1,
            head_dim: 4,
            n_palette: crate::kv_cache::N_PALETTE,
        }
    }

    fn raw(arena: usize, chunk: usize) -> i64 {
        (arena * GID_STRIDE + chunk) as i64
    }

    /// A gid from an untouched arena is rejected without consulting the hash map
    /// — the fast path a whole-process sweep depends on.
    #[test]
    fn an_untouched_arena_is_rejected_by_the_presence_byte() {
        let mut map = CompactionMap::new();
        map.insert(raw(9, 3), ChunkGid::detached(raw(0, 0)), 0x1000);
        assert!(map.get(&ChunkGid::detached(raw(4, 3))).is_none());
        assert!(
            map.get(&ChunkGid::detached(raw(9, 7))).is_none(),
            "same arena, different chunk"
        );
        assert!(map.get(&ChunkGid::detached(raw(9, 3))).is_some());
    }

    /// A sentinel never moves and never hashes.
    #[test]
    fn a_sentinel_gid_is_never_translated() {
        let mut map = CompactionMap::new();
        map.insert(raw(1, 1), ChunkGid::detached(raw(0, 0)), 0x20);
        assert!(map.get(&ChunkGid::detached(-1)).is_none());
    }

    /// An empty map rewrites nothing, and says so rather than handing back an
    /// identical copy.
    #[test]
    fn an_empty_map_rewrites_nothing() {
        let map = CompactionMap::new();
        let mut sweep = Sweep::new(&map, geometry());
        assert!(rewrite_sealed(&[], &mut sweep).unwrap().is_none());
    }

    /// **One allocation, one rewrite, installed twice.**
    ///
    /// Two holders sharing a `HeadGids` allocation — what every view, injected
    /// chunk and adopted turn produces — must come back sharing ONE replacement
    /// allocation. If the sweep rewrote per holder they would hold equal but
    /// distinct gids, and the refcounts would then disagree with the sharing the
    /// cache believes exists.
    #[test]
    fn a_shared_allocation_is_rewritten_once_and_shared_again() {
        let shared = HeadGids::uniform(ChunkGid::detached(raw(5, 2)), 1);
        let seq_a = SealedSequence {
            chunks: vec![chunk_with(shared.clone())],
            token_count: 32,
            chunk_size: 32,
            location: crate::kv_cache::ArenaLocation::Gpu,
        };
        let seq_b = SealedSequence {
            chunks: vec![chunk_with(shared.clone())],
            token_count: 32,
            chunk_size: 32,
            location: crate::kv_cache::ArenaLocation::Gpu,
        };

        let mut map = CompactionMap::new();
        map.insert(raw(5, 2), ChunkGid::detached(raw(0, 1)), 0x4000);
        let mut sweep = Sweep::new(&map, geometry());

        let a = rewrite_sealed(&[seq_a], &mut sweep)
            .unwrap()
            .expect("moved");
        let b = rewrite_sealed(&[seq_b], &mut sweep)
            .unwrap()
            .expect("moved");

        assert_eq!(
            sweep.allocations_rewritten(),
            1,
            "one allocation, one rewrite"
        );
        assert!(
            a[0].chunks[0].gids.is_same_alloc(&b[0].chunks[0].gids),
            "both holders must end up on ONE replacement allocation",
        );
        assert_eq!(a[0].chunks[0].gids.as_slice()[0].raw(), raw(0, 1));
    }

    /// A chunk none of whose gids moved is passed through by value, and the
    /// sequence reports no change at all when that is true of every chunk.
    #[test]
    fn an_untouched_sequence_is_not_replaced() {
        let seq = SealedSequence {
            chunks: vec![chunk_with(HeadGids::uniform(
                ChunkGid::detached(raw(7, 7)),
                1,
            ))],
            token_count: 32,
            chunk_size: 32,
            location: crate::kv_cache::ArenaLocation::Gpu,
        };
        let mut map = CompactionMap::new();
        map.insert(raw(2, 0), ChunkGid::detached(raw(0, 0)), 0x10);
        let mut sweep = Sweep::new(&map, geometry());
        assert!(rewrite_sealed(&[seq], &mut sweep).unwrap().is_none());
        assert_eq!(sweep.allocations_rewritten(), 0);
    }

    /// Patch words come back **sorted by address**, which is what the kernel's
    /// coalescing assumes.
    #[test]
    fn patch_words_are_sorted_by_address() {
        let map = {
            let mut m = CompactionMap::new();
            m.insert(raw(5, 2), ChunkGid::detached(raw(0, 1)), 0xABCD);
            m
        };
        let mut sweep = Sweep::new(&map, geometry());
        // Two records, the higher address emitted first.
        let next = HeadGids::uniform(ChunkGid::detached(raw(0, 1)), 1);
        sweep.emit_patch(0x9000, &next);
        sweep.emit_patch(0x1000, &next);
        let words = sweep.into_patch_words();
        assert!(!words.is_empty());
        assert!(
            words.windows(2).all(|w| w[0].addr <= w[1].addr),
            "patch words must be ascending: {words:?}",
        );
        assert!(words.iter().all(|w| w.value == 0xABCD));
    }

    /// A record is patched once however many holders share its chunk — the words
    /// describe the chunk, not the visit.
    #[test]
    fn a_shared_record_is_patched_once() {
        let map = {
            let mut m = CompactionMap::new();
            m.insert(raw(5, 2), ChunkGid::detached(raw(0, 1)), 0xABCD);
            m
        };
        let mut sweep = Sweep::new(&map, geometry());
        let next = HeadGids::uniform(ChunkGid::detached(raw(0, 1)), 1);
        sweep.emit_patch(0x2000, &next);
        let after_first = sweep.patch.len();
        sweep.emit_patch(0x2000, &next);
        assert_eq!(
            sweep.patch.len(),
            after_first,
            "second visit emitted words again"
        );
    }

    /// A chunk with no device record contributes no words — there is nothing
    /// resident to patch.
    #[test]
    fn a_chunk_with_no_record_emits_nothing() {
        let map = {
            let mut m = CompactionMap::new();
            m.insert(raw(5, 2), ChunkGid::detached(raw(0, 1)), 0xABCD);
            m
        };
        let mut sweep = Sweep::new(&map, geometry());
        let next = HeadGids::uniform(ChunkGid::detached(raw(0, 1)), 1);
        sweep.emit_patch(0, &next);
        assert!(sweep.into_patch_words().is_empty());
    }

    fn chunk_with(gids: HeadGids) -> SealedChunk {
        SealedChunk {
            gids,
            offset: 0,
            token_count: 32,
            k_pal: std::sync::Arc::new(Vec::new()),
            v_pal: std::sync::Arc::new(Vec::new()),
            k_scale: std::sync::Arc::new(Vec::new()),
            v_scale: std::sync::Arc::new(Vec::new()),
            k_fmt: std::sync::Arc::new(Vec::new()),
            v_fmt: std::sync::Arc::new(Vec::new()),
            byte_size: 0,
            meta: None,
        }
    }
}
