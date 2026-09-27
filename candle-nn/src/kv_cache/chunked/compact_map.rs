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
//! # What the device needs afterwards: a NEW record, never a rewritten one
//!
//! Nothing on the GPU stores a gid. Every kernel reaches KV through a resident
//! `KvHead` record holding **resolved addresses**, and a record holds a clone of the
//! `HeadGids` it was serialized from ([`MetaGid`](super::meta_pool::MetaGid)), so it
//! cannot outlive the bands it names.
//!
//! So a rewritten chunk is given a **freshly minted** record built from its new gids
//! ([`compact_mint`](super::compact_mint)), and the original record is left exactly as
//! it is for whoever still holds it. Each side then describes live, consistent ground
//! for its whole life: the new record names the destination, which the rewritten holder
//! keeps alive, and the old one names the source, which its own held gids keep alive
//! until the last holder of that chunk goes.
//!
//! Rewriting the shared record instead is what corrupted K/V. A chunk has exactly
//! **one** record, so a pass that rewrote some of a chunk's holders and missed others
//! had no correct value to put in it: whichever slot it named was kept alive only by the
//! holders naming that same slot, and when those went the rest were still reading
//! through it into re-tenanted ground. `compact_backings` carries the measurements.

use ahash::{AHashMap, AHashSet};

#[cfg(feature = "cuda")]
use super::backing::ChunkedKvBacking;
#[cfg(feature = "cuda")]
use super::compact_mint::{RecordInputs, RecordMint};
use super::gid_pool::ChunkGid;
use super::head_gids::HeadGids;
#[cfg(feature = "cuda")]
use super::meta_pool::MetaGid;
use super::types::{SealedChunk, SealedSequence};

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
    /// Destination addresses by new raw id, so the completeness check can ask
    /// whether a gid a holder names is one of this pass's destinations.
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
    ///
    /// Only the compaction pass asks, and there is no host compaction.
    #[cfg(feature = "cuda")]
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
}

impl Default for CompactionMap {
    fn default() -> Self {
        Self::new()
    }
}

/// State threaded across every holder in one sweep.
///
/// The memo is what makes a shared `HeadGids` allocation rewritten once, so two
/// holders of one allocation come back sharing a single replacement rather than
/// holding equal-but-distinct gids whose refcounts disagree with the sharing the
/// cache believes exists.
pub struct Sweep<'m> {
    map: &'m CompactionMap,
    /// `HeadGids::alloc_id` → its replacement.
    memo: AHashMap<usize, HeadGids>,
    /// Every ORIGINAL allocation this sweep has rewritten, held alive until the sweep
    /// ends.
    ///
    /// **The memo is keyed on an ADDRESS — `Arc::as_ptr` — so an original that dies
    /// mid-sweep lets its key be recycled, and then the memo answers for the wrong
    /// chunk.** `alloc_id`'s own contract is that it is "stable only while the `Arc` is
    /// alive, which is exactly the life of the sweep that uses it"; this is what makes
    /// that true instead of merely hoped for.
    ///
    /// It became reachable when a compaction started minting records: installing a fresh
    /// record drops the old one, which drops *its* clone of the old gids, so original
    /// allocations began dying while the sweep was still running. The next `map_unique`
    /// could then allocate a replacement at a dead original's address — and because
    /// [`Self::rewrite_gids`] consults the memo *before* asking whether anything moved, a
    /// second visit to that holder was handed **another chunk's gids**. Second visits are
    /// ordinary: several batch slots share one substrate, so the scheduler sweeps the same
    /// residences once per slot and relies on a second visit matching nothing.
    ///
    /// One `Arc` clone per rewritten allocation, dropped with the sweep.
    originals: Vec<HeadGids>,
    /// Allocations rewritten — the sweep's own progress figure.
    allocations_rewritten: usize,
    /// Destination gids some visited holder actually named.
    ///
    /// **The gid side's completeness proof.** A pass relocates `map.len()` slots and
    /// frees their sources; that is only sound if every one of them is named by a
    /// holder this sweep rewrote. A holder nobody visits keeps naming the source
    /// slot — which is not itself corruption, because that holder's gid still holds
    /// the source's refcount and its record still names it, so nothing reissues it —
    /// but it is a claim wasted and a holder left behind, and the set of holders this
    /// sweep can reach is maintained by hand. A rising count is how a new holder nobody
    /// swept is discovered.
    witnessed: AHashSet<i64>,
    /// Where a rewritten chunk's fresh record comes from, and the backing that claims
    /// it. `None` off a device, where there are no records at all.
    ///
    /// Held here rather than passed to each holder because the holders are visited
    /// through a closure the caller owns, across three crates — see
    /// [`Self::mint_record`].
    #[cfg(feature = "cuda")]
    mint: Option<(&'m mut RecordMint, &'m ChunkedKvBacking)>,
}

impl<'m> Sweep<'m> {
    /// A sweep that rewrites gids and mints no records.
    ///
    /// For a caller with no device — there are no `KvHead` records to mint — and for the
    /// unit tests, which exercise the gid rewrite on detached gids.
    pub fn new(map: &'m CompactionMap) -> Self {
        Self {
            map,
            memo: AHashMap::new(),
            originals: Vec::new(),
            allocations_rewritten: 0,
            witnessed: AHashSet::new(),
            #[cfg(feature = "cuda")]
            mint: None,
        }
    }

    /// A sweep that mints a fresh record for every chunk whose gids it rewrites.
    ///
    /// The pass's own constructor. See [`compact_mint`](super::compact_mint) for why a
    /// relocated chunk needs a new record rather than a rewritten one.
    #[cfg(feature = "cuda")]
    pub(super) fn with_mint(
        map: &'m CompactionMap,
        mint: &'m mut RecordMint,
        backing: &'m ChunkedKvBacking,
    ) -> Self {
        Self {
            map,
            memo: AHashMap::new(),
            originals: Vec::new(),
            allocations_rewritten: 0,
            witnessed: AHashSet::new(),
            mint: Some((mint, backing)),
        }
    }

    /// The fresh record for a chunk whose gids were just rewritten to `next`, or `None`
    /// when this sweep mints none.
    ///
    /// Called by the two places a chunk's `meta` lives — [`rewrite_sealed`] for a
    /// `SealedChunk` and `SequenceState::rewrite_for_compaction` for a `ChunkWindow` —
    /// in the same visit that installs `next`, which is what keeps the sweep to one
    /// traversal of the holder set.
    ///
    /// **Only for a chunk that had a record.** A live writer window carries none and
    /// must not be given one: it is written through its gids and would then be read
    /// through a record, and the two would diverge on the next token.
    #[cfg(feature = "cuda")]
    pub(super) fn mint_record(
        &mut self,
        next: &HeadGids,
        src: RecordInputs<'_>,
    ) -> candle::Result<Option<MetaGid>> {
        match self.mint.as_mut() {
            Some((mint, backing)) => mint.mint(backing, next, src),
            None => Ok(None),
        }
    }

    /// Relocated slots no visited holder named — the pass's completeness gap.
    ///
    /// Empty is the only sound answer: every slot the pass moved and is about to
    /// free must be named by a holder it rewrote.
    ///
    /// Only the compaction pass asks, and there is no host compaction.
    #[cfg(feature = "cuda")]
    pub(super) fn unwitnessed(&self, map: &CompactionMap) -> Vec<i64> {
        map.destinations()
            .filter(|raw| !self.witnessed.contains(raw))
            .collect()
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
        // Every destination this holder now names — it is the naming that keeps the
        // destination's refcount honest, and the absence of a naming that leaves a
        // source held by a holder the pass could not correct.
        for g in next.as_slice() {
            if map.is_destination(g.raw()) {
                self.witnessed.insert(g.raw());
            }
        }
        // Hold the original alive so its address cannot be recycled under the memo key —
        // see [`Self::originals`]. Taken before the insert so the key is pinned for as long
        // as the entry exists.
        self.originals.push(gids.clone());
        self.memo.insert(id, next.clone());
        self.allocations_rewritten += 1;
        Ok(Some(next))
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
/// A rewritten chunk is given a **freshly minted** `MetaGid` when the sweep has a minter
/// and it had a record to replace; no record is ever rewritten in place. See
/// [`compact_mint`](super::compact_mint).
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
                    let mut c = chunk.clone();
                    // **A fresh record for the new bands, and only if this chunk had
                    // one.** Installing it drops the old record — and with it the clone
                    // of the old gids the old record was holding — which is what lets the
                    // source be reclaimed. A chunk with no record is addressed from its
                    // gids and must not be given one.
                    #[cfg(feature = "cuda")]
                    if c.meta.is_some() {
                        let minted = sweep.mint_record(
                            &next,
                            RecordInputs {
                                k_pal: &c.k_pal,
                                v_pal: &c.v_pal,
                                k_scale: &c.k_scale,
                                v_scale: &c.v_scale,
                                k_fmt: &c.k_fmt,
                                v_fmt: &c.v_fmt,
                            },
                        )?;
                        if let Some(record) = minted {
                            c.meta = Some(record);
                        }
                    }
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
    use crate::kv_cache::chunked::meta_pool::MetaGid;
    use crate::kv_cache::chunked::types::GID_STRIDE;

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
        let mut sweep = Sweep::new(&map);
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
        let mut sweep = Sweep::new(&map);

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

    /// **A second visit to an already-rewritten holder must match nothing — even after the
    /// original allocation has been dropped.**
    ///
    /// The scheduler sweeps the same substrate once per batch slot that shares it, so a
    /// second visit is ordinary and the pass relies on it being a no-op. The memo is keyed
    /// on `HeadGids::alloc_id`, an `Arc` address, so that only holds while the original is
    /// alive — and once a compaction mints records, installing a fresh one drops the old
    /// record's clone of the old gids and originals start dying *mid-sweep*. A replacement
    /// allocated at a dead original's address then collides with its memo key, and because
    /// `rewrite_gids` consults the memo before asking whether anything moved, the holder is
    /// handed **another chunk's gids**.
    ///
    /// Asserted on the **refcount**, not on an address collision. A test that dropped an
    /// original and hoped the allocator reused its address would pass or fail by luck; what
    /// makes the memo key safe is that the original is still *alive*, and a band's strong
    /// count says so deterministically. The second-visit checks ride along.
    #[test]
    fn a_second_visit_matches_nothing_after_the_original_is_dropped() {
        let mut map = CompactionMap::new();
        map.insert(raw(5, 2), ChunkGid::detached(raw(0, 1)), 0x4000);
        map.insert(raw(6, 4), ChunkGid::detached(raw(0, 9)), 0x8000);

        let band = ChunkGid::detached(raw(5, 2));
        let unheld = band.strong_count();
        let mut sweep = Sweep::new(&map);

        // Rewrite, install, and drop the original — the holder's own lifecycle.
        let first = {
            let original = HeadGids::uniform(band.clone(), 1);
            let held = band.strong_count();
            assert!(held > unheld, "the original gid vector holds the band");
            let next = sweep.rewrite_gids(&original).unwrap().expect("moved");
            drop(original);
            assert_eq!(
                band.strong_count(),
                held,
                "the SWEEP still holds the original, so its `alloc_id` cannot be recycled \
                 under the memo key while the memo entry lives",
            );
            next
        };
        assert_eq!(first.as_slice()[0].raw(), raw(0, 1));

        let second = {
            let original = HeadGids::uniform(ChunkGid::detached(raw(6, 4)), 1);
            let next = sweep.rewrite_gids(&original).unwrap().expect("moved");
            drop(original);
            next
        };
        assert_eq!(second.as_slice()[0].raw(), raw(0, 9));

        // Both holders now hold replacements, which name destinations the map has no key
        // for, so a second visit must match nothing rather than answer from the memo.
        assert!(
            sweep.rewrite_gids(&first).unwrap().is_none(),
            "a second visit to the first holder must match nothing",
        );
        assert!(
            sweep.rewrite_gids(&second).unwrap().is_none(),
            "and a replacement must never be answered with another chunk's gids",
        );
        assert_eq!(
            sweep.allocations_rewritten(),
            2,
            "two allocations rewritten, and no third invented by a recycled memo key",
        );

        // The sweep is what was holding it; when it goes, so does the original.
        drop(sweep);
        assert_eq!(band.strong_count(), unheld);
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
        let mut sweep = Sweep::new(&map);
        assert!(rewrite_sealed(&[seq], &mut sweep).unwrap().is_none());
        assert_eq!(sweep.allocations_rewritten(), 0);
    }

    /// **With no minter, a rewritten chunk keeps its record — and the record keeps the
    /// bands it named, so it is still describing live ground.**
    ///
    /// A `Sweep::new` sweep mints nothing: that is the shape off a device, where there
    /// are no `KvHead` records to mint, and it must still leave the pool consistent. The
    /// rewrite replaces `gids`, so the chunk names the destination while `meta` names the
    /// source — which is safe precisely because the record holds a clone of the
    /// `HeadGids` it was serialized from, so that source keeps a refcount and keeps the
    /// bytes the copy read out of it.
    ///
    /// The device path replaces the record instead ([`super::compact_mint`]); what is
    /// pinned here is the *lifetime* claim underneath both, and the thing that would break
    /// it is somebody clearing `meta` in `rewrite_sealed` — which reads as tidying up and
    /// is the whole corruption.
    #[test]
    fn without_a_minter_a_rewritten_chunk_keeps_a_record_that_still_owns_its_bands() {
        let source = ChunkGid::detached(raw(5, 2));
        let bands = HeadGids::uniform(source.clone(), 1);
        let record = MetaGid::from_slot(ChunkGid::detached(raw(9, 9)), bands.clone(), 0xFEED);
        let seq = SealedSequence {
            chunks: vec![SealedChunk {
                meta: Some(record),
                ..chunk_with(bands)
            }],
            token_count: 32,
            chunk_size: 32,
            location: crate::kv_cache::ArenaLocation::Gpu,
        };

        let mut map = CompactionMap::new();
        map.insert(raw(5, 2), ChunkGid::detached(raw(0, 1)), 0x4000);
        let mut sweep = Sweep::new(&map);
        let out = rewrite_sealed(&[seq], &mut sweep).unwrap().expect("moved");
        let chunk = &out[0].chunks[0];

        assert_eq!(
            chunk.gids.as_slice()[0].raw(),
            raw(0, 1),
            "the chunk must name the destination"
        );
        let meta = chunk.meta.as_ref().expect("the record is carried through");
        assert_eq!(meta.device_addr(), 0xFEED, "and it is the SAME record");
        assert_eq!(
            meta.bands().expect("a record names bands").as_slice()[0].raw(),
            raw(5, 2),
            "which still holds — and so still refcounts — the SOURCE band",
        );
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
