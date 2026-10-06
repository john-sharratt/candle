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
//! A record the pass *itself* moved lower is followed the same way: the holder is given
//! a new handle for the copy (`Sweep::follow_record`), and the old handle is left for
//! whoever still holds it. A record's bytes are band addresses, never its own slot, so
//! the copy is a correct record.
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
#[cfg(feature = "cuda")]
use super::gpu_chunks::ChunkPin;
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
    /// `old record raw id → (replacement record slot, its device address)` — the
    /// `KvHead` records this pass copied lower.
    ///
    /// Separate from [`Self::moved`] because a record is followed through a different
    /// field of the same holder: `moved` rewrites a chunk's `gids`, this replaces its
    /// `meta`. A record's bytes name band addresses, not its own slot, so a copy of it is
    /// a correct record for as long as its bands have not moved — see `Sweep::follow_record`.
    records: AHashMap<i64, (ChunkGid, u64)>,
}

impl CompactionMap {
    /// An empty map — nothing moved, so every lookup misses.
    pub fn new() -> Self {
        Self {
            touched: Vec::new(),
            moved: AHashMap::new(),
            new_addr: AHashMap::new(),
            records: AHashMap::new(),
        }
    }

    /// Record that the `KvHead` record at `old_raw` has been copied to `replacement`,
    /// whose device address is `addr`.
    pub fn insert_record(&mut self, old_raw: i64, replacement: ChunkGid, addr: u64) {
        debug_assert!(old_raw >= 0, "a sentinel record never moves");
        self.records.insert(old_raw, (replacement, addr));
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
        let arena = old_raw as usize / super::types::GID_STRIDE;
        if self.touched.len() <= arena {
            self.touched.resize(arena + 1, 0);
        }
        self.touched[arena] = 1;
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
    /// Destination gids of chunks the sweep deliberately left on their sources, because
    /// no fresh record could be minted for them ([`Self::remint`]).
    ///
    /// Kept apart from an unreached holder's destinations: those are a hole in the holder
    /// set, these are a pass that ran out of record room and will move the chunk later.
    #[cfg(feature = "cuda")]
    forgone: AHashSet<i64>,
    /// Where a rewritten chunk's fresh record comes from, and the backing that claims
    /// it. `None` off a device, where there are no records at all.
    ///
    /// Held here rather than passed to each holder because the holders are visited
    /// through a closure the caller owns, across three crates — see
    /// [`Self::mint_record`].
    #[cfg(feature = "cuda")]
    mint: Option<(&'m mut RecordMint, &'m ChunkedKvBacking)>,
    /// Every record a holder gave up this sweep — replaced by a mint or by a relocated
    /// copy — held alive until the sweep ends.
    ///
    /// **The same hazard as [`Self::originals`], on the record side.** A record move is
    /// looked up by its slot's raw id, and a record whose last holder is replaced frees its
    /// slot. The mints this same sweep is claiming can then be handed that slot, and a
    /// second visit to a holder of the fresh mint would find its raw id in the record map
    /// and be given **a copy of the old chunk's record** — another chunk's band pointers.
    /// Holding every retired record keeps its slot occupied until nothing more is looked
    /// up, so no raw id this sweep resolves can change what it names.
    #[cfg(feature = "cuda")]
    retired: Vec<MetaGid>,
    /// Raw ids of records this sweep minted, which [`Self::follow_record`] never follows.
    ///
    /// A mint is claimed from the record pool's free list, and a planned record source
    /// whose chunk went before the claims ran is on that list. A fresh mint landing there
    /// has a raw id the record map knows, and must not be swapped for the copy of what
    /// used to be in its slot.
    #[cfg(feature = "cuda")]
    minted: AHashSet<i64>,
    /// Old raw ids of the relocated records some holder was moved onto — the record
    /// side's completeness figure, as [`Self::witnessed`] is the band side's.
    #[cfg(feature = "cuda")]
    records_followed: AHashSet<i64>,
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
            forgone: AHashSet::new(),
            #[cfg(feature = "cuda")]
            mint: None,
            #[cfg(feature = "cuda")]
            retired: Vec::new(),
            #[cfg(feature = "cuda")]
            minted: AHashSet::new(),
            #[cfg(feature = "cuda")]
            records_followed: AHashSet::new(),
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
            forgone: AHashSet::new(),
            mint: Some((mint, backing)),
            retired: Vec::new(),
            minted: AHashSet::new(),
            records_followed: AHashSet::new(),
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
        let minted = match self.mint.as_mut() {
            Some((mint, backing)) => mint.mint(backing, next, src)?,
            None => None,
        };
        if let Some(record) = &minted {
            self.note_minted(record.raw());
        }
        Ok(minted)
    }

    /// Give a chunk whose gids are moving to `next` a fresh record — or say it must stay
    /// where it is.
    ///
    /// `true` when the holder may install `next`: the chunk carries no record (it is
    /// addressed from its gids), this sweep mints none, or a fresh record for `next` is now
    /// in `meta`. `false` when the record pool had no slot left for it: the chunk then
    /// keeps its source gids **and** its source record, which agree, and a later pass with
    /// record room moves it.
    ///
    /// **Moving the gids without the record splits the chunk.** Reads go through the
    /// record, to the source, which the record keeps alive; the new gids hold a destination
    /// nothing reads. So the chunk held two slots, and every later pass saw the
    /// record-pinned source as live, copied it again, and found no holder naming the copy:
    /// 445, then 13,076 relocated slots unreached on every pass for half an hour, each run
    /// of them starting on the pass after one that reported `records_declined`.
    #[cfg(feature = "cuda")]
    pub(super) fn remint(
        &mut self,
        next: &HeadGids,
        meta: &mut Option<MetaGid>,
        src: RecordInputs<'_>,
    ) -> candle::Result<bool> {
        if meta.is_none() || self.mint.is_none() {
            return Ok(true);
        }
        match self.mint_record(next, src)? {
            Some(record) => {
                self.install_record(meta, record);
                Ok(true)
            }
            None => {
                let map = self.map;
                self.forgone.extend(
                    next.as_slice()
                        .iter()
                        .map(ChunkGid::raw)
                        .filter(|&raw| map.is_destination(raw)),
                );
                Ok(false)
            }
        }
    }

    /// Record that a holder now names `next` — see [`Self::witnessed`]. Called where the
    /// holder installs it, so a chunk left on its source witnesses nothing.
    pub(super) fn witness(&mut self, next: &HeadGids) {
        for g in next.as_slice() {
            if self.map.is_destination(g.raw()) {
                self.witnessed.insert(g.raw());
            }
        }
    }

    /// Record that `raw` is a slot this sweep minted into — see [`Self::minted`].
    #[cfg(feature = "cuda")]
    fn note_minted(&mut self, raw: i64) {
        self.minted.insert(raw);
    }

    /// Put `record` in `meta`, keeping whatever it replaces alive until the sweep ends —
    /// see [`Self::retired`].
    #[cfg(feature = "cuda")]
    pub(super) fn install_record(&mut self, meta: &mut Option<MetaGid>, record: MetaGid) {
        if let Some(old) = meta.replace(record) {
            self.retired.push(old);
        }
    }

    /// Move `meta` onto its record's relocated copy, if the pass copied it; `true` when
    /// it did.
    ///
    /// **A copy is a correct record here, not an approximation of one.** A record's bytes
    /// are band *addresses*, never its own slot, so copying them to another slot yields a
    /// record that describes exactly what the original did. The new handle carries the
    /// same [`MetaGid::bands`] the old one held, so it keeps those bands alive on the same
    /// terms.
    ///
    /// **Called after the band rewrite, and a no-op on anything that rewrite minted.** A
    /// chunk whose bands moved was given a fresh record built from the new bands; its old
    /// record's copy names the *source* and would undo that. The copy is then simply never
    /// installed, and its slot is released with the map. A chunk whose bands moved but
    /// whose mint was declined still names its source, as its old record does — so
    /// following that record's copy is right, and reclaims the old record's slot.
    #[cfg(feature = "cuda")]
    pub(super) fn follow_record(&mut self, meta: &mut Option<MetaGid>) -> bool {
        if !self.follows(meta) {
            return false;
        }
        let Some(old) = meta.as_ref() else {
            return false;
        };
        let raw = old.raw();
        let Some((slot, addr)) = self.map.record(raw) else {
            return false;
        };
        let Some(bands) = old.bands() else {
            return false;
        };
        let next = MetaGid::from_slot(slot.clone(), bands.clone(), *addr);
        self.records_followed.insert(raw);
        self.install_record(meta, next);
        true
    }

    /// Whether [`Self::follow_record`] would move `meta` — asked without touching it,
    /// so a holder that does not move is never cloned to find out.
    #[cfg(feature = "cuda")]
    pub(super) fn follows(&self, meta: &Option<MetaGid>) -> bool {
        let Some(old) = meta.as_ref() else {
            return false;
        };
        let raw = old.raw();
        // A detached record names nothing and has no slot to follow.
        !self.minted.contains(&raw) && self.map.record(raw).is_some() && old.bands().is_some()
    }

    /// What this sweep replaced in the holders it reached: every original band
    /// allocation it rewrote and every record a holder gave up.
    ///
    /// **Taken while the sweep is alive, and only then meaningful.** Both are identified
    /// by address or raw id, which the sweep's own pins ([`Self::originals`],
    /// [`Self::retired`]) keep from being reused; after the sweep drops, a new allocation
    /// can take either.
    #[cfg(feature = "cuda")]
    pub(super) fn replaced(&self) -> Replaced {
        Replaced {
            allocs: self.originals.iter().map(HeadGids::alloc_id).collect(),
            records: self.retired.iter().map(MetaGid::raw).collect(),
        }
    }

    /// Relocated records no visited holder was moved onto. Their copies are wasted and
    /// their sources stay where they are — a chunk whose bands also moved this pass (it
    /// was minted instead) or a holder outside the sweep's reach.
    #[cfg(feature = "cuda")]
    pub(super) fn records_unfollowed(&self) -> usize {
        self.map
            .records_len()
            .saturating_sub(self.records_followed.len())
    }

    /// Relocated slots no visited holder named — the pass's completeness gap.
    ///
    /// Empty is the only sound answer: every slot the pass moved and is about to
    /// free must be named by a holder it rewrote. A destination forgone for want of a
    /// record ([`Self::remint`]) is not a gap and is not counted.
    ///
    /// Only the compaction pass asks, and there is no host compaction.
    #[cfg(feature = "cuda")]
    pub(super) fn unwitnessed(&self, map: &CompactionMap) -> Vec<i64> {
        map.destinations()
            .filter(|raw| !self.witnessed.contains(raw) && !self.forgone.contains(raw))
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
        // Hold the original alive so its address cannot be recycled under the memo key —
        // see [`Self::originals`]. Taken before the insert so the key is pinned for as long
        // as the entry exists.
        self.originals.push(gids.clone());
        self.memo.insert(id, next.clone());
        self.allocations_rewritten += 1;
        Ok(Some(next))
    }
}

/// Band allocations and records a sweep replaced — see [`Sweep::replaced`].
///
/// What decides whether a cached decode buffer is stale: a buffer pinning any of these
/// was serialised from ground the pass has since moved its holders off.
#[cfg(feature = "cuda")]
pub(super) struct Replaced {
    allocs: AHashSet<usize>,
    records: AHashSet<i64>,
}

#[cfg(feature = "cuda")]
impl Replaced {
    /// Whether the sweep replaced nothing, so no buffer can name replaced ground.
    pub(super) fn is_empty(&self) -> bool {
        self.allocs.is_empty() && self.records.is_empty()
    }

    /// Whether a cached buffer holding `pins` names anything the sweep replaced.
    ///
    /// Sound only while `self`'s sweep is alive, and while the buffer's pins are held —
    /// which they are, since they are read from the buffer that holds them.
    pub(super) fn names_any(&self, pins: &[ChunkPin]) -> bool {
        pins.iter().any(|p| {
            self.allocs.contains(&p.bands_alloc_id())
                || p.record_raw().is_some_and(|r| self.records.contains(&r))
        })
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
    // **Copy on write.** A pass moves a handful of chunks and the sweep visits every
    // chunk in the process, so cloning each one into a replacement that is then thrown
    // away was the whole cost of a sweep — a dozen refcount round trips per chunk across
    // every resident turn, ~120 ms a pass on a long-lived substrate. Nothing is cloned
    // until the first chunk that actually changes, and only its sequence and the ones
    // after it are rebuilt.
    let mut out: Option<Vec<SealedSequence>> = None;
    for (i, seq) in seqs.iter().enumerate() {
        let chunks = rewrite_chunks(&seq.chunks, sweep)?;
        match (chunks, out.as_mut()) {
            (Some(chunks), Some(out)) => out.push(with_chunks(seq, chunks)),
            (Some(chunks), None) => {
                let mut fresh = Vec::with_capacity(seqs.len());
                fresh.extend_from_slice(&seqs[..i]);
                fresh.push(with_chunks(seq, chunks));
                out = Some(fresh);
            }
            (None, Some(out)) => out.push(seq.clone()),
            (None, None) => {}
        }
    }
    Ok(out)
}

/// `seq` holding `chunks` in place of its own.
fn with_chunks(seq: &SealedSequence, chunks: Vec<SealedChunk>) -> SealedSequence {
    SealedSequence {
        chunks,
        token_count: seq.token_count,
        chunk_size: seq.chunk_size,
        location: seq.location,
    }
}

/// One sequence's chunks rewritten through the sweep, or `None` when none of them
/// changed. Clones only from the first chunk that changed — see [`rewrite_sealed`].
fn rewrite_chunks(
    chunks: &[SealedChunk],
    sweep: &mut Sweep<'_>,
) -> candle::Result<Option<Vec<SealedChunk>>> {
    let mut out: Option<Vec<SealedChunk>> = None;
    for (i, chunk) in chunks.iter().enumerate() {
        match (rewrite_chunk(chunk, sweep)?, out.as_mut()) {
            (Some(c), Some(out)) => out.push(c),
            (Some(c), None) => {
                let mut fresh = Vec::with_capacity(chunks.len());
                fresh.extend_from_slice(&chunks[..i]);
                fresh.push(c);
                out = Some(fresh);
            }
            (None, Some(out)) => out.push(chunk.clone()),
            (None, None) => {}
        }
    }
    Ok(out)
}

/// One chunk's replacement, or `None` when the sweep leaves it as it is.
fn rewrite_chunk(
    chunk: &SealedChunk,
    sweep: &mut Sweep<'_>,
) -> candle::Result<Option<SealedChunk>> {
    let mut changed: Option<SealedChunk> = None;
    if let Some(next) = sweep.rewrite_gids(&chunk.gids)? {
        let mut c = chunk.clone();
        // **A fresh record for the new bands, and only if this chunk had one.**
        // Installing it retires the old record — and with it, once the sweep ends, the
        // clone of the old gids the old record was holding — which is what lets the
        // source be reclaimed. A chunk with no record is addressed from its gids and
        // must not be given one; a chunk with one that cannot have a fresh one stays
        // whole on its source.
        #[cfg(feature = "cuda")]
        let movable = sweep.remint(
            &next,
            &mut c.meta,
            RecordInputs {
                k_pal: &c.k_pal,
                v_pal: &c.v_pal,
                k_scale: &c.k_scale,
                v_scale: &c.v_scale,
                k_fmt: &c.k_fmt,
                v_fmt: &c.v_fmt,
            },
        )?;
        #[cfg(not(feature = "cuda"))]
        let movable = true;
        if movable {
            sweep.witness(&next);
            c.gids = next;
            changed = Some(c);
        }
    }
    // The record itself may have been copied lower. After the band rewrite, so a record
    // just minted for new bands is never swapped for a copy of the old one.
    #[cfg(feature = "cuda")]
    {
        let meta = changed.as_ref().map_or(&chunk.meta, |c| &c.meta);
        if sweep.follows(meta) {
            let c = changed.get_or_insert_with(|| chunk.clone());
            sweep.follow_record(&mut c.meta);
        }
    }
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::chunked::meta_pool::MetaGid;
    use crate::kv_cache::chunked::types::GID_STRIDE;
    use crate::kv_cache::ArenaLocation;

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
            location: ArenaLocation::Gpu,
        };
        let seq_b = SealedSequence {
            chunks: vec![chunk_with(shared.clone())],
            token_count: 32,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
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
            location: ArenaLocation::Gpu,
        };
        let mut map = CompactionMap::new();
        map.insert(raw(2, 0), ChunkGid::detached(raw(0, 0)), 0x10);
        let mut sweep = Sweep::new(&map);
        assert!(rewrite_sealed(&[seq], &mut sweep).unwrap().is_none());
        assert_eq!(sweep.allocations_rewritten(), 0);
    }

    /// **Copy on write keeps every untouched chunk and sequence, in order.** The sweep
    /// clones nothing until the first change, so the chunks and sequences before it are
    /// copied in afterwards, and everything after it is carried through — the replacement
    /// must be the input with exactly the moved chunk swapped.
    #[test]
    fn a_change_mid_walk_keeps_every_untouched_chunk_in_place() {
        let gids = |arena, chunk| HeadGids::uniform(ChunkGid::detached(raw(arena, chunk)), 1);
        let seq = |chunks: Vec<SealedChunk>| SealedSequence {
            token_count: 32 * chunks.len(),
            chunks,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        };
        let before = seq(vec![chunk_with(gids(7, 0)), chunk_with(gids(7, 1))]);
        let moved = seq(vec![
            chunk_with(gids(7, 2)),
            chunk_with(gids(5, 2)),
            chunk_with(gids(7, 3)),
        ]);
        let after = seq(vec![chunk_with(gids(7, 4))]);
        let input = [before, moved, after];

        let mut map = CompactionMap::new();
        map.insert(raw(5, 2), ChunkGid::detached(raw(0, 1)), 0x4000);
        let mut sweep = Sweep::new(&map);
        let out = rewrite_sealed(&input, &mut sweep).unwrap().expect("moved");

        let names: Vec<Vec<i64>> = out
            .iter()
            .map(|s| {
                s.chunks
                    .iter()
                    .map(|c| c.gids.as_slice()[0].raw())
                    .collect()
            })
            .collect();
        assert_eq!(
            names,
            vec![
                vec![raw(7, 0), raw(7, 1)],
                vec![raw(7, 2), raw(0, 1), raw(7, 3)],
                vec![raw(7, 4)],
            ],
        );
        for (s, (o, i)) in out.iter().zip(input.iter()).enumerate() {
            assert_eq!(o.token_count, i.token_count, "sequence {s}");
            for (c, (oc, ic)) in o.chunks.iter().zip(i.chunks.iter()).enumerate() {
                if (s, c) != (1, 1) {
                    assert!(
                        oc.gids.is_same_alloc(&ic.gids),
                        "untouched chunk {s}.{c} is carried through, not rebuilt",
                    );
                }
            }
        }
    }

    /// **A destination is witnessed where a holder installs it, not where it is
    /// computed.** A chunk the sweep leaves on its source for want of a record computes
    /// its replacement and installs none, and must not read as a holder of the copy — that
    /// is how the completeness count tells an unreached holder from a declined mint.
    #[test]
    fn a_destination_is_witnessed_only_when_a_holder_installs_it() {
        let mut map = CompactionMap::new();
        map.insert(raw(5, 2), ChunkGid::detached(raw(0, 1)), 0x4000);
        let mut sweep = Sweep::new(&map);
        let gids = HeadGids::uniform(ChunkGid::detached(raw(5, 2)), 1);

        sweep.rewrite_gids(&gids).unwrap().expect("moved");
        assert!(
            sweep.witnessed.is_empty(),
            "computing a replacement names nothing",
        );

        let seq = SealedSequence {
            chunks: vec![chunk_with(gids)],
            token_count: 32,
            chunk_size: 32,
            location: crate::kv_cache::ArenaLocation::Gpu,
        };
        rewrite_sealed(&[seq], &mut sweep).unwrap().expect("moved");
        assert_eq!(
            sweep.witnessed.iter().copied().collect::<Vec<_>>(),
            [raw(0, 1)],
            "the holder that installed the replacement names its destination",
        );
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
            location: ArenaLocation::Gpu,
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

    #[cfg(feature = "cuda")]
    fn sealed(chunk: SealedChunk) -> SealedSequence {
        SealedSequence {
            chunks: vec![chunk],
            token_count: 32,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        }
    }

    /// **A holder of a relocated record is moved onto the copy, and nothing else about
    /// the chunk changes.** The copy's handle carries the very `bands` allocation the old
    /// one held, because a copy of a record's bytes names the same bands.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_relocated_record_is_followed_with_its_bands() {
        let bands = HeadGids::uniform(ChunkGid::detached(raw(5, 2)), 1);
        let record = MetaGid::from_slot(ChunkGid::detached(raw(9, 9)), bands.clone(), 0xFEED);
        let seq = sealed(SealedChunk {
            meta: Some(record),
            ..chunk_with(bands.clone())
        });

        let mut map = CompactionMap::new();
        map.insert_record(raw(9, 9), ChunkGid::detached(raw(1, 4)), 0xBEEF);
        let mut sweep = Sweep::new(&map);
        let out = rewrite_sealed(&[seq], &mut sweep)
            .unwrap()
            .expect("a followed record is a change the caller must install");

        let chunk = &out[0].chunks[0];
        assert!(chunk.gids.is_same_alloc(&bands), "the bands did not move");
        let meta = chunk.meta.as_ref().unwrap();
        assert_eq!(meta.raw(), raw(1, 4));
        assert_eq!(meta.device_addr(), 0xBEEF);
        assert!(meta.bands().unwrap().is_same_alloc(&bands));
        assert_eq!(sweep.records_unfollowed(), 0);
    }

    /// **The record a holder gave up stays alive until the sweep ends.** Records are
    /// followed by raw id; a retired record that freed its slot mid-sweep could have that
    /// slot handed to a mint, and a later visit would then follow the mint's raw id to a
    /// copy of another chunk's record. Asserted on the refcount, as the band-side memo test
    /// is, because liveness is what makes the raw id safe.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_retired_record_is_held_until_the_sweep_ends() {
        let bands = HeadGids::uniform(ChunkGid::detached(raw(5, 2)), 1);
        let slot = ChunkGid::detached(raw(9, 9));
        let unheld = slot.strong_count();
        let seq = sealed(SealedChunk {
            meta: Some(MetaGid::from_slot(slot.clone(), bands.clone(), 0xFEED)),
            ..chunk_with(bands)
        });
        let held = slot.strong_count();
        assert!(held > unheld, "the chunk's record holds its slot");

        let mut map = CompactionMap::new();
        map.insert_record(raw(9, 9), ChunkGid::detached(raw(1, 4)), 0xBEEF);
        let mut sweep = Sweep::new(&map);
        let out = rewrite_sealed(&[seq], &mut sweep)
            .unwrap()
            .expect("followed");
        drop(out);
        assert_eq!(
            slot.strong_count(),
            held,
            "with every holder gone, the sweep is still holding the old record",
        );

        // A second visit to a holder of the copy matches nothing.
        let copy = MetaGid::from_slot(
            ChunkGid::detached(raw(1, 4)),
            HeadGids::uniform(ChunkGid::detached(raw(5, 2)), 1),
            0xBEEF,
        );
        let mut again = Some(copy);
        assert!(!sweep.follow_record(&mut again));

        drop(sweep);
        assert_eq!(slot.strong_count(), unheld, "and lets go with the sweep");
    }

    /// **A record this sweep minted is never swapped for a relocated copy**, even when
    /// its raw id is one the record map knows — which is what a mint landing in a planned
    /// source's freed slot looks like. The copy would name the old chunk's bands.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_minted_record_is_never_followed() {
        let mut map = CompactionMap::new();
        map.insert_record(raw(9, 9), ChunkGid::detached(raw(1, 4)), 0xBEEF);
        let mut sweep = Sweep::new(&map);
        sweep.note_minted(raw(9, 9));

        let fresh = MetaGid::from_slot(
            ChunkGid::detached(raw(9, 9)),
            HeadGids::uniform(ChunkGid::detached(raw(0, 1)), 1),
            0xF00D,
        );
        let mut meta = Some(fresh);
        assert!(!sweep.follow_record(&mut meta));
        assert_eq!(meta.unwrap().device_addr(), 0xF00D);
        assert_eq!(
            sweep.records_unfollowed(),
            1,
            "the copy nobody took is reported, not hidden",
        );
    }

    /// Only the record map moves: a map holding record moves and no band moves is not
    /// empty, so the holder walk does not short-circuit past the records.
    #[test]
    fn a_map_with_only_record_moves_is_not_empty() {
        let mut map = CompactionMap::new();
        assert!(map.is_empty());
        map.insert_record(raw(9, 9), ChunkGid::detached(raw(1, 4)), 0xBEEF);
        assert!(!map.is_empty());
        assert_eq!(map.len(), 0, "no band moved");
        assert_eq!(map.records_len(), 1);
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
