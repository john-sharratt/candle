//! Carries the state of one compaction sweep across every holder it visits, so each
//! shared allocation is rewritten once and every holder that installs it shares the
//! result.
//!
//! Memos are keyed on an allocation's address, so every original is kept alive until
//! the sweep ends. Otherwise a freed address could be reused for another chunk and
//! the memo would answer for the wrong one. Retired records are held for the same
//! reason. Moving a chunk's gids without its record would split the chunk, so a
//! chunk whose fresh record cannot be minted stays whole on its source. Completeness
//! is counted in two sets: every slot the pass moved must be named by some holder
//! the sweep rewrote, and a chunk left on its source is counted apart from a holder
//! the sweep never reached.

use super::*;

/// State threaded across every holder in one sweep.
///
/// The memo is what makes a shared `HeadGids` allocation rewritten once, so two
/// holders of one allocation come back sharing a single replacement rather than
/// holding equal-but-distinct gids whose refcounts disagree with the sharing the
/// cache believes exists.
pub struct Sweep<'m> {
    pub(super) map: &'m CompactionMap,
    /// `HeadGids::alloc_id` → its replacement.
    pub(super) memo: AHashMap<usize, HeadGids>,
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
    pub(super) originals: Vec<HeadGids>,
    /// Allocations this sweep found naming no moved gid, by `alloc_id` — a
    /// later visit to one is answered without scanning it again.
    ///
    /// **The same allocation is visited many times.** A residence's hot chunks
    /// are Arc-injected into every slot that places them, a view shares its
    /// parent's prefix, and the scheduler sweeps a shared substrate once per
    /// slot; each visit scanned every gid of every chunk again. An allocation
    /// never changes, so the first scan's answer stands for the sweep.
    pub(super) clean: AHashSet<usize>,
    /// The allocations in [`Self::clean`], held alive until the sweep ends so
    /// no key is recycled for another allocation — see [`Self::originals`].
    pub(super) clean_pins: Vec<HeadGids>,
    /// Visits [`Self::clean`] answered.
    pub(super) clean_reused: usize,
    /// Allocations rewritten — the sweep's own progress figure.
    pub(super) allocations_rewritten: usize,
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
    pub(super) witnessed: AHashSet<i64>,
    /// Allocations [`Self::witness`] has read, by `alloc_id`, and the pins that
    /// keep each key naming its allocation until the sweep ends.
    pub(super) witnessed_allocs: AHashSet<usize>,
    pub(super) witnessed_pins: Vec<HeadGids>,
    /// Destination gids of chunks the sweep deliberately left on their sources, because
    /// no fresh record could be minted for them ([`Self::remint`]).
    ///
    /// Kept apart from an unreached holder's destinations: those are a hole in the holder
    /// set, these are a pass that ran out of record room and will move the chunk later.
    #[cfg(feature = "cuda")]
    pub(super) forgone: AHashSet<i64>,
    /// Where a rewritten chunk's fresh record comes from, and the backing that claims
    /// it. `None` off a device, where there are no records at all.
    ///
    /// Held here rather than passed to each holder because the holders are visited
    /// through a closure the caller owns, across three crates — see
    /// [`Self::mint_record`].
    #[cfg(feature = "cuda")]
    pub(super) mint: Option<(&'m mut RecordMint, &'m ChunkedKvBacking)>,
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
    pub(super) retired: Vec<MetaGid>,
    /// Raw ids of records this sweep minted, which [`Self::follow_record`] never follows.
    ///
    /// A mint is claimed from the record pool's free list, and a planned record source
    /// whose chunk went before the claims ran is on that list. A fresh mint landing there
    /// has a raw id the record map knows, and must not be swapped for the copy of what
    /// used to be in its slot.
    #[cfg(feature = "cuda")]
    pub(super) minted: AHashSet<i64>,
    /// Old raw ids of the relocated records some holder was moved onto — the record
    /// side's completeness figure, as [`Self::witnessed`] is the band side's.
    #[cfg(feature = "cuda")]
    pub(super) records_followed: AHashSet<i64>,
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
            clean: AHashSet::new(),
            clean_pins: Vec::new(),
            clean_reused: 0,
            allocations_rewritten: 0,
            witnessed: AHashSet::new(),
            witnessed_allocs: AHashSet::new(),
            witnessed_pins: Vec::new(),
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
    pub(in super::super) fn with_mint(
        map: &'m CompactionMap,
        mint: &'m mut RecordMint,
        backing: &'m ChunkedKvBacking,
    ) -> Self {
        Self {
            map,
            memo: AHashMap::new(),
            originals: Vec::new(),
            clean: AHashSet::new(),
            clean_pins: Vec::new(),
            clean_reused: 0,
            allocations_rewritten: 0,
            witnessed: AHashSet::new(),
            witnessed_allocs: AHashSet::new(),
            witnessed_pins: Vec::new(),
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
    pub(in super::super) fn mint_record(
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
    pub(in super::super) fn remint(
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
    ///
    /// **Once an allocation.** A replacement is shared by every holder of what it
    /// replaced, and each install witnessed all of its gids again — a chunk's id
    /// repeated across its palettes, a hash probe and an insert apiece. Witnessed
    /// once, its repeats asked once a run, the set is the same.
    pub(in super::super) fn witness(&mut self, next: &HeadGids) {
        let id = next.alloc_id();
        if self.witnessed_allocs.contains(&id) {
            return;
        }
        let mut last: Option<i64> = None;
        for g in next.as_slice() {
            let raw = g.raw();
            if last == Some(raw) {
                continue;
            }
            last = Some(raw);
            if self.map.is_destination(raw) {
                self.witnessed.insert(raw);
            }
        }
        // Pinned before the key is kept, as the memo's originals are.
        self.witnessed_pins.push(next.clone());
        self.witnessed_allocs.insert(id);
    }

    /// Record that `raw` is a slot this sweep minted into — see [`Self::minted`].
    #[cfg(feature = "cuda")]
    pub(super) fn note_minted(&mut self, raw: i64) {
        self.minted.insert(raw);
    }

    /// Put `record` in `meta`, keeping whatever it replaces alive until the sweep ends —
    /// see [`Self::retired`].
    #[cfg(feature = "cuda")]
    pub(in super::super) fn install_record(&mut self, meta: &mut Option<MetaGid>, record: MetaGid) {
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
    pub(in super::super) fn follow_record(&mut self, meta: &mut Option<MetaGid>) -> bool {
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
    pub(in super::super) fn follows(&self, meta: &Option<MetaGid>) -> bool {
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
    pub(in super::super) fn replaced(&self) -> Replaced {
        Replaced {
            allocs: self.originals.iter().map(HeadGids::alloc_id).collect(),
            records: self.retired.iter().map(MetaGid::raw).collect(),
        }
    }

    /// Relocated records no visited holder was moved onto. Their copies are wasted and
    /// their sources stay where they are — a chunk whose bands also moved this pass (it
    /// was minted instead) or a holder outside the sweep's reach.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn records_unfollowed(&self) -> usize {
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
    pub(in super::super) fn unwitnessed(&self, map: &CompactionMap) -> Vec<i64> {
        map.destinations()
            .filter(|raw| !self.witnessed.contains(raw) && !self.forgone.contains(raw))
            .collect()
    }

    pub fn allocations_rewritten(&self) -> usize {
        self.allocations_rewritten
    }

    /// Visits answered by an earlier scan of the same unmoved allocation
    /// ([`Self::clean`]).
    pub fn clean_reused(&self) -> usize {
        self.clean_reused
    }

    /// Whether this sweep can have anything to rewrite in a payload whose
    /// arenas are `set` ([`CompactionMap::may_touch`]): `false`, and
    /// [`rewrite_sealed`] would return `None` for it.
    pub fn may_touch(&self, set: &ArenaSet) -> bool {
        self.map.may_touch(set)
    }

    /// Rewrite one `HeadGids`, or `None` when none of its gids moved.
    ///
    /// Visible to the pass driver so it can walk holders this module knows nothing
    /// about — a live slot's `ChunkWindow` block table as well as a
    /// `SealedSequence` — while still sharing one memo, which is what makes an
    /// allocation held by both come back as one replacement.
    pub(in super::super) fn rewrite_gids(
        &mut self,
        gids: &HeadGids,
    ) -> candle::Result<Option<HeadGids>> {
        // **The presence scan before the memo.** The memo holds only allocations
        // that name a moved gid, and an allocation never changes, so one whose
        // gids moved nowhere cannot be in it. The sweep visits every resident
        // chunk and nearly all of them name no moved gid: scanned first, they
        // are rejected on presence bytes alone, without a hash probe of the memo.
        // An allocation already scanned clean is not scanned again
        // ([`Self::clean`]).
        let id = gids.alloc_id();
        if self.clean.contains(&id) {
            self.clean_reused += 1;
            return Ok(None);
        }
        if !self.map.names_moved(gids.as_slice()) {
            // Pinned before the key is kept, as the memo's originals are.
            self.clean_pins.push(gids.clone());
            self.clean.insert(id);
            return Ok(None);
        }
        if let Some(done) = self.memo.get(&id) {
            return Ok(Some(done.clone()));
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
