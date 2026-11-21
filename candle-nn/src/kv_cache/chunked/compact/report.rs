//! Records what one compaction pass did, for its log line, its tests and the
//! profiler.
//!
//! The frontier before and after is the figure a pass is judged by, not the
//! number of moves. Refused collisions are counted because each one is a
//! corruption prevented. Relocated slots that no holder this sweep reached names
//! are counted as well, since they are the one completeness gap, and zero is the
//! healthy reading. Timings are reported here rather than recorded, because this
//! crate sits below the profiler.

use super::*;

/// What one pass did, for the caller's log line and for the tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompactionReport {
    /// Chunk slots physically copied to a lower address.
    pub moves: usize,
    /// Moves the plan produced, of which the budget let `moves` through.
    pub planned_moves: usize,
    /// Moves dropped from the plan because a migrate group holds their chunk
    /// (`HeldChunks`) — not counted in `planned_moves`.
    pub held: usize,
    /// Pools the census reached before the planning deadline.
    pub pools_censused: usize,
    /// Pools with an arena to rank — what the census set out to reach.
    pub pools_ranked: usize,
    /// Region rank of the first planned move's source — the arena the pass empties
    /// first. When the census reached the top-ranked pool this is the arena holding
    /// the frontier; `None` when nothing was planned.
    pub first_source_rank: Option<usize>,
    /// `HeadGids` allocations rewritten. Lower than `moves` whenever holders share.
    pub allocations_rewritten: usize,
    /// Holder visits the sweep answered from an earlier scan of the same unmoved
    /// allocation (`Sweep::clean_reused`).
    pub clean_reused: usize,
    /// Moves this pass declined because the slot claimed for them was one it had
    /// already planned to read.
    ///
    /// **Non-zero is the guard working, not a fault.** It counts a read/write
    /// collision refused before the launch; suffering one corrupts a chunk silently.
    pub source_collisions: usize,
    /// Relocated slots **no holder this sweep reached names** — the pass's
    /// completeness gap, and the one figure here that is a defect rather than a
    /// measurement.
    ///
    /// The pass claimed those slots and copied into them, and nothing it could see
    /// points at the result: the claim is wasted and the source's holder will never
    /// be corrected. It was 129 of 6767 on one measured pass, which is how the
    /// projection caches were found to be holders nobody swept.
    pub unwitnessed: usize,
    /// Fresh `KvHead` records minted for relocated chunks — one per rewritten gid
    /// allocation that had a record, written by one batched launch.
    ///
    /// Lower than [`Self::allocations_rewritten`] by however many relocated chunks
    /// carried no record: a live writer window is addressed from its gids and is given
    /// none. Zero with a non-zero `moves` means the pass relocated only writer windows.
    pub records_minted: usize,
    /// Relocated chunks that wanted a fresh record and could not have one, because the
    /// pre-provisioned record slots ran out mid-sweep.
    ///
    /// Correct but degraded: each keeps the record it has, which still names its source
    /// and still holds it alive, so nothing is reclaimed for that chunk. A standing
    /// non-zero reading means the reservation is under-provisioning.
    pub records_declined: usize,
    /// Record arenas the pass created up front so the sweep would not have to.
    pub record_arenas_reserved: usize,
    /// Every arena the pass created ahead of demand — the per-pool low destinations and
    /// the record reservation. Whichever of them received nothing the pass releases itself,
    /// and those are counted in `arenas_released`.
    pub fresh_arenas: usize,
    /// `KvHead` records copied to a lower slot, in the same launch as the bands. Counted
    /// in [`Self::moves`] as well, which is every slot the pass copied.
    pub records_moved: usize,
    /// Relocated records no visited holder was moved onto. Expected non-zero: a chunk
    /// whose bands also moved this pass is given a freshly minted record instead, and
    /// its record's copy goes unused. A reading near `records_moved` on passes that
    /// minted little means a record holder the sweep does not reach.
    pub records_unfollowed: usize,
    /// Cached decode buffers that still named ground the sweep replaced after the holder
    /// rewrite had run — decided from each buffer's own pins, so a buffer naming nothing
    /// replaced is neither cleared nor counted.
    ///
    /// **Zero is the healthy reading.** A slot whose chunks moved has its buffer cleared by
    /// its own rewrite, so a buffer left naming replaced ground belongs to a holder the
    /// sweep does not reach. The pins keep that ground alive, so it is not a wrong read,
    /// but it is ground the pass cannot reclaim until the buffer goes; the warning beside
    /// it names the slots.
    pub decode_buffers_cleared: usize,
    /// Arenas released afterwards, and so regions handed back.
    pub arenas_released: usize,
    /// The frontier before and after, in regions — the figure the pass exists to
    /// move.
    pub frontier_before: usize,
    pub frontier_after: usize,
    /// `true` when the time budget stopped the pass with work left. Not a failure:
    /// everything below the cursor is packed and nothing moved upward, so the next
    /// pass resumes closer.
    pub clipped: bool,
    /// `true` when some pool's plan stopped at [`planned_moves_cap`] — the pool has
    /// moves left that this pass never planned. Kept apart from [`Self::clipped`]
    /// because it is not progress by itself: a pool whose top arena cannot move caps
    /// its plan on every pass while reclaiming nothing, and counted as progress that
    /// would re-run the same useless pass after every wave.
    pub plan_capped: bool,
    /// The arena standing highest in the span once the pass is done, across every pool
    /// it packs — band and record. When its region is the frontier it is what holds the
    /// frontier up, and its live count says whether a later pass can move it; when it
    /// stands below the frontier, the region above it belongs to something no pool
    /// owns. A pass whose frontier did not fall reads as nothing without this.
    pub top_arena: Option<TopArena>,
    /// Where the pass spent its wall clock, by phase.
    ///
    /// **Reported rather than recorded here, because this crate sits below the
    /// profiler.** `candle_transformers::models::profile` is where every other span in
    /// a wave lands, and a breakdown that could show the forward's phases but not the
    /// compaction's is the shape that sends you looking in the wrong half. So the pass
    /// measures itself and the caller — which can reach the profiler — files the
    /// numbers under its own span names.
    pub timings: CompactionTimings,
}

/// The highest arena of any packed pool — see [`CompactionReport::top_arena`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopArena {
    pub key: ArenaKey,
    /// Its region's rank: one below the frontier when it is what holds it.
    pub rank: usize,
    /// Slots live in it.
    pub live: usize,
}

/// Wall clock per phase of one compaction pass. See [`CompactionReport::timings`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompactionTimings {
    /// Draining the device before the pass reads anything — every stream, not just
    /// this one.
    ///
    /// Reported separately from [`Self::barrier`] because it is the one phase the
    /// pass does not control the cost of: it waits for whatever was already in
    /// flight, so a large reading is a statement about the work the pass arrived
    /// behind rather than about the pass. Outside the budget for the same reason.
    pub quiesce: Duration,
    /// Ranking the pools, then walking every arena's occupancy bitmap and planning.
    /// Usually the largest share, and the one the budget's planning half bounds.
    /// The sum of the three below.
    pub plan: Duration,
    /// Of `plan`: ranking the pools and reading each censused pool's occupancy.
    pub census: Duration,
    /// Of `plan`: claiming fresh low arenas for the pools holding the frontier.
    pub provision: Duration,
    /// Of `plan`: the two-cursor walks and merging them into one list by source
    /// rank.
    pub walk: Duration,
    /// The host walk that claims each destination slot and computes both addresses.
    pub claim: Duration,
    /// Uploading the three record arrays and launching the batched copy.
    pub copy: Duration,
    /// The barrier between the copies and the records that will name their
    /// destinations.
    pub barrier: Duration,
    /// Rewriting every holder's gids — this crate's block tables, then the caller's —
    /// and claiming a record slot for each rewritten chunk that had one.
    pub sweep: Duration,
    /// The one launch that writes every minted record's bytes.
    pub mint: Duration,
    /// Checking every slot's cached decode buffer against what the sweep replaced, and
    /// dropping the ones that name it. A lock and a host walk per backing, timed apart
    /// from [`Self::mint`] so a slow check does not read as a slow kernel.
    pub invalidate: Duration,
    /// The device-wide sync, and releasing the arenas the pass emptied.
    pub publish: Duration,
}

impl CompactionReport {
    /// Regions the frontier fell by.
    pub fn regions_reclaimed(&self) -> usize {
        self.frontier_before.saturating_sub(self.frontier_after)
    }

    pub fn is_empty(&self) -> bool {
        self.moves == 0
    }
}
