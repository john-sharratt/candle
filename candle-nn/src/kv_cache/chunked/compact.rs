//! The compaction pass: plan, copy, translate, publish.
//!
//! Packs every GPU KV pool to the lowest physical addresses it can reach, so the
//! arena frontier falls and the weight side can have the ground back.
//!
//! # The frontier is the objective
//!
//! `weight_floor` is measured from the top of the highest live arena, because the
//! wave transient tier must stand above it. So what costs expert residency — and
//! therefore decode — is not how many arenas are live or how full they are, but
//! *where the topmost one sits*. Everything here exists to lower that one index.
//!
//! Measured on the 30B-A3B before this existed: a full drain took `live` from
//! 1,999 arenas to 165 while the frontier fell only 1,999 → 1,727. One hundred and
//! sixty-five survivors, scattered upward, denied the weight side 1,562 regions —
//! and `weights` did not move by a single MiB across the whole run.
//!
//! # All of it, or none of it
//!
//! [`ChunkedKvBacking::compact`] is one call because a half-applied pass is the
//! corruption. A chunk's gid *is* its physical location, so moving bytes changes
//! identity, and every holder of that identity must be rewritten in the same
//! window. The holders span three crates: this backing's own block tables, the
//! substrate's residences, and the scheduler's projection caches. The caller's
//! `sweep` closure is how the layers above hand theirs over, and nothing is
//! published until it has returned.
//!
//! An earlier attempt inverted this — it started from an arena and tried to
//! *discover* who pointed into it. There is no index from a gid back to its
//! holders, and its completeness proof (a deduplicated gid count against a
//! refcount) is unsound because `HeadGids` is `Arc<Vec<ChunkGid>>` with a derived
//! `Clone`: every sharing path in the cache shares the allocation and clones no
//! gid at all, so a chunk held by six holders reads `strong_count() == 1`. That
//! pass corrupted conversations. The argument here is structural instead —
//! enumerate the holders, visit every one — which is why the closure is not
//! optional and why the pass refuses rather than guesses when it cannot finish.
//!
//! # The holders this pass does NOT visit, and why it is still complete
//!
//! [`super::types::WriterTail`] holds `ChunkWindow`s that have been *split off* a
//! slot's block table, so they are unreachable from `sequences` and no walk of it
//! can rewrite them. That is safe here for a reason worth writing down, because it
//! is a property of the caller rather than of this module: a tail exists only
//! between `split_off_writer_tail` and `extend_writer_tail`, both called inside one
//! `projection_assembler::apply_projection`, on the **same thread** that drives this
//! pass, and that call completes before the loop reaches the point compaction runs
//! from. So a tail cannot be outstanding while a pass is in flight.
//!
//! If the assembler ever yields between the split and the restore — an await, a
//! hand-off to another thread, a pass that spans loop iterations — that argument
//! dies and this pass starts relocating chunks whose only holder it cannot see.
//! Whoever makes that change has to sweep the tail too, or refuse a pass while one
//! is live.
//!
//! # Order, and why each step waits for the one before
//!
//! 1. One arena window for the whole pass. Taking it per call measured 199,888
//!    refusals against 788 passes on a prior branch, because any forward starting
//!    mid-pass killed every remaining call.
//! 2. Census and plan, per `(class, location)` pool.
//! 3. Claim every destination slot, then copy the bytes.
//! 4. Barrier. The records written next name those slots; nothing may read
//!    through them before the copies that filled them have run.
//! 5. Rewrite this backing's gids, then the caller's, then patch the device
//!    records — one 8-byte store per moved band, from a kernel, so no per-chunk
//!    record is serialised on the host and shipped.
//! 6. Invalidate every affected slot's cached decode buffer. **This is the step
//!    that silently corrupts if skipped**: `sync_decode_gpu_chunks` reuses its
//!    buffer whenever the chunk *count* agrees, and a compaction never changes a
//!    count, so the kernel would walk pre-compaction band addresses.
//! 7. Device-wide sync, then release the emptied arenas.
//!
//! The sync is device-wide rather than per-stream deliberately: it costs one
//! barrier at between-forwards cadence and it removes the whole overlap failure
//! domain, including side streams this module does not know about.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use candle::Result;

use super::arena::ArenaKey;
use super::compact_map::{CompactionMap, PatchWord, RecordGeometry, Sweep};
use super::compact_plan::{plan_pool, ChunkMove};
use super::size_class::SizeClass;
use crate::kv_cache::ArenaLocation;

/// What one pass did, for the caller's log line and for the tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompactionReport {
    /// Chunk slots physically copied to a lower address.
    pub moves: usize,
    /// `HeadGids` allocations rewritten. Lower than `moves` whenever holders share.
    pub allocations_rewritten: usize,
    /// Device pointer words the patch kernel stored.
    pub patched_words: usize,
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
    /// Moved bands whose holder carried no device record — a live chunk window
    /// addressed from its block table, or a record resident on the host. Ordinary:
    /// neither needs a patch word.
    pub patch_no_record: usize,
    /// Moved bands whose record an earlier holder had already emitted. Ordinary: one
    /// record describes one chunk however many holders name it.
    pub patch_dup_record: usize,
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
    pub plan: Duration,
    /// The host walk that claims each destination slot and computes both addresses.
    pub claim: Duration,
    /// Uploading the three record arrays and launching the batched copy.
    pub copy: Duration,
    /// The barrier between the copies and the records that will name their
    /// destinations.
    pub barrier: Duration,
    /// Rewriting every holder's gids — this crate's block tables, then the caller's.
    pub sweep: Duration,
    /// Uploading the patch words and launching the band-pointer store.
    pub patch: Duration,
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

/// Cumulative outcomes of every compaction pass attempted since boot.
///
/// **Counted here rather than at the caller because a refusal is the invisible
/// outcome.** The success path logs; `WaveInFlight` is ordinary contention and must
/// not, so a pass that never once gets the window looks exactly like a pass nobody
/// asked for — and those call for opposite fixes. Measured: 60 s of churn at 51%
/// efficiency with three passes and twenty-seven silent refusals, which read as "the
/// gate never opened".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompactionTally {
    /// Passes entered — the gate opened and [`compact_backings`] was called.
    pub attempts: u64,
    /// Passes that relocated at least one chunk.
    pub passes: u64,
    /// Refused because a forward owned the partition.
    pub wave_in_flight: u64,
    /// Refused because a hot→warm migrate was acting on captured chunk addresses.
    pub migrate_in_flight: u64,
    /// Refused because every pool was already gapless — the expected refusal.
    pub already_packed: u64,
    /// Refused because a step of the pass failed: claims, copy, barrier, or either
    /// side of the rewrite. Any non-zero reading is a defect, and the `tracing`
    /// error beside it names which.
    pub step_failed: u64,
    /// Refused for want of a device reservation.
    pub no_device: u64,
    /// Chunk slots relocated, all passes.
    pub moves: u64,
    /// Arenas released, all passes.
    pub arenas_released: u64,
    /// Regions the frontier fell by, all passes.
    pub regions_reclaimed: u64,
    /// Passes the time budget stopped with work left.
    pub clipped: u64,
    /// Moves dropped because the slot claimed for them was one this pass had already
    /// planned to read.
    ///
    /// **The one that was corrupting K/V.** The census is a snapshot, so a slot it
    /// saw occupied is planned as a source and can be freed before the claims run —
    /// after which the allocator hands it out, correctly, as another move's
    /// destination. Both halves are legitimate and the result is a read/write race
    /// between two concurrent blocks of one launch. Non-zero is expected and healthy:
    /// it counts collisions declined, not collisions suffered.
    pub source_collisions: u64,
}

static TALLY: [std::sync::atomic::AtomicU64; 12] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 12];

fn note(idx: usize, add: u64) {
    TALLY[idx].fetch_add(add, std::sync::atomic::Ordering::Relaxed);
}

/// Monotonic count of passes that relocated something, for a holder to tell
/// whether the gids it captured are still current.
static EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many compactions have published a relocation since boot.
///
/// **For a holder that captures gids and installs them later.** The persistence
/// thread's hot→warm batch is the one that does: it snapshots a turn's sealed
/// sequences under the migrate guard, releases the guard per group so a compaction
/// is free to run, and then installs what it built — which, for the chunks it
/// carried through rather than requantised, means writing pre-compaction gids back
/// over a residence a compaction has since rewritten. The holder and the device
/// record then disagree about where the chunk is, and the reader that resolves
/// through the record gets bytes the selection never chose.
///
/// Comparing this across the capture tells a caller its snapshot is stale. A
/// monotonic count and not the geometry, for the reason
/// `expert_lre::zone_geometry` records about the weight zone: capacity and frontier
/// both come back, and a count that only increases cannot be undone by a pass that
/// happens to restore the shape.
pub fn compaction_epoch() -> u64 {
    EPOCH.load(std::sync::atomic::Ordering::Acquire)
}

/// Every compaction outcome since boot. See [`CompactionTally`].
pub fn compaction_tally() -> CompactionTally {
    let v = |i: usize| TALLY[i].load(std::sync::atomic::Ordering::Relaxed);
    CompactionTally {
        attempts: v(0),
        passes: v(1),
        wave_in_flight: v(2),
        already_packed: v(3),
        no_device: v(4),
        moves: v(5),
        arenas_released: v(6),
        regions_reclaimed: v(7),
        clipped: v(8),
        step_failed: v(9),
        migrate_in_flight: v(10),
        source_collisions: v(11),
    }
}

/// Why a pass could not run or could not finish.
///
/// A refusal is always the safe outcome: nothing has been copied, nothing
/// rewritten, nothing freed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompactionRefused {
    /// **The pass is switched off, because it corrupts K/V.** See
    /// [`compact_backings`] for the evidence and for what has to be true before it
    /// runs again.
    Disabled,
    /// A forward owns the partition. Ordinary between-forwards contention; the
    /// caller notes it and comes back.
    WaveInFlight,
    /// The persistence thread is acting on chunk addresses it has already captured.
    /// Ordinary contention like [`Self::WaveInFlight`], and the reverse direction
    /// defers too — see `migrate_flight`.
    MigrateInFlight,
    /// Not a CUDA device, or no reservation yet.
    NoDevice,
    /// Every pool is already gapless from its lowest arena — the planner had nothing
    /// to propose. The only *expected* refusal, and the only one that is not a
    /// defect.
    ///
    /// **Separated from the failures below because one label for all of them cost a
    /// whole diagnostic cycle.** A tally reading `nothing_to_do=38` over 41 attempts
    /// against a pool at 51% efficiency reads as "the pools are packed and the
    /// efficiency figure is lying", when it could equally have been a census the
    /// allocator disagreed with or a sweep failing on every pass. Those want opposite
    /// fixes, and none of them is the one that reading implies.
    AlreadyPacked,
    /// The plan named destinations the allocator would not hand over, so nothing was
    /// left to copy. The census samples occupancy bitmaps word by word, so a slot it
    /// believed free can genuinely have gone — but *every* claim losing means the
    /// census and the allocator disagree about the pool.
    ClaimsLost,
    /// The copy launch failed. Nothing is rewritten, so every holder still names its
    /// source slot and the claimed destinations are merely wasted.
    CopyFailed,
    /// The pre-rewrite barrier failed, with the same consequence as
    /// [`Self::CopyFailed`].
    BarrierFailed,
    /// A backing's own block tables could not be rewritten.
    HolderRewriteFailed,
    /// The caller's sweep failed. Its holders still name the old slots, which are
    /// still live and still hold the bytes, so the pool is left consistent but
    /// unpacked.
    SweepFailed,
    /// **Not a refusal — a partition invariant broke while planning.** The pass is
    /// abandoned at the same safe point a refusal abandons it (claims are host-only
    /// and the copy has not launched, so nothing is copied, rewritten or freed), but
    /// the two must not be handled alike.
    ///
    /// A refusal means "not now": the pool is fine and the next window will do the
    /// work. This means the plan named ground no arena owns — a slot past an arena's
    /// capacity, or a copy between slots of unequal extent — and the state that
    /// produced it is still there, so the next pass will produce it again. Waiting
    /// changes nothing, and the address the plan implied lands in another tenant's
    /// region, which does not fault because every address in the reservation is
    /// mapped.
    ///
    /// Carries the message because the caller reports it rather than logging it: a
    /// refusal is a `debug` line, and this must not share that fate.
    Fault(String),
}

/// Claims per batch before the clock is consulted.
///
/// A time budget, not a move cap, because chunks are not equal: the ladder runs
/// from 320 B to 16 KiB, so the same "one move" is fifty times the bandwidth at
/// one rung than another and a count cannot bound a duration. Batching keeps the
/// clock check off the per-claim path while staying fine-grained enough that a
/// budget of a few milliseconds is respected.
const MOVES_PER_BATCH: usize = 512;

/// The copy plan, as the migration kernel wants it: one entry per relocated slot,
/// in three parallel arrays.
///
/// Built entirely on the host by [`ChunkedKvBacking::claim_moves`] and issued in a
/// single launch, so the whole pass costs two kernel calls — this copy and the
/// band-pointer patch — however many chunks it relocates.
#[derive(Default)]
struct CopyRecords {
    srcs: Vec<i64>,
    dsts: Vec<i64>,
    lens: Vec<i64>,
    /// Moves dropped because the slot claimed for them was one this pass had already
    /// planned to read — see the note in [`ChunkedKvBacking::claim_moves`].
    ///
    /// Carried here rather than counted only in the global tally so a single pass's
    /// line says whether the collision it declined was real. Without that the
    /// disappearance of a fault is unattributable: the guard that would have caught
    /// the collision downstream finds nothing precisely *because* this declined it,
    /// so silence proves both "it fired" and "there was nothing to fire at".
    source_collisions: usize,
}

/// Pack every GPU KV pool toward the lowest addresses, within `budget`.
///
/// **Takes every backing, because one plan must reach every block table.** Arenas
/// pool globally across same-config layers, so the pool being packed is shared —
/// but each backing owns its *own* `sequences` block tables, one per layer. A pass
/// driven from a single backing would relocate a chunk and rewrite one layer's view
/// of it, leaving the other forty-seven pointing at the vacated slot. That is not a
/// degraded pass, it is wrong attention on every layer but one.
///
/// `sweep` is called **once**, after every backing has rewritten its own block
/// tables and before anything is published, and is handed **the same [`Sweep`]**
/// those rewrites used. That sharing is required, not convenient: holders across the
/// backings, the substrate and the projection caches routinely share one
/// `Arc<Vec<ChunkGid>>` allocation, and a second `Sweep` would give each its own
/// equal-but-distinct replacement — leaving refcounts that disagree with the sharing
/// the cache believes exists. The caller applies it with
/// [`super::compact_map::rewrite_sealed`]. Returning `Err` aborts the pass with
/// nothing published.
///
/// A caller with provably no holders of its own (a batched session, whose only
/// block tables are these backings') passes a closure that does nothing. That is a
/// statement about that caller, not a default.
///
/// # The pass does not run: it corrupts K/V
///
/// Every call returns [`CompactionRefused::Disabled`] before touching anything. The
/// machinery below is complete and is kept whole deliberately — it is what the fix
/// has to be made against, and it carries the instrumentation that found the fault.
///
/// **What goes wrong.** A chunk's location is written down twice: in the `ChunkGid`,
/// which holds the arena slot's refcount, and in the `KvHead` record's band-pointer
/// word, which is the address the paged kernels dereference. This pass rewrites both,
/// and leaves behind records whose pointer names a slot no gid holds. Only the gid
/// keeps a slot alive, so that slot is free: the allocator reissues it, correctly, as
/// a later pass's destination and writes another chunk's K/V into it. The band then
/// reads finite, plausibly-shaped, wrong values.
///
/// **The measurements**, from Qwen3.8-Flash-Next with compaction on and the
/// `tensor-assert` harness in place. Zero orphaned gids against eight orphaned
/// records, so the gid side is sound and the record side is not. Seven of seven
/// clobbered addresses were written by the pass itself as destinations, five of those
/// seven with the record's pointer unchanged across the pass. The pool is
/// self-consistent throughout — no slot is double-allocated — which is why nothing
/// faults and nothing downstream of the pass can see it.
///
/// **Why it is fatal on a recurrent model and survivable elsewhere.** Wrong K/V puts
/// a NaN in the first full-attention layer; a DeltaNet layer then computes its
/// recurrent state from that NaN and the state is *persisted*, so the next wave's
/// logits are entirely NaN and the model emits `!!!!!!!!` forever. The 30B survives
/// the identical wrong K/V because it has no recurrent state to persist it into.
///
/// **What re-enabling requires.** Not a repair at a call site: the two recordings have
/// to stop being able to disagree. `KvHead` records need to move into an arena so they
/// are walkable and relocatable, and the patch has to be driven from that walk rather
/// than from a per-holder sweep that can visit a record twice or not at all — the
/// design is in `docs/vram_span_partition.md`. The gate for turning this back on is
/// the check already wired here: `kv_integrity::report_boundary` must report zero
/// orphaned records and an unchanged content hash across a pass, over a run long
/// enough to compact many times.
#[cfg(feature = "cuda")]
pub fn compact_backings(
    backings: &[super::backing::ChunkedKvBacking],
    budget: Duration,
    sweep: &mut dyn FnMut(&mut Sweep<'_>) -> Result<()>,
) -> std::result::Result<CompactionReport, CompactionRefused> {
    /// Whether the pass may run. **False, because it corrupts K/V** — the header
    /// above has the evidence and the condition for flipping it back.
    const ENABLED: bool = false;
    // Refused before the attempt is counted: a pass that is switched off is not an
    // attempt that failed, and tallying it would put a denominator under a rate
    // nothing is measuring.
    if !ENABLED {
        return Err(CompactionRefused::Disabled);
    }
    note(0, 1);
    let Some(first) = backings.first() else {
        note(4, 1);
        return Err(CompactionRefused::NoDevice);
    };
    // The pass takes its own clock, after it has drained the device — see the note
    // on the budget in `compact_with`.
    let outcome = first.compact_with(backings, budget, sweep);
    match &outcome {
        Ok(r) => {
            note(1, 1);
            // Bumped only for a pass that actually moved something: a pass that
            // relocated nothing invalidates nobody's captured gids, and a counter
            // that ticked anyway would make every holder's snapshot look stale.
            if r.moves > 0 {
                EPOCH.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            }
            note(5, r.moves as u64);
            note(6, r.arenas_released as u64);
            note(7, r.regions_reclaimed() as u64);
            if r.clipped {
                note(8, 1);
            }
        }
        Err(CompactionRefused::WaveInFlight) => note(2, 1),
        Err(CompactionRefused::MigrateInFlight) => note(10, 1),
        Err(CompactionRefused::AlreadyPacked) => note(3, 1),
        Err(CompactionRefused::NoDevice) => note(4, 1),
        // `compact_with` never produces this: the switch at the top of this function
        // returns before the pass is attempted, so nothing reaches the tally.
        Err(CompactionRefused::Disabled) => {}
        Err(
            CompactionRefused::ClaimsLost
            | CompactionRefused::CopyFailed
            | CompactionRefused::BarrierFailed
            | CompactionRefused::HolderRewriteFailed
            | CompactionRefused::SweepFailed
            | CompactionRefused::Fault(_),
        ) => note(9, 1),
    }
    outcome
}

impl super::backing::ChunkedKvBacking {
    /// The body of [`compact_backings`], on the backing that owns the shared pool.
    #[cfg(feature = "cuda")]
    fn compact_with(
        &self,
        backings: &[super::backing::ChunkedKvBacking],
        budget: Duration,
        sweep: &mut dyn FnMut(&mut Sweep<'_>) -> Result<()>,
    ) -> std::result::Result<CompactionReport, CompactionRefused> {
        let candle::Device::Cuda(cuda) = self.device() else {
            return Err(CompactionRefused::NoDevice);
        };
        let stream = cuda.cuda_stream();

        // **Nothing off-thread may be holding a chunk address.** The arena window
        // below excludes forwards, which run on this thread; it says nothing about the
        // persistence thread, whose hot→warm migrate captures a device address per
        // band from gids it has pinned. A pin keeps the arena alive and says nothing
        // about which slot of it the chunk occupies, so relocating one under a migrate
        // makes it copy whatever now sits in the vacated slot into the warm tier —
        // which surfaces, much later, as a sequence answering from another sequence's
        // KV. See `migrate_flight` for the measurement that found it.
        let _locations = super::migrate_flight::try_freeze_chunk_locations()
            .ok_or(CompactionRefused::MigrateInFlight)?;

        // One window for the whole pass — see the module note on 199,888 refusals.
        let _window = super::bump_arena::enter_arena_window(&stream, "a KV compaction")
            .map_err(|_| CompactionRefused::WaveInFlight)?;

        // **Quiesce the whole device before reading anything, not just before
        // publishing.**
        //
        // The pass had a barrier after its copy and another after its patch, and
        // none before it began — so every source slot it read, and every
        // destination it wrote, was read and written against whatever was still in
        // flight when it started. The two exclusions above are host-side: the arena
        // window keeps a *new* forward from opening and the location freeze keeps a
        // *new* migrate from starting, and neither waits for device work that was
        // already launched to retire. The persistence thread's hot→warm migrate is
        // the one that matters — it runs on its own copy stream, so stream-order
        // FIFO says nothing about it, and it reads exactly the band addresses this
        // pass is about to move.
        //
        // Device-wide rather than per stream, for the reason in the module note: one
        // barrier at between-forwards cadence removes the whole overlap domain,
        // including side streams this module does not know about. Ordered after both
        // exclusions deliberately — quiescing first would fence work that the
        // unexcluded window could then replace before the census ran.
        //
        // Refused rather than continued if it fails: nothing has been planned, so
        // this is the cheapest possible place to abandon a pass.
        let quiesce = Instant::now();
        if let Err(e) = self.device().synchronize() {
            tracing::error!(
                target: "candle_nn::kv_cache::compact",
                "compaction could not quiesce the device before planning, \
                 abandoning the pass: {e}",
            );
            return Err(CompactionRefused::BarrierFailed);
        }

        // **The budget starts after the wait, which is why this is taken here and
        // not by the caller.** Every deadline below is the pass's own work — the
        // census it chooses to run and the claims it chooses to make — and the
        // quiesce is neither: it waits for whatever the pass arrived behind. Timed
        // from before the drain, a 50 ms wait against an 80 ms budget would put the
        // planning deadline in the past before the first pool was censused, so every
        // pass would clip having moved nothing, and the harder the device was working
        // the more completely compaction would stop.
        let started = Instant::now();

        let frontier_before = self.frontier_regions().ok_or(CompactionRefused::NoDevice)?;

        // ── Plan every pool, and claim + copy within the budget ──────────────
        let mut map = CompactionMap::new();
        let mut report = CompactionReport {
            frontier_before,
            timings: CompactionTimings {
                quiesce: quiesce.elapsed(),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut pending: Vec<ChunkMove> = Vec::new();
        let mut pool_of: Vec<ArenaKey> = Vec::new();
        // **The census is the expensive half, so the clock is consulted between
        // pools.** `compaction_census` walks every arena's occupancy bitmap — at a
        // frontier of 900 arenas and 50k slots per arena that is tens of millions of
        // bits and a `Vec<u32>` per arena — and it runs once per rung of the ladder.
        // Planning the whole ladder unconditionally spent the entire budget before a
        // single chunk was claimed, and the pass then refused with nothing done: 37 of
        // 40 attempts, reported as claims lost, with the pools at 60% efficiency the
        // whole time.
        //
        // A quarter of the budget for planning, so three quarters reach the claims. A
        // pool left unplanned is not lost — it is where the next pass starts from, and
        // the order below puts the pool holding the frontier first, so what gets
        // planned is what the pass most needs to move. The split was half and half
        // while the ladder was walked in class order and the planner's choice of pool
        // was therefore arbitrary; with the order deliberate, spending less of the
        // budget deciding and more of it moving is the better trade.
        let plan_deadline = budget / 4;
        // **Highest arena first, because address is the quantity under attack.**
        //
        // Walked in ladder order instead, a budgeted pass spends its census and its
        // claims on whichever rungs happen to come first, and the pool that owns the
        // topmost arena — the only pool whose position costs the weight side anything
        // — can be last and never reached. Measured: nine consecutive seconds at 52–59%
        // efficiency with the frontier pinned at 346 and 64 free regions under it,
        // while every pass ran, clipped, and packed the low rungs it could already
        // reach.
        //
        // The order comes from `pool_top_rank`, which answers without the census, so
        // deciding where to spend the budget does not spend it.
        let mut by_rank: Vec<(usize, ArenaKey)> = SizeClass::all()
            .filter_map(|class| {
                let key = ArenaKey::new(class, ArenaLocation::Gpu);
                self.pool_top_rank(key).map(|(rank, _)| (rank, key))
            })
            .collect();
        by_rank.sort_unstable_by_key(|a| std::cmp::Reverse(a.0));
        let mut census_by_pool: Vec<(ArenaKey, Vec<super::compact_plan::ArenaSlots>)> = Vec::new();
        for (_, key) in by_rank {
            if started.elapsed() >= plan_deadline && !census_by_pool.is_empty() {
                report.clipped = true;
                break;
            }
            let Ok(census) = self.compaction_census(key) else {
                continue;
            };
            census_by_pool.push((key, census));
        }
        // **Give the topmost arena's pool somewhere lower to go.** This is the step
        // without which the frontier stops falling while everything else looks
        // perfect. Packing is per pool, so each pool converges onto *its own* lowest
        // arenas — and the pool holding the highest arena in the span may have no
        // lower arena with room, in which case a perfect per-pool pack leaves that one
        // arena exactly where it was and the frontier with it. Measured: every pool
        // gapless, 131 arenas live, and a frontier of 330.
        //
        // One fresh arena for that pool is the whole fix. A fresh arena claims from
        // the region free list, which is lowest-index-first, so it lands in a hole
        // *below* the frontier and the walk then has a destination under the top
        // arena. One per pass, and only while holes exist — otherwise a claim would
        // take a region above the frontier and push it up, which is the opposite of
        // the objective.
        if let Some(fresh) = self.lowest_hole_destination(&census_by_pool) {
            match self.inner.claim_fresh_region(fresh) {
                Ok(_) => {
                    if let Ok(recensus) = self.compaction_census(fresh) {
                        if let Some(slot) = census_by_pool.iter_mut().find(|(k, _)| *k == fresh) {
                            slot.1 = recensus;
                        }
                    }
                }
                Err(e) => tracing::debug!(
                    target: "candle_nn::kv_cache::compact",
                    "no fresh low arena for the top pool, packing without one: {e}",
                ),
            }
        }
        for (key, census) in &census_by_pool {
            let Some(plan) = plan_pool(census, *key, 0) else {
                continue;
            };
            for m in plan.moves {
                pending.push(m);
                pool_of.push(*key);
            }
        }
        if pending.is_empty() {
            return Err(CompactionRefused::AlreadyPacked);
        }
        report.timings.plan = started.elapsed();
        let mut phase = Instant::now();

        // **Claim on the host, copy in one launch.** The budget bounds the claim
        // walk — a host loop over the plan, one free-list pop and two address
        // computations per move — and the copies it produced then go to the device
        // as a single batched record set. Checking the clock per batch rather than
        // per move keeps the branch off the inner path.
        let mut records = CopyRecords::default();
        // Indexes the PLAN, not the records: a claim that loses a race is skipped,
        // so the record count drifts from the plan position and using it to slice
        // `pool_of` would pair later moves with the wrong pool's size class — a
        // wrong `byte_lens` for the copy and a wrong stride for the address.
        // Every slot this pass plans to read, so a claim cannot hand one of them back
        // as a destination — see the note in `claim_moves`. Built from the whole plan
        // rather than per batch, because the collision is across batches: the plan is
        // complete before the first claim, and a destination claimed in the last batch
        // can name a source planned in the first.
        let planned_sources: HashSet<(usize, u32)> =
            pending.iter().map(|m| (m.from.0, m.from.1)).collect();
        let mut planned = 0usize;
        for batch in pending.chunks(MOVES_PER_BATCH) {
            // **The first batch is unconditional.** Planning has already spent part of
            // the budget, and a pass that plans and then claims nothing is worse than
            // no pass at all: it pays the census, the window and the refusal, and
            // leaves the pool exactly as it found it. One batch guarantees every
            // attempt converges by at least 512 slots.
            if planned > 0 && started.elapsed() >= budget {
                report.clipped = true;
                break;
            }
            let keys = &pool_of[planned..planned + batch.len()];
            planned += batch.len();
            // **Passed up, not absorbed.** This used to warn, set `clipped` and
            // carry on to the sweep and the publish — written for a lost claim,
            // where keeping what landed is exactly right. But a lost claim never
            // arrives here: `claim_moves` skips one with a `continue`, because the
            // census is a snapshot and a slot it believed free may have gone. The
            // only things that come out of it as `Err` are an arena table that
            // would not resolve and a plan naming ground no arena owns — faults
            // that recur on the next pass and whose implied addresses land in
            // another tenant's region. So the soft handling caught none of the
            // cases it was for and absorbed every case it was not.
            if let Err(e) = self.claim_moves(batch, keys, &planned_sources, &mut map, &mut records)
            {
                return Err(CompactionRefused::Fault(e.to_string()));
            }
        }
        if map.is_empty() {
            return Err(CompactionRefused::ClaimsLost);
        }
        // From here until the pass ends, the slots it is about to move are declared
        // immutable and any instrumented write into them names its writer. See
        // [`ReadonlySources`] for why it is the sources and not the whole band.
        #[cfg(feature = "tensor-assert")]
        let _sources = ReadonlySources::declare(self.device(), &records);

        // **Everything here that costs a readback is behind `tensor-assert`, and the
        // reason is a measurement it invalidated.**
        //
        // Left unconditional, these checks read from the device per chunk and compared
        // both ends of every copied record — megabytes off the device per pass. A run
        // then completed ONE compaction pass where an uninstrumented run completed
        // twenty, so the workload under observation was not the workload being
        // debugged, and "no faults found" meant "no passes ran". That is the harness's
        // own first danger: an instrument that fences suppresses the race it hunts.
        //
        // What stays unconditional is what costs nothing to be right about: the
        // read/write overlap refusal (host arithmetic), the band-pointer verify (one
        // four-byte readback), the source-collision rejection (a host set lookup) and
        // the pool's ownership guard (one indexed bool).
        //
        // **Every reference every live slot holds, and one content hash per slot.**
        //
        // Complete rather than sampled. A gid or a record pointer naming a slot the
        // refcount tables call free is ground the allocator will reissue while that band
        // still points there; reported here so the first boundary at which a slot's
        // references go bad names the operation that broke them. The hash is what the
        // closing boundary compares against: this pass rewrites addresses and must leave
        // every slot's bytes exactly as they are, so any slot whose hash moves across the
        // pass has a party naming another chunk's K/V.
        //
        // Not free — it walks every band of every slot of every backing, reads each
        // chunk's resident record back, and launches the hash — which is why it is behind
        // the harness rather than standing in the pass.
        #[cfg(feature = "tensor-assert")]
        let entering =
            super::kv_integrity::report_boundary(backings, "entering a compaction", 0, 0, 0, None);

        report.timings.claim = phase.elapsed();
        phase = Instant::now();
        // One launch for every relocated slot, whatever the pool or the rung —
        // `byte_lens` is per record, so a pass spanning the whole ladder is still
        // one grid. A per-move `memcpy_dtod_async` measured ~8 µs of launch
        // overhead each, which put 1,024 moves in an 8 ms budget and left every
        // pass clipped with the frontier exactly where it started.
        if let Err(e) = self.copy_records(&records) {
            tracing::error!(
                target: "candle_nn::kv_cache::compact",
                "compaction copy launch failed, abandoning the pass: {e}",
            );
            // Nothing is rewritten yet, so every holder still names its source
            // slot and the claimed destinations are merely wasted.
            return Err(CompactionRefused::CopyFailed);
        }
        report.moves = records.srcs.len();
        report.source_collisions = records.source_collisions;
        report.timings.copy = phase.elapsed();
        phase = Instant::now();

        // **Bytes before pointers.** The records rewritten below name the
        // destinations; nothing may read through them until the copies landed.
        //
        // Device-wide, not `stream.synchronize()`. It used to be the stream's, with
        // a comment saying it was written that way so "a future side stream cannot
        // silently break it" — which is the opposite of what a stream-scoped
        // barrier does. FIFO covers this pass's own copy-then-patch ordering because
        // both are on `stream`; it says nothing about the persistence thread's copy
        // stream, which is a reader of the very slots being vacated.
        if let Err(e) = self.device().synchronize() {
            tracing::error!(
                target: "candle_nn::kv_cache::compact",
                "compaction barrier failed, abandoning the pass: {e}",
            );
            // Nothing has been rewritten, so the old slots are still every
            // holder's truth and the destinations are merely wasted.
            return Err(CompactionRefused::BarrierFailed);
        }

        // **Did the copy reproduce the bytes?** Sampled, host-side, after the fence.
        //
        // Every other check in this pass is about *addressing* — that a gid names the
        // slot it should, that a record holds the pointer it was given, that a holder
        // was reached. All of them can pass while the bytes themselves are wrong, and
        // the reader cannot tell: quantised KV is a byte string, so a destination
        // holding the wrong bytes decodes to finite, plausibly-shaped, wrong values
        // and surfaces as a NaN several layers later.
        //
        // Behind the feature because it reads BOTH ENDS OF EVERY RECORD — megabytes
        // off the device per pass, which is what starved a run down to a single
        // compaction. Full coverage is the point (a 16-slot sample cannot tell "the
        // copy is correct" from "the corruption missed my sample"), and full coverage
        // is exactly what makes it too expensive to leave on.
        #[cfg(feature = "tensor-assert")]
        if let Err(e) = self.verify_copied_bytes(&records) {
            tracing::error!(
                target: "candle_nn::kv_cache::compact",
                "compaction copy did not reproduce the source bytes: {e}",
            );
            // Nothing is rewritten yet — the holders still name their sources, which
            // still hold the data — so this is still the safe point to abandon.
            return Err(CompactionRefused::Fault(e.to_string()));
        }

        report.timings.barrier = phase.elapsed();
        phase = Instant::now();

        // ── Rewrite every holder: ours, then the caller's ────────────────────
        // **The band count comes from the backing, not from the GQA constant.**
        //
        // `n_palette()` is `LATENT_N_BANDS` on a single-latent backing and
        // `N_PALETTE` for GQA, and its own doc says it drives the KvHead record
        // layout, which is exactly what `emit_patch` indexes. Hardcoding the GQA
        // constant is correct for GQA and silently wrong for the latent path: the
        // record stride and every band offset would be computed for a quarter of the
        // bands there, so the patch would write correct addresses into the wrong
        // words — over the palette, format and scale fields of an earlier head — and
        // leave the upper bands naming vacated slots. Neither shows up as a fault,
        // and the patch verifier cannot see it either, because each word does hold
        // the value it was given.
        let geometry = RecordGeometry {
            n_kv_head: self.n_kv_head(),
            head_dim: self.head_dim(),
            n_palette: self.n_palette(),
        };
        let mut sweep_state = Sweep::new(&map, geometry);
        // **Every backing, one sweep.** Sharing the `Sweep` across them is not an
        // optimisation: layers routinely hold the SAME `HeadGids` allocation, so a
        // per-backing sweep would give each its own equal-but-distinct replacement
        // and the refcounts would then disagree with the sharing the cache believes
        // exists.
        for b in backings {
            if let Err(e) = b.rewrite_own_holders(&mut sweep_state) {
                tracing::error!(
                    target: "candle_nn::kv_cache::compact",
                    "compaction could not rewrite a backing's block tables: {e}",
                );
                return Err(CompactionRefused::HolderRewriteFailed);
            }
        }
        if let Err(e) = sweep(&mut sweep_state) {
            // The caller could not finish. Its holders still name the old slots,
            // which are still live and still hold the bytes — so refusing here
            // leaves a consistent, if unpacked, pool. This is the branch that
            // exists so an incomplete sweep is a refusal and not a corruption.
            tracing::error!(
                target: "candle_nn::kv_cache::compact",
                "caller's sweep failed; compaction abandoned with nothing published: {e}",
            );
            return Err(CompactionRefused::SweepFailed);
        }
        // **Every relocated slot must be named by a holder this sweep rewrote.**
        //
        // The pass is about to free the sources. That is sound only if nothing still
        // points at them, and the sweep is what makes it so — it walks the backings'
        // block tables and the caller's residences and rewrites each gid it finds.
        // A holder it does not reach keeps naming the source, the source is handed to
        // the next claim, and the stale holder then reads another sequence's KV:
        // finite, plausibly shaped and wrong, which is why it surfaces as a NaN many
        // layers later rather than as a fault.
        //
        // Counted against the destinations rather than against `patched_words`,
        // because those are different questions. A live chunk window carries no
        // device record and contributes no patch word, so `patched < moves` is
        // ordinary and says nothing. A destination no holder named at all is not
        // ordinary.
        report.unwitnessed = sweep_state.unwitnessed(&map).len();
        let (no_record, dup_record) = sweep_state.patch_skips();
        report.patch_no_record = no_record;
        report.patch_dup_record = dup_record;
        if report.unwitnessed != 0 {
            // **Reported, not fatal, and the distinction is honest rather than
            // lenient.** An unwitnessed destination is certainly a defect: the pass
            // claimed a slot, copied into it, and no holder it reached names the
            // result, so the claim is wasted and the source's holder will never be
            // corrected. Whether it is also *corruption* is a separate question this
            // check cannot answer — the unreached holder still owns a `ChunkGid`, so
            // its source keeps a refcount and the pool cannot re-tenant that slot,
            // which is the very thing corruption would require.
            //
            // Killing the daemon over a fault whose severity is unestablished would
            // be the wrong trade, and so would staying silent. It is counted on the
            // report and logged at `error` with its attribution.
            tracing::error!(
                target: "candle_nn::kv_cache::compact",
                moved = map.len(),
                unwitnessed = report.unwitnessed,
                "compaction relocated slots no holder it reached names — the sweep's \
                 holder set is incomplete, so those claims are wasted and their \
                 sources will never be rewritten",
            );
        }
        report.allocations_rewritten = sweep_state.allocations_rewritten();
        report.timings.sweep = phase.elapsed();
        phase = Instant::now();

        // ── Patch the device records, then fence ─────────────────────────────
        let words = sweep_state.into_patch_words();
        report.patched_words = words.len();
        // **Past this line the pass has committed, so a failure here is fatal and
        // not a refusal.**
        //
        // Every holder above has already had its gids rewritten in place — the
        // block tables and the caller's sealed sequences now name the destination
        // slots. The device records are the other half of that same publish, and
        // the patch is what performs it. If it does not happen:
        //
        // - the records still name the source bands, while every holder names the
        //   destinations, and
        // - `release_empty_arenas` below then frees the sources, so the records
        //   point into ground the pool has handed back.
        //
        // There is nothing to return to. The rewrite is not undoable — it installed
        // replacements across backings, the substrate and the projection caches
        // sharing one allocation — so reporting a refusal would claim "nothing
        // published" of a pool that is already half published, and the reader of
        // that lie is a scheduler that will go on decoding from it. Wrong attention
        // on every layer, no fault, discovered as a wrong answer.
        //
        // So it fails loudly instead. This is the same judgement as the sweep
        // branch above — an incomplete pass must never be mistaken for a complete
        // one — but the branch above can still refuse because it runs *before*
        // anything is installed, and this one cannot.
        if let Err(e) = self.patch_band_pointers(&words) {
            panic!(
                "compaction published {} relocations and then could not patch the \
                 {} band pointers naming them: {e}. Every holder now names the \
                 destination slots while the device records still name the sources, \
                 and the rewrite cannot be undone — continuing would decode from \
                 records pointing at ground this pass is about to release.",
                map.len(),
                words.len(),
            );
        }

        report.timings.patch = phase.elapsed();
        phase = Instant::now();

        // No separate invalidation step: `SequenceState::rewrite_for_compaction`
        // throws away its own cached decode buffer when it moves anything, which is
        // what makes the step impossible to omit rather than merely documented.

        // Device-wide, not per-stream: one barrier at between-forwards cadence
        // buys the removal of the entire overlap failure domain, side streams this
        // module does not know about included.
        // Fatal for the same reason as the patch above, and one step worse: this
        // barrier is what makes the copies and the patch visible to every other
        // stream before the next line hands their source ground back. Carrying on
        // without it releases regions while kernels may still be reading them, and
        // the pool's next tenant then zeroes ground a live reader is in — which
        // reports as an illegal address in some unrelated kernel, or as nothing at
        // all.
        if let Err(e) = self.device().synchronize() {
            panic!(
                "compaction published {} relocations and then could not fence them: \
                 {e}. The copies and the band-pointer patch are not known to have \
                 retired, and the next step returns their source regions to the \
                 pool — releasing ground that may still have readers.",
                map.len(),
            );
        }

        // **Every reference the pass leaves behind, re-checked now that it has
        // published, and every slot's content against what it held on the way in.** An
        // orphan count that rose across the pass means it rewrote gids and did not update
        // every record that named them, which is the fault that frees a slot while a
        // record still points at it. A content hash that moved means that fault has
        // already been cashed in: some party is reading ground another chunk now owns.
        #[cfg(feature = "tensor-assert")]
        super::kv_integrity::report_boundary(
            backings,
            "after a compaction",
            report.moves,
            report.patch_no_record,
            report.patch_dup_record,
            Some(&entering),
        );

        // The old slots' gids died with the replacements installed above, so the
        // arenas they emptied are tombstoneable now.
        report.arenas_released = self.release_empty_arenas().unwrap_or(0);
        report.frontier_after = self.frontier_regions().unwrap_or(frontier_before);
        report.timings.publish = phase.elapsed();
        Ok(report)
    }

    /// The pool that should be given one fresh low arena this pass, or `None`.
    ///
    /// Answers a question a per-pool pack cannot: which pool is *holding the
    /// frontier up*, and can it get out of its own way? The pool owning the highest
    /// arena in the span is the only one whose position costs anything, and it needs
    /// help only when its lower arenas cannot absorb that arena's live chunks —
    /// otherwise [`plan_pool`] drains it unaided.
    ///
    /// `None` when there are no holes, because a fresh arena would then claim a
    /// region *above* the frontier and raise the very number the pass exists to
    /// lower.
    fn lowest_hole_destination(
        &self,
        census_by_pool: &[(ArenaKey, Vec<super::compact_plan::ArenaSlots>)],
    ) -> Option<ArenaKey> {
        let candle::DeviceLocation::Cuda { gpu_id } = self.device().location() else {
            return None;
        };
        let stats = super::region_pool::region_stats(gpu_id)?;
        // No stranded region below the frontier means no low ground to claim.
        if stats.live_watermark <= stats.live {
            return None;
        }
        // The highest-ranked arena anywhere, and the pool that owns it.
        let (key, top_rank) = census_by_pool
            .iter()
            .flat_map(|(k, c)| c.iter().map(move |a| (*k, a.rank)))
            .max_by_key(|(_, rank)| *rank)?;
        let pool = &census_by_pool.iter().find(|(k, _)| *k == key)?.1;
        let top = pool.iter().find(|a| a.rank == top_rank)?;
        // Room below it, in slots, against what it is holding. A pool that can
        // already absorb its own top arena is packed by the walk alone.
        let room_below: usize = pool
            .iter()
            .filter(|a| a.rank < top_rank)
            .map(|a| a.capacity.saturating_sub(a.occupied.len()))
            .sum();
        if top.occupied.is_empty() || room_below >= top.occupied.len() {
            return None;
        }
        Some(key)
    }

    /// One past the highest live region on this device — the frontier, in regions.
    pub fn frontier_regions(&self) -> Option<usize> {
        let candle::DeviceLocation::Cuda { gpu_id } = self.device().location() else {
            return None;
        };
        #[cfg(feature = "cuda")]
        return super::region_pool::region_stats(gpu_id).map(|s| s.live_watermark);
        #[cfg(not(feature = "cuda"))]
        {
            let _ = gpu_id;
            None
        }
    }

    /// Claim each move's destination slot and record the copy it implies.
    ///
    /// Host-only: no device work happens here. Every claim appends one record to
    /// `records`, which [`Self::copy_records`] then issues as a single launch.
    ///
    /// A claim that fails is skipped rather than fatal: the census is a snapshot
    /// (its occupancy bitmaps are sampled word by word), so a slot it believed free
    /// can have gone. Skipping costs reclaim and nothing else — the source is
    /// untouched and every holder still names it.
    #[cfg(feature = "cuda")]
    fn claim_moves(
        &self,
        moves: &[ChunkMove],
        keys: &[ArenaKey],
        planned_sources: &HashSet<(usize, u32)>,
        map: &mut CompactionMap,
        records: &mut CopyRecords,
    ) -> Result<()> {
        // Resolve only the arenas this batch touches, both ends.
        let needed: HashSet<usize> = moves.iter().flat_map(|m| [m.from.0, m.to.0]).collect();
        let info = self.resolve_arena_info_for(&needed)?;

        for (m, key) in moves.iter().zip(keys) {
            let Some(new_gid) = self.inner.pool.allocate_from_arena(*key, m.to.0) else {
                continue;
            };
            // The allocator is asked for a NAMED arena, never "wherever you
            // like": left to itself it hands back the leftmost arena with room,
            // and draining an arena is precisely what makes it the arena with the
            // most room — so the better a pass worked the more likely each next
            // chunk would land back where it came from.
            debug_assert_eq!(new_gid.arena_idx(), m.to.0);
            // **A destination may not be one of this pass's own sources.**
            //
            // The census is a snapshot, and this is the mirror of the case the
            // comment above records. A slot it saw OCCUPIED is planned as a source;
            // if its chunk is freed before the claims run, the slot is genuinely free
            // and the allocator hands it out — correctly — as some other move's
            // destination. Nothing downstream notices, because both halves are
            // individually legitimate.
            //
            // What it produces is a read/write race *inside one launch*. Every record
            // is a concurrent block, so the record copying INTO the slot runs against
            // the record copying OUT of it, and the second one's chunk arrives as a
            // mixture or as the first one's data entire. Its holder is then rewritten
            // to name ground holding another sequence's K/V — finite, plausibly
            // shaped, and wrong, which is why it surfaces as a NaN in the first
            // attention layer rather than as a fault.
            //
            // The byte check after the copy cannot see it: by the time it compares,
            // the source has been overwritten with the same bytes the destination
            // holds, so the two agree.
            //
            // Skipped rather than repaired, because there is nothing to repair. The
            // claim is dropped with `new_gid` at the `continue`, which frees the slot
            // again, and the move is simply left for the next pass — by which time
            // the census will see the slot as free and never plan it as a source.
            if planned_sources.contains(&(new_gid.arena_idx(), new_gid.chunk_idx() as u32)) {
                note(11, 1);
                records.source_collisions += 1;
                continue;
            }
            // **A slot index past its arena's capacity is an invariant breach, and
            // must be told apart from the ordinary skips below.**
            //
            // `slot_addr` returns `None` for three different things: no such arena,
            // a CPU arena with no device slab, and an index past the arena's end.
            // The first two are ordinary — the census is a snapshot — and skipping
            // them costs reclaim and nothing else. The third is not ordinary: it
            // means the plan named ground this arena does not own, and the address
            // it would have produced lands in the next region, whose tenant may be
            // another arena, the wave transient tier, or a sequence's recurrent
            // store. Sharing one silent `continue` with the benign cases would
            // hide it completely, which is how it stayed hidden.
            for (which, arena_idx, chunk_idx) in [
                ("source", m.from.0, m.from.1 as usize),
                ("destination", new_gid.arena_idx(), new_gid.chunk_idx()),
            ] {
                if let Some(a) = info.get(arena_idx) {
                    if chunk_idx >= a.chunk_capacity as usize {
                        candle::bail!(
                            "compaction planned a {which} slot {chunk_idx} in arena \
                             {arena_idx}, which holds only {} slots. The address that \
                             implies is {} B past the arena's end, inside whatever \
                             tenant holds the next region — and every address in the \
                             reservation is mapped, so writing it would not fault.",
                            a.chunk_capacity,
                            (chunk_idx + 1 - a.chunk_capacity as usize)
                                * a.chunk_byte_stride as usize,
                        )
                    }
                }
            }
            let (Some((src, src_stride)), Some((dst, dst_stride))) = (
                slot_addr(&info, m.from.0, m.from.1 as usize),
                slot_addr(&info, new_gid.arena_idx(), new_gid.chunk_idx()),
            ) else {
                continue;
            };
            // **The length comes from the arenas, and both ends must agree.**
            //
            // It used to be `key.class.bytes()` — the size class's stride — while
            // the two addresses above stepped by the *arena's* stride. Those are
            // equal by convention rather than by construction, and
            // `ResolvedArenaInfo::chunk_byte_stride` says in as many words that
            // the class figure "is generally larger than the payload" and must
            // never be used as a copy length.
            //
            // A length longer than the destination slot does not fault. It writes
            // into the slot above, and from the arena's last slot it writes into
            // the next region — whose tenant may be another arena, the wave
            // transient tier, or a sequence's recurrent store, none of which will
            // notice until something reads a wrong number out of it. Taking the
            // length from the same table that produced the addresses removes the
            // second source of truth, and comparing the two ends catches a plan
            // that paired slots of unequal classes, which no in-bounds check on
            // either end alone would see.
            if src_stride != dst_stride {
                candle::bail!(
                    "compaction planned a copy between slots of unequal extent: \
                     arena {} slot {} holds {src_stride} B, arena {} slot {} holds \
                     {dst_stride} B. A pool is one size class, so this is a planner \
                     or arena-table fault, and copying either length would write \
                     outside one of the two slots.",
                    m.from.0,
                    m.from.1,
                    new_gid.arena_idx(),
                    new_gid.chunk_idx(),
                )
            }
            records.srcs.push(src as i64);
            records.dsts.push(dst as i64);
            records.lens.push(dst_stride);
            let old_raw = (m.from.0 * super::types::GID_STRIDE + m.from.1 as usize) as i64;
            map.insert(old_raw, new_gid, dst);
        }
        Ok(())
    }

    /// Copy every claimed slot in one launch of the migration scatter/gather
    /// kernel.
    ///
    /// One CUDA block per record, a 16-byte vectorised body when the record's ends
    /// and length are all aligned — which for a size-class slot they are, the ladder
    /// being powers of two from 320 B up. Three `i64` arrays cross the bus and
    /// nothing else; the kernel resolves no tables and reads no metadata, because
    /// the addresses were computed by the claim walk that produced the records.
    #[cfg(feature = "cuda")]
    fn copy_records(&self, records: &CopyRecords) -> Result<()> {
        if records.srcs.is_empty() {
            return Ok(());
        }
        // Before the launch, because after it the evidence is gone: the racing pair
        // leaves the source holding the destination's bytes, so every after-the-fact
        // comparison agrees.
        disjoint_ends(records)?;
        let candle::Device::Cuda(cuda) = self.device() else {
            return Ok(());
        };
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::kernels;

        let d_src = cuda.memcpy_stod(&records.srcs)?;
        let d_dst = cuda.memcpy_stod(&records.dsts)?;
        let d_len = cuda.memcpy_stod(&records.lens)?;
        let stream = cuda.cuda_stream();
        let (sp, _sg) = d_src.device_ptr(&stream);
        let (dp, _dg) = d_dst.device_ptr(&stream);
        let (lp, _lg) = d_len.device_ptr(&stream);
        // Every destination this launch will write, against ground declared
        // immutable after load. The kernel is plan-driven and has no bound of its
        // own, and the weight zone shares the reservation with the arenas — so a
        // destination computed from the wrong base writes expert weights and
        // raises nothing. Host-side, so the offender is named rather than the
        // victim discovered downstream.
        #[cfg(feature = "tensor-assert")]
        for (d, l) in records.dsts.iter().zip(records.lens.iter()) {
            candle::readonly_regions::forbid_write(
                "kv compaction destination",
                *d as u64,
                (*l).max(0) as usize,
            );
        }
        // SAFETY: each record names one slot of the reservation, `lens[r]` bytes
        // long, in arenas of the same class and the same memory (the plan is per
        // `ArenaKey`, so location cannot differ). Each destination is held by this
        // pass's own claim; each source is read-only here and stays live until the
        // holders that name it are rewritten below.
        unsafe {
            candle::set_kernel_breadcrumb("run_kv_migrate_copy", file!(), line!());
            kernels::simple::kv_migrate::run_kv_migrate_copy(
                sp as *const i64,
                dp as *const i64,
                lp as *const i64,
                std::ptr::null::<i64>(),
                std::ptr::null::<i64>(),
                std::ptr::null::<i64>(),
                records.srcs.len() as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
        Ok(())
    }

    /// One resident `KvHead` record, as the device holds it.
    ///
    /// The whole record in one transfer rather than a word per band: the integrity
    /// check reads every band pointer of every chunk, and a chunk's record is a couple
    /// of kilobytes against `n_kv_head * n_palette * 2` separate synchronous reads.
    #[cfg(all(feature = "cuda", feature = "tensor-assert"))]
    pub(super) fn read_record(&self, addr: u64, bytes: usize) -> Option<Vec<u8>> {
        let candle::Device::Cuda(cuda) = self.device() else {
            return None;
        };
        cuda.bind_to_thread().ok()?;
        let mut out = vec![0u8; bytes];
        // SAFETY: `addr` is a resident record's base address, taken from the chunk's
        // own `MetaGid`, and `bytes` is that record's serialized length computed from
        // the same geometry the serializer used.
        unsafe {
            candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_sync(&mut out, addr).ok()?;
        }
        Some(out)
    }

    /// Compare every one of the pass's destinations against its source.
    ///
    /// Called after the copy's fence, while both ends are still live — the sources are
    /// not released until the end of the pass, so this is the one window in which the
    /// two can be compared at all.
    ///
    /// Reads raw device addresses rather than tensors because that is what the records
    /// are: `srcs`/`dsts` are the addresses the migrate kernel was given, and checking
    /// anything else would be checking a different claim.
    #[cfg(feature = "tensor-assert")]
    fn verify_copied_bytes(&self, records: &CopyRecords) -> Result<()> {
        /// Bytes this check may pull back per pass, over both ends.
        ///
        /// **Sized to cover a whole pass, not to sample one.** It began at sixteen
        /// slots, which against ten thousand moves is 0.16% coverage — and a check
        /// that inspects one slot in six hundred cannot distinguish "the copy is
        /// correct" from "the corruption missed my sample", which is the only
        /// question it exists to answer. A pass moves a few MB, so reading both ends
        /// of all of it is well under a millisecond on any of the three machines'
        /// links, against a pass that already synchronises the device twice.
        const BUDGET_BYTES: usize = 32 << 20;

        let candle::Device::Cuda(cuda) = self.device() else {
            return Ok(());
        };
        let n = records.srcs.len();
        if n == 0 {
            return Ok(());
        }
        cuda.bind_to_thread().map_err(candle::Error::wrap)?;
        let mut src_buf: Vec<u8> = Vec::new();
        let mut dst_buf: Vec<u8> = Vec::new();
        let mut spent = 0usize;
        let mut checked = 0usize;
        for i in 0..n {
            let len = records.lens[i].max(0) as usize;
            if len == 0 {
                continue;
            }
            // Stop at the budget rather than thinning across the plan: the records
            // are claimed in pool order, so a prefix is one size class — but a
            // complete prefix still proves something about the slots it covered,
            // where a 0.16% stride proves nothing about any of them.
            spent += len * 2;
            if spent > BUDGET_BYTES {
                tracing::debug!(
                    target: "candle_nn::kv_cache::compact",
                    checked, moves = n,
                    "copy verification stopped at its byte budget",
                );
                break;
            }
            checked += 1;
            src_buf.resize(len, 0);
            dst_buf.resize(len, 0);
            // SAFETY: both addresses name `len` bytes of one chunk slot inside the
            // reservation — the same pair the migrate kernel was handed — and the
            // caller has fenced the copy that filled the destination.
            unsafe {
                candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_sync(
                    &mut src_buf,
                    records.srcs[i] as u64,
                )
                .map_err(candle::Error::wrap)?;
                candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_sync(
                    &mut dst_buf,
                    records.dsts[i] as u64,
                )
                .map_err(candle::Error::wrap)?;
            }
            if src_buf != dst_buf {
                let at = src_buf
                    .iter()
                    .zip(&dst_buf)
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                // **Which end moved?** A mismatch after the fence has two readings
                // and they call for opposite fixes: the copy did not land, or the
                // SOURCE was mutated after it did — in which case the destination is
                // correct and the bug is a concurrent writer. Reading both a second
                // time separates them, because a value that changes between two host
                // reads is being written now, and one that does not was already
                // wrong when the copy retired.
                let mut src2 = vec![0u8; len];
                let mut dst2 = vec![0u8; len];
                // SAFETY: as the reads above — same two addresses, same length.
                unsafe {
                    let _ = candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_sync(
                        &mut src2,
                        records.srcs[i] as u64,
                    );
                    let _ = candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_sync(
                        &mut dst2,
                        records.dsts[i] as u64,
                    );
                }
                let src_moved = src2 != src_buf;
                let dst_moved = dst2 != dst_buf;
                let verdict = match (src_moved, dst_moved) {
                    (false, false) => {
                        "both ends stable across two reads — the copy \
                                       did not land"
                    }
                    (true, false) => {
                        "the SOURCE changed between two reads — a \
                                      concurrent writer, and the destination may be \
                                      correct"
                    }
                    (false, true) => {
                        "the DESTINATION changed between two reads — \
                                      something is writing the slot this pass just \
                                      filled"
                    }
                    (true, true) => {
                        "both ends changed between two reads — the pass \
                                     is running against live writers"
                    }
                };
                tracing::error!(
                    target: "candle_nn::kv_cache::compact",
                    record = i, of = n, len, %verdict,
                    src = format_args!("{:#x}", records.srcs[i]),
                    dst = format_args!("{:#x}", records.dsts[i]),
                    "compaction copy verification: {verdict}",
                );
                candle::bail!(
                    "record {i} of {n}: {len} B from {:#x} to {:#x} differ at byte \
                     {at} (source {:#04x}, destination {:#04x}). The destination does \
                     not hold what the source holds, so every holder this pass is \
                     about to point at it will read bytes that decode to plausible, \
                     wrong values.",
                    records.srcs[i],
                    records.dsts[i],
                    src_buf[at],
                    dst_buf[at],
                )
            }
        }
        Ok(())
    }

    /// Rewrite this backing's own block tables, returning the slots touched.
    ///
    /// The live side of the sweep: one `ChunkWindow` per block per slot, whose
    /// `gids` are frequently the very same allocation a sealed chunk holds — an
    /// injected sealed chunk, a view borrowing its parent's blocks, a fork. They
    /// share one `Sweep`, so such an allocation is rewritten once and both holders
    /// receive the same replacement; rewriting them independently would leave two
    /// holders with equal-but-distinct gids and refcounts that disagree with the
    /// sharing the cache believes exists.
    ///
    /// The returned slots are what must have their cached decode buffers thrown
    /// away — see the note on `invalidate_decode_slot` in [`Self::compact`].
    fn rewrite_own_holders(&self, sweep: &mut Sweep<'_>) -> Result<usize> {
        let mut touched = 0usize;
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        for entry in state.sequences.iter_mut() {
            let Some(seq) = entry.as_mut() else { continue };
            // `rewrite_for_compaction` invalidates the slot's cached decode buffer
            // itself when it moves anything, so there is no separate invalidation
            // step for this pass to forget.
            if seq.rewrite_for_compaction(sweep)? {
                touched += 1;
            }
        }
        Ok(touched)
    }

    /// Store each patch word with the device kernel, in one launch.
    ///
    /// Only the table crosses the bus. The alternative — `build_meta_records` per
    /// moved chunk — serialises a whole record on the host and ships it, which for
    /// a few thousand moves is a few thousand small uploads.
    #[cfg(feature = "cuda")]
    fn patch_band_pointers(&self, words: &[PatchWord]) -> Result<()> {
        if words.is_empty() {
            return Ok(());
        }
        let candle::Device::Cuda(cuda) = self.device() else {
            return Ok(());
        };
        use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
        use candle::cuda_backend::kernels;

        // Two arrays, one upload each: the only thing that crosses the bus for the
        // whole patch. Everything else is written by the kernel in place.
        let addrs: Vec<u64> = words.iter().map(|w| w.addr).collect();
        let vals: Vec<u64> = words.iter().map(|w| w.value).collect();
        let d_addrs = cuda.memcpy_stod(&addrs)?;
        let d_vals = cuda.memcpy_stod(&vals)?;
        let stream = cuda.cuda_stream();
        let (a, _ga) = d_addrs.device_ptr(&stream);
        let (v, _gv) = d_vals.device_ptr(&stream);
        candle::set_kernel_breadcrumb("run_kv_ptr_patch", file!(), line!());
        unsafe {
            kernels::simple::kv_ptr_patch::run_kv_ptr_patch(
                a as *const u64,
                v as *const u64,
                words.len() as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }

        // **"The patch ran" is not evidence that it landed. This is.**
        //
        // The verifier was written with the patch and never called, while both its
        // FFI doc and its `.cu` header claimed it backed "the compaction's own
        // self-check". Nothing checked anything: the launch is asynchronous and
        // unchecked, and a band pointer left stale does not fault — every address in
        // the reservation is mapped, so the record reads whatever now occupies the
        // vacated slot. It surfaces as a wrong number many layers downstream, in a
        // different subsystem, long after the pass that caused it returned "ok".
        //
        // One launch and one 4-byte readback per pass, at between-forwards cadence,
        // on a path that already synchronises the device twice. Cheap enough to be
        // unconditional, and being unconditional is the point — a proof compiled out
        // of the build where the fault happens proves nothing.
        let mut d_bad = cuda.memcpy_stod(&[0u32])?;
        {
            let (b, _gb) = d_bad.device_ptr_mut(&stream);
            candle::set_kernel_breadcrumb("run_kv_ptr_verify", file!(), line!());
            unsafe {
                kernels::simple::kv_ptr_patch::run_kv_ptr_verify(
                    a as *const u64,
                    v as *const u64,
                    words.len() as i32,
                    b as *mut u32,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }
        let bad = cuda.memcpy_dtov(&d_bad)?;
        let stale = bad.first().copied().unwrap_or(0);
        if stale != 0 {
            candle::bail!(
                "compaction patched {} band pointers and {stale} of them do not hold \
                 the value they were given. Those records still name the slots this \
                 pass is about to free, so the next claim re-tenants that ground and \
                 the record reads another sequence's KV — finite, plausibly shaped \
                 and wrong, with no fault anywhere.",
                words.len(),
            )
        }
        Ok(())
    }
}

/// Refuse a copy plan in which any slot is both read and written.
///
/// **The kernel is one concurrent block per record, so this is the whole safety
/// condition.** A record writing a slot another record reads races it, and the loser
/// copies a mixture or the winner's chunk entire — then has its holder rewritten to
/// name that ground. Nothing faults and the bytes are plausible; it arrives as a NaN
/// in the first attention layer many launches later.
///
/// `plan_pool` guarantees disjointness and a test pins it, but the guarantee does not
/// survive the claim: `claim_moves` asks the allocator for *any* free slot in the
/// destination arena rather than the one the plan named, and a slot the census saw
/// occupied — and therefore planned as a source — is genuinely free by then if its
/// chunk was released in between. Both halves are legitimate, which is why this was
/// invisible.
///
/// Checked on address *ranges*, not on `(arena, slot)` or on bare addresses. Ranges
/// are what the kernel dereferences, so a length is part of the collision: two slots
/// of different classes can intersect without sharing a start address, and a wrong
/// stride makes that the normal case rather than the exotic one.
///
/// Every written range must be disjoint from every other range in the plan. Two
/// *reads* of the same bytes are harmless — both records copy the same content out —
/// so only a write makes a pair a collision. That covers three distinct faults with
/// one comparison:
///
/// - a destination that is also a source (the census-snapshot case above),
/// - two records writing one slot, where the loser's holder ends up naming the
///   winner's chunk,
/// - a destination overlapping a *different* slot part-way, which is what a wrong
///   length or stride produces.
fn disjoint_ends(records: &CopyRecords) -> Result<()> {
    // (start, end, record, is_write)
    let mut ranges: Vec<(i64, i64, usize, bool)> =
        Vec::with_capacity(records.srcs.len() + records.dsts.len());
    for i in 0..records.srcs.len() {
        let len = records.lens[i].max(0);
        if len == 0 {
            continue;
        }
        ranges.push((records.srcs[i], records.srcs[i] + len, i, false));
        ranges.push((records.dsts[i], records.dsts[i] + len, i, true));
    }
    ranges.sort_unstable_by_key(|&(start, _, _, _)| start);
    for pair in ranges.windows(2) {
        let (a_start, a_end, a_rec, a_write) = pair[0];
        let (b_start, _b_end, b_rec, b_write) = pair[1];
        // Sorted by start, so the only way two ranges meet is the earlier one
        // reaching into the later one.
        if b_start >= a_end {
            continue;
        }
        // Two reads of the same bytes are fine: both records copy the same content.
        if !a_write && !b_write {
            continue;
        }
        let what = match (a_write, b_write) {
            (true, true) => {
                "two records WRITE overlapping ranges, so one chunk lands \
                             on top of the other and the loser's holder is rewritten \
                             to name it"
            }
            _ => {
                "one record READS a range another WRITES, so they race and whichever \
                  loses copies the other's chunk"
            }
        };
        candle::bail!(
            "compaction planned overlapping ranges: record {a_rec} \
             [{a_start:#x}, {a_end:#x}) and record {b_rec} at {b_start:#x} — {what}. \
             The migrate kernel runs one concurrent block per record, and every \
             address involved is mapped, so nothing faults and the bytes decode to \
             plausible, wrong values."
        )
    }
    Ok(())
}

/// Declares this pass's **source** slots immutable for the rest of the pass, and
/// gives them back on the way out however the pass ends.
///
/// # Why the sources, and only the sources
///
/// Every other check in this module verifies a *result* — an address is in bounds, a
/// pointer word holds what it was given, a destination holds what its source held.
/// None of them can answer "who", and by the time a byte is wrong the writer is
/// gone. This one asks the question the other way round: for the window in which
/// nothing at all is supposed to write K/V, declare the bytes this pass is about to
/// move and let `forbid_write` name any writer at the moment of the write.
///
/// Not the whole K/V band, though that is the stronger claim, because this pass
/// writes the band itself: `copy_records` fills every destination and already calls
/// `forbid_write` on each one, so declaring the band makes the pass panic on its own
/// copy. The sources are the half nothing should touch — the copy only reads them,
/// and they stay live until the holders naming them have been rewritten.
///
/// A guard rather than a pair of calls because the pass has several early returns,
/// and a declaration left standing past any of them reports the next legitimate K/V
/// write as a violation. That failure mode is in the harness's own notes: a stale
/// declared region blames an innocent allocation, which is worse than not checking,
/// because a guard that cries wolf is one you stop reading.
///
/// `release_below(weight_floor)` is the release, the same idiom `set_weight_floor`
/// uses: the floor is the top of the K/V side, so it clears exactly what was
/// declared while the expert zone above keeps its own declaration.
#[cfg(all(feature = "cuda", feature = "tensor-assert"))]
struct ReadonlySources {
    floor: u64,
}

#[cfg(all(feature = "cuda", feature = "tensor-assert"))]
impl ReadonlySources {
    fn declare(device: &candle::Device, records: &CopyRecords) -> Option<Self> {
        let candle::DeviceLocation::Cuda { gpu_id } = device.location() else {
            return None;
        };
        let layout = super::region_pool::span_layout(gpu_id)?;
        let mut spans: Vec<(u64, usize)> = records
            .srcs
            .iter()
            .zip(&records.lens)
            .map(|(&s, &l)| (s as u64, l.max(0) as usize))
            .collect();
        if spans.is_empty() {
            return None;
        }
        // Merged, because the sources of one pass are thousands of slots that sit end
        // to end inside a handful of arenas: declared individually they would both
        // overflow the table and turn the check into a thousands-entry scan.
        candle::readonly_regions::declare_merged("kv compaction source", &mut spans);
        Some(Self {
            floor: layout.weight_floor,
        })
    }
}

#[cfg(all(feature = "cuda", feature = "tensor-assert"))]
impl Drop for ReadonlySources {
    fn drop(&mut self) {
        candle::readonly_regions::release_below(self.floor);
    }
}

/// Device address of one chunk slot, and the slot's byte extent, from a resolved
/// arena table.
///
/// The same two lines `serialize_kv_heads` uses to fill a record's band pointers,
/// named once here so a compaction's copy and the record describing it cannot
/// disagree about where a slot is. The stride comes from the resolved arena rather
/// than from the size class: they agree, and taking it from the arena is what makes
/// a disagreement impossible rather than merely unlikely.
///
/// **`chunk_idx` is bounded against the arena's capacity, not against the stride.**
/// The raw-gid namespace is one fixed power of two (`GID_STRIDE`) while an arena
/// holds only `chunks_per_region` slots for its class, so an index can be perfectly
/// valid as a gid and still name ground past this arena's end —
/// [`ResolvedArenaInfo::chunk_capacity`] exists to say exactly that, and says
/// validators must compare against it. Unbounded, the address this returns walks
/// into the next region, which belongs to another tenant: the next arena, the wave
/// transient tier, or a sequence's recurrent store. Nothing faults, because every
/// address in the reservation is mapped.
///
/// Returning the extent alongside the address is what lets a caller copy without
/// inventing a length from a second source. That mattered: the copy planner took
/// its length from the size class while its addresses stepped by the arena's
/// stride, and the two are only equal by convention — see
/// [`ChunkedKvBacking::claim_moves`].
pub(super) fn slot_addr(
    info: &[crate::kv_cache::arena_table::ResolvedArenaInfo],
    arena_idx: usize,
    chunk_idx: usize,
) -> Option<(u64, i64)> {
    let a = info.get(arena_idx)?;
    if a.base_ptr == 0 {
        return None;
    }
    if chunk_idx >= a.chunk_capacity as usize {
        return None;
    }
    Some((
        a.base_ptr + chunk_idx as u64 * a.chunk_byte_stride as u64,
        a.chunk_byte_stride,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The figure the pass is judged by is the frontier's fall, not the move count
    /// — a pass that copies ten thousand chunks and leaves the frontier where it
    /// was has achieved nothing, and the report must make that legible.
    #[test]
    fn the_report_measures_the_frontier_not_the_work() {
        let busy_but_useless = CompactionReport {
            moves: 10_000,
            frontier_before: 1727,
            frontier_after: 1727,
            ..Default::default()
        };
        assert_eq!(busy_but_useless.regions_reclaimed(), 0);
        assert!(!busy_but_useless.is_empty(), "it did do work");

        let cheap_and_good = CompactionReport {
            moves: 40,
            frontier_before: 1727,
            frontier_after: 200,
            ..Default::default()
        };
        assert_eq!(cheap_and_good.regions_reclaimed(), 1527);
    }

    /// A slot index past the arena's capacity has no address, and the address of
    /// an in-range slot carries that arena's own extent.
    ///
    /// **The raw-gid namespace is wider than an arena.** `GID_STRIDE` is one fixed
    /// power of two while an arena holds `chunks_per_region` slots for its class, so
    /// an index can be a valid gid and still name ground past the arena's end.
    /// Unbounded, the copy planner turns that into a write into the next region —
    /// another arena, the wave transient tier, or a sequence's recurrent store — and
    /// because every address in the reservation is mapped it does not fault. It
    /// surfaces as whatever reads those bytes next getting a wrong number, in a
    /// different subsystem from the writer.
    ///
    /// Pure arithmetic over a hand-built table: no device, no pool, and no
    /// allocation — the bound is the whole behaviour under test.
    #[test]
    fn a_slot_past_the_arenas_capacity_has_no_address() {
        use crate::kv_cache::arena_table::ResolvedArenaInfo;
        let info = vec![ResolvedArenaInfo {
            base_ptr: 0x1_0000_0000,
            chunk_byte_stride: 512,
            chunk_capacity: 4,
        }];

        assert_eq!(slot_addr(&info, 0, 0), Some((0x1_0000_0000, 512)));
        assert_eq!(slot_addr(&info, 0, 3), Some((0x1_0000_0000 + 3 * 512, 512)));
        // One past the end is the first address that belongs to somebody else.
        assert_eq!(slot_addr(&info, 0, 4), None, "capacity is exclusive");
        assert_eq!(
            slot_addr(&info, 0, super::super::types::GID_STRIDE - 1),
            None,
            "a valid gid index is not a valid slot index"
        );
        // An arena with no device slab cannot supply an address at all.
        let cpu = vec![ResolvedArenaInfo {
            base_ptr: 0,
            chunk_byte_stride: 512,
            chunk_capacity: 4,
        }];
        assert_eq!(slot_addr(&cpu, 0, 0), None);
        assert_eq!(slot_addr(&info, 1, 0), None, "no such arena");
    }

    /// Every slot an in-range index names lies wholly inside its arena, when the
    /// copy length is the arena's own stride.
    ///
    /// This is the property the planner needs and the reason the length no longer
    /// comes from the size class: `[addr, addr + stride)` is inside
    /// `[base, base + capacity * stride)` for every valid index, which is only true
    /// while the stride used for the length is the stride used for the address.
    #[test]
    fn a_slots_extent_never_leaves_its_arena() {
        use crate::kv_cache::arena_table::ResolvedArenaInfo;
        let stride = 320i64;
        let capacity = 7u32;
        let base = 0x2_0000_0000u64;
        let info = vec![ResolvedArenaInfo {
            base_ptr: base,
            chunk_byte_stride: stride,
            chunk_capacity: capacity,
        }];
        let end = base + capacity as u64 * stride as u64;
        for idx in 0..capacity as usize {
            let (addr, len) = slot_addr(&info, 0, idx).expect("an in-range slot");
            assert!(addr >= base, "slot {idx} starts inside the arena");
            assert!(
                addr + len as u64 <= end,
                "slot {idx} ends at {} which is past the arena's end {end}",
                addr + len as u64,
            );
        }
    }

    /// A plan that reads a slot another record writes is refused before the launch.
    ///
    /// **This is the corruption, reduced to arithmetic.** The migrate kernel is one
    /// concurrent block per record, so a slot appearing on both sides is a
    /// read/write race and the losing record copies the other's chunk — which is then
    /// installed under the losing chunk's holder. It reached production because the
    /// two halves are separately legitimate: `plan_pool` keeps its sources and
    /// destinations disjoint, and `claim_moves` then asks the allocator for any free
    /// slot in the destination arena, which is allowed to be a planned source whose
    /// chunk was released after the census sampled it.
    ///
    /// Asserted on the records rather than on the plan, because the records are what
    /// the kernel is handed and the plan's own disjointness is already pinned
    /// elsewhere — it was true, and not enough.
    #[test]
    fn a_plan_that_reads_what_it_writes_is_refused() {
        // Disjoint: three moves packing downward, nothing read and written.
        let ok = CopyRecords {
            srcs: vec![0x3000, 0x3800, 0x4000],
            dsts: vec![0x1000, 0x1800, 0x2000],
            lens: vec![0x800, 0x800, 0x800],
            source_collisions: 0,
        };
        assert!(disjoint_ends(&ok).is_ok());

        // Record 0 reads 0x2000 while record 2 writes it — the collision the census
        // snapshot allows.
        let racy = CopyRecords {
            srcs: vec![0x2000, 0x3800, 0x4000],
            dsts: vec![0x1000, 0x1800, 0x2000],
            lens: vec![0x800, 0x800, 0x800],
            source_collisions: 0,
        };
        let err = disjoint_ends(&racy).expect_err("a slot read and written must refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("READS a range another WRITES"),
            "the refusal must say a read raced a write, got: {msg}",
        );

        // Two records writing one slot: the same hazard with both ends inverted, and
        // the case a source-only check cannot see.
        let double_write = CopyRecords {
            srcs: vec![0x3000, 0x3800],
            dsts: vec![0x1000, 0x1000],
            lens: vec![0x800, 0x800],
            source_collisions: 0,
        };
        let err = disjoint_ends(&double_write).expect_err("one slot written twice must refuse");
        assert!(
            err.to_string()
                .contains("two records WRITE overlapping ranges"),
            "got: {err}",
        );

        // A destination landing PART-way into another slot — what a wrong length or
        // stride produces, and what an equality test on bare addresses misses.
        let partial = CopyRecords {
            srcs: vec![0x3000, 0x3800],
            dsts: vec![0x1000, 0x1400],
            lens: vec![0x800, 0x800],
            source_collisions: 0,
        };
        assert!(
            disjoint_ends(&partial).is_err(),
            "overlap is about ranges, not equal start addresses"
        );

        // Two records reading the same bytes is harmless — both copy the same content
        // out — so it must NOT be refused.
        let shared_read = CopyRecords {
            srcs: vec![0x3000, 0x3000],
            dsts: vec![0x1000, 0x1800],
            lens: vec![0x800, 0x800],
            source_collisions: 0,
        };
        assert!(
            disjoint_ends(&shared_read).is_ok(),
            "two reads of one range are not a collision"
        );

        // An empty plan is trivially disjoint — a pass that moves nothing is not a
        // pass that races.
        assert!(disjoint_ends(&CopyRecords::default()).is_ok());
    }

    /// A pass that moved nothing reads empty, so a caller can skip its log line and
    /// its `reclaim_spare_ground` without inspecting anything else.
    #[test]
    fn a_pass_that_moved_nothing_is_empty() {
        assert!(CompactionReport::default().is_empty());
    }

    /// A frontier that somehow rose does not underflow into a vast fake reclaim.
    #[test]
    fn a_risen_frontier_reports_no_reclaim() {
        let grew = CompactionReport {
            moves: 1,
            frontier_before: 100,
            frontier_after: 140,
            ..Default::default()
        };
        assert_eq!(grew.regions_reclaimed(), 0);
    }
}
