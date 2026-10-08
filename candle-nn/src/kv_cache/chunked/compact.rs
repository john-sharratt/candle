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

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use candle::Result;

use super::arena::{ArenaKey, ArenaKind};
use super::backing::ChunkedKvBacking;
use super::compact_map::{CompactionMap, Sweep};
use super::compact_plan::{by_source_rank, plan_pool, ArenaSlots, ChunkMove, CompactPlan};
use super::fresh_arenas::FreshArenas;
use super::size_class::SizeClass;
use crate::kv_cache::ArenaLocation;

/// What one pass did, for the caller's log line and for the tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompactionReport {
    /// Chunk slots physically copied to a lower address.
    pub moves: usize,
    /// Moves the plan produced, of which the budget let `moves` through.
    pub planned_moves: usize,
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
    /// The holder sweep failed part-way — either a backing's own block tables or the
    /// caller's residences.
    ///
    /// **Partially applied, fully published, nothing released.** The sweep installs in
    /// place, so everything it reached before the error holds its new gids and its new
    /// record; the pass therefore fills those records, invalidates the cached decode
    /// buffers and fences before returning this, and skips only the release of emptied
    /// arenas. What is left is sound for the same reason an unreached holder is — each
    /// side names ground it keeps alive — and what is lost is the reclaim.
    ///
    /// One variant for both halves because they now have identical consequences; it was
    /// two while a refusal here could still claim "nothing published", which stopped being
    /// true when the sweep started minting records. Carries the message because a caller
    /// reports this rather than logging it.
    SweepFailed(String),
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

/// Moves one pass plans per pool — no more than the pass can claim.
///
/// **Planning what the budget cannot claim is what stopped compaction.** The walk is
/// ~0.12 µs a move and the claim ~0.55 µs, so a pass spends its budget on whichever
/// it is handed more of. Planned without a cap, a fragmented pool at a high frontier
/// produced 570,000–770,000 moves: the walk alone took 75–100 ms of the 80 ms budget,
/// the pass claimed only its one guaranteed batch of 512, and a top arena holding
/// 768–1,280 live chunks was never emptied — the frontier held at 376–517 regions
/// pass after pass, and Qwen3-30B-A3B's engine probe sat at 36% efficiency on the
/// RTX 4090 Laptop. The passes that did lower it had planned 75,000–145,000 and
/// claimed 60,000–115,000.
///
/// **Capping a pool keeps exactly the moves that matter.** The two-cursor walk takes
/// its sources off the right cursor, highest address first, so a capped plan is the
/// pool's top arenas — the ones holding the frontier — and what it drops is the low
/// tail the budget would never have reached. A capped plan marks the pass clipped, so
/// the next resumes from the shorter distance. At this cap even four active pools
/// plan in ~16 ms, leaving the rest of the budget for ~100,000 claims.
///
/// A capped plan is reported as `plan_capped`, not as `clipped`: it is only progress
/// when the pass also reclaimed something.
const PLANNED_MOVES_PER_POOL: usize = 32_768;

/// One pool's plan cap: [`PLANNED_MOVES_PER_POOL`], or one whole arena of the pool
/// when that is more.
///
/// **A pass must be able to empty the arena holding the frontier, not just drain
/// it.** Allocation is leftmost by address, so once a pass has packed every lower
/// arena of a pool full, the only free slots in the pool are the ones it just
/// vacated at the top — and the next allocations land exactly there. An arena left
/// part-drained is refilled before the next pass reaches it. Capped below an
/// arena's capacity, the smallest rung (52,428 slots an arena) could never be
/// emptied in one pass: on Qwen3-30B-A3B the class-0 arena at the frontier was
/// planned first on pass after pass, and its live count went 2,499 → 7,266 →
/// 4,957 → 2,750 → 5,228 with the frontier pinned at 364 regions.
fn planned_moves_cap(key: ArenaKey) -> usize {
    PLANNED_MOVES_PER_POOL.max(key.chunks())
}

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
/// # The pass rewrites gids, and MINTS a record — it never rewrites one
///
/// A chunk's location is written down twice: in the `ChunkGid`, which holds the arena
/// slot's refcount, and in the `KvHead` record's band-pointer word, which is the
/// address the paged kernels dereference. Both are published, and neither is *patched*.
///
/// The gids are rewritten through their holders, because a gid is an owning handle. A
/// relocated chunk is then given a **freshly minted** record built from its new gids
/// ([`RecordMint`](super::compact_mint::RecordMint)), and the record it had is left
/// exactly as it is for whoever still holds it. Two properties make each side sound on
/// its own:
///
/// * A record holds a clone of the `HeadGids` its words were serialized from (see
///   [`MetaGid`](super::meta_pool::MetaGid)), so **a record cannot outlive the bands it
///   names** — the allocator will not reissue a band slot while a record points at it,
///   because the record is one of that slot's refcount holders.
/// * So the rewritten holders name the destination and read a record naming the
///   destination; the holders the sweep did not reach name the source and read a record
///   naming the source, which their own record keeps alive. Neither had to be *found*.
///
/// Installing the minted record drops the old one, and when that was the chunk's last
/// holder the old record's slot frees — taking with it the clone of the *old* gids it was
/// holding, which is the last reference to the source bands. That is what lets
/// `release_empty_arenas` hand the region back, and it is the whole reclaim.
///
/// A **live writer window** carries no record (`meta: None`), is addressed from its own
/// gids, and has its cached decode buffer thrown away — so it reads and writes the
/// destination, consistently, and is never given a record. `sealed ⇒ has a record ⇒
/// immutable` and `live ⇒ meta: None` is the invariant that keeps a written chunk and a
/// record-read chunk from being the same chunk; `set_block_gids` enforces it from the
/// other side by clearing `meta` on any gid mutation, and
/// `a_writer_owned_chunk_with_a_record_is_left_alone` pins the pass's own half.
///
/// ## What a mint moves, and what must therefore be dropped
///
/// A mint changes a **record's** address. A band's address is owned — by the record — but
/// a record's own address is cached as a bare `kvheads_ptr` word in every `TokenSlice`
/// header of every slot's decode buffer, and that buffer is reused whenever the chunk
/// *count* agrees, which a compaction never changes. `rewrite_for_compaction` clears the
/// buffer of each slot whose own chunks moved, which is the set whose *band* addresses
/// changed and is not provably the set whose *record* addresses changed. So a pass that
/// minted anything clears every slot's buffer in every backing.
///
/// ## Why the previous design corrupted K/V
///
/// The pass used to rewrite the records too, and with a record owning nothing it could
/// not be made correct either way round. **A chunk has exactly one record, shared by
/// every holder of that chunk**, so a pass that rewrites some of a chunk's holders to
/// the destination and leaves the rest on the source has no correct value to put in
/// that word: whichever slot it names is kept alive only by the holders naming the same
/// slot, and when those go the others are still reading through the record into
/// re-tenanted ground. Patching to the destination, patching to the source and leaving
/// it alone were all wrong — leaving it alone only became right once the record started
/// holding a refcount.
///
/// Measured on Qwen3.8-Flash-Next with the `tensor-assert` harness: zero orphaned gids
/// against 148 orphaned records already present when a pass *began*, 4,320 pointer/gid
/// disagreements on two slots, and an engine probe answering 8/8, 5/8, 8/8, 5/8 against
/// 8/8 twice with the pass off. It was fatal on a recurrent model and survivable
/// elsewhere: wrong K/V puts a NaN in the first full-attention layer, a DeltaNet layer
/// computes its *persisted* recurrent state from it, and every later wave's logits are
/// NaN, so the model emits `!!!!!!!!` forever. The 30B ate the identical wrong K/V,
/// because it has no recurrent state to keep it in.
///
/// Three more faults were paid for on the way, each found by correlating a log line
/// against the story gate rather than by inspection, and each worth knowing because the
/// shape recurs: a record claim **promoted into a band arena** and filled over live K/V
/// (`stamp_region_promoting` widens the size class, and `ArenaKey::new` is always a band
/// key); a record claim **creating or releasing an arena mid-sweep**, which re-tenants an
/// `arena_idx` that half-rewritten holders still name — `arena slot: 2176 B requested from
/// a 1152 B slot`, 198/92/114 times on corrupt runs against 0 on clean ones; and
/// `Sweep::memo` keyed on `Arc::as_ptr` while minting made originals die mid-sweep, so a
/// replacement could land on a dead original's address and a second visit was handed
/// another chunk's gids.
///
/// `kv_integrity::report_boundary` is wired at both boundaries below under
/// `tensor-assert` and remains the standing check — but **it cannot see most of this
/// class**: it reported zero orphans and zero content changes on a run that answered 2/8,
/// because nothing was orphaned and its hash covers only bands that *live* slots' block
/// tables name. The engine probe's story gate is the oracle, and one clean run means
/// nothing: require several consecutive.
#[cfg(feature = "cuda")]
pub fn compact_backings(
    backings: &[super::backing::ChunkedKvBacking],
    budget: Duration,
    sweep: &mut dyn FnMut(&mut Sweep<'_>) -> Result<()>,
) -> std::result::Result<CompactionReport, CompactionRefused> {
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
        Err(
            CompactionRefused::ClaimsLost
            | CompactionRefused::CopyFailed
            | CompactionRefused::BarrierFailed
            | CompactionRefused::SweepFailed(_)
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
        // From the census to the end of the sweep: a slot freed on another thread
        // inside this window is what the report's gaps are made of.
        #[cfg(feature = "tensor-assert")]
        let window = super::release_watch::PassWindow::open();

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
        //
        // **The `KvHead` record pool is ranked beside the band pools.** Its arenas are
        // regions like any other, so a sparse one high in the span holds the frontier up
        // exactly as a sparse band arena does. The same walk packs it; what differs is how
        // a holder follows a moved record (`Sweep::follow_record`), not how it is planned.
        let record_key = if self.records_are_resident() {
            self.record_layout().ok().map(|l| l.key)
        } else {
            None
        };
        let mut by_rank: Vec<(usize, ArenaKey)> = SizeClass::all()
            .map(|class| ArenaKey::new(class, ArenaLocation::Gpu))
            .chain(record_key)
            .filter_map(|key| self.pool_top_rank(key).map(|(rank, _)| (rank, key)))
            .collect();
        by_rank.sort_unstable_by_key(|a| std::cmp::Reverse(a.0));
        report.pools_ranked = by_rank.len();
        let mut census_by_pool: Vec<(ArenaKey, Vec<ArenaSlots>)> = Vec::new();
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
        report.timings.census = started.elapsed();
        // Every arena the pass creates ahead of demand goes through `fresh`, which keeps
        // the empty sweep off it while the pass needs it and releases the ones nothing
        // landed in itself — see [`FreshArenas`].
        let mut fresh = FreshArenas::new(&*self.inner);
        let provision_started = Instant::now();
        self.provision_low_arenas(&mut census_by_pool, &mut fresh);
        report.timings.provision = provision_started.elapsed();
        let walk_started = Instant::now();
        // **One list over every pool, highest source region first** — see
        // [`by_source_rank`]. The budget clips this list, so its order decides which
        // regions a clipped pass empties.
        let rank_of: HashMap<usize, usize> = census_by_pool
            .iter()
            .flat_map(|(_, census)| census.iter().map(|a| (a.arena_idx, a.rank)))
            .collect();
        let plans: Vec<CompactPlan> = census_by_pool
            .iter()
            .filter_map(|(key, census)| plan_pool(census, *key, planned_moves_cap(*key)))
            .collect();
        // A pool cut short has work left, which the next pass resumes — reported apart
        // from the budget's clip; see `CompactionReport::plan_capped`.
        report.plan_capped = plans.iter().any(|p| p.clipped);
        for (m, key) in by_source_rank(&plans, |idx| rank_of.get(&idx).copied().unwrap_or(0)) {
            pending.push(m);
            pool_of.push(key);
        }
        if pending.is_empty() {
            return Err(CompactionRefused::AlreadyPacked);
        }
        report.planned_moves = pending.len();
        report.pools_censused = census_by_pool.len();
        report.first_source_rank = pending
            .first()
            .and_then(|m| rank_of.get(&m.from.0).copied());
        report.timings.walk = walk_started.elapsed();
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
        // **Every provisioned arena the plan put nothing in goes back now**, before the
        // record reservation below looks for a hole to claim. The per-pool arenas took
        // one hole each, and a reservation that found none would leave the mints nowhere
        // to land. Safe here and only here: no holder has been rewritten yet, so nothing
        // names an empty arena. See [`FreshArenas::release_unused`].
        if let Err(e) = fresh.release_unused() {
            return Err(CompactionRefused::Fault(e.to_string()));
        }
        // The record slots this pass will read, which no mint may be handed — see
        // `ChunkedKvBacking::try_alloc_record_slot`.
        let record_sources: HashSet<(usize, u32)> = pending
            .iter()
            .zip(&pool_of)
            .filter(|(_, key)| matches!(key.kind, ArenaKind::Record { .. }))
            .map(|(m, _)| (m.from.0, m.from.1))
            .collect();
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
            super::kv_integrity::report_boundary(backings, "entering a compaction", 0, 0, None);

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
        // Gids, and a freshly minted record for each rewritten chunk that had one. No
        // record is ever rewritten — see the header.
        let mut mint = match super::compact_mint::RecordMint::new(self) {
            Ok(m) => m,
            Err(e) => {
                tracing::error!(
                    target: "candle_nn::kv_cache::compact",
                    "compaction cannot mint records at this geometry, abandoning the pass \
                     before anything is published: {e}",
                );
                return Err(CompactionRefused::Fault(e.to_string()));
            }
        };
        // **Every record arena this pass needs is created HERE, before a single holder is
        // touched.** A record claim that had to create an arena mid-sweep would register
        // one, and a registration can re-tenant an `arena_idx` that holders the sweep has
        // not reached are still naming — after which their gids resolve against an arena
        // of the wrong stride. That is what `arena slot: 2176 B requested from a 1152 B
        // slot` was, 198 of them on a run that answered 1/8 against none on the runs that
        // answered 8/8. `map.len()` is the upper bound on mints, so this over-provisions
        // by design.
        if let Some(m) = mint.as_mut() {
            // **Bounded by the chunk count, not the move count.** One record per relocated
            // chunk, and a chunk covers `n_kv_head · n_palette · 2` bands — so `map.len()`
            // over-provisions by up to that factor, and every extra arena is a *region*
            // claimed from the same free list. With no hole below the frontier that claim
            // lands above it and raises the number the pass exists to lower, which is
            // exactly why `provision_low_arenas` stops in the same situation. Rounded
            // up, and at least one, so a small pass still provisions.
            let bands_per_chunk = (self.n_kv_head() * self.n_palette() * 2).max(1);
            let want = map.len().div_ceil(bands_per_chunk).max(1);
            if let Err(e) = m.reserve(self, want, record_sources, &mut fresh) {
                tracing::error!(
                    target: "candle_nn::kv_cache::compact",
                    "compaction could not provision record slots, abandoning the pass \
                     before anything is published: {e}",
                );
                return Err(CompactionRefused::Fault(e.to_string()));
            }
        }
        let mut sweep_state = match mint.as_mut() {
            Some(m) => Sweep::with_mint(&map, m, self),
            None => Sweep::new(&map),
        };
        // **Every backing, one sweep.** Sharing the `Sweep` across them is not an
        // optimisation: layers routinely hold the SAME `HeadGids` allocation, so a
        // per-backing sweep would give each its own equal-but-distinct replacement
        // and the refcounts would then disagree with the sharing the cache believes
        // exists.
        // **A sweep that fails part-way has PUBLISHED part-way, and the records it
        // minted still need their bytes.**
        //
        // `rewrite_own_holders` mutates each backing's block tables in place, and the
        // caller's closure installs per residence, so an error on backing *k* or
        // residence *k* leaves everything before it holding fresh `MetaGid`s — whose
        // slots are filled only by the batched launch below. Returning here without
        // that launch would leave live chunks pointing at record slots holding the
        // previous tenant's record: well-formed band pointers into ground that has
        // since been freed and reissued, which is the original corruption exactly.
        //
        // So the outcome is *captured* and the fill, the invalidation and the fence all
        // run regardless; only the release of the emptied arenas is skipped. What that
        // leaves is a partially-applied pass, and a partially-applied pass is sound for
        // the same reason an unreached holder is: the holders that were rewritten name
        // destinations and carry records naming destinations, the holders that were not
        // name sources and carry records naming sources, and each side keeps its own
        // ground alive. The comment this replaces claimed "nothing published", which was
        // true before a sweep minted anything.
        let swept_ok = backings
            .iter()
            .try_fold((), |_, b| {
                b.rewrite_own_holders(&mut sweep_state).map(|_| ())
            })
            .and_then(|_| sweep(&mut sweep_state));
        if let Err(e) = &swept_ok {
            tracing::error!(
                target: "candle_nn::kv_cache::compact",
                "compaction's holder sweep failed part-way; publishing what it installed \
                 and releasing nothing: {e}",
            );
        }
        // **Relocated slots no holder this sweep reached names.**
        //
        // Pure reclaim accounting now, not a correctness proxy. Nothing is freed by
        // this pass unless every party naming it let go, and the records let go only
        // when their chunk does — so an unreached holder costs the ground it is sitting
        // on and cannot cost correctness. Counted because the holder list is maintained
        // by hand and a new holder nobody sweeps is otherwise invisible: a rising
        // reading is how one is discovered.
        report.unwitnessed = sweep_state.unwitnessed(&map).len();
        #[cfg(feature = "tensor-assert")]
        {
            let foreign = window.foreign();
            if foreign != 0 {
                tracing::warn!(
                    target: "candle_nn::kv_cache::compact",
                    foreign,
                    unwitnessed = report.unwitnessed,
                    source_collisions = report.source_collisions,
                    "chunk slots were freed on other threads inside this pass",
                );
            }
        }
        if report.unwitnessed != 0 {
            tracing::error!(
                target: "candle_nn::kv_cache::compact",
                moved = map.len(),
                unwitnessed = report.unwitnessed,
                "compaction relocated slots no holder it reached names — the sweep's \
                 holder set is incomplete, so those claims are wasted and their \
                 sources are not reclaimed",
            );
        }
        report.allocations_rewritten = sweep_state.allocations_rewritten();
        report.records_moved = map.records_len();
        report.records_unfollowed = sweep_state.records_unfollowed();
        report.timings.sweep = phase.elapsed();
        phase = Instant::now();

        // ── Drop every cached decode buffer serialised from replaced ground ──
        // A `kvheads_ptr` is a record's address and the band pointers behind it are the
        // bands', both cached in a slice header and owned by nothing. The live slots are
        // already covered: `rewrite_for_compaction` clears the buffer of every slot whose
        // gids moved or whose record was followed to a copy. This catches a buffer that
        // names replaced ground through a chunk *not* in its slot's own block table — a
        // holder the sweep does not reach — and names the slot, so the finding points at
        // it. Decided from each buffer's own pins, so a buffer that names nothing replaced
        // is kept and not miscounted.
        //
        // **Before the sweep drops**, because the replaced set is identified by allocation
        // address and record raw id, which only the sweep's own pins keep from being
        // reused. The minted records' bytes are not needed for this: it is host-only.
        //
        // Fatal on a poisoned lock: every holder has been installed, so there is nothing to
        // return to, and an unchecked buffer may be naming ground the pass is about to hand
        // back.
        let replaced = sweep_state.replaced();
        // A pass whose claims all landed and whose holders all turned out to be elsewhere
        // replaced nothing; there is no buffer to check against it.
        let checked: &[ChunkedKvBacking] = if replaced.is_empty() { &[] } else { backings };
        for (layer, b) in checked.iter().enumerate() {
            match b.invalidate_decode_buffers_naming(&replaced) {
                Ok(slots) if slots.is_empty() => {}
                Ok(slots) => {
                    tracing::warn!(
                        target: "candle_nn::kv_cache::compact",
                        layer,
                        ?slots,
                        "a cached decode buffer named ground this compaction replaced, but \
                         its slot's own rewrite did not clear it — a holder of those chunks \
                         the sweep does not reach. Cleared here; its old ground stays pinned \
                         until then",
                    );
                    report.decode_buffers_cleared += slots.len();
                }
                Err(e) => panic!(
                    "compaction published {} band and {} record relocations and then could \
                     not check the cached decode buffers against them: {e}. A buffer naming \
                     replaced ground would go on being read.",
                    map.len(),
                    map.records_len(),
                ),
            }
        }
        drop(replaced);
        report.timings.invalidate = phase.elapsed();
        phase = Instant::now();

        // ── Fill the minted records, in one launch ───────────────────────────
        // **Every rewritten chunk already holds its new record; this writes the bytes.**
        // The slots were claimed during the sweep (host-only) so the holder could install
        // the handle in the same visit, and the fill is batched to here because it is a
        // device launch and there is exactly one per pass however many records were
        // minted.
        //
        // Ordered before the barrier below deliberately: that barrier is what makes these
        // writes visible to every other stream, and `release_empty_arenas` after it hands
        // back the regions the old records' deaths just emptied. Nothing reads a minted
        // record in between — the pass holds the arena window, so no forward is in flight.
        //
        // Fatal rather than a refusal, for the same reason the barrier is: every holder's
        // gids and record have already been installed, so there is nothing to return to.
        // A record left unfilled holds whatever its slot held before, which is another
        // chunk's record or uninitialised ground.
        drop(sweep_state);
        if let Some(m) = mint.as_mut() {
            report.records_declined = m.declined();
            report.record_arenas_reserved = m.reserved_arenas();
            if report.records_declined != 0 {
                // Correct but degraded: those chunks stayed whole on their sources —
                // gids and record together (`Sweep::remint`) — so nothing is reclaimed
                // for them this pass and a pass with record room moves them. A standing
                // non-zero reading means the reservation is under-provisioning.
                tracing::warn!(
                    target: "candle_nn::kv_cache::compact",
                    declined = report.records_declined,
                    minted = m.len(),
                    "compaction ran out of pre-provisioned record slots mid-sweep; those \
                     chunks stay on their sources until a later pass",
                );
            }
            match m.flush(self) {
                Ok(n) => report.records_minted = n,
                Err(e) => panic!(
                    "compaction published {} relocations and then could not fill the {} \
                     records it minted for them: {e}. Those holders now name records whose \
                     bytes were never written, so the paged kernels would dereference \
                     whatever the slots held before.",
                    map.len(),
                    m.len(),
                ),
            }
        }

        report.timings.mint = phase.elapsed();
        phase = Instant::now();

        // Device-wide, not per-stream: one barrier at between-forwards cadence
        // buys the removal of the entire overlap failure domain, side streams this
        // module does not know about included.
        // **Fatal, not a refusal: the gid rewrite above is already published.** This
        // barrier is what makes the copies visible to every other stream before the
        // next line hands any emptied region back. Carrying on without it releases
        // regions while kernels may still be reading them, and the pool's next tenant
        // then zeroes ground a live reader is in — which reports as an illegal address
        // in some unrelated kernel, or as nothing at all.
        if let Err(e) = self.device().synchronize() {
            panic!(
                "compaction published {} relocations and then could not fence them: \
                 {e}. The copies are not known to have retired, and the next step \
                 returns emptied regions to the pool — releasing ground that may still \
                 have readers.",
                map.len(),
            );
        }

        // **Every reference the pass leaves behind, re-checked now that it has
        // published, and every slot's content against what it held on the way in.** An
        // orphan count that rose across the pass means some party names a slot the
        // refcount tables call free — which a record now cannot, because it holds one of
        // those refcounts, so a rise here is a *gid* holder nobody swept. A content hash
        // that moved means it has already been cashed in: something is reading ground
        // another chunk now owns.
        #[cfg(feature = "tensor-assert")]
        super::kv_integrity::report_boundary(
            backings,
            "after a compaction",
            report.moves,
            report.unwitnessed,
            Some(&entering),
        );

        // **No emptied source is released after a part-way sweep.** Everything above has
        // been published and fenced, so the pool is consistent — but the holders the sweep
        // did not reach still name sources, and handing regions back is the one step that
        // depends on having reached all of them. Refusing here costs the reclaim and
        // nothing else. The arenas this pass provisioned are still handed back when
        // `fresh` drops on the way out: an empty one is named by nothing, because every
        // destination and every mint in it keeps it occupied.
        if let Err(e) = swept_ok {
            return Err(CompactionRefused::SweepFailed(e.to_string()));
        }

        // Whatever the replacements above emptied is tombstoneable now. A source a
        // surviving holder or its record still names is NOT empty, so this reclaims only
        // the chunks nothing is left pointing at.
        //
        // The arenas this pass provisioned go first, and are counted: until the holder
        // sweep was over a released one was an `arena_idx` open to re-tenancy under
        // holders it had not reached, so they were held until here.
        report.fresh_arenas = fresh.created();
        let fresh_released = fresh.finish();
        report.arenas_released = fresh_released + self.release_empty_arenas().unwrap_or(0);
        report.frontier_after = self.frontier_regions().unwrap_or(frontier_before);
        report.top_arena = SizeClass::all()
            .map(|class| ArenaKey::new(class, ArenaLocation::Gpu))
            .chain(record_key)
            .filter_map(|key| {
                self.pool_top_rank(key)
                    .map(|(rank, live)| TopArena { key, rank, live })
            })
            .max_by_key(|t| t.rank);
        report.timings.publish = phase.elapsed();
        Ok(report)
    }

    /// Give the pools holding the frontier fresh arenas as low in the span as the free
    /// list can place them, **the pool highest on the frontier first**, and add each to
    /// its pool's census so the walk plans into it.
    ///
    /// **One per KV arena holding the frontier, and only those.** The regions worth
    /// emptying are the KV arenas from the frontier down to the first region another
    /// tenant holds, paired with the holes below them (`region_pool::frontier_relocations`)
    /// — every pool among them, because packing is per pool and a pool gets no lower
    /// destination than its own lowest arena otherwise (every pool gapless, 131 arenas
    /// live and a frontier of 330 is a state a per-pool pack alone settles into). An
    /// arena standing under another tenant's region gets none: emptied, it would leave
    /// a hole under a frontier that tenant holds, which the next pass refills from the
    /// next arena down. Sized by "every pool, whatever lies above it", passes moved every
    /// chunk above any hole and the frontier stayed put — Qwen3-30B-A3B, 16 million chunk
    /// moves in one probe. `MAX_FRESH` bounds the claims one pass makes.
    ///
    /// **Why this order places them right.** `census_by_pool` is already sorted by each
    /// pool's highest arena, highest first, and the region free list is lowest-index
    /// first — so claiming in this order hands the lowest hole to the pool standing
    /// highest, the next hole to the next, and so on.
    ///
    /// A claim that lands at or above the lowest region it was meant to empty — another
    /// claim took the hole — ends the provisioning: that arena receives nothing, and the
    /// pass releases it as soon as the claims are in ([`FreshArenas::release_unused`]),
    /// or when [`FreshArenas`] drops on a pass that stops earlier.
    fn provision_low_arenas(
        &self,
        census_by_pool: &mut [(ArenaKey, Vec<ArenaSlots>)],
        fresh: &mut FreshArenas<'_>,
    ) {
        /// Fresh arenas one pass may claim across every pool.
        const MAX_FRESH: usize = 64;
        // **Only for the arenas holding the frontier.** The KV arenas from the frontier
        // down to the first region another tenant holds, one per hole below them: each
        // emptied lowers the frontier. One below another tenant's region, emptied, is a
        // hole under a frontier that tenant still holds, which the next pass refills from
        // the next arena down — every chunk above any hole moved each pass, and the
        // frontier did not (Qwen3-30B-A3B: 16 million chunk moves in one probe).
        let held: HashSet<usize> = census_by_pool
            .iter()
            .flat_map(|(_, census)| census.iter().map(|a| a.rank))
            .collect();
        let relocate = self.regions_to_relocate(&held);
        let budget = relocate.len().min(MAX_FRESH);
        if budget == 0 {
            return;
        }
        // A fresh arena must land below the lowest region it is meant to empty; one the
        // free list placed at or above it receives nothing, and goes back unused.
        let cutoff = relocate[budget - 1];
        let to_empty: HashSet<usize> = relocate[..budget].iter().copied().collect();
        // Per pool, its arenas among those — one fresh arena each.
        let mut wanted: Vec<usize> = census_by_pool
            .iter()
            .map(|(_, census)| census.iter().filter(|a| to_empty.contains(&a.rank)).count())
            .collect();
        let mut made = 0usize;
        loop {
            let mut claimed_any = false;
            for (i, (key, census)) in census_by_pool.iter_mut().enumerate() {
                if census.is_empty() || wanted[i] == 0 {
                    continue;
                }
                if made >= budget {
                    return;
                }
                let arena_idx = match fresh.claim(&self.inner, *key) {
                    Ok(idx) => idx,
                    // No region for this pool is no region for any pool below it either.
                    Err(e) => {
                        tracing::debug!(
                            target: "candle_nn::kv_cache::compact",
                            ?key,
                            "no fresh low arena, packing the remaining pools without one: {e}",
                        );
                        return;
                    }
                };
                made += 1;
                claimed_any = true;
                let Some(slots) = self.fresh_arena_slots(*key, arena_idx) else {
                    wanted[i] = 0;
                    continue;
                };
                if slots.rank >= cutoff {
                    return;
                }
                wanted[i] -= 1;
                census.push(slots);
            }
            if !claimed_any {
                return;
            }
        }
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
            // A band is followed through a holder's gids, a record through its `meta` —
            // the same copy, published to the field that names it.
            match key.kind {
                ArenaKind::Band => map.insert(old_raw, new_gid, dst),
                ArenaKind::Record { .. } => map.insert_record(old_raw, new_gid, dst),
            }
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

    /// A pool's plan cap covers at least one whole arena, so a pass can empty the
    /// arena holding the frontier rather than leave it part-drained for the next
    /// allocations to refill: the smallest rung's arena (16 MiB of 320 B slots)
    /// raises the cap, the largest rung's (16 MiB of 16 KiB slots) does not.
    #[test]
    fn a_pools_plan_cap_covers_one_whole_arena() {
        let smallest = ArenaKey::new(SizeClass::from_index(0).unwrap(), ArenaLocation::Gpu);
        let largest = ArenaKey::new(
            SizeClass::from_index(SizeClass::COUNT - 1).unwrap(),
            ArenaLocation::Gpu,
        );
        assert_eq!(smallest.chunks(), 52_428);
        assert_eq!(planned_moves_cap(smallest), 52_428);
        assert_eq!(largest.chunks(), 1_024);
        assert_eq!(planned_moves_cap(largest), 32_768);
    }

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
