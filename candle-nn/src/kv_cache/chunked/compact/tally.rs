//! Counts every compaction outcome since boot, including the refusals that leave
//! no trace.
//!
//! A pass that never gets the window looks the same as a pass nobody asked for,
//! so each refusal is counted by its cause, and each cause calls for a different
//! fix. A refusal is always the safe outcome, since nothing has been copied,
//! rewritten or freed. The epoch counter lets a holder that captured gids tell
//! whether they are still current.

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
    /// Refused because no pool had an arena a pack could empty — the expected refusal.
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

pub(super) static TALLY: [std::sync::atomic::AtomicU64; 12] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 12];

pub(super) fn note(idx: usize, add: u64) {
    TALLY[idx].fetch_add(add, std::sync::atomic::Ordering::Relaxed);
}

/// Monotonic count of passes that relocated something, for a holder to tell
/// whether the gids it captured are still current.
pub(super) static EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
    /// The persistence thread is writing device slots a pass could copy, or every
    /// move the plan proposed was a chunk a migrate group holds. Ordinary contention
    /// like [`Self::WaveInFlight`], and the reverse direction defers too — see
    /// `migrate_flight`.
    MigrateInFlight,
    /// Not a CUDA device, or no reservation yet.
    NoDevice,
    /// No pool has an arena a pack could empty — the planner had nothing to propose,
    /// since a move that empties no arena releases nothing (`compact_plan::pack_moves`).
    /// The only *expected* refusal, and the only one that is not a defect.
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
