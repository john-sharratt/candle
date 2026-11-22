//! Claiming a free slot for a chunk, and how a starved size class gets ground.
//!
//! A class with no free slot and no region to stamp promotes into a wider class
//! that already has one. That is safe for band chunks, because every read takes
//! its extent from the band's format. A record is never promoted, since its size
//! is model geometry and a record placed in a band arena would be written over
//! live K/V. A deferral from a wave in flight is handed back unchanged, since
//! widening would put the chunk in the wrong class for a passing reason.

use super::*;

/// Stamps [`BackingInner::claim_slot_promoting`] makes before calling a lost claim a
/// failure. A stamped arena is reclaimable at once, so a concurrent sweep can win it;
/// losing that race several times in a row means the sweep and the stamp are fighting
/// over every free region, which is not a race to keep retrying.
pub(super) const STAMP_ATTEMPTS: usize = 4;

/// # Claiming a chunk slot
///
/// The pool's lock-free refcount table is the single source of truth for "this
/// slot is allocated" — `pool.allocate_for` performs the CAS-claim that flips
/// it. Storage only has to ensure the physical arena tensor exists; no separate
/// per-slot bookkeeping is needed.
///
/// 1. Try `pool.allocate_for(key)` — reuses a freed slot from any arena of this class.
/// 2. If no capacity: stamp a region for the class.
/// 3. If no region can be had: take a free slot from a wider class that
///    already has one (scarcity-only promotion, §3.4).
impl BackingInner {
    /// Register a fresh region for `key`, or place the chunk in an
    /// **already-stamped** wider class when no region can be had. Returns the
    /// key the slot actually came from and the arena holding it.
    ///
    /// # Why promotion reuses rather than re-stamps
    ///
    /// Every class's region is the same [`TARGET_ARENA_BYTES`], so if a region
    /// cannot be had for one class it cannot be had for a wider one either —
    /// walking up the ladder *stamping* would fail identically at every rung
    /// while paying `ensure_vram_budget`'s global compaction each time. (It
    /// did, in the first version of this: one refusal became seven, and the
    /// gate lost 30 % of single-stream decode.)
    ///
    /// What promotion is actually for is stated in §3.4: stopping a trickle of
    /// a rare format from stamping a whole 16 MiB region for a class that will
    /// never fill it. That means taking a **free slot in a class that already
    /// has a region**, which costs nothing and is the only outcome that can
    /// succeed where the stamp failed.
    ///
    /// Strictly scarcity-gated, and in this order: a class gets its own region
    /// whenever one is available, so promotion cannot become a background
    /// mixing vector.
    ///
    /// **A stamped arena comes back inside its creation window** (see
    /// [`Self::claim_fresh_region_open`]), so the caller must claim from it by name and
    /// then call `ChunkGidPool::finish_creation(placed, arena_idx)` whatever the claim's
    /// outcome. For a promoted placement the arena already existed and that call is a
    /// no-op, so a caller never needs to know which it got.
    pub(super) fn stamp_region_promoting(&self, key: ArenaKey) -> Result<(ArenaKey, usize)> {
        let stamp_err = match self.claim_fresh_region_open(key) {
            Ok(arena_idx) => return Ok((key, arena_idx)),
            // **A wave-in-flight deferral is not scarcity, and must not be
            // treated as it.** Promotion exists to stop a rare format stamping a
            // whole region for itself when no region can be had; here a region
            // can be had, just not this instant. Widening in response puts the
            // chunk in the wrong class for a reason that will have passed by the
            // next pass — and, worse, the caller's retry loop then reports the
            // failure as `VRAM exhaustion on arena creation` while a fifth of the
            // reservation stands free. Hand the deferral straight back so it
            // reaches the sealing pass as itself.
            #[cfg(feature = "cuda")]
            Err(e) if Self::is_wave_deferral(&e) => return Err(e),
            Err(e) => e,
        };
        // **Reactive empty sweep before anything wasteful.** Seal and
        // requantize churn strands fully-empty arenas that still pin a region
        // each — measured on the Llama-2 MHA gate: 195 empty arenas holding
        // 3 GiB while a class-4096 claim died on "every region occupied". The
        // periodic sweeps (the scheduler's per-wave call, the harness's
        // between-configs call) don't run inside a config's own churn, so the
        // claim that would otherwise fail is exactly the place to pay for one.
        if self.pool.has_reclaimable() && self.release_empty_arenas()? > 0 {
            match self.claim_fresh_region_open(key) {
                Ok(arena_idx) => return Ok((key, arena_idx)),
                #[cfg(feature = "cuda")]
                Err(e) if Self::is_wave_deferral(&e) => return Err(e),
                Err(_) => {}
            }
        }
        // **Only a band claim may be promoted, and a record claim must NOT be.**
        //
        // Promotion widens `key.class` and rebuilds the key with `ArenaKey::new`, which
        // is a `Band` key by construction. For a band that is the whole point — a rare
        // format borrows a wider class's region instead of stamping one for itself. For a
        // `KvHead` record it is silent corruption: the record would be handed a slot in a
        // **band arena** and its bytes written on top of live K/V, which nothing
        // downstream can catch. A record's size is model geometry
        // (`n_kv_head × (head_dim/2 + 26 × n_palette)`), not a rung of the size-class
        // ladder, so there is no wider class it belongs in and the stride check a caller
        // does — "the slot is at least `record_bytes`" — passes happily on a 2 KiB band
        // slot holding somebody's keys.
        //
        // Measured: with a compaction minting a fresh record per relocated chunk (800–1,400
        // claims per pass) against a pool driven to saturation, the Flash-Next engine probe
        // answered wrongly on one to three sessions of eight. A record claim that cannot
        // get a region fails as itself instead, which the seal path and the compaction both
        // already handle.
        if !matches!(key.kind, ArenaKind::Band) {
            return Err(stamp_err);
        }
        // No region. Look for a wider class that already has one with room.
        let mut class = key.class;
        while let Some(next) = class.promote() {
            class = next;
            let wider = ArenaKey::new(class, key.location);
            if let Some(gid) = self.pool.allocate_for(wider) {
                let arena_idx = gid.arena_idx();
                // The gid is dropped here; the caller re-claims from `wider`.
                // Dropping returns the slot, so this is a probe, not a leak.
                drop(gid);
                if self
                    .storage
                    .read(|s| s.has_arena(arena_idx))
                    .unwrap_or(false)
                {
                    record_class_promotion(key.class, class);
                    return Ok((wider, arena_idx));
                }
            }
        }
        Err(stamp_err)
    }

    /// Register and materialise one region for `key`, rolling the registration
    /// back if it cannot be materialised — and return it **still inside its creation
    /// window**.
    ///
    /// Without the rollback the pool would advertise free slots that storage
    /// cannot produce: every later claim into that arena fails the same way,
    /// and `total_arenas` inflates the occupancy diagnostic.
    ///
    /// **The caller closes the window, on every path.** While it is open no sweep can
    /// release the arena, which is what lets the caller claim from it by name without
    /// racing another thread's sweep — and an index released under a claimer can be
    /// re-registered for someone else mid-creation, so the claim would land in a
    /// half-built arena. Once the caller has claimed, or decided not to, it calls
    /// `ChunkGidPool::finish_creation`: a no-op if its claim already closed the window,
    /// and otherwise the step that makes an arena nothing used reclaimable. A window
    /// left open is a region no sweep will ever release.
    pub(in super::super) fn claim_fresh_region_open(&self, key: ArenaKey) -> Result<usize> {
        let arena_idx = self.pool.register_arena(key);
        if let Err(e) = self.ensure_arena_exists(arena_idx, key) {
            self.pool.force_release_arena(arena_idx);
            return Err(e);
        }
        Ok(arena_idx)
    }

    /// One region for `key`, running `hold` on the new index before its creation
    /// window closes, and returned reclaimable.
    ///
    /// For a creator that means to keep the arena empty for a while: whatever `hold`
    /// does to keep it (protecting it, in practice) is in place before any sweep can
    /// see the arena as reclaimable. Doing it after the return leaves a gap in which a
    /// sweep on another thread can take the arena the creator is about to use.
    pub(in super::super) fn claim_fresh_region_then(
        &self,
        key: ArenaKey,
        hold: impl FnOnce(usize),
    ) -> Result<usize> {
        let arena_idx = self.claim_fresh_region_open(key)?;
        hold(arena_idx);
        self.pool.finish_creation(key, arena_idx);
        Ok(arena_idx)
    }

    /// Claim one chunk slot for `key`, **promoting up the size-class ladder**
    /// when the class is starved and no region can be had for it.
    ///
    /// Correctness is untouched by the wider slot: every read takes its extent
    /// from the band's *format* bytes, never from the stride, so a chunk in a
    /// larger class is simply a chunk with more unread pad. Only the waste changes.
    ///
    /// **The claim names the arena the stamp produced**, while that arena's creation
    /// window is still open — so no sweep can release it in between — and the window is
    /// closed after the claim whatever it returned. Left to `allocate_for` the claim
    /// could land in another arena that freed a slot meanwhile, and the stamped one
    /// would then sit empty. A promoted placement is an existing arena other claimers
    /// share, so its free slot can be taken first; that goes round again, and a stamp
    /// that fails ends the loop.
    pub(in super::super) fn claim_slot_promoting(&self, key: ArenaKey) -> Result<ChunkGid> {
        for _ in 0..STAMP_ATTEMPTS {
            if let Some(gid) = self.pool.allocate_for(key) {
                self.ensure_arena_exists(gid.arena_idx(), key)?;
                return Ok(gid);
            }
            let (placed, arena_idx) = self.stamp_region_promoting(key)?;
            let claimed = self.pool.allocate_from_arena(placed, arena_idx);
            self.pool.finish_creation(placed, arena_idx);
            if let Some(gid) = claimed {
                return Ok(gid);
            }
        }
        candle::bail!(
            "claim_slot_promoting: {STAMP_ATTEMPTS} placements for {key:?} each had their \
             free slot taken by another claimer before this one could take it"
        )
    }

    pub(in super::super) fn alloc_chunk_for_key(
        &self,
        key: super::super::arena::ArenaKey,
    ) -> Result<super::super::gid_pool::ChunkGid> {
        self.claim_slot_promoting(key)
    }

    /// Allocate `len` CONSECUTIVE slots in one arena of `key`. Contiguity is a
    /// LOCALITY optimization for the paged select/QREL walk, not a correctness
    /// requirement — each band is addressed through its own gid
    /// (`resolve_band_source`), so scattered bands read correctly, just with
    /// worse spatial locality. Falls back to singleton allocation when no run
    /// is available. Mirrors [`Self::alloc_chunk_for_key`]'s
    /// register-on-exhaustion retry.
    pub(in super::super) fn alloc_chunk_run_for_key(
        &self,
        key: super::super::arena::ArenaKey,
        len: usize,
    ) -> Result<Vec<super::super::gid_pool::ChunkGid>> {
        // A run larger than one arena can NEVER be satisfied — arenas are
        // fixed-capacity slabs and a run must be contiguous within one. Fail
        // with the sizes so this permanent condition is never mistaken for the
        // transient race below (they used to share one message, and a night went
        // into telling them apart).
        let arena_chunks = key.chunks();
        if len > arena_chunks {
            candle::bail!(
                "palette run of {len} chunks exceeds arena capacity {arena_chunks} \
                 for class {} B — cannot be satisfied by any arena",
                key.slot_stride(),
            );
        }
        if let Some(gids) = self.pool.allocate_run_for(key, len) {
            self.ensure_arena_exists(gids[0].arena_idx(), key)?;
            self.replenish_if_nearly_dry(key, len);
            return Ok(gids);
        }
        // No existing arena has tail room: register a fresh one and claim from
        // it BY INDEX. The old shape — register, then re-walk the whole pool —
        // raced: between registration and the re-walk, concurrent claimers (the
        // scheduler's prefills and the persistence thread's elevations allocate
        // the same formats in parallel) could consume the fresh arena's tail,
        // and the single retry then failed spuriously as "fresh arena cannot
        // fit palette run", killing the whole forward. Targeting the registered
        // index removes the which-arena race; losing even that (racers landing
        // in OUR arena via their own global walks) just means another
        // registration, bounded.
        const ATTEMPTS: usize = 4;
        let mut key = key;
        for _ in 0..ATTEMPTS {
            // The whole run widens together when the class cannot get a
            // region, so its bands stay in one arena and keep the locality the
            // run exists for.
            let (placed, arena_idx) = self.stamp_region_promoting(key)?;
            key = placed;
            let _ = arena_chunks;
            let run = self.pool.allocate_run_for_in(key, arena_idx, len);
            // Closed after the claim, whatever it returned — see
            // `stamp_region_promoting`.
            self.pool.finish_creation(key, arena_idx);
            if let Some(gids) = run {
                self.replenish_if_nearly_dry(key, len);
                return Ok(gids);
            }
            // Raced into our fresh arena — the racer may equally have vacated
            // tail room elsewhere; check the whole pool before registering again.
            if let Some(gids) = self.pool.allocate_run_for(key, len) {
                self.ensure_arena_exists(gids[0].arena_idx(), key)?;
                self.replenish_if_nearly_dry(key, len);
                return Ok(gids);
            }
        }
        candle::bail!(
            "palette run of {len} chunks unsatisfied after {ATTEMPTS} fresh arenas \
             (capacity {arena_chunks} each, class {} B) — allocator contention or \
             VRAM exhaustion on arena creation",
            key.slot_stride(),
        )
    }

    /// Claim a sequence of palette runs, `requests[i] = (key, len)`, in order —
    /// the slots and the order of one [`Self::alloc_chunk_run_for_key`] per
    /// request, with that function's per-run bookkeeping done once.
    ///
    /// A run that the pool serves from an existing arena needs two follow-ups:
    /// its arena must exist in storage, and the class must be probed for being
    /// nearly dry. Both are properties of the arena and the class, not of the
    /// run, so a batch of runs from one arena asks each question once. A run the
    /// pool cannot serve falls back to the single-run path, which registers a
    /// fresh arena and retries exactly as it always did.
    pub(in super::super) fn alloc_chunk_runs_for_keys(
        &self,
        requests: &[(super::super::arena::ArenaKey, usize)],
    ) -> Result<Vec<Vec<super::super::gid_pool::ChunkGid>>> {
        let mut runs = Vec::with_capacity(requests.len());
        let mut arenas: Vec<(super::super::arena::ArenaKey, usize)> = Vec::new();
        let mut probes: Vec<(super::super::arena::ArenaKey, usize)> = Vec::new();
        for &(key, len) in requests {
            match self.pool.allocate_run_for(key, len) {
                Some(gids) => {
                    let arena_idx = gids[0].arena_idx();
                    if !arenas.contains(&(key, arena_idx)) {
                        arenas.push((key, arena_idx));
                    }
                    if !probes.iter().any(|&(k, _)| k == key) {
                        probes.push((key, len));
                    }
                    runs.push(gids);
                }
                None => runs.push(self.alloc_chunk_run_for_key(key, len)?),
            }
        }
        for (key, arena_idx) in arenas {
            self.ensure_arena_exists(arena_idx, key)?;
        }
        for (key, len) in probes {
            self.replenish_if_nearly_dry(key, len);
        }
        Ok(runs)
    }

    /// Bulk allocator — mirrors [`Self::alloc_chunk_for_key`]'s
    /// register-on-exhaustion loop but in batch.
    ///
    /// Per pass:
    /// - **One** `pool.allocate_n_for(key, remaining)` returns up to
    ///   `remaining` GIDs; CAS-claim makes the refcount table
    ///   immediately authoritative, no follow-up bookkeeping required.
    /// - **One** `ensure_arena_exists` per unique arena index we
    ///   touched (cheap — the inner check is a `storage.read`).
    ///
    /// If the pool returned fewer GIDs than requested, the format's
    /// pool was exhausted — we register a fresh arena (one
    /// `register_arena + ensure_arena_exists` round) and re-enter the
    /// loop to fill the remainder. Same termination guarantee as the
    /// singular path.
    pub(in super::super) fn alloc_chunks_for_key_bulk(
        &self,
        key: super::super::arena::ArenaKey,
        n: usize,
    ) -> Result<Vec<super::super::gid_pool::ChunkGid>> {
        let (out, stopped) = self.alloc_chunks_for_key_upto(key, n);
        match stopped {
            // What was claimed goes back to the pool as `out` drops.
            Some(e) => Err(e),
            None => Ok(out),
        }
    }

    /// Up to `n` slots of `key`, lowest place first, and why it stopped short
    /// if it did: the pool's free slots first, then fresh regions, widening the
    /// class when it cannot get one (as [`Self::alloc_chunks_for_key_bulk`]).
    /// A refusal ends the claim and keeps what was claimed — what the gap's
    /// lease wants, since a class short of ground defers blocks rather than
    /// fail the gap.
    pub(in super::super) fn alloc_chunks_for_key_upto(
        &self,
        key: ArenaKey,
        n: usize,
    ) -> (Vec<ChunkGid>, Option<Error>) {
        if n == 0 {
            return (Vec::new(), None);
        }
        let mut out: Vec<ChunkGid> = Vec::with_capacity(n);
        let mut key = key;
        // Placements in a row whose slots other claimers took before this batch could.
        // Each placement either feeds the batch or is lost that way; a run of losses is
        // contention, not progress, and is not retried forever.
        let mut fruitless = 0usize;
        let mut stopped = None;
        while out.len() < n {
            let remaining = n - out.len();
            let mut batch = self.pool.allocate_n_for(key, remaining);
            if batch.is_empty() {
                if fruitless == STAMP_ATTEMPTS {
                    stopped = Some(Error::Msg(format!(
                        "alloc_chunks_for_key_bulk: {STAMP_ATTEMPTS} placements for {key:?} \
                         in a row were taken by other claimers before this batch could \
                         claim from them ({} of {n} allocated)",
                        out.len(),
                    )));
                    break;
                }
                // Pool exhausted — stamp a fresh region, widening if this
                // class cannot get one. Later passes then fill from the class
                // that could, so a partially-promoted batch is normal. The batch
                // claims while the stamped arena's creation window is open, and
                // closes it after — see `stamp_region_promoting`.
                let (placed, arena_idx) = match self.stamp_region_promoting(key) {
                    Ok(p) => p,
                    Err(e) => {
                        stopped = Some(e);
                        break;
                    }
                };
                key = placed;
                batch = self.pool.allocate_n_for(key, remaining);
                self.pool.finish_creation(key, arena_idx);
                if batch.is_empty() {
                    fruitless += 1;
                    continue;
                }
            }
            fruitless = 0;
            // Ensure every unique arena index we just got is materialised
            // in storage. Most calls hit the cheap `storage.read`-only
            // path because the arena already exists.
            let mut seen: ahash::HashSet<usize> =
                ahash::HashSet::with_capacity_and_hasher(4, ahash::RandomState::new());
            let mut missing = None;
            for gid in &batch {
                let ai = gid.arena_idx();
                if seen.insert(ai) {
                    if let Err(e) = self.ensure_arena_exists(ai, key) {
                        missing = Some(e);
                        break;
                    }
                }
            }
            if let Some(e) = missing {
                // The batch's slots go back as it drops; what came before stands.
                stopped = Some(e);
                break;
            }
            out.extend(batch);
        }
        // One single-slot probe for the whole batch, at the final key. The bulk
        // allocator takes arbitrary slots rather than contiguous runs, so "could
        // this batch run again" has no cheap probe — but "is the class bone dry"
        // does, and a dry class is the case that costs the sealer a deferred
        // pass.
        if stopped.is_none() {
            self.replenish_if_nearly_dry(key, 1);
        }
        (out, stopped)
    }

    /// Keep the sealer one arena ahead of its demand: if `key`'s class could
    /// not serve `len` more slots right now, ask the next inter-forward gap
    /// for an arena — **before** anything actually runs out.
    ///
    /// This is the sealing buffer. The sealer may fill existing arenas at any
    /// point in a wave but may only *create* one between forwards, so a class
    /// that runs dry mid-wave costs a deferred pass: the selection work is
    /// redone and the hot→warm drain slips a wave. Probing after each
    /// successful allocation converts that into a class that is replenished in
    /// the gap *before* the next pass needs it — steady state never sees the
    /// refusal, and the deferral remains only for the cold start of a class no
    /// one has used yet.
    ///
    /// The probe is an allocate-and-drop, the same pattern
    /// [`Self::stamp_region_promoting`] uses: dropping the gids returns the
    /// slots, so it observes without consuming. CPU classes are skipped — their
    /// creation is never gated, so there is nothing to get ahead of.
    #[cfg(feature = "cuda")]
    pub(super) fn replenish_if_nearly_dry(&self, key: ArenaKey, len: usize) {
        if key.location == ArenaLocation::Cpu {
            return;
        }
        // Read-only: an allocate-and-drop probe here permanently burns run
        // capacity, because a run claim advances the arena's never-used
        // high-water mark and dropped run gids recycle through the singleton
        // stack that run claims never read. Measured consequence of the probe:
        // every successful run allocation consumed double its length, arenas
        // ran dry mid-sweep, and the in-wave refusal this function exists to
        // prevent came back.
        if !self.pool.run_would_fit(key, len.max(1)) {
            self.note_deferred_arena(key);
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub(super) fn replenish_if_nearly_dry(&self, _key: ArenaKey, _len: usize) {}

    /// Ensure that an arena exists at the given index in storage.
    ///
    /// Creates the arena if it does not exist yet — **waiting for the gap
    /// between forwards to do it**. A wave is pre-allocated end to end, and a
    /// new arena is the one KV allocation that cannot be: it moves the arena
    /// frontier the running wave's transient tier was placed against. Filling an
    /// existing arena is unrestricted and is what the sealing thread does for
    /// nearly every chunk; only the creation waits.
    ///
    /// A refusal **records what it wanted** before it propagates, so the next
    /// pass can create it in the gap rather than rediscovering the need at the
    /// same depth and failing the same way — see [`Self::create_deferred_arenas`].
    pub(in super::super) fn ensure_arena_exists(
        &self,
        arena_idx: usize,
        key: ArenaKey,
    ) -> Result<()> {
        // **An arena already standing at the index must be this key's.** One left
        // over from the index's previous tenant has another class's stride, and
        // adopting it puts slots of one size in a pool of another — which the
        // compaction planner then reports as a copy between slots of unequal extent,
        // pass after pass, failing every turn it touches.
        if let Some(standing) = self.storage.read(|s| s.arena_key(arena_idx))? {
            if standing != key {
                candle::bail!(
                    "arena index {arena_idx} was registered for {key:?} but storage still \
                     holds an arena for {standing:?} there — the index was reissued before \
                     its last tenant's storage was released"
                );
            }
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        let _window = match self.arena_window(key) {
            Ok(w) => w,
            Err(e) => {
                self.note_deferred_arena(key);
                return Err(e);
            }
        };
        let arena = self.create_arena(key, arena_idx)?;
        self.storage.try_write(|s| {
            if !s.has_arena(arena_idx) {
                s.push_arena(arena, arena_idx);
            }
            Ok(())
        })?;
        Ok(())
    }
}
