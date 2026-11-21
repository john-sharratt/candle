//! Lets one KV arena be created at a moment when no forward holds the partition.
//!
//! A wave's tier is placed against the arena frontier as it stands when the
//! forward begins, so an arena created mid-wave would move that frontier under
//! it. `enter_arena_window` therefore refuses such a creation instead of
//! waiting, because it is a leaf of the allocator and any wait can close a
//! cycle with a lock the caller holds. The sealing pass that asked retries on
//! its next pass. An open `ArenaWindow` keeps the stream lease for as long as it
//! stands and hands back a standing tier, so a waiting forward proceeds once
//! the window closes.

use super::*;

/// Permission to create KV arenas, held for as long as the creation takes.
///
/// Dropping it lets a forward that is waiting to place its tier proceed.
///
/// **It holds the stream lease it was entered with, for as long as it is
/// open.** A forward that wants the partition waits for every window to close,
/// and creating an arena under the window issues on the stream. With the lease
/// taken only to enter and let go at once, a segment could start recording
/// while the window stood, then wait on it for its tier — while the window's
/// thread, wanting the stream to claim its slab, waited on the recording. Both
/// persistence threads hit that on `.5` twice in one evening, each a 30 s stall
/// ending in a lease panic (2026-10-18). A lease held keeps a new recording from
/// starting; the recording thread runs that wave's segments as issued instead.
pub struct ArenaWindow {
    pub(super) ordinal: usize,
    pub(super) _lease: StreamLease,
}

impl Drop for ArenaWindow {
    fn drop(&mut self) {
        {
            let mut map = lock_domains();
            if let Some(domain) = map.get_mut(&self.ordinal) {
                domain.arena_windows = domain.arena_windows.saturating_sub(1);
            }
        }
        wave_gate().notify_all();
    }
}

/// Take the partition for as long as it takes to create one KV arena, or refuse.
///
/// **A wave is pre-allocated end to end.** Everything it writes is claimed by
/// `admit_wave_kv` and everything it attends over is resident before it opens,
/// and the transient tier is then placed against the arena frontier *as it
/// stands at that moment*. A region claimed after that point is claimed against
/// a partition the wave has already measured: the tier's ceiling either hides
/// the ground (the claim fails with the span half empty) or the arena lands
/// where the tier is about to sit.
///
/// The sealing thread is allowed to allocate KV during a wave — that is the
/// whole point of it running concurrently — but only **into arenas that already
/// exist**. Needing a new one is the case this refuses.
///
/// # It refuses rather than waits, and that is the whole design
///
/// Waiting here was tried and deadlocked the daemon three times, each from a
/// different direction, and the third one showed why the shape is wrong rather
/// than the placement:
///
/// ```text
/// substrate-persistence   migrate_sealed_layers_to_cpu_batch   <- holds state.write()
///                           alloc_chunk_for_key
///                             enter_arena_window   -> Condvar::wait   (for the forward to end)
/// conversation-scheduler  forward_layer_batched_mixed
///                           prefill_capture::maybe_capture
///                             Cache::chunked_writer_start_idx -> state.read()  (for persistence)
/// ```
///
/// This function is a **leaf of the allocator**, reached from arbitrary depth
/// under whatever locks the caller happens to hold — `chunk_ops` holds the
/// sequence-state write lock across the allocation loop, and the forward needs a
/// read on it mid-layer. Any blocking edge here can be closed into a cycle by
/// some caller, and moving the wait above one lock only relocates the problem to
/// the next one. There is no placement that fixes a wait; there is only not
/// waiting.
///
/// **The invariant is unharmed by refusing.** The caller is a sealing pass, and
/// it is already built to fail non-destructively and retry — "the group stays
/// hot-float + consistent and retries next pass". So the arena is still created
/// only between forwards, which is the property that matters; what changes is
/// that the thread goes back to its own work instead of sleeping inside the
/// allocator holding somebody's lock. That is also the codebase's own principle
/// (refuse rather than corrupt) rather than an import into it.
///
/// The one cost is that a sealing pass can be starved by back-to-back forwards.
/// `fresh_claims_during_wave` / `refusals_during_wave` stay zero either way, so
/// the signal for that is the compression backlog, not these.
///
/// # It hands back a standing tier, and must
///
/// A forward's tier is *not* released when the forward ends — it stands until
/// the next forward's phase 0 returns it. So a caller arriving between forwards
/// finds ground that is nobody's and marked as somebody's; it returns it here,
/// which is also what the first deadlock was really about.
///
/// # What this replaced
///
/// Nothing blocked, and the pool budgeted for the breach instead: a per-forward
/// `claim_reserve` seeded a gap above the arena frontier, `fresh_per_forward_peak`
/// grew that gap from observation, and `refusals_during_wave` counted the claims
/// that fell in the gap's shadow anyway. The measurement those carried —
/// "hundreds of fresh regions per forward, because the compressor creates
/// size-class arenas as it chooses formats" — was read as proof the tier could
/// not sit at the arena frontier. It was proof of this.
///
/// # `tenant`
///
/// Who wants the ground, named in both refusals. The span has four tenants and
/// the refusals are about *placement*, so the first question either one raises
/// is which tenant asked — and the message could not answer it. A recurrent
/// store built inside the forward reported itself as "creating a KV arena",
/// which is the one tenant it was not: the hunt for it went through two wrong
/// subsystems before a stack capture named `RecurrentStateStore::new`.
///
/// `&'static str`, so no call site can pay for a `format!` on the success path
/// that every wave takes.
pub fn enter_arena_window(lease: StreamLease, tenant: &'static str) -> Result<ArenaWindow> {
    let stream: &Arc<CudaStream> = &lease;
    let (ordinal, had_tier) = {
        let mut map = lock_domains();
        let (ordinal, domain) = domain_entry(&mut map, stream);
        // **The forward's own thread is a placement bug, not a race**, so it is
        // named separately and *without* the retry marker — a caller that loops
        // on that marker would spin forever on a defect no retry can clear.
        // Every allocation a wave needs is claimed by `admit_wave_kv` before the
        // forward is opened, which is why opening it before admit stopped the
        // engine outright.
        if domain.forward_open && domain.forward_thread == Some(current().id()) {
            candle::bail!(
                "{tenant} wants reservation ground from inside the forward that owns the \
                 partition: this thread opened the wave, so nothing it does can end it. Every \
                 tenant's ground is claimed before the forward opens — K/V by `admit_wave_kv`, \
                 the rest beside it — and ground wanted after that point is ground the \
                 transient tier has already been placed against. Move {tenant}'s claim above \
                 `begin_forward`, and above `plan_wave_transient` with it."
            )
        }
        if domain.forward_open || domain.live_generations > 0 {
            candle::bail!(
                "{KV_ARENA_MID_WAVE} ({tenant}): a forward owns the partition, and the transient tier \
                 was placed against the arena frontier as it stood when that forward began. \
                 Creating an arena now moves the frontier under it. Retry once the wave has \
                 ended — the wave's own storage is claimed by `admit_wave_kv` before it \
                 opens, so nothing it needs depends on this."
            )
        }
        // The partition is ours: nothing is running on it and nothing is holding
        // a span. A tier still standing belongs to a forward that has finished
        // with it, and the ground under it is the ground this arena is about to
        // be carved from.
        //
        // **Its plan goes with it.** The plan is what `begin_wave` re-places
        // when it finds no tier, so a plan left behind is a finished forward's
        // width waiting to be bought again by whoever opens the next guard. A
        // draft walk that priced itself into a prefill's standing tier and then
        // claimed its KV — an arena, so this — found no tier and re-placed the
        // prefill's 8 GB plan against KV that had grown since: refused by one
        // region, every walk, and the decodes on it failed. With the plan gone,
        // the next caller prices itself — unless the caller still running is
        // one that already has, a walk mid-walk ([`WaveDomain::covering`]).
        //
        // **A plan not yet laid out is not a finished forward's.** It is the
        // forward about to open, between its placement and its first phase; its
        // guard re-places it against the frontier this window leaves, and the
        // tier it buys is still that forward's ([`WaveDomain::plan_pending`]).
        let had_tier = domain.placed_at.take().is_some();
        domain.placed_bytes = None;
        if had_tier && !domain.plan_pending {
            domain.reserved_by_forward = false;
            domain.planned = domain.covering;
            domain.scratch_priced = domain.covering_scratch;
        }
        // Counted before the lock drops, so a forward cannot begin placing a
        // tier in the gap between here and the release below.
        domain.arena_windows += 1;
        (ordinal, had_tier)
    };
    // **Outside the map lock, and it has to be.** The lock order everywhere else
    // is map → pool, and a forward inside `place_transient` holds the pool lock
    // while it blocks on the expert pipeline thread — which reaches back for the
    // map lock through `wave_is_live`. Releasing from under the map lock closes
    // that into a cycle.
    if had_tier {
        release_transient(stream);
    }
    Ok(ArenaWindow {
        ordinal,
        _lease: lease,
    })
}

/// Marker every mid-wave arena refusal carries, so a caller that has a retry can
/// tell this apart from a genuine allocation failure.
pub const KV_ARENA_MID_WAVE: &str = "kv arena creation deferred: wave in flight";

/// Declare a forward in progress; the returned value ends it when dropped.
///
/// Held for the whole of `forward_wave_contexts`, so every `?` inside it clears
/// the flag on the way out. See [`WaveDomain::forward_open`] for why
/// `live_generations` cannot do this job.
pub struct ForwardOpen {
    pub(super) ordinal: usize,
}

impl Drop for ForwardOpen {
    fn drop(&mut self) {
        {
            let mut map = lock_domains();
            if let Some(domain) = map.get_mut(&self.ordinal) {
                domain.forward_open = false;
                domain.forward_thread = None;
            }
        }
        wave_gate().notify_all();
    }
}

/// Open the forward's claim on the partition.
///
/// **Call this after admit, not before.** Admit is the forward creating its own
/// arenas, and it goes through `enter_arena_window` like any other creator — so
/// opening the claim first has the forward thread wait on a flag only it can
/// clear.
pub fn begin_forward(stream: &Arc<CudaStream>) -> ForwardOpen {
    let mut map = lock_domains();
    let (ordinal, domain) = domain_entry(&mut map, stream);
    domain.forward_open = true;
    domain.forward_thread = Some(current().id());
    ForwardOpen { ordinal }
}

/// Wait for every open arena window to close, holding the map lock throughout.
///
/// The caller is a forward about to measure the arena frontier; the guard it
/// gets back is the same one it must keep while it decides, so no window can
/// open between the wait and the decision.
pub(super) fn await_arena_windows<'a>(
    mut map: MutexGuard<'a, DomainMap>,
    stream: &Arc<CudaStream>,
) -> MutexGuard<'a, DomainMap> {
    loop {
        let (_, domain) = domain_entry(&mut map, stream);
        if domain.arena_windows == 0 {
            return map;
        }
        map = wave_gate().wait(map).unwrap_or_else(|e| e.into_inner());
    }
}
