//! Advisory counter: is a hot→warm migrate copying residences right now?
//!
//! Used to defer work that would duplicate a migrate's effort — the scheduler
//! postpones a section quantize while a migrate is converting the same
//! residences, so the two don't both do it. That is a *work* decision, not a
//! safety one, and nothing here provides mutual exclusion.
//!
//! # What used to live here, and why it doesn't
//!
//! This file was an arena-topology `RwLock`: shared for operations that
//! captured raw arena base pointers (the persistence thread's migrate, the
//! scheduler's elevate), exclusive for operations that invalidated them (arena
//! free, defrag relocate, arena-vector truncate, `cuMemPoolTrimTo`). A migrate
//! built a per-head base-pointer table, uploaded it, and launched a kernel that
//! dereferenced those pointers — all with no storage lock held — while the
//! scheduler thread was free to unmap the memory underneath it. The lock made
//! the two exclusive, at the cost of a process-global read acquisition on every
//! migrate and every elevate.
//!
//! Two independent changes retired it:
//!
//! - **Nothing invalidates a base pointer any more.** A region of the
//!   reservation is mapped once and stays mapped at the same address for the
//!   process lifetime; "freeing" an arena moves its region between two lists.
//!   Defrag relocation is gone, the arena vector is not truncated, and the pool
//!   trim went with the pool's KV. The ordering that *is* still required — not
//!   re-tenanting a region while an earlier kernel may still be reading it —
//!   belongs to whoever re-tenants, and lives in `region_pool::claim_region`.
//! - **The table stopped being dense over storage.** It is sized from the job
//!   list now, so every pointer in it comes from a gid the caller has pinned.
//!   Even under the old allocator that would have made the neighbour-arena
//!   hazard unreachable; the pin already protected every arena the kernel could
//!   address.
//!
//! `docs/archived/arena_unification.md` §5 (audit A4).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Migrates currently copying residences.
static MIGRATE_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

/// Held shared by a migrate for as long as it is acting on captured chunk
/// addresses, and exclusively by a KV compaction while it relocates them.
///
/// # Why an exclusion is back, when the header says it went away
///
/// The lock this file used to hold guarded **arena base pointers**, and that
/// hazard is genuinely gone: a region is mapped once, at one address, for the
/// process lifetime. What came back is a different hazard at a finer grain. A
/// migrate's table is sized from its job list, so every pointer in it comes from a
/// gid the migrate has pinned — which keeps the *arena* alive and says nothing
/// about where in that arena the chunk sits. A compaction moves chunks between
/// slots of a live arena, so the pin no longer protects the address: the migrate
/// reads the slot the chunk used to be in and writes whatever now occupies it into
/// the warm tier. It does not fault, and the wrong bytes surface later as a
/// sequence answering from another sequence's KV.
///
/// Measured: with compaction gated on a 2 s window, 8 of 8 sessions rewrote the
/// story correctly across many runs; with the same code gated per wave — 114 passes
/// instead of 16 — it fell to 4 of 8.
///
/// # Neither side ever blocks
///
/// Both use the `try_` form and defer. That is not an optimisation, it is what
/// makes the pair free of lock ordering: a migrate holds substrate and block-table
/// locks while it works, and so does a compaction's sweep, so *either* side waiting
/// on the other would need those orders to agree. Neither waits, so there is
/// nothing to agree about. Compaction retries on its next wave, a migrate on the
/// persistence thread's next pass; both are periodic and neither is on a critical
/// path.
static CHUNK_LOCATIONS: RwLock<()> = RwLock::new(());

/// RAII marker: a migrate is copying residences until this drops. Drop-based so
/// an early return or an error still clears the count.
#[must_use = "the flight marker clears on drop; bind it for the migrate's scope"]
pub struct MigrateFlight {
    /// Keeps a compaction from relocating the chunks whose addresses this migrate
    /// has captured. See [`CHUNK_LOCATIONS`].
    _locations: RwLockReadGuard<'static, ()>,
}

impl Drop for MigrateFlight {
    fn drop(&mut self) {
        MIGRATE_IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A compaction pass has been refused and is waiting for a gap.
///
/// **The fairness half of the exclusion, and it is needed in one direction only.**
/// Neither side blocks, so whoever asks during the other's window simply loses — and
/// the two are not symmetric in how often they ask. A hot→warm batch is one long
/// hold per pass; a compaction asks between forwards and is bounded to tens of
/// milliseconds. Left to chance, the long holder wins nearly every time: measured, 78
/// refusals against 107 attempts, precisely during the mass eviction that produced
/// both the fragmentation and the migrate work.
///
/// So a refused pass says so, and the next migrate that would have taken the guard
/// steps aside for one round instead. A compaction clears it when it gets in, and its
/// own gate only lets it ask every few hundred milliseconds, so the migrate keeps the
/// large majority of the time.
static COMPACTION_WAITING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Mark a migrate as in flight until the returned marker drops, or `None` while a
/// compaction holds — or is waiting for — chunk locations.
///
/// **Never blocks, and never waits its turn.** A refusal means the caller should
/// come back on its next pass — see [`CHUNK_LOCATIONS`] for why waiting is the one
/// thing neither side may do. Concurrent migrates simply both count.
pub fn try_migrate_flight() -> Option<MigrateFlight> {
    if COMPACTION_WAITING.load(Ordering::SeqCst) {
        return None;
    }
    let locations = CHUNK_LOCATIONS.try_read().ok()?;
    MIGRATE_IN_FLIGHT.fetch_add(1, Ordering::SeqCst);
    Some(MigrateFlight {
        _locations: locations,
    })
}

/// Exclusive hold on chunk locations for a compaction pass, or `None` while a
/// migrate is acting on addresses it has already captured.
///
/// Never blocks, for the reason on [`CHUNK_LOCATIONS`].
#[must_use = "the freeze lifts on drop; bind it for the compaction's scope"]
pub struct LocationFreeze {
    _locations: RwLockWriteGuard<'static, ()>,
}

pub fn try_freeze_chunk_locations() -> Option<LocationFreeze> {
    match CHUNK_LOCATIONS.try_write() {
        Ok(locations) => {
            // In, so stop holding migrates off — the freeze itself does that now.
            COMPACTION_WAITING.store(false, Ordering::SeqCst);
            Some(LocationFreeze {
                _locations: locations,
            })
        }
        Err(_) => {
            // Refused. Say so, so the next migrate that would have renewed the hold
            // steps aside for a round — see [`COMPACTION_WAITING`].
            COMPACTION_WAITING.store(true, Ordering::SeqCst);
            None
        }
    }
}

/// Stop holding migrates off after a compaction decided not to ask.
///
/// The flag is set by a refusal and cleared by the pass that follows it, so a caller
/// whose gate has since closed — the pools packed themselves, the pass is not worth
/// making — must clear it rather than leave migrates deferring for a pass that is
/// never coming.
pub fn clear_compaction_waiting() {
    COMPACTION_WAITING.store(false, Ordering::SeqCst);
}

/// Whether any migrate is copying residences right now. For deferring duplicated
/// *work* only — the safety question is answered by the guards above, not by this
/// counter, because a counter read cannot keep the answer true for the caller's
/// next instruction.
pub fn migrate_in_flight() -> bool {
    MIGRATE_IN_FLIGHT.load(Ordering::SeqCst) > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test only: the counter and the lock are process-global, so splitting
    // across tests that run in parallel would race on them.
    #[test]
    fn flights_compose_and_exclude_a_compaction() {
        assert!(!migrate_in_flight(), "clean start");
        assert!(
            try_freeze_chunk_locations().is_some(),
            "nothing is holding locations, so a pass may freeze them"
        );

        let a = try_migrate_flight().expect("no compaction is holding the freeze");
        let b = try_migrate_flight().expect("migrates compose");
        assert!(migrate_in_flight(), "in flight while held");
        assert!(
            try_freeze_chunk_locations().is_none(),
            "a compaction must refuse while a migrate holds captured addresses — \
             relocating them is what put the wrong KV in the warm tier"
        );

        drop(a);
        assert!(migrate_in_flight(), "still held by the second marker");
        assert!(
            try_freeze_chunk_locations().is_none(),
            "one remaining migrate is still one too many"
        );

        // A refusal leaves the waiting flag set, which is what stops the long holder
        // from renewing its hold indefinitely — so clear it before asserting that a
        // migrate can proceed again.
        clear_compaction_waiting();

        drop(b);
        assert!(!migrate_in_flight(), "cleared when the last marker drops");
        let freeze = try_freeze_chunk_locations().expect("the last migrate let go");
        assert!(
            try_migrate_flight().is_none(),
            "and the exclusion holds the other way: a migrate defers to the pass \
             rather than capturing addresses it is mid-way through rewriting"
        );
        drop(freeze);
        let resumed = try_migrate_flight().expect("the pass ended, so a migrate proceeds");

        // ── Fairness: a refused pass holds the next migrate off ──────────────
        //
        // Without it the long holder wins nearly every contest — a hot→warm batch is
        // one long hold per pass and a compaction asks between forwards — and chance
        // alone left 78 of 107 passes refused during the very burst that needed them.
        assert!(
            try_freeze_chunk_locations().is_none(),
            "refused while that migrate holds"
        );
        drop(resumed);
        assert!(
            try_migrate_flight().is_none(),
            "the refusal is remembered, so the next migrate steps aside instead of \
             renewing the hold"
        );
        let after_refusal = try_freeze_chunk_locations().expect("the gap the step-aside opened");
        drop(after_refusal);
        let held = try_migrate_flight().expect("the pass getting in released the hold");

        // ── And a pass that stands down must not leave migrates waiting ──────
        assert!(try_freeze_chunk_locations().is_none(), "refused again");
        drop(held);
        clear_compaction_waiting();
        assert!(
            try_migrate_flight().is_some(),
            "the pass stood down, so nothing is holding the migrate off"
        );
    }
}
