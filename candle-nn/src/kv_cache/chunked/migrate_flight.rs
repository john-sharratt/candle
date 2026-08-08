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

/// Migrates currently copying residences.
static MIGRATE_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

/// RAII marker: a migrate is copying residences until this drops. Drop-based so
/// an early return or an error still clears the count.
#[must_use = "the flight marker clears on drop; bind it for the migrate's scope"]
pub struct MigrateFlight;

impl Drop for MigrateFlight {
    fn drop(&mut self) {
        MIGRATE_IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Mark a migrate as in flight until the returned marker drops. Never blocks —
/// concurrent migrates simply both count.
pub fn migrate_flight() -> MigrateFlight {
    MIGRATE_IN_FLIGHT.fetch_add(1, Ordering::SeqCst);
    MigrateFlight
}

/// Whether any migrate is copying residences right now. For deferring
/// duplicated work only — never a safety check, and there is no longer any
/// exclusion here to promote it into one.
pub fn migrate_in_flight() -> bool {
    MIGRATE_IN_FLIGHT.load(Ordering::SeqCst) > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test only: the counter is process-global, so splitting across tests
    // that run in parallel would race on it.
    #[test]
    fn flights_compose_and_clear_on_the_last_drop() {
        assert!(!migrate_in_flight(), "clean start");
        let a = migrate_flight();
        let b = migrate_flight();
        assert!(migrate_in_flight(), "in flight while held");
        drop(a);
        assert!(migrate_in_flight(), "still held by the second marker");
        drop(b);
        assert!(!migrate_in_flight(), "cleared when the last marker drops");
    }
}
