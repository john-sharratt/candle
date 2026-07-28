//! Arena-topology guard: mutual exclusion between operations that **capture raw
//! arena base pointers** (the persistence thread's hot→warm migrate, the
//! scheduler's warm→hot elevate) and operations that **invalidate them** (arena
//! free / relocate / truncate / pool trim).
//!
//! The hot→warm migrate builds a **dense per-head base-pointer table** over every
//! arena in a backing ([`super::ChunkedKvBacking::per_head_table_host`]) —
//! including fully-empty arenas — uploads it, and launches
//! `run_select_kv_format_palette4_paged`, which dereferences those raw device
//! base pointers, then reads the result back. That whole window runs with **no
//! storage lock held** (the slow CUDA work is deliberately unlocked). Meanwhile
//! the scheduler thread frees empty arenas (the per-wave proactive sweep),
//! defragments (relocates + reindexes), truncates the arena vector, and calls
//! `cuMemPoolTrimTo` — the last of which **synchronously unmaps** freed pool
//! memory (it is not stream-ordered). An arena base pointer captured in the
//! table can therefore be unmapped out from under the still-running kernel →
//! `CUDA_ERROR_ILLEGAL_ADDRESS` on the capturing thread.
//!
//! The `Arc<ChunkGid>` snapshot pin protects the migrate's *source* chunks from
//! release, but not the neighbour arenas the dense table also addresses. A
//! one-directional advisory check ("is a migrate in flight?") is insufficient:
//! the sweep can pass the check and then spend hundreds of milliseconds freeing
//! arenas while a migrate *starts* and captures its table mid-sweep. So the
//! exclusion is a process-global `RwLock`:
//!
//! - **Readers (shared)** — pointer capturers: [`enter_migrate`] blocks until no
//!   free/relocate is running, then holds for the whole
//!   build→launch→readback→sync window. Concurrent captures compose.
//! - **Writer (exclusive)** — pointer invalidators: [`try_enter_relief`] takes
//!   the write side non-blocking; on contention the caller skips the pass and
//!   retries a later wave (captures run in ~hundreds of ms, so relief resumes
//!   promptly). Relief must **never block** on the write lock: a pointer
//!   capturer that hits VRAM pressure mid-capture may attempt relief on its own
//!   thread, and a blocking write there would self-deadlock — with `try_write`
//!   it degrades to a skipped relief pass and a propagated allocation error.
//!
//! Process-global rather than per-backing because `cuMemPoolTrimTo` unmaps at
//! the **device pool** granularity — shared across all layer backings — so a
//! trim during any backing's capture is unsafe, and the CUDA memory pool is a
//! single per-process resource (mirroring the existing global CUDA launch
//! state).
//!
//! The [`migrate_in_flight`] boolean survives for advisory (non-safety)
//! deferrals — e.g. the scheduler postpones section quantize while a migrate is
//! copying the same residences, avoiding redundant double-conversion work. It
//! must not be used as a safety check; safety comes only from holding a guard.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// One process-wide lock over arena topology: shared = pointer capture in
/// progress, exclusive = free/relocate/trim in progress.
static ARENA_TOPOLOGY: RwLock<()> = RwLock::new(());

/// Count of pointer captures currently in flight, for the advisory
/// [`migrate_in_flight`] query (a `RwLock` cannot report "readers present").
static MIGRATE_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

/// RAII marker: arena base pointers are captured and may be dereferenced by
/// in-flight kernels for as long as this value is held. Enter it around the
/// whole build→launch→readback→sync window (before `per_head_table_host`,
/// dropped after the post-migrate device sync). Drop-based so an early return /
/// error still releases the guard. Re-entrant on one thread only because relief
/// never *blocks* on the write side — keep it that way.
#[must_use = "the migrate guard releases on drop; bind it for the migrate's scope"]
pub struct MigrateGuard(#[allow(dead_code)] RwLockReadGuard<'static, ()>);

impl Drop for MigrateGuard {
    fn drop(&mut self) {
        MIGRATE_IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// RAII marker: arena topology is being mutated (free / relocate / truncate /
/// trim) for as long as this value is held. Hold it across the **entire**
/// mutation, not just a pre-check — the exclusion, not the acquisition, is the
/// safety property.
#[must_use = "the relief guard releases on drop; bind it for the mutation's scope"]
pub struct ReliefGuard(#[allow(dead_code)] RwLockWriteGuard<'static, ()>);

/// Mark a pointer capture as in flight until the returned guard drops. Blocks
/// while a relief mutation holds the write side (bounded: relief passes are
/// hundreds of milliseconds).
pub fn enter_migrate() -> MigrateGuard {
    let g = ARENA_TOPOLOGY.read().unwrap_or_else(|e| e.into_inner());
    MIGRATE_IN_FLIGHT.fetch_add(1, Ordering::SeqCst);
    MigrateGuard(g)
}

/// Try to begin an arena-topology mutation. `None` while any pointer capture is
/// in flight — the caller skips the pass and retries a later wave. Never
/// blocks, so a capturer that attempts relief on its own thread cannot
/// self-deadlock.
pub fn try_enter_relief() -> Option<ReliefGuard> {
    ARENA_TOPOLOGY.try_write().ok().map(ReliefGuard)
}

/// Advisory: whether any pointer capture is currently in flight. For work
/// deferral only (e.g. postponing a section quantize that would race a migrate
/// copying the same residences) — **not** a safety check; safety requires
/// holding a [`ReliefGuard`].
pub fn migrate_in_flight() -> bool {
    MIGRATE_IN_FLIGHT.load(Ordering::SeqCst) > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test only: the guard is process-global, so splitting across tests that
    // run in parallel would race on the lock and counter.
    #[test]
    fn capture_and_relief_exclude_each_other() {
        assert!(!migrate_in_flight(), "clean start");

        // Captures compose and block relief.
        let a = enter_migrate();
        let b = enter_migrate();
        assert!(migrate_in_flight(), "in flight while held");
        assert!(
            try_enter_relief().is_none(),
            "relief must be refused while captures hold the lock"
        );
        drop(a);
        assert!(migrate_in_flight(), "still held by the nested guard");
        assert!(try_enter_relief().is_none(), "one capture still suffices");
        drop(b);
        assert!(!migrate_in_flight(), "released when the last guard drops");

        // Relief holds exclusively; a second relief is refused; captures resume
        // after it drops.
        let r = try_enter_relief().expect("no captures — relief must acquire");
        assert!(
            try_enter_relief().is_none(),
            "relief is exclusive with itself"
        );
        drop(r);
        let c = enter_migrate();
        assert!(migrate_in_flight());
        drop(c);
        assert!(!migrate_in_flight());
    }
}
