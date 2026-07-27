//! Migration-in-flight guard: mutual exclusion between the persistence thread's
//! hot→warm quantize and the scheduler thread's arena free / relocate / trim.
//!
//! The hot→warm migrate builds a **dense per-head base-pointer table** over every
//! arena in a backing ([`super::ChunkedKvBacking::per_head_table_host`]), uploads
//! it, and launches `run_select_kv_format_palette4_paged` — which dereferences
//! those raw device base pointers — then reads the result back. That whole window
//! runs with **no storage lock held** (the slow CUDA work is deliberately
//! unlocked). Meanwhile the scheduler thread, under VRAM pressure, frees empty
//! arenas, defragments (relocates + reindexes), truncates the arena vector, and
//! calls `cuMemPoolTrimTo` — the last of which **synchronously unmaps** freed pool
//! memory (it is not stream-ordered). A neighbour arena's base pointer captured in
//! the table can therefore be unmapped out from under the still-running select
//! kernel → `CUDA_ERROR_ILLEGAL_ADDRESS` on the persistence thread.
//!
//! The `Arc<ChunkGid>` snapshot pin protects the migrate's *source* arenas from
//! release, but not the neighbour arenas the dense table also addresses, and not
//! against defrag *reindexing* arenas the frozen gids point at. So the fix is a
//! coarse guard: while a migrate is in flight, the scheduler **defers** every
//! arena free / relocate / trim to a later wave (migrates run in ~hundreds of ms,
//! so the deferral is short and relief resumes the next wave).
//!
//! Process-global rather than per-backing because `cuMemPoolTrimTo` unmaps memory
//! at the **device pool** granularity — shared across all layer backings — so a
//! trim during any backing's migrate is unsafe, and the CUDA memory pool is a
//! single per-process resource (mirroring the existing global CUDA launch state).

use std::sync::atomic::{AtomicUsize, Ordering};

/// Count of hot→warm migrates currently reading arena base pointers. The persist
/// thread's batched pass holds one; the count is a counter (not a bool) only so
/// nested/overlapping enters compose correctly.
static MIGRATE_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

/// RAII marker: a hot→warm migrate is reading arena base pointers for as long as
/// this value is held. Enter it around the whole build→launch→readback→sync
/// window (before `per_head_table_host`, dropped after the post-migrate device
/// sync). Drop-based so an early return / error still releases the guard.
#[must_use = "the migrate guard releases on drop; bind it for the migrate's scope"]
pub struct MigrateGuard(());

impl Drop for MigrateGuard {
    fn drop(&mut self) {
        MIGRATE_IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Mark a hot→warm migrate as in flight until the returned guard drops.
pub fn enter_migrate() -> MigrateGuard {
    MIGRATE_IN_FLIGHT.fetch_add(1, Ordering::SeqCst);
    MigrateGuard(())
}

/// Whether any hot→warm migrate is currently reading arena base pointers. The
/// scheduler's arena free / defrag / truncate / trim paths check this and defer
/// when it is `true`, so they never unmap memory the select kernel is reading.
pub fn migrate_in_flight() -> bool {
    MIGRATE_IN_FLIGHT.load(Ordering::SeqCst) > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test only: the guard is process-global, so splitting across tests that
    // run in parallel would race on `MIGRATE_IN_FLIGHT`.
    #[test]
    fn guard_tracks_in_flight_nests_and_releases_on_drop() {
        assert!(!migrate_in_flight(), "clean start");
        let a = enter_migrate();
        assert!(migrate_in_flight(), "in flight while held");
        let b = enter_migrate();
        drop(a);
        assert!(migrate_in_flight(), "still held by the nested guard");
        drop(b);
        assert!(!migrate_in_flight(), "released when the last guard drops");
    }
}
