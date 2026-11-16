//! The host rows a recurrent snapshot is held in, kept between snapshots.
//!
//! A snapshot of a hybrid's recurrent state is one F32 byte row per recurrent
//! layer — ~112 MiB across the 36 DeltaNet layers of Qwen4Exp. A restore decodes
//! a record into fresh rows, and the staged snapshot that holds them is dropped
//! once a store is built from it and its sequence's state has moved on — on the
//! scheduler thread, inside a reprojection's swap: freeing them measured ~4 ms
//! of a calibration exemplar's ~5 ms state move (BlackWorld, 2026-11-16). Rows
//! are the same few sizes for every sequence of a model, so the ones a dropped
//! snapshot held are kept here and filled by the next decode instead of going
//! back to the allocator.
//!
//! **Bounded.** The pool keeps at most [`POOL_BYTES`] of rows; past that a
//! returned row is freed as before. It is host memory, never pinned and never
//! on the card.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use candle::engine_events::note;

/// What the pool has done since the process started: rows filled from it and
/// rows allocated past it, rows kept and rows freed past its bound, each with
/// its bytes — what says whether the bound fits the traffic.
static REUSED: AtomicU64 = AtomicU64::new(0);
static FRESH: AtomicU64 = AtomicU64::new(0);
static FRESH_BYTES: AtomicU64 = AtomicU64::new(0);
static KEPT: AtomicU64 = AtomicU64::new(0);
static FREED: AtomicU64 = AtomicU64::new(0);
static FREED_BYTES: AtomicU64 = AtomicU64::new(0);

/// The most row bytes the pool keeps: two of Qwen4Exp's snapshots — what a
/// restore and the snapshot it replaces hold at once.
pub const POOL_BYTES: usize = 256 << 20;

/// The rows kept, and their total capacity in bytes.
struct Pool {
    rows: Vec<Vec<u8>>,
    bytes: usize,
}

static POOL: Mutex<Pool> = Mutex::new(Pool {
    rows: Vec::new(),
    bytes: 0,
});

fn pool() -> std::sync::MutexGuard<'static, Pool> {
    POOL.lock().unwrap_or_else(|e| e.into_inner())
}

/// A row holding a copy of `bytes`, taken from the pool when it keeps one large
/// enough — the smallest that is — and allocated otherwise.
pub fn row_from(bytes: &[u8]) -> Vec<u8> {
    let kept = {
        let mut p = pool();
        let best = p
            .rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.capacity() >= bytes.len())
            .min_by_key(|(_, r)| r.capacity())
            .map(|(i, _)| i);
        best.map(|i| {
            let row = p.rows.swap_remove(i);
            p.bytes -= row.capacity();
            row
        })
    };
    let mut row = match kept {
        Some(row) => {
            REUSED.fetch_add(1, Ordering::Relaxed);
            row
        }
        None => {
            FRESH.fetch_add(1, Ordering::Relaxed);
            FRESH_BYTES.fetch_add(bytes.len() as u64, Ordering::Relaxed);
            Vec::with_capacity(bytes.len())
        }
    };
    row.clear();
    row.extend_from_slice(bytes);
    row
}

/// Return `row` to the pool, or free it when the pool is full or it holds
/// nothing.
pub fn give(row: Vec<u8>) {
    if row.capacity() == 0 {
        return;
    }
    let mut p = pool();
    if p.bytes + row.capacity() > POOL_BYTES {
        drop(p);
        FREED.fetch_add(1, Ordering::Relaxed);
        FREED_BYTES.fetch_add(row.capacity() as u64, Ordering::Relaxed);
        drop(row);
        report();
        return;
    }
    p.bytes += row.capacity();
    p.rows.push(row);
    drop(p);
    KEPT.fetch_add(1, Ordering::Relaxed);
    report();
}

/// The pool's counters as an engine event (`GET /v1/events`).
fn report() {
    note("recurrent host rows", || {
        let mib = |b: u64| b >> 20;
        format!(
            "reused={} fresh={} fresh_mib={} kept={} freed={} freed_mib={} pooled_mib={}",
            REUSED.load(Ordering::Relaxed),
            FRESH.load(Ordering::Relaxed),
            mib(FRESH_BYTES.load(Ordering::Relaxed)),
            KEPT.load(Ordering::Relaxed),
            FREED.load(Ordering::Relaxed),
            mib(FREED_BYTES.load(Ordering::Relaxed)),
            mib(pooled_bytes() as u64),
        )
    });
}

/// Row bytes the pool keeps now.
pub fn pooled_bytes() -> usize {
    pool().bytes
}

#[cfg(test)]
pub(super) fn clear() {
    let mut p = pool();
    p.rows.clear();
    p.bytes = 0;
}

#[cfg(test)]
pub(super) static TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    /// **A returned row is filled again rather than allocated**: the second
    /// row of a size reuses the first's buffer, and the pool's bytes go back to
    /// zero once it is taken.
    #[test]
    fn a_returned_row_is_filled_again() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        let first = row_from(&[1, 2, 3, 4]);
        let addr = first.as_ptr();
        give(first);
        assert_eq!(pooled_bytes(), 4);
        let second = row_from(&[9, 8, 7]);
        assert_eq!(second, [9, 8, 7]);
        assert_eq!(second.as_ptr(), addr, "the kept buffer, not a new one");
        assert_eq!(pooled_bytes(), 0);
    }

    /// **The smallest row large enough is taken**, and a row too small is
    /// left for a smaller one.
    #[test]
    fn the_smallest_row_large_enough_is_taken() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        let small = Vec::<u8>::with_capacity(8);
        let large = Vec::<u8>::with_capacity(64);
        let (small_at, large_at) = (small.as_ptr(), large.as_ptr());
        give(large);
        give(small);
        let row = row_from(&[0u8; 6]);
        assert_eq!(row.as_ptr(), small_at);
        let big = row_from(&[0u8; 40]);
        assert_eq!(big.as_ptr(), large_at);
        let fresh = row_from(&[0u8; 100]);
        assert!(fresh.as_ptr() != small_at && fresh.as_ptr() != large_at);
    }

    /// **The pool keeps no more than its bound**: a row past it is freed, and
    /// an empty row is never kept.
    #[test]
    fn the_pool_keeps_no_more_than_its_bound() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        give(Vec::with_capacity(POOL_BYTES - 16));
        give(Vec::with_capacity(32));
        assert_eq!(pooled_bytes(), POOL_BYTES - 16);
        give(Vec::new());
        assert_eq!(pooled_bytes(), POOL_BYTES - 16);
        clear();
    }
}
