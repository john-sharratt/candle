//! [`Counting`]: the global allocator that counts. Every call is forwarded to
//! [`System`] and counted in [`CENSUS`] on the calling thread's shard.
//!
//! **Nothing in here may allocate, lock, log or panic.** An allocator that
//! allocates recurses into itself; one that locks can deadlock against a thread
//! holding that lock while it allocates; one that panics aborts the process.
//! So the counting is atomic adds on static memory, the arithmetic wraps, and
//! the thread's shard lives in a `const`-initialised thread-local with no
//! destructor — the kind the platform keeps in static TLS, which needs no
//! allocation to create and is never torn down while its thread runs.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

use super::shards::{Census, SHARDS};

/// The process's census: what [`Counting`] counts into.
pub static CENSUS: Census = Census::new();

/// Set by the first call [`Counting`] counts, so a reading can tell an
/// installed census reading zero from no census at all.
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// The next shard a thread is given.
static NEXT_SHARD: AtomicUsize = AtomicUsize::new(0);

/// No shard chosen yet.
const UNSET: usize = usize::MAX;

thread_local! {
    /// This thread's shard, chosen on its first allocation.
    static SHARD: Cell<usize> = const { Cell::new(UNSET) };
}

/// The calling thread's shard: the one it was given, or the next in turn on its
/// first call. A thread past its thread-locals' lifetime counts on shard 0.
#[inline(always)]
fn shard() -> usize {
    SHARD
        .try_with(|s| {
            let held = s.get();
            if held != UNSET {
                return held;
            }
            let given = NEXT_SHARD.fetch_add(1, Relaxed) % SHARDS;
            s.set(given);
            given
        })
        .unwrap_or(0)
}

/// Whether [`Counting`] is this process's global allocator — whether it has
/// counted anything yet.
pub fn installed() -> bool {
    INSTALLED.load(Relaxed)
}

/// The counting allocator. Install it with
///
/// ```ignore
/// #[global_allocator]
/// static HEAP: heap_census::Counting = heap_census::Counting;
/// ```
pub struct Counting;

impl Counting {
    #[inline(always)]
    fn note_alloc(size: usize) {
        CENSUS.on_alloc(shard(), size);
        if !INSTALLED.load(Relaxed) {
            INSTALLED.store(true, Relaxed);
        }
    }
}

// SAFETY: every method forwards to `System` with the caller's own arguments and
// returns what `System` returned; the counting around it touches only static
// atomics and a `const` thread-local, and never the memory itself.
unsafe impl GlobalAlloc for Counting {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            Counting::note_alloc(layout.size());
        }
        p
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(layout);
        if !p.is_null() {
            Counting::note_alloc(layout.size());
        }
        p
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        CENSUS.on_free(shard(), layout.size());
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = System.realloc(ptr, layout, new_size);
        // A failed realloc leaves the old block in place, counted as it was.
        if !p.is_null() {
            let s = shard();
            CENSUS.on_free(s, layout.size());
            CENSUS.on_alloc(s, new_size);
        }
        p
    }
}
