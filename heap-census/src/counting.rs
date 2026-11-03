//! [`Counting`]: the global allocator that counts. Every call is forwarded to
//! [`System`] and counted in [`CENSUS`] on the calling thread's shard.
//!
//! **A block of [`LABELLED_FROM`] bytes or more carries its scope.** It is
//! allocated with a header in front ([`header_of`]), whose last byte holds the
//! scope the block was made in ([`crate::scope`]); the free reads it back and
//! takes the block off that scope, whichever thread frees it. Whether a block
//! has a header is decided by its layout alone, which the free is handed too,
//! so nothing else records it. A smaller block is counted by class only: the
//! header would be a large share of it, and the growth a scope is wanted for
//! is in the blocks above it.
//!
//! **Nothing in here may allocate, lock, log or panic.** An allocator that
//! allocates recurses into itself; one that locks can deadlock against a thread
//! holding that lock while it allocates; one that panics aborts the process.
//! So the counting is atomic adds on static memory, the arithmetic wraps, and
//! the thread's shard and scope live in `const`-initialised thread-locals with
//! no destructor — the kind the platform keeps in static TLS, which needs no
//! allocation to create and is never torn down while its thread runs.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ptr::{copy_nonoverlapping, null_mut};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

use super::scope::current;
use super::shards::{Census, SHARDS};

/// The process's census: what [`Counting`] counts into.
pub static CENSUS: Census = Census::new();

/// The smallest block that carries its scope.
pub const LABELLED_FROM: usize = 4096;

/// The largest alignment a labelled block may ask for: a header is the
/// alignment's size, so a page-aligned block would pay a page for one byte.
const LABELLED_ALIGN: usize = 64;

/// The smallest header: room for the scope byte at any alignment up to 16.
const MIN_HEADER: usize = 16;

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

/// The header in front of a block of `layout`, in bytes, when it carries its
/// scope: the alignment, at least [`MIN_HEADER`], so the block behind it keeps
/// its alignment.
#[inline(always)]
pub const fn header_of(layout: Layout) -> Option<usize> {
    if layout.size() >= LABELLED_FROM && layout.align() <= LABELLED_ALIGN {
        Some(if layout.align() > MIN_HEADER {
            layout.align()
        } else {
            MIN_HEADER
        })
    } else {
        None
    }
}

/// The layout `System` is asked for: `layout` with `header` bytes in front.
#[inline(always)]
fn outer(layout: Layout, header: usize) -> Option<Layout> {
    let size = layout.size().checked_add(header)?;
    Layout::from_size_align(size, layout.align()).ok()
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
    fn mark_installed() {
        if !INSTALLED.load(Relaxed) {
            INSTALLED.store(true, Relaxed);
        }
    }

    /// Allocate a block of `layout` through `get`, with its header when it
    /// carries one, and count it.
    #[inline(always)]
    unsafe fn allocate(layout: Layout, get: impl FnOnce(Layout) -> *mut u8) -> *mut u8 {
        let Some(header) = header_of(layout) else {
            let p = get(layout);
            if !p.is_null() {
                CENSUS.on_alloc(shard(), layout.size());
                Counting::mark_installed();
            }
            return p;
        };
        let Some(whole) = outer(layout, header) else {
            return null_mut();
        };
        let base = get(whole);
        if base.is_null() {
            return base;
        }
        let scope = current();
        // SAFETY: `base` holds `header` bytes before the block; the scope is
        // their last.
        base.add(header - 1).write(scope);
        CENSUS.on_alloc_scoped(shard(), layout.size(), scope);
        Counting::mark_installed();
        base.add(header)
    }
}

// SAFETY: every method hands `System` a layout it can satisfy and returns
// memory `System` gave: either the block itself, or — for a block that carries
// its scope — the block behind a header of the alignment's size, which keeps it
// aligned. The free undoes exactly what the allocation did, decided by the same
// layout. The counting touches only static atomics and `const` thread-locals.
unsafe impl GlobalAlloc for Counting {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Counting::allocate(layout, |l| System.alloc(l))
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Counting::allocate(layout, |l| System.alloc_zeroed(l))
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        match header_of(layout) {
            None => {
                System.dealloc(ptr, layout);
                CENSUS.on_free(shard(), layout.size());
            }
            Some(header) => {
                let base = ptr.sub(header);
                let scope = base.add(header - 1).read();
                // SAFETY: the allocation succeeded with this outer layout, so
                // it is valid.
                System.dealloc(base, outer(layout, header).unwrap_unchecked());
                CENSUS.on_free_scoped(shard(), layout.size(), scope);
            }
        }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        match (header_of(layout), header_of(new_layout)) {
            (None, None) => {
                let p = System.realloc(ptr, layout, new_size);
                // A failed realloc leaves the old block in place, counted as it
                // was.
                if !p.is_null() {
                    let s = shard();
                    CENSUS.on_free(s, layout.size());
                    CENSUS.on_alloc(s, new_size);
                }
                p
            }
            // The same alignment, so the same header: the block keeps the
            // scope it was made in.
            (Some(header), Some(_)) => {
                let base = ptr.sub(header);
                let scope = base.add(header - 1).read();
                let Some(whole) = new_size.checked_add(header) else {
                    return null_mut();
                };
                let p = System.realloc(base, outer(layout, header).unwrap_unchecked(), whole);
                if p.is_null() {
                    return p;
                }
                let s = shard();
                CENSUS.on_free_scoped(s, layout.size(), scope);
                CENSUS.on_alloc_scoped(s, new_size, scope);
                p.add(header)
            }
            // Across the labelled size, one way or the other: the header comes
            // or goes, so the block moves.
            _ => {
                let p = self.alloc(new_layout);
                if !p.is_null() {
                    copy_nonoverlapping(ptr, p, layout.size().min(new_size));
                    self.dealloc(ptr, layout);
                }
                p
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(size: usize, align: usize) -> Layout {
        Layout::from_size_align(size, align).unwrap()
    }

    /// **A header is the alignment, at least 16 bytes, for a block from 4 KiB
    /// of an alignment up to 64** — and none otherwise.
    #[test]
    fn a_header_is_decided_by_the_layout() {
        assert_eq!(header_of(layout(4095, 8)), None);
        assert_eq!(header_of(layout(4096, 1)), Some(16));
        assert_eq!(header_of(layout(4096, 8)), Some(16));
        assert_eq!(header_of(layout(100_000, 16)), Some(16));
        assert_eq!(header_of(layout(100_000, 32)), Some(32));
        assert_eq!(header_of(layout(100_000, 64)), Some(64));
        assert_eq!(header_of(layout(100_000, 128)), None);
        assert_eq!(header_of(layout(1 << 20, 4096)), None);
    }

    /// **A block behind its header keeps its alignment**, and the scope byte
    /// sits just before it — called directly, without installing.
    #[test]
    fn a_labelled_block_is_aligned_and_names_its_scope() {
        for align in [1, 8, 16, 32, 64] {
            let l = layout(8192, align);
            // SAFETY: freed once with the layout it was allocated with.
            unsafe {
                let p = Counting.alloc(l);
                assert!(!p.is_null());
                assert_eq!(p as usize % align, 0, "align {align}");
                assert_eq!(p.sub(1).read(), current());
                p.write_bytes(0xA5, l.size());
                Counting.dealloc(p, l);
            }
        }
    }

    /// **A realloc keeps the bytes across the labelled size, both ways.**
    #[test]
    fn a_realloc_across_the_labelled_size_keeps_the_bytes() {
        // SAFETY: each pointer is used with the layout it last had.
        unsafe {
            let small = layout(1000, 8);
            let p = Counting.alloc(small);
            for i in 0..1000 {
                p.add(i).write((i % 251) as u8);
            }
            let big = Counting.realloc(p, small, 50_000);
            assert!(!big.is_null());
            assert!((0..1000).all(|i| big.add(i).read() == (i % 251) as u8));
            let bigger = Counting.realloc(big, layout(50_000, 8), 200_000);
            assert!((0..1000).all(|i| bigger.add(i).read() == (i % 251) as u8));
            let back = Counting.realloc(bigger, layout(200_000, 8), 500);
            assert!((0..500).all(|i| back.add(i).read() == (i % 251) as u8));
            Counting.dealloc(back, layout(500, 8));
        }
    }
}
