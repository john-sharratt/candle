//! The device-agnostic page-slot allocator for the gallery arena.
//!
//! A [`PagePool`] hands out fixed-size **page slots** identified by a `u32`
//! global id, decoding `gid → (slab, page-in-slab)` by a fixed `SLAB_PAGES`
//! stride. It is a plain free-list + high-water mark behind the arena's lock —
//! the gallery is (re)paged at most once per turn seal, not per token, so the
//! lock-free Treiber machinery of the KV `GidPool` (`candle-nn` `gid_pool.rs`) is
//! unnecessary here; this mirrors its *pattern* (fixed slabs, recycle on free,
//! grow on exhaustion) without its concurrency cost.
//!
//! The pool tracks only ids and capacity — the physical VRAM slabs live in
//! [`super::storage::GalleryStorage`]. The arena grows the pool
//! ([`PagePool::grow`]) right after it registers a new device slab, so the two
//! stay in lock-step.

/// Pages per 16 MiB slab, for a given page size in `u64` words. `TARGET_ARENA_BYTES`
/// mirrors the KV arena's slab size (`chunked/types.rs`).
pub const TARGET_ARENA_BYTES: usize = 16 * 1024 * 1024;

/// Number of page slots in one slab, given `page_u64` words per page.
pub fn slab_pages(page_u64: usize) -> usize {
    (TARGET_ARENA_BYTES / (page_u64 * std::mem::size_of::<u64>())).max(1)
}

/// A free-list page-slot allocator over a growable slab space.
///
/// `capacity` is the number of slots physically backed by registered slabs;
/// `hwm` is the next never-used slot below `capacity`. Freed slots return to
/// `free` and are handed out before the high-water mark advances.
#[derive(Debug)]
pub struct PagePool {
    slab_pages: u32,
    free: Vec<u32>,
    hwm: u32,
    capacity: u32,
    live: usize,
}

impl PagePool {
    pub fn new(slab_pages: usize) -> Self {
        assert!(slab_pages > 0, "slab must hold at least one page");
        Self {
            slab_pages: slab_pages as u32,
            free: Vec::new(),
            hwm: 0,
            capacity: 0,
            live: 0,
        }
    }

    /// Decode a global page id into `(slab_idx, page_in_slab)`.
    #[inline]
    pub fn locate(&self, gid: u32) -> (usize, usize) {
        (
            (gid / self.slab_pages) as usize,
            (gid % self.slab_pages) as usize,
        )
    }

    /// Slots currently handed out (not yet freed).
    #[inline]
    #[cfg(test)]
    pub fn live(&self) -> usize {
        self.live
    }

    /// Slots physically backed by registered slabs.
    #[inline]
    #[cfg(test)]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Number of slabs registered so far.
    #[inline]
    #[cfg(test)]
    pub fn slab_count(&self) -> usize {
        (self.capacity / self.slab_pages) as usize
    }

    /// Grow the backed capacity by one slab. The arena calls this immediately
    /// after it materialises the corresponding device slab.
    pub fn grow_one_slab(&mut self) {
        self.capacity += self.slab_pages;
    }

    /// Try to hand out one slot without growing — recycle a freed slot, else
    /// bump the high-water mark if a backed slot remains. Returns `None` when a
    /// new slab must be registered first.
    pub fn try_alloc_one(&mut self) -> Option<u32> {
        let gid = self.free.pop().or_else(|| {
            (self.hwm < self.capacity).then(|| {
                let g = self.hwm;
                self.hwm += 1;
                g
            })
        })?;
        self.live += 1;
        Some(gid)
    }

    /// Return a run of slots to the free-list. `live` saturates rather than
    /// underflowing so a stray double-free can't wrap the governor's evictable
    /// accounting to a huge value; a debug build additionally catches the
    /// double-free (which would alias one VRAM slot to two turns).
    pub fn free_run(&mut self, gids: &[u32]) {
        for &g in gids {
            debug_assert!(g < self.capacity, "freeing an out-of-range gid");
            debug_assert!(
                !self.free.contains(&g),
                "double-free of gallery page gid {g}"
            );
            self.live = self.live.saturating_sub(1);
            self.free.push(g);
        }
    }

    /// Whether the next slot needs a new slab (no free slot and hwm at capacity).
    #[inline]
    #[cfg(test)]
    pub fn needs_slab(&self) -> bool {
        self.free.is_empty() && self.hwm >= self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slab_pages_is_16mib_worth() {
        // page = 768 u64 = 6144 B → 16 MiB / 6144 = 2730 pages.
        assert_eq!(slab_pages(768), 2730);
        // A one-word page still yields a positive count.
        assert!(slab_pages(1) > 0);
    }

    #[test]
    fn alloc_bumps_hwm_within_capacity_then_needs_slab() {
        let mut p = PagePool::new(4); // tiny slab: 4 pages
        assert!(p.needs_slab(), "empty pool needs a slab");
        assert_eq!(p.try_alloc_one(), None, "no capacity yet");

        p.grow_one_slab();
        assert_eq!(p.capacity(), 4);
        assert_eq!(p.slab_count(), 1);
        // Four bumps of the high-water mark.
        assert_eq!(p.try_alloc_one(), Some(0));
        assert_eq!(p.try_alloc_one(), Some(1));
        assert_eq!(p.try_alloc_one(), Some(2));
        assert_eq!(p.try_alloc_one(), Some(3));
        assert_eq!(p.live(), 4);
        // Fifth needs a new slab.
        assert!(p.needs_slab());
        assert_eq!(p.try_alloc_one(), None);
    }

    #[test]
    fn freed_slots_recycle_before_hwm() {
        let mut p = PagePool::new(8);
        p.grow_one_slab();
        let a = p.try_alloc_one().unwrap(); // 0
        let b = p.try_alloc_one().unwrap(); // 1
        let _c = p.try_alloc_one().unwrap(); // 2
        assert_eq!((a, b), (0, 1));
        assert_eq!(p.live(), 3);

        // Free 0 and 1 → they recycle (LIFO) before the hwm (3) advances.
        p.free_run(&[a, b]);
        assert_eq!(p.live(), 1);
        assert_eq!(p.try_alloc_one(), Some(1)); // last freed first
        assert_eq!(p.try_alloc_one(), Some(0));
        assert_eq!(p.try_alloc_one(), Some(3)); // now the hwm resumes
        assert_eq!(p.live(), 4);
    }

    #[test]
    fn gid_decodes_to_slab_and_offset() {
        let p = PagePool::new(2730);
        assert_eq!(p.locate(0), (0, 0));
        assert_eq!(p.locate(2729), (0, 2729));
        assert_eq!(p.locate(2730), (1, 0));
        assert_eq!(p.locate(2731), (1, 1));
        assert_eq!(p.locate(5460), (2, 0));
    }

    #[test]
    fn multi_slab_capacity_and_live_accounting() {
        let mut p = PagePool::new(2);
        // Grow to 3 slabs = 6 slots, allocate all 6 across slab boundaries.
        for _ in 0..3 {
            p.grow_one_slab();
        }
        assert_eq!(p.capacity(), 6);
        let mut gids = Vec::new();
        for _ in 0..6 {
            gids.push(p.try_alloc_one().expect("within capacity"));
        }
        assert_eq!(gids, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(p.locate(gids[5]), (2, 1));
        assert!(p.needs_slab());
        assert_eq!(p.live(), 6);

        p.free_run(&gids);
        assert_eq!(p.live(), 0);
        // All six recycle before any new slab is needed.
        assert!(!p.needs_slab());
        for _ in 0..6 {
            assert!(p.try_alloc_one().is_some());
        }
    }
}
