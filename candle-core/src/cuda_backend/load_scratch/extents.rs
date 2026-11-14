//! First-fit extents over a fixed run of bytes: the load scratch's allocator.
//!
//! Offsets, not addresses: the run is `0..capacity` and the caller adds its base.
//! A load takes and frees in roughly stack order — a source, its bands, the
//! source dropped before the next — so first fit keeps the run's low end busy
//! and its high end whole, and coalescing on release is all the compaction it
//! needs.

use std::collections::BTreeMap;

/// The free runs of `0..capacity`, by start offset, never two touching.
pub(super) struct Extents {
    free: BTreeMap<usize, usize>,
    capacity: usize,
}

impl Extents {
    /// The whole of `0..capacity` free.
    pub(super) fn new(capacity: usize) -> Self {
        let mut free = BTreeMap::new();
        if capacity > 0 {
            free.insert(0, capacity);
        }
        Self { free, capacity }
    }

    /// Take `len` bytes at an `align`-aligned offset from the lowest free run
    /// that holds them, or `None` when none does. The alignment pad stays free.
    pub(super) fn take(&mut self, len: usize, align: usize) -> Option<usize> {
        let (start, run, at) = self.free.iter().find_map(|(&start, &run)| {
            let at = start.next_multiple_of(align);
            (at - start + len <= run).then_some((start, run, at))
        })?;
        self.free.remove(&start);
        if at > start {
            self.free.insert(start, at - start);
        }
        let end = at + len;
        if end < start + run {
            self.free.insert(end, start + run - end);
        }
        Some(at)
    }

    /// Give back `len` bytes at `offset`, merged with the free runs either side.
    pub(super) fn release(&mut self, offset: usize, len: usize) {
        let mut start = offset;
        let mut run = len;
        if let Some((&prev, &prev_len)) = self.free.range(..offset).next_back() {
            if prev + prev_len == offset {
                self.free.remove(&prev);
                start = prev;
                run += prev_len;
            }
        }
        if let Some(&next_len) = self.free.get(&(offset + len)) {
            self.free.remove(&(offset + len));
            run += next_len;
        }
        self.free.insert(start, run);
    }

    /// The run's size.
    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Whether every byte is free again.
    pub(super) fn is_whole(&self) -> bool {
        self.capacity == 0 || self.free.get(&0) == Some(&self.capacity)
    }

    /// The free runs, as `(offset, len)`, lowest first.
    #[cfg(test)]
    fn runs(&self) -> Vec<(usize, usize)> {
        self.free.iter().map(|(&s, &l)| (s, l)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Extents;

    /// Takes come from the low end, aligned, and the pad an alignment skips
    /// stays free.
    #[test]
    fn takes_are_first_fit_and_aligned() {
        let mut e = Extents::new(4096);
        assert_eq!(e.take(100, 256), Some(0));
        assert_eq!(e.runs(), vec![(100, 3996)]);
        assert_eq!(e.take(100, 256), Some(256));
        assert_eq!(e.runs(), vec![(100, 156), (356, 3740)]);
        // Fits the pad left in front of 256.
        assert_eq!(e.take(156, 1), Some(100));
        assert_eq!(e.runs(), vec![(356, 3740)]);
    }

    /// A take no run holds is refused and changes nothing.
    #[test]
    fn a_take_past_every_run_is_refused() {
        let mut e = Extents::new(1024);
        assert_eq!(e.take(512, 256), Some(0));
        assert_eq!(e.take(768, 256), None);
        assert_eq!(e.runs(), vec![(512, 512)]);
        assert_eq!(Extents::new(0).take(1, 1), None);
    }

    /// Releases merge with both neighbours, and the run is whole once every
    /// take is back, in whatever order.
    #[test]
    fn releases_coalesce_back_to_whole() {
        let mut e = Extents::new(3072);
        let a = e.take(1024, 256).unwrap();
        let b = e.take(1024, 256).unwrap();
        let c = e.take(1024, 256).unwrap();
        assert_eq!((a, b, c), (0, 1024, 2048));
        assert!(e.runs().is_empty());
        e.release(a, 1024);
        e.release(c, 1024);
        assert_eq!(e.runs(), vec![(0, 1024), (2048, 1024)]);
        assert!(!e.is_whole());
        e.release(b, 1024);
        assert_eq!(e.runs(), vec![(0, 3072)]);
        assert!(e.is_whole());
    }
}
