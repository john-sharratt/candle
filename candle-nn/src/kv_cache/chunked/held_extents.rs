//! First-fit extent bookkeeping for the held heap: which byte ranges of a fixed
//! block are free, with neighbours merged on release.
//!
//! Pure host arithmetic over offsets, so every rule — alignment honoured on the
//! offset, the lowest fitting extent taken, adjacent frees coalesced, a double
//! or overlapping release refused — is tested without a device.

use std::collections::BTreeMap;

/// One range [`Extents::take`] handed out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Taken {
    /// The aligned offset the holder uses.
    pub at: usize,
    /// Where the range starts — `at` less the alignment padding in front of it.
    pub start: usize,
    /// Bytes from `start` the range covers, padding included: what
    /// [`Extents::release`] gives back.
    pub len: usize,
}

/// Free extents of a block of `capacity` bytes, keyed by offset.
#[derive(Debug, Clone)]
pub(crate) struct Extents {
    capacity: usize,
    /// `offset → len`, disjoint and never adjacent: a release merges.
    free: BTreeMap<usize, usize>,
    /// Bytes handed out and not yet released, alignment padding included.
    used: usize,
    /// High-water mark of `used`.
    peak: usize,
}

impl Extents {
    pub(crate) fn new(capacity: usize) -> Self {
        let mut free = BTreeMap::new();
        if capacity > 0 {
            free.insert(0, capacity);
        }
        Self {
            capacity,
            free,
            used: 0,
            peak: 0,
        }
    }

    /// Take `len` bytes at an offset that is a multiple of `align`, from the
    /// lowest free extent that holds them. `None` when nothing fits.
    pub(crate) fn take(&mut self, len: usize, align: usize) -> Option<Taken> {
        debug_assert!(align.is_power_of_two());
        let len = len.max(1);
        let (start, span, at) = self.free.iter().find_map(|(&start, &span)| {
            let at = start.next_multiple_of(align);
            (at + len <= start + span).then_some((start, span, at))
        })?;
        self.free.remove(&start);
        // The tail past the request stays free; the padding in front of `at`
        // is taken with it, so a release returns one contiguous range.
        let end = at + len;
        if end < start + span {
            self.free.insert(end, start + span - end);
        }
        let taken = end - start;
        self.used += taken;
        self.peak = self.peak.max(self.used);
        Some(Taken {
            at,
            start,
            len: taken,
        })
    }

    /// Give back `taken` bytes from `start`, merging with free neighbours.
    ///
    /// # Panics
    ///
    /// On a range outside the block or overlapping one already free — two
    /// holders believing they own the same bytes, which nothing can carry on
    /// from.
    pub(crate) fn release(&mut self, start: usize, taken: usize) {
        let end = start + taken;
        assert!(
            end <= self.capacity,
            "held extents: release of {start}..{end} past the {} B block",
            self.capacity
        );
        if let Some((&prev, &plen)) = self.free.range(..=start).next_back() {
            assert!(
                prev + plen <= start,
                "held extents: release of {start}..{end} overlaps free {prev}..{}",
                prev + plen
            );
        }
        if let Some((&next, &nlen)) = self.free.range(start..).next() {
            assert!(
                end <= next,
                "held extents: release of {start}..{end} overlaps free {next}..{}",
                next + nlen
            );
        }
        let mut lo = start;
        let mut len = taken;
        if let Some((&prev, &plen)) = self.free.range(..start).next_back() {
            if prev + plen == start {
                self.free.remove(&prev);
                lo = prev;
                len += plen;
            }
        }
        if let Some(nlen) = self.free.remove(&end) {
            len += nlen;
        }
        self.free.insert(lo, len);
        self.used -= taken;
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn used(&self) -> usize {
        self.used
    }

    pub(crate) fn peak(&self) -> usize {
        self.peak
    }

    /// The widest single request that would succeed at alignment 1.
    pub(crate) fn largest_free(&self) -> usize {
        self.free.values().copied().max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::{Extents, Taken};

    fn taken(at: usize, start: usize, len: usize) -> Option<Taken> {
        Some(Taken { at, start, len })
    }

    /// Requests are served lowest-first, each at its alignment, and the padding
    /// in front of an aligned start is part of what it takes.
    #[test]
    fn lowest_fit_with_alignment_padding() {
        let mut e = Extents::new(4096);
        assert_eq!(e.take(100, 256), taken(0, 0, 100));
        // Next free byte is 100; aligned to 256 the range is used from 256,
        // taking 156 bytes of padding plus 10.
        assert_eq!(e.take(10, 256), taken(256, 100, 166));
        assert_eq!(e.used(), 266);
        assert_eq!(e.largest_free(), 4096 - 266);
    }

    /// Releasing in any order merges back to one extent.
    #[test]
    fn releases_coalesce_with_both_neighbours() {
        let mut e = Extents::new(1024);
        let a = e.take(256, 256).unwrap();
        let b = e.take(256, 256).unwrap();
        let c = e.take(256, 256).unwrap();
        assert_eq!(
            (a.start, b.start, c.start, a.len, b.len, c.len),
            (0, 256, 512, 256, 256, 256)
        );
        e.release(a.start, a.len);
        e.release(c.start, c.len);
        assert_eq!(
            e.largest_free(),
            512,
            "c merges with the tail, a stands alone"
        );
        e.release(b.start, b.len);
        assert_eq!(e.largest_free(), 1024);
        assert_eq!(e.used(), 0);
        assert_eq!(e.peak(), 768);
    }

    /// A freed hole is reused before the tail.
    #[test]
    fn a_hole_is_refilled_first() {
        let mut e = Extents::new(1024);
        let a = e.take(256, 256).unwrap();
        let _b = e.take(256, 256).unwrap();
        e.release(a.start, a.len);
        assert_eq!(e.take(128, 256), taken(0, 0, 128));
    }

    /// Nothing fits once the block is exhausted, and a zero-byte request still
    /// takes a byte so it has an address of its own.
    #[test]
    fn exhaustion_and_empty_requests() {
        let mut e = Extents::new(512);
        assert_eq!(e.take(512, 256), taken(0, 0, 512));
        assert_eq!(e.take(1, 1), None);
        let mut e = Extents::new(512);
        assert_eq!(e.take(0, 1), taken(0, 0, 1));
        assert_eq!(Extents::new(0).take(1, 1), None);
    }

    #[test]
    #[should_panic(expected = "overlaps free")]
    fn a_double_release_is_refused() {
        let mut e = Extents::new(1024);
        let a = e.take(256, 256).unwrap();
        e.release(a.start, a.len);
        e.release(a.start, a.len);
    }
}
