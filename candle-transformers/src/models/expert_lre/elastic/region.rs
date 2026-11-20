//! The elastic warm tier's pinned memory: equal segments, each one
//! `cuMemAllocHost` block (`docs/elastic_warm_tier.md` §5.1).
//!
//! The pad's resident slots index into it — slot `r` is slot `r % per` of
//! segment `r / per` — so growing appends a segment and shrinking takes the
//! last one off: the slots below never move. Each segment's range is in the
//! pinned range table while the segment exists: added before anything can
//! point into it, removed before it is freed.

use std::sync::{Arc, RwLock};

use candle::quantized::pinned_ranges::PinnedRangeTable;
use candle::vram::PinnedUse;

use super::super::pinned::WarmPool;

/// A segment's size, before it is cut to whole slots.
pub(crate) const SEGMENT_BYTES: usize = 1024 * 1024 * 1024;

/// Slots in a segment of `stride`-byte slots for a tier that wants
/// `want_slots`: a [`SEGMENT_BYTES`]'s worth, but no more than the tier wants,
/// and at least one.
pub(crate) fn segment_slots(stride: usize, want_slots: usize) -> usize {
    (SEGMENT_BYTES / stride.max(1)).clamp(1, want_slots.max(1))
}

pub(crate) struct ResidentRegion {
    stride: usize,
    per: usize,
    segments: RwLock<Vec<WarmPool>>,
    ranges: Arc<PinnedRangeTable>,
}

impl ResidentRegion {
    /// Empty, for segments of `per` slots of `stride` bytes whose ranges go in
    /// `ranges`.
    pub(crate) fn new(stride: usize, per: usize, ranges: Arc<PinnedRangeTable>) -> Self {
        Self {
            stride,
            per: per.max(1),
            segments: RwLock::new(Vec::new()),
            ranges,
        }
    }

    /// Slots a segment holds.
    pub(crate) fn per_segment(&self) -> usize {
        self.per
    }

    /// Bytes a segment holds.
    pub(crate) fn segment_bytes(&self) -> u64 {
        (self.per * self.stride) as u64
    }

    pub(crate) fn segments(&self) -> usize {
        self.read().len()
    }

    /// Slots held, every segment's.
    pub(crate) fn slots(&self) -> usize {
        self.segments() * self.per
    }

    /// Bytes held, every segment's.
    pub(crate) fn held_bytes(&self) -> u64 {
        self.slots() as u64 * self.stride as u64
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, Vec<WarmPool>> {
        self.segments.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Take one more segment: allocated, its range in the table, then
    /// appended. `false` when the driver refuses it or the table is full.
    pub(crate) fn grow(&self) -> bool {
        let Some(pool) = WarmPool::try_alloc(self.per, self.stride, PinnedUse::WeightWarmTier)
        else {
            return false;
        };
        let (lo, hi) = pool.range();
        if self.ranges.add(lo, hi).is_err() {
            return false;
        }
        self.segments
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .push(pool);
        true
    }

    /// Give the last segment back: its range out of the table, then freed.
    /// Only once nothing points into it — the stager has drained its slots.
    pub(crate) fn pop(&self) -> bool {
        let pool = self
            .segments
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .pop();
        match pool {
            Some(pool) => {
                self.ranges.remove(pool.range().0);
                drop(pool);
                true
            }
            None => false,
        }
    }

    /// Slot `r`'s address, host and device alike.
    pub(crate) fn addr(&self, r: usize) -> u64 {
        let segs = self.read();
        let seg = segs.get(r / self.per).unwrap_or_else(|| {
            panic!(
                "elastic warm tier: slot {r} is past the {} slots held",
                segs.len() * self.per
            )
        });
        seg.slot_addr(r % self.per)
    }

    /// Slot `r`'s bytes, for a writer that owns the slot.
    ///
    /// # Safety
    ///
    /// No other reference to slot `r`'s bytes may be live while the returned
    /// slice is, and the segment holding it must not be popped while it is.
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn slot_mut(&self, r: usize) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.addr(r) as *mut u8, self.stride)
    }

    /// Slot `r`'s first `len` bytes.
    pub(crate) fn slot_ref(&self, r: usize, len: usize) -> &[u8] {
        assert!(
            len <= self.stride,
            "elastic warm tier: a {len} B read past the stride"
        );
        // SAFETY: the slot is inside a held segment (`addr` checks), and the
        // caller holds the region, so the segment outlives the borrow unless
        // popped — which the stager does only once nothing reads it.
        unsafe { std::slice::from_raw_parts(self.addr(r) as *const u8, len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A segment is a gigabyte cut to whole slots, never more than the tier
    /// wants and never none.**
    #[test]
    fn a_segment_is_a_gigabyte_of_whole_slots() {
        let stride = 1_998_848; // Flash-Next's expert record, 1.91 MiB.
        assert_eq!(segment_slots(stride, 24_064), 537);
        assert_eq!(
            segment_slots(stride, 100),
            100,
            "a small tier is one segment"
        );
        assert_eq!(segment_slots(stride, 0), 1);
        assert_eq!(
            segment_slots(4 * SEGMENT_BYTES, 10),
            1,
            "a slot larger than a segment"
        );
    }

    /// **Segments grow and shrink at the end, each range in the table while it
    /// is held**: slot addresses index across them in order.
    #[test]
    fn segments_grow_and_shrink_at_the_end_with_their_ranges() -> candle::Result<()> {
        let Ok(_dev) = candle::Device::new_cuda(0) else {
            return Ok(());
        };
        let ranges = Arc::new(PinnedRangeTable::new()?);
        let r = ResidentRegion::new(4096, 4, ranges.clone());
        assert!(r.grow() && r.grow());
        assert_eq!((r.segments(), r.slots()), (2, 8));
        let held = ranges.ranges();
        assert_eq!(held.len(), 2);
        for s in 0..8 {
            let a = r.addr(s);
            let (lo, hi) = held[s / 4];
            assert_eq!(a, lo + (s % 4) as u64 * 4096, "slot {s}");
            assert!(a < hi && ranges.contains(a));
        }
        let gone = r.addr(4);
        assert!(r.pop());
        assert_eq!(r.slots(), 4);
        assert!(
            !ranges.contains(gone),
            "the popped segment's range left the table"
        );
        assert!(ranges.contains(r.addr(3)));
        assert!(r.pop() && !r.pop());
        Ok(())
    }
}
