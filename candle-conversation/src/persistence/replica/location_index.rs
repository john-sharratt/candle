//! Where each record is, by its identity (`docs/replicated_substrate.md` §6.2).
//!
//! Maintenance relocates records, so a location captured earlier is stale; a
//! record owed to another member — sent as an event, or fetched by a catch-up
//! — is read through this index, which every append, relocation and rewrite
//! keeps current.
//!
//! **Eight bytes a record.** An origin's `oseq`s are dense, so each origin is a
//! vector indexed by `oseq` from the first one held, and a location packs into
//! one `u64`: the segment in the top 24 bits and the record's offset in
//! [`ALIGN`]-sized blocks in the low 40. Records are always aligned, and the
//! record's size is read from its header when it is read.

use std::collections::HashMap;
use std::iter::repeat_n;

use super::super::record::ALIGN;
use super::super::segment::SegmentId;

const SEGMENT_BITS: u32 = 24;
const BLOCK_BITS: u32 = 64 - SEGMENT_BITS;

/// One record's place: its segment and its offset in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Located {
    pub segment: SegmentId,
    pub offset: u64,
}

impl Located {
    /// The location as one word. A segment or an offset past what the word
    /// holds is a fault, never a silent truncation.
    fn pack(self) -> u64 {
        assert!(
            self.segment.raw() < (1 << SEGMENT_BITS),
            "segment {} is past the location index's {SEGMENT_BITS}-bit field",
            self.segment.raw()
        );
        assert!(
            self.offset.is_multiple_of(ALIGN as u64),
            "record offset {} is not aligned to {ALIGN}",
            self.offset
        );
        let block = self.offset / ALIGN as u64;
        assert!(
            block < (1 << BLOCK_BITS),
            "record offset {} is past the location index's {BLOCK_BITS}-bit field",
            self.offset
        );
        (self.segment.raw() << BLOCK_BITS) | block
    }

    fn unpack(packed: u64) -> Located {
        Located {
            segment: SegmentId(packed >> BLOCK_BITS),
            offset: (packed & ((1 << BLOCK_BITS) - 1)) * ALIGN as u64,
        }
    }
}

/// One origin's locations, indexed by `oseq - base`. Zero is "none": segment
/// ids start at one, so no record packs to zero.
#[derive(Clone, Debug, Default)]
struct OriginLocations {
    base: u64,
    locs: Vec<u64>,
}

/// See the module documentation.
#[derive(Clone, Debug, Default)]
pub struct LocationIndex {
    origins: HashMap<u64, OriginLocations>,
}

impl LocationIndex {
    /// Record that `(origin, oseq)` is at `at`. A record with no identity is
    /// not indexed: it is never sent or fetched.
    pub fn insert(&mut self, origin: u64, oseq: u64, at: Located) {
        if origin == 0 || oseq == 0 {
            return;
        }
        let packed = at.pack();
        let o = self.origins.entry(origin).or_default();
        if o.locs.is_empty() {
            o.base = oseq;
        }
        if oseq < o.base {
            let grow = (o.base - oseq) as usize;
            o.locs.splice(0..0, repeat_n(0, grow));
            o.base = oseq;
        }
        let i = (oseq - o.base) as usize;
        if i >= o.locs.len() {
            o.locs.resize(i + 1, 0);
        }
        o.locs[i] = packed;
    }

    /// Where `(origin, oseq)` is, if this member holds it.
    pub fn get(&self, origin: u64, oseq: u64) -> Option<Located> {
        let o = self.origins.get(&origin)?;
        let i = oseq.checked_sub(o.base)? as usize;
        match o.locs.get(i).copied() {
            Some(0) | None => None,
            Some(packed) => Some(Located::unpack(packed)),
        }
    }

    /// Forget every record in a segment `gone` names — what dropping those
    /// segments leaves of them is nothing.
    pub fn forget_segments(&mut self, gone: &[SegmentId]) {
        for o in self.origins.values_mut() {
            for slot in &mut o.locs {
                if *slot != 0 && gone.contains(&Located::unpack(*slot).segment) {
                    *slot = 0;
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.origins.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(segment: u64, block: u64) -> Located {
        Located {
            segment: SegmentId(segment),
            offset: block * ALIGN as u64,
        }
    }

    #[test]
    fn a_location_packs_into_one_word() {
        assert_eq!(at(1, 0).pack(), 1 << 40);
        assert_eq!(at(3, 5).pack(), (3 << 40) | 5);
        assert_eq!(Located::unpack((3 << 40) | 5), at(3, 5));
    }

    #[test]
    fn records_are_found_by_identity_in_any_order() {
        let mut idx = LocationIndex::default();
        idx.insert(7, 10, at(1, 2));
        idx.insert(7, 8, at(1, 1));
        idx.insert(7, 12, at(2, 0));
        idx.insert(9, 1, at(2, 3));
        assert_eq!(idx.get(7, 8), Some(at(1, 1)));
        assert_eq!(idx.get(7, 10), Some(at(1, 2)));
        assert_eq!(idx.get(7, 12), Some(at(2, 0)));
        assert_eq!(idx.get(7, 9), None, "a gap");
        assert_eq!(idx.get(7, 7), None, "below the first");
        assert_eq!(idx.get(7, 13), None, "above the last");
        assert_eq!(idx.get(9, 1), Some(at(2, 3)));
        idx.insert(0, 4, at(1, 9));
        idx.insert(7, 0, at(1, 9));
        assert_eq!(idx.get(0, 4), None, "an unstamped record is not indexed");
    }

    /// A relocated copy moves the entry; dropped segments take theirs.
    #[test]
    fn relocation_moves_and_a_drop_forgets() {
        let mut idx = LocationIndex::default();
        idx.insert(7, 1, at(1, 0));
        idx.insert(7, 2, at(1, 1));
        idx.insert(7, 3, at(2, 0));
        idx.insert(7, 1, at(3, 0));
        idx.forget_segments(&[SegmentId(1)]);
        assert_eq!(idx.get(7, 1), Some(at(3, 0)));
        assert_eq!(idx.get(7, 2), None);
        assert_eq!(idx.get(7, 3), Some(at(2, 0)));
    }

    #[test]
    #[should_panic(expected = "not aligned")]
    fn an_unaligned_offset_is_refused() {
        LocationIndex::default().insert(
            7,
            1,
            Located {
                segment: SegmentId(1),
                offset: 100,
            },
        );
    }
}
