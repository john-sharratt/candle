//! Where each record is, by its identity, and what a catch-up lists of it
//! (`docs/replicated_substrate.md` §6.2, §7.2).
//!
//! Maintenance relocates records, so a location captured earlier is stale; a
//! record owed to another member — sent as an event, or fetched by a catch-up
//! — is read through this index, which every append, relocation and rewrite
//! keeps current.
//!
//! **A listing reads no disk.** Beside each location sits the record's
//! [`ListedHeader`], so answering a catch-up's headers is a walk over memory
//! under a short hold of the persistence lock, not a header read per record
//! under a long one.
//!
//! **Forty-eight bytes a record.** An origin's `oseq`s are dense, so each origin
//! is a vector indexed by `oseq` from the first one held. A location packs into
//! one `u64`: the segment in the top 24 bits and the record's offset in
//! [`ALIGN`]-sized blocks in the low 40. Records are always aligned.

use std::collections::HashMap;
use std::iter::repeat_n;

use super::super::record::{RecordHeader, ALIGN};
use super::super::segment::SegmentId;
use super::listed_header::ListedHeader;

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

/// One `oseq`'s entry. A packed location of zero is "none": segment ids start
/// at one, so no record packs to zero.
#[derive(Clone, Copy, Debug)]
struct Slot {
    loc: u64,
    listed: ListedHeader,
}

impl Slot {
    const EMPTY: Slot = Slot {
        loc: 0,
        listed: ListedHeader::NONE,
    };
}

/// One origin's entries, indexed by `oseq - base`.
#[derive(Clone, Debug, Default)]
struct OriginLocations {
    base: u64,
    slots: Vec<Slot>,
}

/// See the module documentation.
#[derive(Clone, Debug, Default)]
pub struct LocationIndex {
    origins: HashMap<u64, OriginLocations>,
}

impl LocationIndex {
    /// Record that the record `header` heads is at `at`. A record with no
    /// identity is not indexed: it is never sent or fetched.
    pub fn insert(&mut self, header: &RecordHeader, at: Located) {
        let (origin, oseq) = (header.origin, header.oseq);
        if origin == 0 || oseq == 0 {
            return;
        }
        let slot = Slot {
            loc: at.pack(),
            listed: ListedHeader::of(header),
        };
        let o = self.origins.entry(origin).or_default();
        if o.slots.is_empty() {
            o.base = oseq;
        }
        if oseq < o.base {
            let grow = (o.base - oseq) as usize;
            o.slots.splice(0..0, repeat_n(Slot::EMPTY, grow));
            o.base = oseq;
        }
        let i = (oseq - o.base) as usize;
        if i >= o.slots.len() {
            o.slots.resize(i + 1, Slot::EMPTY);
        }
        o.slots[i] = slot;
    }

    fn slot(&self, origin: u64, oseq: u64) -> Option<&Slot> {
        let o = self.origins.get(&origin)?;
        let i = oseq.checked_sub(o.base)? as usize;
        o.slots.get(i).filter(|s| s.loc != 0)
    }

    /// Where `(origin, oseq)` is, if this member holds it.
    pub fn get(&self, origin: u64, oseq: u64) -> Option<Located> {
        self.slot(origin, oseq).map(|s| Located::unpack(s.loc))
    }

    /// What a catch-up lists of `(origin, oseq)`, if this member holds it.
    pub fn listed(&self, origin: u64, oseq: u64) -> Option<ListedHeader> {
        self.slot(origin, oseq).map(|s| s.listed)
    }

    /// Forget every record in a segment `gone` names — what dropping those
    /// segments leaves of them is nothing.
    pub fn forget_segments(&mut self, gone: &[SegmentId]) {
        for o in self.origins.values_mut() {
            for slot in &mut o.slots {
                if slot.loc != 0 && gone.contains(&Located::unpack(slot.loc).segment) {
                    *slot = Slot::EMPTY;
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
    use std::mem::size_of;

    use super::*;
    use crate::persistence::record::RecordType;

    fn at(segment: u64, block: u64) -> Located {
        Located {
            segment: SegmentId(segment),
            offset: block * ALIGN as u64,
        }
    }

    fn head(origin: u64, oseq: u64) -> RecordHeader {
        RecordHeader {
            record_type: RecordType::Tokens,
            payload_len: 10,
            stream_id: oseq * 3,
            chunk_index: oseq,
            origin,
            oseq,
            ..RecordHeader::NONE
        }
    }

    #[test]
    fn a_location_packs_into_one_word() {
        assert_eq!(at(1, 0).pack(), 1 << 40);
        assert_eq!(at(3, 5).pack(), (3 << 40) | 5);
        assert_eq!(Located::unpack((3 << 40) | 5), at(3, 5));
    }

    #[test]
    fn an_entry_is_forty_eight_bytes() {
        assert_eq!(size_of::<Slot>(), 48);
    }

    #[test]
    fn records_are_found_by_identity_in_any_order() {
        let mut idx = LocationIndex::default();
        idx.insert(&head(7, 10), at(1, 2));
        idx.insert(&head(7, 8), at(1, 1));
        idx.insert(&head(7, 12), at(2, 0));
        idx.insert(&head(9, 1), at(2, 3));
        assert_eq!(idx.get(7, 8), Some(at(1, 1)));
        assert_eq!(idx.get(7, 10), Some(at(1, 2)));
        assert_eq!(idx.get(7, 12), Some(at(2, 0)));
        assert_eq!(idx.get(7, 9), None, "a gap");
        assert_eq!(idx.get(7, 7), None, "below the first");
        assert_eq!(idx.get(7, 13), None, "above the last");
        assert_eq!(idx.get(9, 1), Some(at(2, 3)));
        idx.insert(&head(0, 4), at(1, 9));
        idx.insert(&head(7, 0), at(1, 9));
        assert_eq!(idx.get(0, 4), None, "an unstamped record is not indexed");
    }

    /// Each held record lists its own header; a gap lists nothing.
    #[test]
    fn a_held_record_lists_its_header() {
        let mut idx = LocationIndex::default();
        idx.insert(&head(7, 10), at(1, 2));
        idx.insert(&head(7, 8), at(1, 1));
        assert_eq!(idx.listed(7, 8), Some(ListedHeader::of(&head(7, 8))));
        assert_eq!(idx.listed(7, 10), Some(ListedHeader::of(&head(7, 10))));
        assert_eq!(idx.listed(7, 9), None);
        assert_eq!(idx.listed(7, 10).unwrap().chunk_index, 10);
    }

    /// A relocated copy moves the entry; dropped segments take theirs, header
    /// and all.
    #[test]
    fn relocation_moves_and_a_drop_forgets() {
        let mut idx = LocationIndex::default();
        idx.insert(&head(7, 1), at(1, 0));
        idx.insert(&head(7, 2), at(1, 1));
        idx.insert(&head(7, 3), at(2, 0));
        idx.insert(&head(7, 1), at(3, 0));
        idx.forget_segments(&[SegmentId(1)]);
        assert_eq!(idx.get(7, 1), Some(at(3, 0)));
        assert_eq!(idx.listed(7, 1), Some(ListedHeader::of(&head(7, 1))));
        assert_eq!(idx.get(7, 2), None);
        assert_eq!(idx.listed(7, 2), None);
        assert_eq!(idx.get(7, 3), Some(at(2, 0)));
    }

    #[test]
    #[should_panic(expected = "not aligned")]
    fn an_unaligned_offset_is_refused() {
        LocationIndex::default().insert(
            &head(7, 1),
            Located {
                segment: SegmentId(1),
                offset: 100,
            },
        );
    }
}
