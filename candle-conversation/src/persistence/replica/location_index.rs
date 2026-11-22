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
//! **Forty-eight bytes a record.** An origin's `oseq`s are dense where records
//! are held, so the index is pages of [`PAGE_SLOTS`] consecutive `oseq`s, a
//! page allocated only while it holds a record. A location packs into one
//! `u64`: the segment in the top 24 bits and the record's offset in
//! [`ALIGN`]-sized blocks in the low 40. Records are always aligned.

use std::collections::HashMap;
use std::mem::size_of;

use serde::Serialize;

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

/// `oseq`s a page holds, as a power of two.
const PAGE_BITS: u32 = 10;

/// Slots in one page: 48 KiB of them.
const PAGE_SLOTS: usize = 1 << PAGE_BITS;

/// [`PAGE_SLOTS`] consecutive `oseq`s' slots, and how many of them name a
/// record.
#[derive(Clone, Debug)]
struct Page {
    slots: Box<[Slot]>,
    held: u32,
}

impl Page {
    fn new() -> Self {
        Self {
            slots: vec![Slot::EMPTY; PAGE_SLOTS].into_boxed_slice(),
            held: 0,
        }
    }
}

/// See the module documentation.
///
/// **Pages, allocated only where a record is held.** An origin's `oseq`s are
/// dense when written, but maintenance drops whole segments, and with them
/// runs of `oseq`s from the middle of an origin's range; a load then indexes
/// only the records left. One vector per origin spanning the first `oseq` to
/// the last kept a slot for every gap: on WhiteWorld (2026-11-22) zend's own
/// origin held 65.9 M slots, 3.43 GB, to name 22.1 M records. Keyed by
/// `(origin, oseq >> PAGE_BITS)`, a page goes when its last record does, and a
/// gap costs nothing. A lookup that finds no page, and one that finds an
/// empty slot, answer alike: not held.
#[derive(Clone, Debug, Default)]
pub struct LocationIndex {
    pages: HashMap<(u64, u64), Page>,
}

/// The page key and the slot within it of `(origin, oseq)`.
fn place(origin: u64, oseq: u64) -> ((u64, u64), usize) {
    (
        (origin, oseq >> PAGE_BITS),
        (oseq & (PAGE_SLOTS as u64 - 1)) as usize,
    )
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
        let (key, i) = place(origin, oseq);
        let page = self.pages.entry(key).or_insert_with(Page::new);
        if page.slots[i].loc == 0 {
            page.held += 1;
        }
        page.slots[i] = slot;
    }

    fn slot(&self, origin: u64, oseq: u64) -> Option<&Slot> {
        let (key, i) = place(origin, oseq);
        self.pages
            .get(&key)
            .map(|p| &p.slots[i])
            .filter(|s| s.loc != 0)
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
    /// segments leaves of them is nothing — and every page left holding none.
    pub fn forget_segments(&mut self, gone: &[SegmentId]) {
        for page in self.pages.values_mut() {
            for slot in page.slots.iter_mut() {
                if slot.loc != 0 && gone.contains(&Located::unpack(slot.loc).segment) {
                    *slot = Slot::EMPTY;
                    page.held -= 1;
                }
            }
        }
        self.pages.retain(|_, p| p.held > 0);
    }

    pub fn clear(&mut self) {
        self.pages.clear();
    }

    /// What the index holds, origin by origin, for the memory census
    /// (`docs/memory_census.md` §5): the span from its first record held to
    /// its last, the slots its pages allocate, and how many name a record.
    /// `capacity` past `held` is the empty slots of pages still holding one.
    /// O(slots).
    pub fn gauge(&self) -> Vec<OriginGauge> {
        let mut by_origin: HashMap<u64, (u64, u64, u64, u64)> = HashMap::new();
        for (&(origin, page_no), page) in &self.pages {
            let base = page_no << PAGE_BITS;
            let mut held = page.slots.iter().enumerate().filter(|(_, s)| s.loc != 0);
            let Some((first, _)) = held.next() else {
                continue;
            };
            let last = held.next_back().map_or(first, |(i, _)| i);
            let g = by_origin.entry(origin).or_insert((u64::MAX, 0, 0, 0));
            g.0 = g.0.min(base + first as u64);
            g.1 = g.1.max(base + last as u64);
            g.2 += PAGE_SLOTS as u64;
            g.3 += page.held as u64;
        }
        let mut out: Vec<OriginGauge> = by_origin
            .into_iter()
            .map(|(origin, (first, last, capacity, held))| OriginGauge {
                origin,
                len: last - first + 1,
                capacity,
                held,
                bytes: capacity * SLOT_BYTES as u64,
            })
            .collect();
        out.sort_by_key(|g| g.origin);
        out
    }
}

/// The bytes one slot takes.
pub const SLOT_BYTES: usize = size_of::<Slot>();

/// One origin's share of the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct OriginGauge {
    pub origin: u64,
    /// Slots, from the first `oseq` held to the last.
    pub len: u64,
    /// Slots its pages allocate.
    pub capacity: u64,
    /// Slots naming a record this member holds.
    pub held: u64,
    /// The capacity in bytes.
    pub bytes: u64,
}

#[cfg(test)]
mod tests {
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
        assert_eq!(SLOT_BYTES, 48);
    }

    /// **The gauge counts what the index keeps, held or not**: the gaps
    /// between records still held are slots that name nothing, and the gauge
    /// says so.
    #[test]
    fn the_gauge_counts_the_gaps_between_records_held() {
        let mut idx = LocationIndex::default();
        for oseq in 1..=1000 {
            let segment = if oseq == 1 || oseq == 1000 { 2 } else { 1 };
            idx.insert(&head(7, oseq), at(segment, oseq));
        }
        idx.insert(&head(9, 5), at(2, 0));
        let g = idx.gauge();
        assert_eq!(g.len(), 2);
        assert_eq!((g[0].origin, g[0].len, g[0].held), (7, 1000, 1000));
        assert!(g[0].capacity >= 1000);
        assert_eq!(g[0].bytes, g[0].capacity * 48);
        assert_eq!((g[1].origin, g[1].len, g[1].held), (9, 1, 1));

        idx.forget_segments(&[SegmentId(1)]);
        let g = idx.gauge();
        assert_eq!((g[0].len, g[0].held), (1000, 2), "the gap between the two");
    }

    /// **A gap in the middle costs nothing**: an origin whose middle pages'
    /// records are all dropped keeps only the pages still holding one —
    /// exactly what a load of the records left indexes.
    #[test]
    fn a_dropped_run_gives_its_pages_back() {
        let mut idx = LocationIndex::default();
        let pages = 6 * PAGE_SLOTS as u64;
        for oseq in 1..pages {
            let page = oseq >> PAGE_BITS;
            let segment = if page == 0 || page == 5 { 2 } else { 1 };
            idx.insert(&head(7, oseq), at(segment, oseq));
        }
        assert_eq!(idx.gauge()[0].capacity, pages);
        idx.forget_segments(&[SegmentId(1)]);
        let g = idx.gauge();
        assert_eq!(g[0].capacity, 2 * PAGE_SLOTS as u64, "two pages left");
        assert_eq!(g[0].held, 2 * PAGE_SLOTS as u64 - 1);
        assert_eq!((g[0].len, g[0].bytes), (pages - 1, g[0].capacity * 48));

        // The same records indexed into a fresh index — a load — allocate
        // the same two pages.
        let mut loaded = LocationIndex::default();
        for oseq in (1..PAGE_SLOTS as u64).chain(5 * PAGE_SLOTS as u64..pages) {
            loaded.insert(&head(7, oseq), at(2, oseq));
        }
        assert_eq!(loaded.gauge(), g);
    }

    /// A record relocated in place is counted once, and a page goes only
    /// when its last record does.
    #[test]
    fn a_page_counts_its_records_once() {
        let mut idx = LocationIndex::default();
        idx.insert(&head(7, 3), at(1, 0));
        idx.insert(&head(7, 3), at(2, 0));
        idx.insert(&head(7, 4), at(1, 1));
        assert_eq!(idx.gauge()[0].held, 2);
        idx.forget_segments(&[SegmentId(1)]);
        assert_eq!(idx.gauge()[0].held, 1);
        assert_eq!(idx.get(7, 3), Some(at(2, 0)));
        idx.forget_segments(&[SegmentId(2)]);
        assert!(idx.gauge().is_empty());
        assert_eq!(idx.get(7, 3), None);
    }

    /// **A trim changes no answer**: every record held is found where it
    /// was, everything else is not held — below, inside and above the run —
    /// and a record indexed after the trim, at either end, is found too.
    #[test]
    fn lookups_answer_the_same_after_a_trim() {
        let mut idx = LocationIndex::default();
        for oseq in 1..=100 {
            let segment = if (40..=60).contains(&oseq) && oseq % 2 == 0 {
                2
            } else {
                1
            };
            idx.insert(&head(7, oseq), at(segment, oseq));
        }
        idx.forget_segments(&[SegmentId(1)]);
        for oseq in 1..=100 {
            let held = (40..=60).contains(&oseq) && oseq % 2 == 0;
            assert_eq!(idx.get(7, oseq), held.then(|| at(2, oseq)), "oseq {oseq}");
            assert_eq!(
                idx.listed(7, oseq),
                held.then(|| ListedHeader::of(&head(7, oseq))),
                "oseq {oseq}"
            );
        }
        idx.insert(&head(7, 5), at(3, 0));
        idx.insert(&head(7, 200), at(3, 1));
        assert_eq!(idx.get(7, 5), Some(at(3, 0)));
        assert_eq!(idx.get(7, 200), Some(at(3, 1)));
        assert_eq!(idx.get(7, 40), Some(at(2, 40)));
        assert_eq!(idx.get(7, 41), None);
    }

    /// An origin with nothing left is gone from the index, and comes back
    /// whole when its records are indexed again.
    #[test]
    fn an_origin_with_nothing_held_is_dropped() {
        let mut idx = LocationIndex::default();
        for oseq in 1..=10 {
            idx.insert(&head(7, oseq), at(1, oseq));
        }
        idx.insert(&head(9, 1), at(2, 0));
        idx.forget_segments(&[SegmentId(1)]);
        let g = idx.gauge();
        assert_eq!(g.iter().map(|g| g.origin).collect::<Vec<_>>(), vec![9]);
        idx.insert(&head(7, 3), at(3, 0));
        assert_eq!(idx.get(7, 3), Some(at(3, 0)));
        assert_eq!(idx.get(7, 2), None);
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
