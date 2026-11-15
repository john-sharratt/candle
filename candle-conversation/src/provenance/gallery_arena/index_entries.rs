//! The scan index's page entries: one per (page, segment), each naming the page,
//! the slots of it the segment scans, and the case they vote for
//! (`docs/gallery_residency.md` §5).
//!
//! A kernel's token space is `entries × 32` slots — slot `j` is slot `j & 31` of
//! entry `j >> 5`, scanned when its bit is set — so the index holds three `u32`s
//! a page, not two a token.

use super::pages::PAGE_TOKENS;

/// A page's slots, one bit each.
const _: () = assert!(PAGE_TOKENS == 32, "an entry's slot mask is one u32");

/// The entries of a scan index, built a scanned token at a time in scan order,
/// and the segment prefixes over them.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct IndexEntries {
    /// The page each entry scans.
    pub page: Vec<u32>,
    /// The slots of its page each entry scans, bit `s` for slot `s`.
    pub slots: Vec<u32>,
    /// The global case each entry's votes go to.
    pub case: Vec<u32>,
    /// Each segment's first slot, `entries × 32`, and the total after the last.
    pub seg_slot: Vec<i32>,
    /// Each segment's first scanned token, and the total after the last: the
    /// tokens a segment's vote is a mean and spread over.
    pub seg_tok: Vec<i32>,
    /// Scanned tokens so far.
    tokens: usize,
}

impl IndexEntries {
    pub fn new() -> Self {
        Self {
            seg_slot: vec![0],
            seg_tok: vec![0],
            ..Self::default()
        }
    }

    /// Scan slot `slot` of page `page` for case `case`: into the open entry when
    /// it is that page and case, else into a new one.
    pub fn push(&mut self, page: u32, slot: usize, case: u32) {
        debug_assert!(
            slot < PAGE_TOKENS,
            "slot {slot} of a {PAGE_TOKENS}-slot page"
        );
        let bit = 1u32 << slot;
        let open = self.page.len().checked_sub(1).filter(|&e| {
            // Within a segment: a segment's entries start after its prefix.
            e * PAGE_TOKENS >= *self.seg_slot.last().expect("a first prefix") as usize
                && self.page[e] == page
                && self.case[e] == case
                && self.slots[e] & bit == 0
        });
        match open {
            Some(e) => self.slots[e] |= bit,
            None => {
                self.page.push(page);
                self.slots.push(bit);
                self.case.push(case);
            }
        }
        self.tokens += 1;
    }

    /// Close the current segment: what follows is the next one's.
    pub fn end_segment(&mut self) {
        self.seg_slot.push((self.page.len() * PAGE_TOKENS) as i32);
        self.seg_tok.push(self.tokens as i32);
    }

    /// Scanned tokens in all.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Slots in all — the kernels' token space.
    pub fn slots_len(&self) -> usize {
        self.page.len() * PAGE_TOKENS
    }

    /// Device bytes of the per-entry and per-segment arrays.
    pub fn bytes(&self) -> u64 {
        ((self.page.len() * 3 + self.seg_slot.len() + self.seg_tok.len()) * 4) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A run of slots on one page for one case is one entry**; a new page, a
    /// new case or a new segment opens the next.
    #[test]
    fn a_page_scanned_for_one_case_is_one_entry() {
        let mut e = IndexEntries::new();
        // Segment 0: page 4 slots 0..=2 for case 7, then page 5 slot 0.
        for s in 0..3 {
            e.push(4, s, 7);
        }
        e.push(5, 0, 7);
        // The same page for another case is another entry.
        e.push(5, 1, 8);
        e.end_segment();
        // Segment 1 reads page 5 again: a new entry, past the segment's prefix.
        e.push(5, 1, 9);
        e.end_segment();
        assert_eq!(e.page, [4, 5, 5, 5]);
        assert_eq!(e.slots, [0b111, 0b1, 0b10, 0b10]);
        assert_eq!(e.case, [7, 7, 8, 9]);
        assert_eq!(e.seg_slot, [0, 96, 128]);
        assert_eq!(e.seg_tok, [0, 5, 6]);
        assert_eq!(e.tokens(), 6);
        assert_eq!(e.slots_len(), 128);
        assert_eq!(e.bytes(), (4 * 3 + 3 + 3) * 4);
    }

    /// **Gaps are bits left clear**: a narrower keep scans a scattered few slots
    /// of a page, and a window's edge a run of them — both one entry.
    #[test]
    fn gaps_in_a_page_are_bits_left_clear() {
        let mut e = IndexEntries::new();
        for s in [0, 3, 31] {
            e.push(2, s, 0);
        }
        // A window cut at slot 30 of page 3.
        for s in 30..32 {
            e.push(3, s, 0);
        }
        e.end_segment();
        assert_eq!(e.page, [2, 3]);
        assert_eq!(
            e.slots,
            [(1 << 0) | (1 << 3) | (1 << 31), (1 << 30) | (1 << 31)]
        );
        assert_eq!(e.seg_tok, [0, 5]);
    }

    /// **The same slot twice is two entries** — a turn two windows of one
    /// segment read over the same tokens counts them twice, as the per-token
    /// index did.
    #[test]
    fn a_slot_scanned_twice_is_two_entries() {
        let mut e = IndexEntries::new();
        e.push(1, 4, 2);
        e.push(1, 4, 2);
        e.end_segment();
        assert_eq!(e.page, [1, 1]);
        assert_eq!(e.slots, [1 << 4, 1 << 4]);
        assert_eq!(e.seg_tok, [0, 2]);
    }

    /// **An empty segment is an empty range**, and an empty index has none.
    #[test]
    fn an_empty_segment_is_an_empty_range() {
        let mut e = IndexEntries::new();
        e.end_segment();
        e.push(0, 0, 0);
        e.end_segment();
        e.end_segment();
        assert_eq!(e.seg_slot, [0, 0, 32, 32]);
        assert_eq!(e.seg_tok, [0, 0, 1, 1]);
    }
}
