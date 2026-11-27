//! A reserved range's places, top first (`docs/placed_section_layout.md`
//! §3.2).
//!
//! The top of the range is nearest the question. The first section to enter
//! takes the top; each one after enters **in front** — immediately below the
//! lowest placed — so nothing already placed moves. A section that leaves
//! leaves a hole where it stood. A full range is a window: an entry still
//! enters in front, and what passes the top — the earliest-entered places
//! that may go, holes and sections alike — leaves, the run sliding up by what
//! they freed. A mandatory section never leaves and never slides out. Between
//! requests the range is compacted: its holes close, the sections moving up
//! in order.

use super::place::{Entry, Place};
use crate::files::key::SectionKey;

/// One range's places and its size.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RangeScope {
    /// The range's positions.
    pub positions: usize,
    /// The places, top first: the first is nearest the question.
    places: Vec<Place>,
}

impl RangeScope {
    /// An empty range of `positions`.
    pub fn new(positions: usize) -> Self {
        Self {
            positions,
            places: Vec::new(),
        }
    }

    /// The places, top first.
    pub fn places(&self) -> &[Place] {
        &self.places
    }

    /// The positions the places take, holes included.
    pub fn used(&self) -> usize {
        self.places.iter().map(Place::positions).sum()
    }

    /// Whether `key` is placed.
    pub fn holds(&self, key: SectionKey) -> bool {
        self.entry(key).is_some()
    }

    /// The placed section `key`, if it is placed.
    pub fn entry(&self, key: SectionKey) -> Option<&Entry> {
        self.places
            .iter()
            .filter_map(Place::section)
            .find(|e| e.key == key)
    }

    /// The placed sections, top first.
    pub fn sections(&self) -> impl Iterator<Item = &Entry> {
        self.places.iter().filter_map(Place::section)
    }

    /// Enter `key`, of `tokens` positions, in front of every placed section,
    /// and return the sections a full range slid out to make room, top first.
    /// `None`, and nothing changed, when it cannot fit even with every place
    /// that may go gone, or it is already placed.
    pub fn enter(
        &mut self,
        key: SectionKey,
        tokens: usize,
        mandatory: bool,
    ) -> Option<Vec<SectionKey>> {
        if self.holds(key) {
            return None;
        }
        let free = self.positions.saturating_sub(self.used());
        let movable: usize = self
            .places
            .iter()
            .filter(|p| p.section().is_none_or(|e| !e.mandatory))
            .map(Place::positions)
            .sum();
        if tokens > free + movable {
            return None;
        }
        let mut free = free;
        let mut out = Vec::new();
        while tokens > free {
            let at = self
                .places
                .iter()
                .position(|p| p.section().is_none_or(|e| !e.mandatory))
                .expect("the movable places cover the entry");
            let gone = self.places.remove(at);
            free += gone.positions();
            if let Place::Section(e) = gone {
                out.push(e.key);
            }
        }
        self.places.push(Place::Section(Entry {
            key,
            tokens,
            mandatory,
        }));
        Some(out)
    }

    /// `key` leaves and its positions become a hole, merged with a hole
    /// beside it. A mandatory section, or one not placed, does not leave.
    pub fn leave(&mut self, key: SectionKey) -> bool {
        let Some(at) = self
            .places
            .iter()
            .position(|p| p.section().is_some_and(|e| e.key == key && !e.mandatory))
        else {
            return false;
        };
        let mut hole = self.places[at].positions();
        let mut start = at;
        if at + 1 < self.places.len() {
            if let Place::Hole(n) = self.places[at + 1] {
                hole += n;
                self.places.remove(at + 1);
            }
        }
        if at > 0 {
            if let Place::Hole(n) = self.places[at - 1] {
                hole += n;
                self.places.remove(at - 1);
                start = at - 1;
            }
        }
        self.places[start] = Place::Hole(hole);
        // A hole at the front is the front of the range, not a place.
        if matches!(self.places.last(), Some(Place::Hole(_))) {
            self.places.pop();
        }
        true
    }

    /// `key` is no longer held, mandatory or not: it leaves as
    /// [`Self::leave`] has a section leave.
    pub fn forget(&mut self, key: SectionKey) -> bool {
        for p in &mut self.places {
            if let Place::Section(e) = p {
                if e.key == key {
                    e.mandatory = false;
                }
            }
        }
        self.leave(key)
    }

    /// Close every hole: each section moves up toward the top, in order.
    pub fn compact(&mut self) {
        self.places.retain(|p| matches!(p, Place::Section(_)));
    }

    /// The range from its bottom to its top, as the prompt holds it: the
    /// front's empty positions as a hole, then each place from the front to
    /// the top.
    pub fn layout(&self) -> Vec<Place> {
        let mut out = Vec::with_capacity(self.places.len() + 1);
        let front = self.positions.saturating_sub(self.used());
        if front > 0 {
            out.push(Place::Hole(front));
        }
        out.extend(self.places.iter().rev().copied());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(s: &RangeScope) -> Vec<u64> {
        s.sections().map(|e| e.key.0).collect()
    }

    fn shape(s: &RangeScope) -> Vec<(char, usize)> {
        s.layout()
            .iter()
            .map(|p| match p {
                Place::Section(e) => (char::from(b'a' + e.key.0 as u8), e.tokens),
                Place::Hole(n) => ('_', *n),
            })
            .collect()
    }

    /// **The first entry takes the top, each after it enters in front, and
    /// nothing placed moves**: a (30) at the top, b (20) below it, c (10)
    /// below b, the front 40 positions of the 100 a hole.
    #[test]
    fn each_entry_enters_in_front_and_nothing_placed_moves() {
        let mut s = RangeScope::new(100);
        assert_eq!(s.enter(SectionKey(0), 30, false), Some(vec![]));
        assert_eq!(s.enter(SectionKey(1), 20, false), Some(vec![]));
        assert_eq!(s.enter(SectionKey(2), 10, false), Some(vec![]));
        assert_eq!(shape(&s), vec![('_', 40), ('c', 10), ('b', 20), ('a', 30)]);
        assert_eq!(s.used(), 60);
        assert_eq!(s.enter(SectionKey(1), 20, false), None, "already placed");
    }

    /// **A section that leaves leaves a hole where it stood**, merged with a
    /// hole beside it; one at the front returns to the front.
    #[test]
    fn a_section_that_leaves_leaves_a_hole() {
        let mut s = RangeScope::new(100);
        for (k, t) in [(0, 30), (1, 20), (2, 10), (3, 5)] {
            s.enter(SectionKey(k), t, false);
        }
        assert!(s.leave(SectionKey(1)));
        assert_eq!(
            shape(&s),
            vec![('_', 35), ('d', 5), ('c', 10), ('_', 20), ('a', 30)]
        );
        assert!(s.leave(SectionKey(2)));
        assert_eq!(shape(&s), vec![('_', 35), ('d', 5), ('_', 30), ('a', 30)]);
        assert!(s.leave(SectionKey(3)));
        assert_eq!(
            shape(&s),
            vec![('_', 70), ('a', 30)],
            "the front is no place"
        );
        assert!(!s.leave(SectionKey(9)));
    }

    /// **A full range slides**: the entry enters in front and the
    /// earliest-entered places that may go pass the top — b, then the hole c
    /// left — the run moving up by what they freed; a, mandatory, stays.
    #[test]
    fn a_full_range_slides_and_a_mandatory_section_stays() {
        let mut s = RangeScope::new(100);
        s.enter(SectionKey(0), 30, true);
        s.enter(SectionKey(1), 20, false);
        s.enter(SectionKey(2), 30, false);
        s.enter(SectionKey(3), 20, false);
        s.leave(SectionKey(2));
        assert_eq!(s.used(), 100);
        // 25 more: b (20) and then the 30-position hole pass the top.
        assert_eq!(s.enter(SectionKey(4), 25, false), Some(vec![SectionKey(1)]));
        assert_eq!(shape(&s), vec![('_', 25), ('e', 25), ('d', 20), ('a', 30)]);
        // 40 more: d passes the top; a, mandatory, stays.
        assert_eq!(s.enter(SectionKey(5), 40, false), Some(vec![SectionKey(3)]));
        assert_eq!(shape(&s), vec![('_', 5), ('f', 40), ('e', 25), ('a', 30)]);
        // Nothing fits that would need the mandatory section's room.
        assert_eq!(s.enter(SectionKey(6), 71, false), None);
        assert_eq!(keys(&s), vec![0, 4, 5]);
        assert!(!s.leave(SectionKey(0)), "a mandatory section never leaves");
        assert!(s.forget(SectionKey(0)), "unless it is no longer held");
        assert_eq!(keys(&s), vec![4, 5]);
    }

    /// **Compaction closes every hole**, the sections keeping their order.
    #[test]
    fn compaction_closes_every_hole() {
        let mut s = RangeScope::new(100);
        for (k, t) in [(0, 30), (1, 20), (2, 10)] {
            s.enter(SectionKey(k), t, false);
        }
        s.leave(SectionKey(1));
        s.compact();
        assert_eq!(shape(&s), vec![('_', 60), ('c', 10), ('a', 30)]);
    }
}
