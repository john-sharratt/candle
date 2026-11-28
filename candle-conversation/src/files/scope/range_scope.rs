//! A reserved range's places, top first (`docs/placed_section_layout.md`
//! §3.2).
//!
//! The top of the range is nearest the question, and the range is held in
//! relevance order: the strongest at the top, each weaker one further from
//! the question ([`RangeScope::reorder`], by each section's momentum). A
//! section enters **in front** — immediately below the lowest placed — and the
//! next reorder takes it to its rank. A section that leaves leaves a hole
//! where it stood, closed by the next reorder. A full range makes room from
//! its front: the places furthest from the question that may go, holes and
//! sections alike, leave. A mandatory section never leaves, but is ordered
//! like any other.

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
    /// and return the sections a full range let go from its front to make
    /// room, furthest from the question first. `None`, and nothing changed,
    /// when it cannot fit even with every place that may go gone, or it is
    /// already placed.
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
                .rposition(|p| p.section().is_none_or(|e| !e.mandatory))
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

    /// Enter `key`, of `tokens` positions and score `value`, at its rank by
    /// `score`, and return the sections it pushed out, furthest from the
    /// question first. A full range makes room from its front, but only past
    /// what scores under `value`: holes and weaker sections go, a mandatory
    /// section or one at least as strong stops it. `None`, and nothing
    /// changed, when it cannot make room so, or it is already placed.
    pub fn enter_over(
        &mut self,
        key: SectionKey,
        tokens: usize,
        value: f32,
        score: impl Fn(SectionKey) -> Option<f32>,
    ) -> Option<Vec<SectionKey>> {
        if self.holds(key) {
            return None;
        }
        let mut free = self.positions.saturating_sub(self.used());
        let mut cut = self.places.len();
        while tokens > free {
            let at = cut.checked_sub(1)?;
            let p = &self.places[at];
            if let Some(e) = p.section() {
                if e.mandatory || score(e.key).unwrap_or(0.0) >= value {
                    return None;
                }
            }
            free += p.positions();
            cut = at;
        }
        let out: Vec<SectionKey> = self
            .places
            .drain(cut..)
            .rev()
            .filter_map(|p| p.section().map(|e| e.key))
            .collect();
        self.places.push(Place::Section(Entry {
            key,
            tokens,
            mandatory: false,
        }));
        let rank = |k: SectionKey| if k == key { Some(value) } else { score(k) };
        self.reorder(rank, 0.0);
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

    /// Order the range by `score`, strongest at the top, and close its holes;
    /// whether anything moved. A section passes the one above it only when its
    /// score beats that one's by more than `margin`, so two a hair apart keep
    /// their places; a section with no score yet counts as 0. Mandatory
    /// sections are ordered like any other.
    pub fn reorder(&mut self, score: impl Fn(SectionKey) -> Option<f32>, margin: f32) -> bool {
        self.reorder_locked(score, margin, |_| false)
    }

    /// [`Self::reorder`] with `locked` sections held: nothing passes a locked
    /// section, so one holds its place unless it climbs itself, past the
    /// unlocked ones above it. A section the index is reading stays where it
    /// is read, and only one it has stopped reading sinks.
    pub fn reorder_locked(
        &mut self,
        score: impl Fn(SectionKey) -> Option<f32>,
        margin: f32,
        locked: impl Fn(SectionKey) -> bool,
    ) -> bool {
        let before = self.places.clone();
        self.compact();
        let s = |p: &Place| p.section().map_or(0.0, |e| score(e.key).unwrap_or(0.0));
        let held = |p: &Place| p.section().is_some_and(|e| locked(e.key));
        for i in 1..self.places.len() {
            let mut j = i;
            while j > 0
                && !held(&self.places[j - 1])
                && s(&self.places[j]) > s(&self.places[j - 1]) + margin
            {
                self.places.swap(j, j - 1);
                j -= 1;
            }
        }
        self.places != before
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

    /// **A full range makes room from its front**: the places furthest from
    /// the question that may go leave — d, then the hole c left — and the
    /// entry takes the front; a, mandatory, stays.
    #[test]
    fn a_full_range_makes_room_from_its_front() {
        let mut s = RangeScope::new(100);
        s.enter(SectionKey(0), 30, true);
        s.enter(SectionKey(1), 20, false);
        s.enter(SectionKey(2), 30, false);
        s.enter(SectionKey(3), 20, false);
        s.leave(SectionKey(2));
        assert_eq!(s.used(), 100);
        // 25 more: d (20) at the front, then the 30-position hole, go.
        assert_eq!(s.enter(SectionKey(4), 25, false), Some(vec![SectionKey(3)]));
        assert_eq!(shape(&s), vec![('_', 25), ('e', 25), ('b', 20), ('a', 30)]);
        // 40 more: e, now the front, goes; a, mandatory, stays.
        assert_eq!(s.enter(SectionKey(5), 40, false), Some(vec![SectionKey(4)]));
        assert_eq!(shape(&s), vec![('_', 10), ('f', 40), ('b', 20), ('a', 30)]);
        // Nothing fits that would need the mandatory section's room.
        assert_eq!(s.enter(SectionKey(6), 71, false), None);
        assert_eq!(keys(&s), vec![0, 1, 5]);
        assert!(!s.leave(SectionKey(0)), "a mandatory section never leaves");
        assert!(s.forget(SectionKey(0)), "unless it is no longer held");
        assert_eq!(keys(&s), vec![1, 5]);
    }

    /// **The range is ordered strongest at the top, the mandatory section
    /// with the rest, its holes closed**: a (mandatory, 100), b (900), c
    /// (500), with a hole where d left, reorder to b, c, a.
    #[test]
    fn the_range_is_ordered_strongest_at_the_top() {
        let mut s = RangeScope::new(100);
        s.enter(SectionKey(0), 10, true);
        s.enter(SectionKey(1), 10, false);
        s.enter(SectionKey(3), 10, false);
        s.enter(SectionKey(2), 10, false);
        s.leave(SectionKey(3));
        let score = |k: SectionKey| match k.0 {
            0 => Some(100.0),
            1 => Some(900.0),
            2 => Some(500.0),
            _ => None,
        };
        assert!(s.reorder(score, 50.0));
        assert_eq!(keys(&s), vec![1, 2, 0]);
        assert_eq!(shape(&s), vec![('_', 70), ('a', 10), ('c', 10), ('b', 10)]);
        assert!(!s.reorder(score, 50.0), "an ordered range does not move");
    }

    /// **A section passes the one above it only by more than the margin**:
    /// b at 540 stays under a at 500 with a margin of 50, and passes it at
    /// 551; one with no score yet counts as 0 and sinks below a scored one.
    #[test]
    fn a_section_passes_only_by_more_than_the_margin() {
        let mut s = RangeScope::new(100);
        s.enter(SectionKey(0), 10, false);
        s.enter(SectionKey(1), 10, false);
        s.enter(SectionKey(2), 10, false);
        let near = |k: SectionKey| match k.0 {
            0 => Some(500.0),
            1 => Some(540.0),
            _ => Some(0.0),
        };
        assert!(!s.reorder(near, 50.0));
        assert_eq!(keys(&s), vec![0, 1, 2]);
        let past = |k: SectionKey| match k.0 {
            0 => Some(500.0),
            1 => Some(551.0),
            2 => Some(60.0),
            _ => None,
        };
        assert!(s.reorder(past, 50.0));
        assert_eq!(keys(&s), vec![1, 0, 2]);
        let unscored = |k: SectionKey| match k.0 {
            0 => None,
            _ => Some(100.0),
        };
        assert!(s.reorder(unscored, 50.0));
        assert_eq!(keys(&s), vec![1, 2, 0]);
    }

    /// **Nothing passes a locked section, and every section climbs only past
    /// unlocked ones**: top first a (300, locked), b (200), c (500, locked),
    /// d (900). c climbs past b but not past a; d past b, up to c. Nothing
    /// moves again while the locks hold. a unlocked, c climbs past it and d
    /// follows, up to c again.
    #[test]
    fn nothing_passes_a_locked_section() {
        let mut s = RangeScope::new(100);
        for k in 0..4 {
            s.enter(SectionKey(k), 10, false);
        }
        assert_eq!(keys(&s), vec![0, 1, 2, 3]);
        let score = |k: SectionKey| match k.0 {
            0 => Some(300.0),
            1 => Some(200.0),
            2 => Some(500.0),
            _ => Some(900.0),
        };
        let lock_ac = |k: SectionKey| k.0 == 0 || k.0 == 2;
        assert!(s.reorder_locked(score, 50.0, lock_ac));
        assert_eq!(keys(&s), vec![0, 2, 3, 1]);
        assert!(
            !s.reorder_locked(score, 50.0, lock_ac),
            "nothing moves while the locks hold"
        );
        let lock_c = |k: SectionKey| k.0 == 2;
        assert!(s.reorder_locked(score, 50.0, lock_c));
        assert_eq!(keys(&s), vec![2, 3, 0, 1]);
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
