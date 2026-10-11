//! Where every live file event and every tombstone of each conversation is.
//!
//! The persistence layer's twin of `npc_locs` for a conversation's file
//! events: locations only — no body is ever held — keyed by timeline, writer
//! and sequence number, which the record header carries. Each member numbers
//! the events it writes for a timeline in a sequence of its own, so two
//! members writing one conversation's files never name the same event
//! (`docs/replicated_substrate.md` §6.2). Built by the open walk, kept current
//! by every append, rebuilt after compaction; compaction and maintenance read
//! it to know what to carry, and a resume reads the events it points at.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use super::payload::VfsEventId;
use crate::persistence::manifest::RecordLoc;

/// One conversation's file events and tombstones.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TimelineVfs {
    /// Live events.
    events: BTreeMap<VfsEventId, RecordLoc>,
    /// Tombstones, by their own name.
    tombstones: BTreeMap<VfsEventId, RecordLoc>,
    /// Events a tombstone killed — so an event met after the tombstone that
    /// killed it, in log order, stays dead.
    killed: HashSet<VfsEventId>,
    /// The writers of the events each tombstone kills.
    killed_writers: HashMap<VfsEventId, BTreeSet<u64>>,
    /// Each writer's highest sequence number seen, live or not.
    max_seq: HashMap<u64, u64>,
}

impl TimelineVfs {
    pub fn events(&self) -> &BTreeMap<VfsEventId, RecordLoc> {
        &self.events
    }

    pub fn tombstones(&self) -> &BTreeMap<VfsEventId, RecordLoc> {
        &self.tombstones
    }

    /// The tombstones a rewrite must keep on a member whose node is `node`:
    /// each that kills an event another member wrote. Another member's event
    /// can arrive after its tombstone, in any order and however late, and
    /// must still die (`docs/replicated_substrate.md` §6.5); this member's own
    /// events were all here before it wrote, or applied, anything that kills
    /// them.
    pub fn kept_tombstones(&self, node: u64) -> impl Iterator<Item = (&VfsEventId, &RecordLoc)> {
        self.tombstones.iter().filter(move |(id, _)| {
            self.killed_writers
                .get(id)
                .is_some_and(|writers| writers.iter().any(|&w| w != node))
        })
    }

    fn saw(&mut self, id: VfsEventId) {
        let max = self.max_seq.entry(id.origin).or_insert(id.seq);
        *max = (*max).max(id.seq);
    }
}

/// Every conversation's file events and tombstones. See the module.
#[derive(Debug, Default)]
pub struct VfsIndex {
    timelines: HashMap<u64, TimelineVfs>,
    /// Timelines whose `Tombstone` has been seen: nothing of theirs is live,
    /// whatever comes after it in the log.
    dead: HashSet<u64>,
}

impl VfsIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Event `id` of `timeline` is at `loc`. Returns whether it is live: not
    /// killed by a tombstone, and its timeline not tombstoned.
    pub fn record_event(&mut self, timeline: u64, id: VfsEventId, loc: RecordLoc) -> bool {
        if self.dead.contains(&timeline) {
            return false;
        }
        let tl = self.timelines.entry(timeline).or_default();
        tl.saw(id);
        if tl.killed.contains(&id) {
            return false;
        }
        tl.events.insert(id, loc);
        true
    }

    /// Tombstone `id` of `timeline`, at `loc`, killing `kills`. Returns the
    /// events it took from the live set — whose bytes are dead now.
    pub fn record_tombstone(
        &mut self,
        timeline: u64,
        id: VfsEventId,
        loc: RecordLoc,
        kills: &[VfsEventId],
    ) -> Vec<VfsEventId> {
        if self.dead.contains(&timeline) {
            return Vec::new();
        }
        let tl = self.timelines.entry(timeline).or_default();
        tl.saw(id);
        tl.tombstones.insert(id, loc);
        tl.killed_writers
            .insert(id, kills.iter().map(|k| k.origin).collect());
        let mut retired = Vec::new();
        for &k in kills {
            tl.killed.insert(k);
            if tl.events.remove(&k).is_some() {
                retired.push(k);
            }
        }
        retired
    }

    /// `timeline` is tombstoned: everything of its goes. Returns the events'
    /// and the tombstones' names, whose bytes are dead now.
    pub fn drop_timeline(&mut self, timeline: u64) -> (Vec<VfsEventId>, Vec<VfsEventId>) {
        self.dead.insert(timeline);
        match self.timelines.remove(&timeline) {
            Some(tl) => (
                tl.events.into_keys().collect(),
                tl.tombstones.into_keys().collect(),
            ),
            None => (Vec::new(), Vec::new()),
        }
    }

    /// The sequence number `origin`'s next record for `timeline` takes: past
    /// every one of its own seen.
    pub fn next_seq(&self, timeline: u64, origin: u64) -> u64 {
        self.timelines
            .get(&timeline)
            .and_then(|tl| tl.max_seq.get(&origin))
            .map_or(0, |m| m + 1)
    }

    pub fn timeline(&self, timeline: u64) -> Option<&TimelineVfs> {
        self.timelines.get(&timeline)
    }

    /// Every timeline with events or tombstones, in id order.
    pub fn timelines(&self) -> Vec<(u64, &TimelineVfs)> {
        let mut out: Vec<(u64, &TimelineVfs)> =
            self.timelines.iter().map(|(&id, tl)| (id, tl)).collect();
        out.sort_unstable_by_key(|(id, _)| *id);
        out
    }

    /// Every location recorded, events and tombstones alike.
    pub fn locations(&self) -> impl Iterator<Item = &RecordLoc> {
        self.timelines
            .values()
            .flat_map(|tl| tl.events.values().chain(tl.tombstones.values()))
    }

    /// Move event `id` from `old` to `new`, if it is still at `old`.
    pub fn repoint_event(&mut self, timeline: u64, id: VfsEventId, old: RecordLoc, new: RecordLoc) {
        if let Some(loc) = self
            .timelines
            .get_mut(&timeline)
            .and_then(|tl| tl.events.get_mut(&id))
        {
            if *loc == old {
                *loc = new;
            }
        }
    }

    /// Move tombstone `id` from `old` to `new`, if it is still there.
    pub fn repoint_tombstone(
        &mut self,
        timeline: u64,
        id: VfsEventId,
        old: RecordLoc,
        new: RecordLoc,
    ) {
        if let Some(loc) = self
            .timelines
            .get_mut(&timeline)
            .and_then(|tl| tl.tombstones.get_mut(&id))
        {
            if *loc == old {
                *loc = new;
            }
        }
    }

    pub fn clear(&mut self) {
        self.timelines.clear();
        self.dead.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::segment::FIRST_SEGMENT;

    const A: u64 = 0xA;
    const B: u64 = 0xB;

    fn at(offset: u64) -> RecordLoc {
        RecordLoc {
            segment: FIRST_SEGMENT,
            offset,
            payload_len: 10,
            record_size: 4096,
        }
    }

    fn ev(origin: u64, seq: u64) -> VfsEventId {
        VfsEventId { origin, seq }
    }

    /// **A tombstone kills the events it names**, and only those; an event
    /// met after its tombstone in log order stays dead.
    #[test]
    fn a_tombstone_kills_what_it_names() {
        let mut index = VfsIndex::new();
        assert!(index.record_event(7, ev(A, 0), at(0)));
        assert!(index.record_event(7, ev(A, 1), at(4096)));
        assert!(index.record_event(7, ev(A, 2), at(8192)));
        assert_eq!(
            index.record_tombstone(7, ev(A, 3), at(12288), &[ev(A, 0), ev(A, 1), ev(A, 5)]),
            vec![ev(A, 0), ev(A, 1)]
        );
        let tl = index.timeline(7).unwrap();
        assert_eq!(
            tl.events().keys().copied().collect::<Vec<_>>(),
            vec![ev(A, 2)]
        );
        assert_eq!(
            tl.tombstones().keys().copied().collect::<Vec<_>>(),
            vec![ev(A, 3)]
        );
        assert!(
            !index.record_event(7, ev(A, 5), at(16384)),
            "killed before it was met"
        );
        assert_eq!(index.next_seq(7, A), 6);
    }

    /// **Two writers number their events independently**: the same `seq`
    /// from each is two events, a tombstone kills the one it names, and each
    /// writer's next number is past only its own.
    #[test]
    fn two_writers_number_their_events_apart() {
        let mut index = VfsIndex::new();
        assert!(index.record_event(7, ev(A, 0), at(0)));
        assert!(index.record_event(7, ev(B, 0), at(4096)));
        assert!(index.record_event(7, ev(B, 1), at(8192)));
        assert_eq!(
            index.record_tombstone(7, ev(A, 1), at(12288), &[ev(B, 0)]),
            vec![ev(B, 0)]
        );
        let tl = index.timeline(7).unwrap();
        assert_eq!(
            tl.events().keys().copied().collect::<Vec<_>>(),
            vec![ev(A, 0), ev(B, 1)]
        );
        assert_eq!(index.next_seq(7, A), 2);
        assert_eq!(index.next_seq(7, B), 2);
        assert_eq!(index.next_seq(7, 0xC), 0);
    }

    /// **A tombstone that kills another member's event is kept**; one that
    /// kills only this member's own, or nothing, is not.
    #[test]
    fn a_tombstone_killing_another_writers_event_is_kept() {
        let mut index = VfsIndex::new();
        index.record_tombstone(7, ev(A, 1), at(0), &[ev(A, 0)]);
        index.record_tombstone(7, ev(A, 2), at(4096), &[ev(A, 0), ev(B, 3)]);
        index.record_tombstone(7, ev(B, 4), at(8192), &[ev(B, 3)]);
        index.record_tombstone(7, ev(A, 5), at(12288), &[]);
        let tl = index.timeline(7).unwrap();
        let kept = |node| {
            tl.kept_tombstones(node)
                .map(|(id, _)| *id)
                .collect::<Vec<_>>()
        };
        assert_eq!(kept(A), vec![ev(A, 2), ev(B, 4)]);
        assert_eq!(kept(B), vec![ev(A, 1), ev(A, 2)]);
    }

    /// **A tombstoned timeline takes everything with it**, and nothing of
    /// its met later is live.
    #[test]
    fn a_tombstoned_timeline_takes_everything() {
        let mut index = VfsIndex::new();
        index.record_event(7, ev(A, 0), at(0));
        index.record_event(8, ev(A, 0), at(4096));
        index.record_tombstone(7, ev(A, 1), at(8192), &[]);
        assert_eq!(index.drop_timeline(7), (vec![ev(A, 0)], vec![ev(A, 1)]));
        assert!(index.timeline(7).is_none());
        assert!(!index.record_event(7, ev(A, 2), at(12288)));
        assert!(index
            .record_tombstone(7, ev(A, 3), at(16384), &[ev(A, 2)])
            .is_empty());
        assert!(index.timeline(7).is_none());
        assert_eq!(index.timelines().len(), 1, "8 is untouched");
        assert_eq!(
            index.next_seq(9, A),
            0,
            "a timeline with nothing starts at 0"
        );
    }

    /// **A relocation moves a record only if it is still where the plan saw
    /// it.**
    #[test]
    fn a_repoint_needs_the_old_location() {
        let mut index = VfsIndex::new();
        index.record_event(7, ev(A, 0), at(0));
        index.record_tombstone(7, ev(A, 1), at(4096), &[]);
        index.repoint_event(7, ev(A, 0), at(8192), at(12288));
        assert_eq!(
            index.timeline(7).unwrap().events()[&ev(A, 0)],
            at(0),
            "stale plan"
        );
        index.repoint_event(7, ev(A, 0), at(0), at(12288));
        index.repoint_tombstone(7, ev(A, 1), at(4096), at(16384));
        let tl = index.timeline(7).unwrap();
        assert_eq!(tl.events()[&ev(A, 0)], at(12288));
        assert_eq!(tl.tombstones()[&ev(A, 1)], at(16384));
        assert_eq!(index.locations().count(), 2);
    }
}
