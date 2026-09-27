//! Where every live file event and every tombstone of each conversation is.
//!
//! The persistence layer's twin of `npc_locs` for a conversation's file
//! events: locations only — no body is ever held — keyed by timeline and
//! sequence number, which the record header carries. Built by the open walk,
//! kept current by every append, rebuilt after compaction; compaction and
//! maintenance read it to know what to carry, and a resume reads the events
//! it points at.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::persistence::manifest::RecordLoc;

/// One conversation's file events and tombstones.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TimelineVfs {
    /// Live events, by sequence number.
    events: BTreeMap<u64, RecordLoc>,
    /// Tombstones, by their own sequence number.
    tombstones: BTreeMap<u64, RecordLoc>,
    /// Sequence numbers a tombstone killed — so an event met after the
    /// tombstone that killed it, in log order, stays dead.
    killed: HashSet<u64>,
    /// The highest sequence number seen, live or not.
    max_seq: Option<u64>,
}

impl TimelineVfs {
    pub fn events(&self) -> &BTreeMap<u64, RecordLoc> {
        &self.events
    }

    pub fn tombstones(&self) -> &BTreeMap<u64, RecordLoc> {
        &self.tombstones
    }

    fn saw(&mut self, seq: u64) {
        self.max_seq = Some(self.max_seq.map_or(seq, |m| m.max(seq)));
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

    /// An event of `timeline` at `seq` is at `loc`. Returns whether it is
    /// live: not killed by a tombstone, and its timeline not tombstoned.
    pub fn record_event(&mut self, timeline: u64, seq: u64, loc: RecordLoc) -> bool {
        if self.dead.contains(&timeline) {
            return false;
        }
        let tl = self.timelines.entry(timeline).or_default();
        tl.saw(seq);
        if tl.killed.contains(&seq) {
            return false;
        }
        tl.events.insert(seq, loc);
        true
    }

    /// A tombstone of `timeline`, its own sequence number `seq`, at `loc`,
    /// killing `kills`. Returns the sequence numbers it took from the live
    /// set — the events whose bytes are dead now.
    pub fn record_tombstone(
        &mut self,
        timeline: u64,
        seq: u64,
        loc: RecordLoc,
        kills: &[u64],
    ) -> Vec<u64> {
        if self.dead.contains(&timeline) {
            return Vec::new();
        }
        let tl = self.timelines.entry(timeline).or_default();
        tl.saw(seq);
        tl.tombstones.insert(seq, loc);
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
    /// and the tombstones' sequence numbers, whose bytes are dead now.
    pub fn drop_timeline(&mut self, timeline: u64) -> (Vec<u64>, Vec<u64>) {
        self.dead.insert(timeline);
        match self.timelines.remove(&timeline) {
            Some(tl) => (
                tl.events.into_keys().collect(),
                tl.tombstones.into_keys().collect(),
            ),
            None => (Vec::new(), Vec::new()),
        }
    }

    /// The sequence number `timeline`'s next record takes: past every one
    /// seen.
    pub fn next_seq(&self, timeline: u64) -> u64 {
        self.timelines
            .get(&timeline)
            .and_then(|tl| tl.max_seq)
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

    /// Move the event at `seq` from `old` to `new`, if it is still at `old`.
    pub fn repoint_event(&mut self, timeline: u64, seq: u64, old: RecordLoc, new: RecordLoc) {
        if let Some(loc) = self
            .timelines
            .get_mut(&timeline)
            .and_then(|tl| tl.events.get_mut(&seq))
        {
            if *loc == old {
                *loc = new;
            }
        }
    }

    /// Move the tombstone at `seq` from `old` to `new`, if it is still there.
    pub fn repoint_tombstone(&mut self, timeline: u64, seq: u64, old: RecordLoc, new: RecordLoc) {
        if let Some(loc) = self
            .timelines
            .get_mut(&timeline)
            .and_then(|tl| tl.tombstones.get_mut(&seq))
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

    fn at(offset: u64) -> RecordLoc {
        RecordLoc {
            segment: FIRST_SEGMENT,
            offset,
            payload_len: 10,
            record_size: 4096,
        }
    }

    /// **A tombstone kills the events it names**, and only those; an event
    /// met after its tombstone in log order stays dead.
    #[test]
    fn a_tombstone_kills_what_it_names() {
        let mut index = VfsIndex::new();
        assert!(index.record_event(7, 0, at(0)));
        assert!(index.record_event(7, 1, at(4096)));
        assert!(index.record_event(7, 2, at(8192)));
        assert_eq!(
            index.record_tombstone(7, 3, at(12288), &[0, 1, 5]),
            vec![0, 1]
        );
        let tl = index.timeline(7).unwrap();
        assert_eq!(tl.events().keys().copied().collect::<Vec<_>>(), vec![2]);
        assert_eq!(tl.tombstones().keys().copied().collect::<Vec<_>>(), vec![3]);
        assert!(
            !index.record_event(7, 5, at(16384)),
            "killed before it was met"
        );
        assert_eq!(index.next_seq(7), 6);
    }

    /// **A tombstoned timeline takes everything with it**, and nothing of
    /// its met later is live.
    #[test]
    fn a_tombstoned_timeline_takes_everything() {
        let mut index = VfsIndex::new();
        index.record_event(7, 0, at(0));
        index.record_event(8, 0, at(4096));
        index.record_tombstone(7, 1, at(8192), &[]);
        assert_eq!(index.drop_timeline(7), (vec![0], vec![1]));
        assert!(index.timeline(7).is_none());
        assert!(!index.record_event(7, 2, at(12288)));
        assert!(index.record_tombstone(7, 3, at(16384), &[2]).is_empty());
        assert!(index.timeline(7).is_none());
        assert_eq!(index.timelines().len(), 1, "8 is untouched");
        assert_eq!(index.next_seq(9), 0, "a timeline with nothing starts at 0");
    }

    /// **A relocation moves a record only if it is still where the plan saw
    /// it.**
    #[test]
    fn a_repoint_needs_the_old_location() {
        let mut index = VfsIndex::new();
        index.record_event(7, 0, at(0));
        index.record_tombstone(7, 1, at(4096), &[]);
        index.repoint_event(7, 0, at(8192), at(12288));
        assert_eq!(index.timeline(7).unwrap().events()[&0], at(0), "stale plan");
        index.repoint_event(7, 0, at(0), at(12288));
        index.repoint_tombstone(7, 1, at(4096), at(16384));
        let tl = index.timeline(7).unwrap();
        assert_eq!(tl.events()[&0], at(12288));
        assert_eq!(tl.tombstones()[&1], at(16384));
        assert_eq!(index.locations().count(), 2);
    }
}
