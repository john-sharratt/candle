//! Where a reprojection placed each turn and section, held until the walk ends
//! and written back to the substrate under one write lock.
//!
//! Writing each range as its piece was placed took a write lock per turn —
//! forty-odd a rebuild, each queued behind whatever reader held the substrate.
//! Held here instead, they are written once the walk ends ([`PendingRanges::flush`])
//! — and on the way out of a walk that stops early with an error too, by
//! [`Drop`]: the slot already holds the pieces placed before the error, and
//! the substrate must say where they are, as it did when each was written at once.

use crate::projection::{Conversation, SectionId, TimelineId, TurnIndex};

/// A turn's placement: its timeline, index, and `[start, end)` blocks.
pub(super) type TurnRange = (TimelineId, TurnIndex, u64, u64);

/// A section's placement: its id and `[start, end)` blocks.
pub(super) type SectionRange = (SectionId, u64, u64);

/// Where placements are written: the substrate, in production.
pub(super) trait RangeSink {
    /// Write every placement, under one lock.
    fn write_ranges(&self, turns: &[TurnRange], sections: &[SectionRange]);
}

impl RangeSink for Conversation {
    fn write_ranges(&self, turns: &[TurnRange], sections: &[SectionRange]) {
        let mut substrate = self.write();
        for &(timeline, index, start, end) in turns {
            substrate.set_block_range(timeline, index, start, end);
        }
        for &(sid, start, end) in sections {
            substrate.set_section_block_range(sid, start, end);
        }
    }
}

/// Placements recorded during one walk and not yet written.
pub(super) struct PendingRanges<'c, S: RangeSink + ?Sized> {
    sink: &'c S,
    turns: Vec<TurnRange>,
    sections: Vec<SectionRange>,
}

impl<'c, S: RangeSink + ?Sized> PendingRanges<'c, S> {
    pub(super) fn new(sink: &'c S) -> Self {
        Self {
            sink,
            turns: Vec::new(),
            sections: Vec::new(),
        }
    }

    pub(super) fn turn(&mut self, range: TurnRange) {
        self.turns.push(range);
    }

    pub(super) fn section(&mut self, range: SectionRange) {
        self.sections.push(range);
    }

    /// Write what is recorded, under one lock, and forget it. Nothing
    /// recorded takes no lock.
    pub(super) fn flush(&mut self) {
        if self.turns.is_empty() && self.sections.is_empty() {
            return;
        }
        self.sink.write_ranges(&self.turns, &self.sections);
        self.turns.clear();
        self.sections.clear();
    }
}

impl<S: RangeSink + ?Sized> Drop for PendingRanges<'_, S> {
    fn drop(&mut self) {
        self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;

    /// Every write the sink was asked for, in order.
    #[derive(Default)]
    struct Recorder {
        writes: RefCell<Vec<(Vec<TurnRange>, Vec<SectionRange>)>>,
    }

    impl RangeSink for Recorder {
        fn write_ranges(&self, turns: &[TurnRange], sections: &[SectionRange]) {
            self.writes
                .borrow_mut()
                .push((turns.to_vec(), sections.to_vec()));
        }
    }

    fn turn(index: u32, start: u64) -> TurnRange {
        (
            TimelineId::from_raw(7).expect("non-zero"),
            TurnIndex(index),
            start,
            start + 4,
        )
    }

    /// **One write for the whole walk**, and nothing written twice: a flush
    /// empties what the drop would have written.
    #[test]
    fn a_walk_writes_its_ranges_once() {
        let sink = Recorder::default();
        {
            let mut pending = PendingRanges::new(&sink);
            pending.turn(turn(0, 0));
            pending.turn(turn(1, 4));
            pending.section((SectionId::new(3), 8, 9));
            pending.flush();
        }
        let writes = sink.writes.borrow();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].0, vec![turn(0, 0), turn(1, 4)]);
        assert_eq!(writes[0].1, vec![(SectionId::new(3), 8, 9)]);
    }

    /// **A walk that stops early still writes what it placed**: the drop
    /// writes the ranges no flush reached.
    #[test]
    fn a_walk_cut_short_writes_what_it_placed() {
        let sink = Recorder::default();
        let cut_short = || -> Result<(), ()> {
            let mut pending = PendingRanges::new(&sink);
            pending.turn(turn(0, 0));
            Err(())?;
            pending.turn(turn(1, 4));
            pending.flush();
            Ok(())
        };
        assert_eq!(cut_short(), Err(()));
        assert_eq!(sink.writes.borrow().as_slice(), &[(vec![turn(0, 0)], vec![])]);
    }

    /// A walk that placed nothing takes no lock.
    #[test]
    fn nothing_placed_writes_nothing() {
        let sink = Recorder::default();
        PendingRanges::new(&sink).flush();
        drop(PendingRanges::new(&sink));
        assert!(sink.writes.borrow().is_empty());
    }
}
