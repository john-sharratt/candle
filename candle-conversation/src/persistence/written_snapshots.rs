//! The recurrent snapshots the writer has appended, handed to whoever releases
//! state once its record is durable.
//!
//! A recurrent model's slot holds a turn's state on the device until the
//! record of that state is on disk; from then on the record is the copy, and
//! the slot's state can go — restored from the record when the slot is next
//! used. This queue is how the slot's owner learns a record is down: the
//! writer pushes each snapshot it appended, and the owner takes those whose
//! location is durable ([`super::sealed_reader::SealedReader::is_durable`]).
//!
//! Lock-free on both sides: the writer pushes from its own thread, and the
//! taker — the scheduler, between waves — pops without waiting on the writer,
//! the persistence lock or the substrate. A snapshot appended but not yet
//! committed is put back for the next take.

use crossbeam::queue::SegQueue;

use super::manifest::RecordLoc;

/// Entries held for a taker that never takes — a host whose model parks no
/// state. Past this the oldest goes: it is long durable, and losing it only
/// means a slot keeps its state until the slot is released.
pub const WRITTEN_SNAPSHOTS_HELD: usize = 4096;

/// One snapshot the writer appended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrittenSnapshot {
    /// The raw id of the timeline whose state it is.
    pub timeline: u64,
    /// The turn it was taken at.
    pub turn_index: u32,
    /// Where the writer put it.
    pub loc: RecordLoc,
}

/// The queue between the writer and the taker.
#[derive(Debug, Default)]
pub struct WrittenSnapshots(SegQueue<WrittenSnapshot>);

impl WrittenSnapshots {
    /// Record a snapshot the writer appended.
    pub fn push(&self, written: WrittenSnapshot) {
        while self.0.len() >= WRITTEN_SNAPSHOTS_HELD {
            self.0.pop();
        }
        self.0.push(written);
    }

    /// Every queued snapshot whose record `durable` says is down, as
    /// `(timeline, turn_index)`; the rest stay queued. Takes only what was
    /// queued when it started, so a writer pushing meanwhile cannot keep it
    /// going.
    pub fn take_durable(&self, durable: impl Fn(&RecordLoc) -> bool) -> Vec<(u64, u32)> {
        let mut taken = Vec::new();
        for _ in 0..self.0.len() {
            let Some(w) = self.0.pop() else {
                break;
            };
            if durable(&w.loc) {
                taken.push((w.timeline, w.turn_index));
            } else {
                self.0.push(w);
            }
        }
        taken
    }

    /// How many are queued.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether none are queued.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::segment::SegmentId;

    fn written(timeline: u64, turn_index: u32, offset: u64) -> WrittenSnapshot {
        WrittenSnapshot {
            timeline,
            turn_index,
            loc: RecordLoc {
                segment: SegmentId(1),
                offset,
                payload_len: 8,
                record_size: 64,
            },
        }
    }

    /// **Only what is durable is taken; the rest waits for the next take.**
    /// Durable here is "ends at or below the committed end", the rule the
    /// sealed reader applies.
    #[test]
    fn only_durable_snapshots_are_taken() {
        let q = WrittenSnapshots::default();
        q.push(written(7, 0, 0));
        q.push(written(8, 2, 1000));
        let committed_end = 64;
        let durable = |loc: &RecordLoc| loc.offset + loc.record_size <= committed_end;
        assert_eq!(q.take_durable(durable), vec![(7, 0)]);
        assert_eq!(q.len(), 1, "the uncommitted one stays queued");
        assert_eq!(q.take_durable(|_| true), vec![(8, 2)]);
        assert!(q.is_empty());
    }

    /// A queue nobody takes from holds at most its bound, dropping the oldest.
    #[test]
    fn an_untaken_queue_is_bounded() {
        let q = WrittenSnapshots::default();
        for i in 0..(WRITTEN_SNAPSHOTS_HELD as u64 + 10) {
            q.push(written(i, 0, 0));
        }
        assert_eq!(q.len(), WRITTEN_SNAPSHOTS_HELD);
        let taken = q.take_durable(|_| true);
        assert_eq!(taken.first(), Some(&(10, 0)), "the oldest went first");
    }
}
