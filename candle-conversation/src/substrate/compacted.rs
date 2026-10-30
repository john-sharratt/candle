//! The substrate's side of a full rewrite of the log — what
//! [`crate::persistence::rewritten::Rewritten::settle`] asks of it once the
//! rewritten log's records are re-applied.

use std::collections::HashSet;

use crate::persistence::identity::RecordIdentity;
use crate::persistence::segment::SegmentId;

use super::stream_index::StreamKey;
use super::Substrate;

impl Substrate {
    /// Drop every location into `old` — segments a rewrite replaced. After the
    /// re-apply, a location still into one of them is a record the rewrite did
    /// not carry: a branch checkpoint over its cap, a distilled timeline's
    /// snapshot, a tombstoned stream's chunks.
    pub fn drop_locations_into(&mut self, old: &HashSet<SegmentId>) {
        self.streams.drop_segments(old);
        let snapshots: Vec<_> = self
            .recurrent_snapshots
            .iter()
            .filter(|(_, l)| old.contains(&l.segment))
            .map(|(sid, _)| *sid)
            .collect();
        for sid in snapshots {
            self.recurrent_snapshots.remove(&sid);
            self.snapshot_ids.remove(&sid);
        }
        let branches: Vec<_> = self
            .branch_checkpoints
            .iter()
            .filter(|(_, l)| old.contains(&l.segment))
            .map(|(sid, _)| *sid)
            .collect();
        for sid in branches {
            self.branch_checkpoints.remove(&sid);
            self.branch_checkpoint_ids.remove(&sid);
        }
        self.gallery_epoch += 1;
    }

    /// Forget the stream at `key`, which a rewrite dropped wholesale, if it is
    /// still declared by the record `decl` the rewrite read. One declared again
    /// since — by a record the rewrite never saw — stands.
    pub fn forget_dropped_stream(&mut self, key: StreamKey, decl: RecordIdentity) {
        let held = self.streams.physical(key).map(|rt| rt.ids.decl);
        if held != Some(decl) {
            return;
        }
        self.streams.remove_writer(key);
        self.tagged_turns.remove(key);
        self.gallery_epoch += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::manifest::RecordLoc;
    use crate::persistence::streams::StreamId;

    fn loc(segment: u64, offset: u64) -> RecordLoc {
        RecordLoc {
            segment: SegmentId(segment),
            offset,
            payload_len: 8,
            record_size: 4096,
        }
    }

    /// **A location into a replaced segment goes; one into the new segment
    /// stays.**
    #[test]
    fn only_locations_into_replaced_segments_are_dropped() {
        let mut s = Substrate::new();
        s.apply_tokens_loc(StreamId(1), loc(3, 0), RecordIdentity::NONE);
        s.apply_tokens_loc(StreamId(2), loc(9, 0), RecordIdentity::NONE);
        s.apply_snapshot_loc(StreamId(5), loc(3, 4096), RecordIdentity::NONE);
        s.apply_snapshot_loc(StreamId(6), loc(9, 4096), RecordIdentity::NONE);
        let old: HashSet<SegmentId> = [SegmentId(3)].into_iter().collect();
        s.drop_locations_into(&old);
        assert_eq!(s.stream_of(StreamId(1)).and_then(|r| r.tokens), None);
        assert_eq!(
            s.stream_of(StreamId(2)).and_then(|r| r.tokens),
            Some(loc(9, 0))
        );
        assert_eq!(s.recurrent_snapshot_loc(StreamId(5)), None);
        assert_eq!(s.recurrent_snapshot_loc(StreamId(6)), Some(loc(9, 4096)));
    }
}
