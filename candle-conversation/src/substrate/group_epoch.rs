//! Each belief group's content epoch: what an assembled group scan is stamped
//! with (`projection::group_assembly`).
//!
//! A group's scan is assembled from its conversations' turns — their windows,
//! signatures, seams and tags — and from the calibrations of its files. The
//! substrate's [`Substrate::gallery_epoch`] moves on every change to any of
//! those anywhere, so stamped with it a group was reassembled whenever any
//! conversation sealed a turn: on BlackWorld (2026-11-18) a third of a single
//! chat's reprojections reassembled every group, 140–440 ms each, though the
//! turns that moved the epoch belonged to none of them.
//!
//! So each change that moves the gallery epoch also moves the epoch of the
//! groups it can change, and no other:
//!
//! - the group of the timeline it names;
//! - the group of the file that timeline calibrates, if it is a calibration;
//! - every group that draws its candidates from one of those
//!   ([`Substrate::set_group_draws`]);
//!
//! and a change that names no timeline moves every group's. A group's epoch is
//! read as the sum of its own and that shared one, which moves whenever either
//! does.

use super::stream_index::StreamKey;
use super::Substrate;
use crate::projection::{GroupId, TimelineId};

impl Substrate {
    /// `group`'s content epoch: unchanged while nothing its scan is assembled
    /// from has changed. See the module documentation.
    pub fn group_epoch(&self, group: GroupId) -> u64 {
        self.all_groups_epoch + self.group_epochs.get(&group).copied().unwrap_or(0)
    }

    /// Something `timeline`'s turns are read for changed: the epochs of the
    /// groups that read them move.
    pub(super) fn timeline_content_changed(&mut self, timeline: TimelineId) {
        let Some(group) = self.timelines.get(&timeline).map(|e| e.group) else {
            // A timeline this version does not hold names no group: its
            // change may still be one some group reads.
            self.all_groups_content_changed();
            return;
        };
        let mut groups = vec![group];
        if let Some(unit) = self.calibrates(timeline) {
            if let Some(entry) = self.timelines.get(&unit) {
                groups.push(entry.group);
            }
        }
        let drawers: Vec<GroupId> = self
            .drawing_groups
            .iter()
            .filter(|(_, source)| groups.contains(source))
            .map(|(drawing, _)| *drawing)
            .collect();
        groups.extend(drawers);
        groups.sort_unstable();
        groups.dedup();
        for g in groups {
            let at = self.group_epochs.get(&g).copied().unwrap_or(0);
            self.group_epochs.insert(g, at + 1);
        }
    }

    /// A value on `key`'s stream changed: the groups reading its turn's
    /// timeline move, or every group's when the stream is no turn.
    pub(super) fn stream_content_changed(&mut self, key: StreamKey) {
        match self
            .streams
            .turn(key)
            .and_then(|t| TimelineId::from_raw(t.decl.timeline_id))
        {
            Some(timeline) => self.timeline_content_changed(timeline),
            None => self.all_groups_content_changed(),
        }
    }

    /// Something changed whose timeline is not known: every group's epoch
    /// moves.
    pub(super) fn all_groups_content_changed(&mut self) {
        self.all_groups_epoch += 1;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::persistence::content_hash::turn_stream_id;
    use crate::persistence::identity::RecordIdentity;
    use crate::persistence::streams::{StreamDecl, StreamId, TurnDecl};
    use crate::projection::{GroupId, LayerId, TimelineId};
    use crate::provenance::{encode_wide_sigs, WideQSig};
    use crate::substrate::Substrate;

    fn group(raw: u32) -> GroupId {
        GroupId::for_test(raw)
    }

    fn timeline(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    /// A substrate holding a turn of timeline `tl` in `group` for each pair.
    fn substrate(turns: &[(u64, u32)]) -> Substrate {
        let mut sub = Substrate::new();
        for &(tl, g) in turns {
            sub.apply_stream_decl(
                turn_stream_id(tl, 0),
                StreamDecl::Turn(TurnDecl {
                    timeline_id: tl,
                    turn_index: 0,
                    turn_id_day: 0,
                    turn_id_seq: tl as u32,
                    role: 1,
                    block_start: 0,
                    block_end: 1,
                    layer_id: 1,
                    group_id: g,
                    anchored_prefix: Vec::new(),
                    view: Vec::new(),
                    segments: Vec::new(),
                    tags: Vec::new(),
                }),
                RecordIdentity::NONE,
            );
        }
        sub
    }

    /// Write a signature on `sid`'s stream.
    fn write_sig(sub: &mut Substrate, sid: StreamId, word: u64) {
        let sig = encode_wide_sigs(&[WideQSig {
            n_heads: 12,
            words: vec![word; 24],
        }]);
        sub.apply_wide_q_sigs(sid, sig, RecordIdentity::NONE);
    }

    /// **A turn's signature moves its own group's epoch and no other's**; a
    /// read moves none.
    #[test]
    fn a_write_moves_only_the_group_it_names() {
        let mut sub = substrate(&[(7, 1), (8, 2)]);
        let (a, b) = (sub.group_epoch(group(1)), sub.group_epoch(group(2)));
        write_sig(&mut sub, turn_stream_id(7, 0), 7);
        assert!(sub.group_epoch(group(1)) > a);
        assert_eq!(sub.group_epoch(group(2)), b, "another group's turn");
        let after = sub.group_epoch(group(1));
        let _ = sub.wide_sig_window(turn_stream_id(7, 0));
        assert_eq!(sub.group_epoch(group(1)), after, "a read");
    }

    /// **A calibration's change moves the group of the file it calibrates**,
    /// and a group drawing from either moves with it.
    #[test]
    fn a_calibration_moves_its_files_group_and_its_drawers() {
        let mut sub = substrate(&[(7, 1), (9, 3)]);
        sub.register_timeline(timeline(8), LayerId::for_test(1), group(2));
        let kv = BTreeMap::from([(Substrate::CALIBRATES_KEY.to_string(), "7".to_string())]);
        sub.merge_custom(timeline(8), &kv, RecordIdentity::NONE);
        sub.set_group_draws(group(4), group(1));
        let before = [1, 2, 3, 4].map(|g| sub.group_epoch(group(g)));
        sub.timeline_content_changed(timeline(8));
        let after = [1, 2, 3, 4].map(|g| sub.group_epoch(group(g)));
        assert!(after[0] > before[0], "the file's group");
        assert!(after[1] > before[1], "the calibration's own group");
        assert_eq!(after[2], before[2], "a group it has nothing to do with");
        assert!(after[3] > before[3], "a group drawing from the file's");
    }

    /// **A change that names no timeline moves every group**: a stream that is
    /// no turn, and a timeline this version does not hold.
    #[test]
    fn a_change_naming_no_timeline_moves_every_group() {
        let mut sub = substrate(&[(7, 1), (8, 2)]);
        let before = [1, 2].map(|g| sub.group_epoch(group(g)));
        write_sig(&mut sub, turn_stream_id(99, 0), 99);
        let after = [1, 2].map(|g| sub.group_epoch(group(g)));
        assert!(after[0] > before[0] && after[1] > before[1]);
        sub.timeline_content_changed(timeline(1234));
        assert!(sub.group_epoch(group(1)) > after[0]);
    }

    /// **A tombstone moves its timeline's group**, as a reset moves every one.
    #[test]
    fn a_tombstone_and_a_reset_move_the_groups_they_change() {
        let mut sub = substrate(&[(7, 1), (8, 2)]);
        let (a, b) = (sub.group_epoch(group(1)), sub.group_epoch(group(2)));
        sub.tombstone_timeline(timeline(7));
        assert!(sub.group_epoch(group(1)) > a);
        assert_eq!(sub.group_epoch(group(2)), b);
        sub.reset();
        assert!(sub.group_epoch(group(2)) > b);
    }
}
