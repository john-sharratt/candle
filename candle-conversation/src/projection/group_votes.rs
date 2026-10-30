//! A belief group's raw votes for a batch of probes, on the GPU gallery arena:
//! every file of the group one case of one competition, scanned as the live
//! belief scan scans a file-scored group, and handed back before the needle
//! gate and the tally — for a caller outside the daemon that weighs, gates and
//! folds the votes itself, against the gallery the daemon already holds
//! resident (`docs/provenance_scoped_signatures.md` §3).

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use super::belief_files::{assemble_file_scans, FileScan};
use super::ids::{GroupId, TimelineId};
use crate::persistence::streams::StreamId;
use crate::projection::Conversation;
use crate::provenance::chunk_exclusion::uncut;
use crate::provenance::gallery_arena::{
    GroupSet, Holder, Keep, PagedSegment, PagedWindow, TokenVotes,
};
use crate::provenance::{GalleryArena, WideQSig};

/// The votes: each case's file, and every probe token's vote per fold group
/// (`TokenVotes::case` indexes `files`).
pub struct GroupVotes {
    pub files: Vec<TimelineId>,
    pub votes: TokenVotes,
}

/// One segment over `files`, file `i` case `i`: each file's own windows, its
/// calibrations' questions too when `calibrations` — pooled in the file's
/// case, for a caller measuring them there — no window of a turn `masked`
/// names, and none of the token ranges `excluded` names by turn, scanned in
/// `groups` (the others vote for nothing) of the tokens `keep` keeps.
fn group_segment<'a>(
    files: &'a [FileScan],
    calibrations: bool,
    masked: &HashSet<StreamId>,
    excluded: &HashMap<StreamId, Vec<Range<usize>>>,
    groups: GroupSet,
    keep: Keep,
) -> PagedSegment<'a> {
    let mut windows = Vec::new();
    for (case, f) in files.iter().enumerate() {
        let own = f.windows.iter().map(|&(k, s, e, _)| (k, s, e));
        let cal = f.cal_windows.iter().copied().filter(|_| calibrations);
        for (k, start, end) in own.chain(cal) {
            let sid = f.arc_sids[k];
            if masked.contains(&sid) {
                continue;
            }
            let cut = excluded.get(&sid).map_or(&[][..], Vec::as_slice);
            for (s, e) in uncut(start, end, cut) {
                windows.push(PagedWindow {
                    sid,
                    fingerprint: f.arcs_kept[k].fingerprint(),
                    turn: (&f.arcs_kept[k]).into(),
                    start: s,
                    end: e,
                    case,
                });
            }
        }
    }
    // Every active file of the group: a read across conversations.
    PagedSegment {
        windows,
        n_cases: files.len(),
        holder: Holder::Shared,
        groups,
        keep,
    }
}

impl Conversation {
    /// `probes`' raw votes against every active file of `group` — its
    /// calibrations' questions in the gallery when `calibrations`, the turns
    /// `masked` names left out, and the token ranges `excluded` names cut from
    /// the gallery so the scan never sees them — on `arena`, in the fold
    /// `groups` (`None` for every group of the arena), of the tokens `keep`
    /// has the card hold.
    #[allow(clippy::too_many_arguments)]
    pub fn scan_group_votes(
        &self,
        group: GroupId,
        calibrations: bool,
        masked: &HashSet<StreamId>,
        excluded: &HashMap<StreamId, Vec<Range<usize>>>,
        probes: &[Vec<WideQSig>],
        groups: Option<GroupSet>,
        keep: Keep,
        arena: &GalleryArena,
    ) -> candle::Result<GroupVotes> {
        let (files, _) = {
            let sub = self.inner.read();
            let timelines: Vec<TimelineId> = sub.active_timelines_for_group(group).collect();
            assemble_file_scans(&sub, timelines)
        };
        let groups = groups.unwrap_or_else(|| arena.all_groups());
        let segment = group_segment(&files, calibrations, masked, excluded, groups, keep);
        let probe_refs: Vec<&[WideQSig]> = probes.iter().map(Vec::as_slice).collect();
        let votes = arena.scan_votes(&[segment], &probe_refs)?;
        Ok(GroupVotes {
            files: files.iter().map(|f| f.timeline).collect(),
            votes,
        })
    }
}

// The tests build lists of one range on purpose — a turn of one exchange, a
// cut covering a whole window — which the lint takes for a mistyped range.
#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::persistence::content_hash::turn_stream_id;
    use crate::projection::ids::TurnIndex;
    use crate::provenance::{encode_wide_sigs, SigWindow};

    fn window(fill: u64) -> SigWindow {
        let sig = WideQSig {
            n_heads: 12,
            words: vec![fill; 24],
        };
        SigWindow::new(Arc::from(encode_wide_sigs(&[sig]))).expect("a stored window")
    }

    /// A file of one turn, plus a calibration's question when `question` names
    /// the calibration's timeline.
    fn file(raw: u64, question: Option<u64>) -> FileScan {
        let tl = TimelineId::for_test(raw);
        let mut arcs_kept = vec![window(raw)];
        let mut arc_sids = vec![turn_stream_id(raw, 0)];
        let mut cal_windows = Vec::new();
        if let Some(cal) = question {
            arcs_kept.push(window(cal));
            arc_sids.push(turn_stream_id(cal, 0));
            cal_windows.push((1, 0, 1));
        }
        FileScan {
            timeline: tl,
            arc_turn: vec![TurnIndex(0)],
            ex_ranges: vec![0..1],
            n_slots: 1,
            arcs_kept,
            arc_sids,
            windows: vec![(0, 0, 1, 0)],
            cal_windows,
            ex_tokens: vec![1],
        }
    }

    /// **Each file is its own case, its questions in or out as asked, masked
    /// turns never.** Two files, the second calibrated: with the questions the
    /// second votes through two windows; without, through its own alone; and a
    /// masked turn leaves the gallery either way.
    #[test]
    fn a_group_segment_holds_each_files_windows_as_its_case() {
        let files = vec![file(71, None), file(72, Some(73))];
        let none = HashSet::new();
        let uncut_all = HashMap::new();
        let cases = |s: &PagedSegment| {
            s.windows
                .iter()
                .map(|w| (w.sid, w.case))
                .collect::<Vec<_>>()
        };

        let all = GroupSet::all(3);
        let every = Keep::All;
        let with = group_segment(&files, true, &none, &uncut_all, all, every);
        assert_eq!(with.n_cases, 2);
        assert_eq!(with.groups, all);
        assert_eq!(
            group_segment(&files, true, &none, &uncut_all, all, Keep::FirstPerCode(1)).keep,
            Keep::FirstPerCode(1)
        );
        assert_eq!(
            cases(&with),
            vec![
                (turn_stream_id(71, 0), 0),
                (turn_stream_id(72, 0), 1),
                (turn_stream_id(73, 0), 1),
            ]
        );

        let without = group_segment(&files, false, &none, &uncut_all, all, every);
        assert_eq!(
            cases(&without),
            vec![(turn_stream_id(71, 0), 0), (turn_stream_id(72, 0), 1)]
        );

        let masked: HashSet<StreamId> = [turn_stream_id(73, 0), turn_stream_id(71, 0)].into();
        assert_eq!(
            cases(&group_segment(
                &files, true, &masked, &uncut_all, all, every
            )),
            vec![(turn_stream_id(72, 0), 1)]
        );

        // The first file's one token cut from its turn leaves it no window.
        let cut: HashMap<StreamId, Vec<Range<usize>>> =
            [(turn_stream_id(71, 0), vec![0..1])].into();
        assert_eq!(
            cases(&group_segment(&files, false, &none, &cut, all, every)),
            vec![(turn_stream_id(72, 0), 1)]
        );
    }
}
