//! Where a belief scan's turn-group half spent its time, summed over the groups
//! one scan visits.
//!
//! The per-group line is `trace` — there are ~17 groups a scan and it was a
//! quarter of a daemon log — so the per-scan `belief scan phase split` engine
//! event (`GET /v1/events`) carries their totals: the one number per phase that says whether the group
//! half grew in its host walk (`select` / `assemble`), its launches (`score`) or
//! its normalization.
//!
//! Thread-local because a scan runs on one thread from start to finish and the
//! groups are scored by a method several callers share; the scan resets it
//! before the groups and takes it after.

use std::cell::Cell;

/// The turn-group half of one belief scan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GroupScanSplit {
    /// Groups scanned.
    pub scanned: u64,
    /// Gallery windows scored, over every group.
    pub windows: u64,
    /// Turns the candidate assembly walked.
    pub turns: u64,
    /// Choosing each group's timelines.
    pub select_us: u64,
    /// Building each group's files and windows from the substrate.
    pub assemble_us: u64,
    /// The scans themselves.
    pub score_us: u64,
    /// Normalizing, fusing and stamping the scores.
    pub normalize_us: u64,
    /// Files the assembly kept.
    pub files: u64,
    /// Of `assemble_us`: walking each file's own turns — the summary check,
    /// the stored window and its seams per turn.
    pub walk_us: u64,
    /// Of `assemble_us`: grouping the turns into exchanges and flattening
    /// their windows.
    pub exchange_us: u64,
    /// Calibration records built rather than found standing, over every file.
    pub cal_built: u64,
    /// Calibration windows the assembly kept, over every file.
    pub cal_windows: u64,
    /// Of `assemble_us`: walking every file's calibrations.
    pub cal_us: u64,
    /// Of `score_us`: laying the windows out as the arena's segments.
    pub segments_us: u64,
    /// Files whose gallery the assembly built, rather than found held on
    /// their timeline (`Substrate::file_gallery`).
    pub galleries_built: u64,
    /// Of `select_us`: the group's timelines in the target's scope.
    pub scope_us: u64,
    /// Of `select_us`: keeping the timelines that carry a tagged group's tags.
    pub tags_us: u64,
    /// Of `assemble_us`: the per-file loop — each held gallery read back (or
    /// built), its windows taken and its calibrations attached.
    pub files_us: u64,
    /// Of `assemble_us`: cutting the group's excluded chunks out.
    pub exclude_us: u64,
    /// Of `assemble_us`: each file's kind, for a group normalized by kind.
    pub kinds_us: u64,
    /// Of `assemble_us`: the target's lineage, for a group that leaves it out.
    pub lineage_us: u64,
    /// Of `assemble_us`: letting go of the substrate version the assembly
    /// read — a version no one else holds is freed here.
    pub release_us: u64,
    /// Groups whose files were kept from an earlier scan rather than gathered
    /// (`group_assembly`).
    pub assemblies_reused: u64,
}

thread_local! {
    /// This thread's scan in progress.
    pub(crate) static GROUP_SCAN_SPLIT: Cell<GroupScanSplit> =
        const { Cell::new(GroupScanSplit {
            scanned: 0,
            windows: 0,
            turns: 0,
            select_us: 0,
            assemble_us: 0,
            score_us: 0,
            normalize_us: 0,
            files: 0,
            walk_us: 0,
            exchange_us: 0,
            cal_built: 0,
            cal_windows: 0,
            cal_us: 0,
            segments_us: 0,
            galleries_built: 0,
            scope_us: 0,
            tags_us: 0,
            files_us: 0,
            exclude_us: 0,
            kinds_us: 0,
            lineage_us: 0,
            release_us: 0,
            assemblies_reused: 0,
        }) };
}

/// Add to this thread's scan in progress.
pub(crate) fn add_to_split(f: impl FnOnce(&mut GroupScanSplit)) {
    let mut s = GROUP_SCAN_SPLIT.get();
    f(&mut s);
    GROUP_SCAN_SPLIT.set(s);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Taking the split hands back what was summed and leaves it empty for the
    /// next scan.
    #[test]
    fn taking_the_split_resets_it() {
        GROUP_SCAN_SPLIT.set(GroupScanSplit {
            scanned: 2,
            score_us: 7,
            ..GroupScanSplit::default()
        });
        let taken = GROUP_SCAN_SPLIT.take();
        assert_eq!(taken.scanned, 2);
        assert_eq!(taken.score_us, 7);
        assert_eq!(GROUP_SCAN_SPLIT.get(), GroupScanSplit::default());
    }

    /// What is added sums onto what the scan holds.
    #[test]
    fn adding_sums_onto_the_scan_in_progress() {
        GROUP_SCAN_SPLIT.take();
        add_to_split(|s| s.cal_windows += 24);
        add_to_split(|s| {
            s.cal_windows += 24;
            s.cal_us += 9;
        });
        let taken = GROUP_SCAN_SPLIT.take();
        assert_eq!(taken.cal_windows, 48);
        assert_eq!(taken.cal_us, 9);
    }
}
