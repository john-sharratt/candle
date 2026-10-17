//! Where a belief scan's turn-group half spent its time, summed over the groups
//! one scan visits.
//!
//! The per-group line is `trace` — there are ~17 groups a scan and it was a
//! quarter of a daemon log — so the per-scan `belief scan phase split` line
//! carries their totals: the one number per phase that says whether the group
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
        }) };
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
}
