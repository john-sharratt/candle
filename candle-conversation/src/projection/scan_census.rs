//! Where the belief scans' time went, summed until taken.
//!
//! A reprojection's `score` part is one belief scan, and the per-scan split
//! (`belief scan phase split`, `GET /v1/events`) keeps only the latest. The
//! reprojection census takes this sum once a minute and writes it as its third
//! line, `reprojection census scan`: the one view of which half of the scans —
//! the collections or the groups' walk, launches and normalization, and of the
//! arena's launches the upload, the kernel or the readback — grew.
//!
//! Thread-local, as the splits it sums are: a scan runs on one thread from
//! start to finish, the scheduler's, which also writes the census.

use std::cell::Cell;

use super::belief_scan_split::GroupScanSplit;
use crate::provenance::gallery_arena::scan_split::ArenaScanSplit;

/// The belief scans since the census last took them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScanCensus {
    pub scans: u64,
    pub collections_us: u64,
    pub groups_us: u64,
    pub select_us: u64,
    pub assemble_us: u64,
    pub score_us: u64,
    pub normalize_us: u64,
    pub segments_us: u64,
    pub assemblies_reused: u64,
    pub groups_scanned: u64,
    pub windows: u64,
    pub launches: u64,
    pub fresh_tokens: u64,
    pub index_us: u64,
    pub fingerprint_us: u64,
    pub upload_us: u64,
    pub kernel_us: u64,
    pub kernel_gpu_us: u64,
    pub readback_us: u64,
    pub rows_us: u64,
    pub tally_us: u64,
}

impl ScanCensus {
    /// The census line, in milliseconds to a tenth.
    pub fn line(&self) -> String {
        let ms = |us: u64| format!("{:.1}", us as f64 / 1000.0);
        format!(
            "scans={} collections_ms={} groups_ms={} select_ms={} assemble_ms={} \
             group_score_ms={} normalize_ms={} segments_ms={} groups_scanned={} \
             assemblies_reused={} windows={} launches={} fresh_tokens={} index_ms={} \
             fingerprint_ms={} upload_ms={} kernel_ms={} kernel_gpu_ms={} readback_ms={} \
             rows_ms={} tally_ms={}",
            self.scans,
            ms(self.collections_us),
            ms(self.groups_us),
            ms(self.select_us),
            ms(self.assemble_us),
            ms(self.score_us),
            ms(self.normalize_us),
            ms(self.segments_us),
            self.groups_scanned,
            self.assemblies_reused,
            self.windows,
            self.launches,
            self.fresh_tokens,
            ms(self.index_us),
            ms(self.fingerprint_us),
            ms(self.upload_us),
            ms(self.kernel_us),
            ms(self.kernel_gpu_us),
            ms(self.readback_us),
            ms(self.rows_us),
            ms(self.tally_us),
        )
    }

    /// One scan: its two halves' times and its two splits.
    fn add(&mut self, collections_us: u64, groups_us: u64, g: &GroupScanSplit, a: &ArenaScanSplit) {
        self.scans += 1;
        self.collections_us += collections_us;
        self.groups_us += groups_us;
        self.select_us += g.select_us;
        self.assemble_us += g.assemble_us;
        self.score_us += g.score_us;
        self.normalize_us += g.normalize_us;
        self.segments_us += g.segments_us;
        self.assemblies_reused += g.assemblies_reused;
        self.groups_scanned += g.scanned;
        self.windows += g.windows;
        self.launches += a.launches;
        self.fresh_tokens += a.fresh_tokens;
        self.index_us += a.index_us;
        self.fingerprint_us += a.fingerprint_us;
        self.upload_us += a.upload_us;
        self.kernel_us += a.kernel_us;
        self.kernel_gpu_us += a.kernel_gpu_us;
        self.readback_us += a.readback_us;
        self.rows_us += a.rows_us;
        self.tally_us += a.tally_us;
    }
}

thread_local! {
    static SCAN_CENSUS: Cell<ScanCensus> = Cell::new(ScanCensus::default());
}

/// Count one scan on this thread.
pub(crate) fn record(collections_us: u64, groups_us: u64, g: &GroupScanSplit, a: &ArenaScanSplit) {
    let mut c = SCAN_CENSUS.get();
    c.add(collections_us, groups_us, g, a);
    SCAN_CENSUS.set(c);
}

/// This thread's scans since the last take, leaving none.
pub fn take() -> ScanCensus {
    SCAN_CENSUS.take()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every scan is summed until taken**, and taking leaves none.
    #[test]
    fn scans_sum_until_taken() {
        take();
        let g = GroupScanSplit {
            select_us: 1_000,
            score_us: 4_500,
            scanned: 3,
            ..GroupScanSplit::default()
        };
        let a = ArenaScanSplit {
            kernel_us: 2_000,
            launches: 2,
            ..ArenaScanSplit::default()
        };
        record(300, 6_000, &g, &a);
        record(300, 6_000, &g, &a);
        let c = take();
        assert_eq!(
            (c.scans, c.select_us, c.score_us, c.kernel_us),
            (2, 2_000, 9_000, 4_000)
        );
        assert_eq!((c.groups_scanned, c.launches), (6, 4));
        assert!(c
            .line()
            .starts_with("scans=2 collections_ms=0.6 groups_ms=12.0 select_ms=2.0"));
        assert_eq!(take(), ScanCensus::default());
    }
}
