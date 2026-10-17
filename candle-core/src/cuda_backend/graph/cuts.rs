//! Where a wave is cut into segments, by the source location that asked.
//!
//! A segment ends wherever the forward meets the device eagerly — a pause for
//! an allocation, an upload or a readback, or a flush for the MoE host
//! protocol — and where those land decides which runs of layers a segment
//! holds. Segments that start and end in different places from one wave to the
//! next are different topologies, so their graphs are never reused: on `.5`
//! (2026-10-17) half of 4,915 new topologies began with a different kernel from
//! the executable they were compared against, about 70 segments a wave. This
//! counts every cut by its caller (`#[track_caller]` from `Device::eager`,
//! `CudaDevice::pause_capture` and `flush_launches`) and reports the sites
//! cutting most, per wave, so the cuts can be moved to fixed places.

use std::collections::HashMap;
use std::panic::Location;

/// Waves between two reports.
pub(super) const REPORT_WAVES: u64 = 512;

/// Sites a report names.
pub(super) const REPORT_SITES: usize = 12;

/// One cut site: the file and line that asked.
pub(super) type Site = (&'static str, u32);

/// One window's cuts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CutReport {
    pub(super) waves: u64,
    pub(super) cuts: u64,
    /// The sites that cut most, most first, with their counts.
    pub(super) sites: Vec<(Site, u64)>,
}

/// See the module documentation.
#[derive(Debug, Default)]
pub(super) struct CutCounter {
    by_site: HashMap<Site, u64>,
    cuts: u64,
    waves: u64,
}

impl CutCounter {
    /// A segment was cut at the request of `at`.
    pub(super) fn cut(&mut self, at: &'static Location<'static>) {
        *self.by_site.entry((at.file(), at.line())).or_default() += 1;
        self.cuts += 1;
    }

    /// A wave began. Returns the window's report — the [`REPORT_SITES`] sites
    /// that cut most — once [`REPORT_WAVES`] have passed, and starts the next.
    pub(super) fn wave(&mut self) -> Option<CutReport> {
        self.waves += 1;
        if self.waves < REPORT_WAVES {
            return None;
        }
        let mut sites: Vec<(Site, u64)> = self.by_site.drain().collect();
        sites.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        sites.truncate(REPORT_SITES);
        let report = CutReport {
            waves: self.waves,
            cuts: self.cuts,
            sites,
        };
        self.waves = 0;
        self.cuts = 0;
        Some(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[track_caller]
    fn here() -> &'static Location<'static> {
        Location::caller()
    }

    /// **The sites that cut most come first**, with their counts, once a
    /// window of `REPORT_WAVES` has passed — and the next window starts empty.
    #[test]
    fn a_window_reports_its_busiest_cut_sites() {
        let mut c = CutCounter::default();
        let a = here();
        let b = here();
        for _ in 0..3 {
            c.cut(a);
        }
        c.cut(b);
        for _ in 0..REPORT_WAVES - 1 {
            assert!(c.wave().is_none());
        }
        let report = c.wave().expect("a full window");
        assert_eq!(
            report,
            CutReport {
                waves: REPORT_WAVES,
                cuts: 4,
                sites: vec![((a.file(), a.line()), 3), ((b.file(), b.line()), 1)],
            }
        );
        for _ in 0..REPORT_WAVES - 1 {
            c.wave();
        }
        assert_eq!(
            c.wave(),
            Some(CutReport {
                waves: REPORT_WAVES,
                cuts: 0,
                sites: vec![],
            })
        );
    }
}
