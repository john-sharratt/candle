//! What a window's `warm` band was spent on: the background launches the loop
//! runs one per pass (`norm_warm`) — a replayed dialogue turn, a raw group
//! scan, a held launch from the chunk-exclusion learner, an ingest warm-up
//! slice.
//!
//! The phase chart draws warm as one band, and a live turn waits on at most
//! one of its launches a pass; which kind costs it, and how long its longest
//! single launch held the loop, is what making it faster needs, so the
//! window's close reports this beside it.

use std::time::Duration;

/// The kinds of background launch, in the order [`WarmSplit::parts_ms`]
/// reports them.
pub(super) const WARM_KINDS: [&str; 4] = ["replay", "scan", "held", "ingest"];

/// A kind of background launch, as an index into [`WARM_KINDS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WarmKind {
    /// A replayed dialogue turn of the normalization warm-up.
    Replay,
    /// A raw group-vote scan.
    Scan,
    /// A held launch sent from another thread — the chunk-exclusion learner's.
    Held,
    /// One slice of an ingest group's per-file warm-up.
    Ingest,
}

/// One window's background launches, by kind: how many, their total time and
/// the longest, in microseconds.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct WarmSplit {
    pub count: [u32; WARM_KINDS.len()],
    pub us: [u64; WARM_KINDS.len()],
    pub max_us: [u64; WARM_KINDS.len()],
}

impl WarmSplit {
    /// One launch of `kind` that held the loop for `took`.
    pub fn add(&mut self, kind: WarmKind, took: Duration) {
        let i = kind as usize;
        let us = took.as_micros() as u64;
        self.count[i] += 1;
        self.us[i] += us;
        self.max_us[i] = self.max_us[i].max(us);
    }

    /// The kinds that ran as `name=count/ms/max_ms`, for the log.
    pub fn parts_ms(&self) -> String {
        WARM_KINDS
            .iter()
            .enumerate()
            .filter(|&(i, _)| self.count[i] > 0)
            .map(|(i, name)| {
                format!(
                    "{name}={}/{}/{}",
                    self.count[i],
                    self.us[i] / 1000,
                    self.max_us[i] / 1000
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each kind counts its own launches, time and longest, and reads back as
    /// `name=count/ms/max_ms` for the kinds that ran.
    #[test]
    fn each_kind_counts_its_launches_and_reads_back_by_name() {
        let ms = Duration::from_millis;
        let mut s = WarmSplit::default();
        s.add(WarmKind::Held, ms(270));
        s.add(WarmKind::Held, ms(30));
        s.add(WarmKind::Replay, ms(45));
        assert_eq!(s.count, [1, 0, 2, 0]);
        assert_eq!(s.us, [45_000, 0, 300_000, 0]);
        assert_eq!(s.max_us, [45_000, 0, 270_000, 0]);
        assert_eq!(s.parts_ms(), "replay=1/45/45 held=2/300/270");
        assert_eq!(WarmSplit::default().parts_ms(), "");
    }
}
