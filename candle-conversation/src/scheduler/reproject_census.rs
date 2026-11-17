//! Where a live reprojection's time goes, summed over a minute.
//!
//! A reprojection runs on the scheduler thread inside the decode quantum, so
//! every part of it is time no slot decodes. The per-reprojection events
//! (`reproject prepare split`, `reproject observe split`, `reproject (zero-copy
//! rebuild)`) keep one recent instance each, which says what one looked like
//! and nothing about where a run's time went. The census adds every part of
//! every reprojection, by how it ended, and the scheduler writes it as one
//! `reprojection census` line when the first reprojection to end after its
//! minute does — one line a minute while reprojections run, none while idle.

use std::time::{Duration, Instant};

/// How often the census is written while reprojections run.
pub(crate) const CENSUS_EVERY: Duration = Duration::from_secs(60);

/// One timed part of a reprojection, in the order it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Part {
    /// The wide-Q probe gathered from the view.
    Probe,
    /// The normalization warm-up queued before the scan.
    Warm,
    /// The belief scan.
    Score,
    /// The selection's hits read and folded into the round momentum.
    Hits,
    /// The working set's momentum observed.
    WorkingSet,
    /// The pool behind the slot refreshed.
    Pool,
    /// The turn-boundary challengers seated.
    Challenger,
    /// The projection over the scored substrate.
    Project,
    /// The selection's identity, compared with the parent's.
    Identity,
    /// The rebuild's lift, judged against the decode floor.
    Lift,
    /// The outstanding gap's rows put in place before the snapshot.
    Gap,
    /// The active turn's tail captured.
    Snapshot,
    /// The tail's index rows sealed.
    Seal,
    /// The recurrent state moved to the parent.
    Move,
    /// The old view's slot freed.
    FreeView,
    /// The old view's sequence released.
    Release,
    /// The parent cut back to its last assembly.
    Truncate,
    /// The new working set brought hot.
    Elevate,
    /// Other conversations kept whole.
    Whole,
    /// The apply's segment build.
    Build,
    /// The shared gap-fill wave this reprojection waited on.
    Glue,
    /// The apply finished.
    Finish,
    /// The captured tail and its index pages re-injected.
    Inject,
    /// The new view carved and the state moved onto it.
    View,
}

impl Part {
    const ALL: [Part; 24] = [
        Part::Probe,
        Part::Warm,
        Part::Score,
        Part::Hits,
        Part::WorkingSet,
        Part::Pool,
        Part::Challenger,
        Part::Project,
        Part::Identity,
        Part::Lift,
        Part::Gap,
        Part::Snapshot,
        Part::Seal,
        Part::Move,
        Part::FreeView,
        Part::Release,
        Part::Truncate,
        Part::Elevate,
        Part::Whole,
        Part::Build,
        Part::Glue,
        Part::Finish,
        Part::Inject,
        Part::View,
    ];

    fn name(self) -> &'static str {
        match self {
            Part::Probe => "probe",
            Part::Warm => "warm",
            Part::Score => "score",
            Part::Hits => "hits",
            Part::WorkingSet => "ws",
            Part::Pool => "pool",
            Part::Challenger => "challenger",
            Part::Project => "project",
            Part::Identity => "identity",
            Part::Lift => "lift",
            Part::Gap => "gap",
            Part::Snapshot => "snapshot",
            Part::Seal => "seal",
            Part::Move => "move",
            Part::FreeView => "free_view",
            Part::Release => "release",
            Part::Truncate => "truncate",
            Part::Elevate => "elevate",
            Part::Whole => "whole",
            Part::Build => "build",
            Part::Glue => "glue",
            Part::Finish => "finish",
            Part::Inject => "inject",
            Part::View => "view",
        }
    }
}

/// How a reprojection ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// The view was rebuilt on the new selection.
    Rebuilt,
    /// The selection was the parent's: nothing was rebuilt.
    Kept,
    /// The rebuild's lift would take the weights below the decode floor.
    Declined,
}

/// See the module documentation.
#[derive(Debug)]
pub(crate) struct ReprojectCensus {
    parts_us: [u64; Part::ALL.len()],
    /// Count and end-to-end microseconds, by [`Outcome`].
    ended: [(u64, u64); 3],
    since: Instant,
}

impl ReprojectCensus {
    pub(crate) fn new(now: Instant) -> Self {
        Self {
            parts_us: [0; Part::ALL.len()],
            ended: [(0, 0); 3],
            since: now,
        }
    }

    /// Count `us` microseconds against `part`.
    pub(crate) fn add(&mut self, part: Part, us: u64) {
        self.parts_us[part as usize] += us;
    }

    /// Count `us` microseconds against `part`, the time since `from`.
    pub(crate) fn add_since(&mut self, part: Part, from: Instant) {
        self.add(part, from.elapsed().as_micros() as u64);
    }

    /// A reprojection ended as `outcome`, `total_us` after it started.
    pub(crate) fn ended(&mut self, outcome: Outcome, total_us: u64) {
        let at = &mut self.ended[outcome as usize];
        at.0 += 1;
        at.1 += total_us;
    }

    /// The census line once [`CENSUS_EVERY`] has passed since the last and a
    /// reprojection ended in it, starting the next; `None` otherwise.
    pub(crate) fn due(&mut self, now: Instant) -> Option<String> {
        let window = now.saturating_duration_since(self.since);
        if window < CENSUS_EVERY {
            return None;
        }
        let ran = self.ended.iter().any(|(n, _)| *n > 0);
        let line = ran.then(|| self.line(window));
        *self = Self::new(now);
        line
    }

    fn line(&self, window: Duration) -> String {
        let ms = |us: u64| format!("{:.1}", us as f64 / 1000.0);
        let [rebuilt, kept, declined] = self.ended;
        let mut out = format!(
            "window_s={} rebuilt={} rebuilt_ms={} kept={} kept_ms={} declined={} declined_ms={}",
            window.as_secs(),
            rebuilt.0,
            ms(rebuilt.1),
            kept.0,
            ms(kept.1),
            declined.0,
            ms(declined.1),
        );
        for part in Part::ALL {
            out.push_str(&format!(
                " {}_ms={}",
                part.name(),
                ms(self.parts_us[part as usize])
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every part and every outcome is in the line, summed**, in the order
    /// a reprojection runs, in milliseconds to a tenth.
    #[test]
    fn the_line_sums_each_part_and_outcome() {
        let t0 = Instant::now();
        let mut c = ReprojectCensus::new(t0);
        c.add(Part::Score, 120_000);
        c.add(Part::Score, 140_500);
        c.add(Part::Hits, 18_260);
        c.add(Part::View, 4_000);
        c.ended(Outcome::Rebuilt, 190_000);
        c.ended(Outcome::Rebuilt, 180_000);
        c.ended(Outcome::Kept, 14_000);
        assert_eq!(
            c.due(t0 + CENSUS_EVERY).as_deref(),
            Some(
                "window_s=60 rebuilt=2 rebuilt_ms=370.0 kept=1 kept_ms=14.0 declined=0 \
                 declined_ms=0.0 probe_ms=0.0 warm_ms=0.0 score_ms=260.5 hits_ms=18.3 \
                 ws_ms=0.0 pool_ms=0.0 challenger_ms=0.0 project_ms=0.0 identity_ms=0.0 \
                 lift_ms=0.0 gap_ms=0.0 snapshot_ms=0.0 seal_ms=0.0 move_ms=0.0 \
                 free_view_ms=0.0 release_ms=0.0 truncate_ms=0.0 elevate_ms=0.0 \
                 whole_ms=0.0 build_ms=0.0 glue_ms=0.0 finish_ms=0.0 inject_ms=0.0 \
                 view_ms=4.0"
            )
        );
    }

    /// **Nothing is written before the minute, and nothing for a minute in
    /// which no reprojection ended**; each line starts the next window empty.
    #[test]
    fn a_line_is_due_once_a_minute_and_only_after_reprojections() {
        let t0 = Instant::now();
        let mut c = ReprojectCensus::new(t0);
        c.ended(Outcome::Declined, 1_000);
        assert_eq!(c.due(t0 + CENSUS_EVERY / 2), None);
        let first = c.due(t0 + CENSUS_EVERY).expect("a reprojection ended");
        assert!(first.contains(" declined=1 declined_ms=1.0 "), "{first}");
        assert_eq!(c.due(t0 + CENSUS_EVERY * 2), None);
        c.add(Part::Glue, 5_000);
        assert_eq!(
            c.due(t0 + CENSUS_EVERY * 3),
            None,
            "a part alone ends nothing"
        );
    }
}
