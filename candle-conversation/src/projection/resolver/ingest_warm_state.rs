//! Which ingest-layer files have had their per-file normalization backgrounds
//! warmed in this process, and how far the warm-up now running has got.
//!
//! A file's backgrounds are learned by self-match ([`crate::IngestWarmer`]),
//! and a file read again is a new timeline whose backgrounds start cold. The
//! warm-up runs after every ingest pass that changed a layer, so it warms only
//! the timelines not warmed before — a pass that read three files costs three
//! files' scans, not the corpus's. Its progress is the `/v1/status` warm-up
//! bar, counted as [`WarmReplayProgress`] counts the dialogue replay.

use std::collections::HashSet;
use std::sync::Mutex;

use super::warm_replay_progress::WarmReplayProgress;
use super::Conversation;
use crate::projection::TimelineId;

/// The warmed timelines and the running warm-up's count.
#[derive(Default)]
pub struct IngestWarmState {
    warmed: Mutex<HashSet<u64>>,
    progress: WarmReplayProgress,
}

impl IngestWarmState {
    /// `timelines` less those already warmed, in their order.
    pub fn unwarmed(&self, timelines: &[TimelineId]) -> Vec<TimelineId> {
        let warmed = self.warmed.lock().unwrap_or_else(|e| e.into_inner());
        timelines
            .iter()
            .copied()
            .filter(|t| !warmed.contains(&t.raw()))
            .collect()
    }

    /// `n` timelines handed to the warm-up.
    pub fn started(&self, n: usize) {
        self.progress.add(n);
    }

    /// `timelines` warmed.
    pub fn warmed(&self, timelines: &[TimelineId]) {
        let mut warmed = self.warmed.lock().unwrap_or_else(|e| e.into_inner());
        for t in timelines {
            warmed.insert(t.raw());
            self.progress.scored_one();
        }
    }

    /// `n` handed-out timelines the warm-up will not reach: it was cancelled,
    /// or its scheduler is gone.
    pub fn abandoned(&self, n: usize) {
        self.progress.dropped(n);
    }

    /// `(warmed, total)` of the warm-up now running; `None` when none is.
    pub fn progress(&self) -> Option<(usize, usize)> {
        self.progress.snapshot()
    }
}

impl Conversation {
    /// This process's ingest warm-up state, shared across clones of the handle.
    pub fn ingest_warm_state(&self) -> &IngestWarmState {
        &self.ingest_warm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    /// **A file warmed once is not warmed again**, and a file read again — a
    /// new timeline — is: the second pass's work is only what the first did
    /// not cover.
    #[test]
    fn a_later_pass_warms_only_what_is_new() {
        let s = IngestWarmState::default();
        assert_eq!(s.unwarmed(&[tl(1), tl(2)]), vec![tl(1), tl(2)]);
        s.started(2);
        s.warmed(&[tl(1), tl(2)]);
        assert_eq!(s.unwarmed(&[tl(1), tl(2), tl(3)]), vec![tl(3)]);
    }

    /// **The bar counts the warm-up running and settles when it is done**;
    /// one abandoned part way still settles.
    #[test]
    fn progress_counts_the_running_warm_up() {
        let s = IngestWarmState::default();
        assert_eq!(s.progress(), None);
        s.started(3);
        s.warmed(&[tl(1)]);
        assert_eq!(s.progress(), Some((1, 3)));
        s.warmed(&[tl(2), tl(3)]);
        assert_eq!(s.progress(), None);
        s.started(4);
        s.warmed(&[tl(4)]);
        s.abandoned(3);
        assert_eq!(s.progress(), None);
    }
}
