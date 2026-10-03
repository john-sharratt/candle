//! A mission whose steps have not moved for too long.

use std::time::Duration;

use crate::engine::guardian::module::Module;
use crate::engine::guardian::view::{Concern, NpcView, Question, Verdict};

pub struct Stall {
    no_progress_for: Duration,
}

impl Stall {
    pub fn new(no_progress_for: Duration) -> Self {
        Self { no_progress_for }
    }
}

impl Module for Stall {
    fn name(&self) -> &'static str {
        "stall"
    }

    fn question(&self, _view: &NpcView) -> Option<Question> {
        None
    }

    /// A mission with no steps has no progress to stand still, so only one with
    /// a step still open can stall.
    fn judge(&self, view: &NpcView, _answer: Option<&str>) -> Verdict {
        let open = view
            .mission
            .as_ref()
            .is_some_and(|m| m.open_step().is_some());
        if open && view.since_progress >= self.no_progress_for {
            Verdict::Concern(Concern::Stalled)
        } else {
            Verdict::Healthy
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::guardian::modules::fixtures::{view, with_mission};

    fn after(mut v: NpcView, secs: u64) -> NpcView {
        v.since_progress = Duration::from_secs(secs);
        v
    }

    #[test]
    fn it_never_asks() {
        let stall = Stall::new(Duration::from_secs(60));
        assert_eq!(stall.question(&with_mission(&["a"])), None);
    }

    #[test]
    fn a_step_left_open_past_the_window_is_stalled() {
        let stall = Stall::new(Duration::from_secs(60));
        assert_eq!(
            stall.judge(&after(with_mission(&["a"]), 59), None),
            Verdict::Healthy
        );
        assert_eq!(
            stall.judge(&after(with_mission(&["a"]), 60), None),
            Verdict::Concern(Concern::Stalled)
        );
    }

    #[test]
    fn a_mission_without_steps_or_without_a_mission_cannot_stall() {
        let stall = Stall::new(Duration::from_secs(60));
        assert_eq!(
            stall.judge(&after(with_mission(&[]), 900), None),
            Verdict::Healthy
        );
        assert_eq!(stall.judge(&after(view(None), 900), None), Verdict::Healthy);
    }
}
