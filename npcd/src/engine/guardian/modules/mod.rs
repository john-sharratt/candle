//! The modules a guardian can be built from, one per file.

pub mod drift;
pub mod journal;
pub mod looping;
pub mod stall;
pub mod step_tracker;

pub use drift::Drift;
pub use journal::Journal;
pub use looping::Looping;
pub use stall::Stall;
pub use step_tracker::StepTracker;

#[cfg(test)]
pub mod fixtures {
    //! Views the module tests share.

    use std::time::Duration;

    use crate::engine::guardian::view::{MissionView, NpcView, Step};

    pub fn view(mission: Option<MissionView>) -> NpcView {
        NpcView {
            npc_id: 7,
            mission,
            since_progress: Duration::ZERO,
            stations: Vec::new(),
            recent_acts: Vec::new(),
            journal: None,
        }
    }

    /// `view` with these acts as its most recent, oldest first.
    pub fn doing(mut v: NpcView, acts: &[&str]) -> NpcView {
        v.recent_acts = acts.iter().map(|a| a.to_string()).collect();
        v
    }

    /// A character carrying "Find the ledger." with the given open steps.
    pub fn with_mission(steps: &[&str]) -> NpcView {
        view(Some(MissionView {
            prompt: "Find the ledger.".to_string(),
            steps: steps
                .iter()
                .map(|s| Step {
                    text: s.to_string(),
                    done: false,
                    reports: false,
                })
                .collect(),
            standing: "What has been asked of you: Find the ledger.".to_string(),
        }))
    }
}
