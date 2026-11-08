//! A drone driving a task of Zen's (`docs/mission_templates.md` §7): a coding
//! or maintenance mission lends its carrier [`acts`] that start, answer and
//! steer one zend task on the thread named for the mission, made as Zen
//! (`service:zen`, `docs/zen_memory_and_guides.md` D8); the task's news comes
//! back on the carrier's handset, and the mission's [`step`] is signed off when
//! the task reports done.
//!
//! **Granted wherever it is carried.** A task is work in the house's
//! repositories, not in a room, so — unlike the acts that build the world —
//! these are offered while the mission granting them is carried, wherever its
//! carrier stands ([`task_grants`]).
//!
//! | Module | |
//! |---|---|
//! | [`acts`] | `task_start`, `task_reply`, `task_nudge` |
//! | [`request`] | the zend call each act makes, and reading zend's answer |
//! | [`step`] | the step the task's completion signs off |

pub mod acts;
pub mod request;
pub mod step;

use serde::{Deserialize, Serialize};

use crate::engine::tools::Within;
use crate::world::Hosted;
pub use acts::{is_mine, TASK_ACTS};
pub use step::{is_task_step, TASK_STEP};

/// The task a carried mission drives, once started: its handle, and a digest
/// of the news last posted to its carrier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    pub handle: String,
    #[serde(default)]
    pub heard: u64,
}

impl Within {
    /// This situation with the task acts `body`'s carried mission grants
    /// added to what it is granted where it stands.
    pub fn with_task_grants(mut self, hosted: &Hosted, body: &str) -> Self {
        for g in task_grants(hosted, body) {
            if !self.grants.contains(&g) {
                self.grants.push(g);
            }
        }
        self
    }
}

/// The task acts `body`'s carried mission grants: wherever it stands.
pub fn task_grants(hosted: &Hosted, body: &str) -> Vec<String> {
    hosted.sim(|s| {
        s.missions
            .active(body)
            .map(|m| m.grants.iter().filter(|g| is_mine(g)).cloned().collect())
            .unwrap_or_default()
    })
}

#[cfg(test)]
mod tests {
    use npc_map::schema::Where;
    use npc_map::world::World;
    use npc_map::MapSet;

    use super::*;
    use crate::engine::mission::{engine_sees, Mission, Origin, StepOutcome, Todo};

    fn vault() -> Hosted {
        let map =
            MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps")).unwrap();
        let h = Hosted::of("creators-vault", World::new(map));
        h.with(|w| {
            w.enter("drone", "Wick", Where::new("vault-casting", "green-room"))
                .unwrap();
        });
        h
    }

    fn coding() -> Mission {
        Mission::new(
            "Fix the door sensor's build.",
            vec![
                Todo::new("start the task"),
                Todo::new(TASK_STEP),
                Todo::new("tell the authorizers what changed"),
            ],
            Origin::Standing {
                id: "fix-door".into(),
                room: "vault-casting/band-one".into(),
            },
        )
        .granting(vec![
            "task_start".into(),
            "task_reply".into(),
            "onboard_part".into(),
        ])
    }

    /// **A task act is granted wherever its mission is carried** — not in its
    /// room alone, as a building act is — and nothing without the mission.
    #[test]
    fn task_acts_are_granted_wherever_the_mission_is_carried() {
        let h = vault();
        assert!(task_grants(&h, "drone").is_empty());
        h.with_sim(|s| s.missions.assign("drone", coding()));
        assert_eq!(task_grants(&h, "drone"), ["task_start", "task_reply"]);
        let within = Within::default().with_task_grants(&h, "drone");
        assert_eq!(
            within.grants,
            ["task_start", "task_reply"],
            "not its room, still granted"
        );
    }

    /// **The task step is the engine's to sign off, and only on done**: the
    /// carrier's word does not close it, and news is posted once per change.
    #[test]
    fn the_task_step_closes_only_when_the_task_is_done() {
        assert!(engine_sees(TASK_STEP));
        let h = vault();
        h.with_sim(|s| s.missions.assign("drone", coding()));
        h.with_sim(|s| {
            assert!(s.missions.set_task(
                "drone",
                Task {
                    handle: "task:mission-1-0/fix-door".into(),
                    heard: 0
                }
            ));
            assert!(s.missions.heard("drone", 42), "first news");
            assert!(
                !s.missions.heard("drone", 42),
                "the same news is not posted twice"
            );
            assert!(s.missions.heard("drone", 43));
        });
        h.with_sim(|s| {
            let m = s.missions.active_mut("drone").unwrap();
            assert!(m.task_done());
            assert_eq!(m.todo[1].outcome, Some(StepOutcome::Achieved));
            assert!(!m.todo[0].done, "only the task step is the task's");
            assert!(!m.todo[2].done);
            assert!(!m.task_done(), "once");
        });
    }
}
