//! Which character is carrying which mission — the world's half of the mission
//! system.
//!
//! The pure [`Mission`](crate::engine::mission::Mission) model owns what a
//! mission *is*: the ask, its steps, the answer it builds, the report that
//! closes it. This owns *whose* it is, so it lives on [`Sim`](crate::sim::Sim),
//! where an act at the command desk can reach it through `with_sim`, where the
//! standing-task nudge can read it, and where it is serialized and persisted
//! with the rest of the world.
//!
//! A body carries at most one open mission at a time — the one it reads each
//! quiet turn and works through. Missions lodged for a body it has not collected
//! wait in a queue, oldest drawn first. A mission it has reported on is kept in
//! `done` so an operator can still read the outcome and answer after the
//! character has moved on to the next.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::engine::mission::bank::{self, Facts};
use crate::engine::mission::{Mission, Outcome};

/// Every character's mission, and the ones lodged for characters yet to collect.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Missions {
    /// The open mission each body is carrying, by body id. At most one.
    active: BTreeMap<String, Mission>,
    /// Missions lodged for a body that it has not collected yet, oldest first.
    lodged: BTreeMap<String, Vec<Mission>>,
    /// The last mission each body finished, kept so an operator can read the
    /// outcome and answer after the character has reported and moved on.
    done: BTreeMap<String, Mission>,
    /// The seed for the next random routine drawn from the bank, bumped each
    /// draw so a character collecting twice does not get the same routine and a
    /// test can pin the sequence.
    seed: u64,
}

impl Missions {
    /// The body's open mission, if it is on one.
    pub fn active(&self, body: &str) -> Option<&Mission> {
        self.active.get(body)
    }

    /// The last mission the body finished, if any.
    pub fn done(&self, body: &str) -> Option<&Mission> {
        self.done.get(body)
    }

    /// The mission to answer an operator's question about this body with — the
    /// open one if it is on one, otherwise the last it finished.
    pub fn latest(&self, body: &str) -> Option<&Mission> {
        self.active.get(body).or_else(|| self.done.get(body))
    }

    /// Whether the body is carrying an open mission right now.
    pub fn is_on_mission(&self, body: &str) -> bool {
        self.active.contains_key(body)
    }

    /// Whether a mission has been lodged for the body and not yet collected.
    pub fn has_lodged(&self, body: &str) -> bool {
        self.lodged.get(body).is_some_and(|queue| !queue.is_empty())
    }

    /// Lodge a mission for a body to collect at the desk. Oldest is drawn first.
    pub fn lodge(&mut self, body: &str, mission: Mission) {
        self.lodged.entry(body.to_string()).or_default().push(mission);
    }

    /// Give the body a mission to carry now, replacing any open one it held.
    pub fn assign(&mut self, body: &str, mission: Mission) {
        self.active.insert(body.to_string(), mission);
    }

    /// Collect a mission at the desk: the next one lodged for this body, or —
    /// when nothing is lodged — a fresh non-destructive routine drawn from the
    /// [`bank`] against who and what is around. Either way it becomes the body's
    /// open mission, and the collected one is returned so the desk can hand back
    /// its brief.
    pub fn collect(&mut self, body: &str, facts: &Facts) -> &Mission {
        let mission = match self.take_lodged(body) {
            Some(lodged) => lodged,
            None => {
                let drawn = bank::random(facts, self.seed);
                self.seed = self.seed.wrapping_add(1);
                drawn
            }
        };
        self.active.insert(body.to_string(), mission);
        &self.active[body]
    }

    /// Draw the next lodged mission for a body, oldest first, if one waits. The
    /// caller assigns it; taking and carrying are two steps so a draw that finds
    /// nothing can fall back to a random one without a mission going missing.
    pub fn take_lodged(&mut self, body: &str) -> Option<Mission> {
        let queue = self.lodged.get_mut(body)?;
        if queue.is_empty() {
            return None;
        }
        let mission = queue.remove(0);
        if queue.is_empty() {
            self.lodged.remove(body);
        }
        Some(mission)
    }

    /// Check a step off the body's open mission. `false` when it has no open
    /// mission or nothing matched — see [`Mission::check_off`].
    pub fn check_off(&mut self, body: &str, step: &str) -> bool {
        self.active.get_mut(body).is_some_and(|m| m.check_off(step))
    }

    /// Add a step to the body's open mission. `false` when it has no open
    /// mission, or the step was blank or a duplicate — see [`Mission::add_todo`].
    pub fn add_todo(&mut self, body: &str, step: &str) -> bool {
        self.active.get_mut(body).is_some_and(|m| m.add_todo(step))
    }

    /// File the body's completion report, closing the open mission and keeping
    /// it in `done` so its outcome and answer can still be read. Returns the
    /// closed mission, or `None` if the body had no open mission to report.
    pub fn report(
        &mut self,
        body: &str,
        outcome: Outcome,
        notes: &str,
        answer: Option<String>,
    ) -> Option<Mission> {
        let mut mission = self.active.remove(body)?;
        mission.complete(outcome, notes, answer);
        self.done.insert(body.to_string(), mission.clone());
        Some(mission)
    }
}

#[cfg(test)]
mod tests {
    use super::Missions;
    use crate::engine::mission::bank::Facts;
    use crate::engine::mission::{Mission, Origin, Outcome, Todo};

    fn a_mission(prompt: &str) -> Mission {
        Mission::new(
            prompt,
            vec![Todo::new("step one"), Todo::new("step two")],
            Origin::Lodged {
                by: "u_op".to_string(),
            },
        )
    }

    #[test]
    fn a_lodged_mission_is_drawn_oldest_first_and_then_the_queue_empties() {
        let mut m = Missions::default();
        assert!(!m.has_lodged("bram"));
        m.lodge("bram", a_mission("first"));
        m.lodge("bram", a_mission("second"));
        assert!(m.has_lodged("bram"));

        assert_eq!(m.take_lodged("bram").unwrap().prompt, "first");
        assert_eq!(m.take_lodged("bram").unwrap().prompt, "second");
        assert!(m.take_lodged("bram").is_none(), "queue empties");
        assert!(!m.has_lodged("bram"));
    }

    #[test]
    fn assigning_a_mission_makes_it_the_active_one() {
        let mut m = Missions::default();
        assert!(!m.is_on_mission("wren"));
        m.assign("wren", a_mission("survey the ridge"));
        assert!(m.is_on_mission("wren"));
        assert_eq!(m.active("wren").unwrap().prompt, "survey the ridge");
        // A second assignment replaces the first — a body carries one at a time.
        m.assign("wren", a_mission("read the canon"));
        assert_eq!(m.active("wren").unwrap().prompt, "read the canon");
    }

    #[test]
    fn steps_are_checked_off_and_added_only_on_an_open_mission() {
        let mut m = Missions::default();
        // No mission: nothing to act on.
        assert!(!m.check_off("pax", "step one"));
        assert!(!m.add_todo("pax", "a new step"));

        m.assign("pax", a_mission("do the thing"));
        assert!(m.check_off("pax", "STEP one"), "trim + case-insensitive");
        assert!(!m.check_off("pax", "step one"), "already done");
        assert!(m.add_todo("pax", "a discovered step"));
        assert!(!m.add_todo("pax", "  a discovered step  "), "open duplicate");
        assert_eq!(m.active("pax").unwrap().todo.len(), 3);
    }

    #[test]
    fn reporting_closes_the_open_mission_and_keeps_it_for_the_operator() {
        let mut m = Missions::default();
        assert!(
            m.report("soren", Outcome::Pass, "n/a", None).is_none(),
            "nothing to report without an open mission"
        );

        m.assign("soren", a_mission("check the record"));
        let closed = m
            .report(
                "soren",
                Outcome::Pass,
                "it holds against the storyline",
                Some("the eastern date is wrong".to_string()),
            )
            .expect("an open mission is reported");
        assert!(!closed.is_open());
        assert_eq!(closed.report.as_ref().unwrap().outcome, Outcome::Pass);

        // The body is now free of its mission, but the operator can still read it.
        assert!(!m.is_on_mission("soren"));
        assert!(m.active("soren").is_none());
        let done = m.done("soren").expect("kept for the operator");
        assert_eq!(done.answer.as_deref(), Some("the eastern date is wrong"));
        assert_eq!(m.latest("soren").unwrap().prompt, "check the record");
    }

    #[test]
    fn collecting_takes_a_lodged_mission_first_then_draws_from_the_bank() {
        let mut m = Missions::default();
        let makers = vec!["Wren".to_string(), "Pax".to_string()];
        let records = vec!["the-charge".to_string()];
        let facts = Facts {
            makers: &makers,
            records: &records,
        };

        // A lodged mission is taken first, verbatim, and becomes active.
        m.lodge("bram", a_mission("the lodged one"));
        assert_eq!(m.collect("bram", &facts).prompt, "the lodged one");
        assert!(m.is_on_mission("bram"));
        assert!(!m.has_lodged("bram"), "the lodged queue is drawn down");

        // With nothing lodged, a routine is drawn from the bank — a valid, open,
        // non-empty, non-destructive mission it can act on.
        let drawn = m.collect("bram", &facts).clone();
        assert!(drawn.is_open());
        assert!(!drawn.todo.is_empty(), "a routine gives steps to act on");
        assert!(matches!(drawn.origin, Origin::Random { .. }));
    }

    #[test]
    fn latest_is_the_open_mission_over_a_finished_one() {
        let mut m = Missions::default();
        m.assign("yen", a_mission("first task"));
        m.report("yen", Outcome::Pass, "done", None);
        m.assign("yen", a_mission("second task"));
        // With both a finished and an open mission, the open one is what an
        // operator's question is about.
        assert_eq!(m.latest("yen").unwrap().prompt, "second task");
    }
}
