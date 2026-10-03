//! Whether what a character has been doing is what it was asked to do.
//!
//! A character's own account of its mission follows the scene rather than the
//! prompt (the `mission` check answers from recent events), so the module asks
//! a closed question built from the mission instead of reading free text.

use crate::engine::guardian::module::Module;
use crate::engine::guardian::view::{Concern, NpcView, Question, Verdict};

pub const ON_MISSION: &str = "part of what I was asked";
pub const OFF_MISSION: &str = "something else";

pub struct Drift;

impl Module for Drift {
    fn name(&self) -> &'static str {
        "drift"
    }

    fn question(&self, view: &NpcView) -> Option<Question> {
        let mission = view.mission.as_ref()?;
        Some(Question {
            text: format!(
                "You were asked: \"{}\" Think about what you have been doing in your last few \
                 actions. Was it part of that, or something else?",
                mission.prompt
            ),
            choices: vec![ON_MISSION.to_string(), OFF_MISSION.to_string()],
        })
    }

    fn judge(&self, _view: &NpcView, answer: Option<&str>) -> Verdict {
        match answer {
            Some(a) if a.trim().eq_ignore_ascii_case(OFF_MISSION) => {
                Verdict::Concern(Concern::OffMission)
            }
            _ => Verdict::Healthy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::guardian::modules::fixtures::{view, with_mission};

    #[test]
    fn it_asks_nothing_of_a_character_with_no_mission() {
        assert_eq!(Drift.question(&view(None)), None);
    }

    #[test]
    fn the_question_carries_the_ask_and_both_answers() {
        let q = Drift.question(&with_mission(&["go there"])).unwrap();
        assert!(q.text.contains("Find the ledger"), "{}", q.text);
        assert_eq!(q.choices, vec![ON_MISSION, OFF_MISSION]);
    }

    #[test]
    fn only_something_else_is_a_concern() {
        let v = with_mission(&[]);
        assert_eq!(
            Drift.judge(&v, Some("Something else")),
            Verdict::Concern(Concern::OffMission)
        );
        assert_eq!(Drift.judge(&v, Some(ON_MISSION)), Verdict::Healthy);
        assert_eq!(Drift.judge(&v, None), Verdict::Healthy);
    }
}
