//! Signs off a step a character has dealt with and not recorded.
//!
//! Characters do not tick their own steps, so a mission's progress never moves
//! without this. The character says how a step went: it was carried out
//! (`achieved`), or it was tried and could not be (`thwarted`). A single answer
//! is not enough to act on — a character asked whether it has finished
//! something will sometimes say so — so the same answer has to come on
//! `confirmations` consecutive scans for the same step, and what it recently did
//! has to be about that step.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::engine::guardian::module::Module;
use crate::engine::guardian::view::{words, NpcView, Question, Verdict};
use crate::engine::mission::StepOutcome;

pub const DONE: &str = "yes, it is done";
pub const COULD_NOT: &str = "I tried and could not do it";
pub const NOT_YET: &str = "not yet";

pub struct StepTracker {
    confirmations: u32,
    /// Per character: the step being confirmed, the outcome claimed for it and
    /// how many scans in a row have said so.
    pending: Mutex<HashMap<u64, (String, StepOutcome, u32)>>,
}

impl StepTracker {
    pub fn new(confirmations: u32) -> Self {
        Self {
            confirmations: confirmations.max(1),
            pending: Mutex::new(HashMap::new()),
        }
    }
}

/// Whether anything the character did is about `step`: a claim of having done
/// it counts only when some act it took names what the step is about.
fn shown_by_acts(step: &str, acts: &[String]) -> bool {
    let about = words(step);
    acts.iter().any(|act| !about.is_disjoint(&words(act)))
}

impl Module for StepTracker {
    fn name(&self) -> &'static str {
        "step_tracker"
    }

    fn question(&self, view: &NpcView) -> Option<Question> {
        let mission = view.mission.as_ref()?;
        if mission.open_step_reports() {
            return None;
        }
        let step = mission.open_step()?;
        Some(Question {
            text: format!(
                "One thing you were asked to do is: \"{step}\" Looking back over what you have \
                 actually done, have you already done that, or tried and found you could not?"
            ),
            choices: vec![DONE.to_string(), COULD_NOT.to_string(), NOT_YET.to_string()],
        })
    }

    fn judge(&self, view: &NpcView, answer: Option<&str>) -> Verdict {
        let mut pending = self.pending.lock().unwrap();
        let mission = view.mission.as_ref();
        let step = mission
            .filter(|m| !m.open_step_reports())
            .and_then(|m| m.open_step());
        let (Some(step), Some(answer)) = (step, answer) else {
            if step.is_none() {
                pending.remove(&view.npc_id);
            }
            return Verdict::Healthy;
        };
        let claimed = if answer.trim().eq_ignore_ascii_case(DONE) {
            Some(StepOutcome::Achieved)
        } else if answer.trim().eq_ignore_ascii_case(COULD_NOT) {
            Some(StepOutcome::Thwarted)
        } else {
            None
        };
        let Some(outcome) = claimed.filter(|_| shown_by_acts(step, &view.recent_acts)) else {
            pending.remove(&view.npc_id);
            return Verdict::Healthy;
        };
        let seen = match pending.get(&view.npc_id) {
            Some((s, o, n)) if s == step && *o == outcome => n + 1,
            _ => 1,
        };
        if seen >= self.confirmations {
            pending.remove(&view.npc_id);
            return Verdict::TickStep(step.to_string(), outcome);
        }
        pending.insert(view.npc_id, (step.to_string(), outcome, seen));
        Verdict::Healthy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::guardian::modules::fixtures::{doing, view, with_mission};

    const WENT: &str = "move_to — over there";

    fn there() -> NpcView {
        doing(with_mission(&["go there"]), &[WENT])
    }

    #[test]
    fn it_asks_about_the_first_open_step_only() {
        let t = StepTracker::new(2);
        let q = t.question(&with_mission(&["go there", "read it"])).unwrap();
        assert!(q.text.contains("go there") && !q.text.contains("read it"));
        assert_eq!(
            q.choices,
            vec!["yes, it is done", "I tried and could not do it", "not yet"]
        );
        assert_eq!(t.question(&with_mission(&[])), None);
        assert_eq!(t.question(&view(None)), None);
    }

    #[test]
    fn the_report_step_is_never_asked_about_or_ticked() {
        let t = StepTracker::new(1);
        let mut v = doing(
            with_mission(&["report it"]),
            &["move_to — the table to report"],
        );
        v.mission.as_mut().unwrap().steps[0].reports = true;
        assert_eq!(t.question(&v), None);
        assert_eq!(t.judge(&v, Some(DONE)), Verdict::Healthy);
    }

    #[test]
    fn a_step_is_ticked_only_after_consecutive_confirmations() {
        let t = StepTracker::new(2);
        let v = there();
        assert_eq!(t.judge(&v, Some(DONE)), Verdict::Healthy);
        assert_eq!(
            t.judge(&v, Some(DONE)),
            Verdict::TickStep("go there".into(), StepOutcome::Achieved)
        );
        assert_eq!(
            t.judge(&v, Some(DONE)),
            Verdict::Healthy,
            "the count restarts"
        );
    }

    #[test]
    fn a_step_it_could_not_do_is_signed_off_thwarted_and_the_claims_do_not_mix() {
        let t = StepTracker::new(2);
        let v = there();
        assert_eq!(t.judge(&v, Some(COULD_NOT)), Verdict::Healthy);
        assert_eq!(
            t.judge(&v, Some(COULD_NOT)),
            Verdict::TickStep("go there".into(), StepOutcome::Thwarted)
        );
        // Done then could-not is two different claims, so neither has run twice.
        assert_eq!(t.judge(&v, Some(DONE)), Verdict::Healthy);
        assert_eq!(t.judge(&v, Some(COULD_NOT)), Verdict::Healthy);
    }

    #[test]
    fn a_claim_nothing_it_did_bears_out_is_not_ticked() {
        let t = StepTracker::new(1);
        let unrelated = doing(with_mission(&["go there"]), &["reflect — what next"]);
        assert_eq!(t.judge(&unrelated, Some(DONE)), Verdict::Healthy);
        assert_eq!(
            t.judge(&with_mission(&["go there"]), Some(DONE)),
            Verdict::Healthy
        );
    }

    #[test]
    fn a_not_yet_or_a_missing_answer_breaks_the_run() {
        let t = StepTracker::new(2);
        let v = there();
        assert_eq!(t.judge(&v, Some(DONE)), Verdict::Healthy);
        assert_eq!(t.judge(&v, Some(NOT_YET)), Verdict::Healthy);
        assert_eq!(t.judge(&v, Some(DONE)), Verdict::Healthy);
        assert_eq!(
            t.judge(&v, Some(DONE)),
            Verdict::TickStep("go there".into(), StepOutcome::Achieved)
        );
    }

    #[test]
    fn a_different_step_does_not_inherit_the_count() {
        let t = StepTracker::new(2);
        let acts = ["scan — the first", "scan — the second"];
        assert_eq!(
            t.judge(&doing(with_mission(&["scan first"]), &acts), Some(DONE)),
            Verdict::Healthy
        );
        assert_eq!(
            t.judge(&doing(with_mission(&["scan second"]), &acts), Some(DONE)),
            Verdict::Healthy
        );
    }

    #[test]
    fn one_confirmation_suffices_when_configured_so() {
        let t = StepTracker::new(1);
        assert_eq!(
            t.judge(&there(), Some(DONE)),
            Verdict::TickStep("go there".into(), StepOutcome::Achieved)
        );
    }
}
