//! Whether a character is going round in circles, judged from what it did.
//!
//! A character asked whether it is repeating itself will say so about a single
//! quiet turn, so the question is not put. The acts it took are the evidence:
//! one act taken again and again, word for word, while nothing on its mission
//! moved.

use std::collections::HashMap;

use crate::engine::guardian::module::Module;
use crate::engine::guardian::view::{Concern, NpcView, Question, Verdict};

/// How many of the latest acts are weighed.
const WINDOW: usize = 8;

/// How many times one act must stand in the window to be a circle.
const REPEATS: usize = 4;

/// Acts that are repeated on purpose: a fight is the same blow struck again.
const REPEATABLE: &[&str] = &["act"];

pub struct Looping;

/// An act as it was taken, without the result that came back after it.
fn taken(rendered: &str) -> &str {
    rendered.split(" → ").next().unwrap_or(rendered).trim()
}

fn name_of(act: &str) -> &str {
    act.split(" — ").next().unwrap_or(act)
}

impl Module for Looping {
    fn name(&self) -> &'static str {
        "looping"
    }

    fn question(&self, _view: &NpcView) -> Option<Question> {
        None
    }

    fn judge(&self, view: &NpcView, _answer: Option<&str>) -> Verdict {
        let progressing = view.mission.is_some() && view.since_progress.is_zero();
        if progressing {
            return Verdict::Healthy;
        }
        let skip = view.recent_acts.len().saturating_sub(WINDOW);
        let mut counts: HashMap<String, usize> = HashMap::new();
        for act in &view.recent_acts[skip..] {
            let act = taken(act);
            if REPEATABLE.contains(&name_of(act)) {
                continue;
            }
            *counts.entry(act.to_lowercase()).or_default() += 1;
        }
        if counts.values().any(|n| *n >= REPEATS) {
            Verdict::Concern(Concern::Looping)
        } else {
            Verdict::Healthy
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::engine::guardian::modules::fixtures::{doing, view, with_mission};

    const SCAN: &str = "scan — the table";
    const ASK: &str = "ask — Yaelis Vayne; where the data chips are";

    fn idle(mut v: NpcView) -> NpcView {
        v.since_progress = Duration::from_secs(90);
        v
    }

    #[test]
    fn it_puts_no_question() {
        assert_eq!(Looping.question(&view(None)), None);
    }

    #[test]
    fn one_act_taken_again_and_again_is_a_circle() {
        let v = doing(view(None), &[SCAN, SCAN, "move_to — the hall", SCAN, SCAN]);
        assert_eq!(Looping.judge(&v, None), Verdict::Concern(Concern::Looping));
    }

    #[test]
    fn the_same_act_with_different_results_is_still_the_same_act() {
        let v = doing(
            view(None),
            &[
                "scan — the table → a console",
                "scan — the table → a console, a lamp",
                "SCAN — the table",
                "scan — the table → nothing new",
            ],
        );
        assert_eq!(Looping.judge(&v, None), Verdict::Concern(Concern::Looping));
    }

    #[test]
    fn a_varied_run_of_acts_is_not_a_circle() {
        let v = doing(
            view(None),
            &[
                SCAN,
                ASK,
                "move_to — the hall",
                SCAN,
                "reflect — what next",
                SCAN,
                ASK,
            ],
        );
        assert_eq!(Looping.judge(&v, None), Verdict::Healthy);
    }

    #[test]
    fn a_few_repeats_are_not_a_circle() {
        let v = doing(view(None), &[SCAN, SCAN, SCAN]);
        assert_eq!(Looping.judge(&v, None), Verdict::Healthy);
    }

    #[test]
    fn repeating_while_the_mission_moves_is_work() {
        let v = doing(with_mission(&["scan the table"]), &[SCAN, SCAN, SCAN, SCAN]);
        assert_eq!(Looping.judge(&v, None), Verdict::Healthy);
        assert_eq!(
            Looping.judge(&idle(v), None),
            Verdict::Concern(Concern::Looping)
        );
    }

    #[test]
    fn a_fight_is_repetition_by_nature() {
        let blow = "act — strike the raider";
        let v = doing(view(None), &[blow, blow, blow, blow, blow]);
        assert_eq!(Looping.judge(&v, None), Verdict::Healthy);
    }

    #[test]
    fn only_the_latest_acts_count() {
        let mut acts = vec![SCAN; 4];
        acts.extend(["move_to — the hall", ASK, "reflect — what next"].repeat(3));
        let v = doing(view(None), &acts);
        assert_eq!(Looping.judge(&v, None), Verdict::Healthy);
    }

    #[test]
    fn a_character_with_no_acts_is_well() {
        assert_eq!(Looping.judge(&view(None), None), Verdict::Healthy);
    }
}
