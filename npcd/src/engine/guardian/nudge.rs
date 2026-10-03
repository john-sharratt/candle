//! What a rung says to the character.
//!
//! Worded as the character's own thought and as a restatement of its errand,
//! never as an order to act in a particular way: the character decides what to
//! do about it.

use crate::engine::guardian::ladder::Rung;
use crate::engine::guardian::view::{words, Concern, MissionView, NpcView, Station};
use crate::engine::mission_acts::REPORT_STUCK;

/// The verb that puts a blocker on the record.
const STUCK_VERB: &str = REPORT_STUCK.name;

/// The words for `rung`, or `None` when the rung says nothing to the character
/// (a flag is for an operator) or has nothing to say (a restatement with no
/// mission to restate).
pub fn text(rung: Rung, concern: Concern, view: &NpcView) -> Option<String> {
    if rung == Rung::Flag {
        return None;
    }
    match (rung, view.mission.as_ref()) {
        (Rung::Flag, _) => None,
        (Rung::Refresh, Some(m)) => Some(m.standing.clone()),
        (Rung::Restate, Some(m)) => Some(format!(
            "Whatever else is asking for you, what you were sent to do still stands: {} Answer \
             whoever is waiting in a word, then get back to it.{}{}",
            m.prompt,
            next(m),
            at_hand(m, &view.stations)
        )),
        (Rung::Nudge, Some(m)) => Some(match concern {
            Concern::Looping => format!(
                "You keep covering the same ground. Stop, and come at what you were asked from \
                 another side: {}{}{} If what you need is not to be had here and nobody can give \
                 it, say so rather than going over it again: {}",
                m.prompt,
                next(m),
                at_hand(m, &view.stations),
                stuck_way(&view.stations)
            ),
            Concern::OffMission | Concern::Stalled => format!(
                "The errand you were given is still yours: {}{}{}",
                m.prompt,
                next(m),
                at_hand(m, &view.stations)
            ),
        }),
        (_, None) => match concern {
            Concern::Looping => Some(
                "You keep covering the same ground. Stop, and do something different: go \
                 somewhere else, or find somebody and talk."
                    .to_string(),
            ),
            Concern::OffMission | Concern::Stalled => None,
        },
    }
}

fn next(m: &MissionView) -> String {
    m.open_step()
        .map(|s| format!(" The next thing is: {s}"))
        .unwrap_or_default()
}

/// The station in the room whose name or verbs are what the open step (or, with
/// none, the ask) is about, named with the address to `invoke`.
fn at_hand(m: &MissionView, stations: &[Station]) -> String {
    let about = words(m.open_step().unwrap_or(&m.prompt));
    stations
        .iter()
        .find(|s| {
            let named = words(&format!("{} {}", s.name, s.verbs.join(" ")));
            !about.is_disjoint(&named)
        })
        .map(|s| {
            format!(
                " You can do it from the {}: `invoke` {}.",
                s.name,
                s.invokable().join(" or ")
            )
        })
        .unwrap_or_default()
}

/// How a character puts what stopped it on the record: the address to `invoke`
/// when the table that takes the report is in the room, otherwise where to go.
fn stuck_way(stations: &[Station]) -> String {
    stations
        .iter()
        .flat_map(Station::invokable)
        .find(|a| a.ends_with(STUCK_VERB))
        .map(|a| format!("`invoke` {a} with what stopped you."))
        .unwrap_or_else(|| {
            format!(
                "go back to the table where work is handed out, `scan` it, and `invoke` its \
                 `{STUCK_VERB}` with what stopped you."
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::guardian::modules::fixtures::view;
    use crate::engine::guardian::view::Step;

    fn mission() -> MissionView {
        MissionView {
            prompt: "Find the ledger.".to_string(),
            steps: vec![
                Step {
                    text: "go to the archive".into(),
                    done: true,
                    reports: false,
                },
                Step {
                    text: "read the ledger".into(),
                    done: false,
                    reports: false,
                },
            ],
            standing: "What has been asked of you: Find the ledger.".to_string(),
        }
    }

    fn seen(mission: Option<MissionView>) -> NpcView {
        view(mission)
    }

    fn console() -> Station {
        Station {
            name: "bridge console".into(),
            address: "http://local/tower/bridge~0".into(),
            verbs: vec!["command_tower".into()],
        }
    }

    fn at_the_console(step: &str) -> NpcView {
        let mut m = mission();
        m.steps[1].text = step.into();
        let mut v = seen(Some(m));
        v.stations = vec![console()];
        v
    }

    #[test]
    fn a_nudge_names_the_station_that_serves_the_open_step() {
        let v = at_the_console("raise the shields from the tower");
        let t = text(Rung::Nudge, Concern::OffMission, &v).unwrap();
        assert!(
            t.ends_with(
                " You can do it from the bridge console: `invoke` \
                 http://local/tower/bridge~0/command_tower."
            ),
            "{t}"
        );
    }

    #[test]
    fn a_restatement_names_it_too() {
        let v = at_the_console("command the tower");
        let t = text(Rung::Restate, Concern::OffMission, &v).unwrap();
        assert!(
            t.contains("`invoke` http://local/tower/bridge~0/command_tower"),
            "{t}"
        );
    }

    #[test]
    fn a_station_the_step_is_not_about_is_not_named() {
        let v = at_the_console("read the ledger");
        let t = text(Rung::Nudge, Concern::OffMission, &v).unwrap();
        assert!(!t.contains("invoke"), "{t}");
    }

    #[test]
    fn a_nudge_restates_the_ask_and_the_next_open_step() {
        let t = text(Rung::Nudge, Concern::OffMission, &seen(Some(mission()))).unwrap();
        assert_eq!(
            t,
            "The errand you were given is still yours: Find the ledger. The next thing is: read \
             the ledger"
        );
    }

    #[test]
    fn a_looping_nudge_asks_for_another_approach() {
        let t = text(Rung::Nudge, Concern::Looping, &seen(Some(mission()))).unwrap();
        assert!(t.starts_with("You keep covering the same ground."), "{t}");
        assert!(t.contains("Find the ledger."), "{t}");
    }

    #[test]
    fn a_looping_nudge_offers_the_way_to_say_what_stopped_it() {
        let t = text(Rung::Nudge, Concern::Looping, &seen(Some(mission()))).unwrap();
        assert!(
            t.ends_with("go back to the table where work is handed out, `scan` it, and `invoke` its `report_stuck` with what stopped you."),
            "{t}"
        );
    }

    #[test]
    fn with_the_table_in_the_room_the_nudge_names_its_address() {
        let mut v = seen(Some(mission()));
        v.stations = vec![Station {
            name: "table".into(),
            address: "http://local/command/order-table~0".into(),
            verbs: vec![
                "collect_mission".into(),
                "report_done".into(),
                "report_stuck".into(),
            ],
        }];
        let t = text(Rung::Nudge, Concern::Looping, &v).unwrap();
        assert!(
            t.ends_with(
                "`invoke` http://local/command/order-table~0/report_stuck with what stopped you."
            ),
            "{t}"
        );
    }

    #[test]
    fn a_restatement_names_the_pull_of_others_and_the_ask() {
        let t = text(Rung::Restate, Concern::OffMission, &seen(Some(mission()))).unwrap();
        assert!(t.starts_with("Whatever else is asking for you"), "{t}");
        assert!(t.contains("read the ledger"), "{t}");
    }

    #[test]
    fn a_refresh_is_the_standing_text_unchanged() {
        assert_eq!(
            text(Rung::Refresh, Concern::Stalled, &seen(Some(mission()))).unwrap(),
            "What has been asked of you: Find the ledger."
        );
    }

    #[test]
    fn a_flag_says_nothing_to_the_character() {
        assert_eq!(
            text(Rung::Flag, Concern::Looping, &seen(Some(mission()))),
            None
        );
    }

    #[test]
    fn without_a_mission_only_circling_has_anything_to_be_told() {
        let v = seen(None);
        assert!(text(Rung::Nudge, Concern::Looping, &v).is_some());
        assert!(text(Rung::Restate, Concern::Looping, &v).is_some());
        assert_eq!(text(Rung::Nudge, Concern::OffMission, &v), None);
        assert_eq!(text(Rung::Refresh, Concern::Stalled, &v), None);
    }

    #[test]
    fn a_mission_with_every_step_ticked_has_no_next_thing() {
        let mut m = mission();
        m.steps.iter_mut().for_each(|s| s.done = true);
        let t = text(Rung::Nudge, Concern::OffMission, &seen(Some(m))).unwrap();
        assert!(!t.contains("The next thing"), "{t}");
    }
}
