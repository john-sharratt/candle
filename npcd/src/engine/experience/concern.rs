//! Which journals hear of a change in the world (`docs/npc_conversations.md`
//! §5.1.1): a conversation knows only what its journal tells it, so a change
//! the NPC did not make and was not in the room for, but which touches it or
//! its party, is a line in its journal.
//!
//! Run where the change is applied — the world's authority — right after it.
//! What is never written: ambient percepts, rooms that did not change, and
//! the comings and goings of characters it has no conversation with. Those
//! stay in the world log, where the main loop's perception reads them.

use crate::sim::Sim;

/// A change, as the journals are told of it: what happened, in a line a
/// reader with no world in front of them can follow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Change {
    /// `body`'s own mission was lodged, assigned, withdrawn, stepped or
    /// authorized.
    Mission { body: String, line: String },
    /// Something in a room changed — a building op, or a device worked by
    /// `by` — which concerns whoever's standing mission is for that room.
    Room {
        place: String,
        by: Option<String>,
        line: String,
    },
    /// Something happened to `member` — a person's visitor body arrived or
    /// left, a mission they lodged changed — which concerns every character
    /// with a conversation with them.
    Party { member: String, line: String },
    /// The world or the house as a whole: a house mode set, the day rolling
    /// over, an operator's reshape. Written once, in `bots/world`.
    World { line: String },
}

/// Who is to hear of a change.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Hears {
    /// A bot, by body.
    Bot(String),
    /// The world's own journal, `bots/world`.
    World,
}

/// The journals `change` concerns, each with the line it is told, in a
/// stable order.
pub fn concern(change: &Change, sim: &Sim) -> Vec<(Hears, String)> {
    let mut out: Vec<(Hears, String)> = match change {
        Change::Mission { body, line } => vec![(Hears::Bot(body.clone()), line.clone())],
        Change::Room { place, by, line } => sim
            .missions
            .carried()
            .filter(|(body, m)| m.room() == Some(place.as_str()) && by.as_deref() != Some(*body))
            .map(|(body, _)| (Hears::Bot(body.to_string()), line.clone()))
            .collect(),
        // A person's direct threads are each with a character: every one of
        // them is a conversation with them.
        Change::Party { member, line } => sim
            .threads
            .direct_of(member)
            .into_iter()
            .flat_map(|t| t.other_ids(member))
            .map(|other| (Hears::Bot(other), line.clone()))
            .collect(),
        Change::World { line } => vec![(Hears::World, line.clone())],
    };
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mission::{Mission, Origin, Todo};
    use crate::sim::phone::Who;

    fn sim() -> Sim {
        let mut s = Sim::default();
        let zen = Who::new("zen", "Zen");
        let wick = Who::new("wick", "Wick");
        let mira = Who::new("u_mira", "Mira");
        s.threads.reach(&mira, &zen);
        s.threads.reach(&mira, &wick);
        s.threads.reach(&zen, &wick);
        let tidy = Mission::new(
            "Keep the lounge tidy.",
            vec![Todo::new("tidy the lounge")],
            Origin::Standing {
                id: "tidy-lounge".into(),
                room: "house/lounge".into(),
            },
        );
        s.missions.assign("wick", tidy);
        s
    }

    fn bot(b: &str, line: &str) -> (Hears, String) {
        (Hears::Bot(b.into()), line.into())
    }

    #[test]
    fn a_mission_change_concerns_its_carrier() {
        let c = Change::Mission {
            body: "zen".into(),
            line: "John withdrew 'Tidy the lounge'.".into(),
        };
        assert_eq!(
            concern(&c, &sim()),
            [bot("zen", "John withdrew 'Tidy the lounge'.")]
        );
    }

    #[test]
    fn a_room_change_concerns_whoever_s_mission_is_for_it_unless_they_made_it() {
        let c = |by: Option<&str>| Change::Room {
            place: "house/lounge".into(),
            by: by.map(String::from),
            line: "The lounge TV was turned off.".into(),
        };
        assert_eq!(
            concern(&c(Some("zen")), &sim()),
            [bot("wick", "The lounge TV was turned off.")]
        );
        assert!(concern(&c(Some("wick")), &sim()).is_empty());
        let elsewhere = Change::Room {
            place: "house/kitchen".into(),
            by: None,
            line: "x".into(),
        };
        assert!(concern(&elsewhere, &sim()).is_empty());
    }

    #[test]
    fn a_party_s_change_concerns_everyone_in_conversation_with_them() {
        let c = Change::Party {
            member: "u_mira".into(),
            line: "Mira arrived in the lounge.".into(),
        };
        assert_eq!(
            concern(&c, &sim()),
            [
                bot("wick", "Mira arrived in the lounge."),
                bot("zen", "Mira arrived in the lounge.")
            ]
        );
    }

    #[test]
    fn a_world_change_is_written_once_to_the_world() {
        let c = Change::World {
            line: "The house was set to away mode.".into(),
        };
        assert_eq!(
            concern(&c, &sim()),
            [(Hears::World, "The house was set to away mode.".to_string())]
        );
    }
}
