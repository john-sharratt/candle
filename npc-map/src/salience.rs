//! How much a thing that happened is worth.
//!
//! Two questions get asked of every change: whether it is worth a turn now,
//! and how hard it should pull once it is in the working set. This answers
//! both, and it answers them **from the reader's side** — the same utterance
//! is a demand to the one it was aimed at and a piece of gossip to everybody
//! else in the room, so weight is a property of [`Witnessed`], never of
//! [`Happening`] alone.
//!
//! # A ladder, not a number
//!
//! The levels are named for what a consumer does with them, so a weight can be
//! read off without knowing anything about the scale. That is deliberate: a
//! float invites tuning, and a tuned float is a decision nobody can find later.
//! Where a number is genuinely wanted — the `salience` field on the perception
//! API is one — [`Weight::as_f32`] renders the ladder, and the rendering is
//! monotone by test so the two can never disagree about order.
//!
//! # Where the ordering comes from
//!
//! Not from taste. Every rung falls out of a rule the world already enforces:
//!
//! - **Being addressed** is the loudest social event there is, and the world
//!   already distinguishes it — direction is carried on the utterance, and
//!   delivery is by place, so *told you* and *told somebody else in front of
//!   you* are already two different facts about one event.
//! - **Your own outcome** is the answer to a question you asked several turns
//!   ago. A body that does not act on arriving stands in a doorway.
//! - **Nothing out of the room** can be loud, because sight is deliberately
//!   poor: you can see a body move and a console light, never what either is
//!   doing. There is nothing in that to act on urgently.

use crate::witness::Witnessed;
use crate::world::Happening;

/// What a change is worth, named for what to do about it.
///
/// Ordered ascending, so `>=` reads the way it looks and
/// [`Weight::PREEMPTS`] is a comparison rather than a list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Weight {
    /// Somewhere else. Worth having in the working set, worth nothing else.
    Ambient,
    /// Here, and worth remembering. Not worth interrupting for.
    Note,
    /// Here, and worth a turn at the next opportunity.
    Wake,
    /// Worth a turn *now*, ahead of whatever was planned.
    Preempt,
}

impl Weight {
    /// The rung at which a change stops waiting its turn.
    pub const PREEMPTS: Weight = Weight::Preempt;

    /// The ladder as a number, for the API surfaces that take one.
    ///
    /// A rendering of the order, not a second source of truth — a test asserts
    /// it is monotone, so the two cannot drift apart.
    pub fn as_f32(self) -> f32 {
        match self {
            Weight::Ambient => 0.1,
            Weight::Note => 0.3,
            Weight::Wake => 0.6,
            Weight::Preempt => 1.0,
        }
    }
}

/// What one change is worth to the body that made it out.
pub fn weight(w: &Witnessed) -> Weight {
    // Only an outcome of the reader's own ever reaches the reader — setting
    // off, sitting down and speaking are filtered on the way out, because a
    // body knows what it just did. What is left is the answer to something it
    // set in motion, and it is always worth acting on.
    if w.mine() {
        return Weight::Preempt;
    }
    // Sight is deliberately poor. A body moving in the next room and a console
    // lighting up are worth knowing and never worth interrupting for.
    if !w.here {
        return Weight::Ambient;
    }
    match &w.what {
        // Aimed at you. Everything else in the room heard it too, and for them
        // it is one rung down.
        Happening::Said { .. } if w.addressed() => Weight::Preempt,
        // Said to the room is said to you among others.
        Happening::Said { to: None, .. } => Weight::Wake,
        // Aimed past you. Information, not a demand.
        Happening::Said { .. } => Weight::Note,
        // A gesture aimed at you is somebody addressing you without words, and
        // waiting for an answer — the same demand speech makes, one rung down
        // because it can be met with a look rather than a reply.
        Happening::Did { .. } if w.addressed() => Weight::Wake,
        // Something somebody did in the room. Worth knowing, never a summons:
        // a room where every shifted chair interrupted everyone would leave
        // nobody able to finish a thought.
        Happening::Did { .. } => Weight::Note,
        // The building carries its own. A vent changing note and a breaker
        // going are the same kind of event and are worth entirely different
        // amounts, so the thing that knows which it was says so.
        Happening::Stirred { weight, .. } => *weight,
        // Somebody came in. You look up.
        Happening::Arrived => Weight::Wake,
        // Somebody left, or the room's furniture changed state — both of which
        // the percept already reports, so as *events* they are only history.
        Happening::Left | Happening::TookStation { .. } | Happening::LeftStation { .. } => {
            Weight::Note
        }
        // Private to the body they happened to, so they arrive only as the
        // reader's own and are caught above. Classified rather than ignored so
        // that a new happening cannot be added without a weight being chosen.
        Happening::SetOut { .. } | Happening::GotThere { .. } | Happening::LostTheWay { .. } => {
            Weight::Note
        }
    }
}

/// Whether one change is worth a turn now.
pub fn preempts(w: &Witnessed) -> bool {
    weight(w) >= Weight::PREEMPTS
}

/// The loudest of a run of changes, or nothing if there were none.
///
/// What a tick gate asks: not what happened, but whether any of it was worth
/// stopping for.
pub fn loudest(seen: &[Witnessed]) -> Option<Weight> {
    seen.iter().map(weight).max()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::MapSet;
    use crate::witness::since;
    use crate::world::{Where, World};

    fn vault() -> World {
        World::new(
            MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/maps"))
                .expect("the vault must load"),
        )
    }

    fn at(node: &str) -> Where {
        Where::new("vault-casting", node)
    }

    /// Three in the green room, one out on the ring that can see into it.
    fn room() -> World {
        let mut w = vault();
        for (id, name, place) in [
            ("m1", "Maker-01", "green-room"),
            ("m2", "Maker-02", "green-room"),
            ("m3", "Maker-03", "green-room"),
            ("out", "Maker-09", "ring-south"),
        ] {
            w.enter(id, name, at(place)).unwrap();
        }
        for id in ["m1", "m2", "m3", "out"] {
            w.mark_seen(id);
        }
        w
    }

    fn only(world: &World, id: &str) -> Weight {
        let seen = since(world, id);
        assert_eq!(seen.len(), 1, "expected exactly one change for {id}");
        weight(&seen[0])
    }

    // -- the ladder itself -------------------------------------------------

    #[test]
    fn the_rungs_are_ordered_the_way_they_are_named() {
        assert!(Weight::Ambient < Weight::Note);
        assert!(Weight::Note < Weight::Wake);
        assert!(Weight::Wake < Weight::Preempt);
    }

    #[test]
    fn the_number_is_a_rendering_of_the_order_and_stays_monotone() {
        let ladder = [Weight::Ambient, Weight::Note, Weight::Wake, Weight::Preempt];
        for pair in ladder.windows(2) {
            assert!(pair[0] < pair[1]);
            assert!(
                pair[0].as_f32() < pair[1].as_f32(),
                "{:?} renders no lower than {:?}",
                pair[0],
                pair[1]
            );
        }
        for rung in ladder {
            let n = rung.as_f32();
            assert!((0.0..=1.0).contains(&n), "{rung:?} renders to {n}");
        }
    }

    #[test]
    fn only_the_top_rung_preempts() {
        assert!(Weight::Preempt >= Weight::PREEMPTS);
        for rung in [Weight::Ambient, Weight::Note, Weight::Wake] {
            assert!(rung < Weight::PREEMPTS, "{rung:?} preempts");
        }
    }

    // -- one event, two readers --------------------------------------------

    #[test]
    fn one_utterance_is_worth_more_to_the_one_it_was_aimed_at() {
        // The property the whole module exists for. Same event, same room,
        // same instant — two weights, because weight is read from the reader's
        // side and the world already knows which of them was addressed.
        let mut w = room();
        w.tell("m1", "m2", "get out of here").unwrap();

        assert_eq!(only(&w, "m2"), Weight::Preempt);
        assert_eq!(only(&w, "m3"), Weight::Note);
        assert!(preempts(&since(&w, "m2")[0]));
        assert!(!preempts(&since(&w, "m3")[0]));
    }

    #[test]
    fn speech_to_the_room_sits_between_being_told_and_overhearing() {
        let mut w = room();
        w.say("m1", "the redoubt burned twice").unwrap();
        let to_the_room = only(&w, "m2");

        let mut w = room();
        w.tell("m1", "m2", "the redoubt burned twice").unwrap();
        let told = only(&w, "m2");
        let overheard = only(&w, "m3");

        assert!(overheard < to_the_room, "{overheard:?} !< {to_the_room:?}");
        assert!(to_the_room < told, "{to_the_room:?} !< {told:?}");
    }

    // -- your own doings ---------------------------------------------------

    #[test]
    fn arriving_where_you_meant_to_go_is_worth_a_turn_now() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("band-one")).unwrap();
        w.set_off("m1", at("green-room")).unwrap();
        w.mark_seen("m1");
        w.settle();
        assert_eq!(only(&w, "m1"), Weight::Preempt);
    }

    #[test]
    fn losing_a_journey_is_worth_a_turn_now() {
        // Your plan is void and nothing else will say so.
        let mut w = vault();
        w.enter("m1", "Maker-01", at("band-one")).unwrap();
        w.set_off("m1", Where::new("vault-command", "command-room"))
            .unwrap();
        w.mark_seen("m1");
        w.tick();
        w.set_off("m1", at("green-room")).unwrap();
        assert_eq!(only(&w, "m1"), Weight::Preempt);
    }

    #[test]
    fn nothing_a_body_merely_did_reaches_it_to_be_weighed() {
        // The rule the outcomes are an exception to: a body is told nothing
        // about what it already knows it did.
        let mut w = room();
        w.mark_seen("m1");
        w.say("m1", "to the room").unwrap();
        w.tell("m1", "m2", "and to you").unwrap();
        w.set_off("m1", at("band-one")).unwrap();
        assert!(since(&w, "m1").is_empty());
        assert_eq!(loudest(&since(&w, "m1")), None);
    }

    // -- the room, and everywhere that is not the room ---------------------

    #[test]
    fn somebody_coming_in_is_worth_a_turn_and_somebody_leaving_is_not() {
        // Both are two changes rather than one, because the green room can see
        // the run outside it — so the question is what the *run* is worth,
        // which is what a gate asks anyway.
        let mut w = room();
        w.set_off("out", at("green-room")).unwrap();
        w.settle();
        assert_eq!(loudest(&since(&w, "m1")), Some(Weight::Wake));

        let mut w = room();
        w.set_off("m2", at("ring-south")).unwrap();
        w.settle();
        assert_eq!(loudest(&since(&w, "m1")), Some(Weight::Note));
    }

    #[test]
    fn coming_in_and_going_out_are_weighed_apart_even_in_one_run() {
        // The rungs the test above reads through a maximum, asserted directly.
        let mut w = room();
        w.set_off("out", at("green-room")).unwrap();
        w.settle();

        let seen = since(&w, "m1");
        let arrived = seen
            .iter()
            .find(|s| s.here && s.what == Happening::Arrived)
            .expect("nobody came in");
        assert_eq!(weight(arrived), Weight::Wake);

        let mut w = room();
        w.set_off("m2", at("ring-south")).unwrap();
        w.settle();
        let seen = since(&w, "m1");
        let left = seen
            .iter()
            .find(|s| s.here && s.what == Happening::Left)
            .expect("nobody left");
        assert_eq!(weight(left), Weight::Note);
    }

    #[test]
    fn the_furniture_changing_state_is_history_because_the_percept_has_it() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("band-one")).unwrap();
        w.enter("m2", "Maker-02", at("band-one")).unwrap();
        w.mark_seen("m2");

        w.take("m1", Some("cindy")).unwrap();
        assert_eq!(only(&w, "m2"), Weight::Note);

        w.mark_seen("m2");
        w.release("m1").unwrap();
        assert_eq!(only(&w, "m2"), Weight::Note);
    }

    #[test]
    fn nothing_seen_from_another_room_is_ever_more_than_ambient() {
        // Exhaustive over what can carry out of a room. Sight is poor on
        // purpose, and no rung above ambient may be reachable through it —
        // the moment one is, a body has a reason not to walk anywhere.
        let mut w = room();
        w.mark_seen("out");
        w.say("m1", "not heard outside").unwrap();
        w.tell("m1", "m2", "nor this").unwrap();
        assert!(since(&w, "out").is_empty());

        // What does carry: bodies moving, and consoles lighting up.
        let mut w = vault();
        w.enter("m1", "Maker-01", at("ring-north")).unwrap();
        w.enter("m2", "Maker-02", at("band-one")).unwrap();
        w.mark_seen("m1");
        w.take("m2", Some("cindy")).unwrap();
        w.release("m2").unwrap();
        w.set_off("m2", at("green-room")).unwrap();
        w.settle();

        let seen = since(&w, "m1");
        assert!(!seen.is_empty(), "nothing carried at all");
        for s in &seen {
            assert_eq!(weight(s), Weight::Ambient, "{:?} carried loudly", s.what);
            assert!(!preempts(s));
        }
    }

    // -- runs of changes ---------------------------------------------------

    #[test]
    fn the_loudest_of_a_run_is_what_a_gate_asks_about() {
        let mut w = room();
        w.say("m3", "somebody should look at the redoubt").unwrap();
        w.tell("m1", "m2", "you have it").unwrap();
        w.tell("m1", "m3", "and you have the gap").unwrap();

        // m2 was addressed once among three; the run is a preempt.
        let mine = since(&w, "m2");
        assert!(mine.len() > 1);
        assert_eq!(loudest(&mine), Some(Weight::Preempt));

        // A body that only overheard the same run is not interrupted by it.
        let mut w = room();
        w.tell("m1", "m2", "you have it").unwrap();
        assert_eq!(loudest(&since(&w, "m3")), Some(Weight::Note));
    }

    #[test]
    fn nothing_at_all_is_not_a_quiet_something() {
        assert_eq!(loudest(&[]), None);
    }
}
