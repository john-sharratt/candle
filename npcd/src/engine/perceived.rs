//! Turning what a body made out into what a character reads.
//!
//! [`npc_map`] answers what one body could perceive from where it was standing.
//! [`crate::engine::event`] answers how a thing that happened reads to a mind.
//! Both render prose, and this is the seam where it is decided which of them
//! renders what — because two renderers for one event is two voices, and the
//! character would hear the join.
//!
//! # Speech keeps its structure; everything else is condensed
//!
//! Two different jobs, so two different treatments.
//!
//! **Speech stays one event per utterance.** Who spoke and who they were aimed
//! at is a fact the console filters on, the relationship layer calibrates
//! against, and the scheduler weighs — flatten it into a sentence and all three
//! lose it. So a `Said` becomes an [`EventKind::Speech`] one for one, and
//! `event.rs` renders it in the narrator's voice like any other speech.
//!
//! **Everything else is condensed into one line.** A body crossing your field
//! of view generates a departure and an arrival, and four of them in a busy
//! corridor is eight events that read as a log. `npc_map::witness::narrate`
//! already collapses those into *Maker-02 left a station dark in band one and
//! left*, which is how a person recalls a minute rather than how a machine
//! reports one — so the movement half comes through as a single
//! [`EventKind::Description`].
//!
//! The split is not aesthetic. It is that speech has *structure worth keeping*
//! and movement has structure worth losing.

use npc_map::salience::weight;
use npc_map::witness::{narrate, Witnessed};
use npc_map::world::{Happening, World};

use crate::engine::event::{Addressed, EventKind, Salience};

/// One thing to put in a character's window, already weighed.
#[derive(Clone, Debug, PartialEq)]
pub struct Perceived {
    pub kind: EventKind,
    pub salience: Salience,
}

/// The situation a body is standing in, as a superseding event.
///
/// Always its own event rather than folded in with the news, because it is a
/// point in time and the news is a change over time — and because
/// [`EventKind::replaces`] is what keeps exactly one of them in the window.
pub fn situation(text: impl Into<String>) -> Perceived {
    Perceived {
        kind: EventKind::Situation { text: text.into() },
        // A situation says the world moved, not that anything wants answering.
        // Weighing it above idle would make every passing body an interruption.
        salience: Salience::IDLE,
    }
}

/// Everything a body made out, as events a character can read.
///
/// This is where an utterance's addressee is resolved into [`Addressed`] — the
/// one place both who was aimed at and who is receiving are in hand at once.
/// Neither has to be passed in: a [`Witnessed`] carries its own reader, so it
/// already knows whether it was the one spoken to.
pub fn digest(world: &World, seen: &[Witnessed]) -> Vec<Perceived> {
    let mut out: Vec<Perceived> = Vec::new();

    for w in seen {
        let Happening::Said { to, words } = &w.what else {
            continue;
        };
        out.push(Perceived {
            kind: EventKind::Speech {
                speaker: w.name.clone(),
                text: words.clone(),
                to: match to {
                    None => Addressed::Room,
                    Some(_) if w.addressed() => Addressed::You,
                    Some(other) => Addressed::Other {
                        who: named(world, other),
                    },
                },
            },
            salience: weight(w).into(),
        });
    }

    // The rest, as one line. `narrate` is given only what it is rendering, so
    // the speech above is not told twice in two voices.
    let rest: Vec<Witnessed> = seen
        .iter()
        .filter(|w| !matches!(w.what, Happening::Said { .. }))
        .cloned()
        .collect();
    if let Some(text) = narrate(world, &rest) {
        out.push(Perceived {
            kind: EventKind::Description { text },
            salience: rest
                .iter()
                .map(|w| Salience::from(weight(w)))
                .fold(Salience::IDLE, Salience::max_of),
        });
    }
    out
}

/// A third party's name, as this reader would say it.
///
/// The reader itself never reaches here — that case is [`Addressed::You`] — so
/// this only ever names somebody else, and falls back to the id for one who has
/// left the world between the utterance and the reading.
fn named(world: &World, id: &str) -> String {
    world
        .actor(id)
        .map(|a| a.name.clone())
        .unwrap_or_else(|| id.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use npc_map::witness::since;
    use npc_map::world::Where;
    use npc_map::MapSet;

    fn vault() -> World {
        World::new(
            MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"))
                .expect("the vault must load"),
        )
    }

    fn at(node: &str) -> Where {
        Where::new("vault-casting", node)
    }

    /// Three in the green room, all caught up.
    fn room() -> World {
        let mut w = vault();
        for (id, name) in [("m1", "Maker-01"), ("m2", "Maker-02"), ("m3", "Maker-03")] {
            w.enter(id, name, at("green-room")).unwrap();
        }
        for id in ["m1", "m2", "m3"] {
            w.mark_seen(id);
        }
        w
    }

    fn speech(p: &[Perceived]) -> Vec<&EventKind> {
        p.iter()
            .map(|p| &p.kind)
            .filter(|k| matches!(k, EventKind::Speech { .. }))
            .collect()
    }

    #[test]
    fn one_utterance_reads_three_ways_by_who_it_was_aimed_at() {
        let mut w = room();
        w.tell("m1", "m2", "get out of here").unwrap();

        let told = digest(&w, &since(&w, "m2"));
        let heard = digest(&w, &since(&w, "m3"));
        assert!(matches!(
            speech(&told)[0],
            EventKind::Speech {
                to: Addressed::You,
                ..
            }
        ));
        assert!(matches!(
            speech(&heard)[0],
            EventKind::Speech { to: Addressed::Other { who }, .. } if who == "Maker-02"
        ));

        let mut w = room();
        w.say("m1", "the redoubt burned twice").unwrap();
        let all = digest(&w, &since(&w, "m2"));
        assert!(matches!(
            speech(&all)[0],
            EventKind::Speech {
                to: Addressed::Room,
                ..
            }
        ));
    }

    #[test]
    fn the_three_readings_render_as_three_different_lines() {
        // The whole reason the distinction is carried: it has to survive into
        // what the character actually reads.
        let lines: Vec<String> = [
            Addressed::You,
            Addressed::Room,
            Addressed::Other {
                who: "Maker-02".into(),
            },
        ]
        .into_iter()
        .map(|to| EventKind::Speech {
            speaker: "Maker-01".into(),
            text: "get out of here".into(),
            to,
        })
        .map(|kind| crate::engine::event::Event::new(1, 0, Salience::NORMAL, kind).prose())
        .collect();
        assert_eq!(lines.len(), 3);
        for pair in [(0, 1), (0, 2), (1, 2)] {
            assert_ne!(lines[pair.0], lines[pair.1], "{lines:?}");
        }
        assert!(lines[0].contains("to you"), "{}", lines[0]);
        assert!(lines[2].contains("to Maker-02"), "{}", lines[2]);
    }

    #[test]
    fn being_addressed_weighs_more_than_overhearing_the_same_words() {
        let mut w = room();
        w.tell("m1", "m2", "get out of here").unwrap();

        let told = digest(&w, &since(&w, "m2"));
        let heard = digest(&w, &since(&w, "m3"));
        assert!(told[0].salience.get() > heard[0].salience.get());
        assert!(
            told[0].salience.preempts(),
            "being addressed did not preempt"
        );
        assert!(!heard[0].salience.preempts(), "overhearing preempted");
    }

    #[test]
    fn every_utterance_survives_as_its_own_event() {
        // Speech must not be condensed. Three things said is three things to
        // answer, and folding them into a sentence loses who said what to whom.
        let mut w = room();
        w.say("m1", "one").unwrap();
        w.tell("m2", "m3", "two").unwrap();
        w.say("m3", "three").unwrap();

        let p = digest(&w, &since(&w, "m1"));
        assert_eq!(speech(&p).len(), 2, "m1 said one of the three itself");
    }

    #[test]
    fn movement_is_condensed_into_one_line_rather_than_one_per_doorway() {
        let mut w = vault();
        w.enter("watch", "Maker-09", at("ring-north")).unwrap();
        for i in 1..=4 {
            w.enter(format!("m{i}"), format!("Maker-{i:02}"), at("band-one"))
                .unwrap();
        }
        w.mark_seen("watch");
        for i in 1..=4 {
            w.set_off(&format!("m{i}"), at("green-room")).unwrap();
        }
        w.settle();

        let p = digest(&w, &since(&w, "watch"));
        let described: Vec<&EventKind> = p
            .iter()
            .map(|p| &p.kind)
            .filter(|k| matches!(k, EventKind::Description { .. }))
            .collect();
        assert_eq!(
            described.len(),
            1,
            "four bodies produced {} lines",
            described.len()
        );
    }

    #[test]
    fn nothing_is_said_twice_in_two_voices() {
        // The join the module exists to prevent: an utterance rendered once as
        // speech and again inside the condensed line.
        let mut w = room();
        w.say("m1", "the redoubt burned twice").unwrap();
        w.set_off("m1", at("band-one")).unwrap();
        w.settle();

        let p = digest(&w, &since(&w, "m2"));
        let spoken = speech(&p).len();
        assert_eq!(spoken, 1);
        for it in &p {
            if let EventKind::Description { text } = &it.kind {
                assert!(
                    !text.contains("redoubt"),
                    "the utterance leaked into the condensed line: {text}"
                );
            }
        }
    }

    #[test]
    fn nothing_at_all_digests_to_nothing_at_all() {
        let w = room();
        assert!(digest(&w, &since(&w, "m1")).is_empty());
        assert!(digest(&w, &[]).is_empty());
    }

    #[test]
    fn a_situation_replaces_the_one_before_it_and_nothing_else_does() {
        let s = situation("You are in band one.");
        assert_eq!(s.kind.replaces().as_deref(), Some("situation"));
        assert!(
            !s.salience.preempts(),
            "a situation interrupted a character"
        );

        let mut w = room();
        w.say("m1", "something").unwrap();
        for it in digest(&w, &since(&w, "m2")) {
            assert!(it.kind.replaces().is_none(), "{:?} superseded", it.kind);
        }
    }

    #[test]
    fn a_condensed_line_carries_the_weight_of_the_loudest_thing_in_it() {
        // Somebody walking in is worth a turn; somebody walking past a doorway
        // two rooms away is not. Folded into one line, the line has to keep the
        // louder of the two or the arrival is silently demoted.
        let mut w = vault();
        w.enter("m1", "Maker-01", at("green-room")).unwrap();
        w.enter("m2", "Maker-02", at("ring-south")).unwrap();
        w.enter("m3", "Maker-03", at("core")).unwrap();
        w.mark_seen("m1");

        w.set_off("m3", at("ring-north")).unwrap(); // out of sight throughout
        w.set_off("m2", at("green-room")).unwrap(); // comes in
        w.settle();

        let p = digest(&w, &since(&w, "m1"));
        let line = p
            .iter()
            .find(|p| matches!(p.kind, EventKind::Description { .. }))
            .expect("something happened");
        assert_eq!(
            line.salience,
            Salience::from(npc_map::Weight::Wake),
            "the arrival was demoted by the traffic around it"
        );
    }
}
