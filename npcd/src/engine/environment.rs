//! Carrying the world to the minds standing in it.
//!
//! Everything either side of this exists: a world that knows who is where, and
//! a scheduler that knows whose turn it is. This is the only thing between
//! them, and it is the one piece of the design with no precedent in the tree.
//!
//! # Placed, not selected
//!
//! What a mind is handed here is **environment-placed**: written into its window
//! at a fixed position, never scored, never competing for budget. That is a
//! third residency class beside the immutable prefix (never scored, never
//! changes) and the gathered substrate (scored, accumulates).
//!
//! It has to be. A percept is not a memory to retrieve — a character that
//! failed to notice the room it is standing in because something scored higher
//! would be broken, not selective.
//!
//! # Pushed, in one pass, after the world moves
//!
//! Perception is prefill and action is decode. Pushing every body's delta in
//! one sweep is what makes the first half true: sixteen minds are told what
//! changed in one pass over one lock, off the decode path entirely. Pulling
//! instead — each mind computing its own situation inside its own turn — would
//! serialise the cheap half behind the expensive one.
//!
//! And most of that sweep is free. A percept is a pure function, so an
//! unchanged situation renders byte-identically and is not sent at all: a body
//! nothing happened to costs nothing, which is what pays for a mind that never
//! blocks.
//!
//! # Two halves, and only one of them supersedes
//!
//! A delta carries a **situation** and a run of **changes**. The situation is
//! the whole of what is true where the body stands, so a second one replaces
//! the first — two in a window is a stale reading of a room the character has
//! left, sitting where attention weights it highest. The changes accumulate,
//! because a thing that happened stays happened.

use npc_map::delta::Delta;
use npc_map::world::World;

use crate::engine::perceived::{digest, situation, Perceived};
use crate::engine::reach;
use crate::engine::tick::Scheduler;
use crate::engine::tools::{self, Mode};
use crate::world::binding::Bindings;
use crate::world::Hosted;

/// A delta as the events a mind receives — the situation, then what happened.
///
/// The order is both how it reads and how it caches: the situation leads, so an
/// unchanged one that *is* re-sent still matches the tokens above it, and the
/// volatile half is appended after that boundary rather than through it.
pub fn carried(world: &World, delta: &Delta) -> Vec<Perceived> {
    let mut out = Vec::new();
    if let Some(here) = &delta.percept {
        // What is within reach goes *in* the situation rather than beside it.
        // Both are functions of where the body stands, so they change together
        // and go stale together — and one band holding both is what guarantees
        // a character never reads the tools of a room it has left.
        let mut text = here.trim_end().to_string();
        if let Some(reach) = reach::line(world, &delta.who) {
            text.push_str("\n\n");
            text.push_str(&reach);
        }
        // What company makes possible, offered here rather than in the prompt
        // because who is standing next to you changes every tick and the prompt
        // is written once. Absent when alone, so a character with nobody to
        // address is never shown a way to address somebody.
        if let Some(near) = company_line(world, &delta.who) {
            text.push_str("\n\n");
            text.push_str(&near);
        }
        out.push(situation(text));
    }
    out.extend(digest(world, &delta.events));
    out
}

/// What a body can do because somebody else is in the room, as a line it reads.
///
/// Nothing when alone — and that emptiness is the mechanism, not an omission.
/// A character shown a way to address people while there is nobody to address
/// will address somebody, because a model handed a field fills it in.
fn company_line(world: &World, body: &str) -> Option<String> {
    let here = world.actor(body)?.at.clone();
    let others: Vec<String> = world
        .actors_at(&here)
        .into_iter()
        .filter(|a| a.id != body)
        .map(|a| a.name.clone())
        .collect();
    if others.is_empty() {
        return None;
    }
    let tools: Vec<&str> = tools::nearby(Mode::Physical)
        .into_iter()
        .map(|t| t.name)
        .collect();
    if tools.is_empty() {
        return None;
    }
    Some(format!(
        "Because {} {} here with you, you can also use: {}. Name them exactly as written.",
        npc_map::text::list(&others),
        if others.len() == 1 { "is" } else { "are" },
        tools.join(", "),
    ))
}

/// Hand one mind what its body has to be told, and say whether there was any.
///
/// The delta is spent whether or not the mind exists in the scheduler: a
/// character retired between the sweep being planned and run must not leave its
/// body's cursor behind to replay the same minute for ever.
pub fn push_one(hosted: &Hosted, sched: &Scheduler, npc_id: u64, body: &str) -> bool {
    let delta = hosted.delta(body);
    if delta.is_empty() {
        return false;
    }
    let (events, at_ms) = hosted.read(|w| (carried(w, &delta), w.now()));
    for Perceived { kind, salience } in events {
        sched.deliver(npc_id, at_ms, salience, kind);
    }
    true
}

/// Hand every mind in a world what its body has to be told.
///
/// Returns how many minds got something. Bodies with no mind are swept anyway —
/// their cursor has to advance, or the day they are finally bound to one they
/// would be told a week of news at once.
pub fn push(hosted: &Hosted, bindings: &Bindings, sched: &Scheduler) -> usize {
    let deltas = hosted.sweep();
    if deltas.is_empty() {
        return 0;
    }
    let at_ms = hosted.read(|w| w.now());
    let mut told = 0;
    for delta in &deltas {
        let Some(npc_id) = bindings.mind_of(hosted.id(), &delta.who) else {
            continue;
        };
        let events = hosted.read(|w| carried(w, delta));
        if events.is_empty() {
            continue;
        }
        for Perceived { kind, salience } in events {
            sched.deliver(npc_id, at_ms, salience, kind);
        }
        told += 1;
    }
    told
}

/// One moment of world time, and everything that follows from it.
///
/// Journeys advance first and perception is taken after, because a body that
/// moved this moment has to be told where it ended up rather than where it set
/// out from. Both under the world's own lock, in that order, always.
pub fn advance(hosted: &Hosted, bindings: &Bindings, sched: &Scheduler) -> Moment {
    let moved = hosted.tick();
    let told = push(hosted, bindings, sched);
    Moment { moved, told }
}

/// What one moment of world time came to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Moment {
    /// Bodies that covered a leg.
    pub moved: usize,
    /// Minds that were told something.
    pub told: usize,
}

impl Moment {
    /// Nothing moved and nobody was told. What most moments are, and what makes
    /// a large cast affordable.
    pub fn is_quiet(&self) -> bool {
        self.moved == 0 && self.told == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::event::EventKind;
    use npc_map::world::Where;

    fn vault() -> Hosted {
        Hosted::load(
            "creators-vault",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
        )
        .expect("the shipped vault must load")
    }

    fn at(node: &str) -> Where {
        Where::new("vault-casting", node)
    }

    /// A world with `n` Makers in the green room, each bound to a mind, all
    /// caught up and grounded.
    fn crew(n: usize) -> (Hosted, Bindings, Scheduler) {
        let h = vault();
        let b = Bindings::new();
        let s = Scheduler::new(256);
        for i in 1..=n {
            let body = format!("m{i}");
            let npc_id = 100 + i as u64;
            h.with(|w| {
                w.enter(&body, format!("Maker-{i:02}"), at("green-room"))
                    .unwrap()
            });
            b.bind(npc_id, h.id(), &body).unwrap();
            s.wake(npc_id, 0, 0);
        }
        push(&h, &b, &s);
        (h, b, s)
    }

    /// What one mind is holding in its window.
    fn window(s: &Scheduler, npc_id: u64) -> Vec<String> {
        s.window_of(npc_id, |win| {
            win.turns().map(|t| t.text.clone()).collect::<Vec<_>>()
        })
        .unwrap_or_default()
    }

    /// Run every mind that is due, and give back what each read.
    fn run(s: &Scheduler, at_ms: u64) -> Vec<(u64, Vec<String>)> {
        s.due_now(at_ms)
            .into_iter()
            .filter_map(|id| {
                s.tick(id, at_ms, at_ms, |_, _| Vec::new())
                    .map(|r| (id, r.perceived))
            })
            .collect()
    }

    /// What a mind is holding after it has taken its turn.
    ///
    /// Pushing fills an inbox; a window fills when the mind is scheduled and
    /// drains it. The two are deliberately separate — perception is cheap and
    /// arrives whenever, thinking is expensive and happens on a tick — so a
    /// test that reads a window without letting the mind run is reading the
    /// wrong side of that line.
    fn read_window(s: &Scheduler, npc_id: u64) -> Vec<String> {
        run(s, 10_000_000);
        window(s, npc_id)
    }

    #[test]
    fn a_body_is_grounded_the_first_time_and_left_alone_after() {
        let h = vault();
        let b = Bindings::new();
        let s = Scheduler::new(64);
        h.with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());
        b.bind(1, h.id(), "m1").unwrap();
        s.wake(1, 0, 0);

        assert_eq!(push(&h, &b, &s), 1);
        assert!(read_window(&s, 1).iter().any(|t| t.starts_with("You are")));
        for _ in 0..10 {
            assert_eq!(push(&h, &b, &s), 0, "an idle body cost something");
        }
    }

    #[test]
    fn a_body_with_no_mind_still_has_its_cursor_advanced() {
        // Otherwise the day it is bound to one, it is told a week at once.
        let h = vault();
        let b = Bindings::new();
        let s = Scheduler::new(64);
        h.with(|w| {
            w.enter("extra", "An Extra", at("green-room")).unwrap();
            w.enter("m1", "Maker-01", at("green-room")).unwrap();
        });
        b.bind(1, h.id(), "m1").unwrap();
        s.wake(1, 0, 0);

        push(&h, &b, &s);
        h.with(|w| w.say("m1", "something").unwrap());
        assert_eq!(push(&h, &b, &s), 0, "only the extra could have heard it");
        // The extra's cursor moved with everyone else's, so binding it now
        // would start it at the present rather than at the beginning.
        assert!(h.peek("extra").is_empty(), "{:?}", h.peek("extra"));
    }

    #[test]
    fn the_situation_arrives_before_what_happened_in_it() {
        let (h, b, s) = crew(2);
        h.with(|w| {
            w.say("m1", "the redoubt burned twice").unwrap();
            w.set_off("m1", at("band-one")).unwrap();
        });
        h.tick();
        push(&h, &b, &s);

        let turns = read_window(&s, 102);
        let here = turns.iter().position(|t| t.starts_with("You are"));
        let news = turns.iter().position(|t| t.contains("redoubt"));
        assert!(here.is_some() && news.is_some(), "{turns:?}");
        assert!(here < news, "{turns:?}");
    }

    #[test]
    fn a_mind_is_only_ever_standing_in_one_place() {
        let (h, b, s) = crew(1);
        for room in ["band-one", "relations", "watch", "core"] {
            h.with(|w| w.set_off("m1", at(room)).unwrap());
            h.tick();
            push(&h, &b, &s);
        }
        let standing: Vec<String> = read_window(&s, 101)
            .into_iter()
            .filter(|t| t.starts_with("You are"))
            .collect();
        assert_eq!(standing.len(), 1, "{standing:?}");
        assert!(standing[0].contains("lift"), "{standing:?}");
    }

    #[test]
    fn a_moment_that_moves_nobody_and_tells_nobody_is_quiet() {
        let (h, b, s) = crew(3);
        let m = advance(&h, &b, &s);
        assert_eq!(m, Moment { moved: 0, told: 0 });
        assert!(m.is_quiet());
    }

    #[test]
    fn a_journey_advances_before_perception_is_taken() {
        // The ordering that matters: a body told where it set out from would be
        // acting on a room it has already left.
        let (h, b, s) = crew(1);
        h.with(|w| w.set_off("m1", at("band-one")).unwrap());

        let m = advance(&h, &b, &s);
        assert_eq!(m.moved, 1);
        assert_eq!(m.told, 1);

        let here = read_window(&s, 101)
            .into_iter()
            .rfind(|t| t.starts_with("You are"))
            .expect("grounded");
        assert!(here.contains("band one"), "{here}");
    }

    #[test]
    fn being_spoken_to_wakes_a_mind_and_overhearing_does_not() {
        let (h, b, s) = crew(3);
        run(&s, 0);
        h.with(|w| w.tell("m1", "m2", "get out of here").unwrap());
        push(&h, &b, &s);

        let woken = run(&s, 1);
        let ids: Vec<u64> = woken.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&102), "the addressed mind was not woken");
        assert!(!ids.contains(&103), "the bystander was woken");
    }

    #[test]
    fn a_mind_in_another_room_is_told_nothing_that_was_said() {
        let h = vault();
        let b = Bindings::new();
        let s = Scheduler::new(64);
        h.with(|w| {
            w.enter("m1", "Maker-01", at("green-room")).unwrap();
            w.enter("out", "Maker-09", at("ring-south")).unwrap();
        });
        b.bind(1, h.id(), "m1").unwrap();
        b.bind(9, h.id(), "out").unwrap();
        s.wake(1, 0, 0);
        s.wake(9, 0, 0);
        push(&h, &b, &s);

        h.with(|w| w.say("m1", "the redoubt burned twice").unwrap());
        push(&h, &b, &s);
        assert!(
            !window(&s, 9).iter().any(|t| t.contains("redoubt")),
            "it carried through a wall: {:?}",
            window(&s, 9)
        );
    }

    #[test]
    fn only_the_minds_something_happened_to_are_told_anything() {
        // The cost of a moment is the number of minds it touched, not the
        // population — which is what makes a large cast affordable at all.
        let h = vault();
        let b = Bindings::new();
        let s = Scheduler::new(256);
        let rooms = ["band-one", "band-two", "green-room", "watch"];
        for i in 1..=16u64 {
            let body = format!("m{i:02}");
            h.with(|w| {
                w.enter(&body, format!("Maker-{i:02}"), at(rooms[i as usize % 4]))
                    .unwrap()
            });
            b.bind(i, h.id(), &body).unwrap();
            s.wake(i, 0, 0);
        }
        push(&h, &b, &s);

        h.with(|w| w.say("m01", "somebody should look at the redoubt").unwrap());
        let told = push(&h, &b, &s);
        assert!(told > 0, "nobody heard it");
        assert!(told < 8, "one utterance reached {told} of 16 minds");
    }

    #[test]
    fn a_mind_is_only_told_things_once() {
        let (h, b, s) = crew(2);
        h.with(|w| w.say("m1", "once").unwrap());
        assert_eq!(push(&h, &b, &s), 1);
        assert_eq!(push(&h, &b, &s), 0, "told again");
        assert_eq!(
            read_window(&s, 102)
                .iter()
                .filter(|t| t.contains("once"))
                .count(),
            1
        );
    }

    #[test]
    fn one_body_can_be_pushed_on_its_own_without_disturbing_the_others() {
        // What a tool act needs: the character that just moved is told at once,
        // rather than waiting for the world's next moment.
        let (h, b, s) = crew(3);
        h.with(|w| w.say("m1", "for the room").unwrap());

        assert!(push_one(&h, &s, 102, "m2"));
        assert!(read_window(&s, 102)
            .iter()
            .any(|t| t.contains("for the room")));
        assert!(
            !window(&s, 103).iter().any(|t| t.contains("for the room")),
            "the third mind was told too"
        );
        // And the rest of the sweep still finds it waiting.
        assert_eq!(push(&h, &b, &s), 1);
    }

    #[test]
    fn pushing_a_body_with_nothing_to_say_reports_it_rather_than_delivering() {
        let (h, _b, s) = crew(1);
        assert!(!push_one(&h, &s, 101, "m1"));
    }

    #[test]
    fn a_delta_for_a_retired_mind_is_spent_rather_than_left_to_replay() {
        // The mind is gone from the scheduler but the body is still in the
        // world. If the cursor did not move, every later sweep would rebuild
        // the same delta for ever.
        let (h, b, s) = crew(2);
        s.retire(102);
        h.with(|w| w.say("m1", "into the void").unwrap());

        push(&h, &b, &s);
        assert!(h.peek("m2").is_empty(), "the cursor was left behind");
    }

    #[test]
    fn nothing_a_mind_reads_carries_the_shape_of_the_machinery() {
        let (h, b, s) = crew(3);
        h.with(|w| {
            w.tell("m1", "m2", "you have the redoubt").unwrap();
            w.set_off("m3", at("band-one")).unwrap();
        });
        advance(&h, &b, &s);

        for npc_id in [101, 102, 103] {
            for line in window(&s, npc_id) {
                for leak in [
                    "vault-casting",
                    "green-room",
                    "Happening",
                    "Witnessed",
                    "Some(",
                    "npc_id",
                    "{",
                    "}",
                ] {
                    assert!(!line.contains(leak), "leaked {leak:?}: {line}");
                }
            }
        }
    }

    #[test]
    fn only_the_situation_claims_a_band() {
        // Read straight off the delta: this is about what `carried` produces,
        // not about anything downstream of it.
        let (h, _b, _s) = crew(2);
        h.with(|w| {
            w.say("m1", "the redoubt burned twice").unwrap();
            w.set_off("m1", at("band-one")).unwrap();
        });
        h.tick();

        let delta = h.peek("m2");
        let events = h.read(|w| carried(w, &delta));
        assert!(events.len() > 1, "nothing to tell apart");
        for e in &events {
            match e.kind {
                EventKind::Situation { .. } => assert!(e.kind.replaces().is_some()),
                _ => assert!(e.kind.replaces().is_none(), "{:?} superseded", e.kind),
            }
        }
    }
}
