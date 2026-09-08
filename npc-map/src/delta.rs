//! What to hand a body when its turn comes round.
//!
//! Everything else in this crate answers a question about the world. This
//! assembles the answers into the one thing a mind is actually given: where it
//! is, and what has happened since it last looked.
//!
//! # Two halves, on purpose
//!
//! A [`Delta`] carries a **percept** and a run of **changes**, and they are not
//! the same kind of thing. The percept is a point in time — the whole of what
//! is true where the body stands, recomputed from scratch, superseding whatever
//! was true before. The changes are a stream — ordered, cursored, accumulating.
//! Rendering them into one block is the last step and the only place they meet.
//!
//! # The percept is omitted when it has not changed
//!
//! [`crate::perceive::percept`] is a pure function, so an unchanged situation
//! renders to **byte-identical** text. Sending it again would place the same
//! tokens twice: the reader would carry two copies of one truth, and the second
//! copy would be the more recent, which is exactly the wrong thing for
//! attention to weight highest.
//!
//! So [`Attention`] remembers what it last showed each body and leaves the
//! percept out when nothing has moved. A body that nothing happened to gets an
//! empty delta and costs nothing — which is what pays for a mind that never
//! blocks.
//!
//! # Two cursors, in two places, deliberately
//!
//! How far a body has *read* is world state and lives on the actor, because
//! [`crate::witness`] needs it to answer what is new. What was last *shown* is
//! a fact about rendering and lives here. Losing either is survivable and in
//! opposite directions: forget what was shown and one percept is repeated;
//! forget what was read and old changes are replayed. Neither can corrupt the
//! world, which is why neither belongs in it.

use std::collections::BTreeMap;

use crate::perceive::percept;
use crate::salience::{loudest, Weight};
use crate::witness::{narrate, since, Witnessed};
use crate::world::{Tick, World};

/// What one body is handed at one moment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Delta {
    /// Whose it is. Carried so a delta that has been queued, batched or sent
    /// still knows, the same reason [`Witnessed`] carries its reader.
    pub who: String,
    /// The moment it was taken.
    pub at: Tick,
    /// Where the body is and what is true there — `None` when that is
    /// unchanged since this reader was last shown it.
    pub percept: Option<String>,
    /// What it made out since it last looked, oldest first.
    pub events: Vec<Witnessed>,
}

impl Delta {
    /// Nothing moved and nothing happened. Not pushed, not prefilled, free.
    pub fn is_empty(&self) -> bool {
        self.percept.is_none() && self.events.is_empty()
    }

    /// The loudest thing in it, or nothing if nothing happened.
    ///
    /// A changed percept alone has no weight: it says the situation moved, not
    /// that anything demanded attention. What demands attention is an event.
    pub fn loudest(&self) -> Option<Weight> {
        loudest(&self.events)
    }

    /// Whether this is worth a turn now, ahead of whatever was planned.
    pub fn preempts(&self) -> bool {
        self.loudest().is_some_and(|w| w >= Weight::PREEMPTS)
    }

    /// The delta as the text a body reads, or nothing when it is empty.
    ///
    /// Situation first, then what happened — which is both how it reads and
    /// how it caches. The stable half leads, so an unchanged percept that *is*
    /// re-sent still matches the tokens above it, and the volatile half is
    /// appended after the boundary rather than through it.
    pub fn render(&self, world: &World) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        if let Some(here) = &self.percept {
            parts.push(here.trim_end().to_string());
        }
        if let Some(news) = narrate(world, &self.events) {
            parts.push(news);
        }
        (!parts.is_empty()).then(|| parts.join("\n\n") + "\n")
    }
}

/// What each body was last shown, so that it is not shown it twice.
///
/// One of these belongs to whatever is driving the world — it is the
/// environment's bookkeeping, not the world's. Nothing here can change what is
/// true; the worst a corrupt `Attention` can do is repeat itself.
#[derive(Clone, Debug, Default)]
pub struct Attention {
    shown: BTreeMap<String, u64>,
}

impl Attention {
    pub fn new() -> Attention {
        Attention::default()
    }

    /// What this body would be handed right now, without handing it over.
    ///
    /// The read-only half of [`Attention::take`]: same answer, no cursor moved
    /// on either side. Asking twice gives the same delta both times.
    pub fn peek(&self, world: &World, id: &str) -> Delta {
        self.compose(world, id)
    }

    /// What this body is handed now, marking it as delivered.
    ///
    /// Advances both cursors: the world's record of how far this body has read,
    /// and this one's record of what it was last shown.
    pub fn take(&mut self, world: &mut World, id: &str) -> Delta {
        let delta = self.compose(world, id);
        if let Some(here) = &delta.percept {
            self.shown.insert(id.to_string(), fingerprint(here));
        }
        if world.actor(id).is_some() {
            world.mark_seen(id);
        }
        delta
    }

    /// Take a delta for every body in the world, keeping the ones with
    /// something in them.
    ///
    /// The shape the environment wants: one sweep after a world tick, and what
    /// comes back is exactly the set of minds with anything to prefill. Bodies
    /// that nothing happened to are absent rather than present-and-empty,
    /// because "costs nothing" has to mean *nothing*, including a row in a
    /// batch.
    pub fn sweep(&mut self, world: &mut World) -> Vec<Delta> {
        let ids: Vec<String> = world.actors().map(|a| a.id.clone()).collect();
        ids.iter()
            .map(|id| self.take(world, id))
            .filter(|d| !d.is_empty())
            .collect()
    }

    /// Forget what a body was last shown, so the next delta grounds it again.
    ///
    /// For a body leaving the world, or one whose context was rebuilt beneath
    /// it — after which what it was *last shown* is no longer what it holds.
    pub fn forget(&mut self, id: &str) {
        self.shown.remove(id);
    }

    /// Whether this body has already been shown exactly this situation.
    pub fn is_current(&self, id: &str, here: &str) -> bool {
        self.shown.get(id) == Some(&fingerprint(here))
    }

    fn compose(&self, world: &World, id: &str) -> Delta {
        let mut delta = Delta {
            who: id.to_string(),
            at: world.now(),
            percept: None,
            events: Vec::new(),
        };
        // A body that is not in the world perceives nothing and has witnessed
        // nothing. An empty delta, not a panic: an actor can leave between a
        // sweep being planned and being run.
        if world.actor(id).is_none() {
            return delta;
        }
        let here = percept(world, id);
        if !here.is_empty() && !self.is_current(id, &here) {
            delta.percept = Some(here);
        }
        delta.events = since(world, id);
        delta
    }
}

/// A stable fingerprint of one rendered percept — FNV-1a, 64 bits.
///
/// Written out rather than taken from `DefaultHasher` because this has to give
/// the same answer in every process and every release: a fingerprint that
/// changed between runs would silently re-ground every body on restart, and
/// one that changed between *hosts* would do it inconsistently.
fn fingerprint(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::MapSet;
    use crate::world::Where;

    fn vault() -> World {
        World::new(
            MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/maps"))
                .expect("the vault must load"),
        )
    }

    fn at(node: &str) -> Where {
        Where::new("vault-casting", node)
    }

    /// One Maker in band one, already shown where it is.
    fn grounded() -> (World, Attention) {
        let mut w = vault();
        let mut a = Attention::new();
        w.enter("m1", "Maker-01", at("band-one")).unwrap();
        assert!(a.take(&mut w, "m1").percept.is_some());
        (w, a)
    }

    // -- the fingerprint ---------------------------------------------------

    #[test]
    fn the_same_text_fingerprints_the_same_way_every_time() {
        assert_eq!(
            fingerprint("You are in band one."),
            fingerprint("You are in band one.")
        );
        assert_eq!(fingerprint(""), fingerprint(""));
    }

    #[test]
    fn text_that_differs_at_all_fingerprints_differently() {
        // Including the cases a weak hash would miss: one character, a
        // transposition, and a difference only in length.
        let cases = [
            ("You are in band one.", "You are in band two."),
            ("Maker-01 is here.", "Maker-10 is here."),
            ("ab", "ba"),
            ("band one", "band one "),
            ("", " "),
        ];
        for (a, b) in cases {
            assert_ne!(fingerprint(a), fingerprint(b), "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn the_fingerprint_is_the_documented_constant_for_a_known_input() {
        // FNV-1a of "a" is fixed by the algorithm, not by this implementation.
        // Pinning it means a well-meaning rewrite cannot quietly change what
        // every stored fingerprint means.
        assert_eq!(fingerprint("a"), 0xaf63_dc4c_8601_ec8c);
    }

    // -- the percept half --------------------------------------------------

    #[test]
    fn the_first_delta_grounds_the_body() {
        let mut w = vault();
        let mut a = Attention::new();
        w.enter("m1", "Maker-01", at("band-one")).unwrap();

        let d = a.take(&mut w, "m1");
        assert_eq!(d.who, "m1");
        assert!(d.percept.as_deref().unwrap().contains("band one"));
        assert!(!d.is_empty());
    }

    #[test]
    fn a_situation_that_has_not_moved_is_not_sent_twice() {
        let (mut w, mut a) = grounded();
        let d = a.take(&mut w, "m1");
        assert_eq!(d.percept, None);
        assert!(d.events.is_empty());
        assert!(d.is_empty(), "an idle body cost something");
    }

    #[test]
    fn a_situation_that_moves_is_sent_again() {
        let (mut w, mut a) = grounded();
        w.enter("m2", "Maker-02", at("band-one")).unwrap();

        let d = a.take(&mut w, "m1");
        let here = d.percept.expect("somebody walked in");
        assert!(here.contains("Maker-02"), "{here}");
    }

    #[test]
    fn a_situation_that_returns_to_an_earlier_one_is_still_sent_again() {
        // The comparison is against what this body was *last shown*, not
        // against every situation it has ever been in. A room that empties back
        // to how it started is news, because the reader's most recent grounding
        // says somebody is in it.
        let (mut w, mut a) = grounded();
        let first = percept(&w, "m1");

        w.enter("m2", "Maker-02", at("band-one")).unwrap();
        assert!(a.take(&mut w, "m1").percept.is_some());

        w.set_off("m2", at("green-room")).unwrap();
        w.settle();
        let back = a.take(&mut w, "m1").percept.expect("the room emptied");
        assert_eq!(back, first, "the situation did not actually return");
    }

    #[test]
    fn what_a_body_holds_is_part_of_its_situation() {
        let (mut w, mut a) = grounded();
        w.take("m1", Some("cindy")).unwrap();
        let here = a.take(&mut w, "m1").percept.expect("it sat down");
        assert!(here.contains("holding cindy"), "{here}");
    }

    #[test]
    fn being_on_the_way_somewhere_is_part_of_its_situation() {
        let (mut w, mut a) = grounded();
        w.set_off("m1", Where::new("vault-command", "command-room"))
            .unwrap();
        let here = a.take(&mut w, "m1").percept.expect("it set off");
        assert!(here.contains("on your way"), "{here}");
    }

    #[test]
    fn forgetting_a_body_grounds_it_again_next_time() {
        let (mut w, mut a) = grounded();
        assert!(a.take(&mut w, "m1").is_empty());
        a.forget("m1");
        assert!(a.take(&mut w, "m1").percept.is_some());
    }

    #[test]
    fn is_current_agrees_with_what_was_sent() {
        let (mut w, mut a) = grounded();
        assert!(a.is_current("m1", &percept(&w, "m1")));
        assert!(!a.is_current("m1", "something else"));
        assert!(!a.is_current("nobody", &percept(&w, "m1")));

        w.enter("m2", "Maker-02", at("band-one")).unwrap();
        assert!(!a.is_current("m1", &percept(&w, "m1")));
        a.take(&mut w, "m1");
        assert!(a.is_current("m1", &percept(&w, "m1")));
    }

    // -- the event half ----------------------------------------------------

    #[test]
    fn changes_are_carried_and_then_spent() {
        let (mut w, mut a) = grounded();
        w.enter("m2", "Maker-02", at("band-one")).unwrap();
        w.say("m2", "the redoubt burned twice").unwrap();

        let d = a.take(&mut w, "m1");
        assert_eq!(d.events.len(), 2, "arrival and utterance");
        assert!(a.take(&mut w, "m1").events.is_empty(), "told twice");
    }

    #[test]
    fn a_delta_only_ever_carries_what_the_body_could_make_out() {
        // The scope rules are witness's; this asserts the delta does not widen
        // them on its way through.
        let (mut w, mut a) = grounded();
        w.enter("far", "Maker-09", Where::new("vault-command", "anteroom"))
            .unwrap();
        w.say("far", "not a word of this carries").unwrap();

        let d = a.take(&mut w, "m1");
        assert!(d.events.is_empty(), "{:?}", d.events);
        assert!(d.is_empty());
    }

    #[test]
    fn the_weight_of_a_delta_is_the_weight_of_its_loudest_change() {
        let (mut w, mut a) = grounded();
        w.enter("m2", "Maker-02", at("band-one")).unwrap();
        a.take(&mut w, "m1");

        w.say("m2", "to the room").unwrap();
        assert_eq!(a.peek(&w, "m1").loudest(), Some(Weight::Wake));
        assert!(!a.peek(&w, "m1").preempts());

        w.tell("m2", "m1", "and to you").unwrap();
        assert!(a.take(&mut w, "m1").preempts(), "being addressed did not");
    }

    #[test]
    fn a_situation_moving_is_not_by_itself_worth_a_turn() {
        // A changed percept says the world moved, not that anything wants
        // answering. Weighing it would make every passing body an interruption.
        let (mut w, mut a) = grounded();
        w.enter("m2", "Maker-02", at("ring-north")).unwrap();
        w.mark_seen("m1");

        let d = a.take(&mut w, "m1");
        assert!(
            d.percept.is_some(),
            "the corridor filled and it did not show"
        );
        assert!(d.events.is_empty());
        assert_eq!(d.loudest(), None);
        assert!(!d.preempts());
    }

    // -- peek and take -----------------------------------------------------

    #[test]
    fn peeking_moves_neither_cursor() {
        let (mut w, mut a) = grounded();
        w.enter("m2", "Maker-02", at("band-one")).unwrap();

        let first = a.peek(&w, "m1");
        assert_eq!(a.peek(&w, "m1"), first, "peeking twice differed");
        assert_eq!(a.take(&mut w, "m1"), first, "taking differed from peeking");
        assert_ne!(a.peek(&w, "m1"), first, "taking moved nothing");
    }

    #[test]
    fn a_body_that_is_not_in_the_world_is_handed_nothing() {
        let (mut w, mut a) = grounded();
        let d = a.take(&mut w, "nobody");
        assert!(d.is_empty());
        assert_eq!(d.who, "nobody");
        assert_eq!(d.percept, None);
        assert!(d.render(&w).is_none());
    }

    // -- rendering ---------------------------------------------------------

    #[test]
    fn the_situation_is_rendered_before_what_happened_in_it() {
        let (mut w, mut a) = grounded();
        w.enter("m2", "Maker-02", at("band-one")).unwrap();
        w.say("m2", "the redoubt burned twice").unwrap();

        let text = a.take(&mut w, "m1").render(&w).expect("something happened");
        let here = text.find("You are").expect("no situation");
        let news = text.find("Maker-02 came in").expect("no news");
        assert!(here < news, "{text}");
        assert!(text.ends_with('\n'));
    }

    #[test]
    fn an_unchanged_situation_renders_only_what_happened() {
        let (mut w, mut a) = grounded();
        // Speech changes nothing about the room, so the percept holds.
        w.enter("m2", "Maker-02", at("band-one")).unwrap();
        a.take(&mut w, "m1");
        w.say("m2", "the redoubt burned twice").unwrap();

        let d = a.take(&mut w, "m1");
        assert_eq!(d.percept, None);
        let text = d.render(&w).expect("it was said");
        assert!(!text.contains("You are"), "{text}");
        assert!(text.starts_with("Maker-02 said"), "{text}");
    }

    #[test]
    fn an_empty_delta_renders_to_nothing_at_all() {
        let (mut w, mut a) = grounded();
        let d = a.take(&mut w, "m1");
        assert!(d.is_empty());
        assert_eq!(d.render(&w), None);
    }

    // -- the sweep ---------------------------------------------------------

    #[test]
    fn a_sweep_grounds_everybody_once_and_then_goes_quiet() {
        let mut w = vault();
        let mut a = Attention::new();
        for i in 1..=6 {
            w.enter(format!("m{i}"), format!("Maker-{i:02}"), at("core"))
                .unwrap();
        }
        // The first sweep has something for everyone; entering is itself a
        // change, so bodies that arrived together also witnessed each other.
        let first = a.sweep(&mut w);
        assert_eq!(first.len(), 6);
        assert!(first.iter().all(|d| d.percept.is_some()));

        // Nothing has happened since, so nothing is pushed at all.
        assert!(a.sweep(&mut w).is_empty(), "an idle world cost something");
    }

    #[test]
    fn a_sweep_carries_only_the_bodies_something_happened_to() {
        let mut w = vault();
        let mut a = Attention::new();
        w.enter("near", "Maker-01", at("band-one")).unwrap();
        w.enter("also", "Maker-02", at("band-one")).unwrap();
        w.enter("far", "Maker-09", Where::new("vault-command", "anteroom"))
            .unwrap();
        a.sweep(&mut w);

        w.say("near", "the redoubt burned twice").unwrap();
        let swept = a.sweep(&mut w);
        let who: Vec<&str> = swept.iter().map(|d| d.who.as_str()).collect();
        assert_eq!(who, vec!["also"], "{who:?}");
    }

    #[test]
    fn a_sweep_after_a_tick_carries_everybody_the_movement_touched() {
        let mut w = vault();
        let mut a = Attention::new();
        w.enter("walker", "Maker-01", at("band-one")).unwrap();
        w.enter("waiting", "Maker-02", at("green-room")).unwrap();
        w.enter("elsewhere", "Maker-09", at("watch")).unwrap();
        a.sweep(&mut w);

        w.set_off("walker", at("green-room")).unwrap();
        w.tick();

        let swept = a.sweep(&mut w);
        let who: Vec<&str> = swept.iter().map(|d| d.who.as_str()).collect();
        // The walker learns it arrived; the room it entered sees it come in;
        // the watch, which can see neither, hears nothing.
        assert!(who.contains(&"walker"), "{who:?}");
        assert!(who.contains(&"waiting"), "{who:?}");
        assert!(!who.contains(&"elsewhere"), "{who:?}");
    }

    #[test]
    fn a_sweep_is_the_same_answer_as_taking_each_body_in_turn() {
        // The batched path and the single path must not be able to disagree,
        // because only one of them is exercised in production.
        let mut w = vault();
        w.enter("m1", "Maker-01", at("band-one")).unwrap();
        w.enter("m2", "Maker-02", at("band-one")).unwrap();
        w.enter("m3", "Maker-03", at("green-room")).unwrap();

        let mut swept_world = w.clone();
        let mut swept = Attention::new();
        let batch = swept.sweep(&mut swept_world);

        let mut one_world = w.clone();
        let mut one = Attention::new();
        let ids: Vec<String> = one_world.actors().map(|a| a.id.clone()).collect();
        let singly: Vec<Delta> = ids
            .iter()
            .map(|id| one.take(&mut one_world, id))
            .filter(|d| !d.is_empty())
            .collect();

        assert_eq!(batch, singly);
    }

    #[test]
    fn a_body_that_left_the_world_is_skipped_rather_than_fatal() {
        let mut w = vault();
        let mut a = Attention::new();
        w.enter("m1", "Maker-01", at("band-one")).unwrap();
        a.sweep(&mut w);
        // Something the environment planned a delta for is gone by the time it
        // runs — the sweep does not, and a direct take gives an empty delta.
        assert!(a.take(&mut w, "departed").is_empty());
        assert!(a.sweep(&mut w).is_empty());
    }
}
