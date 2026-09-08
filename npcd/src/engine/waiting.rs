//! A character waiting for one named thing, aimed at one named person.
//!
//! # A wait is a subscription, not an act that does nothing
//!
//! The act this replaces was not a wait. `wait` took an `until` in free text —
//! "the silence speaks", "they finish speaking" — that nothing in the world
//! could read, so nothing could satisfy it, so it never ended. A character did
//! not wait for anybody; it *decided to wait*, spent a decode, forgot, and
//! decided again four seconds later. Three characters did that at each other
//! for hours while the daemon reported perfect health.
//!
//! A [`Waiting`] is state instead. The character goes quiet, the world is asked
//! each pass whether the thing has happened, and when it has, the character is
//! woken with it in front of it.
//!
//! # Every kind has to be a question the world can answer
//!
//! [`Kind`] is deliberately tiny. A kind the world cannot settle is `until` with
//! more syntax, and the whole value of the change is that something other than
//! the character can decide when the wait is over. These three are settled by
//! looking at the room: who spoke, who is standing in it now, who was and is
//! not.
//!
//! # Two failure modes it is built against
//!
//! **A wait nothing can satisfy.** The person walks out, or is retired, or the
//! thing simply never happens — and the character sleeps forever while every
//! view of it says "blocked", which is what it says for a character that is
//! merely quiet. That is the same silent-success shape as a `<think>` that never
//! closes, so [`Waiting::until_ms`] is not optional and [`Waiting::expired`] is
//! checked on the same pass as satisfaction.
//!
//! **Two characters waiting at each other.** Waiting on somebody wakes them; if
//! they could wait back, each would wake the other to do nothing, which costs
//! more than the deadlock it replaced. They cannot: somebody already waiting on
//! you is struck from the list you may wait on, in the grammar, so the act is
//! unreachable rather than discouraged. See `tools::Choices::Waitable`.

use serde::Serialize;

/// How long a wait stands before the character gives up on it.
///
/// **Not optional.** A wait with no deadline is a character asleep with nothing
/// able to wake it, and that reads as a healthy quiet character from every
/// angle the daemon has.
///
/// Two minutes of world time: long enough that a companion who is thinking,
/// walking a stop, or simply slower than the heartbeat is worth waiting for,
/// and short enough that a character whose wait can never be answered is doing
/// something else within the minute after.
pub const PATIENCE_MS: u64 = 120_000;

/// What would end a wait.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Somebody said something. The commonest by far: it is what a character
    /// asking a question is doing until it gets an answer.
    SomeoneSpeaks,
    /// Somebody came into the room.
    SomeoneArrives,
    /// Somebody left it.
    SomeoneLeaves,
}

impl Kind {
    /// Parse the value the grammar constrained the model to. `None` for
    /// anything else, which cannot happen through the stencil and can through
    /// a hand-written call.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "someone_speaks" => Some(Kind::SomeoneSpeaks),
            "someone_arrives" => Some(Kind::SomeoneArrives),
            "someone_leaves" => Some(Kind::SomeoneLeaves),
            _ => None,
        }
    }

    /// How it reads to the person being waited on.
    ///
    /// Second person and plain, because it arrives as something they perceive
    /// rather than as a label on a state. "Wyneth is waiting for you to speak"
    /// is a fact about the room; "wait_for(someone_speaks)" is a fact about the
    /// machinery.
    pub fn as_seen(self) -> &'static str {
        match self {
            Kind::SomeoneSpeaks => "waiting for you to say something",
            Kind::SomeoneArrives => "waiting for you to arrive",
            Kind::SomeoneLeaves => "waiting for you to leave",
        }
    }

    /// How it reads to the waiter itself, in its own record of what it did.
    pub fn as_done(self) -> &'static str {
        match self {
            Kind::SomeoneSpeaks => "for somebody to speak",
            Kind::SomeoneArrives => "for somebody to arrive",
            Kind::SomeoneLeaves => "for somebody to leave",
        }
    }
}

/// One character's standing wait.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Waiting {
    pub kind: Kind,
    /// The one person it is about, by the name they go by here. `None` waits on
    /// whoever is around, and nobody is told.
    pub who: Option<String>,
    /// World time this wait stops standing. See [`PATIENCE_MS`].
    pub until_ms: u64,
}

impl Waiting {
    pub fn new(kind: Kind, who: Option<String>, now_ms: u64) -> Self {
        Self {
            kind,
            who,
            until_ms: now_ms.saturating_add(PATIENCE_MS),
        }
    }

    /// Whether this wait has stood long enough.
    pub fn expired(&self, now_ms: u64) -> bool {
        now_ms >= self.until_ms
    }

    /// Whether `speaker` saying something ends this wait.
    ///
    /// A named wait is answered only by that person. An unnamed one is answered
    /// by anybody — which is what makes it the cheap ambient wait rather than a
    /// demand on somebody in particular.
    pub fn answered_by_speech(&self, speaker: &str) -> bool {
        if self.kind != Kind::SomeoneSpeaks {
            return false;
        }
        match &self.who {
            None => true,
            Some(who) => who.eq_ignore_ascii_case(speaker.trim()),
        }
    }

    /// Whether the room's occupants now answer this wait.
    ///
    /// `before` and `now` are who was here when the wait was set and who is
    /// here on this pass. Asked of the **world** rather than of the event
    /// stream, because arrivals and departures reach a character as one line of
    /// narrated prose rather than as anything a match could be written against
    /// — and matching on prose is how a rule stops meaning what it says.
    pub fn answered_by_room(&self, before: &[String], now: &[String]) -> bool {
        let named = |set: &[String]| {
            self.who
                .as_ref()
                .map(|w| set.iter().any(|n| n.eq_ignore_ascii_case(w)))
        };
        match self.kind {
            Kind::SomeoneSpeaks => false,
            Kind::SomeoneArrives => match named(now) {
                // Somebody in particular: are they here now, having not been?
                Some(here) => here && !named(before).unwrap_or(false),
                // Anybody: is the room fuller than it was?
                None => now.iter().any(|n| !before.contains(n)),
            },
            Kind::SomeoneLeaves => match named(now) {
                Some(here) => !here && named(before).unwrap_or(false),
                None => before.iter().any(|n| !now.contains(n)),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn who(n: &str) -> Option<String> {
        Some(n.to_string())
    }

    fn names(ns: &[&str]) -> Vec<String> {
        ns.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_named_wait_is_answered_only_by_that_person() {
        let w = Waiting::new(Kind::SomeoneSpeaks, who("Perrin Vastwood"), 0);
        assert!(w.answered_by_speech("Perrin Vastwood"));
        // Case and stray spacing are the world's spelling, not a different
        // person.
        assert!(w.answered_by_speech(" perrin vastwood "));
        assert!(!w.answered_by_speech("Orion Vance"));
    }

    /// The cheap ambient wait: anybody ends it, and nobody is put under any
    /// obligation, which is why it is the one available with nobody named.
    #[test]
    fn an_unnamed_wait_is_answered_by_anybody() {
        let w = Waiting::new(Kind::SomeoneSpeaks, None, 0);
        assert!(w.answered_by_speech("Orion Vance"));
        assert!(w.answered_by_speech("anybody at all"));
    }

    #[test]
    fn speech_does_not_answer_a_wait_on_the_door() {
        for k in [Kind::SomeoneArrives, Kind::SomeoneLeaves] {
            let w = Waiting::new(k, who("Perrin Vastwood"), 0);
            assert!(!w.answered_by_speech("Perrin Vastwood"), "{k:?}");
        }
    }

    #[test]
    fn arriving_is_being_here_now_and_not_before() {
        let w = Waiting::new(Kind::SomeoneArrives, who("Perrin Vastwood"), 0);
        assert!(w.answered_by_room(&names(&[]), &names(&["Perrin Vastwood"])));
        // Already here when the wait was set: standing still is not arriving,
        // or the wait would answer itself the instant it was made.
        assert!(!w.answered_by_room(&names(&["Perrin Vastwood"]), &names(&["Perrin Vastwood"])));
        assert!(!w.answered_by_room(&names(&[]), &names(&["Orion Vance"])));
    }

    #[test]
    fn leaving_is_having_been_here_and_not_being_here_now() {
        let w = Waiting::new(Kind::SomeoneLeaves, who("Perrin Vastwood"), 0);
        assert!(w.answered_by_room(&names(&["Perrin Vastwood"]), &names(&[])));
        assert!(!w.answered_by_room(&names(&[]), &names(&[])));
    }

    #[test]
    fn an_unnamed_wait_on_the_door_takes_anybody_coming_or_going() {
        let arrive = Waiting::new(Kind::SomeoneArrives, None, 0);
        assert!(arrive.answered_by_room(&names(&["A"]), &names(&["A", "B"])));
        assert!(!arrive.answered_by_room(&names(&["A", "B"]), &names(&["A"])));

        let leave = Waiting::new(Kind::SomeoneLeaves, None, 0);
        assert!(leave.answered_by_room(&names(&["A", "B"]), &names(&["A"])));
        assert!(!leave.answered_by_room(&names(&["A"]), &names(&["A", "B"])));
    }

    /// **The deadline is the difference between a quiet character and a lost
    /// one**, and both look identical from outside.
    #[test]
    fn a_wait_gives_up_eventually() {
        let w = Waiting::new(Kind::SomeoneSpeaks, who("Somebody Absent"), 1_000);
        assert!(!w.expired(1_000));
        assert!(!w.expired(1_000 + PATIENCE_MS - 1));
        assert!(w.expired(1_000 + PATIENCE_MS));
        assert!(w.expired(u64::MAX));
    }

    /// Every value the grammar can produce has to parse, or the act is refused
    /// for saying exactly what it was constrained to say.
    #[test]
    fn every_kind_the_grammar_offers_parses() {
        for k in crate::engine::tools::WAIT_KINDS {
            assert!(Kind::parse(k).is_some(), "`{k}` is offered and not parsed");
        }
        assert_eq!(Kind::parse("whenever"), None);
    }
}
