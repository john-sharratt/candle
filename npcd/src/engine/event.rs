//! What arrives in a character's inbox, and how it reads once it gets there.
//!
//! The mind design has one loop per character: block on an inbox, drain it,
//! gather, decode one step, fan out. This module is the inbox's contents — the
//! events themselves, the salience that decides how urgently they are taken, and
//! the prose each one turns into when it reaches the model.
//!
//! # Events become prose, not JSON
//!
//! An event is structured on the wire because a caller has to construct one, and
//! prose by the time the character sees it. That conversion is the whole of
//! [`Event::prose`], and it is deliberate rather than incidental: a character
//! reasons in the language its personality and beliefs are written in, and
//! handing a mind a JSON object asks it to translate before it can think. The
//! world speaks to the character the way a narrator would.
//!
//! # Salience gates the tick, never the write
//!
//! The mind design is explicit that a filter dropping contradictory evidence
//! before it lands makes a delusion permanent. So salience does three things
//! here — it decides whether an arrival *preempts* (forces a tick now), whether
//! it *rouses* (ends a standing wait), and how strongly it biases what the
//! gather pulls — and it never decides whether the event is recorded.
//! Everything that arrives, lands.
//!
//! # The two bars, and what sits between them
//!
//! [`Salience::PREEMPT_AT`] is 0.8 and [`Salience::ROUSES_AT`] is 0.6, so there
//! is a band that ends a wait without cutting into a working character:
//! somebody speaking to the room, somebody walking in, a gesture aimed at you.
//! That band is the whole point — being spoken to in a room you were listening
//! in is exactly the case that must not have to survive two minutes of patience
//! before anybody notices it.

use std::fmt;

use serde::{Deserialize, Serialize};

/// How much this event demands attention, 0.0–1.0.
///
/// A newtype rather than a bare `f32` because the clamping has to happen at
/// exactly one place. A caller can and will send `5.0`, and a salience above one
/// would make every subsequent comparison meaningless.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize)]
pub struct Salience(f32);

impl Salience {
    /// The heartbeat's own salience: enough to wake, not enough to preempt.
    pub const IDLE: Salience = Salience(0.1);
    /// Ordinary world traffic.
    pub const NORMAL: Salience = Salience(0.5);
    /// Took damage, ally down, a direct order. Forces a tick now.
    pub const URGENT: Salience = Salience(0.9);

    /// At or above this, an arrival preempts rather than waiting for the next
    /// scheduled tick.
    pub const PREEMPT_AT: f32 = 0.8;

    /// At or above this, an arrival **ends a standing wait**.
    ///
    /// # Why this is a second, lower bar rather than the same one
    ///
    /// The two questions are not the same question, because the character is
    /// not in the same state when each is asked.
    ///
    /// [`Self::PREEMPT_AT`] asks whether to *cut into* a character that is
    /// already thinking on its own schedule. That bar is high on purpose: a
    /// room where every shifted chair interrupted everybody would leave nobody
    /// able to finish a thought.
    ///
    /// This asks whether to rouse one that has gone quiet on a
    /// [`crate::engine::waiting::Waiting`] — deliberately doing nothing, with
    /// its next thought two minutes out. There is nothing to interrupt, so the
    /// bar is the rung the map already calls "worth a turn at the next
    /// opportunity" (`npc_map::Weight::Wake`): for a waiting character, this
    /// *is* the next opportunity, and the alternative is sitting silent through
    /// the very thing it was listening for.
    ///
    /// **This is what makes a wait a wait rather than a deafness.** A wait is a
    /// subscription to one named thing; without a bar like this, everything
    /// else that happened — being spoken to by the wrong person, somebody
    /// messaging you, walking into the room — landed in the inbox and was read
    /// two minutes later, by which time whoever did it had gone.
    ///
    /// Below the bar and therefore *not* an interrupt, each deliberately:
    /// the heartbeat and the standing task (they are the quiet itself), a
    /// situation (where you are standing is not something that happened),
    /// speech aimed past you, things done in the room, somebody leaving, and a
    /// group thread — which is the phone's ambient traffic in the same way
    /// `Weight::Note` is the room's. A wait that anything at all could end
    /// would be a four-second heartbeat spelled differently.
    pub const ROUSES_AT: f32 = 0.6;

    pub fn new(v: f32) -> Self {
        // NaN compares false against every bound, so it would survive a naive
        // clamp and then poison every ordering it takes part in. Mapped to
        // NORMAL: an event whose urgency is unreadable is ordinary traffic, not
        // an emergency and not something to drop.
        if v.is_nan() {
            return Salience::NORMAL;
        }
        Salience(v.clamp(0.0, 1.0))
    }

    pub fn get(self) -> f32 {
        self.0
    }

    pub fn preempts(self) -> bool {
        self.0 >= Self::PREEMPT_AT
    }

    /// Whether this arrival is worth ending a standing wait for. See
    /// [`Self::ROUSES_AT`].
    ///
    /// Everything that preempts also rouses — the bar is lower — so no caller
    /// has to ask both.
    pub fn rouses(self) -> bool {
        self.0 >= Self::ROUSES_AT
    }

    /// The more urgent of two. `f32` has no `Ord`, so this cannot be `max`.
    pub fn max_of(self, other: Salience) -> Salience {
        if other.0 > self.0 {
            other
        } else {
            self
        }
    }
}

/// **Rousing is the lower bar.** If the two ever crossed, an arrival could cut
/// into a character that was working and leave a waiting one asleep — which is
/// exactly backwards, and it is why no caller has to ask both questions.
///
/// A `const` rather than a test: both bars are compile-time constants, so this
/// can be a fact about the build instead of something a test run has to
/// discover.
const BARS_ARE_ORDERED: () = assert!(Salience::ROUSES_AT <= Salience::PREEMPT_AT);
const _: () = BARS_ARE_ORDERED;

impl Default for Salience {
    fn default() -> Self {
        Salience::NORMAL
    }
}

/// A weight read off a place becomes a salience on the way in.
///
/// `npc-map` grades what one body made out on a four-rung ladder named for what
/// to do about it; the scheduler grades everything on one number. This is the
/// one place they meet, so the correspondence is here rather than spread over
/// the call sites — and `weight_and_salience_agree_about_preempting` is what
/// stops the two drifting into disagreeing about the only question that
/// matters, which is whether a character is interrupted.
impl From<npc_map::Weight> for Salience {
    fn from(w: npc_map::Weight) -> Salience {
        Salience::new(w.as_f32())
    }
}

impl<'de> Deserialize<'de> for Salience {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        // Through `new`, so a wire value is clamped by the same code path as a
        // constructed one. Deriving this would let `9.0` in from the network and
        // nowhere else, which is the worst kind of asymmetry to debug.
        Ok(Salience::new(f32::deserialize(d)?))
    }
}

/// Who an utterance was aimed at, resolved from the receiving character's side.
///
/// Resolved at delivery rather than carried raw, so that rendering needs no
/// second argument and cannot get the comparison wrong: whoever hands the event
/// over knows both who was addressed and who is receiving it, and that is the
/// only place both facts are in hand at once.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "at", rename_all = "snake_case")]
pub enum Addressed {
    /// Said to the room. You are among those it was for.
    #[default]
    Room,
    /// Said to you.
    You,
    /// Said to you, too quietly for anybody else to make out. Still aimed at
    /// you — it asks for an answer like anything else said to you — but the
    /// listener is told it was a whisper, because that is half of what it said.
    Whispered,
    /// Said to somebody else, in front of you.
    Other { who: String },
}

/// The kinds of thing that can happen to a character.
///
/// Deliberately small. Each variant exists because it renders to prose
/// differently, not because the world model has that many nouns — a taxonomy
/// that mirrors the game's object graph would need a new variant per content
/// patch, and every one of them would render the same way.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventKind {
    /// Something happened, described in words. The general case.
    Description { text: String },
    /// Someone said something to, or near, the character.
    Speech {
        speaker: String,
        text: String,
        /// Who it was aimed at, from this character's side. Delivery is by
        /// place — everyone in the room hears it — and direction is a property
        /// of the utterance, so the two have to be carried separately or
        /// overhearing becomes indistinguishable from being addressed.
        #[serde(default)]
        to: Addressed,
    },
    /// Something said to the character on its handset, from wherever the sender
    /// is.
    ///
    /// **Not [`EventKind::Speech`], and the difference is the whole point of a
    /// phone.** Speech is delivered by place: everybody in the room hears it,
    /// and overhearing is the normal case. A message reaches one thread and
    /// nobody else, from somebody who may be nowhere near — so it carries the
    /// thread it arrived on rather than a room, and the character answers it by
    /// naming that thread.
    Message {
        /// The thread, by the name *this* character calls it — which is the
        /// other party for a direct thread and the group's name otherwise. It
        /// is the argument `message` takes, so what the character is told is
        /// what it would have to type back.
        thread: String,
        from: String,
        text: String,
    },
    /// Where the character is and what is true there, right now.
    ///
    /// **Replaces** the previous one — see [`EventKind::replaces`]. A situation
    /// is a point in time, so two of them in the window is one stale reading of
    /// a room the character is no longer standing in, sitting where attention
    /// weights it highest.
    Situation { text: String },
    /// A specific entity was observed doing something.
    Entity {
        entity_id: String,
        observation: String,
    },
    /// A turn taken with nothing new to take it on. Carries no content.
    ///
    /// **Nothing schedules this any more.** It was the idle wake — a character
    /// with an empty inbox thought anyway, so that "nothing arrived" could not
    /// mean "dead forever". What that actually bought was a character writing
    /// an act into its own window every few seconds in a quiet room, until the
    /// window held nothing but its own last line and the next token was
    /// certain. A character with nothing to react to now does not think.
    ///
    /// It survives as the operator's instrument: `/wake` forces a turn on what
    /// the character already has, which is the point of being able to poke a
    /// mind and watch what comes out.
    Heartbeat,
    /// What the character is set on, restated.
    ///
    /// **Replaces** the previous one — see [`EventKind::replaces`]. It is a
    /// standing instruction, not a thing that happened, so a second one is the
    /// current one and the first is a task the character has been taken off.
    ///
    /// This is the turn that keeps a long run on course. A character with work
    /// in front of it has quiet turns rather than empty ones, and what fills
    /// them has to come from **outside** — a nudge derived from the character's
    /// own state would be its own reasoning read back as instruction, which is
    /// the runaway loop with extra steps.
    Nudge { text: String },
    /// The day ended. The character consolidates and its conversation rolls over
    /// — see `engine::sleep`.
    Sleep { day: u64 },
    /// The day began, on a fresh conversation.
    Wake { day: u64 },
    /// An operator speaking directly to the loop, through the `/` notation in the
    /// console. Marked as its own kind so a debugging poke is never mistaken for
    /// something the world did.
    Operator { text: String },
}

impl EventKind {
    /// A short machine-readable tag, for filtering in the Pulse view.
    pub fn tag(&self) -> &'static str {
        match self {
            EventKind::Description { .. } => "description",
            EventKind::Speech { .. } => "speech",
            EventKind::Message { .. } => "message",
            EventKind::Situation { .. } => "situation",
            EventKind::Nudge { .. } => "nudge",
            EventKind::Entity { .. } => "entity",
            EventKind::Heartbeat => "heartbeat",
            EventKind::Sleep { .. } => "sleep",
            EventKind::Wake { .. } => "wake",
            EventKind::Operator { .. } => "operator",
        }
    }

    /// The band this event supersedes within, if it supersedes anything.
    ///
    /// **This is how a point in time lives in an append-only window.** A
    /// situation is the whole of what is true where the character stands, so a
    /// second one does not add to the first — it replaces it, and keeping both
    /// would leave a stale reading of a room the character has left sitting in
    /// the most recent position, which is exactly where attention weights it
    /// highest.
    ///
    /// Everything else accumulates. A thing that happened stays happened.
    pub fn replaces(&self) -> Option<String> {
        match self {
            EventKind::Situation { .. } => Some("situation".to_string()),
            // Its own band, beside the situation. Two standing instructions is
            // a character working to a task it has been taken off, which reads
            // as one that has forgotten what it was doing.
            EventKind::Nudge { .. } => Some("nudge".to_string()),
            _ => None,
        }
    }
}

/// One thing that happened, addressed to one character.
#[derive(Clone, Debug, Serialize)]
pub struct Event {
    /// Monotonic within a daemon run. Lets the Pulse view order arrivals that
    /// share a millisecond, which under a batched world update is most of them.
    pub seq: u64,
    /// World-clock milliseconds, from `crate::clock`. The narrative clock, not
    /// the wall clock — a character reasons in its own world's time.
    pub at_ms: u64,
    pub salience: Salience,
    pub kind: EventKind,
}

impl Event {
    pub fn new(seq: u64, at_ms: u64, salience: Salience, kind: EventKind) -> Self {
        Self {
            seq,
            at_ms,
            salience,
            kind,
        }
    }

    pub fn preempts(&self) -> bool {
        self.salience.preempts()
    }

    /// How this event reads to the character.
    ///
    /// Second person, present tense, no field names, no punctuation the model has
    /// to decode. This is the narrator's voice — the character is being told what
    /// it perceives, not handed a record of what the simulation logged.
    pub fn prose(&self) -> String {
        match &self.kind {
            EventKind::Description { text } => text.trim().to_string(),
            EventKind::Speech { speaker, text, to } => {
                let t = text.trim();
                // Three readings of one utterance, and the difference between
                // them is what makes a shared room worth standing in: being
                // told something, being among those it was said to, and
                // watching somebody else be told it.
                // **Reported, not quoted.** A tool carries intent, never
                // wording — `speak` takes what a character *means* to convey
                // and the catalog refuses a finished line of dialogue — so what
                // arrives here is substance. Quotation marks around it are a
                // claim that these were the words, which they never are: they
                // turned "that I am starting now" into `says: "that I am
                // starting now"`, a sentence nobody has ever spoken. The colon
                // reports; the quotes would fabricate, and this module's whole
                // discipline is narrate acts, never fabricate.
                match to {
                    Addressed::You => format!("{speaker} says to you: {t}"),
                    Addressed::Whispered => format!("{speaker} whispers to you: {t}"),
                    Addressed::Room => format!("{speaker} says: {t}"),
                    Addressed::Other { who } => format!("{speaker} says to {who}: {t}"),
                }
            }
            // **Named as a message, and the thread named with it.** A character
            // that is told only who spoke cannot tell whether it was heard by a
            // room or read off a handset, and the two are answered by different
            // acts. Naming the thread here means the argument `message` needs
            // is already in front of it.
            //
            // Reported rather than quoted, for the reason `Speech` gives: what
            // a phone carries is intent, and quoting it would fabricate wording
            // nobody chose.
            EventKind::Message { thread, from, text } => {
                let t = text.trim();
                match thread == from {
                    // A direct thread is named for the other party, so saying
                    // both would read as "Wren, on Wren".
                    true => format!("{from} messages you: {t}"),
                    false => format!("{from} messages you, on {thread}: {t}"),
                }
            }
            // Both arrive already written — a situation is generated from a
            // map, a nudge is authored beside the task it belongs to — so
            // rendering passes them through rather than decorating them.
            EventKind::Situation { text } | EventKind::Nudge { text } => text.trim().to_string(),
            EventKind::Entity {
                entity_id,
                observation,
            } => format!("You notice {entity_id}: {}", observation.trim()),
            // Not "a heartbeat fired". The character is not aware of the
            // scheduler; it is aware of a moment passing with nothing in it.
            EventKind::Heartbeat => "Time passes quietly. Nothing demands you.".to_string(),
            EventKind::Sleep { .. } => {
                "The day is over. You rest, and let the day settle into what you will keep of it."
                    .to_string()
            }
            EventKind::Wake { day } => {
                format!("You wake. It is day {day}. Yesterday is behind you, kept as memory.")
            }
            EventKind::Operator { text } => text.trim().to_string(),
        }
    }
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{} s={:.2}] {}",
            self.kind.tag(),
            self.salience.get(),
            self.prose()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn salience_is_clamped_at_both_ends() {
        assert_eq!(Salience::new(5.0).get(), 1.0);
        assert_eq!(Salience::new(-3.0).get(), 0.0);
        assert_eq!(Salience::new(0.4).get(), 0.4);
    }

    /// NaN survives a naive clamp — every comparison against it is false — and
    /// then poisons the tick ordering it takes part in.
    #[test]
    fn a_nan_salience_becomes_ordinary_traffic() {
        let s = Salience::new(f32::NAN);
        assert_eq!(s, Salience::NORMAL);
        assert!(!s.preempts());
    }

    /// The wire path must clamp exactly like the constructed one. It did not,
    /// once — `derive(Deserialize)` on the newtype let `9.0` in from the network
    /// and nowhere else.
    #[test]
    fn a_wire_salience_goes_through_the_same_clamp() {
        let s: Salience = serde_json::from_str("9.0").unwrap();
        assert_eq!(s.get(), 1.0);
        let n: Salience = serde_json::from_str("null").unwrap_or(Salience::NORMAL);
        assert_eq!(n, Salience::NORMAL);
    }

    #[test]
    fn only_high_salience_preempts() {
        assert!(!Salience::IDLE.preempts());
        assert!(!Salience::NORMAL.preempts());
        assert!(Salience::URGENT.preempts());
        assert!(
            Salience::new(Salience::PREEMPT_AT).preempts(),
            "the bound is inclusive"
        );
        assert!(!Salience::new(Salience::PREEMPT_AT - 0.01).preempts());
    }

    /// The situation replaces; nothing else does. This is how a point in time
    /// lives in an append-only window, so it is pinned — and so is the fact
    /// that every *change* accumulates, because a thing that happened stays
    /// happened.
    #[test]
    fn the_situation_replaces_and_nothing_else_replaces() {
        let here = EventKind::Situation {
            text: "You are in band one.".into(),
        };
        assert_eq!(here.replaces().as_deref(), Some("situation"));

        // Two situations share a band whatever they say, or the older one
        // survives to be read as current.
        let later = EventKind::Situation {
            text: "You are in the green room.".into(),
        };
        assert_eq!(here.replaces(), later.replaces());

        for k in [
            EventKind::Description { text: "x".into() },
            EventKind::Heartbeat,
            EventKind::Speech {
                speaker: "H".into(),
                text: "x".into(),
                to: Addressed::You,
            },
            EventKind::Entity {
                entity_id: "e".into(),
                observation: "o".into(),
            },
        ] {
            assert!(k.replaces().is_none(), "{k:?} must accumulate, not replace");
        }
    }

    /// Prose is what the character actually reads, so it must never contain the
    /// wire's field names or its punctuation.
    #[test]
    fn prose_carries_no_json_shape() {
        let events = [
            EventKind::Description {
                text: "The gate groans open.".into(),
            },
            EventKind::Speech {
                speaker: "Hess".into(),
                text: "Hold the line.".into(),
                to: Addressed::You,
            },
            EventKind::Situation {
                text: "You are in the green room. Maker-04 is here.".into(),
            },
            EventKind::Entity {
                entity_id: "a scout".into(),
                observation: "moving along the ridge".into(),
            },
            EventKind::Heartbeat,
            EventKind::Wake { day: 3 },
        ];
        for kind in events {
            let p = Event::new(1, 0, Salience::NORMAL, kind.clone()).prose();
            assert!(!p.is_empty(), "{kind:?} rendered to nothing");
            for banned in ["{", "}", "\"kind\"", "salience", "entity_id", "_ms"] {
                assert!(
                    !p.contains(banned),
                    "{kind:?} leaked {banned:?} into the character's view: {p}"
                );
            }
        }
    }

    /// One utterance, three readings. Being told something, being among those
    /// it was said to, and watching somebody else be told it are three
    /// different facts, and whether silence is rude depends on which — not
    /// something to leave to inference from phrasing.
    #[test]
    fn who_an_utterance_was_aimed_at_changes_how_it_reads() {
        let said = |to: Addressed| {
            Event::new(
                1,
                0,
                Salience::NORMAL,
                EventKind::Speech {
                    speaker: "Hess".into(),
                    text: "Well?".into(),
                    to,
                },
            )
            .prose()
        };
        let to_me = said(Addressed::You);
        let to_room = said(Addressed::Room);
        let to_other = said(Addressed::Other {
            who: "Varek".into(),
        });

        assert_ne!(to_me, to_room);
        assert_ne!(to_me, to_other);
        assert_ne!(to_room, to_other);
        assert!(to_me.contains("to you"), "{to_me}");
        assert!(to_other.contains("to Varek"), "{to_other}");
        assert!(!to_room.contains(" to "), "{to_room}");
        for p in [&to_me, &to_room, &to_other] {
            assert!(p.contains("Well?"), "{p}");
            // Reported, never quoted: a tool carries what a character meant to
            // convey, not the words it used, so quotation marks would put a
            // sentence in its mouth that it never spoke.
            assert!(!p.contains('"'), "substance rendered as a quotation: {p}");
        }
    }

    /// The situation is already prose when it arrives — it is generated from a
    /// map, not described by a caller — so rendering must not decorate it.
    #[test]
    fn a_situation_is_passed_through_untouched() {
        let text = "You are in the green room. Maker-04 is here.";
        let p = Event::new(
            1,
            0,
            Salience::IDLE,
            EventKind::Situation {
                text: format!("  {text}\n"),
            },
        )
        .prose();
        assert_eq!(p, text);
    }

    /// The character is not aware it is being scheduled. A heartbeat that says
    /// "heartbeat" teaches the model that the machinery is part of the world.
    #[test]
    fn the_heartbeat_does_not_tell_the_character_about_the_scheduler() {
        let p = Event::new(1, 0, Salience::IDLE, EventKind::Heartbeat).prose();
        for leak in ["heartbeat", "tick", "event", "scheduler", "inbox"] {
            assert!(!p.to_lowercase().contains(leak), "leaked {leak:?}: {p}");
        }
    }

    #[test]
    fn every_kind_has_a_distinct_tag() {
        let kinds = [
            EventKind::Description {
                text: String::new(),
            },
            EventKind::Speech {
                speaker: String::new(),
                text: String::new(),
                to: Addressed::Room,
            },
            EventKind::Situation {
                text: String::new(),
            },
            EventKind::Entity {
                entity_id: String::new(),
                observation: String::new(),
            },
            EventKind::Heartbeat,
            EventKind::Sleep { day: 0 },
            EventKind::Wake { day: 0 },
            EventKind::Operator {
                text: String::new(),
            },
        ];
        let mut tags: Vec<&str> = kinds.iter().map(|k| k.tag()).collect();
        tags.sort_unstable();
        let n = tags.len();
        tags.dedup();
        assert_eq!(tags.len(), n, "two kinds share a tag");
    }

    /// The wire shape is a contract with the console and with any world
    /// simulation driving this. Asserted against exact JSON, not a round trip.
    #[test]
    fn the_wire_shape_is_tagged_by_kind() {
        let k: EventKind = serde_json::from_str(
            r#"{"kind":"speech","speaker":"Hess","text":"Hold.","to":{"at":"you"}}"#,
        )
        .expect("parses");
        assert_eq!(
            k,
            EventKind::Speech {
                speaker: "Hess".into(),
                text: "Hold.".into(),
                to: Addressed::You
            }
        );

        let o: EventKind = serde_json::from_str(
            r#"{"kind":"speech","speaker":"Hess","text":"Hold.","to":{"at":"other","who":"Varek"}}"#,
        )
        .expect("parses");
        assert!(matches!(
            o,
            EventKind::Speech {
                to: Addressed::Other { ref who },
                ..
            } if who == "Varek"
        ));

        // `to` defaults to the room, so a world sim that does not model who an
        // utterance was aimed at still parses — and defaults to the reading
        // that claims least, rather than to being addressed.
        let d: EventKind =
            serde_json::from_str(r#"{"kind":"speech","speaker":"H","text":"x"}"#).expect("parses");
        assert!(matches!(
            d,
            EventKind::Speech {
                to: Addressed::Room,
                ..
            }
        ));
    }

    /// **The rouse bar is a rung of the map's ladder, not a number beside it.**
    ///
    /// `ROUSES_AT` has to be written as a literal — `Weight::as_f32` is not
    /// `const` — so this is what stops it becoming a second, quietly diverging
    /// source of truth. Move `Weight::Wake` and this fails rather than silently
    /// leaving "worth a turn at the next opportunity" unable to end a wait.
    #[test]
    fn the_rouse_bar_is_the_maps_own_wake_rung() {
        use npc_map::Weight;

        assert_eq!(Salience::ROUSES_AT, Weight::Wake.as_f32());
        assert!(Salience::from(Weight::Wake).rouses());
        for quiet in [Weight::Ambient, Weight::Note] {
            assert!(!Salience::from(quiet).rouses(), "{quiet:?} ended a wait");
        }
    }

    /// **Rousing is the lower bar, so everything that preempts also rouses.**
    /// The ordering itself is a compile-time guarantee — see `BARS_ARE_ORDERED`
    /// — and this is the behaviour that guarantee exists for.
    #[test]
    fn anything_that_preempts_also_rouses() {
        for s in [Salience::URGENT, Salience::new(0.8), Salience::new(1.0)] {
            assert!(s.preempts() && s.rouses(), "{s:?}");
        }
    }

    /// The band between the bars: ends a wait, does not cut into a character
    /// that is already thinking. Somebody speaking to the room, somebody
    /// walking in, a gesture aimed at you — the whole reason there are two
    /// bars rather than one.
    #[test]
    fn there_is_a_band_that_rouses_without_preempting() {
        for s in [Salience::new(0.6), Salience::new(0.7), Salience::new(0.79)] {
            assert!(s.rouses(), "{s:?} did not rouse");
            assert!(!s.preempts(), "{s:?} interrupted a working character");
        }
    }

    /// Below the bar, and each for its own stated reason — see `ROUSES_AT`.
    /// A wait anything at all could end is a heartbeat spelled differently.
    #[test]
    fn the_quiet_traffic_does_not_end_a_wait() {
        // The heartbeat and the standing task: they *are* the quiet.
        assert!(!Salience::IDLE.rouses());
        // Ordinary traffic, which is what a group thread is graded as.
        assert!(!Salience::NORMAL.rouses());
    }

    /// The two salience scales must agree about the only question that matters:
    /// whether a character is interrupted. They are separate types in separate
    /// crates and nothing but this holds them together.
    #[test]
    fn weight_and_salience_agree_about_preempting() {
        use npc_map::Weight;

        assert!(Salience::from(Weight::Preempt).preempts());
        for quiet in [Weight::Ambient, Weight::Note, Weight::Wake] {
            assert!(
                !Salience::from(quiet).preempts(),
                "{quiet:?} interrupts a character"
            );
        }

        // And the ladder's order survives the crossing.
        let ladder = [Weight::Ambient, Weight::Note, Weight::Wake, Weight::Preempt];
        for pair in ladder.windows(2) {
            assert!(
                Salience::from(pair[0]).get() < Salience::from(pair[1]).get(),
                "{:?} did not stay below {:?}",
                pair[0],
                pair[1]
            );
        }
    }
}
