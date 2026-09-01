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
//! before it lands makes a delusion permanent. So salience does two things here
//! — it decides whether an arrival *preempts* (forces a tick now) and it biases
//! what the gather pulls — and it never decides whether the event is recorded.
//! Everything that arrives, lands.

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

    /// The more urgent of two. `f32` has no `Ord`, so this cannot be `max`.
    pub fn max_of(self, other: Salience) -> Salience {
        if other.0 > self.0 {
            other
        } else {
            self
        }
    }
}

impl Default for Salience {
    fn default() -> Self {
        Salience::NORMAL
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
        /// Whether it was addressed to this character or merely overheard. The
        /// difference decides whether silence is rude or normal, so it cannot be
        /// left for the model to infer from phrasing.
        #[serde(default)]
        directed: bool,
    },
    /// A spatial picture at a zoom band. Maps **replace** within their band —
    /// see [`EventKind::replaces`].
    Map {
        zoom: String,
        ascii: String,
        #[serde(default)]
        legend: Option<String>,
    },
    /// A specific entity was observed doing something.
    Entity {
        entity_id: String,
        observation: String,
    },
    /// The scheduled wake. Carries no content: its whole purpose is that "nothing
    /// arrived" cannot mean "dead forever".
    Heartbeat,
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
            EventKind::Map { .. } => "map",
            EventKind::Entity { .. } => "entity",
            EventKind::Heartbeat => "heartbeat",
            EventKind::Sleep { .. } => "sleep",
            EventKind::Wake { .. } => "wake",
            EventKind::Operator { .. } => "operator",
        }
    }

    /// The band this event supersedes within, if it supersedes anything.
    ///
    /// Only maps do. Twelve stale tactical maps in the gather is twelve chances
    /// to act on a position that no longer exists, so a new map at a zoom retires
    /// the previous one at that zoom. Descriptions accumulate — a thing that
    /// happened stays happened.
    pub fn replaces(&self) -> Option<String> {
        match self {
            EventKind::Map { zoom, .. } => Some(format!("map:{zoom}")),
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
            EventKind::Speech {
                speaker,
                text,
                directed,
            } => {
                let t = text.trim();
                if *directed {
                    format!("{speaker} says to you: \"{t}\"")
                } else {
                    format!("You overhear {speaker} say: \"{t}\"")
                }
            }
            EventKind::Map {
                zoom,
                ascii,
                legend,
            } => {
                let mut s = format!("You take in your surroundings at {zoom} range:\n\n```\n");
                s.push_str(ascii.trim_end());
                s.push_str("\n```");
                if let Some(l) = legend {
                    // The legend is what makes the glyphs mean anything. Kept
                    // outside the fence so it reads as explanation rather than as
                    // more map.
                    s.push_str("\n\nKey: ");
                    s.push_str(l.trim());
                }
                s
            }
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

    /// Maps replace within a zoom band; nothing else replaces anything. This is
    /// the one departure from append-only in the whole engine, so it is pinned.
    #[test]
    fn maps_replace_within_a_band_and_nothing_else_replaces() {
        let m = EventKind::Map {
            zoom: "tactical".into(),
            ascii: "..#..".into(),
            legend: None,
        };
        assert_eq!(m.replaces().as_deref(), Some("map:tactical"));

        let other = EventKind::Map {
            zoom: "strategic".into(),
            ascii: "..".into(),
            legend: None,
        };
        assert_ne!(
            m.replaces(),
            other.replaces(),
            "two zoom bands must not supersede each other"
        );

        for k in [
            EventKind::Description { text: "x".into() },
            EventKind::Heartbeat,
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
                directed: true,
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

    /// Directed and overheard speech must not read alike — whether silence is
    /// rude depends on it, and that is not something to leave to inference.
    #[test]
    fn directed_speech_reads_differently_from_overheard() {
        let d = Event::new(
            1,
            0,
            Salience::NORMAL,
            EventKind::Speech {
                speaker: "Hess".into(),
                text: "Well?".into(),
                directed: true,
            },
        )
        .prose();
        let o = Event::new(
            1,
            0,
            Salience::NORMAL,
            EventKind::Speech {
                speaker: "Hess".into(),
                text: "Well?".into(),
                directed: false,
            },
        )
        .prose();
        assert_ne!(d, o);
        assert!(d.contains("to you"));
        assert!(o.contains("overhear"));
    }

    /// A map's glyphs are meaningless without its key, and the key must not sit
    /// inside the fence where it reads as more map.
    #[test]
    fn a_map_fences_its_ascii_and_keeps_the_legend_outside() {
        let p = Event::new(
            1,
            0,
            Salience::NORMAL,
            EventKind::Map {
                zoom: "tactical".into(),
                ascii: "#.#\n...".into(),
                legend: Some("# wall, . floor".into()),
            },
        )
        .prose();
        let body = p.split("```").nth(1).expect("a fenced block");
        assert!(body.contains("#.#"));
        assert!(!body.contains("wall"), "the legend leaked inside the fence");
        assert!(p.contains("Key: # wall"));
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
                directed: false,
            },
            EventKind::Map {
                zoom: String::new(),
                ascii: String::new(),
                legend: None,
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
            r#"{"kind":"speech","speaker":"Hess","text":"Hold.","directed":true}"#,
        )
        .expect("parses");
        assert_eq!(
            k,
            EventKind::Speech {
                speaker: "Hess".into(),
                text: "Hold.".into(),
                directed: true
            }
        );
        // `directed` defaults, so a world sim that does not model it still parses.
        let d: EventKind =
            serde_json::from_str(r#"{"kind":"speech","speaker":"H","text":"x"}"#).expect("parses");
        assert!(matches!(
            d,
            EventKind::Speech {
                directed: false,
                ..
            }
        ));
    }
}
