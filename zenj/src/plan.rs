//! A transition plan (§5.3): the moves that take the room from the playing
//! track to the next, each at a beat of the playing track. A plan is data,
//! written before it runs — so it is shown, logged, overridden, and replayed in
//! tests with no rekordbox.

use serde::{Deserialize, Serialize};

use crate::control::{Button, Knob};
use crate::music::TrackId;

/// One move the conductor makes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Move {
    /// Put a knob at a position at once.
    Set { knob: Knob, to: f32 },
    /// Move a knob from wherever it is to `to`, evenly over `beats`.
    Ramp { knob: Knob, to: f32, beats: f64 },
    /// Press a button.
    Press { button: Button },
    /// Start a deck if it is stopped.
    Start { deck: u8 },
    /// Stop a deck if it is playing.
    Stop { deck: u8 },
    /// Sync a deck to the master if its tempo differs from the master's.
    Match { deck: u8 },
}

/// A move at a beat of the playing track, counted from its first beat.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub at: f64,
    #[serde(rename = "move")]
    pub act: Move,
}

/// How a pair is mixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Style {
    /// Both play together; the bass swaps half way.
    Blend,
    /// The next comes in on a phrase boundary, the playing one out at once.
    Cut,
    /// The playing one echoes out as the next starts.
    EchoOut,
    /// The playing one is filtered away, then the next starts.
    FilterSweep,
}

impl Style {
    pub fn name(self) -> &'static str {
        match self {
            Style::Blend => "blend",
            Style::Cut => "cut",
            Style::EchoOut => "echo out",
            Style::FilterSweep => "filter sweep",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub from: TrackId,
    pub to: TrackId,
    pub from_deck: u8,
    pub to_deck: u8,
    pub style: Style,
    /// The playing track's beat at which the next starts.
    pub start: f64,
    /// The playing track's beat by which the move is done.
    pub end: f64,
    /// The steps, in the order they run.
    pub steps: Vec<Step>,
    /// One line on why it is mixed this way.
    pub why: String,
}

impl Plan {
    /// The steps due after beat `from` up to and including `to`.
    pub fn due(&self, from: f64, to: f64) -> impl Iterator<Item = &Step> {
        self.steps.iter().filter(move |s| s.at > from && s.at <= to)
    }

    /// The beat of the plan's first step: when it must be armed by.
    pub fn first(&self) -> f64 {
        self.steps.first().map_or(self.start, |s| s.at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_steps_due_in_a_window_are_those_after_its_start_through_its_end() {
        let s = |at| Step {
            at,
            act: Move::Start { deck: 2 },
        };
        let p = Plan {
            from: TrackId("a".into()),
            to: TrackId("b".into()),
            from_deck: 1,
            to_deck: 2,
            style: Style::Cut,
            start: 8.0,
            end: 8.0,
            steps: vec![s(4.0), s(8.0), s(8.0), s(12.0)],
            why: String::new(),
        };
        assert_eq!(p.due(0.0, 4.0).count(), 1);
        assert_eq!(p.due(4.0, 8.0).count(), 2, "both at the boundary, once");
        assert_eq!(p.due(8.0, 11.9).count(), 0);
        assert_eq!(p.first(), 4.0);
    }

    #[test]
    fn a_plan_reads_back_from_its_json() {
        let p = Plan {
            from: TrackId("a".into()),
            to: TrackId("b".into()),
            from_deck: 1,
            to_deck: 2,
            style: Style::Blend,
            start: 224.0,
            end: 256.0,
            steps: vec![Step {
                at: 224.0,
                act: Move::Ramp {
                    knob: Knob::Fader(2),
                    to: 1.0,
                    beats: 16.0,
                },
            }],
            why: "outro over intro".into(),
        };
        let text = serde_json::to_string(&p).unwrap();
        assert!(text.contains(r#""move":{"ramp":{"knob":"#), "{text}");
        assert_eq!(serde_json::from_str::<Plan>(&text).unwrap(), p);
    }
}
