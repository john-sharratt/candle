//! A lighting cue (§9.4): a move of a transition plan, armed, logged and
//! replayed like any other. The conductor hands the lights a plan's cues
//! the moment it is armed — each with its beat, so they are planned ahead
//! with everything else — and takes them back when the decks are handed
//! back.

use serde::{Deserialize, Serialize};

/// What the conductor tells the lights of the plan's cues.
#[derive(Debug, Clone, PartialEq)]
pub enum Cues {
    /// The armed plan's cues, each at its beat of the playing deck.
    Armed(Vec<(f64, Cue)>),
    /// The plan is gone: no cue stands.
    Clear,
}

/// What a cue does to the room.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cue {
    /// Everything dark, until the next cue or the next phrase.
    Blackout,
    /// One full-bright flash, sent twice.
    Hit,
    /// The look eases to the next track's over `beats`.
    Fade { beats: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cue_reads_back_from_its_json() {
        for c in [Cue::Blackout, Cue::Hit, Cue::Fade { beats: 32.0 }] {
            let text = serde_json::to_string(&c).unwrap();
            assert_eq!(serde_json::from_str::<Cue>(&text).unwrap(), c);
        }
        assert_eq!(
            serde_json::to_string(&Cue::Fade { beats: 8.0 }).unwrap(),
            r#"{"fade":{"beats":8.0}}"#
        );
    }
}
