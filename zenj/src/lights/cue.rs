//! A lighting cue (§9.4): a move of a transition plan, armed, logged and
//! replayed like any other, which the conductor hands to the lights thread
//! when its beat comes.

use serde::{Deserialize, Serialize};

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
