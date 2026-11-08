//! The house's modes (`docs/house_modes.md` §2): what the home is set up for.
//! A mode decides what the devices are doing, whether rekordbox is open, and
//! so whether BlackWorld is held for the decks. Two setups that differ only in
//! how loud or how lively are one mode.

use serde::{Deserialize, Serialize};

/// What the house is set up for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Nobody home.
    Away,
    /// Home, asleep: nothing makes a sound.
    Asleep,
    /// Somebody in, nothing on.
    Home,
    /// Music in the background.
    Listening,
    /// The television: a film, a show, a game.
    Watching,
    /// Zen at the decks.
    Party,
}

impl Mode {
    pub const ALL: [Mode; 6] = [
        Mode::Away,
        Mode::Asleep,
        Mode::Home,
        Mode::Listening,
        Mode::Watching,
        Mode::Party,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Mode::Away => "away",
            Mode::Asleep => "asleep",
            Mode::Home => "home",
            Mode::Listening => "listening",
            Mode::Watching => "watching",
            Mode::Party => "party",
        }
    }

    pub fn parse(s: &str) -> Option<Mode> {
        Mode::ALL
            .into_iter()
            .find(|m| m.name().eq_ignore_ascii_case(s.trim()))
    }

    /// What it is for, in a line.
    pub fn what(self) -> &'static str {
        match self {
            Mode::Away => "nobody is home",
            Mode::Asleep => "the house is asleep",
            Mode::Home => "somebody is in and nothing is on",
            Mode::Listening => "music is on in the background",
            Mode::Watching => "the television is on",
            Mode::Party => "Zen is at the decks",
        }
    }

    /// Whether rekordbox is open in this mode: for a party, and never else.
    pub fn decks_open(self) -> bool {
        self == Mode::Party
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_mode_reads_back_by_its_name() {
        for m in Mode::ALL {
            assert_eq!(Mode::parse(m.name()), Some(m));
        }
        assert_eq!(Mode::parse(" Party "), Some(Mode::Party));
        assert_eq!(Mode::parse("dinner"), None);
    }

    /// **Only a party opens the decks**: every other mode closes them.
    #[test]
    fn only_a_party_opens_the_decks() {
        let open: Vec<Mode> = Mode::ALL.into_iter().filter(|m| m.decks_open()).collect();
        assert_eq!(open, [Mode::Party]);
    }
}
