//! Every control ZenJ names, on the FLX4 and on the virtual controller alike:
//! the faders and knobs, which hold a position, and the buttons, which are
//! pressed. One vocabulary for the maps (`midi::map`), the observed hands
//! (`hands`), the conductor, and the plans it runs.
//!
//! A control is written `deck1.fader`, `mixer.crossfader`, `browser.load2`,
//! `sampler.3`.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// An EQ band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Band {
    High,
    Mid,
    Low,
}

/// A control that holds a position, `0..=1` (a centred knob rests at 0.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Knob {
    Fader(u8),
    Trim(u8),
    Eq(u8, Band),
    /// The deck's colour filter (the FLX4's Smart CFX knob).
    Filter(u8),
    Tempo(u8),
    Crossfader,
    FxLevel,
}

/// A control that is pressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Button {
    Play(u8),
    Cue(u8),
    Sync(u8),
    /// Touching the jog wheel's top.
    Jog(u8),
    /// Hot cue pad 1–8 on a deck.
    HotCue(u8, u8),
    LoopIn(u8),
    LoopOut(u8),
    /// A four-beat auto loop on, or off when one is running.
    AutoLoop(u8),
    /// The beat FX on a deck's channel.
    Fx(u8),
    /// The browser's selected row, loaded to a deck.
    Load(u8),
    BrowseUp,
    BrowseDown,
    /// A sampler slot, 1–16.
    Sampler(u8),
    Automix,
}

/// Any control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Control {
    Knob(Knob),
    Button(Button),
}

impl Knob {
    /// Where the knob rests: faders down, knobs centred, the crossfader in
    /// the middle.
    pub fn rest(self) -> f32 {
        match self {
            Knob::Fader(_) | Knob::FxLevel => 0.0,
            _ => 0.5,
        }
    }

    /// The deck a knob is on; `None` for the mixer's own.
    pub fn deck(self) -> Option<u8> {
        match self {
            Knob::Fader(d) | Knob::Trim(d) | Knob::Eq(d, _) | Knob::Filter(d) | Knob::Tempo(d) => {
                Some(d)
            }
            Knob::Crossfader | Knob::FxLevel => None,
        }
    }
}

impl Button {
    /// The deck a button is on; `None` for the browser's, the sampler's and
    /// Automix.
    pub fn deck(self) -> Option<u8> {
        match self {
            Button::Play(d)
            | Button::Cue(d)
            | Button::Sync(d)
            | Button::Jog(d)
            | Button::HotCue(d, _)
            | Button::LoopIn(d)
            | Button::LoopOut(d)
            | Button::AutoLoop(d)
            | Button::Fx(d)
            | Button::Load(d) => Some(d),
            Button::BrowseUp | Button::BrowseDown | Button::Sampler(_) | Button::Automix => None,
        }
    }

    /// Whether a person touching it is playing the deck: transport and jog,
    /// which hand the whole mixer back (§6.3).
    pub fn is_transport(self) -> bool {
        matches!(self, Button::Play(_) | Button::Cue(_) | Button::Jog(_))
    }
}

fn band_name(b: Band) -> &'static str {
    match b {
        Band::High => "high",
        Band::Mid => "mid",
        Band::Low => "low",
    }
}

impl fmt::Display for Control {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Control::Knob(k) => match k {
                Knob::Fader(d) => write!(f, "deck{d}.fader"),
                Knob::Trim(d) => write!(f, "deck{d}.trim"),
                Knob::Eq(d, b) => write!(f, "deck{d}.eq_{}", band_name(b)),
                Knob::Filter(d) => write!(f, "deck{d}.filter"),
                Knob::Tempo(d) => write!(f, "deck{d}.tempo"),
                Knob::Crossfader => f.write_str("mixer.crossfader"),
                Knob::FxLevel => f.write_str("mixer.fx_level"),
            },
            Control::Button(b) => match b {
                Button::Play(d) => write!(f, "deck{d}.play"),
                Button::Cue(d) => write!(f, "deck{d}.cue"),
                Button::Sync(d) => write!(f, "deck{d}.sync"),
                Button::Jog(d) => write!(f, "deck{d}.jog"),
                Button::HotCue(d, n) => write!(f, "deck{d}.hotcue{n}"),
                Button::LoopIn(d) => write!(f, "deck{d}.loop_in"),
                Button::LoopOut(d) => write!(f, "deck{d}.loop_out"),
                Button::AutoLoop(d) => write!(f, "deck{d}.auto_loop"),
                Button::Fx(d) => write!(f, "deck{d}.fx"),
                Button::Load(d) => write!(f, "browser.load{d}"),
                Button::BrowseUp => f.write_str("browser.up"),
                Button::BrowseDown => f.write_str("browser.down"),
                Button::Sampler(n) => write!(f, "sampler.{n}"),
                Button::Automix => f.write_str("mixer.automix"),
            },
        }
    }
}

/// The decks a name may carry: 1 to 4.
fn deck_number(t: &str) -> Option<u8> {
    let d: u8 = t.parse().ok()?;
    (1..=4).contains(&d).then_some(d)
}

impl FromStr for Control {
    type Err = String;

    fn from_str(s: &str) -> Result<Control, String> {
        let bad = || format!("`{s}` names no control");
        let (group, name) = s.split_once('.').ok_or_else(bad)?;
        let knob = |k| Ok(Control::Knob(k));
        let button = |b| Ok(Control::Button(b));
        if let Some(d) = group.strip_prefix("deck") {
            let d = deck_number(d).ok_or_else(bad)?;
            if let Some(n) = name.strip_prefix("hotcue") {
                let n: u8 = n.parse().map_err(|_| bad())?;
                return if (1..=8).contains(&n) {
                    button(Button::HotCue(d, n))
                } else {
                    Err(bad())
                };
            }
            return match name {
                "fader" => knob(Knob::Fader(d)),
                "trim" => knob(Knob::Trim(d)),
                "eq_high" => knob(Knob::Eq(d, Band::High)),
                "eq_mid" => knob(Knob::Eq(d, Band::Mid)),
                "eq_low" => knob(Knob::Eq(d, Band::Low)),
                "filter" => knob(Knob::Filter(d)),
                "tempo" => knob(Knob::Tempo(d)),
                "play" => button(Button::Play(d)),
                "cue" => button(Button::Cue(d)),
                "sync" => button(Button::Sync(d)),
                "jog" => button(Button::Jog(d)),
                "loop_in" => button(Button::LoopIn(d)),
                "loop_out" => button(Button::LoopOut(d)),
                "auto_loop" => button(Button::AutoLoop(d)),
                "fx" => button(Button::Fx(d)),
                _ => Err(bad()),
            };
        }
        match (group, name) {
            ("mixer", "crossfader") => knob(Knob::Crossfader),
            ("mixer", "fx_level") => knob(Knob::FxLevel),
            ("mixer", "automix") => button(Button::Automix),
            ("browser", "up") => button(Button::BrowseUp),
            ("browser", "down") => button(Button::BrowseDown),
            ("browser", load) => {
                let d = load.strip_prefix("load").and_then(deck_number);
                d.map(|d| Control::Button(Button::Load(d))).ok_or_else(bad)
            }
            ("sampler", n) => {
                let n: u8 = n.parse().map_err(|_| bad())?;
                if (1..=16).contains(&n) {
                    button(Button::Sampler(n))
                } else {
                    Err(bad())
                }
            }
            _ => Err(bad()),
        }
    }
}

impl Serialize for Control {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Control {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Control, D::Error> {
        let text = String::deserialize(d)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

impl Serialize for Knob {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        Control::Knob(*self).serialize(s)
    }
}

impl<'de> Deserialize<'de> for Knob {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Knob, D::Error> {
        match Control::deserialize(d)? {
            Control::Knob(k) => Ok(k),
            Control::Button(b) => Err(serde::de::Error::custom(format!(
                "{} is a button, not a knob",
                Control::Button(b)
            ))),
        }
    }
}

impl Serialize for Button {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        Control::Button(*self).serialize(s)
    }
}

impl<'de> Deserialize<'de> for Button {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Button, D::Error> {
        match Control::deserialize(d)? {
            Control::Button(b) => Ok(b),
            Control::Knob(k) => Err(serde::de::Error::custom(format!(
                "{} is a knob, not a button",
                Control::Knob(k)
            ))),
        }
    }
}

impl From<Knob> for Control {
    fn from(k: Knob) -> Control {
        Control::Knob(k)
    }
}

impl From<Button> for Control {
    fn from(b: Button) -> Control {
        Control::Button(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every control's name reads back as itself.**
    #[test]
    fn every_name_reads_back() {
        let mut all: Vec<Control> = Vec::new();
        for d in 1..=4 {
            for k in [
                Knob::Fader(d),
                Knob::Trim(d),
                Knob::Eq(d, Band::High),
                Knob::Eq(d, Band::Mid),
                Knob::Eq(d, Band::Low),
                Knob::Filter(d),
                Knob::Tempo(d),
            ] {
                all.push(k.into());
            }
            for b in [
                Button::Play(d),
                Button::Cue(d),
                Button::Sync(d),
                Button::Jog(d),
                Button::LoopIn(d),
                Button::LoopOut(d),
                Button::AutoLoop(d),
                Button::Fx(d),
                Button::Load(d),
            ] {
                all.push(b.into());
            }
            for n in 1..=8 {
                all.push(Button::HotCue(d, n).into());
            }
        }
        for n in 1..=16 {
            all.push(Button::Sampler(n).into());
        }
        all.extend([
            Control::Knob(Knob::Crossfader),
            Control::Knob(Knob::FxLevel),
            Control::Button(Button::BrowseUp),
            Control::Button(Button::BrowseDown),
            Control::Button(Button::Automix),
        ]);
        for c in &all {
            assert_eq!(c.to_string().parse::<Control>().as_ref(), Ok(c), "{c}");
        }
        assert_eq!(Control::Knob(Knob::Eq(2, Band::Low)).to_string(), "deck2.eq_low");
        assert_eq!(Control::Button(Button::Load(1)).to_string(), "browser.load1");
    }

    #[test]
    fn a_name_outside_the_vocabulary_is_refused() {
        for s in [
            "deck5.fader",
            "deck0.play",
            "deck1.hotcue9",
            "deck1.hotcue0",
            "deck1.volume",
            "sampler.17",
            "browser.load5",
            "mixer",
            "fader",
            "",
        ] {
            assert_eq!(
                s.parse::<Control>(),
                Err(format!("`{s}` names no control")),
                "{s}"
            );
        }
    }

    #[test]
    fn transport_and_jog_are_playing_the_deck() {
        assert!(Button::Play(1).is_transport() && Button::Jog(2).is_transport());
        assert!(!Button::Sync(1).is_transport() && !Button::Load(1).is_transport());
        assert_eq!(Knob::Fader(1).rest(), 0.0);
        assert_eq!(Knob::Eq(1, Band::Low).rest(), 0.5);
        assert_eq!(Knob::Crossfader.deck(), None);
    }
}
