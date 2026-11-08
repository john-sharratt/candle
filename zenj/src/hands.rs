//! A person and ZenJ on the same mixer (§6.3). ZenJ observes every control
//! the FLX4 sends, so it knows where each physical one sits and whether a
//! person is playing. Moving toward more ZenJ is always explicit — a pad
//! combination or a word to Zen — and moving back is immediate: a touch on a
//! mixer control takes that control back, a touch on a deck's transport or
//! jog takes the whole mixer.
//!
//! **No jump on hand-back.** The FLX4's controls are not motorised: a control
//! ZenJ moved in software sits somewhere the hardware does not. Before
//! handing the mixer back, the conductor walks each one it moved to where the
//! hardware is ([`Hands::walk_back`]), so the first touch is seamless.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::control::{Button, Control, Knob};
use crate::midi::{Reading, Touch};
use crate::plan::Move;

/// Who has the decks: a person, ZenJ beside them, or ZenJ. Named apart from
/// the house's mode (`docs/house_modes.md`), of which the decks are one part.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Holder {
    /// A person: ZenJ keeps *NYE Next* stocked and touches nothing on air.
    #[default]
    Human,
    /// ZenJ also loads the next track on the idle deck and cues it.
    Assist,
    /// ZenJ mixes.
    Zenj,
}

impl Holder {
    pub fn name(self) -> &'static str {
        match self {
            Holder::Human => "human",
            Holder::Assist => "assist",
            Holder::Zenj => "zenj",
        }
    }

    pub fn parse(s: &str) -> Option<Holder> {
        match s {
            "human" => Some(Holder::Human),
            "assist" => Some(Holder::Assist),
            "zenj" => Some(Holder::Zenj),
            _ => None,
        }
    }
}

/// The pad combinations that ask for more ZenJ (§15.1): every button of a
/// combination held at once.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Combos {
    #[serde(default)]
    pub assist: Vec<Control>,
    #[serde(default)]
    pub zenj: Vec<Control>,
}

/// What a touch changed.
#[derive(Debug, Clone, PartialEq)]
pub enum Change {
    /// The person took one control back; ZenJ leaves it alone now.
    TookBack(Knob),
    /// The person took the whole mixer back.
    HandedBack,
    /// A pad combination asked for the decks to change hands.
    Asked(Holder),
}

#[derive(Debug, Default)]
pub struct Hands {
    holder: Holder,
    /// Where each physical knob sits, as the FLX4 last said.
    physical: BTreeMap<Knob, f32>,
    /// Where ZenJ has put each knob it moved, while it holds it.
    held: BTreeMap<Knob, f32>,
    /// Controls the person took back while ZenJ mixes.
    taken: BTreeSet<Knob>,
    /// Buttons held down now.
    down: BTreeSet<Button>,
    combos: Combos,
}

impl Hands {
    pub fn new(combos: Combos) -> Hands {
        Hands {
            combos,
            ..Hands::default()
        }
    }

    pub fn holder(&self) -> Holder {
        self.holder
    }

    /// Give the decks to `holder` explicitly. Back to a person starts the
    /// walk back.
    pub fn set_holder(&mut self, holder: Holder) {
        if holder == Holder::Human {
            self.taken.clear();
            self.held.clear();
        }
        self.holder = holder;
    }

    /// Whether ZenJ may move `knob` now: it mixes, and the person has not
    /// taken that control back.
    pub fn may_move(&self, knob: Knob) -> bool {
        self.holder() == Holder::Zenj && !self.taken.contains(&knob)
    }

    /// ZenJ moved `knob` to `to`.
    pub fn moved(&mut self, knob: Knob, to: f32) {
        self.held.insert(knob, to);
    }

    pub fn physical(&self, knob: Knob) -> Option<f32> {
        self.physical.get(&knob).copied()
    }

    /// What `t` changes.
    pub fn touch(&mut self, t: &Touch) -> Option<Change> {
        match (t.control, t.reading) {
            (Control::Knob(k), Reading::At(at)) => {
                self.physical.insert(k, at);
                if self.holder() != Holder::Human && self.held.remove(&k).is_some() {
                    self.taken.insert(k);
                    return Some(Change::TookBack(k));
                }
                None
            }
            (Control::Button(b), Reading::Pressed) => {
                self.down.insert(b);
                // The conductor walks the mixer back, then lets go.
                if b.is_transport() && self.holder() != Holder::Human {
                    return Some(Change::HandedBack);
                }
                for (holder, combo) in [
                    (Holder::Zenj, &self.combos.zenj),
                    (Holder::Assist, &self.combos.assist),
                ] {
                    let held = !combo.is_empty()
                        && combo
                            .iter()
                            .all(|c| matches!(c, Control::Button(x) if self.down.contains(x)));
                    if held && self.holder() != holder {
                        return Some(Change::Asked(holder));
                    }
                }
                None
            }
            (Control::Button(b), Reading::Released) => {
                self.down.remove(&b);
                None
            }
            _ => None,
        }
    }

    /// The moves that walk every knob ZenJ holds to where the hardware sits,
    /// over `beats`, starting at beat `at`; the knobs are released.
    pub fn walk_back(&mut self, at: f64, beats: f64) -> Vec<(f64, Move)> {
        let held = std::mem::take(&mut self.held);
        held.into_iter()
            .filter_map(|(knob, now)| {
                let to = self.physical(knob).unwrap_or(knob.rest());
                ((to - now).abs() > 1.0 / 127.0).then_some((at, Move::Ramp { knob, to, beats }))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::control::Band;

    fn knob(k: Knob, at: f32) -> Touch {
        Touch {
            control: k.into(),
            reading: Reading::At(at),
            at: Instant::now(),
        }
    }

    fn press(b: Button, down: bool) -> Touch {
        Touch {
            control: b.into(),
            reading: if down {
                Reading::Pressed
            } else {
                Reading::Released
            },
            at: Instant::now(),
        }
    }

    fn combos() -> Combos {
        Combos {
            assist: vec![Button::HotCue(1, 7).into(), Button::HotCue(1, 8).into()],
            zenj: vec![Button::HotCue(2, 7).into(), Button::HotCue(2, 8).into()],
        }
    }

    /// **A touch on a control ZenJ holds takes it back; on transport, the
    /// whole mixer.**
    #[test]
    fn a_touch_takes_back_a_control_or_the_mixer() {
        let mut h = Hands::new(combos());
        h.set_holder(Holder::Zenj);
        h.moved(Knob::Fader(2), 1.0);
        assert!(h.may_move(Knob::Fader(2)));
        assert_eq!(
            h.touch(&knob(Knob::Fader(1), 0.8)),
            None,
            "not held: just noted"
        );
        assert_eq!(
            h.touch(&knob(Knob::Fader(2), 0.6)),
            Some(Change::TookBack(Knob::Fader(2)))
        );
        assert!(!h.may_move(Knob::Fader(2)));
        assert!(h.may_move(Knob::Fader(1)));
        assert_eq!(
            h.touch(&press(Button::Jog(1), true)),
            Some(Change::HandedBack)
        );
        h.set_holder(Holder::Human);
        assert!(!h.may_move(Knob::Fader(1)));
    }

    /// **More ZenJ is only ever asked for**: by a whole combination held at
    /// once, never by part of one.
    #[test]
    fn a_combination_held_whole_asks_for_its_holder() {
        let mut h = Hands::new(combos());
        assert_eq!(h.touch(&press(Button::HotCue(2, 7), true)), None);
        assert_eq!(h.touch(&press(Button::HotCue(2, 7), false)), None);
        assert_eq!(
            h.touch(&press(Button::HotCue(2, 8), true)),
            None,
            "the first was let go"
        );
        assert_eq!(
            h.touch(&press(Button::HotCue(2, 7), true)),
            Some(Change::Asked(Holder::Zenj))
        );
        h.set_holder(Holder::Zenj);
        assert_eq!(h.touch(&press(Button::HotCue(1, 7), true)), None);
        assert_eq!(
            h.touch(&press(Button::HotCue(1, 8), true)),
            Some(Change::Asked(Holder::Assist))
        );
    }

    /// **The walk back takes each held knob to the hardware**, over the
    /// beats given; a knob already there, or the hardware's unseen one at
    /// its rest, is handled too.
    #[test]
    fn the_walk_back_meets_the_hardware() {
        let mut h = Hands::new(combos());
        h.touch(&knob(Knob::Fader(1), 0.2));
        h.touch(&knob(Knob::Eq(2, Band::Low), 0.5));
        h.set_holder(Holder::Zenj);
        h.moved(Knob::Fader(1), 1.0);
        h.moved(Knob::Eq(2, Band::Low), 0.5);
        h.moved(Knob::Filter(2), 0.1);
        let walk = h.walk_back(64.0, 4.0);
        assert_eq!(
            walk,
            vec![
                (
                    64.0,
                    Move::Ramp {
                        knob: Knob::Fader(1),
                        to: 0.2,
                        beats: 4.0
                    }
                ),
                (
                    64.0,
                    Move::Ramp {
                        knob: Knob::Filter(2),
                        to: 0.5,
                        beats: 4.0
                    }
                ),
            ]
        );
        assert!(h.walk_back(68.0, 4.0).is_empty(), "released");
    }
}
