//! Running one plan (§5.3) against the beat: each step when the playing
//! deck's beat reaches it, each ramp moved a little every tick, and every
//! knob move checked against the hands — a control the person took back is
//! left alone. Pure: beats and a snapshot in, sends out, so a plan replays in
//! a test against a recorded trace with no rekordbox.

use std::collections::BTreeMap;

use crate::control::{Button, Knob};
use crate::deck::Snapshot;
use crate::hands::{Hands, Holder};
use crate::lights::Cue;
use crate::plan::{Move, Plan};

/// What the conductor sends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Send {
    Set(Knob, f32),
    Press(Button),
    /// A cue for the lights thread.
    Light(Cue),
}

/// Two tempos closer than this are one.
const SAME_BPM: f32 = 0.05;
/// The least change a ramp sends: one 7-bit step.
const STEP: f32 = 1.0 / 127.0;

#[derive(Debug, Clone, Copy)]
struct Ramp {
    knob: Knob,
    from: f32,
    to: f32,
    start: f64,
    beats: f64,
}

/// One plan, part way through.
#[derive(Debug)]
pub struct Runner {
    pub plan: Plan,
    /// Steps at or before this beat have run.
    ran_to: f64,
    ramps: Vec<Ramp>,
    /// What each knob was last sent.
    sent: BTreeMap<Knob, f32>,
}

impl Runner {
    /// Arm `plan`; nothing before `now` (the playing deck's beat) runs.
    pub fn new(plan: Plan, now: f64) -> Runner {
        Runner {
            plan,
            ran_to: now,
            ramps: Vec::new(),
            sent: BTreeMap::new(),
        }
    }

    /// Whether every step has run and every ramp arrived.
    pub fn done(&self) -> bool {
        self.ran_to >= self.plan.end && self.ramps.is_empty()
    }

    /// Where `knob` is: where it was last sent, else where the hardware
    /// sits, else — the playing deck's fader, which is heard — up, else at
    /// rest.
    fn value_of(&self, knob: Knob, hands: &Hands) -> f32 {
        self.sent
            .get(&knob)
            .copied()
            .or_else(|| hands.physical(knob))
            .unwrap_or(if knob == Knob::Fader(self.plan.from_deck) {
                1.0
            } else {
                knob.rest()
            })
    }

    fn set(&mut self, knob: Knob, to: f32, hands: &mut Hands, out: &mut Vec<Send>) {
        if !hands.may_move(knob) {
            return;
        }
        self.sent.insert(knob, to);
        hands.moved(knob, to);
        out.push(Send::Set(knob, to));
    }

    /// The sends due as the playing deck reaches `beat`, `lead` beats early
    /// to meet the MIDI path's latency.
    pub fn tick(&mut self, beat: f64, lead: f64, deck: &Snapshot, hands: &mut Hands) -> Vec<Send> {
        let mut out = Vec::new();
        let to = beat + lead;
        let due: Vec<(f64, Move)> = self
            .plan
            .due(self.ran_to, to)
            .map(|s| (s.at, s.act))
            .collect();
        if to > self.ran_to {
            self.ran_to = to;
        }
        let zenj = hands.holder() == Holder::Zenj;
        for (at, act) in due {
            match act {
                Move::Set { knob, to } => self.set(knob, to, hands, &mut out),
                Move::Ramp {
                    knob,
                    to: target,
                    beats,
                } => {
                    let from = self.value_of(knob, hands);
                    self.ramps.retain(|r| r.knob != knob);
                    self.ramps.push(Ramp {
                        knob,
                        from,
                        to: target,
                        start: at,
                        beats: beats.max(f64::EPSILON),
                    });
                }
                Move::Press { button } if zenj => out.push(Send::Press(button)),
                Move::Light { cue } if zenj => out.push(Send::Light(cue)),
                Move::Start { deck: d } if zenj => {
                    if !deck.deck(d).is_some_and(|x| x.playing) {
                        out.push(Send::Press(Button::Play(d)));
                    }
                }
                Move::Stop { deck: d } if zenj => {
                    if deck.deck(d).is_some_and(|x| x.playing) {
                        out.push(Send::Press(Button::Play(d)));
                    }
                }
                Move::Match { deck: d } if zenj => {
                    let master = deck.deck(self.plan.from_deck).map(|x| x.bpm);
                    let this = deck.deck(d).map(|x| x.bpm);
                    if let (Some(m), Some(t)) = (master, this) {
                        if (m - t).abs() > SAME_BPM {
                            out.push(Send::Press(Button::Sync(d)));
                        }
                    }
                }
                _ => {}
            }
        }
        let ramps = std::mem::take(&mut self.ramps);
        for r in ramps {
            let f = ((to - r.start) / r.beats).clamp(0.0, 1.0) as f32;
            let v = r.from + (r.to - r.from) * f;
            let last = self.sent.get(&r.knob).copied();
            if f >= 1.0 || last.is_none_or(|l| (l - v).abs() >= STEP) {
                self.set(r.knob, v, hands, &mut out);
            }
            if f < 1.0 && hands.may_move(r.knob) {
                self.ramps.push(r);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::brain::transition::{plan, Pair};
    use crate::control::Band;
    use crate::deck::Deck;
    use crate::hands::Combos;
    use crate::midi::{Reading, Touch};
    use crate::music::track::fixture::club;

    fn snapshot(a_playing: bool, b_playing: bool, b_bpm: f32) -> Snapshot {
        let deck = |n, playing, bpm| Deck {
            number: n,
            track: None,
            bpm,
            position_ms: 0.0,
            beat: None,
            playing,
        };
        Snapshot {
            decks: vec![deck(1, a_playing, 128.0), deck(2, b_playing, b_bpm)],
            master: Some(1),
            blind: None,
        }
    }

    fn mixing() -> Hands {
        let mut h = Hands::new(Combos::default());
        h.set_holder(Holder::Zenj);
        h
    }

    /// **A blend replays against the beat**: the sync and silencing before,
    /// the start on the phrase, the fader ramp in steps, the bass swap, and
    /// the playing deck stopped at the end.
    #[test]
    fn a_blend_replays_beat_by_beat() {
        let a = club("a", 128.0, "8A", 0.6);
        let b = club("b", 126.0, "9A", 0.65);
        let p = plan(&Pair {
            from: &a,
            from_deck: 1,
            to: &b,
            to_deck: 2,
            now: 100.0,
        });
        let mut r = Runner::new(p, 100.0);
        let mut h = mixing();
        let s = snapshot(true, false, 126.0);
        assert!(
            r.tick(200.0, 0.0, &s, &mut h).is_empty(),
            "before the first step"
        );
        let prep = r.tick(208.0, 0.0, &s, &mut h);
        assert_eq!(
            prep,
            vec![
                Send::Press(Button::Sync(2)),
                Send::Set(Knob::Fader(2), 0.0),
                Send::Set(Knob::Filter(2), 0.5),
                Send::Set(Knob::Eq(2, Band::Low), 0.0),
            ]
        );
        let start = r.tick(224.0, 0.0, &s, &mut h);
        assert_eq!(
            start,
            vec![Send::Press(Button::Play(2))],
            "the ramp starts where it is"
        );
        let s = snapshot(true, true, 128.0);
        let mid = r.tick(232.0, 0.0, &s, &mut h);
        assert_eq!(mid, vec![Send::Set(Knob::Fader(2), 0.5)]);
        let swap = r.tick(240.0, 0.0, &s, &mut h);
        assert!(swap.contains(&Send::Set(Knob::Fader(2), 1.0)));
        assert!(swap.contains(&Send::Set(Knob::Eq(1, Band::Low), 0.0)));
        assert!(
            swap.contains(&Send::Set(Knob::Fader(1), 1.0)),
            "the heard deck's fader is up"
        );
        let end = r.tick(256.0, 0.0, &s, &mut h);
        assert!(end.contains(&Send::Set(Knob::Fader(1), 0.0)));
        assert!(
            end.contains(&Send::Press(Button::Play(1))),
            "stopped: it was playing"
        );
        assert!(r.done());
    }

    /// **A control the person took back is left alone**, ramps and all;
    /// and with the mixer handed back, nothing is pressed.
    #[test]
    fn a_taken_control_is_left_alone() {
        let a = club("a", 128.0, "8A", 0.6);
        let b = club("b", 128.0, "8A", 0.6);
        let p = plan(&Pair {
            from: &a,
            from_deck: 1,
            to: &b,
            to_deck: 2,
            now: 100.0,
        });
        let mut r = Runner::new(p, 100.0);
        let mut h = mixing();
        let s = snapshot(true, false, 128.0);
        r.tick(224.0, 0.0, &s, &mut h);
        h.touch(&Touch {
            control: Knob::Fader(2).into(),
            reading: Reading::At(0.3),
            at: Instant::now(),
        });
        let after = r.tick(232.0, 0.0, &s, &mut h);
        assert!(!after
            .iter()
            .any(|x| matches!(x, Send::Set(Knob::Fader(2), _))));
        h.set_holder(Holder::Human);
        assert!(r.tick(256.0, 0.0, &s, &mut h).is_empty());
    }
}
