//! The transition planner (§5.3): from both tracks' phrase maps, their keys,
//! tempos and the energy step between them, where to leave the playing track,
//! how to mix the next one in, and every move of it on the beat.
//!
//! The next track always enters from its first beat, which rekordbox cues on
//! load, so the plan needs no seek. Every move lands on a bar.

use crate::control::{Band, Button, Knob};
use crate::music::{PhraseKind, Profile};
use crate::plan::{Move, Plan, Step, Style};

/// The fewest beats between now and the transition's start: time to load,
/// cue and arm.
pub const LEAD: f64 = 32.0;

/// A blend's length bounds, in beats.
const BLEND_MIN: u32 = 16;
const BLEND_MAX: u32 = 64;
/// A blend's length when neither track's phrases say.
const BLEND_DEFAULT: u32 = 32;

/// The tempo gap past which a pair is not blended: one of them would be
/// stretched audibly for the length of the blend.
const TEMPO_BLEND: f32 = 0.04;
/// The energy rise past which the next comes in on a cut.
const ENERGY_CUT: f32 = 0.25;

/// What the planner is given.
pub struct Pair<'a> {
    pub from: &'a Profile,
    pub from_deck: u8,
    pub to: &'a Profile,
    pub to_deck: u8,
    /// The playing track's beat now.
    pub now: f64,
}

/// How far apart two tempos are, allowing half and double time.
pub fn tempo_gap(a: f32, b: f32) -> f32 {
    [b, b * 2.0, b / 2.0]
        .iter()
        .map(|x| (x / a - 1.0).abs())
        .fold(f32::INFINITY, f32::min)
}

/// The phrase boundary of `track` where the transition starts: its outro if
/// it has one far enough ahead, else the last boundary that leaves `room`
/// beats of track after it, else the first boundary past `earliest`.
fn out_point(track: &Profile, earliest: f64, room: u32) -> u32 {
    let earliest_beat = earliest.ceil().max(0.0) as u32;
    if let Some(outro) = track
        .phrases
        .iter()
        .rev()
        .find(|p| p.kind == PhraseKind::Outro && p.start >= earliest_beat)
    {
        return outro.start;
    }
    let total = track.beats();
    let bounds = track.boundaries();
    bounds
        .iter()
        .rev()
        .find(|&&b| b >= earliest_beat && b + room <= total)
        .or_else(|| bounds.iter().find(|&&b| b >= earliest_beat))
        .copied()
        .unwrap_or_else(|| {
            // Past the last boundary: the next bar.
            earliest_beat.div_ceil(4) * 4
        })
}

/// How many beats the playing track has from `start` to its end.
fn left_after(track: &Profile, start: u32) -> u32 {
    track.beats().saturating_sub(start)
}

/// The blend's length: the shorter of the playing track's outro and the next
/// one's intro, within bounds, on whole bars.
fn blend_len(from: &Profile, start: u32, to: &Profile) -> u32 {
    let outro = from
        .phrases
        .iter()
        .find(|p| p.start == start && p.kind == PhraseKind::Outro)
        .map(|p| p.beats())
        .unwrap_or(left_after(from, start));
    let intro = to
        .phrases
        .first()
        .filter(|p| p.kind == PhraseKind::Intro)
        .map(|p| p.beats());
    let len = match intro {
        Some(i) => outro.min(i),
        None if from.phrases.is_empty() => BLEND_DEFAULT,
        None => outro.min(BLEND_DEFAULT),
    };
    (len.clamp(BLEND_MIN, BLEND_MAX) / 4) * 4
}

/// The style for the pair, and why.
fn style(pair: &Pair<'_>) -> (Style, String) {
    let (a, b) = (pair.from, pair.to);
    let gap = tempo_gap(a.grid.bpm, b.grid.bpm);
    let keys = match (a.key, b.key) {
        (Some(x), Some(y)) => Some(x.fit(y)),
        _ => None,
    };
    let rise = b.energy - a.energy;
    if gap > TEMPO_BLEND {
        return (
            Style::FilterSweep,
            format!(
                "{:.0} → {:.0} BPM is too far to blend: filter out, then in",
                a.grid.bpm, b.grid.bpm
            ),
        );
    }
    if keys.is_some_and(|k| !k.mixes()) {
        let (x, y) = (a.key.unwrap(), b.key.unwrap());
        return (
            Style::EchoOut,
            format!("{x} → {y} clash: echo out rather than lay them together"),
        );
    }
    if rise >= ENERGY_CUT {
        return (
            Style::Cut,
            format!("a lift of {rise:.2}: cut in on the phrase"),
        );
    }
    let has_outro = a.phrases.iter().any(|p| p.kind == PhraseKind::Outro);
    let has_intro = b
        .phrases
        .first()
        .is_some_and(|p| p.kind == PhraseKind::Intro);
    if !a.phrases.is_empty() && !has_outro && !has_intro {
        return (
            Style::EchoOut,
            "no outro and no intro to lay together: echo out".into(),
        );
    }
    let key_line = match keys {
        Some(k) => format!(" ({:?} keys)", k).to_lowercase(),
        None => String::new(),
    };
    (Style::Blend, format!("outro over intro{key_line}"))
}

fn at(at: f64, act: Move) -> Step {
    Step { at, act }
}

/// The plan for `pair`.
pub fn plan(pair: &Pair<'_>) -> Plan {
    let (a, b) = (pair.from_deck, pair.to_deck);
    let (style, why) = style(pair);
    let room = match style {
        Style::Blend => BLEND_DEFAULT,
        _ => 8,
    };
    let start = out_point(pair.from, pair.now + LEAD, room);
    let s = start as f64;
    let low = |deck| Knob::Eq(deck, Band::Low);
    // Before any style: the next deck silent and its bass out, synced.
    let mut steps = vec![
        at(s - 16.0, Move::Match { deck: b }),
        at(
            s - 16.0,
            Move::Set {
                knob: Knob::Fader(b),
                to: 0.0,
            },
        ),
        at(
            s - 16.0,
            Move::Set {
                knob: Knob::Filter(b),
                to: 0.5,
            },
        ),
    ];
    let end = match style {
        Style::Blend => {
            let len = blend_len(pair.from, start, pair.to) as f64;
            let half = (len / 8.0).floor() * 4.0;
            steps.extend([
                at(
                    s - 16.0,
                    Move::Set {
                        knob: low(b),
                        to: 0.0,
                    },
                ),
                at(s, Move::Start { deck: b }),
                at(
                    s,
                    Move::Ramp {
                        knob: Knob::Fader(b),
                        to: 1.0,
                        beats: half,
                    },
                ),
                at(
                    s + half,
                    Move::Set {
                        knob: low(a),
                        to: 0.0,
                    },
                ),
                at(
                    s + half,
                    Move::Set {
                        knob: low(b),
                        to: 0.5,
                    },
                ),
                at(
                    s + half,
                    Move::Ramp {
                        knob: Knob::Fader(a),
                        to: 0.0,
                        beats: len - half,
                    },
                ),
                at(s + len, Move::Stop { deck: a }),
                at(
                    s + len,
                    Move::Set {
                        knob: low(a),
                        to: 0.5,
                    },
                ),
            ]);
            s + len
        }
        Style::Cut => {
            steps.extend([
                at(
                    s - 16.0,
                    Move::Set {
                        knob: low(b),
                        to: 0.5,
                    },
                ),
                at(s, Move::Start { deck: b }),
                at(
                    s,
                    Move::Set {
                        knob: Knob::Fader(b),
                        to: 1.0,
                    },
                ),
                at(
                    s,
                    Move::Set {
                        knob: Knob::Fader(a),
                        to: 0.0,
                    },
                ),
                at(s + 4.0, Move::Stop { deck: a }),
            ]);
            s + 4.0
        }
        Style::EchoOut => {
            steps.extend([
                at(
                    s - 16.0,
                    Move::Set {
                        knob: low(b),
                        to: 0.5,
                    },
                ),
                at(
                    s - 4.0,
                    Move::Set {
                        knob: Knob::FxLevel,
                        to: 0.7,
                    },
                ),
                at(
                    s - 4.0,
                    Move::Press {
                        button: Button::Fx(a),
                    },
                ),
                at(
                    s,
                    Move::Set {
                        knob: Knob::Fader(a),
                        to: 0.0,
                    },
                ),
                at(s, Move::Start { deck: b }),
                at(
                    s,
                    Move::Ramp {
                        knob: Knob::Fader(b),
                        to: 1.0,
                        beats: 4.0,
                    },
                ),
                at(
                    s + 8.0,
                    Move::Press {
                        button: Button::Fx(a),
                    },
                ),
                at(s + 8.0, Move::Stop { deck: a }),
            ]);
            s + 8.0
        }
        Style::FilterSweep => {
            steps.extend([
                at(
                    s - 16.0,
                    Move::Set {
                        knob: low(b),
                        to: 0.5,
                    },
                ),
                at(
                    s - 16.0,
                    Move::Ramp {
                        knob: Knob::Filter(a),
                        to: 0.1,
                        beats: 16.0,
                    },
                ),
                at(
                    s,
                    Move::Set {
                        knob: Knob::Fader(a),
                        to: 0.0,
                    },
                ),
                at(s, Move::Start { deck: b }),
                at(
                    s,
                    Move::Set {
                        knob: Knob::Fader(b),
                        to: 1.0,
                    },
                ),
                at(
                    s + 4.0,
                    Move::Set {
                        knob: Knob::Filter(a),
                        to: 0.5,
                    },
                ),
                at(s + 4.0, Move::Stop { deck: a }),
            ]);
            s + 4.0
        }
    };
    steps.sort_by(|x, y| x.at.total_cmp(&y.at));
    Plan {
        from: pair.from.id.clone(),
        to: pair.to.id.clone(),
        from_deck: a,
        to_deck: b,
        style,
        start: s,
        end,
        steps,
        why,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::music::track::fixture::{club, track};

    fn pair<'a>(from: &'a Profile, to: &'a Profile, now: f64) -> Pair<'a> {
        Pair {
            from,
            from_deck: 1,
            to,
            to_deck: 2,
            now,
        }
    }

    fn moves(p: &Plan) -> Vec<(f64, Move)> {
        p.steps.iter().map(|s| (s.at, s.act)).collect()
    }

    #[test]
    fn tempo_gap_allows_half_and_double_time() {
        assert!((tempo_gap(128.0, 130.0) - 0.015625).abs() < 1e-6);
        assert!(tempo_gap(128.0, 64.0) < 1e-6);
        assert!(tempo_gap(87.0, 174.0) < 1e-6);
        assert!(tempo_gap(128.0, 100.0) > 0.2);
    }

    /// **A blend of two club tracks**: the outro over the intro, 32 beats,
    /// the bass swapped half way, every move on a bar — the exact plan.
    #[test]
    fn two_club_tracks_blend_outro_over_intro() {
        let a = club("a", 128.0, "8A", 0.6);
        let b = club("b", 126.0, "9A", 0.65);
        let p = plan(&pair(&a, &b, 100.0));
        assert_eq!(p.style, Style::Blend);
        assert_eq!(p.why, "outro over intro (adjacent keys)");
        assert_eq!((p.start, p.end), (224.0, 256.0));
        let low = |d| Knob::Eq(d, Band::Low);
        assert_eq!(
            moves(&p),
            vec![
                (208.0, Move::Match { deck: 2 }),
                (
                    208.0,
                    Move::Set {
                        knob: Knob::Fader(2),
                        to: 0.0
                    }
                ),
                (
                    208.0,
                    Move::Set {
                        knob: Knob::Filter(2),
                        to: 0.5
                    }
                ),
                (
                    208.0,
                    Move::Set {
                        knob: low(2),
                        to: 0.0
                    }
                ),
                (224.0, Move::Start { deck: 2 }),
                (
                    224.0,
                    Move::Ramp {
                        knob: Knob::Fader(2),
                        to: 1.0,
                        beats: 16.0
                    }
                ),
                (
                    240.0,
                    Move::Set {
                        knob: low(1),
                        to: 0.0
                    }
                ),
                (
                    240.0,
                    Move::Set {
                        knob: low(2),
                        to: 0.5
                    }
                ),
                (
                    240.0,
                    Move::Ramp {
                        knob: Knob::Fader(1),
                        to: 0.0,
                        beats: 16.0
                    }
                ),
                (256.0, Move::Stop { deck: 1 }),
                (
                    256.0,
                    Move::Set {
                        knob: low(1),
                        to: 0.5
                    }
                ),
            ]
        );
        assert!(
            p.steps.iter().all(|s| s.at % 4.0 == 0.0),
            "every move on a bar"
        );
    }

    /// **The start leaves time to arm**: with the outro too close, the last
    /// boundary that leaves a blend's room is taken — or, past it, the first.
    #[test]
    fn the_start_is_never_inside_the_lead() {
        let a = club("a", 128.0, "8A", 0.6);
        let b = club("b", 128.0, "8A", 0.6);
        let late = plan(&pair(&a, &b, 200.0));
        assert!(late.start >= 200.0 + LEAD, "{}", late.start);
        assert_eq!(late.start, 256.0, "past the outro: the track's end");
        let early = plan(&pair(&a, &b, 0.0));
        assert_eq!(early.start, 224.0);
    }

    #[test]
    fn a_clash_echoes_out_a_lift_cuts_and_a_tempo_gap_sweeps() {
        let a = club("a", 128.0, "8A", 0.5);
        let clash = club("c", 128.0, "3B", 0.5);
        let p = plan(&pair(&a, &clash, 0.0));
        assert_eq!(p.style, Style::EchoOut);
        assert_eq!(
            p.why,
            "8A → 3B clash: echo out rather than lay them together"
        );
        assert!(moves(&p).contains(&(
            220.0,
            Move::Press {
                button: Button::Fx(1)
            }
        )));
        assert_eq!(p.end, 232.0);

        let lift = club("l", 128.0, "8A", 0.8);
        let p = plan(&pair(&a, &lift, 0.0));
        assert_eq!(p.style, Style::Cut);
        assert_eq!(p.why, "a lift of 0.30: cut in on the phrase");
        assert!(moves(&p).contains(&(
            224.0,
            Move::Set {
                knob: Knob::Fader(1),
                to: 0.0
            }
        )));

        let slow = club("s", 100.0, "8A", 0.5);
        let p = plan(&pair(&a, &slow, 0.0));
        assert_eq!(p.style, Style::FilterSweep);
        assert_eq!(
            p.why,
            "128 → 100 BPM is too far to blend: filter out, then in"
        );
        assert!(moves(&p).contains(&(
            208.0,
            Move::Ramp {
                knob: Knob::Filter(1),
                to: 0.1,
                beats: 16.0
            }
        )));
    }

    /// **A short intro shortens the blend** to it, on whole bars.
    #[test]
    fn the_blend_is_as_long_as_the_shorter_of_outro_and_intro() {
        use PhraseKind::*;
        let a = club("a", 128.0, "8A", 0.6);
        let b = track(
            "b",
            128.0,
            "8A",
            0.6,
            200,
            &[(Intro, 0, 16), (Chorus, 16, 184), (Outro, 184, 200)],
        );
        let p = plan(&pair(&a, &b, 0.0));
        assert_eq!((p.style, p.start, p.end), (Style::Blend, 224.0, 240.0));
    }
}
