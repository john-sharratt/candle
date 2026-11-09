//! The lights planned ahead (§9.3): what every bulb is to show over the coming
//! [`HORIZON_BEATS`], sampled every [`STEP`] beats and at each moment that
//! falls between — a cue's beat, midnight — and turned into sends, each at the
//! instant its beat falls less that bulb's own lead. The sender plays the
//! timeline; the planner tops it up as it runs out and makes it again whenever
//! what it was made from changes. A stall upstream — the reader blind a
//! moment, a slow tick — then costs the room nothing for two bars.

use std::time::{Duration, Instant};

use effector_bridge::proto::wiz::Pilot;

use super::director::{Director, Input};
use crate::beat::BeatClock;

/// How far ahead the lights are planned: two bars.
pub const HORIZON_BEATS: f64 = 8.0;
/// How finely: half a beat, the finest a scene moves.
pub const STEP: f64 = 0.5;

/// One send, planned.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Planned {
    /// When it goes.
    pub at: Instant,
    pub bulb: usize,
    pub pilot: Pilot,
    /// A hit: sent at once whatever the budget, and again.
    pub key: bool,
}

/// What the planner works from: the night at `now`.
pub struct Basis<'a> {
    pub director: &'a Director,
    /// The night at `now`; its beat and seconds to midnight are moved to each
    /// sample's.
    pub now_input: Input,
    pub clock: &'a BeatClock,
    pub now: Instant,
    pub lead: &'a dyn Fn(usize) -> Duration,
    pub groups: &'a dyn Fn(Option<&str>) -> Vec<usize>,
}

/// The beats sampled from `from` through `to`: every [`STEP`], and each of
/// `moments` between.
pub fn samples(from: f64, to: f64, moments: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::new();
    let mut b = (from / STEP).ceil() * STEP;
    while b <= to {
        out.push(b);
        b += STEP;
    }
    out.extend(moments.iter().copied().filter(|m| (from..=to).contains(m)));
    out.sort_by(f64::total_cmp);
    out.dedup();
    out
}

/// The sends for the beats `from` through `to`, in the order they go; a bulb
/// is sent a state only when it differs from the one it was planned before.
pub fn plan(f: &Basis<'_>, from: f64, to: f64) -> Vec<Planned> {
    let mut moments = f.director.cue_beats();
    let midnight = f.now + Duration::from_secs_f64(f.now_input.to_midnight.max(0.0));
    if let Some(b) = f.clock.beat_at(midnight) {
        moments.push(b);
    }
    let mut last: Vec<Option<Pilot>> = vec![None; f.now_input.bulbs];
    let mut out = Vec::new();
    for b in samples(from, to, &moments) {
        let Some(t) = f.clock.time_of(b) else { break };
        let ahead = t.saturating_duration_since(f.now).as_secs_f64();
        let inp = Input {
            beat: Some(b),
            to_midnight: f.now_input.to_midnight - ahead,
            ..f.now_input
        };
        let Some(look) = f.director.look(&inp, f.groups) else {
            continue;
        };
        for (i, pilot) in look.bulbs {
            // A hit's window may hold more than one sample: it is sent once.
            if last.get(i).copied().flatten() == Some(pilot) {
                continue;
            }
            if let Some(slot) = last.get_mut(i) {
                *slot = Some(pilot);
            }
            out.push(Planned {
                at: t.checked_sub((f.lead)(i)).unwrap_or(t).max(f.now),
                bulb: i,
                pilot,
                key: look.key,
            });
        }
    }
    out.sort_by_key(|s| s.at);
    out
}

/// What to show now with no beat to plan on: the look at this instant alone.
pub fn now_only(f: &Basis<'_>) -> Vec<Planned> {
    let inp = Input {
        beat: None,
        ..f.now_input
    };
    let Some(look) = f.director.look(&inp, f.groups) else {
        return Vec::new();
    };
    look.bulbs
        .into_iter()
        .map(|(bulb, pilot)| Planned {
            at: f.now,
            bulb,
            pilot,
            key: look.key,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hands::Holder;
    use crate::lights::cue::Cue;
    use crate::lights::director::Ask;
    use crate::lights::palette::HIT;

    fn input() -> Input {
        Input {
            rekordbox: true,
            holder: Holder::Zenj,
            energy: 0.9,
            beat: None,
            to_midnight: 3600.0,
            bulbs: 2,
            dead: 0,
        }
    }

    fn groups(_: Option<&str>) -> Vec<usize> {
        vec![0, 1]
    }

    #[test]
    fn samples_are_every_half_beat_and_each_moment_between() {
        assert_eq!(samples(0.2, 2.0, &[]), [0.5, 1.0, 1.5, 2.0]);
        assert_eq!(
            samples(0.0, 1.0, &[0.75, 9.0]),
            [0.0, 0.5, 0.75, 1.0],
            "a moment past the window is not sampled"
        );
    }

    /// **Two bars are planned, each send its beat's instant less the bulb's
    /// lead, and only changes**: a pulse is two states a beat.
    #[test]
    fn two_bars_are_planned_ahead_by_each_bulbs_lead() {
        let now = Instant::now();
        let mut clock = BeatClock::new();
        clock.read(now, Some((64.0, 120.0)));
        let d = Director::new();
        let lead = |i: usize| Duration::from_millis(if i == 0 { 20 } else { 60 });
        let f = Basis {
            director: &d,
            now_input: input(),
            clock: &clock,
            now,
            lead: &lead,
            groups: &groups,
        };
        let sends = plan(&f, 64.0, 64.0 + HORIZON_BEATS);
        // A pulse: bright on the beat, dim off it — 17 samples, every one a
        // change, for each of two bulbs.
        assert_eq!(sends.len(), 34);
        let first_of = |i| sends.iter().find(|s| s.bulb == i).unwrap().at;
        assert_eq!(first_of(0), now, "the first beat is now, no earlier");
        let beat_65 = clock.time_of(65.0).unwrap();
        let at_65 = |i| sends.iter().filter(|s| s.bulb == i).nth(2).unwrap().at;
        assert_eq!(at_65(0), beat_65 - Duration::from_millis(20));
        assert_eq!(at_65(1), beat_65 - Duration::from_millis(60));
        assert!(sends.windows(2).all(|w| w[0].at <= w[1].at), "in order");
    }

    /// **A cue between the steps is sampled at its own beat**, keyed.
    #[test]
    fn a_cue_between_the_steps_is_sampled_on_its_beat() {
        let now = Instant::now();
        let mut clock = BeatClock::new();
        clock.read(now, Some((64.0, 120.0)));
        let mut d = Director::new();
        d.cues(vec![(66.25, Cue::Hit)]);
        let lead = |_| Duration::ZERO;
        let f = Basis {
            director: &d,
            now_input: input(),
            clock: &clock,
            now,
            lead: &lead,
            groups: &groups,
        };
        let hits: Vec<Planned> = plan(&f, 64.0, 72.0).into_iter().filter(|s| s.key).collect();
        assert_eq!(hits.len(), 2, "both bulbs");
        assert!(hits.iter().all(|s| s.pilot == HIT));
        assert_eq!(hits[0].at, clock.time_of(66.25).unwrap());
    }

    /// **With no beat, what shows now is planned alone** — the house lights,
    /// say — and with rekordbox closed, nothing.
    #[test]
    fn with_no_beat_the_instant_alone() {
        let now = Instant::now();
        let clock = BeatClock::new();
        let mut d = Director::new();
        d.ask(Ask::House(true));
        let lead = |_| Duration::ZERO;
        let f = Basis {
            director: &d,
            now_input: input(),
            clock: &clock,
            now,
            lead: &lead,
            groups: &groups,
        };
        assert!(plan(&f, 0.0, 8.0).is_empty(), "no beat, nothing ahead");
        assert_eq!(now_only(&f).len(), 2);
        let closed = Basis {
            now_input: Input {
                rekordbox: false,
                ..input()
            },
            ..f
        };
        assert!(now_only(&closed).is_empty());
    }
}
