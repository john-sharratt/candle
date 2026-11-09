//! The beat as a clock (§4.2, §8, §9.3): the deck reader's last trusted
//! reading of the playing deck, extrapolated by its tempo, so the screens and
//! the lights can ask what the beat is — or will be — at any instant, and
//! when a coming beat falls. A reading that agrees with the prediction only
//! refines it; one that does not — a seek, a new track, a tempo nudged — is a
//! **jump**, and whatever was planned from the old prediction is invalid.
//!
//! With no reading — the reader blind a moment, rekordbox busy — the clock
//! **coasts** on its last anchor for [`COAST_BEATS`], so a hiccup upstream is
//! not a hiccup in the room.

use std::time::{Duration, Instant};

use crate::deck::Snapshot;

/// The playing deck's beat and tempo in `snap`, as the clock reads them.
pub fn reading(snap: &Snapshot) -> Option<(f64, f64)> {
    let d = snap.on_air()?;
    Some((d.beat?, d.bpm as f64))
}

/// How far a reading may stray from the prediction, in beats, before it is a
/// jump: a fifth of a beat, under 100 ms at any club tempo.
pub const JUMP_BEATS: f64 = 0.2;
/// A change of tempo, in BPM, that is a jump.
pub const JUMP_BPM: f64 = 0.5;
/// How many beats the clock runs on without a reading before it says it does
/// not know.
pub const COAST_BEATS: f64 = 8.0;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Anchor {
    at: Instant,
    beat: f64,
    bpm: f64,
}

/// What a reading did to the clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Read {
    /// It agreed, or there was nothing to read: what was planned stands.
    Steady,
    /// It disagreed, or it began or ended the music: plan again.
    Jump,
}

#[derive(Debug, Clone, Default)]
pub struct BeatClock {
    anchor: Option<Anchor>,
}

impl BeatClock {
    pub fn new() -> BeatClock {
        BeatClock::default()
    }

    /// The reader's `(beat, bpm)` of the playing deck at `now`, or `None` when
    /// it reads nothing playing.
    pub fn read(&mut self, now: Instant, reading: Option<(f64, f64)>) -> Read {
        let Some((beat, bpm)) = reading.filter(|(_, bpm)| *bpm > 0.0) else {
            // Nothing playing: coast, until the coast runs out.
            if self.anchor.is_some() && self.beat_at(now).is_none() {
                self.anchor = None;
                return Read::Jump;
            }
            return Read::Steady;
        };
        let jumped = match (self.anchor, self.beat_at(now)) {
            (Some(a), Some(predicted)) => {
                (predicted - beat).abs() > JUMP_BEATS || (a.bpm - bpm).abs() > JUMP_BPM
            }
            _ => true,
        };
        self.anchor = Some(Anchor { at: now, beat, bpm });
        match jumped {
            true => Read::Jump,
            false => Read::Steady,
        }
    }

    /// The tempo it runs at.
    pub fn bpm(&self) -> Option<f64> {
        self.anchor.map(|a| a.bpm)
    }

    /// The beat at `t`, past or coming — `None` with no anchor, or past the
    /// coast.
    pub fn beat_at(&self, t: Instant) -> Option<f64> {
        let a = self.anchor?;
        let since = match t.checked_duration_since(a.at) {
            Some(d) => d.as_secs_f64(),
            None => -(a.at - t).as_secs_f64(),
        };
        let beat = a.beat + since * a.bpm / 60.0;
        (beat - a.beat <= COAST_BEATS).then_some(beat)
    }

    /// When `beat` falls — `None` with no anchor.
    pub fn time_of(&self, beat: f64) -> Option<Instant> {
        let a = self.anchor?;
        let secs = (beat - a.beat) * 60.0 / a.bpm;
        Some(match secs >= 0.0 {
            true => a.at + Duration::from_secs_f64(secs),
            false => a.at - Duration::from_secs_f64(-secs),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// **The clock runs on by its tempo**: at 120 BPM, half a second is a
    /// beat, forward and back, and a beat's time is found again.
    #[test]
    fn the_clock_runs_on_by_its_tempo() {
        let t = Instant::now();
        let mut c = BeatClock::new();
        assert_eq!(c.beat_at(t), None);
        assert_eq!(
            c.read(t, Some((64.0, 120.0))),
            Read::Jump,
            "the first reading"
        );
        assert_eq!(c.beat_at(t + ms(500)), Some(65.0));
        assert_eq!(c.beat_at(t + ms(2000)), Some(68.0));
        assert_eq!(c.time_of(66.0), Some(t + ms(1000)));
        assert_eq!(c.bpm(), Some(120.0));
    }

    /// **A reading that agrees is steady; one that strays is a jump.**
    #[test]
    fn a_stray_reading_is_a_jump() {
        let t = Instant::now();
        let mut c = BeatClock::new();
        c.read(t, Some((64.0, 120.0)));
        assert_eq!(c.read(t + ms(500), Some((65.05, 120.0))), Read::Steady);
        assert_eq!(
            c.read(t + ms(1000), Some((80.0, 120.0))),
            Read::Jump,
            "a seek"
        );
        assert_eq!(
            c.read(t + ms(1500), Some((81.0, 124.0))),
            Read::Jump,
            "the tempo nudged"
        );
    }

    /// **With no reading it coasts**, for [`COAST_BEATS`], then lets go.
    #[test]
    fn with_no_reading_it_coasts_then_lets_go() {
        let t = Instant::now();
        let mut c = BeatClock::new();
        c.read(t, Some((0.0, 120.0)));
        assert_eq!(c.read(t + ms(2000), None), Read::Steady, "four beats blind");
        assert_eq!(c.beat_at(t + ms(2000)), Some(4.0));
        assert_eq!(c.read(t + ms(5000), None), Read::Jump, "past the coast");
        assert_eq!(c.beat_at(t + ms(5000)), None);
        assert_eq!(c.read(t + ms(6000), None), Read::Steady, "and stays let go");
    }
}
