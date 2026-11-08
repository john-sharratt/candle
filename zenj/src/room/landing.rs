//! How the room is taking what plays (§4.4, §5.2): the last half-minute's
//! level and low end against the last few minutes'. A room singing along and
//! a mix pushed louder both lift it; a floor emptying lets it fall. It says
//! how much likeness to the current track should count in the shortlist, and
//! nothing more — the energy curve stays the plan.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::levels::Levels;

/// The recent window, and the baseline it is held against.
pub const RECENT: Duration = Duration::from_secs(30);
pub const BASELINE: Duration = Duration::from_secs(300);
/// The level swing, in dB, that moves landing from the middle to an end.
const SWING_DB: f32 = 6.0;

#[derive(Debug, Default)]
pub struct Room {
    seen: VecDeque<(Instant, Levels)>,
}

fn mean<'a>(it: impl Iterator<Item = &'a Levels>) -> Option<(f32, f32)> {
    let (mut n, mut db, mut low) = (0f32, 0f32, 0f32);
    for l in it {
        n += 1.0;
        db += l.rms_db;
        low += l.bands[0];
    }
    (n > 0.0).then(|| (db / n, low / n))
}

impl Room {
    pub fn new() -> Room {
        Room::default()
    }

    pub fn push(&mut self, at: Instant, l: Levels) {
        self.seen.push_back((at, l));
        while self
            .seen
            .front()
            .is_some_and(|(t, _)| at.duration_since(*t) > BASELINE)
        {
            self.seen.pop_front();
        }
    }

    /// The last half-minute's mean level, dBFS.
    pub fn level(&self, now: Instant) -> Option<f32> {
        mean(self.recent(now)).map(|(db, _)| db)
    }

    fn recent(&self, now: Instant) -> impl Iterator<Item = &Levels> {
        self.seen
            .iter()
            .filter(move |(t, _)| now.duration_since(*t) <= RECENT)
            .map(|(_, l)| l)
    }

    /// How well it is landing, `0..=1`; 0.5 with too little heard to say.
    pub fn landing(&self, now: Instant) -> f32 {
        let (Some((rdb, rlow)), Some((bdb, blow))) = (
            mean(self.recent(now)),
            mean(self.seen.iter().map(|(_, l)| l)),
        ) else {
            return 0.5;
        };
        let lift = (rdb - bdb) / SWING_DB + (rlow - blow) * 2.0;
        (0.5 + 0.5 * lift).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(db: f32, low: f32) -> Levels {
        Levels {
            rms_db: db,
            bands: [low, 1.0 - low, 0.0],
        }
    }

    #[test]
    fn a_louder_heavier_half_minute_lands() {
        let t0 = Instant::now();
        let mut r = Room::new();
        assert_eq!(r.landing(t0), 0.5);
        for s in 0..240 {
            r.push(t0 + Duration::from_secs(s), at(-20.0, 0.3));
        }
        let now = t0 + Duration::from_secs(240);
        assert!(
            (r.landing(now) - 0.5).abs() < 1e-5,
            "steady: {}",
            r.landing(now)
        );
        for s in 241..=270 {
            r.push(t0 + Duration::from_secs(s), at(-14.0, 0.4));
        }
        let now = t0 + Duration::from_secs(270);
        assert!(r.landing(now) > 0.8, "{}", r.landing(now));
        assert!((r.level(now).unwrap() + 14.0).abs() < 0.5);
        for s in 271..=600 {
            r.push(t0 + Duration::from_secs(s), at(-30.0, 0.2));
        }
        assert_eq!(r.seen.len(), 301, "five minutes kept");
    }
}
