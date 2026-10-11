//! The reconnect delay (§3.2): `d` starts at 250 ms and doubles per consecutive
//! failure, capped at 8 s; each wait is drawn uniformly from `[d/2, d]`, so a
//! gateway restart does not reconnect every member in the same millisecond. An
//! admission resets it.

use std::time::Duration;

use rand::Rng;

use crate::close::AfterClose;

pub const FIRST: Duration = Duration::from_millis(250);
pub const CAP: Duration = Duration::from_secs(8);

#[derive(Debug, Clone)]
pub struct Backoff {
    d: Duration,
}

impl Default for Backoff {
    fn default() -> Self {
        Self { d: FIRST }
    }
}

impl Backoff {
    pub fn new() -> Self {
        Self::default()
    }

    /// The `d` the next failure's wait is drawn under.
    pub fn current(&self) -> Duration {
        self.d
    }

    /// The wait after one more failure, drawn from `[d/2, d]`; `d` then
    /// doubles toward the cap.
    pub fn next(&mut self, rng: &mut impl Rng) -> Duration {
        let wait = draw(self.d, rng);
        self.d = (self.d * 2).min(CAP);
        wait
    }

    /// The wait after a close, or `None` when the socket stops dialling.
    pub fn after(&mut self, after: AfterClose, rng: &mut impl Rng) -> Option<Duration> {
        match after {
            AfterClose::Backoff => Some(self.next(rng)),
            AfterClose::AtCap => {
                self.d = CAP;
                Some(draw(CAP, rng))
            }
            AfterClose::StopSocket | AfterClose::StopApp => None,
        }
    }

    /// An admission: the next failure starts from 250 ms again.
    pub fn reset(&mut self) {
        self.d = FIRST;
    }
}

fn draw(d: Duration, rng: &mut impl Rng) -> Duration {
    let hi = d.as_millis() as u64;
    Duration::from_millis(rng.random_range(hi / 2..=hi))
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::*;

    /// `d` runs 250, 500, 1000, 2000, 4000, 8000, 8000 ms, each wait drawn
    /// within `[d/2, d]`, and an admission starts it over.
    #[test]
    fn delays_double_to_the_cap_within_their_bounds_and_reset() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut b = Backoff::new();
        let schedule = [250u64, 500, 1000, 2000, 4000, 8000, 8000];
        for d in schedule {
            assert_eq!(b.current(), Duration::from_millis(d));
            let wait = b.next(&mut rng).as_millis() as u64;
            assert!(
                (d / 2..=d).contains(&wait),
                "{wait} outside [{}, {d}]",
                d / 2
            );
        }
        b.reset();
        assert_eq!(b.current(), FIRST);
    }

    /// The draws spread across the window rather than sitting at one end.
    #[test]
    fn draws_spread_across_the_window() {
        let mut rng = StdRng::seed_from_u64(1);
        let waits: Vec<u64> = (0..200)
            .map(|_| draw(CAP, &mut rng).as_millis() as u64)
            .collect();
        assert!(waits.iter().any(|w| *w < 5000));
        assert!(waits.iter().any(|w| *w > 7000));
    }

    #[test]
    fn a_close_picks_the_wait_or_stops() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut b = Backoff::new();
        let at_cap = b.after(AfterClose::AtCap, &mut rng).unwrap();
        assert!(at_cap >= CAP / 2 && at_cap <= CAP);
        assert_eq!(b.current(), CAP);
        assert_eq!(b.after(AfterClose::StopApp, &mut rng), None);
        assert_eq!(b.after(AfterClose::StopSocket, &mut rng), None);
        b.reset();
        let first = b.after(AfterClose::Backoff, &mut rng).unwrap();
        assert!(first <= FIRST);
    }
}
