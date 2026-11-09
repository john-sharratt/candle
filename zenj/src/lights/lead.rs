//! How early each bulb is sent its state (§9.3): its own lag, measured. A
//! bulb answers a `getPilot` the moment it gets one, so half a poll's round
//! trip is the network's share of its lag; the rest — the bulb taking a
//! state and lighting it — is the configured `lead_ms`, which spike 8
//! measures in the room. The network's share is a running average, so one
//! slow answer moves it a little and a slow network moves it all the way.

use std::time::Duration;

/// How much of a new round trip the average takes.
const ALPHA: f64 = 0.25;
/// The most a network's share is believed: past this a bulb is struggling,
/// and leading it further would only move its light off the beat.
const CAP: Duration = Duration::from_millis(250);

#[derive(Debug, Clone)]
pub struct Leads {
    base: Duration,
    one_way: Vec<Option<f64>>,
}

impl Leads {
    /// `bulbs` bulbs, each led by `base` until it is measured.
    pub fn new(bulbs: usize, base: Duration) -> Leads {
        Leads {
            base,
            one_way: vec![None; bulbs],
        }
    }

    /// Bulb `i` answered a poll after `rtt`.
    pub fn measured(&mut self, i: usize, rtt: Duration) {
        let Some(slot) = self.one_way.get_mut(i) else {
            return;
        };
        let half = (rtt / 2).min(CAP).as_secs_f64();
        *slot = Some(match *slot {
            Some(avg) => avg + ALPHA * (half - avg),
            None => half,
        });
    }

    /// How early bulb `i` is sent what it is to show.
    pub fn of(&self, i: usize) -> Duration {
        let net = self
            .one_way
            .get(i)
            .copied()
            .flatten()
            .map_or(Duration::ZERO, Duration::from_secs_f64);
        self.base + net
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// **A bulb is led by its own lag**: the base until it answers, then the
    /// base and half its round trip, averaged, and never past the cap.
    #[test]
    fn a_bulb_is_led_by_its_own_lag() {
        let mut l = Leads::new(2, ms(25));
        assert_eq!(l.of(0), ms(25));
        l.measured(0, ms(40));
        assert_eq!(l.of(0), ms(45), "half of 40 on top");
        l.measured(0, ms(120));
        assert_eq!(l.of(0), ms(55), "a quarter of the way to 60");
        assert_eq!(l.of(1), ms(25), "the other unmeasured");
        l.measured(1, Duration::from_secs(5));
        assert_eq!(l.of(1), ms(275), "capped");
        assert_eq!(l.of(9), ms(25), "no such bulb");
    }
}
