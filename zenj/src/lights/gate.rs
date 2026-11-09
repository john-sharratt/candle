//! The rate budget (§9.2): a token bucket per bulb, which coalesces — what a
//! bulb is wanted to show is kept, and sent when the bulb has a token, so a
//! bulb that missed four states arrives at the fifth. Only a change is sent.
//! A key moment — a drop, midnight — is sent at once whatever the bucket
//! says, and again a moment later: UDP has no ack, and one lost datagram is
//! one dark bulb at the moment the room was waiting for.

use std::time::{Duration, Instant};

use effector_bridge::proto::wiz::Pilot;

/// A bulb's sustained rate, in messages a second.
pub const RATE: f64 = 10.0;
/// How many it may send at once.
pub const BURST: f64 = 20.0;
/// How long after a key moment it is sent again.
pub const AGAIN: Duration = Duration::from_millis(15);

#[derive(Debug, Clone)]
struct Slot {
    tokens: f64,
    filled: Instant,
    sent: Option<Pilot>,
    want: Option<Pilot>,
    again: Option<(Instant, Pilot)>,
}

#[derive(Debug, Clone)]
pub struct Gate {
    rate: f64,
    burst: f64,
    slots: Vec<Slot>,
}

impl Gate {
    pub fn new(bulbs: usize, rate: f64, burst: f64, now: Instant) -> Gate {
        Gate {
            rate,
            burst,
            slots: vec![
                Slot {
                    tokens: burst,
                    filled: now,
                    sent: None,
                    want: None,
                    again: None,
                };
                bulbs
            ],
        }
    }

    /// Bulb `i` is wanted to show `p`.
    pub fn want(&mut self, i: usize, p: Pilot) {
        if let Some(s) = self.slots.get_mut(i) {
            s.want = Some(p);
        }
    }

    /// Bulb `i` shows `p` now, whatever its bucket says, and again
    /// [`AGAIN`] later.
    pub fn key(&mut self, i: usize, p: Pilot, now: Instant) -> Option<Pilot> {
        let s = self.slots.get_mut(i)?;
        s.want = Some(p);
        s.sent = Some(p);
        s.tokens -= 1.0;
        s.again = Some((now + AGAIN, p));
        Some(p)
    }

    /// What bulb `i` was last sent is no longer known: it is sent again.
    pub fn forget(&mut self, i: usize) {
        if let Some(s) = self.slots.get_mut(i) {
            s.sent = None;
        }
    }

    /// The sends due at `now`: each bulb's repeat that has come, and each
    /// bulb whose wanted state is not what it was sent and that has a token.
    pub fn due(&mut self, now: Instant) -> Vec<(usize, Pilot)> {
        let (rate, burst) = (self.rate, self.burst);
        let mut out = Vec::new();
        for (i, s) in self.slots.iter_mut().enumerate() {
            let gained = now.saturating_duration_since(s.filled).as_secs_f64() * rate;
            s.tokens = (s.tokens + gained).min(burst);
            s.filled = now;
            if let Some((at, p)) = s.again {
                if now >= at {
                    s.again = None;
                    out.push((i, p));
                    continue;
                }
            }
            let Some(want) = s.want else { continue };
            if s.sent != Some(want) && s.tokens >= 1.0 {
                s.tokens -= 1.0;
                s.sent = Some(want);
                out.push((i, want));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn red(d: u8) -> Pilot {
        Pilot::Colour {
            r: 255,
            g: 0,
            b: 0,
            dimming: d,
        }
    }

    /// **Only a change is sent**, once, however often it is wanted.
    #[test]
    fn only_a_change_is_sent() {
        let t = Instant::now();
        let mut g = Gate::new(2, RATE, BURST, t);
        g.want(0, red(50));
        assert_eq!(g.due(t), vec![(0, red(50))]);
        g.want(0, red(50));
        assert!(g.due(t).is_empty(), "the same state again sends nothing");
        g.forget(0);
        assert_eq!(g.due(t), vec![(0, red(50))], "forgotten, it is sent again");
    }

    /// **A bulb past its budget keeps the latest state** and sends it when a
    /// token comes, skipping the ones it missed.
    #[test]
    fn a_bulb_past_its_budget_coalesces_to_the_latest() {
        let t = Instant::now();
        let mut g = Gate::new(1, 10.0, 1.0, t);
        g.want(0, red(10));
        assert_eq!(g.due(t).len(), 1, "the one token");
        for d in [20, 30, 40, 50] {
            g.want(0, red(d));
            assert!(g.due(t).is_empty(), "no token yet");
        }
        assert_eq!(
            g.due(t + Duration::from_millis(100)),
            vec![(0, red(50))],
            "a tenth of a second later, the latest"
        );
    }

    /// **A key moment goes at once, whatever the bucket, and again after
    /// [`AGAIN`]**.
    #[test]
    fn a_key_moment_is_sent_twice() {
        let t = Instant::now();
        let mut g = Gate::new(1, 10.0, 0.0, t);
        assert_eq!(g.key(0, red(100), t), Some(red(100)));
        assert!(g.due(t + Duration::from_millis(5)).is_empty());
        assert_eq!(g.due(t + AGAIN), vec![(0, red(100))]);
        assert!(g.due(t + AGAIN * 2).is_empty(), "twice, not thrice");
        assert_eq!(g.key(7, red(1), t), None, "no such bulb");
    }
}
