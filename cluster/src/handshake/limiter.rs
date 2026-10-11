//! Failed proofs counted per source IP (§3.3): more than ten in a minute holds
//! that IP's handshakes off for a minute.

use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;
use std::time::{Duration, Instant};

pub const WINDOW: Duration = Duration::from_secs(60);
pub const LIMIT: usize = 10;
pub const HOLD: Duration = Duration::from_secs(60);

#[derive(Debug, Default)]
pub struct ProofLimiter {
    failures: HashMap<IpAddr, VecDeque<Instant>>,
    held_until: HashMap<IpAddr, Instant>,
}

impl ProofLimiter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a handshake from `ip` may start at `now`.
    pub fn allows(&mut self, ip: IpAddr, now: Instant) -> bool {
        match self.held_until.get(&ip) {
            Some(until) if now < *until => false,
            Some(_) => {
                self.held_until.remove(&ip);
                true
            }
            None => true,
        }
    }

    /// A proof from `ip` failed at `now`.
    pub fn failed(&mut self, ip: IpAddr, now: Instant) {
        let window = self.failures.entry(ip).or_default();
        window.push_back(now);
        while window
            .front()
            .is_some_and(|t| now.duration_since(*t) >= WINDOW)
        {
            window.pop_front();
        }
        if window.len() > LIMIT {
            window.clear();
            self.held_until.insert(ip, now + HOLD);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eleven_failures_in_a_minute_hold_that_ip_off_for_a_minute() {
        let mut l = ProofLimiter::new();
        let ip: IpAddr = "10.0.0.9".parse().unwrap();
        let other: IpAddr = "10.0.0.10".parse().unwrap();
        let t0 = Instant::now();
        for i in 0..10 {
            l.failed(ip, t0 + Duration::from_secs(i));
            assert!(l.allows(ip, t0 + Duration::from_secs(i)));
        }
        l.failed(ip, t0 + Duration::from_secs(10));
        assert!(!l.allows(ip, t0 + Duration::from_secs(11)));
        assert!(l.allows(other, t0 + Duration::from_secs(11)));
        assert!(!l.allows(ip, t0 + Duration::from_secs(69)));
        assert!(l.allows(ip, t0 + Duration::from_secs(70)));
    }

    #[test]
    fn failures_spread_past_the_window_never_hold() {
        let mut l = ProofLimiter::new();
        let ip: IpAddr = "10.0.0.9".parse().unwrap();
        let t0 = Instant::now();
        for i in 0..30 {
            l.failed(ip, t0 + Duration::from_secs(i * 7));
            assert!(l.allows(ip, t0 + Duration::from_secs(i * 7)));
        }
    }
}
