//! How often a repository's origin is asked (`docs/zend_branch_ingest.md`
//! §4.1, §4.3): every interval, spread by ±20 % so repositories do not
//! synchronise, and after a failure backing off — doubling from 4 s to at
//! most 5 minutes — until the next success.

use std::time::Duration;

/// The HTTPS probe's interval, and the git probe's over a local path.
pub const FAST: Duration = Duration::from_secs(2);

/// The git probe's interval over SSH, where each probe is a handshake.
pub const SLOW: Duration = Duration::from_secs(30);

/// The first wait after a failure.
const BACKOFF_FIRST: Duration = Duration::from_secs(4);

/// The longest wait after failures.
const BACKOFF_MAX: Duration = Duration::from_secs(300);

/// The spread applied to every wait, as a fraction of it.
const JITTER: f64 = 0.2;

/// When to probe next.
#[derive(Debug, Clone, PartialEq)]
pub struct Cadence {
    interval: Duration,
    failures: u32,
}

impl Cadence {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            failures: 0,
        }
    }

    pub fn succeeded(&mut self) {
        self.failures = 0;
    }

    pub fn failed(&mut self) {
        self.failures = self.failures.saturating_add(1);
    }

    /// Whether the last probe failed.
    pub fn failing(&self) -> bool {
        self.failures > 0
    }

    /// How many probes in a row have failed.
    pub fn failures(&self) -> u32 {
        self.failures
    }

    /// The wait before the next probe, given `spread` in `[-1, 1]` — the
    /// jitter, as a fraction of its bound.
    pub fn next_delay(&self, spread: f64) -> Duration {
        let base = match self.failures {
            0 => self.interval,
            n => {
                let doubled = BACKOFF_FIRST.saturating_mul(1u32 << (n - 1).min(16));
                doubled.min(BACKOFF_MAX)
            }
        };
        let ms = base.as_millis() as f64 * (1.0 + JITTER * spread.clamp(-1.0, 1.0));
        Duration::from_millis(ms.round() as u64)
    }
}

/// A small, seeded source of spreads in `[-1, 1]` — each repository's own
/// sequence, so no two watchers draw the same one.
#[derive(Debug, Clone)]
pub struct Jitter(u64);

impl Jitter {
    /// A sequence seeded from `seed` (a repository's name).
    pub fn new(seed: &str) -> Self {
        // FNV-1a: stable across runs and platforms, and never zero after the
        // `| 1` below, which xorshift needs.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in seed.bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        Self(h | 1)
    }

    /// The next spread.
    pub fn draw(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_healthy_probe_waits_its_interval_spread_by_a_fifth() {
        let c = Cadence::new(FAST);
        assert_eq!(c.next_delay(0.0), Duration::from_secs(2));
        assert_eq!(c.next_delay(1.0), Duration::from_millis(2400));
        assert_eq!(c.next_delay(-1.0), Duration::from_millis(1600));
        assert_eq!(c.next_delay(7.0), Duration::from_millis(2400), "clamped");
    }

    #[test]
    fn failures_back_off_doubling_to_five_minutes_and_a_success_resets() {
        let mut c = Cadence::new(FAST);
        let waits: Vec<u64> = (0..9)
            .map(|_| {
                c.failed();
                c.next_delay(0.0).as_secs()
            })
            .collect();
        assert_eq!(waits, [4, 8, 16, 32, 64, 128, 256, 300, 300]);
        assert!(c.failing());
        assert_eq!(c.failures(), 9);
        c.succeeded();
        assert!(!c.failing());
        assert_eq!(c.failures(), 0);
        assert_eq!(c.next_delay(0.0), FAST);
    }

    /// Every spread is in range, the sequence is the same for the same seed,
    /// and two repositories draw different ones.
    #[test]
    fn jitter_is_in_range_and_seeded_per_repository() {
        let mut a = Jitter::new("candle");
        let draws: Vec<f64> = (0..1000).map(|_| a.draw()).collect();
        assert!(draws.iter().all(|d| (-1.0..=1.0).contains(d)));
        assert!(draws.iter().any(|d| *d > 0.5) && draws.iter().any(|d| *d < -0.5));
        let mut again = Jitter::new("candle");
        assert_eq!(again.draw(), draws[0]);
        assert_ne!(Jitter::new("mind").draw(), draws[0]);
    }
}
