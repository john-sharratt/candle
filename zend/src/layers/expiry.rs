//! When a unit of a layer that expires is read again.
//!
//! A layer declaring `expires_days: N` answers a question whose answer goes
//! out of date even when the file does not change — what is known about the
//! threats to it — so each of its units is read again about every `N` days.
//!
//! **Each file has its own phase, so the layer never churns at once.** Read
//! together, every file of a layer would expire together, and the layer would
//! be read again whole on one day each period. Instead the period is cut at a
//! boundary of the file's own, offset by a digest of its path: a unit stands
//! until a boundary of its file after it was built. The first units of a
//! layer — built together — expire spread evenly over the period, and each
//! file is read again every period after that, so the layer reads `1/N` of
//! its files a day. The phase is a digest, not a random draw, so every member
//! of a cluster finds the same unit expired at the same time.
//!
//! **A unit stands at least [`Expiry::min_standing`].** A unit read just
//! before its file's boundary — rebuilt because the file was edited — skips
//! that boundary and stands to the next one, rather than being read again
//! minutes after it was built.
//!
//! **An expired unit stands until its replacement commits.** It is old, not
//! wrong: the planner keeps it and reads the file again (`super::plan`). Each
//! reading of an expiring unit is keyed by the boundary it will expire at
//! ([`Expiry::term`]), so the replacement is a unit of its own key, never one
//! the expired unit already holds.

use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

/// One day, in milliseconds.
pub const DAY_MS: u64 = 86_400_000;

/// How long a layer's units stand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Expiry {
    period_ms: u64,
}

impl Expiry {
    /// Units read again every `days` days. `None` for zero days, which would
    /// expire a unit the moment it was built.
    pub fn days(days: u32) -> Option<Self> {
        (days > 0).then(|| Self {
            period_ms: u64::from(days) * DAY_MS,
        })
    }

    /// The period, in milliseconds.
    pub fn period_ms(self) -> u64 {
        self.period_ms
    }

    /// The least a unit stands: a thirtieth of the period — a day of thirty.
    pub fn min_standing(self) -> u64 {
        self.period_ms / 30
    }

    /// The offset of `subject`'s boundaries into the period: a digest of its
    /// path, so it is the same on every member and for every version of the
    /// file.
    fn phase(self, subject: &str) -> u64 {
        let digest = Sha256::digest(subject.as_bytes());
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&digest[..8]);
        u64::from_le_bytes(bytes) % self.period_ms
    }

    /// When a unit of `subject` built at `built_ms` expires: the first of the
    /// subject's boundaries at least [`Self::min_standing`] after it — so
    /// never sooner than that, and never more than one period plus that.
    pub fn expires_at(self, subject: &str, built_ms: u64) -> u64 {
        let phase = self.phase(subject);
        let next = ((built_ms + phase) / self.period_ms + 1) * self.period_ms - phase;
        if next - built_ms < self.min_standing() {
            next + self.period_ms
        } else {
            next
        }
    }

    /// The term a unit of `subject` read at `now_ms` stands for — the boundary
    /// it will expire at. What its key is made distinct by: a unit read again
    /// once its term ends is read for the next one.
    pub fn term(self, subject: &str, now_ms: u64) -> u64 {
        self.expires_at(subject, now_ms)
    }

    /// Whether a unit of `subject` built at `built_ms` has expired at
    /// `now_ms`. One whose build time is unknown has: nothing says how long it
    /// has stood, and it stands until it is replaced either way.
    pub fn expired(self, subject: &str, built_ms: Option<u64>, now_ms: u64) -> bool {
        match built_ms {
            Some(built) => now_ms >= self.expires_at(subject, built),
            None => true,
        }
    }
}

/// The wall clock, in milliseconds since the epoch.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: u64 = 1_792_886_400_000; // 2026-10-25T00:00:00Z
    const HOUR_MS: u64 = 3_600_000;

    fn thirty() -> Expiry {
        Expiry::days(30).unwrap()
    }

    #[test]
    fn zero_days_is_no_expiry() {
        assert_eq!(Expiry::days(0), None);
        assert_eq!(Expiry::days(1).unwrap().period_ms(), DAY_MS);
        assert_eq!(thirty().period_ms(), 30 * DAY_MS);
        assert_eq!(thirty().min_standing(), DAY_MS);
    }

    /// A unit stands at least its minimum and at most one period plus it.
    #[test]
    fn a_unit_stands_between_its_minimum_and_one_period_more() {
        let e = thirty();
        for n in 0..500 {
            let subject = format!("candle/src/file_{n}.rs");
            let stood = e.expires_at(&subject, T0) - T0;
            assert!(stood >= e.min_standing(), "{subject} stood {stood} ms");
            assert!(
                stood <= e.period_ms() + e.min_standing(),
                "{subject} stood {stood} ms"
            );
        }
    }

    /// **Read again at its boundary, a unit stands exactly one period.** Its
    /// next boundary is one period after the one it was read at, so once the
    /// layer is under way each file is read once a period — and read a little
    /// late, it still expires at that same boundary: the phase holds.
    #[test]
    fn read_again_at_its_boundary_a_unit_stands_one_period() {
        let e = thirty();
        let subject = "candle/zend/src/main.rs";
        let first = e.expires_at(subject, T0);
        assert_eq!(e.expires_at(subject, first), first + e.period_ms());
        assert_eq!(e.expires_at(subject, first + DAY_MS), first + e.period_ms());
    }

    /// **Rebuilt just before its boundary — the file was edited — a unit skips
    /// it** and stands to the next, rather than being read again within the
    /// hour.
    #[test]
    fn rebuilt_just_before_its_boundary_a_unit_stands_to_the_next() {
        let e = thirty();
        let subject = "candle/zend/src/session.rs";
        let boundary = e.expires_at(subject, T0);
        assert_eq!(
            e.expires_at(subject, boundary - HOUR_MS),
            boundary + e.period_ms()
        );
        // Two days before it, the boundary is far enough off to keep.
        assert_eq!(e.expires_at(subject, boundary - 2 * DAY_MS), boundary);
    }

    /// **Units built together expire spread over the period**, not together:
    /// across many files built at one moment, every tenth of the window they
    /// expire in has some expiring in it.
    #[test]
    fn units_built_together_expire_spread_over_the_period() {
        let e = thirty();
        let mut tenths = [0usize; 10];
        for n in 0..1000 {
            let at = e.expires_at(&format!("candle/src/f{n}.rs"), T0);
            let tenth = ((at - T0 - e.min_standing() - 1) * 10 / e.period_ms()) as usize;
            tenths[tenth] += 1;
        }
        for (i, count) in tenths.iter().enumerate() {
            assert!(
                (50..=150).contains(count),
                "tenth {i} of the period has {count} of 1000 expiries: {tenths:?}"
            );
        }
    }

    /// The phase is the path's alone: the same on every call, and different
    /// between paths.
    #[test]
    fn the_phase_is_the_paths() {
        let e = thirty();
        assert_eq!(e.expires_at("a/x.rs", T0), e.expires_at("a/x.rs", T0));
        assert_ne!(e.expires_at("a/x.rs", T0), e.expires_at("a/y.rs", T0));
    }

    #[test]
    fn a_unit_expires_at_its_boundary_and_not_before() {
        let e = thirty();
        let at = e.expires_at("a/x.rs", T0);
        assert!(!e.expired("a/x.rs", Some(T0), at - 1));
        assert!(e.expired("a/x.rs", Some(T0), at));
        assert!(e.expired("a/x.rs", None, T0));
    }

    /// **The replacement is read for a term of its own.** Read at the moment
    /// the old unit expires, the term is the next boundary, never the one the
    /// old unit stood for — so the two are keyed apart.
    #[test]
    fn a_replacement_is_read_for_the_next_term() {
        let e = thirty();
        let subject = "candle/zend/src/lib.rs";
        let old_term = e.term(subject, T0);
        let new_term = e.term(subject, old_term);
        assert!(new_term > old_term);
        assert_eq!(new_term, old_term + e.period_ms());
        // Every member reading within one term reads for the same one.
        assert_eq!(e.term(subject, T0), e.term(subject, T0 + HOUR_MS));
    }
}
