//! How long an account waits before its next `diary_open` after opening
//! nothing again and again (`docs/private_mode.md` §5.4): the first
//! [`FREE`] such opens are free, then each waits twice the last, up to
//! [`MAX_WAIT_MS`]. One open that unlocks a diary clears it. Kept in memory
//! only, and counted per account — never what was tried.

use std::collections::HashMap;
use std::sync::Mutex;

/// Opens that unlock nothing before the waits begin.
const FREE: u32 = 3;

/// The first wait.
const FIRST_WAIT_MS: u64 = 2_000;

/// The longest wait.
pub const MAX_WAIT_MS: u64 = 60 * 60 * 1000;

/// Each account's opens that unlocked nothing, and when it may try again.
#[derive(Debug, Default)]
pub struct Backoff(Mutex<HashMap<String, (u32, u64)>>);

impl Backoff {
    /// Whether `owner` may open at `now_ms`; how long it must wait, when not.
    pub fn check(&self, owner: &str, now_ms: u64) -> Result<(), u64> {
        let held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        match held.get(owner) {
            Some(&(_, until)) if until > now_ms => Err(until - now_ms),
            _ => Ok(()),
        }
    }

    /// `owner`'s open at `now_ms` unlocked nothing.
    pub fn failed(&self, owner: &str, now_ms: u64) {
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let (fails, until) = held.entry(owner.to_string()).or_insert((0, 0));
        *fails += 1;
        if *fails >= FREE {
            let doublings = (*fails - FREE).min(20);
            *until = now_ms + (FIRST_WAIT_MS << doublings).min(MAX_WAIT_MS);
        }
    }

    /// `owner`'s open unlocked a diary.
    pub fn cleared(&self, owner: &str) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(owner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_that_unlock_nothing_wait_twice_as_long_each_time() {
        let b = Backoff::default();
        for i in 0..FREE - 1 {
            b.failed("o", i as u64);
            assert_eq!(b.check("o", i as u64), Ok(()), "free");
        }
        b.failed("o", 10);
        assert_eq!(b.check("o", 10), Err(FIRST_WAIT_MS));
        assert_eq!(b.check("o", 10 + FIRST_WAIT_MS), Ok(()));
        b.failed("o", 10_000);
        assert_eq!(b.check("o", 10_000), Err(2 * FIRST_WAIT_MS));
        assert_eq!(b.check("other", 10_000), Ok(()), "per account");
        for _ in 0..40 {
            b.failed("o", 0);
        }
        assert_eq!(b.check("o", 0), Err(MAX_WAIT_MS));
        b.cleared("o");
        assert_eq!(b.check("o", 0), Ok(()));
    }
}
