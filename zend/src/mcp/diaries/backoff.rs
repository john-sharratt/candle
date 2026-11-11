//! What guards `diary_open` against guessing (`docs/private_mode.md` §5.4),
//! in memory only, and counting nothing of what was tried:
//!
//! - **[`Backoff`]**: every open is counted against its account *before* its
//!   passphrase is tried, one at a time per account. The first [`FREE`]
//!   opens that unlock nothing are free; after them each waits twice the
//!   last, up to [`MAX_WAIT_MS`]. An open that unlocks a diary takes back its
//!   own count and nothing more, so a passphrase of one's own is no way
//!   round the waits; a count unused for a day is let go. An open refused
//!   before its passphrase could be tried leaves the count as it found it.
//! - **[`Stretching`]**: at most [`AT_ONCE`] passphrases are stretched at a
//!   time on this member, since each takes the stretch's memory.

use std::collections::HashMap;
use std::sync::Mutex;

/// Opens that unlock nothing before the waits begin.
const FREE: u32 = 3;

/// The first wait.
const FIRST_WAIT_MS: u64 = 2_000;

/// The longest wait.
pub const MAX_WAIT_MS: u64 = 60 * 60 * 1000;

/// How long an account's count is kept with nothing new on it.
const FORGOTTEN_AFTER_MS: u64 = 24 * 60 * 60 * 1000;

/// How many passphrases are stretched at once.
pub const AT_ONCE: usize = 2;

/// One account's count.
#[derive(Debug, Clone, Copy, Default)]
struct Count {
    fails: u32,
    until_ms: u64,
    last_ms: u64,
    trying: bool,
}

/// Each account's opens that unlocked nothing.
#[derive(Debug, Default)]
pub struct Backoff(Mutex<HashMap<String, Count>>);

/// One open, counted until it says what it found.
pub struct Attempt<'a> {
    backoff: &'a Backoff,
    owner: String,
    settled: bool,
    /// The account's count as it stood before this open was counted.
    before: Count,
}

impl Backoff {
    /// Count an open by `owner` at `now_ms`, before its passphrase is
    /// tried — or how long it must wait first. One open of an account's runs
    /// at a time.
    pub fn attempt(&self, owner: &str, now_ms: u64) -> Result<Attempt<'_>, u64> {
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let c = held.entry(owner.to_string()).or_default();
        if now_ms.saturating_sub(c.last_ms) > FORGOTTEN_AFTER_MS {
            *c = Count::default();
        }
        if c.until_ms > now_ms {
            return Err(c.until_ms - now_ms);
        }
        if c.trying {
            return Err(FIRST_WAIT_MS);
        }
        let before = *c;
        c.trying = true;
        c.fails += 1;
        c.last_ms = now_ms;
        if c.fails >= FREE {
            let doublings = (c.fails - FREE).min(20);
            c.until_ms = now_ms + (FIRST_WAIT_MS << doublings).min(MAX_WAIT_MS);
        }
        Ok(Attempt {
            backoff: self,
            owner: owner.to_string(),
            settled: false,
            before,
        })
    }
}

impl Attempt<'_> {
    /// What the open found: one that unlocked a diary takes its own count
    /// back, and the wait it set.
    pub fn found(mut self, unlocked: bool) {
        self.settled = true;
        let mut held = self.backoff.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = held.get_mut(&self.owner) {
            c.trying = false;
            if unlocked {
                c.fails = c.fails.saturating_sub(1);
                if c.fails < FREE {
                    c.until_ms = 0;
                }
            }
        }
    }

    /// The open was refused before its passphrase was tried — the memory
    /// unavailable, the member busy stretching others: the account's count
    /// is as it was before it, its wait with it. One open of an account's
    /// runs at a time, so nothing was counted since.
    pub fn untried(mut self) {
        self.settled = true;
        let mut held = self.backoff.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = held.get_mut(&self.owner) {
            *c = self.before;
        }
    }
}

impl Drop for Attempt<'_> {
    /// An open that ended without saying what it found counts as one that
    /// found nothing.
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        let mut held = self.backoff.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = held.get_mut(&self.owner) {
            c.trying = false;
        }
    }
}

/// The passphrases being stretched on this member.
#[derive(Debug, Default)]
pub struct Stretching(Mutex<usize>);

/// One stretch's place, given back when dropped.
pub struct Slot<'a>(&'a Stretching);

impl Stretching {
    /// A place to stretch a passphrase in, or why there is none now.
    pub fn take(&self) -> Result<Slot<'_>, String> {
        let mut n = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if *n >= AT_ONCE {
            return Err("diaries are busy opening others; try again in a few seconds".into());
        }
        *n += 1;
        Ok(Slot(self))
    }
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        let mut n = self.0 .0.lock().unwrap_or_else(|e| e.into_inner());
        *n -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fail(b: &Backoff, owner: &str, at: u64) {
        b.attempt(owner, at).unwrap().found(false);
    }

    #[test]
    fn opens_that_unlock_nothing_wait_twice_as_long_each_time() {
        let b = Backoff::default();
        for i in 0..FREE - 1 {
            fail(&b, "o", i as u64);
            assert!(
                b.attempt("o", i as u64).map(|a| a.found(true)).is_ok(),
                "free"
            );
        }
        fail(&b, "o", 10);
        assert_eq!(b.attempt("o", 10).err(), Some(FIRST_WAIT_MS));
        fail(&b, "o", 10 + FIRST_WAIT_MS);
        assert_eq!(
            b.attempt("o", 10 + FIRST_WAIT_MS).err(),
            Some(2 * FIRST_WAIT_MS)
        );
        assert!(b.attempt("other", 10).is_ok(), "per account");
        let mut at = 10_000_000;
        for _ in 0..40 {
            if let Ok(a) = b.attempt("o", at) {
                a.found(false);
            }
            at += MAX_WAIT_MS;
        }
        assert_eq!(b.attempt("o", at - MAX_WAIT_MS).err(), Some(MAX_WAIT_MS));
        assert!(
            b.attempt("o", at + FORGOTTEN_AFTER_MS + 1).is_ok(),
            "forgotten after a day"
        );
    }

    /// **An open of one's own passphrase is no way round the waits**: it takes
    /// back its own count, never the misses before it.
    #[test]
    fn a_success_takes_back_only_its_own_count() {
        let b = Backoff::default();
        for i in 0..FREE {
            fail(&b, "o", i as u64 * 10_000_000);
        }
        let later = 30_000_000;
        b.attempt("o", later).unwrap().found(true);
        assert!(b.attempt("o", later).is_err(), "the misses still stand");
    }

    /// **An open refused before its passphrase was tried counts nothing**:
    /// however many, the account's free opens and its waits are as they were.
    #[test]
    fn an_untried_open_leaves_the_count_as_it_was() {
        let b = Backoff::default();
        fail(&b, "o", 0);
        for _ in 0..10 {
            b.attempt("o", 1).unwrap().untried();
        }
        fail(&b, "o", 2);
        assert!(b.attempt("o", 3).is_ok(), "the third open is still free");
    }

    /// **One open at a time per account**, and the count stands for one that
    /// ends without saying.
    #[test]
    fn one_open_runs_at_a_time_per_account() {
        let b = Backoff::default();
        let first = b.attempt("o", 0).unwrap();
        assert!(b.attempt("o", 0).is_err(), "while one is trying");
        drop(first);
        assert!(b.attempt("o", 0).is_ok());
    }

    #[test]
    fn at_most_two_stretch_at_once() {
        let s = Stretching::default();
        let a = s.take().unwrap();
        let _b = s.take().unwrap();
        assert!(s.take().is_err());
        drop(a);
        assert!(s.take().is_ok());
    }
}
