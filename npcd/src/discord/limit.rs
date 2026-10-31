//! The host's share of Discord's invalid-request limit (`docs/npcd_discord.md`
//! §2, §6).
//!
//! Discord bans an IP for ten minutes or more past 10,000 `401`, `403` and
//! `429` answers in ten minutes, counted over every token sent from it. So
//! every bot's calls go through one counter: past [`CEILING`] such answers in
//! a [`WINDOW`], nothing more is sent until the window ends, and the log says
//! so once.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How long the count runs before it starts again.
pub const WINDOW: Duration = Duration::from_secs(10 * 60);

/// How many invalid answers a window takes before sending stops: half of
/// Discord's ban line.
pub const CEILING: u32 = 5_000;

/// The invalid answers counted in this window.
#[derive(Debug)]
pub struct InvalidRequests {
    ceiling: u32,
    window: Duration,
    held: Mutex<Option<(Instant, u32)>>,
}

impl Default for InvalidRequests {
    fn default() -> Self {
        Self::new(CEILING, WINDOW)
    }
}

impl InvalidRequests {
    /// A counter that stops sending past `ceiling` invalid answers a `window`.
    pub fn new(ceiling: u32, window: Duration) -> Self {
        Self {
            ceiling,
            window,
            held: Mutex::new(None),
        }
    }

    /// Whether a request may go at `now`; if not, how many seconds until it
    /// may.
    pub fn check(&self, now: Instant) -> Result<(), f64> {
        let mut held = self.held.lock().unwrap();
        match *held {
            Some((since, _)) if now.duration_since(since) >= self.window => {
                *held = None;
                Ok(())
            }
            Some((since, count)) if count >= self.ceiling => {
                Err((self.window - now.duration_since(since)).as_secs_f64())
            }
            _ => Ok(()),
        }
    }

    /// Discord answered `status` at `now`: counted when it is one of the
    /// answers the limit counts.
    pub fn answered(&self, status: u16, now: Instant) {
        if !matches!(status, 401 | 403 | 429) {
            return;
        }
        let mut held = self.held.lock().unwrap();
        let (since, count) = match *held {
            Some((since, count)) if now.duration_since(since) < self.window => (since, count + 1),
            _ => (now, 1),
        };
        *held = Some((since, count));
        if count == self.ceiling {
            tracing::error!(
                "discord: {count} invalid requests in {} s — nothing more is sent to Discord until \
                 the window ends, so the host is not banned",
                self.window.as_secs()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Past the ceiling nothing goes, until the window ends**; only `401`,
    /// `403` and `429` count.
    #[test]
    fn past_the_ceiling_nothing_goes_until_the_window_ends() {
        let limit = InvalidRequests::new(3, Duration::from_secs(600));
        let t0 = Instant::now();
        limit.answered(200, t0);
        limit.answered(404, t0);
        limit.answered(500, t0);
        assert_eq!(limit.check(t0), Ok(()));
        limit.answered(401, t0);
        limit.answered(403, t0 + Duration::from_secs(1));
        assert_eq!(limit.check(t0 + Duration::from_secs(1)), Ok(()));
        limit.answered(429, t0 + Duration::from_secs(2));
        assert_eq!(limit.check(t0 + Duration::from_secs(100)), Err(500.0));
        assert_eq!(limit.check(t0 + Duration::from_secs(600)), Ok(()));
        // The next window counts from its own first.
        limit.answered(401, t0 + Duration::from_secs(601));
        assert_eq!(limit.check(t0 + Duration::from_secs(602)), Ok(()));
    }
}
