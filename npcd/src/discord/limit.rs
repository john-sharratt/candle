//! The host's share of Discord's invalid-request limit (`docs/npcd_discord.md`
//! §2, §6).
//!
//! Discord bans an IP — Cloudflare, in front of its API — for an hour or
//! more past 10,000 `401`, `403` and `429` answers in ten minutes, counted
//! over every token and every bot sent from that IP. Every bot of this
//! process sends through one counter, a **rolling** window of the invalid
//! answers of the last [`WINDOW`]:
//!
//! - under the **throttle** line (5,000 by default) a request goes at once;
//! - from it, requests are spaced one per time slice, `WINDOW / (ceiling −
//!   throttle)` — 200 ms at the defaults — so even if every one of them were
//!   refused, the window could not reach the ceiling;
//! - at the **ceiling** (8,000 by default) nothing goes until the oldest
//!   counted answer leaves the window.
//!
//! A request held back is answered as `DiscordError::Paused` with how long
//! to wait, which every caller holds off for. The lines are configurable
//! (`--discord-invalid-throttle`, `--discord-invalid-ceiling`): every member
//! behind one public address shares Discord's 10,000, so each takes its own
//! part of it.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How far back the count runs: Discord's own window.
pub const WINDOW: Duration = Duration::from_secs(10 * 60);

/// From how many invalid answers in a window requests are throttled.
pub const THROTTLE: u32 = 5_000;

/// At how many invalid answers in a window nothing more is sent: under
/// Discord's ban line of 10,000.
pub const CEILING: u32 = 8_000;

/// The invalid answers of the last window, and when the last request went.
#[derive(Debug)]
pub struct InvalidRequests {
    throttle: u32,
    ceiling: u32,
    window: Duration,
    held: Mutex<Held>,
}

#[derive(Debug, Default)]
struct Held {
    /// When each invalid answer of the window came, oldest first.
    answers: VecDeque<Instant>,
    /// When the last request was let go.
    last_sent: Option<Instant>,
    /// Which line the count last crossed, so each crossing is said once.
    said: Line,
}

/// Where the count stands against the lines.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Line {
    #[default]
    Under,
    Throttled,
    Stopped,
}

impl Default for InvalidRequests {
    fn default() -> Self {
        Self::new(THROTTLE, CEILING, WINDOW)
    }
}

impl InvalidRequests {
    /// A counter that throttles from `throttle` invalid answers a `window`
    /// and stops at `ceiling`. A ceiling at or under the throttle line stops
    /// at the throttle line.
    pub fn new(throttle: u32, ceiling: u32, window: Duration) -> Self {
        Self {
            throttle,
            ceiling: ceiling.max(throttle),
            window,
            held: Mutex::new(Held::default()),
        }
    }

    /// The spacing between requests while throttled: what keeps a window
    /// of all-refused requests from reaching the ceiling.
    pub fn slice(&self) -> Duration {
        match self.ceiling - self.throttle {
            0 => self.window,
            room => self.window / room,
        }
    }

    /// Whether a request may go at `now`; if not, how many seconds until it
    /// may. A request that may go is counted as sent.
    pub fn check(&self, now: Instant) -> Result<(), f64> {
        let mut held = self.held.lock().unwrap();
        self.forget_before(&mut held, now);
        let count = held.answers.len() as u32;
        if count >= self.ceiling {
            let oldest = *held.answers.front().expect("at the ceiling");
            let wait = (oldest + self.window).saturating_duration_since(now);
            self.say(&mut held, Line::Stopped, count);
            return Err(wait.as_secs_f64().max(0.001));
        }
        if count >= self.throttle {
            self.say(&mut held, Line::Throttled, count);
            if let Some(last) = held.last_sent {
                let next = last + self.slice();
                if now < next {
                    return Err((next - now).as_secs_f64());
                }
            }
        } else {
            self.say(&mut held, Line::Under, count);
        }
        held.last_sent = Some(now);
        Ok(())
    }

    /// Discord answered `status` at `now`: counted when it is one of the
    /// answers the limit counts.
    pub fn answered(&self, status: u16, now: Instant) {
        if !matches!(status, 401 | 403 | 429) {
            return;
        }
        let mut held = self.held.lock().unwrap();
        self.forget_before(&mut held, now);
        held.answers.push_back(now);
    }

    /// How many invalid answers the window holds at `now`.
    pub fn counted(&self, now: Instant) -> u32 {
        let mut held = self.held.lock().unwrap();
        self.forget_before(&mut held, now);
        held.answers.len() as u32
    }

    /// Drop the answers older than the window.
    fn forget_before(&self, held: &mut Held, now: Instant) {
        while held
            .answers
            .front()
            .is_some_and(|at| now.saturating_duration_since(*at) >= self.window)
        {
            held.answers.pop_front();
        }
    }

    /// Say once when the count crosses into `line`.
    fn say(&self, held: &mut Held, line: Line, count: u32) {
        if held.said == line {
            return;
        }
        held.said = line;
        let window = self.window.as_secs();
        match line {
            Line::Stopped => tracing::error!(
                "discord: {count} invalid requests in the last {window} s, at the ceiling of {}: \
                 nothing more is sent to Discord until the oldest leaves the window",
                self.ceiling
            ),
            Line::Throttled => tracing::warn!(
                "discord: {count} invalid requests in the last {window} s, past {}: every request \
                 to Discord is spaced {} ms apart so the window stays under {}",
                self.throttle,
                self.slice().as_millis(),
                self.ceiling
            ),
            Line::Under => tracing::info!(
                "discord: {count} invalid requests in the last {window} s: requests go unthrottled"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    /// **Under the throttle line everything goes; from it, one request a
    /// slice; at the ceiling nothing** — and only `401`, `403` and `429`
    /// count.
    #[test]
    fn requests_go_then_are_spaced_then_stop() {
        // Throttle at 3, stop at 5, a 100 s window: a 50 s slice.
        let limit = InvalidRequests::new(3, 5, s(100));
        assert_eq!(limit.slice(), s(50));
        let t0 = Instant::now();
        for status in [200, 404, 500] {
            limit.answered(status, t0);
        }
        assert_eq!(limit.counted(t0), 0);
        for _ in 0..3 {
            assert_eq!(limit.check(t0), Ok(()), "under the line, at once");
        }
        limit.answered(401, t0);
        limit.answered(403, t0);
        limit.answered(429, t0);
        // At the throttle line: the last went at t0, so the next waits out
        // its slice.
        assert_eq!(limit.check(t0 + s(10)), Err(40.0));
        assert_eq!(limit.check(t0 + s(50)), Ok(()));
        assert_eq!(limit.check(t0 + s(60)), Err(40.0));
        limit.answered(429, t0 + s(60));
        limit.answered(429, t0 + s(60));
        // At the ceiling: nothing until the oldest leaves, at t0 + 100.
        assert_eq!(limit.check(t0 + s(99)), Err(1.0));
        assert_eq!(limit.counted(t0 + s(100)), 2);
        assert_eq!(limit.check(t0 + s(100)), Ok(()), "the window rolled");
    }

    /// **The window rolls**: an answer counts for exactly one window from
    /// when it came, not until a fixed window restarts.
    #[test]
    fn each_answer_counts_for_one_window_from_when_it_came() {
        let limit = InvalidRequests::new(10, 20, s(600));
        let t0 = Instant::now();
        limit.answered(429, t0);
        limit.answered(429, t0 + s(300));
        assert_eq!(limit.counted(t0 + s(599)), 2);
        assert_eq!(limit.counted(t0 + s(600)), 1);
        assert_eq!(limit.counted(t0 + s(900)), 0);
    }

    /// **At the defaults the slice keeps a window of refusals under the
    /// ceiling**: 200 ms apart from 5,000, for at most 3,000 more.
    #[test]
    fn at_the_defaults_the_slice_keeps_the_window_under_the_ceiling() {
        let limit = InvalidRequests::default();
        assert_eq!(limit.slice(), Duration::from_millis(200));
        assert_eq!(WINDOW.as_millis() / limit.slice().as_millis(), 3_000);
        let lone = InvalidRequests::new(7, 3, s(60));
        assert_eq!(lone.slice(), s(60), "a ceiling under the line stops at it");
    }
}
