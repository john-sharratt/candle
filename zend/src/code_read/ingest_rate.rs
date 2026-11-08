//! **Files read per window**, logged once a window (`docs/kv_drain_under_pressure.md`
//! §2b): the ingest rate as a number in the log, rather than a count of
//! per-file DEBUG lines.
//!
//! A window closes at the first commit past its length, so a quiet spell is
//! one long window; the line carries the window's real length beside its
//! count and the rate divides the two.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How long one window runs.
const WINDOW: Duration = Duration::from_secs(5 * 60);

/// One window's count, from its first commit.
#[derive(Debug, Clone, Copy)]
struct RateWindow {
    start: Instant,
    files: usize,
}

impl RateWindow {
    /// Count a file committed at `now`. Past the window's length, answer the
    /// closed window's files and length, and start the next with this file.
    fn note(&mut self, now: Instant) -> Option<(usize, Duration)> {
        let length = now.saturating_duration_since(self.start);
        if length < WINDOW {
            self.files += 1;
            return None;
        }
        let closed = (self.files, length);
        *self = RateWindow {
            start: now,
            files: 1,
        };
        Some(closed)
    }
}

static RATE: Mutex<Option<RateWindow>> = Mutex::new(None);

/// Count one file read and committed, and write the window's line when one
/// closes.
pub(super) fn note_committed() {
    let now = Instant::now();
    let closed = {
        let Ok(mut rate) = RATE.lock() else { return };
        match rate.as_mut() {
            Some(w) => w.note(now),
            None => {
                *rate = Some(RateWindow {
                    start: now,
                    files: 1,
                });
                None
            }
        }
    };
    if let Some((files, length)) = closed {
        let secs = length.as_secs_f64();
        tracing::info!(
            target: "zend::code_read::ingest",
            files,
            window_s = length.as_secs(),
            files_per_min = files as f64 * 60.0 / secs,
            "code_read: {files} file(s) read in {:.0} s",
            secs,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A window counts until its length, then answers and restarts with the
    /// file that closed it.**
    #[test]
    fn a_window_answers_its_count_once_past_its_length() {
        let t0 = Instant::now();
        let mut w = RateWindow {
            start: t0,
            files: 1,
        };
        assert_eq!(w.note(t0 + Duration::from_secs(10)), None);
        assert_eq!(w.note(t0 + WINDOW - Duration::from_secs(1)), None);
        let late = t0 + WINDOW + Duration::from_secs(30);
        assert_eq!(w.note(late), Some((3, WINDOW + Duration::from_secs(30))));
        assert_eq!((w.start, w.files), (late, 1));
    }
}
