//! The last hour of readings, one a minute: what a growth verdict is judged
//! over.

use std::collections::VecDeque;

use super::reading::Reading;

/// How many readings are kept: an hour at one a minute, and the one before it.
pub const KEPT: usize = 61;

/// The readings, oldest first.
#[derive(Debug, Default)]
pub struct History {
    readings: VecDeque<Reading>,
}

impl History {
    pub fn new() -> History {
        History {
            readings: VecDeque::with_capacity(KEPT),
        }
    }

    /// Keep `r`, letting the oldest go past [`KEPT`].
    pub fn push(&mut self, r: Reading) {
        if self.readings.len() == KEPT {
            self.readings.pop_front();
        }
        self.readings.push_back(r);
    }

    /// Let every reading go: the process has finished loading, and growth is
    /// judged from here on, not across the load.
    pub fn clear(&mut self) {
        self.readings.clear();
    }

    /// The newest reading.
    pub fn newest(&self) -> Option<&Reading> {
        self.readings.back()
    }

    /// The oldest reading no older than `span_ms` before the newest, and the
    /// newest: the window a verdict is judged over. `None` until two readings
    /// span at least half of it, so a verdict is never drawn from a moment.
    pub fn window(&self, span_ms: u64) -> Option<(&Reading, &Reading)> {
        let newest = self.readings.back()?;
        let from = newest.at_ms.saturating_sub(span_ms);
        let oldest = self.readings.iter().find(|r| r.at_ms >= from)?;
        // Readings are stamped on a monotonic clock, so this never goes
        // backwards; saturating all the same, so a caller that stamped them
        // otherwise gets no window rather than a panic.
        (newest.at_ms.saturating_sub(oldest.at_ms) >= span_ms / 2).then_some((oldest, newest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(at_ms: u64) -> Reading {
        Reading {
            at_ms,
            private_bytes: at_ms,
            heap: None,
            classes: None,
            scopes: None,
            allocated_total: None,
        }
    }

    const MIN: u64 = 60_000;

    #[test]
    fn the_window_spans_from_the_oldest_reading_inside_it() {
        let mut h = History::new();
        for m in 0..=40 {
            h.push(at(m * MIN));
        }
        let (old, new) = h.window(30 * MIN).unwrap();
        assert_eq!((old.at_ms, new.at_ms), (10 * MIN, 40 * MIN));
    }

    /// **No verdict from a moment**: two readings a minute apart span too
    /// little of a 30-minute window to judge it.
    #[test]
    fn a_window_too_short_gives_nothing() {
        let mut h = History::new();
        h.push(at(0));
        h.push(at(MIN));
        assert!(h.window(30 * MIN).is_none());
    }

    /// **A window never reaches back past a clear**: the load's readings are
    /// not judged against what follows it.
    #[test]
    fn a_clear_starts_the_window_again() {
        let mut h = History::new();
        for m in 0..=20 {
            h.push(at(m * MIN));
        }
        h.clear();
        for m in 21..=40 {
            h.push(at(m * MIN));
        }
        let (old, new) = h.window(30 * MIN).unwrap();
        assert_eq!((old.at_ms, new.at_ms), (21 * MIN, 40 * MIN));
    }

    #[test]
    fn an_hour_and_one_is_kept() {
        let mut h = History::new();
        for m in 0..100 {
            h.push(at(m * MIN));
        }
        assert_eq!(h.readings.len(), KEPT);
        assert_eq!(h.readings.front().unwrap().at_ms, 39 * MIN);
    }
}
