//! Where a request's time went on the wave thread: named laps, logged with the
//! request when it held the thread long (`Scheduler::handle_request`). A slow
//! request was named but not its slow step — a `free_sequence` held the wave
//! thread 1,035 ms with one decode waiting behind it (BlackWorld, 2026-10-27)
//! and nothing said which of its nine steps took it.

use std::fmt;
use std::time::Instant;

/// The laps of the request in hand.
#[derive(Default)]
pub(super) struct RequestLaps {
    last: Option<Instant>,
    laps: Vec<(&'static str, u64)>,
}

impl RequestLaps {
    /// Start timing a request, dropping the last one's laps.
    pub(super) fn start(&mut self) {
        self.last = Some(Instant::now());
        self.laps.clear();
    }

    /// Close the step `name`: the time since the previous lap, or the start.
    pub(super) fn lap(&mut self, name: &'static str) {
        let now = Instant::now();
        if let Some(last) = self.last {
            self.laps
                .push((name, now.saturating_duration_since(last).as_micros() as u64));
        }
        self.last = Some(now);
    }

    /// Whether the request in hand lapped at all.
    pub(super) fn any(&self) -> bool {
        !self.laps.is_empty()
    }
}

/// `name=12.3ms` per lap, in the order they ran.
impl fmt::Display for RequestLaps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (name, us)) in self.laps.iter().enumerate() {
            if i > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{name}={:.1}ms", *us as f64 / 1000.0)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each lap by name and milliseconds, in the order it ran.
    #[test]
    fn laps_read_by_name_in_order() {
        let laps = RequestLaps {
            last: None,
            laps: vec![("session_free", 1_035_200), ("model_release", 40)],
        };
        assert_eq!(
            laps.to_string(),
            "session_free=1035.2ms model_release=0.0ms"
        );
        assert!(laps.any());
        assert_eq!(RequestLaps::default().to_string(), "");
    }

    /// A lap is only taken once a request has started, and starting drops the
    /// last request's laps.
    #[test]
    fn a_lap_needs_a_start_and_a_start_clears() {
        let mut laps = RequestLaps::default();
        laps.lap("before");
        assert!(!laps.any());
        laps.start();
        laps.lap("first");
        assert!(laps.any());
        laps.start();
        assert!(!laps.any());
    }
}
