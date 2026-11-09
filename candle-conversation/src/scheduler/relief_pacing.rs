//! How often the section gate's VRAM relief may run: at most a fixed share of
//! the scheduler's time, priced by each relief's own measured cost.
//!
//! The gate runs before every decode step that has no creep, whenever the KV
//! regions are under their setpoint. Unpaced it ran back to back: on
//! BlackWorld (2026-11-09) one dialogue reprojecting over 200K of history
//! recalled ~75 MB/s of cold turns into hot, and five reliefs of 300–1,050 ms
//! — 3.57 s of 25 — packed the pools behind them, every decode wave after one
//! running at 60–98 ms a forward against 34–39.
//!
//! **The share is [`RELIEF_SHARE`], and the interval after a relief is what keeps
//! it there**: `took × (1 − share) / share`. A cheap relief comes back soon; an
//! expensive one waits in proportion. At 10%, an 850 ms relief waits 7.7 s —
//! the deficit the recall builds in that time, ~575 MB, is about half the
//! setpoint (77 regions, ~1.2 GB), so the pools never run past it, and a relief
//! frees 400–560 MB, the intake of the 5–7 s it is spaced by. At 5% the same
//! relief would wait 16 s and the deficit would reach the whole setpoint; at
//! 15% the pacing would not move the 14% measured at all.

use std::time::{Duration, Instant};

/// The most of the scheduler's time the section gate's relief may take.
pub const RELIEF_SHARE: f64 = 0.10;

/// When the gate's relief last ran and how long it took.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReliefPacing {
    last: Option<(Instant, Duration)>,
}

impl ReliefPacing {
    /// Whether a relief may run at `now`.
    pub fn ready(&self, now: Instant) -> bool {
        self.last
            .is_none_or(|(ended, took)| now >= ended + interval_after(took))
    }

    /// A relief that took `took` ended at `ended`.
    pub fn ran(&mut self, ended: Instant, took: Duration) {
        self.last = Some((ended, took));
    }
}

/// The wait after a relief that took `took`, holding the relief to
/// [`RELIEF_SHARE`] of the time.
pub fn interval_after(took: Duration) -> Duration {
    took.mul_f64((1.0 - RELIEF_SHARE) / RELIEF_SHARE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wait_holds_relief_to_its_share() {
        assert_eq!(
            interval_after(Duration::from_millis(850)),
            Duration::from_millis(7_650)
        );
        assert_eq!(
            interval_after(Duration::from_millis(40)),
            Duration::from_millis(360)
        );
        assert_eq!(interval_after(Duration::ZERO), Duration::ZERO);
    }

    #[test]
    fn a_relief_may_run_again_once_its_wait_has_passed() {
        let t0 = Instant::now();
        let mut pacing = ReliefPacing::default();
        assert!(pacing.ready(t0), "nothing has run yet");
        pacing.ran(t0, Duration::from_millis(100));
        assert!(!pacing.ready(t0 + Duration::from_millis(899)));
        assert!(pacing.ready(t0 + Duration::from_millis(900)));
    }
}
