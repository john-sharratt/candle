//! What a wave costs on the scheduler loop, wave by wave.
//!
//! A wave here is one pass of the run loop that ran at least one forward. Its
//! wall-clock runs from the top of the pass to the end of its housekeeping, and
//! its overhead is the part spent outside the quanta — the submission drain and
//! the housekeeping between waves, the same figure the rate planner prices a
//! forward's overhead from (`Scheduler::wave_overhead_us`). The rest is the
//! quanta: the forwards and the work each quantum does around its own.
//!
//! Counted per pass rather than per forward: a co-batched sweep records a row
//! in each class that rode it, and a creep splits one sweep into several layer
//! windows, so neither the class records nor the `forward_wave` calls count
//! waves.

use std::mem::take;

/// One telemetry window's waves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct WaveTiming {
    pub waves: u64,
    pub wall_us: u64,
    pub overhead_us: u64,
}

impl WaveTiming {
    /// Count one wave: `wall_us` from the top of its pass, `overhead_us` of it
    /// outside the quanta.
    pub(super) fn add(&mut self, wall_us: u64, overhead_us: u64) {
        self.waves += 1;
        self.wall_us += wall_us;
        // The overhead is part of the wall it was measured inside.
        self.overhead_us += overhead_us.min(wall_us);
    }

    /// The window's figures, and a fresh window.
    pub(super) fn take(&mut self) -> Self {
        take(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Waves add up, each overhead inside its own wall, and taking the window
    /// leaves it empty.
    #[test]
    fn waves_add_up_and_taking_the_window_empties_it() {
        let mut t = WaveTiming::default();
        t.add(40_000, 6_000);
        t.add(30_000, 45_000);
        assert_eq!(
            t.take(),
            WaveTiming {
                waves: 2,
                wall_us: 70_000,
                overhead_us: 36_000,
            }
        );
        assert_eq!(t, WaveTiming::default());
    }
}
