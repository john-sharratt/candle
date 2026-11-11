//! The run loop's wait when it has no work: blocked on its channel for a
//! request, in slices that end each time the wave window falls due, so the
//! window closes on time while nothing runs.
//!
//! Each closed window is one telemetry sample, so an idle member answers every
//! 5-second slot of `/v1/telemetry` — its card, the weights it holds, zero
//! work — rather than leaving the slots of a whole rates period empty and
//! dropping out of the performance page and of the cluster's sum
//! (`docs/performance_series.md` §2).

use std::time::{Duration, Instant};

use flume::RecvTimeoutError;

use super::interleave::reseed_achievable_weight;
use super::step_census::Activity;
use super::{Scheduler, SchedulerRequest};

impl Scheduler {
    /// Wait up to `wait` for a request, closing the wave window each time it
    /// falls due. Each slice's wait is counted idle as it ends, so a window
    /// closed mid-wait holds the idle time it saw and the next window holds
    /// the rest. `Timeout` once the whole wait has passed with no request.
    pub(super) fn recv_idle(
        &mut self,
        wait: Duration,
    ) -> Result<SchedulerRequest, RecvTimeoutError> {
        let deadline = Instant::now() + wait;
        loop {
            let t = Instant::now();
            let slice = deadline
                .saturating_duration_since(t)
                .min(self.wave_stats.until_due());
            let got = self.rx.recv_timeout(slice);
            let waited = t.elapsed();
            self.step_census.spend(Activity::Idle, waited);
            self.wave_stats.add_idle(waited.as_millis() as u64);
            match got {
                Err(RecvTimeoutError::Timeout) => {
                    if self.wave_stats.due() {
                        self.flush_wave_telemetry();
                        // The flush sweeps emptied arenas back to the free
                        // list; with nothing in flight the achievable weight
                        // residency is exact again, and the figure taken as
                        // the loop fell idle no longer is.
                        reseed_achievable_weight();
                    }
                    if Instant::now() >= deadline {
                        return Err(RecvTimeoutError::Timeout);
                    }
                }
                other => return other,
            }
        }
    }
}
