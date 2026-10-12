//! Backing off a decode-less wave that would not run.
//!
//! With no decode in flight, the wave step carries only prefills and section
//! chunks, and a failure leaves every one of them where it was. The loop came
//! straight back round, rebuilt the same wave against the same ground, and
//! failed it again — about every 2.5 ms, 1,574 identical errors in four seconds,
//! with the card held and nothing else on the scheduler thread getting a turn.
//!
//! The same wave only runs once something changes: a request lands, a relief or
//! compaction pass hands ground back, or the weight side, which refuses to move
//! while a forward or an expert invocation is in flight, concedes between steps.
//! So a failed step waits before the next, doubling from [`FIRST_WAIT`] to
//! [`MAX_WAIT`], and a request ends the wait at once. The error is logged on
//! power-of-two attempts only, so a run of them is a handful of lines.
//!
//! When it is still failing after [`GIVE_UP_AFTER`] attempts, nothing that
//! could change has: no decode runs on this path, so the only work that could
//! return ground is the wave itself. Its members are then failed with the error,
//! which reaches their callers and gives their ground back, rather than the
//! engine standing on a wave it can never place.

use std::time::Duration;

/// The wait after the first failure.
pub(super) const FIRST_WAIT: Duration = Duration::from_micros(500);

/// The longest wait between attempts. Short on purpose: the ground a retry
/// waits for usually comes back within a step or two, and a longer wait is
/// idle card for every one that does.
pub(super) const MAX_WAIT: Duration = Duration::from_millis(5);

/// Consecutive failures after which the wave's members are failed — about five
/// seconds at the cap, long enough for relief, compaction and the weight side's
/// between-step concessions to have run many times over.
pub(super) const GIVE_UP_AFTER: u32 = 1024;

/// What the loop does after a failed step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Next {
    /// Wait this long, or until a request arrives, then try again.
    Retry { wait: Duration, attempt: u32 },
    /// Fail the wave's members.
    GiveUp { attempts: u32 },
}

/// Consecutive failures of the decode-less wave step, and the wait owed before
/// the next attempt.
#[derive(Debug, Default)]
pub(super) struct WaveRetry {
    failures: u32,
    owed: Option<Duration>,
}

impl WaveRetry {
    /// A step failed. Records it and says what the loop does next.
    pub(super) fn failed(&mut self) -> Next {
        self.failures += 1;
        if self.failures >= GIVE_UP_AFTER {
            let attempts = self.failures;
            *self = Self::default();
            return Next::GiveUp { attempts };
        }
        let wait = FIRST_WAIT
            .saturating_mul(1u32 << (self.failures - 1).min(16))
            .min(MAX_WAIT);
        self.owed = Some(wait);
        Next::Retry {
            wait,
            attempt: self.failures,
        }
    }

    /// A step ran. Returns how many failures preceded it, if any, so a recovery
    /// is logged once.
    pub(super) fn succeeded(&mut self) -> Option<u32> {
        let failures = std::mem::take(&mut self.failures);
        self.owed = None;
        (failures > 0).then_some(failures)
    }

    /// The wait owed before the next attempt, handed out once.
    pub(super) fn take_wait(&mut self) -> Option<Duration> {
        self.owed.take()
    }

    /// Whether this failure is logged in full: the first, and then each one
    /// whose attempt number is a power of two, so a run of them reads as a
    /// handful of lines however long it lasts.
    pub(super) fn logs(attempt: u32) -> bool {
        attempt.is_power_of_two()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The waits double from 0.5 ms and stop at 5 ms.
    #[test]
    fn the_wait_doubles_to_its_cap() {
        let mut r = WaveRetry::default();
        let us: Vec<u128> = (0..7)
            .map(|_| match r.failed() {
                Next::Retry { wait, .. } => wait.as_micros(),
                other => panic!("gave up early: {other:?}"),
            })
            .collect();
        assert_eq!(us, vec![500, 1_000, 2_000, 4_000, 5_000, 5_000, 5_000]);
    }

    /// A wait is owed once: the loop waits, then attempts, and does not wait
    /// again until another failure.
    #[test]
    fn a_wait_is_handed_out_once() {
        let mut r = WaveRetry::default();
        assert_eq!(r.take_wait(), None);
        r.failed();
        assert_eq!(r.take_wait(), Some(Duration::from_micros(500)));
        assert_eq!(r.take_wait(), None);
    }

    /// A step that runs clears the count and the owed wait, and reports how
    /// long the run of failures was.
    #[test]
    fn a_step_that_runs_resets_the_backoff() {
        let mut r = WaveRetry::default();
        assert_eq!(r.succeeded(), None);
        r.failed();
        r.failed();
        r.failed();
        assert_eq!(r.succeeded(), Some(3));
        assert_eq!(r.take_wait(), None);
        assert_eq!(
            r.failed(),
            Next::Retry {
                wait: Duration::from_micros(500),
                attempt: 1
            }
        );
    }

    /// The 1,024th consecutive failure gives up, and the count starts over.
    #[test]
    fn it_gives_up_after_1024_failures() {
        let mut r = WaveRetry::default();
        for _ in 0..1023 {
            assert!(matches!(r.failed(), Next::Retry { .. }));
        }
        assert_eq!(r.failed(), Next::GiveUp { attempts: 1024 });
        assert_eq!(r.take_wait(), None);
        assert!(matches!(r.failed(), Next::Retry { attempt: 1, .. }));
    }

    /// Logged in full on attempts 1, 2, 4, … 512 — ten lines for a run that
    /// stops just short of giving up.
    #[test]
    fn a_run_of_failures_logs_a_handful_of_lines() {
        let logged: Vec<u32> = (1..GIVE_UP_AFTER).filter(|&a| WaveRetry::logs(a)).collect();
        assert_eq!(logged, vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512]);
    }
}
