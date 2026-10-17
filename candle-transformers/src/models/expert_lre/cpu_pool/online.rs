//! The split's online loop (`docs/cpu_expert_compute.md` §5.3.2): the
//! measured CPU costs follow what the pool actually takes, and a pool the
//! device keeps waiting on is reported, never silent.
//!
//! **Correction.** Each layer the pool serves is a sample: what it took
//! against what the measured tables predicted for the same experts. The ratio
//! moves every CPU cost through an EWMA whose weight follows wall time
//! (`online_tau_ms`), so a thermal swing reaches the split within a few
//! waves without the loop chasing one layer's noise. A ratio far outside
//! `[1/4, 4]` is a preemption or a page fault, not the pool's speed, and is
//! clamped there.
//!
//! **Lateness.** The device records how long its CPU items waited for the
//! pool (the mapped `wait` word). Waits past [`LATE_NS`] for [`LATE_STREAK`]
//! consecutive CPU layers are logged once per streak, with the likelier cause:
//! the pool itself running slow against its tables, or the handoff.
//!
//! Every input arrives off the critical path: the service runs this after it
//! has published the layer's `done`.

use std::time::Duration;

use super::params::SplitParams;

/// A device wait this long on the pool is late.
pub const LATE_NS: u64 = 50_000;
/// Consecutive late layers before the streak is reported.
pub const LATE_STREAK: u32 = 64;
/// The bounds on one sample's observed / predicted ratio.
const RATIO_BOUNDS: (f32, f32) = (0.25, 4.0);
/// A pool slower than its tables by this much is the cause a late streak names.
const SLOW_POOL: f32 = 1.25;

/// The EWMA weight of a sample `dt` after the last, at time constant `tau`.
pub fn weight(dt: Duration, tau: Duration) -> f32 {
    if tau.is_zero() {
        return 1.0;
    }
    1.0 - (-(dt.as_secs_f32() / tau.as_secs_f32())).exp()
}

/// The factor that moves the CPU costs toward a sample that took `observed`
/// against `predicted`, at weight `w`.
pub fn correction(observed_ns: f32, predicted_ns: f32, w: f32) -> f32 {
    if predicted_ns <= 0.0 || observed_ns <= 0.0 {
        return 1.0;
    }
    let r = (observed_ns / predicted_ns).clamp(RATIO_BOUNDS.0, RATIO_BOUNDS.1);
    1.0 + w * (r - 1.0)
}

/// Why a streak of late layers is late.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LateCause {
    /// The pool took longer than its tables say.
    SlowPool,
    /// The pool kept to its tables; the time went to the handoff.
    Handoff,
}

/// One served layer, as the loop sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation {
    /// Doorbell seen → `done` published.
    pub busy_ns: f32,
    /// What the tables predicted for the same experts, before this sample's
    /// correction.
    pub predicted_ns: f32,
    /// The device's wait on the previous layer the pool served.
    pub device_wait_ns: u64,
}

/// The loop's state between samples.
#[derive(Debug, Default)]
pub struct Online {
    late: u32,
    reported: bool,
    /// The pool's mean slowness over the current late streak.
    slow_sum: f32,
}

impl Online {
    /// Applies `obs` to `params` at weight `w`; returns the cause when this
    /// sample completes a late streak not yet reported.
    pub fn observe(
        &mut self,
        params: &mut SplitParams,
        obs: Observation,
        w: f32,
    ) -> Option<LateCause> {
        params.rescale_cpu(correction(obs.busy_ns, obs.predicted_ns, w));
        if obs.device_wait_ns < LATE_NS {
            *self = Self::default();
            return None;
        }
        self.late += 1;
        if obs.predicted_ns > 0.0 {
            self.slow_sum += obs.busy_ns / obs.predicted_ns;
        }
        if self.late < LATE_STREAK || self.reported {
            return None;
        }
        self.reported = true;
        Some(if self.slow_sum / self.late as f32 > SLOW_POOL {
            LateCause::SlowPool
        } else {
            LateCause::Handoff
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::models::expert_lre::cpu_pool::params::{Knobs, Measured, BUCKETS};

    fn params() -> SplitParams {
        SplitParams {
            enabled: true,
            knobs: Knobs::default(),
            measured: Some(Measured {
                copy_ns: 160e3,
                cpu_read_alone_ns: 20e3,
                cpu_read_shared_ns: 25e3,
                cpu_n_ns: [40e3; BUCKETS.len()],
                gpu_n_ns: [2e3; BUCKETS.len()],
                handoff_ns: 5e3,
                pool_fixed_ns: 0.0,
                row_ns: 1e3,
            }),
            max_experts: 5,
        }
    }

    #[test]
    fn the_weight_follows_wall_time() {
        let tau = Duration::from_millis(500);
        assert_eq!(weight(Duration::ZERO, tau), 0.0);
        assert!((weight(tau, tau) - (1.0 - (-1f32).exp())).abs() < 1e-6);
        assert_eq!(weight(Duration::from_millis(1), Duration::ZERO), 1.0);
    }

    /// A sample twice as slow as predicted moves the costs by the weight's
    /// share of the gap; an absurd ratio is clamped; a missing prediction
    /// moves nothing.
    #[test]
    fn the_correction_is_weighted_and_clamped() {
        assert_eq!(correction(200.0, 100.0, 0.5), 1.5);
        assert_eq!(correction(50.0, 100.0, 1.0), 0.5);
        assert_eq!(correction(10_000.0, 1.0, 1.0), 4.0);
        assert_eq!(correction(1.0, 10_000.0, 1.0), 0.25);
        assert_eq!(correction(100.0, 0.0, 1.0), 1.0);
    }

    /// Converges: a pool steadily 1.5× slower than its tables brings the
    /// tables to it, and then the samples stop moving them.
    #[test]
    fn a_steady_pool_brings_the_tables_to_it() {
        let mut p = params();
        let mut online = Online::default();
        let truth = 1.5 * p.predicted_pool_ns(&[1]).unwrap();
        for _ in 0..200 {
            let predicted = p.predicted_pool_ns(&[1]).unwrap();
            let obs = Observation {
                busy_ns: truth,
                predicted_ns: predicted,
                device_wait_ns: 0,
            };
            assert_eq!(online.observe(&mut p, obs, 0.1), None);
        }
        let settled = p.predicted_pool_ns(&[1]).unwrap();
        assert!((settled / truth - 1.0).abs() < 1e-3, "{settled} vs {truth}");
    }

    /// A streak of late layers is reported once, at its LATE_STREAK-th layer,
    /// naming a slow pool when the pool ran over its tables and the handoff
    /// when it did not; one prompt layer ends the streak.
    #[test]
    fn a_late_streak_is_reported_once_with_its_cause() {
        let late = |busy: f32| Observation {
            busy_ns: busy,
            predicted_ns: 100.0,
            device_wait_ns: LATE_NS,
        };
        for (busy, cause) in [(200.0, LateCause::SlowPool), (100.0, LateCause::Handoff)] {
            let mut p = params();
            let mut online = Online::default();
            for i in 1..LATE_STREAK {
                assert_eq!(online.observe(&mut p, late(busy), 0.0), None, "layer {i}");
            }
            assert_eq!(online.observe(&mut p, late(busy), 0.0), Some(cause));
            assert_eq!(
                online.observe(&mut p, late(busy), 0.0),
                None,
                "once a streak"
            );
            let prompt = Observation {
                device_wait_ns: 0,
                ..late(busy)
            };
            assert_eq!(online.observe(&mut p, prompt, 0.0), None);
            for _ in 1..LATE_STREAK {
                assert_eq!(online.observe(&mut p, late(busy), 0.0), None);
            }
            assert_eq!(
                online.observe(&mut p, late(busy), 0.0),
                Some(cause),
                "a new streak"
            );
        }
    }
}
