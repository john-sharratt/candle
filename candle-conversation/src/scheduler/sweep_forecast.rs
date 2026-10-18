//! What a creep cohort's sweep will lose to others' KV before its last window.
//!
//! A creep cohort's width is fixed when it forms, and its transient tier is
//! placed again at every window of its sweep — minutes later at decode airtime,
//! on ground nothing holds for it in between. The decodes running beside it
//! keep claiming KV, and each claim buys from the same ground: the KV side's
//! free regions and what the weight side will concede above its floor. Priced
//! against the ground at formation, a width can fit every early window and miss
//! a late one by a region. Measured on the RTX 3090 (Flash-Next, 2026-10-18):
//! an 8,192-row cohort formed on ~10 GB of ground; decode KV grew 3.6 → 7.2 GB
//! over its two-minute sweep, the weight side reached its floor, and window 23
//! found 453 regions for a 454-region tier — its members failed with the file
//! ingests they carried.
//!
//! So formation charges the cohort the KV the decodes will add while it sweeps:
//! their measured rate of KV rows, times how long a sweep has measured to take.
//! Both are observations, not a model of airtime: the rate from the decodes'
//! own offsets as they advance, and the duration from formation to the head.
//! With neither measured yet, nothing is withheld — the width is what it was.

use std::collections::HashMap;
use std::time::Instant;

/// Time constant of the decode rate's average, in seconds: long enough to ride
/// over one bursty step, short enough to follow a cohort of decodes arriving.
const RATE_TAU_SECS: f64 = 10.0;
/// A rate sample spans at least this long, so a millisecond step does not
/// weigh in as a rate of thousands of rows a second.
const MIN_SAMPLE_SECS: f64 = 0.5;
/// Weight of the newest sweep in the duration's average.
const SWEEP_WEIGHT: f64 = 0.5;

/// See the module documentation.
#[derive(Debug, Default)]
pub(super) struct SweepForecast {
    /// Each decoding sequence's offset when last sampled.
    offsets: HashMap<usize, usize>,
    /// When the sample in progress began, and the rows it has counted.
    window: Option<(Instant, u64)>,
    /// KV rows the decodes add per second, averaged.
    rows_per_sec: Option<f64>,
    /// When the sweep in flight formed.
    sweep_started: Option<Instant>,
    /// How long a sweep takes, formation to head, averaged.
    sweep_secs: Option<f64>,
}

impl SweepForecast {
    /// The decodes' offsets now. A sequence's growth since its last sample is
    /// KV it wrote; one seen for the first time, or one that went backwards (a
    /// rewind, a fresh slot under a reused id), adds nothing — its offset is
    /// where it stands, not what it just wrote.
    pub(super) fn sample_decodes(
        &mut self,
        now: Instant,
        offsets: impl IntoIterator<Item = (usize, usize)>,
    ) {
        let mut grown = 0u64;
        let mut next = HashMap::with_capacity(self.offsets.len());
        for (seq, at) in offsets {
            if let Some(&was) = self.offsets.get(&seq) {
                grown += at.saturating_sub(was) as u64;
            }
            next.insert(seq, at);
        }
        self.offsets = next;
        let (start, rows) = self.window.get_or_insert((now, 0));
        *rows += grown;
        let secs = now.duration_since(*start).as_secs_f64();
        if secs < MIN_SAMPLE_SECS {
            return;
        }
        let rate = *rows as f64 / secs;
        let w = 1.0 - (-secs / RATE_TAU_SECS).exp();
        self.rows_per_sec = Some(match self.rows_per_sec {
            Some(r) => r + w * (rate - r),
            None => rate,
        });
        self.window = Some((now, 0));
    }

    /// A cohort formed: its sweep starts now.
    pub(super) fn sweep_formed(&mut self, now: Instant) {
        self.sweep_started = Some(now);
    }

    /// The sweep in flight reached the head: its duration joins the average.
    pub(super) fn sweep_done(&mut self, now: Instant) {
        let Some(start) = self.sweep_started.take() else {
            return;
        };
        let secs = now.duration_since(start).as_secs_f64();
        self.sweep_secs = Some(match self.sweep_secs {
            Some(s) => s + SWEEP_WEIGHT * (secs - s),
            None => secs,
        });
    }

    /// The sweep in flight ended short of the head — dropped, failed or set
    /// aside — and measured nothing.
    pub(super) fn sweep_abandoned(&mut self) {
        self.sweep_started = None;
    }

    /// KV rows the decodes will add over one sweep, as measured so far: zero
    /// until both the rate and a sweep's duration have been observed.
    pub(super) fn withheld_rows(&self) -> usize {
        match (self.rows_per_sec, self.sweep_secs) {
            (Some(rate), Some(secs)) => (rate * secs).ceil().max(0.0) as usize,
            _ => 0,
        }
    }

    /// [`Self::withheld_rows`] in bytes, at `kv_bytes_per_row`.
    pub(super) fn withheld_bytes(&self, kv_bytes_per_row: usize) -> usize {
        self.withheld_rows().saturating_mul(kv_bytes_per_row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn at(t0: Instant, secs: f64) -> Instant {
        t0 + Duration::from_secs_f64(secs)
    }

    /// Nothing is withheld until a rate and a sweep have both been measured:
    /// a cohort formed before either is sized as it always was.
    #[test]
    fn nothing_is_withheld_before_both_are_measured() {
        let t0 = Instant::now();
        let mut f = SweepForecast::default();
        assert_eq!(f.withheld_rows(), 0);
        f.sample_decodes(t0, [(1, 100)]);
        f.sample_decodes(at(t0, 1.0), [(1, 140)]);
        assert_eq!(f.withheld_rows(), 0, "a rate alone withholds nothing");
        let mut g = SweepForecast::default();
        g.sweep_formed(t0);
        g.sweep_done(at(t0, 120.0));
        assert_eq!(g.withheld_rows(), 0, "a duration alone withholds nothing");
    }

    /// The forecast is the decodes' rate times a sweep's duration: two
    /// sequences adding 20 rows a second each, over a 120 s sweep.
    #[test]
    fn the_forecast_is_the_decode_rate_over_a_sweep() {
        let t0 = Instant::now();
        let mut f = SweepForecast::default();
        f.sample_decodes(t0, [(1, 1_000), (2, 5_000)]);
        f.sample_decodes(at(t0, 1.0), [(1, 1_020), (2, 5_020)]);
        f.sweep_formed(t0);
        f.sweep_done(at(t0, 120.0));
        assert_eq!(f.withheld_rows(), 40 * 120);
        assert_eq!(f.withheld_bytes(24_576), 40 * 120 * 24_576);
    }

    /// A sequence seen for the first time, and one that went backwards, add no
    /// rows: their offsets say where they stand, not what they just wrote.
    #[test]
    fn arrivals_and_rewinds_write_nothing() {
        let t0 = Instant::now();
        let mut f = SweepForecast::default();
        f.sample_decodes(t0, [(1, 1_000)]);
        // Sequence 2 arrives 90k deep; sequence 1 rewinds a rejected draft.
        f.sample_decodes(at(t0, 1.0), [(1, 996), (2, 90_000)]);
        f.sweep_formed(t0);
        f.sweep_done(at(t0, 10.0));
        assert_eq!(f.withheld_rows(), 0);
    }

    /// Steps shorter than the sample span accumulate into one sample rather
    /// than each weighing in as a burst.
    #[test]
    fn short_steps_accumulate_into_one_sample() {
        let t0 = Instant::now();
        let mut f = SweepForecast::default();
        f.sample_decodes(t0, [(1, 0)]);
        for k in 1..=10 {
            f.sample_decodes(at(t0, 0.1 * k as f64), [(1, 2 * k)]);
        }
        // 20 rows over 1.0 s, measured once the span passed 0.5 s.
        let rate = f.rows_per_sec.expect("a sample was taken");
        assert!((rate - 20.0).abs() < 1e-6, "rate {rate}");
    }

    /// A sweep that never reached the head measures nothing, and the average
    /// follows the sweeps that did.
    #[test]
    fn only_finished_sweeps_are_timed() {
        let t0 = Instant::now();
        let mut f = SweepForecast::default();
        f.sweep_formed(t0);
        f.sweep_abandoned();
        f.sweep_done(at(t0, 500.0));
        assert_eq!(f.sweep_secs, None);
        f.sweep_formed(t0);
        f.sweep_done(at(t0, 100.0));
        f.sweep_formed(at(t0, 100.0));
        f.sweep_done(at(t0, 300.0));
        assert_eq!(f.sweep_secs, Some(150.0));
    }
}
