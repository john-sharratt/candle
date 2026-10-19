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

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Time constant of the decode rate's average, in seconds: long enough to ride
/// over one bursty step, short enough to follow a cohort of decodes arriving.
const RATE_TAU_SECS: f64 = 10.0;
/// A rate sample spans at least this long, so a millisecond step does not
/// weigh in as a rate of thousands of rows a second.
const MIN_SAMPLE_SECS: f64 = 0.5;
/// Weight of the newest sweep in the duration's average.
const SWEEP_WEIGHT: f64 = 0.5;
/// Sweeps a forecast of what a sweep loses takes its peak over.
///
/// **The peak, not the mean, because the constraint is hard.** A window that
/// finds its ground a region short loses every member of its cohort, so a
/// growth priced at its mean is exceeded on about half the sweeps it matters
/// for: on WhiteWorld (f3bf5e313, 2026-10-19) cohorts fitted with 40–130 MiB
/// to spare were lost two to five regions short, each to a sweep that ran
/// longer or took more held ground than the average said. A few sweeps show
/// the load's spread, and few enough follow it when it moves.
const RECENT_SWEEPS: usize = 4;

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
    /// Held ground's arena bytes when the sweep in flight formed.
    held_at_formation: Option<u64>,
    /// The most recent sweeps' durations and held-ground growths, newest last
    /// — what the forecasts take their peak over ([`RECENT_SWEEPS`]).
    recent_secs: VecDeque<f64>,
    recent_held: VecDeque<u64>,
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
        push_recent(&mut self.recent_secs, secs);
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

    /// Held ground's arena bytes as the sweep in flight forms.
    pub(super) fn held_formed(&mut self, held_bytes: u64) {
        self.held_at_formation = Some(held_bytes);
    }

    /// Held ground's arena bytes as the sweep in flight ends — at the head, or
    /// short of it, whose growth up to its end is the growth that ended it —
    /// against what they were at formation, joining the average.
    pub(super) fn held_ended(&mut self, held_bytes: u64) {
        let Some(at) = self.held_at_formation.take() else {
            return;
        };
        push_recent(&mut self.recent_held, held_bytes.saturating_sub(at));
    }

    /// Held ground's growth over one sweep, as measured so far: zero until a
    /// sweep has ended. Held ground provisions between every window of a sweep
    /// (`provision_held`), so what it takes over the whole sweep is what a
    /// cohort must leave room for, not only its next pass: an ingest cohort
    /// fitted with 17 MiB to spare met held ground grown seven regions by its
    /// last window, and was refused one short (WhiteWorld, f3bf5e313,
    /// 2026-10-19).
    pub(super) fn held_growth_bytes(&self) -> u64 {
        self.recent_held.iter().copied().max().unwrap_or(0)
    }

    /// A sweep's measured duration, in seconds — `None` until one has run.
    pub(super) fn sweep_secs(&self) -> Option<f64> {
        self.sweep_secs
    }

    /// KV rows the decodes will add over one sweep, as measured so far: zero
    /// until both the rate and a sweep's duration have been observed.
    pub(super) fn withheld_rows(&self) -> usize {
        // Over the longest recent sweep: see `RECENT_SWEEPS`.
        let longest = self.recent_secs.iter().copied().reduce(f64::max);
        match (self.rows_per_sec, longest) {
            (Some(rate), Some(secs)) => (rate * secs).ceil().max(0.0) as usize,
            _ => 0,
        }
    }

    /// [`Self::withheld_rows`] in bytes, at `kv_bytes_per_row`.
    pub(super) fn withheld_bytes(&self, kv_bytes_per_row: usize) -> usize {
        self.withheld_rows().saturating_mul(kv_bytes_per_row)
    }
}

/// `value` as the newest of a window of at most [`RECENT_SWEEPS`].
fn push_recent<T>(window: &mut VecDeque<T>, value: T) {
    if window.len() == RECENT_SWEEPS {
        window.pop_front();
    }
    window.push_back(value);
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

    /// The forecast runs over the longest of the recent sweeps, not their
    /// average, and forgets a sweep once [`RECENT_SWEEPS`] newer ones have run.
    #[test]
    fn the_forecast_runs_over_the_longest_recent_sweep() {
        let t0 = Instant::now();
        let mut f = SweepForecast::default();
        f.sample_decodes(t0, [(1, 0)]);
        f.sample_decodes(at(t0, 1.0), [(1, 10)]);
        let sweep = |f: &mut SweepForecast, secs: f64| {
            f.sweep_formed(t0);
            f.sweep_done(at(t0, secs));
        };
        sweep(&mut f, 120.0);
        sweep(&mut f, 10.0);
        assert_eq!(
            f.withheld_rows(),
            10 * 120,
            "the long sweep is the forecast"
        );
        for _ in 0..RECENT_SWEEPS {
            sweep(&mut f, 10.0);
        }
        assert_eq!(
            f.withheld_rows(),
            10 * 10,
            "past the window it is forgotten"
        );
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

    /// Held ground's growth over a sweep is measured from formation to the
    /// sweep's end, a short end included, and priced at its recent peak; a
    /// sweep over which it shrank grew by nothing, and does not lower the peak.
    #[test]
    fn held_growth_is_what_held_ground_took_over_a_sweep() {
        let mut f = SweepForecast::default();
        assert_eq!(
            f.held_growth_bytes(),
            0,
            "nothing measured, nothing withheld"
        );
        f.held_formed(100);
        f.held_ended(212);
        assert_eq!(f.held_growth_bytes(), 112);
        f.held_formed(300);
        f.held_ended(200);
        assert_eq!(
            f.held_growth_bytes(),
            112,
            "a shrinking sweep leaves the peak"
        );
        f.held_ended(999);
        assert_eq!(
            f.held_growth_bytes(),
            112,
            "an end with no formation measures nothing"
        );
    }
}
