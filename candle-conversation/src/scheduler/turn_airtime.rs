//! How a `High` prompt and the `High` decodes beside it share the device.
//!
//! A dialogue turn that calls tools alternates the two: a prompt — the user's
//! message, then each tool's result — and a decode run that ends at the next
//! tool call. Neither half finishes a turn alone, so the device is split to
//! keep both moving: each side gets airtime in proportion to the work it has
//! outstanding, in seconds of its own airtime, and the two backlogs drain
//! together. A side that falls behind holds more of the work and so draws more
//! of the device, which is how a waiting prompt rises.
//!
//! This replaced a fixed `R = 2` — decode owed twice the creep's airtime —
//! which held a 50K-token `High` backlog to a third of the device while the
//! decodes beside it ran one to five rows a step: on BlackWorld (a965e3a07,
//! 2026-10-19) heavy-read decoded 221 s and prefilled 87 s of 455, and its
//! prompts drained at 43–277 tok/s where a dedicated forward ran thousands.
//!
//! `R` is the decode backlog over the prompt backlog: decode's airtime for
//! each second of the creep's. Decode rides every step the creep rides, so a
//! co-batched step of one decode-only step `d` plus a slice `s` gives decode
//! `d` and the creep `s`. The step loop takes `R` two ways: when `R ≥ 1` the
//! slice is `d` and decode is owed `(R − 1) d` of decode-only steps after it
//! ([`debt_ratio`]); when `R < 1` nothing is owed and the slice is `d / R`
//! ([`slice_steps`]). Either way decode holds `R / (1 + R)` of the device.

use std::time::Duration;

/// The widest the split may go either way. Past it one side's backlog is noise
/// beside the other's, and the far side still advances once in this many.
const MAX_RATIO: f64 = 64.0;

/// Decode's airtime for each second of the creep's: the decodes' backlog —
/// one mean decode run of `segment_tokens` at `decode_step` a step, which every
/// decode advances through together — over the prompts' backlog — `prompt_rows`
/// at `creep_rows_per_s` of the creep's own airtime. `None` while any input is
/// unmeasured or a side has nothing outstanding; the caller then keeps its
/// default.
pub(super) fn high_airtime_ratio(
    segment_tokens: f64,
    decode_step: Duration,
    prompt_rows: usize,
    creep_rows_per_s: f64,
) -> Option<f64> {
    let decode_s = segment_tokens * decode_step.as_secs_f64();
    if decode_s.is_nan()
        || decode_s <= 0.0
        || creep_rows_per_s.is_nan()
        || creep_rows_per_s <= 0.0
        || prompt_rows == 0
    {
        return None;
    }
    let prompt_s = prompt_rows as f64 / creep_rows_per_s;
    Some((decode_s / prompt_s).clamp(1.0 / MAX_RATIO, MAX_RATIO))
}

/// The debt a co-batched step owes decode, per second of the creep's share of
/// it, for decode to hold `ratio` of the creep's airtime: the decode rows rode
/// that step too, so they already had one share of it, and only the rest is
/// owed in decode-only steps. Nothing when `ratio < 1`: the slice carries the
/// split then ([`slice_steps`]).
pub(super) fn debt_ratio(ratio: f64) -> f64 {
    (ratio - 1.0).max(0.0)
}

/// The longest a `High` decode may go between tokens and still read as a live
/// stream: about four tokens a second. A co-batched step is the decode rows'
/// step and the creep's slice together, so this bounds the slice, and with it
/// how far below 1 the split may push decode's share — whatever the backlogs
/// say, a chat beside a heavy read keeps streaming.
const STREAM_GAP: Duration = Duration::from_millis(250);

/// How long a slice of the creep may run beside the decode rows, as a multiple
/// of a decode-only step `decode_step`: one when decode is owed at least the
/// creep's time, and `1 / R` when it is owed less — a co-batched step then
/// carries the creep's share in the step itself, since the debt cannot go below
/// the steps the creep rides — but never so long that the step outlasts
/// [`STREAM_GAP`]. That cap floors decode's share at `d / STREAM_GAP`; a step
/// already past half the gap keeps the one step of slice it always had, since
/// less would leave the creep nothing to ride.
pub(super) fn slice_steps(ratio: f64, decode_step: Duration) -> f64 {
    if ratio >= 1.0 {
        return 1.0;
    }
    let d = decode_step.as_secs_f64();
    let streaming = if d > 0.0 {
        ((STREAM_GAP.as_secs_f64() - d) / d).max(1.0)
    } else {
        MAX_RATIO
    };
    (1.0 / ratio.max(1.0 / MAX_RATIO)).min(streaming)
}

/// A rate per second of airtime, decayed by the airtime itself: rows the creep
/// advanced over the seconds of device it took to advance them, so the figure
/// is the creep's speed when it runs and not how often it was let run.
#[derive(Debug, Default)]
pub(super) struct AirtimeRate {
    rows: f64,
    secs: f64,
}

impl AirtimeRate {
    /// Memory, in seconds of airtime.
    const TAU_SECS: f64 = 30.0;

    pub(super) fn add(&mut self, rows: usize, airtime: Duration) {
        let dt = airtime.as_secs_f64();
        if dt.is_nan() || dt <= 0.0 {
            return;
        }
        let keep = (-dt / Self::TAU_SECS).exp();
        self.rows = self.rows * keep + rows as f64;
        self.secs = self.secs * keep + dt;
    }

    pub(super) fn per_s(&self) -> Option<f64> {
        (self.secs > 0.0 && self.rows > 0.0).then(|| self.rows / self.secs)
    }
}

/// The mean decode run of a `High` turn segment, in tokens: what a decode
/// generates between its prompt and the tool call or end that closes it.
#[derive(Debug, Default)]
pub(super) struct SegmentMean {
    mean: Option<f64>,
}

impl SegmentMean {
    /// The newest run's weight.
    const WEIGHT: f64 = 0.1;

    pub(super) fn add(&mut self, tokens: usize) {
        let t = tokens as f64;
        self.mean = Some(match self.mean {
            None => t,
            Some(m) => m + Self::WEIGHT * (t - m),
        });
    }

    pub(super) fn tokens(&self) -> Option<f64> {
        self.mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Heavy-read's shape: a 50K-row backlog at 1,500 rows a second of creep
    /// beside decodes running 117-token segments at 50 ms — the decodes hold
    /// 5.9 s of work against the prompts' 33 s, so the creep takes most of the
    /// device, where the old constant gave it a third.
    #[test]
    fn a_deep_prompt_backlog_takes_most_of_the_device() {
        let r = high_airtime_ratio(117.0, Duration::from_millis(50), 50_000, 1_500.0)
            .expect("both sides measured");
        assert!((r - 5.85 / (50_000.0 / 1_500.0)).abs() < 1e-9, "R = {r}");
        let creep_share = 1.0 / (1.0 + r);
        assert!(creep_share > 0.8, "the creep's share {creep_share}");
        // 5.7 decode steps of slice by the backlogs, but a 50 ms step and its
        // slice may not outlast the stream gap: four steps of slice, 80%.
        assert!((slice_steps(r, Duration::from_millis(50)) - 4.0).abs() < 1e-9);
    }

    /// However `R` falls, a co-batched step and the decode-only steps it owes
    /// give decode `R` seconds for each of the creep's.
    #[test]
    fn the_step_loop_holds_the_split_it_was_given() {
        for r in [1.0 / 64.0, 0.2, 0.5, 1.0, 2.0, 4.0, 64.0] {
            let d = 1.0; // a decode-only step
            let slice = slice_steps(r, Duration::from_micros(1)) * d;
            let owed = debt_ratio(r) * slice;
            let decode = d + owed;
            assert!(
                (decode / slice - r).abs() < 1e-9,
                "at R = {r}: {decode} against {slice}"
            );
        }
    }

    /// A chat beside a heavy read keeps streaming: however small `R`, a
    /// co-batched step — the decode step and its slice — stays inside the
    /// stream gap — and a step past half the gap carries no more than itself.
    #[test]
    fn a_decode_beside_a_deep_backlog_keeps_streaming() {
        for d_ms in [10u64, 50, 120, 125] {
            let d = Duration::from_millis(d_ms);
            let step = d.as_secs_f64() * (1.0 + slice_steps(1.0 / MAX_RATIO, d));
            assert!(
                step <= STREAM_GAP.as_secs_f64() + 1e-9,
                "a {d_ms} ms decode step ran {step} s co-batched"
            );
        }
        // A step past half the gap keeps the slice it always had: one step.
        let slow = Duration::from_millis(200);
        assert_eq!(
            slice_steps(1.0 / MAX_RATIO, slow),
            1.0,
            "past half the gap: one step of slice"
        );
    }

    /// A short prompt beside long decode runs yields to them.
    #[test]
    fn a_short_prompt_beside_long_runs_yields() {
        let r = high_airtime_ratio(2_000.0, Duration::from_millis(60), 512, 1_500.0)
            .expect("both sides measured");
        assert!(r > 1.0, "R = {r}");
        assert_eq!(slice_steps(r, Duration::from_millis(60)), 1.0);
    }

    /// Nothing measured, nothing outstanding: the caller keeps its default.
    #[test]
    fn an_unmeasured_side_gives_no_ratio() {
        let step = Duration::from_millis(50);
        assert_eq!(high_airtime_ratio(0.0, step, 1_000, 1_500.0), None);
        assert_eq!(
            high_airtime_ratio(100.0, Duration::ZERO, 1_000, 1_500.0),
            None
        );
        assert_eq!(high_airtime_ratio(100.0, step, 0, 1_500.0), None);
        assert_eq!(high_airtime_ratio(100.0, step, 1_000, 0.0), None);
    }

    /// The split stays inside its bounds however lopsided the backlogs.
    #[test]
    fn the_split_is_bounded_both_ways() {
        let step = Duration::from_millis(50);
        let tiny = high_airtime_ratio(1.0, step, 10_000_000, 100.0).unwrap();
        let huge = high_airtime_ratio(1e9, step, 1, 1e9).unwrap();
        assert_eq!(tiny, 1.0 / MAX_RATIO);
        assert_eq!(huge, MAX_RATIO);
        assert_eq!(slice_steps(tiny, Duration::from_micros(1)), MAX_RATIO);
    }

    /// The creep's rate is its speed per second it ran, whatever the gaps.
    #[test]
    fn the_creep_rate_is_per_second_of_its_own_airtime() {
        let mut a = AirtimeRate::default();
        for _ in 0..20 {
            a.add(150, Duration::from_millis(100));
        }
        let r = a.per_s().unwrap();
        assert!((r - 1_500.0).abs() < 1.0, "read {r}");
        assert_eq!(AirtimeRate::default().per_s(), None);
    }

    #[test]
    fn the_segment_mean_follows_its_runs() {
        let mut m = SegmentMean::default();
        assert_eq!(m.tokens(), None);
        m.add(100);
        assert_eq!(m.tokens(), Some(100.0));
        for _ in 0..50 {
            m.add(200);
        }
        assert!((m.tokens().unwrap() - 200.0).abs() < 1.0);
    }
}
