//! How a `High` prompt and the `High` decodes beside it share the device.
//!
//! A dialogue turn that calls tools alternates the two: a prompt — the user's
//! message, then each tool's result — and a decode run that ends at the next
//! tool call. The device is busy either way, so how it is split does not
//! change how much work it does; it orders what finishes first. **The side with
//! less outstanding work goes first** — the smaller backlog, in seconds of its
//! own airtime, takes the larger share — which is the order that minimises the
//! mean time to completion: a short prompt beside a long decode run reaches
//! its first text in seconds instead of waiting the run out, and short decode
//! runs beside a deep read finish their segments and come back as prompts.
//!
//! **And a prompt that has waited rises to the front.** Its backlog is counted
//! down by its wait against its own expected time, so a prompt that has waited
//! as long as it would take alone holds the split's whole width toward it —
//! however short the decode runs that keep arriving beside it.
//!
//! This replaced, in turn, a fixed `R = 2` — decode owed twice the creep's
//! airtime — which held a 50K-token `High` backlog to a third of the device
//! (BlackWorld, a965e3a07, 2026-10-19: heavy-read decoded 221 s and prefilled
//! 87 s of 455), and then a split by backlog *size* that gave the larger side
//! the larger share: on WhiteWorld (e119fd7d7) a 1,768-row prompt beside one
//! long decode drained at 11 tok/s, its turn silent until it finished.
//!
//! `R` is decode's airtime for each second of the creep's. Decode rides every
//! step the creep rides, so a co-batched step of one decode-only step `d` plus
//! a slice `s` gives decode `d` and the creep `s`. The step loop takes `R` two
//! ways: when `R ≥ 1` the slice is `d` and decode is owed `(R − 1) d` of
//! decode-only steps after it ([`debt_ratio`]); when `R < 1` nothing is owed
//! and the slice is `d / R` ([`slice_steps`]). Either way decode holds
//! `R / (1 + R)` of the device — and never less than a token every
//! [`STREAM_GAP`].

use std::time::Duration;

/// The widest the split may go either way. Past it one side's backlog is noise
/// beside the other's, and the far side still advances once in this many.
const MAX_RATIO: f64 = 64.0;

/// Decode's airtime for each second of the creep's: the prompts' backlog —
/// `prompt_rows` at `creep_rows_per_s` of the creep's own airtime — over the
/// decodes' — one mean decode run of `segment_tokens` at `decode_step` a step,
/// which every decode advances through together — narrowed toward the prompts
/// by the most-due of `prompts`, each `(rows, waited)`. `None` while any input
/// is unmeasured or a side has nothing outstanding; the caller then keeps its
/// default.
///
/// **Each prompt is due by its own time, not the backlog's.** A prompt that
/// has waited its own rows' solo time is overdue, and holds the split at its
/// widest toward the prompts until it is served. Measured against the whole
/// backlog instead, the oldest of a steady stream is always old beside work it
/// is not alone in, and the split pinned at its widest for good: on WhiteWorld
/// (9d69e59c5, 2026-10-19) every `High` step of heavy-read ran at `1/64`, and
/// decode at the stream floor was every busy member's steady state.
pub(super) fn high_airtime_ratio(
    segment_tokens: f64,
    decode_step: Duration,
    prompt_rows: usize,
    creep_rows_per_s: f64,
    prompts: &[(usize, Duration)],
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
    // What is left of each prompt's own time once its wait is taken off — none
    // left, and it is due — and the most due of them.
    let due = prompts
        .iter()
        .filter(|&&(rows, _)| rows > 0)
        .map(|&(rows, waited)| {
            let own = rows as f64 / creep_rows_per_s;
            (1.0 - waited.as_secs_f64() / own).max(0.0)
        })
        .fold(1.0f64, f64::min);
    let ratio = prompt_s / decode_s;
    Some((ratio.clamp(1.0 / MAX_RATIO, MAX_RATIO) * due).max(1.0 / MAX_RATIO))
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
        self.add_value(tokens as f64);
    }

    /// Fold in one measurement that need not be whole.
    pub(super) fn add_value(&mut self, t: f64) {
        if !t.is_finite() {
            return;
        }
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

    /// No prompt has waited.
    const NOW: &[(usize, Duration)] = &[];

    /// The WhiteWorld case: a 1,768-row prompt at 102 rows a second beside a
    /// decode with 245-token runs at 200 ms — 17 s of prompt against 49 s of
    /// decode. The smaller backlog goes first: the prompt takes most of the
    /// device, where splitting by size gave the decode three quarters.
    #[test]
    fn a_short_prompt_beside_a_long_run_goes_first() {
        let r = high_airtime_ratio(245.0, Duration::from_millis(200), 1_768, 102.0, NOW)
            .expect("both sides measured");
        let want = (1_768.0 / 102.0) / (245.0 * 0.2);
        assert!((r - want).abs() < 1e-9, "R = {r}");
        assert!(1.0 / (1.0 + r) > 0.7, "the prompt's share at R = {r}");
    }

    /// Heavy-read's shape: 117-token runs at 50 ms beside a 50K-row backlog at
    /// 1,500 rows a second — 5.9 s of decode against 33 s of prompt. The decode
    /// runs finish first and come back as prompts.
    #[test]
    fn short_decode_runs_beside_a_deep_backlog_finish_first() {
        let r = high_airtime_ratio(117.0, Duration::from_millis(50), 50_000, 1_500.0, NOW)
            .expect("both sides measured");
        assert!(r > 1.0, "R = {r}");
        assert_eq!(slice_steps(r, Duration::from_millis(50)), 1.0);
    }

    /// **A prompt that has waited its own time is due, whatever the decodes
    /// hold.** A busy chat cluster can keep the decode backlog short
    /// indefinitely; the prompt's share still reaches the split's full width
    /// toward it once it has waited as long as it would take alone, and rises
    /// all the way there.
    #[test]
    fn a_waiting_prompt_reaches_the_front_within_its_own_time() {
        let (rows, rate) = (50_000usize, 1_500.0);
        let own = Duration::from_secs_f64(rows as f64 / rate);
        for (seg, step_ms) in [(1.0, 1u64), (20.0, 10), (117.0, 50), (2_000.0, 200)] {
            let step = Duration::from_millis(step_ms);
            let mut last = f64::INFINITY;
            for k in 0..=10 {
                let waited = own.mul_f64(k as f64 / 10.0);
                let r = high_airtime_ratio(seg, step, rows, rate, &[(rows, waited)]).unwrap();
                assert!(r <= last, "R rose as the prompt waited: {r} after {last}");
                last = r;
            }
            assert_eq!(
                last,
                1.0 / MAX_RATIO,
                "due at its own time beside {seg}-token runs"
            );
        }
    }

    /// **One old prompt holds the split only until it is served.** Behind it,
    /// a steady stream of fresh short prompts is due by its own times, not the
    /// old one's: with the old prompt in the backlog the split is at its widest
    /// toward the prompts, and with it gone the split is the backlogs' again —
    /// whatever the total backlog, however long the stream has been arriving.
    #[test]
    fn a_stream_of_fresh_prompts_does_not_pin_the_split() {
        let (rate, seg, step) = (1_000.0, 250.0, Duration::from_millis(100));
        // 25 s of decode run; ten fresh 2,000-row prompts, 2 s each alone.
        let fresh: Vec<(usize, Duration)> = (0..10)
            .map(|i| (2_000usize, Duration::from_millis(100 * i)))
            .collect();
        let old = (4_000usize, Duration::from_secs(60)); // 4 s alone, waited 60 s
        let mut with_old = fresh.clone();
        with_old.push(old);
        let total_with: usize = with_old.iter().map(|p| p.0).sum();
        let r = high_airtime_ratio(seg, step, total_with, rate, &with_old).unwrap();
        assert_eq!(r, 1.0 / MAX_RATIO, "the overdue prompt holds the front");
        let total: usize = fresh.iter().map(|p| p.0).sum();
        let r = high_airtime_ratio(seg, step, total, rate, &fresh).unwrap();
        let base = (total as f64 / rate) / (seg * step.as_secs_f64());
        assert!(
            r > 1.0 / MAX_RATIO,
            "the split pinned once the old prompt left: {r}"
        );
        // The most-due fresh prompt has waited 0.9 s of its own 2 s.
        let want = base * (1.0 - 0.9 / 2.0);
        assert!((r - want).abs() < 1e-9, "R = {r}, want {want}");
        // Waits short of each prompt's own time are not overdue, however many.
        let aged: Vec<(usize, Duration)> = fresh
            .iter()
            .map(|&(rows, _)| (rows, Duration::from_millis(1_500)))
            .collect();
        let r = high_airtime_ratio(seg, step, total, rate, &aged).unwrap();
        assert!(
            r > 1.0 / MAX_RATIO,
            "1.5 s waits on 2 s prompts are not overdue: {r}"
        );
    }

    /// **The clamp and the floor compose.** At the split's full width toward
    /// decode, a slice still rides once in each debt cycle, so a prompt beside
    /// long decode runs advances — and at full width toward the prompt the
    /// decodes still stream.
    #[test]
    fn neither_side_stops_at_the_split_s_widest() {
        let d = Duration::from_millis(50);
        // Toward decode: one step of slice, owed 63 decode-only steps.
        let slice = slice_steps(MAX_RATIO, d);
        let owed = debt_ratio(MAX_RATIO) * slice;
        assert_eq!(slice, 1.0);
        assert!(owed.is_finite() && owed > 0.0);
        let creep_share = slice / (1.0 + owed + slice);
        assert!(
            creep_share >= 1.0 / (MAX_RATIO + 2.0),
            "the creep's share {creep_share}"
        );
        // Toward the prompt: no debt, and a co-batched step inside the gap.
        let r = 1.0 / MAX_RATIO;
        assert_eq!(debt_ratio(r), 0.0);
        let step = d.as_secs_f64() * (1.0 + slice_steps(r, d));
        assert!(
            step <= STREAM_GAP.as_secs_f64() + 1e-9,
            "co-batched step {step} s"
        );
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

    /// Nothing measured, nothing outstanding: the caller keeps its default.
    #[test]
    fn an_unmeasured_side_gives_no_ratio() {
        let step = Duration::from_millis(50);
        assert_eq!(high_airtime_ratio(0.0, step, 1_000, 1_500.0, NOW), None);
        assert_eq!(
            high_airtime_ratio(100.0, Duration::ZERO, 1_000, 1_500.0, NOW),
            None
        );
        assert_eq!(high_airtime_ratio(100.0, step, 0, 1_500.0, NOW), None);
        assert_eq!(high_airtime_ratio(100.0, step, 1_000, 0.0, NOW), None);
    }

    /// The split stays inside its bounds however lopsided the backlogs.
    #[test]
    fn the_split_is_bounded_both_ways() {
        let step = Duration::from_millis(50);
        let toward_decode = high_airtime_ratio(1.0, step, 10_000_000, 100.0, NOW).unwrap();
        let toward_prompt = high_airtime_ratio(1e9, step, 1, 1e9, NOW).unwrap();
        assert_eq!(toward_decode, MAX_RATIO);
        assert_eq!(toward_prompt, 1.0 / MAX_RATIO);
        assert_eq!(
            slice_steps(toward_prompt, Duration::from_micros(1)),
            MAX_RATIO
        );
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
