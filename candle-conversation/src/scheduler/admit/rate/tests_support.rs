//! Holds the shared fixtures and constructors the planner's test modules build from.
//!
//! Link rates for the 4090 Mobile, the 3090 and the Blackwell workstation, the
//! residency of a measured run, and the exponent that reproduces it are defined once
//! here. Helpers build a planner that is fitted, learned or judging, and offer
//! admissions with a stated residency before and after, so each test states the
//! figures it is about and nothing else.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(in super::super) const GIB: u64 = 1 << 30;

    pub(in super::super) const MIB: u64 = 1 << 20;

    /// PCIe 4.0 ×16 as the 4090 Mobile measures it.
    pub(in super::super) const LINK_4090_MOBILE: f64 = 25e9;

    /// The 3090 box: a PCIe 3.0 host at half that.
    pub(in super::super) const LINK_3090: f64 = 12e9;

    /// The Blackwell workstation on PCIe 5.0 ×16.
    pub(in super::super) const LINK_BLACKWELL: f64 = 60e9;

    /// The floor the fill defends on the 16 GB card: hold 4.7 GiB + 512 MiB.
    pub(in super::super) const FLOOR_16GB: u64 = 5_232 * MIB;

    /// Run 15's residency in calibration.
    pub(in super::super) const RESIDENT_RUN15: u64 = 7_611 * MIB;

    /// The effective copy rate run 15 implies once the tokens' compute is taken
    /// off the forward: 12.31 GB non-resident over
    /// `1,004 ms − 474 × 0.55 ms` = 743 ms.
    ///
    /// Re-derived when `c` was measured at 0.55 ms rather than 0.42 (see
    /// [`RateModel::SEED_COMPUTE_SECS_PER_TOKEN`]): a larger compute term
    /// leaves less of the forward for the copy, so the same measurement implies
    /// a *faster* bus. It also brought the two independent run-15 forwards into
    /// much closer agreement — 16.56 against 16.48 GB/s, 0.5% apart, where at
    /// 0.42 ms they were 2% apart. Two forwards of different widths agreeing
    /// more closely under the corrected slope is the strongest evidence that
    /// the slope is the thing that was wrong.
    pub(in super::super) const BW_RUN15: f64 = 15.3e9;

    /// The engine's widest prefill and widest decode set.
    pub(in super::super) const CAP: usize = 8_192;

    pub(in super::super) const DECODE_CAP: usize = 64;

    /// What the caller would report a decode admission dislodging on this
    /// model: its 160 MiB recurrent store and a few K/V regions.
    pub(in super::super) const DECODE_DISLODGE: u64 = 176 * MIB;

    pub(in super::super) const SLOT: f64 = 1_933_312.0;

    pub(in super::super) const LAYERS: f64 = 41.0;

    /// A planner whose decode fit has learned a layer at `a` seconds plus `b`
    /// per missed distinct expert, over one to thirty-two decodes and a spread
    /// of residencies — as `a_decode_fit_extrapolates_by_distinct_experts_not_picks`
    /// teaches it — reset to `resident` with `decodes` running.
    pub(in super::super) fn planner_fitted(
        a: f64,
        b: f64,
        resident: u64,
        decodes: usize,
    ) -> WaveRate {
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        let mut p = planner(LINK_4090_MOBILE);
        for round in 0..40 {
            for d in (1..=32usize).step_by(3) {
                let share = 0.1 + 0.1 * ((round + d) % 9) as f64;
                let at = (total as f64 * share) as u64;
                let m = p.distinct_routed(d * 8) * p.streamed_fraction(at);
                p.observe_decode(d, 0, at, (a + b * m) * LAYERS);
            }
        }
        assert!(p.decode_learned());
        p.reset(resident, GIB, GIB, CAP, DECODE_CAP);
        for _ in 0..decodes {
            p.charge(decode(0), resident);
        }
        p
    }

    /// The 35B's experts at `share` of their whole.
    pub(in super::super) fn share_of_experts(share: f64) -> u64 {
        (ExpertGeometry::QWEN36_35B_A3B.total_bytes() as f64 * share) as u64
    }

    /// The test planner. Its hit exponent puts the curve through the hit the
    /// old seed (`0.7 × resident fraction`) gave at run 15's residency, so the
    /// decode figures worked out there by hand still hold; the deployed seed
    /// is [`DecodeModel::default`]'s.
    pub(in super::super) fn planner(link: f64) -> WaveRate {
        WaveRate::with_link_rate(
            link,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel {
                hit_exponent: run15_hit_exponent(),
                ..DecodeModel::default()
            },
        )
    }

    /// The exponent through a 0.7 × 39.3% hit at run 15's 39.3% residency.
    pub(in super::super) fn run15_hit_exponent() -> f64 {
        let f = RESIDENT_RUN15 as f64 / ExpertGeometry::QWEN36_35B_A3B.total_bytes() as f64;
        (0.7 * f).ln() / f.ln()
    }

    /// The planner as run 15 measured it: effective rate learned, not seeded.
    ///
    /// Taught with a forward wide enough to route across the whole layer
    /// ([`WaveRate::min_learn_rows`]), because a narrower one copies a fraction
    /// of the experts the model divides by and is refused.
    pub(in super::super) fn planner_learned(link: f64, bw: f64) -> WaveRate {
        let mut p = planner(link).with_alpha(1.0);
        let bytes = p.non_resident_bytes(RESIDENT_RUN15) as f64;
        let rows = p.min_learn_rows();
        let secs = bytes / bw + rows as f64 * RateModel::SEED_COMPUTE_SECS_PER_TOKEN;
        p.observe_prefill(rows, RESIDENT_RUN15, secs)
            .expect("a wide enough forward");
        assert!(approx(p.effective_bytes_per_s(), bw, 1e-9));
        p.with_alpha(WaveRate::DEFAULT_ALPHA)
    }

    /// The same planner with the layer time learned to `layer_secs` exactly:
    /// one plain decode at run 15's residency, compute-bound, so the step is
    /// the layer time over every MoE layer.
    pub(in super::super) fn planner_with_layer(link: f64, bw: f64, layer_secs: f64) -> WaveRate {
        let mut p = planner_learned(link, bw).with_alpha(1.0);
        assert!(p.decode_copy_secs(8, RESIDENT_RUN15) < layer_secs);
        p.observe_decode(1, 0, RESIDENT_RUN15, layer_secs * LAYERS);
        assert!(approx(p.layer_secs(), layer_secs, 1e-9));
        p.with_alpha(WaveRate::DEFAULT_ALPHA)
    }

    pub(in super::super) fn approx(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-12)
    }

    pub(in super::super) fn prefill(tokens: usize) -> Admission {
        Admission::Prefill { tokens }
    }

    pub(in super::super) fn decode(draft: usize) -> Admission {
        Admission::Decode { draft }
    }

    /// Offer `a` as the caller would: weights before are the wave's now,
    /// weights after are those less what this admission dislodges.
    pub(in super::super) fn offer(p: &mut WaveRate, a: Admission, dislodge: u64) -> Admit {
        let before = p.resident_now();
        p.try_admit(a, before, before.saturating_sub(dislodge))
    }

    /// Offer `a` with the residency stated outright, before and after, rather than
    /// derived from a dislodge. What the floor cases need: they are about the exact
    /// side of a floor the result lands on, and computing that from a subtraction
    /// puts arithmetic between the test and the line it is testing.
    pub(in super::super) fn offer_at(
        p: &mut WaveRate,
        a: Admission,
        before: u64,
        after: u64,
    ) -> Admit {
        p.try_admit(a, before, after)
    }

    /// A planner whose layer time is both **set** and **learned**: past
    /// [`WaveRate::MIN_DECODE_SAMPLES`], so `judge_promotion` actually judges
    /// rather than taking the unlearned-estimate bypass.
    pub(in super::super) fn planner_judging(layer_secs: f64) -> WaveRate {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, layer_secs);
        while p.decode_samples() < WaveRate::MIN_DECODE_SAMPLES {
            p.observe_decode(1, 0, RESIDENT_RUN15, layer_secs * LAYERS);
        }
        assert!(
            approx(p.layer_secs(), layer_secs, 1e-9),
            "still at the target"
        );
        p
    }

    /// Offer a finished prefill for promotion the way the scheduler does: the
    /// slot is already resident, so `before` is residency with its ground added
    /// back and `after` is residency as it stands.
    pub(in super::super) fn promote(p: &mut WaveRate, draft: usize, held: u64) -> Admit {
        let after = p.resident_now();
        p.judge_promotion(draft, after.saturating_add(held), after)
    }

    /// Offer `a` with `dislodge` until refused; the count admitted and the
    /// refusal.
    pub(in super::super) fn fill_with(
        p: &mut WaveRate,
        a: Admission,
        dislodge: u64,
    ) -> (usize, Refusal) {
        let mut n = 0;
        loop {
            match offer(p, a, dislodge) {
                Admit::Admitted { .. } => n += 1,
                Admit::Refused(r) => return (n, r),
            }
        }
    }

    /// A 250-token exemplar's tier at about a MiB a row.
    pub(in super::super) const EXEMPLAR: usize = 250;

    pub(in super::super) const EXEMPLAR_DISLODGE: u64 = 250 * MIB;
}
