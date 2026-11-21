//! Learns the copy rate, the layer time and the hit exponent from the forwards the
//! engine runs.
//!
//! Each estimate moves by the dampening rather than being latched, so one fast or
//! stalled forward cannot become the permanent truth, and the link rate only rises
//! on sustained evidence. A prefill forward teaches the copy rate only when it is
//! wide enough to route across the whole layer, because a narrower one divides by
//! experts it never touched. A decode forward teaches the layer time only when it
//! was compute-bound, since a bus-bound step says nothing about the layer, and the
//! hit exponent is learned from the expert cache's own hit counters.

use super::*;

impl WaveRate {
    /// Decode forwards the layer time must have seen before a promotion may be
    /// refused on it — see [`Self::judge_promotion`].
    ///
    /// Eight, matching [`CostFit::MIN_SAMPLES`]: enough for the dampened
    /// estimate to have travelled most of the way from its seed to the measured
    /// value (run 33 read 0.00394 at three samples against a converged 0.027),
    /// and few enough that the window costs a handful of waves rather than a
    /// phase.
    pub const MIN_DECODE_SAMPLES: u64 = 8;

    /// Bytes each run of the link probe copies (`candle::cuda_backend::link_probe`).
    pub const PROBE_BYTES: usize = link_probe::PROBE_BYTES;

    /// The estimator never believes a copy rate under this fraction of the
    /// link: below it the observation is measuring something other than the
    /// copy (a stall, a sync, a wave that carried a park), and it is refused.
    pub(super) const MIN_LINK_FRACTION: f64 = 0.02;

    /// Learn what the wave loop costs per forward, from the loop's own clock.
    ///
    /// `secs` is the wall clock a wave spent **outside** its forwards, divided by the
    /// forwards it ran — see [`RateModel::overhead_secs`] for why the model cannot get
    /// this from a forward's own duration and why omitting it made every wave too narrow.
    ///
    /// Dampened by `alpha` like every other estimate, and ignored when it is not a finite
    /// positive number: a wave that ran no forwards has no per-forward cost to report, and
    /// dividing by its zero would poison the term permanently.
    pub fn observe_overhead(&mut self, secs: f64) {
        if !secs.is_finite() || secs <= 0.0 {
            return;
        }
        self.model.overhead_secs += self.alpha * (secs - self.model.overhead_secs);
    }

    /// Prefill forwards the copy rate has learned from.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Decode forwards the layer time has learned from.
    pub fn decode_samples(&self) -> u64 {
        self.decode_samples
    }

    /// The narrowest forward whose copy is worth learning from.
    ///
    /// **The whole prefill model rests on "a forward needs every expert", and
    /// that is a statement about width.** A wave of `W` rows leaves one of a
    /// layer's experts untouched with probability about
    /// `e^(−W · experts_per_token / experts_per_layer)`, so a *narrow* forward
    /// routes to a fraction of the layer and copies a fraction of the bytes —
    /// while [`Self::observe_prefill`] divides the time by *all* of them. The
    /// implied rate is then far too fast, and on this engine that is not
    /// hypothetical: a 26-row section forward read as 119 GB/s on a 25 GB/s
    /// link, and the estimate it poisoned prices every wave after it.
    ///
    /// This is the width at which the untouched share falls under 2% — about
    /// 125 rows on the 35B's 8-of-256 routing — derived from the geometry so it
    /// follows the model rather than being a constant to re-derive per card.
    pub fn min_learn_rows(&self) -> usize {
        let per = self.decode.experts_per_token.max(1) as f64;
        let ratio = self.geometry.experts_per_layer as f64 / per;
        (ratio * 50f64.ln()).ceil() as usize
    }

    // ── the feedback ───────────────────────────────────────────────────────

    /// Learn from one prefill forward: `tokens` rows ran in `forward_secs`
    /// with `resident_bytes` of experts on the card.
    ///
    /// The copy is what is left of the forward after the fixed cost and the
    /// tokens' compute, and the observed rate is the non-resident bytes over
    /// it, folded into the estimate by `alpha`. Returns the observed rate, or
    /// `None` when the forward taught nothing: no bytes to copy (a fully
    /// resident card), a forward shorter than its own compute, or a duration
    /// that is not a number. One under [`Self::MIN_LINK_FRACTION`] of the link
    /// is refused as a stall, not a rate.
    ///
    /// **A forward that beats the link raises the link, if it moved more bytes
    /// than the probe did.** The probe is a few transfers of one size, and a
    /// reading can still land low: taken at scheduler start, while the expert
    /// cache staged its resident set across the same bus, it read 12.03 GB/s on
    /// a PCIe 4.0 ×16 link whose forwards then sustained 15.0. Clamping to such a
    /// reading would pin the estimate below the bus the engine actually has,
    /// permanently, on the strength of the worse measurement. A forward copying at least [`Self::PROBE_BYTES`] is better
    /// evidence about the link than the probe is — it is the same transfer, at
    /// size, under the real workload — so it replaces it; anything smaller is
    /// still clamped, because a rate divided out of a short copy is mostly the
    /// error in the compute model.
    #[cfg(test)]
    pub fn observe_prefill(
        &mut self,
        tokens: usize,
        resident_bytes: u64,
        forward_secs: f64,
    ) -> Option<f64> {
        self.observe_prefill_attending(tokens, 0.0, resident_bytes, forward_secs)
    }

    /// [`Self::observe_prefill`] for a forward whose rows attended `pairs`
    /// query×key pairs between them — each row times the context it attends —
    /// so the fit can price depth apart from width
    /// ([`RateModel::attend_secs_per_pair`]).
    pub fn observe_prefill_attending(
        &mut self,
        tokens: usize,
        pairs: f64,
        resident_bytes: u64,
        forward_secs: f64,
    ) -> Option<f64> {
        if !forward_secs.is_finite() || forward_secs <= 0.0 {
            return None;
        }
        // A forward too narrow to route across the whole layer copied a
        // fraction of the experts this would divide by — see
        // [`Self::min_learn_rows`].
        if tokens < self.min_learn_rows() {
            return None;
        }
        let bytes = self.non_resident_bytes(resident_bytes);
        // The depth the learned-from rows attended, for callers that name none.
        let context = pairs / tokens.max(1) as f64;
        self.context_per_row += self.alpha * (context - self.context_per_row);
        // **Every unknown comes out of the same forwards.** A forward is
        // `T = f + X/bw + W·c` with `X` the non-resident bytes, so one observation
        // is one equation in three unknowns and a *set* of them at different
        // widths and residencies determines all three. Folding it in here, before
        // the copy is divided out, is what stops `c` being a constant somebody
        // measured on one card: `fit` carries the running normal equations and
        // solves them the moment the observations have enough spread to be
        // conditioned.
        //
        // **A forward that copied nothing is folded in too.** It used to be discarded
        // here, on the reasoning that a zero-byte copy teaches nothing about the copy
        // rate — which is true, and threw away everything it *does* teach. On a fully
        // resident card every forward copies nothing, so the fit was fed no observations
        // at all and the per-forward cost stayed at its zero seed; the projected rate was
        // then flat in width and admission refused the second row of every wave as
        // saturated. The copy rate keeps its prior in that regime (`FitSolution::bw` comes
        // back `None`), and the two terms width actually trades against — the fixed cost
        // and the per-token compute — are exactly what these forwards determine.
        self.fit
            .observe(bytes as f64, tokens as f64, pairs, forward_secs, self.alpha);
        if let Some(sol) = self.fit.solve(self.link_bytes_per_s) {
            self.model.compute_secs_per_token = sol.compute_secs_per_token;
            // Zero when the forwards were all at one depth: their attention is in
            // `c` then, and counting it again here would price it twice.
            self.model.attend_secs_per_pair = sol.attend_secs_per_pair.unwrap_or(0.0);
            // The per-forward cost width exists to amortise. Learned, not seeded: it is
            // the wave loop's own overhead — housekeeping, the submission drain, promotion
            // — which is a property of this engine on this machine and not of the model.
            self.model.fixed_secs = sol.fixed_secs;
            self.samples += 1;
            // A fit from forwards that copied nothing leaves the link estimate alone and
            // reports no observation, rather than inventing one from zero bytes.
            if let Some(bw) = sol.bw {
                self.bw_bytes_per_s = bw;
                return Some(bw);
            }
            return None;
        }

        let copy_secs = forward_secs
            - self.model.fixed_secs
            - tokens as f64 * self.model.compute_secs_per_token
            - pairs * self.model.attend_secs_per_pair;
        if copy_secs <= 0.0 {
            return None;
        }
        let raw = bytes as f64 / copy_secs;
        if raw > self.link_bytes_per_s && bytes >= Self::PROBE_BYTES as u64 {
            // **Dampened, not latched.** The link is a ceiling, so raising it on
            // the strength of one forward makes a single fast outlier the new
            // permanent truth — and it can only ever ratchet upward, since
            // nothing below the ceiling moves it back. Moving by `alpha` means
            // sustained evidence raises it and a lone outlier barely does.
            self.link_bytes_per_s += self.alpha * (raw - self.link_bytes_per_s);
        }
        let observed = raw.min(self.link_bytes_per_s);
        if observed < self.link_bytes_per_s * Self::MIN_LINK_FRACTION {
            return None;
        }
        self.bw_bytes_per_s += self.alpha * (observed - self.bw_bytes_per_s);
        self.samples += 1;
        Some(observed)
    }

    /// Learn how well the expert cache converts residency into hits, from the
    /// cache's own counters: `hit` is the fraction of routed experts it found
    /// resident over some interval, at `resident_bytes` on the card.
    ///
    /// The model prices a decode's copy at `1 − resident_fraction^γ`, so what
    /// is learned is the **exponent** that puts the curve through the measured
    /// hit at the residency that produced it: `γ = ln hit / ln fraction`.
    /// Guessing it is the one thing that cannot be done well: the LRU zone and
    /// the Markov predictor together are worth far more than the resident
    /// fraction alone, and by how much depends on the workload's routing — on
    /// the 35B ingest, 0.65 at 39% resident, γ ≈ 0.46.
    ///
    /// Returns the observed exponent, or `None` when the interval says nothing:
    /// a card holding none of the model or all of it (where every residency
    /// gives the same hit), or a hit rate at either end, which no exponent puts
    /// a curve through. Dampened by `alpha` like the other estimates.
    pub fn observe_hit_rate(&mut self, hit: f64, resident_bytes: u64) -> Option<f64> {
        if !hit.is_finite() || hit <= 0.0 || hit >= 1.0 {
            return None;
        }
        let fraction = self.resident_fraction(resident_bytes);
        if fraction <= 0.0 || fraction >= 1.0 {
            return None;
        }
        let observed = hit.ln() / fraction.ln();
        self.decode.hit_exponent += self.alpha * (observed - self.decode.hit_exponent);
        self.hit_samples += 1;
        Some(observed)
    }

    /// Intervals the hit exponent has learned from.
    pub fn hit_samples(&self) -> u64 {
        self.hit_samples
    }

    /// The hit exponent the model currently believes.
    pub fn hit_exponent(&self) -> f64 {
        self.decode.hit_exponent
    }

    /// Whether the two-cost fit has enough spread to be answering, rather than
    /// the planner still running on its seeds.
    pub fn cost_fit_converged(&self) -> bool {
        self.fit.solve(self.link_bytes_per_s).is_some()
    }
}
