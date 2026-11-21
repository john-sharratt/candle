//! Projects the time and rate of a prefill forward from the learned copy rate,
//! per-token compute and fixed costs.
//!
//! A forward's wall-clock cost is the copy of the expert bytes it does not hold,
//! plus compute per row, plus attention per attended pair, plus the fixed loop
//! cost around it, all learned from the engine's own forwards. The fixed cost is
//! included because on a fully resident card the copy is zero and the loop's own
//! overhead is what remains; leaving it out made the rate flat in width. The
//! learned view reports what the card does now, and reports nothing until both
//! halves have learned from a forward, so a cluster never places by defaults.

use super::*;

impl WaveRate {
    /// The learned per-forward loop overhead, in seconds.
    pub fn overhead_secs(&self) -> f64 {
        self.model.overhead_secs
    }

    /// The learned per-forward cost, in seconds — what width amortises.
    ///
    /// Zero until the fit has forwards with enough spread in width to separate it from the
    /// per-token compute, and zero is then also the honest answer for an engine whose
    /// per-forward overhead really is negligible.
    pub fn fixed_secs(&self) -> f64 {
        self.model.fixed_secs
    }

    /// The effective rate as a fraction of the link — how much of the bus the
    /// expert path is actually getting.
    pub fn link_fraction(&self) -> f64 {
        self.bw_bytes_per_s / self.link_bytes_per_s
    }

    /// The time a MoE layer takes in a decode wave when the bus is not the
    /// limit.
    pub fn layer_secs(&self) -> f64 {
        self.layer_secs
    }

    /// What this card does now, by what the model has learned: prefill
    /// throughput at `prefill_rows` and the residency the wave stands at, and
    /// a decode step's time at the wave's decodes and with one more — what one
    /// more decoding turn would cost every decode here. `None` until both
    /// halves have learned from a forward, so a cluster never places by a
    /// model's defaults.
    pub fn learned(
        &self,
        prefill_rows: usize,
    ) -> Option<super::super::super::phase_ring::LearnedRates> {
        if self.samples == 0 || self.decode_samples == 0 {
            return None;
        }
        let step = |routed: usize| self.decode_step_secs(routed, self.resident_now) * 1e3;
        Some(super::super::super::phase_ring::LearnedRates {
            prefill_tok_s: self.rate(prefill_rows.max(1), self.resident_now),
            // The scheduler prices the wide rate (`placed_prefill_near`); the
            // model's own is the rate now.
            prefill_tok_s_plus_wide: self.rate(prefill_rows.max(1), self.resident_now),
            decode_step_ms: step(self.routed_per_layer),
            decode_step_ms_plus_one: step(
                self.routed_per_layer
                    .saturating_add(self.experts_per_decode(0)),
            ),
            decode_step_ms_plus_wide: step(self.routed_per_layer.saturating_add(
                self.experts_per_decode(0) * super::super::super::phase_ring::PLACEMENT_WIDE,
            )),
            // The scheduler fills it in: the rate model sees no queue.
            high_backlog_tokens: 0,
            prompt_tokens: 0.0,
            reply_tokens: 0.0,
            tokens_per_step: 0.0,
        })
    }

    // ── the prefill model ──────────────────────────────────────────────────

    /// Expert bytes a prefill forward copies with `resident` bytes on the card.
    pub fn non_resident_bytes(&self, resident: u64) -> u64 {
        self.geometry.total_bytes().saturating_sub(resident)
    }

    /// Projected time of a prefill forward of `tokens` rows with `resident`
    /// expert bytes on the card, in seconds.
    /// **The wall clock a forward costs the engine**, not the duration of the forward.
    /// The loop's own per-forward overhead is in here for the reason
    /// [`RateModel::overhead_secs`] gives: it is amortised by width exactly as the copy
    /// is, and on a resident card it is the only thing that is.
    pub fn forward_secs(&self, tokens: usize, resident: u64) -> f64 {
        self.forward_secs_attending(tokens, tokens as f64 * self.context_per_row, resident)
    }

    /// The context the rows learned from have attended, on average: the depth
    /// a prompt arriving here is priced at before its own is known.
    pub fn context_per_row(&self) -> f64 {
        self.context_per_row
    }

    /// The shares of a `rows`-wide forward at the learned depth and the
    /// residency now that go to the copy, the rows' compute and their
    /// attention, by the fit. Whether a falling creep rate is the zone
    /// shrinking (the copy's share grows) or contexts deepening (attention's
    /// does) is read from these.
    pub fn forward_shares(&self, rows: usize) -> (f64, f64, f64) {
        let copy = self.non_resident_bytes(self.resident_now) as f64 / self.bw_bytes_per_s;
        let compute = rows as f64 * self.model.compute_secs_per_token;
        let attend = rows as f64 * self.context_per_row * self.model.attend_secs_per_pair;
        let total = self.forward_secs(rows, self.resident_now);
        if total > 0.0 {
            (copy / total, compute / total, attend / total)
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    /// [`Self::forward_secs`] for rows that attend `pairs` query×key pairs
    /// between them — a cohort at a known depth.
    pub fn forward_secs_attending(&self, tokens: usize, pairs: f64, resident: u64) -> f64 {
        self.model.fixed_secs
            + self.model.overhead_secs
            + self.non_resident_bytes(resident) as f64 / self.bw_bytes_per_s
            + tokens as f64 * self.model.compute_secs_per_token
            + pairs * self.model.attend_secs_per_pair
    }

    /// Projected throughput of that forward, tokens a second. Zero for an
    /// empty wave.
    pub fn rate(&self, tokens: usize, resident: u64) -> f64 {
        if tokens == 0 {
            return 0.0;
        }
        tokens as f64 / self.forward_secs(tokens, resident)
    }

    /// The prefill rate ceiling as width goes to infinity with residency held:
    /// every row still costs its compute, and nothing else. What an admission
    /// dislodges is the caller's figure and lowers the real ceiling by it.
    ///
    /// Asserted against rather than decided with: the fill judges each offer on
    /// the gain it buys, never against a ceiling.
    #[cfg(test)]
    pub fn asymptotic_rate(&self) -> f64 {
        1.0 / self.model.compute_secs_per_token
    }

    /// Seconds of compute the model currently believes one prefill row costs.
    ///
    /// The seed until [`CostFit`] has forwards with enough spread to separate
    /// it from the copy; measured from the engine's own forwards after that.
    pub fn compute_secs_per_token(&self) -> f64 {
        self.model.compute_secs_per_token
    }
}
