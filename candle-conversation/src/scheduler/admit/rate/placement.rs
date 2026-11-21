//! Prices a turn placed beside a held cohort and the decodes already running, at the
//! width the trade gives it.
//!
//! A held cohort's width is re-traded against the decodes that would run beside it,
//! and the decode step or the prefill rate a placed turn would get is computed at
//! that traded width and the residency it leaves. A member advertises its next step
//! from this figure, so a wide cohort holding the weights down is priced at the width
//! a placed decode would narrow it to. Each search can start from a hint, the width
//! last answered, so a cohort that has barely moved is found in a probe or two.

use super::*;

impl WaveRate {
    /// The decode step a decode placed beside a held cohort would get, in
    /// milliseconds: one decode more than `decodes` running, the cohort
    /// re-traded against it ([`Self::traded_width`]) and the decodes stepping at
    /// the residency that width leaves. What a member advertises as its next
    /// step, so a wide cohort holding the weights down is priced at the width
    /// a placed decode would narrow it to, not at the width it stands at.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub fn placed_decode_step_ms(
        &self,
        decodes: usize,
        full: usize,
        least: usize,
        chunk: usize,
        prefill_weight: f64,
        weights_at: impl Fn(usize) -> u64,
    ) -> f64 {
        self.placed_decode_step_ms_with(decodes, 1, full, least, chunk, prefill_weight, weights_at)
    }

    /// [`Self::placed_decode_step_ms`] for `more` decodes placed at once — the
    /// step a burst of them would get, each routing its own experts and the
    /// cohort re-traded against all of them.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub fn placed_decode_step_ms_with(
        &self,
        decodes: usize,
        more: usize,
        full: usize,
        least: usize,
        chunk: usize,
        prefill_weight: f64,
        weights_at: impl Fn(usize) -> u64,
    ) -> f64 {
        self.placed_decode_step_near(
            None,
            decodes,
            more,
            full,
            least,
            chunk,
            prefill_weight,
            weights_at,
        )
        .0
    }

    /// [`Self::placed_decode_step_ms_with`] with the trade searched from
    /// `hint` ([`Self::traded_width_near`]), and the width it traded to — the
    /// next search's hint.
    #[allow(clippy::too_many_arguments)]
    pub fn placed_decode_step_near(
        &self,
        hint: Option<usize>,
        decodes: usize,
        more: usize,
        full: usize,
        least: usize,
        chunk: usize,
        prefill_weight: f64,
        weights_at: impl Fn(usize) -> u64,
    ) -> (f64, usize) {
        let decodes = decodes + more;
        let routed = self.routed_for(decodes);
        let width = self.traded_width_near(
            hint,
            full,
            least,
            chunk,
            prefill_weight,
            decodes,
            routed,
            &weights_at,
        );
        (
            self.decode_step_secs(routed, weights_at(width)) * 1e3,
            width,
        )
    }

    /// The prefill rate a turn placed here would drain at, in tokens a second,
    /// before any drain has been measured: the cohort of `full` rows it would
    /// join, traded against the `decodes` running ([`Self::traded_width`]),
    /// in a forward that carries the decodes' `decode_rows` beside it at the
    /// residency that width leaves. With no decode the trade keeps the whole
    /// width, and this is the wide forward's own rate.
    ///
    /// **At the depth the cohort attends.** Its rows attend `context` positions
    /// each on average and the decodes `decode_pairs` between them, so a deep
    /// prompt is priced as one ([`RateModel::attend_secs_per_pair`]).
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub fn placed_prefill_tok_s(
        &self,
        full: usize,
        least: usize,
        chunk: usize,
        prefill_weight: f64,
        decodes: usize,
        decode_rows: usize,
        context: f64,
        decode_pairs: f64,
        weights_at: impl Fn(usize) -> u64,
    ) -> f64 {
        self.placed_prefill_near(
            None,
            full,
            least,
            chunk,
            prefill_weight,
            decodes,
            decode_rows,
            context,
            decode_pairs,
            weights_at,
        )
        .0
    }

    /// [`Self::placed_prefill_tok_s`] with the trade searched from `hint`, and
    /// the width it traded to.
    #[allow(clippy::too_many_arguments)]
    pub fn placed_prefill_near(
        &self,
        hint: Option<usize>,
        full: usize,
        least: usize,
        chunk: usize,
        prefill_weight: f64,
        decodes: usize,
        decode_rows: usize,
        context: f64,
        decode_pairs: f64,
        weights_at: impl Fn(usize) -> u64,
    ) -> (f64, usize) {
        let width = self.traded_width_near(
            hint,
            full,
            least,
            chunk,
            prefill_weight,
            decodes,
            self.routed_for(decodes),
            &weights_at,
        );
        let secs = self.forward_secs_attending(
            width + decode_rows,
            width as f64 * context + decode_pairs,
            weights_at(width),
        );
        let rate = if secs > 0.0 { width as f64 / secs } else { 0.0 };
        (rate, width)
    }
}
