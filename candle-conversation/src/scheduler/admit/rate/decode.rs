//! Projects a decode step's time and rate from the residency the expert cache
//! holds and the experts the decodes route to.
//!
//! A routed expert is found on the card at a rate that rises with residency through
//! a learned exponent, and each miss is streamed over the bus. A layer's copy hides
//! under its own compute time until the bus sets the step, so a decode is admitted
//! only while the decode rate still improves. Distinct experts are counted rather
//! than routed picks, because rows share experts and a layer fetches each one once.
//! Beyond the misses the fit has seen, the step moves from the nearest one it saw by
//! what a miss costs the link.

use super::*;

impl WaveRate {
    /// The decode model as it stands at the residency now: the hit rate it
    /// projects there, and the fitted per-layer seconds at no miss and per
    /// missed expert (`None` until fitted). What a trade's decode side is
    /// priced from, so a decode change that reads too dear can be checked
    /// against the hit rate the cache measured.
    pub fn decode_terms(&self) -> (f64, Option<(f64, f64, f64, f64)>) {
        (
            self.hit_fraction(self.resident_now),
            self.decode_fit.solve_all(),
        )
    }

    // ── the decode model ───────────────────────────────────────────────────

    /// The fraction of the model's experts that are resident, in `[0, 1]`.
    pub fn resident_fraction(&self, resident: u64) -> f64 {
        (resident as f64 / self.geometry.total_bytes() as f64).min(1.0)
    }

    /// The rate at which a routed expert is found on the card at `resident`
    /// bytes: the resident fraction raised to the learned exponent
    /// ([`DecodeModel::hit_exponent`]) — nothing with nothing resident, every
    /// expert only with the whole model resident, and through the hit rate the
    /// cache measured where it measured it.
    pub fn hit_fraction(&self, resident: u64) -> f64 {
        hit_at(self.resident_fraction(resident), self.decode.hit_exponent)
    }

    /// The fraction of a decode's routed experts that have to be streamed with
    /// `resident` bytes on the card: what the zone and its predictor miss.
    /// Falls as residency rises; never negative, never above 1.
    pub fn streamed_fraction(&self, resident: u64) -> f64 {
        1.0 - self.hit_fraction(resident)
    }

    /// Experts one decode of `draft` proposals routes to per layer.
    pub fn experts_per_decode(&self, draft: usize) -> usize {
        self.decode.experts_per_token.saturating_mul(1 + draft)
    }

    /// Bytes a layer streams for decodes routing to `routed` experts between
    /// them with `resident` bytes on the card: no more experts than the layer
    /// has, of which the streamed fraction crosses the bus.
    pub fn decode_layer_bytes(&self, routed: usize, resident: u64) -> f64 {
        let experts = routed.min(self.geometry.experts_per_layer) as f64;
        experts * self.geometry.slot_bytes as f64 * self.streamed_fraction(resident)
    }

    /// Seconds a layer spends copying for those decodes at that residency.
    pub fn decode_copy_secs(&self, routed: usize, resident: u64) -> f64 {
        self.decode_layer_bytes(routed, resident) / self.bw_bytes_per_s
    }

    /// Projected time of one decode step: every MoE layer takes its own time
    /// or its copy, whichever is longer.
    ///
    /// **Once the steps have varied, by what a miss has cost here.** The fit
    /// ([`DecodeFit`]) prices a layer at its learned time with no miss plus its
    /// learned cost per missed expert, so a step grows with the experts its
    /// decodes route however this card serves a miss. Before the widths have
    /// varied enough to fit, the layer's compute or its copy, whichever is
    /// longer.
    pub fn decode_step_secs(&self, routed: usize, resident: u64) -> f64 {
        let per_layer = match self.decode_fit.solve() {
            Some((a, b)) => {
                // The decodes' attention at the depth they attend: `routed`'s
                // decodes at the mean context a decode row has attended.
                let decodes = routed as f64 / self.experts_per_decode(0).max(1) as f64;
                let depth = self.decode_fit.depth().unwrap_or(0.0);
                let m = self.decode_misses(routed, resident);
                // **The slope where the steps taught it, the link beyond.** A
                // fit learned over a narrow spread of misses says what a miss
                // costs there and nothing far from it: restored on WhiteWorld
                // (71b3b203a, 2026-10-20) from steps a miss or so apart, its
                // slope priced one idle decode at 177 ms. Past the misses it saw,
                // the step moves from the nearest it saw by what a miss costs the
                // link — a slot over the copy rate learned.
                let (lo, hi) = self.decode_fit.miss_support();
                let per_miss = self.geometry.slot_bytes as f64 / self.bw_bytes_per_s.max(1.0);
                let fitted = |m: f64| a + b * m;
                let misses_part = if m < lo {
                    (fitted(lo) - per_miss * (lo - m)).max(a)
                } else if m > hi {
                    fitted(hi) + per_miss * (m - hi)
                } else {
                    fitted(m)
                };
                misses_part + depth * decodes * self.decode_context
            }
            None => self.layer_secs.max(self.decode_copy_secs(routed, resident)),
        };
        self.geometry.moe_layers as f64 * per_layer
    }

    /// Fold one step's attended positions into the mean context a decode row
    /// attends: `positions` across its `decodes` rows.
    pub(super) fn learn_decode_context(&mut self, decodes: usize, positions: f64) {
        if decodes == 0 || !positions.is_finite() || positions <= 0.0 {
            return;
        }
        let per_row = positions / decodes as f64;
        self.decode_context = if self.decode_context > 0.0 {
            self.decode_context + self.alpha * (per_row - self.decode_context)
        } else {
            per_row
        };
    }

    /// Experts a layer fetches for decodes routing to `routed` of them with
    /// `resident` bytes on the card: no more than the layer has, of which the
    /// streamed fraction it does not hold.
    pub(super) fn decode_misses(&self, routed: usize, resident: u64) -> f64 {
        self.distinct_routed(routed) * self.streamed_fraction(resident)
    }

    /// The distinct experts a layer serves when its rows route to `routed`
    /// between them: each row picks its top-k of the layer, so rows share
    /// experts, and the layer fetches each one once however many rows chose
    /// it — `E · (1 − (1 − k/E)^(routed/k))` in expectation.
    ///
    /// **Counted as routed, the misses grew with every row.** Twenty-four
    /// decodes route to 192 experts of a 256-expert layer and share them down
    /// to about 136 distinct, so a step's misses were read 40% high, the fit's
    /// slope was learned on that, and the step published at the width it
    /// prices outran the steps the card took: on BlackWorld (dcba2cc3b,
    /// 2026-10-20) 0.7–3.3 s against co-batched steps of about 0.3 s.
    pub(super) fn distinct_routed(&self, routed: usize) -> f64 {
        let e = self.geometry.experts_per_layer as f64;
        let k = self.decode.experts_per_token.max(1) as f64;
        if e <= 0.0 || routed == 0 {
            return 0.0;
        }
        if k >= e {
            return e;
        }
        e * (1.0 - (1.0 - k / e).powf(routed as f64 / k))
    }

    /// Projected decode throughput: `decodes` sequences stepped a second, for a
    /// wave whose decodes route to `routed` experts a layer at `resident`
    /// bytes. Zero for a wave with no decodes.
    pub fn decode_rate(&self, decodes: usize, routed: usize, resident: u64) -> f64 {
        if decodes == 0 {
            return 0.0;
        }
        decodes as f64 / self.decode_step_secs(routed, resident)
    }

    /// How many decodes of `draft` proposals a wave can carry at `resident`
    /// bytes before their copy outruns the layer time — the point past which
    /// the bus sets the step and another decode stops paying. At least one:
    /// the first is always admitted. When the routed set saturates the layer
    /// the copy stops growing with decodes, and if that copy fits the answer is
    /// the cap alone (`usize::MAX`).
    #[cfg(test)]
    pub fn decode_bus_limit(&self, draft: usize, resident: u64) -> usize {
        let per = self.experts_per_decode(draft).max(1);
        let per_expert_secs = self.geometry.slot_bytes as f64 * self.streamed_fraction(resident)
            / self.bw_bytes_per_s;
        let fit = if per_expert_secs > 0.0 {
            (self.layer_secs / per_expert_secs).floor().max(0.0) as usize / per
        } else {
            usize::MAX
        };
        let saturating = self.geometry.experts_per_layer.div_ceil(per);
        if fit >= saturating {
            usize::MAX
        } else {
            fit.max(1)
        }
    }

    /// How long the decode fit remembers a step: the minute the published
    /// rates are logged at and a measured step bound stays fresh within.
    pub(super) const FIT_MEMORY: std::time::Duration = std::time::Duration::from_secs(60);

    /// The weight the decode fit's sums lose to the step it folds in now:
    /// what [`Self::FIT_MEMORY`] forgets of them since the last one.
    ///
    /// **Forgotten in time, not in steps.** At the per-step dampening the
    /// other estimates use, the fit remembered its last seven steps — a few
    /// seconds of a phase stepping hundreds a minute, over which no step
    /// fetched a different number of experts from the rest, so it never
    /// conditioned: on BlackWorld (192818c31, 2026-10-20) a whole heavy-read
    /// of co-batched steps left it unsolved.
    pub(super) fn fit_alpha(&mut self) -> f64 {
        let now = std::time::Instant::now();
        let alpha = self.decode_fit_at.map_or(0.0, |then| {
            let dt = now.saturating_duration_since(then).as_secs_f64();
            1.0 - (-dt / Self::FIT_MEMORY.as_secs_f64()).exp()
        });
        self.decode_fit_at = Some(now);
        alpha
    }

    /// The decode fit's state for the rates line: its steps, its columns'
    /// spread (misses, creep rows, creep pairs in millions), and why it has
    /// no answer when it has none.
    pub fn decode_fit_report(&self) -> (u64, [f64; 4], Option<FitWhy>) {
        let (n, sd) = self.decode_fit.spread();
        (n, sd, self.decode_fit.answer().err())
    }

    /// The fit's per-layer seconds per position a decode row attends, and the
    /// mean context a decode row attends — `None` until fitted.
    pub fn decode_depth(&self) -> Option<(f64, f64)> {
        self.decode_fit.depth().map(|e| (e, self.decode_context))
    }

    /// Learn the decode fit from one co-batched forward: `decodes` sequences
    /// stepped beside a creep of `creep_layer_rows` rows-through-layers, in
    /// `forward_secs` with `resident_bytes` of experts on the card. Only the
    /// fit learns — its creep term takes what the creep cost — and the layer
    /// time, which a creep's rows would inflate, is left to the decode-only
    /// steps ([`Self::observe_decode`]).
    pub fn observe_decode_beside(
        &mut self,
        decodes: usize,
        resident_bytes: u64,
        forward_secs: f64,
        creep_layer_rows: usize,
        creep_layer_pairs: f64,
        positions: f64,
        rows: usize,
    ) {
        if decodes == 0 || creep_layer_rows == 0 || !forward_secs.is_finite() || forward_secs <= 0.0
        {
            return;
        }
        let layers = self.geometry.moe_layers.max(1) as f64;
        let routed = rows.max(decodes).saturating_mul(self.experts_per_decode(0));
        let fit_alpha = self.fit_alpha();
        self.learn_decode_context(decodes, positions);
        self.decode_fit.observe(
            self.decode_misses(routed, resident_bytes),
            positions,
            creep_layer_rows as f64 / layers,
            creep_layer_pairs / layers,
            forward_secs / layers,
            fit_alpha,
        );
    }

    /// Learn from one decode forward: `decodes` sequences each drafting
    /// `draft` proposals stepped in `forward_secs` with `resident_bytes` of
    /// experts on the card.
    ///
    /// A step is `moe_layers` layers of the longer of the layer's own time and
    /// its copy. When the step's per-layer time exceeds the modelled copy the
    /// wave ran compute-bound and that per-layer time *is* the layer time,
    /// folded in by `alpha`. When it does not, the bus set the step and the
    /// forward says nothing about the layer: `None`, and the estimate is left
    /// alone. Also `None` for a forward with no decodes or a duration that is
    /// not a number.
    #[cfg(test)]
    pub fn observe_decode(
        &mut self,
        decodes: usize,
        draft: usize,
        resident_bytes: u64,
        forward_secs: f64,
    ) -> Option<f64> {
        let positions = decodes as f64 * self.decode_context;
        self.observe_decode_at(
            decodes,
            decodes * (1 + draft),
            resident_bytes,
            forward_secs,
            positions,
        )
    }

    /// [`Self::observe_decode`] with the positions its decode rows attended —
    /// the depth the fit's third column prices.
    pub fn observe_decode_at(
        &mut self,
        decodes: usize,
        rows: usize,
        resident_bytes: u64,
        forward_secs: f64,
        positions: f64,
    ) -> Option<f64> {
        if decodes == 0 || !forward_secs.is_finite() || forward_secs <= 0.0 {
            return None;
        }
        // Every row the step scored routes its own top-k: a drafted decode's
        // verify block is 1 + k rows, and counted as one it read the misses a
        // speculative step paid as a plain step's.
        let routed = rows.max(decodes).saturating_mul(self.experts_per_decode(0));
        let per_layer = forward_secs / self.geometry.moe_layers as f64;
        let fit_alpha = self.fit_alpha();
        self.learn_decode_context(decodes, positions);
        // Every step teaches the fit, bus-bound or not: it is the time against
        // the experts fetched that the fit is of.
        self.decode_fit.observe(
            self.decode_misses(routed, resident_bytes),
            positions,
            0.0,
            0.0,
            per_layer,
            fit_alpha,
        );
        if per_layer <= self.decode_copy_secs(routed, resident_bytes) {
            return None;
        }
        self.layer_secs += self.alpha * (per_layer - self.layer_secs);
        self.decode_samples += 1;
        Some(per_layer)
    }
}
