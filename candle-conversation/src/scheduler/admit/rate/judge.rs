//! Decides whether a prefill offer, or a finished prefill's promotion to decode,
//! should be admitted to the wave.
//!
//! A prefill offer is judged on the gain its rows buy against the residency they
//! dislodge. A promotion is judged on whether the decode it becomes can be carried
//! at the residency it leaves, because a decode's copy is paid per layer per step
//! and the residency it spends is multiplied across every step that follows. The
//! promotion judgement does not refuse on a constant that has not been measured:
//! the seed is far too optimistic, and refusing on it would keep the engine inside
//! the window it is refusing on. Nothing on the admission path calls it yet, and a
//! refused promotion turns away one turn without closing the wave.

use super::*;

impl WaveRate {
    /// Whether the decode a prefill is going to become could be carried at
    /// `resident` — asked **while judging that prefill**, so the turn is priced
    /// by both models at the one moment a refusal is still cheap.
    ///
    /// **Pure, and it buys nothing.** Nothing is mutated and nothing is latched
    /// — this is a question, and [`Self::try_join_trading`], which asks it of
    /// every join that goes on to decode, makes the decision and charges the
    /// decode it admits.
    ///
    /// Why the prefill model cannot answer it: a prefill's copy is paid per
    /// forward, so `rate()` amortises it across the chunk's rows and widening
    /// stays a gain down to the floor. A decode's copy is paid per layer and
    /// scales with `1 - hit(resident)`, so the residency the prefill spent is
    /// multiplied across every step of the decode that follows. The two answers
    /// diverge exactly when the weight side is under pressure, which is when it
    /// matters.
    ///
    /// The unlearned window is honoured for the same reason
    /// [`Self::judge_promotion`] honours it, and it matters more here: refusing
    /// on a seed measured 26x optimistic would turn away prefills that have not
    /// yet run the decode forwards the estimate is learned from, and the engine
    /// would never leave the window it is refusing on.
    ///
    /// **No decode floor: the marginal decode rate decides.** A fixed fraction
    /// of achievable residency stood here, first as a wall and then as a line
    /// the crossing offer could not pass. It refused turns the decodes could
    /// carry: measured on the 3090 scratch daemon (2026-10-18, 24 `High`
    /// clients), residency 3–4.5 GiB against a 7.4 GiB floor, 911 offers
    /// refused in eleven minutes and first text p50 160 s where the same rung
    /// had run at 10.9 s. The question that line stood in for is the one this
    /// asks — does the wave decode faster with this turn in it, at the
    /// residency the turn leaves, than without it at the residency it found —
    /// and the prefill floor still bounds what fits.
    pub fn decode_would_carry(
        &self,
        draft: usize,
        resident_before: u64,
        resident_after: u64,
    ) -> Result<f64, Refusal> {
        let routed_after = self
            .routed_per_layer
            .saturating_add(self.experts_per_decode(draft));
        let projected = self.decode_rate(self.decodes + 1, routed_after, resident_after);
        if self.decode_samples < Self::MIN_DECODE_SAMPLES {
            return Ok(projected);
        }
        // Against the wave's decodes as they stand: if adding the decode this
        // turn becomes would not pay for itself — the wave without it at the
        // residency it found, against the wave with it at the residency it
        // leaves — the prefill that creates it should not be admitted either.
        // With no decode in the wave yet there is nothing to compare against
        // and the turn is carried — the same head rule the rest of the model
        // keeps.
        if self.decodes > 0 {
            let current = self.decode_rate(self.decodes, self.routed_per_layer, resident_before);
            Self::judge_gain(current, projected, self.min_gain)?;
        }
        Ok(projected)
    }

    /// Judge a turn that has finished prefilling and is asking to become a
    /// decode — the one decision the decode side gets, and the only place
    /// [`Self::judge_decode`] is ever reached.
    ///
    /// **Why this is not [`Self::try_admit`].** That entry point carries two
    /// behaviours that belong to composing a wave and are wrong here. It
    /// short-circuits on the [`Self::is_full`] latch, so a boundary decision
    /// landing after some *earlier, unrelated* offer was refused would be
    /// refused for that reason rather than on this turn's merits. And it admits
    /// the first candidate after a reset unconditionally, which would hand a
    /// free pass to whichever turn happened to arrive first. A finished prefill
    /// is not competing for a place in this wave; it is asking whether the
    /// decode side can carry it at all.
    ///
    /// **The caller must hand over the counterfactual.** The slot is already
    /// resident — its K/V and store were claimed at prefill admission — so
    /// `weights_before` is residency *as if this turn were not there*
    /// (`resident_now + Cost::dislodged_bytes()`) and `weights_after` is residency
    /// as it actually stands. Offering it any other way asks the model whether
    /// to admit something already admitted, which double-counts its ground and
    /// can only answer yes.
    ///
    /// A refusal does not latch the wave: this turn is being turned away, not
    /// the ones behind it.
    ///
    /// **Not yet on the admission path.** `super::admit_ground` offers prefills
    /// and charges decodes as continuations; nothing calls this. Wiring it means
    /// a finished prefill can be *refused* its promotion and held as a finished
    /// prefill until residency improves, which is a behaviour this engine has
    /// never run and a wedge if residency never does improve. It wants its own
    /// change and its own verification, not a line added at the end of another.
    #[cfg(test)]
    pub fn judge_promotion(
        &mut self,
        draft: usize,
        weights_before: u64,
        weights_after: u64,
    ) -> Admit {
        // **Never refuse on a constant that has not been measured.** `layer_secs`
        // is seeded from the model's advertised decode rate and both measured
        // runs found that seed ~26x optimistic: 0.000813 against a converged
        // 0.018–0.028. While it is still wrong, `T_step`'s `max(layer_secs,
        // copy_secs)` is dominated by the copy term, so every additional decode
        // prices as maximally expensive and the boundary refuses turns it should
        // carry — run 32 evicted 36 that way before the estimate caught up, and
        // run 33 another 10, every one of them `Worse` at three or four decodes.
        //
        // The copy side already refuses to be trusted below `min_learn_rows`;
        // this is the same rule for the compute side. Carrying a turn that
        // should have been refused costs a wave of throughput, while refusing
        // one that should have been carried costs a whole re-prefill — so while
        // the estimate is unlearned, the cheaper error is to admit.
        if self.decode_samples < Self::MIN_DECODE_SAMPLES {
            let projected = self.decode_rate(
                self.decodes + 1,
                self.routed_per_layer
                    .saturating_add(self.experts_per_decode(draft)),
                weights_after,
            );
            self.decodes += 1;
            self.routed_per_layer = self
                .routed_per_layer
                .saturating_add(self.experts_per_decode(draft));
            self.admitted_any = true;
            self.resident_now = weights_after;
            return Admit::Admitted { projected };
        }
        match self.judge_decode(draft, weights_before, weights_after, false) {
            Ok(projected) => {
                self.admitted_any = true;
                self.resident_now = weights_after;
                Admit::Admitted { projected }
            }
            Err(refusal) => Admit::Refused(refusal),
        }
    }

    pub(super) fn judge_prefill(
        &mut self,
        tokens: usize,
        before: u64,
        after: u64,
        first: bool,
    ) -> Result<f64, Refusal> {
        let tokens_after = self.tokens.saturating_add(tokens);
        if !first {
            if tokens_after > self.max_tokens {
                return Err(Refusal::Cap {
                    max_tokens: self.max_tokens,
                });
            }
            if after < self.prefill_floor {
                return Err(Refusal::Floor {
                    resident_after: after,
                    floor: self.prefill_floor,
                });
            }
        }
        let projected = self.rate(tokens_after, after);
        if self.tokens > 0 {
            let current = self.rate(self.tokens, before);
            Self::judge_gain(current, projected, self.prefill_min_gain(before, after))?;
        }
        self.tokens = tokens_after;
        Ok(projected)
    }

    /// The minimum gain a prefill offer must buy: [`Self::min_gain`] when it spends
    /// residency a forward will then have to copy, and nothing when it does not.
    ///
    /// **The minimum exists to protect residency, so it applies only where residency
    /// is spent.** A prefill row that dislodges nothing the forward must copy costs the
    /// wave only its own compute — and a queued prompt pays that compute whenever it
    /// runs. Refusing it does not save the work, it moves it into a later wave that
    /// pays every per-wave cost again: its own forward, its first-token finalise, and
    /// its own decode and seal round, queued behind this one's. Measured on the
    /// Qwen3.6-35B-A3B with every expert resident: eight turns submitted together, the
    /// fifth refused `Saturated { gain: 0.009 }`, and the second four waited out the
    /// first four's whole decode and seal before their prefill finished.
    ///
    /// The `Worse` rule still holds either way — an offer that lowers the wave's rate
    /// is refused whatever it costs.
    pub(super) fn prefill_min_gain(&self, before: u64, after: u64) -> f64 {
        if self.non_resident_bytes(after) > self.non_resident_bytes(before) {
            self.min_gain
        } else {
            0.0
        }
    }

    pub(super) fn judge_decode(
        &mut self,
        draft: usize,
        before: u64,
        after: u64,
        first: bool,
    ) -> Result<f64, Refusal> {
        let decodes_after = self.decodes + 1;
        let routed_after = self
            .routed_per_layer
            .saturating_add(self.experts_per_decode(draft));
        if !first {
            if decodes_after > self.max_decodes {
                return Err(Refusal::DecodeCap {
                    max_decodes: self.max_decodes,
                });
            }
            // The line every admission respects: under the prefill floor the
            // expert cache can no longer serve a forward. Above it, whether a
            // decode pays at the residency it leaves is the gain's question —
            // there is no separate decode floor.
            if after < self.prefill_floor {
                return Err(Refusal::Floor {
                    resident_after: after,
                    floor: self.prefill_floor,
                });
            }
        }
        let projected = self.decode_rate(decodes_after, routed_after, after);
        if self.decodes > 0 {
            let current = self.decode_rate(self.decodes, self.routed_per_layer, before);
            Self::judge_gain(current, projected, self.min_gain)?;
        }
        self.decodes = decodes_after;
        self.routed_per_layer = routed_after;
        Ok(projected)
    }

    /// The one rule both kinds share: the rate must not fall, and must rise by
    /// at least `min_gain`. The tolerances keep a flat rate — an admission that
    /// dislodges nothing on a fully resident card, where every row costs exactly
    /// what it earns — from reading as worse by an ulp.
    pub(super) fn judge_gain(current: f64, projected: f64, min_gain: f64) -> Result<(), Refusal> {
        if projected < current * (1.0 - 1e-9) {
            return Err(Refusal::Worse {
                before: current,
                after: projected,
            });
        }
        let gain = (projected - current) / current;
        if gain + 1e-9 < min_gain {
            return Err(Refusal::Saturated { gain });
        }
        Ok(())
    }
}
