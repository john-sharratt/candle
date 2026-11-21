//! Opens and closes a wave's budget, and admits or charges work into it against the
//! weights it holds.
//!
//! A wave is reset with the residency the weight side holds and the floors the
//! prefill and the decode may not go under. Offers are then admitted in the caller's
//! priority order, and the first is always taken so the wave carries its head. The
//! first refusal of either kind latches the wave full, so residency is spent for
//! whichever offer came first. A charge records a continuation already paid for, such
//! as a stepping decode, and moves the counts and the residency without judging it or
//! latching the wave.

use super::*;

impl WaveRate {
    // ── the budget ─────────────────────────────────────────────────────────

    /// Start composing a wave: `resident_bytes` is what the weight side holds
    /// now, `prefill_floor` and `decode_floor` the residencies a prefill and a
    /// decode may respectively not go under (see [`Self::prefill_floor`] for why
    /// they differ), `max_tokens` the widest prefill the engine will run in one
    /// forward, `max_decodes` the most decodes it will carry.
    pub fn reset(
        &mut self,
        resident_bytes: u64,
        prefill_floor: u64,
        decode_floor: u64,
        max_tokens: usize,
        max_decodes: usize,
    ) {
        self.resident_now = resident_bytes;
        self.prefill_floor = prefill_floor;
        self.decode_floor = decode_floor;
        self.max_tokens = max_tokens;
        self.max_decodes = max_decodes;
        self.tokens = 0;
        self.decodes = 0;
        self.routed_per_layer = 0;
        self.admitted_any = false;
        self.full = false;
    }

    /// Put the model in the state an idle card stands in — `resident_bytes` on
    /// the card, nothing charged — for figures priced with nothing running.
    ///
    /// **Idle is a state, not the last pass's.** The charges a model holds are
    /// the last fill's, made under whatever wave stood then; priced idle from
    /// them, a member advertised a loaded heavy-read's step: on WhiteWorld
    /// (da11a3717, 2026-10-20) a hit of 0.12 and a 3.4 s burst step for ten
    /// idle minutes, the zone long since grown back.
    pub fn settle_idle(&mut self, resident_bytes: u64) {
        self.resident_now = resident_bytes;
        self.tokens = 0;
        self.decodes = 0;
        self.routed_per_layer = 0;
        self.admitted_any = false;
        self.full = false;
    }

    /// Whether the wave reached a limit this pass. Latched by the first
    /// refusal of either kind; cleared by reset.
    #[cfg(test)]
    pub fn is_full(&self) -> bool {
        self.full
    }

    /// Prefill rows admitted into the wave since the last reset.
    #[cfg(test)]
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Decodes admitted into the wave since the last reset.
    #[cfg(test)]
    pub fn decodes(&self) -> usize {
        self.decodes
    }

    /// Experts the admitted decodes route to per layer, before the layer cap.
    #[cfg(test)]
    pub fn routed_per_layer(&self) -> usize {
        self.routed_per_layer
    }

    /// Resident weights as the wave stands: the reset figure, then the last
    /// admitted offer's `weights_after`.
    #[cfg(test)]
    pub fn resident_now(&self) -> u64 {
        self.resident_now
    }

    /// Offer one candidate to the wave, in the caller's priority order, with
    /// the resident weights before it and the resident weights it would leave.
    ///
    /// The caller computes the dislodge — the tier rows, K/V and store the
    /// admission takes from the weight side, at the granularity the allocator
    /// actually claims — because only it can; this model reads the two figures
    /// and nothing else about the cost.
    ///
    /// The first candidate after a reset is admitted whatever it costs: a wave
    /// carries at least its head. After that a refusal of either kind latches
    /// [`Self::is_full`] and every later offer is refused as [`Refusal::Full`].
    pub fn try_admit(
        &mut self,
        admission: Admission,
        weights_before: u64,
        weights_after: u64,
    ) -> Admit {
        if self.full {
            return Admit::Refused(Refusal::Full);
        }
        let first = !self.admitted_any;
        let answer = match admission {
            Admission::Prefill { tokens } => {
                self.judge_prefill(tokens, weights_before, weights_after, first)
            }
            Admission::Decode { draft } => {
                self.judge_decode(draft, weights_before, weights_after, first)
            }
        };
        match answer {
            Ok(projected) => {
                self.admitted_any = true;
                self.resident_now = weights_after;
                Admit::Admitted { projected }
            }
            Err(refusal) => {
                self.full = true;
                Admit::Refused(refusal)
            }
        }
    }

    /// End the wave on a refusal the caller could not turn into a join — what
    /// [`Self::try_admit`] does on every refusal. Every later offer is refused
    /// as [`Refusal::Full`].
    pub fn latch_full(&mut self) {
        self.full = true;
    }

    /// Fold work into the wave that the caller is **not** asking permission
    /// for, and answer with the rate the wave then projects.
    ///
    /// Two kinds of row reach a wave already paid for. A decode step is a
    /// continuation: its ground was bought when its slot was admitted and the
    /// engine owes it the forwards that finish it, so re-judging it refuses
    /// work that has already been paid for — and does it exactly when the
    /// device is tightest, which is when those turns most need to complete and
    /// hand their ground back. And a held creep group rides the next wave
    /// whole, whatever else joins it.
    ///
    /// Neither is a decision, but both are **rows**, and the offers that follow
    /// have to be judged against the wave as it will actually run. So this
    /// records them: the counts move, the residency moves, and — because the
    /// wave is no longer empty — the next offer is judged rather than taken as
    /// the head. Never refuses, and never latches [`Self::is_full`].
    pub fn charge(&mut self, admission: Admission, weights_after: u64) -> f64 {
        let projected = match admission {
            Admission::Prefill { tokens } => {
                self.tokens = self.tokens.saturating_add(tokens);
                self.rate(self.tokens, weights_after)
            }
            Admission::Decode { draft } => {
                self.decodes += 1;
                self.routed_per_layer = self
                    .routed_per_layer
                    .saturating_add(self.experts_per_decode(draft));
                self.decode_rate(self.decodes, self.routed_per_layer, weights_after)
            }
        };
        self.admitted_any = true;
        self.resident_now = weights_after;
        projected
    }
}
