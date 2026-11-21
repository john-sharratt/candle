//! Judges an offer of prefill rows as widening the forward, joining a forward
//! already spent, or outranking work that is running.
//!
//! A widening offer the rate refuses does not close the wave, because the rows are
//! spent and the turn can still join the cohort forming. A join adds no rows, so it
//! is judged against the prefill floor alone, and the decode question is asked of
//! the decode model separately. An outranking offer passes the decode floor that the
//! running lower band yields to, but never the prefill floor, which is the line below
//! which the committed wave no longer fits beside the weights. The cap ends an
//! outranking offer without latching the wave.

use super::*;

impl WaveRate {
    /// Offer prefill rows that would **widen** the forward, judged as
    /// [`Self::try_admit`] judges them — except that a refusal on the rate
    /// alone does not latch the wave full.
    ///
    /// `Saturated` and `Worse` answer one question: whether more rows make this
    /// forward faster. A "no" means its rows are spent, not that the turn
    /// offering them must wait for another cohort — it can still join the one
    /// forming and share the forwards to come ([`Self::try_join`]), which is
    /// what lets the cohort decode at its full width. Latching here turned the
    /// width the rate model chose into a cut through the cohort: on
    /// Qwen3.8-Flash-Next, eight 515-token turns against an 8,192-row budget
    /// admitted five whole, refused the sixth `Saturated { gain: 0.007 }`, and
    /// the last three waited out the first five's decode.
    ///
    /// The floor and the caps still latch: those are the lines every admission,
    /// a join included, has to respect.
    pub fn try_widen(&mut self, tokens: usize, weights_before: u64, weights_after: u64) -> Admit {
        if self.full {
            return Admit::Refused(Refusal::Full);
        }
        let first = !self.admitted_any;
        match self.judge_prefill(tokens, weights_before, weights_after, first) {
            Ok(projected) => {
                self.admitted_any = true;
                self.resident_now = weights_after;
                Admit::Admitted { projected }
            }
            Err(refusal @ (Refusal::Saturated { .. } | Refusal::Worse { .. })) => {
                Admit::Refused(refusal)
            }
            Err(refusal) => {
                self.full = true;
                Admit::Refused(refusal)
            }
        }
    }

    /// Offer a prefill that **joins a forward whose rows are already spent** —
    /// a turn admitted to share the coming forwards rather than to widen this
    /// one.
    ///
    /// It adds no row to this forward, so the prefill model has no gain to
    /// weigh: [`Self::judge_prefill`] would compare the same rows at a lower
    /// residency and call it `Worse`, though the turn's work runs at that
    /// residency whenever it is admitted. What joining changes is the cohort —
    /// its prefill shares the forwards the others are already taking, and its
    /// decode steps beside theirs instead of a whole cohort later — and that
    /// judgement is the decode model's, asked through
    /// [`Self::decode_would_carry`] by [`Self::try_join_trading`], which ends
    /// here. What stays here is the line
    /// every admission respects: residency may not fall under the prefill
    /// floor. A refusal latches [`Self::is_full`], as any other does.
    pub fn try_join(&mut self, weights_after: u64) -> Admit {
        if self.full {
            return Admit::Refused(Refusal::Full);
        }
        if self.admitted_any && weights_after < self.prefill_floor {
            self.full = true;
            return Admit::Refused(Refusal::Floor {
                resident_after: weights_after,
                floor: self.prefill_floor,
            });
        }
        self.admitted_any = true;
        self.resident_now = weights_after;
        Admit::Admitted {
            projected: self.rate(self.tokens, weights_after),
        }
    }

    /// Offer a prefill that **outranks work the engine is already running** — a
    /// band above some active slot's. The row cap and the prefill floor are
    /// judged; the decode floor and the gain rules are not. The fill offers it
    /// here only for a live band, or above the decode floor: an outranking turn
    /// nobody waits on is refused where it would leave residency under the
    /// decode floor (`admit::breaks_the_hold`).
    ///
    /// The floors defend the residency the running work decodes at, and the gain
    /// rules defend the wave's rate. Neither is a reason to hold back a band the
    /// running work yields to: the lower band's slots are what took that
    /// residency, and they keep it until they finish. Measured on `.5`,
    /// 2026-10-17: with fourteen ingest and character slots in flight, a dialogue's
    /// two-row continuation was refused at the decode floor on every pass
    /// (`resident_after` 21.5 GiB against a 25.3 GiB floor) — the person waited
    /// for background work to drain before their tool result was even admitted.
    /// The residency it spends is paid back by the throttle, which yields the
    /// lower bands' creep to this band's decode.
    ///
    /// **The prefill floor is not the lower band's, and it still binds.** What
    /// the case above was held at is the *decode* floor — the residency the
    /// running work decodes at, which the lower band spent and the throttle hands
    /// back. The prefill floor is a different line: below it the wave already
    /// committed and one useful forward no longer fit beside the weights at all,
    /// and [`Self::try_join`] and [`Self::try_widen`] hold it for every other
    /// admission. Judging only the cap let the outranking band break itself:
    /// measured on the 3090 scratch daemon (2026-10-18, zend-load heavy-read),
    /// eight `High` turns admitted outranking calibration work took residency
    /// from 4,318 MiB to 0 against a 3,913 MiB prefill floor, and their own
    /// cohort's sweep then found its 1.71 GB window tier one region short —
    /// 22 of 26 turns lost after 260 s each.
    ///
    /// A refusal on the cap does **not** latch [`Self::is_full`]: the forward's
    /// rows are spent, not the turn's chance, and the caller re-offers it as a
    /// join — or latches the wave itself ([`Self::latch_full`]) when it cannot.
    /// A refusal on the floor latches, as it does for every admission.
    pub fn try_outranking(&mut self, tokens: usize, weights_after: u64) -> Admit {
        if self.full {
            return Admit::Refused(Refusal::Full);
        }
        if self.admitted_any && weights_after < self.prefill_floor {
            self.full = true;
            return Admit::Refused(Refusal::Floor {
                resident_after: weights_after,
                floor: self.prefill_floor,
            });
        }
        let tokens_after = self.tokens.saturating_add(tokens);
        if self.admitted_any && tokens_after > self.max_tokens {
            return Admit::Refused(Refusal::Cap {
                max_tokens: self.max_tokens,
            });
        }
        self.admitted_any = true;
        self.resident_now = weights_after;
        self.tokens = tokens_after;
        Admit::Admitted {
            projected: self.rate(tokens_after, weights_after),
        }
    }
}
