//! Judges a prefill against the running decodes it would slow, weighing each phase
//! by its own rate and the work waiting on it.
//!
//! The prefill gain and the decode loss are each taken as a share of their own rate,
//! weighted by the prompts waiting and the decodes running, because raw tokens a
//! second are not comparable across the two. A join that adds no rows is judged on
//! the running decodes' loss alone, since the decode it becomes is waiting work
//! already counted on the prefill side. The widest width that pays is found by
//! bisection, and the trade window records each trade so a run shows the balance it
//! struck.

use super::*;

/// The trades [`WaveRate::try_widen_trading`] judged over a window — what the
/// scheduler's summary line reports, so a run shows how the balance between
/// prefill and decode moved and what the formula weighed each time.
#[derive(Clone, Copy, Debug, Default)]
pub struct TradeWindow {
    pub judged: u64,
    pub admitted: u64,
    pub prefill_weight: f64,
    pub decode_weight: f64,
    pub prefill_gain: f64,
    pub decode_change: f64,
    /// Offers admitted, and of them how many outranked the running work.
    pub offers_admitted: u64,
    pub outranking: u64,
    /// Offers refused, by the rule that refused them: the prefill floor, the
    /// row or decode cap, the rate (worse or saturated), and the latch.
    pub refused_floor: u64,
    pub refused_cap: u64,
    pub refused_rate: u64,
    pub refused_full: u64,
    /// The residency the last offer of the window would leave, and the
    /// prefill floor it was judged against.
    pub last_after: u64,
    pub last_floor: u64,
}

impl WaveRate {
    /// Offer a prefill of `tokens` rows **whose turn goes on to decode**, judged
    /// on the trade the whole wave makes: the prefill throughput it earns
    /// against the decode throughput it costs every running decode by the
    /// residency it dislodges.
    ///
    /// **One trade, not two vetoes.** It used to be asked the prefill question
    /// (`judge_prefill`) and the decode question (`decode_would_carry`) apart,
    /// and refused if either rate fell at all — so a prefill that slowed the
    /// decodes by a fraction of a percent was refused however much it earned.
    ///
    /// **Each phase is weighed as a share of its own rate, by the work waiting
    /// on it.** The two rates are not comparable as tokens a second — a
    /// prefill forward runs thousands, a decode tens — so summing them lets any
    /// prefill gain drown any decode loss. The trade instead is
    ///
    /// `prefill_weight × ΔP / P_cap  ≥  decode_weight × (−ΔD) / D`
    ///
    /// with `P_cap` the prefill rate of a forward at the wave's whole width at
    /// the residency found, and `D` the running decodes' rate at it.
    /// `prefill_weight` is the prompts waiting, each counted more the longer it
    /// has waited (`Ground::prefill_weight`); `decode_weight` is the decodes
    /// running. So many decodes and few prompts make a dislodge dear and the
    /// prompts wait; many prompts and few decodes make the rows worth the
    /// weights, and the weights go; in between the cheap prefills co-batch and
    /// the balance follows the queues. The decode side is the running decodes'
    /// loss alone: the decode the turn becomes is waiting work, already counted
    /// on the prefill side.
    ///
    /// The prefill floor still binds — under it the cache cannot serve a forward
    /// — and the row cap and the head waiver are `try_widen`'s. The decode is
    /// not committed: it rides a later wave and is charged there. While the
    /// decode model is unlearned its side is left out, for
    /// `decode_would_carry`'s reason, and the prefill gain alone decides.
    /// The trade [`Self::try_widen_trading`] judges, without admitting
    /// anything: widening a wave from `tokens_before` to `tokens_after` rows,
    /// which moves residency from `weights_before` to `weights_after`, gains the
    /// prefill `prefill_weight × ΔP / P_cap` and costs the running decodes
    /// `decode_weight × ΔD / D`. Positive pays.
    ///
    /// What a held cohort's width is re-judged by at each fit, so a sweep
    /// admitted when no decode ran does not keep its width once one does.
    #[allow(clippy::too_many_arguments)]
    pub fn trade_net(
        &self,
        tokens_before: usize,
        tokens_after: usize,
        weights_before: u64,
        weights_after: u64,
        prefill_weight: f64,
        decode_weight: f64,
        decodes: usize,
        routed: usize,
    ) -> f64 {
        let prefill_now = self.rate(tokens_before, weights_before);
        let prefill_with = self.rate(tokens_after, weights_after);
        let p_cap = self.rate(self.max_tokens.max(tokens_after), weights_before);
        let prefill_gain = if p_cap > 0.0 {
            (prefill_with - prefill_now) / p_cap
        } else {
            0.0
        };
        let decode_learned = self.decode_samples >= Self::MIN_DECODE_SAMPLES;
        let decode_now = self.decode_rate(decodes, routed, weights_before);
        let decode_change = if decode_learned && decodes > 0 && decode_now > 0.0 {
            let decode_with = self.decode_rate(decodes, routed, weights_after);
            (decode_with - decode_now) / decode_now
        } else {
            0.0
        };
        prefill_weight * prefill_gain + decode_weight * decode_change
    }

    /// Experts the decodes route to per layer were there `decodes` of them: the
    /// wave's as it stands, and each decode more or fewer adding or taking its
    /// own.
    pub fn routed_for(&self, decodes: usize) -> usize {
        let per = self.experts_per_decode(0);
        if decodes >= self.decodes {
            self.routed_per_layer
                .saturating_add((decodes - self.decodes).saturating_mul(per))
        } else {
            self.routed_per_layer
                .saturating_sub((self.decodes - decodes).saturating_mul(per))
        }
    }

    /// Whether the decode model has learned enough to be traded against.
    pub fn decode_learned(&self) -> bool {
        self.decode_samples >= Self::MIN_DECODE_SAMPLES
    }

    /// The widest a cohort of `full` rows may run beside `decodes` decodes
    /// routing to `routed` experts a layer: the widest chunk-aligned width,
    /// down to `least`, whose last chunk's trade ([`Self::trade_net`]) still
    /// pays, with `weights_at(rows)` the residency a width leaves. `full` when
    /// there is nothing to trade against — no decode, or a decode model not yet
    /// learned.
    ///
    /// Bisected: the decodes' loss grows with the width and the prefill's gain
    /// falls, so the widths that pay are a prefix.
    #[allow(clippy::too_many_arguments)]
    pub fn traded_width(
        &self,
        full: usize,
        least: usize,
        chunk: usize,
        prefill_weight: f64,
        decodes: usize,
        routed: usize,
        weights_at: impl Fn(usize) -> u64,
    ) -> usize {
        self.traded_width_near(
            None,
            full,
            least,
            chunk,
            prefill_weight,
            decodes,
            routed,
            weights_at,
        )
    }

    /// [`Self::traded_width`], searched outward from `hint` — the width the
    /// same question was last answered at. The answer is the same; a cohort
    /// that has barely moved since is found in a probe or two rather than a
    /// whole bisection, each probe being two tiers composed over every layer.
    #[allow(clippy::too_many_arguments)]
    pub fn traded_width_near(
        &self,
        hint: Option<usize>,
        full: usize,
        least: usize,
        chunk: usize,
        prefill_weight: f64,
        decodes: usize,
        routed: usize,
        weights_at: impl Fn(usize) -> u64,
    ) -> usize {
        let least = least.min(full);
        if decodes == 0 || !self.decode_learned() || full <= least {
            return full;
        }
        let chunk = chunk.max(1);
        let rows_at = |i: usize| (i * chunk).min(full);
        let pays = |i: usize| {
            let rows = rows_at(i);
            let before = rows.saturating_sub(chunk);
            self.trade_net(
                before,
                rows,
                weights_at(before),
                weights_at(rows),
                prefill_weight,
                decodes as f64,
                decodes,
                routed,
            ) > 1e-12
        };
        let (lo, hi) = (least.div_ceil(chunk), full.div_ceil(chunk));
        let hint = hint.map(|h| h.div_ceil(chunk));
        largest_paying(lo, hi, hint, pays).map_or(least, rows_at)
    }

    pub fn try_widen_trading(
        &mut self,
        tokens: usize,
        weights_before: u64,
        weights_after: u64,
        prefill_weight: f64,
        decode_weight: f64,
    ) -> Admit {
        if self.full {
            return Admit::Refused(Refusal::Full);
        }
        let first = !self.admitted_any;
        let tokens_after = self.tokens.saturating_add(tokens);
        if !first {
            if tokens_after > self.max_tokens {
                return Admit::Refused(Refusal::Cap {
                    max_tokens: self.max_tokens,
                });
            }
            if weights_after < self.prefill_floor {
                self.full = true;
                return Admit::Refused(Refusal::Floor {
                    resident_after: weights_after,
                    floor: self.prefill_floor,
                });
            }
        }
        let prefill_now = self.rate(self.tokens, weights_before);
        let prefill_with = self.rate(tokens_after, weights_after);
        let projected = prefill_with;
        if !first {
            let p_cap = self.rate(self.max_tokens.max(tokens_after), weights_before);
            let prefill_gain = if p_cap > 0.0 {
                (prefill_with - prefill_now) / p_cap
            } else {
                0.0
            };
            let decode_learned = self.decode_samples >= Self::MIN_DECODE_SAMPLES;
            let decode_now = self.decode_rate(self.decodes, self.routed_per_layer, weights_before);
            // The running decodes' loss alone: the same decodes at the residency
            // the offer leaves against the residency it found. The decode the
            // turn becomes is not on this side — it is waiting work, which the
            // prefill weight already counts — and counting it here added +1/n
            // to every trade, so the decode side read as a gain on nearly every
            // offer (WhiteWorld, 2026-10-18: dD +0.17 to +0.33, every trade
            // admitted).
            let decode_change = if decode_learned && self.decodes > 0 && decode_now > 0.0 {
                let decode_with =
                    self.decode_rate(self.decodes, self.routed_per_layer, weights_after);
                (decode_with - decode_now) / decode_now
            } else {
                0.0
            };
            let net = prefill_weight * prefill_gain + decode_weight * decode_change;
            self.trades.judged += 1;
            self.trades.admitted += u64::from(net > 1e-12);
            self.trades.prefill_weight += prefill_weight;
            self.trades.decode_weight += decode_weight;
            self.trades.prefill_gain += prefill_gain;
            self.trades.decode_change += decode_change;
            // Every trade's terms, so a run shows where the formula was wrong
            // against what the forwards that followed measured.
            tracing::debug!(
                target: "candle_conversation::scheduler::admission",
                tokens,
                prefill_weight,
                decode_weight,
                prefill_gain,
                decode_change,
                net,
                resident_before_mib = weights_before >> 20,
                resident_after_mib = weights_after >> 20,
                "prefill traded against decode",
            );
            // Any net gain pays: each side is already a share of its own rate,
            // weighted by the work waiting on it.
            if net <= 1e-12 {
                // Unlatched, as `try_widen`'s rate refusals are: the forward's
                // rows stop paying, and the turn is re-offered as a join.
                return Admit::Refused(if net < 0.0 {
                    Refusal::Worse {
                        before: prefill_now,
                        after: prefill_with,
                    }
                } else {
                    Refusal::Saturated { gain: net }
                });
            }
        }
        self.admitted_any = true;
        self.resident_now = weights_after;
        self.tokens = tokens_after;
        Admit::Admitted { projected }
    }

    /// The trades judged since the last call, and a fresh window.
    pub fn note_admitted(&mut self, outranking: bool, after: u64, floor: u64) {
        self.trades.offers_admitted += 1;
        self.trades.outranking += u64::from(outranking);
        self.trades.last_after = after;
        self.trades.last_floor = floor;
    }

    /// A refusal the fill made, for the window — see [`TradeWindow`].
    pub fn note_refusal(&mut self, refusal: Refusal, after: u64, floor: u64) {
        let t = &mut self.trades;
        match refusal {
            Refusal::Floor { .. } => t.refused_floor += 1,
            Refusal::Cap { .. } | Refusal::DecodeCap { .. } => t.refused_cap += 1,
            Refusal::Worse { .. } | Refusal::Saturated { .. } => t.refused_rate += 1,
            Refusal::Full => t.refused_full += 1,
        }
        t.last_after = after;
        t.last_floor = floor;
    }

    pub fn take_trades(&mut self) -> TradeWindow {
        std::mem::take(&mut self.trades)
    }

    /// [`Self::try_join`] beside running decodes: a join adds no row, so it has
    /// no prefill gain, but the K/V it claims dislodges residency every running
    /// decode steps slower for.
    ///
    /// **A join that goes on to decode is judged with its decode in the wave.**
    /// The question is [`Self::decode_would_carry`]'s: does the wave step more
    /// decodes a second with this turn decoding in it, at the residency it
    /// leaves, than without it at the residency it found — and by at least the
    /// minimum gain. Where a dislodge is a large share of the zone — a streaming
    /// card, or one already near its floor — the slower step outweighs the extra
    /// row and the join is refused; where it is a sliver, the row pays and the
    /// wave widens.
    ///
    /// **Admitted, its decode is charged.** The joins of one pass decode
    /// together once their prompts are in, so each is priced against the wave
    /// with the ones before it decoding: twenty-four turns of one burst are
    /// twenty-four marginal decodes, not twenty-four first ones, and the burst
    /// stops where the next row no longer pays.
    ///
    /// **Priced without its own decode, no join could ever pay.** The running
    /// decodes' rate at lower residency is below their rate at the residency
    /// found, so every join that claimed a byte was `Worse` — on BlackWorld
    /// (2026-10-27, Flash-Next ingest) 582 of 663 passes stopped there, refusing
    /// a calibration's twenty-four thinking turns for a 0.01% projected loss
    /// (`before: 39.777, after: 39.772`) while 1–3 rows decoded.
    ///
    /// A join that never decodes — a section — has nothing to set against the
    /// loss, and one that would cost the running decodes anything is refused as
    /// `Worse`. `decode_weight` is those decodes' claim, as in
    /// [`Self::try_widen_trading`].
    ///
    /// **A refusal the join undid.** On BlackWorld (dcba2cc3b, 2026-10-20)
    /// heavy-read's outranking tool results were refused on the trade and
    /// admitted at once as joins, judged on the prefill floor alone — `ok=1
    /// rate=1` on almost every pass for five minutes — and the expert zone fell
    /// to 21 GB against its 28.5 GB hold as before the trade was asked at all.
    /// A decoding join is judged on the decodes' rate, not on the floor alone.
    pub fn try_join_trading(
        &mut self,
        weights_before: u64,
        weights_after: u64,
        decode_weight: f64,
        decodes_after: bool,
    ) -> Admit {
        if self.full {
            return Admit::Refused(Refusal::Full);
        }
        if self.admitted_any
            && self.decodes > 0
            && self.decode_learned()
            && weights_after < weights_before
        {
            if decodes_after {
                if let Err(refusal) = self.decode_would_carry(0, weights_before, weights_after) {
                    return Admit::Refused(refusal);
                }
            } else {
                let now = self.decode_rate(self.decodes, self.routed_per_layer, weights_before);
                let with = self.decode_rate(self.decodes, self.routed_per_layer, weights_after);
                if now > 0.0 && decode_weight * (with - now) / now < -1e-12 {
                    return Admit::Refused(Refusal::Worse {
                        before: now,
                        after: with,
                    });
                }
            }
        }
        let admitted = self.try_join(weights_after);
        if decodes_after && matches!(admitted, Admit::Admitted { .. }) {
            self.decodes += 1;
            self.routed_per_layer = self
                .routed_per_layer
                .saturating_add(self.experts_per_decode(0));
        }
        admitted
    }
}
