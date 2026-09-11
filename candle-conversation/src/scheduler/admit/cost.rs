//! What admitting one item would take from the device.
//!
//! # This is arithmetic, not a forecast
//!
//! Every pricing this codebase has thrown away (`docs/wave_feeder.md`
//! §4.5–§4.11) was a *forecast compared against a setpoint*: predict a cost,
//! predict what the card will have, admit what "fits". Each was falsified,
//! because a prediction about memory that has not been allocated is wrong in
//! whichever direction the allocator actually behaves.
//!
//! This is a different question, asked at a different moment. The wave's
//! eviction pass has already run, so the free lists are settled; the three
//! terms below are then read off real state rather than modelled:
//!
//! * **activations** — [`candle_nn::kv_cache::WavePlan::tier_bytes`], the very
//!   function `plan_wave_transient` places the tier from. Not an estimate of
//!   the transient cost: the transient cost.
//! * **recurrent** — a fixed store per sequence, and residency is a lookup. So
//!   the term is exactly zero (already on the device) or exactly one store.
//! * **kv** — blocks this advance needs, priced in the formats a **live**
//!   sequence occupies.
//!
//! The KV term is the one with a history. Pricing the *sealed* formats, which
//! a block only reaches once its turn seals and quantizes, understated the
//! working set by **3.7x** — a live sequence sits in active R16. So this prices
//! `active_kv_formats`, and the distinction is the whole reason the function
//! takes them as an argument rather than reading config.

use candle_nn::kv_cache::{KvFormat, CHUNK_SIZE, REGION_BYTES};

/// What one admission would take, split by tenant so a refusal can say which.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Cost {
    /// K/V for the tokens this admission would feed, in the formats a live
    /// sequence occupies.
    pub kv: u64,
    /// The per-sequence model state, or zero when it is already resident.
    pub recurrent: u64,
    /// The wave transient tier this admission's rows would need.
    pub activations: u64,
    /// Rows this admission puts in the forward — the prefill chunk that rides
    /// this wave, or a decode's verify block of `1 + draft`.
    ///
    /// Distinct from every byte term above, and the only one the throughput
    /// model reads directly: bytes say what the admission *costs*, rows say
    /// what it *earns*. `activations` is this row count priced through the
    /// wave planner; the two must move together or the tier is sized for a
    /// wave the rate was never judged on.
    pub rows: usize,
}

impl Cost {
    pub(crate) fn total(&self) -> u64 {
        self.kv
            .saturating_add(self.recurrent)
            .saturating_add(self.activations)
    }

    /// Ground this admission takes **off the free list**, at the granularity
    /// the allocator actually claims it.
    ///
    /// The tier is not in this sum, and that is right *for this question*: it
    /// makes no region live, and it is released at phase 0 of the forward it
    /// was placed for. What this answers is "what does the allocator claim",
    /// which is what [`super::super::prefill`]'s `buy_kv_ground` purchases —
    /// region-granular, and rounded the same way, so the figure admission is
    /// judged on and the figure admission purchases cannot drift.
    ///
    /// It is **not** the answer to "what does the weight side lose": see
    /// [`Self::dislodged_bytes`].
    pub(crate) fn claimed_bytes(&self) -> u64 {
        let region = REGION_BYTES as u64;
        self.kv
            .saturating_add(self.recurrent)
            .div_ceil(region)
            .saturating_mul(region)
    }

    /// Residency the weight side actually loses to this admission — its claim
    /// **and its tier**.
    ///
    /// The figure the throughput model is judged on, because the trade it
    /// exists to weigh is "do these rows earn back the weights they dislodge",
    /// and the tier dislodges weights exactly as a region claim does.
    ///
    /// **The tier being transient does not make it free.** It is released
    /// between forwards, but the fill publishes it
    /// (`set_least_tier_bytes`) and the weight side's growth is then bounded by
    /// "the gap above the live watermark less the slack **and the least
    /// forward's tier**" (`RegionPool::spare`). So a wave that widens holds
    /// that ground against the weight side for as long as it stays that wide —
    /// wave after wave, not for one forward.
    ///
    /// Charging only [`Self::claimed_bytes`] made a prefill chunk's tier free
    /// to the rate model: it saw the rows the chunk earns and the K/V it
    /// claims, and none of the expert residency its tier concedes. The two
    /// judges of one admission then disagreed about its cost — the first-decode
    /// gate is handed [`Self::total`], which has always included the tier.
    ///
    /// Not region-rounded: the tier is placed in whole regions, but the
    /// rounding is the *placement's* and applies to the wave's whole tier, not
    /// to each admission's slice of it. Rounding here would charge a region per
    /// admission for ground the wave takes once.
    pub(crate) fn dislodged_bytes(&self) -> u64 {
        self.claimed_bytes().saturating_add(self.activations)
    }
}

/// Bytes one 32-token K/V block costs across the whole model, in `k`/`v`.
///
/// Takes the formats rather than reading them so the caller must decide which
/// it means — see the module header on the 3.7x that decision is worth.
pub(crate) fn per_block_kv_bytes(
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    k: KvFormat,
    v: KvFormat,
) -> u64 {
    // `bytes_per_block` is the exact figure for one CHUNK_SIZE-element block —
    // per-element arithmetic cannot round-trip a quantized format (`Q4_0` is 18
    // bytes for 32 elements), so this must not be derived from a rate.
    let per = |f: KvFormat| -> u64 { (kv_heads * head_dim) as u64 * f.bytes_per_block() as u64 };
    (per(k) + per(v)).saturating_mul(layers as u64)
}

/// K/V for `tokens` more tokens on a sequence that already holds `held` tokens.
///
/// Blocks are 32 tokens, so a sequence part-way through a block pays nothing
/// for the rest of it: the cost is the blocks the advance actually opens.
pub(crate) fn kv_bytes_for_advance(held: usize, tokens: usize, per_block: u64) -> u64 {
    if tokens == 0 {
        return 0;
    }
    let before = held.div_ceil(CHUNK_SIZE);
    let after = (held + tokens).div_ceil(CHUNK_SIZE);
    (after.saturating_sub(before) as u64).saturating_mul(per_block)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::DType;

    fn f16() -> KvFormat {
        KvFormat::Float(DType::F16)
    }

    #[test]
    fn a_block_costs_both_halves_across_every_layer() {
        // 4 kv heads x 128 dim x 32 tokens x 2 bytes = 32 KiB per half per layer.
        let one = per_block_kv_bytes(1, 4, 128, f16(), f16());
        assert_eq!(one, 2 * 4 * 128 * 32 * 2);
        assert_eq!(
            per_block_kv_bytes(48, 4, 128, f16(), f16()),
            one * 48,
            "every layer holds a block",
        );
    }

    /// **The advance pays for the blocks it opens, not for its tokens.** A
    /// sequence mid-block extends into space it already owns.
    #[test]
    fn an_advance_inside_an_open_block_is_free() {
        let per = per_block_kv_bytes(1, 1, 1, f16(), f16());
        assert_eq!(kv_bytes_for_advance(0, 32, per), per, "one block");
        assert_eq!(kv_bytes_for_advance(0, 33, per), 2 * per, "spills to two");
        assert_eq!(
            kv_bytes_for_advance(10, 5, per),
            0,
            "still inside block one"
        );
        assert_eq!(kv_bytes_for_advance(32, 1, per), per, "opens block two");
        assert_eq!(kv_bytes_for_advance(5, 0, per), 0, "no advance, no cost");
    }

    /// **A one-token decode step prices nothing thirty-one times in thirty-two**
    /// — which is why a slot's whole authorised generation is charged at
    /// admission instead of a step at a time. Run BV priced decodes by the step
    /// and stood 95 slots open on admissions that each cost zero.
    #[test]
    fn a_single_token_step_is_free_inside_an_open_block() {
        let per = per_block_kv_bytes(1, 1, 1, f16(), f16());
        let free = (0..CHUNK_SIZE)
            .filter(|held| kv_bytes_for_advance(*held, 1, per) == 0)
            .count();
        assert_eq!(
            free,
            CHUNK_SIZE - 1,
            "only the step that crosses a block boundary costs anything",
        );
        assert_eq!(
            kv_bytes_for_advance(0, CHUNK_SIZE * 16, per),
            16 * per,
            "the same generation, priced whole, costs every block it opens",
        );
    }

    #[test]
    fn a_cost_totals_its_three_tenants() {
        let c = Cost {
            kv: 10,
            recurrent: 200,
            activations: 3_000,
            rows: 128,
        };
        assert_eq!(c.total(), 3_210);
        assert_eq!(Cost::default().total(), 0);
    }

    /// **The claim is the K/V and the store, rounded to regions — and not the
    /// tier.** The tier is transient: the forward this admits for releases it
    /// at phase 0, so it never makes a region live and never costs the weight
    /// side residency. Charging it here would refuse admissions twice for the
    /// same ground, once at the gate and once at the purchase.
    #[test]
    fn the_claim_is_the_ground_that_goes_live_rounded_to_regions() {
        let region = REGION_BYTES as u64;
        let c = Cost {
            kv: region + 1,
            recurrent: 0,
            activations: 64 * region,
            rows: 128,
        };
        assert_eq!(c.claimed_bytes(), 2 * region, "rounded up, tier excluded");
        assert_eq!(Cost::default().claimed_bytes(), 0);
        // **And the tier is exactly what separates the two questions.** The
        // allocator claims `claimed_bytes`; the weight side loses that plus the
        // tier, because the fill publishes the tier and the growth term is
        // bounded by it. The rate model is judged on the second — charging it
        // the first made a prefill chunk's tier free to the one model whose job
        // is weighing what an admission dislodges.
        assert_eq!(
            c.dislodged_bytes(),
            2 * region + 64 * region,
            "the claim, rounded, plus the tier, not rounded"
        );
        assert_eq!(
            Cost {
                activations: 0,
                ..c
            }
            .dislodged_bytes(),
            c.claimed_bytes(),
            "with no tier the two questions have the same answer"
        );
        // A decode step claims nothing and still moves the tier, which is the
        // case the omission hid completely.
        let step = Cost {
            kv: 0,
            recurrent: 0,
            activations: 3 * region,
            rows: 8,
        };
        assert_eq!(step.claimed_bytes(), 0);
        assert_eq!(step.dislodged_bytes(), 3 * region);
        assert_eq!(
            Cost {
                kv: region,
                recurrent: region,
                ..Default::default()
            }
            .claimed_bytes(),
            2 * region,
            "the store is claimed ground like the K/V",
        );
    }

    /// A decode step claims nothing — its lease was reserved when its slot was
    /// admitted — so it dislodges no weights however many rows it carries.
    #[test]
    fn a_decode_steps_rows_cost_the_weight_side_nothing() {
        let step = Cost {
            kv: 0,
            recurrent: 0,
            activations: 128 << 20,
            rows: 3,
        };
        assert_eq!(step.claimed_bytes(), 0);
    }
}
