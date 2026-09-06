//! Which published Hermes-4-14B a card can run.
//!
//! The single place the prose guest's quantisation is decided, so the choice
//! cannot drift between the downloader that fetches a file and the loader that
//! opens it — the same reason `zend`'s model choice and `flux`'s quant choice
//! each live in one function.
//!
//! # Why the prose guest needs a ladder
//!
//! A guest is not resident. It claims span ground at the start of a drain, and
//! what it can claim is what the shedding ladder can free with a 9 B dense model
//! already standing in the same reservation.
//!
//! **That figure is not fixed, and the ladder is sized against the low end of
//! it.** Across one evening of drains on the 3090 it ranged from **11,760 MiB**
//! to **13,808 MiB** — the same card, the same resident model, differing by how
//! warm the working set was when the drain began. A rung chosen against the best
//! observation is a rung that loads on a quiet card and fails on a busy one,
//! which is the intermittent class of failure this subsystem has already been
//! bitten by once. So **11,760 MiB (11.48 GiB) is the ceiling used here**, and
//! the good days are margin rather than capacity.
//!
//! # What has to fit, and it is not only the weights
//!
//! [`super::prose::ProseSpec`]'s footprint is `weights + per_token × context`,
//! and both are claimed up front. Hermes-4-14B is Qwen3-14B: 40 layers, 8 KV
//! heads, 128-wide, F32 in this guest's forward — which is
//!
//! ```text
//! 2 × 8 × 128 × 4 × 40  = 327,680 bytes of K/V per token
//!         + 2 × 64 × 4  =     512 bytes of RoPE per token
//!                       ≈ 320 KiB per token
//! ```
//!
//! **1.28 GiB at a 4,096-token context**, which is the figure every rung below
//! is sized against. A ladder written against the file size alone puts the top
//! rung a gigabyte over the ceiling and discovers it after the engine has been
//! evicted to make room.
//!
//! # Why not Hermes 4.3
//!
//! It is the newer tune and it does not fit. The 36 B is 13.6 GB at Q2_K — the
//! whole ceiling, with nothing left for K/V — and 17.6 GB at Q3_K_M. There is
//! no rung of it this card can hold, and Q2 of a 36 B would not be worth the
//! seat if there were.

use candle::quantized::GgmlDType;

/// A published quantisation of Hermes-4-14B.
///
/// Four rungs rather than the two FLUX needs, because this ladder spans 16 GB
/// to 48 GB rather than 16 to 24, and a 14 B dense model's file size moves
/// enough across that range for the middle to be worth having.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HermesQuant {
    /// 7.32 GB. The rung a 16 GB card can hold beside a resident model, and the
    /// only one it can. Q3 on a 14 B is a real quality cost and is the price of
    /// the seat.
    Q3KM,
    /// 9.00 GB. **What a 24 GB card runs.** 8.38 GiB of weights and 1.25 GiB of
    /// K/V at a 4k window is 9.63 GiB, which leaves 1.85 GiB of the 11.48 GiB
    /// ceiling spare — the margin that makes it load on a warm card as well as a
    /// cold one.
    Q4KM,
    /// 10.51 GB. 11.04 GiB with a 4k window, which clears the ceiling by 0.44
    /// GiB — too little to rely on when the figure itself moves by two gigabytes
    /// between drains. It wants a card with real headroom, not a 24 GB one.
    Q5KM,
    /// 12.12 GB. 12.54 GiB with a 4k window: over the 11.48 GiB ceiling outright,
    /// and only under the best-day 13.48 GiB one. A 24 GB card taking this rung
    /// would place most of it and then fail on the RoPE tables, having already
    /// evicted the engine to get that far.
    Q6K,
}

/// Below this much total VRAM, Q5 rather than Q6.
pub const Q6_MIN_TOTAL_VRAM_BYTES: u64 = 32 * (1 << 30);
/// Below this, Q4 rather than Q5.
///
/// **28, not 24.** Q5 needs 11.04 GiB against a 24 GB card's 11.48 GiB of
/// claimable ground, and a margin of 0.44 GiB is not a margin when the ground
/// itself moves by two gigabytes between drains. The 3090 takes Q4 — a rung
/// that always loads beats a rung that usually does.
pub const Q5_MIN_TOTAL_VRAM_BYTES: u64 = 28 * (1 << 30);
/// Below this, Q3 — the 16 GB rung.
pub const Q4_MIN_TOTAL_VRAM_BYTES: u64 = 20 * (1 << 30);

impl HermesQuant {
    /// The rung for a card with `total_vram_bytes` of memory.
    ///
    /// Total rather than free, for the reason `flux::quant_choice` gives: the
    /// decision is which checkpoint this deployment runs, and free memory is a
    /// reading of the moment that would have a daemon pick a different file
    /// depending on what happened to be resident when it started.
    pub fn for_vram(total_vram_bytes: u64) -> Self {
        if total_vram_bytes >= Q6_MIN_TOTAL_VRAM_BYTES {
            Self::Q6K
        } else if total_vram_bytes >= Q5_MIN_TOTAL_VRAM_BYTES {
            Self::Q5KM
        } else if total_vram_bytes >= Q4_MIN_TOTAL_VRAM_BYTES {
            Self::Q4KM
        } else {
            Self::Q3KM
        }
    }

    /// The repository publishing it.
    ///
    /// bartowski's conversion rather than NousResearch's own: the upstream repo
    /// ships BF16 and FP8 safetensors, and these are the GGUF conversions of
    /// that release. Same weights; the original licence still applies.
    pub fn repo(&self) -> &'static str {
        "bartowski/NousResearch_Hermes-4-14B-GGUF"
    }

    pub fn filename(&self) -> &'static str {
        match self {
            Self::Q3KM => "NousResearch_Hermes-4-14B-Q3_K_M.gguf",
            Self::Q4KM => "NousResearch_Hermes-4-14B-Q4_K_M.gguf",
            Self::Q5KM => "NousResearch_Hermes-4-14B-Q5_K_M.gguf",
            Self::Q6K => "NousResearch_Hermes-4-14B-Q6_K.gguf",
        }
    }

    /// The published file's length, for a downloader's progress total when the
    /// server omits `Content-Length`.
    pub fn bytes(&self) -> u64 {
        match self {
            Self::Q3KM => 7_320_000_000,
            Self::Q4KM => 9_000_000_000,
            Self::Q5KM => 10_510_000_000,
            Self::Q6K => 12_120_000_000,
        }
    }

    /// The dominant block type, for reporting.
    pub fn dtype(&self) -> GgmlDType {
        match self {
            Self::Q3KM => GgmlDType::Q3_K,
            Self::Q4KM => GgmlDType::Q4_K,
            Self::Q5KM => GgmlDType::Q5_K,
            Self::Q6K => GgmlDType::Q6_K,
        }
    }
}

/// Ground this rung needs at `context` tokens: the weights plus the K/V and
/// RoPE the guest preallocates for the whole window.
///
/// The figure a deployment should check against what its shedding ladder can
/// actually free, rather than against the card's capacity — the span is not the
/// card, and a resident model is standing in it.
pub fn ground_bytes_at(quant: HermesQuant, context: usize) -> u64 {
    /// Qwen3-14B: 40 layers, 8 KV heads, 128-wide, F32 in this guest's forward.
    const PER_TOKEN_BYTES: u64 = 2 * 8 * 128 * 4 * 40 + 2 * 64 * 4;
    quant.bytes() + PER_TOKEN_BYTES * context as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// The three dev machines, each landing where its capacity says it should.
    #[test]
    fn each_card_gets_the_rung_it_can_hold() {
        // 4090 Mobile.
        assert_eq!(HermesQuant::for_vram(16 * GIB), HermesQuant::Q3KM);
        // 3090 — Q4, not Q5, for the reason `Q5_MIN_TOTAL_VRAM_BYTES` gives.
        assert_eq!(HermesQuant::for_vram(24 * GIB), HermesQuant::Q4KM);
        // RTX PRO 5000 Blackwell.
        assert_eq!(HermesQuant::for_vram(72 * GIB), HermesQuant::Q6K);
    }

    /// The boundaries are inclusive at the bottom of each rung, so a card
    /// sitting exactly on a threshold gets the better file rather than the
    /// worse one.
    #[test]
    fn a_card_on_the_boundary_takes_the_higher_rung() {
        assert_eq!(
            HermesQuant::for_vram(Q6_MIN_TOTAL_VRAM_BYTES),
            HermesQuant::Q6K
        );
        assert_eq!(
            HermesQuant::for_vram(Q5_MIN_TOTAL_VRAM_BYTES),
            HermesQuant::Q5KM
        );
        assert_eq!(
            HermesQuant::for_vram(Q4_MIN_TOTAL_VRAM_BYTES),
            HermesQuant::Q4KM
        );
        assert_eq!(
            HermesQuant::for_vram(Q4_MIN_TOTAL_VRAM_BYTES - 1),
            HermesQuant::Q3KM
        );
    }

    /// **The point of the ladder, and it is sized against the bad day.**
    ///
    /// The rung a 24 GB card picks has to fit the ground its shedding ladder can
    /// free when the working set is warm — the LOW observation, 11,760 MiB, not
    /// the high one. Every rung above it must not fit, or the ladder is choosing
    /// a file that loads on a quiet card and fails on a busy one.
    #[test]
    fn the_24gb_rung_fits_the_worst_observed_ceiling_and_the_next_does_not() {
        // The low end of one evening's drains on the 3090. The high end was
        // 13,808 MiB; sizing against that is what this test exists to prevent.
        const WARM_CEILING: u64 = 11_760 * (1 << 20);
        // `ProseSpec::hermes4_14b`'s ceiling, which is what the lifegen ladder
        // actually claims — not a round number chosen to make the sum work.
        const CONTEXT: usize = 6144;

        let chosen = HermesQuant::for_vram(24 * GIB);
        let needs = ground_bytes_at(chosen, CONTEXT);
        assert!(
            needs < WARM_CEILING,
            "{chosen:?} needs {needs} against a {WARM_CEILING} warm ceiling",
        );
        // And with room to spare, not by a hair: the ceiling itself moves.
        assert!(
            WARM_CEILING - needs > (1 << 30),
            "{chosen:?} clears the warm ceiling by less than a GiB, which is not a margin",
        );

        for over in [HermesQuant::Q5KM, HermesQuant::Q6K] {
            assert!(
                ground_bytes_at(over, CONTEXT) > WARM_CEILING - (1 << 30),
                "{over:?} is close enough to the warm ceiling that a 24 GB card should not take it",
            );
        }
    }

    /// K/V is not a rounding error at this size: a 4k window costs more than a
    /// gigabyte, which is the term a ladder written against file size alone
    /// leaves out.
    #[test]
    fn the_context_term_is_large_enough_to_matter() {
        let weights = HermesQuant::Q5KM.bytes();
        let total = ground_bytes_at(HermesQuant::Q5KM, 4096);
        assert!(total - weights > (1 << 30), "4k of K/V should exceed a GiB");
    }
}
